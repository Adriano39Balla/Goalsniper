"""
goalsniper — model training.

WHAT CHANGED, AND WHY IT MATTERED
---------------------------------

1. THE SPLIT ACTUALLY SPLITS BY TIME NOW.
   The old time_order_split() did:

       df_sorted = df.sort_values("_ts").reset_index(drop=True)
       train_idx = df_sorted.index[:cut].to_numpy()

   reset_index(drop=True) replaces the original row labels with a RangeIndex,
   so df_sorted.index[:cut] is just [0..cut-1] — positional numbers with no
   relationship to the sorted order. Those were then used as a mask against
   df[FEATURES].values, which is in the ORIGINAL order. The result was "the
   first 75% of rows in whatever order Postgres returned them". There was no
   temporal separation at all, and every metric and auto-picked threshold the
   system has ever produced came out of that leaky split.

2. CALIBRATION IS NO LONGER FITTED ON THE EVALUATION SET.
   The old code fitted Platt scaling on y_te, then reported Brier / accuracy /
   precision / F1 on the same y_te, then picked the operating threshold on the
   same y_te. Three uses of one dataset. There is now a three-way time split:
   fit on TRAIN, calibrate and pick thresholds on CAL, report metrics on a
   HOLDOUT that nothing has touched.

3. THE SPLIT IS GROUPED BY MATCH.
   In-play training now uses ALL snapshots per match instead of only the latest
   one (see #4), which means rows within a match are heavily correlated. Splits
   are therefore made at the MATCH level so the same fixture can never appear on
   both sides of a boundary.

4. IN-PLAY TRAINING USES EVERY SNAPSHOT, NOT JUST THE LAST ONE.
   The old query took MAX(created_ts) per match. You harvest every ~3 minutes,
   so ~96% of the data was discarded — and the surviving row was always the
   latest, i.e. minute 80-90. The model learned "Over 2.5 when three goals are
   already in at minute 88", which is trivially true and inflates test metrics,
   while at serving time _candidate_is_sane() blocks exactly those states. The
   model was deployed entirely outside its training distribution.

5. THE SCALER IS BACK, AND IT SHIPS WITH THE MODEL.
   Removing StandardScaler fixed train/serve parity but left LogisticRegression
   applying L2 (C=1.0) to raw features spanning 0/1 flags up to ~1500-point ELO
   ratings. L2 penalises in the units of the feature, so ELO coefficients were
   penalised roughly 1500x harder than binary flags — the best prematch signal
   was being regularized out of existence. Models are now fitted on standardized
   features and the fitted mean/scale are PERSISTED in the model blob, which
   main.py._linpred() applies before scoring. Parity is preserved because the
   transform travels with the weights.

6. C IS TUNED, class_weight IS NOT "balanced".
   Balanced weighting reshapes the intercept so predictions reflect a 50/50
   prior rather than the true base rate — a large upward bias for a ~20% market
   like Over 3.5. You are optimising for calibrated probability, not recall on
   a rare class. C is now selected on the calibration split by log loss.

7. LEAGUE BASE RATES ARE COMPUTED FROM TRAINING MATCHES ONLY.
   _compute_league_rate_map() aggregated over ALL of match_results, so for any
   league near the min_n threshold each test match's own result was inside its
   own league_btts_rate feature. It now takes an explicit set of training
   match_ids.

8. FEATURE LISTS LIVE IN feature_spec.py, SHARED WITH main.py.
   The duplicated derivation logic is gone, along with the ~20 exactly collinear
   or duplicated columns and the ~65 constant-zero live features that used to
   pad PRE_FEATURES. See feature_spec.py for the full removal list.

9. SETTINGS WRITES ARE BUFFERED AND FLUSHED IN ONE TRANSACTION.
   Previously each head wrote its model to `settings` the moment it fitted, so a
   failure part-way through left a half-updated set of models live. All writes
   now accumulate in memory and commit atomically at the end of a successful run.

10. SAMPLE-SIZE FLOOR SCALES WITH FEATURE COUNT.
    MIN_ROWS defaulted to 150 while PRE_FEATURES had ~130 columns — 112 training
    rows for 130 parameters, which forces near-perfect separation and pins
    probabilities at 0 and 1. The floor is now max(MIN_ROWS, rows_per_feature *
    n_features), with a separate floor on distinct matches for the in-play set.

NOT DONE IN THIS PASS, DELIBERATELY
-----------------------------------
A bivariate-Poisson / Dixon-Coles goal model. Over 2.5, Over 3.5 and BTTS are
three views of ONE goal distribution; three unconstrained logistic heads can and
do emit P(Over 3.5) > P(Over 2.5). A joint goal model gives all three coherently
from ~6 parameters and would largely dissolve the sample-size problem. That is a
genuine modelling change, not a bug fix, and it deserves a deliberate decision
rather than being folded into a repair pass. Same for using the de-vigged
closing line as a model FEATURE (the strongest single predictor available):
that needs market probabilities attached to historical snapshots first, which
main.py only started capturing with this release.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import psycopg2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, brier_score_loss, confusion_matrix, f1_score,
    log_loss, precision_score, recall_score,
)

from feature_spec import (
    DEFAULT_LEAGUE_RATES, FEATURES, PRE_FEATURES,
    LEAGUE_RATE_FIELDS_INPLAY, LEAGUE_RATE_FIELDS_PREMATCH,
    build_inplay_features,
)

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PGConnection = psycopg2.extensions.connection

EPS = 1e-6
DEFAULT_LEAGUE_RATE_MIN_N = int(os.getenv("LEAGUE_RATE_MIN_N", "20"))
C_GRID = [float(x) for x in os.getenv("TRAIN_C_GRID", "0.01,0.03,0.1,0.3,1.0,3.0").split(",")]


# ─────────────────────── DB helpers ─────────────────────── #

def _connect(db_url: str) -> PGConnection:
    if not db_url:
        raise SystemExit("DATABASE_URL must be set.")
    if "sslmode=" not in db_url:
        db_url = db_url + ("&" if "?" in db_url else "?") + "sslmode=require"
    conn = psycopg2.connect(db_url)
    conn.autocommit = True
    return conn


def _read_sql(conn: PGConnection, sql: str, params: Tuple = ()) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


def _exec(conn: PGConnection, sql: str, params: Tuple = ()) -> None:
    with conn.cursor() as cur:
        cur.execute(sql, params)


def _as_int(v: Any, default: int = 0) -> int:
    """
    NaN-safe int coercion.

    BUG THIS FIXES: a Postgres NULL comes back as None, but building a
    DataFrame from those rows makes pandas promote any integer column
    containing a NULL to float64, turning the None into np.nan. And np.nan is
    TRUTHY, so a chain like

        int(row["kickoff_ts"] or row["created_ts"] or 0)

    never falls through to the next candidate — it hands np.nan straight to
    int(), which raises "cannot convert float NaN to integer" and aborts the
    entire training run. Every pre-existing tip_snapshots and match_results row
    has a NULL kickoff_ts (the column was added by ALTER TABLE and backfilled as
    NULL), so this fired on the first row of the first load.
    """
    if v is None:
        return default
    try:
        if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
            return default
        if pd.isna(v):
            return default
    except (TypeError, ValueError):
        pass
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _first_int(*candidates: Any, default: int = 0) -> int:
    """First candidate that coerces to a non-zero int, else `default`."""
    for c in candidates:
        v = _as_int(c, 0)
        if v:
            return v
    return default


class SettingsBuffer:
    """
    Accumulates every settings write for a training run and commits them in ONE
    transaction at the end.

    Previously each _train_binary_head() call wrote its model straight to
    `settings` as soon as it fitted, so a failure during prematch training left
    the in-play models updated and the prematch ones stale — a silently
    inconsistent live configuration with no way to tell from the outside.
    """

    def __init__(self, conn: PGConnection):
        self.conn = conn
        self.pending: Dict[str, str] = {}
        self.skipped_locked: List[str] = []

    def set(self, key: str, value: str) -> None:
        self.pending[key] = value

    def get_json(self, key: str) -> Optional[dict]:
        if key in self.pending:
            try:
                return json.loads(self.pending[key])
            except Exception:
                return None
        df = _read_sql(self.conn, "SELECT value FROM settings WHERE key=%s", (key,))
        if df.empty:
            return None
        try:
            return json.loads(df.iloc[0]["value"])
        except Exception:
            return None

    def is_threshold_locked(self, label: str) -> bool:
        df = _read_sql(self.conn, "SELECT value FROM settings WHERE key=%s",
                       (f"conf_threshold_locked:{label}",))
        return (not df.empty) and str(df.iloc[0]["value"]).strip() == "1"

    def set_threshold(self, label: str, thr_pct: float, summary: Dict[str, Any]) -> None:
        if self.is_threshold_locked(label):
            logger.warning("[THRESHOLD] %s is locked — skipping auto-picked value %.2f%%",
                           label, thr_pct)
            self.skipped_locked.append(label)
            return
        self.set(f"conf_threshold:{label}", f"{thr_pct:.2f}")
        # FIX: every threshold is now recorded in the summary. Previously only
        # 1X2 was, so the Telegram training message reported one market of eight.
        summary.setdefault("thresholds", {})[label] = round(float(thr_pct), 2)

    def flush(self) -> int:
        if not self.pending:
            return 0
        prev = self.conn.autocommit
        self.conn.autocommit = False
        try:
            with self.conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO settings(key,value) VALUES(%s,%s) "
                    "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
                    list(self.pending.items()))
            self.conn.commit()
            n = len(self.pending)
            self.pending.clear()
            logger.info("[SETTINGS] committed %d keys atomically", n)
            return n
        except Exception:
            self.conn.rollback()
            raise
        finally:
            self.conn.autocommit = prev


def _log_locked_thresholds(conn: PGConnection) -> List[str]:
    try:
        df = _read_sql(conn, "SELECT key FROM settings "
                             "WHERE key LIKE 'conf_threshold_locked:%%' AND value='1'")
        labels = [str(k)[len("conf_threshold_locked:"):] for k in df["key"].tolist()]
    except Exception as e:
        logger.warning("[THRESHOLD] could not read locked list: %s", e)
        return []
    if labels:
        logger.info("[THRESHOLD] locked markets this run: %s", ", ".join(sorted(labels)))
    else:
        logger.info("[THRESHOLD] no markets currently locked.")
    return labels


def _ensure_training_tables(conn: PGConnection) -> None:
    _exec(conn, "CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)")
    _exec(conn, """CREATE TABLE IF NOT EXISTS prematch_snapshots (
        match_id BIGINT PRIMARY KEY, created_ts BIGINT, payload TEXT)""")
    for stmt in ["ALTER TABLE prematch_snapshots ADD COLUMN IF NOT EXISTS kickoff_ts BIGINT",
                 "ALTER TABLE tip_snapshots ADD COLUMN IF NOT EXISTS kickoff_ts BIGINT",
                 "ALTER TABLE match_results ADD COLUMN IF NOT EXISTS kickoff_ts BIGINT"]:
        try:
            _exec(conn, stmt)
        except Exception:
            pass


# ─────────────────────── League rates (train-only) ─────────────────────── #

def _compute_league_rate_map(conn: PGConnection, train_match_ids: Sequence[int],
                             min_n: int = DEFAULT_LEAGUE_RATE_MIN_N) -> Dict[Any, Dict[str, float]]:
    """
    Per-league BTTS / Over 2.5 / Over 3.5 base rates, computed ONLY over matches
    in the training split.

    The old version aggregated over all of match_results, which meant a test
    match's own outcome was baked into its own league-rate feature whenever that
    league sat near min_n. Small leak, but it inflated precisely the two markets
    the feature was added to rescue.
    """
    ids = [i for i in (_as_int(x, 0) for x in train_match_ids) if i]
    if not ids:
        return {"__GLOBAL__": dict(DEFAULT_LEAGUE_RATES)}
    # NOTE the ::float casts on ALL THREE aggregates. Postgres AVG() over a
    # numeric expression returns `numeric`, which psycopg2 hands back as
    # decimal.Decimal — and Decimal / float raises TypeError. Only `btts` was
    # cast before, so the two CASE-based averages came back as Decimal and blew
    # up the weighted-global calculation below.
    df = _read_sql(conn, """
        SELECT league_id,
               AVG(btts_yes)::float AS btts,
               AVG(CASE WHEN final_goals_h+final_goals_a>2 THEN 1.0 ELSE 0.0 END)::float AS ov25,
               AVG(CASE WHEN final_goals_h+final_goals_a>3 THEN 1.0 ELSE 0.0 END)::float AS ov35,
               COUNT(*)::bigint AS n
        FROM match_results WHERE match_id = ANY(%s) GROUP BY league_id
    """, (ids,))
    if df.empty:
        return {"__GLOBAL__": dict(DEFAULT_LEAGUE_RATES)}

    # Belt and braces: coerce regardless of what the driver returned, so a
    # future schema or driver change cannot reintroduce a Decimal here.
    for col in ("btts", "ov25", "ov35"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    df["n"] = pd.to_numeric(df["n"], errors="coerce").fillna(0).astype("int64")

    out: Dict[Any, Dict[str, float]] = {}
    total_n = float(df["n"].sum()) or 1.0
    out["__GLOBAL__"] = {
        "btts": float((df["btts"].fillna(DEFAULT_LEAGUE_RATES["btts"]) * df["n"]).sum() / total_n),
        "ov25": float((df["ov25"].fillna(DEFAULT_LEAGUE_RATES["ov25"]) * df["n"]).sum() / total_n),
        "ov35": float((df["ov35"].fillna(DEFAULT_LEAGUE_RATES["ov35"]) * df["n"]).sum() / total_n),
    }
    for _, r in df.iterrows():
        lid = r["league_id"]
        if pd.isna(lid) or int(r["n"]) < min_n:
            continue
        out[int(lid)] = {
            "btts": float(r["btts"] if pd.notna(r["btts"]) else DEFAULT_LEAGUE_RATES["btts"]),
            "ov25": float(r["ov25"] if pd.notna(r["ov25"]) else DEFAULT_LEAGUE_RATES["ov25"]),
            "ov35": float(r["ov35"] if pd.notna(r["ov35"]) else DEFAULT_LEAGUE_RATES["ov35"]),
        }
    return out


def _lookup_league_rate(rate_map: Dict[Any, Dict[str, float]], league_id) -> Dict[str, float]:
    if league_id is None or pd.isna(league_id):
        return rate_map["__GLOBAL__"]
    return rate_map.get(int(league_id), rate_map["__GLOBAL__"])


def _apply_league_rates(df: pd.DataFrame, rate_map: Dict[Any, Dict[str, float]],
                        fields: Dict[str, str]) -> pd.DataFrame:
    for kind, col in fields.items():
        df[col] = [
            _lookup_league_rate(rate_map, lid)[kind]
            for lid in df["_league_id"].tolist()
        ]
    return df


def _clean_feature_df(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for col in cols:
        if col not in df.columns:
            df[col] = 0.0
    df[cols] = df[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    return df


# ─────────────────────── Data loading ─────────────────────── #

def load_inplay_data(conn: PGConnection, min_minute: int = 15,
                     max_minute: int = 90) -> pd.DataFrame:
    """
    Loads EVERY harvested snapshot, not just the latest per match.

    Snapshots written before this release used a different payload shape
    ({"stat": {...}, "minute": ..., "gh": ..., "ga": ...}); those are read
    through a compatibility shim so historical harvest is not thrown away.
    """
    rows = _read_sql(conn, """
        SELECT s.match_id, s.created_ts, s.payload, s.kickoff_ts,
               r.final_goals_h, r.final_goals_a, r.btts_yes, r.league_id, r.kickoff_ts AS res_kickoff
        FROM tip_snapshots s
        JOIN match_results r ON r.match_id = s.match_id
    """)
    if rows.empty:
        return pd.DataFrame()

    feats: List[Dict[str, Any]] = []
    legacy = 0
    no_ts = 0
    for _, row in rows.iterrows():
        try:
            payload = json.loads(row["payload"]) or {}
        except Exception:
            continue

        raw = payload.get("raw")
        if raw is None:
            # Legacy schema-1 snapshot.
            legacy += 1
            stat = payload.get("stat") or {}
            raw = dict(stat)
            raw["minute"] = payload.get("minute", 0)
            raw["goals_h"] = payload.get("gh", 0)
            raw["goals_a"] = payload.get("ga", 0)

        minute = float(raw.get("minute", 0) or 0)
        if not (min_minute <= minute <= max_minute):
            continue

        # League rates are injected after the split; a neutral placeholder here.
        f = build_inplay_features(raw, DEFAULT_LEAGUE_RATES)

        gh_f = _as_int(row["final_goals_h"])
        ga_f = _as_int(row["final_goals_a"])
        f["_match_id"] = _as_int(row["match_id"])
        f["_league_id"] = row["league_id"] if pd.notna(row["league_id"]) else None
        # Prefer the fixture's kickoff; fall back to the row's insert time for
        # legacy rows written before kickoff_ts existed.
        f["_ts"] = _first_int(row["kickoff_ts"], row["res_kickoff"], row["created_ts"])
        if not f["_ts"]:
            no_ts += 1
        f["final_goals_sum"] = gh_f + ga_f
        f["final_goals_diff"] = gh_f - ga_f
        f["label_btts"] = 1 if _as_int(row["btts_yes"]) == 1 else 0
        feats.append(f)

    if legacy:
        logger.info("[LOAD] %d legacy-schema in-play snapshots read via compatibility shim", legacy)
    if no_ts:
        logger.warning("[LOAD] %d in-play snapshots have no usable timestamp — they sort to the "
                       "front of the chronological split (i.e. into TRAIN). Harmless for legacy "
                       "rows, but if this count keeps growing, kickoff_ts is not being written.",
                       no_ts)
    if not feats:
        return pd.DataFrame()
    df = pd.DataFrame(feats)
    return _clean_feature_df(df, FEATURES)


def load_prematch_data(conn: PGConnection) -> pd.DataFrame:
    rows = _read_sql(conn, """
        SELECT p.match_id, p.created_ts, p.payload, p.kickoff_ts,
               r.final_goals_h, r.final_goals_a, r.btts_yes, r.league_id,
               r.kickoff_ts AS res_kickoff
        FROM prematch_snapshots p
        JOIN match_results r ON r.match_id = p.match_id
    """)
    if rows.empty:
        return pd.DataFrame()

    feats: List[Dict[str, Any]] = []
    no_ts = 0
    for _, row in rows.iterrows():
        try:
            payload = json.loads(row["payload"]) or {}
            feat = payload.get("feat") or {}
        except Exception:
            continue
        if not feat:
            continue

        f = {k: float(feat.get(k, 0.0) or 0.0) for k in PRE_FEATURES}
        gh_f = _as_int(row["final_goals_h"])
        ga_f = _as_int(row["final_goals_a"])
        f["_match_id"] = _as_int(row["match_id"])
        f["_league_id"] = row["league_id"] if pd.notna(row["league_id"]) else None
        # FIX: kickoff, not insert time. Historical-backfill rows all carried
        # time.time(), so once the split was repaired a 2023 fixture would have
        # sorted AFTER last week's and landed in the holdout.
        f["_ts"] = _first_int(row["kickoff_ts"], row["res_kickoff"],
                              payload.get("kickoff_ts"), row["created_ts"])
        if not f["_ts"]:
            no_ts += 1
        f["final_goals_sum"] = gh_f + ga_f
        f["final_goals_diff"] = gh_f - ga_f
        f["label_btts"] = 1 if _as_int(row["btts_yes"]) == 1 else 0
        feats.append(f)

    if no_ts:
        logger.warning("[LOAD] %d prematch snapshots have no usable timestamp — they sort into "
                       "TRAIN. Re-run /admin/backfill-prematch-history to stamp historical rows "
                       "with their real kickoff time.", no_ts)
    if not feats:
        return pd.DataFrame()
    df = pd.DataFrame(feats)
    return _clean_feature_df(df, PRE_FEATURES)


# ─────────────────────── Splitting ─────────────────────── #

def grouped_time_split(df: pd.DataFrame, cal_size: float, test_size: float,
                       embargo_groups: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Three-way chronological split at the MATCH level.

    Groups (fixtures) are ordered by their kickoff timestamp and assigned whole
    to train / calibration / holdout, so correlated snapshots from one match
    cannot straddle a boundary. An optional embargo drops a number of fixtures
    either side of each boundary.

    Returns boolean masks aligned to df's own row order — which is the bug the
    old implementation had: it built positional indices on a re-indexed sorted
    copy and applied them to the unsorted frame.
    """
    n = len(df)
    if n == 0:
        e = np.zeros(0, dtype=bool)
        return e, e, e

    gcol = "_match_id" if "_match_id" in df.columns else None
    tcol = "_ts" if "_ts" in df.columns else None
    if gcol is None or tcol is None:
        rng = np.random.default_rng(int(os.getenv("TRAIN_SPLIT_SEED", "42")))
        idx = np.arange(n)
        rng.shuffle(idx)
        c1 = int(n * (1 - cal_size - test_size))
        c2 = int(n * (1 - test_size))
        tr = np.zeros(n, dtype=bool); ca = np.zeros(n, dtype=bool); te = np.zeros(n, dtype=bool)
        tr[idx[:c1]] = True; ca[idx[c1:c2]] = True; te[idx[c2:]] = True
        return tr, ca, te

    group_time = df.groupby(gcol)[tcol].min().sort_values()
    groups = list(group_time.index)
    g = len(groups)
    c1 = max(1, int(g * (1 - cal_size - test_size)))
    c2 = max(c1 + 1, int(g * (1 - test_size)))

    train_g = set(groups[:max(0, c1 - embargo_groups)])
    cal_g = set(groups[c1:max(c1, c2 - embargo_groups)])
    test_g = set(groups[c2:])

    gvals = df[gcol].to_numpy()
    tr = np.isin(gvals, list(train_g))
    ca = np.isin(gvals, list(cal_g))
    te = np.isin(gvals, list(test_g))
    logger.info("[SPLIT] fixtures: train=%d cal=%d holdout=%d (rows: %d/%d/%d)",
                len(train_g), len(cal_g), len(test_g), int(tr.sum()), int(ca.sum()), int(te.sum()))
    return tr, ca, te


# ─────────────────────── Model utilities ─────────────────────── #

def _standardize(X_tr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = X_tr.mean(axis=0)
    scale = X_tr.std(axis=0)
    scale[scale < 1e-9] = 1.0  # constant column -> leave as (x - mean)
    return mean, scale


def _fit_lr(X: np.ndarray, y: np.ndarray, C: float) -> Optional[LogisticRegression]:
    if len(np.unique(y)) < 2:
        return None
    # FIX: no class_weight="balanced". It rebases the intercept toward a 50/50
    # prior, which is a large systematic upward bias on low-prevalence markets
    # like Over 3.5. The objective here is a calibrated probability.
    return LogisticRegression(max_iter=3000, solver="liblinear", C=C).fit(X, y)


def _select_C(X_tr, y_tr, X_ca, y_ca) -> Tuple[Optional[LogisticRegression], float]:
    """Pick the regularization strength by log loss on the calibration split."""
    best_m, best_C, best_ll = None, C_GRID[-1], float("inf")
    for C in C_GRID:
        m = _fit_lr(X_tr, y_tr, C)
        if m is None:
            continue
        try:
            p = m.predict_proba(X_ca)[:, 1]
            ll = log_loss(y_ca, np.clip(p, EPS, 1 - EPS), labels=[0, 1])
        except Exception:
            continue
        if ll < best_ll:
            best_m, best_C, best_ll = m, C, ll
    return best_m, best_C


def _logit_vec(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_platt(y_true: np.ndarray, p_raw: np.ndarray) -> Tuple[float, float]:
    z = _logit_vec(p_raw).reshape(-1, 1)
    lr = LogisticRegression(max_iter=1000, solver="lbfgs").fit(z, y_true.astype(int))
    return float(lr.coef_.ravel()[0]), float(lr.intercept_.ravel()[0])


def _apply_platt(p_raw: np.ndarray, a: float, b: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(a * _logit_vec(p_raw) + b)))


def build_model_blob(model: LogisticRegression, features: List[str],
                     mean: np.ndarray, scale: np.ndarray,
                     cal: Tuple[float, float], C: float) -> Dict[str, Any]:
    """
    The blob now carries its own scaler. main.py._linpred() applies
    (x - mean)/scale per feature before the dot product, so the model can be
    fitted on standardized features (making L2 scale-fair) without breaking
    serving parity — which is what the original StandardScaler removal was
    working around.
    """
    return {
        "intercept": float(model.intercept_.ravel()[0]),
        "weights": {name: float(w) for name, w in zip(features, model.coef_.ravel().tolist())},
        "scaler": {"mean": {n: float(m) for n, m in zip(features, mean.tolist())},
                   "scale": {n: float(s) for n, s in zip(features, scale.tolist())}},
        "calibration": {"method": "platt", "a": float(cal[0]), "b": float(cal[1])},
        "C": float(C),
        "n_features": len(features),
    }


def _fmt_line(line: float) -> str:
    return f"{line}".rstrip("0").rstrip(".")


def _validate(X: np.ndarray, y: np.ndarray, feature_names: List[str], context: str) -> bool:
    if X.ndim != 2 or X.shape[0] != len(y):
        logger.warning("[VALIDATE] %s: X/y shape mismatch (X=%s, y=%s)", context, X.shape, len(y))
        return False
    if X.shape[1] != len(feature_names):
        logger.warning("[VALIDATE] %s: %d columns vs %d names", context, X.shape[1], len(feature_names))
        return False
    if not np.all(np.isfinite(X)):
        logger.warning("[VALIDATE] %s: %d non-finite values after cleanup",
                       context, int(np.sum(~np.isfinite(X))))
        return False
    uniq = np.unique(y)
    if not np.all(np.isin(uniq, [0, 1])):
        logger.warning("[VALIDATE] %s: labels not binary (found %s)", context, uniq)
        return False
    return True


# ─────────────────────── Thresholding ─────────────────────── #

THRESHOLD_MARGIN = float(os.getenv("TARGET_PRECISION_MARGIN", "0.03"))
THRESHOLD_FALLBACK = os.getenv("THRESHOLD_FALLBACK", "suppress").strip().lower()


def _pick_threshold(y_true: np.ndarray, p: np.ndarray, target_precision: float,
                    min_preds: int, default_threshold: float,
                    max_thresh_pct: float = 85.0) -> Tuple[float, Dict[str, Any]]:
    """
    Smallest threshold reaching the precision target with at least min_preds
    selections, evaluated on the CALIBRATION split only.

    TWO CHANGES, both forced by what the first two live runs showed.

    1. THE TARGET IS NOW RELATIVE TO THE BASE RATE.
       A fixed TARGET_PRECISION of 0.60 is not a filter for a market whose base
       rate is already 0.59 — Over 2.5 "hit its 60% target" at precision 0.6054
       and 0.6015 on two consecutive runs, i.e. about one point above simply
       betting Over on everything, which is noise on ~1,000 selections. The
       target is now max(TARGET_PRECISION, base_rate + TARGET_PRECISION_MARGIN),
       so a market has to demonstrate it beats its own base rate, not an
       arbitrary constant that may sit below it.

    2. FAILING TO FIND A THRESHOLD NOW SUPPRESSES THE MARKET.
       The old fallback was "best F1". For any market with prevalence above 50%
       that is degenerate: F1 is maximised by predicting the positive class
       everywhere, so the fallback returned the LOWEST possible threshold.
       PRE_BTTS_YES landed there on both runs (method "best_f1", F1 0.6997,
       clipped up to MIN_THRESH) — meaning the market that demonstrably has zero
       signal was handed the most permissive threshold in the system. Backwards.
       A market that cannot show precision above its base rate should go quiet,
       not open up. Set THRESHOLD_FALLBACK=best_f1 to restore the old behaviour.
    """
    y = y_true.astype(int)
    p = np.asarray(p, dtype=float)
    base_rate = float(y.mean()) if len(y) else 0.0
    effective_target = max(float(target_precision), base_rate + THRESHOLD_MARGIN)
    grid = np.arange(0.50, 0.951, 0.005)

    for t in grid:
        pred = (p >= t).astype(int)
        n_pred = int(pred.sum())
        if n_pred < min_preds:
            continue
        prec = float(precision_score(y, pred, zero_division=0))
        if prec >= effective_target:
            return float(t), {"method": "target_precision", "n_at_threshold": n_pred,
                              "precision_at_threshold": round(prec, 4),
                              "base_rate": round(base_rate, 4),
                              "effective_target": round(effective_target, 4),
                              "lift_over_base_pp": round(100.0 * (prec - base_rate), 2)}

    best_t, best_prec, best_n = None, -1.0, 0
    for t in grid:
        pred = (p >= t).astype(int)
        n_pred = int(pred.sum())
        if n_pred < min_preds:
            continue
        prec = float(precision_score(y, pred, zero_division=0))
        if prec > best_prec:
            best_prec, best_t, best_n = prec, float(t), n_pred

    diag = {"base_rate": round(base_rate, 4), "effective_target": round(effective_target, 4),
            "best_precision_found": round(best_prec, 4) if best_prec >= 0 else None,
            "best_n": best_n}

    if THRESHOLD_FALLBACK == "best_f1":
        b_t, b_f1 = None, -1.0
        for t in grid:
            pred = (p >= t).astype(int)
            if int(pred.sum()) < min_preds:
                continue
            f1 = f1_score(y, pred, zero_division=0)
            if f1 > b_f1:
                b_f1, b_t = f1, float(t)
        if b_t is not None:
            diag.update({"method": "best_f1", "f1": round(float(b_f1), 4)})
            return b_t, diag

    diag["method"] = "suppressed"
    logger.warning("[THRESHOLD] no threshold beat base rate %.3f by %.0fpp with >=%d selections "
                   "(best was %.4f). SUPPRESSING this market at %.1f%% — it has not demonstrated "
                   "an edge over simply always betting the majority side.",
                   base_rate, THRESHOLD_MARGIN * 100, min_preds,
                   best_prec if best_prec >= 0 else float("nan"), max_thresh_pct)
    return float(max_thresh_pct) / 100.0, diag


# ─────────────────────── Core fit ─────────────────────── #

def _train_binary_head(
    buf: SettingsBuffer,
    X_all: np.ndarray, y_all: np.ndarray,
    m_tr: np.ndarray, m_ca: np.ndarray, m_te: np.ndarray,
    feature_names: List[str],
    model_key: str,
    threshold_label: Optional[str],
    summary: Dict[str, Any],
    target_precision: float, min_preds: int,
    min_thresh_pct: float, max_thresh_pct: float,
    default_thr_prob: float,
    metrics_name: Optional[str] = None,
) -> Tuple[bool, Dict[str, Any], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Returns (ok, metrics, p_on_cal, p_on_holdout).

    Pipeline: standardize on TRAIN -> select C on CAL -> fit -> Platt on CAL ->
    threshold on CAL -> metrics on HOLDOUT. No dataset is used twice.
    """
    ctx = metrics_name or model_key
    if not _validate(X_all, y_all, feature_names, ctx):
        return False, {}, None, None

    X_tr, y_tr = X_all[m_tr], y_all[m_tr]
    X_ca, y_ca = X_all[m_ca], y_all[m_ca]
    X_te, y_te = X_all[m_te], y_all[m_te]

    if min(len(y_tr), len(y_ca), len(y_te)) < 20:
        logger.info("[SKIP] %s: split too small (train=%d cal=%d holdout=%d)",
                    ctx, len(y_tr), len(y_ca), len(y_te))
        return False, {}, None, None
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_ca)) < 2:
        logger.info("[SKIP] %s: single-class train or calibration split", ctx)
        return False, {}, None, None

    logger.info("[SPLIT] %s: train=%d cal=%d holdout=%d (prevalence %.3f / %.3f / %.3f)",
                ctx, len(y_tr), len(y_ca), len(y_te),
                float(y_tr.mean()), float(y_ca.mean()),
                float(y_te.mean()) if len(y_te) else 0.0)

    mean, scale = _standardize(X_tr)
    Z_tr = (X_tr - mean) / scale
    Z_ca = (X_ca - mean) / scale
    Z_te = (X_te - mean) / scale

    m, C = _select_C(Z_tr, y_tr, Z_ca, y_ca)
    if m is None:
        return False, {}, None, None

    p_ca_raw = m.predict_proba(Z_ca)[:, 1]
    a, b = fit_platt(y_ca, p_ca_raw)
    p_ca = _apply_platt(p_ca_raw, a, b)
    p_te = _apply_platt(m.predict_proba(Z_te)[:, 1], a, b) if len(y_te) else np.array([])

    blob = build_model_blob(m, feature_names, mean, scale, (a, b), C)
    for k in (f"model_latest:{model_key}", f"model:{model_key}"):
        buf.set(k, json.dumps(blob))

    mets: Dict[str, Any] = {"C": C, "n_train": int(len(y_tr)), "n_cal": int(len(y_ca)),
                            "n_holdout": int(len(y_te)), "prevalence": float(y_all.mean()),
                            "n_features": len(feature_names)}

    if len(y_te) and len(np.unique(y_te)) > 1:
        pred = (p_te >= 0.5).astype(int)
        cm = confusion_matrix(y_te, pred)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            logger.warning("[METRICS] %s: degenerate confusion matrix (shape=%s, preds=%s, y=%s)",
                           ctx, cm.shape, np.unique(pred), np.unique(y_te))
            tn = fp = fn = tp = 0
        mets.update({
            "brier": float(brier_score_loss(y_te, p_te)),
            "acc": float(accuracy_score(y_te, pred)),
            "logloss": float(log_loss(y_te, np.clip(p_te, EPS, 1 - EPS), labels=[0, 1])),
            "precision": float(precision_score(y_te, pred, zero_division=0)),
            "recall": float(recall_score(y_te, pred, zero_division=0)),
            "f1": float(f1_score(y_te, pred, zero_division=0)),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "metrics_source": "holdout (untouched by fitting, calibration or thresholding)",
        })
        # Calibration gap on the holdout: the honest headline number.
        mets["mean_predicted"] = float(np.mean(p_te))
        mets["mean_actual"] = float(np.mean(y_te))
        mets["calibration_gap_pct"] = round(100.0 * (mets["mean_actual"] - mets["mean_predicted"]), 2)
        logger.info("[METRICS] %s: acc=%.3f prec=%.3f brier=%.4f calib_gap=%+.2fpp (C=%g)",
                    ctx, mets["acc"], mets["precision"], mets["brier"],
                    mets["calibration_gap_pct"], C)
    else:
        mets["metrics_source"] = "unavailable (empty or single-class holdout)"

    # Top weights are only meaningful now that collinear duplicates are gone.
    mets["feature_importance"] = dict(sorted(
        zip(feature_names, m.coef_.ravel().tolist()), key=lambda x: abs(x[1]), reverse=True)[:10])

    if threshold_label:
        thr_prob, diag = _pick_threshold(y_ca, p_ca, target_precision, min_preds,
                                         default_thr_prob, max_thresh_pct=max_thresh_pct)
        thr_pct = float(np.clip(thr_prob * 100.0, min_thresh_pct, max_thresh_pct))
        buf.set_threshold(threshold_label, thr_pct, summary)
        mets["threshold_pct"] = round(thr_pct, 2)
        mets["threshold_selection"] = diag

    return True, mets, p_ca, p_te


def _effective_min_rows(n_features: int, min_rows_env: int, rows_per_feature: int) -> int:
    """
    FIX: MIN_ROWS defaulted to 150 against ~130 prematch features. Below roughly
    10-20 events per parameter, logistic regression drives toward perfect
    separation, probabilities collapse to 0/1, and everything clears threshold —
    which is a complete explanation for chronic overconfidence on its own.
    """
    return max(int(min_rows_env), int(rows_per_feature) * int(n_features))


# ─────────────────────── Entry point ─────────────────────── #

def train_models(
    db_url: Optional[str] = None,
    min_minute: Optional[int] = None,
    test_size: Optional[float] = None,
    min_rows: Optional[int] = None,
) -> Dict[str, Any]:
    conn = _connect(db_url or os.getenv("DATABASE_URL"))
    _ensure_training_tables(conn)
    _log_locked_thresholds(conn)
    buf = SettingsBuffer(conn)

    min_minute = int(min_minute if min_minute is not None else os.getenv("TRAIN_MIN_MINUTE", 15))
    test_size = float(test_size if test_size is not None else os.getenv("TRAIN_TEST_SIZE", 0.20))
    cal_size = float(os.getenv("TRAIN_CAL_SIZE", "0.20"))
    min_rows_env = int(min_rows if min_rows is not None else os.getenv("MIN_ROWS", 500))
    rows_per_feature = int(os.getenv("ROWS_PER_FEATURE", "20"))
    min_matches_inplay = int(os.getenv("MIN_MATCHES_INPLAY", "300"))
    embargo_groups = int(os.getenv("TRAIN_EMBARGO_GROUPS", "0"))

    ou_lines: List[float] = []
    for t in os.getenv("OU_TRAIN_LINES", "2.5,3.5").split(","):
        t = t.strip()
        if t:
            try:
                ou_lines.append(float(t))
            except Exception:
                pass
    ou_lines = ou_lines or [2.5, 3.5]

    target_precision = float(os.getenv("TARGET_PRECISION", "0.60"))
    min_preds = int(os.getenv("THRESH_MIN_PREDICTIONS", "100"))
    min_thresh = float(os.getenv("MIN_THRESH", "55"))
    max_thresh = float(os.getenv("MAX_THRESH", "85"))

    summary: Dict[str, Any] = {"ok": True, "trained": {}, "metrics": {}, "thresholds": {},
                               "feature_counts": {}, "data_stats": {}, "skipped": {}}

    try:
        # ══════════ In-play ══════════
        df_ip = load_inplay_data(conn, min_minute=min_minute)
        n_ip = len(df_ip)
        n_ip_matches = int(df_ip["_match_id"].nunique()) if n_ip else 0
        need_ip = _effective_min_rows(len(FEATURES), min_rows_env, rows_per_feature)
        summary["data_stats"].update({"inplay_rows": n_ip, "inplay_matches": n_ip_matches,
                                      "inplay_rows_required": need_ip,
                                      "inplay_matches_required": min_matches_inplay})

        if n_ip >= need_ip and n_ip_matches >= min_matches_inplay:
            logger.info("In-play: %d snapshots across %d fixtures, %d features",
                        n_ip, n_ip_matches, len(FEATURES))
            m_tr, m_ca, m_te = grouped_time_split(df_ip, cal_size, test_size, embargo_groups)

            rate_map = _compute_league_rate_map(conn, df_ip.loc[m_tr, "_match_id"].unique())
            df_ip = _apply_league_rates(df_ip, rate_map, LEAGUE_RATE_FIELDS_INPLAY)

            X = df_ip[FEATURES].to_numpy(dtype=float)
            summary["feature_counts"]["inplay"] = len(FEATURES)

            ok, mets, _, _ = _train_binary_head(
                buf, X, df_ip["label_btts"].to_numpy(dtype=int), m_tr, m_ca, m_te, FEATURES,
                "BTTS_YES", "BTTS", summary, target_precision, min_preds,
                min_thresh, max_thresh, 0.65, "BTTS_YES")
            summary["trained"]["BTTS_YES"] = ok
            if ok:
                summary["metrics"]["BTTS_YES"] = mets

            totals = df_ip["final_goals_sum"].to_numpy(dtype=int)
            for line in ou_lines:
                name = f"OU_{_fmt_line(line)}"
                ok, mets, _, _ = _train_binary_head(
                    buf, X, (totals > line).astype(int), m_tr, m_ca, m_te, FEATURES,
                    name, f"Over/Under {_fmt_line(line)}", summary, target_precision, min_preds,
                    min_thresh, max_thresh, 0.65, name)
                summary["trained"][name] = ok
                if ok:
                    summary["metrics"][name] = mets
                    if abs(line - 2.5) < 1e-6:
                        blob = buf.get_json(f"model_latest:{name}")
                        if blob is not None:
                            for k in ("model_latest:O25", "model:O25"):
                                buf.set(k, json.dumps(blob))

            gd = df_ip["final_goals_diff"].to_numpy(dtype=int)
            heads = {}
            for key, y in (("WLD_HOME", (gd > 0)), ("WLD_DRAW", (gd == 0)), ("WLD_AWAY", (gd < 0))):
                ok, mets, p_ca, _ = _train_binary_head(
                    buf, X, y.astype(int), m_tr, m_ca, m_te, FEATURES,
                    key, None, summary, target_precision, min_preds,
                    min_thresh, max_thresh, 0.45, key)
                summary["trained"][key] = ok
                if ok:
                    summary["metrics"][key] = mets
                heads[key] = (ok, p_ca)

            _fit_1x2_threshold(heads, gd, m_ca, buf, summary, "1X2",
                               target_precision, min_preds, min_thresh, max_thresh)
        else:
            reason = (f"have {n_ip} snapshots / {n_ip_matches} fixtures, "
                      f"need {need_ip} / {min_matches_inplay}")
            logger.info("In-Play: not enough data (%s).", reason)
            summary["skipped"]["inplay"] = reason
            summary["trained"]["BTTS_YES"] = False

        # ══════════ Prematch ══════════
        df_pre = load_prematch_data(conn)
        n_pre = len(df_pre)
        need_pre = _effective_min_rows(len(PRE_FEATURES), min_rows_env, rows_per_feature)
        summary["data_stats"].update({"prematch_rows": n_pre, "prematch_rows_required": need_pre})

        if n_pre >= need_pre:
            logger.info("Prematch: %d rows, %d features", n_pre, len(PRE_FEATURES))
            m_tr, m_ca, m_te = grouped_time_split(df_pre, cal_size, test_size, embargo_groups)

            rate_map = _compute_league_rate_map(conn, df_pre.loc[m_tr, "_match_id"].unique())
            df_pre = _apply_league_rates(df_pre, rate_map, LEAGUE_RATE_FIELDS_PREMATCH)

            Xp = df_pre[PRE_FEATURES].to_numpy(dtype=float)
            summary["feature_counts"]["prematch"] = len(PRE_FEATURES)

            ok, mets, _, _ = _train_binary_head(
                buf, Xp, df_pre["label_btts"].to_numpy(dtype=int), m_tr, m_ca, m_te, PRE_FEATURES,
                "PRE_BTTS_YES", "PRE BTTS", summary, target_precision, min_preds,
                min_thresh, max_thresh, 0.65, "PRE_BTTS_YES")
            summary["trained"]["PRE_BTTS_YES"] = ok
            if ok:
                summary["metrics"]["PRE_BTTS_YES"] = mets

            totals = df_pre["final_goals_sum"].to_numpy(dtype=int)
            for line in ou_lines:
                name = f"PRE_OU_{_fmt_line(line)}"
                ok, mets, _, _ = _train_binary_head(
                    buf, Xp, (totals > line).astype(int), m_tr, m_ca, m_te, PRE_FEATURES,
                    name, f"PRE Over/Under {_fmt_line(line)}", summary, target_precision,
                    min_preds, min_thresh, max_thresh, 0.65, name)
                summary["trained"][name] = ok
                if ok:
                    summary["metrics"][name] = mets

            gd = df_pre["final_goals_diff"].to_numpy(dtype=int)
            heads = {}
            # FIX (paired with main.py's 1X2 fix): PRE_WLD_DRAW is now trained.
            # Serving needs a draw probability to build a real 1X2 denominator;
            # without it the only options are a Draw-No-Bet probability priced
            # against 1X2 odds (the old bug) or a hardcoded draw prior.
            for key, y in (("PRE_WLD_HOME", (gd > 0)), ("PRE_WLD_DRAW", (gd == 0)),
                           ("PRE_WLD_AWAY", (gd < 0))):
                ok, mets, p_ca, _ = _train_binary_head(
                    buf, Xp, y.astype(int), m_tr, m_ca, m_te, PRE_FEATURES,
                    key, None, summary, target_precision, min_preds,
                    min_thresh, max_thresh, 0.45, key)
                summary["trained"][key] = ok
                if ok:
                    summary["metrics"][key] = mets
                heads[key.replace("PRE_", "")] = (ok, p_ca)

            _fit_1x2_threshold(heads, gd, m_ca, buf, summary, "PRE 1X2",
                               target_precision, min_preds, min_thresh, max_thresh)
        else:
            reason = f"have {n_pre} rows, need {need_pre}"
            logger.info("Prematch: not enough data (%s).", reason)
            summary["skipped"]["prematch"] = reason
            summary["trained"]["PRE_BTTS_YES"] = False

        summary["locked_thresholds_skipped"] = buf.skipped_locked

        trained_at = pd.Timestamp.now(tz="UTC")
        bundle = {
            "trained_at_utc": trained_at.isoformat(timespec="seconds"),
            **summary["metrics"],
            "features_inplay": FEATURES,
            "features_prematch": PRE_FEATURES,
            "thresholds": summary.get("thresholds", {}),
            "target_precision": target_precision,
            "ou_lines": [float(x) for x in ou_lines],
            "split": {"cal_size": cal_size, "test_size": test_size,
                      "grouped_by": "match_id", "embargo_groups": embargo_groups},
            "data_stats": summary.get("data_stats", {}),
            "skipped": summary.get("skipped", {}),
        }
        buf.set("model_metrics_latest", json.dumps(bundle))
        buf.set(f"model_metrics:{trained_at.strftime('%Y%m%dT%H%M%SZ')}", json.dumps(bundle))

        # Nothing has been written to `settings` until this line. A failure
        # anywhere above leaves the previous, self-consistent model set live.
        buf.flush()

        logger.info("Trained: %s", [k for k, v in summary["trained"].items() if v])
        logger.info("Thresholds: %s", summary["thresholds"])
        return summary

    except Exception as e:
        logger.exception("Training failed: %s", e)
        return {"ok": False, "error": str(e),
                "note": "No settings were written — the previously deployed models are untouched."}
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _fit_1x2_threshold(heads: Dict[str, Tuple[bool, Optional[np.ndarray]]], gd: np.ndarray,
                       m_ca: np.ndarray, buf: SettingsBuffer, summary: Dict[str, Any],
                       label: str, target_precision: float, min_preds: int,
                       min_thresh: float, max_thresh: float) -> None:
    """
    Pick the 1X2 threshold on EXACTLY the statistic serving compares against.

    The old code picked it from three-way argmax correctness against
    max(p_home, p_draw, p_away), while serving applied it as a per-side cut on a
    two-way renormalised probability. Different quantity, different denominator,
    different definition of "correct" — the number carried no information.

    Here: normalise home/draw/away over their sum (the same thing main.py's
    _wld_candidates does), then pool (p_home, home_won) and (p_away, away_won)
    and choose the cut that reaches target precision on that pooled set.
    """
    ok_h, p_h = heads.get("WLD_HOME", (False, None))
    ok_d, p_d = heads.get("WLD_DRAW", (False, None))
    ok_a, p_a = heads.get("WLD_AWAY", (False, None))
    if not (ok_h and ok_a) or p_h is None or p_a is None:
        logger.info("[1X2] %s: home/away heads unavailable — threshold not set", label)
        return

    p_hc = np.clip(p_h, EPS, 1 - EPS)
    p_ac = np.clip(p_a, EPS, 1 - EPS)
    p_dc = np.clip(p_d, EPS, 1 - EPS) if (ok_d and p_d is not None) else np.full_like(p_hc, 0.26)
    s = np.maximum(p_hc + p_dc + p_ac, EPS)
    phn, pan = p_hc / s, p_ac / s

    gd_ca = gd[m_ca]
    y_home = (gd_ca > 0).astype(int)
    y_away = (gd_ca < 0).astype(int)

    probs = np.concatenate([phn, pan])
    ys = np.concatenate([y_home, y_away])
    thr_prob, diag = _pick_threshold(ys, probs, target_precision, min_preds, 0.45,
                                     max_thresh_pct=max_thresh)
    thr_pct = float(np.clip(thr_prob * 100.0, min_thresh, max_thresh))
    buf.set_threshold(label, thr_pct, summary)
    summary.setdefault("metrics", {})[f"{label}_threshold_diag"] = diag
    logger.info("[1X2] %s threshold set to %.2f%% (%s)", label, thr_pct, diag.get("method"))


# ─────────────────────── CLI ─────────────────────── #

def _cli_main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-url", help="Postgres DSN (or use env DATABASE_URL)")
    ap.add_argument("--min-minute", dest="min_minute", type=int,
                    default=int(os.getenv("TRAIN_MIN_MINUTE", 15)))
    ap.add_argument("--test-size", type=float, default=float(os.getenv("TRAIN_TEST_SIZE", 0.20)))
    ap.add_argument("--min-rows", type=int, default=int(os.getenv("MIN_ROWS", 500)))
    args = ap.parse_args()
    print(json.dumps(train_models(
        db_url=args.db_url or os.getenv("DATABASE_URL"),
        min_minute=args.min_minute, test_size=args.test_size, min_rows=args.min_rows), indent=2))


if __name__ == "__main__":
    _cli_main()
