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
   Three-way time split: fit on TRAIN, calibrate and pick thresholds on CAL,
   report metrics on a HOLDOUT that nothing has touched.

3. THE SPLIT IS GROUPED BY MATCH, so correlated in-play snapshots from one
   fixture cannot straddle a boundary.

4. IN-PLAY TRAINING USES EVERY SNAPSHOT, NOT JUST THE LAST ONE.

5. THE SCALER IS BACK, AND IT SHIPS WITH THE MODEL (mean/scale persisted in the
   blob, applied by main.py._linpred), so L2 is scale-fair at fit time without
   breaking serving parity.

6. C IS TUNED on the calibration split; class_weight is NOT "balanced".

7. LEAGUE BASE RATES ARE COMPUTED FROM TRAINING MATCHES ONLY.

8. FEATURE LISTS LIVE IN feature_spec.py, SHARED WITH main.py.

9. SETTINGS WRITES ARE BUFFERED AND FLUSHED IN ONE TRANSACTION.

10. SAMPLE-SIZE FLOOR SCALES WITH FEATURE COUNT.

11. EVERY THRESHOLD IS VERIFIED ON THE HOLDOUT BEFORE IT IS WRITTEN. A
    threshold picked on the calibration split is the best of ~90 grid points and
    is biased upward. If its lift does not survive on the holdout at
    MIN_HOLDOUT_LIFT_SE standard errors over at least MIN_HOLDOUT_SELECTIONS
    selections, the market is suppressed instead.

12. DOUBLE CHANCE AND DRAW NO BET ARE NOW TRAINED AND VERIFIED, NOT DEFAULTED.
    These are algebraic transforms of the 1X2 heads, so they need no new model —
    but they previously had no threshold written at all, which meant main.py's
    _get_market_threshold() fell through to CONF_THRESHOLD (70). That was a hole
    straight through the suppression system: PRE 1X2 could be suppressed at 85
    for failing its holdout while Double Chance, derived from the very same
    heads, fired at 70 on any fixture with a decent home side. Both markets now
    get a threshold picked on CAL and verified on the HOLDOUT exactly like every
    other market, and are suppressed the same way when they fail.

NOT DONE IN THIS PASS, DELIBERATELY
-----------------------------------
A bivariate-Poisson / Dixon-Coles goal model, and using the de-vigged closing
line as a model FEATURE. Both are genuine modelling changes that deserve a
deliberate decision rather than being folded into a repair pass. The closing
line as a feature additionally needs market probabilities attached to historical
snapshots, which main.py only started capturing recently.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
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
    build_inplay_features, derive_dc_dnb,
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
FALLBACK_DRAW_PROB = float(os.getenv("FALLBACK_DRAW_PROB", "0.26"))


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
    entire training run.
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
    transaction at the end, so a mid-run failure leaves the previously deployed
    (self-consistent) model set live rather than a half-updated one.
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
    Per-league BTTS / Over 2.5 / Over 3.5 base rates over TRAINING matches only,
    so a test match's own outcome is never inside its own league-rate feature.
    """
    ids = [i for i in (_as_int(x, 0) for x in train_match_ids) if i]
    if not ids:
        return {"__GLOBAL__": dict(DEFAULT_LEAGUE_RATES)}
    # NOTE the ::float casts on ALL THREE aggregates. Postgres AVG() over a
    # numeric expression returns `numeric`, which psycopg2 hands back as
    # decimal.Decimal — and Decimal / float raises TypeError.
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
        df[col] = [_lookup_league_rate(rate_map, lid)[kind] for lid in df["_league_id"].tolist()]
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

    Snapshots written before the schema-2 release used a different payload shape
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
        # Kickoff, not insert time: historical-backfill rows all carried
        # time.time(), which would sort a 2023 fixture after last week's.
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
    # No class_weight="balanced": it rebases the intercept toward a 50/50 prior,
    # a large systematic upward bias on low-prevalence markets. The objective
    # here is a calibrated probability, not recall on a rare class.
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
    The blob carries its own scaler. main.py._linpred() applies
    (x - mean)/scale per feature before the dot product, so the model can be
    fitted on standardized features (making L2 scale-fair) without breaking
    serving parity.
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

    1. THE TARGET IS RELATIVE TO THE BASE RATE. A fixed 0.60 is not a filter for
       a market whose base rate is already 0.59. Effective target is
       max(TARGET_PRECISION, base_rate + TARGET_PRECISION_MARGIN).
    2. FAILING TO FIND A THRESHOLD SUPPRESSES THE MARKET. The old best-F1
       fallback is degenerate above 50% prevalence (F1 is maximised by
       predicting positive everywhere), so it returned the LOWEST threshold for
       exactly the markets with no signal.
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
                   "(best was %s). SUPPRESSING this market at %.1f%%.",
                   base_rate, THRESHOLD_MARGIN * 100, min_preds,
                   f"{best_prec:.4f}" if best_prec >= 0 else "n/a", max_thresh_pct)
    return float(max_thresh_pct) / 100.0, diag


# ─────────────────────── Holdout verification ─────────────────────── #

MIN_HOLDOUT_SELECTIONS = int(os.getenv("MIN_HOLDOUT_SELECTIONS", "30"))
MIN_HOLDOUT_LIFT_SE = float(os.getenv("MIN_HOLDOUT_LIFT_SE", "2.5"))


def _threshold_on_holdout(y_te: np.ndarray, p_te: np.ndarray, thr_prob: float) -> Dict[str, Any]:
    """
    Re-measure a chosen threshold on data that had no part in choosing it.

    The standard error is computed under the NULL (the base rate), not from the
    observed precision. Using the observed precision collapses the SE to zero
    whenever a threshold selects a handful of rows that all lose.
    """
    if y_te is None or p_te is None or len(y_te) == 0 or len(p_te) == 0:
        return {"note": "no holdout available", "lift_in_std_errors": None}
    sel = np.asarray(p_te) >= float(thr_prob)
    n_sel = int(sel.sum())
    base = float(np.mean(y_te))
    if n_sel == 0:
        return {"n_at_threshold": 0, "base_rate": round(base, 4),
                "lift_in_std_errors": None,
                "note": "threshold selects nothing on the holdout"}

    prec = float(np.mean(np.asarray(y_te)[sel]))
    lift_pp = 100.0 * (prec - base)
    se_pp = 100.0 * math.sqrt(max(base * (1.0 - base), 1e-12) / n_sel)
    out = {"n_at_threshold": n_sel,
           "precision_at_threshold": round(prec, 4),
           "base_rate": round(base, 4),
           "lift_over_base_pp": round(lift_pp, 2),
           "lift_se_pp": round(se_pp, 2),
           "note": "measured on the holdout — this threshold had no say in its own selection"}
    if n_sel < MIN_HOLDOUT_SELECTIONS:
        out["lift_in_std_errors"] = None
        out["note"] = (f"only {n_sel} holdout selections (<{MIN_HOLDOUT_SELECTIONS}) — "
                       f"too few to distinguish signal from noise")
    else:
        out["lift_in_std_errors"] = round(lift_pp / se_pp, 2) if se_pp > 0 else None
    return out


def _holdout_verdict(holdout: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Does the holdout confirm the threshold well enough to trade it?

    HONEST CAVEAT: using the holdout to make this go/no-go call means it is no
    longer a pristine holdout for that decision. The justification is that this
    is a single conservative bit — trade or don't — rather than parameter
    selection, and the two errors are wildly asymmetric: wrongly suppressing a
    market costs nothing but time, wrongly opening one costs money. The only
    genuinely clean confirmation is live closing-line value.
    """
    n = holdout.get("n_at_threshold")
    if not n:
        return False, "threshold selects nothing on the holdout"
    if n < MIN_HOLDOUT_SELECTIONS:
        return False, f"only {n} holdout selections"
    se = holdout.get("lift_in_std_errors")
    if se is None:
        return False, "lift not measurable"
    if se < MIN_HOLDOUT_LIFT_SE:
        return False, (f"holdout lift {holdout.get('lift_over_base_pp')}pp is only {se} standard "
                       f"errors (need {MIN_HOLDOUT_LIFT_SE}) — consistent with noise")
    return True, f"holdout lift {holdout.get('lift_over_base_pp')}pp at {se} standard errors"


def _decide_threshold(y_ca, p_ca, y_te, p_te, label: str, buf: "SettingsBuffer",
                      summary: Dict[str, Any], target_precision: float, min_preds: int,
                      min_thresh: float, max_thresh: float, default_thr_prob: float,
                      ctx: str, extra_diag: Optional[Dict[str, Any]] = None
                      ) -> Tuple[float, Dict[str, Any], Dict[str, Any]]:
    """
    Pick on CAL, verify on HOLDOUT, write only if confirmed.

    Shared by every market — the plain binary heads, 1X2, Double Chance and
    Draw No Bet — so no market can acquire a threshold without passing the same
    bar. That was the gap that let Double Chance run on an unvalidated default.
    """
    thr_prob, diag = _pick_threshold(y_ca, p_ca, target_precision, min_preds,
                                     default_thr_prob, max_thresh_pct=max_thresh)
    if extra_diag:
        diag.update(extra_diag)
    thr_pct = float(np.clip(thr_prob * 100.0, min_thresh, max_thresh))
    holdout = _threshold_on_holdout(y_te, p_te, thr_pct / 100.0)

    if diag.get("method") != "suppressed":
        confirmed, why = _holdout_verdict(holdout)
        holdout["verdict"] = why
        if not confirmed:
            thr_pct = float(max_thresh)
            holdout["action"] = "SUPPRESSED — calibration lift did not survive the holdout"
            logger.warning("[HOLDOUT] %s: %s. Calibration said %s. Suppressing at %.1f%%.",
                           ctx, why, diag.get("lift_over_base_pp"), thr_pct)
        else:
            holdout["action"] = "confirmed — threshold kept"

    buf.set_threshold(label, thr_pct, summary)
    return thr_pct, diag, holdout


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
    threshold on CAL -> verify on HOLDOUT -> metrics on HOLDOUT.
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
        mets["mean_predicted"] = float(np.mean(p_te))
        mets["mean_actual"] = float(np.mean(y_te))
        mets["calibration_gap_pct"] = round(100.0 * (mets["mean_actual"] - mets["mean_predicted"]), 2)
        logger.info("[METRICS] %s: acc=%.3f prec=%.3f brier=%.4f calib_gap=%+.2fpp (C=%g)",
                    ctx, mets["acc"], mets["precision"], mets["brier"],
                    mets["calibration_gap_pct"], C)
    else:
        mets["metrics_source"] = "unavailable (empty or single-class holdout)"

    mets["feature_importance"] = dict(sorted(
        zip(feature_names, m.coef_.ravel().tolist()), key=lambda x: abs(x[1]), reverse=True)[:10])

    if threshold_label:
        thr_pct, diag, holdout = _decide_threshold(
            y_ca, p_ca, y_te, p_te, threshold_label, buf, summary,
            target_precision, min_preds, min_thresh_pct, max_thresh_pct,
            default_thr_prob, ctx)
        mets["threshold_pct"] = round(thr_pct, 2)
        mets["threshold_selection"] = diag
        mets["holdout_at_threshold"] = holdout

    return True, mets, p_ca, p_te


def _effective_min_rows(n_features: int, min_rows_env: int, rows_per_feature: int) -> int:
    """
    Below roughly 10-20 events per parameter, logistic regression drives toward
    perfect separation, probabilities collapse to 0/1, and everything clears
    threshold — a complete explanation for chronic overconfidence on its own.
    """
    return max(int(min_rows_env), int(rows_per_feature) * int(n_features))


# ─────────────────── 1X2 and derived markets ─────────────────── #

def _wld_triple(heads: Dict[str, Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]],
                idx: int) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Normalised (p_home, p_draw, p_away) over Home+Draw+Away for the cal split
    (idx=1) or holdout split (idx=2). Mirrors main.py._wld_probs exactly.
    """
    def _get(key):
        ok, *arrs = heads.get(key, (False, None, None))
        return ok, (arrs[idx - 1] if len(arrs) >= idx else None)

    ok_h, p_h = _get("WLD_HOME")
    ok_d, p_d = _get("WLD_DRAW")
    ok_a, p_a = _get("WLD_AWAY")
    if not (ok_h and ok_a) or p_h is None or p_a is None or len(p_h) == 0:
        return None
    p_hc = np.clip(p_h, EPS, 1 - EPS)
    p_ac = np.clip(p_a, EPS, 1 - EPS)
    if ok_d and p_d is not None and len(p_d) == len(p_hc):
        p_dc = np.clip(p_d, EPS, 1 - EPS)
    else:
        p_dc = np.full_like(p_hc, FALLBACK_DRAW_PROB)
    s = np.maximum(p_hc + p_dc + p_ac, EPS)
    return p_hc / s, p_dc / s, p_ac / s


def _fit_1x2_threshold(heads, gd: np.ndarray, m_ca: np.ndarray, m_te: np.ndarray,
                       buf: SettingsBuffer, summary: Dict[str, Any],
                       label: str, target_precision: float, min_preds: int,
                       min_thresh: float, max_thresh: float) -> bool:
    """
    Pick the 1X2 threshold on EXACTLY the statistic serving compares against:
    the 3-way normalised per-side probability, pooled over home and away.

    Returns True if the threshold survived its holdout. Double Chance and Draw
    No Bet inherit that verdict — they are transforms of these same heads.
    """
    tri_ca = _wld_triple(heads, idx=1)
    if tri_ca is None:
        logger.info("[1X2] %s: home/away heads unavailable — threshold not set", label)
        return False
    phn_ca, _pdn_ca, pan_ca = tri_ca
    gd_ca = gd[m_ca]
    probs_ca = np.concatenate([phn_ca, pan_ca])
    ys_ca = np.concatenate([(gd_ca > 0).astype(int), (gd_ca < 0).astype(int)])

    probs_te = ys_te = np.array([])
    tri_te = _wld_triple(heads, idx=2)
    if tri_te is not None:
        phn_te, _pdn_te, pan_te = tri_te
        gd_te = gd[m_te]
        if len(gd_te) == len(phn_te):
            probs_te = np.concatenate([phn_te, pan_te])
            ys_te = np.concatenate([(gd_te > 0).astype(int), (gd_te < 0).astype(int)])

    caveat = {"base_rate_caveat": (
        "pooled home-or-away win rate; a high lift here is mechanical (favourites win more "
        "than an average side) and is not an edge. Benchmark against the de-vigged market "
        "price instead.")}
    thr_pct, diag, holdout = _decide_threshold(
        ys_ca, probs_ca, ys_te, probs_te, label, buf, summary,
        target_precision, min_preds, min_thresh, max_thresh, 0.45, label, extra_diag=caveat)

    summary.setdefault("metrics", {})[f"{label}_threshold_diag"] = diag
    summary["metrics"][f"{label}_holdout_at_threshold"] = holdout
    confirmed = thr_pct < max_thresh and diag.get("method") != "suppressed"
    logger.info("[1X2] %s threshold %.2f%% (%s); holdout lift %s; confirmed=%s",
                label, thr_pct, diag.get("method"), holdout.get("lift_over_base_pp"), confirmed)
    return confirmed


MIN_DERIVED_LIFT_PP = float(os.getenv("MIN_DERIVED_LIFT_PP", "3.0"))


def _fit_derived_market_thresholds(heads, gd: np.ndarray, m_ca: np.ndarray, m_te: np.ndarray,
                                   buf: SettingsBuffer, summary: Dict[str, Any],
                                   prefix: str, target_precision: float, min_preds: int,
                                   min_thresh: float, max_thresh: float,
                                   parent_confirmed: bool) -> None:
    """
    Double Chance and Draw No Bet, verified rather than defaulted.

    No new model is fitted. The probabilities come from the already calibrated
    1X2 triple via feature_spec.derive_dc_dnb — the same function serving uses —
    and go through the identical pick-on-CAL, verify-on-HOLDOUT path as
    everything else.

    TWO EXTRA GATES, both forced by the first live run of this code:

      PRE 1X2          lift 16.69pp = 2.19 SE  -> SUPPRESSED
      PRE Double Chance lift  1.40pp           -> confirmed at 55%
      PRE Draw No Bet   lift  6.53pp           -> confirmed at 55%

    1. PARENT GATE. Double Chance and Draw No Bet are transforms of the very
       heads that just failed their own holdout. It is incoherent for a derived
       market to trade when the market it is derived from does not, so they now
       inherit the 1X2 verdict: parent suppressed means derived suppressed.

    2. ECONOMIC FLOOR. Double Chance pools three selections across the whole
       holdout (~7,600 rows), which makes the standard error tiny — a 1.40pp
       lift clears 2.5 SE purely on sample size. But DC prices sit at 1.15-1.40,
       where 1.4pp over the base rate does not begin to cover the margin.
       Statistical significance is necessary, not sufficient; a derived market
       must also show at least MIN_DERIVED_LIFT_PP of actual lift.

    Draw No Bet voids on a draw, so drawn fixtures are excluded from its sample
    rather than counted as losses — matching how main.py grades it.
    """
    if not parent_confirmed:
        for name in ("Double Chance", "Draw No Bet"):
            label = f"{prefix}{name}"
            buf.set_threshold(label, float(max_thresh), summary)
            summary.setdefault("metrics", {})[f"{label}_holdout_at_threshold"] = {
                "action": "SUPPRESSED — parent 1X2 market did not pass its own holdout",
                "derived_from": f"{prefix}WLD_* heads",
                "lift_in_std_errors": None,
            }
        logger.warning("[DERIVED] %sDouble Chance / %sDraw No Bet suppressed at %.1f%% — "
                       "the %s1X2 heads they derive from failed their holdout.",
                       prefix, prefix, max_thresh, prefix)
        return

    tri_ca = _wld_triple(heads, idx=1)
    if tri_ca is None:
        logger.info("[DERIVED] %sDouble Chance / Draw No Bet: 1X2 heads unavailable — "
                    "no thresholds set", prefix)
        return
    tri_te = _wld_triple(heads, idx=2)

    def _selections(tri, mask):
        """Returns (dc_probs, dc_labels, dnb_probs, dnb_labels) for one split."""
        if tri is None:
            return None
        ph, pd_, pa = tri
        g = gd[mask]
        if len(g) != len(ph):
            return None
        d = [derive_dc_dnb(a, b, c) for a, b, c in zip(ph, pd_, pa)]
        p_1x = np.array([x["1X"] for x in d])
        p_x2 = np.array([x["X2"] for x in d])
        p_12 = np.array([x["12"] for x in d])
        p_dh = np.array([x["DNB_Home"] for x in d])
        p_da = np.array([x["DNB_Away"] for x in d])
        dc_p = np.concatenate([p_1x, p_x2, p_12])
        dc_y = np.concatenate([(g >= 0).astype(int), (g <= 0).astype(int), (g != 0).astype(int)])
        nd = g != 0  # Draw No Bet: drop draws entirely (stake returned).
        dnb_p = np.concatenate([p_dh[nd], p_da[nd]])
        dnb_y = np.concatenate([(g[nd] > 0).astype(int), (g[nd] < 0).astype(int)])
        return dc_p, dc_y, dnb_p, dnb_y

    sel_ca = _selections(tri_ca, m_ca)
    sel_te = _selections(tri_te, m_te)
    if sel_ca is None:
        return
    dc_p_ca, dc_y_ca, dnb_p_ca, dnb_y_ca = sel_ca
    if sel_te is None:
        dc_p_te = dc_y_te = dnb_p_te = dnb_y_te = np.array([])
    else:
        dc_p_te, dc_y_te, dnb_p_te, dnb_y_te = sel_te

    for name, p_ca, y_ca, p_te, y_te, note in (
        ("Double Chance", dc_p_ca, dc_y_ca, dc_p_te, dc_y_te,
         "pooled 1X/X2/12; base rate is ~2/3 because each selection covers two of three "
         "outcomes. Derived from the 1X2 heads — no independent model."),
        ("Draw No Bet", dnb_p_ca, dnb_y_ca, dnb_p_te, dnb_y_te,
         "draws excluded (stake returned). Derived from the 1X2 heads — no independent model."),
    ):
        label = f"{prefix}{name}"
        if len(y_ca) == 0 or len(np.unique(y_ca)) < 2:
            logger.info("[DERIVED] %s: no usable calibration sample — suppressing at %.1f%%",
                        label, max_thresh)
            buf.set_threshold(label, float(max_thresh), summary)
            continue
        thr_pct, diag, holdout = _decide_threshold(
            y_ca, p_ca, y_te, p_te, label, buf, summary,
            target_precision, min_preds, min_thresh, max_thresh, 0.65, label,
            extra_diag={"derived_from": f"{prefix}WLD_* heads", "note": note})

        # Economic floor, applied after the statistical one.
        lift = holdout.get("lift_over_base_pp")
        if thr_pct < max_thresh and lift is not None and lift < MIN_DERIVED_LIFT_PP:
            buf.set_threshold(label, float(max_thresh), summary)
            thr_pct = float(max_thresh)
            holdout["action"] = (f"SUPPRESSED — holdout lift {lift}pp is below the "
                                 f"{MIN_DERIVED_LIFT_PP}pp economic floor for a derived market. "
                                 f"Significant only because the pooled sample is large; at "
                                 f"1.15-1.40 prices it does not cover the margin.")
            logger.warning("[DERIVED] %s: lift %.2fpp below the %.1fpp economic floor — "
                           "suppressing at %.1f%%.", label, lift, MIN_DERIVED_LIFT_PP, max_thresh)

        summary.setdefault("metrics", {})[f"{label}_threshold_diag"] = diag
        summary["metrics"][f"{label}_holdout_at_threshold"] = holdout
        logger.info("[DERIVED] %s threshold %.2f%% (%s); holdout lift %s",
                    label, thr_pct, diag.get("method"), holdout.get("lift_over_base_pp"))


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
                ok, mets, p_ca, p_te = _train_binary_head(
                    buf, X, y.astype(int), m_tr, m_ca, m_te, FEATURES,
                    key, None, summary, target_precision, min_preds,
                    min_thresh, max_thresh, 0.45, key)
                summary["trained"][key] = ok
                if ok:
                    summary["metrics"][key] = mets
                heads[key] = (ok, p_ca, p_te)

            parent_ok = _fit_1x2_threshold(heads, gd, m_ca, m_te, buf, summary, "1X2",
                                           target_precision, min_preds, min_thresh, max_thresh)
            _fit_derived_market_thresholds(heads, gd, m_ca, m_te, buf, summary, "",
                                           target_precision, min_preds, min_thresh, max_thresh,
                                           parent_confirmed=parent_ok)
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
            for key, y in (("PRE_WLD_HOME", (gd > 0)), ("PRE_WLD_DRAW", (gd == 0)),
                           ("PRE_WLD_AWAY", (gd < 0))):
                ok, mets, p_ca, p_te = _train_binary_head(
                    buf, Xp, y.astype(int), m_tr, m_ca, m_te, PRE_FEATURES,
                    key, None, summary, target_precision, min_preds,
                    min_thresh, max_thresh, 0.45, key)
                summary["trained"][key] = ok
                if ok:
                    summary["metrics"][key] = mets
                heads[key.replace("PRE_", "")] = (ok, p_ca, p_te)

            parent_ok = _fit_1x2_threshold(heads, gd, m_ca, m_te, buf, summary, "PRE 1X2",
                                           target_precision, min_preds, min_thresh, max_thresh)
            _fit_derived_market_thresholds(heads, gd, m_ca, m_te, buf, summary, "PRE ",
                                           target_precision, min_preds, min_thresh, max_thresh,
                                           parent_confirmed=parent_ok)
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

        # Nothing has been written to `settings` until this line.
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
