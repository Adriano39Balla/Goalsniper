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
    ODDS_TRUSTED_FROM_TS,
    DEFAULT_LEAGUE_RATES, FEATURES, CORE_FEATURES, PRE_FEATURES,
    LEAGUE_RATE_FIELDS_INPLAY, LEAGUE_RATE_FIELDS_PREMATCH,
    MARKET_ANCHOR, anchor_logit,
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

# Snapshots harvested before this instant carry market_fair_* values derived
# from contaminated odds: _market_name_normalize() folded team totals, half
# markets and other non-full-match bets into the same market keys, and
# fetch_odds keeps the BEST price per selection, so the wrong (longer) price
# won and the de-vigged "consensus" was not the market's read at all.
#
# Training on those teaches the model a market prior that never existed. They
# are dropped rather than kept, which routes them through
# build_inplay_features' NEUTRAL_MARKET_PRIORS path - the same treatment as
# every snapshot harvested before the feature existed at all. Everything else
# in those snapshots (goals, shots, corners, cards) came from the statistics
# feed and is unaffected, so the rows themselves stay.
#
# The boundary itself lives in feature_spec, shared with main.py's P&L: both
# sides have to agree on when recorded odds became trustworthy, and a second
# copy here would be one more pair of constants to drift apart.
MARKET_FAIR_TRUSTED_FROM_TS = ODDS_TRUSTED_FROM_TS

_MARKET_FAIR_KEYS = ("market_fair_home", "market_fair_draw", "market_fair_away",
                     "market_fair_over25", "market_fair_btts_yes")


def _drop_untrusted_market_fair(raw: Dict[str, Any], created_ts: Optional[int]) -> int:
    """Strip market_fair_* from a snapshot harvested before the odds fix."""
    if created_ts is None or created_ts >= MARKET_FAIR_TRUSTED_FROM_TS:
        return 0
    dropped = 0
    for k in _MARKET_FAIR_KEYS:
        if raw.pop(k, None) is not None:
            dropped = 1
    return dropped


def already_decided_mask(df: pd.DataFrame, head: str) -> Optional[np.ndarray]:
    """
    Rows whose outcome was ALREADY SETTLED at the moment they were harvested.

    An in-play snapshot at 2-1 has already answered "Over 2.5?" and "both
    teams to score?". Those rows are not predictions - the label is a fact
    about the scoreline the features already contain, so the model gets them
    right for free. They inflate accuracy and precision, they dominate
    calibration, and none of them is bettable: nobody prices a market that
    has resolved.

    Their fingerprint is visible in the fitted weights - OU_2.5's two
    strongest features are goals_sum and is_goalfest (goals_sum >= 3, i.e.
    "Over 2.5 has already happened"), and BTTS_YES leans on goals_sum harder
    than on anything else.

    Returns None for heads that cannot be settled early: a side can always
    score, so 1X2/DNB/DC are mathematically open until full time however
    lopsided the scoreline is.
    """
    if "goals_sum" not in df.columns:
        return None
    goals_sum = df["goals_sum"].to_numpy(dtype=float)

    if head.startswith("OU_"):
        try:
            line = float(head.split("_", 1)[1])
        except (ValueError, IndexError):
            return None
        # Over is settled once the line is cleared; Under never settles early.
        return goals_sum > line

    if head == "BTTS_YES":
        if "score_margin" not in df.columns:
            return None
        # min(home, away) = (sum - |diff|) / 2; both have scored when that is >= 1.
        return (goals_sum - df["score_margin"].to_numpy(dtype=float)) >= 2.0

    return None


def decided_diagnostics(df: pd.DataFrame, head: str, y: np.ndarray) -> Optional[Dict[str, Any]]:
    """
    How much of a head's apparent skill is answering settled questions.

    base_rate_undecided is the number that matters: the rate the model is
    actually up against once the free rows are removed. A head whose overall
    base rate is 0.50 but whose undecided base rate is 0.35 has been graded
    on a much easier problem than the one it is asked to bet on.
    """
    mask = already_decided_mask(df, head)
    if mask is None or len(mask) == 0:
        return None
    n = int(len(mask))
    n_decided = int(mask.sum())
    y = np.asarray(y, dtype=int)
    n_pos = int(y.sum())
    undecided = ~mask
    n_undecided = int(undecided.sum())
    return {
        "n_rows": n,
        "n_already_decided": n_decided,
        "decided_share_pct": round(100.0 * n_decided / n, 1) if n else 0.0,
        "share_of_positives_pct": (round(100.0 * int((mask & (y == 1)).sum()) / n_pos, 1)
                                   if n_pos else 0.0),
        "base_rate_all": round(float(y.mean()), 4) if n else 0.0,
        "base_rate_undecided": (round(float(y[undecided].mean()), 4) if n_undecided else None),
        "note": ("Rows whose outcome was already settled by the scoreline when harvested. "
                 "They are free accuracy and cannot be bet. base_rate_undecided is the "
                 "honest benchmark for this head."),
    }


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
    untrusted_fair = 0
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

        untrusted_fair += _drop_untrusted_market_fair(raw, _as_int(row["created_ts"]))

        # Did this snapshot carry a REAL de-vigged market price, or is it about
        # to be filled with a neutral prior? build_inplay_features() substitutes
        # NEUTRAL_MARKET_PRIORS for anything missing, after which the two are
        # indistinguishable in the feature vector - so the question has to be
        # asked here, while the raw payload still knows.
        #
        # It matters because a market-anchored head is learning "where is the
        # market wrong". A row with no market has nothing to be wrong about;
        # training on it with a neutral offset teaches deviation from 0.5,
        # which is a different question and pure noise for this one.
        has_market = {k: (raw.get(k) is not None) for k in _MARKET_FAIR_KEYS}

        # League rates are injected after the split; a neutral placeholder here.
        f = build_inplay_features(raw, DEFAULT_LEAGUE_RATES)
        for k, present in has_market.items():
            f[f"_has_{k}"] = 1 if present else 0

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
    if untrusted_fair:
        logger.warning("[LOAD] %d in-play snapshots harvested before the odds fix — their "
                       "market_fair_* values came from contaminated prices and were dropped "
                       "to the neutral prior. Everything else in those rows is kept. This "
                       "count falls to 0 as the window ages out.", untrusted_fair)
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

def _standardize(X_tr: np.ndarray,
                 sw: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Feature mean/scale, weighted the same way the fit is.

    An unweighted scaler is set by whichever fixtures happened to be harvested
    longest. Since L2 penalises on the standardised scale, that would make the
    penalty itself depend on snapshot counts.
    """
    if sw is None:
        mean, scale = X_tr.mean(axis=0), X_tr.std(axis=0)
    else:
        w = np.asarray(sw, dtype=float)
        tot = w.sum()
        if tot <= 0:
            mean, scale = X_tr.mean(axis=0), X_tr.std(axis=0)
        else:
            mean = (w[:, None] * X_tr).sum(axis=0) / tot
            var = (w[:, None] * (X_tr - mean) ** 2).sum(axis=0) / tot
            scale = np.sqrt(np.maximum(var, 0.0))
    scale = np.asarray(scale, dtype=float).copy()
    scale[scale < 1e-9] = 1.0  # constant column -> leave as (x - mean)
    return np.asarray(mean, dtype=float), scale


def match_weights(match_ids: np.ndarray) -> np.ndarray:
    """
    One match, one observation.

    Nine snapshots of the same fixture share a single outcome. They are not
    nine independent observations, but the fit counted them as such, so the
    objective saw an effective sample ~9x larger than the data contains and
    selected C as if regularisation mattered ~9x less than it does. Systematic
    under-regularisation is precisely how a model acquires the wide, noisy
    deviation from the market that the price gate then selects the profitable
    tail of.

    Each row is weighted 1/(snapshots for its match), and the weights are
    rescaled to sum to the MATCH count. That keeps C's meaning "per
    observation" while making an observation a match rather than a snapshot,
    so the existing C_GRID still spans the useful range.

    It also removes a second, quieter bias: a fixture harvested for 70 minutes
    contributed twice the weight of one harvested for 35, for no reason
    connected to how much either tells us.
    """
    ids = np.asarray(match_ids)
    _, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    w = 1.0 / counts[inverse].astype(float)
    total = w.sum()
    return w * (len(counts) / total) if total > 0 else w


def effective_n(match_ids: Optional[np.ndarray], n_rows: int) -> int:
    """
    How many independent observations a split really holds.

    Not the Kish effective sample size: Kish measures the efficiency loss from
    UNEQUAL weights among rows and is blind to clustering, so on 21 rows across
    3 fixtures it reports 16.2 — a number that reads as sample size and is not.
    Every row of a fixture shares one outcome, so the count of fixtures is the
    answer.
    """
    if match_ids is None:
        return int(n_rows)
    return int(len(np.unique(np.asarray(match_ids))))


def _fit_lr(X: np.ndarray, y: np.ndarray, C: float,
            sw: Optional[np.ndarray] = None) -> Optional[LogisticRegression]:
    if len(np.unique(y)) < 2:
        return None
    # No class_weight="balanced": it rebases the intercept toward a 50/50 prior,
    # a large systematic upward bias on low-prevalence markets. The objective
    # here is a calibrated probability, not recall on a rare class.
    return LogisticRegression(max_iter=3000, solver="liblinear", C=C).fit(X, y, sample_weight=sw)


class OffsetLogit:
    """
    L2-penalised logistic regression with a per-row OFFSET whose coefficient is
    fixed at 1.0:

        logit(p_i) = offset_i + intercept + w . x_i

    scikit-learn has no offset support, so this is fitted directly. The
    objective matches LogisticRegression's lbfgs solver exactly - 0.5*||w||^2
    penalised, intercept unpenalised, C scaling the data term - so C_GRID keeps
    the same meaning across anchored and unanchored heads and the two paths
    stay comparable.

    The offset is the market's log-odds (see feature_spec.MARKET_ANCHOR). Fixing
    its coefficient at 1.0 is what makes the fitted weights a DEVIATION from the
    market rather than a competing opinion about the outcome. Let the fit choose
    that coefficient and it shrinks under the penalty, which is exactly the
    behaviour being removed.

    Exposes coef_/intercept_ so it drops into build_model_blob() unchanged.
    """

    def __init__(self, C: float):
        self.C = float(C)
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray, offset: np.ndarray,
            sample_weight: Optional[np.ndarray] = None) -> "OffsetLogit":
        from scipy.optimize import minimize

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        off = np.asarray(offset, dtype=float)
        n, d = X.shape
        sw = (np.ones(n) if sample_weight is None
              else np.asarray(sample_weight, dtype=float))

        def obj(theta: np.ndarray) -> Tuple[float, np.ndarray]:
            w, c = theta[:d], theta[d]
            z = off + c + X @ w
            # log(1 + exp(z)) without overflow.
            ll = float(np.sum(sw * (np.logaddexp(0.0, z) - y * z)))
            loss = 0.5 * float(w @ w) + self.C * ll
            resid = sw * (1.0 / (1.0 + np.exp(-z)) - y)
            grad = np.empty(d + 1)
            grad[:d] = w + self.C * (X.T @ resid)
            grad[d] = self.C * float(np.sum(resid))
            return loss, grad

        res = minimize(obj, np.zeros(d + 1), jac=True, method="L-BFGS-B",
                       options={"maxiter": 3000})
        self.coef_ = res.x[:d].reshape(1, -1)
        self.intercept_ = np.array([res.x[d]])
        return self

    def decision_function(self, X: np.ndarray, offset: np.ndarray) -> np.ndarray:
        return np.asarray(offset, dtype=float) + float(self.intercept_[0]) + X @ self.coef_.ravel()

    def predict_proba_off(self, X: np.ndarray, offset: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.decision_function(X, offset)))


def _select_C(X_tr, y_tr, X_ca, y_ca,
              off_tr: Optional[np.ndarray] = None,
              off_ca: Optional[np.ndarray] = None,
              sw_tr: Optional[np.ndarray] = None,
              sw_ca: Optional[np.ndarray] = None):
    """
    Pick the regularization strength by log loss on the calibration split.

    With an offset supplied the anchored fit is used; the selection criterion
    is identical either way, so the two paths remain directly comparable.

    The calibration loss is weighted the same way the fit is. Selecting C
    against an UNWEIGHTED cal loss would pick the C that best serves whichever
    fixtures happen to have the most snapshots, undoing on the selection step
    what the weights fix on the fitting step.
    """
    anchored = off_tr is not None and off_ca is not None
    best_m, best_C, best_ll = None, C_GRID[-1], float("inf")
    for C in C_GRID:
        if anchored:
            if len(np.unique(y_tr)) < 2:
                return None, C_GRID[-1]
            try:
                m = OffsetLogit(C).fit(X_tr, y_tr, off_tr, sample_weight=sw_tr)
                p = m.predict_proba_off(X_ca, off_ca)
            except Exception as e:
                logger.warning("[ANCHOR] offset fit failed at C=%g: %s", C, e)
                continue
        else:
            m = _fit_lr(X_tr, y_tr, C, sw=sw_tr)
            if m is None:
                continue
            try:
                p = m.predict_proba(X_ca)[:, 1]
            except Exception:
                continue
        try:
            ll = log_loss(y_ca, np.clip(p, EPS, 1 - EPS), labels=[0, 1],
                          sample_weight=sw_ca)
        except Exception:
            continue
        if ll < best_ll:
            best_m, best_C, best_ll = m, C, ll
    return best_m, best_C


def _logit_vec(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_platt(y_true: np.ndarray, p_raw: np.ndarray,
              sw: Optional[np.ndarray] = None) -> Tuple[float, float]:
    z = _logit_vec(p_raw).reshape(-1, 1)
    lr = LogisticRegression(max_iter=1000, solver="lbfgs").fit(
        z, y_true.astype(int), sample_weight=sw)
    return float(lr.coef_.ravel()[0]), float(lr.intercept_.ravel()[0])


def _apply_platt(p_raw: np.ndarray, a: float, b: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(a * _logit_vec(p_raw) + b)))


def fit_platt_anchored(y_true: np.ndarray, dev: np.ndarray,
                       offset: np.ndarray,
                       sw: Optional[np.ndarray] = None
                       ) -> Tuple[float, float, Dict[str, Any]]:
    """
    Platt scaling for a market-anchored head, applied to the DEVIATION only:

        logit(p) = offset + a * dev + b

    Calibrating the sum instead would multiply the market's log-odds by `a`,
    un-pinning the coefficient anchoring exists to fix.

    `b` needs a second guard that an unanchored head does not. It is a free
    constant, and it soaks up whatever the calibration split's base rate
    happened to be. On a 360-row split the standard error of that base rate is
    ~2.6pp, so a couple of points of pure sampling noise land in `b` routinely
    - and for an anchored model that becomes a CONSTANT edge over the market on
    every future prediction. Measured on synthetic data with no signal at all:
    the model's own weights came out negligible (max 0.07, anchoring working),
    yet `b` alone put 43% of holdout rows more than 2pp above the market,
    against a FAIR_EDGE_MIN_BPS of exactly 2pp. That is a model that would tip
    constantly while knowing nothing.

    An unanchored head hides this because its own scatter dwarfs it. An
    anchored head is supposed to sit on the market unless it has something to
    say, so a constant shift is the whole failure mode.

    So both parameters are held at their null values (a=1, b=0) unless they
    clear CAL_MIN_SE standard errors. A genuinely mis-scaled market still gets
    corrected once there is enough data to show it; noise does not.
    """
    dev = np.asarray(dev, dtype=float)
    off = np.asarray(offset, dtype=float)
    y = np.asarray(y_true, dtype=float)
    m = OffsetLogit(1e6).fit(dev.reshape(-1, 1), y, off, sample_weight=sw)
    a, b = float(m.coef_.ravel()[0]), float(m.intercept_.ravel()[0])

    diag: Dict[str, Any] = {"a_fitted": round(a, 4), "b_fitted": round(b, 4)}
    try:
        p = 1.0 / (1.0 + np.exp(-(off + a * dev + b)))
        # Weighted Fisher information: the guard asks "is this distinguishable
        # from the null", and the answer depends on how many INDEPENDENT
        # observations there are. Unweighted standard errors here would be
        # computed against ~9x the real sample and would wave through exactly
        # the sampling noise the guard exists to catch.
        w = np.clip(p * (1.0 - p), 1e-9, None)
        if sw is not None:
            w = w * np.asarray(sw, dtype=float)
        D = np.column_stack([dev, np.ones_like(dev)])
        cov = np.linalg.pinv(D.T @ (w[:, None] * D))
        se_a, se_b = float(np.sqrt(max(cov[0, 0], 0.0))), float(np.sqrt(max(cov[1, 1], 0.0)))
    except Exception as e:
        logger.warning("[CAL] standard errors unavailable (%s) — holding calibration at "
                       "its null values rather than trusting an unchecked fit", e)
        # No standard errors means no way to know how much to trust either
        # parameter, so neither is applied: the head predicts the market.
        return 0.0, 0.0, {**diag, "a": 0.0, "b": 0.0,
                          "reason": "standard errors unavailable — collapsed to the market"}

    diag.update({"se_a": round(se_a, 4), "se_b": round(se_b, 4)})

    # Both parameters are SHRUNK TOWARD ZERO by how well they are measured,
    # rather than kept or discarded on a significance test.
    #
    # Zero is the conservative direction for both, and this is the part worth
    # being careful about. For `b` that is obvious - a nonzero constant is a
    # permanent shift off the market. For `a` it is the opposite of the
    # intuitive answer: a=1 means "trust the model's own scale completely",
    # which is the LEAST conservative setting, while a=0 collapses the head
    # onto the market price. An earlier version of this held `a` at 1.0 when it
    # was not significantly different from 1.0, and that measurably increased
    # the deviation the price gate sells as edge: the unguarded fit had been
    # supplying useful shrinkage (a~0.4) and the guard removed it.
    #
    # The factor is t^2/(1+t^2) with t the parameter's own t-statistic - the
    # standard reliability weight. A well-measured slope passes through intact;
    # one measured at t~1 is halved; noise collapses to the market. Continuous,
    # so there is no threshold to sit on the wrong side of.
    # The denominator k encodes how strong a prior each parameter faces: the
    # weight is t^2/(k + t^2), so a parameter reaches half its fitted value at
    # t = sqrt(k).
    #
    # They get different priors because they make different claims. `a` says
    # "the model's signal is real at roughly the scale fitted" - an ordinary
    # claim, so k=1 (half weight at t=1). `b` says "the de-vigged market is
    # systematically wrong by a constant, on every fixture" - a strong claim,
    # and the only two things it can be are a real market-wide bias, which
    # would show up loudly given data, or noise. It also applies to every bet
    # forever rather than varying by fixture, so it gets k=4 (half weight at
    # t=2). At t=1.3, which is ordinary sampling noise, that is the difference
    # between a 2.3pp permanent shift and a 1.1pp one.
    def _shrink(v: float, se: float, k: float) -> float:
        if not (se > 0) or not np.isfinite(v):
            return 0.0
        t2 = (v / se) ** 2
        return float(v * t2 / (k + t2))

    a_s, b_s = _shrink(a, se_a, CAL_PRIOR_K_SLOPE), _shrink(b, se_b, CAL_PRIOR_K_SHIFT)
    # A negative slope says the head is ANTI-predictive: apply it and the model
    # bets against its own signal. On a calibration split that is noise
    # essentially every time - a genuinely inverted feature set would be a bug
    # to fix, not an edge to trade - so it floors at the market.
    if a_s < 0.0:
        a_s = 0.0
        diag["slope_was_negative"] = True
    diag.update({"a": round(a_s, 4), "b": round(b_s, 4),
                 "shrinkage_a": round(a_s / a, 3) if a else None})
    logger.info("[CAL] anchored head: a %.3f -> %.3f (se %.3f), b %.3f -> %.3f (se %.3f)",
                a, a_s, se_a, b, b_s, se_b)
    return a_s, b_s, diag


def _apply_platt_anchored(dev: np.ndarray, offset: np.ndarray,
                          a: float, b: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(offset + a * np.asarray(dev, dtype=float) + b)))


def build_model_blob(model, features: List[str],
                     mean: np.ndarray, scale: np.ndarray,
                     cal: Tuple[float, float], C: float,
                     market_anchor: Optional[str] = None) -> Dict[str, Any]:
    """
    The blob carries its own scaler. main.py._linpred() applies
    (x - mean)/scale per feature before the dot product, so the model can be
    fitted on standardized features (making L2 scale-fair) without breaking
    serving parity.

    market_anchor names the feature whose log-odds serving must ADD to the
    linear predictor with a coefficient of 1.0. Its absence means an ordinary
    unanchored model, so old blobs keep serving unchanged.
    """
    blob: Dict[str, Any] = {
        "intercept": float(model.intercept_.ravel()[0]),
        "weights": {name: float(w) for name, w in zip(features, model.coef_.ravel().tolist())},
        "scaler": {"mean": {n: float(m) for n, m in zip(features, mean.tolist())},
                   "scale": {n: float(s) for n, s in zip(features, scale.tolist())}},
        "calibration": {"method": "platt", "a": float(cal[0]), "b": float(cal[1])},
        "C": float(C),
        "n_features": len(features),
    }
    if market_anchor:
        blob["market_anchor"] = market_anchor
    return blob


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
    offset_all: Optional[np.ndarray] = None,
    anchor_name: Optional[str] = None,
    weight_all: Optional[np.ndarray] = None,
    match_ids: Optional[np.ndarray] = None,
) -> Tuple[bool, Dict[str, Any], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Returns (ok, metrics, p_on_cal, p_on_holdout).

    Pipeline: standardize on TRAIN -> select C on CAL -> fit -> Platt on CAL ->
    threshold on CAL -> verify on HOLDOUT -> metrics on HOLDOUT.

    When offset_all is supplied the head is MARKET-ANCHORED: the market's
    log-odds enter the linear predictor with a coefficient fixed at 1.0 and the
    weights express only a deviation from it. See feature_spec.MARKET_ANCHOR.
    Callers pass the offset for the same rows as X_all, or None to fit the
    ordinary way.
    """
    ctx = metrics_name or model_key
    if not _validate(X_all, y_all, feature_names, ctx):
        return False, {}, None, None

    X_tr, y_tr = X_all[m_tr], y_all[m_tr]
    X_ca, y_ca = X_all[m_ca], y_all[m_ca]
    X_te, y_te = X_all[m_te], y_all[m_te]
    if offset_all is not None:
        off_tr, off_ca, off_te = offset_all[m_tr], offset_all[m_ca], offset_all[m_te]
    else:
        off_tr = off_ca = off_te = None
    if weight_all is not None:
        sw_tr, sw_ca = weight_all[m_tr], weight_all[m_ca]
    else:
        sw_tr = sw_ca = None

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

    mean, scale = _standardize(X_tr, sw_tr)
    Z_tr = (X_tr - mean) / scale
    Z_ca = (X_ca - mean) / scale
    Z_te = (X_te - mean) / scale

    m, C = _select_C(Z_tr, y_tr, Z_ca, y_ca, off_tr, off_ca, sw_tr, sw_ca)
    if m is None:
        return False, {}, None, None

    anchored = isinstance(m, OffsetLogit)
    if anchored:
        # The model's own contribution, with the market offset held out:
        # decision_function(X, 0) == intercept + w.x. main.py._score_prob()
        # splits the two the same way.
        dev_ca = m.decision_function(Z_ca, np.zeros(len(y_ca)))
        dev_te = m.decision_function(Z_te, np.zeros(len(y_te))) if len(y_te) else np.array([])
        a, b, cal_diag = fit_platt_anchored(y_ca, dev_ca, off_ca, sw=sw_ca)
        p_ca = _apply_platt_anchored(dev_ca, off_ca, a, b)
        p_te = (_apply_platt_anchored(dev_te, off_te, a, b) if len(y_te) else np.array([]))
    else:
        p_ca_raw = m.predict_proba(Z_ca)[:, 1]
        p_te_raw = m.predict_proba(Z_te)[:, 1] if len(y_te) else np.array([])
        a, b = fit_platt(y_ca, p_ca_raw, sw=sw_ca)
        p_ca = _apply_platt(p_ca_raw, a, b)
        p_te = _apply_platt(p_te_raw, a, b) if len(y_te) else np.array([])

    blob = build_model_blob(m, feature_names, mean, scale, (a, b), C,
                            market_anchor=anchor_name if anchored else None)
    for k in (f"model_latest:{model_key}", f"model:{model_key}"):
        buf.set(k, json.dumps(blob))

    mets: Dict[str, Any] = {"C": C, "n_train": int(len(y_tr)), "n_cal": int(len(y_ca)),
                            "n_holdout": int(len(y_te)), "prevalence": float(y_all.mean()),
                            "n_features": len(feature_names),
                            "market_anchored": bool(anchored),
                            "row_weighting": "per_match" if weight_all is not None else "per_row"}
    if match_ids is not None:
        # What the fit actually has to work with. n_train counts snapshots,
        # which is ~9x larger and reads as sample size without being it.
        mets["n_train_matches"] = effective_n(match_ids[m_tr], int(m_tr.sum()))
        mets["n_cal_matches"] = effective_n(match_ids[m_ca], int(m_ca.sum()))
        mets["n_holdout_matches"] = effective_n(match_ids[m_te], int(m_te.sum()))
    if anchored:
        mets["anchor_feature"] = anchor_name
        mets["calibration_fit"] = cal_diag
        # How far the model moves off the market on unseen rows. This is the
        # number the price gate is really trading on, and it is also the noise
        # the gate selects the upper tail of - so it belongs in the metrics
        # rather than being inferred from tips after the fact.
        if len(y_te):
            market_te = 1.0 / (1.0 + np.exp(-off_te))
            dev = p_te - market_te
            mets["deviation_from_market"] = {
                "mean_abs_pp": round(float(np.mean(np.abs(dev)) * 100), 2),
                "p95_abs_pp": round(float(np.percentile(np.abs(dev), 95) * 100), 2),
                "max_abs_pp": round(float(np.max(np.abs(dev)) * 100), 2),
            }

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


# A market-anchored head is fitted ONLY on rows that carried a real de-vigged
# market price. Rows without one have nothing for the model to deviate FROM.
# Below this many such rows the anchored fit is not attempted and the head
# falls back to the ordinary one - loudly, in the digest, never silently.
MIN_ANCHORED_ROWS = int(os.getenv("MIN_ANCHORED_ROWS", "1500"))
MIN_ANCHORED_MATCHES = int(os.getenv("MIN_ANCHORED_MATCHES", "200"))


def frame_anchor_mask(df: pd.DataFrame) -> Optional[np.ndarray]:
    """
    Rows whose snapshot carried a REAL de-vigged market price for every market
    the anchored heads need, or None when the frame cannot answer the question.

    It is not enough that the columns are populated: build_inplay_features()
    fills anything missing with NEUTRAL_MARKET_PRIORS, and a neutral 0.5 is
    indistinguishable from a genuine 50/50 quote once it reaches the matrix.
    load_inplay_data() therefore records presence at load time, in `_has_*`.
    Its absence here (prematch frames, legacy loaders) means "unknown", which
    must read as not anchored - treating it as "all anchored" would anchor the
    model to neutral priors, the one outcome worth avoiding.

    All the anchored heads are required together because they are fitted on one
    shared row set; see the caller for why.
    """
    flags = [f"_has_{f}" for f in set(MARKET_ANCHOR.values())]
    if any(f not in df.columns for f in flags):
        return None
    mask = np.ones(len(df), dtype=bool)
    for f in flags:
        mask &= df[f].to_numpy(dtype=int) == 1
    return mask


def frame_anchor_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Whether the in-play block can be market-anchored, in numbers, for the digest."""
    mask = frame_anchor_mask(df)
    if mask is None:
        return {"anchored": False,
                "reason": "snapshots do not record whether a real market price was present"}
    n = int(mask.sum())
    matches = int(df.loc[mask, "_match_id"].nunique()) if n else 0
    ok = n >= MIN_ANCHORED_ROWS and matches >= MIN_ANCHORED_MATCHES
    out: Dict[str, Any] = {
        "anchored": ok, "anchored_rows": n, "anchored_matches": matches,
        "rows_required": MIN_ANCHORED_ROWS, "matches_required": MIN_ANCHORED_MATCHES,
        "anchored_share_pct": round(100.0 * n / max(1, len(df)), 1),
        "_mask": mask,
    }
    if not ok:
        out["reason"] = (f"only {n} rows / {matches} fixtures carry a real market price "
                         f"(need {MIN_ANCHORED_ROWS} / {MIN_ANCHORED_MATCHES}). Odds recorded "
                         f"before the market-name fix are not counted, so this number starts "
                         f"near zero and grows as clean snapshots accumulate.")
    return out


# Shrinkage priors for the anchored head's calibration. See fit_platt_anchored().
CAL_PRIOR_K_SLOPE = float(os.getenv("CAL_PRIOR_K_SLOPE", "1.0"))
CAL_PRIOR_K_SHIFT = float(os.getenv("CAL_PRIOR_K_SHIFT", "4.0"))

# Which in-play feature set to fit on: "auto" compares them on the calibration
# split every night, "core" or "full" force one. See feature_spec.CORE_FEATURES
# for what the reduced set drops and why.
FEATURE_SET = (os.getenv("FEATURE_SET", "auto") or "auto").strip().lower()

# "per_match" weights each row 1/(snapshots for its fixture); "per_row" is the
# old behaviour. See match_weights() for the argument, and note that the
# argument is statistical rather than empirical: synthetic data could not
# settle it either way, so this is switchable and reported.
ROW_WEIGHTING = (os.getenv("ROW_WEIGHTING", "per_match") or "per_match").strip().lower()


def collinearity_report(df: pd.DataFrame, cols: List[str],
                        high: float = 0.95, top: int = 5) -> Dict[str, Any]:
    """
    How much of a feature set restates itself, measured on the real data.

    CORE_FEATURES is a hypothesis about which columns are algebraic
    restatements of others. This checks that hypothesis against what is
    actually in the database rather than against an argument about the code,
    and it is the number to look at before trusting - or discarding - the
    reduced set.
    """
    usable = [c for c in cols if c in df.columns]
    if len(usable) < 2:
        return {}
    X = df[usable].to_numpy(dtype=float)
    keep = X.std(axis=0) > 1e-12
    names = [n for n, k in zip(usable, keep) if k]
    if len(names) < 2:
        return {"n_features": len(usable), "note": "no varying columns"}
    C = np.corrcoef(X[:, keep], rowvar=False)
    iu = np.triu_indices(len(names), k=1)
    vals = np.abs(C[iu])
    order = np.argsort(vals)[::-1][:top]
    return {
        "n_features": len(usable),
        "pairs_above_%.2f" % high: int((vals >= high).sum()),
        "max_abs_corr": round(float(vals.max()), 3),
        "most_collinear": [
            {"a": names[iu[0][i]], "b": names[iu[1][i]], "r": round(float(vals[i]), 3)}
            for i in order],
    }


def _cal_loss_for(X: np.ndarray, y: np.ndarray, m_tr, m_ca,
                  offset: Optional[np.ndarray],
                  weights: Optional[np.ndarray]) -> Optional[float]:
    """Weighted calibration-split log loss for one candidate feature matrix."""
    X_tr, X_ca = X[m_tr], X[m_ca]
    y_tr, y_ca = y[m_tr], y[m_ca]
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_ca)) < 2:
        return None
    sw_tr = weights[m_tr] if weights is not None else None
    sw_ca = weights[m_ca] if weights is not None else None
    off_tr = offset[m_tr] if offset is not None else None
    off_ca = offset[m_ca] if offset is not None else None
    mean, scale = _standardize(X_tr, sw_tr)
    m, _C = _select_C((X_tr - mean) / scale, y_tr, (X_ca - mean) / scale, y_ca,
                      off_tr, off_ca, sw_tr, sw_ca)
    if m is None:
        return None
    Z_ca = (X_ca - mean) / scale
    p = (m.predict_proba_off(Z_ca, off_ca) if isinstance(m, OffsetLogit)
         else m.predict_proba(Z_ca)[:, 1])
    try:
        return float(log_loss(y_ca, np.clip(p, EPS, 1 - EPS), labels=[0, 1],
                              sample_weight=sw_ca))
    except Exception:
        return None


def select_feature_set(df: pd.DataFrame, y: np.ndarray, m_tr, m_ca,
                       candidates: Dict[str, List[str]],
                       offset: Optional[np.ndarray] = None,
                       weights: Optional[np.ndarray] = None,
                       ) -> Tuple[str, List[str], Dict[str, Any]]:
    """
    Choose between the full and reduced feature sets on the CALIBRATION split.

    The reduced set is a hypothesis about which of the 56 columns are
    collinear restatements of each other (see feature_spec.CORE_FEATURES). It
    should not be trusted on the strength of that argument alone, and it does
    not have to be: cal is already the model-selection split - it is where C is
    chosen - so asking it one more question is the same kind of decision, not a
    new kind.

    Ties and failures fall back to the full set: the reduced one has to EARN
    the swap, so a bug here degrades to today's behaviour rather than to a
    quietly different model.

    Caveat worth stating: this adds one more binary choice made on a small
    split, so a little of the cal loss improvement will be selection noise.
    The holdout numbers in the digest are the ones to believe.
    """
    scores: Dict[str, Optional[float]] = {}
    for name, cols in candidates.items():
        usable = [c for c in cols if c in df.columns]
        if len(usable) != len(cols):
            scores[name] = None
            continue
        scores[name] = _cal_loss_for(df[cols].to_numpy(dtype=float), y,
                                     m_tr, m_ca, offset, weights)

    diag: Dict[str, Any] = {
        "cal_logloss": {k: (round(v, 5) if v is not None else None) for k, v in scores.items()},
        "n_features": {k: len(v) for k, v in candidates.items()},
    }
    ranked = [(v, k) for k, v in scores.items() if v is not None]
    if not ranked:
        diag["chosen"] = "full"
        diag["reason"] = "no candidate produced a usable calibration loss"
        return "full", candidates["full"], diag

    best_loss, best = min(ranked)
    full_loss = scores.get("full")
    if best != "full" and full_loss is not None and not (best_loss < full_loss):
        best, best_loss = "full", full_loss
    diag["chosen"] = best
    if full_loss is not None and best_loss is not None:
        diag["improvement_vs_full"] = round(full_loss - best_loss, 5)
    return best, candidates[best], diag


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

            # ── Market anchoring, decided ONCE for the whole in-play block ──
            # Not per head. The three WLD heads feed a shared 1X2 threshold
            # fitter that compares their probabilities row-for-row against the
            # same goal-difference array, so heads fitted on different row
            # subsets would misalign. Restricting the frame before the split
            # keeps every downstream array on the same rows, and it is the
            # more defensible choice anyway: all heads should see the same
            # data.
            anchor_rep = frame_anchor_report(df_ip)
            summary["market_anchoring"] = anchor_rep
            if anchor_rep["anchored"]:
                df_ip = df_ip.loc[anchor_rep["_mask"]].reset_index(drop=True)
                logger.info("[ANCHOR] in-play heads anchored to the market: %d rows / "
                            "%d fixtures carry a real de-vigged price (%.1f%% of the set)",
                            anchor_rep["anchored_rows"], anchor_rep["anchored_matches"],
                            anchor_rep["anchored_share_pct"])
            else:
                logger.info("[ANCHOR] in-play heads NOT anchored: %s", anchor_rep.get("reason"))
            anchor_rep.pop("_mask", None)
            # The digest reports what was TRAINED ON. Leaving the pre-subset
            # counts here would show 15,806 rows next to a model fitted on a
            # fraction of them, which is the kind of number that gets trusted.
            summary["data_stats"]["inplay_rows_trained"] = int(len(df_ip))
            summary["data_stats"]["inplay_matches_trained"] = int(df_ip["_match_id"].nunique())

            m_tr, m_ca, m_te = grouped_time_split(df_ip, cal_size, test_size, embargo_groups)

            rate_map = _compute_league_rate_map(conn, df_ip.loc[m_tr, "_match_id"].unique())
            df_ip = _apply_league_rates(df_ip, rate_map, LEAGUE_RATE_FIELDS_INPLAY)

            # ── One match, one observation ──
            # Nine snapshots of a fixture share a single outcome, so counting
            # them as nine independent rows told the fit it had ~9x the sample
            # it has, and C was selected as if regularisation mattered ~9x
            # less. Under-regularisation is how the model acquires the wide
            # deviation from the market that the gate selects the tail of.
            ip_match_ids = df_ip["_match_id"].to_numpy()
            ip_weights = (match_weights(ip_match_ids)
                          if ROW_WEIGHTING == "per_match" else None)
            summary["data_stats"]["inplay_row_weighting"] = ROW_WEIGHTING
            logger.info("[WEIGHTS] in-play: %d rows across %d fixtures, weighting=%s%s",
                        len(df_ip), len(np.unique(ip_match_ids)), ROW_WEIGHTING,
                        " — the fit sees %d observations, not %d" % (
                            len(np.unique(ip_match_ids)), len(df_ip))
                        if ip_weights is not None else "")

            X = df_ip[FEATURES].to_numpy(dtype=float)
            summary["feature_counts"]["inplay"] = len(FEATURES)
            summary["feature_selection"] = {}
            # Measured on the real data, not argued from the code.
            summary["collinearity"] = {
                "full": collinearity_report(df_ip, FEATURES),
                "core": collinearity_report(df_ip, CORE_FEATURES),
            }

            def _anchored_head(head: str, y: np.ndarray, threshold_label, default_thr,
                               metrics_name: str):
                """
                Fit one head, anchored to its own market where it has one.

                Every head here is fitted on the same rows; the only per-head
                difference is WHICH market it is anchored to, and whether it has
                one at all. OU_3.5 does not - only the 2.5 line is quoted in the
                features - so it is fitted the ordinary way rather than being
                anchored to a price for a different question.

                An anchored head drops its own anchor from the feature matrix.
                Leaving it in would let the fit put a second, penalised weight
                on the same signal and partially unpin the offset, which is the
                behaviour being removed.
                """
                feat = MARKET_ANCHOR.get(head) if anchor_rep["anchored"] else None
                off = (np.array([anchor_logit(v)
                                 for v in df_ip[feat].to_numpy(dtype=float)])
                       if feat and feat in df_ip.columns else None)

                # An anchored head drops its own anchor from the feature
                # matrix. Leaving it in would let the fit put a second,
                # penalised weight on the same signal and partially unpin the
                # offset, which is the behaviour being removed.
                def _without_anchor(cols: List[str]) -> List[str]:
                    return [c for c in cols if c != feat] if feat else list(cols)

                candidates = {"full": _without_anchor(FEATURES),
                              "core": _without_anchor(CORE_FEATURES)}
                if FEATURE_SET in candidates:
                    chosen, cols = FEATURE_SET, candidates[FEATURE_SET]
                    fs_diag = {"chosen": chosen, "forced_by": "FEATURE_SET"}
                else:
                    chosen, cols, fs_diag = select_feature_set(
                        df_ip, y, m_tr, m_ca, candidates, offset=off, weights=ip_weights)
                summary["feature_selection"][head] = fs_diag
                logger.info("[FEATURES] %s: %s set (%d columns)%s", head, chosen, len(cols),
                            f", cal logloss {fs_diag.get('cal_logloss')}"
                            if fs_diag.get("cal_logloss") else "")

                Xh = df_ip[cols].to_numpy(dtype=float)
                if off is not None:
                    res = _train_binary_head(
                        buf, Xh, y, m_tr, m_ca, m_te, cols,
                        head, threshold_label, summary, target_precision, min_preds,
                        min_thresh, max_thresh, default_thr, metrics_name,
                        offset_all=off, anchor_name=feat,
                        weight_all=ip_weights, match_ids=ip_match_ids)
                    if res[0]:
                        return res
                    # Never silent: a fallback looks exactly like an anchored
                    # model that learned nothing, and the two need different fixes.
                    logger.warning("[ANCHOR] %s: anchored fit failed, falling back to "
                                   "the unanchored model", head)
                    summary.setdefault("anchor_fallbacks", []).append(head)
                return _train_binary_head(
                    buf, Xh, y, m_tr, m_ca, m_te, cols,
                    head, threshold_label, summary, target_precision, min_preds,
                    min_thresh, max_thresh, default_thr, metrics_name,
                    weight_all=ip_weights, match_ids=ip_match_ids)

            ok, mets, _, _ = _anchored_head(
                "BTTS_YES", df_ip["label_btts"].to_numpy(dtype=int), "BTTS", 0.65, "BTTS_YES")
            summary["trained"]["BTTS_YES"] = ok
            if ok:
                summary["metrics"]["BTTS_YES"] = mets
                _dd = decided_diagnostics(df_ip, "BTTS_YES",
                                          df_ip["label_btts"].to_numpy(dtype=int))
                if _dd:
                    mets["already_decided"] = _dd
                    logger.info("[DECIDED] BTTS_YES: %.1f%% of rows already settled when "
                                "harvested; base rate %.3f overall vs %s undecided",
                                _dd["decided_share_pct"], _dd["base_rate_all"],
                                _dd["base_rate_undecided"])

            totals = df_ip["final_goals_sum"].to_numpy(dtype=int)
            for line in ou_lines:
                name = f"OU_{_fmt_line(line)}"
                ok, mets, _, _ = _anchored_head(
                    name, (totals > line).astype(int),
                    f"Over/Under {_fmt_line(line)}", 0.65, name)
                summary["trained"][name] = ok
                if ok:
                    summary["metrics"][name] = mets
                    _dd = decided_diagnostics(df_ip, name, (totals > line).astype(int))
                    if _dd:
                        mets["already_decided"] = _dd
                        logger.info("[DECIDED] %s: %.1f%% of rows already settled when "
                                    "harvested; base rate %.3f overall vs %s undecided",
                                    name, _dd["decided_share_pct"], _dd["base_rate_all"],
                                    _dd["base_rate_undecided"])
                    if abs(line - 2.5) < 1e-6:
                        blob = buf.get_json(f"model_latest:{name}")
                        if blob is not None:
                            for k in ("model_latest:O25", "model:O25"):
                                buf.set(k, json.dumps(blob))

            gd = df_ip["final_goals_diff"].to_numpy(dtype=int)
            heads = {}
            for key, y in (("WLD_HOME", (gd > 0)), ("WLD_DRAW", (gd == 0)), ("WLD_AWAY", (gd < 0))):
                ok, mets, p_ca, p_te = _anchored_head(key, y.astype(int), None, 0.45, key)
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
