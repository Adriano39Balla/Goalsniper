# file: train_models.py
# Super merged trainer/tuner for goalsniper
# - Learns in-play & prematch logistic models (with calibration)
# - Persists raw-space weights to DB (so main.py can score without sklearn)
# - ROI-aware auto-threshold tuning (CONF_MIN/EV_MIN + MOTD variants)
# - Safe, standalone DB helpers (no import from main.py)

from __future__ import annotations

import os
import re
import json
import time
import math
import logging
from typing import Any, Dict, List, Optional, Tuple, Iterable

import numpy as np
import pandas as pd

import psycopg2
from psycopg2.extras import DictCursor

# Optional heavy deps (sklearn); degrade gracefully if missing
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import (
        brier_score_loss, accuracy_score, log_loss, precision_score,
        roc_auc_score, precision_recall_curve
    )
    _SK_OK = True
except Exception as _e:
    _SK_OK = False

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger("trainer")

# ------------------------------------------------------------------------------
# Env / constants
# ------------------------------------------------------------------------------
DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise SystemExit("DATABASE_URL is required for train_models.py")

def _should_force_ssl(url: str) -> bool:
    if not url.startswith(("postgres://", "postgresql://")):
        return False
    v = os.getenv("DB_SSLMODE_REQUIRE", "1").strip().lower()
    return v not in {"0", "false", "no", ""}

if _should_force_ssl(DB_URL) and "sslmode=" not in DB_URL:
    DB_URL = DB_URL + (("&" if "?" in DB_URL else "?") + "sslmode=require")

STMT_TIMEOUT_MS = int(os.getenv("PG_STATEMENT_TIMEOUT_MS", "15000"))
LOCK_TIMEOUT_MS = int(os.getenv("PG_LOCK_TIMEOUT_MS", "2000"))
IDLE_TX_TIMEOUT_MS = int(os.getenv("PG_IDLE_TX_TIMEOUT_MS", "30000"))
FORCE_UTC = os.getenv("PG_FORCE_UTC", "1").strip().lower() not in {"0","false","no",""}

# Training windows / knobs
RECENCY_HALF_LIFE_DAYS = float(os.getenv("RECENCY_HALF_LIFE_DAYS", "120"))  # 0 disables
RECENCY_MONTHS         = int(os.getenv("RECENCY_MONTHS", "36"))             # cap in-play snapshot window
OU_TRAIN_LINES_RAW     = os.getenv("OU_TRAIN_LINES", os.getenv("OU_LINES", "2.5,3.5"))

TRAIN_MIN_MINUTE       = int(os.getenv("TRAIN_MIN_MINUTE", "15"))
TRAIN_TEST_SIZE        = float(os.getenv("TRAIN_TEST_SIZE", "0.25"))
MIN_ROWS               = int(os.getenv("MIN_ROWS", "150"))

TARGET_PRECISION       = float(os.getenv("TARGET_PRECISION","0.60"))
THRESH_MIN_PREDICTIONS = int(os.getenv("THRESH_MIN_PREDICTIONS","25"))
MIN_THRESH             = float(os.getenv("MIN_THRESH","55"))
MAX_THRESH             = float(os.getenv("MAX_THRESH","85"))

# Global threshold tuning (ROI-aware; used by main.py live gates)
TRAIN_WINDOW_DAYS      = int(os.getenv("TRAIN_WINDOW_DAYS", "28"))
TUNE_WINDOW_DAYS       = int(os.getenv("TUNE_WINDOW_DAYS", "14"))
MIN_BETS_FOR_TRAIN     = int(os.getenv("MIN_BETS_FOR_TRAIN", "150"))
MIN_BETS_FOR_TUNE      = int(os.getenv("MIN_BETS_FOR_TUNE", "80"))
CALIBRATION_BINS       = int(os.getenv("CALIBRATION_BINS", "10"))
EV_CAP                 = float(os.getenv("EV_CAP", "0.50"))

# Settings keys main.py expects to load at boot
CONF_KEY       = "CONF_MIN"
EV_KEY         = "EV_MIN"
MOTD_CONF_KEY  = "MOTD_CONF_MIN"
MOTD_EV_KEY    = "MOTD_EV_MIN"
CAL_KEY        = "calibration_overall"  # for ROI gate mapping of confidence_raw->p

# Feature sets (aligned with old main.py live extraction & prematch snapshot)
FEATURES: List[str] = [
    "minute",
    "goals_h","goals_a","goals_sum","goals_diff",
    "xg_h","xg_a","xg_sum","xg_diff",
    "sot_h","sot_a","sot_sum",
    "sh_total_h","sh_total_a",
    "cor_h","cor_a","cor_sum",
    "pos_h","pos_a","pos_diff",
    "red_h","red_a","red_sum",
    "yellow_h","yellow_a",
]

PRE_FEATURES: List[str] = [
    "pm_ov25_h","pm_ov35_h","pm_btts_h",
    "pm_ov25_a","pm_ov35_a","pm_btts_a",
    "pm_ov25_h2h","pm_ov35_h2h","pm_btts_h2h",
    # placeholders (keep model compatibility with live heads)
    "minute","goals_h","goals_a","goals_sum","goals_diff",
    "xg_h","xg_a","xg_sum","xg_diff","sot_h","sot_a","sot_sum",
    "sh_total_h","sh_total_a","cor_h","cor_a","cor_sum",
    "pos_h","pos_a","pos_diff","red_h","red_a","red_sum",
    "yellow_h","yellow_a",
]

EPS = 1e-6

# ------------------------------------------------------------------------------
# Minimal standalone DB helpers (no import from main.py)
# ------------------------------------------------------------------------------
class _DB:
    def __enter__(self):
        self.conn = psycopg2.connect(DB_URL)
        self.conn.autocommit = True
        with self.conn.cursor() as cur:
            if FORCE_UTC:
                cur.execute("SET TIME ZONE 'UTC'")
            cur.execute("SET statement_timeout = %s", (STMT_TIMEOUT_MS,))
            cur.execute("SET lock_timeout = %s", (LOCK_TIMEOUT_MS,))
            cur.execute("SET idle_in_transaction_session_timeout = %s", (IDLE_TX_TIMEOUT_MS,))
        self.cur = self.conn.cursor(cursor_factory=DictCursor)
        return self.cur
    def __exit__(self, exc_type, exc, tb):
        try:
            self.cur.close()
        finally:
            self.conn.close()

def db_conn():
    return _DB()

def set_setting(key: str, value: str) -> None:
    with db_conn() as c:
        c.execute(
            "INSERT INTO settings(key,value) VALUES(%s,%s) "
            "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
            (key, value),
        )

def get_setting(key: str) -> Optional[str]:
    with db_conn() as c:
        c.execute("SELECT value FROM settings WHERE key=%s", (key,))
        row = c.fetchone()
        return (row[0] if row else None)

def get_setting_json(key: str) -> Optional[dict]:
    try:
        raw = get_setting(key)
        return json.loads(raw) if raw else None
    except Exception as e:
        logger.warning("[TRAIN] get_setting_json failed for key=%s: %s", key, e)
        return None

def _exec(conn, sql: str, params: Tuple = ()) -> None:
    with conn.cursor() as cur:
        cur.execute(sql, params)

def _read_sql(conn, sql: str, params: Tuple = ()) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn, params=params)

def _connect(db_url: Optional[str] = None):
    url = db_url or DB_URL
    if not url:
        raise SystemExit("DATABASE_URL must be set.")
    if _should_force_ssl(url) and "sslmode=" not in url:
        url = url + (("&" if "?" in url else "?") + "sslmode=require")
    conn = psycopg2.connect(url); conn.autocommit = True
    with conn.cursor() as cur:
        if FORCE_UTC:
            cur.execute("SET TIME ZONE 'UTC'")
        cur.execute("SET statement_timeout = %s", (STMT_TIMEOUT_MS,))
        cur.execute("SET lock_timeout = %s", (LOCK_TIMEOUT_MS,))
        cur.execute("SET idle_in_transaction_session_timeout = %s", (IDLE_TX_TIMEOUT_MS,))
    return conn

def _ensure_training_tables(conn) -> None:
    _exec(conn, "CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)")
    _exec(conn, """
        CREATE TABLE IF NOT EXISTS prematch_snapshots (
            match_id   BIGINT PRIMARY KEY,
            created_ts BIGINT,
            payload    TEXT
        )
    """)
    _exec(conn, "CREATE INDEX IF NOT EXISTS idx_pre_snap_ts ON prematch_snapshots (created_ts DESC)")

# ------------------------------------------------------------------------------
# Small utils
# ------------------------------------------------------------------------------
def frange(start: float, stop: float, step: float) -> Iterable[float]:
    v = start
    while v <= stop + 1e-9:
        yield v
        v += step

def _percent(x: float) -> float:
    return float(x) * 100.0

def _fmt_line(line: float) -> str:
    return f"{line}".rstrip("0").rstrip(".")

def _logit_vec(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(float), 1e-6, 1-1e-6)
    return np.log(p/(1.0-p))


# ------------------------------------------------------------------------------
# Data loaders
# ------------------------------------------------------------------------------

def _ensure_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
    df[cols] = df[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df[cols]

def load_inplay_data(conn, min_minute: int = TRAIN_MIN_MINUTE) -> pd.DataFrame:
    q = """
    WITH latest AS (
      SELECT match_id, MAX(created_ts) AS ts
      FROM tip_snapshots
      GROUP BY match_id
    )
    SELECT l.match_id, s.created_ts, s.payload,
           r.final_goals_h, r.final_goals_a, r.btts_yes
    FROM latest l
    JOIN tip_snapshots s ON s.match_id = l.match_id AND s.created_ts = l.ts
    JOIN match_results r ON r.match_id = l.match_id
    """
    rows = _read_sql(conn, q)
    if rows.empty:
        return pd.DataFrame()

    feats: List[Dict[str,Any]] = []
    for _, row in rows.iterrows():
        try:
            payload = json.loads(row["payload"]) if isinstance(row["payload"], str) else (row["payload"] or {})
            stat = payload.get("stat") or {}
            gh = float(payload.get("gh", 0)); ga = float(payload.get("ga", 0))
            f = {
                "_mid": int(row["match_id"]), "_ts": float(row["created_ts"]),
                "minute": float(payload.get("minute", 0)),
                "goals_h": gh, "goals_a": ga, "goals_sum": gh+ga, "goals_diff": gh-ga,
                "xg_h": float(stat.get("xg_h",0)), "xg_a": float(stat.get("xg_a",0)),
                "xg_sum": float(stat.get("xg_h",0)+stat.get("xg_a",0)),
                "xg_diff": float(stat.get("xg_h",0)-stat.get("xg_a",0)),
                "sot_h": float(stat.get("sot_h",0)), "sot_a": float(stat.get("sot_a",0)),
                "sot_sum": float(stat.get("sot_h",0)+stat.get("sot_a",0)),
                "cor_h": float(stat.get("cor_h",0)), "cor_a": float(stat.get("cor_a",0)),
                "cor_sum": float(stat.get("cor_h",0)+stat.get("cor_a",0)),
                "pos_h": float(stat.get("pos_h",0)), "pos_a": float(stat.get("pos_a",0)),
                "pos_diff": float(stat.get("pos_h",0)-stat.get("pos_a",0)),
                "red_h": float(stat.get("red_h",0)), "red_a": float(stat.get("red_a",0)),
                "red_sum": float(stat.get("red_h",0)+stat.get("red_a",0)),
                "sh_total_h": float(stat.get("sh_total_h",0)), "sh_total_a": float(stat.get("sh_total_a",0)),
                "yellow_h": float(stat.get("yellow_h",0)), "yellow_a": float(stat.get("yellow_a",0)),
                "final_goals_h": int(row["final_goals_h"] or 0),
                "final_goals_a": int(row["final_goals_a"] or 0),
                "btts_yes": int(row["btts_yes"] or 0),
                "final_goals_sum": int(row["final_goals_h"] or 0) + int(row["final_goals_a"] or 0),
                "final_goals_diff": int(row["final_goals_h"] or 0) - int(row["final_goals_a"] or 0),
            }
            feats.append(f)
        except Exception:
            continue
    if not feats:
        return pd.DataFrame()

    df = pd.DataFrame(feats)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["minute"] = df["minute"].clip(0, 120)
    df = df[df["minute"] >= float(min_minute)].copy()

    if RECENCY_MONTHS > 0:
        cutoff = pd.Timestamp.utcnow().timestamp() - RECENCY_MONTHS*30*24*3600
        df = df[df["_ts"] >= cutoff].copy()
    return df

def load_prematch_data(conn) -> pd.DataFrame:
    q = """
    SELECT p.match_id, p.created_ts, p.payload,
           r.final_goals_h, r.final_goals_a, r.btts_yes
    FROM prematch_snapshots p
    JOIN match_results r ON r.match_id = p.match_id
    """
    rows = _read_sql(conn, q)
    if rows.empty:
        return pd.DataFrame()

    feats: List[Dict[str,Any]] = []
    for _, row in rows.iterrows():
        try:
            payload = json.loads(row["payload"]) or {}
            feat = (payload.get("feat") or {})
            f = {k: float(feat.get(k, 0.0) or 0.0) for k in PRE_FEATURES}
            gh_f = int(row["final_goals_h"] or 0); ga_f = int(row["final_goals_a"] or 0)
            f["_ts"] = int(row["created_ts"] or 0)
            f["final_goals_sum"]  = gh_f + ga_f
            f["final_goals_diff"] = gh_f - ga_f
            f["label_btts"] = 1 if int(row["btts_yes"] or 0) == 1 else 0
            feats.append(f)
        except Exception:
            continue
    if not feats:
        return pd.DataFrame()
    return pd.DataFrame(feats).replace([np.inf, -np.inf], np.nan).fillna(0.0)

# ------------------------------------------------------------------------------
# Recency weights
# ------------------------------------------------------------------------------
def recency_weights(ts: np.ndarray) -> Optional[np.ndarray]:
    if RECENCY_HALF_LIFE_DAYS <= 0:
        return None
    now = float(pd.Timestamp.utcnow().timestamp())
    dt_days = (now - ts.astype(float)) / (24*3600.0)
    lam = np.log(2) / max(1e-9, RECENCY_HALF_LIFE_DAYS)
    w = np.exp(-lam * np.maximum(0.0, dt_days))
    s = float(np.sum(w))
    return (w/s) if s > 0 else None

# ------------------------------------------------------------------------------
# Modeling helpers (logistic + calibration) — robust & portable
# ------------------------------------------------------------------------------

from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    brier_score_loss, accuracy_score, log_loss, precision_score,
    roc_auc_score, precision_recall_curve
)

def fit_lr_safe(X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None
               ) -> Optional[LogisticRegression]:
    """Stable LR fit with sane defaults; returns None if only one class present."""
    if len(np.unique(y)) < 2:
        return None
    C = float(os.getenv("LR_C", "1.0"))
    return LogisticRegression(
        max_iter=5000,
        solver="saga",
        penalty="l2",
        class_weight="balanced",
        n_jobs=-1,
        C=C,
        random_state=42,
    ).fit(X, y, sample_weight=sample_weight)

def _logit_vec(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(float), 1e-6, 1-1e-6)
    return np.log(p/(1.0-p))

def _fit_calibration(y_true: np.ndarray, p_raw: np.ndarray) -> Tuple[str, object]:
    """
    Choose Platt (on logits) vs Isotonic by validation Brier.
    Returns ('platt', (a,b)) or ('isotonic', IsotonicRegression()).
    """
    y = y_true.astype(int)
    z = _logit_vec(p_raw).reshape(-1, 1)

    lr = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42).fit(z, y)
    a = float(lr.coef_.ravel()[0]); b = float(lr.intercept_.ravel()[0])
    p_platt = 1.0/(1.0+np.exp(-(a*z.ravel()+b)))
    brier_platt = brier_score_loss(y, p_platt)

    best_kind, best_obj, best_brier = "platt", (a, b), brier_platt

    if len(y) >= 300 and 0 < y.mean() < 1:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_raw, y)
        p_iso = iso.predict(p_raw)
        brier_iso = brier_score_loss(y, p_iso)
        if brier_iso + 1e-6 < best_brier:
            best_kind, best_obj, best_brier = "isotonic", iso, brier_iso

    return best_kind, best_obj

def _apply_calibration(p_raw: np.ndarray, cal_kind: str, cal_obj) -> np.ndarray:
    if cal_kind == "platt":
        a, b = cal_obj
        z = _logit_vec(p_raw)
        return 1.0/(1.0+np.exp(-(a*z + b)))
    # isotonic
    return np.asarray(cal_obj.predict(p_raw), dtype=float)

def _weights_dict(model: LogisticRegression, feature_names: List[str]) -> Dict[str, float]:
    return {name: float(w) for name, w in zip(feature_names, model.coef_.ravel().tolist())}

def _standardize_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (X_std, mean, scale). Zero-variance columns left unscaled.
    (We’ll fold mean/scale back into raw-space weights for serving.)
    """
    mean = X.mean(axis=0)
    scale = X.std(axis=0, ddof=0)
    scale = np.where(scale <= 1e-12, 1.0, scale)
    Xs = (X - mean) / scale
    return Xs, mean, scale

def _fold_std_into_weights(model: LogisticRegression, mean: np.ndarray, scale: np.ndarray,
                           features: List[str]) -> Tuple[float, Dict[str, float]]:
    """
    Convert weights learned on standardized features back to raw space:
      w_raw_j = w_std_j / scale_j
      b_raw   = b_std - sum_j w_std_j * mean_j / scale_j
    """
    w_std = model.coef_.ravel().astype(float)
    b_std = float(model.intercept_.ravel()[0])
    w_raw = (w_std / scale).astype(float)
    b_raw = float(b_std - np.sum(w_std * (mean / scale)))
    weights = {name: float(w) for name, w in zip(features, w_raw.tolist())}
    return b_raw, weights

def build_model_blob(model: LogisticRegression, features: List[str],
                     cal_kind: str, cal_obj,
                     mean: Optional[np.ndarray] = None,
                     scale: Optional[np.ndarray] = None) -> Dict[str, object]:
    """
    Serialize model for serving in main.py (no sklearn dependency at runtime).
    If mean/scale provided, fold them into raw-space weights.
    Cal: store Platt directly; Isotonic ≈ Platt via logit→logit linear fit to keep serving simple.
    """
    if mean is not None and scale is not None:
        intercept, weights = _fold_std_into_weights(model, mean, scale, features)
    else:
        intercept = float(model.intercept_.ravel()[0])
        weights = _weights_dict(model, features)

    blob: Dict[str, object] = {
        "intercept": float(intercept),
        "weights": weights,
        "calibration": {"method": "sigmoid", "a": 1.0, "b": 0.0},
    }

    if cal_kind == "platt":
        a, b = cal_obj
        blob["calibration"] = {"method": "platt", "a": float(a), "b": float(b)}
    else:
        # approximate isotonic with a platt-like linear fit in logit space
        x = np.concatenate([
            np.linspace(0.01, 0.10, 30),
            np.linspace(0.10, 0.90, 120),
            np.linspace(0.90, 0.99, 30),
        ])
        y = np.asarray(cal_obj.predict(x), dtype=float)
        zx = _logit_vec(x); zy = _logit_vec(np.clip(y, 1e-4, 1-1e-4))
        a, b = np.polyfit(zx, zy, 1)
        blob["calibration"] = {"method": "platt", "a": float(a), "b": float(b)}

    return blob

# ------------------------------------------------------------------------------
# Splits, filters, and diagnostics
# ------------------------------------------------------------------------------

def time_order_split(df: pd.DataFrame, test_size: float
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Time-ordered split if _ts exists; else random (seeded).
    Returns boolean masks (train_mask, test_mask) for df index order.
    """
    if "_ts" not in df.columns:
        n = len(df); idx = np.arange(n)
        rng = np.random.default_rng(42); rng.shuffle(idx)
        cut = int((1 - test_size) * n)
        tr = np.zeros(n, dtype=bool); te = np.zeros(n, dtype=bool)
        tr[idx[:cut]] = True; te[idx[cut:]] = True
        return tr, te

    df_sorted = df.sort_values("_ts").reset_index(drop=True)
    n = len(df_sorted); cut = int(max(1, (1 - test_size) * n))
    train_idx = df_sorted.index[:cut].to_numpy()
    test_idx  = df_sorted.index[cut:].to_numpy()
    tr = np.zeros(n, dtype=bool); te = np.zeros(n, dtype=bool)
    tr[train_idx] = True; te[test_idx] = True
    return tr, te

def _minute_cutoff_for_market(market: str) -> Optional[int]:
    """
    Market-specific latest snapshot minute to train on (None = no cutoff).
    Falls back to TIP_MAX_MINUTE if set, and allows per-market overrides via MARKET_CUTOFFS.
    """
    m = (market or "").upper()
    if m.startswith("PRE "):
        return None
    if m.startswith("OVER/UNDER"):
        return MARKET_CUTOFFS.get("OU", TIP_MAX_MINUTE)
    if m == "BTTS":
        return MARKET_CUTOFFS.get("BTTS", TIP_MAX_MINUTE)
    if m == "1X2":
        return MARKET_CUTOFFS.get("1X2", TIP_MAX_MINUTE)
    return TIP_MAX_MINUTE

def _ece(y_true: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    """Expected Calibration Error."""
    y = y_true.astype(int); p = np.clip(p.astype(float), 0, 1)
    edges = np.linspace(0.0, 1.0, bins + 1); ece = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i+1]
        mask = (p >= lo) & (p < hi) if i < bins - 1 else (p >= lo) & (p <= hi)
        if not np.any(mask):
            continue
        conf = float(np.mean(p[mask])); acc = float(np.mean(y[mask]))
        ece += (np.sum(mask)/len(p)) * abs(conf - acc)
    return float(ece)

def _mask_undecided(market_key: str, X_all: np.ndarray, y_all: np.ndarray,
                    features: List[str]) -> np.ndarray:
    """
    Keep rows where the outcome isn't trivially decided at snapshot.
      - OU_L: drop rows where goals_sum > line (Over guaranteed)
      - BTTS: drop rows where both teams have already scored
      - 1X2: keep all (never fully locked until FT)
    """
    fidx = {f: i for i, f in enumerate(features)}
    keep = np.ones(len(y_all), dtype=bool)

    def col(name): return X_all[:, fidx[name]] if name in fidx else np.zeros(len(X_all))

    goals_sum = col("goals_sum")
    gh = col("goals_h"); ga = col("goals_a")

    if market_key.startswith("OU_"):
        try:
            ln = float(market_key.split("_", 1)[1].replace(",", "."))
        except Exception:
            ln = 2.5
        keep &= ~(goals_sum > ln)
    elif market_key == "BTTS_YES":
        keep &= ~((gh > 0) & (ga > 0))
    # 1X2_* : no filter

    return keep


# ------------------------------------------------------------------------------
# Single-head trainer and full orchestration
# ------------------------------------------------------------------------------

def _percent(x: float) -> float:
    return float(x) * 100.0

def _pick_threshold_for_target_precision(
    y_true: np.ndarray,
    p_cal: np.ndarray,
    target_precision: float,
    min_preds: int = 25,
    default_threshold: float = 0.65,
) -> float:
    """Choose probability threshold to hit target precision, with sensible fallbacks."""
    y = y_true.astype(int); p = np.asarray(p_cal, dtype=float)
    best_t: Optional[float] = None
    candidates = np.arange(0.50, 0.951, 0.005)

    feasible: List[Tuple[float, float, int]] = []
    for t in candidates:
        pred = (p >= t).astype(int); n_pred = int(pred.sum())
        if n_pred < min_preds:
            continue
        prec = precision_score(y, pred, zero_division=0)
        if prec >= target_precision:
            feasible.append((float(t), float(prec), n_pred))

    if feasible:
        feasible.sort(key=lambda z: (z[0], -z[1]))
        best_t = feasible[0][0]
    else:
        prec, rec, thr = precision_recall_curve(y, p)
        if len(thr) > 0:
            f1 = 2 * (prec * rec) / (prec + rec + 1e-9)
            idx = int(np.argmax(f1[:-1])) if len(f1) > 1 else 0
            best_t = float(thr[max(0, min(idx, len(thr) - 1))])

    if best_t is None:
        best_acc = -1.0
        for t in candidates:
            acc = accuracy_score(y, (p >= t).astype(int))
            if acc > best_acc:
                best_t, best_acc = float(t), acc

    return float(best_t if best_t is not None else default_threshold)

def _fmt_line(line: float) -> str:
    return f"{line}".rstrip("0").rstrip(".")

def _train_binary_head(
    X_all: np.ndarray,
    y_all: np.ndarray,
    ts_all: np.ndarray,
    mask_tr: np.ndarray,
    mask_te: np.ndarray,
    feature_names: List[str],
    model_key: str,
    threshold_label: Optional[str],
    target_precision: float,
    min_preds: int,
    min_thresh_pct: float,
    max_thresh_pct: float,
    default_thr_prob: float,
    metrics_name: Optional[str] = None,
    train_minute_cutoff: Optional[int] = None,
) -> Tuple[bool, Dict[str, object], Optional[np.ndarray]]:
    """
    Train one binary head (e.g., BTTS_YES, OU_2.5, WLD_HOME).
    Standardizes X on train, folds scaler into raw-space weights for serving.
    Applies undecided-snapshot filter on train fold for relevant markets.
    """
    if len(np.unique(y_all)) < 2:
        return False, {}, None

    # Optional minute cutoff for train fold (column 0 is 'minute' by convention)
    if train_minute_cutoff is not None:
        try:
            mt = (X_all[:, 0] <= float(train_minute_cutoff))
            mask_tr = mask_tr & mt
        except Exception:
            pass

    # Filter trivially decided snapshots for this head (train only)
    try:
        undecided = _mask_undecided(model_key, X_all, y_all, feature_names)
        new_mask_tr = mask_tr & undecided
        if np.any(new_mask_tr):
            mask_tr = new_mask_tr
        else:
            log.info("[TRAIN] %s: undecided filter removed all train rows; skipping the filter.", model_key)
    except Exception:
        pass

    X_tr, X_te = X_all[mask_tr], X_all[mask_te]
    y_tr, y_te = y_all[mask_tr], y_all[mask_te]
    ts_tr      = ts_all[mask_tr]

    # Standardize on train fold
    X_tr_std, mu, sd = _standardize_fit(X_tr)
    X_te_std = (X_te - mu) / sd

    # Recency weights
    sw = recency_weights(ts_tr)
    model = fit_lr_safe(X_tr_std, y_tr, sample_weight=sw)
    if model is None:
        return False, {}, None

    # Validate and calibrate on validation fold
    p_raw_te = model.predict_proba(X_te_std)[:, 1]
    cal_kind, cal_obj = _fit_calibration(y_te, p_raw_te)
    p_cal = _apply_calibration(p_raw_te, cal_kind, cal_obj)

    # Metrics
    mets: Dict[str, object] = {
        "brier": float(brier_score_loss(y_te, p_cal)),
        "auc": float(roc_auc_score(y_te, p_cal)) if len(np.unique(y_te)) > 1 else float("nan"),
        "ece": float(_ece(y_te, p_cal)),
        "acc@0.5": float(accuracy_score(y_te, (p_cal >= 0.5).astype(int))),
        "logloss": float(log_loss(y_te, p_cal, labels=[0, 1])),
        "n_test": int(len(y_te)),
        "prevalence": float(np.mean(y_all)),
        "calibration": cal_kind,
    }
    if metrics_name:
        log.info("[METRICS] %s: %s", metrics_name, mets)

    # Serialize (fold scaler into raw-space weights)
    blob = build_model_blob(model, feature_names, cal_kind, cal_obj, mean=mu, scale=sd)
    for k in (f"model_latest:{model_key}", f"model:{model_key}"):
        set_setting(k, json.dumps(blob))

    # Optional: derive/update per-market threshold from validation fold
    if threshold_label:
        thr_prob = _pick_threshold_for_target_precision(
            y_true=y_te,
            p_cal=p_cal,
            target_precision=target_precision,
            min_preds=min_preds,
            default_threshold=default_thr_prob,
        )
        thr_pct = float(np.clip(_percent(thr_prob), min_thresh_pct, max_thresh_pct))
        set_setting(f"conf_threshold:{threshold_label}", f"{thr_pct:.2f}")

    return True, mets, p_cal

def train_models(
    db_url: Optional[str] = None,
    min_minute: Optional[int] = None,
    test_size: Optional[float] = None,
    min_rows: Optional[int] = None,
) -> Dict[str, object]:
    """
    Orchestrates:
      - Load labeled in-play snapshots and prematch snapshots
      - Train BTTS / OU lines / WLD heads (prematch & in-play)
      - Calibrate, persist models to settings as JSON blobs consumable by main.py
      - Learn per-market probability thresholds targeting a precision level
      - Save a compact metrics bundle for observability
    """
    # args/env
    min_minute = int(min_minute if min_minute is not None else os.getenv("TRAIN_MIN_MINUTE", 15))
    test_size  = float(test_size  if test_size  is not None else os.getenv("TRAIN_TEST_SIZE", 0.25))
    min_rows   = int(min_rows   if min_rows   is not None else os.getenv("MIN_ROWS", 150))

    ou_lines = _parse_ou_lines(OU_TRAIN_LINES_RAW)

    target_precision = float(os.getenv("TARGET_PRECISION", "0.60"))
    min_preds        = int(os.getenv("THRESH_MIN_PREDICTIONS", "25"))
    min_thresh       = float(os.getenv("MIN_THRESH", "55"))
    max_thresh       = float(os.getenv("MAX_THRESH", "85"))

    summary: Dict[str, object] = {"ok": True, "trained": {}, "metrics": {}, "thresholds": {}}

    try:
        # ===== In-Play =====
        df_ip = load_inplay_data(min_minute=min_minute)
        if not df_ip.empty and len(df_ip) >= min_rows:
            tr_mask, te_mask = time_order_split(df_ip, test_size=test_size)
            df_ip = df_ip.reset_index(drop=True)
            X_all  = _ensure_columns(df_ip, FEATURES).values
            ts_all = df_ip["_ts"].values

            # BTTS
            ok, mets, _ = _train_binary_head(
                X_all, df_ip["btts_yes"].values.astype(int), ts_all,
                tr_mask, te_mask, FEATURES, "BTTS_YES", "BTTS",
                target_precision, min_preds, min_thresh, max_thresh, 0.65,
                "BTTS_YES", _minute_cutoff_for_market("BTTS")
            )
            summary["trained"]["BTTS_YES"] = ok
            if ok:
                summary["metrics"]["BTTS_YES"] = mets

            # OU lines (label: final_goals_sum > line)
            totals = df_ip["final_goals_sum"].values.astype(int)
            for line in ou_lines:
                name = f"OU_{_fmt_line(line)}"
                ok, mets, _ = _train_binary_head(
                    X_all, (totals > line).astype(int), ts_all,
                    tr_mask, te_mask, FEATURES, name, f"Over/Under {_fmt_line(line)}",
                    target_precision, min_preds, min_thresh, max_thresh, 0.65,
                    name, _minute_cutoff_for_market("Over/Under")
                )
                summary["trained"][name] = ok
                if ok:
                    summary["metrics"][name] = mets
                    # Back-compat alias for O25 if line==2.5
                    if abs(line - 2.5) < 1e-6:
                        blob = get_setting_json(f"model_latest:{name}")
                        if blob is not None:
                            for k in ("model_latest:O25", "model:O25"):
                                set_setting(k, json.dumps(blob))

            # 1X2 heads (3 one-vs-rest)
            gd = df_ip["final_goals_diff"].values.astype(int)
            y_home = (gd > 0).astype(int)
            y_draw = (gd == 0).astype(int)
            y_away = (gd < 0).astype(int)

            ok_h, mets_h, p_h = _train_binary_head(
                X_all, y_home, ts_all, tr_mask, te_mask, FEATURES,
                "WLD_HOME", None, target_precision, min_preds, min_thresh, max_thresh, 0.45,
                "WLD_HOME", _minute_cutoff_for_market("1X2")
            )
            ok_d, mets_d, p_d = _train_binary_head(
                X_all, y_draw, ts_all, tr_mask, te_mask, FEATURES,
                "WLD_DRAW", None, target_precision, min_preds, min_thresh, max_thresh, 0.45,
                "WLD_DRAW", _minute_cutoff_for_market("1X2")
            )
            ok_a, mets_a, p_a = _train_binary_head(
                X_all, y_away, ts_all, tr_mask, te_mask, FEATURES,
                "WLD_AWAY", None, target_precision, min_preds, min_thresh, max_thresh, 0.45,
                "WLD_AWAY", _minute_cutoff_for_market("1X2")
            )

            summary["trained"].update({"WLD_HOME": ok_h, "WLD_DRAW": ok_d, "WLD_AWAY": ok_a})
            if ok_h: summary["metrics"]["WLD_HOME"] = mets_h
            if ok_d: summary["metrics"]["WLD_DRAW"] = mets_d
            if ok_a: summary["metrics"]["WLD_AWAY"] = mets_a

            # Derive single 1X2 threshold from max-normalized head probs
            if ok_h and ok_d and ok_a and (p_h is not None) and (p_d is not None) and (p_a is not None):
                ps = np.clip(p_h, EPS, 1 - EPS) + np.clip(p_d, EPS, 1 - EPS) + np.clip(p_a, EPS, 1 - EPS)
                phn, pdn, pan = p_h / ps, p_d / ps, p_a / ps
                p_max = np.maximum.reduce([phn, pdn, pan])

                gd_te = gd[te_mask]
                y_class = np.zeros_like(gd_te, dtype=int)
                y_class[gd_te == 0] = 1; y_class[gd_te < 0] = 2
                correct = (np.argmax(np.stack([phn, pdn, pan], axis=1), axis=1) == y_class).astype(int)

                thr_prob = _pick_threshold_for_target_precision(
                    y_true=correct, p_cal=p_max,
                    target_precision=target_precision, min_preds=min_preds, default_threshold=0.45
                )
                thr_pct = float(np.clip(_percent(thr_prob), min_thresh, max_thresh))
                set_setting("conf_threshold:1X2", f"{thr_pct:.2f}")
                summary["thresholds"]["1X2"] = thr_pct
        else:
            log.info("In-Play: not enough labeled data (have %d, need >= %d).", len(df_ip), min_rows)
            summary["trained"]["BTTS_YES"] = False

        # ===== Prematch =====
        df_pre = load_prematch_data()
        if not df_pre.empty and len(df_pre) >= min_rows:
            tr_mask, te_mask = time_order_split(df_pre, test_size=test_size)
            df_pre = df_pre.reset_index(drop=True)
            Xp_all = _ensure_columns(df_pre, PRE_FEATURES).values
            ts_pre = df_pre["_ts"].values

            # PRE BTTS
            ok, mets, _ = _train_binary_head(
                Xp_all, df_pre["label_btts"].values.astype(int), ts_pre,
                tr_mask, te_mask, PRE_FEATURES, "PRE_BTTS_YES", "PRE BTTS",
                target_precision, min_preds, min_thresh, max_thresh, 0.65,
                "PRE_BTTS_YES", None
            )
            summary["trained"]["PRE_BTTS_YES"] = ok
            if ok:
                summary["metrics"]["PRE_BTTS_YES"] = mets

            # PRE OU
            totals = df_pre["final_goals_sum"].values.astype(int)
            for line in ou_lines:
                name = f"PRE_OU_{_fmt_line(line)}"
                ok, mets, _ = _train_binary_head(
                    Xp_all, (totals > line).astype(int), ts_pre,
                    tr_mask, te_mask, PRE_FEATURES, name, f"PRE Over/Under {_fmt_line(line)}",
                    target_precision, min_preds, min_thresh, max_thresh, 0.65,
                    name, None
                )
                summary["trained"][name] = ok
                if ok:
                    summary["metrics"][name] = mets

            # PRE 1X2 (draw suppressed)
            gd = df_pre["final_goals_diff"].values.astype(int)
            y_home = (gd > 0).astype(int); y_away = (gd < 0).astype(int)

            ok_h, mets_h, p_h = _train_binary_head(
                Xp_all, y_home, ts_pre, tr_mask, te_mask, PRE_FEATURES,
                "PRE_WLD_HOME", None, target_precision, min_preds, min_thresh, max_thresh, 0.45,
                "PRE_WLD_HOME", None
            )
            ok_a, mets_a, p_a = _train_binary_head(
                Xp_all, y_away, ts_pre, tr_mask, te_mask, PRE_FEATURES,
                "PRE_WLD_AWAY", None, target_precision, min_preds, min_thresh, max_thresh, 0.45,
                "PRE_WLD_AWAY", None
            )
            summary["trained"]["PRE_WLD_HOME"] = ok_h; summary["trained"]["PRE_WLD_AWAY"] = ok_a
            if ok_h: summary["metrics"]["PRE_WLD_HOME"] = mets_h
            if ok_a: summary["metrics"]["PRE_WLD_AWAY"] = mets_a

            if ok_h and ok_a and (p_h is not None) and (p_a is not None):
                ps = np.clip(p_h, EPS, 1 - EPS) + np.clip(p_a, EPS, 1 - EPS)
                phn, pan = p_h / ps, p_a / ps
                p_max = np.maximum(phn, pan)
                gd_te = gd[te_mask]
                y_class = np.where(gd_te > 0, 0, np.where(gd_te < 0, 1, -1))
                mask = (y_class != -1)
                if mask.any():
                    correct = (np.argmax(np.stack([phn, pan], axis=1), axis=1)[mask] == y_class[mask]).astype(int)
                    thr_prob = _pick_threshold_for_target_precision(
                        y_true=correct, p_cal=p_max[mask],
                        target_precision=target_precision, min_preds=min_preds, default_threshold=0.45
                    )
                    thr_pct = float(np.clip(_percent(thr_prob), min_thresh, max_thresh))
                    set_setting("conf_threshold:PRE 1X2", f"{thr_pct:.2f}")
                    summary["thresholds"]["PRE 1X2"] = thr_pct
        else:
            log.info("Prematch: not enough labeled data (have %d, need >= %d).", len(df_pre), min_rows)
            summary["trained"]["PRE_BTTS_YES"] = False

        # ===== Bundle metrics for observability =====
        metrics_bundle = {
            "trained_at_utc": pd.Timestamp.utcnow().isoformat(timespec="seconds") + "Z",
            **summary["metrics"],
            "features_inplay": FEATURES,
            "features_prematch": PRE_FEATURES,
            "thresholds": summary.get("thresholds", {}),
            "target_precision": target_precision,
            "ou_lines": [float(x) for x in ou_lines],
            "min_rows": int(min_rows),
            "test_size": float(test_size),
            "recency_half_life_days": float(RECENCY_HALF_LIFE_DAYS),
            "market_cutoffs": MARKET_CUTOFFS,
            "tip_max_minute": TIP_MAX_MINUTE,
        }
        set_setting("model_metrics_latest", json.dumps(metrics_bundle))
        return summary

    except Exception as e:
        log.exception("Training failed: %s", e)
        return {"ok": False, "error": str(e)}

# ------------------------------------------------------------------------------
# CLI entrypoint
# ------------------------------------------------------------------------------

import argparse

def _cli_main() -> None:
    ap = argparse.ArgumentParser(prog="train_models.py", description="goalsniper — training & tuning")
    ap.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"),
                    help="Python logging level (DEBUG, INFO, WARNING, ERROR)")

    sub = ap.add_subparsers(dest="cmd", required=False)

    # train
    ap_train = sub.add_parser("train", help="train models (in-play + prematch)")
    ap_train.add_argument("--min-minute", type=int, default=int(os.getenv("TRAIN_MIN_MINUTE", 15)),
                          help="minimum in-play snapshot minute to include")
    ap_train.add_argument("--test-size", type=float, default=float(os.getenv("TRAIN_TEST_SIZE", 0.25)),
                          help="holdout fraction for validation")
    ap_train.add_argument("--min-rows", type=int, default=int(os.getenv("MIN_ROWS", 150)),
                          help="minimum labeled rows required to run training")

    # tune
    ap_tune = sub.add_parser("tune", help="auto-tune CONF_MIN / EV_MIN via recent ROI grid-search")
    ap_tune.add_argument("--window-days", type=int, default=int(os.getenv("TUNE_WINDOW_DAYS", "14")),
                         help="lookback window for tuning on realized results")

    # show thresholds
    sub.add_parser("show-thresholds", help="print thresholds currently stored in settings")

    args = ap.parse_args()

    # logging
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="[%(asctime)s] %(levelname)s - %(message)s",
    )

    cmd = args.cmd or "train"

    if cmd == "train":
        res = train_models(
            min_minute=args.min_minute,
            test_size=args.test_size,
            min_rows=args.min_rows,
        )
        print(json.dumps(res, indent=2))
        return

    if cmd == "tune":
        res = auto_tune_thresholds(window_days=args.window_days)
        print(json.dumps(res, indent=2))
        return

    if cmd == "show-thresholds":
        vals = load_thresholds_from_settings()
        print(json.dumps(vals, indent=2))
        return

    # default fallback
    res = train_models(
        min_minute=int(os.getenv("TRAIN_MIN_MINUTE", 15)),
        test_size=float(os.getenv("TRAIN_TEST_SIZE", 0.25)),
        min_rows=int(os.getenv("MIN_ROWS", 150)),
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    _cli_main()
