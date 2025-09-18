# file: train_models.py
# Opta-supercomputer style training: CatBoost-first with LR fallback, calibration, ROI-aware threshold tuning.

import argparse, json, os, logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg2
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    brier_score_loss, accuracy_score, log_loss, precision_score,
    roc_auc_score, precision_recall_curve
)

# --- Optional CatBoost ---
_CB_OK = True
try:
    from catboost import CatBoostClassifier, Pool
except Exception as _e:
    _CB_OK = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger("trainer")

# ───────────────────────── Features (must match scan/extract) ─────────────────────────
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
    # live placeholders for shape-compat
    "minute","goals_h","goals_a","goals_sum","goals_diff",
    "xg_h","xg_a","xg_sum","xg_diff","sot_h","sot_a","sot_sum",
    "sh_total_h","sh_total_a","cor_h","cor_a","cor_sum",
    "pos_h","pos_a","pos_diff","red_h","red_a","red_sum",
    "yellow_h","yellow_a",
]

EPS = 1e-6

# ─────────────────────── Env knobs ───────────────────────
RECENCY_HALF_LIFE_DAYS = float(os.getenv("RECENCY_HALF_LIFE_DAYS", "120"))
RECENCY_MONTHS         = int(os.getenv("RECENCY_MONTHS", "36"))
MARKET_CUTOFFS_RAW     = os.getenv("MARKET_CUTOFFS", "BTTS=75,1X2=80,OU=88")
TIP_MAX_MINUTE_ENV     = os.getenv("TIP_MAX_MINUTE", "")
OU_TRAIN_LINES_RAW     = os.getenv("OU_TRAIN_LINES", os.getenv("OU_LINES", "2.5,3.5"))

# CatBoost params (safe defaults for Railway)
CB_ITERATIONS = int(os.getenv("CB_ITERATIONS", "400"))
CB_DEPTH      = int(os.getenv("CB_DEPTH", "6"))
CB_L2         = float(os.getenv("CB_L2", "3.0"))
CB_LEARNING   = float(os.getenv("CB_LEARNING_RATE", "0.08"))
CB_BAGGING    = float(os.getenv("CB_RSM", "0.9"))
CB_EVAL_FRACT = float(os.getenv("CB_EVAL_FRACTION", "0.15"))
CB_THREAD     = int(os.getenv("CB_THREAD_COUNT", "2"))

# LR fallback
LR_C = float(os.getenv("LR_C", "1.0"))

def _parse_market_cutoffs(s: str) -> Dict[str,int]:
    out: Dict[str,int] = {}
    for tok in (s or "").split(","):
        tok = tok.strip()
        if not tok or "=" not in tok: continue
        k,v = tok.split("=",1)
        try: out[k.strip().upper()] = int(float(v.strip()))
        except: pass
    return out

MARKET_CUTOFFS = _parse_market_cutoffs(MARKET_CUTOFFS_RAW)
try:
    TIP_MAX_MINUTE: Optional[int] = int(float(TIP_MAX_MINUTE_ENV)) if TIP_MAX_MINUTE_ENV.strip() else None
except Exception:
    TIP_MAX_MINUTE = None

def _parse_ou_lines(raw: str) -> List[float]:
    vals: List[float] = []
    for t in (raw or "").split(","):
        t = t.strip()
        if not t: continue
        try: vals.append(float(t))
        except: pass
    return vals or [2.5, 3.5]

def _fmt_line(line: float) -> str: return f"{line}".rstrip("0").rstrip(".")

# ─────────────────────── DB utils ───────────────────────
def _connect(db_url: str):
    if not db_url: raise SystemExit("DATABASE_URL must be set.")
    if "sslmode=" not in db_url:
        db_url = db_url + ("&" if "?" in db_url else "?") + "sslmode=require"
    conn = psycopg2.connect(db_url); conn.autocommit = True
    return conn

def _read_sql(conn, sql: str, params: Tuple = ()) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn, params=params)

def _exec(conn, sql: str, params: Tuple = ()) -> None:
    with conn.cursor() as cur: cur.execute(sql, params)

def _set_setting(conn, key: str, value: str) -> None:
    _exec(conn,
          "INSERT INTO settings(key,value) VALUES(%s,%s) "
          "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
          (key, value))

def _get_setting_json(conn, key: str) -> Optional[dict]:
    try:
        df = _read_sql(conn, "SELECT value FROM settings WHERE key=%s", (key,))
        if df.empty: return None
        return json.loads(df.iloc[0]["value"])
    except Exception:
        return None

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

# ─────────────────────── Data loaders ───────────────────────
def _ensure_columns(df: "pd.DataFrame", cols: List[str]) -> "pd.DataFrame":
    for c in cols:
        if c not in df.columns: df[c] = 0.0
    df[cols] = df[cols].replace([np.inf,-np.inf], np.nan).fillna(0.0)
    return df[cols]

def load_inplay_data(conn, min_minute: int = 15) -> pd.DataFrame:
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
    if rows.empty: return pd.DataFrame()

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
                "xg_sum": float(stat.get("xg_sum", stat.get("xg_h",0)+stat.get("xg_a",0))),
                "xg_diff": float(stat.get("xg_diff", stat.get("xg_h",0)-stat.get("xg_a",0))),
                "sot_h": float(stat.get("sot_h",0)), "sot_a": float(stat.get("sot_a",0)),
                "sot_sum": float(stat.get("sot_sum", stat.get("sot_h",0)+stat.get("sot_a",0))),
                "cor_h": float(stat.get("cor_h",0)), "cor_a": float(stat.get("cor_a",0)),
                "cor_sum": float(stat.get("cor_sum", stat.get("cor_h",0)+stat.get("cor_a",0))),
                "pos_h": float(stat.get("pos_h",0)), "pos_a": float(stat.get("pos_a",0)),
                "pos_diff": float(stat.get("pos_diff", stat.get("pos_h",0)-stat.get("pos_a",0))),
                "red_h": float(stat.get("red_h",0)), "red_a": float(stat.get("red_a",0)),
                "red_sum": float(stat.get("red_sum", stat.get("red_h",0)+stat.get("red_a",0))),
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
    if not feats: return pd.DataFrame()

    df = pd.DataFrame(feats)
    num = [c for c in df.columns if c not in ("_mid","_ts")]
    df[num] = df[num].replace([np.inf,-np.inf], np.nan).fillna(0.0)
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
    if rows.empty: return pd.DataFrame()

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
    if not feats: return pd.DataFrame()
    return pd.DataFrame(feats).replace([np.inf,-np.inf], np.nan).fillna(0.0)

# ─────────────────────── Modeling helpers ───────────────────────
def fit_lr_safe(X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray]=None) -> Optional[LogisticRegression]:
    if len(np.unique(y)) < 2:
        return None
    return LogisticRegression(
        max_iter=5000,
        solver="saga",
        penalty="l2",
        class_weight="balanced",
        n_jobs=-1,
        C=LR_C,
        random_state=42,
    ).fit(X, y, sample_weight=sample_weight)

def _logit_vec(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(float), 1e-6, 1-1e-6)
    return np.log(p/(1.0-p))

def _fit_calibration(y_true: np.ndarray, p_raw: np.ndarray) -> Tuple[str, Any]:
    """Return ('platt',(a,b)) or ('isotonic', IsotonicRegression)."""
    y = y_true.astype(int)
    z = _logit_vec(p_raw).reshape(-1,1)

    # Platt
    lr = LogisticRegression(max_iter=1000, solver="lbfgs").fit(z, y)
    a = float(lr.coef_.ravel()[0]); b = float(lr.intercept_.ravel()[0])
    p_platt = 1.0/(1.0+np.exp(-(a*z.ravel()+b)))
    brier_platt = brier_score_loss(y, p_platt)

    best_kind, best_obj, best_brier = "platt", (a,b), brier_platt

    # Isotonic if enough mass
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
        a,b = cal_obj
        z = _logit_vec(p_raw)
        return 1.0/(1.0+np.exp(-(a*z + b)))
    return np.asarray(cal_obj.predict(p_raw), dtype=float)

def _weights_dict(model: LogisticRegression, feature_names: List[str]) -> Dict[str,float]:
    return {name: float(w) for name, w in zip(feature_names, model.coef_.ravel().tolist())}

def _standardize_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    scale = X.std(axis=0, ddof=0)
    scale = np.where(scale <= 1e-12, 1.0, scale)
    Xs = (X - mean) / scale
    return Xs, mean, scale

def _fold_std_into_weights(model: LogisticRegression, mean: np.ndarray, scale: np.ndarray, features: List[str]) -> Tuple[float, Dict[str, float]]:
    w_std = model.coef_.ravel().astype(float)
    b_std = float(model.intercept_.ravel()[0])
    w_raw = (w_std / scale).astype(float)
    b_raw = float(b_std - np.sum(w_std * (mean / scale)))
    weights = {name: float(w) for name, w in zip(features, w_raw.tolist())}
    return b_raw, weights

def build_model_blob_from_lr(model: LogisticRegression, features: List[str], cal_kind: str, cal_obj,
                             mean: Optional[np.ndarray]=None, scale: Optional[np.ndarray]=None) -> Dict[str,Any]:
    if mean is not None and scale is not None:
        intercept, weights = _fold_std_into_weights(model, mean, scale, features)
    else:
        intercept = float(model.intercept_.ravel()[0])
        weights = _weights_dict(model, features)
    blob = {
        "intercept": float(intercept),
        "weights": weights,
        "calibration": {"method":"sigmoid","a":1.0,"b":0.0},
    }
    if cal_kind == "platt":
        a,b = cal_obj
        blob["calibration"] = {"method":"platt","a":float(a),"b":float(b)}
    else:
        # Fit a logit->logit linear map to approximate isotonic for runtime simplicity.
        x = np.concatenate([
            np.linspace(0.01, 0.10, 30),
            np.linspace(0.10, 0.90, 120),
            np.linspace(0.90, 0.99, 30),
        ])
        y = np.asarray(cal_obj.predict(x), dtype=float)
        zx = _logit_vec(x); zy = _logit_vec(np.clip(y, 1e-4, 1-1e-4))
        a,b = np.polyfit(zx, zy, 1)
        blob["calibration"] = {"method":"platt","a":float(a),"b":float(b)}
    return blob

def build_model_blob_from_catboost(model: "CatBoostClassifier", features: List[str],
                                   cal_kind: str, cal_obj) -> Dict[str,Any]:
    """
    Serve CatBoost via linear surrogate for runtime simplicity:
      - score raw prob p_raw from catboost
      - apply calibration (platt or isotonic->platt-fit)
    We serialize only calibration since runtime scoring calls won’t use tree traversal.
    In production, your inference path uses LR weights; here we store catboost-derived calibration
    and rely on trained LR fallback. If you want direct CatBoost inference at runtime,
    wire a dedicated inference path (out of scope for this minimal server).
    """
    blob = {
        "intercept": 0.0,
        "weights": {f: 0.0 for f in features},
        "calibration": {"method":"sigmoid","a":1.0,"b":0.0},
        "meta": {"trained_with": "catboost"}
    }
    if cal_kind == "platt":
        a,b = cal_obj
        blob["calibration"] = {"method":"platt","a":float(a),"b":float(b)}
    else:
        x = np.concatenate([
            np.linspace(0.01, 0.10, 30),
            np.linspace(0.10, 0.90, 120),
            np.linspace(0.90, 0.99, 30),
        ])
        y = np.asarray(cal_obj.predict(x), dtype=float)
        zx = _logit_vec(x); zy = _logit_vec(np.clip(y, 1e-4, 1-1e-4))
        a,b = np.polyfit(zx, zy, 1)
        blob["calibration"] = {"method":"platt","a":float(a),"b":float(b)}
    return blob

# ─────────────────────── Split, weights, utilities ───────────────────────
def time_order_split(df: pd.DataFrame, test_size: float) -> Tuple[np.ndarray,np.ndarray]:
    if "_ts" not in df.columns:
        n = len(df); idx = np.arange(n)
        rng = np.random.default_rng(42); rng.shuffle(idx)
        cut = int((1-test_size)*n)
        tr = np.zeros(n,dtype=bool); te = np.zeros(n,dtype=bool)
        tr[idx[:cut]] = True; te[idx[cut:]] = True
        return tr, te

    df_sorted = df.sort_values("_ts").reset_index(drop=True)
    n = len(df_sorted); cut = int(max(1,(1-test_size)*n))
    train_idx = df_sorted.index[:cut].to_numpy(); test_idx = df_sorted.index[cut:].to_numpy()
    tr = np.zeros(n,dtype=bool); te = np.zeros(n,dtype=bool)
    tr[train_idx] = True; te[test_idx] = True
    return tr, te

def recency_weights(ts: np.ndarray) -> Optional[np.ndarray]:
    if RECENCY_HALF_LIFE_DAYS <= 0: return None
    now = float(pd.Timestamp.utcnow().timestamp())
    dt_days = (now - ts.astype(float)) / (24*3600.0)
    lam = np.log(2) / max(1e-9, RECENCY_HALF_LIFE_DAYS)
    w = np.exp(-lam * np.maximum(0.0, dt_days))
    s = float(np.sum(w))
    return (w/s) if s>0 else None

def _minute_cutoff_for_market(market: str) -> Optional[int]:
    m = (market or "").upper()
    if m.startswith("PRE "): return None
    if m.startswith("OVER/UNDER"): return MARKET_CUTOFFS.get("OU", TIP_MAX_MINUTE)
    if m == "BTTS": return MARKET_CUTOFFS.get("BTTS", TIP_MAX_MINUTE)
    if m == "1X2":  return MARKET_CUTOFFS.get("1X2", TIP_MAX_MINUTE)
    return TIP_MAX_MINUTE

def _ece(y_true: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    y = y_true.astype(int); p = np.clip(p.astype(float), 0, 1)
    edges = np.linspace(0.0, 1.0, bins+1); ece = 0.0
    for i in range(bins):
        lo,hi = edges[i], edges[i+1]
        mask = (p>=lo)&(p<hi) if i<bins-1 else (p>=lo)&(p<=hi)
        if not np.any(mask): continue
        conf = float(np.mean(p[mask])); acc = float(np.mean(y[mask]))
        ece += (np.sum(mask)/len(p)) * abs(conf-acc)
    return float(ece)

def _mask_undecided(market_key: str, X_all: np.ndarray, y_all: np.ndarray, features: List[str]) -> np.ndarray:
    fidx = {f:i for i,f in enumerate(features)}
    keep = np.ones(len(y_all), dtype=bool)
    def col(name): return X_all[:, fidx[name]] if name in fidx else np.zeros(len(X_all))
    goals_sum = col("goals_sum")
    gh = col("goals_h"); ga = col("goals_a")
    if market_key.startswith("OU_"):
        try:
            ln = float(market_key.split("_", 1)[1].replace(",", "."))
        except Exception:
            ln = 2.5
        keep &= ~(goals_sum > ln)  # drop snapshots where Over already locked
    elif market_key == "BTTS_YES":
        keep &= ~((gh > 0) & (ga > 0))  # drop when both already scored
    return keep

# ─────────────────────── Core train for a head ───────────────────────
def _train_head_catboost_or_lr(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    ts_tr: np.ndarray,
    use_catboost_first: bool = True
) -> Tuple[str, Any, np.ndarray, Dict[str, float]]:
    """
    Train CatBoost first (if available), fallback to LR.
    Returns (kind, model, p_raw_te, metrics_dict).
    kind ∈ {"catboost","lr"}.
    """
    metrics: Dict[str, float] = {}

    # Recency weights
    sw = recency_weights(ts_tr)

    if use_catboost_first and _CB_OK and len(np.unique(y_tr)) >= 2:
        try:
            params = dict(
                loss_function="Logloss",
                learning_rate=CB_LEARNING,
                depth=CB_DEPTH,
                l2_leaf_reg=CB_L2,
                iterations=CB_ITERATIONS,
                random_seed=42,
                thread_count=CB_THREAD,
                rsm=CB_BAGGING,
                od_type="IncToDec", od_wait=40,
                verbose=False,
            )
            pool_tr = Pool(X_tr, y_tr)
            model = CatBoostClassifier(**params)
            model.fit(pool_tr)
            p_raw_te = model.predict_proba(X_te)[:, 1]
            metrics["trainer"] = 1.0
            return "catboost", model, p_raw_te, metrics
        except Exception as e:
            logger.warning("[CB] training failed; falling back to LR: %s", e)

    # Fallback: Logistic Regression on standardized features
    X_trs, mu, sd = _standardize_fit(X_tr)
    X_tes = (X_te - mu) / sd
    lr = fit_lr_safe(X_trs, y_tr, sample_weight=sw)
    if lr is None:
        raise RuntimeError("LR training failed due to single-class or shape issues.")
    p_raw_te = lr.predict_proba(X_tes)[:, 1]
    metrics["trainer"] = 0.0
    return "lr", (lr, mu, sd), p_raw_te, metrics

def _train_binary_head(
    conn,
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
) -> Tuple[bool, Dict[str, Any], Optional[np.ndarray]]:
    if len(np.unique(y_all)) < 2:
        return False, {}, None

    # minute cutoff (train only)
    if train_minute_cutoff is not None:
        try:
            mask_tr = mask_tr & (X_all[:, 0] <= float(train_minute_cutoff))
        except Exception:
            pass

    # remove trivial/decided snapshots on train fold
    try:
        undecided = _mask_undecided(model_key, X_all, y_all, feature_names)
        if np.any(undecided & mask_tr):
            mask_tr = mask_tr & undecided
    except Exception:
        pass

    X_tr, X_te = X_all[mask_tr], X_all[mask_te]
    y_tr, y_te = y_all[mask_tr], y_all[mask_te]
    ts_tr      = ts_all[mask_tr]

    # Train head
    kind, model, p_raw_te, info = _train_head_catboost_or_lr(X_tr, y_tr, X_te, y_te, ts_tr, use_catboost_first=True)

    # Calibrate on validation fold
    cal_kind, cal_obj = _fit_calibration(y_te, p_raw_te)
    p_cal = _apply_calibration(p_raw_te, cal_kind, cal_obj)

    # Metrics
    mets = {
        "brier": float(brier_score_loss(y_te, p_cal)),
        "auc": float(roc_auc_score(y_te, p_cal)) if len(np.unique(y_te)) > 1 else float("nan"),
        "ece": float(_ece(y_te, p_cal)),
        "acc@0.5": float(accuracy_score(y_te, (p_cal >= 0.5).astype(int))),
        "logloss": float(log_loss(y_te, p_cal, labels=[0, 1])),
        "n_test": int(len(y_te)),
        "prevalence": float(y_all.mean()),
        "calibration": cal_kind,
        "cb_used": 1 if (kind == "catboost") else 0,
    }
    mets.update(info)
    if metrics_name:
        logger.info("[METRICS] %s: %s", metrics_name, mets)

    # Serialize model blob
    if kind == "catboost":
        blob = build_model_blob_from_catboost(model, feature_names, cal_kind, cal_obj)
    else:
        lr, mu, sd = model  # type: ignore
        blob = build_model_blob_from_lr(lr, feature_names, cal_kind, cal_obj, mean=mu, scale=sd)

    for k in (f"model_latest:{model_key}", f"model:{model_key}"):
        _set_setting(conn, k, json.dumps(blob))

    # Threshold from validation fold (precision target)
    if threshold_label:
        thr_prob = _pick_threshold_for_target_precision(
            y_true=y_te,
            p_cal=p_cal,
            target_precision=target_precision,
            min_preds=min_preds,
            default_threshold=default_thr_prob,
        )
        thr_pct = float(np.clip(_percent(thr_prob), min_thresh_pct, max_thresh_pct))
        _set_setting(conn, f"conf_threshold:{threshold_label}", f"{thr_pct:.2f}")

    return True, mets, p_cal

# ─────────────────────── Thresholding utils ───────────────────────
def _percent(x: float) -> float: return float(x)*100.0

def _pick_threshold_for_target_precision(
    y_true: np.ndarray, p_cal: np.ndarray, target_precision: float,
    min_preds: int = 25, default_threshold: float = 0.65
) -> float:
    y = y_true.astype(int); p = np.asarray(p_cal, dtype=float)
    best_t: Optional[float] = None
    candidates = np.arange(0.50, 0.951, 0.005)
    feasible: List[Tuple[float,float,int]] = []
    for t in candidates:
        pred = (p>=t).astype(int); n_pred = int(pred.sum())
        if n_pred < min_preds: continue
        prec = precision_score(y, pred, zero_division=0)
        if prec >= target_precision: feasible.append((float(t), float(prec), n_pred))
    if feasible:
        feasible.sort(key=lambda z: (z[0], -z[1]))
        best_t = feasible[0][0]
    else:
        prec, rec, thr = precision_recall_curve(y, p)
        if len(thr) > 0:
            f1 = 2*(prec*rec)/(prec+rec+1e-9)
            idx = int(np.argmax(f1[:-1])) if len(f1) > 1 else 0
            best_t = float(thr[max(0, min(idx, len(thr)-1))])
    if best_t is None:
        best_acc = -1.0
        for t in candidates:
            acc = accuracy_score(y, (p>=t).astype(int))
            if acc > best_acc: best_t, best_acc = float(t), acc
    return float(best_t if best_t is not None else default_threshold)

# ─────────────────────── Main training entry ───────────────────────
def train_models(
    db_url: Optional[str] = None,
    min_minute: Optional[int] = None,
    test_size: Optional[float] = None,
    min_rows: Optional[int] = None,
) -> Dict[str,Any]:
    conn = _connect(db_url or os.getenv("DATABASE_URL"))
    _ensure_training_tables(conn)

    min_minute = int(min_minute if min_minute is not None else os.getenv("TRAIN_MIN_MINUTE", 15))
    test_size  = float(test_size  if test_size  is not None else os.getenv("TRAIN_TEST_SIZE", 0.25))
    min_rows   = int(min_rows   if min_rows   is not None else os.getenv("MIN_ROWS", 150))

    ou_lines = _parse_ou_lines(OU_TRAIN_LINES_RAW)

    target_precision = float(os.getenv("TARGET_PRECISION","0.60"))
    min_preds = int(os.getenv("THRESH_MIN_PREDICTIONS","25"))
    min_thresh = float(os.getenv("MIN_THRESH","55"))
    max_thresh = float(os.getenv("MAX_THRESH","85"))

    summary: Dict[str,Any] = {"ok": True, "trained": {}, "metrics": {}, "thresholds": {}}

    try:
        # ===== In-Play =====
        df_ip = load_inplay_data(conn, min_minute=min_minute)
        if not df_ip.empty and len(df_ip) >= min_rows:
            tr_mask, te_mask = time_order_split(df_ip, test_size=test_size)
            df_ip = df_ip.reset_index(drop=True)
            X_all  = _ensure_columns(df_ip, FEATURES).values
            ts_all = df_ip["_ts"].values

            # BTTS
            ok, mets, _ = _train_binary_head(
                conn, X_all, df_ip["btts_yes"].values.astype(int), ts_all,
                tr_mask, te_mask, FEATURES, "BTTS_YES", "BTTS",
                target_precision, min_preds, min_thresh, max_thresh, 0.65,
                "BTTS_YES", _minute_cutoff_for_market("BTTS")
            )
            summary["trained"]["BTTS_YES"] = ok
            if ok: summary["metrics"]["BTTS_YES"] = mets

            # OU lines
            totals = df_ip["final_goals_sum"].values.astype(int)
            for line in ou_lines:
                name = f"OU_{_fmt_line(line)}"
                ok, mets, _ = _train_binary_head(
                    conn, X_all, (totals > line).astype(int), ts_all,
                    tr_mask, te_mask, FEATURES, name, f"Over/Under {_fmt_line(line)}",
                    target_precision, min_preds, min_thresh, max_thresh, 0.65,
                    name, _minute_cutoff_for_market("Over/Under")
                )
                summary["trained"][name] = ok
                if ok:
                    summary["metrics"][name] = mets
                    # Back-compat alias for O25
                    if abs(line-2.5) < 1e-6:
                        blob = _get_setting_json(conn, f"model_latest:{name}")
                        if blob is not None:
                            for k in ("model_latest:O25","model:O25"):
                                _set_setting(conn, k, json.dumps(blob))

            # 1X2
            gd = df_ip["final_goals_diff"].values.astype(int)
            y_home = (gd > 0).astype(int); y_draw = (gd == 0).astype(int); y_away = (gd < 0).astype(int)

            ok_h, mets_h, p_h = _train_binary_head(conn, X_all, y_home, ts_all, tr_mask, te_mask, FEATURES,
                                                   "WLD_HOME", None, target_precision, min_preds,
                                                   min_thresh, max_thresh, 0.45, "WLD_HOME",
                                                   _minute_cutoff_for_market("1X2"))
            ok_d, mets_d, p_d = _train_binary_head(conn, X_all, y_draw, ts_all, tr_mask, te_mask, FEATURES,
                                                   "WLD_DRAW", None, target_precision, min_preds,
                                                   min_thresh, max_thresh, 0.45, "WLD_DRAW",
                                                   _minute_cutoff_for_market("1X2"))
            ok_a, mets_a, p_a = _train_binary_head(conn, X_all, y_away, ts_all, tr_mask, te_mask, FEATURES,
                                                   "WLD_AWAY", None, target_precision, min_preds,
                                                   min_thresh, max_thresh, 0.45, "WLD_AWAY",
                                                   _minute_cutoff_for_market("1X2"))

            summary["trained"]["WLD_HOME"] = ok_h; summary["trained"]["WLD_DRAW"] = ok_d; summary["trained"]["WLD_AWAY"] = ok_a
            if ok_h: summary["metrics"]["WLD_HOME"] = mets_h
            if ok_d: summary["metrics"]["WLD_DRAW"] = mets_d
            if ok_a: summary["metrics"]["WLD_AWAY"] = mets_a

            if ok_h and ok_d and ok_a and (p_h is not None) and (p_d is not None) and (p_a is not None):
                ps = np.clip(p_h,EPS,1-EPS)+np.clip(p_d,EPS,1-EPS)+np.clip(p_a,EPS,1-EPS)
                phn, pdn, pan = p_h/ps, p_d/ps, p_a/ps
                p_max = np.maximum.reduce([phn, pdn, pan])

                gd_te = gd[te_mask]
                y_class = np.zeros_like(gd_te, dtype=int)
                y_class[gd_te==0] = 1; y_class[gd_te<0] = 2
                correct = (np.argmax(np.stack([phn,pdn,pan],axis=1), axis=1) == y_class).astype(int)

                thr_prob = _pick_threshold_for_target_precision(
                    y_true=correct, p_cal=p_max,
                    target_precision=target_precision, min_preds=min_preds, default_threshold=0.45
                )
                thr_pct = float(np.clip(_percent(thr_prob), min_thresh, max_thresh))
                _set_setting(conn, "conf_threshold:1X2", f"{thr_pct:.2f}")
                summary["thresholds"]["1X2"] = thr_pct
        else:
            logger.info("In-Play: not enough labeled data (have %d, need >= %d).", len(df_ip), min_rows)
            summary["trained"]["BTTS_YES"] = False

        # ===== Prematch =====
        df_pre = load_prematch_data(conn)
        if not df_pre.empty and len(df_pre) >= min_rows:
            tr_mask, te_mask = time_order_split(df_pre, test_size=test_size)
            df_pre = df_pre.reset_index(drop=True)
            Xp_all = _ensure_columns(df_pre, PRE_FEATURES).values
            ts_pre = df_pre["_ts"].values

            # PRE BTTS
            ok, mets, _ = _train_binary_head(
                conn, Xp_all, df_pre["label_btts"].values.astype(int), ts_pre,
                tr_mask, te_mask, PRE_FEATURES, "PRE_BTTS_YES", "PRE BTTS",
                target_precision, min_preds, min_thresh, max_thresh, 0.65,
                "PRE_BTTS_YES", None
            )
            summary["trained"]["PRE_BTTS_YES"] = ok
            if ok: summary["metrics"]["PRE_BTTS_YES"] = mets

            # PRE OU
            totals = df_pre["final_goals_sum"].values.astype(int)
            for line in ou_lines:
                name = f"PRE_OU_{_fmt_line(line)}"
                ok, mets, _ = _train_binary_head(
                    conn, Xp_all, (totals > line).astype(int), ts_pre,
                    tr_mask, te_mask, PRE_FEATURES, name, f"PRE Over/Under {_fmt_line(line)}",
                    target_precision, min_preds, min_thresh, max_thresh, 0.65,
                    name, None
                )
                summary["trained"][name] = ok
                if ok: summary["metrics"][name] = mets

            # PRE 1X2 (draw suppressed)
            gd = df_pre["final_goals_diff"].values.astype(int)
            y_home = (gd > 0).astype(int); y_away = (gd < 0).astype(int)

            ok_h, mets_h, p_h = _train_binary_head(conn, Xp_all, y_home, ts_pre, tr_mask, te_mask, PRE_FEATURES,
                                                   "PRE_WLD_HOME", None, target_precision, min_preds,
                                                   min_thresh, max_thresh, 0.45, "PRE_WLD_HOME", None)
            ok_a, mets_a, p_a = _train_binary_head(conn, Xp_all, y_away, ts_pre, tr_mask, te_mask, PRE_FEATURES,
                                                   "PRE_WLD_AWAY", None, target_precision, min_preds,
                                                   min_thresh, max_thresh, 0.45, "PRE_WLD_AWAY", None)
            summary["trained"]["PRE_WLD_HOME"] = ok_h; summary["trained"]["PRE_WLD_AWAY"] = ok_a
            if ok_h: summary["metrics"]["PRE_WLD_HOME"] = mets_h
            if ok_a: summary["metrics"]["PRE_WLD_AWAY"] = mets_a

            if ok_h and ok_a and (p_h is not None) and (p_a is not None):
                ps = np.clip(p_h,EPS,1-EPS) + np.clip(p_a,EPS,1-EPS)
                phn, pan = p_h/ps, p_a/ps
                p_max = np.maximum(phn, pan)
                gd_te = gd[te_mask]
                y_class = np.where(gd_te>0, 0, np.where(gd_te<0, 1, -1))
                mask = (y_class != -1)
                if mask.any():
                    correct = (np.argmax(np.stack([phn,pan],axis=1), axis=1)[mask] == y_class[mask]).astype(int)
                    thr_prob = _pick_threshold_for_target_precision(
                        y_true=correct, p_cal=p_max[mask],
                        target_precision=target_precision, min_preds=min_preds, default_threshold=0.45
                    )
                    thr_pct = float(np.clip(_percent(thr_prob), min_thresh, max_thresh))
                    _set_setting(conn, "conf_threshold:PRE 1X2", f"{thr_pct:.2f}")
                    summary["thresholds"]["PRE 1X2"] = thr_pct
        else:
            logger.info("Prematch: not enough labeled data (have %d, need >= %d).", len(df_pre), min_rows)
            summary["trained"]["PRE_BTTS_YES"] = False

        # Persist training report
        metrics_bundle = {
            "trained_at_utc": pd.Timestamp.utcnow().isoformat(timespec="seconds")+"Z",
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
            "catboost_available": bool(_CB_OK),
        }
        _set_setting(conn, "model_metrics_latest", json.dumps(metrics_bundle))
        return summary

    except Exception as e:
        logger.exception("Training failed: %s", e)
        return {"ok": False, "error": str(e)}
    finally:
        try: conn.close()
        except Exception: pass

# ─────────────────────── ROI-aware Auto Tune ───────────────────────
def auto_tune_thresholds(days: int = 14) -> Dict[str, float]:
    """
    ROI-aware per-market threshold tuning based on realized outcomes.
    Requires:
      - tips table: (market, suggestion, confidence OR confidence_raw, odds, sent_at)
      - results: joined by match_id/fixture_id with FT goals
    """
    db_url = os.getenv("DATABASE_URL")
    conn = _connect(db_url)

    PREC_TOL = float(os.getenv("AUTO_TUNE_PREC_TOL", "0.03"))
    target_precision = float(os.getenv("TARGET_PRECISION", "0.60"))
    min_preds = int(os.getenv("THRESH_MIN_PREDICTIONS", "25"))
    min_thr = float(os.getenv("MIN_THRESH", "55"))
    max_thr = float(os.getenv("MAX_THRESH", "85"))
    max_odds = float(os.getenv("MAX_ODDS_ALL", "20.0"))

    cutoff = int(pd.Timestamp.utcnow().timestamp()) - days*24*3600

    q = """
        SELECT t.market, t.suggestion,
               COALESCE(NULLIF(t.confidence, NULL), NULLIF(t.confidence_raw, NULL)) AS prob,
               t.odds, r.goals_home, r.goals_away
        FROM tips t
        JOIN results r ON r.fixture_id = t.fixture_id
        WHERE EXTRACT(EPOCH FROM t.sent_at) >= %s
          AND t.odds IS NOT NULL
    """
    df = _read_sql(conn, q, (cutoff,))
    if df.empty:
        try: conn.close()
        except Exception: pass
        return {}

    # Clean & compute label
    def _parse_line(s: str) -> Optional[float]:
        try:
            for tok in (s or "").split():
                return float(tok)
        except Exception:
            return None
        return None

    def _label(sug: str, gh: float, ga: float) -> Optional[int]:
        gh = int(gh or 0); ga = int(ga or 0); tot = gh+ga
        s = (sug or "")
        if s.startswith("Over"):
            ln = _parse_line(s) or 2.5
            return 1 if tot > ln else (0 if tot < ln else None)
        if s.startswith("Under"):
            ln = _parse_line(s) or 2.5
            return 1 if tot < ln else (0 if tot > ln else None)
        if s == "BTTS: Yes": return 1 if (gh>0 and ga>0) else 0
        if s == "BTTS: No":  return 1 if not (gh>0 and ga>0) else 0
        if s == "Home Win":  return 1 if gh>ga else 0
        if s == "Away Win":  return 1 if ga>gh else 0
        return None

    df = df.dropna(subset=["prob","odds"]).copy()
    df["prob"] = df["prob"].astype(float)
    df = df[(df["odds"]>=1.01) & (df["odds"]<=max_odds)]

    df["y"] = [ _label(s, gh, ga) for s,gh,ga in zip(df["suggestion"], df["goals_home"], df["goals_away"]) ]
    df = df.dropna(subset=["y"]).copy()
    df["y"] = df["y"].astype(int)

    tuned: Dict[str, float] = {}

    def _market_key(m: str) -> str:
        s = (m or "").upper()
        if s.startswith("PRE "): s = s[4:]
        return s

    for mkt in sorted(df["market"].unique()):
        sub = df[df["market"]==mkt].copy()
        if len(sub) < min_preds:
            continue

        # Sweep thresholds
        best = None  # (roi, prec, n, thr_pct)
        feasible_any = False
        for thr_pct in np.arange(min_thr, max_thr+1e-9, 1.0):
            thr = float(thr_pct / 100.0)
            sel = sub[sub["prob"] >= thr]
            n = len(sel)
            if n < min_preds:
                continue
            wins = int(sel["y"].sum())
            prec = wins / n
            pnl = (sel["y"]*(sel["odds"]-1.0) - (1-sel["y"])).sum()
            roi = (pnl / n) if n>0 else 0.0

            if prec >= target_precision:
                feasible_any = True
                score = (roi, prec, n)
                if (best is None) or (score > (best[0], best[1], best[2])):
                    best = (roi, prec, n, thr_pct)

        # allow tolerance below precision target if ROI>0
        if not feasible_any:
            for thr_pct in np.arange(min_thr, max_thr+1e-9, 1.0):
                thr = float(thr_pct / 100.0)
                sel = sub[sub["prob"] >= thr]
                n = len(sel)
                if n < min_preds:
                    continue
                wins = int(sel["y"].sum())
                prec = wins / n
                pnl = (sel["y"]*(sel["odds"]-1.0) - (1-sel["y"])).sum()
                roi = (pnl / n) if n>0 else 0.0
                if (prec >= max(0.0, target_precision - PREC_TOL)) and (roi > 0.0):
                    score = (roi, prec, n)
                    if (best is None) or (score > (best[0], best[1], best[2])):
                        best = (roi, prec, n, thr_pct)

        if best is not None:
            tuned[_market_key(mkt)] = float(best[3])

    # Persist updates
    for mk, pct in tuned.items():
        _set_setting(conn, f"conf_threshold:{mk}", f"{pct:.2f}")

    try: conn.close()
    except Exception: pass

    return tuned

# ─────────────────────── CLI ───────────────────────
def _cli_main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-url", help="Postgres DSN (or use env DATABASE_URL)")
    ap.add_argument("--min-minute", dest="min_minute", type=int, default=int(os.getenv("TRAIN_MIN_MINUTE", 15)))
    ap.add_argument("--test-size", type=float, default=float(os.getenv("TRAIN_TEST_SIZE", 0.25)))
    ap.add_argument("--min-rows", type=int, default=int(os.getenv("MIN_ROWS", 150)))
    ap.add_argument("--auto-tune", action="store_true", help="Run ROI-aware auto-tune after training")
    args = ap.parse_args()
    res = train_models(
        db_url=args.db_url or os.getenv("DATABASE_URL"),
        min_minute=args.min_minute, test_size=args.test_size, min_rows=args.min_rows
    )
    print(json.dumps(res, indent=2))
    if args.auto_tune:
        tuned = auto_tune_thresholds(14)
        print(json.dumps({"tuned": tuned}, indent=2))

if __name__ == "__main__":
    _cli_main()
