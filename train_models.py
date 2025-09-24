# file: train_models.py

import argparse, json, os, logging, math
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

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger("trainer")

# ───────────────────────── Feature sets (match main.py) ───────────────────────── #

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

# Prematch snapshots (as saved by main.save_prematch_snapshot via extract_prematch_features)
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

# ─────────────────────── Env knobs ─────────────────────── #

RECENCY_HALF_LIFE_DAYS = float(os.getenv("RECENCY_HALF_LIFE_DAYS", "120"))  # 0 disables
RECENCY_MONTHS         = int(os.getenv("RECENCY_MONTHS", "36"))             # training window cap
MARKET_CUTOFFS_RAW     = os.getenv("MARKET_CUTOFFS", "BTTS=75,1X2=80,OU=88")
TIP_MAX_MINUTE_ENV     = os.getenv("TIP_MAX_MINUTE", "")
OU_TRAIN_LINES_RAW     = os.getenv("OU_TRAIN_LINES", os.getenv("OU_LINES", "2.5,3.5"))

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

# ─────────────────────── DB utils ─────────────────────── #

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

def _get_setting_json(conn, key: str) -> Optional[dict]:
    try:
        df = _read_sql(conn, "SELECT value FROM settings WHERE key=%s", (key,))
        if df.empty: return None
        return json.loads(df.iloc[0]["value"])
    except Exception:
        return None

# ─────────────────────── Data loaders ─────────────────────── #

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

# ─────────────────────── Optional helpers for main.py calibration ─────────────────────── #

def _tip_outcome_for_result(suggestion: str, gh: int, ga: int, btts_yes: int) -> Optional[int]:
    s = (suggestion or "").strip()
    total = int(gh) + int(ga)
    if s.startswith("Over") or s.startswith("Under"):
        # parse line
        try:
            line = None
            for tok in s.split():
                try:
                    line = float(tok)
                    break
                except:
                    pass
            if line is None:
                return None
        except:
            return None
        if s.startswith("Over"):
            if total > line: return 1
            if abs(total - line) < 1e-9: return None
            return 0
        else:
            if total < line: return 1
            if abs(total - line) < 1e-9: return None
            return 0
    if s == "BTTS: Yes": return 1 if int(btts_yes) == 1 else 0
    if s == "BTTS: No":  return 1 if int(btts_yes) == 0 else 0
    if s == "Home Win":  return 1 if gh > ga else 0
    if s == "Away Win":  return 1 if ga > gh else 0
    return None

def load_graded_tips(conn, days: int = 365) -> pd.DataFrame:
    """
    For calibrate_and_retune_from_tips in main.py.
    Returns a DataFrame with columns: [market, suggestion, prob, y]
    """
    cutoff = int(pd.Timestamp.utcnow().timestamp()) - days*24*3600
    q = """
    SELECT t.market, t.suggestion,
           COALESCE(t.confidence_raw, t.confidence/100.0) AS prob,
           t.match_id,
           r.final_goals_h, r.final_goals_a, r.btts_yes
    FROM tips t
    JOIN match_results r ON r.match_id = t.match_id
    WHERE t.created_ts >= %s
      AND t.suggestion <> 'HARVEST'
      AND t.sent_ok = 1
    """
    df = _read_sql(conn, q, (cutoff,))
    if df.empty:
        return df
    ys = []
    for _, row in df.iterrows():
        y = _tip_outcome_for_result(
            str(row["suggestion"]),
            int(row["final_goals_h"] or 0),
            int(row["final_goals_a"] or 0),
            int(row["btts_yes"] or 0),
        )
        ys.append(np.nan if y is None else int(y))
    df["y"] = ys
    df = df.dropna(subset=["prob", "y"]).copy()
    df["prob"] = df["prob"].astype(float).clip(1e-6, 1-1e-6)
    df["y"] = df["y"].astype(int)
    return df[["market", "suggestion", "prob", "y"]]

def _logit_vec(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(float), 1e-6, 1-1e-6)
    return np.log(p/(1.0-p))

def fit_platt(y_true: np.ndarray, p_raw: np.ndarray) -> Tuple[float, float]:
    """
    Simple Platt scaling on logits; returns (a, b) such that sigmoid(a*logit(p_raw)+b)
    """
    y = y_true.astype(int)
    z = _logit_vec(p_raw).reshape(-1,1)
    lr = LogisticRegression(max_iter=1000, solver="lbfgs").fit(z, y)
    a = float(lr.coef_.ravel()[0]); b = float(lr.intercept_.ravel()[0])
    return a, b

def _percent(x: float) -> float: return float(x)*100.0

# ─────────────────────── Modeling helpers ─────────────────────── #

def fit_lr_safe(X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray]=None) -> Optional[LogisticRegression]:
    if len(np.unique(y)) < 2:
        return None
    C = float(os.getenv("LR_C", "1.0"))  # tunable
    return LogisticRegression(
        max_iter=5000,
        solver="saga",
        penalty="l2",
        class_weight="balanced",
        n_jobs=-1,
        C=C,
        random_state=42,
    ).fit(X, y, sample_weight=sample_weight)

def _fit_calibration(y_true: np.ndarray, p_raw: np.ndarray) -> Tuple[str, Any]:
    """Return ('platt',(a,b)) or ('isotonic', IsotonicRegression)."""
    y = y_true.astype(int)
    z = _logit_vec(p_raw).reshape(-1,1)

    # Platt on logits
    lr = LogisticRegression(max_iter=1000, solver="lbfgs").fit(z, y)
    a = float(lr.coef_.ravel()[0]); b = float(lr.intercept_.ravel()[0])
    p_platt = 1.0/(1.0+np.exp(-(a*z.ravel()+b)))
    brier_platt = brier_score_loss(y, p_platt)

    best_kind, best_obj, best_brier = "platt", (a,b), brier_platt

    # Isotonic if enough mass and both classes present
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
    """Return (X_std, mean, scale). Any zero-variance column is left unscaled."""
    mean = X.mean(axis=0)
    scale = X.std(axis=0, ddof=0)
    scale = np.where(scale <= 1e-12, 1.0, scale)
    Xs = (X - mean) / scale
    return Xs, mean, scale

def _fold_std_into_weights(model: LogisticRegression, mean: np.ndarray, scale: np.ndarray, features: List[str]) -> Tuple[float, Dict[str, float]]:
    """Convert weights learned on standardized X back to raw feature space."""
    w_std = model.coef_.ravel().astype(float)
    b_std = float(model.intercept_.ravel()[0])
    w_raw = (w_std / scale).astype(float)
    b_raw = float(b_std - np.sum(w_std * (mean / scale)))
    weights = {name: float(w) for name, w in zip(features, w_raw.tolist())}
    return b_raw, weights

def build_model_blob(model: LogisticRegression, features: List[str], cal_kind: str, cal_obj,
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
        # approximate isotonic with a Platt-like mapping to keep serving simple
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

# ─────────────────────── Thresholding & splits ─────────────────────── #

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

# ─────────────────────── Core training ─────────────────────── #

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
    """
    Keep rows where the outcome isn't trivially decided at snapshot.
    - OU_L: drop rows where goals_sum > line (Over guaranteed)
    - BTTS: drop rows where goals_h>0 and goals_a>0 (Yes guaranteed)
    - 1X2: keep all (never fully decided until FT)
    """
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
        keep &= ~(goals_sum > ln)
    elif market_key == "BTTS_YES":
        keep &= ~((gh > 0) & (ga > 0))
    return keep

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

    if train_minute_cutoff is not None:
        try:
            mt = (X_all[:, 0] <= float(train_minute_cutoff))
            mask_tr = mask_tr & mt
        except Exception:
            pass

    try:
        undecided = _mask_undecided(model_key, X_all, y_all, feature_names)
        new_mask_tr = mask_tr & undecided
        if np.any(new_mask_tr):
            mask_tr = new_mask_tr
    except Exception:
        pass

    X_tr, X_te = X_all[mask_tr], X_all[mask_te]
    y_tr, y_te = y_all[mask_tr], y_all[mask_te]
    ts_tr      = ts_all[mask_tr]

    X_tr_std, mu, sd = _standardize_fit(X_tr)
    X_te_std = (X_te - mu) / sd

    sw = recency_weights(ts_tr)
    model = fit_lr_safe(X_tr_std, y_tr, sample_weight=sw)
    if model is None:
        return False, {}, None

    p_raw_te = model.predict_proba(X_te_std)[:, 1]
    cal_kind, cal_obj = _fit_calibration(y_te, p_raw_te)
    p_cal = _apply_calibration(p_raw_te, cal_kind, cal_obj)

    mets = {
        "brier": float(brier_score_loss(y_te, p_cal)),
        "auc": float(roc_auc_score(y_te, p_cal)) if len(np.unique(y_te)) > 1 else float("nan"),
        "ece": float(_ece(y_te, p_cal)),
        "acc@0.5": float(accuracy_score(y_te, (p_cal >= 0.5).astype(int))),
        "logloss": float(log_loss(y_te, p_cal, labels=[0, 1])),
        "n_test": int(len(y_te)),
        "prevalence": float(y_all.mean()),
        "calibration": cal_kind,
    }
    if metrics_name:
        logger.info("[METRICS] %s: %s", metrics_name, mets)

    blob = build_model_blob(model, feature_names, cal_kind, cal_obj, mean=mu, scale=sd)
    for k in (f"model_latest:{model_key}", f"model:{model_key}"):
        _set_setting(conn, k, json.dumps(blob))

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

# ─────────────────────── Entry point ─────────────────────── #

def _fmt_line(line: float) -> str: return f"{line}".rstrip("0").rstrip(".")

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
        }
        _set_setting(conn, "model_metrics_latest", json.dumps(metrics_bundle))
        return summary

    except Exception as e:
        logger.exception("Training failed: %s", e)
        return {"ok": False, "error": str(e)}
    finally:
        try: conn.close()
        except Exception: pass

# ─────────────────────── CLI ─────────────────────── #

def _cli_main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-url", help="Postgres DSN (or use env DATABASE_URL)")
    ap.add_argument("--min-minute", dest="min_minute", type=int, default=int(os.getenv("TRAIN_MIN_MINUTE", 15)))
    ap.add_argument("--test-size", type=float, default=float(os.getenv("TRAIN_TEST_SIZE", 0.25)))
    ap.add_argument("--min-rows", type=int, default=int(os.getenv("MIN_ROWS", 150)))
    args = ap.parse_args()
    res = train_models(
        db_url=args.db_url or os.getenv("DATABASE_URL"),
        min_minute=args.min_minute, test_size=args.test_size, min_rows=args.min_rows
    )
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    _cli_main()
