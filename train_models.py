# file: train_models.py
"""
Postgres-only training with Platt calibration + per-market auto-thresholding.

Trains two families of models that match main.py:
  In-Play:
    • BTTS_YES (binary LR)
    • OU_{line}  (e.g., OU_2.5, OU_3.5)
    • WLD_HOME, WLD_DRAW, WLD_AWAY (OvR LR heads; serving suppresses Draw)
  Prematch:
    • PRE_BTTS_YES
    • PRE_OU_{line}
    • PRE_WLD_HOME, PRE_WLD_AWAY  (Draw suppressed at serving)

Data sources:
  • In-Play   -> tip_snapshots  (latest snapshot per match)
  • Prematch  -> prematch_snapshots (one snapshot per match)

Models & thresholds written to `settings`:
  • model_latest:* and model:* keys (JSON: intercept, weights, calibration)
  • conf_threshold:* per market (percent; serving uses if present)

Environment knobs (sane defaults):
  DATABASE_URL
  OU_TRAIN_LINES="2.5,3.5"
  TRAIN_MIN_MINUTE=15           (min in-play minute for snapshot inclusion)
  TRAIN_TEST_SIZE=0.25
  TARGET_PRECISION=0.60
  THRESH_MIN_PREDICTIONS=25
  MIN_THRESH=55                 (min allowed threshold %)
  MAX_THRESH=85                 (max allowed threshold %)
  MIN_ROWS=150                  (early-exit if dataset smaller)
"""

import argparse
import json
import os
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    accuracy_score,
    log_loss,
    precision_score,
    f1_score,
)
import psycopg2

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ───────────────────────── Feature sets ───────────────────────── #

# Must match main.py extract_features() keys used by the in-play models
FEATURES: List[str] = [
    "minute",
    "goals_h", "goals_a", "goals_sum", "goals_diff",
    "xg_h", "xg_a", "xg_sum", "xg_diff",
    "sot_h", "sot_a", "sot_sum",
    "cor_h", "cor_a", "cor_sum",
    "pos_h", "pos_a", "pos_diff",
    "red_h", "red_a", "red_sum",
]

# Must match main.py extract_prematch_features() keys for PRE_* models
PRE_FEATURES: List[str] = [
    "pm_gf_h","pm_ga_h","pm_win_h",
    "pm_gf_a","pm_ga_a","pm_win_a",
    "pm_ov25_h","pm_ov35_h","pm_btts_h",
    "pm_ov25_a","pm_ov35_a","pm_btts_a",
    "pm_ov25_h2h","pm_ov35_h2h","pm_btts_h2h",
    "pm_rest_diff",
    # keep the live keys for compatibility (serve-time expects them to exist; they are 0.0 in prematch)
    "minute","goals_h","goals_a","goals_sum","goals_diff",
    "xg_h","xg_a","xg_sum","xg_diff","sot_h","sot_a","sot_sum",
    "cor_h","cor_a","cor_sum","pos_h","pos_a","pos_diff","red_h","red_a","red_sum",
]

EPS = 1e-6


# ─────────────────────── DB helpers ─────────────────────── #

def _connect(db_url: str):
    if not db_url:
        raise SystemExit("DATABASE_URL must be set.")
    if "sslmode=" not in db_url:
        db_url = db_url + ("&" if "?" in db_url else "?") + "sslmode=require"
    conn = psycopg2.connect(db_url)
    conn.autocommit = True
    return "pg", conn


def _read_sql(conn, sql: str, params: Tuple = ()) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn, params=params)


def _exec(conn, sql: str, params: Tuple) -> None:
    with conn.cursor() as cur:
        cur.execute(sql, params)


def _set_setting(conn, key: str, value: str) -> None:
    _exec(conn,
          "INSERT INTO settings(key,value) VALUES(%s,%s) "
          "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
          (key, value))


# ─────────────────────── Data load ─────────────────────── #

def load_inplay_data(conn, min_minute: int = 15) -> pd.DataFrame:
    """
    Uses the latest tip_snapshots per match and joins final results.
    """
    q = """
    WITH latest AS (
      SELECT match_id, MAX(created_ts) AS ts
      FROM tip_snapshots GROUP BY match_id
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

    feats: List[Dict[str, Any]] = []
    for _, row in rows.iterrows():
        try:
            payload = json.loads(row["payload"])
        except Exception:
            continue
        stat = (payload.get("stat") or {}) if isinstance(payload, dict) else {}

        try:
            f: Dict[str, float] = {
                "minute": float(payload.get("minute", 0)),
                "goals_h": float(payload.get("gh", 0)),
                "goals_a": float(payload.get("ga", 0)),
                "xg_h": float(stat.get("xg_h", 0)),
                "xg_a": float(stat.get("xg_a", 0)),
                "sot_h": float(stat.get("sot_h", 0)),
                "sot_a": float(stat.get("sot_a", 0)),
                "cor_h": float(stat.get("cor_h", 0)),
                "cor_a": float(stat.get("cor_a", 0)),
                "pos_h": float(stat.get("pos_h", 0)),
                "pos_a": float(stat.get("pos_a", 0)),
                "red_h": float(stat.get("red_h", 0)),
                "red_a": float(stat.get("red_a", 0)),
            }
        except Exception:
            continue

        # Derived
        f["goals_sum"] = f["goals_h"] + f["goals_a"]
        f["goals_diff"] = f["goals_h"] - f["goals_a"]
        f["xg_sum"] = f["xg_h"] + f["xg_a"]
        f["xg_diff"] = f["xg_h"] - f["xg_a"]
        f["sot_sum"] = f["sot_h"] + f["sot_a"]
        f["cor_sum"] = f["cor_h"] + f["cor_a"]
        f["pos_diff"] = f["pos_h"] - f["pos_a"]
        f["red_sum"] = f["red_h"] + f["red_a"]

        gh_f = int(row["final_goals_h"] or 0)
        ga_f = int(row["final_goals_a"] or 0)
        f["_ts"] = int(row["created_ts"] or 0)
        f["final_goals_sum"] = gh_f + ga_f
        f["final_goals_diff"] = gh_f - ga_f
        f["label_btts"] = 1 if int(row["btts_yes"] or 0) == 1 else 0

        feats.append(f)

    if not feats:
        return pd.DataFrame()

    df = pd.DataFrame(feats)
    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["minute"] = df["minute"].clip(0, 120)
    df = df[df["minute"] >= float(min_minute)].copy()
    return df


def load_prematch_data(conn) -> pd.DataFrame:
    """
    Reads prematch_snapshots and joins final results.
    Payload schema: {"feat": { ...PRE_FEATURES... }}
    """
    q = """
    SELECT p.match_id, p.created_ts, p.payload,
           r.final_goals_h, r.final_goals_a, r.btts_yes
    FROM prematch_snapshots p
    JOIN match_results r ON r.match_id = p.match_id
    """
    rows = _read_sql(conn, q)
    if rows.empty:
        return pd.DataFrame()

    feats: List[Dict[str, Any]] = []
    for _, row in rows.iterrows():
        try:
            payload = json.loads(row["payload"]) or {}
            feat = (payload.get("feat") or {})
        except Exception:
            continue

        f = {k: float(feat.get(k, 0.0) or 0.0) for k in PRE_FEATURES}

        gh_f = int(row["final_goals_h"] or 0)
        ga_f = int(row["final_goals_a"] or 0)
        f["_ts"] = int(row["created_ts"] or 0)
        f["final_goals_sum"]  = gh_f + ga_f
        f["final_goals_diff"] = gh_f - ga_f
        f["label_btts"] = 1 if int(row["btts_yes"] or 0) == 1 else 0

        feats.append(f)

    if not feats:
        return pd.DataFrame()
    df = pd.DataFrame(feats).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


# ─────────────────────── Model utils ─────────────────────── #

def fit_lr_safe(X: np.ndarray, y: np.ndarray) -> Optional[LogisticRegression]:
    """Balanced LR to deal with skew; liblinear is stable for small-ish data."""
    if len(np.unique(y)) < 2:
        return None
    return LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear").fit(X, y)


def weights_dict(model: LogisticRegression, feature_names: List[str]) -> Dict[str, float]:
    return {name: float(w) for name, w in zip(feature_names, model.coef_.ravel().tolist())}


def _logit_vec(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_platt(y_true: np.ndarray, p_raw: np.ndarray) -> Tuple[float, float]:
    """
    1D Platt scaling: logistic regression on logits of raw probabilities.
    Returns a,b so that: p_cal = sigmoid(a * logit(p_raw) + b)
    """
    z = _logit_vec(p_raw).reshape(-1, 1)
    y = y_true.astype(int)
    lr = LogisticRegression(max_iter=1000, solver="lbfgs")
    lr.fit(z, y)
    a = float(lr.coef_.ravel()[0])
    b = float(lr.intercept_.ravel()[0])
    return a, b


def build_model_blob(model: LogisticRegression, features: List[str],
                     cal: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
    blob = {
        "intercept": float(model.intercept_.ravel()[0]),
        "weights": weights_dict(model, features),
        "calibration": {"method": "sigmoid", "a": 1.0, "b": 0.0},
    }
    if cal is not None:
        a, b = cal
        blob["calibration"] = {"method": "platt", "a": float(a), "b": float(b)}
    return blob


def _fmt_line(line: float) -> str:
    return f"{line}".rstrip("0").rstrip(".")


# ─────────────────────── Thresholding ─────────────────────── #

def _percent(x: float) -> float:
    return float(x) * 100.0


def _grid_thresholds(lo=0.50, hi=0.95, step=0.005) -> np.ndarray:
    return np.arange(lo, hi + 1e-9, step)


def _pick_threshold_for_target_precision(
    y_true: np.ndarray,
    p_cal: np.ndarray,
    target_precision: float,
    min_preds: int = 25,
    default_threshold: float = 0.65,
) -> float:
    """
    Choose the smallest threshold t achieving >= target_precision and >= min_preds positives.
    If none, fall back to best F1, then best accuracy, then default.
    Returns threshold in [0..1].
    """
    y = y_true.astype(int)
    p = np.asarray(p_cal).astype(float)
    best_t: Optional[float] = None
    candidates = _grid_thresholds(0.50, 0.95, 0.005)

    feasible: List[Tuple[float, float, int]] = []
    for t in candidates:
        pred = (p >= t).astype(int)
        n_pred = int(pred.sum())
        if n_pred < min_preds:
            continue
        prec = precision_score(y, pred, zero_division=0)
        if prec >= target_precision:
            feasible.append((t, float(prec), n_pred))
    if feasible:
        feasible.sort(key=lambda z: (z[0], -z[1]))  # prefer smallest t, then higher precision
        best_t = feasible[0][0]

    if best_t is None:
        best_f1 = -1.0
        for t in candidates:
            pred = (p >= t).astype(int)
            if int(pred.sum()) < min_preds:
                continue
            f1 = f1_score(y, pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)

    if best_t is None:
        best_acc = -1.0
        for t in candidates:
            pred = (p >= t).astype(int)
            acc = accuracy_score(y, pred)
            if acc > best_acc:
                best_acc = acc
                best_t = float(t)

    return float(best_t if best_t is not None else default_threshold)


# ─────────────────────── Time-based split ─────────────────────── #

def time_order_split(df: pd.DataFrame, test_size: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns boolean masks (train_mask, test_mask) by time order.
    Latest `test_size` fraction goes to test. Stable across markets.
    """
    if "_ts" not in df.columns:
        n = len(df)
        idx = np.arange(n)
        rng = np.random.default_rng(42)
        rng.shuffle(idx)
        cut = int((1 - test_size) * n)
        tr = np.zeros(n, dtype=bool); te = np.zeros(n, dtype=bool)
        tr[idx[:cut]] = True; te[idx[cut:]] = True
        return tr, te

    df_sorted = df.sort_values("_ts").reset_index(drop=True)
    n = len(df_sorted)
    cut = int(max(1, (1 - test_size) * n))
    train_idx = df_sorted.index[:cut].to_numpy()
    test_idx  = df_sorted.index[cut:].to_numpy()
    tr = np.zeros(n, dtype=bool); te = np.zeros(n, dtype=bool)
    tr[train_idx] = True; te[test_idx] = True
    return tr, te


# ─────────────────────── Training core ─────────────────────── #

def _train_binary_head(
    conn,
    X_all: np.ndarray,
    y_all: np.ndarray,
    mask_tr: np.ndarray,
    mask_te: np.ndarray,
    feature_names: List[str],
    model_key: str,
    threshold_key: Optional[str],
    target_precision: float,
    min_preds: int,
    min_thresh_pct: float,
    max_thresh_pct: float,
    default_thr_prob: float,
    metrics_name: Optional[str] = None,
    threshold_label: Optional[str] = None,
) -> Tuple[bool, Dict[str, Any], Optional[np.ndarray]]:
    """
    Fits LR + Platt, writes model blob to settings, returns (trained?, metrics, p_cal_test or None)
    """
    if len(np.unique(y_all)) < 2:
        return False, {}, None

    X_tr, X_te = X_all[mask_tr], X_all[mask_te]
    y_tr, y_te = y_all[mask_tr], y_all[mask_te]

    m = fit_lr_safe(X_tr, y_tr)
    if m is None:
        return False, {}, None

    p_raw = m.predict_proba(X_te)[:, 1]
    a, b = fit_platt(y_te, p_raw)
    z = _logit_vec(p_raw)
    p_cal = 1.0 / (1.0 + np.exp(-(a * z + b)))

    # Persist model
    blob = build_model_blob(m, feature_names, (a, b))
    for k in (f"model_latest:{model_key}", f"model:{model_key}"):
        _set_setting(conn, k, json.dumps(blob))

    # Metrics
    mets = {
        "brier": float(brier_score_loss(y_te, p_cal)),
        "acc": float(accuracy_score(y_te, (p_cal >= 0.5).astype(int))),
        "logloss": float(log_loss(y_te, p_cal, labels=[0, 1])),
        "n_test": int(len(y_te)),
        "prevalence": float(y_all.mean()),
    }
    if metrics_name:
        logger.info("[METRICS] %s: %s", metrics_name, mets)

    # Threshold
    if threshold_key and threshold_label:
        thr_prob = _pick_threshold_for_target_precision(
            y_true=y_te, p_cal=p_cal,
            target_precision=target_precision, min_preds=min_preds,
            default_threshold=default_thr_prob,
        )
        thr_pct = float(np.clip(_percent(thr_prob), min_thresh_pct, max_thresh_pct))
        _set_setting(conn, f"conf_threshold:{threshold_label}", f"{thr_pct:.2f}")

    return True, mets, p_cal


def train_models(
    db_url: Optional[str] = None,
    min_minute: Optional[int] = None,
    test_size: Optional[float] = None,
    min_rows: Optional[int] = None,
) -> Dict[str, Any]:
    # Env + inputs
    db_url = db_url or os.getenv("DATABASE_URL")
    engine, conn = _connect(db_url)

    min_minute = int(min_minute if min_minute is not None else os.getenv("TRAIN_MIN_MINUTE", 15))
    test_size = float(test_size if test_size is not None else os.getenv("TRAIN_TEST_SIZE", 0.25))
    min_rows = int(min_rows if min_rows is not None else os.getenv("MIN_ROWS", 150))

    ou_lines_env = os.getenv("OU_TRAIN_LINES", "2.5,3.5")
    ou_lines: List[float] = []
    for t in ou_lines_env.split(","):
        t = t.strip()
        if not t:
            continue
        try:
            ou_lines.append(float(t))
        except Exception:
            continue
    if not ou_lines:
        ou_lines = [2.5, 3.5]

    target_precision = float(os.getenv("TARGET_PRECISION", "0.60"))
    min_preds = int(os.getenv("THRESH_MIN_PREDICTIONS", "25"))
    min_thresh = float(os.getenv("MIN_THRESH", "55"))
    max_thresh = float(os.getenv("MAX_THRESH", "85"))

    summary: Dict[str, Any] = {"ok": True, "trained": {}, "metrics": {}, "thresholds": {}}

    try:
        # ───────── In-Play training ─────────
        df_ip = load_inplay_data(conn, min_minute=min_minute)
        if not df_ip.empty and len(df_ip) >= min_rows:
            tr_mask, te_mask = time_order_split(df_ip, test_size=test_size)
            X_all = df_ip[FEATURES].values

            # BTTS (in-play)
            yb = df_ip["label_btts"].values.astype(int)
            ok, mets, p_b_cal = _train_binary_head(
                conn, X_all, yb, tr_mask, te_mask, FEATURES,
                model_key="BTTS_YES",
                threshold_key="conf_threshold:BTTS",
                threshold_label="BTTS",
                target_precision=target_precision,
                min_preds=min_preds,
                min_thresh_pct=min_thresh,
                max_thresh_pct=max_thresh,
                default_thr_prob=0.65,
                metrics_name="BTTS_YES",
            )
            summary["trained"]["BTTS_YES"] = ok
            if ok:
                summary["metrics"]["BTTS_YES"] = mets
                # threshold already saved under conf_threshold:BTTS

            # O/U (in-play)
            totals = df_ip["final_goals_sum"].values.astype(int)
            for line in ou_lines:
                y_over = (totals > line).astype(int)
                name = f"OU_{_fmt_line(line)}"
                ok, mets, p_cal = _train_binary_head(
                    conn, X_all, y_over, tr_mask, te_mask, FEATURES,
                    model_key=name,
                    threshold_key=f"conf_threshold:Over/Under {_fmt_line(line)}",
                    threshold_label=f"Over/Under {_fmt_line(line)}",
                    target_precision=target_precision,
                    min_preds=min_preds,
                    min_thresh_pct=min_thresh,
                    max_thresh_pct=max_thresh,
                    default_thr_prob=0.65,
                    metrics_name=name,
                )
                summary["trained"][name] = ok
                if ok:
                    summary["metrics"][name] = mets
                    # Alias for 2.5 -> O25 for serve-time legacy
                    if abs(line - 2.5) < 1e-6:
                        blob = get_setting_json(conn, f"model_latest:{name}")
                        if blob is not None:
                            for k in ("model_latest:O25", "model:O25"):
                                _set_setting(conn, k, json.dumps(blob))

            # 1X2 (in-play) via OvR
            gd = df_ip["final_goals_diff"].values.astype(int)
            y_home = (gd > 0).astype(int)
            y_draw = (gd == 0).astype(int)
            y_away = (gd < 0).astype(int)

            # Heads
            ok_h, mets_h, p_h = _train_binary_head(
                conn, X_all, y_home, tr_mask, te_mask, FEATURES,
                model_key="WLD_HOME",
                threshold_key=None, threshold_label=None,
                target_precision=target_precision, min_preds=min_preds,
                min_thresh_pct=min_thresh, max_thresh_pct=max_thresh,
                default_thr_prob=0.45, metrics_name="WLD_HOME",
            )
            ok_d, mets_d, p_d = _train_binary_head(
                conn, X_all, y_draw, tr_mask, te_mask, FEATURES,
                model_key="WLD_DRAW",
                threshold_key=None, threshold_label=None,
                target_precision=target_precision, min_preds=min_preds,
                min_thresh_pct=min_thresh, max_thresh_pct=max_thresh,
                default_thr_prob=0.45, metrics_name="WLD_DRAW",
            )
            ok_a, mets_a, p_a = _train_binary_head(
                conn, X_all, y_away, tr_mask, te_mask, FEATURES,
                model_key="WLD_AWAY",
                threshold_key=None, threshold_label=None,
                target_precision=target_precision, min_preds=min_preds,
                min_thresh_pct=min_thresh, max_thresh_pct=max_thresh,
                default_thr_prob=0.45, metrics_name="WLD_AWAY",
            )
            summary["trained"]["WLD_HOME"] = ok_h
            summary["trained"]["WLD_DRAW"] = ok_d
            summary["trained"]["WLD_AWAY"] = ok_a
            if ok_h: summary["metrics"]["WLD_HOME"] = mets_h
            if ok_d: summary["metrics"]["WLD_DRAW"] = mets_d
            if ok_a: summary["metrics"]["WLD_AWAY"] = mets_a

            # Threshold for 1X2 = calibrated OvR, renormalized; choose smallest t hitting precision
            if ok_h and ok_d and ok_a and (p_h is not None) and (p_d is not None) and (p_a is not None):
                ps = np.clip(p_h, EPS, 1 - EPS) + np.clip(p_d, EPS, 1 - EPS) + np.clip(p_a, EPS, 1 - EPS)
                phn, pdn, pan = p_h / ps, p_d / ps, p_a / ps
                p_max = np.maximum.reduce([phn, pdn, pan])

                gd_te = gd[te_mask]
                y_class = np.zeros_like(gd_te, dtype=int)  # 0=home,1=draw,2=away
                y_class[gd_te == 0] = 1
                y_class[gd_te < 0]  = 2
                correct = (np.argmax(np.stack([phn, pdn, pan], axis=1), axis=1) == y_class).astype(int)

                thr_prob = _pick_threshold_for_target_precision(
                    y_true=correct, p_cal=p_max,
                    target_precision=target_precision, min_preds=min_preds, default_threshold=0.45,
                )
                thr_pct = float(np.clip(_percent(thr_prob), min_thresh, max_thresh))
                _set_setting(conn, "conf_threshold:1X2", f"{thr_pct:.2f}")
                summary["thresholds"]["1X2"] = thr_pct

        else:
            logger.info("In-Play: not enough labeled data yet (have %d, need >= %d).", len(df_ip), min_rows)
            summary["trained"]["BTTS_YES"] = False

        # ───────── Prematch training (PRE_*) ─────────
        df_pre = load_prematch_data(conn)
        if not df_pre.empty and len(df_pre) >= min_rows:
            tr_mask, te_mask = time_order_split(df_pre, test_size=test_size)
            Xp_all = df_pre[PRE_FEATURES].values

            # PRE BTTS
            yb = df_pre["label_btts"].values.astype(int)
            ok, mets, p_b_cal = _train_binary_head(
                conn, Xp_all, yb, tr_mask, te_mask, PRE_FEATURES,
                model_key="PRE_BTTS_YES",
                threshold_key="conf_threshold:PRE BTTS",
                threshold_label="PRE BTTS",
                target_precision=target_precision,
                min_preds=min_preds,
                min_thresh_pct=min_thresh,
                max_thresh_pct=max_thresh,
                default_thr_prob=0.65,
                metrics_name="PRE_BTTS_YES",
            )
            summary["trained"]["PRE_BTTS_YES"] = ok
            if ok:
                summary["metrics"]["PRE_BTTS_YES"] = mets

            # PRE O/U
            totals = df_pre["final_goals_sum"].values.astype(int)
            for line in ou_lines:
                y_over = (totals > line).astype(int)
                name = f"PRE_OU_{_fmt_line(line)}"
                ok, mets, p_cal = _train_binary_head(
                    conn, Xp_all, y_over, tr_mask, te_mask, PRE_FEATURES,
                    model_key=name,
                    threshold_key=f"conf_threshold:PRE Over/Under {_fmt_line(line)}",
                    threshold_label=f"PRE Over/Under {_fmt_line(line)}",
                    target_precision=target_precision,
                    min_preds=min_preds,
                    min_thresh_pct=min_thresh,
                    max_thresh_pct=max_thresh,
                    default_thr_prob=0.65,
                    metrics_name=name,
                )
                summary["trained"][name] = ok
                if ok:
                    summary["metrics"][name] = mets

            # PRE 1X2 (Draw suppressed at serve but we still train HOME & AWAY heads)
            gd = df_pre["final_goals_diff"].values.astype(int)
            y_home = (gd > 0).astype(int)
            y_away = (gd < 0).astype(int)

            ok_h, mets_h, p_h = _train_binary_head(
                conn, Xp_all, y_home, tr_mask, te_mask, PRE_FEATURES,
                model_key="PRE_WLD_HOME",
                threshold_key=None, threshold_label=None,
                target_precision=target_precision, min_preds=min_preds,
                min_thresh_pct=min_thresh, max_thresh_pct=max_thresh,
                default_thr_prob=0.45, metrics_name="PRE_WLD_HOME",
            )
            ok_a, mets_a, p_a = _train_binary_head(
                conn, Xp_all, y_away, tr_mask, te_mask, PRE_FEATURES,
                model_key="PRE_WLD_AWAY",
                threshold_key=None, threshold_label=None,
                target_precision=target_precision, min_preds=min_preds,
                min_thresh_pct=min_thresh, max_thresh_pct=max_thresh,
                default_thr_prob=0.45, metrics_name="PRE_WLD_AWAY",
            )
            summary["trained"]["PRE_WLD_HOME"] = ok_h
            summary["trained"]["PRE_WLD_AWAY"] = ok_a
            if ok_h: summary["metrics"]["PRE_WLD_HOME"] = mets_h
            if ok_a: summary["metrics"]["PRE_WLD_AWAY"] = mets_a

            if ok_h and ok_a and (p_h is not None) and (p_a is not None):
                ps = np.clip(p_h, EPS, 1 - EPS) + np.clip(p_a, EPS, 1 - EPS)
                phn, pan = p_h / ps, p_a / ps
                p_max = np.maximum(phn, pan)

                # evaluate correctness ignoring draws (mask them out)
                gd_te = gd[te_mask]
                y_class = np.where(gd_te > 0, 0, np.where(gd_te < 0, 1, -1))
                mask = (y_class != -1)
                if mask.any():
                    correct = (np.argmax(np.stack([phn, pan], axis=1), axis=1)[mask] == y_class[mask]).astype(int)
                    thr_prob = _pick_threshold_for_target_precision(
                        y_true=correct, p_cal=p_max[mask],
                        target_precision=target_precision, min_preds=min_preds, default_threshold=0.45,
                    )
                    thr_pct = float(np.clip(_percent(thr_prob), min_thresh, max_thresh))
                    _set_setting(conn, "conf_threshold:PRE 1X2", f"{thr_pct:.2f}")
                    summary["thresholds"]["PRE 1X2"] = thr_pct
        else:
            logger.info("Prematch: not enough labeled data yet (have %d, need >= %d).", len(df_pre), min_rows)
            summary["trained"]["PRE_BTTS_YES"] = False

        # Bundle metrics snapshot for easy inspection
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
        }
        _set_setting(conn, "model_metrics_latest", json.dumps(metrics_bundle))
        return summary

    except Exception as e:
        logger.exception("Training failed: %s", e)
        return {"ok": False, "error": str(e)}
    finally:
        try:
            conn.close()
        except Exception:
            pass


# ─────────────────────── Small helpers for aliasing ─────────────────────── #

def get_setting_json(conn, key: str) -> Optional[dict]:
    try:
        df = _read_sql(conn, "SELECT value FROM settings WHERE key=%s", (key,))
        if df.empty:
            return None
        return json.loads(df.iloc[0]["value"])
    except Exception:
        return None


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
        min_minute=args.min_minute,
        test_size=args.test_size,
        min_rows=args.min_rows,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    _cli_main()
