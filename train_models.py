# file: train_models.py
"""
Postgres-only training with Platt calibration + auto-thresholding.
Trains:
  • BTTS_YES (binary LR)
  • OU_{line} (binary LR for multiple lines)
  • WLD (1X2) via one-vs-rest LR: WLD_HOME, WLD_AWAY

Saves models to settings:
  model_latest:BTTS_YES
  model_latest:OU_{line}      (and O25 alias for 2.5)
  model_latest:WLD_HOME
  model_latest:WLD_AWAY
(also mirrored as model:* keys)

Writes decision thresholds (percent) to settings:
  conf_threshold:BTTS
  conf_threshold:Over/Under {line}
  conf_threshold:1X2
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

FEATURES: List[str] = [
    "minute",
    "goals_h", "goals_a", "goals_sum", "goals_diff",
    "xg_h", "xg_a", "xg_sum", "xg_diff",
    "sot_h", "sot_a", "sot_sum",
    "cor_h", "cor_a", "cor_sum",
    "pos_h", "pos_a", "pos_diff",
    "red_h", "red_a", "red_sum",
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

def load_data(conn, min_minute: int = 15) -> pd.DataFrame:
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

        # Derived features
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
        f["_ts"] = int(row["created_ts"] or 0)          # keep snapshot time for time-based split
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


# ─────────────────────── Model utils ─────────────────────── #

def fit_lr_safe(X: np.ndarray, y: np.ndarray) -> Optional[LogisticRegression]:
    if len(np.unique(y)) < 2:
        return None
    return LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear").fit(X, y)


def weights_dict(model: LogisticRegression, feature_names: List[str]) -> Dict[str, float]:
    return {name: float(w) for name, w in zip(feature_names, model.coef_.ravel().tolist())}


def _logit_vec(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_platt(y_true: np.ndarray, p_raw: np.ndarray) -> Tuple[float, float]:
    # 1D Platt using logistic regression on logits
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
    Search for smallest threshold that achieves >= target_precision with enough positives.
    Fallback to best F1, then best accuracy, then default.
    Returns a threshold in probability space (0..1).
    """
    y = y_true.astype(int)
    p = np.asarray(p_cal).astype(float)
    best_t = None
    candidates = _grid_thresholds(0.5, 0.95, 0.005)

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
        feasible.sort(key=lambda z: (z[0], -z[1]))  # smallest threshold, tie-break by precision
        best_t = feasible[0][0]

    if best_t is None:
        best_f1 = -1.0
        for t in candidates:
            pred = (p >= t).astype(int)
            n_pred = int(pred.sum())
            if n_pred < min_preds:
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
    Latest `test_size` fraction goes to test.
    """
    if "_ts" not in df.columns:
        # fallback to random if timestamps missing
        n = len(df)
        idx = np.arange(n)
        np.random.seed(42)
        np.random.shuffle(idx)
        cut = int((1 - test_size) * n)
        tr = np.zeros(n, dtype=bool); te = np.zeros(n, dtype=bool)
        tr[idx[:cut]] = True; te[idx[cut:]] = True
        return tr, te

    df_sorted = df.sort_values("_ts").reset_index(drop=True)
    n = len(df_sorted)
    cut = int(max(1, (1 - test_size) * n))
    train_idx = df_sorted.index[:cut].to_numpy()
    test_idx = df_sorted.index[cut:].to_numpy()
    tr = np.zeros(n, dtype=bool); te = np.zeros(n, dtype=bool)
    tr[train_idx] = True; te[test_idx] = True
    return tr, te


# ─────────────────────── Training ─────────────────────── #

def train_models(
    db_url: Optional[str] = None,
    min_minute: Optional[int] = None,
    test_size: Optional[float] = None,
    min_rows: int = 150,
) -> Dict[str, Any]:
    db_url = db_url or os.getenv("DATABASE_URL")
    engine, conn = _connect(db_url)

    min_minute = int(min_minute if min_minute is not None else os.getenv("TRAIN_MIN_MINUTE", 15))
    test_size = float(test_size if test_size is not None else os.getenv("TRAIN_TEST_SIZE", 0.25))

    # Markets to train
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

    # Threshold policy
    target_precision = float(os.getenv("TARGET_PRECISION", "0.60"))
    min_preds = int(os.getenv("THRESH_MIN_PREDICTIONS", "25"))
    min_thresh = float(os.getenv("MIN_THRESH", "55"))
    max_thresh = float(os.getenv("MAX_THRESH", "85"))

    summary: Dict[str, Any] = {"ok": True, "trained": {}, "metrics": {}, "features": FEATURES, "thresholds": {}}

    try:
        df = load_data(conn, min_minute)
        if df.empty or len(df) < int(min_rows):
            msg = f"Not enough labeled data yet (have {len(df)}, need >= {min_rows})."
            logger.info(msg)
            return {"ok": False, "reason": msg}

        # Time-based split masks (used across all markets to align test sets)
        tr_mask, te_mask = time_order_split(df, test_size=test_size)
        X_all = df[FEATURES].values

        # --- BTTS ---
        yb = df["label_btts"].values.astype(int)
        Xb_tr, Xb_te = X_all[tr_mask], X_all[te_mask]
        yb_tr, yb_te = yb[tr_mask], yb[te_mask]
        mb = fit_lr_safe(Xb_tr, yb_tr)
        if mb is not None:
            p_te_b_raw = mb.predict_proba(Xb_te)[:, 1]
            a_b, b_b = fit_platt(yb_te, p_te_b_raw)
            p_te_b_cal = 1.0 / (1.0 + np.exp(-(a_b * _logit_vec(p_te_b_raw) + b_b)))
            brier_b = float(brier_score_loss(yb_te, p_te_b_cal))
            acc_b = float(accuracy_score(yb_te, (p_te_b_cal >= 0.5).astype(int)))
            ll_b = float(log_loss(yb_te, p_te_b_cal, labels=[0, 1]))
            blob_btts = build_model_blob(mb, FEATURES, (a_b, b_b))
            for k in ("model_latest:BTTS_YES", "model:BTTS_YES"):
                _set_setting(conn, k, json.dumps(blob_btts))
            summary["trained"]["BTTS_YES"] = True
            summary["metrics"]["BTTS_YES"] = {"brier": brier_b, "acc": acc_b, "logloss": ll_b,
                                              "n_test": int(len(yb_te)), "prevalence": float(yb.mean())}
            thr_btts_prob = _pick_threshold_for_target_precision(
                y_true=yb_te, p_cal=p_te_b_cal,
                target_precision=target_precision, min_preds=min_preds, default_threshold=0.65,
            )
            thr_btts_pct = float(np.clip(_percent(thr_btts_prob), min_thresh, max_thresh))
            _set_setting(conn, "conf_threshold:BTTS", f"{thr_btts_pct:.2f}")
            summary["thresholds"]["BTTS"] = thr_btts_pct
        else:
            summary["trained"]["BTTS_YES"] = False

        # --- O/U lines ---
        total_goals = df["final_goals_sum"].values.astype(int)
        yo_all = {line: (total_goals > line).astype(int) for line in ou_lines}

        for line in ou_lines:
            name = f"OU_{_fmt_line(line)}"
            yo = yo_all[line]
            if yo.sum() == 0 or yo.sum() == len(yo):
                summary["trained"][name] = False
                continue

            Xo_tr, Xo_te = X_all[tr_mask], X_all[te_mask]
            yo_tr, yo_te = yo[tr_mask], yo[te_mask]
            mo = fit_lr_safe(Xo_tr, yo_tr)
            if mo is None:
                summary["trained"][name] = False
                continue

            p_te_o_raw = mo.predict_proba(Xo_te)[:, 1]
            a_o, b_o = fit_platt(yo_te, p_te_o_raw)
            p_te_o_cal = 1.0 / (1.0 + np.exp(-(a_o * _logit_vec(p_te_o_raw) + b_o)))
            brier_o = float(brier_score_loss(yo_te, p_te_o_cal))
            acc_o = float(accuracy_score(yo_te, (p_te_o_cal >= 0.5).astype(int)))
            ll_o = float(log_loss(yo_te, p_te_o_cal, labels=[0, 1]))
            blob_o = build_model_blob(mo, FEATURES, (a_o, b_o))
            for k in (f"model_latest:{name}", f"model:{name}"):
                _set_setting(conn, k, json.dumps(blob_o))
            if abs(line - 2.5) < 1e-6:
                for k in ("model_latest:O25", "model:O25"):
                    _set_setting(conn, k, json.dumps(blob_o))
            summary["trained"][name] = True
            summary["metrics"][name] = {"brier": brier_o, "acc": acc_o, "logloss": ll_o,
                                        "n_test": int(len(yo_te)), "prevalence": float(yo.mean())}
            market_name = f"Over/Under {_fmt_line(line)}"
            thr_ou_prob = _pick_threshold_for_target_precision(
                y_true=yo_te, p_cal=p_te_o_cal,
                target_precision=target_precision, min_preds=min_preds, default_threshold=0.65,
            )
            thr_ou_pct = float(np.clip(_percent(thr_ou_prob), min_thresh, max_thresh))
            _set_setting(conn, f"conf_threshold:{market_name}", f"{thr_ou_pct:.2f}")
            summary["thresholds"][market_name] = thr_ou_pct

        # --- 1X2 (WLD) via one-vs-rest LR ---
        # Labels from final result
        gd = df["final_goals_diff"].values.astype(int)
        y_home = (gd > 0).astype(int)
        y_away = (gd < 0).astype(int)

        def _fit_ovr(name: str, ybin: np.ndarray):
            if len(np.unique(ybin)) < 2:
                return None, None
            X_tr, X_te = X_all[tr_mask], X_all[te_mask]
            y_tr, y_te = ybin[tr_mask], ybin[te_mask]
            m = fit_lr_safe(X_tr, y_tr)
            if m is None:
                return None, None
            p_raw = m.predict_proba(X_te)[:, 1]
            a, b = fit_platt(y_te, p_raw)
            p_cal = 1.0 / (1.0 + np.exp(-(a * _logit_vec(p_raw) + b)))
            bri = float(brier_score_loss(y_te, p_cal))
            acc = float(accuracy_score(y_te, (p_cal >= 0.5).astype(int)))
            ll  = float(log_loss(y_te, p_cal, labels=[0, 1]))
            blob = build_model_blob(m, FEATURES, (a, b))
            for k in (f"model_latest:{name}", f"model:{name}"):
                _set_setting(conn, k, json.dumps(blob))
            summary["metrics"][name] = {"brier": bri, "acc": acc, "logloss": ll,
                                        "n_test": int(len(y_te)), "prevalence": float(ybin.mean())}
            summary["trained"][name] = True
            return (p_cal,)

        wld_models_ok = True
        res_h = _fit_ovr("WLD_HOME", y_home)
        res_a = _fit_ovr("WLD_AWAY", y_away)
        if not (res_h and res_d and res_a):
            wld_models_ok = False

        if wld_models_ok:
            p_h, = res_h
            p_d, = res_d
            p_a, = res_a
            # Renormalize calibrated OvR probs to sum to 1 per sample
            ps = np.clip(p_h, EPS, 1 - EPS) + np.clip(p_d, EPS, 1 - EPS) + np.clip(p_a, EPS, 1 - EPS)
            p_hn, p_dn, p_an = p_h / ps, p_d / ps, p_a / ps
            p_max = np.maximum.reduce([p_hn, p_dn, p_an])

            # True class index for the same test rows (time split keeps alignment)
            gd_te = gd[te_mask]
            y_class = np.zeros_like(gd_te, dtype=int)
            y_class[gd_te == 0] = 1
            y_class[gd_te < 0] = 2
            correct = (np.argmax(np.stack([p_hn, p_dn, p_an], axis=1), axis=1) == y_class).astype(int)

            thr_1x2_prob = _pick_threshold_for_target_precision(
                y_true=correct, p_cal=p_max,
                target_precision=target_precision, min_preds=min_preds, default_threshold=0.45,
            )
            thr_1x2_pct = float(np.clip(_percent(thr_1x2_prob), min_thresh, max_thresh))
            _set_setting(conn, "conf_threshold:1X2", f"{thr_1x2_pct:.2f}")
            summary["thresholds"]["1X2"] = thr_1x2_pct
        else:
            summary["trained"]["WLD_HOME"] = bool(res_h)
            summary["trained"]["WLD_DRAW"] = bool(res_d)
            summary["trained"]["WLD_AWAY"] = bool(res_a)

        # Bundle metrics snapshot
        metrics_bundle = {
            "trained_at_utc": pd.Timestamp.utcnow().isoformat(timespec="seconds") + "Z",
            **summary["metrics"],
            "features": FEATURES,
            "thresholds": summary["thresholds"],
            "target_precision": target_precision,
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


# ─────────────────────── CLI ─────────────────────── #

def _cli_main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-url", help="Postgres DSN (or use env DATABASE_URL)")
    ap.add_argument("--min-minute", dest="min_minute", type=int, default=int(os.getenv("TRAIN_MIN_MINUTE", 15)))
    ap.add_argument("--test-size", type=float, default=float(os.getenv("TRAIN_TEST_SIZE", 0.25)))
    ap.add_argument("--min-rows", type=int, default=150)
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
