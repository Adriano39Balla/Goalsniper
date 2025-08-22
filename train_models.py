#!/usr/bin/env python3
"""
train_models.py  (Postgres / Supabase)

Nightly trainer for Robi:
- Loads latest snapshots + final results from Postgres (Supabase)
- Engineers features consistently with main.py
- Trains two logistic models: Over 2.5 (O25) and BTTS: Yes
- Optional Platt calibration and/or K-fold CV metrics
- Persists compact, framework-free coefficients to settings.model_coeffs
- (Optionally) persists metrics to settings.model_metrics_latest
- (Optionally) exports the same JSON blob to a file

Exit code:
  0  success
  2  not enough labeled data
  3  below min-rows threshold
"""

import argparse
import json
import os
from datetime import datetime
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import psycopg2

from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold

RANDOM_STATE = 42

FEATURES = [
    "minute","goals_h","goals_a","goals_sum","goals_diff",
    "xg_h","xg_a","xg_sum","xg_diff",
    "sot_h","sot_a","sot_sum",
    "cor_h","cor_a","cor_sum",
    "pos_h","pos_a","pos_diff",
    "red_h","red_a","red_sum"
]

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _pg_connect(dsn: str):
    if "sslmode=" not in dsn:
        dsn = dsn + ("&" if "?" in dsn else "?") + "sslmode=require"
    return psycopg2.connect(dsn)

def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)

def _df_from_rows(rows: pd.DataFrame, min_minute: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build feature DataFrames for O2.5 and BTTS labels."""
    if rows.empty:
        return pd.DataFrame(), pd.DataFrame()

    feats = []
    for _, row in rows.iterrows():
        try:
            p = json.loads(row["payload"])
        except Exception:
            continue

        stat = (p.get("stat") or {})
        f = {
            "match_id": int(row["match_id"]),
            "minute": _safe_float(p.get("minute", 0)),
            "goals_h": _safe_float(p.get("gh", 0)),
            "goals_a": _safe_float(p.get("ga", 0)),
            "xg_h": _safe_float(stat.get("xg_h", 0)), "xg_a": _safe_float(stat.get("xg_a", 0)),
            "sot_h": _safe_float(stat.get("sot_h", 0)), "sot_a": _safe_float(stat.get("sot_a", 0)),
            "cor_h": _safe_float(stat.get("cor_h", 0)), "cor_a": _safe_float(stat.get("cor_a", 0)),
            "pos_h": _safe_float(stat.get("pos_h", 0)), "pos_a": _safe_float(stat.get("pos_a", 0)),
            "red_h": _safe_float(stat.get("red_h", 0)), "red_a": _safe_float(stat.get("red_a", 0)),
        }
        f["goals_sum"] = f["goals_h"] + f["goals_a"]
        f["goals_diff"] = f["goals_h"] - f["goals_a"]
        f["xg_sum"] = f["xg_h"] + f["xg_a"]
        f["xg_diff"] = f["xg_h"] - f["xg_a"]
        f["sot_sum"] = f["sot_h"] + f["sot_a"]
        f["cor_sum"] = f["cor_h"] + f["cor_a"]
        f["pos_diff"] = f["pos_h"] - f["pos_a"]
        f["red_sum"] = f["red_h"] + f["red_a"]

        gh_f = int(row.get("final_goals_h") or 0)
        ga_f = int(row.get("final_goals_a") or 0)
        btts_yes = 1 if int(row.get("btts_yes") or 0) == 1 else 0

        f["label_o25"] = 1 if (gh_f + ga_f) >= 3 else 0
        f["label_btts"] = btts_yes

        feats.append(f)

    if not feats:
        return pd.DataFrame(), pd.DataFrame()

    df = pd.DataFrame(feats)

    # Hygiene
    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["minute"] = df["minute"].clip(0, 120)

    df = df[df["minute"] >= float(min_minute)].copy()
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    return df[FEATURES + ["label_o25"]], df[FEATURES + ["label_btts"]]

def load_data_pg(dsn: str, min_minute: int = 15) -> Tuple[pd.DataFrame, pd.DataFrame]:
    q = """
    WITH latest AS (
      SELECT match_id, MAX(created_ts) AS ts
      FROM tip_snapshots GROUP BY match_id
    )
    SELECT l.match_id, s.created_ts, s.payload,
           r.final_goals_h, r.final_goals_a, r.btts_yes
    FROM latest l
    JOIN tip_snapshots s ON s.match_id=l.match_id AND s.created_ts=l.ts
    JOIN match_results r ON r.match_id=l.match_id
    """
    with _pg_connect(dsn) as con:
        rows = pd.read_sql_query(q, con)
    return _df_from_rows(rows, min_minute)

def fit_lr_or_none(X: np.ndarray, y: np.ndarray) -> LogisticRegression | None:
    if len(np.unique(y)) < 2:
        return None
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=RANDOM_STATE
    )
    return model.fit(X, y)

def to_coeffs(model: LogisticRegression, feature_names) -> Dict[str, Any]:
    coef = model.coef_.ravel().tolist()
    intercept = float(model.intercept_.ravel()[0])
    return {"features": list(feature_names), "coef": coef, "intercept": intercept}

def maybe_calibrate(base_model: LogisticRegression, Xtr, ytr, method: str | None):
    if method is None:
        return base_model, False
    if len(np.unique(ytr)) < 2:
        return base_model, False
    cal = CalibratedClassifierCV(base_model, method=method, cv=3)
    cal.fit(Xtr, ytr)
    return cal, True

def _majority_acc(y: np.ndarray) -> float:
    p = y.mean()
    return max(p, 1.0 - p)

def train_and_eval(
    df: pd.DataFrame,
    label_col: str,
    test_size: float,
    calibrate: str | None,
    cv_folds: int = 0
) -> Tuple[Any, Dict[str, Any]]:
    X = df[FEATURES].values
    y = df[label_col].values.astype(int)

    has_pos = y.sum() > 0
    has_neg = (len(y) - y.sum()) > 0
    strat = y if (has_pos and has_neg) else None

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=strat
    )

    base = fit_lr_or_none(Xtr, ytr)
    if base is None:
        raise ValueError(f"{label_col} split became single-class; need more balanced data.")

    clf, calibrated = maybe_calibrate(base, Xtr, ytr, calibrate)

    try:
        pte = clf.predict_proba(Xte)[:, 1]
    except NotFittedError:
        clf.fit(Xtr, ytr)
        pte = clf.predict_proba(Xte)[:, 1]

    metrics = {
        "brier": float(brier_score_loss(yte, pte)),
        "acc": float(accuracy_score(yte, (pte >= 0.5).astype(int))),
        "auc": float(roc_auc_score(yte, pte)) if (has_pos and has_neg) else None,
        "n": int(len(yte)),
        "prevalence": float(y.mean()),
        "majority_acc": float(_majority_acc(y)),
        "calibrated": bool(calibrated),
        "calibration_method": calibrate if calibrated else None,
    }

    if cv_folds and cv_folds > 1 and (has_pos and has_neg):
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
        briers, accs, aucs = [], [], []
        for tr, te in skf.split(X, y):
            m = fit_lr_or_none(X[tr], y[tr])
            if m is None:
                continue
            pp = m.predict_proba(X[te])[:, 1]
            briers.append(brier_score_loss(y[te], pp))
            accs.append(accuracy_score(y[te], (pp >= 0.5).astype(int)))
            aucs.append(roc_auc_score(y[te], pp))
        if briers:
            metrics["cv"] = {
                "folds": cv_folds,
                "brier_mean": float(np.mean(briers)),
                "brier_std": float(np.std(briers)),
                "acc_mean": float(np.mean(accs)),
                "acc_std": float(np.std(accs)),
                "auc_mean": float(np.mean(aucs)),
                "auc_std": float(np.std(aucs)),
            }

    serializable = to_coeffs(base, FEATURES)
    return clf, {"metrics": metrics, "serializable": serializable}

def save_blob_to_settings_pg(dsn: str, key: str, blob: Dict[str, Any]) -> None:
    with _pg_connect(dsn) as con:
        cur = con.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        cur.execute("""
            INSERT INTO settings(key, value)
            VALUES (%s, %s)
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
        """, (key, json.dumps(blob)))
        cur.close()

def _print_top_coeffs(name: str, coeffs: Dict[str, Any], k: int = 8) -> None:
    feats = coeffs["features"]
    vals = coeffs["coef"]
    pairs = sorted(zip(feats, vals), key=lambda t: abs(t[1]), reverse=True)[:k]
    pretty = ", ".join([f"{f}={v:+.3f}" for f, v in pairs])
    print(f"[{name}] top | {pretty} | intercept={coeffs['intercept']:+.3f}")

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-url", default=os.getenv("DATABASE_URL"), help="Postgres DSN (Supabase)")
    ap.add_argument("--min-minute", dest="min_minute", type=int, default=15)
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--min-rows", type=int, default=150)
    ap.add_argument("--calibrate", choices=["sigmoid", "isotonic", "none"], default="sigmoid")
    ap.add_argument("--cv", type=int, default=0, help="If >1, run Stratified K-fold CV (report only).")
    ap.add_argument("--export-path", default=None, help="Optional path to write model_coeffs JSON")
    ap.add_argument("--store-metrics", type=int, default=1, help="Also store latest metrics in settings.model_metrics_latest")
    args = ap.parse_args()

    if not args.db_url:
        print("DATABASE_URL is required (Supabase Postgres).")
        raise SystemExit(2)

    calibrate = None if args.calibrate == "none" else args.calibrate

    # Load
    df_o25, df_btts = load_data_pg(args.db_url, args.min_minute)
    if df_o25.empty or df_btts.empty:
        print("Not enough labeled data yet.")
        raise SystemExit(2)

    n_o25, n_btts = len(df_o25), len(df_btts)
    n = min(n_o25, n_btts)
    print(f"Samples O2.5={n_o25} | BTTS={n_btts}")
    print(f"O2.5 positive rate={df_o25['label_o25'].mean():.3f} | "
          f"BTTS positive rate={df_btts['label_btts'].mean():.3f}")
    if n < args.min_rows:
        print(f"Need more data: {n} < min-rows {args.min_rows}.")
        raise SystemExit(3)

    # Train + evaluate
    _, o25 = train_and_eval(df_o25, "label_o25", args.test_size, calibrate, args.cv)
    _, btts = train_and_eval(df_btts, "label_btts", args.test_size, calibrate, args.cv)

    trained_at = datetime.utcnow().isoformat() + "Z"
    blob = {
        "O25": o25["serializable"],
        "BTTS_YES": btts["serializable"],
        "trained_at_utc": trained_at,
        "model_version": f"lr@{trained_at}",
        "features": FEATURES,
        "metrics": {"o25": o25["metrics"], "btts": btts["metrics"]},
        "training_counts": {"o25": int(n_o25), "btts": int(n_btts), "min_used": int(n)},
        "hyperparams": {
            "min_minute": int(args.min_minute),
            "test_size": float(args.test_size),
            "class_weight": "balanced",
            "solver": "liblinear",
            "random_state": RANDOM_STATE,
            "calibration": calibrate if calibrate else None,
            "cv_folds": int(args.cv) if args.cv else 0,
        }
    }

    # Persist in Postgres settings
    save_blob_to_settings_pg(args.db_url, "model_coeffs", blob)
    print("Saved model_coeffs in settings.")
    _print_top_coeffs("O25", blob["O25"])
    _print_top_coeffs("BTTS_YES", blob["BTTS_YES"])

    if args.export_path:
        with open(args.export_path, "w", encoding="utf-8") as f:
            json.dump(blob, f, ensure_ascii=False, indent=2)
        print(f"Exported model_coeffs to {os.path.abspath(args.export_path)}")

    if args.store_metrics:
        metrics_only = {
            "trained_at_utc": trained_at,
            "metrics": blob["metrics"],
            "training_counts": blob["training_counts"],
            "hyperparams": blob["hyperparams"],
        }
        save_blob_to_settings_pg(args.db_url, "model_metrics_latest", metrics_only)
        print("Saved model_metrics_latest in settings.")

    print(json.dumps({
        "ok": True,
        "trained_at_utc": trained_at,
        "counts": {"o25": n_o25, "btts": n_btts},
        "metrics": {"o25": o25["metrics"], "btts": btts["metrics"]}
    }, ensure_ascii=False))

if __name__ == "__main__":
    main()
