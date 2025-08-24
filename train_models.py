import argparse
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, accuracy_score
from sklearn.model_selection import train_test_split
from datetime import datetime
import psycopg2


FEATURES = [
    "minute","goals_h","goals_a","goals_sum","goals_diff",
    "xg_h","xg_a","xg_sum","xg_diff",
    "sot_h","sot_a","sot_sum",
    "cor_h","cor_a","cor_sum",
    "pos_h","pos_a","pos_diff",
    "red_h","red_a","red_sum"
]

def _connect_pg(db_url: str):
    if not db_url:
        raise SystemExit("--db-url is required (Postgres DSN).")
    if "sslmode=" not in db_url:
        db_url = db_url + ("&" if "?" in db_url else "?") + "sslmode=require"
    conn = psycopg2.connect(db_url)
    conn.autocommit = True
    return conn

def _read_sql(conn, sql: str, params: Tuple = ()):
    return pd.read_sql_query(sql, conn, params=params)

def _exec(conn, sql: str, params: Tuple):
    with conn.cursor() as cur:
        cur.execute(sql, params)

def load_data(conn, min_minute: int = 15) -> pd.DataFrame:
    # take latest snapshot per match and join with results
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
    rows = _read_sql(conn, q)

    feats = []
    for _, row in rows.iterrows():
        try:
            p = json.loads(row["payload"])
        except Exception:
            continue
        stat = (p.get("stat") or {}) if isinstance(p, dict) else {}
        try:
            f = {
                "match_id": int(row["match_id"]),
                "minute": float(p.get("minute", 0)),
                "goals_h": float(p.get("gh", 0)), "goals_a": float(p.get("ga", 0)),
                "xg_h": float((stat or {}).get("xg_h", 0)), "xg_a": float((stat or {}).get("xg_a", 0)),
                "sot_h": float((stat or {}).get("sot_h", 0)), "sot_a": float((stat or {}).get("sot_a", 0)),
                "cor_h": float((stat or {}).get("cor_h", 0)), "cor_a": float((stat or {}).get("cor_a", 0)),
                "pos_h": float((stat or {}).get("pos_h", 0)), "pos_a": float((stat or {}).get("pos_a", 0)),
                "red_h": float((stat or {}).get("red_h", 0)), "red_a": float((stat or {}).get("red_a", 0)),
            }
        except Exception:
            continue

        # Deriveds
        f["goals_sum"] = f["goals_h"] + f["goals_a"]
        f["goals_diff"] = f["goals_h"] - f["goals_a"]
        f["xg_sum"] = f["xg_h"] + f["xg_a"]
        f["xg_diff"] = f["xg_h"] - f["xg_a"]
        f["sot_sum"] = f["sot_h"] + f["sot_a"]
        f["cor_sum"] = f["cor_h"] + f["cor_a"]
        f["pos_diff"] = f["pos_h"] - f["pos_a"]
        f["red_sum"] = f["red_h"] + f["red_a"]

        # Labels from final
        gh_f = int(row["final_goals_h"] or 0)
        ga_f = int(row["final_goals_a"] or 0)
        f["final_total"] = gh_f + ga_f
        f["label_btts"] = 1 if int(row["btts_yes"] or 0) == 1 else 0
        f["label_win_home"] = 1 if gh_f > ga_f else 0
        f["label_draw"] = 1 if gh_f == ga_f else 0
        f["label_win_away"] = 1 if ga_f > gh_f else 0

        feats.append(f)

    if not feats:
        return pd.DataFrame()

    df = pd.DataFrame(feats)
    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["minute"] = df["minute"].clip(0, 120)
    df = df[df["minute"] >= min_minute].copy()
    return df

def fit_lr_safe(X, y) -> Optional[LogisticRegression]:
    if len(np.unique(y)) < 2:
        return None
    return LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear"
    ).fit(X, y)

def to_coeffs(model, feature_names):
    return {
        "features": list(feature_names),
        "coef": model.coef_.ravel().tolist(),
        "intercept": float(model.intercept_.ravel()[0]),
    }

def _train_binary(head_name: str, X: np.ndarray, y: np.ndarray, features: List[str]) -> Optional[Dict[str, Any]]:
    lr = fit_lr_safe(X, y)
    if lr is None:
        print(f"[SKIP] {head_name}: single-class; need more balanced data.")
        return None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42,
        stratify=(y if y.sum() and y.sum() != len(y) else None)
    )
    lr2 = fit_lr_safe(X_tr, y_tr)
    if lr2 is None:
        return None
    p_te = lr2.predict_proba(X_te)[:, 1]
    return {
        "model": lr2,
        "metrics": {
            "brier": float(brier_score_loss(y_te, p_te)),
            "acc": float(accuracy_score(y_te, (p_te >= 0.5).astype(int))),
            "n": int(len(y_te)),
            "prevalence": float(y.mean())
        },
        "blob": to_coeffs(lr2, features)
    }

def _ou_code(th: float) -> str:
    # 2.5 -> "O25", 0.5 -> "O05"
    v = int(round(th * 10))
    return f"O{v:02d}"

def _get_setting_pg(conn, key: str) -> Optional[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT value FROM settings WHERE key=%s", (key,))
        row = cur.fetchone()
        return row[0] if row else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-url", required=True, help="Postgres DSN (required).")
    ap.add_argument("--min-minute", dest="min_minute", type=int, default=15)
    ap.add_argument("--ou-thresholds", type=str, default="0.5,1.5,2.5,3.5,4.5",
                    help="Comma-separated: 0.5,1.5,2.5,3.5,4.5")
    ap.add_argument("--min-rows", type=int, default=150)
    args = ap.parse_args()

    conn = _connect_pg(args.db_url)

    try:
        thresholds = [float(s) for s in args.ou_thresholds.split(",") if s.strip()]
        df = load_data(conn, args.min_minute)
        if df.empty:
            print("Not enough labeled data yet.")
            return

        n = len(df)
        if n < args.min_rows:
            print(f"Need more data: {n} < min-rows {args.min_rows}.")
            return

        X = df[FEATURES].values

        # Train heads
        models: Dict[str, Dict[str, Any]] = {}
        metrics_summary: Dict[str, Any] = {}

        # Over/Under heads
        for th in thresholds:
            y = (df["final_total"].values >= th).astype(int)
            code = _ou_code(th)  # e.g., O25
            res = _train_binary(f"OU {th}", X, y, FEATURES)
            if res:
                models[code] = res
                metrics_summary[code] = res["metrics"]

        # BTTS head
        y_btts = df["label_btts"].values.astype(int)
        res_btts = _train_binary("BTTS", X, y_btts, FEATURES)
        if res_btts:
            models["BTTS"] = res_btts
            metrics_summary["BTTS"] = res_btts["metrics"]

        # 1X2 heads (as three binaries)
        for code, col in [("WIN_HOME","label_win_home"), ("DRAW","label_draw"), ("WIN_AWAY","label_win_away")]:
            y = df[col].values.astype(int)
            res = _train_binary(code, X, y, FEATURES)
            if res:
                models[code] = res
                metrics_summary[code] = res["metrics"]

        if not models:
            print("No models trained (all heads skipped).")
            return

        trained_at = datetime.utcnow().isoformat() + "Z"

        # Save all as model_v2:<CODE>
        for code, res in models.items():
            blob = res["blob"]
            v2 = {
                "intercept": float(blob["intercept"]),
                "weights": {feat: float(w) for feat, w in zip(blob["features"], blob["coef"])},
                "model_type": "logreg_v2",
                "feature_order": list(blob["features"]),
                "calibration": {"method": "sigmoid", "a": 1.0, "b": 0.0},
                "trained_at_utc": trained_at,
            }
            _exec(conn,
                  "INSERT INTO settings(key,value) VALUES(%s,%s) "
                  "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
                  (f"model_v2:{code}", json.dumps(v2)))

        # Backward-compat alias for BTTS
        if "BTTS" in models:
            val = _get_setting_pg(conn, "model_v2:BTTS") or ""
            if val:
                _exec(conn,
                      "INSERT INTO settings(key,value) VALUES(%s,%s) "
                      "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
                      ("model_v2:BTTS_YES", val))

        # Audit blob
        audit = {
            "trained_at_utc": trained_at,
            "metrics": metrics_summary,
            "features": FEATURES,
            "ou_thresholds": thresholds,
            "heads": list(models.keys())
        }
        _exec(conn,
              "INSERT INTO settings(key,value) VALUES(%s,%s) "
              "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
              ("model_coeffs", json.dumps(audit)))

        print(f"Saved {len(models)} models:", ", ".join(sorted(models.keys())))
    finally:
        try:
            conn.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
