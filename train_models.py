import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from datetime import datetime
import psycopg2

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import brier_score_loss, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# consistent, explicit feature order
FEATURES = [
    "minute","goals_h","goals_a","goals_sum","goals_diff",
    "xg_h","xg_a","xg_sum","xg_diff",
    "sot_h","sot_a","sot_sum",
    "cor_h","cor_a","cor_sum",
    "pos_h","pos_a","pos_diff",
    "red_h","red_a","red_sum"
]

OU_THRESHOLDS = [1.5, 2.5, 3.5, 4.5]

# ──────────────────────────────────────────────────────────────────────────────
# DB utils
def _normalize_db_url(db_url: str) -> str:
    if not db_url: return db_url
    if db_url.startswith("postgres://"):
        db_url = "postgresql://" + db_url[len("postgres://"):]
    if "sslmode=" not in db_url:
        db_url = db_url + ("&" if "?" in db_url else "?") + "sslmode=require"
    return db_url

def _connect(db_url: str):
    db_url = _normalize_db_url(db_url)
    conn = psycopg2.connect(db_url)
    conn.autocommit = True
    return conn

def _read_sql(conn, sql: str, params: Tuple = ()):
    return pd.read_sql_query(sql, conn, params=params)

def _exec(conn, sql: str, params: Tuple):
    with conn.cursor() as cur:
        cur.execute(sql, params)

def _get_old_model(conn, code: str) -> Optional[Dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute("SELECT value FROM settings WHERE key=%s", (f"model_v2:{code}",))
        row = cur.fetchone()
        return json.loads(row[0]) if row else None

# ──────────────────────────────────────────────────────────────────────────────
# Feature + Label Loading
def load_snapshots_with_labels(conn, min_minute: int = 15) -> pd.DataFrame:
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
            p = json.loads(row["payload"]) or {}
            stat = p.get("stat") or {}
            f = {
                "match_id": int(row["match_id"]),
                "created_ts": int(row["created_ts"]),
                "minute": float(p.get("minute", 0)),
                "goals_h": float(p.get("gh", 0)), "goals_a": float(p.get("ga", 0)),
                "xg_h": float(stat.get("xg_h", 0)), "xg_a": float(stat.get("xg_a", 0)),
                "sot_h": float(stat.get("sot_h", 0)), "sot_a": float(stat.get("sot_a", 0)),
                "cor_h": float(stat.get("cor_h", 0)), "cor_a": float(stat.get("cor_a", 0)),
                "pos_h": float(stat.get("pos_h", 0)), "pos_a": float(stat.get("pos_a", 0)),
                "red_h": float(stat.get("red_h", 0)), "red_a": float(stat.get("red_a", 0)),
            }
        except Exception:
            continue
        # derived
        f["goals_sum"] = f["goals_h"] + f["goals_a"]
        f["goals_diff"] = f["goals_h"] - f["goals_a"]
        f["xg_sum"] = f["xg_h"] + f["xg_a"]
        f["xg_diff"] = f["xg_h"] - f["xg_a"]
        f["sot_sum"] = f["sot_h"] + f["sot_a"]
        f["cor_sum"] = f["cor_h"] + f["cor_a"]
        f["pos_diff"] = f["pos_h"] - f["pos_a"]
        f["red_sum"] = f["red_h"] + f["red_a"]

        # labels
        gh_f = int(row["final_goals_h"] or 0)
        ga_f = int(row["final_goals_a"] or 0)
        total = gh_f + ga_f
        f["final_total"]     = total
        f["label_btts"]      = 1 if int(row["btts_yes"] or 0) == 1 else 0
        f["label_win_home"]  = 1 if gh_f > ga_f else 0
        f["label_draw"]      = 1 if gh_f == ga_f else 0
        f["label_win_away"]  = 1 if gh_f < ga_f else 0
        feats.append(f)

    if not feats: return pd.DataFrame()
    df = pd.DataFrame(feats)
    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["minute"] = df["minute"].clip(0, 120)
    df = df[df["minute"] >= min_minute].copy()
    return df

# ──────────────────────────────────────────────────────────────────────────────
# Model Training Helpers
def _fit_model(X: np.ndarray, y: np.ndarray, model_type: str = "lr"):
    if model_type == "lr":
        pipe = Pipeline([
            ("scale", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear"))
        ])
        pipe.fit(X, y)
        return pipe
    elif model_type == "rf":
        model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
        model.fit(X, y)
        return model
    elif model_type == "xgb":
        model = XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=(len(y)-sum(y))/max(1,sum(y)),
            use_label_encoder=False, eval_metric="logloss"
        )
        model.fit(X, y)
        return model
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

def _predict_proba(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:,1]
    if hasattr(model, "decision_function"):
        z = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-z))
    raise ValueError("Model has no probability interface")

def _to_json_model(model, feature_names: List[str], model_type: str) -> Dict[str, Any]:
    """
    Export weights for LR (like before), or just store type for tree models.
    """
    if model_type == "lr":
        clf: LogisticRegression = model.named_steps["clf"]
        scale: StandardScaler = model.named_steps["scale"]
        coef = clf.coef_.ravel()
        std = scale.scale_; mean = scale.mean_
        std = np.where(std == 0.0, 1.0, std)
        w_unscaled = coef / std
        b_unscaled = float(clf.intercept_.ravel()[0] - np.dot(coef, mean / std))
        weights = {name: float(w) for name, w in zip(feature_names, w_unscaled.tolist())}
        return {
            "type": "lr",
            "intercept": float(b_unscaled),
            "weights": weights,
            "calibration": {"method": "sigmoid", "a": 1.0, "b": 0.0}
        }
    else:
        # For RF/XGB → just store model type, keep in DB as "black box"
        return {
            "type": model_type,
            "storage": "not_implemented_yet"  # (future: pickle/onnx)
        }

# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-url", default=None)
    ap.add_argument("--model-type", default=os.getenv("MODEL_TYPE", "lr"), help="lr | rf | xgb")
    ap.add_argument("--min-minute", type=int, default=15)
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--min-rows", type=int, default=800)
    args = ap.parse_args()

    db_url = args.db_url or os.getenv("DATABASE_URL")
    if not db_url:
        print("DATABASE_URL required", flush=True); sys.exit(1)

    conn = _connect(db_url)
    try:
        df_all = load_snapshots_with_labels(conn, args.min_minute)
        if df_all.empty or len(df_all) < args.min_rows:
            print("Not enough data yet.", flush=True); sys.exit(0)

        df_all = df_all.sort_values("created_ts").reset_index(drop=True)
        n = len(df_all); k = int(n*(1-args.test_size))
        tr_df, te_df = df_all.iloc[:k], df_all.iloc[k:]

        heads: Dict[str, Any] = {}
        metrics: Dict[str, Any] = {}

        def train_head(y_name: str, code: str):
            y_tr = tr_df[y_name].astype(int).values
            y_te = te_df[y_name].astype(int).values
            if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2: return

            X_tr = tr_df[FEATURES].values; X_te = te_df[FEATURES].values
            model = _fit_model(X_tr, y_tr, args.model_type)
            p_te = _predict_proba(model, X_te)

            auc = roc_auc_score(y_te, p_te)
            brier = brier_score_loss(y_te, p_te)
            acc = accuracy_score(y_te, (p_te>=0.5).astype(int))

            new_json = _to_json_model(model, FEATURES, args.model_type)
            old = _get_old_model(conn, code)

            # rollback check
            if old:
                try:
                    prev_auc = float(old.get("metrics", {}).get("auc", 0))
                    if auc < prev_auc - 0.02:  # tolerate small fluctuation
                        print(f"⚠️ Rollback {code}: new AUC={auc:.3f} worse than old {prev_auc:.3f}")
                        return
                except: pass

            new_json["metrics"] = {"auc": auc, "brier": brier, "acc": acc}
            heads[code] = new_json
            metrics[code] = new_json["metrics"]

        # Train all heads
        for th in OU_THRESHOLDS:
            code = f"O{int(th*10):02d}"
            tr_df[code] = (tr_df["final_total"] >= th).astype(int)
            te_df[code] = (te_df["final_total"] >= th).astype(int)
            train_head(code, code)

        for code, col in [
            ("BTTS", "label_btts"),
            ("WIN_HOME", "label_win_home"),
            ("DRAW", "label_draw"),
            ("WIN_AWAY", "label_win_away"),
        ]:
            train_head(col, code)

        if not heads:
            print("No valid heads trained.", flush=True); sys.exit(0)

        # Save to DB
        for code, model_json in heads.items():
            _exec(conn,
                  "INSERT INTO settings(key,value) VALUES(%s,%s) "
                  "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
                  (f"model_v2:{code}", json.dumps(model_json)))

        audit = {
            "trained_at": datetime.utcnow().isoformat()+"Z",
            "model_type": args.model_type,
            "metrics": metrics
        }
        _exec(conn,
              "INSERT INTO settings(key,value) VALUES(%s,%s) "
              "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
              ("model_audit", json.dumps(audit)))

        print(f"✅ Saved {len(heads)} heads (model_type={args.model_type})", flush=True)
        sys.exit(0)

    except Exception as e:
        print(f"[TRAIN] exception: {e}", flush=True); sys.exit(1)
    finally:
        try: conn.close()
        except: pass

if __name__ == "__main__":
    main()
