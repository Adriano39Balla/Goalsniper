import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import psycopg2

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
def _normalize_db_url(db_url: str) -> str:
    if not db_url: return db_url
    if db_url.startswith("postgres://"):
        db_url = "postgresql://" + db_url[len("postgres://"):]
    if "sslmode=" not in db_url:
        db_url = db_url + ("&" if "?" in db_url else "?") + "sslmode=require"
    return db_url

def _connect(db_url: str):
    conn = psycopg2.connect(_normalize_db_url(db_url))
    conn.autocommit = True
    return conn

def _read_sql(conn, sql: str, params: Tuple = ()):
    return pd.read_sql_query(sql, conn, params=params)

def _exec(conn, sql: str, params: Tuple):
    with conn.cursor() as cur: cur.execute(sql, params)

def _time_holdout_split(df: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("created_ts").reset_index(drop=True)
    n = len(df); k = max(1, int(round(n * (1.0 - test_size))))
    return df.iloc[:k].copy(), df.iloc[k:].copy()

def _calibration_split(df: pd.DataFrame, frac: float = 0.20) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df); k = max(1, int(round(n * (1.0 - frac))))
    return df.iloc[:k].copy(), df.iloc[k:].copy()

def _as_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float).values

def _fit_lr_balanced(X: np.ndarray, y: np.ndarray) -> Pipeline:
    pipe = Pipeline([
        ("scale", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear"))
    ])
    pipe.fit(X, y)
    return pipe

def _decision_function(pipe: Pipeline, X: np.ndarray) -> np.ndarray:
    clf: LogisticRegression = pipe.named_steps["clf"]
    scale: StandardScaler = pipe.named_steps["scale"]
    return clf.decision_function(scale.transform(X))

def _fit_platt(z_tr: np.ndarray, y_tr: np.ndarray) -> Tuple[float, float]:
    z = z_tr.reshape(-1, 1)
    cal = LogisticRegression(max_iter=1000, solver="lbfgs")
    cal.fit(z, y_tr)
    return float(cal.coef_.ravel()[0]), float(cal.intercept_.ravel()[0])

def _platt_proba(z: np.ndarray, a: float, b: float) -> np.ndarray:
    x = np.clip(a * z + b, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))

def _pipe_to_v2(pipe: Pipeline, feature_names: List[str], cal: Optional[Tuple[float, float]]) -> Dict[str, Any]:
    clf: LogisticRegression = pipe.named_steps["clf"]
    scale: StandardScaler = pipe.named_steps["scale"]
    coef = clf.coef_.ravel()
    std = np.where(scale.scale_ == 0.0, 1.0, scale.scale_)
    w_unscaled = coef / std
    b_unscaled = float(clf.intercept_.ravel()[0] - np.dot(coef, scale.mean_ / std))
    weights = {name: float(w) for name, w in zip(feature_names, w_unscaled.tolist())}
    a, b = (cal if cal else (1.0, 0.0))
    return {"intercept": float(b_unscaled),
            "weights": weights,
            "calibration": {"method": "sigmoid", "a": float(a), "b": float(b)}}

def _ou_code(th: float) -> str:
    return f"O{int(round(th * 10)):02d}"

# ──────────────────────────────────────────────────────────────────────────────
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
            f = {"match_id": int(row["match_id"]),
                 "created_ts": int(row["created_ts"]),
                 "minute": float(p.get("minute", 0)),
                 "goals_h": float(p.get("gh", 0)), "goals_a": float(p.get("ga", 0)),
                 "xg_h": float(stat.get("xg_h", 0)), "xg_a": float(stat.get("xg_a", 0)),
                 "sot_h": float(stat.get("sot_h", 0)), "sot_a": float(stat.get("sot_a", 0)),
                 "cor_h": float(stat.get("cor_h", 0)), "cor_a": float(stat.get("cor_a", 0)),
                 "pos_h": float(stat.get("pos_h", 0)), "pos_a": float(stat.get("pos_a", 0)),
                 "red_h": float(stat.get("red_h", 0)), "red_a": float(stat.get("red_a", 0))}
        except Exception:
            continue
        f["goals_sum"] = f["goals_h"] + f["goals_a"]; f["goals_diff"] = f["goals_h"] - f["goals_a"]
        f["xg_sum"] = f["xg_h"] + f["xg_a"]; f["xg_diff"] = f["xg_h"] - f["xg_a"]
        f["sot_sum"] = f["sot_h"] + f["sot_a"]; f["cor_sum"] = f["cor_h"] + f["cor_a"]
        f["pos_diff"] = f["pos_h"] - f["pos_a"]; f["red_sum"] = f["red_h"] + f["red_a"]
        gh_f, ga_f = int(row["final_goals_h"] or 0), int(row["final_goals_a"] or 0)
        f["final_total"] = gh_f + ga_f
        f["label_btts"] = 1 if int(row["btts_yes"] or 0) == 1 else 0
        f["label_win_home"] = 1 if gh_f > ga_f else 0
        f["label_draw"] = 1 if gh_f == ga_f else 0
        f["label_win_away"] = 1 if gh_f < ga_f else 0
        feats.append(f)
    if not feats: return pd.DataFrame()
    df = pd.DataFrame(feats)
    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["minute"] = df["minute"].clip(0, 120)
    return df[df["minute"] >= min_minute].copy()

# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-url", default=None)
    ap.add_argument("--min-minute", type=int, default=15)
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--calib-frac", type=float, default=0.2)
    ap.add_argument("--min-rows", type=int, default=800)
    args = ap.parse_args()

    db_url = args.db_url or os.getenv("DATABASE_URL")
    if not db_url: sys.exit("DATABASE_URL required")

    conn = _connect(db_url)
    try:
        df_all = load_snapshots_with_labels(conn, args.min_minute)
        if df_all.empty or len(df_all) < args.min_rows:
            sys.exit("Not enough labeled data yet.")

        tr_df, te_df = _time_holdout_split(df_all, test_size=args.test_size)
        heads: Dict[str, Dict[str, Any]] = {}; metrics_all: Dict[str, Dict[str, Any]] = {}

        def train_head(y_name: str, code: str):
            nonlocal heads, metrics_all
            y_tr, y_te = tr_df[y_name].astype(int).values, te_df[y_name].astype(int).values
            if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2: return
            tr_core, tr_cal = _calibration_split(tr_df, frac=args.calib_frac)
            pipe = _fit_lr_balanced(_as_matrix(tr_core), tr_core[y_name].astype(int).values)
            a, b = _fit_platt(_decision_function(pipe, _as_matrix(tr_cal)), tr_cal[y_name].astype(int).values)
            z_te = _decision_function(pipe, _as_matrix(te_df))
            p_te = _platt_proba(z_te, a, b)
            metrics_all[code] = {"brier": float(brier_score_loss(y_te, p_te)),
                                 "acc": float(accuracy_score(y_te, (p_te >= 0.5).astype(int))),
                                 "auc": float(roc_auc_score(y_te, p_te)),
                                 "n": int(len(y_te))}
            heads[code] = _pipe_to_v2(pipe, FEATURES, (a, b))

        for th in OU_THRESHOLDS:
            code = _ou_code(th)
            tr_df[code] = (tr_df["final_total"] >= th).astype(int)
            te_df[code] = (te_df["final_total"] >= th).astype(int)
            train_head(code, code)
        for (code, col) in [("BTTS", "label_btts"), ("WIN_HOME", "label_win_home"),
                            ("DRAW", "label_draw"), ("WIN_AWAY", "label_win_away")]:
            train_head(col, code)

        if not heads: sys.exit("No heads trained.")
        for code, v2 in heads.items():
            _exec(conn, "INSERT INTO settings(key,value) VALUES(%s,%s) "
                        "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
                        (f"model_v2:{code}", json.dumps(v2)))
        if "BTTS" in heads:
            _exec(conn, "INSERT INTO settings(key,value) VALUES(%s,%s) "
                        "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
                        ("model_v2:BTTS_YES", json.dumps(heads["BTTS"])))
        audit = {"trained_at_utc": datetime.utcnow().isoformat() + "Z",
                 "features": FEATURES, "metrics": metrics_all}
        _exec(conn, "INSERT INTO settings(key,value) VALUES(%s,%s) "
                    "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
                    ("model_coeffs", json.dumps(audit)))
        print("Saved models and audit.", flush=True)
    finally:
        conn.close()

if __name__ == "__main__":
    main()
