# path: ./train_models.py
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

# why: consistent, explicit feature order for weight export
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
    if not db_url:
        return db_url
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

def _time_holdout_split(df: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # why: avoid leakage from random shuffle; hold out the most recent chunk by created_ts
    df = df.sort_values("created_ts").reset_index(drop=True)
    n = len(df); k = max(1, int(round(n * (1.0 - test_size))))
    return df.iloc[:k].copy(), df.iloc[k:].copy()

def _calibration_split(df: pd.DataFrame, frac: float = 0.20) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # why: dedicate a small slice of train for Platt; keeps test untouched
    n = len(df); k = max(1, int(round(n * (1.0 - frac))))
    return df.iloc[:k].copy(), df.iloc[k:].copy()

def _as_matrix(df: pd.DataFrame) -> np.ndarray:
    X = df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float).values
    return X

def _fit_lr_balanced(X: np.ndarray, y: np.ndarray) -> Pipeline:
    # why: scale improves convergence; liblinear for binary + balanced classes
    pipe = Pipeline([
        ("scale", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear"))
    ])
    pipe.fit(X, y)
    return pipe

def _decision_function(pipe: Pipeline, X: np.ndarray) -> np.ndarray:
    clf: LogisticRegression = pipe.named_steps["clf"]
    scale: StandardScaler = pipe.named_steps["scale"]
    Z = clf.decision_function(scale.transform(X))
    return Z

def _fit_platt(z_tr: np.ndarray, y_tr: np.ndarray) -> Tuple[float, float]:
    # logistic on logits; single feature
    z = z_tr.reshape(-1, 1)
    cal = LogisticRegression(max_iter=1000, solver="lbfgs")
    cal.fit(z, y_tr)
    a = float(cal.coef_.ravel()[0]); b = float(cal.intercept_.ravel()[0])
    return a, b

def _platt_proba(z: np.ndarray, a: float, b: float) -> np.ndarray:
    x = a * z + b
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))

def _pipe_to_v2(pipe: Pipeline, feature_names: List[str], cal: Optional[Tuple[float, float]]) -> Dict[str, Any]:
    clf: LogisticRegression = pipe.named_steps["clf"]
    scale: StandardScaler = pipe.named_steps["scale"]
    # export weights in input feature space (pre‑scaled)
    coef = clf.coef_.ravel()
    std = scale.scale_; mean = scale.mean_
    std = np.where(std == 0.0, 1.0, std)
    w_unscaled = coef / std
    b_unscaled = float(clf.intercept_.ravel()[0] - np.dot(coef, mean / std))
    weights = {name: float(w) for name, w in zip(feature_names, w_unscaled.tolist())}
    a, b = (cal if cal else (1.0, 0.0))
    return {
        "intercept": float(b_unscaled),
        "weights": weights,
        "calibration": {"method": "sigmoid", "a": float(a), "b": float(b)}
    }

def _head(code: str) -> str:
    return code

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

    if not feats:
        return pd.DataFrame()

    df = pd.DataFrame(feats)
    # cleanup
    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["minute"] = df["minute"].clip(0, 120)
    df = df[df["minute"] >= min_minute].copy()
    return df

def build_baselines(df: pd.DataFrame, heads_present: List[str]) -> Dict[str, Dict[str, float]]:
    # legacy baseline table (kept for back‑compat; not used by updated main)
    def bucket_minute(m):
        edges = [15, 30, 45, 60, 75, 90, 120]
        for i, e in enumerate(edges):
            if m <= e: return i
        return len(edges)
    def score_state(row):
        gs = int(row["goals_sum"]);  gd = int(row["goals_diff"])
        if gs >= 3: gs = 3
        d = "H" if gd > 0 else "A" if gd < 0 else "D"
        return f"gs{gs}_{d}"
    base = df.copy()
    base["min_b"] = base["minute"].apply(bucket_minute)
    base["state"] = base.apply(score_state, axis=1)
    def baseline_of(y: np.ndarray, gmin: pd.Series, gstate: pd.Series):
        out = {}
        tmp = pd.DataFrame({"y": y, "min_b": gmin, "state": gstate})
        grp = tmp.groupby(["min_b", "state"])
        cnt = grp["y"].count(); pos = grp["y"].sum()
        for (mb, st), n in cnt.items():
            if n >= 50:
                out[f"{int(mb)}|{str(st)}"] = float(pos[(mb, st)] / n)
        return out
    bl: Dict[str, Dict[str, float]] = {}
    for th in OU_THRESHOLDS:
        code = _ou_code(th)
        if code in heads_present:
            y = (base["final_total"].values >= th).astype(int)
            bl[code] = baseline_of(y, base["min_b"], base["state"])
    if "BTTS" in heads_present:
        bl["BTTS"] = baseline_of(base["label_btts"].values.astype(int), base["min_b"], base["state"])
    for (code, col) in [("WIN_HOME", "label_win_home"), ("DRAW", "label_draw"), ("WIN_AWAY", "label_win_away")]:
        if code in heads_present:
            bl[code] = baseline_of(base[col].values.astype(int), base["min_b"], base["state"])
    return bl

# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-url", default=None)
    ap.add_argument("--min-minute", dest="min_minute", type=int, default=15)
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--calib-frac", type=float, default=0.2)
    ap.add_argument("--min-rows", type=int, default=800)
    args = ap.parse_args()

    db_url = args.db_url or os.getenv("DATABASE_URL")
    if not db_url:
        print("DATABASE_URL required (pass --db-url or set env)", flush=True)
        sys.exit(1)

    conn = _connect(db_url)
    try:
        df_all = load_snapshots_with_labels(conn, args.min_minute)
        if df_all.empty:
            print("Not enough labeled data yet.", flush=True)
            sys.exit(0)

        if len(df_all) < args.min_rows:
            print(f"Need more data: {len(df_all)} < min-rows {args.min_rows}.", flush=True)
            sys.exit(0)

        # time-based holdout
        tr_df, te_df = _time_holdout_split(df_all, test_size=args.test_size)

        heads: Dict[str, Dict[str, Any]] = {}
        metrics_all: Dict[str, Dict[str, Any]] = {}

        def train_head(y_name: str, code: str):
            nonlocal heads, metrics_all
            y_tr = tr_df[y_name].astype(int).values
            y_te = te_df[y_name].astype(int).values
            if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                return
            tr_core, tr_cal = _calibration_split(tr_df, frac=args.calib_frac)
            X_core = _as_matrix(tr_core); y_core = tr_core[y_name].astype(int).values
            X_cal  = _as_matrix(tr_cal);  y_cal  = tr_cal[y_name].astype(int).values
            X_te   = _as_matrix(te_df)

            pipe = _fit_lr_balanced(X_core, y_core)
            z_cal = _decision_function(pipe, X_cal)
            a, b = _fit_platt(z_cal, y_cal)

            z_te = _decision_function(pipe, X_te)
            p_te = _platt_proba(z_te, a, b)

            try:
                auc = roc_auc_score(y_te, p_te)
            except Exception:
                auc = float("nan")
            brier = brier_score_loss(y_te, p_te)
            acc = accuracy_score(y_te, (p_te >= 0.5).astype(int))
            prev = float(y_tr.mean())

            metrics_all[code] = {
                "brier": float(brier), "acc": float(acc), "auc": float(auc),
                "n": int(len(y_te)), "prevalence": float(prev),
                "majority_acc": float(max(prev, 1 - prev)),
                "calibrated": True, "calibration_method": "sigmoid",
                "time_split": True
            }
            heads[code] = _pipe_to_v2(pipe, FEATURES, (a, b))

        # OU heads
        for th in OU_THRESHOLDS:
            code = _ou_code(th)
            tr_df[code] = (tr_df["final_total"] >= th).astype(int)
            te_df[code] = (te_df["final_total"] >= th).astype(int)
            train_head(code, code)

        # BTTS + 1X2
        for (code, col) in [
            ("BTTS", "label_btts"),
            ("WIN_HOME", "label_win_home"),
            ("DRAW", "label_draw"),
            ("WIN_AWAY", "label_win_away"),
        ]:
            train_head(code, col)

        if not heads:
            print("No heads trained (insufficient variation).", flush=True)
            sys.exit(0)

        # Persist models
        for code, v2 in heads.items():
            _exec(conn,
                  "INSERT INTO settings(key,value) VALUES(%s,%s) "
                  "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
                  (f"model_v2:{_head(code)}", json.dumps(v2)))

        if "BTTS" in heads:
            _exec(conn,
                  "INSERT INTO settings(key,value) VALUES(%s,%s) "
                  "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
                  ("model_v2:BTTS_YES", json.dumps(heads["BTTS"])))

        baselines = build_baselines(df_all, list(heads.keys()))
        _exec(conn,
              "INSERT INTO settings(key,value) VALUES(%s,%s) "
              "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
              ("policy:baselines_v1", json.dumps(baselines)))

        policy = {"global": {"top_k_per_match": 2, "min_quota": 1.50}}
        _exec(conn,
              "INSERT INTO settings(key,value) VALUES(%s,%s) "
              "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
              ("policy:heads_v1", json.dumps(policy)))

        audit = {
            "trained_at_utc": datetime.utcnow().isoformat() + "Z",
            "features": FEATURES,
            "metrics": metrics_all
        }
        _exec(conn,
              "INSERT INTO settings(key,value) VALUES(%s,%s) "
              "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
              ("model_coeffs", json.dumps(audit)))

        print("Saved model_v2:* (+ optional baselines/policy for back-compat) and model_coeffs.", flush=True)
        sys.exit(0)

    except Exception as e:
        print(f"[TRAIN] exception: {e}", flush=True)
        sys.exit(1)
    finally:
        conn.close()

        def train_head(y_name: str, code: str):
            nonlocal heads, metrics_all
            y_tr = tr_df[y_name].astype(int).values
            y_te = te_df[y_name].astype(int).values
            if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                return
            # calibration split on train
            tr_core, tr_cal = _calibration_split(tr_df, frac=args.calib_frac)
            X_core = _as_matrix(tr_core); y_core = tr_core[y_name].astype(int).values
            X_cal  = _as_matrix(tr_cal);  y_cal  = tr_cal[y_name].astype(int).values
            X_te   = _as_matrix(te_df)

            pipe = _fit_lr_balanced(X_core, y_core)
            z_cal = _decision_function(pipe, X_cal)
            a, b = _fit_platt(z_cal, y_cal)

            z_te = _decision_function(pipe, X_te)
            p_te = _platt_proba(z_te, a, b)

            try:
                auc = roc_auc_score(y_te, p_te)
            except Exception:
                auc = float("nan")
            brier = brier_score_loss(y_te, p_te)
            acc = accuracy_score(y_te, (p_te >= 0.5).astype(int))
            prev = float(y_tr.mean())

            metrics_all[code] = {
                "brier": float(brier), "acc": float(acc), "auc": float(auc),
                "n_test": int(len(y_te)), "prevalence": float(prev),
                "majority_acc": float(max(prev, 1 - prev)),
                "calibrated": True, "calibration_method": "sigmoid",
                "time_split": True
            }
            heads[code] = _pipe_to_v2(pipe, FEATURES, (a, b))

        # OU heads
        for th in OU_THRESHOLDS:
            code = _ou_code(th)
            tr_df[code] = (tr_df["final_total"] >= th).astype(int)
            te_df[code] = (te_df["final_total"] >= th).astype(int)
            train_head(code, code)
        # BTTS + 1X2
        for (code, col) in [
            ("BTTS", "label_btts"),
            ("WIN_HOME", "label_win_home"),
            ("DRAW", "label_draw"),
            ("WIN_AWAY", "label_win_away"),
        ]:
            train_head(code, col)

        if not heads:
            print("No heads trained (insufficient variation)."); return

        # Persist models
        for code, v2 in heads.items():
            _exec(conn,
                  "INSERT INTO settings(key,value) VALUES(%s,%s) "
                  "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
                  (f"model_v2:{_head(code)}", json.dumps(v2)))

        # Back‑compat alias for BTTS_YES
        if "BTTS" in heads:
            _exec(conn,
                  "INSERT INTO settings(key,value) VALUES(%s,%s) "
                  "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
                  ("model_v2:BTTS_YES", json.dumps(heads["BTTS"])))

        # Optional legacy policy/baselines (main.py doesn't rely on them anymore)
        baselines = build_baselines(df_all, list(heads.keys()))
        _exec(conn,
              "INSERT INTO settings(key,value) VALUES(%s,%s) "
              "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
              ("policy:baselines_v1", json.dumps(baselines)))
        policy = {
            "global": {"top_k_per_match": 2, "min_quota": 1.50}
        }
        _exec(conn,
              "INSERT INTO settings(key,value) VALUES(%s,%s) "
              "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
              ("policy:heads_v1", json.dumps(policy)))

        # Audit blob
        audit = {
            "trained_at_utc": datetime.utcnow().isoformat() + "Z",
            "features": FEATURES,
            "metrics": metrics_all
        }
        _exec(conn,
              "INSERT INTO settings(key,value) VALUES(%s,%s) "
              "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
              ("model_coeffs", json.dumps(audit)))

        print("Saved model_v2:* (+ optional baselines/policy for back‑compat) and model_coeffs.")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
