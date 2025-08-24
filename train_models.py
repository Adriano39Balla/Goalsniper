import argparse, json, psycopg2, time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────────────────────────────────────
FEATURES = [
    "minute","goals_h","goals_a","goals_sum","goals_diff",
    "xg_h","xg_a","xg_sum","xg_diff",
    "sot_h","sot_a","sot_sum",
    "cor_h","cor_a","cor_sum",
    "pos_h","pos_a","pos_diff",
    "red_h","red_a","red_sum"
]

OU_THRESHOLDS = [1.5, 2.5, 3.5, 4.5]   # explicitly drop 0.5

# ──────────────────────────────────────────────────────────────────────────────
def _connect(db_url: str):
    if "sslmode=" not in db_url:
        db_url = db_url + ("&" if "?" in db_url else "?") + "sslmode=require"
    conn = psycopg2.connect(db_url)
    conn.autocommit = True
    return conn

def _read_sql(conn, sql: str, params: Tuple=()):
    return pd.read_sql_query(sql, conn, params=params)

def _exec(conn, sql: str, params: Tuple):
    with conn.cursor() as cur:
        cur.execute(sql, params)

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
        except Exception:
            continue
        stat = p.get("stat") or {}
        try:
            f = {
                "match_id": int(row["match_id"]),
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

        # Deriveds
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
        total = gh_f + ga_f
        f["final_total"] = total
        f["label_btts"] = 1 if int(row["btts_yes"] or 0) == 1 else 0
        f["label_win_home"] = 1 if gh_f > ga_f else 0
        f["label_draw"]     = 1 if gh_f == ga_f else 0
        f["label_win_away"] = 1 if gh_f < ga_f else 0

        feats.append(f)

    if not feats:
        return pd.DataFrame()

    df = pd.DataFrame(feats)
    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["minute"] = df["minute"].clip(0, 120)
    df = df[df["minute"] >= min_minute].copy()
    return df

# ──────────────────────────────────────────────────────────────────────────────
def fit_lr(X, y) -> Optional[LogisticRegression]:
    if len(np.unique(y)) < 2: return None
    return LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear").fit(X, y)

def logits_from_model(m: LogisticRegression, X: np.ndarray) -> np.ndarray:
    # decision_function is linear predictor
    z = m.decision_function(X)
    return z

def fit_platt(z_tr: np.ndarray, y_tr: np.ndarray) -> Tuple[float,float]:
    # logistic over logits: p = 1 / (1 + exp(-(a*z + b)))
    z_tr = z_tr.reshape(-1,1)
    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    clf.fit(z_tr, y_tr)
    a = float(clf.coef_.ravel()[0])
    b = float(clf.intercept_.ravel()[0])
    return a, b

def proba_with_platt(z: np.ndarray, a: float, b: float) -> np.ndarray:
    x = a * z + b
    out = 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))
    return out

def to_v2(m: LogisticRegression, feature_names: List[str], cal: Optional[Tuple[float,float]]) -> Dict[str,Any]:
    weights = dict(zip(feature_names, m.coef_.ravel().tolist()))
    blob = {
        "intercept": float(m.intercept_.ravel()[0]),
        "weights": {k: float(v) for k,v in weights.items()},
        "calibration": {"method": "sigmoid", "a": float(cal[0]) if cal else 1.0, "b": float(cal[1]) if cal else 0.0},
    }
    return blob

def head_code_for_ou(th: float) -> str:
    return f"O{int(round(th*10)):02d}"  # 1.5 -> "O15"

# ──────────────────────────────────────────────────────────────────────────────
def build_baselines(df: pd.DataFrame, heads_present: List[str]) -> Dict[str, Dict[str, float]]:
    def bucket_minute(m):
        edges = [15,30,45,60,75,90,120]
        for i,e in enumerate(edges):
            if m <= e: return i
        return len(edges)

    def score_state(row):
        gs = int(row["goals_sum"])
        gd = int(row["goals_diff"])
        if gs >= 3: gs = 3
        d = "H" if gd > 0 else "A" if gd < 0 else "D"
        return f"gs{gs}_{d}"

    base = df.copy()
    base["min_b"] = base["minute"].apply(bucket_minute)
    base["state"] = base.apply(score_state, axis=1)

    def baseline_of(y: np.ndarray, gmin: pd.Series, gstate: pd.Series):
        out = {}
        tmp = pd.DataFrame({"y": y, "min_b": gmin, "state": gstate})
        grp = tmp.groupby(["min_b","state"])
        cnt = grp["y"].count()
        pos = grp["y"].sum()
        for (mb, st), n in cnt.items():
            if n >= 50:
                out[f"{int(mb)}|{str(st)}"] = float(pos[(mb,st)] / n)
        return out

    bl: Dict[str, Dict[str,float]] = {}
    for th in OU_THRESHOLDS:
        code = head_code_for_ou(th)
        if code in heads_present:
            y = (base["final_total"].values >= th).astype(int)
            bl[code] = baseline_of(y, base["min_b"], base["state"])

    if "BTTS" in heads_present:
        bl["BTTS"] = baseline_of(base["label_btts"].values.astype(int), base["min_b"], base["state"])

    for (code, col) in [("WIN_HOME","label_win_home"), ("DRAW","label_draw"), ("WIN_AWAY","label_win_away")]:
        if code in heads_present:
            bl[code] = baseline_of(base[col].values.astype(int), base["min_b"], base["state"])

    return bl

# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-url", required=True)
    ap.add_argument("--min-minute", dest="min_minute", type=int, default=15)
    ap.add_argument("--test-size", type=float, default=0.25)  # internal use only
    ap.add_argument("--min-rows", type=int, default=800)
    args = ap.parse_args()

    conn = _connect(args.db_url)
    try:
        df = load_snapshots_with_labels(conn, args.min_minute)
        if df.empty:
            print("Not enough labeled data yet.")
            return

        # heads to train
        heads: Dict[str, Dict[str, Any]] = {}
        metrics_all: Dict[str, Dict[str, Any]] = {}

        # common matrix
        X = df[FEATURES].values

        # helper to train/evaluate one head
        def train_head(y: np.ndarray, code: str):
            nonlocal heads, metrics_all
            if len(np.unique(y)) < 2: return
            strat = y if (y.sum() and y.sum() != len(y)) else None
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=args.test_size, random_state=42, stratify=strat)
            m = fit_lr(Xtr, ytr)
            if m is None: return
            z_te = logits_from_model(m, Xte)
            # Platt
            z_tr = logits_from_model(m, Xtr)
            a, b = fit_platt(z_tr, ytr)
            p_te = proba_with_platt(z_te, a, b)

            # metrics
            try: auc = roc_auc_score(yte, p_te)
            except Exception: auc = float("nan")
            brier = brier_score_loss(yte, p_te)
            acc = accuracy_score(yte, (p_te >= 0.5).astype(int))
            prev = float(y.mean())
            metrics_all[code] = {"brier": float(brier), "acc": float(acc), "auc": float(auc),
                                 "n": int(len(yte)), "prevalence": float(prev),
                                 "majority_acc": float(max(prev, 1-prev)),
                                 "calibrated": True, "calibration_method": "sigmoid"}

            heads[code] = to_v2(m, FEATURES, (a,b))

        # O/U heads
        for th in OU_THRESHOLDS:
            code = head_code_for_ou(th)
            y = (df["final_total"].values >= th).astype(int)
            train_head(y, code)

        # BTTS
        train_head(df["label_btts"].values.astype(int), "BTTS")

        # 1X2 (three binary heads)
        train_head(df["label_win_home"].values.astype(int), "WIN_HOME")
        train_head(df["label_draw"].values.astype(int), "DRAW")
        train_head(df["label_win_away"].values.astype(int), "WIN_AWAY")

        if not heads:
            print("No heads trained (insufficient variation).")
            return

        # Save models into settings (v2 format per head)
        for code, v2 in heads.items():
            _exec(conn,
                  "INSERT INTO settings(key,value) VALUES(%s,%s) "
                  "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
                  (f"model_v2:{code}", json.dumps(v2)))

        # Build & save baselines + default policy
        baselines = build_baselines(df, list(heads.keys()))
        _exec(conn,
              "INSERT INTO settings(key,value) VALUES(%s,%s) "
              "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
              ("policy:baselines_v1", json.dumps(baselines)))

        policy = {
            "global": {"top_k_per_match": 2},
            "BTTS": {"min_prob": 0.60, "min_lift": 0.06, "max_minute": 88},
            "O15":  {"min_prob": 0.65, "min_lift": 0.08, "max_minute": 85},
            "O25":  {"min_prob": 0.60, "min_lift": 0.07, "max_minute": 85},
            "O35":  {"min_prob": 0.55, "min_lift": 0.08, "max_minute": 80},
            "O45":  {"min_prob": 0.52, "min_lift": 0.07, "max_minute": 75},
            "WIN_HOME": {"min_prob": 0.55, "min_lift": 0.07, "max_minute": 90},
            "DRAW":     {"min_prob": 0.42, "min_lift": 0.06, "max_minute": 75},
            "WIN_AWAY": {"min_prob": 0.55, "min_lift": 0.07, "max_minute": 90}
        }
        _exec(conn,
              "INSERT INTO settings(key,value) VALUES(%s,%s) "
              "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
              ("policy:heads_v1", json.dumps(policy)))

        # Compose trainer audit blob (summaries only)
        audit = {
            "trained_at_utc": datetime.utcnow().isoformat() + "Z",
            "features": FEATURES,
            "metrics": metrics_all
        }
        _exec(conn,
              "INSERT INTO settings(key,value) VALUES(%s,%s) "
              "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
              ("model_coeffs", json.dumps(audit)))

        print("Saved per‑head model_v2:* plus policy:baselines_v1, policy:heads_v1, and model_coeffs in settings.")

    finally:
        conn.close()

if __name__ == "__main__":
    main()
