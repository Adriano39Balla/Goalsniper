import argparse, json, sqlite3
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

FEATURES = [
    "minute","goals_h","goals_a","goals_sum","goals_diff",
    "xg_h","xg_a","xg_sum","xg_diff",
    "sot_h","sot_a","sot_sum",
    "cor_h","cor_a","cor_sum",
    "pos_h","pos_a","pos_diff",
    "red_h","red_a","red_sum"
]

def load_data(db_path: str, min_minute: int = 15):
    con = sqlite3.connect(db_path)
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
    rows = pd.read_sql_query(q, con)
    if rows.empty:
        return pd.DataFrame(), pd.DataFrame()

    feats = []
    for _, row in rows.iterrows():
        p = json.loads(row["payload"])
        stat = p.get("stat", {})
        f = {
            "match_id": int(row["match_id"]),
            "minute": float(p.get("minute", 0)),
            "goals_h": float(p.get("gh", 0)),
            "goals_a": float(p.get("ga", 0)),
            "xg_h": float(stat.get("xg_h", 0)), "xg_a": float(stat.get("xg_a", 0)),
            "sot_h": float(stat.get("sot_h", 0)), "sot_a": float(stat.get("sot_a", 0)),
            "cor_h": float(stat.get("cor_h", 0)), "cor_a": float(stat.get("cor_a", 0)),
            "pos_h": float(stat.get("pos_h", 0)), "pos_a": float(stat.get("pos_a", 0)),
            "red_h": float(stat.get("red_h", 0)), "red_a": float(stat.get("red_a", 0)),
        }
        f["goals_sum"] = f["goals_h"] + f["goals_a"]
        f["goals_diff"] = f["goals_h"] - f["goals_a"]
        f["xg_sum"] = f["xg_h"] + f["xg_a"]
        f["xg_diff"] = f["xg_h"] - f["xg_a"]
        f["sot_sum"] = f["sot_h"] + f["sot_a"]
        f["cor_sum"] = f["cor_h"] + f["cor_a"]
        f["pos_diff"] = f["pos_h"] - f["pos_a"]
        f["red_sum"] = f["red_h"] + f["red_a"]
        f["label_o25"] = 1 if (row["final_goals_h"] + row["final_goals_a"]) >= 3 else 0
        f["label_btts"] = 1 if int(row["btts_yes"] or 0) == 1 else 0
        feats.append(f)

    df = pd.DataFrame(feats)
    df = df[df["minute"] >= min_minute].copy()
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    return df[FEATURES + ["label_o25"]], df[FEATURES + ["label_btts"]]

def fit_lr(X, y):
    model = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
    model.fit(X, y)
    return model

def to_coeffs(model, feature_names):
    return {
        "features": list(feature_names),
        "coef": model.coef_.ravel().tolist(),
        "intercept": float(model.intercept_.ravel()[0]),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="tip_performance.db")
    ap.add_argument("--min-minute", dest="min_minute", type=int, default=15)
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    df_o25, df_btts = load_data(args.db, args.min_minute)

    if df_o25.empty or df_btts.empty:
        print("Not enough labeled data yet.")
        return

    # O/U 2.5
    Xo, yo = df_o25[FEATURES].values, df_o25["label_o25"].values
    Xo_tr, Xo_te, yo_tr, yo_te = train_test_split(Xo, yo, test_size=args.test_size, random_state=42)
    mo = fit_lr(Xo_tr, yo_tr)
    p_te_o = mo.predict_proba(Xo_te)[:, 1]
    print(f"O2.5 Brier={brier_score_loss(yo_te, p_te_o):.4f}")

    # BTTS Yes
    Xb, yb = df_btts[FEATURES].values, df_btts["label_btts"].values
    Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(Xb, yb, test_size=args.test_size, random_state=42)
    mb = fit_lr(Xb_tr, yb_tr)
    p_te_b = mb.predict_proba(Xb_te)[:, 1]
    print(f"BTTS Brier={brier_score_loss(yb_te, p_te_b):.4f}")

    blob = {"O25": to_coeffs(mo, FEATURES), "BTTS_YES": to_coeffs(mb, FEATURES)}

    con = sqlite3.connect(args.db)
    con.execute("""
        INSERT INTO settings(key,value) VALUES('model_coeffs',?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value
    """, (json.dumps(blob),))
    con.commit()
    con.close()
    print("Saved model_coeffs in settings.")

if __name__ == "__main__":
    main()
