# path: train_models.py
import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, accuracy_score
from sklearn.model_selection import train_test_split
from datetime import datetime

try:
    import psycopg2
except Exception:
    psycopg2 = None

# ──────────────────────────────────────────────────────────────────────────────
# Feature order must match main.py extraction keys
FEATURES = [
    "minute","goals_h","goals_a","goals_sum","goals_diff",
    "xg_h","xg_a","xg_sum","xg_diff",
    "sot_h","sot_a","sot_sum",
    "cor_h","cor_a","cor_sum",
    "pos_h","pos_a","pos_diff",
    "red_h","red_a","red_sum"
]

# ──────────────────────────────────────────────────────────────────────────────
def _connect(db_url: Optional[str]):
    if not db_url:
        raise SystemExit("DATABASE_URL required (pass --db-url or set env)")
    if psycopg2 is None:
        raise SystemExit("psycopg2 is required for Postgres training.")
    # normalize to require SSL like main.py
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

# ──────────────────────────────────────────────────────────────────────────────
def load_snapshots_with_labels(conn, min_minute: int = 15) -> pd.DataFrame:
    """
    Pull latest snapshot per match and join with final labels.
    """
    q = """
    WITH latest AS (
      SELECT match_id, MAX(created_ts) AS ts
      FROM tip_snapshots
      GROUP BY match_id
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

        # Deriveds
        f["goals_sum"] = f["goals_h"] + f["goals_a"]
        f["goals_diff"] = f["goals_h"] - f["goals_a"]
        f["xg_sum"] = f["xg_h"] + f["xg_a"]
        f["xg_diff"] = f["xg_h"] - f["xg_a"]
        f["sot_sum"] = f["sot_h"] + f["sot_a"]
        f["cor_sum"] = f["cor_h"] + f["cor_a"]
        f["pos_diff"] = f["pos_h"] - f["pos_a"]
        f["red_sum"] = f["red_h"] + f["red_a"]

        # Labels
        gh_f = int(row["final_goals_h"] or 0)
        ga_f = int(row["final_goals_a"] or 0)
        total = gh_f + ga_f
        f["label_o25"] = 1 if total >= 3 else 0
        f["label_btts"] = 1 if int(row["btts_yes"] or 0) == 1 else 0

        feats.append(f)

    if not feats:
        return pd.DataFrame()

    df = pd.DataFrame(feats)
    # Hygiene like main.py
    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["minute"] = df["minute"].clip(0, 120)
    df = df[df["minute"] >= min_minute].copy()
    return df

def _fit_lr(X: np.ndarray, y: np.ndarray) -> Optional[LogisticRegression]:
    if len(np.unique(y)) < 2:
        return None
    # No scaler: export weights as-is to match main.py’s linear scorer.
    return LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear"
    ).fit(X, y)

def _to_v2_payload(model: LogisticRegression, feature_names: List[str]) -> Dict[str, Any]:
    """
    Shape that main.py expects under settings key `model_v2:*`
    """
    coef = model.coef_.ravel().tolist()
    intercept = float(model.intercept_.ravel()[0])
    weights = {name: float(w) for name, w in zip(feature_names, coef)}
    return {
        "intercept": intercept,
        "weights": weights,
        "calibration": {"method": "sigmoid", "a": 1.0, "b": 0.0}
    }

# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-url", default=os.getenv("DATABASE_URL"))
    ap.add_argument("--min-minute", type=int, default=int(os.getenv("TRAIN_MIN_MINUTE", "15")))
    ap.add_argument("--test-size", type=float, default=float(os.getenv("TRAIN_TEST_SIZE", "0.25")))
    ap.add_argument("--min-rows", type=int, default=300)
    args = ap.parse_args()

    conn = _connect(args.db_url)
    try:
        df = load_snapshots_with_labels(conn, args.min_minute)
        if df.empty:
            print("Not enough labeled data yet.")
            sys.exit(0)

        # Prepare O2.5
        dfo = df[FEATURES + ["label_o25"]].copy()
        Xo = dfo[FEATURES].values
        yo = dfo["label_o25"].astype(int).values
        strat_o = yo if (yo.sum() and yo.sum() != len(yo)) else None
        Xo_tr, Xo_te, yo_tr, yo_te = train_test_split(Xo, yo, test_size=args.test_size, random_state=42, stratify=strat_o)
        mo = _fit_lr(Xo_tr, yo_tr)
        if mo is None:
            print("O2.5 split became single-class; collect more balanced data.")
            sys.exit(0)
        p_te_o = mo.predict_proba(Xo_te)[:, 1]
        brier_o = brier_score_loss(yo_te, p_te_o)
        acc_o = accuracy_score(yo_te, (p_te_o >= 0.5).astype(int))
        prev_o = float(yo.mean())

        # Prepare BTTS
        dfb = df[FEATURES + ["label_btts"]].copy()
        Xb = dfb[FEATURES].values
        yb = dfb["label_btts"].astype(int).values
        strat_b = yb if (yb.sum() and yb.sum() != len(yb)) else None
        Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(Xb, yb, test_size=args.test_size, random_state=42, stratify=strat_b)
        mb = _fit_lr(Xb_tr, yb_tr)
        if mb is None:
            print("BTTS split became single-class; collect more balanced data.")
            sys.exit(0)
        p_te_b = mb.predict_proba(Xb_te)[:, 1]
        brier_b = brier_score_loss(yb_te, p_te_b)
        acc_b = accuracy_score(yb_te, (p_te_b >= 0.5).astype(int))
        prev_b = float(yb.mean())

        # Persist models in the exact keys/shape main.py reads
        v2_o25 = _to_v2_payload(mo, FEATURES)
        v2_bttsyes = _to_v2_payload(mb, FEATURES)

        _exec(conn,
              "INSERT INTO settings(key,value) VALUES(%s,%s) "
              "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
              ("model_v2:O25", json.dumps(v2_o25)))
        _exec(conn,
              "INSERT INTO settings(key,value) VALUES(%s,%s) "
              "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
              ("model_v2:BTTS_YES", json.dumps(v2_bttsyes)))

        # Also write an audit blob used by dashboards/digests
        audit = {
            "trained_at_utc": datetime.utcnow().isoformat() + "Z",
            "features": FEATURES,
            "metrics": {
                "O25": {"brier": float(brier_o), "acc": float(acc_o), "n": int(len(yo_te)), "prevalence": float(prev_o)},
                "BTTS": {"brier": float(brier_b), "acc": float(acc_b), "n": int(len(yb_te)), "prevalence": float(prev_b)},
            },
        }
        _exec(conn,
              "INSERT INTO settings(key,value) VALUES(%s,%s) "
              "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
              ("model_coeffs", json.dumps(audit)))

        print("Saved model_v2:O25, model_v2:BTTS_YES and model_coeffs.")
        sys.exit(0)

    except Exception as e:
        print(f"[TRAIN] exception: {e}", flush=True)
        sys.exit(1)
    finally:
        try:
            conn.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
