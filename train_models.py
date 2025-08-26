# train_models.py
import argparse, json, sqlite3, sys
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

FEATURES = [
    "minute","goals_h","goals_a","goals_sum","goals_diff",
    "xg_h","xg_a","xg_sum","xg_diff",
    "sot_h","sot_a","sot_sum",
    "cor_h","cor_a","cor_sum",
    "pos_h","pos_a","pos_diff",
    "red_h","red_a","red_sum"
]

def _connect(db_url: Optional[str], db_path: Optional[str]):
    """
    Returns (engine_type, connection, cursor_factory)
    engine_type in {"pg","sqlite"}
    """
    if db_url:
        if psycopg2 is None:
            raise SystemExit("psycopg2 not installed but --db-url was provided.")
        # ensure sslmode=require if missing (mirrors main.py)
        if "sslmode=" not in db_url:
            db_url = db_url + ("&" if "?" in db_url else "?") + "sslmode=require"
        conn = psycopg2.connect(db_url)
        conn.autocommit = True
        return "pg", conn, None
    elif db_path:
        conn = sqlite3.connect(db_path)
        return "sqlite", conn, None
    else:
        raise SystemExit("Provide --db-url (Postgres) or --db (SQLite).")

def _read_sql(engine: str, conn, sql: str, params: Tuple=()):
    if engine == "pg":
        return pd.read_sql_query(sql, conn, params=params)
    else:
        return pd.read_sql_query(sql, conn, params=params)

def _exec(engine: str, conn, sql: str, params: Tuple):
    if engine == "pg":
        with conn.cursor() as cur:
            cur.execute(sql, params)
    else:
        cur = conn.cursor()
        cur.execute(sql, params)
        conn.commit()
        cur.close()

def load_data(engine: str, conn, min_minute: int = 15):
    # Pull the latest snapshot per match and join with final result labels
    # We parse payload JSON in Python (robust across engines)
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
    rows = _read_sql(engine, conn, q)

    if rows.empty:
        return pd.DataFrame(), pd.DataFrame()

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

        # Labels
        gh_f = int(row["final_goals_h"] or 0)
        ga_f = int(row["final_goals_a"] or 0)
        f["label_o25"] = 1 if (gh_f + ga_f) >= 3 else 0
        f["label_btts"] = 1 if int(row["btts_yes"] or 0) == 1 else 0
        feats.append(f)

    if not feats:
        return pd.DataFrame(), pd.DataFrame()

    df = pd.DataFrame(feats)

    # Hygiene
    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["minute"] = df["minute"].clip(0, 120)

    df = df[df["minute"] >= min_minute].copy()
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    return df[FEATURES + ["label_o25"]], df[FEATURES + ["label_btts"]]

def fit_lr_safe(X, y):
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

def retrain_models_job():
    logger.info("Starting retrain job...")
    try:
       from train_models import train_models  
        train_models()
        logger.info("Retrain completed successfully")
    except Exception as e:
        logger.error(f"Retrain job failed: {e}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-url", help="Postgres DSN (preferred, used by main.py)")
    ap.add_argument("--db", help="SQLite path (optional for local dev)")
    ap.add_argument("--min-minute", dest="min_minute", type=int, default=15)
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--min-rows", type=int, default=150)
    args = ap.parse_args()

    engine, conn, _ = _connect(args.db_url, args.db)

    try:
        df_o25, df_btts = load_data(engine, conn, args.min_minute)
        if df_o25.empty or df_btts.empty:
            print("Not enough labeled data yet.")
            return

        n_o25, n_btts = len(df_o25), len(df_btts)
        n = min(n_o25, n_btts)
        print(f"Samples O2.5={n_o25} | BTTS={n_btts}")
        print(f"O2.5 positive rate={df_o25['label_o25'].mean():.3f} | "
              f"BTTS positive rate={df_btts['label_btts'].mean():.3f}")
        if n < args.min_rows:
            print(f"Need more data: {n} < min-rows {args.min_rows}.")
            return

        # --- O/U 2.5 ---
        Xo = df_o25[FEATURES].values
        yo = df_o25["label_o25"].values.astype(int)
        strat_o = yo if (yo.sum() and yo.sum() != len(yo)) else None
        Xo_tr, Xo_te, yo_tr, yo_te = train_test_split(
            Xo, yo, test_size=args.test_size, random_state=42, stratify=strat_o
        )
        mo = fit_lr_safe(Xo_tr, yo_tr)
        if mo is None:
            print("O2.5 split became single-class; collect more balanced data.")
            return
        p_te_o = mo.predict_proba(Xo_te)[:, 1]
        brier_o = brier_score_loss(yo_te, p_te_o)
        acc_o = accuracy_score(yo_te, (p_te_o >= 0.5).astype(int))

        # --- BTTS Yes ---
        Xb = df_btts[FEATURES].values
        yb = df_btts["label_btts"].values.astype(int)
        strat_b = yb if (yb.sum() and yb.sum() != len(yb)) else None
        Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(
            Xb, yb, test_size=args.test_size, random_state=42, stratify=strat_b
        )
        mb = fit_lr_safe(Xb_tr, yb_tr)
        if mb is None:
            print("BTTS split became single-class; collect more balanced data.")
            return
        p_te_b = mb.predict_proba(Xb_te)[:, 1]
        brier_b = brier_score_loss(yb_te, p_te_b)
        acc_b = accuracy_score(yb_te, (p_te_b >= 0.5).astype(int))

        print(f"O2.5 Brier={brier_o:.4f}  Acc={acc_o:.3f}  n={len(yo_te)}  prev={yo.mean():.2f}")
        print(f"BTTS  Brier={brier_b:.4f}  Acc={acc_b:.3f}  n={len(yb_te)}  prev={yb.mean():.2f}")

        blob = {
            "O25": to_coeffs(mo, FEATURES),
            "BTTS_YES": to_coeffs(mb, FEATURES),
            "trained_at_utc": datetime.utcnow().isoformat() + "Z",
            "metrics": {
                "o25": {
                    "brier": float(brier_o), "acc": float(acc_o),
                    "n": int(len(yo_te)), "prevalence": float(yo.mean())
                },
                "btts": {
                    "brier": float(brier_b), "acc": float(acc_b),
                    "n": int(len(yb_te)), "prevalence": float(yb.mean())
                },
            },
            "features": FEATURES,
        }

        # Save into settings(model_coeffs)
        if engine == "pg":
            _exec(engine, conn,
                  "INSERT INTO settings(key,value) VALUES(%s,%s) "
                  "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
                  ("model_coeffs", json.dumps(blob)))
        else:
            _exec(engine, conn,
                  "INSERT INTO settings(key,value) VALUES(?,?) "
                  "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                  ("model_coeffs", json.dumps(blob)))

        print("Saved model_coeffs in settings.")

    finally:
        try:
            conn.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
