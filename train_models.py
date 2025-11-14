# ============================================================
#   GOALSNIPER.AI — MODEL TRAINING (USING DIRECT POSTGRES)
# ============================================================

import os
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from joblib import dump

from supabase import create_client

from app.db_pg import fetch_all
from app.config import (
    MODEL_1X2,
    MODEL_OU25,
    MODEL_BTTS,
    MODEL_DIR,
    SUPABASE_URL,
    SUPABASE_KEY,
)

# ------------------------------------------------------------
# SUPABASE CLIENT JUST FOR STORAGE (models bucket)
# ------------------------------------------------------------
try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("[TRAIN] Supabase client initialized.")
except Exception as e:
    print("[TRAIN] Supabase client init FAILED:", e)
    supabase = None


# ------------------------------------------------------------
# FETCH TRAINING DATA VIA SQL
# ------------------------------------------------------------

def fetch_training_dataset():
    """
    Pull labeled examples directly via SQL.

    We join:
      tips.id           -> tip_results.tip_id
    Only keep rows with result in ('WIN', 'LOSS').
    """

    print("[TRAIN] Loading training data with direct SQL...")

    sql = """
        SELECT
            t.id           AS tip_id,
            t.fixture_id   AS fixture_id,
            t.market       AS market,
            t.selection    AS selection,
            t.prob         AS prob,
            t.odds         AS odds,
            t.minute       AS minute,
            r.result       AS result,
            r.pnl          AS pnl
        FROM tips t
        JOIN tip_results r ON r.tip_id = t.id
        WHERE r.result IN ('WIN', 'LOSS')
    """

    rows = fetch_all(sql)
    if not rows:
        print("[TRAIN] No labeled rows in tips + tip_results.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    print(f"[TRAIN] Loaded {len(df)} labeled examples from SQL.")
    return df


# ------------------------------------------------------------
# MARKET PREPROCESSORS
# ------------------------------------------------------------

def prep_market(df: pd.DataFrame, market: str):
    sub = df[df["market"] == market].copy()
    if sub.empty:
        print(f"[TRAIN] No rows for market={market}")
        return None, None

    # Basic feature set
    sub = sub.dropna(subset=["prob", "odds"])
    if sub.empty:
        print(f"[TRAIN] No non-null rows for market={market}")
        return None, None

    sub["minute"] = sub["minute"].fillna(0)

    X = sub[["prob", "minute", "odds"]].astype(float).values
    y = sub["result"].map({"WIN": 1, "LOSS": 0}).astype(int).values

    if len(X) < 20 or len(set(y)) < 2:
        print(f"[TRAIN] Not enough usable data for {market} → {len(X)} rows")
        return None, None

    print(f"[TRAIN] Prepared {len(X)} rows for market={market}")
    return X, y


# ------------------------------------------------------------
# TRAIN A MODEL
# ------------------------------------------------------------

def train_model(X, y, market_name: str):
    if X is None or y is None:
        print(f"[TRAIN] Skipping {market_name} — no data.")
        return None

    base = LogisticRegression(max_iter=1000)
    model = CalibratedClassifierCV(base, cv=3)

    model.fit(X, y)
    print(f"[TRAIN] {market_name} model trained on {len(X)} samples.")
    return model


# ------------------------------------------------------------
# SAVE MODEL LOCALLY + TO SUPABASE STORAGE
# ------------------------------------------------------------

def save_model(model, path: str, storage_name: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    dump(model, path)
    print(f"[TRAIN] Saved model to {path}")

    if not supabase:
        print("[TRAIN] Skipping upload — Supabase client not available.")
        return

    try:
        with open(path, "rb") as f:
            supabase.storage.from_("models").upload(
                storage_name,
                f,
                file_options={"content-type": "application/octet-stream"},
                upsert=True,
            )
        print(f"[TRAIN] Uploaded {storage_name} → Supabase Storage (bucket=models)")
    except Exception as e:
        print("[TRAIN] Upload to storage failed:", e)


# ------------------------------------------------------------
# SAFE DUMMY MODELS (FALLBACK)
# ------------------------------------------------------------

def make_safe_dummy():
    """
    Create a small but real fitted model so predict_proba works.
    """
    X = np.array([[0.1, 5, 2.0], [0.9, 85, 10.0]])
    y = np.array([0, 1])

    base = LogisticRegression(max_iter=200)
    model = CalibratedClassifierCV(base, cv=2)
    model.fit(X, y)
    return model


def train_dummies():
    print("[TRAIN] Creating safe dummy models...")

    dummy = make_safe_dummy()
    save_model(dummy, MODEL_1X2, "logreg_1x2.pkl")
    save_model(dummy, MODEL_OU25, "logreg_ou25.pkl")
    save_model(dummy, MODEL_BTTS, "logreg_btts.pkl")

    print("[TRAIN] Safe dummy models created & saved.")


# ------------------------------------------------------------
# MAIN TRAINING PIPELINE
# ------------------------------------------------------------

def main():
    print("============================================")
    print("      GOALSNIPER AI — TRAINING START")
    print("============================================")
    print(f"[TRAIN] Timestamp UTC: {datetime.utcnow().isoformat()}")

    df = fetch_training_dataset()

    if df.empty:
        print("[TRAIN] No training data found → using dummy models.")
        return train_dummies()

    # 1X2
    X1, y1 = prep_market(df, "1X2")
    m1 = train_model(X1, y1, "1X2")
    if m1:
        save_model(m1, MODEL_1X2, "logreg_1x2.pkl")

    # OU25
    X2, y2 = prep_market(df, "OU25")
    m2 = train_model(X2, y2, "OU25")
    if m2:
        save_model(m2, MODEL_OU25, "logreg_ou25.pkl")

    # BTTS
    X3, y3 = prep_market(df, "BTTS")
    m3 = train_model(X3, y3, "BTTS")
    if m3:
        save_model(m3, MODEL_BTTS, "logreg_btts.pkl")

    print("[TRAIN] Training complete.")


# ------------------------------------------------------------
# PUBLIC ENTRYPOINT FOR main.py
# ------------------------------------------------------------

def run_full_training():
    print("[TRAIN] Manual run triggered.")
    main()


if __name__ == "__main__":
    main()
