# ============================================================
#   GOALSNIPER.AI — TRAINING ENGINE (FINAL SAFE VERSION)
# ============================================================

import os
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from joblib import dump

from supabase import create_client
from app.db_pg import fetch_all   # direct SQL fetcher
from app.config import (
    MODEL_1X2, MODEL_OU25, MODEL_BTTS, MODEL_DIR,
    SUPABASE_URL, SUPABASE_KEY
)

# ------------------------------------------------------------
# SUPABASE STORAGE
# ------------------------------------------------------------
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
BUCKET = "models"


def upload_file_safe(local_path, storage_name):
    """Safe upload using sync API — NO upsert parameter."""
    try:
        with open(local_path, "rb") as f:
            supabase.storage \
                .from_(BUCKET) \
                .upload(storage_name, f)
        print(f"[TRAIN] Uploaded to storage → {storage_name}")
    except Exception as e:
        print(f"[TRAIN] Upload error for {storage_name}: {e}")


# ------------------------------------------------------------
# FETCH TRAINING DATA FROM SQL
# ------------------------------------------------------------

def fetch_training_dataset():
    sql = """
        SELECT
            t.fixture_id,
            t.market,
            t.selection,
            t.minute,
            t.prob,
            t.odds,
            r.result,
            r.pnl
        FROM tips t
        JOIN tip_results r ON r.tip_id = t.id
        WHERE r.result IN ('WIN','LOSS')
    """
    rows = fetch_all(sql)
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    print(f"[TRAIN] Loaded {len(df)} rows")
    return df


# ------------------------------------------------------------
# MARKET PREP
# ------------------------------------------------------------

MIN_SAMPLES = 10
MIN_CLASS = 2

def prep_market(df, market):
    df = df[df["market"] == market]
    if df.empty:
        return None, None

    df = df.dropna(subset=["prob", "odds"])
    df["minute"] = df["minute"].fillna(0)

    X = df[["prob", "minute", "odds"]].astype(float).values
    y = df["result"].map({"WIN": 1, "LOSS": 0}).astype(int).values

    n0 = (y == 0).sum()
    n1 = (y == 1).sum()
    total = len(y)

    print(f"[TRAIN] {market}: total={total}, WIN={n1}, LOSS={n0}")

    if total < MIN_SAMPLES or min(n0, n1) < MIN_CLASS:
        print(f"[TRAIN] {market}: insufficient data → fallback")
        return None, None

    return X, y


# ------------------------------------------------------------
# SAFE TRAINING LOGIC
# ------------------------------------------------------------

def train_safe_model(X, y, market):
    """
    1. If real data is present → try calibrated CV
    2. If data too small → build dummy X,y
    """

    # If data missing → use dummy set
    if X is None or y is None:
        X = np.array([[0.2, 10, 2.5], [0.7, 80, 1.8]])
        y = np.array([0, 1])
        print(f"[TRAIN] {market}: using dummy synthetic dataset")

    # Try calibrated model
    try:
        base = LogisticRegression(max_iter=1000)
        model = CalibratedClassifierCV(base, cv=2)
        model.fit(X, y)
        print(f"[TRAIN] {market}: calibrated model trained")
        return model
    except Exception as e:
        print(f"[TRAIN] {market}: calibration failed → {e}")

    # Try simple logistic regression
    try:
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        print(f"[TRAIN] {market}: uncalibrated model trained")
        return model
    except Exception as e:
        print(f"[TRAIN] {market}: uncalibrated failed → {e}")

    # FINAL fallback: guaranteed trainable dummy
    print(f"[TRAIN] {market}: using final dummy fallback")
    X = np.array([[0, 0, 1], [1, 90, 10]])
    y = np.array([0, 1])

    model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    return model


# ------------------------------------------------------------
# SAVE MODEL
# ------------------------------------------------------------

def save_model(model, path, storage_name):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dump(model, path)
    print(f"[TRAIN] Saved model → {path}")

    upload_file_safe(path, storage_name)


# ------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------

def main():
    print("============================================")
    print("      GOALSNIPER AI — TRAINING START")
    print("============================================")

    df = fetch_training_dataset()

    # Market list
    markets = [
        ("1X2", MODEL_1X2, "logreg_1x2.pkl"),
        ("OU25", MODEL_OU25, "logreg_ou25.pkl"),
        ("BTTS", MODEL_BTTS, "logreg_btts.pkl"),
    ]

    for market, model_path, storage_name in markets:
        X, y = prep_market(df, market)
        model = train_safe_model(X, y, market)
        save_model(model, model_path, storage_name)

    print("[TRAIN] Training complete")


# Callback for main app
def run_full_training():
    main()


if __name__ == "__main__":
    main()
