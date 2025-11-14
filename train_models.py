# ============================================================
#   GOALSNIPER.AI — MODEL TRAINING (SAFE AGAINST LOW DATA)
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
# SUPABASE STORAGE CLIENT
# ------------------------------------------------------------
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# ------------------------------------------------------------
# FETCH TRAINING DATA
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
            r.pnl,
            t.id AS tip_id
        FROM tips t
        JOIN tip_results r ON r.tip_id = t.id
        WHERE r.result IN ('WIN','LOSS')
    """
    rows = fetch_all(sql)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ------------------------------------------------------------
# SAFE MARKET PREPARATION
# ------------------------------------------------------------

MIN_SAMPLES = 10  # absolute minimum
MIN_CLASS = 2     # minimum samples per class

def prep_market(df, market):
    df = df[df["market"] == market].dropna(subset=["prob", "odds"])
    if df.empty:
        print(f"[TRAIN] No rows for {market}")
        return None, None

    df["minute"] = df["minute"].fillna(0)

    X = df[["prob", "minute", "odds"]].astype(float).values
    y = df["result"].map({"WIN": 1, "LOSS": 0}).astype(int).values

    # Check class counts
    n0 = (y == 0).sum()
    n1 = (y == 1).sum()
    total = len(y)

    print(f"[TRAIN] {market}: total={total}, WIN={n1}, LOSS={n0}")

    if total < MIN_SAMPLES or min(n0, n1) < MIN_CLASS:
        print(f"[TRAIN] Not enough balanced data for {market} → using SAFE FALLBACK")
        return None, None

    return X, y


# ------------------------------------------------------------
# SAFE TRAINING LOGIC
# ------------------------------------------------------------

def train_safe_model(X, y, market):
    """
    1. Try calibrated model
    2. If class count too small → fallback to plain LogisticRegression
    3. If even that fails → dummy model
    """

    # ===== Attempt full calibrated model =====
    try:
        base = LogisticRegression(max_iter=1000)
        model = CalibratedClassifierCV(base, cv=3)

        model.fit(X, y)
        print(f"[TRAIN] {market}: Calibrated model trained")
        return model

    except Exception as e:
        print(f"[TRAIN] {market}: calibration failed → {e}")

    # ===== Fallback to uncalibrated LogisticRegression =====
    try:
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        print(f"[TRAIN] {market}: Uncalibrated model trained")
        return model

    except Exception as e:
        print(f"[TRAIN] {market}: uncalibrated fallback failed → {e}")

    # ===== Final fallback → SAFE dummy =====
    print(f"[TRAIN] {market}: using dummy fallback model")

    dummy_X = np.array([[0.1, 1, 1], [0.9, 89, 10]])
    dummy_y = np.array([0, 1])

    model = LogisticRegression(max_iter=200)
    model.fit(dummy_X, dummy_y)
    return model


# ------------------------------------------------------------
# SAVE MODEL
# ------------------------------------------------------------

def save_model(model, path, name):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dump(model, path)

    try:
        with open(path, "rb") as f:
            supabase.storage.from_("models").upload(
                name,
                f,
                file_options={"content-type": "application/octet-stream"},
                upsert=True,
            )
        print(f"[TRAIN] Uploaded {name} to Supabase")
    except Exception as e:
        print(f"[TRAIN] Upload error for {name}:", e)


# ------------------------------------------------------------
# MAIN TRAINING PIPELINE
# ------------------------------------------------------------

def main():
    print("============================================")
    print("      GOALSNIPER AI — TRAINING START")
    print("============================================")

    df = fetch_training_dataset()

    if df.empty:
        print("[TRAIN] No training data → using dummy models")
        return train_dummies()

    # -------------- Train markets --------------

    for market, path, storage in [
        ("1X2", MODEL_1X2, "logreg_1x2.pkl"),
        ("OU25", MODEL_OU25, "logreg_ou25.pkl"),
        ("BTTS", MODEL_BTTS, "logreg_btts.pkl"),
    ]:
        X, y = prep_market(df, market)
        if X is None:
            print(f"[TRAIN] {market}: no usable data → dummy fallback")
            model = train_safe_model(None, None, market)
        else:
            model = train_safe_model(X, y, market)

        save_model(model, path, storage)

    print("[TRAIN] Training complete")


# ------------------------------------------------------------
def train_dummies():
    for market, path, name in [
        ("1X2", MODEL_1X2, "logreg_1x2.pkl"),
        ("OU25", MODEL_OU25, "logreg_ou25.pkl"),
        ("BTTS", MODEL_BTTS, "logreg_btts.pkl"),
    ]:
        model = train_safe_model(None, None, market)
        save_model(model, path, name)

    print("[TRAIN] Dummy fallback models saved.")


# Entry for main.py
def run_full_training():
    main()


if __name__ == "__main__":
    main()
