# ============================================================
#   GOALSNIPER.AI — MODEL TRAINING ENGINE
# ============================================================

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from joblib import dump

from supabase import create_client
from tqdm import tqdm

from app.config import (
    MODEL_1X2,
    MODEL_OU25,
    MODEL_BTTS,
    MODEL_DIR,
    SUPABASE_URL,
    SUPABASE_KEY,
)

# ------------------------------------------------------------
# INITIALIZE SUPABASE
# ------------------------------------------------------------
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
print("[TRAIN] Supabase connected.")


# ------------------------------------------------------------
# FETCH TRAINING DATA
# ------------------------------------------------------------

def fetch_training_dataset():
    """
    We train from the "tips" + "tip_results" tables.
    This ensures model learns from its own historical performance.
    """

    print("[TRAIN] Fetching training dataset...")

    resp = (
        supabase.table("tips")
        .select("*, tip_results(*)")
        .order("id", desc=False)
        .execute()
    )

    data = resp.data or []
    if not data:
        print("[TRAIN] No training data found. Using dummy dataset.")
        return pd.DataFrame([])

    rows = []
    for tip in data:
        res = tip.get("tip_results")
        if not res:
            continue

        res = res[0]

        rows.append({
            "fixture_id": tip["fixture_id"],
            "market": tip["market"],
            "selection": tip["selection"],
            "prob": tip["prob"],
            "odds": tip["odds"],
            "minute": tip["minute"],
            "result": res["result"],   # WIN / LOSS / PUSH
            "pnl": res["pnl"],
        })

    df = pd.DataFrame(rows)
    print(f"[TRAIN] Loaded {len(df)} historical examples.")
    return df


# ------------------------------------------------------------
# MARKET-SPECIFIC FORMATTERS
# ------------------------------------------------------------

def prepare_1x2(df):
    df = df[df["market"] == "1X2"]
    if df.empty:
        return None, None

    X = df[["prob", "minute", "odds"]].values

    # Encode target: 1 for win, 0 for loss, ignore push
    y = df["result"].map({"WIN": 1, "LOSS": 0}).dropna().astype(int)
    X = X[: len(y)]

    return X, y


def prepare_ou25(df):
    df = df[df["market"] == "OU25"]
    if df.empty:
        return None, None

    X = df[["prob", "minute", "odds"]].values
    y = df["result"].map({"WIN": 1, "LOSS": 0}).dropna().astype(int)
    X = X[: len(y)]

    return X, y


def prepare_btts(df):
    df = df[df["market"] == "BTTS"]
    if df.empty:
        return None, None

    X = df[["prob", "minute", "odds"]].values
    y = df["result"].map({"WIN": 1, "LOSS": 0}).dropna().astype(int)
    X = X[: len(y)]

    return X, y


# ------------------------------------------------------------
# TRAIN A SINGLE LOGISTIC MODEL
# ------------------------------------------------------------

def train_model(X, y):
    if len(X) < 10 or len(set(y)) < 2:
        print("[TRAIN] Not enough data → using dummy model.")
        return None

    clf = LogisticRegression(max_iter=1000)
    calibrated = CalibratedClassifierCV(clf, cv=3)

    calibrated.fit(X, y)
    return calibrated


# ------------------------------------------------------------
# SAVE MODEL TO DISK + SUPABASE STORAGE
# ------------------------------------------------------------

def save_model(model, path, storage_name):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    dump(model, path)
    print(f"[TRAIN] Saved model → {path}")

    try:
        with open(path, "rb") as f:
            supabase.storage.from_("models").upload(
                storage_name,
                f,
                {"content-type": "application/octet-stream"},
            )
        print(f"[TRAIN] Uploaded model to Supabase Storage → {storage_name}")
    except Exception as e:
        print("[TRAIN] Storage upload failed:", e)


# ------------------------------------------------------------
# TRAINING PIPELINE
# ------------------------------------------------------------

def main():
    print("===========================================")
    print("     GOALSNIPER AI — MODEL TRAINING")
    print("===========================================")

    df = fetch_training_dataset()

    # If completely empty → create dummy models
    if df.empty:
        print("[TRAIN] Dataset empty → generating dummy models.")
        return train_dummies()

    # -------------------------
    # TRAIN EACH MARKET
    # -------------------------

    # --- 1X2 ---
    X1, y1 = prepare_1x2(df)
    if X1 is not None:
        m1 = train_model(X1, y1)
        if m1:
            save_model(m1, MODEL_1X2, "logreg_1x2.pkl")

    # --- OU25 ---
    X2, y2 = prepare_ou25(df)
    if X2 is not None:
        m2 = train_model(X2, y2)
        if m2:
            save_model(m2, MODEL_OU25, "logreg_ou25.pkl")

    # --- BTTS ---
    X3, y3 = prepare_btts(df)
    if X3 is not None:
        m3 = train_model(X3, y3)
        if m3:
            save_model(m3, MODEL_BTTS, "logreg_btts.pkl")

    print("[TRAIN] All models trained successfully.")


# ------------------------------------------------------------
# FALLBACK — CREATE DUMMY MODELS
# ------------------------------------------------------------

def train_dummies():
    """
    Creates flat probability models.
    """

    print("[TRAIN] Creating dummy models...")

    dummy = LogisticRegression()
    dummy.coef_ = np.zeros((1, 3))
    dummy.intercept_ = np.zeros(1)
    dummy.classes_ = np.array([0, 1])

    save_model(dummy, MODEL_1X2, "logreg_1x2.pkl")
    save_model(dummy, MODEL_OU25, "logreg_ou25.pkl")
    save_model(dummy, MODEL_BTTS, "logreg_btts.pkl")

    print("[TRAIN] Dummy models created.")


# ------------------------------------------------------------
# MANUAL TRIGGER FROM main.py
# ------------------------------------------------------------

def run_full_training():
    print("[TRAIN] Manual training triggered...")
    main()


# ------------------------------------------------------------
# COMMAND LINE ENTRY
# ------------------------------------------------------------

if __name__ == "__main__":
    main()
