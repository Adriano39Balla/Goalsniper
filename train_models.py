# ============================================================
#   GOALSNIPER.AI — MODEL TRAINING ENGINE (FINAL VERSION)
# ============================================================

import os
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from joblib import dump

from supabase import create_client

from app.config import (
    MODEL_1X2,
    MODEL_OU25,
    MODEL_BTTS,
    MODEL_DIR,
    SUPABASE_URL,
    SUPABASE_KEY,
)

# ------------------------------------------------------------
# CONNECT SUPABASE
# ------------------------------------------------------------
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
print("[TRAIN] Supabase connected.")


# ------------------------------------------------------------
# FETCH TRAINING DATA
# ------------------------------------------------------------

def fetch_training_dataset():
    """
    Loads historical tips with results.
    Schema:
        tips.id -> tip_results.tip_id
    """

    print("[TRAIN] Fetching training dataset...")

    try:
        resp = (
            supabase.table("tips")
            .select("*, tip_results!tip_id(*)")
            .order("id", desc=False)
            .execute()
        )
    except Exception as e:
        print("[TRAIN] ERROR loading data:", e)
        return pd.DataFrame([])

    data = resp.data or []

    rows = []
    for tip in data:
        results = tip.get("tip_results") or []
        if not results:
            continue  # No grading yet

        res = results[0]

        # Skip incomplete rows
        if tip["prob"] is None or tip["odds"] is None:
            continue
        if res["result"] not in ("WIN", "LOSS"):
            continue

        rows.append({
            "fixture_id": tip["fixture_id"],
            "market": tip["market"],
            "selection": tip["selection"],
            "prob": float(tip["prob"]),
            "odds": float(tip["odds"]),
            "minute": tip.get("minute") or 0,
            "result": res["result"],
            "pnl": res["pnl"],
        })

    df = pd.DataFrame(rows)
    print(f"[TRAIN] Loaded {len(df)} labelled examples.")
    return df


# ------------------------------------------------------------
# MARKET PREPROCESSORS
# ------------------------------------------------------------

def prep(df, market):
    df = df[df["market"] == market]
    if df.empty:
        return None, None
    X = df[["prob", "minute", "odds"]].fillna(0).astype(float).values
    y = df["result"].map({"WIN": 1, "LOSS": 0}).astype(int).values
    return X, y


# ------------------------------------------------------------
# TRAIN A MODEL
# ------------------------------------------------------------

def train_model(X, y):
    if len(X) < 30 or len(set(y)) < 2:
        print("[TRAIN] Not enough samples → fallback dummy.")
        return None

    base = LogisticRegression(max_iter=1000)
    model = CalibratedClassifierCV(base, cv=3)
    model.fit(X, y)
    return model


# ------------------------------------------------------------
# SAVE MODEL
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
                file_options={"content-type": "application/octet-stream"},
                upsert=True
            )
        print(f"[TRAIN] Uploaded → Supabase Storage/{storage_name}")
    except Exception as e:
        print("[TRAIN] Upload failed:", e)


# ------------------------------------------------------------
# DUMMY MODEL (SAFE)
# ------------------------------------------------------------

def make_dummy():
    """
    Properly fitted dummy model so predict_proba works.
    """
    X = np.array([[0, 0, 1], [1, 90, 10]])
    y = np.array([0, 1])

    base = LogisticRegression(max_iter=200)
    model = CalibratedClassifierCV(base, cv=2)
    model.fit(X, y)

    return model


def train_dummies():
    print("[TRAIN] Creating dummy models...")

    m = make_dummy()

    save_model(m, MODEL_1X2, "logreg_1x2.pkl")
    save_model(m, MODEL_OU25, "logreg_ou25.pkl")
    save_model(m, MODEL_BTTS, "logreg_btts.pkl")

    print("[TRAIN] Dummy models created.")


# ------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------

def main():
    print("===========================================")
    print("      GOALSNIPER AI — TRAINING START")
    print("===========================================")

    df = fetch_training_dataset()

    if df.empty:
        print("[TRAIN] No historical data → using dummy models.")
        return train_dummies()

    # 1X2
    X1, y1 = prep(df, "1X2")
    if X1 is not None:
        m1 = train_model(X1, y1)
        if m1: save_model(m1, MODEL_1X2, "logreg_1x2.pkl")

    # OU25
    X2, y2 = prep(df, "OU25")
    if X2 is not None:
        m2 = train_model(X2, y2)
        if m2: save_model(m2, MODEL_OU25, "logreg_ou25.pkl")

    # BTTS
    X3, y3 = prep(df, "BTTS")
    if X3 is not None:
        m3 = train_model(X3, y3)
        if m3: save_model(m3, MODEL_BTTS, "logreg_btts.pkl")

    print("[TRAIN] Training complete.")


# ------------------------------------------------------------
# MAIN + MANUAL TRIGGER
# ------------------------------------------------------------

def run_full_training():
    print("[TRAIN] Manual run triggered.")
    main()


if __name__ == "__main__":
    main()
