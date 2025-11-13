import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
from supabase import create_client
from app.features import (
    build_features_1x2,
    build_features_ou25,
    build_features_btts
)
from app.supabase_db import fetch_training_fixtures
from app.config import (
    MODEL_1X2,
    MODEL_OU25,
    MODEL_BTTS,
)

# -------------------------
# Supabase client
# -------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Ensure models folder exists
os.makedirs("models", exist_ok=True)


# -------------------------
# Upload trained models
# -------------------------
def upload_model(local_path: str, upload_name: str):
    print(f"[TRAIN] Uploading {upload_name} to Supabase Storage...")
    with open(local_path, "rb") as f:
        supabase.storage.from_("models").upload(
            upload_name,
            f.read(),
            {
                "content-type": "application/octet-stream",
                "x-upsert": "true"
            }
        )
    print(f"[TRAIN] Uploaded {upload_name}.")


# -------------------------
# MAIN TRAINING PIPELINE
# -------------------------
def main():

    print("[TRAIN] Fetching training fixtures...")
    fixtures = fetch_training_fixtures()

    if not fixtures:
        print("[TRAIN] No fixtures available. Exiting.")
        return

    df = pd.DataFrame(fixtures)
    print(f"[TRAIN] Training on {len(df)} fixtures.")

    # Safety checks
    required_cols = [
        "fixture",
        "stats",
        "odds",
        "prematch_strength",
        "prematch_goal_expectation",
        "prematch_btts_expectation",
        "result_1x2",
        "result_ou25",
        "result_btts",
    ]
    for c in required_cols:
        if c not in df.columns:
            print(f"[TRAIN] Missing column: {c}")
            return

    # --------------------------
    # TRAIN 1X2
    # --------------------------
    X_1x2 = []
    y_1x2 = []

    for _, row in df.iterrows():
        feats, _ = build_features_1x2(
            row["fixture"],
            row["stats"],
            row["odds"],
            row["prematch_strength"]
        )
        X_1x2.append(feats)
        y_1x2.append(row["result_1x2"])

    X_1x2 = np.vstack(X_1x2)
    y_1x2 = np.array(y_1x2)

    print(f"[TRAIN] Training 1X2 model: X={X_1x2.shape}, y={y_1x2.shape}")

    clf_1x2 = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=800,
        n_jobs=-1
    )
    clf_1x2.fit(X_1x2, y_1x2)

    joblib.dump(clf_1x2, MODEL_1X2)
    upload_model(MODEL_1X2, "logreg_1x2.pkl")

    # --------------------------
    # TRAIN OU 2.5
    # --------------------------
    X_ou = []
    y_ou = []

    for _, row in df.iterrows():
        feats, _ = build_features_ou25(
            row["fixture"],
            row["stats"],
            row["odds"],
            row["prematch_goal_expectation"]
        )
        X_ou.append(feats)
        y_ou.append(row["result_ou25"])

    X_ou = np.vstack(X_ou)
    y_ou = np.array(y_ou)

    print(f"[TRAIN] Training OU25 model: X={X_ou.shape}, y={y_ou.shape}")

    clf_ou = LogisticRegression(
        solver="lbfgs",
        max_iter=800,
        n_jobs=-1
    )
    clf_ou.fit(X_ou, y_ou)

    joblib.dump(clf_ou, MODEL_OU25)
    upload_model(MODEL_OU25, "logreg_ou25.pkl")

    # --------------------------
    # TRAIN BTTS
    # --------------------------
    X_btts = []
    y_btts = []

    for _, row in df.iterrows():
        feats, _ = build_features_btts(
            row["fixture"],
            row["stats"],
            row["odds"],
            row["prematch_btts_expectation"]
        )
        X_btts.append(feats)
        y_btts.append(row["result_btts"])

    X_btts = np.vstack(X_btts)
    y_btts = np.array(y_btts)

    print(f"[TRAIN] Training BTTS model: X={X_btts.shape}, y={y_btts.shape}")

    clf_btts = LogisticRegression(
        solver="lbfgs",
        max_iter=800,
        n_jobs=-1
    )
    clf_btts.fit(X_btts, y_btts)

    joblib.dump(clf_btts, MODEL_BTTS)
    upload_model(MODEL_BTTS, "logreg_btts.pkl")

    print("[TRAIN] Training completed successfully.")


if __name__ == "__main__":
    main()
