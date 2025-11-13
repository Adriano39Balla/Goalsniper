import os
import joblib
import numpy as np

from supabase import create_client
from app.config import (
    MODEL_1X2,
    MODEL_OU25,
    MODEL_BTTS,
    MODEL_DIR,
    SUPABASE_URL,
    SUPABASE_KEY,
)

# ================================================================
# SUPABASE STORAGE CLIENT
# ================================================================

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
BUCKET = "models"   # Required Supabase bucket name


# ================================================================
# ENSURE MODEL DIRECTORY EXISTS
# ================================================================
os.makedirs(MODEL_DIR, exist_ok=True)


# ================================================================
# DOWNLOAD FROM SUPABASE STORAGE
# ================================================================

def download_model_if_exists(filename):
    """
    Attempts to download a model file from Supabase Storage.
    If not present → returns False.
    """

    try:
        path_in_bucket = f"{filename}"
        local_path = os.path.join(MODEL_DIR, filename)

        print(f"[MODELS] Trying to download {filename} from Supabase...")

        resp = supabase.storage.from_(BUCKET).download(path_in_bucket)

        if resp:
            with open(local_path, "wb") as f:
                f.write(resp)
            print(f"[MODELS] Downloaded {filename} successfully.")
            return True

    except Exception as e:
        print(f"[MODELS] No Supabase version for {filename}: {e}")

    return False


# ================================================================
# UPLOAD TO SUPABASE STORAGE
# ================================================================

def upload_model_to_supabase(filename):
    """
    Uploads local model file to Supabase Storage.
    """

    local_path = os.path.join(MODEL_DIR, filename)

    try:
        with open(local_path, "rb") as f:
            supabase.storage.from_(BUCKET).upload(
                path=filename,
                file=f,
                file_options={"content-type": "application/octet-stream"},
                upsert=True,
            )
        print(f"[MODELS] Uploaded {filename} → Supabase Storage.")

    except Exception as e:
        print(f"[MODELS] Failed to upload {filename}: {e}")


# ================================================================
# DUMMY MODEL GENERATOR (fallback)
# ================================================================

def make_dummy_classifier(num_outputs):
    """
    Creates a simple dummy model that always predicts uniform probabilities.
    """
    class Dummy:
        def predict_proba(self, X):
            batch = X.shape[0] if len(X.shape) > 1 else 1
            return np.ones((batch, num_outputs)) / num_outputs

    return Dummy()


# ================================================================
# LOAD / CREATE MODELS
# ================================================================

def load_or_create_model(path, outputs, filename):
    """
    1) Try load local file
    2) Try download from Supabase
    3) Generate dummy and upload
    """

    # 1 — local
    if os.path.exists(path):
        print(f"[MODELS] Loaded {path}")
        return joblib.load(path)

    # 2 — Supabase download
    if download_model_if_exists(filename):
        print(f"[MODELS] Loaded {filename} from Supabase")
        return joblib.load(path)

    # 3 — Dummy fallback
    print(f"[MODELS] No model found → creating dummy for {filename}")
    model = make_dummy_classifier(outputs)
    joblib.dump(model, path)
    upload_model_to_supabase(filename)
    return model


# ================================================================
# LOAD ALL MODELS (on startup)
# ================================================================

print("[MODELS] Initializing model system...")

model_1x2 = load_or_create_model(MODEL_1X2, 3, "logreg_1x2.pkl")
model_ou25 = load_or_create_model(MODEL_OU25, 2, "logreg_ou25.pkl")
model_btts = load_or_create_model(MODEL_BTTS, 2, "logreg_btts.pkl")

print("[MODELS] Model system ready.")


# ================================================================
# PREDICTION FUNCTIONS
# ================================================================

def predict_1x2(features: np.ndarray):
    """
    Returns dict: { "home": p, "draw": p, "away": p }
    """

    try:
        X = features.reshape(1, -1)
        probs = model_1x2.predict_proba(X)[0]

        return {
            "home": float(probs[0]),
            "draw": float(probs[1]),
            "away": float(probs[2]),
        }

    except Exception as e:
        print("[MODELS] Error in predict_1x2:", e)
        return {"home": 0.33, "draw": 0.33, "away": 0.33}


def predict_ou25(features: np.ndarray):
    """
    Returns dict: { "over": p, "under": p }
    """

    try:
        X = features.reshape(1, -1)
        probs = model_ou25.predict_proba(X)[0]

        return {
            "over": float(probs[0]),
            "under": float(probs[1]),
        }

    except Exception as e:
        print("[MODELS] Error in predict_ou25:", e)
        return {"over": 0.5, "under": 0.5}


def predict_btts(features: np.ndarray):
    """
    Returns dict: { "yes": p, "no": p }
    """

    try:
        X = features.reshape(1, -1)
        probs = model_btts.predict_proba(X)[0]

        return {
            "yes": float(probs[0]),
            "no": float(probs[1]),
        }

    except Exception as e:
        print("[MODELS] Error in predict_btts:", e)
        return {"yes": 0.5, "no": 0.5}
