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
BUCKET = "models"   # REQUIRED Supabase bucket name


# ================================================================
# ENSURE MODEL DIRECTORY EXISTS
# ================================================================
os.makedirs(MODEL_DIR, exist_ok=True)


# ================================================================
# DUMMY CLASSIFIER (GLOBAL, SAFE, PICKLABLE)
# ================================================================

class DummyClassifier:
    """
    Simple picklable classifier with predict_proba.

    Replaces earlier bad local-function dummy, which was unpicklable.
    """
    def __init__(self, n_outputs):
        self.n_outputs = n_outputs
        self.classes_ = np.arange(n_outputs)

    def predict_proba(self, X):
        batch = X.shape[0] if len(X.shape) > 1 else 1
        return np.ones((batch, self.n_outputs)) / self.n_outputs


def make_dummy_classifier(n_outputs):
    return DummyClassifier(n_outputs)


# ================================================================
# DOWNLOAD FROM SUPABASE STORAGE
# ================================================================

def download_model_if_exists(filename):
    """
    Attempts to download a model file from Supabase Storage.
    Saves locally & returns True if found.
    """

    local_path = os.path.join(MODEL_DIR, filename)

    try:
        print(f"[MODELS] Checking Supabase for {filename}...")
        resp = supabase.storage.from_(BUCKET).download(filename)

        if resp is None:
            return False

        # Some SDK versions wrap bytes in {"data": bytes}
        if isinstance(resp, dict) and "data" in resp:
            resp = resp["data"]

        if isinstance(resp, (bytes, bytearray)):
            with open(local_path, "wb") as f:
                f.write(resp)
            print(f"[MODELS] Downloaded {filename}")
            return True

        return False

    except Exception as e:
        print(f"[MODELS] No Supabase version for {filename}: {e}")
        return False


# ================================================================
# UPLOAD TO SUPABASE STORAGE
# ================================================================

def upload_model_to_supabase(filename):
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
        print(f"[MODELS] Upload failed for {filename}: {e}")


# ================================================================
# LOAD / CREATE MODEL
# ================================================================

def load_or_create_model(path, outputs, filename):
    """
    1. Try load local model
    2. Try download from Supabase
    3. Fallback to dummy model
    4. Upload dummy to Supabase
    """

    # 1 — Local
    if os.path.exists(path):
        try:
            print(f"[MODELS] Loaded local model {filename}")
            return joblib.load(path)
        except Exception as e:
            print(f"[MODELS] Local model corrupted → {e}")

    # 2 — Supabase download
    if download_model_if_exists(filename):
        try:
            print(f"[MODELS] Loaded Supabase model {filename}")
            return joblib.load(path)
        except Exception as e:
            print(f"[MODELS] Supabase model corrupted → {e}")

    # 3 — Dummy fallback
    print(f"[MODELS] Creating dummy model: {filename}")
    model = make_dummy_classifier(outputs)

    joblib.dump(model, path)
    upload_model_to_supabase(filename)

    return model


# ================================================================
# INITIALIZE MODELS
# ================================================================

print("[MODELS] Initializing model system...")

model_1x2 = load_or_create_model(MODEL_1X2, 3, "logreg_1x2.pkl")
model_ou25 = load_or_create_model(MODEL_OU25, 2, "logreg_ou25.pkl")
model_btts = load_or_create_model(MODEL_BTTS, 2, "logreg_btts.pkl")

print("[MODELS] Models loaded successfully.")


# ================================================================
# NORMALIZATION UTIL
# ================================================================

def _normalize(lst):
    s = sum(lst)
    if s <= 0:
        return [1 / len(lst)] * len(lst)
    return [x / s for x in lst]


# ================================================================
# PREDICTION WRAPPERS
# ================================================================

def predict_1x2(features: np.ndarray):
    X = features.reshape(1, -1)

    try:
        raw = model_1x2.predict_proba(X)[0]
    except Exception as e:
        print("[MODELS] predict_1x2 error → dummy:", e)
        raw = [1/3, 1/3, 1/3]

    probs = _normalize([float(raw[0]), float(raw[1]), float(raw[2])])
    return {"home": probs[0], "draw": probs[1], "away": probs[2]}


def predict_ou25(features: np.ndarray):
    X = features.reshape(1, -1)

    try:
        raw = model_ou25.predict_proba(X)[0]
    except Exception as e:
        print("[MODELS] predict_ou25 error → dummy:", e)
        raw = [0.5, 0.5]

    probs = _normalize([float(raw[0]), float(raw[1])])
    return {"over": probs[0], "under": probs[1]}


def predict_btts(features: np.ndarray):
    X = features.reshape(1, -1)

    try:
        raw = model_btts.predict_proba(X)[0]
    except Exception as e:
        print("[MODELS] predict_btts error → dummy:", e)
        raw = [0.5, 0.5]

    probs = _normalize([float(raw[0]), float(raw[1])])
    return {"yes": probs[0], "no": probs[1]}
