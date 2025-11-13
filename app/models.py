import os
import pickle
import joblib
from typing import Dict, Any

from sklearn.linear_model import LogisticRegression
import numpy as np

from app.config import (
    MODEL_1X2,
    MODEL_OU25,
    MODEL_BTTS,
    MODEL_DIR,
)
from app.supabase_db import get_supabase

sb = get_supabase()


# ---------------------------------------------------------
# LOGGING HELPERS
# ---------------------------------------------------------

def log(msg: str):
    print(f"[MODELS] {msg}")


# ---------------------------------------------------------
# CREATE DUMMY MODELS (FALLBACK)
# ---------------------------------------------------------

def create_dummy_model_1x2():
    """
    Simple 3-class logistic regression-like dummy.
    Output ~0.33 / 0.33 / 0.33 always.
    """
    log("Creating dummy 1X2 model...")

    class Dummy1X2:
        def predict_proba(self, X):
            p = np.array([[0.33, 0.34, 0.33]])
            return np.repeat(p, X.shape[0], axis=0)

    model = Dummy1X2()
    joblib.dump(model, MODEL_1X2)
    log(f"Dummy 1X2 saved: {MODEL_1X2}")


def create_dummy_model_ou25():
    """
    Dummy binary classifier for Over/Under 2.5
    Output ~50/50.
    """
    log("Creating dummy OU25 model...")

    class DummyOU25:
        def predict_proba(self, X):
            p = np.array([[0.50, 0.50]])
            return np.repeat(p, X.shape[0], axis=0)

    model = DummyOU25()
    joblib.dump(model, MODEL_OU25)
    log(f"Dummy OU25 saved: {MODEL_OU25}")


def create_dummy_model_btts():
    """
    Dummy BTTS classifier.
    """
    log("Creating dummy BTTS model...")

    class DummyBTTS:
        def predict_proba(self, X):
            p = np.array([[0.50, 0.50]])
            return np.repeat(p, X.shape[0], axis=0)

    model = DummyBTTS()
    joblib.dump(model, MODEL_BTTS)
    log(f"Dummy BTTS saved: {MODEL_BTTS}")


def create_all_dummy_models():
    create_dummy_model_1x2()
    create_dummy_model_ou25()
    create_dummy_model_btts()
    log("All dummy models created.")


# ---------------------------------------------------------
# SUPABASE STORAGE SYNC
# ---------------------------------------------------------

def download_from_storage(remote_name: str, local_path: str) -> bool:
    """
    Attempt to download a trained model from Supabase Storage.
    Returns True if download successful.
    """
    try:
        log(f"Downloading {remote_name} from Supabase Storage...")

        res = sb.storage.from_("models").download(remote_name)
        if not res:
            log(f"Storage returned empty for {remote_name}.")
            return False

        with open(local_path, "wb") as f:
            f.write(res)

        log(f"Downloaded model → {local_path}")
        return True

    except Exception as e:
        log(f"Failed to download {remote_name}: {e}")
        return False


def sync_models_from_storage():
    """
    Attempts to download trained models.
    Falls back to dummy models if nothing exists yet.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    files = {
        "logreg_1x2.pkl": MODEL_1X2,
        "logreg_ou25.pkl": MODEL_OU25,
        "logreg_btts.pkl": MODEL_BTTS,
    }

    downloaded_any = False

    for remote, local in files.items():
        ok = download_from_storage(remote, local)
        if ok:
            downloaded_any = True

    if not downloaded_any:
        log("No trained models found in storage → using dummy models.")
        create_all_dummy_models()


# ---------------------------------------------------------
# LOAD MODEL HELPERS
# ---------------------------------------------------------

def safe_load(path: str):
    try:
        return joblib.load(path)
    except Exception as e:
        log(f"Error loading {path}: {e}")
        return None


# ---------------------------------------------------------
# INITIALIZATION (RUNS ON IMPORT)
# ---------------------------------------------------------

def initialize_models():
    log("Initializing model system...")

    # Ensure folder exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Check if all models exist locally
    missing = []
    for p in [MODEL_1X2, MODEL_OU25, MODEL_BTTS]:
        if not os.path.exists(p):
            missing.append(p)

    if missing:
        log(f"Missing model files: {missing}")
        log("Trying to sync from Supabase...")
        sync_models_from_storage()

    # Final check: create dummy if still missing
    final_missing = []
    for p in [MODEL_1X2, MODEL_OU25, MODEL_BTTS]:
        if not os.path.exists(p):
            final_missing.append(p)

    if final_missing:
        log(f"Final missing models → creating dummy: {final_missing}")
        create_all_dummy_models()

    log("Model system ready.")


# ---------------------------------------------------------
# MODEL PREDICT FUNCTIONS
# ---------------------------------------------------------

# We load models after initialization
initialize_models()

MODEL_1X2_OBJ = safe_load(MODEL_1X2)
MODEL_OU25_OBJ = safe_load(MODEL_OU25)
MODEL_BTTS_OBJ = safe_load(MODEL_BTTS)


def predict_1x2(features: np.ndarray) -> Dict[str, float]:
    """
    Returns: {"home": p1, "draw": p2, "away": p3}
    """
    proba = MODEL_1X2_OBJ.predict_proba(features.reshape(1, -1))[0]
    return {"home": float(proba[0]), "draw": float(proba[1]), "away": float(proba[2])}


def predict_ou25(features: np.ndarray) -> Dict[str, float]:
    proba = MODEL_OU25_OBJ.predict_proba(features.reshape(1, -1))[0]
    return {"under": float(proba[0]), "over": float(proba[1])}


def predict_btts(features: np.ndarray) -> Dict[str, float]:
    proba = MODEL_BTTS_OBJ.predict_proba(features.reshape(1, -1))[0]
    return {"no": float(proba[0]), "yes": float(proba[1])}
