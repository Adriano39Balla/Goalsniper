import os
import pickle
import joblib
from typing import Dict, Any

import numpy as np
from app.config import (
    MODEL_1X2,
    MODEL_OU25,
    MODEL_BTTS,
    MODEL_DIR,
)
from app.supabase_db import get_supabase

sb = get_supabase()


def log(msg: str):
    print(f"[MODELS] {msg}")


# ---------------------------------------------------------
# DUMMY MODEL CLASSES (must be top-level for pickle!)
# ---------------------------------------------------------

class Dummy1X2:
    def predict_proba(self, X):
        p = np.array([[0.33, 0.34, 0.33]])
        return np.repeat(p, X.shape[0], axis=0)


class DummyOU25:
    def predict_proba(self, X):
        p = np.array([[0.50, 0.50]])
        return np.repeat(p, X.shape[0], axis=0)


class DummyBTTS:
    def predict_proba(self, X):
        p = np.array([[0.50, 0.50]])
        return np.repeat(p, X.shape[0], axis=0)


# ---------------------------------------------------------
# CREATE & SAVE DUMMY MODELS
# ---------------------------------------------------------

def create_dummy_model_1x2():
    log("Creating dummy 1X2 model...")
    joblib.dump(Dummy1X2(), MODEL_1X2)
    log(f"Dummy 1X2 saved: {MODEL_1X2}")


def create_dummy_model_ou25():
    log("Creating dummy OU25 model...")
    joblib.dump(DummyOU25(), MODEL_OU25)
    log(f"Dummy OU25 saved: {MODEL_OU25}")


def create_dummy_model_btts():
    log("Creating dummy BTTS model...")
    joblib.dump(DummyBTTS(), MODEL_BTTS)
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
    try:
        log(f"Downloading {remote_name} from Supabase Storage...")
        res = sb.storage.from_("models").download(remote_name)

        if not res:
            log(f"No file returned from storage for {remote_name}.")
            return False

        with open(local_path, "wb") as f:
            f.write(res)

        log(f"Downloaded model → {local_path}")
        return True

    except Exception as e:
        log(f"Failed to download {remote_name}: {e}")
        return False


def sync_models_from_storage():
    os.makedirs(MODEL_DIR, exist_ok=True)

    files = {
        "logreg_1x2.pkl": MODEL_1X2,
        "logreg_ou25.pkl": MODEL_OU25,
        "logreg_btts.pkl": MODEL_BTTS,
    }

    downloaded_any = False

    for remote, local in files.items():
        if download_from_storage(remote, local):
            downloaded_any = True

    if not downloaded_any:
        log("No trained models found → creating dummy models.")
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
# INITIALIZATION (AUTO RUN AT IMPORT)
# ---------------------------------------------------------

def initialize_models():
    log("Initializing model system...")
    os.makedirs(MODEL_DIR, exist_ok=True)

    missing = []
    for p in [MODEL_1X2, MODEL_OU25, MODEL_BTTS]:
        if not os.path.exists(p):
            missing.append(p)

    if missing:
        log(f"Missing model files: {missing}")
        sync_models_from_storage()

    # After sync, check again
    final_missing = []
    for p in [MODEL_1X2, MODEL_OU25, MODEL_BTTS]:
        if not os.path.exists(p):
            final_missing.append(p)

    if final_missing:
        log("Still missing → creating dummy models.")
        create_all_dummy_models()

    log("Model system ready.")


# Run initialization
initialize_models()

# Load into memory
MODEL_1X2_OBJ = safe_load(MODEL_1X2)
MODEL_OU25_OBJ = safe_load(MODEL_OU25)
MODEL_BTTS_OBJ = safe_load(MODEL_BTTS)


# ---------------------------------------------------------
# PUBLIC PREDICT FUNCTIONS
# ---------------------------------------------------------

def predict_1x2(features: np.ndarray) -> Dict[str, float]:
    proba = MODEL_1X2_OBJ.predict_proba(features.reshape(1, -1))[0]
    return {"home": float(proba[0]), "draw": float(proba[1]), "away": float(proba[2])}


def predict_ou25(features: np.ndarray) -> Dict[str, float]:
    proba = MODEL_OU25_OBJ.predict_proba(features.reshape(1, -1))[0]
    return {"under": float(proba[0]), "over": float(proba[1])}


def predict_btts(features: np.ndarray) -> Dict[str, float]:
    proba = MODEL_BTTS_OBJ.predict_proba(features.reshape(1, -1))[0]
    return {"no": float(proba[0]), "yes": float(proba[1])}
