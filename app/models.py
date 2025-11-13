import os
import joblib
import numpy as np
from supabase import create_client
from app.config import (
    MODEL_1X2,
    MODEL_OU25,
    MODEL_BTTS,
)

# -------------------------
# Supabase
# -------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Ensure folder exists
os.makedirs("models", exist_ok=True)


# -------------------------
# DOWNLOAD IF MISSING
# -------------------------
def download_model_if_missing(local_path: str, storage_name: str):
    if os.path.exists(local_path):
        return

    print(f"[MODELS] Missing {local_path}. Downloading from Supabase Storage...")
    try:
        data = supabase.storage.from_("models").download(storage_name)
        with open(local_path, "wb") as f:
            f.write(data)
        print(f"[MODELS] Saved {local_path}")
    except Exception as e:
        print(f"[MODELS] ERROR downloading {storage_name}: {e}")


download_model_if_missing(MODEL_1X2, "logreg_1x2.pkl")
download_model_if_missing(MODEL_OU25, "logreg_ou25.pkl")
download_model_if_missing(MODEL_BTTS, "logreg_btts.pkl")


# -------------------------
# LOAD MODELS
# -------------------------
def load_model(path: str):
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"[MODELS] Error loading model {path} -> {e}")
        return None


logreg_1x2 = load_model(MODEL_1X2)
logreg_ou25 = load_model(MODEL_OU25)
logreg_btts = load_model(MODEL_BTTS)


# -------------------------
# SIMPLE CALIBRATION (NO PyMC)
# -------------------------
def calibrate(p: float) -> float:
    return float(p)


# -------------------------
# PREDICTION INTERFACES
# -------------------------
def predict_1x2(features: np.ndarray):
    if logreg_1x2 is None:
        return {"home": 0.33, "draw": 0.33, "away": 0.34}

    raw = logreg_1x2.predict_proba(features.reshape(1, -1))[0]
    p_home = calibrate(raw[0])
    p_draw = calibrate(raw[1])
    p_away = calibrate(raw[2])

    total = p_home + p_draw + p_away
    return {
        "home": p_home / total,
        "draw": p_draw / total,
        "away": p_away / total,
    }


def predict_ou25(features: np.ndarray):
    if logreg_ou25 is None:
        return {"over": 0.5, "under": 0.5}

    raw = logreg_ou25.predict_proba(features.reshape(1, -1))[0]
    p_under = calibrate(raw[0])
    p_over = calibrate(raw[1])

    total = p_over + p_under
    return {"over": p_over / total, "under": p_under / total}


def predict_btts(features: np.ndarray):
    if logreg_btts is None:
        return {"yes": 0.5, "no": 0.5}

    raw = logreg_btts.predict_proba(features.reshape(1, -1))[0]
    p_no = calibrate(raw[0])
    p_yes = calibrate(raw[1])

    total = p_yes + p_no
    return {"yes": p_yes / total, "no": p_no / total}
