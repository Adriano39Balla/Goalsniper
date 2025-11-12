import joblib
import numpy as np
from typing import Dict, Any
from app.config import (
    MODEL_1X2,
    MODEL_OU25,
    MODEL_BTTS,
)

# ---------------------------------------------------------
# LOADING LOGISTIC REGRESSION MODELS
# ---------------------------------------------------------

def load_logreg(path: str):
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"[MODELS] Error loading model: {path} -> {e}")
        return None

# Load the trained models (or None if missing)
logreg_1x2 = load_logreg(MODEL_1X2)
logreg_ou25 = load_logreg(MODEL_OU25)
logreg_btts = load_logreg(MODEL_BTTS)

# ---------------------------------------------------------
# CALIBRATION (DISABLED)
# ---------------------------------------------------------

def bayesian_calibrate(raw_p: float) -> float:
    """
    Placeholder calibration — returns raw probability.
    """
    return float(raw_p)

# ---------------------------------------------------------
# PREDICTION FUNCTIONS
# ---------------------------------------------------------

def predict_1x2(features: np.ndarray) -> Dict[str, float]:
    if logreg_1x2 is None:
        return {"home": 0.33, "draw": 0.33, "away": 0.34}

    raw = logreg_1x2.predict_proba(features.reshape(1, -1))[0]
    p_home = bayesian_calibrate(raw[0])
    p_draw = bayesian_calibrate(raw[1])
    p_away = bayesian_calibrate(raw[2])

    total = p_home + p_draw + p_away
    return {
        "home": p_home / total,
        "draw": p_draw / total,
        "away": p_away / total,
    }

def predict_ou25(features: np.ndarray) -> Dict[str, float]:
    if logreg_ou25 is None:
        return {"over": 0.5, "under": 0.5}

    raw = logreg_ou25.predict_proba(features.reshape(1, -1))[0]
    p_under = bayesian_calibrate(raw[0])
    p_over = bayesian_calibrate(raw[1])
    total = p_over + p_under
    return {"over": p_over / total, "under": p_under / total}

def predict_btts(features: np.ndarray) -> Dict[str, float]:
    if logreg_btts is None:
        return {"yes": 0.5, "no": 0.5}

    raw = logreg_btts.predict_proba(features.reshape(1, -1))[0]
    p_no = bayesian_calibrate(raw[0])
    p_yes = bayesian_calibrate(raw[1])
    total = p_yes + p_no
    return {"yes": p_yes / total, "no": p_no / total}
