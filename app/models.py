import joblib
import numpy as np
from typing import Dict, Any
import pymc as pm
import aesara.tensor as at
from app.config import (
    MODEL_1X2,
    MODEL_OU25,
    MODEL_BTTS,
    MODEL_CALIBRATION,
)


# ---------------------------------------------------------
# LOADING LOGISTIC REGRESSION MODELS
# ---------------------------------------------------------

def load_logreg(path: str):
    """
    Load a logistic regression (sklearn) model from disk.
    """
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"[MODELS] Error loading model: {path} -> {e}")
        return None


def load_calibration_model(path: str):
    """
    Load Bayesian calibration model (PyMC).
    """
    try:
        return pm.load_trace(path)
    except Exception as e:
        print(f"[MODELS] Could not load calibration model: {e}")
        return None


# ---------------------------------------------------------
# INITIALIZE MODELS ON IMPORT (so main.py is fast)
# ---------------------------------------------------------

logreg_1x2 = load_logreg(MODEL_1X2)
logreg_ou25 = load_logreg(MODEL_OU25)
logreg_btts = load_logreg(MODEL_BTTS)

calibration_trace = load_calibration_model(MODEL_CALIBRATION)


# ---------------------------------------------------------
# BAYESIAN CALIBRATION FUNCTION
# ---------------------------------------------------------

def bayesian_calibrate(raw_p: float) -> float:
    """
    Adjust the model probability using Bayesian calibration.
    If no calibration model exists, return raw.
    """
    if calibration_trace is None:
        return float(raw_p)

    # Example minimal Bayesian calibration model:
    # p_calibrated = a * raw_p + b, where a,b ~ posterior from PyMC
    a_samples = calibration_trace.posterior["a"].values.flatten()
    b_samples = calibration_trace.posterior["b"].values.flatten()

    a = float(np.mean(a_samples))
    b = float(np.mean(b_samples))

    calibrated = a * raw_p + b
    return float(max(0.0, min(1.0, calibrated)))  # clip to [0,1]


# ---------------------------------------------------------
# PREDICTION FUNCTIONS
# ---------------------------------------------------------

def predict_1x2(features: np.ndarray) -> Dict[str, float]:
    """
    Predict {home, draw, away} probabilities.
    """
    if logreg_1x2 is None:
        return {"home": 0.33, "draw": 0.33, "away": 0.34}

    # raw probabilities from model
    raw = logreg_1x2.predict_proba(features.reshape(1, -1))[0]

    # calibrate each class independently
    p_home = bayesian_calibrate(raw[0])
    p_draw = bayesian_calibrate(raw[1])
    p_away = bayesian_calibrate(raw[2])

    # renormalize
    total = p_home + p_draw + p_away
    if total > 0:
        p_home /= total
        p_draw /= total
        p_away /= total

    return {"home": p_home, "draw": p_draw, "away": p_away}


def predict_ou25(features: np.ndarray) -> Dict[str, float]:
    """
    Predict Over/Under 2.5.
    """
    if logreg_ou25 is None:
        return {"over": 0.5, "under": 0.5}

    raw = logreg_ou25.predict_proba(features.reshape(1, -1))[0]
    p_over = bayesian_calibrate(raw[1])   # positive class
    p_under = bayesian_calibrate(raw[0])

    total = p_over + p_under
    return {"over": p_over / total, "under": p_under / total}


def predict_btts(features: np.ndarray) -> Dict[str, float]:
    """
    Predict BTTS yes/no.
    """
    if logreg_btts is None:
        return {"yes": 0.5, "no": 0.5}

    raw = logreg_btts.predict_proba(features.reshape(1, -1))[0]
    p_yes = bayesian_calibrate(raw[1])
    p_no = bayesian_calibrate(raw[0])

    total = p_yes + p_no
    return {"yes": p_yes / total, "no": p_no / total}
