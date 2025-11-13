# app/models.py

import os
from typing import Dict
import numpy as np
from joblib import load
from app.config import MODEL_1X2, MODEL_OU25, MODEL_BTTS

_model_1x2 = None
_model_ou25 = None
_model_btts = None


def _load_model(path: str):
    if not os.path.exists(path):
        print(f"[MODELS] Missing model file: {path}")
        return None
    try:
        m = load(path)
        print(f"[MODELS] Loaded model: {path}")
        return m
    except Exception as e:
        print(f"[MODELS] Error loading {path}: {e}")
        return None


def init_models():
    global _model_1x2, _model_ou25, _model_btts

    if _model_1x2 is None:
        _model_1x2 = _load_model(MODEL_1X2)

    if _model_ou25 is None:
        _model_ou25 = _load_model(MODEL_OU25)

    if _model_btts is None:
        _model_btts = _load_model(MODEL_BTTS)

    if not all([_model_1x2, _model_ou25, _model_btts]):
        print("[MODELS] WARNING: one or more models missing. "
              "Dummy behaviour may occur until training runs.")


def predict_1x2(features: np.ndarray) -> Dict[str, float]:
    """
    Returns calibrated probs for home/draw/away as dict.
    """
    if _model_1x2 is None:
        init_models()
    if _model_1x2 is None:
        # fallback uniform if still missing
        return {"home": 1/3, "draw": 1/3, "away": 1/3}

    X = features.reshape(1, -1)
    proba = _model_1x2.predict_proba(X)[0]
    classes = list(_model_1x2.classes_)
    # classes correspond to: 0=home,1=draw,2=away as per training
    idx = {int(c): i for i, c in enumerate(classes)}

    return {
        "home": float(proba[idx[0]]),
        "draw": float(proba[idx[1]]),
        "away": float(proba[idx[2]]),
    }


def predict_ou25(features: np.ndarray) -> Dict[str, float]:
    """
    Returns probs for over / under 2.5.
    """
    if _model_ou25 is None:
        init_models()
    if _model_ou25 is None:
        return {"over": 0.5, "under": 0.5}

    X = features.reshape(1, -1)
    proba = _model_ou25.predict_proba(X)[0]  # [p_under, p_over] or similar
    classes = list(_model_ou25.classes_)
    idx = {int(c): i for i, c in enumerate(classes)}

    p_over = proba[idx[1]]      # class 1 => goals > 2.5
    p_under = 1.0 - p_over

    return {"over": float(p_over), "under": float(p_under)}


def predict_btts(features: np.ndarray) -> Dict[str, float]:
    """
    Returns probs for BTTS yes/no.
    """
    if _model_btts is None:
        init_models()
    if _model_btts is None:
        return {"yes": 0.5, "no": 0.5}

    X = features.reshape(1, -1)
   
