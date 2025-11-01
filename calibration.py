# calibration.py

import logging
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from typing import Optional, Tuple

log = logging.getLogger(__name__)

def calibrate_probabilities(
    train_probs: np.ndarray,
    y_train: np.ndarray,
    test_probs: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[dict, Optional[np.ndarray]]:
    """
    Calibrates prediction probabilities using Platt scaling (logistic regression).
    Returns a calibration dictionary and optionally calibrated probabilities.
    """
    calibration = {"method": "sigmoid", "a": 1.0, "b": 0.0}
    calibrated_probs = None

    try:
        if len(y_test) <= 50 or len(np.unique(y_train)) < 2:
            return calibration, None

        calibrator = LogisticRegression(max_iter=1000, C=1.0)
        calibrator.fit(train_probs.reshape(-1, 1), y_train)

        # Apply to test set
        calibrated_probs = calibrator.predict_proba(test_probs.reshape(-1, 1))[:, 1]
        orig_brier = brier_score_loss(y_test, test_probs)
        cal_brier = brier_score_loss(y_test, calibrated_probs)

        if cal_brier < orig_brier:
            calibration = {
                "method": "sigmoid",
                "a": float(calibrator.coef_[0][0]),
                "b": float(calibrator.intercept_[0]),
            }
        else:
            calibrated_probs = None  # Keep original if calibration worse

    except Exception as e:
        log.debug("Calibration failed: %s", e)

    return calibration, calibrated_probs
