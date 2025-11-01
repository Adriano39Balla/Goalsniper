import logging
from typing import Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

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
        # ───────── Input checks ───────── #
        if any(arr is None for arr in [train_probs, y_train, test_probs, y_test]):
            raise ValueError("Missing required inputs for calibration.")

        if len(y_test) <= 50 or len(np.unique(y_train)) < 2:
            log.debug("Calibration skipped: not enough test samples or class variety.")
            return calibration, None

        # ───────── Train calibration model ───────── #
        calibrator = LogisticRegression(max_iter=1000, C=1.0)
        calibrator.fit(train_probs.reshape(-1, 1), y_train)

        # ───────── Apply calibration ───────── #
        calibrated_probs = calibrator.predict_proba(test_probs.reshape(-1, 1))[:, 1]

        # ───────── Brier score comparison ───────── #
        orig_brier = brier_score_loss(y_test, test_probs)
        cal_brier = brier_score_loss(y_test, calibrated_probs)

        if cal_brier < orig_brier:
            calibration = {
                "method": "sigmoid",
                "a": float(calibrator.coef_[0][0]),
                "b": float(calibrator.intercept_[0]),
            }
        else:
            log.debug("Calibration discarded: Brier score not improved.")
            calibrated_probs = None

    except Exception as e:
        log.exception("Calibration failed: %s", e)

    return calibration, calibrated_probs
