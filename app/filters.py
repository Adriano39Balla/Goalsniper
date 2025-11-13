from typing import List
from app.predictor import Prediction
from app.config import (
    MIN_CONFIDENCE,
    MAX_CONFIDENCE,
    MIN_EV,
    MIN_ODDS,
    SAFE_MODE,
)


# ------------------------------------------------------------
# HARD FILTERS
# ------------------------------------------------------------

def hard_filter(pred: Prediction) -> bool:
    """
    Hard filters eliminate predictions that make no sense.
    SAFE_MODE bypasses all strict conditions (for dummy models).
    """

    if SAFE_MODE:
        return True  # bypass everything when SAFE_MODE=true

    # odds sanity
    if pred.odds is None or pred.odds < MIN_ODDS:
        return False

    # prob sanity (should never be THAT low)
    if pred.prob < 0.02:
        return False

    return True


# ------------------------------------------------------------
# ADAPTIVE CONFIDENCE THRESHOLD
# ------------------------------------------------------------

def compute_adaptive_confidence_threshold(preds: List[Prediction]) -> float:
    if SAFE_MODE:
        return 0.00  # disable confidence threshold

    if not preds:
        return MIN_CONFIDENCE

    avg_conf = sum(p.prob for p in preds) / len(preds)

    # very strong model output → tighten
    if avg_conf >= 0.75:
        return min(MAX_CONFIDENCE, 0.80)

    # weak day → relax threshold
    if avg_conf < 0.55:
        return max(0.50, MIN_CONFIDENCE - 0.10)

    # normal day
    return MIN_CONFIDENCE


# ------------------------------------------------------------
# ADAPTIVE EV THRESHOLD
# ------------------------------------------------------------

def compute_adaptive_ev_threshold(preds: List[Prediction]) -> float:
    if SAFE_MODE:
        return -999.0  # allow ALL EV (even negative)

    if not preds:
        return MIN_EV

    avg_ev = sum(p.ev for p in preds) / len(preds)

    # hot day → EV high → require more edge
    if avg_ev >= 0.06:
        return max(MIN_EV, 0.05)

    # quiet market → relax
    if avg_ev < 0.02:
        return 0.015

    # normal
    return MIN_EV


# ------------------------------------------------------------
# MAIN FILTER LOGIC
# ------------------------------------------------------------

def filter_predictions(predictions: List[Prediction]) -> List[Prediction]:
    """
    Applies:
        1) Hard filters
        2) Adaptive thresholding
        3) Soft filters
        4) Fallback: never go silent
    """

    if SAFE_MODE:
        # If SAFE_MODE = true → bypass everything
        if predictions:
            # Always send the highest EV prediction
            return [max(predictions, key=lambda p: p.ev)]
        return []

    # ----------------------
    # 1) HARD FILTERS
    # ----------------------
    filtered = [p for p in predictions if hard_filter(p)]
    if not filtered:
        return []

    # ----------------------
    # 2) ADAPTIVE THRESHOLDS
    # ----------------------
    conf_thresh = compute_adaptive_confidence_threshold(filtered)
    ev_thresh = compute_adaptive_ev_threshold(filtered)

    # ----------------------
    # 3) SOFT FILTERS
    # ----------------------
    soft_filtered = []
    for p in filtered:
        if p.prob < conf_thresh:
            continue
        if p.ev < ev_thresh:
            continue
        soft_filtered.append(p)

    # ----------------------
    # 4) FALLBACK
    # ----------------------
    if not soft_filtered:
        # Return the best EV candidate so Telegram never stays silent
        best = max(filtered, key=lambda p: p.ev)
        return [best]

    return soft_filtered
