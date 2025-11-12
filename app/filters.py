from typing import List
from app.predictor import Prediction
from app.config import (
    MIN_CONFIDENCE,
    MAX_CONFIDENCE,
    MIN_EV,
    MIN_ODDS,
)


# ------------------------------------------------------------
# HARD FILTERS (absolute minimum conditions)
# ------------------------------------------------------------

def hard_filter(pred: Prediction) -> bool:
    """
    Strict minimum conditions for a prediction to be considered at all.
    These should almost never be disabled.
    """
    if pred.odds < MIN_ODDS:
        return False

    if pred.prob < 0.02:  # 2% minimum sanity
        return False

    return True


# ------------------------------------------------------------
# ADAPTIVE THRESHOLDING
# ------------------------------------------------------------

def compute_adaptive_confidence_threshold(preds: List[Prediction]) -> float:
    """
    Adapt confidence threshold based on the quality distribution.
    If model output is weak -> relax.
    If model output is strong -> tighten.
    """

    if not preds:
        return MIN_CONFIDENCE

    # average confidence
    avg_conf = sum(p.prob for p in preds) / len(preds)

    # if model is very confident today
    if avg_conf >= 0.75:
        return min(MAX_CONFIDENCE, 0.80)

    # if model is weak today
    if avg_conf < 0.55:
        return max(0.50, MIN_CONFIDENCE - 0.10)  # relax by ~10%

    # normal day
    return MIN_CONFIDENCE


def compute_adaptive_ev_threshold(preds: List[Prediction]) -> float:
    """
    Adaptive EV threshold: prevent overblocking when model is low-edge.
    """

    if not preds:
        return MIN_EV

    avg_ev = sum(p.ev for p in preds) / len(preds)

    # If market EV is high across many matches -> be stricter
    if avg_ev >= 0.06:
        return max(MIN_EV, 0.05)

    # If market EV is low (quiet market) -> relax
    if avg_ev < 0.02:
        return 0.015

    # Default
    return MIN_EV


# ------------------------------------------------------------
# MAIN FILTER FUNCTION
# ------------------------------------------------------------

def filter_predictions(predictions: List[Prediction]) -> List[Prediction]:
    """
    Takes ALL predictions for ALL matches and returns the filtered subset.
    """

    # 1) Hard filters first
    filtered = [p for p in predictions if hard_filter(p)]
    if not filtered:
        return []

    # 2) Compute adaptive thresholds
    conf_thresh = compute_adaptive_confidence_threshold(filtered)
    ev_thresh = compute_adaptive_ev_threshold(filtered)

    # 3) Apply soft filters
    soft_filtered = []
    for p in filtered:
        if p.prob < conf_thresh:
            continue
        if p.ev < ev_thresh:
            continue
        soft_filtered.append(p)

    # 4) If nothing passed soft filters, fallback:
    # → return TOP 1 EV prediction so bot never goes silent.
    if not soft_filtered:
        best = max(filtered, key=lambda p: p.ev)
        return [best]

    return soft_filtered
