from typing import Dict, Any, List
from dataclasses import dataclass
import numpy as np

from app.features import (
    build_features_1x2,
    build_features_ou25,
    build_features_btts,
)
from app.models import (
    predict_1x2,
    predict_ou25,
    predict_btts,
)


@dataclass
class Prediction:
    fixture_id: int
    market: str         # "1X2" | "OU25" | "BTTS"
    selection: str      # e.g. "home", "over", "yes"
    prob: float         # model probability (calibrated)
    odds: float         # market odds
    ev: float           # expected value (prob * odds - 1)
    aux: Dict[str, Any] # extra info (minute, implied probs, etc.)


def _ev(prob: float, odds: float) -> float:
    """
    EV from bettor perspective: prob * odds - 1
    """
    if odds is None or odds <= 1.0:
        return -1.0
    return prob * odds - 1.0


def _safe_float(x):
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


# ----------------------------------------------------------------
# 1X2 PREDICTION
# ----------------------------------------------------------------

def predict_fixture_1x2(
    fixture: Dict[str, Any],
    stats: Dict[str, Any],
    odds_1x2: Dict[str, float],
    prematch_team_strength: Dict[str, float],
) -> List[Prediction]:
    """
    Returns up to 3 predictions (home/draw/away) with EV values.
    If odds are missing or broken, returns [].
    """

    # Validate odds
    odd_home = _safe_float(odds_1x2.get("home"))
    odd_draw = _safe_float(odds_1x2.get("draw"))
    odd_away = _safe_float(odds_1x2.get("away"))

    # Require at least home & away to make sense
    if not odd_home or not odd_away:
        return []

    # Build features
    features, aux = build_features_1x2(
        fixture, stats, odds_1x2, prematch_team_strength
    )

    probs = predict_1x2(features)
    preds: List[Prediction] = []

    # Home
    if odd_home:
        p = probs["home"]
        preds.append(
            Prediction(
                fixture_id=fixture["fixture_id"],
                market="1X2",
                selection="home",
                prob=p,
                odds=odd_home,
                ev=_ev(p, odd_home),
                aux=aux,
            )
        )

    # Draw (optional if we got it)
    if odd_draw:
        p = probs["draw"]
        preds.append(
            Prediction(
                fixture_id=fixture["fixture_id"],
                market="1X2",
                selection="draw",
                prob=p,
                odds=odd_draw,
                ev=_ev(p, odd_draw),
                aux=aux,
            )
        )

    # Away
    if odd_away:
        p = probs["away"]
        preds.append(
            Prediction(
                fixture_id=fixture["fixture_id"],
                market="1X2",
                selection="away",
                prob=p,
                odds=odd_away,
                ev=_ev(p, odd_away),
                aux=aux,
            )
        )

    return preds


# ----------------------------------------------------------------
# OU 2.5 PREDICTION
# ----------------------------------------------------------------

def predict_fixture_ou25(
    fixture: Dict[str, Any],
    stats: Dict[str, Any],
    odds_ou: Dict[str, float],
    prematch_goal_expectation: float,
) -> List[Prediction]:
    """
    Returns predictions for Over/Under 2.5 goals.
    If over/under odds missing, returns [].
    """

    odd_over = _safe_float(odds_ou.get("over"))
    odd_under = _safe_float(odds_ou.get("under"))

    if not odd_over and not odd_under:
        return []

    features, aux = build_features_ou25(
        fixture, stats, odds_ou, prematch_goal_expectation
    )
    probs = predict_ou25(features)

    preds: List[Prediction] = []

    if odd_over:
        p = probs["over"]
        preds.append(
            Prediction(
                fixture_id=fixture["fixture_id"],
                market="OU25",
                selection="over",
                prob=p,
                odds=odd_over,
                ev=_ev(p, odd_over),
                aux=aux,
            )
        )

    if odd_under:
        p = probs["under"]
        preds.append(
            Prediction(
                fixture_id=fixture["fixture_id"],
                market="OU25",
                selection="under",
                prob=p,
                odds=odd_under,
                ev=_ev(p, odd_under),
                aux=aux,
            )
        )

    return preds


# ----------------------------------------------------------------
# BTTS PREDICTION
# ----------------------------------------------------------------

def predict_fixture_btts(
    fixture: Dict[str, Any],
    stats: Dict[str, Any],
    odds_btts: Dict[str, float],
    prematch_btts_expectation: float,
) -> List[Prediction]:
    """
    Returns predictions for BTTS yes / no.
    If yes/no odds missing, returns [].
    """

    odd_yes = _safe_float(odds_btts.get("yes"))
    odd_no = _safe_float(odds_btts.get("no"))

    if not odd_yes and not odd_no:
        return []

    features, aux = build_features_btts(
        fixture, stats, odds_btts, prematch_btts_expectation
    )
    probs = predict_btts(features)

    preds: List[Prediction] = []

    if odd_yes:
        p = probs["yes"]
        preds.append(
            Prediction(
                fixture_id=fixture["fixture_id"],
                market="BTTS",
                selection="yes",
                prob=p,
                odds=odd_yes,
                ev=_ev(p, odd_yes),
                aux=aux,
            )
        )

    if odd_no:
        p = probs["no"]
        preds.append(
            Prediction(
                fixture_id=fixture["fixture_id"],
                market="BTTS",
                selection="no",
                prob=p,
                odds=odd_no,
                ev=_ev(p, odd_no),
                aux=aux,
            )
        )

    return preds
