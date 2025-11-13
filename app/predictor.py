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
    market: str
    selection: str
    prob: float
    odds: float
    ev: float
    aux: Dict[str, Any]


# ------------------------------------------------------------
# SAFE HELPERS
# ------------------------------------------------------------

def _safe_float(x):
    try:
        if x is None:
            return None
        return float(x)
    except:
        return None


def _ev(prob: float, odds: float) -> float:
    """Expected value from bettor perspective."""
    if odds is None or odds <= 1.0:
        return -1.0
    return prob * odds - 1.0


def _safe_predict(fn, features, neutral_probs):
    """
    Calls a model prediction function safely.
    If model is missing, returns neutral_probs.
    """
    try:
        return fn(features)
    except Exception:
        return neutral_probs


# ##########################################################################
#                                1X2 MODEL
# ##########################################################################

def predict_fixture_1x2(
    fixture: Dict[str, Any],
    stats: Dict[str, Any],
    odds_1x2: Dict[str, float],
    prematch_team_strength: Dict[str, float],
) -> List[Prediction]:

    odd_home = _safe_float(odds_1x2.get("home"))
    odd_draw = _safe_float(odds_1x2.get("draw"))
    odd_away = _safe_float(odds_1x2.get("away"))

    # Require at least home + away
    if not odd_home or not odd_away:
        return []

    # Build features
    features, aux = build_features_1x2(
        fixture, stats, odds_1x2, prematch_team_strength
    )

    # Neutral fallback in case model missing or broken:
    neutral = {"home": 0.33, "draw": 0.33, "away": 0.33}

    probs = _safe_predict(predict_1x2, features, neutral)

    preds = []

    # HOME
    if odd_home:
        p = probs.get("home", 0.0)
        preds.append(Prediction(
            fixture_id=fixture["fixture_id"],
            market="1X2",
            selection="home",
            prob=p,
            odds=odd_home,
            ev=_ev(p, odd_home),
            aux=aux,
        ))

    # DRAW
    if odd_draw:
        p = probs.get("draw", 0.0)
        preds.append(Prediction(
            fixture_id=fixture["fixture_id"],
            market="1X2",
            selection="draw",
            prob=p,
            odds=odd_draw,
            ev=_ev(p, odd_draw),
            aux=aux,
        ))

    # AWAY
    if odd_away:
        p = probs.get("away", 0.0)
        preds.append(Prediction(
            fixture_id=fixture["fixture_id"],
            market="1X2",
            selection="away",
            prob=p,
            odds=odd_away,
            ev=_ev(p, odd_away),
            aux=aux,
        ))

    return preds


# ##########################################################################
#                           OVER / UNDER 2.5 MODEL
# ##########################################################################

def predict_fixture_ou25(
    fixture: Dict[str, Any],
    stats: Dict[str, Any],
    odds_ou: Dict[str, float],
    prematch_goal_expectation: float,
) -> List[Prediction]:

    odd_over = _safe_float(odds_ou.get("over"))
    odd_under = _safe_float(odds_ou.get("under"))

    if not odd_over and not odd_under:
        return []

    features, aux = build_features_ou25(
        fixture, stats, odds_ou, prematch_goal_expectation
    )

    neutral = {"over": 0.5, "under": 0.5}
    probs = _safe_predict(predict_ou25, features, neutral)

    preds = []

    if odd_over:
        p = probs.get("over", 0.0)
        preds.append(Prediction(
            fixture_id=fixture["fixture_id"],
            market="OU25",
            selection="over",
            prob=p,
            odds=odd_over,
            ev=_ev(p, odd_over),
            aux=aux,
        ))

    if odd_under:
        p = probs.get("under", 0.0)
        preds.append(Prediction(
            fixture_id=fixture["fixture_id"],
            market="OU25",
            selection="under",
            prob=p,
            odds=odd_under,
            ev=_ev(p, odd_under),
            aux=aux,
        ))

    return preds


# ##########################################################################
#                                BTTS MODEL
# ##########################################################################

def predict_fixture_btts(
    fixture: Dict[str, Any],
    stats: Dict[str, Any],
    odds_btts: Dict[str, float],
    prematch_btts_expectation: float,
) -> List[Prediction]:

    odd_yes = _safe_float(odds_btts.get("yes"))
    odd_no = _safe_float(odds_btts.get("no"))

    if not odd_yes and not odd_no:
        return []

    features, aux = build_features_btts(
        fixture, stats, odds_btts, prematch_btts_expectation
    )

    neutral = {"yes": 0.5, "no": 0.5}
    probs = _safe_predict(predict_btts, features, neutral)

    preds = []

    if odd_yes:
        p = probs.get("yes", 0.0)
        preds.append(Prediction(
            fixture_id=fixture["fixture_id"],
            market="BTTS",
            selection="yes",
            prob=p,
            odds=odd_yes,
            ev=_ev(p, odd_yes),
            aux=aux,
        ))

    if odd_no:
        p = probs.get("no", 0.0)
        preds.append(Prediction(
            fixture_id=fixture["fixture_id"],
            market="BTTS",
            selection="no",
            prob=p,
            odds=odd_no,
            ev=_ev(p, odd_no),
            aux=aux,
        ))

    return preds
