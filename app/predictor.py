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
    if odds <= 1.0:
        return -1.0
    return prob * odds - 1.0


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
    """
    features, aux = build_features_1x2(fixture, stats, odds_1x2, prematch_team_strength)
    probs = predict_1x2(features)

    preds: List[Prediction] = []

    # Home
    if odds_1x2.get("home"):
        p = probs["home"]
        o = float(odds_1x2["home"])
        preds.append(
            Prediction(
                fixture_id=fixture["fixture_id"],
                market="1X2",
                selection="home",
                prob=p,
                odds=o,
                ev=_ev(p, o),
                aux=aux,
            )
        )

    # Draw
    if odds_1x2.get("draw"):
        p = probs["draw"]
        o = float(odds_1x2["draw"])
        preds.append(
            Prediction(
                fixture_id=fixture["fixture_id"],
                market="1X2",
                selection="draw",
                prob=p,
                odds=o,
                ev=_ev(p, o),
                aux=aux,
            )
        )

    # Away
    if odds_1x2.get("away"):
        p = probs["away"]
        o = float(odds_1x2["away"])
        preds.append(
            Prediction(
                fixture_id=fixture["fixture_id"],
                market="1X2",
                selection="away",
                prob=p,
                odds=o,
                ev=_ev(p, o),
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
    features, aux = build_features_ou25(
        fixture, stats, odds_ou, prematch_goal_expectation
    )
    probs = predict_ou25(features)

    preds: List[Prediction] = []

    if odds_ou.get("over"):
        p = probs["over"]
        o = float(odds_ou["over"])
        preds.append(
            Prediction(
                fixture_id=fixture["fixture_id"],
                market="OU25",
                selection="over",
                prob=p,
                odds=o,
                ev=_ev(p, o),
                aux=aux,
            )
        )

    if odds_ou.get("under"):
        p = probs["under"]
        o = float(odds_ou["under"])
        preds.append(
            Prediction(
                fixture_id=fixture["fixture_id"],
                market="OU25",
                selection="under",
                prob=p,
                odds=o,
                ev=_ev(p, o),
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
    features, aux = build_features_btts(
        fixture, stats, odds_btts, prematch_btts_expectation
    )
    probs = predict_btts(features)

    preds: List[Prediction] = []

    if odds_btts.get("yes"):
        p = probs["yes"]
        o = float(odds_btts["yes"])
        preds.append(
            Prediction(
                fixture_id=fixture["fixture_id"],
                market="BTTS",
                selection="yes",
                prob=p,
                odds=o,
                ev=_ev(p, o),
                aux=aux,
            )
        )

    if odds_btts.get("no"):
        p = probs["no"]
        o = float(odds_btts["no"])
        preds.append(
            Prediction(
                fixture_id=fixture["fixture_id"],
                market="BTTS",
                selection="no",
                prob=p,
                odds=o,
                ev=_ev(p, o),
                aux=aux,
            )
        )

    return preds
