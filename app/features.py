from typing import Dict, Any, Optional, Tuple
import numpy as np
import math


def _safe_div(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return a / b


def implied_prob_from_odds(odds: Optional[float]) -> float:
    if not odds or odds <= 1.0:
        return 0.0
    return 1.0 / odds


def _minute_normalized(minute: Optional[int]) -> float:
    if minute is None:
        return 0.0
    return max(0.0, min(1.0, minute / 90.0))


def _goal_diff(home_goals: Optional[int], away_goals: Optional[int]) -> int:
    return (home_goals or 0) - (away_goals or 0)


# ------------------------------------------------------
# FEATURE BUILDER FOR 1X2 / MATCH RESULT
# ------------------------------------------------------

def build_features_1x2(
    fixture: Dict[str, Any],
    stats: Dict[str, Any],
    odds: Dict[str, Any],
    prematch_team_strength: Dict[str, float],
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Build feature vector for 1X2 (home/draw/away) prediction.

    Returns:
      - features: np.ndarray of shape (n_features,)
      - aux: dict with useful extras (current odds, implied probs, minute, etc.)
    """

    minute = fixture.get("minute") or 0
    home_goals = fixture.get("home_goals") or 0
    away_goals = fixture.get("away_goals") or 0
    goal_diff = _goal_diff(home_goals, away_goals)

    # Basic in-play stats (if available)
    home_shots_on = stats.get("home_shots_on", 0)
    away_shots_on = stats.get("away_shots_on", 0)
    total_shots_on = home_shots_on + away_shots_on

    home_attacks = stats.get("home_attacks", 0)
    away_attacks = stats.get("away_attacks", 0)
    total_attacks = home_attacks + away_attacks

    # Team strength from historical rating (prematch input)
    home_strength = prematch_team_strength.get("home", 0.0)
    away_strength = prematch_team_strength.get("away", 0.0)
    strength_diff = home_strength - away_strength

    # Odds (market 1X2 from some chosen bookmaker / avg)
    odds_home = odds.get("home")
    odds_draw = odds.get("draw")
    odds_away = odds.get("away")

    imp_home = implied_prob_from_odds(odds_home)
    imp_draw = implied_prob_from_odds(odds_draw)
    imp_away = implied_prob_from_odds(odds_away)

    overround = max(0.0, (imp_home + imp_draw + imp_away) - 1.0)

    # Normalize implied probs by overround (simple)
    if overround > 0:
        imp_home_adj = imp_home / (1.0 + overround)
        imp_draw_adj = imp_draw / (1.0 + overround)
        imp_away_adj = imp_away / (1.0 + overround)
    else:
        imp_home_adj, imp_draw_adj, imp_away_adj = imp_home, imp_draw, imp_away

    # In-play dominance ratios
    shots_ratio = _safe_div(home_shots_on, total_shots_on)
    attacks_ratio = _safe_div(home_attacks, total_attacks)

    # Time-based features
    minute_norm = _minute_normalized(minute)

    features = np.array(
        [
            # Scoreline
            home_goals,
            away_goals,
            goal_diff,
            # Team strength
            home_strength,
            away_strength,
            strength_diff,
            # Odds-based implied probs
            imp_home_adj,
            imp_draw_adj,
            imp_away_adj,
            overround,
            # In-play stats
            home_shots_on,
            away_shots_on,
            shots_ratio,
            home_attacks,
            away_attacks,
            attacks_ratio,
            # Time
            minute,
            minute_norm,
        ],
        dtype=float,
    )

    aux = {
        "odds_home": float(odds_home or 0.0),
        "odds_draw": float(odds_draw or 0.0),
        "odds_away": float(odds_away or 0.0),
        "imp_home": imp_home_adj,
        "imp_draw": imp_draw_adj,
        "imp_away": imp_away_adj,
        "minute": minute,
    }

    return features, aux


# ------------------------------------------------------
# FEATURE BUILDER FOR OVER/UNDER 2.5
# ------------------------------------------------------

def build_features_ou25(
    fixture: Dict[str, Any],
    stats: Dict[str, Any],
    odds: Dict[str, Any],
    prematch_goal_expectation: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Features for O/U 2.5 goals.
    """

    minute = fixture.get("minute") or 0
    home_goals = fixture.get("home_goals") or 0
    away_goals = fixture.get("away_goals") or 0
    goals_total = home_goals + away_goals

    home_shots_on = stats.get("home_shots_on", 0)
    away_shots_on = stats.get("away_shots_on", 0)
    total_shots_on = home_shots_on + away_shots_on

    # crude proxy for live xG pressure
    pressure_index = home_shots_on + away_shots_on + 0.5 * (stats.get("home_attacks", 0) + stats.get("away_attacks", 0))

    odds_over = odds.get("over")
    odds_under = odds.get("under")
    imp_over = implied_prob_from_odds(odds_over)
    imp_under = implied_prob_from_odds(odds_under)
    overround = max(0.0, (imp_over + imp_under) - 1.0)

    if overround > 0:
        imp_over_adj = imp_over / (1.0 + overround)
        imp_under_adj = imp_under / (1.0 + overround)
    else:
        imp_over_adj, imp_under_adj = imp_over, imp_under

    minute_norm = _minute_normalized(minute)

    features = np.array(
        [
            goals_total,
            home_goals,
            away_goals,
            prematch_goal_expectation,
            imp_over_adj,
            imp_under_adj,
            pressure_index,
            total_shots_on,
            minute,
            minute_norm,
        ],
        dtype=float,
    )

    aux = {
        "odds_over": float(odds_over or 0.0),
        "odds_under": float(odds_under or 0.0),
        "imp_over": imp_over_adj,
        "imp_under": imp_under_adj,
        "minute": minute,
    }

    return features, aux


# ------------------------------------------------------
# FEATURE BUILDER FOR BTTS (Both Teams To Score)
# ------------------------------------------------------

def build_features_btts(
    fixture: Dict[str, Any],
    stats: Dict[str, Any],
    odds: Dict[str, Any],
    prematch_btts_expectation: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Features for BTTS yes/no.
    """

    minute = fixture.get("minute") or 0
    home_goals = fixture.get("home_goals") or 0
    away_goals = fixture.get("away_goals") or 0

    home_shots_on = stats.get("home_shots_on", 0)
    away_shots_on = stats.get("away_shots_on", 0)
    total_shots_on = home_shots_on + away_shots_on

    odds_yes = odds.get("yes")
    odds_no = odds.get("no")
    imp_yes = implied_prob_from_odds(odds_yes)
    imp_no = implied_prob_from_odds(odds_no)
    overround = max(0.0, (imp_yes + imp_no) - 1.0)

    if overround > 0:
        imp_yes_adj = imp_yes / (1.0 + overround)
        imp_no_adj = imp_no / (1.0 + overround)
    else:
        imp_yes_adj, imp_no_adj = imp_yes, imp_no

    minute_norm = _minute_normalized(minute)

    features = np.array(
        [
            home_goals > 0,
            away_goals > 0,
            home_goals + away_goals,
            home_shots_on,
            away_shots_on,
            total_shots_on,
            prematch_btts_expectation,
            imp_yes_adj,
            imp_no_adj,
            minute,
            minute_norm,
        ],
        dtype=float,
    )

    aux = {
        "odds_yes": float(odds_yes or 0.0),
        "odds_no": float(odds_no or 0.0),
        "imp_yes": imp_yes_adj,
        "imp_no": imp_no_adj,
        "minute": minute,
    }

    return features, aux
