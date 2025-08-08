import logging
from typing import Dict, List
from config import Config

logger = logging.getLogger(__name__)

class PredictionEngine:
    def __init__(self):
        self.min_confidence = Config.MIN_CONFIDENCE_THRESHOLD
        self.xg_threshold = 0.9  # Safer default threshold

    def analyze_match(self, match: Dict, stats: Dict) -> List[Dict]:
        """Analyze a live match and return a list of prediction dicts"""
        try:
            predictions = []

            elapsed = match.get("elapsed", 0)
            status = match.get("status", "")

            if not stats or elapsed is None:
                return []

            # Combine stats for simplicity
            combined_xg = self._get_xg(stats)
            total_goals = match.get("home_score", 0) + match.get("away_score", 0)

            confidence = self._calculate_confidence(combined_xg, elapsed, total_goals)

            if confidence >= self.min_confidence:
                predictions.append({
                    "match_id": match["id"],
                    "home_team": match["home_team"],
                    "away_team": match["away_team"],
                    "league": match["league"],
                    "league_id": match["league_id"],
                    "type": "over_2.5_goals",
                    "confidence": confidence,
                    "reasoning": f"Total xG is {combined_xg:.2f} with {total_goals} goals already scored at {elapsed} mins.",
                })

            return predictions

        except Exception as e:
            logger.error(f"Prediction analysis failed: {str(e)}")
            return []

    def _get_xg(self, stats: Dict) -> float:
        """Estimate xG from team stats"""
        try:
            total_xg = 0.0
            for team_stats in stats.values():
                xg = team_stats.get("Expected Goals", 0)
                if isinstance(xg, (int, float)):
                    total_xg += xg
            return round(total_xg, 2)
        except Exception as e:
            logger.error(f"Error extracting xG: {str(e)}")
            return 0.0

    def _calculate_confidence(self, total_xg: float, elapsed: int, goals: int) -> float:
        """Generate confidence score"""
        try:
            # Remove fixed time restrictions (now any match time is allowed)
            if total_xg < self.xg_threshold:
                return 0.0

            # Base confidence on xG and elapsed time
            base = min(total_xg / 3.5, 1.0)  # Normalize xG
            time_factor = min(elapsed / 90, 1.0)
            goal_factor = 0.1 * goals  # small boost per goal

            confidence = base * 0.6 + time_factor * 0.3 + goal_factor * 0.1
            return round(min(confidence, 1.0), 2)

        except Exception as e:
            logger.error(f"Confidence calculation failed: {str(e)}")
            return 0.0
