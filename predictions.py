# predictions.py
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def calculate_confidence_and_suggestion(match):
    """Analyze match stats and produce bet suggestion with confidence."""
    stats_map = {}
    for team_data in match["stats"]:
        team_name = team_data["team"]["name"]
        stats_map[team_name] = {stat["type"]: stat["value"] for stat in team_data["statistics"]}

    if match["home"] not in stats_map or match["away"] not in stats_map:
        return None

    home_stats = stats_map[match["home"]]
    away_stats = stats_map[match["away"]]

    def extract(team_stats, key, cast_type=float):
        val = team_stats.get(key, 0)
        try:
            return cast_type(str(val).replace("%", "").strip() or 0)
        except:
            return 0

    home_shots = extract(home_stats, "Shots on Target", int)
    away_shots = extract(away_stats, "Shots on Target", int)
    home_xg = extract(home_stats, "Expected Goals", float)
    away_xg = extract(away_stats, "Expected Goals", float)
    home_corners = extract(home_stats, "Corner Kicks", int)
    away_corners = extract(away_stats, "Corner Kicks", int)
    home_poss = extract(home_stats, "Ball Possession", int)
    away_poss = extract(away_stats, "Ball Possession", int)

    def team_conf(shots, xg, corners, poss):
        return (
            (shots * 10) * 0.4 +
            (xg * 20) * 0.3 +
            (corners * 5) * 0.2 +
            (poss / 2) * 0.1
        )

    home_conf = team_conf(home_shots, home_xg, home_corners, home_poss)
    away_conf = team_conf(away_shots, away_xg, away_corners, away_poss)

    if home_conf > away_conf:
        high_team = match["home"]
        confidence = home_conf
    else:
        high_team = match["away"]
        confidence = away_conf

    if confidence < 60:
        return None

    return (
        f"⚽ {match['home']} vs {match['away']}\n"
        f"⏱️ Minute: {match['minute']}'\n"
        f"🔥 High pressure: {high_team}\n"
        f"🎯 Suggestion: {high_team} to score next\n"
        f"📊 Confidence: {confidence:.0f}%"
    )
