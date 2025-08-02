import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def calculate_confidence_and_suggestion(match):
    """Analyze match stats and produce bet suggestion with confidence."""

    # Defensive: Ensure keys exist
    if not match or "stats" not in match or "home" not in match or "away" not in match:
        logger.warning("Match data incomplete, skipping prediction")
        return None

    stats_map = {}
    try:
        for team_data in match.get("stats", []):
            team_name = team_data.get("team", {}).get("name")
            if not team_name:
                continue
            stats_map[team_name] = {
                stat.get("type"): stat.get("value", 0)
                for stat in team_data.get("statistics", [])
            }
    except Exception as e:
        logger.error(f"Error parsing match stats: {e}")
        return None

    if match["home"] not in stats_map or match["away"] not in stats_map:
        return None

    home_stats = stats_map[match["home"]]
    away_stats = stats_map[match["away"]]

    def extract(team_stats, key, cast_type=float):
        """Extract numeric stat value safely."""
        val = team_stats.get(key, 0)
        try:
            return cast_type(str(val).replace("%", "").strip() or 0)
        except:
            return 0

    # Extract key stats
    home_shots = extract(home_stats, "Shots on Target", int)
    away_shots = extract(away_stats, "Shots on Target", int)
    home_xg = extract(home_stats, "Expected Goals", float)
    away_xg = extract(away_stats, "Expected Goals", float)
    home_corners = extract(home_stats, "Corner Kicks", int)
    away_corners = extract(away_stats, "Corner Kicks", int)
    home_poss = extract(home_stats, "Ball Possession", int)
    away_poss = extract(away_stats, "Ball Possession", int)

    # Confidence calculation
    def team_conf(shots, xg, corners, poss):
        return (
            (shots * 10) * 0.4 +  # Shots weight
            (xg * 20) * 0.3 +     # xG weight
            (corners * 5) * 0.2 + # Corners weight
            (poss / 2) * 0.1      # Possession weight
        )

    home_conf = team_conf(home_shots, home_xg, home_corners, home_poss)
    away_conf = team_conf(away_shots, away_xg, away_corners, away_poss)

    # Pick team with higher confidence
    if home_conf > away_conf:
        high_team = match["home"]
        confidence = home_conf
    else:
        high_team = match["away"]
        confidence = away_conf

    if confidence < 60:  # Trigger only if ≥ 60%
        logger.info(f"Low confidence ({confidence:.1f}%), skipping suggestion")
        return None

    minute = match.get("minute", "?")

    suggestion = (
        f"⚽ {match['home']} vs {match['away']}\n"
        f"⏱️ Minute: {minute}'\n"
        f"🔥 High pressure: {high_team}\n"
        f"🎯 Suggestion: {high_team} to score next\n"
        f"📊 Confidence: {confidence:.0f}%"
    )

    logger.info(f"Generated suggestion: {suggestion}")
    return suggestion
