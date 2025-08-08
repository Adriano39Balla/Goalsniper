def make_predictions(stats):
    try:
        # Extract safely
        home_stats = stats.get("home", {}).get("response", {})
        away_stats = stats.get("away", {}).get("response", {})

        # Goals average
        home_goals = float(home_stats.get("goals", {}).get("for", {}).get("average", {}).get("total", 0))
        away_goals = float(away_stats.get("goals", {}).get("for", {}).get("average", {}).get("total", 0))
        avg_goals = home_goals + away_goals

        # Over/Under
        over_under = "Over 2.5" if avg_goals > 2.5 else "Under 2.5"

        # Win/Draw/Loss
        home_played = max(1, home_stats.get("fixtures", {}).get("played", {}).get("total", 1))
        away_played = max(1, away_stats.get("fixtures", {}).get("played", {}).get("total", 1))
        home_win_rate = home_stats.get("fixtures", {}).get("wins", {}).get("total", 0) / home_played
        away_win_rate = away_stats.get("fixtures", {}).get("wins", {}).get("total", 0) / away_played

        if home_win_rate > away_win_rate:
            result = "Home Win"
        elif away_win_rate > home_win_rate:
            result = "Away Win"
        else:
            result = "Draw"

        # Double Chance
        double_chance = "1X" if home_win_rate > 0.4 else "X2"

        # Handicap
        home_goals_for = home_stats.get("goals", {}).get("for", {}).get("total", {}).get("total", 0)
        home_goals_against = home_stats.get("goals", {}).get("against", {}).get("total", {}).get("total", 0)
        goal_diff = home_goals_for - home_goals_against
        handicap = "Home -1" if goal_diff > 10 else "No Handicap"

        return {
            "over_under": over_under,
            "result": result,
            "double_chance": double_chance,
            "handicap": handicap
        }
    except Exception as e:
        return {"error": str(e)}
