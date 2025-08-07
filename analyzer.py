import requests
from dotenv import dotenv_values

def analyze_matches(match: dict) -> dict | None:
    stats = get_match_stats(match)
    if not stats:
        return None

    xg_home = stats.get("xg_home", 0)
    xg_away = stats.get("xg_away", 0)
    shots_total = stats.get("shots_total", 0)
    corners = stats.get("corners", 0)
    red_cards = stats.get("reds", 0)
    score_home = match["goals"]["home"] or 0
    score_away = match["goals"]["away"] or 0
    total_goals = score_home + score_away

    if xg_home + xg_away > 0.9 and shots_total > 15 and total_goals < 3:
        confidence = max(70, min(95, int((xg_home + xg_away) * 20)))
        return {
            "match_id": match["fixture"]["id"],
            "team": f"{match['teams']['home']['name']} vs {match['teams']['away']['name']}",
            "league": match["league"]["name"],
            "tip": "Over 2.5 Goals",
            "confidence": confidence
        }

    if corners > 10 and red_cards == 0:
        return {
            "match_id": match["fixture"]["id"],
            "team": f"{match['teams']['home']['name']} vs {match['teams']['away']['name']}",
            "league": match["league"]["name"],
            "tip": "Over 10.5 Corners",
            "confidence": 70
        }

    return None


def get_match_stats(match: dict) -> dict:
    stats_url = f"https://v3.football.api-sports.io/fixtures/statistics"
    fixture_id = match["fixture"]["id"]
    API_KEY = dotenv_values().get("API_KEY")

    res = requests.get(stats_url, params={"fixture": fixture_id}, headers={"x-apisports-key": API_KEY})
    if res.status_code != 200:
        return {}

    data = res.json().get("response", [])
    total_shots = 0
    total_corners = 0
    total_reds = 0
    xg_home = 0
    xg_away = 0

    for team_stats in data:
        stats_list = team_stats.get("statistics", [])
        for item in stats_list:
            if "Shots on Goal" in item["type"]:
                total_shots += item["value"] or 0
            if "Corner Kicks" in item["type"]:
                total_corners += item["value"] or 0
            if "Red Cards" in item["type"]:
                total_reds += item["value"] or 0

    xg_home = match.get("teams", {}).get("home", {}).get("xg", 0) or 0
    xg_away = match.get("teams", {}).get("away", {}).get("xg", 0) or 0

    return {
        "shots_total": total_shots,
        "corners": total_corners,
        "reds": total_reds,
        "xg_home": xg_home,
        "xg_away": xg_away
    }
