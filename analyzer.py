import os
import requests
import logging

logger = logging.getLogger("uvicorn")

API_KEY = os.getenv("API_KEY")
STATS_URL = "https://v3.football.api-sports.io/fixtures/statistics"
HEADERS = {"x-apisports-key": API_KEY}

def analyze_matches(match: dict) -> dict | None:
    elapsed = match["fixture"]["status"].get("elapsed")
    if elapsed is None or not (20 <= elapsed <= 45 or 65 <= elapsed <= 90):
        return None  # Only trigger in late 1st half or late 2nd half

    stats = get_match_stats(match)
    if not stats:
        return None

    goals_home = match["goals"]["home"] or 0
    goals_away = match["goals"]["away"] or 0
    total_goals = goals_home + goals_away

    xg_sum = stats["xg_home"] + stats["xg_away"]
    shots_total = stats["shots_total"]
    corners = stats["corners"]
    red_cards = stats["reds"]

    if xg_sum > 2.0 and shots_total > 15 and total_goals < 3:
        confidence = min(95, int(xg_sum * 20))
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
            "confidence": 80
        }

    return None

def get_match_stats(match: dict) -> dict:
    fixture_id = match["fixture"]["id"]

    try:
        res = requests.get(
            STATS_URL,
            params={"fixture": fixture_id},
            headers=HEADERS
        )
        res.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"[Analyzer] Failed to fetch stats: {e}")
        return {}

    data = res.json().get("response", [])
    total_shots = sum(
        item["value"] or 0
        for team in data
        for item in team.get("statistics", [])
        if item["type"] == "Shots on Goal"
    )

    total_corners = sum(
        item["value"] or 0
        for team in data
        for item in team.get("statistics", [])
        if item["type"] == "Corner Kicks"
    )

    total_reds = sum(
        item["value"] or 0
        for team in data
        for item in team.get("statistics", [])
        if item["type"] == "Red Cards"
    )

    # These are fallback xG values – API usually doesn't provide real xG per team in this endpoint
    xg_home = 1.1
    xg_away = 1.2

    return {
        "shots_total": total_shots,
        "corners": total_corners,
        "reds": total_reds,
        "xg_home": xg_home,
        "xg_away": xg_away
    }
