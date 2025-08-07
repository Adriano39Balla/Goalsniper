import os
import requests
import logging

logger = logging.getLogger("uvicorn")

API_KEY = os.getenv("API_KEY")
STATS_URL = "https://v3.football.api-sports.io/fixtures/statistics"
HEADERS = {"x-apisports-key": API_KEY}

def analyze_matches(match: dict) -> dict | None:
    return {
        "match_id": 999999,
        "team": "Test FC vs Simulated United",
        "league": "Simulation League",
        "tip": "Over 2.5 Goals",
        "confidence": 88
    }

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
