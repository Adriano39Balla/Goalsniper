import os
import requests
from dotenv import load_dotenv
from datetime import datetime, timedelta
import logging

load_dotenv()
logger = logging.getLogger("uvicorn")

API_KEY = os.getenv("API_KEY")
FIXTURES_URL = "https://v3.football.api-sports.io/fixtures"
HEADERS = {"x-apisports-key": API_KEY}

def get_upcoming_predictions(hours_ahead=6):
    now = datetime.utcnow()
    future = now + timedelta(hours=hours_ahead)

    params = {
        "date": now.strftime("%Y-%m-%d")
    }

    try:
        res = requests.get(FIXTURES_URL, headers=HEADERS, params=params)
        res.raise_for_status()
        fixtures = res.json().get("response", [])
    except Exception as e:
        logger.error(f"[Predictor] Failed to fetch fixtures: {e}")
        return []

    tips = []

    for match in fixtures:
        kickoff_ts = match["fixture"]["timestamp"]
        if not kickoff_ts:
            continue

        kickoff = datetime.utcfromtimestamp(kickoff_ts)
        if not (now <= kickoff <= future):
            continue

        stats = estimate_team_strength(match)
        if stats and stats["btts_likely"]:
            tips.append({
                "match_id": match["fixture"]["id"],
                "team": f"{match['teams']['home']['name']} vs {match['teams']['away']['name']}",
                "league": match["league"]["name"],
                "tip": "BTTS (Yes)",
                "confidence": 78  # still static; you can improve later
            })

    return tips

def estimate_team_strength(match):
    home = match["teams"]["home"]
    away = match["teams"]["away"]

    # crude logic: team has recently won
    home_strength = int(home.get("winner") is not None)
    away_strength = int(away.get("winner") is not None)

    return {
        "btts_likely": home_strength and away_strength
    }
