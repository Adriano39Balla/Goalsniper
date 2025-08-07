import os
import requests
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()

API_KEY = os.getenv("API_KEY")
HEADERS = {"x-apisports-key": API_KEY}
FIXTURES_URL = "https://v3.football.api-sports.io/fixtures"

def get_upcoming_predictions(hours_ahead=6):
    now = datetime.utcnow()
    future = now + timedelta(hours=hours_ahead)

    params = {
        "date": now.strftime("%Y-%m-%d")
    }

    try:
        res = requests.get(FIXTURES_URL, headers=HEADERS, params=params)
        fixtures = res.json().get("response", [])
    except Exception as e:
        print(f"[PreMatch] Failed to fetch fixtures: {e}")
        return []

    tips = []

    for match in fixtures:
        kickoff = match["fixture"]["timestamp"]
        if kickoff is None:
            continue
        kickoff_dt = datetime.utcfromtimestamp(kickoff)
        if not (now <= kickoff_dt <= future):
            continue

        stats = estimate_team_strength(match)
        if stats and stats["btts_likely"]:
            tips.append({
                "match_id": match["fixture"]["id"],
                "team": f"{match['teams']['home']['name']} vs {match['teams']['away']['name']}",
                "league": match["league"]["name"],
                "tip": "BTTS (Yes)",
                "confidence": 78
            })

    return tips


def estimate_team_strength(match):
    # Placeholder logic – upgrade with team form, xG history, standings
    home = match["teams"]["home"]
    away = match["teams"]["away"]

    home_strength = int(home.get("winner") is not None)
    away_strength = int(away.get("winner") is not None)

    return {
        "btts_likely": home_strength and away_strength
    }
