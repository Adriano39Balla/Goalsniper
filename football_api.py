# /app/football_api.py
import os
import requests
from datetime import datetime

API_KEY = os.getenv("API_KEY")
BASE_URL = "https://v3.football.api-sports.io"

HEADERS = {"x-apisports-key": API_KEY}

def api_request(url):
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        return {"error": str(e)}

def get_today_fixtures():
    today = datetime.utcnow().strftime("%Y-%m-%d")
    url = f"{BASE_URL}/fixtures?date={today}"
    data = api_request(url)
    fixtures = []
    for match in data.get("response", []):
        fixtures.append({
            "fixture_id": match["fixture"]["id"],
            "home": {
                "id": match["teams"]["home"]["id"],
                "name": match["teams"]["home"]["name"]
            },
            "away": {
                "id": match["teams"]["away"]["id"],
                "name": match["teams"]["away"]["name"]
            },
            "league_id": match["league"]["id"],
            "season": match["league"]["season"]
        })
    return fixtures

def get_team_stats(home_id, away_id, league_id, season):
    url_home = f"{BASE_URL}/teams/statistics?season={season}&team={home_id}&league={league_id}"
    url_away = f"{BASE_URL}/teams/statistics?season={season}&team={away_id}&league={league_id}"
    return {
        "home": api_request(url_home),
        "away": api_request(url_away)
    }
