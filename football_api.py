import os
import logging
import requests
import certifi
from requests.adapters import HTTPAdapter, Retry

# Environment variables
API_KEY = os.getenv("API_KEY")
BASE_URL = "https://v3.football.api-sports.io"

# Logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# API headers
HEADERS = {"x-apisports-key": API_KEY} if API_KEY else {}

# Configure retry session
session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))

def api_request(url):
    """Makes an API request with retries and structured error handling."""
    if not API_KEY:
        logger.error("API_KEY environment variable not set.")
        return {"success": False, "error": "API key missing", "data": None}

    try:
        logger.info(f"Requesting: {url}")
        res = session.get(url, headers=HEADERS, timeout=15, verify=certifi.where())
        res.raise_for_status()
        return {"success": True, "data": res.json()}
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}", exc_info=True)
        return {"success": False, "error": str(e), "data": None}

def get_live_fixtures():
    """Fetch all currently live football fixtures."""
    url = f"{BASE_URL}/fixtures?live=all"
    response = api_request(url)

    if not response.get("success"):
        return []

    fixtures = []
    for match in response["data"].get("response", []):
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
            "season": match["league"]["season"],
            "elapsed": match["fixture"]["status"].get("elapsed"),
            "score": {
                "home": match["goals"]["home"],
                "away": match["goals"]["away"]
            }
        })
    return fixtures

def get_team_stats(home_id, away_id, league_id, season):
    """Fetch statistics for both home and away teams."""
    url_home = f"{BASE_URL}/teams/statistics?season={season}&team={home_id}&league={league_id}"
    url_away = f"{BASE_URL}/teams/statistics?season={season}&team={away_id}&league={league_id}"

    return {
        "home": api_request(url_home),
        "away": api_request(url_away)
    }
