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

# Retry session
session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))

def api_request(endpoint, params=None):
    """General API request with retries."""
    if not API_KEY:
        logger.error("API_KEY not set")
        return None
    try:
        res = session.get(f"{BASE_URL}/{endpoint}", headers=HEADERS, params=params, timeout=15, verify=certifi.where())
        res.raise_for_status()
        return res.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}", exc_info=True)
        return None

def get_live_matches():
    """Fetch all currently live matches with statistics."""
    matches = api_request("fixtures", {"live": "all"})
    if not matches or "response" not in matches:
        return []

    live_data = []
    for match in matches["response"]:
        fixture_id = match["fixture"]["id"]

        stats = api_request("fixtures/statistics", {"fixture": fixture_id})
        stats_data = stats.get("response", []) if stats else []

        live_data.append({
            "fixture_id": fixture_id,
            "minute": match["fixture"]["status"]["elapsed"],
            "home": match["teams"]["home"]["name"],
            "away": match["teams"]["away"]["name"],
            "score_home": match["goals"]["home"],
            "score_away": match["goals"]["away"],
            "stats": stats_data
        })

    return live_data
