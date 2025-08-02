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
if API_KEY:
    HEADERS = {"x-apisports-key": API_KEY}
else:
    logger.error("API_KEY not set in environment variables")
    HEADERS = {}

# Retry session
session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))

def api_request(endpoint, params=None):
    """General API request with retries and error handling."""
    if not API_KEY:
        logger.error("API_KEY missing — cannot make API requests")
        return None
    try:
        res = session.get(
            f"{BASE_URL}/{endpoint}",
            headers=HEADERS,
            params=params,
            timeout=15,
            verify=certifi.where()
        )
        res.raise_for_status()
        data = res.json()
        return data
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}", exc_info=True)
        return None

def get_live_matches():
    """Fetch all currently live matches with statistics."""
    matches = api_request("fixtures", {"live": "all"})
    if not matches or "response" not in matches:
        logger.warning("No live matches returned from API")
        return []

    live_data = []
    for match in matches["response"]:
        try:
            fixture_id = match["fixture"]["id"]
            elapsed = match["fixture"]["status"].get("elapsed") or 0
            home_team = match["teams"]["home"]["name"]
            away_team = match["teams"]["away"]["name"]

            # Get stats for this fixture
            stats = api_request("fixtures/statistics", {"fixture": fixture_id})
            stats_data = stats.get("response", []) if stats else []

            live_data.append({
                "fixture_id": fixture_id,
                "minute": elapsed,
                "home": home_team,
                "away": away_team,
                "score_home": match["goals"]["home"],
                "score_away": match["goals"]["away"],
                "stats": stats_data
            })

        except Exception as e:
            logger.error(f"Error processing match data: {e}", exc_info=True)
            continue

    logger.info(f"Fetched {len(live_data)} live matches")
    return live_data
