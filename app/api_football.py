import requests
import time
import json
from typing import Dict, List, Any
from app.config import API_FOOTBALL_KEY, API_FOOTBALL_URL

HEADERS = {
    "x-apisports-key": API_FOOTBALL_KEY,
    "x-rapidapi-key": API_FOOTBALL_KEY,
}

REQ_TIMEOUT = 8  # seconds
RETRY_COUNT = 2


def api_get(endpoint: str, params: Dict[str, Any] = None) -> Any:
    """Simple GET request wrapper with retry logic."""
    url = f"{API_FOOTBALL_URL}/{endpoint}"

    for attempt in range(1, RETRY_COUNT + 2):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=REQ_TIMEOUT)
            if r.status_code == 200:
                data = r.json()
                return data.get("response", [])
            else:
                print(f"[API-FOOTBALL] HTTP {r.status_code} -> {r.text}")

        except Exception as e:
            print(f"[API-FOOTBALL] Exception: {e}")

        time.sleep(0.5)

    return []


# -------------------------------------------------------------
# LIVE FIXTURES (primary for in-play predictions)
# -------------------------------------------------------------
def get_live_fixtures() -> List[Dict[str, Any]]:
    """Returns all live matches with basic fixture info."""
    return api_get("fixtures", {"live": "all"})


# -------------------------------------------------------------
# PREMATCH FIXTURES
# -------------------------------------------------------------
def get_prematch_fixtures(hours_ahead: int = 2) -> List[Dict[str, Any]]:
    """Returns fixtures starting within the next X hours."""
    return api_get("fixtures", {"next": hours_ahead})


# -------------------------------------------------------------
# IN-PLAY STATS
# -------------------------------------------------------------
def get_fixture_stats(fixture_id: int) -> Dict[str, Any]:
    """Returns detailed fixture stats for a given match."""
    return api_get("fixtures/statistics", {"fixture": fixture_id})


# -------------------------------------------------------------
# LIVE ODDS
# -------------------------------------------------------------
def get_live_odds(fixture_id: int) -> List[Dict[str, Any]]:
    """
    Returns odds for the fixture across all markets and bookmakers.

    Strategy:
      1) Try /odds/live?fixture=ID (as API-Football recommends for live) 
      2) Fallback to /odds?live=all and filter by fixture
      3) Fallback to /odds?fixture=ID (prematch / partial coverage)
    """

    # 1) Primary: /odds/live
    odds_live = api_get("odds/live", {"fixture": fixture_id})
    if odds_live:
        return odds_live

    # 2) Fallback: /odds?live=all then filter
    all_live = api_get("odds", {"live": "all"})
    if all_live:
        filtered = [o for o in all_live if o.get("fixture", {}).get("id") == fixture_id]
        if filtered:
            return filtered

    # 3) Fallback: fixture-scoped odds
    odds = api_get("odds", {"fixture": fixture_id})
    return odds or []


# -------------------------------------------------------------
# NORMALIZATION HELPERS
# -------------------------------------------------------------
def normalize_fixture(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Simplify the raw API data for use in our ML features."""
    fixture = raw.get("fixture", {})
    league = raw.get("league", {})
    teams = raw.get("teams", {})
    goals = raw.get("goals", {})

    return {
        "fixture_id": fixture.get("id"),
        "league_id": league.get("id"),
        "home_team_id": teams.get("home", {}).get("id"),
        "away_team_id": teams.get("away", {}).get("id"),
        "home_goals": goals.get("home"),
        "away_goals": goals.get("away"),
        "status": fixture.get("status", {}).get("short"),
        "minute": fixture.get("status", {}).get("elapsed"),
        "timestamp": fixture.get("timestamp"),
    }
