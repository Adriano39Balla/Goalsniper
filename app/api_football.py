import requests
import time
from typing import Dict, List, Any
from app.config import API_FOOTBALL_KEY, API_FOOTBALL_URL

# API Headers
HEADERS = {
    "x-apisports-key": API_FOOTBALL_KEY,
    "x-rapidapi-key": API_FOOTBALL_KEY,
}

REQ_TIMEOUT = 8
RETRY_COUNT = 2


# =====================================================================
# GENERIC GET REQUEST
# =====================================================================

def api_get(endpoint: str, params: Dict[str, Any] = None) -> Any:
    url = f"{API_FOOTBALL_URL}/{endpoint}"

    for attempt in range(1, RETRY_COUNT + 2):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=REQ_TIMEOUT)

            if r.status_code == 200:
                data = r.json()
                return data.get("response", [])

            print(f"[API-FOOTBALL] HTTP {r.status_code}: {r.text}")

        except Exception as e:
            print(f"[API-FOOTBALL] Exception: {e}")

        time.sleep(0.5)

    return []


# =====================================================================
# LIVE FIXTURES
# =====================================================================

def get_live_fixtures() -> List[Dict[str, Any]]:
    return api_get("fixtures", {"live": "all"})


# =====================================================================
# PREMATCH FIXTURES
# =====================================================================

def get_prematch_fixtures(hours_ahead: int = 2) -> List[Dict[str, Any]]:
    return api_get("fixtures", {"next": hours_ahead})


# =====================================================================
# LIVE STATS
# =====================================================================

def get_fixture_stats(fixture_id: int) -> Dict[str, Any]:
    return api_get("fixtures/statistics", {"fixture": fixture_id})


# =====================================================================
# LIVE ODDS
# =====================================================================

def get_live_odds(fixture_id: int) -> List[Dict[str, Any]]:
    # IMPORTANT: use odds/live (confirmed by API-Football)
    return api_get("odds/live", {"fixture": fixture_id})


# =====================================================================
# FIXTURE NORMALIZER
# =====================================================================

def normalize_fixture(raw: Dict[str, Any]) -> Dict[str, Any]:
    fixture = raw.get("fixture", {})
    teams = raw.get("teams", {})
    goals = raw.get("goals", {})

    return {
        "fixture_id": fixture.get("id"),
        "league_id": fixture.get("league", {}).get("id"),
        "home_team_id": teams.get("home", {}).get("id"),
        "away_team_id": teams.get("away", {}).get("id"),
        "home_goals": goals.get("home"),
        "away_goals": goals.get("away"),
        "status": fixture.get("status", {}).get("short"),
        "minute": fixture.get("status", {}).get("elapsed"),
        "timestamp": fixture.get("timestamp"),
    }


# =====================================================================
# UNIVERSAL ODDS PARSER (FULLTIME 1X2, OU2.5, BTTS)
# =====================================================================

CORE_MARKETS = {
    "1x2": [
        "1x2", "fulltime result", "result", "match result",
        "win/draw/win", "ft result", "3-way result"
    ],
    "btts": [
        "both teams to score", "goal/no goal", "btts", "gg/ng"
    ],
    "ou": [
        "over/under", "o/u", "total goals", "match goals",
        "goals o/u", "over/under line"
    ],
}


def parse_live_odds(raw) -> dict:
    """
    Extracts:
        - Fulltime 1X2
        - FT Over/Under 2.5
        - FT BTTS
    From chaotic API-Football data.
    """

    clean = {
        "1x2": {"home": None, "draw": None, "away": None},
        "ou25": {"over": None, "under": None},
        "btts": {"yes": None, "no": None},
    }

    if not raw:
        return clean

    for item in raw:

        name = item.get("name", "").lower().strip()
        values = item.get("values", [])

        # -------------------------------------------------
        # DETECT 1X2 (FULL TIME ONLY)
        # -------------------------------------------------
        is_1x2 = any(m in name for m in CORE_MARKETS["1x2"])

        # auto-ignore time-shifted markets like "1x2 - 60 minutes"
        if "minutes" in name:
            is_1x2 = False

        if is_1x2:
            for v in values:
                label = v.get("value", "").lower()
                odd = v.get("odd")

                if not odd:
                    continue

                if label in ["home", "1"]:
                    clean["1x2"]["home"] = float(odd)
                elif label in ["draw", "x"]:
                    clean["1x2"]["draw"] = float(odd)
                elif label in ["away", "2"]:
                    clean["1x2"]["away"] = float(odd)

        # -------------------------------------------------
        # DETECT OVER/UNDER 2.5
        # -------------------------------------------------
        is_ou = any(m in name for m in CORE_MARKETS["ou"])

        if is_ou:
            for v in values:
                label = v.get("value", "").lower()
                odd = v.get("odd")
                handicap = str(v.get("handicap", "")).strip()

                if odd is None:
                    continue

                # detect line = 2.5
                if "2.5" in label or handicap == "2.5":

                    if "over" in label:
                        clean["ou25"]["over"] = float(odd)

                    elif "under" in label:
                        clean["ou25"]["under"] = float(odd)

        # -------------------------------------------------
        # DETECT BTTS (FULL TIME)
        # -------------------------------------------------
        is_btts = any(m in name for m in CORE_MARKETS["btts"])

        # ignore 1H/2H BTTS
        if "1st half" in name or "2nd half" in name:
            is_btts = False

        if is_btts:
            for v in values:
                label = v.get("value", "").lower()
                odd = v.get("odd")

                if not odd:
                    continue

                if label in ["yes", "goal", "gg"]:
                    clean["btts"]["yes"] = float(odd)
                elif label in ["no", "no goal", "ng"]:
                    clean["btts"]["no"] = float(odd)

    return clean
