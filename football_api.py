import logging
from typing import List

from config import FOOTBALL_API_URL
from core.http_client import api_get

# ───────────────────────── Logging ───────────────────────── #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] - %(message)s"
)
log = logging.getLogger(__name__)

# ───────────────────────── API Availability ───────────────────────── #

def is_api_enabled() -> bool:
    return bool(FOOTBALL_API_URL)


def test_api_connection(timeout: int = 5) -> bool:
    """Attempts a test API request to check connectivity."""
    if not is_api_enabled():
        log.warning("[FootballAPI] FOOTBALL_API_URL not set in config.")
        return False
    try:
        response = api_get(FOOTBALL_API_URL, {"live": "all"}, timeout=timeout)
        if response is None:
            log.warning("[FootballAPI] Empty response from API during connectivity test.")
            return False
        return True
    except Exception as e:
        log.warning("[FootballAPI] API connectivity test failed: %s", e)
        return False


def get_today_fixture_ids() -> List[int]:
    """
    Fetches today's fixtures and returns their match IDs.
    Used for pre-match odds snapshotting.
    """
    if not is_api_enabled():
        log.warning("[FootballAPI] Cannot fetch fixtures — API is disabled.")
        return []

    try:
        response = api_get(FOOTBALL_API_URL, {"today": 1}, timeout=5)
        if not response or not isinstance(response, list):
            log.warning("[FootballAPI] Unexpected or empty response when fetching fixtures.")
            return []

        ids = []
        for match in response:
            mid = match.get("id")
            if isinstance(mid, int):
                ids.append(mid)
        return ids

    except Exception as e:
        log.exception("[FootballAPI] Failed to fetch today’s fixtures: %s", e)
        return []
