# football_api.py

import os
import logging
from typing import List, Optional

from core.http_client import api_get  # You should move _api_get into http_client.py or similar if needed

log = logging.getLogger(__name__)

# Load from environment
FOOTBALL_API_URL = os.getenv("FOOTBALL_API_URL", "").strip()


def is_api_enabled() -> bool:
    return bool(FOOTBALL_API_URL)


def test_api_connection(timeout: int = 5) -> bool:
    """Attempts a test API request to check connectivity."""
    if not is_api_enabled():
        log.warning("Football API URL not set.")
        return False
    try:
        response = api_get(FOOTBALL_API_URL, {"live": "all"}, timeout=timeout)
        return response is not None
    except Exception as e:
        log.warning("API connectivity test failed: %s", e)
        return False


def get_today_fixture_ids() -> List[int]:
    """
    Fetches today's fixtures and returns their match IDs.
    Used for pre-match odds snapshotting.
    """
    try:
        response = api_get(FOOTBALL_API_URL, {"today": 1}, timeout=5)
        if not response or not isinstance(response, list):
            return []
        ids = []
        for match in response:
            mid = match.get("id")
            if isinstance(mid, int):
                ids.append(mid)
        return ids
    except Exception as e:
        log.exception("Failed to fetch today’s fixtures: %s", e)
        return []
