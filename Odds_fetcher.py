import logging
from core.http_client import api_get
from config import FOOTBALL_API_URL

# ───────────────────────── Logging Setup ───────────────────────── #
logger = logging.getLogger(__name__)

# ───────────────────────── Fetch Function ───────────────────────── #

def fetch_odds_for_match(match_id: int) -> dict:
    try:
        if not FOOTBALL_API_URL:
            logger.warning("FOOTBALL_API_URL not set in config.")
            return {}
        odds = api_get(f"{FOOTBALL_API_URL}/odds", {"id": match_id}) or {}
        return odds
    except Exception as e:
        logger.exception(f"[Odds Fetch] Failed to fetch odds for match_id={match_id}: {e}")
        return {}
