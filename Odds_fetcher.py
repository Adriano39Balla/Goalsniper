from core.http_client import api_get
import os

FOOTBALL_API_URL = os.getenv("FOOTBALL_API_URL", "").strip()

def fetch_odds_for_match(match_id: int) -> dict:
    try:
        if not FOOTBALL_API_URL:
            return {}
        return api_get(f"{FOOTBALL_API_URL}/odds", {"id": match_id}) or {}
    except Exception:
        return {}
