import time, os
import logging
from typing import Optional, Dict, Any, List, Tuple
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

from core.config import config

log = logging.getLogger("goalsniper.api")

# Add these missing API-related constants and utilities
INPLAY_STATUSES = {"1H", "HT", "2H", "ET", "BT", "P"}

# Add global caches for API data (moved from main.py)
STATS_CACHE: Dict[int, Tuple[float, list]] = {}
EVENTS_CACHE: Dict[int, Tuple[float, list]] = {}
ODDS_CACHE: Dict[int, Tuple[float, dict]] = {}
NEG_CACHE: Dict[Tuple[str, int], Tuple[float, bool]] = {}

# Add negative cache TTL
NEG_TTL_SEC = int(os.getenv("NEG_TTL_SEC", "45"))

def _blocked_league(league_obj: dict) -> bool:
    """Check if league should be blocked from processing"""
    name = str((league_obj or {}).get("name", "")).lower()
    country = str((league_obj or {}).get("country", "")).lower()
    typ = str((league_obj or {}).get("type", "")).lower()
    txt = f"{country} {name} {typ}"
    
    # Check block patterns
    block_patterns = ["u17", "u18", "u19", "u20", "u21", "u23", "youth", "junior", "reserve", "res.", "friendlies", "friendly"]
    if any(p in txt for p in block_patterns):
        return True
    
    # Check denied league IDs
    deny_ids = [x.strip() for x in os.getenv("LEAGUE_DENY_IDS", "").split(",") if x.strip()]
    lid = str((league_obj or {}).get("id") or "")
    if lid in deny_ids:
        return True
    
    return False

def _api_get(url: str, params: dict, timeout: int = 15) -> Optional[Dict]:
    """Legacy API function for compatibility (delegates to APIClient)"""
    # Extract endpoint from URL for APIClient
    if url.startswith(config.api.base_url):
        endpoint = url.replace(config.api.base_url + "/", "")
        return api_client.get(endpoint, params)
    else:
        # Fallback to direct request for non-standard URLs
        return api_client._direct_request(url, params, timeout)

class APIClient:
    def __init__(self):
        self.session = requests.Session()
        self.circuit_breaker = {
            "failures": 0,
            "opened_until": 0.0,
            "last_success": 0.0
        }
        
        # Setup retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            respect_retry_after_header=True
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        
        self.headers = {
            "x-apisports-key": config.api.key,
            "Accept": "application/json"
        }

    def _check_circuit_breaker(self) -> bool:
        now = time.time()
        if self.circuit_breaker["opened_until"] > now:
            log.warning("[CB] Circuit open, rejecting request")
            return False
            
        # Reset circuit breaker after cooldown if we have successes
        if (self.circuit_breaker["failures"] > 0 and 
            now - self.circuit_breaker.get("last_success", 0) > config.api.circuit_breaker_cooldown):
            log.info("[CB] Resetting circuit breaker after quiet period")
            self.circuit_breaker["failures"] = 0
            self.circuit_breaker["opened_until"] = 0
            
        return True

    def get(self, endpoint: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self._check_circuit_breaker():
            return None
            
        url = f"{config.api.base_url}/{endpoint}"
        
        try:
            response = self.session.get(
                url,
                headers=self.headers,
                params=params,
                timeout=min(15, config.api.timeout)
            )
            
            # Update circuit breaker state
            now = time.time()
            if response.status_code == 429:
                self.circuit_breaker["failures"] += 1
            elif response.status_code >= 500:
                self.circuit_breaker["failures"] += 1
            else:
                self.circuit_breaker["failures"] = 0
                self.circuit_breaker["last_success"] = now

            if self.circuit_breaker["failures"] >= config.api.circuit_breaker_threshold:
                self.circuit_breaker["opened_until"] = now + config.api.circuit_breaker_cooldown
                log.warning("[CB] API-Football opened for %ss", config.api.circuit_breaker_cooldown)

            return response.json() if response.ok else None
            
        except Exception:
            self.circuit_breaker["failures"] += 1
            if self.circuit_breaker["failures"] >= config.api.circuit_breaker_threshold:
                self.circuit_breaker["opened_until"] = time.time() + config.api.circuit_breaker_cooldown
                log.warning("[CB] API-Football opened due to exceptions")
            return None

    def _direct_request(self, url: str, params: Dict[str, Any], timeout: int = 15) -> Optional[Dict]:
        """Direct request for non-standard URLs (compatibility)"""
        if not self._check_circuit_breaker():
            return None
            
        try:
            response = self.session.get(
                url,
                headers=self.headers,
                params=params,
                timeout=min(timeout, config.api.timeout)
            )
            return response.json() if response.ok else None
        except Exception:
            return None

    def get_fixtures(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        result = self.get("fixtures", params)
        return result.get("response", []) if isinstance(result, dict) else []

    def get_live_matches(self) -> List[Dict[str, Any]]:
        return self.get_fixtures({"live": "all"})

    def get_fixture_events(self, fixture_id: int) -> List[Dict[str, Any]]:
        result = self.get("fixtures/events", {"fixture": fixture_id})
        return result.get("response", []) if isinstance(result, dict) else []

    def get_fixture_statistics(self, fixture_id: int) -> List[Dict[str, Any]]:
        result = self.get("fixtures/statistics", {"fixture": fixture_id})
        return result.get("response", []) if isinstance(result, dict) else []

    def get_odds(self, fixture_id: int, live: bool = True) -> List[Dict[str, Any]]:
        endpoint = "odds/live" if live else "odds"
        result = self.get(endpoint, {"fixture": fixture_id})
        return result.get("response", []) if isinstance(result, dict) else []

    # Add cached data fetching methods (compatibility with main.py)
    def fetch_match_stats(self, fid: int) -> list:
        """Fetch match statistics with caching"""
        now = time.time()
        k = ("stats", fid)
        ts_empty = NEG_CACHE.get(k, (0.0, False))
        if ts_empty[1] and (now - ts_empty[0] < NEG_TTL_SEC):
            return []
        if fid in STATS_CACHE and now - STATS_CACHE[fid][0] < 90:
            return STATS_CACHE[fid][1]
        
        stats = self.get_fixture_statistics(fid)
        STATS_CACHE[fid] = (now, stats)
        if not stats:
            NEG_CACHE[k] = (now, True)
        return stats

    def fetch_match_events(self, fid: int) -> list:
        """Fetch match events with caching"""
        now = time.time()
        k = ("events", fid)
        ts_empty = NEG_CACHE.get(k, (0.0, False))
        if ts_empty[1] and (now - ts_empty[0] < NEG_TTL_SEC):
            return []
        if fid in EVENTS_CACHE and now - EVENTS_CACHE[fid][0] < 90:
            return EVENTS_CACHE[fid][1]
        
        events = self.get_fixture_events(fid)
        EVENTS_CACHE[fid] = (now, events)
        if not events:
            NEG_CACHE[k] = (now, True)
        return events

    def fetch_live_matches_filtered(self) -> List[Dict[str, Any]]:
        """Fetch live matches with filtering and enhanced data"""
        matches = self.get_live_matches()
        filtered_matches = [m for m in matches if not _blocked_league(m.get("league") or {})]
        
        out = []
        for m in filtered_matches:
            fixture = m.get("fixture", {})
            status = fixture.get("status", {})
            elapsed = status.get("elapsed")
            short = (status.get("short") or "").upper()
            
            if elapsed is None or elapsed > 120 or short not in INPLAY_STATUSES:
                continue
                
            fid = fixture.get("id")
            if fid:
                # Enhance match with stats and events
                m["statistics"] = self.fetch_match_stats(fid)
                m["events"] = self.fetch_match_events(fid)
                out.append(m)
                
        return out

    def fetch_odds_cached(self, fid: int) -> Optional[Dict]:
        """Fetch odds with caching"""
        now = time.time()
        k = ("odds", fid)
        ts_empty = NEG_CACHE.get(k, (0.0, False))
        if ts_empty[1] and (now - ts_empty[0] < NEG_TTL_SEC):
            return {}
        if fid in ODDS_CACHE and now - ODDS_CACHE[fid][0] < 120:
            return ODDS_CACHE[fid][1]
        
        odds_data = self.get_odds(fid, live=True)
        # Process odds data into the expected format
        processed_odds = self._process_odds_data(odds_data)
        ODDS_CACHE[fid] = (now, processed_odds)
        if not processed_odds:
            NEG_CACHE[k] = (now, True)
        return processed_odds

    def _process_odds_data(self, odds_data: List[Dict]) -> Dict[str, Any]:
        """Process raw odds data into structured format"""
        # Implementation would go here
        return {}

# Global API client instance
api_client = APIClient()

# Legacy functions for compatibility
def fetch_live_matches() -> List[Dict]:
    return api_client.fetch_live_matches_filtered()

def fetch_match_stats(fid: int) -> list:
    return api_client.fetch_match_stats(fid)

def fetch_match_events(fid: int) -> list:
    return api_client.fetch_match_events(fid)

def fetch_odds(fid: int, *args, **kwargs) -> Dict:
    return api_client.fetch_odds_cached(fid)

# Export for compatibility
__all__ = [
    'api_client',
    'fetch_live_matches', 
    'fetch_match_stats',
    'fetch_match_events', 
    'fetch_odds',
    '_api_get',
    '_blocked_league',
    'INPLAY_STATUSES',
    'STATS_CACHE',
    'EVENTS_CACHE', 
    'ODDS_CACHE',
    'NEG_CACHE'
]
