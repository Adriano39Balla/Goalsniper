import time
import logging
from typing import Optional, Dict, Any, List
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

from core.config import config

log = logging.getLogger("goalsniper.api")

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

# Global API client instance
api_client = APIClient()
