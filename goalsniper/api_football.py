# goalsniper/api_football.py
from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .config import API_KEY, MAX_CONCURRENT_REQUESTS
from .logger import log, warn
from . import filters  # <-- NEW

BASE_URL = os.getenv("API_BASE_URL", "https://v3.football.api-sports.io")
API_TIMEOUT = float(os.getenv("API_TIMEOUT", "30"))
API_RETRIES = int(os.getenv("API_RETRIES", "3"))
API_BACKOFF_INITIAL = float(os.getenv("API_BACKOFF_INITIAL", "0.5"))
API_BACKOFF_MAX = float(os.getenv("API_BACKOFF_MAX", "2.0"))

API_RPS = float(os.getenv("API_RPS", "3"))
API_BURST = int(os.getenv("API_BURST", "6"))

TTL_CURRENT_LEAGUES = int(os.getenv("CACHE_TTL_LEAGUES", "1800"))
TTL_FIXTURES_BY_DATE = int(os.getenv("CACHE_TTL_FIXTURES", "900"))
TTL_LIVE_FIXTURES = int(os.getenv("CACHE_TTL_LIVE", "90"))
TTL_TEAM_STATS = int(os.getenv("CACHE_TTL_TEAM_STATS", "600"))
TTL_ODDS = int(os.getenv("CACHE_TTL_ODDS", "600"))

HEADERS = {"x-apisports-key": API_KEY}
_sem = asyncio.Semaphore(max(1, int(MAX_CONCURRENT_REQUESTS)))

class _TokenBucket:
    def __init__(self, rate: float, capacity: int):
        self.rate = max(0.1, float(rate))
        self.capacity = max(1, int(capacity))
        self.tokens = float(capacity)
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            self._last = now
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            if self.tokens < 1.0:
                need = 1.0 - self.tokens
                await asyncio.sleep(need / self.rate)
                self.tokens = 0.0
            else:
                self.tokens -= 1.0

_bucket = _TokenBucket(API_RPS, API_BURST)

_CacheVal = Tuple[float, Any]
_cache: Dict[str, _CacheVal] = {}

def _ckey(path: str, params: Dict[str, Any]) -> str:
    return f"{path}|{tuple(sorted(params.items()))}"

def _cache_get(key: str, ttl: int) -> Optional[Any]:
    if ttl <= 0:
        return None
    val = _cache.get(key)
    if not val:
        return None
    ts, data = val
    if (time.time() - ts) < ttl:
        return data
    return None

def _cache_put(key: str, data: Any):
    _cache[key] = (time.time(), data)

async def _api_get(client: httpx.AsyncClient, path: str, params: Dict[str, Any], ttl: int) -> Any:
    key = _ckey(path, params)
    cached = _cache_get(key, ttl)
    if cached is not None:
        return cached

    url = f"{BASE_URL}{path}"
    backoff = API_BACKOFF_INITIAL

    for attempt in range(1, API_RETRIES + 1):
        await _bucket.acquire()
        async with _sem:
            try:
                resp = await client.get(url, params=params, headers=HEADERS, timeout=API_TIMEOUT)
                if resp.status_code == 429 or 500 <= resp.status_code < 600:
                    if attempt < API_RETRIES:
                        warn("API retry", resp.status_code, (resp.text or "")[:160])
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * 2.0, API_BACKOFF_MAX)
                        continue
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, dict) and data.get("errors"):
                    raise httpx.HTTPError(str(data["errors"]))
                payload = data.get("response") if isinstance(data, dict) else data
                if ttl > 0:
                    _cache_put(key, payload)
                return payload
            except Exception as e:
                if attempt >= API_RETRIES:
                    raise
                warn("API retry on exception:", str(e))
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, API_BACKOFF_MAX)
    return None

# --------- dynamic filtering helpers ----------
def _league_name(fx: dict) -> str:
    return ((fx.get("league") or {}).get("name") or "").upper()

def _country_name(fx: dict) -> str:
    return ((fx.get("league") or {}).get("country") or "").upper()

async def _is_allowed_fixture(fx: dict) -> bool:
    # Pull live filters (cached in filters.py)
    f = await filters.get_filters()
    lname = _league_name(fx)
    if not lname:
        return False

    excl = f["excludeKeywords"]
    if excl and any(bad in lname for bad in excl):
        return False

    allow_keys = f["allowLeagueKeywords"]
    if allow_keys and not any(k in lname for k in allow_keys):
        return False

    allow_countries = f["allowCountries"]
    if allow_countries:
        c = _country_name(fx)
        if c and (c not in allow_countries):
            return False

    return True

# ----------------- public functions -----------------
async def get_current_leagues(client: httpx.AsyncClient) -> List[Dict[str, Any]]:
    res = await _api_get(client, "/leagues", {"current": "true"}, TTL_CURRENT_LEAGUES)
    leagues: List[Dict[str, Any]] = []
    for item in res or []:
        seasons = item.get("seasons") or []
        cur = next((s for s in seasons if s.get("current")), None)
        lg = item.get("league") or {}
        if cur and lg.get("id"):
            leagues.append({
                "leagueId": lg["id"],
                "leagueName": lg.get("name"),
                "type": lg.get("type"),
                "country": (item.get("country") or {}).get("name"),
                "season": cur.get("year"),
            })
    return leagues

async def get_fixtures_by_date(client: httpx.AsyncClient, league_id: int, season: int, date_iso: str) -> List[Dict[str, Any]]:
    data = await _api_get(
        client, "/fixtures",
        {"league": int(league_id), "season": int(season), "date": date_iso},
        TTL_FIXTURES_BY_DATE,
    ) or []
    out: List[Dict[str, Any]] = []
    for fx in data:
        if await _is_allowed_fixture(fx):
            out.append(fx)
    return out

async def get_live_fixtures(client: httpx.AsyncClient) -> List[Dict[str, Any]]:
    data = await _api_get(client, "/fixtures", {"live": "all"}, TTL_LIVE_FIXTURES) or []
    out: List[Dict[str, Any]] = []
    for fx in data:
        if await _is_allowed_fixture(fx):
            out.append(fx)
    return out

async def get_team_statistics(client: httpx.AsyncClient, league_id: int, season: int, team_id: int) -> Dict[str, Any]:
    return await _api_get(
        client, "/teams/statistics",
        {"league": int(league_id), "season": int(season), "team": int(team_id)},
        TTL_TEAM_STATS,
    ) or {}

async def get_odds_for_fixture(client: httpx.AsyncClient, fixture_id: int):
    return await _api_get(client, "/odds", {"fixture": int(fixture_id)}, TTL_ODDS) or []
