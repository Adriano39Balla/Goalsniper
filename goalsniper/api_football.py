from __future__ import annotations

import asyncio
import json
import os
import time
import random
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .config import API_KEY, MAX_CONCURRENT_REQUESTS
from .logger import warn
from . import filters as cfg_filters  # <-- use the unified filters module

# ---- API base / retries / token bucket ---------------------------------------
BASE_URL = os.getenv("API_BASE_URL", "https://v3.football.api-sports.io")
API_TIMEOUT = float(os.getenv("API_TIMEOUT", "30"))
API_RETRIES = int(os.getenv("API_RETRIES", "3"))
API_BACKOFF_INITIAL = float(os.getenv("API_BACKOFF_INITIAL", "0.5"))
API_BACKOFF_MAX = float(os.getenv("API_BACKOFF_MAX", "2.0"))

# If set to "false", we do NO filtering here; scanner will handle everything.
_APPLY_API_FILTERS = (os.getenv("APPLY_API_FILTERS", "true") or "").strip().lower() in (
    "1", "true", "yes", "on"
)

# token bucket (per process)
_api_rps_env = os.getenv("API_RPS")
if _api_rps_env is None:
    per_min = os.getenv("API_RATE_PER_MIN")
    API_RPS = float(per_min) / 60.0 if per_min else float(os.getenv("API_RPS", "2"))
else:
    API_RPS = float(_api_rps_env)
API_BURST = int(os.getenv("API_BURST", "4"))

# ---- Cache TTLs --------------------------------------------------------------
TTL_CURRENT_LEAGUES  = int(os.getenv("CACHE_TTL_LEAGUES", "1800"))
TTL_FIXTURES_BY_DATE = int(os.getenv("CACHE_TTL_FIXTURES_BYDATE", os.getenv("CACHE_TTL_FIXTURES", "900")))
TTL_LIVE_FIXTURES    = int(os.getenv("CACHE_TTL_LIVE", "90"))
TTL_TEAM_STATS       = int(os.getenv("CACHE_TTL_TEAM_STATS", "600"))
TTL_ODDS             = int(os.getenv("CACHE_TTL_ODDS", "600"))

HEADERS = {"x-apisports-key": API_KEY}
_sem = asyncio.Semaphore(max(1, int(MAX_CONCURRENT_REQUESTS)))

# ---------------- Token bucket ----------------
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

# --------------- small cache ------------------
_CacheVal = Tuple[float, Any]
_cache: Dict[str, _CacheVal] = {}

def _stable_key(path: str, params: Dict[str, Any]) -> str:
    return f"{path}|{json.dumps(params, sort_keys=True, separators=(',', ':'))}"

def _cache_get(key: str, ttl: int) -> Optional[Any]:
    if ttl <= 0:
        return None
    v = _cache.get(key)
    if not v:
        return None
    ts, data = v
    return data if (time.time() - ts) < ttl else None

def _cache_put(key: str, data: Any):
    _cache[key] = (time.time(), data)

# --------------- in‑flight coalescing ---------------
from asyncio import Future
_inflight: Dict[str, Future] = {}
_inflight_lock = asyncio.Lock()

# --------------- core GET with retries + coalescing --------
async def _api_get(client: httpx.AsyncClient, path: str, params: Dict[str, Any], ttl: int) -> Any:
    key = _stable_key(path, params)
    cached = _cache_get(key, ttl)
    if cached is not None:
        return cached

    # single‑flight: coalesce identical concurrent requests
    async with _inflight_lock:
        fut = _inflight.get(key)
        if fut is None:
            fut = asyncio.get_running_loop().create_future()
            _inflight[key] = fut
            leader = True
        else:
            leader = False

    if not leader:
        try:
            return await asyncio.wait_for(fut, timeout=5)
        except asyncio.TimeoutError:
            cached = _cache_get(key, ttl)
            if cached is not None:
                return cached
            return await fut

    url = f"{BASE_URL}{path}"
    backoff = API_BACKOFF_INITIAL
    last_exc: Optional[BaseException] = None
    payload = None

    try:
        for attempt in range(1, API_RETRIES + 1):
            await _bucket.acquire()
            async with _sem:
                try:
                    r = await client.get(url, params=params, headers=HEADERS, timeout=API_TIMEOUT)
                    if r.status_code == 429:
                        ra = r.headers.get("Retry-After")
                        delay = float(ra) if (ra and ra.isdigit()) else backoff
                        await asyncio.sleep(delay + random.uniform(0, 0.25))
                        backoff = min(backoff * 2.0, API_BACKOFF_MAX)
                        if attempt < API_RETRIES:
                            warn("API 429, retrying", delay)
                            continue
                    if r.status_code in (500, 502, 503, 504):
                        if attempt < API_RETRIES:
                            warn("API retry", r.status_code, (r.text or "")[:160])
                            await asyncio.sleep(backoff + random.uniform(0, 0.25))
                            backoff = min(backoff * 2.0, API_BACKOFF_MAX)
                            continue

                    r.raise_for_status()
                    data = r.json()
                    if isinstance(data, dict) and data.get("errors"):
                        raise httpx.HTTPError(str(data["errors"]))
                    payload = data.get("response") if isinstance(data, dict) else data
                    if ttl > 0:
                        _cache_put(key, payload)
                    return payload
                except Exception as e:
                    last_exc = e
                    if attempt >= API_RETRIES:
                        raise
                    warn("API retry on exception:", str(e))
                    await asyncio.sleep(backoff + random.uniform(0, 0.25))
                    backoff = min(backoff * 2.0, API_BACKOFF_MAX)
    finally:
        async with _inflight_lock:
            fut = _inflight.pop(key, None)
            if fut and not fut.done():
                if last_exc is None:
                    fut.set_result(payload if payload is not None else _cache_get(key, ttl))
                else:
                    fut.set_exception(last_exc)

    return None

# --------------- unified filtering (DB/filters module) ---------------
def _lname(fx: dict) -> str:
    return ((fx.get("league") or {}).get("name") or "").upper()

def _country(fx: dict) -> str:
    return ((fx.get("league") or {}).get("country") or "").upper()

async def _apply_unified_filters(fixtures: List[dict]) -> List[dict]:
    if not _APPLY_API_FILTERS:
        # Pass-through: let the scanner handle everything
        return fixtures

    eff = await cfg_filters.get_filters()
    allow_keys = eff.get("allowLeagueKeywords", [])
    ex_keys = eff.get("excludeKeywords", [])
    allow_countries = eff.get("allowCountries", set())

    out: List[dict] = []
    for fx in fixtures or []:
        name = _lname(fx)
        if not name:
            continue
        if ex_keys and any(bad in name for bad in ex_keys):
            continue
        if allow_keys and not any(k in name for k in allow_keys):
            continue
        if allow_countries:
            c = _country(fx)
            if c and c not in allow_countries:
                continue
        out.append(fx)
    return out

# --------------- public API -------------------
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
    return await _apply_unified_filters(data)

async def get_live_fixtures(client: httpx.AsyncClient) -> List[Dict[str, Any]]:
    data = await _api_get(client, "/fixtures", {"live": "all"}, TTL_LIVE_FIXTURES) or []
    return await _apply_unified_filters(data)

async def get_team_statistics(client: httpx.AsyncClient, league_id: int, season: int, team_id: int) -> Dict[str, Any]:
    return await _api_get(
        client, "/teams/statistics",
        {"league": int(league_id), "season": int(season), "team": int(team_id)},
        TTL_TEAM_STATS,
    ) or {}

async def get_odds_for_fixture(client: httpx.AsyncClient, fixture_id: int) -> Any:
    return await _api_get(client, "/odds", {"fixture": int(fixture_id)}, TTL_ODDS) or []
