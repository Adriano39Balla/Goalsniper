# goalsniper/api_football.py

import asyncio
import time
from os import getenv
from typing import Any, Dict, List, Tuple
import httpx

from .config import API_KEY, MAX_CONCURRENT_REQUESTS
from .logger import log

# -----------------------------
# Config (env-driven, no hardcodes)
# -----------------------------
# Optional filters (empty = allow all)
COUNTRY_FLAGS_ALLOW = getenv("COUNTRY_FLAGS_ALLOW", "")  # e.g. "🇩🇪,🇬🇧" or "DE,GB"
LEAGUE_ALLOW_KEYWORDS = [
    s.strip() for s in getenv("LEAGUE_ALLOW_KEYWORDS", "").split(",") if s.strip()
]
EXCLUDE_KEYWORDS = [
    s.strip()
    for s in getenv(
        "EXCLUDE_KEYWORDS",
        # default: keep obvious non-pro comps out
        "U19,U20,U21,U23,YOUTH,WOMEN,FRIENDLY,CLUB FRIENDLIES,RESERVE,AMATEUR,B-TEAM",
    ).split(",")
    if s.strip()
]

# Cache TTLs (seconds)
CACHE_TTL_LEAGUES         = int(getenv("CACHE_TTL_LEAGUES", "1800"))  # 30 min
CACHE_TTL_FIXTURES_BYDATE = int(getenv("CACHE_TTL_FIXTURES_BYDATE", "1800"))
CACHE_TTL_LIVE            = int(getenv("CACHE_TTL_LIVE", "90"))
CACHE_TTL_TEAM_STATS      = int(getenv("CACHE_TTL_TEAM_STATS", "600"))
CACHE_TTL_ODDS            = int(getenv("CACHE_TTL_ODDS", "600"))

# Rate limiting (token bucket per minute)
API_RATE_PER_MIN = int(getenv("API_RATE_PER_MIN", "60"))  # keep ≤ provider limit

# Concurrency gate (from config)
_sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# -----------------------------
# Filtering helpers
# -----------------------------
def _flag_to_iso2(flag: str) -> str:
    cps = [ord(c) - 0x1F1E6 for c in flag if 0x1F1E6 <= ord(c) <= 0x1F1FF]
    if len(cps) == 2:
        return chr(cps[0] + 65) + chr(cps[1] + 65)
    return flag.upper()

def _normalize_country_flags(csv_flags: str) -> set[str]:
    out: set[str] = set()
    for chunk in (csv_flags or "").split(","):
        s = chunk.strip()
        if not s:
            continue
        if any(0x1F1E6 <= ord(c) <= 0x1F1FF for c in s):
            out.add(_flag_to_iso2(s))
        else:
            out.add(s.upper())
    return out

_ALLOW_COUNTRIES = _normalize_country_flags(COUNTRY_FLAGS_ALLOW)
_ALLOW_KEYS = [k.upper() for k in LEAGUE_ALLOW_KEYWORDS]
_EX_KEYS = [k.upper() for k in EXCLUDE_KEYWORDS]

_COUNTRY_TO_ISO = {
    "GERMANY": "DE",
    "ENGLAND": "GB",
    "FRANCE": "FR",
    "SPAIN": "ES",
    "NETHERLANDS": "NL",
    "HOLLAND": "NL",
    "EGYPT": "EG",
    "BELGIUM": "BE",
    "CHINA": "CN",
    "AUSTRALIA": "AU",
    "ITALY": "IT",
    "CROATIA": "HR",
    "AUSTRIA": "AT",
    "PORTUGAL": "PT",
    "ROMANIA": "RO",
    "SCOTLAND": "GB",
    "SWEDEN": "SE",
    "SWITZERLAND": "CH",
    "TURKEY": "TR",
    "USA": "US",
    "UNITED STATES": "US",
}

def _league_name(fx: dict) -> str:
    return ((fx.get("league") or {}).get("name") or "").upper()

def _country_name(fx: dict) -> str:
    return ((fx.get("league") or {}).get("country") or "").upper()

def _country_iso2_from_name(name: str) -> str:
    name = name.strip().upper()
    return _COUNTRY_TO_ISO.get(name, name)

def _is_allowed_fixture(fx: dict) -> bool:
    lname = _league_name(fx)
    if not lname:
        return False
    for bad in _EX_KEYS:
        if bad in lname:
            return False
    if _ALLOW_KEYS and not any(k in lname for k in _ALLOW_KEYS):
        return False
    iso2 = _country_iso2_from_name(_country_name(fx))
    if _ALLOW_COUNTRIES and iso2 not in _ALLOW_COUNTRIES:
        return False
    return True

# -----------------------------
# API core + rate limiter + cache
# -----------------------------
BASE = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}

# token bucket (per-process)
_tokens = API_RATE_PER_MIN
_last_refill = time.monotonic()

async def _wait_token():
    """Smooth outburst across minutes to avoid 429."""
    global _tokens, _last_refill
    while True:
        now = time.monotonic()
        # refill according to elapsed seconds
        add = (now - _last_refill) * (API_RATE_PER_MIN / 60.0)
        if add >= 1.0:
            _tokens = min(API_RATE_PER_MIN, _tokens + int(add))
            _last_refill = now
        if _tokens > 0:
            _tokens -= 1
            return
        await asyncio.sleep(0.2)

# simple in-memory cache for this process
_cache: Dict[str, Tuple[float, Any]] = {}
_cache_ttl = {
    "get_current_leagues": CACHE_TTL_LEAGUES,
    "get_fixtures_by_date": CACHE_TTL_FIXTURES_BYDATE,
    "get_live_fixtures": CACHE_TTL_LIVE,
    "get_team_statistics": CACHE_TTL_TEAM_STATS,
    "get_odds_for_fixture": CACHE_TTL_ODDS,
}

def _cache_key(path: str, params: Dict[str, Any]) -> str:
    return f"{path}|{tuple(sorted(params.items()))}"

async def _api_get(client: httpx.AsyncClient, path: str, params: Dict[str, Any], cache_name: str) -> Any:
    key = _cache_key(path, params)
    ttl = _cache_ttl.get(cache_name, 0)
    now = time.time()

    # serve from cache
    if ttl > 0 and key in _cache:
        ts, data = _cache[key]
        if now - ts < ttl:
            return data

    url = f"{BASE}{path}"
    retries, backoff = 3, 0.5
    for attempt in range(1, retries + 1):
        async with _sem:
            await _wait_token()
            try:
                r = await client.get(url, params=params, headers=HEADERS, timeout=30.0)
                if r.status_code == 429 or 500 <= r.status_code < 600:
                    if attempt < retries:
                        log("API retry", r.status_code, (r.text or "")[:120])
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * 2, 2.0)
                        continue
                r.raise_for_status()
                data = r.json()
                if data.get("errors"):
                    raise httpx.HTTPError(str(data["errors"]))
                resp = data.get("response")
                if ttl > 0:
                    _cache[key] = (time.time(), resp)
                return resp
            except httpx.HTTPError as e:
                if attempt >= retries:
                    raise
                log("API retry on exception:", str(e))
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 2.0)

# -----------------------------
# Public API
# -----------------------------
async def get_current_leagues(client: httpx.AsyncClient) -> List[Dict[str, Any]]:
    res = await _api_get(client, "/leagues", {"current": "true"}, "get_current_leagues")
    leagues: List[Dict[str, Any]] = []
    for item in res or []:
        seasons = item.get("seasons") or []
        cur = next((s for s in seasons if s.get("current")), None)
        if cur and item.get("league", {}).get("id"):
            leagues.append({
                "leagueId": item["league"]["id"],
                "leagueName": item["league"].get("name"),
                "type": item["league"].get("type"),
                "country": item.get("country", {}).get("name"),
                "season": cur.get("year"),
            })
    return leagues

async def get_fixtures_by_date(client: httpx.AsyncClient, league_id: int, season: int, date_iso: str) -> List[Dict[str, Any]]:
    data = await _api_get(
        client,
        "/fixtures",
        {"league": int(league_id), "season": int(season), "date": str(date_iso)},
        "get_fixtures_by_date",
    ) or []
    return [fx for fx in data if _is_allowed_fixture(fx)]

async def get_team_statistics(client: httpx.AsyncClient, league_id: int, season: int, team_id: int) -> Dict[str, Any]:
    return await _api_get(
        client,
        "/teams/statistics",
        {"league": int(league_id), "season": int(season), "team": int(team_id)},
        "get_team_statistics",
    )

async def get_odds_for_fixture(client: httpx.AsyncClient, fixture_id: int):
    return await _api_get(client, "/odds", {"fixture": int(fixture_id)}, "get_odds_for_fixture")

async def get_live_fixtures(client: httpx.AsyncClient) -> List[Dict[str, Any]]:
    data = await _api_get(client, "/fixtures", {"live": "all"}, "get_live_fixtures") or []
    return [fx for fx in data if _is_allowed_fixture(fx)]
