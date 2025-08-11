import asyncio
import time
from typing import Any, Dict, List
import httpx
from .config import API_KEY, MAX_CONCURRENT_REQUESTS
from .logger import log

# --- NEW: filtering config ---
COUNTRY_FLAGS_ALLOW = "🇩🇪,🇬🇧,🇫🇷,🇪🇸,🇳🇱,🇪🇬,🇧🇪,🇨🇳,🇦🇺,🇮🇹,🇭🇷,🇦🇹,🇵🇹,🇷🇴,🇸🇪,🇨🇭,🇹🇷,🇺🇸"

LEAGUE_ALLOW_KEYWORDS = [
    "DFB POKAL",
    "BUNDESLIGA", "2. BUNDESLIGA",
    "PREMIER LEAGUE",
    "LIGUE 1", "LIGUE 2",
    "LA LIGA", "SEGUNDA",
    "EERSTE DIVISIE",
    "EGYPT PREMIER",
    "JUPILER",
    "CHINESE SUPER",
    "A-LEAGUE",
    "COPPA ITALIA",
    "1. HNL",
    "BUNDESLIGA (AUT)", "2. LIGA",
    "PRIMEIRA LIGA",
    "LIGA I",
    "SCOTTISH PREMIERSHIP",
    "ALLSVENSKAN",
    "SUPER LEAGUE",
    "SUPER LIG", "1. LIG",
    "MLS",
]

EXCLUDE_KEYWORDS = [
    "U19", "U20", "U21", "U23", "YOUTH", "WOMEN", "FRIENDLY", "CLUB FRIENDLIES", "RESERVE", "AMATEUR", "B-TEAM"
]

# --- helpers ---
def _flag_to_iso2(flag: str) -> str:
    code_points = [ord(c) - 0x1F1E6 for c in flag if 0x1F1E6 <= ord(c) <= 0x1F1FF]
    if len(code_points) == 2:
        return chr(code_points[0] + 65) + chr(code_points[1] + 65)
    return flag.upper()

def _normalize_country_flags() -> set[str]:
    out = set()
    for chunk in COUNTRY_FLAGS_ALLOW.split(","):
        s = chunk.strip()
        if not s:
            continue
        if any(0x1F1E6 <= ord(c) <= 0x1F1FF for c in s):
            out.add(_flag_to_iso2(s))
        else:
            out.add(s.upper())
    return out

_ALLOW_COUNTRIES = _normalize_country_flags()
_ALLOW_KEYS = [k.upper() for k in LEAGUE_ALLOW_KEYWORDS]
_EX_KEYS = [k.upper() for k in EXCLUDE_KEYWORDS]

_COUNTRY_TO_ISO = {
    "GERMANY": "DE", "ENGLAND": "GB", "FRANCE": "FR", "SPAIN": "ES", "NETHERLANDS": "NL",
    "HOLLAND": "NL", "EGYPT": "EG", "BELGIUM": "BE", "CHINA": "CN", "AUSTRALIA": "AU",
    "ITALY": "IT", "CROATIA": "HR", "AUSTRIA": "AT", "PORTUGAL": "PT", "ROMANIA": "RO",
    "SCOTLAND": "GB", "SWEDEN": "SE", "SWITZERLAND": "CH", "TURKEY": "TR", "USA": "US", "UNITED STATES": "US"
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
    if not any(k in lname for k in _ALLOW_KEYS):
        return False
    c = _country_name(fx)
    iso2 = _country_iso2_from_name(c)
    if _ALLOW_COUNTRIES and iso2 not in _ALLOW_COUNTRIES:
        return False
    return True

# --- API core ---
BASE = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}
_sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# --- Simple in-memory cache ---
_cache: Dict[str, Tuple[float, Any]] = {}
_cache_ttl = {
    "get_current_leagues": 1800,   # 30 min
    "get_fixtures_by_date": 1800,  # 30 min
    "get_live_fixtures": 90,       # 1.5 min
    "get_team_statistics": 600,    # 10 min
    "get_odds_for_fixture": 600,   # 10 min
}

def _cache_key(path: str, params: Dict[str, Any]) -> str:
    return f"{path}|{sorted(params.items())}"

async def _api_get(client: httpx.AsyncClient, path: str, params: Dict[str, Any], cache_name: str) -> Any:
    key = _cache_key(path, params)
    ttl = _cache_ttl.get(cache_name, 0)
    now = time.time()

    # Serve from cache if valid
    if ttl > 0 and key in _cache:
        ts, data = _cache[key]
        if now - ts < ttl:
            return data

    # Otherwise fetch fresh
    url = f"{BASE}{path}"
    retries, backoff = 3, 0.5
    for attempt in range(1, retries + 1):
        async with _sem:
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
                # Store in cache
                if ttl > 0:
                    _cache[key] = (now, data.get("response"))
                return data.get("response")
            except httpx.HTTPError as e:
                if attempt >= retries:
                    raise
                log("API retry on exception:", str(e))
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 2.0)

# --- Functions ---
async def get_current_leagues(client: httpx.AsyncClient) -> List[Dict[str, Any]]:
    res = await _api_get(client, "/leagues", {"current": "true"}, "get_current_leagues")
    leagues = []
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
    data = await _api_get(client, "/fixtures", {"league": league_id, "season": season, "date": date_iso}, "get_fixtures_by_date") or []
    return [fx for fx in data if _is_allowed_fixture(fx)]

async def get_team_statistics(client: httpx.AsyncClient, league_id: int, season: int, team_id: int) -> Dict[str, Any]:
    return await _api_get(client, "/teams/statistics", {"league": league_id, "season": season, "team": team_id}, "get_team_statistics")

async def get_odds_for_fixture(client: httpx.AsyncClient, fixture_id: int):
    return await _api_get(client, "/odds", {"fixture": fixture_id}, "get_odds_for_fixture")

async def get_live_fixtures(client: httpx.AsyncClient) -> List[Dict[str, Any]]:
    data = await _api_get(client, "/fixtures", {"live": "all"}, "get_live_fixtures") or []
    return [fx for fx in data if _is_allowed_fixture(fx)]
