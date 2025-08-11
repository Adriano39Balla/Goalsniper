import asyncio
from typing import Any, Dict, List, Optional, Tuple
import time
import httpx

from .config import API_KEY, MAX_CONCURRENT_REQUESTS
from .logger import log

# ----------------- Filtering config (exported & reused by scanner) -----------------

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
    "U19", "U20", "U21", "U23", "YOUTH", "WOMEN", "FRIENDLY", "CLUB FRIENDLIES",
    "RESERVE", "AMATEUR", "B-TEAM"
]

# ----------------- Helpers for flag/country matching -----------------

def _flag_to_iso2(flag: str) -> str:
    cps = [ord(c) - 0x1F1E6 for c in flag if 0x1F1E6 <= ord(c) <= 0x1F1FF]
    if len(cps) == 2:
        return chr(cps[0] + 65) + chr(cps[1] + 65)
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
_ALLOW_KEYS      = [k.upper() for k in LEAGUE_ALLOW_KEYWORDS]
_EX_KEYS         = [k.upper() for k in EXCLUDE_KEYWORDS]

_COUNTRY_TO_ISO = {
    "GERMANY":"DE","ENGLAND":"GB","FRANCE":"FR","SPAIN":"ES","NETHERLANDS":"NL","HOLLAND":"NL",
    "EGYPT":"EG","BELGIUM":"BE","CHINA":"CN","AUSTRALIA":"AU","ITALY":"IT","CROATIA":"HR",
    "AUSTRIA":"AT","PORTUGAL":"PT","ROMANIA":"RO","SCOTLAND":"GB","SWEDEN":"SE","SWITZERLAND":"CH",
    "TURKEY":"TR","USA":"US","UNITED STATES":"US"
}

def _league_name(fx: dict) -> str:
    return ((fx.get("league") or {}).get("name") or "").upper()

def _country_name(fx: dict) -> str:
    return ((fx.get("league") or {}).get("country") or "").upper()

def _country_iso2_from_name(name: str) -> str:
    return _COUNTRY_TO_ISO.get(name.strip().upper(), name.strip().upper())

def _is_allowed_fixture(fx: dict) -> bool:
    lname = _league_name(fx)
    if not lname:
        return False
    for bad in _EX_KEYS:
        if bad in lname:
            return False
    if not any(k in lname for k in _ALLOW_KEYS):
        return False
    iso2 = _country_iso2_from_name(_country_name(fx))
    if _ALLOW_COUNTRIES and iso2 not in _ALLOW_COUNTRIES:
        return False
    return True

# ----------------- API core (retries + concurrency) -----------------

BASE = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}
_sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

async def _api_get(client: httpx.AsyncClient, path: str, params: Dict[str, Any]) -> Any:
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
                return data.get("response")
            except httpx.HTTPError as e:
                if attempt >= retries:
                    raise
                log("API retry on exception:", str(e))
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 2.0)

# ----------------- Small per‑process caches (reduce duplicate calls) -----------------

# TTLs are conservative – enough for one scan cycle.
_STATS_CACHE: dict[Tuple[int,int,int], Tuple[float, Dict[str, Any]]] = {}
_STATS_TTL = 20 * 60  # 20 minutes

_ODDS_CACHE: dict[int, Tuple[float, Any]] = {}
_ODDS_TTL = 10 * 60   # 10 minutes

def _get_cached_stats(league_id: int, season: int, team_id: int) -> Optional[Dict[str, Any]]:
    key = (int(league_id), int(season), int(team_id))
    rec = _STATS_CACHE.get(key)
    now = time.time()
    if rec and (now - rec[0] <= _STATS_TTL):
        return rec[1]
    return None

def _put_cached_stats(league_id: int, season: int, team_id: int, data: Dict[str, Any]):
    key = (int(league_id), int(season), int(team_id))
    _STATS_CACHE[key] = (time.time(), data)

def _get_cached_odds(fixture_id: int):
    rec = _ODDS_CACHE.get(int(fixture_id))
    now = time.time()
    if rec and (now - rec[0] <= _ODDS_TTL):
        return rec[1]
    return None

def _put_cached_odds(fixture_id: int, data: Any):
    _ODDS_CACHE[int(fixture_id)] = (time.time(), data)

# ----------------- Public API wrappers -----------------

async def get_current_leagues(client: httpx.AsyncClient) -> List[Dict[str, Any]]:
    """
    One request. Scanner will filter aggressively before asking
    for fixtures by date per league.
    """
    res = await _api_get(client, "/leagues", {"current": "true"})
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

async def get_fixtures_by_date(
    client: httpx.AsyncClient,
    league_id: int,
    season: int,
    date_iso: str
) -> List[Dict[str, Any]]:
    """
    Filter at source: only return fixtures from allowed leagues/countries.
    """
    data = await _api_get(client, "/fixtures", {"league": league_id, "season": season, "date": date_iso}) or []
    return [fx for fx in data if _is_allowed_fixture(fx)]

async def get_team_statistics(
    client: httpx.AsyncClient,
    league_id: int,
    season: int,
    team_id: int
) -> Dict[str, Any]:
    """
    Cached per process (TTL ~20m). Dramatically reduces duplicate calls across fixtures.
    """
    cached = _get_cached_stats(league_id, season, team_id)
    if cached is not None:
        return cached
    data = await _api_get(
        client,
        "/teams/statistics",
        {"league": league_id, "season": season, "team": team_id},
    )
    _put_cached_stats(league_id, season, team_id, data)
    return data

async def get_odds_for_fixture(client: httpx.AsyncClient, fixture_id: int):
    """
    Cached per process (TTL ~10m). Odds can be queried multiple times per run.
    """
    cached = _get_cached_odds(fixture_id)
    if cached is not None:
        return cached
    data = await _api_get(client, "/odds", {"fixture": fixture_id})
    _put_cached_odds(fixture_id, data)
    return data

async def get_live_fixtures(client: httpx.AsyncClient) -> List[Dict[str, Any]]:
    """
    Single call for all live fixtures; filtered to your whitelist.
    """
    data = await _api_get(client, "/fixtures", {"live": "all"}) or []
    return [fx for fx in data if _is_allowed_fixture(fx)]
