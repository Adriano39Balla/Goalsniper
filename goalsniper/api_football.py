import asyncio
from typing import Any, Dict, List, Optional
import httpx
from .config import API_KEY, MAX_CONCURRENT_REQUESTS
from .logger import log

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

async def get_current_leagues(client: httpx.AsyncClient) -> List[Dict[str, Any]]:
    res = await _api_get(client, "/leagues", {"current": "true"})
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
    return await _api_get(client, "/fixtures", {"league": league_id, "season": season, "date": date_iso}) or []

async def get_team_statistics(client: httpx.AsyncClient, league_id: int, season: int, team_id: int) -> Dict[str, Any]:
    return await _api_get(client, "/teams/statistics", {"league": league_id, "season": season, "team": team_id})

async def get_odds_for_fixture(client: httpx.AsyncClient, fixture_id: int):
    return await _api_get(client, "/odds", {"fixture": fixture_id})

async def get_live_fixtures(client: httpx.AsyncClient) -> List[Dict[str, Any]]:
    return await _api_get(client, "/fixtures", {"live": "all"}) or []
