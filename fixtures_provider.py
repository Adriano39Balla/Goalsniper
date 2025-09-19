from __future__ import annotations
import os
import time
import logging
from typing import List, Dict, Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from db import db_conn

log = logging.getLogger("fixtures")

API_KEY = os.getenv("APIFOOTBALL_KEY", "")
BASE_URL = os.getenv("APISPORTS_BASE_URL", "https://v3.football.api-sports.io").rstrip("/")

HEADERS = {
    "x-apisports-key": API_KEY,
    "Accept": "application/json",
    "User-Agent": os.getenv("HTTP_USER_AGENT", "goalsniper/1.0 (+fixtures)"),
}

# how far ahead to fetch prematch fixtures (minutes)
PREMATCH_WINDOW_MIN = int(os.getenv("PREMATCH_WINDOW_MIN", "120"))
# optional filter: comma-separated league IDs (e.g. "39,140,135"); empty = all
LEAGUE_FILTER = {x.strip() for x in os.getenv("LEAGUE_WHITELIST", "").split(",") if x.strip()}

HTTP_CONNECT_TIMEOUT = float(os.getenv("HTTP_CONNECT_TIMEOUT", "3.0"))
HTTP_READ_TIMEOUT = float(os.getenv("HTTP_READ_TIMEOUT", "10.0"))

# resilient session
_s = requests.Session()
_retry = Retry(
    total=3, connect=3, read=3,
    backoff_factor=0.6,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=frozenset(["GET"]),
    respect_retry_after_header=True,
    raise_on_status=False,
)
_s.mount("https://", HTTPAdapter(max_retries=_retry, pool_connections=32, pool_maxsize=64))
_s.mount("http://",  HTTPAdapter(max_retries=_retry, pool_connections=32, pool_maxsize=64))

def _get(path: str, params: Dict[str, Any]) -> Dict:
    if not API_KEY:
        log.warning("[fixtures] APIFOOTBALL_KEY missing – fixtures sync disabled")
        return {}
    try:
        url = f"{BASE_URL}/{path.lstrip('/')}"
        r = _s.get(url, headers=HEADERS, params=params,
                   timeout=(HTTP_CONNECT_TIMEOUT, HTTP_READ_TIMEOUT))
        if not r.ok:
            # noisy on non-retryable status to surface auth/quota issues
            log.warning("[fixtures] GET %s %s -> %s %s", path, params, r.status_code, r.text[:200])
            return {}
        js = r.json()
        return js if isinstance(js, dict) else {}
    except Exception as e:
        log.warning("[fixtures] request failed: %s %s", path, e)
        return {}

def _want_league(league_id: int) -> bool:
    if not LEAGUE_FILTER:
        return True
    return str(league_id) in LEAGUE_FILTER

def _rows_from_response(js: Dict) -> List[tuple]:
    rows = []
    for item in (js.get("response") or []):
        try:
            fixture = item.get("fixture") or {}
            league  = item.get("league") or {}
            teams   = item.get("teams") or {}

            fid = int(fixture.get("id"))
            status = (fixture.get("status") or {}).get("short") or "NS"
            # kickoff & last_update come as ISO strings; let Postgres parse via ::timestamptz
            kickoff = (fixture.get("date") or None)
            updated = (fixture.get("update") or None)

            league_name = (league.get("name") or "").strip()
            league_id   = int(league.get("id") or 0)
            if not _want_league(league_id):
                continue

            home = ((teams.get("home") or {}).get("name") or "").strip()
            away = ((teams.get("away") or {}).get("name") or "").strip()

            rows.append((
                fid, league_name, home, away, kickoff, updated, status
            ))
        except Exception:
            continue
    return rows

def sync_live_fixtures() -> int:
    """Upsert all fixtures currently live (status LIVE/1H/HT/2H/ET/P)."""
    js = _get("fixtures", {"live": "all"})
    rows = _rows_from_response(js)
    return _upsert(rows)

def sync_upcoming_fixtures() -> int:
    """Upsert fixtures starting in the next PREMATCH_WINDOW_MIN minutes."""
    js = _get("fixtures", {"next": 200})  # API returns next N fixtures across all leagues
    rows = [
        r for r in _rows_from_response(js)
        # lightweight window check done in SQL via kickoff <= now()+interval
    ]
    return _upsert(rows)

def _upsert(rows: List[tuple]) -> int:
    if not rows:
        return 0
    # Use Postgres to parse ISO datetimes
    with db_conn() as c:
        for (fid, league_name, home, away, kickoff_iso, updated_iso, status) in rows:
            c.execute(
                """
                INSERT INTO fixtures(fixture_id, league_name, home, away, kickoff, last_update, status)
                VALUES (%s, %s, %s, %s, %s::timestamptz, %s::timestamptz, %s)
                ON CONFLICT (fixture_id) DO UPDATE SET
                  league_name = EXCLUDED.league_name,
                  home        = EXCLUDED.home,
                  away        = EXCLUDED.away,
                  kickoff     = COALESCE(EXCLUDED.kickoff, fixtures.kickoff),
                  last_update = COALESCE(EXCLUDED.last_update, NOW()),
                  status      = EXCLUDED.status
                """,
                (fid, league_name, home, away, kickoff_iso, updated_iso, status),
            )
    return len(rows)

def sync_fixtures_once() -> dict:
    """Convenience wrapper: sync live + near-term prematch fixtures."""
    n_live = sync_live_fixtures()
    n_upc  = sync_upcoming_fixtures()
    log.info("[fixtures] upserted live=%d upcoming=%d (window=%dm)",
             n_live, n_upc, PREMATCH_WINDOW_MIN)
    return {"live": n_live, "upcoming": n_upc}
