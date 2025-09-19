# file: fixtures.py
# Pull upcoming fixtures from API-Football and upsert into DB

from __future__ import annotations
import os, time, logging, datetime as dt
from typing import List, Tuple, Optional

import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from psycopg2.extras import execute_values

from db import db_conn

log = logging.getLogger("fixtures")

API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("APISPORTS_BASE_URL", "https://v3.football.api-sports.io").rstrip("/")

HEADERS = {
    "x-apisports-key": API_KEY or "",
    "Accept": "application/json",
    "User-Agent": os.getenv("HTTP_USER_AGENT", "goalsniper/1.0 (+fixtures)"),
}

HTTP_CONNECT_TIMEOUT = float(os.getenv("HTTP_CONNECT_TIMEOUT", "3.0"))
HTTP_READ_TIMEOUT = float(os.getenv("HTTP_READ_TIMEOUT", "10.0"))

session = requests.Session()
retry = Retry(
    total=3, connect=3, read=3, backoff_factor=0.5,
    status_forcelist=[429,500,502,503,504],
    allowed_methods=frozenset(["GET"]),
    respect_retry_after_header=True, raise_on_status=False,
)
adapter = HTTPAdapter(max_retries=retry, pool_connections=64, pool_maxsize=128)
session.mount("https://", adapter)
session.mount("http://", adapter)

def _api_get(path: str, params: dict) -> Optional[dict]:
    if not API_KEY:
        log.warning("[FIX] API_KEY missing; fixtures sync disabled")
        return None
    url = f"{BASE_URL}/{path.lstrip('/')}"
    try:
        r = session.get(url, headers=HEADERS, params=params,
                        timeout=(HTTP_CONNECT_TIMEOUT, HTTP_READ_TIMEOUT))
        if not r.ok:
            if r.status_code == 429:
                log.info("[FIX] rate limited (429) on %s", path)
            else:
                log.warning("[FIX] %s -> %s %s", path, r.status_code, r.text[:120])
            return None
        js = r.json()
        return js if isinstance(js, dict) else None
    except Exception as e:
        log.warning("[FIX] request failed: %s", e)
        return None

def _parse_row(item: dict):
    fx = item.get("fixture") or {}
    teams = item.get("teams") or {}
    league = item.get("league") or {}

    fixture_id = int(fx.get("id") or 0)
    league_name = str(league.get("name") or "")
    home = str((teams.get("home") or {}).get("name") or "")
    away = str((teams.get("away") or {}).get("name") or "")
    kickoff_iso = fx.get("date")  # e.g. "2025-09-19T18:30:00+00:00"
    status_short = str((fx.get("status") or {}).get("short") or "")
    return (fixture_id, league_name, home, away, kickoff_iso, status_short)

def sync_fixtures(days_ahead: int = 2, include_today: bool = True) -> int:
    """
    Upsert fixtures for today..today+days_ahead.
    Returns number of rows upserted.
    """
    today = dt.date.today()
    dates: List[str] = []
    start = 0 if include_today else 1
    for i in range(start, days_ahead + 1):
        dates.append((today + dt.timedelta(days=i)).isoformat())

    rows: List[Tuple] = []
    for d in ([today.isoformat()] if include_today else []) + dates:
        js = _api_get("fixtures", {"date": d})
        for item in (js or {}).get("response", []):
            fid, league_name, home, away, kickoff_iso, status_short = _parse_row(item)
            if not fid or not kickoff_iso:
                continue
            rows.append((
                fid, league_name, home, away,
                kickoff_iso,  # TIMESTAMPTZ
                dt.datetime.utcnow().isoformat() + "Z",
                status_short
            ))

    if not rows:
        return 0

    sql = """
        INSERT INTO fixtures (fixture_id, league_name, home, away, kickoff, last_update, status)
        VALUES %s
        ON CONFLICT (fixture_id) DO UPDATE SET
            league_name = EXCLUDED.league_name,
            home        = EXCLUDED.home,
            away        = EXCLUDED.away,
            kickoff     = EXCLUDED.kickoff,
            last_update = EXCLUDED.last_update,
            status      = EXCLUDED.status
    """
    with db_conn() as c:
        execute_values(c.cur, sql, rows, page_size=300)  # type: ignore[attr-defined]
    log.info("[FIX] upserted %d fixtures", len(rows))
    return len(rows)
