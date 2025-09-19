# file: results_provider.py
# Fetch final scores + BTTS from API-Football and update DB (robust, with retries)

from __future__ import annotations
import os
import time
import logging
from typing import Iterable, Tuple, Optional, List, Dict, Any

import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

from db import db_conn

log = logging.getLogger("results_provider")

# ───────── Env ─────────
API_KEY = os.getenv("APIFOOTBALL_KEY")
BASE_URL = os.getenv("APISPORTS_BASE_URL", "https://v3.football.api-sports.io").rstrip("/")

HTTP_CONNECT_TIMEOUT = float(os.getenv("HTTP_CONNECT_TIMEOUT", "3.0"))
HTTP_RESULTS_TIMEOUT = float(os.getenv("HTTP_RESULTS_TIMEOUT", "10.0"))
HTTP_BACKOFF_FACTOR = float(os.getenv("HTTP_BACKOFF_FACTOR", "0.6"))
HTTP_MAX_RETRIES = int(os.getenv("HTTP_MAX_RETRIES", "3"))

HEADERS = {
    "x-apisports-key": API_KEY or "",
    "Accept": "application/json",
    "User-Agent": os.getenv("HTTP_USER_AGENT", "goalsniper/1.0 (+results)"),
}

# ───────── Session with resilient retries ─────────
_session = requests.Session()
_retry = Retry(
    total=HTTP_MAX_RETRIES,
    connect=HTTP_MAX_RETRIES,
    read=HTTP_MAX_RETRIES,
    backoff_factor=HTTP_BACKOFF_FACTOR,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=frozenset(["GET"]),
    respect_retry_after_header=True,
    raise_on_status=False,
)
_adapter = HTTPAdapter(max_retries=_retry, pool_connections=32, pool_maxsize=64)
_session.mount("https://", _adapter)
_session.mount("http://", _adapter)

def _get(path: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not API_KEY:
        log.debug("[results] APIFOOTBALL_KEY missing; skipping call %s", path)
        return None
    url = f"{BASE_URL}/{path.lstrip('/')}"
    try:
        r = _session.get(url, headers=HEADERS, params=params, timeout=(HTTP_CONNECT_TIMEOUT, HTTP_RESULTS_TIMEOUT))
        if r.status_code == 429:
            # Best effort honor Retry-After (requests/urllib3 will already retry, this is last-resort)
            try:
                ra = int(r.headers.get("Retry-After", "0")) or int(r.json().get("parameters", {}).get("retry_after", 0))
            except Exception:
                ra = 0
            if ra > 0:
                log.warning("[results] 429 rate limited; sleeping %ss (final attempt)", ra)
                time.sleep(min(ra, 15))
                r = _session.get(url, headers=HEADERS, params=params, timeout=(HTTP_CONNECT_TIMEOUT, HTTP_RESULTS_TIMEOUT))
        if not r.ok:
            log.debug("[results] GET %s -> %s (%s)", path, r.status_code, r.text[:200])
            return None
        try:
            js = r.json()
        except ValueError:
            log.warning("[results] invalid JSON from %s", path)
            return None
        return js if isinstance(js, dict) else None
    except Exception as e:
        log.warning("[results] request failed (%s): %s", path, e)
        return None

def _to_int(x, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return default

# ───────── Fetch results ─────────
def fetch_results_for_fixtures(fixture_ids: Iterable[int]) -> List[Tuple[int, int, int, int]]:
    """
    Return [(fixture_id, goals_h, goals_a, btts_yes), ...]
    Uses API-Football /fixtures?id=<id>.
    Robust to partial failures; silently skips not-found/ongoing fixtures.
    """
    out: List[Tuple[int, int, int, int]] = []

    for fid in fixture_ids:
        try:
            js = _get("fixtures", {"id": fid})
            if not js:
                continue
            resp = js.get("response") or []
            if not resp:
                continue

            fix = resp[0]  # id is unique; API returns single item
            goals = (fix.get("goals") or {})
            gh = _to_int(goals.get("home"), 0)
            ga = _to_int(goals.get("away"), 0)

            # Only grade when match is actually over; otherwise skip to avoid premature results.
            # Many API-Football payloads include fixture.status.short (FT/AET/PEN/NS/1H/2H/ET…)
            try:
                short = str(((fix.get("fixture") or {}).get("status") or {}).get("short") or "").upper()
            except Exception:
                short = ""
            finished = short in {"FT", "AET", "PEN"}

            if not finished:
                # If you prefer to accept partials anyway, drop this guard.
                continue

            btts_yes = 1 if (gh > 0 and ga > 0) else 0
            out.append((_to_int(fid), gh, ga, btts_yes))

        except Exception as e:
            log.warning("[results] fixture %s parse error: %s", fid, e)

    return out

# ───────── DB update ─────────
def update_match_results(rows: Iterable[Tuple[int, int, int, int]]) -> int:
    """
    Upsert into match_results(match_id, final_goals_h, final_goals_a, btts_yes, updated_ts).
    Returns number of rows written.
    """
    now = int(time.time())
    n = 0
    try:
        with db_conn() as c:
            for mid, gh, ga, btts in rows:
                c.execute(
                    """
                    INSERT INTO match_results(match_id, final_goals_h, final_goals_a, btts_yes, updated_ts)
                    VALUES (%s,%s,%s,%s,%s)
                    ON CONFLICT (match_id) DO UPDATE SET
                      final_goals_h = EXCLUDED.final_goals_h,
                      final_goals_a = EXCLUDED.final_goals_a,
                      btts_yes      = EXCLUDED.btts_yes,
                      updated_ts    = EXCLUDED.updated_ts
                    """,
                    (_to_int(mid), _to_int(gh), _to_int(ga), 1 if btts else 0, now),
                )
                n += 1
    except Exception as e:
        log.exception("[results] DB upsert failed: %s", e)
    return n
