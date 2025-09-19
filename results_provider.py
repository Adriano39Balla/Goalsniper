# file: results_provider.py
# Fetch final scores + BTTS from API-Football and update DB.

from __future__ import annotations
import os
import time
import logging
from typing import Iterable, Tuple, Optional, List

import requests

from db import db_conn

log = logging.getLogger("results_provider")

# ───────── Env ─────────
API_KEY = os.getenv("APIFOOTBALL_KEY")
BASE_URL = os.getenv("APISPORTS_BASE_URL", "https://v3.football.api-sports.io").rstrip("/")

HEADERS = {
    "x-apisports-key": API_KEY or "",
    "Accept": "application/json",
    "User-Agent": os.getenv("HTTP_USER_AGENT", "goalsniper/1.0 (+results)"),
}

HTTP_TIMEOUT = float(os.getenv("HTTP_RESULTS_TIMEOUT", "10.0"))

# ───────── Fetch results ─────────
def fetch_results_for_fixtures(fixture_ids: Iterable[int]) -> List[Tuple[int, int, int, int]]:
    """
    Return [(fixture_id, goals_h, goals_a, btts_yes), ...]
    Uses API-Football /fixtures endpoint.
    """
    out: List[Tuple[int, int, int, int]] = []
    for fid in fixture_ids:
        try:
            url = f"{BASE_URL}/fixtures"
            resp = requests.get(url, headers=HEADERS, params={"id": fid}, timeout=HTTP_TIMEOUT)
            if not resp.ok:
                log.warning("[results] fixture %s fetch failed (%s)", fid, resp.status_code)
                continue
            js = resp.json()
            data = (js.get("response") or [])
            if not data:
                continue
            fix = data[0]
            goals = fix.get("goals") or {}
            gh = int(goals.get("home") or 0)
            ga = int(goals.get("away") or 0)
            btts = int(gh > 0 and ga > 0)
            out.append((fid, gh, ga, btts))
        except Exception as e:
            log.warning("[results] fixture %s fetch error: %s", fid, e)
    return out

# ───────── DB update ─────────
def update_match_results(rows: Iterable[Tuple[int,int,int,int]]) -> int:
    """
    Upsert into match_results(match_id, final_goals_h, final_goals_a, btts_yes, updated_ts)
    """
    now = int(time.time())
    n = 0
    with db_conn() as c:
        for mid, gh, ga, btts in rows:
            c.execute(
                """
                INSERT INTO match_results(match_id, final_goals_h, final_goals_a, btts_yes, updated_ts)
                VALUES (%s,%s,%s,%s,%s)
                ON CONFLICT (match_id) DO UPDATE SET
                  final_goals_h=EXCLUDED.final_goals_h,
                  final_goals_a=EXCLUDED.final_goals_a,
                  btts_yes=EXCLUDED.btts_yes,
                  updated_ts=EXCLUDED.updated_ts
                """,
                (mid, gh, ga, int(bool(btts)), now),
            )
            n += 1
    return n
