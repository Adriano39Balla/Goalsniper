# file: results_provider.py
# Fetch final scores + BTTS from your data source and update DB.

from __future__ import annotations
import time
from typing import Iterable, Tuple, Optional
from db import db_conn

def fetch_results_for_fixtures(fixture_ids: Iterable[int]) -> Iterable[Tuple[int,int,int,int]]:
    """
    Yield tuples: (fixture_id, goals_h, goals_a, btts_yes)
    Implement with your provider (API-Football /fixtures endpoint or your store).
    """
    # TODO: replace with real API calls; stub returns nothing
    return []

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
