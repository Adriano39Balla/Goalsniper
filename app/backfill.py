# app/backfill.py
import logging
import time
from typing import List, Dict, Any

from app.db import db_conn
from app.api_client import fetch_fixtures_by_ids


def _list_unlabeled_ids(limit: int = 50) -> List[int]:
    """
    Find match_ids that exist in tip_snapshots but have no final result in match_results.
    """
    with db_conn() as conn:
        cur = conn.execute("""
            SELECT DISTINCT s.match_id
            FROM tip_snapshots s
            LEFT JOIN match_results r ON r.match_id = s.match_id
            WHERE r.match_id IS NULL
            ORDER BY s.created_ts DESC
            LIMIT ?
        """, (limit,))
        return [row["match_id"] for row in cur.fetchall()]


def _backfill_for_ids(ids: List[int]) -> int:
    """
    Fetch final results from API for given fixture IDs and insert into match_results.
    """
    if not ids:
        return 0

    fixtures: Dict[int, Dict[str, Any]] = fetch_fixtures_by_ids(ids)
    now = int(time.time())
    inserted = 0

    with db_conn() as conn:
        for fid, fx in fixtures.items():
            try:
                goals = fx.get("goals") or {}
                gh = goals.get("home")
                ga = goals.get("away")
                if gh is None or ga is None:
                    continue

                # Derive BTTS (both teams scored at least once)
                btts_yes = 1 if (gh > 0 and ga > 0) else 0

                conn.execute("""
                    INSERT OR REPLACE INTO match_results(match_id, final_goals_h, final_goals_a, btts_yes, updated_ts)
                    VALUES (?, ?, ?, ?, ?)
                """, (fid, int(gh), int(ga), btts_yes, now))
                inserted += 1
            except Exception as e:
                logging.exception(f"[BACKFILL] failed for {fid}: {e}")

        conn.commit()

    logging.info(f"[BACKFILL] inserted/updated {inserted} results")
    return inserted


def backfill_results_from_snapshots(batch_size: int = 50) -> int:
    """
    Backfill results for all matches in tip_snapshots that are missing final outcomes.
    Runs in batches until no more missing.
    """
    total = 0
    while True:
        ids = _list_unlabeled_ids(limit=batch_size)
        if not ids:
            break
        inserted = _backfill_for_ids(ids)
        total += inserted
        if inserted == 0:
            break
    return total
