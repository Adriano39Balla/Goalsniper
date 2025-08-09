import os
import sqlite3
import asyncio
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict, Any, List

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "goalsniper.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS tips (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  fixture_id INTEGER NOT NULL,
  market TEXT NOT NULL,
  selection TEXT NOT NULL,
  probability REAL NOT NULL,
  confidence REAL NOT NULL,
  league_id INTEGER,
  season INTEGER,
  sent_at TEXT NOT NULL,
  message_id INTEGER,
  outcome INTEGER
);
CREATE INDEX IF NOT EXISTS idx_tips_fixture ON tips(fixture_id);
CREATE INDEX IF NOT EXISTS idx_tips_market ON tips(market);
CREATE INDEX IF NOT EXISTS idx_tips_league ON tips(league_id, market);
"""

def _ensure_dir():
    d = os.path.dirname(DB_PATH)
    os.makedirs(d, exist_ok=True)

def _with_conn(fn):
    def wrapper(*args, **kwargs):
        _ensure_dir()
        conn = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None)  # autocommit
        try:
            conn.row_factory = sqlite3.Row
            conn.executescript(SCHEMA)
            return fn(conn, *args, **kwargs)
        finally:
            conn.close()
    return wrapper

@_with_conn
def _insert_tip_sync(conn: sqlite3.Connection, tip: Dict[str, Any], message_id: Optional[int]) -> int:
    cur = conn.execute(
        """INSERT INTO tips
           (fixture_id, market, selection, probability, confidence,
            league_id, season, sent_at, message_id, outcome)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)""",
        (
            int(tip["fixtureId"]),
            str(tip["market"]),
            str(tip["selection"]),
            float(tip["probability"]),
            float(tip["confidence"]),
            int(tip.get("leagueId") or 0),
            int(tip.get("season") or 0),
            datetime.now(timezone.utc).isoformat(),
            int(message_id or 0),
        ),
    )
    return int(cur.lastrowid)

@_with_conn
def _set_outcome_sync(conn: sqlite3.Connection, tip_id: int, outcome: int) -> None:
    conn.execute("UPDATE tips SET outcome=? WHERE id=?", (int(outcome), int(tip_id)))

@_with_conn
def _market_stats_sync(conn: sqlite3.Connection, market: str) -> Tuple[int, int]:
    row = conn.execute(
        "SELECT SUM(CASE WHEN outcome=1 THEN 1 ELSE 0 END) AS wins,"
        "       SUM(CASE WHEN outcome=0 THEN 1 ELSE 0 END) AS losses "
        "FROM tips WHERE market=? AND outcome IS NOT NULL",
        (market,),
    ).fetchone()
    return int(row["wins"] or 0), int(row["losses"] or 0)

@_with_conn
def _recent_market_samples_sync(conn: sqlite3.Connection, market: str, limit: int) -> List[sqlite3.Row]:
    cur = conn.execute(
        "SELECT probability, outcome, COALESCE(league_id,0) AS league_id "
        "FROM tips WHERE market=? AND outcome IS NOT NULL "
        "ORDER BY id DESC LIMIT ?",
        (market, int(limit)),
    )
    return cur.fetchall()

@_with_conn
def _recent_market_league_samples_sync(conn: sqlite3.Connection, market: str, league_id: int, limit: int) -> List[sqlite3.Row]:
    cur = conn.execute(
        "SELECT probability, outcome, COALESCE(league_id,0) AS league_id "
        "FROM tips WHERE market=? AND league_id=? AND outcome IS NOT NULL "
        "ORDER BY id DESC LIMIT ?",
        (market, int(league_id or 0), int(limit)),
    )
    return cur.fetchall()

# -------- async wrappers (run sync DB ops off the loop) --------

async def insert_tip_return_id(tip: Dict[str, Any], message_id: Optional[int]) -> int:
    return await asyncio.to_thread(_insert_tip_sync, tip, message_id)

async def set_outcome(tip_id: int, outcome: int):
    await asyncio.to_thread(_set_outcome_sync, tip_id, outcome)

async def market_stats(market: str) -> Tuple[int, int]:
    return await asyncio.to_thread(_market_stats_sync, market)

async def recent_market_samples(market: str, limit: int = 400):
    return await asyncio.to_thread(_recent_market_samples_sync, market, limit)

async def recent_market_league_samples(market: str, league_id: int, limit: int = 120):
    return await asyncio.to_thread(_recent_market_league_samples_sync, market, league_id, limit)
