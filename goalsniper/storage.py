import os
import sqlite3
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple, Dict, Any, List

# DB under /data so it survives service restarts (if you attach a disk)
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

# ----------------- sync primitives -----------------

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
            int(tip.get("messageId") or (message_id or 0)),
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

@_with_conn
def _count_sent_since_sync(conn: sqlite3.Connection, since_iso: str) -> int:
    row = conn.execute("SELECT COUNT(*) AS c FROM tips WHERE sent_at >= ?", (since_iso,)).fetchone()
    return int(row["c"] or 0)

@_with_conn
def _fixture_ever_sent_sync(conn: sqlite3.Connection, fixture_id: int) -> int:
    row = conn.execute("SELECT 1 FROM tips WHERE fixture_id=? LIMIT 1", (int(fixture_id),)).fetchone()
    return 1 if row else 0

@_with_conn
def _has_fixture_recent_sync(conn: sqlite3.Connection, fixture_id: int, since_iso: str) -> int:
    row = conn.execute(
        "SELECT 1 FROM tips WHERE fixture_id=? AND sent_at >= ? LIMIT 1",
        (int(fixture_id), since_iso),
    ).fetchone()
    return 1 if row else 0

# ----------------- async wrappers -----------------

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

async def count_sent_today() -> int:
    start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    return await asyncio.to_thread(_count_sent_since_sync, start)

async def fixture_ever_sent(fixture_id: int) -> bool:
    return bool(await asyncio.to_thread(_fixture_ever_sent_sync, fixture_id))

async def has_fixture_tip_recent(fixture_id: int, minutes: int) -> bool:
    since = (datetime.now(timezone.utc) - timedelta(minutes=int(minutes))).isoformat()
    return bool(await asyncio.to_thread(_has_fixture_recent_sync, fixture_id, since))
