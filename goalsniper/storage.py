import os
import aiosqlite
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict, Any, List

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "goalsniper.db")

async def _ensure_dir():
    d = os.path.dirname(DB_PATH)
    os.makedirs(d, exist_ok=True)

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

async def get_db() -> aiosqlite.Connection:
    await _ensure_dir()
    db = await aiosqlite.connect(DB_PATH)
    db.row_factory = aiosqlite.Row
    await db.executescript(SCHEMA)
    return db

async def insert_tip_return_id(tip: Dict[str, Any], message_id: Optional[int]) -> int:
    async with await get_db() as db:
        cur = await db.execute(
            """
            INSERT INTO tips (fixture_id, market, selection, probability, confidence,
                              league_id, season, sent_at, message_id, outcome)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
            """,
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
        await db.commit()
        return cur.lastrowid

async def set_outcome(tip_id: int, outcome: int):
    async with await get_db() as db:
        await db.execute("UPDATE tips SET outcome=? WHERE id=?", (int(outcome), int(tip_id)))
        await db.commit()

async def market_stats(market: str) -> Tuple[int, int]:
    async with await get_db() as db:
        cur = await db.execute(
            "SELECT SUM(CASE WHEN outcome=1 THEN 1 ELSE 0 END) AS wins,"
            "       SUM(CASE WHEN outcome=0 THEN 1 ELSE 0 END) AS losses "
            "FROM tips WHERE market=? AND outcome IS NOT NULL",
            (market,),
        )
        row = await cur.fetchone()
        return int(row["wins"] or 0), int(row["losses"] or 0)

async def recent_market_samples(market: str, limit: int = 400) -> List[aiosqlite.Row]:
    async with await get_db() as db:
        cur = await db.execute(
            "SELECT probability, outcome, COALESCE(league_id,0) AS league_id "
            "FROM tips WHERE market=? AND outcome IS NOT NULL "
            "ORDER BY id DESC LIMIT ?",
            (market, int(limit)),
        )
        return await cur.fetchall()

async def recent_market_league_samples(market: str, league_id: int, limit: int = 120) -> List[aiosqlite.Row]:
    async with await get_db() as db:
        cur = await db.execute(
            "SELECT probability, outcome, COALESCE(league_id,0) AS league_id "
            "FROM tips WHERE market=? AND league_id=? AND outcome IS NOT NULL "
            "ORDER BY id DESC LIMIT ?",
            (market, int(league_id or 0), int(limit)),
        )
        return await cur.fetchall()
