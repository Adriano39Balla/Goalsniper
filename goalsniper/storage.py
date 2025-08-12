import os
import sqlite3
import asyncio
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timezone, timedelta

# DB: env override; default to persistent /data on Railway, else /tmp
DB_PATH = os.getenv(
    "DB_PATH",
    "/data/goalsniper.db" if os.path.isdir("/data") else "/tmp/goalsniper.db"
)

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;
PRAGMA busy_timeout=5000;

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
-- Avoid duplicate tips per fixture/market/selection
CREATE UNIQUE INDEX IF NOT EXISTS ux_tips_fixture_market_sel
  ON tips(fixture_id, market, selection);

CREATE INDEX IF NOT EXISTS idx_tips_fixture ON tips(fixture_id);
CREATE INDEX IF NOT EXISTS idx_tips_market ON tips(market);
CREATE INDEX IF NOT EXISTS idx_tips_league ON tips(league_id, market);
CREATE INDEX IF NOT EXISTS idx_tips_sent_at ON tips(sent_at);
CREATE INDEX IF NOT EXISTS idx_tips_fixture_sent_at ON tips(fixture_id, sent_at);

CREATE TABLE IF NOT EXISTS config (
  key TEXT PRIMARY KEY,
  value TEXT
);
"""

def _ensure_dir():
    d = os.path.dirname(DB_PATH)
    if d and d not in (".", "/"):
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

# ---------- tips primitives ----------
@_with_conn
def _insert_tip_sync(conn: sqlite3.Connection, tip: Dict[str, Any], message_id: Optional[int]) -> int:
    try:
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
    except sqlite3.IntegrityError:
        row = conn.execute(
            "SELECT id FROM tips WHERE fixture_id=? AND market=? AND selection=? ORDER BY id DESC LIMIT 1",
            (int(tip["fixtureId"]), str(tip["market"]), str(tip["selection"]))
        ).fetchone()
        return int(row["id"]) if row else 0

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

@_with_conn
def _get_tip_sync(conn: sqlite3.Connection, tip_id: int) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        "SELECT id, fixture_id, market, selection, probability, confidence,"
        "       league_id, season, sent_at, message_id, outcome "
        "FROM tips WHERE id = ?",
        (int(tip_id),),
    ).fetchone()
    if not row:
        return None
    return {
        "id": int(row["id"]),
        "fixtureId": int(row["fixture_id"]),
        "market": str(row["market"]),
        "selection": str(row["selection"]),
        "probability": float(row["probability"]),
        "confidence": float(row["confidence"]),
        "leagueId": int(row["league_id"] or 0),
        "season": int(row["season"] or 0),
        "sentAt": str(row["sent_at"]),
        "messageId": int(row["message_id"] or 0),
        "outcome": (None if row["outcome"] is None else int(row["outcome"])),
    }

# ---------- daily totals ----------
def _day_bounds_utc(d: datetime) -> Tuple[str, str]:
    d = d.astimezone(timezone.utc)
    start = d.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    return start.isoformat(), end.isoformat()

@_with_conn
def _daily_counts_sync(conn: sqlite3.Connection, start_iso: str, end_iso: str) -> dict:
    row = conn.execute(
        "SELECT COUNT(*) AS sent,"
        " SUM(CASE WHEN outcome=1 THEN 1 ELSE 0 END) AS good,"
        " SUM(CASE WHEN outcome=0 THEN 1 ELSE 0 END) AS bad,"
        " SUM(CASE WHEN outcome IS NULL THEN 1 ELSE 0 END) AS pending "
        "FROM tips WHERE sent_at >= ? AND sent_at < ?",
        (start_iso, end_iso),
    ).fetchone()
    return {
        "sent": int(row["sent"] or 0),
        "good": int(row["good"] or 0),
        "bad": int(row["bad"] or 0),
        "pending": int(row["pending"] or 0),
    }

@_with_conn
def _totals_sync(conn: sqlite3.Connection) -> dict:
    row = conn.execute(
        "SELECT COUNT(*) AS sent,"
        " SUM(CASE WHEN outcome=1 THEN 1 ELSE 0 END) AS good,"
        " SUM(CASE WHEN outcome=0 THEN 1 ELSE 0 END) AS bad,"
        " SUM(CASE WHEN outcome IS NULL THEN 1 ELSE 0 END) AS pending "
        "FROM tips",
    ).fetchone()
    return {
        "sent": int(row["sent"] or 0),
        "good": int(row["good"] or 0),
        "bad": int(row["bad"] or 0),
        "pending": int(row["pending"] or 0),
    }

# ---------- config K/V ----------
@_with_conn
def _get_config_bulk_sync(conn: sqlite3.Connection, keys: List[str]) -> Dict[str, str]:
    if not keys:
        return {}
    placeholders = ",".join("?" for _ in keys)
    cur = conn.execute(f"SELECT key, value FROM config WHERE key IN ({placeholders})", keys)
    found = {str(r["key"]): str(r["value"] or "") for r in cur.fetchall()}
    return {k: found.get(k, "") for k in keys}

@_with_conn
def _set_config_sync(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO config(key,value) VALUES(?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (str(key), str(value or "")),
    )

# ---------- async wrappers ----------
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

async def get_tip_by_id(tip_id: int) -> Optional[Dict[str, Any]]:
    return await asyncio.to_thread(_get_tip_sync, tip_id)

async def daily_counts_for(date_dt: datetime) -> dict:
    start_iso, end_iso = _day_bounds_utc(date_dt)
    return await asyncio.to_thread(_daily_counts_sync, start_iso, end_iso)

async def totals() -> dict:
    return await asyncio.to_thread(_totals_sync)

# Config API
async def get_config_bulk(keys: List[str]) -> Dict[str, str]:
    return await asyncio.to_thread(_get_config_bulk_sync, keys)

async def set_config(key: str, value: str) -> None:
    await asyncio.to_thread(_set_config_sync, key, value)
