# file: db.py
# Robust Postgres pool + schema bootstrap for goalsniper

import os
import json
import time
import logging
from contextlib import contextmanager

import psycopg2
from psycopg2.pool import SimpleConnectionPool
from psycopg2 import OperationalError, InterfaceError

log = logging.getLogger("db")

# --- DSN & pool --------------------------------------------------------------

DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise SystemExit("DATABASE_URL is required")

# Always enforce SSL on managed DBs unless already present
if "sslmode=" not in DB_URL:
    DB_URL = DB_URL + (("&" if "?" in DB_URL else "?") + "sslmode=require")

POOL: SimpleConnectionPool | None = None

def _new_conn():
    # Keepalives help prevent idle disconnects
    return psycopg2.connect(
        DB_URL,
        keepalives=1,
        keepalives_idle=int(os.getenv("PG_KEEPALIVE_IDLE", "30")),
        keepalives_interval=int(os.getenv("PG_KEEPALIVE_INTERVAL", "10")),
        keepalives_count=int(os.getenv("PG_KEEPALIVE_COUNT", "5")),
    )

def _init_pool():
    global POOL
    if POOL is None:
        maxconn = int(os.getenv("DB_POOL_MAX", "6"))
        POOL = SimpleConnectionPool(minconn=1, maxconn=maxconn, dsn=DB_URL)
        log.info("[DB] pool created (max=%d)", maxconn)

# --- Context manager with auto-retry -----------------------------------------

class _PooledCursor:
    """
    Usage:
        with db_conn() as c:
            cur = c.execute("SELECT 1")
            n = cur.fetchone()[0]
    """
    def __init__(self, pool: SimpleConnectionPool):
        self.pool = pool
        self.conn = None
        self.cur = None

    def __enter__(self):
        self._acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        # Close cursor then return conn to pool
        try:
            if self.cur is not None:
                try:
                    self.cur.close()
                except Exception:
                    pass
        finally:
            if self.conn is not None:
                try:
                    self.pool.putconn(self.conn)
                except Exception:
                    pass
            self.conn = None
            self.cur = None

    def _acquire(self):
        _init_pool()
        self.conn = POOL.getconn()  # type: ignore
        self.conn.autocommit = True
        self.cur = self.conn.cursor()

    def _reset(self):
        # Drop broken connection, grab a fresh one
        try:
            if self.cur:
                try:
                    self.cur.close()
                except Exception:
                    pass
        finally:
            try:
                if self.conn:
                    self.pool.putconn(self.conn, close=True)
            except Exception:
                pass
        self.conn = None
        self.cur = None
        self._acquire()

    def execute(self, sql: str, params: tuple | list = ()):
        try:
            self.cur.execute(sql, params or ())
            return self.cur
        except (OperationalError, InterfaceError) as e:
            log.warning("[DB] execute retry after connection error: %s", e)
            self._reset()
            self.cur.execute(sql, params or ())
            return self.cur

    def executemany(self, sql: str, rows: list[tuple]):
        try:
            self.cur.executemany(sql, rows or [])
            return self.cur
        except (OperationalError, InterfaceError) as e:
            log.warning("[DB] executemany retry after connection error: %s", e)
            self._reset()
            self.cur.executemany(sql, rows or [])
            return self.cur

def db_conn() -> _PooledCursor:
    """Get a pooled cursor context manager."""
    _init_pool()
    return _PooledCursor(POOL)  # type: ignore

# --- Schema & migrations ------------------------------------------------------

SCHEMA_SQL = [
    # core tips storage (no id column by design; identity = match_id + created_ts)
    """
    CREATE TABLE IF NOT EXISTS tips (
        match_id        BIGINT,
        league_id       BIGINT,
        league          TEXT,
        home            TEXT,
        away            TEXT,
        market          TEXT,
        suggestion      TEXT,
        confidence      DOUBLE PRECISION,
        confidence_raw  DOUBLE PRECISION,
        score_at_tip    TEXT,
        minute          INTEGER,
        created_ts      BIGINT,
        odds            DOUBLE PRECISION,
        book            TEXT,
        ev_pct          DOUBLE PRECISION,
        sent_ok         INTEGER DEFAULT 1,
        PRIMARY KEY (match_id, created_ts)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS tip_snapshots (
        match_id   BIGINT,
        created_ts BIGINT,
        payload    TEXT,
        PRIMARY KEY (match_id, created_ts)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS prematch_snapshots (
        match_id   BIGINT PRIMARY KEY,
        created_ts BIGINT,
        payload    TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS feedback (
        id SERIAL PRIMARY KEY,
        match_id BIGINT UNIQUE,
        verdict  INTEGER,
        created_ts BIGINT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS settings (
        key   TEXT PRIMARY KEY,
        value TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS match_results (
        match_id       BIGINT PRIMARY KEY,
        final_goals_h  INTEGER,
        final_goals_a  INTEGER,
        btts_yes       INTEGER,
        updated_ts     BIGINT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS odds_history (
        match_id    BIGINT,
        captured_ts BIGINT,
        market      TEXT,
        selection   TEXT,
        odds        DOUBLE PRECISION,
        book        TEXT,
        PRIMARY KEY (match_id, market, selection, captured_ts)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS lineups (
        match_id   BIGINT PRIMARY KEY,
        created_ts BIGINT,
        payload    TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS fixtures (
        fixture_id   BIGINT PRIMARY KEY,
        league_name  TEXT,
        home         TEXT,
        away         TEXT,
        kickoff      TIMESTAMPTZ,
        last_update  TIMESTAMPTZ,
        status       TEXT
    )
    """,
    # helpful indexes
    "CREATE INDEX IF NOT EXISTS idx_tips_created        ON tips(created_ts DESC)",
    "CREATE INDEX IF NOT EXISTS idx_tips_match          ON tips(match_id)",
    "CREATE INDEX IF NOT EXISTS idx_tips_sent           ON tips(sent_ok, created_ts DESC)",
    "CREATE INDEX IF NOT EXISTS idx_snap_by_match       ON tip_snapshots(match_id, created_ts DESC)",
    "CREATE INDEX IF NOT EXISTS idx_results_updated     ON match_results(updated_ts DESC)",
    "CREATE INDEX IF NOT EXISTS idx_odds_hist_match     ON odds_history(match_id, captured_ts DESC)"
    "CREATE INDEX IF NOT EXISTS idx_fixtures_status     ON fixtures(status)",
]


MIGRATIONS_SQL = [
    # idempotent add-columns in case older tables exist
    "ALTER TABLE tips ADD COLUMN IF NOT EXISTS odds           DOUBLE PRECISION",
    "ALTER TABLE tips ADD COLUMN IF NOT EXISTS book           TEXT",
    "ALTER TABLE tips ADD COLUMN IF NOT EXISTS ev_pct         DOUBLE PRECISION",
    "ALTER TABLE tips ADD COLUMN IF NOT EXISTS confidence_raw DOUBLE PRECISION",
    "ALTER TABLE tips ADD COLUMN IF NOT EXISTS sent_ok        INTEGER DEFAULT 1",
]

def init_db() -> None:
    """Create tables and run safe migrations."""
    _init_pool()
    with db_conn() as c:
        for sql in SCHEMA_SQL:
            c.execute(sql)
        for sql in MIGRATIONS_SQL:
            try:
                c.execute(sql)
            except Exception as e:
                # harmless if not applicable
                log.debug("[DB] migration skipped: %s -> %s", sql, e)
    log.info("[DB] schema ready")

# --- Settings helpers (used by app) ------------------------------------------

def get_setting(key: str) -> str | None:
    with db_conn() as c:
        row = c.execute("SELECT value FROM settings WHERE key=%s", (key,)).fetchone()
        return (row[0] if row else None)

def set_setting(key: str, value: str) -> None:
    with db_conn() as c:
        c.execute(
            "INSERT INTO settings(key,value) VALUES(%s,%s) "
            "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
            (key, value),
        )

def get_setting_json(key: str) -> dict | None:
    try:
        raw = get_setting(key)
        return json.loads(raw) if raw else None
    except Exception:
        return None
