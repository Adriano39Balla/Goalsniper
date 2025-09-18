# file: db.py
# Robust Postgres pool + schema bootstrap for goalsniper

import os
import json
import time
import atexit
import logging
from typing import Optional, Iterable, Any, Tuple
from contextlib import contextmanager

import psycopg2
from psycopg2.pool import SimpleConnectionPool
from psycopg2 import OperationalError, InterfaceError, sql
from psycopg2.extras import DictCursor

log = logging.getLogger("db")

# --- DSN & pool --------------------------------------------------------------

DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise SystemExit("DATABASE_URL is required")

def _should_force_ssl(url: str) -> bool:
    # Only auto-append sslmode for postgres URLs; allow opt-out via env.
    if not url.startswith(("postgres://", "postgresql://")):
        return False
    v = os.getenv("DB_SSLMODE_REQUIRE", "1").strip().lower()
    return v not in {"0", "false", "no", ""}

# Enforce SSL on managed DBs unless already present (configurable)
if _should_force_ssl(DB_URL) and "sslmode=" not in DB_URL:
    DB_URL = DB_URL + (("&" if "?" in DB_URL else "?") + "sslmode=require")

POOL: Optional[SimpleConnectionPool] = None

def _new_conn():
    # application_name helps debugging in pg_stat_activity
    app_name = os.getenv("PG_APP_NAME", "goalsniper")
    return psycopg2.connect(
        DB_URL,
        keepalives=1,
        keepalives_idle=int(os.getenv("PG_KEEPALIVE_IDLE", "30")),
        keepalives_interval=int(os.getenv("PG_KEEPALIVE_INTERVAL", "10")),
        keepalives_count=int(os.getenv("PG_KEEPALIVE_COUNT", "5")),
        application_name=app_name,
    )

def _init_pool():
    global POOL
    if POOL is None:
        minconn = int(os.getenv("DB_POOL_MIN", "1"))
        maxconn = int(os.getenv("DB_POOL_MAX", "6"))
        # build pool using connection factory to apply session settings after acquire
        POOL = SimpleConnectionPool(minconn=minconn, maxconn=maxconn, dsn=DB_URL)
        log.info("[DB] pool created (min=%d, max=%d)", minconn, maxconn)
        atexit.register(_close_pool)

def _close_pool():
    global POOL
    if POOL is not None:
        try:
            POOL.closeall()
            log.info("[DB] pool closed")
        except Exception:
            log.warning("[DB] pool close failed", exc_info=True)
        POOL = None

# --- Session settings ---------------------------------------------------------

STMT_TIMEOUT_MS = int(os.getenv("PG_STATEMENT_TIMEOUT_MS", "15000"))  # 15s
LOCK_TIMEOUT_MS = int(os.getenv("PG_LOCK_TIMEOUT_MS", "2000"))        # 2s
IDLE_TX_TIMEOUT_MS = int(os.getenv("PG_IDLE_TX_TIMEOUT_MS", "30000")) # 30s
FORCE_UTC = os.getenv("PG_FORCE_UTC", "1").strip().lower() not in {"0", "false", "no", ""}

def _apply_session_settings(conn) -> None:
    cur = conn.cursor()
    try:
        if FORCE_UTC:
            cur.execute("SET TIME ZONE 'UTC'")
        cur.execute("SET statement_timeout = %s", (STMT_TIMEOUT_MS,))
        cur.execute("SET lock_timeout = %s", (LOCK_TIMEOUT_MS,))
        cur.execute("SET idle_in_transaction_session_timeout = %s", (IDLE_TX_TIMEOUT_MS,))
    finally:
        cur.close()

# --- Context manager with auto-retry -----------------------------------------

MAX_RETRIES = int(os.getenv("DB_MAX_RETRIES", "2"))
BASE_BACKOFF_MS = int(os.getenv("DB_BASE_BACKOFF_MS", "100"))

class _PooledCursor:
    """
    Autocommit cursor borrowed from pool.
    - Retries transient connection errors with bounded exponential backoff.
    - Applies session settings on first acquire and after reconnect.
    - Optional dict rows via cursor_factory=DictCursor.
    """
    def __init__(self, pool: SimpleConnectionPool, dict_rows: bool = False):
        self.pool = pool
        self.conn = None
        self.cur = None
        self.dict_rows = dict_rows

    def __enter__(self):
        self._acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if self.cur is not None:
                try: self.cur.close()
                except Exception: pass
        finally:
            if self.conn is not None:
                try: self.pool.putconn(self.conn)
                except Exception: pass
            self.conn = None
            self.cur = None

    def _acquire(self):
        _init_pool()
        self.conn = POOL.getconn()  # type: ignore
        self.conn.autocommit = True
        _apply_session_settings(self.conn)
        self.cur = self.conn.cursor(cursor_factory=DictCursor if self.dict_rows else None)

    def _reset(self):
        try:
            if self.cur:
                try: self.cur.close()
                except Exception: pass
        finally:
            try:
                if self.conn:
                    # close=True to drop bad connections
                    self.pool.putconn(self.conn, close=True)
            except Exception:
                pass
        self.conn = None
        self.cur = None
        self._acquire()

    def _retry_loop(self, fn, *args, **kwargs):
        attempts = 0
        while True:
            try:
                return fn(*args, **kwargs)
            except (OperationalError, InterfaceError) as e:
                if attempts >= MAX_RETRIES:
                    log.warning("[DB] giving up after %d retries: %s", attempts, e)
                    raise
                backoff = min(2000, BASE_BACKOFF_MS * (2 ** attempts))
                log.warning("[DB] transient error, retrying in %dms: %s", backoff, e)
                self._reset()
                time.sleep(backoff / 1000.0)
                attempts += 1

    def execute(self, sql_text: str, params: Tuple | list = ()):
        return self._retry_loop(self.cur.execute, sql_text, params or ())

    def executemany(self, sql_text: str, rows: Iterable[Tuple]):
        return self._retry_loop(self.cur.executemany, sql_text, list(rows) or [])

def db_conn(dict_rows: bool = False) -> _PooledCursor:
    """
    Borrow a connection as a cursor wrapper (autocommit).
    Set dict_rows=True to get DictCursor.
    """
    _init_pool()
    return _PooledCursor(POOL, dict_rows=dict_rows)  # type: ignore

# --- Explicit transaction scope ----------------------------------------------

@contextmanager
def tx(dict_rows: bool = False):
    """
    Transaction context:
        with tx() as cur:
            cur.execute("...")  # autocommit disabled inside
    Commits on success, rollbacks on error, with transient-connection auto-retry on BEGIN.
    """
    _init_pool()
    conn = None
    cur = None
    attempts = 0
    while True:
        try:
            conn = POOL.getconn()  # type: ignore
            conn.autocommit = False
            _apply_session_settings(conn)
            cur = conn.cursor(cursor_factory=DictCursor if dict_rows else None)
            break
        except (OperationalError, InterfaceError) as e:
            if attempts >= MAX_RETRIES:
                raise
            if conn:
                try: POOL.putconn(conn, close=True)  # type: ignore
                except Exception: pass
            attempts += 1
            time.sleep(min(2000, BASE_BACKOFF_MS * (2 ** attempts)) / 1000.0)

    try:
        yield cur
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        try:
            if cur:
                cur.close()
        finally:
            if conn:
                try:
                    POOL.putconn(conn)
                except Exception:
                    pass

# --- Schema & migrations ------------------------------------------------------

SCHEMA_SQL = [
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
    # ✅ Fixtures table for scan.py
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
    "CREATE INDEX IF NOT EXISTS idx_tips_created            ON tips(created_ts DESC)",
    "CREATE INDEX IF NOT EXISTS idx_tips_match              ON tips(match_id)",
    "CREATE INDEX IF NOT EXISTS idx_tips_sent               ON tips(sent_ok, created_ts DESC)",
    "CREATE INDEX IF NOT EXISTS idx_snap_by_match           ON tip_snapshots(match_id, created_ts DESC)",
    "CREATE INDEX IF NOT EXISTS idx_results_updated         ON match_results(updated_ts DESC)",
    "CREATE INDEX IF NOT EXISTS idx_odds_hist_match         ON odds_history(match_id, captured_ts DESC)",
    "CREATE INDEX IF NOT EXISTS idx_fixtures_status_update  ON fixtures(status, last_update DESC)"
]

MIGRATIONS_SQL = [
    "ALTER TABLE tips ADD COLUMN IF NOT EXISTS odds           DOUBLE PRECISION",
    "ALTER TABLE tips ADD COLUMN IF NOT EXISTS book           TEXT",
    "ALTER TABLE tips ADD COLUMN IF NOT EXISTS ev_pct         DOUBLE PRECISION",
    "ALTER TABLE tips ADD COLUMN IF NOT EXISTS confidence_raw DOUBLE PRECISION",
    "ALTER TABLE tips ADD COLUMN IF NOT EXISTS sent_ok        INTEGER DEFAULT 1",
]

def init_db() -> None:
    _init_pool()
    with db_conn() as c:
        for sql_stmt in SCHEMA_SQL:
            c.execute(sql_stmt)
        for sql_stmt in MIGRATIONS_SQL:
            try:
                c.execute(sql_stmt)
            except Exception as e:
                log.debug("[DB] migration skipped: %s -> %s", sql_stmt, e)
    log.info("[DB] schema ready")

# --- Settings helpers --------------------------------------------------------

def get_setting(key: str) -> Optional[str]:
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

def get_setting_json(key: str) -> Optional[dict]:
    try:
        raw = get_setting(key)
        return json.loads(raw) if raw else None
    except Exception as e:
        log.warning("[DB] get_setting_json failed for key=%s: %s", key, e)
        return None
