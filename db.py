# file: db.py
# Robust Postgres pool + schema bootstrap for goalsniper

import os
import json
import time
import atexit
import logging
from typing import Optional, Iterable, Tuple
from contextlib import contextmanager

import psycopg2
from psycopg2.pool import SimpleConnectionPool
from psycopg2 import OperationalError, InterfaceError, DatabaseError
from psycopg2.extras import DictCursor

log = logging.getLogger("db")

# --- DSN & pool --------------------------------------------------------------

DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise SystemExit("DATABASE_URL is required")

# Normalize and enforce SSL (configurable)
def _should_force_ssl(url: str) -> bool:
    if not url.startswith(("postgres://", "postgresql://")):
        return False
    v = os.getenv("DB_SSLMODE_REQUIRE", "1").strip().lower()
    return v not in {"0", "false", "no", ""}

if DB_URL.startswith("postgres://"):
    # psycopg2 accepts both, but normalize anyway
    DB_URL = "postgresql://" + DB_URL.split("://", 1)[1]

if _should_force_ssl(DB_URL) and "sslmode=" not in DB_URL:
    DB_URL = DB_URL + (("&" if "?" in DB_URL else "?") + "sslmode=require")

POOL: Optional[SimpleConnectionPool] = None

# Connection kwargs (TCP keepalives help with idle disconnects on managed PG)
_CONN_KW = dict(
    keepalives=1,
    keepalives_idle=int(os.getenv("PG_KEEPALIVE_IDLE", "30")),
    keepalives_interval=int(os.getenv("PG_KEEPALIVE_INTERVAL", "10")),
    keepalives_count=int(os.getenv("PG_KEEPALIVE_COUNT", "5")),
    application_name=os.getenv("PG_APP_NAME", "goalsniper"),
)

def _init_pool():
    global POOL
    if POOL is None:
        minconn = int(os.getenv("DB_POOL_MIN", "1"))
        maxconn = int(os.getenv("DB_POOL_MAX", "6"))
        # Pass keepalive kwargs down to psycopg2.connect via the pool
        POOL = SimpleConnectionPool(minconn=minconn, maxconn=maxconn, dsn=DB_URL, **_CONN_KW)
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

STMT_TIMEOUT_MS = int(os.getenv("PG_STATEMENT_TIMEOUT_MS", "15000"))   # 15s
LOCK_TIMEOUT_MS = int(os.getenv("PG_LOCK_TIMEOUT_MS", "2000"))         # 2s
IDLE_TX_TIMEOUT_MS = int(os.getenv("PG_IDLE_TX_TIMEOUT_MS", "30000"))  # 30s
FORCE_UTC = os.getenv("PG_FORCE_UTC", "1").strip().lower() not in {"0", "false", "no", ""}

def _apply_session_settings(conn) -> None:
    """
    Apply per-connection settings. Guard each SET so a dropped socket or
    permission issue cannot kill the process at import/startup.
    """
    try:
        cur = conn.cursor()
        try:
            if FORCE_UTC:
                try:
                    cur.execute("SET TIME ZONE 'UTC'")
                except Exception as e:
                    # Non-fatal: connection may be mid-termination; we'll retry on next checkout
                    log.debug("[DB] SET TIME ZONE skipped: %s", e)
            try:
                cur.execute("SET statement_timeout = %s", (STMT_TIMEOUT_MS,))
            except Exception as e:
                log.debug("[DB] statement_timeout skip: %s", e)
            try:
                cur.execute("SET lock_timeout = %s", (LOCK_TIMEOUT_MS,))
            except Exception as e:
                log.debug("[DB] lock_timeout skip: %s", e)
            try:
                cur.execute("SET idle_in_transaction_session_timeout = %s", (IDLE_TX_TIMEOUT_MS,))
            except Exception as e:
                log.debug("[DB] idle_tx_timeout skip: %s", e)
        finally:
            try: cur.close()
            except Exception: pass
    except Exception as e:
        # Also non-fatal; acquire code will pre-ping and possibly reconnect
        log.debug("[DB] apply_session_settings failed: %s", e)

# --- Helpers ------------------------------------------------------------------

PING_SQL = "SELECT 1"
MAX_RETRIES = int(os.getenv("DB_MAX_RETRIES", "2"))
BASE_BACKOFF_MS = int(os.getenv("DB_BASE_BACKOFF_MS", "100"))

def _sleep_backoff(attempt: int):
    backoff = min(2000, BASE_BACKOFF_MS * (2 ** attempt))
    time.sleep(backoff / 1000.0)

# --- Pooled cursor with retries & pre-ping -----------------------------------

class _PooledCursor:
    """
    Autocommit cursor borrowed from pool.
    - Retries transient connection errors with bounded exponential backoff.
    - Applies session settings on first acquire and after reconnect.
    - Pre-pings connection to avoid handing out dead sockets.
    - Optional dict rows via cursor_factory=DictCursor.
    - execute()/executemany() return the *cursor* so chaining fetch* is OK.
    """
    def __init__(self, pool: SimpleConnectionPool, dict_rows: bool = False):
        self.pool = pool
        self.conn = None
        self.cur = None
        self.dict_rows = dict_rows

    def __enter__(self):
        self._acquire_with_retry()
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

    def _acquire_once(self):
        self.conn = POOL.getconn()  # type: ignore
        self.conn.autocommit = True
        _apply_session_settings(self.conn)
        # Pre-ping to ensure the socket is alive
        try:
            ping = self.conn.cursor()
            ping.execute(PING_SQL)
            ping.close()
        except Exception:
            # If ping fails, discard this connection and raise so caller can retry
            try: self.pool.putconn(self.conn, close=True)
            except Exception: pass
            self.conn = None
            raise
        # Create the working cursor
        self.cur = self.conn.cursor(cursor_factory=DictCursor if self.dict_rows else None)

    def _acquire_with_retry(self):
        _init_pool()
        attempts = 0
        while True:
            try:
                self._acquire_once()
                return
            except (OperationalError, InterfaceError, DatabaseError) as e:
                if attempts >= MAX_RETRIES:
                    log.warning("[DB] acquire failed after %d retries: %s", attempts, e)
                    raise
                log.warning("[DB] acquire transient error, retrying: %s", e)
                _sleep_backoff(attempts)
                attempts += 1

    def _reset(self):
        try:
            if self.cur:
                try: self.cur.close()
                except Exception: pass
        finally:
            try:
                if self.conn:
                    self.pool.putconn(self.conn, close=True)
            except Exception:
                pass
        self.conn = None
        self.cur = None
        self._acquire_with_retry()

    def _retry_loop(self, fn, *args, **kwargs):
        attempts = 0
        while True:
            try:
                return fn(*args, **kwargs)
            except (OperationalError, InterfaceError) as e:
                if attempts >= MAX_RETRIES:
                    log.warning("[DB] giving up after %d retries: %s", attempts, e)
                    raise
                log.warning("[DB] transient error, resetting connection: %s", e)
                self._reset()
                _sleep_backoff(attempts)
                attempts += 1

    def execute(self, sql_text: str, params: Tuple | list = ()):
        def _call():
            self.cur.execute(sql_text, params or ())
            return self.cur
        return self._retry_loop(_call)

    def executemany(self, sql_text: str, rows: Iterable[Tuple]):
        rows = list(rows) or []
        def _call():
            self.cur.executemany(sql_text, rows)
            return self.cur
        return self._retry_loop(_call)

def db_conn(dict_rows: bool = False) -> _PooledCursor:
    _init_pool()
    return _PooledCursor(POOL, dict_rows=dict_rows)  # type: ignore

# --- Explicit transaction scope ----------------------------------------------

@contextmanager
def tx(dict_rows: bool = False):
    _init_pool()
    conn = None
    cur = None
    attempts = 0
    while True:
        try:
            conn = POOL.getconn()  # type: ignore
            conn.autocommit = False
            _apply_session_settings(conn)
            # Pre-ping inside the transaction too
            ping = conn.cursor()
            try:
                ping.execute(PING_SQL)
            finally:
                try: ping.close()
                except Exception: pass
            cur = conn.cursor(cursor_factory=DictCursor if dict_rows else None)
            break
        except (OperationalError, InterfaceError, DatabaseError) as e:
            if attempts >= MAX_RETRIES:
                if conn:
                    try: POOL.putconn(conn, close=True)  # type: ignore
                    except Exception: pass
                raise
            if conn:
                try: POOL.putconn(conn, close=True)  # type: ignore
                except Exception: pass
            log.warning("[DB] tx acquire transient error, retrying: %s", e)
            attempts += 1
            _sleep_backoff(attempts-1)

    try:
        yield cur
        conn.commit()
    except Exception:
        try: conn.rollback()
        except Exception: pass
        raise
    finally:
        try:
            if cur: cur.close()
        finally:
            if conn:
                try: POOL.putconn(conn)
                except Exception: pass

# --- Schema: tables, migrations, then indexes --------------------------------

SCHEMA_TABLES_SQL = [
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
    """CREATE TABLE IF NOT EXISTS settings ( key TEXT PRIMARY KEY, value TEXT )""",
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
]

MIGRATIONS_SQL = [
    "ALTER TABLE tips ADD COLUMN IF NOT EXISTS odds           DOUBLE PRECISION",
    "ALTER TABLE tips ADD COLUMN IF NOT EXISTS book           TEXT",
    "ALTER TABLE tips ADD COLUMN IF NOT EXISTS ev_pct         DOUBLE PRECISION",
    "ALTER TABLE tips ADD COLUMN IF NOT EXISTS confidence_raw DOUBLE PRECISION",
    "ALTER TABLE tips ADD COLUMN IF NOT EXISTS sent_ok        INTEGER DEFAULT 1",
    # ensure fixtures has these cols even on old DBs
    "ALTER TABLE fixtures ADD COLUMN IF NOT EXISTS last_update TIMESTAMPTZ",
    "ALTER TABLE fixtures ADD COLUMN IF NOT EXISTS status TEXT",
]

INDEX_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_tips_created           ON tips(created_ts DESC)",
    "CREATE INDEX IF NOT EXISTS idx_tips_match             ON tips(match_id)",
    "CREATE INDEX IF NOT EXISTS idx_tips_sent              ON tips(sent_ok, created_ts DESC)",
    "CREATE INDEX IF NOT EXISTS idx_snap_by_match          ON tip_snapshots(match_id, created_ts DESC)",
    "CREATE INDEX IF NOT EXISTS idx_results_updated        ON match_results(updated_ts DESC)",
    "CREATE INDEX IF NOT EXISTS idx_odds_hist_match        ON odds_history(match_id, captured_ts DESC)",
    # run AFTER migrations so columns exist:
    "CREATE INDEX IF NOT EXISTS idx_fixtures_status_update ON fixtures(status, last_update DESC)",
]

def init_db() -> None:
    _init_pool()
    with db_conn() as c:
        # 1) create tables (no-op if exist)
        for sql_stmt in SCHEMA_TABLES_SQL:
            c.execute(sql_stmt)
        # 2) run migrations
        for sql_stmt in MIGRATIONS_SQL:
            try:
                c.execute(sql_stmt)
            except Exception as e:
                log.debug("[DB] migration skipped: %s -> %s", sql_stmt, e)
        # 3) create indexes (after columns exist)
        for sql_stmt in INDEX_SQL:
            try:
                c.execute(sql_stmt)
            except Exception as e:
                log.debug("[DB] index skipped: %s -> %s", sql_stmt, e)
    log.info("[DB] schema ready")

# --- Settings helpers --------------------------------------------------------

def get_setting(key: str) -> Optional[str]:
    with db_conn() as c:
        cur = c.execute("SELECT value FROM settings WHERE key=%s", (key,))
        row = cur.fetchone()
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
