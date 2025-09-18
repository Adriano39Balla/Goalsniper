# db.py — pooled Postgres access + schema bootstrap + settings cache

import os
import time
import logging
from typing import Optional, Tuple, Any

from psycopg2.pool import SimpleConnectionPool
import psycopg2

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

log = logging.getLogger("db")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

# ---------- Connection pool ----------

POOL: Optional[SimpleConnectionPool] = None

def _dsn_with_ssl(url: str) -> str:
    if not url:
        raise SystemExit("DATABASE_URL is required")
    if "sslmode=" not in url:
        url = url + (("&" if "?" in url else "?") + "sslmode=require")
    return url

def _init_pool() -> None:
    global POOL
    if POOL:
        return
    db_url = os.getenv("DATABASE_URL", "")
    dsn = _dsn_with_ssl(db_url)
    POOL = SimpleConnectionPool(
        minconn=1,
        maxconn=int(os.getenv("DB_POOL_MAX", "5")),
        dsn=dsn,
    )

class PooledConn:
    """
    Usage:
      with db_conn() as c:
          c.execute("SELECT 1")
          row = c.fetchone()
    """
    def __init__(self, pool: SimpleConnectionPool):
        self.pool = pool
        self.conn: Optional[psycopg2.extensions.connection] = None
        self.cur: Optional[psycopg2.extensions.cursor] = None

    def __enter__(self) -> "PooledConn":
        self.conn = self.pool.getconn()
        self.conn.autocommit = True
        self.cur = self.conn.cursor()
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if self.cur:
                self.cur.close()
        finally:
            if self.conn:
                self.pool.putconn(self.conn)

    # thin cursor facades
    def execute(self, sql: str, params: Tuple[Any, ...] = ()) -> "PooledConn":
        assert self.cur is not None
        self.cur.execute(sql, params or ())
        return self

    def executemany(self, sql: str, seq_of_params):
        assert self.cur is not None
        self.cur.executemany(sql, seq_of_params)
        return self

    def fetchone(self):
        assert self.cur is not None
        return self.cur.fetchone()

    def fetchall(self):
        assert self.cur is not None
        return self.cur.fetchall()

def db_conn() -> PooledConn:
    if POOL is None:
        _init_pool()
    # type: ignore[arg-type]
    return PooledConn(POOL)  # pyright: ignore

# ---------- Schema bootstrap ----------

def init_db() -> None:
    """
    Creates (or evolves) all tables used by main.py/scan/trainers.
    Idempotent and safe to call on every boot.
    """
    with db_conn() as c:
        # tips + snapshots
        c.execute("""
        CREATE TABLE IF NOT EXISTS tips (
            match_id       BIGINT,
            league_id      BIGINT,
            league         TEXT,
            home           TEXT,
            away           TEXT,
            market         TEXT,
            suggestion     TEXT,
            confidence     DOUBLE PRECISION,
            confidence_raw DOUBLE PRECISION,
            score_at_tip   TEXT,
            minute         INTEGER,
            created_ts     BIGINT,
            odds           DOUBLE PRECISION,
            book           TEXT,
            ev_pct         DOUBLE PRECISION,
            sent_ok        INTEGER DEFAULT 1,
            PRIMARY KEY (match_id, created_ts)
        )""")
        c.execute("""
        CREATE TABLE IF NOT EXISTS tip_snapshots (
            match_id   BIGINT,
            created_ts BIGINT,
            payload    TEXT,
            PRIMARY KEY (match_id, created_ts)
        )""")

        # prematch snapshots (used by save_prematch_snapshot / trainer)
        c.execute("""
        CREATE TABLE IF NOT EXISTS prematch_snapshots (
            match_id   BIGINT PRIMARY KEY,
            created_ts BIGINT,
            payload    TEXT
        )""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_pre_snap_ts ON prematch_snapshots (created_ts DESC)")

        # odds history (aggregated, optional)
        c.execute("""
        CREATE TABLE IF NOT EXISTS odds_history (
            match_id    BIGINT,
            captured_ts BIGINT,
            market      TEXT,
            selection   TEXT,
            odds        DOUBLE PRECISION,
            book        TEXT,
            PRIMARY KEY (match_id, market, selection, captured_ts)
        )""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_odds_hist_match ON odds_history (match_id, captured_ts DESC)")

        # results store for grading / digest
        c.execute("""
        CREATE TABLE IF NOT EXISTS match_results (
            match_id       BIGINT PRIMARY KEY,
            final_goals_h  INTEGER,
            final_goals_a  INTEGER,
            btts_yes       INTEGER,
            updated_ts     BIGINT
        )""")

        # lineups cache (prematch features)
        c.execute("""
        CREATE TABLE IF NOT EXISTS lineups (
            match_id   BIGINT PRIMARY KEY,
            created_ts BIGINT,
            payload    TEXT
        )""")

        # misc
        c.execute("CREATE TABLE IF NOT EXISTS feedback (id SERIAL PRIMARY KEY, match_id BIGINT UNIQUE, verdict INTEGER, created_ts BIGINT)")
        c.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)")

        # evolutive columns (defensive)
        c.execute("ALTER TABLE tips ADD COLUMN IF NOT EXISTS odds DOUBLE PRECISION")
        c.execute("ALTER TABLE tips ADD COLUMN IF NOT EXISTS book TEXT")
        c.execute("ALTER TABLE tips ADD COLUMN IF NOT EXISTS ev_pct DOUBLE PRECISION")
        c.execute("ALTER TABLE tips ADD COLUMN IF NOT EXISTS confidence_raw DOUBLE PRECISION")

        # indices for perf
        c.execute("CREATE INDEX IF NOT EXISTS idx_tips_created ON tips (created_ts DESC)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_tips_match   ON tips (match_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_tips_sent    ON tips (sent_ok, created_ts DESC)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_snap_by_match ON tip_snapshots (match_id, created_ts DESC)")

        log.info("[DB] schema OK")

# ---------- Settings helpers (with small TTL cache) ----------

class _TTLCache:
    def __init__(self, ttl_sec: int):
        self.ttl = ttl_sec
        self.data: dict[str, tuple[float, Optional[str]]] = {}

    def get(self, k: str) -> Optional[str]:
        rec = self.data.get(k)
        if not rec:
            return None
        ts, val = rec
        if time.time() - ts > self.ttl:
            self.data.pop(k, None)
            return None
        return val

    def set(self, k: str, v: Optional[str]) -> None:
        self.data[k] = (time.time(), v)

    def invalidate(self, k: Optional[str] = None) -> None:
        if k is None:
            self.data.clear()
        else:
            self.data.pop(k, None)

_SETTINGS_TTL = int(os.getenv("SETTINGS_TTL_SEC", "60"))
_SETTINGS_CACHE = _TTLCache(_SETTINGS_TTL)

def get_setting(key: str) -> Optional[str]:
    with db_conn() as c:
        row = c.execute("SELECT value FROM settings WHERE key=%s", (key,)).fetchone()
        return row[0] if row else None

def set_setting(key: str, value: str) -> None:
    with db_conn() as c:
        c.execute(
            "INSERT INTO settings(key,value) VALUES(%s,%s) "
            "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
            (key, value),
        )
    _SETTINGS_CACHE.invalidate(key)

def get_setting_cached(key: str) -> Optional[str]:
    v = _SETTINGS_CACHE.get(key)
    if v is None:
        v = get_setting(key)
        _SETTINGS_CACHE.set(key, v)
    return v

def invalidate_model_caches_for_key(_key: str) -> None:
    """
    Placeholder for model-cache invalidation. If your main.py also caches models,
    it can import and call its own invalidator after updating settings.
    We keep a stub here so `from db import invalidate_model_caches_for_key`
    is safe even when main owns the model cache.
    """
    return None
