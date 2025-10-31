# db.py

import os
import sys
import logging
import signal
import atexit
import psycopg2
import psycopg2.pool
from contextlib import contextmanager

log = logging.getLogger(__name__)

# PostgreSQL Connection Pool
POOL: psycopg2.pool.SimpleConnectionPool = None
_SHUTDOWN_RAN = False
_SHUTDOWN_HANDLERS_SET = False


def validate_config():
    if not os.getenv("DATABASE_URL"):
        raise RuntimeError("DATABASE_URL not set")


def _init_pool():
    global POOL
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is not configured.")

    POOL = psycopg2.pool.SimpleConnectionPool(
        minconn=1,
        maxconn=10,
        dsn=db_url,
        sslmode='require'
    )
    log.info("[DB] PostgreSQL connection pool initialized.")


@contextmanager
def db_conn():
    """Database connection context manager."""
    conn = POOL.getconn()
    try:
        yield conn.cursor()
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        POOL.putconn(conn)


def init_db():
    """Ensures the necessary tables exist."""
    with db_conn() as c:
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS prematch_snapshots (
                match_id   BIGINT PRIMARY KEY,
                created_ts BIGINT,
                payload    TEXT
            )
            """
        )
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_pre_snap_ts ON prematch_snapshots (created_ts DESC)"
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS tip_snapshots (
                match_id   BIGINT,
                created_ts BIGINT,
                payload    TEXT,
                PRIMARY KEY (match_id, created_ts)
            )
            """
        )
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_snap_by_match ON tip_snapshots (match_id, created_ts DESC)"
        )
        log.info("[DB] Schema initialized.")


def shutdown_handler(signum=None, frame=None, *, from_atexit: bool = False):
    """Clean up once. Don't call sys.exit() when invoked by atexit."""
    global _SHUTDOWN_RAN
    if _SHUTDOWN_RAN:
        return
    _SHUTDOWN_RAN = True

    try:
        who = "atexit" if from_atexit else ("signal" if signum else "manual")
        log.info(f"[SHUTDOWN] Received shutdown ({who}), cleaning up...")
    except Exception:
        pass

    # Close DB pool if open
    try:
        if POOL:
            POOL.closeall()
            log.info("[DB] Connection pool closed.")
    except Exception as e:
        log.warning("[DB] Error closing pool during shutdown: %s", e)

    # Only exit on signal path; atexit must never sys.exit()
    if not from_atexit:
        try:
            sys.exit(0)
        except SystemExit:
            pass


def register_shutdown_handlers():
    global _SHUTDOWN_HANDLERS_SET
    if _SHUTDOWN_HANDLERS_SET:
        return
    _SHUTDOWN_HANDLERS_SET = True

    signal.signal(signal.SIGINT, lambda s, f: shutdown_handler(s, f, from_atexit=False))
    signal.signal(signal.SIGTERM, lambda s, f: shutdown_handler(s, f, from_atexit=False))
    atexit.register(lambda: shutdown_handler(from_atexit=True))
