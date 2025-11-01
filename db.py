import os
import sys
import logging
import signal
import atexit
import psycopg2
import psycopg2.pool
from contextlib import contextmanager

from config import DATABASE_URL

# ───────────────────────────── Logging ───────────────────────────── #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] - %(message)s"
)
log = logging.getLogger(__name__)

# ───────────────────────────── DB Config ───────────────────────────── #
DB_POOL_MIN = int(os.getenv("DB_POOL_MIN", 1))
DB_POOL_MAX = int(os.getenv("DB_POOL_MAX", 10))

POOL: psycopg2.pool.SimpleConnectionPool = None
_SHUTDOWN_RAN = False
_SHUTDOWN_HANDLERS_SET = False

# ───────────────────────────── Pool Setup ───────────────────────────── #

def validate_config():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set in environment or config.py")


def _init_pool():
    global POOL
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not configured.")

    try:
        POOL = psycopg2.pool.SimpleConnectionPool(
            minconn=DB_POOL_MIN,
            maxconn=DB_POOL_MAX,
            dsn=DATABASE_URL,
            sslmode='require'
        )
        log.info("[DB] PostgreSQL connection pool initialized (%d–%d)", DB_POOL_MIN, DB_POOL_MAX)
    except Exception as e:
        log.exception("[DB] Failed to initialize PostgreSQL pool: %s", e)
        raise


@contextmanager
def db_conn():
    """Database connection context manager."""
    conn = POOL.getconn()
    try:
        with conn.cursor() as cursor:
            yield cursor
        conn.commit()
    except Exception as e:
        conn.rollback()
        log.exception("[DB] Error during DB transaction: %s", e)
        raise
    finally:
        POOL.putconn(conn)


# ───────────────────────────── Init Schema ───────────────────────────── #

def init_db():
    """Ensures the necessary tables exist."""
    with db_conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS prematch_snapshots (
                match_id   BIGINT PRIMARY KEY,
                created_ts BIGINT,
                payload    TEXT
            )
        """)
        c.execute("""
            CREATE INDEX IF NOT EXISTS idx_pre_snap_ts 
            ON prematch_snapshots (created_ts DESC)
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS tip_snapshots (
                match_id   BIGINT,
                created_ts BIGINT,
                payload    TEXT,
                PRIMARY KEY (match_id, created_ts)
            )
        """)
        c.execute("""
            CREATE INDEX IF NOT EXISTS idx_snap_by_match 
            ON tip_snapshots (match_id, created_ts DESC)
        """)
    log.info("[DB] Schema initialized successfully.")


# ───────────────────────────── Shutdown & Signals ───────────────────────────── #

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

    try:
        if POOL:
            POOL.closeall()
            log.info("[DB] Connection pool closed.")
    except Exception as e:
        log.warning("[DB] Error closing pool during shutdown: %s", e)

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


# ───────────────────────────── Optional: Health Check ───────────────────────────── #

def check_db_health() -> bool:
    """Returns True if DB is reachable and working."""
    try:
        with db_conn() as cur:
            cur.execute("SELECT 1")
            return cur.fetchone() == (1,)
    except Exception as e:
        log.warning("[DB] Health check failed: %s", e)
        return False
