# app/db.py
import os
import sqlite3
import logging

# Path to SQLite DB (default: tip_performance.db)
DB_PATH = os.getenv("DB_PATH", "tip_performance.db")


def db_conn():
    """
    Get a SQLite connection with safe defaults for concurrent reads/writes.
    """
    conn = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None)  # autocommit-ish
    conn.row_factory = sqlite3.Row  # return dict-like rows
    try:
        conn.execute("PRAGMA journal_mode=WAL")       # better concurrency
        conn.execute("PRAGMA synchronous=NORMAL")     # balance durability/speed
        conn.execute("PRAGMA busy_timeout=5000")      # wait if DB locked
        conn.execute("PRAGMA foreign_keys=ON")        # enforce constraints
    except Exception as e:
        logging.warning(f"[DB] pragma setup failed: {e}")
    return conn


def init_db():
    """
    Initialize all required tables.
    Existing tables are kept as-is.
    Adds a new `models` table for versioned model persistence.
    """
    with db_conn() as conn:
        # tips: stores predictions/suggestions
        conn.execute("""
        CREATE TABLE IF NOT EXISTS tips (
            match_id INTEGER,
            league_id INTEGER,
            league   TEXT,
            home     TEXT,
            away     TEXT,
            market   TEXT,
            suggestion TEXT,
            confidence REAL,
            score_at_tip TEXT,
            minute    INTEGER,
            created_ts INTEGER,
            sent_ok   INTEGER DEFAULT 1,
            PRIMARY KEY (match_id, created_ts)
        )
        """)

        # tip_snapshots: raw harvested data
        conn.execute("""
        CREATE TABLE IF NOT EXISTS tip_snapshots (
            match_id INTEGER,
            created_ts INTEGER,
            payload TEXT,
            PRIMARY KEY (match_id, created_ts)
        )
        """)

        # feedback: user validation
        conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id INTEGER UNIQUE,
            verdict  INTEGER CHECK (verdict IN (0,1)),
            created_ts INTEGER
        )
        """)

        # settings: app-wide configs + model_coeffs (legacy storage)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """)

        # match_results: final match outcomes
        conn.execute("""
        CREATE TABLE IF NOT EXISTS match_results (
            match_id   INTEGER PRIMARY KEY,
            final_goals_h INTEGER,
            final_goals_a INTEGER,
            btts_yes      INTEGER,
            updated_ts    INTEGER
        )
        """)

        # NEW: models table (for versioned ML persistence)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            market TEXT NOT NULL,
            trained_at TEXT NOT NULL,
            coeffs TEXT NOT NULL,
            metrics TEXT,
            hyperparams TEXT,
            active INTEGER DEFAULT 0
        )
        """)

        # Useful indices
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tip_snaps_created ON tip_snapshots(created_ts)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tips_match ON tips(match_id)")

        # A view for hit rate stats (used for analysis)
        conn.execute("DROP VIEW IF EXISTS v_tip_stats")
        conn.execute("""
        CREATE VIEW v_tip_stats AS
        SELECT t.market, t.suggestion,
               AVG(f.verdict) AS hit_rate,
               COUNT(DISTINCT t.match_id) AS n
        FROM (
          SELECT match_id, market, suggestion, MAX(created_ts) AS last_ts
          FROM tips GROUP BY match_id, market, suggestion
        ) lt
        JOIN tips t ON t.match_id=lt.match_id AND t.created_ts=lt.last_ts
        JOIN feedback f ON f.match_id = t.match_id
        GROUP BY t.market, t.suggestion
        """)

        conn.commit()


def set_setting(key: str, value: str):
    """Insert/update a setting (backwards-compatible with your old code)."""
    with db_conn() as conn:
        conn.execute("""
            INSERT INTO settings(key,value) VALUES(?,?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """, (key, value))
        conn.commit()


def get_setting(key: str, default=None):
    """Fetch a setting, or return default if missing."""
    with db_conn() as conn:
        cur = conn.execute("SELECT value FROM settings WHERE key=?", (key,))
        row = cur.fetchone()
        return row[0] if row else default
