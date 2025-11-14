# app/db_pg.py

import os
import psycopg2
import psycopg2.extras

# You can set this in Railway as env var:
# SUPABASE_DB_URL = full postgres connection string from Supabase
DB_URL = os.getenv("SUPABASE_DB_URL") or os.getenv("DATABASE_URL")


def get_pg_connection():
    """
    Returns a psycopg2 connection to Supabase Postgres.
    Uses dict-like rows (RealDictCursor).
    """
    if not DB_URL:
        raise RuntimeError(
            "SUPABASE_DB_URL / DATABASE_URL not set. "
            "Copy the Postgres connection string from Supabase → Settings → Database."
        )

    conn = psycopg2.connect(DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    return conn


def fetch_all(sql: str, params=None):
    """
    Simple helper: run SELECT and return list[dict].
    """
    conn = get_pg_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, params or ())
                rows = cur.fetchall()
                return rows
    finally:
        conn.close()
