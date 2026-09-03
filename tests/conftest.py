"""
Test setup for importing main.py in isolation.

main.py runs `_on_boot()` (real DB pool + schema creation) and
`_start_scheduler_once()` unconditionally at import time - there is no
`if __name__ == "__main__":` guard around them. To unit-test pure functions
that happen to live in main.py (like _stake_units), the module has to be
importable without a real Postgres instance:

  - DATABASE_URL only needs to be a non-empty string (main.py checks
    truthiness, not connectivity, at import time).
  - RUN_SCHEDULER=0 skips starting the APScheduler background jobs.
  - psycopg2.pool.ThreadedConnectionPool is replaced with a fake that never
    touches the network, so _init_pool()/init_db()'s CREATE TABLE calls
    against a fake cursor are harmless no-ops.

This must run before anything imports main, so it lives at module level in
this conftest (pytest always loads conftest.py before collecting test files
in the same directory).
"""
import os

import pytest

os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost/test")
os.environ.setdefault("API_KEY", "test-api-key")
os.environ.setdefault("RUN_SCHEDULER", "0")

import psycopg2.pool  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_api_rate_limit_cooldown():
    """
    _api_get() refuses locally while a per-minute rate limit is in force, and
    that flag is module-level state. A test that trips it would otherwise make
    every later test's API call return None for reasons having nothing to do
    with what those tests assert. Cleared on both sides of every test.
    """
    import main
    main._rate_limit_until = 0.0
    yield
    main._rate_limit_until = 0.0


class _FakeCursor:
    def execute(self, *a, **k):
        return self

    def executemany(self, *a, **k):
        return self

    def fetchone(self):
        return None

    def fetchall(self):
        return []

    def close(self):
        pass


class _FakeConn:
    autocommit = True

    def cursor(self):
        return _FakeCursor()

    def close(self):
        pass


class _FakePool:
    def __init__(self, *a, **k):
        pass

    def getconn(self):
        return _FakeConn()

    def putconn(self, conn, close=False):
        pass

    def closeall(self):
        pass


# main.py does `from psycopg2.pool import ThreadedConnectionPool`, so this
# has to replace the attribute on the real module before that import runs.
psycopg2.pool.ThreadedConnectionPool = _FakePool
