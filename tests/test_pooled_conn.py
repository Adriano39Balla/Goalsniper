"""
PooledConn must never lose a pool slot.

__exit__ does not run when __enter__ raises, so any connection handed out by
getconn() and then rejected inside __enter__ is invisible to the normal
return path. getconn() readily hands back connections the server has since
closed (Postgres restart, idle timeout), and those fail on `.autocommit =`
or `.cursor()` - so without an explicit hand-back there, every dead
connection permanently cost a slot and ~DB_POOL_MAX of them left the app
unable to reach the database until it was restarted.
"""
import psycopg2
import pytest

import main


class _DeadConn:
    """A connection the server has already closed."""

    def __init__(self):
        self.closed_by_us = False

    def __setattr__(self, name, value):
        if name == "autocommit":
            raise psycopg2.InterfaceError("connection already closed")
        object.__setattr__(self, name, value)

    def cursor(self):
        raise psycopg2.InterfaceError("connection already closed")

    def close(self):
        object.__setattr__(self, "closed_by_us", True)


class _LiveConn:
    def __init__(self):
        self.autocommit = False

    def cursor(self):
        return object()

    def close(self):
        pass


class _Pool:
    """Tracks what was handed out and what came back."""

    def __init__(self, conns):
        self._conns = list(conns)
        self.handed_out = []
        self.returned = []

    def getconn(self):
        if not self._conns:
            raise psycopg2.pool.PoolError("exhausted")
        c = self._conns.pop(0)
        self.handed_out.append(c)
        return c

    def putconn(self, conn, close=False):
        self.returned.append((conn, close))


def test_dead_connection_is_returned_not_leaked():
    pool = _Pool([_DeadConn(), _LiveConn()])
    with main.PooledConn(pool) as c:
        assert c is not None
    # Both were handed out; the dead one must have gone back.
    assert len(pool.handed_out) == 2
    dead = pool.handed_out[0]
    assert any(conn is dead for conn, _ in pool.returned), "dead connection leaked"


def test_dead_connection_is_returned_with_close_so_it_is_not_recycled():
    pool = _Pool([_DeadConn(), _LiveConn()])
    with main.PooledConn(pool):
        pass
    dead = pool.handed_out[0]
    closed_flags = [close for conn, close in pool.returned if conn is dead]
    assert closed_flags == [True]


def test_a_dead_connection_is_retried_rather_than_raising():
    # Previously InterfaceError wasn't caught at all, so one dead connection
    # failed the caller outright even with healthy ones behind it.
    pool = _Pool([_DeadConn(), _DeadConn(), _LiveConn()])
    with main.PooledConn(pool) as c:
        assert c.conn is pool.handed_out[-1]


def test_every_dead_connection_in_a_row_still_returns_them_all():
    pool = _Pool([_DeadConn() for _ in range(5)])
    with pytest.raises(psycopg2.InterfaceError):
        with main.PooledConn(pool):
            pass
    assert len(pool.returned) == len(pool.handed_out) == 5


def test_healthy_connection_is_not_closed_on_the_way_in():
    pool = _Pool([_LiveConn()])
    with main.PooledConn(pool) as c:
        assert c.conn is pool.handed_out[0]
    # Returned once, by __exit__, and NOT force-closed.
    assert [close for _, close in pool.returned] == [False]


def test_pool_exhaustion_still_raises_after_its_retries():
    pool = _Pool([])
    with pytest.raises(psycopg2.pool.PoolError):
        with main.PooledConn(pool):
            pass
    # Nothing was handed out, so nothing should have been returned.
    assert pool.returned == []
