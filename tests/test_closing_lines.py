"""
capture_closing_lines() used to query strictly AFTER kickoff and fetch
PREMATCH odds for a match that had already gone live - the prematch market
closes at kickoff, so that call structurally could never succeed. These
tests check the fixed query window (before kickoff, while the market is
still open) and the CLV arithmetic on a successful capture.
"""
import time

import main


class _FakeCursor:
    def __init__(self, rows=None):
        self._rows = rows or []
        self.executed = []

    def execute(self, sql, params=()):
        self.executed.append((sql, params))
        return self

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return None


class _FakeConn:
    def __init__(self, cursor):
        self._cursor = cursor

    def __enter__(self):
        return self._cursor

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def test_disabled_short_circuits_without_querying(monkeypatch):
    monkeypatch.setattr(main, "CLV_ENABLE", False)
    cursor = _FakeCursor()
    monkeypatch.setattr(main, "db_conn", lambda: _FakeConn(cursor))
    assert main.capture_closing_lines() == 0
    assert cursor.executed == []


def test_queries_a_window_before_kickoff_not_after(monkeypatch):
    monkeypatch.setattr(main, "CLV_ENABLE", True)
    monkeypatch.setattr(main.time, "time", lambda: 1_000_000.0)
    cursor = _FakeCursor(rows=[])
    monkeypatch.setattr(main, "db_conn", lambda: _FakeConn(cursor))

    main.capture_closing_lines(limit=50)

    sql, params = cursor.executed[0]
    now = 1_000_000
    assert params == (now, now + main.CLV_CAPTURE_LEAD_MIN * 60, 50)
    assert "kickoff_ts > %s" in sql
    assert "kickoff_ts <= %s" in sql
    # The old, broken query looked strictly backward from "now" - make sure
    # that shape is gone, not just that a differently-broken one replaced it.
    assert "kickoff_ts <= %s AND kickoff_ts >= %s" not in sql


def test_successful_capture_computes_clv_and_writes_it(monkeypatch):
    monkeypatch.setattr(main, "CLV_ENABLE", True)
    select_cursor = _FakeCursor(rows=[(555, 12345, "BTTS", "BTTS: Yes", 2.0, "Bet365",
                                    int(time.time()) + 300, None)])
    update_cursor = _FakeCursor()
    conns = iter([_FakeConn(select_cursor), _FakeConn(update_cursor)])
    monkeypatch.setattr(main, "db_conn", lambda: next(conns))
    monkeypatch.setattr(main, "fetch_odds", lambda fid, live: {
        "BTTS": {"best": {"Yes": {"odds": 1.9, "book": "OtherBook"}},
                 "by_book": {"Yes": {"Bet365": 1.8, "OtherBook": 1.9}},
                 "n_books": 3},
    })

    n = main.capture_closing_lines()

    assert n == 1
    sql, params = update_cursor.executed[0]
    assert "UPDATE tips SET closing_odds" in sql
    closing_odds, clv_pct, lead_sec, mid, cts = params
    # How close to kickoff the benchmark was taken. Without it the CLV series
    # can be read but not judged: a price captured 15 minutes out is a weaker
    # line than one at the bell, and a weaker line flatters CLV.
    assert 0 <= lead_sec <= 300
    # 1.8 is Bet365's close (the book that priced the tip), NOT the 1.9
    # best-across-books - that superset maximum is the bias being avoided.
    assert closing_odds == 1.8
    # (tip_odds / same_book_closing - 1) * 100 = (2.0 / 1.8 - 1) * 100
    assert clv_pct == round((2.0 / 1.8 - 1.0) * 100.0, 3)
    assert mid == 555
    assert cts == 12345


def test_no_matching_odds_skips_without_writing(monkeypatch):
    monkeypatch.setattr(main, "CLV_ENABLE", True)
    select_cursor = _FakeCursor(rows=[(555, 12345, "BTTS", "BTTS: Yes", 2.0, "Bet365",
                                    int(time.time()) + 300, None)])
    update_cursor = _FakeCursor()
    conns = iter([_FakeConn(select_cursor), _FakeConn(update_cursor)])
    monkeypatch.setattr(main, "db_conn", lambda: next(conns))
    monkeypatch.setattr(main, "fetch_odds", lambda fid, live: {})

    n = main.capture_closing_lines()

    assert n == 0
    assert update_cursor.executed == []
