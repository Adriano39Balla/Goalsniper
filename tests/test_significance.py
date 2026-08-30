"""
compute_market_significance() counts sent tips whose fair_prob never got
stored ("tips_without_fair_price_skipped"), split by phase via the tips
table's is_prematch column. These tests feed it canned rows through a fake
db_conn so the phase split can be verified without a real database.
"""
import main


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def execute(self, sql, params=()):
        return self

    def fetchall(self):
        return self._rows


class _FakeConnCtx:
    def __init__(self, rows):
        self._rows = rows

    def __enter__(self):
        return _FakeCursor(self._rows)

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def _row(mkt, sugg, odds, fair, is_prematch, gh, ga, btts):
    return (mkt, sugg, odds, fair, is_prematch, gh, ga, btts)


def test_skipped_no_fair_price_is_split_by_phase(monkeypatch):
    rows = [
        # Prematch, no fair_prob stored, gradeable -> counts as "pre" skip.
        _row("PRE BTTS", "BTTS: Yes", 1.8, None, 1, 2, 1, 1),
        _row("PRE BTTS", "BTTS: No", 1.8, None, 1, 1, 0, 1),
        # Live, no fair_prob stored, gradeable -> counts as "live" skip.
        _row("BTTS", "BTTS: Yes", 1.8, None, 0, 0, 0, 0),
        # Live, HAS a fair_prob -> not a skip at all (goes into by_market).
        _row("BTTS", "BTTS: Yes", 1.8, 0.5, 0, 3, 1, 1),
        # Draw No Bet voids on a draw -> outcome is None -> ungradeable, must
        # not be counted as a fair-price skip even though fair is None too.
        _row("DNB", "Draw No Bet: Home", 1.5, None, 0, 1, 1, 0),
    ]
    monkeypatch.setattr(main, "db_conn", lambda: _FakeConnCtx(rows))

    out = main.compute_market_significance(days=None, min_n=50)

    skipped = out["tips_without_fair_price_skipped"]
    assert skipped == {"pre": 2, "live": 1, "total": 3}


def test_no_rows_produces_zeroed_breakdown(monkeypatch):
    monkeypatch.setattr(main, "db_conn", lambda: _FakeConnCtx([]))
    out = main.compute_market_significance(days=None, min_n=50)
    assert out["tips_without_fair_price_skipped"] == {"pre": 0, "live": 0, "total": 0}
    assert out["by_market"] == {}
