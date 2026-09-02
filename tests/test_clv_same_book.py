"""
CLV must compare a price against the SAME bookmaker's closing price.

The tip price is a maximum across whichever books were quoting at the time,
and that set grows towards kickoff. Comparing it against a closing maximum
over MORE books measures book coverage, not line movement: a maximum over a
superset is almost always the larger number, so tip/closing - 1 comes out
negative whatever the market actually did. That is what produced "beat
close 0% of the time" - a structural artefact, not a verdict on the model.

CLV is the metric that decides whether the edge is real, so a smaller
honest sample beats a larger biased one: if the tip's book isn't quoting at
close, nothing is recorded.
"""
import pytest

import main


def _odds_payload(books):
    """books: {book_name: {"Over": odds, ...}} for the OU_2.5 market."""
    return {"response": [{"bookmakers": [
        {"name": name,
         "bets": [{"name": "Goals Over/Under",
                   "values": [{"value": f"{sel} 2.5", "odd": str(o)} for sel, o in sels.items()]}]}
        for name, sels in books.items()]}]}


def _capture(monkeypatch, tip_row, closing_books):
    """Run capture_closing_lines over one tip; return the recorded update."""
    monkeypatch.setattr(main, "_api_get",
                        lambda url, params, timeout=15: _odds_payload(closing_books))
    main.ODDS_CACHE.invalidate()
    written = []

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, sql, params=()):
            if "UPDATE tips SET closing_odds" in sql:
                written.append(params)
            self._rows = [tip_row] if "SELECT match_id" in sql else []
            return self

        def fetchall(self):
            return getattr(self, "_rows", [])

    monkeypatch.setattr(main, "db_conn", lambda: _Conn())
    monkeypatch.setattr(main, "CLV_ENABLE", True)
    n = main.capture_closing_lines(10)
    return n, written


# tip: (match_id, created_ts, market, suggestion, odds, book)
TIP = (99, 1000, "PRE Over/Under 2.5", "Over 2.5 Goals", 2.10, "Bet365")


def test_clv_is_measured_against_the_same_books_close(monkeypatch):
    # Bet365 drifted 2.10 -> 2.00 (we beat the close). Another book opened
    # later at 2.60; under the old "best across books" comparison that 2.60
    # would have made this look like a loss.
    n, written = _capture(monkeypatch, TIP, {
        "Bet365": {"Over": 2.00, "Under": 1.80},
        "LateBook": {"Over": 2.60, "Under": 1.55},
    })
    assert n == 1
    closing, clv, _mid, _cts = written[0]
    assert closing == pytest.approx(2.00)
    assert clv > 0, "beating the same book's close must read as positive CLV"
    assert clv == pytest.approx((2.10 / 2.00 - 1) * 100, abs=0.01)


def test_a_book_joining_late_cannot_drag_clv_negative(monkeypatch):
    # The exact bias: closing best over a superset of books is almost always
    # higher, so every bet looked like it lost to the close.
    _, written = _capture(monkeypatch, TIP, {
        "Bet365": {"Over": 2.10, "Under": 1.75},          # unchanged
        "SharpBook": {"Over": 3.40, "Under": 1.35},        # only quoting now
    })
    closing, clv, _mid, _cts = written[0]
    assert closing == pytest.approx(2.10)
    assert clv == pytest.approx(0.0, abs=0.01), "an unchanged line is zero CLV, not negative"


def test_drifting_against_us_still_reads_negative(monkeypatch):
    # The fix must not simply flatter the numbers - a real adverse move has
    # to still show up as negative CLV.
    _, written = _capture(monkeypatch, TIP, {"Bet365": {"Over": 2.40, "Under": 1.65}})
    _closing, clv, _mid, _cts = written[0]
    assert clv < 0


def test_nothing_is_recorded_when_that_book_stops_quoting(monkeypatch):
    # Falling back to another book's price would reintroduce the bias.
    n, written = _capture(monkeypatch, TIP, {"SomeOtherBook": {"Over": 2.05, "Under": 1.85}})
    assert n == 0
    assert written == []


def test_nothing_is_recorded_for_a_tip_with_no_book(monkeypatch):
    no_book = (99, 1000, "PRE Over/Under 2.5", "Over 2.5 Goals", 2.10, None)
    n, written = _capture(monkeypatch, no_book, {"Bet365": {"Over": 2.00, "Under": 1.80}})
    assert n == 0
    assert written == []


def test_fetch_odds_exposes_per_book_prices(monkeypatch):
    monkeypatch.setattr(main, "_api_get", lambda url, params, timeout=15: _odds_payload({
        "Bet365": {"Over": 1.90, "Under": 1.95},
        "Pinnacle": {"Over": 2.05, "Under": 1.83},
    }))
    main.ODDS_CACHE.invalidate()
    entry = main.fetch_odds(1234, live=False)["OU_2.5"]
    assert entry["by_book"]["Over"] == {"Bet365": 1.90, "Pinnacle": 2.05}
    # "best" is still the max across books - that is correct for EV, where
    # taking the best available price is exactly the point.
    assert entry["best"]["Over"]["odds"] == pytest.approx(2.05)
