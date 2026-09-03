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


def _tip(odds=2.10, book="Bet365", lead_sec=300, prev_lead=None):
    """A tips row as capture_closing_lines() now selects it."""
    import time as _t
    return (99, 1000, "PRE Over/Under 2.5", "Over 2.5 Goals", odds, book,
            int(_t.time()) + lead_sec, prev_lead)


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


# (match_id, created_ts, market, suggestion, odds, book, kickoff_ts, closing_lead_sec)
TIP = _tip()


def test_clv_is_measured_against_the_same_books_close(monkeypatch):
    # Bet365 drifted 2.10 -> 2.00 (we beat the close). Another book opened
    # later at 2.60; under the old "best across books" comparison that 2.60
    # would have made this look like a loss.
    n, written = _capture(monkeypatch, TIP, {
        "Bet365": {"Over": 2.00, "Under": 1.80},
        "LateBook": {"Over": 2.60, "Under": 1.55},
    })
    assert n == 1
    closing, clv, _lead, _mid, _cts = written[0]
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
    closing, clv, _lead, _mid, _cts = written[0]
    assert closing == pytest.approx(2.10)
    assert clv == pytest.approx(0.0, abs=0.01), "an unchanged line is zero CLV, not negative"


def test_drifting_against_us_still_reads_negative(monkeypatch):
    # The fix must not simply flatter the numbers - a real adverse move has
    # to still show up as negative CLV.
    _, written = _capture(monkeypatch, TIP, {"Bet365": {"Over": 2.40, "Under": 1.65}})
    _closing, clv, _lead, _mid, _cts = written[0]
    assert clv < 0


def test_nothing_is_recorded_when_that_book_stops_quoting(monkeypatch):
    # Falling back to another book's price would reintroduce the bias.
    n, written = _capture(monkeypatch, TIP, {"SomeOtherBook": {"Over": 2.05, "Under": 1.85}})
    assert n == 0
    assert written == []


def test_nothing_is_recorded_for_a_tip_with_no_book(monkeypatch):
    no_book = _tip(book=None)
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


# ───────── the price has to be the one closest to kickoff ─────────
# This used to filter on `closing_odds IS NULL`, so the FIRST successful
# capture won - the earliest one, up to CLV_CAPTURE_LEAD_MIN before kickoff.
# That is not a closing line. The market sharpens as kickoff approaches, so
# scoring a tip against a T-15 price rather than a T-0 one systematically
# FLATTERS CLV - and CLV is the one instrument meant to say whether the edge
# is real before the P&L can. An error in the flattering direction is the
# worst kind here.

def test_a_later_capture_replaces_an_earlier_one(monkeypatch):
    # Already holds a price taken 600s out; now 120s out, so it must refresh.
    n, written = _capture(monkeypatch, _tip(lead_sec=120, prev_lead=600),
                          {"Bet365": {"Over": 1.95, "Under": 1.90}})
    assert n == 1
    closing, _clv, lead, _mid, _cts = written[0]
    assert closing == pytest.approx(1.95)
    assert lead <= 120, "the stored lead must be the newer, closer one"


def test_an_earlier_capture_does_not_replace_a_later_one(monkeypatch):
    # Cannot happen in the forward direction, but a re-run or a clock skew
    # must never move the benchmark AWAY from kickoff.
    n, written = _capture(monkeypatch, _tip(lead_sec=600, prev_lead=120),
                          {"Bet365": {"Over": 1.95, "Under": 1.90}})
    assert n == 0 and written == []


def test_the_first_capture_is_taken_when_none_exists(monkeypatch):
    n, written = _capture(monkeypatch, _tip(lead_sec=400, prev_lead=None),
                          {"Bet365": {"Over": 2.00, "Under": 1.80}})
    assert n == 1
    assert written[0][2] <= 400


def test_how_close_to_kickoff_the_benchmark_was_is_recorded(monkeypatch):
    # Without this the CLV series cannot be judged, only read.
    _, written = _capture(monkeypatch, _tip(lead_sec=90),
                          {"Bet365": {"Over": 2.00, "Under": 1.80}})
    lead = written[0][2]
    assert 0 <= lead <= 90
