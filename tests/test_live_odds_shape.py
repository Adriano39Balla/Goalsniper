"""
The odds API returns two different shapes and only one was handled.

/odds (prematch) nests markets under a list of "bookmakers".
/odds/live is a single aggregated in-play feed that puts the markets
directly under "odds", with no bookmaker layer at all.

fetch_odds only ever looked for "bookmakers", so every live fixture parsed
to zero markets regardless of what the feed contained. In-play candidates
came back no_odds on 100% of scans - 27 consecutive tallies with no other
decision ever recorded - while prematch priced normally through the same
function. Nothing downstream was broken; the prices never arrived.
"""
import pytest

import main


def _prematch_payload():
    return {"response": [{"bookmakers": [
        {"name": "Bet365", "bets": [
            {"name": "Goals Over/Under",
             "values": [{"value": "Over 2.5", "odd": "1.90"},
                        {"value": "Under 2.5", "odd": "1.95"}]}]}]}]}


def _live_payload(values=None):
    """The in-play shape: markets under "odds", no bookmakers key."""
    return {"response": [{
        "fixture": {"id": 42}, "league": {"id": 39}, "teams": {},
        "status": {"elapsed": 63}, "update": "2026-09-02T06:00:00+00:00",
        "odds": [{"id": 5, "name": "Goals Over/Under",
                  "values": values or [{"value": "Over 2.5", "odd": "2.05", "suspended": False},
                                       {"value": "Under 2.5", "odd": "1.80", "suspended": False}]}],
    }]}


def _fetch(monkeypatch, payload, live):
    monkeypatch.setattr(main, "_api_get", lambda url, params, timeout=15: payload)
    main.ODDS_CACHE.invalidate()
    return main.fetch_odds(42, live=live)


def test_live_feed_markets_are_parsed(monkeypatch):
    out = _fetch(monkeypatch, _live_payload(), live=True)
    assert "OU_2.5" in out, "in-play markets must not be silently dropped"
    assert out["OU_2.5"]["best"]["Over"]["odds"] == pytest.approx(2.05)


def test_prematch_shape_still_works(monkeypatch):
    out = _fetch(monkeypatch, _prematch_payload(), live=False)
    assert out["OU_2.5"]["best"]["Over"]["odds"] == pytest.approx(1.90)
    assert out["OU_2.5"]["best"]["Over"]["book"] == "Bet365"


def test_the_live_feed_is_counted_as_exactly_one_source(monkeypatch):
    # It is one aggregated feed, not a panel. Reporting it as more would lend
    # it the credibility of a multi-book consensus it does not have, and
    # n_books gates MIN_BOOKS_FOR_FAIR.
    out = _fetch(monkeypatch, _live_payload(), live=True)
    assert out["OU_2.5"]["n_books"] == 1
    assert out["OU_2.5"]["best"]["Over"]["book"] == main.LIVE_FEED_BOOK


def test_suspended_selections_are_not_treated_as_prices(monkeypatch):
    # A suspended in-play market cannot be taken, so it is not a price.
    out = _fetch(monkeypatch, _live_payload([
        {"value": "Over 2.5", "odd": "2.05", "suspended": True},
        {"value": "Under 2.5", "odd": "1.80", "suspended": False},
    ]), live=True)
    assert "Over" not in (out.get("OU_2.5", {}).get("best") or {})
    assert out["OU_2.5"]["best"]["Under"]["odds"] == pytest.approx(1.80)


def test_an_unrecognised_shape_is_reported_not_swallowed(monkeypatch, caplog):
    # The whole cost of this bug was that it was silent. A response that
    # yields nothing must say what it looked like.
    weird = {"response": [{"fixture": {"id": 42}, "markets_v2": []}]}
    with caplog.at_level("WARNING"):
        out = _fetch(monkeypatch, weird, live=True)
    assert out == {}
    assert any("no usable markets" in r.message for r in caplog.records)
    assert any("markets_v2" in str(r.args) for r in caplog.records)


def test_an_empty_response_is_not_reported_as_a_shape_problem(monkeypatch, caplog):
    # No fixtures priced yet is normal, not a parser fault.
    with caplog.at_level("WARNING"):
        out = _fetch(monkeypatch, {"response": []}, live=True)
    assert out == {}
    assert not any("no usable markets" in r.message for r in caplog.records)


def test_devig_still_works_off_the_single_live_source(monkeypatch):
    out = _fetch(monkeypatch, _live_payload(), live=True)
    fair = out["OU_2.5"]["fair"]
    assert fair["Over"] + fair["Under"] == pytest.approx(1.0, abs=1e-9)


# ───────── the gate the live feed hits next ─────────

def _odds_map(n_books):
    return {"OU_2.5": {"best": {"Over": {"odds": 2.0, "book": main.LIVE_FEED_BOOK}},
                       "fair": {"Over": 0.55}, "n_books": n_books}}


def test_live_book_depth_is_judged_by_its_own_setting(monkeypatch):
    # One aggregated feed can never reach a multi-book threshold, so live
    # would sit at too_few_books forever on the prematch setting.
    monkeypatch.setattr(main, "fetch_odds", lambda fid, live: _odds_map(1))
    monkeypatch.setattr(main, "MIN_BOOKS_FOR_FAIR", 3)
    monkeypatch.setattr(main, "MIN_BOOKS_FOR_FAIR_LIVE", 1)
    res = main._price_gate("Over/Under 2.5", "Over 2.5 Goals", fid=1, prob=0.62, live=True)
    assert res["decision"] == "tipped"


def test_prematch_still_demands_a_real_consensus(monkeypatch):
    # The live allowance must not leak into prematch, which does have books.
    monkeypatch.setattr(main, "fetch_odds", lambda fid, live: _odds_map(1))
    monkeypatch.setattr(main, "MIN_BOOKS_FOR_FAIR", 3)
    monkeypatch.setattr(main, "MIN_BOOKS_FOR_FAIR_LIVE", 1)
    res = main._price_gate("PRE Over/Under 2.5", "Over 2.5 Goals", fid=1, prob=0.62, live=False)
    assert res["decision"] == "too_few_books"


def test_live_defaults_to_the_strict_value_so_nothing_loosens_silently():
    import importlib, os as _os
    assert "MIN_BOOKS_FOR_FAIR_LIVE" not in _os.environ
    assert main.MIN_BOOKS_FOR_FAIR_LIVE == main.MIN_BOOKS_FOR_FAIR


def test_the_diagnostic_names_the_markets_that_were_on_offer(monkeypatch, caplog):
    # Knowing the shape was right but nothing parsed is only half an answer:
    # the next question is whether the feed offered only markets we refuse on
    # purpose, or whether the exclusion list is rejecting a real full-match
    # market. Naming them settles it without another round trip.
    only_refused = {"response": [{
        "fixture": {"id": 7}, "league": {}, "teams": {}, "status": {}, "update": "",
        "odds": [{"name": "Asian Handicap", "values": [{"value": "Home -0.5", "odd": "1.90"}]},
                 {"name": "Corners Over Under", "values": [{"value": "Over 9.5", "odd": "1.85"}]}],
    }]}
    with caplog.at_level("WARNING"):
        out = _fetch(monkeypatch, only_refused, live=True)
    assert out == {}
    msg = " ".join(str(r.args) for r in caplog.records)
    assert "Asian Handicap" in msg and "Corners Over Under" in msg
