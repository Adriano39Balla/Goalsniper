"""
_price_gate() gates a tip candidate through, in order: odds exist -> odds in
range -> a usable fair price -> EV over the raw price -> edge over the fair
price -> the model-sanity cap on that edge. Each test below isolates one gate
by constructing an odds_map that clears every earlier gate and only trips (or
clears) the one under test.

fetch_odds() normally calls the live API; every test monkeypatches
main.fetch_odds directly so _price_gate never touches the network.
"""
import pytest

import main


def _odds_map(mkey, sel, odds, book="Bet365", fair=None, n_books=5):
    entry = {"best": {sel: {"odds": odds, "book": book}}, "n_books": n_books}
    if fair is not None:
        entry["fair"] = {sel: fair}
    return {mkey: entry}


def test_unmapped_market_short_circuits_before_fetching_odds(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("fetch_odds must not be called for an unmapped market")

    monkeypatch.setattr(main, "fetch_odds", _boom)
    res = main._price_gate("Correct Score", "2-1", fid=1, prob=0.5, live=True)
    assert res["decision"] == "unmapped_market"
    assert res["passed"] is False


def test_no_odds_fails_by_default(monkeypatch):
    monkeypatch.setattr(main, "fetch_odds", lambda fid, live: {})
    res = main._price_gate("BTTS", "BTTS: Yes", fid=1, prob=0.6, live=True)
    assert res["decision"] == "no_odds"
    assert res["passed"] is False


def test_no_odds_passes_when_allow_tips_without_odds(monkeypatch):
    monkeypatch.setattr(main, "fetch_odds", lambda fid, live: {})
    monkeypatch.setattr(main, "ALLOW_TIPS_WITHOUT_ODDS", True)
    res = main._price_gate("BTTS", "BTTS: Yes", fid=1, prob=0.6, live=True)
    assert res["decision"] == "no_odds"
    assert res["passed"] is True


def test_odds_below_market_minimum_are_rejected(monkeypatch):
    # MIN_ODDS_BTTS defaults to 1.30.
    odds_map = _odds_map("BTTS", "Yes", odds=1.10, fair=0.6)
    monkeypatch.setattr(main, "fetch_odds", lambda fid, live: odds_map)
    res = main._price_gate("BTTS", "BTTS: Yes", fid=1, prob=0.8, live=True)
    assert res["decision"] == "odds_out_of_range"
    assert res["passed"] is False


def test_odds_above_global_maximum_are_rejected(monkeypatch):
    # MAX_ODDS_ALL defaults to 20.0.
    odds_map = _odds_map("BTTS", "Yes", odds=25.0, fair=0.05)
    monkeypatch.setattr(main, "fetch_odds", lambda fid, live: odds_map)
    res = main._price_gate("BTTS", "BTTS: Yes", fid=1, prob=0.1, live=True)
    assert res["decision"] == "odds_out_of_range"
    assert res["passed"] is False


def test_too_few_books_is_rejected_when_fair_price_required(monkeypatch):
    # MIN_BOOKS_FOR_FAIR defaults to 3; REQUIRE_FAIR_PRICE defaults to on.
    odds_map = _odds_map("BTTS", "Yes", odds=2.0, fair=0.55, n_books=1)
    monkeypatch.setattr(main, "fetch_odds", lambda fid, live: odds_map)
    res = main._price_gate("BTTS", "BTTS: Yes", fid=1, prob=0.62, live=True)
    assert res["decision"] == "too_few_books"
    assert res["passed"] is False


def test_missing_fair_price_is_rejected_when_required(monkeypatch):
    odds_map = _odds_map("BTTS", "Yes", odds=2.0, fair=None)
    monkeypatch.setattr(main, "fetch_odds", lambda fid, live: odds_map)
    res = main._price_gate("BTTS", "BTTS: Yes", fid=1, prob=0.62, live=True)
    assert res["decision"] == "no_fair_price"
    assert res["passed"] is False


def test_missing_fair_price_can_pass_on_ev_alone_when_not_required(monkeypatch):
    odds_map = _odds_map("BTTS", "Yes", odds=2.0, fair=None)
    monkeypatch.setattr(main, "fetch_odds", lambda fid, live: odds_map)
    monkeypatch.setattr(main, "REQUIRE_FAIR_PRICE", False)
    # ev(0.6, 2.0) = 0.6*2 - 1 = 0.20 -> 2000bps, clears EDGE_MIN_BPS (300).
    res = main._price_gate("BTTS", "BTTS: Yes", fid=1, prob=0.6, live=True)
    assert res["decision"] == "tipped"
    assert res["passed"] is True
    assert res["fair_prob"] is None


def test_ev_below_minimum_is_rejected(monkeypatch):
    # ev(0.505, 2.0) = 0.505*2 - 1 = 0.01 -> 100bps, below EDGE_MIN_BPS (300).
    odds_map = _odds_map("BTTS", "Yes", odds=2.0, fair=0.30)
    monkeypatch.setattr(main, "fetch_odds", lambda fid, live: odds_map)
    res = main._price_gate("BTTS", "BTTS: Yes", fid=1, prob=0.505, live=True)
    assert res["decision"] == "ev_below_min"
    assert res["passed"] is False


def test_fair_edge_below_minimum_is_rejected(monkeypatch):
    # ev(0.60, 2.0) = 0.20 clears EDGE_MIN_BPS, but prob(0.60) - fair(0.585)
    # = 150bps is below FAIR_EDGE_MIN_BPS (200).
    odds_map = _odds_map("BTTS", "Yes", odds=2.0, fair=0.585)
    monkeypatch.setattr(main, "fetch_odds", lambda fid, live: odds_map)
    res = main._price_gate("BTTS", "BTTS: Yes", fid=1, prob=0.60, live=True)
    assert res["decision"] == "fair_edge_below_min"
    assert res["passed"] is False


def test_implausible_edge_over_fair_price_is_suppressed(monkeypatch):
    # prob(0.95) - fair(0.50) = 4500bps, way past MAX_MODEL_EDGE_BPS (800):
    # the model claiming a 45-point edge over a liquid market is a model
    # failure, not a real opportunity.
    odds_map = _odds_map("BTTS", "Yes", odds=3.0, fair=0.50)
    monkeypatch.setattr(main, "fetch_odds", lambda fid, live: odds_map)
    res = main._price_gate("BTTS", "BTTS: Yes", fid=1, prob=0.95, live=True)
    assert res["decision"] == "edge_implausible"
    assert res["passed"] is False


def test_candidate_clearing_every_gate_is_tipped(monkeypatch):
    odds_map = _odds_map("BTTS", "Yes", odds=2.0, fair=0.55)
    monkeypatch.setattr(main, "fetch_odds", lambda fid, live: odds_map)
    res = main._price_gate("BTTS", "BTTS: Yes", fid=1, prob=0.62, live=True)
    assert res["decision"] == "tipped"
    assert res["passed"] is True
    assert res["odds"] == pytest.approx(2.0)
    assert res["fair_prob"] == pytest.approx(0.55)
    assert res["ev_pct"] == pytest.approx(24.0)
    assert res["fair_edge_pct"] == pytest.approx(7.0)


def test_over_under_market_maps_to_line_specific_key(monkeypatch):
    # A distinct gate from BTTS/1X2: the market key is built from the line
    # in the suggestion text (_market_key_and_selection), not a fixed name.
    odds_map = _odds_map("OU_2.5", "Over", odds=2.0, fair=0.55)
    monkeypatch.setattr(main, "fetch_odds", lambda fid, live: odds_map)
    res = main._price_gate("Over/Under 2.5", "Over 2.5 Goals", fid=1, prob=0.62, live=True)
    assert res["decision"] == "tipped"
    assert res["passed"] is True
