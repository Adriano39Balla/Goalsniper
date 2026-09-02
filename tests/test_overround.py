"""
Market width, shown on every tip and gated on when it gets silly.

devig() removes the bookmaker's margin by scaling every selection down by the
same factor. Books do not add the margin that way: favourite-longshot bias
puts more of it on the longshot, so proportional de-vig takes too much off the
favourite and hands back a fair probability that is too LOW - a fair price
that looks too long, and an "edge" on the favourite side that is partly our
own arithmetic.

The live BTTS: No tip that lost was exactly that shape: taken at 1.53 (the
favourite side) against a fair 1.65 derived from a market running 9.3% wide.
The wider the market, the more of that 7.5pp edge was method error. So the
overround is not a cost line, it is an error bar on the number the gate is
built around - and past some width the fair price should not be bet against
at all.
"""
import pytest

import main


def _payload(values, name="Goals Over/Under", book="Bet365"):
    return {"response": [{"bookmakers": [{"name": book, "bets": [
        {"name": name, "values": values}]}]}]}


def _fetch(monkeypatch, payload):
    monkeypatch.setattr(main, "_api_get", lambda url, params, timeout=15: payload)
    main.ODDS_CACHE.invalidate()
    return main.fetch_odds(42, live=False)


# ───────── measuring it ─────────

def test_a_two_way_market_reports_its_margin(monkeypatch):
    # 1.53 / 2.28 -> 65.36% + 43.86% = 109.22%, i.e. 9.2% wide. This is the
    # market the losing BTTS: No tip was priced from.
    out = _fetch(monkeypatch, _payload([{"value": "Over 2.5", "odd": "1.53"},
                                        {"value": "Under 2.5", "odd": "2.28"}]))
    assert out["OU_2.5"]["overround"] == pytest.approx(0.0922, abs=1e-3)


def test_a_fair_market_reports_near_zero(monkeypatch):
    out = _fetch(monkeypatch, _payload([{"value": "Over 2.5", "odd": "2.00"},
                                        {"value": "Under 2.5", "odd": "2.00"}]))
    assert out["OU_2.5"]["overround"] == pytest.approx(0.0, abs=1e-9)


def test_double_chance_is_measured_against_its_own_total(monkeypatch):
    # DC selections sum to 2.0, not 1.0. Measuring against 1.0 would report a
    # ~110% margin on a normal market and gate every DC candidate away - the
    # same class of bug that halved every DC fair price.
    out = _fetch(monkeypatch, _payload(
        [{"value": "Home/Draw", "odd": "1.30"},
         {"value": "Draw/Away", "odd": "1.60"},
         {"value": "Home/Away", "odd": "1.35"}], name="Double Chance"))
    assert out["DC"]["overround"] < 0.25


def test_it_is_measured_per_book_not_across_best_prices(monkeypatch):
    # "best" is a maximum across books, so implied probabilities built from it
    # sum to LESS than the truth. Reported that way a two-book market would
    # show a negative overround - the book paying you to bet.
    two_books = {"response": [{"bookmakers": [
        {"name": "A", "bets": [{"name": "Goals Over/Under", "values": [
            {"value": "Over 2.5", "odd": "1.90"}, {"value": "Under 2.5", "odd": "1.95"}]}]},
        {"name": "B", "bets": [{"name": "Goals Over/Under", "values": [
            {"value": "Over 2.5", "odd": "2.05"}, {"value": "Under 2.5", "odd": "1.85"}]}]},
    ]}]}
    out = _fetch(monkeypatch, two_books)
    assert out["OU_2.5"]["overround"] > 0, "a real book never prices under 100%"
    assert out["OU_2.5"]["best"]["Over"]["odds"] == pytest.approx(2.05)


def test_an_incomplete_market_reports_no_overround(monkeypatch):
    # One side quoted says nothing about the margin.
    out = _fetch(monkeypatch, _payload([{"value": "Over 2.5", "odd": "1.90"}]))
    assert out["OU_2.5"]["overround"] is None


# ───────── gating on it ─────────

def _odds_map(overround, fair=0.55, odds=2.0, mkey="OU_2.5"):
    return {mkey: {"best": {"Over": {"odds": odds, "book": "B"},
                            "Home": {"odds": odds, "book": "B"}},
                   "fair": {"Over": fair, "Home": fair},
                   "overround": overround, "n_books": 3}}


def _gate(monkeypatch, overround, market="Over/Under 2.5", sel="Over 2.5 Goals",
          mkey="OU_2.5", prob=0.62):
    monkeypatch.setattr(main, "fetch_odds",
                        lambda fid, live: _odds_map(overround, mkey=mkey))
    monkeypatch.setattr(main, "MIN_BOOKS_FOR_FAIR", 1)
    return main._price_gate(market, sel, fid=1, prob=prob, live=True)


def test_a_normal_market_passes_and_reports_its_width(monkeypatch):
    res = _gate(monkeypatch, 0.06)
    assert res["decision"] == "tipped"
    assert res["overround_pct"] == pytest.approx(6.0)


def test_a_market_wider_than_the_cap_is_refused(monkeypatch):
    monkeypatch.setattr(main, "MAX_OVERROUND_BPS", 1200)
    res = _gate(monkeypatch, 0.19)
    assert res["decision"] == "overround_too_wide"
    assert res["passed"] is False


def test_the_refusal_happens_before_any_edge_is_claimed(monkeypatch):
    # At this width the fair price is the thing in doubt, so an edge measured
    # against it is quantifying our own de-vig error. Reporting an EV here
    # would be reporting that error as an opportunity.
    monkeypatch.setattr(main, "MAX_OVERROUND_BPS", 1200)
    res = _gate(monkeypatch, 0.19)
    assert res["ev_pct"] is None
    assert "fair_edge_pct" not in res


def test_three_way_markets_get_their_own_cap(monkeypatch):
    # A book pricing three outcomes carries a mechanically larger margin. One
    # cap for both either waves 1X2 through or strangles BTTS.
    monkeypatch.setattr(main, "MAX_OVERROUND_BPS", 1200)
    monkeypatch.setattr(main, "MAX_OVERROUND_BPS_3WAY", 1800)
    assert main._max_overround_bps("OU_2.5") == 1200
    assert main._max_overround_bps("BTTS") == 1200
    assert main._max_overround_bps("1X2") == 1800
    assert main._max_overround_bps("DC") == 1800

    wide = _gate(monkeypatch, 0.15, market="1X2", sel="Home Win", mkey="1X2")
    assert wide["decision"] == "tipped", "15% is normal for a three-way market"
    narrow = _gate(monkeypatch, 0.15)
    assert narrow["decision"] == "overround_too_wide"


def test_a_zero_cap_disables_the_gate_without_hiding_the_number(monkeypatch):
    monkeypatch.setattr(main, "MAX_OVERROUND_BPS", 0)
    res = _gate(monkeypatch, 0.40)
    assert res["decision"] == "tipped"
    assert res["overround_pct"] == pytest.approx(40.0)


def test_a_market_with_no_measurable_width_is_not_refused(monkeypatch):
    # Missing width is not evidence of a wide market.
    res = _gate(monkeypatch, None)
    assert res["decision"] == "tipped"
    assert res.get("overround_pct") is None


def test_the_defaults_do_not_gate_a_normal_in_play_market():
    # The live feed has been running around 9% on two-way markets. Shipping a
    # cap that silently kills every live tip would be worse than no cap.
    assert main.MAX_OVERROUND_BPS > 900
    assert main.MAX_OVERROUND_BPS_3WAY > main.MAX_OVERROUND_BPS


# ───────── showing it ─────────

def test_the_tip_message_carries_the_overround():
    msg = main._format_tip_message("A", "B", "L", 37, "1-0", "BTTS: No", 68.3,
                                   raw={"sot_h": 3}, odds=1.53, book="Feed",
                                   ev_pct=4.6, fair_prob=0.608, stake=1.0,
                                   overround_pct=9.3)
    assert "Overround" in msg and "9.3%" in msg


def test_a_tip_without_a_measurable_overround_says_nothing():
    msg = main._format_tip_message("A", "B", "L", 37, "1-0", "BTTS: No", 68.3,
                                   raw={"sot_h": 3}, odds=1.53, book="Feed",
                                   ev_pct=4.6, fair_prob=0.608, stake=1.0)
    assert "Overround" not in msg
    assert "Fair" in msg, "the rest of the money line must be unaffected"
