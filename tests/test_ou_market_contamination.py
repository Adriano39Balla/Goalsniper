"""
Only the MATCH goals over/under may be priced as OU.

API-Football's odds catalogue quotes a plain "Over 2.5" label under many
bets that are not the match total: team totals, first/second half goals,
corners, cards, exact goals. Folding those into OU_2.5 was not a cosmetic
mislabel - fetch_odds keeps the BEST price per selection, and "this team
scores 3+" prices around 4.0-9.0 against ~1.9 for the match total, so the
wrong price won every time. It then drove the EV gate and the P&L, showing
a 116.8% ROI on PRE Over/Under 2.5 at a 52.9% win rate.

These tests use the real bet names as API-Football spells them.
"""
import pytest

import main


# ───────── names that ARE the match total ─────────

@pytest.mark.parametrize("name", [
    "Goals Over/Under",
    "Over/Under",
    "Total Goals",
    "Goals Over/Under ",
])
def test_match_total_names_map_to_ou(name):
    assert main._market_name_normalize(name) == "OU"


# ───────── names that are NOT, and must not be priced as OU ─────────

@pytest.mark.parametrize("name", [
    "Total - Home",                    # team total: "Over 2.5" ~ 4.0-9.0
    "Total - Away",
    "Home Team Total Goals",
    "Away Team Total Goals",
    "Goals Over/Under First Half",     # half total: "Over 2.5" ~ 6.0+
    "Goals Over/Under - Second Half",
    "Corners Over Under",
    "Cards Over/Under",
    "Total Corners",
    "Exact Goals Number",
    "Total Goals Odd/Even",
    "Asian Handicap",
    "Goals Asian Over/Under",
])
def test_other_totals_are_not_folded_into_ou(name):
    assert main._market_name_normalize(name) != "OU"


def test_unmapped_totals_are_skipped_entirely_by_the_parser():
    # An unmapped name must produce no prices at all, not a partial market.
    team_total = {"name": "Total - Home",
                  "values": [{"value": "Over 2.5", "odd": "4.50"},
                             {"value": "Under 2.5", "odd": "1.18"}]}
    assert main._parse_book_market(team_total) is None


def test_match_total_still_parses_normally():
    match_total = {"name": "Goals Over/Under",
                   "values": [{"value": "Over 2.5", "odd": "1.90"},
                              {"value": "Under 2.5", "odd": "1.95"}]}
    key, payload = main._parse_book_market(match_total)
    assert key == "OU_MULTI"
    assert payload["OU_2.5"] == {"Over": 1.90, "Under": 1.95}


# ───────── the end-to-end effect on the stored price ─────────

def _book(*markets):
    return {"response": [{"bookmakers": [{"name": "Bet365", "bets": list(markets)}]}]}


def test_team_total_no_longer_hijacks_the_match_over_price(monkeypatch):
    # The exact shape that inflated the ROI: both markets quote "Over 2.5",
    # and the team total is the bigger number.
    payload = _book(
        {"name": "Goals Over/Under",
         "values": [{"value": "Over 2.5", "odd": "1.90"}, {"value": "Under 2.5", "odd": "1.95"}]},
        {"name": "Total - Home",
         "values": [{"value": "Over 2.5", "odd": "4.50"}, {"value": "Under 2.5", "odd": "1.18"}]},
    )
    monkeypatch.setattr(main, "_api_get", lambda url, params, timeout=15: payload)
    main.ODDS_CACHE.invalidate()

    out = main.fetch_odds(4242, live=False)
    assert out["OU_2.5"]["best"]["Over"]["odds"] == pytest.approx(1.90)


def test_first_half_goals_do_not_hijack_it_either(monkeypatch):
    payload = _book(
        {"name": "Goals Over/Under",
         "values": [{"value": "Over 2.5", "odd": "2.00"}, {"value": "Under 2.5", "odd": "1.85"}]},
        {"name": "Goals Over/Under First Half",
         "values": [{"value": "Over 2.5", "odd": "6.50"}, {"value": "Under 2.5", "odd": "1.08"}]},
    )
    monkeypatch.setattr(main, "_api_get", lambda url, params, timeout=15: payload)
    main.ODDS_CACHE.invalidate()

    out = main.fetch_odds(4243, live=False)
    assert out["OU_2.5"]["best"]["Over"]["odds"] == pytest.approx(2.00)


def test_contaminated_prices_no_longer_reach_the_fair_price(monkeypatch):
    # The de-vigged consensus feeds market_fair_over25, which is now a model
    # input - so a hijacked price corrupted a training feature too, not just
    # the EV gate.
    payload = _book(
        {"name": "Goals Over/Under",
         "values": [{"value": "Over 2.5", "odd": "1.90"}, {"value": "Under 2.5", "odd": "1.95"}]},
        {"name": "Total - Away",
         "values": [{"value": "Over 2.5", "odd": "5.00"}, {"value": "Under 2.5", "odd": "1.15"}]},
    )
    monkeypatch.setattr(main, "_api_get", lambda url, params, timeout=15: payload)
    main.ODDS_CACHE.invalidate()

    fair = main.fetch_odds(4244, live=False)["OU_2.5"]["fair"]
    # Two roughly even prices de-vig to roughly a coin flip; the team total
    # would have dragged this far away from it.
    assert fair["Over"] == pytest.approx(0.5, abs=0.05)


def test_only_the_match_market_counts_toward_book_depth(monkeypatch):
    # n_books gates MIN_BOOKS_FOR_FAIR. Counting a team total as an extra
    # "book" would fake up depth that isn't there.
    payload = _book(
        {"name": "Goals Over/Under",
         "values": [{"value": "Over 2.5", "odd": "1.90"}, {"value": "Under 2.5", "odd": "1.95"}]},
        {"name": "Total - Home",
         "values": [{"value": "Over 2.5", "odd": "4.50"}, {"value": "Under 2.5", "odd": "1.18"}]},
        {"name": "Cards Over/Under",
         "values": [{"value": "Over 2.5", "odd": "1.60"}, {"value": "Under 2.5", "odd": "2.20"}]},
    )
    monkeypatch.setattr(main, "_api_get", lambda url, params, timeout=15: payload)
    main.ODDS_CACHE.invalidate()

    assert main.fetch_odds(4245, live=False)["OU_2.5"]["n_books"] == 1


# ───────── the same hole in every other family ─────────
#
# Over/Under was where it showed up in the P&L, but the substring matching
# had nothing to do with goals: a half-time, extra-time or shootout variant
# mapped onto the full-match market in every family. Half-time prices are
# systematically longer than full-time ones, and fetch_odds keeps the best
# price, so they hijacked the recorded price the same way team totals did.

@pytest.mark.parametrize("name,family", [
    ("Both Teams To Score - First Half", "BTTS"),
    ("Both Teams To Score in Both Halves", "BTTS"),
    ("First Half Winner", "1X2"),
    ("Second Half Winner", "1X2"),
    ("Winner - Extra Time", "1X2"),
    ("Penalty Shootout Winner", "1X2"),
    ("Double Chance - First Half", "DC"),
    ("Draw No Bet (1st Half)", "DNB"),
])
def test_part_match_variants_never_map_to_the_full_match_family(name, family):
    assert main._market_name_normalize(name) != family


@pytest.mark.parametrize("name,family", [
    ("Match Winner", "1X2"),
    ("1X2", "1X2"),
    ("Both Teams To Score", "BTTS"),
    ("Double Chance", "DC"),
    ("Draw No Bet", "DNB"),
    ("Goals Over/Under", "OU"),
])
def test_full_match_markets_still_map(name, family):
    assert main._market_name_normalize(name) == family


def test_half_time_price_no_longer_hijacks_the_full_time_home_price(monkeypatch):
    payload = _book(
        {"name": "Match Winner",
         "values": [{"value": "Home", "odd": "1.80"}, {"value": "Draw", "odd": "3.60"},
                    {"value": "Away", "odd": "4.20"}]},
        {"name": "First Half Winner",
         "values": [{"value": "Home", "odd": "2.60"}, {"value": "Draw", "odd": "2.10"},
                    {"value": "Away", "odd": "6.00"}]},
    )
    monkeypatch.setattr(main, "_api_get", lambda url, params, timeout=15: payload)
    main.ODDS_CACHE.invalidate()

    best = main.fetch_odds(4246, live=False)["1X2"]["best"]
    assert best["Home"]["odds"] == pytest.approx(1.80)
    assert best["Draw"]["odds"] == pytest.approx(3.60)


def test_half_time_btts_no_longer_hijacks_full_time_btts(monkeypatch):
    payload = _book(
        {"name": "Both Teams To Score",
         "values": [{"value": "Yes", "odd": "1.90"}, {"value": "No", "odd": "1.90"}]},
        {"name": "Both Teams To Score - First Half",
         "values": [{"value": "Yes", "odd": "3.60"}, {"value": "No", "odd": "1.28"}]},
    )
    monkeypatch.setattr(main, "_api_get", lambda url, params, timeout=15: payload)
    main.ODDS_CACHE.invalidate()

    assert main.fetch_odds(4247, live=False)["BTTS"]["best"]["Yes"]["odds"] == pytest.approx(1.90)
