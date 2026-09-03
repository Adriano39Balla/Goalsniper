"""
The two API-Football odds endpoints put the goals line in different places.

PREMATCH embeds it in the selection label:
    {"value": "Over 2.5", "odd": "1.95"}

IN-PLAY leaves the label bare and gives the line its own field — which is why
that market is named "Over/Under Line" rather than "Goals Over/Under":
    {"value": "Over", "odd": "1.95", "handicap": "2.5"}

Only the first was parsed. float("over") raises, so every live Over/Under
selection was skipped and no live OU price was ever recorded — taking
market_fair_over25 with it, one of the five features the anchored heads are
pinned to. The training run reported "only 0 rows / 0 fixtures carry a real
market price" across the entire dataset.
"""
import pytest

import main


def test_the_prematch_shape_still_parses():
    parsed = main._parse_book_market({
        "name": "Goals Over/Under",
        "values": [{"value": "Over 2.5", "odd": "1.95"},
                   {"value": "Under 2.5", "odd": "1.90"}],
    })
    assert parsed == ("OU_MULTI", {"OU_2.5": {"Over": 1.95, "Under": 1.90}})


def test_the_in_play_shape_parses_the_line_from_its_own_field():
    parsed = main._parse_book_market({
        "name": "Over/Under Line",
        "values": [{"value": "Over", "odd": "1.95", "handicap": "2.5"},
                   {"value": "Under", "odd": "1.90", "handicap": "2.5"}],
    })
    assert parsed == ("OU_MULTI", {"OU_2.5": {"Over": 1.95, "Under": 1.90}})


def test_several_lines_in_one_market_stay_separate():
    parsed = main._parse_book_market({
        "name": "Over/Under Line",
        "values": [{"value": "Over", "odd": "1.50", "handicap": "1.5"},
                   {"value": "Under", "odd": "2.50", "handicap": "1.5"},
                   {"value": "Over", "odd": "2.05", "handicap": "2.5"},
                   {"value": "Under", "odd": "1.80", "handicap": "2.5"}],
    })
    _, by_line = parsed
    assert by_line["OU_1.5"] == {"Over": 1.50, "Under": 2.50}
    assert by_line["OU_2.5"] == {"Over": 2.05, "Under": 1.80}


@pytest.mark.parametrize("field", ["handicap", "line", "hcp", "total", "points"])
def test_the_line_is_read_from_any_field_the_feed_uses(field):
    parsed = main._parse_book_market({
        "name": "Over/Under Line",
        "values": [{"value": "Over", "odd": "2.00", field: "3.5"},
                   {"value": "Under", "odd": "1.75", field: "3.5"}],
    })
    assert parsed == ("OU_MULTI", {"OU_3.5": {"Over": 2.00, "Under": 1.75}})


def test_a_comma_decimal_line_is_read():
    parsed = main._parse_book_market({
        "name": "Over/Under Line",
        "values": [{"value": "Over", "odd": "1.95", "handicap": "2,5"}],
    })
    assert parsed == ("OU_MULTI", {"OU_2.5": {"Over": 1.95}})


def test_a_selection_with_no_line_anywhere_is_still_skipped():
    # Guessing a line would price a bet against a market it does not belong to.
    assert main._parse_book_market({
        "name": "Over/Under Line",
        "values": [{"value": "Over", "odd": "1.95"}],
    }) is None


def test_a_suspended_live_price_is_still_refused():
    assert main._parse_book_market({
        "name": "Over/Under Line",
        "values": [{"value": "Over", "odd": "1.95", "handicap": "2.5",
                    "suspended": True}],
    }) is None


def test_the_live_market_now_yields_a_devigged_over25(monkeypatch):
    """End to end: a live OU market must reach market_fair_over25."""
    main.ODDS_CACHE.invalidate()
    monkeypatch.setattr(main, "_api_get", lambda url, params, timeout=15: {"response": [{
        "fixture": {"id": 7},
        "odds": [{"name": "Over/Under Line", "values": [
            {"value": "Over", "odd": "2.00", "handicap": "2.5"},
            {"value": "Under", "odd": "2.00", "handicap": "2.5"}]}],
    }]})
    out = main.fetch_odds(7, live=True)
    assert "OU_2.5" in out, "the live Over/Under market must produce a price"
    assert out["OU_2.5"]["fair"]["Over"] == pytest.approx(0.5, abs=1e-6)

    priors = main._market_fair_priors(7, live=True)
    assert priors.get("market_fair_over25") == pytest.approx(0.5, abs=1e-6)
