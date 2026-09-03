"""
API-Football names the same market differently in its prematch and in-play
feeds. The prematch feed says "Match Winner"; the IN-PLAY feed says "Fulltime
Result", which matched none of the 1X2 patterns and was dropped as unmapped.

Live 1X2 prices therefore never arrived, and with them market_fair_home,
market_fair_draw and market_fair_away — three of the five features every
anchored head is pinned to. Production logs show "Fulltime Result" offered on
10 fixtures that then reported "no usable markets", alongside
"[ANCHOR] in-play heads NOT anchored: only 0 rows / 0 fixtures carry a real
market price".
"""
import pytest

import main


@pytest.mark.parametrize("name", [
    "Fulltime Result", "FULLTIME RESULT", "Full Time Result", "FT Result",
    "Match Result", "Match Winner", "1X2", "3-Way Result",
])
def test_the_full_time_result_market_is_recognised_as_1x2(name):
    assert main._market_name_normalize(name) == "1X2"


@pytest.mark.parametrize("name,key", [
    ("Match Goals", "OU"),
    ("Over/Under Line", "OU"),
    ("Goals Over/Under", "OU"),
    ("Both Teams to Score", "BTTS"),
    ("Double Chance", "DC"),
    ("Draw No Bet", "DNB"),
])
def test_the_other_full_match_markets_still_map(name, key):
    assert main._market_name_normalize(name) == key


@pytest.mark.parametrize("name", [
    # A combination bet pays only when BOTH legs land, so it is neither of
    # them. The BTTS matcher keys on "both teams" appearing anywhere and would
    # otherwise claim this one.
    "Result / Both Teams To Score",
    "Result/Both Teams To Score",
    # Wrong scope: a different question that reuses the same words.
    "Total Corners", "Total Cards", "Asian Handicap", "Goals Odd/Even",
    "Home Team Goals", "Away Team Goals", "Final Score",
    "Both Teams To Score (2nd Half)", "To Win 2nd Half",
    "Goals Over/Under First Half",
    # Contains "1x2" but is a timed market, not the 90-minute one.
    "1x2 - 60 minutes",
])
def test_markets_that_are_a_different_bet_stay_unmapped(name):
    assert main._market_name_normalize(name) not in ("BTTS", "DC", "DNB", "1X2", "OU")


def test_a_live_fulltime_result_market_now_prices_1x2():
    """End to end through the parser, with the in-play feed's own value labels."""
    parsed = main._parse_book_market({
        "name": "Fulltime Result",
        "values": [{"value": "Home", "odd": "2.10"},
                   {"value": "Draw", "odd": "3.40"},
                   {"value": "Away", "odd": "3.60"}],
    })
    assert parsed is not None, "the live 1X2 market must produce prices"
    mkey, sels = parsed
    assert mkey == "1X2"
    assert sels == {"Home": 2.10, "Draw": 3.40, "Away": 3.60}


def test_a_combo_market_yields_no_btts_price():
    # Its selections are "Home/Yes" style, so even reaching the BTTS branch it
    # produced nothing — but it must not reach that branch at all.
    parsed = main._parse_book_market({
        "name": "Result / Both Teams To Score",
        "values": [{"value": "Home/Yes", "odd": "4.50"},
                   {"value": "Draw/No", "odd": "6.00"}],
    })
    assert parsed is None
