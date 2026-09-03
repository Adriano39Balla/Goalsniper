"""
Parsing API-Football responses without silently inventing data.

Three defects, each of which produces a plausible-looking number rather than
an error, which is why none of them showed up as a failure:

  1. Statistics and events were matched to a side by TEAM NAME. A name is free
     text and differs between endpoints often enough to matter; a mismatch
     yields an empty statistics dict, which then reads as a genuine 0 for
     every shot, corner and card that team has.
  2. Missing possession arrived as 0/0. The two sides always sum to 100, so
     that is not a plausible observation - and it zeroes game_control_h/a,
     which are possession-weighted.
  3. Results were read from the top-level `goals`, which keeps counting
     through extra time, for every fixture in FINAL_STATUSES - and that set
     includes AET and PEN.
"""
import pytest

import main
from main import extract_raw_inplay, fulltime_goals, fulltime_market_open


def _fixture(stats_team_home=None, events=None, status="2H"):
    return {
        "fixture": {"id": 1, "status": {"elapsed": 60, "short": status}},
        "teams": {"home": {"id": 10, "name": "Home FC"},
                  "away": {"id": 20, "name": "Away FC"}},
        "goals": {"home": 1, "away": 0},
        "events": events or [],
        "statistics": [
            {"team": stats_team_home or {"id": 10, "name": "Home FC"},
             "statistics": [{"type": "Shots on Goal", "value": 5},
                            {"type": "Ball Possession", "value": "58%"},
                            {"type": "Expected Goals", "value": "1.35"}]},
            {"team": {"id": 20, "name": "Away FC"},
             "statistics": [{"type": "Shots on Goal", "value": 2},
                            {"type": "Ball Possession", "value": "42%"}]},
        ],
    }


# ───────── matching a side ─────────

def test_a_renamed_team_still_gets_its_statistics():
    # The id is the same; only the free-text name differs between endpoints.
    raw = extract_raw_inplay(_fixture(stats_team_home={"id": 10, "name": "Home Football Club"}))
    assert raw["sot_h"] == 5, "matching on name alone would report zero shots"
    assert raw["xg_h"] == pytest.approx(1.35)


def test_a_feed_without_ids_still_matches_on_name():
    raw = extract_raw_inplay(_fixture(stats_team_home={"name": "Home FC"}))
    assert raw["sot_h"] == 5


def test_a_third_team_is_ignored_rather_than_guessed():
    raw = extract_raw_inplay(_fixture(stats_team_home={"id": 999, "name": "Someone Else"}))
    assert raw["sot_h"] == 0, "an unmatched block must not be attributed to a side"
    assert raw["sot_a"] == 2, "and must not disturb the side that did match"


def test_red_cards_are_attributed_by_id():
    ev = [{"type": "Card", "detail": "Red Card", "team": {"id": 20, "name": "Away Football Club"}},
          {"type": "Card", "detail": "Second Yellow card", "team": {"id": 10, "name": "Home FC"}},
          {"type": "Card", "detail": "Yellow Card", "team": {"id": 10, "name": "Home FC"}}]
    raw = extract_raw_inplay(_fixture(events=ev))
    assert raw["red_a"] == 1
    assert raw["red_h"] == 1, "a second yellow is a red"


# ───────── possession ─────────

def test_a_missing_possession_feed_reads_as_unknown_not_as_zero():
    fx = _fixture()
    for blk in fx["statistics"]:
        blk["statistics"] = [i for i in blk["statistics"] if i["type"] != "Ball Possession"]
    raw = extract_raw_inplay(fx)
    assert raw["pos_h"] == 50.0 and raw["pos_a"] == 50.0


def test_one_side_quoted_implies_the_other():
    fx = _fixture()
    fx["statistics"][1]["statistics"] = [{"type": "Shots on Goal", "value": 2}]
    raw = extract_raw_inplay(fx)
    assert raw["pos_h"] == 58.0
    assert raw["pos_a"] == pytest.approx(42.0)


def test_real_possession_is_untouched():
    raw = extract_raw_inplay(_fixture())
    assert (raw["pos_h"], raw["pos_a"]) == (58.0, 42.0)


def test_substituted_possession_cannot_satisfy_the_coverage_check():
    # Possession is now always populated, so counting it would let a fixture
    # whose statistics never arrived pass as covered.
    blank = {"minute": 60, "xg_h": 0, "xg_a": 0, "sot_h": 0, "sot_a": 0,
             "cor_h": 0, "cor_a": 0, "pos_h": 50.0, "pos_a": 50.0,
             "total_shots_h": 0, "total_shots_a": 0}
    assert main.stats_coverage_ok(blank, 60) is False


# ───────── the 90-minute score ─────────

def test_a_tie_decided_in_extra_time_is_graded_on_90_minutes():
    # 1-1 after 90, 3-2 after extra time. Every market here settles at 90:
    # Over 2.5 lost, and the 120-minute score says it won.
    fx = {"goals": {"home": 3, "away": 2},
          "score": {"fulltime": {"home": 1, "away": 1},
                    "extratime": {"home": 2, "away": 1}}}
    assert fulltime_goals(fx) == (1, 1)


def test_a_goalless_draw_decided_on_penalties_is_still_goalless():
    fx = {"goals": {"home": 0, "away": 0},
          "score": {"fulltime": {"home": 0, "away": 0},
                    "penalty": {"home": 4, "away": 3}}}
    assert fulltime_goals(fx) == (0, 0)


def test_a_normal_finish_is_unchanged():
    fx = {"goals": {"home": 2, "away": 1},
          "score": {"fulltime": {"home": 2, "away": 1}, "extratime": {}}}
    assert fulltime_goals(fx) == (2, 1)


def test_a_live_fixture_falls_back_to_the_running_score():
    # score.fulltime is null until the match finishes.
    fx = {"goals": {"home": 1, "away": 0}, "score": {"fulltime": {"home": None, "away": None}}}
    assert fulltime_goals(fx) == (1, 0)


def test_a_fixture_with_no_score_block_at_all_does_not_raise():
    assert fulltime_goals({"goals": {"home": 2, "away": 0}}) == (2, 0)
    assert fulltime_goals({}) == (0, 0)


# ───────── which statuses are bettable ─────────

def test_the_full_time_market_is_open_in_normal_play():
    for st in ("1H", "HT", "2H"):
        assert fulltime_market_open({"fixture": {"status": {"short": st}}}) is True


def test_it_is_closed_once_extra_time_is_reached():
    # The bet settled at 90. Anything quoted now is a different bet, and the
    # scoreline now includes goals that do not count toward it.
    for st in ("ET", "BT", "P", "AET", "PEN", "FT"):
        assert fulltime_market_open({"fixture": {"status": {"short": st}}}) is False


def test_an_absent_status_is_treated_as_closed():
    # Conservative: we cannot confirm the market is open, so we do not bet it.
    assert fulltime_market_open({}) is False
