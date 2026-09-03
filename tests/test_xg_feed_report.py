"""
Answering "is the xG channel actually arriving" without another round trip.

A shot on target always carries positive expected goals, so `shots recorded
AND total xG exactly 0.00` proves the channel is absent rather than proving
the game is quiet. That one test separates "nothing is happening" from "we are
blind" — and the two look identical in the feature vector, because a missing
Expected Goals arrives as 0.0.

Reported by league because coverage is a per-competition property of the data
plan: a few dead leagues is a filtering decision, every league dead is an
account-level fault worth more than any model change.
"""
from main import xg_feed_report


def _m(league="England - PL", xg=(0.0, 0.0), sot=(0, 0), shots=(0, 0), fid=1):
    return {"fixture_id": fid, "league": league, "home": "A", "away": "B", "minute": 60,
            "stats": {"xg_h": xg[0], "xg_a": xg[1], "sot_h": sot[0], "sot_a": sot[1],
                      "total_shots_h": shots[0], "total_shots_a": shots[1]}}


def test_a_live_channel_is_recognised():
    rep = xg_feed_report([_m(xg=(0.8, 0.3), sot=(3, 1))])
    assert rep["xg_live"] == 1 and rep["xg_dead_but_shots_recorded"] == 0
    assert "arriving on every fixture" in rep["verdict"]


def test_shots_with_zero_xg_is_counted_as_dead():
    rep = xg_feed_report([_m(sot=(3, 0))])
    assert rep["xg_dead_but_shots_recorded"] == 1
    assert rep["xg_live"] == 0


def test_a_shotless_fixture_is_undecidable_not_dead():
    # No shots and no xG is consistent, and it is real football. Counting it as
    # a fault would invent a problem to explain an ordinary dull half.
    rep = xg_feed_report([_m()])
    assert rep["no_shots_yet_undecidable"] == 1
    assert rep["xg_dead_but_shots_recorded"] == 0
    assert rep["xg_coverage_pct_of_decidable"] is None
    assert "cannot be judged yet" in rep["verdict"]


def test_coverage_is_measured_only_over_decidable_fixtures():
    # Shotless fixtures must not dilute the percentage in either direction.
    rep = xg_feed_report([_m(xg=(1.0, 0.2), sot=(4, 1)), _m(sot=(2, 1)), _m()])
    assert rep["fixtures_in_snapshot"] == 3
    assert rep["xg_coverage_pct_of_decidable"] == 50.0


def test_a_total_outage_is_called_what_it_is():
    rep = xg_feed_report([_m(sot=(3, 1), fid=i) for i in range(6)])
    assert rep["xg_live"] == 0
    assert "account-level fault" in rep["verdict"]
    assert "No live tip can fire" in rep["verdict"]


def test_a_partial_gap_points_at_the_leagues_responsible():
    rep = xg_feed_report([
        _m(league="England - PL", xg=(1.1, 0.4), sot=(5, 2)),
        _m(league="Ecuador - Liga Pro", sot=(3, 0)),
        _m(league="Ecuador - Liga Pro", sot=(2, 2)),
    ])
    assert rep["xg_dead_but_shots_recorded"] == 2
    # The worst league leads, so the fix is obvious from the first line.
    assert list(rep["by_league"])[0] == "Ecuador - Liga Pro"
    assert rep["by_league"]["Ecuador - Liga Pro"]["xg_dead"] == 2
    assert rep["by_league"]["England - PL"]["xg_live"] == 1
    assert "LEAGUE_DENY_IDS" in rep["verdict"]


def test_dead_fixtures_are_named_so_they_can_be_checked_by_hand():
    rep = xg_feed_report([_m(sot=(3, 0), fid=777)])
    ex = rep["dead_examples"][0]
    assert ex["fixture_id"] == 777 and ex["xg"] == 0 and ex["shots"] == 3


def test_the_example_list_is_bounded():
    rep = xg_feed_report([_m(sot=(1, 0), fid=i) for i in range(50)])
    assert len(rep["dead_examples"]) == 10
    assert rep["xg_dead_but_shots_recorded"] == 50, "the count is not truncated"


def test_an_empty_snapshot_says_so_rather_than_reporting_a_fault():
    rep = xg_feed_report([])
    assert rep["fixtures_in_snapshot"] == 0
    assert "nothing to judge" in rep["verdict"]


def test_a_fixture_with_no_stats_block_does_not_break_it():
    rep = xg_feed_report([{"fixture_id": 1, "league": "L", "stats": None}])
    assert rep["fixtures_in_snapshot"] == 1
    assert rep["no_shots_yet_undecidable"] == 1
