"""
_build_live_match_entry() and the _set/_get_live_snapshot() pair back
GET /dashboard/live - the full per-market probability breakdown for every
live match, independent of whether any candidate actually got tipped.
production_scan() itself needs a live DB/API stack to exercise directly, so
these tests cover the pure pieces it's built from.
"""
import main


def test_build_live_match_entry_maps_every_candidate_to_a_market_row():
    candidates = [
        ("BTTS", "BTTS: Yes", 0.62, 55.0),
        ("Over/Under 2.5", "Over 2.5 Goals", 0.58, 55.0),
    ]
    entry = main._build_live_match_entry(
        fid=123, league="Bundesliga", league_id=78, home="Bayern", away="Dortmund",
        score="1-0", minute=63, candidates=candidates)

    assert entry["fixture_id"] == 123
    assert entry["home"] == "Bayern"
    assert entry["away"] == "Dortmund"
    assert entry["minute"] == 63
    assert entry["markets"] == [
        {"market": "BTTS", "suggestion": "BTTS: Yes", "prob_pct": 62.0, "threshold_pct": 55.0},
        {"market": "Over/Under 2.5", "suggestion": "Over 2.5 Goals", "prob_pct": 58.0, "threshold_pct": 55.0},
    ]
    assert entry["hits"] == 2


def test_build_live_match_entry_handles_no_candidates():
    entry = main._build_live_match_entry(
        fid=1, league="L", league_id=1, home="A", away="B",
        score="0-0", minute=10, candidates=[])
    assert entry["markets"] == []
    assert entry["hits"] == 0


def test_build_live_match_entry_counts_only_candidates_clearing_threshold():
    candidates = [
        ("BTTS", "BTTS: Yes", 0.62, 55.0),   # clears
        ("1X2", "Home Win", 0.40, 55.0),      # below
        ("Over/Under 2.5", "Over 2.5 Goals", 0.55, 55.0),  # exactly at threshold, clears
    ]
    entry = main._build_live_match_entry(
        fid=1, league="L", league_id=1, home="A", away="B",
        score="0-0", minute=10, candidates=candidates)
    assert entry["hits"] == 2


def test_live_snapshot_round_trips_and_stamps_updated_ts(monkeypatch):
    monkeypatch.setattr(main.time, "time", lambda: 1_700_000_000.0)
    matches = [main._build_live_match_entry(1, "L", 1, "A", "B", "1-1", 40, [])]

    main._set_live_snapshot(matches)
    snap = main._get_live_snapshot()

    assert snap["updated_ts"] == 1_700_000_000
    assert snap["matches"] == matches


def test_get_live_snapshot_returns_a_copy_not_the_live_list():
    main._set_live_snapshot([main._build_live_match_entry(1, "L", 1, "A", "B", "0-0", 5, [])])
    snap = main._get_live_snapshot()
    snap["matches"].append("mutated")
    assert main._get_live_snapshot()["matches"] != snap["matches"]


def test_empty_snapshot_after_no_live_matches():
    main._set_live_snapshot([main._build_live_match_entry(1, "L", 1, "A", "B", "0-0", 5, [])])
    assert main._get_live_snapshot()["matches"]

    main._set_live_snapshot([])
    assert main._get_live_snapshot()["matches"] == []
