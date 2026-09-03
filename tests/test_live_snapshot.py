"""
_build_live_match_entry() and the _set/_get_live_snapshot() pair back
GET /dashboard/live - the full per-market probability breakdown for every
live match, independent of whether any candidate actually got tipped.

Any candidate clearing its own threshold is now run through the real
_price_gate() so the dashboard can show *why* a high-confidence candidate
wasn't tipped, not just its raw probability - so these tests monkeypatch
main.fetch_odds even though they aren't testing _price_gate itself.
production_scan() needs a live DB/API stack to exercise directly, so these
tests cover the pure pieces it's built from.
"""
import pytest

import main


def _odds_map(mkey, sel, odds, book="Bet365", fair=None, n_books=5):
    entry = {"best": {sel: {"odds": odds, "book": book}}, "n_books": n_books}
    if fair is not None:
        entry["fair"] = {sel: fair}
    return {mkey: entry}


def test_build_live_match_entry_maps_every_candidate_to_a_market_row(monkeypatch):
    # BTTS clears every _price_gate() gate; Over/Under 2.5 has no odds quoted
    # at all, so it's the "high confidence but nothing sent" case in miniature.
    odds_map = _odds_map("BTTS", "Yes", odds=2.0, fair=0.55)
    monkeypatch.setattr(main, "fetch_odds", lambda fid, live: odds_map)

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

    btts, ou = entry["markets"]
    assert btts["market"] == "BTTS" and btts["suggestion"] == "BTTS: Yes"
    assert btts["prob_pct"] == 62.0 and btts["threshold_pct"] == 55.0
    assert btts["decision"] == "tipped"
    assert btts["odds"] == pytest.approx(2.0)
    assert btts["ev_pct"] == pytest.approx(24.0)

    assert ou["market"] == "Over/Under 2.5" and ou["suggestion"] == "Over 2.5 Goals"
    assert ou["prob_pct"] == 58.0 and ou["threshold_pct"] == 55.0
    assert ou["decision"] == "no_odds"
    assert ou["odds"] is None

    # Only the BTTS candidate actually clears the price gate.
    assert entry["hits"] == 1


def test_build_live_match_entry_handles_no_candidates():
    entry = main._build_live_match_entry(
        fid=1, league="L", league_id=1, home="A", away="B",
        score="0-0", minute=10, candidates=[])
    assert entry["markets"] == []
    assert entry["hits"] == 0


def test_kickoff_ts_and_stats_default_when_not_supplied():
    # Older call sites (or tests) that don't pass kickoff_ts/raw shouldn't
    # break - kickoff_ts falls back to 0, stats to None rather than a
    # half-filled dict.
    entry = main._build_live_match_entry(
        fid=1, league="L", league_id=1, home="A", away="B",
        score="0-0", minute=10, candidates=[])
    assert entry["kickoff_ts"] == 0
    assert entry["stats"] is None


def test_kickoff_ts_and_stats_pass_through_when_supplied():
    raw = {"sot_h": 4.0, "sot_a": 2.0, "cor_h": 5.0, "cor_a": 3.0,
           "pos_h": 55.0, "pos_a": 45.0, "yellow_h": 1.0, "yellow_a": 2.0}
    entry = main._build_live_match_entry(
        fid=1, league="L", league_id=1, home="A", away="B",
        score="0-0", minute=10, candidates=[], kickoff_ts=1_700_000_000, raw=raw)
    assert entry["kickoff_ts"] == 1_700_000_000
    # Every supplied value passes through. Asserted as a subset rather than by
    # equality so adding a field to the panel is not a test failure — the
    # contract is "what was given is carried", not "exactly these keys".
    assert raw.items() <= entry["stats"].items()


def test_the_snapshot_carries_what_the_xg_diagnostic_needs():
    # /admin/diagnostics/xg-feed answers "is the xG channel alive" from this
    # snapshot rather than by re-fetching, so these keys have to be here.
    entry = main._build_live_match_entry(
        fid=1, league="L", league_id=1, home="A", away="B", score="0-0", minute=30,
        candidates=[], raw={"xg_h": 0.4, "xg_a": 0.1, "sot_h": 3.0, "sot_a": 1.0,
                            "total_shots_h": 7.0, "total_shots_a": 3.0})
    for k in ("xg_h", "xg_a", "sot_h", "sot_a", "total_shots_h", "total_shots_a"):
        assert k in entry["stats"], k


def test_candidates_below_threshold_never_call_price_gate(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("_price_gate must not run for a below-threshold candidate")

    monkeypatch.setattr(main, "_price_gate", _boom)
    candidates = [
        ("BTTS", "BTTS: Yes", 0.40, 55.0),   # below
        ("1X2", "Home Win", 0.40, 55.0),      # below
    ]
    entry = main._build_live_match_entry(
        fid=1, league="L", league_id=1, home="A", away="B",
        score="0-0", minute=10, candidates=candidates)
    assert [m["decision"] for m in entry["markets"]] == ["below_threshold", "below_threshold"]
    assert entry["hits"] == 0


def test_hits_counts_only_candidates_that_pass_the_price_gate(monkeypatch):
    # One clears the price gate, one clears threshold but has no odds, one
    # never clears threshold at all - only the first should count as a hit.
    odds_map = _odds_map("BTTS", "Yes", odds=2.0, fair=0.55)
    monkeypatch.setattr(main, "fetch_odds", lambda fid, live: odds_map)
    candidates = [
        ("BTTS", "BTTS: Yes", 0.62, 55.0),                    # tipped
        ("Over/Under 2.5", "Over 2.5 Goals", 0.55, 55.0),     # clears threshold, no odds
        ("1X2", "Home Win", 0.40, 55.0),                      # below threshold
    ]
    entry = main._build_live_match_entry(
        fid=1, league="L", league_id=1, home="A", away="B",
        score="0-0", minute=10, candidates=candidates)
    assert entry["hits"] == 1


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


def test_snapshot_carries_scan_counts_so_an_empty_feed_can_explain_itself():
    # "no matches are being played" and "plenty are, none with usable stats"
    # are the same empty list - the counts are what tells them apart.
    main._set_live_snapshot([], live_seen=12, no_coverage=12)
    snap = main._get_live_snapshot()
    assert snap["matches"] == []
    assert snap["live_seen"] == 12
    assert snap["no_coverage"] == 12


def test_scan_counts_default_to_none_when_the_caller_does_not_know():
    main._set_live_snapshot([])
    snap = main._get_live_snapshot()
    assert snap["live_seen"] is None
    assert snap["no_coverage"] is None


def test_empty_snapshot_after_no_live_matches():
    main._set_live_snapshot([main._build_live_match_entry(1, "L", 1, "A", "B", "0-0", 5, [])])
    assert main._get_live_snapshot()["matches"]
    main._set_live_snapshot([])
    assert main._get_live_snapshot()["matches"] == []
