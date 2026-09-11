"""
A failed API call must never be persisted as training data.

fetch_odds(), fetch_match_stats() and _api_last_fixtures() each enforce this
at the CACHE layer, and each says so in its own comment: a call that failed
is not "nothing was there". It was being violated one layer further down, at
the point of persistence:

  IN-PLAY   fetch_match_stats() returns [] on a failed/refused call ->
            extract_raw_inplay() yields zeros for xg/shots/corners/cards ->
            save_snapshot_from_match() wrote that into tip_snapshots ->
            load_inplay_data() has no coverage filter, so the fit is taught
            "no shots and no xG at minute 60" as a real match state.

  PREMATCH  _api_last_fixtures()/_api_h2h() return [] on a failed call ->
            assemble_prematch_features() yields a complete all-zero vector ->
            save_prematch_snapshot() wrote it into prematch_snapshots ->
            load_prematch_data()'s only filter is `if not feat`, which never
            fires because the vector is populated-with-zeros, not empty. And
            because that table upserts on match_id, a rate-limited rescan
            CLOBBERS a good snapshot with zeros.

Production logs from 2026-09-11 show both live: in-play scans reporting
harvested=5/no_coverage=4, and a prematch scan writing 378 snapshots during a
rate-limit cooldown of which 382 fixtures were flagged no_form_data.

What must NOT regress: the BETTING gates (inplay_data_gate - xg_feed_dead,
too_early, too_late, market_already_settled) still never block a harvest. A
fixture with real shot data but a dead xG channel is genuine training data.
That split is what test_scan_snapshot_gates.py pins and it stays intact.
"""
import main


def _match(fid=555, minute=40, status="2H", stats=None):
    """A live fixture. `stats` None means the statistics feed returned nothing."""
    return {
        "fixture": {"id": fid, "status": {"elapsed": minute, "short": status}},
        "teams": {"home": {"id": 10, "name": "Home FC"}, "away": {"id": 20, "name": "Away FC"}},
        "goals": {"home": 1, "away": 0},
        "league": {"id": 39, "name": "Premier League", "country": "England"},
        "events": [],
        "statistics": stats if stats is not None else [],
    }


def _stats_with_shots():
    return [
        {"team": {"name": "Home FC"}, "statistics": [
            {"type": "Shots on Goal", "value": 3},
            {"type": "Corner Kicks", "value": 4},
            {"type": "Ball Possession", "value": "55%"},
        ]},
        {"team": {"name": "Away FC"}, "statistics": [
            {"type": "Shots on Goal", "value": 1},
        ]},
    ]


class _Cursor:
    def __init__(self):
        self._last = ""

    def execute(self, sql, params=()):
        self._last = sql
        return self

    def fetchone(self):
        return None

    def fetchall(self):
        return []


class _Conn:
    def __enter__(self):
        return _Cursor()

    def __exit__(self, *a):
        return False


def _stub_live_scan(monkeypatch, matches):
    monkeypatch.setattr(main, "db_conn", lambda: _Conn())
    monkeypatch.setattr(main, "fetch_live_matches", lambda: matches)
    monkeypatch.setattr(main, "get_league_rates", lambda league_id: main.DEFAULT_LEAGUE_RATES)
    monkeypatch.setattr(main, "fetch_odds", lambda fid, live: {})
    monkeypatch.setattr(main, "load_model_from_settings",
                        lambda name: {"intercept": 0.0, "weights": {}})
    monkeypatch.setattr(main, "_log_predictions", lambda rows: None)
    monkeypatch.setattr(main, "send_telegram", lambda *a, **k: True)
    monkeypatch.setattr(main, "_last_snapshot_ts_bulk", lambda fids: {})
    monkeypatch.setattr(main, "HARVEST_MODE", True)
    monkeypatch.setattr(main, "DUP_COOLDOWN_MIN", 0)
    harvested = []
    monkeypatch.setattr(main, "save_snapshot_from_match",
                        lambda m, raw: harvested.append((m.get("fixture") or {}).get("id")))
    return harvested


# ───────────────────────────── in-play ─────────────────────────────

def test_a_fixture_whose_statistics_feed_returned_nothing_is_not_harvested(monkeypatch):
    # The whole point: this is what a rate-limited /fixtures/statistics call
    # looks like by the time it reaches here - an empty statistics list, which
    # extract_raw_inplay() renders as zeros across the board.
    harvested = _stub_live_scan(monkeypatch, [_match(stats=[])])

    main.production_scan()

    assert harvested == [], (
        "a fixture with no statistics at all was written to tip_snapshots - "
        "that row teaches the model about our own blindness, not about football")


def test_a_fixture_with_real_statistics_is_still_harvested(monkeypatch):
    harvested = _stub_live_scan(monkeypatch, [_match(stats=_stats_with_shots())])

    main.production_scan()

    assert harvested == [555]


def test_no_coverage_is_still_counted_when_the_harvest_is_skipped(monkeypatch):
    # The operator-facing counter must not quietly lose the fixture just
    # because it is no longer harvested.
    _stub_live_scan(monkeypatch, [_match(stats=[])])

    main.production_scan()

    snap = main._get_live_snapshot()
    assert snap["no_coverage"] == 1


def test_a_dead_xg_feed_with_real_shots_is_still_harvested(monkeypatch):
    # inplay_data_gate() blocks this from BETTING (xg_feed_dead) and must go
    # on doing so - but shots and corners genuinely arrived, so it is real
    # training data. This is the split the fix must not collapse.
    monkeypatch.setattr(main, "REQUIRE_XG_FEED", True)
    harvested = _stub_live_scan(monkeypatch, [_match(stats=_stats_with_shots())])

    main.production_scan()

    assert harvested == [555], "an unbettable-but-observed fixture is still training data"


# ───────────────────────────── prematch ─────────────────────────────

def _pre_fixture(fid):
    return {
        "fixture": {"id": fid, "date": "2026-09-12T18:00:00+00:00",
                    "status": {"short": "NS"}},
        "teams": {"home": {"id": 10, "name": "Home FC"}, "away": {"id": 20, "name": "Away FC"}},
        "league": {"id": 39, "name": "Premier League", "country": "England"},
    }


def _stub_prematch(monkeypatch, feat):
    monkeypatch.setattr(main, "db_conn", lambda: _Conn())
    monkeypatch.setattr(main, "_collect_todays_prematch_fixtures", lambda: [_pre_fixture(777)])
    monkeypatch.setattr(main, "_get_prematch_features_bulk",
                        lambda fixtures: ({777: feat}, {777: feat}))
    monkeypatch.setattr(main, "_log_predictions", lambda rows: None)
    monkeypatch.setattr(main, "send_telegram", lambda *a, **k: True)
    monkeypatch.setattr(main, "load_model_from_settings",
                        lambda name: {"intercept": 0.0, "weights": {}})
    monkeypatch.setattr(main, "PREMATCH_DEDUP_ENABLE", False)
    saved = []
    monkeypatch.setattr(main, "save_prematch_snapshot",
                        lambda fid, f, ko: saved.append(fid))
    return saved


def test_an_all_zero_prematch_vector_is_not_written_to_the_training_table(monkeypatch):
    # This is precisely what assemble_prematch_features() returns when every
    # form fetch failed: fully populated, every value 0.0.
    from feature_spec import PRE_FEATURES
    zeros = {k: 0.0 for k in PRE_FEATURES}
    saved = _stub_prematch(monkeypatch, zeros)

    main.prematch_scan_save()

    assert saved == [], (
        "an all-zero form vector was written to prematch_snapshots - and that "
        "table upserts on match_id, so it can also clobber a good snapshot")


def test_a_prematch_vector_with_real_form_is_still_written(monkeypatch):
    from feature_spec import PRE_FEATURES
    feat = {k: 0.0 for k in PRE_FEATURES}
    feat.update({"pm_gf_h": 1.4, "pm_ga_h": 0.9, "pm_win_h": 0.5, "pm_draw_h": 0.2,
                 "pm_gf_a": 1.1, "pm_ga_a": 1.2, "pm_win_a": 0.3, "pm_draw_a": 0.3})
    saved = _stub_prematch(monkeypatch, feat)

    main.prematch_scan_save()

    assert saved == [777]
