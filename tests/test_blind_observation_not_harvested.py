"""
A failed API call must never be persisted as training data.

fetch_odds(), fetch_match_stats() and _api_last_fixtures() each enforce this
at the CACHE layer, and each says so in its own comment: a call that failed is
not "nothing was there". It was being violated one layer further down, at the
point of persistence:

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
            because that table upserts on match_id, a rescan during a
            rate-limit cooldown CLOBBERS a good snapshot.

Production logs from 2026-09-11 show both live: in-play scans reporting
harvested=5/no_coverage=4, and a prematch scan writing 378 snapshots in 3.4
seconds during a 60-second cooldown — i.e. without fetching anything.

WHAT MUST NOT REGRESS, and why this is not the mistake production_scan()'s
own comment warns about. Moving a gate ahead of the harvest block once took
in-play collection to 1 snapshot in six hours, and that lesson stands. It
applies to gates that answer "would we BET this?" — those must never stop
data collection, because a fixture with real shots and a dead xG channel is
genuinely observed data. stats_coverage_ok()'s paired-fields branch answers a
different question: "did the statistics feed return ANYTHING?". A row that
fails it is not a thin observation, it is the absence of one. The tests below
pin both halves of that split.
"""
import main


# ───────────────────────────── in-play ─────────────────────────────

def _match(fid=555, minute=40, stats=None):
    """A live fixture. `stats` None means the statistics feed returned nothing."""
    return {
        "fixture": {"id": fid, "status": {"elapsed": minute, "short": "2H"}},
        "teams": {"home": {"id": 10, "name": "Home FC"}, "away": {"id": 20, "name": "Away FC"}},
        "goals": {"home": 1, "away": 0},
        "league": {"id": 39, "name": "Premier League", "country": "England"},
        "events": [],
        "statistics": stats if stats is not None else [],
    }


def _stats_with_shots(home_sot=3, away_sot=1):
    return [
        {"team": {"name": "Home FC"}, "statistics": [
            {"type": "Shots on Goal", "value": home_sot},
            {"type": "Corner Kicks", "value": 4},
            {"type": "Ball Possession", "value": "55%"},
        ]},
        {"team": {"name": "Away FC"}, "statistics": [
            {"type": "Shots on Goal", "value": away_sot},
            {"type": "Corner Kicks", "value": 2},
            {"type": "Ball Possession", "value": "45%"},
        ]},
    ]


class _Cursor:
    def execute(self, sql, params=()):
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


def test_a_fixture_whose_statistics_feed_returned_nothing_is_not_harvested(monkeypatch):
    # What a rate-limited /fixtures/statistics call looks like by the time it
    # reaches here: an empty statistics list, which extract_raw_inplay()
    # renders as zeros across the board.
    harvested = _stub_live_scan(monkeypatch, [_match(stats=[])])

    main.production_scan()

    assert harvested == [], (
        "a fixture with no statistics at all was written to tip_snapshots — "
        "that row teaches the model about our own blindness, not about football")


def test_a_fixture_with_real_statistics_is_still_harvested(monkeypatch):
    harvested = _stub_live_scan(monkeypatch, [_match(stats=_stats_with_shots())])

    main.production_scan()

    assert harvested == [555]


def test_a_goalless_but_fully_reported_fixture_is_still_harvested(monkeypatch):
    """
    THE CASE THE FIX MUST NOT COLLAPSE. Nothing has happened yet — no shots on
    target either side — but the feed answered and returned field groups. That
    is a real observation of a quiet match, not an absent one, and it is
    exactly the kind of row a totals model needs. stats_coverage_ok() reads
    field PRESENCE, not field values, so it passes.
    """
    harvested = _stub_live_scan(
        monkeypatch, [_match(stats=_stats_with_shots(home_sot=0, away_sot=0))])

    main.production_scan()

    assert harvested == [555], "a genuine 0-0 with a working feed is training data"


def test_no_coverage_is_still_counted_when_the_harvest_is_skipped(monkeypatch):
    # The operator-facing counter must not quietly lose the fixture just
    # because it is no longer harvested.
    _stub_live_scan(monkeypatch, [_match(stats=[])])

    main.production_scan()

    assert main._get_live_snapshot()["no_coverage"] == 1


# ───────────────────────────── prematch ─────────────────────────────

def _pre_fixture(fid=777):
    return {
        "fixture": {"id": fid, "date": "2026-09-20T18:00:00+00:00",
                    "status": {"short": "NS"}},
        "teams": {"home": {"id": 10, "name": "Home FC"}, "away": {"id": 20, "name": "Away FC"}},
        "league": {"id": 39, "name": "Premier League", "country": "England"},
    }


def _stub_prematch(monkeypatch, feat, fid=777):
    monkeypatch.setattr(main, "db_conn", lambda: _Conn())
    monkeypatch.setattr(main, "_collect_todays_prematch_fixtures", lambda: [_pre_fixture(fid)])
    monkeypatch.setattr(main, "_get_prematch_features_bulk",
                        lambda fixtures: ({fid: feat}, {fid: feat}))
    monkeypatch.setattr(main, "_log_predictions", lambda rows: None)
    monkeypatch.setattr(main, "send_telegram", lambda *a, **k: True)
    # A weightless blob makes _linpred() raise, which would swallow the very
    # difference these tests measure; one real weight keeps scoring reachable.
    monkeypatch.setattr(main, "load_model_from_settings",
                        lambda name: {"intercept": 0.0, "weights": {"pm_gf_h": 0.1}})
    monkeypatch.setattr(main, "PREMATCH_DEDUP_ENABLE", False)
    saved = []
    monkeypatch.setattr(main, "save_prematch_snapshot",
                        lambda f, ft, ko: saved.append(f))
    return saved


def _zeros():
    from feature_spec import PRE_FEATURES
    return {k: 0.0 for k in PRE_FEATURES}


def _real_form():
    feat = _zeros()
    feat.update({"pm_gf_h": 1.4, "pm_ga_h": 0.9, "pm_win_h": 0.5, "pm_draw_h": 0.2,
                 "pm_gf_a": 1.1, "pm_ga_a": 1.2, "pm_win_a": 0.3, "pm_draw_a": 0.3})
    return feat


def test_an_all_zero_prematch_vector_is_not_written_to_the_training_table(monkeypatch):
    # Precisely what assemble_prematch_features() returns when every form fetch
    # failed: fully populated, every value 0.0 — so `if not feat` never fires.
    saved = _stub_prematch(monkeypatch, _zeros())

    main.prematch_scan_save()

    assert saved == [], (
        "an all-zero form vector was written to prematch_snapshots — and that "
        "table upserts on match_id, so it can also clobber a good snapshot")


def test_an_all_zero_prematch_vector_is_not_tipped_either(monkeypatch):
    # Every fixture in the same outage gets this identical vector, so the model
    # returns one probability for all of them and the scan can emit a burst of
    # matching tips off data it never received.
    _stub_prematch(monkeypatch, _zeros())
    sent = []
    monkeypatch.setattr(main, "send_telegram", lambda text: sent.append(text) or True)

    assert main.prematch_scan_save() == 0
    assert sent == []


def test_a_prematch_vector_with_real_form_is_still_written(monkeypatch):
    saved = _stub_prematch(monkeypatch, _real_form())

    main.prematch_scan_save()

    assert saved == [777]


def test_one_side_missing_is_enough_to_block(monkeypatch):
    # A half-zero vector is just as unusable as an all-zero one: the model is
    # comparing a real team against a team that has no recorded history.
    feat = _zeros()
    feat.update({"pm_gf_a": 1.1, "pm_ga_a": 1.2, "pm_win_a": 0.3, "pm_draw_a": 0.3})
    saved = _stub_prematch(monkeypatch, feat)

    main.prematch_scan_save()

    assert saved == []


# ───────────────── the gate's own argument ─────────────────
#
# prematch_data_gate() claims all four of gf/ga/win/draw at zero can only mean
# an empty window. Every finished game lands in exactly one outcome, and each
# outcome moves a different one of those four. These pin that argument down,
# because if it were wrong the gate would be discarding real data.

def test_a_goalless_draw_still_counts_as_observed():
    feat = _zeros()
    feat.update({"pm_draw_h": 1.0, "pm_draw_a": 1.0})
    assert main.prematch_data_gate(feat) is None


def test_a_defeat_to_nil_still_counts_as_observed():
    # gf, win and draw are all zero, but a goal was conceded: pm_ga carries it.
    feat = _zeros()
    feat.update({"pm_ga_h": 3.0, "pm_ga_a": 2.0})
    assert main.prematch_data_gate(feat) is None


def test_the_gate_names_which_side_is_missing():
    assert main.prematch_data_gate(_zeros()) == "no_form_data_home"
    half = _zeros()
    half.update({"pm_gf_h": 1.0})
    assert main.prematch_data_gate(half) == "no_form_data_away"
