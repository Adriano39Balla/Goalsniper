"""
production_scan() applies several gates in sequence, and they answer
different questions. The duplicate cooldown answers "should this be TIPPED
again?" - it must not also decide "should this appear on the dashboard?",
because the fixture it hides is, by definition, one that just produced a
tip: the match most worth looking at goes missing from the live view for
the whole cooldown window.

Same split the harvest block above it already documents. These tests drive
the real production_scan() with the API and DB stubbed out.
"""
import pytest

import main


def _match(fid=555, minute=40, home_sot=3, away_sot=1):
    return {
        "fixture": {"id": fid, "status": {"elapsed": minute}},
        "teams": {"home": {"id": 10, "name": "Home FC"}, "away": {"id": 20, "name": "Away FC"}},
        "goals": {"home": 1, "away": 0},
        "league": {"id": 39, "name": "Premier League", "country": "England"},
        "events": [],
        "statistics": [
            {"team": {"name": "Home FC"}, "statistics": [
                {"type": "Shots on Goal", "value": home_sot},
                {"type": "Corner Kicks", "value": 4},
                {"type": "Ball Possession", "value": "55%"},
            ]},
            {"team": {"name": "Away FC"}, "statistics": [
                {"type": "Shots on Goal", "value": away_sot},
            ]},
        ],
    }


class _Cursor:
    """Minimal cursor: reports a recent tip for the cooldown probe only."""

    def __init__(self, executed, cooling_down):
        self.executed = executed
        self.cooling_down = cooling_down
        self._last = ""

    def execute(self, sql, params=()):
        self._last = sql
        self.executed.append(sql)
        return self

    def fetchone(self):
        if self.cooling_down and "FROM tips WHERE match_id" in self._last:
            return (1,)
        return None

    def fetchall(self):
        return []


def _stub_scan(monkeypatch, cooling_down: bool, matches=None):
    executed = []

    class _Conn:
        def __enter__(self):
            return _Cursor(executed, cooling_down)

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(main, "db_conn", lambda: _Conn())
    live = [_match()] if matches is None else matches
    monkeypatch.setattr(main, "fetch_live_matches", lambda: live)
    monkeypatch.setattr(main, "get_league_rates", lambda league_id: main.DEFAULT_LEAGUE_RATES)
    monkeypatch.setattr(main, "fetch_odds", lambda fid, live: {})
    monkeypatch.setattr(main, "load_model_from_settings",
                        lambda name: {"intercept": 0.0, "weights": {}})
    monkeypatch.setattr(main, "_log_predictions", lambda rows: None)
    monkeypatch.setattr(main, "send_telegram", lambda *a, **k: True)
    # Keep the test on the tipping/display split, not the harvest cadence.
    monkeypatch.setattr(main, "HARVEST_MODE", False)
    monkeypatch.setattr(main, "DUP_COOLDOWN_MIN", 30)
    return executed


def test_fixture_in_its_cooldown_window_still_reaches_the_dashboard(monkeypatch):
    _stub_scan(monkeypatch, cooling_down=True)
    main._set_live_snapshot([])

    saved, live_seen = main.production_scan()

    assert live_seen == 1
    # Nothing new tipped - the cooldown did its actual job.
    assert saved == 0
    # ...but the match is still on the dashboard.
    snap = main._get_live_snapshot()
    assert [m["fixture_id"] for m in snap["matches"]] == [555]


def test_cooldown_still_blocks_the_tip_itself(monkeypatch):
    executed = _stub_scan(monkeypatch, cooling_down=True)
    main.production_scan()
    assert not any("INSERT INTO tips" in sql for sql in executed)


def test_fixture_outside_the_cooldown_is_scored_and_displayed(monkeypatch):
    _stub_scan(monkeypatch, cooling_down=False)
    main._set_live_snapshot([])
    main.production_scan()
    snap = main._get_live_snapshot()
    assert [m["fixture_id"] for m in snap["matches"]] == [555]


def test_snapshot_records_how_many_fixtures_were_live_and_uncovered(monkeypatch):
    # One scoreable fixture, one with no usable stats at all: the dashboard
    # needs both numbers to explain an empty or short list.
    blank = _match(fid=777)
    blank["statistics"] = []
    _stub_scan(monkeypatch, cooling_down=False, matches=[_match(), blank])
    main.production_scan()

    snap = main._get_live_snapshot()
    assert snap["live_seen"] == 2
    assert snap["no_coverage"] == 1
    assert [m["fixture_id"] for m in snap["matches"]] == [555]


def test_no_live_matches_reports_zero_rather_than_unknown(monkeypatch):
    _stub_scan(monkeypatch, cooling_down=False, matches=[])
    main.production_scan()
    snap = main._get_live_snapshot()
    assert snap["matches"] == []
    assert snap["live_seen"] == 0
