"""
score_live_matches_now() is the read-only, on-demand twin of
production_scan()'s live-scoring step, backing POST /dashboard/live/refresh
(and the standalone desktop tool that calls it). It must never touch
tips/predictions or send Telegram - these tests monkeypatch main.send_telegram
to blow up if it's ever called, to keep that guarantee honest.
"""
import pytest

import main


def _match(fid=555, minute=40, home_sot=3, away_sot=1, cor=4, pos="55%",
           league_id=39, league_name="Premier League"):
    return {
        "fixture": {"id": fid, "status": {"elapsed": minute}},
        "teams": {"home": {"name": "Home FC"}, "away": {"name": "Away FC"}},
        "goals": {"home": 1, "away": 0},
        "league": {"id": league_id, "name": league_name, "country": ""},
        "events": [],
        "statistics": [
            {"team": {"name": "Home FC"}, "statistics": [
                {"type": "Shots on Goal", "value": home_sot},
                {"type": "Corner Kicks", "value": cor},
                {"type": "Ball Possession", "value": pos},
            ]},
            {"team": {"name": "Away FC"}, "statistics": [
                {"type": "Shots on Goal", "value": away_sot},
            ]},
        ],
    }


def _boom(*a, **k):
    raise AssertionError("score_live_matches_now must never send Telegram")


def _stub_league_rates(monkeypatch):
    # extract_features() -> get_league_rates() hits the DB for league base
    # rates; these tests are about candidate scoring, not that lookup, so
    # bypass it with the same defaults feature_spec falls back to.
    monkeypatch.setattr(main, "get_league_rates", lambda league_id: main.DEFAULT_LEAGUE_RATES)


def _stub_odds(monkeypatch):
    # extract_features() -> _market_fair_priors() -> fetch_odds() would
    # otherwise attempt a real HTTPS call per test (slow, and pointless with
    # a fake API key). Returning {} exercises the real neutral-prior fallback
    # path rather than skipping it.
    monkeypatch.setattr(main, "fetch_odds", lambda fid, live: {})


def test_scores_a_covered_live_fixture(monkeypatch):
    _stub_league_rates(monkeypatch)
    _stub_odds(monkeypatch)
    monkeypatch.setattr(main, "fetch_live_matches", lambda: [_match()])
    monkeypatch.setattr(main, "load_model_from_settings",
                         lambda name: {"intercept": 0.0, "weights": {}})
    monkeypatch.setattr(main, "send_telegram", _boom)

    out, live_seen = main.score_live_matches_now()

    assert live_seen == 1
    assert len(out) == 1
    entry = out[0]
    assert entry["fixture_id"] == 555
    assert entry["home"] == "Home FC"
    assert entry["away"] == "Away FC"
    assert entry["minute"] == 40
    assert entry["markets"]

    btts_yes = next(m for m in entry["markets"] if m["suggestion"] == "BTTS: Yes")
    assert btts_yes["prob_pct"] == pytest.approx(50.0)


def test_never_writes_to_the_tips_table(monkeypatch):
    _stub_league_rates(monkeypatch)
    _stub_odds(monkeypatch)
    monkeypatch.setattr(main, "fetch_live_matches", lambda: [_match()])
    monkeypatch.setattr(main, "load_model_from_settings",
                         lambda name: {"intercept": 0.0, "weights": {}})
    monkeypatch.setattr(main, "send_telegram", _boom)

    executed = []

    class _RecordingCursor:
        def execute(self, sql, params=()):
            executed.append(sql)
            return self

        def fetchone(self):
            return None

        def fetchall(self):
            return []

    class _RecordingConn:
        def __enter__(self):
            return _RecordingCursor()

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(main, "db_conn", lambda: _RecordingConn())
    main.score_live_matches_now()

    assert not any("INSERT INTO tips" in sql for sql in executed)
    assert not any("INSERT INTO predictions" in sql for sql in executed)


def test_fixture_below_minimum_minute_is_excluded(monkeypatch):
    _stub_league_rates(monkeypatch)
    _stub_odds(monkeypatch)
    monkeypatch.setattr(main, "fetch_live_matches", lambda: [_match(minute=3)])
    monkeypatch.setattr(main, "load_model_from_settings",
                         lambda name: {"intercept": 0.0, "weights": {}})
    out, live_seen = main.score_live_matches_now()
    assert live_seen == 1
    assert out == []


def test_fixture_without_stat_coverage_is_excluded(monkeypatch):
    _stub_league_rates(monkeypatch)
    _stub_odds(monkeypatch)
    empty_match = _match(home_sot=0, away_sot=0, cor=0, pos="0%")
    monkeypatch.setattr(main, "fetch_live_matches", lambda: [empty_match])
    monkeypatch.setattr(main, "load_model_from_settings",
                         lambda name: {"intercept": 0.0, "weights": {}})
    out, live_seen = main.score_live_matches_now()
    assert live_seen == 1
    assert out == []


def test_one_bad_fixture_does_not_break_the_rest(monkeypatch):
    _stub_league_rates(monkeypatch)
    _stub_odds(monkeypatch)
    broken = {"fixture": {"id": None}}  # missing everything -> fid resolves to 0
    good = _match(fid=777)
    monkeypatch.setattr(main, "fetch_live_matches", lambda: [broken, good])
    monkeypatch.setattr(main, "load_model_from_settings",
                         lambda name: {"intercept": 0.0, "weights": {}})
    out, live_seen = main.score_live_matches_now()
    assert live_seen == 2
    assert [e["fixture_id"] for e in out] == [777]


def test_no_live_matches_returns_empty(monkeypatch):
    monkeypatch.setattr(main, "fetch_live_matches", lambda: [])
    out, live_seen = main.score_live_matches_now()
    assert out == []
    assert live_seen == 0
