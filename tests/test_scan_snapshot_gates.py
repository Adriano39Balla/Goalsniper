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


def _match(fid=555, minute=40, home_sot=3, away_sot=1, status="2H"):
    return {
        "fixture": {"id": fid, "status": {"elapsed": minute, "short": status}},
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


# ───────── the information gate splits the same way ─────────
# A fixture we refuse to BET is still a fixture worth SEEING, with the reason
# attached. Blocking the snapshot instead would hide exactly the matches
# someone is asking "why wasn't this tipped?" about - the mistake the cooldown
# above already made once.

def _confident_model(monkeypatch, prob=0.90):
    """Force every candidate well over its threshold so only the gates decide."""
    import math
    intercept = math.log(prob / (1.0 - prob))
    monkeypatch.setattr(main, "load_model_from_settings",
                        lambda name: {"intercept": intercept, "weights": {}})
    monkeypatch.setattr(main, "_get_market_threshold", lambda m: 50.0)


def test_a_fixture_with_a_dead_xg_feed_stays_on_the_dashboard(monkeypatch):
    # SOT 3-1 recorded, no xG at all - provably an absent channel.
    _stub_scan(monkeypatch, cooling_down=False)
    _confident_model(monkeypatch)
    main._set_live_snapshot([])

    saved, live_seen = main.production_scan()

    assert live_seen == 1
    assert saved == 0, "not bettable"
    snap = main._get_live_snapshot()
    assert [m["fixture_id"] for m in snap["matches"]] == [555]
    assert snap["matches"][0]["data_block"] == "xg_feed_dead"


def test_the_reason_is_attached_to_every_market_not_just_the_match(monkeypatch):
    # "68% confidence, nothing sent" is the question the dashboard exists to
    # answer, and it is asked per market.
    _stub_scan(monkeypatch, cooling_down=False)
    _confident_model(monkeypatch)
    main._set_live_snapshot([])
    main.production_scan()

    markets = main._get_live_snapshot()["matches"][0]["markets"]
    over = [m for m in markets if m["prob_pct"] >= m["threshold_pct"]]
    assert over, "the stub model should clear every threshold"
    assert all(m["decision"] == "xg_feed_dead" for m in over)


def test_a_blocked_fixture_is_never_run_through_the_price_gate(monkeypatch):
    # Not a budget saving - extract_features() already fetches this fixture's
    # odds as a model input, and ODDS_CACHE makes the second call a hit. The
    # point is that a price-gate verdict on an unbettable fixture would be
    # misleading: "EV too low" invites tuning EDGE_MIN_BPS when the actual
    # problem is that the match was never observed.
    gated = []
    _stub_scan(monkeypatch, cooling_down=False)
    _confident_model(monkeypatch)
    real_gate = main._price_gate
    monkeypatch.setattr(main, "_price_gate",
                        lambda *a, **k: gated.append(a) or real_gate(*a, **k))
    main.production_scan()
    assert gated == []


def test_a_healthy_fixture_is_not_blocked(monkeypatch):
    healthy = _match()
    healthy["statistics"][0]["statistics"].append({"type": "Expected Goals", "value": "1.24"})
    _stub_scan(monkeypatch, cooling_down=False, matches=[healthy])
    _confident_model(monkeypatch)
    main._set_live_snapshot([])
    main.production_scan()

    snap = main._get_live_snapshot()
    assert snap["matches"][0]["data_block"] is None


def test_harvesting_is_not_blocked_by_the_information_gate(monkeypatch):
    # The gate governs betting. Refusing to harvest a fixture because its xG
    # feed is down is how in-play data collection died the last time a tipping
    # check was moved ahead of the harvest block.
    harvested = []
    _stub_scan(monkeypatch, cooling_down=False)
    monkeypatch.setattr(main, "HARVEST_MODE", True)
    monkeypatch.setattr(main, "_last_snapshot_ts_bulk", lambda fids: {})
    monkeypatch.setattr(main, "save_snapshot_from_match",
                        lambda m, raw: harvested.append((m.get("fixture") or {}).get("id")))
    main.production_scan()
    assert harvested == [555], "an unbettable fixture is still training data"


def test_a_fixture_in_extra_time_is_not_tipped(monkeypatch):
    # Every market here settles on 90 minutes. Once a tie reaches extra time
    # the bet is already decided, and the scoreline now includes goals that do
    # not count toward it.
    et = _match(status="ET", minute=105)
    et["statistics"][0]["statistics"].append({"type": "Expected Goals", "value": "1.9"})
    _stub_scan(monkeypatch, cooling_down=False, matches=[et])
    _confident_model(monkeypatch)
    main._set_live_snapshot([])

    saved, _ = main.production_scan()

    assert saved == 0
    assert main._get_live_snapshot()["matches"][0]["data_block"] == "market_already_settled"


def test_a_penalty_shootout_is_not_tipped(monkeypatch):
    ps = _match(status="P", minute=120)
    ps["statistics"][0]["statistics"].append({"type": "Expected Goals", "value": "1.9"})
    _stub_scan(monkeypatch, cooling_down=False, matches=[ps])
    _confident_model(monkeypatch)
    main._set_live_snapshot([])
    main.production_scan()
    assert main._get_live_snapshot()["matches"][0]["data_block"] == "market_already_settled"


def test_a_normal_second_half_fixture_is_still_tippable(monkeypatch):
    ok = _match(status="2H", minute=60)
    ok["statistics"][0]["statistics"].append({"type": "Expected Goals", "value": "1.2"})
    _stub_scan(monkeypatch, cooling_down=False, matches=[ok])
    _confident_model(monkeypatch)
    main._set_live_snapshot([])
    main.production_scan()
    assert main._get_live_snapshot()["matches"][0]["data_block"] is None


def test_half_time_counts_as_open(monkeypatch):
    ht = _match(status="HT", minute=45)
    ht["statistics"][0]["statistics"].append({"type": "Expected Goals", "value": "1.2"})
    _stub_scan(monkeypatch, cooling_down=False, matches=[ht])
    _confident_model(monkeypatch)
    main._set_live_snapshot([])
    main.production_scan()
    assert main._get_live_snapshot()["matches"][0]["data_block"] is None
