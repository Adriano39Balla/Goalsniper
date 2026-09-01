"""
Form & momentum: the home side's HOME record and the away side's AWAY
record, judged against how often teams actually win at that venue in that
league.

The venue split is the whole point - a last-N window mixes both venues, and
"wins at home / never wins away" is exactly the shape those two numbers are
supposed to expose. These tests pin the filtering, the recency weighting
that carries over from team_form_stats(), and the sample floor that stops
one match being reported as a trend.
"""
import pytest

import main
from feature_spec import venue_form_stats


def _game(home_id, away_id, gh, ga, date, status="FT"):
    return {
        "fixture": {"date": date, "status": {"short": status}},
        "teams": {"home": {"id": home_id}, "away": {"id": away_id}},
        "goals": {"home": gh, "away": ga},
    }


# Team 1 at home: won, won, lost.  Team 1 away: lost, lost.
MIXED_WINDOW = [
    _game(1, 99, 2, 0, "2026-08-30T12:00:00+00:00"),   # home win
    _game(98, 1, 3, 1, "2026-08-27T12:00:00+00:00"),   # away loss
    _game(1, 97, 1, 0, "2026-08-24T12:00:00+00:00"),   # home win
    _game(96, 1, 2, 1, "2026-08-21T12:00:00+00:00"),   # away loss
    _game(1, 95, 0, 2, "2026-08-18T12:00:00+00:00"),   # home loss
]


def test_venue_split_only_counts_games_played_at_that_venue():
    home = venue_form_stats(1, MIXED_WINDOW, "home")
    away = venue_form_stats(1, MIXED_WINDOW, "away")
    assert home["played"] == 3
    assert away["played"] == 2


def test_home_and_away_records_differ_for_the_same_team():
    # 2 wins from 3 at home vs 0 from 2 away - collapsing these into one
    # overall win rate is exactly the signal this split exists to keep.
    home = venue_form_stats(1, MIXED_WINDOW, "home")
    away = venue_form_stats(1, MIXED_WINDOW, "away")
    assert home["win"] > away["win"]
    assert away["win"] == pytest.approx(0.0)


def test_venue_weights_are_recomputed_over_the_filtered_window():
    # The most recent HOME game must carry full weight 1.0, not the decayed
    # weight it had at position 0 of the mixed list. Two wins (most recent
    # and third-most-recent home game) and one loss, weights 1.0/0.8/0.64:
    #   win share = (1.0 + 0.8) / (1.0 + 0.8 + 0.64)
    home = venue_form_stats(1, MIXED_WINDOW, "home")
    assert home["win"] == pytest.approx(1.8 / 2.44)


def test_unfinished_fixtures_are_ignored():
    window = MIXED_WINDOW + [_game(1, 94, 5, 0, "2026-09-01T12:00:00+00:00", status="NS")]
    assert venue_form_stats(1, window, "home")["played"] == 3


def test_team_that_never_played_at_that_venue_reports_nothing():
    only_away = [_game(98, 1, 1, 0, "2026-08-27T12:00:00+00:00")]
    st = venue_form_stats(1, only_away, "home")
    assert st["played"] == 0
    assert st["win"] == 0.0


# ───────── verdict banding ─────────

def test_verdict_is_withheld_below_the_sample_floor():
    # 0% off two games is not evidence of anything.
    assert main._venue_verdict(0.0, 0.45, played=2) is None


def test_verdict_bands_above_and_below_the_league_baseline():
    assert main._venue_verdict(0.65, 0.45, played=5)["tone"] == "good"
    assert main._venue_verdict(0.65, 0.45, played=5)["text"] == "well above the league's usual"
    assert main._venue_verdict(0.52, 0.45, played=5)["text"] == "above the league's usual"
    assert main._venue_verdict(0.46, 0.45, played=5)["text"] == "about the league's usual"
    assert main._venue_verdict(0.38, 0.45, played=5)["text"] == "below the league's usual"
    assert main._venue_verdict(0.10, 0.45, played=5)["tone"] == "bad"


def test_verdict_compares_against_the_supplied_baseline_not_a_constant():
    # 30% away wins is poor against a 45% baseline but strong against 15%.
    assert main._venue_verdict(0.30, 0.45, played=5)["tone"] == "bad"
    assert main._venue_verdict(0.30, 0.15, played=5)["tone"] == "good"


# ───────── build_match_form ─────────

def _stub_form_api(monkeypatch, window=MIXED_WINDOW):
    calls = []

    def _fake(team_id, n=5):
        calls.append((team_id, n))
        return window

    monkeypatch.setattr(main, "_api_last_fixtures", _fake)
    monkeypatch.setattr(main, "get_league_venue_rates",
                        lambda league_id: {"home_win": 0.45, "away_win": 0.29, "n": 500})
    return calls


def test_build_match_form_returns_a_card_per_side(monkeypatch):
    _stub_form_api(monkeypatch)
    entry = {"home_id": 1, "away_id": 1, "home": "Home FC", "away": "Away FC", "league_id": 39}
    out = main.build_match_form(entry)

    assert out["available"] is True
    assert out["home"]["team"] == "Home FC" and out["home"]["venue"] == "home"
    assert out["away"]["team"] == "Away FC" and out["away"]["venue"] == "away"
    # Same underlying window, but each side is read at its own venue.
    assert out["home"]["played"] == 3
    assert out["away"]["played"] == 2
    assert out["away"]["win_pct"] == pytest.approx(0.0)


def test_build_match_form_asks_for_the_full_window(monkeypatch):
    calls = _stub_form_api(monkeypatch)
    main.build_match_form({"home_id": 1, "away_id": 2, "home": "H", "away": "A", "league_id": 39})
    assert {n for _, n in calls} == {main.FORM_WINDOW_GAMES}


def test_build_match_form_degrades_without_team_ids(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("must not hit the API without team ids")

    monkeypatch.setattr(main, "_api_last_fixtures", _boom)
    out = main.build_match_form({"home_id": 0, "away_id": 0, "home": "H", "away": "A"})
    assert out["available"] is False
    assert "team ids" in out["reason"]


def test_each_side_is_judged_against_its_own_venue_baseline(monkeypatch):
    # A 40% record is below a 45% home baseline but above a 29% away one -
    # so the same rate must not produce the same verdict on both sides.
    _stub_form_api(monkeypatch)
    monkeypatch.setattr(main, "_venue_verdict",
                        lambda rate, league_rate, played: {"text": f"{league_rate}", "tone": "x"})
    out = main.build_match_form({"home_id": 1, "away_id": 2, "home": "H", "away": "A", "league_id": 39})
    assert out["home"]["verdict"]["text"] == "0.45"
    assert out["away"]["verdict"]["text"] == "0.29"
