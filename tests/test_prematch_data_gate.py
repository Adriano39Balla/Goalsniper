"""
The prematch path had no counterpart to inplay_data_gate().

production_scan() refuses an all-zero observation twice — in
stats_coverage_ok() and again in inplay_data_gate() — on the stated grounds
that such a vector "makes the model output sigmoid(intercept), which carries no
match information: fine to record, not fine to bet on".

prematch_scan_save() checked only `if not feat`. But
assemble_prematch_features() ends with

    return {k: float(f.get(k, 0.0)) for k in PRE_FEATURES}

so it ALWAYS returns a fully-populated dict and `feat` is never falsy. A
fixture whose team-form fetches failed arrived as a complete vector of zeros
and was scored and tipped like any other.
"""
import pytest

import main
from feature_spec import assemble_prematch_features
from main import prematch_data_gate


def _fx(team_id, gf, ga, home=True, day=1):
    return {
        "fixture": {"id": 1, "status": {"short": "FT"},
                    "date": f"2026-09-{day:02d}T15:00:00Z"},
        "teams": {"home": {"id": team_id if home else 999},
                  "away": {"id": 999 if home else team_id}},
        "goals": {"home": gf if home else ga, "away": ga if home else gf},
        "date": f"2026-09-{day:02d}T15:00:00Z",
    }


def _feat(last_h, last_a):
    return assemble_prematch_features(
        home_id=1, away_id=2, last_h=last_h, last_a=last_a, h2h=[],
        kickoff_ts=1700000000, rating_h=1500.0, rating_a=1500.0,
        league_rates={"btts": 0.5, "ov25": 0.5, "ov35": 0.3})


def test_a_total_form_outage_is_not_bettable():
    # Exactly what the poisoned team-form cache produced: both windows empty.
    feat = _feat([], [])
    assert feat, "the vector is fully populated — this is why `if not feat` missed it"
    assert prematch_data_gate(feat) == "no_form_data_home"


def test_one_side_missing_is_enough_to_block():
    feat = _feat([_fx(1, 2, 0)], [])
    assert prematch_data_gate(feat) == "no_form_data_away"


def test_a_fixture_with_real_form_on_both_sides_passes():
    feat = _feat([_fx(1, 2, 0, day=1), _fx(1, 1, 1, day=2)],
                 [_fx(2, 0, 1, home=False, day=1)])
    assert prematch_data_gate(feat) is None


# Every finished game lands in exactly one outcome, and each moves a different
# one of these four to non-zero — so "all four zero" means the window was empty
# and nothing else. These pin that argument down.

def test_a_goalless_draw_still_counts_as_observed():
    # gf and ga are both 0, but a draw was played: pm_draw carries it.
    feat = _feat([_fx(1, 0, 0)], [_fx(2, 0, 0, home=False)])
    assert prematch_data_gate(feat) is None


def test_a_defeat_to_nil_still_counts_as_observed():
    # gf, win and draw are all 0, but a goal was conceded: pm_ga carries it.
    feat = _feat([_fx(1, 0, 3)], [_fx(2, 0, 2, home=False)])
    assert prematch_data_gate(feat) is None


def test_the_gate_blocks_tipping_but_not_harvesting(monkeypatch):
    """
    The gate governs BETTING only. production_scan() harvests before its
    coverage gate deliberately — moving collection behind a betting gate once
    took in-play harvesting down to 1 snapshot in six hours.
    """
    fx = {"fixture": {"id": 77, "date": "2026-09-01T15:00:00Z"},
          "league": {"id": 5, "country": "X", "name": "Y"},
          "teams": {"home": {"name": "H"}, "away": {"name": "A"}}}
    blind = _feat([], [])

    monkeypatch.setattr(main, "_collect_todays_prematch_fixtures", lambda: [fx])
    monkeypatch.setattr(main, "_get_prematch_features_bulk",
                        lambda fixtures: ({77: blind}, {77}))
    harvested, sent = [], []
    monkeypatch.setattr(main, "save_prematch_snapshot",
                        lambda fid, feat, ts: harvested.append(fid))
    monkeypatch.setattr(main, "send_telegram", lambda text: sent.append(text) or True)
    monkeypatch.setattr(main, "_log_predictions", lambda rows: None)

    saved = main.prematch_scan_save()

    assert saved == 0, "a fixture with no form data must not be tipped"
    assert sent == [], "nothing may reach Telegram off an all-zero vector"
    assert harvested == [77], "harvesting must survive the betting gate"
