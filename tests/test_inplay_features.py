"""
build_inplay_features() derivations for the newly-extracted stats fields
(yellow cards, goalkeeper saves, pass accuracy) - added because these were
already present in every /fixtures/statistics response the app was paying
for, just never read out of it.
"""
import pytest

from feature_spec import MIN_COUNT_DENOM, build_inplay_features


def _raw(**overrides):
    base = {
        "minute": 60.0,
        "goals_h": 1.0, "goals_a": 0.0,
        "xg_h": 1.2, "xg_a": 0.5,
        "sot_h": 4.0, "sot_a": 2.0,
        "cor_h": 3.0, "cor_a": 1.0,
        "pos_h": 55.0, "pos_a": 45.0,
        "red_h": 0.0, "red_a": 0.0,
        "total_shots_h": 8.0, "total_shots_a": 5.0,
        "shots_inside_h": 5.0, "shots_inside_a": 2.0,
        "fouls_h": 6.0, "fouls_a": 8.0,
        "yellow_h": 1.0, "yellow_a": 2.0,
        "saves_h": 1.0, "saves_a": 3.0,
        "passes_h": 300.0, "passes_a": 250.0,
        "passes_acc_h": 270.0, "passes_acc_a": 200.0,
    }
    base.update(overrides)
    return base


def test_yellow_cards_are_summed_and_diffed():
    f = build_inplay_features(_raw(), {})
    assert f["yellow_sum"] == pytest.approx(3.0)
    assert f["yellow_diff"] == pytest.approx(-1.0)


def test_save_rate_is_keyed_against_the_opponents_shots_on_target():
    # Home keeper's saves (1) are a response to AWAY's shots on target (2),
    # not home's own - that's what makes this a new signal, not a rescale.
    f = build_inplay_features(_raw(), {})
    assert f["save_rate_h"] == pytest.approx(1.0 / 2.0)
    assert f["save_rate_a"] == pytest.approx(3.0 / 4.0)


def test_save_rate_does_not_divide_by_zero_with_no_shots_on_target():
    f = build_inplay_features(_raw(sot_h=0.0, sot_a=0.0), {})
    assert f["save_rate_h"] == pytest.approx(1.0 / MIN_COUNT_DENOM)
    assert f["save_rate_a"] == pytest.approx(3.0 / MIN_COUNT_DENOM)


def test_pass_accuracy_ratio():
    f = build_inplay_features(_raw(), {})
    assert f["pass_accuracy_h"] == pytest.approx(270.0 / 300.0)
    assert f["pass_accuracy_a"] == pytest.approx(200.0 / 250.0)


def test_pass_accuracy_does_not_divide_by_zero_with_no_passes():
    f = build_inplay_features(_raw(passes_h=0.0, passes_acc_h=0.0), {})
    assert f["pass_accuracy_h"] == pytest.approx(0.0)


def test_missing_new_fields_default_to_zero_for_legacy_snapshots():
    # Snapshots harvested before this change won't carry yellow/saves/passes
    # keys at all; build_inplay_features must degrade to 0.0, not raise.
    legacy_raw = {"minute": 30.0, "goals_h": 0.0, "goals_a": 0.0}
    f = build_inplay_features(legacy_raw, {})
    assert f["yellow_sum"] == 0.0
    assert f["save_rate_h"] == 0.0
    assert f["pass_accuracy_h"] == 0.0
