"""
build_inplay_features()'s market_fair_* pass-through, and the neutral-prior
default for raw dicts that predate this feature entirely (every historically
harvested snapshot, at first). These must default to
feature_spec.NEUTRAL_MARKET_PRIORS, not _f()'s generic 0.0 - a literal 0.0 is
a false "the market says this is impossible" for a probability feature,
unlike the count-based fields (yellow cards, saves) where 0.0 is a
legitimate value.
"""
import pytest

from feature_spec import NEUTRAL_MARKET_PRIORS, build_inplay_features


def _raw(**overrides):
    base = {
        "minute": 60.0,
        "goals_h": 1.0, "goals_a": 0.0,
        "sot_h": 4.0, "sot_a": 2.0,
        "market_fair_home": 0.55, "market_fair_draw": 0.25, "market_fair_away": 0.20,
        "market_fair_over25": 0.62, "market_fair_btts_yes": 0.58,
    }
    base.update(overrides)
    return base


def test_market_fair_probabilities_pass_through_unchanged():
    f = build_inplay_features(_raw(), {})
    assert f["market_fair_home"] == pytest.approx(0.55)
    assert f["market_fair_draw"] == pytest.approx(0.25)
    assert f["market_fair_away"] == pytest.approx(0.20)
    assert f["market_fair_over25"] == pytest.approx(0.62)
    assert f["market_fair_btts_yes"] == pytest.approx(0.58)


def test_missing_market_fair_keys_default_to_neutral_priors_not_zero():
    # A snapshot harvested before this feature existed at all - the keys are
    # simply absent, not present-and-zero.
    legacy_raw = {"minute": 30.0, "goals_h": 0.0, "goals_a": 0.0}
    f = build_inplay_features(legacy_raw, {})
    assert f["market_fair_home"] == NEUTRAL_MARKET_PRIORS["market_fair_home"]
    assert f["market_fair_draw"] == NEUTRAL_MARKET_PRIORS["market_fair_draw"]
    assert f["market_fair_away"] == NEUTRAL_MARKET_PRIORS["market_fair_away"]
    assert f["market_fair_over25"] == NEUTRAL_MARKET_PRIORS["market_fair_over25"]
    assert f["market_fair_btts_yes"] == NEUTRAL_MARKET_PRIORS["market_fair_btts_yes"]
    # Sanity: these are meaningfully different from a bare 0.0 default.
    assert f["market_fair_home"] != 0.0
    assert f["market_fair_over25"] != 0.0


def test_explicit_none_is_treated_the_same_as_absent(monkeypatch):
    raw = _raw(market_fair_home=None)
    f = build_inplay_features(raw, {})
    assert f["market_fair_home"] == NEUTRAL_MARKET_PRIORS["market_fair_home"]
