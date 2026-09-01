"""
Snapshots harvested before the odds fix carry market_fair_* values derived
from contaminated prices (team totals and half markets outbidding the real
full-match price). Training on them teaches a market prior that never
existed, so they're dropped to the neutral prior rather than trusted.

The rest of those snapshots - goals, shots, corners, cards - came from the
statistics feed and is untouched, so the rows themselves must survive.
"""
import pytest

import train_models
from feature_spec import NEUTRAL_MARKET_PRIORS, build_inplay_features

CUTOFF = train_models.MARKET_FAIR_TRUSTED_FROM_TS


def _raw():
    return {
        "minute": 60.0, "goals_h": 1.0, "goals_a": 0.0, "sot_h": 4.0, "sot_a": 2.0,
        "market_fair_home": 0.55, "market_fair_draw": 0.25, "market_fair_away": 0.20,
        "market_fair_over25": 0.62, "market_fair_btts_yes": 0.58,
    }


def test_snapshot_from_before_the_fix_has_its_market_fair_stripped():
    raw = _raw()
    dropped = train_models._drop_untrusted_market_fair(raw, CUTOFF - 1)
    assert dropped == 1
    assert not any(k.startswith("market_fair_") for k in raw)


def test_snapshot_from_after_the_fix_is_left_alone():
    raw = _raw()
    dropped = train_models._drop_untrusted_market_fair(raw, CUTOFF)
    assert dropped == 0
    assert raw["market_fair_over25"] == 0.62


def test_stripped_snapshot_falls_back_to_the_neutral_prior_not_zero():
    # The point of dropping rather than zeroing: 0.0 would teach the model
    # "the market says this is impossible", which is a stronger and more
    # wrong claim than "unknown".
    raw = _raw()
    train_models._drop_untrusted_market_fair(raw, CUTOFF - 1)
    f = build_inplay_features(raw, {})
    assert f["market_fair_over25"] == NEUTRAL_MARKET_PRIORS["market_fair_over25"]
    assert f["market_fair_over25"] != 0.0


def test_the_rest_of_the_snapshot_survives():
    raw = _raw()
    train_models._drop_untrusted_market_fair(raw, CUTOFF - 1)
    f = build_inplay_features(raw, {})
    assert f["goals_sum"] == pytest.approx(1.0)
    assert f["sot_sum"] == pytest.approx(6.0)


def test_snapshot_with_no_timestamp_is_left_alone():
    # No timestamp means legacy data, which predates the feature entirely -
    # there is nothing to strip and no basis for guessing.
    raw = _raw()
    assert train_models._drop_untrusted_market_fair(raw, None) == 0


def test_snapshot_predating_the_feature_reports_nothing_dropped():
    raw = {"minute": 30.0, "goals_h": 0.0, "goals_a": 0.0}
    assert train_models._drop_untrusted_market_fair(raw, CUTOFF - 1) == 0
