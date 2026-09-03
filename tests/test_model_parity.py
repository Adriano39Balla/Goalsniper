"""
Does the deployed model score its own training samples the way training did?

This is the only check that can catch train/serve DRIFT, and drift is the
failure mode neither side's own tests can see: training is internally
consistent, serving is internally consistent, and the bug exists solely in
their disagreement. Both suites stay green while the deployed model quietly
predicts something other than what was fitted.

The market anchor alone put the offset in three places that all had to agree —
the fitted linear predictor, the Platt scaling that must NOT touch it, and
main._anchor_offset(). Two of those three were wrong at some point.
"""
import json

import pytest

import main


def _blob(anchor=None, a=1.0, b=0.0):
    m = {"intercept": -0.2, "weights": {"sot_sum": 0.4, "xg_sum": 0.9},
         "scaler": {"mean": {"sot_sum": 3.0, "xg_sum": 1.0},
                    "scale": {"sot_sum": 2.0, "xg_sum": 0.5}},
         "calibration": {"method": "platt", "a": a, "b": b}}
    if anchor:
        m["market_anchor"] = anchor
    return m


def _install(monkeypatch, blob, samples):
    store = {"model_latest:OU_2.5": json.dumps(blob),
             "model_health:OU_2.5": json.dumps({"golden_samples": samples})}
    monkeypatch.setattr(main, "get_setting_cached", lambda k: store.get(k))
    main._MODELS_CACHE.invalidate()
    main._PARITY_CACHE.invalidate()


def _sample(blob, feats):
    """A sample recorded by a run that agrees with this serving path."""
    return {"features": feats, "prob": main._score_prob(dict(feats), blob)}


def test_a_model_that_agrees_with_its_own_samples_passes(monkeypatch):
    blob = _blob()
    feats = {"sot_sum": 5.0, "xg_sum": 1.4}
    _install(monkeypatch, blob, [_sample(blob, feats)])
    assert main.verify_model_parity("OU_2.5") == (True, None)


def test_drift_in_the_weights_is_caught(monkeypatch):
    blob = _blob()
    feats = {"sot_sum": 5.0, "xg_sum": 1.4}
    sample = _sample(blob, feats)
    drifted = _blob()
    drifted["weights"]["sot_sum"] = 0.55      # deployed model is not the fitted one
    _install(monkeypatch, drifted, [sample])
    ok, why = main.verify_model_parity("OU_2.5")
    assert ok is False
    assert "disagree" in why


def test_drift_in_the_scaler_is_caught(monkeypatch):
    blob = _blob()
    feats = {"sot_sum": 5.0, "xg_sum": 1.4}
    sample = _sample(blob, feats)
    drifted = _blob()
    drifted["scaler"]["scale"]["xg_sum"] = 0.9
    _install(monkeypatch, drifted, [sample])
    assert main.verify_model_parity("OU_2.5")[0] is False


def test_drift_in_the_calibration_is_caught(monkeypatch):
    blob = _blob(a=1.0, b=0.0)
    feats = {"sot_sum": 4.0, "xg_sum": 0.8}
    sample = _sample(blob, feats)
    _install(monkeypatch, _blob(a=1.3, b=0.2), [sample])
    assert main.verify_model_parity("OU_2.5")[0] is False


def test_a_lost_market_anchor_is_caught(monkeypatch):
    # The sharpest case. An anchored model whose anchor stops being applied at
    # serving still produces a perfectly plausible probability — it is simply
    # a different one, off by the market's log-odds.
    anchored = _blob(anchor="market_fair_over25")
    feats = {"sot_sum": 5.0, "xg_sum": 1.4, "market_fair_over25": 0.68}
    sample = _sample(anchored, feats)
    _install(monkeypatch, _blob(), [sample])       # same model, anchor dropped
    ok, why = main.verify_model_parity("OU_2.5")
    assert ok is False and "disagree" in why


def test_an_anchored_model_that_is_intact_passes(monkeypatch):
    anchored = _blob(anchor="market_fair_over25")
    feats = {"sot_sum": 5.0, "xg_sum": 1.4, "market_fair_over25": 0.68}
    _install(monkeypatch, anchored, [_sample(anchored, feats)])
    assert main.verify_model_parity("OU_2.5") == (True, None)


def test_a_head_with_no_samples_is_not_failed(monkeypatch):
    # Heads trained before this existed have none. Refusing those would stop
    # the system on deploy, which is worse than the fault being prevented.
    _install(monkeypatch, _blob(), [])
    assert main.verify_model_parity("OU_2.5") == (True, None)


def test_a_missing_model_is_not_failed_by_this_check(monkeypatch):
    monkeypatch.setattr(main, "get_setting_cached", lambda k: None)
    main._MODELS_CACHE.invalidate()
    main._PARITY_CACHE.invalidate()
    assert main.verify_model_parity("OU_2.5") == (True, None)


def test_an_unscoreable_sample_is_reported_not_swallowed(monkeypatch):
    blob = _blob()
    _install(monkeypatch, blob, [{"features": {"sot_sum": 1.0}, "prob": None}])
    ok, why = main.verify_model_parity("OU_2.5")
    assert ok is False and "could not re-score" in why


def test_a_drifted_head_is_refused_a_bet(monkeypatch):
    # Containment, not a log line: parity is one of the fitness checks.
    blob = _blob()
    feats = {"sot_sum": 5.0, "xg_sum": 1.4}
    sample = _sample(blob, feats)
    drifted = _blob()
    drifted["intercept"] = 0.9
    store = {"model_latest:OU_2.5": json.dumps(drifted),
             "model_health:OU_2.5": json.dumps({"golden_samples": [sample],
                                                "brier_skill": 0.24,
                                                "n_train_matches": 900})}
    monkeypatch.setattr(main, "get_setting_cached", lambda k: store.get(k))
    main._MODELS_CACHE.invalidate()
    main._PARITY_CACHE.invalidate()
    ok, why = main.head_fit_to_bet("OU_2.5")
    assert ok is False and "disagree" in why
