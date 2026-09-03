"""
The system refuses its own impossible output.

Every defect that reached production here had the same shape: code that reads
as correct on its own, producing a number that is impossible read in context.
Source review cannot catch that class — the function is locally right, and the
error only exists in the relationship between it and a consumer written later.
Checking the OUTPUT catches all of them, and costs nothing.

So every check below is a replay of a defect that actually shipped, asserted
against the real numbers that defect produced. A CRITICAL finding marks the
head unfit to bet, because a warning in a nightly digest is not containment
when the scan runs every five minutes.
"""
import pytest

from train_models import validate_training_run


def _head(**kw):
    m = {"brier": 0.19, "brier_skill": 0.22, "acc": 0.69,
         "mean_predicted": 0.5174, "mean_actual": 0.4817,
         "calibration_gap_pct": 3.57, "prevalence": 0.5,
         "n_train": 3243, "n_train_matches": 900,
         "confusion_matrix": {"tp": 372, "fp": 173, "tn": 423, "fn": 182}}
    m.update(kw)
    return m


def _run(**heads):
    return validate_training_run({"metrics": heads})


def _checks(res, head=None):
    return {f["check"] for f in res["findings"]
            if head is None or f["head"] == head}


def test_a_clean_run_produces_nothing():
    res = _run(OU_2_5=_head())
    assert res["n_critical"] == 0
    assert res["unfit_heads"] == {}


# ───────── replay: the calibration-gap sign inversion ─────────

def test_an_inverted_calibration_sign_is_caught():
    # train_models computed actual - predicted while every consumer assumed
    # predicted - actual, so an 8.8pp OVERconfident head was reported as
    # underconfident and the EV warning fired only on the safe heads.
    res = _run(OU_2_5=_head(mean_predicted=0.5918, mean_actual=0.5042,
                            calibration_gap_pct=-8.76))
    assert "calibration_gap_sign" in _checks(res)
    assert "OU_2_5" in res["unfit_heads"]


def test_the_correct_sign_passes():
    res = _run(OU_2_5=_head(mean_predicted=0.5918, mean_actual=0.5042,
                            calibration_gap_pct=8.76))
    assert "calibration_gap_sign" not in _checks(res)


# ───────── replay: heads that predict one class ─────────

def test_a_head_predicting_only_positives_is_caught():
    # PRE_OU_2.5, from the first real run: tn=0, fn=0, and it reported 60.2%
    # accuracy and 100% recall by calling Over on every fixture.
    res = _run(PRE_OU_2_5=_head(acc=0.6019, brier_skill=-0.005,
                                confusion_matrix={"tp": 1749, "fp": 1157, "tn": 0, "fn": 0}))
    assert "single_class_prediction" in _checks(res)
    detail = [f["detail"] for f in res["findings"]
              if f["check"] == "single_class_prediction"][0]
    assert "measuring the base rate" in detail


def test_a_head_predicting_only_negatives_is_caught():
    # PRE_OU_3.5 from the same run, the mirror image — and note it predicted
    # positive ONCE in 2,906 rows rather than never, which is no less
    # degenerate. A test for exactly zero would have missed it.
    res = _run(PRE_OU_3_5=_head(confusion_matrix={"tp": 0, "fp": 1, "tn": 1800, "fn": 1105}))
    assert "single_class_prediction" in _checks(res)


def test_a_head_calling_almost_everything_one_way_is_caught():
    # PRE_BTTS_YES: 98.2% of rows called positive against a 55% base rate.
    res = _run(PRE_BTTS_YES=_head(confusion_matrix={"tp": 1560, "fp": 1294,
                                                    "tn": 13, "fn": 39}))
    assert "single_class_prediction" in _checks(res)


def test_a_head_tracking_a_low_base_rate_is_not_flagged():
    # A rare outcome legitimately draws few positive calls. WLD_AWAY-shaped:
    # 23% predicted positive against a 22% base rate.
    res = _run(WLD_AWAY=_head(confusion_matrix={"tp": 134, "fp": 128,
                                                "tn": 774, "fn": 114}))
    assert "single_class_prediction" not in _checks(res)


def test_a_normal_confusion_matrix_passes():
    assert "single_class_prediction" not in _checks(_run(X=_head()))


# ───────── replay: no skill hiding behind accuracy ─────────

def test_a_head_with_no_skill_is_caught():
    res = _run(PRE_BTTS_YES=_head(brier_skill=-0.005, acc=0.5413,
                                  confusion_matrix={"tp": 1560, "fp": 1294,
                                                    "tn": 13, "fn": 39}))
    assert "no_skill" in _checks(res)
    assert "PRE_BTTS_YES" in res["unfit_heads"]


def test_real_skill_passes():
    assert "no_skill" not in _checks(_run(X=_head(brier_skill=0.23)))


# ───────── replay: the anchor that was not the market ─────────

def test_an_anchor_that_is_not_binding_is_caught():
    # The bug: _market_fair_priors always returned all five market keys, so
    # rows that never carried a price were anchored to a neutral 0.5. The run
    # reported p95 deviations of 49-64pp, which is arithmetically impossible
    # with the anchor coefficient fixed at 1.0.
    res = _run(OU_2_5=_head(market_anchored=True, anchor_feature="market_fair_over25",
                            deviation_from_market={"mean_abs_pp": 24.06,
                                                   "p95_abs_pp": 49.07,
                                                   "max_abs_pp": 49.92}))
    assert "anchor_not_binding" in _checks(res)
    detail = [f["detail"] for f in res["findings"]
              if f["check"] == "anchor_not_binding"][0]
    assert "not the market price" in detail


def test_a_binding_anchor_passes():
    res = _run(OU_2_5=_head(market_anchored=True, anchor_feature="market_fair_over25",
                            deviation_from_market={"mean_abs_pp": 2.4, "p95_abs_pp": 6.1,
                                                   "max_abs_pp": 9.0}))
    assert "anchor_not_binding" not in _checks(res)


def test_an_unanchored_head_is_not_judged_on_deviation():
    res = _run(OU_3_5=_head(market_anchored=False))
    assert "anchor_not_binding" not in _checks(res)


def test_an_anchored_head_must_name_its_anchor():
    res = _run(X=_head(market_anchored=True, deviation_from_market={"p95_abs_pp": 3.0}))
    assert "anchor_unnamed" in _checks(res)


def test_placeholder_rows_inside_the_anchored_set_are_caught():
    # The bug at its source rather than by its symptom: rows admitted to the
    # anchored fit whose market value sits exactly on the neutral prior.
    res = validate_training_run({
        "metrics": {},
        "market_anchoring": {"anchored": True, "placeholder_share_pct": 74.0}})
    assert "placeholder_anchor" in _checks(res)
    assert res["n_critical"] == 1


def test_a_clean_anchored_set_passes():
    res = validate_training_run({
        "metrics": {},
        "market_anchoring": {"anchored": True, "placeholder_share_pct": 0.0}})
    assert res["n_critical"] == 0


# ───────── arithmetic that cannot be true ─────────

def test_more_fixtures_than_rows_is_caught():
    res = _run(X=_head(n_train=100, n_train_matches=900))
    assert "impossible_counts" in _checks(res)


def test_a_probability_outside_zero_to_one_is_caught():
    assert "probability_out_of_range" in _checks(_run(X=_head(mean_predicted=1.4)))


# ───────── warnings do not block ─────────

def test_a_thin_training_set_warns_without_blocking():
    # The circuit breaker has its own fixture floor; blocking here too would
    # double-report the same fact.
    res = _run(WLD_HOME=_head(n_train_matches=166))
    assert "thin_training_set" in _checks(res)
    assert res["n_critical"] == 0
    assert res["unfit_heads"] == {}


# ───────── the whole run, as it actually came back ─────────

def test_the_real_run_is_rejected_for_the_right_reasons():
    res = validate_training_run({
        "market_anchoring": {"anchored": True, "placeholder_share_pct": 74.0},
        "metrics": {
            "OU_2.5": _head(market_anchored=True, anchor_feature="market_fair_over25",
                            n_train_matches=166,
                            deviation_from_market={"p95_abs_pp": 49.07}),
            "PRE_OU_2.5": _head(brier_skill=-0.005, acc=0.6019,
                                confusion_matrix={"tp": 1749, "fp": 1157,
                                                  "tn": 0, "fn": 0}),
            "WLD_AWAY": _head(market_anchored=True, anchor_feature="market_fair_away",
                              mean_predicted=0.3250, mean_actual=0.2157,
                              calibration_gap_pct=10.94,
                              deviation_from_market={"p95_abs_pp": 45.97}),
        }})
    assert res["n_critical"] >= 4
    assert set(res["unfit_heads"]) == {"OU_2.5", "PRE_OU_2.5", "WLD_AWAY"}
    assert "placeholder_anchor" in _checks(res)


def test_a_metrics_entry_that_is_not_a_head_is_ignored():
    # The metrics blob mixes per-head dicts with threshold diagnostics.
    res = validate_training_run({"metrics": {
        "OU_2.5": _head(),
        "1X2_threshold_diag": {"base_rate": 0.39, "method": "target_precision"},
        "a_scalar": 0.42}})
    assert res["n_critical"] == 0
