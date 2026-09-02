"""
Anchoring the model to the market price.

Without an anchor, market_fair_over25 is one of 55 features and the L2 penalty
shrinks it like any other, so the model is free to wander away from the
market's price and call the distance an edge. The price gate then selects
whichever candidates wandered furthest in the profitable direction - which is
to say, it selects the model's own largest errors.

A model with NO skill, just unbiased noise around the market, produces tips
claiming 8-12pp of edge that way, and nothing in the tip's own numbers
distinguishes that from real skill. The four live tips sent so far claimed
+4.6% to +7.5pp against a sanity cap of 8pp - they cluster at the cap, which is
the signature of that selection effect rather than of edge.

Anchoring puts the market's log-odds into the linear predictor with a
coefficient FIXED at 1.0, and removes the anchor from the feature matrix so it
cannot also be fitted. The weights can then only express a deviation from the
market. A head with nothing to say converges to the market instead of to noise.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

import feature_spec
import main
from feature_spec import MARKET_ANCHOR, anchor_logit
from train_models import (MIN_ANCHORED_MATCHES, MIN_ANCHORED_ROWS, OffsetLogit,
                          build_model_blob, frame_anchor_mask, frame_anchor_report)


# ───────── the fitter ─────────

def _data(n=600, d=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w = rng.normal(size=d)
    y = (rng.random(n) < 1 / (1 + np.exp(-(X @ w - 0.3)))).astype(int)
    return X, y


@pytest.mark.parametrize("C", [0.01, 0.1, 1.0, 10.0])
def test_with_a_zero_offset_it_reproduces_sklearn(C):
    # The objective must match LogisticRegression's lbfgs solver exactly -
    # 0.5*||w||^2 penalised, intercept unpenalised, C scaling the data term -
    # or C_GRID means something different on anchored and unanchored heads and
    # the two paths stop being comparable.
    X, y = _data()
    sk = LogisticRegression(C=C, max_iter=5000, solver="lbfgs").fit(X, y)
    of = OffsetLogit(C).fit(X, y, np.zeros(len(y)))
    assert np.abs(sk.coef_.ravel() - of.coef_.ravel()).max() < 1e-3
    assert abs(sk.intercept_[0] - of.intercept_[0]) < 1e-3


def test_the_offset_coefficient_is_pinned_not_fitted():
    # Fitted freely the market's log-odds attract a coefficient that shrinks
    # under the penalty. Pinning it at 1.0 is the whole mechanism.
    rng = np.random.default_rng(3)
    X, _ = _data(seed=3)
    off = rng.normal(0, 1.5, len(X))
    w = rng.normal(size=X.shape[1])
    y = (rng.random(len(X)) < 1 / (1 + np.exp(-(off + X @ w - 0.3)))).astype(int)

    pinned = OffsetLogit(1.0).fit(X, y, off)
    free = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs").fit(
        np.column_stack([X, off]), y)

    assert free.coef_.ravel()[-1] != pytest.approx(1.0, abs=1e-6)
    # The pinned model recovers the same real-feature weights.
    assert np.abs(free.coef_.ravel()[:X.shape[1]] - pinned.coef_.ravel()).max() < 0.15


def test_a_head_with_no_signal_converges_to_the_market_not_to_noise():
    # THE POINT OF THE WHOLE CHANGE. Outcomes are generated purely by the
    # market price; the features are unrelated noise. An anchored head should
    # sit on the market. An unanchored one invents a large opinion, and it is
    # the upper tail of that opinion the price gate would select.
    rng = np.random.default_rng(11)
    n = 1500
    off = rng.normal(0, 1.2, n)
    market = 1 / (1 + np.exp(-off))
    y = (rng.random(n) < market).astype(int)
    noise = rng.normal(size=(n, 6))

    anchored = OffsetLogit(0.1).fit(noise, y, off).predict_proba_off(noise, off)
    plain = LogisticRegression(C=0.1, max_iter=5000, solver="lbfgs").fit(
        noise, y).predict_proba(noise)[:, 1]

    dev_anchored = np.abs(anchored - market).mean()
    dev_plain = np.abs(plain - market).mean()
    assert dev_anchored < 0.05, "an anchored head must stay near the market it was given"
    assert dev_plain > 4 * dev_anchored, "the unanchored head is the noise source"


def test_it_still_learns_a_real_deviation_when_one_exists():
    # Anchoring must not be a straitjacket: a genuine signal the market has
    # missed still has to come through, or the model can never earn anything.
    rng = np.random.default_rng(5)
    n = 2000
    off = rng.normal(0, 1.0, n)
    x = rng.normal(size=(n, 1))
    true_dev = 1.4 * x.ravel()
    y = (rng.random(n) < 1 / (1 + np.exp(-(off + true_dev)))).astype(int)
    m = OffsetLogit(1.0).fit(x, y, off)
    assert m.coef_.ravel()[0] == pytest.approx(1.4, abs=0.25)


# ───────── train/serve parity ─────────

def test_serving_reproduces_the_training_linear_predictor_exactly():
    # A mismatch here means the deployed model silently predicts something
    # other than what was fitted, which no test of either side alone catches.
    # Serving keeps the two halves apart - _linpred() is the model's deviation,
    # _anchor_offset() the market's log-odds - because Platt scaling is applied
    # to the first and never the second. Their sum is the training linear
    # predictor.
    rng = np.random.default_rng(9)
    feats = ["sot_sum", "xg_sum", "minute"]
    n = 400
    X = rng.normal(size=(n, len(feats)))
    off = rng.normal(0, 1.0, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(off + X[:, 0] * 0.6)))).astype(int)

    mean, scale = X.mean(axis=0), X.std(axis=0)
    Z = (X - mean) / scale
    m = OffsetLogit(1.0).fit(Z, y, off)
    blob = build_model_blob(m, feats, mean, scale, (1.0, 0.0), 1.0,
                            market_anchor="market_fair_over25")

    assert blob["market_anchor"] == "market_fair_over25"
    for i in (0, 7, 123, 399):
        market_p = float(1 / (1 + np.exp(-off[i])))
        feat = dict(zip(feats, X[i].tolist()))
        feat["market_fair_over25"] = market_p
        assert main._linpred(feat, blob) + main._anchor_offset(feat, blob) == pytest.approx(
            float(m.decision_function(Z[i:i + 1], off[i:i + 1])[0]), abs=1e-9)
        # The deviation on its own carries no market information.
        assert main._linpred(feat, blob) == pytest.approx(
            float(m.decision_function(Z[i:i + 1], np.zeros(1))[0]), abs=1e-9)


def test_an_unanchored_blob_is_served_exactly_as_before():
    # Old blobs carry no market_anchor key and must be untouched by this.
    blob = {"intercept": 0.25, "weights": {"sot_sum": 0.4},
            "scaler": {"mean": {"sot_sum": 2.0}, "scale": {"sot_sum": 1.0}}}
    assert main._linpred({"sot_sum": 3.0, "market_fair_over25": 0.9}, blob) == pytest.approx(0.65)


def test_a_missing_market_at_serving_time_falls_back_to_the_neutral_prior():
    # Not to zero: anchor_logit(0.0) is -13.8, which would read as "the market
    # says impossible" and drag every prediction to the floor.
    blob = {"intercept": 0.0, "weights": {}, "market_anchor": "market_fair_over25"}
    assert main._anchor_offset({}, blob) == pytest.approx(
        anchor_logit(feature_spec.NEUTRAL_MARKET_PRIORS["market_fair_over25"]))


def test_platt_scaling_never_touches_the_market_offset():
    # An `a` of 1.1 applied to the sum would multiply the market's log-odds
    # too, restoring it as a fitted feature - the exact behaviour anchoring
    # removes. The offset must pass through the calibration untouched.
    blob = {"intercept": 0.0, "weights": {}, "market_anchor": "market_fair_over25",
            "calibration": {"method": "platt", "a": 1.4, "b": 0.3}}
    feat = {"market_fair_over25": 0.70}
    offset = anchor_logit(0.70)
    # b applies to the deviation, which is 0 here, so only b survives.
    assert main._score_prob(feat, blob) == pytest.approx(main._sigmoid(offset + 0.3))
    # a multiplies the deviation alone.
    blob["weights"] = {"sot_sum": 0.5}
    feat["sot_sum"] = 2.0
    assert main._score_prob(feat, blob) == pytest.approx(
        main._sigmoid(offset + 1.4 * (0.5 * 2.0) + 0.3))


def test_an_unanchored_model_calibrates_exactly_as_it_used_to():
    # logit(sigmoid(z)) == z, so the new form reduces to the old one when the
    # offset is zero. Old blobs must not shift by a hair.
    blob = {"intercept": -0.2, "weights": {"sot_sum": 0.5},
            "calibration": {"method": "platt", "a": 1.3, "b": -0.1}}
    feat = {"sot_sum": 3.0}
    linpred = -0.2 + 0.5 * 3.0
    assert main._score_prob(feat, blob) == pytest.approx(
        main._sigmoid(1.3 * linpred - 0.1))


def test_both_sides_take_the_offset_from_the_same_function():
    # Two copies of this clip would be one more pair of constants to drift
    # apart, which is the mistake feature_spec exists to prevent.
    assert main.anchor_logit is anchor_logit
    assert anchor_logit(0.5) == 0.0
    assert anchor_logit(0.0) == pytest.approx(-anchor_logit(1.0))


# ───────── which heads get anchored ─────────

def test_every_anchored_head_names_a_real_market_feature():
    for head, feat in MARKET_ANCHOR.items():
        assert feat in feature_spec.FEATURES
        assert feat in feature_spec.NEUTRAL_MARKET_PRIORS


def test_over_35_is_deliberately_not_anchored():
    # Only the 2.5 line is quoted in the features. Anchoring Over 3.5 to the
    # Over 2.5 price would anchor it to a different question.
    assert "OU_3.5" not in MARKET_ANCHOR
    assert "OU_2.5" in MARKET_ANCHOR


# ───────── eligibility ─────────

def _frame(n_anchored, n_plain, matches_per=10):
    rows = []
    mid = 0
    flags = [f"_has_{f}" for f in set(MARKET_ANCHOR.values())]
    for i in range(n_anchored):
        if i % matches_per == 0:
            mid += 1
        rows.append({"_match_id": mid, **{f: 1 for f in flags}})
    for i in range(n_plain):
        if i % matches_per == 0:
            mid += 1
        rows.append({"_match_id": mid, **{f: 0 for f in flags}})
    return pd.DataFrame(rows)


def test_only_rows_with_a_real_market_price_count():
    df = _frame(n_anchored=30, n_plain=70)
    mask = frame_anchor_mask(df)
    assert int(mask.sum()) == 30


def test_a_neutral_prior_is_not_mistaken_for_a_market_price():
    # build_inplay_features() fills missing market probabilities with
    # NEUTRAL_MARKET_PRIORS, after which 0.5 is indistinguishable from a
    # genuine 50/50 quote in the matrix. Presence is recorded at load time
    # precisely so this question is not asked of the number.
    df = _frame(n_anchored=0, n_plain=50)
    df["market_fair_over25"] = 0.5
    assert int(frame_anchor_mask(df).sum()) == 0


def test_a_frame_that_cannot_answer_the_question_is_not_anchored():
    # Prematch frames and legacy loaders carry no _has_* flags. Reading that
    # as "all rows anchored" would anchor the model to neutral priors.
    rep = frame_anchor_report(pd.DataFrame({"_match_id": [1, 2], "minute": [30.0, 40.0]}))
    assert rep["anchored"] is False
    assert "do not record" in rep["reason"]


def test_too_little_anchored_data_falls_back_rather_than_anchoring_thinly():
    rep = frame_anchor_report(_frame(n_anchored=50, n_plain=5000))
    assert rep["anchored"] is False
    assert rep["anchored_rows"] == 50
    assert str(MIN_ANCHORED_ROWS) in rep["reason"]


def test_the_fallback_reason_explains_why_the_count_starts_near_zero():
    # Odds recorded before the market-name fix are excluded, so this number is
    # near zero on the day the change ships and grows from there. Without that
    # sentence the digest looks like a bug.
    rep = frame_anchor_report(_frame(n_anchored=10, n_plain=100))
    assert "before the market-name fix" in rep["reason"]


def test_enough_anchored_data_switches_the_regime():
    rep = frame_anchor_report(_frame(n_anchored=MIN_ANCHORED_ROWS + 10,
                                     n_plain=100, matches_per=5))
    assert rep["anchored"] is True
    assert rep["anchored_matches"] >= MIN_ANCHORED_MATCHES
    assert rep["anchored_share_pct"] > 90


# ───────── the calibration guards ─────────
# Platt scaling is the last place a skill-less anchored head can still
# manufacture edge, and both parameters can do it in a different way.

def _platt(y, dev, off):
    from train_models import fit_platt_anchored
    return fit_platt_anchored(np.asarray(y, float), np.asarray(dev, float),
                              np.asarray(off, float))


def test_a_constant_shift_from_sampling_noise_is_shrunk_to_near_zero():
    # b is a free constant that absorbs whatever the calibration split's base
    # rate happened to be. On an anchored head that becomes a permanent edge
    # over the market on every future prediction. Poorly measured, it collapses.
    rng = np.random.default_rng(2)
    n = 400
    off = rng.normal(0, 1.0, n)
    dev = rng.normal(0, 0.05, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-off))).astype(int)
    a, b, diag = _platt(y, dev, off)
    # Judged against the bar that matters: FAIR_EDGE_MIN_BPS is 2pp, and near
    # even money a log-odds shift of x is roughly x/4 in probability. The raw
    # fit here is -0.146 (a 3.6pp permanent shift, past the gate on its own);
    # shrunk it must land well inside it.
    assert abs(b) / 4 < 0.015, f"a {abs(b)/4*100:.1f}pp permanent shift is too much"
    assert abs(b) < 0.5 * abs(diag["b_fitted"]), "noise must lose most of its weight"


def test_a_large_well_evidenced_shift_survives():
    # The guard must not be a blanket refusal to calibrate.
    rng = np.random.default_rng(2)
    n = 8000
    off = rng.normal(0, 1.0, n)
    dev = rng.normal(0, 0.05, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(off + 0.6)))).astype(int)
    a, b, diag = _platt(y, dev, off)
    assert b == pytest.approx(0.6, abs=0.15)
    assert "b" not in (diag.get("held") or "")


def test_an_anti_predictive_slope_collapses_to_the_market():
    # A negative slope makes the model bet against its own signal. On a
    # calibration split that is noise essentially every time; a genuinely
    # inverted feature set would be a bug to fix, not an edge to trade.
    rng = np.random.default_rng(8)
    n = 3000
    off = rng.normal(0, 1.0, n)
    dev = rng.normal(0, 1.0, n)
    # Outcomes move OPPOSITE to the deviation.
    y = (rng.random(n) < 1 / (1 + np.exp(-(off - 1.2 * dev)))).astype(int)
    a, b, diag = _platt(y, dev, off)
    assert a == 0.0, "an inverted head must predict the market, not the inverse"
    assert diag["slope_was_negative"] is True


def test_the_slope_shrinks_toward_the_market_not_toward_full_trust():
    # THE CORRECTION THIS ENCODES. a=1 means "trust the model's own scale
    # completely" and is the LEAST conservative setting; a=0 collapses onto the
    # market. An earlier version held a at 1.0 whenever it was not significantly
    # different from 1.0, which measurably RAISED the deviation the price gate
    # sells as edge — the unguarded fit had been supplying useful shrinkage and
    # the guard removed it.
    rng = np.random.default_rng(12)
    n = 250
    off = rng.normal(0, 1.0, n)
    dev = rng.normal(0, 0.3, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-off))).astype(int)
    a, b, diag = _platt(y, dev, off)
    assert 0.0 <= a < 0.5, "a barely-measured slope must move toward the market"


def test_shrinkage_is_continuous_in_the_evidence():
    # No threshold to sit on the wrong side of. Asserted on the shrinkage
    # FACTOR rather than on the slope itself: the fitted slope has its own
    # sampling noise, so the product need not be monotone even when the
    # evidence weight is.
    factors, last = [], None
    for n in (300, 1200, 5000, 20000):
        rng = np.random.default_rng(3)
        off = rng.normal(0, 1.0, n)
        dev = rng.normal(0, 1.0, n)
        y = (rng.random(n) < 1 / (1 + np.exp(-(off + 1.0 * dev)))).astype(int)
        a, _b, d = _platt(y, dev, off)
        factors.append(round(a / d["a_fitted"], 6))
        last = a
    assert factors == sorted(factors), f"evidence weight must rise with n: {factors}"
    assert factors[0] < 0.99 < factors[-1]
    assert last == pytest.approx(1.0, abs=0.15), "and it ends at the fitted value"


def test_the_constant_shift_faces_a_tighter_prior_than_the_slope():
    # b claims the de-vigged market is wrong by a constant on EVERY fixture,
    # forever. a only claims the model's own signal is real at the scale
    # fitted. The stronger claim is held to more evidence.
    from train_models import CAL_PRIOR_K_SHIFT, CAL_PRIOR_K_SLOPE
    assert CAL_PRIOR_K_SHIFT > CAL_PRIOR_K_SLOPE
    # At the same t-statistic the shift keeps strictly less of its fitted value.
    for t in (0.5, 1.0, 1.5, 2.0, 3.0):
        fa = t ** 2 / (CAL_PRIOR_K_SLOPE + t ** 2)
        fb = t ** 2 / (CAL_PRIOR_K_SHIFT + t ** 2)
        assert fb < fa


def test_a_real_signal_keeps_its_slope():
    rng = np.random.default_rng(8)
    n = 3000
    off = rng.normal(0, 1.0, n)
    dev = rng.normal(0, 1.0, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(off + 1.2 * dev)))).astype(int)
    a, b, diag = _platt(y, dev, off)
    assert a == pytest.approx(1.2, abs=0.2)
    assert "a" not in (diag.get("held") or "")


def test_the_guard_reports_what_it_did_rather_than_acting_silently():
    rng = np.random.default_rng(4)
    n = 300
    off = rng.normal(0, 1.0, n)
    _, _, diag = _platt((rng.random(n) < 0.5).astype(int), rng.normal(0, 0.05, n), off)
    for k in ("a_fitted", "b_fitted", "se_a", "se_b", "a", "b"):
        assert k in diag, f"{k} must be visible for the operator to judge the fit"
