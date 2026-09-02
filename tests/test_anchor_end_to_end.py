"""
The anchored path driven through the real _train_binary_head(), not a stub.

Unit tests on OffsetLogit prove the maths; this proves the wiring - that the
offset survives standardisation, C selection, Platt calibration and blob
serialisation, and that a blob built this way scores identically in main.py.
A break anywhere along that chain produces a model that trains correctly and
serves something else, which neither side's tests would catch alone.
"""
import json

import numpy as np
import pytest

import main
from train_models import _train_binary_head


class _Buf(dict):
    def set(self, k, v):
        self[k] = v

    def get_json(self, k):
        return json.loads(self[k]) if k in self else None


def _synthetic(n=1800, seed=4):
    """Outcomes driven by the market plus one real signal the market misses."""
    rng = np.random.default_rng(seed)
    market = np.clip(rng.beta(5, 5, n), 0.05, 0.95)
    signal = rng.normal(size=n)
    noise = rng.normal(size=(n, 3))
    off = np.log(market / (1 - market))
    y = (rng.random(n) < 1 / (1 + np.exp(-(off + 0.8 * signal)))).astype(int)
    X = np.column_stack([signal, noise])
    names = ["sot_diff", "cor_sum", "fouls_sum", "pos_diff"]
    return X, y, off, market, names


def _splits(n):
    tr = np.zeros(n, bool); ca = np.zeros(n, bool); te = np.zeros(n, bool)
    tr[: int(n * 0.6)] = True
    ca[int(n * 0.6): int(n * 0.8)] = True
    te[int(n * 0.8):] = True
    return tr, ca, te


def _train(anchored: bool):
    X, y, off, market, names = _synthetic()
    tr, ca, te = _splits(len(y))
    buf = _Buf()
    ok, mets, _, p_te = _train_binary_head(
        buf, X, y, tr, ca, te, names, "OU_2.5", None, {"metrics": {}},
        0.6, 5, 50.0, 90.0, 0.65, "OU_2.5",
        offset_all=off if anchored else None,
        anchor_name="market_fair_over25" if anchored else None)
    return ok, mets, buf, X, off, market, names, te, p_te


def test_the_anchored_head_trains_and_records_its_anchor():
    ok, mets, buf, *_ = _train(anchored=True)
    assert ok
    assert mets["market_anchored"] is True
    assert mets["anchor_feature"] == "market_fair_over25"
    blob = buf.get_json("model_latest:OU_2.5")
    assert blob["market_anchor"] == "market_fair_over25"


def test_the_unanchored_head_is_unchanged_and_says_so():
    ok, mets, buf, *_ = _train(anchored=False)
    assert ok
    assert mets["market_anchored"] is False
    assert "market_anchor" not in buf.get_json("model_latest:OU_2.5")


def test_the_shipped_blob_scores_identically_in_main():
    # The end the whole change rests on: what was fitted is what gets served.
    ok, mets, buf, X, off, market, names, te, p_te = _train(anchored=True)
    blob = buf.get_json("model_latest:OU_2.5")
    idx = np.flatnonzero(te)
    for i in idx[:: max(1, len(idx) // 10)]:
        feat = dict(zip(names, X[i].tolist()))
        feat["market_fair_over25"] = float(market[i])
        served = main._score_prob(feat, blob)
        assert served == pytest.approx(float(p_te[np.flatnonzero(idx == i)[0]]), abs=1e-6)


def test_the_anchored_head_reports_how_far_it_strays_from_the_market():
    _, mets, *_ = _train(anchored=True)
    dev = mets["deviation_from_market"]
    assert 0 < dev["mean_abs_pp"] < dev["p95_abs_pp"] <= dev["max_abs_pp"]


def test_anchoring_cuts_the_deviation_the_gate_would_select_from():
    # Both models see the same data. The unanchored one has to rediscover the
    # market from 4 features and cannot, so its predictions scatter - and it is
    # the profitable tail of that scatter the price gate turns into tips.
    _, _, _, X, off, market, names, te, p_anc = _train(anchored=True)
    _, _, _, _, _, _, _, te2, p_pln = _train(anchored=False)
    m_te = market[te]
    assert np.abs(p_anc - m_te).mean() < np.abs(p_pln - m_te).mean(), \
        "anchoring must reduce, not increase, distance from the market"


def _skill_less(seed=21, n=1800):
    """Outcomes generated purely by the market; the features are noise."""
    rng = np.random.default_rng(seed)
    market = np.clip(rng.beta(5, 5, n), 0.05, 0.95)
    off = np.log(market / (1 - market))
    y = (rng.random(n) < market).astype(int)
    X = rng.normal(size=(n, 3))
    tr, ca, te = _splits(n)
    ok, mets, _, p_te = _train_binary_head(
        _Buf(), X, y, tr, ca, te, ["a", "b", "c"], "OU_2.5", None, {"metrics": {}},
        0.6, 5, 50.0, 90.0, 0.65, "OU_2.5", offset_all=off,
        anchor_name="market_fair_over25")
    return ok, mets, p_te - market[te]


def test_a_skill_less_head_produces_no_SYSTEMATIC_edge():
    # The property that matters is not that the deviation is zero - three noise
    # features fitted on 1080 rows will always scatter a little - but that it
    # is CENTRED. A constant offset is the dangerous shape: it clears
    # FAIR_EDGE_MIN_BPS on one side for every fixture forever, so the model
    # tips continuously while knowing nothing.
    ok, mets, edge = _skill_less()
    assert ok
    assert abs(edge.mean()) < 0.008, "the deviation must be centred on the market"
    above, below = (edge >= 0.02).mean(), (edge <= -0.02).mean()
    assert above < 3 * max(below, 0.02), "one-sided deviation is a constant bias, not noise"


def test_the_calibration_intercept_is_not_allowed_to_invent_a_constant_edge():
    # THE REGRESSION THIS GUARDS. Platt's b is a free constant fitted on the
    # calibration split, so it absorbs whatever that split's base rate happened
    # to be - here 52.8% against a market mean of 50.9%, a ~2pp sampling
    # fluctuation on 360 rows. Applied to every future prediction that became a
    # permanent +1.9pp "edge": 43% of holdout rows cleared FAIR_EDGE_MIN_BPS
    # while the model's own weights were negligible (max 0.07).
    ok, mets, edge = _skill_less()
    assert ok
    cal = mets["calibration_fit"]
    assert abs(cal["b_fitted"]) > 0.02, "the unguarded fit really does drift"
    assert abs(cal["b"]) < abs(cal["b_fitted"]), "and shrinkage must pull it back"
    assert max(abs(v) for v in mets["feature_importance"].values()) < 0.2, \
        "weights were already near zero — b was the entire phantom edge"


def test_a_real_calibration_error_still_gets_corrected():
    # The guard must not be a blanket refusal to calibrate: a market that is
    # genuinely mis-scaled, with enough data to show it, has to come through.
    rng = np.random.default_rng(31)
    n = 6000
    market = np.clip(rng.beta(5, 5, n), 0.05, 0.95)
    off = np.log(market / (1 - market))
    # Outcomes run a full 0.5 in log-odds above what the market quotes.
    y = (rng.random(n) < 1 / (1 + np.exp(-(off + 0.5)))).astype(int)
    X = rng.normal(size=(n, 3))
    tr, ca, te = _splits(n)
    ok, mets, _, _ = _train_binary_head(
        _Buf(), X, y, tr, ca, te, ["a", "b", "c"], "OU_2.5", None, {"metrics": {}},
        0.6, 5, 50.0, 90.0, 0.65, "OU_2.5", offset_all=off,
        anchor_name="market_fair_over25")
    assert ok
    # The guard must not be a blanket refusal to calibrate. Asserted on the
    # surviving magnitude, not on closeness to the raw fit: the raw fit
    # overshoots here (0.98 against a true 0.5, because the anchored model's
    # own free intercept has already absorbed part of the offset), and
    # shrinkage lands it at ~0.50 — nearer the truth than the number it
    # shrank from.
    cal = mets["calibration_fit"]
    assert cal["b"] > 0.3, "a real, well-evidenced offset must survive"
    assert cal["b"] == pytest.approx(0.5, abs=0.25)


# ───────── the decision the driver makes ─────────

def test_the_digest_reports_what_was_trained_on_not_what_was_harvested():
    # An anchored run trains on the subset carrying real market prices. The
    # harvested total sitting alone next to an anchored model would be a
    # number that reads as sample size and is not.
    res = {"ok": True, "trained": {}, "thresholds": {},
           "data_stats": {"inplay_rows": 15806, "inplay_matches": 1729,
                          "inplay_rows_trained": 4200, "inplay_matches_trained": 460,
                          "prematch_rows": 14226},
           "metrics": {}, "market_anchoring": {"anchored": True, "anchored_rows": 4200,
                                               "anchored_matches": 460,
                                               "anchored_share_pct": 26.6}}
    sent = []
    import main as _m
    import pytest as _p
    mp = _p.MonkeyPatch()
    try:
        mp.setattr(_m, "TRAIN_ENABLE", True)
        mp.setattr(_m, "train_models", lambda: res)
        mp.setattr(_m, "send_telegram", lambda msg: sent.append(msg) or True)
        mp.setattr(_m._MODELS_CACHE, "invalidate", lambda: None)
        mp.setattr(_m._SETTINGS_CACHE, "invalidate", lambda: None)
        _m.auto_train_job()
    finally:
        mp.undo()
    msg = sent[-1]
    assert "15806" in msg, "the harvested total is still worth seeing"
    assert "trained on 4200 rows" in msg
    assert "460 matches" in msg
