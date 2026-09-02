"""
One match, one observation.

Nine snapshots of a fixture share a single outcome. They are not nine
independent observations, but the fit counted them as such, so the objective
saw an effective sample ~9x larger than the data contains and selected C as if
regularisation mattered ~9x less than it does. Systematic under-regularisation
is how the model acquires the wide deviation from the market that the price
gate then sells as edge.

The second, quieter bias: a fixture harvested for 70 minutes carried twice the
weight of one harvested for 35, for no reason connected to how much either
tells us.
"""
import numpy as np
import pytest

from train_models import (ROW_WEIGHTING, _standardize, effective_n,
                          match_weights)


def _ids(*counts):
    return np.concatenate([np.full(c, i) for i, c in enumerate(counts)])


def test_every_fixture_carries_the_same_total_weight():
    ids = _ids(9, 9, 3)
    w = match_weights(ids)
    totals = [w[ids == m].sum() for m in (0, 1, 2)]
    assert totals == pytest.approx([totals[0]] * 3)


def test_a_long_harvest_does_not_outvote_a_short_one():
    # THE BIAS THIS REMOVES. Unweighted, the 9-snapshot fixture had three times
    # the say of the 3-snapshot one purely for having been watched longer.
    ids = _ids(9, 3)
    w = match_weights(ids)
    assert w[ids == 0].sum() == pytest.approx(w[ids == 1].sum())
    assert w[ids == 0][0] < w[ids == 1][0], "per-row weight must fall as snapshots rise"


def test_the_weights_sum_to_the_fixture_count():
    # C scales the data term, so the sum decides how strong the L2 penalty is
    # relative to the likelihood. Summing to fixtures rather than rows keeps
    # C's meaning "per observation" while making an observation a fixture, so
    # the existing C_GRID still spans the useful range.
    ids = _ids(9, 9, 3, 7)
    assert match_weights(ids).sum() == pytest.approx(4.0)


def test_equal_snapshot_counts_reduce_to_uniform_weights():
    # 3 fixtures x 5 snapshots: every row weighs 1/5, which already sums to 3.
    assert match_weights(_ids(5, 5, 5)) == pytest.approx(np.full(15, 0.2))


def test_a_single_snapshot_per_match_is_a_no_op():
    ids = _ids(1, 1, 1, 1)
    assert match_weights(ids) == pytest.approx(np.ones(4))


# ───────── the sample size actually reported ─────────

def test_effective_n_counts_fixtures_not_rows():
    assert effective_n(_ids(9, 9, 3), 21) == 3


def test_it_is_not_the_kish_effective_sample_size():
    # Kish measures the efficiency loss from UNEQUAL weights among rows and is
    # blind to clustering: on these 21 rows across 3 fixtures it returns 16.2,
    # a number that reads as sample size and is not. Every row of a fixture
    # shares one outcome, so the answer is the fixture count.
    ids = _ids(9, 9, 3)
    w = match_weights(ids)
    kish = w.sum() ** 2 / (w ** 2).sum()
    assert kish > 15
    assert effective_n(ids, len(ids)) == 3


def test_no_ids_degrades_to_the_row_count():
    assert effective_n(None, 500) == 500


# ───────── the scaler moves with the weights ─────────

def test_standardisation_is_weighted_too():
    # L2 penalises on the standardised scale, so an unweighted scaler would
    # make the penalty itself depend on snapshot counts.
    X = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [10.0]])
    ids = _ids(9, 1)
    plain_mean, _ = _standardize(X)
    w_mean, _ = _standardize(X, match_weights(ids))
    assert plain_mean[0] == pytest.approx(1.0), "unweighted: the long fixture dominates"
    assert w_mean[0] == pytest.approx(5.0), "weighted: one fixture, one vote"


def test_a_constant_column_still_survives_standardisation():
    X = np.array([[3.0], [3.0], [3.0], [3.0]])
    _, scale = _standardize(X, match_weights(_ids(3, 1)))
    assert scale[0] == 1.0, "a zero scale would divide by zero at serving time"


def test_the_default_is_per_match_and_it_is_switchable():
    # The argument for this is statistical rather than empirical — synthetic
    # data could not settle it either way — so it is reported and reversible.
    import os
    assert ROW_WEIGHTING == "per_match"
    assert "ROW_WEIGHTING" not in os.environ
