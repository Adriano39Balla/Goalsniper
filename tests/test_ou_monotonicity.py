import pytest

from feature_spec import enforce_ou_monotonicity


def test_noop_when_already_non_increasing():
    # Over 1.5 >= Over 2.5 >= Over 3.5 already holds, so nothing should move.
    out = enforce_ou_monotonicity([(1.5, 0.80), (2.5, 0.50), (3.5, 0.20)])
    assert out == {1.5: pytest.approx(0.80), 2.5: pytest.approx(0.50), 3.5: pytest.approx(0.20)}


def test_two_point_crossing_is_averaged():
    # Over 3.5 (0.55) can never be more likely than Over 2.5 (0.40) - it's a
    # strict subset. The isotonic projection pulls both to their mean.
    out = enforce_ou_monotonicity([(2.5, 0.40), (3.5, 0.55)])
    assert out[2.5] == pytest.approx(0.475)
    assert out[3.5] == pytest.approx(0.475)


def test_crossing_is_detected_regardless_of_input_order():
    # The function must sort by line itself; a caller handing in unsorted
    # pairs must get the identical projection as the sorted case above.
    out = enforce_ou_monotonicity([(3.5, 0.55), (2.5, 0.40)])
    assert out[2.5] == pytest.approx(0.475)
    assert out[3.5] == pytest.approx(0.475)


def test_violation_only_pools_the_offending_block():
    # 1.5=0.80, 2.5=0.50, 3.5=0.60: only 2.5/3.5 cross (0.50 < 0.60). 1.5
    # stays untouched; 2.5 and 3.5 are pooled to their mean (0.55).
    out = enforce_ou_monotonicity([(1.5, 0.80), (2.5, 0.50), (3.5, 0.60)])
    assert out[1.5] == pytest.approx(0.80)
    assert out[2.5] == pytest.approx(0.55)
    assert out[3.5] == pytest.approx(0.55)


def test_cascading_violation_merges_backward():
    # 1.5=0.30, 2.5=0.50, 3.5=0.45: merging (2.5,3.5) first gives 0.475, which
    # is still above 1.5's 0.30 and itself violates monotonicity, so the
    # merge must cascade back and pool all three lines together.
    out = enforce_ou_monotonicity([(1.5, 0.30), (2.5, 0.50), (3.5, 0.45)])
    expected = (0.30 + 0.50 + 0.45) / 3
    assert out[1.5] == pytest.approx(expected)
    assert out[2.5] == pytest.approx(expected)
    assert out[3.5] == pytest.approx(expected)


def test_single_point_is_returned_unchanged():
    assert enforce_ou_monotonicity([(2.5, 0.35)]) == {2.5: 0.35}


def test_empty_input_returns_empty_dict():
    assert enforce_ou_monotonicity([]) == {}
