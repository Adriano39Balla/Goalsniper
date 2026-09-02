"""
Measuring how much of a head's apparent skill answers settled questions.

An in-play snapshot at 2-1 has already answered "Over 2.5?" and "both teams
to score?". Those rows are not predictions: the label is a fact about a
scoreline the features already contain. They are free accuracy, they
dominate calibration, and none of them is bettable - no market prices an
outcome that has resolved.

The fingerprint is in the fitted weights: OU_2.5's two strongest features
are goals_sum and is_goalfest (goals_sum >= 3, i.e. "Over 2.5 already
happened"), and BTTS_YES leans on goals_sum harder than anything else.
"""
import numpy as np
import pandas as pd
import pytest

from train_models import already_decided_mask, decided_diagnostics


def _df(rows):
    """rows: list of (goals_h, goals_a) at snapshot time."""
    return pd.DataFrame([{"goals_sum": h + a, "score_margin": abs(h - a)} for h, a in rows])


# ───────── which rows are settled ─────────

def test_over_25_is_settled_once_three_goals_are_in():
    df = _df([(2, 1), (1, 1), (3, 0), (0, 0)])
    assert list(already_decided_mask(df, "OU_2.5")) == [True, False, True, False]


def test_over_35_needs_a_fourth_goal():
    # The same 3-goal snapshot settles Over 2.5 but not Over 3.5.
    df = _df([(2, 1), (2, 2)])
    assert list(already_decided_mask(df, "OU_2.5")) == [True, True]
    assert list(already_decided_mask(df, "OU_3.5")) == [False, True]


def test_btts_is_settled_once_both_sides_have_scored():
    # 3-0 has more goals than 1-1 but has NOT settled BTTS.
    df = _df([(1, 1), (3, 0), (2, 1), (0, 0), (0, 2)])
    assert list(already_decided_mask(df, "BTTS_YES")) == [True, False, True, False, False]


def test_match_result_heads_are_never_settled_early():
    # A side can always score, so these stay mathematically open to the
    # final whistle however lopsided the score.
    df = _df([(4, 0), (0, 3)])
    for head in ("WLD_HOME", "WLD_DRAW", "WLD_AWAY", "1X2", "DNB"):
        assert already_decided_mask(df, head) is None


def test_an_unparseable_head_name_is_ignored_rather_than_guessed():
    assert already_decided_mask(_df([(1, 1)]), "OU_bogus") is None


def test_missing_columns_degrade_to_no_measurement():
    assert already_decided_mask(pd.DataFrame({"minute": [30.0]}), "OU_2.5") is None
    assert already_decided_mask(pd.DataFrame({"goals_sum": [3.0]}), "BTTS_YES") is None


# ───────── what the diagnostic reports ─────────

def test_it_reports_the_share_and_the_honest_base_rate():
    # Four rows: two already over 2.5 (settled, label 1), two goalless
    # snapshots of which one ends over.
    df = _df([(2, 1), (3, 0), (0, 0), (0, 0)])
    y = np.array([1, 1, 1, 0])
    dd = decided_diagnostics(df, "OU_2.5", y)

    assert dd["n_rows"] == 4
    assert dd["n_already_decided"] == 2
    assert dd["decided_share_pct"] == 50.0
    # Two of the three positives were free.
    assert dd["share_of_positives_pct"] == pytest.approx(66.7, abs=0.1)
    # The honest benchmark: 0.75 overall flatters a head that faces 0.50.
    assert dd["base_rate_all"] == 0.75
    assert dd["base_rate_undecided"] == 0.5


def test_no_settled_rows_reports_zero_rather_than_nothing():
    df = _df([(0, 0), (1, 0)])
    dd = decided_diagnostics(df, "OU_2.5", np.array([0, 1]))
    assert dd["decided_share_pct"] == 0.0
    assert dd["base_rate_undecided"] == dd["base_rate_all"]


def test_every_row_settled_leaves_no_undecided_benchmark():
    df = _df([(2, 1), (3, 1)])
    dd = decided_diagnostics(df, "OU_2.5", np.array([1, 1]))
    assert dd["decided_share_pct"] == 100.0
    assert dd["base_rate_undecided"] is None


def test_heads_that_cannot_settle_early_report_nothing():
    assert decided_diagnostics(_df([(4, 0)]), "WLD_HOME", np.array([1])) is None


def test_a_settled_row_is_always_a_positive_label_in_practice():
    # Sanity on the premise: if the mask says Over 2.5 is settled, the final
    # total cannot come back under it.
    df = _df([(2, 1), (4, 0)])
    mask = already_decided_mask(df, "OU_2.5")
    finals = np.array([1, 1])
    assert finals[mask].all()
