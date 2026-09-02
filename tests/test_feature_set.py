"""
Choosing between the full and reduced in-play feature sets.

FEATURES has 56 columns fitted against ~1,700 independent fixtures - about 15
events per feature, the bottom of the range where a logistic fit is stable at
all. Many of the 56 are algebraic transforms of one another, and collinear
features do not add information: they add variance. The fit splits weight
between them close to arbitrarily, the split moves on every retrain, and that
movement is the deviation-from-market the price gate sells as edge.

CORE_FEATURES is a HYPOTHESIS about which columns are restatements. It is not
trusted on the strength of that argument: the calibration split - already where
C is chosen - picks between them every night, and the full set wins ties. A
wrong cut therefore costs a line in the digest rather than a model.
"""
import numpy as np
import pandas as pd
import pytest

import feature_spec
from feature_spec import CORE_FEATURES, FEATURES
from train_models import FEATURE_SET, collinearity_report, select_feature_set


# ───────── the reduced set is well-formed ─────────

def test_core_is_a_strict_subset_of_the_full_set():
    assert set(CORE_FEATURES) <= set(FEATURES)
    assert len(CORE_FEATURES) < len(FEATURES)
    assert len(set(CORE_FEATURES)) == len(CORE_FEATURES), "no duplicates"


def test_the_settled_question_flags_are_gone():
    # is_goalfest is literally "Over 2.5 has already happened" and score_margin
    # /is_leading_* are deterministic functions of (goals_sum, goals_diff),
    # which are both kept. Free accuracy on rows nobody can bet.
    for f in ("is_goalfest", "score_margin", "is_leading_h", "is_leading_a"):
        assert f in FEATURES
        assert f not in CORE_FEATURES


def test_the_features_that_explode_on_missing_data_are_gone():
    # sot_xg_ratio = sot / max(xg, MIN_XG_DENOM). A missing xG feed arrives as
    # 0.0 — we now know that happens, and silently — so this goes unbounded
    # exactly when the data is absent.
    assert "sot_xg_ratio_h" not in CORE_FEATURES
    assert "sot_xg_ratio_a" not in CORE_FEATURES


def test_the_market_priors_and_league_rates_all_survive():
    # These are the calibration anchors, not candidates for pruning.
    for f in list(feature_spec.NEUTRAL_MARKET_PRIORS) + [
            "league_btts_rate", "league_ov25_rate", "league_ov35_rate"]:
        assert f in CORE_FEATURES


def test_a_subset_relationship_is_kept_over_its_own_subset():
    # game_control contains 0.4*(pos/100)*xg as one of three terms, so
    # possession_xg_interaction is a piece of it. The broader one survives.
    assert "game_control_h" in CORE_FEATURES
    assert "possession_xg_interaction_h" not in CORE_FEATURES


# ───────── the selection ─────────

def _frame(n=900, seed=0, signal_cols=("a",)):
    rng = np.random.default_rng(seed)
    cols = ["a", "b", "c", "d", "e", "f"]
    df = pd.DataFrame({c: rng.normal(size=n) for c in cols})
    z = sum(1.2 * df[c].to_numpy() for c in signal_cols)
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    m_tr = np.zeros(n, bool); m_tr[: int(n * 0.7)] = True
    m_ca = ~m_tr
    return df, y, m_tr, m_ca


def test_the_smaller_set_wins_when_the_extra_columns_are_noise():
    df, y, m_tr, m_ca = _frame()
    chosen, cols, diag = select_feature_set(
        df, y, m_tr, m_ca,
        {"full": ["a", "b", "c", "d", "e", "f"], "core": ["a", "b"]})
    assert chosen == "core"
    assert cols == ["a", "b"]
    assert diag["improvement_vs_full"] > 0


def test_the_full_set_wins_when_the_extra_columns_carry_signal():
    df, y, m_tr, m_ca = _frame(signal_cols=("a", "e", "f"))
    chosen, _cols, _diag = select_feature_set(
        df, y, m_tr, m_ca,
        {"full": ["a", "b", "c", "d", "e", "f"], "core": ["a", "b"]})
    assert chosen == "full", "a cut that loses real signal must not be taken"


def test_a_tie_goes_to_the_full_set():
    # The reduced set has to EARN the swap, so an unclear result leaves today's
    # behaviour in place.
    df, y, m_tr, m_ca = _frame()
    chosen, cols, _ = select_feature_set(
        df, y, m_tr, m_ca, {"full": ["a", "b"], "core": ["a", "b"]})
    assert chosen == "full"


def test_a_broken_candidate_degrades_to_the_full_set():
    df, y, m_tr, m_ca = _frame()
    chosen, cols, diag = select_feature_set(
        df, y, m_tr, m_ca,
        {"full": ["a", "b"], "core": ["a", "does_not_exist"]})
    assert chosen == "full"
    assert diag["cal_logloss"]["core"] is None


def test_both_candidates_failing_still_returns_the_full_set():
    df, y, m_tr, m_ca = _frame()
    chosen, cols, diag = select_feature_set(
        df, y, m_tr, m_ca, {"full": ["nope"], "core": ["also_nope"]})
    assert chosen == "full" and cols == ["nope"]
    assert "no candidate" in diag["reason"]


def test_the_comparison_records_both_scores_for_the_operator():
    df, y, m_tr, m_ca = _frame()
    _, _, diag = select_feature_set(
        df, y, m_tr, m_ca, {"full": ["a", "b", "c"], "core": ["a"]})
    assert set(diag["cal_logloss"]) == {"full", "core"}
    assert diag["n_features"] == {"full": 3, "core": 1}


def test_the_default_compares_rather_than_forcing_either_set():
    assert FEATURE_SET == "auto"


# ───────── collinearity, measured rather than argued ─────────

def test_it_finds_an_exact_restatement():
    n = 300
    rng = np.random.default_rng(1)
    a = rng.normal(size=n)
    df = pd.DataFrame({"a": a, "twice_a": 2 * a + 1e-9, "unrelated": rng.normal(size=n)})
    rep = collinearity_report(df, ["a", "twice_a", "unrelated"])
    assert rep["max_abs_corr"] == pytest.approx(1.0, abs=1e-3)
    assert rep["pairs_above_0.95"] == 1
    top = rep["most_collinear"][0]
    assert {top["a"], top["b"]} == {"a", "twice_a"}


def test_an_independent_set_reports_no_redundancy():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({c: rng.normal(size=800) for c in "abcd"})
    rep = collinearity_report(df, list("abcd"))
    assert rep["pairs_above_0.95"] == 0
    assert rep["max_abs_corr"] < 0.3


def test_a_constant_column_does_not_break_the_measurement():
    # league_*_rate is constant whenever the training set is one league.
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"a": rng.normal(size=100), "flat": np.ones(100)})
    rep = collinearity_report(df, ["a", "flat"])
    assert rep.get("note") == "no varying columns" or rep["n_features"] == 2
