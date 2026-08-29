import pytest

from feature_spec import devig


def test_devig_removes_overround_on_a_two_way_market():
    # 55% + 55% implied = 10% overround. True probabilities must sum to 1.0
    # and keep the same ratio between selections.
    out = devig({"Home": 0.55, "Away": 0.55})
    assert out["Home"] == pytest.approx(0.5)
    assert out["Away"] == pytest.approx(0.5)
    assert sum(out.values()) == pytest.approx(1.0)


def test_devig_preserves_relative_odds_ratio():
    # A 2:1 favourite should stay a 2:1 favourite after de-vig.
    out = devig({"Home": 0.60, "Draw": 0.30, "Away": 0.30})
    assert out["Home"] / out["Draw"] == pytest.approx(2.0)
    assert sum(out.values()) == pytest.approx(1.0)


def test_devig_double_chance_uses_market_total_of_two():
    # Double Chance selections are not mutually exclusive (1X, X2, 12 each
    # cover two of three outcomes), so they must sum to 2.0, not 1.0.
    # Normalising to 1.0 here is the exact regression called out in
    # feature_spec.devig()'s docstring: it silently halves every fair price.
    implied = {"1X": 0.90, "X2": 0.70, "12": 0.80}
    out = devig(implied, market_total=2.0)
    assert sum(out.values()) == pytest.approx(2.0)
    # Ratios between selections are unchanged by a uniform rescale.
    assert out["1X"] / out["X2"] == pytest.approx(implied["1X"] / implied["X2"])


def test_devig_is_a_noop_when_market_is_already_fair():
    out = devig({"Home": 0.5, "Away": 0.5})
    assert out["Home"] == pytest.approx(0.5)
    assert out["Away"] == pytest.approx(0.5)


def test_devig_drops_non_positive_probabilities():
    out = devig({"Home": 0.6, "Away": 0.4, "Void": 0.0, "Bad": -0.1})
    assert "Void" not in out
    assert "Bad" not in out
    assert sum(out.values()) == pytest.approx(1.0)


def test_devig_returns_empty_for_all_zero_input():
    assert devig({"Home": 0.0, "Away": 0.0}) == {}


def test_devig_returns_empty_for_empty_input():
    assert devig({}) == {}
