import pytest

from feature_spec import kelly_fraction

import main


def test_kelly_fraction_positive_edge():
    # b = odds - 1 = 1.0; f* = (p*b - (1-p)) / b = (0.6 - 0.4) / 1 = 0.2
    assert kelly_fraction(prob=0.6, odds=2.0) == pytest.approx(0.2)


def test_kelly_fraction_is_zero_at_fair_odds():
    # No edge over the break-even price -> stake nothing.
    assert kelly_fraction(prob=0.5, odds=2.0) == pytest.approx(0.0)


def test_kelly_fraction_negative_when_odds_dont_cover_probability():
    # Model thinks 40% but the price only pays as if it's 50% -> negative
    # Kelly, i.e. "don't bet", not "bet a small amount".
    assert kelly_fraction(prob=0.4, odds=2.0) < 0


def test_kelly_fraction_zero_for_odds_at_or_below_evens():
    # b = odds - 1 <= 0 is guarded explicitly to avoid a division by
    # zero/sign flip, regardless of how large the model's edge looks.
    assert kelly_fraction(prob=0.99, odds=1.0) == 0.0
    assert kelly_fraction(prob=0.99, odds=0.5) == 0.0


def test_stake_units_returns_none_without_odds():
    assert main._stake_units(prob=0.9, odds=None) is None
    assert main._stake_units(prob=0.9, odds=0) is None


def test_stake_units_floors_negative_kelly_to_zero():
    # A losing edge must never produce a negative or None stake - it's a
    # real "stake nothing" decision, distinct from "no odds available".
    assert main._stake_units(prob=0.4, odds=2.0) == 0.0


def test_stake_units_applies_fractional_kelly_below_the_cap():
    # f* = (0.52*1 - 0.48) / 1 = 0.04; fractional (KELLY_FRACTION=0.25) -> 0.01
    # of BANKROLL_UNITS=100 -> 1.00 units. Uses this repo's default env
    # values, matching main.KELLY_FRACTION / main.BANKROLL_UNITS below so the
    # assertion tracks the constants even if their defaults ever change.
    prob, odds = 0.52, 2.0
    raw_fraction = kelly_fraction(prob, odds) * main.KELLY_FRACTION
    expected = round(main.BANKROLL_UNITS * raw_fraction, 2)
    assert raw_fraction < main.MAX_STAKE_PCT / 100.0  # sanity: this case is under the cap
    assert main._stake_units(prob, odds) == pytest.approx(expected)


def test_stake_units_is_capped_at_max_stake_pct():
    # A huge modelled edge (prob=0.9 at odds=3.0) must still be capped at
    # MAX_STAKE_PCT of the bankroll, not scaled up to whatever Kelly demands.
    prob, odds = 0.9, 3.0
    uncapped_fraction = kelly_fraction(prob, odds) * main.KELLY_FRACTION
    assert uncapped_fraction > main.MAX_STAKE_PCT / 100.0  # sanity: this case exceeds the cap
    expected = round(main.BANKROLL_UNITS * (main.MAX_STAKE_PCT / 100.0), 2)
    assert main._stake_units(prob, odds) == pytest.approx(expected)
