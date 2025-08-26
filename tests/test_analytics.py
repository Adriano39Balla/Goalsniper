# file: tests/test_analytics.py
# Pure unit tests for analytics; no DB, no SQLite.

from analytics import (
    poisson_tail,
    ou_over_probability,
    btts_yes_probability,
    wld_probabilities,
    estimate_rates,
)

def test_poisson_tail_monotonic_in_k():
    lam = 1.8
    vals = [poisson_tail(lam, k) for k in range(0, 8)]
    assert all(vals[i] >= vals[i+1] for i in range(len(vals)-1))
    assert 0.0 <= vals[-1] <= 1.0

def test_ou_over_probability_increases_with_rate():
    feat_low = {"minute": 60, "goals_sum": 0, "xg_h": 0.2, "xg_a": 0.1, "sot_sum": 1, "cor_sum": 2}
    feat_high = {"minute": 60, "goals_sum": 0, "xg_h": 1.0, "xg_a": 0.8, "sot_sum": 6, "cor_sum": 8}
    p_low = ou_over_probability(feat_low, 2.5, total_minutes=95)
    p_high = ou_over_probability(feat_high, 2.5, total_minutes=95)
    assert p_high > p_low

def test_btts_yes_probability_edges():
    feat = {"minute": 30, "goals_h": 1, "goals_a": 1}
    assert btts_yes_probability(feat, 95) == 1.0
    feat2 = {"minute": 80, "goals_h": 1, "goals_a": 0, "xg_h": 0.8, "xg_a": 0.4, "sot_sum": 4, "cor_sum": 4}
    p = btts_yes_probability(feat2, 95)
    assert 0.0 <= p <= 1.0

def test_wld_probabilities_sum_to_one_and_directional():
    feat = {"minute": 94, "goals_h": 2, "goals_a": 0}
    p_home, p_draw, p_away = wld_probabilities(feat, total_minutes=95)
    s = p_home + p_draw + p_away
    assert abs(s - 1.0) < 1e-6
    assert p_home > 0.9

def test_estimate_rates_bounds():
    feat = {"minute": 45, "xg_h": 1.2, "xg_a": 0.8, "sot_sum": 8, "cor_sum": 10}
    rates = estimate_rates(feat, 45, 95)
    assert 0.0 <= rates["rate_total"] <= 0.08
    assert 0.0 <= rates["rate_h"] <= 0.06
    assert 0.0 <= rates["rate_a"] <= 0.06
