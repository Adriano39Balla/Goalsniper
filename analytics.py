# file: analytics.py
"""
Lightweight math engine for in-play probabilities.
Uses Poisson projections from live features.
"""

import math
from typing import Dict, Tuple, List


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    # Why: live minutes may be 0 early; avoid division errors
    try:
        return a / b if b else default
    except Exception:
        return default


def _clip(x: float, lo: float, hi: float) -> float:
    return min(max(x, lo), hi)


def poisson_tail(lmbda: float, k: int) -> float:
    """P[X >= k] for Poisson(λ). Stable for small/medium λ."""
    if k <= 0:
        return 1.0
    try:
        pmf0 = math.exp(-lmbda)
    except OverflowError:
        return 0.0
    cdf = pmf0
    pmf = pmf0
    for i in range(1, k):
        pmf *= lmbda / i
        cdf += pmf
        if pmf < 1e-15:
            break
    return _clip(1.0 - cdf, 0.0, 1.0)


def _poisson_pmf_vec(lmbda: float, kmax: int) -> Tuple[float, ...]:
    """PMF[0..kmax] for Poisson(λ) via recursion; normalized."""
    pmfs: List[float] = [0.0] * (kmax + 1)
    try:
        pmfs[0] = math.exp(-lmbda)
    except OverflowError:
        return tuple(pmfs)
    for k in range(1, kmax + 1):
        pmfs[k] = pmfs[k - 1] * (lmbda / k)
    s = sum(pmfs)
    if s > 0:
        pmfs = [p / s for p in pmfs]
    return tuple(pmfs)


def estimate_rates(feat: Dict[str, float], minute: int, total_minutes: int = 95) -> Dict[str, float]:
    """
    Convert in-play stats → remaining-goal intensities (per-minute).
    Why: simple, explainable baseline complementary to ML signals.
    """
    minute_eff = max(1, min(int(minute or 0), total_minutes))
    rem = max(0, total_minutes - minute_eff)

    xg_h = float(feat.get("xg_h", 0.0))
    xg_a = float(feat.get("xg_a", 0.0))
    xg_sum = float(feat.get("xg_sum", xg_h + xg_a))
    sot_sum = float(feat.get("sot_sum", float(feat.get("sot_h", 0.0)) + float(feat.get("sot_a", 0.0))))
    cor_sum = float(feat.get("cor_sum", float(feat.get("cor_h", 0.0)) + float(feat.get("cor_a", 0.0))))

    # Conservative mapping of auxiliary stats to xG-ish rate
    SOT_TO_XG = 0.12
    CK_TO_XG = 0.03

    r_xg_t = _safe_div(xg_sum, minute_eff)
    r_sot_t = _safe_div(sot_sum, minute_eff) * SOT_TO_XG
    r_cor_t = _safe_div(cor_sum, minute_eff) * CK_TO_XG

    r_t = r_xg_t + 0.35 * r_sot_t + 0.15 * r_cor_t
    share_h = _safe_div(xg_h, (xg_h + xg_a), 0.5)
    share_h = _clip(share_h, 0.15, 0.85)
    r_h = r_t * share_h
    r_a = r_t * (1.0 - share_h)

    # Bound extremes for stability
    r_t = _clip(r_t, 0.0, 0.08)
    r_h = _clip(r_h, 0.0, 0.06)
    r_a = _clip(r_a, 0.0, 0.06)

    return {"rate_total": r_t, "rate_h": r_h, "rate_a": r_a, "remaining": rem}


def ou_over_probability(feat: Dict[str, float], line: float, total_minutes: int = 95) -> float:
    """P(final total goals > line)."""
    minute = int(feat.get("minute", 0))
    goals_sum = int(feat.get("goals_sum", 0))
    rates = estimate_rates(feat, minute, total_minutes)
    lam_rem = rates["rate_total"] * rates["remaining"]
    n_needed = max(0, int(math.floor(line + 1e-9) + 1) - goals_sum)  # robust for .5 lines
    if n_needed <= 0:
        return 1.0
    return _clip(poisson_tail(lam_rem, n_needed), 0.0, 1.0)


def btts_yes_probability(feat: Dict[str, float], total_minutes: int = 95) -> float:
    """P(both teams have ≥1 goal by FT)."""
    minute = int(feat.get("minute", 0))
    gh = int(feat.get("goals_h", 0))
    ga = int(feat.get("goals_a", 0))
    if gh > 0 and ga > 0:
        return 1.0

    rates = estimate_rates(feat, minute, total_minutes)
    lam_h = rates["rate_h"] * rates["remaining"]
    lam_a = rates["rate_a"] * rates["remaining"]

    if gh > 0 and ga == 0:
        return _clip(1.0 - math.exp(-lam_a), 0.0, 1.0)
    if ga > 0 and gh == 0:
        return _clip(1.0 - math.exp(-lam_h), 0.0, 1.0)

    ph = 1.0 - math.exp(-lam_h)
    pa = 1.0 - math.exp(-lam_a)
    return _clip(ph * pa, 0.0, 1.0)


def wld_probabilities(feat: Dict[str, float], total_minutes: int = 95) -> Tuple[float, float, float]:
    """(P(Home Win), P(Draw), P(Away Win)) via independent remaining-goal Poissons."""
    minute = int(feat.get("minute", 0))
    gh = int(feat.get("goals_h", 0))
    ga = int(feat.get("goals_a", 0))

    rates = estimate_rates(feat, minute, total_minutes)
    lam_h = rates["rate_h"] * rates["remaining"]
    lam_a = rates["rate_a"] * rates["remaining"]

    K = max(12, int(math.ceil(lam_h + lam_a + 6)))
    pmf_h = _poisson_pmf_vec(lam_h, K)
    pmf_a = _poisson_pmf_vec(lam_a, K)

    base_diff = gh - ga
    p_home = p_draw = p_away = 0.0
    for i, p_i in enumerate(pmf_h):
        if not p_i:
            continue
        for j, p_j in enumerate(pmf_a):
            pij = p_i * p_j
            d = base_diff + (i - j)
            if d > 0:
                p_home += pij
            elif d == 0:
                p_draw += pij
            else:
                p_away += pij

    s = p_home + p_draw + p_away
    if s > 0:
        p_home, p_draw, p_away = p_home / s, p_draw / s, p_away / s
    return _clip(p_home, 0.0, 1.0), _clip(p_draw, 0.0, 1.0), _clip(p_away, 0.0, 1.0)
