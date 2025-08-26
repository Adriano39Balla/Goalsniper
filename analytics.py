# file: analytics.py
"""
Math engine for in-play probabilities (Poisson-based) with soft calibration.
We deliberately avoid extreme 0%/100% outputs to keep tips realistic.
"""

import math
import os
from typing import Dict, Tuple, List

# Softening / clipping knobs (env overrides)
ANALYTICS_MIN_PROB = float(os.getenv("ANALYTICS_MIN_PROB", "0.05"))  # 5% floor/ceiling for binary
ANALYTICS_SOFT_ALPHA = float(os.getenv("ANALYTICS_SOFT_ALPHA", "0.85"))  # 0..1; 1=no softening
WLD_SOFT_ALPHA = float(os.getenv("WLD_SOFT_ALPHA", "0.90"))  # tri-outcome shrink toward uniform


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    try:
        return a / b if b else default
    except Exception:
        return default


def _clip(x: float, lo: float, hi: float) -> float:
    return min(max(x, lo), hi)


def _soften_binary(p: float) -> float:
    """Shrink p toward 0.5 then clip to [m,1-m]."""
    m = ANALYTICS_MIN_PROB
    p = 0.5 + ANALYTICS_SOFT_ALPHA * (p - 0.5)
    return _clip(p, m, 1.0 - m)


def _soften_tri(p_home: float, p_draw: float, p_away: float) -> Tuple[float, float, float]:
    """Shrink 3-way probabilities toward uniform; keeps sum=1."""
    a = _clip(WLD_SOFT_ALPHA, 0.0, 1.0)
    u = (1.0 - a) / 3.0
    ph = u + a * p_home
    pd = u + a * p_draw
    pa = u + a * p_away
    s = ph + pd + pa
    if s > 0:
        ph, pd, pa = ph / s, pd / s, pa / s
    return ph, pd, pa


def poisson_tail(lmbda: float, k: int) -> float:
    """P[X >= k] for Poisson(λ)."""
    if k <= 0:
        return 1.0
    try:
        pmf = math.exp(-lmbda)
    except OverflowError:
        return 0.0
    cdf = pmf
    for i in range(1, k):
        pmf *= lmbda / i
        cdf += pmf
        if pmf < 1e-15:
            break
    return max(0.0, min(1.0, 1.0 - cdf))


def _poisson_pmf_vec(lmbda: float, kmax: int) -> Tuple[float, ...]:
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
    """Estimate per-minute scoring intensity from in-play signals."""
    minute_eff = max(1, min(int(minute or 0), total_minutes))
    rem = max(0, total_minutes - minute_eff)

    xg_h = float(feat.get("xg_h", 0.0))
    xg_a = float(feat.get("xg_a", 0.0))
    xg_sum = float(feat.get("xg_sum", xg_h + xg_a))
    sot_sum = float(feat.get("sot_sum", float(feat.get("sot_h", 0.0)) + float(feat.get("sot_a", 0.0))))
    cor_sum = float(feat.get("cor_sum", float(feat.get("cor_h", 0.0)) + float(feat.get("cor_a", 0.0))))

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

    r_t = _clip(r_t, 0.0, 0.08)
    r_h = _clip(r_h, 0.0, 0.06)
    r_a = _clip(r_a, 0.0, 0.06)

    return {"rate_total": r_t, "rate_h": r_h, "rate_a": r_a, "remaining": rem}


def ou_over_probability(feat: Dict[str, float], line: float, total_minutes: int = 95) -> float:
    """P(final total goals > line), softened and clipped."""
    minute = int(feat.get("minute", 0))
    goals_sum = int(feat.get("goals_sum", 0))
    rates = estimate_rates(feat, minute, total_minutes)
    lam_rem = rates["rate_total"] * rates["remaining"]
    n_needed = max(0, int(math.floor(line + 1e-9) + 1) - goals_sum)
    if n_needed <= 0:
        return _soften_binary(1.0)
    p = poisson_tail(lam_rem, n_needed)
    return _soften_binary(p)


def btts_yes_probability(feat: Dict[str, float], total_minutes: int = 95) -> float:
    """P(both teams score by FT), softened and clipped."""
    minute = int(feat.get("minute", 0))
    gh = int(feat.get("goals_h", 0))
    ga = int(feat.get("goals_a", 0))
    if gh > 0 and ga > 0:
        return _soften_binary(1.0)

    rates = estimate_rates(feat, minute, total_minutes)
    lam_h = rates["rate_h"] * rates["remaining"]
    lam_a = rates["rate_a"] * rates["remaining"]

    if gh > 0 and ga == 0:
        return _soften_binary(1.0 - math.exp(-lam_a))
    if ga > 0 and gh == 0:
        return _soften_binary(1.0 - math.exp(-lam_h))

    ph = 1.0 - math.exp(-lam_h)
    pa = 1.0 - math.exp(-lam_a)
    return _soften_binary(ph * pa)


def wld_probabilities(feat: Dict[str, float], total_minutes: int = 95) -> Tuple[float, float, float]:
    """(Home, Draw, Away) using Poisson enumeration, then soften toward uniform."""
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

    return _soften_tri(p_home, p_draw, p_away)
