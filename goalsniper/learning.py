import math
from typing import Dict, Any, List, Tuple, Optional

from .storage import (
    market_stats,
    recent_market_samples,
    recent_market_league_samples,
    insert_tip_return_id,
    get_tip_by_id,   # NEW: use full tip payload on feedback
)
from .config import MIN_CONFIDENCE_TO_SEND

# Beta prior for precision smoothing (wins/losses)
_ALPHA = 2.0
_BETA = 2.0


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _logit(p: float) -> float:
    p = _clamp01(p)
    p = min(1 - 1e-6, max(1e-6, p))
    return math.log(p / (1 - p))


def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _fit_platt(
    samples: List[Tuple[float, int]],
    l2: float = 2.0,
    iters: int = 25
) -> Tuple[float, float]:
    """
    Simple Platt scaling on (probability, outcome) pairs.
    Returns intercept 'a' and slope 'b'. If not enough data, identity.
    """
    n = len(samples)
    if n < 25:
        return 0.0, 1.0

    x = [_logit(p) for p, _ in samples]
    y = [int(t[1]) for t in samples]

    a, b = 0.0, 1.0
    lam = float(l2)

    for _ in range(iters):
        g0 = g1 = 0.0
        h00 = h01 = h11 = 0.0
        for xi, yi in zip(x, y):
            z = a + b * xi
            pi = _sigmoid(z)
            w = pi * (1 - pi)
            g0 += (yi - pi)
            g1 += (yi - pi) * xi
            h00 += w
            h01 += w * xi
            h11 += w * xi * xi

        # L2
        g0 -= lam * a
        g1 -= lam * b
        h00 += lam
        h11 += lam

        det = h00 * h11 - h01 * h01
        if abs(det) < 1e-9:
            break

        da = ( g0 * h11 - g1 * h01) / det
        db = (-g0 * h01 + g1 * h00) / det

        a += 0.8 * da
        b += 0.8 * db

        if abs(da) + abs(db) < 1e-5:
            break

    return float(a), float(b)


def _apply_platt(a: float, b: float, p: float) -> float:
    return _clamp01(_sigmoid(a + b * _logit(p)))


def _fit_league_intercept(
    a: float,
    b: float,
    samples: List[Tuple[float, int]],
    l2: float = 3.0,
    iters: int = 18
) -> float:
    """One-parameter league intercept atop global Platt."""
    if len(samples) < 40:
        return 0.0

    c = 0.0
    lam = float(l2)
    xs = [_logit(p) for p, _ in samples]
    ys = [int(t[1]) for t in samples]

    for _ in range(iters):
        g = 0.0
        h = 0.0
        for xi, yi in zip(xs, ys):
            z = a + c + b * xi
            pi = _sigmoid(z)
            g += (yi - pi)
            h += pi * (1 - pi)

        g -= lam * c
        h += lam
        if h <= 1e-9:
            break

        dc = g / h
        c += 0.9 * dc

        if abs(dc) < 1e-5:
            break

    return float(c)


async def calibrate_probability(market: str, p: float, league_id: Optional[int] = None) -> float:
    """Return calibrated probability for this market (optionally league-adjusted)."""
    rows = await recent_market_samples(market, limit=400)
    samples = [(float(r["probability"]), int(r["outcome"])) for r in rows]
    a, b = _fit_platt(samples)

    c = 0.0
    if league_id is not None:
        rows_l = await recent_market_league_samples(market, int(league_id), limit=120)
        samples_l = [(float(r["probability"]), int(r["outcome"])) for r in rows_l]
        c = _fit_league_intercept(a, b, samples_l)

    return _clamp01(_sigmoid(a + c + b * _logit(float(p))))


async def dynamic_conf_threshold(market: str) -> float:
    """
    Choose a confidence threshold for this market aiming at a smoothed precision target,
    with a fall-back to the static MIN_CONFIDENCE_TO_SEND when data is scarce.
    """
    rows = await recent_market_samples(market, limit=250)
    if len(rows) < 40:
        return MIN_CONFIDENCE_TO_SEND

    samples = [(float(r["probability"]), int(r["outcome"])) for r in rows]
    a, b = _fit_platt(samples)

    wins, losses = await market_stats(market)
    target_precision = (wins + _ALPHA) / (wins + losses + _ALPHA + _BETA)
    target_precision = max(0.54, min(0.72, target_precision + 0.02))

    thresholds = [x / 100.0 for x in range(8, 96, 2)]
    best_f1 = (0.0, MIN_CONFIDENCE_TO_SEND)
    chosen = None

    for thr in thresholds:
        tp = fp = fn = 0
        for p_raw, y in samples:
            p_cal = _apply_platt(a, b, p_raw)
            conf = abs(p_cal - 0.5) * 2.0
            pred = 1 if conf >= thr else None
            if pred is None:
                if y == 1:
                    fn += 1
                continue
            if (p_cal >= 0.5 and y == 1) or (p_cal < 0.5 and y == 0):
                tp += 1
            else:
                fp += 1
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        if f1 > best_f1[0]:
            best_f1 = (f1, thr)
        if prec >= target_precision and (chosen is None or thr < chosen):
            chosen = thr

    if chosen is not None:
        return float(_clamp01(chosen))
    return float(_clamp01(best_f1[1]))


async def register_sent_tip(tip: Dict[str, Any], message_id: int) -> int:
    """Persist a sent tip (returns DB tip id)."""
    return await insert_tip_return_id(tip, message_id)


# -------- NEW: optional feedback hook --------
async def on_feedback_update(tip_id: int, outcome: int) -> Dict[str, Any]:
    """
    Called after outcome is recorded (👍=1 / 👎=0). We pull the *full* tip payload
    to enable richer learning/telemetry. This function doesn’t need to persist
    anything extra — model calibration already uses DB history — but it returns
    helpful, precomputed info for logs/dashboards.
    """
    tip = await get_tip_by_id(int(tip_id))
    if not tip:
        return {"ok": False, "reason": "tip_not_found"}

    market   = str(tip.get("market") or "")
    leagueId = tip.get("leagueId")
    prob     = float(tip.get("probability") or 0.5)

    # Recompute calibrated prob + a fresh market threshold for observability
    p_cal = await calibrate_probability(market, prob, leagueId)
    conf  = abs(p_cal - 0.5) * 2.0
    thr   = await dynamic_conf_threshold(market)

    return {
        "ok": True,
        "tip": tip,
        "market": market,
        "leagueId": int(leagueId or 0),
        "rawProb": prob,
        "calibratedProb": p_cal,
        "confidence": conf,
        "marketThreshold": thr,
        "outcome": int(outcome),
    }
