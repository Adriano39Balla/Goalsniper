# train_models.py — simple calibration + auto-threshold tuning for goalsniper
from __future__ import annotations

import os
import json
import math
import time
import logging
import statistics
from typing import List, Tuple, Optional, Dict

from db import db_conn, get_setting, set_setting, get_setting_json

log = logging.getLogger("train")

# Defaults (can be overridden via env)
TRAIN_WINDOW_DAYS = int(os.getenv("TRAIN_WINDOW_DAYS", "28"))      # history for calibration
TUNE_WINDOW_DAYS  = int(os.getenv("TUNE_WINDOW_DAYS", "14"))       # history for threshold tuning
MIN_BETS_FOR_TRAIN = int(os.getenv("MIN_BETS_FOR_TRAIN", "150"))   # minimum graded bets to compute calibration
MIN_BETS_FOR_TUNE  = int(os.getenv("MIN_BETS_FOR_TUNE", "80"))     # minimum graded bets for tuning
BINS = int(os.getenv("CALIBRATION_BINS", "10"))                    # equal-width bins in [0,1]
EV_CAP = float(os.getenv("EV_CAP", "0.50"))                        # guard: ignore absurd EV > +50% in tuning

# Settings keys
CAL_KEY = "calibration_overall"   # JSON: { "bins": [[low,high,obs_rate,count], ...] }
CONF_KEY = "CONF_MIN"
EV_KEY = "EV_MIN"
MOTD_CONF_KEY = "MOTD_CONF_MIN"
MOTD_EV_KEY = "MOTD_EV_MIN"

# ───────── Helpers ─────────

def _is_win(sugg: str, gh: Optional[int], ga: Optional[int]) -> Optional[bool]:
    """Decide win/loss given suggestion and final score. Returns None if unknown."""
    if gh is None or ga is None:
        return None
    gh, ga = int(gh), int(ga)
    total = gh + ga
    s = str(sugg or "")
    if s.startswith("Over") or s.startswith("Under"):
        # extract line (first float found)
        ln = None
        for tok in s.split():
            try:
                ln = float(tok)
                break
            except Exception:
                continue
        if ln is None:
            return None
        return (total > ln) if s.startswith("Over") else (total < ln)
    if s == "BTTS: Yes":
        return (gh > 0 and ga > 0)
    if s == "BTTS: No":
        return not (gh > 0 and ga > 0)
    if s == "Home Win":
        return gh > ga
    if s == "Away Win":
        return ga > gh
    return None

def _cutoff(days: int) -> int:
    return int(time.time()) - days * 24 * 3600

def _load_graded_tips(days: int) -> List[Tuple[float, float, str, int, int]]:
    """
    Returns list of tuples:
      (confidence_raw, odds, suggestion, final_goals_h, final_goals_a)
    Only includes rows where result is known and sent_ok=1.
    """
    cutoff = _cutoff(days)
    with db_conn() as c:
        rows = c.execute(
            """
            SELECT t.confidence_raw, t.odds, t.suggestion, r.final_goals_h, r.final_goals_a
            FROM tips t
            JOIN match_results r ON r.match_id = t.match_id
            WHERE t.created_ts >= %s
              AND t.sent_ok = 1
              AND t.suggestion <> 'HARVEST'
            """,
            (cutoff,),
        ).fetchall()
    out = []
    for cr, odds, sugg, gh, ga in rows:
        try:
            out.append((float(cr or 0.0), float(odds or 0.0), str(sugg or ""), int(gh) if gh is not None else None, int(ga) if ga is not None else None))  # type: ignore[arg-type]
        except Exception:
            continue
    return out

# ───────── Calibration ─────────

def _build_calibration(bets: List[Tuple[float, float, str, int, int]]) -> Dict:
    """
    Create equal-width bin calibration over confidence_raw in [0,1].
    Returns schema:
      { "bins": [[low, high, obs, n], ...], "created_ts": 123, "bins_count": BINS }
    """
    if not bets:
        return {"bins": [], "created_ts": int(time.time()), "bins_count": BINS}

    bins = []
    eps = 1e-9
    width = 1.0 / BINS
    for i in range(BINS):
        low = i * width
        high = (i + 1) * width + (eps if i == BINS - 1 else 0.0)
        bins.append([low, high, 0, 0])  # low, high, wins, n

    for conf_raw, _odds, sugg, gh, ga in bets:
        idx = min(int(conf_raw / width), BINS - 1)
        win = _is_win(sugg, gh, ga)
        if win is None:
            continue
        bins[idx][3] += 1  # n
        if win:
            bins[idx][2] += 1  # wins

    # Convert to observed rate per bin
    out_bins = []
    for (low, high, wins, n) in bins:
        obs = (wins / n) if n > 0 else None
        out_bins.append([round(low, 3), round(high, 3), (round(obs, 4) if obs is not None else None), n])

    return {"bins": out_bins, "created_ts": int(time.time()), "bins_count": BINS}

def _prob_from_calibration(cal: Dict, conf_raw: float) -> float:
    """Map raw confidence to calibrated probability via bin lookup (fallback to raw)."""
    try:
        bins = cal.get("bins") or []
        for (low, high, obs, n) in bins:
            if conf_raw >= low and conf_raw < high:
                if obs is not None and n >= 10:  # require minimum support
                    return float(obs)
                break
    except Exception:
        pass
    # fallback: light shrinkage toward 0.5 to avoid overconfidence
    alpha = 0.15
    return (1 - alpha) * float(conf_raw) + alpha * 0.5

def train_models() -> dict:
    """
    Train a simple calibration curve from recent graded bets and store it in settings.
    """
    bets = _load_graded_tips(TRAIN_WINDOW_DAYS)
    if len(bets) < MIN_BETS_FOR_TRAIN:
        msg = f"not enough bets to calibrate: {len(bets)}/{MIN_BETS_FOR_TRAIN}"
        log.info("[TRAIN] %s", msg)
        return {"ok": False, "reason": msg}

    cal = _build_calibration(bets)
    set_setting(CAL_KEY, json.dumps(cal))
    log.info("[TRAIN] calibration saved (%d bins with data)", sum(1 for b in cal["bins"] if b[3] > 0))

    return {"ok": True, "bins_with_data": sum(1 for b in cal["bins"] if b[3] > 0), "total_bets": len(bets)}

# ───────── Threshold tuning ─────────

def _historical_roi_for_thresholds(
    bets: List[Tuple[float, float, str, int, int]],
    cal: Optional[Dict],
    conf_min: float,
    ev_min: float,
) -> Tuple[int, float]:
    """
    Returns (n_bets_used, pnl_units) after applying thresholds on calibrated prob and odds.
    Uses realized outcomes for PnL; EV only gates selection.
    """
    pnl = 0.0
    n = 0
    for conf_raw, odds, sugg, gh, ga in bets:
        if odds is None or odds <= 1.0 or odds > 1000.0:
            continue
        p = _prob_from_calibration(cal, conf_raw) if cal else conf_raw
        # sanity EV cap to avoid extreme values dominating tuning
        ev = max(min(p * odds - 1.0, EV_CAP), -EV_CAP)
        if p < conf_min or ev < ev_min:
            continue
        result = _is_win(sugg, gh, ga)
        if result is None:
            continue
        n += 1
        pnl += (odds - 1.0) if result else -1.0
    return n, pnl

def auto_tune_thresholds(window_days: int = TUNE_WINDOW_DAYS) -> dict:
    """
    Grid-search over (CONF_MIN, EV_MIN) to maximize historical ROI over the recent window.
    Writes results to settings and returns a summary dict.
    """
    bets = _load_graded_tips(window_days)
    if len(bets) < MIN_BETS_FOR_TUNE:
        msg = f"not enough bets to tune: {len(bets)}/{MIN_BETS_FOR_TUNE}"
        log.info("[TUNE] %s", msg)
        return {"ok": False, "reason": msg}

    cal = get_setting_json(CAL_KEY) or None

    # Build grid
    conf_grid = [round(x, 2) for x in frange(0.70, 0.90, 0.01)]
    ev_grid = [round(x, 2) for x in frange(0.00, 0.10, 0.01)]

    best = {"roi": -1e9, "conf": None, "ev": None, "bets": 0, "pnl": 0.0}
    for cmin in conf_grid:
        for emin in ev_grid:
            n, pnl = _historical_roi_for_thresholds(bets, cal, cmin, emin)
            if n < max(30, int(0.05 * len(bets))):  # avoid overfitting tiny sample
                continue
            roi = pnl / max(n, 1)  # units per bet
            # prefer higher n when ROI ties (within small epsilon)
            if (roi > best["roi"] + 1e-6) or (abs(roi - best["roi"]) <= 1e-6 and n > best["bets"]):
                best = {"roi": roi, "conf": cmin, "ev": emin, "bets": n, "pnl": pnl}

    if best["conf"] is None:
        msg = "no viable thresholds found (grid too strict or data too small)"
        log.info("[TUNE] %s", msg)
        return {"ok": False, "reason": msg}

    # Persist thresholds
    set_setting(CONF_KEY, str(best["conf"]))
    set_setting(EV_KEY, str(best["ev"]))

    # Derive MOTD thresholds a bit stricter
    motd_conf = min(0.95, round(best["conf"] + 0.02, 2))
    motd_ev = min(0.20, round(best["ev"] + 0.02, 2))
    set_setting(MOTD_CONF_KEY, str(motd_conf))
    set_setting(MOTD_EV_KEY, str(motd_ev))

    log.info("[TUNE] best conf=%.2f ev=%.2f on %d bets (ROI=%.3f u/bet, PnL=%.2f)",
             best["conf"], best["ev"], best["bets"], best["roi"], best["pnl"])

    return {
        "ok": True,
        "best": best,
        "motd": {"conf": motd_conf, "ev": motd_ev},
        "window_days": window_days,
        "using_calibration": bool(cal),
    }

# ───────── Utils ─────────

def frange(start: float, stop: float, step: float):
    v = start
    while v <= stop + 1e-9:
        yield v
        v += step
