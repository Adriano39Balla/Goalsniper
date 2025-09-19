# file: train_models.py — calibration + auto-threshold tuning for goalsniper
from __future__ import annotations

import os
import json
import time
import logging
from typing import List, Tuple, Optional, Dict, Iterable

from db import db_conn, get_setting, set_setting, get_setting_json

log = logging.getLogger("train")

# ───────── Defaults (override via env) ─────────
TRAIN_WINDOW_DAYS = int(os.getenv("TRAIN_WINDOW_DAYS", "28"))
TUNE_WINDOW_DAYS  = int(os.getenv("TUNE_WINDOW_DAYS", "14"))
MIN_BETS_FOR_TRAIN = int(os.getenv("MIN_BETS_FOR_TRAIN", "150"))
MIN_BETS_FOR_TUNE  = int(os.getenv("MIN_BETS_FOR_TUNE", "80"))
BINS = max(5, int(os.getenv("CALIBRATION_BINS", "10")))  # clamp to reasonable minimum
EV_CAP = float(os.getenv("EV_CAP", "0.50"))               # clamp EV used only for grid scoring

# Settings keys
CAL_KEY = "calibration_overall"   # JSON: { "bins": [[low,high,obs_rate,count], ...], ... }
CONF_KEY = "CONF_MIN"
EV_KEY = "EV_MIN"
MOTD_CONF_KEY = "MOTD_CONF_MIN"
MOTD_EV_KEY = "MOTD_EV_MIN"

# ───────── Helpers ─────────
def _is_win(sugg: str, gh: Optional[int], ga: Optional[int]) -> Optional[bool]:
    """Return True/False for win/loss; None if cannot grade."""
    if gh is None or ga is None:
        return None
    gh, ga = int(gh), int(ga)
    total = gh + ga
    s = str(sugg or "").strip()

    # OU
    if s.startswith("Over") or s.startswith("Under"):
        ln = None
        for tok in s.split():
            try:
                ln = float(tok); break
            except Exception:
                continue
        if ln is None:
            return None
        return (total > ln) if s.startswith("Over") else (total < ln)

    # BTTS
    if s == "BTTS: Yes":
        return (gh > 0 and ga > 0)
    if s == "BTTS: No":
        return not (gh > 0 and ga > 0)

    # 1X2 (no draw tips in our system)
    if s == "Home Win":
        return gh > ga
    if s == "Away Win":
        return ga > gh

    return None

def _cutoff(days: int) -> int:
    return int(time.time()) - days * 24 * 3600

def _sanitize_conf(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
        if v != v:  # NaN
            return None
        return max(0.0, min(1.0, v))
    except Exception:
        return None

def _load_graded_tips(days: int) -> List[Tuple[float, Optional[float], str, Optional[int], Optional[int]]]:
    """
    Returns: (confidence_raw, odds or None, suggestion, final_goals_h, final_goals_a)
    Includes only rows with results and sent_ok=1, excludes internal 'HARVEST'.
    """
    cutoff = _cutoff(days)
    with db_conn() as c:
        c.execute(
            """
            SELECT t.confidence_raw, t.odds, t.suggestion, r.final_goals_h, r.final_goals_a
            FROM tips t
            JOIN match_results r ON r.match_id = t.match_id
            WHERE t.created_ts >= %s
              AND t.sent_ok = 1
              AND t.suggestion <> 'HARVEST'
            """,
            (cutoff,),
        )
        rows = c.fetchall() or []

    out: List[Tuple[float, Optional[float], str, Optional[int], Optional[int]]] = []
    for cr, odds, sugg, gh, ga in rows:
        conf = _sanitize_conf(cr)
        if conf is None:
            continue
        try:
            o: Optional[float] = None
            if odds is not None:
                o = float(odds)
                # normalize obviously bad odds to None (ignored later by EV/tuning criteria)
                if o <= 1.0 or o > 1000.0:
                    o = None
            out.append((conf, o, str(sugg or ""), (None if gh is None else int(gh)), (None if ga is None else int(ga))))
        except Exception:
            continue
    return out

# ───────── Calibration ─────────
def _build_calibration(bets: List[Tuple[float, Optional[float], str, Optional[int], Optional[int]]]) -> Dict:
    """
    Equal-width bin calibration over confidence_raw in [0,1].
    Output: { "bins": [[low, high, obs, n], ...], "created_ts": ts, "bins_count": BINS }
    """
    created_ts = int(time.time())
    if not bets:
        return {"bins": [], "created_ts": created_ts, "bins_count": BINS}

    width = 1.0 / float(BINS)
    eps = 1e-9
    # bins store [low, high, wins, n]
    bins = [[i * width, (i + 1) * width + (eps if i == BINS - 1 else 0.0), 0, 0] for i in range(BINS)]

    for conf_raw, _odds, sugg, gh, ga in bets:
        idx = min(int(conf_raw / width), BINS - 1)
        win = _is_win(sugg, gh, ga)
        if win is None:
            continue
        bins[idx][3] += 1  # n
        if win:
            bins[idx][2] += 1  # wins

    out_bins = []
    for (low, high, wins, n) in bins:
        obs = (wins / n) if n > 0 else None
        out_bins.append([
            round(low, 3),
            round(high, 3),
            (round(obs, 4) if obs is not None else None),
            int(n),
        ])

    return {"bins": out_bins, "created_ts": created_ts, "bins_count": BINS}

def _prob_from_calibration(cal: Optional[Dict], conf_raw: float) -> float:
    """Map raw confidence to calibrated probability via bin lookup. Fallback: shrink toward 0.5."""
    conf_raw = max(0.0, min(1.0, float(conf_raw)))
    if cal:
        try:
            for (low, high, obs, n) in cal.get("bins") or []:
                if conf_raw >= float(low) and conf_raw < float(high):
                    if obs is not None and int(n) >= 10:
                        return float(obs)
                    break
        except Exception:
            pass
    # simple shrinkage toward 0.5 to avoid overconfidence for sparsely populated bins
    alpha = 0.15
    return (1 - alpha) * conf_raw + alpha * 0.5

def train_models() -> dict:
    """Build and persist calibration curve from recent graded bets."""
    bets = _load_graded_tips(TRAIN_WINDOW_DAYS)
    if len(bets) < MIN_BETS_FOR_TRAIN:
        msg = f"not enough bets to calibrate: {len(bets)}/{MIN_BETS_FOR_TRAIN}"
        log.info("[TRAIN] %s", msg)
        return {"ok": False, "reason": msg}

    cal = _build_calibration(bets)
    try:
        set_setting(CAL_KEY, json.dumps(cal))
    except Exception as e:
        log.exception("[TRAIN] failed to store calibration: %s", e)
        return {"ok": False, "reason": "persist_failed"}

    bins_with_data = sum(1 for b in cal["bins"] if b[3] > 0)
    log.info("[TRAIN] calibration saved (%d/%d bins with data, total_bets=%d)",
             bins_with_data, BINS, len(bets))
    return {"ok": True, "bins_with_data": bins_with_data, "total_bets": len(bets)}

# ───────── Threshold tuning ─────────
def _historical_roi_for_thresholds(
    bets: List[Tuple[float, Optional[float], str, Optional[int], Optional[int]]],
    cal: Optional[Dict],
    conf_min: float,
    ev_min: float,
) -> Tuple[int, float]:
    """
    Returns (n_bets_used, pnl_units) after applying thresholds on calibrated prob/EV.
    Uses realized outcomes for PnL; EV is only a gate.
    """
    pnl = 0.0
    n = 0
    for conf_raw, odds, sugg, gh, ga in bets:
        p = _prob_from_calibration(cal, conf_raw)
        # Gate by prob first
        if p < conf_min:
            continue
        # Gate by EV only if we actually have odds
        if odds is not None:
            ev = max(min(p * odds - 1.0, EV_CAP), -EV_CAP)
            if ev < ev_min:
                continue
        result = _is_win(sugg, gh, ga)
        if result is None:
            continue
        n += 1
        if odds is None:
            # If no odds, treat as 1u flat stake with 0 PnL impact for ROI calc
            pnl += 0.0
        else:
            pnl += (odds - 1.0) if result else -1.0
    return n, pnl

def auto_tune_thresholds(window_days: int = TUNE_WINDOW_DAYS) -> dict:
    """
    Grid-search over (CONF_MIN, EV_MIN) to maximize historical ROI over the recent window.
    Persists best thresholds to settings and returns summary.
    """
    bets = _load_graded_tips(window_days)
    if len(bets) < MIN_BETS_FOR_TUNE:
        msg = f"not enough bets to tune: {len(bets)}/{MIN_BETS_FOR_TUNE}"
        log.info("[TUNE] %s", msg)
        return {"ok": False, "reason": msg}

    cal = get_setting_json(CAL_KEY) or None

    conf_grid = [round(x, 2) for x in frange(0.70, 0.90, 0.01)]
    ev_grid   = [round(x, 2) for x in frange(0.00, 0.10, 0.01)]

    best = {"roi": -1e9, "conf": None, "ev": None, "bets": 0, "pnl": 0.0}
    for cmin in conf_grid:
        for emin in ev_grid:
            n, pnl = _historical_roi_for_thresholds(bets, cal, cmin, emin)
            # require enough bets for stability (at least 30 or 5% of sample)
            if n < max(30, int(0.05 * len(bets))):
                continue
            roi = pnl / max(n, 1)  # units per bet
            if (roi > best["roi"] + 1e-9) or (abs(roi - best["roi"]) <= 1e-9 and n > best["bets"]):
                best = {"roi": roi, "conf": cmin, "ev": emin, "bets": n, "pnl": pnl}

    if best["conf"] is None:
        msg = "no viable thresholds found (grid too strict or data too small)"
        log.info("[TUNE] %s", msg)
        return {"ok": False, "reason": msg}

    # Persist thresholds
    try:
        set_setting(CONF_KEY, str(best["conf"]))
        set_setting(EV_KEY, str(best["ev"]))
    except Exception as e:
        log.exception("[TUNE] failed to persist thresholds: %s", e)
        return {"ok": False, "reason": "persist_failed"}

    # Derive stricter MOTD thresholds
    motd_conf = min(0.95, round(best["conf"] + 0.02, 2))
    motd_ev   = min(0.20, round(best["ev"] + 0.02, 2))
    try:
        set_setting(MOTD_CONF_KEY, str(motd_conf))
        set_setting(MOTD_EV_KEY, str(motd_ev))
    except Exception as e:
        log.warning("[TUNE] failed to persist MOTD thresholds: %s", e)

    log.info("[TUNE] conf=%.2f ev=%.2f on %d bets (ROI=%.3f u/bet, PnL=%.2f)",
             best["conf"], best["ev"], best["bets"], best["roi"], best["pnl"])

    return {
        "ok": True,
        "best": best,
        "motd": {"conf": motd_conf, "ev": motd_ev},
        "window_days": window_days,
        "using_calibration": bool(cal),
    }

# ───────── Threshold loader (optional at boot) ─────────
def _try_float(v: Optional[str], default: float) -> float:
    try:
        return float(v) if v is not None else default
    except Exception:
        return default

def load_thresholds_from_settings(defaults: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """
    Read CONF_MIN / EV_MIN / MOTD_* from settings table.
    Use in main.py boot to override env/runtime knobs:
        th = load_thresholds_from_settings()
    """
    defaults = defaults or {
        "CONF_MIN": float(os.getenv("CONF_MIN", "0.75")),
        "EV_MIN":   float(os.getenv("EV_MIN", "0.00")),
        "MOTD_CONF_MIN": float(os.getenv("MOTD_CONF_MIN", "0.78")),
        "MOTD_EV_MIN":   float(os.getenv("MOTD_EV_MIN", "0.05")),
    }
    return {
        "CONF_MIN": _try_float(get_setting(CONF_KEY), defaults["CONF_MIN"]),
        "EV_MIN":   _try_float(get_setting(EV_KEY), defaults["EV_MIN"]),
        "MOTD_CONF_MIN": _try_float(get_setting(MOTD_CONF_KEY), defaults["MOTD_CONF_MIN"]),
        "MOTD_EV_MIN":   _try_float(get_setting(MOTD_EV_KEY), defaults["MOTD_EV_MIN"]),
    }

# ───────── Utils ─────────
def frange(start: float, stop: float, step: float) -> Iterable[float]:
    v = start
    # numeric loop with tolerance to include the stop
    while v <= stop + 1e-12:
        yield v
        v = round(v + step, 10)
