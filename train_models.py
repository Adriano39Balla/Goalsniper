# file: train_models.py — calibration + auto-threshold tuning for goalsniper
from __future__ import annotations

import os
import json
import time
import logging
from typing import List, Tuple, Optional, Dict, Iterable
from contextlib import contextmanager

import psycopg2
from psycopg2.extras import DictCursor

log = logging.getLogger("train")

# ───────────────────────────────────────────────────────────────────────────────
# Config (override via env)
# ───────────────────────────────────────────────────────────────────────────────
TRAIN_WINDOW_DAYS = int(os.getenv("TRAIN_WINDOW_DAYS", "28"))
TUNE_WINDOW_DAYS  = int(os.getenv("TUNE_WINDOW_DAYS", "14"))
MIN_BETS_FOR_TRAIN = int(os.getenv("MIN_BETS_FOR_TRAIN", "150"))
MIN_BETS_FOR_TUNE  = int(os.getenv("MIN_BETS_FOR_TUNE", "80"))
BINS = int(os.getenv("CALIBRATION_BINS", "10"))
EV_CAP = float(os.getenv("EV_CAP", "0.50"))  # clamp EV in tuning to avoid outliers

# Settings keys in DB
CAL_KEY       = "calibration_overall"   # JSON: { "bins": [[low,high,obs_rate,count], ...], ... }
CONF_KEY      = "CONF_MIN"
EV_KEY        = "EV_MIN"
MOTD_CONF_KEY = "MOTD_CONF_MIN"
MOTD_EV_KEY   = "MOTD_EV_MIN"

# ───────────────────────────────────────────────────────────────────────────────
# Local DB helpers (independent of main.py to avoid circular imports)
# ───────────────────────────────────────────────────────────────────────────────
_DATABASE_URL = os.getenv("DATABASE_URL", "")

def _dsn_with_ssl(url: str) -> str:
    if not url:
        raise RuntimeError("DATABASE_URL is required for training module")
    need_ssl = os.getenv("DB_SSLMODE_REQUIRE", "1").strip().lower() not in {"0","false","no",""}
    if need_ssl and "sslmode=" not in url and url.startswith(("postgres://","postgresql://")):
        url = url + (("&" if "?" in url else "?") + "sslmode=require")
    return url

@contextmanager
def db_conn(dict_rows: bool = False):
    dsn = _dsn_with_ssl(_DATABASE_URL)
    conn = psycopg2.connect(dsn)  # autocommit off by default (we'll mostly read)
    try:
        with conn:
            with conn.cursor(cursor_factory=DictCursor if dict_rows else None) as cur:
                yield cur
    finally:
        try:
            conn.close()
        except Exception:
            pass

def get_setting(key: str) -> Optional[str]:
    try:
        with db_conn() as c:
            c.execute("SELECT value FROM settings WHERE key=%s", (key,))
            row = c.fetchone()
            return (row[0] if row else None)
    except Exception as e:
        log.warning("[TRAIN] get_setting(%s) failed: %s", key, e)
        return None

def set_setting(key: str, value: str) -> None:
    try:
        with db_conn() as c:
            c.execute(
                "INSERT INTO settings(key,value) VALUES(%s,%s) "
                "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
                (key, value),
            )
    except Exception as e:
        log.warning("[TRAIN] set_setting(%s) failed: %s", key, e)

def get_setting_json(key: str) -> Optional[dict]:
    try:
        raw = get_setting(key)
        return json.loads(raw) if raw else None
    except Exception as e:
        log.warning("[TRAIN] get_setting_json(%s) failed: %s", key, e)
        return None

# ───────────────────────────────────────────────────────────────────────────────
# Core helpers
# ───────────────────────────────────────────────────────────────────────────────
def _cutoff(days: int) -> int:
    return int(time.time()) - days * 24 * 3600

def _is_win(sugg: str, gh: Optional[int], ga: Optional[int]) -> Optional[bool]:
    """Return True/False for win/loss; None if cannot grade (missing line etc.)."""
    if gh is None or ga is None:
        return None
    gh, ga = int(gh), int(ga)
    total = gh + ga
    s = str(sugg or "")
    if s.startswith("Over") or s.startswith("Under"):
        # extract first float found
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

def _load_graded_tips(days: int) -> List[Tuple[float, float, str, Optional[int], Optional[int]]]:
    """
    Returns: (confidence_raw, odds, suggestion, final_goals_h, final_goals_a)
    Only includes rows with results and sent_ok=1 (joins can still yield NULLs for old rows).
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

    out: List[Tuple[float, float, str, Optional[int], Optional[int]]] = []
    for cr, odds, sugg, gh, ga in rows:
        try:
            out.append((
                float(cr or 0.0),
                float(odds or 0.0),
                str(sugg or ""),
                (None if gh is None else int(gh)),
                (None if ga is None else int(ga)),
            ))
        except Exception:
            continue
    return out

# ───────────────────────────────────────────────────────────────────────────────
# Calibration
# ───────────────────────────────────────────────────────────────────────────────
def _build_calibration(bets: List[Tuple[float, float, str, Optional[int], Optional[int]]]) -> Dict:
    """
    Equal-width bin calibration over confidence_raw in [0,1].
    Output: { "bins": [[low, high, obs, n], ...], "created_ts": ts, "bins_count": BINS }
    """
    created_ts = int(time.time())
    if not bets:
        return {"bins": [], "created_ts": created_ts, "bins_count": BINS}

    width = 1.0 / max(1, BINS)
    eps = 1e-9
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
    if cal:
        try:
            for (low, high, obs, n) in cal.get("bins") or []:
                if conf_raw >= float(low) and conf_raw < float(high):
                    if obs is not None and int(n) >= 10:  # support threshold
                        return float(obs)
                    break
        except Exception:
            pass
    # light shrinkage to avoid overconfidence when no bin support
    alpha = 0.15
    return (1 - alpha) * float(conf_raw) + alpha * 0.5

def train_models() -> dict:
    """Build and persist calibration curve from recent graded bets."""
    bets = _load_graded_tips(TRAIN_WINDOW_DAYS)
    if len(bets) < MIN_BETS_FOR_TRAIN:
        msg = f"not enough bets to calibrate: {len(bets)}/{MIN_BETS_FOR_TRAIN}"
        log.info("[TRAIN] %s", msg)
        return {"ok": False, "reason": msg}

    cal = _build_calibration(bets)
    set_setting(CAL_KEY, json.dumps(cal))
    bins_with_data = sum(1 for b in cal["bins"] if b[3] > 0)
    log.info("[TRAIN] calibration saved (%d/%d bins with data)", bins_with_data, BINS)
    return {"ok": True, "bins_with_data": bins_with_data, "total_bets": len(bets)}

# ───────────────────────────────────────────────────────────────────────────────
# Threshold tuning
# ───────────────────────────────────────────────────────────────────────────────
def _historical_roi_for_thresholds(
    bets: List[Tuple[float, float, str, Optional[int], Optional[int]]],
    cal: Optional[Dict],
    conf_min: float,
    ev_min: float,
) -> Tuple[int, float]:
    pnl = 0.0
    n = 0
    for conf_raw, odds, sugg, gh, ga in bets:
        if odds is None or odds <= 1.0 or odds > 1000.0:
            continue
        p = _prob_from_calibration(cal, conf_raw)
        # clamp EV contribution to avoid extreme outliers driving tuning
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
            # Avoid overfitting to tiny samples
            if n < max(30, int(0.05 * len(bets))):
                continue
            roi = pnl / max(n, 1)
            if (roi > best["roi"] + 1e-6) or (abs(roi - best["roi"]) <= 1e-6 and n > best["bets"]):
                best = {"roi": roi, "conf": cmin, "ev": emin, "bets": n, "pnl": pnl}

    if best["conf"] is None:
        msg = "no viable thresholds found"
        log.info("[TUNE] %s", msg)
        return {"ok": False, "reason": msg}

    # Persist thresholds
    set_setting(CONF_KEY, str(best["conf"]))
    set_setting(EV_KEY, str(best["ev"]))

    # Derive stricter MOTD thresholds
    motd_conf = min(0.95, round(best["conf"] + 0.02, 2))
    motd_ev   = min(0.20, round(best["ev"] + 0.02, 2))
    set_setting(MOTD_CONF_KEY, str(motd_conf))
    set_setting(MOTD_EV_KEY, str(motd_ev))

    log.info("[TUNE] conf=%.2f ev=%.2f on %d bets (ROI=%.3f u/bet, PnL=%.2f)",
             best["conf"], best["ev"], best["bets"], best["roi"], best["pnl"])

    return {
        "ok": True,
        "best": best,
        "motd": {"conf": motd_conf, "ev": motd_ev},
        "window_days": window_days,
        "using_calibration": bool(cal),
    }

# ───────────────────────────────────────────────────────────────────────────────
# Threshold loader for runtime (used by main.py)
# ───────────────────────────────────────────────────────────────────────────────
def _try_float(v: Optional[str], default: float) -> float:
    try:
        return float(v) if v is not None else default
    except Exception:
        return default

def load_thresholds_from_settings(defaults: Optional[Dict[str, float]] = None) -> Dict[str, float]:
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

# ───────────────────────────────────────────────────────────────────────────────
# Utils
# ───────────────────────────────────────────────────────────────────────────────
def frange(start: float, stop: float, step: float) -> Iterable[float]:
    v = start
    while v <= stop + 1e-9:
        yield v
        v += step
