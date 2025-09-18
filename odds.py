# file: odds.py
# odds aggregation, EV logic, and bookmaker sanity filters

from __future__ import annotations

import os
import time
import threading
import statistics
import logging
from typing import Dict, List, Tuple, Optional

import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

log = logging.getLogger("odds")

# ───────── Env ─────────
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("APISPORTS_BASE_URL", "https://v3.football.api-sports.io")
HEADERS = {
    "x-apisports-key": API_KEY or "",
    "Accept": "application/json",
    "User-Agent": os.getenv("HTTP_USER_AGENT", "goalsniper/1.0 (+odds)"),
}

ODDS_SOURCE = os.getenv("ODDS_SOURCE", "auto").strip().lower()  # auto|live|prematch
ODDS_AGGREGATION = os.getenv("ODDS_AGGREGATION", "median").strip().lower()  # median|best
ODDS_OUTLIER_MULT = float(os.getenv("ODDS_OUTLIER_MULT", "1.8"))
ODDS_REQUIRE_N_BOOKS = int(os.getenv("ODDS_REQUIRE_N_BOOKS", "2"))
ODDS_FAIR_MAX_MULT = float(os.getenv("ODDS_FAIR_MAX_MULT", "2.5"))
MAX_ODDS_ALL = float(os.getenv("MAX_ODDS_ALL", "20.0"))
FALLBACK_TO_PREMATCH_ON_EMPTY_LIVE = os.getenv("FALLBACK_TO_PREMATCH_ON_EMPTY_LIVE", "1").lower() not in {"0","false","no"}

# Market-specific floors
MIN_ODDS_OU = float(os.getenv("MIN_ODDS_OU", "1.50"))
MIN_ODDS_BTTS = float(os.getenv("MIN_ODDS_BTTS", "1.50"))
MIN_ODDS_1X2 = float(os.getenv("MIN_ODDS_1X2", "1.50"))
ALLOW_TIPS_WITHOUT_ODDS = os.getenv("ALLOW_TIPS_WITHOUT_ODDS", "0").lower() not in {"0","false","no"}

# HTTP timeouts
HTTP_CONNECT_TIMEOUT = float(os.getenv("HTTP_CONNECT_TIMEOUT", "3.0"))
HTTP_READ_TIMEOUT = float(os.getenv("HTTP_READ_TIMEOUT", "10.0"))

# Cache controls
ODDS_CACHE_TTL_SEC = int(os.getenv("ODDS_CACHE_TTL_SEC", "120"))
ODDS_CACHE_MAX_ITEMS = int(os.getenv("ODDS_CACHE_MAX_ITEMS", "2000"))

# ───────── Session ─────────
session = requests.Session()
retry = Retry(
    total=3,
    backoff_factor=0.6,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=frozenset(["GET"]),
    respect_retry_after_header=True,
    raise_on_status=False,
)
session.mount("https://", HTTPAdapter(max_retries=retry))

# ───────── Cache (thread-safe, bounded) ─────────
# value: (ts, data)
_ODDS_CACHE: Dict[int, Tuple[float, dict]] = {}
_CACHE_LOCK = threading.RLock()

def _cache_get(fid: int) -> Optional[dict]:
    now = time.time()
    with _CACHE_LOCK:
        entry = _ODDS_CACHE.get(fid)
        if not entry:
            return None
        ts, data = entry
        if now - ts <= ODDS_CACHE_TTL_SEC:
            return data
        # expired
        _ODDS_CACHE.pop(fid, None)
        return None

def _cache_put(fid: int, data: dict) -> None:
    with _CACHE_LOCK:
        if len(_ODDS_CACHE) >= ODDS_CACHE_MAX_ITEMS:
            # drop ~10% oldest entries to keep map small
            cutoff = int(ODDS_CACHE_MAX_ITEMS * 0.1) or 1
            # sort by ts ascending
            for key, _ in sorted(_ODDS_CACHE.items(), key=lambda kv: kv[1][0])[:cutoff]:
                _ODDS_CACHE.pop(key, None)
        _ODDS_CACHE[fid] = (time.time(), data)

# ───────── Helpers ─────────
def _api_get(path: str, params: dict) -> Optional[dict]:
    if not API_KEY:
        log.debug("API key missing; odds API disabled")
        return None
    url = f"{BASE_URL.rstrip('/')}/{path.lstrip('/')}"
    try:
        r = session.get(
            url,
            headers=HEADERS,
            params=params,
            timeout=(HTTP_CONNECT_TIMEOUT, HTTP_READ_TIMEOUT),
        )
        if not r.ok:
            # Log minimally to avoid noisy logs on rate limits
            if r.status_code in (429, 500, 502, 503, 504):
                log.debug("odds api non-200 %s for %s params=%s", r.status_code, path, params)
            return None
        js = r.json()
        return js if isinstance(js, dict) else None
    except Exception as e:
        log.debug("odds api request failed for %s: %s", path, e)
        return None

def _market_name_normalize(s: str) -> str:
    s = (s or "").strip().lower()
    if "both teams" in s or "btts" in s:
        return "BTTS"
    if "match winner" in s or "winner" in s or "1x2" in s:
        return "1X2"
    if "over/under" in s or "total" in s or "goals" in s:
        return "OU"
    return s.upper()

def _fmt_line(line: float) -> str:
    return f"{line}".rstrip("0").rstrip(".")

def _aggregate_price(vals: List[Tuple[float, str]], prob_hint: Optional[float]):
    """
    Robust-ish aggregation:
      1) drop obviously invalid odds (<=0)
      2) median-based outlier trimming (cap by median * ODDS_OUTLIER_MULT)
      3) optional fair-odds cap based on prob_hint
      4) choose 'best' or 'median-closest'
    """
    xs = [(float(o or 0.0), str(b)) for (o, b) in vals if (o or 0) > 0]
    if not xs:
        return None, None

    # median & outlier cap
    med = statistics.median([o for (o, _) in xs])
    cap_outlier = med * max(1.0, ODDS_OUTLIER_MULT)
    trimmed = [(o, b) for (o, b) in xs if o <= cap_outlier] or xs

    # fair-odds cap (if probability hint available)
    if prob_hint is not None and prob_hint > 0:
        fair = 1.0 / max(1e-6, float(prob_hint))
        cap_fair = fair * max(1.0, ODDS_FAIR_MAX_MULT)
        trimmed = [(o, b) for (o, b) in trimmed if o <= cap_fair] or trimmed

    if ODDS_AGGREGATION == "best":
        best = max(trimmed, key=lambda t: t[0])
        return float(best[0]), str(best[1])

    # median-of-trimmed
    med2 = statistics.median([o for (o, _) in trimmed])
    pick = min(trimmed, key=lambda t: abs(t[0] - med2))
    # include median count for explainability
    distinct_books = len({b for _, b in trimmed})
    return float(pick[0]), f"{pick[1]} (median of {distinct_books})"

def _min_odds_for_market(market: str) -> float:
    if market.startswith("Over/Under"):
        return MIN_ODDS_OU
    if market == "BTTS":
        return MIN_ODDS_BTTS
    if market == "1X2":
        return MIN_ODDS_1X2
    return 1.01

def _ev(prob: float, odds: float) -> float:
    """Return expected value as decimal (0.05 = +5%)."""
    return prob * max(0.0, float(odds)) - 1.0

# ───────── Fetch odds ─────────
def fetch_odds(fid: int, prob_hints: Optional[Dict[str, float]] = None) -> dict:
    """
    Aggregated odds map:
      { "BTTS": {...}, "1X2": {...}, "OU_2.5": {...}, ... }

    - Prefers /odds/live (when ODDS_SOURCE=auto|live)
    - Falls back to /odds (prematch) when enabled
    - Cached per fixture with TTL + LRU-ish bounding
    """
    cached = _cache_get(fid)
    if cached is not None:
        return cached

    js: dict = {}
    if ODDS_SOURCE in ("auto", "live"):
        tmp = _api_get("odds/live", {"fixture": fid}) or {}
        if tmp.get("response"):
            js = tmp
        elif ODDS_SOURCE == "auto" and FALLBACK_TO_PREMATCH_ON_EMPTY_LIVE:
            js = _api_get("odds", {"fixture": fid}) or {}
    if not js and ODDS_SOURCE == "prematch":
        js = _api_get("odds", {"fixture": fid}) or {}

    by_market: Dict[str, Dict[str, List[Tuple[float, str]]]] = {}
    try:
        for r in (js.get("response") or []):
            for bk in (r.get("bookmakers") or []):
                book_name = (bk.get("name") or "Book").strip()
                for mkt in (bk.get("bets") or []):
                    mname = _market_name_normalize(mkt.get("name", ""))
                    vals = (mkt.get("values") or [])
                    if mname == "BTTS":
                        for v in vals:
                            lbl = (v.get("value") or "").strip().lower()
                            odd = float(v.get("odd") or 0)
                            if "yes" in lbl:
                                by_market.setdefault("BTTS", {}).setdefault("Yes", []).append((odd, book_name))
                            elif "no" in lbl:
                                by_market.setdefault("BTTS", {}).setdefault("No", []).append((odd, book_name))
                    elif mname == "1X2":
                        for v in vals:
                            lbl = (v.get("value") or "").strip().lower()
                            odd = float(v.get("odd") or 0)
                            if lbl in ("home", "1"):
                                by_market.setdefault("1X2", {}).setdefault("Home", []).append((odd, book_name))
                            elif lbl in ("away", "2"):
                                by_market.setdefault("1X2", {}).setdefault("Away", []).append((odd, book_name))
                            # Note: Draw intentionally ignored unless you want it later
                    elif mname == "OU":
                        for v in vals:
                            lbl = (v.get("value") or "").strip().lower()
                            # expected format like "Over 2.5" / "Under 2.5"
                            if "over" in lbl or "under" in lbl:
                                parts = lbl.split()
                                try:
                                    ln = float(parts[-1])
                                    key = f"OU_{_fmt_line(ln)}"
                                    side = "Over" if "over" in lbl else "Under"
                                    odd = float(v.get("odd") or 0)
                                    by_market.setdefault(key, {}).setdefault(side, []).append((odd, book_name))
                                except Exception:
                                    continue
    except Exception as e:
        log.debug("parse odds response failed for fid=%s: %s", fid, e)

    out: Dict[str, Dict[str, dict]] = {}
    for mkey, side_map in by_market.items():
        # require distinct bookmakers for each side
        def _distinct_count(lst: List[Tuple[float, str]]) -> int:
            return len({b for _, b in lst})
        if not all(_distinct_count(lst) >= ODDS_REQUIRE_N_BOOKS for lst in side_map.values()):
            continue

        out[mkey] = {}
        for side, lst in side_map.items():
            hint: Optional[float] = None
            if prob_hints:
                if mkey == "BTTS":
                    if side == "Yes":
                        hint = prob_hints.get("BTTS: Yes")
                    else:
                        yes = prob_hints.get("BTTS: Yes")
                        hint = (1.0 - yes) if yes is not None else None
                elif mkey == "1X2":
                    if side == "Home":
                        hint = prob_hints.get("Home Win")
                    elif side == "Away":
                        hint = prob_hints.get("Away Win")
                elif mkey.startswith("OU_"):
                    try:
                        ln = float(mkey.split("_", 1)[1])
                        over_key = f"Over {_fmt_line(ln)} Goals"
                        if side == "Over":
                            hint = prob_hints.get(over_key)
                        else:
                            ov = prob_hints.get(over_key)
                            hint = (1.0 - ov) if ov is not None else None
                    except Exception:
                        hint = None
            ag, label = _aggregate_price(lst, hint)
            if ag is not None:
                out[mkey][side] = {"odds": float(ag), "book": label}

    _cache_put(fid, out)
    return out

# ───────── Price gate ─────────
def price_gate(market: str, suggestion: str, fid: int, prob: Optional[float] = None):
    """
    Return (pass, odds, book, ev_pct). Enforces:
      - odds required unless ALLOW_TIPS_WITHOUT_ODDS=1
      - odds between floors and MAX_ODDS_ALL
      - EV >= 0 if prob given
    """
    odds_map = fetch_odds(fid)
    odds: Optional[float] = None
    book: Optional[str] = None

    if market == "BTTS":
        d = odds_map.get("BTTS", {})
        tgt = "Yes" if str(suggestion).endswith("Yes") else "No"
        if tgt in d:
            odds, book = d[tgt]["odds"], d[tgt]["book"]

    elif market == "1X2":
        d = odds_map.get("1X2", {})
        if suggestion == "Home Win":
            tgt = "Home"
        elif suggestion == "Away Win":
            tgt = "Away"
        else:
            tgt = None
        if tgt and tgt in d:
            odds, book = d[tgt]["odds"], d[tgt]["book"]

    elif market.startswith("Over/Under"):
        try:
            # suggestions like "Over 2.5 Goals" or "Under 2.5 Goals"
            parts = str(suggestion).split()
            ln = float(parts[1])
            d = odds_map.get(f"OU_{_fmt_line(ln)}", {})
            tgt = "Over" if str(suggestion).startswith("Over") else "Under"
            if tgt in d:
                odds, book = d[tgt]["odds"], d[tgt]["book"]
        except Exception:
            pass

    if odds is None:
        return (ALLOW_TIPS_WITHOUT_ODDS, odds, book, None)

    # Floor/ceiling checks
    if not (_min_odds_for_market(market) <= float(odds) <= MAX_ODDS_ALL):
        return (False, odds, book, None)

    ev_pct = None
    if prob is not None:
        edge = _ev(prob, float(odds))
        ev_pct = round(edge * 100.0, 1)
        if edge < 0:
            return (False, odds, book, ev_pct)

    return (True, float(odds), book, ev_pct)
