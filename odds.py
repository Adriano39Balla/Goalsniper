# file: odds.py
# odds aggregation, EV logic, and bookmaker sanity filters

from __future__ import annotations

import os
import time
import statistics
import logging
from typing import Dict, List, Tuple, Optional

import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from cachetools import TTLCache

log = logging.getLogger("odds")

# ───────── Env ─────────
API_KEY = os.getenv("APIFOOTBALL_KEY")
BASE_URL = os.getenv("APISPORTS_BASE_URL", "https://v3.football.api-sports.io").rstrip("/")

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

# Optional book filters
BOOK_WHITELIST = {b.strip().lower() for b in os.getenv("ODDS_BOOK_WHITELIST", "").split(",") if b.strip()}
BOOK_BLACKLIST = {b.strip().lower() for b in os.getenv("ODDS_BOOK_BLACKLIST", "").split(",") if b.strip()}

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

# ───────── Session (larger pool + resilient retries) ─────────
session = requests.Session()
retry = Retry(
    total=3,
    connect=3,
    read=3,
    backoff_factor=0.6,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=frozenset(["GET"]),
    respect_retry_after_header=True,
    raise_on_status=False,
)
adapter = HTTPAdapter(max_retries=retry, pool_connections=64, pool_maxsize=128)
session.mount("https://", adapter)
session.mount("http://", adapter)

# ───────── Cache (auto-evicting TTL LRU) ─────────
_ODDS_CACHE: TTLCache[int, dict] = TTLCache(maxsize=ODDS_CACHE_MAX_ITEMS, ttl=ODDS_CACHE_TTL_SEC)

def clear_odds_cache(fid: Optional[int] = None) -> None:
    """Clear cache for one fixture or all."""
    if fid is None:
        _ODDS_CACHE.clear()
    else:
        try:
            del _ODDS_CACHE[fid]
        except KeyError:
            pass

def set_odds_cache_ttl(seconds: int) -> None:
    """Adjust TTL at runtime (affects new inserts)."""
    try:
        _ODDS_CACHE.ttl = max(5, int(seconds))
    except Exception:
        pass

def _cache_get(fid: int) -> Optional[dict]:
    return _ODDS_CACHE.get(fid)

def _cache_put(fid: int, data: dict) -> None:
    _ODDS_CACHE[fid] = data

# ───────── Circuit breaker for API failures ─────────
_api_fail_ts = 0.0
API_FAIL_COOLDOWN = float(os.getenv("ODDS_FAIL_COOLDOWN_SEC", "8"))

def _api_cooldown_active() -> bool:
    return (_api_fail_ts and (time.time() - _api_fail_ts) < API_FAIL_COOLDOWN)

def _note_api_failure():
    global _api_fail_ts
    _api_fail_ts = time.time()

# ───────── Helpers ─────────
def _api_get(path: str, params: dict) -> Optional[dict]:
    if not API_KEY:
        log.debug("API key missing; odds API disabled")
        return None
    if _api_cooldown_active():
        log.debug("odds api in cooldown; skipping call %s", path)
        return None
    url = f"{BASE_URL}/{path.lstrip('/')}"
    try:
        r = session.get(
            url,
            headers=HEADERS,
            params=params,
            timeout=(HTTP_CONNECT_TIMEOUT, HTTP_READ_TIMEOUT),
        )
        if not r.ok:
            # Respect Retry-After for 429 via urllib3; still note failure for our local cooldown
            _note_api_failure()
            if r.status_code in (429, 500, 502, 503, 504):
                log.debug("odds api non-200 %s for %s params=%s", r.status_code, path, params)
            else:
                log.warning("odds api %s for %s params=%s body=%s", r.status_code, path, params, r.text[:200])
            return None
        try:
            js = r.json()
        except ValueError as e:
            _note_api_failure()
            log.error("Invalid JSON from odds API (%s): %s", path, e)
            return None
        return js if isinstance(js, dict) else None
    except Exception as e:
        _note_api_failure()
        log.warning("odds api request failed for %s: %s", path, e)
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

def _book_ok(name: str) -> bool:
    n = (name or "").strip().lower()
    if BOOK_WHITELIST and n not in BOOK_WHITELIST:
        return False
    if n in BOOK_BLACKLIST:
        return False
    return True

def _parse_ou_label_to_line(lbl: str) -> Optional[Tuple[str, float]]:
    """
    Accepts various book formats e.g.:
      "Over 2.5", "Under 2.5", "Over 2.5 goals", "Under 3", "over (2.75)"
    Returns ("Over"/"Under", line) or None.
    """
    t = (lbl or "").strip().lower()
    side = "Over" if t.startswith("over") else ("Under" if t.startswith("under") else None)
    if not side:
        return None
    # extract first float-looking token
    num = None
    tok = t.replace("(", " ").replace(")", " ").replace(",", ".").split()
    for w in tok:
        try:
            num = float(w)
            break
        except Exception:
            continue
    if num is None:
        return None
    return side, float(num)

def _aggregate_price(vals: List[Tuple[float, str]], prob_hint: Optional[float]):
    xs = [(float(o or 0.0), str(b)) for (o, b) in vals if (o or 0) > 0 and _book_ok(b)]
    if not xs:
        return None, None

    # median & outlier cap
    med = statistics.median([o for (o, _) in xs])
    cap_outlier = med * max(1.0, ODDS_OUTLIER_MULT)
    trimmed = [(o, b) for (o, b) in xs if o <= cap_outlier] or xs

    # fair-odds cap (based on model/odds-implied hint if provided)
    if prob_hint is not None and prob_hint > 0:
        fair = 1.0 / max(1e-6, float(prob_hint))
        cap_fair = fair * max(1.0, ODDS_FAIR_MAX_MULT)
        trimmed = [(o, b) for (o, b) in trimmed if o <= cap_fair] or trimmed

    if ODDS_AGGREGATION == "best":
        best = max(trimmed, key=lambda t: t[0])
        return float(best[0]), str(best[1])

    # median-of-trimmed for stability across books
    med2 = statistics.median([o for (o, _) in trimmed])
    pick = min(trimmed, key=lambda t: abs(t[0] - med2))
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
    return prob * max(0.0, float(odds)) - 1.0

# ───────── Fetch odds ─────────
def fetch_odds(fid: int, prob_hints: Optional[Dict[str, float]] = None) -> dict:
    cached = _cache_get(fid)
    if cached is not None:
        return cached

    js: dict = {}
    # Prefer live; optionally fallback to prematch
    if ODDS_SOURCE in ("auto", "live"):
        tmp = _api_get("odds/live", {"fixture": fid}) or {}
        if tmp.get("response"):
            js = tmp
        elif ODDS_SOURCE == "auto" and FALLBACK_TO_PREMATCH_ON_EMPTY_LIVE:
            js = _api_get("odds", {"fixture": fid}) or {}
    if not js and ODDS_SOURCE == "prematch":
        js = _api_get("odds", {"fixture": fid}) or {}

    if not js:
        log.debug("No odds available for fixture %s (source=%s)", fid, ODDS_SOURCE)

    by_market: Dict[str, Dict[str, List[Tuple[float, str]]]] = {}
    try:
        for r in (js.get("response") or []):
            for bk in (r.get("bookmakers") or []):
                book_name = (bk.get("name") or "Book").strip()
                if not _book_ok(book_name):
                    continue
                for mkt in (bk.get("bets") or []):
                    mname = _market_name_normalize(mkt.get("name", ""))
                    vals = (mkt.get("values") or [])
                    if mname == "BTTS":
                        for v in vals:
                            lbl = (v.get("value") or "").strip().lower()
                            odd = float(v.get("odd") or 0)
                            if odd <= 1.0 or odd > MAX_ODDS_ALL:
                                continue
                            if "yes" in lbl:
                                by_market.setdefault("BTTS", {}).setdefault("Yes", []).append((odd, book_name))
                            elif "no" in lbl:
                                by_market.setdefault("BTTS", {}).setdefault("No", []).append((odd, book_name))
                    elif mname == "1X2":
                        for v in vals:
                            lbl = (v.get("value") or "").strip().lower()
                            odd = float(v.get("odd") or 0)
                            if odd <= 1.0 or odd > MAX_ODDS_ALL:
                                continue
                            if lbl in ("home", "1"):
                                by_market.setdefault("1X2", {}).setdefault("Home", []).append((odd, book_name))
                            elif lbl in ("away", "2"):
                                by_market.setdefault("1X2", {}).setdefault("Away", []).append((odd, book_name))
                            elif lbl == "draw":
                                # We do not tip draws; keep consistent with scan.py
                                pass
                    elif mname == "OU":
                        for v in vals:
                            lbl = (v.get("value") or "")
                            parsed = _parse_ou_label_to_line(lbl)
                            if not parsed:
                                continue
                            side, ln = parsed
                            odd = float(v.get("odd") or 0)
                            if odd <= 1.0 or odd > MAX_ODDS_ALL:
                                continue
                            key = f"OU_{_fmt_line(ln)}"
                            by_market.setdefault(key, {}).setdefault(side, []).append((odd, book_name))
    except Exception as e:
        log.debug("parse odds response failed for fid=%s: %s", fid, e)

    out: Dict[str, Dict[str, dict]] = {}
    for mkey, side_map in by_market.items():
        def _distinct_count(lst: List[Tuple[float, str]]) -> int:
            return len({b for _, b in lst})
        if not side_map:
            continue
        # Require minimum distinct books for each side present
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
    Return (pass, odds, book, ev_pct).
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
        tgt = "Home" if suggestion == "Home Win" else ("Away" if suggestion == "Away Win" else None)
        if tgt and tgt in d:
            odds, book = d[tgt]["odds"], d[tgt]["book"]

    elif market.startswith("Over/Under"):
        try:
            parts = str(suggestion).split()
            # Suggestion looks like "Over 2.5 Goals" / "Under 3 Goals"
            ln = None
            for w in parts:
                try:
                    ln = float(w.replace(",", "."))
                    break
                except Exception:
                    continue
            if ln is not None:
                d = odds_map.get(f"OU_{_fmt_line(ln)}", {})
                tgt = "Over" if str(suggestion).lower().startswith("over") else "Under"
                if tgt in d:
                    odds, book = d[tgt]["odds"], d[tgt]["book"]
        except Exception:
            pass

    if odds is None:
        return (ALLOW_TIPS_WITHOUT_ODDS, odds, book, None)

    if not (_min_odds_for_market(market) <= float(odds) <= MAX_ODDS_ALL):
        return (False, odds, book, None)

    ev_pct = None
    if prob is not None:
        edge = _ev(prob, float(odds))
        ev_pct = round(edge * 100.0, 1)
        if edge < 0:
            return (False, odds, book, ev_pct)

    return (True, float(odds), book, ev_pct)
