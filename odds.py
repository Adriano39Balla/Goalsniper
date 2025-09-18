# file: odds.py
# odds aggregation, EV logic, and bookmaker sanity filters

import os, time, statistics
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# ───────── Env ─────────
API_KEY = os.getenv("API_KEY")
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY, "Accept": "application/json"}

ODDS_SOURCE = os.getenv("ODDS_SOURCE", "auto").lower()  # auto|live|prematch
ODDS_AGGREGATION = os.getenv("ODDS_AGGREGATION", "median").lower()
ODDS_OUTLIER_MULT = float(os.getenv("ODDS_OUTLIER_MULT", "1.8"))
ODDS_REQUIRE_N_BOOKS = int(os.getenv("ODDS_REQUIRE_N_BOOKS", "2"))
ODDS_FAIR_MAX_MULT = float(os.getenv("ODDS_FAIR_MAX_MULT", "2.5"))
MAX_ODDS_ALL = float(os.getenv("MAX_ODDS_ALL", "20.0"))

# Market-specific floors
MIN_ODDS_OU = float(os.getenv("MIN_ODDS_OU", "1.50"))
MIN_ODDS_BTTS = float(os.getenv("MIN_ODDS_BTTS", "1.50"))
MIN_ODDS_1X2 = float(os.getenv("MIN_ODDS_1X2", "1.50"))
ALLOW_TIPS_WITHOUT_ODDS = os.getenv("ALLOW_TIPS_WITHOUT_ODDS", "0") not in ("0","false","False","no","NO")

# ───────── Session ─────────
session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1,
                                                       status_forcelist=[429,500,502,503,504],
                                                       respect_retry_after_header=True)))

# ───────── Cache ─────────
ODDS_CACHE: dict[int, tuple[float, dict]] = {}

def _api_get(url: str, params: dict, timeout: int = 15):
    if not API_KEY:
        return None
    try:
        r = session.get(url, headers=HEADERS, params=params, timeout=timeout)
        return r.json() if r.ok else None
    except Exception:
        return None

# ───────── Helpers ─────────
def _market_name_normalize(s: str) -> str:
    s = (s or "").lower()
    if "both teams" in s or "btts" in s: return "BTTS"
    if "match winner" in s or "winner" in s or "1x2" in s: return "1X2"
    if "over/under" in s or "total" in s or "goals" in s: return "OU"
    return s

def _fmt_line(line: float) -> str:
    return f"{line}".rstrip("0").rstrip(".")

def _aggregate_price(vals: list[tuple[float,str]], prob_hint: float|None):
    if not vals:
        return None, None
    xs = sorted([o for (o, _) in vals if o > 0])
    if not xs:
        return None, None

    med = statistics.median(xs)
    cleaned = [(o,b) for (o,b) in vals if o <= med * max(1.0, ODDS_OUTLIER_MULT)]
    if not cleaned:
        cleaned = vals

    xs2 = sorted([o for (o, _) in cleaned])
    med2 = statistics.median(xs2)

    if prob_hint is not None and prob_hint > 0:
        fair = 1.0 / max(1e-6, float(prob_hint))
        cap = fair * max(1.0, ODDS_FAIR_MAX_MULT)
        cleaned = [(o,b) for (o,b) in cleaned if o <= cap] or cleaned

    if ODDS_AGGREGATION == "best":
        best = max(cleaned, key=lambda t: t[0])
        return float(best[0]), str(best[1])
    target = med2
    pick = min(cleaned, key=lambda t: abs(t[0] - target))
    return float(pick[0]), f"{pick[1]} (median of {len(xs)})"

def _min_odds_for_market(market: str) -> float:
    if market.startswith("Over/Under"): return MIN_ODDS_OU
    if market == "BTTS": return MIN_ODDS_BTTS
    if market == "1X2": return MIN_ODDS_1X2
    return 1.01

def _ev(prob: float, odds: float) -> float:
    """Return expected value as decimal (0.05 = +5%)."""
    return prob * max(0.0, float(odds)) - 1.0

# ───────── Fetch odds ─────────
def fetch_odds(fid: int, prob_hints: dict[str,float]|None = None) -> dict:
    """
    Aggregated odds map:
      { "BTTS": {...}, "1X2": {...}, "OU_2.5": {...}, ... }
    Prefers /odds/live, falls back to /odds.
    """
    cached = ODDS_CACHE.get(fid)
    if cached and time.time() - cached[0] < 120:
        return cached[1]

    def _fetch(path: str):
        js = _api_get(f"{BASE_URL}/{path}", {"fixture": fid}) or {}
        return js if isinstance(js, dict) else {}

    js = {}
    if ODDS_SOURCE in ("auto", "live"):
        js = _fetch("odds/live")
    if not (js.get("response") or []) and ODDS_SOURCE in ("auto", "prematch"):
        js = _fetch("odds")

    by_market: dict[str, dict[str, list[tuple[float,str]]]] = {}
    try:
        for r in js.get("response", []) or []:
            for bk in (r.get("bookmakers") or []):
                book_name = bk.get("name") or "Book"
                for mkt in (bk.get("bets") or []):
                    mname = _market_name_normalize(mkt.get("name", ""))
                    vals = mkt.get("values") or []
                    if mname == "BTTS":
                        for v in vals:
                            lbl = (v.get("value") or "").lower()
                            if "yes" in lbl:
                                by_market.setdefault("BTTS", {}).setdefault("Yes", []).append((float(v.get("odd") or 0), book_name))
                            elif "no" in lbl:
                                by_market.setdefault("BTTS", {}).setdefault("No", []).append((float(v.get("odd") or 0), book_name))
                    elif mname == "1X2":
                        for v in vals:
                            lbl = (v.get("value") or "").lower()
                            if lbl in ("home","1"):
                                by_market.setdefault("1X2", {}).setdefault("Home", []).append((float(v.get("odd") or 0), book_name))
                            elif lbl in ("away","2"):
                                by_market.setdefault("1X2", {}).setdefault("Away", []).append((float(v.get("odd") or 0), book_name))
                    elif mname == "OU":
                        for v in vals:
                            lbl = (v.get("value") or "").lower()
                            if "over" in lbl or "under" in lbl:
                                try:
                                    ln = float(lbl.split()[-1])
                                    key = f"OU_{_fmt_line(ln)}"
                                    side = "Over" if "over" in lbl else "Under"
                                    by_market.setdefault(key, {}).setdefault(side, []).append((float(v.get("odd") or 0), book_name))
                                except:
                                    pass
    except Exception:
        pass

    out: dict[str, dict[str, dict]] = {}
    for mkey, side_map in by_market.items():
        ok = all(len({b for (_,b) in lst}) >= ODDS_REQUIRE_N_BOOKS for lst in side_map.values())
        if not ok:
            continue
        out[mkey] = {}
        for side, lst in side_map.items():
            hint = None
            if prob_hints:
                if mkey == "BTTS":
                    hint = prob_hints.get("BTTS: Yes") if side == "Yes" else (1.0 - (prob_hints.get("BTTS: Yes") or 0.0))
                elif mkey == "1X2":
                    hint = prob_hints.get("Home Win") if side == "Home" else (prob_hints.get("Away Win") if side == "Away" else None)
                elif mkey.startswith("OU_"):
                    try:
                        ln = float(mkey.split("_", 1)[1])
                        hint = prob_hints.get(f"Over {_fmt_line(ln)} Goals") if side == "Over" else (1.0 - (prob_hints.get(f"Over {_fmt_line(ln)} Goals") or 0.0))
                    except:
                        pass
            ag, label = _aggregate_price(lst, hint)
            if ag is not None:
                out[mkey][side] = {"odds": float(ag), "book": label}

    ODDS_CACHE[fid] = (time.time(), out)
    return out

# ───────── Price gate ─────────
def price_gate(market: str, suggestion: str, fid: int, prob: float|None = None):
    """
    Return (pass, odds, book, ev_pct). Enforces:
      - odds required unless ALLOW_TIPS_WITHOUT_ODDS=1
      - odds between floors and MAX_ODDS_ALL
      - EV >= 0 if prob given
    """
    odds_map = fetch_odds(fid)
    odds, book = None, None

    if market == "BTTS":
        d = odds_map.get("BTTS", {})
        tgt = "Yes" if suggestion.endswith("Yes") else "No"
        if tgt in d: odds, book = d[tgt]["odds"], d[tgt]["book"]
    elif market == "1X2":
        d = odds_map.get("1X2", {})
        tgt = "Home" if suggestion == "Home Win" else ("Away" if suggestion == "Away Win" else None)
        if tgt and tgt in d: odds, book = d[tgt]["odds"], d[tgt]["book"]
    elif market.startswith("Over/Under"):
        try:
            ln = float(suggestion.split()[1])
            d = odds_map.get(f"OU_{_fmt_line(ln)}", {})
            tgt = "Over" if suggestion.startswith("Over") else "Under"
            if tgt in d: odds, book = d[tgt]["odds"], d[tgt]["book"]
        except:
            pass

    if odds is None:
        return (ALLOW_TIPS_WITHOUT_ODDS, odds, book, None)

    if not (_min_odds_for_market(market) <= odds <= MAX_ODDS_ALL):
        return (False, odds, book, None)

    ev_pct = None
    if prob is not None and odds:
        edge = _ev(prob, odds)
        ev_pct = round(edge * 100.0, 1)
        if edge < 0:
            return (False, odds, book, ev_pct)

    return (True, odds, book, ev_pct)
