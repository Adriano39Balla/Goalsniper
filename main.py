#!/usr/bin/env python3
"""
Main orchestrator:
- In-play & prematch predictions
- Odds aggregation, EV gating, thresholds
- Confidence floor, EV floor, min-odds filter
- Advisory locks (no overlap across dynos)
- API-Football rate limiting (<=75k/day ≈ 3.1k/hr)
- Caches (settings/models/odds with TTL)
- DB auto-retries, structured logging
"""

import os, sys, re, time, json, math, logging, random
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import psycopg2
from psycopg2.pool import SimpleConnectionPool
from psycopg2.extras import RealDictCursor

import requests
from requests.adapters import HTTPAdapter, Retry

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ───────────────────────── Logging ───────────────────────── #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("main")

# ───────────────────────── Env & knobs ───────────────────────── #

DATABASE_URL   = os.getenv("DATABASE_URL")
API_FOOTBALL   = os.getenv("API_FOOTBALL_KEY", "")
POOL_MINCONN   = int(os.getenv("POOL_MINCONN", "1"))
POOL_MAXCONN   = int(os.getenv("POOL_MAXCONN", "5"))

CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "75"))  # %
EDGE_MIN_BPS   = float(os.getenv("EDGE_MIN_BPS", "600"))   # basis points
MIN_ODDS       = float(os.getenv("MIN_ODDS", "1.50"))
MARKET_CUTOFFS = {k: int(v) for k,v in (tok.split("=") for tok in os.getenv("MARKET_CUTOFFS","BTTS=75,1X2=80,OU=88").split(","))}

PER_LEAGUE_CAP = int(os.getenv("PER_LEAGUE_CAP","6"))  # per scan
DUP_COOLDOWN   = int(os.getenv("DUP_COOLDOWN_MIN","60")) # minutes
ODDS_TTL       = int(os.getenv("ODDS_TTL","600"))        # seconds
MODELS_TTL     = int(os.getenv("MODELS_TTL","600"))      # seconds
SETTINGS_TTL   = int(os.getenv("SETTINGS_TTL","60"))     # seconds

# Daily rate cap: 75k → ~3.1k/hr → throttle 1 request ~ every 1.2 sec
API_REQ_INTERVAL = float(os.getenv("API_REQ_INTERVAL","1.2"))

# ───────────────────────── DB pool & lock ───────────────────────── #

POOL: Optional[SimpleConnectionPool] = None

def _init_pool() -> None:
    global POOL
    if POOL: return
    dsn = DATABASE_URL
    if "sslmode=" not in dsn:
        dsn += ("&" if "?" in dsn else "?") + "sslmode=require"
    POOL = SimpleConnectionPool(
        minconn=POOL_MINCONN, maxconn=POOL_MAXCONN, dsn=dsn
    )
    log.info("[DB] Pool initialized (%d–%d)", POOL_MINCONN, POOL_MAXCONN)

def _reset_pool() -> None:
    global POOL
    if POOL:
        try: POOL.closeall()
        except: pass
    POOL = None
    _init_pool()

def _run_with_pg_lock(lock_key: int, fn, *a, **kw):
    """Ensure single job per lock_key across dynos"""
    max_attempts = 3
    for attempt in range(1,max_attempts+1):
        _init_pool()
        conn = POOL.getconn()
        try:
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute("SELECT pg_try_advisory_lock(%s)", (lock_key,))
                locked = cur.fetchone()[0]
                if not locked:
                    log.info("[LOCK] key %s busy", lock_key)
                    return None
                try:
                    return fn(conn, *a, **kw)
                finally:
                    cur.execute("SELECT pg_advisory_unlock(%s)", (lock_key,))
        except Exception as e:
            log.warning("[DB] attempt %d failed: %s", attempt, e)
            _reset_pool()
            time.sleep(attempt*1.5)
        finally:
            try: POOL.putconn(conn)
            except: pass
    raise RuntimeError("PG lock runner failed")

# ───────────────────────── Caches ───────────────────────── #

class _TTLCache:
    def __init__(self, ttl:int): self.ttl=ttl; self.d:Dict[Any,Tuple[float,Any]]={}
    def get(self,k):
        v=self.d.get(k); 
        if not v: return None
        if time.time()>v[0]: self.d.pop(k,None); return None
        return v[1]
    def set(self,k,v): self.d[k]=(time.time()+self.ttl,v)
    def clear(self): self.d.clear()

_SETTINGS_CACHE=_TTLCache(SETTINGS_TTL)
_MODELS_CACHE=_TTLCache(MODELS_TTL)
_ODDS_CACHE=_TTLCache(ODDS_TTL)

# ───────────────────────── Odds aggregation ───────────────────────── #

def _trimmed_median(xs:List[float], trim:float=0.2)->Optional[float]:
    arr=np.array([x for x in xs if x and x>1.0],float)
    if arr.size==0: return None
    k=max(1,int(len(arr)*trim))
    arr.sort(); arr=arr[k:-k] if arr.size>2*k else arr
    return float(np.median(arr))

def _aggregate_price(prices:List[Tuple[float,str]])->Tuple[Optional[float],Optional[str]]:
    vals=[p for p,_ in prices if p and p>1.0]
    if not vals: return None,None
    med=_trimmed_median(vals)
    if med is None: return None,None
    return med, f"median({len(vals)})"

# ───────────────────────── API-Football client (global throttle) ───────────────────────── #

BASE_URL = "https://v3.football.api-sports.io"
HEADERS  = {"x-apisports-key": API_FOOTBALL, "Accept": "application/json"}

_session = requests.Session()
_session.mount("https://", HTTPAdapter(max_retries=Retry(
    total=3, backoff_factor=0.8,
    status_forcelist=[429,500,502,503,504],
    respect_retry_after_header=True,
)))
_next_req_at = 0.0

def _throttle() -> None:
    global _next_req_at
    now = time.time()
    if now < _next_req_at:
        time.sleep(max(0.0, _next_req_at - now))
    _next_req_at = time.time() + API_REQ_INTERVAL * (1.0 + random.random()*0.25)  # jitter

def _api_get(path: str, params: dict, timeout: int = 12) -> Optional[dict]:
    """Centralized GET with throttle, retries and 429 jitter."""
    if not API_FOOTBALL:
        return None
    _throttle()
    url = f"{BASE_URL.rstrip('/')}/{path.lstrip('/')}"
    try:
        r = _session.get(url, headers=HEADERS, params=params, timeout=timeout)
        if r.status_code == 429:
            # burst safety: exponential backoff + jitter
            time.sleep(1.0 + random.random())
            return None
        if not r.ok:
            return None
        js = r.json()
        return js if isinstance(js, dict) else None
    except Exception:
        return None

# ───────────────────────── Telegram (optional) ───────────────────────── #

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")
def send_telegram(text: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return False
    try:
        _throttle()  # telegram also counts against pings budget; share throttle
        r = _session.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                          data={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML",
                                "disable_web_page_preview": True},
                          timeout=10)
        return bool(r.ok)
    except Exception:
        return False

# ───────────────────────── Settings helpers ───────────────────────── #

def _get_setting(conn, key: str) -> Optional[str]:
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT value FROM settings WHERE key=%s", (key,))
            r = cur.fetchone()
            return r[0] if r else None
    except Exception:
        return None

def _set_setting(conn, key: str, value: str) -> None:
    with conn.cursor() as cur:
        cur.execute("""INSERT INTO settings(key,value) VALUES(%s,%s)
                       ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value""", (key, value))

def get_setting_cached(key: str) -> Optional[str]:
    v = _SETTINGS_CACHE.get(key)
    if v is not None:
        return v
    _init_pool()
    conn = POOL.getconn()
    try:
        val = _get_setting(conn, key)
        _SETTINGS_CACHE.set(key, val)
        return val
    finally:
        POOL.putconn(conn)

def set_setting_cached(key: str, value: str) -> None:
    _init_pool()
    conn = POOL.getconn()
    try:
        _set_setting(conn, key, value)
        _SETTINGS_CACHE.set(key, value)
    finally:
        POOL.putconn(conn)

# ───────────────────────── Model blobs from settings ───────────────────────── #

MODEL_KEYS_ORDER = ["model_v2:{name}", "model_latest:{name}", "model:{name}"]

def load_model_from_settings(name: str) -> Optional[Dict[str, Any]]:
    cached = _MODELS_CACHE.get(name)
    if cached is not None: return cached
    mdl = None
    _init_pool()
    conn = POOL.getconn()
    try:
        for pat in MODEL_KEYS_ORDER:
            raw = _get_setting(conn, pat.format(name=name))
            if not raw:
                continue
            try:
                tmp = json.loads(raw)
                if not isinstance(tmp, dict):
                    continue
                tmp.setdefault("intercept", 0.0)
                tmp.setdefault("weights", {})
                cal = tmp.get("calibration") or {}
                if isinstance(cal, dict):
                    cal.setdefault("method", "sigmoid"); cal.setdefault("a", 1.0); cal.setdefault("b", 0.0)
                    tmp["calibration"] = cal
                if "stack" not in tmp:
                    tmp["stack"] = {"a_model": 1.0, "a_book": 0.0, "b": 0.0}  # identity stack by default
                mdl = tmp; break
            except Exception as e:
                log.warning("[MODEL] parse %s failed: %s", name, e)
        if mdl is not None:
            _MODELS_CACHE.set(name, mdl)
        return mdl
    finally:
        POOL.putconn(conn)

# ───────────────────────── Probability helpers ───────────────────────── #

EPS = 1e-12
def _sigmoid(x: float) -> float:
    x = float(x)
    if x < -50: return 1e-22
    if x > 50:  return 1 - 1e-22
    return 1.0/(1.0 + math.exp(-x))

def _logit(p: float) -> float:
    p = max(EPS, min(1.0-EPS, float(p)))
    return math.log(p/(1.0-p))

def _linpred(feat: Dict[str, float], weights: Dict[str, float], intercept: float) -> float:
    s = float(intercept or 0.0)
    for k, w in (weights or {}).items():
        s += float(w or 0.0) * float(feat.get(k, 0.0))
    return s

def _pick_bin_params(cal_bins: list|None, minute: float) -> Optional[Tuple[float,float]]:
    if not cal_bins: return None
    try:
        m=float(minute or 0.0)
        for b in cal_bins:
            lo=float((b or {}).get("lo", -1e9)); hi=float((b or {}).get("hi", 1e9))
            if lo <= m < hi:
                return float(b.get("a", 1.0)), float(b.get("b", 0.0))
    except Exception:
        pass
    return None

def _calibrate_global(p: float, cal: Dict[str, Any]) -> float:
    method=(cal or {}).get("method","sigmoid").lower()
    a=float((cal or {}).get("a",1.0)); b=float((cal or {}).get("b",0.0))
    if method == "platt":
        return _sigmoid(a*_logit(p) + b)
    # 'sigmoid' means we already operate in prob space; emulate platt on logits anyway
    return _sigmoid(a*_logit(p) + b)

def _score_prob(feat: Dict[str,float], mdl: Dict[str,Any]) -> float:
    p = _sigmoid(_linpred(feat, mdl.get("weights", {}), float(mdl.get("intercept", 0.0))))
    p = float(max(0.0, min(1.0, p)))
    cal_bins = mdl.get("calibration_by_minute")
    bin_ab = _pick_bin_params(cal_bins, feat.get("minute", 0.0))
    if bin_ab:
        a,b = bin_ab
        return float(_sigmoid(a*_logit(p) + b))
    return float(_calibrate_global(p, mdl.get("calibration") or {}))

def _implied_prob_from_odds(odds: Optional[float]) -> Optional[float]:
    try:
        o=float(odds or 0.0)
        if o<=1.0: return None
        return float(max(0.01, min(0.99, 1.0/o)))
    except Exception:
        return None

def _apply_stack(p_model: float, mdl: Optional[Dict[str,Any]], p_book: Optional[float]) -> float:
    if not mdl: return float(p_model)
    stack = mdl.get("stack") or {}
    a_m = float(stack.get("a_model", 1.0))
    a_b = float(stack.get("a_book", 0.0))
    b0  = float(stack.get("b", 0.0))
    if p_book is None or a_b == 0.0:
        return float(p_model)
    z = a_m*_logit(p_model) + a_b*_logit(p_book) + b0
    return float(_sigmoid(z))

# ───────────────────────── Feature extraction (in-play) ───────────────────────── #

INPLAY_STATUSES = {"1H","HT","2H","ET","BT","P"}

_BLOCK_RE = re.compile(r'\b(u1[7-9]|u2[0-3]|youth|junior|reserve|res\.|friendlies|friendly)\b', re.I)

def _blocked_league(league_obj: dict) -> bool:
    txt = f"{(league_obj or {}).get('country','')} {(league_obj or {}).get('name','')} {(league_obj or {}).get('type','')}".lower()
    if _BLOCK_RE.search(txt): return True
    deny = {x.strip() for x in os.getenv("LEAGUE_DENY_IDS","").split(",") if x.strip()}
    return str((league_obj or {}).get("id") or "") in deny

def _pos_pct(v) -> float:
    try: return float(str(v).replace("%","").strip() or 0)
    except: return 0.0

def _num(v) -> float:
    try:
        if isinstance(v,str) and v.endswith("%"): return float(v[:-1])
        return float(v or 0)
    except: return 0.0

def extract_features_from_live(m: dict) -> Dict[str,float]:
    home = (m.get("teams") or {}).get("home", {}).get("name", "")
    away = (m.get("teams") or {}).get("away", {}).get("name", "")
    gh = int((m.get("goals") or {}).get("home") or 0)
    ga = int((m.get("goals") or {}).get("away") or 0)
    minute = int(((m.get("fixture") or {}).get("status") or {}).get("elapsed") or 0)

    # stats -> dict by team
    stats = {}
    for s in (m.get("statistics") or []):
        tname = (s.get("team") or {}).get("name")
        if tname:
            stats[tname] = { (i.get("type") or ""): i.get("value") for i in (s.get("statistics") or []) }

    sh = stats.get(home, {}) or {}
    sa = stats.get(away, {}) or {}

    xg_h = _num(sh.get("Expected Goals", 0)); xg_a = _num(sa.get("Expected Goals", 0))
    sot_h = _num(sh.get("Shots on Target", sh.get("Shots on Goal", 0))); sot_a = _num(sa.get("Shots on Target", sa.get("Shots on Goal", 0)))
    sh_total_h = _num(sh.get("Total Shots", sh.get("Shots Total", 0))); sh_total_a = _num(sa.get("Total Shots", sa.get("Shots Total", 0)))
    cor_h = _num(sh.get("Corner Kicks", 0)); cor_a = _num(sa.get("Corner Kicks", 0))
    pos_h = _pos_pct(sh.get("Ball Possession", 0)); pos_a = _pos_pct(sa.get("Ball Possession", 0))

    red_h = red_a = yellow_h = yellow_a = 0
    for ev in (m.get("events") or []):
        if str(ev.get("type","")).lower() == "card":
            d = (ev.get("detail","") or "").lower()
            t = (ev.get("team") or {}).get("name") or ""
            if "yellow" in d and "second" not in d:
                if t == home: yellow_h += 1
                elif t == away: yellow_a += 1
            if "red" in d or "second yellow" in d:
                if t == home: red_h += 1
                elif t == away: red_a += 1

    return {
        "minute": float(minute),
        "goals_h": float(gh), "goals_a": float(ga),
        "goals_sum": float(gh+ga), "goals_diff": float(gh-ga),
        "xg_h": float(xg_h), "xg_a": float(xg_a),
        "xg_sum": float(xg_h + xg_a), "xg_diff": float(xg_h - xg_a),
        "sot_h": float(sot_h), "sot_a": float(sot_a), "sot_sum": float(sot_h + sot_a),
        "sh_total_h": float(sh_total_h), "sh_total_a": float(sh_total_a),
        "cor_h": float(cor_h), "cor_a": float(cor_a), "cor_sum": float(cor_h + cor_a),
        "pos_h": float(pos_h), "pos_a": float(pos_a), "pos_diff": float(pos_h - pos_a),
        "red_h": float(red_h), "red_a": float(red_a), "red_sum": float(red_h + red_a),
        "yellow_h": float(yellow_h), "yellow_a": float(yellow_a),
    }

# ───────────────────────── Fetchers with caching (save pings) ───────────────────────── #

def _fetch_live_matches() -> List[dict]:
    """1 call per scan. We enrich each match minimally to save pings."""
    js = _api_get("fixtures", {"live": "all"}) or {}
    arr = js.get("response") or []
    out = []
    for m in arr:
        if _blocked_league(m.get("league") or {}): 
            continue
        st = ((m.get("fixture",{}) or {}).get("status",{}) or {})
        elapsed = st.get("elapsed"); short=(st.get("short") or "").upper()
        if elapsed is None or elapsed > 120 or short not in INPLAY_STATUSES:
            continue
        # fetch statistics & events only for promising matches:
        # Heuristic: save pings by enriching only when elapsed >= 10 and goals or cards or corners present
        fid = int((m.get("fixture") or {}).get("id") or 0)
        if fid <= 0:
            continue
        enrich = False
        g = m.get("goals") or {}
        if (g.get("home") or 0) + (g.get("away") or 0) > 0: enrich = True
        if (elapsed or 0) >= 15: enrich = True  # allow stats after 15' only (rate saver)
        if enrich:
            # 2 extra pings per enriched match (statistics + events) — consider budget!
            st_js = _api_get("fixtures/statistics", {"fixture": fid}) or {}
            ev_js = _api_get("fixtures/events", {"fixture": fid}) or {}
            m["statistics"] = st_js.get("response", []) if isinstance(st_js, dict) else []
            m["events"] = ev_js.get("response", []) if isinstance(ev_js, dict) else []
        else:
            m["statistics"] = []
            m["events"] = []
        out.append(m)
    return out

def _fetch_prematch_of_day() -> List[dict]:
    """Usually 1–2 calls depending on timezones; keep it lean."""
    today = datetime.now().date().strftime("%Y-%m-%d")
    js = _api_get("fixtures", {"date": today}) or {}
    arr = js.get("response") or []
    return [r for r in arr if (((r.get("fixture") or {}).get("status") or {}).get("short") or "").upper() == "NS"
            and not _blocked_league(r.get("league") or {})]

# ───────────────────────── Odds (aggregated) with cache ───────────────────────── #

def _market_name_normalize(s: str) -> str:
    s=(s or "").lower()
    if "both teams" in s or "btts" in s: return "BTTS"
    if "match winner" in s or "winner" in s or "1x2" in s: return "1X2"
    if "over/under" in s or "total" in s or "goals" in s: return "OU"
    return s

def _fmt_line(line: float) -> str:
    return f"{line}".rstrip("0").rstrip(".")

def _parse_ou_line_from_suggestion(s: str) -> Optional[float]:
    m = re.search(r'(\d+(?:\.\d+)?)', s or '')
    return float(m.group(1)) if m else None

def odds_for_fixture(fid: int) -> dict:
    cached = _ODDS_CACHE.get(("fixture_odds", fid))
    if cached is not None:
        return cached

    # Try live odds first (1 ping). If empty, fall back to prematch odds (another ping).
    js = _api_get("odds/live", {"fixture": fid}) or {}
    if not (js.get("response") or []):
        js = _api_get("odds", {"fixture": fid}) or {}

    by_market: dict[str, dict[str, List[Tuple[float,str]]]] = {}
    try:
        for r in js.get("response", []) or []:
            for bk in (r.get("bookmakers") or []):
                book_name = bk.get("name") or "Book"
                for mkt in (bk.get("bets") or []):
                    mname = _market_name_normalize(mkt.get("name", ""))
                    vals = mkt.get("values") or []
                    if mname == "BTTS":
                        for v in vals:
                            lbl = (v.get("value") or "").strip().lower()
                            if "yes" in lbl:
                                by_market.setdefault("BTTS", {}).setdefault("Yes", []).append((float(v.get("odd") or 0), book_name))
                            elif "no" in lbl:
                                by_market.setdefault("BTTS", {}).setdefault("No", []).append((float(v.get("odd") or 0), book_name))
                    elif mname == "1X2":
                        for v in vals:
                            lbl = (v.get("value") or "").strip().lower()
                            if lbl in ("home","1"):
                                by_market.setdefault("1X2", {}).setdefault("Home", []).append((float(v.get("odd") or 0), book_name))
                            elif lbl in ("away","2"):
                                by_market.setdefault("1X2", {}).setdefault("Away", []).append((float(v.get("odd") or 0), book_name))
                    elif mname == "OU":
                        for v in vals:
                            lbl = (v.get("value") or "").lower()
                            if ("over" in lbl) or ("under" in lbl):
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
        out[mkey] = {}
        for side, rows in side_map.items():
            # aggregate with trimmed median & floor
            agg, label = _aggregate_price(rows)
            if agg and agg >= MIN_ODDS:
                out[mkey][side] = {"odds": float(agg), "book": label}
    _ODDS_CACHE.set(("fixture_odds", fid), out)
    return out

# ───────────────────────── Gates & thresholds ───────────────────────── #

def _market_family(market_text: str) -> str:
    s = market_text.upper()
    if s.startswith("OVER/UNDER") or "OVER/UNDER" in s: return "OU"
    if s == "BTTS": return "BTTS"
    if s == "1X2": return "1X2"
    if s.startswith("PRE "): return _market_family(s[4:])
    return s

def _min_odds_for_market(market: str) -> float:
    fam = _market_family(market)
    if fam in {"BTTS","1X2","OU"}:
        return MIN_ODDS
    return MIN_ODDS

def _price_gate(market_text: str, suggestion: str, fid: int) -> Tuple[bool, Optional[float], Optional[str]]:
    odds_map = odds_for_fixture(fid) if API_FOOTBALL else {}
    odds = None; book = None
    if market_text=="BTTS":
        d=odds_map.get("BTTS",{})
        tgt="Yes" if suggestion.endswith("Yes") else "No"
        if tgt in d: odds, book = d[tgt]["odds"], d[tgt]["book"]
    elif market_text=="1X2":
        d=odds_map.get("1X2",{})
        tgt="Home" if suggestion=="Home Win" else ("Away" if suggestion=="Away Win" else None)
        if tgt and tgt in d: odds, book = d[tgt]["odds"], d[tgt]["book"]
    elif market_text.startswith("Over/Under"):
        ln = _parse_ou_line_from_suggestion(suggestion)
        if ln is not None:
            d = odds_map.get(f"OU_{_fmt_line(ln)}", {})
            tgt = "Over" if suggestion.startswith("Over") else "Under"
            if tgt in d: odds, book = d[tgt]["odds"], d[tgt]["book"]

    if odds is None:
        return False, None, None
    if not (_min_odds_for_market(market_text) <= odds <= 20.0):
        return False, odds, book
    return True, odds, book

def _ev(prob: float, odds: float) -> float:
    """EV as decimal (0.06 = +6%)."""
    return prob*odds - 1.0

def _get_market_threshold(m: str) -> float:
    # never allow below global floor
    floor = float(os.getenv("CONF_THRESHOLD_GLOBAL_FLOOR", CONF_THRESHOLD))
    try:
        v = get_setting_cached(f"conf_threshold:{m}")
        thr = float(v) if v is not None else float(CONF_THRESHOLD)
        return max(floor, thr)
    except Exception:
        return float(CONF_THRESHOLD)

# ───────────────────────── Dedup cooldown ───────────────────────── #

def recently_tipped(conn, match_id: int, cooldown_min: int = DUP_COOLDOWN) -> bool:
    cutoff = int(time.time()) - cooldown_min*60
    with conn.cursor() as cur:
        cur.execute("""SELECT 1 FROM tips 
                       WHERE match_id=%s AND created_ts >= %s AND suggestion <> 'HARVEST' 
                       LIMIT 1""", (int(match_id), int(cutoff)))
        return bool(cur.fetchone())

# ───────────────────────── Sane candidate checks ───────────────────────── #

def _candidate_is_sane(sug: str, feat: Dict[str,float]) -> bool:
    gh = int(feat.get("goals_h", 0)); ga = int(feat.get("goals_a", 0)); total = gh + ga
    if sug.startswith("Over"):
        ln = _parse_ou_line_from_suggestion(sug)
        return (ln is not None) and (total < ln)
    if sug.startswith("Under"):
        ln = _parse_ou_line_from_suggestion(sug)
        # conservative but not too strict (avoid dead unders)
        return (ln is not None) and (total <= ln - 0.5)
    if sug == "BTTS: Yes" and (gh > 0 and ga > 0):
        return False
    return True

import flask
from flask import Flask, jsonify, request, abort
from zoneinfo import ZoneInfo

# ───────────────────────── Globals / app ───────────────────────── #

BERLIN_TZ = ZoneInfo("Europe/Berlin")
app = Flask(__name__)

DAILY_DIGEST_WINDOW_DAYS = int(os.getenv("DAILY_DIGEST_WINDOW_DAYS", "1"))
MOTD_MIN_EV_BPS = int(os.getenv("MOTD_MIN_EV_BPS", str(int(EDGE_MIN_BPS))))

PREDICTIONS_PER_MATCH = int(os.getenv("PREDICTIONS_PER_MATCH","1"))
TIP_MIN_MINUTE       = int(os.getenv("TIP_MIN_MINUTE","12"))
TOTAL_MATCH_MINUTES  = int(os.getenv("TOTAL_MATCH_MINUTES","95"))
RUN_SCHEDULER        = str(os.getenv("RUN_SCHEDULER","1")).lower() not in ("0","false","no")

ADMIN_API_KEY        = os.getenv("ADMIN_API_KEY","")

OU_LINES = [float(t.strip()) for t in os.getenv("OU_LINES","2.5,3.5").split(",") if t.strip()]

ALLOWED_SUGGESTIONS = {"BTTS: Yes", "BTTS: No", "Home Win", "Away Win"}
def _fmt_line(line: float) -> str: return f"{line}".rstrip("0").rstrip(".")
for _ln in OU_LINES:
    s = _fmt_line(_ln)
    ALLOWED_SUGGESTIONS.add(f"Over {s} Goals")
    ALLOWED_SUGGESTIONS.add(f"Under {s} Goals")

# ───────────────────────── DB schema & helpers ───────────────────────── #

def init_db(conn=None):
    own_conn = conn is None
    if own_conn:
        _init_pool(); conn = POOL.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS tips (
                match_id BIGINT,
                league_id BIGINT,
                league TEXT,
                home TEXT, away TEXT,
                market TEXT,
                suggestion TEXT,
                confidence DOUBLE PRECISION,
                confidence_raw DOUBLE PRECISION,
                score_at_tip TEXT,
                minute INTEGER,
                created_ts BIGINT,
                odds DOUBLE PRECISION,
                book TEXT,
                ev_pct DOUBLE PRECISION,
                sent_ok INTEGER DEFAULT 1,
                PRIMARY KEY (match_id, created_ts)
            )""")
            cur.execute("""
            CREATE TABLE IF NOT EXISTS tip_snapshots (
                match_id BIGINT,
                created_ts BIGINT,
                payload TEXT,
                PRIMARY KEY (match_id, created_ts)
            )""")
            cur.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id SERIAL PRIMARY KEY,
                match_id BIGINT UNIQUE,
                verdict INTEGER,
                created_ts BIGINT
            )""")
            cur.execute("""CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)""")
            cur.execute("""
            CREATE TABLE IF NOT EXISTS match_results (
                match_id BIGINT PRIMARY KEY,
                final_goals_h INTEGER,
                final_goals_a INTEGER,
                btts_yes INTEGER,
                updated_ts BIGINT
            )""")
            cur.execute("""
            CREATE TABLE IF NOT EXISTS odds_history (
                match_id BIGINT,
                captured_ts BIGINT,
                market TEXT,
                selection TEXT,
                odds DOUBLE PRECISION,
                book TEXT,
                PRIMARY KEY (match_id, market, selection, captured_ts)
            )""")
            cur.execute("""CREATE INDEX IF NOT EXISTS idx_odds_hist_match ON odds_history (match_id, captured_ts DESC)""")
            cur.execute("""
            CREATE TABLE IF NOT EXISTS lineups (
                match_id BIGINT PRIMARY KEY,
                created_ts BIGINT,
                payload TEXT
            )""")
            # Helpful indexes
            cur.execute("""CREATE INDEX IF NOT EXISTS idx_tips_created ON tips (created_ts DESC)""")
            cur.execute("""CREATE INDEX IF NOT EXISTS idx_tips_match ON tips (match_id)""")
            cur.execute("""CREATE INDEX IF NOT EXISTS idx_tips_sent ON tips (sent_ok, created_ts DESC)""")
            cur.execute("""CREATE INDEX IF NOT EXISTS idx_snap_by_match ON tip_snapshots (match_id, created_ts DESC)""")
            cur.execute("""CREATE INDEX IF NOT EXISTS idx_results_updated ON match_results (updated_ts DESC)""")
        log.info("[DB] schema ensured")
    finally:
        if own_conn:
            POOL.putconn(conn)

# ───────────────────────── Snapshots ───────────────────────── #

def save_harvest_snapshot(conn, m: dict, feat: Dict[str,float]) -> None:
    fx = m.get("fixture") or {}; lg = m.get("league") or {}; teams = m.get("teams") or {}
    fid = int(fx.get("id") or 0)
    if not fid: return
    league_id = int(lg.get("id") or 0); league = f"{lg.get('country','')} - {lg.get('name','')}".strip(" -")
    home = (teams.get("home") or {}).get("name",""); away = (teams.get("away") or {}).get("name","")
    gh = int((m.get("goals") or {}).get("home") or 0); ga = int((m.get("goals") or {}).get("away") or 0)
    minute = int(feat.get("minute", 0))
    snap = {
        "minute": minute, "gh": gh, "ga": ga, "league_id": league_id,
        "market": "HARVEST", "suggestion": "HARVEST", "confidence": 0,
        "stat": {
            "xg_h": feat.get("xg_h",0.0), "xg_a": feat.get("xg_a",0.0), "xg_sum": feat.get("xg_sum",0.0), "xg_diff": feat.get("xg_diff",0.0),
            "sot_h": feat.get("sot_h",0.0), "sot_a": feat.get("sot_a",0.0), "sot_sum": feat.get("sot_sum",0.0),
            "cor_h": feat.get("cor_h",0.0), "cor_a": feat.get("cor_a",0.0), "cor_sum": feat.get("cor_sum",0.0),
            "pos_h": feat.get("pos_h",0.0), "pos_a": feat.get("pos_a",0.0), "pos_diff": feat.get("pos_diff",0.0),
            "red_h": feat.get("red_h",0.0), "red_a": feat.get("red_a",0.0), "red_sum": feat.get("red_sum",0.0),
            "sh_total_h": feat.get("sh_total_h",0.0), "sh_total_a": feat.get("sh_total_a",0.0),
            "yellow_h": feat.get("yellow_h",0.0), "yellow_a": feat.get("yellow_a",0.0),
        }
    }
    payload = json.dumps(snap, separators=(",",":"), ensure_ascii=False)
    now = int(time.time())
    with conn.cursor() as cur:
        cur.execute("""INSERT INTO tip_snapshots(match_id, created_ts, payload)
                       VALUES (%s,%s,%s) ON CONFLICT (match_id, created_ts) DO NOTHING""",
                    (fid, now, payload))
        # also write a HARVEST marker row into tips for later joins
        cur.execute("""INSERT INTO tips(match_id,league_id,league,home,away,market,suggestion,confidence,confidence_raw,
                                        score_at_tip,minute,created_ts,sent_ok)
                       VALUES (%s,%s,%s,%s,%s,'HARVEST','HARVEST',0,0,%s,%s,%s,1)""",
                    (fid, league_id, league, home, away, f"{gh}-{ga}", minute, now))

# ───────────────────────── In-play scan ───────────────────────── #

def _within_cutoff(minute:int, market:str)->bool:
    fam = _market_family(market)
    cutoff = MARKET_CUTOFFS.get(fam, TOTAL_MATCH_MINUTES-5)
    return int(minute or 0) <= int(cutoff)

def production_scan(conn=None) -> Tuple[int,int]:
    own_conn = conn is None
    if own_conn:
        _init_pool(); conn = POOL.getconn()
    saved = 0
    try:
        matches = _fetch_live_matches()
        live_seen = len(matches)
        now_ts = int(time.time())
        per_league_counter: Dict[int,int] = {}
        for m in matches:
            try:
                fid = int((m.get("fixture") or {}).get("id") or 0)
                if fid <= 0: continue
                if recently_tipped(conn, fid, DUP_COOLDOWN):
                    continue

                # only evaluate after 12'
                feat = extract_features_from_live(m)
                minute = int(feat.get("minute", 0))
                if minute < TIP_MIN_MINUTE:
                    # light-weight harvest to build data, but do not enrich further
                    continue

                # store harvest snapshot sparsely (every 3 minutes from 15')
                if minute >= 15 and (minute % 3 == 0):
                    try: save_harvest_snapshot(conn, m, feat)
                    except Exception: pass

                lg = m.get("league") or {}; league_id = int(lg.get("id") or 0)
                league = f"{lg.get('country','')} - {lg.get('name','')}".strip(" -")
                teams = m.get("teams") or {}
                home = (teams.get("home") or {}).get("name","")
                away = (teams.get("away") or {}).get("name","")
                score = f"{int((m.get('goals') or {}).get('home') or 0)}-{int((m.get('goals') or {}).get('away') or 0)}"

                candidates: List[Tuple[str, str, float, Optional[Dict[str,Any]]]] = []

                # OU
                for line in OU_LINES:
                    mdl = load_model_from_settings(f"OU_{_fmt_line(line)}") or load_model_from_settings("O25") if abs(line-2.5)<1e-6 else None
                    if not mdl: continue
                    mk = f"Over/Under {_fmt_line(line)}"
                    thr = _get_market_threshold(mk)

                    p_over = _score_prob(feat, mdl)
                    sug_over = f"Over {_fmt_line(line)} Goals"
                    if p_over*100 >= thr and _within_cutoff(minute, mk) and _candidate_is_sane(sug_over, feat):
                        candidates.append((mk, sug_over, p_over, mdl))

                    p_under = 1.0 - p_over
                    sug_under = f"Under {_fmt_line(line)} Goals"
                    if p_under*100 >= thr and _within_cutoff(minute, mk) and _candidate_is_sane(sug_under, feat):
                        candidates.append((mk, sug_under, p_under, mdl))

                # BTTS
                mdl_b = load_model_from_settings("BTTS_YES")
                if mdl_b:
                    thr = _get_market_threshold("BTTS")
                    p_yes = _score_prob(feat, mdl_b)
                    if p_yes*100 >= thr and _within_cutoff(minute, "BTTS") and _candidate_is_sane("BTTS: Yes", feat):
                        candidates.append(("BTTS", "BTTS: Yes", p_yes, mdl_b))
                    p_no = 1.0 - p_yes
                    if p_no*100 >= thr and _within_cutoff(minute, "BTTS") and _candidate_is_sane("BTTS: No", feat):
                        candidates.append(("BTTS", "BTTS: No", p_no, mdl_b))

                # 1X2 (draw suppressed)
                mh = load_model_from_settings("WLD_HOME")
                md = load_model_from_settings("WLD_DRAW")
                ma = load_model_from_settings("WLD_AWAY")
                if mh and md and ma:
                    thr = _get_market_threshold("1X2")
                    ph = _score_prob(feat, mh)
                    pd = _score_prob(feat, md)
                    pa = _score_prob(feat, ma)
                    s = max(EPS, ph+pd+pa); ph, pa = ph/s, pa/s
                    if ph*100 >= thr and _within_cutoff(minute, "1X2"):
                        candidates.append(("1X2", "Home Win", ph, mh))
                    if pa*100 >= thr and _within_cutoff(minute, "1X2"):
                        candidates.append(("1X2", "Away Win", pa, ma))

                if not candidates:
                    continue

                # Odds lookup (cached), EV gate + stacking
                odds_map = odds_for_fixture(fid)
                ranked: List[Tuple[str,str,float,Optional[float],Optional[str],float,float]] = []

                for mk, sug, p_model, mdl_used in candidates:
                    if sug not in ALLOWED_SUGGESTIONS: 
                        continue

                    ok_odds, odds, book = _price_gate(mk, sug, fid)
                    if not ok_odds or odds is None:
                        continue

                    p_book = _implied_prob_from_odds(odds)
                    p_final = _apply_stack(p_model, mdl_used, p_book)
                    edge = _ev(p_final, odds)
                    ev_pct = round(edge*100, 1)
                    if int(round(edge*10000)) < EDGE_MIN_BPS:
                        continue

                    rank_score = (p_final ** 1.2) * (1 + max(0.0, ev_pct)/100.0)
                    ranked.append((mk, sug, p_final, odds, book, ev_pct, rank_score))

                if not ranked:
                    continue

                ranked.sort(key=lambda x: x[6], reverse=True)

                per_match = 0
                base_now = int(time.time())

                for idx, (market_txt, suggestion, p_final, odds, book, ev_pct, _) in enumerate(ranked):
                    if PER_LEAGUE_CAP > 0 and per_league_counter.get(league_id, 0) >= PER_LEAGUE_CAP:
                        continue
                    created_ts = base_now + idx
                    prob_pct = round(float(p_final) * 100.0, 1)
                    with conn.cursor() as cur:
                        cur.execute("""
                        INSERT INTO tips(match_id,league_id,league,home,away,market,suggestion,confidence,confidence_raw,
                                         score_at_tip,minute,created_ts,odds,book,ev_pct,sent_ok)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,0)
                        """, (fid, league_id, league, home, away, market_txt, suggestion,
                              float(prob_pct), float(p_final), score, minute, created_ts,
                              float(odds), (book or None), float(ev_pct)))
                    # structured log + telegram
                    log.info("[TIP] mid=%s mk=%s sug=%s prob=%.3f odds=%.2f ev=%+.1f%% min=%d lg=%s",
                             fid, market_txt, suggestion, p_final, odds, ev_pct, minute, league)
                    msg = (f"⚽️ <b>New Tip!</b>\n"
                           f"<b>Match:</b> {home} vs {away}\n"
                           f"🕒 <b>Minute:</b> {minute}' | <b>Score:</b> {score}\n"
                           f"<b>Tip:</b> {suggestion}\n"
                           f"📈 <b>Confidence:</b> {prob_pct:.1f}%\n"
                           f"💰 <b>Odds:</b> {odds:.2f} @ {book or 'Book'} • <b>EV:</b> {ev_pct:+.1f}%\n"
                           f"🏆 <b>League:</b> {league}")
                    ok = send_telegram(msg)
                    if ok:
                        with conn.cursor() as cur:
                            cur.execute("UPDATE tips SET sent_ok=1 WHERE match_id=%s AND created_ts=%s", (fid, created_ts))
                    saved += 1
                    per_match += 1
                    per_league_counter[league_id] = per_league_counter.get(league_id, 0) + 1
                    if per_match >= max(1, PREDICTIONS_PER_MATCH):
                        break
            except Exception as e:
                log.exception("[SCAN] match failed: %s", e)
                continue
        log.info("[SCAN] saved=%d live_seen=%d", saved, live_seen)
        return saved, live_seen
    finally:
        if own_conn:
            POOL.putconn(conn)

# ───────────────────────── Prematch scan ───────────────────────── #

def extract_prematch_features(fx: dict) -> Dict[str,float]:
    teams = fx.get("teams") or {}
    th = (teams.get("home") or {}).get("id")
    ta = (teams.get("away") or {}).get("id")
    base = {
        # prematch simple priors (placeholders; learnable via snapshots)
        "pm_ov25_h":0.0,"pm_ov35_h":0.0,"pm_btts_h":0.0,
        "pm_ov25_a":0.0,"pm_ov35_a":0.0,"pm_btts_a":0.0,
        "pm_ov25_h2h":0.0,"pm_ov35_h2h":0.0,"pm_btts_h2h":0.0,
        "minute":0.0,"goals_h":0.0,"goals_a":0.0,"goals_sum":0.0,"goals_diff":0.0,
        "xg_h":0.0,"xg_a":0.0,"xg_sum":0.0,"xg_diff":0.0,"sot_h":0.0,"sot_a":0.0,"sot_sum":0.0,
        "sh_total_h":0.0,"sh_total_a":0.0,"cor_h":0.0,"cor_a":0.0,"cor_sum":0.0,
        "pos_h":0.0,"pos_a":0.0,"pos_diff":0.0,"red_h":0.0,"red_a":0.0,"red_sum":0.0,
        "yellow_h":0.0,"yellow_a":0.0,
    }
    return base

def save_prematch_snapshot(conn, fx: dict, feat: Dict[str,float]) -> None:
    fid = int((fx.get("fixture") or {}).get("id") or 0)
    if not fid: return
    now = int(time.time())
    payload = json.dumps({"feat": {k: float(feat.get(k,0.0)) for k in feat.keys()}}, separators=(",",":"), ensure_ascii=False)
    with conn.cursor() as cur:
        cur.execute("""INSERT INTO prematch_snapshots(match_id, created_ts, payload)
                       VALUES (%s,%s,%s)
                       ON CONFLICT (match_id) DO UPDATE SET created_ts=EXCLUDED.created_ts, payload=EXCLUDED.payload""",
                    (fid, now, payload))

def prematch_scan_save(conn=None) -> int:
    own_conn = conn is None
    if own_conn:
        _init_pool(); conn = POOL.getconn()
    saved = 0
    try:
        fixtures = _fetch_prematch_of_day()
        for fx in fixtures:
            fixture = fx.get("fixture") or {}; lg = fx.get("league") or {}; teams = fx.get("teams") or {}
            fid = int(fixture.get("id") or 0)
            if not fid: continue
            league_id = int(lg.get("id") or 0)
            league = f"{lg.get('country','')} - {lg.get('name','')}".strip(" -")
            home = (teams.get("home") or {}).get("name","")
            away = (teams.get("away") or {}).get("name","")

            feat = extract_prematch_features(fx)
            save_prematch_snapshot(conn, fx, feat)

            candidates: List[Tuple[str,str,float,Optional[Dict[str,Any]]]] = []

            # PRE OU
            for line in OU_LINES:
                mdl = load_model_from_settings(f"PRE_OU_{_fmt_line(line)}")
                if not mdl: continue
                p = _score_prob(feat, mdl)
                mk = f"Over/Under {_fmt_line(line)}"
                thr = _get_market_threshold(f"PRE {mk}")
                if p*100 >= thr: candidates.append((f"PRE {mk}", f"Over {_fmt_line(line)} Goals", p, mdl))
                q = 1-p
                if q*100 >= thr: candidates.append((f"PRE {mk}", f"Under {_fmt_line(line)} Goals", q, mdl))

            # PRE BTTS
            mdl = load_model_from_settings("PRE_BTTS_YES")
            if mdl:
                p = _score_prob(feat, mdl)
                thr = _get_market_threshold("PRE BTTS")
                if p*100 >= thr: candidates.append(("PRE BTTS", "BTTS: Yes", p, mdl))
                q = 1-p
                if q*100 >= thr: candidates.append(("PRE BTTS", "BTTS: No", q, mdl))

            # PRE 1X2
            mh = load_model_from_settings("PRE_WLD_HOME")
            ma = load_model_from_settings("PRE_WLD_AWAY")
            if mh and ma:
                ph = _score_prob(feat, mh); pa = _score_prob(feat, ma)
                s = max(EPS, ph+pa); ph, pa = ph/s, pa/s
                thr = _get_market_threshold("PRE 1X2")
                if ph*100 >= thr: candidates.append(("PRE 1X2", "Home Win", ph, mh))
                if pa*100 >= thr: candidates.append(("PRE 1X2", "Away Win", pa, ma))

            if not candidates: 
                continue

            odds_map = odds_for_fixture(fid)
            base_now = int(time.time())
            per_match = 0

            for idx, (mk, sug, p_model, mdl_used) in enumerate(sorted(candidates, key=lambda x: x[2], reverse=True)):
                ok_odds, odds, book = _price_gate(mk.replace("PRE ",""), sug, fid)
                if not ok_odds or odds is None:
                    continue
                p_book = _implied_prob_from_odds(odds)
                p_final = _apply_stack(p_model, mdl_used, p_book)
                edge = _ev(p_final, odds)
                ev_bps = int(round(edge*10000))
                ev_pct = round(edge*100,1)
                if ev_bps < EDGE_MIN_BPS:
                    continue

                created_ts = base_now + idx
                pct = round(p_final*100, 1)
                with conn.cursor() as cur:
                    cur.execute("""
                    INSERT INTO tips(match_id,league_id,league,home,away,market,suggestion,confidence,confidence_raw,
                                     score_at_tip,minute,created_ts,odds,book,ev_pct,sent_ok)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,'0-0',0,%s,%s,%s,%s,0)
                    """, (fid, league_id, league, home, away, mk, sug, float(pct), float(p_final),
                          created_ts, float(odds), (book or None), float(ev_pct)))
                saved += 1
                per_match += 1
                if per_match >= max(1, PREDICTIONS_PER_MATCH):
                    break

        log.info("[PREMATCH] saved=%d", saved)
        return saved
    finally:
        if own_conn:
            POOL.putconn(conn)

# ───────────────────────── Results backfill & grading ───────────────────────── #

def _fixture_by_id(fid: int) -> Optional[dict]:
    js = _api_get("fixtures", {"id": int(fid)}) or {}
    arr = js.get("response") or []
    return arr[0] if arr else None

def _is_final(short: str) -> bool:
    return (short or "").upper() in {"FT","AET","PEN"}

def backfill_results(conn=None, days: int = 14, limit: int = 400) -> int:
    own_conn = conn is None
    if own_conn:
        _init_pool(); conn = POOL.getconn()
    updated = 0
    try:
        cutoff = int(time.time()) - days*24*3600
        with conn.cursor() as cur:
            cur.execute("""
                WITH last AS (
                  SELECT match_id, MAX(created_ts) AS last_ts FROM tips
                  WHERE created_ts >= %s GROUP BY match_id
                )
                SELECT l.match_id FROM last l
                LEFT JOIN match_results r ON r.match_id = l.match_id
                WHERE r.match_id IS NULL
                ORDER BY l.last_ts DESC LIMIT %s
            """, (cutoff, limit))
            rows = cur.fetchall() or []
        for (mid,) in rows:
            fx = _fixture_by_id(int(mid))
            if not fx: continue
            st = (((fx.get("fixture") or {}).get("status") or {}).get("short") or "")
            if not _is_final(st): continue
            g = fx.get("goals") or {}
            gh = int(g.get("home") or 0); ga = int(g.get("away") or 0)
            btts = 1 if (gh>0 and ga>0) else 0
            with conn.cursor() as cur:
                cur.execute("""INSERT INTO match_results(match_id, final_goals_h, final_goals_a, btts_yes, updated_ts)
                               VALUES (%s,%s,%s,%s,%s)
                               ON CONFLICT(match_id) DO UPDATE SET
                                 final_goals_h=EXCLUDED.final_goals_h,
                                 final_goals_a=EXCLUDED.final_goals_a,
                                 btts_yes=EXCLUDED.btts_yes,
                                 updated_ts=EXCLUDED.updated_ts""",
                            (int(mid), gh, ga, btts, int(time.time())))
            updated += 1
        if updated:
            log.info("[RESULTS] backfilled %d", updated)
        return updated
    finally:
        if own_conn:
            POOL.putconn(conn)

# ───────────────────────── Digest & MOTD ───────────────────────── #

def _tip_outcome_for_result(suggestion: str, res: Dict[str,Any]) -> Optional[int]:
    gh=int(res.get("final_goals_h") or 0); ga=int(res.get("final_goals_a") or 0)
    total=gh+ga; btts=int(res.get("btts_yes") or 0)
    s=(suggestion or "").strip()
    if s.startswith("Over") or s.startswith("Under"):
        m = re.search(r'(\d+(?:\.\d+)?)', s)
        if not m: return None
        line = float(m.group(1))
        if s.startswith("Over"):
            if total > line: return 1
            if abs(total-line)<1e-9: return None
            return 0
        else:
            if total < line: return 1
            if abs(total-line)<1e-9: return None
            return 0
    if s=="BTTS: Yes": return 1 if btts==1 else 0
    if s=="BTTS: No":  return 1 if btts==0 else 0
    if s=="Home Win":  return 1 if gh>ga else 0
    if s=="Away Win":  return 1 if ga>gh else 0
    return None

def daily_accuracy_digest(conn=None, window_days: Optional[int]=None) -> Optional[str]:
    own_conn = conn is None
    if own_conn:
        _init_pool(); conn = POOL.getconn()
    try:
        window_days = int(window_days or DAILY_DIGEST_WINDOW_DAYS or 1)
        backfill_results(conn, days=max(1, window_days))
        cutoff_ts = int(datetime.now(BERLIN_TZ).timestamp()) - window_days*24*3600
        with conn.cursor() as cur:
            cur.execute("""
            SELECT t.market, t.suggestion, COALESCE(t.confidence_raw, t.confidence/100.0) AS prob,
                   t.odds, t.created_ts, r.final_goals_h, r.final_goals_a, r.btts_yes
            FROM tips t
            LEFT JOIN match_results r ON r.match_id = t.match_id
            WHERE t.created_ts >= %s AND t.suggestion <> 'HARVEST' AND t.sent_ok = 1
            ORDER BY t.created_ts ASC
            """, (cutoff_ts,))
            rows = cur.fetchall() or []
        if not rows:
            msg = f"📊 Accuracy Digest\nNo tips in the last {window_days}d."
            send_telegram(msg); return msg

        graded = wins = 0
        by_market: Dict[str, Dict[str,float]] = {}
        for (mkt, sug, prob, odds, cts, gh, ga, btts) in rows:
            res = _tip_outcome_for_result(sug, {"final_goals_h":gh,"final_goals_a":ga,"btts_yes":btts})
            if res is None: continue
            graded += 1
            if int(res)==1: wins += 1
            d = by_market.setdefault(mkt or "?", {"graded":0,"wins":0,"stake":0.0,"pnl":0.0})
            d["graded"] += 1
            if int(res)==1: d["wins"] += 1
            try:
                o = float(odds or 0.0)
                if 1.01 <= o <= 20.0:
                    d["stake"] += 1.0
                    d["pnl"] += (o-1.0) if int(res)==1 else -1.0
            except Exception:
                pass

        if graded == 0:
            msg = f"📊 Accuracy Digest\nNo graded tips in the last {window_days}d."
            send_telegram(msg); return msg

        acc = 100.0 * wins / max(1, graded)
        lines = [f"📊 <b>Accuracy Digest</b> (last {window_days}d)",
                 f"Tips sent: {len(rows)} • Graded: {graded} • Wins: {wins} • Accuracy: {acc:.1f}%"]
        for mk in sorted(by_market.keys()):
            st = by_market[mk]
            if st["graded"] <= 0: continue
            a = 100.0 * st["wins"] / max(1, st["graded"])
            roi_txt = ""
            if st["stake"] > 0:
                roi_val = 100.0 * st["pnl"] / st["stake"]
                roi_txt = f" • ROI {roi_val:+.1f}%"
            lines.append(f"• {mk}: {int(st['wins'])}/{int(st['graded'])} ({a:.1f}%){roi_txt}")
        msg = "\n".join(lines)
        send_telegram(msg)
        return msg
    except Exception as e:
        log.exception("[DIGEST] failed: %s", e)
        try: send_telegram(f"📊 Accuracy Digest failed: {str(e)}")
        except Exception: pass
        return None
    finally:
        if own_conn:
            POOL.putconn(conn)

def send_match_of_the_day(conn=None) -> bool:
    own_conn = conn is None
    if own_conn:
        _init_pool(); conn = POOL.getconn()
    try:
        fixtures = _fetch_prematch_of_day()
        if not fixtures:
            return send_telegram("🏅 MOTD: no fixtures today.")
        best = None; fallback = None
        for fx in fixtures:
            fixture = fx.get("fixture") or {}; lg=fx.get("league") or {}; teams=fx.get("teams") or {}
            fid = int(fixture.get("id") or 0)
            home = (teams.get("home") or {}).get("name","")
            away = (teams.get("away") or {}).get("name","")
            league = f"{lg.get('country','')} - {lg.get('name','')}".strip(" -")
            kickoff = (fixture.get("date") or "")
            try:
                dt = datetime.fromisoformat(kickoff.replace("Z","+00:00")).astimezone(BERLIN_TZ)
                kickoff_txt = dt.strftime("%H:%M")
            except Exception:
                kickoff_txt = "TBD"

            feat = extract_prematch_features(fx)
            candidates: List[Tuple[str,str,float,Optional[Dict[str,Any]]]] = []

            for line in OU_LINES:
                mdl = load_model_from_settings(f"PRE_OU_{_fmt_line(line)}")
                if mdl:
                    p = _score_prob(feat, mdl)
                    mk = f"Over/Under {_fmt_line(line)}"
                    thr = _get_market_threshold(f"PRE {mk}")
                    if p*100 >= thr: candidates.append((mk, f"Over {_fmt_line(line)} Goals", p, mdl))
                    q = 1-p
                    if q*100 >= thr: candidates.append((mk, f"Under {_fmt_line(line)} Goals", q, mdl))

            mdl = load_model_from_settings("PRE_BTTS_YES")
            if mdl:
                p = _score_prob(feat, mdl); thr = _get_market_threshold("PRE BTTS")
                if p*100 >= thr: candidates.append(("BTTS","BTTS: Yes", p, mdl))
                q = 1-p
                if q*100 >= thr: candidates.append(("BTTS","BTTS: No", q, mdl))

            mh = load_model_from_settings("PRE_WLD_HOME")
            ma = load_model_from_settings("PRE_WLD_AWAY")
            if mh and ma:
                ph = _score_prob(feat, mh); pa = _score_prob(feat, ma)
                s = max(EPS, ph+pa); ph, pa = ph/s, pa/s
                thr = _get_market_threshold("PRE 1X2")
                if ph*100 >= thr: candidates.append(("1X2","Home Win", ph, mh))
                if pa*100 >= thr: candidates.append(("1X2","Away Win", pa, ma))

            if not candidates: 
                continue

            odds_map = odds_for_fixture(fid)
            for mk, sug, p_model, mdl_used in candidates:
                ok_odds, odds, book = _price_gate(mk, sug, fid)
                if not ok_odds or odds is None: 
                    continue
                p_book = _implied_prob_from_odds(odds)
                p_final = _apply_stack(p_model, mdl_used, p_book)
                ev = _ev(p_final, odds); ev_bps = int(round(ev*10000)); ev_pct = round(ev*100,1)
                score = (p_final ** 1.2) * (1 + max(0.0, ev_pct)/100.0)
                cand = (score, p_final*100, sug, home, away, league, kickoff_txt, odds, book, ev_pct)
                if fallback is None or cand > fallback: fallback = cand
                if ev_bps >= MOTD_MIN_EV_BPS:
                    if best is None or cand > best: best = cand

        chosen = best or fallback
        if not chosen:
            return send_telegram("🏅 MOTD: no pick available.")
        _, prob_pct, sug, home, away, league, kickoff_txt, odds, book, ev_pct = chosen
        msg = ("🏅 <b>Match of the Day</b>\n"
               f"<b>Match:</b> {home} vs {away}\n"
               f"🏆 <b>League:</b> {league}\n"
               f"⏰ <b>Kickoff (Berlin):</b> {kickoff_txt}\n"
               f"<b>Tip:</b> {sug}\n"
               f"📈 <b>Confidence:</b> {prob_pct:.1f}%\n"
               f"💰 <b>Odds:</b> {odds:.2f} @ {book or 'Book'} • <b>EV:</b> {ev_pct:+.1f}%")
        return send_telegram(msg)
    finally:
        if own_conn:
            POOL.putconn(conn)

# ───────────────────────── Scheduler ───────────────────────── #

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

def _start_scheduler():
    if not RUN_SCHEDULER:
        log.info("[SCHED] disabled")
        return
    sched = BackgroundScheduler(timezone=ZoneInfo("UTC"))
    # In-play scan every 5 minutes (lean to protect API budget)
    scan_sec = int(os.getenv("SCAN_INTERVAL_SEC","300"))
    sched.add_job(lambda: _run_with_pg_lock(1001, production_scan), "interval",
                  seconds=scan_sec, id="scan", max_instances=1, coalesce=True, misfire_grace_time=120)
    # Backfill results every 15 minutes
    sched.add_job(lambda: _run_with_pg_lock(1002, backfill_results), "interval",
                  minutes=int(os.getenv("BACKFILL_EVERY_MIN","15")), id="backfill",
                  max_instances=1, coalesce=True, misfire_grace_time=120)
    # Daily digest (Berlin local)
    digest_h = int(os.getenv("DAILY_ACCURACY_HOUR","3")); digest_m = int(os.getenv("DAILY_ACCURACY_MINUTE","6"))
    sched.add_job(lambda: _run_with_pg_lock(1003, daily_accuracy_digest),
                  CronTrigger(hour=digest_h, minute=digest_m, timezone=BERLIN_TZ),
                  id="digest", max_instances=1, coalesce=True, misfire_grace_time=300)
    # MOTD (Berlin local)
    motd_h = int(os.getenv("MOTD_HOUR","19")); motd_m = int(os.getenv("MOTD_MINUTE","15"))
    sched.add_job(lambda: _run_with_pg_lock(1004, send_match_of_the_day),
                  CronTrigger(hour=motd_h, minute=motd_m, timezone=BERLIN_TZ),
                  id="motd", max_instances=1, coalesce=True, misfire_grace_time=300)
    sched.start()
    log.info("[SCHED] started (scan=%ss)", scan_sec)

# ───────────────────────── HTTP API ───────────────────────── #

def _require_admin():
    key = request.headers.get("X-API-Key") or request.args.get("key") or ((request.json or {}).get("key") if request.is_json else None)
    if not ADMIN_API_KEY or key != ADMIN_API_KEY:
        abort(401)

@app.route("/")
def root():
    return jsonify({"ok": True, "name": "goalsniper", "mode": "FULL_AI", "scheduler": RUN_SCHEDULER})

@app.route("/health")
def health():
    try:
        _init_pool(); conn = POOL.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM tips")
                n = cur.fetchone()[0]
            return jsonify({"ok": True, "db": "ok", "tips_count": int(n)})
        finally:
            POOL.putconn(conn)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/init-db", methods=["POST"])
def http_init_db():
    _require_admin()
    _init_pool(); conn = POOL.getconn()
    try:
        init_db(conn)
        return jsonify({"ok": True})
    finally:
        POOL.putconn(conn)

@app.route("/admin/scan", methods=["POST","GET"])
def http_scan():
    _require_admin()
    res = _run_with_pg_lock(1001, production_scan) or (0,0)
    return jsonify({"ok": True, "saved": int(res[0]), "live_seen": int(res[1])})

@app.route("/admin/backfill-results", methods=["POST","GET"])
def http_backfill():
    _require_admin()
    n = _run_with_pg_lock(1002, backfill_results) or 0
    return jsonify({"ok": True, "updated": int(n)})

@app.route("/admin/prematch-scan", methods=["POST","GET"])
def http_prematch_scan():
    _require_admin()
    n = prematch_scan_save()
    return jsonify({"ok": True, "saved": int(n)})

@app.route("/admin/digest", methods=["POST","GET"])
def http_digest():
    _require_admin()
    days_q = request.args.get("days","").strip()
    days = None
    try:
        if days_q: days = max(1, int(float(days_q)))
    except Exception:
        days = None
    msg = daily_accuracy_digest(window_days=days)
    return jsonify({"ok": bool(msg), "window_days": (days if days is not None else DAILY_DIGEST_WINDOW_DAYS)})

@app.route("/admin/motd", methods=["POST","GET"])
def http_motd():
    _require_admin()
    ok = send_match_of_the_day()
    return jsonify({"ok": bool(ok)})

@app.route("/settings/<key>", methods=["GET","POST"])
def http_settings(key: str):
    _require_admin()
    if request.method == "GET":
        val = get_setting_cached(key)
        return jsonify({"ok": True, "key": key, "value": val})
    data = request.get_json(silent=True) or {}
    val = data.get("value")
    if val is None: abort(400)
    set_setting_cached(key, str(val))
    return jsonify({"ok": True})

@app.route("/tips/latest")
def http_latest():
    limit = int(request.args.get("limit","50"))
    limit = max(1, min(500, limit))
    _init_pool(); conn = POOL.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("""SELECT match_id,league,home,away,market,suggestion,confidence,confidence_raw,
                                  score_at_tip,minute,created_ts,odds,book,ev_pct
                           FROM tips WHERE suggestion <> 'HARVEST'
                           ORDER BY created_ts DESC LIMIT %s""", (limit,))
            rows = cur.fetchall() or []
        tips=[]
        for r in rows:
            tips.append({
                "match_id": int(r[0]), "league": r[1], "home": r[2], "away": r[3],
                "market": r[4], "suggestion": r[5], "confidence": float(r[6]),
                "confidence_raw": (float(r[7]) if r[7] is not None else None),
                "score_at_tip": r[8], "minute": int(r[9]), "created_ts": int(r[10]),
                "odds": (float(r[11]) if r[11] is not None else None),
                "book": r[12], "ev_pct": (float(r[13]) if r[13] is not None else None)
            })
        return jsonify({"ok": True, "tips": tips})
    finally:
        POOL.putconn(conn)

# ───────────────────────── Boot ───────────────────────── #

def _on_boot():
    if not DATABASE_URL:
        raise SystemExit("DATABASE_URL is required")
    _init_pool()
    conn = POOL.getconn()
    try:
        init_db(conn)
        set_setting_cached("boot_ts", str(int(time.time())))
    finally:
        POOL.putconn(conn)
    _start_scheduler()
    log.info("🚀 goalsniper FULL_AI started")

_on_boot()

if __name__ == "__main__":
    app.run(host=os.getenv("HOST","0.0.0.0"), port=int(os.getenv("PORT","8080")))
