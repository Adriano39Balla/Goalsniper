# goalsniper — OU 2.5 ONLY (in-play + prematch snapshot) — Railway-ready
# Lean build: predicts ONLY Over/Under 2.5 with robust odds/EV gating and data-quality guards.

import os, json, time, logging, requests, sys, signal, atexit
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import psycopg2
from psycopg2.pool import SimpleConnectionPool
from flask import Flask, jsonify, request, abort
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# ───────── Logging ─────────
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
handler.setFormatter(formatter)
log = logging.getLogger("goalsniper_ou25")
log.handlers = [handler]
log.setLevel(logging.INFO)
log.propagate = False

app = Flask(__name__)

# ───────── Env ─────────
def _require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise SystemExit(f"Missing required environment variable: {name}")
    return v

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# Core env
TELEGRAM_BOT_TOKEN = _require_env("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = _require_env("TELEGRAM_CHAT_ID")
API_KEY            = _require_env("API_KEY")
DATABASE_URL       = _require_env("DATABASE_URL")

# Knobs (focused on OU 2.5 only)
CONF_THRESHOLD     = float(os.getenv("CONF_THRESHOLD", "72"))  # % threshold for sending tips
MAX_TIPS_PER_SCAN  = int(os.getenv("MAX_TIPS_PER_SCAN", "20"))
TIP_MIN_MINUTE     = int(os.getenv("TIP_MIN_MINUTE", "15"))
SCAN_INTERVAL_SEC  = int(os.getenv("SCAN_INTERVAL_SEC", "240"))
TOTAL_MATCH_MINUTES= int(os.getenv("TOTAL_MATCH_MINUTES", "95"))
PER_LEAGUE_CAP     = int(os.getenv("PER_LEAGUE_CAP", "3"))
RUN_SCHEDULER      = os.getenv("RUN_SCHEDULER", "1") not in ("0","false","False","no","NO")

# Odds/EV gates (only OU 2.5)
MIN_ODDS_OU        = float(os.getenv("MIN_ODDS_OU", "1.50"))
MAX_ODDS_ALL       = float(os.getenv("MAX_ODDS_ALL", "20.0"))
EDGE_MIN_BPS       = int(os.getenv("EDGE_MIN_BPS", "500"))  # +5% EV by default
ODDS_REQUIRE_N_BOOKS = int(os.getenv("ODDS_REQUIRE_N_BOOKS", "2"))
ODDS_AGGREGATION   = os.getenv("ODDS_AGGREGATION", "median").lower() # median|best
ODDS_OUTLIER_MULT  = float(os.getenv("ODDS_OUTLIER_MULT", "1.8"))
ODDS_FAIR_MAX_MULT = float(os.getenv("ODDS_FAIR_MAX_MULT", "2.5"))
ALLOW_TIPS_WITHOUT_ODDS = os.getenv("ALLOW_TIPS_WITHOUT_ODDS","0") not in ("0","false","False","no","NO")

# Data-quality guards
REQUIRE_STATS_MINUTE = int(os.getenv("REQUIRE_STATS_MINUTE", "35"))
REQUIRE_DATA_FIELDS  = int(os.getenv("REQUIRE_DATA_FIELDS", "2"))
STALE_GUARD_ENABLE   = os.getenv("STALE_GUARD_ENABLE","1") not in ("0","false","False","no","NO")
STALE_STATS_MAX_SEC  = int(os.getenv("STALE_STATS_MAX_SEC","240"))

# Timezones
TZ_UTC = ZoneInfo("UTC")
BERLIN_TZ = ZoneInfo("Europe/Berlin")

# ───────── External APIs ─────────
BASE_URL = "https://v3.football.api-sports.io"
FOOTBALL_API_URL = f"{BASE_URL}/fixtures"
HEADERS = {"x-apisports-key": API_KEY, "Accept": "application/json"}
INPLAY_STATUSES = {"1H","HT","2H","ET","BT","P"}

session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1,
    status_forcelist=[429,500,502,503,504], respect_retry_after_header=True)))

REQ_TIMEOUT_SEC = float(os.getenv("REQ_TIMEOUT_SEC","8.0"))

# ───────── Telegram ─────────
def send_telegram(text: str) -> bool:
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True},
            timeout=REQ_TIMEOUT_SEC
        )
        return bool(r.ok)
    except Exception:
        return False

# ───────── DB Pool ─────────
POOL: Optional[SimpleConnectionPool] = None

def _normalize_dsn(url: str) -> str:
    if not url: return url
    dsn = url.strip()
    if "sslmode=" not in dsn:
        dsn += ("&" if "?" in dsn else "?") + "sslmode=require"
    return dsn

def _init_pool():
    global POOL
    if POOL: return
    POOL = SimpleConnectionPool(
        minconn=1,
        maxconn=int(os.getenv("PG_MAXCONN","10")),
        dsn=_normalize_dsn(DATABASE_URL),
        connect_timeout=int(os.getenv("PG_CONNECT_TIMEOUT","10")),
        application_name="goalsniper_ou25"
    )
    log.info("[DB] pool initialized.")

class PooledConn:
    def __enter__(self):
        self.conn = POOL.getconn()  # type: ignore
        self.conn.autocommit = True
        self.cur = self.conn.cursor()
        return self
    def __exit__(self, a,b,c):
        try: self.cur.close()
        except: pass
        try: POOL.putconn(self.conn)  # type: ignore
        except: pass
    def execute(self, sql: str, params: tuple|list=()):
        self.cur.execute(sql, params or ())
        return self.cur

def db_conn(): 
    if not POOL: _init_pool()
    return PooledConn()

def init_db():
    with db_conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS tips (
            match_id BIGINT, league_id BIGINT, league TEXT,
            home TEXT, away TEXT, market TEXT, suggestion TEXT,
            confidence DOUBLE PRECISION, confidence_raw DOUBLE PRECISION,
            score_at_tip TEXT, minute INTEGER, created_ts BIGINT,
            odds DOUBLE PRECISION, book TEXT, ev_pct DOUBLE PRECISION,
            sent_ok INTEGER DEFAULT 1,
            PRIMARY KEY (match_id, created_ts))""")
        c.execute("""CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS match_results (
            match_id BIGINT PRIMARY KEY, final_goals_h INTEGER, final_goals_a INTEGER, btts_yes INTEGER, updated_ts BIGINT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS odds_history (
            match_id BIGINT,
            captured_ts BIGINT,
            market TEXT,
            selection TEXT,
            odds DOUBLE PRECISION,
            book TEXT,
            PRIMARY KEY (match_id, market, selection, captured_ts)
        )""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_tips_created ON tips (created_ts DESC)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_tips_match ON tips (match_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_odds_hist_match ON odds_history (match_id, captured_ts DESC)")
    log.info("[DB] schema ensured.")

# ───────── Settings helpers ─────────
def get_setting(key: str) -> Optional[str]:
    with db_conn() as c:
        cur = c.execute("SELECT value FROM settings WHERE key=%s", (key,))
        row = cur.fetchone()
        return (row[0] if row else None)

def set_setting(key: str, value: str) -> None:
    with db_conn() as c:
        c.execute("INSERT INTO settings(key,value) VALUES(%s,%s) ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value", (key,value))

# ───────── API helpers ─────────
def _api_get(url: str, params: dict, timeout: int = 12) -> Optional[dict]:
    try:
        r = session.get(url, headers=HEADERS, params=params, timeout=min(timeout, REQ_TIMEOUT_SEC))
        return r.json() if r.ok else None
    except Exception:
        return None

def fetch_match_stats(fid: int) -> list:
    js = _api_get(f"{FOOTBALL_API_URL}/statistics", {"fixture": fid}) or {}
    return js.get("response",[]) if isinstance(js,dict) else []

def fetch_match_events(fid: int) -> list:
    js = _api_get(f"{FOOTBALL_API_URL}/events", {"fixture": fid}) or {}
    return js.get("response",[]) if isinstance(js,dict) else []

_BLOCK_PATTERNS = ["u17","u18","u19","u20","u21","u23","youth","junior","reserve","res.","friendlies","friendly"]
def _blocked_league(league_obj: dict) -> bool:
    name=str((league_obj or {}).get("name","")).lower()
    country=str((league_obj or {}).get("country","")).lower()
    typ=str((league_obj or {}).get("type","")).lower()
    txt=f"{country} {name} {typ}"
    if any(p in txt for p in _BLOCK_PATTERNS): return True
    deny=[x.strip() for x in os.getenv("LEAGUE_DENY_IDS","").split(",") if x.strip()]
    lid=str((league_obj or {}).get("id") or "")
    return lid in deny

def fetch_live_matches() -> List[dict]:
    js = _api_get(FOOTBALL_API_URL, {"live": "all"}) or {}
    matches = [m for m in (js.get("response",[]) if isinstance(js,dict) else []) if not _blocked_league(m.get("league") or {})]
    out=[]
    for m in matches:
        st=((m.get("fixture") or {}).get("status") or {})
        elapsed=st.get("elapsed"); short=(st.get("short") or "").upper()
        if elapsed is None or elapsed>120 or short not in INPLAY_STATUSES: 
            continue
        fid=(m.get("fixture") or {}).get("id")
        m["statistics"]=fetch_match_stats(fid); m["events"]=fetch_match_events(fid)
        out.append(m)
    return out

# ───────── Feature extraction (lean) ─────────
def _num(v) -> float:
    try:
        if isinstance(v,str) and v.endswith("%"): return float(v[:-1])
        return float(v or 0)
    except: return 0.0

def _pos_pct(v) -> float:
    try: return float(str(v).replace("%","").strip() or 0)
    except: return 0.0

def extract_features(m: dict) -> Dict[str,float]:
    home = m["teams"]["home"]["name"]
    away = m["teams"]["away"]["name"]
    gh = m["goals"]["home"] or 0
    ga = m["goals"]["away"] or 0
    minute = int(((m.get("fixture") or {}).get("status") or {}).get("elapsed") or 0)

    stats = {}
    for s in (m.get("statistics") or []):
        t = (s.get("team") or {}).get("name")
        if t: stats[t] = { (i.get("type") or ""): i.get("value") for i in (s.get("statistics") or []) }

    sh = stats.get(home, {}) or {}
    sa = stats.get(away, {}) or {}

    xg_h = _num(sh.get("Expected Goals", 0))
    xg_a = _num(sa.get("Expected Goals", 0))
    sot_h = _num(sh.get("Shots on Target", sh.get("Shots on Goal", 0)))
    sot_a = _num(sa.get("Shots on Target", sa.get("Shots on Goal", 0)))
    sh_total_h = _num(sh.get("Total Shots", sh.get("Shots Total", 0)))
    sh_total_a = _num(sa.get("Total Shots", sa.get("Shots Total", 0)))
    cor_h = _num(sh.get("Corner Kicks", 0))
    cor_a = _num(sa.get("Corner Kicks", 0))
    pos_h = _pos_pct(sh.get("Ball Possession", 0))
    pos_a = _pos_pct(sa.get("Ball Possession", 0))

    red_h = red_a = 0
    for ev in (m.get("events") or []):
        if (ev.get("type","").lower()=="card"):
            d = (ev.get("detail","") or "").lower()
            t = (ev.get("team") or {}).get("name") or ""
            if "red" in d or "second yellow" in d:
                if t == home: red_h += 1
                elif t == away: red_a += 1

    return {
        "minute": float(minute),
        "goals_h": float(gh), "goals_a": float(ga),
        "goals_sum": float(gh + ga),
        "xg_h": float(xg_h), "xg_a": float(xg_a),
        "xg_sum": float(xg_h + xg_a),
        "sot_h": float(sot_h), "sot_a": float(sot_a), "sot_sum": float(sot_h + sot_a),
        "sh_total_h": float(sh_total_h), "sh_total_a": float(sh_total_a),
        "cor_h": float(cor_h), "cor_a": float(cor_a), "cor_sum": float(cor_h + cor_a),
        "pos_h": float(pos_h), "pos_a": float(pos_a),
        "red_h": float(red_h), "red_a": float(red_a), "red_sum": float(red_h + red_a),
    }

# ───────── Model loading/scoring (OU 2.5 only) ─────────
EPS=1e-12
def _sigmoid(x: float) -> float:
    try:
        if x<-50: return 1e-22
        if x>50:  return 1-1e-22
        import math; return 1/(1+math.exp(-x))
    except: return 0.5

def _logit(p: float) -> float:
    import math; p=max(EPS,min(1-EPS,float(p))); return math.log(p/(1-p))

def _linpred(feat: Dict[str,float], weights: Dict[str,float], intercept: float) -> float:
    s=float(intercept or 0.0)
    for k,w in (weights or {}).items(): s += float(w or 0.0)*float(feat.get(k,0.0))
    return s

def _calibrate(p: float, cal: Dict[str,Any]) -> float:
    method=(cal or {}).get("method","sigmoid"); a=float((cal or {}).get("a",1.0)); b=float((cal or {}).get("b",0.0))
    if method.lower()=="platt": return _sigmoid(a*_logit(p)+b)
    import math; p=max(EPS,min(1-EPS,float(p))); z=math.log(p/(1-p)); return _sigmoid(a*z+b)

def _score_prob(feat: Dict[str,float], mdl: Dict[str,Any]) -> float:
    p=_sigmoid(_linpred(feat, mdl.get("weights",{}), float(mdl.get("intercept",0.0))))
    cal=mdl.get("calibration") or {}
    try: 
        if cal: p=_calibrate(p, cal)
    except: pass
    return max(0.0, min(1.0, float(p)))

def _validate_model_blob(tmp: dict) -> bool:
    return isinstance(tmp, dict) and "weights" in tmp and "intercept" in tmp and isinstance(tmp.get("weights"), dict)

MODEL_KEYS_ORDER = ["model_v2:{name}", "model_latest:{name}", "model:{name}", "pre_{name}"]

def load_model_from_settings(name: str) -> Optional[Dict[str, Any]]:
    for pat in MODEL_KEYS_ORDER:
        raw=get_setting(pat.format(name=name))
        if not raw: continue
        try:
            tmp=json.loads(raw)
            if _validate_model_blob(tmp):
                tmp.setdefault("intercept",0.0); tmp.setdefault("weights",{})
                cal=tmp.get("calibration") or {}
                if isinstance(cal,dict):
                    cal.setdefault("method","sigmoid"); cal.setdefault("a",1.0); cal.setdefault("b",0.0)
                    tmp["calibration"]=cal
                return tmp
        except Exception:
            continue
    return None

def _load_ou25_model() -> Optional[Dict[str,Any]]:
    return load_model_from_settings("OU_2.5") or load_model_from_settings("O25")

def _ou25_live_odds_plausible(odds: Optional[float], minute: int, goals_sum: int) -> bool:
    """Quick plausibility guard for Over 2.5 prices in-play."""
    if odds is None:
        return False
    try:
        m = int(minute); g = int(goals_sum)
    except Exception:
        return True  # don't block if we can't parse

    # Very early 2–0 → price must be very short
    if g >= 2 and m <= 30:
        return odds <= 1.30
    # Late first half 2–0 → even shorter
    if g >= 2 and m <= 45:
        return odds <= 1.20
    # Early 1–0 → still fairly short on O2.5 in many leagues
    if g == 1 and m <= 20:
        return odds <= 1.80

    return True

# ───────── Odds aggregation (OU 2.5 only) ─────────
def _market_name_normalize(s: str) -> str:
    s=(s or "").lower()
    if "over/under" in s or "total" in s or "goals" in s: return "OU"
    return s

def _aggregate_price(vals: List[tuple[float,str]], prob_hint: Optional[float]) -> tuple[Optional[float], Optional[str]]:
    if not vals: return None, None
    xs = sorted([o for (o,_) in vals if (o or 0) > 0])
    if not xs: return None, None
    import statistics
    med = statistics.median(xs)
    filtered = [(o,b) for (o,b) in vals if o <= med * max(1.0, ODDS_OUTLIER_MULT)] or vals
    xs2 = sorted([o for (o,_) in filtered])
    med2 = statistics.median(xs2)
    if prob_hint is not None and prob_hint > 0:
        fair = 1.0 / max(1e-6, float(prob_hint))
        cap = fair * max(1.0, ODDS_FAIR_MAX_MULT)
        filtered = [(o,b) for (o,b) in filtered if o <= cap] or filtered
    if ODDS_AGGREGATION == "best":
        best = max(filtered, key=lambda t: t[0])
        return float(best[0]), str(best[1])
    target = med2
    pick = min(filtered, key=lambda t: abs(t[0] - target))
    return float(pick[0]), f"{pick[1]} (median of {len(xs)})"

def fetch_odds_ou25(fid: int, prob_hint_over: Optional[float] = None) -> Dict[str, Dict[str, Any]]:
    """Return {'Over': {'odds':x,'book':y}, 'Under': {...}} for OU 2.5"""
    js = _api_get(f"{BASE_URL}/odds/live", {"fixture": fid}) or {}
    if not (js.get("response") or []):
        js = _api_get(f"{BASE_URL}/odds", {"fixture": fid}) or {}
    by: Dict[str, List[tuple[float,str]]] = {"Over": [], "Under": []}
    try:
        for r in js.get("response",[]) or []:
            for bk in (r.get("bookmakers") or []):
                book = bk.get("name") or "Book"
                for mkt in (bk.get("bets") or []):
                    if _market_name_normalize(mkt.get("name","")) != "OU":
                        continue
                    for v in (mkt.get("values") or []):
                        lbl = str(v.get("value") or "").lower()
                        if ("over" in lbl) or ("under" in lbl):
                            try:
                                ln = float(lbl.split()[-1])
                            except Exception:
                                continue
                            if abs(ln - 2.5) > 1e-6:
                                continue
                            if "over" in lbl:
                                by["Over"].append((float(v.get("odd") or 0), book))
                            elif "under" in lbl:
                                by["Under"].append((float(v.get("odd") or 0), book))
    except Exception:
        pass

    # require distinct books
    out: Dict[str, Dict[str, Any]] = {}
    for side, lst in by.items():
        if len({b for (_, b) in lst}) < max(1, ODDS_REQUIRE_N_BOOKS):
            continue
        ag, label = _aggregate_price(lst, prob_hint_over if side=="Over" else (1.0-(prob_hint_over or 0.0) if prob_hint_over is not None else None))
        if ag is not None:
            out[side] = {"odds": float(ag), "book": label}
    return out

# ───────── Helpers ─────────
def _league_name(m: dict) -> Tuple[int,str]:
    lg=(m.get("league") or {}) or {}
    return int(lg.get("id") or 0), f"{lg.get('country','')} - {lg.get('name','')}".strip(" -")

def _teams(m: dict) -> Tuple[str,str]:
    t=(m.get("teams") or {}) or {}
    return (t.get("home",{}).get("name",""), t.get("away",{}).get("name",""))

def _pretty_score(m: dict) -> str:
    gh=(m.get("goals") or {}).get("home") or 0; ga=(m.get("goals") or {}).get("away") or 0
    return f"{gh}-{ga}"

def stats_coverage_ok(feat: Dict[str,float], minute: int) -> bool:
    if minute < REQUIRE_STATS_MINUTE:
        return True
    fields = [
        feat.get("xg_sum", 0.0),
        feat.get("sot_sum", 0.0),
        feat.get("cor_sum", 0.0),
        (feat.get("pos_h", 0.0) + feat.get("pos_a", 0.0)),
    ]
    nonzero = sum(1 for v in fields if (v or 0) > 0)
    return nonzero >= max(0, REQUIRE_DATA_FIELDS)

def _parse_ou_line_from_suggestion(s: str) -> Optional[float]:
    try:
        import re
        m = re.search(r'(\d+\.?\d*)', s or "")
        return float(m.group(1)) if m else None
    except Exception:
        return None

def _ev(prob: float, odds: float) -> float:
    return prob*max(0.0, float(odds)) - 1.0

def _min_odds_for_market() -> float: return MIN_ODDS_OU

def _candidate_is_sane_over(feat: Dict[str,float]) -> bool:
    # For Over 2.5: must not already be settled, avoid absurd states
    total = int(feat.get("goals_sum", 0))
    if total >= 3:   # already over line
        return False
    return True

def _candidate_is_sane_under(feat: Dict[str,float], minute: int) -> bool:
    # Under 2.5: only if total <= 1 and minute sufficiently advanced
    total = int(feat.get("goals_sum", 0))
    if total > 1:
        return False
    if minute < max(25, TIP_MIN_MINUTE):  # discourage too-early unders
        return False
    # Red cards increase volatility → avoid when >=1
    if int(feat.get("red_sum", 0)) >= 1:
        return False
    return True

# Stale-feed guard (lean)
_FEED_STATE: Dict[int, Dict[str, Any]] = {}
def _safe_num(x) -> float:
    try:
        if isinstance(x, str) and x.endswith("%"): return float(x[:-1])
        return float(x or 0.0)
    except Exception:
        return 0.0

def _match_fingerprint(m: dict) -> Tuple:
    teams = (m.get("teams") or {})
    home = (teams.get("home") or {}).get("name", "")
    away = (teams.get("away") or {}).get("name", "")

    stats_by_team = {}
    for s in (m.get("statistics") or []):
        tname = ((s.get("team") or {}).get("name") or "").strip()
        if tname:
            stats_by_team[tname] = {str((i.get("type") or "")).lower(): i.get("value") for i in (s.get("statistics") or [])}

    sh = stats_by_team.get(home, {}) or {}
    sa = stats_by_team.get(away, {}) or {}

    def g(d: dict, key_variants: Tuple[str, ...]) -> float:
        for k in key_variants:
            if k in d: return _safe_num(d[k])
        return 0.0

    xg_h = g(sh, ("expected goals",))
    xg_a = g(sa, ("expected goals",))
    sot_h = g(sh, ("shots on target", "shots on goal"))
    sot_a = g(sa, ("shots on target", "shots on goal"))
    cor_h = g(sh, ("corner kicks",))
    cor_a = g(sa, ("corner kicks",))
    gh = int(((m.get("goals") or {}).get("home") or 0) or 0)
    ga = int(((m.get("goals") or {}).get("away") or 0) or 0)
    n_events = len(m.get("events") or [])

    return (round(xg_h + xg_a, 3), int(sot_h + sot_a), int(cor_h + cor_a), gh, ga, n_events)

def is_feed_stale(fid: int, m: dict, minute: int) -> bool:
    if not STALE_GUARD_ENABLE:
        return False
    now = time.time()
    if minute < 10:
        _FEED_STATE[fid] = {"fp": _match_fingerprint(m), "last_change": now, "last_minute": minute}
        return False
    fp = _match_fingerprint(m)
    st = _FEED_STATE.get(fid)
    if st is None:
        _FEED_STATE[fid] = {"fp": fp, "last_change": now, "last_minute": minute}
        return False
    if fp != st.get("fp"):
        st["fp"] = fp; st["last_change"] = now; st["last_minute"] = minute
        return False
    last_min = int(st.get("last_minute") or 0)
    st["last_minute"] = minute
    if minute > last_min and (now - float(st.get("last_change") or now)) >= STALE_STATS_MAX_SEC:
        return True
    return False

# ───────── Production scan (OU 2.5 only) ─────────
def _get_market_threshold_ou25() -> float:
    v = get_setting("conf_threshold:Over/Under 2.5")
    try:
        return float(v) if v is not None else float(CONF_THRESHOLD)
    except Exception:
        return float(CONF_THRESHOLD)

def production_scan() -> Tuple[int, int]:
    try:
        matches = fetch_live_matches()
    except Exception as e:
        log.error("[SCAN] fetch live failed: %s", e); return (0,0)
    live_seen = len(matches)
    if live_seen == 0:
        log.info("[SCAN] no live matches")
        return 0, 0

    mdl = _load_ou25_model()
    if not mdl:
        log.warning("[SCAN] OU 2.5 model missing in settings (keys: OU_2.5 / O25)")
        return 0, live_seen

    saved = 0
    threshold_pct = _get_market_threshold_ou25()
    now_ts = int(time.time())
    per_league_counter: dict[int,int] = {}

    with db_conn() as c:
        for m in matches:
            try:
                fid = int((m.get("fixture") or {}).get("id") or 0)
                if not fid: continue

                feat = extract_features(m)
                minute = int(feat.get("minute", 0))

                if minute < TIP_MIN_MINUTE:
                    continue
                if is_feed_stale(fid, m, minute):
                    continue
                if not stats_coverage_ok(feat, minute):
                    continue

                # Model prob for OVER 2.5
                p_over = _score_prob(feat, mdl)
                p_under = 1.0 - p_over

                # sanity gates
                if p_over*100.0 >= threshold_pct and _candidate_is_sane_over(feat):
                    suggestion = "Over 2.5 Goals"
                    prob = p_over
                elif p_under*100.0 >= threshold_pct and _candidate_is_sane_under(feat, minute):
                    suggestion = "Under 2.5 Goals"
                    prob = p_under
                else:
                    continue

                # Odds/EV gate
                odds_map = fetch_odds_ou25(fid, prob_hint_over=p_over)
                side = "Over" if suggestion.startswith("Over") else "Under"
                rec = odds_map.get(side)
                if not rec:
                    if not ALLOW_TIPS_WITHOUT_ODDS:
                        continue
                    odds = None; book = None; ev_pct = None
                else:
                    odds = float(rec["odds"]); book = rec["book"]; ev_pct = round(_ev(prob, odds)*100.0, 1)
                    if not (_min_odds_for_market() <= odds <= MAX_ODDS_ALL):
                        continue
                    if int(round(_ev(prob, odds)*10000)) < EDGE_MIN_BPS:
                        continue
                    if mk.startswith("Over/Under") and suggestion.startswith("Over 2.5") and odds is not None:
                        if not _ou25_live_odds_plausible(float(odds), minute, int(feat.get("goals_sum", 0))):
                            continue

                league_id, league = _league_name(m)
                if PER_LEAGUE_CAP > 0 and per_league_counter.get(league_id, 0) >= PER_LEAGUE_CAP:
                    continue

                home, away = _teams(m)
                score = _pretty_score(m)
                conf_pct = round(prob*100.0, 1)
                created_ts = now_ts + saved

                c.execute(
                    "INSERT INTO tips(match_id,league_id,league,home,away,market,suggestion,confidence,confidence_raw,"
                    "score_at_tip,minute,created_ts,odds,book,ev_pct,sent_ok) "
                    "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,0)",
                    (fid, league_id, league, home, away, "Over/Under 2.5", suggestion,
                     float(conf_pct), float(prob), score, minute, created_ts,
                     (float(odds) if odds is not None else None),
                     (book or None),
                     (float(ev_pct) if ev_pct is not None else None))
                )

                sent = send_telegram(_format_tip_message(home, away, league, minute, score, suggestion, conf_pct, feat, odds, book, ev_pct))
                if sent:
                    c.execute("UPDATE tips SET sent_ok=1 WHERE match_id=%s AND created_ts=%s", (fid, created_ts))
                saved += 1
                per_league_counter[league_id] = per_league_counter.get(league_id, 0) + 1

                if MAX_TIPS_PER_SCAN and saved >= MAX_TIPS_PER_SCAN:
                    break

            except Exception as e:
                log.exception("[SCAN] per-match failed: %s", e)
                continue

    log.info("[SCAN] saved=%d live_seen=%d", saved, live_seen)
    return saved, live_seen

def _format_tip_message(home, away, league, minute, score, suggestion, prob_pct, feat, odds=None, book=None, ev_pct=None):
    stat = ""
    if any([feat.get("xg_h",0),feat.get("xg_a",0),feat.get("sot_h",0),feat.get("sot_a",0),
            feat.get("cor_h",0),feat.get("cor_a",0),feat.get("pos_h",0),feat.get("pos_a",0),feat.get("red_sum",0)]):
        stat = (f"\n📊 xG {feat.get('xg_h',0):.2f}-{feat.get('xg_a',0):.2f}"
                f" • SOT {int(feat.get('sot_h',0))}-{int(feat.get('sot_a',0))}"
                f" • CK {int(feat.get('cor_h',0))}-{int(feat.get('cor_a',0))}")
        if feat.get("pos_h",0) or feat.get("pos_a",0): 
            stat += f" • POS {int(feat.get('pos_h',0))}%–{int(feat.get('pos_a',0))}%"
        if feat.get("red_sum",0):
            stat += f" • RED {int(feat.get('red_h',0))}-{int(feat.get('red_a',0))}"

    money = ""
    if odds:
        if ev_pct is not None:
            money = f"\n💰 <b>Odds:</b> {odds:.2f} @ {book or 'Book'}  •  <b>EV:</b> {ev_pct:+.1f}%"
        else:
            money = f"\n💰 <b>Odds:</b> {odds:.2f} @ {book or 'Book'}"
    return ("⚽️ <b>OU2.5 Tip</b>\n"
            f"<b>Match:</b> {home} vs {away}\n"
            f"🕒 <b>Minute:</b> {minute}'  |  <b>Score:</b> {score}\n"
            f"<b>Tip:</b> {suggestion}\n"
            f"📈 <b>Confidence:</b> {prob_pct:.1f}%{money}\n"
            f"🏆 <b>League:</b> {league}{stat}")

# ───────── Backfill results (used by digest) ─────────
def _fixture_by_id(mid: int) -> Optional[dict]:
    js=_api_get(FOOTBALL_API_URL, {"id": mid}) or {}
    arr=js.get("response") or [] if isinstance(js,dict) else []
    return arr[0] if arr else None

def _is_final(short: str) -> bool: return (short or "").upper() in {"FT","AET","PEN"}

def backfill_results_for_open_matches(max_rows: int = 200) -> int:
    now_ts=int(time.time()); cutoff=now_ts - 14*24*3600; updated=0
    with db_conn() as c:
        rows=c.execute("""
            WITH last AS (SELECT match_id, MAX(created_ts) last_ts FROM tips WHERE created_ts >= %s GROUP BY match_id)
            SELECT l.match_id FROM last l LEFT JOIN match_results r ON r.match_id=l.match_id
            WHERE r.match_id IS NULL ORDER BY l.last_ts DESC LIMIT %s
        """,(cutoff, max_rows)).fetchall()
    for (mid,) in rows:
        fx=_fixture_by_id(int(mid))
        if not fx: continue
        st=(((fx.get("fixture") or {}).get("status") or {}).get("short") or "")
        if not _is_final(st): continue
        g=fx.get("goals") or {}; gh=int(g.get("home") or 0); ga=int(g.get("away") or 0)
        btts=1 if (gh>0 and ga>0) else 0
        with db_conn() as c2:
            c2.execute("INSERT INTO match_results(match_id, final_goals_h, final_goals_a, btts_yes, updated_ts) "
                       "VALUES(%s,%s,%s,%s,%s) ON CONFLICT(match_id) DO UPDATE SET final_goals_h=EXCLUDED.final_goals_h, "
                       "final_goals_a=EXCLUDED.final_goals_a, btts_yes=EXCLUDED.btts_yes, updated_ts=EXCLUDED.updated_ts",
                       (int(mid), gh, ga, btts, int(time.time())))
        updated+=1
    if updated: log.info("[RESULTS] backfilled %d", updated)
    return updated

# ───────── Daily digest (OU 2.5 focused) ─────────
def daily_accuracy_digest() -> Optional[str]:
    today = datetime.now(BERLIN_TZ).date()
    start_of_day = datetime.combine(today, datetime.min.time(), tzinfo=BERLIN_TZ)
    start_ts = int(start_of_day.timestamp())

    backfill_results_for_open_matches(300)

    with db_conn() as c:
        rows = c.execute("""
            SELECT t.market, t.suggestion, t.confidence, t.confidence_raw, t.created_ts, t.odds,
                   r.final_goals_h, r.final_goals_a, r.btts_yes
            FROM tips t LEFT JOIN match_results r ON r.match_id=t.match_id
            WHERE t.created_ts >= %s AND t.market='Over/Under 2.5' AND t.sent_ok=1
            ORDER BY t.created_ts DESC
        """, (start_ts,)).fetchall()

    total = len(rows)
    graded = wins = 0
    roi_stake = 0.0
    roi_pnl = 0.0
    recent = []

    def _outcome(sugg: str, gh: int, ga: int) -> Optional[int]:
        total = gh + ga
        if sugg.startswith("Over"):
            if total > 2.5: return 1
            if abs(total - 2.5) < 1e-9: return None
            return 0
        else:
            if total < 2.5: return 1
            if abs(total - 2.5) < 1e-9: return None
            return 0

    for (mkt, sugg, conf, conf_raw, ts, odds, gh, ga, btts) in rows:
        tip_time = datetime.fromtimestamp(ts, BERLIN_TZ).strftime("%H:%M")
        recent.append(f"{sugg} ({conf:.1f}%) - {tip_time}")
        if gh is None or ga is None:
            continue
        res = _outcome(sugg, int(gh or 0), int(ga or 0))
        if res is None: 
            continue
        graded += 1
        if res == 1:
            wins += 1
        if odds:
            roi_stake += 1
            if res == 1:
                roi_pnl += float(odds) - 1.0
            else:
                roi_pnl -= 1.0

    if graded == 0:
        msg = f"📊 OU2.5 Digest {today.strftime('%Y-%m-%d')}\nNo graded tips yet."
    else:
        acc = 100.0 * wins / max(1, graded)
        roi = (100.0 * roi_pnl / max(1.0, roi_stake)) if roi_stake > 0 else 0.0
        msg = (f"📊 <b>OU 2.5 Digest</b> — {today.strftime('%Y-%m-%d')}\n"
               f"Tips sent: {total}  •  Graded: {graded}  •  Wins: {wins}  •  Accuracy: {acc:.1f}%  •  ROI {roi:+.1f}%")
        if recent:
            msg += f"\n🕒 Recent: {', '.join(recent[:5])}"
    send_telegram(msg)
    return msg

# ───────── HTTP endpoints ─────────
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")
def _require_admin():
    key = request.headers.get("X-API-Key") or request.args.get("key") or ((request.json or {}).get("key") if request.is_json else None)
    if not ADMIN_API_KEY or key != ADMIN_API_KEY: abort(401)

@app.route("/")
def root(): 
    return jsonify({"ok": True, "name": "goalsniper_ou25", "only_market": "Over/Under 2.5", "scheduler": RUN_SCHEDULER})

@app.route("/health")
def health():
    try:
        with db_conn() as c:
            n = c.execute("SELECT COUNT(*) FROM tips").fetchone()[0]
        api_ok = bool(_api_get(FOOTBALL_API_URL, {"live":"all"}))
        return jsonify({"ok": True, "db": "ok", "tips_count": int(n), "api_connected": api_ok, "timestamp": time.time()})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/init-db", methods=["POST"])
def http_init_db(): _require_admin(); init_db(); return jsonify({"ok": True})

@app.route("/admin/scan", methods=["POST","GET"])
def http_scan(): _require_admin(); s,l=production_scan(); return jsonify({"ok": True, "saved": s, "live_seen": l})

@app.route("/admin/backfill-results", methods=["POST","GET"])
def http_backfill(): _require_admin(); n=backfill_results_for_open_matches(400); return jsonify({"ok": True, "updated": n})

@app.route("/admin/digest", methods=["POST","GET"])
def http_digest(): _require_admin(); msg=daily_accuracy_digest(); return jsonify({"ok": True, "sent": bool(msg)})

@app.route("/tips/latest")
def http_latest():
    limit=int(request.args.get("limit","50"))
    with db_conn() as c:
        rows=c.execute("SELECT match_id,league,home,away,market,suggestion,confidence,confidence_raw,score_at_tip,minute,created_ts,odds,book,ev_pct "
                       "FROM tips WHERE market='Over/Under 2.5' ORDER BY created_ts DESC LIMIT %s",(max(1,min(500,limit)),)).fetchall()
    tips=[]
    for r in rows:
        tips.append({"match_id":int(r[0]),"league":r[1],"home":r[2],"away":r[3],"market":r[4],"suggestion":r[5],
                     "confidence":float(r[6]),"confidence_raw":(float(r[7]) if r[7] is not None else None),
                     "score_at_tip":r[8],"minute":int(r[9]),"created_ts":int(r[10]),
                     "odds": (float(r[11]) if r[11] is not None else None), "book": r[12], "ev_pct": (float(r[13]) if r[13] is not None else None)})
    return jsonify({"ok": True, "tips": tips})

# ───────── Scheduler ─────────
_scheduler_started=False
def _start_scheduler_once():
    global _scheduler_started
    if _scheduler_started or not RUN_SCHEDULER: return
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        sched = BackgroundScheduler(timezone=TZ_UTC)
        sched.add_job(production_scan, "interval", seconds=SCAN_INTERVAL_SEC, id="scan", max_instances=1, coalesce=True)
        sched.add_job(lambda: backfill_results_for_open_matches(400), "interval", minutes=int(os.getenv("BACKFILL_EVERY_MIN","15")), id="backfill", max_instances=1, coalesce=True)
        sched.add_job(daily_accuracy_digest, "cron", hour=int(os.getenv("DAILY_ACCURACY_HOUR","3")), minute=int(os.getenv("DAILY_ACCURACY_MINUTE","6")), id="digest", max_instances=1, coalesce=True, timezone=BERLIN_TZ)
        sched.start()
        _scheduler_started=True
        send_telegram("🚀 goalsniper OU 2.5–only mode started.")
        log.info("[SCHED] started (scan=%ss)", SCAN_INTERVAL_SEC)
    except Exception as e:
        log.exception("[SCHED] failed: %s", e)

# ───────── Boot ─────────
def _on_boot():
    _init_pool()
    init_db()
    _start_scheduler_once()

_on_boot()

if __name__ == "__main__":
    app.run(host=os.getenv("HOST","0.0.0.0"), port=int(os.getenv("PORT","8080")))
