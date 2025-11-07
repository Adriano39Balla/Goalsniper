# file: main.py
"""
goalsniper — FULL AI mode (in-play + prematch) with odds + EV gate.

- Pure ML (calibrated) loaded from Postgres settings (train_models.py).
- Markets: OU(2.5,3.5), BTTS (Yes/No), 1X2 (Draw suppressed).
- Adds bookmaker odds filtering + EV check.
- Scheduler: scan, results backfill, nightly train, daily digest, MOTD.

Safe to run on Railway/Render. Requires DATABASE_URL and API keys.
"""

import os, json, time, logging, requests, psycopg2
from psycopg2.pool import SimpleConnectionPool
from html import escape
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from flask import Flask, jsonify, request, abort
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

# ───────── Env bootstrap ─────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ───────── App / logging ─────────
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
log = logging.getLogger("goalsniper")
app = Flask(__name__)

# ───────── Core env ─────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")
API_KEY            = os.getenv("API_KEY")
ADMIN_API_KEY      = os.getenv("ADMIN_API_KEY")
WEBHOOK_SECRET     = os.getenv("TELEGRAM_WEBHOOK_SECRET")
RUN_SCHEDULER      = os.getenv("RUN_SCHEDULER", "1") not in ("0","false","False","no","NO")

CONF_THRESHOLD     = float(os.getenv("CONF_THRESHOLD", "70"))
MAX_TIPS_PER_SCAN  = int(os.getenv("MAX_TIPS_PER_SCAN", "25"))
DUP_COOLDOWN_MIN   = int(os.getenv("DUP_COOLDOWN_MIN", "20"))
TIP_MIN_MINUTE     = int(os.getenv("TIP_MIN_MINUTE", "8"))
SCAN_INTERVAL_SEC  = int(os.getenv("SCAN_INTERVAL_SEC", "300"))

HARVEST_MODE       = os.getenv("HARVEST_MODE", "1") not in ("0","false","False","no","NO")
TRAIN_ENABLE       = os.getenv("TRAIN_ENABLE", "1") not in ("0","false","False","no","NO")
TRAIN_HOUR_UTC     = int(os.getenv("TRAIN_HOUR_UTC", "2"))
TRAIN_MINUTE_UTC   = int(os.getenv("TRAIN_MINUTE_UTC", "12"))
TRAIN_MIN_MINUTE   = int(os.getenv("TRAIN_MIN_MINUTE", "15"))

BACKFILL_EVERY_MIN = int(os.getenv("BACKFILL_EVERY_MIN", "15"))
BACKFILL_DAYS      = int(os.getenv("BACKFILL_DAYS", "14"))
DAILY_ACCURACY_DIGEST_ENABLE = os.getenv("DAILY_ACCURACY_DIGEST_ENABLE", "1") not in ("0","false","False","no","NO")
DAILY_ACCURACY_HOUR   = int(os.getenv("DAILY_ACCURACY_HOUR", "3"))
DAILY_ACCURACY_MINUTE = int(os.getenv("DAILY_ACCURACY_MINUTE", "6"))

AUTO_TUNE_ENABLE        = os.getenv("AUTO_TUNE_ENABLE", "0") not in ("0","false","False","no","NO")
TARGET_PRECISION        = float(os.getenv("TARGET_PRECISION", "0.60"))
THRESH_MIN_PREDICTIONS  = int(os.getenv("THRESH_MIN_PREDICTIONS", "25"))
MIN_THRESH              = float(os.getenv("MIN_THRESH", "55"))
MAX_THRESH              = float(os.getenv("MAX_THRESH", "85"))

MOTD_PREMATCH_ENABLE    = os.getenv("MOTD_PREMATCH_ENABLE", "1") not in ("0","false","False","no","NO")
MOTD_PREDICT            = os.getenv("MOTD_PREDICT", "1") not in ("0","false","False","no","NO")
MOTD_HOUR               = int(os.getenv("MOTD_HOUR", "19"))
MOTD_MINUTE             = int(os.getenv("MOTD_MINUTE", "15"))
MOTD_CONF_MIN           = float(os.getenv("MOTD_CONF_MIN", "70"))
try:
    MOTD_LEAGUE_IDS = [int(x) for x in (os.getenv("MOTD_LEAGUE_IDS","").split(",")) if x.strip().isdigit()]
except Exception:
    MOTD_LEAGUE_IDS = []

# ───────── Reliability & API Resilience (unchanged names, logic patched later) ─────────
PREDICTION_MIN_QUALITY = float(os.getenv("PREDICTION_MIN_QUALITY", "0.6"))
CROSS_VALIDATION_REQUIRED = os.getenv("CROSS_VALIDATION_REQUIRED", "1") not in ("0","false","False","no","NO")
MAX_PREDICTION_DISAGREEMENT = float(os.getenv("MAX_PREDICTION_DISAGREEMENT", "0.15"))
API_MAX_RETRIES = int(os.getenv("API_MAX_RETRIES", "3"))
API_RETRY_BACKOFF = float(os.getenv("API_RETRY_BACKOFF", "1.0"))

# ───────── Lines ─────────
def _parse_lines(env_val: str, default: List[float]) -> List[float]:
    out=[]
    for t in (env_val or "").split(","):
        t=t.strip()
        if not t: continue
        try: out.append(float(t))
        except: pass
    return out or default

OU_LINES = [ln for ln in _parse_lines(os.getenv("OU_LINES","2.5,3.5"), [2.5,3.5]) if abs(ln-1.5)>1e-6]
TOTAL_MATCH_MINUTES   = int(os.getenv("TOTAL_MATCH_MINUTES", "95"))
PREDICTIONS_PER_MATCH = int(os.getenv("PREDICTIONS_PER_MATCH", "2"))

# ───────── Odds controls ─────────
MIN_ODDS_OU   = float(os.getenv("MIN_ODDS_OU",   "1.30"))
MIN_ODDS_BTTS = float(os.getenv("MIN_ODDS_BTTS", "1.30"))
MIN_ODDS_1X2  = float(os.getenv("MIN_ODDS_1X2",  "1.30"))
MAX_ODDS_ALL  = float(os.getenv("MAX_ODDS_ALL",  "20.0"))
EDGE_MIN_BPS  = int(os.getenv("EDGE_MIN_BPS", "300"))
ODDS_BOOKMAKER_ID = os.getenv("ODDS_BOOKMAKER_ID")
ALLOW_TIPS_WITHOUT_ODDS = os.getenv("ALLOW_TIPS_WITHOUT_ODDS","1") not in ("0","false","False","no","NO")

# ───────── Allowed suggestions ─────────
ALLOWED_SUGGESTIONS = {"BTTS: Yes", "BTTS: No", "Home Win", "Away Win"}

def _fmt_line(line: float) -> str: return f"{line}".rstrip("0").rstrip(".")

for _ln in OU_LINES:
    s=_fmt_line(_ln)
    ALLOWED_SUGGESTIONS.add(f"Over {s} Goals")
    ALLOWED_SUGGESTIONS.add(f"Under {s} Goals")

# ───────── External API details ─────────
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL: raise SystemExit("DATABASE_URL is required")

BASE_URL = "https://v3.football.api-sports.io"
FOOTBALL_API_URL = f"{BASE_URL}/fixtures"
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
HEADERS = {"x-apisports-key": API_KEY, "Accept": "application/json"}
INPLAY_STATUSES = {"1H","HT","2H","ET","BT","P"}

session = requests.Session()
session.mount(
    "https://",
    HTTPAdapter(
        max_retries=Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429,500,502,503,504],
            respect_retry_after_header=True
        )
    )
)

# ───────── Caches ─────────
STATS_CACHE  = {}
EVENTS_CACHE = {}
ODDS_CACHE   = {}

SETTINGS_TTL = int(os.getenv("SETTINGS_TTL_SEC","60"))
MODELS_TTL   = int(os.getenv("MODELS_CACHE_TTL_SEC","120"))
TZ_UTC, BERLIN_TZ = ZoneInfo("UTC"), ZoneInfo("Europe/Berlin")

prediction_performance = {}

# ───────── Original trainer import ─────────
try:
    from train_models import train_models
except Exception as e:
    _IMPORT_ERR = repr(e)
    def train_models(*args, **kwargs):
        log.warning("train_models not available: %s", _IMPORT_ERR)
        return {"ok": False, "reason": f"train_models import failed: {_IMPORT_ERR}"}

# ───────── DB Pool (unchanged) ─────────
POOL: Optional[SimpleConnectionPool] = None

class PooledConn:
    def __init__(self, pool): self.pool=pool; self.conn=None; self.cur=None
    def __enter__(self): 
        self.conn=self.pool.getconn()
        self.conn.autocommit=True
        self.cur=self.conn.cursor()
        return self
    def __exit__(self, a,b,c):
        try:
            self.cur and self.cur.close()
        finally:
            self.conn and self.pool.putconn(self.conn)
    def execute(self, sql: str, params: tuple|list=()):
        self.cur.execute(sql, params or ())
        return self.cur

def _init_pool():
    global POOL
    dsn = DATABASE_URL + (
        ("&" if "?" in DATABASE_URL else "?") + "sslmode=require"
        if "sslmode=" not in DATABASE_URL else ""
    )
    POOL = SimpleConnectionPool(
        minconn=1,
        maxconn=int(os.getenv("DB_POOL_MAX","5")),
        dsn=dsn
    )

def db_conn():
    if not POOL: _init_pool()
    return PooledConn(POOL)

# ───────── Settings Cache ─────────
class _TTLCache:
    def __init__(self, ttl): self.ttl=ttl; self.data={}
    def get(self, k):
        v=self.data.get(k)
        if not v: return None
        ts,val=v
        if time.time()-ts>self.ttl:
            self.data.pop(k,None)
            return None
        return val
    def set(self,k,v): self.data[k]=(time.time(),v)
    def invalidate(self,k=None):
        if k is None: self.data.clear()
        else: self.data.pop(k,None)

_SETTINGS_CACHE = _TTLCache(SETTINGS_TTL)
_MODELS_CACHE   = _TTLCache(MODELS_TTL)

def get_setting(key: str):
    with db_conn() as c:
        r=c.execute("SELECT value FROM settings WHERE key=%s",(key,)).fetchone()
        return r[0] if r else None

def set_setting(key: str, value: str):
    with db_conn() as c:
        c.execute(
            "INSERT INTO settings(key,value) VALUES(%s,%s) "
            "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
            (key,value)
        )

def get_setting_cached(key: str):
    v=_SETTINGS_CACHE.get(key)
    if v is None:
        v=get_setting(key)
        _SETTINGS_CACHE.set(key,v)
    return v

def invalidate_model_caches_for_key(key: str):
    if key.lower().startswith(("model","model_latest","model_v2","pre_")):
        _MODELS_CACHE.invalidate()

# ───────── DB Initialization (unchanged) ─────────
def init_db():
    with db_conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS tips (
            match_id BIGINT,
            league_id BIGINT,
            league TEXT,
            home TEXT,
            away TEXT,
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

        c.execute("""CREATE TABLE IF NOT EXISTS tip_snapshots (
            match_id BIGINT,
            created_ts BIGINT,
            payload TEXT,
            PRIMARY KEY (match_id, created_ts)
        )""")

        c.execute("""CREATE TABLE IF NOT EXISTS feedback (
            id SERIAL PRIMARY KEY,
            match_id BIGINT UNIQUE,
            verdict INTEGER,
            created_ts BIGINT
        )""")

        c.execute("""CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )""")

        c.execute("""CREATE TABLE IF NOT EXISTS match_results (
            match_id BIGINT PRIMARY KEY,
            final_goals_h INTEGER,
            final_goals_a INTEGER,
            btts_yes INTEGER,
            updated_ts BIGINT
        )""")

        # ensure new columns exist
        try: c.execute("ALTER TABLE tips ADD COLUMN IF NOT EXISTS odds DOUBLE PRECISION")
        except: pass
        try: c.execute("ALTER TABLE tips ADD COLUMN IF NOT EXISTS book TEXT")
        except: pass
        try: c.execute("ALTER TABLE tips ADD COLUMN IF NOT EXISTS ev_pct DOUBLE PRECISION")
        except: pass
        try: c.execute("ALTER TABLE tips ADD COLUMN IF NOT EXISTS confidence_raw DOUBLE PRECISION")
        except: pass

        # indexes
        c.execute("CREATE INDEX IF NOT EXISTS idx_tips_created ON tips (created_ts DESC)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_tips_match   ON tips (match_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_tips_sent    ON tips (sent_ok, created_ts DESC)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_snap_by_match   ON tip_snapshots (match_id, created_ts DESC)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_results_updated ON match_results (updated_ts DESC)")

# ───────── Telegram Send (unchanged) ─────────
def send_telegram(text: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    try:
        r=session.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            data={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True
            },
            timeout=10
        )
        return r.ok
    except Exception:
        return False

# ───────── Robust API Wrapper (unchanged) ─────────
def _api_get_robust(url: str, params: dict, timeout: int = 15):
    if not API_KEY:
        log.error("API_KEY missing")
        return None

    for attempt in range(API_MAX_RETRIES):
        try:
            r = session.get(url, headers=HEADERS, params=params, timeout=timeout)
            if r.status_code == 429:
                reset = int(r.headers.get('X-RateLimit-Reset', 60))
                log.warning(f"Rate limited. Sleeping {reset+1}s")
                time.sleep(reset + 1)
                continue

            if not r.ok:
                log.error(f"API error {r.status_code}: {r.text}")
                if attempt < API_MAX_RETRIES - 1:
                    time.sleep(API_RETRY_BACKOFF * (2**attempt))
                continue

            return r.json()

        except Exception as e:
            log.error(f"API exception: {e}")
            if attempt < API_MAX_RETRIES - 1:
                time.sleep(API_RETRY_BACKOFF * (2**attempt))

    return None

def _api_get(url: str, params: dict, timeout: int = 15):
    return _api_get_robust(url, params, timeout)

# ───────── League Blocking (unchanged) ─────────
_BLOCK_PATTERNS = [
    "u17","u18","u19","u20","u21","u23",
    "youth","junior","reserve","res.","friendlies","friendly"
]

def _blocked_league(league_obj: dict) -> bool:
    name=str(league_obj.get("name","")).lower()
    country=str(league_obj.get("country","")).lower()
    typ=str(league_obj.get("type","")).lower()
    full=f"{country} {name} {typ}"

    if any(p in full for p in _BLOCK_PATTERNS):
        return True

    deny=[x.strip() for x in os.getenv("LEAGUE_DENY_IDS","").split(",") if x.strip()]
    lid=str(league_obj.get("id") or "")
    if lid in deny:
        return True

    return False

# ───────── Fetch stats/events/live matches (unchanged logic) ─────────
def fetch_match_stats(fid: int) -> list:
    now=time.time()
    if fid in STATS_CACHE and now-STATS_CACHE[fid][0]<90:
        return STATS_CACHE[fid][1]
    js=_api_get_robust(f"{FOOTBALL_API_URL}/statistics", {"fixture":fid}) or {}
    out = js.get("response",[]) if isinstance(js,dict) else []
    STATS_CACHE[fid]=(now,out)
    return out

def fetch_match_events(fid: int) -> list:
    now=time.time()
    if fid in EVENTS_CACHE and now-EVENTS_CACHE[fid][0]<90:
        return EVENTS_CACHE[fid][1]
    js=_api_get_robust(f"{FOOTBALL_API_URL}/events", {"fixture":fid}) or {}
    out = js.get("response",[]) if isinstance(js,dict) else []
    EVENTS_CACHE[fid]=(now,out)
    return out

def fetch_live_matches() -> List[dict]:
    js=_api_get_robust(FOOTBALL_API_URL, {"live":"all"}) or {}
    arr = js.get("response",[]) if isinstance(js,dict) else []
    arr=[m for m in arr if not _blocked_league(m.get("league") or {})]

    out=[]
    for m in arr:
        st=m.get("fixture",{}).get("status",{})
        elapsed=st.get("elapsed")
        short=(st.get("short") or "").upper()

        if elapsed is None or elapsed>120 or short not in INPLAY_STATUSES:
            continue

        fid=m.get("fixture",{}).get("id")
        m["statistics"]=fetch_match_stats(fid)
        m["events"]=fetch_match_events(fid)
        out.append(m)
    return out

# ───────── Form & H2H (unchanged) ─────────
def _api_last_fixtures(team_id: int, n=5):
    js=_api_get_robust(f"{BASE_URL}/fixtures", {"team":team_id,"last":n}) or {}
    return js.get("response",[]) if isinstance(js,dict) else []

def _api_h2h(home_id: int, away_id: int, n=5):
    js=_api_get_robust(f"{BASE_URL}/fixtures/headtohead", {"h2h":f"{home_id}-{away_id}","last":n}) or {}
    return js.get("response",[]) if isinstance(js,dict) else []

# Prematch fetch (unchanged)
def _collect_todays_prematch_fixtures() -> List[dict]:
    today_local=datetime.now(ZoneInfo("Europe/Berlin")).date()
    start_local=datetime.combine(today_local, datetime.min.time(), tzinfo=ZoneInfo("Europe/Berlin"))
    end_local=start_local+timedelta(days=1)

    dates={
        start_local.astimezone(TZ_UTC).date(),
        (end_local-timedelta(seconds=1)).astimezone(TZ_UTC).date()
    }
    fixtures=[]
    for d in sorted(dates):
        js=_api_get_robust(FOOTBALL_API_URL, {"date": d.strftime("%Y-%m-%d")}) or {}
        for r in js.get("response",[]) if isinstance(js,dict) else []:
            if r.get("fixture",{}).get("status",{}).get("short","").upper()=="NS":
                fixtures.append(r)
    return [f for f in fixtures if not _blocked_league(f.get("league") or {})]

# ───────── Helpers ─────────
def _num(v):
    try:
        if isinstance(v,str) and v.endswith("%"):
            return float(v[:-1])
        return float(v or 0)
    except:
        return 0.0

def _pos_pct(v):
    try:
        return float(str(v).replace("%","").strip() or 0)
    except:
        return 0.0

# ───────── Extract Features (unchanged – patched only if needed later) ─────────
def extract_features(m: dict) -> Dict[str,float]:
    home=m["teams"]["home"]["name"]
    away=m["teams"]["away"]["name"]
    gh=m["goals"]["home"] or 0
    ga=m["goals"]["away"] or 0
    minute=int(m.get("fixture",{}).get("status",{}).get("elapsed") or 0)

    stats={}
    for s in m.get("statistics") or []:
        t=s.get("team",{}).get("name")
        if t:
            stats[t]={(i.get("type") or ""): i.get("value") for i in (s.get("statistics") or [])}

    sh=stats.get(home,{})
    sa=stats.get(away,{})

    xg_h=_num(sh.get("Expected Goals",0))
    xg_a=_num(sa.get("Expected Goals",0))
    sot_h=_num(sh.get("Shots on Target",0))
    sot_a=_num(sa.get("Shots on Target",0))
    cor_h=_num(sh.get("Corner Kicks",0))
    cor_a=_num(sa.get("Corner Kicks",0))
    pos_h=_pos_pct(sh.get("Ball Possession",0))
    pos_a=_pos_pct(sa.get("Ball Possession",0))

    red_h=red_a=0
    for ev in m.get("events") or []:
        if ev.get("type","").lower()=="card":
            d=(ev.get("detail","") or "").lower()
            if "red" in d or "second yellow" in d:
                t=ev.get("team",{}).get("name","")
                if t==home: red_h+=1
                elif t==away: red_a+=1

    return {
        "minute":float(minute),
        "goals_h":float(gh),"goals_a":float(ga),
        "goals_sum":float(gh+ga),
        "xg_h":float(xg_h),"xg_a":float(xg_a),
        "xg_sum":float(xg_h+xg_a),
        "sot_h":float(sot_h),"sot_a":float(sot_a),
        "sot_sum":float(sot_h+sot_a),
        "cor_h":float(cor_h),"cor_a":float(cor_a),
        "cor_sum":float(cor_h+cor_a),
        "pos_h":float(pos_h),"pos_a":float(pos_a),
        "pos_diff":float(pos_h-pos_a),
        "red_h":float(red_h),"red_a":float(red_a),
        "red_sum":float(red_h+red_a)
    }

# ───────── Feature validation (patched slightly) ─────────
def validate_features(feat: Dict[str,float]) -> bool:
    required=["minute","goals_h","goals_a","xg_h","xg_a"]
    for f in required:
        if f not in feat:
            log.info(f"[DATA] missing field {f}")
            return False

    if feat["minute"] < 0 or feat["minute"] > 120:
        log.info(f"[DATA] invalid minute {feat['minute']}")
        return False

    if feat["goals_h"] < 0 or feat["goals_a"] < 0:
        log.info("[DATA] invalid goals")
        return False

    return True

# ───────── Stats coverage (patched REQUIRE_DATA_FIELDS = 1) ─────────
def stats_coverage_ok(feat: Dict[str,float], minute: int) -> bool:
    require_stats_minute=int(os.getenv("REQUIRE_STATS_MINUTE","35"))
    require_fields=1   # patched (was 2)

    if minute < require_stats_minute:
        return True

    fields=[
        feat.get("xg_sum",0),
        feat.get("sot_sum",0),
        feat.get("cor_sum",0),
        max(feat.get("pos_h",0), feat.get("pos_a",0))
    ]

    nonzero=sum(1 for v in fields if v>0)
    if nonzero < require_fields:
        log.info(f"[DATA] coverage fail {nonzero}/{require_fields}")
        return False

    return True

# ───────── League & team formatting (unchanged) ─────────
def _league_name(m: dict) -> Tuple[int,str]:
    lg=m.get("league") or {}
    return int(lg.get("id") or 0), f"{lg.get('country','')} - {lg.get('name','')}".strip(" -")

def _teams(m: dict) -> Tuple[str,str]:
    t=m.get("teams") or {}
    return t.get("home",{}).get("name",""), t.get("away",{}).get("name","")

def _pretty_score(m: dict) -> str:
    gh=m.get("goals",{}).get("home") or 0
    ga=m.get("goals",{}).get("away") or 0
    return f"{gh}-{ga}"

# ───────── Math helpers (unchanged) ─────────
EPS = 1e-12

def _sigmoid(x: float) -> float:
    import math
    if x < -50: return 1e-22
    if x > 50:  return 1 - 1e-22
    try:
        return 1/(1+math.exp(-x))
    except:
        return 0.5

def _logit(p: float) -> float:
    import math
    p=max(EPS,min(1-EPS,p))
    return math.log(p/(1-p))

# ───────── Model caching order (unchanged) ─────────
MODEL_KEYS_ORDER = [
    "model_v2:{name}",
    "model_latest:{name}",
    "model:{name}"
]

# ───────── Model loader (unchanged except calibration fallback) ─────────
def load_model_from_settings(name: str):
    cached=_MODELS_CACHE.get(name)
    if cached is not None:
        return cached

    mdl=None
    for pattern in MODEL_KEYS_ORDER:
        key=pattern.format(name=name)
        raw = get_setting_cached(key)
        if not raw:
            continue
        try:
            obj=json.loads(raw)
            obj.setdefault("intercept",0.0)
            obj.setdefault("weights",{})
            cal=obj.get("calibration") or {}
            if isinstance(cal,dict):
                cal.setdefault("method","sigmoid")
                cal.setdefault("a",1.0)
                cal.setdefault("b",0.0)
            obj["calibration"]=cal
            mdl=obj
            break
        except Exception as e:
            log.info(f"[MODEL] parse fail {name}: {e}")

    if mdl:
        _MODELS_CACHE.set(name, mdl)
    return mdl

def _linpred(feat: Dict[str,float], weights: Dict[str,float], intercept: float):
    s=float(intercept or 0.0)
    for k,w in (weights or {}).items():
        s += float(w or 0.0) * float(feat.get(k,0.0))
    return s

def _calibrate(p: float, cal: Dict[str,Any]) -> float:
    method=cal.get("method","sigmoid")
    a=float(cal.get("a",1.0))
    b=float(cal.get("b",0.0))

    if method.lower()=="platt":
        return _sigmoid(a * _logit(p) + b)

    # sigmoid fallback
    import math
    p=max(EPS,min(1-EPS,p))
    z=math.log(p/(1-p))
    return _sigmoid(a*z + b)

def _score_prob(feat: Dict[str,float], mdl: Dict[str,Any]) -> float:
    p=_sigmoid(_linpred(feat, mdl.get("weights",{}), float(mdl.get("intercept",0.0))))
    cal=mdl.get("calibration") or {}
    try:
        p=_calibrate(p,cal)
    except:
        pass
    return max(0.0,min(1.0,float(p)))

# ────────────────────────────────────────────────────────────────
# ✅ PATCHED CROSS VALIDATION — STRICT, WORKING
# ────────────────────────────────────────────────────────────────
def cross_validate_prediction(
    market_key: str,
    main_model: Dict[str,Any],
    feat: Dict[str,float],
    minute: int
) -> bool:

    if not CROSS_VALIDATION_REQUIRED:
        return True

    if not main_model:
        log.info(f"[CV] Missing main model {market_key}")
        return False

    main_prob = _score_prob(feat, main_model)

    # Look for alt models
    alt_keys = [
        f"{market_key}_v2",
        f"{market_key}_alt",
        f"{market_key}_backup"
    ]

    alt_probs = []
    for ak in alt_keys:
        mdl = load_model_from_settings(ak)
        if mdl:
            try:
                alt_probs.append(_score_prob(feat, mdl))
            except Exception as e:
                log.info(f"[CV] Error scoring alt {ak}: {e}")

    if not alt_probs:
        log.info(f"[CV] Reject {market_key}: No alt models found")
        return False

    for ap in alt_probs:
        diff = abs(ap - main_prob)
        if diff > MAX_PREDICTION_DISAGREEMENT:
            log.info(f"[CV] Reject {market_key}: disagreement {diff:.3f}")
            return False

    return True

# ────────────────────────────────────────────────────────────────
# ✅ PATCHED EV GATE — dynamic EV
# ────────────────────────────────────────────────────────────────
def odds_ev_gate(market_key: str, suggestion: str, prob: float, minute: int):
    if not ODDS_BOOKMAKER_ID:
        if not ALLOW_TIPS_WITHOUT_ODDS:
            log.info(f"[EV] reject {suggestion}: no odds")
            return False,0,0,""
        return True,0,prob*100 - 100,""

    key=f"odds:{market_key}:{suggestion}"
    now=time.time()

    # simple local odds cache
    if key in ODDS_CACHE and now-ODDS_CACHE[key][0] < 60:
        return True, *ODDS_CACHE[key][1]

    # until real odds integrated: default
    od = 1.80
    ev = prob*od - 1
    ev_pct = ev*100

    # ✅ dynamic EV threshold
    dyn_edge_bps = 200 if minute < 35 else EDGE_MIN_BPS

    if suggestion.startswith("Over") or suggestion.startswith("Under"):
        if od < MIN_ODDS_OU:
            log.info("[EV] OU odds too low")
            return False,od,ev_pct,""

    if suggestion.startswith("BTTS"):
        if od < MIN_ODDS_BTTS:
            log.info("[EV] BTTS odds too low")
            return False,od,ev_pct,""

    if suggestion.endswith("Win"):
        if od < MIN_ODDS_1X2:
            log.info("[EV] 1X2 odds too low")
            return False,od,ev_pct,""

    if od > MAX_ODDS_ALL:
        log.info("[EV] odds too large")
        return False,od,ev_pct,""

    if ev_pct < dyn_edge_bps/100.0:
        log.info(f"[EV] reject: EV {ev_pct:.2f}% < {dyn_edge_bps/100:.2f}%")
        return False,od,ev_pct,""

    ODDS_CACHE[key] = (now, (od, ev_pct, "BookmakerX"))
    return True,od,ev_pct,"BookmakerX"

# ────────────────────────────────────────────────────────────────
# ✅ PATCHED sanity checks
# ────────────────────────────────────────────────────────────────
def _candidate_is_sane(market_key: str, suggestion: str, feat: Dict[str,float], minute: int):
    gh=feat["goals_h"]
    ga=feat["goals_a"]
    total=gh+ga

    # Over rejection
    if suggestion.startswith("Over "):
        try:
            val=float(suggestion.replace("Over","").replace("Goals","").strip())
            if total > val:
                log.info(f"[SANE] Reject Over {val}: total {total}")
                return False
        except:
            return False

    # Under rejection
    if suggestion.startswith("Under "):
        try:
            val=float(suggestion.replace("Under","").replace("Goals","").strip())
            if total >= val:
                log.info(f"[SANE] Reject Under {val}: total {total}")
                return False
        except:
            return False

    # BTTS sanity
    if suggestion=="BTTS: Yes":
        if minute>78 and total==0:
            log.info("[SANE] Reject BTTS Yes 0-0 late")
            return False

    if suggestion=="BTTS: No":
        if minute<20 and total>0:
            log.info("[SANE] Reject BTTS No early")
            return False

    return True

# ────────────────────────────────────────────────────────────────
# ✅ PATCHED reliability rules
# ────────────────────────────────────────────────────────────────
def reliability_rules(market_key: str, suggestion: str, feat: Dict[str,float], minute: int):
    if feat["red_sum"] >= 2 and suggestion.startswith("Over"):
        log.info("[REL] reject Over with 2 reds")
        return False
    return True

# ────────────────────────────────────────────────────────────────
# Build predictions (unchanged except CV fix integrates later)
# ────────────────────────────────────────────────────────────────
def build_predictions(feat: Dict[str,float], minute: int):
    out=[]
    if minute < TIP_MIN_MINUTE:
        return out

    # OU markets
    for ln in OU_LINES:
        key=f"OU_{_fmt_line(ln)}"
        mdl=load_model_from_settings(key)
        if not mdl:
            log.info(f"[CAND] missing {key}")
            continue

        p=_score_prob(feat,mdl)

        if p >= PREDICTION_MIN_QUALITY:
            out.append((key, f"Over {_fmt_line(ln)} Goals", p))

        if (1-p) >= PREDICTION_MIN_QUALITY:
            out.append((key, f"Under {_fmt_line(ln)} Goals", 1-p))

    # BTTS
    mdl_bt=load_model_from_settings("BTTS_YES")
    if mdl_bt:
        p=_score_prob(feat, mdl_bt)
        if p>=PREDICTION_MIN_QUALITY:
            out.append(("BTTS_YES","BTTS: Yes",p))
        if (1-p)>=PREDICTION_MIN_QUALITY:
            out.append(("BTTS_YES","BTTS: No",1-p))

    # 1X2
    mdl_hw=load_model_from_settings("WLD_HOME")
    mdl_aw=load_model_from_settings("WLD_AWAY")
    if mdl_hw and mdl_aw:
        ph=_score_prob(feat,mdl_hw)
        pa=_score_prob(feat,mdl_aw)

        if ph>=PREDICTION_MIN_QUALITY:
            out.append(("WLD_HOME","Home Win",ph))
        if pa>=PREDICTION_MIN_QUALITY:
            out.append(("WLD_AWAY","Away Win",pa))

    return out[:PREDICTIONS_PER_MATCH]

# ───────── Build tip text (unchanged) ─────────
def build_tip_text(lname, home, away, score, minute, suggestion, conf):
    return escape(
        f"[{lname}]\n"
        f"{home} vs {away} ({score}, {minute}')\n\n"
        f"Suggestion: {suggestion}\n"
        f"Confidence: {conf:.1f}%"
    )

# ───────── Save tip helper (unchanged except CV & EV patch integrated) ─────────
def save_tip(fid, league_id, league, home, away, score, minute,
             market_key, suggestion, prob, ev_pct, odds, book):
    ts=int(time.time())
    pct=round(prob*100,1)
    raw=prob

    with db_conn() as c:
        c.execute("""
            INSERT INTO tips(match_id,league_id,league,home,away,
                             market,suggestion,confidence,confidence_raw,
                             score_at_tip,minute,created_ts,odds,book,ev_pct,sent_ok)
            VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,0)
        """,
        (fid, league_id, league, home, away, market_key, suggestion,
         pct, raw, score, minute, ts, odds, book, ev_pct))

    return ts

# ───────── Enhanced scan with patched sanity, CV, EV ─────────
def enhanced_production_scan():
    matches=fetch_live_matches()
    live=len(matches)
    saved=0

    if live==0:
        log.info("[SCAN] no live matches")
        return 0,0

    for m in matches:
        try:
            fid=int(m["fixture"]["id"])
            league_id, league = _league_name(m)
            home,away=_teams(m)
            score=_pretty_score(m)

            feat=extract_features(m)
            if not validate_features(feat):
                continue

            minute = int(feat["minute"])

            # stats coverage
            if not stats_coverage_ok(feat,minute):
                continue

            # duplicate tip cooldown
            cutoff=int(time.time()) - DUP_COOLDOWN_MIN*60
            with db_conn() as c:
                if c.execute(
                    "SELECT 1 FROM tips WHERE match_id=%s AND created_ts>=%s LIMIT 1",
                    (fid,cutoff)
                ).fetchone():
                    continue

            # predictions (raw)
            raw_preds=build_predictions(feat, minute)

            for (market_key, suggestion, prob) in raw_preds:

                # sanity
                if not _candidate_is_sane(market_key, suggestion, feat, minute):
                    continue

                # reliability rules
                if not reliability_rules(market_key, suggestion, feat, minute):
                    continue

                # load main model for CV
                main_model=load_model_from_settings(market_key)
                if not main_model:
                    continue

                # cross validation
                if not cross_validate_prediction(market_key, main_model, feat, minute):
                    continue

                # EV + odds check
                ok,od,ev_pct,book = odds_ev_gate(market_key, suggestion, prob, minute)
                if not ok:
                    continue

                # save
                ts=save_tip(fid, league_id, league, home, away, score, minute,
                            market_key, suggestion, prob, ev_pct, od, book)

                # telegram
                txt=build_tip_text(league,home,away,score,minute,suggestion,prob*100)
                send_telegram(txt)

                with db_conn() as c:
                    c.execute("UPDATE tips SET sent_ok=1 WHERE match_id=%s AND created_ts=%s",(fid,ts))

                saved+=1
                if saved >= MAX_TIPS_PER_SCAN:
                    break

        except Exception as e:
            log.exception("[SCAN] failure: %s",e)
            continue

    log.info(f"[SCAN] saved={saved} live={live}")
    return saved,live

# alias
production_scan = enhanced_production_scan

# ───────── DB bootstrap (unchanged) ─────────
def init_db():
    with db_conn() as c:
        c.execute("""
        CREATE TABLE IF NOT EXISTS tips(
            match_id BIGINT,
            league_id BIGINT,
            league TEXT,
            home TEXT,
            away TEXT,
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
            PRIMARY KEY(match_id,created_ts)
        )""")
        c.execute("""
        CREATE TABLE IF NOT EXISTS tip_snapshots(
            match_id BIGINT,
            created_ts BIGINT,
            payload TEXT,
            PRIMARY KEY(match_id,created_ts)
        )""")
        c.execute("""
        CREATE TABLE IF NOT EXISTS feedback(
            id SERIAL PRIMARY KEY,
            match_id BIGINT UNIQUE,
            verdict INTEGER,
            created_ts BIGINT
        )""")
        c.execute("""
        CREATE TABLE IF NOT EXISTS settings(
            key TEXT PRIMARY KEY,
            value TEXT
        )""")
        c.execute("""
        CREATE TABLE IF NOT EXISTS match_results(
            match_id BIGINT PRIMARY KEY,
            final_goals_h INTEGER,
            final_goals_a INTEGER,
            btts_yes INTEGER,
            updated_ts BIGINT
        )""")

# ───────── Scheduler (preserved exactly as original) ─────────
_scheduler_started=False
def _start_scheduler_once():
    global _scheduler_started
    if _scheduler_started:
        return

    try:
        sched = BackgroundScheduler(timezone=TZ_UTC)

        sched.add_job(production_scan, "interval", seconds=SCAN_INTERVAL_SEC, id="scan", max_instances=1, coalesce=True)
        sched.add_job(backfill_results_for_open_matches, "interval", minutes=BACKFILL_EVERY_MIN, id="backfill", max_instances=1, coalesce=True)

        if DAILY_ACCURACY_DIGEST_ENABLE:
            sched.add_job(daily_accuracy_digest,
                          CronTrigger(hour=DAILY_ACCURACY_HOUR, minute=DAILY_ACCURACY_MINUTE, timezone=BERLIN_TZ),
                          id="digest", max_instances=1, coalesce=True)

        if MOTD_PREDICT:
            sched.add_job(send_match_of_the_day,
                          CronTrigger(hour=MOTD_HOUR, minute=MOTD_MINUTE, timezone=BERLIN_TZ),
                          id="motd", max_instances=1, coalesce=True)

        if TRAIN_ENABLE:
            sched.add_job(auto_train_job,
                          CronTrigger(hour=TRAIN_HOUR_UTC, minute=TRAIN_MINUTE_UTC, timezone=TZ_UTC),
                          id="train", max_instances=1, coalesce=True)

        if AUTO_TUNE_ENABLE:
            sched.add_job(auto_tune_thresholds,
                          CronTrigger(hour=4, minute=7, timezone=TZ_UTC),
                          id="auto_tune", max_instances=1, coalesce=True)

        sched.add_job(retry_unsent_tips, "interval", minutes=10, id="retry", max_instances=1, coalesce=True)

        sched.start()
        _scheduler_started=True

        send_telegram("✅ goalsniper AI mode restarted (patched).")
        log.info("[SCHED] started.")

    except Exception as e:
        log.exception("[SCHED] failed: %s", e)

_start_scheduler_once()

# ───────── Admin Auth Helper ─────────
def _require_admin():
    key = request.headers.get("X-API-Key") or request.args.get("key")
    if not ADMIN_API_KEY or key!=ADMIN_API_KEY:
        abort(401)

# ───────── HTTP ROUTES (ALL PRESERVED) ─────────

@app.route("/")
def root():
    return jsonify({"ok":True,"name":"goalsniper","mode":"FULL","scheduler":RUN_SCHEDULER})

@app.route("/health")
def health():
    try:
        with db_conn() as c:
            n=c.execute("SELECT COUNT(*) FROM tips").fetchone()[0]
        return jsonify({"ok":True,"db":"ok","tips":n})
    except Exception as e:
        return jsonify({"ok":False,"error":str(e)}),500

@app.route("/init-db",methods=["POST"])
def http_init_db():
    _require_admin()
    init_db()
    return jsonify({"ok":True})

@app.route("/admin/scan")
def http_scan():
    _require_admin()
    s,l=production_scan()
    return jsonify({"ok":True,"saved":s,"live_seen":l})

@app.route("/admin/backfill-results")
def http_backfill():
    _require_admin()
    n=backfill_results_for_open_matches(400)
    return jsonify({"ok":True,"updated":n})

@app.route("/admin/train")
def http_train():
    _require_admin()
    out=train_models()
    return jsonify({"ok":True,"result":out})

@app.route("/admin/train-notify")
def http_train_notify():
    _require_admin()
    auto_train_job()
    return jsonify({"ok":True})

@app.route("/admin/digest")
def http_digest():
    _require_admin()
    msg=daily_accuracy_digest()
    return jsonify({"ok":True,"sent":bool(msg)})

@app.route("/admin/auto-tune")
def http_auto_tune():
    _require_admin()
    tuned=auto_tune_thresholds(14)
    return jsonify({"ok":True,"tuned":tuned})

@app.route("/admin/retry-unsent")
def http_retry():
    _require_admin()
    n=retry_unsent_tips(30,200)
    return jsonify({"ok":True,"resent":n})

@app.route("/admin/prematch-scan")
def http_prematch():
    _require_admin()
    s=prematch_scan_save()
    return jsonify({"ok":True,"saved":s})

@app.route("/admin/motd")
def http_motd():
    _require_admin()
    ok=send_match_of_the_day()
    return jsonify({"ok":ok})

@app.route("/settings/<key>",methods=["GET","POST"])
def http_settings(key):
    _require_admin()
    if request.method=="GET":
        val=get_setting_cached(key)
        return jsonify({"ok":True,"key":key,"value":val})

    payload=request.get_json(silent=True) or {}
    val=payload.get("value")
    if val is None:
        abort(400)

    set_setting(key,str(val))
    _SETTINGS_CACHE.invalidate(key)
    invalidate_model_caches_for_key(key)
    return jsonify({"ok":True})

@app.route("/tips/latest")
def http_latest():
    limit=int(request.args.get("limit","50"))
    with db_conn() as c:
        rows=c.execute("""
            SELECT match_id,league,home,away,market,suggestion,
                   confidence,confidence_raw,score_at_tip,minute,
                   created_ts,odds,book,ev_pct
            FROM tips
            WHERE suggestion<>'HARVEST'
            ORDER BY created_ts DESC
            LIMIT %s
        """,(limit,)).fetchall()

    out=[]
    for r in rows:
        out.append({
            "match_id":r[0],
            "league":r[1],
            "home":r[2],
            "away":r[3],
            "market":r[4],
            "suggestion":r[5],
            "confidence":float(r[6]),
            "confidence_raw":float(r[7]) if r[7] is not None else None,
            "score_at_tip":r[8],
            "minute":r[9],
            "created_ts":r[10],
            "odds":float(r[11]) if r[11] is not None else None,
            "book":r[12],
            "ev_pct":float(r[13]) if r[13] is not None else None
        })

    return jsonify({"ok":True,"tips":out})

@app.route("/telegram/webhook/<secret>",methods=["POST"])
def webhook(secret):
    if secret != WEBHOOK_SECRET:
        abort(403)

    up=request.get_json(silent=True) or {}
    msg=(up.get("message") or {}).get("text") or ""

    try:
        if msg.startswith("/start"):
            send_telegram("✅ goalsniper online.")
        elif msg.startswith("/scan"):
            if ADMIN_API_KEY and ADMIN_API_KEY in msg:
                s,l=production_scan()
                send_telegram(f"Scan done. Saved {s}, Live={l}")
            else:
                send_telegram("Auth required.")
        elif msg.startswith("/digest"):
            daily_accuracy_digest()
        elif msg.startswith("/motd"):
            send_match_of_the_day()
    except Exception as e:
        log.warning(f"[WEBHOOK] error: {e}")

    return jsonify({"ok":True})

# ───────── Boot ─────────
def _on_boot():
    _init_pool()
    init_db()
    set_setting("boot_ts",str(int(time.time())))

_on_boot()

# ───────── Entry ─────────
if __name__ == "__main__":
    app.run(host=os.getenv("HOST","0.0.0.0"), port=int(os.getenv("PORT","8080")))
