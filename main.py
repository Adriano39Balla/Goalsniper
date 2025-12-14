"""
goalsniper — FULL AI mode (in-play + prematch) with odds + EV gate.

- Pure ML (calibrated) loaded from Postgres settings (train_models.py).
- Markets: OU(2.5,3.5), BTTS (Yes/No), 1X2 (Draw suppressed).
- Adds bookmaker odds filtering + EV check.
- Scheduler: scan, results backfill, nightly train, daily digest, MOTD.

Safe to run on Railway/Render. Requires DATABASE_URL and API keys.
"""

import os, json, time, logging, requests, psycopg2, re, math
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
import collections

# ───────── Env bootstrap ─────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ───────── Enhanced logging configuration ─────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger("goalsniper")
log.setLevel(logging.DEBUG)  # Enable debug logging for intensive monitoring

# Add file handler for detailed logs
try:
    fh = logging.FileHandler('goalsniper.log', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh_formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(name)s - %(funcName)s:%(lineno)d - %(message)s')
    fh.setFormatter(fh_formatter)
    log.addHandler(fh)
except Exception as e:
    log.warning("Could not create file handler: %s", e)

log.info("🚀 Starting goalsniper AI mode with intensive logging...")

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

# ───────── Odds/EV controls ─────────
MIN_ODDS_OU   = float(os.getenv("MIN_ODDS_OU",   "1.30"))
MIN_ODDS_BTTS = float(os.getenv("MIN_ODDS_BTTS", "1.30"))
MIN_ODDS_1X2  = float(os.getenv("MIN_ODDS_1X2",  "1.30"))
MAX_ODDS_ALL  = float(os.getenv("MAX_ODDS_ALL",  "20.0"))
EDGE_MIN_BPS  = int(os.getenv("EDGE_MIN_BPS", "300"))  # 300 = +3.00%
ODDS_BOOKMAKER_ID = os.getenv("ODDS_BOOKMAKER_ID")  # optional API-Football book id
ALLOW_TIPS_WITHOUT_ODDS = os.getenv("ALLOW_TIPS_WITHOUT_ODDS","1") not in ("0","false","False","no","NO")

# ───────── Markets allow-list (draw suppressed) ─────────
ALLOWED_SUGGESTIONS = {"BTTS: Yes", "BTTS: No", "OU 2.5", "OU 3.5", "Home Win", "Away Win"}
def _fmt_line(line: float) -> str: return f"{line}".rstrip("0").rstrip(".")
for _ln in OU_LINES:
    s=_fmt_line(_ln); ALLOWED_SUGGESTIONS.add(f"Over {s} Goals"); ALLOWED_SUGGESTIONS.add(f"Under {s} Goals")

# ───────── External APIs / HTTP session ─────────
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL: 
    log.critical("❌ DATABASE_URL is required")
    raise SystemExit("DATABASE_URL is required")

BASE_URL = "https://v3.football.api-sports.io"
FOOTBALL_API_URL = f"{BASE_URL}/fixtures"
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
HEADERS = {"x-apisports-key": API_KEY, "Accept": "application/json"}
INPLAY_STATUSES = {"1H","HT","2H","ET","BT","P"}

session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1, status_forcelist=[429,500,502,503,504], respect_retry_after_header=True)))

# ───────── Caches & timezones ─────────
class LRUCache:
    def __init__(self, maxsize: int = 1000, ttl: int = 120):
        self.maxsize = maxsize
        self.ttl = ttl
        self.cache = collections.OrderedDict()
    
    def get(self, key):
        if key not in self.cache:
            return None
        timestamp, value = self.cache[key]
        if time.time() - timestamp > self.ttl:
            del self.cache[key]
            return None
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return value
    
    def set(self, key, value):
        if len(self.cache) >= self.maxsize:
            # Remove oldest item
            self.cache.popitem(last=False)
        self.cache[key] = (time.time(), value)
    
    def clear(self):
        self.cache.clear()

STATS_CACHE = LRUCache(maxsize=1000, ttl=90)
EVENTS_CACHE = LRUCache(maxsize=1000, ttl=90)
ODDS_CACHE = LRUCache(maxsize=1000, ttl=120)

SETTINGS_TTL = int(os.getenv("SETTINGS_TTL_SEC","60"))
MODELS_TTL   = int(os.getenv("MODELS_CACHE_TTL_SEC","120"))
TZ_UTC, BERLIN_TZ = ZoneInfo("UTC"), ZoneInfo("Europe/Berlin")

# ───────── TARGET LEAGUES FILTER ─────────
TARGET_LEAGUES = [
    "England Premier League",
    "England Championship",
    "Spain La Liga",
    "Spain La Liga 2",
    "Italy Serie A",
    "Italy Serie B",
    "Germany Bundesliga",
    "Germany 2. Bundesliga",
    "France Ligue 1",
    "France Ligue 2",
    "Belgium Pro League",
    "Portugal Primeira Liga",
    "Portugal Segunda Liga",
    "Brazil Serie A",
    "Brazil Serie B",
    "USA MLS",
    "Netherlands Eredivisie",
    "Argentina Liga Profesional",
    "Poland Ekstraklasa",
    "Japan J League",
    "Sweden Allsvenskan",
    "Croatia HNL",
    "Turkey Super Lig",
    "Mexico Liga MX",
    "Switzerland Super League",
    "Austria Bundesliga",
    "Norway Eliteserien",
    "Colombia Primera A",
    "South Korea K League 1",
    "Scotland Premiership",
    "Scotland Championship",
    "Saudi Pro League",
    "Greece Super League 1",
    "Morocco Botola Pro",
    "Romania Superliga",
    "Algeria Ligue 1",
    "Paraguay Primera Division",
    "Israel Ligat HaAl",
    "Uruguay Primera Division",
    "Costa Rica Primera Division",
    "Egypt Premier League",
    "Chile Primera Division",
    "Slovakia Super Liga",
    "Slovenia PrvaLiga",
    "Bolivia Primera Division",
    "UAE Pro League",
    "Azerbaijan Premier League",
    "South Africa Premier League",
    "Australia A-League",
    "Peru Primera Division",
    "Serbia SuperLiga",
    "Bulgaria First League",
    "Honduras Liga Nacional",
    "Bosnia Premier League",
    "Finland Veikkausliiga",
    "China CSL",
    "Qatar Stars League",
    "Venezuela Primera Division",
    "Canada Premier League",
    "USL Championship",
    "India Super League"
]

def _is_target_league(league_obj: dict) -> bool:
    """Check if league is in the target list"""
    if not league_obj:
        return False
    
    league_name = str(league_obj.get("name", "")).strip()
    country = str(league_obj.get("country", "")).strip()
    
    # Log what we're actually getting from the API
    log.debug(f"🔍 Checking league: country='{country}', name='{league_name}'")
    
    # Try multiple formats
    possible_names = [
        f"{country} - {league_name}",
        f"{league_name}",
        f"{country} {league_name}",
        f"{league_name} - {country}"
    ]
    
    is_target = False
    for possible_name in possible_names:
        if possible_name in TARGET_LEAGUES:
            is_target = True
            log.debug(f"✅ Matched as: {possible_name}")
            break
    
    if is_target:
        log.info(f"✅ League accepted: {country} - {league_name}")
    else:
        log.info(f"❌ League rejected: {country} - {league_name}")
        
    return is_target

# ───────── Optional import: trainer ─────────
try:
    from train_models import train_models
    log.info("✅ Successfully imported train_models")
except Exception as e:
    _IMPORT_ERR = repr(e)
    log.warning("⚠️ train_models not available: %s", _IMPORT_ERR)
    def train_models(*args, **kwargs):  # type: ignore
        log.warning("train_models not available: %s", _IMPORT_ERR)
        return {"ok": False, "reason": f"train_models import failed: {_IMPORT_ERR}"}

# ───────── DB pool & helpers ─────────
POOL: Optional[SimpleConnectionPool] = None
class PooledConn:
    def __init__(self, pool): self.pool=pool; self.conn=None; self.cur=None
    def __enter__(self): self.conn=self.pool.getconn(); self.conn.autocommit=True; self.cur=self.conn.cursor(); return self
    def __exit__(self, a,b,c): 
        try: self.cur and self.cur.close()
        finally: self.conn and self.pool.putconn(self.conn)
    def execute(self, sql: str, params: tuple|list=()):
        self.cur.execute(sql, params or ()); return self.cur

def _init_pool():
    global POOL
    dsn = DATABASE_URL + (("&" if "?" in DATABASE_URL else "?") + "sslmode=require" if "sslmode=" not in DATABASE_URL else "")
    POOL = SimpleConnectionPool(minconn=1, maxconn=int(os.getenv("DB_POOL_MAX","5")), dsn=dsn)
    log.info("✅ Database connection pool initialized")

def db_conn(): 
    if not POOL: _init_pool()
    return PooledConn(POOL)  # type: ignore

# ───────── Settings cache ─────────
class _TTLCache:
    def __init__(self, ttl): self.ttl=ttl; self.data={}
    def get(self, k): 
        v=self.data.get(k); 
        if not v: return None
        ts,val=v
        if time.time()-ts>self.ttl: self.data.pop(k,None); return None
        return val
    def set(self,k,v): self.data[k]=(time.time(),v)
    def invalidate(self,k=None): self.data.clear() if k is None else self.data.pop(k,None)

_SETTINGS_CACHE, _MODELS_CACHE = _TTLCache(SETTINGS_TTL), _TTLCache(MODELS_TTL)

def get_setting(key: str) -> Optional[str]:
    log.debug("📋 Getting setting: %s", key)
    with db_conn() as c:
        r=c.execute("SELECT value FROM settings WHERE key=%s",(key,)).fetchone()
        log.debug("📋 Setting %s value: %s", key, r[0] if r else "None")
        return r[0] if r else None

def set_setting(key: str, value: str) -> None:
    log.info("📝 Setting %s = %s", key, value)
    with db_conn() as c:
        c.execute("INSERT INTO settings(key,value) VALUES(%s,%s) ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value", (key,value))

def get_setting_cached(key: str) -> Optional[str]:
    v=_SETTINGS_CACHE.get(key)
    if v is None: 
        log.debug("🔍 Cache miss for setting: %s", key)
        v=get_setting(key); 
        _SETTINGS_CACHE.set(key,v)
    else:
        log.debug("✅ Cache hit for setting: %s", key)
    return v

def invalidate_model_caches_for_key(key: str):
    if key.lower().startswith(("model","model_latest","model_v2","pre_")):
        log.debug("🔄 Invalidating model cache for key: %s", key)
        _MODELS_CACHE.invalidate()

# ───────── Init DB ─────────
def init_db():
    log.info("🔄 Initializing database schema...")
    with db_conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS tips (
            match_id BIGINT, league_id BIGINT, league TEXT,
            home TEXT, away TEXT, market TEXT, suggestion TEXT,
            confidence DOUBLE PRECISION, confidence_raw DOUBLE PRECISION,
            score_at_tip TEXT, minute INTEGER, created_ts BIGINT,
            odds DOUBLE PRECISION, book TEXT, ev_pct DOUBLE PRECISION,
            sent_ok INTEGER DEFAULT 1,
            PRIMARY KEY (match_id, created_ts))""")
        c.execute("""CREATE TABLE IF NOT EXISTS tip_snapshots (
            match_id BIGINT, created_ts BIGINT, payload TEXT,
            PRIMARY KEY (match_id, created_ts))""")
        c.execute("""CREATE TABLE IF NOT EXISTS feedback (
            id SERIAL PRIMARY KEY, match_id BIGINT UNIQUE, verdict INTEGER, created_ts BIGINT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS match_results (
            match_id BIGINT PRIMARY KEY, final_goals_h INTEGER, final_goals_a INTEGER, btts_yes INTEGER, updated_ts BIGINT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS prematch_snapshots (
            match_id BIGINT PRIMARY KEY,
            created_ts BIGINT,
            payload TEXT)""")
        # Evolutive columns (idempotent)
        try: c.execute("ALTER TABLE tips ADD COLUMN IF NOT EXISTS odds DOUBLE PRECISION")
        except psycopg2.Error as e: log.warning("Could not add odds column: %s", e)
        try: c.execute("ALTER TABLE tips ADD COLUMN IF NOT EXISTS book TEXT")
        except psycopg2.Error as e: log.warning("Could not add book column: %s", e)
        try: c.execute("ALTER TABLE tips ADD COLUMN IF NOT EXISTS ev_pct DOUBLE PRECISION")
        except psycopg2.Error as e: log.warning("Could not add ev_pct column: %s", e)
        try: c.execute("ALTER TABLE tips ADD COLUMN IF NOT EXISTS confidence_raw DOUBLE PRECISION")
        except psycopg2.Error as e: log.warning("Could not add confidence_raw column: %s", e)
        c.execute("CREATE INDEX IF NOT EXISTS idx_tips_created ON tips (created_ts DESC)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_tips_match ON tips (match_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_tips_sent ON tips (sent_ok, created_ts DESC)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_snap_by_match ON tip_snapshots (match_id, created_ts DESC)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_results_updated ON match_results (updated_ts DESC)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_pre_snap_ts ON prematch_snapshots (created_ts DESC)")
    log.info("✅ Database schema initialized")

# ───────── Telegram ─────────
def send_telegram(text: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: 
        log.error("❌ Telegram credentials missing")
        return False
    try:
        log.info("📤 Sending Telegram message...")
        r=session.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                       data={"chat_id":TELEGRAM_CHAT_ID,"text":text,"parse_mode":"HTML","disable_web_page_preview":True}, timeout=10)
        log.info("✅ Telegram sent: status=%s", r.status_code)
        return r.ok
    except Exception as e:
        log.error("❌ Telegram send failed: %s", e)
        return False

# ───────── API helpers ─────────
def _api_get(url: str, params: dict, timeout: int = 15) -> Optional[dict]:
    if not API_KEY: 
        log.error("❌ API_KEY not configured")
        return None
    try:
        log.debug("🌐 API Request: %s, params: %s", url, params)
        r=session.get(url, headers=HEADERS, params=params, timeout=timeout)
        if r.status_code == 429:
            log.warning("⚠️ API rate limit exceeded for %s", url)
            return None
        if r.status_code >= 500:
            log.error("❌ API server error %s for %s", r.status_code, url)
            return None
        if not r.ok:
            log.warning("⚠️ API request failed %s: %s", r.status_code, r.text[:200])
            return None
        log.debug("✅ API Response: %s", r.status_code)
        return r.json()
    except requests.exceptions.Timeout:
        log.error("❌ API timeout for %s", url)
        return None
    except requests.exceptions.ConnectionError:
        log.error("❌ API connection error for %s", url)
        return None
    except Exception as e:
        log.error("❌ API request exception for %s: %s", url, e)
        return None

# ───────── League filter ─────────
_BLOCK_PATTERNS = ["u17","u18","u19","u20","u21","u23","youth","junior","reserve","res.","friendlies","friendly"]
def _blocked_league(league_obj: dict) -> bool:
    name=str((league_obj or {}).get("name","")).lower()
    country=str((league_obj or {}).get("country","")).lower()
    typ=str((league_obj or {}).get("type","")).lower()
    txt=f"{country} {name} {typ}"
    if any(p in txt for p in _BLOCK_PATTERNS): 
        log.debug("🚫 Blocked league (pattern): %s", txt)
        return True
    allow=[x.strip() for x in os.getenv("MOTD_LEAGUE_IDS","").split(",") if x.strip()]  # not used for live
    deny=[x.strip() for x in os.getenv("LEAGUE_DENY_IDS","").split(",") if x.strip()]
    lid=str((league_obj or {}).get("id") or "")
    if lid in deny: 
        log.debug("🚫 Blocked league (ID in deny list): %s", lid)
        return True
    return False

# ───────── Live fetches ─────────
def fetch_match_stats(fid: int) -> list:
    cached = STATS_CACHE.get(fid)
    if cached is not None: 
        log.debug("✅ Stats cache hit for fixture %s", fid)
        return cached
    
    log.debug("📊 Fetching stats for fixture %s", fid)
    js=_api_get(f"{FOOTBALL_API_URL}/statistics", {"fixture": fid}) or {}
    out=js.get("response",[]) if isinstance(js,dict) else []
    STATS_CACHE.set(fid, out)
    log.debug("✅ Stats fetched for fixture %s: %s items", fid, len(out))
    return out

def fetch_match_events(fid: int) -> list:
    cached = EVENTS_CACHE.get(fid)
    if cached is not None: 
        log.debug("✅ Events cache hit for fixture %s", fid)
        return cached
    
    log.debug("📅 Fetching events for fixture %s", fid)
    js=_api_get(f"{FOOTBALL_API_URL}/events", {"fixture": fid}) or {}
    out=js.get("response",[]) if isinstance(js,dict) else []
    EVENTS_CACHE.set(fid, out)
    log.debug("✅ Events fetched for fixture %s: %s items", fid, len(out))
    return out

def fetch_live_matches() -> List[dict]:
    log.info("🔍 Fetching live matches...")
    js=_api_get(FOOTBALL_API_URL, {"live":"all"}) or {}
    matches=[m for m in (js.get("response",[]) if isinstance(js,dict) else []) if not _blocked_league(m.get("league") or {})]
    log.info("📊 Raw live matches: %s", len(matches))
    
    out=[]
    for m in matches:
        st=((m.get("fixture",{}) or {}).get("status",{}) or {})
        elapsed=st.get("elapsed"); short=(st.get("short") or "").upper()
        if elapsed is None or elapsed>120 or short not in INPLAY_STATUSES: 
            continue
        fid=(m.get("fixture",{}) or {}).get("id")
        log.debug("🏃 Processing live match %s at minute %s", fid, elapsed)
        m["statistics"]=fetch_match_stats(fid); m["events"]=fetch_match_events(fid)
        out.append(m)
    
    log.info("✅ Live matches after filtering: %s", len(out))
    return out

# ───────── Prematch helpers ─────────
def _api_last_fixtures(team_id: int, n: int = 5) -> List[dict]:
    log.debug("📋 Fetching last %s fixtures for team %s", n, team_id)
    js=_api_get(f"{BASE_URL}/fixtures", {"team":team_id,"last":n}) or {}
    return js.get("response",[]) if isinstance(js,dict) else []

def _api_h2h(home_id: int, away_id: int, n: int = 5) -> List[dict]:
    log.debug("🤝 Fetching H2H for %s vs %s", home_id, away_id)
    js=_api_get(f"{BASE_URL}/fixtures/headtohead", {"h2h":f"{home_id}-{away_id}","last":n}) or {}
    return js.get("response",[]) if isinstance(js,dict) else []

def _collect_todays_prematch_fixtures() -> List[dict]:
    log.info("📅 Collecting today's prematch fixtures...")
    today_local=datetime.now(ZoneInfo("Europe/Berlin")).date()
    start_local=datetime.combine(today_local, datetime.min.time(), tzinfo=ZoneInfo("Europe/Berlin"))
    end_local=start_local+timedelta(days=1)
    dates_utc={start_local.astimezone(TZ_UTC).date(), (end_local - timedelta(seconds=1)).astimezone(TZ_UTC).date()}
    fixtures=[]
    for d in sorted(dates_utc):
        log.debug("📅 Checking date: %s", d)
        js=_api_get(FOOTBALL_API_URL, {"date": d.strftime("%Y-%m-%d")}) or {}
        for r in js.get("response",[]) if isinstance(js,dict) else []:
            if (((r.get("fixture") or {}).get("status") or {}).get("short") or "").upper() == "NS":
                fixtures.append(r)
    fixtures=[f for f in fixtures if not _blocked_league(f.get("league") or {})]
    log.info("✅ Today's prematch fixtures: %s", len(fixtures))
    return fixtures

# ───────── Feature extraction (live) ─────────
def _num(v) -> float:
    try:
        if isinstance(v,str) and v.endswith("%"): return float(v[:-1])
        return float(v or 0)
    except: return 0.0

def _pos_pct(v) -> float:
    try: return float(str(v).replace("%","").strip() or 0)
    except: return 0.0

def extract_features(m: dict) -> Dict[str,float]:
    log.debug("📈 Extracting features from match data")
    home=m["teams"]["home"]["name"]; away=m["teams"]["away"]["name"]
    gh=m["goals"]["home"] or 0; ga=m["goals"]["away"] or 0
    minute=int(((m.get("fixture") or {}).get("status") or {}).get("elapsed") or 0)
    
    stats={}; 
    for s in (m.get("statistics") or []):
        t=(s.get("team") or {}).get("name"); 
        if t: stats[t]={ (i.get("type") or ""): i.get("value") for i in (s.get("statistics") or []) }
    
    sh=stats.get(home,{}) or {}; sa=stats.get(away,{}) or {}
    xg_h=_num(sh.get("Expected Goals",0)); xg_a=_num(sa.get("Expected Goals",0))
    sot_h=_num(sh.get("Shots on Target",0)); sot_a=_num(sa.get("Shots on Target",0))
    cor_h=_num(sh.get("Corner Kicks",0)); cor_a=_num(sa.get("Corner Kicks",0))
    pos_h=_pos_pct(sh.get("Ball Possession",0)); pos_a=_pos_pct(sa.get("Ball Possession",0))
    red_h=red_a=0
    
    for ev in (m.get("events") or []):
        if (ev.get("type","").lower()=="card"):
            d=(ev.get("detail","") or "").lower()
            if "red" in d or "second yellow" in d:
                t=(ev.get("team") or {}).get("name") or ""
                if t==home: red_h+=1
                elif t==away: red_a+=1
    
    features = {
        "minute":float(minute),
        "goals_h":float(gh),"goals_a":float(ga),"goals_sum":float(gh+ga),"goals_diff":float(gh-ga),
        "xg_h":float(xg_h),"xg_a":float(xg_a),"xg_sum":float(xg_h+xg_a),"xg_diff":float(xg_h-xg_a),
        "sot_h":float(sot_h),"sot_a":float(sot_a),"sot_sum":float(sot_h+sot_a),
        "cor_h":float(cor_h),"cor_a":float(cor_a),"cor_sum":float(cor_h+cor_a),
        "pos_h":float(pos_h),"pos_a":float(pos_a),"pos_diff":float(pos_h-pos_a),
        "red_h":float(red_h),"red_a":float(red_a),"red_sum":float(red_h+red_a)
    }
    
    log.debug("📊 Extracted features: %s", {k: round(v, 2) for k, v in features.items()})
    return features

def validate_features(feat: Dict[str,float]) -> bool:
    """Enhanced feature validation with realistic value checks"""
    log.debug("🔍 Validating features...")
    
    required = ["minute", "goals_h", "goals_a", "xg_h", "xg_a", "sot_h", "sot_a"]
    
    # Check all required fields exist
    if not all(k in feat for k in required):
        log.warning("❌ Missing required features: %s", [k for k in required if k not in feat])
        return False
    
    # Check for reasonable values
    if feat["minute"] < 0 or feat["minute"] > 120:
        log.warning("❌ Invalid minute: %s", feat["minute"])
        return False
    
    if feat["goals_h"] < 0 or feat["goals_a"] < 0:
        log.warning("❌ Invalid goals: %s-%s", feat["goals_h"], feat["goals_a"])
        return False
    
    if feat["xg_h"] < 0 or feat["xg_a"] < 0 or feat["xg_h"] > 10 or feat["xg_a"] > 10:
        log.warning("❌ Invalid xG: %s-%s", feat["xg_h"], feat["xg_a"])
        return False
    
    if feat["sot_h"] < 0 or feat["sot_a"] < 0 or feat["sot_h"] > 50 or feat["sot_a"] > 50:
        log.warning("❌ Invalid SOT: %s-%s", feat["sot_h"], feat["sot_a"])
        return False
    
    # Check possession if available
    if "pos_h" in feat and "pos_a" in feat:
        pos_sum = feat["pos_h"] + feat["pos_a"]
        if abs(pos_sum - 100.0) > 5.0:  # Allow small rounding errors
            log.warning("❌ Invalid possession sum: %s (h=%s, a=%s)", pos_sum, feat["pos_h"], feat["pos_a"])
            return False
    
    log.debug("✅ Features validated successfully")
    return True

def stats_coverage_ok(feat: Dict[str,float], minute: int) -> bool:
    require_stats_minute=int(os.getenv("REQUIRE_STATS_MINUTE","35"))
    require_fields=int(os.getenv("REQUIRE_DATA_FIELDS","2"))
    if minute < require_stats_minute: 
        log.debug("✅ Stats coverage OK (minute %s < %s)", minute, require_stats_minute)
        return True
    fields=[feat.get("xg_sum",0.0), feat.get("sot_sum",0.0), feat.get("cor_sum",0.0),
            max(feat.get("pos_h",0.0), feat.get("pos_a",0.0))]
    nonzero=sum(1 for v in fields if (v or 0)>0)
    result = nonzero >= max(0, require_fields)
    if not result:
        log.warning("⚠️ Insufficient stats coverage: %s/%s fields have data", nonzero, require_fields)
    return result

def _league_name(m: dict) -> Tuple[int,str]:
    lg=(m.get("league") or {}) or {}
    return int(lg.get("id") or 0), f"{lg.get('country','')} - {lg.get('name','')}".strip(" -")

def _teams(m: dict) -> Tuple[str,str]:
    t=(m.get("teams") or {}) or {}
    return (t.get("home",{}).get("name",""), t.get("away",{}).get("name",""))

def _pretty_score(m: dict) -> str:
    gh=(m.get("goals") or {}).get("home") or 0; ga=(m.get("goals") or {}).get("away") or 0
    return f"{gh}-{ga}"

# ───────── Models ─────────
MODEL_KEYS_ORDER=["model_v2:{name}","model_latest:{name}","model:{name}"]
EPS=1e-12

def _sigmoid(x: float) -> float:
    try:
        if x<-50: return 1e-22
        if x>50:  return 1-1e-22
        return 1/(1+math.exp(-x))
    except: return 0.5

def _logit(p: float) -> float:
    p=max(EPS,min(1-EPS,float(p)))
    return math.log(p/(1-p))

def _calibrate(p: float, cal: Dict[str, Any]) -> float:
    """
    Proper calibration using Platt scaling or sigmoid scaling.
    Ensures probabilities are properly calibrated.
    """
    log.debug("🔧 Calibrating probability %s with method: %s", p, cal.get("method", "sigmoid"))
    
    method = (cal or {}).get("method", "sigmoid")
    a = float((cal or {}).get("a", 1.0))
    b = float((cal or {}).get("b", 0.0))
    
    if method.lower() == "platt":
        # Platt scaling: sigmoid(a * logit(p) + b)
        try:
            p_clamped = max(EPS, min(1 - EPS, float(p)))
            logit_val = math.log(p_clamped / (1 - p_clamped))
            result = _sigmoid(a * logit_val + b)
            log.debug("🔧 Platt calibration: %s -> %s (a=%s, b=%s)", p, result, a, b)
            return result
        except Exception as e:
            log.warning("❌ Platt calibration failed: %s", e)
            return p
    elif method.lower() == "sigmoid":
        # Sigmoid scaling: sigmoid(a * logit(p) + b)
        try:
            p_clamped = max(EPS, min(1 - EPS, float(p)))
            logit_val = math.log(p_clamped / (1 - p_clamped))
            result = _sigmoid(a * logit_val + b)
            log.debug("🔧 Sigmoid calibration: %s -> %s (a=%s, b=%s)", p, result, a, b)
            return result
        except Exception as e:
            log.warning("❌ Sigmoid calibration failed: %s", e)
            return p
    else:
        # Direct probability adjustment
        try:
            p_calibrated = a * p + b
            result = max(0.0, min(1.0, p_calibrated))
            log.debug("🔧 Direct calibration: %s -> %s (a=%s, b=%s)", p, result, a, b)
            return result
        except Exception as e:
            log.warning("❌ Direct calibration failed: %s", e)
            return p

def _score_prob(feat: Dict[str, float], mdl: Dict[str, Any]) -> float:
    """Calculate probability with proper calibration and validation"""
    if not mdl or not feat:
        log.warning("❌ Missing model or features")
        return 0.5
    
    try:
        # Calculate raw linear prediction
        weights = mdl.get("weights", {})
        intercept = float(mdl.get("intercept", 0.0))
        raw_pred = intercept
        for k, w in weights.items():
            raw_pred += float(w or 0.0) * float(feat.get(k, 0.0))
        
        # Apply sigmoid to get probability
        p = _sigmoid(raw_pred)
        log.debug("📊 Raw prediction: %s -> Probability: %s", raw_pred, p)
        
        # Apply calibration if available
        cal = mdl.get("calibration") or {}
        if cal:
            p = _calibrate(p, cal)
            log.debug("📊 Calibrated probability: %s", p)
        
        # Ensure probability is valid
        result = max(0.01, min(0.99, float(p)))
        log.debug("✅ Final probability: %s", result)
        return result
    except Exception as e:
        log.error("❌ Error calculating probability: %s", e)
        return 0.5

def _normalize_1x2_probabilities(ph: float, pd: float, pa: float) -> Tuple[float, float, float]:
    """
    Normalize 1X2 probabilities with draw suppression.
    Only normalizes home and away probabilities, ignoring draw.
    """
    log.debug("📊 Normalizing 1X2 probabilities: H=%s, D=%s, A=%s", ph, pd, pa)
    
    try:
        # Suppress draw by not including it in normalization
        s = max(EPS, ph + pa)
        if s > 0:
            ph_norm = ph / s
            pa_norm = pa / s
            # Draw probability is set to very low value but not zero
            pd_norm = min(pd, 0.01)  # Cap draw probability
        else:
            # Fallback if both are zero
            ph_norm, pa_norm, pd_norm = 0.33, 0.33, 0.34
        
        # Ensure they sum to approximately 1
        total = ph_norm + pd_norm + pa_norm
        if total > 0:
            result = (ph_norm / total, pd_norm / total, pa_norm / total)
        else:
            result = (0.33, 0.34, 0.33)
        
        log.debug("✅ Normalized 1X2 probabilities: H=%s, D=%s, A=%s", *result)
        return result
    except Exception as e:
        log.error("❌ Error normalizing 1X2 probabilities: %s", e)
        return (0.33, 0.34, 0.33)

def validate_model(mdl: Dict[str, Any], model_name: str) -> bool:
    """Validate that a model has proper structure and calibration"""
    log.debug("🔍 Validating model: %s", model_name)
    
    if not mdl:
        log.warning("❌ Model %s is empty", model_name)
        return False
    
    # Check required keys
    if "weights" not in mdl:
        log.warning("❌ Model %s missing weights", model_name)
        return False
    
    if "intercept" not in mdl:
        log.warning("❌ Model %s missing intercept", model_name)
        return False
    
    # Check calibration structure if present
    if "calibration" in mdl:
        cal = mdl["calibration"]
        if not isinstance(cal, dict):
            log.warning("❌ Model %s has invalid calibration structure", model_name)
            return False
        
        if "method" not in cal:
            log.warning("❌ Model %s calibration missing method", model_name)
            return False
    
    # Check for extreme weights (warning only, not fatal)
    weights = mdl.get("weights", {})
    extreme_weights = {k: v for k, v in weights.items() if abs(float(v)) > 100}
    if extreme_weights:
        log.warning("⚠️ Model %s has extreme weights: %s", model_name, extreme_weights)
    
    log.debug("✅ Model %s validated successfully", model_name)
    return True

def load_model_from_settings(name: str) -> Optional[Dict[str,Any]]:
    log.debug("📦 Loading model: %s", name)
    
    cached=_MODELS_CACHE.get(name)
    if cached is not None: 
        log.debug("✅ Model cache hit: %s", name)
        return cached
    
    mdl=None
    for pat in MODEL_KEYS_ORDER:
        raw=get_setting_cached(pat.format(name=name))
        if not raw: continue
        
        try:
            tmp=json.loads(raw); 
            tmp.setdefault("intercept",0.0); 
            tmp.setdefault("weights",{})
            cal=tmp.get("calibration") or {}; 
            if isinstance(cal,dict): 
                cal.setdefault("method","sigmoid"); 
                cal.setdefault("a",1.0); 
                cal.setdefault("b",0.0); 
                tmp["calibration"]=cal
            
            # Validate model
            if validate_model(tmp, name):
                mdl=tmp; 
                log.info("✅ Successfully loaded model: %s", name)
                break
            else:
                log.warning("⚠️ Model %s failed validation", name)
                
        except Exception as e:
            log.warning("❌ [MODEL] parse %s failed: %s", name, e)
    
    if mdl is not None: 
        _MODELS_CACHE.set(name, mdl)
    else:
        log.warning("❌ Could not load model: %s", name)
    
    return mdl

def _load_ou_model_for_line(line: float) -> Optional[Dict[str,Any]]:
    name=f"OU_{_fmt_line(line)}"; 
    mdl=load_model_from_settings(name)
    if not mdl and abs(line-2.5)<1e-6:
        mdl = load_model_from_settings("O25")
    return mdl

def _load_wld_models(): 
    return (
        load_model_from_settings("WLD_HOME"), 
        load_model_from_settings("WLD_DRAW"), 
        load_model_from_settings("WLD_AWAY")
    )

# ───────── Odds helpers ─────────
def _ev(prob: float, odds: float) -> float:
    """Return expected value as decimal (e.g. 0.05 = +5%)."""
    if odds <= 1.0 or prob <= 0:
        log.debug("❌ Invalid EV calculation: odds=%s, prob=%s", odds, prob)
        return -1.0  # Invalid
    
    # Convert probability to decimal (0-1 range)
    prob_decimal = max(0.01, min(0.99, float(prob)))
    
    # Calculate expected value: prob * (odds - 1) - (1 - prob)
    # This gives the expected profit per unit staked
    expected_value = (prob_decimal * (odds - 1.0)) - (1.0 - prob_decimal)
    
    # Alternative simpler formula: prob * odds - 1
    ev_simple = (prob_decimal * odds) - 1.0
    
    # Use the larger value (more conservative)
    result = max(expected_value, ev_simple)
    log.debug("💰 EV calculation: prob=%s, odds=%s -> EV=%s", prob_decimal, odds, result)
    return result

def _min_odds_for_market(market: str) -> float:
    """Get minimum odds for a market, handling PRE prefix."""
    # Remove PRE prefix if present
    market_clean = market.replace("PRE ", "")
    
    if market_clean.startswith("Over/Under"):
        result = MIN_ODDS_OU
    elif market_clean == "BTTS":
        result = MIN_ODDS_BTTS
    elif market_clean == "1X2":
        result = MIN_ODDS_1X2
    else:
        result = 1.01  # Default minimum
    
    log.debug("📊 Min odds for market '%s': %s", market_clean, result)
    return result

def _market_name_normalize(s: str) -> str:
    s=(s or "").lower()
    if "both teams" in s or "btts" in s: return "BTTS"
    if "match winner" in s or "winner" in s or "1x2" in s: return "1X2"
    if "over/under" in s or "total" in s or "goals" in s: return "OU"
    return s

def fetch_odds(fid: int) -> dict:
    """
    Returns a dict like:
    {
      "BTTS": {"Yes": {"odds":1.90,"book":"X"}, "No": {...}},
      "1X2":  {"Home": {...}, "Away": {...}},
      "OU_2.5": {"Over": {...}, "Under": {...}},
      "OU_3.5": {...}
    }
    Best-effort parsing of API-Football /odds endpoint; tolerate missing data.
    """
    log.debug("💰 Fetching odds for fixture %s", fid)
    
    cached = ODDS_CACHE.get(fid)
    if cached is not None: 
        log.debug("✅ Odds cache hit for fixture %s", fid)
        return cached
    
    params={"fixture": fid}
    if ODDS_BOOKMAKER_ID: params["bookmaker"] = ODDS_BOOKMAKER_ID
    
    js=_api_get(f"{BASE_URL}/odds", params) or {}
    out={}
    
    try:
        for r in js.get("response",[]) if isinstance(js,dict) else []:
            bookmakers = r.get("bookmakers", [])
            if not bookmakers: continue
            
            for bk in bookmakers:
                book_name = bk.get("name") or "Unknown"
                bets = bk.get("bets") or []
                
                for bet in bets:
                    bet_name = str(bet.get("name", "")).lower()
                    values = bet.get("values") or []
                    
                    # BTTS Market
                    if "both teams to score" in bet_name or "btts" in bet_name:
                        btts_dict = {}
                        for v in values:
                            value_name = str(v.get("value", "")).lower()
                            odd_value = float(v.get("odd") or 0)
                            
                            if "yes" in value_name or "home" in value_name:
                                btts_dict["Yes"] = {"odds": odd_value, "book": book_name}
                            elif "no" in value_name:
                                btts_dict["No"] = {"odds": odd_value, "book": book_name}
                        
                        if btts_dict:
                            out["BTTS"] = btts_dict
                    
                    # 1X2 Market
                    elif "match winner" in bet_name or "1x2" in bet_name or "winner" in bet_name:
                        win_dict = {}
                        for v in values:
                            value_name = str(v.get("value", "")).lower()
                            odd_value = float(v.get("odd") or 0)
                            
                            if "home" in value_name or "1" in value_name:
                                win_dict["Home"] = {"odds": odd_value, "book": book_name}
                            elif "away" in value_name or "2" in value_name:
                                win_dict["Away"] = {"odds": odd_value, "book": book_name}
                            elif "draw" in value_name or "x" in value_name:
                                # Store but we won't use for draw-suppressed markets
                                win_dict["Draw"] = {"odds": odd_value, "book": book_name}
                        
                        if win_dict:
                            out["1X2"] = win_dict
                    
                    # Over/Under Markets
                    elif "over/under" in bet_name or "total goals" in bet_name:
                        for v in values:
                            value_name = str(v.get("value", "")).lower()
                            odd_value = float(v.get("odd") or 0)
                            
                            # Parse line from value name like "Over 2.5"
                            match = re.search(r'(over|under)\s*(\d+\.?\d*)', value_name)
                            if match:
                                side = match.group(1).capitalize()
                                line = float(match.group(2))
                                market_key = f"OU_{_fmt_line(line)}"
                                
                                if market_key not in out:
                                    out[market_key] = {}
                                
                                out[market_key][side] = {"odds": odd_value, "book": book_name}
        
        ODDS_CACHE.set(fid, out)
        log.debug("✅ Odds fetched for fixture %s: %s", fid, {k: len(v) for k, v in out.items()})
    
    except Exception as e:
        log.error("❌ Error parsing odds for fixture %s: %s", fid, e)
        out = {}
    
    return out

def _price_gate(market_text: str, suggestion: str, fid: int, probability: float) -> Tuple[bool, Optional[float], Optional[str], Optional[float]]:
    """
    Return (pass, odds, book, ev_pct). If odds missing:
      - pass if ALLOW_TIPS_WITHOUT_ODDS else block.
    """
    log.debug("🚪 Price gate for: %s - %s (fixture %s, prob=%s)", market_text, suggestion, fid, probability)
    
    odds_map = fetch_odds(fid) if API_KEY else {}
    odds = None
    book = None
    ev_pct = None
    
    # Clean market text for matching
    market_clean = market_text.replace("PRE ", "")
    
    # Map suggestion to odds market format
    if "BTTS" in market_clean:
        d = odds_map.get("BTTS", {})
        tgt = "Yes" if "Yes" in suggestion else "No"
        if tgt in d:
            odds = float(d[tgt].get("odds", 0))
            book = d[tgt].get("book", "Unknown")
            log.debug("💰 Found BTTS odds: %s @ %s", odds, book)
    
    elif "1X2" in market_clean:
        d = odds_map.get("1X2", {})
        if "Home" in suggestion or "home" in suggestion.lower():
            tgt = "Home"
        elif "Away" in suggestion or "away" in suggestion.lower():
            tgt = "Away"
        else:
            tgt = None
        
        if tgt and tgt in d:
            odds = float(d[tgt].get("odds", 0))
            book = d[tgt].get("book", "Unknown")
            log.debug("💰 Found 1X2 odds: %s @ %s", odds, book)
    
    elif "Over/Under" in market_clean or "OU" in market_text:
        # Parse the line from suggestion
        try:
            # Extract number from suggestion like "Over 2.5 Goals"
            match = re.search(r'(\d+\.?\d*)', suggestion)
            if match:
                line = float(match.group(1))
                market_key = f"OU_{_fmt_line(line)}"
                d = odds_map.get(market_key, {})
                
                tgt = "Over" if "Over" in suggestion else "Under"
                if tgt in d:
                    odds = float(d[tgt].get("odds", 0))
                    book = d[tgt].get("book", "Unknown")
                    log.debug("💰 Found OU odds: %s @ %s for line %s", odds, book, line)
        except Exception as e:
            log.debug("❌ Error parsing line from suggestion %s: %s", suggestion, e)
    
    if odds is None:
        log.debug("⚠️ No odds found for %s - %s", market_text, suggestion)
        return (ALLOW_TIPS_WITHOUT_ODDS, None, None, None)
    
    # Validate odds are sane
    if odds <= 1.0 or odds > 100:
        log.warning("❌ Invalid odds %s for fixture %s", odds, fid)
        return (False, odds, book, None)
    
    # Price range gates
    min_odds = _min_odds_for_market(market_clean)
    if not (min_odds <= odds <= MAX_ODDS_ALL):
        log.debug("🚫 Odds %s outside range [%s, %s] for market %s", 
                  odds, min_odds, MAX_ODDS_ALL, market_clean)
        return (False, odds, book, None)
    
    # Calculate EV
    edge = _ev(probability, odds)
    ev_pct = round(edge * 100.0, 1)
    
    # Check edge condition (convert to basis points)
    edge_bps = int(round(edge * 10000))
    min_edge_bps = EDGE_MIN_BPS
    
    if edge_bps < min_edge_bps:
        log.debug("🚫 Edge too low: %s bps < %s bps for odds %s", 
                  edge_bps, min_edge_bps, odds)
        return (False, odds, book, ev_pct)
    
    log.debug("✅ Price gate passed: odds=%s, book=%s, EV=%s%", odds, book, ev_pct)
    return (True, odds, book, ev_pct)

# ───────── Snapshots ─────────
def save_snapshot_from_match(m: dict, feat: Dict[str,float]) -> None:
    fx=m.get("fixture",{}) or {}; lg=m.get("league",{}) or {}
    fid=int(fx.get("id")); league_id=int(lg.get("id") or 0)
    league=f"{lg.get('country','')} - {lg.get('name','')}".strip(" -")
    home=(m.get("teams") or {}).get("home",{}).get("name","")
    away=(m.get("teams") or {}).get("away",{}).get("name","")
    gh=(m.get("goals") or {}).get("home") or 0; ga=(m.get("goals") or {}).get("away") or 0
    minute=int(feat.get("minute",0))
    snapshot={"minute":minute,"gh":gh,"ga":ga,"league_id":league_id,"market":"HARVEST","suggestion":"HARVEST","confidence":0,
              "stat":{"xg_h":feat.get("xg_h",0),"xg_a":feat.get("xg_a",0),"sot_h":feat.get("sot_h",0),"sot_a":feat.get("sot_a",0),
                      "cor_h":feat.get("cor_h",0),"cor_a":feat.get("cor_a",0),"pos_h":feat.get("pos_h",0),"pos_a":feat.get("pos_a",0),
                      "red_h":feat.get("red_h",0),"red_a":feat.get("red_a",0)}}
    now=int(time.time())
    with db_conn() as c:
        try:
            c.execute("INSERT INTO tip_snapshots(match_id, created_ts, payload) VALUES (%s,%s,%s) "
                      "ON CONFLICT (match_id, created_ts) DO UPDATE SET payload=EXCLUDED.payload",
                      (fid, now, json.dumps(snapshot)[:200000]))
            c.execute("INSERT INTO tips(match_id,league_id,league,home,away,market,suggestion,confidence,confidence_raw,score_at_tip,minute,created_ts,sent_ok) "
                      "VALUES (%s,%s,%s,%s,%s,'HARVEST','HARVEST',0.0,0.0,%s,%s,%s,1)",
                      (fid, league_id, league, home, away, f"{gh}-{ga}", minute, now))
            log.debug("✅ Saved snapshot for match %s", fid)
        except psycopg2.Error as e:
            log.error("❌ Error saving snapshot for match %s: %s", fid, e)

def save_prematch_snapshot(fid: int, feat: Dict[str,float]) -> None:
    """Save prematch features for training"""
    now=int(time.time())
    snapshot = {"feat": feat}
    with db_conn() as c:
        try:
            c.execute("INSERT INTO prematch_snapshots (match_id, created_ts, payload) "
                      "VALUES (%s, %s, %s) "
                      "ON CONFLICT (match_id) DO UPDATE SET payload=EXCLUDED.payload, created_ts=EXCLUDED.created_ts",
                      (fid, now, json.dumps(snapshot)))
            log.debug("✅ Saved prematch snapshot for match %s", fid)
        except psycopg2.Error as e:
            log.error("❌ Error saving prematch snapshot for match %s: %s", fid, e)

# ───────── Outcomes/backfill/digest ─────────
def _parse_ou_line_from_suggestion(s: str) -> Optional[float]:
    try:
        for tok in (s or "").split():
            try: return float(tok)
            except: pass
    except: pass
    return None

def _tip_outcome_for_result(suggestion: str, res: Dict[str,Any]) -> Optional[int]:
    gh=int(res.get("final_goals_h") or 0); ga=int(res.get("final_goals_a") or 0)
    total=gh+ga; btts=int(res.get("btts_yes") or 0); s=(suggestion or "").strip()
    if s.startswith("Over") or s.startswith("Under"):
        line=_parse_ou_line_from_suggestion(s); 
        if line is None: return None
        if s.startswith("Over"):
            if total>line: return 1
            if abs(total-line)<1e-9: return None
            return 0
        else:
            if total<line: return 1
            if abs(total-line)<1e-9: return None
            return 0
    if s=="BTTS: Yes": return 1 if btts==1 else 0
    if s=="BTTS: No":  return 1 if btts==0 else 0
    if s=="Home Win":  return 1 if gh>ga else 0
    if s=="Away Win":  return 1 if ga>gh else 0
    return None

def _fixture_by_id(mid: int) -> Optional[dict]:
    js=_api_get(FOOTBALL_API_URL, {"id": mid}) or {}
    arr=js.get("response") or [] if isinstance(js,dict) else []
    return arr[0] if arr else None

def _is_final(short: str) -> bool: return (short or "").upper() in {"FT","AET","PEN"}

def backfill_results_for_open_matches(max_rows: int = 200) -> int:
    log.info("🔄 Backfilling results for open matches...")
    now_ts=int(time.time()); cutoff=now_ts - BACKFILL_DAYS*24*3600; updated=0
    with db_conn() as c:
        rows=c.execute("""
            WITH last AS (SELECT match_id, MAX(created_ts) last_ts FROM tips WHERE created_ts >= %s GROUP BY match_id)
            SELECT l.match_id FROM last l LEFT JOIN match_results r ON r.match_id=l.match_id
            WHERE r.match_id IS NULL ORDER BY l.last_ts DESC LIMIT %s
        """,(cutoff, max_rows)).fetchall()
    
    log.debug("📊 Found %s matches to check for results", len(rows))
    
    for (mid,) in rows:
        fx=_fixture_by_id(int(mid))
        if not fx: 
            log.debug("❌ Could not fetch fixture %s", mid)
            continue
        
        st=(((fx.get("fixture") or {}).get("status") or {}).get("short") or "")
        if not _is_final(st): 
            log.debug("⚠️ Fixture %s not final (status: %s)", mid, st)
            continue
        
        g=fx.get("goals") or {}; gh=int(g.get("home") or 0); ga=int(g.get("away") or 0)
        btts=1 if (gh>0 and ga>0) else 0
        
        with db_conn() as c2:
            try:
                c2.execute("INSERT INTO match_results(match_id, final_goals_h, final_goals_a, btts_yes, updated_ts) "
                           "VALUES(%s,%s,%s,%s,%s) ON CONFLICT(match_id) DO UPDATE SET final_goals_h=EXCLUDED.final_goals_h, "
                           "final_goals_a=EXCLUDED.final_goals_a, btts_yes=EXCLUDED.btts_yes, updated_ts=EXCLUDED.updated_ts",
                           (int(mid), gh, ga, btts, int(time.time())))
                log.debug("✅ Backfilled result for match %s: %s-%s (BTTS: %s)", mid, gh, ga, btts)
                updated+=1
            except psycopg2.Error as e:
                log.error("❌ Error backfilling results for match %s: %s", mid, e)
                continue
    
    if updated: 
        log.info("✅ [RESULTS] backfilled %d matches", updated)
    else:
        log.info("ℹ️ [RESULTS] no matches to backfill")
    
    return updated

def daily_accuracy_digest() -> Optional[str]:
    if not DAILY_ACCURACY_DIGEST_ENABLE: 
        log.info("ℹ️ Daily accuracy digest disabled")
        return None
    
    log.info("📊 Generating daily accuracy digest...")
    now_local=datetime.now(BERLIN_TZ)
    y0=(now_local - timedelta(days=1)).replace(hour=0,minute=0,second=0,microsecond=0); y1=y0+timedelta(days=1)
    
    # Ensure we have latest results
    backfill_results_for_open_matches(400)
    
    with db_conn() as c:
        rows=c.execute("""
            SELECT t.match_id, t.market, t.suggestion, t.confidence, t.confidence_raw, t.created_ts,
                   r.final_goals_h, r.final_goals_a, r.btts_yes
            FROM tips t LEFT JOIN match_results r ON r.match_id=t.match_id
            WHERE t.created_ts >= %s AND t.created_ts < %s AND t.suggestion<>'HARVEST' AND t.sent_ok=1
        """,(int(y0.timestamp()), int(y1.timestamp()))).fetchall()
    
    log.debug("📊 Found %s tips for digest period", len(rows))
    
    total=graded=wins=0; by={}
    for (mid, mkt, sugg, conf, conf_raw, cts, gh, ga, btts) in rows:
        res={"final_goals_h":gh,"final_goals_a":ga,"btts_yes":btts}
        out=_tip_outcome_for_result(sugg,res)
        if out is None: continue
        total+=1; graded+=1; wins+=1 if out==1 else 0
        d=by.setdefault(mkt or "?",{"graded":0,"wins":0}); d["graded"]+=1; d["wins"]+=1 if out==1 else 0
    
    if graded==0:
        msg="📊 Daily Digest\nNo graded tips for yesterday."
        log.info("ℹ️ No graded tips for digest")
    else:
        acc=100.0*wins/max(1,graded)
        lines=[f"📊 <b>Daily Digest</b> (yesterday, Berlin time)",
               f"Tips sent: {total}  •  Graded: {graded}  •  Wins: {wins}  •  Accuracy: {acc:.1f}%"]
        for mk,st in sorted(by.items()):
            if st["graded"]==0: continue
            a=100.0*st["wins"]/st["graded"]; lines.append(f"• {escape(mk)} — {st['wins']}/{st['graded']} ({a:.1f}%)")
        msg="\n".join(lines)
        log.info("✅ Daily digest: %s tips, %s graded, %s wins, %.1f%% accuracy", total, graded, wins, acc)
    
    send_telegram(msg); 
    return msg

# ───────── Thresholds & formatting ─────────
def _get_market_threshold_key(m: str) -> str: return f"conf_threshold:{m}"
def _get_market_threshold(m: str) -> float:
    try:
        v=get_setting_cached(_get_market_threshold_key(m)); 
        result = float(v) if v is not None else float(CONF_THRESHOLD)
        log.debug("📊 Threshold for %s: %s", m, result)
        return result
    except: 
        log.warning("❌ Error getting threshold for %s, using default", m)
        return float(CONF_THRESHOLD)
def _get_market_threshold_pre(m: str) -> float: return _get_market_threshold(f"PRE {m}")

def _format_tip_message(home, away, league, minute, score, suggestion, prob_pct, feat, odds=None, book=None, ev_pct=None):
    stat=""
    if any([feat.get("xg_h",0),feat.get("xg_a",0),feat.get("sot_h",0),feat.get("sot_a",0),feat.get("cor_h",0),feat.get("cor_a",0),
            feat.get("pos_h",0),feat.get("pos_a",0),feat.get("red_h",0),feat.get("red_a",0)]):
        stat=(f"\n📊 xG {feat.get('xg_h',0):.2f}-{feat.get('xg_a',0):.2f}"
              f" • SOT {int(feat.get('sot_h',0))}-{int(feat.get('sot_a',0))}"
              f" • CK {int(feat.get('cor_h',0))}-{int(feat.get('cor_a',0))}")
        if feat.get("pos_h",0) or feat.get("pos_a",0): stat += f" • POS {int(feat.get('pos_h',0))}%–{int(feat.get('pos_a',0))}%"
        if feat.get("red_h",0) or feat.get("red_a",0): stat += f" • RED {int(feat.get('red_h',0))}-{int(feat.get('red_a',0))}"
    money = ""
    if odds:
        if ev_pct is not None:
            money = f"\n💰 <b>Odds:</b> {odds:.2f} @ {book or 'Book'}  •  <b>EV:</b> {ev_pct:+.1f}%"
        else:
            money = f"\n💰 <b>Odds:</b> {odds:.2f} @ {book or 'Book'}"
    return ("⚽️ <b>New Tip!</b>\n"
            f"<b>Match:</b> {escape(home)} vs {escape(away)}\n"
            f"🕒 <b>Minute:</b> {minute}'  |  <b>Score:</b> {escape(score)}\n"
            f"<b>Tip:</b> {escape(suggestion)}\n"
            f"📈 <b>Confidence:</b> {prob_pct:.1f}%{money}\n"
            f"🏆 <b>League:</b> {escape(league)}{stat}")

# ───────── Candidate sanity check ─────────
def _candidate_is_sane(sug: str, feat: Dict[str, float]) -> bool:
    """Enhanced sanity check for suggestions"""
    gh = int(feat.get("goals_h", 0))
    ga = int(feat.get("goals_a", 0))
    total = gh + ga
    minute = int(feat.get("minute", 0))
    
    log.debug("🧠 Checking sanity for: %s (score: %s-%s, minute: %s)", sug, gh, ga, minute)
    
    # Over/Under suggestions
    if sug.startswith("Over") or sug.startswith("Under"):
        match = re.search(r'(\d+\.?\d*)', sug)
        if match:
            line = float(match.group(1))
            
            if sug.startswith("Over"):
                # Don't suggest Over if already above line (with small buffer)
                if total > line - 0.25:
                    log.debug("🚫 Over suggestion rejected: total %s > line %s", total, line)
                    return False
                
                # Don't suggest Over late in game if unlikely to reach
                if minute > 70 and total < line - 1.5:
                    log.debug("🚫 Over suggestion rejected: minute %s, total %s, line %s", minute, total, line)
                    return False
            
            elif sug.startswith("Under"):
                # Don't suggest Under if already at or above line
                if total >= line - 0.25:
                    log.debug("🚫 Under suggestion rejected: total %s >= line %s", total, line)
                    return False
                
                # Don't suggest Under early if game is high-scoring
                if minute < 30 and total > line - 0.5:
                    log.debug("🚫 Under suggestion rejected: minute %s, total %s, line %s", minute, total, line)
                    return False
    
    # BTTS suggestions
    elif "BTTS" in sug:
        if "Yes" in sug:
            # Don't suggest BTTS Yes if already both scored
            if gh > 0 and ga > 0:
                log.debug("🚫 BTTS Yes rejected: both already scored")
                return False
            
            # Don't suggest BTTS Yes late if one team hasn't scored
            if minute > 70 and (gh == 0 or ga == 0):
                log.debug("🚫 BTTS Yes rejected: minute %s, one team hasn't scored", minute)
                return False
        
        elif "No" in sug:
            # Don't suggest BTTS No if both already scored
            if gh > 0 and ga > 0:
                log.debug("🚫 BTTS No rejected: both already scored")
                return False
            
            # Don't suggest BTTS No early if no goals
            if minute < 30 and gh == 0 and ga == 0:
                log.debug("🚫 BTTS No rejected: minute %s, no goals yet", minute)
                return False
    
    # 1X2 suggestions
    elif "Win" in sug:
        # Don't suggest Home Win if already losing by many
        if "Home" in sug and ga - gh >= 2:
            log.debug("🚫 Home Win rejected: losing by %s goals", ga - gh)
            return False
        
        # Don't suggest Away Win if already losing by many
        if "Away" in sug and gh - ga >= 2:
            log.debug("🚫 Away Win rejected: losing by %s goals", gh - ga)
            return False
        
        # Don't suggest win late if unlikely
        if minute > 80:
            goal_diff = gh - ga if "Home" in sug else ga - gh
            if goal_diff < 0:  # Currently losing
                log.debug("🚫 Win suggestion rejected: minute %s, currently losing", minute)
                return False
    
    log.debug("✅ Suggestion passed sanity check")
    return True

# ───────── Scan (in-play) ─────────
def production_scan() -> Tuple[int,int]:
    """Fixed version with proper probability normalization and EV calculation"""
    log.info("🔍 Starting production scan...")
    
    matches = fetch_live_matches()
    live_seen = len(matches)
    
    if live_seen == 0:
        log.info("ℹ️ [PROD] No live matches found")
        return 0, 0
    
    log.info("📊 [PROD] Processing %s live matches", live_seen)
    
    saved = 0
    now_ts = int(time.time())
    
    with db_conn() as c:
        for m in matches:
            try:
                fid = int((m.get("fixture", {}) or {}).get("id") or 0)
                if not fid:
                    log.warning("⚠️ Match has no fixture ID")
                    continue
                
                log.debug("🏃 Processing match %s", fid)
                
                # Use advisory lock
                lock_key = fid % (2 ** 63)
                got_lock = c.execute("SELECT pg_try_advisory_lock(%s)", (lock_key,)).fetchone()[0]
                if not got_lock:
                    log.debug("⚠️ Could not acquire lock for match %s, skipping", fid)
                    continue
                
                try:
                    # Duplicate check
                    if DUP_COOLDOWN_MIN > 0:
                        cutoff = now_ts - DUP_COOLDOWN_MIN * 60
                        if c.execute("SELECT 1 FROM tips WHERE match_id=%s AND created_ts>=%s LIMIT 1",
                                    (fid, cutoff)).fetchone():
                            log.debug("⚠️ Match %s has recent tip, skipping", fid)
                            continue
                    
                    feat = extract_features(m)
                    minute = int(feat.get("minute", 0))
                    
                    if not validate_features(feat):
                        log.warning("❌ Invalid features for match %s", fid)
                        continue
                    
                    if not stats_coverage_ok(feat, minute):
                        log.warning("⚠️ Insufficient stats coverage for match %s at minute %s", fid, minute)
                        continue
                    
                    if minute < TIP_MIN_MINUTE:
                        log.debug("⚠️ Match %s minute %s < minimum %s", fid, minute, TIP_MIN_MINUTE)
                        continue
                    
                    # Save snapshot for training if enabled
                    if HARVEST_MODE and minute >= TRAIN_MIN_MINUTE and minute % 3 == 0:
                        try:
                            save_snapshot_from_match(m, feat)
                        except Exception as e:
                            log.error("❌ Error saving snapshot: %s", e)
                    
                    league_id, league = _league_name(m)
                    home, away = _teams(m)
                    score = _pretty_score(m)
                    
                    log.debug("🏆 Match: %s vs %s (%s), score: %s, minute: %s", home, away, league, score, minute)
                    
                    candidates: List[Tuple[str, str, float]] = []
                    
                    # OU Markets
                    for line in OU_LINES:
                        mdl = _load_ou_model_for_line(line)
                        if not mdl:
                            log.debug("⚠️ No OU model for line %s", line)
                            continue
                        
                        p_over = _score_prob(feat, mdl)
                        p_under = 1.0 - p_over
                        
                        mk = f"Over/Under {_fmt_line(line)}"
                        thr = _get_market_threshold(mk)
                        
                        # Check Over
                        if p_over * 100.0 >= thr and _candidate_is_sane(f"Over {_fmt_line(line)} Goals", feat):
                            candidates.append((mk, f"Over {_fmt_line(line)} Goals", p_over))
                            log.debug("✅ OU Over candidate: %s (prob: %s%%)", f"Over {_fmt_line(line)} Goals", p_over*100)
                        
                        # Check Under
                        if p_under * 100.0 >= thr and _candidate_is_sane(f"Under {_fmt_line(line)} Goals", feat):
                            candidates.append((mk, f"Under {_fmt_line(line)} Goals", p_under))
                            log.debug("✅ OU Under candidate: %s (prob: %s%%)", f"Under {_fmt_line(line)} Goals", p_under*100)
                    
                    # BTTS Market
                    mdl_btts = load_model_from_settings("BTTS_YES")
                    if mdl_btts:
                        p_yes = _score_prob(feat, mdl_btts)
                        p_no = 1.0 - p_yes
                        
                        thr = _get_market_threshold("BTTS")
                        
                        if p_yes * 100.0 >= thr and _candidate_is_sane("BTTS: Yes", feat):
                            candidates.append(("BTTS", "BTTS: Yes", p_yes))
                            log.debug("✅ BTTS Yes candidate: %s%%)", p_yes*100)
                        
                        if p_no * 100.0 >= thr and _candidate_is_sane("BTTS: No", feat):
                            candidates.append(("BTTS", "BTTS: No", p_no))
                            log.debug("✅ BTTS No candidate: %s%%)", p_no*100)
                    else:
                        log.debug("⚠️ No BTTS model available")
                    
                    # 1X2 Market (draw suppressed)
                    mh, md, ma = _load_wld_models()
                    if mh and md and ma:
                        ph = _score_prob(feat, mh)
                        pd = _score_prob(feat, md)
                        pa = _score_prob(feat, ma)
                        
                        # Normalize with draw suppression
                        ph_norm, pd_norm, pa_norm = _normalize_1x2_probabilities(ph, pd, pa)
                        
                        thr = _get_market_threshold("1X2")
                        
                        if ph_norm * 100.0 >= thr:
                            candidates.append(("1X2", "Home Win", ph_norm))
                            log.debug("✅ Home Win candidate: %s%%)", ph_norm*100)
                        
                        if pa_norm * 100.0 >= thr:
                            candidates.append(("1X2", "Away Win", pa_norm))
                            log.debug("✅ Away Win candidate: %s%%)", pa_norm*100)
                    else:
                        log.debug("⚠️ Incomplete 1X2 models: H=%s, D=%s, A=%s", mh is not None, md is not None, ma is not None)
                    
                    # Sort by probability and process candidates
                    candidates.sort(key=lambda x: x[2], reverse=True)
                    log.debug("📊 Found %s candidates for match %s", len(candidates), fid)
                    
                    per_match = 0
                    base_now = int(time.time())
                    
                    for idx, (market_txt, suggestion, prob) in enumerate(candidates):
                        if suggestion not in ALLOWED_SUGGESTIONS:
                            log.debug("🚫 Suggestion not allowed: %s", suggestion)
                            continue
                        
                        if per_match >= max(1, PREDICTIONS_PER_MATCH):
                            log.debug("⚠️ Max predictions per match reached (%s)", PREDICTIONS_PER_MATCH)
                            break
                        
                        # Validate probability
                        if prob <= 0.01 or prob >= 0.99:
                            log.debug("🚫 Invalid probability: %s", prob)
                            continue
                        
                        # Odds/EV gate with probability
                        pass_odds, odds, book, ev_pct = _price_gate(market_txt, suggestion, fid, prob)
                        
                        if not pass_odds:
                            log.debug("🚫 Price gate failed for: %s", suggestion)
                            continue
                        
                        # If EV is too low (negative or very small positive)
                        if ev_pct is not None and ev_pct < -5.0:  # -5% EV threshold
                            log.debug("🚫 EV too low: %s%", ev_pct)
                            continue
                        
                        created_ts = base_now + idx
                        prob_pct = round(prob * 100.0, 1)
                        
                        try:
                            c.execute(
                                "INSERT INTO tips(match_id, league_id, league, home, away, market, suggestion, "
                                "confidence, confidence_raw, score_at_tip, minute, created_ts, odds, book, ev_pct, sent_ok) "
                                "VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 0)",
                                (fid, league_id, league, home, away, market_txt, suggestion,
                                 float(prob_pct), float(prob), score, minute, created_ts,
                                 float(odds) if odds else None,
                                 book,
                                 float(ev_pct) if ev_pct else None)
                            )
                            log.info("💾 Saved tip for match %s: %s @ %s%%", fid, suggestion, prob_pct)
                        except psycopg2.errors.UniqueViolation:
                            log.warning("⚠️ Duplicate tip for match %s at %s", fid, created_ts)
                            continue
                        except psycopg2.Error as e:
                            log.error("❌ Error inserting tip: %s", e)
                            continue
                        
                        # Send notification
                        sent = _send_tip(home, away, league, minute, score, suggestion,
                                        float(prob_pct), feat, odds, book, ev_pct)
                        if sent:
                            c.execute("UPDATE tips SET sent_ok=1 WHERE match_id=%s AND created_ts=%s",
                                     (fid, created_ts))
                            log.info("📤 Telegram sent for tip: %s", suggestion)
                        else:
                            log.error("❌ Failed to send Telegram for tip: %s", suggestion)
                        
                        saved += 1
                        per_match += 1
                        
                        if MAX_TIPS_PER_SCAN and saved >= MAX_TIPS_PER_SCAN:
                            log.info("⚠️ Max tips per scan reached (%s)", MAX_TIPS_PER_SCAN)
                            break
                    
                    if MAX_TIPS_PER_SCAN and saved >= MAX_TIPS_PER_SCAN:
                        log.info("⚠️ Stopping scan due to max tips limit")
                        break
                
                finally:
                    c.execute("SELECT pg_advisory_unlock(%s)", (lock_key,))
                    log.debug("🔓 Released lock for match %s", fid)
            
            except Exception as e:
                log.exception("❌ [PROD] failure for match: %s", e)
                continue
    
    log.info("✅ [PROD] saved=%d live_seen=%d", saved, live_seen)
    return saved, live_seen

def _send_tip(home,away,league,minute,score,suggestion,prob_pct,feat,odds=None,book=None,ev_pct=None)->bool:
    return send_telegram(_format_tip_message(home,away,league,minute,score,suggestion,prob_pct,feat,odds,book,ev_pct))

# ───────── Prematch ─────────
def extract_prematch_features(fx: dict) -> Dict[str,float]:
    """Extract prematch features matching train_models.py PRE_FEATURES"""
    log.debug("📈 Extracting prematch features")
    
    teams=fx.get("teams") or {}; 
    th=(teams.get("home") or {}).get("id"); ta=(teams.get("away") or {}).get("id")
    if not th or not ta: 
        log.warning("❌ Missing team IDs for prematch features")
        return {}
    
    def _rate(games):
        ov25=ov35=btts=played=0
        for g in games:
            st=(((g.get("fixture") or {}).get("status") or {}).get("short") or "").upper()
            if st not in {"FT","AET","PEN"}: continue
            gh=int((g.get("goals") or {}).get("home") or 0); ga=int((g.get("goals") or {}).get("away") or 0)
            played+=1; 
            if gh+ga>2: ov25+=1
            if gh+ga>3: ov35+=1
            if gh>0 and ga>0: btts+=1
        if played==0: return 0,0,0
        return ov25/played, ov35/played, btts/played
    
    last_h=_api_last_fixtures(th,5); last_a=_api_last_fixtures(ta,5); h2h=_api_h2h(th,ta,5)
    ov25_h,ov35_h,btts_h=_rate(last_h); ov25_a,ov35_a,btts_a=_rate(last_a); ov25_h2h,ov35_h2h,btts_h2h=_rate(h2h)
    
    # Additional features needed for training
    pm_gf_h = pm_ga_h = pm_win_h = 0
    pm_gf_a = pm_ga_a = pm_win_a = 0
    pm_rest_diff = 0
    
    # Calculate goals for/against and win rates for home team
    if last_h:
        gf_h = ga_h = win_h = 0
        valid_games = 0
        for g in last_h:
            st=(((g.get("fixture") or {}).get("status") or {}).get("short") or "").upper()
            if st not in {"FT","AET","PEN"}: continue
            gh=int((g.get("goals") or {}).get("home") or 0); ga=int((g.get("goals") or {}).get("away") or 0)
            gf_h += gh; ga_h += ga
            if (g.get("teams") or {}).get("home", {}).get("winner") == True: win_h += 1
            valid_games += 1
        
        if valid_games > 0:
            pm_gf_h = gf_h / valid_games
            pm_ga_h = ga_h / valid_games
            pm_win_h = win_h / valid_games
    
    # Calculate goals for/against and win rates for away team
    if last_a:
        gf_a = ga_a = win_a = 0
        valid_games = 0
        for g in last_a:
            st=(((g.get("fixture") or {}).get("status") or {}).get("short") or "").upper()
            if st not in {"FT","AET","PEN"}: continue
            gh=int((g.get("goals") or {}).get("home") or 0); ga=int((g.get("goals") or {}).get("away") or 0)
            # For away team, they are away in these fixtures
            gf_a += ga; ga_a += gh
            if (g.get("teams") or {}).get("away", {}).get("winner") == True: win_a += 1
            valid_games += 1
        
        if valid_games > 0:
            pm_gf_a = gf_a / valid_games
            pm_ga_a = ga_a / valid_games
            pm_win_a = win_a / valid_games
    
    features = {
        # Main features matching train_models.py
        "pm_gf_h": pm_gf_h, "pm_ga_h": pm_ga_h, "pm_win_h": pm_win_h,
        "pm_gf_a": pm_gf_a, "pm_ga_a": pm_ga_a, "pm_win_a": pm_win_a,
        "pm_ov25_h": ov25_h, "pm_ov35_h": ov35_h, "pm_btts_h": btts_h,
        "pm_ov25_a": ov25_a, "pm_ov35_a": ov35_a, "pm_btts_a": btts_a,
        "pm_ov25_h2h": ov25_h2h, "pm_ov35_h2h": ov35_h2h, "pm_btts_h2h": btts_h2h,
        "pm_rest_diff": pm_rest_diff,
        # Live features (zeroed for compatibility)
        "minute": 0.0, "goals_h": 0.0, "goals_a": 0.0, "goals_sum": 0.0, "goals_diff": 0.0,
        "xg_h": 0.0, "xg_a": 0.0, "xg_sum": 0.0, "xg_diff": 0.0,
        "sot_h": 0.0, "sot_a": 0.0, "sot_sum": 0.0,
        "cor_h": 0.0, "cor_a": 0.0, "cor_sum": 0.0,
        "pos_h": 0.0, "pos_a": 0.0, "pos_diff": 0.0,
        "red_h": 0.0, "red_a": 0.0, "red_sum": 0.0
    }
    
    log.debug("📊 Extracted prematch features: %s", {k: round(v, 2) for k, v in features.items() if v != 0})
    return features

def _kickoff_berlin(utc_iso: str|None) -> str:
    try:
        if not utc_iso: return "TBD"
        dt=datetime.fromisoformat(utc_iso.replace("Z","+00:00"))
        return dt.astimezone(BERLIN_TZ).strftime("%H:%M")
    except: return "TBD"

def _format_motd_message(home, away, league, kickoff_txt, suggestion, prob_pct, odds=None, book=None, ev_pct=None):
    money = ""
    if odds:
        if ev_pct is not None:
            money = f"\n💰 <b>Odds:</b> {odds:.2f} @ {book or 'Book'}  •  <b>EV:</b> {ev_pct:+.1f}%"
        else:
            money = f"\n💰 <b>Odds:</b> {odds:.2f} @ {book or 'Book'}"
    return (
        "🏅 <b>Match of the Day</b>\n"
        f"<b>Match:</b> {escape(home)} vs {escape(away)}\n"
        f"🏆 <b>League:</b> {escape(league)}\n"
        f"⏰ <b>Kickoff (Berlin):</b> {kickoff_txt}\n"
        f"<b>Tip:</b> {escape(suggestion)}\n"
        f"📈 <b>Confidence:</b> {prob_pct:.1f}%{money}"
    )

def prematch_scan_save() -> int:
    log.info("🔍 Starting prematch scan...")
    fixtures=_collect_todays_prematch_fixtures(); 
    if not fixtures: 
        log.info("ℹ️ No prematch fixtures found")
        return 0
    
    saved=0
    log.info("📊 Processing %s prematch fixtures", len(fixtures))
    
    for fx in fixtures:
        fixture=fx.get("fixture") or {}; lg=fx.get("league") or {}; teams=fx.get("teams") or {}
        home=(teams.get("home") or {}).get("name",""); away=(teams.get("away") or {}).get("name","")
        league_id=int((lg.get("id") or 0)); league=f"{lg.get('country','')} - {lg.get('name','')}".strip(" -"); fid=int((fixture.get("id") or 0))
        
        if not fid: 
            log.warning("⚠️ Fixture has no ID")
            continue
        
        log.debug("🏃 Processing prematch fixture %s: %s vs %s", fid, home, away)
        
        feat=extract_prematch_features(fx); 
        if not feat: 
            log.warning("⚠️ Could not extract features for fixture %s", fid)
            continue
        
        # Save prematch snapshot for training
        try:
            save_prematch_snapshot(fid, feat)
        except Exception as e:
            log.error("❌ Error saving prematch snapshot for %s: %s", fid, e)
        
        candidates: List[Tuple[str,str,float]]=[]
        
        # PRE OU via PRE_OU_* models
        for line in OU_LINES:
            mdl=load_model_from_settings(f"PRE_OU_{_fmt_line(line)}")
            if not mdl: 
                log.debug("⚠️ No PRE OU model for line %s", line)
                continue
            
            p=_score_prob(feat, mdl); 
            mk=f"Over/Under {_fmt_line(line)}"; 
            thr=_get_market_threshold_pre(mk)
            
            if p*100.0>=thr:   
                candidates.append((f"PRE {mk}", f"Over {_fmt_line(line)} Goals", p))
                log.debug("✅ PRE OU Over candidate: %s (prob: %s%%)", f"Over {_fmt_line(line)} Goals", p*100)
            
            q=1.0-p
            if q*100.0>=thr:   
                candidates.append((f"PRE {mk}", f"Under {_fmt_line(line)} Goals", q))
                log.debug("✅ PRE OU Under candidate: %s (prob: %s%%)", f"Under {_fmt_line(line)} Goals", q*100)
        
        # PRE BTTS
        mdl=load_model_from_settings("PRE_BTTS_YES")
        if mdl:
            p=_score_prob(feat, mdl); thr=_get_market_threshold_pre("BTTS")
            if p*100.0>=thr: 
                candidates.append(("PRE BTTS","BTTS: Yes",p))
                log.debug("✅ PRE BTTS Yes candidate: %s%%)", p*100)
            q=1.0-p
            if q*100.0>=thr: 
                candidates.append(("PRE BTTS","BTTS: No",q))
                log.debug("✅ PRE BTTS No candidate: %s%%)", q*100)
        else:
            log.debug("⚠️ No PRE BTTS model available")
        
        # PRE 1X2 (draw suppressed)
        mh,ma=load_model_from_settings("PRE_WLD_HOME"), load_model_from_settings("PRE_WLD_AWAY")
        if mh and ma:
            ph=_score_prob(feat,mh); pa=_score_prob(feat,ma)
            s=max(EPS, ph+pa)  # Normalize only home and away
            if s > 0:
                ph, pa = ph/s, pa/s
            thr=_get_market_threshold_pre("1X2")
            if ph*100.0>=thr: 
                candidates.append(("PRE 1X2","Home Win",ph))
                log.debug("✅ PRE Home Win candidate: %s%%)", ph*100)
            if pa*100.0>=thr: 
                candidates.append(("PRE 1X2","Away Win",pa))
                log.debug("✅ PRE Away Win candidate: %s%%)", pa*100)
        else:
            log.debug("⚠️ Incomplete PRE 1X2 models: H=%s, A=%s", mh is not None, ma is not None)
        
        if not candidates: 
            log.debug("ℹ️ No candidates for fixture %s", fid)
            continue
        
        candidates.sort(key=lambda x:x[2], reverse=True)
        log.debug("📊 Found %s PRE candidates for fixture %s", len(candidates), fid)
        
        base_now=int(time.time()); per_match=0
        
        for idx,(mk,sug,prob) in enumerate(candidates):
            if sug not in ALLOWED_SUGGESTIONS:
                log.debug("🚫 Suggestion not allowed: %s", sug)
                continue
            
            if per_match>=max(1,PREDICTIONS_PER_MATCH):
                log.debug("⚠️ Max predictions per match reached (%s)", PREDICTIONS_PER_MATCH)
                break
            
            # Validate probability
            if prob <= 0.01 or prob >= 0.99:
                log.debug("🚫 Invalid probability: %s", prob)
                continue
                
            # Odds/EV gate
            pass_odds, odds, book, ev_pct = _price_gate(mk.replace("PRE ",""), sug, fid, prob)
            if not pass_odds:
                log.debug("🚫 Price gate failed for: %s", sug)
                continue
            
            # Calculate EV
            if odds is not None:
                edge=_ev(prob, odds); ev_pct=round(edge*100.0,1)
                edge_bps = int(round(edge * 10000))
                if edge_bps < EDGE_MIN_BPS: 
                    log.debug("🚫 EV too low: %s bps < %s bps", edge_bps, EDGE_MIN_BPS)
                    continue
            
            created_ts=base_now+idx; raw=float(prob); pct=round(raw*100.0,1)
            
            with db_conn() as c2:
                try:
                    c2.execute("INSERT INTO tips(match_id,league_id,league,home,away,market,suggestion,confidence,confidence_raw,score_at_tip,minute,created_ts,odds,book,ev_pct,sent_ok) "
                               "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,'0-0',0,%s,%s,%s,%s,0)",
                               (fid,league_id,league,home,away,mk,sug,float(pct),raw,created_ts,
                                (float(odds) if odds is not None else None), (book or None), (float(ev_pct) if ev_pct is not None else None)))
                    log.info("💾 Saved PRE tip for match %s: %s @ %s%%", fid, sug, pct)
                    saved+=1; per_match+=1
                except psycopg2.errors.UniqueViolation:
                    log.warning("⚠️ Duplicate prematch tip for match %s at %s", fid, created_ts)
                    continue
                except psycopg2.Error as e:
                    log.error("❌ Error inserting prematch tip for %s: %s", fid, e)
                    continue
    
    log.info("✅ [PREMATCH] saved=%d tips", saved)
    return saved

# Optional min EV for MOTD (basis points, e.g. 300 = +3.00%). 0 disables EV gate.
MOTD_MIN_EV_BPS = int(os.getenv("MOTD_MIN_EV_BPS", "0"))

def send_match_of_the_day() -> bool:
    """Pick the single best prematch tip for today (PRE_* models). Sends to Telegram."""
    log.info("🏅 Starting Match of the Day selection...")
    
    fixtures = _collect_todays_prematch_fixtures()
    if not fixtures:
        log.warning("⚠️ No eligible fixtures for MOTD")
        return send_telegram("🏅 Match of the Day: no eligible fixtures today.")

    # Optional league allow-list just for MOTD
    if MOTD_LEAGUE_IDS:
        fixtures = [
            f for f in fixtures
            if int(((f.get("league") or {}).get("id") or 0)) in MOTD_LEAGUE_IDS
        ]
        if not fixtures:
            log.warning("⚠️ No fixtures in configured MOTD leagues")
            return send_telegram("🏅 Match of the Day: no fixtures in configured leagues.")

    log.info("📊 Evaluating %s fixtures for MOTD", len(fixtures))
    best = None  # (prob_pct, suggestion, home, away, league, kickoff_txt, odds, book, ev_pct)

    for fx in fixtures:
        fixture = fx.get("fixture") or {}
        lg      = fx.get("league") or {}
        teams   = fx.get("teams") or {}
        fid     = int((fixture.get("id") or 0))

        home = (teams.get("home") or {}).get("name","")
        away = (teams.get("away") or {}).get("name","")
        league = f"{lg.get('country','')} - {lg.get('name','')}".strip(" -")
        kickoff_txt = _kickoff_berlin((fixture.get("date") or ""))

        log.debug("🏃 Evaluating MOTD candidate: %s vs %s (%s)", home, away, league)

        feat = extract_prematch_features(fx)
        if not feat:
            log.debug("⚠️ Could not extract features for MOTD candidate")
            continue

        # Collect PRE candidates (same thresholds as prematch_scan_save)
        candidates: List[Tuple[str,str,float]] = []

        for line in OU_LINES:
            mdl = load_model_from_settings(f"PRE_OU_{_fmt_line(line)}")
            if not mdl: continue
            p = _score_prob(feat, mdl)
            mk = f"Over/Under {_fmt_line(line)}"
            thr = _get_market_threshold_pre(mk)
            if p*100.0 >= thr:   candidates.append((mk, f"Over {_fmt_line(line)} Goals", p))
            q = 1.0 - p
            if q*100.0 >= thr:   candidates.append((mk, f"Under {_fmt_line(line)} Goals", q))

        mdl = load_model_from_settings("PRE_BTTS_YES")
        if mdl:
            p = _score_prob(feat, mdl); thr = _get_market_threshold_pre("BTTS")
            if p*100.0 >= thr: candidates.append(("BTTS","BTTS: Yes", p))
            q = 1.0 - p
            if q*100.0 >= thr: candidates.append(("BTTS","BTTS: No",  q))

        mh = load_model_from_settings("PRE_WLD_HOME")
        ma = load_model_from_settings("PRE_WLD_AWAY")
        if mh and ma:
            ph = _score_prob(feat, mh); pa = _score_prob(feat, ma)
            s = max(EPS, ph+pa); ph, pa = ph/s, pa/s  # Normalize only home and away
            thr = _get_market_threshold_pre("1X2")
            if ph*100.0 >= thr: candidates.append(("1X2","Home Win", ph))
            if pa*100.0 >= thr: candidates.append(("1X2","Away Win", pa))

        if not candidates:
            log.debug("⚠️ No candidates for MOTD fixture %s", fid)
            continue

        # Take the single best for this fixture (by probability) then apply odds/EV gate
        candidates.sort(key=lambda x: x[2], reverse=True)
        mk, sug, prob = candidates[0]
        prob_pct = prob * 100.0
        
        if prob_pct < MOTD_CONF_MIN:
            log.debug("🚫 MOTD candidate below confidence threshold: %s < %s", prob_pct, MOTD_CONF_MIN)
            continue

        # Odds/EV (reuse in-play price gate; market text must be without "PRE ")
        pass_odds, odds, book, _ = _price_gate(mk, sug, fid, prob)
        if not pass_odds:
            log.debug("🚫 MOTD candidate failed price gate")
            continue

        ev_pct = None
        if odds is not None:
            edge = _ev(prob, odds)            # decimal (e.g. 0.05)
            ev_bps = int(round(edge * 10000)) # basis points
            ev_pct = round(edge * 100.0, 1)
            if MOTD_MIN_EV_BPS > 0 and ev_bps < MOTD_MIN_EV_BPS:
                log.debug("🚫 MOTD candidate EV too low: %s bps < %s bps", ev_bps, MOTD_MIN_EV_BPS)
                continue

        item = (prob_pct, sug, home, away, league, kickoff_txt, odds, book, ev_pct)
        if best is None or prob_pct > best[0]:
            best = item
            log.debug("✅ New MOTD best: %s @ %s%% (EV: %s%%)", sug, prob_pct, ev_pct)

    if not best:
        log.warning("⚠️ No MOTD pick met thresholds")
        return send_telegram("🏅 Match of the Day: no prematch pick met thresholds.")
    
    prob_pct, sug, home, away, league, kickoff_txt, odds, book, ev_pct = best
    log.info("🏅 Selected MOTD: %s vs %s - %s @ %s%%", home, away, sug, prob_pct)
    
    return send_telegram(_format_motd_message(home, away, league, kickoff_txt, sug, prob_pct, odds, book, ev_pct))

# ───────── Auto-train / tune / retry ─────────
def auto_train_job():
    log.info("🤖 Starting auto-training job...")
    
    if not TRAIN_ENABLE: 
        log.warning("⚠️ Training disabled via TRAIN_ENABLE")
        send_telegram("🤖 Training skipped: TRAIN_ENABLE=0")
        return
    
    send_telegram("🤖 Training started.")
    try:
        res=train_models() or {}; ok=bool(res.get("ok"))
        if not ok:
            reason=res.get("reason") or res.get("error") or "unknown"
            log.error("❌ Training failed: %s", reason)
            send_telegram(f"⚠️ Training finished: <b>SKIPPED</b>\nReason: {escape(str(reason))}")
            return
        
        trained=[k for k,v in (res.get("trained") or {}).items() if v]
        thr=(res.get("thresholds") or {}); mets=(res.get("metrics") or {})
        lines=["🤖 <b>Model training OK</b>"]
        if trained: lines.append("• Trained: " + ", ".join(sorted(trained)))
        if thr: lines.append("• Thresholds: " + "  |  ".join([f"{escape(k)}: {float(v):.1f}%" for k,v in thr.items()]))
        
        log.info("✅ Training completed: %s models trained", len(trained))
        send_telegram("\n".join(lines))
        
    except Exception as e:
        log.exception("❌ [TRAIN] job failed: %s", e)
        send_telegram(f"❌ Training <b>FAILED</b>\n{escape(str(e))}")

def _pick_threshold(y_true,y_prob,target_precision,min_preds,default_pct):
    import numpy as np
    y=np.asarray(y_true,dtype=int); p=np.asarray(y_prob,dtype=float)
    best=default_pct/100.0
    for t in np.arange(MIN_THRESH,MAX_THRESH+1e-9,1.0)/100.0:
        pred=(p>=t).astype(int); n=int(pred.sum())
        if n<min_preds: continue
        tp=int(((pred==1)&(y==1)).sum()); prec=tp/max(1,n)
        if prec>=target_precision: best=float(t); break
    return best*100.0

def auto_tune_thresholds(days: int = 14) -> Dict[str,float]:
    log.info("🎛️ Starting auto-tune thresholds...")
    
    if not AUTO_TUNE_ENABLE: 
        log.info("ℹ️ Auto-tune disabled")
        return {}
    
    cutoff=int(time.time())-days*24*3600
    with db_conn() as c:
        rows=c.execute("""
            SELECT t.market, t.suggestion, COALESCE(t.confidence_raw, t.confidence/100.0) prob,
                   r.final_goals_h, r.final_goals_a, r.btts_yes
            FROM tips t JOIN match_results r ON r.match_id=t.match_id
            WHERE t.created_ts >= %s AND t.suggestion<>'HARVEST' AND t.sent_ok=1
        """,(cutoff,)).fetchall()
    
    log.debug("📊 Found %s historical tips for auto-tune", len(rows))
    
    by={}
    for (mk,sugg,prob,gh,ga,btts) in rows:
        out=_tip_outcome_for_result(sugg, {"final_goals_h":gh,"final_goals_a":ga,"btts_yes":btts})
        if out is None: continue
        by.setdefault(mk, []).append((float(prob), int(out)))
    
    tuned={}
    for mk,arr in by.items():
        if len(arr)<THRESH_MIN_PREDICTIONS: 
            log.debug("⚠️ Insufficient data for market %s: %s < %s", mk, len(arr), THRESH_MIN_PREDICTIONS)
            continue
        
        probs=[p for (p,_) in arr]; wins=[y for (_,y) in arr]
        pct=_pick_threshold(wins, probs, TARGET_PRECISION, THRESH_MIN_PREDICTIONS, CONF_THRESHOLD)
        set_setting(f"conf_threshold:{mk}", f"{pct:.2f}"); 
        _SETTINGS_CACHE.invalidate(f"conf_threshold:{mk}"); 
        tuned[mk]=pct
        log.info("🎛️ Updated threshold for %s: %s%%", mk, pct)
    
    if tuned: 
        send_telegram("🔧 Auto-tune updated thresholds:\n" + "\n".join([f"• {k}: {v:.1f}%" for k,v in tuned.items()]))
        log.info("✅ Auto-tune updated %s thresholds", len(tuned))
    else: 
        send_telegram("🔧 Auto-tune: no updates (insufficient data).")
        log.info("ℹ️ Auto-tune: no updates (insufficient data)")
    
    return tuned

def retry_unsent_tips(minutes: int = 30, limit: int = 200) -> int:
    log.info("🔄 Retrying unsent tips...")
    
    cutoff = int(time.time()) - minutes*60
    retried = 0
    
    with db_conn() as c:
        rows = c.execute(
            "SELECT match_id,league,home,away,market,suggestion,confidence,confidence_raw,score_at_tip,minute,created_ts,odds,book,ev_pct "
            "FROM tips WHERE sent_ok=0 AND created_ts >= %s ORDER BY created_ts ASC LIMIT %s",
            (cutoff, limit)
        ).fetchall()
    
    log.debug("📊 Found %s unsent tips to retry", len(rows))
    
    for (mid, league, home, away, market, sugg, conf, conf_raw, score, minute, cts, odds, book, ev_pct) in rows:
        ok = send_telegram(_format_tip_message(home, away, league, int(minute), score, sugg, float(conf), {}, odds, book, ev_pct))
        if ok:
            with db_conn() as c2:
                c2.execute("UPDATE tips SET sent_ok=1 WHERE match_id=%s AND created_ts=%s", (mid, cts))
            retried += 1
            log.debug("✅ Retried tip for match %s: %s", mid, sugg)
    
    if retried:
        log.info("✅ [RETRY] resent %d tips", retried)
    else:
        log.info("ℹ️ [RETRY] no tips to retry")
    
    return retried

# ───────── Calibration monitoring ─────────
def monitor_calibration(days: int = 30) -> Dict[str, float]:
    """Monitor model calibration by comparing predicted vs actual outcomes"""
    log.info("📊 Monitoring calibration for last %s days...", days)
    
    cutoff = int(time.time()) - days * 24 * 3600
    results = {}
    
    with db_conn() as c:
        # Get all tips with results
        rows = c.execute("""
            SELECT t.market, t.confidence_raw, r.final_goals_h, r.final_goals_a, r.btts_yes
            FROM tips t 
            JOIN match_results r ON r.match_id = t.match_id
            WHERE t.created_ts >= %s 
              AND t.suggestion != 'HARVEST'
              AND t.confidence_raw IS NOT NULL
        """, (cutoff,)).fetchall()
    
    log.debug("📊 Found %s tips with results for calibration monitoring", len(rows))
    
    # Group by market and confidence bins
    by_market = {}
    for market, conf_raw, gh, ga, btts in rows:
        if market not in by_market:
            by_market[market] = {"predictions": [], "outcomes": []}
        
        # Determine actual outcome
        outcome = None
        if "Over/Under" in market:
            # Parse line from market
            match = re.search(r'(\d+\.?\d*)', market)
            if match:
                line = float(match.group(1))
                total = gh + ga
                if "Over" in market:
                    outcome = 1 if total > line else 0
                elif "Under" in market:
                    outcome = 1 if total < line else 0
        
        elif market == "BTTS":
            outcome = 1 if btts == 1 else 0
        
        elif market == "1X2":
            if "Home" in market or "home" in market.lower():
                outcome = 1 if gh > ga else 0
            elif "Away" in market or "away" in market.lower():
                outcome = 1 if ga > gh else 0
        
        if outcome is not None:
            by_market[market]["predictions"].append(float(conf_raw))
            by_market[market]["outcomes"].append(int(outcome))
    
    # Calculate calibration metrics
    for market, data in by_market.items():
        if len(data["predictions"]) < 10:
            log.debug("⚠️ Insufficient data for market %s: %s predictions", market, len(data["predictions"]))
            continue
        
        # Group predictions into bins
        bins = [0.1 * i for i in range(11)]  # 0.0, 0.1, ..., 1.0
        binned = {b: {"sum_pred": 0, "sum_act": 0, "count": 0} for b in bins[:-1]}
        
        for pred, act in zip(data["predictions"], data["outcomes"]):
            for i in range(len(bins) - 1):
                if bins[i] <= pred < bins[i + 1]:
                    binned[bins[i]]["sum_pred"] += pred
                    binned[bins[i]]["sum_act"] += act
                    binned[bins[i]]["count"] += 1
                    break
        
        # Calculate calibration error
        calibration_error = 0
        total_weight = 0
        
        for bin_start, stats in binned.items():
            if stats["count"] > 0:
                avg_pred = stats["sum_pred"] / stats["count"]
                avg_act = stats["sum_act"] / stats["count"]
                error = abs(avg_pred - avg_act)
                calibration_error += error * stats["count"]
                total_weight += stats["count"]
                log.debug("📊 Calibration bin [%s-%s]: pred=%s, act=%s, error=%s", 
                         bin_start, bin_start+0.1, avg_pred, avg_act, error)
        
        if total_weight > 0:
            calibration_error /= total_weight
            results[market] = calibration_error
            log.info("📊 Calibration error for %s: %.4f", market, calibration_error)
    
    return results

def calibration_check_job():
    """Periodic calibration monitoring"""
    log.info("🔧 Starting calibration check job...")
    
    try:
        calibration_errors = monitor_calibration(30)
        
        if calibration_errors:
            lines = ["📊 <b>Calibration Report</b> (30 days)"]
            for market, error in sorted(calibration_errors.items()):
                lines.append(f"• {escape(market)}: {error:.3f} avg error")
            
            # Alert if any market has high calibration error
            high_error = any(err > 0.1 for err in calibration_errors.values())
            if high_error:
                lines.append("\n⚠️ <b>High calibration errors detected!</b>")
                lines.append("Consider retraining models with calibration.")
                log.warning("⚠️ High calibration errors detected: %s", 
                          {k: v for k, v in calibration_errors.items() if v > 0.1})
            
            send_telegram("\n".join(lines))
            log.info("✅ Calibration check completed")
        else:
            log.info("ℹ️ No calibration data available")
    
    except Exception as e:
        log.error("❌ Calibration check failed: %s", e)

# ───────── Scheduler ─────────
def _run_with_pg_lock(lock_key: int, fn, *a, **k):
    log.debug("🔒 Attempting to acquire lock %s", lock_key)
    try:
        with db_conn() as c:
            got=c.execute("SELECT pg_try_advisory_lock(%s)",(lock_key,)).fetchone()[0]
            if not got: 
                log.info("⚠️ [LOCK %s] busy; skipped.", lock_key)
                return None
            log.debug("✅ Acquired lock %s", lock_key)
            try: 
                result = fn(*a,**k)
                log.debug("✅ Completed function with lock %s", lock_key)
                return result
            finally: 
                c.execute("SELECT pg_advisory_unlock(%s)",(lock_key,))
                log.debug("🔓 Released lock %s", lock_key)
    except Exception as e:
        log.exception("❌ [LOCK %s] failed: %s", lock_key, e)
        return None

_scheduler_started=False
def _start_scheduler_once():
    global _scheduler_started
    if _scheduler_started or not RUN_SCHEDULER: 
        log.info("ℹ️ Scheduler already started or disabled")
        return
    
    try:
        sched=BackgroundScheduler(timezone=TZ_UTC)
        
        # Production scan (in-play)
        sched.add_job(
            lambda:_run_with_pg_lock(1001, production_scan),
            "interval",
            seconds=SCAN_INTERVAL_SEC,
            id="scan",
            max_instances=1,
            coalesce=True
        )
        log.info("⏰ Scheduled production scan every %s seconds", SCAN_INTERVAL_SEC)
        
        # Results backfill
        sched.add_job(
            lambda:_run_with_pg_lock(1002, backfill_results_for_open_matches, 400),
            "interval",
            minutes=BACKFILL_EVERY_MIN,
            id="backfill",
            max_instances=1,
            coalesce=True
        )
        log.info("⏰ Scheduled backfill every %s minutes", BACKFILL_EVERY_MIN)
        
        # Daily accuracy digest
        if DAILY_ACCURACY_DIGEST_ENABLE:
            sched.add_job(
                lambda:_run_with_pg_lock(1003, daily_accuracy_digest),
                CronTrigger(hour=DAILY_ACCURACY_HOUR, minute=DAILY_ACCURACY_MINUTE, timezone=BERLIN_TZ),
                id="digest",
                max_instances=1,
                coalesce=True
            )
            log.info("⏰ Scheduled daily digest at %s:%s Berlin time", DAILY_ACCURACY_HOUR, DAILY_ACCURACY_MINUTE)
        
        # Match of the Day
        if MOTD_PREDICT:
            sched.add_job(
                lambda:_run_with_pg_lock(1004, send_match_of_the_day),
                CronTrigger(hour=int(os.getenv("MOTD_HOUR","19")), minute=int(os.getenv("MOTD_MINUTE","15")), timezone=BERLIN_TZ),
                id="motd",
                max_instances=1,
                coalesce=True
            )
            log.info("⏰ Scheduled MOTD at %s:%s Berlin time", MOTD_HOUR, MOTD_MINUTE)
        
        # Training
        if TRAIN_ENABLE:
            sched.add_job(
                lambda:_run_with_pg_lock(1005, auto_train_job),
                CronTrigger(hour=TRAIN_HOUR_UTC, minute=TRAIN_MINUTE_UTC, timezone=TZ_UTC),
                id="train",
                max_instances=1,
                coalesce=True
            )
            log.info("⏰ Scheduled training at %s:%s UTC", TRAIN_HOUR_UTC, TRAIN_MINUTE_UTC)
        
        # Auto-tune thresholds
        if AUTO_TUNE_ENABLE:
            sched.add_job(
                lambda:_run_with_pg_lock(1006, auto_tune_thresholds, 14),
                CronTrigger(hour=4, minute=7, timezone=TZ_UTC),
                id="auto_tune",
                max_instances=1,
                coalesce=True
            )
            log.info("⏰ Scheduled auto-tune at 04:07 UTC")
        
        # Retry unsent tips
        sched.add_job(
            lambda:_run_with_pg_lock(1007, retry_unsent_tips, 30, 200),
            "interval",
            minutes=10,
            id="retry",
            max_instances=1,
            coalesce=True
        )
        log.info("⏰ Scheduled retry every 10 minutes")
        
        # Calibration check (new)
        sched.add_job(
            lambda:_run_with_pg_lock(1008, calibration_check_job),
            CronTrigger(hour=3, minute=0, timezone=TZ_UTC),
            id="calibration_check",
            max_instances=1,
            coalesce=True
        )
        log.info("⏰ Scheduled calibration check at 03:00 UTC")
        
        sched.start()
        _scheduler_started=True
        
        startup_msg = "🚀 goalsniper AI mode (in-play + prematch) started with intensive logging."
        log.info(startup_msg)
        send_telegram(startup_msg)
        
        log.info("✅ [SCHED] started (scan=%ss)", SCAN_INTERVAL_SEC)
        
    except Exception as e:
        log.exception("❌ [SCHED] failed to start: %s", e)

_start_scheduler_once()

# ───────── Admin / auth ─────────
def _require_admin():
    key=request.headers.get("X-API-Key") or request.args.get("key") or ((request.json or {}).get("key") if request.is_json else None)
    if not ADMIN_API_KEY or key != ADMIN_API_KEY: 
        log.warning("❌ Unauthorized admin access attempt")
        abort(401)
    log.debug("✅ Admin authentication successful")

# ───────── HTTP endpoints ─────────
@app.route("/")
def root(): 
    log.info("🌐 Root endpoint accessed")
    return jsonify({"ok": True, "name": "goalsniper", "mode": "FULL_AI", "scheduler": RUN_SCHEDULER, "intensive_logging": True})

@app.route("/health")
def health():
    log.info("🌐 Health check endpoint accessed")
    try:
        with db_conn() as c:
            n=c.execute("SELECT COUNT(*) FROM tips").fetchone()[0]
        return jsonify({"ok": True, "db": "ok", "tips_count": int(n), "timestamp": int(time.time())})
    except Exception as e:
        log.error("❌ Health check failed: %s", e)
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/init-db", methods=["POST"])
def http_init_db(): 
    _require_admin(); 
    log.info("🔄 Manual DB init requested")
    init_db(); 
    return jsonify({"ok": True, "message": "Database initialized"})

@app.route("/admin/scan", methods=["POST","GET"])
def http_scan(): 
    _require_admin(); 
    log.info("🔍 Manual scan requested")
    s,l=production_scan(); 
    return jsonify({"ok": True, "saved": s, "live_seen": l})

@app.route("/admin/backfill-results", methods=["POST","GET"])
def http_backfill(): 
    _require_admin(); 
    log.info("🔄 Manual backfill requested")
    n=backfill_results_for_open_matches(400); 
    return jsonify({"ok": True, "updated": n})

@app.route("/admin/train", methods=["POST","GET"])
def http_train():
    _require_admin()
    log.info("🤖 Manual training requested")
    if not TRAIN_ENABLE: 
        return jsonify({"ok": False, "reason": "training disabled"}), 400
    try: 
        out=train_models(); 
        return jsonify({"ok": True, "result": out})
    except Exception as e:
        log.exception("❌ train_models failed: %s", e)
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/admin/train-notify", methods=["POST","GET"])
def http_train_notify(): 
    _require_admin(); 
    log.info("🔔 Manual train notify requested")
    auto_train_job(); 
    return jsonify({"ok": True})

@app.route("/admin/digest", methods=["POST","GET"])
def http_digest(): 
    _require_admin(); 
    log.info("📊 Manual digest requested")
    msg=daily_accuracy_digest(); 
    return jsonify({"ok": True, "sent": bool(msg)})

@app.route("/admin/auto-tune", methods=["POST","GET"])
def http_auto_tune(): 
    _require_admin(); 
    log.info("🎛️ Manual auto-tune requested")
    tuned=auto_tune_thresholds(14); 
    return jsonify({"ok": True, "tuned": tuned})

@app.route("/admin/retry-unsent", methods=["POST","GET"])
def http_retry_unsent(): 
    _require_admin(); 
    log.info("🔄 Manual retry requested")
    n=retry_unsent_tips(30,200); 
    return jsonify({"ok": True, "resent": n})

@app.route("/admin/prematch-scan", methods=["POST","GET"])
def http_prematch_scan(): 
    _require_admin(); 
    log.info("🔍 Manual prematch scan requested")
    saved=prematch_scan_save(); 
    return jsonify({"ok": True, "saved": int(saved)})

@app.route("/admin/motd", methods=["POST","GET"])
def http_motd():
    _require_admin(); 
    log.info("🏅 Manual MOTD requested")
    ok = send_match_of_the_day(); 
    return jsonify({"ok": bool(ok)})

@app.route("/admin/calibration-check", methods=["POST","GET"])
def http_calibration_check():
    _require_admin(); 
    log.info("🔧 Manual calibration check requested")
    calibration_check_job(); 
    return jsonify({"ok": True, "message": "Calibration check initiated"})

@app.route("/settings/<key>", methods=["GET","POST"])
def http_settings(key: str):
    _require_admin()
    log.info("⚙️ Settings access for key: %s", key)
    if request.method=="GET":
        val=get_setting_cached(key); 
        return jsonify({"ok": True, "key": key, "value": val})
    val=(request.get_json(silent=True) or {}).get("value")
    if val is None: 
        log.warning("❌ No value provided for setting %s", key)
        abort(400)
    set_setting(key, str(val)); 
    _SETTINGS_CACHE.invalidate(key); 
    invalidate_model_caches_for_key(key)
    log.info("✅ Setting %s updated to: %s", key, val)
    return jsonify({"ok": True})

@app.route("/tips/latest")
def http_latest():
    limit=int(request.args.get("limit","50"))
    log.debug("📋 Latest tips requested, limit: %s", limit)
    with db_conn() as c:
        rows=c.execute("SELECT match_id,league,home,away,market,suggestion,confidence,confidence_raw,score_at_tip,minute,created_ts,odds,book,ev_pct "
                       "FROM tips WHERE suggestion<>'HARVEST' ORDER BY created_ts DESC LIMIT %s",(max(1,min(500,limit)),)).fetchall()
    tips=[]
    for r in rows:
        tips.append({"match_id":int(r[0]),"league":r[1],"home":r[2],"away":r[3],"market":r[4],"suggestion":r[5],
                     "confidence":float(r[6]),"confidence_raw":(float(r[7]) if r[7] is not None else None),
                     "score_at_tip":r[8],"minute":int(r[9]),"created_ts":int(r[10]),
                     "odds": (float(r[11]) if r[11] is not None else None), "book": r[12], "ev_pct": (float(r[13]) if r[13] is not None else None)})
    log.debug("✅ Returned %s tips", len(tips))
    return jsonify({"ok": True, "tips": tips, "count": len(tips)})

@app.route("/telegram/webhook/<secret>", methods=["POST"])
def telegram_webhook(secret: str):
    log.debug("🤖 Telegram webhook accessed with secret")
    if (WEBHOOK_SECRET or "") != secret: 
        log.warning("❌ Invalid webhook secret")
        abort(403)
    update=request.get_json(silent=True) or {}
    try:
        msg=(update.get("message") or {}).get("text") or ""
        if msg.startswith("/start"): 
            send_telegram("👋 goalsniper bot (FULL AI mode) is online with intensive logging.")
        elif msg.startswith("/digest"): 
            daily_accuracy_digest()
        elif msg.startswith("/motd"): 
            send_match_of_the_day()
        elif msg.startswith("/scan"):
            parts=msg.split()
            if len(parts)>1 and ADMIN_API_KEY and parts[1]==ADMIN_API_KEY:
                s,l=production_scan(); 
                send_telegram(f"🔁 Scan done. Saved: {s}, Live seen: {l}")
            else: 
                send_telegram("🔒 Admin key required.")
        log.debug("✅ Processed Telegram webhook message: %s", msg[:100])
    except Exception as e:
        log.warning("❌ Telegram webhook parse error: %s", e)
    return jsonify({"ok": True})

# ───────── Boot ─────────
def _on_boot():
    log.info("🚀 Booting goalsniper...")
    _init_pool(); 
    init_db(); 
    set_setting("boot_ts", str(int(time.time())))
    log.info("✅ Boot sequence completed")

_on_boot()

if __name__ == "__main__":
    log.info("🎬 Starting Flask application...")
    app.run(host=os.getenv("HOST","0.0.0.0"), port=int(os.getenv("PORT","8080")), debug=False)
