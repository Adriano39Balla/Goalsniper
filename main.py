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

# ───────── App / logging ─────────
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
log = logging.getLogger("goalsniper")
app = Flask(__name__)

log.info("🚀 Starting goalsniper application")

# ───────── Env bootstrap ─────────
log.info("🔧 Loading environment variables")
try:
    from dotenv import load_dotenv
    load_dotenv()
    log.info("✅ Environment variables loaded from .env file")
except Exception as e:
    log.info("ℹ️ No .env file found or error loading: %s", e)
    pass

# ───────── Core env ─────────
log.info("📋 Loading core environment variables")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")
API_KEY            = os.getenv("API_KEY")
ADMIN_API_KEY      = os.getenv("ADMIN_API_KEY")
WEBHOOK_SECRET     = os.getenv("TELEGRAM_WEBHOOK_SECRET")
RUN_SCHEDULER      = os.getenv("RUN_SCHEDULER", "1") not in ("0","false","False","no","NO")

log.info("⚙️ Loading configuration thresholds and limits")
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
    log.info("🏆 MOTD League IDs loaded: %s", MOTD_LEAGUE_IDS)
except Exception as e:
    log.warning("⚠️ Failed to parse MOTD_LEAGUE_IDS: %s", e)
    MOTD_LEAGUE_IDS = []

# ───────── Lines ─────────
def _parse_lines(env_val: str, default: List[float]) -> List[float]:
    log.debug("📐 Parsing lines from env value: %s", env_val)
    out=[]
    for t in (env_val or "").split(","):
        t=t.strip()
        if not t: continue
        try: out.append(float(t))
        except: pass
    result = out or default
    log.debug("📐 Parsed lines: %s (default: %s)", result, default)
    return result

log.info("🎯 Loading OU lines configuration")
OU_LINES = [ln for ln in _parse_lines(os.getenv("OU_LINES","2.5,3.5"), [2.5,3.5]) if abs(ln-1.5)>1e-6]
log.info("✅ OU lines configured: %s", OU_LINES)

TOTAL_MATCH_MINUTES   = int(os.getenv("TOTAL_MATCH_MINUTES", "95"))
PREDICTIONS_PER_MATCH = int(os.getenv("PREDICTIONS_PER_MATCH", "2"))

# ───────── Odds/EV controls ─────────
log.info("💰 Loading odds and EV controls")
MIN_ODDS_OU   = float(os.getenv("MIN_ODDS_OU",   "1.30"))
MIN_ODDS_BTTS = float(os.getenv("MIN_ODDS_BTTS", "1.30"))
MIN_ODDS_1X2  = float(os.getenv("MIN_ODDS_1X2",  "1.30"))
MAX_ODDS_ALL  = float(os.getenv("MAX_ODDS_ALL",  "20.0"))
EDGE_MIN_BPS  = int(os.getenv("EDGE_MIN_BPS", "300"))  # 300 = +3.00%
ODDS_BOOKMAKER_ID = os.getenv("ODDS_BOOKMAKER_ID")  # optional API-Football book id
ALLOW_TIPS_WITHOUT_ODDS = os.getenv("ALLOW_TIPS_WITHOUT_ODDS","1") not in ("0","false","False","no","NO")

log.info("📊 Odds configuration: MIN_ODDS_OU=%.2f, MIN_ODDS_BTTS=%.2f, MIN_ODDS_1X2=%.2f, MAX_ODDS_ALL=%.2f", 
         MIN_ODDS_OU, MIN_ODDS_BTTS, MIN_ODDS_1X2, MAX_ODDS_ALL)
log.info("📈 EV configuration: EDGE_MIN_BPS=%d, ALLOW_TIPS_WITHOUT_ODDS=%s", EDGE_MIN_BPS, ALLOW_TIPS_WITHOUT_ODDS)

# ───────── Markets allow-list (draw suppressed) ─────────
log.info("🎲 Building allowed suggestions list")
ALLOWED_SUGGESTIONS = {"BTTS: Yes", "BTTS: No", "OU 2.5", "OU 3.5", "Home Win", "Away Win"}
def _fmt_line(line: float) -> str: return f"{line}".rstrip("0").rstrip(".")
for _ln in OU_LINES:
    s=_fmt_line(_ln); ALLOWED_SUGGESTIONS.add(f"Over {s} Goals"); ALLOWED_SUGGESTIONS.add(f"Under {s} Goals")
log.info("✅ Allowed suggestions: %s", sorted(list(ALLOWED_SUGGESTIONS)))

# ───────── External APIs / HTTP session ─────────
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL: 
    log.critical("❌ DATABASE_URL is required but not set")
    raise SystemExit("DATABASE_URL is required")
log.info("✅ DATABASE_URL configured")

BASE_URL = "https://v3.football.api-sports.io"
FOOTBALL_API_URL = f"{BASE_URL}/fixtures"
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
HEADERS = {"x-apisports-key": API_KEY, "Accept": "application/json"}
INPLAY_STATUSES = {"1H","HT","2H","ET","BT","P"}

session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1, status_forcelist=[429,500,502,503,504], respect_retry_after_header=True)))
log.info("🌐 HTTP session configured with retry logic")

# ───────── Caches & timezones ─────────
STATS_CACHE:  Dict[int, Tuple[float, list]] = {}
EVENTS_CACHE: Dict[int, Tuple[float, list]] = {}
ODDS_CACHE:   Dict[int, Tuple[float, dict]] = {}
SETTINGS_TTL = int(os.getenv("SETTINGS_TTL_SEC","60"))
MODELS_TTL   = int(os.getenv("MODELS_CACHE_TTL_SEC","120"))
TZ_UTC, BERLIN_TZ = ZoneInfo("UTC"), ZoneInfo("Europe/Berlin")
log.info("🕐 Timezones configured: UTC, Europe/Berlin")
log.info("🗄️ Cache TTLs: SETTINGS_TTL=%ds, MODELS_TTL=%ds", SETTINGS_TTL, MODELS_TTL)

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
    log.debug("🔍 Starting league target check")
    if not league_obj:
        log.debug("❌ League object is empty")
        return False
    
    league_name = str(league_obj.get("name", "")).strip()
    country = str(league_obj.get("country", "")).strip()
    
    # Log what we're actually getting from the API
    log.debug("🔍 Checking league: country='%s', name='%s'", country, league_name)
    
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
            log.debug("✅ Matched as: %s", possible_name)
            break
    
    if is_target:
        log.debug("✅ League accepted: %s - %s", country, league_name)
    else:
        log.debug("❌ League rejected: %s - %s", country, league_name)
        
    return is_target

# ───────── Optional import: trainer ─────────
try:
    log.info("🤖 Attempting to import train_models module")
    from train_models import train_models
    log.info("✅ Successfully imported train_models")
except Exception as e:
    _IMPORT_ERR = repr(e)
    log.warning("⚠️ Failed to import train_models: %s", _IMPORT_ERR)
    def train_models(*args, **kwargs):  # type: ignore
        log.warning("🚫 train_models not available: %s", _IMPORT_ERR)
        return {"ok": False, "reason": f"train_models import failed: {_IMPORT_ERR}"}

# ───────── DB pool & helpers ─────────
POOL: Optional[SimpleConnectionPool] = None
class PooledConn:
    def __init__(self, pool): 
        log.debug("🔌 Creating PooledConn instance")
        self.pool=pool; self.conn=None; self.cur=None
    
    def __enter__(self): 
        log.debug("🔌 Acquiring connection from pool")
        self.conn=self.pool.getconn(); 
        self.conn.autocommit=True; 
        self.cur=self.conn.cursor(); 
        log.debug("✅ Connection acquired and cursor created")
        return self
    
    def __exit__(self, a,b,c): 
        log.debug("🔌 Releasing connection back to pool")
        try: 
            self.cur and self.cur.close()
            log.debug("✅ Cursor closed")
        finally: 
            self.conn and self.pool.putconn(self.conn)
            log.debug("✅ Connection returned to pool")
    
    def execute(self, sql: str, params: tuple|list=()):
        log.debug("📝 Executing SQL: %s with params: %s", sql[:100] + "..." if len(sql) > 100 else sql, params)
        self.cur.execute(sql, params or ())
        log.debug("✅ SQL executed successfully")
        return self.cur

def _init_pool():
    global POOL
    log.info("🔌 Initializing database connection pool")
    dsn = DATABASE_URL + (("&" if "?" in DATABASE_URL else "?") + "sslmode=require" if "sslmode=" not in DATABASE_URL else "")
    POOL = SimpleConnectionPool(minconn=1, maxconn=int(os.getenv("DB_POOL_MAX","5")), dsn=dsn)
    log.info("✅ Database pool initialized: minconn=1, maxconn=%s", os.getenv("DB_POOL_MAX","5"))

def db_conn(): 
    if not POOL: 
        log.debug("🔌 Pool not initialized, calling _init_pool")
        _init_pool()
    return PooledConn(POOL)  # type: ignore

# ───────── Settings cache ─────────
class _TTLCache:
    def __init__(self, ttl): 
        self.ttl=ttl; 
        self.data={}
        log.debug("🗄️ Created TTL cache with TTL=%ds", ttl)
    
    def get(self, k): 
        log.debug("🗄️ Cache GET for key: %s", k)
        v=self.data.get(k); 
        if not v: 
            log.debug("🗄️ Cache MISS (key not found): %s", k)
            return None
        ts,val=v
        if time.time()-ts>self.ttl: 
            self.data.pop(k,None)
            log.debug("🗄️ Cache EXPIRED for key: %s (age: %.2fs > TTL: %ds)", k, time.time()-ts, self.ttl)
            return None
        log.debug("🗄️ Cache HIT for key: %s (age: %.2fs)", k, time.time()-ts)
        return val
    
    def set(self,k,v): 
        log.debug("🗄️ Cache SET for key: %s", k)
        self.data[k]=(time.time(),v)
    
    def invalidate(self,k=None): 
        if k is None:
            log.debug("🗄️ Cache CLEARED (all keys)")
            self.data.clear()
        else:
            log.debug("🗄️ Cache INVALIDATE for key: %s", k)
            self.data.pop(k,None)

_SETTINGS_CACHE, _MODELS_CACHE = _TTLCache(SETTINGS_TTL), _TTLCache(MODELS_TTL)
log.info("🗄️ Created caches: SETTINGS (TTL=%ds), MODELS (TTL=%ds)", SETTINGS_TTL, MODELS_TTL)

def get_setting(key: str) -> Optional[str]:
    log.debug("⚙️ Getting setting from DB: %s", key)
    start_time = time.time()
    with db_conn() as c:
        r=c.execute("SELECT value FROM settings WHERE key=%s",(key,)).fetchone()
        elapsed = time.time() - start_time
        if r:
            log.debug("⚙️ Setting retrieved: %s=%s (took %.3fs)", key, r[0], elapsed)
            return r[0]
        else:
            log.debug("⚙️ Setting not found: %s (took %.3fs)", key, elapsed)
            return None

def set_setting(key: str, value: str) -> None:
    log.info("⚙️ Setting value in DB: %s=%s", key, value)
    start_time = time.time()
    with db_conn() as c:
        c.execute("INSERT INTO settings(key,value) VALUES(%s,%s) ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value", (key,value))
        elapsed = time.time() - start_time
    log.debug("✅ Setting saved: %s (took %.3fs)", key, elapsed)

def get_setting_cached(key: str) -> Optional[str]:
    log.debug("⚙️ Getting cached setting: %s", key)
    v=_SETTINGS_CACHE.get(key)
    if v is None: 
        log.debug("⚙️ Cache miss, fetching from DB: %s", key)
        v=get_setting(key); 
        _SETTINGS_CACHE.set(key,v)
    return v

def invalidate_model_caches_for_key(key: str):
    if key.lower().startswith(("model","model_latest","model_v2","pre_")):
        log.debug("🗄️ Invalidating model cache for key: %s", key)
        _MODELS_CACHE.invalidate()
    else:
        log.debug("🗄️ No model cache invalidation needed for key: %s", key)

# ───────── Init DB ─────────
def init_db():
    log.info("🗄️ Initializing database schema")
    start_time = time.time()
    with db_conn() as c:
        log.debug("📋 Creating tables if they don't exist")
        c.execute("""CREATE TABLE IF NOT EXISTS tips (
            match_id BIGINT, league_id BIGINT, league TEXT,
            home TEXT, away TEXT, market TEXT, suggestion TEXT,
            confidence DOUBLE PRECISION, confidence_raw DOUBLE PRECISION,
            score_at_tip TEXT, minute INTEGER, created_ts BIGINT,
            odds DOUBLE PRECISION, book TEXT, ev_pct DOUBLE PRECISION,
            sent_ok INTEGER DEFAULT 1,
            PRIMARY KEY (match_id, created_ts))""")
        log.debug("✅ Created/verified tips table")
        
        c.execute("""CREATE TABLE IF NOT EXISTS tip_snapshots (
            match_id BIGINT, created_ts BIGINT, payload TEXT,
            PRIMARY KEY (match_id, created_ts))""")
        log.debug("✅ Created/verified tip_snapshots table")
        
        c.execute("""CREATE TABLE IF NOT EXISTS feedback (
            id SERIAL PRIMARY KEY, match_id BIGINT UNIQUE, verdict INTEGER, created_ts BIGINT)""")
        log.debug("✅ Created/verified feedback table")
        
        c.execute("""CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)""")
        log.debug("✅ Created/verified settings table")
        
        c.execute("""CREATE TABLE IF NOT EXISTS match_results (
            match_id BIGINT PRIMARY KEY, final_goals_h INTEGER, final_goals_a INTEGER, btts_yes INTEGER, updated_ts BIGINT)""")
        log.debug("✅ Created/verified match_results table")
        
        # Evolutive columns (idempotent)
        log.debug("🔧 Checking for evolutive columns")
        try: 
            c.execute("ALTER TABLE tips ADD COLUMN IF NOT EXISTS odds DOUBLE PRECISION")
            log.debug("✅ Added odds column if needed")
        except Exception as e: 
            log.debug("ℹ️ odds column already exists: %s", e)
        
        try: 
            c.execute("ALTER TABLE tips ADD COLUMN IF NOT EXISTS book TEXT")
            log.debug("✅ Added book column if needed")
        except Exception as e: 
            log.debug("ℹ️ book column already exists: %s", e)
        
        try: 
            c.execute("ALTER TABLE tips ADD COLUMN IF NOT EXISTS ev_pct DOUBLE PRECISION")
            log.debug("✅ Added ev_pct column if needed")
        except Exception as e: 
            log.debug("ℹ️ ev_pct column already exists: %s", e)
        
        try: 
            c.execute("ALTER TABLE tips ADD COLUMN IF NOT EXISTS confidence_raw DOUBLE PRECISION")
            log.debug("✅ Added confidence_raw column if needed")
        except Exception as e: 
            log.debug("ℹ️ confidence_raw column already exists: %s", e)
        
        log.debug("🔧 Creating indexes if they don't exist")
        c.execute("CREATE INDEX IF NOT EXISTS idx_tips_created ON tips (created_ts DESC)")
        log.debug("✅ Created/verified idx_tips_created index")
        
        c.execute("CREATE INDEX IF NOT EXISTS idx_tips_match ON tips (match_id)")
        log.debug("✅ Created/verified idx_tips_match index")
        
        c.execute("CREATE INDEX IF NOT EXISTS idx_tips_sent ON tips (sent_ok, created_ts DESC)")
        log.debug("✅ Created/verified idx_tips_sent index")
        
        c.execute("CREATE INDEX IF NOT EXISTS idx_snap_by_match ON tip_snapshots (match_id, created_ts DESC)")
        log.debug("✅ Created/verified idx_snap_by_match index")
        
        c.execute("CREATE INDEX IF NOT EXISTS idx_results_updated ON match_results (updated_ts DESC)")
        log.debug("✅ Created/verified idx_results_updated index")
    
    elapsed = time.time() - start_time
    log.info("✅ Database initialization completed in %.2f seconds", elapsed)

# ───────── Telegram ─────────
def send_telegram(text: str) -> bool:
    log.debug("📱 Preparing to send Telegram message")
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: 
        log.warning("⚠️ Cannot send Telegram: missing BOT_TOKEN or CHAT_ID")
        return False
    try:
        log.debug("📱 Sending Telegram message (first 100 chars): %s", text[:100])
        start_time = time.time()
        r=session.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                       data={"chat_id":TELEGRAM_CHAT_ID,"text":text,"parse_mode":"HTML","disable_web_page_preview":True}, timeout=10)
        elapsed = time.time() - start_time
        if r.ok:
            log.info("✅ Telegram sent successfully (took %.2fs)", elapsed)
            return True
        else:
            log.warning("⚠️ Telegram send failed: %s - %s", r.status_code, r.text)
            return False
    except Exception as e:
        log.error("❌ Telegram send error: %s", e)
        return False

# ───────── API helpers ─────────
def _api_get(url: str, params: dict, timeout: int = 15):
    log.debug("🌐 API GET request: %s with params: %s", url, params)
    if not API_KEY: 
        log.warning("⚠️ No API_KEY available for request")
        return None
    try:
        start_time = time.time()
        r=session.get(url, headers=HEADERS, params=params, timeout=timeout)
        elapsed = time.time() - start_time
        if r.ok:
            log.debug("✅ API request successful (took %.2fs, status: %s)", elapsed, r.status_code)
            return r.json()
        else:
            log.warning("⚠️ API request failed: %s - %s (took %.2fs)", r.status_code, r.text, elapsed)
            return None
    except Exception as e:
        elapsed = time.time() - start_time if 'start_time' in locals() else 0
        log.error("❌ API request error: %s (took %.2fs)", e, elapsed)
        return None

# ───────── League filter ─────────
_BLOCK_PATTERNS = ["u17","u18","u19","u20","u21","u23","youth","junior","reserve","res.","friendlies","friendly"]
def _blocked_league(league_obj: dict) -> bool:
    log.debug("🚫 Checking if league is blocked")
    name=str((league_obj or {}).get("name","")).lower()
    country=str((league_obj or {}).get("country","")).lower()
    typ=str((league_obj or {}).get("type","")).lower()
    txt=f"{country} {name} {typ}"
    
    # Check for youth/reserve patterns
    for p in _BLOCK_PATTERNS:
        if p in txt:
            log.debug("🚫 League blocked by pattern '%s': %s", p, txt)
            return True
    
    # Check for specific league IDs
    allow=[x.strip() for x in os.getenv("MOTD_LEAGUE_IDS","").split(",") if x.strip()]  # not used for live
    deny=[x.strip() for x in os.getenv("LEAGUE_DENY_IDS","").split(",") if x.strip()]
    lid=str((league_obj or {}).get("id") or "")
    
    if lid in deny:
        log.debug("🚫 League blocked by ID in deny list: %s", lid)
        return True
    
    log.debug("✅ League not blocked: %s", txt)
    return False

# ───────── Live fetches ─────────
def fetch_match_stats(fid: int) -> list:
    log.debug("📊 Fetching match stats for fixture %s", fid)
    now=time.time()
    if fid in STATS_CACHE and now-STATS_CACHE[fid][0] < 90: 
        log.debug("📊 Using cached stats for fixture %s (age: %.1fs)", fid, now-STATS_CACHE[fid][0])
        return STATS_CACHE[fid][1]
    
    log.debug("📊 Cache miss, fetching fresh stats for fixture %s", fid)
    js=_api_get(f"{FOOTBALL_API_URL}/statistics", {"fixture": fid}) or {}
    out=js.get("response",[]) if isinstance(js,dict) else []
    STATS_CACHE[fid]=(now,out)
    log.debug("📊 Stats fetched for fixture %s: %s items", fid, len(out))
    return out

def fetch_match_events(fid: int) -> list:
    log.debug("⚽ Fetching match events for fixture %s", fid)
    now=time.time()
    if fid in EVENTS_CACHE and now-EVENTS_CACHE[fid][0] < 90: 
        log.debug("⚽ Using cached events for fixture %s (age: %.1fs)", fid, now-EVENTS_CACHE[fid][0])
        return EVENTS_CACHE[fid][1]
    
    log.debug("⚽ Cache miss, fetching fresh events for fixture %s", fid)
    js=_api_get(f"{FOOTBALL_API_URL}/events", {"fixture": fid}) or {}
    out=js.get("response",[]) if isinstance(js,dict) else []
    EVENTS_CACHE[fid]=(now,out)
    log.debug("⚽ Events fetched for fixture %s: %s items", fid, len(out))
    return out

def fetch_live_matches() -> List[dict]:
    log.info("🔍 Fetching live matches")
    js=_api_get(FOOTBALL_API_URL, {"live":"all"}) or {}
    matches=[m for m in (js.get("response",[]) if isinstance(js,dict) else []) if not _blocked_league(m.get("league") or {})]
    log.info("📈 Found %s live matches (after filtering)", len(matches))
    
    out=[]
    for m in matches:
        st=((m.get("fixture",{}) or {}).get("status",{}) or {})
        elapsed=st.get("elapsed"); short=(st.get("short") or "").upper()
        if elapsed is None or elapsed>120 or short not in INPLAY_STATUSES: 
            log.debug("⏭️ Skipping match: elapsed=%s, short=%s", elapsed, short)
            continue
        
        fid=(m.get("fixture",{}) or {}).get("id")
        log.debug("🔄 Fetching stats and events for fixture %s", fid)
        m["statistics"]=fetch_match_stats(fid); 
        m["events"]=fetch_match_events(fid)
        out.append(m)
    
    log.info("✅ Live matches processed: %s matches ready for analysis", len(out))
    return out

# ───────── Prematch helpers (short) ─────────
def _api_last_fixtures(team_id: int, n: int = 5) -> List[dict]:
    log.debug("📅 Fetching last %s fixtures for team %s", n, team_id)
    js=_api_get(f"{BASE_URL}/fixtures", {"team":team_id,"last":n}) or {}
    result = js.get("response",[]) if isinstance(js,dict) else []
    log.debug("📅 Found %s past fixtures for team %s", len(result), team_id)
    return result

def _api_h2h(home_id: int, away_id: int, n: int = 5) -> List[dict]:
    log.debug("🤝 Fetching H2H for %s vs %s (last %s)", home_id, away_id, n)
    js=_api_get(f"{BASE_URL}/fixtures/headtohead", {"h2h":f"{home_id}-{away_id}","last":n}) or {}
    result = js.get("response",[]) if isinstance(js,dict) else []
    log.debug("🤝 Found %s H2H fixtures for %s vs %s", len(result), home_id, away_id)
    return result

def _collect_todays_prematch_fixtures() -> List[dict]:
    log.info("📅 Collecting today's prematch fixtures")
    today_local=datetime.now(BERLIN_TZ).date()
    start_local=datetime.combine(today_local, datetime.min.time(), tzinfo=ZoneInfo("Europe/Berlin"))
    end_local=start_local+timedelta(days=1)
    dates_utc={start_local.astimezone(TZ_UTC).date(), (end_local - timedelta(seconds=1)).astimezone(TZ_UTC).date()}
    
    log.debug("📅 Date range: %s (Berlin) -> UTC dates: %s", today_local, dates_utc)
    
    fixtures=[]
    for d in sorted(dates_utc):
        log.debug("📅 Fetching fixtures for date: %s", d)
        js=_api_get(FOOTBALL_API_URL, {"date": d.strftime("%Y-%m-%d")}) or {}
        for r in js.get("response",[]) if isinstance(js,dict) else []:
            if (((r.get("fixture") or {}).get("status") or {}).get("short") or "").upper() == "NS":
                fixtures.append(r)
    
    log.debug("📅 Found %s total fixtures, filtering blocked leagues", len(fixtures))
    fixtures=[f for f in fixtures if not _blocked_league(f.get("league") or {})]
    log.info("✅ Today's prematch fixtures: %s matches", len(fixtures))
    return fixtures

# ───────── Feature extraction (live) ─────────
def _num(v) -> float:
    try:
        if isinstance(v,str) and v.endswith("%"): 
            result = float(v[:-1])
            log.debug("🔢 Converted percentage string '%s' to float: %s", v, result)
            return result
        result = float(v or 0)
        log.debug("🔢 Converted '%s' to float: %s", v, result)
        return result
    except Exception as e:
        log.debug("🔢 Failed to convert '%s' to float, using 0.0: %s", v, e)
        return 0.0

def _pos_pct(v) -> float:
    try: 
        result = float(str(v).replace("%","").strip() or 0)
        log.debug("🎯 Converted position percentage '%s' to float: %s", v, result)
        return result
    except Exception as e:
        log.debug("🎯 Failed to convert position percentage '%s', using 0.0: %s", v, e)
        return 0.0

def extract_features(m: dict) -> Dict[str,float]:
    log.debug("📊 Extracting features from match data")
    home=m["teams"]["home"]["name"]; away=m["teams"]["away"]["name"]
    gh=m["goals"]["home"] or 0; ga=m["goals"]["away"] or 0
    minute=int(((m.get("fixture") or {}).get("status") or {}).get("elapsed") or 0)
    
    log.debug("📊 Match: %s vs %s, Score: %s-%s, Minute: %s", home, away, gh, ga, minute)
    
    stats={}
    for s in (m.get("statistics") or []):
        t=(s.get("team") or {}).get("name")
        if t: 
            stats[t]={ (i.get("type") or ""): i.get("value") for i in (s.get("statistics") or []) }
            log.debug("📊 Stats for team %s: %s items", t, len(stats[t]))
    
    sh=stats.get(home,{}) or {}; sa=stats.get(away,{}) or {}
    
    # Extract individual stats
    xg_h=_num(sh.get("Expected Goals",0)); xg_a=_num(sa.get("Expected Goals",0))
    sot_h=_num(sh.get("Shots on Target",0)); sot_a=_num(sa.get("Shots on Target",0))
    cor_h=_num(sh.get("Corner Kicks",0)); cor_a=_num(sa.get("Corner Kicks",0))
    pos_h=_pos_pct(sh.get("Ball Possession",0)); pos_a=_pos_pct(sa.get("Ball Possession",0))
    
    log.debug("📊 Basic stats - xG: %s-%s, SOT: %s-%s, Corners: %s-%s, Possession: %s%%-%s%%", 
              xg_h, xg_a, sot_h, sot_a, cor_h, cor_a, pos_h, pos_a)
    
    # Count red cards
    red_h=red_a=0
    for ev in (m.get("events") or []):
        if (ev.get("type","").lower()=="card"):
            d=(ev.get("detail","") or "").lower()
            if "red" in d or "second yellow" in d:
                t=(ev.get("team") or {}).get("name") or ""
                if t==home: red_h+=1
                elif t==away: red_a+=1
    
    log.debug("📊 Red cards: %s-%s", red_h, red_a)
    
    features = {
        "minute":float(minute),
        "goals_h":float(gh),"goals_a":float(ga),"goals_sum":float(gh+ga),"goals_diff":float(gh-ga),
        "xg_h":float(xg_h),"xg_a":float(xg_a),"xg_sum":float(xg_h+xg_a),"xg_diff":float(xg_h-xg_a),
        "sot_h":float(sot_h),"sot_a":float(sot_a),"sot_sum":float(sot_h+sot_a),
        "cor_h":float(cor_h),"cor_a":float(cor_a),"cor_sum":float(cor_h+cor_a),
        "pos_h":float(pos_h),"pos_a":float(pos_a),"pos_diff":float(pos_h-pos_a),
        "red_h":float(red_h),"red_a":float(red_a),"red_sum":float(red_h+red_a)
    }
    
    log.debug("✅ Features extracted: %s", {k: round(v, 2) for k, v in features.items()})
    return features

def stats_coverage_ok(feat: Dict[str,float], minute: int) -> bool:
    log.debug("📊 Checking stats coverage for minute %s", minute)
    require_stats_minute=int(os.getenv("REQUIRE_STATS_MINUTE","35"))
    require_fields=int(os.getenv("REQUIRE_DATA_FIELDS","2"))
    
    if minute < require_stats_minute: 
        log.debug("✅ Stats coverage OK (minute %s < required %s)", minute, require_stats_minute)
        return True
    
    fields=[feat.get("xg_sum",0.0), feat.get("sot_sum",0.0), feat.get("cor_sum",0.0),
            max(feat.get("pos_h",0.0), feat.get("pos_a",0.0))]
    
    nonzero=sum(1 for v in fields if (v or 0)>0)
    result = nonzero >= max(0, require_fields)
    
    log.debug("📊 Stats coverage check: %s non-zero fields (need %s) -> %s", 
              nonzero, require_fields, "OK" if result else "FAIL")
    return result

def _league_name(m: dict) -> Tuple[int,str]:
    lg=(m.get("league") or {}) or {}
    league_id = int(lg.get("id") or 0)
    league_name = f"{lg.get('country','')} - {lg.get('name','')}".strip(" -")
    log.debug("🏆 League: ID=%s, Name=%s", league_id, league_name)
    return league_id, league_name

def _teams(m: dict) -> Tuple[str,str]:
    t=(m.get("teams") or {}) or {}
    home = t.get("home",{}).get("name","")
    away = t.get("away",{}).get("name","")
    log.debug("👥 Teams: %s vs %s", home, away)
    return home, away

def _pretty_score(m: dict) -> str:
    gh=(m.get("goals") or {}).get("home") or 0; ga=(m.get("goals") or {}).get("away") or 0
    score = f"{gh}-{ga}"
    log.debug("📊 Score: %s", score)
    return score

# ───────── Models ─────────
MODEL_KEYS_ORDER=["model_v2:{name}","model_latest:{name}","model:{name}"]
EPS=1e-12

def _sigmoid(x: float) -> float:
    try:
        if x<-50: 
            result = 1e-22
            log.debug("📈 Sigmoid(%s) -> %s (clamped low)", x, result)
            return result
        if x>50:  
            result = 1-1e-22
            log.debug("📈 Sigmoid(%s) -> %s (clamped high)", x, result)
            return result
        import math; 
        result = 1/(1+math.exp(-x))
        log.debug("📈 Sigmoid(%s) -> %s", x, result)
        return result
    except Exception as e:
        log.error("❌ Sigmoid calculation error: %s, using 0.5", e)
        return 0.5

def _logit(p: float) -> float:
    import math; 
    p=max(EPS,min(1-EPS,float(p))); 
    result = math.log(p/(1-p))
    log.debug("📈 Logit(%s) -> %s", p, result)
    return result

def load_model_from_settings(name: str) -> Optional[Dict[str,Any]]:
    log.debug("🤖 Loading model: %s", name)
    cached=_MODELS_CACHE.get(name)
    if cached is not None: 
        log.debug("🤖 Model cache HIT: %s", name)
        return cached
    
    log.debug("🤖 Model cache MISS: %s", name)
    mdl=None
    for pat in MODEL_KEYS_ORDER:
        key = pat.format(name=name)
        log.debug("🤖 Trying model key: %s", key)
        raw=get_setting_cached(key)
        if not raw: 
            log.debug("🤖 Model key not found: %s", key)
            continue
        
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
            mdl=tmp; 
            log.info("✅ Model loaded successfully: %s", name)
            log.debug("🤖 Model details - intercept: %s, weights: %s keys, calibration: %s", 
                     tmp.get("intercept"), len(tmp.get("weights", {})), tmp.get("calibration"))
            break
        except Exception as e:
            log.warning("⚠️ Model parse failed for %s: %s", name, e)
    
    if mdl is not None: 
        _MODELS_CACHE.set(name, mdl)
    else:
        log.warning("⚠️ Model not found: %s", name)
    return mdl

def _linpred(feat: Dict[str,float], weights: Dict[str,float], intercept: float) -> float:
    log.debug("🧮 Calculating linear prediction")
    s=float(intercept or 0.0)
    for k,w in (weights or {}).items(): 
        feat_val = float(feat.get(k,0.0))
        s += float(w or 0.0) * feat_val
        log.debug("🧮 Weight * feature: %s * %s = %s (cumulative: %s)", w, feat_val, w*feat_val, s)
    log.debug("🧮 Linear prediction result: %s", s)
    return s

def _calibrate(p: float, cal: Dict[str,Any]) -> float:
    method=(cal or {}).get("method","sigmoid"); 
    a=float((cal or {}).get("a",1.0)); 
    b=float((cal or {}).get("b",0.0))
    
    log.debug("🎯 Calibrating probability %s with method %s, a=%s, b=%s", p, method, a, b)
    
    if method.lower()=="platt": 
        result = _sigmoid(a*_logit(p)+b)
        log.debug("🎯 Platt calibration result: %s", result)
        return result
    
    import math; 
    p=max(EPS,min(1-EPS,float(p))); 
    z=math.log(p/(1-p)); 
    result = _sigmoid(a*z+b)
    log.debug("🎯 Sigmoid calibration result: %s", result)
    return result

def _score_prob(feat: Dict[str,float], mdl: Dict[str,Any]) -> float:
    log.debug("📈 Scoring probability with model")
    p=_sigmoid(_linpred(feat, mdl.get("weights",{}), float(mdl.get("intercept",0.0))))
    log.debug("📈 Raw probability before calibration: %s", p)
    
    cal=mdl.get("calibration") or {}
    try: 
        if cal: 
            p=_calibrate(p, cal)
            log.debug("📈 Probability after calibration: %s", p)
    except Exception as e:
        log.warning("⚠️ Calibration failed: %s", e)
        pass
    
    result = max(0.0, min(1.0, float(p)))
    log.debug("📈 Final probability: %s", result)
    return result

def _load_ou_model_for_line(line: float) -> Optional[Dict[str,Any]]:
    name=f"OU_{_fmt_line(line)}"
    log.debug("🎯 Loading OU model for line %s (name: %s)", line, name)
    mdl=load_model_from_settings(name)
    
    if not mdl and abs(line-2.5)<1e-6:
        log.debug("🎯 Falling back to O25 model for line 2.5")
        mdl = load_model_from_settings("O25")
    
    if mdl:
        log.debug("✅ OU model loaded for line %s", line)
    else:
        log.debug("⚠️ No OU model found for line %s", line)
    return mdl

def _load_wld_models(): 
    log.debug("🏆 Loading Win/Lose/Draw models")
    home_model = load_model_from_settings("WLD_HOME")
    draw_model = load_model_from_settings("WLD_DRAW")
    away_model = load_model_from_settings("WLD_AWAY")
    
    log.debug("🏆 WLD models loaded - Home: %s, Draw: %s, Away: %s", 
             "✓" if home_model else "✗", 
             "✓" if draw_model else "✗", 
             "✓" if away_model else "✗")
    
    return home_model, draw_model, away_model

# ───────── Odds helpers ─────────
def _ev(prob: float, odds: float) -> float:
    """Return expected value as decimal (e.g. 0.05 = +5%)."""
    result = prob*max(0.0, float(odds)) - 1.0
    log.debug("💰 EV calculation: prob=%s * odds=%s - 1 = %s", prob, odds, result)
    return result

def _min_odds_for_market(market: str) -> float:
    if market.startswith("Over/Under"): 
        result = MIN_ODDS_OU
    elif market == "BTTS": 
        result = MIN_ODDS_BTTS
    elif market == "1X2":  
        result = MIN_ODDS_1X2
    else:
        result = 1.01
    
    log.debug("💰 Min odds for market '%s': %s", market, result)
    return result

def _odds_cache_get(fid: int) -> Optional[dict]:
    rec=ODDS_CACHE.get(fid)
    if not rec: 
        log.debug("💰 Odds cache MISS for fixture %s", fid)
        return None
    
    ts,data=rec
    if time.time()-ts>120: 
        ODDS_CACHE.pop(fid,None)
        log.debug("💰 Odds cache EXPIRED for fixture %s (age: %.1fs)", fid, time.time()-ts)
        return None
    
    log.debug("💰 Odds cache HIT for fixture %s (age: %.1fs)", fid, time.time()-ts)
    return data

def _market_name_normalize(s: str) -> str:
    s=(s or "").lower()
    if "both teams" in s or "btts" in s: 
        result = "BTTS"
    elif "match winner" in s or "winner" in s or "1x2" in s: 
        result = "1X2"
    elif "over/under" in s or "total" in s or "goals" in s: 
        result = "OU"
    else:
        result = s
    
    log.debug("🏷️ Market name normalized: '%s' -> '%s'", s, result)
    return result

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
    cached=_odds_cache_get(fid)
    if cached is not None: 
        log.debug("💰 Returning cached odds for fixture %s", fid)
        return cached
    
    params={"fixture": fid}
    if ODDS_BOOKMAKER_ID: 
        params["bookmaker"] = ODDS_BOOKMAKER_ID
        log.debug("💰 Using specific bookmaker: %s", ODDS_BOOKMAKER_ID)
    
    js=_api_get(f"{BASE_URL}/odds", params) or {}
    out={}
    try:
        log.debug("💰 Processing odds response for fixture %s", fid)
        for r in js.get("response",[]) if isinstance(js,dict) else []:
            book=(r.get("bookmakers") or [])
            if not book: 
                log.debug("💰 No bookmakers in response")
                continue
            
            bk=book[0]; book_name=bk.get("name") or "Book"
            log.debug("💰 Processing bookmaker: %s", book_name)
            
            for mkt in (bk.get("bets") or []):
                mname=_market_name_normalize(mkt.get("name",""))
                vals=mkt.get("values") or []
                log.debug("💰 Processing market: %s with %s values", mname, len(vals))
                
                # BTTS
                if mname=="BTTS":
                    d={}
                    for v in vals:
                        lbl=(v.get("value") or "").strip().lower()
                        if "yes" in lbl: d["Yes"]={"odds":float(v.get("odd") or 0), "book":book_name}
                        if "no"  in lbl: d["No"] ={"odds":float(v.get("odd") or 0), "book":book_name}
                    if d: 
                        out["BTTS"]=d
                        log.debug("💰 BTTS odds found: %s", d)
                
                # 1X2
                elif mname=="1X2":
                    d={}
                    for v in vals:
                        lbl=(v.get("value") or "").strip().lower()
                        if lbl in ("home","1"): d["Home"]={"odds":float(v.get("odd") or 0),"book":book_name}
                        if lbl in ("away","2"): d["Away"]={"odds":float(v.get("odd") or 0),"book":book_name}
                    if d: 
                        out["1X2"]=d
                        log.debug("💰 1X2 odds found: %s", d)
                
                # OU lines
                elif mname=="OU":
                    # values like "Over 2.5", "Under 2.5"
                    by_line={}
                    for v in vals:
                        lbl=(v.get("value") or "").lower()
                        if "over" in lbl or "under" in lbl:
                            try:
                                ln=float(lbl.split()[-1])
                                key=f"OU_{_fmt_line(ln)}"
                                side="Over" if "over" in lbl else "Under"
                                by_line.setdefault(key,{}).update({side: {"odds":float(v.get("odd") or 0),"book":book_name}})
                                log.debug("💰 OU odds: %s %s @ %s", side, ln, v.get("odd"))
                            except Exception as e:
                                log.debug("💰 Failed to parse OU label '%s': %s", lbl, e)
                    for k,v in by_line.items(): 
                        out[k]=v
                        log.debug("💰 Added OU market %s: %s", k, v)
        
        ODDS_CACHE[fid]=(time.time(), out)
        log.info("✅ Odds fetched for fixture %s: %s markets", fid, len(out))
    except Exception as e:
        log.error("❌ Error fetching odds for fixture %s: %s", fid, e)
        out={}
    
    return out

def _price_gate(market_text: str, suggestion: str, fid: int) -> Tuple[bool, Optional[float], Optional[str], Optional[float]]:
    """
    Return (pass, odds, book, ev_pct). If odds missing:
      - pass if ALLOW_TIPS_WITHOUT_ODDS else block.
    """
    log.debug("💰 Price gate check for market '%s', suggestion '%s', fixture %s", 
              market_text, suggestion, fid)
    
    odds_map=fetch_odds(fid) if API_KEY else {}
    log.debug("💰 Odds map for fixture %s: %s", fid, list(odds_map.keys()))
    
    odds=None; book=None
    if market_text=="BTTS":
        d=odds_map.get("BTTS",{})
        tgt="Yes" if suggestion.endswith("Yes") else "No"
        if tgt in d: 
            odds=d[tgt]["odds"]; book=d[tgt]["book"]
            log.debug("💰 BTTS odds found: %s @ %s (book: %s)", tgt, odds, book)
    elif market_text=="1X2":
        d=odds_map.get("1X2",{})
        tgt="Home" if suggestion=="Home Win" else ("Away" if suggestion=="Away Win" else None)
        if tgt and tgt in d: 
            odds=d[tgt]["odds"]; book=d[tgt]["book"]
            log.debug("💰 1X2 odds found: %s @ %s (book: %s)", tgt, odds, book)
    elif market_text.startswith("Over/Under"):
        ln=_fmt_line(float(suggestion.split()[1]))
        d=odds_map.get(f"OU_{ln}",{})
        tgt="Over" if suggestion.startswith("Over") else "Under"
        if tgt in d: 
            odds=d[tgt]["odds"]; book=d[tgt]["book"]
            log.debug("💰 OU odds found: %s %s @ %s (book: %s)", tgt, ln, odds, book)

    if odds is None:
        log.debug("💰 No odds found for suggestion")
        if ALLOW_TIPS_WITHOUT_ODDS:
            log.debug("💰 Allowing tip without odds (ALLOW_TIPS_WITHOUT_ODDS=True)")
            return (ALLOW_TIPS_WITHOUT_ODDS, None, None, None)
        else:
            log.debug("💰 Blocking tip without odds (ALLOW_TIPS_WITHOUT_ODDS=False)")
            return (False, odds, book, None)

    # price range gates
    min_odds=_min_odds_for_market(market_text)
    if not (min_odds <= odds <= MAX_ODDS_ALL):
        log.debug("💰 Odds %s outside range [%s, %s] -> BLOCKED", odds, min_odds, MAX_ODDS_ALL)
        return (False, odds, book, None)

    log.debug("💰 Price gate PASSED: odds=%s, book=%s", odds, book)
    return (True, odds, book, None)

# ───────── Snapshots ─────────
def save_snapshot_from_match(m: dict, feat: Dict[str,float]) -> None:
    log.debug("📸 Saving snapshot from match")
    fx=m.get("fixture",{}) or {}; lg=m.get("league",{}) or {}
    fid=int(fx.get("id")); league_id=int(lg.get("id") or 0)
    league=f"{lg.get('country','')} - {lg.get('name','')}".strip(" -")
    home=(m.get("teams") or {}).get("home",{}).get("name","")
    away=(m.get("teams") or {}).get("away",{}).get("name","")
    gh=(m.get("goals") or {}).get("home") or 0; ga=(m.get("goals") or {}).get("away") or 0
    minute=int(feat.get("minute",0))
    
    snapshot={
        "minute":minute,
        "gh":gh,"ga":ga,
        "league_id":league_id,
        "market":"HARVEST",
        "suggestion":"HARVEST",
        "confidence":0,
        "stat":{
            "xg_h":feat.get("xg_h",0),
            "xg_a":feat.get("xg_a",0),
            "sot_h":feat.get("sot_h",0),
            "sot_a":feat.get("sot_a",0),
            "cor_h":feat.get("cor_h",0),
            "cor_a":feat.get("cor_a",0),
            "pos_h":feat.get("pos_h",0),
            "pos_a":feat.get("pos_a",0),
            "red_h":feat.get("red_h",0),
            "red_a":feat.get("red_a",0)
        }
    }
    
    now=int(time.time())
    log.debug("📸 Creating snapshot for fixture %s at minute %s", fid, minute)
    
    with db_conn() as c:
        c.execute("INSERT INTO tip_snapshots(match_id, created_ts, payload) VALUES (%s,%s,%s) "
                  "ON CONFLICT (match_id, created_ts) DO UPDATE SET payload=EXCLUDED.payload",
                  (fid, now, json.dumps(snapshot)[:200000]))
        log.debug("✅ Snapshot saved to tip_snapshots")
        
        c.execute("INSERT INTO tips(match_id,league_id,league,home,away,market,suggestion,confidence,confidence_raw,score_at_tip,minute,created_ts,sent_ok) "
                  "VALUES (%s,%s,%s,%s,%s,'HARVEST','HARVEST',0.0,0.0,%s,%s,%s,1)",
                  (fid, league_id, league, home, away, f"{gh}-{ga}", minute, now))
        log.debug("✅ Harvest record saved to tips")
    
    log.info("📸 Snapshot saved for %s vs %s (minute %s)", home, away, minute)

# ───────── Outcomes/backfill/digest (short) ─────────
def _parse_ou_line_from_suggestion(s: str) -> Optional[float]:
    try:
        log.debug("📐 Parsing OU line from suggestion: %s", s)
        for tok in (s or "").split():
            try: 
                result = float(tok)
                log.debug("📐 Found line: %s", result)
                return result
            except: 
                continue
    except Exception as e:
        log.debug("📐 Failed to parse OU line: %s", e)
        pass
    return None

def _tip_outcome_for_result(suggestion: str, res: Dict[str,Any]) -> Optional[int]:
    log.debug("📊 Calculating tip outcome for suggestion: %s", suggestion)
    gh=int(res.get("final_goals_h") or 0); ga=int(res.get("final_goals_a") or 0)
    total=gh+ga; btts=int(res.get("btts_yes") or 0); s=(suggestion or "").strip()
    
    log.debug("📊 Final score: %s-%s (total: %s, BTTS: %s)", gh, ga, total, btts)
    
    if s.startswith("Over") or s.startswith("Under"):
        line=_parse_ou_line_from_suggestion(s); 
        if line is None: 
            log.debug("📊 Could not parse line from suggestion")
            return None
        
        if s.startswith("Over"):
            if total>line: 
                log.debug("📊 OVER WIN: total %s > line %s", total, line)
                return 1
            if abs(total-line)<1e-9: 
                log.debug("📊 OVER PUSH: total %s == line %s", total, line)
                return None
            log.debug("📊 OVER LOSS: total %s <= line %s", total, line)
            return 0
        else:
            if total<line: 
                log.debug("📊 UNDER WIN: total %s < line %s", total, line)
                return 1
            if abs(total-line)<1e-9: 
                log.debug("📊 UNDER PUSH: total %s == line %s", total, line)
                return None
            log.debug("📊 UNDER LOSS: total %s >= line %s", total, line)
            return 0
    
    if s=="BTTS: Yes": 
        result = 1 if btts==1 else 0
        log.debug("📊 BTTS: Yes -> %s (actual BTTS: %s)", "WIN" if result==1 else "LOSS", btts)
        return result
    
    if s=="BTTS: No":  
        result = 1 if btts==0 else 0
        log.debug("📊 BTTS: No -> %s (actual BTTS: %s)", "WIN" if result==1 else "LOSS", btts)
        return result
    
    if s=="Home Win":  
        result = 1 if gh>ga else 0
        log.debug("📊 Home Win -> %s (score: %s-%s)", "WIN" if result==1 else "LOSS", gh, ga)
        return result
    
    if s=="Away Win":  
        result = 1 if ga>gh else 0
        log.debug("📊 Away Win -> %s (score: %s-%s)", "WIN" if result==1 else "LOSS", gh, ga)
        return result
    
    log.debug("📊 Unknown suggestion type: %s", s)
    return None

def _fixture_by_id(mid: int) -> Optional[dict]:
    log.debug("🔍 Fetching fixture by ID: %s", mid)
    js=_api_get(FOOTBALL_API_URL, {"id": mid}) or {}
    arr=js.get("response") or [] if isinstance(js,dict) else []
    result = arr[0] if arr else None
    if result:
        log.debug("✅ Found fixture %s", mid)
    else:
        log.debug("⚠️ Fixture not found: %s", mid)
    return result

def _is_final(short: str) -> bool: 
    result = (short or "").upper() in {"FT","AET","PEN"}
    log.debug("📊 Is final status '%s'? %s", short, result)
    return result

def backfill_results_for_open_matches(max_rows: int = 200) -> int:
    log.info("🔄 Starting backfill for open matches (max %s rows)", max_rows)
    now_ts=int(time.time()); cutoff=now_ts - BACKFILL_DAYS*24*3600; updated=0
    
    log.debug("🔄 Looking for matches without results (cutoff: %s days ago)", BACKFILL_DAYS)
    with db_conn() as c:
        rows=c.execute("""
            WITH last AS (SELECT match_id, MAX(created_ts) last_ts FROM tips WHERE created_ts >= %s GROUP BY match_id)
            SELECT l.match_id FROM last l LEFT JOIN match_results r ON r.match_id=l.match_id
            WHERE r.match_id IS NULL ORDER BY l.last_ts DESC LIMIT %s
        """,(cutoff, max_rows)).fetchall()
    
    log.debug("🔄 Found %s matches without results", len(rows))
    
    for (mid,) in rows:
        log.debug("🔄 Processing match %s", mid)
        fx=_fixture_by_id(int(mid))
        if not fx: 
            log.debug("🔄 Fixture not found: %s", mid)
            continue
        
        st=(((fx.get("fixture") or {}).get("status") or {}).get("short") or "")
        if not _is_final(st): 
            log.debug("🔄 Match not final: %s (status: %s)", mid, st)
            continue
        
        g=fx.get("goals") or {}; gh=int(g.get("home") or 0); ga=int(g.get("away") or 0)
        btts=1 if (gh>0 and ga>0) else 0
        
        log.debug("🔄 Final score for %s: %s-%s (BTTS: %s)", mid, gh, ga, btts)
        
        with db_conn() as c2:
            c2.execute("INSERT INTO match_results(match_id, final_goals_h, final_goals_a, btts_yes, updated_ts) "
                       "VALUES(%s,%s,%s,%s,%s) ON CONFLICT(match_id) DO UPDATE SET final_goals_h=EXCLUDED.final_goals_h, "
                       "final_goals_a=EXCLUDED.final_goals_a, btts_yes=EXCLUDED.btts_yes, updated_ts=EXCLUDED.updated_ts",
                       (int(mid), gh, ga, btts, int(time.time())))
        
        updated+=1
        log.debug("🔄 Updated result for match %s", mid)
    
    if updated: 
        log.info("✅ Backfill completed: %s matches updated", updated)
    else:
        log.info("ℹ️ Backfill: no matches needed updating")
    return updated

def daily_accuracy_digest() -> Optional[str]:
    if not DAILY_ACCURACY_DIGEST_ENABLE: 
        log.info("📊 Daily digest disabled")
        return None
    
    log.info("📊 Generating daily accuracy digest")
    now_local=datetime.now(BERLIN_TZ)
    y0=(now_local - timedelta(days=1)).replace(hour=0,minute=0,second=0,microsecond=0); y1=y0+timedelta(days=1)
    
    log.debug("📊 Date range: %s to %s (Berlin time)", y0, y1)
    
    # Ensure we have latest results
    backfill_results_for_open_matches(400)
    
    with db_conn() as c:
        rows=c.execute("""
            SELECT t.match_id, t.market, t.suggestion, t.confidence, t.confidence_raw, t.created_ts,
                   r.final_goals_h, r.final_goals_a, r.btts_yes
            FROM tips t LEFT JOIN match_results r ON r.match_id=t.match_id
            WHERE t.created_ts >= %s AND t.created_ts < %s AND t.suggestion<>'HARVEST' AND t.sent_ok=1
        """,(int(y0.timestamp()), int(y1.timestamp()))).fetchall()
    
    log.debug("📊 Found %s tips from yesterday", len(rows))
    
    total=graded=wins=0; by={}
    for (mid, mkt, sugg, conf, conf_raw, cts, gh, ga, btts) in rows:
        res={"final_goals_h":gh,"final_goals_a":ga,"btts_yes":btts}
        out=_tip_outcome_for_result(sugg,res)
        if out is None: 
            log.debug("📊 Tip %s (match %s) not gradable", sugg, mid)
            continue
        
        total+=1; graded+=1; wins+=1 if out==1 else 0
        d=by.setdefault(mkt or "?",{"graded":0,"wins":0}); d["graded"]+=1; d["wins"]+=1 if out==1 else 0
        log.debug("📊 Tip outcome: %s -> %s (market: %s)", sugg, "WIN" if out==1 else "LOSS", mkt)
    
    log.debug("📊 Summary: total=%s, graded=%s, wins=%s", total, graded, wins)
    
    if graded==0:
        msg="📊 Daily Digest\nNo graded tips for yesterday."
        log.info("📊 No graded tips for digest")
    else:
        acc=100.0*wins/max(1,graded)
        lines=[f"📊 <b>Daily Digest</b> (yesterday, Berlin time)",
               f"Tips sent: {total}  •  Graded: {graded}  •  Wins: {wins}  •  Accuracy: {acc:.1f}%"]
        
        log.info("📊 Digest accuracy: %.1f%% (%s/%s)", acc, wins, graded)
        
        for mk,st in sorted(by.items()):
            if st["graded"]==0: continue
            a=100.0*st["wins"]/st["graded"]; 
            lines.append(f"• {escape(mk)} — {st['wins']}/{st['graded']} ({a:.1f}%)")
            log.debug("📊 Market %s: %s/%s (%.1f%%)", mk, st['wins'], st['graded'], a)
        
        msg="\n".join(lines)
    
    send_telegram(msg); 
    log.info("✅ Daily digest sent to Telegram")
    return msg

# ───────── Thresholds & formatting ─────────
def _get_market_threshold_key(m: str) -> str: 
    key = f"conf_threshold:{m}"
    log.debug("⚙️ Market threshold key: %s", key)
    return key

def _get_market_threshold(m: str) -> float:
    try:
        key = _get_market_threshold_key(m)
        v=get_setting_cached(key)
        if v is not None:
            result = float(v)
            log.debug("⚙️ Market threshold for '%s': %s (from DB)", m, result)
            return result
        else:
            result = float(CONF_THRESHOLD)
            log.debug("⚙️ Market threshold for '%s': %s (default)", m, result)
            return result
    except Exception as e:
        log.error("❌ Error getting market threshold for '%s': %s, using default %s", m, e, CONF_THRESHOLD)
        return float(CONF_THRESHOLD)

def _get_market_threshold_pre(m: str) -> float: 
    result = _get_market_threshold(f"PRE {m}")
    log.debug("⚙️ Prematch threshold for '%s': %s", m, result)
    return result

def _format_tip_message(home, away, league, minute, score, suggestion, prob_pct, feat, odds=None, book=None, ev_pct=None):
    log.debug("✍️ Formatting tip message")
    stat=""
    if any([feat.get("xg_h",0),feat.get("xg_a",0),feat.get("sot_h",0),feat.get("sot_a",0),feat.get("cor_h",0),feat.get("cor_a",0),
            feat.get("pos_h",0),feat.get("pos_a",0),feat.get("red_h",0),feat.get("red_a",0)]):
        stat=(f"\n📊 xG {feat.get('xg_h',0):.2f}-{feat.get('xg_a',0):.2f}"
              f" • SOT {int(feat.get('sot_h',0))}-{int(feat.get('sot_a',0))}"
              f" • CK {int(feat.get('cor_h',0))}-{int(feat.get('cor_a',0))}")
        if feat.get("pos_h",0) or feat.get("pos_a",0): 
            stat += f" • POS {int(feat.get('pos_h',0))}%–{int(feat.get('pos_a',0))}%"
        if feat.get("red_h",0) or feat.get("red_a",0): 
            stat += f" • RED {int(feat.get('red_h',0))}-{int(feat.get('red_a',0))}"
    
    money = ""
    if odds:
        if ev_pct is not None:
            money = f"\n💰 <b>Odds:</b> {odds:.2f} @ {book or 'Book'}  •  <b>EV:</b> {ev_pct:+.1f}%"
        else:
            money = f"\n💰 <b>Odds:</b> {odds:.2f} @ {book or 'Book'}"
    
    message = ("⚽️ <b>New Tip!</b>\n"
            f"<b>Match:</b> {escape(home)} vs {escape(away)}\n"
            f"🕒 <b>Minute:</b> {minute}'  |  <b>Score:</b> {escape(score)}\n"
            f"<b>Tip:</b> {escape(suggestion)}\n"
            f"📈 <b>Confidence:</b> {prob_pct:.1f}%{money}\n"
            f"🏆 <b>League:</b> {escape(league)}{stat}")
    
    log.debug("✅ Tip message formatted (length: %s chars)", len(message))
    return message

# ───────── Scan (in-play) ─────────
def _candidate_is_sane(sug: str, feat: Dict[str,float]) -> bool:
    log.debug("🧠 Checking if candidate is sane: %s", sug)
    gh=int(feat.get("goals_h",0)); ga=int(feat.get("goals_a",0)); total=gh+ga
    
    if sug.startswith("Over"):
        ln=_parse_ou_line_from_suggestion(sug)
        if ln is None: 
            log.debug("🧠 Could not parse line from Over suggestion")
            return False
        if total > ln - 1e-9: 
            log.debug("🧠 Over %s not sane: total %s already exceeds line", ln, total)
            return False
        log.debug("🧠 Over %s is sane: total %s <= line", ln, total)
    
    if sug.startswith("Under"):
        ln=_parse_ou_line_from_suggestion(sug)
        if ln is None: 
            log.debug("🧠 Could not parse line from Under suggestion")
            return False
        if total >= ln - 1e-9: 
            log.debug("🧠 Under %s not sane: total %s already at or above line", ln, total)
            return False
        log.debug("🧠 Under %s is sane: total %s < line", ln, total)
    
    if sug.startswith("BTTS") and (gh>0 and ga>0): 
        log.debug("🧠 BTTS not sane: both teams have already scored")
        return False
    
    log.debug("✅ Candidate is sane: %s", sug)
    return True

def production_scan() -> Tuple[int,int]:
    log.info("🚀 Starting production scan")
    start_time = time.time()
    
    matches=fetch_live_matches(); live_seen=len(matches)
    log.info("📈 Live matches found: %s", live_seen)
    
    if live_seen==0: 
        log.info("✅ Production scan complete: no live matches")
        return 0,0
    
    saved=0; now_ts=int(time.time())
    with db_conn() as c:
        for i, m in enumerate(matches):
            try:
                log.debug("🔍 Processing match %s/%s", i+1, len(matches))
                fid=int((m.get("fixture",{}) or {}).get("id") or 0)
                if not fid: 
                    log.debug("⏭️ Skipping match without fixture ID")
                    continue
                
                log.debug("🎯 Processing fixture %s", fid)
                
                if DUP_COOLDOWN_MIN>0:
                    cutoff=now_ts - DUP_COOLDOWN_MIN*60
                    if c.execute("SELECT 1 FROM tips WHERE match_id=%s AND created_ts>=%s LIMIT 1",(fid,cutoff)).fetchone(): 
                        log.debug("⏭️ Skipping duplicate match %s (within cooldown)", fid)
                        continue
                
                feat=extract_features(m); minute=int(feat.get("minute",0))
                log.debug("📊 Match features extracted: minute %s", minute)
                
                if not stats_coverage_ok(feat, minute): 
                    log.debug("⏭️ Skipping due to insufficient stats coverage")
                    continue
                
                if minute < TIP_MIN_MINUTE: 
                    log.debug("⏭️ Skipping: minute %s < minimum %s", minute, TIP_MIN_MINUTE)
                    continue
                
                if HARVEST_MODE and minute>=TRAIN_MIN_MINUTE and minute%3==0:
                    try: 
                        save_snapshot_from_match(m, feat)
                        log.debug("📸 Harvest snapshot saved")
                    except Exception as e:
                        log.error("❌ Failed to save snapshot: %s", e)

                league_id, league=_league_name(m); home,away=_teams(m); score=_pretty_score(m)
                log.debug("🏆 %s vs %s (%s) - %s", home, away, league, score)
                
                candidates: List[Tuple[str,str,float]]=[]

                # OU predictions
                log.debug("🎯 Running OU predictions")
                for line in OU_LINES:
                    mdl=_load_ou_model_for_line(line)
                    if not mdl: 
                        log.debug("⏭️ No OU model for line %s", line)
                        continue
                    
                    p_over=_score_prob(feat, mdl)
                    mk=f"Over/Under {_fmt_line(line)}"; thr=_get_market_threshold(mk)
                    
                    if p_over*100.0 >= thr and _candidate_is_sane(f"Over {_fmt_line(line)} Goals", feat):
                        candidates.append((mk, f"Over {_fmt_line(line)} Goals", p_over))
                        log.debug("✅ Over %s candidate: prob=%.1f%%, threshold=%.1f%%", line, p_over*100, thr)
                    
                    p_under=1.0-p_over
                    if p_under*100.0 >= thr and _candidate_is_sane(f"Under {_fmt_line(line)} Goals", feat):
                        candidates.append((mk, f"Under {_fmt_line(line)} Goals", p_under))
                        log.debug("✅ Under %s candidate: prob=%.1f%%, threshold=%.1f%%", line, p_under*100, thr)

                # BTTS predictions
                log.debug("🎯 Running BTTS predictions")
                mdl_btts=load_model_from_settings("BTTS_YES")
                if mdl_btts:
                    p=_score_prob(feat, mdl_btts); thr=_get_market_threshold("BTTS")
                    if p*100.0>=thr and _candidate_is_sane("BTTS: Yes", feat): 
                        candidates.append(("BTTS","BTTS: Yes",p))
                        log.debug("✅ BTTS: Yes candidate: prob=%.1f%%, threshold=%.1f%%", p*100, thr)
                    
                    q=1.0-p
                    if q*100.0>=thr and _candidate_is_sane("BTTS: No", feat):  
                        candidates.append(("BTTS","BTTS: No",q))
                        log.debug("✅ BTTS: No candidate: prob=%.1f%%, threshold=%.1f%%", q*100, thr)

                # 1X2 predictions (no draw)
                log.debug("🎯 Running 1X2 predictions")
                mh,md,ma=_load_wld_models()
                if mh and md and ma:
                    ph=_score_prob(feat,mh); pd=_score_prob(feat,md); pa=_score_prob(feat,ma)
                    s=max(EPS,ph+pd+pa); ph,pa=ph/s,pa/s
                    thr=_get_market_threshold("1X2")
                    
                    if ph*100.0>=thr: 
                        candidates.append(("1X2","Home Win",ph))
                        log.debug("✅ Home Win candidate: prob=%.1f%%, threshold=%.1f%%", ph*100, thr)
                    
                    if pa*100.0>=thr: 
                        candidates.append(("1X2","Away Win",pa))
                        log.debug("✅ Away Win candidate: prob=%.1f%%, threshold=%.1f%%", pa*100, thr)

                candidates.sort(key=lambda x:x[2], reverse=True)
                log.debug("📊 Candidates found: %s", [(c[1], round(c[2]*100,1)) for c in candidates])
                
                per_match=0; base_now=int(time.time())
                for idx,(market_txt,suggestion,prob) in enumerate(candidates):
                    if suggestion not in ALLOWED_SUGGESTIONS: 
                        log.debug("⏭️ Skipping non-allowed suggestion: %s", suggestion)
                        continue
                    
                    if per_match >= max(1,PREDICTIONS_PER_MATCH): 
                        log.debug("⏭️ Max predictions per match reached: %s", PREDICTIONS_PER_MATCH)
                        break

                    # Odds/EV gate
                    log.debug("💰 Applying price/EV gate for: %s", suggestion)
                    pass_odds, odds, book, _ = _price_gate(market_txt, suggestion, fid)
                    if not pass_odds: 
                        log.debug("⏭️ Failed price gate for: %s", suggestion)
                        continue
                    
                    ev_pct=None
                    if odds is not None:
                        edge=_ev(prob, odds)  # decimal (e.g. 0.05)
                        ev_pct=round(edge*100.0,1)
                        ev_bps = int(round(edge*10000))
                        if ev_bps < EDGE_MIN_BPS:  # basis points compare
                            log.debug("⏭️ EV too low: %s bps < minimum %s bps", ev_bps, EDGE_MIN_BPS)
                            continue
                        log.debug("✅ EV passed: %s bps >= minimum %s bps", ev_bps, EDGE_MIN_BPS)

                    created_ts=base_now+idx
                    raw=float(prob); prob_pct=round(raw*100.0,1)
                    
                    log.debug("💾 Saving tip to database: %s @ %.1f%%", suggestion, prob_pct)
                    
                    # PATCH: reuse the existing connection `c`
                    c.execute(
                        "INSERT INTO tips(match_id,league_id,league,home,away,market,suggestion,confidence,confidence_raw,score_at_tip,minute,created_ts,odds,book,ev_pct,sent_ok) "
                        "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,0)",
                        (fid,league_id,league,home,away,market_txt,suggestion,float(prob_pct),raw,score,minute,created_ts,
                         (float(odds) if odds is not None else None), (book or None), (float(ev_pct) if ev_pct is not None else None))
                    )

                    log.debug("📱 Sending tip to Telegram")
                    sent=_send_tip(home,away,league,minute,score,suggestion,float(prob_pct),feat,odds,book,ev_pct)
                    if sent:
                        c.execute("UPDATE tips SET sent_ok=1 WHERE match_id=%s AND created_ts=%s",(fid,created_ts))
                        log.debug("✅ Tip sent successfully")
                    else:
                        log.warning("⚠️ Failed to send tip to Telegram")

                    saved+=1; per_match+=1
                    log.info("✅ Tip saved: %s vs %s - %s @ %.1f%%", home, away, suggestion, prob_pct)
                    
                    if MAX_TIPS_PER_SCAN and saved>=MAX_TIPS_PER_SCAN: 
                        log.info("⏹️ Max tips per scan reached: %s", MAX_TIPS_PER_SCAN)
                        break
                
                if MAX_TIPS_PER_SCAN and saved>=MAX_TIPS_PER_SCAN: 
                    log.info("⏹️ Stopping scan after reaching max tips")
                    break
                    
            except Exception as e:
                log.exception("❌ Error processing match: %s", e)
                continue
    
    elapsed = time.time() - start_time
    log.info("✅ Production scan completed in %.2f seconds: saved=%d, live_seen=%d", elapsed, saved, live_seen)
    return saved, live_seen

# ───────── Prematch (compact: save-only, thresholds respected) ─────────
def extract_prematch_features(fx: dict) -> Dict[str,float]:
    log.debug("📊 Extracting prematch features")
    teams=fx.get("teams") or {}; th=(teams.get("home") or {}).get("id"); ta=(teams.get("away") or {}).get("id")
    if not th or not ta: 
        log.debug("❌ Missing team IDs for prematch features")
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
        if played==0: 
            log.debug("📊 No played games in sample")
            return 0,0,0
        log.debug("📊 Rate calculation: played=%s, ov25=%s, ov35=%s, btts=%s", played, ov25, ov35, btts)
        return ov25/played, ov35/played, btts/played
    
    log.debug("📊 Fetching past fixtures for teams")
    last_h=_api_last_fixtures(th,5); last_a=_api_last_fixtures(ta,5); h2h=_api_h2h(th,ta,5)
    
    ov25_h,ov35_h,btts_h=_rate(last_h); ov25_a,ov35_a,btts_a=_rate(last_a); ov25_h2h,ov35_h2h,btts_h2h=_rate(h2h)
    
    features = {
        "pm_ov25_h":ov25_h,"pm_ov35_h":ov35_h,"pm_btts_h":btts_h,
        "pm_ov25_a":ov25_a,"pm_ov35_a":ov35_a,"pm_btts_a":btts_a,
        "pm_ov25_h2h":ov25_h2h,"pm_ov35_h2h":ov35_h2h,"pm_btts_h2h":btts_h2h,
        "minute":0.0,"goals_h":0.0,"goals_a":0.0,"goals_sum":0.0,"goals_diff":0.0,
        "xg_h":0.0,"xg_a":0.0,"xg_sum":0.0,"xg_diff":0.0,"sot_h":0.0,"sot_a":0.0,"sot_sum":0.0,
        "cor_h":0.0,"cor_a":0.0,"cor_sum":0.0,"pos_h":0.0,"pos_a":0.0,"pos_diff":0.0,
        "red_h":0.0,"red_a":0.0,"red_sum":0.0
    }
    
    log.debug("✅ Prematch features extracted: %s", {k: round(v, 2) for k, v in features.items()})
    return features

def _kickoff_berlin(utc_iso: str|None) -> str:
    try:
        if not utc_iso: 
            log.debug("⏰ No UTC time provided")
            return "TBD"
        dt=datetime.fromisoformat(utc_iso.replace("Z","+00:00"))
        result = dt.astimezone(BERLIN_TZ).strftime("%H:%M")
        log.debug("⏰ Converted UTC %s to Berlin %s", utc_iso, result)
        return result
    except Exception as e:
        log.debug("⏰ Failed to parse time '%s': %s", utc_iso, e)
        return "TBD"

def _format_motd_message(home, away, league, kickoff_txt, suggestion, prob_pct, odds=None, book=None, ev_pct=None):
    log.debug("✍️ Formatting MOTD message")
    money = ""
    if odds:
        if ev_pct is not None:
            money = f"\n💰 <b>Odds:</b> {odds:.2f} @ {book or 'Book'}  •  <b>EV:</b> {ev_pct:+.1f}%"
        else:
            money = f"\n💰 <b>Odds:</b> {odds:.2f} @ {book or 'Book'}"
    
    message = (
        "🏅 <b>Match of the Day</b>\n"
        f"<b>Match:</b> {escape(home)} vs {escape(away)}\n"
        f"🏆 <b>League:</b> {escape(league)}\n"
        f"⏰ <b>Kickoff (Berlin):</b> {kickoff_txt}\n"
        f"<b>Tip:</b> {escape(suggestion)}\n"
        f"📈 <b>Confidence:</b> {prob_pct:.1f}%{money}"
    )
    
    log.debug("✅ MOTD message formatted (length: %s chars)", len(message))
    return message

def _send_tip(home,away,league,minute,score,suggestion,prob_pct,feat,odds=None,book=None,ev_pct=None)->bool:
    log.debug("📱 Sending tip via Telegram")
    message = _format_tip_message(home,away,league,minute,score,suggestion,prob_pct,feat,odds,book,ev_pct)
    result = send_telegram(message)
    log.debug("📱 Tip send result: %s", "SUCCESS" if result else "FAILED")
    return result

def prematch_scan_save() -> int:
    log.info("🌅 Starting prematch scan")
    start_time = time.time()
    
    fixtures=_collect_todays_prematch_fixtures(); 
    log.info("📅 Found %s prematch fixtures", len(fixtures))
    
    if not fixtures: 
        log.info("✅ Prematch scan complete: no fixtures")
        return 0
    
    saved=0
    for i, fx in enumerate(fixtures):
        try:
            log.debug("🔍 Processing prematch fixture %s/%s", i+1, len(fixtures))
            fixture=fx.get("fixture") or {}; lg=fx.get("league") or {}; teams=fx.get("teams") or {}
            home=(teams.get("home") or {}).get("name",""); away=(teams.get("away") or {}).get("name","")
            league_id=int((lg.get("id") or 0)); league=f"{lg.get('country','')} - {lg.get('name','')}".strip(" -"); fid=int((fixture.get("id") or 0))
            
            log.debug("🏆 %s vs %s (%s)", home, away, league)
            
            feat=extract_prematch_features(fx); 
            if not fid or not feat: 
                log.debug("⏭️ Skipping fixture without features")
                continue
            
            candidates: List[Tuple[str,str,float]]=[]
            
            # PRE OU via PRE_OU_* models
            log.debug("🎯 Running prematch OU predictions")
            for line in OU_LINES:
                mdl=load_model_from_settings(f"PRE_OU_{_fmt_line(line)}")
                if not mdl: 
                    log.debug("⏭️ No prematch OU model for line %s", line)
                    continue
                
                p=_score_prob(feat, mdl); mk=f"Over/Under {_fmt_line(line)}"; thr=_get_market_threshold_pre(mk)
                
                if p*100.0>=thr:   
                    candidates.append((f"PRE {mk}", f"Over {_fmt_line(line)} Goals", p))
                    log.debug("✅ Prematch Over %s candidate: prob=%.1f%%, threshold=%.1f%%", line, p*100, thr)
                
                q=1.0-p
                if q*100.0>=thr:   
                    candidates.append((f"PRE {mk}", f"Under {_fmt_line(line)} Goals", q))
                    log.debug("✅ Prematch Under %s candidate: prob=%.1f%%, threshold=%.1f%%", line, q*100, thr)
            
            # PRE BTTS
            log.debug("🎯 Running prematch BTTS predictions")
            mdl=load_model_from_settings("PRE_BTTS_YES")
            if mdl:
                p=_score_prob(feat, mdl); thr=_get_market_threshold_pre("BTTS")
                if p*100.0>=thr: 
                    candidates.append(("PRE BTTS","BTTS: Yes",p))
                    log.debug("✅ Prematch BTTS: Yes candidate: prob=%.1f%%, threshold=%.1f%%", p*100, thr)
                
                q=1.0-p
                if q*100.0>=thr: 
                    candidates.append(("PRE BTTS","BTTS: No",q))
                    log.debug("✅ Prematch BTTS: No candidate: prob=%.1f%%, threshold=%.1f%%", q*100, thr)
            
            # PRE 1X2 (draw suppressed)
            log.debug("🎯 Running prematch 1X2 predictions")
            mh,ma=load_model_from_settings("PRE_WLD_HOME"), load_model_from_settings("PRE_WLD_AWAY")
            if mh and ma:
                ph=_score_prob(feat,mh); pa=_score_prob(feat,ma); s=max(EPS,ph+pa); ph,pa=ph/s,pa/s
                thr=_get_market_threshold_pre("1X2")
                
                if ph*100.0>=thr: 
                    candidates.append(("PRE 1X2","Home Win",ph))
                    log.debug("✅ Prematch Home Win candidate: prob=%.1f%%, threshold=%.1f%%", ph*100, thr)
                
                if pa*100.0>=thr: 
                    candidates.append(("PRE 1X2","Away Win",pa))
                    log.debug("✅ Prematch Away Win candidate: prob=%.1f%%, threshold=%.1f%%", pa*100, thr)
            
            if not candidates: 
                log.debug("⏭️ No candidates for this fixture")
                continue
            
            candidates.sort(key=lambda x:x[2], reverse=True)
            log.debug("📊 Prematch candidates found: %s", [(c[1], round(c[2]*100,1)) for c in candidates])
            
            base_now=int(time.time()); per_match=0
            for idx,(mk,sug,prob) in enumerate(candidates):
                if sug not in ALLOWED_SUGGESTIONS: 
                    log.debug("⏭️ Skipping non-allowed suggestion: %s", sug)
                    continue
                
                if per_match>=max(1,PREDICTIONS_PER_MATCH): 
                    log.debug("⏭️ Max predictions per match reached: %s", PREDICTIONS_PER_MATCH)
                    break
                
                # Odds/EV gate
                log.debug("💰 Applying price/EV gate for prematch: %s", sug)
                pass_odds, odds, book, _ = _price_gate(mk.replace("PRE ",""), sug, fid)
                if not pass_odds: 
                    log.debug("⏭️ Failed price gate for prematch: %s", sug)
                    continue
                
                ev_pct=None
                if odds is not None:
                    edge=_ev(prob, odds); ev_pct=round(edge*100.0,1)
                    ev_bps = int(round(edge*10000))
                    if ev_bps < EDGE_MIN_BPS: 
                        log.debug("⏭️ EV too low for prematch: %s bps < minimum %s bps", ev_bps, EDGE_MIN_BPS)
                        continue
                    log.debug("✅ Prematch EV passed: %s bps >= minimum %s bps", ev_bps, EDGE_MIN_BPS)
                
                created_ts=base_now+idx; raw=float(prob); pct=round(raw*100.0,1)
                
                log.debug("💾 Saving prematch tip to database: %s @ %.1f%%", sug, pct)
                with db_conn() as c2:
                    c2.execute("INSERT INTO tips(match_id,league_id,league,home,away,market,suggestion,confidence,confidence_raw,score_at_tip,minute,created_ts,odds,book,ev_pct,sent_ok) "
                               "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,'0-0',0,%s,%s,%s,%s,0)",
                               (fid,league_id,league,home,away,mk,sug,float(pct),raw,created_ts,
                                (float(odds) if odds is not None else None), (book or None), (float(ev_pct) if ev_pct is not None else None)))
                
                saved+=1; per_match+=1
                log.info("✅ Prematch tip saved: %s vs %s - %s @ %.1f%%", home, away, sug, pct)
        
        except Exception as e:
            log.exception("❌ Error processing prematch fixture: %s", e)
            continue
    
    elapsed = time.time() - start_time
    log.info("✅ Prematch scan completed in %.2f seconds: saved=%d", elapsed, saved)
    return saved

# ───────── Auto-train / tune / retry (unchanged signatures) ─────────
def auto_train_job():
    log.info("🤖 Starting auto-train job")
    if not TRAIN_ENABLE: 
        log.info("⏭️ Training disabled (TRAIN_ENABLE=0)")
        send_telegram("🤖 Training skipped: TRAIN_ENABLE=0"); 
        return
    
    send_telegram("🤖 Training started.")
    try:
        log.info("🤖 Calling train_models function")
        res=train_models() or {}; ok=bool(res.get("ok"))
        if not ok:
            reason=res.get("reason") or res.get("error") or "unknown"
            log.warning("🤖 Training failed: %s", reason)
            send_telegram(f"⚠️ Training finished: <b>SKIPPED</b>\nReason: {escape(str(reason))}"); 
            return
        
        trained=[k for k,v in (res.get("trained") or {}).items() if v]
        thr=(res.get("thresholds") or {}); mets=(res.get("metrics") or {})
        
        log.info("🤖 Training successful: trained %s models", len(trained))
        lines=["🤖 <b>Model training OK</b>"]
        if trained: 
            lines.append("• Trained: " + ", ".join(sorted(trained)))
            log.debug("🤖 Trained models: %s", trained)
        
        if thr: 
            lines.append("• Thresholds: " + "  |  ".join([f"{escape(k)}: {float(v):.1f}%" for k,v in thr.items()]))
            log.debug("🤖 New thresholds: %s", thr)
        
        send_telegram("\n".join(lines))
        log.info("✅ Auto-train job completed successfully")
        
    except Exception as e:
        log.exception("❌ Training job failed: %s", e); 
        send_telegram(f"❌ Training <b>FAILED</b>\n{escape(str(e))}")

def _pick_threshold(y_true,y_prob,target_precision,min_preds,default_pct):
    log.debug("🎯 Picking optimal threshold")
    import numpy as np
    y=np.asarray(y_true,dtype=int); p=np.asarray(y_prob,dtype=float)
    best=default_pct/100.0
    
    log.debug("🎯 Data: %s samples, target precision: %s, min predictions: %s", len(y), target_precision, min_preds)
    
    for t in np.arange(MIN_THRESH,MAX_THRESH+1e-9,1.0)/100.0:
        pred=(p>=t).astype(int); n=int(pred.sum())
        if n<min_preds: 
            log.debug("🎯 Threshold %.3f: only %s predictions (< %s)", t, n, min_preds)
            continue
        
        tp=int(((pred==1)&(y==1)).sum()); prec=tp/max(1,n)
        log.debug("🎯 Threshold %.3f: %s predictions, %s true positives, precision: %.3f", t, n, tp, prec)
        
        if prec>=target_precision: 
            best=float(t)
            log.debug("🎯 Found threshold %.3f meeting target precision %.3f", best, target_precision)
            break
    
    result = best*100.0
    log.debug("🎯 Selected threshold: %.1f%%", result)
    return result

# Optional min EV for MOTD (basis points, e.g. 300 = +3.00%). 0 disables EV gate.
MOTD_MIN_EV_BPS = int(os.getenv("MOTD_MIN_EV_BPS", "0"))
log.info("🏅 MOTD minimum EV: %s bps", MOTD_MIN_EV_BPS)

def send_match_of_the_day() -> bool:
    """Pick the single best prematch tip for today (PRE_* models). Sends to Telegram."""
    log.info("🏅 Starting Match of the Day selection")
    
    fixtures = _collect_todays_prematch_fixtures()
    if not fixtures:
        log.info("🏅 No fixtures found for today")
        return send_telegram("🏅 Match of the Day: no eligible fixtures today.")

    # Optional league allow-list just for MOTD
    if MOTD_LEAGUE_IDS:
        log.info("🏅 Filtering fixtures by league IDs: %s", MOTD_LEAGUE_IDS)
        fixtures = [
            f for f in fixtures
            if int(((f.get("league") or {}).get("id") or 0)) in MOTD_LEAGUE_IDS
        ]
        log.info("🏅 After league filtering: %s fixtures", len(fixtures))
        
        if not fixtures:
            log.info("🏅 No fixtures in configured leagues")
            return send_telegram("🏅 Match of the Day: no fixtures in configured leagues.")

    best = None  # (prob_pct, suggestion, home, away, league, kickoff_txt, odds, book, ev_pct)
    log.debug("🏅 Evaluating %s fixtures for MOTD", len(fixtures))

    for i, fx in enumerate(fixtures):
        log.debug("🏅 Processing fixture %s/%s for MOTD", i+1, len(fixtures))
        fixture = fx.get("fixture") or {}
        lg      = fx.get("league") or {}
        teams   = fx.get("teams") or {}
        fid     = int((fixture.get("id") or 0))

        home = (teams.get("home") or {}).get("name","")
        away = (teams.get("away") or {}).get("name","")
        league = f"{lg.get('country','')} - {lg.get('name','')}".strip(" -")
        kickoff_txt = _kickoff_berlin((fixture.get("date") or ""))

        log.debug("🏅 Evaluating: %s vs %s (%s)", home, away, league)

        feat = extract_prematch_features(fx)
        if not feat:
            log.debug("⏭️ No features for %s vs %s", home, away)
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
            s = max(EPS, ph+pa); ph, pa = ph/s, pa/s
            thr = _get_market_threshold_pre("1X2")
            if ph*100.0 >= thr: candidates.append(("1X2","Home Win", ph))
            if pa*100.0 >= thr: candidates.append(("1X2","Away Win", pa))

        if not candidates:
            log.debug("⏭️ No candidates for %s vs %s", home, away)
            continue

        # Take the single best for this fixture (by probability) then apply odds/EV gate
        candidates.sort(key=lambda x: x[2], reverse=True)
        mk, sug, prob = candidates[0]
        prob_pct = prob * 100.0
        
        log.debug("🏅 Best candidate for %s vs %s: %s @ %.1f%%", home, away, sug, prob_pct)
        
        if prob_pct < MOTD_CONF_MIN:
            log.debug("⏭️ Confidence too low: %.1f%% < minimum %.1f%%", prob_pct, MOTD_CONF_MIN)
            continue

        # Odds/EV (reuse in-play price gate; market text must be without "PRE ")
        pass_odds, odds, book, _ = _price_gate(mk, sug, fid)
        if not pass_odds:
            log.debug("⏭️ Failed price gate for %s", sug)
            continue

        ev_pct = None
        if odds is not None:
            edge = _ev(prob, odds)            # decimal (e.g. 0.05)
            ev_bps = int(round(edge * 10000)) # basis points
            ev_pct = round(edge * 100.0, 1)
            if MOTD_MIN_EV_BPS > 0 and ev_bps < MOTD_MIN_EV_BPS:
                log.debug("⏭️ EV too low: %s bps < minimum %s bps", ev_bps, MOTD_MIN_EV_BPS)
                continue
            log.debug("✅ EV acceptable: %s bps", ev_bps)

        item = (prob_pct, sug, home, away, league, kickoff_txt, odds, book, ev_pct)
        if best is None or prob_pct > best[0]:
            best = item
            log.debug("🏅 New best candidate: %s @ %.1f%%", sug, prob_pct)

    if not best:
        log.info("🏅 No suitable MOTD pick found")
        return send_telegram("🏅 Match of the Day: no prematch pick met thresholds.")
    
    prob_pct, sug, home, away, league, kickoff_txt, odds, book, ev_pct = best
    log.info("🏅 Selected MOTD: %s vs %s - %s @ %.1f%%", home, away, sug, prob_pct)
    
    return send_telegram(_format_motd_message(home, away, league, kickoff_txt, sug, prob_pct, odds, book, ev_pct))

def auto_tune_thresholds(days: int = 14) -> Dict[str,float]:
    log.info("🔧 Starting auto-tune thresholds (last %s days)", days)
    if not AUTO_TUNE_ENABLE: 
        log.info("⏭️ Auto-tune disabled")
        return {}
    
    cutoff=int(time.time())-days*24*3600
    log.debug("🔧 Cutoff timestamp: %s (%s days ago)", cutoff, days)
    
    with db_conn() as c:
        rows=c.execute("""
            SELECT t.market, t.suggestion, COALESCE(t.confidence_raw, t.confidence/100.0) prob,
                   r.final_goals_h, r.final_goals_a, r.btts_yes
            FROM tips t JOIN match_results r ON r.match_id=t.match_id
            WHERE t.created_ts >= %s AND t.suggestion<>'HARVEST' AND t.sent_ok=1
        """,(cutoff,)).fetchall()
    
    log.debug("🔧 Found %s tips for threshold tuning", len(rows))
    
    by={}
    for (mk,sugg,prob,gh,ga,btts) in rows:
        out=_tip_outcome_for_result(sugg, {"final_goals_h":gh,"final_goals_a":ga,"btts_yes":btts})
        if out is None: 
            log.debug("🔧 Tip not gradable: %s", sugg)
            continue
        
        by.setdefault(mk, []).append((float(prob), int(out)))
    
    log.debug("🔧 Graded tips by market: %s", {k: len(v) for k, v in by.items()})
    
    tuned={}
    for mk,arr in by.items():
        if len(arr)<THRESH_MIN_PREDICTIONS: 
            log.debug("🔧 Market '%s': insufficient data (%s < %s)", mk, len(arr), THRESH_MIN_PREDICTIONS)
            continue
        
        probs=[p for (p,_) in arr]; wins=[y for (_,y) in arr]
        pct=_pick_threshold(wins, probs, TARGET_PRECISION, THRESH_MIN_PREDICTIONS, CONF_THRESHOLD)
        
        log.info("🔧 Tuning market '%s': new threshold %.1f%% (from %s samples)", mk, pct, len(arr))
        
        set_setting(f"conf_threshold:{mk}", f"{pct:.2f}"); 
        _SETTINGS_CACHE.invalidate(f"conf_threshold:{mk}"); 
        tuned[mk]=pct
    
    if tuned:
        log.info("✅ Auto-tune updated %s thresholds", len(tuned))
        send_telegram("🔧 Auto-tune updated thresholds:\n" + "\n".join([f"• {k}: {v:.1f}%" for k,v in tuned.items()]))
    else: 
        log.info("ℹ️ Auto-tune: no updates (insufficient data)")
        send_telegram("🔧 Auto-tune: no updates (insufficient data).")
    
    return tuned

def retry_unsent_tips(minutes: int = 30, limit: int = 200) -> int:
    log.info("🔄 Retrying unsent tips (last %s minutes, limit %s)", minutes, limit)
    cutoff = int(time.time()) - minutes*60
    retried = 0
    
    with db_conn() as c:
        rows = c.execute(
            "SELECT match_id,league,home,away,market,suggestion,confidence,confidence_raw,score_at_tip,minute,created_ts,odds,book,ev_pct "
            "FROM tips WHERE sent_ok=0 AND created_ts >= %s ORDER BY created_ts ASC LIMIT %s",
            (cutoff, limit)
        ).fetchall()

        log.debug("🔄 Found %s unsent tips to retry", len(rows))
        
        for (mid, league, home, away, market, sugg, conf, conf_raw, score, minute, cts, odds, book, ev_pct) in rows:
            log.debug("🔄 Retrying tip: %s vs %s - %s", home, away, sugg)
            ok = send_telegram(_format_tip_message(home, away, league, int(minute), score, sugg, float(conf), {}, odds, book, ev_pct))
            if ok:
                c.execute("UPDATE tips SET sent_ok=1 WHERE match_id=%s AND created_ts=%s", (mid, cts))
                retried += 1
                log.debug("✅ Successfully resent tip for %s vs %s", home, away)
            else:
                log.warning("⚠️ Failed to resend tip for %s vs %s", home, away)
    
    if retried:
        log.info("✅ Retry completed: %s tips resent", retried)
    else:
        log.info("ℹ️ No tips needed retrying")
    return retried

# ───────── Scheduler ─────────
def _run_with_pg_lock(lock_key: int, fn, *a, **k):
    log.debug("🔒 Attempting to acquire lock %s for function %s", lock_key, fn.__name__)
    try:
        with db_conn() as c:
            got=c.execute("SELECT pg_try_advisory_lock(%s)",(lock_key,)).fetchone()[0]
            if not got: 
                log.info("🔒 Lock %s busy; skipped %s.", lock_key, fn.__name__)
                return None
            
            log.debug("🔒 Lock %s acquired for %s", lock_key, fn.__name__)
            try: 
                result = fn(*a,**k)
                log.debug("🔒 Function %s completed with lock %s", fn.__name__, lock_key)
                return result
            finally: 
                c.execute("SELECT pg_advisory_unlock(%s)",(lock_key,))
                log.debug("🔒 Lock %s released", lock_key)
    except Exception as e:
        log.exception("🔒 Lock %s failed for %s: %s", lock_key, fn.__name__, e)
        return None

_scheduler_started=False
def _start_scheduler_once():
    global _scheduler_started
    if _scheduler_started or not RUN_SCHEDULER: 
        log.info("⏭️ Scheduler already started or disabled")
        return
    
    try:
        log.info("⏰ Starting scheduler")
        sched=BackgroundScheduler(timezone=TZ_UTC)
        
        # Production scan job
        sched.add_job(
            lambda:_run_with_pg_lock(1001,production_scan),
            "interval",
            seconds=SCAN_INTERVAL_SEC,
            id="scan",
            max_instances=1,
            coalesce=True
        )
        log.info("⏰ Scheduled scan job every %s seconds", SCAN_INTERVAL_SEC)
        
        # Backfill job
        sched.add_job(
            lambda:_run_with_pg_lock(1002,backfill_results_for_open_matches,400),
            "interval",
            minutes=BACKFILL_EVERY_MIN,
            id="backfill",
            max_instances=1,
            coalesce=True
        )
        log.info("⏰ Scheduled backfill job every %s minutes", BACKFILL_EVERY_MIN)
        
        # Daily accuracy digest
        if DAILY_ACCURACY_DIGEST_ENABLE:
            sched.add_job(
                lambda:_run_with_pg_lock(1003,daily_accuracy_digest),
                CronTrigger(hour=DAILY_ACCURACY_HOUR, minute=DAILY_ACCURACY_MINUTE, timezone=BERLIN_TZ),
                id="digest", 
                max_instances=1, 
                coalesce=True
            )
            log.info("⏰ Scheduled daily digest at %02d:%02d Berlin time", DAILY_ACCURACY_HOUR, DAILY_ACCURACY_MINUTE)
        
        # MOTD job
        if MOTD_PREDICT:
            sched.add_job(
                lambda:_run_with_pg_lock(1004,send_match_of_the_day),
                CronTrigger(hour=int(os.getenv("MOTD_HOUR","19")), minute=int(os.getenv("MOTD_MINUTE","15")), timezone=BERLIN_TZ),
                id="motd", 
                max_instances=1, 
                coalesce=True
            )
            log.info("⏰ Scheduled MOTD at %02d:%02d Berlin time", int(os.getenv("MOTD_HOUR","19")), int(os.getenv("MOTD_MINUTE","15")))
        
        # Training job
        if TRAIN_ENABLE:
            sched.add_job(
                lambda:_run_with_pg_lock(1005,auto_train_job),
                CronTrigger(hour=TRAIN_HOUR_UTC, minute=TRAIN_MINUTE_UTC, timezone=TZ_UTC),
                id="train", 
                max_instances=1, 
                coalesce=True
            )
            log.info("⏰ Scheduled training at %02d:%02d UTC", TRAIN_HOUR_UTC, TRAIN_MINUTE_UTC)
        
        # Auto-tune job
        if AUTO_TUNE_ENABLE:
            sched.add_job(
                lambda:_run_with_pg_lock(1006,auto_tune_thresholds,14),
                CronTrigger(hour=4, minute=7, timezone=TZ_UTC),
                id="auto_tune", 
                max_instances=1, 
                coalesce=True
            )
            log.info("⏰ Scheduled auto-tune at 04:07 UTC")
        
        # Retry unsent tips job
        sched.add_job(
            lambda:_run_with_pg_lock(1007,retry_unsent_tips,30,200),
            "interval",
            minutes=10,
            id="retry",
            max_instances=1,
            coalesce=True
        )
        log.info("⏰ Scheduled retry job every 10 minutes")
        
        sched.start(); 
        _scheduler_started=True
        send_telegram("🚀 goalsniper AI mode (in-play + prematch) started.")
        log.info("✅ Scheduler started with %s jobs", len(sched.get_jobs()))
        
    except Exception as e:
        log.exception("❌ Scheduler failed to start: %s", e)

_start_scheduler_once()

# ───────── Admin / auth ─────────
def _require_admin():
    log.debug("🔒 Checking admin authentication")
    key=request.headers.get("X-API-Key") or request.args.get("key") or ((request.json or {}).get("key") if request.is_json else None)
    
    if not ADMIN_API_KEY or key != ADMIN_API_KEY: 
        log.warning("🔒 Admin authentication failed for key: %s", key)
        abort(401)
    
    log.debug("✅ Admin authentication successful")

# ───────── HTTP endpoints ─────────
@app.route("/")
def root(): 
    log.info("🌐 Root endpoint accessed")
    return jsonify({"ok": True, "name": "goalsniper", "mode": "FULL_AI", "scheduler": RUN_SCHEDULER})

@app.route("/health")
def health():
    log.info("🏥 Health check requested")
    try:
        with db_conn() as c:
            n=c.execute("SELECT COUNT(*) FROM tips").fetchone()[0]
        log.info("✅ Health check passed: DB ok, %s tips", n)
        return jsonify({"ok": True, "db": "ok", "tips_count": int(n)})
    except Exception as e:
        log.error("❌ Health check failed: %s", e)
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/init-db", methods=["POST"])
def http_init_db(): 
    log.info("🗄️ Admin requested database initialization")
    _require_admin(); 
    init_db(); 
    log.info("✅ Database initialization completed via HTTP")
    return jsonify({"ok": True})

@app.route("/admin/scan", methods=["POST","GET"])
def http_scan(): 
    log.info("🚀 Admin requested manual scan")
    _require_admin(); 
    s,l=production_scan(); 
    log.info("✅ Manual scan completed: saved=%s, live_seen=%s", s, l)
    return jsonify({"ok": True, "saved": s, "live_seen": l})

@app.route("/admin/backfill-results", methods=["POST","GET"])
def http_backfill(): 
    log.info("🔄 Admin requested backfill")
    _require_admin(); 
    n=backfill_results_for_open_matches(400); 
    log.info("✅ Backfill completed: updated=%s", n)
    return jsonify({"ok": True, "updated": n})

@app.route("/admin/train", methods=["POST","GET"])
def http_train():
    log.info("🤖 Admin requested manual training")
    _require_admin()
    if not TRAIN_ENABLE: 
        log.warning("⚠️ Training disabled")
        return jsonify({"ok": False, "reason": "training disabled"}), 400
    
    try: 
        out=train_models(); 
        log.info("✅ Manual training completed")
        return jsonify({"ok": True, "result": out})
    except Exception as e:
        log.exception("❌ Manual training failed: %s", e)
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/admin/train-notify", methods=["POST","GET"])
def http_train_notify(): 
    log.info("🤖 Admin requested training notification")
    _require_admin(); 
    auto_train_job(); 
    log.info("✅ Training notification sent")
    return jsonify({"ok": True})

@app.route("/admin/digest", methods=["POST","GET"])
def http_digest(): 
    log.info("📊 Admin requested daily digest")
    _require_admin(); 
    msg=daily_accuracy_digest(); 
    log.info("✅ Daily digest sent: %s", bool(msg))
    return jsonify({"ok": True, "sent": bool(msg)})

@app.route("/admin/auto-tune", methods=["POST","GET"])
def http_auto_tune(): 
    log.info("🔧 Admin requested auto-tune")
    _require_admin(); 
    tuned=auto_tune_thresholds(14); 
    log.info("✅ Auto-tune completed: %s thresholds tuned", len(tuned))
    return jsonify({"ok": True, "tuned": tuned})

@app.route("/admin/retry-unsent", methods=["POST","GET"])
def http_retry_unsent(): 
    log.info("🔄 Admin requested retry unsent")
    _require_admin(); 
    n=retry_unsent_tips(30,200); 
    log.info("✅ Retry unsent completed: %s tips resent", n)
    return jsonify({"ok": True, "resent": n})

@app.route("/admin/prematch-scan", methods=["POST","GET"])
def http_prematch_scan(): 
    log.info("🌅 Admin requested prematch scan")
    _require_admin(); 
    saved=prematch_scan_save(); 
    log.info("✅ Prematch scan completed: %s tips saved", saved)
    return jsonify({"ok": True, "saved": int(saved)})

@app.route("/admin/motd", methods=["POST","GET"])
def http_motd():
    log.info("🏅 Admin requested MOTD")
    _require_admin(); 
    ok = send_match_of_the_day(); 
    log.info("✅ MOTD sent: %s", ok)
    return jsonify({"ok": bool(ok)})

@app.route("/settings/<key>", methods=["GET","POST"])
def http_settings(key: str):
    log.info("⚙️ Settings endpoint accessed for key: %s", key)
    _require_admin()
    
    if request.method=="GET":
        val=get_setting_cached(key); 
        log.info("⚙️ GET setting %s: %s", key, val)
        return jsonify({"ok": True, "key": key, "value": val})
    
    val=(request.get_json(silent=True) or {}).get("value")
    if val is None: 
        log.warning("⚠️ No value provided for setting %s", key)
        abort(400)
    
    log.info("⚙️ SET setting %s=%s", key, val)
    set_setting(key, str(val)); 
    _SETTINGS_CACHE.invalidate(key); 
    invalidate_model_caches_for_key(key)
    
    log.info("✅ Setting updated and caches invalidated")
    return jsonify({"ok": True})

@app.route("/tips/latest")
def http_latest():
    limit=int(request.args.get("limit","50"))
    log.info("📋 Latest tips requested (limit: %s)", limit)
    
    with db_conn() as c:
        rows=c.execute("SELECT match_id,league,home,away,market,suggestion,confidence,confidence_raw,score_at_tip,minute,created_ts,odds,book,ev_pct "
                       "FROM tips WHERE suggestion<>'HARVEST' ORDER BY created_ts DESC LIMIT %s",(max(1,min(500,limit)),)).fetchall()
    
    tips=[]
    for r in rows:
        tips.append({
            "match_id":int(r[0]),
            "league":r[1],
            "home":r[2],
            "away":r[3],
            "market":r[4],
            "suggestion":r[5],
            "confidence":float(r[6]),
            "confidence_raw":(float(r[7]) if r[7] is not None else None),
            "score_at_tip":r[8],
            "minute":int(r[9]),
            "created_ts":int(r[10]),
            "odds": (float(r[11]) if r[11] is not None else None), 
            "book": r[12], 
            "ev_pct": (float(r[13]) if r[13] is not None else None)
        })
    
    log.info("✅ Returned %s latest tips", len(tips))
    return jsonify({"ok": True, "tips": tips})

@app.route("/telegram/webhook/<secret>", methods=["POST"])
def telegram_webhook(secret: str):
    log.info("📱 Telegram webhook received")
    if (WEBHOOK_SECRET or "") != secret: 
        log.warning("🔒 Telegram webhook secret mismatch")
        abort(403)
    
    update=request.get_json(silent=True) or {}
    log.debug("📱 Webhook update: %s", json.dumps(update)[:200])
    
    try:
        msg=(update.get("message") or {}).get("text") or ""
        log.info("📱 Telegram message: %s", msg)
        
        if msg.startswith("/start"): 
            send_telegram("👋 goalsniper bot (FULL AI mode) is online.")
            log.info("✅ Responded to /start command")
        elif msg.startswith("/digest"): 
            daily_accuracy_digest()
            log.info("✅ Responded to /digest command")
        elif msg.startswith("/motd"): 
            send_match_of_the_day()
            log.info("✅ Responded to /motd command")
        elif msg.startswith("/scan"):
            parts=msg.split()
            if len(parts)>1 and ADMIN_API_KEY and parts[1]==ADMIN_API_KEY:
                s,l=production_scan(); 
                send_telegram(f"🔁 Scan done. Saved: {s}, Live seen: {l}")
                log.info("✅ Responded to /scan command")
            else: 
                send_telegram("🔒 Admin key required.")
                log.warning("⚠️ Unauthorized /scan attempt")
    except Exception as e:
        log.warning("❌ Telegram webhook parse error: %s", e)
    
    return jsonify({"ok": True})

# ───────── Boot ─────────
def _on_boot():
    log.info("🚀 Starting application boot sequence")
    start_time = time.time()
    
    _init_pool(); 
    log.info("✅ Database pool initialized")
    
    init_db(); 
    log.info("✅ Database schema initialized")
    
    set_setting("boot_ts", str(int(time.time())))
    log.info("✅ Boot timestamp set")
    
    elapsed = time.time() - start_time
    log.info("✅ Application boot completed in %.2f seconds", elapsed)

_on_boot()

if __name__ == "__main__":
    host = os.getenv("HOST","0.0.0.0")
    port = int(os.getenv("PORT","8080"))
    log.info("🌍 Starting Flask application on %s:%s", host, port)
    app.run(host=host, port=port)
