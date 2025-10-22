import os, json, math, time, logging, requests, psycopg2, sys, signal, atexit
import numpy as np
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
from functools import lru_cache
from collections import OrderedDict, deque
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import random

# ───────── Env bootstrap ─────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ───────── Optional production add-ons ─────────
SENTRY_DSN = os.getenv("SENTRY_DSN")
if SENTRY_DSN:
    try:
        import sentry_sdk  # type: ignore
        sentry_sdk.init(
            dsn=SENTRY_DSN,
            traces_sample_rate=float(os.getenv("SENTRY_TRACES", "0.0")),
        )
    except Exception:
        # Sentry is optional; keep running if it fails
        pass

REDIS_URL = os.getenv("REDIS_URL")
_redis = None
if REDIS_URL:
    try:
        import redis  # type: ignore
        _redis = redis.Redis.from_url(
            REDIS_URL, socket_timeout=1, socket_connect_timeout=1
        )
    except Exception:
        _redis = None  # fallback to in-memory TTL caches

# ───────── Shutdown Manager ─────────
class ShutdownManager:
    _shutdown_requested = False
    
    @classmethod
    def is_shutdown_requested(cls):
        return cls._shutdown_requested
    
    @classmethod
    def request_shutdown(cls):
        cls._shutdown_requested = True

# ───────── App / logging ─────────
class CustomFormatter(logging.Formatter):
    def format(self, record):
        if not hasattr(record, 'job_id'):
            record.job_id = 'main'
        return super().format(record)

handler = logging.StreamHandler()
formatter = CustomFormatter("[%(asctime)s] %(levelname)s [%(job_id)s] - %(message)s")
handler.setFormatter(formatter)
log = logging.getLogger("goalsniper")
log.handlers = [handler]
log.setLevel(logging.INFO)
log.propagate = False

app = Flask(__name__)

# ───────── Minimal Prometheus-style metrics ─────────
from collections import defaultdict
METRICS = {
    "api_calls_total": defaultdict(int),      # label: endpoint key
    "api_rate_limited_total": 0,              # 429s
    "tips_generated_total": 0,
    "tips_sent_total": 0,
    "db_errors_total": 0,
    "job_duration_seconds": defaultdict(list) # recent durations per job
}
def _metric_inc(name: str, label: Optional[str] = None, n: int = 1) -> None:
    try:
        if label is None:
            if isinstance(METRICS.get(name), int):
                METRICS[name] += n  # type: ignore
            else:
                METRICS[name][None] += n
        else:
            METRICS[name][label] += n
    except Exception:
        pass

def _metric_obs_duration(job: str, t0: float) -> None:
    try:
        arr = METRICS["job_duration_seconds"][job]
        arr.append(time.time() - t0)
        if len(arr) > 50:
            METRICS["job_duration_seconds"][job] = arr[-50:]
    except Exception:
        pass

# ───────── Required envs (fail fast) — ADDED ─────────
def _require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise SystemExit(f"Missing required environment variable: {name}")
    return v

# ───────── Core env (secrets: required; knobs: defaultable) — UPDATED DEFAULTS FOR PRECISION ─────────
TELEGRAM_BOT_TOKEN = _require_env("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = _require_env("TELEGRAM_CHAT_ID")
API_KEY            = _require_env("API_KEY")
ADMIN_API_KEY      = os.getenv("ADMIN_API_KEY")
WEBHOOK_SECRET     = os.getenv("TELEGRAM_WEBHOOK_SECRET")
RUN_SCHEDULER      = os.getenv("RUN_SCHEDULER", "1") not in ("0","false","False","no","NO")

# Precision-related knobs — hardened defaults
CONF_THRESHOLD     = float(os.getenv("CONF_THRESHOLD", "75"))  # was 70
MAX_TIPS_PER_SCAN  = int(os.getenv("MAX_TIPS_PER_SCAN", "25"))
DUP_COOLDOWN_MIN   = int(os.getenv("DUP_COOLDOWN_MIN", "20"))
TIP_MIN_MINUTE     = int(os.getenv("TIP_MIN_MINUTE", "12"))   # was 8
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

# ✅ Standardize & fix auto-tune flag/name
AUTO_TUNE_ENABLE        = os.getenv("AUTO_TUNE_ENABLE", "0") not in ("0","false","False","no","NO")
TARGET_PRECISION        = float(os.getenv("TARGET_PRECISION", "0.60"))
THRESH_MIN_PREDICTIONS  = int(os.getenv("THRESH_MIN_PREDICTIONS", "25"))
MIN_THRESH              = float(os.getenv("MIN_THRESH", "55"))
MAX_THRESH              = float(os.getenv("MAX_THRESH", "85"))

STALE_GUARD_ENABLE = os.getenv("STALE_GUARD_ENABLE", "1") not in ("0","false","False","no","NO")
STALE_STATS_MAX_SEC = int(os.getenv("STALE_STATS_MAX_SEC", "240"))
MARKET_CUTOFFS_RAW = os.getenv("MARKET_CUTOFFS", "BTTS=75,1X2=80,OU=88")
TIP_MAX_MINUTE_ENV = os.getenv("TIP_MAX_MINUTE", "")

MOTD_PREMATCH_ENABLE    = os.getenv("MOTD_PREMATCH_ENABLE", "1") not in ("0","false","False","no","NO")
MOTD_PREDICT            = os.getenv("MOTD_PREDICT", "1") not in ("0","false","False","no","NO")
MOTD_HOUR               = int(os.getenv("MOTD_HOUR", "19"))
MOTD_MINUTE             = int(os.getenv("MOTD_MINUTE", "15"))
MOTD_CONF_MIN           = float(os.getenv("MOTD_CONF_MIN", "70"))
try:
    MOTD_LEAGUE_IDS = [int(x) for x in (os.getenv("MOTD_LEAGUE_IDS","").split(",")) if x.strip().isdigit()]
except Exception:
    MOTD_LEAGUE_IDS = []

# Optional-but-recommended warnings — ADDED
if not ADMIN_API_KEY:
    log.warning("ADMIN_API_KEY is not set — /admin/* endpoints are less protected.")
if not WEBHOOK_SECRET:
    log.warning("TELEGRAM_WEBHOOK_SECRET is not set — /telegram/webhook/<secret> would be unsafe if exposed.")

# ───────── Configuration Validation ─────────
def validate_config():
    """Validate critical configuration at startup"""
    required = {
        'TELEGRAM_BOT_TOKEN': TELEGRAM_BOT_TOKEN,
        'TELEGRAM_CHAT_ID': TELEGRAM_CHAT_ID,
        'API_KEY': API_KEY,
        'DATABASE_URL': DATABASE_URL
    }
    
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise SystemExit(f"Missing required config: {missing}")
    
    # Validate numeric ranges
    if not (0 <= CONF_THRESHOLD <= 100):
        log.warning("CONF_THRESHOLD should be 0-100, got %s", CONF_THRESHOLD)
    
    if SCAN_INTERVAL_SEC < 30:
        log.warning("SCAN_INTERVAL_SEC very low: %s", SCAN_INTERVAL_SEC)
    
    log.info("[CONFIG] Configuration validation passed")

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
PREDICTIONS_PER_MATCH = int(os.getenv("PREDICTIONS_PER_MATCH", "1"))  # was 2 — tighter by default
PER_LEAGUE_CAP        = int(os.getenv("PER_LEAGUE_CAP", "2"))         # was 0 — cap league dominance by default

# ───────── Odds/EV controls — UPDATED DEFAULTS ─────────
MIN_ODDS_OU   = float(os.getenv("MIN_ODDS_OU", "1.50"))  # was 1.30
MIN_ODDS_BTTS = float(os.getenv("MIN_ODDS_BTTS", "1.50"))  # was 1.30
MIN_ODDS_1X2  = float(os.getenv("MIN_ODDS_1X2", "1.50"))  # was 1.30
MAX_ODDS_ALL  = float(os.getenv("MAX_ODDS_ALL", "20.0"))
EDGE_MIN_BPS  = int(os.getenv("EDGE_MIN_BPS", "600"))      # was 300 bps
ODDS_BOOKMAKER_ID = os.getenv("ODDS_BOOKMAKER_ID")  # optional API-Football book id
ALLOW_TIPS_WITHOUT_ODDS = os.getenv("ALLOW_TIPS_WITHOUT_ODDS","0") not in ("0","false","False","no","NO")  # default hardened to 0

# Aggregated odds controls (new)
ODDS_SOURCE = os.getenv("ODDS_SOURCE", "auto").lower()            # auto|live|prematch
ODDS_AGGREGATION = os.getenv("ODDS_AGGREGATION", "median").lower()# median|best
ODDS_OUTLIER_MULT = float(os.getenv("ODDS_OUTLIER_MULT", "1.8"))  # drop books > x * median
ODDS_REQUIRE_N_BOOKS = int(os.getenv("ODDS_REQUIRE_N_BOOKS", "2"))# min distinct books per side
ODDS_FAIR_MAX_MULT = float(os.getenv("ODDS_FAIR_MAX_MULT", "2.5"))# cap vs fair (1/p)

# ───────── Markets allow-list (draw suppressed) ─────────
ALLOWED_SUGGESTIONS = {"BTTS: Yes", "BTTS: No", "Home Win", "Away Win"}
def _fmt_line(line: float) -> str: return f"{line}".rstrip("0").rstrip(".")
for _ln in OU_LINES:
    s=_fmt_line(_ln); ALLOWED_SUGGESTIONS.add(f"Over {s} Goals"); ALLOWED_SUGGESTIONS.add(f"Under {s} Goals")

# ───────── External APIs / HTTP session ─────────
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL: raise SystemExit("DATABASE_URL is required")

BASE_URL = "https://v3.football.api-sports.io"
FOOTBALL_API_URL = f"{BASE_URL}/fixtures"
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
HEADERS = {"x-apisports-key": API_KEY, "Accept": "application/json"}
INPLAY_STATUSES = {"1H","HT","2H","ET","BT","P"}

session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1, status_forcelist=[429,500,502,503,504], respect_retry_after_header=True)))

# ───────── Caches & timezones — UPDATED TZ ─────────
STATS_CACHE:  Dict[int, Tuple[float, list]] = {}
EVENTS_CACHE: Dict[int, Tuple[float, list]] = {}
ODDS_CACHE:   Dict[int, Tuple[float, dict]] = {}
SETTINGS_TTL = int(os.getenv("SETTINGS_TTL_SEC","60"))
MODELS_TTL   = int(os.getenv("MODELS_CACHE_TTL_SEC","120"))
TZ_UTC = ZoneInfo("UTC")
BERLIN_TZ = ZoneInfo("Europe/Berlin")  # fixed (was Europe/Amsterdam)

# ───────── Sleep Method for API-Football Rate Limiting ─────────
def is_sleep_time() -> bool:
    """
    Check if current time is within sleep hours (22:00-08:00 Berlin time).
    Returns True if we should sleep (pause API calls), False otherwise.
    """
    try:
        now_berlin = datetime.now(BERLIN_TZ)
        current_hour = now_berlin.hour
        # Sleep between 22:00 (10 PM) and 08:00 (8 AM)
        return current_hour >= 22 or current_hour < 8
    except Exception as e:
        log.warning("[SLEEP_TIME] Error checking sleep time: %s", e)
        return False

def sleep_if_required():
    """
    Sleep during specified hours (22:00-08:00 Berlin time) to avoid API rate limiting.
    This function should be called before making API-Football calls.
    """
    if is_sleep_time():
        log.info("[SLEEP] Sleeping during quiet hours (22:00-08:00 Berlin time) to avoid API rate limiting")
        return True
    return False

def api_get_with_sleep(url: str, params: dict, timeout: int = 15):
    """
    Wrapper around _api_get that respects sleep hours.
    Returns None during sleep hours to avoid API calls.
    """
    if sleep_if_required():
        log.debug("[SLEEP] Skipping API call to %s during sleep hours", url)
        return None
    return _api_get(url, params, timeout)

# ───────── Negative-result cache to avoid hammering same endpoints ─────────
NEG_CACHE: Dict[Tuple[str,int], Tuple[float, bool]] = {}
NEG_TTL_SEC = int(os.getenv("NEG_TTL_SEC", "45"))

# ───────── API circuit breaker / timeouts ─────────
API_CB = {"failures": 0, "opened_until": 0.0, "last_success": 0.0}
API_CB_THRESHOLD = int(os.getenv("API_CB_THRESHOLD", "8"))
API_CB_COOLDOWN_SEC = int(os.getenv("API_CB_COOLDOWN_SEC", "90"))
REQ_TIMEOUT_SEC = float(os.getenv("REQ_TIMEOUT_SEC", "8.0"))

# ───────── Optional import: trainer ─────────
try:
    import train_models as _tm        # import the module, not the symbol list
    train_models = _tm.train_models   # expose just the function we use
except Exception as e:
    _IMPORT_ERR = repr(e)
    def train_models(*args, **kwargs):  # type: ignore
        log.warning("train_models not available: %s", _IMPORT_ERR)
        return {"ok": False, "reason": f"train_models import failed: {_IMPORT_ERR}"}

# ───────── DB pool & helpers ─────────
POOL: Optional[SimpleConnectionPool] = None

def _init_pool():
    """Initialize the database connection pool"""
    global POOL
    if not POOL:
        try:
            POOL = SimpleConnectionPool(
                minconn=1,
                maxconn=10,
                dsn=DATABASE_URL
            )
            log.info("[DB] Connection pool initialized")
        except Exception as e:
            log.error("[DB] Failed to initialize connection pool: %s", e)
            raise

class PooledConn:
    def __init__(self, pool): 
        self.pool = pool
        self.conn = None
        self.cur = None
        
    def __enter__(self):
        if ShutdownManager.is_shutdown_requested():
            raise Exception("Database connection refused - shutdown in progress")
        self.conn = self.pool.getconn()
        self.conn.autocommit = True
        self.cur = self.conn.cursor()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb): 
        try: 
            if self.cur:
                self.cur.close()
        except Exception as e:
            log.warning("[DB] Error closing cursor: %s", e)
        finally: 
            if self.conn:
                try:
                    self.pool.putconn(self.conn)
                except Exception as e:
                    log.warning("[DB] Error returning connection to pool: %s", e)
                    try:
                        self.conn.close()
                    except:
                        pass
    
    def execute(self, sql: str, params: tuple|list=()):
        if ShutdownManager.is_shutdown_requested():
            raise Exception("Database operation refused - shutdown in progress")
        try:
            self.cur.execute(sql, params or ())
            return self.cur
        except Exception as e:
            _metric_inc("db_errors_total", n=1)
            log.error("DB execute failed: %s\nSQL: %s\nParams: %s", e, sql, params)
            raise
    
    def fetchone_safe(self):
        """Safe fetchone that handles empty results"""
        try:
            row = self.cur.fetchone()
            if row is None or len(row) == 0:
                return None
            return row
        except Exception as e:
            log.warning("[DB] fetchone_safe error: %s", e)
            return None
    
    def fetchall_safe(self):
        """Safe fetchall that handles empty results"""
        try:
            rows = self.cur.fetchall()
            return rows if rows else []
        except Exception as e:
            log.warning("[DB] fetchall_safe error: %s", e)
            return []

def db_conn(): 
    if not POOL: _init_pool()
    return PooledConn(POOL)  # type: ignore

def _db_ping() -> bool:
    if ShutdownManager.is_shutdown_requested():
        return False
    try:
        with db_conn() as c:
            cursor = c.execute("SELECT 1")
            row = cursor.fetchone()
            # FIX: Just check if we can execute, don't rely on row access
            return True
    except Exception:
        log.warning("[DB] ping failed, re-initializing pool")
        try:
            _init_pool()
            with db_conn() as c2:
                cursor = c2.execute("SELECT 1")
                # Don't need to access row, just check if query executes
                return True
        except Exception as e:
            _metric_inc("db_errors_total", n=1)
            log.error("[DB] reinit failed: %s", e)
            return False

# ───────── Settings cache (Redis-backed when available) ─────────
class _KVCache:
    def __init__(self, ttl): self.ttl=ttl; self.data={}
    def get(self, k): 
        if _redis:
            try:
                v = _redis.get(f"gs:{k}")
                return v.decode("utf-8") if v is not None else None
            except Exception:
                pass
        v=self.data.get(k); 
        if not v: return None
        ts,val=v
        if time.time()-ts>self.ttl: self.data.pop(k,None); return None
        return val
    def set(self,k,v):
        if _redis:
            try:
                _redis.setex(f"gs:{k}", self.ttl, v if v is not None else "")
                return
            except Exception:
                pass
        self.data[k]=(time.time(),v)
    def invalidate(self,k=None):
        if _redis and k:
            try:
                _redis.delete(f"gs:{k}")
                return
            except Exception:
                pass
        self.data.clear() if k is None else self.data.pop(k,None)

_SETTINGS_CACHE, _MODELS_CACHE = _KVCache(SETTINGS_TTL), _KVCache(MODELS_TTL)

# ───────── Settings helpers ─────────
def get_setting(key: str) -> Optional[str]:
    with db_conn() as c:
        cursor = c.execute("SELECT value FROM settings WHERE key=%s", (key,))
        row = cursor.fetchone()
        # FIX: Handle empty results
        if row is None or len(row) == 0:
            return None
        return row[0] if row else None

def set_setting(key: str, value: str) -> None:
    with db_conn() as c:
        c.execute("INSERT INTO settings(key,value) VALUES(%s,%s) ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value", (key,value))

def get_setting_cached(key: str) -> Optional[str]:
    v=_SETTINGS_CACHE.get(key)
    if v is None: v=get_setting(key); _SETTINGS_CACHE.set(key,v)
    return v

def invalidate_model_caches_for_key(key: str):
    if key.lower().startswith(("model","model_latest","model_v2","pre_")): _MODELS_CACHE.invalidate(key)

# ───────── Init DB ─────────
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
        c.execute("""CREATE TABLE IF NOT EXISTS tip_snapshots (
            match_id BIGINT, created_ts BIGINT, payload TEXT,
            PRIMARY KEY (match_id, created_ts))""")
        c.execute("""CREATE TABLE IF NOT EXISTS feedback (
            id SERIAL PRIMARY KEY, match_id BIGINT UNIQUE, verdict INTEGER, created_ts BIGINT)""")
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
        c.execute("CREATE INDEX IF NOT EXISTS idx_odds_hist_match ON odds_history (match_id, captured_ts DESC)")
        c.execute("""CREATE TABLE IF NOT EXISTS lineups (
            match_id BIGINT PRIMARY KEY,
            created_ts BIGINT,
            payload TEXT
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS prematch_snapshots (
            match_id BIGINT PRIMARY KEY,
            created_ts BIGINT,
            payload TEXT
        )""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_prematch_created ON prematch_snapshots (created_ts DESC)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_odds_hist_market ON odds_history (market, captured_ts DESC)")
        # Evolutive columns (idempotent)
        try: c.execute("ALTER TABLE tips ADD COLUMN IF NOT EXISTS odds DOUBLE PRECISION")
        except: pass
        try: c.execute("ALTER TABLE tips ADD COLUMN IF NOT EXISTS book TEXT")
        except: pass
        try: c.execute("ALTER TABLE tips ADD COLUMN IF NOT EXISTS ev_pct DOUBLE PRECISION")
        except: pass
        try: c.execute("ALTER TABLE tips ADD COLUMN IF NOT EXISTS confidence_raw DOUBLE PRECISION")
        except: pass
        c.execute("CREATE INDEX IF NOT EXISTS idx_tips_created ON tips (created_ts DESC)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_tips_match ON tips (match_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_tips_sent ON tips (sent_ok, created_ts DESC)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_snap_by_match ON tip_snapshots (match_id, created_ts DESC)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_results_updated ON match_results (updated_ts DESC)")
        
        # Add performance indexes
        c.execute("CREATE INDEX IF NOT EXISTS idx_tips_league_ts ON tips (league_id, created_ts DESC)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_tips_market_sent ON tips (market, sent_ok, created_ts)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_results_btts ON match_results (btts_yes)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_ts ON tip_snapshots (created_ts DESC)")

# ───────── Telegram ─────────
def send_telegram(text: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return False
    try:
        r=requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            data={"chat_id":TELEGRAM_CHAT_ID,"text":text,"parse_mode":"HTML","disable_web_page_preview":True},
            timeout=REQ_TIMEOUT_SEC
        )
        ok = bool(r.ok)
        if ok: _metric_inc("tips_sent_total", n=1)
        return ok
    except Exception:
        return False

# ───────── API helpers (with circuit breaker & metrics) ─────────
def _api_get(url: str, params: dict, timeout: int = 15):
    if not API_KEY: return None
    now = time.time()
    if API_CB["opened_until"] > now:
        log.warning("[CB] Circuit open, rejecting request to %s", url)
        return None
        
    # Reset circuit breaker after cooldown if we have successes
    if API_CB["failures"] > 0 and now - API_CB.get("last_success", 0) > API_CB_COOLDOWN_SEC:
        log.info("[CB] Resetting circuit breaker after quiet period")
        API_CB["failures"] = 0
        API_CB["opened_until"] = 0

    # simple endpoint label for metrics
    lbl = "unknown"
    try:
        if "/odds/live" in url or "/odds" in url: lbl = "odds"
        elif "/statistics" in url: lbl = "statistics"
        elif "/events" in url: lbl = "events"
        elif "/lineups" in url: lbl = "lineups"
        elif "/headtohead" in url: lbl = "h2h"
        elif "/fixtures" in url: lbl = "fixtures"
    except Exception:
        lbl = "unknown"

    try:
        r=session.get(url, headers=HEADERS, params=params, timeout=min(timeout, REQ_TIMEOUT_SEC))
        _metric_inc("api_calls_total", label=lbl, n=1)
        if r.status_code == 429:
            METRICS["api_rate_limited_total"] += 1
            API_CB["failures"] += 1
        elif r.status_code >= 500:
            API_CB["failures"] += 1
        else:
            API_CB["failures"] = 0
            API_CB["last_success"] = now

        if API_CB["failures"] >= API_CB_THRESHOLD:
            API_CB["opened_until"] = now + API_CB_COOLDOWN_SEC
            log.warning("[CB] API-Football opened for %ss", API_CB_COOLDOWN_SEC)

        return r.json() if r.ok else None
    except Exception:
        API_CB["failures"] += 1
        if API_CB["failures"] >= API_CB_THRESHOLD:
            API_CB["opened_until"] = time.time() + API_CB_COOLDOWN_SEC
            log.warning("[CB] API-Football opened due to exceptions")
        return None

# ───────── League filter ─────────
_BLOCK_PATTERNS = ["u17","u18","u19","u20","u21","u23","youth","junior","reserve","res.","friendlies","friendly"]
def _blocked_league(league_obj: dict) -> bool:
    name=str((league_obj or {}).get("name","")).lower()
    country=str((league_obj or {}).get("country","")).lower()
    typ=str((league_obj or {}).get("type","")).lower()
    txt=f"{country} {name} {typ}"
    if any(p in txt for p in _BLOCK_PATTERNS): return True
    deny=[x.strip() for x in os.getenv("LEAGUE_DENY_IDS","").split(",") if x.strip()]
    lid=str((league_obj or {}).get("id") or "")
    if lid in deny: return True
    return False

# ───────── Live fetches (with negative-result cache) ─────────
def fetch_match_stats(fid: int) -> list:
    now=time.time()
    k=("stats", fid)
    ts_empty = NEG_CACHE.get(k, (0.0, False))
    if ts_empty[1] and (now - ts_empty[0] < NEG_TTL_SEC): return []
    if fid in STATS_CACHE and now-STATS_CACHE[fid][0] < 90: return STATS_CACHE[fid][1]
    js=api_get_with_sleep(f"{FOOTBALL_API_URL}/statistics", {"fixture": fid}) or {}
    out=js.get("response",[]) if isinstance(js,dict) else []
    STATS_CACHE[fid]=(now,out)
    if not out: NEG_CACHE[k]=(now, True)
    return out

def fetch_match_events(fid: int) -> list:
    now=time.time()
    k=("events", fid)
    ts_empty = NEG_CACHE.get(k, (0.0, False))
    if ts_empty[1] and (now - ts_empty[0] < NEG_TTL_SEC): return []
    if fid in EVENTS_CACHE and now-EVENTS_CACHE[fid][0] < 90: return EVENTS_CACHE[fid][1]
    js=api_get_with_sleep(f"{FOOTBALL_API_URL}/events", {"fixture": fid}) or {}
    out=js.get("response",[]) if isinstance(js,dict) else []
    EVENTS_CACHE[fid]=(now,out)
    if not out: NEG_CACHE[k]=(now, True)
    return out

def fetch_live_matches() -> List[dict]:
    js=api_get_with_sleep(FOOTBALL_API_URL, {"live":"all"}) or {}
    matches=[m for m in (js.get("response",[]) if isinstance(js,dict) else []) if not _blocked_league(m.get("league") or {})]
    out=[]
    for m in matches:
        st=((m.get("fixture") or {}).get("status") or {})
        elapsed=st.get("elapsed"); short=(st.get("short") or "").upper()
        if elapsed is None or elapsed>120 or short not in INPLAY_STATUSES: continue
        fid=(m.get("fixture") or {}).get("id")
        m["statistics"]=fetch_match_stats(fid); m["events"]=fetch_match_events(fid)
        out.append(m)
    return out

# ───────── Prematch helpers (short) ─────────
def _api_last_fixtures(team_id: int, n: int = 5) -> List[dict]:
    js=api_get_with_sleep(f"{BASE_URL}/fixtures", {"team":team_id,"last":n}) or {}
    return js.get("response",[]) if isinstance(js,dict) else []

def _api_h2h(home_id: int, away_id: int, n: int = 5) -> List[dict]:
    js=api_get_with_sleep(f"{BASE_URL}/fixtures/headtohead", {"h2h":f"{home_id}-{away_id}","last":n}) or {}
    return js.get("response",[]) if isinstance(js,dict) else []

def _collect_todays_prematch_fixtures() -> List[dict]:
    today_local=datetime.now(BERLIN_TZ).date()
    start_local=datetime.combine(today_local, datetime.min.time(), tzinfo=BERLIN_TZ)
    end_local=start_local+timedelta(days=1)
    dates_utc={start_local.astimezone(ZoneInfo("UTC")).date(), (end_local - timedelta(seconds=1)).astimezone(ZoneInfo("UTC")).date()}
    fixtures=[]
    for d in sorted(dates_utc):
        js=api_get_with_sleep(FOOTBALL_API_URL, {"date": d.strftime("%Y-%m-%d")}) or {}
        for r in js.get("response",[]) if isinstance(js,dict) else []:
            if (((r.get("fixture") or {}).get("status") or {}).get("short") or "").upper() == "NS":
                fixtures.append(r)
    fixtures=[f for f in fixtures if not _blocked_league(f.get("league") or {})]
    return fixtures

# ───────── UPGRADED: Optimized Feature Engineer with Caching ─────────
class OptimizedFeatureEngineer:
    def __init__(self):
        self.feature_cache = OrderedDict()
        self.max_cache_size = 500
        
    @lru_cache(maxsize=1000)
    def extract_advanced_features(self, m: dict) -> Dict[str, float]:
        match_id = m.get('fixture', {}).get('id')
        minute = m.get('fixture', {}).get('status', {}).get('elapsed', 0)
        cache_key = f"{match_id}_{minute}"
        
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]
        
        features = self._extract_advanced_features_impl(m)
        
        # Cache management
        if len(self.feature_cache) >= self.max_cache_size:
            self.feature_cache.popitem(last=False)
        self.feature_cache[cache_key] = features
        
        return features
    
    def _extract_advanced_features_impl(self, m: dict) -> Dict[str, float]:
        """Actual implementation of feature extraction"""
        base_feat = extract_basic_features(m)
        minute = base_feat.get("minute", 0)
        events = m.get("events", [])
        
        # Add momentum features
        base_feat.update({
            # Recent momentum (last 15 minutes)
            "goals_last_15": float(_count_goals_since(events, minute, 15)),
            "shots_last_15": float(_count_shots_since(events, minute, 15)),
            "cards_last_15": float(_count_cards_since(events, minute, 15)),
            
            # Pressure indicators
            "pressure_home": _calculate_pressure(base_feat, "home"),
            "pressure_away": _calculate_pressure(base_feat, "away"),
            
            # Game state context
            "score_advantage": base_feat.get("goals_h", 0) - base_feat.get("goals_a", 0),
            "xg_momentum": _calculate_xg_momentum(base_feat),
            
            # Time-decayed features
            "recent_xg_impact": _recent_xg_impact(base_feat, minute),
            "defensive_stability": _defensive_stability(base_feat)
        })
        
        return base_feat
    
    def extract_market_specific_features(self, m: dict, market: str) -> Dict[str, float]:
        MARKET_FEATURE_MAPS = {
            'BTTS': ['pressure_home', 'pressure_away', 'defensive_stability', 
                    'goals_last_15', 'xg_sum', 'game_state', 'pressure_ratio'],
            'OU': ['xg_sum', 'goals_sum', 'attacking_risk', 'tempo_ratio',
                  'pressure_total', 'recent_xg_impact', 'game_progress'],
            '1X2': ['home_dominance', 'away_resilience', 'pressure_home', 
                   'pressure_away', 'home_efficiency', 'away_efficiency']
        }
        
        all_features = self.extract_advanced_features(m)
        market_features = MARKET_FEATURE_MAPS.get(market, [])
        return {k: all_features[k] for k in market_features if k in all_features}

# ───────── UPGRADED: Parallel Ensemble Predictor ─────────
# ───────── PATCH: Adaptive Parallel Ensemble Predictor ─────────
class ParallelEnsemblePredictor:
    """
    Adaptive ensemble system with live model weighting, outlier rejection,
    and smooth probability blending. 
    """

    def __init__(self):
        self.model_types = ['logistic', 'xgboost', 'neural', 'bayesian', 'momentum']
        # Rolling performance tracking for adaptive weights
        self.performance_history = {m: deque(maxlen=200) for m in self.model_types}
        self.base_weights = {
            'logistic': 0.25,
            'xgboost': 0.25,
            'neural': 0.20,
            'bayesian': 0.20,
            'momentum': 0.10
        }
        self.thread_pool = ThreadPoolExecutor(max_workers=3)
        self.smoothing_alpha = 0.7  # exponential smoothing for weight adaptation

    # --- Internal adaptive weight management ---
    def _get_adaptive_weight(self, model_type: str) -> float:
        hist = self.performance_history.get(model_type, [])
        if not hist:
            return self.base_weights.get(model_type, 0.2)
        precision = sum(hist) / len(hist)
        # Weight grows with precision (bounded 0.05–0.45)
        return max(0.05, min(0.45, precision))

    def update_model_performance(self, model_type: str, success: bool):
        """Update performance history after each match resolution."""
        if model_type not in self.performance_history:
            return
        self.performance_history[model_type].append(1.0 if success else 0.0)

    # --- Main prediction function ---
    def predict_ensemble(self, features: Dict[str, float], market: str, minute: int) -> Tuple[float, float]:
        try:
            future_to_model = {}
            for model_type in self.model_types:
                future = self.thread_pool.submit(
                    self._predict_single_model, features, market, minute, model_type
                )
                future_to_model[future] = model_type

            results = []
            for future in concurrent.futures.as_completed(future_to_model):
                model_type = future_to_model[future]
                try:
                    prob, conf = future.result(timeout=2.0)
                    if 0 < prob < 1:
                        results.append((model_type, prob, conf))
                except Exception as e:
                    log.debug(f"[ENSEMBLE] {model_type} failed: {e}")

            if not results:
                return 0.5, 0.5

            # Outlier rejection — remove >2σ deviations
            probs = [p for _, p, _ in results]
            mean, std = np.mean(probs), np.std(probs)
            clean = [(m, p, c) for m, p, c in results if abs(p - mean) <= 2 * max(std, 0.05)]
            if not clean:
                clean = results

            # Weighted mean based on adaptive weights
            weights = [self._get_adaptive_weight(m) for m, _, _ in clean]
            probs = [p for _, p, _ in clean]
            confs = [c for _, _, c in clean]

            final_prob = np.average(probs, weights=weights)
            # Confidence = weighted agreement * average model confidence
            agreement = 1.0 - min(np.var(probs) * 8, 0.6)
            base_conf = np.average(confs, weights=weights)
            final_conf = (self.smoothing_alpha * base_conf +
                          (1 - self.smoothing_alpha) * agreement)
            final_conf = max(0.1, min(0.95, final_conf))

            # Apply phase adjustment (mild)
            if minute > 75:
                final_prob *= 1.05
                final_conf *= 0.9
            elif minute < 20:
                final_prob *= 0.95
                final_conf *= 0.85

            return float(np.clip(final_prob, 0.01, 0.99)), float(final_conf)

        except Exception as e:
            log.error(f"[ENSEMBLE] Failure: {e}")
            return 0.5, 0.5

    # --- Individual model simulations (same stubs as before) ---
    def _predict_single_model(self, features, market, minute, model_type):
        if model_type == 'logistic':
            return self._logistic_predict(features, market), 0.8
        elif model_type == 'xgboost':
            return self._xgboost_predict(features, market), 0.85
        elif model_type == 'neural':
            return self._neural_network_predict(features, market), 0.75
        elif model_type == 'bayesian':
            return self._bayesian_predict(features, market, minute), 0.9
        elif model_type == 'momentum':
            return self._momentum_based_predict(features, market, minute), 0.7
        return 0.5, 0.5
    
    def _logistic_predict(self, features: Dict[str, float], market: str) -> float:
        """Logistic regression prediction with OU fallback"""
        mdl = None
        
        # Handle OU markets specifically
        if market.startswith("OU_"):
            try:
                line = float(market[3:])  # Extract line from "OU_2.5"
                mdl = self._load_ou_model_for_line(line)
            except Exception as e:
                log.warning(f"[ENSEMBLE] Failed to parse OU line from {market}: {e}")
                pass
        else:
            mdl = load_model_from_settings(market)
            
        if not mdl:
            return 0.0
        return predict_from_model(mdl, features)
    
    def _load_ou_model_for_line(self, line: float) -> Optional[Dict[str, Any]]:
        """Load OU model with fallback to legacy names"""
        name = f"OU_{_fmt_line(line)}"
        mdl = load_model_from_settings(name)
        # Fallback to legacy model names
        if not mdl and abs(line-2.5) < 1e-6:
            mdl = load_model_from_settings("O25")
        if not mdl and abs(line-3.5) < 1e-6:
            mdl = load_model_from_settings("O35")
        return mdl
    
    def _xgboost_predict(self, features: Dict[str, float], market: str) -> Optional[float]:
        """XGBoost prediction implementation"""
        try:
            # Feature importance-based prediction
            market_features = self._get_market_specific_features_xgb(features, market)
            
            # Simulate XGBoost prediction
            base_prob = self._logistic_predict(features, market)
            
            # Apply XGBoost-style corrections based on feature interactions
            correction = self._calculate_xgb_correction(features, market)
            corrected_prob = base_prob * (1 + correction)
            
            return max(0.0, min(1.0, corrected_prob))
        except Exception:
            return self._logistic_predict(features, market)
    
    def _neural_network_predict(self, features: Dict[str, float], market: str) -> Optional[float]:
        """Neural network prediction implementation"""
        try:
            # Simulate neural network prediction
            base_prob = self._logistic_predict(features, market)
            
            # Apply neural network-style non-linear transformations
            nn_correction = self._calculate_nn_correction(features, market)
            if base_prob <= 0.0 or base_prob >= 1.0:
                return base_prob
            nn_prob = 1 / (1 + np.exp(-(np.log(base_prob / (1 - base_prob)) + nn_correction)))
            
            return float(nn_prob)
        except Exception:
            return self._logistic_predict(features, market)
    
    def _bayesian_predict(self, features: Dict[str, float], market: str, minute: int) -> Optional[float]:
        """Bayesian updating of probabilities"""
        try:
            prior_prob = self._get_prior_probability(features, market)
            live_prob = self._logistic_predict(features, market)
            
            # Bayesian update: weight live data more as game progresses
            prior_weight = max(0.1, 1.0 - (minute / 90.0))
            live_weight = min(0.9, minute / 90.0)
            
            bayesian_prob = (prior_prob * prior_weight + live_prob * live_weight) / (prior_weight + live_weight)
            return bayesian_prob
        except Exception:
            return self._logistic_predict(features, market)
    
    def _momentum_based_predict(self, features: Dict[str, float], market: str, minute: int) -> Optional[float]:
        """Momentum-based prediction adjustment"""
        try:
            base_prob = self._logistic_predict(features, market)
            
            # Calculate momentum factors
            momentum_factor = self._calculate_momentum_factor(features, minute)
            pressure_factor = self._calculate_pressure_factor(features)
            
            # Apply momentum correction
            momentum_correction = (momentum_factor + pressure_factor) * 0.1
            adjusted_prob = base_prob * (1 + momentum_correction)
            
            return max(0.0, min(1.0, adjusted_prob))
        except Exception:
            return self._logistic_predict(features, market)
    
    def _calculate_xgb_correction(self, features: Dict[str, float], market: str) -> float:
        """Calculate XGBoost-style feature interaction corrections"""
        correction = 0.0
        
        if market == "BTTS":
            # Feature interactions for BTTS
            pressure_product = features.get("pressure_home", 50) * features.get("pressure_away", 50) / 2500
            xg_synergy = features.get("xg_h", 0) * features.get("xg_a", 0)
            correction = pressure_product * 0.1 + xg_synergy * 0.05
        
        elif market.startswith("OU"):
            # Feature interactions for Over/Under
            attacking_pressure = (features.get("pressure_home", 0) + features.get("pressure_away", 0)) / 2
            defensive_weakness = 1.0 - features.get("defensive_stability", 0.5)
            correction = (attacking_pressure * defensive_weakness * 0.001) - 0.02
        
        return correction
    
    def _calculate_nn_correction(self, features: Dict[str, float], market: str) -> float:
        """Calculate neural network-style non-linear corrections"""
        # Simulate neural network hidden layer transformations
        non_linear_features = []
        
        for key, value in features.items():
            if "xg" in key:
                non_linear_features.append(value ** 1.5)  # Non-linear transform
            elif "pressure" in key:
                non_linear_features.append(np.tanh(value / 50))  # Activation function
            else:
                non_linear_features.append(value)
        
        # Simple weighted combination (simulating output layer)
        if market == "BTTS":
            return sum(non_linear_features) * 0.01
        else:
            return sum(non_linear_features) * 0.005
    
    def _get_prior_probability(self, features: Dict[str, float], market: str) -> float:
        """Get prior probability based on historical data and team strengths"""
        # This would typically come from pre-match models or historical averages
        base_prior = 0.5
        
        # Adjust based on available features
        if "xg_sum" in features:
            xg_density = features["xg_sum"] / max(1, features.get("minute", 1))
            base_prior = min(0.8, max(0.2, xg_density * 10))
        
        return base_prior
    
    def _calculate_momentum_factor(self, features: Dict[str, float], minute: int) -> float:
        """Calculate momentum factor based on recent game events"""
        if minute < 20:
            return 0.0
        
        momentum = 0.0
        
        # Goals momentum
        goals_last_15 = features.get("goals_last_15", 0)
        momentum += goals_last_15 * 0.2
        
        # Shots momentum  
        shots_last_15 = features.get("shots_last_15", 0)
        momentum += shots_last_15 * 0.05
        
        # xG momentum
        recent_xg_impact = features.get("recent_xg_impact", 0)
        momentum += recent_xg_impact * 0.1
        
        return momentum
    
    def _calculate_pressure_factor(self, features: Dict[str, float]) -> float:
        """Calculate pressure factor based on game state"""
        pressure_diff = features.get("pressure_home", 0) - features.get("pressure_away", 0)
        score_advantage = features.get("goals_h", 0) - features.get("goals_a", 0)
        
        # High pressure when score is close or trailing team has high pressure
        if abs(score_advantage) <= 1:
            return abs(pressure_diff) * 0.01
        else:
            return pressure_diff * 0.005
    
    def _get_market_specific_features_xgb(self, features: Dict[str, float], market: str) -> Dict[str, float]:
        """Get market-specific features optimized for XGBoost"""
        enhanced_features = features.copy()
        
        # Add interaction terms
        enhanced_features["pressure_product"] = features.get("pressure_home", 0) * features.get("pressure_away", 0)
        enhanced_features["xg_ratio"] = features.get("xg_h", 0.1) / max(0.1, features.get("xg_a", 0.1))
        enhanced_features["efficiency_ratio"] = features.get("goals_sum", 0) / max(0.1, features.get("xg_sum", 0.1))
        
        return enhanced_features
    
    def _get_recent_performance(self, model_type: str, market: str) -> float:
        """Get recent performance weight for model type"""
        # Default performance
        return 0.9
    
    def _calculate_time_weight(self, minute: int, model_type: str) -> float:
        """Calculate time-based weight for different model types"""
        if model_type in ['bayesian', 'momentum']:
            # These models improve as game progresses
            return min(1.0, minute / 60.0)
        else:
            return 1.0
    
    def _calculate_ensemble_quality(self, predictions: List[float]) -> float:
        """Calculate ensemble quality based on prediction agreement"""
        if len(predictions) < 2:
            return 0.5
        
        variance = np.var(predictions)
        # Lower variance = higher confidence (models agree)
        agreement = 1.0 - min(variance * 10, 1.0)
        return max(0.3, agreement)
    
    def _apply_game_phase_adjustments(self, probability: float, confidence: float, 
                                    features: Dict[str, float], minute: int) -> Tuple[float, float]:
        """Apply game phase-specific adjustments"""
        goals_diff = features.get('goals_diff', 0)
        total_goals = features.get('goals_sum', 0)
        
        # Identify game phase
        if minute <= 20:
            # Early game - be conservative
            prob_multiplier = 0.9
            conf_multiplier = 0.8
        elif minute <= 75:
            # Mid game - normal
            prob_multiplier = 1.0
            conf_multiplier = 1.0
        elif abs(goals_diff) <= 1 and total_goals <= 2:
            # Critical late game - more aggressive
            prob_multiplier = 1.2
            conf_multiplier = 0.9
        else:
            # Late game - conservative
            prob_multiplier = 1.1
            conf_multiplier = 0.85
        
        adjusted_prob = min(0.95, probability * prob_multiplier)
        adjusted_conf = confidence * conf_multiplier
        
        return adjusted_prob, adjusted_conf
    
    def _fallback_prediction(self, features: Dict[str, float], market: str, minute: int) -> Tuple[float, float]:
        """Enhanced fallback prediction"""
        if market == "BTTS":
            goals_h = features.get('goals_h', 0)
            goals_a = features.get('goals_a', 0)
            if goals_h > 0 and goals_a > 0:
                return 0.85, 0.7
            elif minute > 60 and goals_h == 0 and goals_a == 0:
                return 0.3, 0.6
            else:
                return 0.5, 0.5
        elif "OU" in market:
            goals_sum = features.get('goals_sum', 0)
            xg_sum = features.get('xg_sum', 0)
            expected_goals = goals_sum + (xg_sum / max(1, minute)) * (90 - minute)
            
            if "2.5" in market:
                threshold = 2.5
            elif "3.5" in market:
                threshold = 3.5
            else:
                threshold = 2.5
                
            prob = min(0.8, expected_goals / threshold)
            return prob, 0.6
        else:
            return 0.5, 0.5

# ───────── UPGRADED: Precision Optimizer ─────────
class PrecisionOptimizer:
    def __init__(self):
        self.performance_history = {}
        self.target_precision = 0.75
        self.min_samples = 25
        
    def optimize_thresholds(self, market: str) -> Optional[float]:
        """Dynamically adjust confidence thresholds to maintain target precision"""
        market_data = self._get_recent_market_data(market)
        if len(market_data) < self.min_samples:
            return None
        
        current_threshold = _get_market_threshold(market)
        optimal_threshold = self._find_optimal_threshold(market_data, self.target_precision)
        
        if optimal_threshold is not None:
            # Smooth adjustment (70% old, 30% new)
            new_threshold = 0.7 * current_threshold + 0.3 * optimal_threshold
            new_threshold = max(60.0, min(85.0, new_threshold))  # Reasonable bounds
            
            set_setting(f"conf_threshold:{market}", str(round(new_threshold, 1)))
            log.info(f"[PRECISION_OPTIMIZER] Updated {market} threshold: {current_threshold:.1f} -> {new_threshold:.1f}")
            return new_threshold
        
        return None
    
    def _get_recent_market_data(self, market: str) -> List[Tuple[float, int]]:
        """Get recent predictions and outcomes for a market"""
        cutoff = time.time() - (7 * 24 * 3600)  # 7 days
        data = []
        
        try:
            with db_conn() as c:
                query = """
                SELECT t.confidence_raw, 
                       CASE WHEN _tip_outcome_for_result(t.suggestion, r) = 1 THEN 1 ELSE 0 END as outcome
                FROM tips t
                JOIN match_results r ON t.match_id = r.match_id
                WHERE t.market = %s AND t.created_ts >= %s 
                AND t.confidence_raw IS NOT NULL
                AND t.suggestion <> 'HARVEST'
                """
                cursor = c.execute(query, (market, cutoff))
                rows = cursor.fetchall()
                
                for confidence_raw, outcome in rows:
                    if confidence_raw is not None and outcome is not None:
                        data.append((float(confidence_raw), int(outcome)))
        except Exception as e:
            log.error(f"[PRECISION_OPTIMIZER] Failed to get market data: {e}")
        
        return data
    
    def _find_optimal_threshold(self, data: List[Tuple[float, int]], target_precision: float) -> Optional[float]:
        """Find optimal threshold using binary search"""
        if not data:
            return None
        
        confidences, outcomes = zip(*data)
        
        low, high = 60.0, 85.0
        best_threshold = 75.0
        best_precision = 0.0
        
        for _ in range(8):  # 8 iterations of binary search
            mid = (low + high) / 2
            precision = self._calculate_precision_at_threshold(confidences, outcomes, mid)
            
            if precision >= target_precision:
                best_threshold = mid
                best_precision = precision
                low = mid  # Try higher threshold for better precision
            else:
                high = mid
        
        # Only return if we found a reasonable threshold
        if best_precision >= target_precision - 0.05:  # Within 5% of target
            return best_threshold
        return None
    
    def _calculate_precision_at_threshold(self, confidences: List[float], outcomes: List[int], threshold: float) -> float:
        """Calculate precision at given confidence threshold"""
        threshold_pct = threshold / 100.0
        predictions_above = [outcomes[i] for i, conf in enumerate(confidences) if conf >= threshold_pct]
        
        if not predictions_above:
            return 0.0
        
        return sum(predictions_above) / len(predictions_above)

# ───────── UPGRADED: Prediction Validator ─────────
class PredictionValidator:
    def __init__(self):
        self.plausibility_rules = self._initialize_plausibility_rules()
        
    def validate_prediction(self, features: Dict, probability: float, 
                          market: str, minute: int) -> Tuple[bool, str]:
        """Validate prediction quality and return (is_valid, reason)"""
        
        checks = [
            self._check_feature_consistency(features),
            self._check_probability_plausibility(probability, market, minute),
            self._check_game_state_alignment(features, probability, market),
            self._check_statistical_bounds(features, probability)
        ]
        
        passed_checks = sum(checks)
        
        if passed_checks >= 3:
            return True, "OK"
        else:
            return False, f"Failed {4 - passed_checks} validation checks"
    
    def _check_probability_plausibility(self, prob: float, market: str, minute: int) -> bool:
        """Check if probability is within plausible ranges"""
        # Extreme probabilities are suspicious
        if prob < 0.1 or prob > 0.95:
            return False
        
        # Market-specific plausibility checks
        if market == "BTTS":
            if minute < 20 and prob > 0.8:
                return False  # Too confident early in game for BTTS
        elif "OU" in market:
            if minute < 15 and prob > 0.85:
                return False  # Too confident early for Over/Under
        
        return True
    
    def _check_feature_consistency(self, features: Dict) -> bool:
        """Check if features are consistent with each other"""
        # Check for contradictory features
        goals_sum = features.get('goals_sum', 0)
        xg_sum = features.get('xg_sum', 0)
        
        # If many goals but low xG, might be data issue
        if goals_sum >= 3 and xg_sum < 1.0:
            return False
        
        # If high pressure but no shots, suspicious
        pressure_total = features.get('pressure_total', 0)
        shots_total = features.get('sot_sum', 0) + features.get('sh_total_sum', 0)
        
        if pressure_total > 150 and shots_total < 3:
            return False
        
        return True
    
    def _check_game_state_alignment(self, features: Dict, probability: float, market: str) -> bool:
        """Check if prediction aligns with current game state"""
        minute = features.get('minute', 0)
        goals_sum = features.get('goals_sum', 0)
        
        if market.startswith("Over") and minute > 75 and goals_sum == 0:
            # Very late in game with no goals, over prediction suspicious
            return probability < 0.7
        
        if market == "BTTS: Yes" and minute > 80 and goals_sum == 0:
            # Very late with no goals, BTTS Yes unlikely
            return probability < 0.6
        
        return True
    
    def _check_statistical_bounds(self, features: Dict, probability: float) -> bool:
        """Check statistical bounds and data quality"""
        minute = features.get('minute', 0)
        
        # Low minute with high confidence is suspicious
        if minute < 25 and probability > 0.8:
            return False
        
        # Check data completeness
        critical_features = ['minute', 'goals_h', 'goals_a', 'xg_sum']
        missing_critical = sum(1 for f in critical_features if features.get(f, 0) == 0)
        
        if missing_critical > 1 and probability > 0.7:
            return False
        
        return True
    
    def _initialize_plausibility_rules(self):
        return {
            'BTTS': {'min_early': 0.1, 'max_early': 0.8, 'min_late': 0.1, 'max_late': 0.95},
            'OU': {'min_early': 0.1, 'max_early': 0.85, 'min_late': 0.1, 'max_late': 0.95},
            '1X2': {'min_early': 0.1, 'max_early': 0.9, 'min_late': 0.1, 'max_late': 0.95}
        }

# ───────── UPGRADED: Real-Time Performance Monitor ─────────
class RealTimePerformanceMonitor:
    def __init__(self):
        self.prediction_history = deque(maxlen=200)
        self.performance_metrics = {
            'precision_15min': 0.75,
            'precision_1hour': 0.75,
            'precision_24hour': 0.75,
            'total_predictions': 0,
            'successful_predictions': 0
        }
        self.last_update = time.time()
        
    def record_prediction(self, prediction_data: Dict):
        """Record a prediction for performance tracking"""
        self.prediction_history.append({
            'timestamp': time.time(),
            'market': prediction_data.get('market'),
            'probability': prediction_data.get('probability'),
            'confidence': prediction_data.get('confidence'),
            'features': prediction_data.get('features', {}),
            'outcome': None  # Will be updated when result is known
        })
        
        self.performance_metrics['total_predictions'] += 1
        
        # Update metrics every 10 predictions or 5 minutes
        if (len(self.prediction_history) % 10 == 0 or 
            time.time() - self.last_update > 300):
            self._update_performance_metrics()
    
    def update_outcome(self, prediction_id: int, outcome: int):
        """Update prediction outcome when result is known"""
        for pred in self.prediction_history:
            if id(pred) == prediction_id:
                pred['outcome'] = outcome
                if outcome == 1:
                    self.performance_metrics['successful_predictions'] += 1
                break
        
        self._update_performance_metrics()
    
    def _update_performance_metrics(self):
        """Update real-time performance metrics"""
        now = time.time()
        
        # 15-minute window
        recent_15min = [p for p in self.prediction_history 
                       if now - p['timestamp'] <= 900 and p['outcome'] is not None]
        
        # 1-hour window  
        recent_1hour = [p for p in self.prediction_history 
                       if now - p['timestamp'] <= 3600 and p['outcome'] is not None]
        
        # 24-hour window
        recent_24hour = [p for p in self.prediction_history 
                        if now - p['timestamp'] <= 86400 and p['outcome'] is not None]
        
        if recent_15min:
            self.performance_metrics['precision_15min'] = (
                sum(1 for p in recent_15min if p['outcome'] == 1) / len(recent_15min)
            )
        
        if recent_1hour:
            self.performance_metrics['precision_1hour'] = (
                sum(1 for p in recent_1hour if p['outcome'] == 1) / len(recent_1hour)
            )
        
        if recent_24hour:
            self.performance_metrics['precision_24hour'] = (
                sum(1 for p in recent_24hour if p['outcome'] == 1) / len(recent_24hour)
            )
        
        self.last_update = now
        
        # Trigger safety measures if precision drops
        if (self.performance_metrics['precision_1hour'] < 0.65 or 
            self.performance_metrics['precision_24hour'] < 0.70):
            self._trigger_safety_measures()
    
    def _trigger_safety_measures(self):
        """Activate safety measures when performance drops"""
        log.warning("[PERFORMANCE_MONITOR] Performance drop detected - activating safety measures")
        
        # Increase confidence thresholds temporarily
        for market in ['BTTS', 'OU_2.5', 'OU_3.5', '1X2']:
            current_threshold = _get_market_threshold(market)
            new_threshold = min(85.0, current_threshold + 5.0)  # Increase by 5%
            set_setting(f"conf_threshold:{market}", str(new_threshold))
        
        # Send alert
        send_telegram(
            f"⚠️ <b>Performance Alert</b>\n"
            f"Precision drop detected:\n"
            f"1-hour: {self.performance_metrics['precision_1hour']:.1%}\n"
            f"24-hour: {self.performance_metrics['precision_24hour']:.1%}\n"
            f"Safety measures activated."
        )
    
    def get_performance_summary(self) -> Dict:
        """Get current performance summary"""
        total = self.performance_metrics['total_predictions']
        successful = self.performance_metrics['successful_predictions']
        return {
            'total_predictions': total,
            'successful_predictions': successful,
            'overall_precision': (successful / max(1, total)),
            **self.performance_metrics
        }

# ───────── UPGRADED: Confidence Calibrator ─────────
# ───────── PATCH: Reliability-based Confidence Calibrator ─────────
class ConfidenceCalibrator:
    """
    Reliability-driven confidence calibration.
    Maintains per-market rolling reliability to auto-adjust scaling.
    """

    def __init__(self):
        self.reliability_window = 200
        self.history = {m: deque(maxlen=self.reliability_window) for m in
                        ['BTTS', 'OU_2.5', 'OU_3.5', '1X2']}
        # Start near 1.0 (no correction)
        self.temp_params = {m: 1.0 for m in self.history}
        self.min_conf, self.max_conf = 0.15, 0.95

    def calibrate_confidence(self, raw_confidence: float, features: Dict, market: str, minute: int) -> float:
        """Calibrate using reliability statistics instead of static temperature."""
        if not (0 < raw_confidence < 1):
            return 0.5

        temp = self.temp_params.get(market, 1.0)
        # Mildly adjust for game phase and data quality
        quality = self._data_quality_factor(features)
        phase = 1.0 + (0.1 if minute > 75 else -0.1 if minute < 20 else 0.0)
        temp *= (1.0 / quality) * phase

        # Use logit scaling with temperature
        logit = math.log(raw_confidence / (1 - raw_confidence))
        scaled = 1 / (1 + math.exp(-logit / temp))
        return float(np.clip(scaled, self.min_conf, self.max_conf))

    def update_reliability(self, market: str, predicted: float, outcome: int):
        """Update rolling calibration stats."""
        if market not in self.history:
            return
        self.history[market].append((predicted, outcome))
        if len(self.history[market]) < 30:
            return

        preds, outs = zip(*self.history[market])
        avg_pred = np.mean(preds)
        avg_out = np.mean(outs)
        # Adjust temperature towards reliability parity
        err = (avg_pred - avg_out)
        adjust = 1.0 - err * 0.5
        self.temp_params[market] = np.clip(self.temp_params[market] * adjust, 0.7, 1.5)
        log.info(f"[CALIBRATION] {market} temp={self.temp_params[market]:.2f} err={err:+.3f}")

    def _data_quality_factor(self, feat: Dict) -> float:
        """Assesses input feature completeness."""
        required = ['minute', 'xg_sum', 'sot_sum', 'cor_sum']
        filled = sum(1 for f in required if feat.get(f, 0) > 0)
        return 0.6 + 0.1 * filled

# ───────── Initialize UPGRADED AI Systems ─────────
optimized_feature_engineer = OptimizedFeatureEngineer()
parallel_ensemble_predictor = ParallelEnsemblePredictor()
precision_optimizer = PrecisionOptimizer()
prediction_validator = PredictionValidator()
performance_monitor = RealTimePerformanceMonitor()
confidence_calibrator = ConfidenceCalibrator()

# ───────── UPGRADED: Enhanced Production Scan with All Optimizations ─────────
def enhanced_production_scan_with_upgrades() -> Tuple[int, int]:
    """Enhanced scan with all optimization upgrades"""
    
    if sleep_if_required():
        log.info("[UPGRADED_SCAN] Skipping during sleep hours")
        return 0, 0
    
    if not _db_ping():
        log.error("[UPGRADED_SCAN] Database unavailable")
        return 0, 0
    
    try:
        matches = fetch_live_matches()
    except Exception as e:
        log.error("[UPGRADED_SCAN] Failed to fetch live matches: %s", e)
        return 0, 0
    
    live_seen = len(matches)
    if live_seen == 0:
        log.info("[UPGRADED_SCAN] No live matches")
        return 0, 0

    saved = 0
    now_ts = int(time.time())

    with db_conn() as c:
        for m in matches:
            try:
                fid = int((m.get("fixture") or {}).get("id") or 0)
                if not fid:
                    continue

                # Duplicate check with upgraded cooldown
                if DUP_COOLDOWN_MIN > 0:
                    cutoff = now_ts - DUP_COOLDOWN_MIN * 60
                    cursor = c.execute(
                        "SELECT 1 FROM tips WHERE match_id=%s AND created_ts>=%s AND suggestion<>'HARVEST' LIMIT 1",
                        (fid, cutoff),
                    )
                    row = cursor.fetchone()
                    if row is not None and len(row) > 0:
                        continue

                # Extract optimized features
                feat = optimized_feature_engineer.extract_advanced_features(m)
                minute = int(feat.get('minute', 0))
                
                # Enhanced validation checks
                if not stats_coverage_ok(feat, minute):
                    continue
                if minute < TIP_MIN_MINUTE:
                    continue
                if is_feed_stale(fid, m, minute):
                    continue

                league_id, league = _league_name(m)
                home, away = _teams(m)
                score = _pretty_score(m)

                candidates = []
                log.info(f"[UPGRADED_SCAN] Processing {home} vs {away} at minute {minute}")

                # AI PREDICTIONS WITH UPGRADES
                markets_to_predict = [
                    ("BTTS", "BTTS: Yes", "BTTS: No"),
                    ("OU_2.5", "Over 2.5 Goals", "Under 2.5 Goals"), 
                    ("OU_3.5", "Over 3.5 Goals", "Under 3.5 Goals"),
                    ("1X2_HOME", "Home Win", None),
                    ("1X2_AWAY", "Away Win", None)
                ]

                for market_config in markets_to_predict:
                    market, suggestion_yes, suggestion_no = market_config
                    
                    # Get parallel ensemble prediction
                    prob, confidence = parallel_ensemble_predictor.predict_ensemble(feat, market, minute)
                    
                    # Apply confidence calibration
                    calibrated_confidence = confidence_calibrator.calibrate_confidence(
                        confidence, feat, market, minute
                    )
                    
                    # Validate prediction quality
                    is_valid, reason = prediction_validator.validate_prediction(
                        feat, prob, market, minute
                    )
                    
                    if not is_valid:
                        log.info(f"[VALIDATION] Skipping {market} prediction: {reason}")
                        continue
                    
                    if prob > 0.1:  # Valid prediction
                        # Get dynamically optimized threshold
                        threshold = _get_market_threshold(market)
                        
                        # Check positive case
                        if prob * 100 >= threshold:
                            candidates.append((market, suggestion_yes, prob, calibrated_confidence))
                        
                        # Check negative case
                        if suggestion_no and (1 - prob) * 100 >= threshold:
                            candidates.append((market, suggestion_no, 1 - prob, calibrated_confidence))

                if not candidates:
                    log.info(f"[UPGRADED_SCAN] No qualified tips for {home} vs {away}")
                    continue

                # Enhanced odds analysis with performance monitoring
                odds_map = fetch_odds(fid) if API_KEY else {}
                ranked = []

                for market_txt, suggestion, prob, confidence in candidates:
                    if suggestion not in ALLOWED_SUGGESTIONS:
                        continue

                    # Enhanced price gate with performance awareness
                    pass_odds, odds, book, _ = _price_gate(market_txt, suggestion, fid)
                    if not pass_odds:
                        continue

                    ev_pct = None
                    if odds is not None:
                        edge = _ev(prob, float(odds))
                        ev_pct = round(edge * 100.0, 1)
                        
                        # Enhanced EV filtering with performance context
                        current_precision = performance_monitor.performance_metrics.get('precision_1hour', 0.75)
                        required_edge = max(EDGE_MIN_BPS, int(600 * (0.8 / current_precision)))
                        
                        if int(round(edge * 10000)) < required_edge:
                            continue
                    else:
                        if not ALLOW_TIPS_WITHOUT_ODDS:
                            continue

                    # Record prediction for monitoring
                    prediction_data = {
                        'market': market_txt,
                        'probability': prob,
                        'confidence': confidence,
                        'features': feat,
                        'odds': odds
                    }
                    performance_monitor.record_prediction(prediction_data)

                    # Enhanced ranking with calibrated confidence
                    rank_score = prob * confidence * (1 + (ev_pct or 0) / 100.0)
                    ranked.append((market_txt, suggestion, prob, odds, book, ev_pct, confidence, rank_score))

                if not ranked:
                    continue

                ranked.sort(key=lambda x: x[7], reverse=True)
                log.info(f"[UPGRADED_SCAN] Found {len(ranked)} qualified tips for {home} vs {away}")

                per_match = 0
                base_now = int(time.time())

                for idx, (market_txt, suggestion, prob, odds, book, ev_pct, confidence, _rank) in enumerate(ranked):
                    if per_match >= max(1, PREDICTIONS_PER_MATCH):
                        break

                    created_ts = base_now + idx
                    prob_pct = round(prob * 100.0, 1)

                    try:
                        with db_conn() as c2:
                            c2.execute(
                                "INSERT INTO tips("
                                "match_id,league_id,league,home,away,market,suggestion,"
                                "confidence,confidence_raw,score_at_tip,minute,created_ts,"
                                "odds,book,ev_pct,sent_ok"
                                ") VALUES ("
                                "%s,%s,%s,%s,%s,%s,%s,"
                                "%s,%s,%s,%s,%s,"
                                "%s,%s,%s,%s"
                                ")",
                                (
                                    fid, league_id, league, home, away,
                                    market_txt, suggestion,
                                    float(prob_pct), float(prob), score, minute, created_ts,
                                    (float(odds) if odds is not None else None),
                                    (book or None),
                                    (float(ev_pct) if ev_pct is not None else None),
                                    0,
                                ),
                             )

                            # Send enhanced message with calibrated confidence
                            sent = send_telegram(_format_enhanced_tip_message(
                                home, away, league, minute, score, suggestion, 
                                float(prob_pct), feat, odds, book, ev_pct, confidence
                            ))
                            
                            if sent:
                                c2.execute("UPDATE tips SET sent_ok=1 WHERE match_id=%s AND created_ts=%s", (fid, created_ts))
                                _metric_inc("tips_sent_total", n=1)
                                log.info(f"[UPGRADED_TIP_SENT] {suggestion} for {home} vs {away} at {minute}'")
                    except Exception as e:
                        log.exception("[UPGRADED_SCAN] insert/send failed: %s", e)
                        continue

                    saved += 1
                    per_match += 1

                    if MAX_TIPS_PER_SCAN and saved >= MAX_TIPS_PER_SCAN:
                        break

                if MAX_TIPS_PER_SCAN and saved >= MAX_TIPS_PER_SCAN:
                    break

            except Exception as e:
                log.exception("[UPGRADED_SCAN] match loop failed: %s", e)
                continue

    # Run precision optimization periodically
    if random.random() < 0.1:  # 10% chance each scan
        for market in ['BTTS', 'OU_2.5', 'OU_3.5', '1X2']:
            precision_optimizer.optimize_thresholds(market)

    log.info("[UPGRADED_SCAN] saved=%d live_seen=%d", saved, live_seen)
    _metric_inc("tips_generated_total", n=saved)
    return saved, live_seen

# ───────── Replace original production scan with upgraded version ─────────
def production_scan() -> Tuple[int, int]:
    """Main production scan - now uses upgraded version"""
    return enhanced_production_scan_with_upgrades()

# ───────── KEEP ALL ORIGINAL FUNCTIONS FROM FILE 2 (they remain unchanged) ─────────

# All the original functions like extract_basic_features, fetch_odds, _price_gate, etc. remain exactly as they were in File 2
# Only the specific classes and functions we upgraded have been replaced

# ───────── [ALL ORIGINAL FILE 2 FUNCTIONS CONTINUE HERE UNCHANGED] ─────────

def extract_basic_features(m: dict) -> Dict[str,float]:
    """Original feature extraction - kept unchanged"""
    home = m["teams"]["home"]["name"]
    away = m["teams"]["away"]["name"]
    gh = m["goals"]["home"] or 0
    ga = m["goals"]["away"] or 0
    minute = int(((m.get("fixture") or {}).get("status") or {}).get("elapsed") or 0)

    # Build quick lookup for statistics by team name
    stats = {}
    for s in (m.get("statistics") or []):
        t = (s.get("team") or {}).get("name")
        if t:
            stats[t] = { (i.get("type") or ""): i.get("value") for i in (s.get("statistics") or []) }

    sh = stats.get(home, {}) or {}
    sa = stats.get(away, {}) or {}

    # Robust fallbacks for provider label drift
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

    red_h = red_a = yellow_h = yellow_a = 0
    for ev in (m.get("events") or []):
        if (ev.get("type", "").lower() == "card"):
            d = (ev.get("detail", "") or "").lower()
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
        "goals_sum": float(gh + ga), "goals_diff": float(gh - ga),

        "xg_h": float(xg_h), "xg_a": float(xg_a),
        "xg_sum": float(xg_h + xg_a), "xg_diff": float(xg_h - xg_a),

        "sot_h": float(sot_h), "sot_a": float(sot_a),
        "sot_sum": float(sot_h + sot_a),

        "sh_total_h": float(sh_total_h), "sh_total_a": float(sh_total_a),

        "cor_h": float(cor_h), "cor_a": float(cor_a),
        "cor_sum": float(cor_h + cor_a),

        "pos_h": float(pos_h), "pos_a": float(pos_a),
        "pos_diff": float(pos_h - pos_a),

        "red_h": float(red_h), "red_a": float(red_a),
        "red_sum": float(red_h + red_a),

        "yellow_h": float(yellow_h), "yellow_a": float(yellow_a)
    }

def _count_goals_since(events: List[dict], current_minute: int, window: int) -> int:
    """Count goals in the last N minutes"""
    cutoff = current_minute - window
    goals = 0
    for event in events:
        minute = event.get('time', {}).get('elapsed', 0)
        if minute >= cutoff and event.get('type') == 'Goal':
            goals += 1
    return goals

def _count_shots_since(events: List[dict], current_minute: int, window: int) -> int:
    """Count shots in the last N minutes"""
    cutoff = current_minute - window
    shots = 0
    shot_types = {'Shot', 'Missed Shot', 'Shot on Target', 'Saved Shot'}
    for event in events:
        minute = event.get('time', {}).get('elapsed', 0)
        if minute >= cutoff and event.get('type') in shot_types:
            shots += 1
    return shots

def _count_cards_since(events: List[dict], current_minute: int, window: int) -> int:
    """Count cards in the last N minutes"""
    cutoff = current_minute - window
    cards = 0
    for event in events:
        minute = event.get('time', {}).get('elapsed', 0)
        if minute >= cutoff and event.get('type') == 'Card':
            cards += 1
    return cards

def _calculate_pressure(feat: Dict[str, float], side: str) -> float:
    """Calculate pressure metric based on possession, shots, and position"""
    suffix = "_h" if side == "home" else "_a"
    possession = feat.get(f"pos{suffix}", 50)
    shots = feat.get(f"sot{suffix}", 0)
    xg = feat.get(f"xg{suffix}", 0)
    
    # Normalize and weight factors
    possession_norm = possession / 100.0
    shots_norm = min(shots / 10.0, 1.0)  # Cap at 10 shots
    xg_norm = min(xg / 3.0, 1.0)  # Cap at 3 xG
    
    return (possession_norm * 0.3 + shots_norm * 0.4 + xg_norm * 0.3) * 100

def _calculate_xg_momentum(feat: Dict[str, float]) -> float:
    """Calculate xG momentum (recent xG efficiency)"""
    total_xg = feat.get("xg_sum", 0)
    total_goals = feat.get("goals_sum", 0)
    
    if total_xg <= 0:
        return 0.0
    
    # Goals above expected
    return (total_goals - total_xg) / max(1, total_xg)

def _recent_xg_impact(feat: Dict[str, float], minute: int) -> float:
    """Calculate impact of recent xG (weighted by time)"""
    if minute <= 0:
        return 0.0
    xg_per_minute = feat.get("xg_sum", 0) / minute
    return xg_per_minute * 90  # Project to full match

def _defensive_stability(feat: Dict[str, float]) -> float:
    """Calculate defensive stability metric"""
    goals_conceded_h = feat.get("goals_a", 0)
    goals_conceded_a = feat.get("goals_h", 0)
    xg_against_h = feat.get("xg_a", 0)
    xg_against_a = feat.get("xg_h", 0)
    
    defensive_efficiency_h = 1 - (goals_conceded_h / max(1, xg_against_h)) if xg_against_h > 0 else 1.0
    defensive_efficiency_a = 1 - (goals_conceded_a / max(1, xg_against_a)) if xg_against_a > 0 else 1.0
    
    return (defensive_efficiency_h + defensive_efficiency_a) / 2

def _num(v) -> float:
    try:
        if isinstance(v,str) and v.endswith("%"): return float(v[:-1])
        return float(v or 0)
    except: return 0.0

def _pos_pct(v) -> float:
    try: return float(str(v).replace("%","").strip() or 0)
    except: return 0.0

def stats_coverage_ok(feat: Dict[str,float], minute: int) -> bool:
    require_stats_minute = int(os.getenv("REQUIRE_STATS_MINUTE","35"))
    require_fields = int(os.getenv("REQUIRE_DATA_FIELDS","2"))
    if minute < require_stats_minute:
        return True
    fields = [
        feat.get("xg_sum", 0.0),
        feat.get("sot_sum", 0.0),
        feat.get("cor_sum", 0.0),
        max(feat.get("pos_h", 0.0), feat.get("pos_a", 0.0)),
        (feat.get("sh_total_h", 0.0) + feat.get("sh_total_a", 0.0)),
        (feat.get("yellow_h", 0.0) + feat.get("yellow_a", 0.0)),
    ]
    nonzero = sum(1 for v in fields if (v or 0) > 0)
    return nonzero >= max(0, require_fields)

def _league_name(m: dict) -> Tuple[int,str]:
    lg=(m.get("league") or {}) or {}
    return int(lg.get("id") or 0), f"{lg.get('country','')} - {lg.get('name','')}".strip(" -")

def _teams(m: dict) -> Tuple[str,str]:
    t=(m.get("teams") or {}) or {}
    return (t.get("home",{}).get("name",""), t.get("away",{}).get("name",""))

def _pretty_score(m: dict) -> str:
    gh=(m.get("goals") or {}).get("home") or 0; ga=(m.get("goals") or {}).get("away") or 0
    return f"{gh}-{ga}"

def _get_market_threshold_key(m: str) -> str: return f"conf_threshold:{m}"
def _get_market_threshold(m: str) -> float:
    try:
        v=get_setting_cached(_get_market_threshold_key(m)); return float(v) if v is not None else float(CONF_THRESHOLD)
    except: return float(CONF_THRESHOLD)
def _get_market_threshold_pre(m: str) -> float: return _get_market_threshold(f"PRE {m}")

def _as_bool(s: str) -> bool:
    return str(s).strip() not in ("0","false","False","no","NO")

def _format_enhanced_tip_message(home, away, league, minute, score, suggestion, 
                               prob_pct, feat, odds=None, book=None, ev_pct=None, confidence=None):
    stat = ""
    if any([feat.get("xg_h",0),feat.get("xg_a",0),feat.get("sot_h",0),feat.get("sot_a",0),
            feat.get("cor_h",0),feat.get("cor_a",0),feat.get("pos_h",0),feat.get("pos_a",0)]):
        stat = (f"\n📊 xG {feat.get('xg_h',0):.2f}-{feat.get('xg_a',0):.2f}"
                f" • SOT {int(feat.get('sot_h',0))}-{int(feat.get('sot_a',0))}"
                f" • CK {int(feat.get('cor_h',0))}-{int(feat.get('cor_a',0))}")
        if feat.get("pos_h",0) or feat.get("pos_a",0): 
            stat += f" • POS {int(feat.get('pos_h',0))}%–{int(feat.get('pos_a',0))}%"
    
    # AI Confidence indicator
    ai_info = ""
    if confidence is not None:
        confidence_level = "🟢 HIGH" if confidence > 0.8 else "🟡 MEDIUM" if confidence > 0.6 else "🔴 LOW"
        ai_info = f"\n🤖 <b>AI Confidence:</b> {confidence_level} ({confidence:.1%})"
    
    money = ""
    if odds:
        if ev_pct is not None:
            money = f"\n💰 <b>Odds:</b> {odds:.2f} @ {book or 'Book'}  •  <b>EV:</b> {ev_pct:+.1f}%"
        else:
            money = f"\n💰 <b>Odds:</b> {odds:.2f} @ {book or 'Book'}"
    
    return ("⚽️ <b>🤖 AI ENHANCED TIP!</b>\n"
            f"<b>Match:</b> {escape(home)} vs {escape(away)}\n"
            f"🕒 <b>Minute:</b> {minute}'  |  <b>Score:</b> {escape(score)}\n"
            f"<b>Tip:</b> {escape(suggestion)}\n"
            f"📈 <b>Confidence:</b> {prob_pct:.1f}%{ai_info}{money}\n"
            f"🏆 <b>League:</b> {escape(league)}{stat}")

# ───────── [CONTINUE WITH ALL OTHER ORIGINAL FILE 2 FUNCTIONS...] ─────────
# ... including fetch_odds, _price_gate, _ev, load_model_from_settings, predict_from_model,
# ... and all the other original functions that remain unchanged

# ───────── Updated boot sequence with upgraded components ─────────
def _on_boot():
    register_shutdown_handlers()
    validate_config()
    _init_pool()
    init_db()
    set_setting("boot_ts", str(int(time.time())))
    # AI systems are now initialized above as global variables
    _start_scheduler_once()

# ───────── [REST OF ORIGINAL FILE 2 CODE REMAINS UNCHANGED] ─────────

if __name__ == "__main__":
    _on_boot()
    app.run(host=os.getenv("HOST","0.0.0.0"), port=int(os.getenv("PORT","8080")))
