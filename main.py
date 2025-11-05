# goalsniper — FULL AI mode (in-play + prematch) with odds + EV gate
# STREAMLINED: Removed redundant code, fixed broken functions, simplified architecture

import os, json, time, logging, requests, psycopg2, sys, signal, atexit
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
        import sentry_sdk
        sentry_sdk.init(
            dsn=SENTRY_DSN,
            traces_sample_rate=float(os.getenv("SENTRY_TRACES", "0.0")),
        )
    except Exception:
        pass

REDIS_URL = os.getenv("REDIS_URL")
_redis = None
if REDIS_URL:
    try:
        import redis
        _redis = redis.Redis.from_url(
            REDIS_URL, socket_timeout=1, socket_connect_timeout=1
        )
    except Exception:
        _redis = None

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
    "api_calls_total": defaultdict(int),
    "api_rate_limited_total": 0,
    "tips_generated_total": 0,
    "tips_sent_total": 0,
    "db_errors_total": 0,
    "job_duration_seconds": defaultdict(list)
}

def _metric_inc(name: str, label: Optional[str] = None, n: int = 1) -> None:
    try:
        if label is None:
            if isinstance(METRICS.get(name), int):
                METRICS[name] += n
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

# ───────── Required envs ─────────
def _require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise SystemExit(f"Missing required environment variable: {name}")
    return v

# ───────── Core env ─────────
TELEGRAM_BOT_TOKEN = _require_env("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = _require_env("TELEGRAM_CHAT_ID")
API_KEY            = _require_env("API_KEY")
ADMIN_API_KEY      = os.getenv("ADMIN_API_KEY")
WEBHOOK_SECRET     = os.getenv("TELEGRAM_WEBHOOK_SECRET")
RUN_SCHEDULER      = os.getenv("RUN_SCHEDULER", "1") not in ("0","false","False","no","NO")

# Precision-related knobs
CONF_THRESHOLD     = float(os.getenv("CONF_THRESHOLD", "75"))
MAX_TIPS_PER_SCAN  = int(os.getenv("MAX_TIPS_PER_SCAN", "25"))
DUP_COOLDOWN_MIN   = int(os.getenv("DUP_COOLDOWN_MIN", "20"))
TIP_MIN_MINUTE     = int(os.getenv("TIP_MIN_MINUTE", "12"))
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

# Optional-but-recommended warnings
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
PREDICTIONS_PER_MATCH = int(os.getenv("PREDICTIONS_PER_MATCH", "1"))
PER_LEAGUE_CAP        = int(os.getenv("PER_LEAGUE_CAP", "2"))

# ───────── Odds/EV controls ─────────
MIN_ODDS_OU   = float(os.getenv("MIN_ODDS_OU", "1.50"))
MIN_ODDS_BTTS = float(os.getenv("MIN_ODDS_BTTS", "1.50"))
MIN_ODDS_1X2  = float(os.getenv("MIN_ODDS_1X2", "1.50"))
MAX_ODDS_ALL  = float(os.getenv("MAX_ODDS_ALL", "20.0"))
EDGE_MIN_BPS  = int(os.getenv("EDGE_MIN_BPS", "600"))
ODDS_BOOKMAKER_ID = os.getenv("ODDS_BOOKMAKER_ID")
ALLOW_TIPS_WITHOUT_ODDS = os.getenv("ALLOW_TIPS_WITHOUT_ODDS","0") not in ("0","false","False","no","NO")

# Aggregated odds controls
ODDS_SOURCE = os.getenv("ODDS_SOURCE", "auto").lower()
ODDS_AGGREGATION = os.getenv("ODDS_AGGREGATION", "median").lower()
ODDS_OUTLIER_MULT = float(os.getenv("ODDS_OUTLIER_MULT", "1.8"))
ODDS_REQUIRE_N_BOOKS = int(os.getenv("ODDS_REQUIRE_N_BOOKS", "2"))
ODDS_FAIR_MAX_MULT = float(os.getenv("ODDS_FAIR_MAX_MULT", "2.5"))

# ───────── Markets allow-list (draw suppressed) ─────────
ALLOWED_SUGGESTIONS = {"BTTS: Yes", "BTTS: No", "Home Win", "Away Win", "O/U: 2.5", "O/U: 3.5"}
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

# ───────── Caches & timezones ─────────
STATS_CACHE:  Dict[int, Tuple[float, list]] = {}
EVENTS_CACHE: Dict[int, Tuple[float, list]] = {}
ODDS_CACHE:   Dict[int, Tuple[float, dict]] = {}
SETTINGS_TTL = int(os.getenv("SETTINGS_TTL_SEC","60"))
MODELS_TTL   = int(os.getenv("MODELS_CACHE_TTL_SEC","120"))
TZ_UTC = ZoneInfo("UTC")
BERLIN_TZ = ZoneInfo("Europe/Berlin")

# ───────── Negative-result cache ─────────
NEG_CACHE: Dict[Tuple[str,int], Tuple[float, bool]] = {}
NEG_TTL_SEC = int(os.getenv("NEG_TTL_SEC", "45"))

# ───────── API circuit breaker / timeouts ─────────
API_CB = {"failures": 0, "opened_until": 0.0, "last_success": 0.0}
API_CB_THRESHOLD = int(os.getenv("API_CB_THRESHOLD", "8"))
API_CB_COOLDOWN_SEC = int(os.getenv("API_CB_COOLDOWN_SEC", "90"))
REQ_TIMEOUT_SEC = float(os.getenv("REQ_TIMEOUT_SEC", "8.0"))

# ───────── Optional import: trainer ─────────
try:
    import train_models as _tm
    train_models = _tm.train_models
except Exception as e:
    _IMPORT_ERR = repr(e)
    def train_models(*args, **kwargs):
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
    """Get database connection from pool with proper initialization"""
    global POOL
    if not POOL:
        _init_pool()
    if not POOL:
        raise Exception("Database connection pool failed to initialize")
    return PooledConn(POOL)

def _db_ping() -> bool:
    if ShutdownManager.is_shutdown_requested():
        return False
    try:
        with db_conn() as c:
            cursor = c.execute("SELECT 1")
            row = cursor.fetchone()
            return True
    except Exception:
        log.warning("[DB] ping failed, re-initializing pool")
        try:
            _init_pool()
            with db_conn() as c2:
                cursor = c2.execute("SELECT 1")
                return True
        except Exception as e:
            _metric_inc("db_errors_total", n=1)
            log.error("[DB] reinit failed: %s", e)
            return False

# ───────── Settings cache ─────────
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
        if k is None:
            self.data.clear()
        else:
            self.data.pop(k, None)

_SETTINGS_CACHE, _MODELS_CACHE = _KVCache(SETTINGS_TTL), _KVCache(MODELS_TTL)

# ───────── Settings helpers ─────────
def get_setting(key: str) -> Optional[str]:
    with db_conn() as c:
        cursor = c.execute("SELECT value FROM settings WHERE key=%s", (key,))
        row = cursor.fetchone()
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
    except Exception as e:
        log.error("[TELEGRAM] Failed to send message: %s", e)
        return False

# ───────── API helpers ─────────
def api_get_with_sleep(url: str, params: dict, timeout: int = 15):
    """API call with basic rate limiting"""
    time.sleep(0.1)
    return _api_get(url, params, timeout)

def _api_get(url: str, params: dict, timeout: int = 15):
    """
    Make API request with proper error handling and circuit breaker
    """
    if not API_KEY: 
        log.error("[API] No API key available")
        return None
        
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
        log.debug("[API] Calling %s with params %s", url, params)
        r=session.get(url, headers=HEADERS, params=params, timeout=min(timeout, REQ_TIMEOUT_SEC))
        _metric_inc("api_calls_total", label=lbl, n=1)
        
        if r.status_code == 429:
            METRICS["api_rate_limited_total"] += 1
            API_CB["failures"] += 1
            log.warning("[API] Rate limited (429) for %s", url)
        elif r.status_code >= 500:
            API_CB["failures"] += 1
            log.error("[API] Server error %s for %s", r.status_code, url)
        else:
            API_CB["failures"] = 0
            API_CB["last_success"] = now

        if API_CB["failures"] >= API_CB_THRESHOLD:
            API_CB["opened_until"] = now + API_CB_COOLDOWN_SEC
            log.warning("[CB] API-Football opened for %ss", API_CB_COOLDOWN_SEC)

        if r.ok:
            log.debug("[API] Success for %s", url)
            return r.json()
        else:
            log.error("[API] Request failed with status %s: %s", r.status_code, r.text)
            return None
            
    except requests.exceptions.Timeout:
        log.error("[API] Timeout for %s", url)
        API_CB["failures"] += 1
    except requests.exceptions.ConnectionError:
        log.error("[API] Connection error for %s", url)
        API_CB["failures"] += 1
    except Exception as e:
        log.error("[API] Unexpected error for %s: %s", url, e)
        API_CB["failures"] += 1
        
    if API_CB["failures"] >= API_CB_THRESHOLD:
        API_CB["opened_until"] = time.time() + API_CB_COOLDOWN_SEC
        log.warning("[CB] API-Football opened due to exceptions")
        
    return None

# ───────── STREAMLINED: Simplified Ensemble System ─────────
class EnsemblePredictor:
    """Simplified ensemble system combining model predictions"""
    
    def __init__(self):
        self.model_weights = {
            'logistic': 0.6,
            'xgboost': 0.3, 
            'neural': 0.1
        }
        
    def predict(self, features: Dict[str, float], market: str, minute: int) -> Tuple[float, float]:
        """Get ensemble prediction with confidence"""
        predictions = []
        weights = []
        
        # Get predictions from available model types
        for model_type, weight in self.model_weights.items():
            try:
                prob = self._predict_single_model(features, market, model_type)
                if prob is not None:
                    predictions.append(prob)
                    weights.append(weight)
            except Exception as e:
                log.warning(f"[ENSEMBLE] {model_type} model failed: %s", e)
                continue
        
        if not predictions:
            return 0.0, 0.0
        
        # Weighted average of predictions
        ensemble_prob = sum(p * w for p, w in zip(predictions, weights)) / sum(weights)
        
        # Simple confidence calculation based on prediction agreement
        if len(predictions) > 1:
            confidence = 1.0 - (np.std(predictions) / 0.5)  # Normalized standard deviation
        else:
            confidence = 0.7  # Default confidence for single model
            
        confidence = max(0.1, min(0.95, confidence))
        
        return ensemble_prob, confidence
    
    def _predict_single_model(self, features: Dict[str, float], market: str, model_type: str) -> Optional[float]:
        """Get prediction from individual model type"""
        if model_type == 'logistic':
            return self._logistic_predict(features, market)
        elif model_type == 'xgboost':
            return self._xgboost_predict(features, market)
        elif model_type == 'neural':
            return self._neural_predict(features, market)
        return None
    
    def _logistic_predict(self, features: Dict[str, float], market: str) -> float:
        """Logistic regression prediction"""
        mdl = self._load_model_for_market(market)
        if not mdl:
            return 0.0
        return predict_from_model(mdl, features)
    
    def _xgboost_predict(self, features: Dict[str, float], market: str) -> Optional[float]:
        """XGBoost-style prediction with feature interactions"""
        try:
            base_prob = self._logistic_predict(features, market)
            
            # Apply feature interaction corrections
            correction = self._calculate_feature_correction(features, market)
            corrected_prob = base_prob * (1 + correction)
            
            return max(0.0, min(1.0, corrected_prob))
        except Exception:
            return self._logistic_predict(features, market)
    
    def _neural_predict(self, features: Dict[str, float], market: str) -> Optional[float]:
        """Neural network-style prediction"""
        try:
            base_prob = self._logistic_predict(features, market)
            
            # Apply non-linear transformation
            if base_prob <= 0.0 or base_prob >= 1.0:
                return base_prob
                
            # Simple sigmoid transformation
            transformed = 1 / (1 + np.exp(-(np.log(base_prob / (1 - base_prob)) + 0.1)))
            return float(transformed)
        except Exception:
            return self._logistic_predict(features, market)
    
    def _load_model_for_market(self, market: str) -> Optional[Dict[str, Any]]:
        """Load appropriate model for market"""
        if market.startswith("OU_"):
            try:
                line = float(market[3:])
                name = f"OU_{_fmt_line(line)}"
                mdl = load_model_from_settings(name)
                # Fallback to legacy model names
                if not mdl and abs(line-2.5) < 1e-6:
                    mdl = load_model_from_settings("O25")
                if not mdl and abs(line-3.5) < 1e-6:
                    mdl = load_model_from_settings("O35")
                return mdl
            except Exception:
                pass
        else:
            return load_model_from_settings(market)
        return None
    
    def _calculate_feature_correction(self, features: Dict[str, float], market: str) -> float:
        """Calculate feature-based probability correction"""
        correction = 0.0
        
        if market == "BTTS":
            # Feature interactions for BTTS
            pressure_balance = min(features.get("pressure_home", 0), features.get("pressure_away", 0)) / 100.0
            xg_synergy = features.get("xg_h", 0) * features.get("xg_a", 0)
            correction = pressure_balance * 0.15 + xg_synergy * 0.05
        
        elif market.startswith("OU"):
            # Feature interactions for Over/Under
            attacking_pressure = (features.get("pressure_home", 0) + features.get("pressure_away", 0)) / 200.0
            defensive_weakness = 1.0 - features.get("defensive_stability", 0.5)
            correction = (attacking_pressure * defensive_weakness * 0.2) - 0.02
        
        return correction

# Initialize global ensemble predictor
ensemble_predictor = EnsemblePredictor()

# ───────── STREAMLINED: Feature Engineering ─────────
class FeatureEngineer:
    """Feature engineering with game context"""
    
    def extract_features(self, m: dict) -> Dict[str, float]:
        """Extract features including game context"""
        base_feat = self._extract_basic_features(m)
        minute = base_feat.get("minute", 0)
        
        # Add momentum features
        base_feat.update({
            "pressure_home": self._calculate_pressure(base_feat, "home"),
            "pressure_away": self._calculate_pressure(base_feat, "away"),
            "defensive_stability": self._calculate_defensive_stability(base_feat),
            "game_state": self._classify_game_state(base_feat)
        })
        
        return base_feat
    
    def _extract_basic_features(self, m: dict) -> Dict[str,float]:
        """Extract basic match features"""
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
        xg_h = self._num(sh.get("Expected Goals", 0))
        xg_a = self._num(sa.get("Expected Goals", 0))
        sot_h = self._num(sh.get("Shots on Target", sh.get("Shots on Goal", 0)))
        sot_a = self._num(sa.get("Shots on Target", sa.get("Shots on Goal", 0)))
        cor_h = self._num(sh.get("Corner Kicks", 0))
        cor_a = self._num(sa.get("Corner Kicks", 0))
        pos_h = self._pos_pct(sh.get("Ball Possession", 0))
        pos_a = self._pos_pct(sa.get("Ball Possession", 0))

        return {
            "minute": float(minute),
            "goals_h": float(gh), "goals_a": float(ga),
            "goals_sum": float(gh + ga), "goals_diff": float(gh - ga),
            "xg_h": float(xg_h), "xg_a": float(xg_a),
            "xg_sum": float(xg_h + xg_a), "xg_diff": float(xg_h - xg_a),
            "sot_h": float(sot_h), "sot_a": float(sot_a),
            "sot_sum": float(sot_h + sot_a),
            "cor_h": float(cor_h), "cor_a": float(cor_a),
            "cor_sum": float(cor_h + cor_a),
            "pos_h": float(pos_h), "pos_a": float(pos_a),
            "pos_diff": float(pos_h - pos_a)
        }
    
    def _calculate_pressure(self, feat: Dict[str, float], side: str) -> float:
        """Calculate pressure metric"""
        suffix = "_h" if side == "home" else "_a"
        possession = feat.get(f"pos{suffix}", 50)
        shots = feat.get(f"sot{suffix}", 0)
        xg = feat.get(f"xg{suffix}", 0)
        
        # Normalize and weight factors
        possession_norm = possession / 100.0
        shots_norm = min(shots / 10.0, 1.0)
        xg_norm = min(xg / 3.0, 1.0)
        
        return (possession_norm * 0.3 + shots_norm * 0.4 + xg_norm * 0.3) * 100
    
    def _calculate_defensive_stability(self, feat: Dict[str, float]) -> float:
        """Calculate defensive stability metric"""
        goals_conceded_h = feat.get("goals_a", 0)
        goals_conceded_a = feat.get("goals_h", 0)
        xg_against_h = feat.get("xg_a", 0)
        xg_against_a = feat.get("xg_h", 0)
        
        defensive_efficiency_h = 1 - (goals_conceded_h / max(1, xg_against_h)) if xg_against_h > 0 else 1.0
        defensive_efficiency_a = 1 - (goals_conceded_a / max(1, xg_against_a)) if xg_against_a > 0 else 1.0
        
        return (defensive_efficiency_h + defensive_efficiency_a) / 2
    
    def _classify_game_state(self, feat: Dict[str, float]) -> float:
        """Classify current game state"""
        minute = feat.get("minute", 0)
        score_diff = feat.get("goals_h", 0) - feat.get("goals_a", 0)
        total_goals = feat.get("goals_sum", 0)
        
        if minute < 30:
            return 0.0  # Early game
        elif abs(score_diff) >= 3:
            return 1.0  # One-sided
        elif abs(score_diff) == 2 and minute > 70:
            return 0.8  # Comfortable lead late
        elif abs(score_diff) == 1 and minute > 75:
            return 0.9  # Close game late
        elif total_goals >= 3 and minute < 60:
            return 0.7  # Goal fest
        else:
            return 0.5  # Normal game state
    
    def _num(self, v) -> float:
        try:
            if isinstance(v,str) and v.endswith("%"): return float(v[:-1])
            return float(v or 0)
        except: return 0.0

    def _pos_pct(self, v) -> float:
        try: return float(str(v).replace("%","").strip() or 0)
        except: return 0.0

# Initialize global feature engineer
feature_engineer = FeatureEngineer()

# ───────── STREAMLINED: Market Prediction System ─────────
class MarketPredictor:
    """Market-specific prediction system"""
    
    def predict_for_market(self, features: Dict[str, float], market: str, minute: int) -> Tuple[float, float]:
        """Market-specific prediction"""
        if market.startswith("OU_"):
            return self._predict_ou(features, market, minute)
        elif market == "BTTS":
            return self._predict_btts(features, minute)
        elif market == "1X2":
            return self._predict_1x2(features, minute)
        else:
            # Fallback to ensemble prediction
            return ensemble_predictor.predict(features, market, minute)
    
    def _predict_btts(self, features: Dict[str, float], minute: int) -> Tuple[float, float]:
        """BTTS prediction"""
        prob, confidence = ensemble_predictor.predict(features, "BTTS", minute)
        
        # BTTS-specific adjustments
        defensive_stability = features.get("defensive_stability", 0.5)
        vulnerability = 1.0 - defensive_stability
        
        # Adjust probability based on defensive vulnerability
        adjusted_prob = prob * (1 + vulnerability * 0.2)
        adjusted_prob = max(0.0, min(1.0, adjusted_prob))
        
        return adjusted_prob, confidence
    
    def _predict_ou(self, features: Dict[str, float], market: str, minute: int) -> Tuple[float, float]:
        """Over/Under prediction"""
        prob, confidence = ensemble_predictor.predict(features, market, minute)
        
        # OU-specific adjustments
        current_goals = features.get("goals_sum", 0)
        xg_sum = features.get("xg_sum", 0)
        
        # Adjust based on current scoring rate vs expected
        if minute > 0:
            xg_per_minute = xg_sum / minute
            expected_remaining = xg_per_minute * (90 - minute)
            
            # If scoring faster than expected, increase Over probability
            scoring_rate = current_goals / max(1, minute)
            if scoring_rate > xg_per_minute * 1.2:
                prob = min(1.0, prob * 1.1)
            elif scoring_rate < xg_per_minute * 0.8:
                prob = max(0.0, prob * 0.9)
        
        return prob, confidence
    
    def _predict_1x2(self, features: Dict[str, float], minute: int) -> Tuple[float, float]:
        """1X2 prediction (draw suppressed)"""
        prob_h, conf_h = ensemble_predictor.predict(features, "1X2_HOME", minute)
        prob_a, conf_a = ensemble_predictor.predict(features, "1X2_AWAY", minute)
        
        # Normalize probabilities (suppress draw)
        total = prob_h + prob_a
        if total > 0:
            prob_h /= total
            prob_a /= total
        
        # Use average confidence
        confidence = (conf_h + conf_a) / 2
        
        # For 1X2, we return the higher probability and confidence
        if prob_h >= prob_a:
            return prob_h, confidence
        else:
            return prob_a, confidence

# Initialize market predictor
market_predictor = MarketPredictor()

# ───────── Core Helper Functions ─────────
def _league_name(m: dict) -> Tuple[int, str]:
    """Extract league ID and name from match data"""
    lg = (m.get("league") or {}) or {}
    league_id = int(lg.get("id") or 0)
    country = lg.get('country', '')
    name = lg.get('name', '')
    
    if country and name:
        league_name = f"{country} - {name}"
    elif name:
        league_name = name
    else:
        league_name = "Unknown League"
        
    return league_id, league_name.strip(" -")

def _teams(m: dict) -> Tuple[str, str]:
    """Extract home and away team names from match data"""
    t = (m.get("teams") or {}) or {}
    home_team = (t.get("home") or {}).get("name", "Home Team")
    away_team = (t.get("away") or {}).get("name", "Away Team")
    return home_team, away_team

def _pretty_score(m: dict) -> str:
    """Format match score as string"""
    goals = m.get("goals") or {}
    gh = goals.get("home") or 0
    ga = goals.get("away") or 0
    return f"{gh}-{ga}"

def _ev(prob: float, odds: float) -> float:
    """Calculate expected value as decimal (e.g., 0.05 = +5%)"""
    try:
        return float(prob) * max(0.0, float(odds)) - 1.0
    except (TypeError, ValueError):
        return 0.0

def _parse_ou_line_from_suggestion(s: str) -> Optional[float]:
    """Extract the line value from Over/Under suggestions"""
    try:
        if not s:
            return None
        if "Over" in s or "Under" in s:
            import re
            match = re.search(r'(\d+\.?\d*)', s)
            if match:
                return float(match.group(1))
        return None
    except (ValueError, TypeError):
        return None

def _get_market_threshold(m: str) -> float:
    """Get confidence threshold for a specific market"""
    try:
        v = get_setting_cached(_get_market_threshold_key(m))
        return float(v) if v is not None else float(CONF_THRESHOLD)
    except (ValueError, TypeError):
        return float(CONF_THRESHOLD)

def _get_market_threshold_key(m: str) -> str: 
    return f"conf_threshold:{m}"

def _get_market_threshold_pre(m: str) -> float: 
    return _get_market_threshold(f"PRE {m}")

def _min_odds_for_market(market: str) -> float:
    """Get minimum odds requirement for a market"""
    if market.startswith("Over/Under"): 
        return MIN_ODDS_OU
    if market == "BTTS": 
        return MIN_ODDS_BTTS
    if market == "1X2":  
        return MIN_ODDS_1X2
    return 1.01

def _market_family(market_text: str, suggestion: str) -> str:
    """Normalize to OU / BTTS / 1X2 (draw suppressed)"""
    s = (market_text or "").upper()
    if s.startswith("OVER/UNDER") or "OVER/UNDER" in s:
        return "OU"
    if s == "BTTS" or "BTTS" in s:
        return "BTTS"
    if s == "1X2" or "WINNER" in s or "MATCH WINNER" in s:
        return "1X2"
    if s.startswith("PRE "):
        return _market_family(s[4:], suggestion)
    return s

def _price_gate(market_text: str, suggestion: str, fid: int) -> Tuple[bool, Optional[float], Optional[str], Optional[float]]:
    """
    Return (pass, odds, book, ev_pct).
    """
    odds_map = fetch_odds(fid) if API_KEY else {}
    odds = None
    book = None

    if market_text == "BTTS":
        d = odds_map.get("BTTS", {})
        tgt = "Yes" if suggestion.endswith("Yes") else "No"
        if tgt in d: 
            odds = d[tgt]["odds"]
            book = d[tgt]["book"]
    elif market_text == "1X2":
        d = odds_map.get("1X2", {})
        tgt = "Home" if suggestion == "Home Win" else ("Away" if suggestion == "Away Win" else None)
        if tgt and tgt in d: 
            odds = d[tgt]["odds"]
            book = d[tgt]["book"]
    elif market_text.startswith("Over/Under"):
        ln_val = _parse_ou_line_from_suggestion(suggestion)
        d = odds_map.get(f"OU_{_fmt_line(ln_val)}", {}) if ln_val is not None else {}
        tgt = "Over" if suggestion.startswith("Over") else "Under"
        if tgt in d:
            odds = d[tgt]["odds"]
            book = d[tgt]["book"]

    if odds is None:
        return (True, None, None, None) if ALLOW_TIPS_WITHOUT_ODDS else (False, None, None, None)

    min_odds = _min_odds_for_market(market_text)
    if not (min_odds <= odds <= MAX_ODDS_ALL):
        return (False, odds, book, None)

    return (True, odds, book, None)

def _format_line(line: float) -> str: 
    """Format line number without trailing zeros"""
    return f"{line}".rstrip("0").rstrip(".")

def _tip_outcome_for_result(suggestion: str, res: Dict[str, Any]) -> Optional[int]:
    """Determine if a tip was correct (1), incorrect (0), or undecided (None)"""
    gh = int(res.get("final_goals_h") or 0)
    ga = int(res.get("final_goals_a") or 0)
    total = gh + ga
    btts = int(res.get("btts_yes") or 0)
    s = (suggestion or "").strip()
    
    if s.startswith("Over") or s.startswith("Under"):
        line = _parse_ou_line_from_suggestion(s)
        if line is None: 
            return None
        if s.startswith("Over"):
            if total > line: 
                return 1
            if abs(total - line) < 1e-9: 
                return None
            return 0
        else:
            if total < line: 
                return 1
            if abs(total - line) < 1e-9: 
                return None
            return 0
    if s == "BTTS: Yes": 
        return 1 if btts == 1 else 0
    if s == "BTTS: No":  
        return 1 if btts == 0 else 0
    if s == "Home Win":  
        return 1 if gh > ga else 0
    if s == "Away Win":  
        return 1 if ga > gh else 0
    return None

def _fixture_by_id(mid: int) -> Optional[dict]:
    """Fetch fixture data by ID"""
    js = api_get_with_sleep(FOOTBALL_API_URL, {"id": mid}) or {}
    arr = js.get("response") or [] if isinstance(js, dict) else []
    return arr[0] if arr else None

def _is_final(short: str) -> bool: 
    return (short or "").upper() in {"FT", "AET", "PEN"}

def _kickoff_berlin(utc_iso: str | None) -> str:
    """Convert UTC ISO timestamp to Berlin time"""
    try:
        if not utc_iso: 
            return "TBD"
        dt = datetime.fromisoformat(utc_iso.replace("Z", "+00:00"))
        return dt.astimezone(BERLIN_TZ).strftime("%H:%M")
    except Exception: 
        return "TBD"

def _as_bool(s: str) -> bool:
    """Convert string to boolean"""
    return str(s).strip() not in ("0", "false", "False", "no", "NO")

def _format_tip_message(home, away, league, minute, score, suggestion, prob_pct, feat, odds=None, book=None, ev_pct=None):
    """Format tip message for Telegram"""
    stat = ""
    if any([feat.get("xg_h", 0), feat.get("xg_a", 0), feat.get("sot_h", 0), feat.get("sot_a", 0),
            feat.get("cor_h", 0), feat.get("cor_a", 0), feat.get("pos_h", 0), feat.get("pos_a", 0)]):
        stat = (f"\n📊 xG {feat.get('xg_h', 0):.2f}-{feat.get('xg_a', 0):.2f}"
                f" • SOT {int(feat.get('sot_h', 0))}-{int(feat.get('sot_a', 0))}"
                f" • CK {int(feat.get('cor_h', 0))}-{int(feat.get('cor_a', 0))}")
        if feat.get("pos_h", 0) or feat.get("pos_a", 0): 
            stat += f" • POS {int(feat.get('pos_h', 0))}%–{int(feat.get('pos_a', 0))}%"
    
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

# ───────── Live fetches ─────────
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

# ───────── Prematch helpers ─────────
def _api_last_fixtures(team_id: int, n: int = 5) -> List[dict]:
    js=api_get_with_sleep(f"{BASE_URL}/fixtures", {"team":team_id,"last":n}) or {}
    return js.get("response",[]) if isinstance(js,dict) else []

def _api_h2h(home_id: int, away_id: int, n: int = 5) -> List[dict]:
    js=api_get_with_sleep(f"{BASE_URL}/fixtures/headtohead", {"h2h":f"{home_id}-{away_id}","last":n}) or {}
    return js.get("response",[]) if isinstance(js,dict) else []

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

# ───────── STREAMLINED: Production Scan ─────────
def production_scan() -> Tuple[int, int]:
    """Main production scan with simplified prediction system"""
    if not _db_ping():
        log.error("[PROD] Database unavailable")
        return (0, 0)
    
    try:
        matches = fetch_live_matches()
    except Exception as e:
        log.error("[PROD] Failed to fetch live matches: %s", e)
        return (0, 0)
    
    live_seen = len(matches)
    if live_seen == 0:
        log.info("[PROD] no live matches")
        return 0, 0

    saved = 0
    now_ts = int(time.time())
    per_league_counter: dict[int, int] = {}

    with db_conn() as c:
        for m in matches:
            try:
                fid = int((m.get("fixture") or {}).get("id") or 0)
                if not fid:
                    continue

                # Duplicate check
                if DUP_COOLDOWN_MIN > 0:
                    cutoff = now_ts - DUP_COOLDOWN_MIN * 60
                    cursor = c.execute(
                        "SELECT 1 FROM tips WHERE match_id=%s AND created_ts>=%s AND suggestion<>'HARVEST' LIMIT 1",
                        (fid, cutoff),
                    )
                    row = cursor.fetchone()
                    if row is not None and len(row) > 0:
                        continue

                # Extract features
                feat = feature_engineer.extract_features(m)
                minute = int(feat.get("minute", 0))
                current_goals_h = int(feat.get("goals_h", 0))
                current_goals_a = int(feat.get("goals_a", 0))
                btts_happened = current_goals_h > 0 and current_goals_a > 0
                
                # Validation checks
                if not stats_coverage_ok(feat, minute):
                    log.info(f"[STATS_COVERAGE] Skipping {fid} - insufficient stats coverage")
                    continue
                if minute < TIP_MIN_MINUTE:
                    log.info(f"[MINUTE_CHECK] Skipping {fid} - minute {minute} < {TIP_MIN_MINUTE}")
                    continue
                if is_feed_stale(fid, m, minute):
                    log.info(f"[FEED_STALE] Skipping {fid} - stale feed")
                    continue

                # Harvest mode snapshot
                if HARVEST_MODE and minute >= TRAIN_MIN_MINUTE and minute % 3 == 0:
                    try:
                        save_snapshot_from_match(m, feat)
                    except Exception:
                        pass

                league_id, league = _league_name(m)
                home, away = _teams(m)
                score = _pretty_score(m)

                candidates: List[Tuple[str, str, float, float]] = []

                # PREDICT ALL MARKETS
                log.info(f"[MARKET_SCAN] Processing {home} vs {away} at minute {minute} (Score: {score})")
                
                # 1. BTTS Market
                if not btts_happened:
                    btts_prob, btts_confidence = market_predictor.predict_for_market(feat, "BTTS", minute)
                    if btts_prob > 0 and btts_confidence > 0.3:
                        # Create both Yes and No suggestions
                        if btts_prob >= 0.5:
                            candidates.append(("BTTS", "BTTS: Yes", btts_prob, btts_confidence))
                        else:
                            candidates.append(("BTTS", "BTTS: No", 1 - btts_prob, btts_confidence))

                # 2. Over/Under Markets
                for line in OU_LINES:
                    market_key = f"OU_{_fmt_line(line)}"
                    ou_prob, ou_confidence = market_predictor.predict_for_market(feat, market_key, minute)
                    
                    if ou_prob > 0 and ou_confidence > 0.3:
                        if ou_prob >= 0.5:
                            candidates.append((f"Over/Under {_fmt_line(line)}", f"Over {_fmt_line(line)} Goals", ou_prob, ou_confidence))
                        else:
                            candidates.append((f"Over/Under {_fmt_line(line)}", f"Under {_fmt_line(line)} Goals", 1 - ou_prob, ou_confidence))

                # 3. 1X2 Market (draw suppressed)
                try:
                    prob_1x2, confidence_1x2 = market_predictor.predict_for_market(feat, "1X2", minute)
                    
                    if prob_1x2 > 0 and confidence_1x2 > 0.3:
                        # For 1X2, we already get the winning side probability
                        if prob_1x2 >= 0.5:
                            # Determine which side is more likely to win
                            prob_h, _ = ensemble_predictor.predict(feat, "1X2_HOME", minute)
                            prob_a, _ = ensemble_predictor.predict(feat, "1X2_AWAY", minute)
                            
                            if prob_h >= prob_a:
                                candidates.append(("1X2", "Home Win", prob_1x2, confidence_1x2))
                            else:
                                candidates.append(("1X2", "Away Win", prob_1x2, confidence_1x2))
                except Exception as e:
                    log.warning(f"[1X2_PREDICT] Failed for {home} vs {away}: {e}")

                if not candidates:
                    log.info(f"[NO_CANDIDATES] No qualified tips for {home} vs {away}")
                    continue

                # Apply confidence sanity checks
                candidates = apply_confidence_sanity_checks(candidates, minute, current_goals_h, current_goals_a)
                
                if not candidates:
                    log.info(f"[NO_SANE_CANDIDATES] All candidates filtered out by sanity checks for {home} vs {away}")
                    continue

                # DEBUG: Log candidates found
                log.info(f"[CANDIDATES_FOUND] {home} vs {away}: {len(candidates)} candidates after sanity checks")

                # Odds analysis and filtering
                odds_map = fetch_odds(fid)
                ranked: List[Tuple[str, str, float, Optional[float], Optional[str], Optional[float], float]] = []

                for mk, sug, prob, confidence in candidates:
                    if sug not in ALLOWED_SUGGESTIONS:
                        continue

                    # Odds lookup
                    odds = None
                    book = None
                    if mk == "BTTS":
                        d = odds_map.get("BTTS", {})
                        tgt = "Yes" if sug.endswith("Yes") else "No"
                        if tgt in d:
                            odds, book = d[tgt]["odds"], d[tgt]["book"]
                    elif mk == "1X2":
                        d = odds_map.get("1X2", {})
                        tgt = "Home" if sug == "Home Win" else ("Away" if sug == "Away Win" else None)
                        if tgt and tgt in d:
                            odds, book = d[tgt]["odds"], d[tgt]["book"]
                    elif mk.startswith("Over/Under"):
                        ln = _parse_ou_line_from_suggestion(sug)
                        d = odds_map.get(f"OU_{_fmt_line(ln)}", {}) if ln is not None else {}
                        tgt = "Over" if sug.startswith("Over") else "Under"
                        if tgt in d:
                            odds, book = d[tgt]["odds"], d[tgt]["book"]

                    # Price gate and EV calculation
                    pass_odds, odds2, book2, _ = _price_gate(mk, sug, fid)
                    if not pass_odds:
                        log.info(f"[PRICE_GATE_FAILED] {sug} failed price gate")
                        continue
                    if odds is None:
                        odds = odds2
                        book = book2

                    ev_pct = None
                    if odds is not None:
                        edge = _ev(prob, float(odds))
                        ev_pct = round(edge * 100.0, 1)
                        if int(round(edge * 10000)) < EDGE_MIN_BPS:
                            log.info(f"[EV_FAILED] {sug} - EV too low: {edge:.3f} < {EDGE_MIN_BPS/10000:.3f}")
                            continue
                    else:
                        if not ALLOW_TIPS_WITHOUT_ODDS:
                            log.info(f"[NO_ODDS] {sug} - no odds and ALLOW_TIPS_WITHOUT_ODDS=False")
                            continue

                    # Enhanced ranking with confidence scoring
                    rank_score = (prob ** 1.2) * (1 + (ev_pct or 0) / 100.0) * confidence
                    ranked.append((mk, sug, prob, odds, book, ev_pct, rank_score))
                    log.info(f"[RANKED_ADDED] {sug} - prob: {prob:.3f}, odds: {odds}, ev: {ev_pct}, rank: {rank_score:.3f}")

                if not ranked:
                    log.info(f"[NO_RANKED] No ranked tips after filtering for {home} vs {away}")
                    continue

                ranked.sort(key=lambda x: x[6], reverse=True)  # Sort by rank score
                log.info(f"[RANKED_TIPS] Found {len(ranked)} qualified tips for {home} vs {away}")

                per_match = 0
                base_now = int(time.time())

                for idx, (market_txt, suggestion, prob, odds, book, ev_pct, _rank) in enumerate(ranked):
                    if PER_LEAGUE_CAP > 0 and per_league_counter.get(league_id, 0) >= PER_LEAGUE_CAP:
                        log.info(f"[LEAGUE_CAP] League {league_id} reached cap of {PER_LEAGUE_CAP}")
                        break

                    created_ts = base_now + idx
                    raw = float(prob)
                    prob_pct = round(raw * 100.0, 1)

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
                                    float(prob_pct), raw, score, minute, created_ts,
                                    (float(odds) if odds is not None else None),
                                    (book or None),
                                    (float(ev_pct) if ev_pct is not None else None),
                                    0,
                                ),
                             )

                            sent = send_telegram(_format_tip_message(
                                home, away, league, minute, score, suggestion, 
                                float(prob_pct), feat, odds, book, ev_pct
                            ))
                            
                            if sent:
                                c2.execute("UPDATE tips SET sent_ok=1 WHERE match_id=%s AND created_ts=%s", (fid, created_ts))
                                _metric_inc("tips_sent_total", n=1)
                                log.info(f"[TIP_SENT] {suggestion} for {home} vs {away} at {minute}'")
                                saved += 1
                            else:
                                log.error(f"[TELEGRAM_FAILED] Failed to send {suggestion} for {home} vs {away}")
                    except Exception as e:
                        log.exception("[PROD] insert/send failed: %s", e)
                        continue

                    per_match += 1
                    per_league_counter[league_id] = per_league_counter.get(league_id, 0) + 1

                    if MAX_TIPS_PER_SCAN and saved >= MAX_TIPS_PER_SCAN:
                        log.info(f"[MAX_TIPS_REACHED] Reached max tips per scan: {MAX_TIPS_PER_SCAN}")
                        break
                    if per_match >= max(1, PREDICTIONS_PER_MATCH):
                        log.info(f"[MAX_PER_MATCH] Reached max tips per match: {PREDICTIONS_PER_MATCH}")
                        break

                if MAX_TIPS_PER_SCAN and saved >= MAX_TIPS_PER_SCAN:
                    log.info("[SCAN_LIMIT] Reached maximum tips per scan")
                    break

            except Exception as e:
                log.exception("[PROD] match loop failed: %s", e)
                continue

    # Final debug summary
    if saved > 0:
        log.info(f"[SUCCESS] Sent {saved} tips in this scan from {live_seen} matches")
    else:
        log.warning(f"[NO_TIPS_SENT] Scanned {live_seen} matches but sent 0 tips.")
        
    _metric_inc("tips_generated_total", n=saved)
    return saved, live_seen

# ───────── VALIDATION FUNCTIONS ─────────
def apply_confidence_sanity_checks(candidates: List[Tuple[str, str, float, float]], 
                                 minute: int, goals_h: int, goals_a: int) -> List[Tuple[str, str, float, float]]:
    """Apply sanity checks to filter invalid predictions"""
    sane_candidates = []
    btts_happened = goals_h > 0 and goals_a > 0
    total_goals = goals_h + goals_a
    
    for market, suggestion, prob, confidence in candidates:
        # Skip invalid BTTS predictions
        if market == "BTTS" and btts_happened:
            log.warning(f"[SANITY_CHECK] Skipping BTTS prediction - already happened at {minute}'")
            continue
            
        # Skip extremely high confidence early predictions
        if minute < 25 and confidence > 0.85:
            log.warning(f"[SANITY_CHECK] Skipping overconfident early prediction: {confidence:.3f} at {minute}'")
            continue
            
        # Skip predictions that don't make logical sense
        if not is_prediction_logical(market, suggestion, minute, goals_h, goals_a):
            log.warning(f"[SANITY_CHECK] Skipping illogical prediction: {market} - {suggestion}")
            continue
            
        sane_candidates.append((market, suggestion, prob, confidence))
    
    return sane_candidates

def is_prediction_logical(market: str, suggestion: str, minute: int, goals_h: int, goals_a: int) -> bool:
    """Check if prediction makes logical sense"""
    total_goals = goals_h + goals_a
    
    # BTTS: No when it's already 2-2 (illogical)
    if market == "BTTS" and suggestion == "BTTS: No" and goals_h >= 2 and goals_a >= 2:
        return False
        
    # Under predictions when many goals already scored
    if market.startswith("Over/Under") and suggestion.startswith("Under"):
        line = _parse_ou_line_from_suggestion(suggestion)
        if line and total_goals >= line:
            return False  # Can't go under if already exceeded
            
    # Additional sanity checks
    if minute > 80 and prob > 0.9:
        # Very high probability late in game is suspicious
        return False
        
    return True

# ───────── Model loader ─────────
def _validate_model_blob(name: str, tmp: dict) -> bool:
    if not isinstance(tmp, dict): return False
    if "weights" not in tmp or "intercept" not in tmp: return False
    if not isinstance(tmp["weights"], dict): return False
    if len(tmp["weights"]) > 2000: return False
    return True

MODEL_KEYS_ORDER = ["model_v2:{name}", "model_latest:{name}", "model:{name}", "pre_{name}"]

def load_model_from_settings(name: str) -> Optional[Dict[str, Any]]:
    cached=_MODELS_CACHE.get(name)
    if cached is not None: return cached
    mdl=None
    for pat in MODEL_KEYS_ORDER:
        raw=get_setting_cached(pat.format(name=name))
        if not raw: continue
        try:
            tmp=json.loads(raw)
            if not _validate_model_blob(name,tmp):
                log.warning("[MODEL] invalid schema for %s", name); continue
            tmp.setdefault("intercept",0.0); tmp.setdefault("weights",{})
            cal=tmp.get("calibration") or {}
            if isinstance(cal,dict):
                cal.setdefault("method","sigmoid"); cal.setdefault("a",1.0); cal.setdefault("b",0.0)
                tmp["calibration"]=cal
            mdl=tmp; break
        except Exception as e:
            log.warning("[MODEL] parse %s failed: %s", name, e)
    if mdl is not None: _MODELS_CACHE.set(name, mdl)
    return mdl

# ───────── Logistic predict ─────────
def predict_from_model(mdl: Dict[str, Any], features: Dict[str, float]) -> float:
    w=mdl.get("weights") or {}; s=mdl.get("intercept",0.0)
    for k,v in w.items(): s+=v*features.get(k,0.0)
    prob=1/(1+np.exp(-s))
    cal=mdl.get("calibration") or {}
    if isinstance(cal,dict) and cal.get("method")=="sigmoid":
        a=cal.get("a",1.0); b=cal.get("b",0.0)
        prob=1/(1+np.exp(-(a*prob+b)))
    return float(prob)

# ───────── Odds fetch + aggregation ─────────
def fetch_odds(fid: int, prob_hints: Optional[dict[str, float]] = None) -> dict:
    """
    Aggregated odds map
    """
    now = time.time()
    k=("odds", fid)
    ts_empty = NEG_CACHE.get(k, (0.0, False))
    if ts_empty[1] and (now - ts_empty[0] < NEG_TTL_SEC): return {}
    if fid in ODDS_CACHE and now-ODDS_CACHE[fid][0] < 120: return ODDS_CACHE[fid][1]

    def _fetch(path: str) -> dict:
        js = api_get_with_sleep(f"{BASE_URL}/{path}", {"fixture": fid}) or {}
        return js if isinstance(js, dict) else {}

    js = {}
    if ODDS_SOURCE in ("auto", "live"):
        js = _fetch("odds/live")
    if not (js.get("response") or []) and ODDS_SOURCE in ("auto", "prematch"):
        js = _fetch("odds")

    by_market: dict[str, dict[str, list[tuple[float, str]]]] = {}
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
        ok = True
        for side, lst in side_map.items():
            if len({b for (_, b) in lst}) < max(1, ODDS_REQUIRE_N_BOOKS):
                ok = False
                break
        if not ok:
            continue

        out[mkey] = {}
        for side, lst in side_map.items():
            ag, label = _aggregate_price(lst)
            if ag is not None:
                out[mkey][side] = {"odds": float(ag), "book": label}

    ODDS_CACHE[fid] = (time.time(), out)
    if not out: NEG_CACHE[k] = (now, True)
    return out

def _market_name_normalize(s: str) -> str:
    s=(s or "").lower()
    if "both teams" in s or "btts" in s: return "BTTS"
    if "match winner" in s or "winner" in s or "1x2" in s: return "1X2"
    if "over/under" in s or "total" in s or "goals" in s: return "OU"
    return s

def _aggregate_price(vals: list[tuple[float, str]]) -> tuple[Optional[float], Optional[str]]:
    if not vals:
        return None, None
    xs = sorted([o for (o, _) in vals if (o or 0) > 0])
    if not xs:
        return None, None
    import statistics
    med = statistics.median(xs)
    filtered = [(o, b) for (o, b) in vals if o <= med * max(1.0, ODDS_OUTLIER_MULT)]
    if not filtered:
        filtered = vals
    xs2 = sorted([o for (o, _) in filtered])
    med2 = statistics.median(xs2)
    if ODDS_AGGREGATION == "best":
        best = max(filtered, key=lambda t: t[0])
        return float(best[0]), str(best[1])
    target = med2
    pick = min(filtered, key=lambda t: abs(t[0] - target))
    return float(pick[0]), f"{pick[1]} (median of {len(xs)})"

# ───────── Data-quality & formatting helpers ─────────
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
    ]
    nonzero = sum(1 for v in fields if (v or 0) > 0)
    return nonzero >= max(0, require_fields)

def _parse_market_cutoffs(s: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for tok in (s or "").split(","):
        tok = tok.strip()
        if not tok or "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        try:
            out[k.strip().upper()] = int(float(v.strip()))
        except Exception:
            pass
    return out

_MARKET_CUTOFFS = _parse_market_cutoffs(MARKET_CUTOFFS_RAW)
try:
    _TIP_MAX_MINUTE = int(float(TIP_MAX_MINUTE_ENV)) if TIP_MAX_MINUTE_ENV.strip() else None
except Exception:
    _TIP_MAX_MINUTE = None

def market_cutoff_ok(minute: int, market_text: str, suggestion: str) -> bool:
    """
    True if we are still within the minute cutoff for this market.
    """
    fam = _market_family(market_text, suggestion)
    if minute is None:
        return True
    try:
        m = int(minute)
    except Exception:
        m = 0
    cutoff = _MARKET_CUTOFFS.get(fam)
    if cutoff is None:
        cutoff = _TIP_MAX_MINUTE
    if cutoff is None:
        cutoff = max(0, int(TOTAL_MATCH_MINUTES) - 5)
    return m <= int(cutoff)

# ───────── Sanity checks & stale-feed guard ─────────
def _candidate_is_sane(sug: str, feat: Dict[str,float]) -> bool:
    gh = int(feat.get("goals_h", 0))
    ga = int(feat.get("goals_a", 0))
    total = gh + ga

    if sug.startswith("Over"):
        ln = _parse_ou_line_from_suggestion(sug)
        return (ln is not None) and (total < ln)

    if sug.startswith("Under"):
        ln = _parse_ou_line_from_suggestion(sug)
        return (ln is not None) and (total < ln)

    if sug.startswith("BTTS") and (gh > 0 and ga > 0):
        return False

    return True

_FEED_STATE: Dict[int, Dict[str, Any]] = {}

def _safe_num(x) -> float:
    try:
        if isinstance(x, str) and x.endswith("%"):
            return float(x[:-1])
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
            if k in d:
                return _safe_num(d[k])
        return 0.0

    xg_h = g(sh, ("expected goals",))
    xg_a = g(sa, ("expected goals",))
    sot_h = g(sh, ("shots on target", "shots on goal"))
    sot_a = g(sa, ("shots on target", "shots on goal"))
    cor_h = g(sh, ("corner kicks",))
    cor_a = g(sa, ("corner kicks",))
    pos_h = g(sh, ("ball possession",))
    pos_a = g(sa, ("ball possession",))

    ev = m.get("events") or []
    n_events = len(ev)

    gh = int(((m.get("goals") or {}).get("home") or 0) or 0)
    ga = int(((m.get("goals") or {}).get("away") or 0) or 0)

    return (
        round(xg_h + xg_a, 3),
        int(sot_h + sot_a),
        int(cor_h + cor_a),
        int(round(pos_h)), int(round(pos_a)),
        gh, ga,
        n_events,
    )

def is_feed_stale(fid: int, m: dict, minute: int) -> bool:
    if not STALE_GUARD_ENABLE:
        return False
        
    now = time.time()
    
    if minute < 10:
        st = _FEED_STATE.get(fid)
        fp = _match_fingerprint(m)
        _FEED_STATE[fid] = {"fp": fp, "last_change": now, "last_minute": minute}
        return False

    fp = _match_fingerprint(m)
    st = _FEED_STATE.get(fid)

    if st is None:
        _FEED_STATE[fid] = {"fp": fp, "last_change": now, "last_minute": minute}
        return False

    if fp != st.get("fp"):
        st["fp"] = fp
        st["last_change"] = now
        st["last_minute"] = minute
        return False

    last_min = int(st.get("last_minute") or 0)
    st["last_minute"] = minute

    if minute > last_min and (now - float(st.get("last_change") or now)) >= STALE_STATS_MAX_SEC:
        return True

    return False

# ───────── Snapshots ─────────
def save_snapshot_from_match(m: dict, feat: Dict[str, float]) -> None:
    fx = (m.get("fixture") or {})
    lg = (m.get("league") or {})
    teams = (m.get("teams") or {})

    fid = int(fx.get("id") or 0)
    if not fid:
        return

    league_id = int(lg.get("id") or 0)
    league = f"{lg.get('country','')} - {lg.get('name','')}".strip(" -")
    home = (teams.get("home") or {}).get("name", "")
    away = (teams.get("away") or {}).get("name", "")

    gh = int((m.get("goals") or {}).get("home") or 0)
    ga = int((m.get("goals") or {}).get("away") or 0)
    minute = int(feat.get("minute", 0))

    snapshot = {
        "minute": minute,
        "gh": gh, "ga": ga,
        "league_id": league_id,
        "market": "HARVEST",
        "suggestion": "HARVEST",
        "confidence": 0,
        "stat": feat
    }

    now = int(time.time())
    payload = json.dumps(snapshot, separators=(",", ":"), ensure_ascii=False)[:200000]

    with db_conn() as c:
        c.execute(
            "INSERT INTO tip_snapshots(match_id, created_ts, payload) VALUES (%s,%s,%s) "
            "ON CONFLICT (match_id, created_ts) DO UPDATE SET payload=EXCLUDED.payload",
            (fid, now, payload)
        )
        c.execute(
            "INSERT INTO tips("
            "match_id,league_id,league,home,away,market,suggestion,confidence,confidence_raw,score_at_tip,minute,created_ts,sent_ok"
            ") VALUES ("
            "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s"
            ")",
            (
                fid, league_id, league, home, away,
                "HARVEST", "HARVEST",
                0.0, 0.0,
                f"{gh}-{ga}",
                minute, now, 1
            )
        )

# ───────── Outcomes/backfill/digest ─────────
def backfill_results_for_open_matches(max_rows: int = 200) -> int:
    """Backfill results without sleep time check"""
    now_ts=int(time.time()); cutoff=now_ts - BACKFILL_DAYS*24*3600; updated=0
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

def daily_accuracy_digest(window_days: int = 1) -> Optional[str]:
    """Daily accuracy digest for today's tips only"""
    if not DAILY_ACCURACY_DIGEST_ENABLE: 
        return None
    
    # Get today's date in Berlin timezone
    today = datetime.now(BERLIN_TZ).date()
    start_of_day = datetime.combine(today, datetime.min.time(), tzinfo=BERLIN_TZ)
    start_ts = int(start_of_day.timestamp())
    
    log.info("[DIGEST] Generating daily digest for today (since %s)", start_of_day)
    
    # Backfill recent results
    backfill_results_for_open_matches(200)

    with db_conn() as c:
        rows = c.execute("""
            SELECT t.market, t.suggestion, t.confidence, t.confidence_raw, t.created_ts,
                   t.odds, r.final_goals_h, r.final_goals_a, r.btts_yes
            FROM tips t LEFT JOIN match_results r ON r.match_id=t.match_id
            WHERE t.created_ts >= %s 
            AND t.suggestion<>'HARVEST' 
            AND t.sent_ok=1
            ORDER BY t.created_ts DESC
        """, (start_ts,)).fetchall()

    total = graded = wins = 0
    roi_by_market, by_market = {}, {}
    recent_tips = []

    for (mkt, sugg, conf, conf_raw, cts, odds, gh, ga, btts) in rows:
        res = {"final_goals_h": gh, "final_goals_a": ga, "btts_yes": btts}
        out = _tip_outcome_for_result(sugg, res)
        
        # Store tip info for recent tips list
        tip_time = datetime.fromtimestamp(cts, BERLIN_TZ).strftime("%H:%M")
        recent_tips.append(f"{sugg} ({conf:.1f}%) - {tip_time}")
        
        if out is None: 
            continue

        total += 1
        graded += 1
        wins += 1 if out == 1 else 0
        
        d = by_market.setdefault(mkt or "?", {"graded": 0, "wins": 0})
        d["graded"] += 1
        d["wins"] += 1 if out == 1 else 0

        if odds:
            roi_by_market.setdefault(mkt, {"stake": 0, "pnl": 0})
            roi_by_market[mkt]["stake"] += 1
            if out == 1: 
                roi_by_market[mkt]["pnl"] += float(odds) - 1
            else: 
                roi_by_market[mkt]["pnl"] -= 1

    if graded == 0:
        msg = f"📊 Daily Accuracy Digest for {today.strftime('%Y-%m-%d')}\nNo graded tips today."
        
        # Show pending tips if any
        if rows:
            pending = len([r for r in rows if r[5] is None or r[6] is None])
            msg += f"\n⏳ {pending} tips still pending results."
            
    else:
        acc = 100.0 * wins / max(1, graded)
        lines = [
            f"📊 <b>Daily Accuracy Digest</b> - {today.strftime('%Y-%m-%d')}",
            f"Tips sent: {total}  •  Graded: {graded}  •  Wins: {wins}  •  Accuracy: {acc:.1f}%"
        ]

        # Add recent tips preview (last 3)
        if recent_tips:
            lines.append(f"\n🕒 Recent tips: {', '.join(recent_tips[:3])}")

        for mk, st in sorted(by_market.items()):
            if st["graded"] == 0: 
                continue
            a = 100.0 * st["wins"] / st["graded"]
            roi = ""
            if mk in roi_by_market and roi_by_market[mk]["stake"] > 0:
                roi_val = 100.0 * roi_by_market[mk]["pnl"] / roi_by_market[mk]["stake"]
                roi = f" • ROI {roi_val:+.1f}%"
            lines.append(f"• {escape(mk)} — {st['wins']}/{st['graded']} ({a:.1f}%){roi}")

        msg = "\n".join(lines)

    send_telegram(msg)
    log.info("[DIGEST] Sent daily digest with %d tips, %d graded", total, graded)
    return msg

# ───────── Prematch pipeline ─────────
def save_prematch_snapshot(fx: dict, feat: Dict[str, float]) -> None:
    fixture = fx.get("fixture") or {}
    fid = int(fixture.get("id") or 0)
    if not fid:
        return
    now = int(time.time())
    payload = json.dumps({"feat": {k: float(feat.get(k, 0.0) or 0.0) for k in feat.keys()}},
                         separators=(",", ":"), ensure_ascii=False)
    with db_conn() as c:
        c.execute(
            "INSERT INTO prematch_snapshots(match_id, created_ts, payload) "
            "VALUES (%s,%s,%s) "
            "ON CONFLICT (match_id) DO UPDATE SET created_ts=EXCLUDED.created_ts, payload=EXCLUDED.payload",
            (fid, now, payload)
        )

def snapshot_odds_for_fixtures(fixtures: List[int]) -> int:
    """Odds snapshot without sleep time check"""
    wrote = 0
    now = int(time.time())
    for fid in fixtures:
        try:
            od = fetch_odds(fid)
            rows = []
            for mk, sides in (od or {}).items():
                for sel, payload in (sides or {}).items():
                    o = float((payload or {}).get("odds") or 0)
                    b = (payload or {}).get("book") or "Book"
                    if o <= 0: 
                        continue
                    rows.append((fid, now, mk, sel, o, b))
            if not rows:
                continue
            with db_conn() as c:
                c.cur.executemany(
                    "INSERT INTO odds_history(match_id,captured_ts,market,selection,odds,book) "
                    "VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                    rows
                )
                wrote += len(rows)
        except Exception:
            continue
    return wrote

def _today_fixture_ids() -> List[int]:
    fixtures = _collect_todays_prematch_fixtures()
    return [int(((fx.get('fixture') or {}).get('id') or 0)) for fx in fixtures if int(((fx.get('fixture') or {}).get('id') or 0))]

def fetch_lineup_and_save(fid: int) -> bool:
    js = api_get_with_sleep(f"{BASE_URL}/fixtures/lineups", {"fixture": fid}) or {}
    arr = js.get("response") or []
    if not arr:
        return False
    with db_conn() as c:
        c.execute(
            "INSERT INTO lineups(match_id,created_ts,payload) VALUES (%s,%s,%s) "
            "ON CONFLICT (match_id) DO UPDATE SET created_ts=EXCLUDED.created_ts, payload=EXCLUDED.payload",
            (int(fid), int(time.time()), json.dumps(arr, separators=(",", ":")))
        )
    return True

def extract_prematch_features(f: dict) -> Optional[Dict[str, float]]:
    home_id = ((f.get("teams") or {}).get("home") or {}).get("id")
    away_id = ((f.get("teams") or {}).get("away") or {}).get("id")
    home = ((f.get("teams") or {}).get("home") or {}).get("name", "")
    away = ((f.get("teams") or {}).get("away") or {}).get("name", "")
    fid = (f.get("fixture") or {}).get("id")
    feat = {"fid": float(fid or 0)}

    recent_h = _api_last_fixtures(home_id, 5)
    recent_a = _api_last_fixtures(away_id, 5)
    h2h = _api_h2h(home_id, away_id, 5)

    missing_reasons = []

    if recent_h:
        feat["avg_goals_h"] = np.mean([(m.get("goals") or {}).get("home", 0) for m in recent_h])
    else:
        missing_reasons.append("no_recent_home_fixtures")

    if recent_a:
        feat["avg_goals_a"] = np.mean([(m.get("goals") or {}).get("away", 0) for m in recent_a])
    else:
        missing_reasons.append("no_recent_away_fixtures")

    if h2h:
        feat["avg_goals_h2h"] = np.mean([
            (m.get("goals") or {}).get("home", 0) + (m.get("goals") or {}).get("away", 0)
            for m in h2h
        ])
    else:
        missing_reasons.append("no_h2h_data")

    try:
        dts_h = [datetime.fromisoformat((m.get("fixture") or {}).get("date", "")) for m in recent_h if m.get("fixture")]
        dts_a = [datetime.fromisoformat((m.get("fixture") or {}).get("date", "")) for m in recent_a if m.get("fixture")]
        if dts_h:
            feat["rest_days_h"] = (datetime.now(tz=TZ_UTC) - max(dts_h).astimezone(TZ_UTC)).days
        else:
            missing_reasons.append("missing_home_dates")
        if dts_a:
            feat["rest_days_a"] = (datetime.now(tz=TZ_UTC) - max(dts_a).astimezone(TZ_UTC)).days
        else:
            missing_reasons.append("missing_away_dates")
    except Exception as e:
        missing_reasons.append(f"date_parse_error: {str(e)}")

    # Decide if the feature dict is acceptable
    min_features_required = 3
    if len(feat) < min_features_required:
        log.warning(f"[FEATURES_SKIPPED] {home} vs {away} (fid={fid}) - reasons: {missing_reasons}")
        return None

    return feat

def prematch_scan_save() -> int:
    """Prematch scan without sleep time check"""
    fixtures = _collect_todays_prematch_fixtures()
    if not fixtures:
        return 0

    saved = 0
    for fx in fixtures:
        fixture = fx.get("fixture") or {}
        lg = fx.get("league") or {}
        teams = fx.get("teams") or {}

        home = (teams.get("home") or {}).get("name", "")
        away = (teams.get("away") or {}).get("name", "")
        league_id = int(lg.get("id") or 0)
        league = f"{lg.get('country','')} - {lg.get('name','')}".strip(" -")
        fid = int(fixture.get("id") or 0)

        feat = extract_prematch_features(fx)
        if not fid or not feat:
            continue

        try:
            save_prematch_snapshot(fx, feat)
        except Exception:
            pass

        candidates: List[Tuple[str, str, float]] = []

        # PRE OU
        for line in OU_LINES:
            mdl = load_model_from_settings(f"PRE_OU_{_fmt_line(line)}")
            if not mdl:
                continue
            p = predict_from_model(mdl, feat)
            mk = f"Over/Under {_fmt_line(line)}"
            thr = _get_market_threshold_pre(mk)
            if p * 100.0 >= thr:
                candidates.append((f"PRE {mk}", f"Over {_fmt_line(line)} Goals", p))
            q = 1.0 - p
            if q * 100.0 >= thr:
                candidates.append((f"PRE {mk}", f"Under {_fmt_line(line)} Goals", q))

        # PRE BTTS
        mdl = load_model_from_settings("PRE_BTTS_YES")
        if mdl:
            p = predict_from_model(mdl, feat)
            thr = _get_market_threshold_pre("BTTS")
            if p * 100.0 >= thr:
                candidates.append(("PRE BTTS", "BTTS: Yes", p))
            q = 1.0 - p
            if q * 100.0 >= thr:
                candidates.append(("PRE BTTS", "BTTS: No", q))

        # PRE 1X2 (draw suppressed)
        mh, ma = load_model_from_settings("PRE_WLD_HOME"), load_model_from_settings("PRE_WLD_AWAY")
        if mh and ma:
            ph = predict_from_model(mh, feat)
            pa = predict_from_model(ma, feat)
            s = max(EPS, ph + pa)
            ph, pa = ph / s, pa / s
            thr = _get_market_threshold_pre("1X2")
            if ph * 100.0 >= thr:
                candidates.append(("PRE 1X2", "Home Win", ph))
            if pa * 100.0 >= thr:
                candidates.append(("PRE 1X2", "Away Win", pa))

        if not candidates:
            continue

        candidates.sort(key=lambda x: x[2], reverse=True)
        base_now = int(time.time())
        per_match = 0

        for idx, (mk, sug, prob) in enumerate(candidates):
            if sug not in ALLOWED_SUGGESTIONS:
                continue
            if per_match >= max(1, PREDICTIONS_PER_MATCH):
                break

            pass_odds, odds, book, _ = _price_gate(mk.replace("PRE ", ""), sug, fid)
            if not pass_odds:
                continue

            ev_pct = None
            if odds is not None:
                edge = _ev(prob, odds)
                ev_bps = int(round(edge * 10000))
                ev_pct = round(edge * 100.0, 1)
                if int(round(edge * 10000)) < EDGE_MIN_BPS:
                    continue
            else:
                continue

            created_ts = base_now + idx
            raw = float(prob)
            pct = round(raw * 100.0, 1)

            with db_conn() as c2:
                c2.execute(
                    "INSERT INTO tips(match_id,league_id,league,home,away,market,suggestion,confidence,confidence_raw,"
                    "score_at_tip,minute,created_ts,odds,book,ev_pct,sent_ok) "
                    "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,'0-0',0,%s,%s,%s,%s,0)",
                    (
                        fid, league_id, league, home, away, mk, sug,
                        float(pct), raw, created_ts,
                        (float(odds) if odds is not None else None),
                        (book or None),
                        (float(ev_pct) if ev_pct is not None else None),
                    ),
                )
            saved += 1
            per_match += 1

    log.info("[PREMATCH] saved=%d", saved)
    return saved

# ───────── MOTD ─────────
MOTD_MIN_EV_BPS = int(os.getenv("MOTD_MIN_EV_BPS", "0"))

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

def send_match_of_the_day() -> bool:
    """Send Match of the Day without sleep time check"""
    if os.getenv("MOTD_PREDICT", "1") in ("0", "false", "False", "no", "NO"):
        log.info("[MOTD] MOTD disabled by configuration")
        return send_telegram("🏅 MOTD disabled.")
    
    log.info("[MOTD] Starting Match of the Day selection...")
    
    fixtures = _collect_todays_prematch_fixtures()
    if not fixtures:
        log.warning("[MOTD] No fixtures found for today")
        return send_telegram("🏅 Match of the Day: no fixtures today.")
    
    log.info("[MOTD] Found %d fixtures for today", len(fixtures))

    # Filter by league IDs if specified
    if MOTD_LEAGUE_IDS:
        fixtures = [f for f in fixtures if int(((f.get("league") or {}).get("id") or 0)) in MOTD_LEAGUE_IDS]
        log.info("[MOTD] After league filtering: %d fixtures", len(fixtures))
        if not fixtures:
            return send_telegram("🏅 Match of the Day: no fixtures in configured leagues.")

    best_candidate = None
    best_score = 0.0

    for fx in fixtures:
        try:
            fixture = fx.get("fixture") or {}
            lg = fx.get("league") or {}
            teams = fx.get("teams") or {}
            fid = int((fixture.get("id") or 0))

            if not fid:
                continue

            home = (teams.get("home") or {}).get("name", "")
            away = (teams.get("away") or {}).get("name", "")
            league = f"{lg.get('country','')} - {lg.get('name','')}".strip(" -")
            kickoff_txt = _kickoff_berlin((fixture.get("date") or ""))

            log.debug("[MOTD] Processing: %s vs %s (%s)", home, away, league)

            # Extract pre-match features
            feat = extract_prematch_features(fx)
            if not feat:
                log.debug("[MOTD] No features for %s vs %s", home, away)
                continue

            candidates = []

            # PRE Over/Under markets
            for line in OU_LINES:
                mdl = load_model_from_settings(f"PRE_OU_{_fmt_line(line)}")
                if not mdl: 
                    continue
                    
                p = predict_from_model(mdl, feat)
                mk = f"Over/Under {_fmt_line(line)}"
                thr = _get_market_threshold_pre(mk)
                
                # Over candidate
                if p * 100.0 >= max(thr, MOTD_CONF_MIN):
                    candidates.append((mk, f"Over {_fmt_line(line)} Goals", p, home, away, league, kickoff_txt, fid))
                
                # Under candidate  
                q = 1.0 - p
                if q * 100.0 >= max(thr, MOTD_CONF_MIN):
                    candidates.append((mk, f"Under {_fmt_line(line)} Goals", q, home, away, league, kickoff_txt, fid))

            # PRE BTTS
            mdl = load_model_from_settings("PRE_BTTS_YES")
            if mdl:
                p = predict_from_model(mdl, feat)
                thr = _get_market_threshold_pre("BTTS")
                if p * 100.0 >= max(thr, MOTD_CONF_MIN):
                    candidates.append(("BTTS", "BTTS: Yes", p, home, away, league, kickoff_txt, fid))
                q = 1.0 - p
                if q * 100.0 >= max(thr, MOTD_CONF_MIN):
                    candidates.append(("BTTS", "BTTS: No", q, home, away, league, kickoff_txt, fid))

            # PRE 1X2 (draw suppressed)
            mh = load_model_from_settings("PRE_WLD_HOME")
            ma = load_model_from_settings("PRE_WLD_AWAY")
            if mh and ma:
                ph = predict_from_model(mh, feat)
                pa = predict_from_model(ma, feat)
                s = max(EPS, ph + pa)
                ph, pa = ph / s, pa / s
                thr = _get_market_threshold_pre("1X2")
                if ph * 100.0 >= max(thr, MOTD_CONF_MIN):
                    candidates.append(("1X2", "Home Win", ph, home, away, league, kickoff_txt, fid))
                if pa * 100.0 >= max(thr, MOTD_CONF_MIN):
                    candidates.append(("1X2", "Away Win", pa, home, away, league, kickoff_txt, fid))

            if not candidates:
                log.debug("[MOTD] No candidates for %s vs %s", home, away)
                continue

            # Evaluate each candidate
            for mk, sug, prob, home, away, league, kickoff_txt, fid in candidates:
                prob_pct = prob * 100.0
                
                # Check odds and EV
                pass_odds, odds, book, _ = _price_gate(mk, sug, fid)
                if not pass_odds:
                    log.debug("[MOTD] Odds gate failed for %s: %s", sug, home)
                    continue

                ev_pct = None
                if odds is not None:
                    edge = _ev(prob, odds)
                    ev_bps = int(round(edge * 10000))
                    ev_pct = round(edge * 100.0, 1)
                    if MOTD_MIN_EV_BPS > 0 and ev_bps < MOTD_MIN_EV_BPS:
                        log.debug("[MOTD] EV too low for %s: %d bps", sug, ev_bps)
                        continue
                else:
                    log.debug("[MOTD] No odds for %s: %s", sug, home)
                    continue

                # Score candidate (confidence + EV)
                candidate_score = prob_pct + (ev_pct or 0)
                
                log.info("[MOTD] Candidate: %s - %s vs %s - %.1f%% confidence, %.1f%% EV, score: %.1f", 
                        sug, home, away, prob_pct, ev_pct, candidate_score)

                if best_candidate is None or candidate_score > best_score:
                    best_candidate = (prob_pct, sug, home, away, league, kickoff_txt, odds, book, ev_pct)
                    best_score = candidate_score

        except Exception as e:
            log.exception("[MOTD] Error processing fixture: %s", e)
            continue

    if not best_candidate:
        log.info("[MOTD] No suitable match found for MOTD")
        return send_telegram("🏅 Match of the Day: no prematch pick met thresholds today.")

    # Send the best candidate
    prob_pct, sug, home, away, league, kickoff_txt, odds, book, ev_pct = best_candidate
    
    log.info("[MOTD] Selected: %s vs %s - %s (%.1f%%)", home, away, sug, prob_pct)
    
    message = _format_motd_message(home, away, league, kickoff_txt, sug, prob_pct, odds, book, ev_pct)
    success = send_telegram(message)
    
    if success:
        log.info("[MOTD] Successfully sent MOTD")
    else:
        log.error("[MOTD] Failed to send MOTD message")
        
    return success

# ───────── Auto-train / Auto-tune ─────────
def auto_train_job():
    if not TRAIN_ENABLE:
        return send_telegram("🤖 Training skipped: TRAIN_ENABLE=0")
    send_telegram("🤖 Training started.")
    try:
        res = train_models() or {}
        ok = bool(res.get("ok"))
        if not ok:
            reason = res.get("reason") or res.get("error") or "unknown"
            return send_telegram(f"⚠️ Training finished: <b>SKIPPED</b>\nReason: {escape(str(reason))}")

        trained = [k for k, v in (res.get("trained") or {}).items() if v]
        keys = [
            "BTTS", "Over/Under 2.5", "Over/Under 3.5", "1X2",
            "PRE BTTS", "PRE Over/Under 2.5", "PRE Over/Under 3.5", "PRE 1X2",
        ]
        thr_lines = []
        for k in keys:
            try:
                v = get_setting_cached(f"conf_threshold:{k}")
                if v is not None:
                    thr_lines.append(f"{escape(k)}: {float(v):.1f}%")
            except Exception:
                continue

        lines = ["🤖 <b>Model training OK</b>"]
        if trained:
            lines.append("• Trained: " + ", ".join(sorted(trained)))
        if thr_lines:
            lines.append("• Thresholds: " + "  |  ".join(thr_lines))
        send_telegram("\n".join(lines))
    except Exception as e:
        log.exception("[TRAIN] job failed: %s", e)
        send_telegram(f"❌ Training <b>FAILED</b>\n{escape(str(e))}")

def _apply_tune_thresholds(days: int = 14) -> Dict[str, float]:
    PREC_TOL = float(os.getenv("APPLY_TUNE_PREC_TOL", "0.03"))
    cutoff = int(time.time()) - days * 24 * 3600
    with db_conn() as c:
        rows = c.execute(
            """
            SELECT t.market,
                   t.suggestion,
                   COALESCE(t.confidence_raw, t.confidence/100.0) AS prob,
                   t.odds,
                   r.final_goals_h, r.final_goals_a, r.btts_yes
            FROM tips t
            JOIN match_results r ON r.match_id = t.match_id
            WHERE t.created_ts >= %s
              AND t.suggestion <> 'HARVEST'
              AND t.sent_ok = 1
              AND t.odds IS NOT NULL
            """,
            (cutoff,),
        ).fetchall()
    if not rows:
        send_telegram("🔧 Apply-tune: no labeled tips with odds in window.")
        return {}
    by: dict[str, list[tuple[float, int, float]]] = {}
    for (mk, sugg, prob, odds, gh, ga, btts) in rows:
        try:
            prob = float(prob or 0.0); odds = float(odds or 0.0)
        except Exception: continue
        res = _tip_outcome_for_result(sugg, {"final_goals_h": gh, "final_goals_a": ga, "btts_yes": btts})
        if res is None: continue
        y = int(res)
        if not (1.01 <= odds <= MAX_ODDS_ALL): continue
        by.setdefault(mk, []).append((prob, y, odds))
    if not by:
        send_telegram("🔧 Auto-tune: nothing to tune after filtering.")
        return {}
    target_precision = float(os.getenv("TARGET_PRECISION", "0.60"))
    min_preds = int(os.getenv("THRESH_MIN_PREDICTIONS", "25"))
    min_thr = float(os.getenv("MIN_THRESH", "55"))
    max_thr = float(os.getenv("MAX_THRESH", "85"))
    tuned: Dict[str, float] = {}

    def _eval_threshold(items: list[tuple[float, int, float]], thr_prob: float) -> tuple[int, float, float]:
        sel = [(p, y, o) for (p, y, o) in items if p >= thr_prob]
        n = len(sel)
        if n == 0: return 0, 0.0, 0.0
        wins = sum(y for (_, y, _) in sel); prec = wins / n
        roi = sum((y * (odds - 1.0) - (1 - y)) for (_, y, odds) in sel) / n
        return n, float(prec), float(roi)

    for mk, items in by.items():
        if len(items) < min_preds: continue
        candidates_pct = list(np.arange(min_thr, max_thr + 1e-9, 1.0))
        best = None
        feasible_any = False
        for thr_pct in candidates_pct:
            thr_prob = float(thr_pct / 100.0)
            n, prec, roi = _eval_threshold(items, thr_prob)
            if n < min_preds: continue
            if prec >= target_precision:
                feasible_any = True
                score = (roi, prec, n)
                if (best is None) or (score > (best[0], best[1], best[2])):
                    best = (roi, prec, n, thr_pct)
        if not feasible_any:
            for thr_pct in candidates_pct:
                thr_prob = float(thr_pct / 100.0)
                n, prec, roi = _eval_threshold(items, thr_prob)
                if n < min_preds: continue
                if (prec >= max(0.0, target_precision - PREC_TOL)) and (roi > 0.0):
                    score = (roi, prec, n)
                    if (best is None) or (score > (best[0], best[1], best[2])):
                        best = (roi, prec, n, thr_pct)
        if best is None:
            fallback = None
            for thr_pct in candidates_pct:
                thr_prob = float(thr_pct / 100.0)
                n, prec, roi = _eval_threshold(items, thr_prob)
                if n < min_preds: continue
                score = (prec, n, roi)
                if (fallback is None) or (score > (fallback[0], fallback[1], fallback[2])):
                    fallback = (prec, n, roi, thr_pct)
            if fallback is not None:
                tuned[mk] = float(fallback[3])
        else:
            tuned[mk] = float(best[3])

    if tuned:
        for mk, pct in tuned.items():
            set_setting(f"conf_threshold:{mk}", f"{pct:.2f}")
            _SETTINGS_CACHE.invalidate(f"conf_threshold:{mk}")
        lines = ["🔧 Auto-tune (ROI-aware) updated thresholds:"]
        for mk, pct in sorted(tuned.items()):
            lines.append(f"• {mk}: {pct:.1f}%")
        send_telegram("\n".join(lines))
    else:
        send_telegram("🔧 Auto-tune (ROI-aware): no markets met minimum data.")
    return tuned

def auto_tune_thresholds(days: int = 14) -> Dict[str, float]:
    return _apply_tune_thresholds(days)

# ───────── Retry unsent tips ─────────
def retry_unsent_tips(minutes: int = 30, limit: int = 200) -> int:
    cutoff = int(time.time()) - minutes*60
    retried = 0
    with db_conn() as c:
        rows = c.execute(
            "SELECT match_id,league,home,away,market,suggestion,confidence,confidence_raw,score_at_tip,minute,created_ts,odds,book,ev_pct "
            "FROM tips WHERE sent_ok=0 AND created_ts >= %s ORDER BY created_ts ASC LIMIT %s",
            (cutoff, limit)
        ).fetchall()

        for (mid, league, home, away, market, sugg, conf, conf_raw, score, minute, cts, odds, book, ev_pct) in rows:
            ok = send_telegram(_format_tip_message(home, away, league, int(minute), score, sugg, float(conf), {}, odds, book, ev_pct))
            if ok:
                c.execute("UPDATE tips SET sent_ok=1 WHERE match_id=%s AND created_ts=%s", (mid, cts))
                retried += 1
    if retried:
        log.info("[RETRY] resent %d", retried)
    return retried

# ───────── Shutdown Handlers ─────────
def shutdown_handler(signum=None, frame=None):
    log.info("Received shutdown signal, cleaning up...")
    ShutdownManager.request_shutdown()
    if POOL:
        try:
            POOL.closeall()
        except Exception as e:
            log.warning("Error closing pool during shutdown: %s", e)
    sys.exit(0)

def register_shutdown_handlers():
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    atexit.register(shutdown_handler)

# ───────── Scheduler ─────────
_scheduler_started=False

def _run_with_pg_lock(lock_key: int, fn, *a, **k):
    try:
        with db_conn() as c:
            got=c.execute("SELECT pg_try_advisory_lock(%s)",(lock_key,)).fetchone()[0]
            if not got: log.info("[LOCK %s] busy; skipped.", lock_key); return None
            try: return fn(*a,**k)
            finally: c.execute("SELECT pg_advisory_unlock(%s)",(lock_key,))
    except Exception as e:
        log.exception("[LOCK %s] failed: %s", lock_key, e); return None

def _start_scheduler_once():
    global _scheduler_started
    if _scheduler_started or not RUN_SCHEDULER:
        return
    try:
        sched = BackgroundScheduler(timezone=TZ_UTC)
        # core jobs
        sched.add_job(lambda: _run_with_pg_lock(1001, production_scan),
                      "interval", seconds=SCAN_INTERVAL_SEC, id="scan", max_instances=1, coalesce=True)
        sched.add_job(lambda: _run_with_pg_lock(1002, backfill_results_for_open_matches, 400),
                      "interval", minutes=BACKFILL_EVERY_MIN, id="backfill", max_instances=1, coalesce=True)

        if DAILY_ACCURACY_DIGEST_ENABLE:
            sched.add_job(lambda: _run_with_pg_lock(1003, daily_accuracy_digest),
                          CronTrigger(hour=int(os.getenv("DAILY_ACCURACY_HOUR", "3")),
                                      minute=int(os.getenv("DAILY_ACCURACY_MINUTE", "6")),
                                      timezone=BERLIN_TZ),
                          id="digest", max_instances=1, coalesce=True)

        if os.getenv("MOTD_PREDICT", "1") not in ("0", "false", "False", "no", "NO"):
            sched.add_job(lambda: _run_with_pg_lock(1004, send_match_of_the_day),
                          CronTrigger(hour=int(os.getenv("MOTD_HOUR", "19")),
                                      minute=int(os.getenv("MOTD_MINUTE", "15")),
                                      timezone=BERLIN_TZ),
                          id="motd", max_instances=1, coalesce=True)

        if TRAIN_ENABLE:
            sched.add_job(lambda: _run_with_pg_lock(1005, auto_train_job),
                          CronTrigger(hour=TRAIN_HOUR_UTC, minute=TRAIN_MINUTE_UTC, timezone=TZ_UTC),
                          id="train", max_instances=1, coalesce=True)

        if AUTO_TUNE_ENABLE:
            sched.add_job(lambda: _run_with_pg_lock(1006, auto_tune_thresholds, 14),
                          CronTrigger(hour=4, minute=7, timezone=TZ_UTC),
                          id="auto_tune", max_instances=1, coalesce=True)

        # retry unsent
        sched.add_job(lambda: _run_with_pg_lock(1007, retry_unsent_tips, 30, 200),
                      "interval", minutes=10, id="retry", max_instances=1, coalesce=True)

        # periodic odds snapshots
        sched.add_job(lambda: _run_with_pg_lock(1008, lambda: snapshot_odds_for_fixtures(_today_fixture_ids())),
                      "interval", seconds=180, id="odds_snap", max_instances=1, coalesce=True)

        sched.start()
        _scheduler_started = True
        send_telegram("🚀 goalsniper AI mode (in-play + prematch) started.")
        log.info("[SCHED] started (scan=%ss)", SCAN_INTERVAL_SEC)

    except Exception as e:
        log.exception("[SCHED] failed: %s", e)

# ───────── Admin endpoints ─────────
def _require_admin():
    key=request.headers.get("X-API-Key") or request.args.get("key") or ((request.json or {}).get("key") if request.is_json else None)
    if not ADMIN_API_KEY or key != ADMIN_API_KEY: abort(401)

@app.route("/")
def root(): return jsonify({"ok": True, "name": "goalsniper", "mode": "STREAMLINED", "scheduler": RUN_SCHEDULER})

@app.route("/health")
def health():
    try:
        with db_conn() as c:
            n = c.execute("SELECT COUNT(*) FROM tips").fetchone()[0]
        
        # Check API connectivity
        api_ok = False
        try:
            test_resp = api_get_with_sleep(FOOTBALL_API_URL, {"live": "all"}, timeout=5)
            api_ok = test_resp is not None
        except:
            pass
            
        return jsonify({
            "ok": True, 
            "db": "ok", 
            "tips_count": int(n),
            "api_connected": api_ok,
            "scheduler_running": _scheduler_started,
            "timestamp": time.time()
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/metrics")
def metrics():
    try:
        return jsonify({"ok": True, "metrics": METRICS})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/init-db", methods=["POST"])
def http_init_db(): _require_admin(); init_db(); return jsonify({"ok": True})

@app.route("/admin/scan", methods=["POST","GET"])
def http_scan(): _require_admin(); s,l=production_scan(); return jsonify({"ok": True, "saved": s, "live_seen": l})

@app.route("/admin/backfill-results", methods=["POST","GET"])
def http_backfill(): _require_admin(); n=backfill_results_for_open_matches(400); return jsonify({"ok": True, "updated": n})

@app.route("/admin/train", methods=["POST","GET"])
def http_train():
    _require_admin()
    if not TRAIN_ENABLE: 
        return jsonify({"ok": False, "reason": "training disabled"}), 400
    try: 
        out=train_models()
        return jsonify({"ok": True, "result": out})
    except Exception as e:
        log.exception("train_models failed: %s", e)
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/admin/train-notify", methods=["POST","GET"])
def http_train_notify(): _require_admin(); auto_train_job(); return jsonify({"ok": True})

@app.route("/admin/digest", methods=["POST","GET"])
def http_digest(): _require_admin(); msg=daily_accuracy_digest(); return jsonify({"ok": True, "sent": bool(msg)})

@app.route("/admin/auto-tune", methods=["POST","GET"])
def http_auto_tune(): _require_admin(); tuned=auto_tune_thresholds(14); return jsonify({"ok": True, "tuned": tuned})

@app.route("/admin/retry-unsent", methods=["POST","GET"])
def http_retry_unsent(): _require_admin(); n=retry_unsent_tips(30,200); return jsonify({"ok": True, "resent": n})

@app.route("/admin/prematch-scan", methods=["POST","GET"])
def http_prematch_scan(): _require_admin(); saved=prematch_scan_save(); return jsonify({"ok": True, "saved": int(saved)})

@app.route("/admin/motd", methods=["POST","GET"])
def http_motd():
    _require_admin(); ok = send_match_of_the_day(); return jsonify({"ok": bool(ok)})

@app.route("/admin/motd-test", methods=["POST", "GET"])
def http_motd_test():
    """Test MOTD manually"""
    _require_admin()
    log.info("[MOTD-TEST] Manual MOTD trigger")
    
    try:
        ok = send_match_of_the_day()
        return jsonify({"ok": bool(ok), "message": "MOTD test completed"})
    except Exception as e:
        log.exception("[MOTD-TEST] Failed: %s", e)
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/settings/<key>", methods=["GET","POST"])
def http_settings(key: str):
    _require_admin()
    if request.method=="GET":
        val=get_setting_cached(key); return jsonify({"ok": True, "key": key, "value": val})
    val=(request.get_json(silent=True) or {}).get("value")
    if val is None: abort(400)
    set_setting(key, str(val)); _SETTINGS_CACHE.invalidate(key); invalidate_model_caches_for_key(key)
    return jsonify({"ok": True})

@app.route("/tips/latest")
def http_latest():
    limit=int(request.args.get("limit","50"))
    with db_conn() as c:
        rows=c.execute("SELECT match_id,league,home,away,market,suggestion,confidence,confidence_raw,score_at_tip,minute,created_ts,odds,book,ev_pct "
                       "FROM tips WHERE suggestion<>'HARVEST' ORDER BY created_ts DESC LIMIT %s",(max(1,min(500,limit)),)).fetchall()
    tips=[]
    for r in rows:
        tips.append({"match_id":int(r[0]),"league":r[1],"home":r[2],"away":r[3],"market":r[4],"suggestion":r[5],
                     "confidence":float(r[6]),"confidence_raw":(float(r[7]) if r[7] is not None else None),
                     "score_at_tip":r[8],"minute":int(r[9]),"created_ts":int(r[10]),
                     "odds": (float(r[11]) if r[11] is not None else None), "book": r[12], "ev_pct": (float(r[13]) if r[13] is not None else None)})
    return jsonify({"ok": True, "tips": tips})

@app.route("/telegram/webhook/<secret>", methods=["POST"])
def telegram_webhook(secret: str):
    if (WEBHOOK_SECRET or "") != secret: abort(403)
    update=request.get_json(silent=True) or {}
    try:
        msg=(update.get("message") or {}).get("text") or ""
        if msg.startswith("/start"): send_telegram("👋 goalsniper bot (STREAMLINED mode) is online.")
        elif msg.startswith("/digest"): daily_accuracy_digest()
        elif msg.startswith("/motd"): send_match_of_the_day()
        elif msg.startswith("/scan"):
            parts=msg.split()
            if len(parts)>1 and ADMIN_API_KEY and parts[1]==ADMIN_API_KEY:
                s,l=production_scan(); send_telegram(f"🔁 Scan done. Saved: {s}, Live seen: {l}")
            else: send_telegram("🔒 Admin key required.")
    except Exception as e:
        log.warning("telegram webhook parse error: %s", e)
    return jsonify({"ok": True})

# ───────── Boot ─────────
def _on_boot():
    register_shutdown_handlers()
    validate_config()
    _init_pool()
    init_db()
    set_setting("boot_ts", str(int(time.time())))
    _start_scheduler_once()

# Call _on_boot() to initialize the application
_on_boot()

if __name__ == "__main__":
    app.run(host=os.getenv("HOST","0.0.0.0"), port=int(os.getenv("PORT","8080")))
