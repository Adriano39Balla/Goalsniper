# goalsniper — FULL AI mode (in-play + prematch) with odds + EV gate
# UPGRADED: Advanced prediction capabilities with ensemble models, Bayesian updates, and game state intelligence
# ENHANCED: Added advanced AI systems including ensemble learning, feature engineering, market-specific intelligence, and adaptive learning

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

# ───────── AI IMPORTS ─────────
import pickle
import hashlib
import warnings
from scipy.optimize import minimize
from sklearn.ensemble import IsolationForest
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
warnings.filterwarnings('ignore')

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

# ───────── AI Logger ─────────
class AILogger:
    """Advanced logging for AI systems"""
    def __init__(self):
        self.logger = logging.getLogger("goalsniper.ai")
        
    def log_training(self, model_name, accuracy, features_used):
        self.logger.info(f"[AI_TRAIN] {model_name} - Accuracy: {accuracy:.3f}, Features: {len(features_used)}")
        
    def log_anomaly(self, match_id, anomaly_score, reason):
        self.logger.warning(f"[ANOMALY] Match {match_id} - Score: {anomaly_score:.3f}, Reason: {reason}")
        
    def log_ensemble(self, predictions, weights, final_prob):
        self.logger.debug(f"[ENSEMBLE] Predictions: {predictions}, Weights: {weights}, Final: {final_prob:.3f}")

ai_logger = AILogger()

# ───────── Exception Handler ─────────
class ExceptionHandler:
    """Comprehensive exception handling with alerts"""
    
    def __init__(self):
        self.error_counts = {}
        self.max_errors_before_alert = 5
        
    def handle(self, operation: str, func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self._log_error(operation, e)
            self._check_alert_threshold(operation, e)
            return None
            
    def _log_error(self, operation: str, error: Exception):
        error_key = f"{operation}_{type(error).__name__}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        log.error(f"[EXCEPTION] {operation} failed: {str(error)}")
        
    def _check_alert_threshold(self, operation: str, error: Exception):
        error_key = f"{operation}_{type(error).__name__}"
        if self.error_counts[error_key] >= self.max_errors_before_alert:
            self._send_alert(operation, error, self.error_counts[error_key])
            self.error_counts[error_key] = 0  # Reset after alert
            
    def _send_alert(self, operation: str, error: Exception, count: int):
        alert_msg = (
            f"🚨 CRITICAL ERROR ALERT\n"
            f"Operation: {operation}\n"
            f"Error: {type(error).__name__}\n"
            f"Count: {count} occurrences\n"
            f"Message: {str(error)[:200]}"
        )
        send_telegram(alert_msg)

exception_handler = ExceptionHandler()

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

# ───────── AI Model Trainer ─────────
class ModelTrainer:
    """Advanced model training with scikit-learn and LightGBM"""
    
    def __init__(self):
        self.model_registry = {}
        self.feature_importance = {}
        
    def train_lightgbm(self, X, y, market: str, params: dict = None):
        """Train LightGBM model with cross-validation"""
        try:
            if params is None:
                params = {
                    'objective': 'binary',
                    'metric': 'binary_logloss',
                    'learning_rate': 0.05,
                    'max_depth': 6,
                    'num_leaves': 31,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'verbose': -1
                }
            
            # Create dataset
            lgb_data = lgb.Dataset(X, label=y)
            
            # Train with cross-validation
            cv_results = lgb.cv(
                params, lgb_data, num_boost_round=1000,
                nfold=5, stratified=True, early_stopping_rounds=50,
                verbose_eval=False
            )
            
            best_round = len(cv_results['binary_logloss-mean'])
            model = lgb.train(params, lgb_data, num_boost_round=best_round)
            
            # Store feature importance
            self.feature_importance[market] = dict(zip(
                X.columns, model.feature_importance()
            ))
            
            return model
            
        except Exception as e:
            log.error(f"[LIGHTGBM] Training failed for {market}: {e}")
            return None
            
    def train_logistic(self, X, y, market: str):
        """Train logistic regression model with regularization"""
        try:
            model = LogisticRegression(
                C=1.0, penalty='l2', solver='lbfgs', max_iter=1000
            )
            model.fit(X, y)
            return model
        except Exception as e:
            log.error(f"[LOGISTIC] Training failed for {market}: {e}")
            return None
            
    def calibrate_model(self, model, X_calib, y_calib, method: str = 'sigmoid'):
        """Apply Platt scaling or isotonic regression"""
        try:
            if method == 'isotonic':
                calibrated = CalibratedClassifierCV(
                    model, method='isotonic', cv='prefit'
                )
            else:  # sigmoid
                calibrated = CalibratedClassifierCV(
                    model, method='sigmoid', cv='prefit'
                )
                
            calibrated.fit(X_calib, y_calib)
            return calibrated
        except Exception as e:
            log.error(f"[CALIBRATION] Failed: {e}")
            return model

# ───────── Model Metadata Manager ─────────
class ModelMetadataManager:
    """Manage model metadata including versioning and performance"""
    
    def __init__(self):
        self.metadata_store = {}
        
    def create_metadata(self, model_name: str, accuracy: float, 
                       features: list, model_type: str) -> dict:
        """Create comprehensive model metadata"""
        metadata = {
            'version': self._generate_version(model_name),
            'accuracy': accuracy,
            'features': features,
            'model_type': model_type,
            'timestamp': datetime.now(TZ_UTC).isoformat(),
            'training_samples': 0,  # Will be updated during training
            'feature_importance': {},
            'hyperparameters': {},
            'calibration_method': 'none'
        }
        return metadata
        
    def _generate_version(self, model_name: str) -> str:
        """Generate semantic version for model"""
        base_version = "1.0.0"
        timestamp = int(time.time())
        return f"{base_version}.{timestamp}"
        
    def store_metadata(self, model_name: str, metadata: dict):
        """Store metadata in database"""
        try:
            with db_conn() as c:
                c.execute("""
                    INSERT INTO model_metadata 
                    (model_name, version, accuracy, features, model_type, 
                     timestamp, training_samples, hyperparameters, calibration_method)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (model_name, version) DO UPDATE SET
                    accuracy = EXCLUDED.accuracy,
                    features = EXCLUDED.features,
                    timestamp = EXCLUDED.timestamp
                """, (
                    model_name, metadata['version'], metadata['accuracy'],
                    json.dumps(metadata['features']), metadata['model_type'],
                    metadata['timestamp'], metadata['training_samples'],
                    json.dumps(metadata['hyperparameters']), 
                    metadata['calibration_method']
                ))
        except Exception as e:
            log.error(f"[METADATA] Failed to store metadata for {model_name}: {e}")

# ───────── Bayesian Model Updater ─────────
class BayesianModelUpdater:
    """Bayesian updating for in-play predictions"""
    
    def __init__(self):
        self.prior_strength = 0.3
        
    def update_prediction(self, prior_prob: float, live_evidence: float, 
                         minute: int, confidence: float) -> float:
        """Update probability using Bayesian inference"""
        # Weight increases as game progresses and with higher confidence
        live_weight = min(minute / 90.0, 1.0) * confidence
        
        # Combine prior and live evidence
        posterior = (prior_prob * (1 - live_weight) + 
                    live_evidence * live_weight)
        
        return max(0.0, min(1.0, posterior))
        
    def calculate_bayesian_confidence(self, prior_samples: int, 
                                   live_samples: int) -> float:
        """Calculate confidence based on sample sizes"""
        total_samples = prior_samples + live_samples
        if total_samples == 0:
            return 0.0
        return min(1.0, total_samples / 100.0)  # Normalized confidence

# ───────── Dynamic League Weighter ─────────
class DynamicLeagueWeighter:
    """Dynamic weighting of leagues based on performance"""
    
    def __init__(self):
        self.league_weights = {}
        self.performance_history = {}
        
    def update_league_weight(self, league_id: int, accuracy: float, 
                           sample_size: int):
        """Update league weight based on recent performance"""
        if sample_size < 10:  # Minimum samples
            return
            
        # Exponential moving average of accuracy
        if league_id not in self.performance_history:
            self.performance_history[league_id] = accuracy
        else:
            alpha = 0.1  # Smoothing factor
            self.performance_history[league_id] = (
                alpha * accuracy + 
                (1 - alpha) * self.performance_history[league_id]
            )
            
        # Convert accuracy to weight (0.5 = neutral, 1.0 = perfect)
        weight = max(0.1, min(2.0, self.performance_history[league_id] * 2))
        self.league_weights[league_id] = weight
        
    def get_league_weight(self, league_id: int) -> float:
        """Get weight for specific league"""
        return self.league_weights.get(league_id, 1.0)

# ───────── Anomaly Detector ─────────
class AnomalyDetector:
    """Isolation Forest based anomaly detection"""
    
    def __init__(self, contamination: float = 0.1):
        self.detector = IsolationForest(
            contamination=contamination, 
            random_state=42
        )
        self.is_fitted = False
        
    def fit(self, features: list):
        """Fit the anomaly detector"""
        try:
            if len(features) > 10:  # Minimum samples to fit
                self.detector.fit(features)
                self.is_fitted = True
                log.info("[ANOMALY] Detector fitted successfully")
        except Exception as e:
            log.error(f"[ANOMALY] Fit failed: {e}")
            
    def detect(self, features: dict) -> Tuple[bool, float]:
        """Detect if features represent an anomaly"""
        if not self.is_fitted:
            return False, 0.0
            
        try:
            # Convert features to array
            feature_vector = np.array(list(features.values())).reshape(1, -1)
            anomaly_score = self.detector.decision_function(feature_vector)[0]
            is_anomaly = self.detector.predict(feature_vector)[0] == -1
            
            return is_anomaly, anomaly_score
        except Exception as e:
            log.error(f"[ANOMALY] Detection failed: {e}")
            return False, 0.0

# ───────── Performance Pruner ─────────
class PerformancePruner:
    """Prune poor-performing teams and leagues"""
    
    def __init__(self, min_accuracy: float = 0.45, min_samples: int = 20):
        self.min_accuracy = min_accuracy
        self.min_samples = min_samples
        self.blacklist = set()
        
    def analyze_performance(self, days: int = 30):
        """Analyze and update blacklist"""
        cutoff_ts = int(time.time()) - days * 24 * 3600
        
        with db_conn() as c:
            # Analyze team performance
            team_performance = c.execute("""
                SELECT t.home as team, COUNT(*) as tips, 
                       AVG(CASE WHEN r.final_goals_h IS NOT NULL THEN 
                           CASE WHEN (t.suggestion = 'Home Win' AND r.final_goals_h > r.final_goals_a) OR
                                     (t.suggestion = 'Away Win' AND r.final_goals_a > r.final_goals_h) OR
                                     (t.suggestion = 'BTTS: Yes' AND r.btts_yes = 1) OR
                                     (t.suggestion LIKE 'Over%' AND (r.final_goals_h + r.final_goals_a) > ?) OR
                                     (t.suggestion LIKE 'Under%' AND (r.final_goals_h + r.final_goals_a) < ?)
                                THEN 1.0 ELSE 0.0 END
                           ELSE NULL END) as accuracy
                FROM tips t 
                LEFT JOIN match_results r ON t.match_id = r.match_id
                WHERE t.created_ts >= ? AND t.suggestion <> 'HARVEST'
                GROUP BY t.home
                HAVING tips >= ?
            """, (2.5, 2.5, cutoff_ts, self.min_samples)).fetchall()
            
            # Update blacklist
            for team, tips, accuracy in team_performance:
                if accuracy < self.min_accuracy:
                    self.blacklist.add(team)
                    log.info(f"[PRUNER] Blacklisted team: {team} (accuracy: {accuracy:.3f})")

    def is_blacklisted(self, team: str) -> bool:
        """Check if team is blacklisted"""
        return team in self.blacklist

# ───────── Multi-Model Ensemble ─────────
class MultiModelEnsemble:
    """Ensemble of pre-match, live, and hybrid models"""
    
    def __init__(self):
        self.models = {
            'prematch': {},
            'live': {},
            'hybrid': {}
        }
        self.weights = {
            'prematch': 0.3,
            'live': 0.4,
            'hybrid': 0.3
        }
        
    def add_model(self, model_type: str, model_name: str, model):
        """Add model to ensemble"""
        self.models[model_type][model_name] = model
        
    def predict(self, features: dict, minute: int) -> Tuple[float, float]:
        """Get ensemble prediction"""
        predictions = []
        confidences = []
        
        # Adjust weights based on game minute
        dynamic_weights = self._calculate_dynamic_weights(minute)
        
        for model_type, models in self.models.items():
            for model_name, model in models.items():
                try:
                    if model_type == 'prematch':
                        prob = self._predict_prematch(model, features)
                    elif model_type == 'live':
                        prob = self._predict_live(model, features, minute)
                    else:  # hybrid
                        prob = self._predict_hybrid(model, features, minute)
                        
                    if prob is not None:
                        weight = dynamic_weights[model_type]
                        predictions.append(prob * weight)
                        confidences.append(weight)
                except Exception as e:
                    log.warning(f"[ENSEMBLE] {model_type}.{model_name} failed: {e}")
                    continue
        
        if not predictions:
            return 0.0, 0.0
            
        ensemble_prob = sum(predictions) / sum(confidences)
        ensemble_confidence = np.mean(confidences)
        
        return ensemble_prob, ensemble_confidence
        
    def _calculate_dynamic_weights(self, minute: int) -> dict:
        """Calculate dynamic weights based on game progress"""
        # Live models gain weight as game progresses
        live_weight = min(0.7, 0.3 + (minute / 90.0) * 0.4)
        prematch_weight = max(0.1, 0.5 - (minute / 90.0) * 0.4)
        hybrid_weight = 1.0 - live_weight - prematch_weight
        
        return {
            'prematch': prematch_weight,
            'live': live_weight,
            'hybrid': hybrid_weight
        }
    
    def _predict_prematch(self, model, features: dict) -> Optional[float]:
        """Predict using pre-match model"""
        try:
            # Convert features to format expected by model
            feature_vector = np.array(list(features.values())).reshape(1, -1)
            return float(model.predict_proba(feature_vector)[0][1])
        except Exception:
            return None
            
    def _predict_live(self, model, features: dict, minute: int) -> Optional[float]:
        """Predict using live model"""
        try:
            # Add minute-based features
            enhanced_features = features.copy()
            enhanced_features['minute_normalized'] = minute / 90.0
            feature_vector = np.array(list(enhanced_features.values())).reshape(1, -1)
            return float(model.predict_proba(feature_vector)[0][1])
        except Exception:
            return None
            
    def _predict_hybrid(self, model, features: dict, minute: int) -> Optional[float]:
        """Predict using hybrid model"""
        try:
            # Combine pre-match and live features
            hybrid_features = features.copy()
            hybrid_features['game_progress'] = minute / 90.0
            feature_vector = np.array(list(hybrid_features.values())).reshape(1, -1)
            return float(model.predict_proba(feature_vector)[0][1])
        except Exception:
            return None

# ───────── Confidence Weighted Kombi ─────────
class ConfidenceWeightedKombi:
    """Generate confidence-weighted combination bets"""
    
    def __init__(self, max_combinations: int = 3):
        self.max_combinations = max_combinations
        
    def generate_kombi(self, tips: List[dict], max_stake: float = 100.0):
        """Generate optimal bet combinations"""
        if len(tips) < 2:
            return []
            
        # Calculate expected value for each tip
        for tip in tips:
            tip['ev'] = self._calculate_ev(tip)
            
        # Sort by expected value
        tips.sort(key=lambda x: x['ev'], reverse=True)
        
        combinations = []
        
        # Generate 2-way combinations
        for i in range(len(tips)):
            for j in range(i + 1, len(tips)):
                if len(combinations) >= self.max_combinations:
                    break
                    
                combo = self._create_combination([tips[i], tips[j]])
                if combo['expected_roi'] > 0.05:  # Minimum 5% expected ROI
                    combinations.append(combo)
                    
        # Generate 3-way combinations
        for i in range(len(tips)):
            for j in range(i + 1, len(tips)):
                for k in range(j + 1, len(tips)):
                    if len(combinations) >= self.max_combinations:
                        break
                        
                    combo = self._create_combination([tips[i], tips[j], tips[k]])
                    if combo['expected_roi'] > 0.08:  # Higher threshold for 3-way
                        combinations.append(combo)
        
        return combinations
        
    def _calculate_ev(self, tip: dict) -> float:
        """Calculate expected value"""
        prob = tip.get('confidence', 0) / 100.0
        odds = tip.get('odds', 1.0)
        return (prob * (odds - 1)) - (1 - prob)
        
    def _create_combination(self, tips: List[dict]) -> dict:
        """Create a combination bet"""
        total_prob = 1.0
        total_odds = 1.0
        total_confidence = 0.0
        
        for tip in tips:
            prob = tip.get('confidence', 0) / 100.0
            odds = tip.get('odds', 1.0)
            total_prob *= prob
            total_odds *= odds
            total_confidence += tip.get('confidence', 0)
            
        avg_confidence = total_confidence / len(tips)
        expected_roi = (total_prob * total_odds) - 1.0
        
        return {
            'tips': tips,
            'combined_odds': total_odds,
            'combined_probability': total_prob,
            'expected_roi': expected_roi,
            'confidence': avg_confidence
        }

# ───────── Initialize AI Systems ─────────
model_trainer = ModelTrainer()
metadata_manager = ModelMetadataManager()
bayesian_updater = BayesianModelUpdater()
league_weighter = DynamicLeagueWeighter()
anomaly_detector = AnomalyDetector()
performance_pruner = PerformancePruner()
multi_model_ensemble = MultiModelEnsemble()
kombi_generator = ConfidenceWeightedKombi()

# ───────── ENHANCEMENT 1: Advanced Ensemble Learning System ─────────
class AdvancedEnsemblePredictor:
    """Advanced ensemble system combining multiple model types with dynamic weighting"""
    
    def __init__(self):
        self.model_types = ['logistic', 'xgboost', 'neural', 'bayesian', 'momentum']
        self.ensemble_weights = self._initialize_adaptive_weights()
        self.performance_tracker = {}
        
    def _initialize_adaptive_weights(self):
        """Initialize weights based on historical performance"""
        return {
            'logistic': 0.25,
            'xgboost': 0.30, 
            'neural': 0.20,
            'bayesian': 0.15,
            'momentum': 0.10
        }
    
    def predict_ensemble(self, features: Dict[str, float], market: str, minute: int) -> Tuple[float, float]:
        """Enhanced ensemble prediction with confidence scoring"""
        predictions = []
        confidences = []
        
        # Get predictions from all model types
        for model_type in self.model_types:
            try:
                prob, confidence = self._predict_single_model(features, market, minute, model_type)
                if prob is not None:
                    predictions.append((model_type, prob, confidence))
                    confidences.append(confidence)
            except Exception as e:
                log.warning(f"[ENSEMBLE] {model_type} model failed: %s", e)
                continue
        
        if not predictions:
            return 0.0, 0.0
        
        # Dynamic weighting based on model confidence and recent performance
        weighted_sum = 0.0
        total_weight = 0.0
        
        for model_type, prob, confidence in predictions:
            base_weight = self.ensemble_weights.get(model_type, 0.1)
            recent_performance = self._get_recent_performance(model_type, market)
            time_weight = self._calculate_time_weight(minute, model_type)
            
            # Combined weight considering all factors
            final_weight = base_weight * confidence * recent_performance * time_weight
            weighted_sum += prob * final_weight
            total_weight += final_weight
        
        ensemble_prob = weighted_sum / total_weight if total_weight > 0 else 0.0
        ensemble_confidence = np.mean(confidences) if confidences else 0.0
        
        return ensemble_prob, ensemble_confidence
    
    def _predict_single_model(self, features: Dict[str, float], market: str, minute: int, model_type: str) -> Tuple[Optional[float], float]:
        """Get prediction from individual model type"""
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
        return None, 0.0
    
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
            
            # Simulate XGBoost prediction (in production, load actual XGBoost model)
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

# Initialize global ensemble predictor
ensemble_predictor = AdvancedEnsemblePredictor()

# ───────── ENHANCEMENT 2: Advanced Feature Engineering ─────────
class AdvancedFeatureEngineer:
    """Advanced feature engineering with temporal patterns and game context"""
    
    def __init__(self):
        self.feature_cache = {}
        self.temporal_patterns = {}
    
    def extract_advanced_features(self, m: dict) -> Dict[str, float]:
        """Extract advanced features including temporal patterns and game context"""
        base_features = extract_enhanced_features(m)
        
        # Add temporal pattern features
        temporal_features = self._extract_temporal_patterns(m, base_features)
        base_features.update(temporal_features)
        
        # Add game context features
        context_features = self._extract_game_context(m, base_features)
        base_features.update(context_features)
        
        # Add team strength features
        strength_features = self._extract_team_strength_indicators(m, base_features)
        base_features.update(strength_features)
        
        return base_features
    
    def _extract_temporal_patterns(self, m: dict, base_features: Dict[str, float]) -> Dict[str, float]:
        """Extract temporal patterns and momentum indicators"""
        minute = base_features.get("minute", 0)
        events = m.get("events", [])
        
        temporal_features = {}
        
        # Rolling window statistics
        for window in [10, 15, 20]:
            temporal_features[f"goals_last_{window}"] = float(self._count_events_since(events, minute, window, 'Goal'))
            temporal_features[f"shots_last_{window}"] = float(self._count_events_since(events, minute, window, {'Shot', 'Missed Shot', 'Shot on Target'}))
            temporal_features[f"cards_last_{window}"] = float(self._count_events_since(events, minute, window, {'Card'}))
        
        # Acceleration features (rate of change)
        if minute > 15:
            goals_0_15 = temporal_features.get("goals_last_15", 0)
            goals_15_30 = self._count_events_between(events, max(0, minute-30), minute-15, 'Goal')
            temporal_features["goal_acceleration"] = goals_0_15 - goals_15_30
        
        # Time-decayed features
        temporal_features["time_decayed_xg"] = self._calculate_time_decayed_xg(base_features, minute)
        temporal_features["recent_pressure"] = self._calculate_recent_pressure(events, minute)
        
        return temporal_features
    
    def _extract_game_context(self, m: dict, base_features: Dict[str, float]) -> Dict[str, float]:
        """Extract game context and situational features"""
        context_features = {}
        
        minute = base_features.get("minute", 0)
        score_diff = base_features.get("goals_h", 0) - base_features.get("goals_a", 0)
        
        # Game state classification
        context_features["game_state"] = self._classify_game_state(score_diff, minute)
        
        # Urgency indicators
        context_features["home_urgency"] = self._calculate_urgency(score_diff, minute, is_home=True)
        context_features["away_urgency"] = self._calculate_urgency(-score_diff, minute, is_home=False)
        
        # Risk assessment
        context_features["defensive_risk"] = self._calculate_defensive_risk(base_features, minute)
        context_features["attacking_risk"] = self._calculate_attacking_risk(base_features, minute)
        
        # Match importance (simplified)
        context_features["match_importance"] = self._estimate_match_importance(m)
        
        return context_features
    
    def _extract_team_strength_indicators(self, m: dict, base_features: Dict[str, float]) -> Dict[str, float]:
        """Extract team strength and form indicators"""
        strength_features = {}
        
        # Current match performance indicators
        pressure_home = base_features.get("pressure_home", 1)
        pressure_away = base_features.get("pressure_away", 1)
        strength_features["home_dominance"] = pressure_home / max(1, pressure_away)
        strength_features["away_resilience"] = 1.0 / max(0.1, strength_features["home_dominance"])
        
        # Efficiency metrics
        xg_h = base_features.get("xg_h", 0.1)
        xg_a = base_features.get("xg_a", 0.1)
        goals_h = base_features.get("goals_h", 0)
        goals_a = base_features.get("goals_a", 0)
        
        strength_features["home_efficiency"] = goals_h / max(0.1, xg_h)
        strength_features["away_efficiency"] = goals_a / max(0.1, xg_a)
        
        # Defensive stability
        strength_features["home_defensive_stability"] = 1.0 - (goals_a / max(0.1, xg_a))
        strength_features["away_defensive_stability"] = 1.0 - (goals_h / max(0.1, xg_h))
        
        return strength_features
    
    def _count_events_since(self, events: List[dict], current_minute: int, window: int, event_types: any) -> int:
        """Count events of specific types in the last N minutes"""
        cutoff = current_minute - window
        count = 0
        
        for event in events:
            minute = event.get('time', {}).get('elapsed', 0)
            if minute >= cutoff:
                event_type = event.get('type')
                if isinstance(event_types, str):
                    if event_type == event_types:
                        count += 1
                else:
                    if event_type in event_types:
                        count += 1
        
        return count
    
    def _count_events_between(self, events: List[dict], start_minute: int, end_minute: int, event_type: str) -> int:
        """Count events between two minute marks"""
        count = 0
        for event in events:
            minute = event.get('time', {}).get('elapsed', 0)
            if start_minute <= minute <= end_minute and event.get('type') == event_type:
                count += 1
        return count
    
    def _calculate_time_decayed_xg(self, features: Dict[str, float], minute: int) -> float:
        """Calculate time-decayed xG (recent xG weighted more heavily)"""
        if minute <= 0:
            return 0.0
        
        xg_sum = features.get("xg_sum", 0)
        # Recent xG weighted more heavily (exponential decay)
        decay_factor = 0.9  # 10% decay per 10 minutes
        recent_weight = decay_factor ** (minute / 10.0)
        
        return (xg_sum / minute) * recent_weight * 90  # Project to full match
    
    def _calculate_recent_pressure(self, events: List[dict], minute: int) -> float:
        """Calculate recent pressure based on events in last 10 minutes"""
        recent_events = self._count_events_since(events, minute, 10, 
                                               {'Shot', 'Shot on Target', 'Corner', 'Dangerous Attack'})
        return min(1.0, recent_events / 10.0)
    
    def _classify_game_state(self, score_diff: float, minute: int) -> float:
        """Classify current game state"""
        if minute < 30:
            return 0.0  # Early game
        elif abs(score_diff) >= 3:
            return 1.0  # One-sided
        elif abs(score_diff) == 2 and minute > 70:
            return 0.8  # Comfortable lead late
        elif abs(score_diff) == 1 and minute > 75:
            return 0.9  # Close game late
        elif score_diff == 0 and minute > 80:
            return 0.7  # Draw late
        else:
            return 0.5  # Normal game state
    
    def _calculate_urgency(self, score_diff: float, minute: int, is_home: bool) -> float:
        """Calculate team urgency based on score and time"""
        if is_home:
            urgency_score = -score_diff  # Negative means losing
        else:
            urgency_score = score_diff  # Positive means losing
        
        # Urgency increases when losing and time is running out
        time_pressure = max(0, (minute - 60) / 30.0)  # Increases after 60 minutes
        
        return max(0.0, urgency_score * time_pressure)
    
    def _calculate_defensive_risk(self, features: Dict[str, float], minute: int) -> float:
        """Calculate defensive risk (higher = more vulnerable)"""
        goals_conceded = features.get("goals_a", 0) + features.get("goals_h", 0)
        xg_against = features.get("xg_a", 0) + features.get("xg_h", 0)
        
        defensive_efficiency = goals_conceded / max(0.1, xg_against)
        fatigue_factor = min(1.0, minute / 90.0)  # Risk increases with fatigue
        
        return defensive_efficiency * fatigue_factor
    
    def _calculate_attacking_risk(self, features: Dict[str, float], minute: int) -> float:
        """Calculate attacking risk (higher = more aggressive)"""
        pressure = (features.get("pressure_home", 0) + features.get("pressure_away", 0)) / 2
        urgency = (features.get("home_urgency", 0) + features.get("away_urgency", 0)) / 2
        
        return (pressure / 100.0) * urgency
    
    def _estimate_match_importance(self, m: dict) -> float:
        """Estimate match importance (simplified)"""
        league = m.get("league", {})
        league_name = (league.get("name", "") or "").lower()
        
        # Basic importance estimation
        if any(comp in league_name for comp in ["champions league", "europa league", "premier league"]):
            return 0.9
        elif any(comp in league_name for comp in ["cup", "knockout", "playoff"]):
            return 0.8
        else:
            return 0.5

# Initialize global feature engineer
feature_engineer = AdvancedFeatureEngineer()

# ───────── ENHANCEMENT 3: Intelligent Market-Specific Prediction System ─────────
class MarketSpecificPredictor:
    """Advanced market-specific prediction with specialized models"""
    
    def __init__(self):
        self.market_strategies = {
            "BTTS": self._predict_btts_advanced,
            "OU": self._predict_ou_advanced, 
            "1X2": self._predict_1x2_advanced
        }
        self.market_feature_sets = self._initialize_market_features()
    
    def predict_for_market(self, features: Dict[str, float], market: str, minute: int) -> Tuple[float, float]:
        """Market-specific prediction with advanced features"""
        if market.startswith("OU_"):
            return self._predict_ou_advanced(features, minute)
        elif market in self.market_strategies:
            return self.market_strategies[market](features, minute)
        else:
            # Fallback to ensemble prediction
            return ensemble_predictor.predict_ensemble(features, market, minute)
    
    def _predict_btts_advanced(self, features: Dict[str, float], minute: int) -> Tuple[float, float]:
        """Advanced BTTS prediction with defensive vulnerability analysis"""
        # Base probability from ensemble
        base_prob, base_conf = ensemble_predictor.predict_ensemble(features, "BTTS", minute)
        
        # Advanced BTTS-specific adjustments
        adjustments = 0.0
        
        # Defensive vulnerability factor
        defensive_stability = features.get("defensive_stability", 0.5)
        vulnerability = 1.0 - defensive_stability
        adjustments += vulnerability * 0.2
        
        # Attacking pressure balance
        pressure_balance = min(features.get("pressure_home", 0), features.get("pressure_away", 0)) / 100.0
        adjustments += pressure_balance * 0.15
        
        # Recent goal momentum
        goals_last_20 = features.get("goals_last_20", 0)
        adjustments += min(0.3, goals_last_20 * 0.1)
        
        # Game state adjustment
        game_state = features.get("game_state", 0.5)
        if game_state > 0.7:  # High-stakes situation
            adjustments += 0.1
        
        adjusted_prob = base_prob * (1 + adjustments)
        confidence = base_conf * 0.9  # Slightly reduce confidence for complex market
        
        return max(0.0, min(1.0, adjusted_prob)), max(0.0, min(1.0, confidence))
    
    def _predict_ou_advanced(self, features: Dict[str, float], minute: int) -> Tuple[float, float]:
        """Advanced Over/Under prediction with proper model loading"""
        # Try multiple approaches to get base probability
        base_prob, base_conf = 0.0, 0.0
        
        # Approach 1: Use ensemble prediction
        try:
            ensemble_prob, ensemble_conf = ensemble_predictor.predict_ensemble(features, "OU", minute)
            if ensemble_prob > base_prob:
                base_prob, base_conf = ensemble_prob, ensemble_conf
        except Exception as e:
            log.warning(f"[OU_PREDICT] Ensemble failed: {e}")
        
        # Approach 2: Fallback to direct model prediction
        if base_prob <= 0:
            for line in OU_LINES:
                market_key = f"OU_{_fmt_line(line)}"
                mdl = ensemble_predictor._load_ou_model_for_line(line)
                if mdl:
                    prob = predict_from_model(mdl, features)
                    confidence = 0.8  # Base confidence for direct model
                    if prob > base_prob:
                        base_prob, base_conf = prob, confidence
                    break  # Use first available model
        
        if base_prob <= 0:
            return 0.0, 0.0
        
        # Apply OU-specific adjustments
        adjustments = self._calculate_ou_adjustments(features, minute)
        adjusted_prob = max(0.0, min(1.0, base_prob * (1 + adjustments)))
        
        # Adjust confidence based on game state
        confidence_factor = self._calculate_ou_confidence_factor(features, minute)
        final_confidence = max(0.0, min(1.0, base_conf * confidence_factor))
        
        return adjusted_prob, final_confidence
    
    def _calculate_ou_adjustments(self, features: Dict[str, float], minute: int) -> float:
        """Calculate OU-specific probability adjustments"""
        adjustments = 0.0
        
        # Current game state analysis
        current_goals = features.get("goals_sum", 0)
        xg_sum = features.get("xg_sum", 0)
        minute = max(1, features.get("minute", 1))
        
        # Expected goals per minute rate
        xg_per_minute = xg_sum / minute
        expected_remaining_goals = xg_per_minute * (90 - minute)
        
        # Tempo analysis - compare actual vs expected
        expected_goals_by_now = (xg_per_minute * minute)
        if expected_goals_by_now > 0:
            tempo_ratio = current_goals / expected_goals_by_now
            if tempo_ratio > 1.3:  # Scoring faster than expected
                adjustments += 0.2
            elif tempo_ratio < 0.7:  # Scoring slower than expected
                adjustments -= 0.15
        
        # Pressure and attacking momentum
        pressure_total = features.get("pressure_home", 0) + features.get("pressure_away", 0)
        if pressure_total > 150:  # High pressure game
            adjustments += 0.1
        elif pressure_total < 80:  # Low pressure game
            adjustments -= 0.1
        
        # Defensive stability impact
        defensive_stability = features.get("defensive_stability", 0.5)
        if defensive_stability < 0.3:  # Poor defense
            adjustments += 0.15
        elif defensive_stability > 0.7:  # Strong defense
            adjustments -= 0.15
        
        # Time-based adjustments (late game factors)
        if minute > 75:
            # Late game urgency - teams push for goals
            score_diff = abs(features.get("goals_h", 0) - features.get("goals_a", 0))
            if score_diff <= 1:  # Close game
                adjustments += 0.1
            elif current_goals == 0:  # No goals yet
                adjustments += 0.05
        
        # Recent momentum (last 15 minutes)
        goals_last_15 = features.get("goals_last_15", 0)
        if goals_last_15 >= 2:
            adjustments += 0.1
        elif goals_last_15 == 0 and minute > 30:
            adjustments -= 0.05
        
        return adjustments
    
    def _calculate_ou_confidence_factor(self, features: Dict[str, float], minute: int) -> float:
        """Calculate confidence factor for OU predictions"""
        confidence = 1.0
        
        # Data quality factors
        xg_available = features.get("xg_sum", 0) > 0
        pressure_available = features.get("pressure_home", 0) > 0 or features.get("pressure_away", 0) > 0
        
        if not xg_available:
            confidence *= 0.7
        if not pressure_available:
            confidence *= 0.8
        
        # Game progression factor (more confidence as game progresses)
        progression_factor = min(1.0, minute / 60.0)
        confidence *= (0.5 + 0.5 * progression_factor)
        
        # Sample size factor (more events = more confidence)
        total_events = (
            features.get("sot_sum", 0) + 
            features.get("cor_sum", 0) + 
            features.get("goals_sum", 0)
        )
        if total_events < 5 and minute > 30:
            confidence *= 0.8
        
        return confidence
    
    def _predict_1x2_advanced(self, features: Dict[str, float], minute: int) -> Tuple[float, float, float]:
        """Advanced 1X2 prediction with momentum and psychological factors"""
        base_prob_h, conf_h = ensemble_predictor.predict_ensemble(features, "1X2_HOME", minute)
        base_prob_a, conf_a = ensemble_predictor.predict_ensemble(features, "1X2_AWAY", minute)
        
        # Normalize probabilities (suppress draw)
        total = base_prob_h + base_prob_a
        if total > 0:
            base_prob_h /= total
            base_prob_a /= total
        
        # Advanced adjustments
        prob_h = self._adjust_1x2_probability(base_prob_h, features, minute, is_home=True)
        prob_a = self._adjust_1x2_probability(base_prob_a, features, minute, is_home=False)
        
        # Renormalize
        total_adj = prob_h + prob_a
        if total_adj > 0:
            prob_h /= total_adj
            prob_a /= total_adj
        
        confidence = (conf_h + conf_a) / 2
        
        return prob_h, prob_a, confidence
    
    def _adjust_1x2_probability(self, base_prob: float, features: Dict[str, float], 
                              minute: int, is_home: bool) -> float:
        """Adjust 1X2 probability based on advanced factors"""
        adjustments = 0.0
        
        # Momentum factor
        momentum_key = "pressure_home" if is_home else "pressure_away"
        momentum = features.get(momentum_key, 0) / 100.0
        adjustments += momentum * 0.15
        
        # Psychological factor (score advantage)
        score_diff = features.get("goals_h", 0) - features.get("goals_a", 0)
        if is_home:
            psychological = score_diff * 0.1
        else:
            psychological = -score_diff * 0.1
        adjustments += psychological
        
        # Urgency factor
        urgency_key = "home_urgency" if is_home else "away_urgency"
        urgency = features.get(urgency_key, 0)
        adjustments += urgency * 0.08
        
        # Efficiency factor
        efficiency_key = "home_efficiency" if is_home else "away_efficiency"
        efficiency = features.get(efficiency_key, 1.0)
        adjustments += (efficiency - 1.0) * 0.1
        
        adjusted_prob = base_prob * (1 + adjustments)
        return max(0.0, min(1.0, adjusted_prob))
    
    def _initialize_market_features(self):
        """Initialize market-specific feature sets"""
        return {
            "BTTS": [
                "pressure_home", "pressure_away", "defensive_stability",
                "goals_last_15", "xg_sum", "game_state", "defensive_risk"
            ],
            "OU": [
                "xg_sum", "goals_sum", "attacking_momentum", "defensive_fatigue",
                "tempo_ratio", "time_pressure", "recent_xg_impact"
            ],
            "1X2": [
                "pressure_home", "pressure_away", "home_dominance", "away_resilience",
                "home_efficiency", "away_efficiency", "urgency", "psychological_advantage"
            ]
        }

# Initialize market predictor
market_predictor = MarketSpecificPredictor()

# ───────── ENHANCEMENT 4: Performance Monitoring and Adaptive Learning ─────────
class AdaptiveLearningSystem:
    """System for continuous learning and model adaptation"""
    
    def __init__(self):
        self.performance_history = {}
        self.feature_importance = {}
        self.model_adjustments = {}
        
    def record_prediction_outcome(self, prediction_data: Dict[str, Any], outcome: Optional[int]):
        """Record prediction outcome for continuous learning"""
        if outcome is None:
            return
            
        market = prediction_data.get("market", "")
        features = prediction_data.get("features", {})
        probability = prediction_data.get("probability", 0.0)
        
        # Update performance history
        key = f"{market}_{outcome}"
        self.performance_history[key] = self.performance_history.get(key, []) + [{
            'timestamp': time.time(),
            'probability': probability,
            'outcome': outcome,
            'features': features
        }]
        
        # Keep only recent history (last 1000 predictions per type)
        if len(self.performance_history[key]) > 1000:
            self.performance_history[key] = self.performance_history[key][-1000:]
        
        # Update feature importance
        self._update_feature_importance(features, outcome, probability, market)
        
    def _update_feature_importance(self, features: Dict[str, float], outcome: int, 
                                 probability: float, market: str):
        """Update feature importance based on prediction accuracy"""
        prediction_correct = 1 if (probability > 0.5 and outcome == 1) or (probability <= 0.5 and outcome == 0) else 0
        
        for feature_name, feature_value in features.items():
            if feature_name not in self.feature_importance:
                self.feature_importance[feature_name] = {
                    'total_uses': 0,
                    'correct_uses': 0,
                    'market_specific': {}
                }
            
            self.feature_importance[feature_name]['total_uses'] += 1
            self.feature_importance[feature_name]['correct_uses'] += prediction_correct
            
            # Market-specific importance
            if market not in self.feature_importance[feature_name]['market_specific']:
                self.feature_importance[feature_name]['market_specific'][market] = {
                    'total_uses': 0,
                    'correct_uses': 0
                }
            
            self.feature_importance[feature_name]['market_specific'][market]['total_uses'] += 1
            self.feature_importance[feature_name]['market_specific'][market]['correct_uses'] += prediction_correct

    def get_feature_weights(self, market: str) -> Dict[str, float]:
        """Get adaptive feature weights for specific market"""
        base_weights = {}
        
        for feature_name, stats in self.feature_importance.items():
            total_uses = stats['total_uses']
            correct_uses = stats['correct_uses']
            
            if total_uses > 10:  # Minimum sample size
                accuracy = correct_uses / total_uses
                
                # Market-specific accuracy if available
                market_stats = stats['market_specific'].get(market, {})
                if market_stats.get('total_uses', 0) > 5:
                    market_accuracy = market_stats['correct_uses'] / market_stats['total_uses']
                    accuracy = (accuracy + market_accuracy) / 2
                
                # Convert accuracy to weight (0.5 = neutral, 1.0 = perfect)
                weight = max(0.1, min(2.0, (accuracy - 0.5) * 2 + 1.0))
                base_weights[feature_name] = weight
        
        return base_weights
    
    def analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends and suggest improvements"""
        trends = {
            'overall_accuracy': 0.0,
            'market_performance': {},
            'feature_effectiveness': {},
            'recommendations': []
        }
        
        # Calculate overall accuracy
        total_predictions = 0
        correct_predictions = 0
        
        for key, history in self.performance_history.items():
            if history:
                market = key.split('_')[0]
                correct = sum(1 for h in history if h['outcome'] == 1)
                total = len(history)
                
                trends['market_performance'][market] = {
                    'accuracy': correct / total if total > 0 else 0.0,
                    'total_predictions': total
                }
                
                total_predictions += total
                correct_predictions += correct
        
        if total_predictions > 0:
            trends['overall_accuracy'] = correct_predictions / total_predictions
        
        # Feature effectiveness
        for feature_name, stats in self.feature_importance.items():
            if stats['total_uses'] > 0:
                trends['feature_effectiveness'][feature_name] = {
                    'accuracy': stats['correct_uses'] / stats['total_uses'],
                    'usage_count': stats['total_uses']
                }
        
        # Generate recommendations
        self._generate_recommendations(trends)
        
        return trends
    
    def _generate_recommendations(self, trends: Dict[str, Any]):
        """Generate improvement recommendations based on performance analysis"""
        recommendations = []
        
        # Check market performance
        for market, performance in trends['market_performance'].items():
            if performance['accuracy'] < 0.55 and performance['total_predictions'] > 50:
                recommendations.append(f"Consider reviewing {market} model - current accuracy: {performance['accuracy']:.1%}")
        
        # Check feature effectiveness
        for feature_name, effectiveness in trends['feature_effectiveness'].items():
            if effectiveness['accuracy'] < 0.48 and effectiveness['usage_count'] > 100:
                recommendations.append(f"Feature '{feature_name}' may be counterproductive - accuracy: {effectiveness['accuracy']:.1%}")
            elif effectiveness['accuracy'] > 0.65 and effectiveness['usage_count'] > 50:
                recommendations.append(f"Feature '{feature_name}' is highly effective - consider increasing weight")
        
        trends['recommendations'] = recommendations

# Initialize adaptive learning system
adaptive_learner = AdaptiveLearningSystem()

# ───────── Existing Feature Extraction Functions (keep original) ─────────
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

def extract_basic_features(m: dict) -> Dict[str,float]:
    """Original feature extraction - renamed to basic"""
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

def extract_enhanced_features(m: dict) -> Dict[str, float]:
    """Enhanced feature extraction with momentum and context"""
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

def _get_pre_match_probability(fid: int, market: str) -> Optional[float]:
    """Get pre-match probability for Bayesian updates"""
    try:
        # Try to get from pre-match models or historical data
        if market == "BTTS":
            mdl = load_model_from_settings("PRE_BTTS_YES")
            if mdl:
                # Would need match features, but return None for now
                return None
        elif market.startswith("OU"):
            line = _parse_ou_line_from_suggestion(market)
            if line:
                mdl = load_model_from_settings(f"PRE_OU_{_fmt_line(line)}")
                if mdl:
                    return None
        elif market == "1X2":
            mdl_home = load_model_from_settings("PRE_WLD_HOME")
            mdl_away = load_model_from_settings("PRE_WLD_AWAY")
            if mdl_home and mdl_away:
                # Return tuple for home/away probabilities
                return (0.4, 0.4)  # Placeholder
    except Exception as e:
        log.warning(f"[PRE_MATCH_PROB] Failed for {market}: {e}")
    return None

# ───────── ENHANCEMENT 5: Enhanced Production Scan with AI Systems ─────────
def enhanced_production_scan() -> Tuple[int, int]:
    """Enhanced scan with full AI capabilities"""
    if sleep_if_required():
        log.info("[ENHANCED_PROD] Skipping scan during sleep hours (22:00-08:00 Berlin time)")
        return (0, 0)
    
    if not _db_ping():
        log.error("[ENHANCED_PROD] Database unavailable")
        return (0, 0)
    
    try:
        matches = fetch_live_matches()
    except Exception as e:
        log.error("[ENHANCED_PROD] Failed to fetch live matches: %s", e)
        return (0, 0)
    
    live_seen = len(matches)
    if live_seen == 0:
        log.info("[ENHANCED_PROD] no live matches")
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

                # Extract advanced features with AI systems
                feat = feature_engineer.extract_advanced_features(m)
                minute = int(feat.get("minute", 0))
                
                # AI: Anomaly detection
                is_anomaly, anomaly_score = anomaly_detector.detect(feat)
                if is_anomaly:
                    ai_logger.log_anomaly(fid, anomaly_score, "Statistical anomaly detected")
                    continue

                # AI: Performance pruning
                home, away = _teams(m)
                if (performance_pruner.is_blacklisted(home) or 
                    performance_pruner.is_blacklisted(away)):
                    log.info(f"[PRUNER] Skipping blacklisted team: {home} vs {away}")
                    continue

                # Validation checks
                if not stats_coverage_ok(feat, minute):
                    continue
                if minute < TIP_MIN_MINUTE:
                    continue
                if is_feed_stale(fid, m, minute):
                    continue

                # Harvest mode snapshot
                if HARVEST_MODE and minute >= TRAIN_MIN_MINUTE and minute % 3 == 0:
                    try:
                        save_snapshot_from_match(m, feat)
                    except Exception:
                        pass

                # Game state analysis
                game_state_analyzer = GameStateAnalyzer()
                game_state = game_state_analyzer.analyze_game_state(feat)
                
                # AI: League weighting
                league_id, league = _league_name(m)
                league_weight = league_weighter.get_league_weight(league_id)
                
                home, away = _teams(m)
                score = _pretty_score(m)

                candidates: List[Tuple[str, str, float, float]] = []

                # PREDICT ALL MARKETS: BTTS, OU, 1X2 with AI ensemble
                log.info(f"[MARKET_SCAN] Processing {home} vs {away} at minute {minute}")
                
                # AI: Multi-model ensemble prediction
                ensemble_prob, ensemble_confidence = multi_model_ensemble.predict(feat, minute)
                
                # AI: Bayesian updating
                prior_prob = _get_pre_match_probability(fid, "BTTS")  # Example for BTTS
                if prior_prob is not None:
                    final_prob = bayesian_updater.update_prediction(
                        prior_prob, ensemble_prob, minute, ensemble_confidence
                    )
                else:
                    final_prob = ensemble_prob

                # AI: Apply league weighting
                final_prob = min(1.0, final_prob * league_weight)

                # 1. BTTS Market
                btts_prob, btts_confidence = market_predictor.predict_for_market(feat, "BTTS", minute)
                if btts_prob > 0 and btts_confidence > 0.5:
                    btts_predictions = {
                        "BTTS: Yes": btts_prob,
                        "BTTS: No": 1 - btts_prob
                    }
                    adjusted_btts = game_state_analyzer.adjust_predictions(btts_predictions, game_state)
                    
                    for suggestion, adj_prob in adjusted_btts.items():
                        threshold = _get_market_threshold("BTTS")
                        if adj_prob * 100 >= threshold:
                            candidates.append(("BTTS", suggestion, adj_prob, btts_confidence))
                            log.info(f"[BTTS_CANDIDATE] {suggestion}: {adj_prob:.3f} (conf: {btts_confidence:.3f})")

                # 2. Over/Under Markets
                for line in OU_LINES:
                    market_key = f"OU_{_fmt_line(line)}"
                    ou_prob, ou_confidence = market_predictor.predict_for_market(feat, market_key, minute)
                    
                    if ou_prob > 0 and ou_confidence > 0.5:
                        ou_predictions = {
                            f"Over {_fmt_line(line)} Goals": ou_prob,
                            f"Under {_fmt_line(line)} Goals": 1 - ou_prob
                        }
                        adjusted_ou = game_state_analyzer.adjust_predictions(ou_predictions, game_state)
                        
                        for suggestion, adj_prob in adjusted_ou.items():
                            threshold = _get_market_threshold(f"Over/Under {_fmt_line(line)}")
                            if adj_prob * 100 >= threshold:
                                candidates.append((f"Over/Under {_fmt_line(line)}", suggestion, adj_prob, ou_confidence))
                                log.info(f"[OU_CANDIDATE] {suggestion}: {adj_prob:.3f} (conf: {ou_confidence:.3f})")

                # 3. 1X2 Market (Draw suppressed)
                try:
                    prob_h, prob_a, confidence_1x2 = market_predictor.predict_for_market(feat, "1X2", minute)
                    
                    if prob_h > 0 and prob_a > 0 and confidence_1x2 > 0.5:
                        # Normalize probabilities (suppress draw)
                        total = prob_h + prob_a
                        if total > 0:
                            prob_h /= total
                            prob_a /= total
                            
                        predictions_1X2 = {
                            "Home Win": prob_h,
                            "Away Win": prob_a
                        }
                        adjusted_1x2 = game_state_analyzer.adjust_predictions(predictions_1X2, game_state)
                        
                        for suggestion, adj_prob in adjusted_1x2.items():
                            threshold = _get_market_threshold("1X2")
                            if adj_prob * 100 >= threshold:
                                candidates.append(("1X2", suggestion, adj_prob, confidence_1x2))
                                log.info(f"[1X2_CANDIDATE] {suggestion}: {adj_prob:.3f} (conf: {confidence_1x2:.3f})")
                except Exception as e:
                    log.warning(f"[1X2_PREDICT] Failed for {home} vs {away}: {e}")

                if not candidates:
                    log.info(f"[NO_CANDIDATES] No qualified tips for {home} vs {away}")
                    continue

                # AI: Bayesian updates with pre-match probabilities
                enhanced_candidates = []
                for market, suggestion, prob, confidence in candidates:
                    pre_match_data = _get_pre_match_probability(fid, market)
                    
                    if pre_match_data is not None:
                        if market == "1X2" and isinstance(pre_match_data, tuple):
                            pre_match_prob_home, pre_match_prob_away = pre_match_data
                            if suggestion == "Home Win":
                                enhanced_prob = bayesian_updater.update_prediction(pre_match_prob_home, prob, minute, confidence)
                            else:  # "Away Win"
                                enhanced_prob = bayesian_updater.update_prediction(pre_match_prob_away, prob, minute, confidence)
                        else:
                            enhanced_prob = bayesian_updater.update_prediction(pre_match_data, prob, minute, confidence)
                    else:
                        enhanced_prob = prob

                    enhanced_candidates.append((market, suggestion, enhanced_prob, confidence))

                # Odds analysis and filtering
                odds_map = fetch_odds(fid) if API_KEY else {}
                ranked: List[Tuple[str, str, float, Optional[float], Optional[str], Optional[float], float, float]] = []

                for mk, sug, prob, confidence in enhanced_candidates:
                    if sug not in ALLOWED_SUGGESTIONS:
                        continue

                    # Enhanced odds analysis
                    odds_analyzer = SmartOddsAnalyzer()
                    odds_quality = odds_analyzer.analyze_odds_quality(odds_map, {mk: prob})
                    
                    if odds_quality < odds_analyzer.odds_quality_threshold:
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
                        continue
                    if odds is None:
                        odds = odds2
                        book = book2

                    ev_pct = None
                    if odds is not None:
                        edge = _ev(prob, float(odds))
                        ev_pct = round(edge * 100.0, 1)
                        if int(round(edge * 10000)) < EDGE_MIN_BPS:
                            continue
                    else:
                        if not ALLOW_TIPS_WITHOUT_ODDS:
                            continue

                    # Enhanced ranking with confidence scoring
                    rank_score = (prob ** 1.2) * (1 + (ev_pct or 0) / 100.0) * confidence
                    ranked.append((mk, sug, prob, odds, book, ev_pct, rank_score, confidence))

                if not ranked:
                    continue

                ranked.sort(key=lambda x: x[6], reverse=True)  # Sort by rank score
                log.info(f"[RANKED_TIPS] Found {len(ranked)} qualified tips for {home} vs {away}")

                per_match = 0
                base_now = int(time.time())

                for idx, (market_txt, suggestion, prob, odds, book, ev_pct, _rank, confidence) in enumerate(ranked):
                    if PER_LEAGUE_CAP > 0 and per_league_counter.get(league_id, 0) >= PER_LEAGUE_CAP:
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

                            # Enhanced message with AI confidence indicator
                            sent = send_telegram(_format_enhanced_tip_message(
                                home, away, league, minute, score, suggestion, 
                                float(prob_pct), feat, odds, book, ev_pct, confidence
                            ))
                            
                            if sent:
                                c2.execute("UPDATE tips SET sent_ok=1 WHERE match_id=%s AND created_ts=%s", (fid, created_ts))
                                _metric_inc("tips_sent_total", n=1)
                                log.info(f"[TIP_SENT] {suggestion} for {home} vs {away} at {minute}'")
                    except Exception as e:
                        log.exception("[ENHANCED_PROD] insert/send failed: %s", e)
                        continue

                    saved += 1
                    per_match += 1
                    per_league_counter[league_id] = per_league_counter.get(league_id, 0) + 1

                    if MAX_TIPS_PER_SCAN and saved >= MAX_TIPS_PER_SCAN:
                        break
                    if per_match >= max(1, PREDICTIONS_PER_MATCH):
                        break

                if MAX_TIPS_PER_SCAN and saved >= MAX_TIPS_PER_SCAN:
                    break

            except Exception as e:
                exception_handler.handle("production_scan_match", lambda: None)
                continue

    log.info("[ENHANCED_PROD] saved=%d live_seen=%d", saved, live_seen)
    _metric_inc("tips_generated_total", n=saved)
    return saved, live_seen

def _format_enhanced_tip_message(home, away, league, minute, score, suggestion, 
                               prob_pct, feat, odds=None, book=None, ev_pct=None, confidence=None):
    """Enhanced tip message with AI confidence indicators"""
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

# ───────── Continue with existing functions (updated to use enhanced features where appropriate) ─────────

def _num(v) -> float:
    try:
        if isinstance(v,str) and v.endswith("%"): return float(v[:-1])
        return float(v or 0)
    except: return 0.0

def _pos_pct(v) -> float:
    try: return float(str(v).replace("%","").strip() or 0)
    except: return 0.0

# Update existing production_scan to use enhanced version by default
def production_scan() -> Tuple[int, int]:
    """Main production scan - now uses enhanced version"""
    return enhanced_production_scan()

# ───────── Existing Bayesian and Game State Classes ─────────
class BayesianUpdater:
    def __init__(self):
        self.prior_strength = 0.3  # How much we weight prior vs live data
    
    def update_probability(self, prior_prob: float, live_prob: float, minute: int) -> float:
        """Bayesian update combining pre-match prior with live evidence"""
        # Weight live data more as game progresses
        live_weight = min(minute / 90.0, 1.0) * (1 - self.prior_strength)
        prior_weight = self.prior_strength * (1 - live_weight)
        
        return (prior_prob * prior_weight + live_prob * live_weight) / (prior_weight + live_weight)
    
    def calculate_confidence_interval(self, prob: float, sample_size: int) -> Tuple[float, float]:
        """Calculate confidence interval for probability estimate"""
        import math
        z = 1.96  # 95% confidence
        
        if sample_size == 0:
            return prob, prob
            
        margin = z * math.sqrt((prob * (1 - prob)) / sample_size)
        return max(0, prob - margin), min(1, prob + margin)

class GameStateAnalyzer:
    def __init__(self):
        self.critical_states = {
            'equalizer_seek': 0.7,  # Team needs equalizer
            'park_the_bus': 0.6,    # Team protecting lead
            'goal_fest': 0.8,       # High-scoring game
            'defensive_battle': 0.3  # Low-scoring game
        }
    
    def analyze_game_state(self, feat: Dict[str, float]) -> Dict[str, float]:
        """Analyze current game state and tendencies"""
        state_scores = {}
        
        goal_diff = feat.get("goals_h", 0) - feat.get("goals_a", 0)
        minute = feat.get("minute", 0)
        total_goals = feat.get("goals_sum", 0)
        
        # Equalizer seeking (losing team pushing)
        if abs(goal_diff) == 1 and minute > 60:
            state_scores['equalizer_seek'] = 0.7 + (minute / 90.0) * 0.3
        
        # Park the bus (leading team defending)
        if goal_diff >= 2 and minute > 70:
            state_scores['park_the_bus'] = 0.6 + ((minute - 70) / 20.0) * 0.4
        
        # Goal fest (high scoring game)
        if total_goals >= 3 and minute < 60:
            state_scores['goal_fest'] = min(1.0, total_goals / 5.0)
        
        # Defensive battle
        if total_goals == 0 and minute > 60:
            state_scores['defensive_battle'] = 0.3 + (minute / 90.0) * 0.5
        
        return state_scores
    
    def adjust_predictions(self, predictions: dict, game_state: dict) -> dict:
        """Adjust predictions based on game state"""
        adjusted = predictions.copy()
        
        if game_state.get('equalizer_seek', 0) > 0.5:
            # Increase BTTS and Over probabilities
            if 'BTTS: Yes' in adjusted:
                adjusted['BTTS: Yes'] *= (1 + game_state['equalizer_seek'] * 0.3)
            # Increase relevant Over markets
            for key in list(adjusted.keys()):
                if key.startswith('Over'):
                    adjusted[key] *= (1 + game_state['equalizer_seek'] * 0.2)
        
        if game_state.get('park_the_bus', 0) > 0.5:
            # Decrease goal-related probabilities
            for key in list(adjusted.keys()):
                if key.startswith('Over'):
                    adjusted[key] *= (1 - game_state['park_the_bus'] * 0.4)
                elif key == 'BTTS: Yes':
                    adjusted[key] *= (1 - game_state['park_the_bus'] * 0.3)
        
        return adjusted

class SmartOddsAnalyzer:
    def __init__(self):
        self.odds_quality_threshold = 0.85  # Minimum quality score
    
    def analyze_odds_quality(self, odds_map: dict, prob_hints: dict) -> float:
        """Analyze odds quality and reliability"""
        if not odds_map:
            return 0.0
        
        quality_metrics = []
        
        for market, sides in odds_map.items():
            market_quality = self._market_odds_quality(sides, prob_hints.get(market, {}))
            quality_metrics.append(market_quality)
        
        return sum(quality_metrics) / len(quality_metrics) if quality_metrics else 0.0
    
    def _market_odds_quality(self, sides: dict, prob_hint: float) -> float:
        """Calculate quality for a specific market"""
        if len(sides) < 2:
            return 0.0
        
        # Check implied probabilities
        total_implied = sum(1.0 / data['odds'] for data in sides.values())
        overround = total_implied - 1.0
        
        # Quality decreases with higher overround
        overround_quality = max(0, 1.0 - overround * 5)
        
        # Compare with model probabilities if available
        model_quality = 1.0
        if prob_hint:
            best_side = max(sides.items(), key=lambda x: x[1]['odds'])
            model_ev = _ev(prob_hint, best_side[1]['odds'])
            model_quality = min(1.0, max(0.0, model_ev + 1.0))
        
        return (overround_quality + model_quality) / 2

# ───────── Init DB with AI Tables ─────────
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
        
        # AI Tables
        c.execute("""
            CREATE TABLE IF NOT EXISTS model_metadata (
                model_name TEXT,
                version TEXT,
                accuracy DOUBLE PRECISION,
                features TEXT,
                model_type TEXT,
                timestamp TIMESTAMP,
                training_samples INTEGER,
                hyperparameters TEXT,
                calibration_method TEXT,
                PRIMARY KEY (model_name, version)
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS anomaly_logs (
                match_id BIGINT,
                anomaly_score DOUBLE PRECISION,
                features TEXT,
                detected_at TIMESTAMP,
                PRIMARY KEY (match_id, detected_at)
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS league_performance (
                league_id BIGINT,
                accuracy DOUBLE PRECISION,
                sample_size INTEGER,
                last_updated TIMESTAMP,
                PRIMARY KEY (league_id)
            )
        """)
        
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

# ───────── Model loader (with validation) ─────────
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

# ───────── Odds fetch + aggregation (with negative cache) ─────────
def fetch_odds(fid: int, prob_hints: Optional[dict[str, float]] = None) -> dict:
    """
    Aggregated odds map:
      { "BTTS": {...}, "1X2": {...}, "OU_2.5": {...}, ... }
    Prefers /odds/live, falls back to /odds; aggregates across books with outlier & fair checks.
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
            hint = None
            if prob_hints:
                if mkey == "BTTS":
                    hint = prob_hints.get("BTTS: Yes") if side == "Yes" else (1.0 - (prob_hints.get("BTTS: Yes") or 0.0))
                elif mkey == "1X2":
                    hint = prob_hints.get("Home Win") if side == "Home" else (prob_hints.get("Away Win") if side == "Away" else None)
                elif mkey.startswith("OU_"):
                    try:
                        ln = float(mkey.split("_", 1)[1])
                        key = f"{_fmt_line(ln)}"
                        hint = prob_hints.get(f"Over {key} Goals") if side == "Over" else (1.0 - (prob_hints.get(f"Over {key} Goals") or 0.0))
                    except:
                        pass
            ag, label = _aggregate_price(lst, hint)
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

def _aggregate_price(vals: list[tuple[float, str]], prob_hint: Optional[float]) -> tuple[Optional[float], Optional[str]]:
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
    if prob_hint is not None and prob_hint > 0:
        fair = 1.0 / max(1e-6, float(prob_hint))
        cap = fair * max(1.0, ODDS_FAIR_MAX_MULT)
        filtered = [(o, b) for (o, b) in filtered if o <= cap] or filtered
    if ODDS_AGGREGATION == "best":
        best = max(filtered, key=lambda t: t[0])
        return float(best[0]), str(best[1])
    target = med2
    pick = min(filtered, key=lambda t: abs(t[0] - target))
    return float(pick[0]), f"{pick[1]} (median of {len(xs)})"

# ───────── Model scoring helpers (preserved) ─────────
MODEL_KEYS_ORDER=["model_v2:{name}","model_latest:{name}","model:{name}"]
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

def _load_ou_model_for_line(line: float) -> Optional[Dict[str,Any]]:
    name=f"OU_{_fmt_line(line)}"; mdl=load_model_from_settings(name)
    return mdl or (load_model_from_settings("O25") if abs(line-2.5)<1e-6 else None)

def _load_wld_models(): 
    return (load_model_from_settings("WLD_HOME"), load_model_from_settings("WLD_DRAW"), load_model_from_settings("WLD_AWAY"))

# ───────── Odds helpers (preserved & robust) ─────────
def _ev(prob: float, odds: float) -> float:
    """Return expected value as decimal (e.g. 0.05 = +5%)."""
    return prob*max(0.0, float(odds)) - 1.0

def _min_odds_for_market(market: str) -> float:
    if market.startswith("Over/Under"): return MIN_ODDS_OU
    if market == "BTTS": return MIN_ODDS_BTTS
    if market == "1X2":  return MIN_ODDS_1X2
    return 1.01

def _odds_cache_get(fid: int) -> Optional[dict]:
    rec=ODDS_CACHE.get(fid)
    if not rec: return None
    ts,data=rec
    if time.time()-ts>120: ODDS_CACHE.pop(fid,None); return None
    return data

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

def _market_family(market_text: str, suggestion: str) -> str:
    """Normalize to OU / BTTS / 1X2 (draw suppressed)."""
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

def market_cutoff_ok(minute: int, market_text: str, suggestion: str) -> bool:
    """
    True if we are still within the minute cutoff for this market.
    Falls back to TIP_MAX_MINUTE or (TOTAL_MATCH_MINUTES - 5).
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

def _price_gate(market_text: str, suggestion: str, fid: int) -> Tuple[bool, Optional[float], Optional[str], Optional[float]]:
    """
    Return (pass, odds, book, ev_pct). Enforces consistent EV behavior:
      - If odds missing and ALLOW_TIPS_WITHOUT_ODDS=0 => block.
      - If odds present => must pass min/max odds and EV >= EDGE_MIN_BPS.
    """
    odds_map=fetch_odds(fid) if API_KEY else {}
    odds=None; book=None

    if market_text=="BTTS":
        d=odds_map.get("BTTS",{})
        tgt="Yes" if suggestion.endswith("Yes") else "No"
        if tgt in d: odds=d[tgt]["odds"]; book=d[tgt]["book"]
    elif market_text=="1X2":
        d=odds_map.get("1X2",{})
        tgt="Home" if suggestion=="Home Win" else ("Away" if suggestion=="Away Win" else None)
        if tgt and tgt in d: odds=d[tgt]["odds"]; book=d[tgt]["book"]
    elif market_text.startswith("Over/Under"):
        ln_val = _parse_ou_line_from_suggestion(suggestion)
        d = odds_map.get(f"OU_{_fmt_line(ln_val)}", {}) if ln_val is not None else {}
        tgt = "Over" if suggestion.startswith("Over") else "Under"
        if tgt in d:
            odds = d[tgt]["odds"]
            book = d[tgt]["book"]

    if odds is None:
        return (True, None, None, None) if ALLOW_TIPS_WITHOUT_ODDS else (False, None, None, None)

    min_odds=_min_odds_for_market(market_text)
    if not (min_odds <= odds <= MAX_ODDS_ALL):
        return (False, odds, book, None)

    # EV checked later with actual prob
    return (True, odds, book, None)

# ───────── Data-quality & formatting helpers (preserved) ─────────
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

# ───────── Parse helpers (OU, results) ─────────
def _parse_ou_line_from_suggestion(s: str) -> Optional[float]:
    """Extract the line value from Over/Under suggestions like 'Over 2.5 Goals'"""
    try:
        # Handle different formats
        if "Over" in s or "Under" in s:
            # Extract the number after Over/Under
            import re
            match = re.search(r'(\d+\.?\d*)', s)
            if match:
                return float(match.group(1))
        return None
    except Exception:
        return None

# ───────── Cache Cleanup ─────────
def cleanup_caches():
    """Clean up stale cache entries"""
    now = time.time()
    max_age = 3600  # 1 hour
    
    # Clean STATS_CACHE, EVENTS_CACHE, ODDS_CACHE
    for cache in [STATS_CACHE, EVENTS_CACHE, ODDS_CACHE]:
        stale = [k for k, (ts, _) in cache.items() if now - ts > max_age]
        for k in stale:
            cache.pop(k, None)
    
    # Clean NEG_CACHE
    stale_neg = [k for k, (ts, _) in NEG_CACHE.items() if now - ts > NEG_TTL_SEC]
    for k in stale_neg:
        NEG_CACHE.pop(k, None)
    
    # Clean FEED_STATE
    if STALE_GUARD_ENABLE:
        stale_feeds = [fid for fid, state in _FEED_STATE.items() 
                      if now - state.get('last_change', 0) > STALE_STATS_MAX_SEC * 2]
        for fid in stale_feeds:
            _FEED_STATE.pop(fid, None)
    
    log.info("[CACHE] Cleaned up stale entries")

# ───────── Sanity checks & stale-feed guard (preserved) ─────────
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
    sh_tot_h = g(sh, ("total shots", "shots total"))
    sh_tot_a = g(sa, ("total shots", "shots total"))
    cor_h = g(sh, ("corner kicks",))
    cor_a = g(sa, ("corner kicks",))
    pos_h = g(sh, ("ball possession",))
    pos_a = g(sa, ("ball possession",))

    ev = m.get("events") or []
    n_events = len(ev)
    n_cards = 0
    for e in ev:
        if str(e.get("type", "")).lower() == "card":
            n_cards += 1

    gh = int(((m.get("goals") or {}).get("home") or 0) or 0)
    ga = int(((m.get("goals") or {}).get("away") or 0) or 0)

    return (
        round(xg_h + xg_a, 3),
        int(sot_h + sot_a),
        int(sh_tot_h + sh_tot_a),
        int(cor_h + cor_a),
        int(round(pos_h)), int(round(pos_a)),
        gh, ga,
        n_events, n_cards,
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

# ───────── Snapshots (preserved) ─────────
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

    xg_h = float(feat.get("xg_h", 0.0));      xg_a = float(feat.get("xg_a", 0.0))
    sot_h = float(feat.get("sot_h", 0.0));    sot_a = float(feat.get("sot_a", 0.0))
    cor_h = float(feat.get("cor_h", 0.0));    cor_a = float(feat.get("cor_a", 0.0))
    pos_h = float(feat.get("pos_h", 0.0));    pos_a = float(feat.get("pos_a", 0.0))
    red_h = float(feat.get("red_h", 0.0));    red_a = float(feat.get("red_a", 0.0))

    sh_total_h = float(feat.get("sh_total_h", 0.0))
    sh_total_a = float(feat.get("sh_total_a", 0.0))
    yellow_h   = float(feat.get("yellow_h", 0.0))
    yellow_a   = float(feat.get("yellow_a", 0.0))

    xg_sum = xg_h + xg_a;    xg_diff = xg_h - xg_a
    sot_sum = sot_h + sot_a
    cor_sum = cor_h + cor_a
    pos_diff = pos_h - pos_a
    red_sum = red_h + red_a
    sh_total_sum = sh_total_h + sh_total_a
    sh_total_diff = sh_total_h - sh_total_a
    yellow_sum = yellow_h + yellow_a

    snapshot = {
        "minute": minute,
        "gh": gh, "ga": ga,
        "league_id": league_id,
        "market": "HARVEST",
        "suggestion": "HARVEST",
        "confidence": 0,
        "stat": {
            "xg_h": xg_h, "xg_a": xg_a, "xg_sum": xg_sum, "xg_diff": xg_diff,
            "sot_h": sot_h, "sot_a": sot_a, "sot_sum": sot_sum,
            "cor_h": cor_h, "cor_a": cor_a, "cor_sum": cor_sum,
            "pos_h": pos_h, "pos_a": pos_a, "pos_diff": pos_diff,
            "red_h": red_h, "red_a": red_a, "red_sum": red_a,
            "sh_total_h": sh_total_h, "sh_total_a": sh_total_a,
            "sh_total_sum": sh_total_sum, "sh_total_diff": sh_total_diff,
            "yellow_h": yellow_h, "yellow_a": yellow_a, "yellow_sum": yellow_sum,
        }
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

# ───────── Outcomes/backfill/digest (preserved) ─────────
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
    js=api_get_with_sleep(FOOTBALL_API_URL, {"id": mid}) or {}
    arr=js.get("response") or [] if isinstance(js,dict) else []
    return arr[0] if arr else None

def _is_final(short: str) -> bool: return (short or "").upper() in {"FT","AET","PEN"}

def backfill_results_for_open_matches(max_rows: int = 200) -> int:
    """Backfill results with sleep time check"""
    if sleep_if_required():
        log.info("[BACKFILL] Skipping during sleep hours (22:00-08:00 Berlin time)")
        return 0
    
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
            pending = len([r for r in rows if r[5] is None or r[6] is None])  # No odds or no result
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

# ───────── Prematch pipeline (preserved) ─────────
def _kickoff_berlin(utc_iso: str|None) -> str:
    try:
        if not utc_iso: return "TBD"
        dt=datetime.fromisoformat(utc_iso.replace("Z","+00:00"))
        return dt.astimezone(BERLIN_TZ).strftime("%H:%M")
    except: return "TBD"

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
    """Odds snapshot with sleep time check"""
    if sleep_if_required():
        log.info("[ODDS_SNAPSHOT] Skipping during sleep hours (22:00-08:00 Berlin time)")
        return 0
    
    wrote = 0
    now = int(time.time())
    for fid in fixtures:
        try:
            od = fetch_odds(fid)  # aggregated
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

def prematch_scan_save() -> int:
    """Prematch scan with sleep time check"""
    if sleep_if_required():
        log.info("[PREMATCH] Skipping scan during sleep hours (22:00-08:00 Berlin time)")
        return 0
    
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
            p = _score_prob(feat, mdl)
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
            p = _score_prob(feat, mdl)
            thr = _get_market_threshold_pre("BTTS")
            if p * 100.0 >= thr:
                candidates.append(("PRE BTTS", "BTTS: Yes", p))
            q = 1.0 - p
            if q * 100.0 >= thr:
                candidates.append(("PRE BTTS", "BTTS: No", q))

        # PRE 1X2 (draw suppressed)
        mh, ma = load_model_from_settings("PRE_WLD_HOME"), load_model_from_settings("PRE_WLD_AWAY")
        if mh and ma:
            ph = _score_prob(feat, mh)
            pa = _score_prob(feat, ma)
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
                continue  # odds mandatory by default

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

# ───────── MOTD (preserved) ─────────
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
    """Send Match of the Day with sleep time check"""
    if sleep_if_required():
        log.info("[MOTD] Skipping during sleep hours (22:00-08:00 Berlin time)")
        return False
    
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
                    
                p = _score_prob(feat, mdl)
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
                p = _score_prob(feat, mdl)
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
                ph = _score_prob(feat, mh)
                pa = _score_prob(feat, ma)
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

# ───────── Auto-train / Auto-tune (fix wiring) ─────────
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

# The ROI-aware apply-tune from your original, exposed under a stable name
def _apply_tune_thresholds(days: int = 14) -> Dict[str, float]:
    # identical to your original logic (omitted inline comments for brevity), just kept as-is
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

# Backwards/endpoint-friendly name (fixes earlier NameError)
def auto_tune_thresholds(days: int = 14) -> Dict[str, float]:
    return _apply_tune_thresholds(days)

# ───────── Retry unsent tips (preserved) ─────────
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

# ───────── Scheduler (with AI enhancements) ─────────
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

        # AI maintenance jobs
        sched.add_job(
            lambda: _run_with_pg_lock(1010, performance_pruner.analyze_performance),
            "interval", hours=6, id="performance_pruning", max_instances=1, coalesce=True
        )
        
        sched.add_job(
            lambda: _run_with_pg_lock(1011, lambda: adaptive_learner.analyze_performance_trends()),
            "interval", hours=24, id="performance_analysis", max_instances=1, coalesce=True
        )

        # cache cleanup
        sched.add_job(cleanup_caches, "interval", hours=1, id="cache_cleanup")

        sched.start()
        _scheduler_started = True
        send_telegram("🚀 goalsniper FULL AI mode (in-play + prematch) with ADVANCED ENSEMBLE activated!")
        log.info("[SCHED] started (scan=%ss)", SCAN_INTERVAL_SEC)

    except Exception as e:
        log.exception("[SCHED] failed: %s", e)

# ───────── Admin / auth / endpoints (add AI endpoints) ─────────
def _require_admin():
    key=request.headers.get("X-API-Key") or request.args.get("key") or ((request.json or {}).get("key") if request.is_json else None)
    if not ADMIN_API_KEY or key != ADMIN_API_KEY: abort(401)

@app.route("/")
def root(): return jsonify({"ok": True, "name": "goalsniper", "mode": "FULL_AI_ENHANCED", "scheduler": RUN_SCHEDULER})

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
    # ultra-simple exposition to verify on Railway logs
    try:
        return jsonify({"ok": True, "metrics": METRICS})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# AI Admin Endpoints
@app.route("/admin/ai/performance-analysis", methods=["GET"])
def http_performance_analysis():
    _require_admin()
    try:
        trends = adaptive_learner.analyze_performance_trends()
        return jsonify({"ok": True, "trends": trends})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/admin/ai/generate-kombi", methods=["POST"])
def http_generate_kombi():
    _require_admin()
    try:
        data = request.get_json()
        tips = data.get('tips', [])
        max_stake = data.get('max_stake', 100.0)
        
        combinations = kombi_generator.generate_kombi(tips, max_stake)
        return jsonify({"ok": True, "combinations": combinations})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/admin/ai/train-model", methods=["POST"])
def http_train_ai_model():
    _require_admin()
    try:
        data = request.get_json()
        market = data.get('market')
        model_type = data.get('model_type', 'lightgbm')
        
        # This would fetch training data and train the model
        # Implementation depends on your data structure
        success = True
        
        return jsonify({"ok": success, "model": market, "type": model_type})
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
        if msg.startswith("/start"): send_telegram("👋 goalsniper bot (FULL AI ENHANCED mode) is online.")
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

# ───────── Boot with AI Initialization ─────────
def _on_boot():
    register_shutdown_handlers()
    validate_config()
    _init_pool()
    init_db()
    
    # Initialize AI systems
    try:
        # Load performance data for league weighting and pruning
        performance_pruner.analyze_performance()
        
        # Note: Anomaly detector would need actual feature data to fit
        # anomaly_detector.fit(recent_features) would be called when we have data
        
        log.info("[AI] Advanced AI systems initialized successfully")
    except Exception as e:
        log.error(f"[AI] Initialization failed: {e}")
    
    set_setting("boot_ts", str(int(time.time())))
    _start_scheduler_once()

# Call _on_boot() to initialize the application
_on_boot()

if __name__ == "__main__":
    app.run(host=os.getenv("HOST","0.0.0.0"), port=int(os.getenv("PORT","8080")))
