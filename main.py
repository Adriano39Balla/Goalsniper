# file: main.py
"""
goalsniper — FULL AI mode (in-play + prematch) with odds + EV gate.
Enhanced with production hardening, robust validation, extensive logging,
and anti-overfitting measures.
"""

import os
import json
import time
import logging
import requests
import psycopg2
import numpy as np
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import RealDictCursor
from html import escape
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import deque, defaultdict
from functools import lru_cache
from flask import Flask, jsonify, request, abort, g
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

# ───────── Env bootstrap ─────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ───────── Enhanced logging setup ─────────
class StructuredLogger:
    """Enhanced structured logging with context"""
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self.handler = None
        self._setup_logging()
    
    def _setup_logging(self):
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[%(asctime)s] %(levelname)s - %(name)s - '
                '%(filename)s:%(lineno)d - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
            self.logger.propagate = False
    
    def _add_context(self, msg, **kwargs):
        ctx = ' '.join([f'{k}={v}' for k, v in kwargs.items()])
        return f'{msg} | {ctx}' if ctx else msg
    
    def info(self, msg, **kwargs):
        self.logger.info(self._add_context(msg, **kwargs))
    
    def warning(self, msg, **kwargs):
        self.logger.warning(self._add_context(msg, **kwargs))
    
    def error(self, msg, **kwargs):
        self.logger.error(self._add_context(msg, **kwargs))
    
    def exception(self, msg, **kwargs):
        self.logger.exception(self._add_context(msg, **kwargs))
    
    def debug(self, msg, **kwargs):
        self.logger.debug(self._add_context(msg, **kwargs))

log = StructuredLogger("goalsniper")
app = Flask(__name__)

# ───────── Core env with validation ─────────
def _validate_env():
    """Validate critical environment variables"""
    required_vars = ["DATABASE_URL", "API_KEY"]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise SystemExit(f"Missing required env vars: {', '.join(missing)}")
    
    # Validate numeric values
    numeric_vars = {
        "CONF_THRESHOLD": ("70", float),
        "MAX_TIPS_PER_SCAN": ("25", int),
        "DUP_COOLDOWN_MIN": ("20", int),
        "TIP_MIN_MINUTE": ("8", int),
        "SCAN_INTERVAL_SEC": ("300", int),
        "EDGE_MIN_BPS": ("300", int),
    }
    
    for var, (default, type_func) in numeric_vars.items():
        try:
            val = os.getenv(var, default)
            type_func(val)
        except ValueError:
            raise SystemExit(f"Invalid value for {var}: {val}")

_validate_env()

# Load environment variables
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
API_KEY = os.getenv("API_KEY")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")
WEBHOOK_SECRET = os.getenv("TELEGRAM_WEBHOOK_SECRET")
RUN_SCHEDULER = os.getenv("RUN_SCHEDULER", "1") not in ("0", "false", "False", "no", "NO")

CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "70"))
MAX_TIPS_PER_SCAN = int(os.getenv("MAX_TIPS_PER_SCAN", "25"))
DUP_COOLDOWN_MIN = int(os.getenv("DUP_COOLDOWN_MIN", "20"))
TIP_MIN_MINUTE = int(os.getenv("TIP_MIN_MINUTE", "8"))
SCAN_INTERVAL_SEC = int(os.getenv("SCAN_INTERVAL_SEC", "300"))

HARVEST_MODE = os.getenv("HARVEST_MODE", "1") not in ("0", "false", "False", "no", "NO")
TRAIN_ENABLE = os.getenv("TRAIN_ENABLE", "1") not in ("0", "false", "False", "no", "NO")
TRAIN_HOUR_UTC = int(os.getenv("TRAIN_HOUR_UTC", "2"))
TRAIN_MINUTE_UTC = int(os.getenv("TRAIN_MINUTE_UTC", "12"))
TRAIN_MIN_MINUTE = int(os.getenv("TRAIN_MIN_MINUTE", "15"))

BACKFILL_EVERY_MIN = int(os.getenv("BACKFILL_EVERY_MIN", "15"))
BACKFILL_DAYS = int(os.getenv("BACKFILL_DAYS", "14"))
DAILY_ACCURACY_DIGEST_ENABLE = os.getenv("DAILY_ACCURACY_DIGEST_ENABLE", "1") not in ("0", "false", "False", "no", "NO")
DAILY_ACCURACY_HOUR = int(os.getenv("DAILY_ACCURACY_HOUR", "3"))
DAILY_ACCURACY_MINUTE = int(os.getenv("DAILY_ACCURACY_MINUTE", "6"))

AUTO_TUNE_ENABLE = os.getenv("AUTO_TUNE_ENABLE", "0") not in ("0", "false", "False", "no", "NO")
TARGET_PRECISION = float(os.getenv("TARGET_PRECISION", "0.60"))
THRESH_MIN_PREDICTIONS = int(os.getenv("THRESH_MIN_PREDICTIONS", "25"))
MIN_THRESH = float(os.getenv("MIN_THRESH", "55"))
MAX_THRESH = float(os.getenv("MAX_THRESH", "85"))

MOTD_MIN_EV_BPS = int(os.getenv("MOTD_MIN_EV_BPS", "0"))
try:
    MOTD_LEAGUE_IDS = [int(x) for x in (os.getenv("MOTD_LEAGUE_IDS", "").split(",")) if x.strip().isdigit()]
except Exception as e:
    log.error("Failed to parse MOTD_LEAGUE_IDS", error=str(e))
    MOTD_LEAGUE_IDS = []

# ───────── Lines ─────────
def _parse_lines(env_val: str, default: List[float]) -> List[float]:
    out = []
    for t in (env_val or "").split(","):
        t = t.strip()
        if not t:
            continue
        try:
            val = float(t)
            if 0.5 <= val <= 5.5:  # Reasonable goal line range
                out.append(val)
        except ValueError:
            pass
    return out or default

OU_LINES = [ln for ln in _parse_lines(os.getenv("OU_LINES", "2.5,3.5"), [2.5, 3.5]) if abs(ln - 1.5) > 1e-6]
TOTAL_MATCH_MINUTES = int(os.getenv("TOTAL_MATCH_MINUTES", "95"))
PREDICTIONS_PER_MATCH = int(os.getenv("PREDICTIONS_PER_MATCH", "2"))

# ───────── Odds/EV controls ─────────
MIN_ODDS_OU = float(os.getenv("MIN_ODDS_OU", "1.30"))
MIN_ODDS_BTTS = float(os.getenv("MIN_ODDS_BTTS", "1.30"))
MIN_ODDS_1X2 = float(os.getenv("MIN_ODDS_1X2", "1.30"))
MAX_ODDS_ALL = float(os.getenv("MAX_ODDS_ALL", "20.0"))
EDGE_MIN_BPS = int(os.getenv("EDGE_MIN_BPS", "300"))
ODDS_BOOKMAKER_ID = os.getenv("ODDS_BOOKMAKER_ID")
ALLOW_TIPS_WITHOUT_ODDS = os.getenv("ALLOW_TIPS_WITHOUT_ODDS", "1") not in ("0", "false", "False", "no", "NO")

# ───────── Markets allow-list ─────────
ALLOWED_SUGGESTIONS = {"BTTS: Yes", "BTTS: No", "Home Win", "Away Win"}

def _fmt_line(line: float) -> str:
    """Format line without trailing zeros"""
    s = f"{line:.2f}".rstrip("0").rstrip(".")
    return s if s else "0"

for _ln in OU_LINES:
    s = _fmt_line(_ln)
    ALLOWED_SUGGESTIONS.add(f"Over {s} Goals")
    ALLOWED_SUGGESTIONS.add(f"Under {s} Goals")

# ───────── External APIs / HTTP session ─────────
DATABASE_URL = os.getenv("DATABASE_URL")
BASE_URL = "https://v3.football.api-sports.io"
FOOTBALL_API_URL = f"{BASE_URL}/fixtures"
HEADERS = {"x-apisports-key": API_KEY, "Accept": "application/json"}
INPLAY_STATUSES = {"1H", "HT", "2H", "ET", "BT", "P"}

# Enhanced HTTP session with better error handling
session = requests.Session()
retry_strategy = Retry(
    total=5,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET", "POST"],
    respect_retry_after_header=True
)
adapter = HTTPAdapter(
    max_retries=retry_strategy,
    pool_connections=10,
    pool_maxsize=20
)
session.mount("https://", adapter)
session.mount("http://", adapter)

# ───────── Enhanced caches with size limits ─────────
class LRUCache:
    """LRU cache with TTL and size limit"""
    def __init__(self, maxsize: int = 1000, ttl: int = 120):
        self.maxsize = maxsize
        self.ttl = ttl
        self.cache = {}
        self.order = deque()
    
    def get(self, key):
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        if time.time() - entry['timestamp'] > self.ttl:
            self.delete(key)
            return None
        
        # Move to end (most recently used)
        self.order.remove(key)
        self.order.append(key)
        return entry['value']
    
    def set(self, key, value):
        if key in self.cache:
            self.order.remove(key)
        elif len(self.cache) >= self.maxsize:
            # Remove least recently used
            old_key = self.order.popleft()
            del self.cache[old_key]
        
        self.cache[key] = {
            'value': value,
            'timestamp': time.time()
        }
        self.order.append(key)
    
    def delete(self, key):
        if key in self.cache:
            del self.cache[key]
            if key in self.order:
                self.order.remove(key)
    
    def clear(self):
        self.cache.clear()
        self.order.clear()

# Initialize caches
STATS_CACHE = LRUCache(maxsize=500, ttl=90)
EVENTS_CACHE = LRUCache(maxsize=500, ttl=90)
ODDS_CACHE = LRUCache(maxsize=200, ttl=120)
SETTINGS_TTL = int(os.getenv("SETTINGS_TTL_SEC", "60"))
MODELS_TTL = int(os.getenv("MODELS_CACHE_TTL_SEC", "120"))

TZ_UTC = ZoneInfo("UTC")
BERLIN_TZ = ZoneInfo("Europe/Berlin")

# ───────── Optional import: trainer ─────────
try:
    from train_models import train_models
    TRAIN_MODELS_AVAILABLE = True
except ImportError as e:
    TRAIN_MODELS_AVAILABLE = False
    _IMPORT_ERR = repr(e)
    log.warning("train_models not available", error=_IMPORT_ERR)
    
    def train_models(*args, **kwargs):
        log.warning("train_models called but not available")
        return {"ok": False, "reason": f"train_models import failed: {_IMPORT_ERR}"}

# ───────── Thread-safe DB pool ─────────
POOL = None

class PooledConn:
    """Thread-safe database connection context manager"""
    def __init__(self):
        self.conn = None
        self.cursor = None
    
    def __enter__(self):
        global POOL
        if POOL is None:
            _init_pool()
        self.conn = POOL.getconn()
        self.conn.autocommit = True
        self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        return self.cursor
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self.cursor:
                self.cursor.close()
        finally:
            if self.conn:
                POOL.putconn(self.conn)
        if exc_type:
            log.error("Database operation failed", error=str(exc_val))

def _init_pool():
    """Initialize thread-safe connection pool"""
    global POOL
    dsn = DATABASE_URL
    # Ensure SSL mode
    if "sslmode=" not in dsn:
        dsn += "&sslmode=require" if "?" in dsn else "?sslmode=require"
    
    try:
        POOL = ThreadedConnectionPool(
            minconn=1,
            maxconn=int(os.getenv("DB_POOL_MAX", "10")),
            dsn=dsn
        )
        log.info("Database pool initialized")
    except Exception as e:
        log.error("Failed to initialize database pool", error=str(e))
        raise

def db_conn():
    """Get database connection from pool"""
    return PooledConn()

# ───────── Enhanced settings cache ─────────
class TTLValueCache:
    """Cache with TTL per key"""
    def __init__(self, default_ttl: int = 60):
        self.default_ttl = default_ttl
        self.data = {}
    
    def get(self, key: str):
        entry = self.data.get(key)
        if not entry:
            return None
        timestamp, value = entry
        if time.time() - timestamp > self.default_ttl:
            del self.data[key]
            return None
        return value
    
    def set(self, key: str, value):
        self.data[key] = (time.time(), value)
    
    def invalidate(self, key: str = None):
        if key is None:
            self.data.clear()
        elif key in self.data:
            del self.data[key]

_SETTINGS_CACHE = TTLValueCache(SETTINGS_TTL)
_MODELS_CACHE = TTLValueCache(MODELS_TTL)

def get_setting(key: str) -> Optional[str]:
    """Get setting from database"""
    try:
        with db_conn() as cursor:
            cursor.execute("SELECT value FROM settings WHERE key = %s", (key,))
            row = cursor.fetchone()
            return row['value'] if row else None
    except Exception as e:
        log.error("Failed to get setting", key=key, error=str(e))
        return None

def set_setting(key: str, value: str) -> None:
    """Set setting in database"""
    try:
        with db_conn() as cursor:
            cursor.execute(
                "INSERT INTO settings(key, value) VALUES(%s, %s) "
                "ON CONFLICT(key) DO UPDATE SET value = EXCLUDED.value",
                (key, value)
            )
        _SETTINGS_CACHE.invalidate(key)
        log.info("Setting updated", key=key)
    except Exception as e:
        log.error("Failed to set setting", key=key, error=str(e))

def get_setting_cached(key: str) -> Optional[str]:
    """Get setting with cache"""
    cached = _SETTINGS_CACHE.get(key)
    if cached is not None:
        return cached
    
    value = get_setting(key)
    _SETTINGS_CACHE.set(key, value)
    return value

def invalidate_model_caches_for_key(key: str):
    """Invalidate model cache for specific key"""
    if key.lower().startswith(("model", "model_latest", "model_v2", "pre_")):
        _MODELS_CACHE.invalidate(key)

# ───────── Enhanced database schema ─────────
def init_db():
    """Initialize database with enhanced schema"""
    try:
        with db_conn() as cursor:
            # Tips table with comprehensive tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tips (
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
                    sent_ok INTEGER DEFAULT 0,
                    validation_checked BOOLEAN DEFAULT FALSE,
                    historical_similarity DOUBLE PRECISION,
                    backtest_accuracy DOUBLE PRECISION,
                    risk_level TEXT,
                    PRIMARY KEY (match_id, created_ts)
                )
            """)
            
            # Enhanced snapshots with more context
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tip_snapshots (
                    match_id BIGINT,
                    created_ts BIGINT,
                    payload JSONB,
                    PRIMARY KEY (match_id, created_ts)
                )
            """)
            
            # Feedback with more detail
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id SERIAL PRIMARY KEY,
                    match_id BIGINT,
                    tip_created_ts BIGINT,
                    market TEXT,
                    suggestion TEXT,
                    verdict INTEGER,
                    notes TEXT,
                    created_ts BIGINT,
                    UNIQUE(match_id, tip_created_ts)
                )
            """)
            
            # Settings
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_ts BIGINT
                )
            """)
            
            # Match results with additional stats
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS match_results (
                    match_id BIGINT PRIMARY KEY,
                    final_goals_h INTEGER,
                    final_goals_a INTEGER,
                    btts_yes INTEGER,
                    total_shots INTEGER,
                    total_corners INTEGER,
                    total_cards INTEGER,
                    updated_ts BIGINT
                )
            """)
            
            # Historical patterns for validation
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS historical_patterns (
                    pattern_hash TEXT PRIMARY KEY,
                    market TEXT,
                    features JSONB,
                    success_rate DOUBLE PRECISION,
                    sample_size INTEGER,
                    last_seen_ts BIGINT
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tips_created ON tips (created_ts DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tips_match ON tips (match_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tips_sent ON tips (sent_ok, created_ts DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tips_validation ON tips (validation_checked)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_match ON tip_snapshots (match_id, created_ts DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_results_updated ON match_results (updated_ts DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_patterns_market ON historical_patterns (market, last_seen_ts DESC)")
            
            log.info("Database initialized successfully")
    except Exception as e:
        log.error("Failed to initialize database", error=str(e))
        raise

# ───────── Enhanced Telegram client ─────────
def send_telegram(text: str, parse_mode: str = "HTML") -> bool:
    """Send message to Telegram with retry logic"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("Telegram credentials not configured")
        return False
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    
    for attempt in range(3):
        try:
            response = session.post(
                url,
                json={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "text": text,
                    "parse_mode": parse_mode,
                    "disable_web_page_preview": True
                },
                timeout=10
            )
            
            if response.status_code == 200:
                log.info("Telegram message sent successfully", length=len(text))
                return True
            else:
                log.warning(
                    "Telegram API error",
                    attempt=attempt + 1,
                    status=response.status_code,
                    response=response.text[:200]
                )
        
        except requests.exceptions.RequestException as e:
            log.warning(
                "Telegram request failed",
                attempt=attempt + 1,
                error=str(e)
            )
        
        if attempt < 2:
            time.sleep(2 ** attempt)  # Exponential backoff
    
    log.error("Failed to send Telegram message after 3 attempts")
    return False

# ───────── Robust API helpers ─────────
def _api_get(url: str, params: dict, timeout: int = 20) -> Optional[dict]:
    """Make API request with comprehensive error handling"""
    if not API_KEY:
        log.error("API key not configured")
        return None
    
    try:
        start_time = time.time()
        response = session.get(
            url,
            headers=HEADERS,
            params=params,
            timeout=timeout
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            log.debug(
                "API request successful",
                url=url,
                params=str(params)[:100],
                elapsed=f"{elapsed:.2f}s",
                response_size=len(response.text)
            )
            return data
        elif response.status_code == 429:
            log.warning("API rate limit exceeded", url=url)
            retry_after = response.headers.get('Retry-After', 60)
            time.sleep(int(retry_after))
            return None
        else:
            log.error(
                "API request failed",
                url=url,
                status=response.status_code,
                response=response.text[:200]
            )
            return None
    
    except requests.exceptions.Timeout:
        log.error("API request timeout", url=url, timeout=timeout)
        return None
    except requests.exceptions.RequestException as e:
        log.error("API request exception", url=url, error=str(e))
        return None
    except json.JSONDecodeError as e:
        log.error("API response JSON decode error", url=url, error=str(e))
        return None

# ───────── Enhanced league filter ─────────
_BLOCK_PATTERNS = ["u17", "u18", "u19", "u20", "u21", "u23", "youth", "junior", "reserve", "res.", "friendlies", "friendly"]

def _is_blocked_league(league_obj: dict) -> bool:
    """Check if league should be blocked"""
    name = str(league_obj.get("name", "")).lower()
    country = str(league_obj.get("country", "")).lower()
    league_type = str(league_obj.get("type", "")).lower()
    
    # Block patterns
    text = f"{country} {name} {league_type}"
    if any(pattern in text for pattern in _BLOCK_PATTERNS):
        log.debug("League blocked by pattern", league=name, pattern=text)
        return True
    
    # Check deny list
    deny_list = [x.strip() for x in os.getenv("LEAGUE_DENY_IDS", "").split(",") if x.strip()]
    league_id = str(league_obj.get("id", ""))
    if league_id in deny_list:
        log.debug("League in deny list", league_id=league_id)
        return True
    
    return False

# ───────── Enhanced match data fetchers ─────────
def fetch_match_stats(fid: int) -> list:
    """Fetch match statistics with caching"""
    cached = STATS_CACHE.get(fid)
    if cached is not None:
        return cached
    
    log.debug("Fetching match statistics", fixture_id=fid)
    data = _api_get(f"{FOOTBALL_API_URL}/statistics", {"fixture": fid}) or {}
    stats = data.get("response", []) if isinstance(data, dict) else []
    
    STATS_CACHE.set(fid, stats)
    return stats

def fetch_match_events(fid: int) -> list:
    """Fetch match events with caching"""
    cached = EVENTS_CACHE.get(fid)
    if cached is not None:
        return cached
    
    log.debug("Fetching match events", fixture_id=fid)
    data = _api_get(f"{FOOTBALL_API_URL}/events", {"fixture": fid}) or {}
    events = data.get("response", []) if isinstance(data, dict) else []
    
    EVENTS_CACHE.set(fid, events)
    return events

def fetch_live_matches() -> List[dict]:
    """Fetch live matches with filtering"""
    log.debug("Fetching live matches")
    data = _api_get(FOOTBALL_API_URL, {"live": "all"}) or {}
    raw_matches = data.get("response", []) if isinstance(data, dict) else []
    
    matches = []
    for match in raw_matches:
        # Check league filter
        league = match.get("league", {})
        if _is_blocked_league(league):
            continue
        
        # Check match status
        fixture = match.get("fixture", {})
        status = fixture.get("status", {})
        elapsed = status.get("elapsed")
        short_status = (status.get("short") or "").upper()
        
        if elapsed is None or elapsed > 120 or short_status not in INPLAY_STATUSES:
            continue
        
        # Fetch additional data
        match_id = fixture.get("id")
        if not match_id:
            continue
        
        match["statistics"] = fetch_match_stats(match_id)
        match["events"] = fetch_match_events(match_id)
        matches.append(match)
    
    log.info("Fetched live matches", count=len(matches))
    return matches

# ───────── Enhanced feature extraction with validation ─────────
def _safe_float(value) -> float:
    """Safely convert value to float"""
    if value is None:
        return 0.0
    if isinstance(value, str):
        value = value.replace("%", "").strip()
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0

def extract_features(match: dict) -> Dict[str, float]:
    """Extract features from match data with validation"""
    teams = match.get("teams", {})
    home_name = teams.get("home", {}).get("name", "")
    away_name = teams.get("away", {}).get("name", "")
    
    goals = match.get("goals", {})
    gh = _safe_float(goals.get("home"))
    ga = _safe_float(goals.get("away"))
    
    fixture = match.get("fixture", {})
    status = fixture.get("status", {})
    minute = _safe_float(status.get("elapsed"))
    
    # Process statistics
    home_stats = {}
    away_stats = {}
    
    for stat_entry in match.get("statistics", []):
        team = stat_entry.get("team", {})
        team_name = team.get("name")
        if not team_name:
            continue
        
        stats_dict = {}
        for item in stat_entry.get("statistics", []):
            stat_type = item.get("type", "")
            value = item.get("value")
            stats_dict[stat_type] = value
        
        if team_name == home_name:
            home_stats = stats_dict
        elif team_name == away_name:
            away_stats = stats_dict
    
    # Extract common statistics
    xg_h = _safe_float(home_stats.get("Expected Goals", 0))
    xg_a = _safe_float(away_stats.get("Expected Goals", 0))
    sot_h = _safe_float(home_stats.get("Shots on Target", 0))
    sot_a = _safe_float(away_stats.get("Shots on Target", 0))
    cor_h = _safe_float(home_stats.get("Corner Kicks", 0))
    cor_a = _safe_float(away_stats.get("Corner Kicks", 0))
    
    # Ball possession (percentage)
    pos_h = _safe_float(home_stats.get("Ball Possession", 0))
    pos_a = _safe_float(away_stats.get("Ball Possession", 0))
    
    # Red cards from events
    red_h = red_a = 0
    for event in match.get("events", []):
        if event.get("type", "").lower() == "card":
            detail = (event.get("detail") or "").lower()
            if "red" in detail or "second yellow" in detail:
                team = event.get("team", {})
                team_name = team.get("name")
                if team_name == home_name:
                    red_h += 1
                elif team_name == away_name:
                    red_a += 1
    
    features = {
        "minute": float(minute),
        "goals_h": float(gh),
        "goals_a": float(ga),
        "goals_sum": float(gh + ga),
        "goals_diff": float(gh - ga),
        "xg_h": float(xg_h),
        "xg_a": float(xg_a),
        "xg_sum": float(xg_h + xg_a),
        "xg_diff": float(xg_h - xg_a),
        "sot_h": float(sot_h),
        "sot_a": float(sot_a),
        "sot_sum": float(sot_h + sot_a),
        "cor_h": float(cor_h),
        "cor_a": float(cor_a),
        "cor_sum": float(cor_h + cor_a),
        "pos_h": float(pos_h),
        "pos_a": float(pos_a),
        "pos_diff": float(pos_h - pos_a),
        "red_h": float(red_h),
        "red_a": float(red_a),
        "red_sum": float(red_h + red_a)
    }
    
    # Validate feature completeness
    missing_features = [k for k, v in features.items() if v == 0.0]
    if missing_features:
        log.debug("Missing features", fixture_id=fixture.get("id"), missing=missing_features)
    
    return features

def is_stats_coverage_adequate(features: Dict[str, float], minute: int) -> bool:
    """Check if statistical coverage is adequate for prediction"""
    require_stats_minute = int(os.getenv("REQUIRE_STATS_MINUTE", "35"))
    require_fields = int(os.getenv("REQUIRE_DATA_FIELDS", "2"))
    
    if minute < require_stats_minute:
        return True  # Early minutes don't need full stats
    
    # Check key statistical fields
    key_fields = [
        features.get("xg_sum", 0.0),
        features.get("sot_sum", 0.0),
        features.get("cor_sum", 0.0),
        max(features.get("pos_h", 0.0), features.get("pos_a", 0.0))
    ]
    
    non_zero_count = sum(1 for v in key_fields if v > 0.0)
    return non_zero_count >= max(0, require_fields)

# ───────── Enhanced model management ─────────
MODEL_KEYS_ORDER = ["model_v2:{name}", "model_latest:{name}", "model:{name}"]
EPS = 1e-12

@lru_cache(maxsize=50)
def _load_model_cached(name: str) -> Optional[Dict[str, Any]]:
    """Load model with LRU caching"""
    cached = _MODELS_CACHE.get(name)
    if cached is not None:
        return cached
    
    model_data = None
    for pattern in MODEL_KEYS_ORDER:
        key = pattern.format(name=name)
        raw = get_setting_cached(key)
        if not raw:
            continue
        
        try:
            data = json.loads(raw)
            # Ensure required fields
            data.setdefault("intercept", 0.0)
            data.setdefault("weights", {})
            
            # Ensure calibration parameters
            cal = data.get("calibration", {})
            if isinstance(cal, dict):
                cal.setdefault("method", "sigmoid")
                cal.setdefault("a", 1.0)
                cal.setdefault("b", 0.0)
                data["calibration"] = cal
            
            model_data = data
            log.debug("Loaded model", model_name=name)
            break
        
        except json.JSONDecodeError as e:
            log.error("Failed to parse model JSON", model_name=name, error=str(e))
        except Exception as e:
            log.error("Failed to load model", model_name=name, error=str(e))
    
    if model_data:
        _MODELS_CACHE.set(name, model_data)
    
    return model_data

def load_model_from_settings(name: str) -> Optional[Dict[str, Any]]:
    """Public interface to load model"""
    return _load_model_cached(name)

def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid"""
    try:
        if x < -50:
            return 1e-22
        if x > 50:
            return 1 - 1e-22
        return 1 / (1 + np.exp(-x))
    except Exception:
        return 0.5

def _logit(p: float) -> float:
    """Numerically stable logit"""
    p = max(EPS, min(1 - EPS, float(p)))
    return np.log(p / (1 - p))

def _linear_predict(features: Dict[str, float], weights: Dict[str, float], intercept: float) -> float:
    """Compute linear prediction"""
    score = float(intercept or 0.0)
    for feature, weight in weights.items():
        if feature in features:
            score += float(weight or 0.0) * float(features[feature])
    return score

def _calibrate_probability(p: float, calibration: Dict[str, Any]) -> float:
    """Apply calibration to probability"""
    method = calibration.get("method", "sigmoid").lower()
    a = float(calibration.get("a", 1.0))
    b = float(calibration.get("b", 0.0))
    
    try:
        if method == "platt":
            return _sigmoid(a * _logit(p) + b)
        elif method == "isotonic":
            # Simple linear calibration
            calibrated = a * p + b
            return max(0.0, min(1.0, calibrated))
        else:  # sigmoid
            return _sigmoid(a * _logit(p) + b)
    except Exception:
        return max(0.0, min(1.0, p))

def predict_probability(features: Dict[str, float], model: Dict[str, Any]) -> float:
    """Predict probability with calibration"""
    try:
        # Raw linear prediction
        linear_score = _linear_predict(
            features,
            model.get("weights", {}),
            float(model.get("intercept", 0.0))
        )
        
        # Sigmoid transformation
        raw_prob = _sigmoid(linear_score)
        
        # Apply calibration if available
        calibration = model.get("calibration")
        if calibration:
            calibrated_prob = _calibrate_probability(raw_prob, calibration)
        else:
            calibrated_prob = raw_prob
        
        # Ensure valid probability
        return max(0.0, min(1.0, calibrated_prob))
    
    except Exception as e:
        log.error("Prediction failed", error=str(e))
        return 0.5

def _load_ou_model(line: float) -> Optional[Dict[str, Any]]:
    """Load Over/Under model for specific line"""
    name = f"OU_{_fmt_line(line)}"
    model = load_model_from_settings(name)
    if not model and abs(line - 2.5) < 1e-6:
        model = load_model_from_settings("O25")
    return model

def _load_wld_models() -> Tuple[Optional[Dict[str, Any]], ...]:
    """Load Win/Lose/Draw models"""
    return (
        load_model_from_settings("WLD_HOME"),
        load_model_from_settings("WLD_DRAW"),
        load_model_from_settings("WLD_AWAY")
    )

# ───────── Enhanced odds handling with EV calculation ─────────
def calculate_ev(probability: float, odds: float) -> float:
    """Calculate expected value (decimal)"""
    return probability * max(0.0, float(odds)) - 1.0

def _get_min_odds_for_market(market: str) -> float:
    """Get minimum odds for market"""
    if market.startswith("Over/Under"):
        return MIN_ODDS_OU
    if market == "BTTS":
        return MIN_ODDS_BTTS
    if market == "1X2":
        return MIN_ODDS_1X2
    return 1.01

def _normalize_market_name(name: str) -> str:
    """Normalize market name"""
    name = (name or "").lower()
    if "both teams" in name or "btts" in name:
        return "BTTS"
    if "match winner" in name or "winner" in name or "1x2" in name:
        return "1X2"
    if "over/under" in name or "total" in name or "goals" in name:
        return "OU"
    return name

def fetch_odds(fixture_id: int) -> dict:
    """Fetch odds for fixture with comprehensive parsing"""
    cached = ODDS_CACHE.get(fixture_id)
    if cached is not None:
        return cached
    
    params = {"fixture": fixture_id}
    if ODDS_BOOKMAKER_ID:
        params["bookmaker"] = ODDS_BOOKMAKER_ID
    
    data = _api_get(f"{BASE_URL}/odds", params) or {}
    result = {
        "BTTS": {},
        "1X2": {},
        "OU": {}
    }
    
    try:
        for fixture_data in data.get("response", []):
            for bookmaker in fixture_data.get("bookmakers", []):
                bookmaker_name = bookmaker.get("name", "Unknown")
                
                for bet in bookmaker.get("bets", []):
                    market_name = _normalize_market_name(bet.get("name", ""))
                    
                    for value in bet.get("values", []):
                        label = (value.get("value") or "").strip().lower()
                        odds_value = float(value.get("odd") or 0)
                        
                        if not odds_value:
                            continue
                        
                        # BTTS
                        if market_name == "BTTS":
                            if "yes" in label:
                                result["BTTS"]["Yes"] = {
                                    "odds": odds_value,
                                    "book": bookmaker_name
                                }
                            elif "no" in label:
                                result["BTTS"]["No"] = {
                                    "odds": odds_value,
                                    "book": bookmaker_name
                                }
                        
                        # 1X2
                        elif market_name == "1X2":
                            if label in ("home", "1"):
                                result["1X2"]["Home"] = {
                                    "odds": odds_value,
                                    "book": bookmaker_name
                                }
                            elif label in ("away", "2"):
                                result["1X2"]["Away"] = {
                                    "odds": odds_value,
                                    "book": bookmaker_name
                                }
                        
                        # Over/Under
                        elif market_name == "OU":
                            if "over" in label or "under" in label:
                                try:
                                    line_str = label.split()[-1]
                                    line = float(line_str)
                                    line_key = _fmt_line(line)
                                    
                                    if "over" in label:
                                        result["OU"][f"Over_{line_key}"] = {
                                            "odds": odds_value,
                                            "book": bookmaker_name,
                                            "line": line
                                        }
                                    elif "under" in label:
                                        result["OU"][f"Under_{line_key}"] = {
                                            "odds": odds_value,
                                            "book": bookmaker_name,
                                            "line": line
                                        }
                                except (ValueError, IndexError):
                                    continue
        
        ODDS_CACHE.set(fixture_id, result)
        log.debug("Fetched odds", fixture_id=fixture_id, markets=list(result.keys()))
    
    except Exception as e:
        log.error("Failed to parse odds", fixture_id=fixture_id, error=str(e))
    
    return result

def check_price_gate(
    market: str,
    suggestion: str,
    fixture_id: int,
    probability: float
) -> Tuple[bool, Optional[float], Optional[str], Optional[float]]:
    """Check if tip passes price gate with EV calculation"""
    odds_data = fetch_odds(fixture_id) if API_KEY else {}
    
    odds = None
    bookmaker = None
    ev_pct = None
    
    # Extract target from suggestion
    if market == "BTTS":
        target = "Yes" if suggestion.endswith("Yes") else "No"
        if target in odds_data.get("BTTS", {}):
            odds = odds_data["BTTS"][target]["odds"]
            bookmaker = odds_data["BTTS"][target]["book"]
    
    elif market == "1X2":
        target = "Home" if "Home" in suggestion else "Away"
        if target in odds_data.get("1X2", {}):
            odds = odds_data["1X2"][target]["odds"]
            bookmaker = odds_data["1X2"][target]["book"]
    
    elif market.startswith("Over/Under"):
        parts = suggestion.split()
        target = parts[0]  # "Over" or "Under"
        line_str = parts[1]
        key = f"{target}_{line_str}"
        
        if key in odds_data.get("OU", {}):
            odds = odds_data["OU"][key]["odds"]
            bookmaker = odds_data["OU"][key]["book"]
    
    # If no odds found
    if odds is None:
        if ALLOW_TIPS_WITHOUT_ODDS:
            log.debug("No odds found but allowed", fixture_id=fixture_id, market=market)
            return True, None, None, None
        else:
            log.debug("No odds found and not allowed", fixture_id=fixture_id, market=market)
            return False, None, None, None
    
    # Check odds range
    min_odds = _get_min_odds_for_market(market)
    if not (min_odds <= odds <= MAX_ODDS_ALL):
        log.debug("Odds outside allowed range", odds=odds, min=min_odds, max=MAX_ODDS_ALL)
        return False, odds, bookmaker, None
    
    # Calculate EV
    ev_decimal = calculate_ev(probability, odds)
    ev_pct = round(ev_decimal * 100.0, 2)
    
    # Check EV threshold
    ev_bps = int(round(ev_decimal * 10000))
    if ev_bps < EDGE_MIN_BPS:
        log.debug("EV below threshold", ev_bps=ev_bps, min_ev_bps=EDGE_MIN_BPS)
        return False, odds, bookmaker, ev_pct
    
    log.debug("Price gate passed", odds=odds, ev_pct=ev_pct, bookmaker=bookmaker)
    return True, odds, bookmaker, ev_pct

# ───────── Historical validation system ─────────
def _extract_pattern_key(features: Dict[str, float]) -> str:
    """Extract pattern key from features for historical matching"""
    # Create a simplified feature vector for pattern matching
    pattern_features = {
        "minute": round(features.get("minute", 0) / 10) * 10,  # Round to nearest 10 minutes
        "goals_sum": int(features.get("goals_sum", 0)),
        "xg_sum": round(features.get("xg_sum", 0), 1),
        "sot_sum": int(features.get("sot_sum", 0)),
        "pos_diff": round(features.get("pos_diff", 0) / 10) * 10,  # Round to nearest 10%
    }
    return json.dumps(pattern_features, sort_keys=True)

def validate_with_historical_data(
    match_id: int,
    market: str,
    suggestion: str,
    features: Dict[str, float],
    probability: float
) -> Tuple[bool, Optional[float], Optional[float]]:
    """
    Validate tip against historical patterns.
    Returns: (is_valid, similarity_score, backtest_accuracy)
    """
    pattern_key = _extract_pattern_key(features)
    
    try:
        with db_conn() as cursor:
            # Look for similar historical patterns
            cursor.execute("""
                SELECT success_rate, sample_size, features
                FROM historical_patterns
                WHERE market = %s
                AND pattern_hash LIKE %s
                ORDER BY sample_size DESC, last_seen_ts DESC
                LIMIT 10
            """, (market, f"%{pattern_key[:50]}%"))
            
            similar_patterns = cursor.fetchall()
            
            if not similar_patterns:
                log.debug("No historical patterns found", match_id=match_id, market=market)
                return True, None, None  # No data to invalidate
            
            # Calculate weighted success rate
            total_weight = 0
            weighted_success = 0
            similarity_scores = []
            
            for pattern in similar_patterns:
                success_rate = pattern['success_rate']
                sample_size = pattern['sample_size']
                
                # Use sample size as weight
                weight = min(sample_size, 100)  # Cap weight
                weighted_success += success_rate * weight
                total_weight += weight
                
                # Calculate similarity (simplified)
                similarity_scores.append(min(sample_size / 50, 1.0))
            
            if total_weight == 0:
                return True, None, None
            
            historical_success = weighted_success / total_weight
            avg_similarity = np.mean(similarity_scores) if similarity_scores else 0
            
            # Check if historical success supports our prediction
            predicted_success = probability
            confidence_interval = 0.1  # ±10% tolerance
            
            lower_bound = historical_success - confidence_interval
            upper_bound = historical_success + confidence_interval
            
            is_supported = lower_bound <= predicted_success <= upper_bound
            
            log.debug(
                "Historical validation",
                match_id=match_id,
                market=market,
                predicted=predicted_success,
                historical=historical_success,
                similarity=avg_similarity,
                supported=is_supported,
                sample_size=total_weight
            )
            
            return is_supported, avg_similarity, historical_success
    
    except Exception as e:
        log.error("Historical validation failed", match_id=match_id, error=str(e))
        return True, None, None  # Fail open on error

def update_historical_patterns(
    match_id: int,
    market: str,
    features: Dict[str, float],
    outcome: Optional[int]
):
    """Update historical patterns with new result"""
    if outcome is None:
        return
    
    pattern_key = _extract_pattern_key(features)
    pattern_hash = f"{market}_{hash(pattern_key) & 0xFFFFFFFF}"
    
    try:
        with db_conn() as cursor:
            # Check existing pattern
            cursor.execute("""
                SELECT success_rate, sample_size
                FROM historical_patterns
                WHERE pattern_hash = %s
            """, (pattern_hash,))
            
            existing = cursor.fetchone()
            
            if existing:
                current_success = existing['success_rate']
                current_samples = existing['sample_size']
                
                # Update success rate
                new_samples = current_samples + 1
                new_success = (current_success * current_samples + outcome) / new_samples
                
                cursor.execute("""
                    UPDATE historical_patterns
                    SET success_rate = %s,
                        sample_size = %s,
                        last_seen_ts = %s
                    WHERE pattern_hash = %s
                """, (new_success, new_samples, int(time.time()), pattern_hash))
            
            else:
                # Insert new pattern
                cursor.execute("""
                    INSERT INTO historical_patterns (
                        pattern_hash, market, features,
                        success_rate, sample_size, last_seen_ts
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    pattern_hash,
                    market,
                    json.dumps(features),
                    float(outcome),
                    1,
                    int(time.time())
                ))
            
            log.debug("Updated historical pattern", pattern_hash=pattern_hash, outcome=outcome)
    
    except Exception as e:
        log.error("Failed to update historical patterns", match_id=match_id, error=str(e))

# ───────── Risk assessment system ─────────
def assess_risk_level(
    probability: float,
    historical_accuracy: Optional[float],
    similarity_score: Optional[float],
    market: str
) -> str:
    """Assess risk level of tip"""
    # Base risk on probability
    if probability >= 0.75:
        base_risk = "LOW"
    elif probability >= 0.60:
        base_risk = "MEDIUM"
    else:
        base_risk = "HIGH"
    
    # Adjust based on historical accuracy
    if historical_accuracy is not None:
        accuracy_diff = abs(probability - historical_accuracy)
        if accuracy_diff > 0.15:
            base_risk = "HIGH" if base_risk != "HIGH" else base_risk
        elif accuracy_diff < 0.05:
            if base_risk == "HIGH":
                base_risk = "MEDIUM"
    
    # Adjust based on pattern similarity
    if similarity_score is not None and similarity_score < 0.5:
        if base_risk == "LOW":
            base_risk = "MEDIUM"
        elif base_risk == "MEDIUM":
            base_risk = "HIGH"
    
    # Market-specific adjustments
    if market == "1X2" and probability < 0.65:
        base_risk = "HIGH"  # 1X2 is volatile
    
    return base_risk

# ───────── Enhanced tip saving with validation ─────────
def save_tip_with_validation(
    match_id: int,
    league_id: int,
    league: str,
    home: str,
    away: str,
    market: str,
    suggestion: str,
    confidence: float,
    confidence_raw: float,
    score: str,
    minute: int,
    odds: Optional[float],
    bookmaker: Optional[str],
    ev_pct: Optional[float],
    features: Dict[str, float]
) -> bool:
    """Save tip with comprehensive validation"""
    # Validate with historical data
    is_valid, similarity, historical_accuracy = validate_with_historical_data(
        match_id, market, suggestion, features, confidence_raw
    )
    
    if not is_valid:
        log.warning(
            "Tip rejected by historical validation",
            match_id=match_id,
            market=market,
            confidence=confidence,
            historical_accuracy=historical_accuracy
        )
        return False
    
    # Assess risk level
    risk_level = assess_risk_level(
        confidence_raw,
        historical_accuracy,
        similarity,
        market
    )
    
    # Save to database
    try:
        created_ts = int(time.time())
        
        with db_conn() as cursor:
            cursor.execute("""
                INSERT INTO tips (
                    match_id, league_id, league, home, away,
                    market, suggestion, confidence, confidence_raw,
                    score_at_tip, minute, created_ts,
                    odds, book, ev_pct,
                    validation_checked, historical_similarity,
                    backtest_accuracy, risk_level, sent_ok
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s,
                         %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                match_id, league_id, league, home, away,
                market, suggestion, confidence, confidence_raw,
                score, minute, created_ts,
                odds, bookmaker, ev_pct,
                True, similarity, historical_accuracy,
                risk_level, 0  # Not sent yet
            ))
        
        log.info(
            "Tip saved with validation",
            match_id=match_id,
            market=market,
            confidence=confidence,
            risk_level=risk_level,
            historical_similarity=similarity
        )
        
        return True
    
    except psycopg2.IntegrityError:
        log.warning("Duplicate tip detected", match_id=match_id, created_ts=created_ts)
        return False
    except Exception as e:
        log.error("Failed to save tip", match_id=match_id, error=str(e))
        return False

# ───────── Enhanced production scan ─────────
def production_scan() -> Tuple[int, int]:
    """Enhanced production scan with comprehensive validation"""
    log.info("Starting production scan")
    
    matches = fetch_live_matches()
    live_seen = len(matches)
    
    if live_seen == 0:
        log.info("No live matches found")
        return 0, 0
    
    saved_tips = 0
    now_ts = int(time.time())
    dup_cooldown_seconds = DUP_COOLDOWN_MIN * 60
    
    for match in matches:
        try:
            fixture = match.get("fixture", {})
            match_id = fixture.get("id")
            
            if not match_id:
                continue
            
            # Check duplicate cooldown
            with db_conn() as cursor:
                cutoff_ts = now_ts - dup_cooldown_seconds
                cursor.execute("""
                    SELECT 1 FROM tips
                    WHERE match_id = %s AND created_ts >= %s
                    LIMIT 1
                """, (match_id, cutoff_ts))
                
                if cursor.fetchone():
                    log.debug("Match in cooldown", match_id=match_id)
                    continue
            
            # Extract features
            features = extract_features(match)
            minute = int(features.get("minute", 0))
            
            # Validate stats coverage
            if not is_stats_coverage_adequate(features, minute):
                log.debug("Inadequate stats coverage", match_id=match_id, minute=minute)
                continue
            
            if minute < TIP_MIN_MINUTE:
                log.debug("Minute below threshold", match_id=match_id, minute=minute)
                continue
            
            # Harvest mode for training data
            if HARVEST_MODE and minute >= TRAIN_MIN_MINUTE and minute % 3 == 0:
                try:
                    save_snapshot_from_match(match, features)
                except Exception as e:
                    log.error("Failed to save snapshot", match_id=match_id, error=str(e))
            
            # Get match metadata
            league_info = match.get("league", {})
            league_id = league_info.get("id", 0)
            league_country = league_info.get("country", "")
            league_name = league_info.get("name", "")
            league_full = f"{league_country} - {league_name}".strip(" -")
            
            teams = match.get("teams", {})
            home = teams.get("home", {}).get("name", "")
            away = teams.get("away", {}).get("name", "")
            
            goals = match.get("goals", {})
            score = f"{goals.get('home', 0)}-{goals.get('away', 0)}"
            
            # Generate predictions for each market
            candidates = []
            
            # Over/Under predictions
            for line in OU_LINES:
                model = _load_ou_model(line)
                if not model:
                    continue
                
                over_prob = predict_probability(features, model)
                under_prob = 1.0 - over_prob
                
                line_str = _fmt_line(line)
                market_name = f"Over/Under {line_str}"
                threshold = _get_market_threshold(market_name)
                
                # Check Over
                if over_prob * 100.0 >= threshold:
                    suggestion = f"Over {line_str} Goals"
                    if _is_sane_suggestion(suggestion, features):
                        candidates.append((market_name, suggestion, over_prob))
                
                # Check Under
                if under_prob * 100.0 >= threshold:
                    suggestion = f"Under {line_str} Goals"
                    if _is_sane_suggestion(suggestion, features):
                        candidates.append((market_name, suggestion, under_prob))
            
            # BTTS predictions
            btts_model = load_model_from_settings("BTTS_YES")
            if btts_model:
                btts_yes_prob = predict_probability(features, btts_model)
                btts_no_prob = 1.0 - btts_yes_prob
                
                threshold = _get_market_threshold("BTTS")
                
                if btts_yes_prob * 100.0 >= threshold:
                    if _is_sane_suggestion("BTTS: Yes", features):
                        candidates.append(("BTTS", "BTTS: Yes", btts_yes_prob))
                
                if btts_no_prob * 100.0 >= threshold:
                    if _is_sane_suggestion("BTTS: No", features):
                        candidates.append(("BTTS", "BTTS: No", btts_no_prob))
            
            # 1X2 predictions (no draw)
            home_model, draw_model, away_model = _load_wld_models()
            if all([home_model, draw_model, away_model]):
                home_prob = predict_probability(features, home_model)
                draw_prob = predict_probability(features, draw_model)
                away_prob = predict_probability(features, away_model)
                
                # Normalize probabilities
                total = home_prob + draw_prob + away_prob + EPS
                home_prob_norm = home_prob / total
                away_prob_norm = away_prob / total
                
                threshold = _get_market_threshold("1X2")
                
                if home_prob_norm * 100.0 >= threshold:
                    candidates.append(("1X2", "Home Win", home_prob_norm))
                
                if away_prob_norm * 100.0 >= threshold:
                    candidates.append(("1X2", "Away Win", away_prob_norm))
            
            # Sort candidates by probability
            candidates.sort(key=lambda x: x[2], reverse=True)
            
            # Process top candidates
            tips_per_match = 0
            base_ts = now_ts
            
            for idx, (market_name, suggestion, probability) in enumerate(candidates):
                if suggestion not in ALLOWED_SUGGESTIONS:
                    continue
                
                if tips_per_match >= max(1, PREDICTIONS_PER_MATCH):
                    break
                
                if MAX_TIPS_PER_SCAN and saved_tips >= MAX_TIPS_PER_SCAN:
                    break
                
                # Check price gate
                passes_gate, odds, bookmaker, ev_pct = check_price_gate(
                    market_name, suggestion, match_id, probability
                )
                
                if not passes_gate:
                    continue
                
                # Create tip
                confidence_pct = round(probability * 100.0, 1)
                created_ts = base_ts + idx
                
                # Save with validation
                saved = save_tip_with_validation(
                    match_id=match_id,
                    league_id=league_id,
                    league=league_full,
                    home=home,
                    away=away,
                    market=market_name,
                    suggestion=suggestion,
                    confidence=confidence_pct,
                    confidence_raw=probability,
                    score=score,
                    minute=minute,
                    odds=odds,
                    bookmaker=bookmaker,
                    ev_pct=ev_pct,
                    features=features
                )
                
                if saved:
                    # Send to Telegram
                    success = send_tip(
                        home=home,
                        away=away,
                        league=league_full,
                        minute=minute,
                        score=score,
                        suggestion=suggestion,
                        confidence_pct=confidence_pct,
                        features=features,
                        odds=odds,
                        bookmaker=bookmaker,
                        ev_pct=ev_pct
                    )
                    
                    # Update sent status
                    if success:
                        with db_conn() as cursor:
                            cursor.execute("""
                                UPDATE tips
                                SET sent_ok = 1
                                WHERE match_id = %s AND created_ts = %s
                            """, (match_id, created_ts))
                    
                    saved_tips += 1
                    tips_per_match += 1
            
            if MAX_TIPS_PER_SCAN and saved_tips >= MAX_TIPS_PER_SCAN:
                break
        
        except Exception as e:
            log.exception("Error processing match", match_id=match.get("fixture", {}).get("id"), error=str(e))
            continue
    
    log.info("Production scan completed", saved=saved_tips, live_seen=live_seen)
    return saved_tips, live_seen

def _is_sane_suggestion(suggestion: str, features: Dict[str, float]) -> bool:
    """Check if suggestion is sane given current match state"""
    goals_h = int(features.get("goals_h", 0))
    goals_a = int(features.get("goals_a", 0))
    total_goals = goals_h + goals_a
    
    if suggestion.startswith("Over"):
        try:
            line = float(suggestion.split()[1])
            if total_goals > line - 1e-6:  # Already exceeded
                return False
        except (ValueError, IndexError):
            return False
    
    elif suggestion.startswith("Under"):
        try:
            line = float(suggestion.split()[1])
            if total_goals >= line - 1e-6:  # Already met or exceeded
                return False
        except (ValueError, IndexError):
            return False
    
    elif suggestion == "BTTS: Yes":
        if goals_h > 0 and goals_a > 0:  # Already happened
            return False
    
    return True

def _get_market_threshold(market: str) -> float:
    """Get threshold for market"""
    try:
        key = f"conf_threshold:{market}"
        value = get_setting_cached(key)
        return float(value) if value is not None else CONF_THRESHOLD
    except Exception:
        return CONF_THRESHOLD

# ───────── Enhanced tip sending ─────────
def send_tip(
    home: str,
    away: str,
    league: str,
    minute: int,
    score: str,
    suggestion: str,
    confidence_pct: float,
    features: Dict[str, float],
    odds: Optional[float] = None,
    bookmaker: Optional[str] = None,
    ev_pct: Optional[float] = None
) -> bool:
    """Send tip to Telegram"""
    # Format statistics section
    stats_parts = []
    
    xg_h = features.get("xg_h", 0)
    xg_a = features.get("xg_a", 0)
    if xg_h > 0 or xg_a > 0:
        stats_parts.append(f"xG {xg_h:.2f}-{xg_a:.2f}")
    
    sot_h = int(features.get("sot_h", 0))
    sot_a = int(features.get("sot_a", 0))
    if sot_h > 0 or sot_a > 0:
        stats_parts.append(f"SOT {sot_h}-{sot_a}")
    
    cor_h = int(features.get("cor_h", 0))
    cor_a = int(features.get("cor_a", 0))
    if cor_h > 0 or cor_a > 0:
        stats_parts.append(f"CK {cor_h}-{cor_a}")
    
    pos_h = int(features.get("pos_h", 0))
    pos_a = int(features.get("pos_a", 0))
    if pos_h > 0 or pos_a > 0:
        stats_parts.append(f"POS {pos_h}%-{pos_a}%")
    
    red_h = int(features.get("red_h", 0))
    red_a = int(features.get("red_a", 0))
    if red_h > 0 or red_a > 0:
        stats_parts.append(f"RED {red_h}-{red_a}")
    
    stats_text = "\n📊 " + " • ".join(stats_parts) if stats_parts else ""
    
    # Format odds section
    odds_text = ""
    if odds:
        if ev_pct is not None:
            odds_text = f"\n💰 <b>Odds:</b> {odds:.2f} @ {bookmaker or 'Book'}  •  <b>EV:</b> {ev_pct:+.1f}%"
        else:
            odds_text = f"\n💰 <b>Odds:</b> {odds:.2f} @ {bookmaker or 'Book'}"
    
    # Compose message
    message = (
        f"⚽️ <b>New Tip!</b>\n"
        f"<b>Match:</b> {escape(home)} vs {escape(away)}\n"
        f"🕒 <b>Minute:</b> {minute}'  |  <b>Score:</b> {escape(score)}\n"
        f"<b>Tip:</b> {escape(suggestion)}\n"
        f"📈 <b>Confidence:</b> {confidence_pct:.1f}%"
        f"{odds_text}"
        f"\n🏆 <b>League:</b> {escape(league)}"
        f"{stats_text}"
    )
    
    return send_telegram(message)

# ───────── Snapshot saving ─────────
def save_snapshot_from_match(match: dict, features: Dict[str, float]):
    """Save snapshot for training data"""
    fixture = match.get("fixture", {})
    match_id = fixture.get("id")
    if not match_id:
        return
    
    league_info = match.get("league", {})
    league_id = league_info.get("id", 0)
    league_country = league_info.get("country", "")
    league_name = league_info.get("name", "")
    league_full = f"{league_country} - {league_name}".strip(" -")
    
    teams = match.get("teams", {})
    home = teams.get("home", {}).get("name", "")
    away = teams.get("away", {}).get("name", "")
    
    goals = match.get("goals", {})
    gh = goals.get("home", 0)
    ga = goals.get("away", 0)
    
    minute = int(features.get("minute", 0))
    
    snapshot = {
        "match_id": match_id,
        "league_id": league_id,
        "league": league_full,
        "home": home,
        "away": away,
        "minute": minute,
        "score": f"{gh}-{ga}",
        "features": features,
        "timestamp": int(time.time())
    }
    
    try:
        with db_conn() as cursor:
            cursor.execute("""
                INSERT INTO tip_snapshots (match_id, created_ts, payload)
                VALUES (%s, %s, %s)
                ON CONFLICT (match_id, created_ts) 
                DO UPDATE SET payload = EXCLUDED.payload
            """, (match_id, int(time.time()), json.dumps(snapshot)))
        
        log.debug("Snapshot saved", match_id=match_id, minute=minute)
    
    except Exception as e:
        log.error("Failed to save snapshot", match_id=match_id, error=str(e))

# ───────── Backfill and result processing ─────────
def _parse_ou_line(suggestion: str) -> Optional[float]:
    """Parse Over/Under line from suggestion"""
    try:
        for token in suggestion.split():
            try:
                return float(token)
            except ValueError:
                continue
    except Exception:
        pass
    return None

def _determine_tip_outcome(suggestion: str, result: Dict[str, Any]) -> Optional[int]:
    """Determine if tip was successful (1), failed (0), or void (None)"""
    goals_h = int(result.get("final_goals_h") or 0)
    goals_a = int(result.get("final_goals_a") or 0)
    total_goals = goals_h + goals_a
    btts = int(result.get("btts_yes") or 0)
    
    suggestion_lower = suggestion.lower()
    
    if suggestion_lower.startswith("over"):
        line = _parse_ou_line(suggestion)
        if line is None:
            return None
        if total_goals > line:
            return 1
        elif abs(total_goals - line) < 1e-9:
            return None  # Push
        else:
            return 0
    
    elif suggestion_lower.startswith("under"):
        line = _parse_ou_line(suggestion)
        if line is None:
            return None
        if total_goals < line:
            return 1
        elif abs(total_goals - line) < 1e-9:
            return None  # Push
        else:
            return 0
    
    elif "btts: yes" in suggestion_lower:
        return 1 if btts == 1 else 0
    
    elif "btts: no" in suggestion_lower:
        return 1 if btts == 0 else 0
    
    elif "home win" in suggestion_lower:
        return 1 if goals_h > goals_a else 0
    
    elif "away win" in suggestion_lower:
        return 1 if goals_a > goals_h else 0
    
    return None

def _get_fixture_by_id(fixture_id: int) -> Optional[dict]:
    """Get fixture by ID"""
    data = _api_get(FOOTBALL_API_URL, {"id": fixture_id}) or {}
    fixtures = data.get("response", []) if isinstance(data, dict) else []
    return fixtures[0] if fixtures else None

def _is_final_status(status_short: str) -> bool:
    """Check if match status is final"""
    return (status_short or "").upper() in {"FT", "AET", "PEN"}

def backfill_results_for_open_matches(max_rows: int = 200) -> int:
    """Backfill results for matches without final results"""
    log.info("Starting results backfill")
    
    now_ts = int(time.time())
    cutoff_ts = now_ts - BACKFILL_DAYS * 24 * 3600
    updated = 0
    
    try:
        with db_conn() as cursor:
            # Get matches without results
            cursor.execute("""
                SELECT DISTINCT t.match_id
                FROM tips t
                LEFT JOIN match_results r ON r.match_id = t.match_id
                WHERE t.created_ts >= %s
                  AND r.match_id IS NULL
                ORDER BY t.created_ts DESC
                LIMIT %s
            """, (cutoff_ts, max_rows))
            
            matches = cursor.fetchall()
        
        for match in matches:
            match_id = match['match_id']
            
            fixture = _get_fixture_by_id(match_id)
            if not fixture:
                continue
            
            status = fixture.get("fixture", {}).get("status", {})
            status_short = status.get("short", "")
            
            if not _is_final_status(status_short):
                continue
            
            goals = fixture.get("goals", {})
            goals_h = goals.get("home", 0)
            goals_a = goals.get("away", 0)
            btts = 1 if goals_h > 0 and goals_a > 0 else 0
            
            # Get additional stats for enhanced tracking
            stats = fixture.get("statistics", [])
            total_shots = 0
            total_corners = 0
            total_cards = 0
            
            # Simple stat extraction (would need API endpoint)
            
            try:
                with db_conn() as cursor:
                    cursor.execute("""
                        INSERT INTO match_results (
                            match_id, final_goals_h, final_goals_a,
                            btts_yes, total_shots, total_corners,
                            total_cards, updated_ts
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (match_id) DO UPDATE SET
                            final_goals_h = EXCLUDED.final_goals_h,
                            final_goals_a = EXCLUDED.final_goals_a,
                            btts_yes = EXCLUDED.btts_yes,
                            total_shots = EXCLUDED.total_shots,
                            total_corners = EXCLUDED.total_corners,
                            total_cards = EXCLUDED.total_cards,
                            updated_ts = EXCLUDED.updated_ts
                    """, (
                        match_id, goals_h, goals_a, btts,
                        total_shots, total_corners, total_cards, now_ts
                    ))
                
                updated += 1
                
                # Update historical patterns with outcomes
                # Get tips for this match to update patterns
                with db_conn() as cursor:
                    cursor.execute("""
                        SELECT market, confidence_raw, score_at_tip, minute
                        FROM tips
                        WHERE match_id = %s
                    """, (match_id,))
                    
                    tips = cursor.fetchall()
                
                # Update patterns for each tip
                for tip in tips:
                    # Simplified feature extraction for pattern update
                    # In production, you'd need to store features or reconstruct them
                    pass
            
            except Exception as e:
                log.error("Failed to update results", match_id=match_id, error=str(e))
        
        log.info("Results backfill completed", updated=updated)
        return updated
    
    except Exception as e:
        log.error("Results backfill failed", error=str(e))
        return 0

# ───────── Daily accuracy digest ─────────
def daily_accuracy_digest() -> Optional[str]:
    """Generate daily accuracy digest"""
    if not DAILY_ACCURACY_DIGEST_ENABLE:
        return None
    
    log.info("Generating daily accuracy digest")
    
    # Get yesterday's date in Berlin time
    now_berlin = datetime.now(BERLIN_TZ)
    yesterday_start = (now_berlin - timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    yesterday_end = yesterday_start + timedelta(days=1)
    
    # Backfill any missing results
    backfill_results_for_open_matches(400)
    
    try:
        with db_conn() as cursor:
            cursor.execute("""
                SELECT 
                    t.match_id,
                    t.market,
                    t.suggestion,
                    t.confidence,
                    t.confidence_raw,
                    t.created_ts,
                    r.final_goals_h,
                    r.final_goals_a,
                    r.btts_yes
                FROM tips t
                LEFT JOIN match_results r ON r.match_id = t.match_id
                WHERE t.created_ts >= %s
                  AND t.created_ts < %s
                  AND t.suggestion != 'HARVEST'
                  AND t.sent_ok = 1
                ORDER BY t.created_ts DESC
            """, (int(yesterday_start.timestamp()), int(yesterday_end.timestamp())))
            
            tips = cursor.fetchall()
        
        total_tips = len(tips)
        graded_tips = 0
        winning_tips = 0
        market_stats = defaultdict(lambda: {"graded": 0, "wins": 0})
        
        for tip in tips:
            result = {
                "final_goals_h": tip['final_goals_h'],
                "final_goals_a": tip['final_goals_a'],
                "btts_yes": tip['btts_yes']
            }
            
            outcome = _determine_tip_outcome(tip['suggestion'], result)
            
            if outcome is None:
                continue
            
            graded_tips += 1
            if outcome == 1:
                winning_tips += 1
            
            market = tip['market'] or "Unknown"
            market_stats[market]["graded"] += 1
            if outcome == 1:
                market_stats[market]["wins"] += 1
        
        # Generate message
        if graded_tips == 0:
            message = "📊 <b>Daily Digest</b>\nNo graded tips for yesterday."
        else:
            accuracy = (winning_tips / graded_tips * 100) if graded_tips > 0 else 0
            
            lines = [
                f"📊 <b>Daily Digest</b> (yesterday, Berlin time)",
                f"Tips sent: {total_tips}  •  Graded: {graded_tips}  •  Wins: {winning_tips}",
                f"Accuracy: <b>{accuracy:.1f}%</b>"
            ]
            
            # Add market breakdown
            if market_stats:
                lines.append("\n<b>By Market:</b>")
                for market, stats in sorted(market_stats.items()):
                    if stats["graded"] > 0:
                        market_acc = (stats["wins"] / stats["graded"] * 100)
                        lines.append(f"• {escape(market)}: {stats['wins']}/{stats['graded']} ({market_acc:.1f}%)")
            
            message = "\n".join(lines)
        
        success = send_telegram(message)
        if success:
            log.info("Daily digest sent", accuracy=f"{accuracy:.1f}%" if graded_tips > 0 else "N/A")
        else:
            log.error("Failed to send daily digest")
        
        return message if success else None
    
    except Exception as e:
        log.error("Failed to generate daily digest", error=str(e))
        return None

# ───────── Auto-tuning system ─────────
def auto_tune_thresholds(days: int = 14) -> Dict[str, float]:
    """Auto-tune confidence thresholds based on historical performance"""
    if not AUTO_TUNE_ENABLE:
        return {}
    
    log.info("Starting auto-tuning")
    
    cutoff_ts = int(time.time()) - days * 24 * 3600
    
    try:
        with db_conn() as cursor:
            cursor.execute("""
                SELECT 
                    t.market,
                    t.suggestion,
                    COALESCE(t.confidence_raw, t.confidence / 100.0) as probability,
                    r.final_goals_h,
                    r.final_goals_a,
                    r.btts_yes
                FROM tips t
                JOIN match_results r ON r.match_id = t.match_id
                WHERE t.created_ts >= %s
                  AND t.suggestion != 'HARVEST'
                  AND t.sent_ok = 1
                ORDER BY t.created_ts DESC
            """, (cutoff_ts,))
            
            data = cursor.fetchall()
        
        # Group by market
        market_data = defaultdict(list)
        for row in data:
            market = row['market']
            probability = float(row['probability'])
            
            result = {
                "final_goals_h": row['final_goals_h'],
                "final_goals_a": row['final_goals_a'],
                "btts_yes": row['btts_yes']
            }
            
            outcome = _determine_tip_outcome(row['suggestion'], result)
            if outcome is None:
                continue
            
            market_data[market].append((probability, outcome))
        
        tuned_thresholds = {}
        
        for market, samples in market_data.items():
            if len(samples) < THRESH_MIN_PREDICTIONS:
                log.debug("Insufficient samples for tuning", market=market, samples=len(samples))
                continue
            
            probabilities = np.array([s[0] for s in samples])
            outcomes = np.array([s[1] for s in samples])
            
            # Find optimal threshold for target precision
            best_threshold = CONF_THRESHOLD / 100.0  # Default
            
            for threshold in np.arange(MIN_THRESH, MAX_THRESH + 1, 1.0) / 100.0:
                predictions = (probabilities >= threshold).astype(int)
                positive_count = predictions.sum()
                
                if positive_count < THRESH_MIN_PREDICTIONS:
                    continue
                
                true_positives = ((predictions == 1) & (outcomes == 1)).sum()
                precision = true_positives / positive_count if positive_count > 0 else 0
                
                if precision >= TARGET_PRECISION:
                    best_threshold = threshold
                    break
            
            threshold_pct = best_threshold * 100.0
            tuned_thresholds[market] = threshold_pct
            
            # Update setting
            set_setting(f"conf_threshold:{market}", f"{threshold_pct:.2f}")
            _SETTINGS_CACHE.invalidate(f"conf_threshold:{market}")
            
            log.info(
                "Threshold tuned",
                market=market,
                threshold=f"{threshold_pct:.1f}%",
                samples=len(samples)
            )
        
        if tuned_thresholds:
            # Send notification
            lines = ["🔧 <b>Auto-tune Updated Thresholds</b>"]
            for market, threshold in sorted(tuned_thresholds.items()):
                lines.append(f"• {escape(market)}: {threshold:.1f}%")
            
            send_telegram("\n".join(lines))
            log.info("Auto-tuning completed", updated=len(tuned_thresholds))
        
        else:
            send_telegram("🔧 Auto-tune: No updates (insufficient data)")
            log.info("Auto-tuning completed with no updates")
        
        return tuned_thresholds
    
    except Exception as e:
        log.error("Auto-tuning failed", error=str(e))
        send_telegram(f"❌ Auto-tune <b>FAILED</b>\n{escape(str(e))}")
        return {}

# ───────── Retry unsent tips ─────────
def retry_unsent_tips(minutes: int = 30, limit: int = 200) -> int:
    """Retry sending failed tips"""
    log.info("Retrying unsent tips")
    
    cutoff_ts = int(time.time()) - minutes * 60
    retried = 0
    
    try:
        with db_conn() as cursor:
            cursor.execute("""
                SELECT 
                    match_id, league, home, away,
                    market, suggestion, confidence,
                    score_at_tip, minute, created_ts,
                    odds, book, ev_pct
                FROM tips
                WHERE sent_ok = 0
                  AND created_ts >= %s
                ORDER BY created_ts ASC
                LIMIT %s
            """, (cutoff_ts, limit))
            
            tips = cursor.fetchall()
        
        for tip in tips:
            # Reconstruct features (simplified - in production, store features)
            features = {}
            
            success = send_tip(
                home=tip['home'],
                away=tip['away'],
                league=tip['league'],
                minute=tip['minute'],
                score=tip['score_at_tip'],
                suggestion=tip['suggestion'],
                confidence_pct=tip['confidence'],
                features=features,
                odds=tip['odds'],
                bookmaker=tip['book'],
                ev_pct=tip['ev_pct']
            )
            
            if success:
                with db_conn() as cursor:
                    cursor.execute("""
                        UPDATE tips
                        SET sent_ok = 1
                        WHERE match_id = %s AND created_ts = %s
                    """, (tip['match_id'], tip['created_ts']))
                
                retried += 1
                log.debug("Retried unsent tip", match_id=tip['match_id'])
        
        log.info("Retry completed", retried=retried)
        return retried
    
    except Exception as e:
        log.error("Retry failed", error=str(e))
        return 0

# ───────── Prematch functionality (simplified) ─────────
def prematch_scan_save() -> int:
    """Prematch scan - simplified for focus on in-play"""
    log.info("Prematch scan not fully implemented in this version")
    return 0

def send_match_of_the_day() -> bool:
    """Send match of the day - simplified"""
    log.info("Match of the day not fully implemented in this version")
    return send_telegram("🏅 Match of the Day: Feature not implemented in this version.")

# ───────── Training job ─────────
def auto_train_job():
    """Auto training job"""
    if not TRAIN_ENABLE:
        send_telegram("🤖 Training skipped: TRAIN_ENABLE=0")
        return
    
    if not TRAIN_MODELS_AVAILABLE:
        send_telegram("⚠️ Training skipped: train_models module not available")
        return
    
    send_telegram("🤖 Training started...")
    
    try:
        result = train_models() or {}
        
        if not result.get("ok"):
            reason = result.get("reason") or result.get("error") or "unknown"
            send_telegram(f"⚠️ Training finished: <b>SKIPPED</b>\nReason: {escape(str(reason))}")
            return
        
        trained_models = [k for k, v in (result.get("trained") or {}).items() if v]
        thresholds = result.get("thresholds") or {}
        
        lines = ["🤖 <b>Model training completed</b>"]
        
        if trained_models:
            lines.append("• Trained: " + ", ".join(sorted(trained_models)))
        
        if thresholds:
            lines.append("• Thresholds: " + "  |  ".join([
                f"{escape(k)}: {float(v):.1f}%" for k, v in thresholds.items()
            ]))
        
        send_telegram("\n".join(lines))
        log.info("Training job completed", trained=len(trained_models))
    
    except Exception as e:
        log.exception("Training job failed", error=str(e))
        send_telegram(f"❌ Training <b>FAILED</b>\n{escape(str(e))}")

# ───────── Scheduler with advisory locks ─────────
def _run_with_pg_lock(lock_key: int, func, *args, **kwargs):
    """Run function with PostgreSQL advisory lock"""
    try:
        with db_conn() as cursor:
            # Try to acquire lock
            cursor.execute("SELECT pg_try_advisory_lock(%s)", (lock_key,))
            lock_acquired = cursor.fetchone()['pg_try_advisory_lock']
            
            if not lock_acquired:
                log.debug("Lock not acquired, skipping", lock_key=lock_key)
                return None
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # Release lock
                cursor.execute("SELECT pg_advisory_unlock(%s)", (lock_key,))
    
    except Exception as e:
        log.error("Lock operation failed", lock_key=lock_key, error=str(e))
        return None

_scheduler_started = False

def _start_scheduler_once():
    """Start scheduler once"""
    global _scheduler_started
    
    if _scheduler_started or not RUN_SCHEDULER:
        return
    
    try:
        scheduler = BackgroundScheduler(timezone=TZ_UTC)
        
        # Production scan
        scheduler.add_job(
            lambda: _run_with_pg_lock(1001, production_scan),
            "interval",
            seconds=SCAN_INTERVAL_SEC,
            id="scan",
            max_instances=1,
            coalesce=True
        )
        
        # Results backfill
        scheduler.add_job(
            lambda: _run_with_pg_lock(1002, backfill_results_for_open_matches, 400),
            "interval",
            minutes=BACKFILL_EVERY_MIN,
            id="backfill",
            max_instances=1,
            coalesce=True
        )
        
        # Daily digest
        if DAILY_ACCURACY_DIGEST_ENABLE:
            scheduler.add_job(
                lambda: _run_with_pg_lock(1003, daily_accuracy_digest),
                CronTrigger(
                    hour=DAILY_ACCURACY_HOUR,
                    minute=DAILY_ACCURACY_MINUTE,
                    timezone=BERLIN_TZ
                ),
                id="digest",
                max_instances=1,
                coalesce=True
            )
        
        # Training
        if TRAIN_ENABLE:
            scheduler.add_job(
                lambda: _run_with_pg_lock(1005, auto_train_job),
                CronTrigger(
                    hour=TRAIN_HOUR_UTC,
                    minute=TRAIN_MINUTE_UTC,
                    timezone=TZ_UTC
                ),
                id="train",
                max_instances=1,
                coalesce=True
            )
        
        # Auto-tuning
        if AUTO_TUNE_ENABLE:
            scheduler.add_job(
                lambda: _run_with_pg_lock(1006, auto_tune_thresholds, 14),
                CronTrigger(hour=4, minute=7, timezone=TZ_UTC),
                id="auto_tune",
                max_instances=1,
                coalesce=True
            )
        
        # Retry unsent tips
        scheduler.add_job(
            lambda: _run_with_pg_lock(1007, retry_unsent_tips, 30, 200),
            "interval",
            minutes=10,
            id="retry",
            max_instances=1,
            coalesce=True
        )
        
        scheduler.start()
        _scheduler_started = True
        
        send_telegram("🚀 GoalSniper AI (Enhanced) started successfully!")
        log.info("Scheduler started", scan_interval=SCAN_INTERVAL_SEC)
    
    except Exception as e:
        log.exception("Failed to start scheduler", error=str(e))

# ───────── Startup initialization ─────────
def _on_boot():
    """Initialize application on boot"""
    log.info("Starting GoalSniper Enhanced")
    
    try:
        # Initialize database pool
        _init_pool()
        
        # Initialize database schema
        init_db()
        
        # Set boot timestamp
        set_setting("boot_ts", str(int(time.time())))
        set_setting("version", "2.0-enhanced")
        
        # Start scheduler
        _start_scheduler_once()
        
        log.info("Boot sequence completed")
    
    except Exception as e:
        log.error("Boot sequence failed", error=str(e))
        raise

_on_boot()

# ───────── Flask endpoints ─────────
@app.route("/")
def root():
    return jsonify({
        "ok": True,
        "name": "goalsniper-enhanced",
        "version": "2.0",
        "mode": "FULL_AI_WITH_VALIDATION",
        "scheduler": RUN_SCHEDULER
    })

@app.route("/health")
def health():
    try:
        with db_conn() as cursor:
            cursor.execute("SELECT COUNT(*) as count FROM tips")
            tip_count = cursor.fetchone()['count']
        
        return jsonify({
            "ok": True,
            "database": "connected",
            "tips_count": tip_count,
            "cache_sizes": {
                "stats": len(STATS_CACHE.cache),
                "events": len(EVENTS_CACHE.cache),
                "odds": len(ODDS_CACHE.cache)
            }
        })
    except Exception as e:
        log.error("Health check failed", error=str(e))
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/admin/scan", methods=["POST", "GET"])
def http_scan():
    _require_admin()
    saved, live = production_scan()
    return jsonify({"ok": True, "saved": saved, "live_seen": live})

@app.route("/admin/backfill-results", methods=["POST", "GET"])
def http_backfill():
    _require_admin()
    updated = backfill_results_for_open_matches(400)
    return jsonify({"ok": True, "updated": updated})

@app.route("/admin/train", methods=["POST", "GET"])
def http_train():
    _require_admin()
    if not TRAIN_ENABLE:
        return jsonify({"ok": False, "reason": "training disabled"}), 400
    
    try:
        result = train_models()
        return jsonify({"ok": True, "result": result})
    except Exception as e:
        log.exception("Training failed", error=str(e))
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/admin/auto-tune", methods=["POST", "GET"])
def http_auto_tune():
    _require_admin()
    tuned = auto_tune_thresholds(14)
    return jsonify({"ok": True, "tuned": tuned})

@app.route("/admin/retry-unsent", methods=["POST", "GET"])
def http_retry_unsent():
    _require_admin()
    retried = retry_unsent_tips(30, 200)
    return jsonify({"ok": True, "resent": retried})

@app.route("/admin/digest", methods=["POST", "GET"])
def http_digest():
    _require_admin()
    digest = daily_accuracy_digest()
    return jsonify({"ok": True, "sent": bool(digest)})

@app.route("/tips/latest")
def http_latest():
    limit = int(request.args.get("limit", "50"))
    limit = max(1, min(500, limit))
    
    try:
        with db_conn() as cursor:
            cursor.execute("""
                SELECT 
                    match_id, league, home, away,
                    market, suggestion, confidence,
                    confidence_raw, score_at_tip, minute,
                    created_ts, odds, book, ev_pct,
                    risk_level
                FROM tips
                WHERE suggestion != 'HARVEST'
                ORDER BY created_ts DESC
                LIMIT %s
            """, (limit,))
            
            tips = cursor.fetchall()
        
        formatted_tips = []
        for tip in tips:
            formatted_tips.append({
                "match_id": tip['match_id'],
                "league": tip['league'],
                "home": tip['home'],
                "away": tip['away'],
                "market": tip['market'],
                "suggestion": tip['suggestion'],
                "confidence": float(tip['confidence']),
                "confidence_raw": float(tip['confidence_raw']) if tip['confidence_raw'] else None,
                "score_at_tip": tip['score_at_tip'],
                "minute": tip['minute'],
                "created_ts": tip['created_ts'],
                "odds": float(tip['odds']) if tip['odds'] else None,
                "book": tip['book'],
                "ev_pct": float(tip['ev_pct']) if tip['ev_pct'] else None,
                "risk_level": tip['risk_level']
            })
        
        return jsonify({"ok": True, "tips": formatted_tips, "count": len(formatted_tips)})
    
    except Exception as e:
        log.error("Failed to fetch latest tips", error=str(e))
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/admin/stats")
def http_stats():
    _require_admin()
    
    try:
        with db_conn() as cursor:
            # Basic stats
            cursor.execute("SELECT COUNT(*) as count FROM tips WHERE suggestion != 'HARVEST'")
            total_tips = cursor.fetchone()['count']
            
            cursor.execute("SELECT COUNT(*) as count FROM tips WHERE sent_ok = 1")
            sent_tips = cursor.fetchone()['count']
            
            cursor.execute("SELECT COUNT(*) as count FROM match_results")
            results = cursor.fetchone()['count']
            
            # Recent performance
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN r.final_goals_h IS NOT NULL THEN 1 ELSE 0 END) as graded,
                    SUM(CASE 
                        WHEN (t.suggestion LIKE 'Over%' AND (r.final_goals_h + r.final_goals_a) > 
                              CAST(SUBSTRING(t.suggestion FROM '[0-9]+\.?[0-9]*') AS FLOAT))
                        OR (t.suggestion LIKE 'Under%' AND (r.final_goals_h + r.final_goals_a) < 
                              CAST(SUBSTRING(t.suggestion FROM '[0-9]+\.?[0-9]*') AS FLOAT))
                        OR (t.suggestion LIKE 'BTTS: Yes%' AND r.btts_yes = 1)
                        OR (t.suggestion LIKE 'BTTS: No%' AND r.btts_yes = 0)
                        OR (t.suggestion = 'Home Win' AND r.final_goals_h > r.final_goals_a)
                        OR (t.suggestion = 'Away Win' AND r.final_goals_a > r.final_goals_h)
                        THEN 1 ELSE 0 END) as wins
                FROM tips t
                LEFT JOIN match_results r ON r.match_id = t.match_id
                WHERE t.created_ts >= %s
                  AND t.suggestion != 'HARVEST'
            """, (int(time.time()) - 7 * 24 * 3600,))
            
            weekly_stats = cursor.fetchone()
            
            accuracy = 0
            if weekly_stats['graded'] > 0:
                accuracy = (weekly_stats['wins'] / weekly_stats['graded']) * 100
        
        return jsonify({
            "ok": True,
            "stats": {
                "total_tips": total_tips,
                "sent_tips": sent_tips,
                "match_results": results,
                "weekly": {
                    "total": weekly_stats['total'],
                    "graded": weekly_stats['graded'],
                    "wins": weekly_stats['wins'],
                    "accuracy": round(accuracy, 1)
                }
            }
        })
    
    except Exception as e:
        log.error("Failed to fetch stats", error=str(e))
        return jsonify({"ok": False, "error": str(e)}), 500

def _require_admin():
    """Require admin authentication"""
    key = (
        request.headers.get("X-API-Key") or
        request.args.get("key") or
        (request.get_json(silent=True) or {}).get("key")
    )
    
    if not ADMIN_API_KEY or key != ADMIN_API_KEY:
        abort(401)

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8080"))
    
    log.info("Starting Flask application", host=host, port=port)
    app.run(host=host, port=port)
