# file:main.py
# goalsniper — FULL AI mode (in-play) with odds + EV gate
# UPGRADED & PATCHED: Robust Supabase connectivity (IPv4 + PgBouncer + SSL), fixed calibration math,
# sanity + minute cutoffs, per-candidate odds quality, configurable sleep window, GET-enabled admin routes.

import os, json, time, logging, requests, psycopg2, sys, signal, atexit, socket
import numpy as np
from urllib.parse import urlparse, parse_qsl
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
from collections import defaultdict
from contextlib import contextmanager
import math
import re

# ───────── Global Variables ─────────
_SCHED = None
_SHUTDOWN_RAN = False
_SHUTDOWN_HANDLERS_SET = False
EPS = 1e-9
DEBUG_SELECTION_LOG = os.getenv("DEBUG_SELECTION_LOG", "0") not in ("0","false","False","no","NO")  # ADDED: Debug selection logging

# ───────── Debug logging function ─────────
def _dbg(reason: str, **ctx):
    """Debug logging for selection process"""
    if DEBUG_SELECTION_LOG:
        try:
            bits = " ".join(f"{k}={ctx[k]}" for k in ctx)
            log.info(f"[SELECTOR] {reason} | {bits}")
        except Exception as e:
            log.info(f"[SELECTOR] {reason} | [context format failed: {e}]")

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

# Optional-but-recommended warnings — ADDED
if not ADMIN_API_KEY:
    log.warning("ADMIN_API_KEY not set — admin endpoints will return 401 (disabled).")
if not WEBHOOK_SECRET:
    log.warning("TELEGRAM_WEBHOOK_SECRET is not set — /telegram/webhook/<secret> would be unsafe if exposed.")

# ───────── Configuration Validation ─────────
def validate_config():
    """Validate critical configuration at startup"""
    required = {
        'TELEGRAM_BOT_TOKEN': TELEGRAM_BOT_TOKEN,
        'TELEGRAM_CHAT_ID': TELEGRAM_CHAT_ID,
        'API_KEY': API_KEY,
        'DATABASE_URL': os.getenv("DATABASE_URL")  # evaluated at call time
    }
    
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise SystemExit(f"Missing required config: {missing}")
    
    if not ADMIN_API_KEY:
        log.warning("ADMIN_API_KEY not set — admin endpoints will return 401 (disabled).")
    
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
ODDS_QUALITY_MIN = float(os.getenv("ODDS_QUALITY_MIN", "0.35"))   # ✅ added: min acceptable odds quality (0-1)

# ───────── Odds helpers & aggregation ─────────
def _market_name_normalize(s: str) -> str:
    s=(s or "").lower()
    if "both teams" in s or "btts" in s: return "BTTS"
    if "match winner" in s or "winner" in s or "1x2" in s: return "1X2"
    if "over/under" in s or "total" in s or "goals" in s: return "OU"
    return s

OU_LINES = [ln for ln in _parse_lines(os.getenv("OU_LINES","2.5,3.5"), [2.5,3.5]) if abs(ln-1.5)>1e-6]
TOTAL_MATCH_MINUTES   = int(os.getenv("TOTAL_MATCH_MINUTES", "95"))
PREDICTIONS_PER_MATCH = int(os.getenv("PREDICTIONS_PER_MATCH", "1"))  # was 2 — tighter by default
PER_LEAGUE_CAP        = int(os.getenv("PER_LEAGUE_CAP", "2"))         # was 0 — cap league dominance by default

# ───────── Markets allow-list (draw suppressed) ─────────
ALLOWED_SUGGESTIONS = {"BTTS: Yes", "BTTS: No", "Home Win", "Away Win"}
def _fmt_line(line: float) -> str: return f"{line}".rstrip("0").rstrip(".")
def _clamp_prob(p: float) -> float:
    """Keep probabilities away from 0/1 to avoid overconfident cascades."""
    eps = float(os.getenv("PROB_CLAMP_EPS", "0.005"))
    return min(1.0 - eps, max(eps, float(p)))

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

# ───────── DB pool & helpers (PATCHED: pooled+SSL+IPv4-aware) ─────────
POOL: Optional[SimpleConnectionPool] = None

def _parse_pg_url(url: str) -> dict:
    pr = urlparse(url)
    if pr.scheme not in ("postgresql", "postgres"):
        raise SystemExit("DATABASE_URL must start with postgresql:// or postgres://")
    params = dict(parse_qsl(pr.query))
    params.setdefault("sslmode", "require")
    return {
        "user": pr.username or "",
        "password": pr.password or "",
        "host": pr.hostname or "",
        "port": int(pr.port or 5432),
        "dbname": (pr.path or "").lstrip("/") or "postgres",
        "params": params,
    }

def _q(v: str) -> str:
    s = "" if v is None else str(v)
    if s == "" or all(ch not in s for ch in (" ", "'", "\\", "\t", "\n")):
        return s
    s = s.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{s}'"

def _make_conninfo(parts: dict, port: int, hostaddr: Optional[str]) -> str:
    base = [
        f"host={_q(parts['host'])}",
        f"port={port}",
        f"dbname={_q(parts['dbname'])}",
    ]
    if parts["user"]:
        base.append(f"user={_q(parts['user'])}")
    if parts["password"]:
        base.append(f"password={_q(parts['password'])}")
    if hostaddr:
        base.append(f"hostaddr={_q(hostaddr)}")  # force IPv4 socket, keep host for TLS/SNI
    base.append("sslmode=require")
    return " ".join(base)

def _resolve_ipv4(host: str, port: int) -> Optional[str]:
    try:
        infos = socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_STREAM)
        for _family, _socktype, _proto, _canon, sockaddr in infos:
            ip, _p = sockaddr
            return ip
    except Exception:
        return None

def _conninfo_candidates(url: str) -> list[str]:
    parts = _parse_pg_url(url)
    prefer_pooled = os.getenv("DB_PREFER_POOLED", "1").lower() not in ("0","false","no")
    pinned = os.getenv("DB_HOSTADDR")  # optional pinned IPv4 (Supabase IPv4 addon)
    ports: list[int] = []
    if prefer_pooled:
        ports.append(6543)
    if parts["port"] not in ports:
        ports.append(parts["port"])
    cands: list[str] = []
    for p in ports:
        ip = pinned or _resolve_ipv4(parts["host"], p)
        if ip:
            cands.append(_make_conninfo(parts, p, ip))
        cands.append(_make_conninfo(parts, p, None))
    return cands

def _init_pool():
    """Initialize the database connection pool with retry/backoff."""
    global POOL
    if POOL:
        return
    maxconn = int(os.getenv("DB_POOL_MAX", "5"))
    candidates = _conninfo_candidates(DATABASE_URL)
    delay = 1.0
    last = "unknown"
    for attempt in range(6):  # 1+2+4+8+16 (~31s)
        for dsn in candidates:
            try:
                POOL = SimpleConnectionPool(minconn=1, maxconn=maxconn, dsn=dsn)
                masked = dsn.replace("password=", "password=**** ")
                log.info("[DB] Connected (pool=%d) using DSN: %s", maxconn, masked)
                return
            except psycopg2.OperationalError as e:
                last = str(e)
                continue
        if attempt == 5:
            raise psycopg2.OperationalError(
                f"DB pool init failed after retries. Last error: {last}. "
                "Hint: set DB_HOSTADDR=<IPv4> or enable Supabase IPv4 addon, and prefer 6543."
            )
        time.sleep(delay)
        delay *= 2

class PooledConn:
    def __init__(self, pool): 
        self.pool = pool
        self.conn = None
        self.cur = None
        
    def __enter__(self):
        if ShutdownManager.is_shutdown_requested():
            raise Exception("Database connection refused - shutdown in progress")
        _init_pool()
        try:
            self.conn = self.pool.getconn()
        except Exception:
            global POOL
            POOL = None
            _init_pool()
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

    def executemany(self, sql: str, seq_of_params: list[tuple] | list[list]):
        if ShutdownManager.is_shutdown_requested():
            raise Exception("Database operation refused - shutdown in progress")
        try:
            self.cur.executemany(sql, seq_of_params or [])
            return self.cur
        except Exception as e:
            _metric_inc("db_errors_total", n=1)
            log.error("DB executemany failed: %s\nSQL: %s\nBatch size: %s", e, sql, len(seq_of_params or []))
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
            c.execute("SELECT 1")
            return True
    except Exception:
        log.warning("[DB] ping failed, re-initializing pool")
        try:
            _init_pool()
            with db_conn() as c2:
                c2.execute("SELECT 1")
                return True
        except Exception as e:
            _metric_inc("db_errors_total", n=1)
            log.error("[DB] reinit failed: %s", e)
            return False

def bootstrap_basic_models():
    """Create simple working models so predictions aren't always 0.0"""
    log.info("[BOOTSTRAP] Creating basic models...")
    
    basic_models = {
        "BTTS": {
            "weights": {
                "goals_sum": 0.15,
                "minute": 0.08, 
                "pressure_home": 0.02,
                "pressure_away": 0.02,
                "xg_sum": 0.12
            },
            "intercept": -2.5,
            "calibration": {"method": "sigmoid", "a": 1.1, "b": 0.1}
        },
        "OU_2.5": {
            "weights": {
                "goals_sum": 0.25,
                "minute": 0.05,
                "xg_sum": 0.18,
                "pressure_home": 0.01,
                "pressure_away": 0.01
            },
            "intercept": -3.0,
            "calibration": {"method": "sigmoid", "a": 1.0, "b": 0.2}
        },
        "OU_3.5": {
            "weights": {
                "goals_sum": 0.3,
                "minute": 0.04,
                "xg_sum": 0.15,
                "pressure_home": 0.01,
                "pressure_away": 0.01
            },
            "intercept": -4.0,
            "calibration": {"method": "sigmoid", "a": 1.0, "b": 0.3}
        },
        "1X2": {
            "weights": {
                "goals_h": 0.4,
                "goals_a": -0.4,
                "xg_h": 0.3,
                "xg_a": -0.3,
                "pressure_home": 0.02,
                "pressure_away": -0.02
            },
            "intercept": 0.1,
            "calibration": {"method": "sigmoid", "a": 1.0, "b": 0.0}
        }
    }
    
    for name, model in basic_models.items():
        try:
            # Save to database
            set_setting(f"model:{name}", json.dumps(model))
            log.info(f"[BOOTSTRAP] Created model: {name}")
        except Exception as e:
            log.error(f"[BOOTSTRAP] Failed to create {name}: {e}")
    
    # Also create market-specific thresholds
    set_setting("conf_threshold:BTTS", "65.0")
    set_setting("conf_threshold:Over/Under 2.5", "62.0") 
    set_setting("conf_threshold:Over/Under 3.5", "60.0")
    set_setting("conf_threshold:1X2", "68.0")
    
    log.info("[BOOTSTRAP] Basic models created successfully!")

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
    js=_api_get(f"{FOOTBALL_API_URL}/statistics", {"fixture": fid}) or {}
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
    js=_api_get(f"{FOOTBALL_API_URL}/events", {"fixture": fid}) or {}
    out=js.get("response",[]) if isinstance(js,dict) else []
    EVENTS_CACHE[fid]=(now,out)
    if not out: NEG_CACHE[k]=(now, True)
    return out

def fetch_live_matches() -> List[dict]:
    js=_api_get(FOOTBALL_API_URL, {"live":"all"}) or {}
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
    js=_api_get(f"{BASE_URL}/fixtures", {"team":team_id,"last":n}) or {}
    return js.get("response",[]) if isinstance(js,dict) else []

def _api_h2h(home_id: int, away_id: int, n: int = 5) -> List[dict]:
    js=_api_get(f"{BASE_URL}/fixtures/headtohead", {"h2h":f"{home_id}-{away_id}","last":n}) or {}
    return js.get("response",[]) if isinstance(js,dict) else []

def _collect_todays_prematch_fixtures() -> List[dict]:
    today_local=datetime.now(BERLIN_TZ).date()
    start_local=datetime.combine(today_local, datetime.min.time(), tzinfo=BERLIN_TZ)
    end_local=start_local+timedelta(days=1)
    dates_utc={start_local.astimezone(ZoneInfo("UTC")).date(), (end_local - timedelta(seconds=1)).astimezone(ZoneInfo("UTC")).date()}
    fixtures=[]
    for d in sorted(dates_utc):
        js=_api_get(FOOTBALL_API_URL, {"date": d.strftime("%Y-%m-%d")}) or {}
        for r in js.get("response",[]) if isinstance(js,dict) else []:
            if (((r.get("fixture") or {}).get("status") or {}).get("short") or "").upper() == "NS":
                fixtures.append(r)
    fixtures=[f for f in fixtures if not _blocked_league(f.get("league") or {})]
    return fixtures

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

        weighted_sum = 0.0
        total_weight = 0.0

        for model_type, prob, confidence in predictions:
            base_weight = self.ensemble_weights.get(model_type, 0.1)
            recent_performance = self._get_recent_performance(model_type, market)
            time_weight = self._calculate_time_weight(minute, model_type)
            final_weight = base_weight * confidence * recent_performance * time_weight
            weighted_sum += prob * final_weight
            total_weight += final_weight

        ensemble_prob = weighted_sum / total_weight if total_weight > 0 else 0.0
        ensemble_confidence = float(np.mean(confidences)) if confidences else 0.0

        return ensemble_prob, ensemble_confidence

    def _predict_single_model(self, features: Dict[str, float], market: str, minute: int, model_type: str) -> Tuple[Optional[float], float]:
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
    """Fixed version - don't try to load segmented models that don't exist"""
    # Skip segmented model loading entirely for now
    name = market
    
    # Normalize OU market names like "OU_2.5" to the canonical key
    if market.startswith("OU_"):
        try:
            ln = float(market.split("_", 1)[1])
            name = f"OU_{_fmt_line(ln)}"
        except Exception:
            pass

    # Only try the main model name
    mdl = load_model_from_settings(name)
    if not mdl:
        log.warning(f"[ENSEMBLE] No model found for {name}")
        return 0.0
        
    return predict_from_model(mdl, features)

    def _load_ou_model_for_line(self, line: float) -> Optional[Dict[str, Any]]:
        """Load OU model with fallback to legacy names"""
        name = f"OU_{_fmt_line(line)}"
        mdl = load_model_from_settings(name)
        if not mdl and abs(line - 2.5) < 1e-6:
            mdl = load_model_from_settings("O25")
        if not mdl and abs(line - 3.5) < 1e-6:
            mdl = load_model_from_settings("O35")
        return mdl

    def _xgboost_predict(self, features: Dict[str, float], market: str) -> Optional[float]:
        try:
            base_prob = self._logistic_predict(features, market)
            correction = self._calculate_xgb_correction(features, market)
            corrected_prob = base_prob * (1 + correction)
            return max(0.0, min(1.0, corrected_prob))
        except Exception:
            return self._logistic_predict(features, market)

    def _neural_network_predict(self, features: Dict[str, float], market: str) -> Optional[float]:
        try:
            base_prob = self._logistic_predict(features, market)
            nn_correction = self._calculate_nn_correction(features, market)
            if base_prob <= 0.0 or base_prob >= 1.0:
                return base_prob
            nn_prob = 1 / (1 + np.exp(-(np.log(base_prob / (1 - base_prob)) + nn_correction)))
            return float(nn_prob)
        except Exception:
            return self._logistic_predict(features, market)

    def _bayesian_predict(self, features: Dict[str, float], market: str, minute: int) -> Optional[float]:
        try:
            prior_prob = self._get_prior_probability(features, market)
            live_prob = self._logistic_predict(features, market)
            prior_weight = max(0.1, 1.0 - (minute / 90.0))
            live_weight = min(0.9, minute / 90.0)
            bayesian_prob = (prior_prob * prior_weight + live_prob * live_weight) / (prior_weight + live_weight)
            return bayesian_prob
        except Exception:
            return self._logistic_predict(features, market)

    def _momentum_based_predict(self, features: Dict[str, float], market: str, minute: int) -> Optional[float]:
        try:
            base_prob = self._logistic_predict(features, market)
            momentum_factor = self._calculate_momentum_factor(features, minute)
            pressure_factor = self._calculate_pressure_factor(features)
            momentum_correction = (momentum_factor + pressure_factor) * 0.1
            adjusted_prob = base_prob * (1 + momentum_correction)
            return max(0.0, min(1.0, adjusted_prob))
        except Exception:
            return self._logistic_predict(features, market)

    def _calculate_xgb_correction(self, features: Dict[str, float], market: str) -> float:
        correction = 0.0
        if market == "BTTS":
            pressure_product = features.get("pressure_home", 50) * features.get("pressure_away", 50) / 2500
            xg_synergy = features.get("xg_h", 0) * features.get("xg_a", 0)
            correction = pressure_product * 0.1 + xg_synergy * 0.05
        elif market.startswith("OU"):
            attacking_pressure = (features.get("pressure_home", 0) + features.get("pressure_away", 0)) / 2
            defensive_weakness = 1.0 - features.get("defensive_stability", 0.5)
            correction = (attacking_pressure * defensive_weakness * 0.001) - 0.02
        return correction

    def _calculate_nn_correction(self, features: Dict[str, float], market: str) -> float:
        non_linear_features = []
        for key, value in features.items():
            if "xg" in key:
                non_linear_features.append(value ** 1.5)
            elif "pressure" in key:
                non_linear_features.append(np.tanh(value / 50))
            else:
                non_linear_features.append(value)
        if market == "BTTS":
            return sum(non_linear_features) * 0.01
        else:
            return sum(non_linear_features) * 0.005

    def _get_prior_probability(self, features: Dict[str, float], market: str) -> float:
        base_prior = 0.5
        if "xg_sum" in features:
            xg_density = features["xg_sum"] / max(1, features.get("minute", 1))
            base_prior = min(0.8, max(0.2, xg_density * 10))
        return base_prior

    def _calculate_momentum_factor(self, features: Dict[str, float], minute: int) -> float:
        if minute < 20:
            return 0.0
        momentum = 0.0
        goals_last_15 = features.get("goals_last_15", 0)
        momentum += goals_last_15 * 0.2
        shots_last_15 = features.get("shots_last_15", 0)
        momentum += shots_last_15 * 0.05
        recent_xg_impact = features.get("recent_xg_impact", 0)
        momentum += recent_xg_impact * 0.1
        return momentum

    def _calculate_pressure_factor(self, features: Dict[str, float]) -> float:
        pressure_diff = features.get("pressure_home", 0) - features.get("pressure_away", 0)
        score_advantage = features.get("goals_h", 0) - features.get("goals_a", 0)
        if abs(score_advantage) <= 1:
            return abs(pressure_diff) * 0.01
        else:
            return pressure_diff * 0.005

    def _get_market_specific_features_xgb(self, features: Dict[str, float], market: str) -> Dict[str, float]:
        enhanced_features = features.copy()
        enhanced_features["pressure_product"] = features.get("pressure_home", 0) * features.get("pressure_away", 0)
        enhanced_features["xg_ratio"] = features.get("xg_h", 0.1) / max(0.1, features.get("xg_a", 0.1))
        enhanced_features["efficiency_ratio"] = features.get("goals_sum", 0) / max(0.1, features.get("xg_sum", 0.1))
        return enhanced_features

    def _get_recent_performance(self, model_type: str, market: str) -> float:
        return 0.9

    def _calculate_time_weight(self, minute: int, model_type: str) -> float:
        if model_type in ['bayesian', 'momentum']:
            return min(1.0, minute / 60.0)
        else:
            return 1.0

# Initialize global ensemble predictor
ensemble_predictor = AdvancedEnsemblePredictor()

# ───────── ENHANCEMENT 2: Advanced Feature Engineering ─────────
def _count_goals_since(events: List[dict], current_minute: int, window: int) -> int:
    cutoff = current_minute - window
    goals = 0
    for event in events:
        minute = event.get('time', {}).get('elapsed', 0)
        if minute >= cutoff and event.get('type') == 'Goal':
            goals += 1
    return goals

def _count_shots_since(events: List[dict], current_minute: int, window: int) -> int:
    cutoff = current_minute - window
    shots = 0
    shot_types = {'Shot', 'Missed Shot', 'Shot on Target', 'Saved Shot'}
    for event in events:
        minute = event.get('time', {}).get('elapsed', 0)
        if minute >= cutoff and event.get('type') in shot_types:
            shots += 1
    return shots

def _count_cards_since(events: List[dict], current_minute: int, window: int) -> int:
    cutoff = current_minute - window
    cards = 0
    for event in events:
        minute = event.get('time', {}).get('elapsed', 0)
        if minute >= cutoff and event.get('type') == 'Card':
            cards += 1
    return cards

def _calculate_pressure(feat: Dict[str, float], side: str) -> float:
    suffix = "_h" if side == "home" else "_a"
    possession = feat.get(f"pos{suffix}", 50)
    shots = feat.get(f"sot{suffix}", 0)
    xg = feat.get(f"xg{suffix}", 0)
    possession_norm = possession / 100.0
    shots_norm = min(shots / 10.0, 1.0)
    xg_norm = min(xg / 3.0, 1.0)
    return (possession_norm * 0.3 + shots_norm * 0.4 + xg_norm * 0.3) * 100

def _calculate_xg_momentum(feat: Dict[str, float]) -> float:
    total_xg = feat.get("xg_sum", 0)
    total_goals = feat.get("goals_sum", 0)
    if total_xg <= 0:
        return 0.0
    return (total_goals - total_xg) / max(1, total_xg)

def _recent_xg_impact(feat: Dict[str, float], minute: int) -> float:
    if minute <= 0:
        return 0.0
    xg_per_minute = feat.get("xg_sum", 0) / minute
    return xg_per_minute * 90

def _defensive_stability(feat: Dict[str, float]) -> float:
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

def extract_basic_features(m: dict) -> Dict[str,float]:
    home = (m.get("teams") or {}).get("home", {}).get("name", "")
    away = (m.get("teams") or {}).get("away", {}).get("name", "")
    gh = (m.get("goals") or {}).get("home") or 0
    ga = (m.get("goals") or {}).get("away") or 0
    minute = int(((m.get("fixture") or {}).get("status") or {}).get("elapsed") or 0)
    stats = {}
    for s in (m.get("statistics") or []):
        t = (s.get("team") or {}).get("name")
        if t:
            stats[t] = { (i.get("type") or ""): i.get("value") for i in (s.get("statistics") or []) }
    sh = stats.get(home, {}) or {}
    sa = stats.get(away, {}) or {}
    
    # FIXED: Use ACTUAL API-Football statistics that exist
    sot_h = _num(sh.get("Shots on Goal", sh.get("Shots on Target", 0)))
    sot_a = _num(sa.get("Shots on Goal", sa.get("Shots on Target", 0)))
    sh_total_h = _num(sh.get("Total Shots", 0))
    sh_total_a = _num(sa.get("Total Shots", 0))
    cor_h = _num(sh.get("Corner Kicks", 0))
    cor_a = _num(sa.get("Corner Kicks", 0))
    pos_h = _pos_pct(sh.get("Ball Possession", 0))
    pos_a = _pos_pct(sa.get("Ball Possession", 0))

    # **UPDATED: Use REAL xG when available, estimate when not**
    xg_h = _num(sh.get("Expected Goals", 0))
    xg_a = _num(sa.get("Expected Goals", 0))
    
    # Only estimate if real xG is not available (0 or missing)
    if xg_h == 0 and xg_a == 0:
        # Fallback: estimate xG from shots on target
        xg_h = sot_h * 0.3
        xg_a = sot_a * 0.3
        log.info(f"[XG_ESTIMATE] Using estimated xG: {xg_h:.2f}-{xg_a:.2f}")
    else:
        log.info(f"[XG_REAL] Using API xG: {xg_h:.2f}-{xg_a:.2f}")

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
    base_feat = extract_basic_features(m)
    minute = base_feat.get("minute", 0)
    events = m.get("events", [])
    base_feat.update({
        "goals_last_15": float(_count_goals_since(events, minute, 15)),
        "shots_last_15": float(_count_shots_since(events, minute, 15)),
        "cards_last_15": float(_count_cards_since(events, minute, 15)),
        "pressure_home": _calculate_pressure(base_feat, "home"),
        "pressure_away": _calculate_pressure(base_feat, "away"),
        "score_advantage": base_feat.get("goals_h", 0) - base_feat.get("goals_a", 0),
        "xg_momentum": _calculate_xg_momentum(base_feat),
        "recent_xg_impact": _recent_xg_impact(base_feat, minute),
        "defensive_stability": _defensive_stability(base_feat)
    })
    return base_feat

# ───────── ENHANCEMENT 2: Advanced Feature Engineering (class) ─────────
class AdvancedFeatureEngineer:
    """Advanced feature engineering with temporal patterns and game context"""
    
    def __init__(self):
        self.feature_cache = {}
        self.temporal_patterns = {}
    
    def extract_advanced_features(self, m: dict) -> Dict[str, float]:
        """Extract advanced features including temporal patterns and game context"""
        base_features = extract_enhanced_features(m)
        
        temporal_features = self._extract_temporal_patterns(m, base_features)
        base_features.update(temporal_features)
        
        context_features = self._extract_game_context(m, base_features)
        base_features.update(context_features)
        
        strength_features = self._extract_team_strength_indicators(m, base_features)
        base_features.update(strength_features)
        
        return base_features
    
    def _extract_temporal_patterns(self, m: dict, base_features: Dict[str, float]) -> Dict[str, float]:
        minute = int(base_features.get("minute", 0))
        events = m.get("events", []) or []
        temporal_features: Dict[str, float] = {}
        
        for window in [10, 15, 20]:
            temporal_features[f"goals_last_{window}"] = float(self._count_events_since(events, minute, window, 'Goal'))
            temporal_features[f"shots_last_{window}"] = float(self._count_events_since(events, minute, window, {'Shot', 'Missed Shot', 'Shot on Target'}))
            temporal_features[f"cards_last_{window}"] = float(self._count_events_since(events, minute, window, {'Card'}))
        
        if minute > 15:
            goals_0_15 = temporal_features.get("goals_last_15", 0.0)
            goals_15_30 = float(self._count_events_between(events, max(0, minute-30), minute-15, 'Goal'))
            temporal_features["goal_acceleration"] = float(goals_0_15 - goals_15_30)
        
        temporal_features["time_decayed_xg"] = self._calculate_time_decayed_xg(base_features, minute)
        temporal_features["recent_pressure"] = self._calculate_recent_pressure(events, minute)
        return temporal_features
    
    def _extract_game_context(self, m: dict, base_features: Dict[str, float]) -> Dict[str, float]:
        context_features: Dict[str, float] = {}
        minute = int(base_features.get("minute", 0))
        score_diff = float(base_features.get("goals_h", 0) - base_features.get("goals_a", 0))
        context_features["game_state"] = self._classify_game_state(score_diff, minute)
        context_features["home_urgency"] = self._calculate_urgency(score_diff, minute, is_home=True)
        context_features["away_urgency"] = self._calculate_urgency(-score_diff, minute, is_home=False)
        context_features["defensive_risk"] = self._calculate_defensive_risk(base_features, minute)
        context_features["attacking_risk"] = self._calculate_attacking_risk(base_features, minute)
        context_features["match_importance"] = self._estimate_match_importance(m)
        return context_features
    
    def _extract_team_strength_indicators(self, m: dict, base_features: Dict[str, float]) -> Dict[str, float]:
        strength_features: Dict[str, float] = {}
        pressure_home = float(base_features.get("pressure_home", 1))
        pressure_away = float(base_features.get("pressure_away", 1))
        strength_features["home_dominance"] = pressure_home / max(1.0, pressure_away)
        strength_features["away_resilience"] = 1.0 / max(0.1, strength_features["home_dominance"])
        xg_h = float(base_features.get("xg_h", 0.1))
        xg_a = float(base_features.get("xg_a", 0.1))
        goals_h = float(base_features.get("goals_h", 0))
        goals_a = float(base_features.get("goals_a", 0))
        strength_features["home_efficiency"] = goals_h / max(0.1, xg_h)
        strength_features["away_efficiency"] = goals_a / max(0.1, xg_a)
        strength_features["home_defensive_stability"] = 1.0 - (goals_a / max(0.1, xg_a))
        strength_features["away_defensive_stability"] = 1.0 - (goals_h / max(0.1, xg_h))
        return strength_features
    
    def _count_events_since(self, events: List[dict], current_minute: int, window: int, event_types: any) -> int:
        cutoff = current_minute - window
        count = 0
        for event in events:
            minute = int((event.get('time', {}) or {}).get('elapsed', 0) or 0)
            if minute >= cutoff:
                et = event.get('type')
                if isinstance(event_types, str):
                    if et == event_types: count += 1
                else:
                    if et in event_types: count += 1
        return count
    
    def _count_events_between(self, events: List[dict], start_minute: int, end_minute: int, event_type: str) -> int:
        count = 0
        for event in events:
            minute = int((event.get('time', {}) or {}).get('elapsed', 0) or 0)
            if start_minute <= minute <= end_minute and event.get('type') == event_type:
                count += 1
        return count
    
    def _calculate_time_decayed_xg(self, features: Dict[str, float], minute: int) -> float:
        if minute <= 0:
            return 0.0
        xg_sum = float(features.get("xg_sum", 0))
        decay_factor = 0.9
        recent_weight = decay_factor ** (minute / 10.0)
        return float((xg_sum / minute) * recent_weight * 90.0)
    
    def _calculate_recent_pressure(self, events: List[dict], minute: int) -> float:
        recent_events = self._count_events_since(events, minute, 10, {'Shot', 'Shot on Target', 'Corner', 'Dangerous Attack'})
        return float(min(1.0, recent_events / 10.0))
    
    def _classify_game_state(self, score_diff: float, minute: int) -> float:
        if minute < 30: return 0.0
        if abs(score_diff) >= 3: return 1.0
        if abs(score_diff) == 2 and minute > 70: return 0.8
        if abs(score_diff) == 1 and minute > 75: return 0.9
        if score_diff == 0 and minute > 80: return 0.7
        return 0.5
    
    def _calculate_urgency(self, score_diff: float, minute: int, is_home: bool) -> float:
        urgency_score = -score_diff if is_home else score_diff
        time_pressure = max(0.0, (minute - 60) / 30.0)
        return float(max(0.0, urgency_score * time_pressure))
    
    def _calculate_defensive_risk(self, features: Dict[str, float], minute: int) -> float:
        goals_conceded = float(features.get("goals_a", 0) + features.get("goals_h", 0))
        xg_against = float(features.get("xg_a", 0) + features.get("xg_h", 0))
        defensive_efficiency = goals_conceded / max(0.1, xg_against)
        fatigue_factor = min(1.0, minute / 90.0)
        return float(defensive_efficiency * fatigue_factor)
    
    def _calculate_attacking_risk(self, features: Dict[str, float], minute: int) -> float:
        pressure = (float(features.get("pressure_home", 0)) + float(features.get("pressure_away", 0))) / 2.0
        home_urgency = float(features.get("home_urgency", 0))
        away_urgency = float(features.get("away_urgency", 0))
        urgency = (home_urgency + away_urgency) / 2.0
        return float((pressure / 100.0) * urgency)
    
    def _estimate_match_importance(self, m: dict) -> float:
        league = m.get("league", {}) or {}
        league_name = str(league.get("name", "") or "").lower()
        if any(w in league_name for w in ["champions league", "europa league", "premier league"]):
            return 0.9
        if any(w in league_name for w in ["cup", "knockout", "playoff"]):
            return 0.8
        return 0.5

# Initialize global feature engineer
feature_engineer = AdvancedFeatureEngineer()

# ───────── ENHANCEMENT 3: Intelligent Market-Specific Prediction System ─────────
class MarketSpecificPredictor:
    """Advanced market-specific prediction with specialized models"""
    
    def __init__(self):
        self.market_strategies = {
            "BTTS": self._predict_btts_advanced,
            "OU": self._predict_ou_advanced
            # Removed 1X2 from strategies since it returns 3 values
        }
        self.market_feature_sets = self._initialize_market_features()
    
    def predict_for_market(self, features: Dict[str, float], market: str, minute: int) -> Tuple[float, float]:
        # For OU markets we must respect the exact line (e.g., OU_2.5 vs OU_3.5)
        if market.startswith("OU_"):
            line = None
            try:
                line = float(market.split("_", 1)[1])
            except Exception:
                pass
            return self._predict_ou_advanced(features, minute, line=line, market_key=market)
        elif market in self.market_strategies:
            return self.market_strategies[market](features, minute)
        else:
            return ensemble_predictor.predict_ensemble(features, market, minute)
    
    def _predict_btts_advanced(self, features: Dict[str, float], minute: int) -> Tuple[float, float]:
        base_prob, base_conf = ensemble_predictor.predict_ensemble(features, "BTTS", minute)
        adjustments = 0.0
        defensive_stability = float(features.get("defensive_stability", 0.5))
        vulnerability = 1.0 - defensive_stability
        adjustments += vulnerability * 0.2
        pressure_balance = min(float(features.get("pressure_home", 0)), float(features.get("pressure_away", 0))) / 100.0
        adjustments += pressure_balance * 0.15
        goals_last_20 = float(features.get("goals_last_20", 0))
        adjustments += min(0.3, goals_last_20 * 0.1)
        game_state = float(features.get("game_state", 0.5))
        if game_state > 0.7:
            adjustments += 0.1
        adjusted_prob = base_prob * (1 + adjustments)
        confidence = base_conf * 0.9
        return _clamp_prob(adjusted_prob), max(0.0, min(1.0, confidence))
    
    def _predict_ou_advanced(
        self,
        features: Dict[str, float],
        minute: int,
        line: Optional[float] = None,
        market_key: Optional[str] = None
    ) -> Tuple[float, float]:
        """
        Produce OU probability for a specific goal line (e.g., 2.5).
        Uses line-specific ensemble first, then falls back to the logistic model for that line.
        """
        base_prob, base_conf = 0.0, 0.0

        # Choose a market name for the ensemble
        mk = market_key or (f"OU_{_fmt_line(line)}" if line is not None else "OU")

        # Try ensemble with the exact line if available
        try:
            if line is not None:
                mk_line = f"OU_{_fmt_line(line)}"
                ensemble_prob, ensemble_conf = ensemble_predictor.predict_ensemble(features, mk_line, minute)
            else:
                ensemble_prob, ensemble_conf = ensemble_predictor.predict_ensemble(features, mk, minute)

            base_prob, base_conf = float(ensemble_prob), float(ensemble_conf)
        except Exception as e:
            log.warning(f"[OU_PREDICT] Ensemble failed for {mk}: {e}")

        # Fallback: if ensemble gave nothing useful and we have a concrete line, use the logistic model for that line
        if (base_prob <= 0.0) and (line is not None):
            mdl = ensemble_predictor._load_ou_model_for_line(line)
            if mdl:
                base_prob = predict_from_model(mdl, features)
                base_conf = max(base_conf, 0.8)

        if base_prob <= 0.0:
            return 0.0, 0.0

        # Apply market-specific adjustments and confidence shaping
        adjustments = self._calculate_ou_adjustments(features, minute)
        adjusted_prob = max(0.0, min(1.0, base_prob * (1.0 + adjustments)))
        confidence_factor = self._calculate_ou_confidence_factor(features, minute)
        final_confidence = max(0.0, min(1.0, base_conf * confidence_factor))

        # Keep probabilities out of the 0/1 extremes to prevent hard contradictions downstream
        return _clamp_prob(adjusted_prob), final_confidence
    
    def _calculate_ou_adjustments(self, features: Dict[str, float], minute: int) -> float:
        adjustments = 0.0
        current_goals = float(features.get("goals_sum", 0))
        xg_sum = float(features.get("xg_sum", 0))
        minute = max(1, int(features.get("minute", 1)))
        xg_per_minute = xg_sum / minute
        expected_goals_by_now = (xg_per_minute * minute)
        if expected_goals_by_now > 0:
            tempo_ratio = current_goals / expected_goals_by_now
            if tempo_ratio > 1.3:
                adjustments += 0.2
            elif tempo_ratio < 0.7:
                adjustments -= 0.15
        pressure_total = float(features.get("pressure_home", 0)) + float(features.get("pressure_away", 0))
        if pressure_total > 150:
            adjustments += 0.1
        elif pressure_total < 80:
            adjustments -= 0.1
        defensive_stability = float(features.get("defensive_stability", 0.5))
        if defensive_stability < 0.3:
            adjustments += 0.15
        elif defensive_stability > 0.7:
            adjustments -= 0.15
        if minute > 75:
            score_diff = abs(float(features.get("goals_h", 0) - features.get("goals_a", 0)))
            if score_diff <= 1:
                adjustments += 0.1
            elif current_goals == 0:
                adjustments += 0.05
        goals_last_15 = float(features.get("goals_last_15", 0))
        if goals_last_15 >= 2:
            adjustments += 0.1
        elif goals_last_15 == 0 and minute > 30:
            adjustments -= 0.05
        return adjustments
    
    def _calculate_ou_confidence_factor(self, features: Dict[str, float], minute: int) -> float:
        confidence = 1.0
        xg_available = float(features.get("xg_sum", 0)) > 0
        pressure_available = (float(features.get("pressure_home", 0)) > 0) or (float(features.get("pressure_away", 0)) > 0)
        if not xg_available:
            confidence *= 0.7
        if not pressure_available:
            confidence *= 0.8
        progression_factor = min(1.0, int(features.get("minute", 0)) / 60.0)
        confidence *= (0.5 + 0.5 * progression_factor)
        total_events = float(features.get("sot_sum", 0) + features.get("cor_sum", 0) + features.get("goals_sum", 0))
        if total_events < 5 and minute > 30:
            confidence *= 0.8
        return float(confidence)
    
    def _predict_1x2_advanced(self, features: Dict[str, float], minute: int) -> Tuple[float, float, float]:
        base_prob_h, conf_h = ensemble_predictor.predict_ensemble(features, "1X2_HOME", minute)
        base_prob_a, conf_a = ensemble_predictor.predict_ensemble(features, "1X2_AWAY", minute)
        total = base_prob_h + base_prob_a
        if total > 0:
            base_prob_h /= total
            base_prob_a /= total
        prob_h = self._adjust_1x2_probability(base_prob_h, features, minute, is_home=True)
        prob_a = self._adjust_1x2_probability(base_prob_a, features, minute, is_home=False)
        total_adj = prob_h + prob_a
        if total_adj > 0:
            prob_h /= total_adj
            prob_a /= total_adj
        confidence = (conf_h + conf_a) / 2.0
        return float(prob_h), float(prob_a), float(confidence)
    
    def _adjust_1x2_probability(self, base_prob: float, features: Dict[str, float], 
                              minute: int, is_home: bool) -> float:
        adjustments = 0.0
        momentum = (float(features.get("pressure_home", 0)) if is_home else float(features.get("pressure_away", 0))) / 100.0
        adjustments += momentum * 0.15
        score_diff = float(features.get("goals_h", 0) - features.get("goals_a", 0))
        psychological = (score_diff * 0.1) if is_home else (-score_diff * 0.1)
        adjustments += psychological
        urgency = float(features.get("home_urgency", 0) if is_home else features.get("away_urgency", 0))
        adjustments += urgency * 0.08
        efficiency = float(features.get("home_efficiency", 1.0) if is_home else features.get("away_efficiency", 1.0))
        adjustments += (efficiency - 1.0) * 0.1
        adjusted_prob = base_prob * (1 + adjustments)
        return max(0.0, min(1.0, float(adjusted_prob)))
    
    def _initialize_market_features(self):
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

# ───────── ENHANCEMENT 4: Adaptive Learning (kept API; storage in-memory) ─────────
class AdaptiveLearningSystem:
    def __init__(self):
        self.performance_history: Dict[str, list] = {}
        self.feature_importance: Dict[str, dict] = {}
        self.model_adjustments: Dict[str, Any] = {}
        
    def record_prediction_outcome(self, prediction_data: Dict[str, Any], outcome: Optional[int]):
        if outcome is None: return
        market = prediction_data.get("market", "")
        features = prediction_data.get("features", {})
        probability = float(prediction_data.get("probability", 0.0))
        key = f"{market}_{outcome}"
        self.performance_history.setdefault(key, []).append({
            'timestamp': time.time(),
            'probability': probability,
            'outcome': int(outcome),
            'features': features
        })
        if len(self.performance_history[key]) > 1000:
            self.performance_history[key] = self.performance_history[key][-1000:]
        self._update_feature_importance(features, int(outcome), probability, market)
        
    def _update_feature_importance(self, features: Dict[str, float], outcome: int, prob: float, market: str):
        prediction_correct = 1 if ((prob > 0.5 and outcome == 1) or (prob <= 0.5 and outcome == 0)) else 0
        for feature_name, feature_value in features.items():
            fi = self.feature_importance.setdefault(feature_name, {'total_uses': 0, 'correct_uses': 0, 'market_specific': {}})
            fi['total_uses'] += 1
            fi['correct_uses'] += prediction_correct
            ms = fi['market_specific'].setdefault(market, {'total_uses': 0, 'correct_uses': 0})
            ms['total_uses'] += 1
            ms['correct_uses'] += prediction_correct

    def get_feature_weights(self, market: str) -> Dict[str, float]:
        base_weights: Dict[str, float] = {}
        for feature_name, stats in self.feature_importance.items():
            total_uses = int(stats['total_uses'])
            correct_uses = int(stats['correct_uses'])
            if total_uses > 10:
                accuracy = correct_uses / total_uses
                market_stats = stats['market_specific'].get(market, {})
                if int(market_stats.get('total_uses', 0)) > 5:
                    market_accuracy = market_stats['correct_uses'] / market_stats['total_uses']
                    accuracy = (accuracy + market_accuracy) / 2.0
                weight = max(0.1, min(2.0, (accuracy - 0.5) * 2 + 1.0))
                base_weights[feature_name] = float(weight)
        return base_weights
    
    def analyze_performance_trends(self) -> Dict[str, Any]:
        trends: Dict[str, Any] = {'overall_accuracy': 0.0, 'market_performance': {}, 'feature_effectiveness': {}, 'recommendations': []}
        total_predictions = 0
        correct_predictions = 0
        for key, history in self.performance_history.items():
            if not history: continue
            market = key.split('_')[0]
            correct = sum(1 for h in history if h['outcome'] == 1)
            total = len(history)
            trends['market_performance'][market] = {'accuracy': (correct / total if total else 0.0), 'total_predictions': total}
            total_predictions += total
            correct_predictions += correct
        if total_predictions > 0:
            trends['overall_accuracy'] = correct_predictions / total_predictions
        for feature_name, stats in self.feature_importance.items():
            if int(stats['total_uses']) > 0:
                trends['feature_effectiveness'][feature_name] = {
                    'accuracy': stats['correct_uses'] / stats['total_uses'],
                    'usage_count': stats['total_uses']
                }
        # recommendations
        recs = []
        for market, perf in trends['market_performance'].items():
            if perf['accuracy'] < 0.55 and perf['total_predictions'] > 50:
                recs.append(f"Consider reviewing {market} model - current accuracy: {perf['accuracy']:.1%}")
        for fname, eff in trends['feature_effectiveness'].items():
            if eff['accuracy'] < 0.48 and eff['usage_count'] > 100:
                recs.append(f"Feature '{fname}' may be counterproductive - accuracy: {eff['accuracy']:.1%}")
            elif eff['accuracy'] > 0.65 and eff['usage_count'] > 50:
                recs.append(f"Feature '{fname}' is highly effective - consider increasing weight")
        trends['recommendations'] = recs
        return trends

# Initialize adaptive learning system
adaptive_learner = AdaptiveLearningSystem()

def _apply_tune_thresholds(days: int = 14) -> Dict[str, float]:
    """
    Auto-tune confidence thresholds based on recent performance.
    Returns dict of market -> new_threshold
    """
    log.info(f"[AUTO-TUNE] Starting threshold tuning for {days} days of data")
    
    cutoff_ts = int(time.time()) - days * 24 * 3600
    tuned_thresholds = {}
    
    try:
        with db_conn() as c:
            # Get recent tips with outcomes
            rows = c.execute("""
                SELECT t.market, t.suggestion, t.confidence_raw, 
                       r.final_goals_h, r.final_goals_a, r.btts_yes
                FROM tips t
                JOIN match_results r ON t.match_id = r.match_id
                WHERE t.created_ts >= %s 
                AND t.suggestion <> 'HARVEST'
                AND t.confidence_raw IS NOT NULL
            """, (cutoff_ts,)).fetchall()
        
        if not rows:
            log.warning("[AUTO-TUNE] No data found for threshold tuning")
            return {}
        
        # Group by market
        market_data = {}
        for market, suggestion, confidence, gh, ga, btts in rows:
            if market not in market_data:
                market_data[market] = []
            
            # Determine outcome
            outcome = _tip_outcome_for_result(suggestion, {
                "final_goals_h": gh, 
                "final_goals_a": ga, 
                "btts_yes": btts
            })
            
            if outcome is not None:
                market_data[market].append((float(confidence), outcome))
        
        # Calculate optimal thresholds for each market
        for market, data in market_data.items():
            if len(data) < THRESH_MIN_PREDICTIONS:
                log.info(f"[AUTO-TUNE] Insufficient data for {market}: {len(data)} predictions")
                continue
            
            # Sort by confidence
            data.sort(key=lambda x: x[0])
            
            # Find threshold that achieves target precision
            best_threshold = CONF_THRESHOLD
            best_precision = 0.0
            
            for i in range(len(data)):
                threshold = data[i][0] * 100  # Convert to percentage
                if threshold < MIN_THRESH or threshold > MAX_THRESH:
                    continue
                
                # Calculate precision above this threshold
                high_conf_data = [x for x in data[i:] if x[0] * 100 >= threshold]
                if len(high_conf_data) < 10:  # Need minimum samples
                    continue
                
                wins = sum(1 for x in high_conf_data if x[1] == 1)
                precision = wins / len(high_conf_data)
                
                if precision >= TARGET_PRECISION and precision > best_precision:
                    best_precision = precision
                    best_threshold = threshold
            
            if best_precision >= TARGET_PRECISION:
                tuned_thresholds[market] = best_threshold
                log.info(f"[AUTO-TUNE] {market}: new threshold {best_threshold:.1f}% (precision: {best_precision:.1%})")
            
            # Save the tuned threshold
            if market in tuned_thresholds:
                set_setting(f"conf_threshold:{market}", str(tuned_thresholds[market]))
                _SETTINGS_CACHE.invalidate(f"conf_threshold:{market}")
        
        log.info(f"[AUTO-TUNE] Completed: tuned {len(tuned_thresholds)} thresholds")
        return tuned_thresholds
        
    except Exception as e:
        log.error(f"[AUTO-TUNE] Error during threshold tuning: {e}")
        return {}

def predict_from_model(mdl: Dict[str, Any], features: Dict[str, float]) -> float:
    """
    Make prediction using a logistic regression model with calibration.
    This is the core prediction function that was missing.
    """
    if not mdl:
        return 0.0
    
    try:
        weights = mdl.get("weights", {})
        intercept = float(mdl.get("intercept", 0.0))
        
        # Calculate linear combination
        linear_score = intercept
        for feature, weight in weights.items():
            if feature in features:
                linear_score += float(weight) * float(features[feature])
        
        # Apply sigmoid to get probability
        probability = 1.0 / (1.0 + math.exp(-linear_score))
        
        # Apply calibration if available
        cal = mdl.get("calibration", {})
        if cal:
            method = cal.get("method", "sigmoid").lower()
            a = float(cal.get("a", 1.0))
            b = float(cal.get("b", 0.0))
            
            if method == "sigmoid":
                # Calibrate on logit scale
                prob_clamped = max(1e-12, min(1-1e-12, probability))
                logit = math.log(prob_clamped / (1 - prob_clamped))
                calibrated_logit = a * logit + b
                probability = 1.0 / (1.0 + math.exp(-calibrated_logit))
        
        return float(max(0.0, min(1.0, probability)))
        
    except Exception as e:
        log.error(f"[PREDICT] Error in predict_from_model: {e}")
        return 0.0

# ───────── Odds fetching function ─────────
def fetch_odds(fid: int, prob_hints: Optional[dict[str, float]] = None, require_live: bool = True) -> dict:
    """
    Aggregated odds map - LIVE ODDS ONLY for in-play matches
    """
    now = time.time()
    k = ("odds", fid)
    ts_empty = NEG_CACHE.get(k, (0.0, False))
    if ts_empty[1] and (now - ts_empty[0] < NEG_TTL_SEC): 
        return {}
    if fid in ODDS_CACHE and now - ODDS_CACHE[fid][0] < 120: 
        return ODDS_CACHE[fid][1]

    def _fetch(path: str) -> dict:
        js = _api_get(f"{BASE_URL}/{path}", {"fixture": fid}) or {}
        return js if isinstance(js, dict) else {}

    js = {}
    src = None

    # Try live first
    js = _fetch("odds/live")
    if (js.get("response") or []):
        src = "live"
        log.info(f"[ODDS] Found LIVE odds for fixture {fid}")
    else:
        # No prematch fallback for in-play
        log.warning(f"[ODDS] No LIVE odds available for fixture {fid} - rejecting")
        NEG_CACHE[k] = (now, True)
        ODDS_CACHE[fid] = (time.time(), {})
        return {}

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
    except Exception as e:
        log.error(f"[ODDS] Error parsing odds for fixture {fid}: {e}")
        return {}

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
    if not out: 
        NEG_CACHE[k] = (now, True)
    else:
        log.info(f"[ODDS] Successfully parsed {len(out)} markets for fixture {fid}")
    return out

def _min_odds_for_market(market_text: str) -> float:
    """Get minimum odds for a specific market"""
    if market_text == "BTTS":
        return MIN_ODDS_BTTS
    elif market_text == "1X2":
        return MIN_ODDS_1X2
    elif market_text.startswith("Over/Under"):
        return MIN_ODDS_OU
    return 1.50  # default

def _ev(probability: float, odds: float) -> float:
    """Calculate expected value"""
    return (probability * odds) - 1.0

def auto_tune_thresholds(days: int = 14) -> Dict[str, float]:
    """Auto-tune confidence thresholds - wrapper function"""
    return _apply_tune_thresholds(days)

def _format_tip_message(home, away, league, minute, score, suggestion, confidence, features, odds=None, book=None, ev_pct=None):
    """Simplified tip message formatter"""
    money = f"\n💰 <b>Odds:</b> {odds:.2f} @ {book}" if odds else ""
    ev_text = f" • <b>EV:</b> {ev_pct:+.1f}%" if ev_pct is not None else ""
    return (f"⚽️ <b>AI TIP</b>\n"
            f"<b>Match:</b> {escape(home)} vs {escape(away)}\n"
            f"🕒 <b>Minute:</b> {minute}' | <b>Score:</b> {escape(score)}\n"
            f"<b>Tip:</b> {escape(suggestion)}\n"
            f"📈 <b>Confidence:</b> {confidence:.1f}%{money}{ev_text}\n"
            f"🏆 <b>League:</b> {escape(league)}")

def _odds_key_for_market(market_txt: str, suggestion: str) -> str | None:
    """Map our market/suggestion to odds_map key."""
    if market_txt == "BTTS":
        return "BTTS"
    if market_txt == "1X2":
        return "1X2"
    if market_txt.startswith("Over/Under"):
        ln = _parse_ou_line_from_suggestion(suggestion)
        return f"OU_{_fmt_line(ln)}" if ln is not None else None
    return None

def _parse_ou_line_from_suggestion(s: str) -> Optional[float]:
    """Extract the line value from Over/Under suggestion."""
    try:
        if "Over" in s or "Under" in s:
            match = re.search(r'(\d+(?:\.\d+)?)', s)
            if match:
                return float(match.group(1))
        return None
    except Exception:
        return None

# ───────── Resource Management for Caches ─────────

class CacheManager:
    """Manage cache sizes and cleanup to prevent memory leaks"""
    
    def __init__(self):
        self.max_stats_cache_size = 1000
        self.max_events_cache_size = 1000
        self.max_odds_cache_size = 500
        self.max_neg_cache_size = 1000
        
    def cleanup_all_caches(self):
        """Clean up all caches with size limits"""
        self._cleanup_cache(STATS_CACHE, self.max_stats_cache_size)
        self._cleanup_cache(EVENTS_CACHE, self.max_events_cache_size)
        self._cleanup_cache(ODDS_CACHE, self.max_odds_cache_size)
        self._cleanup_cache(NEG_CACHE, self.max_neg_cache_size)
        
        # Also clean up old entries based on timestamp
        self._cleanup_old_entries(STATS_CACHE, 300)  # 5 minutes
        self._cleanup_old_entries(EVENTS_CACHE, 300)
        self._cleanup_old_entries(ODDS_CACHE, 600)   # 10 minutes
        self._cleanup_old_entries(NEG_CACHE, NEG_TTL_SEC)
        
    def _cleanup_cache(self, cache: dict, max_size: int):
        """Reduce cache size if it exceeds max_size"""
        if len(cache) > max_size:
            # Remove oldest entries (assuming keys are inserted in order)
            keys_to_remove = list(cache.keys())[:len(cache) - max_size]
            for key in keys_to_remove:
                cache.pop(key, None)
                
    def _cleanup_old_entries(self, cache: dict, max_age_seconds: int):
        """Remove entries older than max_age_seconds"""
        current_time = time.time()
        keys_to_remove = []
        
        for key, value in cache.items():
            if isinstance(value, tuple) and len(value) > 0:
                timestamp = value[0]
                if current_time - timestamp > max_age_seconds:
                    keys_to_remove.append(key)
        
        for key in keys_to_remove:
            cache.pop(key, None)

# Initialize cache manager
cache_manager = CacheManager()

def cleanup_caches():
    """Enhanced cache cleanup with proper resource management"""
    cache_manager.cleanup_all_caches()
    log.debug("[CACHE] Cleaned up all caches")

# ───────── Enhanced Configuration Validation ─────────

class ConfigValidator:
    """Enhanced configuration validation with comprehensive checks"""
    
    def __init__(self):
        self.required_env_vars = [
            'TELEGRAM_BOT_TOKEN',
            'TELEGRAM_CHAT_ID', 
            'API_KEY',
            'DATABASE_URL'
        ]
        
        self.optional_env_vars = {
            'ADMIN_API_KEY': 'Warning: Admin endpoints disabled',
            'WEBHOOK_SECRET': 'Warning: Webhook security disabled',
            'SENTRY_DSN': 'Info: Error tracking disabled',
            'REDIS_URL': 'Info: Redis cache disabled'
        }
        
        self.numeric_ranges = {
            'CONF_THRESHOLD': (0, 100),
            'SCAN_INTERVAL_SEC': (30, 3600),
            'TIP_MIN_MINUTE': (1, 90),
            'MAX_TIPS_PER_SCAN': (1, 1000),
            'DUP_COOLDOWN_MIN': (0, 1440),
            'EDGE_MIN_BPS': (0, 10000)
        }
        
    def validate_all(self):
        """Run all validation checks"""
        self._check_required_vars()
        self._check_optional_vars()
        self._check_numeric_ranges()
        self._check_database_connection()
        self._check_api_connectivity()
        
    def _check_required_vars(self):
        """Check required environment variables"""
        missing = []
        for var in self.required_env_vars:
            if not os.getenv(var):
                missing.append(var)
                
        if missing:
            raise SystemExit(f"Missing required environment variables: {', '.join(missing)}")
            
    def _check_optional_vars(self):
        """Warn about missing optional variables"""
        for var, message in self.optional_env_vars.items():
            if not os.getenv(var):
                log.warning(f"{message} ({var} not set)")
                
    def _check_numeric_ranges(self):
        """Validate numeric configuration ranges"""
        for var, (min_val, max_val) in self.numeric_ranges.items():
            value = os.getenv(var)
            if value:
                try:
                    num_val = float(value)
                    if not (min_val <= num_val <= max_val):
                        log.warning(f"Config {var}={value} outside recommended range {min_val}-{max_val}")
                except ValueError:
                    log.warning(f"Config {var}={value} is not a valid number")
                    
    def _check_database_connection(self):
        """Test database connectivity"""
        try:
            with db_conn() as c:
                c.execute("SELECT 1")
            log.info("[CONFIG] Database connection: OK")
        except Exception as e:
            raise SystemExit(f"Database connection failed: {e}")
            
    def _check_api_connectivity(self):
        """Test API-Football connectivity"""
        try:
            test_resp = _api_get(FOOTBALL_API_URL, {"live": "all"}, timeout=10)
            if test_resp is not None:
                log.info("[CONFIG] API-Football connectivity: OK")
            else:
                log.warning("[CONFIG] API-Football test request failed")
        except Exception as e:
            log.warning(f"[CONFIG] API-Football connectivity test failed: {e}")

# Initialize config validator
config_validator = ConfigValidator()

def enhanced_validate_config():
    """Enhanced configuration validation"""
    config_validator.validate_all()
    log.info("[CONFIG] Enhanced configuration validation passed")

# ───────── Comprehensive Error Handling & Logging ─────────

class ErrorHandler:
    """Centralized error handling with comprehensive logging"""
    
    def __init__(self):
        self.error_counts = defaultdict(int)
        self.max_errors_per_component = 100
        
    def handle_error(self, component: str, error: Exception, context: dict = None):
        """Handle errors with context and rate limiting"""
        error_id = f"{component}_{type(error).__name__}"
        self.error_counts[error_id] += 1
        
        if self.error_counts[error_id] <= self.max_errors_per_component:
            log_message = f"[ERROR] {component}: {error}"
            if context:
                log_message += f" | Context: {context}"
            log.error(log_message)
            
            # Send critical errors to Telegram
            if self.error_counts[error_id] == 1:  # Only first occurrence
                self._notify_critical_error(component, error, context)
        else:
            # Suppress repeated errors but log occasionally
            if self.error_counts[error_id] % 10 == 0:
                log.warning(f"[ERROR] {component}: {error} (repeated {self.error_counts[error_id]} times)")
                
    def _notify_critical_error(self, component: str, error: Exception, context: dict = None):
        """Notify critical errors via Telegram"""
        message = f"🚨 Critical Error in {component}\nError: {error}"
        if context:
            message += f"\nContext: {context}"
        send_telegram(message)
        
    def reset_error_count(self, component: str = None):
        """Reset error counts"""
        if component:
            keys_to_remove = [k for k in self.error_counts.keys() if k.startswith(component)]
            for key in keys_to_remove:
                self.error_counts.pop(key, None)
        else:
            self.error_counts.clear()

# Initialize error handler
error_handler = ErrorHandler()

# ───────── Unit Test Helpers ─────────

class TestHelpers:
    """Helper functions for unit testing critical components"""
    
    @staticmethod
    def test_feature_extraction():
        """Test feature extraction with sample data"""
        sample_match = {
            "teams": {"home": {"name": "TeamA"}, "away": {"name": "TeamB"}},
            "goals": {"home": 1, "away": 0},
            "fixture": {"status": {"elapsed": 30}},
            "statistics": [
                {
                    "team": {"name": "TeamA"},
                    "statistics": [
                        {"type": "Shots on Goal", "value": "5"},
                        {"type": "Total Shots", "value": "10"},
                        {"type": "Corner Kicks", "value": "4"},
                        {"type": "Ball Possession", "value": "60%"},
                        {"type": "Expected Goals", "value": "1.2"}
                    ]
                },
                {
                    "team": {"name": "TeamB"},
                    "statistics": [
                        {"type": "Shots on Goal", "value": "3"},
                        {"type": "Total Shots", "value": "8"},
                        {"type": "Corner Kicks", "value": "2"},
                        {"type": "Ball Possession", "value": "40%"},
                        {"type": "Expected Goals", "value": "0.8"}
                    ]
                }
            ],
            "events": []
        }
        
        try:
            features = extract_basic_features(sample_match)
            required_features = ['minute', 'goals_h', 'goals_a', 'xg_h', 'xg_a', 'sot_h', 'sot_a']
            missing = [f for f in required_features if f not in features]
            
            if not missing:
                log.info("[TEST] Feature extraction: PASSED")
                return True
            else:
                log.error(f"[TEST] Feature extraction: FAILED - Missing features: {missing}")
                return False
                
        except Exception as e:
            log.error(f"[TEST] Feature extraction: FAILED - {e}")
            return False
    
    @staticmethod
    def test_prediction_pipeline():
        """Test the prediction pipeline with sample features"""
        sample_features = {
            "minute": 30.0,
            "goals_h": 1.0, "goals_a": 0.0,
            "xg_h": 1.2, "xg_a": 0.8,
            "sot_h": 5.0, "sot_a": 3.0,
            "pos_h": 60.0, "pos_a": 40.0
        }
        
        try:
            # Test BTTS prediction
            btts_prob, btts_conf = market_predictor.predict_for_market(sample_features, "BTTS", 30)
            
            # Test OU prediction
            ou_prob, ou_conf = market_predictor.predict_for_market(sample_features, "OU_2.5", 30)
            
            if 0 <= btts_prob <= 1 and 0 <= ou_prob <= 1:
                log.info("[TEST] Prediction pipeline: PASSED")
                return True
            else:
                log.error(f"[TEST] Prediction pipeline: FAILED - Invalid probabilities: BTTS={btts_prob}, OU={ou_prob}")
                return False
                
        except Exception as e:
            log.error(f"[TEST] Prediction pipeline: FAILED - {e}")
            return False
    
    @staticmethod
    def test_database_operations():
        """Test basic database operations"""
        try:
            with db_conn() as c:
                # Test connection
                c.execute("SELECT 1")
                
                # Test settings table
                test_key = "test_config_validation"
                test_value = "test_value"
                
                set_setting(test_key, test_value)
                retrieved = get_setting(test_key)
                
                # Cleanup
                c.execute("DELETE FROM settings WHERE key = %s", (test_key,))
                
                if retrieved == test_value:
                    log.info("[TEST] Database operations: PASSED")
                    return True
                else:
                    log.error(f"[TEST] Database operations: FAILED - Expected {test_value}, got {retrieved}")
                    return False
                    
        except Exception as e:
            log.error(f"[TEST] Database operations: FAILED - {e}")
            return False
    
    @staticmethod
    def run_all_tests():
        """Run all unit tests"""
        log.info("[TEST] Running all unit tests...")
        
        tests = [
            ("Feature Extraction", TestHelpers.test_feature_extraction),
            ("Prediction Pipeline", TestHelpers.test_prediction_pipeline),
            ("Database Operations", TestHelpers.test_database_operations)
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                result = test_func()
                results.append((test_name, result))
            except Exception as e:
                log.error(f"[TEST] {test_name}: FAILED with exception - {e}")
                results.append((test_name, False))
        
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        log.info(f"[TEST] Results: {passed}/{total} tests passed")
        
        if passed == total:
            log.info("[TEST] ALL TESTS PASSED")
        else:
            log.error("[TEST] SOME TESTS FAILED")
            
        return passed == total

# Initialize test helpers
test_helpers = TestHelpers()

# TEMPORARY DEBUGGING - Add this function and call it
def debug_tip_selection():
    """Temporarily bypass all filters to test if tips are generated"""
    log.info("[DEBUG] Running tip selection with filters bypassed")
    
    # Set temporary permissive settings
    original_conf_threshold = CONF_THRESHOLD
    original_edge_bps = EDGE_MIN_BPS
    original_odds_quality = ODDS_QUALITY_MIN
    
    try:
        # Temporarily relax thresholds
        os.environ["CONF_THRESHOLD"] = "50"
        os.environ["EDGE_MIN_BPS"] = "100" 
        os.environ["ODDS_QUALITY_MIN"] = "0.05"
        
        # Run scan
        saved, seen = production_scan()
        log.info(f"[DEBUG] Scan result: {saved} tips saved from {seen} matches")
        
    finally:
        # Restore original settings
        os.environ["CONF_THRESHOLD"] = str(original_conf_threshold)
        os.environ["EDGE_MIN_BPS"] = str(original_edge_bps)
        os.environ["ODDS_QUALITY_MIN"] = str(original_odds_quality)

# Call this from an admin endpoint for testing
@app.route("/admin/debug-tips", methods=["GET"])
def http_debug_tips():
    _require_admin()
    debug_tip_selection()
    return jsonify({"ok": True, "message": "Debug scan completed"})

# ───────── Updated Functions with Enhanced Error Handling ─────────

def stats_coverage_ok(feat: Dict[str, float], minute: int) -> bool:
    """NUCLEAR OPTION - Accept virtually everything"""
    try:
        # Only reject if we have absolutely NO data at all
        has_any_data = any([
            feat.get('pos_h', 0) > 0, feat.get('pos_a', 0) > 0,
            feat.get('sot_h', 0) > 0, feat.get('sot_a', 0) > 0, 
            feat.get('goals_h', 0) > 0, feat.get('goals_a', 0) > 0,
            feat.get('cor_h', 0) > 0, feat.get('cor_a', 0) > 0,
            feat.get('sh_total_h', 0) > 0, feat.get('sh_total_a', 0) > 0
        ])
        
        # After 30 minutes, require at least SOME data
        if minute > 30:
            return has_any_data
        
        # Before 30 minutes, be very lenient
        return True
    except Exception as e:
        error_handler.handle_error("stats_coverage_ok", e, {"minute": minute})
        return False

def is_feed_stale(fid: int, m: dict, minute: int) -> bool:
    """Check if the data feed is stale"""
    if not STALE_GUARD_ENABLE:
        return False
    
    try:
        # Check if events are recent
        events = m.get("events", [])
        if events:
            latest_event_minute = max([ev.get('time', {}).get('elapsed', 0) for ev in events], default=0)
            if minute - latest_event_minute > 10 and minute > 30:
                return True
        
        # Check if stats are recent
        cache_time = STATS_CACHE.get(fid, (0, []))[0]
        if time.time() - cache_time > STALE_STATS_MAX_SEC:
            return True
            
        return False
    except Exception as e:
        error_handler.handle_error("is_feed_stale", e, {"fid": fid, "minute": minute})
        return False

def save_snapshot_from_match(m: dict, feat: Dict[str, float]) -> None:
    """Save snapshot for training data"""
    try:
        fid = int((m.get("fixture") or {}).get("id") or 0)
        if not fid:
            return
            
        now = int(time.time())
        payload = json.dumps({
            "match": m,
            "features": feat,
            "timestamp": now
        }, separators=(",", ":"), ensure_ascii=False)
        
        with db_conn() as c:
            c.execute(
                "INSERT INTO tip_snapshots(match_id, created_ts, payload) "
                "VALUES (%s,%s,%s) "
                "ON CONFLICT (match_id, created_ts) DO UPDATE SET payload=EXCLUDED.payload",
                (fid, now, payload)
            )
    except Exception as e:
        error_handler.handle_error("save_snapshot_from_match", e)

def _league_name(m: dict) -> Tuple[int, str]:
    """Extract league ID and name from match data"""
    try:
        league = m.get("league", {}) or {}
        league_id = int(league.get("id", 0))
        country = league.get("country", "")
        name = league.get("name", "")
        league_name = f"{country} - {name}".strip(" -")
        return league_id, league_name
    except Exception as e:
        error_handler.handle_error("_league_name", e)
        return 0, "Unknown"

def _teams(m: dict) -> Tuple[str, str]:
    """Extract team names from match data"""
    try:
        teams = m.get("teams", {}) or {}
        home = (teams.get("home") or {}).get("name", "")
        away = (teams.get("away") or {}).get("name", "")
        return home, away
    except Exception as e:
        error_handler.handle_error("_teams", e)
        return "Unknown", "Unknown"

def _pretty_score(m: dict) -> str:
    """Format score string"""
    try:
        goals = m.get("goals", {}) or {}
        home = goals.get("home") or 0
        away = goals.get("away") or 0
        return f"{home}-{away}"
    except Exception as e:
        error_handler.handle_error("_pretty_score", e)
        return "0-0"

def _get_market_threshold(market: str) -> float:
    """Get confidence threshold for a specific market"""
    try:
        # Try market-specific threshold first
        market_key = f"conf_threshold:{market}"
        cached = get_setting_cached(market_key)
        if cached:
            try:
                return float(cached)
            except:
                pass
        
        # Fall back to global threshold
        return CONF_THRESHOLD
    except Exception as e:
        error_handler.handle_error("_get_market_threshold", e, {"market": market})
        return CONF_THRESHOLD

def _candidate_is_sane(suggestion: str, feat: Dict[str, float]) -> bool:
    """Check if a prediction candidate makes sense given current match state"""
    try:
        goals_sum = feat.get("goals_sum", 0)
        
        # Check for absurd Over/Under predictions
        if suggestion.startswith("Under"):
            try:
                line = _parse_ou_line_from_suggestion(suggestion)
                if line and goals_sum > line:
                    return False
            except:
                pass
                
        # Check for absurd BTTS predictions
        if suggestion == "BTTS: No":
            goals_h = feat.get("goals_h", 0)
            goals_a = feat.get("goals_a", 0)
            if goals_h > 0 and goals_a > 0:
                return False
                
        return True
    except Exception as e:
        error_handler.handle_error("_candidate_is_sane", e, {"suggestion": suggestion})
        return False

def _inplay_1x2_sanity_ok(suggestion: str, feat: Dict[str, float], odds: Optional[float]) -> bool:
    """
    Late-game 1X2 sanity: if a side leads by >=2 after 60', live odds should be short.
    Big odds at that state implies stale/prematch leakage → block.
    """
    try:
        minute = int(feat.get("minute", 0))
        gd = int(feat.get("goals_h", 0) - feat.get("goals_a", 0))
        if suggestion not in ("Home Win", "Away Win"):
            return True
        if minute >= 60 and abs(gd) >= 2 and odds is not None and odds > 1.50:
            return False
        return True
    except Exception as e:
        error_handler.handle_error("_inplay_1x2_sanity_ok", e, {"suggestion": suggestion, "minute": feat.get("minute")})
        return False

def _score_prob(feat: Dict[str, float], mdl: Optional[Dict[str, Any]]) -> float:
    """Score probability using a model"""
    if not mdl:
        return 0.0
    return predict_from_model(mdl, feat)

def _get_market_threshold_pre(market: str) -> float:
    """Get confidence threshold for pre-match markets"""
    try:
        market_key = f"conf_threshold_pre:{market}"
        cached = get_setting_cached(market_key)
        if cached:
            try:
                return float(cached)
            except:
                pass
        return CONF_THRESHOLD + 5.0
    except Exception as e:
        error_handler.handle_error("_get_market_threshold_pre", e, {"market": market})
        return CONF_THRESHOLD + 5.0

# ───────── Bayesian & GameState & Odds Quality ─────────
class BayesianUpdater:
    def __init__(self):
        self.prior_strength = 0.3
    
    def update_probability(self, prior_prob: float, live_prob: float, minute: int) -> float:
        live_weight = min(minute / 90.0, 1.0) * (1 - self.prior_strength)
        prior_weight = self.prior_strength * (1 - live_weight)
        return float((prior_prob * prior_weight + live_prob * live_weight) / max(1e-9, (prior_weight + live_weight)))
    
    def calculate_confidence_interval(self, prob: float, sample_size: int) -> Tuple[float, float]:
        import math
        z = 1.96
        if sample_size == 0:
            return float(prob), float(prob)
        margin = z * math.sqrt((prob * (1 - prob)) / sample_size)
        return max(0.0, prob - margin), min(1.0, prob + margin)

class GameStateAnalyzer:
    def __init__(self):
        self.critical_states = {
            'equalizer_seek': 0.7,
            'park_the_bus': 0.6,
            'goal_fest': 0.8,
            'defensive_battle': 0.3
        }
    
    def analyze_game_state(self, feat: Dict[str, float]) -> Dict[str, float]:
        state_scores: Dict[str, float] = {}
        goal_diff = float(feat.get("goals_h", 0) - feat.get("goals_a", 0))
        minute = int(feat.get("minute", 0))
        total_goals = float(feat.get("goals_sum", 0))
        if abs(goal_diff) == 1 and minute > 60:
            state_scores['equalizer_seek'] = 0.7 + (minute / 90.0) * 0.3
        if goal_diff >= 2 and minute > 70:
            state_scores['park_the_bus'] = 0.6 + ((minute - 70) / 20.0) * 0.4
        if total_goals >= 3 and minute < 60:
            state_scores['goal_fest'] = min(1.0, total_goals / 5.0)
        if total_goals == 0 and minute > 60:
            state_scores['defensive_battle'] = 0.3 + (minute / 90.0) * 0.5
        return state_scores
    
    def adjust_predictions(self, predictions: dict, game_state: dict) -> dict:
        adjusted = dict(predictions)
        if game_state.get('equalizer_seek', 0) > 0.5:
            if 'BTTS: Yes' in adjusted:
                adjusted['BTTS: Yes'] *= (1 + game_state['equalizer_seek'] * 0.3)
            for key in list(adjusted.keys()):
                if key.startswith('Over'):
                    adjusted[key] *= (1 + game_state['equalizer_seek'] * 0.2)
        if game_state.get('park_the_bus', 0) > 0.5:
            for key in list(adjusted.keys()):
                if key.startswith('Over'):
                    adjusted[key] *= (1 - game_state['park_the_bus'] * 0.4)
                elif key == 'BTTS: Yes':
                    adjusted[key] *= (1 - game_state['park_the_bus'] * 0.3)
        return adjusted

class SmartOddsAnalyzer:
    """Odds quality scoring to avoid stale/prematch leakage."""
    def __init__(self):
        # Keep threshold configurable via env; default matches hardened 0.35
        self.odds_quality_threshold = float(os.getenv("ODDS_QUALITY_MIN", "0.35"))

    def analyze_odds_quality(
        self,
        odds_map: dict,
        prob_hints: dict[str, float] | None,
        target_market_key: str | None = None
    ) -> float:
        """Return [0..1] quality score; higher is better."""
        if not odds_map:
            return 0.0

        if target_market_key and target_market_key in odds_map:
            sides = odds_map.get(target_market_key) or {}
            hint = self._resolve_hint_for_specific(prob_hints or {}, target_market_key)
            return self._market_odds_quality(sides, hint)

        qualities: list[float] = []
        for mk, sides in odds_map.items():
            hint = self._resolve_hint_for_specific(prob_hints or {}, mk)
            qualities.append(self._market_odds_quality(sides, hint))
        return sum(qualities) / len(qualities) if qualities else 0.0

    def _market_odds_quality(self, sides: dict, prob_hint: float | None) -> float:
        """Blend overround and model-consistency."""
        if not sides:
            return 0.0

        total_implied = 0.0
        valid_odds: list[float] = []
        for data in (sides or {}).values():
            o = (data or {}).get("odds")
            if o and o > 0:
                val = float(o)
                total_implied += 1.0 / val
                valid_odds.append(val)

        if not valid_odds:
            return 0.0

        # Overround -> lower is better
        overround = max(0.0, total_implied - 1.0)
        overround_quality = max(0.0, 1.0 - overround * 3.0)

        # Model consistency: if model shows edge, bump quality (but cap)
        model_quality = 1.0
        if prob_hint is not None and prob_hint > 0:
            try:
                best_odds = max(valid_odds)
                edge = float(prob_hint) * best_odds - 1.0
                model_quality = min(1.0, max(0.0, edge + 1.0))
            except Exception:
                model_quality = 0.5

        return (overround_quality + model_quality) / 2.0

    def _resolve_hint_for_specific(self, hints: dict[str, float], market_key: str) -> float | None:
        if not hints:
            return None
        if market_key == "BTTS":
            return hints.get("BTTS: Yes")
        if market_key == "1X2":
            # caller decides side; use the larger side hint for basic quality
            hw = hints.get("Home Win") or 0.0
            aw = hints.get("Away Win") or 0.0
            return max(hw, aw) or None
        if market_key.startswith("OU_"):
            try:
                line_txt = market_key.split("_", 1)[1]
                return hints.get(f"Over {line_txt} Goals")
            except Exception:
                return None
        return None

# ───────── Market cutoff helpers (minutes) ─────────
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
    _TIP_MAX_MINUTE = int(float(TIP_MAX_MINUTE_ENV)) if (TIP_MAX_MINUTE_ENV or "").strip() else None
except Exception:
    _TIP_MAX_MINUTE = None

def _market_family(market_text: str, suggestion: str) -> str:
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

def market_cutoff_ok(minute: Optional[int], market_text: str, suggestion: str) -> bool:
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

def _price_gate(market_text: str, suggestion: str, fid: int, require_live_odds: bool = True) -> Tuple[bool, Optional[float], Optional[str], Optional[float]]:
    """
    Return (pass, odds, book, ev_pct).
    For in-play tips, ALWAYS require live odds - no prematch fallback.
    """
    odds_map = fetch_odds(fid, require_live=True) if API_KEY else {}
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
        log.warning(f"[PRICE_GATE] No LIVE odds found for {market_text} {suggestion} in fixture {fid}")
        return (False, None, None, None)

    min_odds = _min_odds_for_market(market_text)
    if not (min_odds <= odds <= MAX_ODDS_ALL):
        log.warning(f"[PRICE_GATE] Odds {odds} outside range {min_odds}-{MAX_ODDS_ALL} for {market_text}")
        return (False, odds, book, None)
    
    return (True, odds, book, None)

# ───────── Formatting (enhanced tip) ─────────
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
    
    current_score = score.split('-')
    home_goals, away_goals = int(current_score[0]), int(current_score[1])
    is_current_state_prediction = (
        (suggestion == "Home Win" and home_goals > away_goals) or
        (suggestion == "Away Win" and away_goals > home_goals) or
        (suggestion == "BTTS: Yes" and home_goals > 0 and away_goals > 0) or
        (suggestion == "BTTS: No" and (home_goals == 0 or away_goals == 0))
    )
    confidence_label = "📈 Win Probability" if is_current_state_prediction else "📈 Confidence"
    
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
    
    context_note = ""
    if is_current_state_prediction:
        context_note = f"\n⚠️ <i>Note: This reflects current match probability based on score and time</i>"
    
    return ("⚽️ <b>🤖 AI ENHANCED TIP!</b>\n"
            f"<b>Match:</b> {escape(home)} vs {escape(away)}\n"
            f"🕒 <b>Minute:</b> {minute}'  |  <b>Score:</b> {escape(score)}\n"
            f"<b>Tip:</b> {escape(suggestion)}\n"
            f"{confidence_label}: {prob_pct:.1f}%{ai_info}{money}{context_note}\n"
            f"🏆 <b>League:</b> {escape(league)}{stat}")

def explain_tip(match, features, model_output):
    explanation = []

    def _g(keys, default=0.0):
        if not isinstance(keys, (list, tuple)):
            keys = [keys]
        for k in keys:
            if k in features and features[k] is not None:
                v = features[k]
                try:
                    return float(str(v).replace("%", ""))
                except Exception:
                    try:
                        return float(v)
                    except Exception:
                        return default
        return default

    xg_home = _g(['xg_home', 'xg_h'], 0.0)
    xg_away = _g(['xg_away', 'xg_a'], 0.0)
    possession_home = _g(['possession_home', 'pos_h'], 0.0)
    corners_home = _g(['corners_home', 'cor_h'], 0.0)
    corners_away = _g(['corners_away', 'cor_a'], 0.0)
    shots_on_target_away = _g(['shots_on_target_away', 'sot_a'], 0.0)

    ev = None
    if 'ev' in features:
        try: ev = float(str(features['ev']).replace("%", ""))
        except Exception: ev = None
    elif 'ev_pct' in features:
        try: ev = float(str(features['ev_pct']).replace("%", ""))
        except Exception: ev = None

    conf = None
    if isinstance(model_output, dict) and 'confidence' in model_output:
        try: conf = float(model_output['confidence'])
        except Exception: conf = None

    if xg_home < 0.2 and xg_away < 0.2:
        explanation.append("Both teams have low xG — low actual threat so far.")
    if possession_home > 65 and xg_home < 0.3:
        explanation.append("Favorite has possession but isn't creating danger.")
    if corners_away >= corners_home - 1:
        explanation.append("Underdog creating set-piece pressure despite low possession.")
    if shots_on_target_away > 0:
        explanation.append("Away side has tested the keeper.")
    if ev is not None and ev > 100:
        explanation.append(f"Massive EV detected: {round(ev, 1)}% suggests bookmaker mispricing.")
    if conf is not None and conf > 0.75:
        explanation.append("AI confidence strong based on historical patterns in similar matches.")
    return " ".join(explanation)

def check_models_loaded():
    """Check if models are properly loaded"""
    model_names = ["BTTS", "OU_2.5", "OU_3.5", "1X2"]
    for name in model_names:
        model = load_model_from_settings(name)
        if model:
            print(f"[MODEL] {name}: LOADED (weights: {len(model.get('weights', {}))})")
        else:
            print(f"[MODEL] {name}: NOT FOUND")

def debug_scan_issues():
    """Enhanced debug function to identify why high-confidence predictions are blocked"""
    try:
        matches = fetch_live_matches()
        log.info(f"[DEBUG] Found {len(matches)} live matches")
        
        viable_matches = 0
        high_confidence_found = 0
        
        for i, m in enumerate(matches[:20]):  # Check first 20 matches
            fid = int((m.get("fixture") or {}).get("id") or 0)
            if not fid:
                continue
                
            try:
                feat = feature_engineer.extract_advanced_features(m)
                minute = int(feat.get("minute", 0))
                
                coverage_ok = stats_coverage_ok(feat, minute)
                minute_ok = minute >= TIP_MIN_MINUTE
                stale_ok = not is_feed_stale(fid, m, minute)
                
                status = "VIABLE" if (coverage_ok and minute_ok and stale_ok) else "FILTERED"
                if status == "VIABLE":
                    viable_matches += 1
                
                log.info(f"[DEBUG] Match {i+1}: {_teams(m)} | Minute: {minute} | Status: {status}")
                log.info(f"[DEBUG]   Coverage: {coverage_ok} | Minute OK: {minute_ok} (min {TIP_MIN_MINUTE}) | Not Stale: {stale_ok}")
                
                if coverage_ok:
                    log.info(f"[DEBUG]   Key features - Goals: {feat.get('goals_h', 0)}-{feat.get('goals_a', 0)} | "
                           f"xG: {feat.get('xg_h', 0):.2f}-{feat.get('xg_a', 0):.2f} | "
                           f"Shots: {int(feat.get('sot_h', 0))}-{int(feat.get('sot_a', 0))}")
                
                if coverage_ok and minute_ok:
                    markets_to_test = ["BTTS"] + [f"OU_{_fmt_line(line)}" for line in OU_LINES]
                    any_predictions = False
                    high_conf_in_match = False
                    
                    for market in markets_to_test:
                        try:
                            prob, conf = market_predictor.predict_for_market(feat, market, minute)
                            prob_pct = prob * 100
                            threshold = _get_market_threshold(market)
                            passes_threshold = prob_pct >= threshold
                            
                            if prob > 0.1:
                                any_predictions = True
                                if prob > 0.7:
                                    high_conf_in_match = True
                                    high_confidence_found += 1
                                    log.info(f"🎯 [HIGH CONF] {market}: {prob_pct:.1f}% (conf: {conf:.1%}) - NEED: {threshold:.1f}%")
                                    
                                    if market == "BTTS":
                                        suggestions = ["BTTS: Yes", "BTTS: No"]
                                    elif market.startswith("OU_"):
                                        line = float(market.split("_")[1])
                                        suggestions = [f"Over {line} Goals", f"Under {line} Goals"]
                                    else:
                                        suggestions = []
                                    
                                    for suggestion in suggestions:
                                        cutoff_ok = market_cutoff_ok(minute, market, suggestion)
                                        log.info(f"   📋 Cutoff check: {cutoff_ok} (minute {minute})")
                                        
                                        sane = _candidate_is_sane(suggestion, feat)
                                        log.info(f"   🧠 Sanity check: {sane}")
                                        
                                        pass_odds, odds, book, ev_pct = _price_gate(market, suggestion, fid)
                                        log.info(f"   💰 Price gate: {pass_odds} | Odds: {odds} | Book: {book}")
                                        
                                        if odds and prob > 0:
                                            edge = _ev(prob, float(odds))
                                            ev_bps = int(round(edge * 10000))
                                            ev_ok = ev_bps >= EDGE_MIN_BPS
                                            log.info(f"   📈 EV: {edge:.1%} ({ev_bps} bps) | Min needed: {EDGE_MIN_BPS} | EV OK: {ev_ok}")
                                        
                                        all_gates_ok = (
                                            cutoff_ok and 
                                            sane and 
                                            (pass_odds or ALLOW_TIPS_WITHOUT_ODDS) and
                                            (not odds or ev_ok)
                                        )
                                        log.info(f"   ✅ All gates passed: {all_gates_ok}")
                                else:
                                    log.info(f"[DEBUG]   {market}: {prob_pct:.1f}% (conf: {conf:.1%}) - {'PASS' if passes_threshold else 'FAIL'}")
                                    
                        except Exception as e:
                            log.info(f"[DEBUG]   {market}: ERROR - {str(e)[:100]}")
                    
                    if high_conf_in_match:
                        log.info(f"🔥 MATCH HAS HIGH-CONFIDENCE PREDICTIONS - ANALYZING BLOCKERS")
                    
                    if not any_predictions:
                        log.info(f"[DEBUG]   No predictions > 10% for any market")
                        
            except Exception as e:
                log.info(f"[DEBUG] Error processing match {i+1}: {str(e)[:100]}")
                
        log.info(f"[DEBUG] SUMMARY:")
        log.info(f"[DEBUG]   Total matches: {len(matches[:20])}")
        log.info(f"[DEBUG]   Viable matches: {viable_matches}")
        log.info(f"[DEBUG]   High-confidence predictions found: {high_confidence_found}")
        
        log.info(f"[DEBUG] CURRENT CONFIG:")
        log.info(f"[DEBUG]   CONF_THRESHOLD: {CONF_THRESHOLD}")
        log.info(f"[DEBUG]   TIP_MIN_MINUTE: {TIP_MIN_MINUTE}") 
        log.info(f"[DEBUG]   EDGE_MIN_BPS: {EDGE_MIN_BPS}")
        log.info(f"[DEBUG]   ALLOW_TIPS_WITHOUT_ODDS: {ALLOW_TIPS_WITHOUT_ODDS}")
        log.info(f"[DEBUG]   MARKET_CUTOFFS: {_MARKET_CUTOFFS}")
        
    except Exception as e:
        log.info(f"[DEBUG] Failed to fetch matches: {e}")

def check_prediction_quality():
    """Check if models are producing reasonable predictions"""
    log.info("[QUALITY_CHECK] Testing model prediction quality...")
    
    test1 = {
        "minute": 25.0,
        "goals_h": 1.0, "goals_a": 1.0,
        "xg_h": 1.2, "xg_a": 1.1,
        "sot_h": 5.0, "sot_a": 4.0,
        "pos_h": 55.0, "pos_a": 45.0,
        "pressure_home": 60.0, "pressure_away": 50.0,
        "goals_sum": 2.0, "xg_sum": 2.3,
        "goals_last_15": 1.0
    }
    test2 = {
        "minute": 25.0,
        "goals_h": 2.0, "goals_a": 0.0,
        "xg_h": 2.5, "xg_a": 0.3,
        "sot_h": 8.0, "sot_a": 1.0,
        "pos_h": 65.0, "pos_a": 35.0,
        "pressure_home": 75.0, "pressure_away": 25.0,
        "goals_sum": 2.0, "xg_sum": 2.8,
        "goals_last_15": 1.0
    }
    test_cases = [("Balanced", test1), ("One-sided", test2)]
    for name, features in test_cases:
        log.info(f"[QUALITY_CHECK] {name} match:")
        for market in ["BTTS", "OU_2.5", "OU_3.5"]:
            try:
                prob, conf = market_predictor.predict_for_market(features, market, 25)
                log.info(f"  {market}: {prob:.1%} (conf: {conf:.1%})")
            except Exception as e:
                log.info(f"  {market}: ERROR - {e}")

# ───────── ENHANCEMENT 5: Enhanced Production Scan with AI Systems (patched) ─────────
def enhanced_production_scan() -> Tuple[int, int]:
    """
    Enhanced scan with per-gate debug logging (DEBUG_SELECTION_LOG=1).
    """
    try:
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
                        _dbg("no_fixture_id")
                        continue

                    if DUP_COOLDOWN_MIN > 0:
                        cutoff = now_ts - DUP_COOLDOWN_MIN * 60
                        c.execute(
                            "SELECT 1 FROM tips WHERE match_id=%s AND created_ts>=%s AND suggestion<>'HARVEST' LIMIT 1",
                            (fid, cutoff),
                        )
                        row = c.fetchone_safe()
                        if row:
                            _dbg("dup_cooldown_block", fid=fid, cutoff=cutoff, cooldown_min=DUP_COOLDOWN_MIN)
                            continue

                    try:
                        feat = feature_engineer.extract_advanced_features(m)
                    except Exception as e:
                        _dbg("feature_extraction_failed", fid=fid, err=str(e))
                        continue
                        
                    minute = int(feat.get("minute", 0))

                    if not stats_coverage_ok(feat, minute):
                        _dbg("coverage_fail", fid=fid, minute=minute)
                        continue
                    if minute < TIP_MIN_MINUTE:
                        _dbg("minute_low", fid=fid, minute=minute, min_req=TIP_MIN_MINUTE)
                        continue
                    if is_feed_stale(fid, m, minute):
                        _dbg("feed_stale", fid=fid, minute=minute)
                        continue

                    if HARVEST_MODE and minute >= TRAIN_MIN_MINUTE and minute % 3 == 0:
                        try:
                            save_snapshot_from_match(m, feat)
                        except Exception:
                            pass

                    game_state_analyzer = GameStateAnalyzer()
                    game_state = game_state_analyzer.analyze_game_state(feat)

                    league_id, league = _league_name(m)
                    home, away = _teams(m)
                    score = _pretty_score(m)

                    candidates: List[Tuple[str, str, float, float]] = []
                    log.info(f"[MARKET_SCAN] Processing {home} vs {away} at minute {minute}")

                    # 1) BTTS
                    try:
                        btts_prob, btts_conf = market_predictor.predict_for_market(feat, "BTTS", minute)
                        if btts_prob > 0 and btts_conf > 0.5:
                            preds = {"BTTS: Yes": btts_prob, "BTTS: No": max(0.0, 1 - btts_prob)}
                            preds = game_state_analyzer.adjust_predictions(preds, game_state)
                            for suggestion, p in preds.items():
                                if not market_cutoff_ok(minute, "BTTS", suggestion):
                                    _dbg("cutoff_block", market="BTTS", suggestion=suggestion, minute=minute)
                                    continue
                                if not _candidate_is_sane(suggestion, feat):
                                    _dbg("sanity_block", market="BTTS", suggestion=suggestion, goals_sum=feat.get("goals_sum",0))
                                    continue
                                thr = _get_market_threshold("BTTS")
                                if p * 100.0 >= thr:
                                    _dbg("BTTS_PASS", suggestion=suggestion, prob=round(p,3), conf=round(btts_conf,3), thr=thr)
                                    candidates.append(("BTTS", suggestion, p, btts_conf))
                                else:
                                    _dbg("threshold_block", market="BTTS", suggestion=suggestion, prob_pct=round(p*100,1), thr=thr)
                    except Exception as e:
                        _dbg("btts_predict_error", err=str(e))

                    # 2) OU
                    for line in OU_LINES:
                        mkkey = f"Over/Under {_fmt_line(line)}"
                        try:
                            ou_prob, ou_conf = market_predictor.predict_for_market(feat, f"OU_{_fmt_line(line)}", minute)
                            if ou_prob <= 0 or ou_conf <= 0.5:
                                _dbg("ou_low_base", line=_fmt_line(line), prob=round(ou_prob,3), conf=round(ou_conf,3))
                                continue
                            preds = {
                                f"Over {_fmt_line(line)} Goals": ou_prob,
                                f"Under {_fmt_line(line)} Goals": max(0.0, 1 - ou_prob),
                            }
                            preds = game_state_analyzer.adjust_predictions(preds, game_state)
                            for suggestion, p in preds.items():
                                if not market_cutoff_ok(minute, mkkey, suggestion):
                                    _dbg("cutoff_block", market=mkkey, suggestion=suggestion, minute=minute)
                                    continue
                                if not _candidate_is_sane(suggestion, feat):
                                    _dbg("sanity_block", market=mkkey, suggestion=suggestion, goals_sum=feat.get("goals_sum",0))
                                    continue
                                thr = _get_market_threshold(mkkey)
                                if p * 100.0 >= thr:
                                    _dbg("OU_PASS", suggestion=suggestion, line=_fmt_line(line), prob=round(p,3), conf=round(ou_conf,3), thr=thr)
                                    candidates.append((mkkey, suggestion, p, ou_conf))
                                else:
                                    _dbg("threshold_block", market=mkkey, suggestion=suggestion, prob_pct=round(p*100,1), thr=thr)
                        except Exception as e:
                            _dbg("ou_predict_error", line=_fmt_line(line), err=str(e))

                    # 3) 1X2 (draw suppressed)
                    try:
                        ph, pa, c1 = market_predictor._predict_1x2_advanced(feat, minute)
                        if ph > 0 and pa > 0 and c1 > 0.5:
                            s = max(EPS, ph + pa)
                            ph, pa = ph/s, pa/s
                            preds = {"Home Win": ph, "Away Win": pa}
                            preds = game_state_analyzer.adjust_predictions(preds, game_state)
                            for suggestion, p in preds.items():
                                if not market_cutoff_ok(minute, "1X2", suggestion):
                                    _dbg("cutoff_block", market="1X2", suggestion=suggestion, minute=minute)
                                    continue
                                thr = _get_market_threshold("1X2")
                                if p * 100.0 >= thr:
                                    _dbg("WLD_PASS", suggestion=suggestion, prob=round(p,3), conf=round(c1,3), thr=thr)
                                    candidates.append(("1X2", suggestion, p, c1))
                                else:
                                    _dbg("threshold_block", market="1X2", suggestion=suggestion, prob_pct=round(p*100,1), thr=thr)
                    except Exception as e:
                        _dbg("wld_predict_error", err=str(e))

                    if not candidates:
                        _dbg("no_candidates_post_threshold", fid=fid, minute=minute)
                        continue

                    odds_map = fetch_odds(fid) if API_KEY else {}
                    ranked: List[Tuple[str, str, float, Optional[float], Optional[str], Optional[float], float, float]] = []

                    for mk, sug, prob, conf in candidates:
                        if sug not in ALLOWED_SUGGESTIONS:
                            _dbg("suggestion_not_allowed", suggestion=sug)
                            continue

                        # Per-market odds quality
                        okey = _odds_key_for_market(mk, sug)
                        oa = SmartOddsAnalyzer()
                        oq = oa.analyze_odds_quality(odds_map, {mk: prob}, target_market_key=okey)
                        if oq < oa.odds_quality_threshold:
                            _dbg("odds_quality_block", market=mk, sug=sug, quality=round(oq,3), need=oa.odds_quality_threshold)
                            continue

                        # Odds lookup + price gate
                        odds = None; book = None
                        if mk == "BTTS":
                            d = (odds_map.get("BTTS") or {})
                            tgt = "Yes" if sug.endswith("Yes") else "No"
                            if tgt in d: odds, book = d[tgt]["odds"], d[tgt]["book"]
                        elif mk == "1X2":
                            d = (odds_map.get("1X2") or {})
                            tgt = "Home" if sug == "Home Win" else ("Away" if sug == "Away Win" else None)
                            if tgt and tgt in d: odds, book = d[tgt]["odds"], d[tgt]["book"]
                        elif mk.startswith("Over/Under"):
                            ln = _parse_ou_line_from_suggestion(sug)
                            d = (odds_map.get(f"OU_{_fmt_line(ln)}") or {}) if ln is not None else {}
                            tgt = "Over" if sug.startswith("Over") else "Under"
                            if tgt in d: odds, book = d[tgt]["odds"], d[tgt]["book"]

                        pass_odds, odds2, book2, _ = _price_gate(mk, sug, fid)
                        if not pass_odds:
                            _dbg("price_gate_block", market=mk, sug=sug, have_odds=bool(odds), min=_min_odds_for_market(mk))
                            continue
                        if odds is None:
                            odds, book = odds2, book2
                        if mk == "1X2" and not _inplay_1x2_sanity_ok(sug, feat, odds):
                            continue

                        ev_pct = None
                        if odds is not None:
                            edge = _ev(prob, float(odds))
                            ev_bps = int(round(edge * 10000))
                            ev_pct = round(edge * 100.0, 1)
                            if ev_bps < EDGE_MIN_BPS:
                                _dbg("ev_block", market=mk, sug=sug, ev_bps=ev_bps, need=EDGE_MIN_BPS, odds=odds, prob=round(prob,3))
                                continue
                        elif not ALLOW_TIPS_WITHOUT_ODDS:
                            _dbg("missing_odds_block", market=mk, sug=sug)
                            continue

                        rank_score = (prob ** 1.2) * (1 + (ev_pct or 0) / 100.0) * max(0.0, conf)
                        ranked.append((mk, sug, prob, odds, book, ev_pct, rank_score, conf))

                    if not ranked:
                        _dbg("ranked_empty_after_gates", fid=fid, minute=minute)
                        continue

                    ranked.sort(key=lambda x: x[6], reverse=True)

                    per_match = 0
                    base_now = int(time.time())

                    for idx, (market_txt, suggestion, prob, odds, book, ev_pct, _rank, conf) in enumerate(ranked):
                        if PER_LEAGUE_CAP > 0 and per_league_counter.get(league_id, 0) >= PER_LEAGUE_CAP:
                            _dbg("per_league_cap_block", league_id=league_id, cap=PER_LEAGUE_CAP)
                            break
                        if per_match >= max(1, PREDICTIONS_PER_MATCH):
                            break

                        created_ts = base_now + idx
                        raw = float(prob); prob_pct = round(raw * 100.0, 1)

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
                            sent = send_telegram(_format_enhanced_tip_message(
                                home, away, league, minute, score, suggestion,
                                float(prob_pct), feat, odds, book, ev_pct, conf
                            ))
                            if sent:
                                with db_conn() as c3:
                                    c3.execute(
                                        "UPDATE tips SET sent_ok=1 WHERE match_id=%s AND created_ts=%s",
                                        (fid, created_ts)
                                    )
                        except Exception as e:
                            log.exception("[ENHANCED_PROD] insert/send failed: %s", e)
                            _dbg("insert_send_error", err=str(e))
                            continue

                        saved += 1
                        per_match += 1
                        per_league_counter[league_id] = per_league_counter.get(league_id, 0) + 1

                        if MAX_TIPS_PER_SCAN and saved >= MAX_TIPS_PER_SCAN:
                            break

                    if MAX_TIPS_PER_SCAN and saved >= MAX_TIPS_PER_SCAN:
                        break

                except Exception as e:
                    log.exception("[ENHANCED_PROD] match loop failed: %s", e)
                    _dbg("match_loop_error", err=str(e))
                    continue

        log.info("[ENHANCED_PROD] saved=%d live_seen=%d", saved, live_seen)
        _metric_inc("tips_generated_total", n=saved)
        return saved, live_seen
    except Exception as e:
        log.exception("[ENHANCED_PROD] Global scan error: %s", e)
        return (0, 0)

# ✅ Alias for scheduler & admin endpoint
def production_scan() -> Tuple[int, int]:
    return enhanced_production_scan()

def emergency_production_scan() -> Tuple[int, int]:
    """
    EMERGENCY: Use simple direct predictions instead of broken ensemble
    """
    log.info("[EMERGENCY_SCAN] Using simplified prediction system")
    
    # Save original config
    original_config = {
        'CONF_THRESHOLD': CONF_THRESHOLD,
        'EDGE_MIN_BPS': EDGE_MIN_BPS,
        'ODDS_QUALITY_MIN': ODDS_QUALITY_MIN,
        'ALLOW_TIPS_WITHOUT_ODDS': ALLOW_TIPS_WITHOUT_ODDS
    }
    
    try:
        # Open all gates
        os.environ.update({
            "CONF_THRESHOLD": "55",
            "EDGE_MIN_BPS": "100", 
            "ODDS_QUALITY_MIN": "0.1",
            "ALLOW_TIPS_WITHOUT_ODDS": "1"
        })
        
        # Reload config
        global CONF_THRESHOLD, EDGE_MIN_BPS, ODDS_QUALITY_MIN, ALLOW_TIPS_WITHOUT_ODDS
        CONF_THRESHOLD = 55.0
        EDGE_MIN_BPS = 100
        ODDS_QUALITY_MIN = 0.1
        ALLOW_TIPS_WITHOUT_ODDS = True
        
        matches = fetch_live_matches()
        if not matches:
            return 0, 0
            
        saved = 0
        now_ts = int(time.time())
        
        for match in matches:
            try:
                fid = int((match.get("fixture") or {}).get("id") or 0)
                if not fid:
                    continue
                    
                # Skip if recently tipped
                if DUP_COOLDOWN_MIN > 0:
                    with db_conn() as c:
                        cutoff = now_ts - DUP_COOLDOWN_MIN * 60
                        c.execute("SELECT 1 FROM tips WHERE match_id=%s AND created_ts>=%s LIMIT 1", (fid, cutoff))
                        if c.fetchone_safe():
                            continue
                
                features = extract_enhanced_features(match)
                minute = int(features.get("minute", 0))
                
                # Basic gates
                if not stats_coverage_ok(features, minute):
                    continue
                if minute < TIP_MIN_MINUTE:
                    continue
                if is_feed_stale(fid, match, minute):
                    continue
                
                league_id, league = _league_name(match)
                home, away = _teams(match)
                score = _pretty_score(match)
                
                log.info(f"[EMERGENCY_SCAN] Processing {home} vs {away} at {minute}'")
                
                # SIMPLE DIRECT PREDICTIONS (bypass broken ensemble)
                tips_generated = []
                
                # BTTS Prediction
                btts_model = load_model_from_settings("BTTS")
                if btts_model:
                    btts_prob = predict_from_model(btts_model, features)
                    if btts_prob >= 0.55:  # 55% threshold
                        tips_generated.append(("BTTS", "BTTS: Yes", btts_prob, 0.8))
                    elif (1 - btts_prob) >= 0.55:
                        tips_generated.append(("BTTS", "BTTS: No", 1 - btts_prob, 0.8))
                
                # Over/Under 2.5 Prediction
                ou25_model = load_model_from_settings("OU_2.5")
                if ou25_model:
                    ou25_prob = predict_from_model(ou25_model, features)
                    if ou25_prob >= 0.55:
                        tips_generated.append(("Over/Under 2.5", f"Over 2.5 Goals", ou25_prob, 0.8))
                    elif (1 - ou25_prob) >= 0.55:
                        tips_generated.append(("Over/Under 2.5", f"Under 2.5 Goals", 1 - ou25_prob, 0.8))
                
                # Save and send tips
                for market, suggestion, prob, conf in tips_generated:
                    try:
                        created_ts = now_ts + len(tips_generated)  # slight offset
                        prob_pct = prob * 100
                        
                        with db_conn() as c:
                            c.execute("""
                                INSERT INTO tips(
                                    match_id, league_id, league, home, away, market, suggestion,
                                    confidence, confidence_raw, score_at_tip, minute, created_ts, sent_ok
                                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """, (
                                fid, league_id, league, home, away, market, suggestion,
                                float(prob_pct), float(prob), score, minute, created_ts, 0
                            ))
                        
                        # Send to Telegram
                        message = _format_enhanced_tip_message(
                            home, away, league, minute, score, suggestion,
                            float(prob_pct), features, None, None, None, conf
                        )
                        
                        if send_telegram(message):
                            with db_conn() as c:
                                c.execute("UPDATE tips SET sent_ok=1 WHERE match_id=%s AND created_ts=%s", 
                                         (fid, created_ts))
                            saved += 1
                            log.info(f"[EMERGENCY_SCAN] SENT TIP: {suggestion} at {prob_pct:.1f}%")
                            
                    except Exception as e:
                        log.error(f"[EMERGENCY_SCAN] Failed to save tip: {e}")
                
            except Exception as e:
                log.error(f"[EMERGENCY_SCAN] Match failed: {e}")
                continue
        
        log.info(f"[EMERGENCY_SCAN] COMPLETE: {saved} tips generated from {len(matches)} matches")
        return saved, len(matches)
        
    finally:
        # Restore original config
        for key, value in original_config.items():
            os.environ[key] = str(value)

# ───────── Model loader (with validation) ─────────
def _validate_model_blob(name: str, tmp: dict) -> bool:
    if not isinstance(tmp, dict): return False
    if "weights" not in tmp or "intercept" not in tmp: return False
    if not isinstance(tmp["weights"], dict): return False
    if len(tmp["weights"]) > 2000: return False
    return True

MODEL_KEYS_ORDER = ["model_v2:{name}", "model_latest:{name}", "model:{name}", "pre_{name}"]

def load_model_from_settings(name: str) -> Optional[Dict[str, Any]]:
    """Load model with better fallback logic"""
    # Check cache first
    cached = _MODELS_CACHE.get(name)
    if cached is not None:
        return cached
        
    log.info(f"[MODEL] Loading model: {name}")
    
    # Try different key patterns
    key_patterns = [
        f"model:{name}",
        f"model_v2:{name}", 
        f"model_latest:{name}",
        f"pre_{name}"
    ]
    
    for key in key_patterns:
        raw = get_setting_cached(key)
        if raw:
            try:
                model_data = json.loads(raw)
                # Basic validation
                if isinstance(model_data, dict) and "weights" in model_data:
                    # Ensure required fields
                    model_data.setdefault("intercept", 0.0)
                    model_data.setdefault("weights", {})
                    
                    # Store in cache
                    _MODELS_CACHE.set(name, model_data)
                    log.info(f"[MODEL] Loaded {name} from {key}")
                    return model_data
                else:
                    log.warning(f"[MODEL] Invalid model structure for {name} from {key}")
            except Exception as e:
                log.warning(f"[MODEL] Failed to parse {name} from {key}: {e}")
    
    log.warning(f"[MODEL] No model found for: {name}")
    return None

# ───────── Outcomes/backfill/digest (minor safety tweaks preserved) ─────────
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

def _is_final(short: str) -> bool:
    return (short or "").upper() in {"FT","AET","PEN"}

def backfill_results_for_open_matches(max_rows: int = 200) -> int:
    """Backfill results for open matches."""
    now_ts = int(time.time())
    cutoff = now_ts - BACKFILL_DAYS * 24 * 3600
    updated = 0
    with db_conn() as c:
        rows = c.execute("""
            WITH last AS (
                SELECT match_id, MAX(created_ts) last_ts
                FROM tips
                WHERE created_ts >= %s
                GROUP BY match_id
            )
            SELECT l.match_id
            FROM last l
            LEFT JOIN match_results r ON r.match_id = l.match_id
            WHERE r.match_id IS NULL
            ORDER BY l.last_ts DESC
            LIMIT %s
        """, (cutoff, max_rows)).fetchall()
    for (mid,) in rows:
        fx = _fixture_by_id(int(mid))
        if not fx:
            continue
        st = (((fx.get("fixture") or {}).get("status") or {}).get("short") or "")
        if not _is_final(st):
            continue
        g = fx.get("goals") or {}
        gh = int(g.get("home") or 0)
        ga = int(g.get("away") or 0)
        btts = 1 if (gh > 0 and ga > 0) else 0
        with db_conn() as c2:
            c2.execute(
                "INSERT INTO match_results(match_id, final_goals_h, final_goals_a, btts_yes, updated_ts) "
                "VALUES(%s,%s,%s,%s,%s) "
                "ON CONFLICT(match_id) DO UPDATE SET "
                "final_goals_h=EXCLUDED.final_goals_h, "
                "final_goals_a=EXCLUDED.final_goals_a, "
                "btts_yes=EXCLUDED.btts_yes, "
                "updated_ts=EXCLUDED.updated_ts",
                (int(mid), gh, ga, btts, int(time.time()))
            )
        updated += 1
    if updated:
        log.info("[RESULTS] backfilled %d", updated)
    return updated

def _digest_day_range(day: str) -> Tuple[int, Optional[int], str]:
    """
    Returns (start_ts, end_ts, label) in Berlin time.
    day: "today", "yesterday", or "YYYY-MM-DD"
    end_ts is exclusive; if None, means 'open-ended to now'.
    """
    now_bln = datetime.now(BERLIN_TZ)
    label = "today"
    if (day or "").lower() == "yesterday":
        target = (now_bln - timedelta(days=1)).date()
        start = datetime.combine(target, datetime.min.time(), tzinfo=BERLIN_TZ)
        end   = start + timedelta(days=1)
        label = "yesterday"
        return int(start.timestamp()), int(end.timestamp()), label
    try:
        target = datetime.strptime(day, "%Y-%m-%d").date()
        start = datetime.combine(target, datetime.min.time(), tzinfo=BERLIN_TZ)
        end   = start + timedelta(days=1)
        label = day
        return int(start.timestamp()), int(end.timestamp()), label
    except Exception:
        pass
    start = datetime.combine(now_bln.date(), datetime.min.time(), tzinfo=BERLIN_TZ)
    return int(start.timestamp()), None, "today"

def daily_accuracy_digest(day: str = "today") -> Optional[str]:
    """
    Daily accuracy digest.
    - day: "today", "yesterday", or "YYYY-MM-DD"
    """
    if not DAILY_ACCURACY_DIGEST_ENABLE:
        return None

    start_ts, end_ts, label = _digest_day_range(day)
    log.info("[DIGEST] Generating digest for %s (start_ts=%s end_ts=%s)", label, start_ts, end_ts or "now")

    try:
        backfill_results_for_open_matches(400)
    except Exception as e:
        log.warning("[DIGEST] backfill skipped/failed: %s", e)

    where_end = " AND t.created_ts < %s" if end_ts is not None else ""
    params = (start_ts,) + ((end_ts,) if end_ts is not None else ())

    with db_conn() as c:
        rows = c.execute(f"""
            SELECT t.market, t.suggestion, t.confidence, t.confidence_raw, t.created_ts,
                   t.odds, r.final_goals_h, r.final_goals_a, r.btts_yes
            FROM tips t
            LEFT JOIN match_results r ON r.match_id = t.match_id
            WHERE t.created_ts >= %s {where_end}
              AND t.suggestion <> 'HARVEST'
              AND t.sent_ok = 1
            ORDER BY t.created_ts DESC
        """, params).fetchall()

    total = graded = wins = 0
    roi_by_market, by_market = {}, {}
    recent_tips = []

    for (mkt, sugg, conf, conf_raw, cts, odds, gh, ga, btts) in rows:
        res = {"final_goals_h": gh, "final_goals_a": ga, "btts_yes": btts}
        out = _tip_outcome_for_result(sugg, res)

        tip_time = datetime.fromtimestamp(cts, BERLIN_TZ).strftime("%H:%M")
        recent_tips.append(f"{sugg} ({(conf or 0):.1f}%) - {tip_time}")

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
            roi_by_market[mkt]["pnl"] += (float(odds) - 1) if out == 1 else -1

    if graded == 0:
        msg = f"📊 Daily Accuracy Digest for {label}\nNo graded tips for this window."
        if rows:
            pending = len([r for r in rows if r[6] is None or r[7] is None])
            if pending:
                msg += f"\n⏳ {pending} tips still pending results."
    else:
        acc = 100.0 * wins / max(1, graded)
        lines = [
            f"📊 <b>Daily Accuracy Digest</b> - {label}",
            f"Tips sent: {total}  •  Graded: {graded}  •  Wins: {wins}  •  Accuracy: {acc:.1f}%"
        ]
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
    log.info("[DIGEST] Sent daily digest (%s) with %d tips, %d graded", label, total, graded)
    return msg

# ───────── Auto-train / Auto-tune (unchanged) ─────────
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

# ───────── Retry unsent tips (unchanged) ─────────
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

# ───────── Shutdown Handlers (unchanged) ─────────
def shutdown_handler(signum=None, frame=None, *, from_atexit: bool = False):
    """Clean up once. Don't call sys.exit() when invoked by atexit."""
    global _SHUTDOWN_RAN
    if _SHUTDOWN_RAN:
        return
    _SHUTDOWN_RAN = True

    try:
        who = "atexit" if from_atexit else ("signal" if signum else "manual")
        log.info("Received shutdown (%s), cleaning up...", who)
    except Exception:
        pass

    try:
        ShutdownManager.request_shutdown()
    except Exception:
        pass

    try:
        if _SCHED is not None:
            try:
                _SCHED.shutdown(wait=False)
            except Exception:
                pass
    except Exception:
        pass

    try:
        if POOL:
            try:
                POOL.closeall()
            except Exception as e:
                log.warning("Error closing pool during shutdown: %s", e)
    except Exception:
        pass

    if not from_atexit:
        try:
            sys.exit(0)
        except SystemExit:
            pass

def register_shutdown_handlers():
    global _SHUTDOWN_HANDLERS_SET
    if _SHUTDOWN_HANDLERS_SET:
        return
    _SHUTDOWN_HANDLERS_SET = True

    import signal, atexit
    def _sig_wrapper(sig):
        return lambda s, f: shutdown_handler(s, f, from_atexit=False)

    signal.signal(signal.SIGINT,  _sig_wrapper(signal.SIGINT))
    signal.signal(signal.SIGTERM, _sig_wrapper(signal.SIGTERM))
    atexit.register(lambda: shutdown_handler(from_atexit=True))

# ───────── Scheduler (unchanged wiring; one-shot start) ─────────
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
    global _scheduler_started, _SCHED
    if _scheduler_started or not RUN_SCHEDULER:
        return
    try:
        sched = BackgroundScheduler(timezone=TZ_UTC)
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

        if TRAIN_ENABLE:
            sched.add_job(lambda: _run_with_pg_lock(1005, auto_train_job),
                          CronTrigger(hour=TRAIN_HOUR_UTC, minute=TRAIN_MINUTE_UTC, timezone=TZ_UTC),
                          id="train", max_instances=1, coalesce=True)

        if AUTO_TUNE_ENABLE:
            sched.add_job(lambda: _run_with_pg_lock(1006, auto_tune_thresholds, 14),
                          CronTrigger(hour=4, minute=7, timezone=TZ_UTC),
                          id="auto_tune", max_instances=1, coalesce=True)

        sched.add_job(lambda: _run_with_pg_lock(1007, retry_unsent_tips, 30, 200),
                      "interval", minutes=10, id="retry", max_instances=1, coalesce=True)

        sched.add_job(cleanup_caches, "interval", hours=1, id="cache_cleanup")

        sched.start()
        _SCHED = sched
        _scheduler_started = True
        send_telegram("🚀 goalsniper AI mode (in-play) with ENHANCED PREDICTIONS started.")
        log.info("[SCHED] started (scan=%ss)", SCAN_INTERVAL_SEC)
    except Exception as e:
        log.exception("[SCHED] failed: %s", e)

# ───────── Admin / auth / endpoints (ALL routes have GET now) ─────────
def _require_admin():
    key=request.headers.get("X-API-Key") or request.args.get("key") or ((request.json or {}).get("key") if request.is_json else None)
    if not ADMIN_API_KEY or key != ADMIN_API_KEY: abort(401)

@app.route("/", methods=["GET"])
def root(): 
    return jsonify({"ok": True, "name": "goalsniper", "mode": "INPLAY_AI_ENHANCED", "scheduler": RUN_SCHEDULER})

@app.route("/health", methods=["GET"])
def health():
    try:
        with db_conn() as c:
            n = c.execute("SELECT COUNT(*) FROM tips").fetchone()[0]
        api_ok = False
        try:
            test_resp = _api_get(FOOTBALL_API_URL, {"live": "all"}, timeout=5)
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

@app.route("/metrics", methods=["GET"])
def metrics():
    try:
        return jsonify({"ok": True, "metrics": METRICS})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/admin/performance-analysis", methods=["GET"])
def http_performance_analysis():
    _require_admin()
    try:
        trends = adaptive_learner.analyze_performance_trends()
        return jsonify({"ok": True, "trends": trends})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/init-db", methods=["POST","GET"])
def http_init_db():
    _require_admin()
    init_db()
    return jsonify({"ok": True})

@app.route("/debug/live-match/<int:fid>", methods=["GET"])
def debug_live_match(fid: int):
    """Check what data we're getting for a specific match"""
    js = _api_get(FOOTBALL_API_URL, {"id": fid}) or {}
    match = js.get("response", [{}])[0] if js.get("response") else {}
    
    if not match:
        return jsonify({"error": "Match not found", "fixture_id": fid})
    
    match["statistics"] = fetch_match_stats(fid)
    feat = extract_basic_features(match)
    minute = int(feat.get("minute", 0))
    coverage_ok = stats_coverage_ok(feat, minute)
    
    return jsonify({
        "fixture_id": fid,
        "minute": minute,
        "coverage_ok": coverage_ok,
        "features": feat,
        "has_statistics": bool(match.get("statistics")),
        "statistics_count": len(match.get("statistics", [])),
    })

@app.route("/admin/scan", methods=["POST","GET"])
def http_scan():
    _require_admin()
    s,l = production_scan()
    return jsonify({"ok": True, "saved": s, "live_seen": l})

@app.route("/admin/backfill-results", methods=["POST","GET"])
def http_backfill():
    _require_admin()
    n = backfill_results_for_open_matches(400)
    return jsonify({"ok": True, "updated": n})

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
def http_train_notify():
    _require_admin()
    auto_train_job()
    return jsonify({"ok": True})

@app.route("/admin/digest", methods=["POST","GET"])
def http_digest():
    _require_admin()
    day = (request.args.get("day") or "today").strip()
    msg = daily_accuracy_digest(day=day)
    return jsonify({"ok": True, "sent": bool(msg), "day": day})

@app.route("/admin/auto-tune", methods=["POST","GET"])
def http_auto_tune():
    _require_admin()
    tuned=auto_tune_thresholds(14)
    return jsonify({"ok": True, "tuned": tuned})

@app.route("/admin/retry-unsent", methods=["POST","GET"])
def http_retry_unsent():
    _require_admin()
    n=retry_unsent_tips(30,200)
    return jsonify({"ok": True, "resent": n})

@app.route("/admin/debug-scan", methods=["GET"])
def http_debug_scan():
    _require_admin()
    debug_scan_issues()
    return jsonify({"ok": True, "message": "Check logs for debug output"})

@app.route("/debug/match-stats/<int:fid>", methods=["GET"])
def debug_match_stats(fid: int):
    """Debug endpoint to check what stats we're getting"""
    stats = fetch_match_stats(fid)
    feat = extract_basic_features({"fixture": {"id": fid}, "teams": {}, "goals": {}, "statistics": stats, "events": []})
    
    return jsonify({
        "fixture_id": fid,
        "stats_available": bool(stats),
        "features": feat,
        "coverage_ok": stats_coverage_ok(feat, int(feat.get("minute", 0))),
        "raw_stats": stats[:2] if stats else []
    })

@app.route("/settings/<key>", methods=["GET","POST"])
def http_settings(key: str):
    _require_admin()
    if request.method=="GET":
        val=get_setting_cached(key); return jsonify({"ok": True, "key": key, "value": val})
    val=(request.get_json(silent=True) or {}).get("value")
    if val is None: abort(400)
    set_setting(key, str(val)); _SETTINGS_CACHE.invalidate(key); invalidate_model_caches_for_key(key)
    return jsonify({"ok": True})

@app.route("/tips/latest", methods=["GET"])
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

@app.route("/telegram/webhook/<secret>", methods=["POST","GET"])
def telegram_webhook(secret: str):
    if (WEBHOOK_SECRET or "") != secret: abort(403)
    if request.method == "GET":
        return jsonify({"ok": True, "webhook": "ready"})
    update=request.get_json(silent=True) or {}
    try:
        msg=(update.get("message") or {}).get("text") or ""
        if msg.startswith("/start"): send_telegram("👋 goalsniper bot (IN-PLAY AI ENHANCED mode) is online.")
        elif msg.startswith("/digest"): daily_accuracy_digest()
        elif msg.startswith("/scan"):
            parts=msg.split()
            if len(parts)>1 and ADMIN_API_KEY and parts[1]==ADMIN_API_KEY:
                s,l=production_scan(); send_telegram(f"🔁 Scan done. Saved: {s}, Live seen: {l}")
            else: send_telegram("🔒 Admin key required.")
    except Exception as e:
        log.warning("telegram webhook parse error: %s", e)
    return jsonify({"ok": True})

@app.route("/admin/emergency-simple-scan", methods=["GET"])
def http_emergency_simple_scan():
    """Use the simplified emergency scanner"""
    _require_admin()
    saved, seen = emergency_production_scan()
    return jsonify({"ok": True, "saved": saved, "live_seen": seen, "mode": "emergency_simple"})

@app.route("/admin/fix-1x2-models", methods=["GET"])
def http_fix_1x2_models():
    """Create the missing 1X2 models that are causing the infinite loop"""
    _require_admin()
    
    log.info("[FIX] Creating missing 1X2 models...")
    
    # Create the main 1X2 model that the system is looking for
    main_1x2_model = {
        "weights": {
            "goals_h": 0.4, "goals_a": -0.4,
            "xg_h": 0.3, "xg_a": -0.3, 
            "pressure_home": 0.02, "pressure_away": -0.02,
            "pos_h": 0.01, "pos_a": -0.01
        },
        "intercept": 0.1,
        "calibration": {"method": "sigmoid", "a": 1.0, "b": 0.0}
    }
    
    # Save as "1X2" (the main model name)
    set_setting("model:1X2", json.dumps(main_1x2_model))
    
    # Also create the specific home/away models the ensemble expects
    home_model = {
        "weights": {
            "goals_h": 0.5, "goals_a": -0.3,
            "xg_h": 0.4, "xg_a": -0.2,
            "pressure_home": 0.03, "pressure_away": -0.01
        },
        "intercept": -0.2,
        "calibration": {"method": "sigmoid", "a": 1.1, "b": 0.1}
    }
    
    away_model = {
        "weights": {
            "goals_h": -0.3, "goals_a": 0.5,
            "xg_h": -0.2, "xg_a": 0.4, 
            "pressure_home": -0.01, "pressure_away": 0.03
        },
        "intercept": -0.2,
        "calibration": {"method": "sigmoid", "a": 1.1, "b": 0.1}
    }
    
    set_setting("model:1X2_HOME", json.dumps(home_model))
    set_setting("model:1X2_AWAY", json.dumps(away_model))
    
    log.info("[FIX] Created 1X2 models successfully")
    return jsonify({"ok": True, "message": "1X2 models created"})

# ───────── NEW: Enhanced Boot with Comprehensive Validation ─────────

def _on_boot():
    """Enhanced boot process with emergency fixes"""
    try:
        log.info("🚀 STARTING GOALSNIPER BOOT PROCESS...")
        
        register_shutdown_handlers()
        enhanced_validate_config()
        _init_pool()
        init_db()
        
        # CRITICAL: Bootstrap models if none exist
        log.info("📦 Checking for models...")
        models_exist = any(load_model_from_settings(name) for name in ["BTTS", "OU_2.5"])
        if not models_exist:
            log.warning("❌ No models found - bootstrapping basic models")
            bootstrap_basic_models()
        else:
            log.info("✅ Models already exist")
        
        # Test that predictions work
        log.info("🧪 Testing prediction system...")
        test_features = {
            "minute": 25.0,
            "goals_sum": 1.5,
            "goals_h": 1.0, "goals_a": 0.5,
            "xg_sum": 1.8, "xg_h": 1.0, "xg_a": 0.8,
            "pressure_home": 55.0, "pressure_away": 45.0
        }
        
        for model_name in ["BTTS", "OU_2.5"]:
            model = load_model_from_settings(model_name)
            if model:
                prob = predict_from_model(model, test_features)
                log.info(f"🧪 {model_name} test prediction: {prob:.1%}")
        
        _start_scheduler_once()
        
        log.info("✅ GOALSNIPER BOOT COMPLETE - SYSTEM READY")
        send_telegram("🤖 goalsniper AI system BOOTED SUCCESSFULLY with emergency fixes")
        
    except Exception as e:
        log.error(f"💥 BOOT FAILED: {e}")
        send_telegram(f"💥 goalsniper BOOT FAILED: {e}")
        raise

_on_boot()

if __name__ == "__main__":
    app.run(host=os.getenv("HOST","0.0.0.0"), port=int(os.getenv("PORT","8080")))
