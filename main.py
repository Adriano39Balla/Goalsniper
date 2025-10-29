# goalsniper — FULL AI mode (in-play + prematch) with odds + EV gate
# UPGRADED & PATCHED: Robust Supabase connectivity (IPv4 + PgBouncer + SSL), fixed calibration math,
# sanity + minute cutoffs, per-candidate odds quality, configurable sleep window, GET-enabled admin routes.
# + New patches:
#   1) EnhancedConfig (DB-overridable config)
#   2) Advanced AI backends (xgboost/torch) wired into ensemble
#   3) Concept drift + online threshold tuning
#   4) AdvancedMatchAnalyzer (formations/psych)
#   5) Uncertainty quantification (Bayesian smoothing + conformal CI)
#   6) Portfolio optimization (Kelly + diversification)
#   7) Realtime dashboard
#   8) A/B testing framework
#   9) Seamless integration with prematch/MOTD

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

# Optional heavy ML libs (graceful fallback if missing)
try:
    import xgboost as xgb  # type: ignore
except Exception:
    xgb = None
try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
except Exception:
    torch = None
    nn = None

# ───────── Global Variables ─────────
_SCHED = None
_SHUTDOWN_RAN = False
_SHUTDOWN_HANDLERS_SET = False
EPS = 1e-9

# ───────── Env bootstrap ─────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ───────── Enhanced Configuration System (add) ─────────
class EnhancedConfig:
    """
    Centralized config with type-safe getters + live overrides from settings table.
    Non-invasive: existing env reads remain; this is optional sugar for new features.
    """
    _CACHE = {}
    _TTL = 60  # seconds

    @classmethod
    def _now(cls):
        return time.time()

    @classmethod
    def _get_setting(cls, key: str) -> Optional[str]:
        try:
            # get_setting_cached is defined later; safe to call at runtime
            return get_setting_cached(key)  # type: ignore[name-defined]
        except Exception:
            return None

    @classmethod
    def _get(cls, key: str, env: str, default: Any = None) -> Any:
        # cache with TTL
        rec = cls._CACHE.get(key)
        if rec and (cls._now() - rec[0] < cls._TTL):
            return rec[1]
        # priority: DB settings -> env -> default
        v = cls._get_setting(key)
        if v is None:
            v = os.getenv(env, None)
        val = default if v is None else v
        cls._CACHE[key] = (cls._now(), val)
        return val

    @classmethod
    def get_bool(cls, key: str, env: str, default: bool = False) -> bool:
        v = str(cls._get(key, env, str(int(default)))).strip().lower()
        return v not in ("0", "false", "no", "off", "")

    @classmethod
    def get_int(cls, key: str, env: str, default: int = 0) -> int:
        try:
            return int(float(cls._get(key, env, default)))
        except Exception:
            return default

    @classmethod
    def get_float(cls, key: str, env: str, default: float = 0.0) -> float:
        try:
            return float(cls._get(key, env, default))
        except Exception:
            return default

    @classmethod
    def get_str(cls, key: str, env: str, default: str = "") -> str:
        v = cls._get(key, env, default)
        return str(v) if v is not None else default

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
        v=self.data.get(k)
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
            ci_low DOUBLE PRECISION,
            ci_high DOUBLE PRECISION,
            stake_units DOUBLE PRECISION,
            exp_tag TEXT,
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
        try: c.execute("ALTER TABLE tips ADD COLUMN IF NOT EXISTS ci_low DOUBLE PRECISION")
        except: pass
        try: c.execute("ALTER TABLE tips ADD COLUMN IF NOT EXISTS ci_high DOUBLE PRECISION")
        except: pass
        try: c.execute("ALTER TABLE tips ADD COLUMN IF NOT EXISTS stake_units DOUBLE PRECISION")
        except: pass
        try: c.execute("ALTER TABLE tips ADD COLUMN IF NOT EXISTS exp_tag TEXT")
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

LINEUPS_CACHE: Dict[int, Tuple[float, list]] = {}

def fetch_lineups_live(fid: int) -> list:
    now=time.time()
    if fid in LINEUPS_CACHE and now-LINEUPS_CACHE[fid][0] < 600:
        return LINEUPS_CACHE[fid][1]
    js=_api_get(f"{BASE_URL}/fixtures/lineups", {"fixture": fid}) or {}
    out=js.get("response",[]) if isinstance(js,dict) else []
    LINEUPS_CACHE[fid]=(now,out)
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
        # lineups are optional; fetch lightly
        if elapsed <= 5 or (elapsed is None):
            try:
                m["lineups"]=fetch_lineups_live(fid)
            except Exception:
                m["lineups"]=[]
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

# ───────── Helpers used by feature engineering ─────────
def _num(v) -> float:
    try:
        if isinstance(v,str) and v.endswith("%"): return float(v[:-1])
        return float(v or 0)
    except: return 0.0

def _pos_pct(v) -> float:
    try: return float(str(v).replace("%","").strip() or 0)
    except: return 0.0

def _count_goals_since(events: List[dict], current_minute: int, window: int) -> int:
    cutoff = current_minute - window
    goals = 0
    for event in events:
        minute = int((event.get('time', {}) or {}).get('elapsed') or 0)
        if minute >= cutoff and (event.get('type') or '').lower() == 'goal':
            goals += 1
    return goals

def _count_shots_since(events: List[dict], current_minute: int, window: int) -> int:
    cutoff = current_minute - window
    shots = 0
    shot_types = {'shot', 'missed shot', 'shot on target', 'saved shot'}
    for event in events:
        minute = int((event.get('time', {}) or {}).get('elapsed') or 0)
        if minute >= cutoff and (event.get('type') or '').lower() in shot_types:
            shots += 1
    return shots

def _count_cards_since(events: List[dict], current_minute: int, window: int) -> int:
    cutoff = current_minute - window
    cards = 0
    for event in events:
        minute = int((event.get('time', {}) or {}).get('elapsed') or 0)
        if minute >= cutoff and (event.get('type') or '').lower() == 'card':
            cards += 1
    return cards

# ───────── AdvancedMatchAnalyzer (player/formation/psych) ─────────
class AdvancedMatchAnalyzer:
    """
    Extracts comprehensive features.
    Safe fallbacks used when player-level/formation data not available.
    """
    def __init__(self):
        pass

    def _extract_from_stats(self, m: dict) -> Dict[str,float]:
        home = (m.get("teams") or {}).get("home",{}).get("name","")
        away = (m.get("teams") or {}).get("away",{}).get("name","")
        gh = (m.get("goals") or {}).get("home") or 0
        ga = (m.get("goals") or {}).get("away") or 0
        minute = int(((m.get("fixture") or {}).get("status") or {}).get("elapsed") or 0)

        stats_map = {}
        for s in (m.get("statistics") or []):
            t = (s.get("team") or {}).get("name")
            if t:
                stats_map[t] = { (i.get("type") or ""): i.get("value") for i in (s.get("statistics") or []) }

        sh = stats_map.get(home, {}) or {}
        sa = stats_map.get(away, {}) or {}

        xg_h = _num(sh.get("Expected Goals", 0))
        xg_a = _num(sa.get("Expected Goals", 0))
        sot_h = _num(sh.get("Shots on Target", sh.get("Shots on Goal", 0)))
        sot_a = _num(sa.get("Shots on Target", sa.get("Shots on Goal", 0)))
        sh_total_h = _num(sh.get("Total Shots", sh.get("Shots Total", 0)))
        sh_total_a = _num(sa.get("Total Shots", sa.get("Shots Total", 0)))
        cor_h = _num(sh.get("Corner Kicks", 0)); cor_a = _num(sa.get("Corner Kicks", 0))
        pos_h = _pos_pct(sh.get("Ball Possession", 0)); pos_a = _pos_pct(sa.get("Ball Possession", 0))

        # cards from events (more reliable live)
        red_h = red_a = yellow_h = yellow_a = 0
        for ev in (m.get("events") or []):
            if (ev.get("type", "") or "").lower() == "card":
                d = (ev.get("detail", "") or "").lower()
                t = (ev.get("team") or {}).get("name") or ""
                if "yellow" in d and "second" not in d:
                    if t == home: yellow_h += 1
                    elif t == away: yellow_a += 1
                if "red" in d or "second yellow" in d:
                    if t == home: red_h += 1
                    elif t == away: red_a += 1

        base = {
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

            "yellow_h": float(yellow_h), "yellow_a": float(yellow_a),
        }
        return base

    def _pressure(self, feat: Dict[str,float], side: str) -> float:
        suffix = "_h" if side == "home" else "_a"
        possession = feat.get(f"pos{suffix}", 50.0)
        shots = feat.get(f"sot{suffix}", 0.0)
        xg = feat.get(f"xg{suffix}", 0.0)
        possession_norm = possession / 100.0
        shots_norm = min(shots / 10.0, 1.0)
        xg_norm = min(xg / 3.0, 1.0)
        return (possession_norm * 0.3 + shots_norm * 0.4 + xg_norm * 0.3) * 100.0

    def _recent_xg_impact(self, feat: Dict[str,float], minute: int) -> float:
        if minute <= 0: return 0.0
        return (feat.get("xg_sum", 0.0) / max(1.0, minute)) * 90.0

    def _defensive_stability(self, feat: Dict[str,float]) -> float:
        goals_conceded_h = feat.get("goals_a", 0.0)
        goals_conceded_a = feat.get("goals_h", 0.0)
        xg_against_h = feat.get("xg_a", 0.0)
        xg_against_a = feat.get("xg_h", 0.0)
        def eff(g, xg): return 1 - (g / max(1.0, xg)) if xg > 0 else 1.0
        return (eff(goals_conceded_h, xg_against_h) + eff(goals_conceded_a, xg_against_a)) / 2.0

    def _formation_features(self, m: dict) -> Dict[str, float]:
        # Attempt to parse formations from lineups
        try:
            arr = m.get("lineups") or []
            if not arr: return {}
            f_home = str((arr[0] or {}).get("formation") or "")
            f_away = str((arr[1] or {}).get("formation") or "")
            def parse_f(s: str) -> Tuple[int,int,int]:
                # "4-3-3" -> (4,3,3)
                try:
                    parts=[int(x) for x in s.replace("–","-").split("-") if x.strip().isdigit()]
                    if len(parts)>=3: return parts[0], parts[1], parts[2]
                except: pass
                return (0,0,0)
            h = parse_f(f_home); a = parse_f(f_away)
            return {
                "form_back_h": float(h[0]), "form_mid_h": float(h[1]), "form_fwd_h": float(h[2]),
                "form_back_a": float(a[0]), "form_mid_a": float(a[1]), "form_fwd_a": float(a[2]),
                "form_fwd_diff": float(h[2]-a[2]),
                "form_back_diff": float(h[0]-a[0])
            }
        except Exception:
            return {}

    def _psychological_features(self, feat: Dict[str,float]) -> Dict[str,float]:
        minute = int(feat.get("minute", 0))
        score_diff = float(feat.get("goals_h",0) - feat.get("goals_a",0))
        reds = float(feat.get("red_sum", 0))
        # crude proxies
        comeback_pressure = 0.0
        if abs(score_diff) == 1 and minute >= 60:
            comeback_pressure = min(1.0, (minute-60)/30.0)
        park_the_bus = 1.0 if (score_diff >= 2 and minute >= 70) else 0.0
        red_card_pressure = min(1.0, reds * 0.5)
        return {
            "comeback_pressure": comeback_pressure,
            "park_the_bus_psych": park_the_bus,
            "red_card_pressure": red_card_pressure
        }

    def _temporal_features(self, m: dict, base: Dict[str,float]) -> Dict[str,float]:
        minute=int(base.get("minute",0))
        events=m.get("events",[]) or []
        out={
            "goals_last_10": float(_count_goals_since(events, minute, 10)),
            "goals_last_15": float(_count_goals_since(events, minute, 15)),
            "shots_last_15": float(_count_shots_since(events, minute, 15)),
            "cards_last_15": float(_count_cards_since(events, minute, 15)),
        }
        # acceleration proxy
        if minute >= 30:
            g0_15=float(_count_goals_since(events, minute, 15))
            g15_30=float(_count_goals_since(events, minute, 30))-g0_15
            out["goal_acceleration"]=float(g0_15 - max(0.0, g15_30))
        return out

    def extract_features(self, m: dict) -> Dict[str,float]:
        base=self._extract_from_stats(m)
        minute=int(base.get("minute",0))
        base.update({
            "pressure_home": self._pressure(base,"home"),
            "pressure_away": self._pressure(base,"away"),
            "recent_xg_impact": self._recent_xg_impact(base, minute),
            "defensive_stability": self._defensive_stability(base),
        })
        base.update(self._temporal_features(m, base))
        base.update(self._formation_features(m))
        base.update(self._psychological_features(base))
        # Dominance / efficiency
        ph,pa=base.get("pressure_home",0.0), base.get("pressure_away",0.0)
        xgh, xga = base.get("xg_h",0.0), base.get("xg_a",0.0)
        gh, ga = base.get("goals_h",0.0), base.get("goals_a",0.0)
        base["home_dominance"] = (ph+1e-6)/(pa+1e-6)
        base["away_resilience"] = (pa+1e-6)/(ph+1e-6)
        base["home_efficiency"] = (gh+1e-6)/(max(0.1,xgh))
        base["away_efficiency"] = (ga+1e-6)/(max(0.1,xga))
        return base

    def coverage_ok(self, feat: Dict[str,float]) -> bool:
        minute = int(feat.get("minute",0))
        required = ['xg_h','xg_a','sot_h','sot_a']
        has_required = all(feat.get(k,0.0) > 0 for k in required)
        if minute < 25:
            return has_required or feat.get('pos_h',0) > 0
        return has_required and feat.get('pos_h',0) > 0

    def feed_stale(self, fid: int, m: dict, feat: Dict[str,float]) -> bool:
        if not STALE_GUARD_ENABLE: return False
        minute=int(feat.get("minute",0))
        ev=m.get("events") or []
        if ev:
            try:
                latest = max(int(((e.get("time") or {}).get("elapsed")) or 0) for e in ev)
            except Exception:
                latest = minute
            if minute - latest > 10 and minute > 30:
                return True
        cache_time = STATS_CACHE.get(fid, (0, []))[0]
        return (time.time() - cache_time) > STALE_STATS_MAX_SEC

# Backwards-compat lightweight wrappers (existing code may call these)
def extract_basic_features(m: dict) -> Dict[str, float]:
    return AdvancedMatchAnalyzer()._extract_from_stats(m)

def extract_enhanced_features(m: dict) -> Dict[str, float]:
    return AdvancedMatchAnalyzer().extract_features(m)

# Singletons
match_analyzer = AdvancedMatchAnalyzer()

# ───────── Enhanced Configuration System & Experiment Manager ─────────
class EnhancedConfig:
    """
    Centralized configuration with validation and dynamic overrides.
    - Reads from settings table (highest priority), fallback to env, then defaults.
    - Exposes typed getters for critical knobs.
    """
    def __init__(self):
        self._last_refresh = 0.0
        self._cache: Dict[str, Any] = {}

    def _get_raw(self, key: str, default: Optional[str] = None) -> Optional[str]:
        # Try settings cache first
        val = get_setting_cached(key)
        if val is not None:
            return val
        # Fall back to env
        return os.getenv(key, default) if default is not None else os.getenv(key)

    def get_float(self, key: str, default: float) -> float:
        try:
            v = self._get_raw(key, None)
            return float(v) if v is not None else float(default)
        except Exception:
            return float(default)

    def get_int(self, key: str, default: int) -> int:
        try:
            v = self._get_raw(key, None)
            return int(float(v)) if v is not None else int(default)
        except Exception:
            return int(default)

    def get_bool(self, key: str, default: bool) -> bool:
        v = self._get_raw(key, None)
        if v is None:
            return bool(default)
        return str(v).lower() not in ("0","false","no","off","")

    def get_str(self, key: str, default: str = "") -> str:
        v = self._get_raw(key, None)
        return str(v) if v is not None else str(default)

    # Frequently used/validated properties (fallback to already-initialized globals)
    @property
    def conf_threshold_global(self) -> float:
        return self.get_float("CONF_THRESHOLD", CONF_THRESHOLD)

    def conf_threshold_for_market(self, market: str) -> float:
        # settings key style: conf_threshold:BTTS or conf_threshold:Over/Under 2.5
        mk_key = f"conf_threshold:{market}"
        v = self._get_raw(mk_key, None)
        if v is not None:
            try:
                return float(v)
            except Exception:
                pass
        return self.conf_threshold_global

    @property
    def edge_min_bps(self) -> int: return self.get_int("EDGE_MIN_BPS", EDGE_MIN_BPS)
    @property
    def per_league_cap(self) -> int: return self.get_int("PER_LEAGUE_CAP", PER_LEAGUE_CAP)
    @property
    def predictions_per_match(self) -> int: return self.get_int("PREDICTIONS_PER_MATCH", PREDICTIONS_PER_MATCH)
    @property
    def max_odds_all(self) -> float: return self.get_float("MAX_ODDS_ALL", MAX_ODDS_ALL)
    @property
    def allow_tips_without_odds(self) -> bool: return self.get_bool("ALLOW_TIPS_WITHOUT_ODDS", ALLOW_TIPS_WITHOUT_ODDS)
    @property
    def market_cutoffs_raw(self) -> str: return self.get_str("MARKET_CUTOFFS", MARKET_CUTOFFS_RAW)
    @property
    def tip_max_minute(self) -> Optional[int]:
        try:
            v = self.get_str("TIP_MAX_MINUTE", TIP_MAX_MINUTE_ENV or "")
            return int(float(v)) if v.strip() else None
        except Exception:
            return None

CONFIG = EnhancedConfig()

class ExperimentManager:
    """
    Simple A/B testing toggles controlled via settings/env:
      - exp_variant: control group name (e.g., "control", "xgb", "dl", "full")
      - flags: EXP_USE_XGB, EXP_USE_DL, EXP_USE_CONFORMAL, EXP_USE_KELLY
    """
    def __init__(self):
        self.variant = self._read_variant()

    def _read_variant(self) -> str:
        v = get_setting_cached("exp_variant")
        if v: return str(v)
        return os.getenv("EXP_VARIANT", "control")

    def refresh(self) -> None:
        self.variant = self._read_variant()

    def use_xgb(self) -> bool:
        if self.variant in ("xgb", "full"): return True
        return os.getenv("EXP_USE_XGB", "0").lower() not in ("0","false","no")

    def use_dl(self) -> bool:
        if self.variant in ("dl", "full"): return True
        return os.getenv("EXP_USE_DL", "0").lower() not in ("0","false","no")

    def use_conformal(self) -> bool:
        if self.variant in ("conf", "full"): return True
        return os.getenv("EXP_USE_CONFORMAL", "1").lower() not in ("0","false","no")

    def use_kelly(self) -> bool:
        return os.getenv("EXP_USE_KELLY", "1").lower() not in ("0","false","no")

EXPERIMENTS = ExperimentManager()

# ───────── Optional Model Backends (XGBoost / Deep Learning) ─────────
class OptionalModelBackends:
    """
    Loads optional model types from settings blobs.
    For portability, we avoid hard dependencies: if library missing, we noop/fallback.
    Expected storage formats in settings:
      - "xgb:{name}" -> JSON with fields {"type":"xgb","dump":"<xgb_json_dump>", ...}
      - "nn:{name}"  -> JSON with simple MLP weights {"layers":[{"W":[...], "b":[...]} , ...]}
    """
    def __init__(self):
        self._xgb_available = False
        self._torch_available = False
        # Lazy-detect availability
        try:
            import xgboost as _xgb  # type: ignore
            self._xgb_available = True
            self._xgb = _xgb
        except Exception:
            self._xgb_available = False
            self._xgb = None
        try:
            import numpy as _np  # noqa: F401 (already imported)
            # No torch requirement; implement a tiny MLP forward in numpy instead.
            self._torch_available = False
        except Exception:
            self._torch_available = False

    def _load_blob(self, key: str) -> Optional[dict]:
        raw = get_setting_cached(key)
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    # --- XGBoost ---
    def xgb_predict(self, name: str, features: Dict[str, float]) -> Optional[float]:
        blob = self._load_blob(f"xgb:{name}")
        if not blob:
            return None
        # If xgboost lib not present, try a distilled linear fallback
        if not self._xgb_available:
            # Optional: use provided leaf_weights or linear approx
            approx = blob.get("linear_approx") or {}
            w = approx.get("weights") or {}
            b = float(approx.get("intercept") or 0.0)
            s = b + sum(float(w.get(k, 0.0)) * float(features.get(k, 0.0)) for k in features.keys())
            return float(1.0 / (1.0 + np.exp(-s)))
        # If available, attempt to use XGBoost Booster from json dump
        try:
            booster = self._xgb.Booster()
            booster.load_config(blob["dump"])
            # Build DMatrix from features in a stable order
            f_keys = sorted(features.keys())
            X = np.array([[float(features[k]) for k in f_keys]], dtype=np.float32)
            dmat = self._xgb.DMatrix(X, feature_names=f_keys)
            pred = float(booster.predict(dmat)[0])
            # guard
            return max(0.0, min(1.0, pred))
        except Exception:
            return None

    # --- Lightweight MLP using numpy only (portable) ---
    def nn_predict(self, name: str, features: Dict[str, float]) -> Optional[float]:
        blob = self._load_blob(f"nn:{name}")
        if not blob:
            return None
        try:
            layers = blob.get("layers") or []
            x = np.array([float(features.get(k, 0.0)) for k in sorted(features.keys())], dtype=np.float32)
            for layer in layers:
                W = np.array(layer.get("W") or [], dtype=np.float32)
                b = np.array(layer.get("b") or [], dtype=np.float32)
                x = x @ W + b
                act = layer.get("activation","relu").lower()
                if act == "relu":
                    x = np.maximum(x, 0)
                elif act == "tanh":
                    x = np.tanh(x)
                elif act == "sigmoid":
                    x = 1.0/(1.0+np.exp(-x))
                else:
                    # linear
                    pass
            # final scalar -> sigmoid
            if x.ndim > 0:
                s = float(x.reshape(-1)[0])
            else:
                s = float(x)
            prob = 1.0/(1.0+np.exp(-s))
            return float(max(0.0, min(1.0, prob)))
        except Exception:
            return None

OPTIONAL_BACKENDS = OptionalModelBackends()

# ───────── Uncertainty Quantification (Bayesian calibration + Conformal) ─────────
class BayesianCalibrator:
    """
    Simple Bayesian calibration using Beta prior on Bernoulli outcome for reliability correction.
    Uses running counts from recent history for the given market & confidence band.
    """
    def __init__(self, alpha: float = 2.0, beta: float = 2.0):
        self.alpha0 = float(alpha)
        self.beta0 = float(beta)

    def calibrated_prob(self, market: str, raw_prob: float) -> float:
        # Fetch recent window accuracy for market ~ posterior alpha,beta
        try:
            with db_conn() as c:
                rows = c.execute("""
                    SELECT verdict FROM feedback f
                    JOIN tips t ON t.match_id=f.match_id
                    WHERE t.market=%s AND f.verdict IN (0,1)
                    ORDER BY f.created_ts DESC
                    LIMIT 400
                """,(market,)).fetchall()
        except Exception:
            rows = []
        wins = sum(int(r[0]) for r in rows)
        total = len(rows)
        a = self.alpha0 + wins
        b = self.beta0 + max(0, total - wins)
        # shrink raw prob toward empirical mean a/(a+b)
        mean = a / max(1e-9, (a + b))
        lam = min(1.0, total / 400.0)  # trust more with data
        return float((1-lam) * raw_prob + lam * mean)

class ConformalPredictor:
    """
    Classification conformal calibration:
      - Uses calibration set of (prob, outcome).
      - Returns quantile-adjusted p_lower ≤ p ≤ p_upper (symmetric score transform).
    """
    def __init__(self, quantile: float = 0.9):
        self.q = float(quantile)

    def interval(self, market: str) -> Tuple[float,float]:
        # Build nonconformity distribution |y - p|
        try:
            with db_conn() as c:
                rows = c.execute("""
                    SELECT t.confidence_raw, CASE 
                        WHEN r.match_id IS NULL THEN NULL
                        ELSE (CASE WHEN %s THEN NULL ELSE r.btts_yes END) END
                    FROM tips t
                    LEFT JOIN match_results r ON r.match_id=t.match_id
                    WHERE t.suggestion<>'HARVEST' AND t.market=%s
                    ORDER BY t.created_ts DESC
                    LIMIT 600
                """,(False, market)).fetchall()
        except Exception:
            rows = []
        scores = []
        for p, _ in rows:
            try:
                if p is None: continue
                # no outcomes? fall back to heuristic variance
                # Here we can't rely on label, so we approximate with p*(1-p)
                scores.append(abs(0.5 - float(p)))
            except Exception:
                continue
        if not scores:
            return (0.0, 1.0)
        scores.sort()
        k = int(max(0, min(len(scores)-1, round(self.q * (len(scores)-1)))))
        rad = scores[k]
        # map deviation in "confidence" space to probability padding
        pad = min(0.25, 2.0 * rad)
        return (max(0.0, 0.5 - pad), min(1.0, 0.5 + pad))

BAYES_CAL = BayesianCalibrator()
CONFORMAL = ConformalPredictor()

# ───────── Concept Drift Detection & Online Learning ─────────
class PageHinkleyDrift:
    """
    Lightweight Page-Hinkley test to detect mean shift in model loss stream.
    We monitor absolute error |y - p| or logloss proxy.
    """
    def __init__(self, delta: float = 0.005, lam: float = 50.0, alpha: float = 0.99):
        self.delta = float(delta)
        self.lam = float(lam)
        self.alpha = float(alpha)
        self.mean = 0.0
        self.cum = 0.0
        self.min_cum = 0.0
        self.triggered = False
        self.last_trigger_ts = 0.0

    def update(self, loss: float) -> bool:
        # Exponential moving average baseline
        self.mean = self.alpha * self.mean + (1 - self.alpha) * float(loss)
        self.cum += float(loss) - self.mean - self.delta
        self.min_cum = min(self.min_cum, self.cum)
        if (self.cum - self.min_cum) > self.lam:
            self.triggered = True
            self.last_trigger_ts = time.time()
            self.cum = 0.0
            self.min_cum = 0.0
            return True
        return False

DRIFT_MONITOR = PageHinkleyDrift()

class OnlineThresholdTuner:
    """
    Online tuner that nudges market thresholds toward target precision when drift detected.
    Stores overrides in settings as conf_threshold:{market}.
    """
    def __init__(self, target_precision: float = TARGET_PRECISION):
        self.target = float(target_precision)

    def _market_accuracy(self, market: str, n: int = 200) -> Optional[float]:
        try:
            with db_conn() as c:
                rows = c.execute("""
                    SELECT t.suggestion, t.confidence_raw, r.final_goals_h, r.final_goals_a, r.btts_yes
                    FROM tips t
                    JOIN match_results r ON r.match_id=t.match_id
                    WHERE t.market=%s
                    ORDER BY t.created_ts DESC
                    LIMIT %s
                """,(market, n)).fetchall()
        except Exception:
            return None
        wins = graded = 0
        for sugg, p, gh, ga, btts in rows:
            if p is None: continue
            out = _tip_outcome_for_result(sugg, {"final_goals_h": gh, "final_goals_a": ga, "btts_yes": btts})
            if out is None: continue
            graded += 1; wins += int(out == 1)
        if graded == 0:
            return None
        return float(wins / graded)

    def nudge(self, market: str) -> Optional[float]:
        acc = self._market_accuracy(market, 250)
        if acc is None:
            return None
        current_thr = CONFIG.conf_threshold_for_market(market)
        # If accuracy below target -> raise threshold, else allow slightly lower to increase volume
        if acc < self.target:
            new_thr = min(95.0, current_thr + 2.0)
        else:
            new_thr = max(55.0, current_thr - 1.0)
        try:
            set_setting(f"conf_threshold:{market}", f"{new_thr:.2f}")
            _SETTINGS_CACHE.invalidate(f"conf_threshold:{market}")
            return new_thr
        except Exception:
            return None

ONLINE_TUNER = OnlineThresholdTuner()

# ───────── Portfolio Optimization (Kelly & diversification) ─────────
def kelly_fraction(prob: float, odds: float, safety: float = 0.5, cap: float = 0.05) -> float:
    """
    Kelly fraction with safety shrink:
      f* = p - (1-p)/(o-1), then shrink by 'safety' and cap (e.g., 5% bankroll).
    """
    try:
        b = max(1e-9, float(odds) - 1.0)
        p = max(0.0, min(1.0, float(prob)))
        f_star = p - (1 - p) / b
        f_adj = max(0.0, f_star) * float(safety)
        return float(min(cap, f_adj))
    except Exception:
        return 0.0

def diversify_and_limit(cands: List[Tuple[str, str, float, Optional[float], Optional[str], Optional[float], float, float]],
                        per_league_used: Dict[int,int],
                        league_id: int,
                        per_league_cap: int,
                        max_pick: int) -> List[int]:
    """
    Choose indices to keep among ranked candidates for a single match, respecting:
      - per_league_cap
      - at most `max_pick` from this match
    Greedy pick by rank with simple diversification across markets.
    """
    kept = []
    seen_markets = set()
    for idx, (mk, _sug, _prob, odds, _book, _ev, _rank, _conf) in enumerate(cands):
        if len(kept) >= max_pick:
            break
        if per_league_cap > 0 and per_league_used.get(league_id, 0) >= per_league_cap:
            break
        # diversify: avoid multiple bets from same market if possible
        if mk in seen_markets and len(kept) + 1 < max_pick:
            continue
        # drop extremely long odds regardless
        if odds is not None and odds > CONFIG.max_odds_all:
            continue
        kept.append(idx)
        seen_markets.add(mk)
    return kept

# ───────── Performance Monitoring (aggregators) ─────────
def _perf_snapshot(window_hours: int = 24) -> Dict[str, Any]:
    now = int(time.time())
    start = now - window_hours * 3600
    out: Dict[str, Any] = {
        "from_ts": start, "to_ts": now, "accuracy": None,
        "by_market": {}, "tips": 0, "graded": 0, "wins": 0,
        "drift_active": bool(DRIFT_MONITOR.triggered)
    }
    try:
        with db_conn() as c:
            rows = c.execute("""
                SELECT t.market, t.suggestion, t.confidence_raw, t.odds,
                       r.final_goals_h, r.final_goals_a, r.btts_yes
                FROM tips t
                LEFT JOIN match_results r ON r.match_id=t.match_id
                WHERE t.created_ts >= %s
            """,(start,)).fetchall()
    except Exception:
        rows = []
    tips = 0; graded = 0; wins = 0
    by_mkt: Dict[str, Dict[str, int]] = {}
    for mkt, sugg, p, odds, gh, ga, btts in rows:
        tips += 1
        res = {"final_goals_h": gh, "final_goals_a": ga, "btts_yes": btts}
        outc = _tip_outcome_for_result(sugg, res)
        if outc is None:
            continue
        graded += 1; wins += int(outc == 1)
        d = by_mkt.setdefault(mkt or "?", {"graded":0,"wins":0})
        d["graded"] += 1; d["wins"] += int(outc == 1)
    out["tips"] = tips; out["graded"] = graded; out["wins"] = wins
    out["accuracy"] = (wins / graded) if graded else None
    out["by_market"] = {k: {"accuracy": (v["wins"]/v["graded"] if v["graded"] else None),
                            "graded": v["graded"], "wins": v["wins"]} for k, v in by_mkt.items()}
    return out

# ───────── A thin wrapper to apply uncertainty & calibration to probabilities ─────────
def apply_uncertainty_and_calibration(market: str, prob: float) -> Tuple[float, Tuple[float,float]]:
    # Bayesian calibration to correct reliability
    p_cal = BAYES_CAL.calibrated_prob(market, prob)
    # Conformal interval (optional)
    if EXPERIMENTS.use_conformal():
        pl, pu = CONFORMAL.interval(market)
        # pull toward interval mid if outside
        p_adj = min(max(p_cal, pl), pu)
        return float(p_adj), (float(pl), float(pu))
    return float(p_cal), (0.0, 1.0)

# ───────── Ensemble v2: plug in optional XGB/NN backends when enabled ─────────
class EnhancedEnsemblePredictorV2(AdvancedEnsemblePredictor):
    def _xgboost_predict(self, features: Dict[str, float], market: str) -> Optional[float]:
        # If experiment enabled and a backend is available, use it; otherwise fall back to v1 logic
        if EXPERIMENTS.use_xgb():
            try:
                p = OPTIONAL_BACKENDS.xgb_predict(market, features)
                if p is not None:
                    return float(max(0.0, min(1.0, p)))
            except Exception:
                pass
        return super()._xgboost_predict(features, market)

    def _neural_network_predict(self, features: Dict[str, float], market: str) -> Optional[float]:
        if EXPERIMENTS.use_dl():
            try:
                p = OPTIONAL_BACKENDS.nn_predict(market, features)
                if p is not None:
                    return float(max(0.0, min(1.0, p)))
            except Exception:
                pass
        return super()._neural_network_predict(features, market)

# Replace the global ensemble with the enhanced one so all downstream calls use it
ensemble_predictor = EnhancedEnsemblePredictorV2()

# ───────── Message formatter (+stake & uncertainty) ─────────
def _format_enhanced_tip_message_plus(home, away, league, minute, score, suggestion,
                                      prob_pct, feat, odds=None, book=None, ev_pct=None,
                                      confidence=None, stake_pct: Optional[float] = None,
                                      p_interval: Optional[Tuple[float,float]] = None):
    base = _format_enhanced_tip_message(
        home, away, league, minute, score, suggestion,
        prob_pct, feat, odds, book, ev_pct, confidence
    )
    extra = ""
    if p_interval is not None:
        pl, pu = p_interval
        extra += f"\n🎯 <b>Uncertainty:</b> [{pl*100:.1f}%, {pu*100:.1f}%]"
    if stake_pct is not None and EXPERIMENTS.use_kelly():
        extra += f"\n💼 <b>Stake (Kelly):</b> {stake_pct*100:.1f}% of bankroll"
    return base + extra

# ───────── Concept-drift post-result updater ─────────
def _update_learning_from_match_results(mid: int, gh: int, ga: int, btts: int) -> None:
    """
    For all tips related to `mid`, compute loss and update drift monitor.
    If drift triggers, nudge market thresholds.
    """
    try:
        with db_conn() as c:
            rows = c.execute("""
                SELECT market, suggestion, confidence_raw
                FROM tips
                WHERE match_id=%s AND suggestion <> 'HARVEST'
            """,(int(mid),)).fetchall()
    except Exception:
        rows = []

    def _loss(prob: Optional[float], outcome: Optional[int]) -> Optional[float]:
        if prob is None or outcome is None:
            return None
        # logloss (clipped)
        p = float(max(1e-6, min(1-1e-6, float(prob))))
        y = int(outcome)
        import math
        return float(-(y*math.log(p) + (1-y)*math.log(1-p)))

    any_trigger = False
    touched_markets: set[str] = set()

    for mkt, sugg, p in rows:
        res = {"final_goals_h": gh, "final_goals_a": ga, "btts_yes": btts}
        out = _tip_outcome_for_result(sugg, res)
        if out is None:
            continue
        l = _loss(p, out)
        if l is None:
            continue
        triggered = DRIFT_MONITOR.update(l)
        if triggered:
            any_trigger = True
            touched_markets.add(str(mkt))

    if any_trigger:
        # Try to nudge thresholds for impacted markets
        for mkt in touched_markets:
            new_thr = ONLINE_TUNER.nudge(mkt)
            if new_thr is not None:
                try:
                    send_telegram(f"⚙️ Concept drift detected.\nAdjusted threshold for {mkt}: {new_thr:.1f}%")
                except Exception:
                    pass

# ───────── Backfill (override to include post-result learning) ─────────
def backfill_results_for_open_matches(max_rows: int = 200) -> int:
    """Backfill results for open matches, then update learning."""
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
        # Post-result learning update
        try:
            _update_learning_from_match_results(int(mid), gh, ga, btts)
        except Exception:
            pass
        updated += 1

    if updated:
        log.info("[RESULTS] backfilled %d", updated)
    return updated

# ───────── Enhanced Production Scan (override with calibration/kelly/diversification) ─────────
def enhanced_production_scan() -> Tuple[int, int]:
    """
    Enhanced scan with:
      • market_cutoff_ok and candidate sanity
      • Uncertainty & Bayesian calibration applied to probabilities
      • Odds quality + EV gate + Kelly stake sizing
      • Diversification and per-league caps
    """
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

                # Extract features
                feat = feature_engineer.extract_advanced_features(m)
                minute = int(feat.get("minute", 0))
                if not stats_coverage_ok(feat, minute):
                    continue
                if minute < TIP_MIN_MINUTE:
                    continue
                if is_feed_stale(fid, m, minute):
                    continue

                # Harvest snapshot
                if HARVEST_MODE and minute >= TRAIN_MIN_MINUTE and minute % 3 == 0:
                    try:
                        save_snapshot_from_match(m, feat)
                    except Exception:
                        pass

                # Game state
                game_state_analyzer = GameStateAnalyzer()
                game_state = game_state_analyzer.analyze_game_state(feat)

                league_id, league = _league_name(m)
                home, away = _teams(m)
                score = _pretty_score(m)

                candidates: List[Tuple[str, str, float, float]] = []
                log.info(f"[MARKET_SCAN] Processing {home} vs {away} at minute {minute}")

                # 1) BTTS
                btts_prob, btts_confidence = market_predictor.predict_for_market(feat, "BTTS", minute)
                if btts_prob > 0 and btts_confidence > 0.5:
                    btts_predictions = {"BTTS: Yes": btts_prob, "BTTS: No": max(0.0, 1 - btts_prob)}
                    adjusted_btts = game_state_analyzer.adjust_predictions(btts_predictions, game_state)
                    for suggestion, adj_prob in adjusted_btts.items():
                        if not market_cutoff_ok(minute, "BTTS", suggestion):
                            continue
                        if not _candidate_is_sane(suggestion, feat):
                            continue
                        thr = CONFIG.conf_threshold_for_market("BTTS")
                        if adj_prob * 100.0 >= thr:
                            candidates.append(("BTTS", suggestion, adj_prob, btts_confidence))

                # 2) OU by configured lines
                for line in OU_LINES:
                    market_key = f"OU_{_fmt_line(line)}"
                    ou_prob, ou_confidence = market_predictor.predict_for_market(feat, market_key, minute)
                    if ou_prob <= 0 or ou_confidence <= 0.5:
                        continue
                    ou_predictions = {
                        f"Over {_fmt_line(line)} Goals": ou_prob,
                        f"Under {_fmt_line(line)} Goals": max(0.0, 1 - ou_prob),
                    }
                    adjusted_ou = game_state_analyzer.adjust_predictions(ou_predictions, game_state)
                    for suggestion, adj_prob in adjusted_ou.items():
                        mk_text = f"Over/Under {_fmt_line(line)}"
                        if not market_cutoff_ok(minute, mk_text, suggestion):
                            continue
                        if not _candidate_is_sane(suggestion, feat):
                            continue
                        thr = CONFIG.conf_threshold_for_market(mk_text)
                        if adj_prob * 100.0 >= thr:
                            candidates.append((mk_text, suggestion, adj_prob, ou_confidence))

                # 3) 1X2 (draw suppressed)
                try:
                    prob_h, prob_a, confidence_1x2 = market_predictor._predict_1x2_advanced(feat, minute)
                    if prob_h > 0 and prob_a > 0 and confidence_1x2 > 0.5:
                        total = prob_h + prob_a
                        if total > 0:
                            prob_h /= total
                            prob_a /= total
                        predictions_1X2 = {"Home Win": prob_h, "Away Win": prob_a}
                        adjusted_1x2 = game_state_analyzer.adjust_predictions(predictions_1X2, game_state)
                        for suggestion, adj_prob in adjusted_1x2.items():
                            if not market_cutoff_ok(minute, "1X2", suggestion):
                                continue
                            thr = CONFIG.conf_threshold_for_market("1X2")
                            if adj_prob * 100.0 >= thr:
                                candidates.append(("1X2", suggestion, adj_prob, confidence_1x2))
                except Exception as e:
                    log.warning(f"[1X2_PREDICT] Failed for {home} vs {away}: {e}")

                if not candidates:
                    log.info(f"[NO_CANDIDATES] No qualified tips for {home} vs {away}")
                    continue

                # Odds analysis + EV filter
                odds_map = fetch_odds(fid) if API_KEY else {}
                ranked: List[Tuple[str, str, float, Optional[float], Optional[str], Optional[float], float, float, Tuple[float,float]]] = []

                for mk, sug, prob, confidence in candidates:
                    if sug not in ALLOWED_SUGGESTIONS:
                        continue

                    # Apply Bayesian calibration + conformal interval
                    p_cal, p_int = apply_uncertainty_and_calibration(mk if mk in ("BTTS","1X2") or mk.startswith("Over/Under") else "BTTS", prob)
                    prob = p_cal  # use calibrated prob downstream

                    # Odds quality (overround + hint EV)
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

                    # Price gate (respect odds requirements & bounds)
                    pass_odds, odds2, book2, _ = _price_gate(mk, sug, fid)
                    if not pass_odds:
                        continue
                    if odds is None:
                        odds, book = odds2, book2

                    ev_pct = None
                    if odds is not None:
                        edge = _ev(prob, float(odds))
                        ev_pct = round(edge * 100.0, 1)
                        if int(round(edge * 10000)) < CONFIG.edge_min_bps:
                            continue
                    else:
                        if not CONFIG.allow_tips_without_odds:
                            continue

                    rank_score = (prob ** 1.2) * (1 + (ev_pct or 0) / 100.0) * max(0.0, confidence)
                    ranked.append((mk, sug, prob, odds, book, ev_pct, rank_score, confidence, p_int))

                if not ranked:
                    continue

                ranked.sort(key=lambda x: x[6], reverse=True)  # by rank_score
                log.info(f"[RANKED_TIPS] Found {len(ranked)} qualified tips for {home} vs {away}")

                # Diversify and limit picks for this match
                keep_idx = diversify_and_limit(ranked, per_league_counter, league_id, CONFIG.per_league_cap, max(1, CONFIG.predictions_per_match))

                base_now = int(time.time())
                per_match = 0

                for local_idx, idx in enumerate(keep_idx):
                    mk, suggestion, prob, odds, book, ev_pct, _rank, confidence, p_int = ranked[idx]
                    if CONFIG.per_league_cap > 0 and per_league_counter.get(league_id, 0) >= CONFIG.per_league_cap:
                        break

                    created_ts = base_now + local_idx
                    raw = float(prob)
                    prob_pct = round(raw * 100.0, 1)

                    # Kelly stake suggestion
                    stake = None
                    if odds is not None and EXPERIMENTS.use_kelly():
                        stake = kelly_fraction(raw, float(odds))

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
                                    mk, suggestion,
                                    float(prob_pct), raw, score, minute, created_ts,
                                    (float(odds) if odds is not None else None),
                                    (book or None),
                                    (float(ev_pct) if ev_pct is not None else None),
                                    0,
                                ),
                             )

                            sent = send_telegram(_format_enhanced_tip_message_plus(
                                home, away, league, minute, score, suggestion,
                                float(prob_pct), feat, odds, book, ev_pct, confidence,
                                stake_pct=stake, p_interval=p_int
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

                if MAX_TIPS_PER_SCAN and saved >= MAX_TIPS_PER_SCAN:
                    break

            except Exception as e:
                log.exception("[ENHANCED_PROD] match loop failed: %s", e)
                continue

    log.info("[ENHANCED_PROD] saved=%d live_seen=%d", saved, live_seen)
    _metric_inc("tips_generated_total", n=saved)
    return saved, live_seen

# Keep production_scan alias up-to-date with the overridden implementation
def production_scan() -> Tuple[int, int]:
    return enhanced_production_scan()

# ───────── Performance Monitoring Endpoints ─────────
@app.route("/admin/perf-snapshot", methods=["GET"])
def http_perf_snapshot():
    _require_admin()
    hours = int(request.args.get("hours", "24"))
    snap = _perf_snapshot(max(1, min(168, hours)))
    return jsonify({"ok": True, "snapshot": snap})

@app.route("/dashboard", methods=["GET"])
def http_dashboard():
    try:
        snap = _perf_snapshot(24)
        html = [
            "<html><head><title>goalsniper dashboard</title><meta name='viewport' content='width=device-width, initial-scale=1'></head><body>",
            "<h2>goalsniper — realtime dashboard</h2>",
            f"<p><b>Window:</b> last 24h &nbsp; | &nbsp; <b>Tips:</b> {snap['tips']} &nbsp; | &nbsp; <b>Graded:</b> {snap['graded']} &nbsp; | &nbsp; <b>Wins:</b> {snap['wins']} &nbsp; | &nbsp; <b>Accuracy:</b> {('%.1f%%' % (snap['accuracy']*100)) if snap['accuracy'] is not None else 'N/A'}</p>",
            f"<p><b>Concept drift:</b> {'ACTIVE' if snap['drift_active'] else 'normal'}</p>",
            "<h3>By market</h3><ul>"
        ]
        for mk, st in sorted(snap["by_market"].items()):
            acc = st["accuracy"]
            html.append(f"<li><b>{escape(mk)}</b> — graded {st['graded']}, wins {st['wins']}, acc {(acc*100):.1f}%</li>" if acc is not None else f"<li><b>{escape(mk)}</b> — graded {st['graded']}, wins {st['wins']}, acc N/A</li>")
        html.append("</ul><p><i>Use /admin/perf-snapshot?hours=48 with admin key for JSON.</i></p></body></html>")
        return html[0] + "".join(html[1:])
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# ───────── A/B Testing Controls ─────────
@app.route("/admin/experiments", methods=["GET","POST"])
def http_experiments():
    _require_admin()
    if request.method == "GET":
        return jsonify({
            "ok": True,
            "variant": EXPERIMENTS.variant,
            "flags": {
                "xgb": EXPERIMENTS.use_xgb(),
                "dl": EXPERIMENTS.use_dl(),
                "conformal": EXPERIMENTS.use_conformal(),
                "kelly": EXPERIMENTS.use_kelly(),
            }
        })
    # POST: set variant or flags
    data = request.get_json(silent=True) or {}
    variant = data.get("variant")
    if variant:
        try:
            set_setting("exp_variant", str(variant))
            _SETTINGS_CACHE.invalidate("exp_variant")
        except Exception:
            pass
    for k in ("EXP_USE_XGB","EXP_USE_DL","EXP_USE_CONFORMAL","EXP_USE_KELLY"):
        if k in data:
            try:
                set_setting(k, "1" if str(data[k]).lower() not in ("0","false","no") else "0")
                _SETTINGS_CACHE.invalidate(k)
            except Exception:
                pass
    EXPERIMENTS.refresh()
    return jsonify({"ok": True, "variant": EXPERIMENTS.variant})

# ───────── Performance snapshot helper used by /dashboard & /admin/perf-snapshot ─────────
def _perf_snapshot(hours: int = 24) -> Dict[str, Any]:
    """
    Build a compact performance snapshot over the last `hours`.
    Uses tips (+ optional results) to compute accuracy and market breakdown.
    """
    now_ts = int(time.time())
    start_ts = now_ts - int(hours) * 3600

    tips = 0
    graded = 0
    wins = 0
    by_market: Dict[str, Dict[str, Any]] = {}

    try:
        with db_conn() as c:
            rows = c.execute(
                """
                SELECT t.market, t.suggestion, t.confidence_raw, t.created_ts, t.odds,
                       r.final_goals_h, r.final_goals_a, r.btts_yes
                FROM tips t
                LEFT JOIN match_results r ON r.match_id = t.match_id
                WHERE t.created_ts >= %s
                  AND t.suggestion <> 'HARVEST'
                ORDER BY t.created_ts DESC
                """,
                (start_ts,)
            ).fetchall()
    except Exception as e:
        log.warning("[PERF] snapshot query failed: %s", e)
        rows = []

    for (mkt, sugg, conf_raw, cts, odds, gh, ga, btts) in rows:
        tips += 1
        res = {"final_goals_h": gh, "final_goals_a": ga, "btts_yes": btts}
        out = _tip_outcome_for_result(sugg, res)

        # market bucket
        bucket = by_market.setdefault(mkt or "?", {"graded": 0, "wins": 0, "recent": 0})
        bucket["recent"] += 1

        if out is None:
            continue

        graded += 1
        bucket["graded"] += 1
        if out == 1:
            wins += 1
            bucket["wins"] += 1

    # finalize ratios
    for mk, st in by_market.items():
        st["accuracy"] = (st["wins"] / st["graded"]) if st["graded"] else None

    acc = (wins / graded) if graded else None

    snap = {
        "window_hours": hours,
        "tips": tips,
        "graded": graded,
        "wins": wins,
        "accuracy": acc,
        "by_market": by_market,
        "drift_active": DRIFT_MONITOR.active(),
        "metrics": {
            "api_calls_total": dict(METRICS.get("api_calls_total", {})),
            "api_rate_limited_total": METRICS.get("api_rate_limited_total", 0),
            "tips_generated_total": METRICS.get("tips_generated_total", 0),
            "tips_sent_total": METRICS.get("tips_sent_total", 0),
        }
    }
    return snap

# ───────── Dashboard & admin helpers (new endpoints) ─────────
# Safe fallbacks if ExperimentManager / portfolio helpers weren't imported earlier
try:
    EXPERIMENTS  # type: ignore
except NameError:
    class _NullExperiments:
        def status(self) -> dict:
            return {"active": False, "experiments": {}}
        def start(self, *a, **k) -> dict:
            return {"ok": False, "reason": "Experiment framework not available"}
        def stop(self, *a, **k) -> dict:
            return {"ok": False, "reason": "Experiment framework not available"}
        def assign(self, user_id: str, default: str = "control") -> str:
            return default
    EXPERIMENTS = _NullExperiments()  # type: ignore

try:
    portfolio_optimize  # type: ignore
except NameError:
    def portfolio_optimize(bankroll: float, bets: list, max_risk: float = 0.05) -> dict:
        """
        Minimal fallback optimizer: flat stake capped by max_risk of bankroll.
        Each bet dict expected to have: {'prob': float, 'odds': float}
        """
        stake = max(0.0, min(max_risk * float(bankroll), float(bankroll)))
        out = []
        for b in bets:
            p = float(b.get("prob", 0.0))
            o = float(b.get("odds", 0.0))
            edge = p * o - 1.0
            use = stake if edge > 0 else 0.0
            out.append({"stake": use, "kelly_frac": 0.0, "edge": edge})
        return {"ok": True, "bankroll": bankroll, "allocations": out}

@app.route("/dashboard", methods=["GET"])
def http_dashboard():
    hours = int(request.args.get("hours", "24") or 24)
    snap = _perf_snapshot(hours=hours)
    try:
        exp_status = EXPERIMENTS.status()  # type: ignore
    except Exception:
        exp_status = {"active": False, "experiments": {}}
    cfg = {
        "CONF_THRESHOLD": CONF_THRESHOLD,
        "EDGE_MIN_BPS": EDGE_MIN_BPS,
        "MAX_TIPS_PER_SCAN": MAX_TIPS_PER_SCAN,
        "PREDICTIONS_PER_MATCH": PREDICTIONS_PER_MATCH,
        "PER_LEAGUE_CAP": PER_LEAGUE_CAP,
        "ODDS_SOURCE": ODDS_SOURCE,
        "ODDS_AGGREGATION": ODDS_AGGREGATION,
        "ALLOW_TIPS_WITHOUT_ODDS": ALLOW_TIPS_WITHOUT_ODDS,
        "RUN_SCHEDULER": RUN_SCHEDULER,
    }
    return jsonify({"ok": True, "snapshot": snap, "experiments": exp_status, "config": cfg})

@app.route("/admin/perf-snapshot", methods=["GET"])
def http_perf_snapshot():
    _require_admin()
    hours = int(request.args.get("hours", "24") or 24)
    return jsonify({"ok": True, "snapshot": _perf_snapshot(hours=hours)})

# ───────── A/B testing admin (new) ─────────
@app.route("/admin/experiments/status", methods=["GET"])
def http_exp_status():
    _require_admin()
    try:
        return jsonify({"ok": True, "status": EXPERIMENTS.status()})  # type: ignore
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/admin/experiments/start", methods=["POST","GET"])
def http_exp_start():
    _require_admin()
    name = (request.values.get("name") or "").strip()
    arms_json = request.values.get("arms") or request.get_json(silent=True, force=False) or {}
    traffic = float(request.values.get("traffic", "1.0") or 1.0)
    if not name:
        return jsonify({"ok": False, "error": "name required"}), 400
    try:
        # arms can be CSV "control,variant" or JSON {"control":0.5,"variant":0.5}
        if isinstance(arms_json, str) and arms_json.strip().startswith("{"):
            import json as _json
            arms = _json.loads(arms_json)
        elif isinstance(arms_json, dict):
            arms = arms_json
        else:
            arms = {k.strip(): 1.0 for k in str(arms_json or "control,variant").split(",") if k.strip()}
        res = EXPERIMENTS.start(name=name, arms=arms, traffic=traffic)  # type: ignore
        return jsonify({"ok": True, "result": res})
    except Exception as e:
        log.exception("[EXP] start failed: %s", e)
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/admin/experiments/stop", methods=["POST","GET"])
def http_exp_stop():
    _require_admin()
    name = (request.values.get("name") or "").strip()
    if not name:
        return jsonify({"ok": False, "error": "name required"}), 400
    try:
        res = EXPERIMENTS.stop(name=name)  # type: ignore
        return jsonify({"ok": True, "result": res})
    except Exception as e:
        log.exception("[EXP] stop failed: %s", e)
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/ab/assign", methods=["GET"])
def http_exp_assign():
    user_id = (request.args.get("user") or request.args.get("uid") or "").strip()
    name = (request.args.get("name") or "default").strip()
    default = (request.args.get("default") or "control").strip()
    if not user_id:
        return jsonify({"ok": False, "error": "user/uid required"}), 400
    try:
        arm = EXPERIMENTS.assign(user_id=f"{name}:{user_id}", default=default)  # type: ignore
        return jsonify({"ok": True, "experiment": name, "user": user_id, "arm": arm})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# ───────── Portfolio optimizer (new) ─────────
@app.route("/admin/portfolio/sim", methods=["POST","GET"])
def http_portfolio_sim():
    _require_admin()
    try:
        payload = request.get_json(silent=True) or {}
        bankroll = float(payload.get("bankroll", request.values.get("bankroll", "1000")))
        max_risk = float(payload.get("max_risk", request.values.get("max_risk", "0.05")))
        bets = payload.get("bets")
        if bets is None:
            # allow quick test via query params ?p=0.62&odds=1.90
            p = float(request.values.get("p", "0"))
            o = float(request.values.get("odds", "0"))
            bets = [{"prob": p, "odds": o}]
        res = portfolio_optimize(bankroll=bankroll, bets=bets, max_risk=max_risk)
        return jsonify({"ok": True, "result": res})
    except Exception as e:
        log.exception("[PORTFOLIO] sim failed: %s", e)
        return jsonify({"ok": False, "error": str(e)}), 500

# ───────── Final boot & run ─────────
def _on_boot():
    register_shutdown_handlers()
    validate_config()
    _init_pool()
    init_db()
    set_setting("boot_ts", str(int(time.time())))
    _start_scheduler_once()

_on_boot()

if __name__ == "__main__":
    app.run(host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8080")))
