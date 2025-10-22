# goalsniper — In-play only (BTTS / OU / 1X2 w/o Draw)
# Focus: stability, performance, speed, precision, calibration & EV discipline
# Notes:
#  - All prematch/MOTD code removed.
#  - Keep: advanced ensemble infra, feature engineering, EV/odds gates, metrics, digest.
#  - Conservative defaults preserved (CONF_THRESHOLD, EDGE_MIN_BPS, odds mins).
#  - Fixed earlier NameErrors/SyntaxErrors and ordering issues.

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
from collections import defaultdict

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
        _redis = None

# ───────── Shutdown Manager ─────────
class ShutdownManager:
    _shutdown_requested = False
    @classmethod
    def is_shutdown_requested(cls): return cls._shutdown_requested
    @classmethod
    def request_shutdown(cls): cls._shutdown_requested = True

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

# ───────── Required envs (fail fast) ─────────
def _require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise SystemExit(f"Missing required environment variable: {name}")
    return v

# ───────── Core env (secrets: required; knobs: defaultable) ─────────
TELEGRAM_BOT_TOKEN = _require_env("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = _require_env("TELEGRAM_CHAT_ID")
API_KEY            = _require_env("API_KEY")
ADMIN_API_KEY      = os.getenv("ADMIN_API_KEY")
RUN_SCHEDULER      = os.getenv("RUN_SCHEDULER", "1") not in ("0","false","False","no","NO")

# Precision-related knobs
CONF_THRESHOLD     = float(os.getenv("CONF_THRESHOLD", "75"))
MAX_TIPS_PER_SCAN  = int(os.getenv("MAX_TIPS_PER_SCAN", "25"))
DUP_COOLDOWN_MIN   = int(os.getenv("DUP_COOLDOWN_MIN", "20"))
TIP_MIN_MINUTE     = int(os.getenv("TIP_MIN_MINUTE", "12"))
SCAN_INTERVAL_SEC  = int(os.getenv("SCAN_INTERVAL_SEC", "300"))

HARVEST_MODE       = os.getenv("HARVEST_MODE", "1") not in ("0","false","False","no","NO")  # send tips to Telegram
TRAIN_ENABLE       = os.getenv("TRAIN_ENABLE", "1") not in ("0","false","False","no","NO")
TRAIN_HOUR_UTC     = int(os.getenv("TRAIN_HOUR_UTC", "2"))
TRAIN_MINUTE_UTC   = int(os.getenv("TRAIN_MINUTE_UTC", "12"))

BACKFILL_EVERY_MIN = int(os.getenv("BACKFILL_EVERY_MIN", "15"))

DAILY_ACCURACY_DIGEST_ENABLE = os.getenv("DAILY_ACCURACY_DIGEST_ENABLE", "1") not in ("0","false","False","no","NO")
DAILY_ACCURACY_HOUR   = int(os.getenv("DAILY_ACCURACY_HOUR", "3"))
DAILY_ACCURACY_MINUTE = int(os.getenv("DAILY_ACCURACY_MINUTE", "6"))

# Auto-tune thresholds (ROI-aware)
AUTO_TUNE_ENABLE        = os.getenv("AUTO_TUNE_ENABLE", "0") not in ("0","false","False","no","NO")
TARGET_PRECISION        = float(os.getenv("TARGET_PRECISION", "0.60"))
THRESH_MIN_PREDICTIONS  = int(os.getenv("THRESH_MIN_PREDICTIONS", "25"))
MIN_THRESH              = float(os.getenv("MIN_THRESH", "55"))
MAX_THRESH              = float(os.getenv("MAX_THRESH", "85"))

STALE_GUARD_ENABLE = os.getenv("STALE_GUARD_ENABLE", "1") not in ("0","false","False","no","NO")
STALE_STATS_MAX_SEC = int(os.getenv("STALE_STATS_MAX_SEC", "240"))
MARKET_CUTOFFS_RAW = os.getenv("MARKET_CUTOFFS", "BTTS=75,1X2=80,OU=88")
TIP_MAX_MINUTE_ENV = os.getenv("TIP_MAX_MINUTE", "")

# Lines and in-play scope
def _parse_lines(env_val: str, default: List[float]) -> List[float]:
    out=[]; 
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
EDGE_MIN_BPS  = int(os.getenv("EDGE_MIN_BPS", "600"))  # 600 bps = +6% EV

ODDS_SOURCE = os.getenv("ODDS_SOURCE", "auto").lower()            # auto|live|prematch
ODDS_AGGREGATION = os.getenv("ODDS_AGGREGATION", "median").lower()# median|best
ODDS_OUTLIER_MULT = float(os.getenv("ODDS_OUTLIER_MULT", "1.8"))
ODDS_REQUIRE_N_BOOKS = int(os.getenv("ODDS_REQUIRE_N_BOOKS", "2"))
ODDS_FAIR_MAX_MULT = float(os.getenv("ODDS_FAIR_MAX_MULT", "2.5"))
ODDS_BOOKMAKER_ID = os.getenv("ODDS_BOOKMAKER_ID")  # optional API-Football book id
ALLOW_TIPS_WITHOUT_ODDS = os.getenv("ALLOW_TIPS_WITHOUT_ODDS","0") not in ("0","false","False","no","NO")

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
session.mount("https://", HTTPAdapter(max_retries=Retry(
    total=3, backoff_factor=1, status_forcelist=[429,500,502,503,504], respect_retry_after_header=True
)))

# ───────── Caches & timezones ─────────
STATS_CACHE:  Dict[int, Tuple[float, list]] = {}
EVENTS_CACHE: Dict[int, Tuple[float, list]] = {}
ODDS_CACHE:   Dict[int, Tuple[float, dict]] = {}
SETTINGS_TTL = int(os.getenv("SETTINGS_TTL_SEC","60"))
MODELS_TTL   = int(os.getenv("MODELS_CACHE_TTL_SEC","120"))
TZ_UTC = ZoneInfo("UTC")
BERLIN_TZ = ZoneInfo("Europe/Berlin")

# ───────── Sleep Method for API-Football Rate Limiting ─────────
def is_sleep_time() -> bool:
    """
    Sleep between 22:00-08:00 Berlin time to protect quota (your knob).
    """
    try:
        now_berlin = datetime.now(BERLIN_TZ)
        return now_berlin.hour >= 22 or now_berlin.hour < 8
    except Exception:
        return False

def sleep_if_required() -> bool:
    if is_sleep_time():
        log.info("[SLEEP] Quiet hours (22–08 Berlin) — skipping API calls")
        return True
    return False

# ───────── Negative-result cache ─────────
NEG_CACHE: Dict[Tuple[str,int], Tuple[float, bool]] = {}
NEG_TTL_SEC = int(os.getenv("NEG_TTL_SEC", "45"))

# ───────── API circuit breaker / timeouts ─────────
API_CB = {"failures": 0, "opened_until": 0.0, "last_success": 0.0}
API_CB_THRESHOLD = int(os.getenv("API_CB_THRESHOLD", "8"))
API_CB_COOLDOWN_SEC = int(os.getenv("API_CB_COOLDOWN_SEC", "90"))
REQ_TIMEOUT_SEC = float(os.getenv("REQ_TIMEOUT_SEC", "8.0"))

# ───────── API helpers (with circuit breaker & metrics) ─────────
def _api_get(url: str, params: dict, timeout: int = 15):
    if not API_KEY: return None
    now = time.time()
    if API_CB["opened_until"] > now:
        log.warning("[CB] Circuit open, rejecting request to %s", url)
        return None

    # Reset breaker after cooldown
    if API_CB["failures"] > 0 and now - API_CB.get("last_success", 0) > API_CB_COOLDOWN_SEC:
        API_CB["failures"] = 0
        API_CB["opened_until"] = 0

    lbl = "unknown"
    try:
        if "/odds/live" in url or "/odds" in url: lbl = "odds"
        elif "/statistics" in url: lbl = "statistics"
        elif "/events" in url: lbl = "events"
        elif "/fixtures" in url: lbl = "fixtures"
    except Exception:
        pass

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

def api_get_with_sleep(url: str, params: dict, timeout: int = 15):
    """
    Wrapper that respects quiet hours. Returns None during sleep hours.
    """
    if sleep_if_required():
        log.debug("[SLEEP] Skipping API call to %s", url)
        return None
    return _api_get(url, params, timeout)

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

# ───────── DB pool & helpers ─────────
POOL: Optional[SimpleConnectionPool] = None

def _init_pool():
    global POOL
    if not POOL:
        POOL = SimpleConnectionPool(minconn=1, maxconn=10, dsn=DATABASE_URL)
        log.info("[DB] Connection pool initialized")

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
            if self.cur: self.cur.close()
        finally:
            if self.conn:
                try: self.pool.putconn(self.conn)
                except Exception: 
                    try: self.conn.close()
                    except: pass
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
        try:
            row = self.cur.fetchone()
            if row is None or len(row) == 0: return None
            return row
        except Exception:
            return None
    def fetchall_safe(self):
        try:
            rows = self.cur.fetchall()
            return rows if rows else []
        except Exception:
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
        return row[0]

def set_setting(key: str, value: str) -> None:
    with db_conn() as c:
        c.execute(
            "INSERT INTO settings(key,value) VALUES(%s,%s) "
            "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
            (key,value)
        )

def get_setting_cached(key: str) -> Optional[str]:
    v=_SETTINGS_CACHE.get(key)
    if v is None: 
        v=get_setting(key); 
        _SETTINGS_CACHE.set(key,v)
    return v

def invalidate_model_caches_for_key(key: str):
    if key.lower().startswith(("model","model_latest","model_v2")):
        _MODELS_CACHE.invalidate(key)

# ───────── Init DB (prematch/MOTD removed) ─────────
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

# ───────── Live fetches (stats/events/fixtures) ─────────

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

# ───────── Helpers ─────────

def _num(v) -> float:
    try:
        if isinstance(v,str) and v.endswith("%"): return float(v[:-1])
        return float(v or 0)
    except: return 0.0

def _pos_pct(v) -> float:
    try: return float(str(v).replace("%","").strip() or 0)
    except: return 0.0

def _safe_num(x) -> float:
    try:
        if isinstance(x, str) and x.endswith("%"):
            return float(x[:-1])
        return float(x or 0.0)
    except Exception:
        return 0.0

# ───────── Feature extraction (basic + enhanced) ─────────

def extract_basic_features(m: dict) -> Dict[str,float]:
    """Provider-robust basic features from API-Football live stats."""
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

def _count_events_since(events: List[dict], current_minute: int, window: int, types: set|str) -> int:
    cutoff = current_minute - window
    count = 0
    for event in events:
        minute = event.get('time', {}).get('elapsed', 0)
        if minute >= cutoff:
            et = event.get('type')
            if isinstance(types, str):
                if et == types: count += 1
            else:
                if et in types: count += 1
    return count

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
    if total_xg <= 0: return 0.0
    return (total_goals - total_xg) / max(1, total_xg)

def _recent_xg_impact(feat: Dict[str, float], minute: int) -> float:
    if minute <= 0: return 0.0
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

def extract_enhanced_features(m: dict) -> Dict[str, float]:
    base_feat = extract_basic_features(m)
    minute = base_feat.get("minute", 0)
    events = m.get("events", [])

    base_feat.update({
        "goals_last_15": float(_count_events_since(events, minute, 15, 'Goal')),
        "shots_last_15": float(_count_events_since(events, minute, 15, {'Shot', 'Missed Shot', 'Shot on Target', 'Saved Shot'})),
        "cards_last_15": float(_count_events_since(events, minute, 15, {'Card'})),

        "pressure_home": _calculate_pressure(base_feat, "home"),
        "pressure_away": _calculate_pressure(base_feat, "away"),

        "score_advantage": base_feat.get("goals_h", 0) - base_feat.get("goals_a", 0),
        "xg_momentum": _calculate_xg_momentum(base_feat),

        "recent_xg_impact": _recent_xg_impact(base_feat, minute),
        "defensive_stability": _defensive_stability(base_feat),
    })
    return base_feat

# ───────── Probability calibration & reliability safeguards ─────────

def _apply_sigmoid_calibration(p: float, cal: Dict[str,Any]) -> float:
    """Sigmoid/Platt-style lightweight reliability adjustment (a,b from DB)."""
    method=(cal or {}).get("method","sigmoid").lower()
    a=float((cal or {}).get("a",1.0)); b=float((cal or {}).get("b",0.0))
    p=float(max(1e-6, min(1-1e-6, p)))
    if method in ("sigmoid","platt"):
        import math
        z=math.log(p/(1-p))
        p=1/(1+math.exp(-(a*z+b)))
    return float(max(1e-6, min(1-1e-6, p)))

def _shrink_extremes(p: float, shrink: float = 0.95) -> float:
    """Conservative shrink to fight overconfidence, e.g. 0.95 toward 0.5."""
    return 0.5 + (p - 0.5) * float(shrink)

def _apply_market_reliability(p: float, market: str) -> float:
    """Optional per-market isotonic/reliability curve from settings: reliability:{market}=[(in,out),...]"""
    key=f"reliability:{market}"
    raw=get_setting_cached(key)
    if not raw: 
        return p
    try:
        pairs=json.loads(raw)
        x=float(p)
        # simple piecewise linear interpolation
        pts=sorted([(float(a),float(b)) for (a,b) in pairs], key=lambda t:t[0])
        if not pts: return p
        if x<=pts[0][0]: return pts[0][1]
        if x>=pts[-1][0]: return pts[-1][1]
        for i in range(1,len(pts)):
            x0,y0=pts[i-1]; x1,y1=pts[i]
            if x0<=x<=x1:
                t=(x-x0)/max(1e-9,(x1-x0))
                return float(y0 + t*(y1-y0))
    except Exception:
        pass
    return p

def _calibrated_prob(raw_p: float, market: str, model_cal: Optional[Dict[str,Any]] = None) -> float:
    """Compose: model calibration -> shrink extremes -> market reliability curve."""
    p=float(raw_p)
    if model_cal:
        p=_apply_sigmoid_calibration(p, model_cal)
    p=_shrink_extremes(p, float(os.getenv("CALIB_SHRINK", "0.94")))
    p=_apply_market_reliability(p, market)
    return float(max(1e-6, min(1-1e-6, p)))

# ───────── Model loader + logistic scoring (kept) ─────────

def _validate_model_blob(name: str, tmp: dict) -> bool:
    if not isinstance(tmp, dict): return False
    if "weights" not in tmp or "intercept" not in tmp: return False
    if not isinstance(tmp["weights"], dict): return False
    if len(tmp["weights"]) > 2000: return False
    return True

MODEL_KEYS_ORDER = ["model_v2:{name}", "model_latest:{name}", "model:{name}"]

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

def predict_from_model(mdl: Dict[str, Any], features: Dict[str, float], market: str) -> float:
    """Linear logistic + conservative calibration stack."""
    w=mdl.get("weights") or {}; s=float(mdl.get("intercept",0.0))
    for k,v in w.items(): s+=float(v)*float(features.get(k,0.0))
    import math
    prob = 1/(1+math.exp(-s))
    prob = _calibrated_prob(prob, market, mdl.get("calibration") or {})
    return float(prob)

# ───────── Advanced Ensemble — in-play only ─────────

class AdvancedEnsemblePredictor:
    """Blend logistic + adjusted proxies for XGB/NN/momentum with dynamic weights."""
    def __init__(self):
        self.model_types = ['logistic', 'xgboost_proxy', 'nn_proxy', 'bayesian_live', 'momentum']
        self.ensemble_weights = {
            'logistic': 0.35, 'xgboost_proxy': 0.25, 'nn_proxy': 0.15, 'bayesian_live': 0.15, 'momentum': 0.10
        }
        self.performance_tracker = {}

    def predict_ensemble(self, features: Dict[str, float], market: str, minute: int) -> Tuple[float, float]:
        preds=[]
        for mt in self.model_types:
            try:
                p, c = self._predict_single(features, market, minute, mt)
                if p is not None:
                    preds.append((mt, float(max(1e-6,min(1-1e-6,p))), float(max(0.1,min(1.0,c)))))
            except Exception as e:
                log.debug("[ENSEMBLE] %s failed: %s", mt, e)
        if not preds: return 0.0, 0.0

        wsum=0.0; psum=0.0
        for mt,p,c in preds:
            base_w=self.ensemble_weights.get(mt,0.1)
            time_w = min(1.0, (minute or 0)/60.0) if mt in ("bayesian_live","momentum") else 1.0
            final_w = base_w * c * time_w
            psum += p*final_w; wsum += final_w
        p = psum/max(wsum,1e-9)
        conf = float(np.mean([c for _,_,c in preds])) if preds else 0.0
        # market-level reliability pass
        p = _apply_market_reliability(p, market)
        return float(p), float(conf)

    def _predict_single(self, features, market, minute, model_type):
        if model_type == 'logistic':
            mdl = self._load_market_model(market)
            if not mdl: return 0.5, 0.6
            return predict_from_model(mdl, features, market), 0.8
        if model_type == 'xgboost_proxy':
            # slight curvature/interaction proxy
            base = self._safe_base(features, market)
            adj = 0.02 * np.tanh(features.get("pressure_home",0)+features.get("pressure_away",0) - 100.0)
            return float(max(1e-6,min(1-1e-6, base + adj))), 0.7
        if model_type == 'nn_proxy':
            base = self._safe_base(features, market)
            curve = np.tanh((features.get("xg_sum",0)+features.get("sot_sum",0)*0.3)/3.0)*0.03
            return float(max(1e-6,min(1-1e-6, base + curve))), 0.65
        if model_type == 'bayesian_live':
            base = self._safe_base(features, market)
            prior = 0.5  # no prematch prior (in-play only)
            w = min(0.95, (minute or 0)/90.0)
            p = (prior*(1-w) + base*w)
            return float(p), 0.8
        if model_type == 'momentum':
            base = self._safe_base(features, market)
            goals_last_15 = float(features.get("goals_last_15",0))
            shots_last_15 = float(features.get("shots_last_15",0))
            bump = min(0.05, 0.02*goals_last_15 + 0.005*shots_last_15)
            if minute and minute>75: bump *= 0.5  # late-game caution
            return float(max(1e-6,min(1-1e-6, base + bump))), 0.6
        return None, 0.0

    def _safe_base(self, features, market):
        mdl = self._load_market_model(market)
        if not mdl: return 0.5
        return predict_from_model(mdl, features, market)

    def _load_market_model(self, market: str) -> Optional[Dict[str,Any]]:
        # Market identifiers: "BTTS", "OU_2.5", "OU_3.5", "1X2_HOME", "1X2_AWAY"
        if market.startswith("OU_"):
            return load_model_from_settings(market)
        return load_model_from_settings(market)

ensemble_predictor = AdvancedEnsemblePredictor()

# ───────── Market-specific predictor (draw suppressed for 1X2) ─────────

class MarketSpecificPredictor:
    def predict_btts(self, feat: Dict[str,float], minute:int) -> Tuple[float,float]:
        p, c = ensemble_predictor.predict_ensemble(feat, "BTTS", minute)
        # mild adjustments: balance & defensive quality
        bal = min(feat.get("pressure_home",0), feat.get("pressure_away",0))/100.0
        vuln = 1.0 - feat.get("defensive_stability", 0.5)
        p = max(1e-6,min(1-1e-6, p + 0.10*bal + 0.05*vuln))
        p = _apply_market_reliability(p, "BTTS")
        return float(p), float(max(0.0,min(1.0,c*0.95)))

    def predict_ou(self, feat: Dict[str,float], line: float, minute:int) -> Tuple[float,float]:
        market = f"OU_{_fmt_line(line)}"
        p, c = ensemble_predictor.predict_ensemble(feat, market, minute)
        # tempo/pressure influence
        pressure_total = feat.get("pressure_home",0)+feat.get("pressure_away",0)
        tempo_adj = 0.08 if pressure_total>150 else (-0.06 if pressure_total<80 else 0.0)
        p = max(1e-6,min(1-1e-6, p + tempo_adj))
        p = _apply_market_reliability(p, market)
        return float(p), float(max(0.0,min(1.0,c)))

    def predict_1x2_draw_suppressed(self, feat: Dict[str,float], minute:int) -> Tuple[float,float,float]:
        ph,_ = ensemble_predictor.predict_ensemble(feat, "1X2_HOME", minute)
        pa,_ = ensemble_predictor.predict_ensemble(feat, "1X2_AWAY", minute)
        s=max(1e-6, ph+pa)
        ph/=s; pa/=s
        # small momentum/psychology effects
        score_diff = feat.get("goals_h",0)-feat.get("goals_a",0)
        ph = max(1e-6,min(1-1e-6, ph + 0.03*np.tanh(score_diff)))
        pa = max(1e-6,min(1-1e-6, pa - 0.03*np.tanh(score_diff)))
        # renormalize
        s=max(1e-6, ph+pa); ph/=s; pa/=s
        ph=_apply_market_reliability(ph, "1X2_HOME")
        pa=_apply_market_reliability(pa, "1X2_AWAY")
        conf=0.7
        return float(ph), float(pa), float(conf)

market_predictor = MarketSpecificPredictor()

# ───────── Odds fetch + aggregation (LIVE only) ─────────
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

def fetch_odds(fid: int, prob_hints: Optional[dict[str, float]] = None) -> dict:
    """
    Aggregated LIVE odds map (no prematch fallback):
      { "BTTS": {...}, "1X2": {...}, "OU_2.5": {...}, ... }
    """
    now = time.time()
    k=("odds", fid)
    ts_empty = NEG_CACHE.get(k, (0.0, False))
    if ts_empty[1] and (now - ts_empty[0] < NEG_TTL_SEC): return {}
    if fid in ODDS_CACHE and now-ODDS_CACHE[fid][0] < 120: return ODDS_CACHE[fid][1]

    def _fetch_live() -> dict:
        js = api_get_with_sleep(f"{BASE_URL}/odds/live", {"fixture": fid}) or {}
        return js if isinstance(js, dict) else {}

    js = _fetch_live()

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

# ───────── EV / price gates ─────────
def _ev(prob: float, odds: float) -> float:
    """Return expected value as decimal (e.g. 0.05 = +5%)."""
    return float(prob)*max(0.0, float(odds)) - 1.0

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

def _parse_ou_line_from_suggestion(s: str) -> Optional[float]:
    """Extract line value from 'Over 2.5 Goals' / 'Under 3.5 Goals'."""
    try:
        import re
        m = re.search(r'(\d+\.?\d*)', s or "")
        return float(m.group(1)) if m else None
    except Exception:
        return None

def _market_family(market_text: str, suggestion: str) -> str:
    s = (market_text or "").upper()
    if s.startswith("OVER/UNDER") or "OVER/UNDER" in s:
        return "OU"
    if s == "BTTS" or "BTTS" in s:
        return "BTTS"
    if s == "1X2" or "WINNER" in s or "MATCH WINNER" in s:
        return "1X2"
    return s

def market_cutoff_ok(minute: int, market_text: str, suggestion: str) -> bool:
    """Minute cutoff by market family; uses MARKET_CUTOFFS / TIP_MAX_MINUTE / fallback."""
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
    Return (pass, odds, book, ev_pct). Enforces conservative odds/EV behavior:
      - If odds missing and ALLOW_TIPS_WITHOUT_ODDS=0 => block.
      - If odds present => must pass min/max odds; EV checked later with actual prob.
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

    return (True, odds, book, None)

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

# ───────── Formatting helpers ─────────
def _league_name(m: dict) -> Tuple[int,str]:
    lg=(m.get("league") or {}) or {}
    return int(lg.get("id") or 0), f"{lg.get('country','')} - {lg.get('name','')}".strip(" -")

def _teams(m: dict) -> Tuple[str,str]:
    t=(m.get("teams") or {}) or {}
    return (t.get("home",{}).get("name",""), t.get("away",{}).get("name",""))

def _pretty_score(m: dict) -> str:
    gh=(m.get("goals") or {}).get("home") or 0; ga=(m.get("goals") or {}).get("away") or 0
    return f"{gh}-{ga}"

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

# ───────── Game state analyzer ─────────
class GameStateAnalyzer:
    def __init__(self):
        self.critical_states = {
            'equalizer_seek': 0.7,  # Team needs equalizer
            'park_the_bus': 0.6,    # Team protecting lead
            'goal_fest': 0.8,       # High-scoring game
            'defensive_battle': 0.3 # Low-scoring game
        }
    
    def analyze_game_state(self, feat: Dict[str, float]) -> Dict[str, float]:
        state_scores = {}
        goal_diff = feat.get("goals_h", 0) - feat.get("goals_a", 0)
        minute = feat.get("minute", 0)
        total_goals = feat.get("goals_sum", 0)
        
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
        adjusted = predictions.copy()
        if game_state.get('equalizer_seek', 0) > 0.5:
            if 'BTTS: Yes' in adjusted:
                adjusted['BTTS: Yes'] *= (1 + game_state['equalizer_seek'] * 0.25)
            for key in list(adjusted.keys()):
                if key.startswith('Over'):
                    adjusted[key] *= (1 + game_state['equalizer_seek'] * 0.15)
        if game_state.get('park_the_bus', 0) > 0.5:
            for key in list(adjusted.keys()):
                if key.startswith('Over'):
                    adjusted[key] *= (1 - game_state['park_the_bus'] * 0.35)
                elif key == 'BTTS: Yes':
                    adjusted[key] *= (1 - game_state['park_the_bus'] * 0.25)
        return adjusted

# ───────── Production scan (in-play only) ─────────
def production_scan() -> Tuple[int, int]:
    """Main in-play scan with conservative calibration + EV gate."""
    if sleep_if_required():
        log.info("[SCAN] Skipping during sleep hours (22:00-08:00 Berlin time)")
        return (0, 0)
    
    if not _db_ping():
        log.error("[SCAN] Database unavailable")
        return (0, 0)

    try:
        matches = fetch_live_matches()
    except Exception as e:
        log.error("[SCAN] Failed to fetch live matches: %s", e)
        return (0, 0)
    
    live_seen = len(matches)
    if live_seen == 0:
        log.info("[SCAN] no live matches")
        return 0, 0

    saved = 0
    now_ts = int(time.time())
    per_league_counter: dict[int, int] = {}
    gsa = GameStateAnalyzer()

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

                feat = extract_enhanced_features(m)
                minute = int(feat.get("minute", 0))
                
                if not stats_coverage_ok(feat, minute):
                    continue
                if minute < TIP_MIN_MINUTE:
                    continue
                if is_feed_stale(fid, m, minute):
                    continue

                league_id, league = _league_name(m)
                home, away = _teams(m)
                score = _pretty_score(m)
                candidates: List[Tuple[str, str, float, float]] = []

                # Game state analysis once per match
                game_state = gsa.analyze_game_state(feat)

                # 1) BTTS
                btts_prob, btts_conf = market_predictor.predict_btts(feat, minute)
                if btts_prob > 0 and btts_conf > 0.5:
                    btts_predictions = { "BTTS: Yes": btts_prob, "BTTS: No": 1 - btts_prob }
                    adjusted = gsa.adjust_predictions(btts_predictions, game_state)
                    for suggestion, adj_prob in adjusted.items():
                        thr = _get_market_threshold("BTTS")
                        if adj_prob * 100 >= thr:
                            candidates.append(("BTTS", suggestion, float(adj_prob), float(btts_conf)))

                # 2) Over/Under lines
                for line in OU_LINES:
                    ou_prob, ou_conf = market_predictor.predict_ou(feat, line, minute)
                    if ou_prob > 0 and ou_conf > 0.5:
                        pred = { f"Over {_fmt_line(line)} Goals": ou_prob,
                                 f"Under {_fmt_line(line)} Goals": 1 - ou_prob }
                        adjusted = gsa.adjust_predictions(pred, game_state)
                        for suggestion, adj_prob in adjusted.items():
                            thr = _get_market_threshold(f"Over/Under {_fmt_line(line)}")
                            if adj_prob * 100 >= thr:
                                candidates.append((f"Over/Under {_fmt_line(line)}", suggestion, float(adj_prob), float(ou_conf)))

                # 3) 1X2 (draw suppressed)
                try:
                    prob_h, prob_a, conf_1x2 = market_predictor.predict_1x2_draw_suppressed(feat, minute)
                    if prob_h > 0 and prob_a > 0 and conf_1x2 > 0.5:
                        total = prob_h + prob_a
                        if total > 0:
                            prob_h /= total; prob_a /= total
                        pred_1x2 = { "Home Win": prob_h, "Away Win": prob_a }
                        adjusted = gsa.adjust_predictions(pred_1x2, game_state)
                        for suggestion, adj_prob in adjusted.items():
                            thr = _get_market_threshold("1X2")
                            if adj_prob * 100 >= thr:
                                candidates.append(("1X2", suggestion, float(adj_prob), float(conf_1x2)))
                except Exception as e:
                    log.debug("[1X2] skip: %s", e)

                if not candidates:
                    continue

                # Odds analysis and EV filtering
                odds_map = fetch_odds(fid)
                ranked: List[Tuple[str, str, float, Optional[float], Optional[str], Optional[float], float, float]] = []

                for mk, sug, prob, conf in candidates:
                    if sug not in ALLOWED_SUGGESTIONS:
                        continue
                    if not market_cutoff_ok(minute, mk, sug):
                        continue
                    if not _candidate_is_sane(sug, feat):
                        continue

                    # Lookup odds and EV
                    pass_odds, odds, book, _ = _price_gate(mk, sug, fid)
                    if not pass_odds:
                        continue

                    ev_pct = None
                    if odds is not None:
                        edge = _ev(prob, float(odds))
                        ev_pct = round(edge * 100.0, 1)
                        if int(round(edge * 10000)) < EDGE_MIN_BPS:
                            continue
                    else:
                        if not ALLOW_TIPS_WITHOUT_ODDS:
                            continue

                    # Rank by conservative score
                    rank_score = (prob ** 1.15) * (1 + (ev_pct or 0) / 120.0) * (0.9 + 0.1*conf)
                    ranked.append((mk, sug, prob, odds, book, ev_pct, rank_score, conf))

                if not ranked:
                    continue

                ranked.sort(key=lambda x: x[6], reverse=True)

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

                            sent = send_telegram(_format_enhanced_tip_message(
                                home, away, league, minute, score, suggestion, 
                                float(prob_pct), feat, odds, book, ev_pct, confidence
                            ))
                            
                            if sent:
                                c2.execute("UPDATE tips SET sent_ok=1 WHERE match_id=%s AND created_ts=%s", (fid, created_ts))
                                _metric_inc("tips_sent_total", n=1)
                                log.info("[TIP_SENT] %s | %s vs %s | %s' | %.1f%% | EV=%s",
                                         suggestion, home, away, minute, prob_pct, f"{ev_pct:+.1f}%" if ev_pct is not None else "n/a")
                    except Exception as e:
                        log.exception("[SCAN] insert/send failed: %s", e)
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
                log.exception("[SCAN] match loop failed: %s", e)
                continue

    log.info("[SCAN] saved=%d live_seen=%d", saved, live_seen)
    _metric_inc("tips_generated_total", n=saved)
    return saved, live_seen

# ───────── Results backfill & daily digest (in-play tips only) ─────────
def _fixture_by_id(mid: int) -> Optional[dict]:
    js=api_get_with_sleep(FOOTBALL_API_URL, {"id": mid}) or {}
    arr=js.get("response") or [] if isinstance(js,dict) else []
    return arr[0] if arr else None

def _is_final(short: str) -> bool: 
    return (short or "").upper() in {"FT","AET","PEN"}

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

def backfill_results_for_open_matches(max_rows: int = 200) -> int:
    """Backfill results for recent tips whose matches are now final."""
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
    """Daily accuracy digest for today's tips (in-play only)."""
    if not DAILY_ACCURACY_DIGEST_ENABLE: 
        return None
    
    today = datetime.now(BERLIN_TZ).date()
    start_of_day = datetime.combine(today, datetime.min.time(), tzinfo=BERLIN_TZ)
    start_ts = int(start_of_day.timestamp())
    
    log.info("[DIGEST] Generating daily digest for today (since %s)", start_of_day)
    backfill_results_for_open_matches(400)

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
        if rows:
            pending = len([r for r in rows if r[5] is None or r[6] is None])
            msg += f"\n⏳ {pending} tips still pending results."
    else:
        acc = 100.0 * wins / max(1, graded)
        lines = [
            f"📊 <b>Daily Accuracy Digest</b> - {today.strftime('%Y-%m-%d')}",
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
    log.info("[DIGEST] Sent daily digest with %d tips, %d graded", total, graded)
    return msg

# ───────── Retry unsent tips (sends only; no new generation) ─────────
def _format_tip_message(home, away, league, minute, score, suggestion, prob_pct, feat, odds=None, book=None, ev_pct=None):
    stat=""
    if any([feat.get("xg_h",0),feat.get("xg_a",0),feat.get("sot_h",0),feat.get("sot_a",0),
            feat.get("cor_h",0),feat.get("cor_a",0),
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

# ───────── Metrics & health helpers ─────────
if "METRICS" not in globals():
    METRICS: dict[str, float] = {}

def _metric_inc(name: str, n: float = 1.0) -> None:
    try:
        METRICS[name] = METRICS.get(name, 0.0) + float(n)
    except Exception:
        pass

def _metric_get_all() -> str:
    # Prometheus-style text for simplicity
    out = []
    for k, v in sorted(METRICS.items()):
        safe = k.replace(" ", "_")
        out.append(f"{safe} {v}")
    return "\n".join(out) + "\n"

# ───────── Basic HTML escape guard (used in messages) ─────────
try:
    from html import escape as _html_escape  # noqa
except Exception:
    _html_escape = lambda s, quote=True: str(s)

if "escape" not in globals():
    def escape(s) -> str:
        return _html_escape(str(s), quote=False)

# ───────── Timezone guard (Berlin default) ─────────
try:
    from zoneinfo import ZoneInfo  # Python 3.11 OK
    BERLIN_TZ = BERLIN_TZ if "BERLIN_TZ" in globals() else ZoneInfo("Europe/Berlin")
except Exception:
    from datetime import timezone, timedelta as _td
    BERLIN_TZ = timezone(_td(hours=0))  # UTC fallback

# ───────── PG advisory lock wrapper ─────────
def _run_with_pg_lock(lock_id: int, fn, timeout_sec: int = 2) -> bool:
    """
    Run `fn()` only if we can take the advisory lock.
    Returns True if executed, False otherwise.
    """
    try:
        with db_conn() as c:
            got = c.execute("SELECT pg_try_advisory_lock(%s)", (int(lock_id),)).fetchone()
            if not got or not got[0]:
                return False
            try:
                fn()
                return True
            finally:
                try:
                    c.execute("SELECT pg_advisory_unlock(%s)", (int(lock_id),))
                except Exception:
                    pass
    except Exception as e:
        log.error("[LOCK %s] failed: %s", lock_id, e)
        return False

# ───────── DB ping guard ─────────
def _db_ping() -> bool:
    try:
        with db_conn() as c:
            r = c.execute("SELECT 1").fetchone()
            return bool(r and r[0] == 1)
    except Exception:
        return False

# ───────── Simple HTTP admin (no extra deps) ─────────
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

HTTP_BIND = os.getenv("HTTP_BIND", "0.0.0.0")
HTTP_PORT = int(os.getenv("PORT", os.getenv("HTTP_PORT", "8080")))

class _AdminHTTP(BaseHTTPRequestHandler):
    def _write(self, code=200, body: str = "ok\n", ctype="text/plain; charset=utf-8"):
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body.encode("utf-8"))))
            self.end_headers()
            self.wfile.write(body.encode("utf-8"))
        except Exception:
            pass

    def do_GET(self):
        p = urlparse(self.path)
        if p.path == "/health":
            ok = _db_ping()
            self._write(200 if ok else 500, "ok\n" if ok else "db down\n")
            return
        if p.path == "/metrics":
            self._write(200, _metric_get_all())
            return
        if p.path == "/version":
            ver = os.getenv("APP_VERSION", "inplay-labs-1")
            self._write(200, f"{ver}\n")
            return
        self._write(404, "not found\n")

    def do_POST(self):
        p = urlparse(self.path)
        qs = parse_qs(p.query or "")
        if p.path == "/scan-now":
            ran = _run_with_pg_lock(2001, lambda: production_scan())
            self._write(200, f"scan {'started' if ran else 'skipped(lock)'}\n")
            return
        if p.path == "/retry":
            mins = int(qs.get("mins", ["30"])[0])
            n = retry_unsent_tips(mins, 200)
            self._write(200, f"retried {n}\n")
            return
        if p.path == "/backfill":
            n = backfill_results_for_open_matches(400)
            self._write(200, f"backfilled {n}\n")
            return
        if p.path == "/digest":
            msg = daily_accuracy_digest(1) or "digest disabled\n"
            self._write(200, (msg if isinstance(msg, str) else "ok\n"))
            return
        self._write(404, "not found\n")

def _start_http_admin():
    def _serve():
        httpd = HTTPServer((HTTP_BIND, HTTP_PORT), _AdminHTTP)
        log.info("[HTTP] admin listening on %s:%s", HTTP_BIND, HTTP_PORT)
        httpd.serve_forever()
    t = threading.Thread(target=_serve, name="http-admin", daemon=True)
    t.start()
    return t

# ───────── Lightweight in-process scheduler (no APScheduler) ─────────
def _env_bool(key: str, default: str = "1") -> bool:
    return os.getenv(key, default) not in ("0", "false", "False", "no", "NO")

SCAN_EVERY_SEC        = globals().get("SCAN_EVERY_SEC",        int(os.getenv("SCAN_EVERY_SEC", "420")))   # 7 min
RETRY_EVERY_SEC       = globals().get("RETRY_EVERY_SEC",       int(os.getenv("RETRY_EVERY_SEC", "300")))  # 5 min
BACKFILL_EVERY_SEC    = globals().get("BACKFILL_EVERY_SEC",    int(os.getenv("BACKFILL_EVERY_SEC", "900"))) # 15 min
RUN_SCHEDULER         = _env_bool("RUN_SCHEDULER", "1")
RUN_HTTP              = _env_bool("RUN_HTTP", "1")
DAILY_ACCURACY_DIGEST_ENABLE = globals().get("DAILY_ACCURACY_DIGEST_ENABLE", _env_bool("DAILY_ACCURACY_DIGEST_ENABLE", "1"))

class _Scheduler(threading.Thread):
    def __init__(self):
        super().__init__(name="inplay-scheduler", daemon=True)
        self._stop = threading.Event()
        self._t_last = {"scan": 0, "retry": 0, "backfill": 0}
        self._last_digest_date = None

    def stop(self):
        self._stop.set()

    def _maybe_run(self, key: str, every: int, fn, lock_id: int):
        now = time.time()
        if now - self._t_last[key] >= every:
            def _wrapped():
                try:
                    fn()
                except Exception as e:
                    log.exception("[SCHED:%s] failed: %s", key, e)
            ran = _run_with_pg_lock(lock_id, _wrapped)
            self._t_last[key] = now if ran else self._t_last[key]

    def _maybe_daily_digest(self):
        if not DAILY_ACCURACY_DIGEST_ENABLE:
            return
        now_local = datetime.now(BERLIN_TZ)
        today = now_local.date()
        hhmm = now_local.strftime("%H%M")
        # Send digest once around 23:50–23:59 local time
        if (self._last_digest_date != today) and ("235" in hhmm or hhmm >= "2350"):
            ran = _run_with_pg_lock(2010, lambda: daily_accuracy_digest(1))
            if ran:
                self._last_digest_date = today

    def run(self):
        log.info("[SCHED] loop started (scan=%ss retry=%ss backfill=%ss)", SCAN_EVERY_SEC, RETRY_EVERY_SEC, BACKFILL_EVERY_SEC)
        while not self._stop.is_set():
            try:
                self._maybe_run("scan", SCAN_EVERY_SEC, production_scan, 2003)
                self._maybe_run("retry", RETRY_EVERY_SEC, lambda: retry_unsent_tips(30, 200), 2004)
                self._maybe_run("backfill", BACKFILL_EVERY_SEC, lambda: backfill_results_for_open_matches(400), 2005)
                self._maybe_daily_digest()
            except Exception as e:
                log.exception("[SCHED] tick failed: %s", e)
            time.sleep(2.0)

def _start_scheduler_once() -> Optional[_Scheduler]:
    if not RUN_SCHEDULER:
        log.info("[SCHED] disabled by env")
        return None
    s = _Scheduler()
    s.start()
    return s

# ───────── Boot sequence ─────────
def _print_boot_banner():
    log.info("────────────────────────────────────────────────────────")
    log.info(" In-Play Engine (no prematch/MOTD) — booting")
    log.info("  • Scheduler: %s  |  HTTP admin: %s", "ON" if RUN_SCHEDULER else "OFF", "ON" if RUN_HTTP else "OFF")
    log.info("  • Scan interval: %ss, Retry: %ss, Backfill: %ss", SCAN_EVERY_SEC, RETRY_EVERY_SEC, BACKFILL_EVERY_SEC)
    log.info("  • Sleep window: %s", os.getenv("SLEEP_WINDOW", "22:00–08:00 Berlin"))
    log.info("────────────────────────────────────────────────────────")

# Main entrypoint
if __name__ == "__main__":
    _print_boot_banner()

    # Warm DB
    if _db_ping():
        log.info("[BOOT] DB OK")
    else:
        log.warning("[BOOT] DB unavailable; continuing, will retry on demand")

    # Start admin HTTP if requested
    if RUN_HTTP:
        _start_http_admin()

    # Start scheduler
    _start_scheduler_once()

    # Keep process alive
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        log.info("shutting down…")
