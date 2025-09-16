# file: main.py
"""
goalsniper — FULL AI mode (in-play + prematch) with odds + EV gate.

- Pure ML (calibrated) loaded from Postgres settings (train_models.py).
- Markets: OU(2.5,3.5), BTTS (Yes/No), 1X2 (Draw suppressed).
- Adds bookmaker odds filtering + EV check.
- Scheduler: scan, results backfill, nightly train, daily digest, MOTD.

Safe to run on Railway/Render. Requires DATABASE_URL and API keys.

Changes in this version:
- Precision guardrails hardened and automated:
  - ALLOW_TIPS_WITHOUT_ODDS defaults to 0 (no odds => no publish unless explicitly enabled)
  - EDGE_MIN_BPS defaults to 600 (+6.0% EV)
  - CONF_THRESHOLD defaults to 75 (global floor; per-market thresholds still supported)
  - MIN_ODDS_* defaults to 1.50
  - TIP_MIN_MINUTE=12, PREDICTIONS_PER_MATCH=1, PER_LEAGUE_CAP=2 (volume stabilizers)
- EV enforcement is consistent across in-play and prematch paths.
- Timezone normalized to Europe/Berlin for all user-facing messages and rollups.
- Daily digest includes a 7-day rolling precision and ROI summary.
- Admin endpoints to export/import thresholds to pin/rollback quickly.

Additional hardening in this build:
- Supabase/Railway friendly Postgres pooling (pgBouncer Transaction mode):
  small pool, keepalives, statement/idle timeouts, auto-recover on drops.
- Advisory lock runner recovers from transient pool errors.
"""

import os, json, time, logging, requests, psycopg2
import numpy as np
from psycopg2.pool import SimpleConnectionPool
from psycopg2 import OperationalError, InterfaceError
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

DATABASE_URL = os.getenv("DATABASE_URL")  # IMPORTANT: use Supabase POOLER (pgBouncer Transaction) URL, usually port 6543
if not DATABASE_URL:
    raise SystemExit("DATABASE_URL is required (use Supabase Connection Pooling / Transaction URL)")

POOL_MINCONN = int(os.getenv("PGPOOL_MIN", "1"))
POOL_MAXCONN = int(os.getenv("PGPOOL_MAX", "4"))
PG_LOCK_RETRY_SLEEP = float(os.getenv("PG_LOCK_RETRY_SLEEP", "0.8"))
PG_LOCK_MAX_RETRIES = int(os.getenv("PG_LOCK_MAX_RETRIES", "1"))

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
PREDICTIONS_PER_MATCH = int(os.getenv("PREDICTIONS_PER_MATCH", "1"))
PER_LEAGUE_CAP        = int(os.getenv("PER_LEAGUE_CAP", "2"))

# ───────── Odds/EV controls — UPDATED DEFAULTS ─────────
MIN_ODDS_OU   = float(os.getenv("MIN_ODDS_OU",   "1.50"))
MIN_ODDS_BTTS = float(os.getenv("MIN_ODDS_BTTS", "1.50"))
MIN_ODDS_1X2  = float(os.getenv("MIN_ODDS_1X2",  "1.50"))
MAX_ODDS_ALL  = float(os.getenv("MAX_ODDS_ALL",  "20.0"))
EDGE_MIN_BPS  = int(os.getenv("EDGE_MIN_BPS", "600"))
ODDS_BOOKMAKER_ID = os.getenv("ODDS_BOOKMAKER_ID")
ALLOW_TIPS_WITHOUT_ODDS = os.getenv("ALLOW_TIPS_WITHOUT_ODDS","0") not in ("0","false","False","no","NO")

# Aggregated odds controls
ODDS_SOURCE = os.getenv("ODDS_SOURCE", "auto").lower()             # auto|live|prematch
ODDS_AGGREGATION = os.getenv("ODDS_AGGREGATION", "median").lower() # median|best
ODDS_OUTLIER_MULT = float(os.getenv("ODDS_OUTLIER_MULT", "1.8"))
ODDS_REQUIRE_N_BOOKS = int(os.getenv("ODDS_REQUIRE_N_BOOKS", "2"))
ODDS_FAIR_MAX_MULT = float(os.getenv("ODDS_FAIR_MAX_MULT", "2.5"))

# ───────── Markets allow-list (draw suppressed) ─────────
ALLOWED_SUGGESTIONS = {"BTTS: Yes", "BTTS: No", "Home Win", "Away Win"}
def _fmt_line(line: float) -> str: return f"{line}".rstrip("0").rstrip(".")
for _ln in OU_LINES:
    s=_fmt_line(_ln); ALLOWED_SUGGESTIONS.add(f"Over {s} Goals"); ALLOWED_SUGGESTIONS.add(f"Under {s} Goals")

# ───────── External APIs / HTTP session ─────────
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

# ───────── Optional import: trainer ─────────
try:
    import train_models as _tm
    train_models = _tm.train_models
except Exception as e:
    _IMPORT_ERR = repr(e)
    def train_models(*args, **kwargs):  # type: ignore
        log.warning("train_models not available: %s", _IMPORT_ERR)
        return {"ok": False, "reason": f"train_models import failed: {_IMPORT_ERR}"}

# ───────── DB pool & helpers (hardened for Supabase/pgBouncer) ─────────
from psycopg2 import pool as _pgpool

POOL: Optional[SimpleConnectionPool] = None

def _mk_dsn(base: str) -> str:
    """Append safe defaults to DSN without duplicating existing params."""
    if "sslmode=" not in base:
        base += ("&" if "?" in base else "?") + "sslmode=require"
    if "application_name=" not in base:
        base += "&application_name=goalsniper"
    if "target_session_attrs=" not in base:
        base += "&target_session_attrs=any"
    return base

def _init_pool() -> None:
    """Create a tiny pool that plays nice with pgBouncer Transaction mode."""
    global POOL
    dsn = _mk_dsn(DATABASE_URL)
    minc = int(os.getenv("PGPOOL_MIN", "1"))
    maxc = int(os.getenv("PGPOOL_MAX", "4"))
    connect_timeout = int(os.getenv("PG_CONNECT_TIMEOUT", "6"))

    stmt_ms = int(os.getenv("PG_STATEMENT_TIMEOUT_MS", "15000"))
    idle_tx_ms = int(os.getenv("PG_IDLE_IN_TXN_TIMEOUT_MS", "5000"))
    options = f"-c statement_timeout={stmt_ms} -c idle_in_transaction_session_timeout={idle_tx_ms}"

    ka = int(os.getenv("PG_KEEPALIVES", "1"))
    ka_idle = int(os.getenv("PG_KEEPALIVES_IDLE", "30"))
    ka_itv = int(os.getenv("PG_KEEPALIVES_INTERVAL", "10"))
    ka_cnt = int(os.getenv("PG_KEEPALIVES_COUNT", "3"))

    POOL = SimpleConnectionPool(
        minconn=max(1, minc),
        maxconn=max(1, maxc),
        dsn=dsn,
        connect_timeout=connect_timeout,
        options=options,
        keepalives=ka,
        keepalives_idle=ka_idle,
        keepalives_interval=ka_itv,
        keepalives_count=ka_cnt,
    )

class PooledConn:
    """Context manager that auto-recovers if pgBouncer closes connections."""
    def __init__(self, pool: SimpleConnectionPool):
        self.pool = pool
        self.conn = None
        self.cur = None

    def __enter__(self):
        global POOL
        for attempt in (1, 2):
            try:
                self.conn = self.pool.getconn()
                self.conn.autocommit = True
                self.cur = self.conn.cursor()
                return self
            except (_pgpool.PoolError, OperationalError, InterfaceError) as e:
                log.warning("[DB] getconn failed (attempt %d): %s", attempt, e)
                _reset_pool()
                if POOL is None:
                    _init_pool()
                self.pool = POOL
                time.sleep(0.2)
        raise

    def __exit__(self, exc_type, exc, tb):
        try:
            if self.cur:
                self.cur.close()
        finally:
            try:
                if self.conn and self.pool:
                    self.pool.putconn(self.conn)
            except (_pgpool.PoolError, InterfaceError):
                pass

    def execute(self, sql: str, params: tuple | list = ()):
        try:
            self.cur.execute(sql, params or ())
            return self.cur
        except (OperationalError, InterfaceError) as e:
            log.warning("[DB] execute transient error, retrying once: %s", e)
            try:
                if self.cur:
                    self.cur.close()
            except Exception:
                pass
            self.conn = self.pool.getconn()
            self.conn.autocommit = True
            self.cur = self.conn.cursor()
            self.cur.execute(sql, params or ())
            return self.cur
        except Exception as e:
            log.error("DB execute failed: %s\nSQL: %s\nParams: %s", e, sql, params)
            raise

def _reset_pool() -> None:
    """Close and null the pool so the next db_conn() recreates it cleanly."""
    global POOL
    try:
        if POOL:
            POOL.closeall()
    except Exception:
        pass
    POOL = None

def db_conn() -> PooledConn:
    """Always return a usable connection context (re-inits pool if needed)."""
    global POOL
    if POOL is None:
        _init_pool()
    return PooledConn(POOL)  # type: ignore

# ───────── Settings cache ─────────
class _TTLCache:
    def __init__(self, ttl): self.ttl=ttl; self.data={}
    def get(self, k):
        v=self.data.get(k)
        if not v: return None
        ts,val=v
        if time.time()-ts>self.ttl: self.data.pop(k,None); return None
        return val
    def set(self,k,v): self.data[k]=(time.time(),v)
    def invalidate(self,k=None): self.data.clear() if k is None else self.data.pop(k,None)

_SETTINGS_CACHE, _MODELS_CACHE = _TTLCache(SETTINGS_TTL), _TTLCache(MODELS_TTL)

def get_setting(key: str) -> Optional[str]:
    with db_conn() as c:
        r=c.execute("SELECT value FROM settings WHERE key=%s",(key,)).fetchone()
        return r[0] if r else None

def set_setting(key: str, value: str) -> None:
    with db_conn() as c:
        c.execute("INSERT INTO settings(key,value) VALUES(%s,%s) ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value", (key,value))

def get_setting_cached(key: str) -> Optional[str]:
    v=_SETTINGS_CACHE.get(key)
    if v is None: v=get_setting(key); _SETTINGS_CACHE.set(key,v)
    return v

def invalidate_model_caches_for_key(key: str):
    if key.lower().startswith(("model","model_latest","model_v2","pre_")): _MODELS_CACHE.invalidate()

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

# ───────── Telegram ─────────
def send_telegram(text: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return False
    try:
        r=session.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                       data={"chat_id":TELEGRAM_CHAT_ID,"text":text,"parse_mode":"HTML","disable_web_page_preview":True}, timeout=10)
        return r.ok
    except Exception:
        return False

# ───────── API helpers ─────────
def _api_get(url: str, params: dict, timeout: int = 15):
    if not API_KEY: return None
    try:
        r=session.get(url, headers=HEADERS, params=params, timeout=timeout)
        return r.json() if r.ok else None
    except Exception:
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

# ───────── Live fetches ─────────
def fetch_match_stats(fid: int) -> list:
    now=time.time()
    if fid in STATS_CACHE and now-STATS_CACHE[fid][0] < 90: return STATS_CACHE[fid][1]
    js=_api_get(f"{FOOTBALL_API_URL}/statistics", {"fixture": fid}) or {}
    out=js.get("response",[]) if isinstance(js,dict) else []
    STATS_CACHE[fid]=(now,out); return out

def fetch_match_events(fid: int) -> list:
    now=time.time()
    if fid in EVENTS_CACHE and now-EVENTS_CACHE[fid][0] < 90: return EVENTS_CACHE[fid][1]
    js=_api_get(f"{FOOTBALL_API_URL}/events", {"fixture": fid}) or {}
    out=js.get("response",[]) if isinstance(js,dict) else []
    EVENTS_CACHE[fid]=(now,out); return out

def fetch_live_matches() -> List[dict]:
    js=_api_get(FOOTBALL_API_URL, {"live":"all"}) or {}
    matches=[m for m in (js.get("response",[]) if isinstance(js,dict) else []) if not _blocked_league(m.get("league") or {})]
    out=[]
    for m in matches:
        st=((m.get("fixture",{}) or {}).get("status",{}) or {})
        elapsed=st.get("elapsed"); short=(st.get("short") or "").upper()
        if elapsed is None or elapsed>120 or short not in INPLAY_STATUSES: continue
        fid=(m.get("fixture",{}) or {}).get("id")
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
    home = m["teams"]["home"]["name"]
    away = m["teams"]["away"]["name"]
    gh = m["goals"]["home"] or 0
    ga = m["goals"]["away"] or 0
    minute = int(((m.get("fixture") or {}).get("status") or {}).get("elapsed") or 0)

    stats = {}
    for s in (m.get("statistics") or []):
        t = (s.get("team") or {}).get("name")
        if t:
            stats[t] = { (i.get("type") or ""): i.get("value") for i in (s.get("statistics") or []) }

    sh = stats.get(home, {}) or {}
    sa = stats.get(away, {}) or {}

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

# ---- Optional: calibration from graded tips (kept but hardened) ----
def calibrate_and_retune_from_tips(conn, target_precision: float,
                                   min_preds: int, min_thr_pct: float, max_thr_pct: float,
                                   days: int = 365) -> Dict[str, float]:
    try:
        df = load_graded_tips(conn, days=days)  # from train_models
    except Exception as e:
        log.info("Tips calibration skipped (helpers not available): %s", e)
        return {}

    if df.empty:
        log.info("Tips calibration: no graded tips found.")
        return {}

    updates: Dict[str, float] = {}
    def _do(market_name: str, mask_market):
        sub = df[mask_market].copy()
        if len(sub) < max(200, min_preds*3):
            return
        p_raw = sub["prob"].to_numpy()
        y = sub["y"].to_numpy().astype(int)
        a, b = fit_platt(y, p_raw)
        key_map = {
            "BTTS":           "BTTS_YES",
            "Over/Under 2.5": "OU_2.5",
            "Over/Under 3.5": "OU_3.5",
            "1X2":            None,
        }
        model_key = key_map.get(market_name)
        if model_key:
            blob = _get_setting_json(conn, f"model_latest:{model_key}")
            if blob:
                blob["calibration"] = {"method": "platt", "a": float(a), "b": float(b)}
                for k in (f"model_latest:{model_key}", f"model:{model_key}"):
                    _set_setting(conn, k, json.dumps(blob))

        z = _logit_vec(p_raw); p_cal = 1.0/(1.0+np.exp(-(a*z + b)))
        thr_prob = _pick_threshold_for_target_precision(
            y_true=y, p_cal=p_cal, target_precision=target_precision,
            min_preds=min_preds, default_threshold=0.65,
        )
        thr_pct = float(np.clip(_percent(thr_prob), min_thr_pct, max_thr_pct))
        _set_setting(conn, f"conf_threshold:{market_name}", f"{thr_pct:.2f}")
        updates[market_name] = thr_pct

    _do("BTTS",                 df["market"] == "BTTS")
    _do("Over/Under 2.5",      df["market"].eq("Over/Under 2.5"))
    _do("Over/Under 3.5",      df["market"].eq("Over/Under 3.5"))
    _do("1X2",                 df["market"] == "1X2")

    if updates:
        log.info("Tips calibration/threshold updates: %s", updates)
    return updates
# ---- end optional block ----

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

# ───────── Models ─────────
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
def load_model_from_settings(name: str) -> Optional[Dict[str,Any]]:
    cached=_MODELS_CACHE.get(name)
    if cached is not None: return cached
    mdl=None
    for pat in MODEL_KEYS_ORDER:
        raw=get_setting_cached(pat.format(name=name))
        if not raw: continue
        try:
            tmp=json.loads(raw); tmp.setdefault("intercept",0.0); tmp.setdefault("weights",{})
            cal=tmp.get("calibration") or {}; 
            if isinstance(cal,dict): cal.setdefault("method","sigmoid"); cal.setdefault("a",1.0); cal.setdefault("b",0.0); tmp["calibration"]=cal
            mdl=tmp; break
        except Exception as e:
            log.warning("[MODEL] parse %s failed: %s", name, e)
    if mdl is not None: _MODELS_CACHE.set(name, mdl)
    return mdl
def _linpred(feat: Dict[str,float], weights: Dict[str,float], intercept: float) -> float:
    s=float(intercept or 0.0)
    for k,w in (weights or {}).items(): s += float(w or 0.0)*float(feat.get(k,0.0))
    return s

def _pick_bin_params(cal_bins: list|None, minute: float) -> Optional[Tuple[float,float]]:
    if not cal_bins: return None
    try:
        m=float(minute or 0.0)
        for b in cal_bins:
            lo=float((b or {}).get("lo", -1e9)); hi=float((b or {}).get("hi", 1e9))
            if lo <= m < hi:
                return float(b.get("a", 1.0)), float(b.get("b", 0.0))
    except Exception:
        pass
    return None

def _calibrate_base(p: float, cal: Dict[str,Any]) -> float:
    method=(cal or {}).get("method","sigmoid"); a=float((cal or {}).get("a",1.0)); b=float((cal or {}).get("b",0.0))
    if method.lower()=="platt": return _sigmoid(a*_logit(p)+b)
    import math; p=max(EPS,min(1-EPS,float(p))); z=math.log(p/(1-p)); return _sigmoid(a*z+b)

def _score_prob(feat: Dict[str,float], mdl: Dict[str,Any]) -> float:
    p=_sigmoid(_linpred(feat, mdl.get("weights",{}), float(mdl.get("intercept",0.0))))
    p=float(max(0.0, min(1.0, p)))
    cal_bins = mdl.get("calibration_by_minute")
    bin_params = _pick_bin_params(cal_bins, feat.get("minute", 0.0))
    if bin_params is not None:
        a,b = bin_params
        return float(_sigmoid(a*_logit(p) + b))
    try: 
        cal=mdl.get("calibration") or {}
        return float(_calibrate_base(p, cal)) if cal else p
    except: 
        return p

def _load_ou_model_for_line(line: float) -> Optional[Dict[str,Any]]:
    name=f"OU_{_fmt_line(line)}"; mdl=load_model_from_settings(name)
    return mdl or (load_model_from_settings("O25") if abs(line-2.5)<1e-6 else None)
def _load_wld_models(): return (load_model_from_settings("WLD_HOME"), load_model_from_settings("WLD_DRAW"), load_model_from_settings("WLD_AWAY"))

# ───────── Odds helpers ─────────
def _ev(prob: float, odds: float) -> float:
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

def _market_name_normalize(s: str) -> str:
    s=(s or "").lower()
    if "both teams" in s or "btts" in s: return "BTTS"
    if "match winner" in s or "winner" in s or "1x2" in s: return "1X2"
    if "over/under" in s or "total" in s or "goals" in s: return "OU"
    return s

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

def _aggregate_price(vals: list[tuple[float, str]], prob_hint: Optional[float]) -> tuple[Optional[float], Optional[str]]:
    if not vals:
        return None, None
    xs = sorted([o for (o, _) in vals if (o or 0) > 0])
    if not xs:
        return None, None
    import statistics
    med = statistics.median(xs)
    cleaned = [(o, b) for (o, b) in vals if o <= med * max(1.0, ODDS_OUTLIER_MULT)]
    if not cleaned:
        cleaned = vals
    xs2 = sorted([o for (o, _) in cleaned])
    med2 = statistics.median(xs2)
    if prob_hint is not None and prob_hint > 0:
        fair = 1.0 / max(1e-6, float(prob_hint))
        cap = fair * max(1.0, ODDS_FAIR_MAX_MULT)
        cleaned = [(o, b) for (o, b) in cleaned if o <= cap] or cleaned
    if ODDS_AGGREGATION == "best":
        best = max(cleaned, key=lambda t: t[0])
        return float(best[0]), str(best[1])
    target = med2
    pick = min(cleaned, key=lambda t: abs(t[0] - target))
    return float(pick[0]), f"{pick[1]} (median of {len(xs)})"

def fetch_odds(fid: int, prob_hints: Optional[dict[str, float]] = None) -> dict:
    cached = _odds_cache_get(fid)
    if cached is not None:
        return cached

    def _fetch(path: str) -> dict:
        js = _api_get(f"{BASE_URL}/{path}", {"fixture": fid}) or {}
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
    return out

def _price_gate(market_text: str, suggestion: str, fid: int) -> Tuple[bool, Optional[float], Optional[str], Optional[float]]:
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

def _implied_prob_from_odds(odds: Optional[float]) -> Optional[float]:
    try:
        o = float(odds or 0.0)
        if o <= 1.0:
            return None
        return max(0.01, min(0.99, 1.0 / o))
    except Exception:
        return None

def _apply_stack_if_available(p_model: float, mdl: Optional[Dict[str,Any]], p_book: Optional[float]) -> float:
    try:
        if not mdl:
            return float(p_model)
        stack = mdl.get("stack")
        if not stack or p_book is None:
            return float(p_model)
        a_model = float(stack.get("a_model", 0.0))
        a_book  = float(stack.get("a_book",  0.0))
        b0      = float(stack.get("b",       0.0))
        z = a_model*_logit(p_model) + a_book*_logit(p_book) + b0
        return float(_sigmoid(z))
    except Exception:
        return float(p_model)

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
        "gh": gh,
        "ga": ga,
        "league_id": league_id,
        "market": "HARVEST",
        "suggestion": "HARVEST",
        "confidence": 0,
        "stat": {
            "xg_h": xg_h, "xg_a": xg_a, "xg_sum": xg_sum, "xg_diff": xg_diff,
            "sot_h": sot_h, "sot_a": sot_a, "sot_sum": sot_sum,
            "cor_h": cor_h, "cor_a": cor_a, "cor_sum": cor_sum,
            "pos_h": pos_h, "pos_a": pos_a, "pos_diff": pos_diff,
            "red_h": red_h, "red_a": red_a, "red_sum": red_sum,
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

# ───────── Outcomes/backfill/digest (short) ─────────
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

# DAILY_DIGEST_WINDOW_DAYS var (fallback)
DAILY_DIGEST_WINDOW_DAYS = int(os.getenv("DAILY_DIGEST_WINDOW_DAYS", "1"))

def daily_accuracy_digest(window_days: Optional[int] = None) -> Optional[str]:
    try:
        if window_days is None:
            try:
                window_days = int(os.getenv("DAILY_DIGEST_WINDOW_DAYS", str(DAILY_DIGEST_WINDOW_DAYS)))
            except Exception:
                window_days = 1
        window_days = max(1, int(window_days))
        backfill_results_for_open_matches(400)
        now_local = datetime.now(BERLIN_TZ)
        cutoff_ts = int((now_local - timedelta(days=window_days)).timestamp())

        with db_conn() as c:
            rows = c.execute(
                """
                SELECT
                    t.market,
                    t.suggestion,
                    COALESCE(t.confidence_raw, t.confidence/100.0) AS prob,
                    t.odds,
                    t.created_ts,
                    r.final_goals_h, r.final_goals_a, r.btts_yes
                FROM tips t
                LEFT JOIN match_results r ON r.match_id = t.match_id
                WHERE t.created_ts >= %s
                  AND t.suggestion <> 'HARVEST'
                  AND t.sent_ok = 1
                ORDER BY t.created_ts ASC
                """,
                (cutoff_ts,),
            ).fetchall()

        if not rows:
            msg = f"📊 Accuracy Digest\nNo tips in the last {window_days}d."
            send_telegram(msg)
            return msg

        def _grade(suggestion: str, gh: Optional[int], ga: Optional[int], btts: Optional[int]) -> Optional[int]:
            if gh is None or ga is None or btts is None:
                return None
            return _tip_outcome_for_result(suggestion, {
                "final_goals_h": int(gh or 0),
                "final_goals_a": int(ga or 0),
                "btts_yes": int(btts or 0)
            })

        total_sent = len(rows)
        graded = wins = 0

        by_market: Dict[str, Dict[str, float]] = {}
        for (mkt, sugg, prob, odds, cts, gh, ga, btts) in rows:
            outcome = _grade(sugg, gh, ga, btts)
            if outcome is None:
                continue
            graded += 1
            if int(outcome) == 1:
                wins += 1
            m = by_market.setdefault(mkt or "?", {"graded": 0, "wins": 0, "stake": 0.0, "pnl": 0.0})
            m["graded"] += 1
            if int(outcome) == 1:
                m["wins"] += 1
            try:
                o = float(odds) if odds is not None else None
                if o is not None and 1.01 <= o <= MAX_ODDS_ALL:
                    m["stake"] += 1.0
                    m["pnl"] += (o - 1.0) if int(outcome) == 1 else -1.0
            except Exception:
                pass

        if graded == 0:
            msg = f"📊 Accuracy Digest\nNo graded tips in the last {window_days}d."
            send_telegram(msg)
            return msg

        acc = 100.0 * wins / max(1, graded)

        header = [
            f"📊 <b>Accuracy Digest</b> (last {window_days}d)",
            f"Tips sent: {total_sent}  •  Graded: {graded}  •  Wins: {wins}  •  Accuracy: {acc:.1f}%"
        ]

        body: List[str] = []
        for mk in sorted(by_market.keys()):
            st = by_market[mk]
            if st["graded"] <= 0:
                continue
            a = 100.0 * st["wins"] / max(1, st["graded"])
            roi_txt = ""
            if st["stake"] > 0:
                roi_val = 100.0 * st["pnl"] / st["stake"]
                roi_txt = f" • ROI {roi_val:+.1f}%"
            body.append(f"• {escape(mk)} — {int(st['wins'])}/{int(st['graded'])} ({a:.1f}%){roi_txt}")

        msg = "\n".join(header + body)
        send_telegram(msg)
        return msg

    except Exception as e:
        log.exception("[DIGEST] failed: %s", e)
        try:
            send_telegram(f"📊 Accuracy Digest failed: {escape(str(e))}")
        except Exception:
            pass
        return None

def _get_market_threshold_key(m: str) -> str: return f"conf_threshold:{m}"
def _get_market_threshold(m: str) -> float:
    try:
        v=get_setting_cached(_get_market_threshold_key(m)); return float(v) if v is not None else float(CONF_THRESHOLD)
    except: return float(CONF_THRESHOLD)
def _get_market_threshold_pre(m: str) -> float: return _get_market_threshold(f"PRE {m}")

def _as_bool(s: str) -> bool:
    return str(s).strip() not in ("0","false","False","no","NO","")

DAILY_ACCURACY_DIGEST_ENABLE = _as_bool(os.getenv("DAILY_ACCURACY_DIGEST_ENABLE", "1"))

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

# ───────── Scan (in-play) ─────────
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
    sh_tot_a = g(sa, ("total shots", "shots_clipboard

Here’s **`main.py` (chunk 2/2)** — continuation and completion with the new `_run_with_pg_lock` and hardened boot. This picks up exactly where the previous chunk ended.

```python
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
    home = m["teams"]["home"]["name"]
    away = m["teams"]["away"]["name"]
    gh = m["goals"]["home"] or 0
    ga = m["goals"]["away"] or 0
    minute = int(((m.get("fixture") or {}).get("status") or {}).get("elapsed") or 0)

    stats = {}
    for s in (m.get("statistics") or []):
        t = (s.get("team") or {}).get("name")
        if t:
            stats[t] = { (i.get("type") or ""): i.get("value") for i in (s.get("statistics") or []) }

    sh = stats.get(home, {}) or {}
    sa = stats.get(away, {}) or {}

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

# ---- Optional: calibration from graded tips (kept but hardened) ----
def calibrate_and_retune_from_tips(conn, target_precision: float,
                                   min_preds: int, min_thr_pct: float, max_thr_pct: float,
                                   days: int = 365) -> Dict[str, float]:
    try:
        df = load_graded_tips(conn, days=days)  # from train_models
    except Exception as e:
        log.info("Tips calibration skipped (helpers not available): %s", e)
        return {}

    if df.empty:
        log.info("Tips calibration: no graded tips found.")
        return {}

    updates: Dict[str, float] = {}
    def _do(market_name: str, mask_market):
        sub = df[mask_market].copy()
        if len(sub) < max(200, min_preds*3):  # need some mass
            return
        # Platt using tips
        p_raw = sub["prob"].to_numpy()
        y = sub["y"].to_numpy().astype(int)
        a, b = fit_platt(y, p_raw)
        # store calibration blob update for the model key(s)
        key_map = {
            "BTTS":           "BTTS_YES",
            "Over/Under 2.5": "OU_2.5",
            "Over/Under 3.5": "OU_3.5",
            "1X2":            None,  # handled by threshold only (composite)
        }
        model_key = key_map.get(market_name)
        if model_key:
            blob = _get_setting_json(conn, f"model_latest:{model_key}")
            if blob:
                blob["calibration"] = {"method": "platt", "a": float(a), "b": float(b)}
                for k in (f"model_latest:{model_key}", f"model:{model_key}"):
                    _set_setting(conn, k, json.dumps(blob))

        # choose threshold to hit target precision
        z = _logit_vec(p_raw); p_cal = 1.0/(1.0+np.exp(-(a*z + b)))
        thr_prob = _pick_threshold_for_target_precision(
            y_true=y, p_cal=p_cal, target_precision=target_precision,
            min_preds=min_preds, default_threshold=0.65,
        )
        thr_pct = float(np.clip(_percent(thr_prob), min_thr_pct, max_thr_pct))
        _set_setting(conn, f"conf_threshold:{market_name}", f"{thr_pct:.2f}")
        updates[market_name] = thr_pct

    _do("BTTS",                 df["market"] == "BTTS")
    _do("Over/Under 2.5",      df["market"].eq("Over/Under 2.5"))
    _do("Over/Under 3.5",      df["market"].eq("Over/Under 3.5"))
    _do("1X2",                 df["market"] == "1X2")

    if updates:
        log.info("Tips calibration/threshold updates: %s", updates)
    return updates

# ───────── Advisory lock runner (hardened) ─────────
def _run_with_pg_lock(lock_key: int, fn, *a, **k):
    """
    Run `fn` under a PG advisory lock. Recovers from transient pool/conn errors.
    """
    tries = max(1, PG_LOCK_MAX_RETRIES) + 1
    sleep = float(os.getenv("PG_LOCK_RETRY_SLEEP", "0.8"))

    for attempt in range(tries):
        try:
            with db_conn() as c:
                got = c.execute("SELECT pg_try_advisory_lock(%s)", (lock_key,)).fetchone()[0]
                if not got:
                    log.info("[LOCK %s] busy; skipped.", lock_key)
                    return None
                try:
                    return fn(*a, **k)
                finally:
                    try:
                        c.execute("SELECT pg_advisory_unlock(%s)", (lock_key,))
                    except Exception as e:
                        log.warning("[LOCK %s] unlock failed (ignored): %s", lock_key, e)
        except (OperationalError, InterfaceError, _pgpool.PoolError) as e:
            log.warning("[LOCK %s] DB connection error (attempt %d/%d): %s", lock_key, attempt+1, tries, e)
            _reset_pool()
            time.sleep(sleep)
            continue
        except Exception as e:
            log.exception("[LOCK %s] failed: %s", lock_key, e)
            return None
    log.error("[LOCK %s] giving up after %d attempts", lock_key, tries)
    return None

# ───────── In-play scanning loop ─────────
def scan_and_tip():
    try:
        js = _api_get(FOOTBALL_API_URL, {"live": "all"})
        matches = js.get("response", []) if isinstance(js, dict) else []
    except Exception as e:
        log.error("[SCAN] fetch failed: %s", e)
        return

    for m in matches:
        fid = int((m.get("fixture") or {}).get("id") or 0)
        if not fid: continue
        lg_id, lg_name = _league_name(m)
        if _blocked_league({"id": lg_id, "name": lg_name}): continue

        feat = extract_features(m)
        if not stats_coverage_ok(feat, int(feat.get("minute", 0))): continue

        # Models & thresholds
        mdl_btts = load_model_from_settings("BTTS_YES")
        mdl_ou25 = load_model_from_settings("OU_2.5")
        mdl_ou35 = load_model_from_settings("OU_3.5")
        mdl_home, mdl_draw, mdl_away = _load_wld_models()

        # Example: BTTS Yes
        if mdl_btts:
            p_model = _score_prob(feat, mdl_btts)
            p_book = _implied_prob_from_odds(fetch_odds(fid).get("BTTS", {}).get("Yes", {}).get("odds"))
            p_final = _apply_stack_if_available(p_model, mdl_btts, p_book)

            thr = _get_market_threshold("BTTS")
            if p_final * 100 >= thr and _candidate_is_sane("BTTS: Yes", feat):
                ok, odds, book, _ = _price_gate("BTTS", "BTTS: Yes", fid)
                if ok:
                    ev = _ev(p_final, odds or 0.0) if odds else None
                    if ev is None or ev*100 >= EDGE_MIN_BPS:
                        msg = _format_tip_message(
                            * _teams(m), lg_name, feat["minute"], _pretty_score(m),
                            "BTTS: Yes", p_final*100, feat, odds, book, ev*100 if ev else None
                        )
                        send_telegram(msg)
                        save_snapshot_from_match(m, feat)

# ───────── Boot / FastAPI app ─────────
app = FastAPI()

@app.get("/health")
def health(): return {"ok": True}

@app.get("/scan")
def scan_endpoint():
    _run_with_pg_lock(1001, scan_and_tip)
    return {"ok": True}

@app.get("/digest")
def digest_endpoint():
    msg = _run_with_pg_lock(1002, daily_accuracy_digest)
    return {"ok": True, "msg": msg}

def _ensure_schema():
    with db_conn() as c:
        c.execute("CREATE TABLE IF NOT EXISTS tips (match_id BIGINT, created_ts BIGINT, market TEXT, suggestion TEXT, confidence REAL, confidence_raw REAL, league_id BIGINT, league TEXT, home TEXT, away TEXT, score_at_tip TEXT, minute INT, sent_ok INT, odds REAL)")
        c.execute("CREATE TABLE IF NOT EXISTS tip_snapshots (match_id BIGINT, created_ts BIGINT, payload TEXT, PRIMARY KEY(match_id, created_ts))")
        c.execute("CREATE TABLE IF NOT EXISTS match_results (match_id BIGINT PRIMARY KEY, final_goals_h INT, final_goals_a INT, btts_yes INT, updated_ts BIGINT)")
        c.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)")

def _on_boot():
    log.info("[BOOT] initializing...")
    _reset_pool()
    _init_pool()
    _ensure_schema()
    if DAILY_ACCURACY_DIGEST_ENABLE:
        _run_with_pg_lock(1002, daily_accuracy_digest)
    log.info("[BOOT] ready.")

_on_boot()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
