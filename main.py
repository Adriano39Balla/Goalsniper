#!/usr/bin/env python3
"""
goalsniper — single-server, fully AI-powered betting predictions backend
Markets: Over/Under (OU), BTTS, 1X2 (draw suppressed)
Core: Logistic regression + Naive Bayesian evidence overlay + probability calibration
Data: API-Football (fixtures, stats, events, odds), Supabase Postgres (psycopg2)
Ops: Flask API + APScheduler jobs, Redis-backed caches (optional), EV/odds gate, Telegram notifications
Manual controls: scan, training, digest, auto-tune, backfill, health
"""
import os, json, time, logging, statistics
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timedelta
from collections import defaultdict

import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

import numpy as np

from flask import Flask, jsonify, request, abort
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from psycopg2.pool import SimpleConnectionPool
import psycopg2

try:
    from zoneinfo import ZoneInfo
except Exception:
    from backports.zoneinfo import ZoneInfo  # type: ignore

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from pydantic_settings import BaseSettings, SettingsConfigDict


# ──────────────────────────────────────────────────────────────────────────────
# Settings
# ──────────────────────────────────────────────────────────────────────────────

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False)

    # API-Football
    API_FOOTBALL_KEY: str = ""
    API_FOOTBALL_URL: str = "https://v3.football.api-sports.io"

    # Core toggles
    RUN_SCHEDULER: bool = True
    HARVEST_MODE: bool = True              # save live snapshots for training
    TRAIN_ENABLE: bool = True              # allow scheduled training
    AUTO_TUNE_ENABLE: bool = True

    # Odds/EV
    MIN_ODDS_OU: float = 1.50
    MIN_ODDS_BTTS: float = 1.50
    MIN_ODDS_1X2: float = 1.50
    MAX_ODDS_ALL: float = 20.0
    EDGE_MIN_BPS: int = 600                # +6.00% edge minimum

    # Markets, thresholds (percent)
    OU_LINES: str = "2.5,3.5"
    CONF_THRESHOLD: float = 75.0           # default if per-market not set (percent)
    PRE_CONF_THRESHOLD: float = 75.0

    # Timings
    SCAN_INTERVAL_SEC: int = 300           # live scan cadence
    NEG_TTL_SEC: int = 45
    SETTINGS_TTL_SEC: int = 60
    MODELS_CACHE_TTL_SEC: int = 120
    STATS_CACHE_TTL_SEC: int = 90
    ODDS_CACHE_TTL_SEC: int = 120
    STALE_STATS_MAX_SEC: int = 240
    TIP_MIN_MINUTE: int = 12               # don't tip too early
    TOTAL_MATCH_MINUTES: int = 95

    # Prematch / digest / training
    DAILY_ACCURACY_DIGEST_ENABLE: bool = True
    DAILY_ACCURACY_HOUR: int = 3
    DAILY_ACCURACY_MINUTE: int = 6
    TRAIN_HOUR_UTC: int = 2
    TRAIN_MINUTE_UTC: int = 12

    BACKFILL_EVERY_MIN: int = 15
    BACKFILL_DAYS: int = 14

    # Telegram
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""

    # Admin
    ADMIN_API_KEY: Optional[str] = None
    TELEGRAM_WEBHOOK_SECRET: Optional[str] = None

    # DB (Supabase Postgres)
    DATABASE_URL: str = ""

    # Optional integrations
    SENTRY_DSN: Optional[str] = None
    REDIS_URL: Optional[str] = None

    # Misc
    ODDS_SOURCE: str = "auto"              # auto|live|prematch
    ODDS_AGGREGATION: str = "best"         # ← aggressive default per your request (best|median)
    ODDS_OUTLIER_MULT: float = 1.8
    ODDS_REQUIRE_N_BOOKS: int = 2
    ODDS_FAIR_MAX_MULT: float = 2.5
    PER_LEAGUE_CAP: int = 2
    PREDICTIONS_PER_MATCH: int = 1
    MAX_TIPS_PER_SCAN: int = 25

    APPLY_TUNE_PREC_TOL: float = 0.03
    TARGET_PRECISION: float = 0.60
    THRESH_MIN_PREDICTIONS: int = 25
    MIN_THRESH: float = 55.0
    MAX_THRESH: float = 85.0

settings = Settings()

# ──────────────────────────────────────────────────────────────────────────────
# App / logging / metrics
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
log = logging.getLogger("goalsniper")
app = Flask(__name__)

METRICS = {
    "api_calls_total": defaultdict(int),
    "api_rate_limited_total": 0,
    "tips_generated_total": 0,
    "tips_sent_total": 0,
    "db_errors_total": 0,
    "job_duration_seconds": defaultdict(list),
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

# ───────── Optional import: trainer ─────────
try:
    import train_models as _tm        
    train_models = _tm.train_models   
except Exception as e:
    _IMPORT_ERR = repr(e)
    def train_models(*args, **kwargs):  # type: ignore
        log.warning("train_models not available: %s", _IMPORT_ERR)
        return {"ok": False, "reason": f"train_models import failed: {_IMPORT_ERR}"}

# ──────────────────────────────────────────────────────────────────────────────
# Optional add-ons
# ──────────────────────────────────────────────────────────────────────────────
if settings.SENTRY_DSN:
    try:
        import sentry_sdk  # type: ignore
        sentry_sdk.init(dsn=settings.SENTRY_DSN, traces_sample_rate=float(os.getenv("SENTRY_TRACES", "0.0")))
    except Exception:
        pass

_redis = None
if settings.REDIS_URL:
    try:
        import redis  # type: ignore
        _redis = redis.Redis.from_url(settings.REDIS_URL, socket_timeout=1, socket_connect_timeout=1)
    except Exception:
        _redis = None

# ──────────────────────────────────────────────────────────────────────────────
# HTTP client with retries
# ──────────────────────────────────────────────────────────────────────────────
session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], respect_retry_after_header=True)))

HEADERS = {"x-apisports-key": settings.API_FOOTBALL_KEY, "Accept": "application/json"}
BASE_URL = settings.API_FOOTBALL_URL
FOOTBALL_API_URL = f"{BASE_URL}/fixtures"
TZ_UTC = ZoneInfo("UTC")
BERLIN_TZ = ZoneInfo("Europe/Berlin")

INPLAY_STATUSES = {"1H", "HT", "2H", "ET", "BT", "P"}

# ──────────────────────────────────────────────────────────────────────────────
# DB Pool
# ──────────────────────────────────────────────────────────────────────────────
POOL: Optional[SimpleConnectionPool] = None

class PooledConn:
    def __init__(self, pool): self.pool = pool; self.conn=None; self.cur=None
    def __enter__(self):
        self.conn = self.pool.getconn(); self.conn.autocommit=True; self.cur=self.conn.cursor(); return self
    def __exit__(self, a,b,c):
        try: self.cur and self.cur.close()
        finally: self.conn and self.pool.putconn(self.conn)
    def execute(self, sql: str, params=()):
        try:
            self.cur.execute(sql, params or ())
            return self.cur
        except Exception as e:
            _metric_inc("db_errors_total", n=1)
            log.error("DB execute failed: %s\nSQL: %s\nParams: %s", e, sql, params)
            raise

def _init_pool():
    global POOL
    dsn = settings.DATABASE_URL
    if "sslmode=" not in dsn:
        dsn += ("&" if "?" in dsn else "?") + "sslmode=require"
    POOL = SimpleConnectionPool(minconn=1, maxconn=int(os.getenv("DB_POOL_MAX","5")), dsn=dsn)

def db_conn():
    if not POOL: _init_pool()
    return PooledConn(POOL)  # type: ignore

# ──────────────────────────────────────────────────────────────────────────────
# Simple caches
# ──────────────────────────────────────────────────────────────────────────────
class _KVCache:
    def __init__(self, ttl): self.ttl=ttl; self.data={}
    def get(self, k):
        if _redis:
            try:
                v = _redis.get(f"gs:{k}")
                return v.decode("utf-8") if v is not None else None
            except Exception: pass
        v = self.data.get(k)
        if not v: return None
        ts, val = v
        if time.time() - ts > self.ttl:
            self.data.pop(k, None); return None
        return val
    def set(self, k, v):
        if _redis:
            try:
                _redis.setex(f"gs:{k}", self.ttl, v if v is not None else "")
                return
            except Exception: pass
        self.data[k] = (time.time(), v)
    def invalidate(self, k=None):
        if _redis and k:
            try:
                _redis.delete(f"gs:{k}"); return
            except Exception: pass
        self.data.clear() if k is None else self.data.pop(k, None)

SETTINGS_CACHE, MODELS_CACHE, ODDS_CACHE, STATS_CACHE, EVENTS_CACHE = (_KVCache(settings.SETTINGS_TTL_SEC),
                                                                       _KVCache(settings.MODELS_CACHE_TTL_SEC),
                                                                       _KVCache(settings.ODDS_CACHE_TTL_SEC),
                                                                       _KVCache(settings.STATS_CACHE_TTL_SEC),
                                                                       _KVCache(settings.STATS_CACHE_TTL_SEC))

NEG_CACHE: Dict[Tuple[str, int], Tuple[float, bool]] = {}
API_CB = {"failures": 0, "opened_until": 0.0}
API_CB_THRESHOLD = int(os.getenv("API_CB_THRESHOLD", "8"))
API_CB_COOLDOWN_SEC = int(os.getenv("API_CB_COOLDOWN_SEC", "90"))
REQ_TIMEOUT_SEC = float(os.getenv("REQ_TIMEOUT_SEC", "8.0"))

# ──────────────────────────────────────────────────────────────────────────────
# DB schema & settings helpers
# ──────────────────────────────────────────────────────────────────────────────
def init_db():
    with db_conn() as c:
        # 1) Create tables if missing (no PK assumptions here)
        c.execute("""CREATE TABLE IF NOT EXISTS tips (
            match_id BIGINT,
            league_id BIGINT,
            league TEXT,
            home TEXT, away TEXT,
            market TEXT, suggestion TEXT,
            confidence DOUBLE PRECISION,
            confidence_raw DOUBLE PRECISION,
            score_at_tip TEXT,
            minute INTEGER,
            created_ts BIGINT,
            odds DOUBLE PRECISION, book TEXT, ev_pct DOUBLE PRECISION,
            sent_ok INTEGER DEFAULT 1
        )""")

        c.execute("""CREATE TABLE IF NOT EXISTS tip_snapshots (
            match_id BIGINT, created_ts BIGINT, payload TEXT,
            PRIMARY KEY (match_id, created_ts)
        )""")

        c.execute("""CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY, value TEXT
        )""")

        c.execute("""CREATE TABLE IF NOT EXISTS match_results (
            match_id BIGINT PRIMARY KEY,
            final_goals_h INTEGER, final_goals_a INTEGER,
            btts_yes INTEGER, updated_ts BIGINT
        )""")

        c.execute("""CREATE TABLE IF NOT EXISTS odds_history (
            match_id BIGINT,
            captured_ts BIGINT,
            market TEXT,
            selection TEXT,
            odds DOUBLE PRECISION,
            book TEXT,
            PRIMARY KEY (match_id, market, selection, captured_ts)
        )""")

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

        # 2) Column introspection for 'tips'
        cols = set(
            x[0] for x in c.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name='tips'"
            ).fetchall()
        )

        def _add(coldef: str, colname: str):
            if colname not in cols:
                c.execute(f"ALTER TABLE tips ADD COLUMN IF NOT EXISTS {coldef}")
                cols.add(colname)

        # Ensure ALL expected columns exist (covers very old schemas)
        _add("match_id BIGINT", "match_id")
        _add("league_id BIGINT", "league_id")
        _add("league TEXT", "league")
        _add("home TEXT", "home")
        _add("away TEXT", "away")
        _add("market TEXT", "market")
        _add("suggestion TEXT", "suggestion")
        _add("confidence DOUBLE PRECISION", "confidence")
        _add("confidence_raw DOUBLE PRECISION", "confidence_raw")
        _add("score_at_tip TEXT", "score_at_tip")
        _add("minute INTEGER", "minute")
        _add("created_ts BIGINT", "created_ts")
        _add("odds DOUBLE PRECISION", "odds")
        _add("book TEXT", "book")
        _add("ev_pct DOUBLE PRECISION", "ev_pct")
        _add("sent_ok INTEGER", "sent_ok")

        # 3) Indexes/constraints — only if the required columns exist
        def _has(*need): return all(n in cols for n in need)

        if _has("match_id", "created_ts"):
            c.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_tips_match_created ON tips (match_id, created_ts)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_tips_created ON tips (created_ts DESC)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_tips_match ON tips (match_id)")
        if _has("sent_ok", "created_ts"):
            c.execute("CREATE INDEX IF NOT EXISTS idx_tips_sent ON tips (sent_ok, created_ts DESC)")

        # Other table indexes (these don't depend on old columns)
        c.execute("CREATE INDEX IF NOT EXISTS idx_snap_by_match ON tip_snapshots (match_id, created_ts DESC)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_results_updated ON match_results (updated_ts DESC)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_odds_hist_match ON odds_history (match_id, captured_ts DESC)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_odds_hist_market ON odds_history (market, captured_ts DESC)")

def get_setting(key: str) -> Optional[str]:
    with db_conn() as c:
        r = c.execute("SELECT value FROM settings WHERE key=%s", (key,)).fetchone()
        return r[0] if r else None

def set_setting(key: str, value: str) -> None:
    with db_conn() as c:
        c.execute("INSERT INTO settings(key,value) VALUES(%s,%s) ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value", (key, value))

def get_setting_cached(key: str) -> Optional[str]:
    v = SETTINGS_CACHE.get(key)
    if v is None:
        v = get_setting(key)
        SETTINGS_CACHE.set(key, v)
    return v

def invalidate_model_caches_for_key(key: str):
    if key.lower().startswith(("model", "model_latest", "model_v2", "pre_")):
        MODELS_CACHE.invalidate(key)

# ──────────────────────────────────────────────────────────────────────────────
# Telegram
# ──────────────────────────────────────────────────────────────────────────────
def send_telegram(text: str) -> bool:
    if not settings.TELEGRAM_BOT_TOKEN or not settings.TELEGRAM_CHAT_ID:
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/sendMessage",
            data={"chat_id": settings.TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True},
            timeout=REQ_TIMEOUT_SEC
        )
        ok = bool(r.ok)
        if ok:
            _metric_inc("tips_sent_total", n=1)
        return ok
    except Exception:
        return False

# ──────────────────────────────────────────────────────────────────────────────
# API-Football client with CB + negative cache
# ──────────────────────────────────────────────────────────────────────────────
def _api_get(url: str, params: dict, timeout: int = 15):
    now = time.time()
    if API_CB["opened_until"] > now:
        return None
    lbl = "unknown"
    try:
        p = str(url)
        if "/odds" in p: lbl = "odds"
        elif "/statistics" in p: lbl = "statistics"
        elif "/events" in p: lbl = "events"
        elif "/fixtures" in p: lbl = "fixtures"
    except Exception:
        pass
    try:
        r = session.get(url, headers=HEADERS, params=params, timeout=min(timeout, REQ_TIMEOUT_SEC))
        _metric_inc("api_calls_total", label=lbl, n=1)
        if r.status_code == 429:
            METRICS["api_rate_limited_total"] += 1
            API_CB["failures"] += 1
        elif r.status_code >= 500:
            API_CB["failures"] += 1
        else:
            API_CB["failures"] = 0
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

# ──────────────────────────────────────────────────────────────────────────────
# Feature extraction helpers
# ──────────────────────────────────────────────────────────────────────────────
def _num(v) -> float:
    try:
        if isinstance(v, str) and v.endswith("%"): return float(v[:-1])
        return float(v or 0)
    except: return 0.0

def _pos_pct(v) -> float:
    try: return float(str(v).replace("%","").strip() or 0)
    except: return 0.0

def _fmt_line(line: float) -> str: return f"{line}".rstrip("0").rstrip(".")

def _blocked_league(league_obj: dict) -> bool:
    name=str((league_obj or {}).get("name","")).lower()
    country=str((league_obj or {}).get("country","")).lower()
    typ=str((league_obj or {}).get("type","")).lower()
    txt=f"{country} {name} {typ}"
    patterns = ["u17","u18","u19","u20","u21","u23","youth","junior","reserve","res.","friendlies","friendly"]
    if any(p in txt for p in patterns): return True
    deny=[x.strip() for x in os.getenv("LEAGUE_DENY_IDS","").split(",") if x.strip()]
    lid=str((league_obj or {}).get("id") or "")
    if lid in deny: return True
    return False

def fetch_match_stats(fid: int) -> list:
    now=time.time(); k=("stats",fid)
    ts_empty = NEG_CACHE.get(k, (0.0, False))
    if ts_empty[1] and (now - ts_empty[0] < settings.NEG_TTL_SEC): return []
    v = STATS_CACHE.get(fid)
    if v is not None:
        try_ts = json.loads(v)[0]
        if now - try_ts < settings.STATS_CACHE_TTL_SEC: return json.loads(v)[1]
    js=_api_get(f"{FOOTBALL_API_URL}/statistics", {"fixture": fid}) or {}
    out=js.get("response",[]) if isinstance(js,dict) else []
    STATS_CACHE.set(fid, json.dumps([now, out]))
    if not out: NEG_CACHE[k]=(now, True)
    return out

def fetch_match_events(fid: int) -> list:
    now=time.time(); k=("events",fid)
    ts_empty = NEG_CACHE.get(k, (0.0, False))
    if ts_empty[1] and (now - ts_empty[0] < settings.NEG_TTL_SEC): return []
    v = EVENTS_CACHE.get(fid)
    if v is not None:
        try_ts = json.loads(v)[0]
        if now - try_ts < settings.STATS_CACHE_TTL_SEC: return json.loads(v)[1]
    js=_api_get(f"{FOOTBALL_API_URL}/events", {"fixture": fid}) or {}
    out=js.get("response",[]) if isinstance(js,dict) else []
    EVENTS_CACHE.set(fid, json.dumps([now, out]))
    if not out: NEG_CACHE[k]=(now, True)
    return out

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

    xg_h = _num(sh.get("Expected Goals", 0));        xg_a = _num(sa.get("Expected Goals", 0))
    sot_h = _num(sh.get("Shots on Target", sh.get("Shots on Goal", 0)))
    sot_a = _num(sa.get("Shots on Target", sa.get("Shots on Goal", 0)))
    sh_total_h = _num(sh.get("Total Shots", sh.get("Shots Total", 0)))
    sh_total_a = _num(sa.get("Total Shots", sa.get("Shots Total", 0)))
    cor_h = _num(sh.get("Corner Kicks", 0));         cor_a = _num(sa.get("Corner Kicks", 0))
    pos_h = _pos_pct(sh.get("Ball Possession", 0));  pos_a = _pos_pct(sa.get("Ball Possession", 0))

    # Cards from events
    red_h = red_a = yellow_h = yellow_a = 0
    for ev in (m.get("events") or []):
        if (str(ev.get("type","")).lower() == "card"):
            d = (ev.get("detail") or "").lower()
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
        "sot_h": float(sot_h), "sot_a": float(sot_a), "sot_sum": float(sot_h + sot_a),
        "sh_total_h": float(sh_total_h), "sh_total_a": float(sh_total_a),
        "cor_h": float(cor_h), "cor_a": float(cor_a), "cor_sum": float(cor_h + cor_a),
        "pos_h": float(pos_h), "pos_a": float(pos_a), "pos_diff": float(pos_h - pos_a),
        "red_h": float(red_h), "red_a": float(red_a), "red_sum": float(red_h + red_a),
        "yellow_h": float(yellow_h), "yellow_a": float(yellow_a)
    }

# ──────────────────────────────────────────────────────────────────────────────
# Model loading (weights stored as JSON in settings table)
# ──────────────────────────────────────────────────────────────────────────────
EPS=1e-12

def _sigmoid(x: float) -> float:
    try:
        if x<-50: return 1e-22
        if x>50:  return 1-1e-22
        return 1/(1+np.exp(-x))
    except Exception: return 0.5

def _linpred(feat: Dict[str,float], weights: Dict[str,float], intercept: float) -> float:
    s=float(intercept or 0.0)
    for k,w in (weights or {}).items(): s += float(w or 0.0)*float(feat.get(k,0.0))
    return s

def _calibrate(p: float, cal: Dict[str,Any]) -> float:
    method=(cal or {}).get("method","sigmoid"); a=float((cal or {}).get("a",1.0)); b=float((cal or {}).get("b",0.0))
    if method.lower() in ("platt","sigmoid"):
        import math; p=max(EPS,min(1-EPS,float(p))); z=math.log(p/(1-p)); return _sigmoid(a*z+b)
    return max(0.0, min(1.0, float(p)))

def _apply_naive_bayes_overlay(p_prior: float, feat: Dict[str, float], bayes: Dict[str, Any]) -> float:
    """
    Naive Bayes overlay:
    bayes = { "features": { "sot_sum": {"bins":[0,2,5,10,99], "lr":[0.7, 1.0, 1.3, 1.6] }, ... } }
    posterior_odds = prior_odds * Π LR_i(bin_i)
    """
    prior_odds = p_prior / max(EPS, 1 - p_prior)
    feats = bayes.get("features", {}) or {}
    odds = prior_odds
    for name, spec in feats.items():
        v = float(feat.get(name, 0.0))
        bins = list(spec.get("bins", []))
        lrs = list(spec.get("lr", []))
        if not bins or not lrs or len(lrs) != len(bins) - 1:
            continue
        idx = None
        for i in range(len(bins)-1):
            if bins[i] <= v < bins[i+1]:
                idx = i
                break
        if idx is None:
            idx = len(lrs) - 1
        lr = float(lrs[idx])
        odds *= max(0.01, min(100.0, lr))
    post = odds / (1 + odds)
    return max(0.0, min(1.0, post))

def _score_prob(feat: Dict[str,float], mdl: Dict[str,Any]) -> float:
    p=_sigmoid(_linpred(feat, mdl.get("weights",{}), float(mdl.get("intercept",0.0))))
    bayes=mdl.get("bayes")
    if isinstance(bayes, dict):
        try:
            p = _apply_naive_bayes_overlay(p, feat, bayes)
        except Exception:
            pass
    cal=mdl.get("calibration") or {}
    try:
        if cal: p=_calibrate(p, cal)
    except Exception:
        pass
    return max(0.0, min(1.0, float(p)))

def load_model_from_settings(name: str) -> Optional[Dict[str, Any]]:
    v = MODELS_CACHE.get(name)
    if v is not None:
        try:
            return json.loads(v)
        except Exception:
            MODELS_CACHE.invalidate(name)
    raw = get_setting_cached(name)
    if not raw:
        return None
    try:
        js = json.loads(raw)
        MODELS_CACHE.set(name, raw)
        return js
    except Exception:
        return None

def _load_ou_model_for_line(line: float) -> Optional[Dict[str,Any]]:
    name=f"OU_{_fmt_line(line)}"; mdl=load_model_from_settings(name)
    return mdl or (load_model_from_settings("O25") if abs(line-2.5)<1e-6 else None)

def _load_wld_models():
    return (load_model_from_settings("WLD_HOME"),
            load_model_from_settings("WLD_DRAW"),
            load_model_from_settings("WLD_AWAY"))

def _load_training_rows_from_snapshots(days: int = 365, per_match: str = "latest") -> List[Tuple[dict, str, int, Optional[float]]]:
    """
    Build training rows directly from tip_snapshots + match_results (no tips/odds needed).
    Returns tuples shaped like the tips loader: (feat_dict, 'Suggestion', label(0/1), None)
    We generate:
      - 'Over 2.5'  (label = 1 if final goals > 2.5)
      - 'Over 3.5'  (same logic)
      - 'BTTS: Yes' (label = 1 if btts_yes == 1)
      - 'Home Win'  (label = 1 if home final > away final)
      - 'Away Win'  (label = 1 if away final > home final)
    """
    cutoff = int(time.time()) - days * 24 * 3600
    with db_conn() as c:
        if per_match == "latest":
            q = """
                SELECT s.match_id, MAX(s.created_ts) AS created_ts, MIN(s.payload) AS payload
                FROM tip_snapshots s
                WHERE s.created_ts >= %s
                GROUP BY s.match_id
            """
        else:  # "all" (can be huge)
            q = "SELECT s.match_id, s.created_ts, s.payload FROM tip_snapshots s WHERE s.created_ts >= %s"
        rows = c.execute(q, (cutoff,)).fetchall()

        # join in bulk to reduce API calls
        mids = [int(r[0]) for r in rows]
        have = set()
        if mids:
            fmt = ",".join(["%s"] * len(mids))
            got = c.execute(f"SELECT match_id, final_goals_h, final_goals_a, btts_yes FROM match_results WHERE match_id IN ({fmt})", tuple(mids)).fetchall()
            for (mid, gh, ga, btts) in got:
                have.add(int(mid))
            # optional: silently skip those without results yet

    out: List[Tuple[dict, str, int, Optional[float]]] = []
    for (mid, cts, payload) in rows:
        if int(mid) not in have:
            continue
        try:
            snap = json.loads(payload or "{}")
            feat = snap.get("stat") or snap  # support both {stat:{...}} and flat payloads
            # pull results (a second query per row here is fine; you can cache if needed)
            with db_conn() as c2:
                gh, ga, btts = c2.execute(
                    "SELECT final_goals_h, final_goals_a, btts_yes FROM match_results WHERE match_id=%s",
                    (int(mid),)
                ).fetchone()
            total = int(gh or 0) + int(ga or 0)

            # OU labels
            out.append((feat, "Over 2.5 Goals", 1 if total > 2.5 else 0, None))
            out.append((feat, "Over 3.5 Goals", 1 if total > 3.5 else 0, None))

            # BTTS
            out.append((feat, "BTTS: Yes", 1 if int(btts or 0) == 1 else 0, None))

            # 1X2 (draw suppressed) — provide both streams so home/away models each learn
            out.append((feat, "Home Win", 1 if int(gh or 0) > int(ga or 0) else 0, None))
            out.append((feat, "Away Win", 1 if int(ga or 0) > int(gh or 0) else 0, None))
        except Exception:
            continue
    return out

# ──────────────────────────────────────────────────────────────────────────────
# Odds aggregation (+ detailed reasons)
# ──────────────────────────────────────────────────────────────────────────────
def _market_name_normalize(s: str) -> str:
    s=(s or "").lower()
    if "both teams" in s or "btts" in s: return "BTTS"
    if "match winner" in s or "winner" in s or "1x2" in s: return "1X2"
    if "over/under" in s or "total" in s or "goals" in s: return "OU"
    return s

def _aggregate_price(vals: List[Tuple[float, str]], prob_hint: Optional[float]) -> Tuple[Optional[float], Optional[str], Dict[str,Any]]:
    meta: Dict[str,Any] = {"books": len({b for (_, b) in vals}), "outlier_mult": settings.ODDS_OUTLIER_MULT, "fair_cap_mult": settings.ODDS_FAIR_MAX_MULT}
    if not vals: 
        meta["reason"] = "no offers"
        return None, None, meta
    xs = sorted([o for (o, _) in vals if (o or 0) > 0])
    if not xs: 
        meta["reason"] = "non-positive offers"
        return None, None, meta
    med = statistics.median(xs)
    cleaned = [(o, b) for (o, b) in vals if o <= med * max(1.0, settings.ODDS_OUTLIER_MULT)]
    if not cleaned:
        meta["reason"] = "all offers filtered as outliers"
        cleaned = vals
    xs2 = sorted([o for (o, _) in cleaned])
    med2 = statistics.median(xs2)
    fair_cap_applied = False
    if prob_hint is not None and prob_hint > 0:
        fair = 1.0 / max(1e-6, float(prob_hint))
        cap = fair * max(1.0, settings.ODDS_FAIR_MAX_MULT)
        before = len(cleaned)
        cleaned = [(o, b) for (o, b) in cleaned if o <= cap] or cleaned
        fair_cap_applied = len(cleaned) < before
    meta["fair_cap_applied"] = bool(fair_cap_applied)
    meta["offers_considered"] = len(cleaned)
    if settings.ODDS_AGGREGATION == "best":
        best = max(cleaned, key=lambda t: t[0])
        return float(best[0]), str(best[1]), meta
    target = med2
    pick = min(cleaned, key=lambda t: abs(t[0] - target))
    return float(pick[0]), f"{pick[1]} (median of {len(xs)})", meta

def fetch_odds(fid: int, prob_hints: Optional[dict] = None) -> dict:
    """
    Returns aggregated odds plus a _debug map with per-side reasons when prices are missing.
    out example:
    {
      "BTTS": {"Yes":{"odds":1.95,"book":"..."}, "No": {...}},
      "1X2": {"Home": {...}, "Away": {...}},
      "OU_2.5": {"Over": {...}, "Under": {...}},
      "_debug": {"BTTS:Yes":{"reason":"insufficient distinct bookmakers","book_count":1,"require":2}, ...}
    }
    """
    v = ODDS_CACHE.get(fid)
    if v is not None:
        try:
            return json.loads(v)
        except Exception:
            ODDS_CACHE.invalidate(fid)

    js = {}
    def _fetch(path: str) -> dict:
        r = _api_get(f"{BASE_URL}/{path}", {"fixture": fid}) or {}
        return r if isinstance(r, dict) else {}
    if settings.ODDS_SOURCE in ("auto","live"):
        js = _fetch("odds/live")
    if not (js.get("response") or []) and settings.ODDS_SOURCE in ("auto","prematch"):
        js = _fetch("odds")

    by_market: dict[str, dict[str, list[tuple[float, str]]]] = {}
    debug: Dict[str, Dict[str, Any]] = {}

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
        for side, lst in side_map.items():
            distinct_books = len({b for (_, b) in lst})
            if distinct_books < max(1, settings.ODDS_REQUIRE_N_BOOKS):
                debug_key = f"{mkey}:{side}"
                debug[debug_key] = {"reason": "insufficient distinct bookmakers", "book_count": distinct_books, "require": settings.ODDS_REQUIRE_N_BOOKS}
                continue

            # compute price
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
            ag, label, meta = _aggregate_price(lst, hint)
            if ag is None:
                debug[f"{mkey}:{side}"] = {"reason": meta.get("reason","aggregation failed"), **meta}
                continue
            out.setdefault(mkey, {})[side] = {"odds": float(ag), "book": label}

    if debug:
        out["_debug"] = debug

    ODDS_CACHE.set(fid, json.dumps(out))
    return out

# ──────────────────────────────────────────────────────────────────────────────
# Gating & scoring helpers
# ──────────────────────────────────────────────────────────────────────────────
def _ev(prob: float, odds: float) -> float:
    return prob * max(0.0, float(odds)) - 1.0

def _min_odds_for_market(market: str) -> float:
    if market.startswith("Over/Under"): return settings.MIN_ODDS_OU
    if market == "BTTS": return settings.MIN_ODDS_BTTS
    if market == "1X2":  return settings.MIN_ODDS_1X2
    return 1.01

def _market_family(market_text: str) -> str:
    s = (market_text or "").upper()
    if s.startswith("OVER/UNDER") or "OVER/UNDER" in s: return "OU"
    if s == "BTTS" or "BTTS" in s: return "BTTS"
    if s == "1X2" or "WINNER" in s or "MATCH WINNER" in s: return "1X2"
    return s

def _parse_ou_line_from_suggestion(s: str) -> Optional[float]:
    try:
        for tok in (s or "").split():
            try: return float(tok)
            except: pass
    except: pass
    return None

def _get_threshold(key: str, default_pct: float) -> float:
    try:
        v = get_setting_cached(key)
        return float(v) if v is not None else float(default_pct)
    except Exception:
        return float(default_pct)

def _format_tip_message(home, away, league, minute, score, suggestion, prob_pct, odds=None, book=None, ev_pct=None):
    money = ""
    if odds:
        if ev_pct is not None:
            money = f"\n💰 <b>Odds:</b> {odds:.2f} @ {book or 'Book'}  •  <b>EV:</b> {ev_pct:+.1f}%"
        else:
            money = f"\n💰 <b>Odds:</b> {odds:.2f} @ {book or 'Book'}"
    return ("⚽️ <b>New Tip!</b>\n"
            f"<b>Match:</b> {home} vs {away}\n"
            f"🕒 <b>Minute:</b> {minute}'  |  <b>Score:</b> {score}\n"
            f"<b>Tip:</b> {suggestion}\n"
            f"📈 <b>Confidence:</b> {prob_pct:.1f}%{money}\n"
            f"🏆 <b>League:</b> {league}")

def _league_name(m: dict) -> Tuple[int,str]:
    lg=(m.get("league") or {}) or {}
    return int(lg.get("id") or 0), f"{lg.get('country','')} - {lg.get('name','')}".strip(" -")

def _teams(m: dict) -> Tuple[str,str]:
    t=(m.get("teams") or {}) or {}
    return (t.get("home",{}).get("name",""), t.get("away",{}).get("name",""))

def _pretty_score(m: dict) -> str:
    gh=(m.get("goals") or {}).get("home") or 0; ga=(m.get("goals") or {}).get("away") or 0
    return f"{gh}-{ga}"

def market_cutoff_ok(minute: int, market_text: str, default_cutoff: Optional[int] = None) -> bool:
    fam = _market_family(market_text)
    cutoff = get_setting_cached(f"market_cutoff:{fam}")
    try:
        cutoff_i = int(float(cutoff)) if cutoff is not None else (default_cutoff if default_cutoff is not None else (settings.TOTAL_MATCH_MINUTES - 5))
    except Exception:
        cutoff_i = settings.TOTAL_MATCH_MINUTES - 5
    try:
        m = int(minute)
    except Exception:
        m = 0
    return m <= cutoff_i

def _odds_debug_reason(odds_map: dict, mk: str, sug: str) -> str:
    dbg = odds_map.get("_debug") or {}
    key = None
    if mk == "BTTS":
        key = "BTTS:Yes" if sug.endswith("Yes") else "BTTS:No"
    elif mk == "1X2":
        key = "1X2:Home" if sug == "Home Win" else ("1X2:Away" if sug == "Away Win" else None)
    elif mk.startswith("Over/Under"):
        ln = _parse_ou_line_from_suggestion(sug)
        if ln is not None:
            side = "Over" if sug.startswith("Over") else "Under"
            key = f"OU_{_fmt_line(ln)}:{side}"
    if key and key in dbg:
        d = dbg[key]
        parts = [str(d.get("reason","missing"))]
        if "book_count" in d: parts.append(f"books={d['book_count']}/{d.get('require', '?')}")
        if "offers_considered" in d: parts.append(f"offers_considered={d['offers_considered']}")
        if "fair_cap_applied" in d: parts.append(f"fair_cap_applied={d['fair_cap_applied']}")
        return "; ".join(parts)
    return "market/side not available from odds endpoint(s) or filtered by requirements"

# ──────────────────────────────────────────────────────────────────────────────
# Core scan
# ──────────────────────────────────────────────────────────────────────────────
ALLOWED_SUGGESTIONS = {"BTTS: Yes", "BTTS: No", "Home Win", "Away Win", "Over", "Under"}

def _allowed_with_ou_lines():
    out=set(ALLOWED_SUGGESTIONS)
    for s in settings.OU_LINES.split(","):
        s=s.strip()
        if not s: continue
        out.add(f"Over {s} Goals"); out.add(f"Under {s} Goals")
    return out

def production_scan() -> Tuple[int, int]:
    # ping DB first
    try:
        with db_conn() as c:
            c.execute("SELECT 1")
    except Exception:
        log.warning("[DB] ping failed, reinit")
        try:
            _init_pool()
        except Exception:
            return (0, 0)

    matches = fetch_live_matches()
    live_seen = len(matches)
    if live_seen == 0:
        log.info("[SCAN] no live")
        return 0, 0

    saved = 0
    now_ts = int(time.time())
    per_league_counter: dict[int, int] = {}
    allowed = _allowed_with_ou_lines()

    with db_conn() as c:
        for m in matches:
            try:
                fid = int((m.get("fixture", {}) or {}).get("id") or 0)
                if not fid: continue

                # dup cooldown per match (20m)
                cutoff = now_ts - 20 * 60
                if c.execute("SELECT 1 FROM tips WHERE match_id=%s AND created_ts>=%s AND suggestion<>'HARVEST' LIMIT 1", (fid, cutoff)).fetchone():
                    continue

                feat = extract_features(m)
                minute = int(feat.get("minute", 0))
                if minute < settings.TIP_MIN_MINUTE: continue

                league_id, league = _league_name(m)
                home, away = _teams(m)
                score = _pretty_score(m)

                candidates: List[Tuple[str, str, float]] = []

                # OU
                for token in settings.OU_LINES.split(","):
                    token = token.strip()
                    if not token: continue
                    try: line = float(token)
                    except Exception: continue
                    mdl = _load_ou_model_for_line(line)
                    if not mdl: continue
                    mk = f"Over/Under {_fmt_line(line)}"
                    thr = _get_threshold(f"conf_threshold:{mk}", settings.CONF_THRESHOLD)
                    p_over = _score_prob(feat, mdl)
                    if p_over*100.0 >= thr and market_cutoff_ok(minute, mk):
                        candidates.append((mk, f"Over {_fmt_line(line)} Goals", p_over))
                    p_under = 1.0 - p_over
                    if p_under*100.0 >= thr and market_cutoff_ok(minute, mk):
                        candidates.append((mk, f"Under {_fmt_line(line)} Goals", p_under))

                # BTTS
                mdl_btts = load_model_from_settings("BTTS_YES")
                if mdl_btts:
                    mk = "BTTS"
                    thr = _get_threshold(f"conf_threshold:{mk}", settings.CONF_THRESHOLD)
                    p_yes = _score_prob(feat, mdl_btts)
                    if p_yes*100.0 >= thr and market_cutoff_ok(minute, mk):
                        candidates.append((mk, "BTTS: Yes", p_yes))
                    p_no = 1.0 - p_yes
                    if p_no*100.0 >= thr and market_cutoff_ok(minute, mk):
                        candidates.append((mk, "BTTS: No", p_no))

                # 1X2 (draw suppressed)
                mh, md, ma = _load_wld_models()
                if mh and md and ma:
                    mk = "1X2"
                    thr = _get_threshold(f"conf_threshold:{mk}", settings.CONF_THRESHOLD)
                    ph = _score_prob(feat, mh)
                    pd = _score_prob(feat, md)
                    pa = _score_prob(feat, ma)
                    s = max(EPS, ph + pd + pa)
                    ph, pa = ph / s, pa / s
                    if ph*100.0 >= thr and market_cutoff_ok(minute, mk):
                        candidates.append((mk, "Home Win", ph))
                    if pa*100.0 >= thr and market_cutoff_ok(minute, mk):
                        candidates.append((mk, "Away Win", pa))

                if not candidates: continue

                # odds + EV
                odds_map = fetch_odds(fid)
                ranked: List[Tuple[str, str, float, Optional[float], Optional[str], Optional[float], float]] = []

                for mk, sug, prob in candidates:
                    if sug not in allowed: continue

                    odds = None; book = None
                    if mk == "BTTS":
                        d = odds_map.get("BTTS", {})
                        tgt = "Yes" if sug.endswith("Yes") else "No"
                        if tgt in d: odds, book = d[tgt]["odds"], d[tgt]["book"]
                    elif mk == "1X2":
                        d = odds_map.get("1X2", {})
                        tgt = "Home" if sug == "Home Win" else ("Away" if sug == "Away Win" else None)
                        if tgt and tgt in d: odds, book = d[tgt]["odds"], d[tgt]["book"]
                    elif mk.startswith("Over/Under"):
                        ln = _parse_ou_line_from_suggestion(sug)
                        d = odds_map.get(f"OU_{_fmt_line(ln)}", {}) if ln is not None else {}
                        tgt = "Over" if sug.startswith("Over") else "Under"
                        if tgt in d: odds, book = d[tgt]["odds"], d[tgt]["book"]

                    if odds is None:
                        reason = _odds_debug_reason(odds_map, mk, sug)
                        log.info("[ODDS] no price for fid=%s %s / %s — %s", fid, mk, sug, reason)
                        continue

                    # odds bounds
                    min_odds = _min_odds_for_market(mk)
                    if not (min_odds <= float(odds) <= settings.MAX_ODDS_ALL):
                        log.info("[ODDS] rejected fid=%s %s / %s — odds %.2f outside [%.2f, %.2f]",
                                 fid, mk, sug, float(odds), min_odds, settings.MAX_ODDS_ALL)
                        continue

                    ev = _ev(prob, float(odds)); ev_pct = round(ev*100.0, 1)
                    if int(round(ev*10000)) < settings.EDGE_MIN_BPS:
                        log.info("[EV] rejected fid=%s %s / %s — EV %+0.1f%% < %d bps",
                                 fid, mk, sug, ev_pct, settings.EDGE_MIN_BPS)
                        continue

                    rank_score = (prob ** 1.2) * (1 + (ev_pct or 0)/100.0)
                    ranked.append((mk, sug, prob, odds, book, ev_pct, rank_score))

                if not ranked: continue
                ranked.sort(key=lambda x: x[6], reverse=True)

                # insert/save + telegram
                per_match = 0
                base_now = int(time.time())
                for idx, (market_txt, suggestion, prob, odds, book, ev_pct, _rank) in enumerate(ranked):
                    if settings.PER_LEAGUE_CAP > 0 and per_league_counter.get(league_id, 0) >= settings.PER_LEAGUE_CAP:
                        continue
                    created_ts = base_now + idx
                    prob_pct = round(float(prob) * 100.0, 1)
                    with db_conn() as c2:
                        c2.execute(
                            "INSERT INTO tips(match_id,league_id,league,home,away,market,suggestion,"
                            "confidence,confidence_raw,score_at_tip,minute,created_ts,"
                            "odds,book,ev_pct,sent_ok) "
                            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,0)",
                            (
                                fid, league_id, league, home, away,
                                market_txt, suggestion,
                                float(prob_pct), float(prob), score, minute, created_ts,
                                (float(odds) if odds is not None else None),
                                (book or None),
                                (float(ev_pct) if ev_pct is not None else None),
                            ),
                        )
                    # notify
                    send_telegram(_format_tip_message(home, away, league, minute, score, suggestion, float(prob_pct), odds, book, ev_pct))
                    with db_conn() as c3:
                        c3.execute("UPDATE tips SET sent_ok=1 WHERE match_id=%s AND created_ts=%s", (fid, created_ts))
                    _metric_inc("tips_generated_total", n=1)
                    saved += 1; per_match += 1; per_league_counter[league_id] = per_league_counter.get(league_id, 0) + 1
                    if settings.MAX_TIPS_PER_SCAN and saved >= settings.MAX_TIPS_PER_SCAN: break
                    if per_match >= max(1, settings.PREDICTIONS_PER_MATCH): break
                if settings.MAX_TIPS_PER_SCAN and saved >= settings.MAX_TIPS_PER_SCAN: break

            except Exception as e:
                log.exception("[SCAN] match loop failed: %s", e)
                continue

    log.info("[SCAN] saved=%d live_seen=%d", saved, live_seen)
    return saved, live_seen

# ──────────────────────────────────────────────────────────────────────────────
# Results backfill + digest
# ──────────────────────────────────────────────────────────────────────────────
def _fixture_by_id(mid: int) -> Optional[dict]:
    js=_api_get(FOOTBALL_API_URL, {"id": mid}) or {}
    arr=js.get("response") or [] if isinstance(js,dict) else []
    return arr[0] if arr else None

def _is_final(short: str) -> bool: return (short or "").upper() in {"FT","AET","PEN"}

def backfill_results_for_open_matches(max_rows: int = 200) -> int:
    now_ts=int(time.time()); cutoff=now_ts - settings.BACKFILL_DAYS*24*3600; updated=0
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

def backfill_results_from_snapshots(max_rows: int = 200, days: int = 365) -> int:
    """
    Use tip_snapshots to discover match_ids (even if no tips were sent),
    fetch final results from API-Football, and write into match_results.
    """
    cutoff = int(time.time()) - days * 24 * 3600
    updated = 0
    with db_conn() as c:
        rows = c.execute(
            """
            SELECT DISTINCT s.match_id
            FROM tip_snapshots s
            LEFT JOIN match_results r ON r.match_id = s.match_id
            WHERE s.created_ts >= %s AND r.match_id IS NULL
            ORDER BY s.created_ts DESC
            LIMIT %s
            """,
            (cutoff, max_rows),
        ).fetchall()

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
                "ON CONFLICT(match_id) DO UPDATE SET final_goals_h=EXCLUDED.final_goals_h, "
                "final_goals_a=EXCLUDED.final_goals_a, btts_yes=EXCLUDED.btts_yes, updated_ts=EXCLUDED.updated_ts",
                (int(mid), gh, ga, btts, int(time.time())),
            )
        updated += 1

    if updated:
        log.info("[RESULTS] backfilled from snapshots: %d", updated)
    return updated

def _tip_outcome_for_result(suggestion: str, res: Dict[str,Any]) -> Optional[int]:
    gh=int(res.get("final_goals_h") or 0); ga=int(res.get("final_goals_a") or 0)
    total=gh+ga; btts=int(res.get("btts_yes") or 0); s=(suggestion or "").strip()
    def _parse_line(s: str)->Optional[float]:
        for tok in (s or "").split():
            try: return float(tok)
            except: pass
        return None
    if s.startswith("Over") or s.startswith("Under"):
        line=_parse_line(s); 
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

def daily_accuracy_digest(window_days: int = 7) -> Optional[str]:
    if not settings.DAILY_ACCURACY_DIGEST_ENABLE: return None
    backfill_results_for_open_matches(400)

    cutoff=int((datetime.now(BERLIN_TZ)-timedelta(days=window_days)).timestamp())
    with db_conn() as c:
        rows=c.execute("""
            SELECT t.market, t.suggestion, t.confidence, t.confidence_raw, t.created_ts,
                   t.odds, r.final_goals_h, r.final_goals_a, r.btts_yes
            FROM tips t LEFT JOIN match_results r ON r.match_id=t.match_id
            WHERE t.created_ts >= %s AND t.suggestion<>'HARVEST' AND t.sent_ok=1
        """,(cutoff,)).fetchall()

    total=graded=wins=0
    roi_by_market, by_market = {}, {}

    for (mkt, sugg, conf, conf_raw, cts, odds, gh, ga, btts) in rows:
        res={"final_goals_h":gh,"final_goals_a":ga,"btts_yes":btts}
        out=_tip_outcome_for_result(sugg,res)
        if out is None: continue

        total+=1; graded+=1; wins+=1 if out==1 else 0
        d=by_market.setdefault(mkt or "?",{"graded":0,"wins":0}); d["graded"]+=1; d["wins"]+=1 if out==1 else 0

        if odds:
            roi_by_market.setdefault(mkt, {"stake":0,"pnl":0})
            roi_by_market[mkt]["stake"]+=1
            if out==1: roi_by_market[mkt]["pnl"]+=float(odds)-1
            else: roi_by_market[mkt]["pnl"]-=1

    if graded==0:
        msg="📊 Accuracy Digest\nNo graded tips in window."
    else:
        acc=100.0*wins/max(1,graded)
        lines=[f"📊 <b>Accuracy Digest</b> (last {window_days}d)",
               f"Tips sent: {total}  •  Graded: {graded}  •  Wins: {wins}  •  Accuracy: {acc:.1f}%"]

        for mk,st in sorted(by_market.items()):
            if st["graded"]==0: continue
            a=100.0*st["wins"]/st["graded"]
            roi=""
            if mk in roi_by_market and roi_by_market[mk]["stake"]>0:
                roi_val=100.0*roi_by_market[mk]["pnl"]/roi_by_market[mk]["stake"]
                roi=f" • ROI {roi_val:+.1f}%"
            lines.append(f"• {mk} — {st['wins']}/{st['graded']} ({a:.1f}%){roi}")
        msg="\n".join(lines)

    send_telegram(msg); return msg

# ──────────────────────────────────────────────────────────────────────────────
# Scheduler
# ──────────────────────────────────────────────────────────────────────────────
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
    if _scheduler_started or not settings.RUN_SCHEDULER:
        return
    try:
        sched = BackgroundScheduler(timezone=TZ_UTC)
        sched.add_job(lambda: _run_with_pg_lock(1001, production_scan),
                      "interval", seconds=settings.SCAN_INTERVAL_SEC, id="scan", max_instances=1, coalesce=True)
        sched.add_job(lambda: _run_with_pg_lock(1002, backfill_results_for_open_matches, 400),
                      "interval", minutes=settings.BACKFILL_EVERY_MIN, id="backfill", max_instances=1, coalesce=True)

        if settings.DAILY_ACCURACY_DIGEST_ENABLE:
            sched.add_job(lambda: _run_with_pg_lock(1003, daily_accuracy_digest),
                          CronTrigger(hour=settings.DAILY_ACCURACY_HOUR, minute=settings.DAILY_ACCURACY_MINUTE, timezone=BERLIN_TZ),
                          id="digest", max_instances=1, coalesce=True)

        if settings.TRAIN_ENABLE:
            from train_models import train_models  # lazy import so this file can run standalone
            sched.add_job(lambda: _run_with_pg_lock(1005, train_models),
                          CronTrigger(hour=settings.TRAIN_HOUR_UTC, minute=settings.TRAIN_MINUTE_UTC, timezone=TZ_UTC),
                          id="train", max_instances=1, coalesce=True)

        if settings.AUTO_TUNE_ENABLE:
            from train_models import auto_tune_thresholds
            sched.add_job(lambda: _run_with_pg_lock(1006, auto_tune_thresholds, 14),
                          CronTrigger(hour=4, minute=7, timezone=TZ_UTC),
                          id="auto_tune", max_instances=1, coalesce=True)

        sched.start()
        _scheduler_started = True
        send_telegram("🚀 goalsniper AI mode started.")
        log.info("[SCHED] started (scan=%ss)", settings.SCAN_INTERVAL_SEC)
    except Exception as e:
        log.exception("[SCHED] failed: %s", e)

# ──────────────────────────────────────────────────────────────────────────────
# Admin & API
# ──────────────────────────────────────────────────────────────────────────────
def _require_admin():
    key=request.headers.get("X-API-Key") or request.args.get("key") or ((request.json or {}).get("key") if request.is_json else None)
    if not settings.ADMIN_API_KEY or key != settings.ADMIN_API_KEY: abort(401)

@app.route("/")
def root(): return jsonify({"ok": True, "name": "goalsniper", "mode": "FULL_AI", "scheduler": settings.RUN_SCHEDULER})

@app.route("/health")
def health():
    try:
        with db_conn() as c:
            n=c.execute("SELECT COUNT(*) FROM tips").fetchone()[0]
        return jsonify({"ok": True, "db": "ok", "tips_count": int(n)})
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

@app.route("/admin/backfill-results-snapshots", methods=["POST","GET"])
def http_backfill_snapshots():
    _require_admin()
    n = backfill_results_from_snapshots(1000, 365)  # scan a lot on demand
    return jsonify({"ok": True, "updated": int(n)})

@app.route("/admin/train", methods=["POST","GET"])
def http_train():
    _require_admin()
    try:
        from train_models import train_models
        out = train_models()
        return jsonify({"ok": True, "result": out})
    except Exception as e:
        log.exception("train_models failed: %s", e); return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/admin/digest", methods=["POST","GET"])
def http_digest():
    _require_admin()
    msg = daily_accuracy_digest()
    return jsonify({"ok": True, "sent": bool(msg)})

@app.route("/admin/auto-tune", methods=["POST","GET"])
def http_auto_tune():
    _require_admin()
    try:
        from train_models import auto_tune_thresholds
        tuned=auto_tune_thresholds(14)
        return jsonify({"ok": True, "tuned": tuned})
    except Exception as e:
        log.exception("auto_tune failed: %s", e); return jsonify({"ok": False, "error": str(e)}), 500

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

# ──────────────────────────────────────────────────────────────────────────────
# Boot
# ──────────────────────────────────────────────────────────────────────────────
def _on_boot():
    _init_pool()
    init_db()
    set_setting("boot_ts", str(int(time.time())))
    _start_scheduler_once()

_on_boot()

if __name__ == "__main__":
    # For Railway, $PORT is provided; Flask builtin server for local dev; use gunicorn in prod
    app.run(host=os.getenv("HOST","0.0.0.0"), port=int(os.getenv("PORT","8080")), threaded=True)
