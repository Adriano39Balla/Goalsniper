"""
goalsniper — FULL AI mode (in-play + prematch) with odds + EV gate.

- Pure ML (calibrated) loaded from Postgres settings (train_models.py).
- Markets: OU(2.5,3.5), BTTS (Yes/No), 1X2 (Draw suppressed).
- Adds bookmaker odds filtering + EV check.
- Scheduler: scan, results backfill, nightly train, daily digest, MOTD.

Safe to run on Railway/Render. Requires DATABASE_URL and API keys.

PATCH NOTES (this revision):
  1. [BUG-FIX] Live 1X2 renormalization now divides by (Home+Away) instead of
     (Home+Draw+Away). Dividing by the discarded draw probability was
     systematically deflating both Home/Away probabilities and suppressing
     legitimate tips below threshold.
  2. [RELIABILITY] Switched SimpleConnectionPool -> ThreadedConnectionPool
     (Flask request threads + APScheduler background thread both pull from
     the pool concurrently; SimpleConnectionPool does not guarantee
     thread-safety).
  3. [RELIABILITY] PooledConn now detects broken connections
     (OperationalError/InterfaceError) on exit and discards them instead of
     returning a poisoned connection to the pool.
  4. [PERF] production_scan() and prematch_scan_save() no longer hold a
     single DB connection open across the entire per-match loop while doing
     blocking HTTP calls (odds fetch, Telegram send). Connections are now
     acquired briefly, only around the actual DB statement.
  5. [PERF] Prematch team-form / H2H lookups (_api_last_fixtures/_api_h2h)
     are now cached (TTL) and fetched concurrently per fixture, and fixture
     feature extraction across a whole scan is parallelized with a bounded
     thread pool. This cache is shared between prematch_scan_save() and
     send_match_of_the_day(), so running both no longer doubles API calls.
  6. [MINOR] request.json (deprecated) -> request.get_json(silent=True).
  7. [DEPLOY SAFETY] Added `from __future__ import annotations`. The file
     uses PEP 604 union syntax (`tuple|list`, `str|None`) in a couple of
     type hints, which raises TypeError at import time on Python <3.10.
     Nothing pins a Python version in requirements.txt/railway.yaml, so
     this makes the file safe regardless of what Nixpacks resolves to.
"""
from __future__ import annotations

import os, json, time, logging, requests, psycopg2
from psycopg2.pool import ThreadedConnectionPool
from html import escape
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
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

# ───────── Core env ─────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")
API_KEY            = os.getenv("API_KEY")
ADMIN_API_KEY      = os.getenv("ADMIN_API_KEY")
WEBHOOK_SECRET     = os.getenv("TELEGRAM_WEBHOOK_SECRET")
RUN_SCHEDULER      = os.getenv("RUN_SCHEDULER", "1") not in ("0","false","False","no","NO")

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

# PATCH: prematch scan is now scheduled automatically (was previously only
# reachable via the /admin/prematch-scan endpoint). Runs on an interval
# rather than once a day so fixtures added later in the day still get
# picked up, and so team-form/Elo-derived features get refreshed closer to
# kickoff rather than staying stale from a single morning snapshot.
PREMATCH_SCAN_ENABLE        = os.getenv("PREMATCH_SCAN_ENABLE", "1") not in ("0","false","False","no","NO")
PREMATCH_SCAN_INTERVAL_MIN  = int(os.getenv("PREMATCH_SCAN_INTERVAL_MIN", "180"))

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
except Exception:
    MOTD_LEAGUE_IDS = []

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
PREDICTIONS_PER_MATCH = int(os.getenv("PREDICTIONS_PER_MATCH", "2"))

# ───────── Odds/EV controls ─────────
MIN_ODDS_OU   = float(os.getenv("MIN_ODDS_OU",   "1.30"))
MIN_ODDS_BTTS = float(os.getenv("MIN_ODDS_BTTS", "1.30"))
MIN_ODDS_1X2  = float(os.getenv("MIN_ODDS_1X2",  "1.30"))
MAX_ODDS_ALL  = float(os.getenv("MAX_ODDS_ALL",  "20.0"))
EDGE_MIN_BPS  = int(os.getenv("EDGE_MIN_BPS", "300"))  # 300 = +3.00%
ODDS_BOOKMAKER_ID = os.getenv("ODDS_BOOKMAKER_ID")  # optional API-Football book id
ALLOW_TIPS_WITHOUT_ODDS = os.getenv("ALLOW_TIPS_WITHOUT_ODDS","1") not in ("0","false","False","no","NO")

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
# PATCH: pool_maxsize was left at requests' default (10). Concurrent
# fetching added earlier (fetch_live_matches hydrates up to 8 matches at
# once, each doing 2 concurrent sub-requests for stats+events) can open up
# to ~16 simultaneous connections to v3.football.api-sports.io through
# this one shared session — anything past 10 was being discarded and
# re-opened from scratch (fresh TCP+TLS handshake) instead of reused,
# which is exactly what pooling exists to avoid. Sized above the realistic
# concurrent ceiling across all thread pools in this file, configurable
# via env since ceiling depends on MAX_WORKERS-style constants elsewhere.
HTTP_POOL_MAXSIZE = int(os.getenv("HTTP_POOL_MAXSIZE", "30"))
session.mount("https://", HTTPAdapter(
    max_retries=Retry(total=3, backoff_factor=1, status_forcelist=[429,500,502,503,504], respect_retry_after_header=True),
    pool_connections=HTTP_POOL_MAXSIZE, pool_maxsize=HTTP_POOL_MAXSIZE))

# ───────── Caches & timezones ─────────
STATS_CACHE:  Dict[int, Tuple[float, list]] = {}
EVENTS_CACHE: Dict[int, Tuple[float, list]] = {}
ODDS_CACHE:   Dict[int, Tuple[float, dict]] = {}
# PATCH: shared TTL cache for prematch team-form / H2H lookups, reused by
# prematch_scan_save() and send_match_of_the_day() so they don't each
# re-fetch the same team's last-5 fixtures separately.
TEAM_FORM_CACHE: Dict[tuple, Tuple[float, list]] = {}
TEAM_FORM_TTL = int(os.getenv("TEAM_FORM_CACHE_TTL_SEC", "1800"))  # 30 min

SETTINGS_TTL = int(os.getenv("SETTINGS_TTL_SEC","60"))
MODELS_TTL   = int(os.getenv("MODELS_CACHE_TTL_SEC","120"))
TZ_UTC, BERLIN_TZ = ZoneInfo("UTC"), ZoneInfo("Europe/Berlin")

# ───────── Optional import: trainer ─────────
try:
    from train_models import train_models
except Exception as e:
    _IMPORT_ERR = repr(e)
    def train_models(*args, **kwargs):  # type: ignore
        log.warning("train_models not available: %s", _IMPORT_ERR)
        return {"ok": False, "reason": f"train_models import failed: {_IMPORT_ERR}"}

# ───────── DB pool & helpers ─────────
POOL: Optional[ThreadedConnectionPool] = None

class PooledConn:
    """
    Context manager for a pooled DB connection/cursor.

    PATCH: on exit, if the connection raised an OperationalError/
    InterfaceError (broken pipe, server closed connection, etc.) it is
    discarded (closed) instead of being returned to the pool. Returning a
    dead connection to the pool means the *next* caller to get it fails
    immediately with an unrelated-looking error; discarding it lets the
    pool open a fresh one on demand.
    """
    def __init__(self, pool): self.pool=pool; self.conn=None; self.cur=None
    def __enter__(self):
        self.conn=self.pool.getconn(); self.conn.autocommit=True; self.cur=self.conn.cursor(); return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self.cur: self.cur.close()
        except Exception:
            pass
        finally:
            if self.conn is not None:
                broken = exc_type is not None and issubclass(exc_type, (psycopg2.OperationalError, psycopg2.InterfaceError))
                try:
                    self.pool.putconn(self.conn, close=broken)
                except Exception:
                    try: self.conn.close()
                    except Exception: pass
    def execute(self, sql: str, params: tuple|list=()):
        self.cur.execute(sql, params or ()); return self.cur

def _init_pool():
    global POOL
    dsn = DATABASE_URL + (("&" if "?" in DATABASE_URL else "?") + "sslmode=require" if "sslmode=" not in DATABASE_URL else "")
    # PATCH: ThreadedConnectionPool instead of SimpleConnectionPool — this
    # app is accessed from multiple threads at once (Flask request threads
    # + the APScheduler background thread), and SimpleConnectionPool is
    # documented as not being safe to share across threads.
    POOL = ThreadedConnectionPool(minconn=1, maxconn=int(os.getenv("DB_POOL_MAX","5")), dsn=dsn)

def db_conn():
    if not POOL: _init_pool()
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
        # PATCH: Elo-style team strength ratings, updated after each result.
        c.execute("""CREATE TABLE IF NOT EXISTS team_ratings (
            team_id BIGINT PRIMARY KEY, rating DOUBLE PRECISION NOT NULL DEFAULT 1500.0, updated_ts BIGINT)""")
        # PATCH: this was missing. save_prematch_snapshot() (called from
        # prematch_scan_save()) writes to this table, but only
        # train_models.py's _ensure_training_tables() was creating it —
        # which only runs when training actually executes. Without this,
        # the prematch scheduler job would fail every run with
        # "relation prematch_snapshots does not exist" (silently swallowed
        # by a try/except), meaning zero prematch training data would ever
        # get collected. Schema matches train_models.py's version exactly.
        c.execute("""CREATE TABLE IF NOT EXISTS prematch_snapshots (
            match_id BIGINT PRIMARY KEY, created_ts BIGINT, payload TEXT)""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_pre_snap_ts ON prematch_snapshots (created_ts DESC)")
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

def _fetch_stats_and_events(fid: int) -> Tuple[list, list]:
    """PATCH: fetch stats+events for one fixture concurrently instead of
    sequentially (each is an independent HTTP call)."""
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_stats = ex.submit(fetch_match_stats, fid)
        f_events = ex.submit(fetch_match_events, fid)
        return f_stats.result(), f_events.result()

def fetch_live_matches() -> List[dict]:
    js=_api_get(FOOTBALL_API_URL, {"live":"all"}) or {}
    matches=[m for m in (js.get("response",[]) if isinstance(js,dict) else []) if not _blocked_league(m.get("league") or {})]
    eligible=[]
    for m in matches:
        st=((m.get("fixture",{}) or {}).get("status",{}) or {})
        elapsed=st.get("elapsed"); short=(st.get("short") or "").upper()
        if elapsed is None or elapsed>120 or short not in INPLAY_STATUSES: continue
        eligible.append(m)

    # PATCH: fetch stats/events for all eligible matches concurrently
    # (bounded worker pool) instead of one match at a time.
    def _hydrate(m: dict) -> dict:
        fid=(m.get("fixture",{}) or {}).get("id")
        stats, events = _fetch_stats_and_events(fid)
        m["statistics"]=stats; m["events"]=events
        return m

    if not eligible: return []
    with ThreadPoolExecutor(max_workers=min(8, max(1,len(eligible)))) as ex:
        out=list(ex.map(_hydrate, eligible))
    return out

# ───────── Prematch helpers (short) ─────────
def _api_last_fixtures(team_id: int, n: int = 5) -> List[dict]:
    key=("last", team_id, n)
    now=time.time()
    cached=TEAM_FORM_CACHE.get(key)
    if cached and now-cached[0] < TEAM_FORM_TTL: return cached[1]
    js=_api_get(f"{BASE_URL}/fixtures", {"team":team_id,"last":n}) or {}
    out = js.get("response",[]) if isinstance(js,dict) else []
    TEAM_FORM_CACHE[key]=(now,out)
    return out

def _api_h2h(home_id: int, away_id: int, n: int = 5) -> List[dict]:
    key=("h2h", home_id, away_id, n)
    now=time.time()
    cached=TEAM_FORM_CACHE.get(key)
    if cached and now-cached[0] < TEAM_FORM_TTL: return cached[1]
    js=_api_get(f"{BASE_URL}/fixtures/headtohead", {"h2h":f"{home_id}-{away_id}","last":n}) or {}
    out = js.get("response",[]) if isinstance(js,dict) else []
    TEAM_FORM_CACHE[key]=(now,out)
    return out

def _collect_todays_prematch_fixtures() -> List[dict]:
    today_local=datetime.now(ZoneInfo("Europe/Berlin")).date()
    start_local=datetime.combine(today_local, datetime.min.time(), tzinfo=ZoneInfo("Europe/Berlin"))
    end_local=start_local+timedelta(days=1)
    dates_utc={start_local.astimezone(TZ_UTC).date(), (end_local - timedelta(seconds=1)).astimezone(TZ_UTC).date()}
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
    """
    PATCH: extended to compute the FULL feature set train_models.py trains
    on (FEATURES list), not just the ~20 basic fields. Every derived
    feature below is ported verbatim from train_models.py's
    load_inplay_data() so serving and training compute identical values
    from identical raw inputs — previously ~40 of the model's learned
    weights were being multiplied by 0.0 at serving time because this
    function never populated them.
    """
    home=m["teams"]["home"]["name"]; away=m["teams"]["away"]["name"]
    gh=m["goals"]["home"] or 0; ga=m["goals"]["away"] or 0
    minute=int(((m.get("fixture") or {}).get("status") or {}).get("elapsed") or 0)
    stats={}
    for s in (m.get("statistics") or []):
        t=(s.get("team") or {}).get("name")
        if t: stats[t]={ (i.get("type") or ""): i.get("value") for i in (s.get("statistics") or []) }
    sh=stats.get(home,{}) or {}; sa=stats.get(away,{}) or {}
    # PATCH: API-Football's real /fixtures/statistics response uses the key
    # "Shots on Goal" (confirmed against their official example response),
    # not "Shots on Target" — the old key never matched anything, so sot_h/
    # sot_a (and everything derived from them: shot_accuracy, conversion_rate,
    # momentum_score, attack_pressure, sot_xg_ratio) were silently 0 on
    # every live prediction. xG coverage varies by league/plan on
    # API-Football and is frequently null/absent for a given fixture — that's
    # real API behavior, not a bug — so we check a couple of plausible key
    # spellings and fall back to 0.0 rather than assume one exact casing.
    xg_h=_num(sh.get("Expected Goals", sh.get("expected_goals", 0)))
    xg_a=_num(sa.get("Expected Goals", sa.get("expected_goals", 0)))
    sot_h=_num(sh.get("Shots on Goal",0)); sot_a=_num(sa.get("Shots on Goal",0))
    cor_h=_num(sh.get("Corner Kicks",0)); cor_a=_num(sa.get("Corner Kicks",0))
    pos_h=_pos_pct(sh.get("Ball Possession",0)); pos_a=_pos_pct(sa.get("Ball Possession",0))
    total_shots_h=_num(sh.get("Total Shots",0)); total_shots_a=_num(sa.get("Total Shots",0))
    shots_inside_h=_num(sh.get("Shots insidebox",0)); shots_inside_a=_num(sa.get("Shots insidebox",0))
    fouls_h=_num(sh.get("Fouls",0)); fouls_a=_num(sa.get("Fouls",0))
    red_h=red_a=0
    for ev in (m.get("events") or []):
        if (ev.get("type","").lower()=="card"):
            d=(ev.get("detail","") or "").lower()
            if "red" in d or "second yellow" in d:
                t=(ev.get("team") or {}).get("name") or ""
                if t==home: red_h+=1
                elif t==away: red_a+=1

    f: Dict[str,float] = {
        "minute":float(minute),
        "goals_h":float(gh),"goals_a":float(ga),
        "xg_h":float(xg_h),"xg_a":float(xg_a),
        "sot_h":float(sot_h),"sot_a":float(sot_a),
        "cor_h":float(cor_h),"cor_a":float(cor_a),
        "pos_h":float(pos_h),"pos_a":float(pos_a),
        "red_h":float(red_h),"red_a":float(red_a),
        "total_shots_h":float(total_shots_h),"total_shots_a":float(total_shots_a),
        "shots_inside_h":float(shots_inside_h),"shots_inside_a":float(shots_inside_a),
        "fouls_h":float(fouls_h),"fouls_a":float(fouls_a),
    }
    f["goals_sum"]=f["goals_h"]+f["goals_a"]; f["goals_diff"]=f["goals_h"]-f["goals_a"]
    f["xg_sum"]=f["xg_h"]+f["xg_a"]; f["xg_diff"]=f["xg_h"]-f["xg_a"]
    f["sot_sum"]=f["sot_h"]+f["sot_a"]
    f["cor_sum"]=f["cor_h"]+f["cor_a"]
    f["pos_diff"]=f["pos_h"]-f["pos_a"]
    f["red_sum"]=f["red_h"]+f["red_a"]

    mnt=f["minute"]; goals_sum=f["goals_sum"]; xg_sum=f["xg_sum"]; sot_sum=f["sot_sum"]
    total_shots_sum=f["total_shots_h"]+f["total_shots_a"]
    if mnt>0:
        f["goals_per_minute"]=goals_sum/mnt; f["xg_per_minute"]=xg_sum/mnt
        f["sot_per_minute"]=sot_sum/mnt; f["shots_per_minute"]=total_shots_sum/mnt
    else:
        f["goals_per_minute"]=f["xg_per_minute"]=f["sot_per_minute"]=f["shots_per_minute"]=0.0

    f["momentum_score"]=(f["xg_per_minute"]*0.5 + f["sot_per_minute"]*0.3 + f["shots_per_minute"]*0.2)
    f["shot_accuracy_h"]=f["sot_h"]/max(f["total_shots_h"],1)
    f["shot_accuracy_a"]=f["sot_a"]/max(f["total_shots_a"],1)
    f["shot_quality_h"]=f["shots_inside_h"]/max(f["total_shots_h"],1)
    f["shot_quality_a"]=f["shots_inside_a"]/max(f["total_shots_a"],1)
    f["conversion_rate_h"]=f["goals_h"]/max(f["sot_h"],1)
    f["conversion_rate_a"]=f["goals_a"]/max(f["sot_a"],1)
    f["xg_efficiency_h"]=f["goals_h"]-f["xg_h"]
    f["xg_efficiency_a"]=f["goals_a"]-f["xg_a"]
    f["attack_pressure_h"]=(f["sot_h"]*0.4+f["xg_h"]*0.4+f["cor_h"]*0.2)
    f["attack_pressure_a"]=(f["sot_a"]*0.4+f["xg_a"]*0.4+f["cor_a"]*0.2)
    f["attack_pressure_diff"]=f["attack_pressure_h"]-f["attack_pressure_a"]
    f["game_control_h"]=(f["pos_h"]/100)*f["attack_pressure_h"]
    f["game_control_a"]=(f["pos_a"]/100)*f["attack_pressure_a"]
    f["is_first_half"]=1.0 if mnt<=45 else 0.0
    f["is_second_half"]=1.0 if mnt>45 else 0.0
    f["is_final_15"]=1.0 if mnt>75 else 0.0
    f["score_margin"]=abs(f["goals_h"]-f["goals_a"])
    f["is_leading_h"]=1.0 if f["goals_h"]>f["goals_a"] else 0.0
    f["is_leading_a"]=1.0 if f["goals_a"]>f["goals_h"] else 0.0
    f["is_draw"]=1.0 if f["goals_h"]==f["goals_a"] else 0.0
    f["is_goalfest"]=1.0 if f["goals_sum"]>=3 else 0.0
    fouls_sum=f["fouls_h"]+f["fouls_a"]
    f["fouls_per_minute"]=fouls_sum/max(mnt,1)
    f["discipline_score_h"]=1.0/max(f["fouls_h"]+f["red_h"]*10,1)
    f["discipline_score_a"]=1.0/max(f["fouls_a"]+f["red_a"]*10,1)
    f["possession_xg_interaction_h"]=(f["pos_h"]/100)*f["xg_h"]
    f["possession_xg_interaction_a"]=(f["pos_a"]/100)*f["xg_a"]
    f["sot_xg_ratio_h"]=f["sot_h"]/max(f["xg_h"],0.1)
    f["sot_xg_ratio_a"]=f["sot_a"]/max(f["xg_a"],0.1)
    f["match_minute_normalized"]=mnt/90.0
    f["time_weighted_xg_h"]=f["xg_h"]*(mnt/90.0)
    f["time_weighted_xg_a"]=f["xg_a"]*(mnt/90.0)
    return f

def stats_coverage_ok(feat: Dict[str,float], minute: int) -> bool:
    require_stats_minute=int(os.getenv("REQUIRE_STATS_MINUTE","35"))
    require_fields=int(os.getenv("REQUIRE_DATA_FIELDS","2"))
    if minute < require_stats_minute: return True
    fields=[feat.get("xg_sum",0.0), feat.get("sot_sum",0.0), feat.get("cor_sum",0.0),
            max(feat.get("pos_h",0.0), feat.get("pos_a",0.0))]
    nonzero=sum(1 for v in fields if (v or 0)>0)
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
            cal=tmp.get("calibration") or {}
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
def _load_wld_models(): return (load_model_from_settings("WLD_HOME"), load_model_from_settings("WLD_DRAW"), load_model_from_settings("WLD_AWAY"))

# ───────── Odds helpers ─────────
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

def _market_name_normalize(s: str) -> str:
    s=(s or "").lower()
    if "both teams" in s or "btts" in s: return "BTTS"
    if "match winner" in s or "winner" in s or "1x2" in s: return "1X2"
    if "over/under" in s or "total" in s or "goals" in s: return "OU"
    return s

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
    cached=_odds_cache_get(fid)
    if cached is not None: return cached
    params={"fixture": fid}
    if ODDS_BOOKMAKER_ID: params["bookmaker"] = ODDS_BOOKMAKER_ID
    js=_api_get(f"{BASE_URL}/odds", params) or {}
    out={}
    try:
        for r in js.get("response",[]) if isinstance(js,dict) else []:
            book=(r.get("bookmakers") or [])
            if not book: continue
            bk=book[0]; book_name=bk.get("name") or "Book"
            for mkt in (bk.get("bets") or []):
                mname=_market_name_normalize(mkt.get("name",""))
                vals=mkt.get("values") or []
                # BTTS
                if mname=="BTTS":
                    d={}
                    for v in vals:
                        lbl=(v.get("value") or "").strip().lower()
                        if "yes" in lbl: d["Yes"]={"odds":float(v.get("odd") or 0), "book":book_name}
                        if "no"  in lbl: d["No"] ={"odds":float(v.get("odd") or 0), "book":book_name}
                    if d: out["BTTS"]=d
                # 1X2
                elif mname=="1X2":
                    d={}
                    for v in vals:
                        lbl=(v.get("value") or "").strip().lower()
                        if lbl in ("home","1"): d["Home"]={"odds":float(v.get("odd") or 0),"book":book_name}
                        if lbl in ("away","2"): d["Away"]={"odds":float(v.get("odd") or 0),"book":book_name}
                    if d: out["1X2"]=d
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
                            except: pass
                    for k,v in by_line.items(): out[k]=v
        ODDS_CACHE[fid]=(time.time(), out)
    except Exception:
        out={}
    return out

def _price_gate(market_text: str, suggestion: str, fid: int) -> Tuple[bool, Optional[float], Optional[str], Optional[float]]:
    """
    Return (pass, odds, book, ev_pct). If odds missing:
      - pass if ALLOW_TIPS_WITHOUT_ODDS else block.
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
        ln=_fmt_line(float(suggestion.split()[1]))
        d=odds_map.get(f"OU_{ln}",{})
        tgt="Over" if suggestion.startswith("Over") else "Under"
        if tgt in d: odds=d[tgt]["odds"]; book=d[tgt]["book"]

    if odds is None:
        return (ALLOW_TIPS_WITHOUT_ODDS, None, None, None)

    # price range gates
    min_odds=_min_odds_for_market(market_text)
    if not (min_odds <= odds <= MAX_ODDS_ALL):
        return (False, odds, book, None)

    return (True, odds, book, None)

# ───────── Team ratings (Elo) ─────────
# PATCH: backs the pm_rating_* / pm_home_adv_rating / pm_away_adv_rating
# features that PRE_FEATURES (train_models.py) expects but nothing was
# previously computing. Standard Elo with a home-field constant.
ELO_DEFAULT = float(os.getenv("ELO_DEFAULT", "1500.0"))
ELO_K       = float(os.getenv("ELO_K", "20.0"))
ELO_HOME_ADV= float(os.getenv("ELO_HOME_ADV", "60.0"))

def get_team_rating(team_id: int) -> float:
    if not team_id: return ELO_DEFAULT
    with db_conn() as c:
        r=c.execute("SELECT rating FROM team_ratings WHERE team_id=%s",(team_id,)).fetchone()
        return float(r[0]) if r else ELO_DEFAULT

def get_team_ratings_bulk(team_ids: List[int]) -> Dict[int,float]:
    ids=[t for t in set(team_ids) if t]
    if not ids: return {}
    with db_conn() as c:
        rows=c.execute("SELECT team_id, rating FROM team_ratings WHERE team_id = ANY(%s)",(ids,)).fetchall()
    out={int(t):ELO_DEFAULT for t in ids}
    for tid,rating in rows: out[int(tid)]=float(rating)
    return out

def update_team_ratings(home_id: int, away_id: int, gh: int, ga: int) -> None:
    if not home_id or not away_id: return
    rh=get_team_rating(home_id); ra=get_team_rating(away_id)
    exp_h=1.0/(1.0+10**(((ra)-(rh+ELO_HOME_ADV))/400.0))
    score_h = 1.0 if gh>ga else (0.5 if gh==ga else 0.0)
    new_rh = rh + ELO_K*(score_h-exp_h)
    new_ra = ra + ELO_K*((1.0-score_h)-(1.0-exp_h))
    now=int(time.time())
    with db_conn() as c:
        c.execute("INSERT INTO team_ratings(team_id,rating,updated_ts) VALUES(%s,%s,%s) "
                  "ON CONFLICT(team_id) DO UPDATE SET rating=EXCLUDED.rating, updated_ts=EXCLUDED.updated_ts",
                  (home_id, float(new_rh), now))
        c.execute("INSERT INTO team_ratings(team_id,rating,updated_ts) VALUES(%s,%s,%s) "
                  "ON CONFLICT(team_id) DO UPDATE SET rating=EXCLUDED.rating, updated_ts=EXCLUDED.updated_ts",
                  (away_id, float(new_ra), now))

# ───────── Snapshots ─────────
def save_snapshot_from_match(m: dict, feat: Dict[str,float]) -> None:
    fx=m.get("fixture",{}) or {}; lg=m.get("league",{}) or {}
    fid=int(fx.get("id")); league_id=int(lg.get("id") or 0)
    league=f"{lg.get('country','')} - {lg.get('name','')}".strip(" -")
    home=(m.get("teams") or {}).get("home",{}).get("name","")
    away=(m.get("teams") or {}).get("away",{}).get("name","")
    gh=(m.get("goals") or {}).get("home") or 0; ga=(m.get("goals") or {}).get("away") or 0
    minute=int(feat.get("minute",0))
    snapshot={"minute":minute,"gh":gh,"ga":ga,"league_id":league_id,"market":"HARVEST","suggestion":"HARVEST","confidence":0,
              "stat":{"xg_h":feat.get("xg_h",0),"xg_a":feat.get("xg_a",0),"sot_h":feat.get("sot_h",0),"sot_a":feat.get("sot_a",0),
                      "cor_h":feat.get("cor_h",0),"cor_a":feat.get("cor_a",0),"pos_h":feat.get("pos_h",0),"pos_a":feat.get("pos_a",0),
                      "red_h":feat.get("red_h",0),"red_a":feat.get("red_a",0),
                      # PATCH: these three were missing, so train_models.py's
                      # load_inplay_data() always saw them as 0 — meaning
                      # total_shots/shots_inside/fouls (and every feature
                      # derived from them: shot_accuracy, shot_quality,
                      # attack_pressure, discipline_score, etc.) were trained
                      # on all-zero data regardless of the real match stats.
                      "total_shots_h":feat.get("total_shots_h",0),"total_shots_a":feat.get("total_shots_a",0),
                      "shots_inside_h":feat.get("shots_inside_h",0),"shots_inside_a":feat.get("shots_inside_a",0),
                      "fouls_h":feat.get("fouls_h",0),"fouls_a":feat.get("fouls_a",0)}}
    now=int(time.time())
    with db_conn() as c:
        c.execute("INSERT INTO tip_snapshots(match_id, created_ts, payload) VALUES (%s,%s,%s) "
                  "ON CONFLICT (match_id, created_ts) DO UPDATE SET payload=EXCLUDED.payload",
                  (fid, now, json.dumps(snapshot)[:200000]))
        c.execute("INSERT INTO tips(match_id,league_id,league,home,away,market,suggestion,confidence,confidence_raw,score_at_tip,minute,created_ts,sent_ok) "
                  "VALUES (%s,%s,%s,%s,%s,'HARVEST','HARVEST',0.0,0.0,%s,%s,%s,1)",
                  (fid, league_id, league, home, away, f"{gh}-{ga}", minute, now))

def save_prematch_snapshot(fid: int, feat: Dict[str,float]) -> None:
    """
    PATCH: this was the missing piece — nothing previously wrote to
    prematch_snapshots, so PRE_* models had zero training data no matter
    how long the bot ran. Called once per collected fixture in
    prematch_scan_save(), independent of whether that fixture ends up
    generating a tip, so training data accumulates from every fixture seen.
    """
    payload={"feat": {k:v for k,v in feat.items() if not k.startswith("_")}}
    now=int(time.time())
    with db_conn() as c:
        c.execute("INSERT INTO prematch_snapshots(match_id, created_ts, payload) VALUES (%s,%s,%s) "
                  "ON CONFLICT (match_id) DO UPDATE SET created_ts=EXCLUDED.created_ts, payload=EXCLUDED.payload",
                  (fid, now, json.dumps(payload)[:200000]))

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
        line=_parse_ou_line_from_suggestion(s)
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
        # PATCH: update Elo ratings now that we know the result — this is
        # what feeds pm_rating_h/pm_rating_a/pm_rating_diff for future
        # prematch predictions. Nothing previously updated team_ratings at all.
        try:
            th=((fx.get("teams") or {}).get("home") or {}).get("id")
            ta=((fx.get("teams") or {}).get("away") or {}).get("id")
            if th and ta: update_team_ratings(int(th), int(ta), gh, ga)
        except Exception:
            log.warning("[ELO] rating update failed for match %s", mid)
        updated+=1
    if updated: log.info("[RESULTS] backfilled %d", updated)
    return updated

def daily_accuracy_digest() -> Optional[str]:
    if not DAILY_ACCURACY_DIGEST_ENABLE: return None
    now_local=datetime.now(BERLIN_TZ)
    y0=(now_local - timedelta(days=1)).replace(hour=0,minute=0,second=0,microsecond=0); y1=y0+timedelta(days=1)
    backfill_results_for_open_matches(400)
    with db_conn() as c:
        rows=c.execute("""
            SELECT t.match_id, t.market, t.suggestion, t.confidence, t.confidence_raw, t.created_ts,
                   r.final_goals_h, r.final_goals_a, r.btts_yes
            FROM tips t LEFT JOIN match_results r ON r.match_id=t.match_id
            WHERE t.created_ts >= %s AND t.created_ts < %s AND t.suggestion<>'HARVEST' AND t.sent_ok=1
        """,(int(y0.timestamp()), int(y1.timestamp()))).fetchall()
    total=graded=wins=0; by={}
    for (mid, mkt, sugg, conf, conf_raw, cts, gh, ga, btts) in rows:
        res={"final_goals_h":gh,"final_goals_a":ga,"btts_yes":btts}
        out=_tip_outcome_for_result(sugg,res)
        if out is None: continue
        total+=1; graded+=1; wins+=1 if out==1 else 0
        d=by.setdefault(mkt or "?",{"graded":0,"wins":0}); d["graded"]+=1; d["wins"]+=1 if out==1 else 0
    if graded==0:
        msg="📊 Daily Digest\nNo graded tips for yesterday."
    else:
        acc=100.0*wins/max(1,graded)
        lines=[f"📊 <b>Daily Digest</b> (yesterday, Berlin time)",
               f"Tips sent: {total}  •  Graded: {graded}  •  Wins: {wins}  •  Accuracy: {acc:.1f}%"]
        for mk,st in sorted(by.items()):
            if st["graded"]==0: continue
            a=100.0*st["wins"]/st["graded"]; lines.append(f"• {escape(mk)} — {st['wins']}/{st['graded']} ({a:.1f}%)")
        msg="\n".join(lines)
    send_telegram(msg); return msg

# ───────── Thresholds & formatting ─────────
def _get_market_threshold_key(m: str) -> str: return f"conf_threshold:{m}"
def _get_market_threshold(m: str) -> float:
    try:
        v=get_setting_cached(_get_market_threshold_key(m)); return float(v) if v is not None else float(CONF_THRESHOLD)
    except: return float(CONF_THRESHOLD)
def _get_market_threshold_pre(m: str) -> float: return _get_market_threshold(f"PRE {m}")

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

# ───────── Scan (in-play) ─────────
def _candidate_is_sane(sug: str, feat: Dict[str,float]) -> bool:
    gh=int(feat.get("goals_h",0)); ga=int(feat.get("goals_a",0)); total=gh+ga
    if sug.startswith("Over"):
        ln=_parse_ou_line_from_suggestion(sug)
        if ln is None: return False
        if total > ln - 1e-9: return False
    if sug.startswith("Under"):
        ln=_parse_ou_line_from_suggestion(sug)
        if ln is None: return False
        if total >= ln - 1e-9: return False
    if sug.startswith("BTTS") and (gh>0 and ga>0): return False
    return True

def production_scan() -> Tuple[int,int]:
    matches=fetch_live_matches(); live_seen=len(matches)
    if live_seen==0: log.info("[PROD] no live"); return 0,0
    saved=0; now_ts=int(time.time())

    for m in matches:
        try:
            fid=int((m.get("fixture",{}) or {}).get("id") or 0)
            if not fid: continue

            # PATCH: short-lived connection just for the dup-check, instead
            # of holding one connection open for the whole per-match loop
            # (which also does blocking HTTP calls below: odds + Telegram).
            if DUP_COOLDOWN_MIN>0:
                cutoff=now_ts - DUP_COOLDOWN_MIN*60
                with db_conn() as c:
                    dup = c.execute("SELECT 1 FROM tips WHERE match_id=%s AND created_ts>=%s LIMIT 1",(fid,cutoff)).fetchone()
                if dup:
                    continue

            feat=extract_features(m); minute=int(feat.get("minute",0))
            if not stats_coverage_ok(feat, minute): continue
            if minute < TIP_MIN_MINUTE: continue
            if HARVEST_MODE and minute>=TRAIN_MIN_MINUTE and minute%3==0:
                try: save_snapshot_from_match(m, feat)
                except: pass

            league_id, league=_league_name(m); home,away=_teams(m); score=_pretty_score(m)
            candidates: List[Tuple[str,str,float]]=[]

            # OU
            for line in OU_LINES:
                mdl=_load_ou_model_for_line(line)
                if not mdl: continue
                p_over=_score_prob(feat, mdl)
                mk=f"Over/Under {_fmt_line(line)}"; thr=_get_market_threshold(mk)
                if p_over*100.0 >= thr and _candidate_is_sane(f"Over {_fmt_line(line)} Goals", feat):
                    candidates.append((mk, f"Over {_fmt_line(line)} Goals", p_over))
                p_under=1.0-p_over
                if p_under*100.0 >= thr and _candidate_is_sane(f"Under {_fmt_line(line)} Goals", feat):
                    candidates.append((mk, f"Under {_fmt_line(line)} Goals", p_under))

            # BTTS
            mdl_btts=load_model_from_settings("BTTS_YES")
            if mdl_btts:
                p=_score_prob(feat, mdl_btts); thr=_get_market_threshold("BTTS")
                if p*100.0>=thr and _candidate_is_sane("BTTS: Yes", feat): candidates.append(("BTTS","BTTS: Yes",p))
                q=1.0-p
                if q*100.0>=thr and _candidate_is_sane("BTTS: No", feat):  candidates.append(("BTTS","BTTS: No",q))

            # 1X2 (no draw)
            mh,md,ma=_load_wld_models()
            if mh and md and ma:
                ph=_score_prob(feat,mh); pd=_score_prob(feat,md); pa=_score_prob(feat,ma)
                # PATCH [BUG-FIX]: renormalize over Home+Away only. The draw
                # is suppressed from output entirely, so dividing by
                # (ph+pd+pa) was deflating both surviving probabilities and
                # silently suppressing tips that should have cleared
                # threshold. Draw-No-Bet style renormalization:
                s=max(EPS,ph+pa); ph,pa=ph/s,pa/s
                thr=_get_market_threshold("1X2")
                if ph*100.0>=thr: candidates.append(("1X2","Home Win",ph))
                if pa*100.0>=thr: candidates.append(("1X2","Away Win",pa))

            candidates.sort(key=lambda x:x[2], reverse=True)
            per_match=0; base_now=int(time.time())
            for idx,(market_txt,suggestion,prob) in enumerate(candidates):
                if suggestion not in ALLOWED_SUGGESTIONS: continue
                if per_match >= max(1,PREDICTIONS_PER_MATCH): break

                # Odds/EV gate (network call — no DB connection held here)
                pass_odds, odds, book, _ = _price_gate(market_txt, suggestion, fid)
                if not pass_odds:
                    continue
                ev_pct=None
                if odds is not None:
                    edge=_ev(prob, odds)  # decimal (e.g. 0.05)
                    ev_pct=round(edge*100.0,1)
                    if int(round(edge*10000)) < EDGE_MIN_BPS:  # basis points compare
                        continue

                created_ts=base_now+idx
                raw=float(prob); prob_pct=round(raw*100.0,1)

                # PATCH: short-lived connection just for the insert.
                with db_conn() as c:
                    c.execute(
                        "INSERT INTO tips(match_id,league_id,league,home,away,market,suggestion,confidence,confidence_raw,score_at_tip,minute,created_ts,odds,book,ev_pct,sent_ok) "
                        "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,0)",
                        (fid,league_id,league,home,away,market_txt,suggestion,float(prob_pct),raw,score,minute,created_ts,
                         (float(odds) if odds is not None else None), (book or None), (float(ev_pct) if ev_pct is not None else None))
                    )

                sent=_send_tip(home,away,league,minute,score,suggestion,float(prob_pct),feat,odds,book,ev_pct)
                if sent:
                    with db_conn() as c:
                        c.execute("UPDATE tips SET sent_ok=1 WHERE match_id=%s AND created_ts=%s",(fid,created_ts))

                saved+=1; per_match+=1
                if MAX_TIPS_PER_SCAN and saved>=MAX_TIPS_PER_SCAN: break
            if MAX_TIPS_PER_SCAN and saved>=MAX_TIPS_PER_SCAN: break
        except Exception as e:
            log.exception("[PROD] failure: %s", e)
            continue
    log.info("[PROD] saved=%d live_seen=%d", saved, live_seen)
    return saved, live_seen

# ───────── Prematch (compact: save-only, thresholds respected) ─────────
def _team_form_stats(team_id: int, games: List[dict]) -> Dict[str,float]:
    """
    PATCH: per-team form stats computed from the team's own perspective
    (goals for/against, W/D/L, last match date) — needed for
    pm_gf_*/pm_ga_*/pm_win_*/pm_draw_*/pm_loss_*/pm_rest_diff, none of
    which the previous implementation computed at all.
    """
    gf=ga=win=draw=loss=played=0
    last_ts=None
    for g in games:
        st=(((g.get("fixture") or {}).get("status") or {}).get("short") or "").upper()
        if st not in {"FT","AET","PEN"}: continue
        th=((g.get("teams") or {}).get("home") or {}).get("id")
        ta=((g.get("teams") or {}).get("away") or {}).get("id")
        gh=int((g.get("goals") or {}).get("home") or 0); ga_=int((g.get("goals") or {}).get("away") or 0)
        if team_id==th: my,opp=gh,ga_
        elif team_id==ta: my,opp=ga_,gh
        else: continue
        gf+=my; ga+=opp; played+=1
        if my>opp: win+=1
        elif my==opp: draw+=1
        else: loss+=1
        try:
            d=(g.get("fixture") or {}).get("date")
            if d:
                ts=datetime.fromisoformat(d.replace("Z","+00:00")).timestamp()
                if last_ts is None or ts>last_ts: last_ts=ts
        except Exception: pass
    if played==0:
        return {"gf":0.0,"ga":0.0,"win":0.0,"draw":0.0,"loss":0.0,"played":0,"last_ts":None}
    return {"gf":gf/played,"ga":ga/played,"win":win/played,"draw":draw/played,"loss":loss/played,"played":played,"last_ts":last_ts}

def _rate_totals(games: List[dict]) -> Tuple[float,float,float]:
    ov25=ov35=btts=played=0
    for g in games:
        st=(((g.get("fixture") or {}).get("status") or {}).get("short") or "").upper()
        if st not in {"FT","AET","PEN"}: continue
        gh=int((g.get("goals") or {}).get("home") or 0); ga=int((g.get("goals") or {}).get("away") or 0)
        played+=1
        if gh+ga>2: ov25+=1
        if gh+ga>3: ov35+=1
        if gh>0 and ga>0: btts+=1
    if played==0: return 0.0,0.0,0.0
    return ov25/played, ov35/played, btts/played

def _h2h_counts(h2h: List[dict], home_id: int, away_id: int) -> Tuple[float,float,float]:
    hw=aw=dr=played=0
    for g in h2h:
        st=(((g.get("fixture") or {}).get("status") or {}).get("short") or "").upper()
        if st not in {"FT","AET","PEN"}: continue
        th=((g.get("teams") or {}).get("home") or {}).get("id")
        gh=int((g.get("goals") or {}).get("home") or 0); ga=int((g.get("goals") or {}).get("away") or 0)
        played+=1
        if gh==ga: dr+=1
        else:
            winner_home = gh>ga
            winner_id = th if winner_home else ((g.get("teams") or {}).get("away") or {}).get("id")
            if winner_id==home_id: hw+=1
            elif winner_id==away_id: aw+=1
    if played==0: return 0.0,0.0,0.0
    return hw/played, aw/played, dr/played

def extract_prematch_features(fx: dict) -> Dict[str,float]:
    teams=fx.get("teams") or {}; th=(teams.get("home") or {}).get("id"); ta=(teams.get("away") or {}).get("id")
    if not th or not ta: return {}

    # PATCH: fetch last-5-home, last-5-away, and H2H concurrently instead of
    # sequentially — each is an independent HTTP call, and results are also
    # cached (TEAM_FORM_CACHE) so repeat calls across a scan / MOTD run are free.
    with ThreadPoolExecutor(max_workers=3) as ex:
        f_last_h = ex.submit(_api_last_fixtures, th, 5)
        f_last_a = ex.submit(_api_last_fixtures, ta, 5)
        f_h2h    = ex.submit(_api_h2h, th, ta, 5)
        last_h, last_a, h2h = f_last_h.result(), f_last_a.result(), f_h2h.result()

    ov25_h,ov35_h,btts_h=_rate_totals(last_h); ov25_a,ov35_a,btts_a=_rate_totals(last_a); ov25_h2h,ov35_h2h,btts_h2h=_rate_totals(h2h)
    hw_h2h,aw_h2h,dr_h2h=_h2h_counts(h2h, th, ta)

    form_h=_team_form_stats(th, last_h); form_a=_team_form_stats(ta, last_a)

    # PATCH: Elo ratings, backed by the team_ratings table (updated after
    # each result in backfill_results_for_open_matches()).
    ratings=get_team_ratings_bulk([th, ta])
    rating_h=ratings.get(th, ELO_DEFAULT); rating_a=ratings.get(ta, ELO_DEFAULT)

    form_points_h = form_h["win"]*3.0 + form_h["draw"]*1.0
    form_points_a = form_a["win"]*3.0 + form_a["draw"]*1.0

    # Rest days vs this fixture's kickoff
    rest_h = rest_a = 3.0  # neutral default when unknown
    try:
        kickoff_ts=datetime.fromisoformat(((fx.get("fixture") or {}).get("date") or "").replace("Z","+00:00")).timestamp()
        if form_h["last_ts"]: rest_h=max(0.0, (kickoff_ts - form_h["last_ts"])/86400.0)
        if form_a["last_ts"]: rest_a=max(0.0, (kickoff_ts - form_a["last_ts"])/86400.0)
    except Exception:
        pass

    attack_strength_h, defense_strength_h = form_h["gf"], form_h["ga"]
    attack_strength_a, defense_strength_a = form_a["gf"], form_a["ga"]
    expected_h = (attack_strength_h + defense_strength_a)/2.0
    expected_a = (attack_strength_a + defense_strength_h)/2.0
    expected_total = expected_h + expected_a
    rating_diff = rating_h - rating_a
    form_points_diff = form_points_h - form_points_a

    return {
        "pm_gf_h":form_h["gf"],"pm_ga_h":form_h["ga"],"pm_win_h":form_h["win"],"pm_draw_h":form_h["draw"],"pm_loss_h":form_h["loss"],
        "pm_gf_a":form_a["gf"],"pm_ga_a":form_a["ga"],"pm_win_a":form_a["win"],"pm_draw_a":form_a["draw"],"pm_loss_a":form_a["loss"],
        "pm_ov25_h":ov25_h,"pm_ov35_h":ov35_h,"pm_btts_h":btts_h,
        "pm_ov25_a":ov25_a,"pm_ov35_a":ov35_a,"pm_btts_a":btts_a,
        "pm_ov25_h2h":ov25_h2h,"pm_ov35_h2h":ov35_h2h,"pm_btts_h2h":btts_h2h,
        "pm_home_wins_h2h":hw_h2h,"pm_away_wins_h2h":aw_h2h,"pm_draws_h2h":dr_h2h,
        "pm_rating_h":rating_h,"pm_rating_a":rating_a,"pm_rating_diff":rating_diff,
        "pm_home_adv_rating":rating_h+ELO_HOME_ADV,"pm_away_adv_rating":rating_a,
        "pm_form_points_h":form_points_h,"pm_form_points_a":form_points_a,"pm_form_points_diff":form_points_diff,
        "pm_goal_difference_h":form_h["gf"]-form_h["ga"],"pm_goal_difference_a":form_a["gf"]-form_a["ga"],
        "pm_attack_strength_h":attack_strength_h,"pm_attack_strength_a":attack_strength_a,
        "pm_defense_strength_h":defense_strength_h,"pm_defense_strength_a":defense_strength_a,
        "pm_expected_total":expected_total,"pm_expected_total_diff":expected_h-expected_a,
        "pm_rest_diff":rest_h-rest_a,
        "pm_rating_form_interaction":rating_diff*form_points_diff,
        "pm_attack_defense_ratio":(attack_strength_h+attack_strength_a)/max(defense_strength_h+defense_strength_a, 0.1),
        # Live features are 0 for prematch (matches main.py's in-play keys)
        "minute":0.0,"goals_h":0.0,"goals_a":0.0,"goals_sum":0.0,"goals_diff":0.0,
        "xg_h":0.0,"xg_a":0.0,"xg_sum":0.0,"xg_diff":0.0,"sot_h":0.0,"sot_a":0.0,"sot_sum":0.0,
        "cor_h":0.0,"cor_a":0.0,"cor_sum":0.0,"pos_h":0.0,"pos_a":0.0,"pos_diff":0.0,"red_h":0.0,"red_a":0.0,"red_sum":0.0,
        "total_shots_h":0.0,"total_shots_a":0.0,"shots_inside_h":0.0,"shots_inside_a":0.0,"fouls_h":0.0,"fouls_a":0.0,
        "goals_per_minute":0.0,"xg_per_minute":0.0,"sot_per_minute":0.0,"shots_per_minute":0.0,"momentum_score":0.0,
        "shot_accuracy_h":0.0,"shot_accuracy_a":0.0,"shot_quality_h":0.0,"shot_quality_a":0.0,
        "conversion_rate_h":0.0,"conversion_rate_a":0.0,"xg_efficiency_h":0.0,"xg_efficiency_a":0.0,
        "attack_pressure_h":0.0,"attack_pressure_a":0.0,"attack_pressure_diff":0.0,
        "game_control_h":0.0,"game_control_a":0.0,"is_first_half":0.0,"is_second_half":0.0,"is_final_15":0.0,
        "score_margin":0.0,"is_leading_h":0.0,"is_leading_a":0.0,"is_draw":0.0,"is_goalfest":0.0,
        "fouls_per_minute":0.0,"discipline_score_h":0.0,"discipline_score_a":0.0,
        "possession_xg_interaction_h":0.0,"possession_xg_interaction_a":0.0,"sot_xg_ratio_h":0.0,"sot_xg_ratio_a":0.0,
        "match_minute_normalized":0.0,"time_weighted_xg_h":0.0,"time_weighted_xg_a":0.0,
        # internal, not a model feature — used by save_prematch_snapshot()
        "_home_id": float(th), "_away_id": float(ta),
    }

def _kickoff_berlin(utc_iso: str|None) -> str:
    try:
        if not utc_iso: return "TBD"
        dt=datetime.fromisoformat(utc_iso.replace("Z","+00:00"))
        return dt.astimezone(BERLIN_TZ).strftime("%H:%M")
    except: return "TBD"

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

def _send_tip(home,away,league,minute,score,suggestion,prob_pct,feat,odds=None,book=None,ev_pct=None)->bool:
    return send_telegram(_format_tip_message(home,away,league,minute,score,suggestion,prob_pct,feat,odds,book,ev_pct))

def prematch_scan_save() -> int:
    fixtures=_collect_todays_prematch_fixtures()
    if not fixtures: return 0
    saved=0

    # PATCH: extract features for all of today's fixtures concurrently
    # (bounded worker pool) instead of one fixture at a time — each
    # extraction does 3 (cached) HTTP calls, so this is the dominant cost
    # of a prematch scan.
    with ThreadPoolExecutor(max_workers=8) as ex:
        feats=list(ex.map(extract_prematch_features, fixtures))

    for fx, feat in zip(fixtures, feats):
        fixture=fx.get("fixture") or {}; lg=fx.get("league") or {}; teams=fx.get("teams") or {}
        home=(teams.get("home") or {}).get("name",""); away=(teams.get("away") or {}).get("name","")
        league_id=int((lg.get("id") or 0)); league=f"{lg.get('country','')} - {lg.get('name','')}".strip(" -"); fid=int((fixture.get("id") or 0))
        if not fid or not feat: continue
        try: save_prematch_snapshot(fid, feat)
        except Exception: pass
        candidates: List[Tuple[str,str,float]]=[]
        # PRE OU via PRE_OU_* models
        for line in OU_LINES:
            mdl=load_model_from_settings(f"PRE_OU_{_fmt_line(line)}")
            if not mdl: continue
            p=_score_prob(feat, mdl); mk=f"Over/Under {_fmt_line(line)}"; thr=_get_market_threshold_pre(mk)
            if p*100.0>=thr:   candidates.append((f"PRE {mk}", f"Over {_fmt_line(line)} Goals", p))
            q=1.0-p
            if q*100.0>=thr:   candidates.append((f"PRE {mk}", f"Under {_fmt_line(line)} Goals", q))
        # PRE BTTS
        mdl=load_model_from_settings("PRE_BTTS_YES")
        if mdl:
            p=_score_prob(feat, mdl); thr=_get_market_threshold_pre("BTTS")
            if p*100.0>=thr: candidates.append(("PRE BTTS","BTTS: Yes",p))
            q=1.0-p
            if q*100.0>=thr: candidates.append(("PRE BTTS","BTTS: No",q))
        # PRE 1X2 (draw suppressed — only home/away models exist, so no
        # renormalization bug here: s = ph+pa already)
        mh,ma=load_model_from_settings("PRE_WLD_HOME"), load_model_from_settings("PRE_WLD_AWAY")
        if mh and ma:
            ph=_score_prob(feat,mh); pa=_score_prob(feat,ma); s=max(EPS,ph+pa); ph,pa=ph/s,pa/s
            thr=_get_market_threshold_pre("1X2")
            if ph*100.0>=thr: candidates.append(("PRE 1X2","Home Win",ph))
            if pa*100.0>=thr: candidates.append(("PRE 1X2","Away Win",pa))
        if not candidates: continue
        candidates.sort(key=lambda x:x[2], reverse=True)
        base_now=int(time.time()); per_match=0
        for idx,(mk,sug,prob) in enumerate(candidates):
            if sug not in ALLOWED_SUGGESTIONS: continue
            if per_match>=max(1,PREDICTIONS_PER_MATCH): break
            # Odds/EV gate
            pass_odds, odds, book, _ = _price_gate(mk.replace("PRE ",""), sug, fid)
            if not pass_odds: continue
            ev_pct=None
            if odds is not None:
                edge=_ev(prob, odds); ev_pct=round(edge*100.0,1)
                if int(round(edge*10000)) < EDGE_MIN_BPS: continue
            created_ts=base_now+idx; raw=float(prob); pct=round(raw*100.0,1)
            with db_conn() as c2:
                c2.execute("INSERT INTO tips(match_id,league_id,league,home,away,market,suggestion,confidence,confidence_raw,score_at_tip,minute,created_ts,odds,book,ev_pct,sent_ok) "
                           "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,'0-0',0,%s,%s,%s,%s,0)",
                           (fid,league_id,league,home,away,mk,sug,float(pct),raw,created_ts,
                            (float(odds) if odds is not None else None), (book or None), (float(ev_pct) if ev_pct is not None else None)))
            saved+=1; per_match+=1
    log.info("[PREMATCH] saved=%d", saved); return saved

# ───────── Auto-train / tune / retry (unchanged signatures) ─────────
def auto_train_job():
    if not TRAIN_ENABLE: send_telegram("🤖 Training skipped: TRAIN_ENABLE=0"); return
    send_telegram("🤖 Training started.")
    try:
        res=train_models() or {}; ok=bool(res.get("ok"))
        if not ok:
            reason=res.get("reason") or res.get("error") or "unknown"
            send_telegram(f"⚠️ Training finished: <b>SKIPPED</b>\nReason: {escape(str(reason))}"); return
        trained=[k for k,v in (res.get("trained") or {}).items() if v]
        thr=(res.get("thresholds") or {}); mets=(res.get("metrics") or {})
        lines=["🤖 <b>Model training OK</b>"]
        if trained: lines.append("• Trained: " + ", ".join(sorted(trained)))
        if thr: lines.append("• Thresholds: " + "  |  ".join([f"{escape(k)}: {float(v):.1f}%" for k,v in thr.items()]))
        send_telegram("\n".join(lines))
    except Exception as e:
        log.exception("[TRAIN] job failed: %s", e); send_telegram(f"❌ Training <b>FAILED</b>\n{escape(str(e))}")

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

# Optional min EV for MOTD (basis points, e.g. 300 = +3.00%). 0 disables EV gate.
MOTD_MIN_EV_BPS = int(os.getenv("MOTD_MIN_EV_BPS", "0"))

def send_match_of_the_day() -> bool:
    """Pick the single best prematch tip for today (PRE_* models). Sends to Telegram."""
    fixtures = _collect_todays_prematch_fixtures()
    if not fixtures:
        return send_telegram("🏅 Match of the Day: no eligible fixtures today.")

    # Optional league allow-list just for MOTD
    if MOTD_LEAGUE_IDS:
        fixtures = [
            f for f in fixtures
            if int(((f.get("league") or {}).get("id") or 0)) in MOTD_LEAGUE_IDS
        ]
        if not fixtures:
            return send_telegram("🏅 Match of the Day: no fixtures in configured leagues.")

    # PATCH: same concurrent + cached feature extraction as prematch_scan_save().
    # If prematch_scan_save() already ran today, TEAM_FORM_CACHE hits mean
    # this loop makes ~zero additional team-form/H2H API calls.
    with ThreadPoolExecutor(max_workers=8) as ex:
        feats=list(ex.map(extract_prematch_features, fixtures))

    best = None  # (prob_pct, suggestion, home, away, league, kickoff_txt, odds, book, ev_pct)

    for fx, feat in zip(fixtures, feats):
        fixture = fx.get("fixture") or {}
        lg      = fx.get("league") or {}
        teams   = fx.get("teams") or {}
        fid     = int((fixture.get("id") or 0))

        home = (teams.get("home") or {}).get("name","")
        away = (teams.get("away") or {}).get("name","")
        league = f"{lg.get('country','')} - {lg.get('name','')}".strip(" -")
        kickoff_txt = _kickoff_berlin((fixture.get("date") or ""))

        if not feat:
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
            continue

        # Take the single best for this fixture (by probability) then apply odds/EV gate
        candidates.sort(key=lambda x: x[2], reverse=True)
        mk, sug, prob = candidates[0]
        prob_pct = prob * 100.0
        if prob_pct < MOTD_CONF_MIN:
            continue

        # Odds/EV (reuse in-play price gate; market text must be without "PRE ")
        pass_odds, odds, book, _ = _price_gate(mk, sug, fid)
        if not pass_odds:
            continue

        ev_pct = None
        if odds is not None:
            edge = _ev(prob, odds)            # decimal (e.g. 0.05)
            ev_bps = int(round(edge * 10000)) # basis points
            ev_pct = round(edge * 100.0, 1)
            if MOTD_MIN_EV_BPS > 0 and ev_bps < MOTD_MIN_EV_BPS:
                continue

        item = (prob_pct, sug, home, away, league, kickoff_txt, odds, book, ev_pct)
        if best is None or prob_pct > best[0]:
            best = item

    if not best:
        return send_telegram("🏅 Match of the Day: no prematch pick met thresholds.")
    prob_pct, sug, home, away, league, kickoff_txt, odds, book, ev_pct = best
    return send_telegram(_format_motd_message(home, away, league, kickoff_txt, sug, prob_pct, odds, book, ev_pct))

def auto_tune_thresholds(days: int = 14) -> Dict[str,float]:
    if not AUTO_TUNE_ENABLE: return {}
    cutoff=int(time.time())-days*24*3600
    with db_conn() as c:
        rows=c.execute("""
            SELECT t.market, t.suggestion, COALESCE(t.confidence_raw, t.confidence/100.0) prob,
                   r.final_goals_h, r.final_goals_a, r.btts_yes
            FROM tips t JOIN match_results r ON r.match_id=t.match_id
            WHERE t.created_ts >= %s AND t.suggestion<>'HARVEST' AND t.sent_ok=1
        """,(cutoff,)).fetchall()
    by={}
    for (mk,sugg,prob,gh,ga,btts) in rows:
        out=_tip_outcome_for_result(sugg, {"final_goals_h":gh,"final_goals_a":ga,"btts_yes":btts})
        if out is None: continue
        by.setdefault(mk, []).append((float(prob), int(out)))
    tuned={}
    for mk,arr in by.items():
        if len(arr)<THRESH_MIN_PREDICTIONS: continue
        probs=[p for (p,_) in arr]; wins=[y for (_,y) in arr]
        pct=_pick_threshold(wins, probs, TARGET_PRECISION, THRESH_MIN_PREDICTIONS, CONF_THRESHOLD)
        set_setting(f"conf_threshold:{mk}", f"{pct:.2f}"); _SETTINGS_CACHE.invalidate(f"conf_threshold:{mk}"); tuned[mk]=pct
    if tuned: send_telegram("🔧 Auto-tune updated thresholds:\n" + "\n".join([f"• {k}: {v:.1f}%" for k,v in tuned.items()]))
    else: send_telegram("🔧 Auto-tune: no updates (insufficient data).")
    return tuned

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

# ───────── Scheduler ─────────
def _run_with_pg_lock(lock_key: int, fn, *a, **k):
    try:
        with db_conn() as c:
            got=c.execute("SELECT pg_try_advisory_lock(%s)",(lock_key,)).fetchone()[0]
            if not got: log.info("[LOCK %s] busy; skipped.", lock_key); return None
            try: return fn(*a,**k)
            finally: c.execute("SELECT pg_advisory_unlock(%s)",(lock_key,))
    except Exception as e:
        log.exception("[LOCK %s] failed: %s", lock_key, e); return None

_scheduler_started=False
def _start_scheduler_once():
    global _scheduler_started
    if _scheduler_started or not RUN_SCHEDULER: return
    try:
        sched=BackgroundScheduler(timezone=TZ_UTC)
        sched.add_job(lambda:_run_with_pg_lock(1001,production_scan),"interval",seconds=SCAN_INTERVAL_SEC,id="scan",max_instances=1,coalesce=True)
        sched.add_job(lambda:_run_with_pg_lock(1002,backfill_results_for_open_matches,400),"interval",minutes=BACKFILL_EVERY_MIN,id="backfill",max_instances=1,coalesce=True)
        if PREMATCH_SCAN_ENABLE:
            sched.add_job(lambda:_run_with_pg_lock(1008,prematch_scan_save),"interval",minutes=PREMATCH_SCAN_INTERVAL_MIN,id="prematch_scan",max_instances=1,coalesce=True)
        if DAILY_ACCURACY_DIGEST_ENABLE:
            sched.add_job(lambda:_run_with_pg_lock(1003,daily_accuracy_digest),
                          CronTrigger(hour=DAILY_ACCURACY_HOUR, minute=DAILY_ACCURACY_MINUTE, timezone=BERLIN_TZ),
                          id="digest", max_instances=1, coalesce=True)
        if MOTD_PREDICT:
            sched.add_job(lambda:_run_with_pg_lock(1004,send_match_of_the_day),
                          CronTrigger(hour=int(os.getenv("MOTD_HOUR","19")), minute=int(os.getenv("MOTD_MINUTE","15")), timezone=BERLIN_TZ),
                          id="motd", max_instances=1, coalesce=True)
        if TRAIN_ENABLE:
            sched.add_job(lambda:_run_with_pg_lock(1005,auto_train_job),
                          CronTrigger(hour=TRAIN_HOUR_UTC, minute=TRAIN_MINUTE_UTC, timezone=TZ_UTC),
                          id="train", max_instances=1, coalesce=True)
        if AUTO_TUNE_ENABLE:
            sched.add_job(lambda:_run_with_pg_lock(1006,auto_tune_thresholds,14),
                          CronTrigger(hour=4, minute=7, timezone=TZ_UTC),
                          id="auto_tune", max_instances=1, coalesce=True)
        sched.add_job(lambda:_run_with_pg_lock(1007,retry_unsent_tips,30,200),"interval",minutes=10,id="retry",max_instances=1,coalesce=True)
        sched.start(); _scheduler_started=True
        send_telegram("🚀 goalsniper AI mode (in-play + prematch) started.")
        log.info("[SCHED] started (scan=%ss)", SCAN_INTERVAL_SEC)
    except Exception as e:
        log.exception("[SCHED] failed: %s", e)

_start_scheduler_once()

# ───────── Admin / auth ─────────
def _require_admin():
    # PATCH: request.json is deprecated in newer Flask/Werkzeug in favor of
    # request.get_json(silent=True) (also avoids raising on non-JSON bodies).
    body = request.get_json(silent=True) if request.is_json else None
    key=request.headers.get("X-API-Key") or request.args.get("key") or ((body or {}).get("key") if body else None)
    if not ADMIN_API_KEY or key != ADMIN_API_KEY: abort(401)

# ───────── HTTP endpoints ─────────
@app.route("/")
def root(): return jsonify({"ok": True, "name": "goalsniper", "mode": "FULL_AI", "scheduler": RUN_SCHEDULER})

@app.route("/health")
def health():
    try:
        with db_conn() as c:
            n=c.execute("SELECT COUNT(*) FROM tips").fetchone()[0]
        return jsonify({"ok": True, "db": "ok", "tips_count": int(n)})
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
    if not TRAIN_ENABLE: return jsonify({"ok": False, "reason": "training disabled"}), 400
    try: out=train_models(); return jsonify({"ok": True, "result": out})
    except Exception as e:
        log.exception("train_models failed: %s", e); return jsonify({"ok": False, "error": str(e)}), 500

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
        if msg.startswith("/start"): send_telegram("👋 goalsniper bot (FULL AI mode) is online.")
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
    _init_pool(); init_db(); set_setting("boot_ts", str(int(time.time())))

_on_boot()

if __name__ == "__main__":
    app.run(host=os.getenv("HOST","0.0.0.0"), port=int(os.getenv("PORT","8080")))
