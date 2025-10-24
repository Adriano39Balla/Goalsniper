"""
goalsniper — FULL AI mode (in-play + prematch) with odds + EV gate.

- Pure ML (calibrated) loaded from Postgres settings (train_models.py).
- Markets: OU(2.5,3.5), BTTS (Yes/No), 1X2 (Draw suppressed).
- Adds bookmaker odds filtering + EV check.
- Scheduler: scan, results backfill, nightly train, daily digest, MOTD.

Safe to run on Railway/Render. Requires DATABASE_URL and API keys.
"""

import os, json, time, logging, requests, psycopg2
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

# >>> ADDED: for IPv4 resolution & DSN helpers
import socket
from urllib.parse import urlparse, parse_qsl
# <<<

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

# ───────── Required envs (fail fast) ─────────
def _require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise SystemExit(f"Missing required environment variable: {name}")
    return v

# Enforce hard requirements
DATABASE_URL = _require_env("DATABASE_URL")
API_KEY = _require_env("API_KEY")
TELEGRAM_BOT_TOKEN = _require_env("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = _require_env("TELEGRAM_CHAT_ID")

# Optional-but-strongly-recommended (don’t fail; just warn if missing)
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")
WEBHOOK_SECRET = os.getenv("TELEGRAM_WEBHOOK_SECRET")
if not ADMIN_API_KEY:
    log.warning("ADMIN_API_KEY is not set — /admin/* endpoints are less protected.")
if not WEBHOOK_SECRET:
    log.warning("TELEGRAM_WEBHOOK_SECRET is not set — webhook endpoint would be unsafe if exposed.")

# ───────── Core env ─────────
RUN_SCHEDULER      = os.getenv("RUN_SCHEDULER", "1") not in ("0","false","False","no","NO")
CONF_THRESHOLD     = float(os.getenv("CONF_THRESHOLD", "70"))
MAX_TIPS_PER_SCAN  = int(os.getenv("MAX_TIPS_PER_SCAN", "25"))
DUP_COOLDOWN_MIN   = int(os.getenv("DUP_COOLDOWN_MIN", "20"))
TIP_MIN_MINUTE     = int(os.getenv("TIP_MIN_MINUTE", "8"))
SCAN_INTERVAL_SEC  = int(os.getenv("SCAN_INTERVAL_SEC", "300"))

TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
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
PER_LEAGUE_CAP        = int(os.getenv("PER_LEAGUE_CAP", "0"))  # FIX: previously undefined

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
session.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1, status_forcelist=[429,500,502,503,504], respect_retry_after_header=True)))

# ───────── Caches & timezones ─────────
STATS_CACHE:  Dict[int, Tuple[float, list]] = {}
EVENTS_CACHE: Dict[int, Tuple[float, list]] = {}
ODDS_CACHE:   Dict[int, Tuple[float, dict]] = {}
SETTINGS_TTL = int(os.getenv("SETTINGS_TTL_SEC","60"))
MODELS_TTL   = int(os.getenv("MODELS_CACHE_TTL_SEC","120"))
TZ_UTC, BERLIN_TZ = ZoneInfo("UTC"), ZoneInfo("Europe/Berlin")

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
def _resolve_ipv4(host: str, port: int) -> Optional[str]:
    """
    Prefer a local AF_INET resolve; if that fails, try DNS-over-HTTPS (Google/Cloudflare).
    Returns the first IPv4 string or None.
    """
    if not host:
        return None
    # 1) Local resolver (A record)
    try:
        infos = socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_STREAM)
        for _family, _socktype, _proto, _canonname, sockaddr in infos:
            ip, _p = sockaddr
            return ip
    except Exception as e:
        log.warning("[DNS] IPv4 resolve failed for %s:%s: %s — trying DoH fallback", host, port, e)
    # 2) DoH fallback
    try:
        urls = [
            f"https://dns.google/resolve?name={host}&type=A",
            f"https://cloudflare-dns.com/dns-query?name={host}&type=A",
        ]
        for u in urls:
            r = requests.get(u, headers={"accept": "application/dns-json"}, timeout=4)
            if not r.ok:
                continue
            data = r.json() if isinstance(r.json, type(lambda: None)) else r.json()
            for ans in (data or {}).get("Answer", []) or []:
                ip = ans.get("data")
                # quick sanity: dot-quad IPv4
                if isinstance(ip, str) and ip.count(".") == 3:
                    return ip
    except Exception as e:
        log.warning("[DNS] DoH fallback failed for %s: %s — using hostname fallback", host, e)
    return None

def _parse_pg_url(url: str) -> Dict[str, Any]:
    pr = urlparse(url)
    if pr.scheme not in ("postgresql", "postgres"):
        raise SystemExit("DATABASE_URL must start with postgresql:// or postgres://")
    user = pr.username or ""
    password = pr.password or ""
    host = pr.hostname or ""
    port = pr.port or 5432
    dbname = (pr.path or "").lstrip("/") or "postgres"
    params = dict(parse_qsl(pr.query))
    params.setdefault("sslmode", "require")
    return {"user": user, "password": password, "host": host, "port": int(port), "dbname": dbname, "params": params}

def _q(v: str) -> str:
    s = "" if v is None else str(v)
    if s == "" or all(ch not in s for ch in (" ", "'", "\\", "\t", "\n")):
        return s
    s = s.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{s}'"

def _make_conninfo(parts: Dict[str, Any], port: int, hostaddr: Optional[str]) -> str:
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

def _conninfo_candidates(url: str) -> List[str]:
    """
    Prefer pooled (6543) first, then original port (often 5432).
    For each, try with IPv4 hostaddr (if we can resolve), then without.
    Allow pinning via DB_HOSTADDR=<ipv4>.
    """
    parts = _parse_pg_url(url)
    env_hostaddr = os.getenv("DB_HOSTADDR")
    prefer_pooled = os.getenv("DB_PREFER_POOLED", "1") not in ("0", "false", "False", "no", "NO")

    ports: List[int] = []
    if prefer_pooled:
        ports.append(6543)
    if parts["port"] not in ports:
        ports.append(parts["port"])

    cands: List[str] = []
    for p in ports:
        ipv4 = env_hostaddr or _resolve_ipv4(parts["host"], p)
        if ipv4:
            cands.append(_make_conninfo(parts, p, ipv4))
        cands.append(_make_conninfo(parts, p, None))
    return cands

POOL: Optional[SimpleConnectionPool] = None

def _init_pool():
    """Create a small global pool with retry/backoff and IPv4/pooled preference."""
    global POOL
    if POOL:
        return
    maxconn = int(os.getenv("DB_POOL_MAX", "5"))
    candidates = _conninfo_candidates(DATABASE_URL)

    delay = 1.0
    last = "unknown"
    for attempt in range(6):  # ~1+2+4+8+16 (~31s)
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
                "DB pool init failed after retries. "
                f"Last error: {last}. "
                "Hint: set DB_HOSTADDR=<your Supabase IPv4> to pin IPv4 if DNS A-records are flaky."
            )
        time.sleep(delay)
        delay *= 2

class PooledConn:
    def __init__(self, pool): self.pool=pool; self.conn=None; self.cur=None
    def __enter__(self):
        _init_pool()
        try:
            self.conn=self.pool.getconn()
        except Exception:
            global POOL
            POOL=None
            _init_pool()
            self.conn=self.pool.getconn()
        self.conn.autocommit=True
        self.cur=self.conn.cursor()
        return self
    def __exit__(self, a,b,c): 
        try: self.cur and self.cur.close()
        finally: self.conn and self.pool.putconn(self.conn)
    def execute(self, sql: str, params: tuple|list=()):
        try:
            self.cur.execute(sql, params or ())
            return self.cur
        except Exception as e:
            log.error("DB execute failed: %s\nSQL: %s\nParams: %s", e, sql, params)
            raise

def db_conn(): 
    if not POOL: _init_pool()
    return PooledConn(POOL)  # type: ignore

# ───────── Settings cache ─────────
class _TTLCache:
    def __init__(self, ttl): self.ttl=ttl; self.data={}
    def get(self, k): 
        v=self.data.get(k); 
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
    today_local=datetime.now(ZoneInfo("Europe/Berlin")).date()
    start_local=datetime.combine(today_local, datetime.min.time(), tzinfo=ZoneInfo("Europe/Berlin"))
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
    cutoff = min(cutoff, 80)
    return m <= int(cutoff)

def fetch_odds(fid: int) -> dict:
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
                if mname=="BTTS":
                    d={}
                    for v in vals:
                        lbl=(v.get("value") or "").strip().lower()
                        if "yes" in lbl: d["Yes"]={"odds":float(v.get("odd") or 0), "book":book_name}
                        if "no"  in lbl: d["No"] ={"odds":float(v.get("odd") or 0), "book":book_name}
                    if d: out["BTTS"]=d
                elif mname=="1X2":
                    d={}
                    for v in vals:
                        lbl=(v.get("value") or "").strip().lower()
                        if lbl in ("home","1"): d["Home"]={"odds":float(v.get("odd") or 0),"book":book_name}
                        if lbl in ("away","2"): d["Away"]={"odds":float(v.get("odd") or 0),"book":book_name}
                    if d: out["1X2"]=d
                elif mname=="OU":
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
        return (False, None, None, None)  # require odds for precision

    min_odds=_min_odds_for_market(market_text)
    if not (min_odds <= odds <= MAX_ODDS_ALL):
        return (False, odds, book, None)

    return (True, odds, book, None)

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
            "INSERT INTO tips(match_id,league_id,league,home,away,market,suggestion,confidence,confidence_raw,score_at_tip,minute,created_ts,sent_ok) "
            "VALUES (%s,%s,%s,%s,%s,'HARVEST','HARVEST',0.0,0.0,%s,%s,%s,1)",
            (fid, league_id, league, home, away, f"{gh}-{ga}", minute, now)
        )

# (rest of your logic unchanged — production_scan, prematch, scheduler, endpoints)

# ───────── Boot ─────────
def _on_boot():
    _init_pool(); init_db(); set_setting("boot_ts", str(int(time.time())))

_on_boot()

if __name__ == "__main__":
    app.run(host=os.getenv("HOST","0.0.0.0"), port=int(os.getenv("PORT","8080")))
