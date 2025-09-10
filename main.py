# file: main.py
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

# ───────── Core env (secrets: required; knobs: defaultable) — UPDATED ─────────
TELEGRAM_BOT_TOKEN = _require_env("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = _require_env("TELEGRAM_CHAT_ID")
API_KEY            = _require_env("API_KEY")
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

# ───────── Caches & timezones (set to Europe/Amsterdam) — UPDATED ─────────
STATS_CACHE:  Dict[int, Tuple[float, list]] = {}
EVENTS_CACHE: Dict[int, Tuple[float, list]] = {}
ODDS_CACHE:   Dict[int, Tuple[float, dict]] = {}
SETTINGS_TTL = int(os.getenv("SETTINGS_TTL_SEC","60"))
MODELS_TTL   = int(os.getenv("MODELS_CACHE_TTL_SEC","120"))
TZ_UTC, BERLIN_TZ = ZoneInfo("UTC"), ZoneInfo("Europe/Amsterdam")  # NOTE: BERLIN_TZ now points to Amsterdam

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
class PooledConn:
    def __init__(self, pool): self.pool=pool; self.conn=None; self.cur=None
    def __enter__(self): self.conn=self.pool.getconn(); self.conn.autocommit=True; self.cur=self.conn.cursor(); return self
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

def _init_pool():
    global POOL
    dsn = DATABASE_URL + (("&" if "?" in DATABASE_URL else "?") + "sslmode=require" if "sslmode=" not in DATABASE_URL else "")
    POOL = SimpleConnectionPool(minconn=1, maxconn=int(os.getenv("DB_POOL_MAX","5")), dsn=dsn)

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
    today_local=datetime.now(ZoneInfo("Europe/Amsterdam")).date()
    start_local=datetime.combine(today_local, datetime.min.time(), tzinfo=ZoneInfo("Europe/Amsterdam"))
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
    sh_total_a = _num(sa.get("Total Shots", sh.get("Shots Total", 0)))
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

    # Markets
    _do("BTTS",                 df["market"] == "BTTS")
    _do("Over/Under 2.5",      df["market"].eq("Over/Under 2.5"))
    _do("Over/Under 3.5",      df["market"].eq("Over/Under 3.5"))
    _do("1X2",                 df["market"] == "1X2")

    if updates:
        log.info("Tips calibration/threshold updates: %s", updates)  # FIX: logger -> log
    return updates
# ---- end optional block ----

def stats_coverage_ok(feat: Dict[str,float], minute: int) -> bool:
    require_stats_minute = int(os.getenv("REQUIRE_STATS_MINUTE","35"))
    require_fields = int(os.getenv("REQUIRE_DATA_FIELDS","2"))
    if minute < require_stats_minute:
        return True
    # Allow shots/yellows to satisfy coverage when xG is missing
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

def fetch_odds(fid: int) -> dict:
    """
    Returns a dict like:
      { "BTTS": {...}, "1X2": {...}, "OU_2.5": {...}, ... }
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
        ln_val = _parse_ou_line_from_suggestion(suggestion)
        d = odds_map.get(f"OU_{_fmt_line(ln_val)}", {}) if ln_val is not None else {}
        tgt = "Over" if suggestion.startswith("Over") else "Under"
        if tgt in d:
            odds = d[tgt]["odds"]
            book = d[tgt]["book"]

    if odds is None:
        # Honor flag: allow tips to pass when odds are missing (EV check will be skipped)
        return (ALLOW_TIPS_WITHOUT_ODDS, None, None, None)

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
    gh = int(feat.get("goals_h", 0))
    ga = int(feat.get("goals_a", 0))
    total = gh + ga

    if sug.startswith("Over"):
        ln = _parse_ou_line_from_suggestion(sug)
        # Only send Over if current total is strictly below the line
        return (ln is not None) and (total < ln)

    if sug.startswith("Under"):
        ln = _parse_ou_line_from_suggestion(sug)
        # Only send Under if current total is strictly below the line (if already >= line, it's too late)
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
    if minute < 10:
        st = _FEED_STATE.get(fid)
        fp = _match_fingerprint(m)
        _FEED_STATE[fid] = {"fp": fp, "last_change": time.time(), "last_minute": minute}
        return False

    now = time.time()
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

def production_scan() -> Tuple[int, int]:
    matches = fetch_live_matches()
    live_seen = len(matches)
    if live_seen == 0:
        log.info("[PROD] no live")
        return 0, 0

    saved = 0
    now_ts = int(time.time())
    per_league_counter: dict[int, int] = {}

    with db_conn() as c:
        for m in matches:
            try:
                fid = int((m.get("fixture", {}) or {}).get("id") or 0)
                if not fid:
                    continue

                if DUP_COOLDOWN_MIN > 0:
                    cutoff = now_ts - DUP_COOLDOWN_MIN * 60
                    # Ignore HARVEST rows in cooldown check
                    if c.execute(
                        "SELECT 1 FROM tips WHERE match_id=%s AND created_ts>=%s AND suggestion<>'HARVEST' LIMIT 1",
                        (fid, cutoff),
                    ).fetchone():
                        continue

                feat = extract_features(m)
                minute = int(feat.get("minute", 0))
                if not stats_coverage_ok(feat, minute):
                    continue
                if minute < TIP_MIN_MINUTE:
                    continue

                if is_feed_stale(fid, m, minute):
                    continue

                if HARVEST_MODE and minute >= TRAIN_MIN_MINUTE and minute % 3 == 0:
                    try:
                        save_snapshot_from_match(m, feat)
                    except Exception:
                        pass

                league_id, league = _league_name(m)
                home, away = _teams(m)
                score = _pretty_score(m)

                candidates: List[Tuple[str, str, float]] = []

                # OU
                for line in OU_LINES:
                    mdl = _load_ou_model_for_line(line)
                    if not mdl:
                        continue
                    mk = f"Over/Under {_fmt_line(line)}"
                    thr = _get_market_threshold(mk)

                    p_over = _score_prob(feat, mdl)
                    sug_over = f"Over {_fmt_line(line)} Goals"
                    if (
                        p_over * 100.0 >= thr
                        and _candidate_is_sane(sug_over, feat)
                        and market_cutoff_ok(minute, mk, sug_over)
                    ):
                        candidates.append((mk, sug_over, p_over))

                    p_under = 1.0 - p_over
                    sug_under = f"Under {_fmt_line(line)} Goals"
                    if (
                        p_under * 100.0 >= thr
                        and _candidate_is_sane(sug_under, feat)
                        and market_cutoff_ok(minute, mk, sug_under)
                    ):
                        candidates.append((mk, sug_under, p_under))

                # BTTS
                mdl_btts = load_model_from_settings("BTTS_YES")
                if mdl_btts:
                    mk = "BTTS"
                    thr = _get_market_threshold(mk)

                    p_yes = _score_prob(feat, mdl_btts)
                    if (
                        p_yes * 100.0 >= thr
                        and _candidate_is_sane("BTTS: Yes", feat)
                        and market_cutoff_ok(minute, mk, "BTTS: Yes")
                    ):
                        candidates.append((mk, "BTTS: Yes", p_yes))

                    p_no = 1.0 - p_yes
                    if (
                        p_no * 100.0 >= thr
                        and _candidate_is_sane("BTTS: No", feat)
                        and market_cutoff_ok(minute, mk, "BTTS: No")
                    ):
                        candidates.append((mk, "BTTS: No", p_no))

                # 1X2 (draw suppressed)
                mh, md, ma = _load_wld_models()
                if mh and md and ma:
                    mk = "1X2"
                    thr = _get_market_threshold(mk)
                    ph = _score_prob(feat, mh)
                    pd = _score_prob(feat, md)
                    pa = _score_prob(feat, ma)
                    s = max(EPS, ph + pd + pa)
                    ph, pa = ph / s, pa / s

                    if ph * 100.0 >= thr and market_cutoff_ok(minute, mk, "Home Win"):
                        candidates.append((mk, "Home Win", ph))
                    if pa * 100.0 >= thr and market_cutoff_ok(minute, mk, "Away Win"):
                        candidates.append((mk, "Away Win", pa))

                if not candidates:
                    continue

                odds_map = fetch_odds(fid) if API_KEY else {}
                ranked: List[Tuple[str, str, float, Optional[float], Optional[str], Optional[float], float]] = []

                for mk, sug, prob in candidates:
                    if sug not in ALLOWED_SUGGESTIONS:
                        continue

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

                    rank_score = (prob ** 1.2) * (1 + (ev_pct or 0) / 100.0)
                    ranked.append((mk, sug, prob, odds, book, ev_pct, rank_score))

                if not ranked:
                    continue

                ranked.sort(key=lambda x: x[6], reverse=True)

                per_match = 0
                base_now = int(time.time())

                for idx, (market_txt, suggestion, prob, odds, book, ev_pct, _rank) in enumerate(ranked):
                    if PER_LEAGUE_CAP > 0 and per_league_counter.get(league_id, 0) >= PER_LEAGUE_CAP:
                        continue

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
                                    0,   # sent_ok initially false
                                ),
                             )

                            sent = _send_tip(home, away, league, minute, score, suggestion, float(prob_pct), feat, odds, book, ev_pct)
                            if sent:
                                c2.execute("UPDATE tips SET sent_ok=1 WHERE match_id=%s AND created_ts=%s", (fid, created_ts))
                    except Exception as e:
                        log.exception("[PROD] insert/send failed: %s", e)
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
                log.exception("[PROD] match loop failed: %s", e)
                continue

    log.info("[PROD] saved=%d live_seen=%d", saved, live_seen)
    return saved, live_seen

# ───────── Prematch (compact: save-only, thresholds respected) ─────────
def extract_prematch_features(fx: dict) -> Dict[str,float]:
    teams=fx.get("teams") or {}; th=(teams.get("home") or {}).get("id"); ta=(teams.get("away") or {}).get("id")
    if not th or not ta: return {}
    def _rate(games):
        ov25=ov35=btts=played=0
        for g in games:
            st=(((g.get("fixture") or {}).get("status") or {}).get("short") or "").upper()
            if st not in {"FT","AET","PEN"}: continue
            gh=int((g.get("goals") or {}).get("home") or 0); ga=int((g.get("goals") or {}).get("away") or 0)
            played+=1; 
            if gh+ga>2: ov25+=1
            if gh+ga>3: ov35+=1
            if gh>0 and ga>0: btts+=1
        if played==0: return 0,0,0
        return ov25/played, ov35/played, btts/played
    last_h=_api_last_fixtures(th,5); last_a=_api_last_fixtures(ta,5); h2h=_api_h2h(th,ta,5)
    ov25_h,ov35_h,btts_h=_rate(last_h); ov25_a,ov35_a,btts_a=_rate(last_a); ov25_h2h,ov35_h2h,btts_h2h=_rate(h2h)
    return {"pm_ov25_h":ov25_h,"pm_ov35_h":ov35_h,"pm_btts_h":btts_h,
            "pm_ov25_a":ov25_a,"pm_ov35_a":ov35_a,"pm_btts_a":btts_a,
            "pm_ov25_h2h":ov25_h2h,"pm_ov35_h2h":ov35_h2h,"pm_btts_h2h":btts_h2h,
            "minute":0.0,"goals_h":0.0,"goals_a":0.0,"goals_sum":0.0,"goals_diff":0.0,
            "xg_h":0.0,"xg_a":0.0,"xg_sum":0.0,"xg_diff":0.0,"sot_h":0.0,"sot_a":0.0,"sot_sum":0.0,
            "cor_h":0.0,"cor_a":0.0,"cor_sum":0.0,"pos_h":0.0,"pos_a":0.0,"pos_diff":0.0,"red_h":0.0,"red_a":0.0,"red_sum":0.0}

def _kickoff_berlin(utc_iso: str|None) -> str:  # name kept for compatibility; uses Amsterdam tz
    try:
        if not utc_iso: return "TBD"
        dt=datetime.fromisoformat(utc_iso.replace("Z","+00:00"))
        return dt.astimezone(BERLIN_TZ).strftime("%H:%M")  # BERLIN_TZ is Europe/Amsterdam
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
        f"⏰ <b>Kickoff (Amsterdam):</b> {kickoff_txt}\n"
        f"<b>Tip:</b> {escape(suggestion)}\n"
        f"📈 <b>Confidence:</b> {prob_pct:.1f}%{money}"
    )

def _send_tip(home,away,league,minute,score,suggestion,prob_pct,feat,odds=None,book=None,ev_pct=None)->bool:
    return send_telegram(_format_tip_message(home,away,league,minute,score,suggestion,prob_pct,feat,odds,book,ev_pct))

def prematch_scan_save() -> int:
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
                ev_pct = round(edge * 100.0, 1)
                if int(round(edge * 10000)) < EDGE_MIN_BPS:
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

def backfill_prematch_snapshots(days: int = 7, limit_per_day: int = 800) -> int:
    if not API_KEY:
        return 0
    wrote = 0
    tz = ZoneInfo("Europe/Amsterdam")
    now_local = datetime.now(tz).replace(hour=0, minute=0, second=0, microsecond=0)
    for d in range(days):
        day = (now_local - timedelta(days=d)).date()
        js = _api_get(FOOTBALL_API_URL, {"date": day.strftime("%Y-%m-%d")}) or {}
        fixtures = []
        if isinstance(js, dict):
            for r in js.get("response", []):
                status = ((r.get("fixture") or {}).get("status") or {}).get("short") or ""
                if status.upper() == "NS":
                    fixtures.append(r)
        fixtures = [f for f in fixtures if not _blocked_league(f.get("league") or {})][:limit_per_day]
        for fx in fixtures:
            feat = extract_prematch_features(fx)
            if not feat:
                continue
            try:
                save_prematch_snapshot(fx, feat)
                wrote += 1
            except Exception:
                continue
    return wrote

# ───────── Auto-train / tune / retry (unchanged signatures) ─────────
def auto_train_job():
    if not TRAIN_ENABLE:
        send_telegram("🤖 Training skipped: TRAIN_ENABLE=0")
        return
    send_telegram("🤖 Training started.")
    try:
        res = train_models() or {}
        ok = bool(res.get("ok"))
        if not ok:
            reason = res.get("reason") or res.get("error") or "unknown"
            send_telegram(f"⚠️ Training finished: <b>SKIPPED</b>\nReason: {escape(str(reason))}")
            return

        # What the trainer reported
        trained = [k for k, v in (res.get("trained") or {}).items() if v]
        mets    = (res.get("metrics") or {})

        # Always fetch the live thresholds actually stored in DB,
        # so the message shows BTTS/OU too (not just 1X2).
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

MOTD_MIN_EV_BPS = int(os.getenv("MOTD_MIN_EV_BPS", "0"))

def send_match_of_the_day() -> bool:
    if not MOTD_PREDICT:
        return send_telegram("🏅 MOTD disabled.")
    fixtures = _collect_todays_prematch_fixtures()
    if not fixtures:
        return send_telegram("🏅 Match of the Day: no eligible fixtures today.")

    if MOTD_LEAGUE_IDS:
        fixtures = [f for f in fixtures if int(((f.get("league") or {}).get("id") or 0)) in MOTD_LEAGUE_IDS]
        if not fixtures:
            return send_telegram("🏅 Match of the Day: no fixtures in configured leagues.")

    best = None

    for fx in fixtures:
        fixture = fx.get("fixture") or {}
        lg      = fx.get("league") or {}
        teams   = fx.get("teams") or {}
        fid     = int((fixture.get("id") or 0))

        home = (teams.get("home") or {}).get("name","")
        away = (teams.get("away") or {}).get("name","")
        league = f"{lg.get('country','')} - {lg.get('name','')}".strip(" -")
        kickoff_txt = _kickoff_berlin((fixture.get("date") or ""))

        feat = extract_prematch_features(fx)
        if not feat:
            continue

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

        candidates.sort(key=lambda x: x[2], reverse=True)
        mk, sug, prob = candidates[0]
        prob_pct = prob * 100.0
        if prob_pct < max(MOTD_CONF_MIN, 75):  # stricter floor
            continue

        pass_odds, odds, book, _ = _price_gate(mk, sug, fid)
        if not pass_odds:
            continue

        ev_pct = None
        if odds is not None:
            edge = _ev(prob, odds)
            ev_bps = int(round(edge * 10000))
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
    key=request.headers.get("X-API-Key") or request.args.get("key") or ((request.json or {}).get("key") if request.is_json else None)
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

@app.route("/admin/backfill-prematch", methods=["POST","GET"])
def http_backfill_prematch():
    _require_admin()
    days = int(request.args.get("days", "7"))
    n = backfill_prematch_snapshots(days=days)
    return jsonify({"ok": True, "snapshots_written": int(n)})

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
