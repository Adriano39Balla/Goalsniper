"""
goalsniper — FULL AI mode (in-play + prematch) with advanced features.

Enhanced with sophisticated features:
- Game state and momentum metrics
- Team strength ratings (ELO-inspired)
- Advanced form metrics with exponential decay
- Shot quality and efficiency metrics
- Learning system to adapt from mistakes
"""

import os, json, time, math, logging, requests, psycopg2
import numpy as np
from psycopg2.pool import SimpleConnectionPool
from html import escape
from zoneinfo import ZoneInfo
from scipy.stats import norm, binom_test
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

AUTO_TUNE_ENABLE        = os.getenv("AUTO_TUNE_ENABLE", "1") not in ("0","false","False","no","NO")
TARGET_PRECISION        = float(os.getenv("TARGET_PRECISION", "0.60"))
THRESH_MIN_PREDICTIONS  = int(os.getenv("THRESH_MIN_PREDICTIONS", "25"))
MIN_THRESH              = float(os.getenv("MIN_THRESH", "55"))
MAX_THRESH              = float(os.getenv("MAX_THRESH", "85"))

# Learning system settings
LEARNING_ENABLE         = os.getenv("LEARNING_ENABLE", "1") not in ("0","false","False","no","NO")
LEARNING_HISTORY_DAYS   = int(os.getenv("LEARNING_HISTORY_DAYS", "30"))
LEARNING_MIN_SAMPLES    = int(os.getenv("LEARNING_MIN_SAMPLES", "100"))
LEARNING_UPDATE_HOUR    = int(os.getenv("LEARNING_UPDATE_HOUR", "1"))
LEARNING_UPDATE_MINUTE  = int(os.getenv("LEARNING_UPDATE_MINUTE", "0"))

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
session.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1, status_forcelist=[429,500,502,503,504], respect_retry_after_header=True)))

# ───────── Caches & timezones ─────────
STATS_CACHE:  Dict[int, Tuple[float, list]] = {}
EVENTS_CACHE: Dict[int, Tuple[float, list]] = {}
ODDS_CACHE:   Dict[int, Tuple[float, dict]] = {}
SETTINGS_TTL = int(os.getenv("SETTINGS_TTL_SEC","60"))
MODELS_TTL   = int(os.getenv("MODELS_CACHE_TTL_SEC","120"))
TZ_UTC, BERLIN_TZ = ZoneInfo("UTC"), ZoneInfo("Europe/Berlin")

# ───────── Team ELO Ratings Cache ─────────
TEAM_RATINGS_CACHE: Dict[int, Dict[str, float]] = {}  # team_id -> {rating, home_rating, away_rating, last_updated}
TEAM_RATINGS_TTL = 3600  # 1 hour

# ───────── Learning System Cache ─────────
LEARNING_PATTERNS_CACHE: Dict[str, Dict[str, Any]] = {}  # pattern_key -> pattern_data
LEARNING_CACHE_TTL = 3600  # 1 hour

# ───────── Optional import: trainer ─────────
try:
    from train_models import train_models
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
        self.cur.execute(sql, params or ()); return self.cur

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
        c.execute("""CREATE TABLE IF NOT EXISTS prematch_snapshots (
            match_id BIGINT PRIMARY KEY, created_ts BIGINT, payload TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS feedback (
            id SERIAL PRIMARY KEY, match_id BIGINT UNIQUE, verdict INTEGER, created_ts BIGINT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS match_results (
            match_id BIGINT PRIMARY KEY, final_goals_h INTEGER, final_goals_a INTEGER, btts_yes INTEGER, updated_ts BIGINT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS team_ratings (
            team_id BIGINT PRIMARY KEY, rating DOUBLE PRECISION, home_rating DOUBLE PRECISION,
            away_rating DOUBLE PRECISION, matches_played INTEGER, last_updated BIGINT)""")
        # Learning system tables
        c.execute("""CREATE TABLE IF NOT EXISTS error_patterns (
            id SERIAL PRIMARY KEY,
            pattern_type TEXT NOT NULL,
            pattern_key TEXT NOT NULL,
            error_rate DOUBLE PRECISION,
            samples_count INTEGER,
            last_detected_ts BIGINT,
            corrective_action TEXT,
            created_ts BIGINT,
            UNIQUE(pattern_type, pattern_key)
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS performance_log (
            id SERIAL PRIMARY KEY,
            log_date DATE NOT NULL,
            market TEXT NOT NULL,
            total_tips INTEGER,
            successful_tips INTEGER,
            accuracy_rate DOUBLE PRECISION,
            avg_confidence DOUBLE PRECISION,
            created_ts BIGINT,
            UNIQUE(log_date, market)
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
        c.execute("CREATE INDEX IF NOT EXISTS idx_prematch_by_match ON prematch_snapshots (match_id, created_ts DESC)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_results_updated ON match_results (updated_ts DESC)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_team_ratings_updated ON team_ratings (last_updated DESC)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_error_patterns ON error_patterns (pattern_type, error_rate DESC)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_performance_log ON performance_log (log_date DESC, market)")

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

# ───────── Enhanced League filter ─────────
_BLOCK_PATTERNS = ["u17","u18","u19","u20","u21","u23","youth","junior","reserve","res.","friendlies","friendly","academy","development","u-"]
def _blocked_league(league_obj: dict) -> bool:
    """
    Enhanced league blocking with better pattern matching and logging
    """
    if not league_obj:
        return False
    
    name = str(league_obj.get("name","")).lower()
    country = str(league_obj.get("country","")).lower()
    typ = str(league_obj.get("type","")).lower()
    
    # Combine all text for pattern matching
    combined_text = f"{country} {name} {typ}"
    
    # Check for youth/reserve/friendly patterns
    for pattern in _BLOCK_PATTERNS:
        if pattern in combined_text:
            log.debug(f"Blocked league: {combined_text} (matched pattern: {pattern})")
            return True
    
    # Check league ID deny list from environment
    deny_ids = [x.strip() for x in os.getenv("LEAGUE_DENY_IDS","").split(",") if x.strip().isdigit()]
    league_id = str(league_obj.get("id") or "")
    
    if league_id in deny_ids:
        log.debug(f"Blocked league ID: {league_id} (in deny list)")
        return True
    
    # Additional checks for specific league characteristics
    # Check for youth indicators in name
    youth_indicators = ["youth", "junior", "u19", "u18", "u17", "u20", "u21", "u23", "academy", "reserve", "b team"]
    for indicator in youth_indicators:
        if indicator in name.lower():
            log.debug(f"Blocked league: {name} (contains youth indicator: {indicator})")
            return True
    
    # Check league type
    if typ in ["Cup", "Friendlies"]:
        log.debug(f"Blocked league type: {typ} for {name}")
        return True
    
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

# ───────── Team Rating System (ELO-inspired) ─────────
def calculate_elo_update(winner_rating: float, loser_rating: float, home_advantage: float = 0.0, k_factor: float = 32.0) -> Tuple[float, float]:
    """Calculate ELO rating changes for two teams"""
    # Expected score for team A
    expected_a = 1 / (1 + 10 ** ((loser_rating - winner_rating + home_advantage) / 400))
    # Actual score (1 for win, 0 for loss)
    actual_a = 1.0
    # Rating change
    change = k_factor * (actual_a - expected_a)
    return change, -change

def calculate_confidence_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple:
    """Calculate Wilson score interval for binomial proportion"""
    if trials == 0:
        return (0, 0)
    
    p = successes / trials
    z = norm.ppf(1 - (1 - confidence) / 2)
    
    denominator = 1 + z**2 / trials
    centre_adjusted_probability = p + z**2 / (2 * trials)
    adjusted_standard_deviation = math.sqrt(
        (p * (1 - p) + z**2 / (4 * trials)) / trials
    )
    
    lower_bound = (
        centre_adjusted_probability - z * adjusted_standard_deviation
    ) / denominator
    
    upper_bound = (
        centre_adjusted_probability + z * adjusted_standard_deviation
    ) / denominator
    
    return (max(0, lower_bound), min(1, upper_bound))

def update_team_rating(team_id: int, opponent_rating: float, result: int, is_home: bool, k_factor: float = 32.0):
    """Update team rating after a match (result: 1=win, 0.5=draw, 0=loss)"""
    with db_conn() as c:
        # Get current rating
        row = c.execute("SELECT rating, home_rating, away_rating, matches_played FROM team_ratings WHERE team_id=%s", (team_id,)).fetchone()
        
        if row:
            rating, home_rating, away_rating, matches = row[0], row[1], row[2], row[3]
        else:
            # Initialize with base rating
            rating = 1500.0
            home_rating = 1500.0
            away_rating = 1500.0
            matches = 0
        
        # Adjust K-factor based on matches played (higher for new teams)
        effective_k = k_factor * (40.0 / max(matches, 1)) if matches < 40 else k_factor
        
        # Calculate expected score
        expected = 1 / (1 + 10 ** ((opponent_rating - rating) / 400))
        
        # Calculate rating change
        change = effective_k * (result - expected)
        new_rating = rating + change
        
        # Update home/away rating
        if is_home:
            home_rating = home_rating + change * 0.5  # Less impact on specialized rating
        else:
            away_rating = away_rating + change * 0.5
        
        # Save to database
        c.execute("""
            INSERT INTO team_ratings (team_id, rating, home_rating, away_rating, matches_played, last_updated)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (team_id) DO UPDATE SET
                rating = EXCLUDED.rating,
                home_rating = EXCLUDED.home_rating,
                away_rating = EXCLUDED.away_rating,
                matches_played = EXCLUDED.matches_played,
                last_updated = EXCLUDED.last_updated
        """, (team_id, new_rating, home_rating, away_rating, matches + 1, int(time.time())))
        
        return new_rating

def get_team_rating(team_id: int) -> Dict[str, float]:
    """Get team rating from cache or database"""
    now = time.time()
    
    # Check cache
    if team_id in TEAM_RATINGS_CACHE:
        cached = TEAM_RATINGS_CACHE[team_id]
        if now - cached.get("last_updated", 0) < TEAM_RATINGS_TTL:
            return cached
    
    # Fetch from database
    with db_conn() as c:
        row = c.execute("SELECT rating, home_rating, away_rating, matches_played FROM team_ratings WHERE team_id=%s", (team_id,)).fetchone()
    
    if row:
        rating_data = {
            "rating": float(row[0] or 1500.0),
            "home_rating": float(row[1] or 1500.0),
            "away_rating": float(row[2] or 1500.0),
            "matches_played": int(row[3] or 0),
            "last_updated": now
        }
    else:
        rating_data = {
            "rating": 1500.0,
            "home_rating": 1500.0,
            "away_rating": 1500.0,
            "matches_played": 0,
            "last_updated": now
        }
    
    # Update cache
    TEAM_RATINGS_CACHE[team_id] = rating_data
    return rating_data

def update_ratings_from_finished_match(match_data: dict):
    """Update ELO ratings for both teams after a finished match"""
    fixture = match_data.get("fixture") or {}
    teams = match_data.get("teams") or {}
    goals = match_data.get("goals") or {}
    
    home_id = (teams.get("home") or {}).get("id")
    away_id = (teams.get("away") or {}).get("id")
    home_goals = int(goals.get("home") or 0)
    away_goals = int(goals.get("away") or 0)
    
    if not home_id or not away_id:
        return
    
    # Get current ratings
    home_rating_data = get_team_rating(home_id)
    away_rating_data = get_team_rating(away_id)
    
    home_rating = home_rating_data["rating"]
    away_rating = away_rating_data["rating"]
    
    # Determine result
    if home_goals > away_goals:
        home_result = 1.0
        away_result = 0.0
    elif home_goals < away_goals:
        home_result = 0.0
        away_result = 1.0
    else:
        home_result = 0.5
        away_result = 0.5
    
    # Update ratings
    update_team_rating(home_id, away_rating, home_result, is_home=True)
    update_team_rating(away_id, home_rating, away_result, is_home=False)

# ───────── Prematch helpers ─────────
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
    dates_utc={start_local.astimezone(TZ_UTC).date(), (end_local - timedelta(seconds=1)).astimezone(TZ_UTC).date()}
    fixtures=[]
    for d in sorted(dates_utc):
        js=_api_get(FOOTBALL_API_URL, {"date": d.strftime("%Y-%m-%d")}) or {}
        for r in js.get("response",[]) if isinstance(js,dict) else []:
            if (((r.get("fixture") or {}).get("status") or {}).get("short") or "").upper() == "NS":
                fixtures.append(r)
    fixtures=[f for f in fixtures if not _blocked_league(f.get("league") or {})]
    return fixtures

# ───────── Feature extraction with Advanced Features ─────────
def _num(v) -> float:
    try:
        if isinstance(v,str) and v.endswith("%"): return float(v[:-1])
        return float(v or 0)
    except: return 0.0

def _pos_pct(v) -> float:
    try: return float(str(v).replace("%","").strip() or 0)
    except: return 0.0

def extract_features(m: dict) -> Dict[str,float]:
    """Extract live match features including advanced metrics"""
    home=m["teams"]["home"]["name"]; away=m["teams"]["away"]["name"]
    gh=m["goals"]["home"] or 0; ga=m["goals"]["away"] or 0
    minute=int(((m.get("fixture") or {}).get("status") or {}).get("elapsed") or 0)
    
    stats={}; 
    for s in (m.get("statistics") or []):
        t=(s.get("team") or {}).get("name"); 
        if t: stats[t]={ (i.get("type") or ""): i.get("value") for i in (s.get("statistics") or []) }
    
    sh=stats.get(home,{}) or {}; sa=stats.get(away,{}) or {}
    
    # Basic stats
    xg_h=_num(sh.get("Expected Goals",0)); xg_a=_num(sa.get("Expected Goals",0))
    sot_h=_num(sh.get("Shots on Target",0)); sot_a=_num(sa.get("Shots on Target",0))
    cor_h=_num(sh.get("Corner Kicks",0)); cor_a=_num(sa.get("Corner Kicks",0))
    pos_h=_pos_pct(sh.get("Ball Possession",0)); pos_a=_pos_pct(sa.get("Ball Possession",0))
    
    # Additional stats for advanced features
    total_shots_h = _num(sh.get("Total Shots", 0))
    total_shots_a = _num(sa.get("Total Shots", 0))
    shots_inside_box_h = _num(sh.get("Shots insidebox", 0))
    shots_inside_box_a = _num(sa.get("Shots insidebox", 0))
    fouls_h = _num(sh.get("Fouls", 0))
    fouls_a = _num(sa.get("Fouls", 0))
    
    red_h=red_a=0
    for ev in (m.get("events") or []):
        if (ev.get("type","").lower()=="card"):
            d=(ev.get("detail","") or "").lower()
            if "red" in d or "second yellow" in d:
                t=(ev.get("team") or {}).get("name") or ""
                if t==home: red_h+=1
                elif t==away: red_a+=1
    
    # Basic features
    features = {
        "minute": float(minute),
        "goals_h": float(gh), "goals_a": float(ga), 
        "goals_sum": float(gh+ga), "goals_diff": float(gh-ga),
        "xg_h": float(xg_h), "xg_a": float(xg_a), 
        "xg_sum": float(xg_h+xg_a), "xg_diff": float(xg_h-xg_a),
        "sot_h": float(sot_h), "sot_a": float(sot_a), 
        "sot_sum": float(sot_h+sot_a),
        "cor_h": float(cor_h), "cor_a": float(cor_a), 
        "cor_sum": float(cor_h+cor_a),
        "pos_h": float(pos_h), "pos_a": float(pos_a), 
        "pos_diff": float(pos_h-pos_a),
        "red_h": float(red_h), "red_a": float(red_a), 
        "red_sum": float(red_h+red_a),
        
        # Additional basic stats
        "total_shots_h": float(total_shots_h), "total_shots_a": float(total_shots_a),
        "shots_inside_h": float(shots_inside_box_h), "shots_inside_a": float(shots_inside_box_a),
        "fouls_h": float(fouls_h), "fouls_a": float(fouls_a),
    }
    
    # ───────── ADVANCED FEATURES ─────────
    
    # 1. Game State & Momentum Features
    if minute > 0:
        # Rate of events per minute
        features["goals_per_minute"] = features["goals_sum"] / minute
        features["xg_per_minute"] = features["xg_sum"] / minute
        features["sot_per_minute"] = features["sot_sum"] / minute
        features["shots_per_minute"] = (total_shots_h + total_shots_a) / minute
        
        # Recent momentum (last 15 minutes proxy)
        # Note: This is simplified - ideally would track event timestamps
        features["momentum_score"] = (features["xg_per_minute"] * 0.5 + 
                                      features["sot_per_minute"] * 0.3 + 
                                      features["shots_per_minute"] * 0.2)
    else:
        features["goals_per_minute"] = 0.0
        features["xg_per_minute"] = 0.0
        features["sot_per_minute"] = 0.0
        features["shots_per_minute"] = 0.0
        features["momentum_score"] = 0.0
    
    # 2. Shot Quality & Efficiency Features
    # Shot accuracy
    features["shot_accuracy_h"] = sot_h / max(total_shots_h, 1)
    features["shot_accuracy_a"] = sot_a / max(total_shots_a, 1)
    
    # Shot quality (proportion of shots inside box)
    features["shot_quality_h"] = shots_inside_box_h / max(total_shots_h, 1)
    features["shot_quality_a"] = shots_inside_box_a / max(total_shots_a, 1)
    
    # Conversion rates
    features["conversion_rate_h"] = gh / max(sot_h, 1)
    features["conversion_rate_a"] = ga / max(sot_a, 1)
    
    # xG efficiency (over/underperformance)
    features["xg_efficiency_h"] = gh - xg_h
    features["xg_efficiency_a"] = ga - xg_a
    
    # 3. Pressure & Game Control Features
    # Attack pressure score
    features["attack_pressure_h"] = (sot_h * 0.4 + xg_h * 0.4 + cor_h * 0.2)
    features["attack_pressure_a"] = (sot_a * 0.4 + xg_a * 0.4 + cor_a * 0.2)
    features["attack_pressure_diff"] = features["attack_pressure_h"] - features["attack_pressure_a"]
    
    # Game control score (possession-adjusted)
    features["game_control_h"] = (pos_h / 100) * features["attack_pressure_h"]
    features["game_control_a"] = (pos_a / 100) * features["attack_pressure_a"]
    
    # 4. Game Phase Features
    features["is_first_half"] = 1.0 if minute <= 45 else 0.0
    features["is_second_half"] = 1.0 if minute > 45 else 0.0
    features["is_final_15"] = 1.0 if minute > 75 else 0.0
    
    # 5. Score State Features
    features["score_margin"] = abs(gh - ga)
    features["is_leading_h"] = 1.0 if gh > ga else 0.0
    features["is_leading_a"] = 1.0 if ga > gh else 0.0
    features["is_draw"] = 1.0 if gh == ga else 0.0
    features["is_goalfest"] = 1.0 if gh + ga >= 3 else 0.0
    
    # 6. Discipline & Foul Features
    features["fouls_per_minute"] = (fouls_h + fouls_a) / max(minute, 1)
    features["discipline_score_h"] = 1.0 / max(fouls_h + red_h * 10, 1)
    features["discipline_score_a"] = 1.0 / max(fouls_a + red_a * 10, 1)
    
    # 7. Cross-Feature Interactions
    # These are often predictive as they capture non-linear relationships
    features["possession_xg_interaction_h"] = (pos_h / 100) * xg_h
    features["possession_xg_interaction_a"] = (pos_a / 100) * xg_a
    features["sot_xg_ratio_h"] = sot_h / max(xg_h, 0.1)
    features["sot_xg_ratio_a"] = sot_a / max(xg_a, 0.1)
    
    # 8. Match Context Features
    features["match_minute_normalized"] = minute / 90.0
    features["time_weighted_xg_h"] = xg_h * (minute / 90.0)
    features["time_weighted_xg_a"] = xg_a * (minute / 90.0)
    
    return features

def extract_prematch_features(fx: dict) -> Dict[str,float]:
    """Extract prematch features with advanced metrics"""
    teams=fx.get("teams") or {}
    home_team=teams.get("home") or {}; away_team=teams.get("away") or {}
    home_id=home_team.get("id"); away_id=away_team.get("id")
    
    if not home_id or not away_id:
        return {}
    
    # Fetch recent form with exponential decay weighting
    last_h=_api_last_fixtures(home_id,5) if home_id else []
    last_a=_api_last_fixtures(away_id,5) if away_id else []
    h2h=_api_h2h(home_id, away_id,5) if home_id and away_id else []
    
    # ───────── ADVANCED FORM CALCULATION WITH EXPONENTIAL DECAY ─────────
    def calculate_weighted_form(matches: List[dict], is_home_for_team: bool = True) -> Dict[str, float]:
        """Calculate form metrics with exponential decay weighting (recent matches more important)"""
        if not matches:
            return {
                "gf": 0.0, "ga": 0.0, "win": 0.0, "draw": 0.0, "loss": 0.0,
                "ov25": 0.0, "ov35": 0.0, "btts": 0.0,
                "points": 0.0, "xg_for": 0.0, "xg_against": 0.0
            }
        
        # Exponential decay weights: most recent gets weight 1.0, then 0.8, 0.64, 0.512, 0.41
        weights = [0.8 ** i for i in range(len(matches))]  # [1.0, 0.8, 0.64, 0.512, 0.41]
        total_weight = sum(weights)
        
        gf = ga = wins = draws = losses = 0.0
        ov25 = ov35 = btts = 0.0
        points = 0.0
        
        for i, match in enumerate(reversed(matches)):  # Reverse to get chronological order
            st = (((match.get("fixture") or {}).get("status") or {}).get("short") or "").upper()
            if st not in {"FT", "AET", "PEN"}:
                continue
                
            goals = match.get("goals") or {}
            gh = int(goals.get("home") or 0)
            ga_match = int(goals.get("away") or 0)
            
            # Determine if this team is home or away in this match
            teams_match = match.get("teams") or {}
            home_team_match = (teams_match.get("home") or {}).get("id")
            
            if is_home_for_team:
                # Team is home in these matches
                gf += gh * weights[i]
                ga += ga_match * weights[i]
                if gh > ga_match:
                    wins += weights[i]
                    points += 3 * weights[i]
                elif gh == ga_match:
                    draws += weights[i]
                    points += 1 * weights[i]
                else:
                    losses += weights[i]
            else:
                # Team is away in these matches
                gf += ga_match * weights[i]
                ga += gh * weights[i]
                if ga_match > gh:
                    wins += weights[i]
                    points += 3 * weights[i]
                elif ga_match == gh:
                    draws += weights[i]
                    points += 1 * weights[i]
                else:
                    losses += weights[i]
            
            total = gh + ga_match
            if total > 2:
                ov25 += weights[i]
            if total > 3:
                ov35 += weights[i]
            if gh > 0 and ga_match > 0:
                btts += weights[i]
        
        return {
            "gf": gf / total_weight if total_weight > 0 else 0.0,
            "ga": ga / total_weight if total_weight > 0 else 0.0,
            "win": wins / total_weight if total_weight > 0 else 0.0,
            "draw": draws / total_weight if total_weight > 0 else 0.0,
            "loss": losses / total_weight if total_weight > 0 else 0.0,
            "ov25": ov25 / total_weight if total_weight > 0 else 0.0,
            "ov35": ov35 / total_weight if total_weight > 0 else 0.0,
            "btts": btts / total_weight if total_weight > 0 else 0.0,
            "points": points / total_weight if total_weight > 0 else 0.0
        }
    
    # Calculate weighted form
    home_form = calculate_weighted_form(last_h, is_home_for_team=True)
    away_form = calculate_weighted_form(last_a, is_home_for_team=False)
    
    # Calculate H2H stats
    h2h_stats = {"ov25": 0.0, "ov35": 0.0, "btts": 0.0, "home_wins": 0.0, "away_wins": 0.0, "draws": 0.0}
    h2h_games = 0
    for match in h2h:
        st = (((match.get("fixture") or {}).get("status") or {}).get("short") or "").upper()
        if st not in {"FT", "AET", "PEN"}:
            continue
            
        goals = match.get("goals") or {}
        gh = int(goals.get("home") or 0)
        ga = int(goals.get("away") or 0)
        total = gh + ga
        
        h2h_stats["ov25"] += 1 if total > 2 else 0
        h2h_stats["ov35"] += 1 if total > 3 else 0
        h2h_stats["btts"] += 1 if gh > 0 and ga > 0 else 0
        
        if gh > ga:
            h2h_stats["home_wins"] += 1
        elif ga > gh:
            h2h_stats["away_wins"] += 1
        else:
            h2h_stats["draws"] += 1
        h2h_games += 1
    
    if h2h_games > 0:
        h2h_stats = {k: v / h2h_games for k, v in h2h_stats.items()}
    
    # Get team ratings
    home_rating_data = get_team_rating(home_id)
    away_rating_data = get_team_rating(away_id)
    
    # Calculate advanced features
    home_rating = home_rating_data["rating"]
    away_rating = away_rating_data["rating"]
    home_home_rating = home_rating_data["home_rating"]
    away_away_rating = away_rating_data["away_rating"]
    
    # ───────── BUILD ADVANCED FEATURE DICTIONARY ─────────
    features = {
        # Team form features (weighted)
        "pm_gf_h": home_form["gf"], "pm_ga_h": home_form["ga"],
        "pm_win_h": home_form["win"], "pm_draw_h": home_form["draw"], "pm_loss_h": home_form["loss"],
        "pm_gf_a": away_form["gf"], "pm_ga_a": away_form["ga"],
        "pm_win_a": away_form["win"], "pm_draw_a": away_form["draw"], "pm_loss_a": away_form["loss"],
        
        # Over/Under features (weighted)
        "pm_ov25_h": home_form["ov25"], "pm_ov35_h": home_form["ov35"], "pm_btts_h": home_form["btts"],
        "pm_ov25_a": away_form["ov25"], "pm_ov35_a": away_form["ov35"], "pm_btts_a": away_form["btts"],
        
        # H2H features
        "pm_ov25_h2h": h2h_stats.get("ov25", 0.0), "pm_ov35_h2h": h2h_stats.get("ov35", 0.0),
        "pm_btts_h2h": h2h_stats.get("btts", 0.0), "pm_home_wins_h2h": h2h_stats.get("home_wins", 0.0),
        "pm_away_wins_h2h": h2h_stats.get("away_wins", 0.0), "pm_draws_h2h": h2h_stats.get("draws", 0.0),
        
        # Team Strength Features (ELO-based)
        "pm_rating_h": home_rating, "pm_rating_a": away_rating,
        "pm_rating_diff": home_rating - away_rating,
        "pm_home_adv_rating": home_home_rating - home_rating,  # Home advantage
        "pm_away_adv_rating": away_away_rating - away_rating,  # Away performance
        
        # Advanced Form Metrics
        "pm_form_points_h": home_form["points"], "pm_form_points_a": away_form["points"],
        "pm_form_points_diff": home_form["points"] - away_form["points"],
        "pm_goal_difference_h": home_form["gf"] - home_form["ga"],
        "pm_goal_difference_a": away_form["gf"] - away_form["ga"],
        
        # Attack vs Defense Strength
        "pm_attack_strength_h": home_form["gf"] / max(home_form["ga"], 0.1),
        "pm_attack_strength_a": away_form["gf"] / max(away_form["ga"], 0.1),
        "pm_defense_strength_h": 1.0 / max(home_form["ga"], 0.1),
        "pm_defense_strength_a": 1.0 / max(away_form["ga"], 0.1),
        
        # Expected Goals Proxy
        "pm_expected_total": (home_form["gf"] + away_form["gf"]) / 2,
        "pm_expected_total_diff": home_form["gf"] - away_form["gf"],
        
        # Rest days (placeholder - would need fixture dates)
        "pm_rest_diff": 0.0,
        
        # Live features (set to 0 for prematch)
        "minute": 0.0,
        "goals_h": 0.0, "goals_a": 0.0, "goals_sum": 0.0, "goals_diff": 0.0,
        "xg_h": 0.0, "xg_a": 0.0, "xg_sum": 0.0, "xg_diff": 0.0,
        "sot_h": 0.0, "sot_a": 0.0, "sot_sum": 0.0,
        "cor_h": 0.0, "cor_a": 0.0, "cor_sum": 0.0,
        "pos_h": 0.0, "pos_a": 0.0, "pos_diff": 0.0,
        "red_h": 0.0, "red_a": 0.0, "red_sum": 0.0,
        
        # Advanced live features (set to 0 for prematch)
        "total_shots_h": 0.0, "total_shots_a": 0.0,
        "shots_inside_h": 0.0, "shots_inside_a": 0.0,
        "fouls_h": 0.0, "fouls_a": 0.0,
        "goals_per_minute": 0.0, "xg_per_minute": 0.0, "sot_per_minute": 0.0,
        "shots_per_minute": 0.0, "momentum_score": 0.0,
        "shot_accuracy_h": 0.0, "shot_accuracy_a": 0.0,
        "shot_quality_h": 0.0, "shot_quality_a": 0.0,
        "conversion_rate_h": 0.0, "conversion_rate_a": 0.0,
        "xg_efficiency_h": 0.0, "xg_efficiency_a": 0.0,
        "attack_pressure_h": 0.0, "attack_pressure_a": 0.0,
        "attack_pressure_diff": 0.0,
        "game_control_h": 0.0, "game_control_a": 0.0,
        "is_first_half": 0.0, "is_second_half": 0.0, "is_final_15": 0.0,
        "score_margin": 0.0, "is_leading_h": 0.0, "is_leading_a": 0.0,
        "is_draw": 0.0, "is_goalfest": 0.0,
        "fouls_per_minute": 0.0, "discipline_score_h": 0.0, "discipline_score_a": 0.0,
        "possession_xg_interaction_h": 0.0, "possession_xg_interaction_a": 0.0,
        "sot_xg_ratio_h": 0.0, "sot_xg_ratio_a": 0.0,
        "match_minute_normalized": 0.0, "time_weighted_xg_h": 0.0, "time_weighted_xg_a": 0.0,
    }
    
    # Calculate interaction features
    features["pm_rating_form_interaction"] = features["pm_rating_diff"] * features["pm_form_points_diff"]
    features["pm_attack_defense_ratio"] = features["pm_attack_strength_h"] / max(features["pm_defense_strength_a"], 0.1)
    
    return features

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

def _price_gate(market_text: str, suggestion: str, fid: int, probability: float) -> Tuple[bool, Optional[float], Optional[str], Optional[float]]:
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
    
    # Calculate EV
    ev_pct = (_ev(probability, odds) * 100.0) if probability > 0 else None

    return (True, odds, book, ev_pct)

# ───────── Snapshots ─────────
def save_snapshot_from_match(m: dict, feat: Dict[str,float]) -> None:
    fx=m.get("fixture",{}) or {}; lg=m.get("league",{}) or {}
    fid=int(fx.get("id")); league_id=int(lg.get("id") or 0)
    league=f"{lg.get('country','')} - {lg.get('name','')}".strip(" -")
    home=(m.get("teams") or {}).get("home",{}).get("name","")
    away=(m.get("teams") or {}).get("away",{}).get("name","")
    gh=(m.get("goals") or {}).get("home") or 0; ga=(m.get("goals") or {}).get("away") or 0
    minute=int(feat.get("minute",0))
    
    # Include advanced features in snapshot
    snapshot={
        "minute": minute,
        "gh": gh, "ga": ga,
        "league_id": league_id,
        "market": "HARVEST",
        "suggestion": "HARVEST",
        "confidence": 0,
        "stat": {
            "xg_h": feat.get("xg_h",0), "xg_a": feat.get("xg_a",0),
            "sot_h": feat.get("sot_h",0), "sot_a": feat.get("sot_a",0),
            "cor_h": feat.get("cor_h",0), "cor_a": feat.get("cor_a",0),
            "pos_h": feat.get("pos_h",0), "pos_a": feat.get("pos_a",0),
            "red_h": feat.get("red_h",0), "red_a": feat.get("red_a",0),
            "total_shots_h": feat.get("total_shots_h",0), "total_shots_a": feat.get("total_shots_a",0),
            "shots_inside_h": feat.get("shots_inside_h",0), "shots_inside_a": feat.get("shots_inside_a",0),
            "fouls_h": feat.get("fouls_h",0), "fouls_a": feat.get("fouls_a",0),
        },
        "advanced": {
            "goals_per_minute": feat.get("goals_per_minute",0),
            "xg_per_minute": feat.get("xg_per_minute",0),
            "sot_per_minute": feat.get("sot_per_minute",0),
            "shot_accuracy_h": feat.get("shot_accuracy_h",0),
            "shot_accuracy_a": feat.get("shot_accuracy_a",0),
            "attack_pressure_h": feat.get("attack_pressure_h",0),
            "attack_pressure_a": feat.get("attack_pressure_a",0),
        }
    }
    
    now=int(time.time())
    with db_conn() as c:
        c.execute("INSERT INTO tip_snapshots(match_id, created_ts, payload) VALUES (%s,%s,%s) "
                  "ON CONFLICT (match_id, created_ts) DO UPDATE SET payload=EXCLUDED.payload",
                  (fid, now, json.dumps(snapshot)[:200000]))
        c.execute("INSERT INTO tips(match_id,league_id,league,home,away,market,suggestion,confidence,confidence_raw,score_at_tip,minute,created_ts,sent_ok) "
                  "VALUES (%s,%s,%s,%s,%s,'HARVEST','HARVEST',0.0,0.0,%s,%s,%s,1)",
                  (fid, league_id, league, home, away, f"{gh}-{ga}", minute, now))

def save_prematch_snapshot(fid: int, feat: Dict[str, float]) -> None:
    """Save prematch features for training"""
    now = int(time.time())
    snapshot = {"feat": feat}
    with db_conn() as c:
        c.execute(
            "INSERT INTO prematch_snapshots(match_id, created_ts, payload) VALUES (%s,%s,%s) "
            "ON CONFLICT (match_id) DO UPDATE SET created_ts=EXCLUDED.created_ts, payload=EXCLUDED.payload",
            (fid, now, json.dumps(snapshot)[:200000])
        )

# ───────── Outcomes/backfill/digest ─────────
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
        
        # Update team ratings from finished match
        try:
            update_ratings_from_finished_match(fx)
        except Exception as e:
            log.warning("Failed to update ratings for match %s: %s", mid, e)
        
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
    # Include advanced stats if available
    advanced_stats = ""
    if any([feat.get("xg_h",0), feat.get("xg_a",0), feat.get("attack_pressure_h",0), feat.get("attack_pressure_a",0)]):
        advanced_stats = (f"\n📊 xG {feat.get('xg_h',0):.2f}-{feat.get('xg_a',0):.2f}"
                         f" • SOT {int(feat.get('sot_h',0))}-{int(feat.get('sot_a',0))}"
                         f" • Attack Pressure: {feat.get('attack_pressure_h',0):.1f}-{feat.get('attack_pressure_a',0):.1f}")
        if feat.get("shot_accuracy_h",0) or feat.get("shot_accuracy_a",0):
            advanced_stats += f"\n🎯 Shot Accuracy: {feat.get('shot_accuracy_h',0)*100:.0f}%-{feat.get('shot_accuracy_a',0)*100:.0f}%"
    
    money = ""
    if odds:
        if ev_pct is not None:
            money = f"\n💰 <b>Odds:</b> {odds:.2f} @ {book or 'Book'}  •  <b>EV:</b> {ev_pct:+.1f}%"
        else:
            money = f"\n💰 <b>Odds:</b> {odds:.2f} @ {book or 'Book'}"
    
    return ("⚽️ <b>New Tip!</b> [ADVANCED]\n"
            f"<b>Match:</b> {escape(home)} vs {escape(away)}\n"
            f"🕒 <b>Minute:</b> {minute}'  |  <b>Score:</b> {escape(score)}\n"
            f"<b>Tip:</b> {escape(suggestion)}\n"
            f"📈 <b>Confidence:</b> {prob_pct:.1f}%{money}\n"
            f"🏆 <b>League:</b> {escape(league)}{advanced_stats}")

# ───────── Scan (in-play) ─────────
def _candidate_is_sane(sug: str, feat: Dict[str,float]) -> bool:
    gh=int(feat.get("goals_h",0)); ga=int(feat.get("goals_a",0)); total=gh+ga
    if sug.startswith("Over"):
        ln=_parse_ou_line_from_suggestion(sug)
        if ln is None: return False
        if total > ln - 1e-9: return False  # Already over the line
    if sug.startswith("Under"):
        ln=_parse_ou_line_from_suggestion(sug)
        if ln is None: return False
        if total >= ln - 1e-9: return False  # Already at or over the line
    if sug.startswith("BTTS") and (gh>0 and ga>0): return False  # BTTS already happened
    return True

def production_scan() -> Tuple[int,int]:
    matches=fetch_live_matches(); live_seen=len(matches)
    if live_seen==0: log.info("[PROD] no live"); return 0,0
    saved=0; now_ts=int(time.time())
    with db_conn() as c:
        for m in matches:
            try:
                fid=int((m.get("fixture",{}) or {}).get("id") or 0)
                if not fid: continue
                if DUP_COOLDOWN_MIN>0:
                    cutoff=now_ts - DUP_COOLDOWN_MIN*60
                    if c.execute("SELECT 1 FROM tips WHERE match_id=%s AND created_ts>=%s LIMIT 1",(fid,cutoff)).fetchone(): 
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
                    s=max(EPS,ph+pd+pa); ph,pa=ph/s,pa/s
                    thr=_get_market_threshold("1X2")
                    if ph*100.0>=thr: candidates.append(("1X2","Home Win",ph))
                    if pa*100.0>=thr: candidates.append(("1X2","Away Win",pa))

                candidates.sort(key=lambda x:x[2], reverse=True)
                per_match=0; base_now=int(time.time())
                for idx,(market_txt,suggestion,prob) in enumerate(candidates):
                    if suggestion not in ALLOWED_SUGGESTIONS: continue
                    if per_match >= max(1,PREDICTIONS_PER_MATCH): break

                    # Odds/EV gate
                    pass_odds, odds, book, ev_pct = _price_gate(market_txt, suggestion, fid, prob)
                    if not pass_odds: 
                        continue
                    
                    if odds is not None:
                        if ev_pct is not None and int(round(ev_pct * 10)) < (EDGE_MIN_BPS / 10):
                            continue

                    created_ts=base_now+idx
                    raw=float(prob); prob_pct=round(raw*100.0,1)

                    c.execute(
                        "INSERT INTO tips(match_id,league_id,league,home,away,market,suggestion,confidence,confidence_raw,score_at_tip,minute,created_ts,odds,book,ev_pct,sent_ok) "
                        "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,0)",
                        (fid,league_id,league,home,away,market_txt,suggestion,float(prob_pct),raw,score,minute,created_ts,
                         (float(odds) if odds is not None else None), (book or None), (float(ev_pct) if ev_pct is not None else None))
                    )

                    sent=_send_tip(home,away,league,minute,score,suggestion,float(prob_pct),feat,odds,book,ev_pct)
                    if sent:
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
        "🏅 <b>Match of the Day</b> [ADVANCED]\n"
        f"<b>Match:</b> {escape(home)} vs {escape(away)}\n"
        f"🏆 <b>League:</b> {escape(league)}\n"
        f"⏰ <b>Kickoff (Berlin):</b> {kickoff_txt}\n"
        f"<b>Tip:</b> {escape(suggestion)}\n"
        f"📈 <b>Confidence:</b> {prob_pct:.1f}%{money}"
    )

def _send_tip(home,away,league,minute,score,suggestion,prob_pct,feat,odds=None,book=None,ev_pct=None)->bool:
    return send_telegram(_format_tip_message(home,away,league,minute,score,suggestion,prob_pct,feat,odds,book,ev_pct))

def prematch_scan_save() -> int:
    fixtures=_collect_todays_prematch_fixtures(); 
    if not fixtures: return 0
    saved=0
    for fx in fixtures:
        fixture=fx.get("fixture") or {}; lg=fx.get("league") or {}; teams=fx.get("teams") or {}
        home=(teams.get("home") or {}).get("name",""); away=(teams.get("away") or {}).get("name","")
        league_id=int((lg.get("id") or 0)); league=f"{lg.get('country','')} - {lg.get('name','')}".strip(" -"); fid=int((fixture.get("id") or 0))
        feat=extract_prematch_features(fx); 
        if not fid or not feat: continue
        
        # Save prematch snapshot for training
        try:
            save_prematch_snapshot(fid, feat)
        except Exception as e:
            log.warning("[PREMATCH] Failed to save snapshot for %s: %s", fid, e)
        
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
        # PRE 1X2 (draw suppressed)
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
            pass_odds, odds, book, ev_pct = _price_gate(mk.replace("PRE ",""), sug, fid, prob)
            if not pass_odds: continue
            
            if odds is not None and ev_pct is not None:
                if int(round(ev_pct * 10)) < (EDGE_MIN_BPS / 10):
                    continue
            created_ts=base_now+idx; raw=float(prob); pct=round(raw*100.0,1)
            with db_conn() as c2:
                c2.execute("INSERT INTO tips(match_id,league_id,league,home,away,market,suggestion,confidence,confidence_raw,score_at_tip,minute,created_ts,odds,book,ev_pct,sent_ok) "
                           "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,'0-0',0,%s,%s,%s,%s,0)",
                           (fid,league_id,league,home,away,mk,sug,float(pct),raw,created_ts,
                            (float(odds) if odds is not None else None), (book or None), (float(ev_pct) if ev_pct is not None else None)))
            saved+=1; per_match+=1
    log.info("[PREMATCH] saved=%d", saved); return saved

# ───────── Learning System to Adapt from Mistakes ─────────
def detect_error_patterns(days: int = LEARNING_HISTORY_DAYS) -> Dict[str, Any]:
    """Analyze past mistakes to detect patterns and suggest corrections"""
    if not LEARNING_ENABLE:
        return {"enabled": False, "patterns": []}
    
    cutoff = int(time.time()) - days * 24 * 3600
    patterns = []
    
    with db_conn() as c:
        # Analyze by market type
        rows = c.execute("""
            SELECT t.market, t.suggestion, t.confidence, t.confidence_raw, 
                   r.final_goals_h, r.final_goals_a, r.btts_yes,
                   t.created_ts, t.league_id, t.league
            FROM tips t 
            JOIN match_results r ON r.match_id = t.match_id
            WHERE t.created_ts >= %s AND t.suggestion <> 'HARVEST' 
            ORDER BY t.created_ts DESC
            LIMIT 1000
        """, (cutoff,)).fetchall()
    
    if not rows:
        return {"patterns": [], "total_samples": 0}
    
    # Group by market and analyze
    by_market = {}
    for row in rows:
        market = row[0] or "Unknown"
        if market not in by_market:
            by_market[market] = {"total": 0, "correct": 0, "samples": []}
        
        suggestion = row[1]
        outcome = _tip_outcome_for_result(suggestion, {
            "final_goals_h": row[4],
            "final_goals_a": row[5],
            "btts_yes": row[6]
        })
        
        by_market[market]["total"] += 1
        if outcome == 1:
            by_market[market]["correct"] += 1
        by_market[market]["samples"].append({
            "suggestion": suggestion,
            "confidence": row[2],
            "outcome": outcome,
            "league": row[9]
        })
    
    # Detect patterns
    for market, data in by_market.items():
        if data["total"] < LEARNING_MIN_SAMPLES:
            continue
        
        accuracy = data["correct"] / data["total"]
        avg_confidence = sum(s["confidence"] for s in data["samples"]) / data["total"]
        
        # Pattern 1: Overconfidence in certain markets
        if avg_confidence > 75 and accuracy < 0.5:
            patterns.append({
                "type": "overconfidence",
                "market": market,
                "avg_confidence": avg_confidence,
                "accuracy": accuracy,
                "samples": data["total"],
                "suggestion": f"Consider lowering confidence threshold for {market} from current {_get_market_threshold(market):.1f}%"
            })
        
        # Pattern 2: League-specific issues
        league_stats = {}
        for sample in data["samples"]:
            league = sample["league"]
            if league not in league_stats:
                league_stats[league] = {"total": 0, "correct": 0}
            league_stats[league]["total"] += 1
            if sample["outcome"] == 1:
                league_stats[league]["correct"] += 1
        
        for league, stats in league_stats.items():
            if stats["total"] >= 10:
                league_acc = stats["correct"] / stats["total"]
                if league_acc < 0.4:  # Poor performance in this league
                    patterns.append({
                        "type": "league_specific",
                        "market": market,
                        "league": league,
                        "accuracy": league_acc,
                        "samples": stats["total"],
                        "suggestion": f"Avoid {market} tips in {league} (accuracy: {league_acc:.1%})"
                    })
        
        # Pattern 3: Suggestion type issues
        suggestion_stats = {}
        for sample in data["samples"]:
            sugg = sample["suggestion"]
            if sugg not in suggestion_stats:
                suggestion_stats[sugg] = {"total": 0, "correct": 0}
            suggestion_stats[sugg]["total"] += 1
            if sample["outcome"] == 1:
                suggestion_stats[sugg]["correct"] += 1
        
        for sugg, stats in suggestion_stats.items():
            if stats["total"] >= 5:
                sugg_acc = stats["correct"] / stats["total"]
                if sugg_acc < 0.4:
                    patterns.append({
                        "type": "suggestion_specific",
                        "market": market,
                        "suggestion": sugg,
                        "accuracy": sugg_acc,
                        "samples": stats["total"],
                        "suggestion_text": f"Avoid {sugg} in {market} market (accuracy: {sugg_acc:.1%})"
                    })
    
    # Save detected patterns to database
    now = int(time.time())
    for pattern in patterns:
        with db_conn() as c:
            c.execute("""
                INSERT INTO error_patterns 
                (pattern_type, pattern_key, error_rate, samples_count, last_detected_ts, corrective_action, created_ts)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (pattern_type, pattern_key) 
                DO UPDATE SET 
                    error_rate = EXCLUDED.error_rate,
                    samples_count = EXCLUDED.samples_count,
                    last_detected_ts = EXCLUDED.last_detected_ts,
                    corrective_action = EXCLUDED.corrective_action
            """, (
                pattern["type"],
                f"{pattern.get('market')}_{pattern.get('league', '')}_{pattern.get('suggestion', '')}",
                1.0 - pattern.get("accuracy", 0.5),
                pattern["samples"],
                now,
                pattern.get("suggestion") or pattern.get("suggestion_text", ""),
                now
            ))
    
    return {
        "patterns": patterns,
        "total_samples": sum(data["total"] for data in by_market.values()),
        "avg_accuracy": sum(data["correct"] for data in by_market.values()) / sum(data["total"] for data in by_market.values()) if by_market else 0
    }

def apply_learning_corrections() -> Dict[str, Any]:
    """Apply learned corrections to improve future predictions"""
    if not LEARNING_ENABLE:
        return {"applied": 0, "corrections": []}
    
    corrections_applied = 0
    corrections = []
    
    with db_conn() as c:
        # Get recent error patterns
        rows = c.execute("""
            SELECT pattern_type, pattern_key, error_rate, samples_count, corrective_action
            FROM error_patterns 
            WHERE last_detected_ts >= %s 
            AND samples_count >= %s
            ORDER BY error_rate DESC
            LIMIT 20
        """, (int(time.time()) - 7*24*3600, LEARNING_MIN_SAMPLES)).fetchall()
    
    for row in rows:
        pattern_type, pattern_key, error_rate, samples_count, corrective_action = row
        
        # Apply corrections based on pattern type
        if pattern_type == "overconfidence" and "market" in pattern_key:
            market = pattern_key.split("_")[0]
            current_threshold = _get_market_threshold(market)
            
            # Increase threshold if overconfidence detected
            if error_rate > 0.6 and samples_count >= 20:
                new_threshold = min(current_threshold + 5.0, MAX_THRESH)
                set_setting(f"conf_threshold:{market}", f"{new_threshold:.1f}")
                _SETTINGS_CACHE.invalidate(f"conf_threshold:{market}")
                
                corrections.append({
                    "type": "threshold_adjustment",
                    "market": market,
                    "old_threshold": current_threshold,
                    "new_threshold": new_threshold,
                    "reason": f"High error rate ({error_rate:.1%}) with {samples_count} samples"
                })
                corrections_applied += 1
        
        elif pattern_type == "league_specific":
            # Could implement league-specific filters here
            corrections.append({
                "type": "league_warning",
                "action": corrective_action,
                "error_rate": error_rate,
                "samples": samples_count
            })
    
    # Log performance metrics
    log_performance_metrics()
    
    return {
        "applied": corrections_applied,
        "corrections": corrections,
        "total_patterns_analyzed": len(rows)
    }

def log_performance_metrics():
    """Log daily performance metrics for tracking"""
    today = datetime.now(BERLIN_TZ).date()
    today_start = int(datetime.combine(today, datetime.min.time()).timestamp())
    yesterday_start = today_start - 24*3600
    
    with db_conn() as c:
        # Get yesterday's performance
        rows = c.execute("""
            SELECT t.market, COUNT(*) as total,
                   SUM(CASE WHEN _tip_outcome_for_result(t.suggestion, 
                        json_build_object('final_goals_h', r.final_goals_h, 
                                         'final_goals_a', r.final_goals_a, 
                                         'btts_yes', r.btts_yes)) = 1 THEN 1 ELSE 0 END) as correct,
                   AVG(t.confidence) as avg_conf
            FROM tips t 
            JOIN match_results r ON r.match_id = t.match_id
            WHERE t.created_ts >= %s AND t.created_ts < %s 
            AND t.suggestion <> 'HARVEST'
            GROUP BY t.market
        """, (yesterday_start, today_start)).fetchall()
    
    now = int(time.time())
    for row in rows:
        market, total, correct, avg_conf = row
        if total > 0:
            accuracy = correct / total if total > 0 else 0
            with db_conn() as c2:
                c2.execute("""
                    INSERT INTO performance_log 
                    (log_date, market, total_tips, successful_tips, accuracy_rate, avg_confidence, created_ts)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (log_date, market) 
                    DO UPDATE SET 
                        total_tips = EXCLUDED.total_tips,
                        successful_tips = EXCLUDED.successful_tips,
                        accuracy_rate = EXCLUDED.accuracy_rate,
                        avg_confidence = EXCLUDED.avg_confidence,
                        created_ts = EXCLUDED.created_ts
                """, (today - timedelta(days=1), market, total, correct, accuracy, avg_conf or 0, now))

def learning_system_update():
    """Main learning system update job"""
    if not LEARNING_ENABLE:
        return
    
    log.info("[LEARNING] Starting learning system update")
    
    # Step 1: Detect error patterns
    patterns = detect_error_patterns(LEARNING_HISTORY_DAYS)
    
    # Step 2: Apply corrections
    corrections = apply_learning_corrections()
    
    # Step 3: Send summary if significant findings
    if patterns.get("patterns") and corrections.get("applied", 0) > 0:
        message = "🧠 <b>Learning System Update</b>\n"
        message += f"Analyzed {patterns.get('total_samples', 0)} samples\n"
        message += f"Found {len(patterns.get('patterns', []))} patterns\n"
        message += f"Applied {corrections.get('applied', 0)} corrections\n\n"
        
        for i, correction in enumerate(corrections.get("corrections", [])[:3]):
            if correction["type"] == "threshold_adjustment":
                message += f"• {correction['market']}: {correction['old_threshold']:.1f}% → {correction['new_threshold']:.1f}%\n"
        
        send_telegram(message)
    
    log.info(f"[LEARNING] Update complete. Patterns: {len(patterns.get('patterns', []))}, Corrections: {corrections.get('applied', 0)}")

# ───────── Auto-train / tune / retry ─────────
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
        pred=(p>=t).ast(int); n=int(pred.sum())
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

    best = None  # (prob_pct, suggestion, home, away, league, kickoff_txt, odds, book, ev_pct)

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
        pass_odds, odds, book, ev_pct = _price_gate(mk, sug, fid, prob)
        if not pass_odds:
            continue

        if odds is not None and ev_pct is not None:
            if MOTD_MIN_EV_BPS > 0 and int(round(ev_pct * 10)) < (MOTD_MIN_EV_BPS / 10):
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
        # Learning system jobs
        if LEARNING_ENABLE:
            sched.add_job(lambda:_run_with_pg_lock(1008,learning_system_update),
                          CronTrigger(hour=LEARNING_UPDATE_HOUR, minute=LEARNING_UPDATE_MINUTE, timezone=TZ_UTC),
                          id="learning_update", max_instances=1, coalesce=True)
            sched.add_job(lambda:_run_with_pg_lock(1009,detect_error_patterns,LEARNING_HISTORY_DAYS),
                          CronTrigger(hour=LEARNING_UPDATE_HOUR+1, minute=LEARNING_UPDATE_MINUTE, timezone=TZ_UTC),
                          id="error_patterns", max_instances=1, coalesce=True)
        sched.start(); _scheduler_started=True
        send_telegram("🚀 goalsniper ADVANCED mode (in-play + prematch) started with ELO ratings and Learning System.")
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
def root(): return jsonify({"ok": True, "name": "goalsniper", "mode": "FULL_AI_ADVANCED", "scheduler": RUN_SCHEDULER})

@app.route("/health")
def health():
    try:
        with db_conn() as c:
            n=c.execute("SELECT COUNT(*) FROM tips").fetchone()[0]
            r=c.execute("SELECT COUNT(*) FROM team_ratings").fetchone()[0]
            p=c.execute("SELECT COUNT(*) FROM error_patterns").fetchone()[0]
        return jsonify({"ok": True, "db": "ok", "tips_count": int(n), "team_ratings": int(r), "error_patterns": int(p)})
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

@app.route("/admin/update-ratings", methods=["POST","GET"])
def http_update_ratings():
    """Manually trigger rating updates from recent matches"""
    _require_admin()
    cutoff = int(time.time()) - 30*24*3600  # Last 30 days
    updated = 0
    
    with db_conn() as c:
        rows = c.execute("""
            SELECT match_id FROM match_results 
            WHERE updated_ts >= %s 
            ORDER BY updated_ts DESC LIMIT 1000
        """, (cutoff,)).fetchall()
    
    for (match_id,) in rows:
        fx = _fixture_by_id(int(match_id))
        if fx:
            try:
                update_ratings_from_finished_match(fx)
                updated += 1
            except Exception as e:
                log.warning("Failed to update ratings for match %s: %s", match_id, e)
    
    return jsonify({"ok": True, "updated": updated})

@app.route("/team-rating/<int:team_id>")
def http_team_rating(team_id: int):
    """Get team rating"""
    rating_data = get_team_rating(team_id)
    return jsonify({"ok": True, "team_id": team_id, "rating": rating_data})

@app.route("/admin/learning/patterns", methods=["GET"])
def http_learning_patterns():
    """Get detected error patterns"""
    _require_admin()
    patterns = detect_error_patterns(LEARNING_HISTORY_DAYS)
    return jsonify({"ok": True, "patterns": patterns})

@app.route("/admin/learning/apply", methods=["POST"])
def http_learning_apply():
    """Apply learning corrections"""
    _require_admin()
    corrections = apply_learning_corrections()
    return jsonify({"ok": True, "corrections": corrections})

@app.route("/admin/learning/performance", methods=["GET"])
def http_learning_performance():
    """Get performance metrics"""
    _require_admin()
    days = int(request.args.get("days", "7"))
    cutoff = int(time.time()) - days * 24 * 3600
    
    with db_conn() as c:
        rows = c.execute("""
            SELECT log_date, market, total_tips, successful_tips, accuracy_rate, avg_confidence
            FROM performance_log 
            WHERE created_ts >= %s
            ORDER BY log_date DESC, market
        """, (cutoff,)).fetchall()
    
    metrics = []
    for row in rows:
        metrics.append({
            "date": row[0].isoformat() if hasattr(row[0], 'isoformat') else str(row[0]),
            "market": row[1],
            "total_tips": row[2],
            "successful_tips": row[3],
            "accuracy_rate": float(row[4]),
            "avg_confidence": float(row[5]) if row[5] else 0
        })
    
    return jsonify({"ok": True, "metrics": metrics})

@app.route("/settings/<key>", methods=["GET","POST"])
def http_settings(key: str):
    _require_admin()
    if request.method=="GET":
        val=get_setting_cached(key); return jsonify({"ok": True, "key": key, "value": val})
    val=(request.get_json(silent=True) or {}).get("value")
    if val is None: abort(400)
    set_setting(key, str(val)); _SETTINGS_CACHE.invalidate(key); invalidate_model_caches_for_key(key)
    return jsonify({"ok": True})

@app.route("/admin/diagnostics/edge-analysis", methods=["GET"])
def edge_analysis():
    """Comprehensive edge analysis - THE TRUTH TELLER"""
    
    days = int(request.args.get("days", "60"))
    min_tips_per_scenario = int(request.args.get("min_samples", "20"))
    
    cutoff_ts = int(time.time()) - days * 24 * 3600
    
    with db_conn() as c:
        # Get ALL tips with outcomes and odds
        tips_data = c.execute("""
            SELECT 
                t.match_id,
                t.league,
                t.market,
                t.suggestion,
                t.confidence,
                t.confidence_raw,
                t.score_at_tip,
                t.minute,
                t.odds,
                t.book,
                t.ev_pct,
                r.final_goals_h,
                r.final_goals_a,
                r.btts_yes,
                t.created_ts
            FROM tips t
            LEFT JOIN match_results r ON r.match_id = t.match_id
            WHERE t.created_ts >= %s
            AND t.suggestion <> 'HARVEST'
            AND t.sent_ok = 1
            ORDER BY t.created_ts DESC
        """, (cutoff_ts,)).fetchall()
    
    if not tips_data:
        return jsonify({"error": "No tips found in the specified period"})
    
    # Calculate outcomes and profits
    analyzed_tips = []
    total_staked = 0
    total_return = 0
    
    for tip in tips_data:
        (match_id, league, market, suggestion, confidence, confidence_raw, 
         score_at_tip, minute, odds, book, ev_pct, final_gh, final_ga, btts, created_ts) = tip
        
        # Skip if no result yet
        if final_gh is None:
            continue
        
        # Determine outcome
        result = {
            "final_goals_h": final_gh,
            "final_goals_a": final_ga,
            "btts_yes": btts
        }
        outcome = _tip_outcome_for_result(suggestion, result)
        
        # Calculate profit (assuming 1 unit stake)
        profit = 0
        if outcome == 1 and odds:
            profit = odds - 1  # Win: odds - stake
        elif outcome == 0 and odds:
            profit = -1  # Loss: lose stake
        
        analyzed_tips.append({
            "match_id": match_id,
            "league": league,
            "market": market,
            "suggestion": suggestion,
            "confidence": float(confidence) if confidence else None,
            "odds": float(odds) if odds else None,
            "minute": minute,
            "score_at_tip": score_at_tip,
            "outcome": outcome,  # 1=win, 0=loss, None=push
            "profit": profit,
            "ev_pct": float(ev_pct) if ev_pct else None,
            "expected_value": _ev(confidence/100.0, odds) if confidence and odds else None
        })
        
        if odds:
            total_staked += 1
            total_return += profit
    
    # Calculate overall metrics
    total_tips = len(analyzed_tips)
    winning_tips = sum(1 for t in analyzed_tips if t["outcome"] == 1)
    losing_tips = sum(1 for t in analyzed_tips if t["outcome"] == 0)
    pushes = sum(1 for t in analyzed_tips if t["outcome"] is None)
    
    overall_roi = (total_return / total_staked * 100) if total_staked > 0 else 0
    win_rate = (winning_tips / (winning_tips + losing_tips) * 100) if (winning_tips + losing_tips) > 0 else 0
    
    # 1. Analyze by MARKET
    market_analysis = {}
    for tip in analyzed_tips:
        market = tip["market"]
        if market not in market_analysis:
            market_analysis[market] = {
                "tips": 0, "wins": 0, "losses": 0, "pushes": 0,
                "total_staked": 0, "total_return": 0,
                "avg_odds": [], "avg_confidence": []
            }
        
        ma = market_analysis[market]
        ma["tips"] += 1
        if tip["outcome"] == 1:
            ma["wins"] += 1
        elif tip["outcome"] == 0:
            ma["losses"] += 1
        else:
            ma["pushes"] += 1
        
        if tip["odds"]:
            ma["total_staked"] += 1
            ma["total_return"] += tip["profit"]
            ma["avg_odds"].append(tip["odds"])
        
        if tip["confidence"]:
            ma["avg_confidence"].append(tip["confidence"])
    
    # Calculate market metrics
    for market, data in market_analysis.items():
        if data["total_staked"] > 0:
            data["roi"] = (data["total_return"] / data["total_staked"] * 100)
        else:
            data["roi"] = 0
        
        if data["wins"] + data["losses"] > 0:
            data["win_rate"] = (data["wins"] / (data["wins"] + data["losses"]) * 100)
        else:
            data["win_rate"] = 0
        
        data["avg_odds"] = sum(data["avg_odds"]) / len(data["avg_odds"]) if data["avg_odds"] else 0
        data["avg_confidence"] = sum(data["avg_confidence"]) / len(data["avg_confidence"]) if data["avg_confidence"] else 0
        
        # Expected vs Actual
        if data["tips"] >= min_tips_per_scenario:
            implied_win_rate = (1 / data["avg_odds"]) * 100 if data["avg_odds"] else 0
            data["edge_over_market"] = data["win_rate"] - implied_win_rate
        else:
            data["edge_over_market"] = None
    
    # 2. Analyze by SUGGESTION TYPE
    suggestion_analysis = {}
    for tip in analyzed_tips:
        suggestion = tip["suggestion"]
        if suggestion not in suggestion_analysis:
            suggestion_analysis[suggestion] = {
                "tips": 0, "wins": 0, "losses": 0,
                "total_staked": 0, "total_return": 0,
                "avg_odds": [], "avg_confidence": []
            }
        
        sa = suggestion_analysis[suggestion]
        sa["tips"] += 1
        if tip["outcome"] == 1:
            sa["wins"] += 1
        elif tip["outcome"] == 0:
            sa["losses"] += 1
        
        if tip["odds"]:
            sa["total_staked"] += 1
            sa["total_return"] += tip["profit"]
            sa["avg_odds"].append(tip["odds"])
        
        if tip["confidence"]:
            sa["avg_confidence"].append(tip["confidence"])
    
    # Calculate suggestion metrics
    for suggestion, data in suggestion_analysis.items():
        if data["total_staked"] > 0:
            data["roi"] = (data["total_return"] / data["total_staked"] * 100)
            data["avg_odds"] = sum(data["avg_odds"]) / len(data["avg_odds"])
        else:
            data["roi"] = 0
            data["avg_odds"] = 0
        
        if data["wins"] + data["losses"] > 0:
            data["win_rate"] = (data["wins"] / (data["wins"] + data["losses"]) * 100)
        else:
            data["win_rate"] = 0
        
        data["avg_confidence"] = sum(data["avg_confidence"]) / len(data["avg_confidence"]) if data["avg_confidence"] else 0
        
        # Statistical significance test
        if data["tips"] >= min_tips_per_scenario:
            # Z-test for binomial proportion
            implied_prob = 1 / data["avg_odds"] if data["avg_odds"] > 0 else 0
            actual_prob = data["wins"] / data["tips"]
            n = data["tips"]
            
            if implied_prob > 0 and implied_prob < 1:
                se = math.sqrt(implied_prob * (1 - implied_prob) / n)
                z_score = (actual_prob - implied_prob) / se if se > 0 else 0
                data["z_score"] = z_score
                data["p_value"] = 2 * (1 - norm.cdf(abs(z_score)))  # Two-tailed test
                data["significant"] = abs(z_score) > 1.96  # 95% confidence
            else:
                data["z_score"] = None
                data["p_value"] = None
                data["significant"] = False
        else:
            data["z_score"] = None
            data["p_value"] = None
            data["significant"] = False
    
    # 3. Analyze by MINUTE BUCKET
    minute_buckets = {
        "0-20": (0, 20),
        "20-40": (20, 40),
        "40-60": (40, 60),
        "60-75": (60, 75),
        "75+": (75, 200)
    }
    
    minute_analysis = {}
    for bucket_name, (min_start, min_end) in minute_buckets.items():
        bucket_tips = [t for t in analyzed_tips if min_start <= t["minute"] < min_end]
        
        if not bucket_tips:
            continue
        
        wins = sum(1 for t in bucket_tips if t["outcome"] == 1)
        losses = sum(1 for t in bucket_tips if t["outcome"] == 0)
        stakes = sum(1 for t in bucket_tips if t["odds"])
        returns = sum(t["profit"] for t in bucket_tips if t["odds"])
        
        minute_analysis[bucket_name] = {
            "tips": len(bucket_tips),
            "wins": wins,
            "losses": losses,
            "win_rate": (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0,
            "roi": (returns / stakes * 100) if stakes > 0 else 0,
            "avg_minute": sum(t["minute"] for t in bucket_tips) / len(bucket_tips)
        }
    
    # 4. Analyze by SCORE AT TIP
    score_analysis = {}
    for tip in analyzed_tips:
        score = tip["score_at_tip"]
        if score not in score_analysis:
            score_analysis[score] = {
                "tips": 0, "wins": 0, "losses": 0,
                "total_staked": 0, "total_return": 0
            }
        
        sa = score_analysis[score]
        sa["tips"] += 1
        if tip["outcome"] == 1:
            sa["wins"] += 1
        elif tip["outcome"] == 0:
            sa["losses"] += 1
        
        if tip["odds"]:
            sa["total_staked"] += 1
            sa["total_return"] += tip["profit"]
    
    # Calculate score metrics
    for score, data in score_analysis.items():
        if data["total_staked"] > 0:
            data["roi"] = (data["total_return"] / data["total_staked"] * 100)
        else:
            data["roi"] = 0
        
        if data["wins"] + data["losses"] > 0:
            data["win_rate"] = (data["wins"] / (data["wins"] + data["losses"]) * 100)
        else:
            data["win_rate"] = 0
    
    # 5. CONFIDENCE CALIBRATION
    confidence_buckets = {
        "50-55": (50, 55),
        "55-60": (55, 60),
        "60-65": (60, 65),
        "65-70": (65, 70),
        "70-75": (70, 75),
        "75-80": (75, 80),
        "80-85": (80, 85),
        "85-90": (85, 90),
        "90-95": (90, 95),
        "95-100": (95, 100)
    }
    
    calibration_analysis = {}
    for bucket_name, (conf_low, conf_high) in confidence_buckets.items():
        bucket_tips = [t for t in analyzed_tips 
                      if t["confidence"] and conf_low <= t["confidence"] < conf_high]
        
        if len(bucket_tips) < 5:  # Too few samples
            continue
        
        wins = sum(1 for t in bucket_tips if t["outcome"] == 1)
        total = len(bucket_tips)
        
        calibration_analysis[bucket_name] = {
            "tips": total,
            "expected_win_rate": (conf_low + conf_high) / 2,
            "actual_win_rate": (wins / total * 100) if total > 0 else 0,
            "calibration_error": ((wins / total * 100) - (conf_low + conf_high) / 2) if total > 0 else 0
        }
    
    # 6. Find TOP PROFITABLE SCENARIOS (combination analysis)
    profitable_scenarios = []
    
    # Define scenario combinations to analyze
    scenarios_to_check = []
    
    for tip in analyzed_tips:
        if tip["odds"] and tip["confidence"]:
            # Create scenario key
            scenario_key = f"{tip['market']}|{tip['suggestion']}|{tip['score_at_tip']}|{tip['minute']//10*10}"
            scenarios_to_check.append((scenario_key, tip))
    
    # Group and analyze scenarios
    scenario_groups = {}
    for scenario_key, tip in scenarios_to_check:
        if scenario_key not in scenario_groups:
            scenario_groups[scenario_key] = []
        scenario_groups[scenario_key].append(tip)
    
    # Calculate metrics for each scenario
    for scenario_key, tips in scenario_groups.items():
        if len(tips) < min_tips_per_scenario:
            continue
        
        wins = sum(1 for t in tips if t["outcome"] == 1)
        losses = sum(1 for t in tips if t["outcome"] == 0)
        total_return = sum(t["profit"] for t in tips if t["odds"])
        total_staked = sum(1 for t in tips if t["odds"])
        
        if total_staked == 0:
            continue
        
        win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
        roi = total_return / total_staked * 100
        avg_odds = sum(t["odds"] for t in tips if t["odds"]) / total_staked
        
        # Calculate expected value based on average confidence
        avg_confidence = sum(t["confidence"] for t in tips if t["confidence"]) / len(tips)
        expected_roi = _ev(avg_confidence/100.0, avg_odds) * 100
        
        profitable_scenarios.append({
            "scenario": scenario_key,
            "market": tips[0]["market"],
            "suggestion": tips[0]["suggestion"],
            "score": tips[0]["score_at_tip"],
            "minute_range": f"{tips[0]['minute']//10*10}-{tips[0]['minute']//10*10+10}",
            "samples": len(tips),
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 1),
            "roi": round(roi, 1),
            "avg_odds": round(avg_odds, 2),
            "avg_confidence": round(avg_confidence, 1),
            "expected_roi": round(expected_roi, 1),
            "edge_actual_vs_expected": round(roi - expected_roi, 1)
        })
    
    # Sort scenarios by ROI
    profitable_scenarios.sort(key=lambda x: x["roi"], reverse=True)
    top_scenarios = profitable_scenarios[:10]
    worst_scenarios = profitable_scenarios[-10:] if len(profitable_scenarios) >= 10 else []
    
    # 7. Statistical Significance Tests
    overall_significance = {
        "total_tips": total_tips,
        "winning_tips": winning_tips,
        "losing_tips": losing_tips,
        "win_rate": round(win_rate, 1),
        "overall_roi": round(overall_roi, 1)
    }
    
    # Test if overall ROI is statistically significant
    if total_staked > 0:
        # Standard error of ROI
        roi_std = math.sqrt(abs(overall_roi/100) * (1 - abs(overall_roi/100)) / total_staked)
        roi_z = (overall_roi/100) / roi_std if roi_std > 0 else 0
        overall_significance["roi_z_score"] = round(roi_z, 2)
        overall_significance["roi_p_value"] = round(2 * (1 - norm.cdf(abs(roi_z))), 4)
        overall_significance["roi_significant"] = abs(roi_z) > 1.96
    else:
        overall_significance["roi_z_score"] = None
        overall_significance["roi_p_value"] = None
        overall_significance["roi_significant"] = False
    
    # Compile final report
    report = {
        "period_days": days,
        "total_tips_analyzed": total_tips,
        "overall_metrics": {
            "win_rate": round(win_rate, 1),
            "roi": round(overall_roi, 1),
            "total_staked": total_staked,
            "total_return": round(total_return, 2),
            "profit_per_tip": round(total_return / total_staked, 3) if total_staked > 0 else 0,
            "push_rate": round(pushes / total_tips * 100, 1) if total_tips > 0 else 0
        },
        "significance": overall_significance,
        "by_market": market_analysis,
        "by_suggestion": suggestion_analysis,
        "by_minute": minute_analysis,
        "by_score": score_analysis,
        "calibration": calibration_analysis,
        "top_profitable_scenarios": top_scenarios,
        "worst_scenarios": worst_scenarios,
        "diagnostic_conclusion": _generate_diagnostic_conclusion(
            overall_roi, win_rate, total_tips, market_analysis
        )
    }
    
    return jsonify(report)

def _generate_diagnostic_conclusion(roi, win_rate, total_tips, market_analysis):
    """Generate a plain English conclusion about edge"""
    
    conclusions = []
    
    # ROI conclusion
    if roi > 5:
        conclusions.append(f"✅ STRONG EDGE: {roi:.1f}% ROI suggests significant profitability")
    elif roi > 2:
        conclusions.append(f"⚠️ MODEST EDGE: {roi:.1f}% ROI is positive but small - needs more data")
    elif roi > 0:
        conclusions.append(f"📊 MARGINAL EDGE: {roi:.1f}% ROI might not survive variance")
    elif roi > -2:
        conclusions.append(f"⚠️ NO CLEAR EDGE: {roi:.1f}% ROI suggests model is roughly break-even")
    else:
        conclusions.append(f"❌ NEGATIVE EDGE: {roi:.1f}% ROI suggests model is losing money")
    
    # Sample size warning
    if total_tips < 100:
        conclusions.append(f"📉 WARNING: Only {total_tips} tips analyzed - need minimum 500 for statistical confidence")
    elif total_tips < 500:
        conclusions.append(f"⚠️ CAUTION: {total_tips} tips analyzed - borderline for statistical significance")
    else:
        conclusions.append(f"📈 GOOD SAMPLE SIZE: {total_tips} tips provides reasonable confidence")
    
    # Market-specific insights
    profitable_markets = [m for m, data in market_analysis.items() 
                         if data.get("roi", 0) > 3 and data.get("tips", 0) >= 20]
    
    if profitable_markets:
        conclusions.append(f"🎯 PROFITABLE MARKETS: {', '.join(profitable_markets)}")
    
    losing_markets = [m for m, data in market_analysis.items() 
                     if data.get("roi", 0) < -5 and data.get("tips", 0) >= 10]
    
    if losing_markets:
        conclusions.append(f"💣 LOSING MARKETS: {', '.join(losing_markets)} - consider disabling")
    
    # Final recommendation
    if roi > 2 and total_tips >= 200:
        conclusions.append("🎉 RECOMMENDATION: System shows real edge - continue with bankroll management")
    elif roi > 0 and total_tips >= 200:
        conclusions.append("🤔 RECOMMENDATION: Edge is marginal - optimize top scenarios, cut losers")
    else:
        conclusions.append("🚫 RECOMMENDATION: No clear edge found - need model improvements before real money")
    
    return "\n".join(conclusions)

@app.route("/admin/diagnostics/monte-carlo", methods=["GET"])
def monte_carlo_simulation():
    """Run Monte Carlo simulations to assess bankroll risk"""
    
    days = int(request.args.get("days", "60"))
    initial_bankroll = float(request.args.get("bankroll", "1000.0"))
    stake_percent = float(request.args.get("stake_percent", "2.0"))
    simulations = int(request.args.get("simulations", "10000"))
    
    cutoff_ts = int(time.time()) - days * 24 * 3600
    
    with db_conn() as c:
        # Get all tips with outcomes
        tips = c.execute("""
            SELECT t.odds, 
                   CASE WHEN _tip_outcome_for_result(t.suggestion, 
                         json_build_object('final_goals_h', r.final_goals_h,
                                          'final_goals_a', r.final_goals_a,
                                          'btts_yes', r.btts_yes)) = 1 
                        THEN 1 ELSE 0 END as outcome
            FROM tips t
            JOIN match_results r ON r.match_id = t.match_id
            WHERE t.created_ts >= %s
            AND t.suggestion <> 'HARVEST'
            AND t.odds IS NOT NULL
            AND t.sent_ok = 1
        """, (cutoff_ts,)).fetchall()
    
    if not tips:
        return jsonify({"error": "No tips found for simulation"})
    
    # Extract win probabilities from historical data
    tip_outcomes = []
    for odds, outcome in tips:
        if odds:
            tip_outcomes.append({
                "odds": float(odds),
                "outcome": outcome,
                "profit_if_win": float(odds) - 1,
                "profit_if_lose": -1
            })
    
    # Run Monte Carlo simulations
    simulation_results = []
    
    for sim in range(simulations):
        bankroll = initial_bankroll
        bankroll_history = [bankroll]
        
        # Shuffle tip outcomes randomly
        import random
        shuffled_outcomes = random.sample(tip_outcomes, len(tip_outcomes))
        
        for tip in shuffled_outcomes:
            # Calculate stake (percentage of current bankroll)
            stake = bankroll * (stake_percent / 100)
            
            # Apply outcome
            if tip["outcome"] == 1:
                bankroll += stake * tip["profit_if_win"]
            else:
                bankroll -= stake
            
            bankroll_history.append(bankroll)
            
            # Stop if bankrupt
            if bankroll <= 0:
                bankroll = 0
                break
        
        simulation_results.append({
            "final_bankroll": bankroll,
            "max_drawdown": min(bankroll_history) / initial_bankroll * 100 if bankroll_history else 0,
            "peak": max(bankroll_history) if bankroll_history else 0,
            "trough": min(bankroll_history) if bankroll_history else 0
        })
    
    # Analyze simulation results
    final_values = [r["final_bankroll"] for r in simulation_results]
    max_drawdowns = [r["max_drawdown"] for r in simulation_results]
    
    # Calculate statistics
    analysis = {
        "initial_bankroll": initial_bankroll,
        "stake_percent": stake_percent,
        "simulations": simulations,
        "historical_tips_used": len(tip_outcomes),
        
        "final_bankroll_stats": {
            "mean": sum(final_values) / len(final_values),
            "median": sorted(final_values)[len(final_values) // 2],
            "std_dev": math.sqrt(sum((x - sum(final_values)/len(final_values))**2 for x in final_values) / len(final_values)),
            "min": min(final_values),
            "max": max(final_values),
            "q10": sorted(final_values)[int(len(final_values) * 0.1)],
            "q90": sorted(final_values)[int(len(final_values) * 0.9)]
        },
        
        "risk_metrics": {
            "probability_of_ruin": sum(1 for v in final_values if v <= 0) / len(final_values) * 100,
            "probability_of_doubling": sum(1 for v in final_values if v >= initial_bankroll * 2) / len(final_values) * 100,
            "avg_max_drawdown": sum(max_drawdowns) / len(max_drawdowns),
            "worst_drawdown": min(max_drawdowns),
            "sharpe_ratio": (sum(final_values)/len(final_values) - initial_bankroll) / max(0.001, math.sqrt(sum((x - sum(final_values)/len(final_values))**2 for x in final_values) / len(final_values)))
        },
        
        "recommendations": _generate_risk_recommendations(
            initial_bankroll, stake_percent, simulation_results
        )
    }
    
    return jsonify(analysis)

def _generate_risk_recommendations(initial_bankroll, stake_percent, results):
    """Generate risk management recommendations"""
    final_values = [r["final_bankroll"] for r in results]
    ruin_prob = sum(1 for v in final_values if v <= 0) / len(final_values) * 100
    
    recommendations = []
    
    if ruin_prob > 20:
        recommendations.append(f"🚨 EXTREME RISK: {ruin_prob:.1f}% chance of ruin - REDUCE STAKE SIZE IMMEDIATELY")
        recommendations.append(f"   Suggested max stake: {stake_percent/2:.1f}% (currently {stake_percent:.1f}%)")
    elif ruin_prob > 10:
        recommendations.append(f"⚠️ HIGH RISK: {ruin_prob:.1f}% chance of ruin - consider reducing stake size")
    elif ruin_prob > 5:
        recommendations.append(f"📊 MODERATE RISK: {ruin_prob:.1f}% chance of ruin - acceptable for aggressive bankroll")
    elif ruin_prob > 1:
        recommendations.append(f"✅ ACCEPTABLE RISK: {ruin_prob:.1f}% chance of ruin - standard for professional betting")
    else:
        recommendations.append(f"🎯 LOW RISK: {ruin_prob:.1f}% chance of ruin - very safe bankroll management")
    
    # Kelly criterion check
    avg_return = sum(final_values) / len(final_values)
    expected_growth = avg_return / initial_bankroll - 1
    
    if stake_percent > expected_growth * 100 * 2:  # More than half-Kelly
        recommendations.append(f"📉 OVER-BETTING: Stake {stake_percent:.1f}% may be too high for expected edge")
        optimal_stake = max(0.5, min(5.0, expected_growth * 100 * 0.5))  # Quarter-Kelly
        recommendations.append(f"   Recommended stake: {optimal_stake:.1f}% (quarter-Kelly)")
    
    return recommendations

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

@app.route("/admin/diagnostics/edge-detector", methods=["GET"])
def real_time_edge_detector():
    """Real-time edge detection for current matches"""
    
    # Get current live matches
    live_matches = fetch_live_matches()
    
    edge_detections = []
    
    for match in live_matches:
        fid = int((match.get("fixture", {}) or {}).get("id") or 0)
        league_id, league = _league_name(match)
        home, away = _teams(match)
        score = _pretty_score(match)
        minute = int(((match.get("fixture") or {}).get("status") or {}).get("elapsed") or 0)
        
        # Extract features
        feat = extract_features(match)
        
        # Get market odds
        odds_data = fetch_odds(fid)
        
        # Analyze each market for edge
        market_edges = []
        
        # Over/Under markets
        for line in OU_LINES:
            mdl = _load_ou_model_for_line(line)
            if not mdl or f"OU_{_fmt_line(line)}" not in odds_data:
                continue
            
            # Our probability
            our_prob = _score_prob(feat, mdl)
            
            # Market implied probability
            market_over = odds_data[f"OU_{_fmt_line(line)}"].get("Over", {}).get("odds")
            market_under = odds_data[f"OU_{_fmt_line(line)}"].get("Under", {}).get("odds")
            
            if market_over:
                market_implied = 1 / market_over
                edge_over = our_prob - market_implied
                ev_over = _ev(our_prob, market_over) * 100
                
                if abs(edge_over) > 0.05:  # 5% edge
                    market_edges.append({
                        "market": f"Over {_fmt_line(line)}",
                        "our_prob": round(our_prob * 100, 1),
                        "market_implied": round(market_implied * 100, 1),
                        "edge": round(edge_over * 100, 1),
                        "ev": round(ev_over, 1),
                        "odds": market_over,
                        "book": odds_data[f"OU_{_fmt_line(line)}"]["Over"].get("book")
                    })
        
        # 1X2 market
        mh, md, ma = _load_wld_models()
        if mh and ma and "1X2" in odds_data:
            ph = _score_prob(feat, mh)
            pa = _score_prob(feat, ma)
            
            # Normalize (ignore draw)
            s = max(EPS, ph + pa)
            ph, pa = ph/s, pa/s
            
            market_home = odds_data["1X2"].get("Home", {}).get("odds")
            market_away = odds_data["1X2"].get("Away", {}).get("odds")
            
            if market_home:
                market_implied_h = 1 / market_home
                edge_h = ph - market_implied_h
                ev_h = _ev(ph, market_home) * 100
                
                if abs(edge_h) > 0.05:
                    market_edges.append({
                        "market": "Home Win",
                        "our_prob": round(ph * 100, 1),
                        "market_implied": round(market_implied_h * 100, 1),
                        "edge": round(edge_h * 100, 1),
                        "ev": round(ev_h, 1),
                        "odds": market_home,
                        "book": odds_data["1X2"]["Home"].get("book")
                    })
        
        # Sort by absolute edge
        market_edges.sort(key=lambda x: abs(x["edge"]), reverse=True)
        
        if market_edges:
            edge_detections.append({
                "match": f"{home} vs {away}",
                "league": league,
                "minute": minute,
                "score": score,
                "edges": market_edges[:3]  # Top 3 edges
            })
    
    return jsonify({
        "timestamp": int(time.time()),
        "matches_analyzed": len(live_matches),
        "edges_found": len(edge_detections),
        "detections": edge_detections
    })

@app.route("/admin/diagnostics/dashboard")
def diagnostics_dashboard():
    """HTML dashboard for visual diagnostics"""
    
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Edge Diagnostic Dashboard</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body { font-family: Arial; margin: 20px; }
            .card { border: 1px solid #ddd; padding: 15px; margin: 10px; border-radius: 5px; }
            .positive { color: green; }
            .negative { color: red; }
            .warning { color: orange; }
            .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }
            button { padding: 10px 20px; margin: 5px; cursor: pointer; }
            .summary { background: #f5f5f5; padding: 15px; border-radius: 5px; }
        </style>
    </head>
    <body>
        <h1>⚡ Edge Diagnostic Dashboard</h1>
        
        <div class="summary">
            <h3>Quick Actions</h3>
            <button onclick="runEdgeAnalysis()">Run Edge Analysis</button>
            <button onclick="runMonteCarlo()">Run Risk Simulation</button>
            <button onclick="runRealTimeEdge()">Scan Live Edges</button>
            <button onclick="exportData()">Export All Data</button>
        </div>
        
        <div class="grid">
            <div class="card">
                <h3>Overall Performance</h3>
                <div id="overall-stats">Click "Run Edge Analysis"</div>
                <canvas id="roi-chart" width="400" height="200"></canvas>
            </div>
            
            <div class="card">
                <h3>Market Breakdown</h3>
                <div id="market-breakdown"></div>
                <canvas id="market-chart" width="400" height="200"></canvas>
            </div>
            
            <div class="card">
                <h3>Calibration Check</h3>
                <div id="calibration-data"></div>
                <canvas id="calibration-chart" width="400" height="200"></canvas>
            </div>
            
            <div class="card">
                <h3>Risk Analysis</h3>
                <div id="risk-stats">Click "Run Risk Simulation"</div>
                <canvas id="risk-chart" width="400" height="200"></canvas>
            </div>
            
            <div class="card" style="grid-column: span 2;">
                <h3>Top & Worst Scenarios</h3>
                <div style="display: flex; gap: 20px;">
                    <div style="flex: 1;">
                        <h4>Top 5 Profitable</h4>
                        <div id="top-scenarios"></div>
                    </div>
                    <div style="flex: 1;">
                        <h4>Top 5 Losing</h4>
                        <div id="worst-scenarios"></div>
                    </div>
                </div>
            </div>
            
            <div class="card" style="grid-column: span 2;">
                <h3>Real-Time Edge Detection</h3>
                <div id="live-edges"></div>
                <button onclick="runRealTimeEdge()">Refresh Live Edges</button>
            </div>
        </div>
        
        <script>
            async function runEdgeAnalysis() {
                const response = await fetch('/admin/diagnostics/edge-analysis?days=90');
                const data = await response.json();
                
                // Update overall stats
                document.getElementById('overall-stats').innerHTML = `
                    <p><b>Win Rate:</b> <span class="${data.overall_metrics.win_rate > 50 ? 'positive' : 'negative'}">${data.overall_metrics.win_rate}%</span></p>
                    <p><b>ROI:</b> <span class="${data.overall_metrics.roi > 0 ? 'positive' : 'negative'}">${data.overall_metrics.roi}%</span></p>
                    <p><b>Tips Analyzed:</b> ${data.total_tips_analyzed}</p>
                    <p><b>Conclusion:</b> ${data.diagnostic_conclusion.split('\\n').join('<br>')}</p>
                `;
                
                // Update market breakdown
                let marketHTML = '';
                for (const [market, stats] of Object.entries(data.by_market)) {
                    marketHTML += `
                        <p><b>${market}:</b> ${stats.tips} tips, 
                        ROI: <span class="${stats.roi > 0 ? 'positive' : 'negative'}">${stats.roi?.toFixed(1) || 0}%</span>,
                        Win Rate: ${stats.win_rate?.toFixed(1) || 0}%</p>
                    `;
                }
                document.getElementById('market-breakdown').innerHTML = marketHTML;
                
                // Update calibration
                let calHTML = '';
                for (const [bucket, stats] of Object.entries(data.calibration)) {
                    const diff = stats.calibration_error;
                    calHTML += `
                        <p><b>${bucket}%:</b> Expected ${stats.expected_win_rate}%, Actual ${stats.actual_win_rate}%
                        <span class="${Math.abs(diff) < 5 ? 'positive' : 'warning'}"> (${diff > 0 ? '+' : ''}${diff.toFixed(1)}%)</span></p>
                    `;
                }
                document.getElementById('calibration-data').innerHTML = calHTML;
                
                // Update scenarios
                let topHTML = '';
                data.top_profitable_scenarios.forEach(scenario => {
                    topHTML += `<p>${scenario.scenario}: ${scenario.roi}% ROI (${scenario.samples} samples)</p>`;
                });
                document.getElementById('top-scenarios').innerHTML = topHTML;
                
                let worstHTML = '';
                data.worst_scenarios.forEach(scenario => {
                    worstHTML += `<p>${scenario.scenario}: ${scenario.roi}% ROI (${scenario.samples} samples)</p>`;
                });
                document.getElementById('worst-scenarios').innerHTML = worstHTML;
                
                // Create charts
                createCharts(data);
            }
            
            async function runMonteCarlo() {
                const response = await fetch('/admin/diagnostics/monte-carlo?bankroll=1000&stake_percent=2&simulations=10000');
                const data = await response.json();
                
                document.getElementById('risk-stats').innerHTML = `
                    <p><b>Ruin Probability:</b> <span class="${data.risk_metrics.probability_of_ruin < 5 ? 'positive' : 'warning'}">${data.risk_metrics.probability_of_ruin.toFixed(1)}%</span></p>
                    <p><b>Expected Final Bankroll:</b> $${data.final_bankroll_stats.mean.toFixed(2)}</p>
                    <p><b>Worst Case:</b> $${data.final_bankroll_stats.min.toFixed(2)}</p>
                    <p><b>Best Case:</b> $${data.final_bankroll_stats.max.toFixed(2)}</p>
                    <p><b>Recommendations:</b> ${data.recommendations.join('<br>')}</p>
                `;
            }
            
            async function runRealTimeEdge() {
                const response = await fetch('/admin/diagnostics/edge-detector');
                const data = await response.json();
                
                let edgesHTML = '';
                data.detections.forEach(detection => {
                    edgesHTML += `<div style="border: 1px solid #ccc; padding: 10px; margin: 10px 0;">
                        <b>${detection.match}</b> (${detection.minute}', ${detection.score})<br>`;
                    
                    detection.edges.forEach(edge => {
                        edgesHTML += `
                            ${edge.market}: Our ${edge.our_prob}% vs Market ${edge.market_implied}% 
                            <span class="${edge.edge > 0 ? 'positive' : 'negative'}">(${edge.edge > 0 ? '+' : ''}${edge.edge}% edge)</span><br>
                        `;
                    });
                    
                    edgesHTML += '</div>';
                });
                
                document.getElementById('live-edges').innerHTML = edgesHTML || 'No significant edges detected in live matches.';
            }
            
            function createCharts(data) {
                // ROI over time chart
                const roiCtx = document.getElementById('roi-chart').getContext('2d');
                new Chart(roiCtx, {
                    type: 'line',
                    data: {
                        labels: Object.keys(data.by_market),
                        datasets: [{
                            label: 'ROI by Market',
                            data: Object.values(data.by_market).map(m => m.roi || 0),
                            borderColor: 'rgb(75, 192, 192)',
                            backgroundColor: 'rgba(75, 192, 192, 0.2)'
                        }]
                    }
                });
                
                // Calibration chart
                const calCtx = document.getElementById('calibration-chart').getContext('2d');
                const calData = Object.entries(data.calibration || {});
                new Chart(calCtx, {
                    type: 'scatter',
                    data: {
                        datasets: [{
                            label: 'Calibration',
                            data: calData.map(([bucket, stats]) => ({
                                x: stats.expected_win_rate,
                                y: stats.actual_win_rate
                            })),
                            borderColor: 'rgb(255, 99, 132)',
                            backgroundColor: 'rgba(255, 99, 132, 0.5)'
                        }]
                    },
                    options: {
                        scales: {
                            x: { title: { display: true, text: 'Expected Win Rate (%)' }},
                            y: { title: { display: true, text: 'Actual Win Rate (%)' }}
                        }
                    }
                });
            }
            
            function exportData() {
                // Implement CSV export
                alert('Export functionality would be implemented here');
            }
            
            // Run analysis on page load
            window.onload = runEdgeAnalysis;
        </script>
    </body>
    </html>
    """

@app.route("/telegram/webhook/<secret>", methods=["POST"])
def telegram_webhook(secret: str):
    if (WEBHOOK_SECRET or "") != secret: abort(403)
    update=request.get_json(silent=True) or {}
    try:
        msg=(update.get("message") or {}).get("text") or ""
        if msg.startswith("/start"): send_telegram("👋 goalsniper ADVANCED bot (FULL AI mode with ELO ratings and Learning System) is online.")
        elif msg.startswith("/digest"): daily_accuracy_digest()
        elif msg.startswith("/motd"): send_match_of_the_day()
        elif msg.startswith("/scan"):
            parts=msg.split()
            if len(parts)>1 and ADMIN_API_KEY and parts[1]==ADMIN_API_KEY:
                s,l=production_scan(); send_telegram(f"🔁 Scan done. Saved: {s}, Live seen: {l}")
            else: send_telegram("🔒 Admin key required.")
        elif msg.startswith("/learning"):
            if LEARNING_ENABLE:
                patterns = detect_error_patterns(7)
                message = "🧠 <b>Learning System Report</b> (7 days)\n"
                if patterns.get("patterns"):
                    message += f"Found {len(patterns['patterns'])} patterns\n"
                    for i, pattern in enumerate(patterns["patterns"][:3]):
                        message += f"{i+1}. {pattern.get('suggestion', pattern.get('suggestion_text', 'Pattern'))}\n"
                else:
                    message += "No significant patterns detected."
                send_telegram(message)
            else:
                send_telegram("Learning system is disabled.")
    except Exception as e:
        log.warning("telegram webhook parse error: %s", e)
    return jsonify({"ok": True})

# ───────── Boot ─────────
def _on_boot():
    _init_pool(); init_db(); set_setting("boot_ts", str(int(time.time())))
    log.info("✅ Advanced features enabled: ELO ratings, weighted form, shot quality metrics, Learning System")

_on_boot()

if __name__ == "__main__":
    app.run(host=os.getenv("HOST","0.0.0.0"), port=int(os.getenv("PORT","8080")))
