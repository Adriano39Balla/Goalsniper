"""
Postgres-only Flask backend for live football tips — FULL AI MODE.
Pure ML scoring (no math fallbacks), Platt-calibrated models from DB.
Markets: O/U (1.5 removed), BTTS (Yes/No), 1X2 (Draw blocked at send-time).
Automation: harvest snapshots, nightly train, periodic backfill, daily digest, MOTD (prematch optional).
Scan interval: 5 minutes.
"""

import os
import json
import time
import logging
import requests
import psycopg2
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

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ────────────────────────────────
# App / logging
# ────────────────────────────────
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
app = Flask(__name__)

# ────────────────────────────────
# Env / constants
# ────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")
API_KEY            = os.getenv("API_KEY")
ADMIN_API_KEY      = os.getenv("ADMIN_API_KEY")
WEBHOOK_SECRET     = os.getenv("TELEGRAM_WEBHOOK_SECRET")

RUN_SCHEDULER      = os.getenv("RUN_SCHEDULER", "1") not in ("0","false","False","no","NO")

CONF_THRESHOLD     = float(os.getenv("CONF_THRESHOLD", "65"))  # default if no per-market threshold
MAX_TIPS_PER_SCAN  = int(os.getenv("MAX_TIPS_PER_SCAN", "25"))
DUP_COOLDOWN_MIN   = int(os.getenv("DUP_COOLDOWN_MIN", "20"))
TIP_MIN_MINUTE     = int(os.getenv("TIP_MIN_MINUTE", "8"))

SCAN_INTERVAL_SEC  = int(os.getenv("SCAN_INTERVAL_SEC", "300"))

# Harvest & training
HARVEST_MODE       = os.getenv("HARVEST_MODE", "1") not in ("0","false","False","no","NO")
TRAIN_ENABLE       = os.getenv("TRAIN_ENABLE", "1") not in ("0","false","False","no","NO")
TRAIN_HOUR_UTC     = int(os.getenv("TRAIN_HOUR_UTC", "2"))
TRAIN_MINUTE_UTC   = int(os.getenv("TRAIN_MINUTE_UTC", "12"))
TRAIN_MIN_MINUTE   = int(os.getenv("TRAIN_MIN_MINUTE", "15"))

# Backfill & digest
BACKFILL_EVERY_MIN = int(os.getenv("BACKFILL_EVERY_MIN", "15"))
BACKFILL_DAYS      = int(os.getenv("BACKFILL_DAYS", "14"))
DAILY_ACCURACY_DIGEST_ENABLE = os.getenv("DAILY_ACCURACY_DIGEST_ENABLE", "1") not in ("0","false","False","no","NO")
DAILY_ACCURACY_HOUR   = int(os.getenv("DAILY_ACCURACY_HOUR", "3"))
DAILY_ACCURACY_MINUTE = int(os.getenv("DAILY_ACCURACY_MINUTE", "6"))

# Auto-tune thresholds (optional)
AUTO_TUNE_ENABLE        = os.getenv("AUTO_TUNE_ENABLE", "0") not in ("0","false","False","no","NO")
TARGET_PRECISION        = float(os.getenv("TARGET_PRECISION", "0.60"))
THRESH_MIN_PREDICTIONS  = int(os.getenv("THRESH_MIN_PREDICTIONS", "25"))
MIN_THRESH              = float(os.getenv("MIN_THRESH", "55"))
MAX_THRESH              = float(os.getenv("MAX_THRESH", "85"))

# MOTD prematch scoring gate
MOTD_PREMATCH_ENABLE    = os.getenv("MOTD_PREMATCH_ENABLE", "0") not in ("0","false","False","no","NO")

# Markets / lines: remove 1.5
def _parse_lines(env_val: str, default: List[float]) -> List[float]:
    out: List[float] = []
    for tok in (env_val or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(float(tok))
        except Exception:
            pass
    return out or default

OU_LINES: List[float] = _parse_lines(os.getenv("OU_LINES", "0.5,2.5,3.5"), [0.5, 2.5, 3.5])
OU_LINES = [ln for ln in OU_LINES if abs(ln - 1.5) > 1e-6]

TOTAL_MATCH_MINUTES   = int(os.getenv("TOTAL_MATCH_MINUTES", "95"))
PREDICTIONS_PER_MATCH = int(os.getenv("PREDICTIONS_PER_MATCH", "2"))

# Allowed suggestions — Draw removed
ALLOWED_SUGGESTIONS = {"BTTS: Yes", "BTTS: No", "Home Win", "Away Win"}
for _ln in OU_LINES:
    s = f"{_ln}".rstrip("0").rstrip(".")
    ALLOWED_SUGGESTIONS.add(f"Over {s} Goals")
    ALLOWED_SUGGESTIONS.add(f"Under {s} Goals")

# DB
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise SystemExit("DATABASE_URL is required (Postgres).")

# External APIs
BASE_URL         = "https://v3.football.api-sports.io"
FOOTBALL_API_URL = f"{BASE_URL}/fixtures"
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
HEADERS          = {"x-apisports-key": API_KEY, "Accept": "application/json"}
INPLAY_STATUSES  = {"1H","HT","2H","ET","BT","P"}

# HTTP session
session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], respect_retry_after_header=True)
session.mount("https://", HTTPAdapter(max_retries=retries))

# Caches
STATS_CACHE:  Dict[int, Tuple[float, list]] = {}
EVENTS_CACHE: Dict[int, Tuple[float, list]] = {}
SETTINGS_TTL  = int(os.getenv("SETTINGS_TTL_SEC","60"))
MODELS_TTL    = int(os.getenv("MODELS_CACHE_TTL_SEC","120"))

# Timezones
TZ_UTC    = ZoneInfo("UTC")
BERLIN_TZ = ZoneInfo("Europe/Berlin")

# ────────────────────────────────
# Optional import: trainer
# ────────────────────────────────
try:
    from train_models import train_models  # real trainer
except Exception as e:
    _IMPORT_ERR = repr(e)  # capture once, safe global

    def train_models(*args, **kwargs) -> Dict[str, Any]:
        # clean fallback that never references an undefined name
        logger.warning("train_models not available: %s", _IMPORT_ERR)
        return {"ok": False, "reason": f"train_models import failed: {_IMPORT_ERR}"}

# ────────────────────────────────
# DB pool
# ────────────────────────────────
POOL: SimpleConnectionPool | None = None

class PooledConn:
    def __init__(self, pool: SimpleConnectionPool): self.pool = pool; self.conn=None; self.cur=None
    def __enter__(self):
        self.conn = self.pool.getconn()
        self.conn.autocommit = True
        self.cur = self.conn.cursor()
        return self
    def __exit__(self, exc_type, exc, tb):
        try:
            if self.cur: self.cur.close()
        finally:
            if self.conn: self.pool.putconn(self.conn)
    def execute(self, sql: str, params: tuple|list=()):
        self.cur.execute(sql, params or ())
        return self.cur

def _init_pool():
    global POOL, DATABASE_URL
    dsn = DATABASE_URL
    if "sslmode=" not in dsn:
        dsn = dsn + ("&" if "?" in dsn else "?") + "sslmode=require"
    POOL = SimpleConnectionPool(minconn=1, maxconn=int(os.getenv("DB_POOL_MAX", "5")), dsn=dsn)

def db_conn() -> PooledConn:
    if POOL is None:
        _init_pool()
    return PooledConn(POOL)

# ────────────────────────────────
# Settings helpers + TTL cache
# ────────────────────────────────
class _TTLCache:
    def __init__(self, ttl_sec: int): self.ttl = ttl_sec; self.data: dict[str, tuple[float,Any]] = {}
    def get(self, key: str):
        v = self.data.get(key)
        if not v: return None
        ts, val = v
        if time.time() - ts > self.ttl:
            self.data.pop(key, None); return None
        return val
    def set(self, key: str, value: Any): self.data[key] = (time.time(), value)
    def invalidate(self, key: Optional[str]=None):
        if key is None: self.data.clear()
        else: self.data.pop(key, None)

_SETTINGS_CACHE = _TTLCache(SETTINGS_TTL)
_MODELS_CACHE   = _TTLCache(MODELS_TTL)

def get_setting(key: str) -> Optional[str]:
    with db_conn() as conn:
        row = conn.execute("SELECT value FROM settings WHERE key=%s", (key,)).fetchone()
        return row[0] if row else None

def set_setting(key: str, value: str) -> None:
    with db_conn() as conn:
        conn.execute(
            "INSERT INTO settings(key,value) VALUES(%s,%s) ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
            (key, value),
        )

def get_setting_cached(key: str) -> Optional[str]:
    val = _SETTINGS_CACHE.get(key)
    if val is not None:
        return val
    val = get_setting(key)
    _SETTINGS_CACHE.set(key, val)
    return val

def invalidate_model_caches_for_key(key: str):
    """If a settings key looks like model*, invalidate model caches."""
    if key.lower().startswith(("model", "model_latest", "model_v2")):
        _MODELS_CACHE.invalidate()

# ────────────────────────────────
# Init DB (+ indexes)
# ────────────────────────────────
def init_db():
    with db_conn() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS tips (
            match_id   BIGINT,
            league_id  BIGINT,
            league     TEXT,
            home       TEXT,
            away       TEXT,
            market     TEXT,
            suggestion TEXT,
            confidence DOUBLE PRECISION,
            confidence_raw DOUBLE PRECISION,
            score_at_tip TEXT,
            minute     INTEGER,
            created_ts BIGINT,
            sent_ok    INTEGER DEFAULT 1,
            PRIMARY KEY (match_id, created_ts)
        )""")
        conn.execute("""
        CREATE TABLE IF NOT EXISTS tip_snapshots (
            match_id BIGINT,
            created_ts BIGINT,
            payload TEXT,
            PRIMARY KEY (match_id, created_ts)
        )""")
        conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id SERIAL PRIMARY KEY,
            match_id BIGINT UNIQUE,
            verdict  INTEGER,
            created_ts BIGINT
        )""")
        conn.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )""")
        conn.execute("""
        CREATE TABLE IF NOT EXISTS match_results (
            match_id   BIGINT PRIMARY KEY,
            final_goals_h INTEGER,
            final_goals_a INTEGER,
            btts_yes      INTEGER,
            updated_ts    BIGINT
        )""")
        # Add columns if missing
        try: conn.execute("ALTER TABLE tips ADD COLUMN IF NOT EXISTS confidence_raw DOUBLE PRECISION")
        except Exception: pass
        # Indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tips_created ON tips (created_ts DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tips_match ON tips (match_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tips_sent ON tips (sent_ok, created_ts DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_snap_by_match ON tip_snapshots (match_id, created_ts DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_results_updated ON match_results (updated_ts DESC)")

# ────────────────────────────────
# Telegram
# ────────────────────────────────
def send_telegram(message: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML", "disable_web_page_preview": True}
    try:
        res = session.post(f"{TELEGRAM_API_URL}/sendMessage", data=payload, timeout=10)
        return res.ok
    except Exception:
        return False

# ────────────────────────────────
# API helpers
# ────────────────────────────────
def _api_get(url: str, params: dict, timeout: int = 15):
    if not API_KEY:
        return None
    try:
        res = session.get(url, headers=HEADERS, params=params, timeout=timeout)
        if not res.ok:
            return None
        return res.json()
    except Exception:
        return None

# League filters (block youth/reserves/friendlies)
_BLOCK_PATTERNS = ["u17","u18","u19","u20","u21","u23","youth","junior","reserve","res.","friendlies","friendly"]

def _blocked_league(league_obj: dict) -> bool:
    name = str((league_obj or {}).get("name","")).lower()
    country = str((league_obj or {}).get("country","")).lower()
    typ = str((league_obj or {}).get("type","")).lower()
    txt = f"{country} {name} {typ}"
    if any(pat in txt for pat in _BLOCK_PATTERNS):
        return True
    allow = [x.strip() for x in os.getenv("LEAGUE_ALLOW_IDS","").split(",") if x.strip()]
    deny  = [x.strip() for x in os.getenv("LEAGUE_DENY_IDS","").split(",") if x.strip()]
    lid = str((league_obj or {}).get("id") or "")
    if lid in deny:
        return True
    if allow:
        return lid not in allow
    return False

# Stats/events fetch
def fetch_match_stats(fixture_id: int) -> Optional[List[Dict[str, Any]]]:
    now = time.time()
    if fixture_id in STATS_CACHE:
        ts, data = STATS_CACHE[fixture_id]
        if now - ts < 90:
            return data
    js = _api_get(f"{FOOTBALL_API_URL}/statistics", {"fixture": fixture_id})
    stats = js.get("response", []) if isinstance(js, dict) else None
    STATS_CACHE[fixture_id] = (now, stats or [])
    return stats

def fetch_match_events(fixture_id: int) -> Optional[List[Dict[str, Any]]]:
    now = time.time()
    if fixture_id in EVENTS_CACHE:
        ts, data = EVENTS_CACHE[fixture_id]
        if now - ts < 90:
            return data
    js = _api_get(f"{FOOTBALL_API_URL}/events", {"fixture": fixture_id})
    evs = js.get("response", []) if isinstance(js, dict) else None
    EVENTS_CACHE[fixture_id] = (now, evs or [])
    return evs

def fetch_live_matches() -> List[Dict[str, Any]]:
    js = _api_get(FOOTBALL_API_URL, {"live": "all"})
    if not isinstance(js, dict):
        return []
    matches = js.get("response", []) or []
    matches = [m for m in matches if not _blocked_league(m.get("league") or {})]
    out = []
    for m in matches:
        status  = (m.get("fixture", {}) or {}).get("status", {}) or {}
        elapsed = status.get("elapsed")
        short   = (status.get("short") or "").upper()
        if elapsed is None or elapsed > 120: continue
        if short not in INPLAY_STATUSES: continue
        fid = (m.get("fixture", {}) or {}).get("id")
        m["statistics"] = fetch_match_stats(fid) or []
        m["events"]     = fetch_match_events(fid) or []
        out.append(m)
    return out

# ────────────────────────────────
# Features
# ────────────────────────────────
def _num(v) -> float:
    try:
        if isinstance(v, str) and v.endswith("%"): return float(v[:-1])
        return float(v or 0)
    except Exception: return 0.0

def _pos_pct(v) -> float:
    try: return float(str(v).replace("%","").strip() or 0)
    except Exception: return 0.0

def extract_features(match: Dict[str, Any]) -> Dict[str, float]:
    home_name = match["teams"]["home"]["name"]
    away_name = match["teams"]["away"]["name"]
    gh = match["goals"]["home"] or 0
    ga = match["goals"]["away"] or 0
    minute = int(((match.get("fixture") or {}).get("status") or {}).get("elapsed") or 0)

    stats_blocks = match.get("statistics") or []
    stats: Dict[str, Dict[str, Any]] = {}
    for s in stats_blocks:
        tname = (s.get("team") or {}).get("name")
        if tname:
            stats[tname] = { (i.get("type") or ""): i.get("value") for i in (s.get("statistics") or []) }
    sh = stats.get(home_name, {}) or {}
    sa = stats.get(away_name, {}) or {}

    xg_h  = _num(sh.get("Expected Goals", 0))
    xg_a  = _num(sa.get("Expected Goals", 0))
    sot_h = _num(sh.get("Shots on Target", 0))
    sot_a = _num(sa.get("Shots on Target", 0))
    cor_h = _num(sh.get("Corner Kicks", 0))
    cor_a = _num(sa.get("Corner Kicks", 0))
    pos_h = _pos_pct(sh.get("Ball Possession", 0))
    pos_a = _pos_pct(sa.get("Ball Possession", 0))

    red_h = red_a = 0
    for ev in (match.get("events") or []):
        try:
            if (ev.get("type","").lower() == "card"):
                detail = (ev.get("detail","") or "").lower()
                if ("red" in detail) or ("second yellow" in detail):
                    tname = (ev.get("team") or {}).get("name") or ""
                    if tname == home_name: red_h += 1
                    elif tname == away_name: red_a += 1
        except Exception:
            pass

    return {
        "minute": float(minute),
        "goals_h": float(gh), "goals_a": float(ga),
        "goals_sum": float(gh + ga), "goals_diff": float(gh - ga),
        "xg_h": float(xg_h), "xg_a": float(xg_a),
        "xg_sum": float(xg_h + xg_a), "xg_diff": float(xg_h - xg_a),
        "sot_h": float(sot_h), "sot_a": float(sot_a), "sot_sum": float(sot_h + sot_a),
        "cor_h": float(cor_h), "cor_a": float(cor_a), "cor_sum": float(cor_h + cor_a),
        "pos_h": float(pos_h), "pos_a": float(pos_a), "pos_diff": float(pos_h - pos_a),
        "red_h": float(red_h), "red_a": float(red_a), "red_sum": float(red_h + red_a),
    }

def stats_coverage_ok(feat: Dict[str, float], minute: int) -> bool:
    require_stats_minute = int(os.getenv("REQUIRE_STATS_MINUTE", "35"))
    require_fields = int(os.getenv("REQUIRE_DATA_FIELDS", "2"))
    if minute < require_stats_minute: return True
    fields = [feat.get("xg_sum",0.0), feat.get("sot_sum",0.0), feat.get("cor_sum",0.0),
              max(feat.get("pos_h",0.0), feat.get("pos_a",0.0))]
    nonzero = sum(1 for v in fields if (v or 0) > 0)
    return nonzero >= max(0, require_fields)

def _pretty_score(m: Dict[str, Any]) -> str:
    gh = (m.get("goals") or {}).get("home") or 0
    ga = (m.get("goals") or {}).get("away") or 0
    return f"{gh}-{ga}"

def _league_name(m: Dict[str, Any]) -> Tuple[int, str]:
    lg = (m.get("league") or {}) or {}
    league_id = int(lg.get("id") or 0)
    league = f"{lg.get('country','')} - {lg.get('name','')}".strip(" -")
    return league_id, league

def _teams(m: Dict[str, Any]) -> Tuple[str, str]:
    t = (m.get("teams") or {}) or {}
    return (t.get("home", {}).get("name", ""), t.get("away", {}).get("name", ""))

# ────────────────────────────────
# ML model utilities + TTL cache
# ────────────────────────────────
MODEL_KEYS_ORDER = ["model_v2:{name}", "model_latest:{name}", "model:{name}"]
EPS = 1e-12

def _sigmoid(x: float) -> float:
    try:
        if x < -50: return 1e-22
        if x >  50: return 1.0 - 1e-22
        import math
        return 1.0 / (1.0 + math.exp(-x))
    except Exception:
        return 0.5

def _logit(p: float) -> float:
    import math
    p = max(EPS, min(1 - EPS, float(p)))
    return math.log(p / (1 - p))

def load_model_from_settings(name: str) -> Optional[Dict[str, Any]]:
    cached = _MODELS_CACHE.get(name)
    if cached is not None:
        return cached
    mdl = None
    for pat in MODEL_KEYS_ORDER:
        key = pat.format(name=name)
        raw = get_setting_cached(key)
        if not raw:
            continue
        try:
            tmp = json.loads(raw)
            tmp.setdefault("intercept", 0.0)
            tmp.setdefault("weights", {})
            cal = tmp.get("calibration") or {}
            if isinstance(cal, dict):
                cal.setdefault("method", "sigmoid")
                cal.setdefault("a", 1.0)
                cal.setdefault("b", 0.0)
                tmp["calibration"] = cal
            mdl = tmp
            break
        except Exception as e:
            logger.warning("[MODEL] failed to parse %s: %s", key, e)
    if mdl is not None:
        _MODELS_CACHE.set(name, mdl)
    return mdl

def _linpred(feat: Dict[str, float], weights: Dict[str, float], intercept: float) -> float:
    s = float(intercept or 0.0)
    for k, w in (weights or {}).items():
        s += float(w or 0.0) * float(feat.get(k, 0.0))
    return s

def _calibrate(p: float, cal: Dict[str, Any]) -> float:
    method = (cal or {}).get("method", "sigmoid")
    a = float((cal or {}).get("a", 1.0)); b = float((cal or {}).get("b", 0.0))
    if method == "platt":
        return _sigmoid(a * _logit(p) + b)
    import math
    p = max(EPS, min(1 - EPS, float(p)))
    z = math.log(p / (1.0 - p))
    return _sigmoid(a * z + b)

def _score_prob(feat: Dict[str, float], mdl: Dict[str, Any]) -> float:
    lp = _linpred(feat, mdl.get("weights", {}), float(mdl.get("intercept", 0.0)))
    p = _sigmoid(lp)
    cal = mdl.get("calibration") or {}
    try:
        if cal:
            p = _calibrate(p, cal)
    except Exception:
        pass
    return max(0.0, min(1.0, float(p)))

def _fmt_line(line: float) -> str:
    return f"{line}".rstrip("0").rstrip(".")

def _load_ou_model_for_line(line: float) -> Optional[Dict[str, Any]]:
    name = f"OU_{_fmt_line(line)}"
    mdl = load_model_from_settings(name)
    if not mdl and abs(line - 2.5) < 1e-6:
        mdl = load_model_from_settings("O25")
    return mdl

def _load_wld_models() -> Tuple[Optional[Dict], Optional[Dict], Optional[Dict]]:
    return (
        load_model_from_settings("WLD_HOME"),
        load_model_from_settings("WLD_DRAW"),
        load_model_from_settings("WLD_AWAY"),
    )

# ────────────────────────────────
# Harvest
# ────────────────────────────────
def save_snapshot_from_match(m: Dict[str, Any], feat: Dict[str, float]) -> None:
    fx = m.get("fixture", {}) or {}; lg = m.get("league", {}) or {}
    fid = int(fx.get("id")); league_id = int(lg.get("id") or 0)
    league = f"{lg.get('country', '')} - {lg.get('name', '')}".strip(" -")
    home = (m.get("teams") or {}).get("home", {}).get("name", "")
    away = (m.get("teams") or {}).get("away", {}).get("name", "")
    gh = (m.get("goals") or {}).get("home") or 0
    ga = (m.get("goals") or {}).get("away") or 0
    minute = int(feat.get("minute", 0))

    snapshot = {
        "minute": minute, "gh": gh, "ga": ga, "league_id": league_id,
        "market": "HARVEST", "suggestion": "HARVEST", "confidence": 0,
        "stat": {
            "xg_h": feat.get("xg_h", 0), "xg_a": feat.get("xg_a", 0),
            "sot_h": feat.get("sot_h", 0), "sot_a": feat.get("sot_a", 0),
            "cor_h": feat.get("cor_h", 0), "cor_a": feat.get("cor_a", 0),
            "pos_h": feat.get("pos_h", 0), "pos_a": feat.get("pos_a", 0),
            "red_h": feat.get("red_h", 0), "red_a": feat.get("red_a", 0)
        }
    }

    now = int(time.time())
    with db_conn() as conn:
        # store the raw snapshot (for training)
        conn.execute(
            "INSERT INTO tip_snapshots(match_id, created_ts, payload) "
            "VALUES (%s,%s,%s) "
            "ON CONFLICT (match_id, created_ts) DO UPDATE SET payload=EXCLUDED.payload",
            (fid, now, json.dumps(snapshot)[:200000]),
        )
        # store a HARVEST marker row in tips — but mark as already 'sent'
        conn.execute(
            "INSERT INTO tips(match_id,league_id,league,home,away,market,suggestion,confidence,score_at_tip,minute,created_ts,sent_ok) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,1)",
            (fid, league_id, league, home, away, "HARVEST", "HARVEST", 0.0, f"{gh}-{ga}", minute, now),
        )

# ────────────────────────────────
# Outcomes & backfill
# ────────────────────────────────
def _parse_ou_line_from_suggestion(s: str) -> Optional[float]:
    try:
        for tok in (s or "").split():
            try: return float(tok)
            except Exception: pass
    except Exception: pass
    return None

def _tip_outcome_for_result(suggestion: str, res: Dict[str, Any]) -> Optional[int]:
    gh = int(res.get("final_goals_h") or 0)
    ga = int(res.get("final_goals_a") or 0)
    total = gh + ga
    btts = int(res.get("btts_yes") or 0)
    s = (suggestion or "").strip()
    if s.startswith("Over") or s.startswith("Under"):
        line = _parse_ou_line_from_suggestion(s)
        if line is None: return None
        if s.startswith("Over"):
            if total > line: return 1
            if abs(total - line) < 1e-9: return None
            return 0
        else:
            if total < line: return 1
            if abs(total - line) < 1e-9: return None
            return 0
    if s == "BTTS: Yes": return 1 if btts == 1 else 0
    if s == "BTTS: No":  return 1 if btts == 0 else 0
    if s == "Home Win":  return 1 if gh > ga else 0
    if s == "Away Win":  return 1 if ga > gh else 0
    if s == "Draw":      return 1 if gh == ga else 0
    return None

def _fixture_by_id(match_id: int) -> Optional[Dict[str, Any]]:
    js = _api_get(FOOTBALL_API_URL, {"id": match_id})
    if not isinstance(js, dict): return None
    arr = js.get("response") or []
    return arr[0] if arr else None

def _is_final_status(short: str) -> bool:
    short = (short or "").upper()
    return short in {"FT", "AET", "PEN"}

def backfill_results_for_open_matches(max_rows: int = 200) -> int:
    now_ts = int(time.time())
    cutoff = now_ts - BACKFILL_DAYS * 24 * 3600
    updated = 0
    with db_conn() as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT t.match_id
            FROM tips t
            LEFT JOIN match_results r ON r.match_id = t.match_id
            WHERE r.match_id IS NULL AND t.created_ts >= %s
            ORDER BY t.created_ts DESC
            LIMIT %s
            """,
            (cutoff, max_rows),
        ).fetchall()

    for (mid,) in rows:
        try:
            fx = _fixture_by_id(int(mid))
            if not fx: continue
            status = ((fx.get("fixture") or {}).get("status") or {}).get("short", "")
            if not _is_final_status(status): continue
            goals = fx.get("goals") or {}
            gh = int(goals.get("home") or 0)
            ga = int(goals.get("away") or 0)
            btts = 1 if (gh > 0 and ga > 0) else 0
            with db_conn() as conn2:
                conn2.execute(
                    "INSERT INTO match_results(match_id, final_goals_h, final_goals_a, btts_yes, updated_ts) "
                    "VALUES(%s,%s,%s,%s,%s) "
                    "ON CONFLICT (match_id) DO UPDATE SET final_goals_h=EXCLUDED.final_goals_h, "
                    "final_goals_a=EXCLUDED.final_goals_a, btts_yes=EXCLUDED.btts_yes, updated_ts=EXCLUDED.updated_ts",
                    (int(mid), gh, ga, btts, int(time.time())),
                )
            updated += 1
        except Exception as e:
            logger.warning("[RESULTS] update failed for %s: %s", mid, e)
            continue
    if updated: logger.info("[RESULTS] backfilled %d results", updated)
    return updated

def daily_accuracy_digest() -> Optional[str]:
    if not DAILY_ACCURACY_DIGEST_ENABLE: return None
    now_local = datetime.now(BERLIN_TZ)
    y_start = (now_local - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    y_end = y_start + timedelta(days=1)
    y_start_ts = int(y_start.timestamp()); y_end_ts = int(y_end.timestamp())

    backfill_results_for_open_matches(max_rows=400)

    with db_conn() as conn:
        rows = conn.execute(
            """
            SELECT t.match_id, t.market, t.suggestion, t.confidence, t.created_ts,
                   r.final_goals_h, r.final_goals_a, r.btts_yes
            FROM tips t
            LEFT JOIN match_results r ON r.match_id = t.match_id
            WHERE t.created_ts >= %s AND t.created_ts < %s
              AND t.suggestion <> 'HARVEST'
              AND t.sent_ok = 1
            """,
            (y_start_ts, y_end_ts),
        ).fetchall()

    total = graded = wins = 0
    by_market: Dict[str, Dict[str, int]] = {}
    for (mid, market, sugg, conf, cts, gh, ga, btts) in rows:
        res = {"final_goals_h": gh, "final_goals_a": ga, "btts_yes": btts}
        out = _tip_outcome_for_result(sugg, res)
        if out is None: continue
        total += 1; graded += 1; wins += 1 if out == 1 else 0
        m = by_market.setdefault(market or "?", {"graded": 0, "wins": 0})
        m["graded"] += 1; m["wins"] += 1 if out == 1 else 0

    if graded == 0:
        msg = "📊 Daily Digest\nNo graded tips for yesterday."
    else:
        acc = 100.0 * wins / max(1, graded)
        lines = [f"📊 <b>Daily Digest</b> (yesterday, Berlin time)",
                 f"Tips sent: {total}  •  Graded: {graded}  •  Wins: {wins}  •  Accuracy: {acc:.1f}%"]
        for mk, st in sorted(by_market.items()):
            if st["graded"] == 0: continue
            a = 100.0 * st["wins"] / st["graded"]
            lines.append(f"• {escape(mk)} — {st['wins']}/{st['graded']} ({a:.1f}%)")
        msg = "\n".join(lines)
    send_telegram(msg); return msg

# ── Match of the Day (MOTD): FUTURE matches (today only)
MOTD_PREDICT  = os.getenv("MOTD_PREDICT", "1") not in ("0","false","False","no","NO")
MOTD_HOUR     = int(os.getenv("MOTD_HOUR", "19"))
MOTD_MINUTE   = int(os.getenv("MOTD_MINUTE", "15"))
MOTD_CONF_MIN = float(os.getenv("MOTD_CONF_MIN", "70"))
try:
    MOTD_LEAGUE_IDS = [int(x) for x in (os.getenv("MOTD_LEAGUE_IDS","").split(",")) if x.strip().isdigit()]
except Exception:
    MOTD_LEAGUE_IDS = []

def _kickoff_berlin(utc_iso: str|None) -> str:
    try:
        if not utc_iso: return "TBD"
        dt_utc = datetime.fromisoformat(utc_iso.replace("Z","+00:00"))
        return dt_utc.astimezone(BERLIN_TZ).strftime("%H:%M")
    except Exception: return "TBD"

def _api_fixtures_for_date_utc(date_utc: datetime) -> List[dict]:
    js = _api_get(FOOTBALL_API_URL, {"date": date_utc.strftime("%Y-%m-%d")})
    if not isinstance(js, dict): return []
    out = []
    for r in js.get("response", []) or []:
        st = (((r.get("fixture") or {}).get("status") or {}).get("short") or "").upper()
        if st == "NS": out.append(r)
    return out

def _prematch_features() -> Dict[str, float]:
    return {"minute":0.0,"goals_h":0.0,"goals_a":0.0,"goals_sum":0.0,"goals_diff":0.0,
            "xg_h":0.0,"xg_a":0.0,"xg_sum":0.0,"xg_diff":0.0,"sot_h":0.0,"sot_a":0.0,"sot_sum":0.0,
            "cor_h":0.0,"cor_a":0.0,"cor_sum":0.0,"pos_h":0.0,"pos_a":0.0,"pos_diff":0.0,"red_h":0.0,"red_a":0.0,"red_sum":0.0}

def _format_motd_message(home, away, league, kickoff_txt, suggestion, prob_pct) -> str:
    return ("🏅 <b>Match of the Day</b>\n"
            f"<b>Match:</b> {escape(home)} vs {escape(away)}\n"
            f"🏆 <b>League:</b> {escape(league)}\n"
            f"⏰ <b>Kickoff (Berlin):</b> {kickoff_txt}\n"
            f"<b>Tip:</b> {escape(suggestion)}\n"
            f"📈 <b>Confidence:</b> {prob_pct:.1f}%")

def send_match_of_the_day() -> bool:
    if not MOTD_PREDICT:
        return False
    if not MOTD_PREMATCH_ENABLE:
        return send_telegram("🏅 Match of the Day: prematch models disabled (MOTD_PREMATCH_ENABLE=0).")

    # Berlin 'today' -> query overlapping UTC dates
    today_local = datetime.now(BERLIN_TZ).date()
    start_local = datetime.combine(today_local, datetime.min.time(), tzinfo=BERLIN_TZ)
    end_local   = start_local + timedelta(days=1)
    dates_utc = {start_local.astimezone(TZ_UTC).date(),
                 (end_local - timedelta(seconds=1)).astimezone(TZ_UTC).date()}

    fixtures: List[dict] = []
    for d in sorted(dates_utc):
        fixtures.extend(_api_fixtures_for_date_utc(datetime(d.year,d.month,d.day,tzinfo=TZ_UTC)))
    fixtures = [f for f in fixtures if not _blocked_league(f.get("league") or {})]
    if MOTD_LEAGUE_IDS:
        fixtures = [f for f in fixtures if int(((f.get("league") or {}).get("id") or 0)) in MOTD_LEAGUE_IDS]
    if not fixtures:
        return send_telegram("🏅 Match of the Day: no eligible fixtures today.")

    feat0 = _prematch_features()
    best = None  # (prob_pct, suggestion, home, away, league, kickoff_txt)

    for fx in fixtures:
        fixture = fx.get("fixture") or {}; lg = fx.get("league") or {}; teams = fx.get("teams") or {}
        home = (teams.get("home") or {}).get("name",""); away = (teams.get("away") or {}).get("name","")
        league = f"{lg.get('country','')} - {lg.get('name','')}".strip(" -")
        kickoff_txt = _kickoff_berlin((fixture.get("date") or ""))

        candidates: List[Tuple[str,str,float]] = []
        # OU
        for line in OU_LINES:
            mdl_line = _load_ou_model_for_line(line)
            if not mdl_line: continue
            p_over = _score_prob(feat0, mdl_line)
            market_name = f"Over/Under {_fmt_line(line)}"
            thr = _get_market_threshold(market_name)
            if p_over*100.0 >= thr: candidates.append((market_name, f"Over {_fmt_line(line)} Goals", p_over))
            p_under = 1.0 - p_over
            if p_under*100.0 >= thr: candidates.append((market_name, f"Under {_fmt_line(line)} Goals", p_under))
        # BTTS
        mdl_btts = load_model_from_settings("BTTS_YES")
        if mdl_btts:
            p_btts = _score_prob(feat0, mdl_btts); thr_b = _get_market_threshold("BTTS")
            if p_btts*100.0 >= thr_b: candidates.append(("BTTS","BTTS: Yes", p_btts))
            p_btts_no = 1.0 - p_btts
            if p_btts_no*100.0 >= thr_b: candidates.append(("BTTS","BTTS: No", p_btts_no))
        # 1X2
        mh, md, ma = _load_wld_models()
        if mh and md and ma:
            ph = _score_prob(feat0, mh); pd = _score_prob(feat0, md); pa = _score_prob(feat0, ma)
            s = max(EPS, ph+pd+pa); ph,pd,pa = ph/s,pd/s,pa/s
            thr_1x2 = _get_market_threshold("1X2")
            if ph*100.0 >= thr_1x2: candidates.append(("1X2","Home Win", ph))
            if pa*100.0 >= thr_1x2: candidates.append(("1X2","Away Win", pa))

        if not candidates: continue
        _, suggestion, prob = max(candidates, key=lambda x:x[2])
        prob_pct = prob*100.0
        if prob_pct < MOTD_CONF_MIN: continue
        item = (prob_pct, suggestion, home, away, league, kickoff_txt)
        if (best is None) or (prob_pct > best[0]): best = item

    if not best:
        return send_telegram("🏅 Match of the Day: no pick met the confidence/thresholds today.")
    prob_pct, suggestion, home, away, league, kickoff_txt = best
    return send_telegram(_format_motd_message(home, away, league, kickoff_txt, suggestion, prob_pct))

# ────────────────────────────────
# Training + auto-tune + retry outbox
# ────────────────────────────────
def auto_train_job() -> None:
    if not TRAIN_ENABLE:
        send_telegram("🤖 Training skipped: TRAIN_ENABLE=0"); return
    send_telegram("🤖 Training started.")
    try:
        res = train_models() or {}; ok = bool(res.get("ok"))
        if not ok:
            reason = res.get("reason") or res.get("error") or "unknown reason"
            send_telegram(f"⚠️ Training finished: <b>SKIPPED</b>\nReason: {escape(str(reason))}")
            return
        trained = [k for k,v in (res.get("trained") or {}).items() if v]
        thr     = (res.get("thresholds") or {}); mets = (res.get("metrics") or {})
        lines = ["🤖 <b>Model training OK</b>"]
        if trained: lines.append("• Trained: " + ", ".join(sorted(trained)))
        if thr:     lines.append("• Thresholds: " + "  |  ".join([f"{escape(k)}: {float(v):.1f}%" for k,v in thr.items()]))
        def _m(n): m=mets.get(n) or {}; 
        metric_lines=[]
        for n in ["BTTS_YES","OU_2.5","OU_3.5","WLD_HOME","WLD_AWAY"]:
            m = mets.get(n) or {}
            if m: metric_lines.append(f"{n}: acc {m.get('acc',0):.2f}, brier {m.get('brier',0):.3f}, logloss {m.get('logloss',0):.3f}")
        if metric_lines: lines.append("• Metrics:\n  " + "\n  ".join(metric_lines))
        send_telegram("\n".join(lines))
    except Exception as e:
        logger.exception("[TRAIN] job failed: %s", e)
        send_telegram(f"❌ Training <b>FAILED</b>\n{escape(str(e))}")

def _pick_threshold(y_true: List[int], y_prob: List[float],
                    target_precision: float, min_preds: int,
                    default_pct: float) -> float:
    import numpy as np
    y = np.asarray(y_true, dtype=int); p = np.asarray(y_prob, dtype=float)
    best = default_pct/100.0
    candidates = np.arange(MIN_THRESH, MAX_THRESH+1e-9, 1.0)/100.0
    feasible = []
    for t in candidates:
        pred = (p >= t).astype(int); n = int(pred.sum())
        if n < min_preds: continue
        tp = int(((pred==1)&(y==1)).sum()); prec = tp / max(1,n)
        if prec >= target_precision: feasible.append((t,prec,n))
    if feasible:
        feasible.sort(key=lambda z: (z[0], -z[1])); best = feasible[0][0]
    return float(best*100.0)

def auto_tune_thresholds(days: int = 14) -> Dict[str, float]:
    """Simple precision-based tuner using stored confidence_raw on last N days."""
    if not AUTO_TUNE_ENABLE:
        return {}
    cutoff = int(time.time()) - days*24*3600
    with db_conn() as conn:
        rows = conn.execute(
            """
            SELECT t.market, t.suggestion, COALESCE(t.confidence_raw, t.confidence/100.0) as prob,
                   r.final_goals_h, r.final_goals_a, r.btts_yes
            FROM tips t
            JOIN match_results r ON r.match_id = t.match_id
            WHERE t.created_ts >= %s AND t.suggestion <> 'HARVEST' AND t.sent_ok=1
            """,
            (cutoff,)
        ).fetchall()
    by_market: Dict[str, List[Tuple[float,int]]] = {}
    for (market, sugg, prob, gh, ga, btts) in rows:
        res = {"final_goals_h": gh, "final_goals_a": ga, "btts_yes": btts}
        out = _tip_outcome_for_result(sugg, res)
        if out is None: continue
        by_market.setdefault(market, []).append((float(prob), int(out)))

    tuned: Dict[str,float] = {}
    for mk, arr in by_market.items():
        if len(arr) < THRESH_MIN_PREDICTIONS: continue
        probs = [p for (p,_) in arr]; wins = [y for (_,y) in arr]
        pct = _pick_threshold(wins, probs, TARGET_PRECISION, THRESH_MIN_PREDICTIONS, default_pct=CONF_THRESHOLD)
        set_setting(f"conf_threshold:{mk}", f"{pct:.2f}")
        _SETTINGS_CACHE.invalidate(f"conf_threshold:{mk}")
        tuned[mk] = pct
    if tuned:
        send_telegram("🔧 Auto-tune updated thresholds:\n" + "\n".join([f"• {k}: {v:.1f}%" for k,v in tuned.items()]))
    else:
        send_telegram("🔧 Auto-tune: no updates (insufficient data).")
    return tuned

def retry_unsent_tips(minutes: int = 30, limit: int = 200) -> int:
    """Retry Telegram sends for tips with sent_ok=0 in the recent window."""
    cutoff = int(time.time()) - minutes*60
    with db_conn() as conn:
        rows = conn.execute(
            "SELECT match_id,league,home,away,market,suggestion,confidence,score_at_tip,minute,created_ts "
            "FROM tips WHERE sent_ok=0 AND created_ts >= %s ORDER BY created_ts ASC LIMIT %s",
            (cutoff, limit)
        ).fetchall()
    retried = 0
    for (mid, league, home, away, market, sugg, conf, score, minute, cts) in rows:
        msg = ("⚽️ <b>New Tip!</b>\n"
               f"<b>Match:</b> {escape(home)} vs {escape(away)}\n"
               f"🕒 <b>Minute:</b> {int(minute)}'  |  <b>Score:</b> {escape(score)}\n"
               f"<b>Tip:</b> {escape(sugg)}\n"
               f"📈 <b>Confidence:</b> {float(conf):.1f}%\n"
               f"🏆 <b>League:</b> {escape(league)}")
        ok = send_telegram(msg)
        if ok:
            with db_conn() as conn2:
                conn2.execute("UPDATE tips SET sent_ok=1 WHERE match_id=%s AND created_ts=%s", (mid, cts))
            retried += 1
    if retried:
        logger.info("[RETRY] resent %d tips", retried)
    return retried

# ────────────────────────────────
# Threshold helpers, tip format/send, production scan
# ────────────────────────────────
def _get_market_threshold_key(market: str) -> str: return f"conf_threshold:{market}"

def _get_market_threshold(market: str) -> float:
    try:
        val = get_setting_cached(_get_market_threshold_key(market))
        return float(val) if val is not None else float(CONF_THRESHOLD)
    except Exception:
        return float(CONF_THRESHOLD)

def _format_tip_message(home, away, league, minute, score_txt, suggestion, prob_pct, feat) -> str:
    stat_line = ""
    if any([feat.get("xg_h", 0), feat.get("xg_a", 0), feat.get("sot_h", 0), feat.get("sot_a", 0),
            feat.get("cor_h", 0), feat.get("cor_a", 0), feat.get("pos_h", 0), feat.get("pos_a", 0),
            feat.get("red_h", 0), feat.get("red_a", 0)]):
        stat_line = (f"\n📊 xG {feat.get('xg_h',0):.2f}-{feat.get('xg_a',0):.2f}"
                     f" • SOT {int(feat.get('sot_h',0))}-{int(feat.get('sot_a',0))}"
                     f" • CK {int(feat.get('cor_h',0))}-{int(feat.get('cor_a',0))}")
        if feat.get("pos_h", 0) or feat.get("pos_a", 0):
            stat_line += f" • POS {int(feat.get('pos_h',0))}%–{int(feat.get('pos_a',0))}%"
        if feat.get("red_h", 0) or feat.get("red_a", 0):
            stat_line += f" • RED {int(feat.get('red_h',0))}-{int(feat.get('red_a',0))}"
    return ("⚽️ <b>New Tip!</b>\n"
            f"<b>Match:</b> {escape(home)} vs {escape(away)}\n"
            f"🕒 <b>Minute:</b> {minute}'  |  <b>Score:</b> {escape(score_txt)}\n"
            f"<b>Tip:</b> {escape(suggestion)}\n"
            f"📈 <b>Confidence:</b> {prob_pct:.1f}%\n"
            f"🏆 <b>League:</b> {escape(league)}{stat_line}")

def _send_tip(home, away, league, minute, score_txt, suggestion, prob_pct, feat) -> bool:
    return send_telegram(_format_tip_message(home, away, league, minute, score_txt, suggestion, prob_pct, feat))

def production_scan() -> Tuple[int, int]:
    matches = fetch_live_matches()
    live_seen = len(matches)
    if live_seen == 0:
        logger.info("[PROD] no live matches"); return 0, 0
    saved = 0; now_ts = int(time.time())
    with db_conn() as conn:
        for m in matches:
            try:
                fid = int((m.get("fixture", {}) or {}).get("id") or 0)
                if not fid: continue
                if DUP_COOLDOWN_MIN > 0:
                    cutoff = now_ts - (DUP_COOLDOWN_MIN * 60)
                    dup = conn.execute("SELECT 1 FROM tips WHERE match_id=%s AND created_ts>=%s LIMIT 1",(fid,cutoff)).fetchone()
                    if dup: continue
                feat = extract_features(m); minute = int(feat.get("minute",0))
                if not stats_coverage_ok(feat, minute): continue
                if minute < TIP_MIN_MINUTE: continue
                if HARVEST_MODE and minute >= TRAIN_MIN_MINUTE and minute % 3 == 0:
                    try: save_snapshot_from_match(m, feat)
                    except Exception: pass
                league_id, league = _league_name(m); home, away = _teams(m); score_txt = _pretty_score(m)
                candidates: List[Tuple[str,str,float]] = []
                for line in OU_LINES:
                    mdl_line = _load_ou_model_for_line(line)
                    if not mdl_line: continue
                    p_over = _score_prob(feat, mdl_line); market_name = f"Over/Under {_fmt_line(line)}"
                    thr = _get_market_threshold(market_name)
                    if p_over*100.0 >= thr: candidates.append((market_name, f"Over {_fmt_line(line)} Goals", p_over))
                    p_under = 1.0 - p_over
                    if p_under*100.0 >= thr: candidates.append((market_name, f"Under {_fmt_line(line)} Goals", p_under))
                mdl_btts = load_model_from_settings("BTTS_YES")
                if mdl_btts:
                    p_btts = _score_prob(feat, mdl_btts); thr_b = _get_market_threshold("BTTS")
                    if p_btts*100.0 >= thr_b: candidates.append(("BTTS","BTTS: Yes", p_btts))
                    p_btts_no = 1.0 - p_btts
                    if p_btts_no*100.0 >= thr_b: candidates.append(("BTTS","BTTS: No", p_btts_no))
                mh, md, ma = _load_wld_models()
                if mh and md and ma:
                    ph = _score_prob(feat, mh); pd = _score_prob(feat, md); pa = _score_prob(feat, ma)
                    s = max(EPS, ph+pd+pa); ph,pd,pa = ph/s,pd/s,pa/s
                    thr_1x2 = _get_market_threshold("1X2")
                    if ph*100.0 >= thr_1x2: candidates.append(("1X2","Home Win", ph))
                    if pa*100.0 >= thr_1x2: candidates.append(("1X2","Away Win", pa))
                candidates.sort(key=lambda x:x[2], reverse=True)
                per_match = 0; base_now = int(time.time())
                for idx,(market_txt,suggestion,prob) in enumerate(candidates):
                    if suggestion not in ALLOWED_SUGGESTIONS: continue
                    if per_match >= max(1,PREDICTIONS_PER_MATCH): break
                    created_ts = base_now + idx; prob_pct = round(prob*100.0, 1)
                    with db_conn() as conn2:
                        conn2.execute(
                            "INSERT INTO tips(match_id,league_id,league,home,away,market,suggestion,confidence,score_at_tip,minute,created_ts,sent_ok) "
                            "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,0)",
                            (fid, league_id, league, home, away, market_txt, suggestion, float(prob_pct), score_txt, minute, created_ts),
                        )
                        # Do not send any HARVEST markers to Telegram
                        if suggestion != "HARVEST":
                            sent_ok = _send_tip(home, away, league, minute, score_txt, suggestion, float(prob_pct), feat)
                            if sent_ok:
                                conn2.execute(
                                    "UPDATE tips SET sent_ok=1 WHERE match_id=%s AND created_ts=%s",
                                    (fid, created_ts),
                                )
                    saved += 1; per_match += 1
                    if MAX_TIPS_PER_SCAN and saved >= MAX_TIPS_PER_SCAN: break
                if MAX_TIPS_PER_SCAN and saved >= MAX_TIPS_PER_SCAN: break
            except Exception as e:
                logger.exception("[PROD] failure on match: %s", e); continue
    logger.info(f"[PROD] saved={saved} live_seen={live_seen}"); return saved, live_seen

# ────────────────────────────────
# Leader lock (advisory)
# ────────────────────────────────
def _run_with_pg_lock(lock_key: int, fn, *args, **kwargs):
    try:
        with db_conn() as conn:
            got = conn.execute("SELECT pg_try_advisory_lock(%s)", (lock_key,)).fetchone()[0]
            if not got:
                logger.info("[LOCK %s] another instance is running; skipped.", lock_key); return None
            try: return fn(*args, **kwargs)
            finally: conn.execute("SELECT pg_advisory_unlock(%s)", (lock_key,))
    except Exception as e:
        logger.exception("[LOCK %s] failed: %s", lock_key, e); return None

# ────────────────────────────────
# Scheduler (gated by RUN_SCHEDULER)
# ────────────────────────────────
_scheduler_started = False
def _start_scheduler_once() -> None:
    global _scheduler_started
    if _scheduler_started or not RUN_SCHEDULER: return
    try:
        sched = BackgroundScheduler(timezone=TZ_UTC)
        sched.add_job(lambda: _run_with_pg_lock(1001, production_scan), "interval",
                      seconds=SCAN_INTERVAL_SEC, id="scan_loop", max_instances=1, coalesce=True)
        sched.add_job(lambda: _run_with_pg_lock(1002, backfill_results_for_open_matches, 400), "interval",
                      minutes=BACKFILL_EVERY_MIN, id="backfill_loop", max_instances=1, coalesce=True)
        if DAILY_ACCURACY_DIGEST_ENABLE:
            sched.add_job(lambda: _run_with_pg_lock(1003, daily_accuracy_digest),
                          CronTrigger(hour=DAILY_ACCURACY_HOUR, minute=DAILY_ACCURACY_MINUTE, timezone=TZ_UTC),
                          id="daily_digest", max_instances=1, coalesce=True)
        if MOTD_PREDICT:
            sched.add_job(lambda: _run_with_pg_lock(1004, send_match_of_the_day),
                          CronTrigger(hour=MOTD_HOUR, minute=MOTD_MINUTE, timezone=BERLIN_TZ),
                          id="motd_daily", max_instances=1, coalesce=True)
        if TRAIN_ENABLE:
            sched.add_job(lambda: _run_with_pg_lock(1005, auto_train_job),
                          CronTrigger(hour=TRAIN_HOUR_UTC, minute=TRAIN_MINUTE_UTC, timezone=TZ_UTC),
                          id="nightly_train", max_instances=1, coalesce=True)
        if AUTO_TUNE_ENABLE:
            sched.add_job(lambda: _run_with_pg_lock(1006, auto_tune_thresholds, 14),
                          CronTrigger(hour=4, minute=7, timezone=TZ_UTC),
                          id="auto_tune", max_instances=1, coalesce=True)
        # retry unsent
        sched.add_job(lambda: _run_with_pg_lock(1007, retry_unsent_tips, 30, 200), "interval",
                      minutes=10, id="retry_unsent", max_instances=1, coalesce=True)
        sched.start(); _scheduler_started = True
        send_telegram("🚀 Full AI mode backend started."); 
        logger.info("[SCHED] started (scan=%ss, backfill=%smin, run=%s)", SCAN_INTERVAL_SEC, BACKFILL_EVERY_MIN, RUN_SCHEDULER)
    except Exception as e:
        logger.exception("[SCHED] failed to start: %s", e)

_start_scheduler_once()

# ────────────────────────────────
# Admin / auth
# ────────────────────────────────
def _require_admin() -> None:
    key = request.headers.get("X-API-Key") or request.args.get("key") or ((request.json or {}).get("key") if request.is_json else None)
    if not ADMIN_API_KEY or key != ADMIN_API_KEY: abort(401)

# ────────────────────────────────
# Flask endpoints
# ────────────────────────────────
@app.route("/")
def root(): return jsonify({"ok": True, "name": "live-tips-backend", "mode": "FULL_AI", "scheduler": RUN_SCHEDULER})

@app.route("/health")
def health():
    try:
        with db_conn() as conn:
            c = conn.execute("SELECT COUNT(*) FROM tips").fetchone()[0]
        return jsonify({"ok": True, "db": "ok", "tips_count": int(c)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# Admin ops
@app.route("/init-db", methods=["POST"])
def http_init_db():
    _require_admin(); init_db(); return jsonify({"ok": True})

@app.route("/admin/scan", methods=["POST", "GET"])
def http_scan():
    _require_admin(); saved, live = production_scan(); return jsonify({"ok": True, "saved": saved, "live_seen": live})

@app.route("/admin/backfill-results", methods=["POST", "GET"])
def http_backfill():
    _require_admin(); n = backfill_results_for_open_matches(400); return jsonify({"ok": True, "updated": n})

@app.route("/admin/train", methods=["POST", "GET"])
def http_train():
    _require_admin()
    if not TRAIN_ENABLE: return jsonify({"ok": False, "reason": "training disabled"}), 400
    try:
        out = train_models(); return jsonify({"ok": True, "result": out})
    except Exception as e:
        logger.exception("train_models failed: %s", e); return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/admin/train-notify", methods=["POST","GET"])
def http_train_notify():
    _require_admin(); auto_train_job(); return jsonify({"ok": True})

@app.route("/admin/digest", methods=["POST","GET"])
def http_digest():
    _require_admin(); msg = daily_accuracy_digest(); return jsonify({"ok": True, "sent": bool(msg)})

@app.route("/admin/motd", methods=["POST","GET"])
def http_motd():
    _require_admin(); ok = send_match_of_the_day(); return jsonify({"ok": bool(ok)})

@app.route("/admin/auto-tune", methods=["POST","GET"])
def http_auto_tune():
    _require_admin(); tuned = auto_tune_thresholds(14); return jsonify({"ok": True, "tuned": tuned})

@app.route("/admin/retry-unsent", methods=["POST","GET"])
def http_retry_unsent():
    _require_admin(); n = retry_unsent_tips(30, 200); return jsonify({"ok": True, "resent": n})

# Settings (secured) + cache invalidation
@app.route("/settings/<key>", methods=["GET", "POST"])
def http_settings(key: str):
    _require_admin()
    if request.method == "GET":
        val = get_setting_cached(key); return jsonify({"ok": True, "key": key, "value": val})
    js = request.get_json(silent=True) or {}; val = js.get("value")
    if val is None: abort(400)
    set_setting(key, str(val))
    _SETTINGS_CACHE.invalidate(key); invalidate_model_caches_for_key(key)
    return jsonify({"ok": True})

# Tips read APIs
@app.route("/tips/latest")
def http_latest_tips():
    limit = int(request.args.get("limit", "50"))
    with db_conn() as conn:
        rows = conn.execute(
            "SELECT match_id,league,home,away,market,suggestion,confidence,confidence_raw,score_at_tip,minute,created_ts "
            "FROM tips WHERE suggestion<>'HARVEST' ORDER BY created_ts DESC LIMIT %s",
            (max(1, min(500, limit)),),
        ).fetchall()
    tips = []
    for r in rows:
        tips.append({
            "match_id": int(r[0]), "league": r[1], "home": r[2], "away": r[3], "market": r[4],
            "suggestion": r[5], "confidence": float(r[6]),
            "confidence_raw": (float(r[7]) if r[7] is not None else None),
            "score_at_tip": r[8], "minute": int(r[9]), "created_ts": int(r[10]),
        })
    return jsonify({"ok": True, "tips": tips})

@app.route("/tips/<int:match_id>")
def http_tips_by_match(match_id: int):
    with db_conn() as conn:
        rows = conn.execute(
            "SELECT league,home,away,market,suggestion,confidence,confidence_raw,score_at_tip,minute,created_ts,sent_ok "
            "FROM tips WHERE match_id=%s ORDER BY created_ts ASC",
            (match_id,),
        ).fetchall()
    out = []
    for r in rows:
        out.append({
            "league": r[0], "home": r[1], "away": r[2], "market": r[3], "suggestion": r[4],
            "confidence": float(r[5]),
            "confidence_raw": (float(r[6]) if r[6] is not None else None),
            "score_at_tip": r[7], "minute": int(r[8]),
            "created_ts": int(r[9]), "sent_ok": int(r[10]),
        })
    return jsonify({"ok": True, "match_id": match_id, "tips": out})

# Telegram webhook
@app.route("/telegram/webhook/<secret>", methods=["POST"])
def telegram_webhook(secret: str):
    if (WEBHOOK_SECRET or "") != secret: abort(403)
    update = request.get_json(silent=True) or {}
    try:
        msg = (update.get("message") or {}).get("text") or ""
        if msg.startswith("/start"):
            send_telegram("👋 Live tips bot (FULL AI mode) is online.")
        elif msg.startswith("/digest"):
            daily_accuracy_digest()
        elif msg.startswith("/motd"):
            send_match_of_the_day()
        elif msg.startswith("/scan"):
            parts = msg.split()
            if len(parts) > 1 and ADMIN_API_KEY and parts[1] == ADMIN_API_KEY:
                saved, live = production_scan()
                send_telegram(f"🔁 Scan done. Saved: {saved}, Live seen: {live}")
            else:
                send_telegram("🔒 Admin key required.")
    except Exception as e:
        logger.warning("telegram webhook parse error: %s", e)
    return jsonify({"ok": True})

# ────────────────────────────────
# App startup
# ────────────────────────────────
def _on_boot():
    _init_pool(); init_db(); set_setting("boot_ts", str(int(time.time())))

_on_boot()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    host = os.getenv("HOST", "0.0.0.0")
    app.run(host=host, port=port)
