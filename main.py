"""
Postgres-only Flask backend for live football tips — FULL AI MODE.
Pure ML scoring (no math fallbacks), Platt-calibrated models from DB.
Markets: O/U (1.5 removed), BTTS (Yes/No), 1X2 (Draw blocked at send-time).
Automation: harvest snapshots, nightly train, periodic backfill, daily digest, MOTD, daily auto-tune.
Scan interval: 5 minutes.
"""

import os
import json
import time
import logging
import requests
import psycopg2
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

# ── Logging / app
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
app = Flask(__name__)

# ── ENV
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")
API_KEY            = os.getenv("API_KEY")
ADMIN_API_KEY      = os.getenv("ADMIN_API_KEY")
WEBHOOK_SECRET     = os.getenv("TELEGRAM_WEBHOOK_SECRET")

CONF_THRESHOLD   = float(os.getenv("CONF_THRESHOLD", "65"))
MAX_TIPS_PER_SCAN= int(os.getenv("MAX_TIPS_PER_SCAN", "25"))
DUP_COOLDOWN_MIN = int(os.getenv("DUP_COOLDOWN_MIN", "20"))
TIP_MIN_MINUTE   = int(os.getenv("TIP_MIN_MINUTE", "8"))
SCAN_INTERVAL_SEC= int(os.getenv("SCAN_INTERVAL_SEC", "300"))  # 5 min

HARVEST_MODE     = os.getenv("HARVEST_MODE", "1") not in ("0","false","False","no","NO")
TRAIN_ENABLE     = os.getenv("TRAIN_ENABLE", "1") not in ("0","false","False","no","NO")
TRAIN_HOUR_UTC   = int(os.getenv("TRAIN_HOUR_UTC", "2"))
TRAIN_MINUTE_UTC = int(os.getenv("TRAIN_MINUTE_UTC", "12"))
TRAIN_MIN_MINUTE = int(os.getenv("TRAIN_MIN_MINUTE", "15"))

BACKFILL_EVERY_MIN     = int(os.getenv("BACKFILL_EVERY_MIN", "15"))
BACKFILL_WINDOW_HOURS  = int(os.getenv("BACKFILL_WINDOW_HOURS", "36"))
DAILY_ACCURACY_DIGEST_ENABLE = os.getenv("DAILY_ACCURACY_DIGEST_ENABLE", "1") not in ("0","false","False","no","NO")
DAILY_ACCURACY_HOUR    = int(os.getenv("DAILY_ACCURACY_HOUR", "3"))
DAILY_ACCURACY_MINUTE  = int(os.getenv("DAILY_ACCURACY_MINUTE", "6"))

# Auto-tune thresholds
AUTO_TUNE_ENABLE   = os.getenv("AUTO_TUNE_ENABLE","1") not in ("0","false","False","no","NO")
TARGET_PRECISION   = float(os.getenv("TARGET_PRECISION","0.60"))
TUNE_STEP          = float(os.getenv("TUNE_STEP","2.5"))        # pct points
MIN_THRESH_CLAMP   = float(os.getenv("MIN_THRESH","55"))
MAX_THRESH_CLAMP   = float(os.getenv("MAX_THRESH","85"))
TUNE_LOOKBACK_DAYS = int(os.getenv("TUNE_LOOKBACK_DAYS","1"))

# League blocking
BLOCK_YOUTH_FRIENDLY = os.getenv("BLOCK_YOUTH_FRIENDLY", "1") not in ("0","false","False","no","NO")
BLOCK_PATTERNS_RAW   = (os.getenv("LEAGUE_BLOCK_PATTERNS") or
                        "Friendly,Friendlies,U19,U20,U21,U18,U23,Youth,Reserve,Res.,B Team,B-Team,Development,Academy,Testimonial,All Stars").split(",")
LEAGUE_BLOCK_PATTERNS = [p.strip().lower() for p in BLOCK_PATTERNS_RAW if p.strip()]
try:
    LEAGUE_BLOCK_IDS = [int(x) for x in (os.getenv("LEAGUE_BLOCK_IDS","").split(",")) if x.strip().isdigit()]
except Exception:
    LEAGUE_BLOCK_IDS = []

# Markets / lines
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
OU_LINES = [ln for ln in OU_LINES if abs(ln - 1.5) > 1e-6]  # ensure 1.5 out

TOTAL_MATCH_MINUTES   = int(os.getenv("TOTAL_MATCH_MINUTES", "95"))
PREDICTIONS_PER_MATCH = int(os.getenv("PREDICTIONS_PER_MATCH", "2"))

# Allowed suggestions (no Draw)
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
BASE_URL = "https://v3.football.api-sports.io"
FOOTBALL_API_URL = f"{BASE_URL}/fixtures"
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
HEADERS = {"x-apisports-key": API_KEY, "Accept": "application/json"}
INPLAY_STATUSES = {"1H","HT","2H","ET","BT","P"}

# HTTP
session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], respect_retry_after_header=True)
session.mount("https://", HTTPAdapter(max_retries=retries))

# Caches / TZ
STATS_CACHE: Dict[int, Tuple[float, list]] = {}
EVENTS_CACHE: Dict[int, Tuple[float, list]] = {}
TZ_UTC = ZoneInfo("UTC")
BERLIN_TZ = ZoneInfo("Europe/Berlin")

# Optional: trainer
try:
    from train_models import train_models
except Exception as _imp_err:
    def train_models() -> Dict[str, Any]:
        logger.warning("train_models not available: %s", _imp_err)
        return {"ok": False, "reason": "train_models not available"}

# ── DB adapter
class PgCursor:
    def __init__(self, cur): self.cur = cur
    def fetchone(self): return self.cur.fetchone()
    def fetchall(self): return self.cur.fetchall()

class PgConn:
    def __init__(self, dsn: str): self.dsn = dsn; self.conn=None; self.cur=None
    def __enter__(self):
        if "sslmode=" not in self.dsn:
            self.dsn = self.dsn + ("&" if "?" in self.dsn else "?") + "sslmode=require"
        self.conn = psycopg2.connect(self.dsn)
        self.conn.autocommit = True
        self.cur = self.conn.cursor()
        return self
    def __exit__(self, exc_type, exc, tb):
        try:
            if self.cur: self.cur.close()
        finally:
            if self.conn: self.conn.close()
    def execute(self, sql: str, params: tuple|list=()):
        self.cur.execute(sql, params or ())
        return PgCursor(self.cur)

def db_conn() -> PgConn:
    return PgConn(DATABASE_URL)

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

# ── Init DB
def init_db():
    with db_conn() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS tips (
            match_id BIGINT,
            league_id BIGINT,
            league   TEXT,
            home     TEXT,
            away     TEXT,
            market   TEXT,
            suggestion TEXT,
            confidence DOUBLE PRECISION,
            score_at_tip TEXT,
            minute    INTEGER,
            created_ts BIGINT,
            sent_ok   INTEGER DEFAULT 1,
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

# ── Telegram
def send_telegram(message: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    try:
        res = session.post(f"{TELEGRAM_API_URL}/sendMessage", data=payload, timeout=10)
        return res.ok
    except Exception:
        return False

# ── API helpers
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

# ── League blocking
def _league_blocked(lg: Dict[str, Any]) -> bool:
    if not BLOCK_YOUTH_FRIENDLY:
        return False
    try:
        lid = int((lg or {}).get("id") or 0)
        if lid and lid in set(LEAGUE_BLOCK_IDS):
            return True
        name = ((lg or {}).get("name") or "").lower()
        country = ((lg or {}).get("country") or "").lower()
        ltype = ((lg or {}).get("type") or "").lower()
        hay = f"{name} {country} {ltype}"
        for pat in LEAGUE_BLOCK_PATTERNS:
            if pat and pat in hay:
                return True
    except Exception:
        pass
    return False

# ── Live fixtures
def fetch_live_matches() -> List[Dict[str, Any]]:
    js = _api_get(FOOTBALL_API_URL, {"live": "all"})
    if not isinstance(js, dict):
        return []
    matches = js.get("response", []) or []
    out = []
    for m in matches:
        lg = (m.get("league") or {}) or {}
        if _league_blocked(lg):
            continue
        status = (m.get("fixture", {}) or {}).get("status", {}) or {}
        elapsed = status.get("elapsed")
        short = (status.get("short") or "").upper()
        if elapsed is None or elapsed > 120:
            continue
        if short not in INPLAY_STATUSES:
            continue
        fid = (m.get("fixture", {}) or {}).get("id")
        m["statistics"] = fetch_match_stats(fid) or []
        m["events"] = fetch_match_events(fid) or []
        out.append(m)
    return out

# ── Features
def _num(v) -> float:
    try:
        if isinstance(v, str) and v.endswith("%"):
            return float(v[:-1])
        return float(v or 0)
    except Exception:
        return 0.0

def _pos_pct(v) -> float:
    try:
        return float(str(v).replace("%", "").strip() or 0)
    except Exception:
        return 0.0

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

    xg_h = _num(sh.get("Expected Goals", 0))
    xg_a = _num(sa.get("Expected Goals", 0))
    sot_h = _num(sh.get("Shots on Target", 0))
    sot_a = _num(sa.get("Shots on Target", 0))
    cor_h = _num(sh.get("Corner Kicks", 0))
    cor_a = _num(sa.get("Corner Kicks", 0))
    pos_h = _pos_pct(sh.get("Ball Possession", 0))
    pos_a = _pos_pct(sa.get("Ball Possession", 0))

    red_h = red_a = 0
    for ev in (match.get("events") or []):
        try:
            if (ev.get("type", "").lower() == "card"):
                detail = (ev.get("detail", "") or "").lower()
                if ("red" in detail) or ("second yellow" in detail"):
                    tname = (ev.get("team") or {}).get("name") or ""
                    if tname == home_name: red_h += 1
                    elif tname == away_name: red_a += 1
        except Exception:
            pass

    return {
        "minute": float(minute),
        "goals_h": float(gh),
        "goals_a": float(ga),
        "goals_sum": float(gh + ga),
        "goals_diff": float(gh - ga),
        "xg_h": float(xg_h),
        "xg_a": float(xg_a),
        "xg_sum": float(xg_h + xg_a),
        "xg_diff": float(xg_h - xg_a),
        "sot_h": float(sot_h),
        "sot_a": float(sot_a),
        "sot_sum": float(sot_h + sot_a),
        "cor_h": float(cor_h),
        "cor_a": float(cor_a),
        "cor_sum": float(cor_h + cor_a),
        "pos_h": float(pos_h),
        "pos_a": float(pos_a),
        "pos_diff": float(pos_h - pos_a),
        "red_h": float(red_h),
        "red_a": float(red_a),
        "red_sum": float(red_h + red_a),
    }

def stats_coverage_ok(feat: Dict[str, float], minute: int) -> bool:
    require_stats_minute = int(os.getenv("REQUIRE_STATS_MINUTE", "35"))
    require_fields = int(os.getenv("REQUIRE_DATA_FIELDS", "2"))
    if minute < require_stats_minute:
        return True
    fields = [feat.get("xg_sum", 0.0), feat.get("sot_sum", 0.0), feat.get("cor_sum", 0.0),
              max(feat.get("pos_h", 0.0), feat.get("pos_a", 0.0))]
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

# ── ML model utils
MODEL_KEYS_ORDER = ["model_v2:{name}", "model_latest:{name}", "model:{name}"]

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
    p = max(1e-12, min(1 - 1e-12, float(p)))
    return math.log(p / (1 - p))

def load_model_from_settings(name: str) -> Optional[Dict[str, Any]]:
    for pat in MODEL_KEYS_ORDER:
        key = pat.format(name=name)
        raw = get_setting(key)
        if not raw:
            continue
        try:
            mdl = json.loads(raw)
            mdl.setdefault("intercept", 0.0)
            mdl.setdefault("weights", {})
            cal = mdl.get("calibration") or {}
            if isinstance(cal, dict):
                cal.setdefault("method", "sigmoid")
                cal.setdefault("a", 1.0)
                cal.setdefault("b", 0.0)
                mdl["calibration"] = cal
            return mdl
        except Exception as e:
            logger.warning("[MODEL] failed to parse %s: %s", key, e)
    return None

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
    p = max(1e-12, min(1 - 1e-12, float(p)))
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

# ── Harvest snapshot
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
        "stat": {"xg_h": feat.get("xg_h", 0), "xg_a": feat.get("xg_a", 0),
                 "sot_h": feat.get("sot_h", 0), "sot_a": feat.get("sot_a", 0),
                 "cor_h": feat.get("cor_h", 0), "cor_a": feat.get("cor_a", 0),
                 "pos_h": feat.get("pos_h", 0), "pos_a": feat.get("pos_a", 0),
                 "red_h": feat.get("red_h", 0), "red_a": feat.get("red_a", 0)}
    }
    now = int(time.time())
    with db_conn() as conn:
        conn.execute(
            "INSERT INTO tip_snapshots(match_id, created_ts, payload) VALUES (%s,%s,%s) "
            "ON CONFLICT (match_id, created_ts) DO UPDATE SET payload=EXCLUDED.payload",
            (fid, now, json.dumps(snapshot)[:200000]),
        )
        conn.execute(
            "INSERT INTO tips(match_id,league_id,league,home,away,market,suggestion,confidence,score_at_tip,minute,created_ts,sent_ok) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,0)",
            (fid, league_id, league, home, away, "HARVEST", "HARVEST", 0.0, f"{gh}-{ga}", minute, now),
        )

# ── Tip outcome (grading)
def _parse_ou_line_from_suggestion(s: str) -> Optional[float]:
    try:
        for tok in (s or "").split():
            try:
                return float(tok)
            except Exception:
                pass
    except Exception:
        pass
    return None

def _tip_outcome_for_result(suggestion: str, res: Dict[str, Any]) -> Optional[int]:
    gh = int(res.get("final_goals_h") or 0)
    ga = int(res.get("final_goals_a") or 0)
    total = gh + ga
    btts = int(res.get("btts_yes") or 0)
    s = (suggestion or "").strip()
    if s.startswith("Over") or s.startswith("Under"):
        line = _parse_ou_line_from_suggestion(s)
        if line is None:
            return None
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

# ── Backfill + digest
def _fixture_by_id(match_id: int) -> Optional[Dict[str, Any]]:
    js = _api_get(FOOTBALL_API_URL, {"id": match_id})
    if not isinstance(js, dict):
        return None
    arr = js.get("response") or []
    return arr[0] if arr else None

def _is_final_status(short: str) -> bool:
    short = (short or "").upper()
    return short in {"FT", "AET", "PEN"}

def backfill_results_for_open_matches(max_rows: int = 200) -> int:
    now_ts = int(time.time())
    cutoff = now_ts - 2 * 24 * 3600
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
            if not fx:
                continue
            status = ((fx.get("fixture") or {}).get("status") or {}).get("short", "")
            if not _is_final_status(status):
                continue
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
    if updated:
        logger.info("[RESULTS] backfilled %d results", updated)
    return updated

def daily_accuracy_digest() -> Optional[str]:
    if not DAILY_ACCURACY_DIGEST_ENABLE:
        return None
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
    for (_mid, market, sugg, _conf, _cts, gh, ga, btts) in rows:
        res = {"final_goals_h": gh, "final_goals_a": ga, "btts_yes": btts}
        out = _tip_outcome_for_result(sugg, res)
        if out is None:
            continue
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
    send_telegram(msg)
    return msg

# ── MOTD (future match today, with blocking)
MOTD_PREDICT   = os.getenv("MOTD_PREDICT", "1") not in ("0","false","False","no","NO")
MOTD_HOUR      = int(os.getenv("MOTD_HOUR", "19"))
MOTD_MINUTE    = int(os.getenv("MOTD_MINUTE", "15"))
MOTD_CONF_MIN  = float(os.getenv("MOTD_CONF_MIN", "70"))
try:
    MOTD_LEAGUE_IDS = [int(x) for x in (os.getenv("MOTD_LEAGUE_IDS","").split(",")) if x.strip().isdigit()]
except Exception:
    MOTD_LEAGUE_IDS = []

def _kickoff_berlin(utc_iso: str|None) -> str:
    try:
        if not utc_iso:
            return "TBD"
        dt_utc = datetime.fromisoformat(utc_iso.replace("Z", "+00:00"))
        return dt_utc.astimezone(BERLIN_TZ).strftime("%H:%M")
    except Exception:
        return "TBD"

def _api_fixtures_for_date_utc(date_utc: datetime) -> list[dict]:
    js = _api_get(FOOTBALL_API_URL, {"date": date_utc.strftime("%Y-%m-%d")})
    if not isinstance(js, dict):
        return []
    out = []
    for r in js.get("response", []) or []:
        st = (((r.get("fixture") or {}).get("status") or {}).get("short") or "").upper()
        if st == "NS":
            out.append(r)
    return out

def _prematch_features() -> Dict[str, float]:
    return {
        "minute": 0.0,
        "goals_h": 0.0, "goals_a": 0.0, "goals_sum": 0.0, "goals_diff": 0.0,
        "xg_h": 0.0, "xg_a": 0.0, "xg_sum": 0.0, "xg_diff": 0.0,
        "sot_h": 0.0, "sot_a": 0.0, "sot_sum": 0.0,
        "cor_h": 0.0, "cor_a": 0.0, "cor_sum": 0.0,
        "pos_h": 0.0, "pos_a": 0.0, "pos_diff": 0.0,
        "red_h": 0.0, "red_a": 0.0, "red_sum": 0.0,
    }

def _format_motd_message(home, away, league, kickoff_txt, suggestion, market_name, prob_pct) -> str:
    return (
        "🏅 <b>Match of the Day</b>\n"
        f"<b>Match:</b> {escape(home)} vs {escape(away)}\n"
        f"🏆 <b>League:</b> {escape(league)}\n"
        f"⏰ <b>Kickoff (Berlin):</b> {kickoff_txt}\n"
        f"<b>Tip:</b> {escape(suggestion)}\n"
        f"📈 <b>Confidence:</b> {prob_pct:.1f}%"
    )

def send_match_of_the_day() -> bool:
    if not MOTD_PREDICT:
        return False
    today_local = datetime.now(BERLIN_TZ).date()
    start_local = datetime.combine(today_local, datetime.min.time(), tzinfo=BERLIN_TZ)
    end_local   = start_local + timedelta(days=1)
    dates_utc = {start_local.astimezone(TZ_UTC).date(), (end_local - timedelta(seconds=1)).astimezone(TZ_UTC).date()}
    fixtures: list[dict] = []
    for d in sorted(dates_utc):
        fixtures.extend(_api_fixtures_for_date_utc(datetime(d.year, d.month, d.day, tzinfo=TZ_UTC)))
    # Block low-quality comps & optional league filter
    fixtures = [f for f in fixtures if not _league_blocked(f.get("league") or {})]
    if MOTD_LEAGUE_IDS:
        fixtures = [f for f in fixtures if int(((f.get("league") or {}).get("id") or 0)) in MOTD_LEAGUE_IDS]
    if not fixtures:
        return send_telegram("🏅 Match of the Day: no eligible fixtures today.")
    feat0 = _prematch_features()
    best = None  # (prob_pct, market_name, suggestion, home, away, league, kickoff_txt)
    for fx in fixtures:
        fixture = fx.get("fixture") or {}
        lg      = fx.get("league")  or {}
        teams   = fx.get("teams")   or {}
        home = (teams.get("home") or {}).get("name", "")
        away = (teams.get("away") or {}).get("name", "")
        league = f"{lg.get('country','')} - {lg.get('name','')}".strip(" -")
        kickoff_txt = _kickoff_berlin((fixture.get("date") or ""))
        candidates: list[tuple[str, str, float]] = []
        # O/U
        for line in OU_LINES:
            mdl_line = _load_ou_model_for_line(line)
            if not mdl_line: continue
            p_over = _score_prob(feat0, mdl_line)
            market_name = f"Over/Under {_fmt_line(line)}"
            thr = _get_market_threshold(market_name)
            if p_over * 100.0 >= thr:
                candidates.append((market_name, f"Over {_fmt_line(line)} Goals", p_over))
            p_under = 1.0 - p_over
            if p_under * 100.0 >= thr:
                candidates.append((market_name, f"Under {_fmt_line(line)} Goals", p_under))
        # BTTS
        mdl_btts = load_model_from_settings("BTTS_YES")
        if mdl_btts:
            p_btts = _score_prob(feat0, mdl_btts)
            thr_b = _get_market_threshold("BTTS")
            if p_btts * 100.0 >= thr_b:
                candidates.append(("BTTS", "BTTS: Yes", p_btts))
            p_btts_no = 1.0 - p_btts
            if p_btts_no * 100.0 >= thr_b:
                candidates.append(("BTTS", "BTTS: No", p_btts_no))
        # 1X2
        mh, md, ma = _load_wld_models()
        if mh and md and ma:
            ph = _score_prob(feat0, mh); pd = _score_prob(feat0, md); pa = _score_prob(feat0, ma)
            s = max(1e-9, ph + pd + pa); ph, pd, pa = ph/s, pd/s, pa/s
            thr_1x2 = _get_market_threshold("1X2")
            if ph * 100.0 >= thr_1x2: candidates.append(("1X2", "Home Win", ph))
            if pa * 100.0 >= thr_1x2: candidates.append(("1X2", "Away Win", pa))
        if not candidates:
            continue
        market_name, suggestion, prob = max(candidates, key=lambda x: x[2])
        prob_pct = prob * 100.0
        if prob_pct < MOTD_CONF_MIN:
            continue
        item = (prob_pct, market_name, suggestion, home, away, league, kickoff_txt)
        if (best is None) or (prob_pct > best[0]):
            best = item
    if not best:
        return send_telegram("🏅 Match of the Day: no pick met the confidence/thresholds today.")
    prob_pct, market_name, suggestion, home, away, league, kickoff_txt = best
    return send_telegram(_format_motd_message(home, away, league, kickoff_txt, suggestion, market_name, prob_pct))

# ── Training job (with Telegram)
def auto_train_job() -> None:
    if not TRAIN_ENABLE:
        send_telegram("🤖 Training skipped: TRAIN_ENABLE=0")
        return
    send_telegram("🤖 Training started.")
    try:
        res = train_models() or {}
        ok   = bool(res.get("ok"))
        if not ok:
            reason = res.get("reason") or res.get("error") or "unknown reason"
            send_telegram(f"⚠️ Training finished: <b>SKIPPED</b>\nReason: {escape(str(reason))}")
            return
        trained = [k for k, v in (res.get("trained") or {}).items() if v]
        thr     = (res.get("thresholds") or {})
        mets    = (res.get("metrics") or {})
        lines = ["🤖 <b>Model training OK</b>"]
        if trained: lines.append("• Trained: " + ", ".join(sorted(trained)))
        if thr:     lines.append("• Thresholds: " + "  |  ".join([f"{escape(k)}: {float(v):.1f}%" for k, v in thr.items()]))
        def _m(name):
            m = mets.get(name) or {}
            if m:
                return f"{name}: acc {m.get('acc',0):.2f}, brier {m.get('brier',0):.3f}, logloss {m.get('logloss',0):.3f}"
            return None
        metric_lines = list(filter(None, [
            _m("BTTS_YES"),
            _m("OU_2.5"), _m("OU_3.5"),
            _m("WLD_HOME"), _m("WLD_AWAY"),
        ]))
        if metric_lines:
            lines.append("• Metrics:\n  " + "\n  ".join(metric_lines))
        send_telegram("\n".join(lines))
    except Exception as e:
        logger.exception("[TRAIN] job failed: %s", e)
        send_telegram(f"❌ Training <b>FAILED</b>\n{escape(str(e))}")

# ── Auto-tune thresholds (daily)
def _market_threshold_key(market: str) -> str:
    return f"conf_threshold:{market}"

def _get_or_default_threshold(market: str) -> float:
    try:
        val = get_setting(_market_threshold_key(market))
        return float(val) if val is not None else float(CONF_THRESHOLD)
    except Exception:
        return float(CONF_THRESHOLD)

def _set_threshold(market: str, val_pct: float) -> None:
    clamped = max(MIN_THRESH_CLAMP, min(MAX_THRESH_CLAMP, float(val_pct)))
    set_setting(_market_threshold_key(market), f"{clamped:.2f}")

def _graded_precision_for_market(market: str, start_ts: int, end_ts: int) -> tuple[int,int,float]:
    with db_conn() as conn:
        rows = conn.execute("""
            SELECT t.suggestion, r.final_goals_h, r.final_goals_a, r.btts_yes
            FROM tips t
            LEFT JOIN match_results r ON r.match_id = t.match_id
            WHERE t.created_ts BETWEEN %s AND %s
              AND t.market = %s
              AND t.suggestion <> 'HARVEST'
              AND t.sent_ok = 1
        """, (start_ts, end_ts, market)).fetchall()
    graded = wins = 0
    for (sugg, gh, ga, btts) in rows:
        out = _tip_outcome_for_result(sugg, {"final_goals_h": gh, "final_goals_a": ga, "btts_yes": btts})
        if out is None:
            continue
        graded += 1; wins += 1 if out == 1 else 0
    prec = (wins/graded) if graded else 0.0
    return graded, wins, prec

def _yesterday_window_utc() -> tuple[int,int]:
    now = datetime.now(TZ_UTC)
    y_start = (now - timedelta(days=TUNE_LOOKBACK_DAYS)).replace(hour=0, minute=0, second=0, microsecond=0)
    y_end = y_start + timedelta(days=TUNE_LOOKBACK_DAYS)
    return int(y_start.timestamp()), int(y_end.timestamp())

def auto_tune_thresholds() -> dict:
    if not AUTO_TUNE_ENABLE:
        return {"ok": False, "reason": "auto-tune disabled"}
    start_ts, end_ts = _yesterday_window_utc()
    markets = ["BTTS", "1X2"] + [f"Over/Under { _fmt_line(ln) }" for ln in OU_LINES]
    changes = {}
    for mk in markets:
        graded, wins, prec = _graded_precision_for_market(mk, start_ts, end_ts)
        if graded < 25:
            continue
        cur = _get_or_default_threshold(mk)
        new_thr = cur
        if prec < TARGET_PRECISION - 0.02:
            new_thr = min(MAX_THRESH_CLAMP, cur + TUNE_STEP)
        elif prec > TARGET_PRECISION + 0.02:
            new_thr = max(MIN_THRESH_CLAMP, cur - TUNE_STEP)
        if abs(new_thr - cur) >= 1e-9:
            _set_threshold(mk, new_thr)
            changes[mk] = {"graded": graded, "wins": wins, "prec": round(prec,3), "old": cur, "new": new_thr}
    if changes:
        send_telegram("🛠️ Auto-tune updated thresholds:\n" + "\n".join(
            [f"• {k}: {v['old']:.1f}% → {v['new']:.1f}% (prec {v['prec']*100:.1f}%, n={v['graded']})" for k,v in changes.items()]
        ))
    else:
        send_telegram("🛠️ Auto-tune ran: no changes (insufficient data or already on target).")
    return {"ok": True, "changes": changes}

# ── Tip formatting/sending
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
    return (
        "⚽️ <b>New Tip!</b>\n"
        f"<b>Match:</b> {escape(home)} vs {escape(away)}\n"
        f"🕒 <b>Minute:</b> {minute}'  |  <b>Score:</b> {escape(score_txt)}\n"
        f"<b>Tip:</b> {escape(suggestion)}\n"
        f"📈 <b>Confidence:</b> {prob_pct:.1f}%\n"
        f"🏆 <b>League:</b> {escape(league)}"
        f"{stat_line}"
    )

def _send_tip(home, away, league, minute, score_txt, suggestion, prob_pct, feat) -> bool:
    return send_telegram(_format_tip_message(home, away, league, minute, score_txt, suggestion, prob_pct, feat))

# ── Production scan (ML-only)
def _get_market_threshold(market: str) -> float:
    try:
        val = get_setting(f"conf_threshold:{market}")
        return float(val) if val is not None else float(CONF_THRESHOLD)
    except Exception:
        return float(CONF_THRESHOLD)

def production_scan() -> Tuple[int, int]:
    matches = fetch_live_matches()
    live_seen = len(matches)
    if live_seen == 0:
        logger.info("[PROD] no live matches")
        return 0, 0
    saved = 0
    now_ts = int(time.time())
    with db_conn() as conn:
        for m in matches:
            try:
                fid = int((m.get("fixture", {}) or {}).get("id") or 0)
                if not fid: continue
                # cooldown
                if DUP_COOLDOWN_MIN > 0:
                    cutoff = now_ts - (DUP_COOLDOWN_MIN * 60)
                    dup = conn.execute(
                        "SELECT 1 FROM tips WHERE match_id=%s AND created_ts>=%s LIMIT 1",
                        (fid, cutoff),
                    ).fetchone()
                    if dup:
                        continue
                feat = extract_features(m)
                minute = int(feat.get("minute", 0))
                if not stats_coverage_ok(feat, minute): continue
                if minute < TIP_MIN_MINUTE: continue
                # harvest
                if HARVEST_MODE and minute >= TRAIN_MIN_MINUTE and minute % 3 == 0:
                    try: save_snapshot_from_match(m, feat)
                    except Exception: pass
                league_id, league = _league_name(m)
                home, away = _teams(m)
                score_txt = _pretty_score(m)
                candidates: List[Tuple[str, str, float]] = []
                # O/U
                for line in OU_LINES:
                    mdl_line = _load_ou_model_for_line(line)
                    if not mdl_line: continue
                    p_over = _score_prob(feat, mdl_line)
                    market_name = f"Over/Under {_fmt_line(line)}"
                    thr = _get_market_threshold(market_name)
                    if p_over * 100.0 >= thr:
                        candidates.append((market_name, f"Over {_fmt_line(line)} Goals", p_over))
                    p_under = 1.0 - p_over
                    if p_under * 100.0 >= thr:
                        candidates.append((market_name, f"Under {_fmt_line(line)} Goals", p_under))
                # BTTS
                mdl_btts = load_model_from_settings("BTTS_YES")
                if mdl_btts:
                    p_btts = _score_prob(feat, mdl_btts)
                    thr_b = _get_market_threshold("BTTS")
                    if p_btts * 100.0 >= thr_b:
                        candidates.append(("BTTS", "BTTS: Yes", p_btts))
                    p_btts_no = 1.0 - p_btts
                    if p_btts_no * 100.0 >= thr_b:
                        candidates.append(("BTTS", "BTTS: No", p_btts_no))
                # 1X2 (no Draw)
                mh, md, ma = _load_wld_models()
                if mh and md and ma:
                    ph = _score_prob(feat, mh)
                    pd = _score_prob(feat, md)
                    pa = _score_prob(feat, ma)
                    s = max(1e-9, ph + pd + pa)
                    ph, pd, pa = ph/s, pd/s, pa/s
                    thr_1x2 = _get_market_threshold("1X2")
                    if ph * 100.0 >= thr_1x2: candidates.append(("1X2", "Home Win", ph))
                    if pa * 100.0 >= thr_1x2: candidates.append(("1X2", "Away Win", pa))
                # send top few
                candidates.sort(key=lambda x: x[2], reverse=True)
                per_match = 0
                base_now = int(time.time())
                for idx, (market_txt, suggestion, prob) in enumerate(candidates):
                    if suggestion not in ALLOWED_SUGGESTIONS: continue
                    if per_match >= max(1, PREDICTIONS_PER_MATCH): break
                    created_ts = base_now + idx
                    prob_pct = round(prob * 100.0, 1)
                    with db_conn() as conn2:
                        conn2.execute(
                            "INSERT INTO tips(match_id,league_id,league,home,away,market,suggestion,confidence,score_at_tip,minute,created_ts,sent_ok) "
                            "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,0)",
                            (fid, league_id, league, home, away, market_txt, suggestion, float(prob_pct), score_txt, minute, created_ts),
                        )
                        sent_ok = _send_tip(home, away, league, minute, score_txt, suggestion, float(prob_pct), feat)
                        if sent_ok:
                            conn2.execute("UPDATE tips SET sent_ok=1 WHERE match_id=%s AND created_ts=%s", (fid, created_ts))
                    saved += 1; per_match += 1
                    if MAX_TIPS_PER_SCAN and saved >= MAX_TIPS_PER_SCAN: break
                if MAX_TIPS_PER_SCAN and saved >= MAX_TIPS_PER_SCAN: break
            except Exception as e:
                logger.exception("[PROD] failure on match: %s", e)
                continue
    logger.info(f"[PROD] saved={saved} live_seen={live_seen}")
    return saved, live_seen

# ── Scheduler
_scheduler_started = False
def _start_scheduler_once() -> None:
    global _scheduler_started
    if _scheduler_started:
        return
    try:
        sched = BackgroundScheduler(timezone=TZ_UTC)
        sched.add_job(production_scan, "interval", seconds=SCAN_INTERVAL_SEC, id="scan_loop", max_instances=1, coalesce=True)
        sched.add_job(lambda: backfill_results_for_open_matches(400), "interval", minutes=BACKFILL_EVERY_MIN,
                      id="backfill_loop", max_instances=1, coalesce=True)
        if DAILY_ACCURACY_DIGEST_ENABLE:
            sched.add_job(daily_accuracy_digest,
                          CronTrigger(hour=DAILY_ACCURACY_HOUR, minute=DAILY_ACCURACY_MINUTE, timezone=TZ_UTC),
                          id="daily_digest", max_instances=1, coalesce=True)
        if MOTD_PREDICT:
            sched.add_job(
                send_match_of_the_day,
                CronTrigger(hour=MOTD_HOUR, minute=MOTD_MINUTE, timezone=BERLIN_TZ),
                id="motd_daily",
                max_instances=1,
                coalesce=True
            )
        if AUTO_TUNE_ENABLE:
            sched.add_job(
                auto_tune_thresholds,
                CronTrigger(hour=4, minute=12, timezone=BERLIN_TZ),
                id="auto_tune",
                max_instances=1,
                coalesce=True
            )
        if TRAIN_ENABLE:
            sched.add_job(
                auto_train_job,
                CronTrigger(hour=TRAIN_HOUR_UTC, minute=TRAIN_MINUTE_UTC, timezone=TZ_UTC),
                id="nightly_train", max_instances=1, coalesce=True
            )
        sched.start()
        _scheduler_started = True
        send_telegram("🚀 Full AI mode backend started.")
        logger.info("[SCHED] started (scan=%ss backfill=%smin train=%02d:%02dZ digest=%02d:%02dZ)",
                    SCAN_INTERVAL_SEC, BACKFILL_EVERY_MIN, TRAIN_HOUR_UTC, TRAIN_MINUTE_UTC,
                    DAILY_ACCURACY_HOUR, DAILY_ACCURACY_MINUTE)
    except Exception as e:
        logger.exception("[SCHED] failed to start: %s", e)

_start_scheduler_once()

# ── Admin / auth
def _require_admin() -> None:
    key = request.headers.get("X-API-Key") or request.args.get("key") or (request.json or {}).get("key") if request.is_json else None
    if not ADMIN_API_KEY or key != ADMIN_API_KEY:
        abort(401)

# ── Routes
@app.route("/")
def root():
    return jsonify({"ok": True, "name": "live-tips-backend", "mode": "FULL_AI"})

@app.route("/health")
def health():
    try:
        with db_conn() as conn:
            c = conn.execute("SELECT COUNT(*) FROM tips").fetchone()[0]
        return jsonify({"ok": True, "db": "ok", "tips_count": int(c)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/init-db", methods=["POST"])
def http_init_db():
    _require_admin()
    init_db()
    return jsonify({"ok": True})

@app.route("/admin/scan", methods=["POST", "GET"])
def http_scan():
    _require_admin()
    saved, live = production_scan()
    return jsonify({"ok": True, "saved": saved, "live_seen": live})

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
        out = train_models()
        return jsonify({"ok": True, "result": out})
    except Exception as e:
        logger.exception("train_models failed: %s", e)
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/admin/train-notify", methods=["POST","GET"])
def http_train_notify():
    _require_admin()
    auto_train_job()
    return jsonify({"ok": True})

@app.route("/admin/digest", methods=["POST","GET"])
def http_digest():
    _require_admin()
    msg = daily_accuracy_digest()
    return jsonify({"ok": True, "sent": bool(msg)})

@app.route("/admin/motd", methods=["POST","GET"])
def http_motd():
    _require_admin()
    ok = send_match_of_the_day()
    return jsonify({"ok": bool(ok)})

@app.route("/admin/auto-tune", methods=["POST","GET"])
def http_auto_tune():
    _require_admin()
    res = auto_tune_thresholds()
    return jsonify({"ok": True, "result": res})

@app.route("/settings/<key>", methods=["GET", "POST"])
def http_settings(key: str):
    if request.method == "GET":
        val = get_setting(key)
        return jsonify({"ok": True, "key": key, "value": val})
    _require_admin()
    js = request.get_json(silent=True) or {}
    val = js.get("value")
    if val is None:
        abort(400)
    set_setting(key, str(val))
    return jsonify({"ok": True})

@app.route("/tips/latest")
def http_latest_tips():
    limit = int(request.args.get("limit", "50"))
    with db_conn() as conn:
        rows = conn.execute(
            "SELECT match_id,league,home,away,market,suggestion,confidence,score_at_tip,minute,created_ts "
            "FROM tips WHERE suggestion<>'HARVEST' ORDER BY created_ts DESC LIMIT %s",
            (max(1, min(500, limit)),),
        ).fetchall()
    tips = []
    for r in rows:
        tips.append({
            "match_id": int(r[0]), "league": r[1], "home": r[2], "away": r[3], "market": r[4],
            "suggestion": r[5], "confidence": float(r[6]), "score_at_tip": r[7],
            "minute": int(r[8]), "created_ts": int(r[9]),
        })
    return jsonify({"ok": True, "tips": tips})

@app.route("/tips/<int:match_id>")
def http_tips_by_match(match_id: int):
    with db_conn() as conn:
        rows = conn.execute(
            "SELECT league,home,away,market,suggestion,confidence,score_at_tip,minute,created_ts,sent_ok "
            "FROM tips WHERE match_id=%s ORDER BY created_ts ASC",
            (match_id,),
        ).fetchall()
    out = []
    for r in rows:
        out.append({
            "league": r[0], "home": r[1], "away": r[2], "market": r[3], "suggestion": r[4],
            "confidence": float(r[5]), "score_at_tip": r[6], "minute": int(r[7]),
            "created_ts": int(r[8]), "sent_ok": int(r[9]),
        })
    return jsonify({"ok": True, "match_id": match_id, "tips": out})

@app.route("/telegram/webhook/<secret>", methods=["POST"])
def telegram_webhook(secret: str):
    if (WEBHOOK_SECRET or "") != secret:
        abort(403)
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

# ── Boot
def _on_boot():
    init_db()
    set_setting("boot_ts", str(int(time.time())))

_on_boot()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    host = os.getenv("HOST", "0.0.0.0")
    app.run(host=host, port=port)
