"""
Postgres-only Flask backend for live football tips with blended math+ML.
Softened probabilities (no 0%/100%), Platt-calibrated ML, multi-line O/U, BTTS, 1X2.
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

from analytics import ou_over_probability, btts_yes_probability, wld_probabilities

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
app = Flask(__name__)

# ── Env
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
API_KEY = os.getenv("API_KEY")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")
PUBLIC_BASE_URL = (os.getenv("PUBLIC_BASE_URL") or "").rstrip("/")
WEBHOOK_SECRET = os.getenv("TELEGRAM_WEBHOOK_SECRET")
HEARTBEAT_ENABLE = os.getenv("HEARTBEAT_ENABLE", "1") not in ("0","false","False","no","NO")

HARVEST_MODE = os.getenv("HARVEST_MODE", "1") not in ("0","false","False","no","NO")
CONF_THRESHOLD = int(os.getenv("CONF_THRESHOLD", "65"))
MAX_TIPS_PER_SCAN = int(os.getenv("MAX_TIPS_PER_SCAN", "25"))
DUP_COOLDOWN_MIN = int(os.getenv("DUP_COOLDOWN_MIN", "20"))

O25_LATE_MINUTE = int(os.getenv("O25_LATE_MINUTE", "88"))
O25_LATE_MIN_GOALS = int(os.getenv("O25_LATE_MIN_GOALS", "2"))
BTTS_LATE_MINUTE = int(os.getenv("BTTS_LATE_MINUTE", "88"))
UNDER_SUPPRESS_AFTER_MIN = int(os.getenv("UNDER_SUPPRESS_AFTER_MIN", "82"))

ONLY_MODEL_MODE = os.getenv("ONLY_MODEL_MODE", "0") not in ("0","false","False","no","NO")
REQUIRE_STATS_MINUTE = int(os.getenv("REQUIRE_STATS_MINUTE", "35"))
REQUIRE_DATA_FIELDS = int(os.getenv("REQUIRE_DATA_FIELDS", "2"))

LEAGUE_PRIORITY_IDS = [int(x) for x in (os.getenv("MOTD_LEAGUE_IDS", "39,140,135,78,61,2").split(",")) if x.strip().isdigit()]
MOTD_PREDICT = os.getenv("MOTD_PREDICT", "1") not in ("0","false","False","no","NO")
MOTD_MIN_SAMPLES = int(os.getenv("MOTD_MIN_SAMPLES", "30"))
MOTD_CONF_MIN = int(os.getenv("MOTD_CONF_MIN", "65"))

DAILY_ACCURACY_DIGEST_ENABLE = os.getenv("DAILY_ACCURACY_DIGEST_ENABLE", "1") not in ("0","false","False","no","NO")
DAILY_ACCURACY_HOUR = int(os.getenv("DAILY_ACCURACY_HOUR", "3"))
DAILY_ACCURACY_MINUTE = int(os.getenv("DAILY_ACCURACY_MINUTE", "6"))

TIP_MIN_MINUTE = int(os.getenv("TIP_MIN_MINUTE", "8")) 
EARLY_CONF_CAP_ENABLE = os.getenv("EARLY_CONF_CAP_ENABLE", "1") not in ("0","false","False","no","NO")
EARLY_CONF_CAP_MINUTE = int(os.getenv("EARLY_CONF_CAP_MINUTE", "15"))  
EARLY_CONF_CAP_LO = float(os.getenv("EARLY_CONF_CAP_LO", "70.0"))

# ── Autonomous mode / policy tuning
AUTO_TUNE_ENABLE = os.getenv("AUTO_TUNE_ENABLE", "1") not in ("0","false","False","no","NO")
TARGET_PRECISION = float(os.getenv("TARGET_PRECISION", "0.60"))   # aim precision per market
TUNE_STEP = float(os.getenv("TUNE_STEP", "2.5"))                  # pct points to nudge thresholds
MIN_THRESH = float(os.getenv("MIN_THRESH", "55"))
MAX_THRESH = float(os.getenv("MAX_THRESH", "85"))
MARKET_SCORE_ENABLE = os.getenv("MARKET_SCORE_ENABLE", "0") not in ("0","false","False","no","NO")
MARKET_SCORE_DAYS = int(os.getenv("MARKET_SCORE_DAYS", "7"))

def _parse_lines(env_val: str, default: List[float]) -> List[float]:
    out: List[float] = []
    for tok in (env_val or "").split(","):
        tok = tok.strip()
        try:
            if tok:
                out.append(float(tok))
        except Exception:
            continue
    return out or default

OU_LINES: List[float] = _parse_lines(os.getenv("OU_LINES", "0.5,1.5,2.5,3.5"), [0.5, 1.5, 2.5, 3.5])
TOTAL_MATCH_MINUTES = int(os.getenv("TOTAL_MATCH_MINUTES", "95"))
PREDICTIONS_PER_MATCH = int(os.getenv("PREDICTIONS_PER_MATCH", "2"))

BLEND_ML_WEIGHT = float(os.getenv("BLEND_ML_WEIGHT", "0.7"))
BLEND_ML_WEIGHT = 0.0 if BLEND_ML_WEIGHT < 0 else (1.0 if BLEND_ML_WEIGHT > 1.0 else BLEND_ML_WEIGHT)
BLEND_MATH_WEIGHT = 1.0 - BLEND_ML_WEIGHT
# If ONLY_MODEL_MODE is enabled, let ML fully drive probabilities (math is disabled)
if ONLY_MODEL_MODE:
    BLEND_ML_WEIGHT = 1.0
    BLEND_MATH_WEIGHT = 0.0

# Final-prob softening & display caps
GLOBAL_MIN_PROB = float(os.getenv("GLOBAL_MIN_PROB", "0.02"))
GLOBAL_SOFT_ALPHA = float(os.getenv("GLOBAL_SOFT_ALPHA", "0.85"))
DISPLAY_MIN_PCT = float(os.getenv("DISPLAY_MIN_PCT", "1.0"))
DISPLAY_MAX_PCT = float(os.getenv("DISPLAY_MAX_PCT", "99.0"))
ALLOWED_SUGGESTIONS = {"BTTS: Yes", "BTTS: No", "Home Win", "Draw", "Away Win"}
for ln in OU_LINES:
    s = f"{ln}".rstrip("0").rstrip(".")
    ALLOWED_SUGGESTIONS.add(f"Over {s} Goals")
    ALLOWED_SUGGESTIONS.add(f"Under {s} Goals")

TRAIN_ENABLE = os.getenv("TRAIN_ENABLE", "1") not in ("0","false","False","no","NO")
TRAIN_MIN_MINUTE = int(os.getenv("TRAIN_MIN_MINUTE", "15"))
TRAIN_TEST_SIZE = float(os.getenv("TRAIN_TEST_SIZE", "0.25"))

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise SystemExit("DATABASE_URL is required (Postgres).")

# External APIs
BASE_URL = "https://v3.football.api-sports.io"
FOOTBALL_API_URL = f"{BASE_URL}/fixtures"
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
HEADERS = {"x-apisports-key": API_KEY, "Accept": "application/json"}
INPLAY_STATUSES = ["1H","HT","2H","ET","BT","P"]

# HTTP
session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], respect_retry_after_header=True)
session.mount("https://", HTTPAdapter(max_retries=retries))

# Caches
STATS_CACHE: Dict[int, Tuple[float, list]] = {}
EVENTS_CACHE: Dict[int, Tuple[float, list]] = {}
CAL_CACHE: Dict[str, Any] = {"ts": 0, "bins": []}

# Optional import
try:
    from train_models import train_models
except Exception as _imp_err:
    def train_models() -> Dict[str, Any]:
        logger.warning("train_models not available: %s", _imp_err)
        return {"ok": False, "reason": "train_models not available"}

# ── DB adapter (Postgres-only)
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

# ── Settings
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
def send_telegram(message: str, inline_keyboard: Optional[list] = None) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    if inline_keyboard:
        payload["reply_markup"] = json.dumps({"inline_keyboard": inline_keyboard})
    try:
        res = session.post(f"{TELEGRAM_API_URL}/sendMessage", data=payload, timeout=10)
        return res.ok
    except Exception:
        return False

# ── External API helpers
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

def fetch_live_matches() -> List[Dict[str, Any]]:
    js = _api_get(FOOTBALL_API_URL, {"live": "all"})
    if not isinstance(js, dict):
        return []
    matches = js.get("response", []) or []
    out = []
    allowed = set(INPLAY_STATUSES)
    for m in matches:
        status = (m.get("fixture", {}) or {}).get("status", {}) or {}
        elapsed = status.get("elapsed")
        short = (status.get("short") or "").upper()
        if elapsed is None or elapsed > 120:
            continue
        if short not in allowed:
            continue
        fid = (m.get("fixture", {}) or {}).get("id")
        m["statistics"] = fetch_match_stats(fid) or []
        m["events"] = fetch_match_events(fid) or []
        out.append(m)
    return out

# ── Feature extraction
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
                if ("red" in detail) or ("second yellow" in detail):
                    tname = (ev.get("team") or {}).get("name") or ""
                    if tname == home_name:
                        red_h += 1
                    elif tname == away_name:
                        red_a += 1
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
    if minute < REQUIRE_STATS_MINUTE:
        return True
    fields = [
        feat.get("xg_sum", 0.0),
        feat.get("sot_sum", 0.0),
        feat.get("cor_sum", 0.0),
        max(feat.get("pos_h", 0.0), feat.get("pos_a", 0.0)),
    ]
    nonzero = sum(1 for v in fields if (v or 0) > 0)
    return nonzero >= max(0, REQUIRE_DATA_FIELDS)

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

# ── Models & calibration
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
    logit = math.log(p / (1.0 - p))
    return _sigmoid(a * logit + b)

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

def _soften_final_prob(p: float) -> float:
    p = 0.5 + GLOBAL_SOFT_ALPHA * (p - 0.5)
    return max(GLOBAL_MIN_PROB, min(1.0 - GLOBAL_MIN_PROB, p))

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
    msg = (
        "⚽️ <b>New Tip!</b>\n"
        f"<b>Match:</b> {escape(home)} vs {escape(away)}\n"
        f"🕒 <b>Minute:</b> {minute}'  |  <b>Score:</b> {escape(score_txt)}\n"
        f"<b>Tip:</b> {escape(suggestion)}\n"
        f"📈 <b>Confidence:</b> {prob_pct:.1f}%\n"
        f"🏆 <b>League:</b> {escape(league)}"
        f"{stat_line}"
    )
    return msg

def _send_tip(home, away, league, minute, score_txt, suggestion, prob_pct, feat) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    return send_telegram(_format_tip_message(home, away, league, minute, score_txt, suggestion, prob_pct, feat))

# ── Autonomous policy: per-market thresholds + auto-train/promote + optional market scoring
def _get_market_threshold_key(market: str) -> str:
    return f"conf_threshold:{market}"

def _get_market_threshold_default(market: str) -> float:
    return float(CONF_THRESHOLD)

def _get_market_threshold(market: str) -> float:
    try:
        val = get_setting(_get_market_threshold_key(market))
        return float(val) if val is not None else _get_market_threshold_default(market)
    except Exception:
        return _get_market_threshold_default(market)

def _set_market_threshold(market: str, val: float) -> None:
    try:
        set_setting(_get_market_threshold_key(market), f"{float(val):.2f}")
    except Exception:
        pass

def _recent_precision_for_market(market: str, days: int = 3) -> Tuple[int,int,float]:
    end = int(time.time())
    start = end - days*24*3600
    with db_conn() as conn:
        rows = conn.execute("""
          SELECT t.suggestion, r.final_goals_h, r.final_goals_a, r.btts_yes
          FROM tips t
          LEFT JOIN match_results r ON r.match_id = t.match_id
          WHERE t.created_ts BETWEEN %s AND %s
            AND t.market = %s
            AND t.suggestion <> 'HARVEST' AND t.sent_ok=1
        """, (start, end, market)).fetchall()
    graded = wins = 0
    for (sugg, gh, ga, btts) in rows:
        out = _tip_outcome_for_result(sugg, {"final_goals_h": gh, "final_goals_a": ga, "btts_yes": btts})
        if out is None:
            continue
        graded += 1
        wins += 1 if out == 1 else 0
    prec = (wins/graded) if graded else 0.0
    return graded, wins, prec

def auto_tune_thresholds():
    if not AUTO_TUNE_ENABLE:
        return
    markets = ["BTTS", "1X2"] + [f"Over/Under { _fmt_line(ln) }" for ln in OU_LINES]
    for mk in markets:
        graded, wins, prec = _recent_precision_for_market(mk, days=3)
        if graded < 30:  # need signal
            continue
        cur = _get_market_threshold(mk)
        if prec < TARGET_PRECISION - 0.02:
            cur = min(MAX_THRESH, cur + TUNE_STEP)
        elif prec > TARGET_PRECISION + 0.02:
            cur = max(MIN_THRESH, cur - TUNE_STEP)
        _set_market_threshold(mk, cur)
        logger.info("[AUTO-TUNE] %s graded=%d wins=%d prec=%.3f -> new_threshold=%.1f",
                    mk, graded, wins, prec, cur)

def _compare_and_promote(current_metrics: dict, new_metrics: dict) -> bool:
    """Return True if new metrics are better (lower logloss & brier; tie-break by accuracy)."""
    try:
        c_ll = float((current_metrics or {}).get("logloss", 1e9))
        n_ll = float((new_metrics or {}).get("logloss", 1e9))
        c_br = float((current_metrics or {}).get("brier", 1e9))
        n_br = float((new_metrics or {}).get("brier", 1e9))
        c_acc = float((current_metrics or {}).get("acc", 0.0))
        n_acc = float((new_metrics or {}).get("acc", 0.0))
        better = (n_ll < c_ll and n_br < c_br) or (abs(n_ll - c_ll) < 1e-6 and n_br < c_br) \
                 or (n_ll <= c_ll and n_br <= c_br and n_acc > c_acc)
        return better
    except Exception:
        return False

def _auto_train_and_promote():
    if not TRAIN_ENABLE:
        return
    try:
        res = train_models()
        if not (isinstance(res, dict) and res.get("ok")):
            logger.info("[AUTO-TRAIN] skipped/failed: %s", res)
            return
        latest = get_setting("model_metrics_latest")
        if not latest:
            return
        metrics = json.loads(latest)
        active = get_setting("model_metrics_active")
        active_m = json.loads(active) if active else {}
        improved = False
        for name, met in metrics.items():
            if name in ("trained_at_utc", "features"):
                continue
            old = (active_m or {}).get(name, {})
            if _compare_and_promote(old, met):
                improved = True
        if improved:
            set_setting("model_metrics_active", latest)
            logger.info("[AUTO-TRAIN] promoted models based on metrics")
        else:
            logger.info("[AUTO-TRAIN] new models not better; kept current")
    except Exception as e:
        logger.warning("[AUTO-TRAIN] error: %s", e)

# Optional: market scoring (small multiplier by recent precision). Off by default.
def _market_score(market: str, days: int = None) -> float:
    if not MARKET_SCORE_ENABLE:
        return 1.0
    days = days or MARKET_SCORE_DAYS
    graded, wins, prec = _recent_precision_for_market(market, days)
    if graded < 50:
        return 1.0
    # map precision 0.45..0.75 → 0.9..1.1
    return 0.9 + 0.2 * max(0.0, min(1.0, (prec - 0.45) / 0.30))

# ── Results backfill & accuracy
BERLIN_TZ = ZoneInfo("Europe/Berlin")

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
    """
    Find matches we tipped on but don't have results for yet; query API, store finals.
    """
    now_ts = int(time.time())
    cutoff = now_ts - 2 * 24 * 3600  # don't chase ancient games too aggressively
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

def _parse_ou_line_from_suggestion(s: str) -> Optional[float]:
    try:
        # "Over 2.5 Goals" / "Under 1.5 Goals"
        parts = (s or "").split()
        for tok in parts:
            try:
                return float(tok)
            except Exception:
                pass
    except Exception:
        pass
    return None

def _tip_outcome_for_result(suggestion: str, res: Dict[str, Any]) -> Optional[int]:
    """
    Returns 1=win, 0=loss, None=push/unknown/not gradable.
    """
    if not res:
        return None
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

    if s == "BTTS: Yes":
        return 1 if btts == 1 else 0
    if s == "BTTS: No":
        return 1 if btts == 0 else 0

    if s == "Home Win":
        return 1 if gh > ga else 0
    if s == "Away Win":
        return 1 if ga > gh else 0
    if s == "Draw":
        return 1 if gh == ga else 0
    return None

def daily_accuracy_digest() -> Optional[str]:
    """
    Compute yesterday's accuracy (Berlin time) and send Telegram digest.
    """
    if not DAILY_ACCURACY_DIGEST_ENABLE:
        return None

    now_local = datetime.now(BERLIN_TZ)
    y_start = (now_local - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    y_end = y_start + timedelta(days=1)

    y_start_ts = int(y_start.timestamp())
    y_end_ts = int(y_end.timestamp())

    # Ensure we have as many finals as possible
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
        total += 1
        res = {"final_goals_h": gh, "final_goals_a": ga, "btts_yes": btts}
        out = _tip_outcome_for_result(sugg, res)
        if out is None:
            continue
        graded += 1
        wins += 1 if out == 1 else 0
        m = by_market.setdefault(market or "?", {"graded": 0, "wins": 0})
        m["graded"] += 1
        m["wins"] += 1 if out == 1 else 0

    if graded == 0:
        msg = "📊 Daily Digest\nNo graded tips for yesterday."
    else:
        acc = 100.0 * wins / max(1, graded)
        lines = [f"📊 <b>Daily Digest</b> (yesterday, Berlin time)",
                 f"Tips sent: {total}  •  Graded: {graded}  •  Wins: {wins}  •  Accuracy: {acc:.1f}%"]
        for mk, st in sorted(by_market.items()):
            if st["graded"] == 0:
                continue
            a = 100.0 * st["wins"] / st["graded"]
            lines.append(f"• {escape(mk)} — {st['wins']}/{st['graded']} ({a:.1f}%)")
        msg = "\n".join(lines)

    send_telegram(msg)
    return msg

# ── Admin / Auth helpers
def _require_admin() -> None:
    key = request.headers.get("X-API-Key") or request.args.get("key") or (request.json or {}).get("key") if request.is_json else None
    if not ADMIN_API_KEY or key != ADMIN_API_KEY:
        abort(401)

def _ou_n_needed_over(goals_sum: int, line: float) -> int:
    # how many goals needed to finish above the line
    import math
    return max(0, int(math.floor(line + 1e-9) + 1) - int(goals_sum))

def _ou_over_already_won(goals_sum: int, line: float) -> bool:
    return _ou_n_needed_over(goals_sum, line) <= 0

def _ou_under_already_lost(goals_sum: int, line: float) -> bool:
    # under X.5 is already lost when goals_sum >= ceil(line+epsilon)
    import math
    return int(goals_sum) >= int(math.floor(line + 1e-9) + 1)

def _btts_decided(gh: int, ga: int) -> bool:
    return (gh > 0 and ga > 0)  # BTTS:Yes already true; also BTTS:No impossible

def _apply_early_conf_cap(prob_pct: float, minute: int) -> float:
    """
    Linearly cap displayed confidence early-game to avoid unrealistic 90%+ at 1'.
    At 0': cap = EARLY_CONF_CAP_LO (e.g., 70%)
    At EARLY_CONF_CAP_MINUTE or later: cap = 100%
    Between: linear interpolation.
    """
    if not EARLY_CONF_CAP_ENABLE:
        return prob_pct
    m = max(1, int(EARLY_CONF_CAP_MINUTE))
    if minute >= m:
        return prob_pct
    lo = max(0.0, min(100.0, float(EARLY_CONF_CAP_LO)))
    cap = lo + (100.0 - lo) * (minute / m)
    return min(prob_pct, cap)

# ── Flask endpoints
@app.route("/")
def root():
    return jsonify({"ok": True, "name": "live-tips-backend", "version": 1})

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

@app.route("/admin/backfill-results", methods=["POST"])
def http_backfill():
    _require_admin()
    n = backfill_results_for_open_matches()
    return jsonify({"ok": True, "updated": n})

@app.route("/admin/train", methods=["POST"])
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

@app.route("/tips/latest")
def http_latest_tips():
    limit = int(request.args.get("limit", "50"))
    market = request.args.get("market")
    sent_only = request.args.get("sent_only", "1") not in ("0","false","False","no","NO")
    include_harvest = request.args.get("include_harvest", "0") not in ("0","false","False","no","NO")
    where = []
    params: List[Any] = []
    if sent_only:
        where.append("sent_ok=1")
    if not include_harvest:
        where.append("suggestion<>'HARVEST'")
    if market:
        where.append("market=%s")
        params.append(market)
    sql = "SELECT match_id,league,home,away,market,suggestion,confidence,score_at_tip,minute,created_ts FROM tips"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY created_ts DESC LIMIT %s"
    params.append(max(1, min(500, limit)))
    with db_conn() as conn:
        rows = conn.execute(sql, tuple(params)).fetchall()
    tips = []
    for r in rows:
        tips.append({
            "match_id": int(r[0]),
            "league": r[1],
            "home": r[2],
            "away": r[3],
            "market": r[4],
            "suggestion": r[5],
            "confidence": float(r[6]),
            "score_at_tip": r[7],
            "minute": int(r[8]),
            "created_ts": int(r[9]),
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

@app.route("/feedback", methods=["POST"])
def http_feedback():
    js = request.get_json(silent=True) or {}
    mid = js.get("match_id")
    verdict = js.get("verdict")
    if mid is None or verdict not in (0, 1):
        abort(400)
    with db_conn() as conn:
        conn.execute(
            "INSERT INTO feedback(match_id, verdict, created_ts) VALUES(%s,%s,%s) "
            "ON CONFLICT (match_id) DO UPDATE SET verdict=EXCLUDED.verdict, created_ts=EXCLUDED.created_ts",
            (int(mid), int(verdict), int(time.time()))
        )
    return jsonify({"ok": True})

@app.route("/telegram/webhook/<secret>", methods=["POST"])
def telegram_webhook(secret: str):
    if (WEBHOOK_SECRET or "") != secret:
        abort(403)
    update = request.get_json(silent=True) or {}
    # Simple command responder to keep Telegram webhook alive
    try:
        msg = (update.get("message") or {}).get("text") or ""
        chat_id = ((update.get("message") or {}).get("chat") or {}).get("id")
        if msg.startswith("/start"):
            send_telegram("👋 Live tips bot is online.")
        elif msg.startswith("/ping"):
            send_telegram("✅ Pong")
        elif msg.startswith("/digest"):
            daily_accuracy_digest()
        elif msg.startswith("/scan"):
            # Protect with admin key if provided as "/scan <key>"
            parts = msg.split()
            if len(parts) > 1 and ADMIN_API_KEY and parts[1] == ADMIN_API_KEY:
                saved, live = production_scan()
                send_telegram(f"🔁 Scan done. Saved: {saved}, Live seen: {live}")
            else:
                send_telegram("🔒 Admin key required.")
        elif msg.startswith("/latest"):
            with db_conn() as conn:
                r = conn.execute(
                    "SELECT home,away,market,suggestion,confidence,minute FROM tips "
                    "WHERE suggestion<>'HARVEST' ORDER BY created_ts DESC LIMIT 5"
                ).fetchall()
            if not r:
                send_telegram("No tips yet.")
            else:
                lines = []
                for (h,a,mk,sugg,conf,minute) in r:
                    lines.append(f"{escape(h)}–{escape(a)} • {mk} • {sugg} • {conf:.1f}% @ {minute}'")
                send_telegram("Last tips:\n" + "\n".join(lines))
    except Exception as e:
        logger.warning("telegram webhook parse error: %s", e)
    return jsonify({"ok": True})

# ── Production scan
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
                if not fid:
                    continue

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
                if not stats_coverage_ok(feat, minute):
                    continue
                if minute < TIP_MIN_MINUTE:
                    continue

                # HARVEST snapshots for training (rate-limit to every ~3 min by minute%3)
                if HARVEST_MODE and minute >= TRAIN_MIN_MINUTE and minute % 3 == 0:
                    try:
                        save_snapshot_from_match(m, feat)
                    except Exception as e:
                        logger.warning("[HARVEST] snapshot failed for %s: %s", fid, e)

                league_id, league = _league_name(m)
                home, away = _teams(m)
                score_txt = _pretty_score(m)

                candidates: List[Tuple[str, str, float]] = []

                # O/U lines (math+ML blend or ML-only per ONLY_MODEL_MODE)
                goals_sum_now = int(feat.get("goals_sum", 0))
                for line in OU_LINES:
                    # Skip trivial/decided cases
                    if _ou_over_already_won(goals_sum_now, line):
                        continue  # Over already true; don't tip
                    if _ou_under_already_lost(goals_sum_now, line):
                        # Under already impossible; no point scoring probabilities
                        pass  # we'll still compute p_over for Over side; Under is skipped below

                    p_over_math = ou_over_probability(feat, line, TOTAL_MATCH_MINUTES)
                    mdl_line = _load_ou_model_for_line(line)
                    if mdl_line:
                        p_over_ml = _score_prob(feat, mdl_line)
                        p_over = BLEND_ML_WEIGHT * p_over_ml + BLEND_MATH_WEIGHT * p_over_math
                    else:
                        p_over = p_over_math
                    p_over = _soften_final_prob(p_over)

                    market_name = f"Over/Under {_fmt_line(line)}"
                    thr_ou = _get_market_threshold(market_name)

                    # Over side (only if not already won)
                    if p_over * 100.0 >= thr_ou:
                        prob_adj = p_over * _market_score(market_name)
                        candidates.append((market_name, f"Over {_fmt_line(line)} Goals", prob_adj))

                    # Under side: skip if already impossible or late suppress; otherwise consider
                    if minute <= UNDER_SUPPRESS_AFTER_MIN and not _ou_under_already_lost(goals_sum_now, line):
                        p_under = _soften_final_prob(1.0 - p_over)
                        if p_under * 100.0 >= thr_ou:
                            prob_adj = p_under * _market_score(market_name)
                            candidates.append((market_name, f"Under {_fmt_line(line)} Goals", prob_adj))

                # BTTS
                gh_now = int(feat.get("goals_h", 0))
                ga_now = int(feat.get("goals_a", 0))

                # If BTTS is already decided (both scored), skip BTTS tips entirely
                if not _btts_decided(gh_now, ga_now):
                    p_btts_math = btts_yes_probability(feat, TOTAL_MATCH_MINUTES)
                    mdl_btts = load_model_from_settings("BTTS_YES")
                    if mdl_btts:
                        p_btts_ml = _score_prob(feat, mdl_btts)
                        p_btts = BLEND_ML_WEIGHT * p_btts_ml + BLEND_MATH_WEIGHT * p_btts_math
                    else:
                        p_btts = p_btts_math
                    p_btts = _soften_final_prob(p_btts)

                    thr_btts = _get_market_threshold("BTTS")
                    if not (minute >= BTTS_LATE_MINUTE and (gh_now == 0 or ga_now == 0)):
                        if p_btts * 100.0 >= thr_btts:
                            prob_adj = p_btts * _market_score("BTTS")
                            candidates.append(("BTTS", "BTTS: Yes", prob_adj))

                    p_btts_no = _soften_final_prob(1.0 - p_btts)
                    if minute <= UNDER_SUPPRESS_AFTER_MIN and p_btts_no * 100.0 >= thr_btts:
                        prob_adj = p_btts_no * _market_score("BTTS")
                        candidates.append(("BTTS", "BTTS: No", prob_adj))

                # 1X2 (pick only best side above threshold)
                p_home, p_draw, p_away = wld_probabilities(feat, TOTAL_MATCH_MINUTES)
                wld_map = [("1X2", "Home Win", p_home), ("1X2", "Draw", p_draw), ("1X2", "Away Win", p_away)]
                best_market = max(wld_map, key=lambda x: x[2])
                best_market = (best_market[0], best_market[1], _soften_final_prob(best_market[2]))
                thr_1x2 = _get_market_threshold("1X2")
                if best_market[2] * 100.0 >= thr_1x2:
                    prob_adj = best_market[2] * _market_score("1X2")
                    candidates.append((best_market[0], best_market[1], prob_adj))

                if not candidates:
                    continue

                candidates.sort(key=lambda x: x[2], reverse=True)
                per_match = 0
                base_now = int(time.time())

                for idx, (market_txt, suggestion, prob) in enumerate(candidates):
                    if suggestion not in ALLOWED_SUGGESTIONS:
                        continue
                    if per_match >= max(1, PREDICTIONS_PER_MATCH):
                        break

                    created_ts = base_now + idx
                    prob_pct = round(prob * 100.0, 1)
                    prob_pct = _apply_early_conf_cap(prob_pct, minute)
                    if prob_pct >= 100.0: prob_pct = DISPLAY_MAX_PCT
                    if prob_pct <= 0.0:  prob_pct = DISPLAY_MIN_PCT

                    with db_conn() as conn2:
                        conn2.execute(
                            "INSERT INTO tips(match_id,league_id,league,home,away,market,suggestion,confidence,score_at_tip,minute,created_ts,sent_ok) "
                            "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,0)",
                            (fid, league_id, league, home, away, market_txt, suggestion, float(prob_pct), score_txt, minute, created_ts),
                        )
                        sent_ok = _send_tip(home, away, league, minute, score_txt, suggestion, float(prob_pct), feat)
                        if sent_ok:
                            conn2.execute("UPDATE tips SET sent_ok=1 WHERE match_id=%s AND created_ts=%s", (fid, created_ts))

                    saved += 1
                    per_match += 1
                    if MAX_TIPS_PER_SCAN and saved >= MAX_TIPS_PER_SCAN:
                        break

                if MAX_TIPS_PER_SCAN and saved >= MAX_TIPS_PER_SCAN:
                    break

            except Exception as e:
                logger.exception("[PROD] failure on match: %s", e)
                continue

    logger.info(f"[PROD] saved={saved} live_seen={live_seen}")
    return saved, live_seen

# ── Scheduler setup
scheduler: Optional[BackgroundScheduler] = None

def _start_scheduler():
    global scheduler
    if scheduler is not None:
        return
    scheduler = BackgroundScheduler(timezone=BERLIN_TZ)

    # Frequent live scan
    scheduler.add_job(production_scan, "interval", seconds=30, id="scan", max_instances=1, coalesce=True)

    # Results backfill every 10 minutes
    scheduler.add_job(backfill_results_for_open_matches, "interval", minutes=10, id="results_backfill",
                      kwargs={"max_rows": 400}, max_instances=1, coalesce=True)

    # Heartbeat: record setting & optional log
    if HEARTBEAT_ENABLE:
        def _heartbeat():
            set_setting("heartbeat_ts", str(int(time.time())))
            logger.info("[HEARTBEAT] alive")
        scheduler.add_job(_heartbeat, "interval", minutes=5, id="heartbeat", max_instances=1, coalesce=True)

    # Daily accuracy digest at configured local time
    if DAILY_ACCURACY_DIGEST_ENABLE:
        trig = CronTrigger(hour=DAILY_ACCURACY_HOUR, minute=DAILY_ACCURACY_MINUTE, timezone=BERLIN_TZ)
        scheduler.add_job(daily_accuracy_digest, trig, id="daily_digest", max_instances=1, coalesce=True)

    # Auto-train & promote (daily, Berlin morning) + kick once after boot
    scheduler.add_job(_auto_train_and_promote, CronTrigger(hour=4, minute=12, timezone=BERLIN_TZ),
                      id="auto_train_promote", max_instances=1, coalesce=True)
    scheduler.add_job(_auto_train_and_promote, "date",
                      run_date=datetime.now(BERLIN_TZ)+timedelta(seconds=5),
                      id="auto_train_boot", replace_existing=True)

    # Auto-tune per-market thresholds hourly
    scheduler.add_job(auto_tune_thresholds, "interval", minutes=60, id="auto_tune", max_instances=1, coalesce=True)

    scheduler.start()
    logger.info("[SCHED] started")

# ── App startup
def _on_boot():
    init_db()
    set_setting("boot_ts", str(int(time.time())))
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        try:
            send_telegram("🚀 Live tips backend started.")
        except Exception:
            pass
    if os.getenv("DISABLE_SCHEDULER", "0") in ("1","true","True","YES","yes"):
        logger.info("[SCHED] disabled by env")
    else:
        _start_scheduler()

_on_boot()

if __name__ == "__main__":
    # Production WSGI servers (gunicorn/uvicorn) will ignore this and import app only.
    port = int(os.getenv("PORT", "8080"))
    host = os.getenv("HOST", "0.0.0.0")
    app.run(host=host, port=port)
