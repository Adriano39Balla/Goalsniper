# file: main.py
"""
Flask backend for live football tips with blended math+ML.
Now includes:
- Probability softening & caps (no 0%/100%),
- Platt-calibrated ML (from train_models.py),
- Multi-line O/U, BTTS Yes/No, and 1X2.
"""

import os
import json
import time
import logging
import requests
import psycopg2
from html import escape
from zoneinfo import ZoneInfo
from datetime import datetime
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
CONF_THRESHOLD = int(os.getenv("CONF_THRESHOLD", "65"))  # slightly higher default
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

def _parse_lines(env_val: str, default: List[float]) -> List[float]:
    out = []
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

# Final-prob softening & display caps
GLOBAL_MIN_PROB = float(os.getenv("GLOBAL_MIN_PROB", "0.02"))   # 2%..98%
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
    raise SystemExit("DATABASE_URL is required (Supabase Postgres).")

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

# ── DB adapter & helpers (unchanged essentials)
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

# ── Init DB / Telegram / API / Feature extraction (as provided earlier)
# (kept identical to your previous working version; shortened here to essential parts you run)
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

def send_telegram(message: str, inline_keyboard: Optional[list] = None) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML", "disable_web_page_preview": True}
    if inline_keyboard:
        payload["reply_markup"] = json.dumps({"inline_keyboard": inline_keyboard})
    try:
        res = session.post(f"{TELEGRAM_API_URL}/sendMessage", data=payload, timeout=10)
        return res.ok
    except Exception:
        return False

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
    allowed = {"1H","HT","2H","ET","BT","P"}
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

# ── Models & scoring
MODEL_KEYS_ORDER = ["model_v2:{name}", "model_latest:{name}", "model:{name}"]

def _sigmoid(x: float) -> float:
    try:
        if x < -50: return 1e-22
        if x >  50: return 1.0 - 1e-22
        import math
        return 1.0 / (1.0 + math.exp(-x))
    except Exception:
        return 0.5

def load_model_from_settings(name: str) -> Optional[Dict[str, Any]]:
    for pat in MODEL_KEYS_ORDER:
        key = pat.format(name=name)
        raw = get_setting(key)
        if not raw: continue
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
        return _sigmoid(a * _logit(max(1e-12, min(1-1e-12, p))) + b)
    import math
    p = max(1e-12, min(1 - 1e-12, float(p)))
    logit = math.log(p / (1.0 - p))
    return _sigmoid(a * logit + b)

def _logit(p: float) -> float:
    import math
    p = max(1e-12, min(1 - 1e-12, float(p)))
    return math.log(p / (1 - p))

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

def _fmt_line(line: float) -> str:
    return f"{line}".rstrip("0").rstrip(".")

def _load_ou_model_for_line(line: float) -> Optional[Dict[str, Any]]:
    name = f"OU_{_fmt_line(line)}"
    mdl = load_model_from_settings(name)
    if not mdl and abs(line - 2.5) < 1e-6:
        mdl = load_model_from_settings("O25")
    return mdl

def _soften_final_prob(p: float) -> float:
    # Why: avoid extremes at output
    p = 0.5 + GLOBAL_SOFT_ALPHA * (p - 0.5)
    return max(GLOBAL_MIN_PROB, min(1.0 - GLOBAL_MIN_PROB, p))

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

                league_id, league = _league_name(m)
                home, away = _teams(m)
                score_txt = _pretty_score(m)

                candidates: List[Tuple[str, str, float]] = []

                # O/U lines (blend math + ML if available)
                for line in OU_LINES:
                    p_over_math = ou_over_probability(feat, line, TOTAL_MATCH_MINUTES)
                    mdl_line = _load_ou_model_for_line(line)
                    if mdl_line:
                        p_over_ml = _score_prob(feat, mdl_line)
                        p_over = BLEND_ML_WEIGHT * p_over_ml + BLEND_MATH_WEIGHT * p_over_math
                    else:
                        p_over = p_over_math
                    p_over = _soften_final_prob(p_over)

                    if p_over * 100.0 >= CONF_THRESHOLD:
                        candidates.append((f"OU_{_fmt_line(line)}", f"Over {_fmt_line(line)} Goals", p_over))

                    if minute <= UNDER_SUPPRESS_AFTER_MIN:
                        p_under = _soften_final_prob(1.0 - p_over)
                        if p_under * 100.0 >= CONF_THRESHOLD:
                            candidates.append((f"OU_{_fmt_line(line)}", f"Under {_fmt_line(line)} Goals", p_under))

                # BTTS
                p_btts_math = btts_yes_probability(feat, TOTAL_MATCH_MINUTES)
                mdl_btts = load_model_from_settings("BTTS_YES")
                if mdl_btts:
                    p_btts_ml = _score_prob(feat, mdl_btts)
                    p_btts = BLEND_ML_WEIGHT * p_btts_ml + BLEND_MATH_WEIGHT * p_btts_math
                else:
                    p_btts = p_btts_math
                p_btts = _soften_final_prob(p_btts)

                if not (minute >= BTTS_LATE_MINUTE and (feat.get("goals_h", 0) == 0 or feat.get("goals_a", 0) == 0)):
                    if p_btts * 100.0 >= CONF_THRESHOLD:
                        candidates.append(("BTTS", "BTTS: Yes", p_btts))
                p_btts_no = _soften_final_prob(1.0 - p_btts)
                if minute <= UNDER_SUPPRESS_AFTER_MIN and p_btts_no * 100.0 >= CONF_THRESHOLD:
                    candidates.append(("BTTS", "BTTS: No", p_btts_no))

                # 1X2 (math only; analytics already softened)
                p_home, p_draw, p_away = wld_probabilities(feat, TOTAL_MATCH_MINUTES)
                wld_map = [("1X2", "Home Win", p_home), ("1X2", "Draw", p_draw), ("1X2", "Away Win", p_away)]
                best_market = max(wld_map, key=lambda x: x[2])
                best_market = (best_market[0], best_market[1], _soften_final_prob(best_market[2]))
                if best_market[2] * 100.0 >= CONF_THRESHOLD:
                    candidates.append(best_market)

                if not candidates:
                    continue

                candidates.sort(key=lambda x: x[2], reverse=True)
                per_match = 0
                base_now = int(time.time())

                for idx, (market_code, suggestion, prob) in enumerate(candidates):
                    if suggestion not in ALLOWED_SUGGESTIONS:
                        continue
                    if per_match >= max(1, PREDICTIONS_PER_MATCH):
                        break

                    created_ts = base_now + idx
                    prob_pct = round(prob * 100.0, 1)
                    # Never display 0%/100% in UI
                    if prob_pct >= 100.0: prob_pct = DISPLAY_MAX_PCT
                    if prob_pct <= 0.0: prob_pct = DISPLAY_MIN_PCT

                    market_txt = f"Over/Under {market_code.split('_',1)[1]}" if market_code.startswith("OU_") else market_code

                    conn.execute(
                        "INSERT INTO tips(match_id,league_id,league,home,away,market,suggestion,confidence,score_at_tip,minute,created_ts,sent_ok) "
                        "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,0)",
                        (fid, league_id, league, home, away, market_txt, suggestion, float(prob_pct), score_txt, minute, created_ts),
                    )

                    sent_ok = _send_tip(home, away, league, minute, score_txt, suggestion, float(prob_pct), feat)
                    if sent_ok:
                        conn.execute("UPDATE tips SET sent_ok=1 WHERE match_id=%s AND created_ts=%s", (fid, created_ts))

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

# (harvest/backfill/stats/MOTD/routes/scheduler remain same as in your working file; omitted for brevity)

# ── Security headers, routes, and main() stay identical to your prior version, except CONF defaults above.
# Ensure you keep your existing routes and scheduler jobs here (unchanged) or paste from previous working file.
# For brevity, not duplicating the whole routing section again.

# Minimal home + entrypoint for completeness in this snippet:
@app.route("/")
def home():
    mode = "HARVEST" if HARVEST_MODE else "PRODUCTION"
    return f"🤖 Robi Superbrain is active ({mode}) · DB=Postgres"

def main():
    init_db()
    scheduler = BackgroundScheduler()
    if HARVEST_MODE:
        scheduler.add_job(lambda: None, "interval", minutes=60, id="noop", replace_existing=True)  # keep scheduler alive
    else:
        scheduler.add_job(production_scan, CronTrigger(minute="*/5", timezone=ZoneInfo("Europe/Berlin")),
                          id="production_scan", replace_existing=True)
    scheduler.start()
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
