import os
import json
import time
import logging
import requests
import psycopg2
import subprocess, shlex
from html import escape
from zoneinfo import ZoneInfo
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from flask import Flask, jsonify, request, abort
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

# ──────────────────────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
app = Flask(__name__)

# ── Env ───────────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")
API_KEY            = os.getenv("API_KEY")
ADMIN_API_KEY      = os.getenv("ADMIN_API_KEY")
PUBLIC_BASE_URL    = (os.getenv("PUBLIC_BASE_URL") or "").rstrip("/")
WEBHOOK_SECRET     = os.getenv("TELEGRAM_WEBHOOK_SECRET")
HEARTBEAT_ENABLE   = os.getenv("HEARTBEAT_ENABLE", "1") not in ("0","false","False","no","NO")

HARVEST_MODE       = os.getenv("HARVEST_MODE", "1") not in ("0","false","False","no","NO")
CONF_THRESHOLD     = int(os.getenv("CONF_THRESHOLD", "60"))
MAX_TIPS_PER_SCAN  = int(os.getenv("MAX_TIPS_PER_SCAN", "25"))
DUP_COOLDOWN_MIN   = int(os.getenv("DUP_COOLDOWN_MIN", "20"))

O25_LATE_MINUTE     = int(os.getenv("O25_LATE_MINUTE", "88"))
O25_LATE_MIN_GOALS  = int(os.getenv("O25_LATE_MIN_GOALS", "2"))
BTTS_LATE_MINUTE    = int(os.getenv("BTTS_LATE_MINUTE", "88"))
UNDER_SUPPRESS_AFTER_MIN = int(os.getenv("UNDER_SUPPRESS_AFTER_MIN", "82"))

ONLY_MODEL_MODE          = os.getenv("ONLY_MODEL_MODE", "0") not in ("0","false","False","no","NO")
REQUIRE_STATS_MINUTE     = int(os.getenv("REQUIRE_STATS_MINUTE", "35"))
REQUIRE_DATA_FIELDS      = int(os.getenv("REQUIRE_DATA_FIELDS", "2"))

LEAGUE_PRIORITY_IDS = [int(x) for x in (os.getenv("MOTD_LEAGUE_IDS", "39,140,135,78,61,2").split(",")) if x.strip().isdigit()]
MOTD_PREDICT        = os.getenv("MOTD_PREDICT", "1") not in ("0","false","False","no","NO")
MOTD_MIN_SAMPLES    = int(os.getenv("MOTD_MIN_SAMPLES", "30"))
MOTD_CONF_MIN       = int(os.getenv("MOTD_CONF_MIN", "65"))

ALLOWED_SUGGESTIONS = {"Over 2.5 Goals", "Under 2.5 Goals", "BTTS: Yes", "BTTS: No"}

TRAIN_ENABLE       = os.getenv("TRAIN_ENABLE", "1") not in ("0","false","False","no","NO")
TRAIN_MIN_MINUTE   = int(os.getenv("TRAIN_MIN_MINUTE", "15"))
TRAIN_TEST_SIZE    = float(os.getenv("TRAIN_TEST_SIZE", "0.25"))

# Postgres (Supabase) — REQUIRED
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise SystemExit("DATABASE_URL is required (Supabase Postgres).")

# ── External APIs ─────────────────────────────────────────────────────────────
BASE_URL         = "https://v3.football.api-sports.io"
FOOTBALL_API_URL = f"{BASE_URL}/fixtures"
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
HEADERS          = {"x-apisports-key": API_KEY, "Accept": "application/json"}
INPLAY_STATUSES  = ["1H","HT","2H","ET","BT","P"]

# ── HTTP session ──────────────────────────────────────────────────────────────
session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], respect_retry_after_header=True)
session.mount("https://", HTTPAdapter(max_retries=retries))

# ── Caches ────────────────────────────────────────────────────────────────────
STATS_CACHE: Dict[int, Tuple[float, list]] = {}
EVENTS_CACHE: Dict[int, Tuple[float, list]] = {}
CAL_CACHE: Dict[str, Any] = {"ts": 0, "bins": []}

# ──────────────────────────────────────────────────────────────────────────────
# Postgres adapter
# ──────────────────────────────────────────────────────────────────────────────
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

# ── Init DB ───────────────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────────────────────────
# Telegram
# ──────────────────────────────────────────────────────────────────────────────
def send_telegram(message: str, inline_keyboard: Optional[list] = None) -> bool:
    """
    Sends an HTML-formatted message. We don't globally escape because pieces of
    the text are already escaped selectively where needed.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    if inline_keyboard:
        payload["reply_markup"] = json.dumps({"inline_keyboard": inline_keyboard})

    try:
        res = session.post(f"{TELEGRAM_API_URL}/sendMessage", data=payload, timeout=10)
        return res.ok
    except Exception:
        return False

# ──────────────────────────────────────────────────────────────────────────────
# External API helpers
# ──────────────────────────────────────────────────────────────────────────────
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

    allowed = {"1H", "HT", "2H", "ET", "BT", "P"}  # in-play status filter
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

# ──────────────────────────────────────────────────────────────────────────────
# Fixture helpers
# ──────────────────────────────────────────────────────────────────────────────
MAX_IDS_PER_REQ = 20

def fetch_fixtures_by_ids(ids: List[int]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    if not ids:
        return out
    for i in range(0, len(ids), MAX_IDS_PER_REQ):
        chunk = ids[i : i + MAX_IDS_PER_REQ]
        js = _api_get(FOOTBALL_API_URL, {"ids": ",".join(map(str, chunk)), "timezone": "UTC"})
        resp = js.get("response", []) if isinstance(js, dict) else []
        for fx in resp:
            fid = (fx.get("fixture") or {}).get("id")
            if fid:
                out[int(fid)] = fx

        # Fallback: request missing individually
        missing = [fid for fid in chunk if fid not in out]
        for fid in missing:
            js1 = _api_get(FOOTBALL_API_URL, {"id": fid, "timezone": "UTC"})
            r1 = (js1.get("response") if isinstance(js1, dict) else []) or []
            if r1:
                out[int(fid)] = r1[0]
        time.sleep(0.15)
    return out

def fetch_fixtures_by_ids_hyphen(ids: List[int]) -> Dict[int, Dict[str, Any]]:
    if not ids:
        return {}
    out: Dict[int, Dict[str, Any]] = {}
    for i in range(0, len(ids), MAX_IDS_PER_REQ):
        chunk = ids[i : i + MAX_IDS_PER_REQ]
        js = _api_get(FOOTBALL_API_URL, {"ids": "-".join(map(str, chunk)), "timezone": "UTC"})
        resp = js.get("response", []) if isinstance(js, dict) else []
        for fx in resp:
            fid = (fx.get("fixture") or {}).get("id")
            if fid:
                out[int(fid)] = fx
        time.sleep(0.15)
    return out

# ──────────────────────────────────────────────────────────────────────────────
# Feature extraction & snapshots
# ──────────────────────────────────────────────────────────────────────────────
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

    red_h = 0
    red_a = 0
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
        "sot_h": float(sot_h),
        "sot_a": float(sot_a),
        "sot_sum": float(sot_h + sot_a),
        "cor_h": float(cor_h),
        "cor_a": float(cor_a),
        "cor_sum": float(cor_h + cor_a),
        "pos_h": float(pos_h),
        "pos_a": float(pos_a),
        "red_h": float(red_h),
        "red_a": float(red_a),
        "red_sum": float(red_h + red_a),
    }

def stats_coverage_ok(feat: Dict[str, float], minute: int) -> bool:
    """
    After REQUIRE_STATS_MINUTE, demand at least REQUIRE_DATA_FIELDS of the
    following signals to be non-zero: xG_sum, SOT_sum, COR_sum, max(pos_h,pos_a).
    """
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
    fx = m.get("fixture", {}) or {}
    lg = m.get("league", {}) or {}

    fid = int(fx.get("id"))
    league_id = int(lg.get("id") or 0)
    league = f"{lg.get('country', '')} - {lg.get('name', '')}".strip(" -")

    home = (m.get("teams") or {}).get("home", {}).get("name", "")
    away = (m.get("teams") or {}).get("away", {}).get("name", "")
    gh = (m.get("goals") or {}).get("home") or 0
    ga = (m.get("goals") or {}).get("away") or 0
    minute = int(feat.get("minute", 0))

    snapshot = {
        "minute": minute,
        "gh": gh,
        "ga": ga,
        "league_id": league_id,
        "market": "HARVEST",
        "suggestion": "HARVEST",
        "confidence": 0,
        "stat": {
            "xg_h": feat.get("xg_h", 0),
            "xg_a": feat.get("xg_a", 0),
            "sot_h": feat.get("sot_h", 0),
            "sot_a": feat.get("sot_a", 0),
            "cor_h": feat.get("cor_h", 0),
            "cor_a": feat.get("cor_a", 0),
            "pos_h": feat.get("pos_h", 0),
            "pos_a": feat.get("pos_a", 0),
            "red_h": feat.get("red_h", 0),
            "red_a": feat.get("red_a", 0),
        },
    }

    now = int(time.time())
    with db_conn() as conn:
        # Upsert snapshot
        conn.execute(
            """
            INSERT INTO tip_snapshots(match_id, created_ts, payload)
            VALUES (%s,%s,%s)
            ON CONFLICT (match_id, created_ts) DO UPDATE SET payload=EXCLUDED.payload
            """,
            (fid, now, json.dumps(snapshot)[:200000]),
        )

        # Also store a HARVEST row in tips (unsent)
        conn.execute(
            """
            INSERT INTO tips(match_id,league_id,league,home,away,market,suggestion,confidence,score_at_tip,minute,created_ts,sent_ok)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,0)
            """,
            (fid, league_id, league, home, away, "HARVEST", "HARVEST", 0.0, f"{gh}-{ga}", minute, now),
        )

# ──────────────────────────────────────────────────────────────────────────────
# Model loading & scoring
# ──────────────────────────────────────────────────────────────────────────────
MODEL_KEYS_ORDER = [
    "model_v2:{name}",
    "model_latest:{name}",
    "model:{name}",
]

def _sigmoid(x: float) -> float:
    try:
        if x < -50:
            return 1e-22
        if x > 50:
            return 1.0 - 1e-22
        import math

        return 1.0 / (1.0 + math.exp(-x))
    except Exception:
        return 0.5

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
            logging.warning("[MODEL] failed to parse %s: %s", key, e)
    return None

def _linpred(feat: Dict[str, float], weights: Dict[str, float], intercept: float) -> float:
    s = float(intercept or 0.0)
    for k, w in (weights or {}).items():
        s += float(w or 0.0) * float(feat.get(k, 0.0))
    return s

def _calibrate(p: float, cal: Dict[str, Any]) -> float:
    """
    Apply Platt-style or generic sigmoid calibration on the *probability* domain.
    """
    method = (cal or {}).get("method", "sigmoid")
    a = float((cal or {}).get("a", 1.0))
    b = float((cal or {}).get("b", 0.0))

    if method == "platt":
        return _sigmoid(a * float(p) + b)

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

# ──────────────────────────────────────────────────────────────────────────────
# Tip formatting/sending
# ──────────────────────────────────────────────────────────────────────────────
def _format_tip_message(
    home: str,
    away: str,
    league: str,
    minute: int,
    score_txt: str,
    suggestion: str,
    prob_pct: float,
    feat: Dict[str, float],
) -> str:
    stat_line = ""
    if any(
        [
            feat.get("xg_h", 0),
            feat.get("xg_a", 0),
            feat.get("sot_h", 0),
            feat.get("sot_a", 0),
            feat.get("cor_h", 0),
            feat.get("cor_a", 0),
            feat.get("pos_h", 0),
            feat.get("pos_a", 0),
            feat.get("red_h", 0),
            feat.get("red_a", 0),
        ]
    ):
        stat_line = (
            f"\n📊 xG {feat.get('xg_h',0):.2f}-{feat.get('xg_a',0):.2f}"
            f" • SOT {int(feat.get('sot_h',0))}-{int(feat.get('sot_a',0))}"
            f" • CK {int(feat.get('cor_h',0))}-{int(feat.get('cor_a',0))}"
        )
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

# ──────────────────────────────────────────────────────────────────────────────
# Production scan (models → tips)
# ──────────────────────────────────────────────────────────────────────────────
def production_scan() -> Tuple[int, int]:
    """
    Generate tips using models for O25 and BTTS: Yes.
    Rules:
      • require minimal stats coverage after REQUIRE_STATS_MINUTE
      • cooldown per match (DUP_COOLDOWN_MIN)
      • late-minute suppressors for O25 and BTTS
      • created_ts de-dup to avoid primary-key collisions
    """
    matches = fetch_live_matches()
    live_seen = len(matches)
    if live_seen == 0:
        logging.info("[PROD] no live matches")
        return 0, 0

    mdl_o25 = load_model_from_settings("O25")
    mdl_btts = load_model_from_settings("BTTS_YES")
    if not mdl_o25 and not mdl_btts:
        logging.warning("[PROD] no models available in settings — skipping.")
        return 0, live_seen

    saved = 0
    now_ts = int(time.time())

    with db_conn() as conn:
        for m in matches:
            try:
                fid = int((m.get("fixture", {}) or {}).get("id") or 0)
                if not fid:
                    continue

                # Cooldown: don't produce too often for same match
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

                preds: List[Tuple[str, str, float]] = []  # (market_code, suggestion, prob)

                # Over 2.5
                if mdl_o25:
                    p_over = _score_prob(feat, mdl_o25)
                    # If it's very late and the game has < required goals, skip
                    if not (minute >= O25_LATE_MINUTE and feat.get("goals_sum", 0) < O25_LATE_MIN_GOALS):
                        if p_over * 100.0 >= CONF_THRESHOLD:
                            preds.append(("O25", "Over 2.5 Goals", p_over))

                # BTTS: Yes
                if mdl_btts:
                    p_btts = _score_prob(feat, mdl_btts)
                    # Very late and at least one side on 0 → often dead, skip
                    if not (minute >= BTTS_LATE_MINUTE and (feat.get("goals_h", 0) == 0 or feat.get("goals_a", 0) == 0)):
                        if p_btts * 100.0 >= CONF_THRESHOLD:
                            preds.append(("BTTS", "BTTS: Yes", p_btts))

                if not preds:
                    continue

                preds.sort(key=lambda x: x[2], reverse=True)  # highest prob first

                league_id, league = _league_name(m)
                home, away = _teams(m)
                score_txt = _pretty_score(m)

                # Insert & send (cap by MAX_TIPS_PER_SCAN)
                base_now = int(time.time())
                for idx, (market_code, suggestion, prob) in enumerate(preds):
                    created_ts = base_now + idx  # avoid PK clash if we insert >1 row this second
                    market_txt = "Over/Under 2.5" if market_code == "O25" else "BTTS"
                    prob_pct = float(prob * 100.0)

                    # 1) insert as unsent
                    conn.execute(
                        """
                        INSERT INTO tips(match_id,league_id,league,home,away,market,suggestion,confidence,score_at_tip,minute,created_ts,sent_ok)
                        VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,0)
                        """,
                        (
                            fid,
                            league_id,
                            league,
                            home,
                            away,
                            market_txt,
                            suggestion,
                            prob_pct,
                            score_txt,
                            minute,
                            created_ts,
                        ),
                    )

                    # 2) try to send now
                    sent_ok = _send_tip(home, away, league, minute, score_txt, suggestion, prob_pct, feat)

                    # 3) mark as sent if delivered
                    if sent_ok:
                        conn.execute("UPDATE tips SET sent_ok=1 WHERE match_id=%s AND created_ts=%s", (fid, created_ts))

                    saved += 1
                    if MAX_TIPS_PER_SCAN and saved >= MAX_TIPS_PER_SCAN:
                        break

                if MAX_TIPS_PER_SCAN and saved >= MAX_TIPS_PER_SCAN:
                    break

            except Exception as e:
                logging.exception(f"[PROD] failure on match: {e}")
                continue

    logging.info(f"[PROD] saved={saved} live_seen={live_seen}")
    return saved, live_seen

# ── Routes & Security Headers ─────────────────────────────────────────────────
# ── Routes & Security Headers ─────────────────────────────────────────────────
@app.after_request
def add_security_headers(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["Cache-Control"] = "no-store"
    return resp

def _require_api_key():
    key = request.headers.get("X-API-Key") or request.args.get("key")
    if not ADMIN_API_KEY or key != ADMIN_API_KEY:
        abort(401)

@app.route("/")
def home():
    mode = "HARVEST" if HARVEST_MODE else "PRODUCTION"
    return f"🤖 Robi Superbrain is active ({mode}) · DB=Postgres"

# ── Harvest (bulk) ────────────────────────────────────────────────────────────
@app.route("/harvest/bulk_leagues", methods=["POST"])
def harvest_bulk_leagues_post():
    _require_api_key()
    body = request.get_json(silent=True) or {}
    leagues = body.get("leagues") or []
    seasons = body.get("seasons") or [2024]
    return jsonify(_run_bulk_leagues(leagues, seasons))

# /harvest/bulk_leagues?key=...&seasons=2023,2024
@app.route("/harvest/bulk_leagues", methods=["GET"])
def harvest_bulk_leagues_get():
    _require_api_key()
    seasons_q = (request.args.get("seasons") or "").strip()
    seasons = [int(s) for s in seasons_q.split(",") if s.strip().isdigit()] or [2024]
    leagues = []  # will use curated defaults
    return jsonify(_run_bulk_leagues(leagues, seasons))

@app.route("/harvest")
def harvest_route():
    _require_api_key()
    saved, live_seen = harvest_scan()
    return jsonify({"ok": True, "live_seen": live_seen, "snapshots_saved": saved})

@app.route("/harvest/league_ids")
def harvest_league_ids_route():
    _require_api_key()
    try:
        league = int(request.args.get("league"))
        season = int(request.args.get("season"))
    except Exception:
        abort(400)
    statuses_param = request.args.get("statuses", "")
    include_inplay = request.args.get("include_inplay", "0") not in ("0","false","False","no","NO")
    statuses = [s for s in statuses_param.split("-") if s] if statuses_param else ["FT","AET","PEN"]
    if include_inplay:
        for s in INPLAY_STATUSES:
            if s not in statuses:
                statuses.append(s)
    ids = get_fixture_ids_for_league(league, season, statuses)
    return jsonify({"ok": True, "league": league, "season": season, "statuses": "-".join(statuses), "count": len(ids), "ids": ids})

@app.route("/harvest/league_snapshots")
def harvest_league_snapshots_route():
    _require_api_key()
    try:
        league = int(request.args.get("league"))
        season = int(request.args.get("season"))
    except Exception:
        abort(400)
    include_inplay = request.args.get("include_inplay", "0") not in ("0","false","False","no","NO")
    statuses_param = request.args.get("statuses", "")
    extra_statuses = [s for s in statuses_param.split("-") if s] if statuses_param else None
    limit_param = request.args.get("limit")
    limit = int(limit_param) if (limit_param and str(limit_param).isdigit()) else None
    delay_ms = int(request.args.get("delay_ms", 1000))
    summary = create_synthetic_snapshots_for_league(league, season, include_inplay, extra_statuses, limit, delay_ms)
    return jsonify(summary)

# ── Backfill finals ───────────────────────────────────────────────────────────
@app.route("/backfill")
def backfill_route():
    _require_api_key()
    try:
        hours = int(request.args.get("hours", 48))
    except Exception:
        hours = 48
    hours = max(1, min(hours, 168))
    updated, checked = backfill_results_from_snapshots(hours=hours)
    return jsonify({"ok": True, "hours": hours, "checked": checked, "updated": updated})

@app.route("/backfill/all")
def backfill_all_unlabeled_route():
    _require_api_key()
    limit = request.args.get("limit")
    ids = _list_unlabeled_ids(int(limit) if (limit and str(limit).isdigit()) else None)
    updated, checked = _backfill_for_ids(ids)
    return jsonify({"ok": True, "mode": "all_unlabeled", "checked": checked, "updated": updated})

@app.route("/debug/backfill_preview")
def debug_backfill_preview():
    _require_api_key()
    hours = request.args.get("hours")
    if hours is not None:
        try:
            h = max(1, min(int(hours), 168))
        except Exception:
            abort(400)
        since = int(time.time()) - h * 3600
        with db_conn() as conn:
            rows = conn.execute("""
                SELECT DISTINCT match_id FROM tip_snapshots
                WHERE created_ts >= %s
                  AND match_id NOT IN (SELECT match_id FROM match_results)
                ORDER BY match_id
            """, (since,)).fetchall()
            ids = [r[0] for r in rows]
    else:
        ids = _list_unlabeled_ids(int(request.args.get("limit", "0")) or None)
    fx = fetch_fixtures_by_ids(ids)
    finished_status = {"FT","AET","PEN"}
    out = []
    for fid in ids:
        m = fx.get(fid)
        if not m:
            out.append({"match_id": fid, "fetched": False})
            continue
        st = ((m.get("fixture") or {}).get("status") or {}).get("short")
        gh = (m.get("goals") or {}).get("home")
        ga = (m.get("goals") or {}).get("away")
        out.append({"match_id": fid, "fetched": True, "status": st, "goals_h": gh, "goals_a": ga, "will_update": st in finished_status})
    return jsonify({"count": len(ids), "candidates": out})

# ── Prediction / Models ───────────────────────────────────────────────────────
@app.route("/predict/scan")
def predict_scan_route():
    """Manual trigger for one production scan (admin only)."""
    _require_api_key()
    saved, live = production_scan()
    return jsonify({"ok": True, "mode": "PRODUCTION", "live_seen": live, "tips_saved": saved})

@app.route("/predict/models")
def predict_models_route():
    """Inspect current models loaded from settings."""
    _require_api_key()
    o25 = load_model_from_settings("O25")
    btts = load_model_from_settings("BTTS_YES")
    return jsonify({
        "O25_found": bool(o25),
        "BTTS_YES_found": bool(btts),
        "keys_tried": MODEL_KEYS_ORDER,
    })

@app.route("/train", methods=["POST", "GET"])
def train_route():
    _require_api_key()
    return jsonify(retrain_models_job())

# ── Stats / Admin ────────────────────────────────────────────────────────────
@app.route("/stats/snapshots_count")
def snapshots_count():
    _require_api_key()
    with db_conn() as conn:
        snap = conn.execute("SELECT COUNT(*) FROM tip_snapshots").fetchone()[0]
        tips = conn.execute("SELECT COUNT(*) FROM tips").fetchone()[0]
        res  = conn.execute("SELECT COUNT(*) FROM match_results").fetchone()[0]
        unlabeled = conn.execute("""
            SELECT COUNT(DISTINCT s.match_id)
            FROM tip_snapshots s
            LEFT JOIN match_results r ON r.match_id = s.match_id
            WHERE r.match_id IS NULL
        """).fetchone()[0]
    return jsonify({"tip_snapshots": int(snap), "tips_rows": int(tips), "match_results": int(res), "unlabeled_match_ids": int(unlabeled)})

@app.route("/stats/progress")
def stats_progress():
    _require_api_key()
    now = int(time.time())
    day_ago = now - 24*3600
    week_ago = now - 7*24*3600
    c24 = _counts_since(day_ago)
    c7  = _counts_since(week_ago)
    return jsonify({
        "window_24h": {"snapshots_added": c24["snap_24h"], "results_added": c24["res_24h"]},
        "totals": {"tip_snapshots": c24["snap_total"], "tips_rows": c24["tips_total"], "match_results": c24["res_total"], "unlabeled_match_ids": c24["unlabeled"]},
        "window_7d": {"snapshots_added_est": c7["snap_24h"], "results_added_est": c7["res_24h"]},
    })

@app.route("/stats/config")
def stats_config():
    _require_api_key()
    return jsonify({
        "HARVEST_MODE": HARVEST_MODE,
        "CONF_THRESHOLD": CONF_THRESHOLD,
        "MAX_TIPS_PER_SCAN": MAX_TIPS_PER_SCAN,
        "DUP_COOLDOWN_MIN": DUP_COOLDOWN_MIN,
        "REQUIRE_STATS_MINUTE": REQUIRE_STATS_MINUTE,
        "REQUIRE_DATA_FIELDS": REQUIRE_DATA_FIELDS,
        "UNDER_SUPPRESS_AFTER_MIN": UNDER_SUPPRESS_AFTER_MIN,
        "O25_LATE_MINUTE": O25_LATE_MINUTE,
        "O25_LATE_MIN_GOALS": O25_LATE_MIN_GOALS,
        "BTTS_LATE_MINUTE": BTTS_LATE_MINUTE,
        "ONLY_MODEL_MODE": ONLY_MODEL_MODE,
        "API_KEY_set": bool(API_KEY),
        "TELEGRAM_CHAT_ID_set": bool(TELEGRAM_CHAT_ID),
        "TRAIN_ENABLE": TRAIN_ENABLE,
        "TRAIN_MIN_MINUTE": TRAIN_MIN_MINUTE,
        "TRAIN_TEST_SIZE": TRAIN_TEST_SIZE,
        "MOTD_PREDICT": MOTD_PREDICT,
        "MOTD_MIN_SAMPLES": MOTD_MIN_SAMPLES,
        "MOTD_CONF_MIN": MOTD_CONF_MIN,
        "LEAGUE_PRIORITY_IDS": LEAGUE_PRIORITY_IDS,
    })

@app.route("/debug/env")
def debug_env():
    _require_api_key()
    def mark(val): return {"set": bool(val), "len": (len(val) if val else 0)}
    return jsonify({
        "API_KEY":            mark(API_KEY),
        "ADMIN_API_KEY":      {"set": bool(ADMIN_API_KEY)},
        "TELEGRAM_BOT_TOKEN": mark(TELEGRAM_BOT_TOKEN),
        "TELEGRAM_CHAT_ID":   mark(TELEGRAM_CHAT_ID),
        "HARVEST_MODE":       HARVEST_MODE,
        "TRAIN_ENABLE":       TRAIN_ENABLE,
        "TRAIN_MIN_MINUTE":   TRAIN_MIN_MINUTE,
        "TRAIN_TEST_SIZE":    TRAIN_TEST_SIZE,
        "DB": "Postgres",
    })

# ── MOTD ─────────────────────────────────────────────────────────────────────
@app.route("/motd/preview")
def motd_preview_route():
    _require_api_key()
    limit = request.args.get("limit")
    try:
        limit = int(limit) if limit and str(limit).isdigit() else 5
    except Exception:
        limit = 5
    cands = motd_candidates(limit=limit)
    return jsonify({"ok": True, "candidates": cands, "already_sent_today": bool(_motd_already_announced_for_today())})

@app.route("/motd/announce", methods=["POST"])
def motd_announce_route():
    _require_api_key()
    res = motd_announce()
    return jsonify(res)

# ── Entrypoint / Scheduler ────────────────────────────────────────────────────
def main():
    # Init DB tables
    init_db()

    # Load configs
    mode = "HARVEST" if HARVEST_MODE else "PRODUCTION"
    logging.info(f"🤖 Robi Superbrain starting in {mode} mode.")

    scheduler = BackgroundScheduler()

    if HARVEST_MODE:
        # Harvest snapshots every 10 minutes
        scheduler.add_job(
            harvest_scan,
            CronTrigger(minute="*/10", timezone=ZoneInfo("Europe/Berlin")),
            id="harvest",
            replace_existing=True,
        )
        # Backfill final results every 15 minutes
        scheduler.add_job(
            backfill_results_from_snapshots,
            "interval",
            minutes=15,
            id="backfill",
            replace_existing=True,
        )
        logging.info("⛏️ Running in HARVEST mode.")
    else:
        # Production: generate tips every 5 minutes
        scheduler.add_job(
            production_scan,
            CronTrigger(minute="*/5", timezone=ZoneInfo("Europe/Berlin")),
            id="production_scan",
            replace_existing=True,
        )
        logging.info("🎯 Running in PRODUCTION mode.")

        # Nightly training at 03:00 Europe/Berlin
    scheduler.add_job(
        retrain_models_job,
        CronTrigger(hour=3, minute=0, timezone=ZoneInfo("Europe/Berlin")),
        id="train",
        replace_existing=True,
        misfire_grace_time=3600,
        coalesce=True,
    )

    # Nightly digest at 03:02 Europe/Berlin
    scheduler.add_job(
        nightly_digest_job,
        CronTrigger(hour=3, minute=2, timezone=ZoneInfo("Europe/Berlin")),
        id="digest",
        replace_existing=True,
        misfire_grace_time=3600,
        coalesce=True,
    )

    # Daily MOTD announce (time configurable via env; defaults to 09:00)
    if MOTD_PREDICT:
        motd_hour = int(os.getenv("MOTD_HOUR", "9"))      # default 09:00
        motd_minute = int(os.getenv("MOTD_MINUTE", "0"))  # default :00
        scheduler.add_job(
            motd_announce,
            CronTrigger(hour=motd_hour, minute=motd_minute, timezone=ZoneInfo("Europe/Berlin")),
            id="motd_announce",
            replace_existing=True,
            misfire_grace_time=3600,
            coalesce=True,
        )
        logging.info(f"🌟 MOTD daily announcement enabled at {motd_hour:02d}:{motd_minute:02d} Europe/Berlin.")

    # Start Flask app
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
