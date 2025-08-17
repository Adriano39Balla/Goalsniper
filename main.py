import os
import json
import time
import math
import logging
import requests
import sqlite3
from html import escape
from zoneinfo import ZoneInfo
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from flask import Flask, jsonify, request, abort
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
app = Flask(__name__)

# ── Env ───────────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")
API_KEY            = os.getenv("API_KEY")                  # admin endpoints
APISPORTS_KEY      = os.getenv("APISPORTS_KEY")            # API-Football
PUBLIC_BASE_URL    = os.getenv("PUBLIC_BASE_URL")
BET_URL_TMPL       = os.getenv("BET_URL")
WATCH_URL_TMPL     = os.getenv("WATCH_URL")
WEBHOOK_SECRET     = os.getenv("TELEGRAM_WEBHOOK_SECRET")
HEARTBEAT_ENABLE   = os.getenv("HEARTBEAT_ENABLE", "1") not in ("0","false","False","no","NO")

# Mode switch: Harvest (Sun–Thu) vs Production (Fri–Sat)
HARVEST_MODE       = os.getenv("HARVEST_MODE", "1") not in ("0","false","False","no","NO")

# Core knobs
CONF_THRESHOLD     = int(os.getenv("CONF_THRESHOLD", "60"))     # tip gate
MAX_TIPS_PER_SCAN  = int(os.getenv("MAX_TIPS_PER_SCAN", "25"))  # 0 = unlimited
DUP_COOLDOWN_MIN   = int(os.getenv("DUP_COOLDOWN_MIN", "20"))   # anti-spam

# Dynamic volume guard (safe to keep on)
DYN_ENABLE          = os.getenv("DYNAMIC_ENABLE", "1") not in ("0","false","False","no","NO")
DYN_BAND_STR        = os.getenv("DYNAMIC_TARGET_BAND", "2,15")
try:
    DYN_MIN, DYN_MAX = [int(x) for x in DYN_BAND_STR.split(",")]
except Exception:
    DYN_MIN, DYN_MAX = 2, 15
DYN_TARGET_HITRATE  = float(os.getenv("DYN_TARGET_HITRATE", "0.56"))

# Late windows / guards
O25_LATE_MINUTE     = int(os.getenv("O25_LATE_MINUTE", "88"))
O25_LATE_MIN_GOALS  = int(os.getenv("O25_LATE_MIN_GOALS", "2"))
BTTS_LATE_MINUTE    = int(os.getenv("BTTS_LATE_MINUTE", "88"))
UNDER_SUPPRESS_AFTER_MIN = int(os.getenv("UNDER_SUPPRESS_AFTER_MIN", "82"))

# Data sufficiency
ONLY_MODEL_MODE          = os.getenv("ONLY_MODEL_MODE", "0") not in ("0","false","False","no","NO")
REQUIRE_STATS_MINUTE     = int(os.getenv("REQUIRE_STATS_MINUTE", "35"))
REQUIRE_DATA_FIELDS      = int(os.getenv("REQUIRE_DATA_FIELDS", "2"))   # need ≥2 among xG,SOT,CK,POS

# MOTD
LEAGUE_PRIORITY_IDS = [int(x) for x in (os.getenv("MOTD_LEAGUE_IDS", "39,140,135,78,61,2").split(",")) if x.strip().isdigit()]
MOTD_PREDICT        = os.getenv("MOTD_PREDICT", "1") not in ("0","false","False","no","NO")
MOTD_MIN_SAMPLES    = int(os.getenv("MOTD_MIN_SAMPLES", "30"))
MOTD_CONF_MIN       = int(os.getenv("MOTD_CONF_MIN", "65"))

# Allowed tips (no Over 1.5)
ALLOWED_SUGGESTIONS = {"Over 2.5 Goals", "Under 2.5 Goals", "BTTS: Yes", "BTTS: No"}

# ── External APIs ─────────────────────────────────────────────────────────────
BASE_URL         = "https://v3.football.api-sports.io"
FOOTBALL_API_URL = f"{BASE_URL}/fixtures"
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
HEADERS          = {"x-apisports-key": APISPORTS_KEY, "Accept": "application/json"}

# ── HTTP session ──────────────────────────────────────────────────────────────
session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))

# ── In-memory caches ──────────────────────────────────────────────────────────
STATS_CACHE: Dict[int, Tuple[float, list]] = {}  # fixture_id -> (ts, stats)
CAL_CACHE: Dict[str, Any] = {"ts": 0, "bins": []}

# ── DB ────────────────────────────────────────────────────────────────────────
DB_PATH = "tip_performance.db"
def db_conn(): return sqlite3.connect(DB_PATH)

def init_db():
    with db_conn() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS tips (
            match_id INTEGER,
            league_id INTEGER,
            league   TEXT,
            home     TEXT,
            away     TEXT,
            market   TEXT,
            suggestion TEXT,
            confidence REAL,
            score_at_tip TEXT,
            minute    INTEGER,
            created_ts INTEGER,
            sent_ok   INTEGER DEFAULT 1,
            PRIMARY KEY (match_id, created_ts)
        )""")
        try: conn.execute("ALTER TABLE tips ADD COLUMN sent_ok INTEGER DEFAULT 1")
        except Exception: pass

        conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id INTEGER UNIQUE,
            verdict  INTEGER CHECK (verdict IN (0,1)),
            created_ts INTEGER
        )""")

        conn.execute("""
        CREATE TABLE IF NOT EXISTS tip_snapshots (
            match_id INTEGER,
            created_ts INTEGER,
            payload TEXT,
            PRIMARY KEY (match_id, created_ts)
        )""")

        conn.execute("""
        CREATE TABLE IF NOT EXISTS match_results (
            match_id   INTEGER PRIMARY KEY,
            final_goals_h INTEGER,
            final_goals_a INTEGER,
            btts_yes      INTEGER,
            updated_ts    INTEGER
        )""")

        conn.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )""")

        conn.execute("DROP VIEW IF EXISTS v_tip_stats")
        conn.execute("""
        CREATE VIEW v_tip_stats AS
        SELECT t.market, t.suggestion,
               AVG(f.verdict) AS hit_rate,
               COUNT(DISTINCT t.match_id) AS n
        FROM (
          SELECT match_id, market, suggestion, MAX(created_ts) AS last_ts
          FROM tips GROUP BY match_id, market, suggestion
        ) lt
        JOIN tips t ON t.match_id=lt.match_id AND t.created_ts=lt.last_ts
        JOIN feedback f ON f.match_id = t.match_id
        GROUP BY t.market, t.suggestion
        """)
        conn.commit()

def set_setting(key: str, value: str):
    with db_conn() as conn:
        conn.execute(
            "INSERT INTO settings(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        conn.commit()

def get_setting(key: str, default: Optional[str] = None) -> Optional[str]:
    with db_conn() as conn:
        cur = conn.execute("SELECT value FROM settings WHERE key=?", (key,))
        row = cur.fetchone()
        return row[0] if row else default

def save_tip(match_id: int, league_id: int, league: str, home: str, away: str,
             market: str, suggestion: str, confidence: float,
             score_at_tip: str, minute: int, sent_ok: int = 1):
    now = int(time.time())
    with db_conn() as conn:
        conn.execute("""
          INSERT INTO tips(match_id,league_id,league,home,away,market,suggestion,confidence,score_at_tip,minute,created_ts,sent_ok)
          VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, (match_id, league_id, league, home, away, market, suggestion, confidence, score_at_tip, minute, now, sent_ok))
        conn.commit()

def upsert_feedback(match_id: int, verdict: int, ts: Optional[int] = None):
    if ts is None: ts = int(time.time())
    with db_conn() as conn:
        conn.execute("""
          INSERT INTO feedback(match_id, verdict, created_ts)
          VALUES (?,?,?)
          ON CONFLICT(match_id) DO UPDATE SET verdict=excluded.verdict, created_ts=excluded.created_ts
        """, (match_id, verdict, ts))
        conn.commit()

def feedback_exists(match_id: int) -> bool:
    with db_conn() as conn:
        cur = conn.execute("SELECT 1 FROM feedback WHERE match_id=? LIMIT 1", (match_id,))
        return cur.fetchone() is not None

def get_historic_hit_rate(market: str, suggestion: str) -> Tuple[float, int]:
    with db_conn() as conn:
        cur = conn.execute(
            "SELECT COALESCE(hit_rate,0.5), COALESCE(n,0) FROM v_tip_stats WHERE market=? AND suggestion=?",
            (market, suggestion),
        )
        row = cur.fetchone()
        return (row[0], row[1]) if row else (0.5, 0)

def recent_tip_exists(match_id: int, cooldown_min: int) -> bool:
    cutoff = int(time.time()) - cooldown_min * 60
    with db_conn() as conn:
        cur = conn.execute("SELECT 1 FROM tips WHERE match_id=? AND created_ts>=? LIMIT 1", (match_id, cutoff))
        return cur.fetchone() is not None

# ── Telegram ──────────────────────────────────────────────────────────────────
def send_telegram(message: str, inline_keyboard: Optional[list] = None) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logging.error("Missing Telegram credentials")
        return False
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    if inline_keyboard:
        payload["reply_markup"] = json.dumps({"inline_keyboard": inline_keyboard})
    try:
        res = session.post(f"{TELEGRAM_API_URL}/sendMessage", data=payload, timeout=10)
        if not res.ok:
            logging.error(f"[Telegram] sendMessage FAILED status={res.status_code} body={res.text}")
        else:
            logging.info("[Telegram] sendMessage OK")
        return res.ok
    except Exception as e:
        logging.exception(f"[Telegram] Exception: {e}")
        return False

def answer_callback(callback_id: str, text: str):
    try:
        session.post(f"{TELEGRAM_API_URL}/answerCallbackQuery",
                     data={"callback_query_id": callback_id, "text": text, "show_alert": False}, timeout=10)
    except Exception:
        pass

# ── API helpers ───────────────────────────────────────────────────────────────
def _api_get(url: str, params: dict, timeout: int = 15):
    try:
        res = session.get(url, headers=HEADERS, params=params, timeout=timeout)
        if not res.ok:
            logging.error(f"[API] {url} status={res.status_code} body={res.text[:300]}")
            return None
        return res.json()
    except Exception as e:
        logging.exception(f"[API] error {url}: {e}")
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

def fetch_live_matches() -> List[Dict[str, Any]]:
    js = _api_get(FOOTBALL_API_URL, {"live": "all"})
    if not isinstance(js, dict): return []
    matches = js.get("response", []) or []
    filtered = []
    for m in matches:
        status = (m.get("fixture", {}) or {}).get("status", {}) or {}
        elapsed = status.get("elapsed")
        if elapsed is None or elapsed > 90:
            continue
        fid = (m.get("fixture", {}) or {}).get("id")
        try:
            stats = fetch_match_stats(fid)
            m["statistics"] = stats or []
        except Exception:
            m["statistics"] = []
        filtered.append(m)
    logging.info(f"[FETCH] live={len(matches)} kept={len(filtered)}")
    return filtered

def fetch_today_fixtures_utc() -> List[Dict[str, Any]]:
    today_utc = datetime.now(timezone.utc).date()
    js = _api_get(FOOTBALL_API_URL, {"date": today_utc.isoformat(), "timezone": "UTC"})
    return js.get("response", []) if isinstance(js, dict) else []

def fetch_fixtures_by_ids(ids: List[int]) -> Dict[int, Dict[str, Any]]:
    if not ids: return {}
    js = _api_get(FOOTBALL_API_URL, {"ids": ",".join(str(i) for i in ids), "timezone": "UTC"})
    resp = js.get("response", []) if isinstance(js, dict) else []
    return { (fx.get("fixture") or {}).get("id"): fx for fx in resp }

# ── Calibration (works after feedback) ────────────────────────────────────────
def compute_calibration_bins(cache_secs: int = 600):
    now = time.time()
    if now - CAL_CACHE["ts"] < cache_secs and CAL_CACHE["bins"]:
        return CAL_CACHE["bins"]
    bins = [(50,54),(55,59),(60,64),(65,69),(70,74),(75,79),(80,84),(85,100)]
    results = []
    with db_conn() as conn:
        for lo, hi in bins:
            cur = conn.execute("""
              SELECT COUNT(*), AVG(f.verdict)
              FROM tips t JOIN feedback f ON f.match_id = t.match_id
              WHERE t.confidence BETWEEN ? AND ?
            """, (lo, hi))
            n, hr = cur.fetchone()
            hr = hr if hr is not None else 0.5
            results.append({"lo": lo, "hi": hi, "n": n or 0, "hit": float(hr)})
    CAL_CACHE.update({"ts": now, "bins": results})
    return results

def calibrate_confidence(pct: float) -> float:
    bins = compute_calibration_bins()
    for b in bins:
        if b["lo"] <= pct <= b["hi"]:
            if b["n"] >= 30:
                return round(b["hit"] * 100.0)
            break
    return pct

# ── Model support ─────────────────────────────────────────────────────────────
def _load_model():
    with db_conn() as conn:
        cur = conn.execute("SELECT value FROM settings WHERE key='model_coeffs'")
        row = cur.fetchone()
        if not row:
            return {}
    try:
        return json.loads(row[0])
    except Exception:
        logging.exception("[MODEL] Failed to load model_coeffs JSON")
        return {}

def _sigmoid(z: float) -> float:
    try: return 1.0 / (1.0 + math.exp(-z))
    except OverflowError: return 0.0 if z < 0 else 1.0

def _num(v) -> float:
    try:
        if isinstance(v, str) and v.endswith('%'):
            return float(v[:-1])
        return float(v or 0)
    except Exception:
        return 0.0

# ── Feature extraction ────────────────────────────────────────────────────────
def extract_features(match: Dict[str, Any]) -> Dict[str, float]:
    home_name = match["teams"]["home"]["name"]
    away_name = match["teams"]["away"]["name"]
    gh = match["goals"]["home"] or 0
    ga = match["goals"]["away"] or 0
    minute = int((match.get("fixture", {}).get("status", {}) or {}).get("elapsed") or 0)

    stats_blocks = match.get("statistics") or []
    stats: Dict[str, Dict[str, Any]] = {}
    for s in stats_blocks:
        tname = (s.get("team") or {}).get("name")
        if tname:
            stats[tname] = {i["type"]: i["value"] for i in (s.get("statistics") or [])}

    sh = stats.get(home_name, {}); sa = stats.get(away_name, {})

    def pct(v):
        try: return float(str(v).replace('%', '').strip() or 0)
        except Exception: return 0.0

    xg_h = _num(sh.get("Expected Goals", 0));       xg_a = _num(sa.get("Expected Goals", 0))
    sot_h = _num(sh.get("Shots on Target", 0));     sot_a = _num(sa.get("Shots on Target", 0))
    cor_h = _num(sh.get("Corner Kicks", 0));        cor_a = _num(sa.get("Corner Kicks", 0))
    pos_h = pct(sh.get("Ball Possession", 0));      pos_a = pct(sa.get("Ball Possession", 0))
    red_h = _num(sh.get("Red Cards", 0));           red_a = _num(sa.get("Red Cards", 0))

    return {
        "minute": float(minute),
        "goals_h": float(gh), "goals_a": float(ga),
        "goals_sum": float(gh + ga), "goals_diff": float(gh - ga),
        "xg_h": float(xg_h), "xg_a": float(xg_a), "xg_sum": float(xg_h + xg_a), "xg_diff": float(xg_h - xg_a),
        "sot_h": float(sot_h), "sot_a": float(sot_a), "sot_sum": float(sot_h + sot_a),
        "cor_h": float(cor_h), "cor_a": float(cor_a), "cor_sum": float(cor_h + cor_a),
        "pos_h": float(pos_h), "pos_a": float(pos_a), "pos_diff": float(pos_h - pos_a),
        "red_h": float(red_h), "red_a": float(red_a), "red_sum": float(red_h + red_a),
    }

# ── Sufficiency / triggers ────────────────────────────────────────────────────
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

def has_intensity_trigger(feat: Dict[str, float]) -> bool:
    return (feat.get("xg_sum", 0.0) >= 0.6) or (feat.get("sot_sum", 0.0) >= 3.0) or (feat.get("cor_sum", 0.0) >= 6.0)

# ── Predict (model-first) ─────────────────────────────────────────────────────
def predict_with_model(model: Dict[str, Any], feat: Dict[str, float]) -> Optional[float]:
    try:
        names = model["features"]; coef = model["coef"]; w0 = float(model.get("intercept", 0.0))
        z = w0 + sum(float(coef[i]) * float(feat.get(names[i], 0.0)) for i in range(len(names)))
        return _sigmoid(z)
    except Exception:
        logging.exception("[MODEL] Prediction error")
        return None

def decide_market(match: Dict[str, Any]) -> Tuple[str, str, float, Dict[str, float]]:
    feat = extract_features(match)
    minute = int(feat["minute"])
    gh = int(feat["goals_h"]); ga = int(feat["goals_a"])
    total_goals = gh + ga

    # hard stops
    if minute >= 90:
        return "Over/Under 2.5 Goals", "Under 2.5 Goals", 0.0, feat
    if minute >= 30 and feat["xg_sum"] == 0 and feat["sot_sum"] == 0 and feat["cor_sum"] == 0:
        return "Over/Under 2.5 Goals", "Over 2.5 Goals", 0.0, feat
    if not stats_coverage_ok(feat, minute):
        return "Over/Under 2.5 Goals", "Over 2.5 Goals", 0.0, feat
    if minute >= 15 and not has_intensity_trigger(feat):
        return "Over/Under 2.5 Goals", "Over 2.5 Goals", 0.0, feat

    models = _load_model()
    effective_model_only = (ONLY_MODEL_MODE or (not HARVEST_MODE))
    if effective_model_only and not models:
        # Weekend/production without a model → do not send (still snapshot upstream)
        logging.warning("[MODEL] Model-only mode active but no model loaded")
        return "Over/Under 2.5 Goals", "Over 2.5 Goals", 0.0, feat

    p_o25 = predict_with_model(models.get("O25", {}), feat) if models.get("O25") else None
    p_btts = predict_with_model(models.get("BTTS_YES", {}), feat) if models.get("BTTS_YES") else None

    candidates: List[Tuple[str, str, float]] = []
    if p_o25 is not None:
        candidates.append(("Over/Under 2.5 Goals", "Over 2.5 Goals", p_o25))
        candidates.append(("Over/Under 2.5 Goals", "Under 2.5 Goals", 1.0 - p_o25))
    if p_btts is not None:
        candidates.append(("BTTS", "BTTS: Yes", p_btts))
        candidates.append(("BTTS", "BTTS: No", 1.0 - p_btts))

    if not candidates:
        # Heuristic fallback allowed only in Harvest (so we still learn coverage / volume)
        if not HARVEST_MODE:
            return "Over/Under 2.5 Goals", "Over 2.5 Goals", 0.0, feat
        score = 50.0
        if feat["xg_sum"] >= 2.4 and total_goals <= 2 and minute >= 25: score += 18
        if feat["sot_sum"] >= 6: score += 10
        if feat["cor_sum"] >= 10: score += 5
        if minute < 20: score -= 8
        if minute >= 25 and feat["xg_h"] >= 0.9 and feat["xg_a"] >= 0.9 and feat["sot_h"] >= 3 and feat["sot_a"] >= 3 and total_goals <= 3:
            market, suggestion, base = "BTTS", "BTTS: Yes", max(score, 62)
        elif feat["xg_sum"] < 1.4 and feat["sot_sum"] < 4 and total_goals == 0 and minute > 35:
            market, suggestion, base = "BTTS", "BTTS: No", 62
        elif minute > 70 and total_goals <= 1:
            market, suggestion, base = "Over/Under 2.5 Goals", "Under 2.5 Goals", max(score, 62)
        else:
            market, suggestion, base = "Over/Under 2.5 Goals", "Over 2.5 Goals", score
        base = max(35.0, min(90.0, base))
        return market, suggestion, int(round(base)), feat

    allowed = [(m, s, float(p)) for (m, s, p) in candidates if s in ALLOWED_SUGGESTIONS]
    if not allowed:
        return "Over/Under 2.5 Goals", "Over 2.5 Goals", 0.0, feat
    market, suggestion, prob = max(allowed, key=lambda x: x[2])
    raw_pct = int(round(min(95.0, max(35.0, prob * 100.0))))
    return market, suggestion, raw_pct, feat

# ── Guards ────────────────────────────────────────────────────────────────────
def is_settled_or_impossible(suggestion: str, gh: int, ga: int) -> bool:
    total = gh + ga
    both_scored = (gh > 0 and ga > 0)
    if suggestion == "Over 2.5 Goals":  return total >= 3
    if suggestion == "Under 2.5 Goals": return total >= 3
    if suggestion == "BTTS: Yes":       return both_scored
    if suggestion == "BTTS: No":        return both_scored
    return False

def is_too_late(suggestion: str, minute: int, gh: int, ga: int, feat: Optional[Dict[str,float]] = None) -> bool:
    feat = feat or {}
    total = gh + ga
    xg_sum = feat.get("xg_sum", 0.0)
    sot_sum = feat.get("sot_sum", 0.0)
    cor_sum = feat.get("cor_sum", 0.0)
    red_sum = feat.get("red_sum", 0.0)

    # settled/impossible
    if suggestion == "Over 2.5 Goals" and total >= 3:  return True
    if suggestion == "Under 2.5 Goals" and total >= 3: return True
    both_scored = (gh > 0 and ga > 0)
    if suggestion == "BTTS: Yes" and both_scored:      return True
    if suggestion == "BTTS: No"  and both_scored:      return True

    # Under brakes
    if suggestion == "Under 2.5 Goals":
        if minute >= 80 and red_sum >= 1:  return True
        if minute >= 80 and (xg_sum >= 1.0 or sot_sum >= 2 or cor_sum >= 6): return True
        if minute >= 88:
            if not (total <= 1 and xg_sum < 0.8 and sot_sum <= 1 and cor_sum <= 4 and red_sum == 0):
                return True

    # BTTS late guard
    if suggestion == "BTTS: Yes" and minute >= 85:
        if (gh == 0 and feat.get("sot_h",0) == 0) or (ga == 0 and feat.get("sot_a",0) == 0):
            return True

    if suggestion == "Over 2.5 Goals" and minute >= O25_LATE_MINUTE and total < O25_LATE_MIN_GOALS:
        return True
    if suggestion == "BTTS: Yes" and minute >= BTTS_LATE_MINUTE and (gh == 0 or ga == 0):
        return True
    return False

# ── Snapshots ─────────────────────────────────────────────────────────────────
def save_snapshot(match: Dict[str, Any], feat: Dict[str, float]):
    fx = match.get("fixture", {}) or {}
    lg = match.get("league", {}) or {}
    teams = match.get("teams", {}) or {}
    goals = match.get("goals", {}) or {}
    payload = {
        "minute": int(feat.get("minute", 0)),
        "gh": int(goals.get("home") or 0),
        "ga": int(goals.get("away") or 0),
        "league_id": int(lg.get("id") or 0),
        "home": teams.get("home", {}).get("name",""),
        "away": teams.get("away", {}).get("name",""),
        "stat": {
            "xg_h": feat.get("xg_h",0), "xg_a": feat.get("xg_a",0),
            "sot_h": feat.get("sot_h",0), "sot_a": feat.get("sot_a",0),
            "cor_h": feat.get("cor_h",0), "cor_a": feat.get("cor_a",0),
            "pos_h": feat.get("pos_h",0), "pos_a": feat.get("pos_a",0),
            "red_h": feat.get("red_h",0), "red_a": feat.get("red_a",0),
        }
    }
    try:
        with db_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO tip_snapshots(match_id, created_ts, payload) VALUES (?,?,?)",
                (int(fx.get("id")), int(time.time()), json.dumps(payload)[:200000])
            )
            conn.commit()
    except Exception:
        logging.exception("[SNAPSHOT] failed")

# ── Messaging ─────────────────────────────────────────────────────────────────
def build_tip_message(match: Dict[str, Any],
                      chosen: Optional[Tuple[str,str,float,Dict[str,float]]] = None) -> Tuple[str, list, Dict[str, Any]]:
    fixture = match["fixture"]; league_block = match.get("league") or {}
    league_id = league_block.get("id") or 0
    league = escape(f"{league_block.get('country','')} - {league_block.get('name','')}".strip(" -"))
    minute = fixture["status"].get("elapsed") or 0
    match_id = fixture["id"]
    home = escape(match["teams"]["home"]["name"]); away = escape(match["teams"]["away"]["name"])
    g_home = match["goals"]["home"] or 0; g_away = match["goals"]["away"] or 0

    market, suggestion, confidence, stat = chosen if chosen else decide_market(match)

    if suggestion not in ALLOWED_SUGGESTIONS or confidence < CONF_THRESHOLD:
        return "", [], {"skip": True}
    if is_settled_or_impossible(suggestion, g_home, g_away) or is_too_late(suggestion, int(minute), g_home, g_away, stat):
        return "", [], {"skip": True}

    stat_line = ""
    if any([stat.get("xg_h",0), stat.get("xg_a",0), stat.get("sot_h",0), stat.get("sot_a",0), stat.get("cor_h",0), stat.get("cor_a",0), stat.get("pos_h",0), stat.get("pos_a",0), stat.get("red_h",0), stat.get("red_a",0)]):
        stat_line = (
            f"\n📊 xG H {stat.get('xg_h',0):.2f} / A {stat.get('xg_a',0):.2f}"
            f" • SOT {int(stat.get('sot_h',0))}–{int(stat.get('sot_a',0))}"
            f" • CK {int(stat.get('cor_h',0))}–{int(stat.get('cor_a',0))}"
        )
        if stat.get("pos_h",0) or stat.get("pos_a",0):
            stat_line += f" • POS {int(stat.get('pos_h',0))}%–{int(stat.get('pos_a',0))}%"
        if stat.get("red_h",0) or stat.get("red_a",0):
            stat_line += f" • RED {int(stat.get('red_h',0))}–{int(stat.get('red_a',0))}"

    lines = [
        "⚽️ <b>New Tip!</b>",
        f"<b>Match:</b> {home} vs {away}",
        f"⏱ <b>Minute:</b> {minute}'  |  <b>Score:</b> {g_home}–{g_away}",
        f"<b>Tip:</b> {escape(suggestion)}",
        f"📈 <b>Confidence:</b> {int(confidence)}%",
        f"🏆 <b>League:</b> {league}{stat_line}",
    ]
    msg = "\n".join(lines)

    kb = [[
        {"text": "👍 Correct", "callback_data": json.dumps({"t": "correct", "id": match_id})},
        {"text": "👎 Wrong",   "callback_data": json.dumps({"t": "wrong",   "id": match_id})},
    ]]
    row = []
    if BET_URL_TMPL:
        try: row.append({"text": "💰 Bet Now", "url": BET_URL_TMPL.format(home=home, away=away, fixture_id=match_id)})
        except Exception: pass
    if WATCH_URL_TMPL:
        try: row.append({"text": "📺 Watch Match", "url": WATCH_URL_TMPL.format(home=home, away=away, fixture_id=match_id)})
        except Exception: pass
    if row: kb.append(row)

    meta = {
        "match_id": match_id, "league_id": int(league_id), "league": league,
        "home": home, "away": away, "confidence": int(confidence),
        "score_at_tip": f"{g_home}-{g_away}", "minute": int(minute)
    }
    return msg, kb, meta

# ── Orchestration ─────────────────────────────────────────────────────────────
last_heartbeat = 0
def maybe_send_heartbeat():
    global last_heartbeat
    if HEARTBEAT_ENABLE and time.time() - last_heartbeat > 1200:
        send_telegram("✅ Robi Superbrain is online and scanning…")
        last_heartbeat = time.time()

def match_alert():
    logging.info("🔍 Scanning live matches…")
    matches = fetch_live_matches()
    if not matches:
        logging.warning("[SCAN] No matches returned – check APISPORTS_KEY/plan or time of day.")
        maybe_send_heartbeat()
        return

    sent = 0
    for match in matches:
        if MAX_TIPS_PER_SCAN and sent >= MAX_TIPS_PER_SCAN:
            break

        fid = (match.get("fixture", {}) or {}).get("id")

        # snapshot ALWAYS (training)
        feat_snap = extract_features(match)
        save_snapshot(match, feat_snap)

        if recent_tip_exists(fid, DUP_COOLDOWN_MIN):
            continue

        market, suggestion, conf, feat = decide_market(match)
        gh = match["goals"]["home"] or 0; ga = match["goals"]["away"] or 0
        minute = int((match.get("fixture", {}) or {}).get("status", {}).get("elapsed") or 0)

        if suggestion not in ALLOWED_SUGGESTIONS or conf < CONF_THRESHOLD:
            continue
        if is_settled_or_impossible(suggestion, gh, ga) or is_too_late(suggestion, minute, gh, ga, feat):
            continue

        msg, kb, meta = build_tip_message(match, (market, suggestion, conf, feat))
        if not msg:
            continue
        logging.info(f"[TIP] sending fixture={fid} conf={meta['confidence']} suggestion={suggestion}")
        ok = send_telegram(msg, kb)
        save_tip(meta["match_id"], int(meta["league_id"]), meta["league"], meta["home"], meta["away"],
                 market, suggestion, float(meta["confidence"]), meta["score_at_tip"], int(meta["minute"]),
                 sent_ok=1 if ok else 0)
        if ok:
            sent += 1

    logging.info(f"[TIP] Sent={sent}")
    maybe_send_heartbeat()

def autolabel_unscored():
    since = int(time.time()) - 36*3600
    with db_conn() as conn:
        cur = conn.execute("""
          SELECT DISTINCT match_id FROM tip_snapshots
          WHERE created_ts>=? AND match_id NOT IN (SELECT match_id FROM match_results)
        """, (since,))
        ids = [r[0] for r in cur.fetchall()]
    if not ids: return
    for i in range(0, len(ids), 20):
        batch = ids[i:i+20]
        fx = fetch_fixtures_by_ids(batch)
        for fid in batch:
            m = fx.get(fid)
            if not m: continue
            status = ((m.get("fixture") or {}).get("status") or {}).get("short")
            if status not in ("FT","AET","PEN","PST","CANC","ABD","AWD","WO"):
                continue
            gh = int((m.get("goals") or {}).get("home") or 0)
            ga = int((m.get("goals") or {}).get("away") or 0)
            with db_conn() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO match_results(match_id, final_goals_h, final_goals_a, btts_yes, updated_ts)
                    VALUES (?,?,?,?,?)
                """, (int(fid), gh, ga, 1 if (gh>0 and ga>0) else 0, int(time.time())))
                conn.commit()
    logging.info(f"[AUTO] labeled_up_to={len(ids)}")

# ── MOTD (simple card; lean can be added later) ───────────────────────────────
def pick_match_of_the_day(fixtures: List[Dict[str,Any]]) -> Optional[Dict[str,Any]]:
    fixtures = [f for f in fixtures if (f.get("fixture") or {}).get("status",{}).get("short") in ("NS","TBD")]
    if not fixtures: return None
    pri = {lid:i for i,lid in enumerate(LEAGUE_PRIORITY_IDS)}
    fixtures.sort(key=lambda f: (pri.get((f.get("league") or {}).get("id"), len(LEAGUE_PRIORITY_IDS)+1),
                                 (f.get("fixture") or {}).get("timestamp") or 10**12))
    return fixtures[0]

def send_match_of_the_day():
    fixtures = fetch_today_fixtures_utc()
    motd = pick_match_of_the_day(fixtures)
    if not motd:
        logging.info("[MOTD] No suitable fixture found for today.")
        return
    lg = motd.get("league") or {}; fx = motd.get("fixture") or {}
    home = escape((motd.get("teams") or {}).get("home",{}).get("name",""))
    away = escape((motd.get("teams") or {}).get("away",{}).get("name",""))
    league = escape(f"{lg.get('country','')} - {lg.get('name','')}".strip(" -"))
    ts = fx.get("timestamp"); when = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC") if ts else "TBD"
    fid = fx.get("id")
    lines = [
        "🌟 <b>Match of the Day</b>",
        f"<b>Match:</b> {home} vs {away}",
        f"🗓 <b>Kickoff:</b> {when}",
        f"🏆 <b>League:</b> {league}",
    ]
    send_telegram("\n".join(lines), [])
    logging.info(f"[MOTD] Sent for fixture={fid}")

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return "🤖 Robi Superbrain is active and learning."

@app.route("/match-alert")
def manual_match_alert():
    match_alert()
    return jsonify({"status": "ok"})

@app.route("/motd")
def motd_route():
    send_match_of_the_day()
    return jsonify({"status": "ok"})

@app.route("/debug/scan")
def debug_scan():
    data = fetch_live_matches()
    return jsonify({
        "count": len(data),
        "fixture_ids": [ (m.get("fixture",{}) or {}).get("id") for m in data ],
        "has_stats": [ bool(m.get("statistics")) for m in data ],
    })

@app.route("/stats/snapshots_count")
def snapshots_count():
    with db_conn() as conn:
        cur = conn.execute("SELECT COUNT(*) FROM tip_snapshots")
        n = cur.fetchone()[0]
    return jsonify({"tip_snapshots": n})

# Telegram webhook
@app.route("/telegram-webhook", methods=["POST"])
def telegram_webhook():
    if WEBHOOK_SECRET and request.headers.get("X-Telegram-Bot-Api-Secret-Token") != WEBHOOK_SECRET:
        return "forbidden", 403
    update = request.get_json(force=True, silent=True) or {}
    cq = update.get("callback_query")
    if not cq:
        return "ok"
    try:
        payload = json.loads(cq.get("data") or "{}")
        verdict = 1 if payload.get("t") == "correct" else 0
        match_id = int(payload.get("id"))
        upsert_feedback(match_id, verdict)
        answer_callback(cq["id"], "Thanks! Feedback recorded ✅" if verdict else "Got it. Marked as wrong ❌")
    except Exception:
        logging.exception("webhook error")
        try: answer_callback(cq.get("id",""), "Sorry, couldn’t record feedback.")
        except Exception: pass
    return "ok"

# Admin: config
def _require_api_key():
    key = request.headers.get("X-API-Key") or request.args.get("key")
    if not API_KEY or key != API_KEY:
        abort(401)

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
        "APISPORTS_KEY_set": bool(APISPORTS_KEY),
        "TELEGRAM_CHAT_ID_set": bool(TELEGRAM_CHAT_ID),
    })

# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not APISPORTS_KEY:
        logging.error("APISPORTS_KEY is not set — live fetch will return 0 matches.")
    init_db()
    scheduler = BackgroundScheduler()

     # Scanning windows (Europe/Berlin). Night pause saves credits.
if HARVEST_MODE:
    # Busy-window harvest: every 2 minutes, 09:00–21:59 Berlin, Sun→Thu
    scheduler.add_job(
        match_alert,
        CronTrigger(day_of_week="sun,mon,tue,wed,thu",
                    hour="9-21", minute="*/2",
                    timezone=ZoneInfo("Europe/Berlin")),
        id="scan", replace_existing=True
    )
else:
    # Production: every 5 minutes, 07:00–21:59 Berlin, Fri/Sat
    scheduler.add_job(
        match_alert,
        CronTrigger(day_of_week="fri,sat",
                    hour="7-21", minute="*/5",
                    timezone=ZoneInfo("Europe/Berlin")),
        id="scan", replace_existing=True
    )
    # If you also want weekdays in production, uncomment this:
    # scheduler.add_job(
    #     match_alert,
    #     CronTrigger(day_of_week="mon,tue,wed,thu",
    #                 hour="7-21", minute="*/5",
    #                 timezone=ZoneInfo("Europe/Berlin")),
    #     id="scan_week", replace_existing=True
    # )

    # MOTD 08:00 UTC
    scheduler.add_job(
        send_match_of_the_day, CronTrigger(hour=8, minute=0, timezone="UTC"),
        id="motd", replace_existing=True, misfire_grace_time=3600, coalesce=True
    )

    # Auto-label every 10 minutes
    scheduler.add_job(autolabel_unscored, "interval", minutes=10, id="autolabel", replace_existing=True)

    scheduler.start()
    logging.info("⏱️ Scheduler started (harvest=%s)", HARVEST_MODE)
    port = int(os.getenv("PORT", 5000))
    logging.info("✅ Robi Superbrain started.")
    app.run(host="0.0.0.0", port=port)
