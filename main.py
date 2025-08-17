import os
import json
import time
import math
import logging
import requests
import sqlite3
from html import escape
from datetime import datetime, timezone, date
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
API_KEY            = os.getenv("API_KEY")                 # admin endpoints
APISPORTS_KEY      = os.getenv("APISPORTS_KEY")           # API-Football (no fallback)
PUBLIC_BASE_URL    = os.getenv("PUBLIC_BASE_URL")
BET_URL_TMPL       = os.getenv("BET_URL")
WATCH_URL_TMPL     = os.getenv("WATCH_URL")
WEBHOOK_SECRET     = os.getenv("TELEGRAM_WEBHOOK_SECRET")
HEARTBEAT_ENABLE   = os.getenv("HEARTBEAT_ENABLE", "0") not in ("0","false","False","no","NO")

# Core knobs
CONF_THRESHOLD     = int(os.getenv("CONF_THRESHOLD", "60"))     # send only if >= 60%
MAX_TIPS_PER_SCAN  = int(os.getenv("MAX_TIPS_PER_SCAN", "25"))  # soft cap per scan (set 0 to disable)
DUP_COOLDOWN_MIN   = int(os.getenv("DUP_COOLDOWN_MIN", "20"))   # don't resend same fixture within N min

# Late windows (per-market)
O25_LATE_MINUTE     = int(os.getenv("O25_LATE_MINUTE", "88"))
O25_LATE_MIN_GOALS  = int(os.getenv("O25_LATE_MIN_GOALS", "2"))
BTTS_LATE_MINUTE    = int(os.getenv("BTTS_LATE_MINUTE", "88"))

# === Data sufficiency & anti-trivial guards (tightened) ===
ONLY_MODEL_MODE          = os.getenv("ONLY_MODEL_MODE", "0") not in ("0","false","False","no","NO")
REQUIRE_STATS_MINUTE     = int(os.getenv("REQUIRE_STATS_MINUTE", "35"))   # stricter: start enforcing at 35'
REQUIRE_DATA_FIELDS      = int(os.getenv("REQUIRE_DATA_FIELDS", "2"))     # need at least two of {xG,SOT,CK}
UNDER_SUPPRESS_AFTER_MIN = int(os.getenv("UNDER_SUPPRESS_AFTER_MIN", "82"))

# MOTD
LEAGUE_PRIORITY_IDS = [int(x) for x in (os.getenv("MOTD_LEAGUE_IDS", "39,140,135,78,61,2").split(",")) if x.strip().isdigit()]
MOTD_PREDICT        = os.getenv("MOTD_PREDICT", "1") not in ("0","false","False","no","NO")
MOTD_MIN_SAMPLES    = int(os.getenv("MOTD_MIN_SAMPLES", "30"))
MOTD_CONF_MIN       = int(os.getenv("MOTD_CONF_MIN", "65"))

# Allowed tips (NO Over 1.5)
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
STATS_CACHE: Dict[int, Tuple[float, list]] = {}  # fixture_id -> (timestamp, stats_response)

# ── DB ────────────────────────────────────────────────────────────────────────
DB_PATH = "tip_performance.db"

def db_conn():
    return sqlite3.connect(DB_PATH)

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
        try:
            conn.execute("ALTER TABLE tips ADD COLUMN sent_ok INTEGER DEFAULT 1")
        except Exception:
            pass
        conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id INTEGER UNIQUE,
            verdict  INTEGER CHECK (verdict IN (0,1)),
            created_ts INTEGER
        )""")
        conn.execute("DROP VIEW IF EXISTS v_tip_stats")
        conn.execute("""
        CREATE VIEW IF NOT EXISTS v_tip_stats AS
        SELECT t.market, t.suggestion,
               AVG(f.verdict) AS hit_rate,
               COUNT(DISTINCT t.match_id) AS n
        FROM (
          SELECT match_id, market, suggestion, MAX(created_ts) AS last_ts
          FROM tips
          GROUP BY match_id, market, suggestion
        ) lt
        JOIN tips t ON t.match_id=lt.match_id AND t.created_ts=lt.last_ts
        JOIN feedback f ON f.match_id = t.match_id
        GROUP BY t.market, t.suggestion
        """)
        conn.commit()

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
    # 90s cache
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

# ── Model & features ──────────────────────────────────────────────────────────
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
    try:
        return 1.0 / (1.0 + math.exp(-z))
    except OverflowError:
        return 0.0 if z < 0 else 1.0

def _num(v) -> float:
    try:
        if isinstance(v, str) and v.endswith('%'):
            return float(v[:-1])
        return float(v or 0)
    except Exception:
        return 0.0

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
    xg_h = _num(sh.get("Expected Goals", 0)); xg_a = _num(sa.get("Expected Goals", 0))
    sot_h = _num(sh.get("Shots on Target", 0)); sot_a = _num(sa.get("Shots on Target", 0))
    cor_h = _num(sh.get("Corner Kicks", 0));   cor_a = _num(sa.get("Corner Kicks", 0))
    pos_h = _num(sh.get("Ball Possession", 0)); pos_a = _num(sa.get("Ball Possession", 0))
    red_h = _num(sh.get("Red Cards", 0));       red_a = _num(sa.get("Red Cards", 0))
    return {
        "minute": float(minute),
        "goals_h": float(gh), "goals_a": float(ga),
        "goals_sum": float(gh + ga), "goals_diff": float(gh - ga),
        "xg_h": float(xg_h), "xg_a": float(xg_a), "xg_sum": float(xg_h + xg_a),
        "sot_h": float(sot_h), "sot_a": float(sot_a), "sot_sum": float(sot_h + sot_a),
        "cor_h": float(cor_h), "cor_a": float(cor_a), "cor_sum": float(cor_h + cor_a),
        "pos_h": float(pos_h), "pos_a": float(pos_a),
        "red_h": float(red_h), "red_a": float(red_a),
    }

def predict_with_model(model: Dict[str, Any], feat: Dict[str, float]) -> Optional[float]:
    try:
        names = model["features"]
        coef = model["coef"]
        w0 = float(model.get("intercept", 0.0))
        z = w0 + sum(float(coef[i]) * float(feat.get(names[i], 0.0)) for i in range(len(names)))
        return _sigmoid(z)
    except Exception:
        logging.exception("[MODEL] Prediction error")
        return None

# === Data sufficiency / “no free wins” gates ================================
def stats_coverage_ok(feat: Dict[str, float], minute: int) -> bool:
    """After REQUIRE_STATS_MINUTE, require >= REQUIRE_DATA_FIELDS non-zero among {xg_sum,sot_sum,cor_sum}."""
    if minute < REQUIRE_STATS_MINUTE:
        return True
    fields = [feat.get("xg_sum", 0.0), feat.get("sot_sum", 0.0), feat.get("cor_sum", 0.0)]
    nonzero = sum(1 for v in fields if (v or 0) > 0)
    return nonzero >= max(0, REQUIRE_DATA_FIELDS)

def has_intensity_trigger(feat: Dict[str, float]) -> bool:
    """Require at least one sign there’s actual pressure."""
    return (feat.get("xg_sum", 0.0) >= 0.6) or (feat.get("sot_sum", 0.0) >= 3.0) or (feat.get("cor_sum", 0.0) >= 6.0)

# ── Tip logic (model-first, heuristic-fallback, with guards) ─────────────────
def decide_market(match: Dict[str, Any]) -> Tuple[str, str, float, Dict[str, float]]:
    feat = extract_features(match)
    minute = int(feat["minute"])
    gh = int(feat["goals_h"]); ga = int(feat["goals_a"])
    total_goals = gh + ga

    # Hard stops
    if minute >= 90:
        return "Over/Under 2.5 Goals", "Under 2.5 Goals", 0.0, feat  # never send at 90'
    if minute >= 30 and feat["xg_sum"] == 0 and feat["sot_sum"] == 0 and feat["cor_sum"] == 0:
        return "Over/Under 2.5 Goals", "Over 2.5 Goals", 0.0, feat

    # Stats coverage gate (after 35')
    if not stats_coverage_ok(feat, minute):
        return "Over/Under 2.5 Goals", "Over 2.5 Goals", 0.0, feat

    # Require at least one intensity trigger any time after 15'
    if minute >= 15 and not has_intensity_trigger(feat):
        return "Over/Under 2.5 Goals", "Over 2.5 Goals", 0.0, feat

    models = _load_model()
    if ONLY_MODEL_MODE and not models:
        logging.warning("[MODEL] ONLY_MODEL_MODE is ON but no model loaded.")
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
        # light heuristic if no model JSON is present
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
        hist_rate, n = get_historic_hit_rate(market, suggestion)
        bump = (hist_rate - 0.5) * 14.0 * min(1.0, n / 50.0)
        raw_conf_pct = round(max(35.0, min(95.0, base + bump)))
        return market, suggestion, raw_conf_pct, feat

    # choose best candidate among allowed
    allowed = [(m, s, float(p)) for (m, s, p) in candidates if s in ALLOWED_SUGGESTIONS]
    if not allowed:
        return "Over/Under 2.5 Goals", "Over 2.5 Goals", 0.0, feat

    market, suggestion, prob = max(allowed, key=lambda x: x[2])
    raw_pct = int(round(min(95.0, max(35.0, prob * 100.0))))
    hist_rate, n = get_historic_hit_rate(market, suggestion)
    raw_pct = int(round(min(95.0, max(35.0, raw_pct + (hist_rate - 0.5) * 10.0 * min(1.0, n / 50.0)))))
    return market, suggestion, raw_pct, feat

# Guards for “too late” or “already decided”
def is_settled_or_impossible(suggestion: str, gh: int, ga: int) -> bool:
    total = gh + ga
    both_scored = (gh > 0 and ga > 0)
    if suggestion == "Over 2.5 Goals":  return total >= 3
    if suggestion == "Under 2.5 Goals": return total >= 3
    if suggestion == "BTTS: Yes":       return both_scored
    if suggestion == "BTTS: No":        return both_scored
    return False

def is_too_late(suggestion: str, minute: int, gh: int, ga: int, feat: Optional[Dict[str,float]]=None) -> bool:
    total = gh + ga
    if suggestion == "Over 2.5 Goals" and minute >= O25_LATE_MINUTE and total < O25_LATE_MIN_GOALS:
        return True
    if suggestion == "BTTS: Yes" and minute >= BTTS_LATE_MINUTE and (gh == 0 or ga == 0):
        return True
    # suppress trivial “Under 2.5” very late without evidence
    if suggestion == "Under 2.5 Goals" and minute >= UNDER_SUPPRESS_AFTER_MIN:
        xg_sum = (feat or {}).get("xg_sum", 0.0)
        sot_sum = (feat or {}).get("sot_sum", 0.0)
        cor_sum = (feat or {}).get("cor_sum", 0.0)
        if (xg_sum + sot_sum) < 0.8 and cor_sum < 4:
            return True
    return False

# ── Message builder & sender ─────────────────────────────────────────────────
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

    # stats line (xG / SOT / CK / POS / RED)
    stat_line = ""
    if any([stat.get("xg_h",0), stat.get("xg_a",0), stat.get("sot_h",0), stat.get("sot_a",0), stat.get("cor_h",0), stat.get("cor_a",0)]):
        pos_h = int(stat.get("pos_h",0)); pos_a = int(stat.get("pos_a",0))
        red_h = int(stat.get("red_h",0)); red_a = int(stat.get("red_a",0))
        stat_line = (
            f"\n🇮🇹 xG H {stat.get('xg_h',0):.2f} / A {stat.get('xg_a',0):.2f} • "
            f"SOT {int(stat.get('sot_h',0))}–{int(stat.get('sot_a',0))} • "
            f"CK {int(stat.get('cor_h',0))}–{int(stat.get('cor_a',0))} • "
            f"POS {pos_h}%–{pos_a}% • RED {red_h}–{red_a}"
        )

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
        if recent_tip_exists(fid, DUP_COOLDOWN_MIN):
            continue

        chosen = decide_market(match)
        market, suggestion, conf, feat = chosen
        gh = match["goals"]["home"] or 0; ga = match["goals"]["away"] or 0
        minute = int((match.get("fixture", {}) or {}).get("status", {}).get("elapsed") or 0)

        if suggestion not in ALLOWED_SUGGESTIONS or conf < CONF_THRESHOLD:
            continue
        if is_settled_or_impossible(suggestion, gh, ga) or is_too_late(suggestion, minute, gh, ga, feat):
            continue

        msg, kb, meta = build_tip_message(match, chosen)
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

# ── MOTD (unchanged selection; will include prior-based lean if configured) ──
def get_best_prior_for_league(league_id: int, min_samples: int) -> Optional[Tuple[str, str, float, int]]:
    with db_conn() as conn:
        cur = conn.execute("""
            SELECT t.market, t.suggestion, AVG(f.verdict) AS hit, COUNT(DISTINCT t.match_id) AS n
            FROM (
              SELECT match_id, market, suggestion, MAX(created_ts) AS last_ts
              FROM tips
              GROUP BY match_id, market, suggestion
            ) lt
            JOIN tips t ON t.match_id=lt.match_id AND t.created_ts=lt.last_ts
            JOIN feedback f ON f.match_id = t.match_id
            WHERE t.league_id = ?
            GROUP BY t.market, t.suggestion
            HAVING n >= ?
            ORDER BY hit DESC, n DESC
            LIMIT 1
        """, (int(league_id), int(min_samples)))
        row = cur.fetchone()
        if not row: return None
        return (row[0], row[1], float(row[2] or 0.5), int(row[3] or 0))

def get_best_prior_global(min_samples: int) -> Optional[Tuple[str, str, float, int]]:
    with db_conn() as conn:
        cur = conn.execute("""
            SELECT t.market, t.suggestion, AVG(f.verdict) AS hit, COUNT(DISTINCT t.match_id) AS n
            FROM (
              SELECT match_id, market, suggestion, MAX(created_ts) AS last_ts
              FROM tips
              GROUP BY match_id, market, suggestion
            ) lt
            JOIN tips t ON t.match_id=lt.match_id AND t.created_ts=lt.last_ts
            JOIN feedback f ON f.match_id = t.match_id
            GROUP BY t.market, t.suggestion
            HAVING n >= ?
            ORDER BY hit DESC, n DESC
            LIMIT 1
        """, (int(min_samples),))
        row = cur.fetchone()
        if not row: return None
        return (row[0], row[1], float(row[2] or 0.5), int(row[3] or 0))

def pick_match_of_the_day(fixtures: List[Dict[str,Any]]) -> Optional[Dict[str,Any]]:
    fixtures = [f for f in fixtures if (f.get("fixture") or {}).get("status",{}).get("short") in ("NS","TBD")]
    if not fixtures: return None
    candidates = []
    for f in fixtures:
        lg = (f.get("league") or {}); lid = lg.get("id") or 0
        prior = get_best_prior_for_league(int(lid), MOTD_MIN_SAMPLES) or get_best_prior_global(MOTD_MIN_SAMPLES)
        if not prior:
            continue
        pm_market, pm_suggestion, pm_hit, pm_n = prior
        pct = int(round(pm_hit * 100))
        if pm_suggestion in ALLOWED_SUGGESTIONS and pct >= MOTD_CONF_MIN:
            ts = (f.get("fixture") or {}).get("timestamp") or 10**12
            pri_index = (LEAGUE_PRIORITY_IDS.index(lid) + 1) if lid in LEAGUE_PRIORITY_IDS else (len(LEAGUE_PRIORITY_IDS) + 2)
            candidates.append((pm_hit, pm_n, -pri_index, ts, f))
    if not candidates:
        return None
    candidates.sort(key=lambda t: (-t[0], -t[1], t[2], t[3]))
    return candidates[0][4]

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
    league_id = lg.get("id") or 0
    ts = fx.get("timestamp"); when = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC") if ts else "TBD"
    fid = fx.get("id")

    lines = [
        "🌟 <b>Match of the Day</b>",
        f"<b>Match:</b> {home} vs {away}",
        f"🗓 <b>Kickoff:</b> {when}",
        f"🏆 <b>League:</b> {league}",
    ]
    if MOTD_PREDICT:
        prior = get_best_prior_for_league(int(league_id), MOTD_MIN_SAMPLES) or get_best_prior_global(MOTD_MIN_SAMPLES)
        if prior:
            pm_market, pm_suggestion, pm_hit, pm_n = prior
            pct = int(round(pm_hit * 100))
            if pm_suggestion in ALLOWED_SUGGESTIONS and pct >= MOTD_CONF_MIN:
                lines.append(f"🔮 <b>Lean:</b> {escape(pm_suggestion)}")
                lines.append(f"📈 <b>Confidence:</b> {pct}% <i>(prior, n={pm_n})</i>")
    msg = "\n".join(lines)
    kb = []
    row = []
    if BET_URL_TMPL:
        try: row.append({"text": "💰 Bet Now", "url": BET_URL_TMPL.format(home=home, away=away, fixture_id=fid)})
        except Exception: pass
    if WATCH_URL_TMPL:
        try: row.append({"text": "📺 Watch (when live)", "url": WATCH_URL_TMPL.format(home=home, away=away, fixture_id=fid)})
        except Exception: pass
    if row: kb.append(row)
    send_telegram(msg, kb)
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
    })

@app.route("/debug/send")
def debug_send():
    limit = int(request.args.get("n", 3))
    sent = 0
    matches = fetch_live_matches()
    for m in matches[:limit]:
        chosen = decide_market(m)
        market, suggestion, conf, feat = chosen
        gh = m["goals"]["home"] or 0; ga = m["goals"]["away"] or 0
        minute = int((m.get("fixture", {}) or {}).get("status", {}).get("elapsed") or 0)
        if suggestion not in ALLOWED_SUGGESTIONS or conf < CONF_THRESHOLD:
            continue
        if is_settled_or_impossible(suggestion, gh, ga) or is_too_late(suggestion, minute, gh, ga, feat):
            continue
        msg, kb, meta = build_tip_message(m, chosen)
        if not msg: continue
        ok = send_telegram(msg, kb)
        save_tip(meta["match_id"], int(meta["league_id"]), meta["league"], meta["home"], meta["away"],
                 market, suggestion, float(meta["confidence"]), meta["score_at_tip"], int(meta["minute"]),
                 sent_ok=1 if ok else 0)
        if ok: sent += 1
    return jsonify({"requested": limit, "sent": sent, "total_matches": len(matches)})

# Telegram webhook for callback buttons
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

# Admin: quick config dump
def _require_api_key():
    key = request.headers.get("X-API-Key") or request.args.get("key")
    if not API_KEY or key != API_KEY:
        abort(401)

@app.route("/stats/config")
def stats_config():
    _require_api_key()
    return jsonify({
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
    init_db()
    scheduler = BackgroundScheduler()
    scheduler.add_job(match_alert, "interval", minutes=5, id="scan", replace_existing=True)
    scheduler.add_job(send_match_of_the_day, CronTrigger(hour=8, minute=0, timezone="UTC"),
                      id="motd", replace_existing=True, misfire_grace_time=3600, coalesce=True)
    scheduler.start()
    logging.info("⏱️ Scheduler started (scan=5m, MOTD=08:00 UTC)")
    port = int(os.getenv("PORT", 5000))
    logging.info("✅ Robi Superbrain started.")
    app.run(host="0.0.0.0", port=port)
