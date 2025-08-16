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
from requests.adapters import HTTPAdapter, Retry
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
app = Flask(__name__)

# ── Env ───────────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")
API_KEY            = os.getenv("API_KEY")                 # admin endpoints
APISPORTS_KEY      = os.getenv("APISPORTS_KEY", API_KEY)  # API-Football
PUBLIC_BASE_URL    = os.getenv("PUBLIC_BASE_URL")
BET_URL_TMPL       = os.getenv("BET_URL")
WATCH_URL_TMPL     = os.getenv("WATCH_URL")
WEBHOOK_SECRET     = os.getenv("TELEGRAM_WEBHOOK_SECRET")

# Core knobs
CONF_THRESHOLD     = int(os.getenv("CONF_THRESHOLD", "60"))     # send only if >= 60%
MAX_TIPS_PER_SCAN  = int(os.getenv("MAX_TIPS_PER_SCAN", "8"))   # soft cap per scan (set 0 to disable)
DUP_COOLDOWN_MIN   = int(os.getenv("DUP_COOLDOWN_MIN", "20"))   # don't resend same fixture within N min

# Dynamic threshold targeting 2–15 tips/day
DYN_BAND_STR        = os.getenv("DYNAMIC_TARGET_BAND", "2,15")
try:
    DYN_MIN, DYN_MAX = [int(x) for x in DYN_BAND_STR.split(",")]
except Exception:
    DYN_MIN, DYN_MAX = 2, 15
DYN_ENABLE          = os.getenv("DYNAMIC_ENABLE", "1") not in ("0","false","False","no","NO")

# Late windows (per-market)
O25_LATE_MINUTE     = int(os.getenv("O25_LATE_MINUTE", "88"))
O25_LATE_MIN_GOALS  = int(os.getenv("O25_LATE_MIN_GOALS", "2"))  # allow O2.5 at 88'+ only if ≥2 goals
BTTS_LATE_MINUTE    = int(os.getenv("BTTS_LATE_MINUTE", "88"))

# MOTD leagues priority + predictor knobs
LEAGUE_PRIORITY_IDS = [int(x) for x in (os.getenv("MOTD_LEAGUE_IDS", "39,140,135,78,61,2").split(",")) if x.strip().isdigit()]
MOTD_PREDICT        = os.getenv("MOTD_PREDICT", "1") not in ("0","false","False","no","NO")
MOTD_MIN_SAMPLES    = int(os.getenv("MOTD_MIN_SAMPLES", "30"))
MOTD_CONF_MIN       = int(os.getenv("MOTD_CONF_MIN", "65"))      # require ≥65% to display lean

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
CAL_CACHE: Dict[str, Any] = {"ts": 0, "bins": []}  # calibration cache

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
            PRIMARY KEY (match_id, created_ts)
        )""")
        conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id INTEGER UNIQUE,         -- unique to upsert final label
            verdict  INTEGER CHECK (verdict IN (0,1)),
            created_ts INTEGER
        )""")
        conn.execute("""
        CREATE VIEW IF NOT EXISTS v_tip_stats AS
          SELECT t.market, t.suggestion, AVG(f.verdict) AS hit_rate, COUNT(*) AS n
          FROM tips t JOIN feedback f ON f.match_id = t.match_id
          GROUP BY t.market, t.suggestion
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS tip_snapshots (
            match_id INTEGER,
            created_ts INTEGER,
            payload TEXT,
            PRIMARY KEY (match_id, created_ts)
        )""")
        conn.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )""")
        conn.commit()

def set_setting(key: str, value: str):
    with db_conn() as conn:
        conn.execute("INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))
        conn.commit()

def get_setting(key: str, default: Optional[str] = None) -> Optional[str]:
    with db_conn() as conn:
        cur = conn.execute("SELECT value FROM settings WHERE key=?", (key,))
        row = cur.fetchone()
        return row[0] if row else default

def save_tip(match_id: int, league_id: int, league: str, home: str, away: str,
             market: str, suggestion: str, confidence: float,
             score_at_tip: str, minute: int, snapshot: Optional[dict] = None):
    now = int(time.time())
    with db_conn() as conn:
        conn.execute("""
          INSERT INTO tips(match_id,league_id,league,home,away,market,suggestion,confidence,score_at_tip,minute,created_ts)
          VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (match_id, league_id, league, home, away, market, suggestion, confidence, score_at_tip, minute, now))
        if snapshot:
            conn.execute("""
              INSERT OR REPLACE INTO tip_snapshots(match_id, created_ts, payload)
              VALUES (?,?,?)
            """, (match_id, now, json.dumps(snapshot)[:200000]))
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
    js = _api_get(FOOTBALL_API_URL, {"date": date.today().isoformat(), "timezone": "UTC"})
    return js.get("response", []) if isinstance(js, dict) else []

def fetch_fixtures_by_ids(ids: List[int]) -> Dict[int, Dict[str, Any]]:
    if not ids:
        return {}
    js = _api_get(FOOTBALL_API_URL, {"ids": ",".join(str(i) for i in ids), "timezone": "UTC"})
    resp = js.get("response", []) if isinstance(js, dict) else []
    return { (fx.get("fixture") or {}).get("id"): fx for fx in resp }

# ── Calibration ───────────────────────────────────────────────────────────────
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

def calibrate_confidence(p: float) -> float:
    bins = compute_calibration_bins()
    for b in bins:
        if b["lo"] <= p <= b["hi"]:
            if b["n"] >= 30:
                return round(b["hit"] * 100.0)
            break
    return p

# ── Tip logic ─────────────────────────────────────────────────────────────────
def _num(v) -> float:
    try:
        if isinstance(v, str) and v.endswith('%'):
            return float(v[:-1])
        return float(v or 0)
    except Exception:
        return 0.0

def decide_market(match: Dict[str, Any]) -> Tuple[str, str, float, Dict[str, float]]:
    """Allowed: Over 2.5, Under 2.5, BTTS: Yes, BTTS: No"""
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

    total_goals = gh + ga
    xg_sum = xg_h + xg_a
    sot_sum = sot_h + sot_a
    corners = cor_h + cor_a

    # Default lean Over 2.5
    market, suggestion, score = "Over/Under 2.5 Goals", "Over 2.5 Goals", 50.0

    # No stats: prefer Under late/low, else mild Over 2.5
    if not stats:
        if minute >= 70 and total_goals <= 1:
            conf = 60
            return "Over/Under 2.5 Goals", "Under 2.5 Goals", conf, {"xg_h":0,"xg_a":0,"sot_h":0,"sot_a":0,"cor_h":0,"cor_a":0}
        conf = 55
        return "Over/Under 2.5 Goals", "Over 2.5 Goals", conf, {"xg_h":0,"xg_a":0,"sot_h":0,"sot_a":0,"cor_h":0,"cor_a":0}

    if xg_sum >= 2.4 and total_goals <= 2 and minute >= 25: score += 18
    if sot_sum >= 6: score += 10
    if corners >= 10: score += 5
    if minute < 20: score -= 8

    # BTTS rules
    if minute >= 25 and xg_h >= 0.9 and xg_a >= 0.9 and sot_h >= 3 and sot_a >= 3 and total_goals <= 3:
        market, suggestion, score = "BTTS", "Yes", max(score, 62)
    if xg_sum < 1.4 and sot_sum < 4 and total_goals == 0 and minute > 35:
        market, suggestion, score = "BTTS", "No", 62
    if minute > 70 and total_goals <= 1 and suggestion != "BTTS: Yes":
        market, suggestion = "Over/Under 2.5 Goals", "Under 2.5 Goals"
        score = max(score, 62)

    score = max(35.0, min(90.0, score))
    hist_rate, n = get_historic_hit_rate(market, suggestion)
    bump = (hist_rate - 0.5) * 14.0 * min(1.0, n / 50.0)  # cap ±7%
    raw_conf = round(max(35.0, min(95.0, score + bump)))
    cal_conf = calibrate_confidence(raw_conf)

    statline = {"xg_h": xg_h, "xg_a": xg_a, "sot_h": sot_h, "sot_a": sot_a, "cor_h": cor_h, "cor_a": cor_a}
    return market, suggestion, cal_conf, statline

# Guards
def is_settled_or_impossible(suggestion: str, gh: int, ga: int) -> bool:
    total = gh + ga
    both_scored = (gh > 0 and ga > 0)
    if suggestion == "Over 2.5 Goals":  return total >= 3
    if suggestion == "Under 2.5 Goals": return total >= 3
    if suggestion == "BTTS: Yes":       return both_scored
    if suggestion == "BTTS: No":        return both_scored
    return False

def is_too_late(suggestion: str, minute: int, gh: int, ga: int) -> bool:
    total = gh + ga
    if suggestion == "Over 2.5 Goals" and minute >= O25_LATE_MINUTE and total < O25_LATE_MIN_GOALS:
        return True
    if suggestion == "BTTS: Yes" and minute >= BTTS_LATE_MINUTE and (gh == 0 or ga == 0):
        return True
    return False

# ── Orchestration ─────────────────────────────────────────────────────────────
last_heartbeat = 0
def maybe_send_heartbeat():
    global last_heartbeat
    if time.time() - last_heartbeat > 1200:
        send_telegram("✅ Robi Superbrain is online and scanning…")
        last_heartbeat = time.time()

def get_effective_threshold() -> int:
    v = get_setting("conf_threshold_override", None)
    try:
        if v is not None:
            return int(v)
    except Exception:
        pass
    return CONF_THRESHOLD_BASE

def adjust_threshold_if_needed():
    if not DYN_ENABLE:
        return
    since = int(time.time()) - 24*3600
    with db_conn() as conn:
        cur = conn.execute("SELECT COUNT(*) FROM tips WHERE created_ts>=?", (since,))
        count_24h = cur.fetchone()[0]
    eff = get_effective_threshold()
    new_eff = eff
    if count_24h > DYN_MAX:
        new_eff = min(80, eff + 2)
    elif count_24h < DYN_MIN:
        new_eff = max(50, eff - 2)
    if new_eff != eff:
        set_setting("conf_threshold_override", str(new_eff))
        logging.info(f"[DYN] tips_24h={count_24h} adjust threshold {eff} -> {new_eff}")
    else:
        logging.info(f"[DYN] tips_24h={count_24h} keep threshold {eff}")

def format_human_tip(match: Dict[str, Any],
                     chosen: Optional[Tuple[str,str,float,Dict[str,float]]] = None) -> Tuple[str, list, Dict[str, Any]]:
    fixture = match["fixture"]; league_block = match.get("league") or {}
    league_id = league_block.get("id") or 0
    league = escape(f"{league_block.get('country','')} - {league_block.get('name','')}".strip(" -"))
    minute = fixture["status"].get("elapsed") or 0
    match_id = fixture["id"]
    home = escape(match["teams"]["home"]["name"]); away = escape(match["teams"]["away"]["name"])
    g_home = match["goals"]["home"] or 0; g_away = match["goals"]["away"] or 0

    market, suggestion, confidence, stat = chosen if chosen else decide_market(match)

    eff_threshold = get_effective_threshold()
    if suggestion not in ALLOWED_SUGGESTIONS or confidence < eff_threshold:
        return "", [], {"match_id": match_id, "skip": True, "confidence": confidence}
    if is_settled_or_impossible(suggestion, g_home, g_away) or is_too_late(suggestion, int(minute), g_home, g_away):
        return "", [], {"match_id": match_id, "skip": True, "confidence": confidence}

    snapshot = {
        "minute": int(minute), "gh": g_home, "ga": g_away,
        "league_id": int(league_id), "market": market, "suggestion": suggestion,
        "confidence": int(confidence), "stat": stat
    }
    save_tip(match_id, int(league_id), league, home, away, market, suggestion, float(confidence),
             f"{g_home}-{g_away}", int(minute), snapshot=snapshot)

    stat_line = ""
    if any(stat.values()):
        stat_line = f"\n📊 xG H {stat['xg_h']:.2f} / A {stat['xg_a']:.2f} • SOT {int(stat['sot_h'])}–{int(stat['sot_a'])} • CK {int(stat['cor_h'])}–{int(stat['cor_a'])}"

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

    meta = {"match_id": match_id, "league_id": int(league_id), "confidence": confidence}
    return msg, kb, meta

def match_alert():
    logging.info("🔍 Scanning live matches…")
    matches = fetch_live_matches()
    if not matches:
        logging.warning("[SCAN] No matches returned – check APISPORTS_KEY/plan or time of day.")
        return

    sent = 0
    for match in matches:
        if MAX_TIPS_PER_SCAN and sent >= MAX_TIPS_PER_SCAN:
            break
        fid = (match.get("fixture", {}) or {}).get("id")
        if recent_tip_exists(fid, DUP_COOLDOWN_MIN):
            continue
        chosen = decide_market(match)
        market, suggestion, conf, _ = chosen
        gh = match["goals"]["home"] or 0; ga = match["goals"]["away"] or 0
        minute = int((match.get("fixture", {}) or {}).get("status", {}).get("elapsed") or 0)
        if suggestion not in ALLOWED_SUGGESTIONS or conf < get_effective_threshold():
            continue
        if is_settled_or_impossible(suggestion, gh, ga) or is_too_late(suggestion, minute, gh, ga):
            continue
        msg, kb, meta = format_human_tip(match, chosen)
        if meta.get("skip"):
            continue
        logging.info(f"[TIP] sending fixture={fid} conf={meta['confidence']} suggestion={suggestion}")
        if send_telegram(msg, kb):
            sent += 1

    logging.info(f"[TIP] Sent={sent}")
    maybe_send_heartbeat()

# ── Auto-labeler ──────────────────────────────────────────────────────────────
def autolabel_unscored():
    since = int(time.time()) - 36*3600
    with db_conn() as conn:
        cur = conn.execute("""
          SELECT DISTINCT match_id FROM tips
          WHERE created_ts>=? AND match_id NOT IN (SELECT match_id FROM feedback)
        """, (since,))
        ids = [r[0] for r in cur.fetchall()]
    if not ids:
        return
    for i in range(0, len(ids), 20):
        batch = ids[i:i+20]
        fx = fetch_fixtures_by_ids(batch)
        for fid in batch:
            m = fx.get(fid)
            if not m: continue
            status = ((m.get("fixture") or {}).get("status") or {}).get("short")
            if status not in ("FT","AET","PEN"):
                continue
            gh = (m.get("goals") or {}).get("home") or 0
            ga = (m.get("goals") or {}).get("away") or 0
            with db_conn() as conn:
                cur = conn.execute("""
                  SELECT suggestion FROM tips WHERE match_id=? ORDER BY created_ts DESC LIMIT 1
                """, (fid,))
                row = cur.fetchone()
                if not row: 
                    continue
                s = row[0]
            total = gh + ga
            verdict = None
            if s == "Over 2.5 Goals": verdict = 1 if total >= 3 else 0
            elif s == "Under 2.5 Goals": verdict = 1 if total <= 2 else 0
            elif s == "BTTS: Yes": verdict = 1 if (gh > 0 and ga > 0) else 0
            elif s == "BTTS: No": verdict = 1 if (gh == 0 or ga == 0) else 0
            if verdict is not None and not feedback_exists(fid):
                upsert_feedback(fid, verdict)
                CAL_CACHE["ts"] = 0  # invalidate calibration cache
                logging.info(f"[AUTO] Labeled match {fid} -> {verdict}")

# ── MOTD (with prior-based lean ≥ MOTD_CONF_MIN) ──────────────────────────────
def get_best_prior_for_league(league_id: int, min_samples: int) -> Optional[Tuple[str, str, float, int]]:
    with db_conn() as conn:
        cur = conn.execute("""
            SELECT t.market, t.suggestion, AVG(f.verdict) AS hit, COUNT(*) AS n
            FROM tips t JOIN feedback f ON f.match_id = t.match_id
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
            SELECT t.market, t.suggestion, AVG(f.verdict) AS hit, COUNT(*) AS n
            FROM tips t JOIN feedback f ON f.match_id = t.match_id
            GROUP BY t.market, t.suggestion
            HAVING n >= ?
            ORDER BY hit DESC, n DESC
            LIMIT 1
        """, (int(min_samples),))
        row = cur.fetchone()
        if not row: return None
        return (row[0], row[1], float(row[2] or 0.5), int(row[3] or 0))

def pick_match_of_the_day(fixtures: List[Dict[str,Any]]) -> Optional[Dict[str,Any]]:
    pri = {lid:i for i,lid in enumerate(LEAGUE_PRIORITY_IDS)}
    fixtures = [f for f in fixtures if (f.get("fixture") or {}).get("status",{}).get("short") in ("NS","TBD")]
    if not fixtures: return None
    def key_fn(f):
        lg = (f.get("league") or {})
        lid = lg.get("id", 10**9)
        idx = pri.get(lid, len(LEAGUE_PRIORITY_IDS)+1)
        kick = (f.get("fixture") or {}).get("timestamp") or 10**12
        return (idx, kick)
    fixtures.sort(key=key_fn)
    return fixtures[0]

def send_match_of_the_day():
    fixtures = fetch_today_fixtures_utc()
    motd = pick_match_of_the_day(fixtures)
    if not motd:
        logging.info("[MOTD] No suitable fixture found for today.")
        return

    lg = motd.get("league") or {}
    fx = motd.get("fixture") or {}
    home = escape((motd.get("teams") or {}).get("home",{}).get("name",""))
    away = escape((motd.get("teams") or {}).get("away",{}).get("name",""))
    league = escape(f"{lg.get('country','')} - {lg.get('name','')}".strip(" -"))
    league_id = lg.get("id") or 0
    ts = fx.get("timestamp")
    when = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC") if ts else "TBD"
    fid = fx.get("id")

    lines = [
        "🌟 <b>Match of the Day</b>",
        f"<b>Match:</b> {home} vs {away}",
        f"🗓 <b>Kickoff:</b> {when}",
        f"🏆 <b>League:</b> {league}",
    ]

    if MOTD_PREDICT:
        prior = get_best_prior_for_league(int(league_id), MOTD_MIN_SAMPLES) or \
                get_best_prior_global(MOTD_MIN_SAMPLES)
        if prior:
            pm_market, pm_suggestion, pm_hit, pm_n = prior
            pct = int(round(pm_hit * 100))
            if pm_suggestion in ALLOWED_SUGGESTIONS and pct >= MOTD_CONF_MIN:
                lines.append(f"🔮 <b>Lean:</b> {escape(pm_suggestion)}")
                lines.append(f"📈 <b>Confidence:</b> {pct}% <i>(prior, n={pm_n})</i>")
            else:
                logging.info(f"[MOTD] Prior insufficient or not allowed (suggestion={pm_suggestion}, pct={pct}).")
        else:
            logging.info("[MOTD] No prior available; skipping lean.")

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
    logging.info(f"[MOTD] Sent for fixture={fid} (predict={MOTD_PREDICT})")

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

@app.route("/debug/format")
def debug_format():
    matches = fetch_live_matches()
    if not matches:
        return jsonify({"error": "no matches"}), 200
    i = 0
    try: i = int(request.args.get("i", 0))
    except Exception: pass
    i = max(0, min(i, len(matches)-1))
    chosen = decide_market(matches[i])
    msg, kb, meta = format_human_tip(matches[i], chosen)
    return jsonify({"fixture_id": meta.get("match_id"), "confidence": meta.get("confidence"),
                    "message": msg, "inline_keyboard": kb})

@app.route("/debug/send")
def debug_send():
    limit = int(request.args.get("n", 3))
    sent = 0
    matches = fetch_live_matches()
    for m in matches[:limit]:
        chosen = decide_market(m)
        market, suggestion, conf, _ = chosen
        gh = m["goals"]["home"] or 0; ga = m["goals"]["away"] or 0
        minute = int((m.get("fixture", {}) or {}).get("status", {}).get("elapsed") or 0)
        if suggestion not in ALLOWED_SUGGESTIONS or conf < get_effective_threshold():
            continue
        if is_settled_or_impossible(suggestion, gh, ga) or is_too_late(suggestion, minute, gh, ga):
            continue
        msg, kb, meta = format_human_tip(m, chosen)
        if meta.get("skip"): continue
        ok = send_telegram(msg, kb)
        logging.info(f"[DEBUG/SEND] fixture={m.get('fixture',{}).get('id')} ok={ok}")
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
        CAL_CACHE["ts"] = 0  # invalidate calibration cache on new label
        answer_callback(cq["id"], "Thanks! Feedback recorded ✅" if verdict else "Got it. Marked as wrong ❌")
    except Exception:
        logging.exception("webhook error")
        try: answer_callback(cq.get("id",""), "Sorry, couldn’t record feedback.")
        except Exception: pass
    return "ok"

# Admin: stats (protected)
def _require_api_key():
    key = request.headers.get("X-API-Key") or request.args.get("key")
    if not API_KEY or key != API_KEY:
        abort(401)

@app.route("/stats/hitrate")
def stats_hitrate():
    _require_api_key()
    with db_conn() as conn:
        cur = conn.execute("""
          SELECT market, suggestion, ROUND(hit_rate*100,1), n
          FROM v_tip_stats ORDER BY n DESC, hit_rate DESC
        """)
        rows = [{"market": r[0], "suggestion": r[1], "hit_rate_pct": r[2], "n": r[3]} for r in cur.fetchall()]
    return jsonify({"rows": rows})

@app.route("/stats/calibration")
def stats_calibration():
    _require_api_key()
    return jsonify({"bins": compute_calibration_bins()})

@app.route("/stats/volume")
def stats_volume():
    _require_api_key()
    with db_conn() as conn:
        since = int(time.time()) - 24*3600
        cur = conn.execute("SELECT COUNT(*) FROM tips WHERE created_ts>=?", (since,))
        n = cur.fetchone()[0]
        cur = conn.execute("""
          SELECT suggestion, COUNT(*) FROM tips WHERE created_ts>=?
          GROUP BY suggestion ORDER BY COUNT(*) DESC
        """, (since,))
        per = [{"suggestion": r[0], "count": r[1]} for r in cur.fetchall()]
    return jsonify({"last_24h": n, "by_suggestion": per, "threshold_effective": get_effective_threshold()})

@app.route("/stats/config")
def stats_config():
    _require_api_key()
    return jsonify({
        "CONF_THRESHOLD_BASE": CONF_THRESHOLD_BASE,
        "threshold_effective": get_effective_threshold(),
        "MAX_TIPS_PER_SCAN": MAX_TIPS_PER_SCAN,
        "DUP_COOLDOWN_MIN": DUP_COOLDOWN_MIN,
        "DYN_ENABLE": DYN_ENABLE,
        "DYN_MIN": DYN_MIN, "DYN_MAX": DYN_MAX,
        "O25_LATE_MINUTE": O25_LATE_MINUTE, "O25_LATE_MIN_GOALS": O25_LATE_MIN_GOALS,
        "BTTS_LATE_MINUTE": BTTS_LATE_MINUTE,
        "MOTD_PREDICT": MOTD_PREDICT,
        "MOTD_MIN_SAMPLES": MOTD_MIN_SAMPLES,
        "MOTD_CONF_MIN": MOTD_CONF_MIN,
        "APISPORTS_KEY_set": bool(APISPORTS_KEY),
        "TELEGRAM_CHAT_ID_set": bool(TELEGRAM_CHAT_ID),
    })

# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    scheduler = BackgroundScheduler()
    # Live scan every 5 minutes
    scheduler.add_job(match_alert, "interval", minutes=5, id="scan", replace_existing=True)
    # MOTD 08:00 UTC
    scheduler.add_job(send_match_of_the_day, CronTrigger(hour=8, minute=0, timezone="UTC"),
                      id="motd", replace_existing=True, misfire_grace_time=3600, coalesce=True)
    # Auto-label every 15 minutes
    scheduler.add_job(autolabel_unscored, "interval", minutes=15, id="autolabel", replace_existing=True)
    # Dynamic threshold daily
    scheduler.add_job(adjust_threshold_if_needed, CronTrigger(hour=23, minute=59, timezone="UTC"),
                      id="dynthresh", replace_existing=True)

    scheduler.start()
    logging.info("⏱️ Scheduler started (scan=5m, autolabel=15m, MOTD=08:00 UTC, dyn=23:59 UTC)")
    port = int(os.getenv("PORT", 5000))
    logging.info("✅ Robi Superbrain started.")
    app.run(host="0.0.0.0", port=port)
