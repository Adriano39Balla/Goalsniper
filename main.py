import os
import json
import time
import math
import logging
import requests
import sqlite3
from flask import Flask, jsonify, request
from typing import List, Dict, Any, Optional, Tuple
from requests.adapters import HTTPAdapter, Retry
from apscheduler.schedulers.background import BackgroundScheduler

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

app = Flask(__name__)

# --- Env ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
API_KEY = os.getenv("API_KEY")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL")  # e.g. https://Goalsniper-backend.onrender.com
BET_URL = os.getenv("BET_URL")
WATCH_URL = os.getenv("WATCH_URL")

# --- External APIs ---
FOOTBALL_API_URL = "https://v3.football.api-sports.io/fixtures"
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
HEADERS = {"x-apisports-key": API_KEY, "Accept": "application/json"}

# --- DB ---
DB_PATH = "tip_performance.db"

# --- HTTP session with retries ---
session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))

# ---------------- DB ----------------
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tips (
                match_id INTEGER PRIMARY KEY,
                league TEXT,
                home TEXT,
                away TEXT,
                market TEXT,
                suggestion TEXT,
                confidence REAL,
                score_at_tip TEXT,
                minute INTEGER,
                created_ts INTEGER
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id INTEGER,
                verdict INTEGER CHECK (verdict IN (0,1)),
                created_ts INTEGER
            )
        """)
        # Simple aggregates to "learn" from history without heavy ML.
        conn.execute("""
            CREATE VIEW IF NOT EXISTS v_tip_stats AS
            SELECT market, suggestion,
                   AVG(verdict) AS hit_rate,
                   COUNT(*) AS n
            FROM tips t
            JOIN feedback f ON f.match_id = t.match_id
            GROUP BY market, suggestion
        """)
        conn.commit()

# Lightweight helper to fetch historic hit rates for confidence tuning
def get_historic_hit_rate(market: str, suggestion: str) -> Tuple[float, int]:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "SELECT COALESCE(hit_rate, 0.5), COALESCE(n, 0) FROM v_tip_stats WHERE market=? AND suggestion=?",
            (market, suggestion),
        )
        row = cur.fetchone()
        return (row[0], row[1]) if row else (0.5, 0)

def save_tip(match_id: int, league: str, home: str, away: str,
             market: str, suggestion: str, confidence: float,
             score_at_tip: str, minute: int):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT OR REPLACE INTO tips(match_id,league,home,away,market,suggestion,confidence,score_at_tip,minute,created_ts)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (match_id, league, home, away, market, suggestion, confidence, score_at_tip, minute, int(time.time())))
        conn.commit()

def record_feedback(match_id: int, verdict: int):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("INSERT INTO feedback(match_id, verdict, created_ts) VALUES (?,?,?)",
                     (match_id, verdict, int(time.time())))
        conn.commit()

# ---------------- Telegram ----------------
def send_telegram(message: str, inline_keyboard: Optional[list] = None) -> bool:
    if not TELEGRAM_CHAT_ID or not TELEGRAM_BOT_TOKEN:
        logging.error("Missing Telegram credentials")
        return False

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
    }
    if inline_keyboard:
        payload["reply_markup"] = json.dumps({"inline_keyboard": inline_keyboard})

    try:
        res = session.post(f"{TELEGRAM_API_URL}/sendMessage", data=payload, timeout=10)
        if not res.ok:
            logging.error(f"[Telegram] Failed: {res.status_code} - {res.text}")
        return res.ok
    except Exception as e:
        logging.error(f"[Telegram] Exception: {e}")
        return False

# ---------------- Football API ----------------
def fetch_match_stats(fixture_id: int) -> Optional[List[Dict[str, Any]]]:
    try:
        res = session.get(
            f"https://v3.football.api-sports.io/fixtures/statistics",
            headers=HEADERS,
            params={"fixture": fixture_id},
            timeout=10
        )
        res.raise_for_status()
        return res.json().get("response", [])
    except Exception as e:
        logging.warning(f"[SKIP] Stats fetch failed for {fixture_id}: {e}")
        return None

def fetch_live_matches() -> List[Dict[str, Any]]:
    try:
        res = session.get(FOOTBALL_API_URL, headers=HEADERS, params={"live": "all"}, timeout=10)
        res.raise_for_status()
        matches = res.json().get("response", [])

        filtered = []
        for m in matches:
            status = (m.get("fixture", {}) or {}).get("status", {}) or {}
            elapsed = status.get("elapsed")
            # Skip if no minute or junk time
            if elapsed is None or elapsed > 90:
                continue
            fid = m.get("fixture", {}).get("id")
            m["statistics"] = fetch_match_stats(fid) or []
            if m["statistics"]:
                filtered.append(m)

        return filtered
    except Exception as e:
        logging.error(f"API error: {e}")
        return []

# ---------------- Tip logic (human-friendly) ----------------
def build_stat_map(stats: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for team_block in stats:
        team = (team_block.get("team") or {}).get("name")
        metric_map: Dict[str, float] = {}
        for item in team_block.get("statistics") or []:
            typ, val = item.get("type"), item.get("value")
            # Normalize numeric-ish values (e.g., "61%", None)
            if isinstance(val, str) and val.endswith("%"):
                try:
                    val = float(val.replace("%", "").strip())
                except Exception:
                    val = 0
            elif val is None:
                val = 0
            metric_map[typ] = float(val)
        if team:
            out[team] = metric_map
    return out

def decide_market(match: Dict[str, Any]) -> Tuple[str, str, float]:
    """
    Returns (market, suggestion, confidence[0-100])

    Markets supported for now to match your screenshot style:
    - "Over/Under 2.5 Goals"
    - "BTTS"
    """
    home = match["teams"]["home"]["name"]
    away = match["teams"]["away"]["name"]
    goals_home = match["goals"]["home"] or 0
    goals_away = match["goals"]["away"] or 0
    minute = match["fixture"]["status"]["elapsed"] or 0

    stats_map = build_stat_map(match["statistics"])
    s_home = stats_map.get(home, {})
    s_away = stats_map.get(away, {})

    # Features (robust to missing stats)
    xg_sum = (s_home.get("Expected Goals", 0.0) + s_away.get("Expected Goals", 0.0))
    shots_on = (s_home.get("Shots on Target", 0.0) + s_away.get("Shots on Target", 0.0))
    big_chances = (s_home.get("Big Chances", 0.0) + s_away.get("Big Chances", 0.0))
    corners = (s_home.get("Corner Kicks", 0.0) + s_away.get("Corner Kicks", 0.0))
    total_goals = goals_home + goals_away

    # Simple interpretable rules -> pick market + base score
    score = 50.0
    market = "Over/Under 2.5 Goals"
    suggestion = "Over 2.5 Goals"

    # Push towards Over if good chance creation but score still low
    if xg_sum >= 2.0 and total_goals <= 2:
        score += 18
    if shots_on >= 6:
        score += 10
    if big_chances >= 3:
        score += 8
    if corners >= 10:
        score += 5

    # Early minutes: reduce confidence naturally
    if minute < 20:
        score -= 8
    if minute > 70 and total_goals <= 1:
        # late & still low: favor Under and adjust market text
        market = "Over/Under 2.5 Goals"
        suggestion = "Under 2.5 Goals"
        score = 55 + (70 - min(minute, 90)) * 0.2  # taper

    # Low xG / few chances -> prefer BTTS No (defensive game vibe)
    if xg_sum < 1.4 and shots_on < 4 and total_goals == 0 and minute > 35:
        market = "BTTS"
        suggestion = "No"
        score = 62

    # Clamp base to [35, 90]
    score = max(35.0, min(90.0, score))

    # "Learning" bump from historic accuracy of same-market suggestions
    hist_rate, n = get_historic_hit_rate(market, suggestion)  # (0..1)
    # Weight the bump by sample size (cap impact to ±7%)
    bump = (hist_rate - 0.5) * 14.0 * min(1.0, n / 50.0)
    final_conf = max(35.0, min(95.0, score + bump))

    return market, suggestion, round(final_conf, 0)

def format_human_tip(match: Dict[str, Any]) -> Tuple[str, list]:
    """
    Returns (message_text, inline_keyboard)
    """
    fixture = match["fixture"]
    league_block = match.get("league") or {}
    league = f"{league_block.get('country','')} - {league_block.get('name','')}".strip(" -")
    minute = fixture["status"]["elapsed"] or 0
    match_id = fixture["id"]
    home = match["teams"]["home"]["name"]
    away = match["teams"]["away"]["name"]
    g_home = match["goals"]["home"] or 0
    g_away = match["goals"]["away"] or 0

    market, suggestion, confidence = decide_market(match)

    # Persist tip so feedback can be learned
    save_tip(
        match_id=match_id,
        league=league,
        home=home,
        away=away,
        market=market,
        suggestion=suggestion,
        confidence=float(confidence),
        score_at_tip=f"{g_home}-{g_away}",
        minute=int(minute),
    )

    # Message like your 2nd screenshot
    # Example:
    # ⚽️ New Tip!
    # Match: Lechia Gdansk vs Motor Lublin
    # Tip: Over 2.5 Goals
    # 📈 Confidence: 58%
    # 🏆 League: Poland - Ekstraklasa
    lines = [
        "⚽️ <b>New Tip!</b>",
        f"<b>Match:</b> {home} vs {away}",
        f"<b>Tip:</b> {suggestion if market == 'BTTS' else suggestion}",
        f"📈 <b>Confidence:</b> {int(confidence)}%",
        f"🏆 <b>League:</b> {league}",
    ]
    msg = "\n".join(lines)

    # Inline buttons
    keyboard = []
    # Feedback (use URLs so we don't need Telegram webhook/callback handler)
    if PUBLIC_BASE_URL:
        keyboard.append([
            {"text": "👍 Correct", "url": f"{PUBLIC_BASE_URL}/fb?match_id={match_id}&v=1"},
            {"text": "👎 Wrong",   "url": f"{PUBLIC_BASE_URL}/fb?match_id={match_id}&v=0"},
        ])
    # Optional action links
    action_row = []
    if BET_URL:
        action_row.append({"text": "💰 Bet Now", "url": BET_URL})
    if WATCH_URL:
        action_row.append({"text": "📺 Watch Match", "url": WATCH_URL})
    if action_row:
        keyboard.append(action_row)

    return msg, keyboard

# ---------------- Scheduler task ----------------
last_heartbeat = 0
def maybe_send_heartbeat():
    global last_heartbeat
    if time.time() - last_heartbeat > 1200:  # Every 20 mins
        send_telegram("✅ Robi Superbrain is online and scanning matches…")
        last_heartbeat = time.time()

def match_alert():
    logging.info("🔍 Scanning live matches…")
    matches = fetch_live_matches()
    logging.info(f"[SCAN] {len(matches)} matches after filtering")
    sent = 0
    for match in matches:
        try:
            msg, kb = format_human_tip(match)
            if send_telegram(msg, kb):
                sent += 1
        except Exception as e:
            logging.exception(f"Failed to format/send tip for fixture {match.get('fixture',{}).get('id')}: {e}")
    logging.info(f"[TIP] Sent {sent} tips")
    maybe_send_heartbeat()

# ---------------- Routes ----------------
@app.route("/")
def home():
    return "🤖 Robi Superbrain is active and watching the game."

@app.route("/match-alert")
def manual_match_alert():
    match_alert()
    return jsonify({"status": "ok"})

@app.route("/fb")
def feedback():
    """
    GET /fb?match_id=123&v=1   (1 = correct, 0 = wrong)
    Stores feedback and replies with a tiny thank-you.
    """
    match_id = request.args.get("match_id", type=int)
    verdict = request.args.get("v", type=int)
    if match_id is None or verdict not in (0, 1):
        return jsonify({"ok": False, "error": "bad params"}), 400
    record_feedback(match_id, verdict)
    txt = "🙏 Thanks for the feedback!"
    # also acknowledge in chat (silent)
    try:
        send_telegram(f"📊 Feedback saved for match <code>{match_id}</code>: {'Correct' if verdict==1 else 'Wrong'}")
    except Exception:
        pass
    return jsonify({"ok": True, "message": txt})

# ---------------- Main ----------------
if __name__ == "__main__":
    init_db()
    scheduler = BackgroundScheduler()
    scheduler.add_job(match_alert, "interval", minutes=5)
    scheduler.start()
    port = int(os.getenv("PORT", 5000))
    logging.info("✅ Robi Superbrain started and scanning every 5 minutes.")
    app.run(host="0.0.0.0", port=port)
