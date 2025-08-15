import os
import json
import time
import logging
import requests
import sqlite3
from flask import Flask, jsonify, request, abort
from typing import List, Dict, Any, Optional, Tuple
from requests.adapters import HTTPAdapter, Retry
from apscheduler.schedulers.background import BackgroundScheduler

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

app = Flask(__name__)

# ── Env ───────────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")
API_KEY            = os.getenv("API_KEY")  # used for API-Sports header
PUBLIC_BASE_URL    = os.getenv("PUBLIC_BASE_URL")  # e.g. https://your-service.up.railway.app
BET_URL_TMPL       = os.getenv("BET_URL")      # optional; e.g. https://book.com/?fixture={fixture_id}
WATCH_URL_TMPL     = os.getenv("WATCH_URL")    # optional; e.g. https://livescore.com/?fixture={fixture_id}
WEBHOOK_SECRET     = os.getenv("TELEGRAM_WEBHOOK_SECRET")  # verify Telegram webhook

# ── External APIs ─────────────────────────────────────────────────────────────
FOOTBALL_API_URL = "https://v3.football.api-sports.io/fixtures"
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
HEADERS          = {"x-apisports-key": API_KEY, "Accept": "application/json"}

# ── DB ────────────────────────────────────────────────────────────────────────
DB_PATH = "tip_performance.db"

# ── HTTP session ──────────────────────────────────────────────────────────────
session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))

# ───────────────────────────── DB helpers ─────────────────────────────────────
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tips (
                match_id INTEGER PRIMARY KEY,
                league   TEXT,
                home     TEXT,
                away     TEXT,
                market   TEXT,        -- e.g. "Over/Under 2.5 Goals", "BTTS"
                suggestion TEXT,      -- e.g. "Over 2.5 Goals", "No"
                confidence REAL,      -- % we sent
                score_at_tip TEXT,    -- "1-0"
                minute    INTEGER,
                created_ts INTEGER
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id INTEGER,
                verdict  INTEGER CHECK (verdict IN (0,1)),  -- 1=correct, 0=wrong
                created_ts INTEGER
            )
        """)
        conn.execute("""
            CREATE VIEW IF NOT EXISTS v_tip_stats AS
            SELECT market, suggestion, AVG(verdict) AS hit_rate, COUNT(*) AS n
            FROM tips t JOIN feedback f ON f.match_id = t.match_id
            GROUP BY market, suggestion
        """)
        conn.commit()

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

def get_historic_hit_rate(market: str, suggestion: str) -> Tuple[float, int]:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "SELECT COALESCE(hit_rate,0.5), COALESCE(n,0) FROM v_tip_stats WHERE market=? AND suggestion=?",
            (market, suggestion),
        )
        row = cur.fetchone()
        return (row[0], row[1]) if row else (0.5, 0)

# ─────────────────────────── Telegram helpers ─────────────────────────────────
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

def answer_callback(callback_id: str, text: str):
    try:
        session.post(f"{TELEGRAM_API_URL}/answerCallbackQuery",
                     data={"callback_query_id": callback_id, "text": text, "show_alert": False}, timeout=10)
    except Exception:
        pass

# ───────────────────────── Football API ───────────────────────────────────────
def fetch_match_stats(fixture_id: int) -> Optional[List[Dict[str, Any]]]:
    try:
        res = session.get(
            "https://v3.football.api-sports.io/fixtures/statistics",
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
            elapsed = (m.get("fixture", {}) or {}).get("status", {}).get("elapsed")
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

# ───────────────────────── Tip logic ──────────────────────────────────────────
def _num(v) -> float:
    try:
        if isinstance(v, str) and v.endswith('%'):
            return float(v[:-1])
        return float(v or 0)
    except Exception:
        return 0.0

def decide_market(match: Dict[str, Any]) -> Tuple[str, str, float]:
    """
    Pick a human-friendly market/suggestion + confidence.
    Markets: Over/Under 2.5, BTTS, Next Goal (leaning).
    """
    home = match["teams"]["home"]["name"]
    away = match["teams"]["away"]["name"]
    gh = match["goals"]["home"] or 0
    ga = match["goals"]["away"] or 0
    minute = match["fixture"]["status"]["elapsed"] or 0

    # Build stat dicts
    stats = {s["team"]["name"]: {i["type"]: i["value"] for i in s["statistics"]} for s in match["statistics"]}
    sh, sa = stats.get(home, {}), stats.get(away, {})

    xg_sum   = _num(sh.get("Expected Goals", 0)) + _num(sa.get("Expected Goals", 0))
    sot_sum  = _num(sh.get("Shots on Target", 0)) + _num(sa.get("Shots on Target", 0))
    corners  = _num(sh.get("Corner Kicks", 0)) + _num(sa.get("Corner Kicks", 0))
    poss_h   = _num(sh.get("Ball Possession", 0))
    poss_a   = _num(sa.get("Ball Possession", 0))
    total_goals = gh + ga

    # Start Over/Under as default
    market = "Over/Under 2.5 Goals"
    suggestion = "Over 2.5 Goals"
    score = 50.0

    if xg_sum >= 2.4 and total_goals <= 2 and minute >= 25:
        score += 18
    if sot_sum >= 6:
        score += 10
    if corners >= 10:
        score += 5
    if minute < 20:
        score -= 8

    # Defensive vibe -> BTTS: No
    if xg_sum < 1.4 and sot_sum < 4 and total_goals == 0 and minute > 35:
        market, suggestion, score = "BTTS", "No", 62

    # Late & still low -> Under
    if minute > 70 and total_goals <= 1:
        market, suggestion = "Over/Under 2.5 Goals", "Under 2.5 Goals"
        score = max(score, 60)

    # Clamp and add historical bump
    score = max(35.0, min(90.0, score))
    hist_rate, n = get_historic_hit_rate(market, suggestion)
    bump = (hist_rate - 0.5) * 14.0 * min(1.0, n / 50.0)  # cap ±7% based on sample
    confidence = round(max(35.0, min(95.0, score + bump)))

    return market, suggestion, confidence

def format_human_tip(match: Dict[str, Any]) -> Tuple[str, list, Dict[str, Any]]:
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

    # Persist the tip (for later feedback)
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

    lines = [
        "⚽️ <b>New Tip!</b>",
        f"<b>Match:</b> {home} vs {away}",
        f"<b>Tip:</b> {suggestion}",
        f"📈 <b>Confidence:</b> {int(confidence)}%",
        f"🏆 <b>League:</b> {league}",
    ]
    msg = "\n".join(lines)

    # Inline buttons (callback-based); optional Bet/Watch row
    kb = [[
        {"text": "👍 Correct", "callback_data": json.dumps({"t": "correct", "id": match_id})},
        {"text": "👎 Wrong",   "callback_data": json.dumps({"t": "wrong",   "id": match_id})},
    ]]
    link_row = []
    if BET_URL_TMPL:
        try:
            link_row.append({"text": "💰 Bet Now", "url": BET_URL_TMPL.format(home=home, away=away, fixture_id=match_id)})
        except Exception:
            pass
    if WATCH_URL_TMPL:
        try:
            link_row.append({"text": "📺 Watch Match", "url": WATCH_URL_TMPL.format(home=home, away=away, fixture_id=match_id)})
        except Exception:
            pass
    if link_row:
        kb.append(link_row)

    # meta is returned so caller doesn’t recompute
    meta = {"match_id": match_id}
    return msg, kb, meta

# ───────────────────────── Scheduler / Orchestration ──────────────────────────
last_heartbeat = 0
def maybe_send_heartbeat():
    global last_heartbeat
    if time.time() - last_heartbeat > 1200:
        send_telegram("✅ Robi Superbrain is online and scanning matches…")
        last_heartbeat = time.time()

def match_alert():
    logging.info("🔍 Scanning live matches…")
    matches = fetch_live_matches()
    logging.info(f"[SCAN] {len(matches)} matches after filtering")
    sent = 0
    for match in matches:
        try:
            msg, kb, _meta = format_human_tip(match)
            if send_telegram(msg, kb):
                sent += 1
        except Exception as e:
            logging.exception(f"Failed to format/send tip for fixture {match.get('fixture',{}).get('id')}: {e}")
    logging.info(f"[TIP] Sent {sent} tips")
    maybe_send_heartbeat()

# ───────────────────────── Routes ─────────────────────────────────────────────
@app.route("/")
def home():
    return "🤖 Robi Superbrain is active and learning."

@app.route("/match-alert")
def manual_match_alert():
    match_alert()
    return jsonify({"status": "ok"})

# Telegram webhook for callback buttons
@app.route("/telegram-webhook", methods=["POST"])
def telegram_webhook():
    # Optional security: verify Telegram secret header if you set it on setWebhook
    if WEBHOOK_SECRET:
        if request.headers.get("X-Telegram-Bot-Api-Secret-Token") != WEBHOOK_SECRET:
            return "forbidden", 403

    update = request.get_json(force=True, silent=True) or {}
    cq = update.get("callback_query")
    if not cq:
        # ignore non-callback updates
        return "ok"

    try:
        data = cq.get("data")
        payload = json.loads(data) if isinstance(data, str) else (data or {})
        verdict = 1 if payload.get("t") == "correct" else 0
        match_id = int(payload.get("id"))
        record_feedback(match_id, verdict)
        answer_callback(cq["id"], "Thanks! Feedback recorded ✅" if verdict else "Got it. Marked as wrong ❌")
    except Exception as e:
        logging.exception(f"webhook error: {e}")
        try:
            answer_callback(cq.get("id",""), "Sorry, couldn’t record feedback.")
        except Exception:
            pass
    return "ok"

# Optional: server-side feedback API (if you want to label outcomes later)
@app.route("/feedback", methods=["POST"])
def feedback_api():
    key = request.headers.get("X-API-Key") or request.args.get("key")
    if not API_KEY or key != API_KEY:
        abort(401)
    data = request.get_json(force=True, silent=True) or {}
    try:
        match_id = int(data["match_id"])
        verdict  = 1 if data["won"] in (True, 1, "1", "true", "True") else 0
    except Exception:
        abort(400, "Provide match_id and won")
    record_feedback(match_id, verdict)
    return jsonify({"updated": True})

# ───────────────────────── Entrypoint ─────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    scheduler = BackgroundScheduler()
    scheduler.add_job(match_alert, "interval", minutes=5)
    scheduler.start()
    port = int(os.getenv("PORT", 5000))
    logging.info("✅ Robi Superbrain started — scanning every 5 minutes.")
    app.run(host="0.0.0.0", port=port)
