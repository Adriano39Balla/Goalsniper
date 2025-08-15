import os
import json
import time
import logging
import requests
import sqlite3
from flask import Flask, jsonify
from typing import List, Dict, Any, Optional, Tuple
from requests.adapters import HTTPAdapter, Retry
from apscheduler.schedulers.background import BackgroundScheduler

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

app = Flask(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
API_KEY = os.getenv("API_KEY")
FOOTBALL_API_URL = "https://v3.football.api-sports.io/fixtures"
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
HEADERS = {"x-apisports-key": API_KEY, "Accept": "application/json"}
DB_PATH = "tip_performance.db"

# Optional, no placeholders: only render these buttons if set
BET_URL_TMPL = os.getenv("BET_URL")      # e.g. https://mybook.com/?match={home}-{away}
WATCH_URL_TMPL = os.getenv("WATCH_URL")  # e.g. https://livescore.com/?fixture={fixture_id}

session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tip_stats (
                match_id TEXT PRIMARY KEY,
                tip TEXT,
                score TEXT,
                correct INTEGER
            )
        """)
        conn.commit()

def _inline_buttons(match: Dict[str, Any], tip_text: str) -> Dict[str, Any]:
    """Build an inline keyboard. ‘Correct/Wrong’ are callback buttons.
    Bet/Watch links are added only if corresponding env vars are set."""
    home = match["teams"]["home"]["name"]
    away = match["teams"]["away"]["name"]
    fid = match["fixture"]["id"]

    rows = [
        [
            {"text": "👍 Correct", "callback_data": json.dumps({"t": "correct", "id": fid, "tip": tip_text})},
            {"text": "👎 Wrong",   "callback_data": json.dumps({"t": "wrong", "id": fid, "tip": tip_text})}
        ]
    ]
    link_row = []
    if BET_URL_TMPL:
        try:
            link_row.append({"text": "💰 Bet Now", "url": BET_URL_TMPL.format(home=home, away=away, fixture_id=fid)})
        except Exception:
            pass
    if WATCH_URL_TMPL:
        try:
            link_row.append({"text": "📺 Watch Match", "url": WATCH_URL_TMPL.format(home=home, away=away, fixture_id=fid)})
        except Exception:
            pass
    if link_row:
        rows.append(link_row)

    return {"inline_keyboard": rows}

def send_telegram(message: str, match: Optional[Dict[str, Any]] = None, tip_text: str = "") -> bool:
    if not TELEGRAM_CHAT_ID or not TELEGRAM_BOT_TOKEN:
        logging.error("Missing Telegram credentials")
        return False

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
    }
    if match:
        payload["reply_markup"] = json.dumps(_inline_buttons(match, tip_text))

    try:
        res = session.post(f"{TELEGRAM_API_URL}/sendMessage", data=payload, timeout=10)
        if not res.ok:
            logging.error(f"[Telegram] Failed: {res.status_code} - {res.text}")
        return res.ok
    except Exception as e:
        logging.error(f"[Telegram] Exception: {e}")
        return False

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
            elapsed = m.get("fixture", {}).get("status", {}).get("elapsed")
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

def _num(v) -> float:
    try:
        if isinstance(v, str) and v.endswith('%'):
            return float(v[:-1])
        return float(v or 0)
    except Exception:
        return 0.0

def _confidence_from_stats(s_home: Dict[str, Any], s_away: Dict[str, Any], score_h: int, score_a: int) -> float:
    """Quick, explainable confidence 0–100 based on pressure/xG/score state."""
    xg_h = _num(s_home.get("Expected Goals", 0))
    xg_a = _num(s_away.get("Expected Goals", 0))
    sot_h = _num(s_home.get("Shots on Target", 0))
    sot_a = _num(s_away.get("Shots on Target", 0))
    cor_h = _num(s_home.get("Corner Kicks", 0))
    cor_a = _num(s_away.get("Corner Kicks", 0))
    poss_h = _num(s_home.get("Ball Possession", 0))
    poss_a = _num(s_away.get("Ball Possession", 0))

    xg_total = xg_h + xg_a
    pressure = (sot_h + sot_a) * 5 + (cor_h + cor_a) * 2
    balance = abs(poss_h - poss_a)

    raw = 0
    raw += min(xg_total / 3.5, 1.0) * 45          # how open the game is
    raw += min(pressure / 12.0, 1.0) * 35         # finishing pressure
    raw += min(balance / 40.0, 1.0) * 10          # one-sided momentum
    raw += min((score_h + score_a) / 3.0, 1.0) * 10  # goals already
    return round(max(20.0, min(95.0, raw * 100.0 / 100.0)))  # clamp and avoid super low

def _choose_tip(match: Dict[str, Any]) -> Tuple[str, int]:
    """Return (tip_text, confidence%)."""
    home = match["teams"]["home"]["name"]
    away = match["teams"]["away"]["name"]
    score_home = match["goals"]["home"] or 0
    score_away = match["goals"]["away"] or 0
    stats = match["statistics"]
    data = {s["team"]["name"]: {i["type"]: i["value"] for i in s["statistics"]} for s in stats}
    if home not in data or away not in data:
        return ("No clear edge", 40)

    s_home = data[home]; s_away = data[away]
    xg_h = _num(s_home.get("Expected Goals", 0)); xg_a = _num(s_away.get("Expected Goals", 0))
    sot_h = _num(s_home.get("Shots on Target", 0)); sot_a = _num(s_away.get("Shots on Target", 0))
    cor_h = _num(s_home.get("Corner Kicks", 0));   cor_a = _num(s_away.get("Corner Kicks", 0))
    poss_h = _num(s_home.get("Ball Possession", 0)); poss_a = _num(s_away.get("Ball Possession", 0))
    elapsed = match["fixture"]["status"]["elapsed"] or 0

    total_score = (score_home + score_away)
    xg_total = xg_h + xg_a
    sot_total = sot_h + sot_a
    corners_total = cor_h + cor_a

    # Simple, readable rules → one human tip
    if xg_total >= 2.4 and total_score <= 2 and elapsed >= 25:
        tip = "Over 2.5 Goals"
    elif xg_total <= 1.6 and sot_total <= 3 and total_score <= 1 and elapsed >= 20:
        tip = "Under 2.5 Goals"
    elif total_score == 0 and xg_h < 0.6 and xg_a < 0.6 and sot_total <= 2:
        tip = "BTTS: No"
    else:
        # Next goal leaning: pick side with higher pressure
        press_h = sot_h*3 + cor_h*1 + (poss_h-50)*0.2
        press_a = sot_a*3 + cor_a*1 + (poss_a-50)*0.2
        tip = f"Next Goal: {home if press_h >= press_a else away}"

    conf = _confidence_from_stats(s_home, s_away, score_home, score_away)
    return tip, conf

def _format_human_message(match: Dict[str, Any], tip_text: str, conf: int) -> str:
    home = match["teams"]["home"]["name"]
    away = match["teams"]["away"]["name"]
    league = match.get("league", {}).get("country", "") + " - " + match.get("league", {}).get("name", "")
    elapsed = match["fixture"]["status"]["elapsed"] or 0
    score_home = match["goals"]["home"] or 0
    score_away = match["goals"]["away"] or 0

    # Exactly like your target style
    msg = (
        "⚽ <b>New Tip!</b>\n"
        f"<b>Match:</b> {home} vs {away}\n"
        f"<b>Tip:</b> {tip_text}\n"
        f"<b>Confidence:</b> {conf}%\n"
        f"<b>Minute:</b> {elapsed}'  <b>Score:</b> {score_home}–{score_away}\n"
        f"🏆 <b>League:</b> {league.strip(' - ')}"
    )
    return msg

def generate_tip(match: Dict[str, Any]) -> Optional[Tuple[str, str, int]]:
    """Return (message, tip_text, confidence) or None if not enough data."""
    try:
        tip_text, conf = _choose_tip(match)
        message = _format_human_message(match, tip_text, conf)
        return message, tip_text, conf
    except Exception as e:
        logging.exception(f"Tip generation failed: {e}")
        return None

last_heartbeat = 0
def maybe_send_heartbeat():
    global last_heartbeat
    if time.time() - last_heartbeat > 1200:  # Every 20 mins
        send_telegram("✅ Robi Superbrain is online and scanning matches...")
        last_heartbeat = time.time()

def match_alert():
    logging.info("🔍 Scanning live matches...")
    matches = fetch_live_matches()
    logging.info(f"[SCAN] {len(matches)} matches after filtering")
    sent = 0
    for match in matches:
        result = generate_tip(match)
        if not result:
            continue
        message, tip_text, _ = result
        if send_telegram(message, match=match, tip_text=tip_text):
            sent += 1
    logging.info(f"[TIP] Sent {sent} tips")
    maybe_send_heartbeat()

@app.route("/")
def home():
    return "🤖 Robi Superbrain is active and watching the game."

@app.route("/match-alert")
def manual_match_alert():
    match_alert()
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    init_db()
    scheduler = BackgroundScheduler()
    scheduler.add_job(match_alert, "interval", minutes=5)
    scheduler.start()
    port = int(os.getenv("PORT", 5000))
    logging.info("✅ Robi Superbrain started and scanning every 5 minutes.")
    app.run(host="0.0.0.0", port=port)
