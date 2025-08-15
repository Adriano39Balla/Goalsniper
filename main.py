
import os
import json
import time
import logging
import requests
import sqlite3
from flask import Flask, jsonify
from typing import List, Dict, Any, Optional
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

def send_telegram(message: str) -> bool:
    if not TELEGRAM_CHAT_ID or not TELEGRAM_BOT_TOKEN:
        logging.error("Missing Telegram credentials")
        return False

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "reply_markup": json.dumps({
            "inline_keyboard": [
                [{"text": "💰 Bet Now", "url": "https://your-betting-site.com"}],
                [{"text": "📊 Watch Match", "url": "https://livescore-api.com"}]
            ]
        })
    }

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

def generate_tip(match: Dict[str, Any]) -> Optional[str]:
    match_id = match["fixture"]["id"]
    home = match["teams"]["home"]["name"]
    away = match["teams"]["away"]["name"]
    score_home = match["goals"]["home"]
    score_away = match["goals"]["away"]
    elapsed = match["fixture"]["status"]["elapsed"]

    stats = match["statistics"]
    data = {s["team"]["name"]: {i["type"]: i["value"] for i in s["statistics"]} for s in stats}
    if home not in data or away not in data:
        return None

    s_home = data[home]
    s_away = data[away]
    
    xg_home = float(s_home.get("Expected Goals", 0) or 0)
    xg_away = float(s_away.get("Expected Goals", 0) or 0)
    shots_home = int(s_home.get("Shots on Target", 0) or 0)
    shots_away = int(s_away.get("Shots on Target", 0) or 0)
    corners_home = int(s_home.get("Corner Kicks", 0) or 0)
    corners_away = int(s_away.get("Corner Kicks", 0) or 0)

    tip_lines = []
    total_score = score_home + score_away
    pressure_threshold = 5
    over_goal_trigger = xg_home + xg_away >= 2.5 and total_score <= 2

    for team, stats in [(home, s_home), (away, s_away)]:
        shots = int(stats.get("Shots on Target", 0) or 0)
        corners = int(stats.get("Corner Kicks", 0) or 0)
        poss = int(str(stats.get("Ball Possession", "0")).replace('%', '') or 0)

        if shots >= pressure_threshold or corners >= pressure_threshold or poss >= 60:
            tip_lines.append(
                f"📈 High attacking pressure by <b>{team}</b>\n"
                f"📊 Possession {poss}%, {corners} corners, {shots} shots on target"
            )
            if over_goal_trigger:
                tip_lines.append(f"🤖 Suggested Bet: {team} to score next / Over 2.5 Goals likely")

    if not tip_lines:
        tip_lines.append("📌 Stats suggest a balanced game in progress.")

    return (
        f"⚽️ Match in progress: {home} vs {away}\n"
        f"⏱️ Minute: {elapsed}'\n"
        f"🔢 Score: {score_home} - {score_away}\n\n" +
        "\n".join(tip_lines)
    )

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
        tip = generate_tip(match)
        if tip and send_telegram(tip):
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
