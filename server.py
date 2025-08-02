import os
import logging
import requests
import sqlite3
from flask import Flask, jsonify
from typing import List, Dict, Any, Optional
from apscheduler.schedulers.background import BackgroundScheduler

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment Variables
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
API_KEY = os.getenv("API_KEY")

FOOTBALL_API_URL = "https://v3.football.api-sports.io/fixtures"
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
HEADERS = {"x-apisports-key": API_KEY, "Accept": "application/json"}
DB_PATH = "tip_performance.db"
CONFIDENCE_THRESHOLD = 70  # Only send if >= 70%

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
        return False
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    try:
        res = requests.post(f"{TELEGRAM_API_URL}/sendMessage", json=payload, timeout=10)
        return res.ok
    except Exception as e:
        logger.error(f"[Telegram] {e}")
        return False

def fetch_match_stats(fixture_id: int) -> Optional[List[Dict[str, Any]]]:
    try:
        res = requests.get(
            "https://v3.football.api-sports.io/fixtures/statistics",
            headers=HEADERS,
            params={"fixture": fixture_id},
            timeout=10
        )
        res.raise_for_status()
        data = res.json().get("response", [])
        return data if data else None
    except Exception as e:
        logger.warning(f"[SKIP] Stats fetch failed for {fixture_id}: {e}")
        return None

def fetch_live_matches() -> List[Dict[str, Any]]:
    try:
        res = requests.get(FOOTBALL_API_URL, headers=HEADERS, params={"live": "all"}, timeout=10)
        res.raise_for_status()
        matches = res.json().get("response", [])
        for m in matches:
            fid = m.get("fixture", {}).get("id")
            m["statistics"] = fetch_match_stats(fid) or []
        return matches
    except Exception as e:
        logger.error(f"API error: {e}")
        return []

def calculate_confidence(s_home: dict, s_away: dict) -> int:
    """Simple confidence calculation based on match stats"""
    score = 0
    score += min(int(s_home.get("Shots on Target", 0) or 0), 10) * 5
    score += min(int(s_away.get("Shots on Target", 0) or 0), 10) * 5
    score += min(int(s_home.get("Corner Kicks", 0) or 0), 10) * 3
    score += min(int(s_away.get("Corner Kicks", 0) or 0), 10) * 3
    score += abs(int(str(s_home.get("Ball Possession", "0")).replace('%', '') or 0) -
                 int(str(s_away.get("Ball Possession", "0")).replace('%', '') or 0))
    return min(score, 100)

def generate_tip(match: Dict[str, Any]) -> Optional[str]:
    match_id = match["fixture"]["id"]
    home = match["teams"]["home"]["name"]
    away = match["teams"]["away"]["name"]
    score_home = match["goals"]["home"]
    score_away = match["goals"]["away"]
    elapsed = match["fixture"]["status"]["elapsed"]

    if elapsed is None or elapsed > 90:
        return None

    stats = match.get("statistics", [])
    if not stats:
        logger.info(f"[SKIP] No stats found for match {match_id}")
        return None

    data = {s["team"]["name"]: {i["type"]: i["value"] for i in s["statistics"]} for s in stats}
    if home not in data or away not in data:
        logger.info(f"[SKIP] Incomplete stats for match {match_id}")
        return None

    s_home = data[home]
    s_away = data[away]

    confidence = calculate_confidence(s_home, s_away)
    if confidence < CONFIDENCE_THRESHOLD:
        return None

    return (
        f"⚽️ Match in progress: {home} vs {away}\n"
        f"⏱️ Minute: {elapsed}'\n"
        f"🔢 Score: {score_home} - {score_away}\n"
        f"📊 Confidence: {confidence}%\n\n"
        f"🤖 Suggested Bet: High attacking pressure detected!"
    )

@app.route("/")
def home():
    return "🤖 Robi Superbrain is active and scanning live matches."

@app.route("/match-alert")
def match_alert():
    matches = fetch_live_matches()
    sent = 0
    for match in matches:
        tip = generate_tip(match)
        if tip and send_telegram(tip):
            sent += 1
    return jsonify({"status": "ok", "matches": len(matches), "tips_sent": sent})

def scheduled_task():
    logger.info("Running scheduled match scan...")
    match_alert()

if __name__ == "__main__":
    init_db()
    scheduler = BackgroundScheduler()
    scheduler.add_job(scheduled_task, "interval", minutes=5)  # Every 5 minutes
    scheduler.start()

    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
