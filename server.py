import os
import logging
import requests
import sqlite3
from flask import Flask, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
from typing import List, Dict, Any, Optional

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

CONFIDENCE_THRESHOLD = 70  # 70%

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
        logger.error("Missing Telegram credentials.")
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
        return res.json().get("response", []) or None
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

def normalize(value: float, max_value: float) -> float:
    try:
        return min(float(value) / max_value, 1.0)
    except (ValueError, TypeError):
        return 0.0

def calculate_confidence(s_home, s_away) -> float:
    shots_home = int(s_home.get("Shots on Target", 0) or 0)
    shots_away = int(s_away.get("Shots on Target", 0) or 0)
    corners_home = int(s_home.get("Corner Kicks", 0) or 0)
    corners_away = int(s_away.get("Corner Kicks", 0) or 0)
    possession_home = int(str(s_home.get("Ball Possession", "0")).replace('%', '') or 0)
    possession_away = int(str(s_away.get("Ball Possession", "0")).replace('%', '') or 0)
    xg_home = float(s_home.get("Expected Goals", 0) or 0)
    xg_away = float(s_away.get("Expected Goals", 0) or 0)

    # Combine stats from both teams
    shots = shots_home + shots_away
    corners = corners_home + corners_away
    possession = max(possession_home, possession_away)  # stronger team's possession
    xg = xg_home + xg_away

    # Weighted confidence calculation
    confidence = (
        normalize(shots, 8) * 0.4 +
        normalize(corners, 10) * 0.2 +
        normalize(possession, 100) * 0.2 +
        normalize(xg, 3) * 0.2
    ) * 100

    return round(confidence, 2)

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
    
def check_matches():
    logger.info("Checking live matches...")
    matches = fetch_live_matches()
    sent = 0
    for match in matches:
        tip = generate_tip(match)
        if tip and send_telegram(tip):
            sent += 1
    logger.info(f"Tips sent: {sent}")

@app.route("/")
def home():
    return "🤖 Robi Superbrain is active and watching the game."

@app.route("/match-alert")
def match_alert():
    check_matches()
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    init_db()
    scheduler = BackgroundScheduler()
    scheduler.add_job(check_matches, "interval", minutes=5)
    scheduler.start()
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
