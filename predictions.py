import os
import logging
import requests
import certifi
from math import floor
from football_api import get_live_fixtures, get_live_stats

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def send_to_telegram(message: str):
    """Send message to Telegram."""
    try:
        telegram_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML"
        }
        res = requests.post(telegram_url, json=payload, timeout=10, verify=certifi.where())
        res.raise_for_status()
    except Exception as e:
        logger.error(f"Telegram send error: {e}")

def calculate_confidence(stats):
    """Calculate confidence score from live stats."""
    score = 0
    try:
        shots = int(stats.get("Shots on Goal", 0))
        corners = int(stats.get("Corner Kicks", 0))
        possession = int(stats.get("Ball Possession", "0%").replace("%", ""))
        xg = float(stats.get("Expected Goals", 0))

        score += min(shots * 5, 25)
        score += min(corners * 3, 15)
        score += min(possession / 2, 15)
        score += min(xg * 10, 45)
    except Exception as e:
        logger.error(f"Error calculating confidence: {e}")

    return min(100, floor(score))

def run_live_predictions():
    fixtures = get_live_fixtures()
    if not fixtures:
        send_to_telegram("⚽ No live matches found.")
        return

    for match in fixtures:
        live_stats = get_live_stats(match["fixture_id"])
        home_name = match["home"]["name"]
        away_name = match["away"]["name"]

        if home_name in live_stats:
            home_conf = calculate_confidence(live_stats[home_name])
        else:
            home_conf = 0

        if away_name in live_stats:
            away_conf = calculate_confidence(live_stats[away_name])
        else:
            away_conf = 0

        confidence = max(home_conf, away_conf)

        # ✅ Only trigger alert if confidence is 75% or higher
        if confidence >= 75:
            msg = (
                f"⚽ <b>{home_name}</b> vs <b>{away_name}</b>\n"
                f"⏱️ {match['elapsed']}'\n"
                f"🔢 Score: {match['score']['home']} - {match['score']['away']}\n"
                f"📊 Confidence: <b>{confidence}%</b>\n"
            )
            send_to_telegram(msg)
