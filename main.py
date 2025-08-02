import os
import logging
import requests
from flask import Flask
from apscheduler.schedulers.background import BackgroundScheduler
from football_api import get_live_matches
from predictions import calculate_confidence_and_suggestion

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger
app = Flask(__name__)

TELEGRAM_TOKEN = os.getenv("BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("CHAT_ID")

def send_telegram(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("Telegram credentials missing")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        res = requests.post(url, json=payload, timeout=10)
        return res.ok
    except Exception as e:
        logger.error(f"Telegram send failed: {e}")
        return False

def check_matches():
    logger.info("Checking live matches...")
    matches = get_live_matches()
    for match in matches:
        tip = calculate_confidence_and_suggestion(match)
        if tip:
            send_telegram(tip)

@app.route("/")
def home():
    return "GoalSniper is live and watching in-play matches ⚽🔥"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
    scheduler = BackgroundScheduler()
    scheduler.add_job(check_matches, "interval", minutes=10)
    scheduler.start()

    logger.info("Scheduler started. Press Ctrl+C to exit.")

    try:
        while True:
            pass
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
