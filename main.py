import logging
from flask import Flask
from apscheduler.schedulers.background import BackgroundScheduler
from football_api import get_live_matches
from predictions import calculate_confidence_and_suggestion
from telegram import send_telegram_message
import time
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Track last heartbeat message
last_heartbeat = 0
HEARTBEAT_INTERVAL = 25 * 60  # 25 minutes in seconds

def check_matches():
    global last_heartbeat

    logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Scheduler tick: checking live matches...")

    try:
        # Build API URL for logging (without exposing the API key fully)
        api_key = os.getenv("API_KEY", "UNKNOWN")
        safe_key = api_key[:5] + "****" if api_key else "MISSING"
        api_url = f"https://api.sportsdata.io/v4/soccer/scores/json/LiveGame?key={safe_key}"
        logger.info(f"Calling Football API: {api_url}")

        matches = get_live_matches()

        if matches is None:
            logger.error("No response from API. Possible network or credentials issue.")
            return

        logger.info(f"API returned {len(matches)} matches.")

        for match in matches:
            tip = calculate_confidence_and_suggestion(match)
            if tip:
                logger.info(f"Match found meeting criteria: {tip}")
                send_telegram_message(tip)

    except Exception as e:
        logger.error(f"Error while checking matches: {e}")

    # Send occasional heartbeat message to Telegram
    now = time.time()
    if now - last_heartbeat > HEARTBEAT_INTERVAL:
        send_telegram_message("✅ Bot is still scanning for matches...")
        last_heartbeat = now

@app.route("/")
def home():
    return "GoalSniper is live and watching in-play matches ⚽🔥"

def start_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(check_matches, "interval", minutes=10)
    scheduler.start()
    logger.info("✅ Scheduler started and running every 10 minutes.")


if __name__ == "__main__":
    start_scheduler()
    app.run(host="0.0.0.0", port=10000)
