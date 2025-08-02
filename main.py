import logging
from flask import Flask
from apscheduler.schedulers.background import BackgroundScheduler
from football_api import get_live_matches
from predictions import calculate_confidence_and_suggestion
from telegram import send_telegram_message

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

def check_matches():
    logger.info("Checking live matches...")
    matches = get_live_matches()

    if not matches:
        logger.info("No live matches found.")
        return

    for match in matches:
        tip = calculate_confidence_and_suggestion(match)
        if tip:
            logger.info(f"Prediction triggered for: {match['home']} vs {match['away']}")
            send_telegram_message(tip)
        else:
            logger.info(f"No prediction for: {match['home']} vs {match['away']}")

def send_heartbeat():
    """Send a 'still working' message every 20 mins."""
    logger.info("Sending heartbeat to Telegram...")
    send_telegram_message("🔍 GoalSniper is live — scanning matches...")

@app.route("/")
def home():
    return "GoalSniper is live and watching in-play matches ⚽🔥"

def start_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(check_matches, "interval", minutes=10)
    scheduler.add_job(send_heartbeat, "interval", minutes=20)
    scheduler.start()
    logger.info("Scheduler started...")

if __name__ == "__main__":
    start_scheduler()
    app.run(host="0.0.0.0", port=10000)
else:
    start_scheduler()
