# main.py
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
    for match in matches:
        tip = calculate_confidence_and_suggestion(match)
        if tip:
            send_telegram_message(tip)

@app.route("/")
def home():
    return "GoalSniper is live and watching in-play matches ⚽🔥"

if __name__ == "__main__":
    scheduler = BackgroundScheduler()
    scheduler.add_job(check_matches, "interval", minutes=10)
    scheduler.start()
    logger.info("Scheduler started. Press Ctrl+C to exit.")

    app.run(host="0.0.0.0", port=10000)
