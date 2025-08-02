from flask import Flask, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
import logging

from predictions import run_daily_predictions
from telegram import send_telegram_message

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Root route to avoid 404s ---
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "ok", "message": "GoalSniper API is live 🚀"})

# --- Predict route ---
@app.route("/predict", methods=["GET"])
def predict():
    result = run_daily_predictions()
    send_telegram_message(f"Daily prediction result: {result}")
    return jsonify(result)

# --- Daily scheduler ---
def scheduled_task():
    logger.info("Running scheduled task: daily predictions.")
    result = run_daily_predictions()
    send_telegram_message(f"Daily prediction result: {result}")

scheduler = BackgroundScheduler()
scheduler.add_job(scheduled_task, 'cron', hour=8, minute=0)  # Every day at 08:00 UTC
scheduler.start()

# --- Keep scheduler alive ---
@app.before_first_request
def init_scheduler():
    logger.info("Scheduler initialized and running.")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
