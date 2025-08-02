import os
import logging
from flask import Flask
from apscheduler.schedulers.background import BackgroundScheduler
from predictions import run_live_predictions

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

scheduler = BackgroundScheduler()

@app.route("/")
def home():
    return "⚽ Live Match Prediction Bot is running."

@app.route("/predict")
def predict():
    run_live_predictions()
    return "✅ Predictions sent."

def start_scheduler():
    scheduler.add_job(run_live_predictions, "interval", minutes=10)
    scheduler.start()

if __name__ == "__main__":
    start_scheduler()
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
