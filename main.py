from flask import Flask, jsonify
import logging
from apscheduler.schedulers.background import BackgroundScheduler
from predictions import run_live_predictions

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)

# Background scheduler
scheduler = BackgroundScheduler()

def scheduled_task():
    logger.info("⏱️ Scheduled task: Running live predictions...")
    try:
        run_live_predictions()
    except Exception as e:
        logger.error(f"❌ Scheduled predictions error: {e}")

# Add job to run every 10 minutes
scheduler.add_job(scheduled_task, "interval", minutes=10)
scheduler.start()

@app.route("/")
def home():
    return "🤖 Robi Superbrain is monitoring live matches every 10 minutes..."

@app.route("/predict", methods=["GET"])
def predict():
    try:
        logger.info("📡 Manual trigger: Running live predictions...")
        run_live_predictions()
        return jsonify({"status": "success", "message": "Live predictions executed."})
    except Exception as e:
        logger.error(f"❌ Error running predictions: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    import os
    try:
        port = int(os.getenv("PORT", 5000))
        app.run(host="0.0.0.0", port=port, use_reloader=False)
    finally:
        scheduler.shutdown()
