import os
import logging
from flask import Flask, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
from football_api import get_today_fixtures, get_team_stats
from prediction import make_predictions
from telegram import send_telegram_message
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

def run_daily_predictions():
    logger.info(f"Running daily predictions at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    try:
        fixtures = get_today_fixtures()
        if not fixtures:
            logger.info("No fixtures found today.")
            return

        predictions = []
        for fixture in fixtures:
            stats = get_team_stats(
                fixture["home"]["id"],
                fixture["away"]["id"],
                fixture["league_id"],
                fixture["season"]
            )
            pred = make_predictions(stats)
            predictions.append({
                "match": f"{fixture['home']['name']} vs {fixture['away']['name']}",
                "predictions": pred
            })

        # Build Telegram message
        message = "\n\n".join(
            f"📊 {p['match']}\n"
            f"• Over/Under: {p['predictions'].get('over_under', 'N/A')}\n"
            f"• Double Chance: {p['predictions'].get('double_chance', 'N/A')}\n"
            f"• Handicap: {p['predictions'].get('handicap', 'N/A')}\n"
            f"• Result: {p['predictions'].get('result', 'N/A')}"
            for p in predictions
        )
        send_telegram_message(message)
        logger.info("Predictions sent successfully.")
    except Exception as e:
        logger.error(f"Error in daily prediction job: {e}", exc_info=True)

@app.route("/predict", methods=["GET"])
def predict_endpoint():
    try:
        run_daily_predictions()
        return jsonify({"status": "sent manually"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    scheduler = BackgroundScheduler()
    # Run every day at 10:00 UTC
    scheduler.add_job(run_daily_predictions, "cron", hour=10, minute=0)
    scheduler.start()

    logger.info("Scheduler started. App running.")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
