# /app/main.py
import os
import logging
from flask import Flask, jsonify
from football_api import get_today_fixtures, get_team_stats
from prediction import make_predictions
from telegram import send_telegram_message

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route("/predict", methods=["GET"])
def predict():
    try:
        fixtures = get_today_fixtures()
        if not fixtures:
            logger.info("No fixtures found today.")
            return jsonify({"status": "no fixtures today"}), 200

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

        # Send Telegram message
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
        return jsonify({"status": "sent", "predictions": predictions}), 200

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
