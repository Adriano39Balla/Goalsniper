import os
import requests
from datetime import datetime
from flask import Flask, jsonify
from dotenv import load_dotenv
from your_prediction_module import get_predictions  # Replace with your actual import

# Load environment variables
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

app = Flask(__name__)

def send_telegram_message(message: str):
    """Send a message to the configured Telegram chat."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Error sending message: {e}")

@app.route("/predict", methods=["GET"])
def run_daily_predictions():
    """Run match predictions and send them to Telegram."""
    try:
        predictions = get_predictions()  # Your prediction function

        if not predictions:
            return jsonify({"status": "ok", "message": "No predictions available"}), 200

        # Format predictions message
        today = datetime.utcnow().strftime("%Y-%m-%d")
        message_lines = [f"⚽ <b>Today's Predictions ({today})</b>\n"]
        for p in predictions:
            # Example: "Team A vs Team B — 2-1"
            message_lines.append(f"<b>{p['home']}</b> vs <b>{p['away']}</b> — {p['prediction']}")

        message_text = "\n".join(message_lines)

        # Send to Telegram
        send_telegram_message(message_text)

        return jsonify({"status": "ok", "message": "Predictions sent"}), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
