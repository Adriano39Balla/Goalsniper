import os
import logging
import requests
import certifi
from datetime import datetime

# Logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Environment variables
API_KEY = os.getenv("API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Telegram send function
def send_to_telegram(message: str):
    """Send message to Telegram chat."""
    try:
        telegram_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML"
        }
        response = requests.post(telegram_url, json=payload, timeout=10, verify=certifi.where())
        response.raise_for_status()
        logger.info("Message sent to Telegram successfully.")
    except Exception as e:
        logger.error(f"Error sending message to Telegram: {e}", exc_info=True)

# Main prediction function
def run_daily_predictions():
    try:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        logger.info(f"Running daily predictions for {today}...")

        url = "https://v3.football.api-sports.io/fixtures"
        params = {"date": today}
        headers = {"x-apisports-key": API_KEY}

        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=15,
            verify=certifi.where()
        )
        response.raise_for_status()
        data = response.json()

        fixtures = data.get("response", [])
        logger.info(f"Fixtures found: {len(fixtures)}")

        # Format message for Telegram
        if fixtures:
            message_lines = [f"⚽ <b>Today's Fixtures ({today})</b>\n"]
            for fixture in fixtures[:10]:  # limit to first 10 for message size
                league = fixture.get("league", {}).get("name", "Unknown League")
                home = fixture.get("teams", {}).get("home", {}).get("name", "Home")
                away = fixture.get("teams", {}).get("away", {}).get("name", "Away")
                status = fixture.get("fixture", {}).get("status", {}).get("short", "TBD")
                message_lines.append(f"<b>{home}</b> vs <b>{away}</b> ({league}) - {status}")

            telegram_message = "\n".join(message_lines)
            send_to_telegram(telegram_message)
        else:
            send_to_telegram(f"No fixtures found for {today}.")

        return {"success": True, "data": data}

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        send_to_telegram(f"❌ Error running daily predictions: {e}")
        return {"success": False, "error": str(e)}
