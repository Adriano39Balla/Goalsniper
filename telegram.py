# telegram.py
import os
import requests
import logging

logger = logging.getLogger(__name__)

# Match the env vars from Render
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_message(message: str):
    """Send a message to Telegram with full logging."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }

    try:
        logging.info(f"Sending to Telegram: {message}")
        response = requests.post(url, json=payload)
        logging.info(f"Telegram status: {response.status_code}")
        logging.info(f"Telegram response: {response.text}")

        if response.status_code != 200:
            raise Exception(f"Telegram send failed: {response.text}")

    except Exception as e:
        logging.error(f"Error sending Telegram message: {e}")
        raise  # So we can see the failure in logs
