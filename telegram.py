import os
import requests
import logging

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
logger = logging.getLogger(__name__)

def send_telegram_message(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
        res = requests.post(url, data=payload, timeout=10)
        res.raise_for_status()
        logger.info("Telegram message sent successfully.")
    except Exception as e:
        logger.error(f"Telegram API error: {e}")
