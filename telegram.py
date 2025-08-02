# /app/telegram.py
import os
import requests
import logging

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
logger = logging.getLogger(__name__)

def send_telegram_message(message):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": message}
        res = requests.post(url, data=payload, timeout=10)
        if res.status_code != 200:
            logger.error(f"Telegram send error: {res.text}")
    except Exception as e:
        logger.error(f"Telegram API error: {e}")
