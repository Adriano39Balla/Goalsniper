# telegram.py
import os
import requests
import logging

logger = logging.getLogger(__name__)

# Match the env vars from Render
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_message(message: str) -> bool:
    """Send a message to Telegram."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID missing in environment variables")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"  # Allow HTML formatting
    }

    try:
        res = requests.post(url, json=payload, timeout=10)
        res.raise_for_status()
        logger.info("✅ Telegram message sent successfully.")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Telegram API error: {e}")
        return False
