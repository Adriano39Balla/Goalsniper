import os
import requests
import logging

# Environment variables
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def send_telegram_message(message: str) -> bool:
    """
    Sends a message to a Telegram chat using the bot API.
    Returns True if successful, False otherwise.
    """
    # Check required credentials
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("Telegram bot token or chat ID not set in environment variables.")
        return False

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}

        res = requests.post(url, data=payload, timeout=10)

        if res.status_code == 200:
            logger.info("Telegram message sent successfully.")
            return True
        else:
            logger.error(f"Telegram send error [{res.status_code}]: {res.text}")
            return False

    except requests.exceptions.RequestException as e:
        logger.error(f"Telegram API request failed: {e}", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"Unexpected error in send_telegram_message: {e}", exc_info=True)
        return False
