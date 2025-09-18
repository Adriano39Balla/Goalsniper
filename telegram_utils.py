# telegram_utils.py — safe Telegram bot sender with retries

import os
import time
import logging
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

log = logging.getLogger("telegram")

# Env config
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
SEND_MESSAGES = os.getenv("SEND_MESSAGES", "true").lower() == "true"

API_BASE = f"https://api.telegram.org/bot{BOT_TOKEN}"

def send_telegram(text: str, disable_preview: bool = True) -> bool:
    """
    Send a message to Telegram.
    Retries up to 3 times on errors.
    Returns True if successful, False otherwise.
    """
    if not (BOT_TOKEN and CHAT_ID and SEND_MESSAGES):
        log.info("[TELEGRAM] skipped: not configured or disabled")
        return False

    url = f"{API_BASE}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": disable_preview,
    }

    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, timeout=15)
            if resp.status_code == 200:
                return True
            else:
                log.warning("[TELEGRAM] attempt %d failed: %s", attempt+1, resp.text)
        except Exception as e:
            log.warning("[TELEGRAM] exception on attempt %d: %s", attempt+1, e)
        time.sleep(2 * (attempt + 1))  # exponential backoff
    return False
