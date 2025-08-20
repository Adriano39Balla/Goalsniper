# app/telegram.py
import os
import logging
import requests
from typing import Optional

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}" if TELEGRAM_BOT_TOKEN else None


def _post(endpoint: str, payload: dict) -> Optional[dict]:
    if not BASE_URL:
        logging.warning("[TELEGRAM] BOT_TOKEN not set; skipping send.")
        return None
    try:
        url = f"{BASE_URL}/{endpoint}"
        res = requests.post(url, json=payload, timeout=10)
        if not res.ok:
            logging.error(f"[TELEGRAM] send failed {res.status_code}: {res.text}")
            return None
        return res.json()
    except Exception as e:
        logging.exception(f"[TELEGRAM] post error: {e}")
        return None


def send_telegram(msg: str, chat_id: Optional[str] = None, disable_preview: bool = True) -> bool:
    """
    Send a message to Telegram chat.
    """
    if not BASE_URL or not TELEGRAM_CHAT_ID:
        logging.debug("[TELEGRAM] disabled; msg=%s", msg)
        return False
    payload = {
        "chat_id": chat_id or TELEGRAM_CHAT_ID,
        "text": msg,
        "disable_web_page_preview": disable_preview,
        "parse_mode": "HTML",
    }
    resp = _post("sendMessage", payload)
    return bool(resp)


def answer_callback(callback_query_id: str, text: str = "", show_alert: bool = False) -> bool:
    """
    Answer a callback query (e.g. from inline button presses).
    """
    if not BASE_URL:
        return False
    payload = {
        "callback_query_id": callback_query_id,
        "text": text,
        "show_alert": show_alert,
    }
    resp = _post("answerCallbackQuery", payload)
    return bool(resp)
