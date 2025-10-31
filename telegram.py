# telegram.py

import os
import logging
import html
import requests
from flask import request, abort, jsonify

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, WEBHOOK_SECRET, ADMIN_API_KEY
from core.predictions import production_scan
from core.digest import daily_accuracy_digest
from core.motd import send_match_of_the_day

log = logging.getLogger(__name__)

BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"


def send_telegram(text: str) -> bool:
    """Send a formatted message to Telegram bot channel."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    try:
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        r = requests.post(BASE_URL, json=payload, timeout=5)
        return r.status_code == 200
    except Exception as e:
        log.warning("Telegram send failed: %s", e)
        return False


def escape(text: str) -> str:
    """Escape HTML for Telegram safety."""
    return html.escape(str(text or ""))


def handle_telegram_webhook(secret: str, update: dict = None):
    """Handles incoming Telegram webhook commands."""
    if (WEBHOOK_SECRET or "") != secret:
        abort(403)

    if request.method == "GET":
        return jsonify({"ok": True, "webhook": "ready"})

    update = update or request.get_json(silent=True) or {}
    msg = (update.get("message") or {}).get("text") or ""

    try:
        if msg.startswith("/start"):
            send_telegram("👋 goalsniper bot (FULL AI ENHANCED mode) is online.")
        elif msg.startswith("/digest"):
            daily_accuracy_digest()
        elif msg.startswith("/motd"):
            send_match_of_the_day()
        elif msg.startswith("/scan"):
            parts = msg.split()
            if len(parts) > 1 and ADMIN_API_KEY and parts[1] == ADMIN_API_KEY:
                saved, seen = production_scan()
                send_telegram(f"🔁 Scan done. Saved: {saved}, Live seen: {seen}")
            else:
                send_telegram("🔒 Admin key required.")
    except Exception as e:
        log.warning("Telegram webhook parse error: %s", e)

    return jsonify({"ok": True})
