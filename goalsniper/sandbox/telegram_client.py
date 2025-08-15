import logging
import requests
from . import config  # if you place files flat (same folder), change to: import config

log = logging.getLogger(__name__)

def send_text(text, parse_mode=None):
    url = f"https://api.telegram.org/bot{config.TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": config.CHAT_ID, "text": text, "disable_web_page_preview": True}
    if parse_mode:
        payload["parse_mode"] = parse_mode
    r = requests.post(url, json=payload, timeout=15)
    try:
        data = r.json()
    except Exception:
        data = {"ok": False, "error": "non-json response", "body": r.text}
    log.info("sendMessage status=%s ok=%s desc=%s", r.status_code, data.get("ok"), data.get("description"))
    if not data.get("ok"):
        raise RuntimeError(f"Telegram error: {data.get('description')} (status {r.status_code})")
    return data
