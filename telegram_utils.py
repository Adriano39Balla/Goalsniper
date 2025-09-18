# telegram_utils.py — safe Telegram bot sender with retries & rate-limit handling

import os
import time
import random
import logging
import requests
import html

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

log = logging.getLogger("telegram")

# Env config
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
SEND_MESSAGES = os.getenv("SEND_MESSAGES", "true").lower() in ("1", "true", "yes")

API_BASE = f"https://api.telegram.org/bot{BOT_TOKEN}" if BOT_TOKEN else ""

# Message constraints
MAX_LEN = 4000  # Telegram hard max 4096 chars; leave some buffer

# Circuit breaker state
_last_fail_ts = 0
FAIL_COOLDOWN_SEC = int(os.getenv("TELEGRAM_FAIL_COOLDOWN_SEC", "60"))

def _should_skip() -> bool:
    global _last_fail_ts
    if not (BOT_TOKEN and CHAT_ID and SEND_MESSAGES):
        return True
    if _last_fail_ts and time.time() - _last_fail_ts < FAIL_COOLDOWN_SEC:
        return True
    return False

def send_telegram(text: str, disable_preview: bool = True) -> bool:
    """
    Send a message to Telegram.
    - Escapes unsafe HTML
    - Truncates if >4096 chars
    - Retries with exponential backoff
    - Handles 429 rate limits
    Returns True if successful, False otherwise.
    """
    global _last_fail_ts

    if _should_skip():
        log.debug("[TELEGRAM] skipped (disabled or cooldown)")
        return False

    url = f"{API_BASE}/sendMessage"
    safe_text = html.escape(str(text or ""))[:MAX_LEN]

    payload = {
        "chat_id": CHAT_ID,
        "text": safe_text,
        "parse_mode": "HTML",  # safer than Markdown
        "disable_web_page_preview": disable_preview,
    }

    backoff = 1.5
    for attempt in range(3):
        try:
            resp = requests.post(
                url, json=payload, timeout=(3, 8)  # connect/read timeouts
            )
            if resp.status_code == 200:
                return True
            if resp.status_code == 429:
                try:
                    retry_after = int(resp.json().get("parameters", {}).get("retry_after", 5))
                except Exception:
                    retry_after = 5
                log.warning("[TELEGRAM] rate limited, sleeping %ds", retry_after)
                time.sleep(retry_after)
                continue
            log.warning("[TELEGRAM] attempt %d failed: %s", attempt + 1, resp.text)
        except Exception as e:
            log.warning("[TELEGRAM] exception on attempt %d: %s", attempt + 1, e)

        sleep_for = backoff + random.uniform(0, 0.5)
        time.sleep(sleep_for)
        backoff *= 2

    _last_fail_ts = time.time()
    return False
