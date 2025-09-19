# file: telegram_utils.py — safe Telegram bot sender with retries, rate-limit handling, chunking

import os
import time
import random
import logging
import requests
import html
import re
from typing import List, Optional, Dict, Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

log = logging.getLogger("telegram")

# ───────── Env config ─────────
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_IDS = [c.strip() for c in os.getenv("TELEGRAM_CHAT_ID", "").split(",") if c.strip()]
THREAD_ID = os.getenv("TELEGRAM_THREAD_ID")  # optional topic/thread id (int)
SEND_MESSAGES = os.getenv("SEND_MESSAGES", "true").lower() in ("1", "true", "yes")

PARSE_MODE = os.getenv("TELEGRAM_PARSE_MODE", "HTML").strip()  # HTML | Markdown | MarkdownV2 | none
ALLOW_HTML_TAGS = os.getenv("TELEGRAM_ALLOW_HTML_TAGS", "0").lower() in ("1", "true", "yes")
DISABLE_NOTIFICATION = os.getenv("TELEGRAM_SILENT", "0").lower() in ("1", "true", "yes")  # silent push

API_BASE = f"https://api.telegram.org/bot{BOT_TOKEN}" if BOT_TOKEN else ""

TG_HARD_MAX = 4096
MAX_LEN = min(int(os.getenv("TELEGRAM_MAX_LEN", TG_HARD_MAX)), TG_HARD_MAX)

# ───────── Circuit breaker state ─────────
_last_fail_ts = 0.0
# Cooldowns for generic failures and auth failures (401/403)
FAIL_COOLDOWN_SEC = int(os.getenv("TELEGRAM_FAIL_COOLDOWN_SEC", "60"))
AUTH_FAIL_COOLDOWN_SEC = int(os.getenv("TELEGRAM_AUTH_FAIL_COOLDOWN_SEC", "600"))

# ───────── HTTP session with retries ─────────
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

_session = requests.Session()
_retry = Retry(
    total=3,
    connect=3,
    read=3,
    backoff_factor=0.7,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=frozenset(["POST"]),
    respect_retry_after_header=True,
    raise_on_status=False,
)
_adapter = HTTPAdapter(max_retries=_retry, pool_connections=32, pool_maxsize=64)
_session.mount("https://", _adapter)
_session.mount("http://", _adapter)

def _should_skip() -> bool:
    """Global gate based on config and recent hard failures."""
    global _last_fail_ts
    if not SEND_MESSAGES:
        return True
    if not (BOT_TOKEN and CHAT_IDS):
        # Only log once on import to avoid spam
        return True
    if _last_fail_ts and (time.time() - _last_fail_ts) < FAIL_COOLDOWN_SEC:
        return True
    return False

# ───────── Parse-mode helpers ─────────
# Telegram escaping rules:
#  - MarkdownV2 reserves: _ * [ ] ( ) ~ ` > # + - = | { } . !
#  - Markdown legacy: _ * [ ] ( )
_MD_V2_CHARS = r'[_\*$begin:math:display$$end:math:display$$begin:math:text$$end:math:text$~`>#+\-=|{}\.!]'
_MD_LEGACY_CHARS = r'[_\*\[\]\(\)]'

def _escape_md(text: str) -> str:
    return re.sub(r'(' + _MD_LEGACY_CHARS + r')', r'\\\1', text)

def _escape_md_v2(text: str) -> str:
    return re.sub(r'(' + _MD_V2_CHARS + r')', r'\\\1', text)

def _prepare_text(text: str) -> str:
    """
    Return body string according to PARSE_MODE.
    - HTML: escapes unless ALLOW_HTML_TAGS=1
    - Markdown / MarkdownV2: escapes reserved chars
    - none: plain text
    """
    t = str(text or "")
    mode = PARSE_MODE.lower()
    if mode == "html":
        return (t if ALLOW_HTML_TAGS else html.escape(t))[:MAX_LEN]
    if mode == "markdownv2":
        return _escape_md_v2(t)[:MAX_LEN]
    if mode == "markdown":
        return _escape_md(t)[:MAX_LEN]
    return t[:MAX_LEN]

# ───────── Chunking ─────────
def _split_message(s: str, limit: int) -> List[str]:
    """Split into chunks <= limit, preferring paragraph/line/space boundaries."""
    if len(s) <= limit:
        return [s]
    chunks: List[str] = []
    cur = s
    while len(cur) > limit:
        cut = cur.rfind("\n\n", 0, limit)
        if cut == -1: cut = cur.rfind("\n", 0, limit)
        if cut == -1: cut = cur.rfind(" ", 0, limit)
        if cut == -1: cut = limit
        chunks.append(cur[:cut].rstrip())
        cur = cur[cut:].lstrip()
    if cur:
        chunks.append(cur)
    return chunks

# ───────── Sender ─────────
def _post_send_message(payload: Dict[str, Any]) -> requests.Response:
    """Low-level POST with last-resort handling for 429 Retry-After."""
    url = f"{API_BASE}/sendMessage"
    r = _session.post(url, json=payload, timeout=(3, 10))
    if r.status_code == 429:
        # Respect Retry-After if present (urllib3 already retries; this is last attempt)
        try:
            ra = int(r.headers.get("Retry-After", "0"))
            if ra <= 0:
                ra = int(r.json().get("parameters", {}).get("retry_after", 0))
        except Exception:
            ra = 0
        if ra > 0:
            sleep_s = min(ra, 15)
            log.warning("[TELEGRAM] 429 rate limited, sleeping %ds then retry", sleep_s)
            time.sleep(sleep_s)
            r = _session.post(url, json=payload, timeout=(3, 10))
    return r

def send_telegram(text: str, disable_preview: bool = True, disable_notification: Optional[bool] = None) -> bool:
    """
    Send a message to Telegram.
      - Respects TELEGRAM_PARSE_MODE (HTML / Markdown / MarkdownV2 / none)
      - Escapes according to parse mode
      - Splits long messages safely (<=4096 chars)
      - Retries with backoff and honors 429 Retry-After
      - Supports multiple chat IDs and optional THREAD_ID (topics)
    Returns True if at least one send succeeds.
    """
    global _last_fail_ts

    if _should_skip():
        log.debug("[TELEGRAM] skipped (disabled, missing config, or cooldown)")
        return False

    mode = PARSE_MODE if PARSE_MODE.lower() in ("html", "markdown", "markdownv2") else None
    body = _prepare_text(text)
    parts = _split_message(body, MAX_LEN)

    succeeded_any = False
    hard_error = False

    for chat_id in CHAT_IDS:
        for idx, part in enumerate(parts, 1):
            payload: Dict[str, Any] = {
                "chat_id": chat_id,
                "text": part,
                "disable_web_page_preview": disable_preview,
                "disable_notification": DISABLE_NOTIFICATION if disable_notification is None else bool(disable_notification),
            }
            if mode:
                payload["parse_mode"] = "MarkdownV2" if mode.lower() == "markdownv2" else mode

            if THREAD_ID:
                try:
                    payload["message_thread_id"] = int(THREAD_ID)
                except Exception:
                    pass

            backoff = 1.5
            for attempt in range(3):
                try:
                    resp = _post_send_message(payload)
                    code = resp.status_code

                    if code == 200:
                        if not succeeded_any:
                            log.debug("[TELEGRAM] delivered to chat_id=%s (len=%d)", chat_id, len(part))
                        succeeded_any = True
                        break

                    # Handle auth errors more strictly (bad token or blocked bot)
                    if code in (401, 403):
                        hard_error = True
                        log.error("[TELEGRAM] auth/perm error %s: %s", code, resp.text[:200])
                        break

                    # Other non-OK
                    log.warning("[TELEGRAM] attempt %d/%d failed (%s): %s",
                                attempt + 1, 3, code, resp.text[:200])

                except Exception as e:
                    log.warning("[TELEGRAM] exception on attempt %d/%d: %s", attempt + 1, 3, e)

                time.sleep(backoff + random.uniform(0, 0.5))
                backoff *= 2

            if idx < len(parts):
                time.sleep(0.25 + random.uniform(0, 0.2))

    if not succeeded_any:
        # Trip circuit breaker
        _last_fail_ts = time.time()
        if hard_error:
            # Extend cooldown substantially for auth problems
            _last_fail_ts -= (FAIL_COOLDOWN_SEC - min(FAIL_COOLDOWN_SEC, AUTH_FAIL_COOLDOWN_SEC))
    return succeeded_any

def send_telegram_safe(text: str, **kwargs) -> bool:
    """Safe wrapper that never raises; logs and returns False on failure."""
    try:
        return send_telegram(text, **kwargs)
    except Exception as e:
        log.warning("[TELEGRAM] send_telegram_safe failed: %s", e)
        return False
