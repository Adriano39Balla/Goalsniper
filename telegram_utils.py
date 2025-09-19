# file: telegram_utils.py — safe Telegram bot sender with retries, rate-limit handling, chunking

import os
import time
import random
import logging
import requests
import html
import re
from typing import List

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

log = logging.getLogger("telegram")

# Env config
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_IDS = [c.strip() for c in os.getenv("TELEGRAM_CHAT_ID", "").split(",") if c.strip()]
THREAD_ID = os.getenv("TELEGRAM_THREAD_ID")  # optional: topics in supergroups
SEND_MESSAGES = os.getenv("SEND_MESSAGES", "true").lower() in ("1", "true", "yes")

# Parse mode: HTML (default), Markdown, MarkdownV2, or none
PARSE_MODE = os.getenv("TELEGRAM_PARSE_MODE", "HTML").strip()

API_BASE = f"https://api.telegram.org/bot{BOT_TOKEN}" if BOT_TOKEN else ""

# Message constraints
TG_HARD_MAX = 4096
MAX_LEN = min(int(os.getenv("TELEGRAM_MAX_LEN", TG_HARD_MAX)), TG_HARD_MAX)

# Circuit breaker state
_last_fail_ts = 0
FAIL_COOLDOWN_SEC = int(os.getenv("TELEGRAM_FAIL_COOLDOWN_SEC", "60"))

# HTTP session with retries
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
    global _last_fail_ts
    if not (BOT_TOKEN and CHAT_IDS and SEND_MESSAGES):
        return True
    if _last_fail_ts and time.time() - _last_fail_ts < FAIL_COOLDOWN_SEC:
        return True
    return False

# ───────── Parse-mode helpers ─────────

_MD_V2_CHARS = r'[_*[\]()~`>#+\-=|{}.!]'  # Telegram MarkdownV2 reserved chars

def _escape_md(text: str) -> str:
    # Legacy Markdown (not V2): escape only _ * ` [ ]
    return re.sub(r'([_*`$begin:math:display$$end:math:display$])', r'\\\1', text)

def _escape_md_v2(text: str) -> str:
    return re.sub(r'(' + _MD_V2_CHARS + r')', r'\\\1', text)

def _prepare_text(text: str) -> str:
    """
    Return body string according to PARSE_MODE.
    - HTML: escape all HTML (safe, no formatting unless you pass tags yourself)
    - Markdown/MarkdownV2: escape reserved chars so your *bold* etc. render safely
    - none: send as plain text
    """
    t = str(text or "")
    if PARSE_MODE.lower() == "html":
        return html.escape(t)[:MAX_LEN]
    if PARSE_MODE.lower() == "markdownv2":
        return _escape_md_v2(t)[:MAX_LEN]
    if PARSE_MODE.lower() == "markdown":
        return _escape_md(t)[:MAX_LEN]
    # none
    return t[:MAX_LEN]

# ───────── Chunking ─────────

def _split_message(s: str, limit: int) -> List[str]:
    """Split into safe chunks <= limit, preferring paragraph and line breaks."""
    if len(s) <= limit:
        return [s]

    chunks: List[str] = []
    cur = s
    while len(cur) > limit:
        cut = cur.rfind("\n\n", 0, limit)
        if cut == -1:
            cut = cur.rfind("\n", 0, limit)
        if cut == -1:
            cut = cur.rfind(" ", 0, limit)
        if cut == -1:
            cut = limit  # hard split
        chunks.append(cur[:cut].rstrip())
        cur = cur[cut:].lstrip()
    if cur:
        chunks.append(cur)
    return chunks

# ───────── Sender ─────────

def send_telegram(text: str, disable_preview: bool = True) -> bool:
    """
    Send a message to Telegram.
    - Respects TELEGRAM_PARSE_MODE: HTML (default), Markdown, MarkdownV2, none
    - Escapes/normalizes content for the chosen parse mode
    - Splits long messages safely (<=4096 chars)
    - Retries with exponential backoff and honors 429 rate limits
    - Supports multiple chat IDs and optional THREAD_ID (topics)
    Returns True if at least one send succeeds.
    """
    global _last_fail_ts

    if _should_skip():
        log.debug("[TELEGRAM] skipped (disabled, missing config, or cooldown)")
        return False

    url = f"{API_BASE}/sendMessage"
    body = _prepare_text(text)
    parts = _split_message(body, MAX_LEN)

    succeeded = False
    for chat_id in CHAT_IDS:
        for idx, part in enumerate(parts, 1):
            payload = {
                "chat_id": chat_id,
                "text": part,
                "disable_web_page_preview": disable_preview,
            }
            if PARSE_MODE.lower() in ("html", "markdown", "markdownv2"):
                payload["parse_mode"] = "MarkdownV2" if PARSE_MODE.lower() == "markdownv2" else PARSE_MODE

            if THREAD_ID:
                try:
                    payload["message_thread_id"] = int(THREAD_ID)
                except Exception:
                    pass

            backoff = 1.5
            for attempt in range(3):
                try:
                    resp = _session.post(url, json=payload, timeout=(3, 10))
                    if resp.status_code == 200:
                        succeeded = True
                        break
                    if resp.status_code == 429:
                        # Bot API returns {"parameters":{"retry_after":N}}
                        try:
                            retry_after = int(resp.json().get("parameters", {}).get("retry_after", 5))
                        except Exception:
                            retry_after = 5
                        log.warning("[TELEGRAM] rate limited, sleeping %ds", retry_after)
                        time.sleep(retry_after)
                        continue
                    log.warning("[TELEGRAM] attempt %d/%d failed (%s): %s",
                                attempt + 1, 3, resp.status_code, resp.text[:200])
                except Exception as e:
                    log.warning("[TELEGRAM] exception on attempt %d/%d: %s", attempt + 1, 3, e)

                time.sleep(backoff + random.uniform(0, 0.5))
                backoff *= 2

            # light pacing between chunks to avoid flood (esp. in groups)
            if idx < len(parts):
                time.sleep(0.3 + random.uniform(0, 0.2))

    if not succeeded:
        _last_fail_ts = time.time()
    return succeeded
