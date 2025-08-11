import re
import json
import httpx
from typing import Optional, Dict

from .config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from .logger import log, warn

BASE = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

# ---- helpers ----------------------------------------------------------------

_MAX_LEN = 4096
_CTRL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")  # strip non-printable (keep \n)

def _sanitize_html(text: str) -> str:
    """Keep it simple: escape only the risky chars we actually use with <b> tags."""
    text = _CTRL_RE.sub("", text or "")
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # We sometimes add <b>…</b> ourselves, so allow those back:
    text = text.replace("&lt;b&gt;", "<b>").replace("&lt;/b&gt;", "</b>")
    # Telegram 4096 char limit
    if len(text) > _MAX_LEN:
        text = text[: _MAX_LEN - 1] + "…"
    return text

def _safe_reply_markup(markup: Optional[Dict]) -> Optional[Dict]:
    # Keep callback_data below Telegram’s 64-byte limit and ensure it’s JSON-safe
    if not markup:
        return None
    try:
        js = json.dumps(markup)
        if len(js) > 800:  # extremely defensive; inline keyboards are tiny for us
            warn("reply_markup too large; truncating")
            return None
    except Exception:
        return None
    return markup

async def _post_json(client: httpx.AsyncClient, method: str, payload: Dict) -> Dict:
    r = await client.post(f"{BASE}/{method}", json=payload, timeout=30.0)
    # Raise to enter the fallback path
    r.raise_for_status()
    data = r.json()
    if not data.get("ok"):
        raise httpx.HTTPStatusError(str(data), request=r.request, response=r)
    return data

def _extract_telegram_error(e: Exception) -> str:
    try:
        resp = getattr(e, "response", None)
        if resp is not None:
            js = resp.json()
            return f"{js.get('description') or js}"
    except Exception:
        pass
    return str(e)

# ---- public API --------------------------------------------------------------

def _render_tip_text(tip: dict) -> str:
    market = (tip.get("market") or "").upper()
    selection = (tip.get("selection") or "").upper().strip()

    if market == "1X2":
        return "Home Win" if selection == "HOME" else "Away Win" if selection == "AWAY" else "Draw"

    if market in ("OVER_UNDER_2.5", "OVER/UNDER"):
        parts = selection.split()
        if len(parts) == 2 and parts[0] in ("OVER", "UNDER"):
            return f"{parts[0].title()} {parts[1]} Goals"
        return selection.title()

    if market == "BTTS":
        return f"BTTS: {'Yes' if selection.startswith('Y') else 'No'}"

    if market == "1ST_HALF_OU":
        parts = selection.split()
        if len(parts) == 2 and parts[0] in ("OVER", "UNDER"):
            return f"1st Half {parts[0].title()} {parts[1]}"
        return f"1st Half {selection.title()}"

    return selection.title()

def format_tip_message(tip: dict) -> str:
    header = "⚽️ <b>New Tip!</b>"
    match_line = f"🏟️ <b>Match:</b> {tip.get('home','Home')} vs {tip.get('away','Away')}"
    tip_line = f"📊 <b>Tip:</b> {_render_tip_text(tip)}"
    conf_pct = f"{round(float(tip.get('confidence', 0) * 100))}%"
    confidence_line = f"📈 <b>Confidence:</b> {conf_pct}"
    league_name = tip.get("leagueName") or ""
    country = tip.get("country") or ""
    league_label = f"{country} - {league_name}" if country and league_name else (league_name or country or "Unknown")
    league_line = f"🏆 <b>League:</b> {league_label}"
    return "\n".join([header, match_line, tip_line, confidence_line, league_line])

async def send_text(client: httpx.AsyncClient, text: str, reply_markup: dict | None = None) -> int:
    """
    Safe sender:
      1) Try HTML parse_mode with sanitized text
      2) On 400, retry as plain text (no parse_mode)
      3) If still 400, log full payload sample and give up gracefully
    Returns the Telegram message_id (0 on hard failure).
    """
    sanitized = _sanitize_html(text)
    markup = _safe_reply_markup(reply_markup)

    payload_html = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": sanitized,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
        "reply_markup": markup or None,
    }

    try:
        data = await _post_json(client, "sendMessage", payload_html)
        return int(((data or {}).get("result") or {}).get("message_id") or 0)
    except httpx.HTTPStatusError as e:
        # 400 often means HTML/formatting issue → fall back to plain
        warn("Telegram 400 (HTML), retrying plain:", _extract_telegram_error(e))
        try:
            payload_plain = {
                "chat_id": TELEGRAM_CHAT_ID,
                "text": _CTRL_RE.sub("", (text or ""))[:_MAX_LEN],
                "disable_web_page_preview": True,
                "reply_markup": markup or None,
            }
            data = await _post_json(client, "sendMessage", payload_plain)
            return int(((data or {}).get("result") or {}).get("message_id") or 0)
        except Exception as e2:
            err = _extract_telegram_error(e2)
            # Log the offending content (truncated) so we can diagnose
            sample = (text or "")[:240].replace("\n", " | ")
            warn(f"Telegram send failed (plain). err={err} sample='{sample}…'")
            return 0
    except Exception as e:
        warn("Telegram send failed:", _extract_telegram_error(e))
        return 0

# --- buttons (feedback) -------------------------------------------------------

def _feedback_keyboard_for_tip_id(tip_id: int) -> dict:
    # Callback format expected by /telegram/webhook: fb:<tip_id>:<1|0>
    return {
        "inline_keyboard": [[
            {"text": "👍 Correct", "callback_data": f"fb:{int(tip_id)}:1"},
            {"text": "👎 Wrong",   "callback_data": f"fb:{int(tip_id)}:0"},
        ]]
    }

async def attach_feedback_buttons(client: httpx.AsyncClient, message_id: int, tip_id: int):
    kb = _feedback_keyboard_for_tip_id(tip_id)
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "message_id": int(message_id),
        "reply_markup": kb,
    }
    try:
        await _post_json(client, "editMessageReplyMarkup", payload)
    except Exception as e:
        warn("editMessageReplyMarkup failed:", _extract_telegram_error(e))

# --- scanner-facing API -------------------------------------------------------

async def send_tip_plain(client: httpx.AsyncClient, tip: dict) -> int:
    """
    Used by scanner.py to send a tip WITHOUT buttons first. Buttons are attached
    afterwards with the DB tip_id to wire feedback learning.
    """
    text = format_tip_message(tip)
    msg_id = await send_text(client, text)
    if not msg_id:
        # Keep scanning going, but log for analysis
        warn("send_tip_plain: message_id=0 (Telegram rejected message)")
    return msg_id
