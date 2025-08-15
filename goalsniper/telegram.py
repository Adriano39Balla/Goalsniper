import re
import json
import httpx
import asyncio
import random
from typing import Optional, Dict
from datetime import datetime

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
    # allow <b>…</b> we add ourselves
    text = text.replace("&lt;b&gt;", "<b>").replace("&lt;/b&gt;", "</b>")
    if len(text) > _MAX_LEN:
        text = text[: _MAX_LEN - 1] + "…"
    return text

def _safe_reply_markup(markup: Optional[Dict]) -> Optional[Dict]:
    if not markup or not isinstance(markup, dict):
        return None
    try:
        js = json.dumps(markup)
        if len(js) > 800:
            warn("reply_markup too large; dropping it")
            return None
    except Exception:
        return None
    return markup

async def _post_json(client: httpx.AsyncClient, method: str, payload: Dict) -> Dict:
    r = await client.post(f"{BASE}/{method}", json=payload, timeout=30.0)
    if r.status_code == 429:
        ra = r.headers.get("Retry-After")
        delay = float(ra) if (ra and ra.isdigit()) else 1.0
        await asyncio.sleep(delay + random.uniform(0, 0.25))
        r = await client.post(f"{BASE}/{method}", json=payload, timeout=30.0)
    if r.status_code == 400:
        try:
            js = r.json()
            desc = js.get("description") or js
            raise httpx.HTTPStatusError(str(desc), request=r.request, response=r)
        except Exception:
            r.raise_for_status()
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

def _nice_market_line(tip: dict) -> str:
    market = (tip.get("market") or "").upper()
    selection = (tip.get("selection") or "").upper().strip()

    if market == "1X2":
        label = "Home Win" if selection == "HOME" else ("Away Win" if selection == "AWAY" else "Draw")
        return f"🎯 <b>Pick:</b> {label}"

    if market in ("OVER_UNDER_2.5", "OVER/UNDER"):
        parts = selection.split()
        if len(parts) == 2 and parts[0] in ("OVER", "UNDER"):
            return f"🎯 <b>Pick:</b> {parts[0].title()} {parts[1]} Goals"
        return f"🎯 <b>Pick:</b> {selection.title()}"

    if market == "BTTS":
        return f"🎯 <b>Pick:</b> BTTS — {'Yes' if selection.startswith('Y') else 'No'}"

    if market in ("1ST_HALF_OU", "OVER_UNDER_1H"):
        parts = selection.split()
        if len(parts) == 2 and parts[0] in ("OVER", "UNDER"):
            return f"🎯 <b>Pick:</b> 1st Half {parts[0].title()} {parts[1]}"
        return f"🎯 <b>Pick:</b> 1st Half {selection.title()}"

    return f"🎯 <b>Pick:</b> {selection.title()}"

def _confidence_bar(conf: float) -> str:
    # 10 chars bar: ▒ to █
    blocks = int(max(0.0, min(1.0, conf)) * 10)
    return "█" * blocks + "░" * (10 - blocks)

def _fmt_kickoff(tip: dict) -> str:
    # tip["kickOff"] is ISO; show HH:MM UTC and date
    ko = tip.get("kickOff")
    if not ko:
        return ""
    try:
        dt = datetime.fromisoformat(ko.replace("Z", "+00:00"))
        return f"{dt.strftime('%Y-%m-%d %H:%M')} UTC"
    except Exception:
        return str(ko)

# Build a fake tip in the same structure as your real tips
fake_tip = {
    "fixtureId": 999999,
    "leagueId": 12345,
    "leagueName": "Test League",
    "homeTeam": "Python FC",
    "awayTeam": "Telegram United",
    "market": "1X2",           # match result market
    "selection": "HOME",       # HOME / DRAW / AWAY
    "probability": 0.85,       # raw probability
    "confidence": 0.70,        # must be float, >= min conf to send
    "note": "This is a test match to verify Telegram tip delivery."
}

def format_tip_message(tip: dict) -> str:
    home = tip.get("home", "Home")
    away = tip.get("away", "Away")
    league_name = tip.get("leagueName") or ""
    country = tip.get("country") or ""
    league_label = f"{country} — {league_name}" if country and league_name else (league_name or country or "Unknown")

    conf = float(tip.get("confidence", 0.0))
    conf_pct = round(conf * 100)
    conf_line = f"📈 <b>Confidence:</b> {conf_pct}%  [{_confidence_bar(conf)}]"

    live = bool(tip.get("live"))
    minute = int(tip.get("minute") or 0)
    score  = str(tip.get("score") or "0-0")
    status_line = f"⏱️ <b>Live:</b> {minute}'  |  <b>Score:</b> {score}" if live else f"🕒 <b>Kickoff:</b> {_fmt_kickoff(tip)}"

    lines = [
        "⚽️ <b>Goalsniper Tip</b>",
        f"🏟️ <b>Match:</b> {home} vs {away}",
        f"🏆 <b>League:</b> {league_label}",
        _nice_market_line(tip),
        conf_line,
        status_line,
        "",
        "Tap below to grade the tip after the match 👇",
    ]
    return "\n".join(lines)

async def send_text(client: httpx.AsyncClient, text: str, reply_markup: dict | None = None) -> int:
    """
    Safe sender:
      1) Try HTML parse_mode with sanitized text
      2) On 400, retry as plain text (no parse_mode)
      3) If still 400, log and return 0
    """
    sanitized = _sanitize_html(text)
    markup = _safe_reply_markup(reply_markup)

    payload_html = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": sanitized,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    if markup is not None:
        payload_html["reply_markup"] = markup

    try:
        data = await _post_json(client, "sendMessage", payload_html)
        return int(((data or {}).get("result") or {}).get("message_id") or 0)
    except httpx.HTTPStatusError as e:
        warn("Telegram 400 (HTML), retrying plain:", _extract_telegram_error(e))
        try:
            payload_plain = {
                "chat_id": TELEGRAM_CHAT_ID,
                "text": _CTRL_RE.sub("", (text or ""))[:_MAX_LEN],
                "disable_web_page_preview": True,
            }
            if markup is not None:
                payload_plain["reply_markup"] = markup
            data = await _post_json(client, "sendMessage", payload_plain)
            return int(((data or {}).get("result") or {}).get("message_id") or 0)
        except Exception as e2:
            err = _extract_telegram_error(e2)
            sample = (text or "")[:240].replace("\n", " | ")
            warn(f"Telegram send failed (plain). err={err} sample='{sample}…'")
            return 0
    except Exception as e:
        warn("Telegram send failed:", _extract_telegram_error(e))
        return 0

# --- buttons (feedback) -------------------------------------------------------

def _feedback_keyboard_for_tip_id(tip_id: int) -> dict:
    return {
        "inline_keyboard": [[
            {"text": "👍 Correct", "callback_data": f"fb:{int(tip_id)}:1"},
            {"text": "👎 Wrong",   "callback_data": f"fb:{int(tip_id)}:0"},
        ]]
    }

async def attach_feedback_buttons(client: httpx.AsyncClient, message_id: int, tip_id: int):
    if not message_id:
        warn("attach_feedback_buttons: skip (message_id=0)")
        return
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

async def answer_callback_query(client: httpx.AsyncClient, callback_query_id: str, text: str = "Thanks!"):
    try:
        await _post_json(client, "answerCallbackQuery", {
            "callback_query_id": callback_query_id,
            "text": text,
            "show_alert": False
        })
    except Exception as e:
        warn("answerCallbackQuery failed:", _extract_telegram_error(e))

# --- scanner-facing API -------------------------------------------------------

async def send_tip_plain(client: httpx.AsyncClient, tip: dict) -> int:
    text = format_tip_message(tip)
    msg_id = await send_text(client, text)
    if not msg_id:
        warn("send_tip_plain: message_id=0 (Telegram rejected message)")
    return msg_id

# --- MOTD (Match of the Day) --------------------------------------------------

def format_motd_message(tip: dict) -> str:
    """
    Pretty MOTD message. Reuses the same fields a normal tip uses,
    but adds the trophy header and a couple of extra numbers.
    """
    base = format_tip_message(tip).splitlines()
    # Replace the header line
    if base and base[0].startswith("⚽️"):
        base[0] = "🏆 <b>Goalsniper — Match of the Day</b>"
    # Add expected goals if available
    xg = tip.get("expectedGoals")
    if xg is not None:
        for i, line in enumerate(base):
            if line.startswith("🕒 ") or line.startswith("⏱️ "):
                base.insert(i, f"🔮 <b>Model xG:</b> {xg:.2f}")
                break
    return "\n".join(base)

async def send_motd(client: httpx.AsyncClient, tip: dict) -> int:
    """
    Sends a MOTD message (no buttons; we want a clean highlight).
    Returns Telegram message_id or 0 on failure.
    """
    text = format_motd_message(tip)
    return await send_text(client, text)
