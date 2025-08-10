import httpx
from .config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

BASE = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

def _render_tip_text(tip: dict) -> str:
    market = (tip.get("market") or "").upper()
    selection = (tip.get("selection") or "").upper().strip()

    if market == "1X2":
        return "Home Win" if selection == "HOME" else "Away Win" if selection == "AWAY" else "Draw"

    if market == "OVER_UNDER_2.5":
        parts = selection.split()
        if len(parts) == 2 and parts[0] in ("OVER", "UNDER"):
            return f"{parts[0].title()} {parts[1]} Goals"
        return selection.title()

    if market == "BTTS":
        return f"BTTS: {'Yes' if selection.startswith('Y') else 'No'}"

    if market == "ASIAN_HANDICAP":
        if selection.startswith("HOME ") or selection.startswith("AWAY "):
            side, line = selection.split(" ", 1)
            return f"Asian Handicap: {side.title()} {line}"
        if "AH 0" in selection:
            return "Asian Handicap: 0 (Draw No Bet)"
        return f"Asian Handicap: {selection.title()}"

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

async def send_telegram_message(client: httpx.AsyncClient, text: str, reply_markup: dict | None = None) -> int:
    r = await client.post(
        f"{BASE}/sendMessage",
        json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
            "reply_markup": reply_markup or None,
        },
        timeout=30.0,
    )
    r.raise_for_status()
    data = r.json()
    return int(((data or {}).get("result") or {}).get("message_id") or 0)

async def edit_message_markup(client: httpx.AsyncClient, message_id: int, reply_markup: dict):
    r = await client.post(
        f"{BASE}/editMessageReplyMarkup",
        json={"chat_id": TELEGRAM_CHAT_ID, "message_id": message_id, "reply_markup": reply_markup},
        timeout=30.0,
    )
    r.raise_for_status()

# ---------- Feedback keyboards ----------

def feedback_keyboard_by_tip_id(tip_id: int) -> dict:
    """Build inline keyboard with fb:<tip_id>:<1|0> so webhook learning works."""
    return {
        "inline_keyboard": [[
            {"text": "👍 Correct", "callback_data": f"fb:{int(tip_id)}:1"},
            {"text": "👎 Wrong",   "callback_data": f"fb:{int(tip_id)}:0"},
        ]]
    }

# ---------- Public helpers used by scanner ----------

async def send_tip_plain(client: httpx.AsyncClient, tip: dict) -> int:
    """Send the tip message WITHOUT buttons (we don't have tip_id yet)."""
    text = format_tip_message(tip)
    return await send_telegram_message(client, text, None)

async def attach_feedback_buttons(client: httpx.AsyncClient, message_id: int, tip_id: int) -> None:
    """After inserting into DB, attach the correct fb buttons."""
    kb = feedback_keyboard_by_tip_id(int(tip_id))
    await edit_message_markup(client, int(message_id), kb)

# Backwards-compat alias (if something still calls it)
async def send_tip_message(client: httpx.AsyncClient, tip: dict) -> int:
    # Not used in the new flow, but keep as plain send for compatibility
    return await send_tip_plain(client, tip)

# Other aliases for scanner discovery
send_tip = send_tip_message
send_message_with_feedback = send_tip_message
send_message = send_tip_message
push_tip = send_tip_message
send = send_tip_message
