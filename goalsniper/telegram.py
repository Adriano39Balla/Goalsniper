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
    await client.post(
        f"{BASE}/editMessageReplyMarkup",
        json={"chat_id": TELEGRAM_CHAT_ID, "message_id": message_id, "reply_markup": reply_markup},
        timeout=30.0,
    )

# ---------- NEW: tip sender with inline feedback buttons ----------

def _feedback_keyboard(tip: dict) -> dict:
    # include enough identifiers so webhook learning can record it
    fid = tip.get("fixtureId") or 0
    market = (tip.get("market") or "NA").upper()
    sel = (tip.get("selection") or "NA").upper()
    ok_cb = f"tip:{fid}:{market}:{sel}:ok"
    bad_cb = f"tip:{fid}:{market}:{sel}:bad"
    return {
        "inline_keyboard": [[
            {"text": "👍 Correct", "callback_data": ok_cb},
            {"text": "👎 Wrong",   "callback_data": bad_cb},
        ]]
    }

async def send_tip_message(client: httpx.AsyncClient, tip: dict) -> int:
    """
    Preferred entrypoint used by scanner.py. Sends a formatted tip with 👍/👎 buttons.
    Returns Telegram message_id.
    """
    text = format_tip_message(tip)
    kb = _feedback_keyboard(tip)
    return await send_telegram_message(client, text, kb)

# aliases so scanner can find any of these names
send_tip = send_tip_message
send_message_with_feedback = send_tip_message
send_message = send_tip_message
push_tip = send_tip_message
send = send_tip_message
