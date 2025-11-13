import requests
import time
from typing import Optional
from app.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID


def telegram_url():
    return f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"


# --------------------------------------------------------
# BASIC SEND MESSAGE (UPGRADED)
# --------------------------------------------------------

def send_telegram(text: str, retries: int = 3) -> bool:
    """
    Sends a Telegram message with retry handling, including:
    - rate limit handling (429)
    - bot blocked (403)
    - safe logging
    """

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[TELEGRAM] Missing bot token or chat_id")
        return False

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    url = telegram_url()

    for attempt in range(1, retries + 1):

        try:
            r = requests.post(url, data=payload, timeout=5)

            # success
            if r.status_code == 200:
                return True

            # rate limit
            if r.status_code == 429:
                retry_after = r.json().get("parameters", {}).get("retry_after", 1)
                print(f"[TELEGRAM] Rate limited. Waiting {retry_after}s...")
                time.sleep(retry_after)
                continue

            # bot blocked
            if r.status_code == 403:
                print("[TELEGRAM] Bot blocked or chat_id invalid.")
                return False

            # other errors
            print(f"[TELEGRAM] HTTP {r.status_code}: {r.text}")

        except Exception as e:
            print(f"[TELEGRAM] Exception: {e}")

        time.sleep(0.5)

    return False


# --------------------------------------------------------
# FORMAT A SINGLE PREDICTION
# --------------------------------------------------------

def format_prediction(pred) -> str:

    return (
        f"⚽ <b>GOALSNIPER AI</b>\n"
        f"📌 <b>Fixture ID:</b> {pred.fixture_id}\n"
        f"⏱ <b>Minute:</b> {pred.aux.get('minute', 0)}'\n\n"
        f"🎯 <b>Market:</b> {pred.market}\n"
        f"🟩 <b>Bet:</b> {pred.selection.upper()}\n\n"
        f"📈 <b>Probability:</b> {pred.prob*100:.1f}%\n"
        f"💸 <b>Odds:</b> {pred.odds:.2f}\n"
        f"📊 <b>EV:</b> {pred.ev*100:.1f}%\n"
    )


# --------------------------------------------------------
# SEND MULTIPLE PREDICTIONS
# --------------------------------------------------------

def send_predictions(pred_list):
    count = 0
    for pred in pred_list:
        msg = format_prediction(pred)
        if send_telegram(msg):
            count += 1
    return count
