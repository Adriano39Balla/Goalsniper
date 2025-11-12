import requests
import time
from typing import Optional
from app.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

TG_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"


# --------------------------------------------------------
# BASIC SEND MESSAGE
# --------------------------------------------------------

def send_telegram(text: str, retries: int = 2) -> bool:
    """
    Sends a Telegram message with retry handling.
    Never throws; always returns True/False.
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

    for attempt in range(1, retries + 2):
        try:
            r = requests.post(TG_URL, data=payload, timeout=5)
            if r.status_code == 200:
                return True

            print(f"[TELEGRAM] Error {r.status_code}: {r.text}")

        except Exception as e:
            print(f"[TELEGRAM] Exception: {e}")

        time.sleep(0.5)

    return False


# --------------------------------------------------------
# FORMAT A SINGLE PREDICTION AS TELEGRAM MESSAGE
# --------------------------------------------------------

def format_prediction(pred) -> str:
    """
    Expected input: Prediction dataclass instance.
    """

    fid = pred.fixture_id
    market = pred.market
    selection = pred.selection.upper()

    p = pred.prob
    o = pred.odds
    ev = pred.ev
    minute = pred.aux.get("minute", 0)

    return (
        f"⚽ <b>GOALSNIPER AI</b>\n"
        f"📌 <b>Fixture ID:</b> {fid}\n"
        f"⏱ <b>Minute:</b> {minute}'\n\n"
        f"🎯 <b>Market:</b> {market}\n"
        f"🟩 <b>Bet:</b> {selection}\n\n"
        f"📈 <b>Probability:</b> {p*100:.1f}%\n"
        f"💸 <b>Odds:</b> {o:.2f}\n"
        f"📊 <b>EV:</b> {ev*100:.1f}%\n"
    )


# --------------------------------------------------------
# SEND A LIST OF PREDICTIONS
# --------------------------------------------------------

def send_predictions(pred_list):
    """
    Takes a list of Prediction objects and sends each to Telegram.
    Returns number of successful messages.
    """
    count = 0

    for pred in pred_list:
        msg = format_prediction(pred)
        if send_telegram(msg):
            count += 1

    return count
