import requests
import logging

logger = logging.getLogger("uvicorn")

def send_tip_message(tip: dict, bot_token: str, chat_id: str):
    message = (
        f"⚽️ *New Tip!*\n"
        f"🏟 *Match:* {tip['team']}\n"
        f"📊 *Tip:* {tip['tip']}\n"
        f"📈 *Confidence:* {tip['confidence']}%\n"
        f"🏆 *League:* {tip['league']}"
    )

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }

    try:
        res = requests.post(url, json=payload)
        res.raise_for_status()
        logger.info(f"[Telegram] Tip sent to chat {chat_id}")
    except requests.RequestException as e:
        logger.error(f"[Telegram] Error sending message: {e}")
