from telegram import Bot
from telegram.error import TelegramError
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from utils import logger

bot = Bot(token=TELEGRAM_BOT_TOKEN)

def send_message(text):
    try:
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text)
        logger.info("Telegram message sent.")
    except TelegramError as e:
        logger.error(f"Telegram error: {e}")
