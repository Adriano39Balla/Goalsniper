import logging
import sys

# ---- logging you can actually read
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# ----- DIRECT SEND TEST (bypasses all bot frameworks) -----
if __name__ == "__main__" and "--direct" in sys.argv:
    from formatter import format_tip_html
    from tips import fake_match
    from telegram_client import send_text

    html = format_tip_html(fake_match())
    print("Sending direct test message…")
    send_text(html, parse_mode="HTML")
    print("✅ If you saw the message in Telegram, credentials & permissions are correct.")
    sys.exit(0)

# ----- /run COMMAND via python-telegram-bot v20+ -----
# pip install python-telegram-bot==20.6 requests
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from formatter import format_tip_html
from tips import fake_match
from telegram_client import send_text
from config import TELEGRAM_TOKEN, CHAT_ID

async def run_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Optional: restrict to the configured chat id to avoid posting elsewhere
    try:
        html = format_tip_html(fake_match())
        # Use our direct sender so we get consistent error reporting
        send_text(html, parse_mode="HTML")
        await update.message.reply_text("✅ Sent test tip to target chat.")
    except Exception as e:
        await update.message.reply_text(f"❌ Failed: {e}")

def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("run", run_cmd))
    logging.info("Bot polling… send /run to the bot in any chat where it exists.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
