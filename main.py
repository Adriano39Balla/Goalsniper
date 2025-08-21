#!/usr/bin/env python3
"""
main.py

Flask entrypoint for Goalsniper.
- Exposes routes (/harvest, /backfill, /train)
- Runs scheduled jobs:
    * Harvest snapshots every 15 minutes
    * Backfill results every 30 minutes
    * Train nightly at 03:00 UTC
- Sends Telegram digest after nightly training
"""

from flask import Flask
from apscheduler.schedulers.background import BackgroundScheduler
from app.routes import bp as routes_bp
from app.training import run_training
from app.harvest import harvest_route
from app.backfill import backfill_route
import telegram
import os


# Telegram config from Railway environment variables
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")


def send_digest(summary: dict):
    """Send Telegram digest with per-market metrics."""
    if not TELEGRAM_CHAT_ID or not TELEGRAM_TOKEN:
        print("⚠️ Telegram not configured, skipping digest")
        return

    bot = telegram.Bot(token=TELEGRAM_TOKEN)

    msg = "🌙 Nightly Training Digest\n\n"
    for market, metrics in summary.get("metrics", {}).items():
        acc = metrics.get("acc")
        auc = metrics.get("auc")
        msg += f"📊 {market}: acc={acc:.2f}, auc={auc:.2f if auc else 'N/A'}\n"

    try:
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
        print("✅ Nightly digest sent to Telegram")
    except Exception as e:
        print(f"⚠️ Failed to send digest: {e}")


# --- Scheduled jobs ---

def scheduled_harvest():
    print("🔄 Scheduled harvest triggered...")
    return harvest_route()

def scheduled_backfill():
    print("📥 Scheduled backfill triggered...")
    return backfill_route()

def scheduled_training():
    print("🌙 Scheduled nightly training triggered...")
    result = run_training()
    if result["ok"]:
        send_digest(result["summary"])
    else:
        print("⚠️ Nightly training failed:", result)


# --- App Factory ---

def create_app():
    app = Flask(__name__)
    app.register_blueprint(routes_bp)

    # BackgroundScheduler
    scheduler = BackgroundScheduler()

    # Schedule harvest every 15 minutes
    scheduler.add_job(scheduled_harvest, "interval", minutes=15)

    # Schedule backfill every 30 minutes
    scheduler.add_job(scheduled_backfill, "interval", minutes=30)

    # Schedule training every night at 03:00 UTC
    scheduler.add_job(scheduled_training, "cron", hour=3, minute=0)

    scheduler.start()

    return app


# Railway/Gunicorn looks for "app"
app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
