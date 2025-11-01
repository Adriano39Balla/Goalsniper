import os
import pytz
import logging

# ─────────────────────────── Timezones ─────────────────────────── #
TZ_UTC = pytz.UTC
BERLIN_TZ = pytz.timezone("Europe/Berlin")

# ─────────────────────────── Logging ─────────────────────────── #
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] - %(message)s")

# ─────────────────────────── Feature Toggles ─────────────────────────── #
TRAIN_ENABLE: bool = os.getenv("TRAIN_ENABLE", "1") not in ("0", "false", "False", "no", "NO")
AUTO_TUNE_ENABLE: bool = os.getenv("AUTO_TUNE_ENABLE", "0") not in ("0", "false", "False", "no", "NO")
RUN_SCHEDULER: bool = os.getenv("RUN_SCHEDULER", "1") not in ("0", "false", "False", "no", "NO")
DAILY_ACCURACY_DIGEST_ENABLE: bool = os.getenv("DAILY_ACCURACY_DIGEST_ENABLE", "0") not in ("0", "false", "False", "no", "NO")

# ─────────────────────────── Scheduler Timing ─────────────────────────── #
SCAN_INTERVAL_SEC: int = int(os.getenv("SCAN_INTERVAL_SEC", "20"))
BACKFILL_EVERY_MIN: int = int(os.getenv("BACKFILL_EVERY_MIN", "8"))
TRAIN_HOUR_UTC: int = int(os.getenv("TRAIN_HOUR_UTC", "5"))
TRAIN_MINUTE_UTC: int = int(os.getenv("TRAIN_MINUTE_UTC", "10"))

# ─────────────────────────── Telegram / Admin ─────────────────────────── #
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
WEBHOOK_SECRET: str = os.getenv("WEBHOOK_SECRET", "")
ADMIN_API_KEY: str = os.getenv("ADMIN_API_KEY", "")

# ─────────────────────────── Football API / Auth ─────────────────────────── #
FOOTBALL_API_URL: str = os.getenv("FOOTBALL_API_URL", "").strip()
API_KEY: str = os.getenv("API_KEY", "").strip()
MODEL_VERSION: str = os.getenv("MODEL_VERSION", "model_v2")

# ─────────────────────────── Flask Server ─────────────────────────── #
FLASK_HOST: str = os.getenv("HOST", "0.0.0.0")
FLASK_PORT: int = int(os.getenv("PORT", "8080"))

# ─────────────────────────── Database ─────────────────────────── #
DATABASE_URL: str = os.getenv("DATABASE_URL", "")

# ─────────────────────────── Global Metrics Container (use with care) ─────────────────────────── #
METRICS = {}

# ─────────────────────────── Sanity Checks ─────────────────────────── #
if not TELEGRAM_BOT_TOKEN:
    logging.warning("⚠️ TELEGRAM_BOT_TOKEN is not set.")

if not FOOTBALL_API_URL:
    logging.warning("⚠️ FOOTBALL_API_URL is not set.")

if not DATABASE_URL:
    logging.warning("⚠️ DATABASE_URL is not set.")
