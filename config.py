# config.py

import os
import pytz

# ─────────────────────────── Timezones ─────────────────────────── #
TZ_UTC = pytz.UTC
BERLIN_TZ = pytz.timezone("Europe/Berlin")

# ─────────────────────────── Feature Toggles ─────────────────────────── #
TRAIN_ENABLE = os.getenv("TRAIN_ENABLE", "1") not in ("0", "false", "False", "no", "NO")
AUTO_TUNE_ENABLE = os.getenv("AUTO_TUNE_ENABLE", "0") not in ("0", "false", "False", "no", "NO")
RUN_SCHEDULER = os.getenv("RUN_SCHEDULER", "1") not in ("0", "false", "False", "no", "NO")
DAILY_ACCURACY_DIGEST_ENABLE = os.getenv("DAILY_ACCURACY_DIGEST_ENABLE", "0") not in ("0", "false", "False", "no", "NO")

# ─────────────────────────── Scheduler Timing ─────────────────────────── #
SCAN_INTERVAL_SEC = int(os.getenv("SCAN_INTERVAL_SEC", "20"))
BACKFILL_EVERY_MIN = int(os.getenv("BACKFILL_EVERY_MIN", "8"))
TRAIN_HOUR_UTC = int(os.getenv("TRAIN_HOUR_UTC", "5"))
TRAIN_MINUTE_UTC = int(os.getenv("TRAIN_MINUTE_UTC", "10"))

# ─────────────────────────── Telegram / Admin ─────────────────────────── #
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")

# ─────────────────────────── Football API ─────────────────────────── #
FOOTBALL_API_URL = os.getenv("FOOTBALL_API_URL", "").strip()
FOOTBALL_API_KEY = os.getenv("FOOTBALL_API_KEY", "").strip()

# ─────────────────────────── Metrics ─────────────────────────── #
METRICS = {}

# ─────────────────────────── Other ─────────────────────────── #
MODEL_VERSION = os.getenv("MODEL_VERSION", "model_v2")

# Default Flask host/port (used in main.py for app.run)
FLASK_HOST = os.getenv("HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("PORT", "8080"))
