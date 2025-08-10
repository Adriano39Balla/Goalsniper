import os
from dotenv import load_dotenv

load_dotenv()

def must(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required env: {name}")
    return v

# Required
PORT = int(os.getenv("PORT", "10000"))
API_KEY = must("API_KEY")  # changed from APISPORTS_KEY
TELEGRAM_BOT_TOKEN = must("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = must("TELEGRAM_CHAT_ID")

# Daily cap & duplicate policy
DAILY_TIP_CAP = int(os.getenv("DAILY_TIP_CAP", "15"))
DUPLICATE_SUPPRESS_FOREVER = os.getenv("DUPLICATE_SUPPRESS_FOREVER", "true").lower() == "true"

# Controls (defaults are safe)
RUN_TOKEN = os.getenv("RUN_TOKEN", "change_me_to_a_long_random_string")
SCAN_DAYS = max(1, int(os.getenv("SCAN_DAYS", "1")))
MAX_CONCURRENT_REQUESTS = max(1, int(os.getenv("MAX_CONCURRENT_REQUESTS", "6")))
STATS_REQUEST_DELAY_MS = max(0, int(os.getenv("STATS_REQUEST_DELAY_MS", "200")))
MAX_TIPS_PER_RUN = max(1, int(os.getenv("MAX_TIPS_PER_RUN", "150")))
MIN_CONFIDENCE_TO_SEND = max(0.0, min(1.0, float(os.getenv("MIN_CONFIDENCE_TO_SEND", "0.58"))))
LEAGUE_EXCLUDE_TYPES = [s.strip() for s in os.getenv("LEAGUE_EXCLUDE_TYPES", "").split(",") if s.strip()]

# Live scanning
LIVE_ENABLED = os.getenv("LIVE_ENABLED", "true").lower() == "true"
LIVE_MINUTE_MIN = int(os.getenv("LIVE_MINUTE_MIN", "10"))
LIVE_MINUTE_MAX = int(os.getenv("LIVE_MINUTE_MAX", "85"))
LIVE_MAX_FIXTURES = int(os.getenv("LIVE_MAX_FIXTURES", "80"))

# Telegram webhook protection (optional)
TELEGRAM_WEBHOOK_TOKEN = os.getenv("TELEGRAM_WEBHOOK_TOKEN", "")

__all__ = [
    "PORT", "API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID",
    "RUN_TOKEN", "SCAN_DAYS", "MAX_CONCURRENT_REQUESTS", "STATS_REQUEST_DELAY_MS",
    "MAX_TIPS_PER_RUN", "MIN_CONFIDENCE_TO_SEND", "LEAGUE_EXCLUDE_TYPES",
    "LIVE_ENABLED", "LIVE_MINUTE_MIN", "LIVE_MINUTE_MAX", "LIVE_MAX_FIXTURES",
    "TELEGRAM_WEBHOOK_TOKEN",
]

# Throttle & dedupe
THROTTLE_TIPS_PER_WINDOW = int(os.getenv("THROTTLE_TIPS_PER_WINDOW", "10"))
THROTTLE_WINDOW_MINUTES = int(os.getenv("THROTTLE_WINDOW_MINUTES", "10"))
# suppress duplicate tips for same fixture if we’ve sent any within this window
DUPLICATE_SUPPRESS_MINUTES = int(os.getenv("DUPLICATE_SUPPRESS_MINUTES", "360"))  # 6h
