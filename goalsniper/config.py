import os
from dotenv import load_dotenv

load_dotenv()

def _bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in ("1","true","yes","y","on")

def _int(name: str, default: str) -> int:
    return int(os.getenv(name, default))

def _float01(name: str, default: str) -> float:
    v = float(os.getenv(name, default))
    return max(0.0, min(1.0, v))

def must(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required env: {name}")
    return v

def must_any(*names: str) -> str:
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    raise RuntimeError(f"Missing required env: one of {', '.join(names)}")

# Required
PORT = _int("PORT", "10000")
API_KEY = must_any("API_KEY", "APISPORTS_KEY")  # accepts either name
TELEGRAM_BOT_TOKEN = must("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = must("TELEGRAM_CHAT_ID")

# Daily cap & duplicate policy
DAILY_TIP_CAP = _int("DAILY_TIP_CAP", "15")
DUPLICATE_SUPPRESS_FOREVER = _bool("DUPLICATE_SUPPRESS_FOREVER", "true")

# Controls
RUN_TOKEN = os.getenv("RUN_TOKEN", "change_me_to_a_long_random_string")
SCAN_DAYS = max(1, _int("SCAN_DAYS", "1"))
MAX_CONCURRENT_REQUESTS = max(1, _int("MAX_CONCURRENT_REQUESTS", "6"))
STATS_REQUEST_DELAY_MS = max(0, _int("STATS_REQUEST_DELAY_MS", "200"))
MAX_TIPS_PER_RUN = max(1, _int("MAX_TIPS_PER_RUN", "150"))
MIN_CONFIDENCE_TO_SEND = _float01("MIN_CONFIDENCE_TO_SEND", "0.58")
LEAGUE_EXCLUDE_TYPES = [s.strip() for s in os.getenv("LEAGUE_EXCLUDE_TYPES", "").split(",") if s.strip()]

# Live scanning
LIVE_ENABLED = _bool("LIVE_ENABLED", "true")
LIVE_MINUTE_MIN = _int("LIVE_MINUTE_MIN", "10")
LIVE_MINUTE_MAX = _int("LIVE_MINUTE_MAX", "85")
LIVE_MAX_FIXTURES = _int("LIVE_MAX_FIXTURES", "80")

# Telegram webhook protection (optional)
TELEGRAM_WEBHOOK_TOKEN = os.getenv("TELEGRAM_WEBHOOK_TOKEN", "")

# Throttle & dedupe
THROTTLE_TIPS_PER_WINDOW = _int("THROTTLE_TIPS_PER_WINDOW", "10")
THROTTLE_WINDOW_MINUTES = _int("THROTTLE_WINDOW_MINUTES", "10")
DUPLICATE_SUPPRESS_MINUTES = _int("DUPLICATE_SUPPRESS_MINUTES", "360")  # 6h

__all__ = [
    "PORT","API_KEY","TELEGRAM_BOT_TOKEN","TELEGRAM_CHAT_ID",
    "RUN_TOKEN","SCAN_DAYS","MAX_CONCURRENT_REQUESTS","STATS_REQUEST_DELAY_MS",
    "MAX_TIPS_PER_RUN","MIN_CONFIDENCE_TO_SEND","LEAGUE_EXCLUDE_TYPES",
    "LIVE_ENABLED","LIVE_MINUTE_MIN","LIVE_MINUTE_MAX","LIVE_MAX_FIXTURES",
    "TELEGRAM_WEBHOOK_TOKEN","DAILY_TIP_CAP","DUPLICATE_SUPPRESS_FOREVER",
    "THROTTLE_TIPS_PER_WINDOW","THROTTLE_WINDOW_MINUTES","DUPLICATE_SUPPRESS_MINUTES",
]
