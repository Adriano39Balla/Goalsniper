# goalsniper/config.py

"""
Centralized configuration (env-driven).
- No secrets defaulted: required keys must be set in environment.
- All non-secret behavior can be tuned via env vars (documented below).
"""

import os
from dotenv import load_dotenv

load_dotenv()  # optional: allows .env locally


# ------------------------------
# helpers
# ------------------------------
def _bool(name: str, default: str = "false") -> bool:
    return (os.getenv(name, default) or "").strip().lower() in ("1", "true", "yes", "y", "on")

def _int(name: str, default: str) -> int:
    return int(os.getenv(name, default))

def _float01(name: str, default: str) -> float:
    v = float(os.getenv(name, default))
    return 0.0 if v < 0 else 1.0 if v > 1 else v

def must(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required env: {name}")
    return v


# ------------------------------
# required credentials
# ------------------------------
API_KEY = must("API_KEY")  # API-Sports key
TELEGRAM_BOT_TOKEN = must("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = must("TELEGRAM_CHAT_ID")
RUN_TOKEN = must("RUN_TOKEN")  # bearer for /run and /digest (no placeholders)

# optional webhook token to secure /telegram/webhook/<token>
TELEGRAM_WEBHOOK_TOKEN = os.getenv("TELEGRAM_WEBHOOK_TOKEN", "")

# server port (Render honors PORT)
PORT = _int("PORT", "10000")


# ------------------------------
# scanning / tip policy
# ------------------------------
SCAN_DAYS = max(1, _int("SCAN_DAYS", "1"))  # days ahead to scan fixtures-by-date

# per-run limit for tips (protects from bursts)
MAX_TIPS_PER_RUN = max(1, _int("MAX_TIPS_PER_RUN", "25"))

# minimum confidence to consider a tip (pre-learning gate)
MIN_CONFIDENCE_TO_SEND = _float01("MIN_CONFIDENCE_TO_SEND", "0.78")

# overall daily cap
DAILY_TIP_CAP = max(1, _int("DAILY_TIP_CAP", "15"))

# duplicate suppression
DUPLICATE_SUPPRESS_FOREVER = _bool("DUPLICATE_SUPPRESS_FOREVER", "true")
DUPLICATE_SUPPRESS_MINUTES = _int("DUPLICATE_SUPPRESS_MINUTES", "360")  # if FOREVER is false

# live scanning window
LIVE_ENABLED = _bool("LIVE_ENABLED", "true")
LIVE_MINUTE_MIN = _int("LIVE_MINUTE_MIN", "10")
LIVE_MINUTE_MAX = _int("LIVE_MINUTE_MAX", "80")
LIVE_MAX_FIXTURES = _int("LIVE_MAX_FIXTURES", "30")

# pacing between sequential stats requests (ms)
STATS_REQUEST_DELAY_MS = max(50, _int("STATS_REQUEST_DELAY_MS", "150"))

# concurrency for outbound API calls
MAX_CONCURRENT_REQUESTS = max(1, _int("MAX_CONCURRENT_REQUESTS", "4"))

# throttle short-term bursts (optional UI-side pacing)
THROTTLE_TIPS_PER_WINDOW = max(1, _int("THROTTLE_TIPS_PER_WINDOW", "8"))
THROTTLE_WINDOW_MINUTES = max(1, _int("THROTTLE_WINDOW_MINUTES", "10"))


# ------------------------------
# api_football.py specific knobs
# ------------------------------
# token-bucket rate limiting (per process)
API_RATE_PER_MIN = max(1, _int("API_RATE_PER_MIN", "60"))

# response cache TTLs (seconds)
CACHE_TTL_LEAGUES = max(0, _int("CACHE_TTL_LEAGUES", "1800"))          # 30 min
CACHE_TTL_FIXTURES_BYDATE = max(0, _int("CACHE_TTL_FIXTURES_BYDATE", "1800"))
CACHE_TTL_LIVE = max(0, _int("CACHE_TTL_LIVE", "90"))
CACHE_TTL_TEAM_STATS = max(0, _int("CACHE_TTL_TEAM_STATS", "600"))
CACHE_TTL_ODDS = max(0, _int("CACHE_TTL_ODDS", "600"))

# optional filters (empty => allow all; set via env to restrict)
COUNTRY_FLAGS_ALLOW = os.getenv("COUNTRY_FLAGS_ALLOW", "")  # e.g. "🇩🇪,🇬🇧" or "DE,GB"
LEAGUE_ALLOW_KEYWORDS = os.getenv("LEAGUE_ALLOW_KEYWORDS", "")
EXCLUDE_KEYWORDS = os.getenv(
    "EXCLUDE_KEYWORDS",
    "U19,U20,U21,U23,YOUTH,WOMEN,FRIENDLY,CLUB FRIENDLIES,RESERVE,AMATEUR,B-TEAM",
)


__all__ = [
    # required
    "API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "RUN_TOKEN",
    # server
    "PORT", "TELEGRAM_WEBHOOK_TOKEN",
    # scan/tip policy
    "SCAN_DAYS", "MAX_TIPS_PER_RUN", "MIN_CONFIDENCE_TO_SEND", "DAILY_TIP_CAP",
    "DUPLICATE_SUPPRESS_FOREVER", "DUPLICATE_SUPPRESS_MINUTES",
    "LIVE_ENABLED", "LIVE_MINUTE_MIN", "LIVE_MINUTE_MAX", "LIVE_MAX_FIXTURES",
    "STATS_REQUEST_DELAY_MS", "MAX_CONCURRENT_REQUESTS",
    "THROTTLE_TIPS_PER_WINDOW", "THROTTLE_WINDOW_MINUTES",
    # api football knobs
    "API_RATE_PER_MIN",
    "CACHE_TTL_LEAGUES", "CACHE_TTL_FIXTURES_BYDATE", "CACHE_TTL_LIVE",
    "CACHE_TTL_TEAM_STATS", "CACHE_TTL_ODDS",
    "COUNTRY_FLAGS_ALLOW", "LEAGUE_ALLOW_KEYWORDS", "EXCLUDE_KEYWORDS",
]
