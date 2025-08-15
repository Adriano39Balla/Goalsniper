"""
Centralized configuration (env-driven).
"""

import os
from dotenv import load_dotenv

load_dotenv()  # optional for local dev

# ---------- helpers ----------
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

# ---------- required credentials ----------
# Accept either API_KEY or APISPORTS_KEY
_API_KEY = os.getenv("API_KEY") or os.getenv("APISPORTS_KEY")
if not _API_KEY:
    raise RuntimeError("Missing required env: API_KEY (or APISPORTS_KEY)")
API_KEY = _API_KEY

TELEGRAM_BOT_TOKEN = must("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = must("TELEGRAM_CHAT_ID")
RUN_TOKEN = must("RUN_TOKEN")
TELEGRAM_WEBHOOK_TOKEN = os.getenv("TELEGRAM_WEBHOOK_TOKEN", "")

# Server
PORT = int(os.getenv("PORT", "8080"))

# ---------- scanning / tip policy ----------
SCAN_DAYS = max(1, _int("SCAN_DAYS", "1"))

MAX_TIPS_PER_RUN = max(1, _int("MAX_TIPS_PER_RUN", "25"))
MIN_CONFIDENCE_TO_SEND = _float01("MIN_CONFIDENCE_TO_SEND", "0.58")
DAILY_TIP_CAP = max(1, _int("DAILY_TIP_CAP", "15"))

DUPLICATE_SUPPRESS_FOREVER = _bool("DUPLICATE_SUPPRESS_FOREVER", "true")
DUPLICATE_SUPPRESS_MINUTES = _int("DUPLICATE_SUPPRESS_MINUTES", "360")

LIVE_ENABLED = _bool("LIVE_ENABLED", "true")
LIVE_MINUTE_MIN = _int("LIVE_MINUTE_MIN", "10")
LIVE_MINUTE_MAX = _int("LIVE_MINUTE_MAX", "85")
LIVE_MAX_FIXTURES = _int("LIVE_MAX_FIXTURES", "80")

STATS_REQUEST_DELAY_MS = max(50, _int("STATS_REQUEST_DELAY_MS", "200"))
MAX_CONCURRENT_REQUESTS = max(1, _int("MAX_CONCURRENT_REQUESTS", "6"))

THROTTLE_TIPS_PER_WINDOW = max(1, _int("THROTTLE_TIPS_PER_WINDOW", "8"))
THROTTLE_WINDOW_MINUTES = max(1, _int("THROTTLE_WINDOW_MINUTES", "10"))

# ---------- api_football knobs ----------
API_RATE_PER_MIN = max(1, _int("API_RATE_PER_MIN", "60"))

CACHE_TTL_LEAGUES = max(0, _int("CACHE_TTL_LEAGUES", "1800"))
CACHE_TTL_FIXTURES_BYDATE = max(0, _int("CACHE_TTL_FIXTURES_BYDATE", "1800"))
CACHE_TTL_LIVE = max(0, _int("CACHE_TTL_LIVE", "90"))
CACHE_TTL_TEAM_STATS = max(0, _int("CACHE_TTL_TEAM_STATS", "600"))
CACHE_TTL_ODDS = max(0, _int("CACHE_TTL_ODDS", "600"))

# ---------- league filters ----------
# Empty allow-lists => allow all countries/leagues
COUNTRY_FLAGS_ALLOW = os.getenv("COUNTRY_FLAGS_ALLOW", "")
LEAGUE_ALLOW_KEYWORDS = os.getenv("LEAGUE_ALLOW_KEYWORDS", "")

# Keep only the junk out (case-insensitive, matched in league names)
EXCLUDE_KEYWORDS = os.getenv(
    "EXCLUDE_KEYWORDS",
    # friendlies + youth/reserve/academy/B teams + misc exhibitions
    "FRIENDLY,FRIENDLIES,CLUB FRIENDLY,CLUB FRIENDLIES,TEST,TRIAL,"
    "U17,U18,U19,U20,U21,U23,YOUTH,YTH,ACADEMY,RESERVE,RESERVES,B-TEAM,"
    "ALL STARS,ALL-STARS,LEGENDS"
)

__all__ = [
    "API_KEY","TELEGRAM_BOT_TOKEN","TELEGRAM_CHAT_ID","RUN_TOKEN",
    "PORT","TELEGRAM_WEBHOOK_TOKEN",
    "SCAN_DAYS","MAX_TIPS_PER_RUN","MIN_CONFIDENCE_TO_SEND","DAILY_TIP_CAP",
    "DUPLICATE_SUPPRESS_FOREVER","DUPLICATE_SUPPRESS_MINUTES",
    "LIVE_ENABLED","LIVE_MINUTE_MIN","LIVE_MINUTE_MAX","LIVE_MAX_FIXTURES",
    "STATS_REQUEST_DELAY_MS","MAX_CONCURRENT_REQUESTS",
    "THROTTLE_TIPS_PER_WINDOW","THROTTLE_WINDOW_MINUTES",
    "API_RATE_PER_MIN",
    "CACHE_TTL_LEAGUES","CACHE_TTL_FIXTURES_BYDATE","CACHE_TTL_LIVE",
    "CACHE_TTL_TEAM_STATS","CACHE_TTL_ODDS",
    "COUNTRY_FLAGS_ALLOW","LEAGUE_ALLOW_KEYWORDS","EXCLUDE_KEYWORDS",
]
