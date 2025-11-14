import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# -------------------------------------------------------
# API-FOOTBALL
# -------------------------------------------------------
API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY")
API_FOOTBALL_URL = "https://v3.football.api-sports.io"

# -------------------------------------------------------
# SUPABASE SETTINGS
# -------------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Table names
TABLE_FIXTURES = "fixtures"
TABLE_ODDS = "odds"
TABLE_STATS = "stats"
TABLE_TIPS = "tips"
TABLE_RESULTS = "tip_results"

# -------------------------------------------------------
# TELEGRAM SETTINGS
# -------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# -------------------------------------------------------
# MODEL PATHS
# -------------------------------------------------------
MODEL_DIR = "models/"
MODEL_1X2 = os.path.join(MODEL_DIR, "logreg_1x2.pkl")
MODEL_OU25 = os.path.join(MODEL_DIR, "logreg_ou25.pkl")
MODEL_BTTS = os.path.join(MODEL_DIR, "logreg_btts.pkl")

# -------------------------------------------------------
# FILTER SETTINGS (WITH SAFE MODE SUPPORT)
# -------------------------------------------------------
SAFE_MODE = os.getenv("SAFE_MODE", "false").lower() == "true"

# Default thresholds (adaptive logic will modify)
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.55"))  # LOWERED from 0.65 → 0.55
MIN_EV = float(os.getenv("MIN_EV", "0.02"))                  # LOWERED from 0.03 → 0.02
MIN_ODDS = float(os.getenv("MIN_ODDS", "1.40"))

MAX_CONFIDENCE = 0.90

# -------------------------------------------------------
# SCHEDULER
# -------------------------------------------------------
LIVE_PREDICTION_INTERVAL = int(os.getenv("LIVE_INTERVAL", "30"))
PREMATCH_PREDICTION_INTERVAL = int(os.getenv("PREMATCH_INTERVAL", "600"))

# -------------------------------------------------------
# RAILWAY-SPECIFIC SETTINGS
# -------------------------------------------------------
# Railway uses PORT environment variable automatically
PORT = int(os.getenv("PORT", 8000))

# -------------------------------------------------------
# OTHER
# -------------------------------------------------------
DEBUG_MODE = os.getenv("DEBUG", "0") == "1"
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")

# -------------------------------------------------------
# VALIDATION (Helpful for debugging)
# -------------------------------------------------------
def validate_config():
    """Validate critical configuration settings"""
    errors = []
    
    if not API_FOOTBALL_KEY:
        errors.append("API_FOOTBALL_KEY is missing")
    
    if not SUPABASE_URL:
        errors.append("SUPABASE_URL is missing")
    
    if not SUPABASE_KEY:
        errors.append("SUPABASE_KEY is missing")
    
    if not TELEGRAM_BOT_TOKEN:
        errors.append("TELEGRAM_BOT_TOKEN is missing")
    
    if not TELEGRAM_CHAT_ID:
        errors.append("TELEGRAM_CHAT_ID is missing")
    
    return errors

# Optional: Auto-validate on import if in debug mode
if DEBUG_MODE:
    config_errors = validate_config()
    if config_errors:
        print("⚠️  Configuration warnings:")
        for error in config_errors:
            print(f"   - {error}")
