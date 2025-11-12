import os
from dotenv import load_dotenv

# Load environment variables from .env (Railway will override these automatically)
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
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")  # single-channel system

# -------------------------------------------------------
# MODEL PATHS
# -------------------------------------------------------
MODEL_DIR = "models/"
MODEL_1X2 = os.path.join(MODEL_DIR, "logreg_1x2.pkl")
MODEL_OU25 = os.path.join(MODEL_DIR, "logreg_ou25.pkl")
MODEL_BTTS = os.path.join(MODEL_DIR, "logreg_btts.pkl")
MODEL_CALIBRATION = os.path.join(MODEL_DIR, "bayesian_calibration.pkl")

# -------------------------------------------------------
# ADAPTIVE FILTER THRESHOLDS
# -------------------------------------------------------
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.65"))   # 65%
MIN_EV = float(os.getenv("MIN_EV", "0.03"))                   # +3% EV
MIN_ODDS = float(os.getenv("MIN_ODDS", "1.40"))               # Odds ≥ 1.40

# Upper limit (system will adapt)
MAX_CONFIDENCE = 0.90

# -------------------------------------------------------
# INFERENCE & SCHEDULER SETTINGS
# -------------------------------------------------------
LIVE_PREDICTION_INTERVAL = int(os.getenv("LIVE_INTERVAL", "30"))  # seconds
PREMATCH_PREDICTION_INTERVAL = int(os.getenv("PREMATCH_INTERVAL", "600"))

# -------------------------------------------------------
# OTHER SETTINGS
# -------------------------------------------------------
DEBUG_MODE = os.getenv("DEBUG", "0") == "1"
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
