# config.py

import os

class Config:
    # API Football configuration
    API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY")
    API_FOOTBALL_BASE_URL = "https://v3.football.api-sports.io"

    # Telegram configuration
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

    # Prediction settings
    MIN_CONFIDENCE_THRESHOLD = 0.7
    MAX_PREDICTIONS_PER_DAY = 10

    # Major leagues (fixed typo in Ligue 1)
    MAJOR_LEAGUES = [
        39,   # Premier League
        140,  # La Liga
        78,   # Bundesliga
        135,  # Serie A
        61,   # Ligue 1
        2,    # UEFA Champions League
        3     # UEFA Europa League
    ]

    # Rate limiting
    API_REQUESTS_PER_MINUTE = 100
    REQUEST_DELAY = 0.6
