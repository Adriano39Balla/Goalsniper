import os
from typing import Optional
from pydantic import BaseSettings

class Settings(BaseSettings):
    # API Football
    API_FOOTBALL_KEY: str = os.getenv("API_FOOTBALL_KEY", "")
    API_FOOTBALL_URL: str = "https://v3.football.api-sports.io"
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")
    
    # Telegram
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
    
    # Model Settings
    MODEL_UPDATE_HOURS: int = 6
    MIN_CONFIDENCE_THRESHOLD: float = 0.65
    LIVE_UPDATE_INTERVAL: int = 30  # seconds
    
    # Feature Engineering
    ROLLING_WINDOW: int = 10
    FEATURE_LOOKBACK: int = 15
    
    class Config:
        env_file = ".env"

settings = Settings()
