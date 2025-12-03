from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    API_FOOTBALL_KEY: str
    TELEGRAM_BOT_TOKEN: str
    TELEGRAM_CHAT_ID: str
    DATABASE_URL: str
    REDIS_URL: Optional[str] = None
    
    class Config:
        env_file = ".env"

class BettingConfig:
    min_confidence: float = 0.65
    min_ev: float = 0.1
    max_tips_per_match: int = 3
    bankroll_percentage: float = 0.02
    odds_format: str = "decimal"
