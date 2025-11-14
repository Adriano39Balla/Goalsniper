# config/settings.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # v2-style settings config
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # API Football
    API_FOOTBALL_KEY: str = ""
    API_FOOTBALL_URL: str = "https://v3.football.api-sports.io"

    # Database
    DATABASE_URL: str = ""
    SUPABASE_URL: str = ""
    SUPABASE_KEY: str = ""

    # Telegram
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""

    # Model Settings
    MODEL_UPDATE_HOURS: int = 6
    MIN_CONFIDENCE_THRESHOLD: float = 0.65
    LIVE_UPDATE_INTERVAL: int = 30  # seconds

    # Feature Engineering
    ROLLING_WINDOW: int = 10
    FEATURE_LOOKBACK: int = 15

settings = Settings()
