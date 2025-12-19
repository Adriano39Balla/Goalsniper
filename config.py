"""
Configuration Management
Centralized settings using Pydantic for validation and type safety
"""

from pydantic_settings import BaseSettings
from pydantic import Field, validator
from typing import Optional
import os
from pathlib import Path


class Settings(BaseSettings):
    """
    Application settings with environment variable support
    """
    
    # API-Football Configuration
    API_FOOTBALL_KEY: str = Field(..., description="API-Football API key")
    API_FOOTBALL_BASE_URL: str = Field(
        default="https://v3.football.api-sports.io",
        description="API-Football base URL"
    )
    
    # Supabase Configuration
    SUPABASE_URL: str = Field(..., description="Supabase project URL")
    SUPABASE_KEY: str = Field(..., description="Supabase anon key")
    DB_HOST: str = Field(..., description="Database host")
    DB_PORT: int = Field(default=5432, description="Database port")
    DB_NAME: str = Field(default="postgres", description="Database name")
    DB_USER: str = Field(default="postgres", description="Database user")
    DB_PASSWORD: str = Field(..., description="Database password")
    
    # Telegram Configuration
    TELEGRAM_BOT_TOKEN: str = Field(..., description="Telegram bot token")
    TELEGRAM_CHAT_ID: str = Field(..., description="Telegram chat ID")
    
    # Application Configuration
    ENVIRONMENT: str = Field(default="production", description="Environment (production/development)")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    MIN_CONFIDENCE_THRESHOLD: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for predictions"
    )
    MAX_PREDICTIONS_PER_HOUR: int = Field(
        default=50,
        description="Maximum predictions per hour"
    )
    
    # Model Configuration
    MODEL_TYPE: str = Field(
        default="lightgbm",
        description="ML model type (lightgbm/xgboost)"
    )
    RETRAIN_INTERVAL_HOURS: int = Field(
        default=24,
        description="Hours between automatic retraining"
    )
    BACKFILL_DAYS: int = Field(
        default=90,
        description="Days of historical data to backfill"
    )
    
    # API Control Configuration
    API_HOST: str = Field(default="0.0.0.0", description="API host")
    API_PORT: int = Field(default=8000, description="API port")
    API_SECRET_KEY: str = Field(..., description="API authentication secret key")
    
    # Railway Configuration
    PORT: Optional[int] = Field(default=None, description="Railway port (auto-assigned)")
    
    @validator('PORT', pre=True, always=True)
    def set_port(cls, v, values):
        """Use Railway PORT if available, otherwise use API_PORT"""
        return v or values.get('API_PORT', 8000)
    
    @validator('MODEL_TYPE')
    def validate_model_type(cls, v):
        """Validate model type"""
        allowed = ['lightgbm', 'xgboost', 'catboost']
        if v not in allowed:
            raise ValueError(f"MODEL_TYPE must be one of {allowed}")
        return v
    
    @validator('ENVIRONMENT')
    def validate_environment(cls, v):
        """Validate environment"""
        allowed = ['production', 'development', 'staging']
        if v not in allowed:
            raise ValueError(f"ENVIRONMENT must be one of {allowed}")
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Create global settings instance
try:
    settings = Settings()
except Exception as e:
    print(f"Error loading settings: {e}")
    print("Please ensure .env file exists with all required variables")
    raise


# Create necessary directories
def setup_directories():
    """Create required directories if they don't exist"""
    directories = [
        "models",
        "logs",
        "data"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


# Setup on import
setup_directories()
