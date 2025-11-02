import os
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

log = logging.getLogger("goalsniper.config")

@dataclass
class DatabaseConfig:
    url: str
    pool_max: int = 5
    prefer_pooled: bool = True
    hostaddr: Optional[str] = None

@dataclass
class APIConfig:
    key: str
    base_url: str = "https://v3.football.api-sports.io"
    timeout: float = 8.0
    circuit_breaker_threshold: int = 8
    circuit_breaker_cooldown: int = 90

@dataclass
class TelegramConfig:
    bot_token: str
    chat_id: str
    webhook_secret: Optional[str] = None

@dataclass
class ModelConfig:
    confidence_threshold: float = 75.0
    max_tips_per_scan: int = 25
    dup_cooldown_min: int = 20
    tip_min_minute: int = 12
    train_min_minute: int = 15
    predictions_per_match: int = 1
    per_league_cap: int = 2

@dataclass
class OddsConfig:
    min_odds_ou: float = 1.50
    min_odds_btts: float = 1.50
    min_odds_1x2: float = 1.50
    max_odds_all: float = 20.0
    edge_min_bps: int = 600
    source: str = "auto"
    aggregation: str = "median"
    outlier_mult: float = 1.8
    require_n_books: int = 2
    quality_min: float = 0.35

@dataclass
class SchedulerConfig:
    enabled: bool = True
    scan_interval_sec: int = 300
    backfill_every_min: int = 15
    backfill_days: int = 14
    train_hour_utc: int = 2
    train_minute_utc: int = 12

class Config:
    def __init__(self):
        self._validate_required_envs()
        self._load_config()
        
    def _validate_required_envs(self):
        required = {
            'TELEGRAM_BOT_TOKEN': os.getenv('TELEGRAM_BOT_TOKEN'),
            'TELEGRAM_CHAT_ID': os.getenv('TELEGRAM_CHAT_ID'),
            'API_KEY': os.getenv('API_KEY'),
            'DATABASE_URL': os.getenv('DATABASE_URL')
        }
        
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise SystemExit(f"Missing required environment variables: {missing}")

    def _load_config(self):
        # Database
        self.database = DatabaseConfig(
            url=os.getenv('DATABASE_URL'),
            pool_max=int(os.getenv('DB_POOL_MAX', '5')),
            prefer_pooled=os.getenv('DB_PREFER_POOLED', '1').lower() not in ('0', 'false', 'no'),
            hostaddr=os.getenv('DB_HOSTADDR')
        )

        # API
        self.api = APIConfig(
            key=os.getenv('API_KEY'),
            timeout=float(os.getenv('REQ_TIMEOUT_SEC', '8.0')),
            circuit_breaker_threshold=int(os.getenv('API_CB_THRESHOLD', '8')),
            circuit_breaker_cooldown=int(os.getenv('API_CB_COOLDOWN_SEC', '90'))
        )

        # Telegram
        self.telegram = TelegramConfig(
            bot_token=os.getenv('TELEGRAM_BOT_TOKEN'),
            chat_id=os.getenv('TELEGRAM_CHAT_ID'),
            webhook_secret=os.getenv('TELEGRAM_WEBHOOK_SECRET')
        )

        # Models
        self.models = ModelConfig(
            confidence_threshold=float(os.getenv('CONF_THRESHOLD', '75')),
            max_tips_per_scan=int(os.getenv('MAX_TIPS_PER_SCAN', '25')),
            dup_cooldown_min=int(os.getenv('DUP_COOLDOWN_MIN', '20')),
            tip_min_minute=int(os.getenv('TIP_MIN_MINUTE', '12')),
            train_min_minute=int(os.getenv('TRAIN_MIN_MINUTE', '15')),
            predictions_per_match=int(os.getenv('PREDICTIONS_PER_MATCH', '1')),
            per_league_cap=int(os.getenv('PER_LEAGUE_CAP', '2'))
        )

        # Odds
        self.odds = OddsConfig(
            min_odds_ou=float(os.getenv('MIN_ODDS_OU', '1.50')),
            min_odds_btts=float(os.getenv('MIN_ODDS_BTTS', '1.50')),
            min_odds_1x2=float(os.getenv('MIN_ODDS_1X2', '1.50')),
            max_odds_all=float(os.getenv('MAX_ODDS_ALL', '20.0')),
            edge_min_bps=int(os.getenv('EDGE_MIN_BPS', '600')),
            source=os.getenv('ODDS_SOURCE', 'auto').lower(),
            aggregation=os.getenv('ODDS_AGGREGATION', 'median').lower(),
            outlier_mult=float(os.getenv('ODDS_OUTLIER_MULT', '1.8')),
            require_n_books=int(os.getenv('ODDS_REQUIRE_N_BOOKS', '2')),
            quality_min=float(os.getenv('ODDS_QUALITY_MIN', '0.35'))
        )

        # Scheduler
        self.scheduler = SchedulerConfig(
            enabled=os.getenv('RUN_SCHEDULER', '1').lower() not in ('0', 'false', 'no'),
            scan_interval_sec=int(os.getenv('SCAN_INTERVAL_SEC', '300')),
            backfill_every_min=int(os.getenv('BACKFILL_EVERY_MIN', '15')),
            backfill_days=int(os.getenv('BACKFILL_DAYS', '14')),
            train_hour_utc=int(os.getenv('TRAIN_HOUR_UTC', '2')),
            train_minute_utc=int(os.getenv('TRAIN_MINUTE_UTC', '12'))
        )

        # Feature flags
        self.harvest_mode = os.getenv('HARVEST_MODE', '1').lower() not in ('0', 'false', 'no')
        self.train_enable = os.getenv('TRAIN_ENABLE', '1').lower() not in ('0', 'false', 'no')
        self.auto_tune_enable = os.getenv('AUTO_TUNE_ENABLE', '0').lower() not in ('0', 'false', 'no')
        self.stale_guard_enable = os.getenv('STALE_GUARD_ENABLE', '1').lower() not in ('0', 'false', 'no')
        self.allow_tips_without_odds = os.getenv('ALLOW_TIPS_WITHOUT_ODDS', '0').lower() not in ('0', 'false', 'no')

        # OU Lines
        self.ou_lines = self._parse_ou_lines()

    def _parse_ou_lines(self) -> List[float]:
        env_val = os.getenv('OU_LINES', '2.5,3.5')
        out = []
        for t in env_val.split(','):
            t = t.strip()
            if not t:
                continue
            try:
                out.append(float(t))
            except:
                pass
        return out or [2.5, 3.5]

    def validate(self):
        """Validate configuration"""
        if not (0 <= self.models.confidence_threshold <= 100):
            log.warning("CONF_THRESHOLD should be 0-100, got %s", self.models.confidence_threshold)
        
        if self.scheduler.scan_interval_sec < 30:
            log.warning("SCAN_INTERVAL_SEC very low: %s", self.scheduler.scan_interval_sec)
        
        log.info("[CONFIG] Configuration validation passed")

# Global config instance
config = Config()
