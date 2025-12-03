# Extended config.py: includes Settings, BettingConfig and a simple SQLite-backed DatabaseManager
from pydantic_settings import BaseSettings
from typing import Optional, Dict, Any
from pathlib import Path
import sqlite3
import json
from datetime import datetime, timedelta

class Settings(BaseSettings):
    API_FOOTBALL_KEY: str = ""
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""
    DATABASE_URL: str = "sqlite:///data/db.sqlite"
    REDIS_URL: Optional[str] = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

class BettingConfig:
    # thresholds and limits used by PredictionEngine.generate_tips
    min_confidence: float = 0.65
    min_ev: float = 0.1
    max_tips_per_match: int = 3
    bankroll_percentage: float = 0.02
    odds_format: str = "decimal"

class DatabaseManager:
    """
    Minimal sqlite-backed DatabaseManager implementing the methods expected by main.py:
      - check_recent_prediction(fixture_id) -> bool
      - store_match(match_data)
      - store_prediction(prediction_dict)
      - get_daily_performance() -> dict

    This implementation is intentionally simple and designed for local testing.
    For production use, replace with a more robust implementation (SQLAlchemy, migrations, connection pooling, etc).
    """

    def __init__(self, database_url: str = "sqlite:///data/db.sqlite"):
        # Normalize sqlite path format 'sqlite:///<path>'
        db_file = database_url
        if db_file.startswith("sqlite:///"):
            db_file = db_file.replace("sqlite:///", "")
        # Ensure data directory exists
        Path(db_file).parent.mkdir(parents=True, exist_ok=True)

        # Connect to sqlite (allow simple multi-thread usage)
        self.conn = sqlite3.connect(db_file, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        cur = self.conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS matches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fixture_id INTEGER,
            ts TEXT,
            payload TEXT
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fixture_id INTEGER,
            ts TEXT,
            payload TEXT
        )
        """)
        cur.execute("""
        CREATE INDEX IF NOT EXISTS ix_predictions_fixture ON predictions(fixture_id)
        """)
        self.conn.commit()

    def check_recent_prediction(self, fixture_id: int, minutes: int = 60) -> bool:
        """
        Return True if a prediction exists for fixture_id within last `minutes` minutes.
        main.py calls this to avoid re-processing the same match too often.
        """
        cur = self.conn.cursor()
        since = (datetime.utcnow() - timedelta(minutes=minutes)).isoformat()
        cur.execute("SELECT 1 FROM predictions WHERE fixture_id = ? AND ts >= ? LIMIT 1", (fixture_id, since))
        row = cur.fetchone()
        return bool(row)

    def store_match(self, match_data: Dict[str, Any]):
        """Store raw match payload for later analysis."""
        cur = self.conn.cursor()
        fixture_id = match_data.get("fixture", {}).get("id")
        payload = json.dumps(match_data)
        cur.execute("INSERT INTO matches (fixture_id, ts, payload) VALUES (?, ?, ?)",
                    (fixture_id, datetime.utcnow().isoformat(), payload))
        self.conn.commit()

    def store_prediction(self, prediction: Dict[str, Any]):
        """Store prediction result payload."""
        cur = self.conn.cursor()
        fixture_id = prediction.get("match_id")
        payload = json.dumps(prediction)
        cur.execute("INSERT INTO predictions (fixture_id, ts, payload) VALUES (?, ?, ?)",
                    (fixture_id, datetime.utcnow().isoformat(), payload))
        self.conn.commit()

    def get_daily_performance(self) -> Dict[str, float]:
        """
        Minimal daily performance summary used by BettingPredictor.send_daily_report.
        Returns a dict with keys used by main.py: win_rate, roi (both stubbed here), and predictions_last_24h.
        """
        cur = self.conn.cursor()
        since = (datetime.utcnow() - timedelta(days=1)).isoformat()
        cur.execute("SELECT COUNT(*) as c FROM predictions WHERE ts >= ?", (since,))
        row = cur.fetchone()
        total = row["c"] if row else 0
        # TODO: compute real win_rate and roi based on stored outcomes; return 0.0 as placeholder
        return {"predictions_last_24h": total, "win_rate": 0.0, "roi": 0.0}

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass
