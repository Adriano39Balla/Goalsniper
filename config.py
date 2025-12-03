import os
from pydantic_settings import BaseSettings
from typing import Optional, List, Dict, Any
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
import json
from datetime import datetime
import pandas as pd
import numpy as np
from loguru import logger

class Settings(BaseSettings):
    API_FOOTBALL_KEY: str = ""
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/betting_predictor"
    REDIS_URL: Optional[str] = None
    
    class Config:
        env_file = ".env"

class BettingConfig:
    min_confidence: float = 0.65
    min_ev: float = 0.1
    max_tips_per_match: int = 3
    bankroll_percentage: float = 0.02
    odds_format: str = "decimal"

# Define consistent feature columns for all models
FEATURE_COLUMNS = [
    'home_team_rating', 'away_team_rating', 'home_score', 'away_score',
    'goal_difference', 'total_goals', 'minute', 'time_ratio',
    'league_rank', 'hour_of_day', 'day_of_week', 'month',
    'momentum', 'scoring_pressure', 'home_possession', 'away_possession',
    'home_shots_on_goal', 'away_shots_on_goal', 'implied_prob_home',
    'implied_prob_draw', 'implied_prob_away'
]

class DatabaseManager:
    """Database manager for Supabase/PostgreSQL"""
    
    def __init__(self, connection_string: Optional[str] = None):
        self.settings = Settings()
        self.connection_string = connection_string or self.settings.DATABASE_URL
        self.init_database()
    
    def init_database(self):
        """Initialize database tables if they don't exist"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Create matches table
                    cur.execute("""
                    CREATE TABLE IF NOT EXISTS matches (
                        id SERIAL PRIMARY KEY,
                        match_id INTEGER UNIQUE,
                        home_team VARCHAR(255),
                        away_team VARCHAR(255),
                        home_score INTEGER,
                        away_score INTEGER,
                        timestamp TIMESTAMP,
                        status VARCHAR(50),
                        league VARCHAR(255),
                        country VARCHAR(100),
                        home_odds DECIMAL(10,2),
                        draw_odds DECIMAL(10,2),
                        away_odds DECIMAL(10,2),
                        over_2_5_odds DECIMAL(10,2),
                        btts_yes_odds DECIMAL(10,2),
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                    """)
                    
                    # Create predictions table
                    cur.execute("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id SERIAL PRIMARY KEY,
                        match_id INTEGER,
                        predictions JSONB,
                        probabilities JSONB,
                        tips JSONB,
                        confidence DECIMAL(5,4),
                        features JSONB,
                        expected_value JSONB,
                        outcome VARCHAR(50),
                        is_correct BOOLEAN,
                        timestamp TIMESTAMP DEFAULT NOW(),
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                    """)
                    
                    # Create training_sessions table
                    cur.execute("""
                    CREATE TABLE IF NOT EXISTS training_sessions (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP,
                        model_type VARCHAR(50),
                        avg_accuracy DECIMAL(5,4),
                        avg_roc_auc DECIMAL(5,4),
                        training_samples INTEGER,
                        metrics JSONB,
                        model_path TEXT,
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                    """)
                    
                    # Create performance_metrics table
                    cur.execute("""
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id SERIAL PRIMARY KEY,
                        date DATE,
                        predictions_made INTEGER,
                        tips_sent INTEGER,
                        correct_predictions INTEGER,
                        total_predictions INTEGER,
                        roi DECIMAL(10,4),
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                    """)
                    
                    # Create indices
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_matches_match_id ON matches(match_id)")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_matches_timestamp ON matches(timestamp)")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_predictions_match_id ON predictions(match_id)")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp)")
                    
                    conn.commit()
                    logger.info("Database tables initialized successfully")
                    
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    @contextmanager
    def get_connection(self):
        """Get database connection with context manager"""
        conn = None
        try:
            conn = psycopg2.connect(self.connection_string, cursor_factory=RealDictCursor)
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def execute_query(self, query: str, params: tuple = None, fetch: bool = True):
        """Execute SQL query"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params or ())
                    if fetch and query.strip().upper().startswith('SELECT'):
                        return cur.fetchall()
                    elif fetch:
                        return cur.fetchone()
        except Exception as e:
            logger.error(f"Query execution error: {e}")
            return []
    
    def store_prediction(self, prediction: dict):
        """Store prediction in database"""
        query = """
        INSERT INTO predictions 
        (match_id, predictions, probabilities, tips, confidence, features, expected_value, timestamp)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (match_id) DO UPDATE SET
            predictions = EXCLUDED.predictions,
            probabilities = EXCLUDED.probabilities,
            tips = EXCLUDED.tips,
            confidence = EXCLUDED.confidence,
            features = EXCLUDED.features,
            expected_value = EXCLUDED.expected_value,
            timestamp = EXCLUDED.timestamp
        """
        
        self.execute_query(query, (
            prediction.get('match_id'),
            json.dumps(prediction.get('predictions', {})),
            json.dumps(prediction.get('probabilities', {})),
            json.dumps(prediction.get('tips', [])),
            prediction.get('confidence', 0),
            json.dumps(prediction.get('features', {})),
            json.dumps(prediction.get('expected_value', {})),
            prediction.get('timestamp')
        ), fetch=False)
    
    def check_recent_prediction(self, match_id: int, minutes: int = 5) -> bool:
        """Check if prediction was made recently for this match"""
        query = """
        SELECT COUNT(*) as count FROM predictions 
        WHERE match_id = %s AND timestamp > NOW() - INTERVAL '%s minutes'
        """
        result = self.execute_query(query, (match_id, minutes))
        return result[0]['count'] > 0 if result else False
    
    def get_daily_performance(self) -> dict:
        """Get daily performance metrics"""
        query = """
        SELECT 
            COUNT(*) as total_predictions,
            SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) as correct_predictions,
            AVG(CASE WHEN is_correct THEN 1.0 ELSE 0.0 END) as win_rate,
            COALESCE(AVG(roi), 0) as roi
        FROM predictions 
        WHERE DATE(timestamp) = CURRENT_DATE
        """
        
        result = self.execute_query(query)
        if result:
            row = result[0]
            return {
                'total_predictions': row['total_predictions'] or 0,
                'correct_predictions': row['correct_predictions'] or 0,
                'win_rate': float(row['win_rate'] or 0),
                'roi': float(row['roi'] or 0)
            }
        return {'total_predictions': 0, 'correct_predictions': 0, 'win_rate': 0, 'roi': 0}
    
    def get_recent_accuracy(self, days: int = 7) -> float:
        """Get recent prediction accuracy"""
        query = """
        SELECT 
            AVG(CASE WHEN is_correct THEN 1.0 ELSE 0.0 END) as accuracy
        FROM predictions 
        WHERE timestamp > NOW() - INTERVAL '%s days'
        AND is_correct IS NOT NULL
        """
        
        result = self.execute_query(query, (days,))
        if result and result[0]['accuracy']:
            return float(result[0]['accuracy'])
        return 0.0
    
    def store_match(self, match_data: dict):
        """Store match data"""
        query = """
        INSERT INTO matches 
        (match_id, home_team, away_team, home_score, away_score, timestamp, 
         status, league, country, home_odds, draw_odds, away_odds,
         over_2_5_odds, btts_yes_odds)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (match_id) DO UPDATE SET
            home_score = EXCLUDED.home_score,
            away_score = EXCLUDED.away_score,
            status = EXCLUDED.status,
            timestamp = EXCLUDED.timestamp
        """
        
        # Extract odds from match data
        odds = match_data.get('odds', {})
        
        self.execute_query(query, (
            match_data.get('fixture', {}).get('id'),
            match_data.get('teams', {}).get('home', {}).get('name'),
            match_data.get('teams', {}).get('away', {}).get('name'),
            match_data.get('goals', {}).get('home'),
            match_data.get('goals', {}).get('away'),
            match_data.get('fixture', {}).get('date'),
            match_data.get('fixture', {}).get('status', {}).get('long'),
            match_data.get('league', {}).get('name'),
            match_data.get('league', {}).get('country'),
            odds.get('home', 0),
            odds.get('draw', 0),
            odds.get('away', 0),
            odds.get('over_2_5', 0),
            odds.get('btts_yes', 0)
        ), fetch=False)
