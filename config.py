import os
from pydantic_settings import BaseSettings
from typing import Optional, List, Dict, Any, Tuple
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
    # Calibrated thresholds
    min_confidence: float = 0.68  # Increased from 0.65
    min_ev: float = 0.15  # Increased from 0.1
    max_tips_per_match: int = 2  # Reduced from 3
    bankroll_percentage: float = 0.02
    odds_format: str = "decimal"
    
    # Time-based calibration
    late_game_minute: int = 80
    extra_time_minute: int = 90
    high_scoring_threshold: float = 2.5  # Goals per game
    
    # Market-specific thresholds
    home_win_confidence: float = 0.75
    over_25_confidence: float = 0.72
    btts_confidence: float = 0.70
    
    # Minimum minutes for valid prediction
    min_minutes_for_prediction: int = 20

# Define consistent feature columns for all models
FEATURE_COLUMNS = [
    'home_team_rating', 'away_team_rating', 'home_score', 'away_score',
    'goal_difference', 'total_goals', 'minute', 'time_ratio',
    'league_rank', 'hour_of_day', 'day_of_week', 'month',
    'momentum', 'scoring_pressure', 'home_possession', 'away_possession',
    'home_shots_on_goal', 'away_shots_on_goal', 'implied_prob_home',
    'implied_prob_draw', 'implied_prob_away', 'is_late_game',
    'is_extra_time', 'goals_needed_for_over', 'can_btts_happen'
]

class CalibrationEngine:
    """Engine to calibrate predictions based on game state"""
    
    @staticmethod
    def calibrate_for_game_state(match_data: Dict, predictions: Dict, probabilities: Dict) -> Tuple[Dict, Dict]:
        """Calibrate predictions based on current game state"""
        minute = match_data.get('fixture', {}).get('status', {}).get('elapsed', 0)
        home_score = match_data.get('goals', {}).get('home', 0)
        away_score = match_data.get('goals', {}).get('away', 0)
        total_goals = home_score + away_score
        
        calibrated_predictions = predictions.copy()
        calibrated_probabilities = probabilities.copy()
        
        # Apply game state logic
        if minute > 0:
            # BTTS calibration
            if 'btts' in calibrated_predictions:
                # If both teams have scored, BTTS is already YES
                if home_score > 0 and away_score > 0:
                    calibrated_predictions['btts'] = 1
                    calibrated_probabilities['btts'] = max(calibrated_probabilities.get('btts', 0), 0.95)
                # If one team hasn't scored and it's late game, reduce probability
                elif minute > 80 and (home_score == 0 or away_score == 0):
                    calibrated_probabilities['btts'] *= 0.7
            
            # Over 2.5 calibration
            if 'over_2_5' in calibrated_predictions:
                goals_needed = 3 - total_goals
                if goals_needed <= 0:
                    # Already over 2.5
                    calibrated_predictions['over_2_5'] = 1
                    calibrated_probabilities['over_2_5'] = 0.99
                elif minute > 80 and goals_needed > 0:
                    # Late game, needs goals - reduce probability
                    time_left = max(0, 90 - minute)
                    probability_factor = time_left / 15  # 15 minutes is reasonable time for a goal
                    calibrated_probabilities['over_2_5'] *= probability_factor
            
            # Home win calibration
            if 'home_win' in calibrated_predictions:
                if minute > 80:
                    if home_score > away_score:
                        # Already winning, high probability
                        calibrated_predictions['home_win'] = 1
                        calibrated_probabilities['home_win'] = max(calibrated_probabilities.get('home_win', 0), 0.85)
                    elif home_score < away_score:
                        # Losing late, very low probability
                        calibrated_probabilities['home_win'] *= 0.3
                    elif home_score == away_score:
                        # Drawing late, medium probability
                        calibrated_probabilities['home_win'] *= 0.7
        
        return calibrated_predictions, calibrated_probabilities
    
    @staticmethod
    def should_suppress_tip(tip: Dict, match_data: Dict, other_tips: List[Dict]) -> bool:
        """Determine if a tip should be suppressed based on game logic"""
        minute = match_data.get('fixture', {}).get('status', {}).get('elapsed', 0)
        home_score = match_data.get('goals', {}).get('home', 0)
        away_score = match_data.get('goals', {}).get('away', 0)
        total_goals = home_score + away_score
        
        tip_type = tip['type']
        
        # Suppress tips in very late game
        if minute > 105 and tip_type in ['Over/Under', '1X2']:
            return True
        
        # Check for conflicting tips
        if tip_type == '1X2' and tip['prediction'] == 'Home Win':
            # Don't recommend Home Win if it's a draw late in game
            if minute > 80 and home_score == away_score:
                return True
        
        if tip_type == 'Over/Under' and tip['prediction'] == 'Over 2.5 Goals':
            # If we need multiple goals very late, suppress
            if minute > 85 and (3 - total_goals) > 1:
                return True
        
        # Don't recommend BTTS if one team hasn't scored and it's very late
        if tip_type == 'BTTS' and minute > 85 and (home_score == 0 or away_score == 0):
            return True
        
        return False
    
    @staticmethod
    def calculate_time_adjusted_probability(base_prob: float, minute: int, prediction_type: str) -> float:
        """Adjust probability based on game time"""
        if minute <= 0:
            return base_prob
        
        # Different decay rates for different prediction types
        decay_rates = {
            '1X2': 0.7,  # Outcome becomes more certain as time passes
            'Over/Under': 0.8,  # Goals become less likely in late game
            'BTTS': 0.75  # BTTS becomes less likely if teams haven't scored
        }
        
        decay_rate = decay_rates.get(prediction_type, 0.8)
        
        # Apply time decay after 60 minutes
        if minute > 60:
            time_factor = 1 - ((minute - 60) / 30) * (1 - decay_rate)  # Linear decay from 60-90 minutes
            time_factor = max(0.3, time_factor)  # Don't go below 30%
            return base_prob * time_factor
        
        return base_prob

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
                    
                    # Create calibration_logs table
                    cur.execute("""
                    CREATE TABLE IF NOT EXISTS calibration_logs (
                        id SERIAL PRIMARY KEY,
                        match_id INTEGER,
                        original_probabilities JSONB,
                        calibrated_probabilities JSONB,
                        calibration_factors JSONB,
                        timestamp TIMESTAMP DEFAULT NOW()
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
    
    def log_calibration(self, match_id: int, original_probs: Dict, calibrated_probs: Dict, factors: Dict):
        """Log calibration changes"""
        query = """
        INSERT INTO calibration_logs 
        (match_id, original_probabilities, calibrated_probabilities, calibration_factors)
        VALUES (%s, %s, %s, %s)
        """
        
        self.execute_query(query, (
            match_id,
            json.dumps(original_probs),
            json.dumps(calibrated_probs),
            json.dumps(factors)
        ), fetch=False)
    
    def check_recent_prediction(self, match_id: int, minutes: int = 5) -> bool:
        """Check if prediction was made recently for this match"""
        query = """
        SELECT COUNT(*) as count FROM predictions 
        WHERE match_id = %s AND timestamp > NOW() - INTERVAL '%s minutes'
        """
        result = self.execute_query(query, (match_id, minutes))
        return result[0]['count'] > 0 if result else False
