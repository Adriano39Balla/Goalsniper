import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
from typing import List, Dict, Any, Optional
import logging
from config.settings import settings

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        self.conn = None
        self.connect()
    
    def connect(self):
        """Establish database connection"""
        try:
            if not settings.DATABASE_URL:
                logger.warning("DATABASE_URL not set, using in-memory storage")
                return
                
            self.conn = psycopg2.connect(
                settings.DATABASE_URL,
                cursor_factory=RealDictCursor
            )
            logger.info("Connected to database successfully")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            # Continue without database for development
            self.conn = None
    
    def execute_query(self, query: str, params: tuple = None) -> List[Dict]:
        """Execute query and return results"""
        if not self.conn:
            logger.warning("No database connection, returning empty results")
            return []
            
        try:
            with self.conn.cursor() as cur:
                cur.execute(query, params or ())
                if query.strip().upper().startswith('SELECT'):
                    return cur.fetchall()
                self.conn.commit()
                return []
        except psycopg2.InterfaceError:
            logger.warning("Database connection lost, reconnecting...")
            self.connect()
            return self.execute_query(query, params)
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return []
    
    def save_prediction(self, prediction_data: Dict[str, Any]) -> bool:
        """Save prediction to database"""
        if not self.conn:
            logger.warning("No database connection, skipping save")
            return False
            
        query = """
        INSERT INTO predictions (
            fixture_id, home_team, away_team, league_id, league_name,
            prediction_time, home_win_prob, away_win_prob, draw_prob,
            over_25_prob, under_25_prob, btts_yes_prob, btts_no_prob,
            confidence, recommended_bet, bet_type, stake_confidence,
            model_version, live_minute, current_score
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        params = (
            prediction_data.get('fixture_id', 0), 
            prediction_data.get('home_team', 'Unknown'), 
            prediction_data.get('away_team', 'Unknown'), 
            prediction_data.get('league_id', 0),
            prediction_data.get('league_name', 'Unknown'),
            prediction_data.get('prediction_time', pd.Timestamp.now()),
            prediction_data.get('home_win_prob', 0.33),
            prediction_data.get('away_win_prob', 0.33),
            prediction_data.get('draw_prob', 0.34),
            prediction_data.get('over_25_prob', 0.5),
            prediction_data.get('under_25_prob', 0.5),
            prediction_data.get('btts_yes_prob', 0.5),
            prediction_data.get('btts_no_prob', 0.5),
            prediction_data.get('confidence', 0.5),
            prediction_data.get('recommended_bet', 'NO_BET'),
            prediction_data.get('bet_type', 'NO_BET'),
            prediction_data.get('stake_confidence', 0.5),
            prediction_data.get('model_version', 'ensemble_v1'),
            prediction_data.get('live_minute'),
            prediction_data.get('current_score')
        )
        try:
            self.execute_query(query, params)
            return True
        except Exception as e:
            logger.error(f"Failed to save prediction: {e}")
            return False
    
    def save_bet_result(self, fixture_id: int, bet_type: str, success: bool, actual_odds: float = None):
        """Save bet result for model learning"""
        if not self.conn:
            return
            
        query = """
        INSERT INTO bet_results (
            fixture_id, bet_type, success, actual_odds, processed
        ) VALUES (%s, %s, %s, %s, %s)
        """
        self.execute_query(query, (fixture_id, bet_type, success, actual_odds, False))
    
    def get_training_data(self, limit: int = 10000) -> pd.DataFrame:
        """Get historical data for model training"""
        if not self.conn:
            logger.warning("No database connection, returning empty DataFrame")
            return pd.DataFrame()
            
        query = """
        SELECT * FROM historical_matches 
        WHERE home_goals IS NOT NULL AND away_goals IS NOT NULL
        ORDER BY fixture_date DESC LIMIT %s
        """
        results = self.execute_query(query, (limit,))
        return pd.DataFrame(results) if results else pd.DataFrame()
    
    def get_live_fixtures(self) -> List[Dict]:
        """Get currently live fixtures"""
        if not self.conn:
            return []
            
        query = """
        SELECT * FROM fixtures 
        WHERE status = 'Live' AND elapsed > 0
        """
        return self.execute_query(query)

db = DatabaseManager()
