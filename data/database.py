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
            self.conn = psycopg2.connect(
                settings.DATABASE_URL,
                cursor_factory=RealDictCursor
            )
            logger.info("Connected to database successfully")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def execute_query(self, query: str, params: tuple = None) -> List[Dict]:
        """Execute query and return results"""
        try:
            with self.conn.cursor() as cur:
                cur.execute(query, params or ())
                if query.strip().upper().startswith('SELECT'):
                    return cur.fetchall()
                self.conn.commit()
                return []
        except psycopg2.InterfaceError:
            self.connect()
            return self.execute_query(query, params)
    
    def save_prediction(self, prediction_data: Dict[str, Any]) -> bool:
        """Save prediction to database"""
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
            prediction_data['fixture_id'], prediction_data['home_team'], 
            prediction_data['away_team'], prediction_data['league_id'],
            prediction_data['league_name'], prediction_data['prediction_time'],
            prediction_data['home_win_prob'], prediction_data['away_win_prob'],
            prediction_data['draw_prob'], prediction_data['over_25_prob'],
            prediction_data['under_25_prob'], prediction_data['btts_yes_prob'],
            prediction_data['btts_no_prob'], prediction_data['confidence'],
            prediction_data['recommended_bet'], prediction_data['bet_type'],
            prediction_data['stake_confidence'], prediction_data['model_version'],
            prediction_data.get('live_minute'), prediction_data.get('current_score')
        )
        try:
            self.execute_query(query, params)
            return True
        except Exception as e:
            logger.error(f"Failed to save prediction: {e}")
            return False
    
    def save_bet_result(self, fixture_id: int, bet_type: str, success: bool, actual_odds: float = None):
        """Save bet result for model learning"""
        query = """
        INSERT INTO bet_results (
            fixture_id, bet_type, success, actual_odds, processed
        ) VALUES (%s, %s, %s, %s, %s)
        """
        self.execute_query(query, (fixture_id, bet_type, success, actual_odds, False))
    
    def get_training_data(self, limit: int = 10000) -> pd.DataFrame:
        """Get historical data for model training"""
        query = """
        SELECT * FROM historical_matches 
        WHERE home_goals IS NOT NULL AND away_goals IS NOT NULL
        ORDER BY fixture_date DESC LIMIT %s
        """
        results = self.execute_query(query, (limit,))
        return pd.DataFrame(results)
    
    def get_live_fixtures(self) -> List[Dict]:
        """Get currently live fixtures"""
        query = """
        SELECT * FROM fixtures 
        WHERE status = 'Live' AND elapsed > 0
        """
        return self.execute_query(query)

db = DatabaseManager()
