import psycopg2
from psycopg2.extras import RealDictCursor, Json
from contextlib import contextmanager
import os
from datetime import datetime
from typing import Optional, Dict, List, Any
from .logger import logger

class DatabaseManager:
    """Supabase PostgreSQL database manager"""
    
    def __init__(self):
        self.connection_params = {
            'host': os.getenv('DB_HOST', 'db.supabase.co'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME', 'postgres'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD'),
            'sslmode': 'require'
        }
    
    @contextmanager
    def get_connection(self):
        """Get database connection with context manager"""
        conn = None
        try:
            conn = psycopg2.connect(**self.connection_params)
            conn.autocommit = False
            yield conn
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()
    
    def execute_query(self, query: str, params: tuple = None, fetch: bool = False):
        """Execute SQL query"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, params or ())
                if fetch:
                    return cursor.fetchall()
                conn.commit()
    
    def save_prediction(self, prediction_data: Dict[str, Any]) -> Optional[str]:
        """Save prediction to database"""
    
        try:
            query = """
            INSERT INTO predictions (
                match_id, league_id, home_team, away_team,
                prediction_type, prediction_value, confidence,
                probability, features, model_version,
                created_at, match_time
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """
        
            # Extract predictions
            predictions = prediction_data.get('predictions', {})
        
            # For now, save first prediction type found
            # You might want to modify this to save all prediction types
            for pred_type, pred_data in predictions.items():
                params = (
                    prediction_data.get('match_id'),
                    prediction_data.get('league_id'),
                    prediction_data.get('home_team'),
                    prediction_data.get('away_team'),
                    pred_type,
                    pred_data.get('prediction'),
                    float(pred_data.get('confidence', 0)),
                    Json({'probability': pred_data.get('probability')}),
                    Json(prediction_data.get('features', {})),
                    os.getenv('MODEL_VERSION', 'v1.0'),
                    datetime.utcnow(),
                    prediction_data.get('match_time')
                )
            
                result = self.execute_query(query, params, fetch=True)
                if result:
                    return str(result[0]['id'])
        
            return None
        
        except Exception as e:
            logger.error(f"Error saving prediction: {e}")
            return None
    
    def save_bet_result(self, prediction_id: str, result_data: Dict[str, Any]):
        """Save actual bet result for self-learning"""
        query = """
        INSERT INTO bet_results (
            prediction_id, actual_result, is_correct,
            profit_loss, odds_used, bet_amount,
            result_details, analyzed_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        params = (
            prediction_id,
            result_data['actual_result'],
            result_data['is_correct'],
            result_data.get('profit_loss', 0),
            result_data.get('odds_used', 1.0),
            result_data.get('bet_amount', 0),
            Json(result_data.get('details', {})),
            datetime.utcnow()
        )
        
        self.execute_query(query, params)
    
    def get_training_data(self, limit: int = 10000) -> List[Dict]:
        """Retrieve training data from database"""
        query = """
        SELECT 
            p.features,
            br.actual_result,
            p.prediction_type,
            p.confidence,
            p.match_time
        FROM predictions p
        JOIN bet_results br ON p.id = br.prediction_id
        WHERE br.actual_result IS NOT NULL
        ORDER BY p.match_time DESC
        LIMIT %s
        """
        
        return self.execute_query(query, (limit,), fetch=True)
    
    def get_live_matches(self) -> List[Dict]:
        """Get currently live matches"""
        query = """
        SELECT 
            m.match_id, m.league_id, m.home_team, m.away_team,
            m.current_score, m.minute, m.match_stats,
            m.last_updated, m.api_match_data
        FROM live_matches m
        WHERE m.status = 'LIVE'
        AND m.minute BETWEEN 20 AND 80
        ORDER BY m.league_priority DESC, m.minute DESC
        """
        
        return self.execute_query(query, fetch=True)
    
    def create_tables(self):
        """Create necessary tables if they don't exist"""
        tables_sql = [
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                match_id INTEGER UNIQUE,
                league_id INTEGER,
                home_team VARCHAR(100),
                away_team VARCHAR(100),
                prediction_type VARCHAR(20),
                prediction_value VARCHAR(50),
                confidence FLOAT,
                probability JSONB,
                features JSONB,
                model_version VARCHAR(20),
                created_at TIMESTAMP DEFAULT NOW(),
                match_time TIMESTAMP,
                UNIQUE(match_id, prediction_type)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS bet_results (
                id SERIAL PRIMARY KEY,
                prediction_id INTEGER REFERENCES predictions(id),
                actual_result VARCHAR(50),
                is_correct BOOLEAN,
                profit_loss FLOAT,
                odds_used FLOAT,
                bet_amount FLOAT,
                result_details JSONB,
                analyzed_at TIMESTAMP DEFAULT NOW()
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS live_matches (
                id SERIAL PRIMARY KEY,
                match_id INTEGER UNIQUE,
                league_id INTEGER,
                home_team VARCHAR(100),
                away_team VARCHAR(100),
                current_score VARCHAR(10),
                minute INTEGER,
                status VARCHAR(20),
                match_stats JSONB,
                api_match_data JSONB,
                last_updated TIMESTAMP DEFAULT NOW(),
                league_priority INTEGER DEFAULT 1
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS model_performance (
                id SERIAL PRIMARY KEY,
                model_type VARCHAR(50),
                version VARCHAR(20),
                accuracy FLOAT,
                precision FLOAT,
                recall FLOAT,
                f1_score FLOAT,
                roc_auc FLOAT,
                training_date TIMESTAMP DEFAULT NOW(),
                parameters JSONB,
                feature_importance JSONB
            );
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_predictions_match_time 
            ON predictions(match_time);
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_bet_results_analyzed 
            ON bet_results(analyzed_at);
            """
        ]
        
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                for sql in tables_sql:
                    cursor.execute(sql)
                conn.commit()
        
        logger.info("Database tables created/verified successfully")
