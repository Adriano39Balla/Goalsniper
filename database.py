import psycopg2
from psycopg2.extras import RealDictCursor
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import json
from contextlib import contextmanager

class DatabaseManager:
    def __init__(self):
        self.connection_string = os.getenv('DATABASE_URL')
    
    @contextmanager
    def get_connection(self):
        conn = psycopg2.connect(self.connection_string, cursor_factory=RealDictCursor)
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def execute_query(self, query: str, params: tuple = None) -> List[Dict]:
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params or ())
                if query.strip().upper().startswith('SELECT'):
                    return cur.fetchall()
                return []
    
    def store_prediction(self, prediction: Dict):
        query = """
        INSERT INTO predictions 
        (match_id, predictions, probabilities, tips, confidence, features, timestamp)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        self.execute_query(query, (
            prediction['match_id'],
            json.dumps(prediction['predictions']),
            json.dumps(prediction['probabilities']),
            json.dumps(prediction['tips']),
            prediction['confidence'],
            json.dumps(prediction['features']),
            prediction['timestamp']
        ))

    def check_recent_prediction(self, match_id: int, minutes: int = 15) -> bool:
        """
        Returns True if a prediction for this match already exists 
        within the last X minutes.
        """
        query = """
        SELECT timestamp FROM predictions
        WHERE match_id = %s
        AND timestamp >= NOW() - INTERVAL '%s minutes'
        ORDER BY timestamp DESC
        LIMIT 1
        """
        rows = self.execute_query(query, (match_id, minutes))
        return len(rows) > 0
    

    def get_recent_predictions(self, minutes: int = 60):
        """
        Fetch all predictions stored within the last X minutes.
        Useful for monitoring or auto-tuning.
        """
        query = """
        SELECT * FROM predictions
        WHERE timestamp >= NOW() - INTERVAL '%s minutes'
        ORDER BY timestamp DESC
        """
        return self.execute_query(query, (minutes,))
    

    def log_error(self, source: str, message: str, details: dict = None):
        """
        Store backend errors for debugging + learning loops.
        Create an errors table first.
        """
        query = """
        INSERT INTO errors (source, message, details, timestamp)
        VALUES (%s, %s, %s, NOW())
        """
        self.execute_query(query, (
            source,
            message,
            json.dumps(details or {})
        ))
