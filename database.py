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
