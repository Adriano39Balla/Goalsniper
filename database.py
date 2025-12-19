"""
Database Manager for Supabase Integration
Handles all database operations with psycopg2 for optimal performance
"""

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from psycopg2.pool import ThreadedConnectionPool
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
from contextlib import contextmanager

from loguru import logger
from config import settings


class DatabaseManager:
    """
    High-performance database manager with connection pooling
    """
    
    def __init__(self):
        self.pool = ThreadedConnectionPool(
            minconn=2,
            maxconn=10,
            host=settings.DB_HOST,
            port=settings.DB_PORT,
            database=settings.DB_NAME,
            user=settings.DB_USER,
            password=settings.DB_PASSWORD
        )
        logger.info("Database connection pool initialized")
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = self.pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            self.pool.putconn(conn)
    
    async def initialize_schema(self):
        """
        Create all required database tables
        """
        logger.info("Initializing database schema...")
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Table 1: Live matches data
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS live_matches (
                    id SERIAL PRIMARY KEY,
                    fixture_id INTEGER NOT NULL,
                    league_id INTEGER,
                    league_name VARCHAR(255),
                    home_team VARCHAR(255),
                    away_team VARCHAR(255),
                    elapsed_minutes INTEGER,
                    status VARCHAR(10),
                    home_goals INTEGER,
                    away_goals INTEGER,
                    home_shots_total INTEGER,
                    away_shots_total INTEGER,
                    home_shots_on_target INTEGER,
                    away_shots_on_target INTEGER,
                    home_possession INTEGER,
                    away_possession INTEGER,
                    home_attacks INTEGER,
                    away_attacks INTEGER,
                    home_dangerous_attacks INTEGER,
                    away_dangerous_attacks INTEGER,
                    home_corners INTEGER,
                    away_corners INTEGER,
                    home_yellow_cards INTEGER,
                    away_yellow_cards INTEGER,
                    home_red_cards INTEGER,
                    away_red_cards INTEGER,
                    home_fouls INTEGER,
                    away_fouls INTEGER,
                    home_offsides INTEGER,
                    away_offsides INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_fixture_id ON live_matches(fixture_id);
                CREATE INDEX IF NOT EXISTS idx_timestamp ON live_matches(timestamp);
                CREATE INDEX IF NOT EXISTS idx_status ON live_matches(status);
            """)
            
            # Table 2: Predictions
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id SERIAL PRIMARY KEY,
                    fixture_id INTEGER NOT NULL,
                    market VARCHAR(100),
                    prediction VARCHAR(50),
                    probability FLOAT,
                    calibrated_probability FLOAT,
                    confidence_score FLOAT,
                    expected_value FLOAT,
                    model_version VARCHAR(50),
                    features_used TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    outcome VARCHAR(20),
                    correct BOOLEAN,
                    profit FLOAT,
                    odds FLOAT,
                    sent_to_telegram BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_pred_fixture ON predictions(fixture_id);
                CREATE INDEX IF NOT EXISTS idx_pred_market ON predictions(market);
                CREATE INDEX IF NOT EXISTS idx_pred_timestamp ON predictions(timestamp);
                CREATE INDEX IF NOT EXISTS idx_pred_outcome ON predictions(outcome);
            """)
            
            # Table 3: Model performance tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    id SERIAL PRIMARY KEY,
                    market VARCHAR(100),
                    accuracy FLOAT,
                    log_loss FLOAT,
                    brier_score FLOAT,
                    auc_roc FLOAT,
                    calibration_error FLOAT,
                    profit_loss FLOAT,
                    total_predictions INTEGER,
                    winning_predictions INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    model_version VARCHAR(50)
                );
                
                CREATE INDEX IF NOT EXISTS idx_perf_market ON model_performance(market);
                CREATE INDEX IF NOT EXISTS idx_perf_timestamp ON model_performance(timestamp);
            """)
            
            # Table 4: Training history
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_history (
                    id SERIAL PRIMARY KEY,
                    market VARCHAR(100),
                    samples_count INTEGER,
                    training_accuracy FLOAT,
                    validation_accuracy FLOAT,
                    training_time_seconds FLOAT,
                    model_version VARCHAR(50),
                    hyperparameters TEXT,
                    feature_importance TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_train_market ON training_history(market);
                CREATE INDEX IF NOT EXISTS idx_train_timestamp ON training_history(timestamp);
            """)
            
            # Table 5: System logs
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_logs (
                    id SERIAL PRIMARY KEY,
                    level VARCHAR(20),
                    message TEXT,
                    component VARCHAR(100),
                    details TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_logs_level ON system_logs(level);
                CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON system_logs(timestamp);
            """)
            
            # Table 6: Configuration and state
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_state (
                    id SERIAL PRIMARY KEY,
                    key VARCHAR(100) UNIQUE,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            conn.commit()
            logger.info("Database schema initialized successfully")
    
    async def save_live_match(self, match_data: Dict):
        """Save live match data"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO live_matches (
                    fixture_id, league_id, league_name, home_team, away_team,
                    elapsed_minutes, status, home_goals, away_goals,
                    home_shots_total, away_shots_total, home_shots_on_target, away_shots_on_target,
                    home_possession, away_possession, home_attacks, away_attacks,
                    home_dangerous_attacks, away_dangerous_attacks, home_corners, away_corners,
                    home_yellow_cards, away_yellow_cards, home_red_cards, away_red_cards,
                    home_fouls, away_fouls, home_offsides, away_offsides, timestamp
                ) VALUES (
                    %(fixture_id)s, %(league_id)s, %(league_name)s, %(home_team)s, %(away_team)s,
                    %(elapsed_minutes)s, %(status)s, %(home_goals)s, %(away_goals)s,
                    %(home_shots_total)s, %(away_shots_total)s, %(home_shots_on_target)s, %(away_shots_on_target)s,
                    %(home_possession)s, %(away_possession)s, %(home_attacks)s, %(away_attacks)s,
                    %(home_dangerous_attacks)s, %(away_dangerous_attacks)s, %(home_corners)s, %(away_corners)s,
                    %(home_yellow_cards)s, %(away_yellow_cards)s, %(home_red_cards)s, %(away_red_cards)s,
                    %(home_fouls)s, %(away_fouls)s, %(home_offsides)s, %(away_offsides)s, %(timestamp)s
                )
            """, match_data)
    
    async def save_historical_matches(self, df: pd.DataFrame):
        """Bulk save historical matches"""
        if df.empty:
            return
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Convert DataFrame to list of tuples
            records = df.to_dict('records')
            
            for record in records:
                try:
                    await self.save_live_match(record)
                except Exception as e:
                    logger.error(f"Error saving match {record.get('fixture_id')}: {e}")
        
        logger.info(f"Saved {len(df)} historical matches")
    
    async def save_prediction(self, prediction: Dict):
        """Save prediction to database"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO predictions (
                    fixture_id, market, prediction, probability, calibrated_probability,
                    confidence_score, expected_value, model_version, features_used, timestamp
                ) VALUES (
                    %(fixture_id)s, %(market)s, %(prediction)s, %(probability)s, %(calibrated_probability)s,
                    %(confidence_score)s, %(expected_value)s, %(model_version)s, %(features_used)s, %(timestamp)s
                )
                RETURNING id
            """, prediction)
            
            prediction_id = cursor.fetchone()[0]
            return prediction_id
    
    async def update_prediction_outcome(
        self, 
        prediction_id: int, 
        outcome: str, 
        correct: bool, 
        profit: float,
        odds: float
    ):
        """Update prediction with actual outcome"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE predictions
                SET outcome = %s, correct = %s, profit = %s, odds = %s
                WHERE id = %s
            """, (outcome, correct, profit, odds, prediction_id))
    
    async def save_model_performance(self, performance_dict: Dict):
        """Save model performance metrics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            for market, perf in performance_dict.items():
                cursor.execute("""
                    INSERT INTO model_performance (
                        market, accuracy, log_loss, brier_score, auc_roc,
                        calibration_error, profit_loss, total_predictions,
                        winning_predictions, model_version
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                """, (
                    market, perf.accuracy, perf.log_loss, perf.brier_score,
                    perf.auc_roc, perf.calibration_error, perf.profit_loss,
                    perf.total_predictions, perf.winning_predictions,
                    'v1'
                ))
    
    async def get_recent_predictions(self, days: int = 7) -> pd.DataFrame:
        """Get recent predictions for analysis"""
        with self.get_connection() as conn:
            query = """
                SELECT * FROM predictions
                WHERE timestamp > NOW() - INTERVAL '%s days'
                ORDER BY timestamp DESC
            """
            df = pd.read_sql_query(query, conn, params=(days,))
            return df
    
    async def get_training_data(self, days: int = 30) -> pd.DataFrame:
        """Get historical match data for training"""
        with self.get_connection() as conn:
            query = """
                SELECT * FROM live_matches
                WHERE timestamp > NOW() - INTERVAL '%s days'
                ORDER BY timestamp ASC
            """
            df = pd.read_sql_query(query, conn, params=(days,))
            return df
    
    async def get_market_performance(self, market: str, days: int = 30) -> Optional[Dict]:
        """Get performance metrics for a specific market"""
        with self.get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT * FROM model_performance
                WHERE market = %s AND timestamp > NOW() - INTERVAL '%s days'
                ORDER BY timestamp DESC
                LIMIT 1
            """, (market, days))
            
            result = cursor.fetchone()
            return dict(result) if result else None
    
    async def get_system_state(self, key: str) -> Optional[str]:
        """Get system state value"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT value FROM system_state WHERE key = %s
            """, (key,))
            
            result = cursor.fetchone()
            return result[0] if result else None
    
    async def set_system_state(self, key: str, value: str):
        """Set system state value"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO system_state (key, value, updated_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (key) DO UPDATE
                SET value = EXCLUDED.value, updated_at = NOW()
            """, (key, value))
    
    async def log_system_event(self, level: str, message: str, component: str, details: str = None):
        """Log system event to database"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO system_logs (level, message, component, details)
                VALUES (%s, %s, %s, %s)
            """, (level, message, component, details))
    
    async def get_statistics_dashboard(self) -> Dict:
        """Get comprehensive statistics for dashboard"""
        with self.get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Total predictions
            cursor.execute("SELECT COUNT(*) as total FROM predictions")
            total_predictions = cursor.fetchone()['total']
            
            # Accuracy by market
            cursor.execute("""
                SELECT market, 
                       COUNT(*) as total,
                       SUM(CASE WHEN correct THEN 1 ELSE 0 END) as correct,
                       AVG(CASE WHEN correct THEN 1.0 ELSE 0.0 END) as accuracy,
                       SUM(profit) as total_profit
                FROM predictions
                WHERE outcome IS NOT NULL
                GROUP BY market
                ORDER BY accuracy DESC
            """)
            market_stats = cursor.fetchall()
            
            # Recent performance (last 7 days)
            cursor.execute("""
                SELECT DATE(timestamp) as date,
                       COUNT(*) as predictions,
                       AVG(CASE WHEN correct THEN 1.0 ELSE 0.0 END) as accuracy,
                       SUM(profit) as profit
                FROM predictions
                WHERE timestamp > NOW() - INTERVAL '7 days'
                  AND outcome IS NOT NULL
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
            """)
            daily_stats = cursor.fetchall()
            
            return {
                'total_predictions': total_predictions,
                'market_statistics': [dict(row) for row in market_stats],
                'daily_performance': [dict(row) for row in daily_stats]
            }
    
    def close(self):
        """Close all database connections"""
        self.pool.closeall()
        logger.info("Database connections closed")


async def test_database():
    """Test database operations"""
    logger.add("logs/database_{time}.log")
    
    db = DatabaseManager()
    
    try:
        # Initialize schema
        await db.initialize_schema()
        logger.info("Schema initialized")
        
        # Test system state
        await db.set_system_state('last_scan', datetime.now().isoformat())
        value = await db.get_system_state('last_scan')
        logger.info(f"System state test: {value}")
        
        # Test statistics
        stats = await db.get_statistics_dashboard()
        logger.info(f"Dashboard stats: {stats}")
        
    finally:
        db.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_database())
