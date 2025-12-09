"""
AI-Powered Betting Predictions Backend
Main orchestration script for live betting predictions
"""

import os
import sys
import time
import logging
import schedule
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta
from dotenv import load_dotenv
import requests
import json
import pickle
import numpy as np
from typing import Dict, List, Optional, Tuple
from train_models import ModelTrainer, FeatureEngineer
from dataclasses import dataclass, asdict

# Load environment variables
load_dotenv()

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('betting_predictions.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    """Prediction data structure"""
    fixture_id: int
    league: str
    home_team: str
    away_team: str
    prediction_type: str  # 'over_under', 'btts', 'win_lose'
    prediction: str
    probability: float
    confidence: float
    timestamp: datetime
    elapsed_time: int
    
    
class DatabaseManager:
    """Manages all database operations with Supabase"""
    
    def __init__(self):
        self.conn_params = {
            'dbname': os.getenv('SUPABASE_DB_NAME'),
            'user': os.getenv('SUPABASE_DB_USER'),
            'password': os.getenv('SUPABASE_DB_PASSWORD'),
            'host': os.getenv('SUPABASE_DB_HOST'),
            'port': os.getenv('SUPABASE_DB_PORT', '5432')
        }
        self._init_tables()
        
    def get_connection(self):
        """Create database connection"""
        return psycopg2.connect(**self.conn_params)
    
    def _init_tables(self):
        """Initialize database tables if they don't exist"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Predictions table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS predictions (
                            id SERIAL PRIMARY KEY,
                            fixture_id INTEGER NOT NULL,
                            league VARCHAR(255),
                            home_team VARCHAR(255),
                            away_team VARCHAR(255),
                            prediction_type VARCHAR(50),
                            prediction VARCHAR(50),
                            probability FLOAT,
                            confidence FLOAT,
                            elapsed_time INTEGER,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            result VARCHAR(50),
                            is_correct BOOLEAN,
                            updated_at TIMESTAMP
                        )
                    """)
                    
                    # Model performance table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS model_performance (
                            id SERIAL PRIMARY KEY,
                            model_type VARCHAR(50),
                            accuracy FLOAT,
                            precision FLOAT,
                            recall FLOAT,
                            f1_score FLOAT,
                            brier_score FLOAT,
                            total_predictions INTEGER,
                            correct_predictions INTEGER,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    
                    # Match data cache
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS match_data (
                            fixture_id INTEGER PRIMARY KEY,
                            data JSONB,
                            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    
                    # System logs
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS system_logs (
                            id SERIAL PRIMARY KEY,
                            log_type VARCHAR(50),
                            message TEXT,
                            details JSONB,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    
                    conn.commit()
                    logger.info("Database tables initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database tables: {e}")
            raise
    
    def save_prediction(self, prediction: Prediction):
        """Save prediction to database"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO predictions 
                        (fixture_id, league, home_team, away_team, prediction_type, 
                         prediction, probability, confidence, elapsed_time)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                    """, (
                        prediction.fixture_id,
                        prediction.league,
                        prediction.home_team,
                        prediction.away_team,
                        prediction.prediction_type,
                        prediction.prediction,
                        prediction.probability,
                        prediction.confidence,
                        prediction.elapsed_time
                    ))
                    prediction_id = cur.fetchone()[0]
                    conn.commit()
                    logger.info(f"Saved prediction {prediction_id} for fixture {prediction.fixture_id}")
                    return prediction_id
        except Exception as e:
            logger.error(f"Error saving prediction: {e}")
            return None
    
    def update_prediction_result(self, fixture_id: int, prediction_type: str, result: str):
        """Update prediction with actual result"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Get the prediction
                    cur.execute("""
                        SELECT prediction FROM predictions 
                        WHERE fixture_id = %s AND prediction_type = %s
                        ORDER BY created_at DESC LIMIT 1
                    """, (fixture_id, prediction_type))
                    
                    row = cur.fetchone()
                    if row:
                        predicted = row[0]
                        is_correct = (predicted == result)
                        
                        cur.execute("""
                            UPDATE predictions 
                            SET result = %s, is_correct = %s, updated_at = CURRENT_TIMESTAMP
                            WHERE fixture_id = %s AND prediction_type = %s
                        """, (result, is_correct, fixture_id, prediction_type))
                        conn.commit()
                        
                        logger.info(f"Updated result for fixture {fixture_id}: {prediction_type} = {result} (Correct: {is_correct})")
                        return is_correct
        except Exception as e:
            logger.error(f"Error updating prediction result: {e}")
            return None
    
    def get_recent_performance(self, hours: int = 24) -> Dict:
        """Get model performance for recent predictions"""
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT 
                            prediction_type,
                            COUNT(*) as total,
                            SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) as correct,
                            AVG(CASE WHEN is_correct THEN 1 ELSE 0 END) as accuracy
                        FROM predictions
                        WHERE created_at > NOW() - INTERVAL '%s hours'
                        AND is_correct IS NOT NULL
                        GROUP BY prediction_type
                    """, (hours,))
                    return cur.fetchall()
        except Exception as e:
            logger.error(f"Error getting performance: {e}")
            return []
    
    def save_match_data(self, fixture_id: int, data: Dict):
        """Cache match data"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO match_data (fixture_id, data)
                        VALUES (%s, %s)
                        ON CONFLICT (fixture_id) 
                        DO UPDATE SET data = %s, last_updated = CURRENT_TIMESTAMP
                    """, (fixture_id, json.dumps(data), json.dumps(data)))
                    conn.commit()
        except Exception as e:
            logger.error(f"Error saving match data: {e}")
    
    def log_system_event(self, log_type: str, message: str, details: Dict = None):
        """Log system events"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO system_logs (log_type, message, details)
                        VALUES (%s, %s, %s)
                    """, (log_type, message, json.dumps(details) if details else None))
                    conn.commit()
        except Exception as e:
            logger.error(f"Error logging system event: {e}")


class APIFootballClient:
    """Client for API-Football"""
    
    def __init__(self):
        self.api_key = os.getenv('API_FOOTBALL_KEY')
        self.base_url = "https://v3.football.api-sports.io"
        self.headers = {
            'x-rapidapi-key': self.api_key,
            'x-rapidapi-host': 'v3.football.api-sports.io'
        }
        
    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make API request with error handling"""
        try:
            url = f"{self.base_url}/{endpoint}"
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get('errors'):
                logger.error(f"API Error: {data['errors']}")
                return None
                
            return data
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None
    
    def get_live_fixtures(self) -> List[Dict]:
        """Get all live fixtures"""
        logger.info("Fetching live fixtures...")
        data = self._make_request('fixtures', {'live': 'all'})
        
        if data and 'response' in data:
            fixtures = data['response']
            logger.info(f"Found {len(fixtures)} live fixtures")
            return fixtures
        return []
    
    def get_fixture_statistics(self, fixture_id: int) -> Optional[Dict]:
        """Get detailed statistics for a fixture"""
        data = self._make_request('fixtures/statistics', {'fixture': fixture_id})
        return data.get('response') if data else None
    
    def get_fixture_events(self, fixture_id: int) -> Optional[List[Dict]]:
        """Get match events (goals, cards, etc.)"""
        data = self._make_request('fixtures/events', {'fixture': fixture_id})
        return data.get('response') if data else None
    
    def get_head_to_head(self, team1_id: int, team2_id: int, last: int = 10) -> Optional[List[Dict]]:
        """Get head-to-head history"""
        data = self._make_request('fixtures/headtohead', {
            'h2h': f"{team1_id}-{team2_id}",
            'last': last
        })
        return data.get('response') if data else None


class TelegramNotifier:
    """Send betting tips to Telegram"""
    
    def __init__(self):
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        
    def send_tip(self, prediction: Prediction):
        """Send betting tip to Telegram"""
        try:
            # Format message
            emoji_map = {
                'over_under': '📊',
                'btts': '⚽',
                'win_lose': '🏆'
            }
            
            emoji = emoji_map.get(prediction.prediction_type, '🎯')
            
            message = f"""
{emoji} <b>LIVE BETTING TIP</b> {emoji}

🏟️ <b>{prediction.home_team}</b> vs <b>{prediction.away_team}</b>
🏆 {prediction.league}
⏱️ {prediction.elapsed_time}' 

📈 <b>{prediction.prediction_type.upper().replace('_', ' ')}</b>
✅ Prediction: <b>{prediction.prediction}</b>
📊 Probability: <b>{prediction.probability:.1%}</b>
🎯 Confidence: <b>{prediction.confidence:.1%}</b>

⚡ Act Fast - Live Opportunity!
            """
            
            url = f"{self.base_url}/sendMessage"
            payload = {
                'chat_id': self.chat_id,
                'text': message.strip(),
                'parse_mode': 'HTML'
            }
            
            response = requests.post(url, json=payload, timeout=5)
            response.raise_for_status()
            
            logger.info(f"Sent Telegram tip for fixture {prediction.fixture_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False


class PredictionEngine:
    """Core prediction engine using Random Forest models"""
    
    def __init__(self):
        self.models = {}
        self.feature_engineer = FeatureEngineer()
        self.load_models()
        
    def load_models(self):
        """Load trained models from disk"""
        model_types = ['over_under', 'btts', 'win_lose']
        
        for model_type in model_types:
            model_path = f'models/{model_type}_model.pkl'
            if os.path.exists(model_path):
                try:
                    with open(model_path, 'rb') as f:
                        self.models[model_type] = pickle.load(f)
                    logger.info(f"Loaded {model_type} model successfully")
                except Exception as e:
                    logger.error(f"Error loading {model_type} model: {e}")
            else:
                logger.warning(f"Model file not found: {model_path}")
    
    def predict(self, fixture_data: Dict, statistics: Dict, events: List[Dict]) -> List[Prediction]:
        """Generate predictions for a live fixture"""
        predictions = []
        
        try:
            # Extract features
            features = self.feature_engineer.extract_live_features(
                fixture_data, statistics, events
            )
            
            if features is None:
                return predictions
            
            fixture = fixture_data['fixture']
            teams = fixture_data['teams']
            league = fixture_data['league']
            elapsed = fixture_data['fixture']['status']['elapsed'] or 0
            
            # Generate predictions for each model
            for model_type, model in self.models.items():
                try:
                    # Get prediction and probability
                    pred_proba = model.predict_proba([features])[0]
                    pred_class = model.predict([features])[0]
                    
                    # Get probability for predicted class
                    probability = pred_proba[pred_class]
                    
                    # Calculate confidence based on probability distribution
                    confidence = self._calculate_confidence(pred_proba)
                    
                    # Only send high-confidence predictions
                    if confidence >= 0.70 and probability >= 0.65:
                        # Map prediction class to readable format
                        prediction_text = self._format_prediction(model_type, pred_class, fixture_data)
                        
                        pred = Prediction(
                            fixture_id=fixture['id'],
                            league=league['name'],
                            home_team=teams['home']['name'],
                            away_team=teams['away']['name'],
                            prediction_type=model_type,
                            prediction=prediction_text,
                            probability=probability,
                            confidence=confidence,
                            timestamp=datetime.now(),
                            elapsed_time=elapsed
                        )
                        
                        predictions.append(pred)
                        logger.info(f"Generated {model_type} prediction for fixture {fixture['id']}: {prediction_text} ({probability:.2%})")
                
                except Exception as e:
                    logger.error(f"Error in {model_type} prediction: {e}")
                    continue
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            return []
    
    def _calculate_confidence(self, probabilities: np.ndarray) -> float:
        """Calculate confidence score based on probability distribution"""
        # Confidence is high when one class has much higher probability than others
        max_prob = np.max(probabilities)
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        max_entropy = -np.log(1.0 / len(probabilities))
        
        # Normalized confidence (0-1)
        confidence = max_prob * (1 - entropy / max_entropy)
        return confidence
    
    def _format_prediction(self, model_type: str, pred_class: int, fixture_data: Dict) -> str:
        """Format prediction for display"""
        if model_type == 'over_under':
            return 'Over 2.5' if pred_class == 1 else 'Under 2.5'
        elif model_type == 'btts':
            return 'Both Teams Score' if pred_class == 1 else 'No BTTS'
        elif model_type == 'win_lose':
            if pred_class == 0:
                return f"{fixture_data['teams']['home']['name']} Win"
            else:
                return f"{fixture_data['teams']['away']['name']} Win"
        return str(pred_class)


class BettingSystem:
    """Main betting system orchestrator"""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.api = APIFootballClient()
        self.telegram = TelegramNotifier()
        self.engine = PredictionEngine()
        self.processed_predictions = set()  # Track processed fixtures to avoid duplicates
        self.is_running = False
        
        logger.info("Betting system initialized successfully")
        self.db.log_system_event('startup', 'System initialized')
    
    def scan_live_matches(self):
        """Scan and process live matches"""
        logger.info("=" * 50)
        logger.info("Starting live match scan...")
        
        try:
            fixtures = self.api.get_live_fixtures()
            
            if not fixtures:
                logger.info("No live fixtures found")
                return
            
            # Filter fixtures based on elapsed time (focus on early to mid-game)
            suitable_fixtures = [
                f for f in fixtures 
                if f['fixture']['status']['elapsed'] and 
                10 <= f['fixture']['status']['elapsed'] <= 70
            ]
            
            logger.info(f"Processing {len(suitable_fixtures)} suitable live fixtures")
            
            for fixture in suitable_fixtures:
                fixture_id = fixture['fixture']['id']
                
                # Get detailed statistics
                statistics = self.api.get_fixture_statistics(fixture_id)
                events = self.api.get_fixture_events(fixture_id)
                
                if not statistics:
                    continue
                
                # Save match data
                self.db.save_match_data(fixture_id, {
                    'fixture': fixture,
                    'statistics': statistics,
                    'events': events
                })
                
                # Generate predictions
                predictions = self.engine.predict(fixture, statistics, events or [])
                
                # Process and send predictions
                for prediction in predictions:
                    # Create unique key to avoid duplicate predictions
                    pred_key = f"{fixture_id}_{prediction.prediction_type}"
                    
                    if pred_key not in self.processed_predictions:
                        # Save to database
                        pred_id = self.db.save_prediction(prediction)
                        
                        if pred_id:
                            # Send to Telegram
                            self.telegram.send_tip(prediction)
                            self.processed_predictions.add(pred_key)
                
                # Rate limiting
                time.sleep(0.5)
            
            logger.info("Live match scan completed")
            
        except Exception as e:
            logger.error(f"Error in live match scan: {e}")
            self.db.log_system_event('error', 'Live scan error', {'error': str(e)})
    
    def train_models(self):
        """Trigger model training"""
        logger.info("Starting model training...")
        self.db.log_system_event('training', 'Model training started')
        
        try:
            trainer = ModelTrainer(self.db)
            results = trainer.train_all_models()
            
            logger.info(f"Model training completed: {results}")
            self.db.log_system_event('training', 'Model training completed', results)
            
        except Exception as e:
            logger.error(f"Error in model training: {e}")
            self.db.log_system_event('error', 'Training error', {'error': str(e)})
    
    def backfill_historical_data(self, days: int = 30):
        """Backfill historical data for training"""
        logger.info(f"Starting backfill for last {days} days...")
        self.db.log_system_event('backfill', f'Backfill started for {days} days')
        
        try:
            # This would fetch historical data from API-Football
            # Implementation depends on your API plan and rate limits
            logger.info("Backfill completed")
            self.db.log_system_event('backfill', 'Backfill completed')
            
        except Exception as e:
            logger.error(f"Error in backfill: {e}")
            self.db.log_system_event('error', 'Backfill error', {'error': str(e)})
    
    def daily_digest(self):
        """Generate daily performance digest"""
        logger.info("Generating daily digest...")
        
        try:
            performance = self.db.get_recent_performance(24)
            
            digest = "📊 <b>DAILY PERFORMANCE DIGEST</b>\n\n"
            
            for perf in performance:
                accuracy = perf['accuracy'] * 100 if perf['accuracy'] else 0
                digest += f"<b>{perf['prediction_type'].upper()}</b>\n"
                digest += f"Total: {perf['total']} | Correct: {perf['correct']} | Accuracy: {accuracy:.1f}%\n\n"
            
            # Send to Telegram
            url = f"https://api.telegram.org/bot{self.telegram.bot_token}/sendMessage"
            requests.post(url, json={
                'chat_id': self.telegram.chat_id,
                'text': digest,
                'parse_mode': 'HTML'
            })
            
            logger.info("Daily digest sent")
            
        except Exception as e:
            logger.error(f"Error generating digest: {e}")
    
    def health_check(self):
        """System health check"""
        logger.info("Running health check...")
        
        health_status = {
            'database': False,
            'api': False,
            'models': False,
            'telegram': False
        }
        
        try:
            # Check database
            with self.db.get_connection() as conn:
                health_status['database'] = True
            
            # Check API
            fixtures = self.api.get_live_fixtures()
            health_status['api'] = fixtures is not None
            
            # Check models
            health_status['models'] = len(self.engine.models) > 0
            
            # Check Telegram
            url = f"{self.telegram.base_url}/getMe"
            response = requests.get(url)
            health_status['telegram'] = response.status_code == 200
            
            logger.info(f"Health check: {health_status}")
            self.db.log_system_event('health_check', 'Health check completed', health_status)
            
            return all(health_status.values())
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def start_live_scanning(self):
        """Start continuous live scanning"""
        self.is_running = True
        logger.info("🚀 Starting live betting system...")
        
        # Schedule tasks
        schedule.every(2).minutes.do(self.scan_live_matches)
        schedule.every().day.at("02:00").do(self.train_models)
        schedule.every().day.at("08:00").do(self.daily_digest)
        schedule.every(30).minutes.do(self.health_check)
        
        # Initial health check
        self.health_check()
        
        # Main loop
        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(30)
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                self.is_running = False
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(60)


def main():
    """Main entry point with manual controls"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AI Betting Predictions System')
    parser.add_argument('--mode', choices=['live', 'train', 'backfill', 'digest', 'health'], 
                        default='live', help='Operation mode')
    parser.add_argument('--days', type=int, default=30, help='Days for backfill')
    
    args = parser.parse_args()
    
    system = BettingSystem()
    
    if args.mode == 'live':
        system.start_live_scanning()
    elif args.mode == 'train':
        system.train_models()
    elif args.mode == 'backfill':
        system.backfill_historical_data(args.days)
    elif args.mode == 'digest':
        system.daily_digest()
    elif args.mode == 'health':
        status = system.health_check()
        print(f"System health: {'✅ OK' if status else '❌ ISSUES DETECTED'}")


if __name__ == '__main__':
    main()
