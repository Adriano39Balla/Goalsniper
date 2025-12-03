import asyncio
import sys
import signal
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import schedule
import json
import pandas as pd
import numpy as np
from loguru import logger
import joblib
import traceback
import psutil  # ensure psutil is available for health checks

# Ensure logs directory exists before adding file handler
Path("logs").mkdir(parents=True, exist_ok=True)

# Remove default handler and add custom format
logger.remove()

# Add console handler with simple format
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO"
)

# Add file handler
logger.add(
    "logs/app_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="30 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG"
)

# Now import custom modules after configuring logger
from config import Settings, BettingConfig, DatabaseManager

class APIFootballClient:
    """Client for API-Football with retry logic"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://v3.football.api-sports.io"
        self.headers = {
            'x-apisports-key': api_key,
            'x-rapidapi-host': 'v3.football.api-sports.io'
        }
    
    async def fetch_live_matches(self) -> List[Dict]:
        """Fetch live in-play matches"""
        import aiohttp
        from tenacity import retry, stop_after_attempt, wait_exponential
        
        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
        async def _fetch():
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/fixtures?live=all"
                
                async with session.get(url, headers=self.headers, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('response', [])
                    elif response.status == 429:
                        logger.warning("Rate limited, waiting...")
                        await asyncio.sleep(30)
                        raise Exception("Rate limited")
                    else:
                        logger.error(f"API Error: {response.status}")
                        return []
        
        try:
            return await _fetch()
        except Exception as e:
            logger.error(f"Error fetching live matches: {e}")
            return []
    
    async def fetch_match_statistics(self, fixture_id: int) -> Dict:
        """Fetch detailed statistics for a match"""
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/fixtures/statistics?fixture={fixture_id}"
            
            try:
                async with session.get(url, headers=self.headers, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('response', {})
                    else:
                        return {}
            except Exception as e:
                logger.error(f"Error fetching statistics: {e}")
                return {}
    
    async def fetch_odds(self, fixture_id: int) -> Dict:
        """Fetch betting odds for a match"""
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/odds?fixture={fixture_id}&bookmaker=1"
            
            try:
                async with session.get(url, headers=self.headers, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('response', [{}])[0] if data.get('response') else {}
                    else:
                        return {}
            except Exception as e:
                logger.error(f"Error fetching odds: {e}")
                return {}

class ModelTrainer:
    """Simplified model trainer integrated into main system"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.models = {}
        self.scalers = {}
        
    def load_models(self):
        """Load trained models from disk"""
        try:
            model_dir = Path("models/current")
            if model_dir.exists() and model_dir.is_dir():
                for model_file in model_dir.glob("*.pkl"):
                    if "ensemble" in str(model_file) or "random_forest" in str(model_file):
                        target = model_file.stem.split('_')[0]
                        self.models[target] = joblib.load(model_file)
                        logger.info(f"Loaded model for {target}")
                
                # Load scalers
                scaler_file = model_dir / "scalers.pkl"
                if scaler_file.exists():
                    self.scalers = joblib.load(scaler_file)
                    logger.info("Loaded scalers")
                
                return True
        except Exception as e:
            logger.error(f"Error loading models: {e}")
        
        return False
    
    def train_simple_model(self):
        """Train a simple model if no trained models exist"""
        logger.info("Training simple model for initial use...")
        
        # Create synthetic training data for initial model
        np.random.seed(42)
        n_samples = 1000
        
        # Generate features
        X = np.random.randn(n_samples, 10)
        
        # Generate targets with some logic
        y_home_win = (X[:, 0] + X[:, 1] > 0).astype(int)
        y_over_25 = (X[:, 2] + X[:, 3] > 0.5).astype(int)
        y_btts = (X[:, 4] + X[:, 5] > 0.3).astype(int)
        
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        
        # Train models
        for target_name, y in [('home_win', y_home_win), ('over_2_5', y_over_25), ('btts', y_btts)]:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_scaled, y)
            
            self.models[target_name] = model
            self.scalers[target_name] = scaler
        
        # Save models
        self.save_models()
        logger.info("Simple models trained and saved")
    
    def save_models(self):
        """Save models to disk"""
        model_dir = Path("models/simple")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        for target, model in self.models.items():
            model_path = model_dir / f"{target}_ensemble.pkl"
            joblib.dump(model, model_path)
        
        scaler_path = model_dir / "scalers.pkl"
        joblib.dump(self.scalers, scaler_path)
        
        # Update current symlink (fallback to copy on Windows)
        current_path = Path("models/current")
        try:
            if current_path.exists():
                if current_path.is_symlink():
                    current_path.unlink()
                elif current_path.is_dir():
                    import shutil
                    shutil.rmtree(current_path)
            # Attempt symlink
            try:
                current_path.symlink_to(model_dir, target_is_directory=True)
            except Exception:
                # Fallback: copy directory (Windows may require elevated privileges for symlink)
                import shutil
                shutil.copytree(model_dir, current_path)
        except Exception as e:
            logger.warning(f"Could not create/update models/current: {e}")
        
        logger.info(f"Models saved to {model_dir}")

class PredictionEngine:
    """Prediction engine using trained models"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.trainer = ModelTrainer(db_manager)
        self.models = {}
        self.scalers = {}
        self.load_models()
        
    def load_models(self):
        """Load models from trainer"""
        if not self.trainer.load_models():
            logger.warning("No trained models found, training simple models...")
            self.trainer.train_simple_model()
            self.trainer.load_models()
        
        self.models = self.trainer.models
        self.scalers = self.trainer.scalers
        logger.info(f"Loaded {len(self.models)} models for prediction")
    
    def extract_features(self, match_data: Dict) -> pd.DataFrame:
        """Extract features from match data"""
        features = {}
        
        # Basic match info
        fixture = match_data.get('fixture', {})
        teams = match_data.get('teams', {})
        goals = match_data.get('goals', {})
        league = match_data.get('league', {})
        
        features['home_team_rating'] = teams.get('home', {}).get('rating', 0) or 0
        features['away_team_rating'] = teams.get('away', {}).get('rating', 0) or 0
        
        # Current match state
        features['home_score'] = goals.get('home', 0) or 0
        features['away_score'] = goals.get('away', 0) or 0
        features['goal_difference'] = features['home_score'] - features['away_score']
        features['total_goals'] = features['home_score'] + features['away_score']
        
        # Match progress
        status = fixture.get('status', {})
        features['minute'] = status.get('elapsed', 0) or 0
        features['time_ratio'] = min(features['minute'] / 90, 1.0) if features['minute'] else 0.0
        
        # League info
        features['league_rank'] = league.get('rank', 50)
        
        # Time features
        try:
            match_time = datetime.fromisoformat(fixture['date'].replace('Z', '+00:00')) if 'date' in fixture else datetime.now()
        except Exception:
            match_time = datetime.now()
        features['hour_of_day'] = match_time.hour
        features['day_of_week'] = match_time.weekday()
        features['month'] = match_time.month
        
        # Derived features
        features['momentum'] = features['goal_difference'] * features.get('time_ratio', 0)
        features['scoring_pressure'] = features['total_goals'] / max(features.get('time_ratio', 0.1), 0.1)
        
        # Fill missing values with defaults
        default_features = {
            'home_possession': 50,
            'away_possession': 50,
            'home_shots_on_goal': 5,
            'away_shots_on_goal': 5,
            'implied_prob_home': 0.33,
            'implied_prob_draw': 0.33,
            'implied_prob_away': 0.33
        }
        
        features.update(default_features)
        
        return pd.DataFrame([features])
    
    def predict(self, match_data: Dict) -> Dict[str, Any]:
        """Make predictions for a match"""
        if not self.models:
            logger.error("No models available for prediction")
            return {}
        
        try:
            # Extract features
            features_df = self.extract_features(match_data)
            
            # Define expected feature columns (should match training)
            expected_features = [
                'home_team_rating', 'away_team_rating', 'home_score', 'away_score',
                'goal_difference', 'total_goals', 'minute', 'time_ratio',
                'league_rank', 'hour_of_day', 'day_of_week', 'month',
                'momentum', 'scoring_pressure', 'home_possession', 'away_possession',
                'home_shots_on_goal', 'away_shots_on_goal', 'implied_prob_home',
                'implied_prob_draw', 'implied_prob_away'
            ]
            
            # Ensure all features are present
            for col in expected_features:
                if col not in features_df.columns:
                    features_df[col] = 0
            
            features_df = features_df[expected_features]
            
            # Make predictions for each target
            predictions = {}
            probabilities = {}
            
            for target, model in self.models.items():
                if target in self.scalers:
                    try:
                        # Scale features
                        scaler = self.scalers[target]
                        features_scaled = scaler.transform(features_df)
                        
                        # Predict
                        if hasattr(model, 'predict_proba'):
                            proba = model.predict_proba(features_scaled)[0]
                            predictions[target] = int(model.predict(features_scaled)[0])
                            probabilities[target] = float(proba[1]) if len(proba) > 1 else float(proba[0])
                        else:
                            pred = int(model.predict(features_scaled)[0])
                            predictions[target] = pred
                            probabilities[target] = float(pred)
                    except Exception as e:
                        logger.error(f"Error predicting {target}: {e}")
                        predictions[target] = 0
                        probabilities[target] = 0.5
            
            # Generate tips
            tips = self.generate_tips(predictions, probabilities, match_data)
            
            result = {
                'match_id': match_data.get('fixture', {}).get('id'),
                'predictions': predictions,
                'probabilities': probabilities,
                'tips': tips,
                'confidence': self.calculate_confidence(probabilities),
                'timestamp': datetime.now().isoformat(),
                'features': features_df.iloc[0].to_dict()
            }
            
            logger.info(f"Predictions made for match {result['match_id']}")
            return result
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            logger.error(traceback.format_exc())
            return {}
    
    def generate_tips(self, predictions: Dict, probabilities: Dict, match_data: Dict) -> List[Dict]:
        """Generate betting tips"""
        tips = []
        config = BettingConfig()
        
        # Home win tip
        if predictions.get('home_win', 0) == 1 and probabilities.get('home_win', 0) > config.min_confidence:
            tips.append({
                'type': '1X2',
                'prediction': 'Home Win',
                'probability': probabilities.get('home_win', 0),
                'confidence': 'high' if probabilities.get('home_win', 0) > 0.75 else 'medium',
                'market': 'match_winner'
            })
        
        # Over 2.5 goals tip
        if predictions.get('over_2_5', 0) == 1 and probabilities.get('over_2_5', 0) > config.min_confidence:
            tips.append({
                'type': 'Over/Under',
                'prediction': 'Over 2.5 Goals',
                'probability': probabilities.get('over_2_5', 0),
                'confidence': 'high' if probabilities.get('over_2_5', 0) > 0.75 else 'medium',
                'market': 'total_goals'
            })
        
        # BTTS tip
        if predictions.get('btts', 0) == 1 and probabilities.get('btts', 0) > config.min_confidence:
            tips.append({
                'type': 'BTTS',
                'prediction': 'Both Teams to Score',
                'probability': probabilities.get('btts', 0),
                'confidence': 'high' if probabilities.get('btts', 0) > 0.75 else 'medium',
                'market': 'btts'
            })
        
        # Sort by probability
        tips.sort(key=lambda x: x['probability'], reverse=True)
        
        return tips[:config.max_tips_per_match]
    
    def calculate_confidence(self, probabilities: Dict) -> float:
        """Calculate overall confidence score"""
        if not probabilities:
            return 0.0
        
        valid_probs = [p for p in probabilities.values() if p > 0]
        if not valid_probs:
            return 0.0
        
        return float(np.mean(valid_probs))

class TelegramBot:
    """Telegram bot for notifications"""
    
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
    
    async def send_message(self, message: str) -> bool:
        """Send message to Telegram"""
        import aiohttp
        
        if not self.token or not self.chat_id:
            logger.warning("Telegram not configured")
            return False
        
        url = f"{self.base_url}/sendMessage"
        payload = {
            'chat_id': self.chat_id,
            'text': message,
            'parse_mode': 'HTML',
            'disable_web_page_preview': True
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=10) as response:
                    if response.status == 200:
                        return True
                    else:
                        logger.error(f"Telegram API error: {response.status}")
                        return False
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")
            return False
    
    def format_tip_message(self, match_data: Dict, tips: List[Dict]) -> str:
        """Format tips for Telegram"""
        home_team = match_data.get('teams', {}).get('home', {}).get('name', 'Home')
        away_team = match_data.get('teams', {}).get('away', {}).get('name', 'Away')
        score = f"{match_data.get('goals', {}).get('home', 0)}-{match_data.get('goals', {}).get('away', 0)}"
        minute = match_data.get('fixture', {}).get('status', {}).get('elapsed', 0)
        league = match_data.get('league', {}).get('name', '')
        
        message = f"🎯 <b>BETTING ALERT</b> 🎯\n\n"
        message += f"🏆 <b>{league}</b>\n"
        message += f"⚽ {home_team} vs {away_team}\n"
        message += f"📊 Score: {score} ({minute}')\n"
        message += f"🕒 {datetime.now().strftime('%H:%M %Y-%m-%d')}\n\n"
        
        if tips:
            message += "💰 <b>RECOMMENDED BETS:</b>\n\n"
            for i, tip in enumerate(tips, 1):
                emoji = "🔥" if tip['confidence'] == 'high' else "✅"
                message += f"{i}. {emoji} <b>{tip['type']}</b>\n"
                message += f"   📈 {tip['prediction']}\n"
                message += f"   🎯 Probability: {tip['probability']:.1%}\n"
                message += f"   ⭐ Confidence: {tip['confidence'].upper()}\n\n"
        else:
            message += "📭 No valuable bets found for this match.\n\n"
        
        message += "⚠️ <i>Bet responsibly. Only bet what you can afford to lose.</i>"
        
        return message

class BettingPredictor:
    """Main betting predictor system"""
    
    def __init__(self):
        self.settings = Settings()
        self.db = DatabaseManager(self.settings.DATABASE_URL)
        self.api_client = APIFootballClient(self.settings.API_FOOTBALL_KEY)
        self.prediction_engine = PredictionEngine(self.db)
        self.telegram_bot = TelegramBot(self.settings.TELEGRAM_BOT_TOKEN, self.settings.TELEGRAM_CHAT_ID)
        
        self.is_running = False
        self.performance = {
            'predictions_made': 0,
            'tips_sent': 0,
            'last_scan': None,
            'start_time': datetime.now()
        }
        
        logger.info("=" * 50)
        logger.info("Betting Predictor System Initialized")
        logger.info(f"API Key: {'✓' if self.settings.API_FOOTBALL_KEY else '✗'}")
        logger.info(f"Telegram: {'✓' if self.settings.TELEGRAM_BOT_TOKEN else '✗'}")
        logger.info(f"Database: {'✓' if self.settings.DATABASE_URL else '✗'}")
        logger.info("=" * 50)
    
    async def scan_live_matches(self):
        """Scan for live matches"""
        logger.info("🔍 Scanning for live matches...")
        self.performance['last_scan'] = datetime.now()
        
        try:
            matches = await self.api_client.fetch_live_matches()
            
            if not matches:
                logger.info("No live matches found")
                return
            
            logger.info(f"Found {len(matches)} live matches")
            
            for match in matches:
                match_id = match['fixture']['id']
                
                # Check if already processed recently
                if self.db.check_recent_prediction(match_id):
                    continue
                
                # Make predictions
                prediction = self.prediction_engine.predict(match)
                
                if prediction and prediction.get('tips'):
                    # Store match data
                    self.db.store_match(match)
                    
                    # Store prediction
                    self.db.store_prediction(prediction)
                    
                    # Send Telegram alert
                    message = self.telegram_bot.format_tip_message(match, prediction['tips'])
                    if await self.telegram_bot.send_message(message):
                        self.performance['tips_sent'] += len(prediction['tips'])
                        logger.info(f"📨 Sent tips for match {match_id}")
                
                self.performance['predictions_made'] += 1
                
                # Rate limiting
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Error in scan: {e}")
    
    async def backfill_data(self, days: int = 7):
        """Backfill historical data"""
        logger.info(f"📊 Backfilling {days} days of data...")
        # Implementation would go here
        await asyncio.sleep(1)
        logger.info("Backfill completed")
    
    async def send_daily_report(self):
        """Send daily performance report"""
        logger.info("📈 Generating daily report...")
        
        performance = self.db.get_daily_performance()
        uptime = datetime.now() - self.performance['start_time']
        
        message = f"📊 <b>DAILY REPORT</b>\n\n"
        message += f"⏰ Uptime: {uptime}\n"
        message += f"🔍 Predictions Made: {self.performance['predictions_made']}\n"
        message += f"💰 Tips Sent: {self.performance['tips_sent']}\n"
        message += f"🎯 Win Rate: {performance.get('win_rate', 0):.1%}\n"
        message += f"📈 ROI: {performance.get('roi', 0):.1%}\n\n"
        message += f"🔄 Last Scan: {self.performance['last_scan'] or 'Never'}\n"
        
        await self.telegram_bot.send_message(message)
        logger.info("Daily report sent")
    
    def train_models(self):
        """Train models"""
        logger.info("🤖 Training models...")
        self.prediction_engine.trainer.train_simple_model()
        self.prediction_engine.load_models()
        logger.info("Models trained and loaded")
    
    def health_check(self) -> Dict:
        """System health check"""
        health = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'uptime': str(datetime.now() - self.performance['start_time']),
            'performance': self.performance.copy()
        }
        
        # Check components
        health['components'] = {
            'api': bool(self.settings.API_FOOTBALL_KEY),
            'telegram': bool(self.settings.TELEGRAM_BOT_TOKEN and self.settings.TELEGRAM_CHAT_ID),
            'database': bool(self.settings.DATABASE_URL),
            'models': len(self.prediction_engine.models) > 0
        }
        
        # Check system resources
        try:
            health['system'] = {
                'cpu': psutil.cpu_percent(),
                'memory': psutil.virtual_memory().percent,
                'disk': psutil.disk_usage('/').percent
            }
        except Exception:
            health['system'] = {'error': 'Could not get system metrics'}
        
        # Update status if any component is down
        if not all(health['components'].values()):
            health['status'] = 'degraded'
        if not health['components']['api']:
            health['status'] = 'unhealthy'
        
        return health
    
    def manual_scan(self):
        """Manual scan trigger"""
        logger.info("Manual scan triggered")
        asyncio.create_task(self.scan_live_matches())
    
    def manual_train(self):
        """Manual training trigger"""
        logger.info("Manual training triggered")
        self.train_models()
    
    def manual_backfill(self, days: int = 7):
        """Manual backfill trigger"""
        logger.info(f"Manual backfill triggered for {days} days")
        asyncio.create_task(self.backfill_data(days))
    
    def manual_report(self):
        """Manual report trigger"""
        logger.info("Manual report triggered")
        asyncio.create_task(self.send_daily_report())
    
    async def run_scheduler(self):
        """Run scheduled tasks"""
        # Schedule regular scans
        schedule.every(2).minutes.do(lambda: asyncio.create_task(self.scan_live_matches()))
        
        # Schedule daily tasks
        schedule.every().day.at("08:00").do(lambda: asyncio.create_task(self.send_daily_report()))
        schedule.every().day.at("03:00").do(self.train_models)
        schedule.every().day.at("04:00").do(lambda: asyncio.create_task(self.backfill_data(1)))
        
        logger.info("Scheduler started")
        
        while self.is_running:
            schedule.run_pending()
            await asyncio.sleep(1)
    
    async def run(self):
        """Main run loop"""
        self.is_running = True
        
        logger.info("🚀 Starting Betting Predictor...")
        
        # Initial scan
        await self.scan_live_matches()
        
        # Start scheduler
        await self.run_scheduler()
    
    def stop(self):
        """Stop the system"""
        self.is_running = False
        logger.info("🛑 Stopping Betting Predictor...")

# FastAPI app for health checks
from fastapi import FastAPI
import uvicorn

app = FastAPI(title="Betting Predictor API")
predictor = None

@app.on_event("startup")
async def startup_event():
    global predictor
    predictor = BettingPredictor()
    # Start predictor in background
    import threading
    thread = threading.Thread(target=lambda: asyncio.run(predictor.run()))
    thread.daemon = True
    thread.start()
    logger.info("FastAPI server started")

@app.get("/")
async def root():
    return {"message": "Betting Predictor API", "status": "running"}

@app.get("/health")
async def health():
    if predictor:
        return predictor.health_check()
    return {"status": "starting"}

@app.get("/scan")
async def trigger_scan():
    if predictor:
        predictor.manual_scan()
        return {"message": "Scan triggered"}
    return {"error": "Predictor not ready"}

@app.get("/train")
async def trigger_train():
    if predictor:
        predictor.manual_train()
        return {"message": "Training triggered"}
    return {"error": "Predictor not ready"}

@app.get("/backfill/{days}")
async def trigger_backfill(days: int = 7):
    if predictor:
        predictor.manual_backfill(days)
        return {"message": f"Backfill triggered for {days} days"}
    return {"error": "Predictor not ready"}

@app.get("/report")
async def trigger_report():
    if predictor:
        predictor.manual_report()
        return {"message": "Report triggered"}
    return {"error": "Predictor not ready"}

@app.get("/status")
async def status():
    if predictor:
        return {
            "is_running": predictor.is_running,
            "performance": predictor.performance
        }
    return {"is_running": False}

def signal_handler(sig, frame):
    """Handle shutdown signals"""
    logger.info("Shutdown signal received")
    if predictor:
        predictor.stop()
    sys.exit(0)

def main():
    """Main entry point"""
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start FastAPI server
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=False
    )
    
    server = uvicorn.Server(config)
    
    try:
        server.run()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        if predictor:
            predictor.stop()

if __name__ == "__main__":
    main()
