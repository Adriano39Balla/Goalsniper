import asyncio
import logging
import sys
import signal
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import schedule
import json
import pandas as pd
import numpy as np
from loguru import logger
import joblib
import traceback

# Custom modules
from database import DatabaseManager
from train_models import ModelTrainer
from config import Settings, BettingConfig

# Configure loguru
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:icon}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add(
    "logs/app_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="30 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function} - {message}",
    level="DEBUG"
)

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
        
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/fixtures?live=all"
            
            try:
                async with session.get(url, headers=self.headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('response', [])
                    else:
                        logger.error(f"API Error: {response.status}")
                        return []
            except Exception as e:
                logger.error(f"Error fetching live matches: {e}")
                return []
    
    async def fetch_match_statistics(self, fixture_id: int) -> Dict:
        """Fetch detailed statistics for a match"""
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/fixtures/statistics?fixture={fixture_id}"
            
            try:
                async with session.get(url, headers=self.headers) as response:
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
            url = f"{self.base_url}/odds?fixture={fixture_id}"
            
            try:
                async with session.get(url, headers=self.headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('response', [{}])[0] if data.get('response') else {}
                    else:
                        return {}
            except Exception as e:
                logger.error(f"Error fetching odds: {e}")
                return {}

class PredictionEngine:
    """Advanced prediction engine using trained models"""
    
    def __init__(self):
        self.settings = Settings()
        self.db = DatabaseManager()
        self.model_trainer = ModelTrainer(self.db)
        self.models = {}
        self.scalers = {}
        self.load_models()
        
    def load_models(self):
        """Load the latest trained models"""
        if not self.model_trainer.load_latest_models():
            logger.warning("No trained models found. Please run training first.")
            self.models = {}
            self.scalers = {}
        else:
            self.models = self.model_trainer.models
            self.scalers = self.model_trainer.scalers
    
    def extract_features(self, match_data: Dict, stats: Dict, odds: Dict) -> pd.DataFrame:
        """Extract features from match data for prediction"""
        features = {}
        
        # Basic match info
        features['home_team_rating'] = match_data.get('teams', {}).get('home', {}).get('rating', 0)
        features['away_team_rating'] = match_data.get('teams', {}).get('away', {}).get('rating', 0)
        
        # Current score state
        features['home_score'] = match_data.get('goals', {}).get('home', 0)
        features['away_score'] = match_data.get('goals', {}).get('away', 0)
        features['goal_difference'] = features['home_score'] - features['away_score']
        features['total_goals'] = features['home_score'] + features['away_score']
        
        # Match progress
        features['minute'] = match_data.get('fixture', {}).get('status', {}).get('elapsed', 0)
        features['time_ratio'] = min(features['minute'] / 90, 1.0)
        
        # Statistics
        if stats:
            home_stats = stats.get('statistics', [{}])[0] if stats.get('statistics') else {}
            away_stats = stats.get('statistics', [{}])[1] if stats.get('statistics') and len(stats.get('statistics', [])) > 1 else {}
            
            features['home_possession'] = home_stats.get('possession', 50)
            features['away_possession'] = away_stats.get('possession', 50)
            features['home_shots_on_goal'] = home_stats.get('shots on goal', 0)
            features['away_shots_on_goal'] = away_stats.get('shots on goal', 0)
            features['home_shots_off_goal'] = home_stats.get('shots off goal', 0)
            features['away_shots_off_goal'] = away_stats.get('shots off goal', 0)
            features['home_total_shots'] = home_stats.get('total shots', 0)
            features['away_total_shots'] = away_stats.get('total shots', 0)
        
        # Odds
        if odds and 'bookmakers' in odds:
            bookmaker = odds['bookmakers'][0] if odds['bookmakers'] else {}
            if 'bets' in bookmaker:
                for bet in bookmaker['bets']:
                    if bet['name'] == 'Match Winner':
                        for value in bet['values']:
                            if value['value'] == 'Home':
                                features['home_odds'] = float(value['odd'])
                            elif value['value'] == 'Draw':
                                features['draw_odds'] = float(value['odd'])
                            elif value['value'] == 'Away':
                                features['away_odds'] = float(value['odd'])
        
        # Implied probabilities
        if 'home_odds' in features:
            features['implied_prob_home'] = 1 / features['home_odds']
            features['implied_prob_draw'] = 1 / features['draw_odds']
            features['implied_prob_away'] = 1 / features['away_odds']
        
        # Derived features
        features['possession_difference'] = features.get('home_possession', 50) - features.get('away_possession', 50)
        features['shot_difference'] = features.get('home_total_shots', 0) - features.get('away_total_shots', 0)
        features['expected_goals_momentum'] = features.get('home_shots_on_goal', 0) * 0.3 + features.get('away_shots_on_goal', 0) * 0.3
        
        # Time features
        match_time = datetime.fromisoformat(match_data['fixture']['date'].replace('Z', '+00:00'))
        features['hour_of_day'] = match_time.hour
        features['day_of_week'] = match_time.weekday()
        features['month'] = match_time.month
        
        return pd.DataFrame([features])
    
    def predict_match(self, match_data: Dict) -> Dict[str, Any]:
        """Make predictions for a live match"""
        if not self.models:
            logger.warning("Models not loaded. Cannot make predictions.")
            return {}
        
        try:
            # Fetch additional data
            fixture_id = match_data['fixture']['id']
            
            # In production, you'd fetch these async
            stats = {}
            odds = {}
            
            # Extract features
            features_df = self.extract_features(match_data, stats, odds)
            
            # Prepare feature list (should match training)
            feature_columns = [
                'home_team_rating', 'away_team_rating', 'home_score', 'away_score',
                'goal_difference', 'total_goals', 'minute', 'time_ratio',
                'possession_difference', 'shot_difference', 'expected_goals_momentum',
                'hour_of_day', 'day_of_week', 'month'
            ]
            
            # Add odds features if available
            if 'implied_prob_home' in features_df.columns:
                feature_columns.extend(['implied_prob_home', 'implied_prob_draw', 'implied_prob_away'])
            
            # Ensure all features are present
            for col in feature_columns:
                if col not in features_df.columns:
                    features_df[col] = 0
            
            features_df = features_df[feature_columns]
            
            # Scale features and make predictions
            predictions = {}
            probabilities = {}
            
            for target, model in self.models.items():
                if target in self.scalers:
                    # Scale features
                    scaler = self.scalers[target]
                    features_scaled = scaler.transform(features_df)
                    
                    # Predict
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(features_scaled)[0]
                        predictions[target] = model.predict(features_scaled)[0]
                        probabilities[target] = proba[1] if len(proba) > 1 else proba[0]
                    else:
                        predictions[target] = model.predict(features_scaled)[0]
                        probabilities[target] = predictions[target]
            
            # Calculate expected value
            ev_calculations = self.calculate_expected_value(predictions, probabilities, odds)
            
            # Generate betting tips
            tips = self.generate_betting_tips(predictions, probabilities, ev_calculations, match_data)
            
            result = {
                'match_id': fixture_id,
                'predictions': predictions,
                'probabilities': probabilities,
                'expected_value': ev_calculations,
                'tips': tips,
                'confidence': self.calculate_confidence(probabilities),
                'timestamp': datetime.now().isoformat(),
                'features': features_df.iloc[0].to_dict()
            }
            
            logger.info(f"Predictions made for match {fixture_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error predicting match: {e}")
            logger.error(traceback.format_exc())
            return {}
    
    def calculate_expected_value(self, predictions: Dict, probabilities: Dict, odds: Dict) -> Dict:
        """Calculate expected value for each betting market"""
        ev_results = {}
        
        # Home win EV
        if 'home_odds' in odds and 'home_win' in probabilities:
            home_odds = odds.get('home_odds', 2.0)
            home_prob = probabilities.get('home_win', 0.5)
            ev_results['home_win_ev'] = (home_odds - 1) * home_prob - (1 - home_prob)
        
        # Over 2.5 EV
        if 'over_2_5_odds' in odds and 'over_2_5' in probabilities:
            over_odds = odds.get('over_2_5_odds', 2.0)
            over_prob = probabilities.get('over_2_5', 0.5)
            ev_results['over_2_5_ev'] = (over_odds - 1) * over_prob - (1 - over_prob)
        
        # BTTS EV
        if 'btts_yes_odds' in odds and 'btts' in probabilities:
            btts_odds = odds.get('btts_yes_odds', 2.0)
            btts_prob = probabilities.get('btts', 0.5)
            ev_results['btts_ev'] = (btts_odds - 1) * btts_prob - (1 - btts_prob)
        
        return ev_results
    
    def generate_betting_tips(self, predictions: Dict, probabilities: Dict, 
                            ev_calculations: Dict, match_data: Dict) -> List[Dict]:
        """Generate betting tips based on predictions and EV"""
        tips = []
        config = BettingConfig()
        
        # Check home win tip
        if (predictions.get('home_win', 0) == 1 and 
            probabilities.get('home_win', 0) > config.min_confidence and
            ev_calculations.get('home_win_ev', -1) > config.min_ev):
            
            tips.append({
                'type': '1X2',
                'prediction': 'Home Win',
                'probability': probabilities.get('home_win'),
                'ev': ev_calculations.get('home_win_ev'),
                'confidence': 'high' if probabilities.get('home_win', 0) > 0.7 else 'medium'
            })
        
        # Check away win tip
        if (predictions.get('away_win', 0) == 1 and 
            probabilities.get('away_win', 0) > config.min_confidence and
            ev_calculations.get('away_win_ev', -1) > config.min_ev):
            
            tips.append({
                'type': '1X2',
                'prediction': 'Away Win',
                'probability': probabilities.get('away_win'),
                'ev': ev_calculations.get('away_win_ev'),
                'confidence': 'high' if probabilities.get('away_win', 0) > 0.7 else 'medium'
            })
        
        # Check over 2.5 tip
        if (predictions.get('over_2_5', 0) == 1 and 
            probabilities.get('over_2_5', 0) > config.min_confidence and
            ev_calculations.get('over_2_5_ev', -1) > config.min_ev):
            
            tips.append({
                'type': 'Over/Under',
                'prediction': 'Over 2.5 Goals',
                'probability': probabilities.get('over_2_5'),
                'ev': ev_calculations.get('over_2_5_ev'),
                'confidence': 'high' if probabilities.get('over_2_5', 0) > 0.7 else 'medium'
            })
        
        # Check BTTS tip
        if (predictions.get('btts', 0) == 1 and 
            probabilities.get('btts', 0) > config.min_confidence and
            ev_calculations.get('btts_ev', -1) > config.min_ev):
            
            tips.append({
                'type': 'BTTS',
                'prediction': 'Both Teams to Score - Yes',
                'probability': probabilities.get('btts'),
                'ev': ev_calculations.get('btts_ev'),
                'confidence': 'high' if probabilities.get('btts', 0) > 0.7 else 'medium'
            })
        
        # Sort tips by EV (descending)
        tips.sort(key=lambda x: x.get('ev', 0), reverse=True)
        
        return tips
    
    def calculate_confidence(self, probabilities: Dict) -> float:
        """Calculate overall confidence score"""
        if not probabilities:
            return 0.0
        
        # Weighted average of probabilities
        weights = {
            'home_win': 1.0,
            'away_win': 1.0,
            'over_2_5': 0.8,
            'btts': 0.8
        }
        
        total_weight = 0
        weighted_sum = 0
        
        for target, prob in probabilities.items():
            if target in weights:
                weighted_sum += prob * weights[target]
                total_weight += weights[target]
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0

class TelegramBot:
    """Telegram bot for sending alerts"""
    
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
    
    async def send_message(self, message: str) -> bool:
        """Send message to Telegram"""
        import aiohttp
        
        if not self.token or not self.chat_id:
            logger.warning("Telegram bot not configured")
            return False
        
        url = f"{self.base_url}/sendMessage"
        payload = {
            'chat_id': self.chat_id,
            'text': message,
            'parse_mode': 'HTML'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        logger.info("Telegram message sent successfully")
                        return True
                    else:
                        logger.error(f"Failed to send Telegram message: {response.status}")
                        return False
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")
            return False
    
    def format_tip_message(self, match_data: Dict, tips: List[Dict]) -> str:
        """Format betting tips for Telegram"""
        home_team = match_data.get('teams', {}).get('home', {}).get('name', 'Home')
        away_team = match_data.get('teams', {}).get('away', {}).get('name', 'Away')
        score = f"{match_data.get('goals', {}).get('home', 0)}-{match_data.get('goals', {}).get('away', 0)}"
        minute = match_data.get('fixture', {}).get('status', {}).get('elapsed', 0)
        
        message = f"⚽ <b>LIVE BETTING ALERT</b> ⚽\n\n"
        message += f"🏟️ <b>{home_team} vs {away_team}</b>\n"
        message += f"📊 Score: {score} ({minute}')\n"
        message += f"⏰ Time: {datetime.now().strftime('%H:%M')}\n\n"
        message += "🎯 <b>RECOMMENDED BETS:</b>\n\n"
        
        for i, tip in enumerate(tips[:3], 1):  # Top 3 tips only
            emoji = "🔥" if tip.get('confidence') == 'high' else "✅"
            message += f"{i}. {emoji} <b>{tip['type']}</b>\n"
            message += f"   📈 Prediction: {tip['prediction']}\n"
            message += f"   🎲 Probability: {tip['probability']:.1%}\n"
            message += f"   💰 Expected Value: {tip['ev']:.3f}\n"
            message += f"   ⭐ Confidence: {tip['confidence'].upper()}\n\n"
        
        message += "⚠️ <i>Bet responsibly. Past performance is not indicative of future results.</i>"
        
        return message

class BettingPredictor:
    """Main betting predictor system"""
    
    def __init__(self):
        self.settings = Settings()
        self.db = DatabaseManager()
        self.api_client = APIFootballClient(self.settings.API_FOOTBALL_KEY)
        self.prediction_engine = PredictionEngine()
        self.telegram_bot = TelegramBot(self.settings.TELEGRAM_BOT_TOKEN, self.settings.TELEGRAM_CHAT_ID)
        self.is_running = False
        
        # Performance tracking
        self.performance_metrics = {
            'predictions_made': 0,
            'tips_sent': 0,
            'accuracy_tracking': [],
            'last_training': None
        }
        
        logger.info("Betting Predictor initialized")
    
    async def scan_live_matches(self):
        """Scan for live matches and make predictions"""
        logger.info("Scanning for live matches...")
        
        try:
            # Fetch live matches
            live_matches = await self.api_client.fetch_live_matches()
            
            if not live_matches:
                logger.info("No live matches found")
                return
            
            logger.info(f"Found {len(live_matches)} live matches")
            
            # Process each match
            for match in live_matches:
                fixture_id = match['fixture']['id']
                
                # Check if we already processed this match recently
                if self.db.check_recent_prediction(fixture_id):
                    continue
                
                # Make predictions
                predictions = self.prediction_engine.predict_match(match)
                
                if predictions and predictions.get('tips'):
                    # Send Telegram alert
                    message = self.telegram_bot.format_tip_message(match, predictions['tips'])
                    await self.telegram_bot.send_message(message)
                    
                    # Store prediction
                    self.db.store_prediction(predictions)
                    
                    # Update metrics
                    self.performance_metrics['tips_sent'] += len(predictions['tips'])
                    logger.info(f"Tips sent for match {fixture_id}")
                
                self.performance_metrics['predictions_made'] += 1
                
                # Avoid API rate limiting
                await asyncio.sleep(1)
        
        except Exception as e:
            logger.error(f"Error scanning live matches: {e}")
    
    async def backfill_historical_data(self, days: int = 30):
        """Backfill historical data for training"""
        logger.info(f"Backfilling historical data for {days} days")
        
        # This would implement API calls to fetch historical data
        # For now, just log
        logger.info("Backfill completed")
    
    async def daily_digest(self):
        """Send daily performance digest"""
        logger.info("Preparing daily digest")
        
        # Calculate daily performance
        daily_stats = self.db.get_daily_performance()
        
        message = f"📊 <b>DAILY PERFORMANCE DIGEST</b>\n\n"
        message += f"📈 Predictions Made: {self.performance_metrics['predictions_made']}\n"
        message += f"🎯 Tips Sent: {self.performance_metrics['tips_sent']}\n"
        message += f"💰 Estimated ROI: {daily_stats.get('roi', 0):.2%}\n"
        message += f"✅ Win Rate: {daily_stats.get('win_rate', 0):.2%}\n\n"
        message += f"🔄 Last Training: {self.performance_metrics['last_training'] or 'Never'}\n"
        
        await self.telegram_bot.send_message(message)
    
    def train_models(self):
        """Trigger model training"""
        logger.info("Starting model training...")
        
        try:
            trainer = ModelTrainer(self.db)
            result = trainer.train_all_models(model_type='ensemble', days_back=180)
            
            if result['success']:
                self.performance_metrics['last_training'] = datetime.now().isoformat()
                self.prediction_engine.load_models()  # Reload new models
                
                # Send training summary
                asyncio.create_task(self.send_training_summary(result))
                
                logger.info("Model training completed successfully")
            else:
                logger.error(f"Model training failed: {result.get('error')}")
        
        except Exception as e:
            logger.error(f"Error in model training: {e}")
    
    async def send_training_summary(self, result: Dict):
        """Send training summary to Telegram"""
        message = f"🤖 <b>MODEL TRAINING COMPLETE</b>\n\n"
        message += f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        message += f"📊 Samples: {result.get('training_samples', 0)}\n"
        message += f"🎯 Avg Accuracy: {result.get('avg_accuracy', 0):.4f}\n"
        message += f"📈 Avg ROC AUC: {result.get('avg_roc_auc', 0):.4f}\n"
        message += f"📁 Model Path: {result.get('model_dir', 'N/A')}\n"
        
        await self.telegram_bot.send_message(message)
    
    def health_check(self) -> Dict:
        """Perform system health check"""
        health = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'components': {},
            'metrics': self.performance_metrics.copy()
        }
        
        # Check database
        try:
            self.db.execute_query("SELECT 1")
            health['components']['database'] = 'healthy'
        except Exception as e:
            health['components']['database'] = f'unhealthy: {str(e)}'
            health['status'] = 'degraded'
        
        # Check models
        if self.prediction_engine.models:
            health['components']['models'] = f'loaded ({len(self.prediction_engine.models)} models)'
        else:
            health['components']['models'] = 'not loaded'
            health['status'] = 'degraded'
        
        # Check API key
        if self.settings.API_FOOTBALL_KEY:
            health['components']['api_key'] = 'configured'
        else:
            health['components']['api_key'] = 'missing'
            health['status'] = 'unhealthy'
        
        # Check Telegram
        if self.settings.TELEGRAM_BOT_TOKEN and self.settings.TELEGRAM_CHAT_ID:
            health['components']['telegram'] = 'configured'
        else:
            health['components']['telegram'] = 'not configured'
        
        # System metrics
        import psutil
        health['system'] = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent
        }
        
        return health
    
    async def auto_tune(self):
        """Auto-tune system parameters"""
        logger.info("Auto-tuning system parameters...")
        
        # This would implement automatic parameter optimization
        # For now, just trigger retraining if performance drops
        
        recent_performance = self.db.get_recent_accuracy()
        if recent_performance < 0.55:  # If accuracy drops below 55%
            logger.warning(f"Performance dropped to {recent_performance:.2%}, triggering retraining")
            self.train_models()
    
    def run_manual_scan(self):
        """Manual scan trigger"""
        logger.info("Manual scan triggered")
        asyncio.create_task(self.scan_live_matches())
    
    def run_manual_training(self):
        """Manual training trigger"""
        logger.info("Manual training triggered")
        self.train_models()
    
    def run_backfill(self, days: int = 30):
        """Manual backfill trigger"""
        logger.info(f"Manual backfill triggered for {days} days")
        asyncio.create_task(self.backfill_historical_data(days))
    
    def send_daily_digest(self):
        """Manual daily digest trigger"""
        logger.info("Manual daily digest triggered")
        asyncio.create_task(self.daily_digest())
    
    def run_auto_tune(self):
        """Manual auto-tune trigger"""
        logger.info("Manual auto-tune triggered")
        asyncio.create_task(self.auto_tune())
    
    async def run(self):
        """Main run loop"""
        self.is_running = True
        
        logger.info("Starting Betting Predictor...")
        
        # Schedule tasks
        schedule.every(5).minutes.do(lambda: asyncio.create_task(self.scan_live_matches()))
        schedule.every().day.at("02:00").do(self.train_models)
        schedule.every().day.at("08:00").do(lambda: asyncio.create_task(self.daily_digest()))
        schedule.every().hour.do(lambda: asyncio.create_task(self.auto_tune()))
        schedule.every().day.at("04:00").do(lambda: asyncio.create_task(self.backfill_historical_data(1)))
        
        # Initial tasks
        asyncio.create_task(self.scan_live_matches())
        
        # Keep running
        while self.is_running:
            schedule.run_pending()
            await asyncio.sleep(1)
    
    def stop(self):
        """Stop the system"""
        self.is_running = False
        logger.info("Betting Predictor stopped")

# FastAPI Health Endpoint (optional)
from fastapi import FastAPI
app = FastAPI()

predictor = None

@app.on_event("startup")
async def startup():
    global predictor
    predictor = BettingPredictor()
    asyncio.create_task(predictor.run())

@app.get("/health")
async def health():
    if predictor:
        return predictor.health_check()
    return {"status": "starting"}

@app.get("/scan")
async def manual_scan():
    if predictor:
        predictor.run_manual_scan()
        return {"status": "scan_triggered"}
    return {"status": "predictor_not_ready"}

@app.get("/train")
async def manual_train():
    if predictor:
        predictor.run_manual_training()
        return {"status": "training_triggered"}
    return {"status": "predictor_not_ready"}

@app.get("/backfill/{days}")
async def manual_backfill(days: int = 30):
    if predictor:
        predictor.run_backfill(days)
        return {"status": f"backfill_triggered_{days}_days"}
    return {"status": "predictor_not_ready"}

def main():
    """Main entry point"""
    import uvicorn
    
    # Start FastAPI server for health checks
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    
    # Also run the predictor directly
    predictor = BettingPredictor()
    
    # Signal handling
    def signal_handler(signum, frame):
        logger.info("Shutdown signal received")
        predictor.stop()
        exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run both
    try:
        # Run FastAPI in background
        import threading
        server_thread = threading.Thread(target=server.run)
        server_thread.start()
        
        # Run predictor
        asyncio.run(predictor.run())
        
    except KeyboardInterrupt:
        predictor.stop()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        predictor.stop()

if __name__ == "__main__":
    main()
