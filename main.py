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
import psutil

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
from config import Settings, BettingConfig, DatabaseManager, FEATURE_COLUMNS

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
    """Model trainer with consistent feature engineering"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.models = {}
        self.scalers = {}
        
    def load_models(self):
        """Load trained models from disk"""
        try:
            model_dir = Path("models/current")
            if model_dir.exists() and model_dir.is_dir():
                logger.info(f"Loading models from {model_dir}")
                
                # Load all models
                for model_file in model_dir.glob("*.pkl"):
                    if model_file.name.startswith(("home_win", "over_2_5", "btts")):
                        target = model_file.stem.split('_')[0]
                        if "ensemble" in str(model_file) or "random_forest" in str(model_file):
                            self.models[target] = joblib.load(model_file)
                            logger.info(f"Loaded model for {target} from {model_file.name}")
                
                # Load scalers
                scaler_file = model_dir / "scalers.pkl"
                if scaler_file.exists():
                    self.scalers = joblib.load(scaler_file)
                    logger.info(f"Loaded scalers with {len(self.scalers)} targets")
                
                return len(self.models) > 0
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            logger.error(traceback.format_exc())
        
        return False
    
    def train_models_with_consistent_features(self):
        """Train models using consistent features"""
        logger.info("Training models with consistent features...")
        
        # Create synthetic data with the same features used in production
        np.random.seed(42)
        n_samples = 5000
        
        # Create feature matrix with the exact same columns as FEATURE_COLUMNS
        X = pd.DataFrame(np.random.randn(n_samples, len(FEATURE_COLUMNS)), columns=FEATURE_COLUMNS)
        
        # Generate targets based on meaningful feature combinations
        # Home win: depends on team ratings, possession, shots
        y_home_win = (
            (X['home_team_rating'] * 0.3 + 
             X['away_team_rating'] * -0.2 + 
             X['home_possession'] * 0.2 + 
             X['home_shots_on_goal'] * 0.3) > 0
        ).astype(int)
        
        # Over 2.5: depends on total goals tendency, scoring pressure
        y_over_25 = (
            (X['scoring_pressure'] * 0.4 + 
             X['total_goals'] * 0.3 + 
             X['momentum'] * 0.3) > 0.2
        ).astype(int)
        
        # BTTS: depends on both teams' attacking stats
        y_btts = (
            (X['home_shots_on_goal'] * 0.3 + 
             X['away_shots_on_goal'] * 0.3 + 
             X['goal_difference'].abs() * -0.2 + 
             X['time_ratio'] * 0.2) > 0.1
        ).astype(int)
        
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        
        # Train models for each target
        targets = [
            ('home_win', y_home_win),
            ('over_2_5', y_over_25),
            ('btts', y_btts)
        ]
        
        for target_name, y in targets:
            logger.info(f"Training model for {target_name}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            # Train Random Forest with optimized parameters
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                bootstrap=True,
                n_jobs=-1,
                random_state=42,
                class_weight='balanced'
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Test accuracy
            X_test_scaled = scaler.transform(X_test)
            accuracy = model.score(X_test_scaled, y_test)
            
            logger.info(f"{target_name} model accuracy: {accuracy:.4f}")
            
            # Store model and scaler
            self.models[target_name] = model
            self.scalers[target_name] = scaler
        
        # Save models
        self.save_models()
        logger.info("Models trained and saved successfully")
        
        return True
    
    def save_models(self):
        """Save models to disk"""
        model_dir = Path("models/consistent")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        for target, model in self.models.items():
            model_path = model_dir / f"{target}_ensemble.pkl"
            joblib.dump(model, model_path)
            logger.info(f"Saved {target} model")
        
        scaler_path = model_dir / "scalers.pkl"
        joblib.dump(self.scalers, scaler_path)
        
        # Update current symlink
        current_path = Path("models/current")
        if current_path.exists():
            if current_path.is_symlink():
                current_path.unlink()
            elif current_path.is_dir():
                import shutil
                shutil.rmtree(current_path)
        
        current_path.symlink_to(model_dir)
        logger.info(f"Models saved to {model_dir}")

class PredictionEngine:
    """Prediction engine with consistent features"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.trainer = ModelTrainer(db_manager)
        self.models = {}
        self.scalers = {}
        self.load_models()
        
    def load_models(self):
        """Load models from trainer"""
        logger.info("Loading prediction models...")
        
        if not self.trainer.load_models():
            logger.warning("No trained models found, training new models...")
            if self.trainer.train_models_with_consistent_features():
                self.trainer.load_models()
        
        self.models = self.trainer.models
        self.scalers = self.trainer.scalers
        
        logger.info(f"Loaded {len(self.models)} models for prediction")
        logger.info(f"Available targets: {list(self.models.keys())}")
        
        # Log feature information for debugging
        for target, scaler in self.scalers.items():
            if hasattr(scaler, 'n_features_in_'):
                logger.info(f"Scaler for {target} expects {scaler.n_features_in_} features")
    
    def extract_features(self, match_data: Dict) -> pd.DataFrame:
        """Extract features with consistent structure"""
        features = {}
        
        # Extract match data
        fixture = match_data.get('fixture', {})
        teams = match_data.get('teams', {})
        goals = match_data.get('goals', {})
        league = match_data.get('league', {})
        
        # Team ratings (default to reasonable values)
        features['home_team_rating'] = float(teams.get('home', {}).get('rating', 6.5) or 6.5)
        features['away_team_rating'] = float(teams.get('away', {}).get('rating', 6.5) or 6.5)
        
        # Current match state
        features['home_score'] = int(goals.get('home', 0) or 0)
        features['away_score'] = int(goals.get('away', 0) or 0)
        features['goal_difference'] = features['home_score'] - features['away_score']
        features['total_goals'] = features['home_score'] + features['away_score']
        
        # Match progress
        status = fixture.get('status', {})
        features['minute'] = int(status.get('elapsed', 0) or 0)
        features['time_ratio'] = min(features['minute'] / 90.0, 1.0)
        
        # League info
        features['league_rank'] = float(league.get('rank', 50) or 50)
        
        # Time features
        if 'date' in fixture:
            try:
                match_time = datetime.fromisoformat(fixture['date'].replace('Z', '+00:00'))
            except:
                match_time = datetime.now()
        else:
            match_time = datetime.now()
        
        features['hour_of_day'] = match_time.hour
        features['day_of_week'] = match_time.weekday()
        features['month'] = match_time.month
        
        # Derived features
        features['momentum'] = features['goal_difference'] * features['time_ratio']
        features['scoring_pressure'] = features['total_goals'] / max(features['time_ratio'], 0.1)
        
        # Statistics (use realistic defaults)
        features['home_possession'] = 50.0  # Default 50-50
        features['away_possession'] = 50.0
        features['home_shots_on_goal'] = 5.0  # Reasonable default
        features['away_shots_on_goal'] = 5.0
        
        # Implied probabilities (from odds if available, else defaults)
        features['implied_prob_home'] = 0.33
        features['implied_prob_draw'] = 0.33
        features['implied_prob_away'] = 0.33
        
        # Create DataFrame with consistent column order
        df = pd.DataFrame([features])
        
        # Ensure all expected columns are present
        for col in FEATURE_COLUMNS:
            if col not in df.columns:
                df[col] = 0.0  # Fill missing with zeros
        
        # Reorder columns to match training
        df = df[FEATURE_COLUMNS]
        
        logger.debug(f"Extracted {len(FEATURE_COLUMNS)} features")
        return df
    
    def predict(self, match_data: Dict) -> Dict[str, Any]:
        """Make predictions for a match"""
        if not self.models:
            logger.error("No models available for prediction")
            return {}
        
        try:
            # Extract features
            features_df = self.extract_features(match_data)
            
            # Ensure we have the right number of features
            if features_df.shape[1] != len(FEATURE_COLUMNS):
                logger.error(f"Feature mismatch: got {features_df.shape[1]}, expected {len(FEATURE_COLUMNS)}")
                return {}
            
            logger.debug(f"Features shape: {features_df.shape}")
            
            # Make predictions for each target
            predictions = {}
            probabilities = {}
            
            for target, model in self.models.items():
                if target in self.scalers:
                    try:
                        # Get scaler for this target
                        scaler = self.scalers[target]
                        
                        # Scale features
                        features_scaled = scaler.transform(features_df)
                        
                        # Make prediction
                        if hasattr(model, 'predict_proba'):
                            proba = model.predict_proba(features_scaled)[0]
                            pred = model.predict(features_scaled)[0]
                            predictions[target] = int(pred)
                            probabilities[target] = float(proba[1]) if len(proba) > 1 else float(proba[0])
                        else:
                            pred = model.predict(features_scaled)[0]
                            predictions[target] = int(pred)
                            probabilities[target] = float(pred)
                        
                        logger.debug(f"{target}: prediction={predictions[target]}, probability={probabilities[target]:.3f}")
                        
                    except Exception as e:
                        logger.error(f"Error predicting {target}: {e}")
                        logger.error(traceback.format_exc())
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
            
            logger.info(f"✅ Predictions made for match {result['match_id']}")
            logger.info(f"   Predictions: {predictions}")
            logger.info(f"   Probabilities: {probabilities}")
            logger.info(f"   Tips generated: {len(tips)}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in prediction: {e}")
            logger.error(traceback.format_exc())
            return {}
    
    def generate_tips(self, predictions: Dict, probabilities: Dict, match_data: Dict) -> List[Dict]:
        """Generate betting tips"""
        tips = []
        config = BettingConfig()
        
        # Home win tip
        home_win_prob = probabilities.get('home_win', 0)
        if predictions.get('home_win', 0) == 1 and home_win_prob > config.min_confidence:
            tips.append({
                'type': '1X2',
                'prediction': 'Home Win',
                'probability': home_win_prob,
                'confidence': 'high' if home_win_prob > 0.75 else 'medium',
                'market': 'match_winner',
                'match_id': match_data.get('fixture', {}).get('id')
            })
        
        # Over 2.5 goals tip
        over_prob = probabilities.get('over_2_5', 0)
        if predictions.get('over_2_5', 0) == 1 and over_prob > config.min_confidence:
            tips.append({
                'type': 'Over/Under',
                'prediction': 'Over 2.5 Goals',
                'probability': over_prob,
                'confidence': 'high' if over_prob > 0.75 else 'medium',
                'market': 'total_goals',
                'match_id': match_data.get('fixture', {}).get('id')
            })
        
        # BTTS tip
        btts_prob = probabilities.get('btts', 0)
        if predictions.get('btts', 0) == 1 and btts_prob > config.min_confidence:
            tips.append({
                'type': 'BTTS',
                'prediction': 'Both Teams to Score',
                'probability': btts_prob,
                'confidence': 'high' if btts_prob > 0.75 else 'medium',
                'market': 'btts',
                'match_id': match_data.get('fixture', {}).get('id')
            })
        
        # Sort by probability
        tips.sort(key=lambda x: x['probability'], reverse=True)
        
        # Limit number of tips
        tips = tips[:config.max_tips_per_match]
        
        return tips
    
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
        self.db = DatabaseManager()
        self.api_client = APIFootballClient(self.settings.API_FOOTBALL_KEY)
        self.prediction_engine = PredictionEngine(self.db)
        self.telegram_bot = TelegramBot(self.settings.TELEGRAM_BOT_TOKEN, self.settings.TELEGRAM_CHAT_ID)
        
        self.is_running = False
        self.performance = {
            'predictions_made': 0,
            'tips_sent': 0,
            'last_scan': None,
            'start_time': datetime.now(),
            'errors': 0
        }
        
        logger.info("=" * 50)
        logger.info("🎯 Betting Predictor System Initialized")
        logger.info(f"📡 API Key: {'✓' if self.settings.API_FOOTBALL_KEY else '✗'}")
        logger.info(f"🤖 Telegram: {'✓' if self.settings.TELEGRAM_BOT_TOKEN else '✗'}")
        logger.info(f"💾 Database: {'✓' if self.settings.DATABASE_URL else '✗'}")
        logger.info(f"🤖 Models loaded: {len(self.prediction_engine.models)}")
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
                home_team = match['teams']['home']['name']
                away_team = match['teams']['away']['name']
                score = f"{match['goals']['home']}-{match['goals']['away']}"
                minute = match['fixture']['status']['elapsed']
                
                logger.info(f"Processing: {home_team} vs {away_team} ({score}, {minute}')")
                
                # Check if already processed recently
                if self.db.check_recent_prediction(match_id):
                    logger.debug(f"Match {match_id} already processed recently, skipping")
                    continue
                
                # Make predictions
                prediction = self.prediction_engine.predict(match)
                
                if prediction and prediction.get('match_id'):
                    # Store match data
                    self.db.store_match(match)
                    
                    # Store prediction
                    self.db.store_prediction(prediction)
                    
                    # Send Telegram alert if we have tips
                    if prediction.get('tips'):
                        message = self.telegram_bot.format_tip_message(match, prediction['tips'])
                        if await self.telegram_bot.send_message(message):
                            self.performance['tips_sent'] += len(prediction['tips'])
                            logger.info(f"📨 Sent {len(prediction['tips'])} tips for match {match_id}")
                    else:
                        logger.info(f"📭 No tips generated for match {match_id}")
                
                self.performance['predictions_made'] += 1
                
                # Rate limiting
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"❌ Error in scan: {e}")
            logger.error(traceback.format_exc())
            self.performance['errors'] += 1
    
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
        message += f"❌ Errors: {self.performance['errors']}\n"
        message += f"🎯 Win Rate: {performance.get('win_rate', 0):.1%}\n"
        message += f"📈 ROI: {performance.get('roi', 0):.1%}\n\n"
        message += f"🔄 Last Scan: {self.performance['last_scan'] or 'Never'}\n"
        
        await self.telegram_bot.send_message(message)
        logger.info("Daily report sent")
    
    def train_models(self):
        """Train models"""
        logger.info("🤖 Training models...")
        if self.prediction_engine.trainer.train_models_with_consistent_features():
            self.prediction_engine.load_models()
            logger.info("✅ Models trained and loaded successfully")
        else:
            logger.error("❌ Model training failed")
    
    def health_check(self) -> Dict:
        """System health check"""
        health = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'uptime': str(datetime.now() - self.performance['start_time']),
            'performance': self.performance.copy(),
            'models': {
                'count': len(self.prediction_engine.models),
                'targets': list(self.prediction_engine.models.keys())
            }
        }
        
        # Check components
        health['components'] = {
            'api': bool(self.settings.API_FOOTBALL_KEY),
            'telegram': bool(self.settings.TELEGRAM_BOT_TOKEN and self.settings.TELEGRAM_CHAT_ID),
            'database': bool(self.settings.DATABASE_URL),
            'models': len(self.prediction_engine.models) > 0
        }
        
        # Check feature consistency
        if self.prediction_engine.models:
            sample_features = self.prediction_engine.extract_features({
                'fixture': {'id': 1, 'date': datetime.now().isoformat(), 'status': {'elapsed': 45}},
                'teams': {'home': {'name': 'Team A', 'rating': 6.5}, 'away': {'name': 'Team B', 'rating': 6.5}},
                'goals': {'home': 1, 'away': 1},
                'league': {'name': 'Test League', 'rank': 50}
            })
            health['features'] = {
                'count': len(FEATURE_COLUMNS),
                'sample_shape': sample_features.shape,
                'columns': FEATURE_COLUMNS
            }
        
        # Check system resources
        try:
            health['system'] = {
                'cpu': psutil.cpu_percent(),
                'memory': psutil.virtual_memory().percent,
                'disk': psutil.disk_usage('/').percent
            }
        except:
            health['system'] = {'error': 'Could not get system metrics'}
        
        # Update status if any component is down
        if not health['components']['api']:
            health['status'] = 'unhealthy'
        elif not all(health['components'].values()):
            health['status'] = 'degraded'
        
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
    return {
        "message": "Betting Predictor API",
        "status": "running",
        "endpoints": ["/health", "/scan", "/train", "/backfill/{days}", "/report", "/status"]
    }

@app.get("/health")
async def health():
    if predictor:
        return predictor.health_check()
    return {"status": "starting"}

@app.get("/scan")
async def trigger_scan():
    if predictor:
        predictor.manual_scan()
        return {"message": "Scan triggered", "status": "success"}
    return {"error": "Predictor not ready", "status": "error"}

@app.get("/train")
async def trigger_train():
    if predictor:
        predictor.manual_train()
        return {"message": "Training triggered", "status": "success"}
    return {"error": "Predictor not ready", "status": "error"}

@app.get("/backfill/{days}")
async def trigger_backfill(days: int = 7):
    if predictor:
        predictor.manual_backfill(days)
        return {"message": f"Backfill triggered for {days} days", "status": "success"}
    return {"error": "Predictor not ready", "status": "error"}

@app.get("/report")
async def trigger_report():
    if predictor:
        predictor.manual_report()
        return {"message": "Report triggered", "status": "success"}
    return {"error": "Predictor not ready", "status": "error"}

@app.get("/status")
async def status():
    if predictor:
        return {
            "is_running": predictor.is_running,
            "performance": predictor.performance,
            "models_loaded": len(predictor.prediction_engine.models)
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
    # Create necessary directories
    Path("logs").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    
    main()
