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
from config import Settings, BettingConfig, DatabaseManager, FEATURE_COLUMNS, CalibrationEngine

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
                        matches = data.get('response', [])
                        # Filter for matches with reasonable minutes
                        filtered_matches = []
                        for match in matches:
                            minute = match.get('fixture', {}).get('status', {}).get('elapsed', 0)
                            if minute >= 20:  # Only include matches with at least 20 minutes played
                                filtered_matches.append(match)
                        return filtered_matches
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
    """Model trainer with game-state aware features"""
    
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
    
    def train_models_with_game_state_features(self):
        """Train models with game-state aware features"""
        logger.info("Training models with game-state features...")
        
        # Create synthetic data with game-state features
        np.random.seed(42)
        n_samples = 10000
        
        # Create feature matrix with game-state aware features
        X = pd.DataFrame(np.random.randn(n_samples, len(FEATURE_COLUMNS)), columns=FEATURE_COLUMNS)
        
        # Add game-state logic to features
        X['is_late_game'] = (X['minute'] > 80).astype(int)
        X['is_extra_time'] = (X['minute'] > 90).astype(int)
        X['goals_needed_for_over'] = np.maximum(0, 3 - X['total_goals'])
        X['can_btts_happen'] = ((X['home_score'] > 0) & (X['away_score'] > 0)).astype(int)
        
        # Generate targets with game-state logic
        # Home win: less likely in late game when drawing
        base_home_win = (
            X['home_team_rating'] * 0.3 + 
            X['away_team_rating'] * -0.2 + 
            X['home_possession'] * 0.15
        )
        # Reduce probability in late game when scores are level
        late_game_penalty = X['is_late_game'] * (X['home_score'] == X['away_score']) * -0.3
        y_home_win = ((base_home_win + late_game_penalty) > 0).astype(int)
        
        # Over 2.5: less likely in late game when more goals needed
        base_over_25 = (
            X['scoring_pressure'] * 0.4 + 
            X['total_goals'] * 0.3
        )
        # Penalize when many goals needed late
        goals_needed_penalty = X['goals_needed_for_over'] * X['is_late_game'] * -0.2
        y_over_25 = ((base_over_25 + goals_needed_penalty) > 0.2).astype(int)
        
        # BTTS: already happened or not
        y_btts = X['can_btts_happen'].copy()
        # Add some noise for prediction
        y_btts = (y_btts + np.random.randn(n_samples) * 0.1 > 0.5).astype(int)
        
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
            
            # Train Random Forest with game-state aware parameters
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
            
            # Feature importance analysis
            feature_importance = dict(zip(FEATURE_COLUMNS, model.feature_importances_))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            
            logger.info(f"{target_name} model accuracy: {accuracy:.4f}")
            logger.info(f"Top features: {top_features}")
            
            # Store model and scaler
            self.models[target_name] = model
            self.scalers[target_name] = scaler
        
        # Save models
        self.save_models()
        logger.info("Models trained and saved successfully")
        
        return True
    
    def save_models(self):
        """Save models to disk"""
        model_dir = Path("models/calibrated")
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
    """Calibrated prediction engine with game-state awareness"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.trainer = ModelTrainer(db_manager)
        self.models = {}
        self.scalers = {}
        self.calibration_engine = CalibrationEngine()
        self.config = BettingConfig()
        self.load_models()
        
    def load_models(self):
        """Load models from trainer"""
        logger.info("Loading prediction models...")
        
        if not self.trainer.load_models():
            logger.warning("No trained models found, training new models...")
            if self.trainer.train_models_with_game_state_features():
                self.trainer.load_models()
        
        self.models = self.trainer.models
        self.scalers = self.trainer.scalers
        
        logger.info(f"Loaded {len(self.models)} models for prediction")
        logger.info(f"Available targets: {list(self.models.keys())}")
        
        return True
    
    def extract_features(self, match_data: Dict) -> pd.DataFrame:
        """Extract features with game-state awareness"""
        features = {}
        
        # Extract match data
        fixture = match_data.get('fixture', {})
        teams = match_data.get('teams', {})
        goals = match_data.get('goals', {})
        league = match_data.get('league', {})
        
        # Basic features
        features['home_team_rating'] = float(teams.get('home', {}).get('rating', 6.5) or 6.5)
        features['away_team_rating'] = float(teams.get('away', {}).get('rating', 6.5) or 6.5)
        
        # Current match state
        features['home_score'] = int(goals.get('home', 0) or 0)
        features['away_score'] = int(goals.get('away', 0) or 0)
        features['goal_difference'] = features['home_score'] - features['away_score']
        features['total_goals'] = features['home_score'] + features['away_score']
        
        # Match progress
        status = fixture.get('status', {})
        minute = int(status.get('elapsed', 0) or 0)
        features['minute'] = minute
        features['time_ratio'] = min(minute / 90.0, 1.0)
        
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
        features['home_possession'] = 50.0
        features['away_possession'] = 50.0
        features['home_shots_on_goal'] = 5.0
        features['away_shots_on_goal'] = 5.0
        
        # Implied probabilities
        features['implied_prob_home'] = 0.33
        features['implied_prob_draw'] = 0.33
        features['implied_prob_away'] = 0.33
        
        # Game-state features
        features['is_late_game'] = 1 if minute > self.config.late_game_minute else 0
        features['is_extra_time'] = 1 if minute > self.config.extra_time_minute else 0
        features['goals_needed_for_over'] = max(0, 3 - features['total_goals'])
        features['can_btts_happen'] = 1 if features['home_score'] > 0 and features['away_score'] > 0 else 0
        
        # Create DataFrame with consistent column order
        df = pd.DataFrame([features])
        
        # Ensure all expected columns are present
        for col in FEATURE_COLUMNS:
            if col not in df.columns:
                df[col] = 0.0
        
        # Reorder columns to match training
        df = df[FEATURE_COLUMNS]
        
        logger.debug(f"Extracted {len(FEATURE_COLUMNS)} features")
        logger.debug(f"Game state: minute={minute}, score={features['home_score']}-{features['away_score']}")
        
        return df
    
    def predict(self, match_data: Dict) -> Dict[str, Any]:
        """Make calibrated predictions for a match"""
        if not self.models:
            logger.error("No models available for prediction")
            return {}
        
        minute = match_data.get('fixture', {}).get('status', {}).get('elapsed', 0)
        
        # Skip predictions for very early or very late games
        if minute < self.config.min_minutes_for_prediction:
            logger.info(f"Match at {minute}' - too early for prediction")
            return {}
        
        if minute > 110:  # Very late in extra time
            logger.info(f"Match at {minute}' - too late for meaningful prediction")
            return {}
        
        try:
            # Extract features
            features_df = self.extract_features(match_data)
            
            # Make predictions for each target
            raw_predictions = {}
            raw_probabilities = {}
            
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
                            raw_predictions[target] = int(pred)
                            raw_probabilities[target] = float(proba[1]) if len(proba) > 1 else float(proba[0])
                        else:
                            pred = model.predict(features_scaled)[0]
                            raw_predictions[target] = int(pred)
                            raw_probabilities[target] = float(pred)
                        
                    except Exception as e:
                        logger.error(f"Error predicting {target}: {e}")
                        raw_predictions[target] = 0
                        raw_probabilities[target] = 0.5
            
            # Apply calibration based on game state
            calibrated_predictions, calibrated_probabilities = self.calibration_engine.calibrate_for_game_state(
                match_data, raw_predictions, raw_probabilities
            )
            
            # Log calibration changes
            calibration_factors = {}
            for target in calibrated_probabilities:
                if target in raw_probabilities:
                    if raw_probabilities[target] > 0:
                        calibration_factors[target] = calibrated_probabilities[target] / raw_probabilities[target]
            
            self.db.log_calibration(
                match_data.get('fixture', {}).get('id'),
                raw_probabilities,
                calibrated_probabilities,
                calibration_factors
            )
            
            # Generate tips
            tips = self.generate_calibrated_tips(calibrated_predictions, calibrated_probabilities, match_data)
            
            result = {
                'match_id': match_data.get('fixture', {}).get('id'),
                'raw_predictions': raw_predictions,
                'raw_probabilities': raw_probabilities,
                'calibrated_predictions': calibrated_predictions,
                'calibrated_probabilities': calibrated_probabilities,
                'tips': tips,
                'confidence': self.calculate_confidence(calibrated_probabilities),
                'timestamp': datetime.now().isoformat(),
                'features': features_df.iloc[0].to_dict(),
                'calibration_factors': calibration_factors
            }
            
            logger.info(f"✅ Predictions made for match {result['match_id']} at {minute}'")
            logger.info(f"   Raw probabilities: { {k: f'{v:.3f}' for k, v in raw_probabilities.items()} }")
            logger.info(f"   Calibrated probabilities: { {k: f'{v:.3f}' for k, v in calibrated_probabilities.items()} }")
            logger.info(f"   Tips generated: {len(tips)}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in prediction: {e}")
            logger.error(traceback.format_exc())
            return {}
    
    def generate_calibrated_tips(self, predictions: Dict, probabilities: Dict, match_data: Dict) -> List[Dict]:
        """Generate calibrated betting tips"""
        tips = []
        config = self.config
        
        minute = match_data.get('fixture', {}).get('status', {}).get('elapsed', 0)
        home_score = match_data.get('goals', {}).get('home', 0)
        away_score = match_data.get('goals', {}).get('away', 0)
        total_goals = home_score + away_score
        
        # Check each prediction type with calibrated thresholds
        potential_tips = []
        
        # Home win tip
        home_win_prob = probabilities.get('home_win', 0)
        if (predictions.get('home_win', 0) == 1 and 
            home_win_prob > config.home_win_confidence):
            
            # Additional checks for home win
            confidence = None
            if minute > 80:
                if home_score > away_score:
                    confidence = 'high'
                elif home_score == away_score:
                    # Drawing late - less confident
                    if home_win_prob > 0.8:
                        confidence = 'medium'
                    # If not confident enough, skip this tip
                else:
                    # Losing late - skip
                    pass
            else:
                confidence = 'high' if home_win_prob > 0.8 else 'medium'
            
            if confidence:
                potential_tips.append({
                    'type': '1X2',
                    'prediction': 'Home Win',
                    'probability': home_win_prob,
                    'confidence': confidence,
                    'market': 'match_winner',
                    'match_id': match_data.get('fixture', {}).get('id')
                })
        
        # Over 2.5 goals tip
        over_prob = probabilities.get('over_2_5', 0)
        if (predictions.get('over_2_5', 0) == 1 and 
            over_prob > config.over_25_confidence):
            
            goals_needed = 3 - total_goals
            confidence = None
            
            if minute > 80:
                if goals_needed <= 0:
                    confidence = 'high'
                elif goals_needed == 1:
                    if over_prob > 0.75:
                        confidence = 'medium'
                    # If not confident enough, skip
                else:
                    # Need multiple goals late - skip
                    pass
            else:
                confidence = 'high' if over_prob > 0.78 else 'medium'
            
            if confidence:
                potential_tips.append({
                    'type': 'Over/Under',
                    'prediction': 'Over 2.5 Goals',
                    'probability': over_prob,
                    'confidence': confidence,
                    'market': 'total_goals',
                    'match_id': match_data.get('fixture', {}).get('id')
                })
        
        # BTTS tip
        btts_prob = probabilities.get('btts', 0)
        if (predictions.get('btts', 0) == 1 and 
            btts_prob > config.btts_confidence):
            
            confidence = None
            
            if home_score > 0 and away_score > 0:
                # Already happened
                confidence = 'high'
            elif minute > 85 and (home_score == 0 or away_score == 0):
                # Very late and one team hasn't scored
                if btts_prob > 0.8:
                    confidence = 'low'
                # If not confident enough, skip
            else:
                confidence = 'high' if btts_prob > 0.77 else 'medium'
            
            if confidence:
                potential_tips.append({
                    'type': 'BTTS',
                    'prediction': 'Both Teams to Score',
                    'probability': btts_prob,
                    'confidence': confidence,
                    'market': 'btts',
                    'match_id': match_data.get('fixture', {}).get('id')
                })
        
        # Apply suppression logic
        filtered_tips = []
        for tip in potential_tips:
            if not self.calibration_engine.should_suppress_tip(tip, match_data, filtered_tips):
                filtered_tips.append(tip)
        
        # Sort by probability
        filtered_tips.sort(key=lambda x: x['probability'], reverse=True)
        
        # Limit number of tips and ensure they make sense together
        final_tips = []
        for tip in filtered_tips[:config.max_tips_per_match]:
            # Don't recommend both Home Win and Over 2.5 in very late game
            if tip['type'] == '1X2' and tip['prediction'] == 'Home Win':
                has_over_tip = any(t['type'] == 'Over/Under' for t in final_tips)
                if has_over_tip and minute > 85:
                    continue  # Skip home win if we already have over tip late
            
            final_tips.append(tip)
        
        logger.info(f"Generated {len(final_tips)} tips after calibration")
        return final_tips
    
    def calculate_confidence(self, probabilities: Dict) -> float:
        """Calculate overall confidence score with game-state awareness"""
        if not probabilities:
            return 0.0
        
        # Weight probabilities by their reliability
        weights = {
            'home_win': 1.0,
            'over_2_5': 0.9,
            'btts': 0.8
        }
        
        weighted_sum = 0
        total_weight = 0
        
        for target, prob in probabilities.items():
            if target in weights:
                weighted_sum += prob * weights[target]
                total_weight += weights[target]
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0

class TelegramBot:
    """Telegram bot with calibrated messaging"""
    
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
        """Format calibrated tips for Telegram"""
        home_team = match_data.get('teams', {}).get('home', {}).get('name', 'Home')
        away_team = match_data.get('teams', {}).get('away', {}).get('name', 'Away')
        score = f"{match_data.get('goals', {}).get('home', 0)}-{match_data.get('goals', {}).get('away', 0)}"
        minute = match_data.get('fixture', {}).get('status', {}).get('elapsed', 0)
        league = match_data.get('league', {}).get('name', '')
        
        # Game state indicator
        game_state = ""
        if minute > 90:
            game_state = "⏰ EXTRA TIME "
        elif minute > 80:
            game_state = "⏰ LATE GAME "
        
        message = f"🎯 <b>CALIBRATED BETTING ALERT</b> 🎯\n"
        message += f"{game_state}\n\n"
        message += f"🏆 <b>{league}</b>\n"
        message += f"⚽ {home_team} vs {away_team}\n"
        message += f"📊 Score: {score} ({minute}')\n"
        message += f"🕒 {datetime.now().strftime('%H:%M %Y-%m-%d')}\n\n"
        
        if tips:
            message += "💰 <b>CALIBRATED RECOMMENDATIONS:</b>\n\n"
            for i, tip in enumerate(tips, 1):
                # Use appropriate emoji based on confidence and game state
                if tip['confidence'] == 'high':
                    emoji = "🔥"
                elif tip['confidence'] == 'medium':
                    emoji = "✅"
                else:
                    emoji = "⚠️"
                
                # Add time-awareness note for late games
                time_note = ""
                if minute > 80:
                    if tip['type'] == 'Over/Under' and tip['prediction'] == 'Over 2.5 Goals':
                        goals_needed = 3 - (int(score.split('-')[0]) + int(score.split('-')[1]))
                        time_note = f" (needs {goals_needed} more goal{'s' if goals_needed > 1 else ''})"
                    elif tip['type'] == '1X2' and tip['prediction'] == 'Home Win':
                        if score.split('-')[0] == score.split('-')[1]:
                            time_note = " (drawing late)"
                
                message += f"{i}. {emoji} <b>{tip['type']}</b>\n"
                message += f"   📈 {tip['prediction']}{time_note}\n"
                message += f"   🎯 Probability: {tip['probability']:.1%}\n"
                message += f"   ⭐ Confidence: {tip['confidence'].upper()}\n\n"
        else:
            message += "📭 <b>No valuable bets found</b>\n"
            message += "Game state doesn't support confident predictions.\n\n"
        
        # Add game state warning for late games
        if minute > 85:
            message += "⚠️ <b>LATE GAME WARNING:</b> Limited time remaining.\n"
        
        message += "💡 <i>Calibrated for current game state. Bet responsibly.</i>"
        
        return message

class BettingPredictor:
    """Main calibrated betting predictor system"""
    
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
            'tips_suppressed': 0,
            'last_scan': None,
            'start_time': datetime.now(),
            'errors': 0
        }
        
        logger.info("=" * 50)
        logger.info("🎯 CALIBRATED Betting Predictor System Initialized")
        logger.info(f"📡 API Key: {'✓' if self.settings.API_FOOTBALL_KEY else '✗'}")
        logger.info(f"🤖 Telegram: {'✓' if self.settings.TELEGRAM_BOT_TOKEN else '✗'}")
        logger.info(f"💾 Database: {'✓' if self.settings.DATABASE_URL else '✗'}")
        logger.info(f"🤖 Models loaded: {len(self.prediction_engine.models)}")
        logger.info(f"⚙️ Calibration: Active (game-state aware)")
        logger.info("=" * 50)
    
    async def scan_live_matches(self):
        """Scan for live matches with calibration"""
        logger.info("🔍 Scanning for live matches with calibration...")
        self.performance['last_scan'] = datetime.now()
        
        try:
            matches = await self.api_client.fetch_live_matches()
            
            if not matches:
                logger.info("No live matches found")
                return
            
            logger.info(f"Found {len(matches)} live matches (min 20' played)")
            
            for match in matches:
                match_id = match['fixture']['id']
                home_team = match['teams']['home']['name']
                away_team = match['teams']['away']['name']
                score = f"{match['goals']['home']}-{match['goals']['away']}"
                minute = match['fixture']['status']['elapsed']
                league = match.get('league', {}).get('name', 'Unknown')
                
                logger.info(f"Processing: {home_team} vs {away_team} ({league})")
                logger.info(f"  Score: {score} at {minute}'")
                
                # Skip if already processed recently
                if self.db.check_recent_prediction(match_id):
                    logger.debug(f"Match {match_id} already processed, skipping")
                    continue
                
                # Make calibrated predictions
                prediction = self.prediction_engine.predict(match)
                
                if prediction and prediction.get('match_id'):
                    # Store match data
                    self.db.store_match(match)
                    
                    # Store prediction
                    self.db.store_prediction(prediction)
                    
                    # Send Telegram alert if we have tips
                    if prediction.get('tips'):
                        tips = prediction['tips']
                        message = self.telegram_bot.format_tip_message(match, tips)
                        if await self.telegram_bot.send_message(message):
                            self.performance['tips_sent'] += len(tips)
                            logger.info(f"📨 Sent {len(tips)} calibrated tips for match {match_id}")
                            
                            # Log calibration details
                            if 'calibration_factors' in prediction:
                                logger.debug(f"Calibration factors: {prediction['calibration_factors']}")
                    else:
                        logger.info(f"📭 No tips generated after calibration for match {match_id}")
                        self.performance['tips_suppressed'] += 1
                
                self.performance['predictions_made'] += 1
                
                # Rate limiting
                await asyncio.sleep(1.5)
                
        except Exception as e:
            logger.error(f"❌ Error in scan: {e}")
            logger.error(traceback.format_exc())
            self.performance['errors'] += 1
    
    async def send_daily_report(self):
        """Send daily performance report with calibration stats"""
        logger.info("📈 Generating daily calibration report...")
        
        performance = self.db.get_daily_performance()
        uptime = datetime.now() - self.performance['start_time']
        
        # Calculate calibration effectiveness
        total_predictions = self.performance['predictions_made']
        tips_suppressed = self.performance['tips_suppressed']
        suppression_rate = (tips_suppressed / max(total_predictions, 1)) * 100
        
        message = f"📊 <b>DAILY CALIBRATION REPORT</b>\n\n"
        message += f"⏰ Uptime: {uptime}\n"
        message += f"🔍 Predictions Made: {self.performance['predictions_made']}\n"
        message += f"💰 Tips Sent: {self.performance['tips_sent']}\n"
        message += f"🚫 Tips Suppressed: {self.performance['tips_suppressed']}\n"
        message += f"📉 Suppression Rate: {suppression_rate:.1f}%\n"
        message += f"❌ Errors: {self.performance['errors']}\n"
        message += f"🎯 Win Rate: {performance.get('win_rate', 0):.1%}\n"
        message += f"📈 ROI: {performance.get('roi', 0):.1%}\n\n"
        message += f"🔄 Last Scan: {self.performance['last_scan'] or 'Never'}\n"
        message += f"⚙️ Calibration: <b>ACTIVE</b> (game-state aware)\n"
        
        await self.telegram_bot.send_message(message)
        logger.info("Daily calibration report sent")
