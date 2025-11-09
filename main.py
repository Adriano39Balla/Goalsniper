import asyncio
import aiohttp
import pandas as pd
import numpy as np
import pickle
import schedule
import time
from datetime import datetime, timedelta
from supabase import create_client
from telegram import Bot
from telegram.error import TelegramError
import os
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class AIPoweredBettingPredictor:
    def __init__(self):
        self.supabase = self._init_supabase()
        self.telegram_bot = self._init_telegram()
        self.models = self._load_models()
        self.session = None
        self.API_KEY = os.environ.get("API_FOOTBALL_KEY")
        self.BASE_URL = "https://v3.football.api-sports.io"
        
    def _init_supabase(self):
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        return create_client(url, key)
    
    def _init_telegram(self):
        token = os.environ.get("TELEGRAM_BOT_TOKEN")
        return Bot(token=token) if token else None
    
    def _load_models(self):
        try:
            with open('ensemble_models.pkl', 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            print("Models not found. Please run train_models.py first.")
            return None
    
    async def init_session(self):
        """Initialize async HTTP session"""
        if not self.session:
            self.session = aiohttp.ClientSession(
                headers={'x-apisports-key': self.API_KEY}
            )
    
    async def fetch_live_matches(self) -> List[Dict]:
        """Fetch live in-play matches with high frequency"""
        try:
            async with self.session.get(
                f"{self.BASE_URL}/fixtures",
                params={'live': 'all', 'timezone': 'Europe/London'}
            ) as response:
                data = await response.json()
                return data.get('response', [])
        except Exception as e:
            print(f"Error fetching live matches: {e}")
            return []
    
    async def fetch_match_details(self, fixture_id: int) -> Dict:
        """Fetch comprehensive match details"""
        endpoints = [
            f"/fixtures?id={fixture_id}",
            f"/fixtures/statistics?fixture={fixture_id}",
            f"/fixtures/events?fixture={fixture_id}",
            f"/predictions?fixture={fixture_id}"
        ]
        
        match_data = {}
        for endpoint in endpoints:
            try:
                async with self.session.get(f"{self.BASE_URL}{endpoint}") as response:
                    data = await response.json()
                    key = endpoint.split('?')[0].split('/')[-1]
                    match_data[key] = data.get('response', [])
            except Exception as e:
                print(f"Error fetching {endpoint}: {e}")
        
        return match_data
    
    def extract_live_features(self, match_data: Dict) -> pd.DataFrame:
        """Extract sophisticated features for live predictions"""
        features = {}
        
        try:
            fixture = match_data.get('fixtures', [{}])[0]
            statistics = match_data.get('statistics', [])
            events = match_data.get('events', [])
            
            # Basic match info
            features['match_minute'] = fixture.get('fixture', {}).get('status', {}).get('elapsed', 0)
            features['score_home'] = fixture.get('goals', {}).get('home', 0)
            features['score_away'] = fixture.get('goals', {}).get('away', 0)
            
            # In-play dynamics
            features['goal_momentum'] = self._calculate_goal_momentum(events)
            features['attack_pressure'] = self._calculate_attack_pressure(statistics)
            features['momentum_shift'] = self._detect_momentum_shift(events)
            
            # Statistical dominance
            features['possession_ratio'] = self._get_possession_ratio(statistics)
            features['shot_efficiency'] = self._get_shot_efficiency(statistics)
            features['dangerous_attacks'] = self._get_dangerous_attacks(statistics)
            
            # Contextual factors
            features['time_decay'] = np.exp(-features['match_minute'] / 90)
            features['goal_expectancy'] = self._calculate_goal_expectancy(statistics, features)
            
            return pd.DataFrame([features])
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return pd.DataFrame()
    
    def _calculate_goal_momentum(self, events: List) -> float:
        """Calculate goal scoring momentum based on recent events"""
        recent_events = [e for e in events if e.get('time', {}).get('elapsed', 0) > 70]
        goals = [e for e in recent_events if e.get('type') == 'Goal']
        return len(goals) / (len(recent_events) + 1e-6)
    
    def _calculate_attack_pressure(self, statistics: List) -> float:
        """Calculate current attacking pressure"""
        if not statistics:
            return 0.0
        
        home_stats = statistics[0].get('statistics', [{}])[0] if len(statistics) > 0 else {}
        away_stats = statistics[1].get('statistics', [{}])[0] if len(statistics) > 1 else {}
        
        pressure_indicators = [
            home_stats.get('shots_on_goal', 0),
            home_stats.get('shots_off_goal', 0),
            away_stats.get('shots_on_goal', 0), 
            away_stats.get('shots_off_goal', 0)
        ]
        
        return sum(pressure_indicators) / (max(features['match_minute'], 1) / 90)
    
    def _detect_momentum_shift(self, events: List) -> float:
        """Detect momentum shifts in the match"""
        recent_cards = len([e for e in events if e.get('type') in ['Card', 'Substitution']])
        recent_attacks = len([e for e in events if e.get('type') in ['Attempt', 'Corner']])
        
        return (recent_attacks - recent_cards) / 10.0
    
    def _get_possession_ratio(self, statistics: List) -> float:
        """Get possession ratio"""
        if not statistics or len(statistics) < 2:
            return 0.5
        
        home_possession = float(statistics[0].get('statistics', [{}])[0].get('possession', 0) or 0)
        return home_possession / 100.0
    
    def _get_shot_efficiency(self, statistics: List) -> float:
        """Calculate shot efficiency"""
        if not statistics:
            return 0.0
        
        home_stats = statistics[0].get('statistics', [{}])[0] if len(statistics) > 0 else {}
        shots_on_target = home_stats.get('shots_on_goal', 0)
        total_shots = home_stats.get('total_shots', 1)
        
        return shots_on_target / total_shots
    
    def _get_dangerous_attacks(self, statistics: List) -> float:
        """Calculate dangerous attacks metric"""
        if not statistics:
            return 0.0
        
        home_stats = statistics[0].get('statistics', [{}])[0] if len(statistics) > 0 else {}
        away_stats = statistics[1].get('statistics', [{}])[0] if len(statistics) > 1 else {}
        
        dangerous_attacks = (
            home_stats.get('attacks', 0) + 
            away_stats.get('attacks', 0) +
            home_stats.get('dangerous_attacks', 0) +
            away_stats.get('dangerous_attacks', 0)
        )
        
        return dangerous_attacks / 100.0
    
    def _calculate_goal_expectancy(self, statistics: List, features: Dict) -> float:
        """Calculate expected goals based on current match state"""
        base_xg = (features['score_home'] + features['score_away']) / 2
        time_factor = (90 - features['match_minute']) / 90
        pressure_factor = features['attack_pressure']
        
        return base_xg + (pressure_factor * time_factor)
    
    def predict_over_25_probability(self, features: pd.DataFrame) -> float:
        """Generate ensemble prediction for Over 2.5 goals"""
        if self.models is None or features.empty:
            return 0.0
        
        try:
            predictions = []
            weights = {'xgb': 0.35, 'lgb': 0.30, 'catboost': 0.25, 'logistic_calibrated': 0.10}
            
            for model_name, weight in weights.items():
                if model_name in self.models['models']:
                    model = self.models['models'][model_name]
                    pred = model.predict_proba(features)[:, 1]
                    predictions.append(pred * weight)
            
            if predictions:
                ensemble_pred = np.sum(predictions, axis=0)
                # Apply Bayesian calibration if available
                if 'calibration_model' in self.models:
                    ensemble_pred = self.models['calibration_model']
                
                return float(ensemble_pred[0])
            
            return 0.0
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return 0.0
    
    async def analyze_and_predict(self, match_data: Dict) -> Optional[Dict]:
        """Complete analysis and prediction pipeline"""
        features = self.extract_live_features(match_data)
        
        if features.empty:
            return None
        
        probability = self.predict_over_25_probability(features)
        confidence = self._calculate_confidence(probability, features)
        
        # Only consider high-confidence predictions
        if confidence > 0.75 and probability > 0.65:
            fixture = match_data.get('fixtures', [{}])[0]
            return {
                'fixture_id': fixture.get('fixture', {}).get('id'),
                'home_team': fixture.get('teams', {}).get('home', {}).get('name'),
                'away_team': fixture.get('teams', {}).get('away', {}).get('name'),
                'probability': probability,
                'confidence': confidence,
                'current_score': f"{features['score_home'].iloc[0]}-{features['score_away'].iloc[0]}",
                'minute': features['match_minute'].iloc[0],
                'timestamp': datetime.now().isoformat()
            }
        
        return None
    
    def _calculate_confidence(self, probability: float, features: pd.DataFrame) -> float:
        """Calculate prediction confidence based on multiple factors"""
        base_confidence = probability
        
        # Boost confidence for strong signals
        if features['attack_pressure'].iloc[0] > 0.7:
            base_confidence *= 1.2
        
        if features['goal_momentum'].iloc[0] > 0.5:
            base_confidence *= 1.1
        
        # Reduce confidence for late-game predictions with low scores
        if features['match_minute'].iloc[0] > 75 and features['score_home'].iloc[0] + features['score_away'].iloc[0] < 1:
            base_confidence *= 0.7
        
        return min(base_confidence, 1.0)
    
    async def send_telegram_alert(self, prediction: Dict):
        """Send formatted prediction to Telegram"""
        if not self.telegram_bot:
            print("Telegram bot not configured")
            return
        
        try:
            message = f"""
🎯 **AI BETTING PREDICTION ALERT** 🎯

⚽ **Match**: {prediction['home_team']} vs {prediction['away_team']}
📊 **Current Score**: {prediction['current_score']} (Minute: {prediction['minute']})
🎲 **Prediction**: OVER 2.5 GOALS
📈 **Probability**: {prediction['probability']:.1%}
✅ **Confidence**: {prediction['confidence']:.1%}
⏰ **Time**: {datetime.now().strftime('%H:%M:%S')}

*AI-powered analysis based on live in-play data*
            """
            
            chat_id = os.environ.get("TELEGRAM_CHAT_ID")
            await self.telegram_bot.send_message(
                chat_id=chat_id,
                text=message,
                parse_mode='Markdown'
            )
            
            print(f"Telegram alert sent for {prediction['home_team']} vs {prediction['away_team']}")
            
        except TelegramError as e:
            print(f"Telegram error: {e}")
    
    async def track_prediction_outcome(self, prediction: Dict):
        """Track prediction outcomes for self-learning"""
        try:
            # Store prediction in Supabase
            self.supabase.table('predictions').insert({
                'fixture_id': prediction['fixture_id'],
                'prediction': 'over_2.5',
                'probability': prediction['probability'],
                'confidence': prediction['confidence'],
                'timestamp': prediction['timestamp'],
                'status': 'active'
            }).execute()
        except Exception as e:
            print(f"Error tracking prediction: {e}")
    
    async def run_prediction_cycle(self):
        """Main prediction cycle"""
        print(f"Running prediction cycle at {datetime.now()}")
        
        await self.init_session()
        live_matches = await self.fetch_live_matches()
        
        predictions_sent = 0
        for match in live_matches:
            fixture_id = match['fixture']['id']
            
            # Skip if we already predicted this match recently
            existing_pred = self.supabase.table('predictions').select('*').eq('fixture_id', fixture_id).execute()
            if existing_pred.data:
                continue
            
            match_data = await self.fetch_match_details(fixture_id)
            prediction = await self.analyze_and_predict(match_data)
            
            if prediction:
                await self.send_telegram_alert(prediction)
                await self.track_prediction_outcome(prediction)
                predictions_sent += 1
                
                # Small delay to avoid rate limiting
                await asyncio.sleep(1)
        
        print(f"Cycle complete. Sent {predictions_sent} predictions")
    
    async def start_monitoring(self):
        """Start continuous monitoring"""
        print("Starting AI Betting Predictor Monitoring...")
        
        # Run immediately on start
        await self.run_prediction_cycle()
        
        # Schedule regular runs
        schedule.every(1).minutes.do(
            lambda: asyncio.create_task(self.run_prediction_cycle())
        )
        
        # Run scheduled tasks
        while True:
            schedule.run_pending()
            await asyncio.sleep(1)

async def main():
    predictor = AIPoweredBettingPredictor()
    await predictor.start_monitoring()

if __name__ == "__main__":
    asyncio.run(main())
