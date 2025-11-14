from fastapi import FastAPI, BackgroundTasks
import uvicorn
import joblib
import asyncio
import aiohttp
import os
from datetime import datetime
import pandas as pd
from supabase import create_client
import requests
from telegram import Bot
from telegram.error import TelegramError

# Import training components
from train_models import AdvancedBettingPredictor, DataFetcher

# Configuration
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
API_FOOTBALL_KEY = os.getenv('API_FOOTBALL_KEY')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# Initialize FastAPI
app = FastAPI(title="AI Betting Predictor")

# Global variables
predictor = None
data_fetcher = None
telegram_bot = None

class PredictionEngine:
    def __init__(self):
        self.min_confidence = 0.65
        self.min_probability = 0.60
        self.min_edge = 0.05  # 5% edge over bookmaker
        
    async def analyze_live_matches(self):
        """Analyze live matches for betting opportunities"""
        print("Analyzing live matches...")
        
        live_matches = await data_fetcher.fetch_live_matches()
        valuable_tips = []
        
        for match in live_matches:
            try:
                # Extract match data for prediction
                match_data = self._prepare_match_data(match)
                
                # Generate predictions
                predictions = predictor.predict_with_confidence(match_data)
                
                # Filter valuable tips
                valuable_tips.extend(
                    self._filter_valuable_tips(match, predictions)
                )
                
            except Exception as e:
                print(f"Error analyzing match {match.get('id')}: {e}")
                continue
        
        return valuable_tips
    
    def _prepare_match_data(self, match):
        """Prepare match data for prediction"""
        fixture = match.get('fixture', {})
        teams = match.get('teams', {})
        goals = match.get('goals', {})
        league = match.get('league', {})
        
        # Basic match info
        match_data = {
            'match_id': fixture.get('id'),
            'league_id': league.get('id'),
            'home_team': teams.get('home', {}).get('name'),
            'away_team': teams.get('away', {}).get('name'),
            'is_live': fixture.get('status', {}).get('short') in ['1H', '2H', 'HT'],
            'minutes_played': fixture.get('status', {}).get('elapsed', 0),
            'current_home_goals': goals.get('home', 0),
            'current_away_goals': goals.get('away', 0),
            'league_avg_goals': league.get('goals', {}).get('avg', 2.5)
        }
        
        # Add live-specific features
        if match_data['is_live']:
            match_data.update(self._extract_live_features(match))
        
        return match_data
    
    def _extract_live_features(self, match):
        """Extract live match features"""
        events = match.get('events', [])
        statistics = match.get('statistics', [])
        
        # Calculate momentum indicators
        recent_events = [e for e in events if e.get('time', {}).get('elapsed', 0) >= 70]  # Last 20 mins
        recent_attacks = len([e for e in recent_events if e.get('type') in ['Goal', 'Shot on Target', 'Corner']])
        
        # Calculate form and momentum from recent events
        home_attacks = len([e for e in recent_events if e.get('team', {}).get('id') == match['teams']['home']['id']])
        away_attacks = len([e for e in recent_events if e.get('team', {}).get('id') == match['teams']['away']['id']])
        
        return {
            'recent_attacks': recent_attacks,
            'home_momentum': home_attacks / max(recent_attacks, 1),
            'away_momentum': away_attacks / max(recent_attacks, 1),
            'momentum_indicator': (home_attacks - away_attacks) / max(recent_attacks, 1)
        }
    
    def _filter_valuable_tips(self, match, predictions):
        """Filter predictions to find valuable betting tips"""
        valuable_tips = []
        
        for bet_type, prediction in predictions.items():
            if (prediction['confidence'] >= self.min_confidence and
                prediction['probability'] >= self.min_probability and
                prediction['edge'] >= self.min_edge):
                
                tip = {
                    'match_id': match['fixture']['id'],
                    'home_team': match['teams']['home']['name'],
                    'away_team': match['teams']['away']['name'],
                    'league': match['league']['name'],
                    'bet_type': bet_type.upper(),
                    'probability': round(prediction['probability'] * 100, 2),
                    'confidence': round(prediction['confidence'] * 100, 2),
                    'edge': round(prediction['edge'] * 100, 2),
                    'prediction': prediction['prediction'],
                    'timestamp': datetime.now().isoformat(),
                    'minutes_played': match['fixture']['status']['elapsed'] if match['fixture']['status']['elapsed'] else 0,
                    'current_score': f"{match['goals']['home']}-{match['goals']['away']}"
                }
                
                valuable_tips.append(tip)
        
        return valuable_tips

class TelegramNotifier:
    def __init__(self):
        self.bot = Bot(token=TELEGRAM_BOT_TOKEN) if TELEGRAM_BOT_TOKEN else None
    
    async def send_tip(self, tip):
        """Send betting tip to Telegram"""
        if not self.bot:
            print("Telegram bot not configured")
            return
        
        try:
            message = self._format_tip_message(tip)
            await self.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=message,
                parse_mode='HTML'
            )
            print(f"Tip sent to Telegram: {tip['bet_type']} for {tip['home_team']} vs {tip['away_team']}")
        except TelegramError as e:
            print(f"Telegram error: {e}")
    
    def _format_tip_message(self, tip):
        """Format tip as HTML message"""
        emoji = "🎯" if tip['confidence'] > 75 else "⚡"
        
        return f"""
{emoji} <b>AI BETTING TIP FOUND</b> {emoji}

🏆 <b>League:</b> {tip['league']}
⚽ <b>Match:</b> {tip['home_team']} vs {tip['away_team']}
📊 <b>Current Score:</b> {tip['current_score']} ({tip['minutes_played']}')
🎲 <b>Bet Type:</b> {tip['bet_type']}
📈 <b>Probability:</b> {tip['probability']}%
💪 <b>Confidence:</b> {tip['confidence']}%
💰 <b>Edge:</b> +{tip['edge']}%

⏰ <i>Generated at: {datetime.now().strftime('%H:%M:%S')}</i>
        """

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    global predictor, data_fetcher, telegram_bot
    
    print("Starting AI Betting Predictor...")
    
    # Initialize components
    data_fetcher = DataFetcher()
    telegram_bot = TelegramNotifier()
    
    # Load trained models
    try:
        predictor = joblib.load('models/trained_predictor.joblib')
        print("Trained models loaded successfully")
    except FileNotFoundError:
        print("No trained models found. Please run training first.")
        # Initialize empty predictor
        predictor = AdvancedBettingPredictor()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": predictor is not None
    }

@app.get("/predict/live")
async def predict_live_matches(background_tasks: BackgroundTasks):
    """Analyze live matches and return predictions"""
    try:
        prediction_engine = PredictionEngine()
        valuable_tips = await prediction_engine.analyze_live_matches()
        
        # Send tips via Telegram in background
        for tip in valuable_tips:
            background_tasks.add_task(telegram_bot.send_tip, tip)
        
        return {
            "status": "success",
            "tips_found": len(valuable_tips),
            "tips": valuable_tips,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.post("/learn")
async def learn_from_outcomes(bet_outcomes: list):
    """Endpoint to learn from bet outcomes"""
    try:
        predictor.learn_from_mistakes(bet_outcomes)
        
        # Retrain models periodically with new data
        if len(predictor.calibration_data) >= 100:  # Retrain after 100 new samples
            background_tasks = BackgroundTasks()
            background_tasks.add_task(retrain_models)
        
        return {
            "status": "success",
            "message": f"Learned from {len(bet_outcomes)} outcomes",
            "calibration_data_size": len(predictor.calibration_data)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

async def retrain_models():
    """Retrain models with new calibration data"""
    print("Retraining models with new data...")
    
    # Combine historical data with calibration data
    historical_data = await data_fetcher.fetch_historical_data(30)  # Last 30 days
    
    if historical_data and hasattr(predictor, 'calibration_data'):
        # Convert calibration data to training format
        calibration_training_data = [
            {**item['match_data'], 'actual_result': item['correct_prediction']}
            for item in predictor.calibration_data
        ]
        
        # Combine datasets and retrain
        combined_data = historical_data + calibration_training_data
        predictor.train_models(combined_data)
        
        # Save updated models
        joblib.dump(predictor, 'models/trained_predictor.joblib')
        
        print("Models retrained successfully!")
        
        # Clear calibration data
        predictor.calibration_data = []

@app.get("/stats")
async def get_system_stats():
    """Get system statistics and performance metrics"""
    if not predictor:
        return {"error": "Predictor not initialized"}
    
    return {
        "models_trained": len(predictor.models) > 0,
        "bayesian_priors_count": len(predictor.bayesian_net.prior_probabilities),
        "calibration_data_size": len(predictor.calibration_data),
        "feature_count": len(predictor.feature_columns) if predictor.feature_columns else 0,
        "last_updated": datetime.now().isoformat()
    }

# Background task for continuous live analysis
async def continuous_live_analysis():
    """Continuously analyze live matches"""
    while True:
        try:
            prediction_engine = PredictionEngine()
            valuable_tips = await prediction_engine.analyze_live_matches()
            
            for tip in valuable_tips:
                await telegram_bot.send_tip(tip)
            
            # Wait before next analysis
            await asyncio.sleep(60)  # Analyze every minute
            
        except Exception as e:
            print(f"Continuous analysis error: {e}")
            await asyncio.sleep(30)

if __name__ == "__main__":
    # Start continuous analysis in background
    asyncio.create_task(continuous_live_analysis())
    
    # Start FastAPI server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=False
    )
