#!/usr/bin/env python3
"""
Football Prediction System Backend
Autonomous market selection with extensive logging
"""
import os
import sys
import json
import time
import functools
import urllib.parse
from datetime import datetime, timedelta
from threading import Thread, Lock
from typing import Dict, List, Optional, Tuple, Any
import psycopg2

from psycopg2.pool import SimpleConnectionPool
from flask import Flask, request, jsonify
from flask_cors import CORS
from loguru import logger
from apscheduler.schedulers.background import BackgroundScheduler
from scipy.stats import poisson
import pandas as pd
import numpy as np
import requests
import joblib
import pickle
import psutil

# Import modules
from database import DatabaseManager
from train_models import ModelTrainer

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables with thread safety
db = DatabaseManager()
trainer = ModelTrainer()
scheduler = BackgroundScheduler()
model_lock = Lock()

class APICache:
    """Simple API response cache with TTL"""
    def __init__(self, ttl_minutes: int = 60):
        self.cache = {}
        self.ttl = timedelta(minutes=ttl_minutes)
        self.lock = Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value if not expired"""
        with self.lock:
            if key in self.cache:
                data, timestamp = self.cache[key]
                if datetime.now() - timestamp < self.ttl:
                    logger.debug(f"Cache hit for key: {key}")
                    return data
                else:
                    del self.cache[key]
            return None
    
    def set(self, key: str, value: Any) -> None:
        """Set cached value with timestamp"""
        with self.lock:
            self.cache[key] = (value, datetime.now())
            logger.debug(f"Cache set for key: {key}")
    
    def clear(self) -> None:
        """Clear all cached data"""
        with self.lock:
            self.cache.clear()
            logger.info("Cache cleared")

class FootballPredictor:
    def __init__(self):
        self.api_key = os.getenv('API_FOOTBALL_KEY')
        if not self.api_key:
            logger.error("API_FOOTBALL_KEY not found in environment variables")
            raise ValueError("API_FOOTBALL_KEY is required")
            
        self.api_base = "https://v3.football.api-sports.io"
        self.headers = {
            'x-apisports-key': self.api_key,
            'x-rapidapi-host': 'v3.football.api-sports.io'
        }
        self.telegram_bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.models_loaded = False
        self.api_call_count = 0
        self.api_cache = APICache(ttl_minutes=30)
        self.setup_logging()
        
    def setup_logging(self):
        """Configure extensive logging"""
        # Remove default logger
        logger.remove()
        
        # Console logging
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level="INFO"
        )
        
        # File logging - General
        logger.add(
            "logs/system_{time:YYYY-MM-DD}.log",
            rotation="00:00",
            retention="30 days",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG"
        )
        
        # File logging - Predictions only
        logger.add(
            "logs/predictions_{time:YYYY-MM-DD}.log",
            rotation="00:00",
            retention="30 days",
            format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
            level="INFO",
            filter=lambda record: "PREDICTION" in record["message"] or "VALUE_BET" in record["message"]
        )
        
        # File logging - Errors only
        logger.add(
            "logs/errors_{time:YYYY-MM-DD}.log",
            rotation="00:00",
            retention="30 days",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="ERROR"
        )
        
        # File logging - API calls
        logger.add(
            "logs/api_{time:YYYY-MM-DD}.log",
            rotation="00:00",
            retention="30 days",
            format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
            level="DEBUG",
            filter=lambda record: "API call" in record["message"]
        )
        
        logger.info("Logging system initialized")
    
    def load_models(self):
        """Load trained ML models"""
        try:
            with model_lock:
                if trainer.load_models():
                    self.models_loaded = True
                    logger.success("Models loaded successfully")
                    return True
                else:
                    logger.warning("No models found, training required")
                    return False
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def fetch_api_data(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Fetch data from API-Football with caching"""
        try:
            # Create cache key
            cache_key = f"{endpoint}:{json.dumps(params, sort_keys=True) if params else 'no_params'}"
            
            # Check cache first
            cached_data = self.api_cache.get(cache_key)
            if cached_data:
                return cached_data
            
            url = f"{self.api_base}/{endpoint}"
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            self.api_call_count += 1
            
            if response.status_code == 200:
                data = response.json()
                if data.get('response'):
                    logger.debug(f"API call successful: {endpoint}")
                    # Cache the response
                    self.api_cache.set(cache_key, data['response'])
                    return data['response']
                else:
                    logger.warning(f"No data returned from API: {endpoint}")
                    return None
            elif response.status_code == 429:
                logger.warning("API rate limit reached, waiting 60 seconds...")
                time.sleep(60)
                # Retry once after waiting
                return self.fetch_api_data(endpoint, params)
            else:
                logger.error(f"API error {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            logger.error(f"API timeout for endpoint: {endpoint}")
            return None
        except Exception as e:
            logger.error(f"Error fetching API data: {e}")
            return None
    
    def fetch_upcoming_matches(self, hours_ahead: int = 48) -> List[Dict]:
        """Fetch upcoming matches"""
        try:
            today = datetime.now()
            from_date = today.strftime('%Y-%m-%d')
            to_date = (today + timedelta(hours=hours_ahead)).strftime('%Y-%m-%d')
            
            params = {
                'date': f'{from_date}-{to_date}',
                'timezone': 'UTC'
            }
            
            matches = self.fetch_api_data('fixtures', params)
            if matches:
                logger.info(f"Found {len(matches)} upcoming matches")
                return matches
            return []
            
        except Exception as e:
            logger.error(f"Error fetching upcoming matches: {e}")
            return []
    
    def _extract_odds_from_bookmaker(self, bookmaker: Dict) -> Dict:
        """Extract odds from bookmaker data"""
        odds = {}
        
        for bet in bookmaker['bets']:
            if bet['name'] == 'Match Winner':
                for outcome in bet['values']:
                    if outcome['value'] == 'Home':
                        odds['home_odds'] = float(outcome['odd'])
                    elif outcome['value'] == 'Draw':
                        odds['draw_odds'] = float(outcome['odd'])
                    elif outcome['value'] == 'Away':
                        odds['away_odds'] = float(outcome['odd'])
            
            elif bet['name'] == 'Over/Under':
                if bet['id'] == 5:  # Over/Under 2.5
                    for outcome in bet['values']:
                        if outcome['value'] == 'Over 2.5':
                            odds['over_25_odds'] = float(outcome['odd'])
                        elif outcome['value'] == 'Under 2.5':
                            odds['under_25_odds'] = float(outcome['odd'])
            
            elif bet['name'] == 'Both Teams to Score':
                for outcome in bet['values']:
                    if outcome['value'] == 'Yes':
                        odds['btts_yes_odds'] = float(outcome['odd'])
                    elif outcome['value'] == 'No':
                        odds['btts_no_odds'] = float(outcome['odd'])
        
        return odds
    
    def fetch_match_odds(self, fixture_id: int) -> Optional[Dict]:
        """Fetch odds for a specific match"""
        try:
            params = {'fixture': fixture_id}
            odds_data = self.fetch_api_data('odds', params)
            
            if odds_data and len(odds_data) > 0:
                # Try to find the best bookmaker
                best_bookmaker = None
                best_score = 0
                
                for bookmaker_data in odds_data[0].get('bookmakers', []):
                    score = 0
                    bookmaker = bookmaker_data
                    
                    for bet in bookmaker.get('bets', []):
                        if bet['name'] == 'Match Winner':
                            score += 3
                        elif bet['name'] == 'Over/Under' and bet.get('id') == 5:
                            score += 2
                        elif bet['name'] == 'Both Teams to Score':
                            score += 2
                    
                    if score > best_score:
                        best_score = score
                        best_bookmaker = bookmaker
                
                if best_bookmaker:
                    odds = self._extract_odds_from_bookmaker(best_bookmaker)
                    logger.debug(f"Odds fetched for fixture {fixture_id}: {odds}")
                    return odds
                
            return None
            
        except Exception as e:
            logger.error(f"Error fetching odds for fixture {fixture_id}: {e}")
            return None
    
    def fetch_team_statistics(self, team_id: int, league_id: int, season: int) -> Optional[Dict]:
        """Fetch team statistics with caching"""
        try:
            cache_key = f"team_stats:{team_id}:{league_id}:{season}"
            cached_stats = self.api_cache.get(cache_key)
            if cached_stats:
                return cached_stats
            
            params = {
                'team': team_id,
                'league': league_id,
                'season': season
            }
            
            stats = self.fetch_api_data('teams/statistics', params)
            if stats:
                self.api_cache.set(cache_key, stats)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error fetching team stats: {e}")
            return None
    
    def prepare_features(self, match_data: Dict, home_stats: Dict, away_stats: Dict) -> pd.DataFrame:
        """Prepare features for model prediction"""
        try:
            features = {}
            
            # Basic match features
            features['home_team_id'] = match_data['teams']['home']['id']
            features['away_team_id'] = match_data['teams']['away']['id']
            features['league_id'] = match_data['league']['id']
            
            # Team statistics features
            if home_stats and away_stats:
                home_form = home_stats.get('form', '')
                away_form = away_stats.get('form', '')
                
                # Convert form string to numeric (W=3, D=1, L=0)
                def form_to_points(form_str):
                    if not form_str:
                        return 0
                    points = {'W': 3, 'D': 1, 'L': 0}
                    recent_form = form_str[-5:] if len(form_str) >= 5 else form_str
                    return sum(points.get(char, 0) for char in recent_form) / len(recent_form)
                
                features['home_form'] = form_to_points(home_form)
                features['away_form'] = form_to_points(away_form)
                
                # Goal statistics
                home_goals = home_stats.get('goals', {})
                away_goals = away_stats.get('goals', {})
                
                features['home_avg_goals'] = home_goals.get('for', {}).get('average', {}).get('total', 0) or 0
                features['home_avg_conceded'] = home_goals.get('against', {}).get('average', {}).get('total', 0) or 0
                features['away_avg_goals'] = away_goals.get('for', {}).get('average', {}).get('total', 0) or 0
                features['away_avg_conceded'] = away_goals.get('against', {}).get('average', {}).get('total', 0) or 0
                
                # Calculate strengths
                features['home_att_strength'] = features['home_avg_goals']
                features['home_def_strength'] = features['home_avg_conceded']
                features['away_att_strength'] = features['away_avg_goals']
                features['away_def_strength'] = features['away_avg_conceded']
            
            # Create derived features
            features['home_goal_diff'] = features.get('home_avg_goals', 0) - features.get('home_avg_conceded', 0)
            features['away_goal_diff'] = features.get('away_avg_goals', 0) - features.get('away_avg_conceded', 0)
            features['att_strength_diff'] = features.get('home_att_strength', 0) - features.get('away_att_strength', 0)
            features['def_strength_diff'] = features.get('home_def_strength', 0) - features.get('away_def_strength', 0)
            
            # Convert to DataFrame
            df = pd.DataFrame([features])
            
            # Ensure all expected features are present
            expected_features = trainer.features if hasattr(trainer, 'features') else []
            for feature in expected_features:
                if feature not in df.columns:
                    df[feature] = 0
            
            return df
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return pd.DataFrame()
    
    def predict_match(self, match_data: Dict) -> Dict:
        """Generate predictions for all markets"""
        try:
            with model_lock:
                if not self.models_loaded:
                    logger.warning("Models not loaded, attempting to load...")
                    if not self.load_models():
                        return {}
                
                # Fetch team statistics
                league_id = match_data['league']['id']
                season = match_data['league']['season']
                home_team_id = match_data['teams']['home']['id']
                away_team_id = match_data['teams']['away']['id']
                
                home_stats = self.fetch_team_statistics(home_team_id, league_id, season)
                away_stats = self.fetch_team_statistics(away_team_id, league_id, season)
                
                # Prepare features
                features_df = self.prepare_features(match_data, home_stats, away_stats)
                if features_df.empty:
                    return {}
                
                # Generate predictions from all models
                predictions = {}
                
                # 1. Match Result predictions
                if 'result' in trainer.models:
                    result_pred = trainer.models['result'].predict_proba(features_df)[0]
                    # Get class order from model
                    if hasattr(trainer.models['result'], 'classes_'):
                        classes = trainer.models['result'].classes_
                        for i, class_label in enumerate(classes):
                            if class_label == 1:  # Home win
                                predictions['home_win'] = float(result_pred[i])
                            elif class_label == 0:  # Draw
                                predictions['draw'] = float(result_pred[i])
                            elif class_label == -1:  # Away win
                                predictions['away_win'] = float(result_pred[i])
                    else:
                        # Fallback to default ordering
                        predictions['home_win'] = float(result_pred[2]) if len(result_pred) == 3 else 0.33
                        predictions['draw'] = float(result_pred[1]) if len(result_pred) == 3 else 0.33
                        predictions['away_win'] = float(result_pred[0]) if len(result_pred) == 3 else 0.33
                
                # 2. Over/Under predictions
                if 'over_under' in trainer.models:
                    ou_pred = trainer.models['over_under'].predict_proba(features_df)[0]
                    if len(ou_pred) == 2:
                        predictions['over_25'] = float(ou_pred[1])
                        predictions['under_25'] = float(ou_pred[0])
                    else:
                        predictions['over_25'] = 0.5
                        predictions['under_25'] = 0.5
                
                # 3. BTTS predictions
                if 'btts' in trainer.models:
                    btts_pred = trainer.models['btts'].predict_proba(features_df)[0]
                    if len(btts_pred) == 2:
                        predictions['btts_yes'] = float(btts_pred[1])
                        predictions['btts_no'] = float(btts_pred[0])
                    else:
                        predictions['btts_yes'] = 0.5
                        predictions['btts_no'] = 0.5
                
                # 4. Poisson predictions for exact scores
                if 'poisson' in trainer.models:
                    poisson_model = trainer.models['poisson']
                    lambda_home = poisson_model.get('lambda_home', 1.5)
                    lambda_away = poisson_model.get('lambda_away', 1.2)
                    
                    # Calculate probabilities for common scorelines
                    score_probs = {}
                    for i in range(0, 5):  # Home goals
                        for j in range(0, 5):  # Away goals
                            prob = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)
                            score_probs[f'{i}-{j}'] = float(prob)
                    
                    predictions['score_probs'] = score_probs
                    predictions['expected_goals_home'] = float(lambda_home)
                    predictions['expected_goals_away'] = float(lambda_away)
                
                # Add match metadata
                predictions['fixture_id'] = match_data['fixture']['id']
                predictions['home_team'] = match_data['teams']['home']['name']
                predictions['away_team'] = match_data['teams']['away']['name']
                predictions['league'] = match_data['league']['name']
                predictions['timestamp'] = match_data['fixture']['timestamp']
                
                logger.info(f"PREDICTION: {predictions['home_team']} vs {predictions['away_team']} - Predictions generated")
                
                return predictions
                
        except Exception as e:
            logger.error(f"Error predicting match: {e}")
            return {}
    
    def select_best_market(self, predictions: Dict, odds: Dict) -> Optional[Dict]:
        """Autonomously select the best market based on expected value"""
        try:
            if not predictions or not odds:
                return None
            
            # Calculate expected value for each market
            value_bets = []
            
            market_mapping = {
                'home_win': ('home_odds', predictions.get('home_win', 0)),
                'draw': ('draw_odds', predictions.get('draw', 0)),
                'away_win': ('away_odds', predictions.get('away_win', 0)),
                'over_25': ('over_25_odds', predictions.get('over_25', 0)),
                'under_25': ('under_25_odds', predictions.get('under_25', 0)),
                'btts_yes': ('btts_yes_odds', predictions.get('btts_yes', 0)),
                'btts_no': ('btts_no_odds', predictions.get('btts_no', 0))
            }
            
            for market, (odds_key, pred_prob) in market_mapping.items():
                if odds_key in odds and odds[odds_key] and odds[odds_key] > 0 and pred_prob > 0:
                    # Calculate expected value
                    ev = (pred_prob * odds[odds_key]) - 1
                    
                    # Calculate Kelly Criterion fraction
                    if odds[odds_key] > 1:
                        kelly_fraction = (pred_prob * odds[odds_key] - 1) / (odds[odds_key] - 1)
                    else:
                        kelly_fraction = 0
                    
                    if ev > 0.05:  # Minimum 5% EV threshold
                        value_bet = {
                            'market': market,
                            'predicted_probability': round(pred_prob, 3),
                            'implied_probability': round(1 / odds[odds_key], 3),
                            'odds': round(odds[odds_key], 2),
                            'expected_value': round(ev, 3),
                            'kelly_fraction': round(kelly_fraction, 3),
                            'edge': round(pred_prob - (1 / odds[odds_key]), 3)
                        }
                        value_bets.append(value_bet)
            
            # Sort by expected value (descending)
            value_bets.sort(key=lambda x: x['expected_value'], reverse=True)
            
            if value_bets:
                best_bet = value_bets[0]
                logger.info(f"VALUE_BET: Selected {best_bet['market']} with EV: {best_bet['expected_value']}")
                return best_bet
            
            return None
            
        except Exception as e:
            logger.error(f"Error selecting best market: {e}")
            return None
    
    def send_telegram_alert(self, match_info: Dict, value_bet: Dict):
        """Send value bet alert to Telegram"""
        try:
            if not self.telegram_bot_token or not self.telegram_chat_id:
                logger.warning("Telegram credentials not configured")
                return
            
            message = f"""
⚽ *VALUE BET ALERT* ⚽

*Match:* {match_info['home_team']} vs {match_info['away_team']}
*League:* {match_info['league']}
*Time:* {datetime.fromtimestamp(match_info['timestamp']).strftime('%Y-%m-%d %H:%M')}

*Recommended Bet:* {value_bet['market'].replace('_', ' ').title()}
*Odds:* {value_bet['odds']}
*Predicted Probability:* {value_bet['predicted_probability']*100:.1f}%
*Implied Probability:* {value_bet['implied_probability']*100:.1f}%
*Edge:* {value_bet['edge']*100:.1f}%
*Expected Value:* {value_bet['expected_value']*100:.1f}%
*Kelly Fraction:* {value_bet['kelly_fraction']*100:.1f}%

✅ *Confidence:* {'High' if value_bet['expected_value'] > 0.15 else 'Medium' if value_bet['expected_value'] > 0.08 else 'Low'}
            """
            
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            payload = {
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                logger.success(f"Telegram alert sent for {match_info['home_team']} vs {match_info['away_team']}")
            else:
                logger.error(f"Failed to send Telegram alert: {response.text}")
                
        except Exception as e:
            logger.error(f"Error sending Telegram alert: {e}")
    
    def scan_upcoming_matches(self):
        """Scan all upcoming matches for value bets"""
        try:
            logger.info("Starting live scan of upcoming matches...")
            
            # Fetch upcoming matches
            matches = self.fetch_upcoming_matches(hours_ahead=72)
            if not matches:
                logger.warning("No upcoming matches found")
                return
            
            value_bets_found = 0
            
            for match in matches:
                try:
                    # Fetch odds
                    fixture_id = match['fixture']['id']
                    odds = self.fetch_match_odds(fixture_id)
                    
                    if not odds:
                        continue
                    
                    # Generate predictions
                    predictions = self.predict_match(match)
                    if not predictions:
                        continue
                    
                    # Select best market
                    value_bet = self.select_best_market(predictions, odds)
                    
                    if value_bet:
                        # Prepare match info
                        match_info = {
                            'fixture_id': fixture_id,
                            'home_team': match['teams']['home']['name'],
                            'away_team': match['teams']['away']['name'],
                            'league': match['league']['name'],
                            'timestamp': match['fixture']['timestamp']
                        }
                        
                        # Save to database
                        self.save_value_bet(match_info, value_bet, predictions)
                        
                        # Send Telegram alert
                        self.send_telegram_alert(match_info, value_bet)
                        
                        value_bets_found += 1
                        
                        # Avoid rate limiting
                        time.sleep(1.5)
                        
                except Exception as e:
                    logger.error(f"Error processing match {match.get('fixture', {}).get('id')}: {e}")
                    continue
            
            logger.info(f"Live scan complete. Found {value_bets_found} value bets.")
            
        except Exception as e:
            logger.error(f"Error in live scan: {e}")
    
    def save_value_bet(self, match_info: Dict, value_bet: Dict, predictions: Dict):
        """Save value bet to database"""
        try:
            query = """
            INSERT INTO value_bets 
            (fixture_id, home_team, away_team, league, match_time, 
             market, odds, predicted_probability, implied_probability, 
             expected_value, kelly_fraction, edge, confidence, 
             all_predictions, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            """
            
            params = (
                match_info['fixture_id'],
                match_info['home_team'],
                match_info['away_team'],
                match_info['league'],
                datetime.fromtimestamp(match_info['timestamp']),
                value_bet['market'],
                value_bet['odds'],
                value_bet['predicted_probability'],
                value_bet['implied_probability'],
                value_bet['expected_value'],
                value_bet['kelly_fraction'],
                value_bet['edge'],
                'High' if value_bet['expected_value'] > 0.15 else 'Medium' if value_bet['expected_value'] > 0.08 else 'Low',
                json.dumps(predictions)
            )
            
            db.execute_query(query, params, fetch=False)
            logger.debug(f"Value bet saved for fixture {match_info['fixture_id']}")
            
        except Exception as e:
            logger.error(f"Error saving value bet: {e}")
    
    def generate_daily_digest(self):
        """Generate daily performance digest"""
        try:
            logger.info("Generating daily digest...")
            
            # Get yesterday's date
            yesterday = datetime.now() - timedelta(days=1)
            start_date = yesterday.replace(hour=0, minute=0, second=0)
            end_date = yesterday.replace(hour=23, minute=59, second=59)
            
            # Query performance metrics
            query = """
            SELECT 
                COUNT(*) as total_bets,
                SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN result = 'LOSS' THEN 1 ELSE 0 END) as losses,
                AVG(expected_value) as avg_ev,
                SUM(profit_loss) as total_pnl,
                AVG(odds) as avg_odds
            FROM bet_results
            WHERE bet_date BETWEEN %s AND %s
            """
            
            result = db.execute_query(query, (start_date, end_date))
            
            if not result.empty:
                metrics = result.iloc[0]
                total_bets = int(metrics['total_bets'] or 0)
                wins = int(metrics['wins'] or 0)
                losses = int(metrics['losses'] or 0)
                
                digest = {
                    'date': yesterday.strftime('%Y-%m-%d'),
                    'total_bets': total_bets,
                    'wins': wins,
                    'losses': losses,
                    'win_rate': round(wins / total_bets * 100, 2) if total_bets > 0 else 0,
                    'avg_ev': round(float(metrics['avg_ev'] or 0), 3),
                    'total_pnl': round(float(metrics['total_pnl'] or 0), 2),
                    'roi': round(float(metrics['total_pnl'] or 0) / total_bets * 100, 2) if total_bets > 0 else 0
                }
                
                # Save digest
                self.save_daily_digest(digest)
                
                # Send to Telegram
                self.send_daily_digest(digest)
                
                logger.info(f"Daily digest generated: {digest}")
                
            else:
                logger.info("No bets found for yesterday")
                
        except Exception as e:
            logger.error(f"Error generating daily digest: {e}")
    
    def save_daily_digest(self, digest: Dict):
        """Save daily digest to database"""
        try:
            query = """
            INSERT INTO daily_digests 
            (date, total_bets, wins, losses, win_rate, avg_ev, total_pnl, roi, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (date) DO UPDATE SET
                total_bets = EXCLUDED.total_bets,
                wins = EXCLUDED.wins,
                losses = EXCLUDED.losses,
                win_rate = EXCLUDED.win_rate,
                avg_ev = EXCLUDED.avg_ev,
                total_pnl = EXCLUDED.total_pnl,
                roi = EXCLUDED.roi,
                created_at = NOW()
            """
            
            params = (
                digest['date'],
                digest['total_bets'],
                digest['wins'],
                digest['losses'],
                digest['win_rate'],
                digest['avg_ev'],
                digest['total_pnl'],
                digest['roi']
            )
            
            db.execute_query(query, params, fetch=False)
            
        except Exception as e:
            logger.error(f"Error saving daily digest: {e}")
    
    def send_daily_digest(self, digest: Dict):
        """Send daily digest to Telegram"""
        try:
            if not self.telegram_bot_token or not self.telegram_chat_id:
                return
            
            message = f"""
📊 *DAILY PERFORMANCE DIGEST* 📊

*Date:* {digest['date']}

*Statistics:*
• Total Bets: {digest['total_bets']}
• Wins: {digest['wins']}
• Losses: {digest['losses']}
• Win Rate: {digest['win_rate']}%
• Avg EV: {digest['avg_ev']}
• Total P&L: ${digest['total_pnl']}
• ROI: {digest['roi']}%

{self.get_performance_emoji(digest['roi'])}
            """
            
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            payload = {
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            requests.post(url, json=payload, timeout=10)
            
        except Exception as e:
            logger.error(f"Error sending daily digest: {e}")
    
    def get_performance_emoji(self, roi: float) -> str:
        """Get performance emoji based on ROI"""
        if roi > 10:
            return "🎯🔥 EXCELLENT DAY! 🔥🎯"
        elif roi > 5:
            return "✅ Great performance!"
        elif roi > 0:
            return "↗️ Positive day"
        elif roi > -5:
            return "⚠️ Small loss"
        else:
            return "🔻 Review needed"
    
    def backfill_historical_data(self, days: int = 365):
        """Backfill historical match data"""
        try:
            if days < 1 or days > 730:
                logger.error(f"Invalid days parameter: {days}. Must be between 1 and 730.")
                return
            
            logger.info(f"Starting backfill for last {days} days...")
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            current_date = start_date
            matches_processed = 0
            
            while current_date <= end_date:
                try:
                    date_str = current_date.strftime('%Y-%m-%d')
                    logger.info(f"Backfilling data for {date_str}...")
                    
                    params = {
                        'date': date_str,
                        'timezone': 'UTC'
                    }
                    
                    matches = self.fetch_api_data('fixtures', params)
                    
                    if matches:
                        for match in matches:
                            try:
                                # Save match data
                                self.save_match_data(match)
                                matches_processed += 1
                                
                                # Fetch and save odds if match hasn't started
                                if match['fixture']['status']['short'] not in ['FT', 'AET', 'PEN']:
                                    odds = self.fetch_match_odds(match['fixture']['id'])
                                    if odds:
                                        self.save_odds_data(match['fixture']['id'], odds)
                                
                            except Exception as e:
                                logger.error(f"Error processing match {match.get('fixture', {}).get('id')}: {e}")
                                continue
                    
                    # Avoid rate limiting
                    time.sleep(2.5)
                    current_date += timedelta(days=1)
                    
                except Exception as e:
                    logger.error(f"Error backfilling date {current_date}: {e}")
                    continue
            
            logger.success(f"Backfill complete. Processed {matches_processed} matches.")
            
        except Exception as e:
            logger.error(f"Error in backfill: {e}")
    
    def save_match_data(self, match: Dict):
        """Save match data to database"""
        try:
            query = """
            INSERT INTO matches 
            (fixture_id, league_id, league_name, season, round,
             home_team_id, home_team, away_team_id, away_team,
             goals_home, goals_away, status, timestamp, venue,
             referee, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
            ON CONFLICT (fixture_id) 
            DO UPDATE SET
                goals_home = EXCLUDED.goals_home,
                goals_away = EXCLUDED.goals_away,
                status = EXCLUDED.status,
                updated_at = NOW()
            """
            
            fixture = match['fixture']
            league = match['league']
            teams = match['teams']
            goals = match['goals']
            
            params = (
                fixture['id'],
                league['id'],
                league['name'],
                league['season'],
                match.get('league', {}).get('round', ''),
                teams['home']['id'],
                teams['home']['name'],
                teams['away']['id'],
                teams['away']['name'],
                goals.get('home', 0),
                goals.get('away', 0),
                fixture['status']['short'],
                fixture['timestamp'],
                fixture.get('venue', {}).get('name', ''),
                fixture.get('referee', '')
            )
            
            db.execute_query(query, params, fetch=False)
            
        except Exception as e:
            logger.error(f"Error saving match data: {e}")
    
    def save_odds_data(self, fixture_id: int, odds: Dict):
        """Save odds data to database"""
        try:
            query = """
            INSERT INTO odds 
            (fixture_id, home_odds, draw_odds, away_odds,
             over_25_odds, under_25_odds, btts_yes_odds, btts_no_odds,
             timestamp, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (fixture_id) 
            DO UPDATE SET
                home_odds = EXCLUDED.home_odds,
                draw_odds = EXCLUDED.draw_odds,
                away_odds = EXCLUDED.away_odds,
                over_25_odds = EXCLUDED.over_25_odds,
                under_25_odds = EXCLUDED.under_25_odds,
                btts_yes_odds = EXCLUDED.btts_yes_odds,
                btts_no_odds = EXCLUDED.btts_no_odds,
                timestamp = EXCLUDED.timestamp
            """
            
            params = (
                fixture_id,
                odds.get('home_odds'),
                odds.get('draw_odds'),
                odds.get('away_odds'),
                odds.get('over_25_odds'),
                odds.get('under_25_odds'),
                odds.get('btts_yes_odds'),
                odds.get('btts_no_odds'),
                datetime.now()
            )
            
            db.execute_query(query, params, fetch=False)
            
        except Exception as e:
            logger.error(f"Error saving odds data: {e}")
    
    def health_check(self) -> Dict:
        """Perform system health check"""
        try:
            health = {
                'timestamp': datetime.now().isoformat(),
                'status': 'healthy',
                'components': {}
            }
            
            # Database health
            try:
                db_result = db.execute_query("SELECT 1 as health", fetch=True)
                health['components']['database'] = 'healthy' if not db_result.empty else 'unhealthy'
            except Exception as e:
                logger.error(f"Database health check failed: {e}")
                health['components']['database'] = 'unhealthy'
            
            # API health
            try:
                api_test = self.fetch_api_data('status')
                health['components']['api_football'] = 'healthy' if api_test else 'unhealthy'
            except Exception as e:
                logger.error(f"API health check failed: {e}")
                health['components']['api_football'] = 'unhealthy'
            
            # Models health
            health['components']['models'] = 'loaded' if self.models_loaded else 'not_loaded'
            
            # Disk space
            try:
                if hasattr(os, 'statvfs'):
                    statvfs = os.statvfs('/')
                    free_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)
                    health['components']['disk_space'] = f'{free_gb:.1f} GB free'
                else:
                    import shutil
                    total, used, free = shutil.disk_usage("/")
                    health['components']['disk_space'] = f'{free / (1024**3):.1f} GB free'
            except Exception as e:
                logger.error(f"Disk space check failed: {e}")
                health['components']['disk_space'] = 'unknown'
            
            # Memory usage
            try:
                memory = psutil.virtual_memory()
                health['components']['memory'] = f'{memory.percent}% used'
            except Exception as e:
                logger.error(f"Memory check failed: {e}")
                health['components']['memory'] = 'unknown'
            
            # API call count
            health['components']['api_calls_today'] = self.api_call_count
            
            # Check if any component is unhealthy
            if any(status == 'unhealthy' for status in health['components'].values()):
                health['status'] = 'degraded'
            
            return health
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'status': 'unhealthy',
                'error': str(e)
            }

# Initialize predictor
predictor = FootballPredictor()

# Flask Error Handlers
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({
        'status': 'error',
        'message': 'Resource not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Server Error: {error}")
    return jsonify({
        'status': 'error',
        'message': 'Internal server error'
    }), 500

@app.errorhandler(429)
def rate_limit_error(error):
    return jsonify({
        'status': 'error',
        'message': 'Rate limit exceeded'
    }), 429

@app.errorhandler(400)
def bad_request_error(error):
    return jsonify({
        'status': 'error',
        'message': 'Bad request'
    }), 400

# Flask Routes
@app.route('/')
def index():
    return jsonify({
        'message': 'Football Prediction System API',
        'version': '1.0.0',
        'endpoints': {
            '/live_scan': 'Trigger live match scan',
            '/train': 'Trigger model training',
            '/daily_digest': 'Generate daily performance report',
            '/auto_tune': 'Auto-tune model parameters',
            '/backfill': 'Backfill historical data',
            '/health': 'System health check',
            '/metrics': 'System metrics',
            '/predict/<fixture_id>': 'Predict specific match',
            '/cache/clear': 'Clear API cache'
        }
    })

@app.route('/live_scan', methods=['GET'])
def trigger_live_scan():
    """Trigger live match scanning"""
    try:
        Thread(target=predictor.scan_upcoming_matches, daemon=True).start()
        logger.info("Live scan triggered via API")
        return jsonify({
            'status': 'success',
            'message': 'Live scan started in background'
        })
    except Exception as e:
        logger.error(f"Error triggering live scan: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/train', methods=['GET'])
def trigger_training():
    """Trigger model training"""
    try:
        def train_models():
            trainer.setup_logger()
            success = trainer.train_all_models()
            if success:
                predictor.load_models()
        
        Thread(target=train_models, daemon=True).start()
        logger.info("Model training triggered via API")
        
        return jsonify({
            'status': 'success',
            'message': 'Model training started in background'
        })
    except Exception as e:
        logger.error(f"Error triggering training: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/daily_digest', methods=['GET'])
def trigger_daily_digest():
    """Generate daily performance digest"""
    try:
        Thread(target=predictor.generate_daily_digest, daemon=True).start()
        logger.info("Daily digest triggered via API")
        
        return jsonify({
            'status': 'success',
            'message': 'Daily digest generation started'
        })
    except Exception as e:
        logger.error(f"Error triggering daily digest: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/auto_tune', methods=['GET'])
def trigger_auto_tune():
    """Trigger model auto-tuning"""
    try:
        Thread(target=trainer.train_all_models, daemon=True).start()
        logger.info("Auto-tuning triggered via API")
        
        return jsonify({
            'status': 'success',
            'message': 'Auto-tuning started in background'
        })
    except Exception as e:
        logger.error(f"Error triggering auto-tune: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/backfill', methods=['GET'])
def trigger_backfill():
    """Trigger historical data backfill"""
    try:
        days = request.args.get('days', default=30, type=int)
        
        # Validate input
        if days < 1 or days > 730:
            return jsonify({
                'status': 'error',
                'message': 'Days must be between 1 and 730'
            }), 400
            
        Thread(target=predictor.backfill_historical_data, args=(days,), daemon=True).start()
        logger.info(f"Backfill triggered for {days} days via API")
        
        return jsonify({
            'status': 'success',
            'message': f'Backfill started for {days} days in background'
        })
    except Exception as e:
        logger.error(f"Error triggering backfill: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """System health check endpoint"""
    try:
        health = predictor.health_check()
        return jsonify(health)
    except Exception as e:
        logger.error(f"Health check endpoint error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/metrics', methods=['GET'])
def system_metrics():
    """System metrics endpoint"""
    try:
        import psutil
        
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        metrics_data = {
            'system': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_gb': round(memory.used / (1024**3), 2),
                'memory_total_gb': round(memory.total / (1024**3), 2),
                'disk_percent': disk.percent,
                'disk_free_gb': round(disk.free / (1024**3), 2)
            },
            'application': {
                'models_loaded': predictor.models_loaded,
                'api_calls_today': predictor.api_call_count,
                'cache_size': len(predictor.api_cache.cache)
            }
        }
        
        return jsonify(metrics_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/cache/clear', methods=['POST'])
def clear_cache():
    """Clear API cache"""
    try:
        predictor.api_cache.clear()
        return jsonify({
            'status': 'success',
            'message': 'API cache cleared'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/<int:fixture_id>', methods=['GET'])
def predict_match(fixture_id):
    """Predict specific match"""
    try:
        # Validate fixture_id
        if fixture_id <= 0:
            return jsonify({
                'status': 'error',
                'message': 'Invalid fixture ID'
            }), 400
        
        # Fetch match data
        params = {'id': fixture_id}
        match_data = predictor.fetch_api_data('fixtures', params)
        
        if not match_data or len(match_data) == 0:
            return jsonify({
                'status': 'error',
                'message': 'Match not found'
            }), 404
        
        match = match_data[0]
        
        # Generate predictions
        predictions = predictor.predict_match(match)
        
        if not predictions:
            return jsonify({
                'status': 'error',
                'message': 'Failed to generate predictions'
            }), 500
        
        # Fetch odds
        odds = predictor.fetch_match_odds(fixture_id)
        
        # Select best market
        value_bet = predictor.select_best_market(predictions, odds) if odds else None
        
        response = {
            'status': 'success',
            'match': {
                'fixture_id': fixture_id,
                'home_team': match['teams']['home']['name'],
                'away_team': match['teams']['away']['name'],
                'league': match['league']['name'],
                'timestamp': match['fixture']['timestamp'],
                'date': datetime.fromtimestamp(match['fixture']['timestamp']).strftime('%Y-%m-%d %H:%M')
            },
            'predictions': predictions,
            'odds': odds,
            'value_bet': value_bet
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error predicting match {fixture_id}: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

def setup_scheduler():
    """Setup scheduled tasks"""
    try:
        # Schedule daily digest at 9 AM
        scheduler.add_job(
            func=predictor.generate_daily_digest,
            trigger='cron',
            hour=9,
            minute=0,
            id='daily_digest'
        )
        
        # Schedule model retraining every Sunday at 2 AM
        scheduler.add_job(
            func=trainer.train_all_models,
            trigger='cron',
            day_of_week='sun',
            hour=2,
            minute=0,
            id='weekly_training'
        )
        
        # Schedule live scan every 2 hours
        scheduler.add_job(
            func=predictor.scan_upcoming_matches,
            trigger='interval',
            hours=2,
            id='live_scan'
        )
        
        # Schedule cache clearing at midnight
        scheduler.add_job(
            func=predictor.api_cache.clear,
            trigger='cron',
            hour=0,
            minute=0,
            id='clear_cache'
        )
        
        scheduler.start()
        logger.info("Scheduler started")
        
    except Exception as e:
        logger.error(f"Error setting up scheduler: {e}")

def initialize_system():
    """Initialize the prediction system"""
    try:
        logger.info("Initializing Football Prediction System...")
        
        # Load models
        predictor.load_models()
        
        # Setup scheduler
        setup_scheduler()
        
        # Initial health check
        health = predictor.health_check()
        logger.info(f"System initialized. Health: {health['status']}")
        
    except Exception as e:
        logger.error(f"System initialization failed: {e}")

# Create necessary database tables on startup
def create_tables():
    """Create necessary database tables"""
    try:
        # First create tables without indexes
        table_sqls = [
            """
            CREATE TABLE IF NOT EXISTS matches (
                id SERIAL PRIMARY KEY,
                fixture_id INTEGER UNIQUE,
                league_id INTEGER,
                league_name VARCHAR(100),
                season INTEGER,
                round VARCHAR(50),
                home_team_id INTEGER,
                home_team VARCHAR(100),
                away_team_id INTEGER,
                away_team VARCHAR(100),
                goals_home INTEGER DEFAULT 0,
                goals_away INTEGER DEFAULT 0,
                status VARCHAR(20),
                timestamp BIGINT,
                venue VARCHAR(200),
                referee VARCHAR(100),
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS odds (
                id SERIAL PRIMARY KEY,
                fixture_id INTEGER UNIQUE,
                home_odds DECIMAL(6,2),
                draw_odds DECIMAL(6,2),
                away_odds DECIMAL(6,2),
                over_25_odds DECIMAL(6,2),
                under_25_odds DECIMAL(6,2),
                btts_yes_odds DECIMAL(6,2),
                btts_no_odds DECIMAL(6,2),
                timestamp TIMESTAMP,
                created_at TIMESTAMP DEFAULT NOW(),
                FOREIGN KEY (fixture_id) REFERENCES matches(fixture_id) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS value_bets (
                id SERIAL PRIMARY KEY,
                fixture_id INTEGER,
                home_team VARCHAR(100),
                away_team VARCHAR(100),
                league VARCHAR(100),
                match_time TIMESTAMP,
                market VARCHAR(50),
                odds DECIMAL(6,2),
                predicted_probability DECIMAL(5,3),
                implied_probability DECIMAL(5,3),
                expected_value DECIMAL(5,3),
                kelly_fraction DECIMAL(5,3),
                edge DECIMAL(5,3),
                confidence VARCHAR(20),
                all_predictions JSONB,
                created_at TIMESTAMP DEFAULT NOW(),
                FOREIGN KEY (fixture_id) REFERENCES matches(fixture_id) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS bet_results (
                id SERIAL PRIMARY KEY,
                value_bet_id INTEGER REFERENCES value_bets(id) ON DELETE CASCADE,
                fixture_id INTEGER,
                market VARCHAR(50),
                odds DECIMAL(6,2),
                stake DECIMAL(8,2),
                result VARCHAR(10),
                profit_loss DECIMAL(8,2),
                bet_date TIMESTAMP,
                settled_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT NOW()
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS daily_digests (
                id SERIAL PRIMARY KEY,
                date DATE UNIQUE,
                total_bets INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                win_rate DECIMAL(5,2) DEFAULT 0,
                avg_ev DECIMAL(5,3) DEFAULT 0,
                total_pnl DECIMAL(10,2) DEFAULT 0,
                roi DECIMAL(5,2) DEFAULT 0,
                created_at TIMESTAMP DEFAULT NOW()
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS team_stats (
                id SERIAL PRIMARY KEY,
                team_id INTEGER,
                league_id INTEGER,
                season INTEGER,
                home_avg_goals DECIMAL(4,2) DEFAULT 0,
                home_avg_conceded DECIMAL(4,2) DEFAULT 0,
                home_form VARCHAR(10),
                home_att_strength DECIMAL(4,2) DEFAULT 0,
                home_def_strength DECIMAL(4,2) DEFAULT 0,
                away_avg_goals DECIMAL(4,2) DEFAULT 0,
                away_avg_conceded DECIMAL(4,2) DEFAULT 0,
                away_form VARCHAR(10),
                away_att_strength DECIMAL(4,2) DEFAULT 0,
                away_def_strength DECIMAL(4,2) DEFAULT 0,
                updated_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(team_id, league_id, season)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS head_to_head (
                id SERIAL PRIMARY KEY,
                home_team_id INTEGER,
                away_team_id INTEGER,
                total_matches INTEGER DEFAULT 0,
                home_wins INTEGER DEFAULT 0,
                away_wins INTEGER DEFAULT 0,
                draws INTEGER DEFAULT 0,
                last_updated TIMESTAMP DEFAULT NOW(),
                UNIQUE(home_team_id, away_team_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS system_logs (
                id SERIAL PRIMARY KEY,
                level VARCHAR(20),
                message TEXT,
                module VARCHAR(100),
                function VARCHAR(100),
                created_at TIMESTAMP DEFAULT NOW()
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS model_versions (
                id SERIAL PRIMARY KEY,
                version VARCHAR(50),
                model_type VARCHAR(50),
                accuracy DECIMAL(5,3),
                precision DECIMAL(5,3),
                recall DECIMAL(5,3),
                f1_score DECIMAL(5,3),
                features_count INTEGER,
                training_date DATE,
                created_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(version, model_type)
            )
            """
        ]
        
        # Create indexes separately
        index_sqls = [
            # Indexes for matches table
            "CREATE INDEX IF NOT EXISTS idx_matches_fixture_id ON matches (fixture_id);",
            "CREATE INDEX IF NOT EXISTS idx_matches_league ON matches (league_id, season);",
            "CREATE INDEX IF NOT EXISTS idx_matches_teams ON matches (home_team_id, away_team_id);",
            "CREATE INDEX IF NOT EXISTS idx_matches_timestamp ON matches (timestamp);",
            
            # Indexes for odds table
            "CREATE INDEX IF NOT EXISTS idx_odds_fixture ON odds (fixture_id);",
            "CREATE INDEX IF NOT EXISTS idx_odds_timestamp ON odds (timestamp);",
            
            # Indexes for value_bets table
            "CREATE INDEX IF NOT EXISTS idx_value_bets_fixture ON value_bets (fixture_id);",
            "CREATE INDEX IF NOT EXISTS idx_value_bets_created ON value_bets (created_at);",
            "CREATE INDEX IF NOT EXISTS idx_value_bets_match_time ON value_bets (match_time);",
            "CREATE INDEX IF NOT EXISTS idx_value_bets_league ON value_bets (league);",
            
            # Indexes for bet_results table
            "CREATE INDEX IF NOT EXISTS idx_bet_results_date ON bet_results (bet_date);",
            "CREATE INDEX IF NOT EXISTS idx_bet_results_fixture ON bet_results (fixture_id);",
            "CREATE INDEX IF NOT EXISTS idx_bet_results_result ON bet_results (result);",
            
            # Indexes for daily_digests table
            "CREATE INDEX IF NOT EXISTS idx_daily_digests_date ON daily_digests (date);",
            
            # Indexes for team_stats table
            "CREATE INDEX IF NOT EXISTS idx_team_stats_team ON team_stats (team_id);",
            "CREATE INDEX IF NOT EXISTS idx_team_stats_league ON team_stats (league_id, season);",
            
            # Indexes for head_to_head table
            "CREATE INDEX IF NOT EXISTS idx_h2h_teams ON head_to_head (home_team_id, away_team_id);",
            
            # Indexes for system_logs table
            "CREATE INDEX IF NOT EXISTS idx_system_logs_created ON system_logs (created_at);",
            "CREATE INDEX IF NOT EXISTS idx_system_logs_level ON system_logs (level);",
            
            # Indexes for model_versions table
            "CREATE INDEX IF NOT EXISTS idx_model_versions_date ON model_versions (training_date);",
            "CREATE INDEX IF NOT EXISTS idx_model_versions_type ON model_versions (model_type);"
        ]
        
        # Create tables first
        for table_sql in table_sqls:
            db.execute_query(table_sql, fetch=False)
        
        # Then create indexes
        for index_sql in index_sqls:
            try:
                db.execute_query(index_sql, fetch=False)
            except Exception as e:
                logger.warning(f"Could not create index, may already exist: {e}")
        
        logger.success("Database tables and indexes created/verified")
        
    except Exception as e:
        logger.error(f"Error creating tables: {e}")
        # Don't raise to allow system to continue with existing tables

if __name__ == '__main__':
    # Initialize database
    create_tables()
    
    # Initialize system
    initialize_system()
    
    # Run Flask app
    port = int(os.getenv('PORT', 8080))
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
