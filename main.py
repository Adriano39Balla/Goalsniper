#!/usr/bin/env python3
"""
Football Prediction System Backend
Autonomous market selection with extensive logging
"""
import os
import json
import time
from datetime import datetime, timedelta
from threading import Thread
from typing import Dict, List, Optional, Tuple

from flask import Flask, request, jsonify
from flask_cors import CORS
from loguru import logger
from apscheduler.schedulers.background import BackgroundScheduler
import pandas as pd
import numpy as np
import requests
import joblib
import pickle

# Import modules
from database import DatabaseManager
from train_models import ModelTrainer

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables
db = DatabaseManager()
trainer = ModelTrainer()
scheduler = BackgroundScheduler()

class FootballPredictor:
    def __init__(self):
        self.api_key = os.getenv('API_FOOTBALL_KEY')
        self.api_base = "https://v3.football.api-sports.io"
        self.headers = {
            'x-apisports-key': self.api_key,
            'x-rapidapi-host': 'v3.football.api-sports.io'
        }
        self.telegram_bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.models_loaded = False
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
        
        logger.info("Logging system initialized")
    
    def load_models(self):
        """Load trained ML models"""
        try:
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
        """Fetch data from API-Football"""
        try:
            url = f"{self.api_base}/{endpoint}"
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('response'):
                    logger.debug(f"API call successful: {endpoint}")
                    return data['response']
                else:
                    logger.warning(f"No data returned from API: {endpoint}")
                    return None
            elif response.status_code == 429:
                logger.warning("API rate limit reached, waiting...")
                time.sleep(60)
                return self.fetch_api_data(endpoint, params)
            else:
                logger.error(f"API error {response.status_code}: {response.text}")
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
    
    def fetch_match_odds(self, fixture_id: int) -> Optional[Dict]:
        """Fetch odds for a specific match"""
        try:
            params = {'fixture': fixture_id}
            odds_data = self.fetch_api_data('odds', params)
            
            if odds_data and len(odds_data) > 0:
                # Extract odds from first bookmaker
                bookmaker = odds_data[0]['bookmakers'][0]
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
                
                logger.debug(f"Odds fetched for fixture {fixture_id}: {odds}")
                return odds
                
            return None
            
        except Exception as e:
            logger.error(f"Error fetching odds for fixture {fixture_id}: {e}")
            return None
    
    def fetch_team_statistics(self, team_id: int, league_id: int, season: int) -> Optional[Dict]:
        """Fetch team statistics"""
        try:
            params = {
                'team': team_id,
                'league': league_id,
                'season': season
            }
            
            stats = self.fetch_api_data('teams/statistics', params)
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
                    points = {'W': 3, 'D': 1, 'L': 0}
                    return sum(points.get(char, 0) for char in form_str[-5:]) / 5
                
                features['home_form'] = form_to_points(home_form) if home_form else 0
                features['away_form'] = form_to_points(away_form) if away_form else 0
                
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
                predictions['home_win'] = float(result_pred[2]) if len(result_pred) == 3 else 0.33
                predictions['draw'] = float(result_pred[1]) if len(result_pred) == 3 else 0.33
                predictions['away_win'] = float(result_pred[0]) if len(result_pred) == 3 else 0.33
            
            # 2. Over/Under predictions
            if 'over_under' in trainer.models:
                ou_pred = trainer.models['over_under'].predict_proba(features_df)[0]
                predictions['over_25'] = float(ou_pred[1]) if len(ou_pred) == 2 else 0.5
                predictions['under_25'] = 1 - predictions['over_25']
            
            # 3. BTTS predictions
            if 'btts' in trainer.models:
                btts_pred = trainer.models['btts'].predict_proba(features_df)[0]
                predictions['btts_yes'] = float(btts_pred[1]) if len(btts_pred) == 2 else 0.5
                predictions['btts_no'] = 1 - predictions['btts_yes']
            
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
                if odds_key in odds and odds[odds_key] > 0 and pred_prob > 0:
                    # Calculate expected value
                    ev = (pred_prob * odds[odds_key]) - 1
                    
                    # Calculate Kelly Criterion fraction
                    kelly_fraction = (pred_prob * odds[odds_key] - 1) / (odds[odds_key] - 1) if odds[odds_key] > 1 else 0
                    
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
            
            response = requests.post(url, json=payload)
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
                        time.sleep(1)
                        
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
                
                digest = {
                    'date': yesterday.strftime('%Y-%m-%d'),
                    'total_bets': int(metrics['total_bets']),
                    'wins': int(metrics['wins']),
                    'losses': int(metrics['losses']),
                    'win_rate': round(metrics['wins'] / metrics['total_bets'] * 100, 2) if metrics['total_bets'] > 0 else 0,
                    'avg_ev': round(float(metrics['avg_ev']), 3),
                    'total_pnl': round(float(metrics['total_pnl']), 2),
                    'roi': round(float(metrics['total_pnl']) / metrics['total_bets'] * 100, 2) if metrics['total_bets'] > 0 else 0
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
            
            requests.post(url, json=payload)
            
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
                    time.sleep(2)
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
                goals['home'],
                goals['away'],
                fixture['status']['short'],
                fixture['timestamp'],
                fixture['venue']['name'] if fixture.get('venue') else '',
                fixture['referee'] if fixture.get('referee') else ''
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
                db_result = db.execute_query("SELECT 1 as health")
                health['components']['database'] = 'healthy' if not db_result.empty else 'unhealthy'
            except:
                health['components']['database'] = 'unhealthy'
            
            # API health
            try:
                api_test = self.fetch_api_data('status')
                health['components']['api_football'] = 'healthy' if api_test else 'unhealthy'
            except:
                health['components']['api_football'] = 'unhealthy'
            
            # Models health
            health['components']['models'] = 'loaded' if self.models_loaded else 'not_loaded'
            
            # Disk space (simplified)
            try:
                statvfs = os.statvfs('/')
                free_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)
                health['components']['disk_space'] = f'{free_gb:.1f} GB free'
            except:
                health['components']['disk_space'] = 'unknown'
            
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
            '/predict/<fixture_id>': 'Predict specific match'
        }
    })

@app.route('/live_scan', methods=['GET'])
def trigger_live_scan():
    """Trigger live match scanning"""
    try:
        Thread(target=predictor.scan_upcoming_matches).start()
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
        
        Thread(target=train_models).start()
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
        Thread(target=predictor.generate_daily_digest).start()
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
        # This would implement hyperparameter optimization
        # For now, just trigger a retraining
        Thread(target=trainer.train_all_models).start()
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
        
        Thread(target=predictor.backfill_historical_data, args=(days,)).start()
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

@app.route('/predict/<int:fixture_id>', methods=['GET'])
def predict_match(fixture_id):
    """Predict specific match"""
    try:
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
                'timestamp': match['fixture']['timestamp']
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
        tables = [
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
                goals_home INTEGER,
                goals_away INTEGER,
                status VARCHAR(20),
                timestamp BIGINT,
                venue VARCHAR(200),
                referee VARCHAR(100),
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS odds (
                id SERIAL PRIMARY KEY,
                fixture_id INTEGER UNIQUE REFERENCES matches(fixture_id),
                home_odds DECIMAL(6,2),
                draw_odds DECIMAL(6,2),
                away_odds DECIMAL(6,2),
                over_25_odds DECIMAL(6,2),
                under_25_odds DECIMAL(6,2),
                btts_yes_odds DECIMAL(6,2),
                btts_no_odds DECIMAL(6,2),
                timestamp TIMESTAMP,
                created_at TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS value_bets (
                id SERIAL PRIMARY KEY,
                fixture_id INTEGER REFERENCES matches(fixture_id),
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
                created_at TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS bet_results (
                id SERIAL PRIMARY KEY,
                value_bet_id INTEGER REFERENCES value_bets(id),
                fixture_id INTEGER,
                market VARCHAR(50),
                odds DECIMAL(6,2),
                stake DECIMAL(8,2),
                result VARCHAR(10),
                profit_loss DECIMAL(8,2),
                bet_date TIMESTAMP,
                settled_at TIMESTAMP,
                created_at TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS daily_digests (
                id SERIAL PRIMARY KEY,
                date DATE UNIQUE,
                total_bets INTEGER,
                wins INTEGER,
                losses INTEGER,
                win_rate DECIMAL(5,2),
                avg_ev DECIMAL(5,3),
                total_pnl DECIMAL(10,2),
                roi DECIMAL(5,2),
                created_at TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS team_stats (
                id SERIAL PRIMARY KEY,
                team_id INTEGER,
                league_id INTEGER,
                season INTEGER,
                home_avg_goals DECIMAL(4,2),
                home_avg_conceded DECIMAL(4,2),
                home_form VARCHAR(10),
                home_att_strength DECIMAL(4,2),
                home_def_strength DECIMAL(4,2),
                away_avg_goals DECIMAL(4,2),
                away_avg_conceded DECIMAL(4,2),
                away_form VARCHAR(10),
                away_att_strength DECIMAL(4,2),
                away_def_strength DECIMAL(4,2),
                updated_at TIMESTAMP,
                UNIQUE(team_id, league_id, season)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS head_to_head (
                id SERIAL PRIMARY KEY,
                home_team_id INTEGER,
                away_team_id INTEGER,
                total_matches INTEGER,
                home_wins INTEGER,
                away_wins INTEGER,
                draws INTEGER,
                last_updated TIMESTAMP,
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
            """
        ]
        
        for table_sql in tables:
            db.execute_query(table_sql, fetch=False)
        
        logger.success("Database tables created/verified")
        
    except Exception as e:
        logger.error(f"Error creating tables: {e}")

if __name__ == '__main__':
    # Initialize database
    create_tables()
    
    # Initialize system
    initialize_system()
    
    # Run Flask app
    port = int(os.getenv('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
