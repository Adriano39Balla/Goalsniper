import os
import sys
import time
import schedule
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import psycopg2
from psycopg2.extras import RealDictCursor
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import telegram

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('prediction_engine.log')
    ]
)
logger = logging.getLogger(__name__)

# Configuration
DATABASE_URL = os.getenv('DATABASE_URL')
API_FOOTBALL_KEY = os.getenv('API_FOOTBALL_KEY')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
API_FOOTBALL_HOST = 'v3.football.api-sports.io'

if not all([DATABASE_URL, API_FOOTBALL_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID]):
    logger.error("Missing environment variables. Please check your .env file.")
    sys.exit(1)

# Database connection with proper error handling
def get_db_connection():
    """Get a fresh database connection with proper settings"""
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = True  # Enable autocommit to avoid transaction issues
    return conn

# Initialize database tables
def init_database():
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Event predictions table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS event_predictions (
            id SERIAL PRIMARY KEY,
            fixture_id INTEGER NOT NULL,
            league_id INTEGER NOT NULL,
            minute INTEGER NOT NULL,
            predictions JSONB NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    
    # Market suggestions table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS market_suggestions (
            id SERIAL PRIMARY KEY,
            fixture_id INTEGER NOT NULL,
            market_type VARCHAR(50) NOT NULL,
            probability FLOAT NOT NULL,
            confidence_score FLOAT NOT NULL,
            expected_value FLOAT NOT NULL,
            parameters JSONB NOT NULL,
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    
    # Results tracking table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS market_results (
            id SERIAL PRIMARY KEY,
            market_suggestion_id INTEGER REFERENCES market_suggestions(id),
            outcome VARCHAR(20) NOT NULL,
            actual_probability FLOAT,
            profit_loss FLOAT,
            match_state JSONB NOT NULL,
            analyzed_at TIMESTAMP DEFAULT NOW()
        )
    """)
    
    # League statistics table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS league_stats (
            league_id INTEGER PRIMARY KEY,
            league_name VARCHAR(100) NOT NULL,
            reliability_score FLOAT DEFAULT 0.8,
            total_predictions INTEGER DEFAULT 0,
            correct_predictions INTEGER DEFAULT 0,
            avg_profit_loss FLOAT DEFAULT 0,
            last_updated TIMESTAMP DEFAULT NOW()
        )
    """)
    
    # Create indexes for performance
    cur.execute("CREATE INDEX IF NOT EXISTS idx_market_suggestions_fixture ON market_suggestions(fixture_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_event_predictions_fixture ON event_predictions(fixture_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_market_results_created ON market_results(analyzed_at)")
    
    conn.commit()
    cur.close()
    conn.close()
    logger.info("Database initialized")

# API Football client (Fixed with better error handling)
class APIFootballClient:
    def __init__(self, api_key: str, host: str = API_FOOTBALL_HOST):
        self.api_key = api_key
        self.host = host
        self.base_url = f"https://{host}"
        self.headers = {
            'x-apisports-key': api_key,
            'x-rapidapi-host': host
        }
    
    def get_live_matches(self) -> List[Dict]:
        """Get all live matches"""
        url = f"{self.base_url}/fixtures"
        params = {'live': 'all'}
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()
            if response.status_code == 200:
                data = response.json()
                return data.get('response', [])
            else:
                logger.error(f"API returned status {response.status_code}: {response.text}")
                return []
        except requests.exceptions.Timeout:
            logger.error("API request timed out")
            return []
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching live matches: {e}")
            return []
    
    def get_match_statistics(self, fixture_id: int) -> Dict:
        """Get detailed match statistics"""
        url = f"{self.base_url}/fixtures/statistics"
        params = {'fixture': fixture_id}
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()
            if response.status_code == 200:
                data = response.json()
                return self._parse_statistics(data.get('response', []))
            else:
                logger.warning(f"No statistics for fixture {fixture_id}")
                return {}
        except Exception as e:
            logger.warning(f"Error fetching match stats for {fixture_id}: {e}")
            return {}
    
    def get_match_events(self, fixture_id: int) -> List[Dict]:
        """Get match events"""
        url = f"{self.base_url}/fixtures/events"
        params = {'fixture': fixture_id}
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()
            if response.status_code == 200:
                data = response.json()
                return data.get('response', [])
            return []
        except Exception as e:
            logger.warning(f"Error fetching match events for {fixture_id}: {e}")
            return []
    
    def get_fixture_details(self, fixture_id: int) -> Dict:
        """Get fixture details"""
        url = f"{self.base_url}/fixtures"
        params = {'id': fixture_id}
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()
            if response.status_code == 200:
                data = response.json()
                return data.get('response', [{}])[0]
            return {}
        except Exception as e:
            logger.warning(f"Error fetching fixture {fixture_id}: {e}")
            return {}
    
    def _parse_statistics(self, stats_data: List) -> Dict:
        """Parse statistics into a usable format"""
        parsed = {}
        for team_stats in stats_data:
            team_id = team_stats.get('team', {}).get('id')
            if not team_id:
                continue
                
            stats = {}
            for stat in team_stats.get('statistics', []):
                stat_type = stat.get('type')
                if stat_type:
                    value = stat.get('value')
                    # Convert percentage strings to floats
                    if isinstance(value, str) and '%' in value:
                        try:
                            value = float(value.strip('%')) / 100
                        except:
                            value = 0
                    stats[stat_type] = value
            parsed[team_id] = stats
        return parsed

# Layer 1: Event Probability Engine
class EventProbabilityEngine:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        
    def extract_features(self, match_data: Dict, current_minute: int) -> Dict[str, float]:
        """Extract features from match data"""
        fixture = match_data.get('fixture', {})
        teams = match_data.get('teams', {})
        score = match_data.get('goals', {})
        stats = match_data.get('statistics', {})
        
        home_team = teams.get('home', {})
        away_team = teams.get('away', {})
        home_id = home_team.get('id')
        away_id = away_team.get('id')
        
        home_stats = stats.get(home_id, {})
        away_stats = stats.get(away_id, {})
        
        # Get current scores safely
        home_score = score.get('home')
        away_score = score.get('away')
        
        if home_score is None:
            home_score = 0
        if away_score is None:
            away_score = 0
            
        # Safely extract statistics with defaults
        def get_stat(stat_dict, key, default=0):
            value = stat_dict.get(key, default)
            if value is None:
                return default
            return value
        
        # Calculate derived metrics
        features = {
            'minute': current_minute,
            'home_score': home_score,
            'away_score': away_score,
            'score_delta': home_score - away_score,
            
            # Attack metrics
            'home_xg': get_stat(home_stats, 'expected_goals', 0.0),
            'away_xg': get_stat(away_stats, 'expected_goals', 0.0),
            'xg_delta': get_stat(home_stats, 'expected_goals', 0.0) - get_stat(away_stats, 'expected_goals', 0.0),
            
            # Shot metrics
            'home_shots_on': get_stat(home_stats, 'shots on goal', 0),
            'away_shots_on': get_stat(away_stats, 'shots on goal', 0),
            'home_shots_off': get_stat(home_stats, 'shots off goal', 0),
            'away_shots_off': get_stat(away_stats, 'shots off goal', 0),
            'shot_pressure': (get_stat(home_stats, 'shots on goal', 0) + 
                            get_stat(home_stats, 'shots off goal', 0)) / max(current_minute, 1),
            
            # Possession and control
            'home_possession': self._parse_possession(get_stat(home_stats, 'ball possession', '50%')),
            'possession_trend': 0.5,  # Placeholder - would calculate from trend
            
            # Corner metrics
            'home_corners': get_stat(home_stats, 'corner kicks', 0),
            'away_corners': get_stat(away_stats, 'corner kicks', 0),
            'corner_rate': (get_stat(home_stats, 'corner kicks', 0) + 
                          get_stat(away_stats, 'corner kicks', 0)) / max(current_minute, 1) * 90,
            
            # Discipline
            'home_yellow': get_stat(home_stats, 'yellow cards', 0),
            'away_yellow': get_stat(away_stats, 'yellow cards', 0),
            'home_red': get_stat(home_stats, 'red cards', 0),
            'away_red': get_stat(away_stats, 'red cards', 0),
            
            # Dangerous attacks
            'home_attacks': get_stat(home_stats, 'dangerous_attacks', 0),
            'away_attacks': get_stat(away_stats, 'dangerous_attacks', 0),
        }
        
        # Calculate attack ratio safely
        total_attacks = features['home_attacks'] + features['away_attacks']
        features['attack_ratio'] = features['home_attacks'] / max(total_attacks, 1)
        
        return features
    
    def _parse_possession(self, possession_value):
        """Parse possession value from various formats"""
        if possession_value is None:
            return 0.5
            
        if isinstance(possession_value, (int, float)):
            if possession_value > 1:  # Assume percentage as whole number
                return possession_value / 100
            return possession_value
        elif isinstance(possession_value, str):
            if '%' in possession_value:
                try:
                    return float(possession_value.strip('%')) / 100
                except:
                    return 0.5
            else:
                try:
                    return float(possession_value)
                except:
                    return 0.5
        return 0.5
    
    def calculate_probabilities(self, features: Dict) -> Dict[str, float]:
        """Calculate event probabilities using learned models"""
        
        minute = features['minute']
        score_delta = features['score_delta']
        xg_delta = features['xg_delta']
        shot_pressure = features['shot_pressure']
        corner_rate = features['corner_rate']
        
        # Adjust for game state
        time_factor = min(minute / 90, 1)
        comeback_factor = max(0, -score_delta * 0.1) if score_delta < 0 else 0
        
        # Calculate probabilities
        base_goal_rate = 0.025  # Base probability per minute
        
        probabilities = {
            # Goal probabilities - time decay in last minutes
            'goal_next_5min': min(0.95, base_goal_rate * 5 + shot_pressure * 2 + abs(score_delta) * 0.05),
            'goal_next_10min': min(0.95, base_goal_rate * 10 + shot_pressure * 1.5 + abs(score_delta) * 0.08),
            'goal_next_15min': min(0.95, base_goal_rate * 15 * (1 - time_factor * 0.3) + shot_pressure * 1.2),
            
            # Team-specific probabilities
            'home_goal_probability': max(0, min(1, 0.5 + (xg_delta * 0.3) - (max(score_delta, 0) * 0.15))),
            'away_goal_probability': max(0, min(1, 0.5 - (xg_delta * 0.3) + (max(-score_delta, 0) * 0.15))),
            
            # Expected final goals
            'expected_final_goals': max(0, features['home_score'] + features['away_score'] + 
                                      (xg_delta * 0.5 * (90 - minute) / 90) + comeback_factor * 0.5),
            
            # Corner probabilities
            'corner_next_10min': min(0.9, (corner_rate / 9) * (1 - time_factor * 0.2)),
            'corner_pressure_home': features['home_corners'] / max(features['home_corners'] + features['away_corners'], 1),
            
            # Card probabilities
            'yellow_card_next_10min': min(0.8, (features['home_yellow'] + features['away_yellow']) / max(minute, 1) * 10 * (1 + time_factor)),
            'red_card_risk_away': min(0.5, features['away_yellow'] * 0.15 + features['away_red'] * 0.3),
            'red_card_risk_home': min(0.5, features['home_yellow'] * 0.15 + features['home_red'] * 0.3),
            
            # Game state
            'comeback_probability': comeback_factor,
            'shutdown_probability': max(0, score_delta * 0.2) if score_delta > 1 else 0,
            
            # Control metrics
            'possession_pressure': features['home_possession'] * (1 + features['attack_ratio']),
            'momentum_shift': abs(features['attack_ratio'] - 0.5) * 1.5
        }
        
        # Apply time decay in last 10 minutes
        if minute > 80:
            decay_factor = 1 - ((minute - 80) / 20)
            for key in ['goal_next_5min', 'goal_next_10min', 'goal_next_15min']:
                if key in probabilities:
                    probabilities[key] *= decay_factor
        
        # Ensure probabilities are between 0 and 1
        for key in probabilities:
            probabilities[key] = max(0, min(1, probabilities[key]))
        
        return probabilities

# Layer 2: Market Opportunity Generator
class MarketOpportunityGenerator:
    def __init__(self):
        self.market_definitions = {
            'next_goal': {
                'condition': lambda p: p.get('goal_next_10min', 0) > 0.35,
                'probability_field': 'goal_next_10min',
                'description': 'Next Goal'
            },
            'over_2.5': {
                'condition': lambda p: p.get('expected_final_goals', 0) >= 2.5,
                'probability_field': 'expected_final_goals',
                'description': 'Over 2.5 Goals'
            },
            'next_corner': {
                'condition': lambda p: p.get('corner_next_10min', 0) > 0.4,
                'probability_field': 'corner_next_10min',
                'description': 'Next Corner'
            },
            'home_to_score': {
                'condition': lambda p: p.get('home_goal_probability', 0) > 0.6,
                'probability_field': 'home_goal_probability',
                'description': 'Home Team to Score'
            },
            'away_to_score': {
                'condition': lambda p: p.get('away_goal_probability', 0) > 0.6,
                'probability_field': 'away_goal_probability',
                'description': 'Away Team to Score'
            },
            'yellow_card': {
                'condition': lambda p: p.get('yellow_card_next_10min', 0) > 0.5,
                'probability_field': 'yellow_card_next_10min',
                'description': 'Yellow Card Next 10min'
            },
            'both_teams_score': {
                'condition': lambda p: p.get('home_goal_probability', 0) > 0.4 and 
                                      p.get('away_goal_probability', 0) > 0.4,
                'probability_field': lambda p: min(p.get('home_goal_probability', 0), 
                                                  p.get('away_goal_probability', 0)),
                'description': 'Both Teams to Score'
            }
        }
    
    def generate_opportunities(self, probabilities: Dict, current_state: Dict) -> List[Dict]:
        """Generate market opportunities from probabilities"""
        opportunities = []
        
        for market_id, definition in self.market_definitions.items():
            try:
                if definition['condition'](probabilities):
                    prob_field = definition['probability_field']
                    if callable(prob_field):
                        probability = prob_field(probabilities)
                    else:
                        probability = probabilities.get(prob_field, 0)
                    
                    opportunity = {
                        'market_type': market_id,
                        'description': definition['description'],
                        'probability': probability,
                        'current_state': current_state.copy(),
                        'parameters': self._get_market_parameters(market_id, probabilities, current_state)
                    }
                    opportunities.append(opportunity)
            except Exception as e:
                logger.warning(f"Error generating opportunity for {market_id}: {e}")
                continue
        
        return opportunities
    
    def _get_market_parameters(self, market_id: str, probabilities: Dict, state: Dict) -> Dict:
        """Get specific parameters for each market type"""
        params = {
            'minute': state.get('minute', 0),
            'score': f"{state.get('home_score', 0)}-{state.get('away_score', 0)}",
            'xg_home': state.get('home_xg', 0),
            'xg_away': state.get('away_xg', 0)
        }
        return params

# Layer 3: Value & Confidence Filter
class ValueConfidenceFilter:
    def __init__(self):
        self.min_confidence = 0.65
        self.min_value = 1.1
        
    def calculate_confidence_score(self, opportunity: Dict, league_id: int, 
                                 minute: int, probabilities: Dict, db_conn) -> Dict:
        """Calculate confidence score for an opportunity"""
        try:
            # Get historical accuracy
            hist_accuracy = self._get_historical_accuracy(
                opportunity['market_type'], 
                league_id, 
                minute,
                probabilities,
                db_conn
            )
            
            # Get league reliability
            league_weight = self._get_league_weight(league_id, db_conn)
            
            # Minute window weight
            minute_weight = self._get_minute_weight(minute, opportunity['market_type'])
            
            # Game state compatibility
            game_state_weight = self._get_game_state_weight(opportunity['current_state'], 
                                                           opportunity['market_type'])
            
            # Calculate final score
            base_prob = opportunity['probability']
            confidence_score = (
                base_prob
                * hist_accuracy
                * league_weight
                * minute_weight
                * game_state_weight
            )
            
            # Calculate expected value (simplified)
            expected_value = self._calculate_expected_value(base_prob, opportunity['market_type'])
            
            return {
                'confidence_score': confidence_score,
                'expected_value': expected_value,
                'historical_accuracy': hist_accuracy,
                'league_weight': league_weight,
                'minute_weight': minute_weight,
                'game_state_weight': game_state_weight,
                'final_decision': confidence_score >= self.min_confidence and expected_value >= self.min_value
            }
        except Exception as e:
            logger.error(f"Error calculating confidence score: {e}")
            return {
                'confidence_score': 0,
                'expected_value': 0,
                'historical_accuracy': 0.5,
                'league_weight': 0.8,
                'minute_weight': 1.0,
                'game_state_weight': 1.0,
                'final_decision': False
            }
    
    def _get_historical_accuracy(self, market_type: str, league_id: int, 
                               minute: int, probabilities: Dict, db_conn) -> float:
        """Get historical accuracy for this market in this context"""
        try:
            cur = db_conn.cursor()
            
            # Query historical performance
            query = """
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins
                FROM market_results mr
                JOIN market_suggestions ms ON mr.market_suggestion_id = ms.id
                WHERE ms.market_type = %s
                AND ms.fixture_id IN (
                    SELECT fixture_id FROM event_predictions 
                    WHERE league_id = %s AND ABS(minute - %s) <= 15
                )
            """
            
            cur.execute(query, (market_type, league_id, minute))
            result = cur.fetchone()
            cur.close()
            
            if result and result[0] > 10:
                accuracy = result[1] / result[0]
            else:
                # Default accuracy based on market type
                default_accuracies = {
                    'next_goal': 0.45,
                    'over_2.5': 0.55,
                    'next_corner': 0.60,
                    'home_to_score': 0.65,
                    'away_to_score': 0.65,
                    'yellow_card': 0.50,
                    'both_teams_score': 0.52
                }
                accuracy = default_accuracies.get(market_type, 0.5)
            
            return accuracy
        except Exception as e:
            logger.warning(f"Error getting historical accuracy: {e}")
            return 0.5
    
    def _get_league_weight(self, league_id: int, db_conn) -> float:
        """Get league reliability score"""
        try:
            cur = db_conn.cursor()
            cur.execute("""
                SELECT reliability_score 
                FROM league_stats 
                WHERE league_id = %s
            """, (league_id,))
            
            result = cur.fetchone()
            cur.close()
            
            return result[0] if result else 0.8
        except Exception as e:
            logger.warning(f"Error getting league weight: {e}")
            return 0.8
    
    def _get_minute_weight(self, minute: int, market_type: str) -> float:
        """Get weight based on minute window"""
        if market_type == 'next_goal':
            if minute < 20:
                return 0.7
            elif minute < 70:
                return 1.0
            else:
                return 0.8
        elif market_type == 'over_2.5':
            if minute < 30:
                return 0.9
            elif minute < 60:
                return 1.0
            else:
                return 0.6
        elif market_type == 'next_corner':
            if 25 < minute < 75:
                return 1.0
            else:
                return 0.7
        else:
            return 1.0
    
    def _get_game_state_weight(self, game_state: Dict, market_type: str) -> float:
        """Get weight based on game state compatibility"""
        score_delta = game_state.get('score_delta', 0)
        red_cards = game_state.get('home_red', 0) + game_state.get('away_red', 0)
        
        if market_type == 'next_goal':
            if abs(score_delta) >= 3:
                return 0.3
            elif red_cards > 0:
                return 1.2
            else:
                return 1.0
        elif market_type == 'over_2.5':
            if game_state.get('home_score', 0) + game_state.get('away_score', 0) >= 2:
                return 1.2
            else:
                return 1.0
        else:
            return 1.0
    
    def _calculate_expected_value(self, probability: float, market_type: str) -> float:
        """Calculate expected value"""
        avg_odds = {
            'next_goal': 2.0,
            'over_2.5': 1.9,
            'next_corner': 1.8,
            'home_to_score': 1.5,
            'away_to_score': 1.6,
            'yellow_card': 2.2,
            'both_teams_score': 1.8
        }
        
        odds = avg_odds.get(market_type, 2.0)
        ev = (probability * (odds - 1)) - ((1 - probability) * 1)
        return ev

# Layer 4: Self-Learning Feedback Loop
class SelfLearningFeedbackLoop:
    def __init__(self):
        pass
    
    def store_prediction(self, fixture_id: int, league_id: int, minute: int,
                        predictions: Dict, market_suggestions: List[Dict], db_conn):
        """Store predictions and market suggestions"""
        try:
            cur = db_conn.cursor()
            
            # Store event predictions
            cur.execute("""
                INSERT INTO event_predictions (fixture_id, league_id, minute, predictions)
                VALUES (%s, %s, %s, %s)
                RETURNING id
            """, (fixture_id, league_id, minute, json.dumps(predictions)))
            
            # Store market suggestions
            for suggestion in market_suggestions:
                cur.execute("""
                    INSERT INTO market_suggestions 
                    (fixture_id, market_type, probability, confidence_score, 
                     expected_value, parameters)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    fixture_id,
                    suggestion['market_type'],
                    suggestion['probability'],
                    suggestion.get('confidence_score', 0),
                    suggestion.get('expected_value', 0),
                    json.dumps(suggestion.get('parameters', {}))
                ))
            
            db_conn.commit()
            cur.close()
        except Exception as e:
            logger.error(f"Error storing prediction: {e}")
            db_conn.rollback()
            raise
    
    def update_outcome(self, fixture_id: int, db_conn):
        """Update outcomes for completed matches"""
        try:
            cur = db_conn.cursor()
            
            # Update league statistics
            cur.execute("""
                INSERT INTO league_stats (league_id, league_name, reliability_score, total_predictions, correct_predictions)
                VALUES (%s, 'Unknown League', 0.8, 0, 0)
                ON CONFLICT (league_id) DO NOTHING
            """, (fixture_id,))
            
            db_conn.commit()
            cur.close()
        except Exception as e:
            logger.error(f"Error updating outcome: {e}")
            db_conn.rollback()

# Telegram Bot Integration
class TelegramNotifier:
    def __init__(self, token: str, chat_id: str):
        self.bot = telegram.Bot(token=token)
        self.chat_id = chat_id
    
    def send_tip(self, fixture_info: Dict, market_suggestion: Dict, confidence: Dict):
        """Send tip to Telegram"""
        try:
            message = self._format_tip_message(fixture_info, market_suggestion, confidence)
            self.bot.send_message(chat_id=self.chat_id, text=message, parse_mode='HTML')
            logger.info(f"Tip sent to Telegram: {market_suggestion['description']}")
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")
    
    def _format_tip_message(self, fixture_info: Dict, market_suggestion: Dict, 
                          confidence: Dict) -> str:
        """Format tip message for Telegram"""
        league = fixture_info.get('league', {}).get('name', 'Unknown League')
        home = fixture_info.get('teams', {}).get('home', {}).get('name', 'Home')
        away = fixture_info.get('teams', {}).get('away', {}).get('name', 'Away')
        score = fixture_info.get('goals', {})
        minute = fixture_info.get('fixture', {}).get('status', {}).get('elapsed', 0)
        
        emoji = "✅" if confidence.get('final_decision', False) else "⚠️"
        
        message = f"""
{emoji} <b>PREDICTION ALERT</b> {emoji}

🏆 <b>{league}</b>
⚽ {home} vs {away}
⏱️ Minute: {minute}'
📊 Score: {score.get('home', 0)}-{score.get('away', 0)}

🎯 <b>Market:</b> {market_suggestion['description']}
📈 <b>Probability:</b> {market_suggestion['probability']:.2%}
💪 <b>Confidence:</b> {confidence.get('confidence_score', 0):.2%}
💰 <b>Expected Value:</b> {confidence.get('expected_value', 0):.2f}

⚡ <b>Decision:</b> {"STRONG VALUE" if confidence.get('final_decision') else "MONITOR ONLY"}

#FootballPrediction #LiveBetting
        """
        
        return message

# Main Engine
class PredictionEngine:
    def __init__(self):
        self.api_client = APIFootballClient(API_FOOTBALL_KEY)
        self.event_engine = EventProbabilityEngine()
        self.market_generator = MarketOpportunityGenerator()
        self.telegram_notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        self.value_filter = ValueConfidenceFilter()
        self.learning_loop = SelfLearningFeedbackLoop()
        
        # Initialize database
        init_database()
        
        # Control flags
        self.is_live_scanning = False
        self.is_training = False
        
    def _analyze_match(self, match_data: Dict):
        """Analyze a single match with fresh database connection"""
        try:
            fixture_id = match_data['fixture']['id']
            league_id = match_data['league']['id']
            minute = match_data['fixture']['status']['elapsed']
            
            # Skip matches that are finished or haven't started
            status = match_data['fixture']['status'].get('short', '')
            if status in ['FT', 'AET', 'PEN', 'SUSP', 'PST', 'CANC', 'ABD', 'AWD', 'WO']:
                logger.info(f"Skipping match {fixture_id} with status {status}")
                return
            
            logger.info(f"Analyzing match {fixture_id} at minute {minute}")
            
            # Get detailed statistics with timeout
            stats = self.api_client.get_match_statistics(fixture_id)
            if not stats:
                logger.warning(f"No statistics available for match {fixture_id}")
                return
                
            events = self.api_client.get_match_events(fixture_id)
            
            # Combine data
            match_data['statistics'] = stats
            match_data['events'] = events
            
            # Extract features
            features = self.event_engine.extract_features(match_data, minute)
            
            # Calculate probabilities
            probabilities = self.event_engine.calculate_probabilities(features)
            
            # Generate market opportunities
            opportunities = self.market_generator.generate_opportunities(
                probabilities, 
                features
            )
            
            if not opportunities:
                logger.info(f"No opportunities generated for match {fixture_id}")
                return
            
            # Create fresh database connection for this match
            db_conn = get_db_connection()
            try:
                # Filter and score opportunities
                valuable_opportunities = []
                for opportunity in opportunities:
                    confidence = self.value_filter.calculate_confidence_score(
                        opportunity, league_id, minute, probabilities, db_conn
                    )
                    
                    opportunity['confidence_score'] = confidence['confidence_score']
                    opportunity['expected_value'] = confidence['expected_value']
                    opportunity['confidence_breakdown'] = confidence
                    
                    if confidence['final_decision']:
                        valuable_opportunities.append(opportunity)
                        
                        # Send to Telegram
                        self.telegram_notifier.send_tip(match_data, opportunity, confidence)
                
                # Store predictions
                if opportunities:
                    self.learning_loop.store_prediction(
                        fixture_id, league_id, minute, probabilities, opportunities, db_conn
                    )
                
                logger.info(f"Found {len(valuable_opportunities)} valuable opportunities for match {fixture_id}")
                
            finally:
                db_conn.close()
                
        except KeyError as e:
            logger.warning(f"Missing data in match {match_data.get('fixture', {}).get('id', 'unknown')}: {e}")
        except Exception as e:
            logger.error(f"Error analyzing match {match_data.get('fixture', {}).get('id', 'unknown')}: {e}")
    
    def live_scan(self):
        """Scan live matches and generate predictions"""
        if self.is_live_scanning:
            logger.warning("Live scan already in progress")
            return
        
        self.is_live_scanning = True
        logger.info("Starting live scan...")
        
        try:
            live_matches = self.api_client.get_live_matches()
            logger.info(f"Found {len(live_matches)} live matches")
            
            for match in live_matches:
                try:
                    self._analyze_match(match)
                except Exception as e:
                    logger.error(f"Error in match analysis: {e}")
                
                # Small delay to avoid rate limiting
                time.sleep(2)
        
        except Exception as e:
            logger.error(f"Error in live scan: {e}")
        finally:
            self.is_live_scanning = False
            logger.info("Live scan completed")
    
    def train_models(self):
        """Train/re-train the models"""
        if self.is_training:
            logger.warning("Training already in progress")
            return
        
        self.is_training = True
        logger.info("Starting model training...")
        
        try:
            # Import training module
            from train_models import train_all_models
            
            # Train models with fresh connection
            db_conn = get_db_connection()
            try:
                results = train_all_models(db_conn)
                logger.info(f"Model training completed: {results}")
            finally:
                db_conn.close()
                
        except Exception as e:
            logger.error(f"Error in training: {e}")
        finally:
            self.is_training = False
    
    def daily_digest(self):
        """Generate daily digest"""
        db_conn = get_db_connection()
        try:
            cur = db_conn.cursor()
            
            # Get yesterday's predictions
            yesterday = datetime.now() - timedelta(days=1)
            
            cur.execute("""
                SELECT 
                    COUNT(*) as total_predictions,
                    COUNT(DISTINCT fixture_id) as matches_analyzed,
                    AVG(confidence_score) as avg_confidence,
                    SUM(CASE WHEN expected_value > 1.1 THEN 1 ELSE 0 END) as valuable_opportunities
                FROM market_suggestions
                WHERE created_at >= %s
            """, (yesterday,))
            
            stats = cur.fetchone()
            
            # Get top performing markets
            cur.execute("""
                SELECT 
                    market_type,
                    COUNT(*) as suggestions,
                    AVG(confidence_score) as avg_confidence
                FROM market_suggestions
                WHERE created_at >= %s
                GROUP BY market_type
                ORDER BY avg_confidence DESC
                LIMIT 5
            """, (yesterday,))
            
            top_markets = cur.fetchall()
            
            # Format digest message
            digest = f"""
📊 DAILY DIGEST - {datetime.now().strftime('%Y-%m-%d')}

🔍 Analysis Summary:
• Total Predictions: {stats[0] or 0}
• Matches Analyzed: {stats[1] or 0}
• Average Confidence: {(stats[2] or 0):.2%}
• Valuable Opportunities: {stats[3] or 0}

🏆 Top Performing Markets:
"""
            
            for market in top_markets:
                digest += f"• {market[0]}: {market[1]} suggestions, {(market[2] or 0):.2%} avg confidence\n"
            
            # Send digest to Telegram
            try:
                self.telegram_notifier.bot.send_message(
                    chat_id=TELEGRAM_CHAT_ID,
                    text=digest,
                    parse_mode='HTML'
                )
                logger.info("Daily digest sent")
            except Exception as e:
                logger.error(f"Error sending daily digest: {e}")
                
        finally:
            db_conn.close()
    
    def auto_tune(self):
        """Auto-tune parameters based on performance"""
        logger.info("Starting auto-tuning...")
        
        db_conn = get_db_connection()
        try:
            cur = db_conn.cursor()
            
            # Analyze recent performance
            cur.execute("""
                SELECT 
                    market_type,
                    AVG(CASE WHEN mr.outcome = 'win' THEN 1 ELSE 0 END) as win_rate,
                    COUNT(*) as total
                FROM market_results mr
                JOIN market_suggestions ms ON mr.market_suggestion_id = ms.id
                WHERE mr.analyzed_at >= NOW() - INTERVAL '7 days'
                GROUP BY market_type
                HAVING COUNT(*) > 10
            """)
            
            performance = cur.fetchall()
            
            # Adjust thresholds based on performance
            adjustments = []
            for market_type, win_rate, total in performance:
                if win_rate < 0.4:
                    adjustments.append((market_type, 'increase threshold'))
                elif win_rate > 0.6:
                    adjustments.append((market_type, 'decrease threshold'))
            
            logger.info(f"Auto-tuning adjustments: {adjustments}")
            
        finally:
            db_conn.close()
        
        return adjustments
    
    def backfill_historical(self, days: int = 30):
        """Backfill historical data"""
        logger.info(f"Backfilling historical data for {days} days...")
        
        # This would fetch historical matches and analyze them
        # Implementation depends on API access to historical data
        logger.info("Historical backfill placeholder - implement with historical API access")
    
    def health_check(self) -> Dict:
        """Check system health"""
        health = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'components': {}
        }
        
        # Check database
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("SELECT 1")
            cur.close()
            conn.close()
            health['components']['database'] = 'healthy'
        except Exception as e:
            health['components']['database'] = f'unhealthy: {str(e)}'
            health['status'] = 'degraded'
        
        # Check API
        try:
            test_matches = self.api_client.get_live_matches()
            health['components']['api_football'] = f'healthy ({len(test_matches)} live matches)'
        except Exception as e:
            health['components']['api_football'] = f'unhealthy: {str(e)}'
            health['status'] = 'degraded'
        
        # Check Telegram
        try:
            self.telegram_notifier.bot.get_me()
            health['components']['telegram'] = 'healthy'
        except Exception as e:
            health['components']['telegram'] = f'unhealthy: {str(e)}'
            health['status'] = 'degraded'
        
        return health

# Manual Control Functions
class ManualControl:
    def __init__(self, engine: PredictionEngine):
        self.engine = engine
    
    def run_live_scan(self):
        """Manually trigger live scan"""
        self.engine.live_scan()
    
    def run_training(self):
        """Manually trigger training"""
        self.engine.train_models()
    
    def run_daily_digest(self):
        """Manually trigger daily digest"""
        self.engine.daily_digest()
    
    def run_auto_tune(self):
        """Manually trigger auto-tuning"""
        return self.engine.auto_tune()
    
    def run_backfill(self, days: int = 30):
        """Manually trigger backfill"""
        self.engine.backfill_historical(days)
    
    def run_health_check(self):
        """Run health check and return results"""
        return self.engine.health_check()

# Main application
def main():
    # Initialize engine
    engine = PredictionEngine()
    controller = ManualControl(engine)
    
    # Schedule tasks
    schedule.every(10).minutes.do(engine.live_scan)
    schedule.every().day.at("02:00").do(engine.train_models)
    schedule.every().day.at("08:00").do(engine.daily_digest)
    schedule.every().sunday.at("03:00").do(engine.auto_tune)
    
    logger.info("Prediction Engine Started")
    logger.info("Scheduled tasks:")
    logger.info("- Live scan every 10 minutes")
    logger.info("- Model training at 02:00 daily")
    logger.info("- Daily digest at 08:00 daily")
    logger.info("- Auto-tuning every Sunday at 03:00")
    
    # Initial health check
    health = engine.health_check()
    logger.info(f"Initial health check: {health}")
    
    # Run scheduled tasks
    while True:
        try:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            time.sleep(300)  # Wait 5 minutes on error

if __name__ == "__main__":
    main()
