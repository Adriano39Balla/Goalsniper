import os
import sys
import time
import schedule
import json
import logging
import asyncio
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
from telegram.error import TelegramError

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

# Suppress specific warnings
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

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
    conn.autocommit = True
    return conn

# Initialize database tables
def init_database():
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
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
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_market_suggestions_fixture 
            ON market_suggestions(fixture_id)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_event_predictions_fixture 
            ON event_predictions(fixture_id)
        """)
        
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
    finally:
        cur.close()
        conn.close()

# API Football client
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
            data = response.json()
            
            # Handle API response structure
            if 'response' in data:
                matches = data['response']
                logger.info(f"API returned {len(matches)} live matches")
                return matches
            else:
                logger.warning(f"Unexpected API response format: {data}")
                return []
                
        except requests.exceptions.Timeout:
            logger.error("API request timed out")
            return []
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching live matches: {e}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing API response: {e}")
            return []
    
    def get_match_statistics(self, fixture_id: int) -> Dict:
        """Get detailed match statistics"""
        url = f"{self.base_url}/fixtures/statistics"
        params = {'fixture': fixture_id}
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if 'response' in data:
                return self._parse_statistics(data['response'])
            else:
                logger.warning(f"No statistics in response for fixture {fixture_id}")
                return {}
                
        except Exception as e:
            logger.debug(f"No statistics for fixture {fixture_id}: {e}")
            return {}
    
    def _parse_statistics(self, stats_data: List) -> Dict:
        """Parse statistics into a usable format"""
        if not stats_data:
            return {}
            
        parsed = {}
        for team_stats in stats_data:
            team = team_stats.get('team', {})
            team_id = team.get('id')
            if not team_id:
                continue
                
            stats = {}
            for stat in team_stats.get('statistics', []):
                stat_type = stat.get('type')
                if not stat_type:
                    continue
                    
                value = stat.get('value')
                # Convert percentage strings to floats
                if isinstance(value, str) and '%' in value:
                    try:
                        value = float(value.strip('%')) / 100
                    except (ValueError, AttributeError):
                        value = 0
                elif value is None:
                    value = 0
                    
                stats[stat_type] = value
                
            parsed[team_id] = stats
            
        return parsed

# Layer 1: Event Probability Engine
class EventProbabilityEngine:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def extract_features(self, match_data: Dict, current_minute: int) -> Dict[str, float]:
        """Extract features from match data"""
        try:
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
            home_score = score.get('home', 0) or 0
            away_score = score.get('away', 0) or 0
            
            # Helper function to safely extract stats
            def get_stat(stat_dict, key, default=0):
                value = stat_dict.get(key, default)
                if value is None:
                    return default
                return value
            
            # Calculate basic metrics
            features = {
                'minute': current_minute,
                'home_score': home_score,
                'away_score': away_score,
                'score_delta': home_score - away_score,
                
                # Attack metrics
                'home_xg': float(get_stat(home_stats, 'expected_goals', 0)),
                'away_xg': float(get_stat(away_stats, 'expected_goals', 0)),
                'xg_delta': float(get_stat(home_stats, 'expected_goals', 0)) - 
                          float(get_stat(away_stats, 'expected_goals', 0)),
                
                # Shot metrics
                'home_shots_on': get_stat(home_stats, 'shots on goal', 0),
                'away_shots_on': get_stat(away_stats, 'shots on goal', 0),
                'home_shots_off': get_stat(home_stats, 'shots off goal', 0),
                'away_shots_off': get_stat(away_stats, 'shots off goal', 0),
                
                # Possession
                'home_possession': self._parse_possession(get_stat(home_stats, 'ball possession', '50%')),
                
                # Corners
                'home_corners': get_stat(home_stats, 'corner kicks', 0),
                'away_corners': get_stat(away_stats, 'corner kicks', 0),
                
                # Cards
                'home_yellow': get_stat(home_stats, 'yellow cards', 0),
                'away_yellow': get_stat(away_stats, 'yellow cards', 0),
                'home_red': get_stat(home_stats, 'red cards', 0),
                'away_red': get_stat(away_stats, 'red cards', 0),
                
                # Attacks
                'home_attacks': get_stat(home_stats, 'dangerous_attacks', 0),
                'away_attacks': get_stat(away_stats, 'dangerous_attacks', 0),
            }
            
            # Calculate derived metrics
            minute_for_calc = max(current_minute, 1)
            
            features['shot_pressure'] = (
                features['home_shots_on'] + features['home_shots_off']
            ) / minute_for_calc
            
            features['corner_rate'] = (
                features['home_corners'] + features['away_corners']
            ) / minute_for_calc * 90
            
            total_attacks = features['home_attacks'] + features['away_attacks']
            features['attack_ratio'] = (
                features['home_attacks'] / max(total_attacks, 1)
            )
            
            features['possession_trend'] = 0.5  # Default value
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            # Return minimal features
            return {
                'minute': current_minute,
                'home_score': 0,
                'away_score': 0,
                'score_delta': 0,
                'home_xg': 0,
                'away_xg': 0,
                'xg_delta': 0,
                'home_shots_on': 0,
                'away_shots_on': 0,
                'home_shots_off': 0,
                'away_shots_off': 0,
                'shot_pressure': 0,
                'home_possession': 0.5,
                'possession_trend': 0.5,
                'home_corners': 0,
                'away_corners': 0,
                'corner_rate': 0,
                'home_yellow': 0,
                'away_yellow': 0,
                'home_red': 0,
                'away_red': 0,
                'home_attacks': 0,
                'away_attacks': 0,
                'attack_ratio': 0.5
            }
    
    def _parse_possession(self, possession_value):
        """Parse possession value from various formats"""
        if possession_value is None:
            return 0.5
            
        if isinstance(possession_value, (int, float)):
            if possession_value > 1:
                return possession_value / 100
            return possession_value
        elif isinstance(possession_value, str):
            if '%' in possession_value:
                try:
                    return float(possession_value.strip('%')) / 100
                except (ValueError, AttributeError):
                    return 0.5
            else:
                try:
                    return float(possession_value)
                except (ValueError, AttributeError):
                    return 0.5
        return 0.5
    
    def calculate_probabilities(self, features: Dict) -> Dict[str, float]:
        """Calculate event probabilities"""
        try:
            minute = features['minute']
            score_delta = features['score_delta']
            xg_delta = features['xg_delta']
            shot_pressure = features['shot_pressure']
            corner_rate = features['corner_rate']
            
            # Time factor (games slow down towards the end)
            time_factor = min(minute / 90, 1)
            time_decay = max(0, 1 - (max(minute - 80, 0) / 20))
            
            # Base calculations
            base_goal_rate = 0.025
            
            probabilities = {
                # Goal probabilities
                'goal_next_5min': min(0.9, base_goal_rate * 5 + shot_pressure * 1.5),
                'goal_next_10min': min(0.95, base_goal_rate * 10 + shot_pressure * 1.2),
                'goal_next_15min': min(0.98, base_goal_rate * 15 + shot_pressure),
                
                # Team probabilities
                'home_goal_probability': max(0, min(1, 0.5 + (xg_delta * 0.2))),
                'away_goal_probability': max(0, min(1, 0.5 - (xg_delta * 0.2))),
                
                # Expected goals
                'expected_final_goals': max(0, 
                    features['home_score'] + features['away_score'] + 
                    (abs(xg_delta) * 0.3 * (90 - minute) / 90)
                ),
                
                # Corner probabilities
                'corner_next_10min': min(0.9, (corner_rate / 9) * time_decay),
                'corner_pressure_home': (
                    features['home_corners'] / 
                    max(features['home_corners'] + features['away_corners'], 1)
                ),
                
                # Card probabilities
                'yellow_card_next_10min': min(0.8, 
                    (features['home_yellow'] + features['away_yellow']) / 
                    max(minute, 1) * 10
                ),
                'red_card_risk_away': min(0.5, features['away_yellow'] * 0.1),
                'red_card_risk_home': min(0.5, features['home_yellow'] * 0.1),
                
                # Game state
                'comeback_probability': max(0, -score_delta * 0.1) if score_delta < 0 else 0,
                'shutdown_probability': max(0, score_delta * 0.15) if score_delta > 1 else 0,
                
                # Control metrics
                'possession_pressure': features['home_possession'],
                'momentum_shift': abs(features['attack_ratio'] - 0.5) * 1.2
            }
            
            # Apply time decay for late game
            if minute > 80:
                for key in ['goal_next_5min', 'goal_next_10min', 'goal_next_15min', 'corner_next_10min']:
                    if key in probabilities:
                        probabilities[key] *= time_decay
            
            # Clip probabilities
            for key in probabilities:
                probabilities[key] = max(0, min(1, probabilities[key]))
            
            return probabilities
            
        except Exception as e:
            logger.error(f"Error calculating probabilities: {e}")
            return {
                'goal_next_5min': 0.1,
                'goal_next_10min': 0.15,
                'goal_next_15min': 0.2,
                'home_goal_probability': 0.5,
                'away_goal_probability': 0.5,
                'expected_final_goals': 2.5,
                'corner_next_10min': 0.3,
                'corner_pressure_home': 0.5,
                'yellow_card_next_10min': 0.2,
                'red_card_risk_away': 0.05,
                'red_card_risk_home': 0.05,
                'comeback_probability': 0.1,
                'shutdown_probability': 0.1,
                'possession_pressure': 0.5,
                'momentum_shift': 0
            }

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
                        'current_state': current_state,
                        'parameters': {
                            'minute': current_state.get('minute', 0),
                            'score': f"{current_state.get('home_score', 0)}-{current_state.get('away_score', 0)}"
                        }
                    }
                    opportunities.append(opportunity)
            except Exception as e:
                logger.debug(f"Error generating opportunity for {market_id}: {e}")
                continue
        
        return opportunities

# Layer 3: Value & Confidence Filter
class ValueConfidenceFilter:
    def __init__(self):
        self.min_confidence = 0.65
        self.min_value = 1.1
        
    def calculate_confidence_score(self, opportunity: Dict, league_id: int, 
                                 minute: int, probabilities: Dict) -> Dict:
        """Calculate confidence score for an opportunity"""
        try:
            # Default weights
            hist_accuracy = 0.5
            league_weight = 0.8
            minute_weight = 1.0
            game_state_weight = 1.0
            
            # Calculate minute weight
            market_type = opportunity['market_type']
            minute_weight = self._get_minute_weight(minute, market_type)
            
            # Calculate game state weight
            game_state = opportunity['current_state']
            game_state_weight = self._get_game_state_weight(game_state, market_type)
            
            # Calculate confidence
            base_prob = opportunity['probability']
            confidence_score = (
                base_prob
                * hist_accuracy
                * league_weight
                * minute_weight
                * game_state_weight
            )
            
            # Calculate expected value
            expected_value = self._calculate_expected_value(base_prob, market_type)
            
            # Final decision
            final_decision = (
                confidence_score >= self.min_confidence and 
                expected_value >= self.min_value
            )
            
            return {
                'confidence_score': confidence_score,
                'expected_value': expected_value,
                'historical_accuracy': hist_accuracy,
                'league_weight': league_weight,
                'minute_weight': minute_weight,
                'game_state_weight': game_state_weight,
                'final_decision': final_decision
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
        else:
            return 1.0
    
    def _get_game_state_weight(self, game_state: Dict, market_type: str) -> float:
        """Get weight based on game state compatibility"""
        score_delta = game_state.get('score_delta', 0)
        
        if market_type == 'next_goal':
            if abs(score_delta) >= 3:
                return 0.3
            else:
                return 1.0
        elif market_type == 'over_2.5':
            total_goals = game_state.get('home_score', 0) + game_state.get('away_score', 0)
            if total_goals >= 2:
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

# Telegram Bot Integration (Fixed Async)
class TelegramNotifier:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.bot = None
        self._initialize_bot()
    
    def _initialize_bot(self):
        """Initialize the Telegram bot"""
        try:
            self.bot = telegram.Bot(token=self.token)
            logger.info("Telegram bot initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Telegram bot: {e}")
            self.bot = None
    
    async def _send_tip_async(self, fixture_info: Dict, market_suggestion: Dict, confidence: Dict):
        """Send tip asynchronously"""
        if not self.bot:
            logger.error("Telegram bot not initialized")
            return
        
        try:
            message = self._format_tip_message(fixture_info, market_suggestion, confidence)
            await self.bot.send_message(
                chat_id=self.chat_id, 
                text=message, 
                parse_mode='HTML',
                disable_web_page_preview=True
            )
            logger.info(f"Tip sent to Telegram: {market_suggestion['description']}")
        except TelegramError as e:
            logger.error(f"Telegram API error: {e}")
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")
    
    def send_tip(self, fixture_info: Dict, market_suggestion: Dict, confidence: Dict):
        """Send tip synchronously"""
        if not self.bot:
            logger.warning("Telegram bot not available")
            return
        
        try:
            # Run async function in sync context
            asyncio.run(self._send_tip_async(fixture_info, market_suggestion, confidence))
        except RuntimeError as e:
            if "event loop" in str(e):
                # Create new event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(
                        self._send_tip_async(fixture_info, market_suggestion, confidence)
                    )
                finally:
                    loop.close()
            else:
                logger.error(f"Error sending Telegram tip: {e}")
    
    async def _check_bot_async(self):
        """Check bot connection asynchronously"""
        if not self.bot:
            return False
        try:
            await self.bot.get_me()
            return True
        except Exception:
            return False
    
    def check_bot(self):
        """Check bot connection synchronously"""
        if not self.bot:
            return False
        
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self._check_bot_async())
            loop.close()
            return result
        except Exception as e:
            logger.error(f"Error checking Telegram bot: {e}")
            return False
    
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

# Layer 4: Self-Learning Feedback Loop
class SelfLearningFeedbackLoop:
    def store_prediction(self, fixture_id: int, league_id: int, minute: int,
                        predictions: Dict, market_suggestions: List[Dict], db_conn):
        """Store predictions and market suggestions"""
        try:
            cur = db_conn.cursor()
            
            # Store event predictions
            cur.execute("""
                INSERT INTO event_predictions (fixture_id, league_id, minute, predictions)
                VALUES (%s, %s, %s, %s)
            """, (fixture_id, league_id, minute, json.dumps(predictions)))
            
            # Store market suggestions
            for suggestion in market_suggestions:
                cur.execute("""
                    INSERT INTO market_suggestions 
                    (fixture_id, market_type, probability, confidence_score, 
                     expected_value, parameters)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    fixture_id,
                    suggestion['market_type'],
                    suggestion['probability'],
                    suggestion.get('confidence_score', 0),
                    suggestion.get('expected_value', 0),
                    json.dumps(suggestion.get('parameters', {}))
                ))
            
            logger.debug(f"Stored predictions for fixture {fixture_id}")
            
        except Exception as e:
            logger.error(f"Error storing prediction for fixture {fixture_id}: {e}")
            db_conn.rollback()
        finally:
            cur.close()

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
            # Basic validation
            if not match_data or 'fixture' not in match_data:
                logger.warning("Invalid match data received")
                return
            
            fixture_id = match_data['fixture']['id']
            league_id = match_data['league']['id']
            
            # Check match status
            status = match_data['fixture']['status'].get('short', '')
            if status in ['FT', 'AET', 'PEN', 'SUSP', 'PST', 'CANC', 'ABD', 'AWD', 'WO']:
                logger.debug(f"Skipping finished match {fixture_id} (status: {status})")
                return
            
            minute = match_data['fixture']['status'].get('elapsed', 0)
            if minute <= 0:
                logger.debug(f"Skipping match {fixture_id} (minute: {minute})")
                return
            
            logger.info(f"Analyzing match {fixture_id} at minute {minute}")
            
            # Get statistics
            stats = self.api_client.get_match_statistics(fixture_id)
            if not stats:
                logger.debug(f"No statistics for match {fixture_id}")
                return
            
            # Prepare match data
            match_data['statistics'] = stats
            
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
                logger.debug(f"No opportunities for match {fixture_id}")
                return
            
            # Score opportunities
            valuable_opportunities = []
            for opportunity in opportunities:
                confidence = self.value_filter.calculate_confidence_score(
                    opportunity, league_id, minute, probabilities
                )
                
                opportunity['confidence_score'] = confidence['confidence_score']
                opportunity['expected_value'] = confidence['expected_value']
                opportunity['confidence_breakdown'] = confidence
                
                if confidence['final_decision']:
                    valuable_opportunities.append(opportunity)
                    
                    # Send to Telegram
                    self.telegram_notifier.send_tip(match_data, opportunity, confidence)
            
            # Store predictions if we have any opportunities
            if opportunities:
                db_conn = get_db_connection()
                try:
                    self.learning_loop.store_prediction(
                        fixture_id, league_id, minute, probabilities, 
                        opportunities, db_conn
                    )
                finally:
                    db_conn.close()
            
            logger.info(f"Match {fixture_id}: {len(opportunities)} opportunities, "
                       f"{len(valuable_opportunities)} valuable")
            
        except KeyError as e:
            logger.warning(f"Missing key in match data: {e}")
        except Exception as e:
            logger.error(f"Error analyzing match: {e}")
    
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
                    time.sleep(1)  # Rate limiting
                except Exception as e:
                    logger.error(f"Error processing match: {e}")
        
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
            # Import and run training
            from train_models import train_all_models
            db_conn = get_db_connection()
            try:
                results = train_all_models(db_conn)
                logger.info(f"Model training completed: {results}")
            finally:
                db_conn.close()
                
        except ImportError as e:
            logger.error(f"Cannot import train_models: {e}")
        except Exception as e:
            logger.error(f"Error in training: {e}")
        finally:
            self.is_training = False
    
    def daily_digest(self):
        """Generate daily digest"""
        logger.info("Generating daily digest...")
        
        db_conn = get_db_connection()
        try:
            cur = db_conn.cursor()
            
            # Get yesterday's predictions
            yesterday = datetime.now() - timedelta(days=1)
            
            cur.execute("""
                SELECT 
                    COUNT(*) as total_predictions,
                    COUNT(DISTINCT fixture_id) as matches_analyzed,
                    AVG(confidence_score) as avg_confidence
                FROM market_suggestions
                WHERE created_at >= %s
            """, (yesterday,))
            
            stats = cur.fetchone()
            
            # Format message
            message = f"""
📊 DAILY DIGEST - {datetime.now().strftime('%Y-%m-%d')}

🔍 Analysis Summary:
• Total Predictions: {stats[0] or 0}
• Matches Analyzed: {stats[1] or 0}
• Average Confidence: {(stats[2] or 0):.2%}

⚡ System Status: OPERATIONAL
            """
            
            # Try to send via Telegram
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                async def send_digest():
                    await self.telegram_notifier.bot.send_message(
                        chat_id=TELEGRAM_CHAT_ID,
                        text=message,
                        parse_mode='HTML'
                    )
                
                loop.run_until_complete(send_digest())
                loop.close()
                logger.info("Daily digest sent")
            except Exception as e:
                logger.error(f"Error sending daily digest: {e}")
                
        except Exception as e:
            logger.error(f"Error generating daily digest: {e}")
        finally:
            db_conn.close()
    
    def auto_tune(self):
        """Auto-tune parameters based on performance"""
        logger.info("Starting auto-tuning...")
        
        # Simple auto-tuning logic
        adjustments = []
        logger.info("Auto-tuning completed (placeholder)")
        
        return adjustments
    
    def backfill_historical(self, days: int = 30):
        """Backfill historical data"""
        logger.info(f"Backfilling historical data for {days} days...")
        logger.info("Historical backfill placeholder")
    
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
        
        # Check Telegram (without async issues)
        try:
            telegram_ok = self.telegram_notifier.check_bot()
            health['components']['telegram'] = 'healthy' if telegram_ok else 'unhealthy: bot check failed'
            if not telegram_ok:
                health['status'] = 'degraded'
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
    
    logger.info("=" * 50)
    logger.info("MARKET-AGNOSTIC PREDICTION ENGINE v1.0")
    logger.info("=" * 50)
    logger.info("Scheduled tasks initialized:")
    logger.info("- Live scan every 10 minutes")
    logger.info("- Model training at 02:00 daily")
    logger.info("- Daily digest at 08:00 daily")
    logger.info("- Auto-tuning every Sunday at 03:00")
    logger.info("=" * 50)
    
    # Initial health check
    health = engine.health_check()
    logger.info(f"Initial health check: {health}")
    
    # Run initial scan
    logger.info("Running initial scan...")
    engine.live_scan()
    
    # Main loop
    logger.info("Entering main scheduler loop...")
    while True:
        try:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("Shutting down gracefully...")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            time.sleep(300)  # Wait 5 minutes on error

if __name__ == "__main__":
    main()
