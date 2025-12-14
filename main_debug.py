import os
import sys
import time
import schedule
import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any
import psycopg2
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import telegram
from telegram.error import TelegramError

# Load environment variables
load_dotenv()

# Enhanced logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('debug_engine.log')
    ]
)
logger = logging.getLogger(__name__)

# Configuration
DATABASE_URL = os.getenv('DATABASE_URL')
API_FOOTBALL_KEY = os.getenv('API_FOOTBALL_KEY')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# Debug API Client
class DebugAPIFootballClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://v3.football.api-sports.io"
        self.headers = {
            'x-apisports-key': api_key,
            'x-rapidapi-host': 'v3.football.api-sports.io'
        }
    
    def get_live_matches(self) -> List[Dict]:
        """Get all live matches with debug info"""
        url = f"{self.base_url}/fixtures"
        params = {'live': 'all'}
        try:
            logger.debug(f"Fetching live matches from {url}")
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            logger.debug(f"API Response Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                logger.debug(f"API Response Keys: {list(data.keys())}")
                
                if 'response' in data:
                    matches = data['response']
                    logger.info(f"Found {len(matches)} live matches")
                    
                    # Log match details
                    for match in matches:
                        fixture_id = match.get('fixture', {}).get('id', 'N/A')
                        home = match.get('teams', {}).get('home', {}).get('name', 'Home')
                        away = match.get('teams', {}).get('away', {}).get('name', 'Away')
                        minute = match.get('fixture', {}).get('status', {}).get('elapsed', 0)
                        league = match.get('league', {}).get('name', 'Unknown')
                        logger.debug(f"Match: {home} vs {away} (ID: {fixture_id}, Minute: {minute}, League: {league})")
                    
                    return matches
                else:
                    logger.error(f"No 'response' key in API data: {data}")
                    return []
            else:
                logger.error(f"API Error {response.status_code}: {response.text}")
                return []
        except Exception as e:
            logger.error(f"Error fetching live matches: {e}")
            return []
    
    def get_match_statistics(self, fixture_id: int) -> Dict:
        """Get match statistics with debug"""
        url = f"{self.base_url}/fixtures/statistics"
        params = {'fixture': fixture_id}
        try:
            logger.debug(f"Fetching stats for fixture {fixture_id}")
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                logger.debug(f"Stats API Response for {fixture_id}: {data}")
                
                if 'response' in data and data['response']:
                    stats = self._parse_statistics(data['response'])
                    logger.debug(f"Parsed stats keys: {list(stats.keys())}")
                    return stats
                else:
                    logger.warning(f"No statistics data for fixture {fixture_id}")
                    return {}
            else:
                logger.warning(f"Stats API error for {fixture_id}: {response.status_code}")
                return {}
        except Exception as e:
            logger.error(f"Error getting stats for {fixture_id}: {e}")
            return {}
    
    def _parse_statistics(self, stats_data: List) -> Dict:
        """Parse statistics"""
        parsed = {}
        for team_stats in stats_data:
            team = team_stats.get('team', {})
            team_id = team.get('id')
            if team_id:
                stats = {}
                for stat in team_stats.get('statistics', []):
                    stat_type = stat.get('type')
                    value = stat.get('value', 0)
                    stats[stat_type] = value
                parsed[team_id] = stats
                logger.debug(f"Team {team_id} stats: {list(stats.keys())}")
        return parsed

# Simple Probability Engine for Debug
class DebugProbabilityEngine:
    def extract_features(self, match_data: Dict, minute: int) -> Dict:
        """Extract simplified features"""
        try:
            logger.debug("Extracting features...")
            
            # Basic match info
            teams = match_data.get('teams', {})
            home_team = teams.get('home', {})
            away_team = teams.get('away', {})
            
            home_id = home_team.get('id')
            away_id = away_team.get('id')
            
            # Get statistics
            stats = match_data.get('statistics', {})
            home_stats = stats.get(home_id, {})
            away_stats = stats.get(away_id, {})
            
            # Get scores
            goals = match_data.get('goals', {})
            home_score = goals.get('home', 0) or 0
            away_score = goals.get('away', 0) or 0
            
            logger.debug(f"Scores: {home_score}-{away_score}")
            logger.debug(f"Home stats available: {len(home_stats)} items")
            logger.debug(f"Away stats available: {len(away_stats)} items")
            
            # Extract key stats with defaults
            features = {
                'minute': minute,
                'home_score': home_score,
                'away_score': away_score,
                'score_delta': home_score - away_score,
                
                # Attempt to get xG if available
                'home_xg': self._get_float_stat(home_stats, 'expected_goals', 0),
                'away_xg': self._get_float_stat(away_stats, 'expected_goals', 0),
                
                # Shots
                'home_shots_on': self._get_int_stat(home_stats, 'shots on goal', 0),
                'away_shots_on': self._get_int_stat(away_stats, 'shots on goal', 0),
                
                # Possession
                'home_possession': self._parse_possession(home_stats.get('ball possession', '50%')),
                
                # Corners
                'home_corners': self._get_int_stat(home_stats, 'corner kicks', 0),
                'away_corners': self._get_int_stat(away_stats, 'corner kicks', 0),
                
                # Cards
                'home_yellow': self._get_int_stat(home_stats, 'yellow cards', 0),
                'away_yellow': self._get_int_stat(away_stats, 'yellow cards', 0),
            }
            
            # Log extracted features
            logger.debug(f"Extracted features: {json.dumps(features, indent=2)}")
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return {}
    
    def _get_float_stat(self, stats: Dict, key: str, default: float) -> float:
        """Get float statistic"""
        value = stats.get(key, default)
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, str):
            try:
                return float(value)
            except:
                return default
        return default
    
    def _get_int_stat(self, stats: Dict, key: str, default: int) -> int:
        """Get integer statistic"""
        value = stats.get(key, default)
        if isinstance(value, (int, float)):
            return int(value)
        elif isinstance(value, str):
            try:
                return int(float(value))
            except:
                return default
        return default
    
    def _parse_possession(self, possession: Any) -> float:
        """Parse possession value"""
        if isinstance(possession, (int, float)):
            if possession > 1:
                return possession / 100
            return possession
        elif isinstance(possession, str):
            if '%' in possession:
                try:
                    return float(possession.strip('%')) / 100
                except:
                    return 0.5
            else:
                try:
                    return float(possession)
                except:
                    return 0.5
        return 0.5
    
    def calculate_probabilities(self, features: Dict) -> Dict:
        """Calculate simple probabilities"""
        try:
            minute = features.get('minute', 0)
            home_score = features.get('home_score', 0)
            away_score = features.get('away_score', 0)
            home_xg = features.get('home_xg', 0)
            away_xg = features.get('away_xg', 0)
            
            total_goals = home_score + away_score
            xg_total = home_xg + away_xg
            
            # Simple probability calculations
            time_left = max(90 - minute, 0)
            base_goal_rate = 0.025  # 2.5% per minute
            
            probabilities = {
                'goal_next_10min': min(0.8, base_goal_rate * 10 + (xg_total / max(minute, 1)) * 5),
                'home_goal_probability': 0.4 + (home_xg * 0.3) - (max(home_score - away_score, 0) * 0.1),
                'away_goal_probability': 0.4 + (away_xg * 0.3) - (max(away_score - home_score, 0) * 0.1),
                'expected_final_goals': total_goals + (xg_total * time_left / 90),
                'corner_next_10min': 0.3,  # Default
                'yellow_card_next_10min': 0.2,  # Default
            }
            
            logger.debug(f"Calculated probabilities: {probabilities}")
            return probabilities
            
        except Exception as e:
            logger.error(f"Error calculating probabilities: {e}")
            return {}

# Debug Market Generator
class DebugMarketGenerator:
    def generate_opportunities(self, probabilities: Dict, features: Dict) -> List[Dict]:
        """Generate market opportunities with debug"""
        opportunities = []
        
        # Check for next goal
        if probabilities.get('goal_next_10min', 0) > 0.3:
            opportunities.append({
                'market_type': 'next_goal',
                'description': 'Next Goal',
                'probability': probabilities['goal_next_10min'],
                'current_state': features
            })
        
        # Check for over 2.5
        if probabilities.get('expected_final_goals', 0) >= 2.5:
            opportunities.append({
                'market_type': 'over_2.5',
                'description': 'Over 2.5 Goals',
                'probability': min(0.8, probabilities['expected_final_goals'] / 3),
                'current_state': features
            })
        
        # Check for home to score
        if probabilities.get('home_goal_probability', 0) > 0.5:
            opportunities.append({
                'market_type': 'home_to_score',
                'description': 'Home Team to Score',
                'probability': probabilities['home_goal_probability'],
                'current_state': features
            })
        
        logger.debug(f"Generated {len(opportunities)} opportunities")
        return opportunities

# Debug Telegram Notifier
class DebugTelegramNotifier:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        
        # Test connection
        self.test_connection()
    
    def test_connection(self):
        """Test Telegram connection"""
        try:
            bot = telegram.Bot(token=self.token)
            # Get bot info synchronously
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            bot_info = loop.run_until_complete(bot.get_me())
            loop.close()
            
            logger.info(f"✅ Telegram Bot Connected: @{bot_info.username}")
            return True
        except Exception as e:
            logger.error(f"❌ Telegram connection failed: {e}")
            return False
    
    def send_debug_info(self, match_count: int, opportunities: List[Dict]):
        """Send debug info to Telegram"""
        try:
            bot = telegram.Bot(token=self.token)
            
            message = f"""
🔍 DEBUG REPORT - Market-Agnostic Engine

📊 System Status: ACTIVE
⚽ Live Matches Found: {match_count}
🎯 Opportunities Generated: {len(opportunities)}

🔄 Next scan in 10 minutes

#Debug #PredictionEngine
            """
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(
                bot.send_message(chat_id=self.chat_id, text=message, parse_mode='HTML')
            )
            loop.close()
            
            logger.info("Debug info sent to Telegram")
            
        except Exception as e:
            logger.error(f"Failed to send debug info: {e}")

# Main Debug Engine
class DebugPredictionEngine:
    def __init__(self):
        self.api_client = DebugAPIFootballClient(API_FOOTBALL_KEY)
        self.prob_engine = DebugProbabilityEngine()
        self.market_generator = DebugMarketGenerator()
        self.telegram_notifier = DebugTelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        
        logger.info("=" * 60)
        logger.info("DEBUG MODE ACTIVATED")
        logger.info("=" * 60)
    
    def run_debug_scan(self):
        """Run a debug scan to identify issues"""
        logger.info("🚀 Starting debug scan...")
        
        # Step 1: Get live matches
        live_matches = self.api_client.get_live_matches()
        
        if not live_matches:
            logger.error("❌ No live matches found!")
            return
        
        all_opportunities = []
        
        # Step 2: Analyze each match
        for match in live_matches[:2]:  # Limit to 2 for debugging
            try:
                fixture_id = match.get('fixture', {}).get('id')
                league = match.get('league', {}).get('name', 'Unknown')
                home = match.get('teams', {}).get('home', {}).get('name', 'Home')
                away = match.get('teams', {}).get('away', {}).get('name', 'Away')
                minute = match.get('fixture', {}).get('status', {}).get('elapsed', 0)
                
                logger.info(f"\n🔍 Analyzing: {home} vs {away}")
                logger.info(f"   League: {league}, Minute: {minute}, ID: {fixture_id}")
                
                # Step 3: Get statistics
                stats = self.api_client.get_match_statistics(fixture_id)
                
                if not stats:
                    logger.warning(f"   ⚠️ No statistics available for this match")
                    continue
                
                # Add stats to match data
                match['statistics'] = stats
                
                # Step 4: Extract features
                features = self.prob_engine.extract_features(match, minute)
                
                if not features:
                    logger.warning(f"   ⚠️ Could not extract features")
                    continue
                
                # Step 5: Calculate probabilities
                probabilities = self.prob_engine.calculate_probabilities(features)
                
                if not probabilities:
                    logger.warning(f"   ⚠️ Could not calculate probabilities")
                    continue
                
                # Step 6: Generate opportunities
                opportunities = self.market_generator.generate_opportunities(probabilities, features)
                
                logger.info(f"   ✅ Found {len(opportunities)} opportunities")
                
                for opp in opportunities:
                    logger.info(f"     • {opp['description']}: {opp['probability']:.1%}")
                    all_opportunities.append(opp)
                
            except Exception as e:
                logger.error(f"   ❌ Error analyzing match: {e}")
                continue
        
        # Step 7: Send debug report
        if all_opportunities:
            logger.info(f"\n🎯 Total opportunities found: {len(all_opportunities)}")
            
            # Send to Telegram if we found opportunities
            self.telegram_notifier.send_debug_info(len(live_matches), all_opportunities)
            
            # Also send actual tips
            self.send_telegram_tips(all_opportunities[:3])  # Send top 3
        else:
            logger.info(f"\n📭 No opportunities found in this scan")
            
            # Still send debug report
            self.telegram_notifier.send_debug_info(len(live_matches), [])
    
    def send_telegram_tips(self, opportunities: List[Dict]):
        """Send actual tips to Telegram"""
        try:
            bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
            
            for opp in opportunities:
                message = f"""
🎯 PREDICTION FOUND!

Market: {opp['description']}
Probability: {opp['probability']:.1%}
Confidence: High ⭐

⚡ Action: Consider this opportunity

#LiveBetting #FootballPrediction
                """
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(
                    bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode='HTML')
                )
                loop.close()
                
                logger.info(f"Tip sent: {opp['description']}")
                time.sleep(1)  # Rate limiting
                
        except Exception as e:
            logger.error(f"Error sending tips: {e}")

# Run debug
def main():
    logger.info("Starting Market-Agnostic Prediction Engine - DEBUG MODE")
    
    # Check environment variables
    required_vars = ['DATABASE_URL', 'API_FOOTBALL_KEY', 'TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing environment variables: {missing_vars}")
        return
    
    logger.info("✅ All environment variables are set")
    
    # Create and run debug engine
    engine = DebugPredictionEngine()
    
    # Run initial debug scan
    engine.run_debug_scan()
    
    # Schedule regular scans
    schedule.every(10).minutes.do(engine.run_debug_scan)
    
    logger.info("\n📅 Scheduler started. Next scan in 10 minutes...")
    logger.info("Press Ctrl+C to stop\n")
    
    # Run scheduler
    while True:
        try:
            schedule.run_pending()
            time.sleep(60)
        except KeyboardInterrupt:
            logger.info("\n👋 Shutting down debug engine...")
            break
        except Exception as e:
            logger.error(f"Scheduler error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
