from fastapi import FastAPI, BackgroundTasks, HTTPException
from contextlib import asynccontextmanager
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import pickle
import json
from dataclasses import dataclass, asdict
import numpy as np
from supabase import create_client, Client
import telegram
from telegram.ext import Application
import requests
import os
from scipy.stats import poisson, norm
import betfairlightweight
from betfairlightweight import filters
from train_models import AdvancedPredictor, MarketAnalyzer, EnsembleModel, FeatureEngineer
import aiohttp
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
predictor: Optional[AdvancedPredictor] = None
supabase: Optional[Client] = None
telegram_bot: Optional[telegram.Bot] = None
betfair_trading: Optional[betfairlightweight.APIClient] = None
scheduler = AsyncIOScheduler()

@dataclass
class MatchPrediction:
    fixture_id: int
    home_team: str
    away_team: str
    league: str
    timestamp: datetime
    predictions: Dict[str, dict]
    recommended_market: str
    confidence: float
    expected_value: float
    kelly_criterion: float
    model_version: str
    features: Dict[str, float]
    weather_impact: float
    market_odds: Dict[str, float]
    
@dataclass
class ModelMetrics:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    profit_margin: float
    sharpe_ratio: float
    max_drawdown: float
    timestamp: datetime

class DataCollector:
    """Collects data from API-Football with intelligent scheduling"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://v3.football.api-sports.io"
        self.headers = {
            'x-apisports-key': api_key,
            'x-rapidapi-host': 'v3.football.api-sports.io'
        }
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def initialize(self):
        """Initialize async session"""
        self.session = aiohttp.ClientSession(headers=self.headers)
        
    async def fetch_live_fixtures(self) -> List[Dict]:
        """Fetch current day fixtures with intelligent retry logic"""
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
            
            async with self.session.get(
                f"{self.base_url}/fixtures",
                params={
                    'date': today,
                    'timezone': 'UTC',
                    'status': 'NS-TBD'  # Not Started - To Be Defined
                }
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('response', [])
                else:
                    logger.error(f"API Error: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error fetching fixtures: {str(e)}")
            return []
    
    async def fetch_fixture_stats(self, fixture_id: int) -> Dict:
        """Get detailed statistics for a specific fixture"""
        try:
            endpoints = ['fixtures/statistics', 'fixtures/events', 
                        'fixtures/lineups', 'fixtures/players']
            all_stats = {}
            
            for endpoint in endpoints:
                async with self.session.get(
                    f"{self.base_url}/{endpoint}",
                    params={'fixture': fixture_id}
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        all_stats[endpoint.split('/')[-1]] = data.get('response', {})
                        
            return all_stats
        except Exception as e:
            logger.error(f"Error fetching stats: {str(e)}")
            return {}
    
    async def fetch_historical_data(self, team_id: int, season: int) -> List[Dict]:
        """Fetch historical match data for a team"""
        try:
            async with self.session.get(
                f"{self.base_url}/fixtures",
                params={
                    'team': team_id,
                    'season': season,
                    'last': 20  # Last 20 matches
                }
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    return data.get('response', [])
                return []
        except Exception as e:
            logger.error(f"Error fetching historical: {str(e)}")
            return []
    
    async def fetch_team_news(self, team_id: int) -> List[Dict]:
        """Fetch team news, injuries, and suspensions"""
        try:
            async with self.session.get(
                f"{self.base_url}/injuries",
                params={'team': team_id, 'season': datetime.now().year}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('response', [])
                return []
        except Exception as e:
            logger.error(f"Error fetching team news: {str(e)}")
            return []
    
    async def fetch_weather_data(self, city: str, country: str, match_time: datetime) -> Dict:
        """Fetch weather data for match location"""
        try:
            api_key = os.getenv('OPENWEATHER_API_KEY')
            if not api_key:
                return {'condition': 'Clear', 'temperature': 20, 'humidity': 50, 'wind_speed': 5}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://api.openweathermap.org/data/2.5/forecast",
                    params={
                        'q': f"{city},{country}",
                        'appid': api_key,
                        'units': 'metric'
                    }
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        # Find closest forecast to match time
                        forecasts = data.get('list', [])
                        closest = min(forecasts, key=lambda x: abs(
                            datetime.fromtimestamp(x['dt']) - match_time
                        ))
                        return {
                            'condition': closest['weather'][0]['main'],
                            'temperature': closest['main']['temp'],
                            'humidity': closest['main']['humidity'],
                            'wind_speed': closest['wind']['speed']
                        }
                    return {'condition': 'Clear', 'temperature': 20, 'humidity': 50, 'wind_speed': 5}
        except Exception as e:
            logger.error(f"Error fetching weather: {str(e)}")
            return {'condition': 'Clear', 'temperature': 20, 'humidity': 50, 'wind_speed': 5}

class BetfairOddsCollector:
    """Collect real-time odds from Betfair exchange"""
    
    def __init__(self, username: str, password: str, app_key: str):
        self.username = username
        self.password = password
        self.app_key = app_key
        self.trading = None
        self.market_cache = {}
        
    async def initialize(self):
        """Initialize Betfair connection"""
        try:
            self.trading = betfairlightweight.APIClient(
                username=self.username,
                password=self.password,
                app_key=self.app_key
            )
            await asyncio.to_thread(self.trading.login)
            logger.info("Betfair connection established")
        except Exception as e:
            logger.error(f"Error connecting to Betfair: {str(e)}")
    
    async def get_market_odds(self, event_id: str) -> Dict[str, float]:
        """Get current market odds for an event"""
        try:
            market_filter = filters.market_filter(event_ids=[event_id])
            market_catalogue = await asyncio.to_thread(
                self.trading.betting.list_market_catalogue,
                filter=market_filter,
                max_results=10,
                market_projection=['RUNNER_DESCRIPTION', 'MARKET_START_TIME', 'COMPETITION']
            )
            
            odds_data = {}
            for market in market_catalogue:
                market_id = market.market_id
                market_book = await asyncio.to_thread(
                    self.trading.betting.list_market_book,
                    market_ids=[market_id],
                    price_projection=filters.price_projection(
                        price_data=filters.price_data(ex_best_offers=True)
                    )
                )
                
                if market_book:
                    runners = market_book[0].runners
                    for runner in runners:
                        if runner.ex.available_to_back:
                            best_price = runner.ex.available_to_back[0].price
                            odds_data[market.market_name] = best_price
            
            return odds_data
        except Exception as e:
            logger.error(f"Error fetching Betfair odds: {str(e)}")
            return {}

class PredictionEngine:
    """Orchestrates the entire prediction pipeline"""
    
    def __init__(self, supabase_client: Client, telegram_bot_token: str):
        self.supabase = supabase_client
        self.telegram_bot = telegram.Bot(token=telegram_bot_token)
        self.data_collector = None
        self.betfair_collector = None
        self.predictor = AdvancedPredictor()
        self.market_analyzer = MarketAnalyzer()
        self.ensemble = EnsembleModel()
        self.feature_engineer = FeatureEngineer()
        self.model_version = "1.0.0"
        self.bankroll = 10000.0  # Starting bankroll
        self.stake_history = []
        self.performance_metrics = {
            'total_bets': 0,
            'wins': 0,
            'losses': 0,
            'total_staked': 0.0,
            'total_return': 0.0,
            'current_bankroll': 10000.0
        }
        
    async def initialize(self, api_football_key: str):
        """Initialize all components"""
        self.data_collector = DataCollector(api_football_key)
        await self.data_collector.initialize()
        
        # Initialize Betfair if credentials available
        if all([os.getenv('BETFAIR_USERNAME'), os.getenv('BETFAIR_PASSWORD'), os.getenv('BETFAIR_APP_KEY')]):
            self.betfair_collector = BetfairOddsCollector(
                os.getenv('BETFAIR_USERNAME'),
                os.getenv('BETFAIR_PASSWORD'),
                os.getenv('BETFAIR_APP_KEY')
            )
            await self.betfair_collector.initialize()
        
        # Load or train initial model
        await self.load_or_train_model()
        
        # Start prediction cycle
        asyncio.create_task(self.prediction_cycle())
        
    async def load_or_train_model(self):
        """Load existing model or train a new one"""
        try:
            # Try to load latest model from Supabase
            response = self.supabase.table('models')\
                .select('*')\
                .order('created_at', desc=True)\
                .limit(1)\
                .execute()
            
            if response.data:
                model_data = response.data[0]
                self.predictor = pickle.loads(bytes(model_data['model_data']))
                self.model_version = model_data['version']
                logger.info(f"Loaded model version {self.model_version}")
            else:
                await self.train_and_save_model()
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            await self.train_and_save_model()
    
    async def train_and_save_model(self):
        """Train model with latest data and save to Supabase"""
        try:
            # Fetch training data
            training_data = await self.fetch_training_data()
            
            if len(training_data) < 100:
                logger.warning(f"Insufficient training data: {len(training_data)} samples")
                return
            
            # Train model
            self.predictor.train(training_data)
            
            # Create new version
            self.model_version = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Save to Supabase
            model_bytes = pickle.dumps(self.predictor)
            self.supabase.table('models').insert({
                'version': self.model_version,
                'model_data': model_bytes,
                'accuracy': self.predictor.metrics.accuracy,
                'precision': self.predictor.metrics.precision,
                'recall': self.predictor.metrics.recall,
                'f1_score': self.predictor.metrics.f1,
                'roc_auc': self.predictor.metrics.roc_auc,
                'training_size': len(training_data),
                'created_at': datetime.now().isoformat()
            }).execute()
            
            logger.info(f"Trained and saved model {self.model_version} with {len(training_data)} samples")
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
    
    async def fetch_training_data(self) -> pd.DataFrame:
        """Fetch training data from Supabase"""
        try:
            response = self.supabase.table('historical_predictions')\
                .select('*')\
                .limit(10000)\
                .execute()
            
            if not response.data:
                return pd.DataFrame()
            
            df = pd.DataFrame(response.data)
            
            # Convert JSON fields
            json_columns = ['features', 'market_odds', 'weather_data', 'team_news']
            for col in json_columns:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: json.loads(x) if x else {})
            
            return df
        except Exception as e:
            logger.error(f"Error fetching training data: {str(e)}")
            return pd.DataFrame()
    
    async def prediction_cycle(self):
        """Main prediction cycle"""
        while True:
            try:
                # Fetch current fixtures
                fixtures = await self.data_collector.fetch_live_fixtures()
                
                predictions = []
                for fixture in fixtures:
                    # Make prediction
                    prediction = await self.analyze_fixture(fixture)
                    predictions.append(prediction)
                    
                    # Store prediction
                    await self.store_prediction(prediction)
                    
                    # Calculate optimal stake
                    stake = self.calculate_stake(prediction)
                    
                    # Send to Telegram if confidence > threshold
                    if prediction.confidence > 0.65 and stake > 0:
                        await self.send_telegram_alert(prediction, stake)
                
                # Update model with results if available
                await self.update_model_with_results()
                
                # Update performance metrics
                await self.update_performance_metrics()
                
                # Schedule next cycle
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Prediction cycle error: {str(e)}")
                await asyncio.sleep(60)
    
    async def analyze_fixture(self, fixture: Dict) -> MatchPrediction:
        """Analyze a single fixture and generate predictions"""
        try:
            # Extract features
            features = await self.extract_features(fixture)
            
            # Fetch market odds
            market_odds = await self.fetch_market_odds(fixture)
            
            # Generate predictions for all markets
            market_predictions = {}
            for market in ['1X2', 'Over/Under', 'BTTS', 'Asian Handicap', 'Correct Score']:
                pred = self.predictor.predict(features, market, market_odds)
                market_predictions[market] = pred
            
            # Analyze best market
            market_analysis = self.market_analyzer.analyze(market_predictions)
            
            # Apply ensemble voting
            final_prediction = self.ensemble.vote(market_predictions)
            
            # Get weather impact
            weather_impact = await self.calculate_weather_impact(fixture)
            
            # Calculate Kelly criterion for bankroll management
            kelly = self.calculate_kelly_criterion(
                final_prediction['probability'],
                market_odds.get(market_analysis['best_market'], 2.0)
            )
            
            return MatchPrediction(
                fixture_id=fixture['fixture']['id'],
                home_team=fixture['teams']['home']['name'],
                away_team=fixture['teams']['away']['name'],
                league=fixture['league']['name'],
                timestamp=datetime.fromtimestamp(fixture['fixture']['timestamp']),
                predictions=market_predictions,
                recommended_market=market_analysis['best_market'],
                confidence=final_prediction['confidence'],
                expected_value=final_prediction['expected_value'],
                kelly_criterion=kelly,
                model_version=self.model_version,
                features=features,
                weather_impact=weather_impact,
                market_odds=market_odds
            )
            
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            raise
    
    async def extract_features(self, fixture: Dict) -> Dict[str, float]:
        """Extract advanced features from fixture data"""
        features = {}
        
        # Basic team features
        home_id = fixture['teams']['home']['id']
        away_id = fixture['teams']['away']['id']
        
        features['home_win_rate'] = await self.calculate_win_rate(home_id)
        features['away_win_rate'] = await self.calculate_win_rate(away_id)
        features['home_goal_avg'] = await self.calculate_goal_average(home_id, 'for')
        features['away_goal_avg'] = await self.calculate_goal_average(away_id, 'for')
        features['home_concede_avg'] = await self.calculate_goal_average(home_id, 'against')
        features['away_concede_avg'] = await self.calculate_goal_average(away_id, 'against')
        
        # Form features with exponential weighting
        features['home_form'] = await self.get_weighted_form(home_id)
        features['away_form'] = await self.get_weighted_form(away_id)
        
        # Head to head
        features['h2h_advantage'] = await self.get_h2h_advantage(home_id, away_id)
        
        # Poisson distribution parameters
        lambda_home = features['home_goal_avg'] * (features['away_concede_avg'] / league_avg)
        lambda_away = features['away_goal_avg'] * (features['home_concede_avg'] / league_avg)
        features['poisson_home'] = lambda_home
        features['poisson_away'] = lambda_away
        
        # Market efficiency features
        features['market_efficiency'] = await self.calculate_market_efficiency(fixture['fixture']['id'])
        
        # Team news impact
        features['team_news_impact'] = await self.calculate_team_news_impact(home_id, away_id)
        
        return features
    
    async def calculate_win_rate(self, team_id: int) -> float:
        """Calculate team's win rate with smoothing"""
        try:
            response = self.supabase.table('team_stats')\
                .select('wins', 'draws', 'losses')\
                .eq('team_id', team_id)\
                .execute()
            
            if response.data:
                stats = response.data[0]
                total = stats['wins'] + stats['draws'] + stats['losses']
                if total > 0:
                    # Add-1 smoothing for small sample sizes
                    return (stats['wins'] + 1) / (total + 3)
            return 0.33  # Default for unknown teams
        except Exception as e:
            logger.error(f"Error calculating win rate: {str(e)}")
            return 0.33
    
    async def calculate_goal_average(self, team_id: int, stat_type: str) -> float:
        """Calculate average goals for/against with smoothing"""
        try:
            response = self.supabase.table('team_goal_stats')\
                .select('goals_for', 'goals_against', 'matches_played')\
                .eq('team_id', team_id)\
                .execute()
            
            if response.data:
                stats = response.data[0]
                total_matches = stats['matches_played']
                if total_matches > 0:
                    if stat_type == 'for':
                        return (stats['goals_for'] + 1) / (total_matches + 2)
                    else:
                        return (stats['goals_against'] + 1) / (total_matches + 2)
            return 1.0  # League average default
        except Exception as e:
            logger.error(f"Error calculating goal average: {str(e)}")
            return 1.0
    
    async def get_weighted_form(self, team_id: int) -> float:
        """Get team's weighted form (exponential decay)"""
        try:
            response = self.supabase.table('recent_matches')\
                .select('result', 'match_date', 'is_home')\
                .eq('team_id', team_id)\
                .order('match_date', desc=True)\
                .limit(10)\
                .execute()
            
            if not response.data:
                return 0.5
            
            total_weight = 0
            weighted_sum = 0
            
            for i, match in enumerate(response.data):
                weight = math.exp(-i * 0.3)  # Exponential decay
                
                if match['result'] == 'W':
                    weighted_sum += 1.0 * weight
                elif match['result'] == 'D':
                    weighted_sum += 0.5 * weight
                else:
                    weighted_sum += 0.0 * weight
                
                total_weight += weight
            
            return weighted_sum / total_weight if total_weight > 0 else 0.5
        except Exception as e:
            logger.error(f"Error calculating weighted form: {str(e)}")
            return 0.5
    
    async def get_h2h_advantage(self, home_id: int, away_id: int) -> float:
        """Calculate head-to-head advantage with smoothing"""
        try:
            response = self.supabase.table('head_to_head')\
                .select('home_wins', 'away_wins', 'draws')\
                .eq('home_team_id', home_id)\
                .eq('away_team_id', away_id)\
                .execute()
            
            if response.data:
                h2h = response.data[0]
                total = h2h['home_wins'] + h2h['away_wins'] + h2h['draws']
                if total > 0:
                    # Add-1 smoothing
                    return (h2h['home_wins'] + 1) / (total + 3)
            return 0.5
        except Exception as e:
            logger.error(f"Error calculating H2H: {str(e)}")
            return 0.5
    
    async def calculate_market_efficiency(self, fixture_id: int) -> float:
        """Calculate market efficiency score"""
        try:
            response = self.supabase.table('market_odds_history')\
                .select('odds', 'timestamp', 'volume')\
                .eq('fixture_id', fixture_id)\
                .order('timestamp', desc=True)\
                .limit(20)\
                .execute()
            
            if len(response.data) < 5:
                return 0.5
            
            # Calculate efficiency based on odds movement and volume
            odds_series = [x['odds'] for x in response.data]
            volume_series = [x['volume'] for x in response.data]
            
            # Simple efficiency metric (could be more sophisticated)
            odds_std = np.std(odds_series)
            volume_mean = np.mean(volume_series)
            
            efficiency = 1 / (1 + odds_std) * (volume_mean / 10000)
            return min(efficiency, 1.0)
        except Exception as e:
            logger.error(f"Error calculating market efficiency: {str(e)}")
            return 0.5
    
    async def calculate_team_news_impact(self, home_id: int, away_id: int) -> float:
        """Calculate impact of team news on match outcome"""
        try:
            home_news = await self.data_collector.fetch_team_news(home_id)
            away_news = await self.data_collector.fetch_team_news(away_id)
            
            impact = 0.0
            
            # Calculate impact based on player importance and news type
            for news in home_news:
                if news['type'] == 'injured':
                    impact -= 0.1  # Negative impact for home injuries
                elif news['type'] == 'suspended':
                    impact -= 0.15
            
            for news in away_news:
                if news['type'] == 'injured':
                    impact += 0.1  # Positive impact for away injuries
                elif news['type'] == 'suspended':
                    impact += 0.15
            
            return max(min(impact, 0.3), -0.3)  # Cap at ±30% impact
        except Exception as e:
            logger.error(f"Error calculating team news impact: {str(e)}")
            return 0.0
    
    async def fetch_market_odds(self, fixture: Dict) -> Dict[str, float]:
        """Fetch market odds from Betfair or calculate implied odds"""
        market_odds = {}
        
        # Try Betfair first
        if self.betfair_collector:
            try:
                event_id = f"EVENT_{fixture['fixture']['id']}"
                betfair_odds = await self.betfair_collector.get_market_odds(event_id)
                market_odds.update(betfair_odds)
            except Exception as e:
                logger.warning(f"Could not fetch Betfair odds: {str(e)}")
        
        # Calculate implied odds from predictions as fallback
        if not market_odds:
            # Get basic 1X2 odds from prediction
            home_win_prob = await self.calculate_win_rate(fixture['teams']['home']['id'])
            away_win_prob = await self.calculate_win_rate(fixture['teams']['away']['id'])
            draw_prob = 1 - home_win_prob - away_win_prob
            
            if home_win_prob > 0:
                market_odds['home_win'] = 1.0 / home_win_prob * 0.95  # Apply 5% margin
            if away_win_prob > 0:
                market_odds['away_win'] = 1.0 / away_win_prob * 0.95
            if draw_prob > 0:
                market_odds['draw'] = 1.0 / draw_prob * 0.95
        
        return market_odds
    
    async def calculate_weather_impact(self, fixture: Dict) -> float:
        """Calculate weather impact on match"""
        try:
            venue = fixture.get('fixture', {}).get('venue', {})
            city = venue.get('city', 'London')
            country = fixture.get('league', {}).get('country', 'England')
            match_time = datetime.fromtimestamp(fixture['fixture']['timestamp'])
            
            weather_data = await self.data_collector.fetch_weather_data(city, country, match_time)
            
            # Calculate impact based on weather conditions
            condition = weather_data.get('condition', 'Clear').lower()
            temp = weather_data.get('temperature', 20)
            wind = weather_data.get('wind_speed', 0)
            humidity = weather_data.get('humidity', 50)
            
            impact = 0.0
            
            # Temperature impact (optimal around 20°C)
            temp_impact = -abs(temp - 20) * 0.005
            impact += temp_impact
            
            # Wind impact
            wind_impact = -min(wind * 0.01, 0.1)
            impact += wind_impact
            
            # Condition impact
            condition_impacts = {
                'clear': 0.0,
                'clouds': -0.02,
                'rain': -0.05,
                'snow': -0.1,
                'thunderstorm': -0.15
            }
            impact += condition_impacts.get(condition, 0.0)
            
            return max(min(impact, 0.1), -0.1)  # Cap at ±10% impact
        except Exception as e:
            logger.error(f"Error calculating weather impact: {str(e)}")
            return 0.0
    
    def calculate_kelly_criterion(self, probability: float, odds: float) -> float:
        """
        Calculate Kelly Criterion: f* = (p * (b + 1) - 1) / b
        Where:
        - p = probability of winning
        - b = decimal odds - 1
        """
        if odds <= 1:
            return 0.0
        
        b = odds - 1
        kelly_raw = (probability * (b + 1) - 1) / b
        
        # Apply fractional Kelly (quarter Kelly) with bounds
        kelly_fractional = kelly_raw * 0.25
        
        # Apply sensible bounds: 0% to 5% of bankroll
        kelly_bounded = max(0.0, min(kelly_fractional, 0.05))
        
        return kelly_bounded
    
    def calculate_stake(self, prediction: MatchPrediction) -> float:
        """Calculate stake amount using Kelly Criterion"""
        odds = prediction.market_odds.get(prediction.recommended_market, 2.0)
        
        if odds <= 1 or prediction.kelly_criterion <= 0:
            return 0.0
        
        stake = self.bankroll * prediction.kelly_criterion
        
        # Apply minimum and maximum stake limits
        min_stake = 10.0
        max_stake = 1000.0
        
        if stake < min_stake:
            return 0.0
        elif stake > max_stake:
            return max_stake
        else:
            return stake
    
    async def store_prediction(self, prediction: MatchPrediction):
        """Store prediction in Supabase"""
        try:
            self.supabase.table('predictions').insert({
                'fixture_id': prediction.fixture_id,
                'home_team': prediction.home_team,
                'away_team': prediction.away_team,
                'league': prediction.league,
                'match_time': prediction.timestamp.isoformat(),
                'predicted_market': prediction.recommended_market,
                'confidence': prediction.confidence,
                'expected_value': prediction.expected_value,
                'kelly_criterion': prediction.kelly_criterion,
                'model_version': prediction.model_version,
                'all_predictions': json.dumps(prediction.predictions),
                'features': json.dumps(prediction.features),
                'market_odds': json.dumps(prediction.market_odds),
                'weather_impact': prediction.weather_impact,
                'created_at': datetime.now().isoformat()
            }).execute()
        except Exception as e:
            logger.error(f"Error storing prediction: {str(e)}")
    
    async def send_telegram_alert(self, prediction: MatchPrediction, stake: float):
        """Send prediction alert to Telegram"""
        try:
            # Format odds for display
            odds = prediction.market_odds.get(prediction.recommended_market, 2.0)
            
            # Calculate potential returns
            potential_return = stake * odds
            potential_profit = potential_return - stake
            
            message = f"""
            ⚽️ *NEW PREDICTION ALERT* ⚽️
            
            🏆 *Match*: {prediction.home_team} vs {prediction.away_team}
            📊 *League*: {prediction.league}
            ⏰ *Time*: {prediction.timestamp.strftime('%Y-%m-%d %H:%M')}
            
            🎯 *Recommended Market*: {prediction.recommended_market}
            📈 *Odds*: {odds:.2f}
            
            💪 *Confidence*: {prediction.confidence:.1%}
            📊 *Expected Value*: {prediction.expected_value:+.2%}
            🎲 *Kelly Bet*: {prediction.kelly_criterion:.1%} of bankroll
            💰 *Recommended Stake*: £{stake:.2f}
            💵 *Potential Profit*: £{potential_profit:.2f}
            
            🔍 *Key Insights*:
            • Home Win Rate: {prediction.features.get('home_win_rate', 0):.1%}
            • Away Win Rate: {prediction.features.get('away_win_rate', 0):.1%}
            • H2H Advantage: {prediction.features.get('h2h_advantage', 0):.1%}
            • Expected Goals (Home): {prediction.features.get('poisson_home', 0):.2f}
            • Expected Goals (Away): {prediction.features.get('poisson_away', 0):.2f}
            • Weather Impact: {prediction.weather_impact:+.1%}
            
            ⚙️ *Model Version*: {prediction.model_version}
            🏦 *Current Bankroll*: £{self.bankroll:.2f}
            
            ⚠️ *Disclaimer*: Predictions are probabilistic. Bet responsibly.
            """
            
            await self.telegram_bot.send_message(
                chat_id=os.getenv('TELEGRAM_CHAT_ID'),
                text=message,
                parse_mode='Markdown'
            )
            
            logger.info(f"Sent Telegram alert for fixture {prediction.fixture_id}")
            
        except Exception as e:
            logger.error(f"Error sending Telegram alert: {str(e)}")
    
    async def update_model_with_results(self):
        """Update model with actual match results for continuous learning"""
        try:
            # Find predictions that have concluded
            response = self.supabase.table('predictions')\
                .select('*')\
                .is_('actual_result', 'null')\
                .lt('match_time', datetime.now().isoformat())\
                .execute()
            
            for prediction in response.data:
                # Fetch actual result
                actual_result = await self.fetch_actual_result(prediction['fixture_id'])
                
                if actual_result:
                    # Update prediction with result
                    self.supabase.table('predictions')\
                        .update({
                            'actual_result': json.dumps(actual_result),
                            'outcome': actual_result.get('outcome'),
                            'profit_loss': actual_result.get('profit_loss', 0)
                        })\
                        .eq('id', prediction['id'])\
                        .execute()
                    
                    # Store for training
                    self.supabase.table('historical_predictions').insert({
                        'fixture_id': prediction['fixture_id'],
                        'prediction': prediction['all_predictions'],
                        'actual_result': json.dumps(actual_result),
                        'was_correct': actual_result.get('was_correct', False),
                        'profit_loss': actual_result.get('profit_loss', 0),
                        'model_version': prediction['model_version'],
                        'timestamp': datetime.now().isoformat()
                    }).execute()
                    
                    # Update bankroll
                    if 'profit_loss' in actual_result:
                        self.bankroll += actual_result['profit_loss']
            
            # Retrain if enough new data
            await self.check_and_retrain()
            
        except Exception as e:
            logger.error(f"Error updating model: {str(e)}")
    
    async def fetch_actual_result(self, fixture_id: int) -> Optional[Dict]:
        """Fetch actual match result from API"""
        try:
            async with self.data_collector.session.get(
                f"{self.data_collector.base_url}/fixtures",
                params={'id': fixture_id}
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    fixture = data.get('response', [{}])[0]
                    
                    home_score = fixture['goals']['home']
                    away_score = fixture['goals']['away']
                    
                    # Determine outcome
                    if home_score > away_score:
                        outcome = 'home_win'
                    elif away_score > home_score:
                        outcome = 'away_win'
                    else:
                        outcome = 'draw'
                    
                    return {
                        'home_score': home_score,
                        'away_score': away_score,
                        'outcome': outcome,
                        'full_time': {
                            'home': home_score,
                            'away': away_score
                        }
                    }
            return None
        except Exception as e:
            logger.error(f"Error fetching actual result: {str(e)}")
            return None
    
    async def check_and_retrain(self):
        """Check if retraining is needed"""
        try:
            # Count new results since last training
            response = self.supabase.table('historical_predictions')\
                .select('id', count='exact')\
                .gte('timestamp', self.predictor.last_trained.isoformat())\
                .execute()
            
            new_samples = response.count or 0
            
            # Retrain if significant new data (100+ samples)
            if new_samples >= 100:
                logger.info(f"Retraining with {new_samples} new samples")
                await self.train_and_save_model()
                
        except Exception as e:
            logger.error(f"Error checking retrain: {str(e)}")
    
    async def update_performance_metrics(self):
        """Update performance metrics for dashboard"""
        try:
            response = self.supabase.table('historical_predictions')\
                .select('was_correct', 'profit_loss')\
                .execute()
            
            if response.data:
                wins = sum(1 for p in response.data if p['was_correct'])
                losses = len(response.data) - wins
                total_staked = sum(abs(p.get('profit_loss', 0)) / 0.9 for p in response.data)  # Approximate staked amount
                total_return = sum(p.get('profit_loss', 0) for p in response.data)
                
                self.performance_metrics = {
                    'total_bets': len(response.data),
                    'wins': wins,
                    'losses': losses,
                    'win_rate': wins / len(response.data) if len(response.data) > 0 else 0,
                    'total_staked': total_staked,
                    'total_return': total_return,
                    'roi': (total_return / total_staked * 100) if total_staked > 0 else 0,
                    'current_bankroll': self.bankroll
                }
                
                # Store metrics
                self.supabase.table('performance_metrics').insert({
                    **self.performance_metrics,
                    'timestamp': datetime.now().isoformat()
                }).execute()
                
        except Exception as e:
            logger.error(f"Error updating performance metrics: {str(e)}")

# FastAPI application
app = FastAPI(title="Advanced Football Predictor Pro", version="2.0.0")
prediction_engine: Optional[PredictionEngine] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global prediction_engine
    
    # Startup
    logger.info("Starting Advanced Football Predictor Pro...")
    
    # Validate environment variables
    required_env_vars = ['SUPABASE_URL', 'SUPABASE_KEY', 'TELEGRAM_BOT_TOKEN', 
                        'TELEGRAM_CHAT_ID', 'API_FOOTBALL_KEY']
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        raise ValueError(f"Missing environment variables: {missing_vars}")
    
    # Initialize Supabase
    supabase_client = create_client(
        os.getenv('SUPABASE_URL'),
        os.getenv('SUPABASE_KEY')
    )
    
    # Initialize prediction engine
    prediction_engine = PredictionEngine(
        supabase_client,
        os.getenv('TELEGRAM_BOT_TOKEN')
    )
    
    # Initialize with API key
    await prediction_engine.initialize(os.getenv('API_FOOTBALL_KEY'))
    
    # Schedule periodic tasks
    scheduler.add_job(
        prediction_engine.update_model_with_results,
        'interval',
        minutes=30
    )
    
    scheduler.add_job(
        prediction_engine.train_and_save_model,
        CronTrigger(hour=3, minute=0)  # Daily at 3 AM
    )
    
    scheduler.add_job(
        prediction_engine.update_performance_metrics,
        'interval',
        hours=1
    )
    
    scheduler.start()
    
    yield
    
    # Shutdown
    scheduler.shutdown()
    if prediction_engine.data_collector.session:
        await prediction_engine.data_collector.session.close()
    logger.info("Shutting down...")

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not prediction_engine:
        return {"status": "initializing", "timestamp": datetime.now().isoformat()}
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_version": prediction_engine.model_version,
        "bankroll": prediction_engine.bankroll,
        "performance": prediction_engine.performance_metrics
    }

@app.get("/predict/{fixture_id}")
async def predict_fixture(fixture_id: int):
    """Get prediction for specific fixture"""
    if not prediction_engine:
        raise HTTPException(status_code=503, detail="System initializing")
    
    try:
        # Fetch fixture data
        async with prediction_engine.data_collector.session.get(
            f"{prediction_engine.data_collector.base_url}/fixtures",
            params={'id': fixture_id}
        ) as response:
            
            if response.status != 200:
                raise HTTPException(status_code=404, detail="Fixture not found")
            
            data = await response.json()
            fixtures = data.get('response', [])
            
            if not fixtures:
                raise HTTPException(status_code=404, detail="Fixture not found")
            
            fixture = fixtures[0]
            
            # Make prediction
            prediction = await prediction_engine.analyze_fixture(fixture)
            
            return {
                "fixture": f"{prediction.home_team} vs {prediction.away_team}",
                "league": prediction.league,
                "match_time": prediction.timestamp.isoformat(),
                "prediction": {
                    "recommended_market": prediction.recommended_market,
                    "odds": prediction.market_odds.get(prediction.recommended_market, 0),
                    "confidence": prediction.confidence,
                    "expected_value": prediction.expected_value,
                    "kelly_criterion": prediction.kelly_criterion
                },
                "features": prediction.features,
                "model_version": prediction.model_version
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain")
async def trigger_retraining(background_tasks: BackgroundTasks):
    """Manually trigger model retraining"""
    if not prediction_engine:
        raise HTTPException(status_code=503, detail="System initializing")
    
    background_tasks.add_task(prediction_engine.train_and_save_model)
    
    return {
        "status": "retraining_started",
        "message": "Model retraining initiated in background",
        "model_version": prediction_engine.model_version
    }

@app.get("/stats")
async def get_system_stats():
    """Get system statistics and performance metrics"""
    if not prediction_engine:
        raise HTTPException(status_code=503, detail="System initializing")
    
    try:
        # Get model performance
        response = supabase.table('models')\
            .select('*')\
            .order('created_at', desc=True)\
            .limit(5)\
            .execute()
        
        models = response.data
        
        # Get prediction accuracy
        accuracy_response = supabase.table('historical_predictions')\
            .select('was_correct', count='exact')\
            .execute()
        
        total = accuracy_response.count or 0
        correct = sum(1 for p in accuracy_response.data if p['was_correct'])
        accuracy = correct / total if total > 0 else 0
        
        return {
            "performance": prediction_engine.performance_metrics,
            "model_performance": {
                "current_accuracy": accuracy,
                "total_predictions": total,
                "model_versions": [{
                    'version': m['version'],
                    'accuracy': m['accuracy'],
                    'trained_at': m['created_at']
                } for m in models]
            },
            "system": {
                "status": "operational",
                "last_trained": models[0]['created_at'] if models else None,
                "bankroll": prediction_engine.bankroll
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/bankroll")
async def get_bankroll():
    """Get current bankroll and stake history"""
    if not prediction_engine:
        raise HTTPException(status_code=503, detail="System initializing")
    
    return {
        "current_bankroll": prediction_engine.bankroll,
        "stake_history": prediction_engine.stake_history[-50:],  # Last 50 stakes
        "performance": prediction_engine.performance_metrics
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
