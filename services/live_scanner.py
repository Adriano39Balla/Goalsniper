import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
from concurrent.futures import ThreadPoolExecutor

from utils.api_client import APIFootballClient
from utils.database import DatabaseManager
from utils.logger import logger
from models.random_forest import BettingPredictor
from services.predictor import PredictionService

class LiveMatchScanner:
    """Real-time live match scanner for in-play betting"""
    
    def __init__(self, api_client: APIFootballClient, db: DatabaseManager):
        self.api = api_client
        self.db = db
        self.predictor = BettingPredictor()
        self.prediction_service = PredictionService()
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.active_scans = {}
        
    async def scan_live_matches(self, leagues: Optional[List[int]] = None):
        """Scan for live matches and generate predictions"""
        
        try:
            # Get live matches from API
            live_matches = await self.api.get_live_matches(leagues)
            
            if not live_matches:
                logger.info("No live matches found")
                return []
            
            logger.info(f"Found {len(live_matches)} live matches")
            
            predictions = []
            for match in live_matches:
                # Only process matches between 20-80 minutes
                minute = match.get('fixture', {}).get('status', {}).get('elapsed', 0)
                
                if 20 <= minute <= 80:
                    try:
                        prediction = await self.analyze_match(match)
                        if prediction:
                            predictions.append(prediction)
                    except Exception as e:
                        logger.error(f"Error analyzing match {match.get('fixture', {}).get('id')}: {e}")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error scanning live matches: {e}")
            return []
    
    async def analyze_match(self, match_data: Dict) -> Optional[Dict]:
        """Analyze a single live match"""
        
        match_id = match_data['fixture']['id']
        
        # Check if already being analyzed
        if match_id in self.active_scans:
            return None
        
        self.active_scans[match_id] = True
        
        try:
            # Get detailed match statistics
            stats = await self.api.get_match_statistics(match_id)
            
            if not stats:
                return None
            
            # Extract features for prediction
            features = self.extract_live_features(match_data, stats)
            
            # Make predictions
            predictions = self.predictor.predict_match(features)
            
            if not predictions:
                return None
            
            # Calculate confidence and filter
            filtered_predictions = self.filter_predictions(predictions)
            
            if filtered_predictions:
                prediction_data = {
                    'match_id': match_id,
                    'league_id': match_data['league']['id'],
                    'home_team': match_data['teams']['home']['name'],
                    'away_team': match_data['teams']['away']['name'],
                    'current_score': match_data['goals'],
                    'minute': match_data['fixture']['status']['elapsed'],
                    'predictions': filtered_predictions,
                    'features': features,
                    'match_time': datetime.utcnow(),
                    'api_data': match_data
                }
                
                # Save to database
                prediction_id = self.db.save_prediction(prediction_data)
                prediction_data['prediction_id'] = prediction_id
                
                logger.info(f"Prediction generated for match {match_id}")
                return prediction_data
            
            return None
            
        finally:
            del self.active_scans[match_id]
    
    def extract_live_features(self, match_data: Dict, stats: Dict) -> Dict[str, np.ndarray]:
        """Extract features from live match data"""
        
        # Basic match info
        minute = match_data['fixture']['status']['elapsed']
        score = match_data['goals']
        home_score = score.get('home', 0)
        away_score = score.get('away', 0)
        
        # Statistics
        home_stats = {}
        away_stats = {}
        
        for stat in stats.get('statistics', []):
            if stat['team']['id'] == match_data['teams']['home']['id']:
                home_stats = self.parse_statistics(stat['statistics'])
            else:
                away_stats = self.parse_statistics(stat['statistics'])
        
        # Calculate derived features
        features = {}
        
        # 1X2 features
        features['1X2'] = np.array([
            minute / 90,  # Match progress
            home_score,
            away_score,
            home_stats.get('shots_on_goal', 0),
            away_stats.get('shots_on_goal', 0),
            home_stats.get('possession', 50) / 100,
            home_stats.get('pass_accuracy', 50) / 100,
            away_stats.get('pass_accuracy', 50) / 100,
            home_stats.get('corners', 0),
            away_stats.get('corners', 0),
            home_stats.get('yellow_cards', 0),
            away_stats.get('yellow_cards', 0),
            # Add momentum features
            (home_score - away_score),  # Goal difference
            (home_stats.get('shots_on_goal', 0) - away_stats.get('shots_on_goal', 0)),
            # Recent form (would need historical data)
        ])
        
        # Over/Under features
        total_goals = home_score + away_score
        expected_goals = (home_stats.get('expected_goals', 0) + 
                         away_stats.get('expected_goals', 0))
        
        features['over_under'] = np.array([
            minute / 90,
            total_goals,
            expected_goals,
            home_stats.get('shots_total', 0),
            away_stats.get('shots_total', 0),
            home_stats.get('shots_on_goal', 0),
            away_stats.get('shots_on_goal', 0),
            (home_stats.get('shots_on_goal', 0) + 
             away_stats.get('shots_on_goal', 0)) / max(minute, 1),
            home_stats.get('corners', 0),
            away_stats.get('corners', 0),
            # Attack intensity
            (home_stats.get('attacks', 0) + away_stats.get('attacks', 0)) / max(minute, 1),
        ])
        
        # BTTS features
        both_scored = int(home_score > 0 and away_score > 0)
        
        features['btts'] = np.array([
            minute / 90,
            both_scored,
            home_score,
            away_score,
            home_stats.get('shots_on_goal', 0),
            away_stats.get('shots_on_goal', 0),
            home_stats.get('shots_inside_box', 0),
            away_stats.get('shots_inside_box', 0),
            home_stats.get('dangerous_attacks', 0) / max(minute, 1),
            away_stats.get('dangerous_attacks', 0) / max(minute, 1),
            # Defensive pressure
            home_stats.get('tackles', 0),
            away_stats.get('tackles', 0),
        ])
        
        return features
    
    def parse_statistics(self, stats_list: List[Dict]) -> Dict[str, float]:
        """Parse API statistics into a dictionary"""
        parsed = {}
        
        for stat in stats_list:
            value = stat.get('value')
            if value is None:
                continue
            
            # Convert percentage strings
            if isinstance(value, str) and '%' in value:
                try:
                    value = float(value.strip('%'))
                except:
                    continue
            
            parsed[stat['type'].lower().replace(' ', '_')] = value
        
        return parsed
    
    def filter_predictions(self, predictions: Dict, 
                          confidence_threshold: float = 0.65,
                          probability_threshold: float = 0.6) -> Dict:
        """Filter predictions based on confidence and thresholds"""
        
        filtered = {}
        
        for pred_type, pred_data in predictions.items():
            if pred_type == '1X2':
                # Suppress draws, focus on clear winners
                home_win = pred_data.get('home_win', 0)
                away_win = pred_data.get('away_win', 0)
                draw = pred_data.get('draw', 0)
                
                # Find highest probability that's not draw
                if home_win > away_win and home_win > draw:
                    confidence = home_win - max(away_win, draw)
                    if confidence >= confidence_threshold and home_win >= probability_threshold:
                        filtered[pred_type] = {
                            'prediction': 'home_win',
                            'probability': home_win,
                            'confidence': confidence
                        }
                elif away_win > home_win and away_win > draw:
                    confidence = away_win - max(home_win, draw)
                    if confidence >= confidence_threshold and away_win >= probability_threshold:
                        filtered[pred_type] = {
                            'prediction': 'away_win',
                            'probability': away_win,
                            'confidence': confidence
                        }
            
            elif pred_type == 'over_under':
                over = pred_data.get('over', 0)
                under = pred_data.get('under', 0)
                
                confidence = abs(over - under)
                if confidence >= confidence_threshold:
                    if over > under and over >= probability_threshold:
                        filtered[pred_type] = {
                            'prediction': 'over',
                            'probability': over,
                            'confidence': confidence
                        }
                    elif under > over and under >= probability_threshold:
                        filtered[pred_type] = {
                            'prediction': 'under',
                            'probability': under,
                            'confidence': confidence
                        }
            
            elif pred_type == 'btts':
                yes = pred_data.get('yes', 0)
                no = pred_data.get('no', 0)
                
                confidence = abs(yes - no)
                if confidence >= confidence_threshold:
                    if yes > no and yes >= probability_threshold:
                        filtered[pred_type] = {
                            'prediction': 'yes',
                            'probability': yes,
                            'confidence': confidence
                        }
                    elif no > yes and no >= probability_threshold:
                        filtered[pred_type] = {
                            'prediction': 'no',
                            'probability': no,
                            'confidence': confidence
                        }
        
        return filtered
    
    async def continuous_scan(self, scan_interval: int = 30):
        """Continuous scanning of live matches"""
        
        logger.info("Starting continuous live match scanning...")
        
        while True:
            try:
                start_time = time.time()
                
                # Scan for live matches
                predictions = await self.scan_live_matches()
                
                # Process predictions
                for prediction in predictions:
                    await self.process_prediction(prediction)
                
                # Calculate sleep time
                elapsed = time.time() - start_time
                sleep_time = max(scan_interval - elapsed, 5)
                
                logger.debug(f"Scan completed in {elapsed:.2f}s. Sleeping for {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in continuous scan: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def process_prediction(self, prediction: Dict):
        """Process a generated prediction"""
        
        # Send to Telegram
        await self.send_telegram_alert(prediction)
        
        # Log for monitoring
        logger.info(f"Processed prediction: {prediction}")
        
        # Could add more processing here (e.g., auto-betting integration)
    
    async def send_telegram_alert(self, prediction: Dict):
        """Send prediction alert to Telegram"""
        
        # This would integrate with your Telegram bot
        # Implementation depends on your bot setup
        pass
