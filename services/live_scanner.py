import asyncio
import aiohttp
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
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
        
    async def scan_live_matches(self, leagues: Optional[List[int]] = None) -> List[Dict]:
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
                fixture = match.get('fixture', {})
                status = fixture.get('status', {})
                minute = status.get('elapsed', 0)
                
                if 20 <= minute <= 80:
                    try:
                        prediction = await self.analyze_match(match)
                        if prediction:
                            predictions.append(prediction)
                    except Exception as e:
                        logger.error(f"Error analyzing match {fixture.get('id')}: {e}")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error scanning live matches: {e}")
            return []
    
    async def analyze_match(self, match_data: Dict) -> Optional[Dict]:
        """Analyze a single live match"""
        
        match_id = match_data['fixture']['id']
        
        # Check if already being analyzed
        if match_id in self.active_scans:
            logger.debug(f"Match {match_id} already being analyzed, skipping")
            return None
        
        self.active_scans[match_id] = True
        
        try:
            # Get detailed match statistics
            stats = await self.api.get_match_statistics(match_id)
            
            if not stats:
                logger.warning(f"No statistics available for match {match_id}")
                return None
            
            # Extract features for prediction
            features = self.extract_live_features(match_data, stats)
            
            # Make predictions
            predictions = self.predictor.predict_match(features)
            
            if not predictions:
                logger.debug(f"No predictions generated for match {match_id}")
                return None
            
            # Calculate confidence and filter
            filtered_predictions = self.filter_predictions(predictions)
            
            if filtered_predictions:
                # Prepare prediction data
                fixture = match_data['fixture']
                teams = match_data['teams']
                goals = match_data.get('goals', {})
                league = match_data.get('league', {})
                
                prediction_data = {
                    'match_id': match_id,
                    'league_id': league.get('id'),
                    'home_team': teams['home']['name'],
                    'away_team': teams['away']['name'],
                    'current_score': f"{goals.get('home', 0)}-{goals.get('away', 0)}",
                    'minute': fixture['status']['elapsed'],
                    'predictions': filtered_predictions,
                    'features': features,
                    'match_time': datetime.utcnow(),
                    'api_data': match_data
                }
                
                # Save to database
                try:
                    prediction_id = self.db.save_prediction(prediction_data)
                    prediction_data['prediction_id'] = prediction_id
                    
                    logger.info(f"Prediction generated for match {match_id}: {filtered_predictions}")
                    return prediction_data
                except Exception as e:
                    logger.error(f"Error saving prediction for match {match_id}: {e}")
                    return None
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing match {match_id}: {e}")
            return None
        finally:
            if match_id in self.active_scans:
                del self.active_scans[match_id]
    
    def extract_live_features(self, match_data: Dict, stats: Dict) -> Dict[str, np.ndarray]:
        """Extract features from live match data"""
        
        # Extract match info
        fixture = match_data.get('fixture', {})
        status = fixture.get('status', {})
        teams = match_data.get('teams', {})
        goals = match_data.get('goals', {})
        
        minute = status.get('elapsed', 0)
        home_score = goals.get('home', 0)
        away_score = goals.get('away', 0)
        
        # Parse statistics
        home_stats = {}
        away_stats = {}
        
        if isinstance(stats, list) and len(stats) > 0:
            for team_stat in stats:
                team_info = team_stat.get('team', {})
                statistics = team_stat.get('statistics', [])
                
                if team_info['id'] == teams['home']['id']:
                    home_stats = self.parse_statistics(statistics)
                elif team_info['id'] == teams['away']['id']:
                    away_stats = self.parse_statistics(statistics)
        
        # Calculate derived features
        total_goals = home_score + away_score
        goal_difference = home_score - away_score
        
        # 1X2 features
        features_1x2 = [
            minute / 90.0,  # Match progress
            float(home_score),
            float(away_score),
            float(home_stats.get('shots_on_goal', 0)),
            float(away_stats.get('shots_on_goal', 0)),
            float(home_stats.get('possession', 50)) / 100.0,
            float(home_stats.get('pass_accuracy', 50)) / 100.0,
            float(away_stats.get('pass_accuracy', 50)) / 100.0,
            float(home_stats.get('corners', 0)),
            float(away_stats.get('corners', 0)),
            float(home_stats.get('yellow_cards', 0)),
            float(away_stats.get('yellow_cards', 0)),
            float(goal_difference),
            float(home_stats.get('shots_on_goal', 0) - away_stats.get('shots_on_goal', 0)),
            0.5,  # Placeholder for recent form (would need historical data)
        ]
        
        # Over/Under features
        home_xg = home_stats.get('expected_goals', 0.0)
        away_xg = away_stats.get('expected_goals', 0.0)
        expected_goals = float(home_xg) + float(away_xg)
        
        features_over_under = [
            minute / 90.0,
            float(total_goals),
            float(expected_goals),
            float(home_stats.get('shots_total', 0)),
            float(away_stats.get('shots_total', 0)),
            float(home_stats.get('shots_on_goal', 0)),
            float(away_stats.get('shots_on_goal', 0)),
            float(home_stats.get('shots_on_goal', 0) + away_stats.get('shots_on_goal', 0)) / max(minute, 1),
            float(home_stats.get('corners', 0)),
            float(away_stats.get('corners', 0)),
            float(home_stats.get('attacks', 0) + away_stats.get('attacks', 0)) / max(minute, 1),
        ]
        
        # BTTS features
        both_scored = 1.0 if home_score > 0 and away_score > 0 else 0.0
        
        features_btts = [
            minute / 90.0,
            both_scored,
            float(home_score),
            float(away_score),
            float(home_stats.get('shots_on_goal', 0)),
            float(away_stats.get('shots_on_goal', 0)),
            float(home_stats.get('shots_inside_box', 0)),
            float(away_stats.get('shots_inside_box', 0)),
            float(home_stats.get('dangerous_attacks', 0)) / max(minute, 1),
            float(away_stats.get('dangerous_attacks', 0)) / max(minute, 1),
            float(home_stats.get('tackles', 0)),
            float(away_stats.get('tackles', 0)),
        ]
        
        return {
            '1X2': np.array(features_1x2, dtype=np.float32),
            'over_under': np.array(features_over_under, dtype=np.float32),
            'btts': np.array(features_btts, dtype=np.float32)
        }
    
    def parse_statistics(self, stats_list: List[Dict]) -> Dict[str, float]:
        """Parse API statistics into a dictionary"""
        parsed = {}
        
        if not stats_list:
            return parsed
        
        for stat in stats_list:
            stat_type = stat.get('type', '').lower().replace(' ', '_')
            value = stat.get('value')
            
            if value is None:
                continue
            
            try:
                # Convert percentage strings
                if isinstance(value, str):
                    if '%' in value:
                        value = value.replace('%', '')
                    # Try to convert to float
                    value = float(value)
                
                parsed[stat_type] = float(value)
            except (ValueError, TypeError):
                # Skip if conversion fails
                continue
        
        return parsed
    
    def filter_predictions(self, predictions: Dict, 
                          confidence_threshold: float = 0.65,
                          probability_threshold: float = 0.6) -> Dict:
        """Filter predictions based on confidence and thresholds"""
        
        filtered = {}
        
        for pred_type, pred_data in predictions.items():
            if pred_type == '1X2':
                # Home, Draw, Away probabilities
                home_win = pred_data.get('home_win', 0)
                away_win = pred_data.get('away_win', 0)
                draw = pred_data.get('draw', 0)
                
                # Find highest probability that's not draw
                max_prob = max(home_win, away_win, draw)
                
                if max_prob == home_win and home_win > draw:
                    confidence = home_win - max(away_win, draw)
                    if confidence >= confidence_threshold and home_win >= probability_threshold:
                        filtered[pred_type] = {
                            'prediction': 'home_win',
                            'probability': float(home_win),
                            'confidence': float(confidence)
                        }
                elif max_prob == away_win and away_win > draw:
                    confidence = away_win - max(home_win, draw)
                    if confidence >= confidence_threshold and away_win >= probability_threshold:
                        filtered[pred_type] = {
                            'prediction': 'away_win',
                            'probability': float(away_win),
                            'confidence': float(confidence)
                        }
            
            elif pred_type == 'over_under':
                over = pred_data.get('over', 0)
                under = pred_data.get('under', 0)
                
                confidence = abs(over - under)
                if confidence >= confidence_threshold:
                    if over > under and over >= probability_threshold:
                        filtered[pred_type] = {
                            'prediction': 'over',
                            'probability': float(over),
                            'confidence': float(confidence)
                        }
                    elif under > over and under >= probability_threshold:
                        filtered[pred_type] = {
                            'prediction': 'under',
                            'probability': float(under),
                            'confidence': float(confidence)
                        }
            
            elif pred_type == 'btts':
                yes = pred_data.get('yes', 0)
                no = pred_data.get('no', 0)
                
                confidence = abs(yes - no)
                if confidence >= confidence_threshold:
                    if yes > no and yes >= probability_threshold:
                        filtered[pred_type] = {
                            'prediction': 'yes',
                            'probability': float(yes),
                            'confidence': float(confidence)
                        }
                    elif no > yes and no >= probability_threshold:
                        filtered[pred_type] = {
                            'prediction': 'no',
                            'probability': float(no),
                            'confidence': float(confidence)
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
                
                if predictions:
                    logger.info(f"Scan completed in {elapsed:.2f}s. Found {len(predictions)} predictions.")
                else:
                    logger.debug(f"Scan completed in {elapsed:.2f}s. No predictions found.")
                
                logger.debug(f"Sleeping for {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)
                
            except asyncio.CancelledError:
                logger.info("Continuous scan cancelled")
                break
            except Exception as e:
                logger.error(f"Error in continuous scan: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def process_prediction(self, prediction: Dict):
        """Process a generated prediction"""
        
        try:
            # Send to Telegram
            await self.send_telegram_alert(prediction)
            
            # Log for monitoring
            logger.info(f"Processed prediction for match {prediction.get('match_id')}")
            
            # Store in cache for potential bet placement
            await self.store_for_bet_placement(prediction)
            
        except Exception as e:
            logger.error(f"Error processing prediction: {e}")
    
    async def send_telegram_alert(self, prediction: Dict):
        """Send prediction alert to Telegram"""
        
        try:
            from utils.telegram_bot import TelegramBot
            telegram = TelegramBot()
            
            # Format message
            match_info = {
                'home_team': prediction.get('home_team'),
                'away_team': prediction.get('away_team'),
                'current_score': prediction.get('current_score'),
                'minute': prediction.get('minute'),
                'match_time': prediction.get('match_time')
            }
            
            formatted_prediction = {
                'match_info': match_info,
                'predictions': prediction.get('predictions', {})
            }
            
            await telegram.send_prediction(formatted_prediction)
            
        except ImportError:
            logger.warning("Telegram bot not configured")
        except Exception as e:
            logger.error(f"Error sending Telegram alert: {e}")
    
    async def store_for_bet_placement(self, prediction: Dict):
        """Store prediction for potential bet placement"""
        
        # This is where you would implement logic to decide if a bet should be placed
        # and store it in a queue or database for further processing
        
        # For now, just log it
        predictions = prediction.get('predictions', {})
        if predictions:
            logger.info(f"Prediction stored for potential bet: {predictions}")
