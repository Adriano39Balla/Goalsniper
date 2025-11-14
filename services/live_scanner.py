import asyncio
import logging
from typing import List, Dict, Any
from datetime import datetime, timedelta
from data.api_client import api_client
from data.database import db
from models.ensemble import EnsemblePredictor
from services.telegram_bot import telegram_bot

logger = logging.getLogger(__name__)

class LiveScanner:
    def __init__(self):
        self.ensemble_predictor = EnsemblePredictor()
        self.scanning = False
        self.last_scan = None
        self.processed_fixtures = set()
    
    async def start_scan(self):
        """Start live scanning process"""
        if self.scanning:
            logger.info("Scan already in progress")
            return
        
        self.scanning = True
        logger.info("Starting live scan")
        
        try:
            # Load models
            self.ensemble_predictor.load_models()
            
            # Continuous scanning loop
            while self.scanning:
                await self._scan_iteration()
                await asyncio.sleep(30)  # Scan every 30 seconds
                
        except Exception as e:
            logger.error(f"Live scan error: {e}")
            self.scanning = False
            raise
    
    async def stop_scan(self):
        """Stop live scanning"""
        self.scanning = False
        logger.info("Live scan stopped")
    
    async def _scan_iteration(self):
        """Single scan iteration"""
        try:
            # Get live fixtures
            live_fixtures = await api_client.get_live_fixtures()
            
            for fixture in live_fixtures:
                fixture_id = fixture['fixture']['id']
                
                # Skip if already processed recently
                if fixture_id in self.processed_fixtures:
                    continue
                
                # Process fixture
                await self._process_live_fixture(fixture)
                self.processed_fixtures.add(fixture_id)
            
            # Clean old processed fixtures (older than 4 hours)
            self._clean_processed_fixtures()
            
            self.last_scan = datetime.now()
            
        except Exception as e:
            logger.error(f"Scan iteration error: {e}")
    
    async def _process_live_fixture(self, fixture: Dict[str, Any]):
        """Process a single live fixture"""
        try:
            fixture_id = fixture['fixture']['id']
            logger.info(f"Processing live fixture: {fixture_id}")
            
            # Get detailed fixture data
            fixture_data = await self._prepare_fixture_data(fixture)
            
            # Generate prediction
            prediction = self.ensemble_predictor.predict(fixture_data)
            
            # Get recommended bet
            bet_recommendation = self.ensemble_predictor.get_recommended_bet(prediction)
            
            # Prepare prediction record
            prediction_record = {
                'fixture_id': fixture_id,
                'home_team': fixture['teams']['home']['name'],
                'away_team': fixture['teams']['away']['name'],
                'league_id': fixture['league']['id'],
                'league_name': fixture['league']['name'],
                'prediction_time': datetime.now(),
                'home_win_prob': prediction['home_win'],
                'away_win_prob': prediction['away_win'],
                'draw_prob': prediction['draw'],
                'over_25_prob': prediction['over_25'],
                'under_25_prob': prediction['under_25'],
                'btts_yes_prob': prediction['btts_yes'],
                'btts_no_prob': prediction['btts_no'],
                'confidence': bet_recommendation['confidence'],
                'recommended_bet': bet_recommendation['bet_type'],
                'bet_type': bet_recommendation['bet_type'],
                'stake_confidence': bet_recommendation.get('stake_confidence', 0.0),
                'model_version': 'ensemble_v1',
                'live_minute': fixture['fixture']['status']['elapsed'],
                'current_score': f"{fixture['goals']['home']}-{fixture['goals']['away']}"
            }
            
            # Save to database
            db.save_prediction(prediction_record)
            
            # Send to Telegram if confidence is high enough
            if (bet_recommendation['bet_type'] != 'NO_BET' and 
                bet_recommendation['confidence'] > 0.65):
                await telegram_bot.send_prediction(prediction_record)
            
            logger.info(f"Fixture {fixture_id} processed successfully")
            
        except Exception as e:
            logger.error(f"Error processing fixture {fixture_id}: {e}")
    
    async def _prepare_fixture_data(self, fixture: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare fixture data for model prediction"""
        fixture_id = fixture['fixture']['id']
        
        # Get additional statistics
        stats = await api_client.get_fixture_stats(fixture_id)
        events = await api_client.get_fixture_events(fixture_id)
        
        # Extract relevant features
        fixture_data = {
            'fixture_id': fixture_id,
            'home_team': fixture['teams']['home']['name'],
            'away_team': fixture['teams']['away']['name'],
            'league_id': fixture['league']['id'],
            'current_minute': fixture['fixture']['status']['elapsed'],
            'home_goals': fixture['goals']['home'],
            'away_goals': fixture['goals']['away'],
            'home_shots': stats.get('home', {}).get('shots', {}).get('total', 0),
            'away_shots': stats.get('away', {}).get('shots', {}).get('total', 0),
            'home_shots_on_target': stats.get('home', {}).get('shots', {}).get('on', 0),
            'away_shots_on_target': stats.get('away', {}).get('shots', {}).get('on', 0),
            'home_possession': stats.get('home', {}).get('possession', 0),
            'away_possession': stats.get('away', {}).get('possession', 0),
        }
        
        # Add recent form features (would need historical data)
        # This is a simplified version - you'd want to add more sophisticated features
        
        return fixture_data
    
    def _clean_processed_fixtures(self):
        """Clean old processed fixtures"""
        current_time = datetime.now()
        self.processed_fixtures = {
            fixture_id for fixture_id in self.processed_fixtures
            # Keep only fixtures from last 4 hours
            if current_time - timedelta(hours=4) < current_time
        }
    
    async def manual_scan(self):
        """Manual scan trigger"""
        logger.info("Manual scan triggered")
        await self._scan_iteration()
