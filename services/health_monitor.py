import logging
from datetime import datetime, timedelta
from typing import Dict, Any
from data.database import db
from data.api_client import api_client

logger = logging.getLogger(__name__)

class HealthMonitor:
    def __init__(self):
        self.last_health_check = None
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        self.last_health_check = datetime.now()
        
        health_status = {
            'timestamp': self.last_health_check.isoformat(),
            'models': await self._check_models(),
            'database': await self._check_database(),
            'api': await self._check_api_connection(),
            'performance': await self._get_performance_metrics(),
            'last_update': self._get_last_update(),
        }
        
        return health_status
    
    async def _check_models(self) -> Dict[str, bool]:
        """Check model health"""
        try:
            from models.ensemble import EnsemblePredictor
            predictor = EnsemblePredictor()
            predictor.load_models()
            
            # Test prediction with dummy data
            test_data = {
                'home_attack_strength_cat': 2,
                'away_attack_strength_cat': 2,
                'home_defense_strength_cat': 2,
                'away_defense_strength_cat': 2,
                'home_form_cat': 2,
                'away_form_cat': 2,
                'h2h_goal_diff_cat': 2
            }
            
            prediction = predictor.predict(test_data)
            
            return {
                'bayesian': True,
                'logistic': True,
                'ensemble': True
            }
        except Exception as e:
            logger.error(f"Model health check failed: {e}")
            return {
                'bayesian': False,
                'logistic': False,
                'ensemble': False
            }
    
    async def _check_database(self) -> bool:
        """Check database connection"""
        try:
            db.execute_query("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    async def _check_api_connection(self) -> bool:
        """Check API connection"""
        try:
            fixtures = await api_client.get_live_fixtures()
            return fixtures is not None
        except Exception as e:
            logger.error(f"API health check failed: {e}")
            return False
    
    async def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        try:
            # Get today's predictions
            query = """
            SELECT COUNT(*) as total_predictions,
                   AVG(confidence) as avg_confidence,
                   SUM(CASE WHEN recommended_bet != 'NO_BET' THEN 1 ELSE 0 END) as bets_recommended
            FROM predictions 
            WHERE DATE(prediction_time) = CURRENT_DATE
            """
            result = db.execute_query(query)
            
            # Calculate accuracy (simplified - you'd want actual bet results)
            accuracy_query = """
            SELECT COUNT(*) as total_bets,
                   SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful_bets
            FROM bet_results 
            WHERE DATE(created_at) >= CURRENT_DATE - INTERVAL '7 days'
            """
            accuracy_result = db.execute_query(accuracy_query)
            
            total_bets = accuracy_result[0]['total_bets'] if accuracy_result else 0
            successful_bets = accuracy_result[0]['successful_bets'] if accuracy_result else 0
            accuracy_rate = successful_bets / total_bets if total_bets > 0 else 0
            
            return {
                'predictions_today': result[0]['total_predictions'] if result else 0,
                'bets_recommended_today': result[0]['bets_recommended'] if result else 0,
                'avg_confidence': float(result[0]['avg_confidence']) if result and result[0]['avg_confidence'] else 0.0,
                'accuracy_rate': accuracy_rate
            }
        except Exception as e:
            logger.error(f"Performance metrics error: {e}")
            return {
                'predictions_today': 0,
                'bets_recommended_today': 0,
                'avg_confidence': 0.0,
                'accuracy_rate': 0.0
            }
    
    def _get_last_update(self) -> str:
        """Get last update timestamp"""
        if self.last_health_check:
            return self.last_health_check.strftime("%Y-%m-%d %H:%M:%S")
        return "Never"
    
    async def generate_daily_report(self) -> Dict[str, Any]:
        """Generate daily performance report"""
        health = await self.get_health_status()
        
        # Get top leagues
        league_query = """
        SELECT league_name, COUNT(*) as game_count
        FROM predictions 
        WHERE DATE(prediction_time) = CURRENT_DATE
        GROUP BY league_name 
        ORDER BY game_count DESC 
        LIMIT 5
        """
        league_results = db.execute_query(league_query)
        
        return {
            'health_status': health,
            'predictions_today': health['performance']['predictions_today'],
            'bets_recommended': health['performance']['bets_recommended_today'],
            'accuracy': health['performance']['accuracy_rate'],
            'top_leagues': [(row['league_name'], row['game_count']) for row in league_results],
            'model_health': all(health['models'].values())
        }
