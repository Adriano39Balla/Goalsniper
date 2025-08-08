# services/scheduler.py

import logging
from services.api_football import APIFootballService
from services.prediction_engine import PredictionEngine
from services.telegram_bot import bot_service
from utils.data_storage import storage
from config import Config

logger = logging.getLogger(__name__)

class FootballScheduler:
    def __init__(self):
        self.api_service = APIFootballService()
        self.prediction_engine = PredictionEngine()

    def scan_live_matches(self):
        try:
            logger.info("Scanning for live matches...")
            live_matches = self.api_service.get_live_matches()
            predictions_sent = 0

            for match in live_matches:
                match_id = match['id']
                storage.store_live_match(match_id, match)
                match_stats = self.api_service.get_match_statistics(match_id)
                if match_stats:
                    storage.store_match_statistics(match_id, match_stats)

                predictions = self.prediction_engine.analyze_match(match, match_stats or {})
                for prediction in predictions:
                    if prediction['confidence'] >= Config.MIN_CONFIDENCE_THRESHOLD:
                        if not self._prediction_already_sent(prediction):
                            storage.add_prediction(prediction)
                            if bot_service.send_prediction(prediction):
                                predictions_sent += 1
            logger.info(f"Scan complete. {predictions_sent} predictions sent.")
        except Exception as e:
            logger.error(f"Error scanning live matches: {str(e)}")

    def scan_upcoming_matches(self):
        try:
            logger.info("Scanning upcoming matches...")
            self.api_service.get_upcoming_matches()
        except Exception as e:
            logger.error(f"Error scanning upcoming matches: {str(e)}")

    def send_daily_summary(self):
        try:
            summary = storage.get_recent_predictions(hours=24)
            performance = storage.get_performance_metrics()
            bot_service.send_daily_summary(summary, performance)
        except Exception as e:
            logger.error(f"Error sending summary: {str(e)}")

    def cleanup_old_data(self):
        try:
            storage.cleanup_old_data(days=7)
        except Exception as e:
            logger.error(f"Cleanup error: {str(e)}")

    def _prediction_already_sent(self, prediction):
        recent_predictions = storage.get_recent_predictions(hours=2)
        for recent_pred in recent_predictions:
            if (recent_pred['match_id'] == prediction['match_id'] and 
                recent_pred['type'] == prediction['type']):
                return True
        return False

scheduler_service = FootballScheduler()
