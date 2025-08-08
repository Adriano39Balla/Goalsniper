import logging
from typing import Dict, List
from datetime import datetime, timedelta

from models import Match, Prediction
from app import db

logger = logging.getLogger(__name__)

class DataStorage:
    def store_live_match(self, api_match_id: int, match_data: Dict) -> None:
        """Save or update a live match record in the DB"""
        try:
            match = Match.query.filter_by(api_match_id=api_match_id).first()

            if match:
                match.home_score = match_data.get("home_score")
                match.away_score = match_data.get("away_score")
                match.status = match_data.get("status")
                match.match_date = match_data.get("date")
            else:
                match = Match(
                    api_match_id=api_match_id,
                    home_team=match_data.get("home_team"),
                    away_team=match_data.get("away_team"),
                    league_id=match_data.get("league_id"),
                    league_name=match_data.get("league"),
                    match_date=match_data.get("date"),
                    status=match_data.get("status"),
                    home_score=match_data.get("home_score"),
                    away_score=match_data.get("away_score")
                )
                db.session.add(match)

            db.session.commit()
        except Exception as e:
            logger.error(f"Error storing live match: {str(e)}")
            db.session.rollback()

    def store_match_statistics(self, api_match_id: int, stats: Dict) -> None:
        """Store or update JSON-formatted statistics for a match"""
        try:
            match = Match.query.filter_by(api_match_id=api_match_id).first()
            if match:
                match.statistics = stats
                db.session.commit()
        except Exception as e:
            logger.error(f"Error storing match statistics: {str(e)}")
            db.session.rollback()

    def add_prediction(self, prediction: Dict) -> None:
        """Save a prediction to the database"""
        try:
            match = Match.query.filter_by(api_match_id=prediction["match_id"]).first()
            if not match:
                logger.warning(f"No match found for prediction: {prediction['match_id']}")
                return

            new_prediction = Prediction(
                match_id=match.id,
                prediction_type=prediction["type"],
                confidence=prediction["confidence"],
                reasoning=prediction.get("reasoning", ""),
                sent_to_telegram=True  # since we're sending it right away
            )
            db.session.add(new_prediction)
            db.session.commit()
        except Exception as e:
            logger.error(f"Error adding prediction: {str(e)}")
            db.session.rollback()

    def get_recent_predictions(self, hours: int = 2) -> List[Dict]:
        """Fetch recent predictions from the last N hours"""
        try:
            time_threshold = datetime.utcnow() - timedelta(hours=hours)
            predictions = Prediction.query.filter(Prediction.predicted_at >= time_threshold).all()

            return [
                {
                    "id": pred.id,
                    "match_id": pred.match.api_match_id,
                    "home_team": pred.match.home_team,
                    "away_team": pred.match.away_team,
                    "type": pred.prediction_type,
                    "confidence": pred.confidence,
                    "reasoning": pred.reasoning,
                    "timestamp": pred.predicted_at,
                }
                for pred in predictions
            ]
        except Exception as e:
            logger.error(f"Error fetching recent predictions: {str(e)}")
            return []

    def get_performance_metrics(self) -> Dict:
        """Placeholder performance metrics"""
        try:
            total = Prediction.query.count()
            correct = Prediction.query.filter_by(actual_outcome=True).count()

            accuracy = (correct / total) if total else 0.0

            return {
                "total_predictions": total,
                "correct_predictions": correct,
                "accuracy": accuracy
            }
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            return {
                "total_predictions": 0,
                "correct_predictions": 0,
                "accuracy": 0.0
            }

    def cleanup_old_data(self, days: int = 7) -> None:
        """Delete predictions and matches older than N days"""
        try:
            cutoff = datetime.utcnow() - timedelta(days=days)

            old_predictions = Prediction.query.filter(Prediction.predicted_at < cutoff).all()
            for pred in old_predictions:
                db.session.delete(pred)

            old_matches = Match.query.filter(Match.match_date < cutoff).all()
            for match in old_matches:
                db.session.delete(match)

            db.session.commit()
            logger.info("Old data cleaned successfully.")
        except Exception as e:
            logger.error(f"Error cleaning up old data: {str(e)}")
            db.session.rollback()

# Global instance
storage = DataStorage()
