import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta

from core.database import db
from core.config import config

log = logging.getLogger("goalsniper.training")

class TrainingService:
    def __init__(self):
        self.feature_engineer = FeatureEngineer()  # From your features module

    def load_training_data(self, days: int = 60, min_minute: int = 20) -> List[Dict[str, Any]]:
        """Load training data from tip_snapshots"""
        cutoff_ts = int((datetime.now() - timedelta(days=days)).timestamp())
        
        with db.get_cursor() as c:
            c.execute("""
                SELECT payload 
                FROM tip_snapshots 
                WHERE created_ts >= %s 
                ORDER BY created_ts DESC
                LIMIT 5000
            """, (cutoff_ts,))
            
            training_data = []
            for (payload,) in c.fetchall():
                try:
                    data = json.loads(payload)
                    match_data = data.get("match", {})
                    features = data.get("features", {})
                    
                    # Only use in-play data with sufficient minutes
                    minute = int(features.get("minute", 0))
                    if minute >= min_minute:
                        training_data.append({
                            "match": match_data,
                            "features": features,
                            "timestamp": data.get("timestamp", 0)
                        })
                except Exception as e:
                    log.warning("Failed to parse training sample: %s", e)
                    continue
            
            log.info("Loaded %d training samples (minute >= %d)", len(training_data), min_minute)
            return training_data

    def train_models(self, days: int = 60) -> Dict[str, Any]:
        """Main training function"""
        log.info("Starting model training (in-play only, %d days)", days)
        
        try:
            training_data = self.load_training_data(days)
            if not training_data:
                return {"ok": False, "error": "No training data available"}
            
            # IN-PLAY MARKETS ONLY
            markets_to_train = [
                ("BTTS", "BTTS: Yes"),
                ("BTTS", "BTTS: No"),
                ("Over/Under 2.5", "Over 2.5 Goals"),
                ("Over/Under 2.5", "Under 2.5 Goals"),
                ("Over/Under 3.5", "Over 3.5 Goals"), 
                ("Over/Under 3.5", "Under 3.5 Goals"),
                ("1X2", "Home Win"),
                ("1X2", "Away Win")
            ]
            
            trained_models = {}
            
            for market, suggestion in markets_to_train:
                try:
                    model = self._train_model_for_market(market, suggestion, training_data)
                    if model:
                        # Save to database
                        model_key = self._get_model_key(market)
                        self._save_model(model_key, model)
                        trained_models[f"{market} {suggestion}"] = True
                        log.info("Saved model: %s", model_key)
                    
                except Exception as e:
                    log.error("Failed to train %s %s: %s", market, suggestion, e)
                    trained_models[f"{market} {suggestion}"] = False
            
            return {
                "ok": True,
                "trained": trained_models,
                "total_samples": len(training_data),
                "message": f"Trained {sum(1 for v in trained_models.values() if v)}/{len(trained_models)} models"
            }
            
        except Exception as e:
            log.exception("Training failed: %s", e)
            return {"ok": False, "error": str(e)}

    def _train_model_for_market(self, market: str, suggestion: str, training_data: List[Dict]) -> Optional[Dict[str, Any]]:
        """Train a model for a specific market and suggestion"""
        features_list, labels = self._prepare_features_and_labels(training_data, market, suggestion)
        
        if len(features_list) < 100:
            log.warning("Insufficient samples for %s %s: %d", market, suggestion, len(features_list))
            return None
        
        # Convert features to matrix
        feature_names = list(features_list[0].keys())
        X = np.array([[feat.get(name, 0) for name in feature_names] for feat in features_list])
        y = np.array(labels)
        
        if len(np.unique(y)) < 2:
            log.warning("Only one class in labels for %s %s", market, suggestion)
            return None
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        # Calibrate and return model data
        calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv=3)
        calibrated_model.fit(X_train, y_train)
        
        # Extract weights and intercept
        if hasattr(calibrated_model, 'calibrated_classifiers_'):
            base_estimator = calibrated_model.calibrated_classifiers_[0].base_estimator
            weights = dict(zip(feature_names, base_estimator.coef_[0]))
            intercept = float(base_estimator.intercept_[0])
        else:
            weights = dict(zip(feature_names, model.coef_[0]))
            intercept = float(model.intercept_[0])
        
        return {
            "weights": weights,
            "intercept": intercept,
            "feature_names": feature_names,
            "train_score": model.score(X_train, y_train),
            "test_score": model.score(X_test, y_test),
            "samples": len(X_train),
            "calibration": {"method": "sigmoid", "a": 1.0, "b": 0.0}
        }

    def _prepare_features_and_labels(self, training_data: List[Dict], market: str, suggestion: str) -> Tuple[List[Dict], List[int]]:
        """Prepare features and labels for training"""
        features_list = []
        labels = []
        
        for data in training_data:
            outcome = self._calculate_outcome(data["match"], market, suggestion)
            if outcome is not None:
                features_list.append(data["features"])
                labels.append(outcome)
        
        log.info("Market %s - %s: %d samples", market, suggestion, len(features_list))
        return features_list, labels

    def _calculate_outcome(self, match_data: dict, market: str, suggestion: str) -> Optional[int]:
        """Calculate if a prediction would have been correct"""
        # [Your outcome calculation logic]
        pass

    def _get_model_key(self, market: str) -> str:
        """Convert market name to model key format"""
        if market.startswith("Over/Under"):
            line = market.split()[-1]
            return f"OU_{line}"
        return market

    def _save_model(self, model_key: str, model_data: Dict[str, Any]):
        """Save model to database"""
        with db.get_cursor() as c:
            c.execute(
                "INSERT INTO settings(key, value) VALUES(%s, %s) ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
                (model_key, json.dumps(model_data, separators=(",", ":")))
            )

if __name__ == "__main__":
    trainer = TrainingService()
    result = trainer.train_models()
    print(json.dumps(result, indent=2))
