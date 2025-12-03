import numpy as np
from typing import Dict, Any
from datetime import datetime

from utils.logger import logger
from models.random_forest import BettingPredictor as RFPredictor

class PredictionService:
    """Service for making and managing predictions"""
    
    def __init__(self):
        self.predictor = RFPredictor()
        self.load_models()
        
    def load_models(self):
        """Load trained models"""
        
        try:
            # Load 1X2 model
            self.predictor.models['1X2'].load('data/models/1X2_model.joblib')
            
            # Load Over/Under model
            self.predictor.models['over_under'].load('data/models/over_under_model.joblib')
            
            # Load BTTS model
            self.predictor.models['btts'].load('data/models/btts_model.joblib')
            
            logger.info("All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def predict(self, features: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Make predictions using all models"""
        
        try:
            return self.predictor.predict_match(features)
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {}
    
    def calculate_confidence(self, probabilities: Dict[str, float]) -> float:
        """Calculate overall confidence score"""
        
        if not probabilities:
            return 0.0
        
        # Simple average of probabilities
        total = sum(probabilities.values())
        count = len(probabilities)
        
        return total / count if count > 0 else 0.0
    
    def should_place_bet(self, prediction: Dict, 
                        min_confidence: float = 0.65,
                        min_probability: float = 0.6) -> bool:
        """Determine if a bet should be placed"""
        
        predictions = prediction.get('predictions', {})
        
        if not predictions:
            return False
        
        # Check each prediction type
        for pred_type, pred_data in predictions.items():
            confidence = pred_data.get('confidence', 0)
            probability = pred_data.get('probability', 0)
            
            if confidence >= min_confidence and probability >= min_probability:
                return True
        
        return False
