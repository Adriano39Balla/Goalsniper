import numpy as np
from typing import Dict, List
import logging
from .bayesian_network import BayesianPredictor
from .logistic_regression import LogisticRegressionPredictor
from config.settings import settings

logger = logging.getLogger(__name__)

class EnsemblePredictor:
    def __init__(self):
        self.bayesian_net = BayesianPredictor()
        self.logistic_reg = LogisticRegressionPredictor()
        self.model_weights = {
            'bayesian': 0.4,
            'logistic': 0.6
        }
        self.confidence_threshold = settings.MIN_CONFIDENCE_THRESHOLD
    
    def load_models(self):
        """Load all trained models"""
        try:
            self.bayesian_net.load_model()
            self.logistic_reg.load_models()
            logger.info("All models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def predict(self, match_data: Dict) -> Dict[str, float]:
        """Get ensemble prediction with calibrated probabilities"""
        try:
            # Get predictions from all models
            bayesian_preds = self.bayesian_net.predict(match_data)
            logistic_preds = self.logistic_reg.predict(match_data)
            
            # Weighted ensemble averaging
            ensemble_preds = {}
            for key in bayesian_preds.keys():
                if key in logistic_preds:
                    ensemble_preds[key] = (
                        self.model_weights['bayesian'] * bayesian_preds[key] +
                        self.model_weights['logistic'] * logistic_preds[key]
                    )
            
            # Apply probability calibration adjustments
            calibrated_preds = self._calibrate_probabilities(ensemble_preds)
            
            return calibrated_preds
            
        except Exception as e:
            logger.error(f"Ensemble prediction error: {e}")
            return self._get_fallback_probabilities()
    
    def _calibrate_probabilities(self, preds: Dict[str, float]) -> Dict[str, float]:
        """Apply probability calibration based on historical accuracy"""
        # Simple probability smoothing
        calibrated = preds.copy()
        
        # Ensure probabilities sum to 1 for mutually exclusive outcomes
        if 'home_win' in calibrated and 'draw' in calibrated and 'away_win' in calibrated:
            total = calibrated['home_win'] + calibrated['draw'] + calibrated['away_win']
            if total > 0:
                calibrated['home_win'] /= total
                calibrated['draw'] /= total
                calibrated['away_win'] /= total
        
        # Apply confidence threshold
        for key in calibrated:
            if calibrated[key] < 0.05:  # Minimum probability threshold
                calibrated[key] = 0.05
            elif calibrated[key] > 0.95:  # Maximum probability threshold
                calibrated[key] = 0.95
        
        return calibrated
    
    def calculate_confidence(self, preds: Dict[str, float]) -> float:
        """Calculate overall prediction confidence"""
        # Confidence based on probability clarity
        max_prob = max(preds.values())
        confidence = (max_prob - (1 / len(preds))) / (1 - (1 / len(preds)))
        return max(0.0, min(1.0, confidence))
    
    def get_recommended_bet(self, preds: Dict[str, float]) -> Dict[str, any]:
        """Get recommended bet based on predictions"""
        confidence = self.calculate_confidence(preds)
        
        if confidence < self.confidence_threshold:
            return {'bet_type': 'NO_BET', 'confidence': confidence}
        
        # Find best bet opportunity
        bet_opportunities = []
        
        # 1X2 market
        if preds['home_win'] > 0.6:
            bet_opportunities.append(('1X2_HOME', preds['home_win']))
        if preds['away_win'] > 0.6:
            bet_opportunities.append(('1X2_AWAY', preds['away_win']))
        
        # BTTS market
        if preds['btts_yes'] > 0.65:
            bet_opportunities.append(('BTTS_YES', preds['btts_yes']))
        elif preds['btts_no'] > 0.65:
            bet_opportunities.append(('BTTS_NO', preds['btts_no']))
        
        # Over/Under market
        if preds['over_25'] > 0.62:
            bet_opportunities.append(('OVER_25', preds['over_25']))
        elif preds['under_25'] > 0.62:
            bet_opportunities.append(('UNDER_25', preds['under_25']))
        
        if not bet_opportunities:
            return {'bet_type': 'NO_BET', 'confidence': confidence}
        
        # Select best opportunity
        best_bet = max(bet_opportunities, key=lambda x: x[1])
        
        return {
            'bet_type': best_bet[0],
            'confidence': best_bet[1],
            'stake_confidence': min(1.0, best_bet[1] * confidence)
        }
    
    def _get_fallback_probabilities(self) -> Dict[str, float]:
        """Return fallback probabilities if prediction fails"""
        return {
            'home_win': 0.33,
            'draw': 0.33,
            'away_win': 0.34,
            'btts_yes': 0.5,
            'btts_no': 0.5,
            'over_25': 0.5,
            'under_25': 0.5
        }
