import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import time

from core.config import config
from core.database import db
from services.features import feature_engineer

log = logging.getLogger("goalsniper.ai")

class AdvancedEnsemblePredictor:
    """Advanced ensemble system combining multiple model types with dynamic weighting"""

    def __init__(self):
        self.model_types = ['logistic', 'xgboost', 'neural', 'bayesian', 'momentum']
        self.ensemble_weights = self._initialize_adaptive_weights()
        self.performance_tracker = {}

    def _initialize_adaptive_weights(self):
        """Initialize weights based on historical performance"""
        return {
            'logistic': 0.25,
            'xgboost': 0.30,
            'neural': 0.20,
            'bayesian': 0.15,
            'momentum': 0.10
        }

    def predict_ensemble(self, features: Dict[str, float], market: str, minute: int) -> Tuple[float, float]:
        """Enhanced ensemble prediction with confidence scoring"""
        predictions = []
        confidences = []

        for model_type in self.model_types:
            try:
                prob, confidence = self._predict_single_model(features, market, minute, model_type)
                if prob is not None:
                    predictions.append((model_type, prob, confidence))
                    confidences.append(confidence)
            except Exception as e:
                log.warning(f"[ENSEMBLE] {model_type} model failed: %s", e)
                continue

        if not predictions:
            return 0.0, 0.0

        weighted_sum = 0.0
        total_weight = 0.0

        for model_type, prob, confidence in predictions:
            base_weight = self.ensemble_weights.get(model_type, 0.1)
            recent_performance = self._get_recent_performance(model_type, market)
            time_weight = self._calculate_time_weight(minute, model_type)
            final_weight = base_weight * confidence * recent_performance * time_weight
            weighted_sum += prob * final_weight
            total_weight += final_weight

        ensemble_prob = weighted_sum / total_weight if total_weight > 0 else 0.0
        ensemble_confidence = float(np.mean(confidences)) if confidences else 0.0

        return ensemble_prob, ensemble_confidence

    def _predict_single_model(self, features: Dict[str, float], market: str, minute: int, model_type: str) -> Tuple[Optional[float], float]:
        if model_type == 'logistic':
            return self._logistic_predict(features, market), 0.8
        elif model_type == 'xgboost':
            return self._xgboost_predict(features, market), 0.85
        elif model_type == 'neural':
            return self._neural_network_predict(features, market), 0.75
        elif model_type == 'bayesian':
            return self._bayesian_predict(features, market, minute), 0.9
        elif model_type == 'momentum':
            return self._momentum_based_predict(features, market, minute), 0.7
        return None, 0.0

    def _logistic_predict(self, features: Dict[str, float], market: str) -> float:
        """Load the exact model name (segmented by minute) and score it."""
        minute = float(features.get("minute", 0.0))
        seg = "early" if minute <= 35 else ("mid" if minute <= 70 else "late")

        name = market
        # Normalize OU market names like "OU_2.5" to the canonical key
        if market.startswith("OU_"):
            try:
                ln = float(market.split("_", 1)[1])
                from core.config import _fmt_line
                name = f"OU_{_fmt_line(ln)}"
            except Exception:
                pass

        # try segmented model first, then fallback
        mdl = self._load_model_from_settings(f"{name}@{seg}") or self._load_model_from_settings(name)
        return self._predict_from_model(mdl, features) if mdl else 0.0

    def _load_ou_model_for_line(self, line: float) -> Optional[Dict[str, Any]]:
        """Load OU model with fallback to legacy names"""
        from core.config import _fmt_line
        name = f"OU_{_fmt_line(line)}"
        mdl = self._load_model_from_settings(name)
        if not mdl and abs(line - 2.5) < 1e-6:
            mdl = self._load_model_from_settings("O25")
        if not mdl and abs(line - 3.5) < 1e-6:
            mdl = self._load_model_from_settings("O35")
        return mdl

    def _xgboost_predict(self, features: Dict[str, float], market: str) -> Optional[float]:
        try:
            base_prob = self._logistic_predict(features, market)
            correction = self._calculate_xgb_correction(features, market)
            corrected_prob = base_prob * (1 + correction)
            return max(0.0, min(1.0, corrected_prob))
        except Exception:
            return self._logistic_predict(features, market)

    def _neural_network_predict(self, features: Dict[str, float], market: str) -> Optional[float]:
        try:
            base_prob = self._logistic_predict(features, market)
            nn_correction = self._calculate_nn_correction(features, market)
            if base_prob <= 0.0 or base_prob >= 1.0:
                return base_prob
            nn_prob = 1 / (1 + np.exp(-(np.log(base_prob / (1 - base_prob)) + nn_correction)))
            return float(nn_prob)
        except Exception:
            return self._logistic_predict(features, market)

    def _bayesian_predict(self, features: Dict[str, float], market: str, minute: int) -> Optional[float]:
        try:
            prior_prob = self._get_prior_probability(features, market)
            live_prob = self._logistic_predict(features, market)
            prior_weight = max(0.1, 1.0 - (minute / 90.0))
            live_weight = min(0.9, minute / 90.0)
            bayesian_prob = (prior_prob * prior_weight + live_prob * live_weight) / (prior_weight + live_weight)
            return bayesian_prob
        except Exception:
            return self._logistic_predict(features, market)

    def _momentum_based_predict(self, features: Dict[str, float], market: str, minute: int) -> Optional[float]:
        try:
            base_prob = self._logistic_predict(features, market)
            momentum_factor = self._calculate_momentum_factor(features, minute)
            pressure_factor = self._calculate_pressure_factor(features)
            momentum_correction = (momentum_factor + pressure_factor) * 0.1
            adjusted_prob = base_prob * (1 + momentum_correction)
            return max(0.0, min(1.0, adjusted_prob))
        except Exception:
            return self._logistic_predict(features, market)

    def _calculate_xgb_correction(self, features: Dict[str, float], market: str) -> float:
        correction = 0.0
        if market == "BTTS":
            pressure_product = features.get("pressure_home", 50) * features.get("pressure_away", 50) / 2500
            xg_synergy = features.get("xg_h", 0) * features.get("xg_a", 0)
            correction = pressure_product * 0.1 + xg_synergy * 0.05
        elif market.startswith("OU"):
            attacking_pressure = (features.get("pressure_home", 0) + features.get("pressure_away", 0)) / 2
            defensive_weakness = 1.0 - features.get("defensive_stability", 0.5)
            correction = (attacking_pressure * defensive_weakness * 0.001) - 0.02
        return correction

    def _calculate_nn_correction(self, features: Dict[str, float], market: str) -> float:
        non_linear_features = []
        for key, value in features.items():
            if "xg" in key:
                non_linear_features.append(value ** 1.5)
            elif "pressure" in key:
                non_linear_features.append(np.tanh(value / 50))
            else:
                non_linear_features.append(value)
        if market == "BTTS":
            return sum(non_linear_features) * 0.01
        else:
            return sum(non_linear_features) * 0.005

    def _get_prior_probability(self, features: Dict[str, float], market: str) -> float:
        base_prior = 0.5
        if "xg_sum" in features:
            xg_density = features["xg_sum"] / max(1, features.get("minute", 1))
            base_prior = min(0.8, max(0.2, xg_density * 10))
        return base_prior

    def _calculate_momentum_factor(self, features: Dict[str, float], minute: int) -> float:
        if minute < 20:
            return 0.0
        momentum = 0.0
        goals_last_15 = features.get("goals_last_15", 0)
        momentum += goals_last_15 * 0.2
        shots_last_15 = features.get("shots_last_15", 0)
        momentum += shots_last_15 * 0.05
        recent_xg_impact = features.get("recent_xg_impact", 0)
        momentum += recent_xg_impact * 0.1
        return momentum

    def _calculate_pressure_factor(self, features: Dict[str, float]) -> float:
        pressure_diff = features.get("pressure_home", 0) - features.get("pressure_away", 0)
        score_advantage = features.get("goals_h", 0) - features.get("goals_a", 0)
        if abs(score_advantage) <= 1:
            return abs(pressure_diff) * 0.01
        else:
            return pressure_diff * 0.005

    def _get_market_specific_features_xgb(self, features: Dict[str, float], market: str) -> Dict[str, float]:
        enhanced_features = features.copy()
        enhanced_features["pressure_product"] = features.get("pressure_home", 0) * features.get("pressure_away", 0)
        enhanced_features["xg_ratio"] = features.get("xg_h", 0.1) / max(0.1, features.get("xg_a", 0.1))
        enhanced_features["efficiency_ratio"] = features.get("goals_sum", 0) / max(0.1, features.get("xg_sum", 0.1))
        return enhanced_features

    def _get_recent_performance(self, model_type: str, market: str) -> float:
        # TODO: Implement actual performance tracking
        return 0.9

    def _calculate_time_weight(self, minute: int, model_type: str) -> float:
        if model_type in ['bayesian', 'momentum']:
            return min(1.0, minute / 60.0)
        else:
            return 1.0

    def _load_model_from_settings(self, name: str) -> Optional[Dict[str, Any]]:
        """Load model from database settings"""
        # This would connect to your model storage
        # For now, return None as placeholder
        return None

    def _predict_from_model(self, mdl: Dict[str, Any], features: Dict[str, float]) -> float:
        """Predict probability using a model"""
        if not mdl:
            return 0.0
        # Simplified prediction - would use actual model weights
        return 0.5  # Placeholder

# Global AI predictor instance
ensemble_predictor = AdvancedEnsemblePredictor()
