import os
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from core.config import config
from services.ai_predictor import ensemble_predictor

log = logging.getLogger("goalsniper.markets")

# Add these missing utility functions at the top level
def _parse_ou_line_from_suggestion(s: str) -> Optional[float]:
    """Parse OU line from suggestion text"""
    try:
        if "Over" in s or "Under" in s:
            import re
            match = re.search(r'(\d+(?:\.\d+)?)', s)
            if match:
                return float(match.group(1))
        return None
    except Exception:
        return None

def calculate_ev(probability: float, odds: float) -> float:
    """
    Calculate Expected Value (EV) as decimal percentage.
    
    Args:
        probability: Probability as decimal (0.0-1.0)
        odds: Decimal odds (e.g., 2.0 for even money)
    
    Returns:
        EV as decimal (e.g., 0.05 = +5% edge, -0.10 = -10% edge)
    """
    try:
        return float(probability) * max(0.0, float(odds)) - 1.0
    except (ValueError, TypeError):
        return -1.0  # Return negative EV on error

def calculate_ev_percentage(probability: float, odds: float) -> float:
    """
    Calculate Expected Value as percentage.
    
    Args:
        probability: Probability as decimal (0.0-1.0)
        odds: Decimal odds
    
    Returns:
        EV as percentage (e.g., 5.0 = +5%, -10.0 = -10%)
    """
    ev_decimal = calculate_ev(probability, odds)
    return ev_decimal * 100.0

def calculate_ev_bps(probability: float, odds: float) -> int:
    """
    Calculate Expected Value in basis points (bps).
    
    Args:
        probability: Probability as decimal (0.0-1.0)
        odds: Decimal odds
    
    Returns:
        EV in basis points (e.g., 500 = +5%, -250 = -2.5%)
    """
    ev_decimal = calculate_ev(probability, odds)
    return int(round(ev_decimal * 10000))

def _ev(prob: float, odds: float) -> float:
    """Legacy EV function alias for compatibility with existing code"""
    return calculate_ev(prob, odds)

def _odds_key_for_market(market_txt: str, suggestion: str) -> str | None:
    """Map our market/suggestion to odds_map key."""
    if market_txt == "BTTS":
        return "BTTS"
    if market_txt == "1X2":
        return "1X2"
    if market_txt.startswith("Over/Under"):
        ln = _parse_ou_line_from_suggestion(suggestion)
        from core.config import _fmt_line  # Import from config
        return f"OU_{_fmt_line(ln)}" if ln is not None else None
    return None

def _market_family(market_text: str, suggestion: str) -> str:
    """Get market family for cutoff purposes (standalone function)"""
    s = (market_text or "").upper()
    if s.startswith("OVER/UNDER") or "OVER/UNDER" in s:
        return "OU"
    if s == "BTTS" or "BTTS" in s:
        return "BTTS"
    if s == "1X2" or "WINNER" in s or "MATCH WINNER" in s:
        return "1X2"
    if s.startswith("PRE "):
        return _market_family(s[4:], suggestion)
    return s

def market_cutoff_ok(minute: Optional[int], market_text: str, suggestion: str) -> bool:
    """Check if market is within minute cutoff (standalone function)"""
    fam = _market_family(market_text, suggestion)
    if minute is None:
        return True
    try:
        m = int(minute)
    except Exception:
        m = 0
    
    # Get cutoff from environment or use default
    cutoff = None
    market_cutoffs_raw = os.getenv("MARKET_CUTOFFS", "BTTS=75,1X2=80,OU=88")
    cutoffs = {}
    for tok in market_cutoffs_raw.split(","):
        tok = tok.strip()
        if not tok or "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        try:
            cutoffs[k.strip().upper()] = int(float(v.strip()))
        except Exception:
            pass
    
    cutoff = cutoffs.get(fam)
    if cutoff is None:
        tip_max_minute_env = os.getenv("TIP_MAX_MINUTE", "")
        try:
            cutoff = int(float(tip_max_minute_env)) if tip_max_minute_env else None
        except Exception:
            cutoff = None
    if cutoff is None:
        cutoff = max(0, int(os.getenv("TOTAL_MATCH_MINUTES", "95")) - 5)
    return m <= int(cutoff)

def _candidate_is_sane(suggestion: str, feat: Dict[str, float]) -> bool:
    """Check if a prediction candidate makes sense given current match state"""
    minute = int(feat.get("minute", 0))
    goals_sum = feat.get("goals_sum", 0)
    
    # Check for absurd Over/Under predictions
    if suggestion.startswith("Under"):
        try:
            line = _parse_ou_line_from_suggestion(suggestion)
            if line and goals_sum > line:
                return False  # Can't go under if already over
        except:
            pass
            
    # Check for absurd BTTS predictions
    if suggestion == "BTTS: No" and goals_sum >= 2:
        # If both teams have scored, BTTS: No doesn't make sense
        goals_h = feat.get("goals_h", 0)
        goals_a = feat.get("goals_a", 0)
        if goals_h > 0 and goals_a > 0:
            return False
            
    return True

def _inplay_1x2_sanity_ok(suggestion: str, feat: Dict[str, float], odds: Optional[float]) -> bool:
    """
    Late-game 1X2 sanity: if a side leads by >=2 after 60', live odds should be short.
    Big odds at that state implies stale/prematch leakage → block.
    """
    minute = int(feat.get("minute", 0))
    gd = int(feat.get("goals_h", 0) - feat.get("goals_a", 0))
    if suggestion not in ("Home Win", "Away Win"):
        return True
    if minute >= 60 and abs(gd) >= 2 and odds is not None and odds > 1.50:
        return False
    return True

def _min_odds_for_market(market: str) -> float:
    """Get minimum odds for a market (standalone function)"""
    if market.startswith("Over/Under"): 
        return float(os.getenv("MIN_ODDS_OU", "1.50"))
    if market == "BTTS": 
        return float(os.getenv("MIN_ODDS_BTTS", "1.50"))
    if market == "1X2":  
        return float(os.getenv("MIN_ODDS_1X2", "1.50"))
    return 1.01

class MarketValidator:
    """Market validation and odds processing"""
    
    def __init__(self):
        self.market_cutoffs = self._parse_market_cutoffs()
    
    def _parse_market_cutoffs(self) -> Dict[str, int]:
        """Parse market cutoff configurations"""
        cutoffs = {}
        raw_cutoffs = os.getenv("MARKET_CUTOFFS", "BTTS=75,1X2=80,OU=88")
        
        for token in raw_cutoffs.split(","):
            token = token.strip()
            if not token or "=" not in token:
                continue
            key, value = token.split("=", 1)
            try:
                cutoffs[key.strip().upper()] = int(float(value.strip()))
            except:
                pass
        
        return cutoffs
    
    def is_market_cutoff_ok(self, minute: int, market: str, suggestion: str) -> bool:
        """Check if market is within minute cutoff"""
        return market_cutoff_ok(minute, market, suggestion)
    
    def _get_market_family(self, market: str, suggestion: str) -> str:
        """Get market family for cutoff purposes"""
        return _market_family(market, suggestion)
    
    def _get_tip_max_minute(self) -> Optional[int]:
        """Get global tip maximum minute"""
        try:
            return int(float(os.getenv("TIP_MAX_MINUTE", ""))) if os.getenv("TIP_MAX_MINUTE") else None
        except:
            return None
    
    def is_prediction_sane(self, suggestion: str, features: Dict[str, float]) -> bool:
        """Check if prediction makes sense given current match state"""
        return _candidate_is_sane(suggestion, features)
    
    def _parse_ou_line_from_suggestion(self, suggestion: str) -> Optional[float]:
        """Parse OU line from suggestion text"""
        return _parse_ou_line_from_suggestion(suggestion)

class OddsProcessor:
    """Odds processing and validation"""
    
    def __init__(self):
        self.odds_quality_min = config.odds.quality_min
    
    def calculate_ev(self, probability: float, odds: float) -> float:
        """Calculate expected value"""
        return _ev(probability, odds)
    
    def get_min_odds_for_market(self, market: str) -> float:
        """Get minimum odds for a market"""
        return _min_odds_for_market(market)
    
    def validate_odds_quality(self, odds_data: Dict, probability: float, market: str) -> bool:
        """Validate odds quality"""
        if not odds_data:
            return False
        
        # Implementation of odds quality validation
        # [Your existing odds quality logic]
        return True

class MarketSpecificPredictor:
    """Advanced market-specific prediction with specialized models"""
    
    def __init__(self):
        self.market_strategies = {
            "BTTS": self._predict_btts_advanced,
            "OU": self._predict_ou_advanced
        }
    
    def predict_for_market(self, features: Dict[str, float], market: str, minute: int) -> Tuple[float, float]:
        """Predict probability and confidence for a specific market"""
        if market.startswith("OU_"):
            line = None
            try:
                line = float(market.split("_", 1)[1])
            except Exception:
                pass
            return self._predict_ou_advanced(features, minute, line=line, market_key=market)
        elif market in self.market_strategies:
            return self.market_strategies[market](features, minute)
        else:
            return ensemble_predictor.predict_ensemble(features, market, minute)
    
    def _predict_btts_advanced(self, features: Dict[str, float], minute: int) -> Tuple[float, float]:
        """Advanced BTTS prediction with game state adjustments"""
        base_prob, base_conf = ensemble_predictor.predict_ensemble(features, "BTTS", minute)
        adjustments = 0.0
        defensive_stability = float(features.get("defensive_stability", 0.5))
        vulnerability = 1.0 - defensive_stability
        adjustments += vulnerability * 0.2
        pressure_balance = min(float(features.get("pressure_home", 0)), float(features.get("pressure_away", 0))) / 100.0
        adjustments += pressure_balance * 0.15
        goals_last_20 = float(features.get("goals_last_20", 0))
        adjustments += min(0.3, goals_last_20 * 0.1)
        game_state = float(features.get("game_state", 0.5))
        if game_state > 0.7:
            adjustments += 0.1
        adjusted_prob = base_prob * (1 + adjustments)
        confidence = base_conf * 0.9
        from core.config import _clamp_prob
        return _clamp_prob(adjusted_prob), max(0.0, min(1.0, confidence))
    
    def _predict_ou_advanced(self, features: Dict[str, float], minute: int, line: Optional[float] = None, market_key: Optional[str] = None) -> Tuple[float, float]:
        """Advanced Over/Under prediction with line-specific adjustments"""
        base_prob, base_conf = 0.0, 0.0
        mk = market_key or (f"OU_{line}" if line is not None else "OU")

        try:
            if line is not None:
                mk_line = f"OU_{line}"
                ensemble_prob, ensemble_conf = ensemble_predictor.predict_ensemble(features, mk_line, minute)
            else:
                ensemble_prob, ensemble_conf = ensemble_predictor.predict_ensemble(features, mk, minute)

            base_prob, base_conf = float(ensemble_prob), float(ensemble_conf)
        except Exception as e:
            log.warning(f"[OU_PREDICT] Ensemble failed for {mk}: {e}")

        if (base_prob <= 0.0) and (line is not None):
            mdl = ensemble_predictor._load_ou_model_for_line(line)
            if mdl:
                base_prob = ensemble_predictor._predict_from_model(mdl, features)
                base_conf = max(base_conf, 0.8)

        if base_prob <= 0.0:
            return 0.0, 0.0

        adjustments = self._calculate_ou_adjustments(features, minute)
        adjusted_prob = max(0.0, min(1.0, base_prob * (1.0 + adjustments)))
        confidence_factor = self._calculate_ou_confidence_factor(features, minute)
        final_confidence = max(0.0, min(1.0, base_conf * confidence_factor))

        from core.config import _clamp_prob
        return _clamp_prob(adjusted_prob), final_confidence
    
    def _calculate_ou_adjustments(self, features: Dict[str, float], minute: int) -> float:
        """Calculate adjustments for Over/Under predictions based on match context"""
        adjustments = 0.0
        current_goals = float(features.get("goals_sum", 0))
        xg_sum = float(features.get("xg_sum", 0))
        minute = max(1, int(features.get("minute", 1)))
        xg_per_minute = xg_sum / minute
        expected_goals_by_now = (xg_per_minute * minute)
        if expected_goals_by_now > 0:
            tempo_ratio = current_goals / expected_goals_by_now
            if tempo_ratio > 1.3:
                adjustments += 0.2
            elif tempo_ratio < 0.7:
                adjustments -= 0.15
        pressure_total = float(features.get("pressure_home", 0)) + float(features.get("pressure_away", 0))
        if pressure_total > 150:
            adjustments += 0.1
        elif pressure_total < 80:
            adjustments -= 0.1
        defensive_stability = float(features.get("defensive_stability", 0.5))
        if defensive_stability < 0.3:
            adjustments += 0.15
        elif defensive_stability > 0.7:
            adjustments -= 0.15
        if minute > 75:
            score_diff = abs(float(features.get("goals_h", 0) - features.get("goals_a", 0)))
            if score_diff <= 1:
                adjustments += 0.1
            elif current_goals == 0:
                adjustments += 0.05
        goals_last_15 = float(features.get("goals_last_15", 0))
        if goals_last_15 >= 2:
            adjustments += 0.1
        elif goals_last_15 == 0 and minute > 30:
            adjustments -= 0.05
        return adjustments
    
    def _calculate_ou_confidence_factor(self, features: Dict[str, float], minute: int) -> float:
        """Calculate confidence factor for Over/Under predictions"""
        confidence = 1.0
        xg_available = float(features.get("xg_sum", 0)) > 0
        pressure_available = (float(features.get("pressure_home", 0)) > 0) or (float(features.get("pressure_away", 0)) > 0)
        if not xg_available:
            confidence *= 0.7
        if not pressure_available:
            confidence *= 0.8
        progression_factor = min(1.0, int(features.get("minute", 0)) / 60.0)
        confidence *= (0.5 + 0.5 * progression_factor)
        total_events = float(features.get("sot_sum", 0) + features.get("cor_sum", 0) + features.get("goals_sum", 0))
        if total_events < 5 and minute > 30:
            confidence *= 0.8
        return float(confidence)
    
    def _predict_1x2_advanced(self, features: Dict[str, float], minute: int) -> Tuple[float, float, float]:
        """Advanced 1X2 prediction (Home, Away, Draw probabilities)"""
        base_prob_h, conf_h = ensemble_predictor.predict_ensemble(features, "1X2_HOME", minute)
        base_prob_a, conf_a = ensemble_predictor.predict_ensemble(features, "1X2_AWAY", minute)
        total = base_prob_h + base_prob_a
        if total > 0:
            base_prob_h /= total
            base_prob_a /= total
        prob_h = self._adjust_1x2_probability(base_prob_h, features, minute, is_home=True)
        prob_a = self._adjust_1x2_probability(base_prob_a, features, minute, is_home=False)
        total_adj = prob_h + prob_a
        if total_adj > 0:
            prob_h /= total_adj
            prob_a /= total_adj
        confidence = (conf_h + conf_a) / 2.0
        return float(prob_h), float(prob_a), float(confidence)
    
    def _adjust_1x2_probability(self, base_prob: float, features: Dict[str, float], 
                              minute: int, is_home: bool) -> float:
        """Adjust 1X2 probability based on game context"""
        adjustments = 0.0
        momentum = (float(features.get("pressure_home", 0)) if is_home else float(features.get("pressure_away", 0))) / 100.0
        adjustments += momentum * 0.15
        score_diff = float(features.get("goals_h", 0) - features.get("goals_a", 0))
        psychological = (score_diff * 0.1) if is_home else (-score_diff * 0.1)
        adjustments += psychological
        urgency = float(features.get("home_urgency", 0) if is_home else features.get("away_urgency", 0))
        adjustments += urgency * 0.08
        efficiency = float(features.get("home_efficiency", 1.0) if is_home else features.get("away_efficiency", 1.0))
        adjustments += (efficiency - 1.0) * 0.1
        adjusted_prob = base_prob * (1 + adjustments)
        return max(0.0, min(1.0, float(adjusted_prob)))

# Global market utilities
market_validator = MarketValidator()
odds_processor = OddsProcessor()
market_predictor = MarketSpecificPredictor()

# Export important functions for use in other modules
__all__ = [
    'market_validator',
    'odds_processor', 
    'market_predictor',
    '_parse_ou_line_from_suggestion',
    '_odds_key_for_market',
    '_market_family',
    'market_cutoff_ok',
    '_candidate_is_sane',
    '_inplay_1x2_sanity_ok',
    '_min_odds_for_market',
    '_ev',
    'calculate_ev',
    'calculate_ev_percentage',
    'calculate_ev_bps'
]
