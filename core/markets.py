import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from core.config import config

log = logging.getLogger("goalsniper.markets")

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
        market_family = self._get_market_family(market, suggestion)
        cutoff = self.market_cutoffs.get(market_family)
        
        if cutoff is None:
            # Use global cutoff or default
            tip_max_minute = self._get_tip_max_minute()
            if tip_max_minute is None:
                tip_max_minute = max(0, int(os.getenv("TOTAL_MATCH_MINUTES", "95")) - 5)
            cutoff = tip_max_minute
        
        return minute <= cutoff
    
    def _get_market_family(self, market: str, suggestion: str) -> str:
        """Get market family for cutoff purposes"""
        market_upper = market.upper()
        
        if market_upper.startswith("OVER/UNDER") or "OVER/UNDER" in market_upper:
            return "OU"
        if market_upper == "BTTS" or "BTTS" in market_upper:
            return "BTTS"
        if market_upper == "1X2" or "WINNER" in market_upper or "MATCH WINNER" in market_upper:
            return "1X2"
        if market_upper.startswith("PRE "):
            return self._get_market_family(market_upper[4:], suggestion)
        
        return market_upper
    
    def _get_tip_max_minute(self) -> Optional[int]:
        """Get global tip maximum minute"""
        try:
            return int(float(os.getenv("TIP_MAX_MINUTE", ""))) if os.getenv("TIP_MAX_MINUTE") else None
        except:
            return None
    
    def is_prediction_sane(self, suggestion: str, features: Dict[str, float]) -> bool:
        """Check if prediction makes sense given current match state"""
        minute = int(features.get("minute", 0))
        goals_sum = features.get("goals_sum", 0)
        
        # Check for absurd Over/Under predictions
        if suggestion.startswith("Under"):
            line = self._parse_ou_line_from_suggestion(suggestion)
            if line and goals_sum > line:
                return False  # Can't go under if already over
        
        # Check for absurd BTTS predictions
        if suggestion == "BTTS: No" and goals_sum >= 2:
            goals_h = features.get("goals_h", 0)
            goals_a = features.get("goals_a", 0)
            if goals_h > 0 and goals_a > 0:
                return False
        
        return True
    
    def _parse_ou_line_from_suggestion(self, suggestion: str) -> Optional[float]:
        """Parse OU line from suggestion text"""
        try:
            if "Over" in suggestion or "Under" in suggestion:
                import re
                match = re.search(r'(\d+(?:\.\d+)?)', suggestion)
                if match:
                    return float(match.group(1))
            return None
        except:
            return None

class OddsProcessor:
    """Odds processing and validation"""
    
    def __init__(self):
        self.odds_quality_min = config.odds.quality_min
    
    def calculate_ev(self, probability: float, odds: float) -> float:
        """Calculate expected value"""
        try:
            return float(probability) * max(0.0, float(odds)) - 1.0
        except:
            return -1.0
    
    def get_min_odds_for_market(self, market: str) -> float:
        """Get minimum odds for a market"""
        if market.startswith("Over/Under"):
            return config.odds.min_odds_ou
        if market == "BTTS":
            return config.odds.min_odds_btts
        if market == "1X2":
            return config.odds.min_odds_1x2
        return 1.01
    
    def validate_odds_quality(self, odds_data: Dict, probability: float, market: str) -> bool:
        """Validate odds quality"""
        if not odds_data:
            return False
        
        # Implementation of odds quality validation
        # [Your existing odds quality logic]
        return True

# Global market utilities
market_validator = MarketValidator()
odds_processor = OddsProcessor()
