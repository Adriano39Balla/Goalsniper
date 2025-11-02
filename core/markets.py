import os
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from core.config import config

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

def _ev(prob: float, odds: float) -> float:
    """Return expected value as decimal (e.g. 0.05 = +5%)."""
    try:
        return float(prob) * max(0.0, float(odds)) - 1.0
    except Exception:
        return -1.0

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

# Global market utilities
market_validator = MarketValidator()
odds_processor = OddsProcessor()

# Export important functions for use in other modules
__all__ = [
    'market_validator',
    'odds_processor', 
    '_parse_ou_line_from_suggestion',
    '_odds_key_for_market',
    '_market_family',
    'market_cutoff_ok',
    '_candidate_is_sane',
    '_inplay_1x2_sanity_ok',
    '_min_odds_for_market',
    '_ev'
]
