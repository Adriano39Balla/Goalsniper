import logging
from typing import Dict, Any, Optional

log = logging.getLogger("goalsniper.tip_calculator")

def calculate_tip_outcome(suggestion: str, result: Dict[str, Any]) -> Optional[int]:
    """
    Calculate tip outcome (1=win, 0=loss, None=push/no result)
    """
    home_goals = int(result.get("final_goals_h") or 0)
    away_goals = int(result.get("final_goals_a") or 0)
    total_goals = home_goals + away_goals
    btts_yes = int(result.get("btts_yes") or 0)
    
    suggestion = (suggestion or "").strip()
    
    # Over/Under outcomes
    if suggestion.startswith("Over") or suggestion.startswith("Under"):
        line = _parse_ou_line_from_suggestion(suggestion)
        if line is None:
            return None
            
        if suggestion.startswith("Over"):
            if total_goals > line:
                return 1
            if abs(total_goals - line) < 1e-9:  # Push on exact line
                return None
            return 0
        else:  # Under
            if total_goals < line:
                return 1
            if abs(total_goals - line) < 1e-9:  # Push on exact line
                return None
            return 0
    
    # BTTS outcomes
    if suggestion == "BTTS: Yes":
        return 1 if btts_yes == 1 else 0
    if suggestion == "BTTS: No":
        return 1 if btts_yes == 0 else 0
    
    # 1X2 outcomes
    if suggestion == "Home Win":
        return 1 if home_goals > away_goals else 0
    if suggestion == "Away Win":
        return 1 if away_goals > home_goals else 0
    
    return None

def _parse_ou_line_from_suggestion(suggestion: str) -> Optional[float]:
    """Parse OU line from suggestion text"""
    try:
        if "Over" in suggestion or "Under" in suggestion:
            import re
            match = re.search(r'(\d+(?:\.\d+)?)', suggestion)
            if match:
                return float(match.group(1))
        return None
    except Exception:
        return None

def calculate_tip_accuracy(tips: List[Dict], results: Dict[int, Dict]) -> Dict[str, Any]:
    """
    Calculate accuracy metrics for tips
    """
    total_tips = 0
    graded_tips = 0
    winning_tips = 0
    total_stake = 0
    total_pnl = 0
    
    market_performance = {}
    
    for tip in tips:
        match_id = tip.get("match_id")
        suggestion = tip.get("suggestion")
        odds = tip.get("odds")
        
        if match_id not in results:
            continue
            
        result = results[match_id]
        outcome = calculate_tip_outcome(suggestion, result)
        
        if outcome is None:  # Push or no result
            continue
            
        total_tips += 1
        graded_tips += 1
        
        if outcome == 1:
            winning_tips += 1
            if odds:
                total_pnl += (float(odds) - 1)
                total_stake += 1
        else:
            if odds:
                total_pnl -= 1
                total_stake += 1
        
        # Track market performance
        market = tip.get("market", "Unknown")
        if market not in market_performance:
            market_performance[market] = {"graded": 0, "wins": 0, "stake": 0, "pnl": 0}
        
        market_performance[market]["graded"] += 1
        if outcome == 1:
            market_performance[market]["wins"] += 1
        if odds:
            market_performance[market]["stake"] += 1
            market_performance[market]["pnl"] += (float(odds) - 1) if outcome == 1 else -1
    
    accuracy = (winning_tips / graded_tips * 100) if graded_tips > 0 else 0
    roi = (total_pnl / total_stake * 100) if total_stake > 0 else 0
    
    return {
        "total_tips": total_tips,
        "graded_tips": graded_tips,
        "winning_tips": winning_tips,
        "accuracy": accuracy,
        "total_stake": total_stake,
        "total_pnl": total_pnl,
        "roi": roi,
        "market_performance": market_performance
    }

__all__ = ['calculate_tip_outcome', 'calculate_tip_accuracy']
