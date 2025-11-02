import logging
import numpy as np
from typing import Dict, Any, List
from datetime import datetime
from zoneinfo import ZoneInfo

log = logging.getLogger("goalsniper.features")

# Core utility functions for feature calculation
def _calculate_pressure(feat: Dict[str, float], side: str) -> float:
    """Calculate pressure for home or away team"""
    suffix = "_h" if side == "home" else "_a"
    possession = feat.get(f"pos{suffix}", 50)
    shots = feat.get(f"sot{suffix}", 0)
    xg = feat.get(f"xg{suffix}", 0)
    possession_norm = possession / 100.0
    shots_norm = min(shots / 10.0, 1.0)
    xg_norm = min(xg / 3.0, 1.0)
    return (possession_norm * 0.3 + shots_norm * 0.4 + xg_norm * 0.3) * 100

def _calculate_xg_momentum(feat: Dict[str, float]) -> float:
    """Calculate xG momentum (goals vs expected goals)"""
    total_xg = feat.get("xg_sum", 0)
    total_goals = feat.get("goals_sum", 0)
    if total_xg <= 0:
        return 0.0
    return (total_goals - total_xg) / max(1, total_xg)

def _recent_xg_impact(feat: Dict[str, float], minute: int) -> float:
    """Calculate recent xG impact per minute"""
    if minute <= 0:
        return 0.0
    xg_per_minute = feat.get("xg_sum", 0) / minute
    return xg_per_minute * 90

def _defensive_stability(feat: Dict[str, float]) -> float:
    """Calculate defensive stability for both teams"""
    goals_conceded_h = feat.get("goals_a", 0)
    goals_conceded_a = feat.get("goals_h", 0)
    xg_against_h = feat.get("xg_a", 0)
    xg_against_a = feat.get("xg_h", 0)
    defensive_efficiency_h = 1 - (goals_conceded_h / max(1, xg_against_h)) if xg_against_h > 0 else 1.0
    defensive_efficiency_a = 1 - (goals_conceded_a / max(1, xg_against_a)) if xg_against_a > 0 else 1.0
    return (defensive_efficiency_h + defensive_efficiency_a) / 2

# Event analysis functions
def _count_goals_since(events: List[dict], current_minute: int, window: int) -> int:
    """Count goals in the last N minutes"""
    cutoff = current_minute - window
    goals = 0
    for event in events:
        minute = event.get('time', {}).get('elapsed', 0)
        if minute >= cutoff and event.get('type') == 'Goal':
            goals += 1
    return goals

def _count_shots_since(events: List[dict], current_minute: int, window: int) -> int:
    """Count shots in the last N minutes"""
    cutoff = current_minute - window
    shots = 0
    shot_types = {'Shot', 'Missed Shot', 'Shot on Target', 'Saved Shot'}
    for event in events:
        minute = event.get('time', {}).get('elapsed', 0)
        if minute >= cutoff and event.get('type') in shot_types:
            shots += 1
    return shots

def _count_cards_since(events: List[dict], current_minute: int, window: int) -> int:
    """Count cards in the last N minutes"""
    cutoff = current_minute - window
    cards = 0
    for event in events:
        minute = event.get('time', {}).get('elapsed', 0)
        if minute >= cutoff and event.get('type') == 'Card':
            cards += 1
    return cards

# Standalone feature extraction functions
def extract_basic_features(match_data: Dict[str, Any]) -> Dict[str, float]:
    """Standalone function for basic feature extraction (for compatibility)"""
    return FeatureEngineer()._extract_basic_features(match_data)

def extract_enhanced_features(match_data: Dict[str, Any]) -> Dict[str, float]:
    """Standalone function for enhanced feature extraction (for compatibility)"""
    base_feat = extract_basic_features(match_data)
    minute = base_feat.get("minute", 0)
    events = match_data.get("events", [])
    
    base_feat.update({
        "goals_last_15": float(_count_goals_since(events, minute, 15)),
        "shots_last_15": float(_count_shots_since(events, minute, 15)),
        "cards_last_15": float(_count_cards_since(events, minute, 15)),
        "pressure_home": _calculate_pressure(base_feat, "home"),
        "pressure_away": _calculate_pressure(base_feat, "away"),
        "score_advantage": base_feat.get("goals_h", 0) - base_feat.get("goals_a", 0),
        "xg_momentum": _calculate_xg_momentum(base_feat),
        "recent_xg_impact": _recent_xg_impact(base_feat, minute),
        "defensive_stability": _defensive_stability(base_feat)
    })
    return base_feat

# Data quality and validation functions
def stats_coverage_ok(feat: Dict[str, float], minute: int) -> bool:
    """Check if we have sufficient data coverage for predictions"""
    # Only reject if we have absolutely NO data at all
    has_any_data = any([
        feat.get('pos_h', 0) > 0, feat.get('pos_a', 0) > 0,
        feat.get('sot_h', 0) > 0, feat.get('sot_a', 0) > 0, 
        feat.get('goals_h', 0) > 0, feat.get('goals_a', 0) > 0,
        feat.get('cor_h', 0) > 0, feat.get('cor_a', 0) > 0,
        feat.get('sh_total_h', 0) > 0, feat.get('sh_total_a', 0) > 0
    ])
    
    # After 30 minutes, require at least SOME data
    if minute > 30:
        return has_any_data
    
    # Before 30 minutes, be very lenient
    return True

def is_feed_stale(fid: int, match_data: Dict[str, Any], minute: int) -> bool:
    """Check if the data feed is stale"""
    # Note: This function needs access to STATS_CACHE from main module
    # For now, we'll provide a basic implementation
    # The main module should handle the full implementation with cache access
    events = match_data.get("events", [])
    if events:
        latest_event_minute = max([ev.get('time', {}).get('elapsed', 0) for ev in events], default=0)
        if minute - latest_event_minute > 10 and minute > 30:
            return True
    return False

class FeatureEngineer:
    """Advanced feature engineering with temporal patterns and game context"""
    
    def __init__(self):
        self.feature_cache = {}
        self.temporal_patterns = {}
        self.berlin_tz = ZoneInfo("Europe/Berlin")
    
    def extract_advanced_features(self, match_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract advanced features including temporal patterns and game context"""
        base_features = self._extract_basic_features(match_data)
        
        temporal_features = self._extract_temporal_patterns(match_data, base_features)
        base_features.update(temporal_features)
        
        context_features = self._extract_game_context(match_data, base_features)
        base_features.update(context_features)
        
        strength_features = self._extract_team_strength_indicators(match_data, base_features)
        base_features.update(strength_features)
        
        # Add pressure calculations
        base_features.update({
            "pressure_home": self._calculate_pressure(base_features, "home"),
            "pressure_away": self._calculate_pressure(base_features, "away"),
            "xg_momentum": self._calculate_xg_momentum(base_features),
            "recent_xg_impact": self._recent_xg_impact(base_features, int(base_features.get("minute", 0))),
            "defensive_stability": self._defensive_stability(base_features)
        })
        
        return base_features
    
    def _extract_basic_features(self, match_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract basic match features - MUST MATCH training data extraction"""
        # ... your existing implementation ...
        home = match_data["teams"]["home"]["name"]
        away = match_data["teams"]["away"]["name"]
        goals = match_data.get("goals", {})
        gh = goals.get("home", 0) or 0
        ga = goals.get("away", 0) or 0
        
        fixture = match_data.get("fixture", {})
        status = fixture.get("status", {})
        minute = int(status.get("elapsed", 0))
        
        stats = {}
        for stat_entry in match_data.get("statistics", []):
            team = stat_entry.get("team", {})
            team_name = team.get("name")
            if team_name:
                stats[team_name] = {
                    item.get("type", ""): item.get("value") 
                    for item in stat_entry.get("statistics", [])
                }
        
        home_stats = stats.get(home, {})
        away_stats = stats.get(away, {})
        
        # Extract statistics with consistent naming
        sot_h = self._num(home_stats.get("Shots on Goal", home_stats.get("Shots on Target", 0)))
        sot_a = self._num(away_stats.get("Shots on Goal", away_stats.get("Shots on Target", 0)))
        shots_total_h = self._num(home_stats.get("Total Shots", 0))
        shots_total_a = self._num(away_stats.get("Total Shots", 0))
        corners_h = self._num(home_stats.get("Corner Kicks", 0))
        corners_a = self._num(away_stats.get("Corner Kicks", 0))
        possession_h = self._pos_pct(home_stats.get("Ball Possession", 0))
        possession_a = self._pos_pct(away_stats.get("Ball Possession", 0))
        
        # xG handling - same logic as training
        xg_h = self._num(home_stats.get("Expected Goals", 0))
        xg_a = self._num(away_stats.get("Expected Goals", 0))
        
        # Only estimate if real xG is not available
        if xg_h == 0 and xg_a == 0:
            xg_h = sot_h * 0.3
            xg_a = sot_a * 0.3
            log.info(f"[XG_ESTIMATE] Using estimated xG: {xg_h:.2f}-{xg_a:.2f}")
        else:
            log.info(f"[XG_REAL] Using API xG: {xg_h:.2f}-{xg_a:.2f}")
        
        # Card statistics
        yellow_h, yellow_a, red_h, red_a = self._extract_cards(match_data.get("events", []), home, away)
        
        return {
            "minute": float(minute),
            "goals_h": float(gh), "goals_a": float(ga),
            "goals_sum": float(gh + ga), "goals_diff": float(gh - ga),
            
            "xg_h": float(xg_h), "xg_a": float(xg_a),
            "xg_sum": float(xg_h + xg_a), "xg_diff": float(xg_h - xg_a),
            
            "sot_h": float(sot_h), "sot_a": float(sot_a),
            "sot_sum": float(sot_h + sot_a),
            
            "sh_total_h": float(shots_total_h), "sh_total_a": float(shots_total_a),
            
            "cor_h": float(corners_h), "cor_a": float(corners_a),
            "cor_sum": float(corners_h + corners_a),
            
            "pos_h": float(possession_h), "pos_a": float(possession_a),
            "pos_diff": float(possession_h - possession_a),
            
            "red_h": float(red_h), "red_a": float(red_a),
            "red_sum": float(red_h + red_a),
            
            "yellow_h": float(yellow_h), "yellow_a": float(yellow_a)
        }
    
    def _extract_temporal_patterns(self, match_data: Dict[str, Any], features: Dict[str, float]) -> Dict[str, float]:
        """Extract temporal patterns from match events"""
        minute = int(features.get("minute", 0))
        events = match_data.get("events", [])
        temporal_features = {}
        
        for window in [10, 15, 20]:
            temporal_features[f"goals_last_{window}"] = float(
                self._count_events_since(events, minute, window, 'Goal')
            )
            temporal_features[f"shots_last_{window}"] = float(
                self._count_events_since(events, minute, window, 
                                       {'Shot', 'Missed Shot', 'Shot on Target'})
            )
            temporal_features[f"cards_last_{window}"] = float(
                self._count_events_since(events, minute, window, {'Card'})
            )
        
        if minute > 15:
            goals_0_15 = temporal_features.get("goals_last_15", 0.0)
            goals_15_30 = float(self._count_events_between(events, max(0, minute-30), minute-15, 'Goal'))
            temporal_features["goal_acceleration"] = float(goals_0_15 - goals_15_30)
        
        temporal_features["time_decayed_xg"] = self._calculate_time_decayed_xg(features, minute)
        temporal_features["recent_pressure"] = self._calculate_recent_pressure(events, minute)
        
        return temporal_features
    
    def _extract_game_context(self, match_data: Dict[str, Any], features: Dict[str, float]) -> Dict[str, float]:
        """Extract game context features"""
        context_features = {}
        minute = int(features.get("minute", 0))
        score_diff = float(features.get("goals_h", 0) - features.get("goals_a", 0))
        
        context_features["game_state"] = self._classify_game_state(score_diff, minute)
        context_features["home_urgency"] = self._calculate_urgency(score_diff, minute, is_home=True)
        context_features["away_urgency"] = self._calculate_urgency(-score_diff, minute, is_home=False)
        context_features["defensive_risk"] = self._calculate_defensive_risk(features, minute)
        context_features["attacking_risk"] = self._calculate_attacking_risk(features, minute)
        context_features["match_importance"] = self._estimate_match_importance(match_data)
        
        return context_features
    
    def _extract_team_strength_indicators(self, match_data: Dict[str, Any], features: Dict[str, float]) -> Dict[str, float]:
        """Extract team strength indicators"""
        strength_features = {}
        pressure_home = float(features.get("pressure_home", 1))
        pressure_away = float(features.get("pressure_away", 1))
        
        strength_features["home_dominance"] = pressure_home / max(1.0, pressure_away)
        strength_features["away_resilience"] = 1.0 / max(0.1, strength_features["home_dominance"])
        
        xg_h = float(features.get("xg_h", 0.1))
        xg_a = float(features.get("xg_a", 0.1))
        goals_h = float(features.get("goals_h", 0))
        goals_a = float(features.get("goals_a", 0))
        
        strength_features["home_efficiency"] = goals_h / max(0.1, xg_h)
        strength_features["away_efficiency"] = goals_a / max(0.1, xg_a)
        strength_features["home_defensive_stability"] = 1.0 - (goals_a / max(0.1, xg_a))
        strength_features["away_defensive_stability"] = 1.0 - (goals_h / max(0.1, xg_h))
        
        return strength_features
    
    # Pressure and momentum calculation methods
    def _calculate_pressure(self, feat: Dict[str, float], side: str) -> float:
        return _calculate_pressure(feat, side)
    
    def _calculate_xg_momentum(self, feat: Dict[str, float]) -> float:
        return _calculate_xg_momentum(feat)
    
    def _recent_xg_impact(self, feat: Dict[str, float], minute: int) -> float:
        return _recent_xg_impact(feat, minute)
    
    def _defensive_stability(self, feat: Dict[str, float]) -> float:
        return _defensive_stability(feat)
    
    # ... rest of your existing methods (_num, _pos_pct, _extract_cards, etc.) ...
    def _num(self, value) -> float:
        """Convert any value to float, handling percentages"""
        try:
            if isinstance(value, str) and value.endswith("%"):
                return float(value[:-1])
            return float(value or 0)
        except:
            return 0.0
    
    def _pos_pct(self, value) -> float:
        """Convert possession percentage to float"""
        try:
            return float(str(value).replace("%", "").strip() or 0)
        except:
            return 0.0
    
    def _extract_cards(self, events: List[Dict], home_team: str, away_team: str) -> tuple:
        """Extract card statistics from events"""
        yellow_h = yellow_a = red_h = red_a = 0
        
        for event in events:
            if event.get("type", "").lower() == "card":
                detail = (event.get("detail", "") or "").lower()
                team = (event.get("team", {}).get("name") or "")
                
                if "yellow" in detail and "second" not in detail:
                    if team == home_team:
                        yellow_h += 1
                    elif team == away_team:
                        yellow_a += 1
                if "red" in detail or "second yellow" in detail:
                    if team == home_team:
                        red_h += 1
                    elif team == away_team:
                        red_a += 1
        
        return yellow_h, yellow_a, red_h, red_a
    
    def _count_events_since(self, events: List[Dict], current_minute: int, 
                          window: int, event_types: any) -> int:
        """Count events in the last N minutes"""
        cutoff = current_minute - window
        count = 0
        
        for event in events:
            minute = int((event.get('time', {}) or {}).get('elapsed', 0) or 0)
            if minute >= cutoff:
                event_type = event.get('type')
                if isinstance(event_types, str):
                    if event_type == event_types:
                        count += 1
                else:
                    if event_type in event_types:
                        count += 1
        return count
    
    def _count_events_between(self, events: List[Dict], start_minute: int, 
                            end_minute: int, event_type: str) -> int:
        """Count events between two minute marks"""
        count = 0
        for event in events:
            minute = int((event.get('time', {}) or {}).get('elapsed', 0) or 0)
            if start_minute <= minute <= end_minute and event.get('type') == event_type:
                count += 1
        return count
    
    def _calculate_time_decayed_xg(self, features: Dict[str, float], minute: int) -> float:
        """Calculate time-decayed xG"""
        if minute <= 0:
            return 0.0
        xg_sum = float(features.get("xg_sum", 0))
        decay_factor = 0.9
        recent_weight = decay_factor ** (minute / 10.0)
        return float((xg_sum / minute) * recent_weight * 90.0)
    
    def _calculate_recent_pressure(self, events: List[Dict], minute: int) -> float:
        """Calculate recent pressure from events"""
        recent_events = self._count_events_since(
            events, minute, 10, 
            {'Shot', 'Shot on Target', 'Corner', 'Dangerous Attack'}
        )
        return float(min(1.0, recent_events / 10.0))
    
    def _classify_game_state(self, score_diff: float, minute: int) -> float:
        """Classify current game state"""
        if minute < 30:
            return 0.0
        if abs(score_diff) >= 3:
            return 1.0
        if abs(score_diff) == 2 and minute > 70:
            return 0.8
        if abs(score_diff) == 1 and minute > 75:
            return 0.9
        if score_diff == 0 and minute > 80:
            return 0.7
        return 0.5
    
    def _calculate_urgency(self, score_diff: float, minute: int, is_home: bool) -> float:
        """Calculate team urgency"""
        urgency_score = -score_diff if is_home else score_diff
        time_pressure = max(0.0, (minute - 60) / 30.0)
        return float(max(0.0, urgency_score * time_pressure))
    
    def _calculate_defensive_risk(self, features: Dict[str, float], minute: int) -> float:
        """Calculate defensive risk"""
        goals_conceded = float(features.get("goals_a", 0) + features.get("goals_h", 0))
        xg_against = float(features.get("xg_a", 0) + features.get("xg_h", 0))
        defensive_efficiency = goals_conceded / max(0.1, xg_against)
        fatigue_factor = min(1.0, minute / 90.0)
        return float(defensive_efficiency * fatigue_factor)
    
    def _calculate_attacking_risk(self, features: Dict[str, float], minute: int) -> float:
        """Calculate attacking risk"""
        pressure = (float(features.get("pressure_home", 0)) + float(features.get("pressure_away", 0))) / 2.0
        home_urgency = float(features.get("home_urgency", 0))
        away_urgency = float(features.get("away_urgency", 0))
        urgency = (home_urgency + away_urgency) / 2.0
        return float((pressure / 100.0) * urgency)
    
    def _estimate_match_importance(self, match_data: Dict[str, Any]) -> float:
        """Estimate match importance"""
        league = match_data.get("league", {}) or {}
        league_name = str(league.get("name", "") or "").lower()
        
        if any(w in league_name for w in ["champions league", "europa league", "premier league"]):
            return 0.9
        if any(w in league_name for w in ["cup", "knockout", "playoff"]):
            return 0.8
        return 0.5

# Global feature engineer instance
feature_engineer = FeatureEngineer()

# Export important functions for use in other modules
__all__ = [
    'feature_engineer', 
    'extract_basic_features', 
    'extract_enhanced_features',
    'stats_coverage_ok', 
    'is_feed_stale',
    '_calculate_pressure',
    '_calculate_xg_momentum',
    '_defensive_stability'
]
