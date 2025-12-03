import numpy as np
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime

class FeatureEngineer:
    """Feature engineering for betting predictions"""
    
    def __init__(self):
        self.feature_cache = {}
        
    def create_match_features(self, match_data: Dict, historical_data: Dict = None) -> Dict[str, np.ndarray]:
        """Create comprehensive features for match prediction"""
        
        features = {}
        
        # Basic features
        basic_features = self._extract_basic_features(match_data)
        
        # Statistical features
        statistical_features = self._extract_statistical_features(match_data)
        
        # Momentum features
        momentum_features = self._extract_momentum_features(match_data)
        
        # Historical features if available
        if historical_data:
            historical_features = self._extract_historical_features(match_data, historical_data)
        else:
            historical_features = {}
        
        # Combine for each prediction type
        features['1X2'] = self._combine_1x2_features(
            basic_features, statistical_features, 
            momentum_features, historical_features
        )
        
        features['over_under'] = self._combine_ou_features(
            basic_features, statistical_features,
            momentum_features, historical_features
        )
        
        features['btts'] = self._combine_btts_features(
            basic_features, statistical_features,
            momentum_features, historical_features
        )
        
        return features
    
    def _extract_basic_features(self, match_data: Dict) -> Dict[str, float]:
        """Extract basic match features"""
        
        fixture = match_data.get('fixture', {})
        teams = match_data.get('teams', {})
        goals = match_data.get('goals', {})
        score = match_data.get('score', {})
        
        features = {
            'match_minute': fixture.get('status', {}).get('elapsed', 0),
            'home_score': goals.get('home', 0),
            'away_score': goals.get('away', 0),
            'total_goals': goals.get('home', 0) + goals.get('away', 0),
            'goal_difference': goals.get('home', 0) - goals.get('away', 0),
            'is_home_team': 1,  # From perspective of home team
            'league_id': match_data.get('league', {}).get('id', 0),
            'league_level': self._get_league_level(match_data.get('league', {}).get('id', 0))
        }
        
        # Half-time score if available
        halftime = score.get('halftime', {})
        if halftime:
            features.update({
                'ht_home_score': halftime.get('home', 0),
                'ht_away_score': halftime.get('away', 0),
                'ht_total_goals': halftime.get('home', 0) + halftime.get('away', 0)
            })
        
        return features
    
    def _extract_statistical_features(self, match_data: Dict) -> Dict[str, float]:
        """Extract statistical features from match data"""
        
        statistics = match_data.get('statistics', [])
        features = {}
        
        if not statistics:
            return features
        
        # Parse statistics
        for team_stats in statistics:
            team_id = team_stats.get('team', {}).get('id')
            stats_list = team_stats.get('statistics', [])
            
            prefix = 'home_' if team_id == match_data['teams']['home']['id'] else 'away_'
            
            for stat in stats_list:
                stat_type = stat.get('type', '').lower().replace(' ', '_')
                value = stat.get('value')
                
                if value is not None:
                    # Convert percentage strings
                    if isinstance(value, str) and '%' in value:
                        try:
                            value = float(value.strip('%'))
                        except:
                            value = 0
                    
                    features[f'{prefix}{stat_type}'] = float(value) if value else 0
        
        # Calculate derived statistics
        if 'home_shots_on_goal' in features and 'away_shots_on_goal' in features:
            features['total_shots_on_goal'] = features['home_shots_on_goal'] + features['away_shots_on_goal']
            features['shot_difference'] = features['home_shots_on_goal'] - features['away_shots_on_goal']
        
        if 'home_possession' in features:
            features['possession_difference'] = features['home_possession'] - (100 - features.get('home_possession', 0))
        
        return features
    
    def _extract_momentum_features(self, match_data: Dict) -> Dict[str, float]:
        """Extract momentum-based features"""
        
        events = match_data.get('events', [])
        features = {
            'recent_goals': 0,
            'recent_cards': 0,
            'recent_substitutions': 0,
            'momentum_score': 0,
            'pressure_index': 0
        }
        
        if not events:
            return features
        
        # Analyze recent events (last 15 minutes)
        current_minute = match_data['fixture']['status']['elapsed']
        recent_window = max(current_minute - 15, 0)
        
        recent_events = [
            e for e in events 
            if e.get('time', {}).get('elapsed', 0) >= recent_window
        ]
        
        # Count event types
        goal_events = ['Goal', 'Penalty']
        card_events = ['Yellow Card', 'Red Card', 'Yellow->Red']
        
        for event in recent_events:
            event_type = event.get('type', '')
            event_detail = event.get('detail', '')
            
            if event_type in goal_events or event_detail in goal_events:
                features['recent_goals'] += 1
                
                # Weight recent goals heavily for momentum
                team_id = event.get('team', {}).get('id')
                if team_id == match_data['teams']['home']['id']:
                    features['momentum_score'] += 2  # Home goal
                else:
                    features['momentum_score'] -= 2  # Away goal
            
            elif event_type in card_events or event_detail in card_events:
                features['recent_cards'] += 1
                
                # Cards can indicate pressure
                team_id = event.get('team', {}).get('id')
                if team_id == match_data['teams']['home']['id']:
                    features['pressure_index'] += 1  # Home team under pressure
                else:
                    features['pressure_index'] -= 1  # Away team under pressure
            
            elif event_type == 'substitution':
                features['recent_substitutions'] += 1
        
        # Calculate momentum score
        features['momentum_score'] += features.get('shot_difference', 0) * 0.1
        features['momentum_score'] += features.get('possession_difference', 0) * 0.05
        
        return features
    
    def _extract_historical_features(self, match_data: Dict, historical_data: Dict) -> Dict[str, float]:
        """Extract historical/contextual features"""
        
        features = {}
        
        # Team form (would need historical data)
        # This is a placeholder for actual implementation
        features['home_form'] = historical_data.get('home_form', 0.5)
        features['away_form'] = historical_data.get('away_form', 0.5)
        features['h2h_advantage'] = historical_data.get('h2h_advantage', 0)
        
        # Recent performance metrics
        features['home_goals_scored_avg'] = historical_data.get('home_goals_scored_avg', 1.5)
        features['home_goals_conceded_avg'] = historical_data.get('home_goals_conceded_avg', 1.2)
        features['away_goals_scored_avg'] = historical_data.get('away_goals_scored_avg', 1.3)
        features['away_goals_conceded_avg'] = historical_data.get('away_goals_conceded_avg', 1.4)
        
        # Calculate expected goals difference
        features['expected_goal_difference'] = (
            features['home_goals_scored_avg'] - features['away_goals_conceded_avg']
        ) - (
            features['away_goals_scored_avg'] - features['home_goals_conceded_avg']
        )
        
        return features
    
    def _combine_1x2_features(self, basic: Dict, stats: Dict, 
                             momentum: Dict, historical: Dict) -> np.ndarray:
        """Combine features for 1X2 prediction"""
        
        features = [
            basic.get('match_minute', 0) / 90,  # Match progress
            basic.get('home_score', 0),
            basic.get('away_score', 0),
            basic.get('goal_difference', 0),
            
            # Statistics
            stats.get('home_possession', 50) / 100,
            stats.get('home_shots_on_goal', 0),
            stats.get('away_shots_on_goal', 0),
            stats.get('shot_difference', 0),
            stats.get('home_pass_accuracy', 50) / 100,
            stats.get('away_pass_accuracy', 50) / 100,
            
            # Momentum
            momentum.get('momentum_score', 0),
            momentum.get('recent_goals', 0),
            momentum.get('pressure_index', 0),
            
            # Historical
            historical.get('home_form', 0.5),
            historical.get('away_form', 0.5),
            historical.get('h2h_advantage', 0),
            historical.get('expected_goal_difference', 0),
            
            # Context
            basic.get('league_level', 1) / 5,
            basic.get('is_home_team', 1)
        ]
        
        return np.array(features, dtype=np.float32)
    
    def _combine_ou_features(self, basic: Dict, stats: Dict,
                            momentum: Dict, historical: Dict) -> np.ndarray:
        """Combine features for Over/Under prediction"""
        
        features = [
            basic.get('match_minute', 0) / 90,
            basic.get('total_goals', 0),
            basic.get('home_score', 0),
            basic.get('away_score', 0),
            
            # Attack statistics
            stats.get('total_shots_on_goal', 0),
            stats.get('home_shots_on_goal', 0),
            stats.get('away_shots_on_goal', 0),
            stats.get('home_shots_total', 0),
            stats.get('away_shots_total', 0),
            stats.get('home_corners', 0),
            stats.get('away_corners', 0),
            
            # Attack intensity
            (stats.get('home_shots_on_goal', 0) + stats.get('away_shots_on_goal', 0)) 
            / max(basic.get('match_minute', 1), 1),
            
            # Momentum
            momentum.get('recent_goals', 0),
            momentum.get('momentum_score', 0),
            
            # Historical
            historical.get('home_goals_scored_avg', 1.5),
            historical.get('home_goals_conceded_avg', 1.2),
            historical.get('away_goals_scored_avg', 1.3),
            historical.get('away_goals_conceded_avg', 1.4),
            
            # Expected goals
            stats.get('home_expected_goals', 0),
            stats.get('away_expected_goals', 0),
            
            # Context
            basic.get('league_level', 1) / 5
        ]
        
        return np.array(features, dtype=np.float32)
    
    def _combine_btts_features(self, basic: Dict, stats: Dict,
                              momentum: Dict, historical: Dict) -> np.ndarray:
        """Combine features for BTTS prediction"""
        
        both_scored = int(basic.get('home_score', 0) > 0 and basic.get('away_score', 0) > 0)
        
        features = [
            basic.get('match_minute', 0) / 90,
            both_scored,
            basic.get('home_score', 0),
            basic.get('away_score', 0),
            
            # Attack statistics
            stats.get('home_shots_on_goal', 0),
            stats.get('away_shots_on_goal', 0),
            stats.get('home_shots_inside_box', 0),
            stats.get('away_shots_inside_box', 0),
            stats.get('home_dangerous_attacks', 0),
            stats.get('away_dangerous_attacks', 0),
            
            # Attack intensity per minute
            stats.get('home_dangerous_attacks', 0) / max(basic.get('match_minute', 1), 1),
            stats.get('away_dangerous_attacks', 0) / max(basic.get('match_minute', 1), 1),
            
            # Defense statistics
            stats.get('home_tackles', 0),
            stats.get('away_tackles', 0),
            stats.get('home_clearances', 0),
            stats.get('away_clearances', 0),
            
            # Momentum
            momentum.get('recent_goals', 0),
            momentum.get('pressure_index', 0),
            
            # Historical
            historical.get('home_goals_scored_avg', 1.5),
            historical.get('home_goals_conceded_avg', 1.2),
            historical.get('away_goals_scored_avg', 1.3),
            historical.get('away_goals_conceded_avg', 1.4),
            
            # Context
            basic.get('league_level', 1) / 5
        ]
        
        return np.array(features, dtype=np.float32)
    
    def _get_league_level(self, league_id: int) -> int:
        """Get league quality level"""
        
        # Top leagues
        top_leagues = [39, 78, 61, 135, 94]  # Premier League, Bundesliga, Ligue 1, Serie A, La Liga
        
        # Second tier
        second_tier = [40, 79, 62, 136, 95]
        
        if league_id in top_leagues:
            return 1
        elif league_id in second_tier:
            return 2
        else:
            return 3
    
    def normalize_features(self, features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Normalize features to [0, 1] range"""
        
        normalized = {}
        
        for key, feature_array in features.items():
            if len(feature_array) == 0:
                continue
            
            # Min-max normalization
            min_val = np.min(feature_array)
            max_val = np.max(feature_array)
            
            if max_val > min_val:
                normalized[key] = (feature_array - min_val) / (max_val - min_val)
            else:
                normalized[key] = feature_array
        
        return normalized
