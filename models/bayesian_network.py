import pandas as pd
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
from typing import Dict, List, Tuple, Any
import logging
from joblib import dump, load
import os

logger = logging.getLogger(__name__)

class BayesianPredictor:
    def __init__(self):
        self.model = None
        self.inference = None
        self.features = []
        self.model_path = "models/bayesian_model.joblib"
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for Bayesian Network"""
        # Feature engineering
        df = df.copy()
        
        # Team strength features
        df['home_attack_strength'] = df['home_goals_avg'] / df['league_goals_avg']
        df['away_attack_strength'] = df['away_goals_avg'] / df['league_goals_avg']
        df['home_defense_strength'] = df['home_goals_conceded_avg'] / df['league_goals_avg']
        df['away_defense_strength'] = df['away_goals_conceded_avg'] / df['league_goals_avg']
        
        # Form features
        df['home_form'] = df['home_points_5games'] / 15.0  # Normalize to 0-1
        df['away_form'] = df['away_points_5games'] / 15.0
        
        # Head-to-head features
        df['h2h_goal_diff'] = df['h2h_home_goals_avg'] - df['h2h_away_goals_avg']
        
        # Discretize continuous variables for Bayesian Network
        features_to_discretize = [
            'home_attack_strength', 'away_attack_strength',
            'home_defense_strength', 'away_defense_strength',
            'home_form', 'away_form', 'h2h_goal_diff'
        ]
        
        for feature in features_to_discretize:
            if feature in df.columns:
                df[f'{feature}_cat'] = pd.cut(df[feature], bins=5, labels=False)
        
        return df
    
    def build_model(self):
        """Build Bayesian Network structure"""
        self.model = BayesianNetwork([
            # Attack strengths influence goals
            ('home_attack_strength_cat', 'home_goals'),
            ('away_attack_strength_cat', 'away_goals'),
            
            # Defense strengths influence goals conceded
            ('home_defense_strength_cat', 'away_goals'),
            ('away_defense_strength_cat', 'home_goals'),
            
            # Form influences performance
            ('home_form_cat', 'home_goals'),
            ('away_form_cat', 'away_goals'),
            
            # H2H influences both teams
            ('h2h_goal_diff_cat', 'home_goals'),
            ('h2h_goal_diff_cat', 'away_goals'),
            
            # Goals determine match outcomes
            ('home_goals', 'match_result'),
            ('away_goals', 'match_result'),
            ('home_goals', 'btts'),
            ('away_goals', 'btts'),
            ('home_goals', 'over_25'),
            ('away_goals', 'over_25')
        ])
    
    def train(self, df: pd.DataFrame):
        """Train Bayesian Network"""
        try:
            # Prepare data
            training_data = self.prepare_features(df)
            
            # Build model structure
            self.build_model()
            
            # Fit model with Bayesian estimation
            self.model.fit(
                training_data,
                estimator=BayesianEstimator,
                prior_type='BDeu',  # Bayesian Dirichlet equivalent uniform
                equivalent_sample_size=10
            )
            
            # Initialize inference engine
            self.inference = VariableElimination(self.model)
            
            # Save model
            os.makedirs('models', exist_ok=True)
            dump(self.model, self.model_path)
            logger.info("Bayesian Network trained successfully")
            
        except Exception as e:
            logger.error(f"Error training Bayesian Network: {e}")
            raise
    
    def predict(self, match_data: Dict) -> Dict[str, float]:
        """Predict probabilities for a match"""
        if not self.inference:
            raise ValueError("Model not trained")
        
        try:
            # Prepare evidence
            evidence = {}
            for feature in ['home_attack_strength_cat', 'away_attack_strength_cat',
                          'home_defense_strength_cat', 'away_defense_strength_cat',
                          'home_form_cat', 'away_form_cat', 'h2h_goal_diff_cat']:
                if feature in match_data:
                    evidence[feature] = match_data[feature]
            
            # Query probabilities
            result_probs = self.inference.query(
                variables=['match_result'],
                evidence=evidence
            )
            btts_probs = self.inference.query(
                variables=['btts'],
                evidence=evidence
            )
            over25_probs = self.inference.query(
                variables=['over_25'],
                evidence=evidence
            )
            
            return {
                'home_win': float(result_probs.values[2]),  # Assuming order: [Away, Draw, Home]
                'draw': float(result_probs.values[1]),
                'away_win': float(result_probs.values[0]),
                'btts_yes': float(btts_probs.values[1]),
                'btts_no': float(btts_probs.values[0]),
                'over_25': float(over25_probs.values[1]),
                'under_25': float(over25_probs.values[0])
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return self._get_fallback_probabilities()
    
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
    
    def load_model(self):
        """Load trained model"""
        try:
            self.model = load(self.model_path)
            self.inference = VariableElimination(self.model)
            logger.info("Bayesian Network loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
