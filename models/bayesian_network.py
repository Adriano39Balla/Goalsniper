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
        if df.empty:
            return df
            
        df = df.copy()
        
        # Create basic features if they don't exist
        if 'home_goals' not in df.columns:
            df['home_goals'] = 0
        if 'away_goals' not in df.columns:
            df['away_goals'] = 0
            
        # Calculate basic averages if needed
        if 'home_goals_avg' not in df.columns:
            df['home_goals_avg'] = df.groupby('home_team')['home_goals'].transform('mean').fillna(1.5)
        if 'away_goals_avg' not in df.columns:
            df['away_goals_avg'] = df.groupby('away_team')['away_goals'].transform('mean').fillna(1.5)
        if 'league_goals_avg' not in df.columns:
            df['league_goals_avg'] = (df['home_goals'] + df['away_goals']).mean()
        
        # Team strength features
        df['home_attack_strength'] = df['home_goals_avg'] / df['league_goals_avg'].replace(0, 1)
        df['away_attack_strength'] = df['away_goals_avg'] / df['league_goals_avg'].replace(0, 1)
        df['home_defense_strength'] = df['home_goals_avg'] / df['league_goals_avg'].replace(0, 1)  # Simplified
        df['away_defense_strength'] = df['away_goals_avg'] / df['league_goals_avg'].replace(0, 1)
        
        # Form features (simplified)
        if 'home_points_5games' not in df.columns:
            df['home_points_5games'] = 7  # Default value
        if 'away_points_5games' not in df.columns:
            df['away_points_5games'] = 7
            
        df['home_form'] = df['home_points_5games'] / 15.0
        df['away_form'] = df['away_points_5games'] / 15.0
        
        # Head-to-head features (simplified)
        if 'h2h_home_goals_avg' not in df.columns:
            df['h2h_home_goals_avg'] = df['home_goals_avg']
        if 'h2h_away_goals_avg' not in df.columns:
            df['h2h_away_goals_avg'] = df['away_goals_avg']
            
        df['h2h_goal_diff'] = df['h2h_home_goals_avg'] - df['h2h_away_goals_avg']
        
        # Discretize continuous variables for Bayesian Network
        features_to_discretize = [
            'home_attack_strength', 'away_attack_strength',
            'home_defense_strength', 'away_defense_strength',
            'home_form', 'away_form', 'h2h_goal_diff'
        ]
        
        for feature in features_to_discretize:
            if feature in df.columns:
                # Handle NaN values
                df[feature] = df[feature].fillna(0.5)
                df[f'{feature}_cat'] = pd.cut(df[feature], bins=5, labels=[0, 1, 2, 3, 4])
        
        # Create target variables for training
        df['match_result'] = np.select(
            [df['home_goals'] > df['away_goals'], 
             df['home_goals'] == df['away_goals']],
            [2, 1],  # 2=home_win, 1=draw, 0=away_win
            default=0
        )
        df['btts'] = ((df['home_goals'] > 0) & (df['away_goals'] > 0)).astype(int)
        df['over_25'] = (df['home_goals'] + df['away_goals'] > 2.5).astype(int)
        
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
            if df.empty:
                logger.warning("No data to train Bayesian Network")
                return
                
            # Prepare data
            training_data = self.prepare_features(df)
            
            # Select only columns that exist in the model
            model_columns = ['home_attack_strength_cat', 'away_attack_strength_cat',
                           'home_defense_strength_cat', 'away_defense_strength_cat',
                           'home_form_cat', 'away_form_cat', 'h2h_goal_diff_cat',
                           'home_goals', 'away_goals', 'match_result', 'btts', 'over_25']
            
            available_columns = [col for col in model_columns if col in training_data.columns]
            training_data = training_data[available_columns]
            
            # Build model structure
            self.build_model()
            
            # Fit model with Bayesian estimation
            self.model.fit(
                training_data,
                estimator=BayesianEstimator,
                prior_type='BDeu',
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
            # Create a simple fallback model
            self._create_fallback_model()
    
    def _create_fallback_model(self):
        """Create a simple fallback model when training fails"""
        from pgmpy.models import BayesianNetwork
        from pgmpy.factors.discrete import TabularCPD
        
        # Simple fallback model
        self.model = BayesianNetwork([('A', 'B')])
        # Add simple CPDs
        cpd_a = TabularCPD('A', 2, [[0.5], [0.5]])
        cpd_b = TabularCPD('B', 2, [[0.5, 0.5], [0.5, 0.5]], evidence=['A'], evidence_card=[2])
        self.model.add_cpds(cpd_a, cpd_b)
        self.inference = VariableElimination(self.model)
    
    def predict(self, match_data: Dict) -> Dict[str, float]:
        """Predict probabilities for a match"""
        if not self.inference:
            return self._get_fallback_probabilities()
        
        try:
            # Prepare evidence
            evidence = {}
            for feature in ['home_attack_strength_cat', 'away_attack_strength_cat',
                          'home_defense_strength_cat', 'away_defense_strength_cat',
                          'home_form_cat', 'away_form_cat', 'h2h_goal_diff_cat']:
                if feature in match_data:
                    evidence[feature] = int(match_data[feature])
            
            # If no evidence, return fallback
            if not evidence:
                return self._get_fallback_probabilities()
            
            # Query probabilities (with error handling)
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
                'home_win': float(result_probs.values[2]) if hasattr(result_probs, 'values') else 0.33,
                'draw': float(result_probs.values[1]) if hasattr(result_probs, 'values') else 0.33,
                'away_win': float(result_probs.values[0]) if hasattr(result_probs, 'values') else 0.34,
                'btts_yes': float(btts_probs.values[1]) if hasattr(btts_probs, 'values') else 0.5,
                'btts_no': float(btts_probs.values[0]) if hasattr(btts_probs, 'values') else 0.5,
                'over_25': float(over25_probs.values[1]) if hasattr(over25_probs, 'values') else 0.5,
                'under_25': float(over25_probs.values[0]) if hasattr(over25_probs, 'values') else 0.5
            }
            
        except Exception as e:
            logger.error(f"Bayesian prediction error: {e}")
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
            if os.path.exists(self.model_path):
                self.model = load(self.model_path)
                self.inference = VariableElimination(self.model)
                logger.info("Bayesian Network loaded successfully")
            else:
                logger.warning("Bayesian model not found, using fallback")
                self._create_fallback_model()
        except Exception as e:
            logger.error(f"Error loading Bayesian model: {e}")
            self._create_fallback_model()
