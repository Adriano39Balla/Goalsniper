import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
from typing import Dict, List, Tuple
import logging
from joblib import dump, load
import os

logger = logging.getLogger(__name__)

class LogisticRegressionPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.model_path = "models/logistic_models.joblib"
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create sophisticated features for logistic regression"""
        df = df.copy()
        
        # Advanced feature engineering
        features = []
        
        # Attack/defense metrics
        features.extend([
            'home_goals_avg', 'away_goals_avg',
            'home_goals_conceded_avg', 'away_goals_conceded_avg',
            'home_shots_avg', 'away_shots_avg',
            'home_shots_on_target_avg', 'away_shots_on_target_avg'
        ])
        
        # Form metrics (weighted recent performance)
        for window in [5, 10]:
            df[f'home_form_{window}'] = df[f'home_points_{window}games'] / (window * 3)
            df[f'away_form_{window}'] = df[f'away_points_{window}games'] / (window * 3)
            features.extend([f'home_form_{window}', f'away_form_{window}'])
        
        # Strength metrics
        df['home_attack_strength'] = df['home_goals_avg'] / df['league_goals_avg']
        df['away_attack_strength'] = df['away_goals_avg'] / df['league_goals_avg']
        df['home_defense_strength'] = 1 - (df['home_goals_conceded_avg'] / df['league_goals_avg'])
        df['away_defense_strength'] = 1 - (df['away_goals_conceded_avg'] / df['league_goals_avg'])
        features.extend([
            'home_attack_strength', 'away_attack_strength',
            'home_defense_strength', 'away_defense_strength'
        ])
        
        # Head-to-head features
        features.extend([
            'h2h_home_goals_avg', 'h2h_away_goals_avg',
            'h2h_home_wins', 'h2h_away_wins'
        ])
        
        # League context
        features.extend([
            'league_goals_avg', 'league_btts_ratio'
        ])
        
        self.feature_columns = [f for f in features if f in df.columns]
        return df[self.feature_columns].fillna(0)
    
    def create_targets(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Create target variables for different bet types"""
        targets = {}
        
        # 1X2 outcomes
        targets['result'] = np.select(
            [df['home_goals'] > df['away_goals'], 
             df['home_goals'] == df['away_goals']],
            ['home_win', 'draw'],
            default='away_win'
        )
        
        # BTTS
        targets['btts'] = ((df['home_goals'] > 0) & (df['away_goals'] > 0)).astype(int)
        
        # Over/Under 2.5
        targets['over_25'] = (df['home_goals'] + df['away_goals'] > 2.5).astype(int)
        
        return targets
    
    def train(self, df: pd.DataFrame):
        """Train logistic regression models for all bet types"""
        try:
            # Prepare features and targets
            X = self.create_features(df)
            targets = self.create_targets(df)
            
            # Train separate models for each bet type
            for target_name, y in targets.items():
                if target_name == 'result':
                    # Multi-class for 1X2
                    model = LogisticRegression(
                        multi_class='multinomial',
                        solver='lbfgs',
                        max_iter=1000,
                        C=0.1,
                        class_weight='balanced'
                    )
                else:
                    # Binary classification for BTTS and Over/Under
                    model = LogisticRegression(
                        max_iter=1000,
                        C=0.1,
                        class_weight='balanced'
                    )
                
                # Scale features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Calibrate probabilities
                calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=5)
                calibrated_model.fit(X_scaled, y)
                
                self.models[target_name] = calibrated_model
                self.scalers[target_name] = scaler
            
            # Save models
            os.makedirs('models', exist_ok=True)
            dump({'models': self.models, 'scalers': self.scalers}, self.model_path)
            logger.info("Logistic Regression models trained successfully")
            
        except Exception as e:
            logger.error(f"Error training Logistic Regression: {e}")
            raise
    
    def predict(self, match_data: Dict) -> Dict[str, float]:
        """Predict probabilities for a match"""
        try:
            # Convert match data to feature vector
            feature_df = pd.DataFrame([match_data])
            X = self.create_features(feature_df)
            
            predictions = {}
            
            # Predict result (1X2)
            if 'result' in self.models:
                X_scaled = self.scalers['result'].transform(X)
                result_probs = self.models['result'].predict_proba(X_scaled)[0]
                
                # Assuming order: ['away_win', 'draw', 'home_win']
                predictions.update({
                    'home_win': float(result_probs[2]),
                    'draw': float(result_probs[1]),
                    'away_win': float(result_probs[0])
                })
            
            # Predict BTTS
            if 'btts' in self.models:
                X_scaled = self.scalers['btts'].transform(X)
                btts_probs = self.models['btts'].predict_proba(X_scaled)[0]
                predictions.update({
                    'btts_yes': float(btts_probs[1]),
                    'btts_no': float(btts_probs[0])
                })
            
            # Predict Over/Under 2.5
            if 'over_25' in self.models:
                X_scaled = self.scalers['over_25'].transform(X)
                over25_probs = self.models['over_25'].predict_proba(X_scaled)[0]
                predictions.update({
                    'over_25': float(over25_probs[1]),
                    'under_25': float(over25_probs[0])
                })
            
            return predictions
            
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
    
    def load_models(self):
        """Load trained models"""
        try:
            data = load(self.model_path)
            self.models = data['models']
            self.scalers = data['scalers']
            logger.info("Logistic Regression models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
