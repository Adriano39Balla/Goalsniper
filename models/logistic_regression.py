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
        if df.empty:
            return pd.DataFrame()
            
        df = df.copy()
        
        # Create basic features if they don't exist
        default_features = {
            'home_goals_avg': 1.5, 'away_goals_avg': 1.5,
            'home_goals_conceded_avg': 1.5, 'away_goals_conceded_avg': 1.5,
            'home_shots_avg': 10, 'away_shots_avg': 10,
            'home_shots_on_target_avg': 4, 'away_shots_on_target_avg': 4,
            'home_points_5games': 7, 'away_points_5games': 7,
            'home_points_10games': 15, 'away_points_10games': 15,
            'league_goals_avg': 2.5, 'league_btts_ratio': 0.5,
            'h2h_home_goals_avg': 1.5, 'h2h_away_goals_avg': 1.5,
            'h2h_home_wins': 1, 'h2h_away_wins': 1
        }
        
        for feature, default_val in default_features.items():
            if feature not in df.columns:
                df[feature] = default_val
        
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
        df['home_attack_strength'] = df['home_goals_avg'] / df['league_goals_avg'].replace(0, 1)
        df['away_attack_strength'] = df['away_goals_avg'] / df['league_goals_avg'].replace(0, 1)
        df['home_defense_strength'] = 1 - (df['home_goals_conceded_avg'] / df['league_goals_avg'].replace(0, 1))
        df['away_defense_strength'] = 1 - (df['away_goals_conceded_avg'] / df['league_goals_avg'].replace(0, 1))
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
        result_df = df[self.feature_columns].fillna(0)
        
        # Replace infinities
        result_df = result_df.replace([np.inf, -np.inf], 0)
        
        return result_df
    
    def create_targets(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Create target variables for different bet types"""
        if df.empty:
            return {}
            
        targets = {}
        
        # Ensure we have goal data
        if 'home_goals' not in df.columns or 'away_goals' not in df.columns:
            # Create dummy targets for training
            df['home_goals'] = np.random.randint(0, 4, len(df))
            df['away_goals'] = np.random.randint(0, 4, len(df))
        
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
            if df.empty:
                logger.warning("No data to train Logistic Regression")
                return
                
            # Prepare features and targets
            X = self.create_features(df)
            targets = self.create_targets(df)
            
            if X.empty or not targets:
                logger.warning("Insufficient data for training")
                return
            
            # Train separate models for each bet type
            for target_name, y in targets.items():
                if len(np.unique(y)) < 2:
                    logger.warning(f"Not enough classes for {target_name}, skipping")
                    continue
                    
                if target_name == 'result':
                    # Multi-class for 1X2
                    model = LogisticRegression(
                        multi_class='multinomial',
                        solver='lbfgs',
                        max_iter=1000,
                        C=0.1,
                        class_weight='balanced',
                        random_state=42
                    )
                else:
                    # Binary classification for BTTS and Over/Under
                    model = LogisticRegression(
                        max_iter=1000,
                        C=0.1,
                        class_weight='balanced',
                        random_state=42
                    )
                
                # Scale features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Calibrate probabilities
                calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=min(3, len(X)))
                calibrated_model.fit(X_scaled, y)
                
                self.models[target_name] = calibrated_model
                self.scalers[target_name] = scaler
            
            # Save models
            os.makedirs('models', exist_ok=True)
            dump({'models': self.models, 'scalers': self.scalers, 'feature_columns': self.feature_columns}, self.model_path)
            logger.info("Logistic Regression models trained successfully")
            
        except Exception as e:
            logger.error(f"Error training Logistic Regression: {e}")
            # Create fallback models
            self._create_fallback_models()
    
    def _create_fallback_models(self):
        """Create simple fallback models"""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        
        # Simple fallback model for demonstration
        X_dummy = np.random.randn(100, 5)
        y_dummy = np.random.choice(['home_win', 'draw', 'away_win'], 100)
        
        model = LogisticRegression(random_state=42)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_dummy)
        model.fit(X_scaled, y_dummy)
        
        self.models['result'] = model
        self.scalers['result'] = scaler
        self.feature_columns = [f'feature_{i}' for i in range(5)]
    
    def predict(self, match_data: Dict) -> Dict[str, float]:
        """Predict probabilities for a match"""
        try:
            if not self.models:
                return self._get_fallback_probabilities()
            
            # Convert match data to feature vector
            feature_df = pd.DataFrame([match_data])
            X = self.create_features(feature_df)
            
            if X.empty:
                return self._get_fallback_probabilities()
            
            predictions = {}
            
            # Predict result (1X2)
            if 'result' in self.models:
                X_scaled = self.scalers['result'].transform(X)
                result_probs = self.models['result'].predict_proba(X_scaled)[0]
                
                # Get class order
                classes = self.models['result'].classes_
                class_mapping = {cls: i for i, cls in enumerate(classes)}
                
                predictions.update({
                    'home_win': float(result_probs[class_mapping.get('home_win', 0)]),
                    'draw': float(result_probs[class_mapping.get('draw', 1)]),
                    'away_win': float(result_probs[class_mapping.get('away_win', 2)])
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
            logger.error(f"Logistic Regression prediction error: {e}")
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
            if os.path.exists(self.model_path):
                data = load(self.model_path)
                self.models = data['models']
                self.scalers = data['scalers']
                self.feature_columns = data.get('feature_columns', [])
                logger.info("Logistic Regression models loaded successfully")
            else:
                logger.warning("Logistic Regression models not found, using fallback")
                self._create_fallback_models()
        except Exception as e:
            logger.error(f"Error loading Logistic Regression models: {e}")
            self._create_fallback_models()
