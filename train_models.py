import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from supabase import create_client
import requests
import asyncio
import aiohttp
from datetime import datetime, timedelta
import os

# Configuration
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
API_FOOTBALL_KEY = os.getenv('API_FOOTBALL_KEY')

class BayesianNetwork:
    def __init__(self):
        self.prior_probabilities = {}
        self.conditional_probabilities = {}
        
    def update_priors(self, historical_data):
        """Update prior probabilities based on historical data"""
        # Calculate base rates for different leagues and conditions
        for league in historical_data['league_id'].unique():
            league_data = historical_data[historical_data['league_id'] == league]
            
            # Prior for Over/Under
            total_goals = league_data['home_team_goals'] + league_data['away_team_goals']
            self.prior_probabilities[f'{league}_over_2.5'] = (total_goals > 2.5).mean()
            self.prior_probabilities[f'{league}_over_1.5'] = (total_goals > 1.5).mean()
            
            # Prior for BTTS
            self.prior_probabilities[f'{league}_btts'] = (
                (league_data['home_team_goals'] > 0) & 
                (league_data['away_team_goals'] > 0)
            ).mean()
            
            # Prior for Home Win
            self.prior_probabilities[f'{league}_home_win'] = (
                league_data['home_team_goals'] > league_data['away_team_goals']
            ).mean()

class AdvancedBettingPredictor:
    def __init__(self):
        self.bayesian_net = BayesianNetwork()
        self.models = {}
        self.feature_columns = []
        self.calibration_data = []
        
    def create_features(self, match_data):
        """Create sophisticated features for ML models"""
        features = {}
        
        # Team strength features
        features['home_attack_strength'] = match_data.get('home_team_goals_avg', 1.5)
        features['away_attack_strength'] = match_data.get('away_team_goals_avg', 1.5)
        features['home_defense_strength'] = match_data.get('home_team_conceded_avg', 1.2)
        features['away_defense_strength'] = match_data.get('away_team_conceded_avg', 1.2)
        
        # Form features (last 5 games)
        features['home_form'] = match_data.get('home_team_form', 0.5)
        features['away_form'] = match_data.get('away_team_form', 0.5)
        
        # Momentum features
        features['home_momentum'] = match_data.get('home_team_momentum', 0)
        features['away_momentum'] = match_data.get('away_team_momentum', 0)
        
        # Contextual features
        features['league_avg_goals'] = match_data.get('league_avg_goals', 2.5)
        features['importance'] = match_data.get('match_importance', 0.5)
        features['rivalry'] = match_data.get('is_rivalry', 0)
        
        # Live match features
        if match_data.get('is_live', False):
            features['minutes_played'] = match_data.get('minutes_played', 0)
            features['current_score_home'] = match_data.get('current_home_goals', 0)
            features['current_score_away'] = match_data.get('current_away_goals', 0)
            features['momentum_indicator'] = match_data.get('momentum_indicator', 0.5)
            features['recent_attacks'] = match_data.get('recent_attacks', 0)
        
        return features
    
    def train_models(self, training_data):
        """Train multiple models for different bet types"""
        print("Training advanced betting models...")
        
        # Prepare features and targets
        X = []
        y_over_25 = []
        y_btts = []
        y_home_win = []
        
        for match in training_data:
            features = self.create_features(match)
            X.append(list(features.values()))
            
            # Targets
            total_goals = match['home_team_goals'] + match['away_team_goals']
            y_over_25.append(1 if total_goals > 2.5 else 0)
            y_btts.append(1 if match['home_team_goals'] > 0 and match['away_team_goals'] > 0 else 0)
            y_home_win.append(1 if match['home_team_goals'] > match['away_team_goals'] else 0)
        
        X = np.array(X)
        self.feature_columns = list(features.keys())
        
        # Train models with calibration
        bet_types = {
            'over_25': y_over_25,
            'btts': y_btts, 
            'home_win': y_home_win
        }
        
        for bet_type, y in bet_types.items():
            # Base model
            base_model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            
            # Calibrated model for better probability estimates
            calibrated_model = CalibratedClassifierCV(
                base_model, 
                method='isotonic', 
                cv=3
            )
            
            calibrated_model.fit(X, y)
            self.models[bet_type] = calibrated_model
            
            print(f"Trained and calibrated model for {bet_type}")
    
    def predict_with_confidence(self, match_data):
        """Generate predictions with calibrated probabilities and confidence scores"""
        features = self.create_features(match_data)
        X = np.array([list(features.values())])
        
        predictions = {}
        
        for bet_type, model in self.models.items():
            # Get calibrated probabilities
            proba = model.predict_proba(X)[0]
            
            # Bayesian adjustment using prior knowledge
            league = match_data.get('league_id', 'default')
            prior_key = f"{league}_{bet_type}"
            prior_prob = self.bayesian_net.prior_probabilities.get(prior_key, 0.5)
            
            # Blend prior with model prediction (Bayesian update)
            alpha = 0.1  # Strength of prior
            adjusted_prob = (1 - alpha) * proba[1] + alpha * prior_prob
            
            # Calculate confidence based on feature strength and model certainty
            feature_strength = np.std(list(features.values()))
            model_confidence = np.max(proba)
            confidence_score = 0.7 * model_confidence + 0.3 * feature_strength
            
            predictions[bet_type] = {
                'probability': float(adjusted_prob),
                'confidence': float(confidence_score),
                'prediction': adjusted_prob > 0.5,
                'edge': float(adjusted_prob - 0.5)  # Positive edge over bookmaker implied probability
            }
        
        return predictions
    
    def learn_from_mistakes(self, bet_outcomes):
        """Self-learning from incorrect predictions"""
        for outcome in bet_outcomes:
            match_data = outcome['match_data']
            actual_result = outcome['actual_result']
            predicted_result = outcome['predicted_result']
            
            if predicted_result != actual_result:
                # Add to calibration data for model retraining
                self.calibration_data.append({
                    'match_data': match_data,
                    'correct_prediction': actual_result
                })
                
                # Update Bayesian priors
                self._update_bayesian_network(match_data, actual_result)
    
    def _update_bayesian_network(self, match_data, actual_result):
        """Update Bayesian network based on new evidence"""
        league = match_data.get('league_id', 'default')
        
        # Update priors based on actual outcomes
        for bet_type in ['over_25', 'btts', 'home_win']:
            prior_key = f"{league}_{bet_type}"
            current_prior = self.bayesian_net.prior_probabilities.get(prior_key, 0.5)
            
            # Simple exponential moving average update
            learning_rate = 0.01
            new_prior = (1 - learning_rate) * current_prior + learning_rate * actual_result[bet_type]
            
            self.bayesian_net.prior_probabilities[prior_key] = new_prior

class DataFetcher:
    def __init__(self):
        self.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.api_headers = {
            'x-rapidapi-key': API_FOOTBALL_KEY,
            'x-rapidapi-host': 'api-football-v1.p.rapidapi.com'
        }
    
    async def fetch_historical_data(self, days=365):
        """Fetch historical match data for training"""
        print(f"Fetching historical data for last {days} days...")
        
        # In production, implement actual API calls
        # This is a simplified version
        historical_data = []
        
        # Fetch from Supabase first
        response = self.supabase.table('historical_matches')\
            .select('*')\
            .gte('date', f'{(datetime.now() - timedelta(days=days)).date()}')\
            .execute()
        
        return response.data
    
    async def fetch_live_matches(self):
        """Fetch current live matches for prediction"""
        print("Fetching live matches...")
        
        async with aiohttp.ClientSession() as session:
            url = "https://api-football-v1.p.rapidapi.com/v3/fixtures"
            params = {'live': 'all'}
            
            async with session.get(url, headers=self.api_headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('response', [])
                else:
                    print(f"API Error: {response.status}")
                    return []

async def main_training():
    """Main training function"""
    print("Starting model training pipeline...")
    
    # Initialize components
    predictor = AdvancedBettingPredictor()
    data_fetcher = DataFetcher()
    
    try:
        # Fetch training data
        historical_data = await data_fetcher.fetch_historical_data(180)  # Last 6 months
        
        if historical_data:
            # Train models
            predictor.train_models(historical_data)
            
            # Save trained models
            joblib.dump(predictor, 'models/trained_predictor.joblib')
            joblib.dump(predictor.feature_columns, 'models/feature_columns.joblib')
            
            print("Model training completed successfully!")
        else:
            print("No historical data found for training")
            
    except Exception as e:
        print(f"Training error: {e}")

if __name__ == "__main__":
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Run training
    asyncio.run(main_training())
