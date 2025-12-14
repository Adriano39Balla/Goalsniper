#!/usr/bin/env python3
"""
Football Prediction Model Training Module
"""
import os
import sys
import pickle
import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from loguru import logger
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
from scipy.stats import poisson
import joblib

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import DatabaseManager

class ModelTrainer:
    def __init__(self):
        self.db = DatabaseManager()
        self.models = {}
        self.features = []
        
    def setup_logger(self):
        """Configure logging"""
        logger.add("logs/training_{time}.log", rotation="500 MB", retention="10 days")
        
    def load_training_data(self, leagues=None, seasons=None):
        """
        Load historical match data for training
        """
        try:
            logger.info("Loading training data from database...")
            
            query = """
            SELECT 
                m.*,
                ht.home_avg_goals,
                ht.home_avg_conceded,
                ht.home_form,
                ht.home_att_strength,
                ht.home_def_strength,
                at.away_avg_goals,
                at.away_avg_conceded,
                at.away_form,
                at.away_att_strength,
                at.away_def_strength,
                h2h.total_matches,
                h2h.home_wins,
                h2h.away_wins,
                h2h.draws,
                o.home_odds,
                o.draw_odds,
                o.away_odds,
                o.over_25_odds,
                o.under_25_odds,
                o.btts_yes_odds,
                o.btts_no_odds
            FROM matches m
            LEFT JOIN team_stats ht ON m.home_team_id = ht.team_id 
                AND m.league_id = ht.league_id 
                AND m.season = ht.season
            LEFT JOIN team_stats at ON m.away_team_id = at.team_id 
                AND m.league_id = at.league_id 
                AND m.season = at.season
            LEFT JOIN head_to_head h2h ON m.home_team_id = h2h.home_team_id 
                AND m.away_team_id = h2h.away_team_id
            LEFT JOIN odds o ON m.fixture_id = o.fixture_id
            WHERE m.status = 'Match Finished'
                AND m.goals_home IS NOT NULL
                AND m.goals_away IS NOT NULL
            """
            
            params = []
            if leagues:
                query += " AND m.league_id IN %s"
                params.append(tuple(leagues))
            if seasons:
                query += " AND m.season IN %s"
                params.append(tuple(seasons))
                
            df = self.db.execute_query(query, params if params else None)
            
            if df.empty:
                logger.warning("No training data found!")
                return None
                
            logger.success(f"Loaded {len(df)} matches for training")
            return df
            
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            return None
    
    def create_features(self, df):
        """
        Create engineered features for model training
        """
        logger.info("Creating features...")
        
        # Basic features
        features = pd.DataFrame()
        
        # Goal difference features
        features['home_goal_diff'] = df['home_avg_goals'] - df['home_avg_conceded']
        features['away_goal_diff'] = df['away_avg_goals'] - df['away_avg_conceded']
        
        # Form features
        features['home_form'] = df['home_form']
        features['away_form'] = df['away_form']
        
        # Strength features
        features['att_strength_diff'] = df['home_att_strength'] - df['away_att_strength']
        features['def_strength_diff'] = df['home_def_strength'] - df['away_def_strength']
        
        # Head to head features
        features['h2h_home_win_pct'] = df['home_wins'] / df['total_matches'].replace(0, 1)
        features['h2h_away_win_pct'] = df['away_wins'] / df['total_matches'].replace(0, 1)
        features['h2h_draw_pct'] = df['draws'] / df['total_matches'].replace(0, 1)
        
        # League position difference (if available)
        if 'home_league_position' in df.columns and 'away_league_position' in df.columns:
            features['league_pos_diff'] = df['away_league_position'] - df['home_league_position']
        
        # Recent performance (last 5 matches average)
        if 'home_goals_scored_last5' in df.columns:
            features['home_goals_scored_last5'] = df['home_goals_scored_last5']
            features['away_goals_scored_last5'] = df['away_goals_scored_last5']
            features['home_goals_conceded_last5'] = df['home_goals_conceded_last5']
            features['away_goals_conceded_last5'] = df['away_goals_conceded_last5']
        
        # Create target variables
        targets = {}
        
        # Match result (1=Home Win, 0=Draw, -1=Away Win)
        targets['result'] = np.where(df['goals_home'] > df['goals_away'], 1,
                                    np.where(df['goals_home'] == df['goals_away'], 0, -1))
        
        # Over 2.5 goals
        targets['over_25'] = ((df['goals_home'] + df['goals_away']) > 2.5).astype(int)
        
        # Both teams to score
        targets['btts'] = ((df['goals_home'] > 0) & (df['goals_away'] > 0)).astype(int)
        
        # Exact goals (for Poisson)
        targets['total_goals'] = df['goals_home'] + df['goals_away']
        targets['home_goals'] = df['goals_home']
        targets['away_goals'] = df['goals_away']
        
        self.features = features.columns.tolist()
        logger.success(f"Created {len(self.features)} features")
        
        return features, targets
    
    def train_poisson_model(self, X, y_home, y_away):
        """
        Train Poisson model for goal prediction
        """
        logger.info("Training Poisson model...")
        
        # Calculate lambda parameters
        lambda_home = np.mean(y_home)
        lambda_away = np.mean(y_away)
        
        model = {
            'lambda_home': lambda_home,
            'lambda_away': lambda_away,
            'type': 'poisson'
        }
        
        return model
    
    def train_xgboost_model(self, X, y, model_name):
        """
        Train XGBoost model with hyperparameter tuning
        """
        logger.info(f"Training XGBoost model for {model_name}...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        # Create model
        xgb_model = xgb.XGBClassifier(
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        
        # Randomized search
        random_search = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=param_grid,
            n_iter=10,
            cv=3,
            verbose=0,
            random_state=42,
            n_jobs=-1
        )
        
        random_search.fit(X_train, y_train)
        
        # Best model
        best_model = random_search.best_estimator_
        
        # Evaluate
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        
        logger.success(f"{model_name} - Accuracy: {accuracy:.3f}, Precision: {precision:.3f}")
        
        return best_model
    
    def train_random_forest(self, X, y, model_name):
        """
        Train Random Forest model
        """
        logger.info(f"Training Random Forest model for {model_name}...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.success(f"{model_name} - Accuracy: {accuracy:.3f}")
        
        return model
    
    def calculate_value_bets(self, predictions, odds, threshold=0.05):
        """
        Calculate expected value for each market
        """
        value_bets = []
        
        # Calculate implied probabilities from odds
        implied_probs = {
            'home_win': 1 / odds['home_odds'] if odds['home_odds'] else 0,
            'draw': 1 / odds['draw_odds'] if odds['draw_odds'] else 0,
            'away_win': 1 / odds['away_odds'] if odds['away_odds'] else 0,
            'over_25': 1 / odds['over_25_odds'] if odds['over_25_odds'] else 0,
            'under_25': 1 / odds['under_25_odds'] if odds['under_25_odds'] else 0,
            'btts_yes': 1 / odds['btts_yes_odds'] if odds['btts_yes_odds'] else 0,
            'btts_no': 1 / odds['btts_no_odds'] if odds['btts_no_odds'] else 0
        }
        
        # Calculate expected value for each market
        for market, pred_prob in predictions.items():
            if market in implied_probs and implied_probs[market] > 0:
                ev = (pred_prob * odds.get(f"{market}_odds", 0)) - 1
                
                if ev > threshold:  # Only consider bets with positive EV > threshold
                    value_bet = {
                        'market': market,
                        'predicted_probability': pred_prob,
                        'implied_probability': implied_probs[market],
                        'odds': odds.get(f"{market}_odds", 0),
                        'expected_value': ev,
                        'edge': pred_prob - implied_probs[market]
                    }
                    value_bets.append(value_bet)
        
        # Sort by expected value
        value_bets.sort(key=lambda x: x['expected_value'], reverse=True)
        
        return value_bets
    
    def train_all_models(self, initial=False):
        """
        Train all prediction models
        """
        try:
            logger.info("Starting model training...")
            
            # Load data
            df = self.load_training_data()
            if df is None or df.empty:
                logger.error("No data available for training")
                return False
            
            # Create features
            X, targets = self.create_features(df)
            
            # Train models for each market
            self.models = {}
            
            # 1. Match Result Model (1X2)
            self.models['result'] = self.train_xgboost_model(
                X, targets['result'], 'Match Result'
            )
            
            # 2. Over/Under 2.5 Goals Model
            self.models['over_under'] = self.train_xgboost_model(
                X, targets['over_25'], 'Over/Under 2.5'
            )
            
            # 3. BTTS Model
            self.models['btts'] = self.train_xgboost_model(
                X, targets['btts'], 'Both Teams to Score'
            )
            
            # 4. Poisson Model for goal prediction
            self.models['poisson'] = self.train_poisson_model(
                X, targets['home_goals'], targets['away_goals']
            )
            
            # 5. Random Forest ensemble
            self.models['ensemble'] = self.train_random_forest(
                X, targets['result'], 'Ensemble'
            )
            
            # Save models
            self.save_models()
            
            # Log feature importance
            if hasattr(self.models['result'], 'feature_importances_'):
                importance = pd.DataFrame({
                    'feature': self.features,
                    'importance': self.models['result'].feature_importances_
                }).sort_values('importance', ascending=False)
                
                logger.info("Top 10 important features:")
                for idx, row in importance.head(10).iterrows():
                    logger.info(f"  {row['feature']}: {row['importance']:.4f}")
            
            logger.success("All models trained successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            return False
    
    def save_models(self):
        """Save trained models to disk"""
        try:
            os.makedirs('models', exist_ok=True)
            
            for model_name, model in self.models.items():
                filename = f"models/{model_name}_model.pkl"
                joblib.dump(model, filename)
                logger.info(f"Saved {model_name} model to {filename}")
            
            # Save features list
            with open('models/features.pkl', 'wb') as f:
                pickle.dump(self.features, f)
                
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def load_models(self):
        """Load trained models from disk"""
        try:
            model_files = {
                'result': 'models/result_model.pkl',
                'over_under': 'models/over_under_model.pkl',
                'btts': 'models/btts_model.pkl',
                'poisson': 'models/poisson_model.pkl',
                'ensemble': 'models/ensemble_model.pkl'
            }
            
            for model_name, filename in model_files.items():
                if os.path.exists(filename):
                    self.models[model_name] = joblib.load(filename)
                    logger.info(f"Loaded {model_name} model")
                else:
                    logger.warning(f"Model file not found: {filename}")
            
            # Load features
            if os.path.exists('models/features.pkl'):
                with open('models/features.pkl', 'rb') as f:
                    self.features = pickle.load(f)
                    
            return len(self.models) > 0
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description='Train football prediction models')
    parser.add_argument('--initial', action='store_true', help='Initial training with backfill')
    parser.add_argument('--leagues', nargs='+', help='League IDs to train on')
    parser.add_argument('--seasons', nargs='+', help='Seasons to train on')
    
    args = parser.parse_args()
    
    # Setup trainer
    trainer = ModelTrainer()
    trainer.setup_logger()
    
    if args.initial:
        logger.info("Starting initial training...")
        # Here you would call backfill data first
        # For now, just train with available data
        success = trainer.train_all_models(initial=True)
    else:
        logger.info("Starting incremental training...")
        success = trainer.train_all_models()
    
    if success:
        logger.success("Training completed successfully!")
        sys.exit(0)
    else:
        logger.error("Training failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
