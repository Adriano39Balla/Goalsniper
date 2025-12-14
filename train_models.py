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
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from scipy.stats import poisson
import joblib
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import DatabaseManager

class ModelTrainer:
    def __init__(self):
        self.db = DatabaseManager()
        self.models = {}
        self.features = []
        self.scaler = StandardScaler()
        self.model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def setup_logger(self):
        """Configure logging"""
        logger.remove()  # Remove existing handlers
        
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level="INFO"
        )
        
        logger.add(
            "logs/training_{time:YYYY-MM-DD}.log",
            rotation="00:00",
            retention="30 days",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG"
        )
        
        logger.info("Training logger initialized")
        
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
            WHERE m.status = 'FT'  -- Only finished matches
                AND m.goals_home IS NOT NULL
                AND m.goals_away IS NOT NULL
                AND m.goals_home >= 0
                AND m.goals_away >= 0
            """
            
            params = []
            if leagues:
                query += " AND m.league_id IN %s"
                params.append(tuple(leagues))
            if seasons:
                query += " AND m.season IN %s"
                params.append(tuple(seasons))
                
            query += " ORDER BY m.timestamp DESC LIMIT 10000"
                
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
        
        # Ensure required columns exist with default values
        required_columns = [
            'home_avg_goals', 'home_avg_conceded', 'home_form',
            'home_att_strength', 'home_def_strength',
            'away_avg_goals', 'away_avg_conceded', 'away_form',
            'away_att_strength', 'away_def_strength',
            'total_matches', 'home_wins', 'away_wins', 'draws'
        ]
        
        for col in required_columns:
            if col not in df.columns:
                df[col] = 0
        
        # Goal difference features
        features['home_goal_diff'] = df['home_avg_goals'] - df['home_avg_conceded']
        features['away_goal_diff'] = df['away_avg_goals'] - df['away_avg_conceded']
        
        # Form features (convert string form to numeric)
        def form_to_numeric(form_str):
            if not isinstance(form_str, str):
                return 0
            points = {'W': 1, 'D': 0, 'L': -1}
            recent_form = form_str[-5:] if len(form_str) >= 5 else form_str
            return sum(points.get(char, 0) for char in recent_form) / len(recent_form)
        
        features['home_form_numeric'] = df['home_form'].apply(form_to_numeric)
        features['away_form_numeric'] = df['away_form'].apply(form_to_numeric)
        
        # Strength features
        features['att_strength_diff'] = df['home_att_strength'] - df['away_att_strength']
        features['def_strength_diff'] = df['home_def_strength'] - df['away_def_strength']
        
        # Head to head features (with smoothing for low sample sizes)
        total_matches = df['total_matches'].replace(0, 1)
        features['h2h_home_win_pct'] = (df['home_wins'] + 1) / (total_matches + 3)
        features['h2h_away_win_pct'] = (df['away_wins'] + 1) / (total_matches + 3)
        features['h2h_draw_pct'] = (df['draws'] + 1) / (total_matches + 3)
        
        # League features
        features['league_id'] = df['league_id']
        
        # Additional statistical features
        features['total_avg_goals'] = (df['home_avg_goals'] + df['away_avg_goals']) / 2
        features['total_avg_conceded'] = (df['home_avg_conceded'] + df['away_avg_conceded']) / 2
        
        # Create target variables
        targets = {}
        
        # Match result (1=Home Win, 0=Draw, -1=Away Win)
        goals_home = df['goals_home'].astype(int)
        goals_away = df['goals_away'].astype(int)
        targets['result'] = np.where(goals_home > goals_away, 1,
                                    np.where(goals_home == goals_away, 0, -1))
        
        # Over 2.5 goals
        targets['over_25'] = ((goals_home + goals_away) > 2.5).astype(int)
        
        # Both teams to score
        targets['btts'] = ((goals_home > 0) & (goals_away > 0)).astype(int)
        
        # Exact goals (for Poisson)
        targets['total_goals'] = goals_home + goals_away
        targets['home_goals'] = goals_home
        targets['away_goals'] = goals_away
        
        self.features = features.columns.tolist()
        logger.success(f"Created {len(self.features)} features")
        
        # Scale features
        if len(features) > 0:
            features_scaled = self.scaler.fit_transform(features)
            features = pd.DataFrame(features_scaled, columns=features.columns)
        
        return features, targets
    
    def train_poisson_model(self, X, y_home, y_away):
        """
        Train Poisson model for goal prediction
        """
        logger.info("Training Poisson model...")
        
        # Calculate lambda parameters
        lambda_home = max(0.1, np.mean(y_home))
        lambda_away = max(0.1, np.mean(y_away))
        
        # Fit Poisson distribution parameters
        model = {
            'lambda_home': float(lambda_home),
            'lambda_away': float(lambda_away),
            'home_goals_mean': float(np.mean(y_home)),
            'away_goals_mean': float(np.mean(y_away)),
            'home_goals_std': float(np.std(y_home)),
            'away_goals_std': float(np.std(y_away)),
            'type': 'poisson',
            'version': self.model_version
        }
        
        logger.success(f"Poisson model trained: λ_home={lambda_home:.2f}, λ_away={lambda_away:.2f}")
        return model
    
    def train_xgboost_model(self, X, y, model_name):
        """
        Train XGBoost model with hyperparameter tuning
        """
        logger.info(f"Training XGBoost model for {model_name}...")
        
        if len(X) < 100:
            logger.warning(f"Insufficient data for {model_name} training: {len(X)} samples")
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )
        
        # Hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        # Create model
        xgb_model = xgb.XGBClassifier(
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            eval_metric='logloss'
        )
        
        # Randomized search
        random_search = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=param_grid,
            n_iter=10,
            cv=3,
            verbose=0,
            random_state=42,
            n_jobs=-1,
            scoring='accuracy'
        )
        
        random_search.fit(X_train, y_train)
        
        # Best model
        best_model = random_search.best_estimator_
        best_params = random_search.best_params_
        
        # Evaluate
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Cross-validation score
        cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')
        
        logger.success(
            f"{model_name} - Accuracy: {accuracy:.3f}, "
            f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, "
            f"CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}"
        )
        
        # Save model metrics
        self.save_model_metrics(model_name, best_model, accuracy, precision, recall, f1)
        
        return best_model
    
    def train_random_forest(self, X, y, model_name):
        """
        Train Random Forest model
        """
        logger.info(f"Training Random Forest model for {model_name}...")
        
        if len(X) < 50:
            logger.warning(f"Insufficient data for {model_name} training: {len(X)} samples")
            return None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        logger.success(
            f"{model_name} - Accuracy: {accuracy:.3f}, "
            f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}"
        )
        
        # Save model metrics
        self.save_model_metrics(model_name, model, accuracy, precision, recall, f1)
        
        return model
    
    def save_model_metrics(self, model_name, model, accuracy, precision, recall, f1):
        """Save model performance metrics to database"""
        try:
            query = """
            INSERT INTO model_versions 
            (version, model_type, accuracy, precision, recall, f1_score, features_count, training_date, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
            """
            
            features_count = len(self.features) if hasattr(self, 'features') else 0
            
            params = (
                self.model_version,
                model_name,
                round(accuracy, 4),
                round(precision, 4),
                round(recall, 4),
                round(f1, 4),
                features_count,
                datetime.now().date()
            )
            
            self.db.execute_query(query, params, fetch=False)
            
        except Exception as e:
            logger.error(f"Error saving model metrics: {e}")
    
    def calculate_value_bets(self, predictions, odds, threshold=0.05):
        """
        Calculate expected value for each market
        """
        value_bets = []
        
        # Calculate implied probabilities from odds
        implied_probs = {}
        for market_key, odds_key in [
            ('home_win', 'home_odds'),
            ('draw', 'draw_odds'),
            ('away_win', 'away_odds'),
            ('over_25', 'over_25_odds'),
            ('under_25', 'under_25_odds'),
            ('btts_yes', 'btts_yes_odds'),
            ('btts_no', 'btts_no_odds')
        ]:
            if odds_key in odds and odds[odds_key] and odds[odds_key] > 0:
                implied_probs[market_key] = 1 / odds[odds_key]
            else:
                implied_probs[market_key] = 0
        
        # Calculate expected value for each market
        for market, pred_prob in predictions.items():
            if market in implied_probs and implied_probs[market] > 0 and pred_prob > 0:
                odds_value = odds.get(f"{market}_odds", 0)
                if odds_value and odds_value > 0:
                    ev = (pred_prob * odds_value) - 1
                    
                    if ev > threshold:  # Only consider bets with positive EV > threshold
                        value_bet = {
                            'market': market,
                            'predicted_probability': pred_prob,
                            'implied_probability': implied_probs[market],
                            'odds': odds_value,
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
            
            if X.empty:
                logger.error("No features created from training data")
                return False
            
            # Train models for each market
            self.models = {}
            trained_count = 0
            
            # 1. Match Result Model (1X2)
            result_model = self.train_xgboost_model(
                X, targets['result'], 'Match Result'
            )
            if result_model:
                self.models['result'] = result_model
                trained_count += 1
            
            # 2. Over/Under 2.5 Goals Model
            ou_model = self.train_xgboost_model(
                X, targets['over_25'], 'Over/Under 2.5'
            )
            if ou_model:
                self.models['over_under'] = ou_model
                trained_count += 1
            
            # 3. BTTS Model
            btts_model = self.train_xgboost_model(
                X, targets['btts'], 'Both Teams to Score'
            )
            if btts_model:
                self.models['btts'] = btts_model
                trained_count += 1
            
            # 4. Poisson Model for goal prediction
            poisson_model = self.train_poisson_model(
                X, targets['home_goals'], targets['away_goals']
            )
            if poisson_model:
                self.models['poisson'] = poisson_model
                trained_count += 1
            
            # 5. Random Forest ensemble
            rf_model = self.train_random_forest(
                X, targets['result'], 'Random Forest'
            )
            if rf_model:
                self.models['ensemble'] = rf_model
                trained_count += 1
            
            # Save models
            if trained_count > 0:
                self.save_models()
            else:
                logger.error("No models were successfully trained")
                return False
            
            # Log feature importance for result model
            if 'result' in self.models and hasattr(self.models['result'], 'feature_importances_'):
                try:
                    importance = pd.DataFrame({
                        'feature': self.features,
                        'importance': self.models['result'].feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    logger.info("Top 10 important features for result prediction:")
                    for idx, row in importance.head(10).iterrows():
                        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
                except Exception as e:
                    logger.warning(f"Could not calculate feature importance: {e}")
            
            logger.success(f"All models trained successfully! ({trained_count} models)")
            return True
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def save_models(self):
        """Save trained models to disk with versioning"""
        try:
            # Create versioned directory
            model_dir = f"models/{self.model_version}"
            os.makedirs(model_dir, exist_ok=True)
            
            # Create symlink to latest
            latest_dir = "models/latest"
            if os.path.exists(latest_dir):
                if os.path.islink(latest_dir):
                    os.unlink(latest_dir)
                elif os.path.isdir(latest_dir):
                    import shutil
                    shutil.rmtree(latest_dir)
            
            # Save each model
            for model_name, model in self.models.items():
                filename = f"{model_dir}/{model_name}_model.pkl"
                
                if model_name == 'poisson':
                    # Poisson model is a dict, use pickle
                    with open(filename, 'wb') as f:
                        pickle.dump(model, f)
                else:
                    # Scikit-learn/XGBoost models
                    joblib.dump(model, filename)
                
                logger.info(f"Saved {model_name} model to {filename}")
            
            # Save features and scaler
            with open(f'{model_dir}/features.pkl', 'wb') as f:
                pickle.dump(self.features, f)
            
            joblib.dump(self.scaler, f'{model_dir}/scaler.pkl')
            
            # Save metadata
            metadata = {
                'version': self.model_version,
                'created_at': datetime.now().isoformat(),
                'model_count': len(self.models),
                'features': self.features,
                'feature_count': len(self.features)
            }
            
            with open(f'{model_dir}/metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Create symlink
            os.symlink(self.model_version, latest_dir, target_is_directory=True)
            
            logger.success(f"Models saved to {model_dir}")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def load_models(self):
        """Load trained models from disk"""
        try:
            latest_dir = "models/latest"
            if not os.path.exists(latest_dir):
                logger.warning("No 'latest' model directory found")
                return False
            
            if not os.path.islink(latest_dir) and not os.path.isdir(latest_dir):
                logger.error("'latest' is not a valid directory or symlink")
                return False
            
            # Resolve the actual directory
            actual_dir = os.path.realpath(latest_dir)
            
            # Load models
            model_files = {
                'result': f'{actual_dir}/result_model.pkl',
                'over_under': f'{actual_dir}/over_under_model.pkl',
                'btts': f'{actual_dir}/btts_model.pkl',
                'poisson': f'{actual_dir}/poisson_model.pkl',
                'ensemble': f'{actual_dir}/ensemble_model.pkl'
            }
            
            loaded_count = 0
            for model_name, filename in model_files.items():
                if os.path.exists(filename):
                    try:
                        if model_name == 'poisson':
                            with open(filename, 'rb') as f:
                                self.models[model_name] = pickle.load(f)
                        else:
                            self.models[model_name] = joblib.load(filename)
                        
                        loaded_count += 1
                        logger.info(f"Loaded {model_name} model")
                    except Exception as e:
                        logger.error(f"Error loading {model_name} model: {e}")
                else:
                    logger.warning(f"Model file not found: {filename}")
            
            # Load features
            features_file = f'{actual_dir}/features.pkl'
            if os.path.exists(features_file):
                with open(features_file, 'rb') as f:
                    self.features = pickle.load(f)
            
            # Load scaler
            scaler_file = f'{actual_dir}/scaler.pkl'
            if os.path.exists(scaler_file):
                self.scaler = joblib.load(scaler_file)
            
            logger.success(f"Loaded {loaded_count} models from {actual_dir}")
            return loaded_count > 0
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description='Train football prediction models')
    parser.add_argument('--initial', action='store_true', help='Initial training with backfill')
    parser.add_argument('--leagues', nargs='+', type=int, help='League IDs to train on')
    parser.add_argument('--seasons', nargs='+', type=int, help='Seasons to train on')
    parser.add_argument('--limit', type=int, default=10000, help='Limit training data size')
    
    args = parser.parse_args()
    
    # Setup trainer
    trainer = ModelTrainer()
    trainer.setup_logger()
    
    logger.info(f"Starting training with parameters: initial={args.initial}, "
                f"leagues={args.leagues}, seasons={args.seasons}")
    
    if args.initial:
        logger.info("Starting initial training...")
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
