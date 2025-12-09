"""
Model training module with self-learning capabilities
Trains Random Forest models for betting predictions
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
import pickle
import os
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Extract and engineer features from match data"""
    
    def extract_live_features(self, fixture: Dict, statistics: List[Dict], events: List[Dict]) -> Optional[np.ndarray]:
        """Extract features from live match data"""
        try:
            features = []
            
            # Match state features
            elapsed = fixture['fixture']['status']['elapsed'] or 0
            features.append(elapsed)
            features.append(elapsed / 90)  # Normalized time
            
            # Score features
            home_goals = fixture['goals']['home'] or 0
            away_goals = fixture['goals']['away'] or 0
            features.extend([home_goals, away_goals, home_goals + away_goals])
            features.append(home_goals - away_goals)  # Goal difference
            
            # Initialize stats
            home_stats = {'shots': 0, 'shots_on_target': 0, 'attacks': 0, 
                         'dangerous_attacks': 0, 'possession': 50, 'corners': 0,
                         'fouls': 0, 'yellow_cards': 0, 'red_cards': 0}
            away_stats = home_stats.copy()
            
            # Extract statistics
            if statistics:
                for team_stat in statistics:
                    team_type = 'home' if team_stat['team']['id'] == fixture['teams']['home']['id'] else 'away'
                    stats_dict = home_stats if team_type == 'home' else away_stats
                    
                    for stat in team_stat.get('statistics', []):
                        stat_type = stat['type'].lower().replace(' ', '_')
                        value = stat['value']
                        
                        if stat_type == 'total_shots':
                            stats_dict['shots'] = self._parse_value(value)
                        elif stat_type == 'shots_on_goal':
                            stats_dict['shots_on_target'] = self._parse_value(value)
                        elif stat_type == 'total_attacks':
                            stats_dict['attacks'] = self._parse_value(value)
                        elif stat_type == 'dangerous_attacks':
                            stats_dict['dangerous_attacks'] = self._parse_value(value)
                        elif stat_type == 'ball_possession':
                            stats_dict['possession'] = self._parse_value(value, is_percentage=True)
                        elif stat_type == 'corner_kicks':
                            stats_dict['corners'] = self._parse_value(value)
                        elif stat_type == 'fouls':
                            stats_dict['fouls'] = self._parse_value(value)
                        elif stat_type == 'yellow_cards':
                            stats_dict['yellow_cards'] = self._parse_value(value)
                        elif stat_type == 'red_cards':
                            stats_dict['red_cards'] = self._parse_value(value)
            
            # Add statistical features
            features.extend([
                home_stats['shots'],
                away_stats['shots'],
                home_stats['shots_on_target'],
                away_stats['shots_on_target'],
                home_stats['attacks'],
                away_stats['attacks'],
                home_stats['dangerous_attacks'],
                away_stats['dangerous_attacks'],
                home_stats['possession'],
                away_stats['possession'],
                home_stats['corners'],
                away_stats['corners'],
                home_stats['fouls'],
                away_stats['fouls'],
                home_stats['yellow_cards'],
                away_stats['yellow_cards'],
                home_stats['red_cards'],
                away_stats['red_cards']
            ])
            
            # Derived features
            total_shots = home_stats['shots'] + away_stats['shots']
            features.append(total_shots)
            features.append(home_stats['shots'] - away_stats['shots'])  # Shot difference
            
            total_attacks = home_stats['attacks'] + away_stats['attacks']
            features.append(total_attacks)
            
            # Possession balance
            features.append(abs(home_stats['possession'] - away_stats['possession']))
            
            # Shot efficiency
            home_shot_eff = home_stats['shots_on_target'] / max(home_stats['shots'], 1)
            away_shot_eff = away_stats['shots_on_target'] / max(away_stats['shots'], 1)
            features.extend([home_shot_eff, away_shot_eff])
            
            # Pressure features
            home_pressure = (home_stats['attacks'] + home_stats['dangerous_attacks']) / max(elapsed, 1)
            away_pressure = (away_stats['attacks'] + away_stats['dangerous_attacks']) / max(elapsed, 1)
            features.extend([home_pressure, away_pressure])
            
            # Event-based features
            if events:
                goal_times = [e['time']['elapsed'] for e in events if e['type'] == 'Goal']
                features.append(len(goal_times))
                
                # Recent momentum (goals in last 15 mins)
                recent_goals = len([t for t in goal_times if elapsed - t <= 15])
                features.append(recent_goals)
            else:
                features.extend([0, 0])
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None
    
    def _parse_value(self, value, is_percentage: bool = False) -> float:
        """Parse statistical value"""
        if value is None:
            return 0.0
        
        if isinstance(value, (int, float)):
            return float(value)
        
        if isinstance(value, str):
            # Remove percentage sign
            if '%' in value:
                value = value.replace('%', '')
            
            try:
                return float(value)
            except:
                return 0.0
        
        return 0.0


class ModelTrainer:
    """Train and optimize Random Forest models"""
    
    def __init__(self, db_manager):
        self.db = db_manager
        self.feature_engineer = FeatureEngineer()
        os.makedirs('models', exist_ok=True)
        
    def fetch_training_data(self, min_samples: int = 1000) -> pd.DataFrame:
        """Fetch historical data from database for training"""
        try:
            with self.db.get_connection() as conn:
                # Fetch predictions with results
                query = """
                    SELECT 
                        p.*,
                        md.data as match_data
                    FROM predictions p
                    JOIN match_data md ON p.fixture_id = md.fixture_id
                    WHERE p.is_correct IS NOT NULL
                    ORDER BY p.created_at DESC
                    LIMIT %s
                """
                
                df = pd.read_sql_query(query, conn, params=(min_samples * 3,))
                logger.info(f"Fetched {len(df)} records for training")
                return df
                
        except Exception as e:
            logger.error(f"Error fetching training data: {e}")
            return pd.DataFrame()
    
    def prepare_dataset(self, df: pd.DataFrame, prediction_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare dataset for specific prediction type"""
        # Filter for prediction type
        df_filtered = df[df['prediction_type'] == prediction_type].copy()
        
        if len(df_filtered) < 100:
            logger.warning(f"Insufficient data for {prediction_type}: {len(df_filtered)} samples")
            return None, None
        
        X = []
        y = []
        
        for _, row in df_filtered.iterrows():
            try:
                # Parse match data
                match_data = row['match_data']
                
                # Extract features
                features = self.feature_engineer.extract_live_features(
                    match_data.get('fixture', {}),
                    match_data.get('statistics', []),
                    match_data.get('events', [])
                )
                
                if features is not None:
                    X.append(features)
                    
                    # Convert result to binary
                    label = 1 if row['is_correct'] else 0
                    y.append(label)
                    
            except Exception as e:
                logger.error(f"Error processing row: {e}")
                continue
        
        if len(X) == 0:
            return None, None
        
        return np.array(X), np.array(y)
    
    def train_model(self, X: np.ndarray, y: np.ndarray, model_type: str) -> Dict:
        """Train Random Forest model with calibration"""
        logger.info(f"Training {model_type} model with {len(X)} samples...")
        
        # Split data (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Training set: {len(X_train)}, Test set: {len(X_test)}")
        
        # Define Random Forest with optimized parameters
        rf_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2'],
            'class_weight': ['balanced']
        }
        
        # Base model
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        # Grid search for best parameters
        logger.info("Running hyperparameter optimization...")
        grid_search = GridSearchCV(
            rf, rf_params, cv=5, scoring='f1', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        best_rf = grid_search.best_estimator_
        logger.info(f"Best parameters: {grid_search.best_params_}")
        
        # Calibrate probabilities for better confidence estimates
        logger.info("Calibrating probabilities...")
        calibrated_model = CalibratedClassifierCV(best_rf, method='isotonic', cv=5)
        calibrated_model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = calibrated_model.predict(X_test)
        y_proba = calibrated_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        brier = brier_score_loss(y_test, y_proba)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'brier_score': brier,
            'test_samples': len(X_test),
            'train_samples': len(X_train)
        }
        
        logger.info(f"{model_type} Performance:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1-Score: {f1:.4f}")
        logger.info(f"  Brier Score: {brier:.4f}")
        
        # Save model
        model_path = f'models/{model_type}_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(calibrated_model, f)
        logger.info(f"Model saved to {model_path}")
        
        # Save performance to database
        with self.db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO model_performance 
                    (model_type, accuracy, precision, recall, f1_score, brier_score, 
                     total_predictions, correct_predictions)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    model_type, accuracy, precision, recall, f1, brier,
                    len(X_test), int(accuracy * len(X_test))
                ))
                conn.commit()
        
        return metrics
    
    def train_all_models(self) -> Dict:
        """Train all prediction models"""
        results = {}
        
        # Fetch training data
        df = self.fetch_training_data()
        
        if df.empty:
            logger.error("No training data available")
            return results
        
        # Train each model type
        model_types = ['over_under', 'btts', 'win_lose']
        
        for model_type in model_types:
            logger.info(f"\n{'='*50}")
            logger.info(f"Training {model_type.upper()} model")
            logger.info(f"{'='*50}")
            
            X, y = self.prepare_dataset(df, model_type)
            
            if X is None or len(X) < 100:
                logger.warning(f"Skipping {model_type} - insufficient data")
                continue
            
            metrics = self.train_model(X, y, model_type)
            results[model_type] = metrics
        
        logger.info("\n" + "="*50)
        logger.info("Training Summary:")
        for model_type, metrics in results.items():
            logger.info(f"\n{model_type.upper()}:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    logger.info(f"  {metric}: {value:.4f}")
                else:
                    logger.info(f"  {metric}: {value}")
        
        return results


if __name__ == '__main__':
    # Allow standalone training
    from main import DatabaseManager
    
    db = DatabaseManager()
    trainer = ModelTrainer(db)
    trainer.train_all_models()
