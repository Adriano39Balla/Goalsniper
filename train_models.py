import pandas as pd
import numpy as np
import pickle
import joblib
from datetime import datetime, timedelta
import logging
import sys
import os
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Custom imports
sys.path.append(str(Path(__file__).parent))
from database import DatabaseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Advanced model trainer for betting predictions"""
    
    def __init__(self, db_manager=None):
        self.db_manager = db_manager or DatabaseManager()
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.target_columns = ['home_win', 'draw', 'away_win', 'over_2_5', 'btts']
        
    def fetch_training_data(self, days_back: int = 365) -> pd.DataFrame:
        """Fetch historical data for training"""
        logger.info(f"Fetching training data for last {days_back} days")
        
        query = """
        SELECT 
            *,
            CASE WHEN home_score > away_score THEN 1 ELSE 0 END as home_win,
            CASE WHEN home_score = away_score THEN 1 ELSE 0 END as draw,
            CASE WHEN home_score < away_score THEN 1 ELSE 0 END as away_win,
            CASE WHEN home_score + away_score > 2.5 THEN 1 ELSE 0 END as over_2_5,
            CASE WHEN home_score > 0 AND away_score > 0 THEN 1 ELSE 0 END as btts
        FROM matches 
        WHERE timestamp > NOW() - INTERVAL '%s days'
        AND status = 'FT'
        AND home_score IS NOT NULL
        AND away_score IS NOT NULL
        ORDER BY timestamp
        """
        
        df = self.db_manager.execute_query(query, (days_back,))
        
        if df.empty:
            logger.warning("No training data found")
            return df
        
        logger.info(f"Loaded {len(df)} matches for training")
        return df
    
    def create_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        """Create advanced features for prediction"""
        logger.info("Creating advanced features")
        
        # Make a copy to avoid modifying original
        df_features = df.copy()
        
        # Basic features
        features = {
            'momentum_features': [],
            'form_features': [],
            'h2h_features': [],
            'statistical_features': [],
            'time_features': []
        }
        
        # Team momentum features (last 5 games)
        for team_type in ['home', 'away']:
            df_features[f'{team_type}_form_last5'] = 0
            df_features[f'{team_type}_goals_scored_last5'] = 0
            df_features[f'{team_type}_goals_conceded_last5'] = 0
            features['momentum_features'].extend([
                f'{team_type}_form_last5',
                f'{team_type}_goals_scored_last5',
                f'{team_type}_goals_conceded_last5'
            ])
        
        # Head-to-head features
        df_features['h2h_avg_goals'] = 2.5  # Default
        features['h2h_features'].append('h2h_avg_goals')
        
        # Statistical features
        df_features['total_goals_avg'] = df_features.get('avg_goals', 2.5)
        df_features['btts_percentage'] = df_features.get('btts_rate', 0.5)
        features['statistical_features'].extend(['total_goals_avg', 'btts_percentage'])
        
        # Time features
        df_features['hour_of_day'] = pd.to_datetime(df_features['timestamp']).dt.hour
        df_features['day_of_week'] = pd.to_datetime(df_features['timestamp']).dt.dayofweek
        df_features['month'] = pd.to_datetime(df_features['timestamp']).dt.month
        features['time_features'].extend(['hour_of_day', 'day_of_week', 'month'])
        
        # Market features if available
        if 'home_odds' in df_features.columns:
            df_features['implied_prob_home'] = 1 / df_features['home_odds']
            df_features['implied_prob_draw'] = 1 / df_features['draw_odds']
            df_features['implied_prob_away'] = 1 / df_features['away_odds']
            features['market_features'] = [
                'implied_prob_home', 'implied_prob_draw', 'implied_prob_away'
            ]
        
        # Combine all features
        all_features = []
        for feature_list in features.values():
            all_features.extend(feature_list)
        
        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        df_features[all_features] = imputer.fit_transform(df_features[all_features])
        
        logger.info(f"Created {len(all_features)} features")
        return df_features[all_features], features
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[Dict[str, tuple], Dict[str, Any]]:
        """Prepare data for each target variable"""
        logger.info("Preparing training data")
        
        if df.empty:
            raise ValueError("No data available for training")
        
        # Create features
        X, feature_groups = self.create_features(df)
        
        # Prepare datasets for each target
        datasets = {}
        for target in self.target_columns:
            if target in df.columns:
                y = df[target].values
                
                # Split with time series validation
                tscv = TimeSeriesSplit(n_splits=5)
                
                # Use 80/20 split respecting time order
                split_idx = int(len(X) * 0.8)
                X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
                
                # Scale features
                scaler = RobustScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                self.scalers[target] = scaler
                datasets[target] = (X_train_scaled, X_test_scaled, y_train, y_test)
                
                logger.info(f"{target}: Train={len(X_train)}, Test={len(X_test)}")
        
        return datasets, feature_groups
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray, 
                           target_name: str) -> RandomForestClassifier:
        """Train optimized Random Forest model"""
        logger.info(f"Training Random Forest for {target_name}")
        
        # Optimized hyperparameters based on target
        rf_params = {
            'n_estimators': 200,
            'max_depth': 15 if target_name in ['home_win', 'away_win'] else 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'bootstrap': True,
            'n_jobs': -1,
            'random_state': 42,
            'class_weight': 'balanced_subsample'
        }
        
        rf = RandomForestClassifier(**rf_params)
        rf.fit(X_train, y_train)
        
        # Calibrate probabilities
        calibrated_rf = CalibratedClassifierCV(rf, method='sigmoid', cv=5)
        calibrated_rf.fit(X_train, y_train)
        
        # Store feature importance
        self.feature_importance[target_name] = {
            'features': list(range(X_train.shape[1])),
            'importance': calibrated_rf.base_estimator.feature_importances_.tolist()
        }
        
        return calibrated_rf
    
    def train_ensemble(self, X_train: np.ndarray, y_train: np.ndarray,
                      target_name: str) -> VotingClassifier:
        """Train ensemble model combining multiple algorithms"""
        logger.info(f"Training Ensemble model for {target_name}")
        
        # Define individual models
        rf = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        lgb_model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            num_leaves=31,
            random_state=42,
            n_jobs=-1
        )
        
        # Create voting classifier
        ensemble = VotingClassifier(
            estimators=[
                ('rf', rf),
                ('xgb', xgb_model),
                ('lgb', lgb_model)
            ],
            voting='soft',
            weights=[1.5, 1.0, 1.0]  # Weight RF more heavily as requested
        )
        
        ensemble.fit(X_train, y_train)
        
        # Calibrate the ensemble
        calibrated_ensemble = CalibratedClassifierCV(ensemble, method='isotonic', cv=5)
        calibrated_ensemble.fit(X_train, y_train)
        
        return calibrated_ensemble
    
    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray, 
                      target_name: str) -> Dict[str, float]:
        """Evaluate model performance comprehensively"""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0,
            'brier_score': np.mean((y_pred_proba - y_test) ** 2),
            'log_loss': -np.mean(y_test * np.log(y_pred_proba + 1e-10) + 
                                (1 - y_test) * np.log(1 - y_pred_proba + 1e-10))
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        metrics['true_positives'] = cm[1, 1]
        metrics['false_positives'] = cm[0, 1]
        metrics['true_negatives'] = cm[0, 0]
        metrics['false_negatives'] = cm[1, 0]
        
        logger.info(f"{target_name} Evaluation:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def save_models(self, model_type: str = 'ensemble'):
        """Save trained models and scalers"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = Path(f"models/{timestamp}")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save models
        for target, model in self.models.items():
            model_path = model_dir / f"{target}_{model_type}.pkl"
            joblib.dump(model, model_path)
            logger.info(f"Saved model for {target} to {model_path}")
        
        # Save scalers
        scalers_path = model_dir / "scalers.pkl"
        joblib.dump(self.scalers, scalers_path)
        
        # Save feature importance
        importance_path = model_dir / "feature_importance.pkl"
        joblib.dump(self.feature_importance, importance_path)
        
        # Update current model pointer
        current_path = Path("models/current")
        if current_path.exists():
            current_path.unlink()
        current_path.symlink_to(model_dir)
        
        logger.info(f"Models saved to {model_dir}")
        return str(model_dir)
    
    def load_latest_models(self) -> bool:
        """Load the latest trained models"""
        current_path = Path("models/current")
        if not current_path.exists():
            logger.warning("No current models found")
            return False
        
        try:
            # Load models
            for target in self.target_columns:
                model_path = current_path / f"{target}_ensemble.pkl"
                if model_path.exists():
                    self.models[target] = joblib.load(model_path)
            
            # Load scalers
            scalers_path = current_path / "scalers.pkl"
            if scalers_path.exists():
                self.scalers = joblib.load(scalers_path)
            
            logger.info("Loaded latest models successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def train_all_models(self, model_type: str = 'ensemble', days_back: int = 365) -> Dict:
        """Train all prediction models"""
        logger.info("Starting model training pipeline")
        
        try:
            # Fetch data
            df = self.fetch_training_data(days_back)
            
            if df.empty or len(df) < 100:
                logger.warning(f"Insufficient data: {len(df)} samples. Need at least 100.")
                return {"success": False, "message": "Insufficient training data"}
            
            # Prepare data
            datasets, feature_groups = self.prepare_data(df)
            
            # Train models for each target
            all_metrics = {}
            for target, (X_train, X_test, y_train, y_test) in datasets.items():
                logger.info(f"\nTraining model for {target}")
                
                # Train model
                if model_type == 'ensemble':
                    model = self.train_ensemble(X_train, y_train, target)
                else:
                    model = self.train_random_forest(X_train, y_train, target)
                
                # Evaluate
                metrics = self.evaluate_model(model, X_test, y_test, target)
                
                # Store model
                self.models[target] = model
                all_metrics[target] = metrics
            
            # Save models
            model_dir = self.save_models(model_type)
            
            # Calculate overall performance
            avg_accuracy = np.mean([m['accuracy'] for m in all_metrics.values()])
            avg_roc_auc = np.mean([m['roc_auc'] for m in all_metrics.values() 
                                  if m['roc_auc'] > 0])
            
            result = {
                "success": True,
                "model_dir": model_dir,
                "metrics": all_metrics,
                "avg_accuracy": avg_accuracy,
                "avg_roc_auc": avg_roc_auc,
                "training_samples": len(df),
                "feature_groups": feature_groups
            }
            
            logger.info(f"\nTraining completed:")
            logger.info(f"Average Accuracy: {avg_accuracy:.4f}")
            logger.info(f"Average ROC AUC: {avg_roc_auc:.4f}")
            logger.info(f"Models saved to: {model_dir}")
            
            # Log to database
            self.log_training_session(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
    
    def log_training_session(self, result: Dict):
        """Log training session to database"""
        try:
            query = """
            INSERT INTO training_sessions 
            (timestamp, model_type, avg_accuracy, avg_roc_auc, 
             training_samples, metrics, model_path)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            
            self.db_manager.execute_query(
                query,
                (
                    datetime.now(),
                    'ensemble',
                    result['avg_accuracy'],
                    result['avg_roc_auc'],
                    result['training_samples'],
                    str(result['metrics']),
                    result['model_dir']
                )
            )
            logger.info("Training session logged to database")
        except Exception as e:
            logger.error(f"Failed to log training session: {e}")

def main():
    """Main training function"""
    trainer = ModelTrainer()
    
    print("Starting model training...")
    print("=" * 50)
    
    result = trainer.train_all_models(model_type='ensemble', days_back=180)
    
    if result['success']:
        print(f"\n✅ Training successful!")
        print(f"📊 Average Accuracy: {result['avg_accuracy']:.4f}")
        print(f"🎯 Average ROC AUC: {result['avg_roc_auc']:.4f}")
        print(f"📁 Models saved to: {result['model_dir']}")
    else:
        print(f"\n❌ Training failed: {result.get('error', 'Unknown error')}")
    
    return result

if __name__ == "__main__":
    main()
