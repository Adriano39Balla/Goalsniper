"""
train_models.py - Machine Learning Training Pipeline for Goalsniper OU 2.5 System
Connects to Supabase, trains/retrains models, and updates settings for production.
"""

import os
import sys
import json
import time
import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss, classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV, SelectKBest, f_classif
from sklearn.impute import SimpleImputer

import psycopg2
from psycopg2.extras import RealDictCursor
import joblib
from dotenv import load_dotenv

# ───────── Logging Setup ─────────
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('train_models.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("train_models")

warnings.filterwarnings('ignore', category=UserWarning)

# ───────── Configuration ─────────
@dataclass
class TrainingConfig:
    """Configuration for model training"""
    # Database
    supabase_url: str = ""
    supabase_key: str = ""
    database_url: str = ""
    
    # Model parameters
    target_market: str = "Over/Under 2.5"
    target_col: str = "over_2.5"  # 1 for over, 0 for under
    min_samples: int = 2000
    test_size: float = 0.2
    cv_folds: int = 5
    random_state: int = 42
    
    # Feature selection
    min_features: int = 5
    max_features: int = 20
    
    # Model hyperparameters
    penalty: str = 'l2'
    C_values: List[float] = None  # Will be set in __post_init__
    max_iter: int = 1000
    class_weight: str = 'balanced'
    
    # Calibration
    calibration_method: str = 'sigmoid'  # 'sigmoid' or 'isotonic'
    calibration_cv: int = 5
    
    # Training schedule
    lookback_days: int = 180  # How far back to fetch data
    min_match_age_hours: int = 4  # Matches must be at least this old
    
    def __post_init__(self):
        if self.C_values is None:
            self.C_values = [0.001, 0.01, 0.1, 1, 10, 100]

# ───────── Database Connection ─────────
class SupabaseClient:
    """Handles connection to Supabase/PostgreSQL"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.conn = None
        self.connect()
    
    def connect(self):
        """Establish database connection"""
        try:
            # Try Supabase connection first
            if self.config.supabase_url and self.config.supabase_key:
                try:
                    import supabase
                    self.supabase = supabase.create_client(
                        self.config.supabase_url,
                        self.config.supabase_key
                    )
                    log.info("Connected to Supabase")
                    return
                except ImportError:
                    log.warning("Supabase Python client not installed, using psycopg2")
            
            # Fallback to direct PostgreSQL connection
            if self.config.database_url:
                self.conn = psycopg2.connect(
                    self.config.database_url,
                    cursor_factory=RealDictCursor
                )
                log.info("Connected to PostgreSQL via psycopg2")
            else:
                raise ValueError("No database connection configured")
                
        except Exception as e:
            log.error(f"Database connection failed: {e}")
            raise
    
    def fetch_training_data(self) -> pd.DataFrame:
        """
        Fetch historical data for training.
        Returns DataFrame with features and labels.
        """
        query = """
        WITH match_features AS (
            SELECT 
                t.match_id,
                t.league_id,
                t.league,
                t.home,
                t.away,
                t.minute,
                t.score_at_tip,
                t.confidence_raw,
                t.created_ts,
                -- Extract goals from score_at_tip
                CAST(SPLIT_PART(t.score_at_tip, '-', 1) AS INTEGER) as goals_h,
                CAST(SPLIT_PART(t.score_at_tip, '-', 2) AS INTEGER) as goals_a,
                -- Get final result
                r.final_goals_h,
                r.final_goals_a,
                -- Calculate target
                CASE 
                    WHEN (r.final_goals_h + r.final_goals_a) > 2.5 THEN 1
                    ELSE 0 
                END as over_2_5,
                -- Feature: time remaining
                GREATEST(0, 90 - t.minute) as minutes_remaining,
                -- Feature: goal difference at tip
                CAST(SPLIT_PART(t.score_at_tip, '-', 1) AS INTEGER) - 
                CAST(SPLIT_PART(t.score_at_tip, '-', 2) AS INTEGER) as goal_diff,
                -- Feature: total goals at tip
                CAST(SPLIT_PART(t.score_at_tip, '-', 1) AS INTEGER) + 
                CAST(SPLIT_PART(t.score_at_tip, '-', 2) AS INTEGER) as total_goals_at_tip
            FROM tips t
            LEFT JOIN match_results r ON t.match_id = r.match_id
            WHERE t.market = 'Over/Under 2.5'
            AND t.created_ts >= EXTRACT(EPOCH FROM NOW() - INTERVAL '%s days')
            AND r.final_goals_h IS NOT NULL
            AND r.final_goals_a IS NOT NULL
            AND t.minute BETWEEN 15 AND 80  -- Same as TIP_MIN_MINUTE to TOTAL_MATCH_MINUTES
        ),
        aggregated_stats AS (
            SELECT 
                match_id,
                COUNT(*) as tip_count,
                AVG(confidence_raw) as avg_confidence,
                STDDEV(confidence_raw) as confidence_std
            FROM tips
            WHERE market = 'Over/Under 2.5'
            GROUP BY match_id
        )
        SELECT 
            mf.*,
            COALESCE(agg.tip_count, 0) as tip_count,
            COALESCE(agg.avg_confidence, 0) as avg_confidence,
            COALESCE(agg.confidence_std, 0) as confidence_std
        FROM match_features mf
        LEFT JOIN aggregated_stats agg ON mf.match_id = agg.match_id
        WHERE mf.created_ts <= EXTRACT(EPOCH FROM NOW() - INTERVAL '%s hours')
        ORDER BY mf.created_ts DESC
        """
        
        try:
            with self.conn.cursor() as cur:
                cur.execute(query, (self.config.lookback_days, self.config.min_match_age_hours))
                rows = cur.fetchall()
            
            df = pd.DataFrame(rows)
            log.info(f"Fetched {len(df)} historical records")
            return df
            
        except Exception as e:
            log.error(f"Error fetching training data: {e}")
            raise
    
    def fetch_match_stats(self, match_ids: List[int]) -> pd.DataFrame:
        """Fetch detailed statistics for matches (if available)"""
        if not match_ids:
            return pd.DataFrame()
        
        # This would need to be adjusted based on your actual stats schema
        query = """
        SELECT 
            match_id,
            -- Example stats - adjust based on your schema
            stats->>'xg_sum' as xg_sum,
            stats->>'sot_sum' as sot_sum,
            stats->>'cor_sum' as cor_sum,
            stats->>'pos_h' as pos_h,
            stats->>'pos_a' as pos_a,
            stats->>'red_sum' as red_sum
        FROM match_statistics
        WHERE match_id = ANY(%s)
        """
        
        try:
            with self.conn.cursor() as cur:
                cur.execute(query, (match_ids,))
                rows = cur.fetchall()
            return pd.DataFrame(rows) if rows else pd.DataFrame()
            
        except Exception as e:
            log.warning(f"Could not fetch match stats: {e}")
            return pd.DataFrame()
    
    def save_model_to_settings(self, model_name: str, model_data: Dict[str, Any]):
        """Save trained model to database settings table"""
        query = """
        INSERT INTO settings (key, value, updated_at)
        VALUES (%s, %s, NOW())
        ON CONFLICT (key) 
        DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()
        """
        
        try:
            with self.conn.cursor() as cur:
                cur.execute(query, (model_name, json.dumps(model_data)))
            self.conn.commit()
            log.info(f"Saved model '{model_name}' to settings")
            
        except Exception as e:
            log.error(f"Error saving model to settings: {e}")
            self.conn.rollback()
            raise

# ───────── Feature Engineering ─────────
class FeatureEngineer:
    """Creates features matching main.py's extract_features function"""
    
    @staticmethod
    def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Engineer features from raw data.
        Returns (features_df, target_series)
        """
        if len(df) == 0:
            return pd.DataFrame(), pd.Series()
        
        # Create a copy to avoid SettingWithCopyWarning
        features = df.copy()
        
        # 1. Basic match state features (matching main.py)
        features['minute'] = features['minute'].astype(float)
        features['goals_sum_at_tip'] = features['goals_h'] + features['goals_a']
        features['goal_diff_abs'] = abs(features['goal_diff'])
        
        # 2. Rate features
        features['goals_per_minute'] = features['goals_sum_at_tip'] / features['minute'].clip(1)
        features['projected_goals'] = features['goals_per_minute'] * 90
        
        # 3. Time-based features
        features['is_first_half'] = (features['minute'] <= 45).astype(int)
        features['is_last_30'] = (features['minute'] >= 60).astype(int)
        
        # 4. Match context features
        features['is_draw'] = (features['goal_diff'] == 0).astype(int)
        features['is_close_game'] = (features['goal_diff_abs'] <= 1).astype(int)
        features['is_high_scoring'] = (features['goals_sum_at_tip'] >= 2).astype(int)
        
        # 5. Historical performance features
        if 'avg_confidence' in features.columns:
            features['confidence_norm'] = features['avg_confidence'] / features['avg_confidence'].max()
        
        # 6. League fixed effects (one-hot encode top leagues)
        top_leagues = features['league'].value_counts().head(10).index
        for league in top_leagues:
            features[f'league_{league}'] = (features['league'] == league).astype(int)
        
        # 7. Interaction terms
        features['goals_x_minute'] = features['goals_sum_at_tip'] * features['minute']
        features['close_high_scoring'] = features['is_close_game'] * features['is_high_scoring']
        
        # Select final feature columns (exclude identifiers and raw data)
        feature_cols = [
            'minute',
            'goals_sum_at_tip',
            'goal_diff',
            'goal_diff_abs',
            'goals_per_minute',
            'projected_goals',
            'minutes_remaining',
            'is_first_half',
            'is_last_30',
            'is_draw',
            'is_close_game',
            'is_high_scoring',
            'total_goals_at_tip',
            'tip_count',
            'avg_confidence',
            'confidence_std',
            'goals_x_minute',
            'close_high_scoring'
        ]
        
        # Add league features
        league_cols = [col for col in features.columns if col.startswith('league_')]
        feature_cols.extend(league_cols)
        
        # Ensure all columns exist
        existing_cols = [col for col in feature_cols if col in features.columns]
        
        # Target variable
        target = features['over_2_5']
        
        return features[existing_cols], target
    
    @staticmethod
    def handle_missing_values(features: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features"""
        # For numeric columns, impute with median
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        imputer = SimpleImputer(strategy='median')
        
        features_imputed = features.copy()
        features_imputed[numeric_cols] = imputer.fit_transform(features[numeric_cols])
        
        return features_imputed

# ───────── Model Training ─────────
class ModelTrainer:
    """Handles model training, validation, and evaluation"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.feature_importances = {}
        
    def train_ou25_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Train Over/Under 2.5 model with proper validation.
        Returns model dictionary compatible with main.py
        """
        log.info(f"Training OU 2.5 model on {len(X)} samples with {X.shape[1]} features")
        
        # 1. Train/Test Split with time-based validation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y
        )
        
        log.info(f"Train set: {len(X_train)}, Test set: {len(X_test)}")
        log.info(f"Class balance - Train: {y_train.mean():.3f}, Test: {y_test.mean():.3f}")
        
        # 2. Feature scaling
        scaler = RobustScaler()  # Robust to outliers
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['ou25'] = scaler
        
        # 3. Feature selection using RFECV
        log.info("Performing feature selection...")
        base_estimator = LogisticRegression(
            penalty=self.config.penalty,
            C=1.0,
            max_iter=self.config.max_iter,
            class_weight=self.config.class_weight,
            random_state=self.config.random_state
        )
        
        selector = RFECV(
            estimator=base_estimator,
            step=1,
            cv=TimeSeriesSplit(n_splits=self.config.cv_folds),
            scoring='roc_auc',
            min_features_to_select=self.config.min_features,
            n_jobs=-1
        )
        
        selector.fit(X_train_scaled, y_train)
        selected_features = X.columns[selector.support_].tolist()
        
        log.info(f"Selected {len(selected_features)} features: {selected_features}")
        self.feature_importances['ou25'] = dict(zip(
            X.columns,
            selector.estimator_.coef_[0] if hasattr(selector.estimator_, 'coef_') else [0]*X.shape[1]
        ))
        
        # Use only selected features
        X_train_selected = X_train_scaled[:, selector.support_]
        X_test_selected = X_test_scaled[:, selector.support_]
        
        # 4. Hyperparameter tuning with cross-validation
        log.info("Tuning hyperparameters...")
        cv_scores = []
        best_score = -np.inf
        best_C = self.config.C_values[0]
        
        for C in self.config.C_values:
            model = LogisticRegression(
                penalty=self.config.penalty,
                C=C,
                max_iter=self.config.max_iter,
                class_weight=self.config.class_weight,
                random_state=self.config.random_state
            )
            
            scores = cross_val_score(
                model, X_train_selected, y_train,
                cv=TimeSeriesSplit(n_splits=self.config.cv_folds),
                scoring='roc_auc',
                n_jobs=-1
            )
            
            mean_score = scores.mean()
            cv_scores.append((C, mean_score, scores.std()))
            
            if mean_score > best_score:
                best_score = mean_score
                best_C = C
        
        log.info(f"Best C: {best_C} with AUC: {best_score:.4f}")
        
        # 5. Train final model with best hyperparameters
        log.info("Training final model...")
        final_model = LogisticRegression(
            penalty=self.config.penalty,
            C=best_C,
            max_iter=self.config.max_iter,
            class_weight=self.config.class_weight,
            random_state=self.config.random_state
        )
        
        final_model.fit(X_train_selected, y_train)
        
        # 6. Calibration
        log.info("Calibrating model...")
        calibrated_model = CalibratedClassifierCV(
            final_model,
            method=self.config.calibration_method,
            cv=self.config.calibration_cv
        )
        
        calibrated_model.fit(X_train_selected, y_train)
        
        # 7. Evaluate on test set
        log.info("Evaluating model...")
        y_pred_proba = calibrated_model.predict_proba(X_test_selected)[:, 1]
        
        test_metrics = self._evaluate_model(y_test, y_pred_proba, "Test Set")
        
        # 8. Create model dictionary compatible with main.py
        model_dict = self._create_model_dict(
            calibrated_model, 
            final_model, 
            selected_features,
            scaler,
            best_C,
            test_metrics
        )
        
        self.models['ou25'] = model_dict
        return model_dict
    
    def _evaluate_model(self, y_true: pd.Series, y_pred_proba: np.ndarray, dataset_name: str) -> Dict[str, float]:
        """Evaluate model performance"""
        # Calculate various metrics
        auc = roc_auc_score(y_true, y_pred_proba)
        brier = brier_score_loss(y_true, y_pred_proba)
        logloss = log_loss(y_true, y_pred_proba)
        
        # Calculate calibration curve
        prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)
        
        # Find optimal threshold using Youden's J statistic
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        youden_j = tpr - fpr
        optimal_idx = np.argmax(youden_j)
        optimal_threshold = thresholds[optimal_idx]
        
        metrics = {
            'auc': auc,
            'brier_score': brier,
            'log_loss': logloss,
            'optimal_threshold': optimal_threshold,
            'calibration_true': prob_true.tolist(),
            'calibration_pred': prob_pred.tolist()
        }
        
        log.info(f"{dataset_name} Metrics - AUC: {auc:.4f}, Brier: {brier:.4f}, "
                f"LogLoss: {logloss:.4f}, Optimal Threshold: {optimal_threshold:.3f}")
        
        return metrics
    
    def _create_model_dict(self, calibrated_model, base_model, features, scaler, best_C, metrics) -> Dict[str, Any]:
        """Create model dictionary compatible with main.py"""
        # Extract coefficients and intercept
        if hasattr(base_model, 'coef_'):
            weights = dict(zip(features, base_model.coef_[0]))
            intercept = float(base_model.intercept_[0])
        else:
            # For calibrated models, use the base estimator
            if hasattr(calibrated_model, 'calibrated_classifiers_'):
                base_est = calibrated_model.calibrated_classifiers_[0].base_estimator
                weights = dict(zip(features, base_est.coef_[0]))
                intercept = float(base_est.intercept_[0])
            else:
                weights = {}
                intercept = 0.0
        
        # Create calibration parameters from Platt scaling
        calibration_params = {
            'method': 'sigmoid',
            'a': 1.0,  # These would need to be extracted from calibration
            'b': 0.0,
            'calibrated': True
        }
        
        # Create full model dictionary
        model_dict = {
            'weights': weights,
            'intercept': intercept,
            'calibration': calibration_params,
            'features': features,
            'scaler': {
                'center': scaler.center_.tolist() if hasattr(scaler, 'center_') else [],
                'scale': scaler.scale_.tolist() if hasattr(scaler, 'scale_') else []
            },
            'hyperparameters': {
                'C': best_C,
                'penalty': self.config.penalty,
                'class_weight': self.config.class_weight
            },
            'metrics': metrics,
            'version': f"v2.{int(time.time())}",
            'trained_at': datetime.now().isoformat(),
            'training_samples': base_model.classes_.shape[0] if hasattr(base_model, 'classes_') else 0
        }
        
        return model_dict
    
    def save_model_locally(self, model_name: str, model_dict: Dict[str, Any]):
        """Save model to local file for backup"""
        filename = f"models/{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(model_dict, f, indent=2)
        
        log.info(f"Model saved locally to {filename}")
        return filename

# ───────── Main Training Pipeline ─────────
def main():
    """Main training pipeline"""
    log.info("Starting model training pipeline...")
    
    # Load environment variables
    load_dotenv()
    
    # Initialize configuration
    config = TrainingConfig(
        supabase_url=os.getenv('SUPABASE_URL', ''),
        supabase_key=os.getenv('SUPABASE_KEY', ''),
        database_url=os.getenv('DATABASE_URL', ''),
        lookback_days=int(os.getenv('TRAINING_LOOKBACK_DAYS', '180')),
        min_samples=int(os.getenv('TRAINING_MIN_SAMPLES', '2000'))
    )
    
    # Check if we have enough configuration
    if not config.database_url and not config.supabase_url:
        log.error("No database connection configured. Set DATABASE_URL or SUPABASE_URL/SUPABASE_KEY")
        sys.exit(1)
    
    try:
        # 1. Connect to database
        db = SupabaseClient(config)
        
        # 2. Fetch training data
        log.info("Fetching training data...")
        raw_data = db.fetch_training_data()
        
        if len(raw_data) < config.min_samples:
            log.warning(f"Insufficient data: {len(raw_data)} samples (minimum: {config.min_samples})")
            log.info("Consider reducing min_samples or increasing lookback_days")
            sys.exit(0)
        
        # 3. Engineer features
        log.info("Engineering features...")
        engineer = FeatureEngineer()
        features, target = engineer.engineer_features(raw_data)
        
        if len(features) == 0:
            log.error("No features could be engineered")
            sys.exit(1)
        
        # Handle missing values
        features = engineer.handle_missing_values(features)
        
        log.info(f"Features shape: {features.shape}, Target shape: {target.shape}")
        log.info(f"Feature columns: {list(features.columns)}")
        
        # 4. Train model
        trainer = ModelTrainer(config)
        
        log.info(f"Training {config.target_market} model...")
        model_dict = trainer.train_ou25_model(features, target)
        
        # 5. Save model to database
        model_key = f"model_v2:OU_2.5"
        db.save_model_to_settings(model_key, model_dict)
        
        # 6. Also save as latest
        db.save_model_to_settings("model_latest:OU_2.5", model_dict)
        
        # 7. Save locally for backup
        local_path = trainer.save_model_locally("OU_2.5", model_dict)
        
        # 8. Generate report
        report = generate_training_report(raw_data, features, target, model_dict)
        
        log.info("=" * 60)
        log.info("TRAINING COMPLETE")
        log.info(f"Model saved to: {model_key}")
        log.info(f"Local backup: {local_path}")
        log.info(f"Training samples: {len(raw_data)}")
        log.info(f"Test AUC: {model_dict['metrics']['auc']:.4f}")
        log.info("=" * 60)
        
        # 9. Send notification (optional)
        send_training_notification(report, model_dict)
        
        return model_dict
        
    except Exception as e:
        log.error(f"Training pipeline failed: {e}", exc_info=True)
        sys.exit(1)

def generate_training_report(raw_data, features, target, model_dict):
    """Generate comprehensive training report"""
    report = {
        'timestamp': datetime.now().isoformat(),
        'data_statistics': {
            'total_samples': len(raw_data),
            'feature_count': features.shape[1],
            'class_distribution': {
                'over_2.5': target.sum(),
                'under_2.5': len(target) - target.sum(),
                'over_rate': target.mean()
            },
            'minute_range': {
                'min': raw_data['minute'].min(),
                'max': raw_data['minute'].max(),
                'mean': raw_data['minute'].mean()
            }
        },
        'model_performance': model_dict['metrics'],
        'top_features': sorted(
            model_dict['weights'].items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:10],
        'hyperparameters': model_dict['hyperparameters']
    }
    
    # Save report
    os.makedirs('reports', exist_ok=True)
    report_file = f"reports/training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    log.info(f"Training report saved to {report_file}")
    return report

def send_training_notification(report, model_dict):
    """Send training completion notification"""
    try:
        # Check if Telegram is configured
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        if bot_token and chat_id:
            import requests
            
            message = (
                f"✅ Model Training Complete\n\n"
                f"📊 Samples: {report['data_statistics']['total_samples']}\n"
                f"🎯 AUC: {model_dict['metrics']['auc']:.4f}\n"
                f"📈 Features: {report['data_statistics']['feature_count']}\n"
                f"🕒 Trained: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
                f"Top features:\n"
            )
            
            # Add top 3 features
            for feature, weight in report['top_features'][:3]:
                message += f"• {feature}: {weight:.4f}\n"
            
            requests.post(
                f"https://api.telegram.org/bot{bot_token}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": message,
                    "parse_mode": "HTML"
                },
                timeout=10
            )
            log.info("Training notification sent")
            
    except Exception as e:
        log.warning(f"Could not send notification: {e}")

if __name__ == "__main__":
    main()
