#!/usr/bin/env python3
"""
train_models.py - Enhanced training script for goalsniper OU 2.5
Focused on improving model calibration and threshold optimization
"""

import os, json, time, logging, sys, warnings, traceback
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy import stats, optimize
import psycopg2
from psycopg2.extras import RealDictCursor
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# ML imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.model_selection import (
    train_test_split, 
    TimeSeriesSplit,
    cross_val_score,
    RandomizedSearchCV
)
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    log_loss,
    brier_score_loss,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("train_models")

# Load environment
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    logger.error("DATABASE_URL environment variable required")
    sys.exit(1)

# Training parameters from your logs analysis
MIN_SAMPLES = int(os.getenv("MIN_TRAINING_SAMPLES", "300"))
CURRENT_THRESHOLD = 72.0  # From your logs: 71.08% was below threshold
TARGET_ACCURACY = 65.0  # Target accuracy for threshold adjustment
MIN_CONFIDENCE_FOR_TRAINING = 60.0  # Minimum confidence to include in training

# Database connection
def get_db_connection():
    """Get database connection"""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise

def analyze_current_performance():
    """Analyze current model performance from logs and database"""
    logger.info("Analyzing current model performance...")
    
    query = """
        SELECT 
            COUNT(*) as total_tips,
            SUM(CASE WHEN r.final_goals_h IS NOT NULL AND r.final_goals_a IS NOT NULL THEN 1 ELSE 0 END) as graded_tips,
            SUM(CASE WHEN (r.final_goals_h + r.final_goals_a) > 2.5 AND t.suggestion = 'Over 2.5 Goals' THEN 1
                     WHEN (r.final_goals_h + r.final_goals_a) < 2.5 AND t.suggestion = 'Under 2.5 Goals' THEN 1
                     ELSE 0 END) as correct_tips,
            AVG(t.confidence) as avg_confidence,
            AVG(t.odds) as avg_odds,
            AVG(t.ev_pct) as avg_ev
        FROM tips t
        LEFT JOIN match_results r ON t.match_id = r.match_id
        WHERE t.market = 'Over/Under 2.5'
          AND t.sent_ok = 1
          AND t.created_ts >= %s
    """
    
    week_ago = int(time.time()) - (7 * 24 * 3600)
    
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (week_ago,))
                result = cur.fetchone()
                
                if result:
                    total_tips, graded_tips, correct_tips, avg_conf, avg_odds, avg_ev = result
                    
                    accuracy = (correct_tips / max(1, graded_tips)) * 100 if graded_tips else 0
                    
                    logger.info(f"Performance last 7 days:")
                    logger.info(f"  Total tips sent: {total_tips}")
                    logger.info(f"  Graded tips: {graded_tips}")
                    logger.info(f"  Correct tips: {correct_tips}")
                    logger.info(f"  Accuracy: {accuracy:.1f}%")
                    logger.info(f"  Avg confidence: {avg_conf:.1f}%")
                    logger.info(f"  Avg odds: {avg_odds:.2f}")
                    logger.info(f"  Avg EV: {avg_ev:.1f}%")
                    
                    return {
                        'accuracy': accuracy,
                        'total_tips': total_tips,
                        'graded_tips': graded_tips,
                        'correct_tips': correct_tips,
                        'avg_confidence': avg_conf,
                        'avg_odds': avg_odds,
                        'avg_ev': avg_ev
                    }
    
    except Exception as e:
        logger.error(f"Error analyzing performance: {e}")
    
    return None

def load_training_data_with_metadata(days_back: int = 180):
    """Load training data with detailed metadata"""
    logger.info(f"Loading training data from last {days_back} days")
    
    cutoff_ts = int(time.time()) - (days_back * 24 * 3600)
    
    query = """
        SELECT 
            t.match_id,
            t.league,
            t.home,
            t.away,
            t.minute,
            t.score_at_tip,
            t.suggestion,
            t.confidence,
            t.confidence_raw,
            t.odds,
            t.ev_pct,
            t.learning_features,
            r.final_goals_h,
            r.final_goals_a,
            t.created_ts,
            CASE 
                WHEN (r.final_goals_h + r.final_goals_a) > 2.5 THEN 1
                ELSE 0
            END as over_2_5_result
        FROM tips t
        LEFT JOIN match_results r ON t.match_id = r.match_id
        WHERE t.market = 'Over/Under 2.5'
          AND t.sent_ok = 1
          AND t.created_ts >= %s
          AND t.learning_features IS NOT NULL
        ORDER BY t.created_ts
    """
    
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (cutoff_ts,))
                rows = cur.fetchall()
        
        logger.info(f"Retrieved {len(rows)} tips from database")
        
        if len(rows) < MIN_SAMPLES:
            logger.warning(f"Insufficient data: {len(rows)} < {MIN_SAMPLES}")
            return pd.DataFrame(), pd.Series(), pd.DataFrame()
        
        # Process data
        features_list = []
        targets = []
        metadata = []
        
        for row in rows:
            try:
                # Parse learning features
                if isinstance(row['learning_features'], str):
                    learning_features = json.loads(row['learning_features'])
                else:
                    learning_features = row['learning_features']
                
                if not learning_features or 'features' not in learning_features:
                    continue
                
                feat = learning_features['features']
                
                # Only include if we have sufficient stats
                if not has_sufficient_stats(feat, row['minute']):
                    continue
                
                # Only include if original confidence was reasonable
                if row['confidence'] and row['confidence'] < MIN_CONFIDENCE_FOR_TRAINING:
                    continue
                
                # Extract features
                feature_dict = extract_enhanced_features(feat, row['minute'])
                
                # Target: 1 for Over 2.5, 0 for Under 2.5
                if row['final_goals_h'] is None or row['final_goals_a'] is None:
                    continue
                
                total_goals = row['final_goals_h'] + row['final_goals_a']
                target = 1 if total_goals > 2.5 else 0
                
                features_list.append(feature_dict)
                targets.append(target)
                
                metadata.append({
                    'match_id': row['match_id'],
                    'league': row['league'],
                    'minute': row['minute'],
                    'score_at_tip': row['score_at_tip'],
                    'suggestion': row['suggestion'],
                    'confidence': row['confidence'],
                    'confidence_raw': row['confidence_raw'],
                    'odds': row['odds'],
                    'ev_pct': row['ev_pct'],
                    'total_goals': total_goals,
                    'timestamp': row['created_ts'],
                    'over_2_5_result': row['over_2_5_result']
                })
                
            except Exception as e:
                logger.warning(f"Error processing row: {e}")
                continue
        
        # Create DataFrames
        X = pd.DataFrame(features_list)
        y = pd.Series(targets, name='target')
        meta_df = pd.DataFrame(metadata)
        
        logger.info(f"Processed {len(X)} samples")
        logger.info(f"Class distribution: {y.value_counts().to_dict()}")
        logger.info(f"Over 2.5 rate: {y.mean():.2%}")
        
        # Analyze feature availability
        analyze_feature_availability(X)
        
        return X, y, meta_df
        
    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        traceback.print_exc()
        return pd.DataFrame(), pd.Series(), pd.DataFrame()

def has_sufficient_stats(features: Dict, minute: int) -> bool:
    """Check if match has sufficient statistics for training"""
    # Same logic as in main app
    if minute < 35:  # REQUIRE_STATS_MINUTE
        return True
    
    required_fields = [
        features.get("xg_sum", 0.0),
        features.get("sot_sum", 0.0),
        features.get("cor_sum", 0.0),
        (features.get("pos_h", 0.0) + features.get("pos_a", 0.0)),
    ]
    
    nonzero = sum(1 for v in required_fields if (v or 0) > 0)
    return nonzero >= 2  # REQUIRE_DATA_FIELDS

def extract_enhanced_features(feat: Dict, minute: int) -> Dict:
    """Extract enhanced features with minute-specific adjustments"""
    features = {
        # Basic features
        'minute': float(minute),
        'goals_h': float(feat.get('goals_h', 0)),
        'goals_a': float(feat.get('goals_a', 0)),
        'goals_sum': float(feat.get('goals_sum', 0)),
        'xg_h': float(feat.get('xg_h', 0)),
        'xg_a': float(feat.get('xg_a', 0)),
        'xg_sum': float(feat.get('xg_sum', 0)),
        'sot_h': float(feat.get('sot_h', 0)),
        'sot_a': float(feat.get('sot_a', 0)),
        'sot_sum': float(feat.get('sot_sum', 0)),
        'cor_h': float(feat.get('cor_h', 0)),
        'cor_a': float(feat.get('cor_a', 0)),
        'cor_sum': float(feat.get('cor_sum', 0)),
        'pos_h': float(feat.get('pos_h', 0)),
        'pos_a': float(feat.get('pos_a', 0)),
        'red_h': float(feat.get('red_h', 0)),
        'red_a': float(feat.get('red_a', 0)),
        'red_sum': float(feat.get('red_sum', 0)),
    }
    
    # Minute-specific features
    features['minute_squared'] = features['minute'] ** 2
    features['minute_log'] = np.log1p(features['minute'])
    
    # Game state features
    features['goal_difference'] = abs(features['goals_h'] - features['goals_a'])
    features['is_draw'] = 1.0 if features['goals_h'] == features['goals_a'] else 0.0
    features['home_leading'] = 1.0 if features['goals_h'] > features['goals_a'] else 0.0
    features['away_leading'] = 1.0 if features['goals_a'] > features['goals_h'] else 0.0
    
    # Rate features (per minute)
    if minute > 0:
        features['goals_per_minute'] = features['goals_sum'] / minute
        features['xg_per_minute'] = features['xg_sum'] / minute
        features['sot_per_minute'] = features['sot_sum'] / minute
        features['cor_per_minute'] = features['cor_sum'] / minute
    
    # Efficiency features
    sh_total = feat.get('sh_total_h', 0) + feat.get('sh_total_a', 0)
    if sh_total > 0:
        features['shot_efficiency'] = features['sot_sum'] / sh_total
    
    if features['xg_sum'] > 0:
        features['xg_efficiency'] = features['goals_sum'] / features['xg_sum']
    
    # Momentum features
    features['recent_goals'] = features['goals_sum']  # Could enhance with time-weighted
    features['xg_momentum'] = features['xg_sum'] / max(1, minute/45)  # Normalized to 45 minutes
    
    # Match importance (simplified)
    features['late_game'] = 1.0 if minute > 75 else 0.0
    features['early_game'] = 1.0 if minute < 30 else 0.0
    
    # Fill NaN values
    for key in list(features.keys()):
        if pd.isna(features[key]):
            features[key] = 0.0
    
    return features

def analyze_feature_availability(X: pd.DataFrame):
    """Analyze which features are available in the data"""
    logger.info("Feature availability analysis:")
    
    availability = {}
    for col in X.columns:
        non_null = X[col].notna().sum()
        non_zero = (X[col] != 0).sum()
        availability[col] = {
            'non_null': non_null,
            'non_null_pct': non_null / len(X) * 100,
            'non_zero': non_zero,
            'non_zero_pct': non_zero / len(X) * 100
        }
    
    # Sort by availability
    sorted_features = sorted(availability.items(), key=lambda x: x[1]['non_null_pct'], reverse=True)
    
    logger.info("Top 20 most available features:")
    for feature, stats in sorted_features[:20]:
        logger.info(f"  {feature}: {stats['non_null_pct']:.1f}% not null, {stats['non_zero_pct']:.1f}% non-zero")
    
    return availability

def preprocess_features_robust(X: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Robust preprocessing handling sparse features"""
    logger.info("Preprocessing features...")
    
    # Separate numeric and binary features
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    
    # Create preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ])
    
    # Fit and transform
    X_processed = preprocessor.fit_transform(X)
    X_processed = pd.DataFrame(X_processed, columns=numeric_features)
    
    # Feature selection - keep only features with sufficient variance
    variances = X_processed.var()
    low_variance_features = variances[variances < 0.01].index.tolist()
    
    if low_variance_features:
        logger.info(f"Dropping {len(low_variance_features)} low variance features")
        X_processed = X_processed.drop(columns=low_variance_features)
    
    preprocessing_info = {
        'numeric_features': numeric_features,
        'low_variance_features_dropped': low_variance_features,
        'preprocessor': preprocessor,
        'feature_names': X_processed.columns.tolist()
    }
    
    logger.info(f"Processed features shape: {X_processed.shape}")
    
    return X_processed, preprocessing_info

def train_calibrated_model(X: pd.DataFrame, y: pd.Series, model_type: str = 'logistic'):
    """Train a calibrated model with proper validation"""
    logger.info(f"Training {model_type} model...")
    
    # Time-based split (chronological)
    split_idx = int(len(X) * 0.7)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    
    logger.info(f"Train size: {len(X_train)}, Validation size: {len(X_val)}")
    
    if model_type == 'logistic':
        # Logistic Regression with regularization
        param_grid = {
            'C': np.logspace(-3, 3, 7),
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'class_weight': [None, 'balanced']
        }
        
        base_model = LogisticRegression(
            max_iter=1000,
            random_state=42
        )
        
        search = RandomizedSearchCV(
            base_model,
            param_grid,
            n_iter=20,
            cv=TimeSeriesSplit(n_splits=3),
            scoring='roc_auc',
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
        
        search.fit(X_train, y_train)
        model = search.best_estimator_
        model_info = {
            'model_type': 'logistic_regression',
            'best_params': search.best_params_,
            'cv_score': float(search.best_score_)
        }
        
    elif model_type == 'random_forest':
        # Random Forest
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': [None, 'balanced', 'balanced_subsample']
        }
        
        base_model = RandomForestClassifier(
            random_state=42,
            n_jobs=-1
        )
        
        search = RandomizedSearchCV(
            base_model,
            param_grid,
            n_iter=20,
            cv=TimeSeriesSplit(n_splits=3),
            scoring='roc_auc',
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
        
        search.fit(X_train, y_train)
        model = search.best_estimator_
        model_info = {
            'model_type': 'random_forest',
            'best_params': search.best_params_,
            'cv_score': float(search.best_score_)
        }
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Calibrate the model
    calibrated_model = CalibratedClassifierCV(
        model,
        method='isotonic',
        cv='prefit'
    )
    
    # Fit calibration on validation set
    calibrated_model.fit(X_val, y_val)
    
    # Evaluate
    train_pred = calibrated_model.predict_proba(X_train)[:, 1]
    val_pred = calibrated_model.predict_proba(X_val)[:, 1]
    
    metrics = {
        'train_auc': roc_auc_score(y_train, train_pred),
        'val_auc': roc_auc_score(y_val, val_pred),
        'train_brier': brier_score_loss(y_train, train_pred),
        'val_brier': brier_score_loss(y_val, val_pred),
        'train_log_loss': log_loss(y_train, train_pred),
        'val_log_loss': log_loss(y_val, val_pred)
    }
    
    logger.info(f"Training AUC: {metrics['train_auc']:.4f}")
    logger.info(f"Validation AUC: {metrics['val_auc']:.4f}")
    logger.info(f"Validation Brier: {metrics['val_brier']:.4f}")
    
    return calibrated_model, model_info, metrics

def optimize_confidence_threshold(model, X_val: pd.DataFrame, y_val: pd.Series, 
                                target_accuracy: float = 65.0):
    """Optimize confidence threshold for target accuracy"""
    logger.info("Optimizing confidence threshold...")
    
    # Get predicted probabilities
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # Test different thresholds
    thresholds = np.linspace(0.5, 0.9, 41)  # 0.5 to 0.9 in steps of 0.01
    results = []
    
    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        
        if y_pred.sum() == 0:  # No positive predictions
            continue
            
        # Calculate metrics
        acc = accuracy_score(y_val, y_pred)
        precision = (y_pred & y_val).sum() / max(1, y_pred.sum())
        recall = (y_pred & y_val).sum() / max(1, y_val.sum())
        
        # Distance from target accuracy (we want to be as close as possible)
        accuracy_distance = abs(acc * 100 - target_accuracy)
        
        results.append({
            'threshold': thresh,
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'predictions': y_pred.sum(),
            'accuracy_distance': accuracy_distance,
            'score': precision * recall * min(1, y_pred.sum() / len(y_val) * 10)  # Balance quality and quantity
        })
    
    if not results:
        logger.warning("No valid thresholds found")
        return 0.7, {}
    
    results_df = pd.DataFrame(results)
    
    # Find threshold closest to target accuracy with reasonable predictions
    valid_results = results_df[results_df['predictions'] >= max(5, len(y_val) * 0.05)]  # At least 5 predictions or 5%
    
    if len(valid_results) == 0:
        valid_results = results_df
    
    # Choose threshold: minimize distance to target accuracy, maximize score
    valid_results['combined_score'] = 1 / (valid_results['accuracy_distance'] + 1) + valid_results['score']
    best_idx = valid_results['combined_score'].idxmax()
    best_threshold = valid_results.loc[best_idx, 'threshold']
    
    # Calculate final metrics at best threshold
    y_pred = (y_pred_proba >= best_threshold).astype(int)
    final_acc = accuracy_score(y_val, y_pred)
    final_precision = (y_pred & y_val).sum() / max(1, y_pred.sum())
    final_recall = (y_pred & y_val).sum() / max(1, y_val.sum())
    
    threshold_info = {
        'optimal_threshold': float(best_threshold),
        'optimal_threshold_pct': float(best_threshold * 100),
        'accuracy_at_threshold': float(final_acc * 100),
        'precision_at_threshold': float(final_precision * 100),
        'recall_at_threshold': float(final_recall * 100),
        'predictions_at_threshold': int(y_pred.sum()),
        'total_samples': len(y_val),
        'all_thresholds': results_df.to_dict('records')
    }
    
    logger.info(f"Optimal threshold: {best_threshold:.3f} ({best_threshold*100:.1f}%)")
    logger.info(f"Accuracy at threshold: {final_acc*100:.1f}%")
    logger.info(f"Precision at threshold: {final_precision*100:.1f}%")
    logger.info(f"Predictions made: {y_pred.sum()}/{len(y_val)} ({y_pred.sum()/len(y_val)*100:.1f}%)")
    
    return best_threshold, threshold_info

def save_model_with_calibration(model, model_info: Dict, metrics: Dict,
                               threshold_info: Dict, preprocessing_info: Dict,
                               feature_names: List[str]):
    """Save model in format compatible with main application"""
    logger.info("Saving model in application format...")
    
    # Extract weights for linear model
    weights = {}
    intercept = 0.0
    
    if hasattr(model, 'calibrated_classifiers_'):
        base_estimator = model.calibrated_classifiers_[0].base_estimator
        
        if hasattr(base_estimator, 'coef_') and hasattr(base_estimator, 'intercept_'):
            # Logistic regression
            coef = base_estimator.coef_[0]
            weights = dict(zip(feature_names, coef.tolist()))
            intercept = float(base_estimator.intercept_[0])
            
            logger.info(f"Model intercept: {intercept:.4f}")
            logger.info(f"Top 5 positive weights:")
            for feat, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True)[:5]:
                logger.info(f"  {feat}: {weight:.4f}")
            
            logger.info(f"Top 5 negative weights:")
            for feat, weight in sorted(weights.items(), key=lambda x: x[1])[:5]:
                logger.info(f"  {feat}: {weight:.4f}")
    
    # Create model blob
    model_blob = {
        'weights': weights,
        'intercept': intercept,
        'calibration': {
            'method': 'platt',  # Main app uses Platt scaling
            'a': 1.0,  # Will be updated by calibration
            'b': 0.0
        },
        'feature_names': feature_names,
        'training_info': {
            'timestamp': int(time.time()),
            'model_type': model_info.get('model_type', 'unknown'),
            'metrics': metrics,
            'threshold_optimization': threshold_info,
            'preprocessing': {
                'feature_count': len(feature_names),
                'low_variance_features_dropped': preprocessing_info.get('low_variance_features_dropped', [])
            },
            'performance_summary': {
                'recommended_threshold': threshold_info.get('optimal_threshold_pct', CURRENT_THRESHOLD),
                'expected_accuracy': threshold_info.get('accuracy_at_threshold', 0),
                'validation_auc': metrics.get('val_auc', 0)
            }
        }
    }
    
    # Save to database
    model_key = "model_latest:OU_2.5"
    model_json = json.dumps(model_blob, indent=2)
    
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO settings (key, value) 
                    VALUES (%s, %s)
                    ON CONFLICT (key) 
                    DO UPDATE SET value = EXCLUDED.value
                """, (model_key, model_json))
                
                # Also save with timestamp
                timestamp_key = f"model_v{int(time.time())}:OU_2.5"
                cur.execute("""
                    INSERT INTO settings (key, value) 
                    VALUES (%s, %s)
                    ON CONFLICT (key) 
                    DO UPDATE SET value = EXCLUDED.value
                """, (timestamp_key, model_json))
                
                conn.commit()
        
        logger.info(f"Model saved to database with key: {model_key}")
        
        # Save locally
        os.makedirs("models", exist_ok=True)
        local_file = f"models/ou25_model_{int(time.time())}.json"
        with open(local_file, 'w') as f:
            json.dump(model_blob, f, indent=2)
        
        logger.info(f"Model saved locally: {local_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return False

def generate_diagnostic_plots(model, X_val: pd.DataFrame, y_val: pd.Series,
                             threshold_info: Dict, output_dir: str = "diagnostics"):
    """Generate diagnostic plots for model evaluation"""
    os.makedirs(output_dir, exist_ok=True)
    
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # 1. Calibration curve
    prob_true, prob_pred = calibration_curve(y_val, y_pred_proba, n_bins=10)
    
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.plot(prob_pred, prob_true, 's-', label='Model')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. ROC curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    
    plt.subplot(2, 2, 2)
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # 3. Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_val, y_pred_proba)
    avg_precision = average_precision_score(y_val, y_pred_proba)
    
    plt.subplot(2, 2, 3)
    plt.plot(recall, precision, label=f'PR curve (AP = {avg_precision:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    
    # 4. Threshold analysis
    plt.subplot(2, 2, 4)
    thresholds = [t['threshold'] for t in threshold_info.get('all_thresholds', [])]
    accuracies = [t['accuracy'] * 100 for t in threshold_info.get('all_thresholds', [])]
    predictions = [t['predictions'] for t in threshold_info.get('all_thresholds', [])]
    
    if thresholds:
        plt.plot(thresholds, accuracies, 'b-', label='Accuracy (%)')
        plt.axvline(x=threshold_info.get('optimal_threshold', 0.7), color='r', 
                   linestyle='--', label=f'Optimal ({threshold_info.get("optimal_threshold", 0.7):.3f})')
        plt.axhline(y=TARGET_ACCURACY, color='g', linestyle=':', label=f'Target ({TARGET_ACCURACY}%)')
        plt.xlabel('Confidence Threshold')
        plt.ylabel('Accuracy (%)')
        plt.title('Threshold Optimization')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, f"diagnostics_{int(time.time())}.png")
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Diagnostic plots saved: {plot_file}")
    
    return plot_file

def main():
    """Main training pipeline"""
    logger.info("=" * 80)
    logger.info("Starting OU 2.5 Model Training")
    logger.info("=" * 80)
    
    # 1. Analyze current performance
    current_perf = analyze_current_performance()
    
    # 2. Load training data
    X, y, meta_df = load_training_data_with_metadata(days_back=90)  # 90 days for more recent data
    
    if len(X) < MIN_SAMPLES:
        logger.error(f"Insufficient training data: {len(X)} < {MIN_SAMPLES}")
        return
    
    logger.info(f"Training on {len(X)} samples")
    
    # 3. Preprocess features
    X_processed, preprocessing_info = preprocess_features_robust(X)
    
    # 4. Train model
    model, model_info, metrics = train_calibrated_model(X_processed, y, model_type='logistic')
    
    # 5. Optimize confidence threshold
    # Use the last 30% as validation for threshold optimization
    split_idx = int(len(X_processed) * 0.7)
    X_val_thresh = X_processed.iloc[split_idx:]
    y_val_thresh = y.iloc[split_idx:]
    
    optimal_threshold, threshold_info = optimize_confidence_threshold(
        model, X_val_thresh, y_val_thresh, target_accuracy=TARGET_ACCURACY
    )
    
    # 6. Generate diagnostic plots
    plot_file = generate_diagnostic_plots(model, X_val_thresh, y_val_thresh, threshold_info)
    
    # 7. Save model
    success = save_model_with_calibration(
        model, model_info, metrics, threshold_info,
        preprocessing_info, X_processed.columns.tolist()
    )
    
    if success:
        # 8. Generate summary report
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Model trained successfully!")
        logger.info(f"Training samples: {len(X)}")
        logger.info(f"Validation AUC: {metrics.get('val_auc', 0):.4f}")
        logger.info(f"Validation Brier: {metrics.get('val_brier', 0):.4f}")
        logger.info(f"Recommended confidence threshold: {threshold_info.get('optimal_threshold_pct', 0):.1f}%")
        logger.info(f"Expected accuracy at threshold: {threshold_info.get('accuracy_at_threshold', 0):.1f}%")
        logger.info(f"Current threshold in app: {CURRENT_THRESHOLD}%")
        
        if current_perf:
            logger.info(f"\nCurrent performance (7 days):")
            logger.info(f"  Accuracy: {current_perf.get('accuracy', 0):.1f}%")
            logger.info(f"  Tips sent: {current_perf.get('total_tips', 0)}")
            logger.info(f"  Avg confidence: {current_perf.get('avg_confidence', 0):.1f}%")
        
        logger.info("\nRecommendations:")
        if threshold_info.get('optimal_threshold_pct', 72) < CURRENT_THRESHOLD - 2:
            logger.info(f"  ⬇️  Consider lowering confidence threshold from {CURRENT_THRESHOLD}% to {threshold_info.get('optimal_threshold_pct', 0):.1f}%")
            logger.info(f"     This would increase tips from ~{current_perf.get('total_tips', 0)} to ~{threshold_info.get('predictions_at_threshold', 0)}")
        elif threshold_info.get('optimal_threshold_pct', 72) > CURRENT_THRESHOLD + 2:
            logger.info(f"  ⬆️  Consider raising confidence threshold from {CURRENT_THRESHOLD}% to {threshold_info.get('optimal_threshold_pct', 0):.1f}%")
            logger.info(f"     This would improve accuracy but reduce tip volume")
        else:
            logger.info(f"  ✅ Current threshold of {CURRENT_THRESHOLD}% is near optimal")
        
        logger.info(f"\nNext steps:")
        logger.info(f"  1. Update CONF_THRESHOLD in .env to {threshold_info.get('optimal_threshold_pct', 0):.1f}")
        logger.info(f"  2. Restart the application")
        logger.info(f"  3. Monitor performance for 24-48 hours")
        
        logger.info("\n" + "=" * 80)
        
        # Create a simple config update suggestion
        config_suggestion = f"""
# Suggested .env update based on training:
CONF_THRESHOLD={threshold_info.get('optimal_threshold_pct', 72.0):.1f}
# Previous value: CONF_THRESHOLD={CURRENT_THRESHOLD}
# Expected accuracy: {threshold_info.get('accuracy_at_threshold', 0):.1f}%
# Training date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
        
        config_file = f"models/config_suggestion_{int(time.time())}.txt"
        with open(config_file, 'w') as f:
            f.write(config_suggestion)
        
        logger.info(f"Config suggestion saved: {config_file}")
        
    else:
        logger.error("Model training failed!")
    
    logger.info("Training completed.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train OU 2.5 model")
    parser.add_argument("--days", type=int, default=90, help="Days of historical data")
    parser.add_argument("--target-accuracy", type=float, default=65.0, help="Target accuracy for threshold optimization")
    
    args = parser.parse_args()
    
    main()
