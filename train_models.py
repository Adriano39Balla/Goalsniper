#!/usr/bin/env python3
"""
train_models.py - Train and evaluate OU 2.5 models for goalsniper
Enhanced with cross-validation, feature engineering, and model persistence
"""

import os, json, time, logging, sys, pickle, warnings
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
import traceback
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats
import psycopg2
from psycopg2.extras import RealDictCursor

# ML imports
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import (
    train_test_split, 
    TimeSeriesSplit,
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV
)
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    log_loss,
    brier_score_loss,
    precision_recall_curve,
    average_precision_score,
    classification_report,
    confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Optional imports for advanced models
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost not available, skipping XGBoost models")

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print("LightGBM not available, skipping LightGBM models")

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

# Training parameters
MIN_SAMPLES = int(os.getenv("MIN_TRAINING_SAMPLES", "500"))
TRAIN_TEST_SPLIT = float(os.getenv("TRAIN_TEST_SPLIT", "0.8"))
CV_FOLDS = int(os.getenv("CV_FOLDS", "5"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
N_JOBS = int(os.getenv("N_JOBS", "-1"))

# Feature engineering
INCLUDE_DERIVED_FEATURES = os.getenv("INCLUDE_DERIVED_FEATURES", "1") == "1"
MINUTE_BINS = json.loads(os.getenv("MINUTE_BINS", "[15, 30, 45, 60, 75]"))
FEATURE_SELECTION_K = int(os.getenv("FEATURE_SELECTION_K", "20"))
DROP_CORRELATION_THRESHOLD = float(os.getenv("DROP_CORRELATION_THRESHOLD", "0.95"))

# Model training
MODEL_TYPES = os.getenv("MODEL_TYPES", "logistic,random_forest,gradient_boosting").split(",")
CALIBRATION_METHOD = os.getenv("CALIBRATION_METHOD", "isotonic")  # isotonic or sigmoid
ENSEMBLE_METHOD = os.getenv("ENSEMBLE_METHOD", "voting")  # voting, stacking, or best

# Database connection
def get_db_connection():
    """Get database connection with retry logic"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            conn = psycopg2.connect(DATABASE_URL)
            return conn
        except Exception as e:
            logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(1)

# Feature extraction from learning_features JSON
def extract_features_from_json(learning_features: Dict) -> Dict[str, float]:
    """Extract features from learning_features JSON column"""
    if not learning_features:
        return {}
    
    feat = learning_features.get("features", {})
    if not feat:
        return {}
    
    # Basic features
    features = {
        "minute": float(feat.get("minute", 0)),
        "goals_h": float(feat.get("goals_h", 0)),
        "goals_a": float(feat.get("goals_a", 0)),
        "goals_sum": float(feat.get("goals_sum", 0)),
        "xg_h": float(feat.get("xg_h", 0)),
        "xg_a": float(feat.get("xg_a", 0)),
        "xg_sum": float(feat.get("xg_sum", 0)),
        "sot_h": float(feat.get("sot_h", 0)),
        "sot_a": float(feat.get("sot_a", 0)),
        "sot_sum": float(feat.get("sot_sum", 0)),
        "sh_total_h": float(feat.get("sh_total_h", 0)),
        "sh_total_a": float(feat.get("sh_total_a", 0)),
        "cor_h": float(feat.get("cor_h", 0)),
        "cor_a": float(feat.get("cor_a", 0)),
        "cor_sum": float(feat.get("cor_sum", 0)),
        "pos_h": float(feat.get("pos_h", 0)),
        "pos_a": float(feat.get("pos_a", 0)),
        "red_h": float(feat.get("red_h", 0)),
        "red_a": float(feat.get("red_a", 0)),
        "red_sum": float(feat.get("red_sum", 0)),
    }
    
    # Derived features (if enabled)
    if INCLUDE_DERIVED_FEATURES:
        # Rate features (per minute)
        if features["minute"] > 0:
            features["goals_per_minute"] = features["goals_sum"] / features["minute"]
            features["xg_per_minute"] = features["xg_sum"] / features["minute"]
            features["sot_per_minute"] = features["sot_sum"] / features["minute"]
            features["cor_per_minute"] = features["cor_sum"] / features["minute"]
        
        # Efficiency features
        if features["sh_total_sum"] > 0:
            features["shot_efficiency"] = features["goals_sum"] / features["sh_total_sum"]
            features["sot_efficiency"] = features["sot_sum"] / features["sh_total_sum"]
        
        if features["xg_sum"] > 0:
            features["xg_efficiency"] = features["goals_sum"] / features["xg_sum"]
        
        # Momentum features
        features["goal_difference"] = abs(features["goals_h"] - features["goals_a"])
        features["xg_difference"] = abs(features["xg_h"] - features["xg_a"])
        
        # Pressure features
        features["attacking_pressure"] = (features["sot_sum"] + features["cor_sum"]) / max(1, features["minute"])
        
        # Match state features
        features["is_draw"] = 1.0 if features["goals_h"] == features["goals_a"] else 0.0
        features["home_leading"] = 1.0 if features["goals_h"] > features["goals_a"] else 0.0
        features["away_leading"] = 1.0 if features["goals_a"] > features["goals_h"] else 0.0
        
        # Minute bin features
        for i, bin_edge in enumerate(MINUTE_BINS):
            features[f"minute_gt_{bin_edge}"] = 1.0 if features["minute"] > bin_edge else 0.0
    
    return features

def load_training_data(days_back: int = 180) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Load training data from database
    
    Returns:
        X: Features DataFrame
        y: Target Series (1 for Over 2.5, 0 for Under 2.5)
        feature_names: List of feature names
    """
    logger.info(f"Loading training data from last {days_back} days")
    
    cutoff_ts = int(time.time()) - (days_back * 24 * 3600)
    
    query = """
        SELECT 
            t.learning_features,
            r.final_goals_h,
            r.final_goals_a,
            t.suggestion,
            t.minute,
            t.created_ts,
            t.match_id
        FROM tips t
        JOIN match_results r ON t.match_id = r.match_id
        WHERE t.created_ts >= %s
          AND t.market = 'Over/Under 2.5'
          AND t.sent_ok = 1
          AND t.learning_features IS NOT NULL
          AND r.final_goals_h IS NOT NULL
          AND r.final_goals_a IS NOT NULL
          AND t.suggestion IN ('Over 2.5 Goals', 'Under 2.5 Goals')
        ORDER BY t.created_ts DESC
    """
    
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (cutoff_ts,))
                rows = cur.fetchall()
        
        logger.info(f"Retrieved {len(rows)} samples from database")
        
        if len(rows) < MIN_SAMPLES:
            logger.warning(f"Insufficient samples: {len(rows)} < {MIN_SAMPLES}")
            return pd.DataFrame(), pd.Series(), []
        
        # Process data
        features_list = []
        targets = []
        metadata = []
        
        for row in rows:
            try:
                # Parse learning features
                learning_features = json.loads(row['learning_features']) if isinstance(row['learning_features'], str) else row['learning_features']
                
                # Extract features
                feat_dict = extract_features_from_json(learning_features)
                if not feat_dict:
                    continue
                
                # Determine target (1 for Over 2.5, 0 for Under 2.5)
                total_goals = (row['final_goals_h'] or 0) + (row['final_goals_a'] or 0)
                suggestion = row['suggestion']
                
                # For Over tips: target is 1 if total_goals > 2.5
                if "Over" in suggestion:
                    target = 1 if total_goals > 2.5 else 0
                # For Under tips: target is 0 if total_goals < 2.5
                else:
                    target = 0 if total_goals < 2.5 else 1
                
                # Skip if match hasn't ended or is exactly 2.5 goals (rare)
                if total_goals == 2.5:
                    continue
                
                features_list.append(feat_dict)
                targets.append(target)
                metadata.append({
                    'match_id': row['match_id'],
                    'minute': row['minute'],
                    'timestamp': row['created_ts'],
                    'total_goals': total_goals,
                    'suggestion': suggestion
                })
                
            except Exception as e:
                logger.warning(f"Error processing row: {e}")
                continue
        
        # Convert to DataFrame
        X = pd.DataFrame(features_list)
        y = pd.Series(targets, name='target')
        meta_df = pd.DataFrame(metadata)
        
        logger.info(f"Processed {len(X)} valid samples")
        logger.info(f"Class distribution: {y.value_counts().to_dict()}")
        logger.info(f"Features shape: {X.shape}")
        
        return X, y, X.columns.tolist(), meta_df
        
    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        traceback.print_exc()
        return pd.DataFrame(), pd.Series(), [], pd.DataFrame()

def preprocess_features(X: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Preprocess features: handle missing values, scale, select features
    
    Returns:
        X_processed: Processed features
        preprocessor: Fitted preprocessing objects
    """
    logger.info("Preprocessing features")
    
    # Store original columns
    original_columns = X.columns.tolist()
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    X_imputed = pd.DataFrame(X_imputed, columns=X.columns)
    
    # Remove highly correlated features
    corr_matrix = X_imputed.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > DROP_CORRELATION_THRESHOLD)]
    
    if to_drop:
        logger.info(f"Dropping {len(to_drop)} highly correlated features: {to_drop}")
        X_imputed = X_imputed.drop(columns=to_drop)
    
    # Scale features
    scaler = RobustScaler()  # Robust to outliers
    X_scaled = scaler.fit_transform(X_imputed)
    X_scaled = pd.DataFrame(X_scaled, columns=X_imputed.columns)
    
    # Feature selection
    if len(X_scaled.columns) > FEATURE_SELECTION_K:
        selector = SelectKBest(score_func=f_classif, k=min(FEATURE_SELECTION_K, len(X_scaled.columns)))
        # We'll fit this later with y during training
        logger.info(f"Will select top {FEATURE_SELECTION_K} features")
    
    preprocessor = {
        'imputer': imputer,
        'scaler': scaler,
        'dropped_features': to_drop,
        'original_columns': original_columns,
        'selected_columns': X_scaled.columns.tolist()
    }
    
    return X_scaled, preprocessor

def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series, 
                             preprocessor: Dict) -> Tuple[Any, Dict]:
    """Train and calibrate logistic regression model"""
    logger.info("Training logistic regression model")
    
    # Feature selection
    if len(X_train.columns) > FEATURE_SELECTION_K:
        selector = SelectKBest(score_func=f_classif, k=min(FEATURE_SELECTION_K, len(X_train.columns)))
        X_train_selected = selector.fit_transform(X_train, y_train)
        selected_features = X_train.columns[selector.get_support()].tolist()
        X_train = pd.DataFrame(X_train_selected, columns=selected_features)
        preprocessor['selector'] = selector
        preprocessor['selected_features'] = selected_features
    
    # Hyperparameter tuning
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'class_weight': [None, 'balanced']
    }
    
    base_model = LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=1000,
        n_jobs=N_JOBS
    )
    
    # Use randomized search for speed
    search = RandomizedSearchCV(
        base_model,
        param_grid,
        n_iter=20,
        cv=TimeSeriesSplit(n_splits=3),
        scoring='roc_auc',
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
        verbose=0
    )
    
    search.fit(X_train, y_train)
    
    logger.info(f"Best logistic regression params: {search.best_params_}")
    logger.info(f"Best CV score: {search.best_score_:.4f}")
    
    # Calibrate the model
    calibrated_model = CalibratedClassifierCV(
        search.best_estimator_,
        method=CALIBRATION_METHOD,
        cv='prefit'  # Use the already trained model
    )
    
    # For calibration, we need to fit on a holdout set
    X_cal, X_val, y_cal, y_val = train_test_split(
        X_train, y_train, test_size=0.3, random_state=RANDOM_STATE, stratify=y_train
    )
    
    search.best_estimator_.fit(X_cal, y_cal)
    calibrated_model.fit(X_val, y_val)
    
    model_info = {
        'model_type': 'logistic_regression',
        'best_params': search.best_params_,
        'cv_score': float(search.best_score_),
        'feature_importance': dict(zip(X_train.columns, search.best_estimator_.coef_[0])),
        'intercept': float(search.best_estimator_.intercept_[0]),
        'calibration_method': CALIBRATION_METHOD
    }
    
    return calibrated_model, model_info

def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series,
                       preprocessor: Dict) -> Tuple[Any, Dict]:
    """Train random forest model"""
    logger.info("Training random forest model")
    
    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'class_weight': [None, 'balanced', 'balanced_subsample']
    }
    
    base_model = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        verbose=0
    )
    
    search = RandomizedSearchCV(
        base_model,
        param_grid,
        n_iter=20,
        cv=TimeSeriesSplit(n_splits=3),
        scoring='roc_auc',
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
        verbose=0
    )
    
    search.fit(X_train, y_train)
    
    logger.info(f"Best random forest params: {search.best_params_}")
    logger.info(f"Best CV score: {search.best_score_:.4f}")
    
    # Calibrate
    calibrated_model = CalibratedClassifierCV(
        search.best_estimator_,
        method=CALIBRATION_METHOD,
        cv='prefit'
    )
    
    X_cal, X_val, y_cal, y_val = train_test_split(
        X_train, y_train, test_size=0.3, random_state=RANDOM_STATE, stratify=y_train
    )
    
    search.best_estimator_.fit(X_cal, y_cal)
    calibrated_model.fit(X_val, y_val)
    
    # Feature importance
    feature_importance = dict(zip(X_train.columns, search.best_estimator_.feature_importances_))
    
    model_info = {
        'model_type': 'random_forest',
        'best_params': search.best_params_,
        'cv_score': float(search.best_score_),
        'feature_importance': feature_importance,
        'calibration_method': CALIBRATION_METHOD
    }
    
    return calibrated_model, model_info

def train_gradient_boosting(X_train: pd.DataFrame, y_train: pd.Series,
                           preprocessor: Dict) -> Tuple[Any, Dict]:
    """Train gradient boosting model"""
    logger.info("Training gradient boosting model")
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'subsample': [0.8, 0.9, 1.0],
        'max_features': ['sqrt', 'log2', None]
    }
    
    base_model = GradientBoostingClassifier(
        random_state=RANDOM_STATE,
        verbose=0
    )
    
    search = RandomizedSearchCV(
        base_model,
        param_grid,
        n_iter=20,
        cv=TimeSeriesSplit(n_splits=3),
        scoring='roc_auc',
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
        verbose=0
    )
    
    search.fit(X_train, y_train)
    
    logger.info(f"Best gradient boosting params: {search.best_params_}")
    logger.info(f"Best CV score: {search.best_score_:.4f}")
    
    # Calibrate
    calibrated_model = CalibratedClassifierCV(
        search.best_estimator_,
        method=CALIBRATION_METHOD,
        cv='prefit'
    )
    
    X_cal, X_val, y_cal, y_val = train_test_split(
        X_train, y_train, test_size=0.3, random_state=RANDOM_STATE, stratify=y_train
    )
    
    search.best_estimator_.fit(X_cal, y_cal)
    calibrated_model.fit(X_val, y_val)
    
    feature_importance = dict(zip(X_train.columns, search.best_estimator_.feature_importances_))
    
    model_info = {
        'model_type': 'gradient_boosting',
        'best_params': search.best_params_,
        'cv_score': float(search.best_score_),
        'feature_importance': feature_importance,
        'calibration_method': CALIBRATION_METHOD
    }
    
    return calibrated_model, model_info

def train_xgboost_model(X_train: pd.DataFrame, y_train: pd.Series,
                       preprocessor: Dict) -> Tuple[Any, Dict]:
    """Train XGBoost model (if available)"""
    if not XGB_AVAILABLE:
        return None, {}
    
    logger.info("Training XGBoost model")
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2],
        'reg_alpha': [0, 0.1, 1.0],
        'reg_lambda': [1.0, 1.5, 2.0]
    }
    
    base_model = XGBClassifier(
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        verbosity=0,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    search = RandomizedSearchCV(
        base_model,
        param_grid,
        n_iter=20,
        cv=TimeSeriesSplit(n_splits=3),
        scoring='roc_auc',
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
        verbose=0
    )
    
    search.fit(X_train, y_train)
    
    logger.info(f"Best XGBoost params: {search.best_params_}")
    logger.info(f"Best CV score: {search.best_score_:.4f}")
    
    # Calibrate
    calibrated_model = CalibratedClassifierCV(
        search.best_estimator_,
        method=CALIBRATION_METHOD,
        cv='prefit'
    )
    
    X_cal, X_val, y_cal, y_val = train_test_split(
        X_train, y_train, test_size=0.3, random_state=RANDOM_STATE, stratify=y_train
    )
    
    search.best_estimator_.fit(X_cal, y_cal)
    calibrated_model.fit(X_val, y_val)
    
    feature_importance = dict(zip(X_train.columns, search.best_estimator_.feature_importances_))
    
    model_info = {
        'model_type': 'xgboost',
        'best_params': search.best_params_,
        'cv_score': float(search.best_score_),
        'feature_importance': feature_importance,
        'calibration_method': CALIBRATION_METHOD
    }
    
    return calibrated_model, model_info

def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series, 
                  model_info: Dict) -> Dict:
    """Evaluate model performance"""
    logger.info(f"Evaluating {model_info['model_type']}")
    
    # Predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Calculate metrics
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'roc_auc': float(roc_auc_score(y_test, y_pred_proba)),
        'log_loss': float(log_loss(y_test, y_pred_proba)),
        'brier_score': float(brier_score_loss(y_test, y_pred_proba)),
        'average_precision': float(average_precision_score(y_test, y_pred_proba)),
        'positive_rate': float(y_pred.mean()),
        'true_positive_rate': float(y_test.mean())
    }
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    metrics.update({
        'precision_0': float(report['0']['precision']),
        'recall_0': float(report['0']['recall']),
        'f1_0': float(report['0']['f1-score']),
        'precision_1': float(report['1']['precision']),
        'recall_1': float(report['1']['recall']),
        'f1_1': float(report['1']['f1-score']),
        'macro_avg_f1': float(report['macro avg']['f1-score']),
        'weighted_avg_f1': float(report['weighted avg']['f1-score'])
    })
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics.update({
        'tn': int(cm[0, 0]),
        'fp': int(cm[0, 1]),
        'fn': int(cm[1, 0]),
        'tp': int(cm[1, 1])
    })
    
    # Calibration curve
    prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
    metrics['calibration_curve'] = {
        'prob_true': prob_true.tolist(),
        'prob_pred': prob_pred.tolist()
    }
    
    logger.info(f"Test accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Test ROC AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"Test Brier score: {metrics['brier_score']:.4f}")
    
    return metrics

def save_model_to_db(model: Any, model_info: Dict, metrics: Dict, 
                    preprocessor: Dict, feature_names: List[str]):
    """Save trained model to database settings table"""
    logger.info(f"Saving {model_info['model_type']} model to database")
    
    # Prepare model blob
    model_blob = {
        'model_type': model_info['model_type'],
        'weights': {},
        'intercept': 0.0,
        'calibration': {
            'method': CALIBRATION_METHOD,
            'a': 1.0,
            'b': 0.0
        },
        'feature_names': feature_names,
        'preprocessor': preprocessor,
        'training_info': {
            'timestamp': int(time.time()),
            'metrics': metrics,
            'model_info': {k: v for k, v in model_info.items() if k != 'feature_importance'},
            'feature_importance': model_info.get('feature_importance', {}),
            'training_samples': metrics.get('training_samples', 0)
        }
    }
    
    # Extract weights for linear models
    if model_info['model_type'] == 'logistic_regression':
        # Get the base estimator from the calibrated model
        base_estimator = model.calibrated_classifiers_[0].base_estimator
        if hasattr(base_estimator, 'coef_'):
            weights = base_estimator.coef_[0]
            model_blob['weights'] = dict(zip(feature_names, weights.tolist()))
            model_blob['intercept'] = float(base_estimator.intercept_[0])
    
    # For tree-based models, store feature importance as weights
    elif 'feature_importance' in model_info:
        model_blob['weights'] = model_info['feature_importance']
    
    # Store calibration parameters if available
    if hasattr(model, 'calibrated_classifiers_'):
        calibrated_clf = model.calibrated_classifiers_[0]
        if hasattr(calibrated_clf, 'calibrators_'):
            # This is simplified - actual calibration parameters depend on implementation
            pass
    
    # Save to database
    model_key = f"model_latest:OU_2.5"
    model_json = json.dumps(model_blob, indent=2)
    
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Save as latest model
                cur.execute("""
                    INSERT INTO settings (key, value) 
                    VALUES (%s, %s)
                    ON CONFLICT (key) 
                    DO UPDATE SET value = EXCLUDED.value
                """, (model_key, model_json))
                
                # Also save with timestamp for versioning
                timestamp_key = f"model_v{int(time.time())}:OU_2.5"
                cur.execute("""
                    INSERT INTO settings (key, value) 
                    VALUES (%s, %s)
                    ON CONFLICT (key) 
                    DO UPDATE SET value = EXCLUDED.value
                """, (timestamp_key, model_json))
                
                conn.commit()
        
        logger.info(f"Model saved to database with key: {model_key}")
        
        # Also save locally for backup
        local_filename = f"models/ou25_model_{int(time.time())}.json"
        os.makedirs("models", exist_ok=True)
        with open(local_filename, 'w') as f:
            json.dump(model_blob, f, indent=2)
        logger.info(f"Model saved locally: {local_filename}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving model to database: {e}")
        return False

def create_ensemble_model(models: List[Tuple[Any, Dict, Dict]]) -> Tuple[Any, Dict]:
    """Create ensemble model from multiple trained models"""
    logger.info("Creating ensemble model")
    
    # Simple voting ensemble (average probabilities)
    class VotingEnsemble:
        def __init__(self, models):
            self.models = models
        
        def predict_proba(self, X):
            probas = []
            for model, _, _ in self.models:
                proba = model.predict_proba(X)[:, 1]
                probas.append(proba)
            
            avg_proba = np.mean(probas, axis=0)
            return np.column_stack([1 - avg_proba, avg_proba])
        
        def predict(self, X):
            proba = self.predict_proba(X)[:, 1]
            return (proba >= 0.5).astype(int)
    
    ensemble_model = VotingEnsemble(models)
    
    ensemble_info = {
        'model_type': 'ensemble_voting',
        'component_models': [info['model_type'] for _, info, _ in models],
        'ensemble_method': 'average_probability'
    }
    
    return ensemble_model, ensemble_info

def analyze_feature_importance(models: List[Tuple[Any, Dict, Dict]], 
                             feature_names: List[str]) -> pd.DataFrame:
    """Analyze and compare feature importance across models"""
    logger.info("Analyzing feature importance")
    
    importance_data = []
    
    for model, model_info, _ in models:
        model_type = model_info['model_type']
        
        if 'feature_importance' in model_info:
            for feature, importance in model_info['feature_importance'].items():
                importance_data.append({
                    'model_type': model_type,
                    'feature': feature,
                    'importance': importance
                })
    
    if importance_data:
        df = pd.DataFrame(importance_data)
        
        # Create summary
        summary = df.groupby('feature')['importance'].agg(['mean', 'std', 'count']).round(4)
        summary = summary.sort_values('mean', ascending=False)
        
        logger.info("Top 10 features by average importance:")
        for idx, row in summary.head(10).iterrows():
            logger.info(f"  {idx}: {row['mean']:.4f} ± {row['std']:.4f}")
        
        return summary
    
    return pd.DataFrame()

def generate_training_report(models: List[Tuple[Any, Dict, Dict]], 
                           overall_metrics: Dict) -> str:
    """Generate comprehensive training report"""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("MODEL TRAINING REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().isoformat()}")
    report_lines.append(f"Total models trained: {len(models)}")
    report_lines.append("")
    
    # Overall metrics
    report_lines.append("OVERALL PERFORMANCE:")
    report_lines.append("-" * 40)
    for key, value in overall_metrics.items():
        if isinstance(value, float):
            report_lines.append(f"{key}: {value:.4f}")
        else:
            report_lines.append(f"{key}: {value}")
    report_lines.append("")
    
    # Individual model performance
    report_lines.append("INDIVIDUAL MODEL PERFORMANCE:")
    report_lines.append("-" * 40)
    
    for model, model_info, metrics in models:
        report_lines.append(f"\n{model_info['model_type'].upper()}:")
        report_lines.append(f"  ROC AUC: {metrics.get('roc_auc', 0):.4f}")
        report_lines.append(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
        report_lines.append(f"  Brier Score: {metrics.get('brier_score', 0):.4f}")
        report_lines.append(f"  Log Loss: {metrics.get('log_loss', 0):.4f}")
    
    # Recommendations
    report_lines.append("\nRECOMMENDATIONS:")
    report_lines.append("-" * 40)
    
    # Find best model
    best_model = max(models, key=lambda x: x[2].get('roc_auc', 0))
    best_type = best_model[1]['model_type']
    best_auc = best_model[2].get('roc_auc', 0)
    
    report_lines.append(f"Best model: {best_type} (ROC AUC: {best_auc:.4f})")
    
    if best_auc < 0.6:
        report_lines.append("⚠️  Model performance is poor. Consider:")
        report_lines.append("  - Collecting more training data")
        report_lines.append("  - Improving feature engineering")
        report_lines.append("  - Checking for data quality issues")
    elif best_auc < 0.7:
        report_lines.append("⚠️  Model performance is moderate. Consider:")
        report_lines.append("  - Adding more features")
        report_lines.append("  - Trying different model architectures")
        report_lines.append("  - Increasing training data")
    elif best_auc < 0.8:
        report_lines.append("✅  Model performance is good.")
    else:
        report_lines.append("🎯  Model performance is excellent!")
    
    report_lines.append("\n" + "=" * 80)
    
    report = "\n".join(report_lines)
    
    # Save report to file
    os.makedirs("reports", exist_ok=True)
    report_file = f"reports/training_report_{int(time.time())}.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    logger.info(f"Training report saved to: {report_file}")
    
    return report

def main():
    """Main training function"""
    logger.info("Starting model training pipeline")
    
    # Load data
    X, y, feature_names, meta_df = load_training_data(days_back=180)
    
    if len(X) == 0:
        logger.error("No training data available")
        return
    
    logger.info(f"Training on {len(X)} samples with {len(feature_names)} features")
    
    # Split data (time-based split for time series)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=1-TRAIN_TEST_SPLIT, 
        random_state=RANDOM_STATE,
        shuffle=False  # Important for time series
    )
    
    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    # Preprocess features
    X_train_processed, preprocessor = preprocess_features(X_train)
    X_test_processed, _ = preprocess_features(X_test)
    
    # Ensure same columns in test set
    X_test_processed = X_test_processed[preprocessor['selected_columns']]
    
    # Train models
    trained_models = []
    
    for model_type in MODEL_TYPES:
        model_type = model_type.strip()
        
        try:
            if model_type == 'logistic':
                model, model_info = train_logistic_regression(
                    X_train_processed, y_train, preprocessor
                )
            elif model_type == 'random_forest':
                model, model_info = train_random_forest(
                    X_train_processed, y_train, preprocessor
                )
            elif model_type == 'gradient_boosting':
                model, model_info = train_gradient_boosting(
                    X_train_processed, y_train, preprocessor
                )
            elif model_type == 'xgboost' and XGB_AVAILABLE:
                model, model_info = train_xgboost_model(
                    X_train_processed, y_train, preprocessor
                )
            else:
                logger.warning(f"Unknown model type: {model_type}")
                continue
            
            if model is not None:
                # Evaluate
                metrics = evaluate_model(model, X_test_processed, y_test, model_info)
                metrics['training_samples'] = len(X_train)
                
                trained_models.append((model, model_info, metrics))
                
                # Save to database
                save_model_to_db(
                    model, model_info, metrics, preprocessor, 
                    X_train_processed.columns.tolist()
                )
                
        except Exception as e:
            logger.error(f"Error training {model_type}: {e}")
            traceback.print_exc()
    
    if not trained_models:
        logger.error("No models were successfully trained")
        return
    
    # Create ensemble model
    if len(trained_models) > 1 and ENSEMBLE_METHOD != 'best':
        ensemble_model, ensemble_info = create_ensemble_model(trained_models)
        ensemble_metrics = evaluate_model(
            ensemble_model, X_test_processed, y_test, ensemble_info
        )
        ensemble_metrics['training_samples'] = len(X_train)
        
        # Save ensemble model
        save_model_to_db(
            ensemble_model, ensemble_info, ensemble_metrics, preprocessor,
            X_train_processed.columns.tolist()
        )
        
        trained_models.append((ensemble_model, ensemble_info, ensemble_metrics))
    
    # Feature importance analysis
    feature_importance_df = analyze_feature_importance(trained_models, feature_names)
    
    # Overall metrics
    overall_metrics = {
        'total_samples': len(X),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'class_balance_train': {
            'class_0': int((y_train == 0).sum()),
            'class_1': int((y_train == 1).sum())
        },
        'class_balance_test': {
            'class_0': int((y_test == 0).sum()),
            'class_1': int((y_test == 1).sum())
        },
        'best_model_auc': max(m[2].get('roc_auc', 0) for m in trained_models),
        'feature_count': len(feature_names),
        'selected_feature_count': len(preprocessor['selected_columns'])
    }
    
    # Generate report
    report = generate_training_report(trained_models, overall_metrics)
    logger.info("\n" + report)
    
    logger.info("Model training completed successfully")

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Train OU 2.5 models")
    parser.add_argument("--days-back", type=int, default=180,
                       help="Number of days of historical data to use")
    parser.add_argument("--model-types", type=str,
                       default="logistic,random_forest,gradient_boosting",
                       help="Comma-separated list of model types to train")
    parser.add_argument("--min-samples", type=int, default=500,
                       help="Minimum samples required for training")
    parser.add_argument("--test-size", type=float, default=0.2,
                       help="Test set size ratio")
    
    args = parser.parse_args()
    
    # Update parameters from command line
    if args.days_back:
        os.environ["DAYS_BACK"] = str(args.days_back)
    if args.model_types:
        os.environ["MODEL_TYPES"] = args.model_types
    if args.min_samples:
        os.environ["MIN_TRAINING_SAMPLES"] = str(args.min_samples)
    if args.test_size:
        os.environ["TRAIN_TEST_SPLIT"] = str(1 - args.test_size)
    
    main()
