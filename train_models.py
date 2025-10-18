# train_models.py - Model training module for goalsniper
import os
import json
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
log = logging.getLogger("trainer")

# Database connection (shared with main.py)
DATABASE_URL = os.getenv("DATABASE_URL")

def db_execute(query: str, params: tuple = ()) -> list:
    """
    Execute query using main.py's connection pool
    This avoids creating a separate connection pool
    """
    try:
        # Import from main to use the same connection pool
        from main import db_conn
        
        with db_conn() as c:
            cursor = c.execute(query, params)
            if query.strip().upper().startswith('SELECT'):
                return cursor.fetchall()
            return []
    except ImportError:
        # Fallback for standalone execution
        log.warning("Could not import from main, using direct connection")
        import psycopg2
        conn = None
        try:
            conn = psycopg2.connect(DATABASE_URL)
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(query, params)
                if query.strip().upper().startswith('SELECT'):
                    return cur.fetchall()
            return []
        except Exception as e:
            log.error("DB query failed: %s - %s", query, e)
            raise
        finally:
            if conn:
                conn.close()
    except Exception as e:
        log.error("DB query failed: %s - %s", query, e)
        raise

# Feature configuration (keep your existing feature sets)
FEATURE_SETS = {
    "BTTS": [
        "minute", "goals_h", "goals_a", "goals_sum", "goals_diff",
        "xg_h", "xg_a", "xg_sum", "xg_diff",
        "sot_h", "sot_a", "sot_sum",
        "cor_h", "cor_a", "cor_sum",
        "pos_h", "pos_a", "pos_diff",
        "red_h", "red_a", "red_sum",
        "yellow_h", "yellow_a"
    ],
    "OU_2.5": [
        "minute", "goals_h", "goals_a", "goals_sum", "goals_diff",
        "xg_h", "xg_a", "xg_sum", "xg_diff",
        "sot_h", "sot_a", "sot_sum",
        "cor_h", "cor_a", "cor_sum",
        "pos_h", "pos_a", "pos_diff"
    ],
    "OU_3.5": [
        "minute", "goals_h", "goals_a", "goals_sum", "goals_diff",
        "xg_h", "xg_a", "xg_sum", "xg_diff",
        "sot_h", "sot_a", "sot_sum",
        "cor_h", "cor_a", "cor_sum"
    ],
    "WLD_HOME": [
        "minute", "goals_h", "goals_a", "goals_diff",
        "xg_h", "xg_a", "xg_diff",
        "sot_h", "sot_a",
        "cor_h", "cor_a",
        "pos_h", "pos_a", "pos_diff",
        "red_h", "red_a"
    ],
    "WLD_AWAY": [
        "minute", "goals_h", "goals_a", "goals_diff",
        "xg_h", "xg_a", "xg_diff",
        "sot_h", "sot_a",
        "cor_h", "cor_a",
        "pos_h", "pos_a", "pos_diff",
        "red_h", "red_a"
    ],
    "WLD_DRAW": [
        "minute", "goals_h", "goals_a", "goals_diff",
        "xg_h", "xg_a", "xg_diff",
        "sot_h", "sot_a",
        "cor_h", "cor_a",
        "pos_h", "pos_a", "pos_diff"
    ]
}

# Prematch feature sets
PREMATCH_FEATURE_SETS = {
    "PRE_BTTS": ["avg_goals_h", "avg_goals_a", "avg_goals_h2h", "rest_days_h", "rest_days_a"],
    "PRE_OU_2.5": ["avg_goals_h", "avg_goals_a", "avg_goals_h2h", "rest_days_h", "rest_days_a"],
    "PRE_OU_3.5": ["avg_goals_h", "avg_goals_a", "avg_goals_h2h", "rest_days_h", "rest_days_a"],
    "PRE_WLD_HOME": ["avg_goals_h", "avg_goals_a", "avg_goals_h2h", "rest_days_h", "rest_days_a"],
    "PRE_WLD_AWAY": ["avg_goals_h", "avg_goals_a", "avg_goals_h2h", "rest_days_h", "rest_days_a"]
}

def test_integration():
    """Test if train_models can connect to the database"""
    try:
        result = db_execute("SELECT 1 as test")
        log.info("✅ train_models database connection successful")
        return True
    except Exception as e:
        log.error("❌ train_models database connection failed: %s", e)
        return False

def load_training_data(days: int = 30) -> pd.DataFrame:
    """
    Load training data from snapshots and results
    """
    if not test_integration():
        return pd.DataFrame()
        
    cutoff = int((datetime.now() - timedelta(days=days)).timestamp())
    
    query = """
    SELECT 
        ts.payload as snapshot,
        mr.final_goals_h,
        mr.final_goals_a,
        mr.btts_yes
    FROM tip_snapshots ts
    JOIN match_results mr ON ts.match_id = mr.match_id
    WHERE ts.created_ts >= %s
    AND ts.payload IS NOT NULL
    AND ts.payload != 'null'
    AND mr.final_goals_h IS NOT NULL
    AND mr.final_goals_a IS NOT NULL
    ORDER BY ts.created_ts DESC
    """
    
    try:
        rows = db_execute(query, (cutoff,))
        data = []
        
        for snapshot, goals_h, goals_a, btts_yes in rows:
            try:
                snap_data = json.loads(snapshot)
                stat = snap_data.get('stat', {})
                
                # Extract features
                features = {
                    'minute': snap_data.get('minute', 0),
                    'goals_h': goals_h,
                    'goals_a': goals_a,
                    'goals_sum': goals_h + goals_a,
                    'goals_diff': goals_h - goals_a,
                    'xg_h': stat.get('xg_h', 0),
                    'xg_a': stat.get('xg_a', 0),
                    'xg_sum': stat.get('xg_sum', 0),
                    'xg_diff': stat.get('xg_diff', 0),
                    'sot_h': stat.get('sot_h', 0),
                    'sot_a': stat.get('sot_a', 0),
                    'sot_sum': stat.get('sot_sum', 0),
                    'cor_h': stat.get('cor_h', 0),
                    'cor_a': stat.get('cor_a', 0),
                    'cor_sum': stat.get('cor_sum', 0),
                    'pos_h': stat.get('pos_h', 0),
                    'pos_a': stat.get('pos_a', 0),
                    'pos_diff': stat.get('pos_diff', 0),
                    'red_h': stat.get('red_h', 0),
                    'red_a': stat.get('red_a', 0),
                    'red_sum': stat.get('red_sum', 0),
                    'yellow_h': stat.get('yellow_h', 0),
                    'yellow_a': stat.get('yellow_a', 0),
                    'yellow_sum': stat.get('yellow_sum', 0),
                    'btts_yes': btts_yes,
                    'total_goals': goals_h + goals_a
                }
                
                data.append(features)
            except json.JSONDecodeError:
                continue
                
        return pd.DataFrame(data)
    except Exception as e:
        log.error("Failed to load training data: %s", e)
        return pd.DataFrame()

def load_prematch_training_data(days: int = 90) -> pd.DataFrame:
    """
    Load prematch training data
    """
    if not test_integration():
        return pd.DataFrame()
        
    cutoff = int((datetime.now() - timedelta(days=days)).timestamp())
    
    query = """
    SELECT 
        ps.payload as snapshot,
        mr.final_goals_h,
        mr.final_goals_a,
        mr.btts_yes
    FROM prematch_snapshots ps
    JOIN match_results mr ON ps.match_id = mr.match_id
    WHERE ps.created_ts >= %s
    AND ps.payload IS NOT NULL
    AND ps.payload != 'null'
    AND mr.final_goals_h IS NOT NULL
    AND mr.final_goals_a IS NOT NULL
    """
    
    try:
        rows = db_execute(query, (cutoff,))
        data = []
        
        for snapshot, goals_h, goals_a, btts_yes in rows:
            try:
                snap_data = json.loads(snapshot)
                feat = snap_data.get('feat', {})
                
                features = {
                    'avg_goals_h': feat.get('avg_goals_h', 0),
                    'avg_goals_a': feat.get('avg_goals_a', 0),
                    'avg_goals_h2h': feat.get('avg_goals_h2h', 0),
                    'rest_days_h': feat.get('rest_days_h', 0),
                    'rest_days_a': feat.get('rest_days_a', 0),
                    'final_goals_h': goals_h,
                    'final_goals_a': goals_a,
                    'btts_yes': btts_yes,
                    'total_goals': goals_h + goals_a
                }
                
                data.append(features)
            except json.JSONDecodeError:
                continue
                
        return pd.DataFrame(data)
    except Exception as e:
        log.error("Failed to load prematch training data: %s", e)
        return pd.DataFrame()

def create_labels(df: pd.DataFrame, target_type: str) -> np.ndarray:
    """
    Create labels for different prediction targets
    """
    if target_type == "BTTS_YES":
        return (df['btts_yes'] == 1).astype(int).values
    elif target_type == "OU_2.5":
        return (df['total_goals'] > 2.5).astype(int).values
    elif target_type == "OU_3.5":
        return (df['total_goals'] > 3.5).astype(int).values
    elif target_type == "WLD_HOME":
        return (df['final_goals_h'] > df['final_goals_a']).astype(int).values
    elif target_type == "WLD_AWAY":
        return (df['final_goals_a'] > df['final_goals_h']).astype(int).values
    elif target_type == "WLD_DRAW":
        return (df['final_goals_h'] == df['final_goals_a']).astype(int).values
    else:
        raise ValueError(f"Unknown target type: {target_type}")

def train_logistic_model(X: np.ndarray, y: np.ndarray, model_name: str) -> Dict[str, Any]:
    """
    Train a logistic regression model with calibration
    """
    if len(X) < 50:
        log.warning("[TRAIN] Insufficient data for %s: %d samples", model_name, len(X))
        return None
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    if len(np.unique(y_train)) < 2:
        log.warning("[TRAIN] Only one class in training data for %s", model_name)
        return None
    
    try:
        # Train model
        model = LogisticRegression(
            penalty='l2',
            C=1.0,
            random_state=42,
            max_iter=1000,
            class_weight='balanced'
        )
        
        # Calibrate
        calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=3)
        calibrated_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = calibrated_model.predict(X_test)
        y_prob = calibrated_model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5
        
        log.info("[TRAIN] %s - Accuracy: %.3f, Precision: %.3f, AUC: %.3f", 
                model_name, accuracy, precision, auc)
        
        # Extract model parameters for our custom format
        base_model = calibrated_model.calibrated_classifiers_[0].base_estimator
        weights = dict(zip([f"f_{i}" for i in range(X.shape[1])], base_model.coef_[0]))
        intercept = float(base_model.intercept_[0])
        
        # Get calibration parameters
        # For simplicity, we'll use a sigmoid calibration
        # In production, you might want to extract the actual isotonic calibration
        calibration = {
            "method": "sigmoid",
            "a": 1.0,
            "b": 0.0
        }
        
        return {
            "weights": weights,
            "intercept": intercept,
            "calibration": calibration,
            "performance": {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "auc": float(auc),
                "n_samples": len(X)
            },
            "feature_names": [f"f_{i}" for i in range(X.shape[1])]
        }
        
    except Exception as e:
        log.error("[TRAIN] Failed to train %s: %s", model_name, e)
        return None

def save_model_to_db(model_name: str, model_data: Dict[str, Any]):
    """
    Save trained model to database settings table
    """
    try:
        # Convert to JSON-serializable format
        serializable_data = {
            "weights": model_data["weights"],
            "intercept": model_data["intercept"],
            "calibration": model_data["calibration"],
            "trained_at": datetime.now().isoformat(),
            "performance": model_data.get("performance", {})
        }
        
        model_json = json.dumps(serializable_data, ensure_ascii=False)
        
        # Save to database
        query = """
        INSERT INTO settings (key, value) 
        VALUES (%s, %s)
        ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
        """
        db_execute(query, (model_name, model_json))
        
        log.info("[TRAIN] Saved model %s to database", model_name)
        return True
        
    except Exception as e:
        log.error("[TRAIN] Failed to save model %s: %s", model_name, e)
        return False

def get_existing_model_performance(model_name: str) -> Optional[float]:
    """
    Get performance of existing model from database
    """
    try:
        query = "SELECT value FROM settings WHERE key = %s"
        rows = db_execute(query, (model_name,))
        
        if rows:
            model_data = json.loads(rows[0][0])
            return model_data.get("performance", {}).get("accuracy", 0)
        return None
    except Exception:
        return None

def should_retrain_model(model_name: str, new_accuracy: float, min_improvement: float = 0.02) -> bool:
    """
    Decide if model should be retrained based on performance improvement
    """
    existing_acc = get_existing_model_performance(model_name)
    
    if existing_acc is None:
        log.info("[TRAIN] No existing model found for %s, will train", model_name)
        return True
    
    improvement = new_accuracy - existing_acc
    if improvement >= min_improvement:
        log.info("[TRAIN] %s improvement: %.3f -> %.3f (Δ=%.3f), retraining", 
                model_name, existing_acc, new_accuracy, improvement)
        return True
    else:
        log.info("[TRAIN] %s improvement insufficient: %.3f -> %.3f (Δ=%.3f), skipping", 
                model_name, existing_acc, new_accuracy, improvement)
        return False

def train_inplay_models(df: pd.DataFrame, retrain_all: bool = False) -> Dict[str, bool]:
    """
    Train all in-play models
    """
    results = {}
    
    # Filter data for meaningful training (minute > 20, at least some stats)
    train_df = df[
        (df['minute'] > 20) & 
        (df['xg_sum'] > 0) & 
        (df['sot_sum'] > 0)
    ].copy()
    
    if len(train_df) < 100:
        log.warning("[TRAIN] Insufficient in-play training data: %d samples", len(train_df))
        return {model: False for model in FEATURE_SETS.keys()}
    
    for model_name, features in FEATURE_SETS.items():
        try:
            log.info("[TRAIN] Training %s with %d samples", model_name, len(train_df))
            
            # Prepare features and labels
            feature_cols = [f for f in features if f in train_df.columns]
            X = train_df[feature_cols].fillna(0).values
            y = create_labels(train_df, model_name)
            
            # Check class balance
            if len(np.unique(y)) < 2:
                log.warning("[TRAIN] Skipping %s - insufficient class variety", model_name)
                results[model_name] = False
                continue
            
            # Train model
            model_data = train_logistic_model(X, y, model_name)
            
            if model_data and model_data["performance"]["accuracy"] > 0.5:
                # Check if we should retrain
                if retrain_all or should_retrain_model(model_name, model_data["performance"]["accuracy"]):
                    if save_model_to_db(model_name, model_data):
                        results[model_name] = True
                    else:
                        results[model_name] = False
                else:
                    results[model_name] = False
            else:
                log.warning("[TRAIN] %s training failed or accuracy too low", model_name)
                results[model_name] = False
                
        except Exception as e:
            log.error("[TRAIN] Error training %s: %s", model_name, e)
            results[model_name] = False
    
    return results

def train_prematch_models(df: pd.DataFrame, retrain_all: bool = False) -> Dict[str, bool]:
    """
    Train all prematch models
    """
    results = {}
    
    if len(df) < 50:
        log.warning("[TRAIN] Insufficient prematch training data: %d samples", len(df))
        return {f"PRE_{model}": False for model in PREMATCH_FEATURE_SETS.keys()}
    
    for base_model_name, features in PREMATCH_FEATURE_SETS.items():
        model_name = f"PRE_{base_model_name}"
        try:
            log.info("[TRAIN] Training %s with %d samples", model_name, len(df))
            
            # Prepare features and labels
            feature_cols = [f for f in features if f in df.columns]
            X = df[feature_cols].fillna(0).values
            y = create_labels(df, base_model_name)
            
            # Check class balance
            if len(np.unique(y)) < 2:
                log.warning("[TRAIN] Skipping %s - insufficient class variety", model_name)
                results[model_name] = False
                continue
            
            # Train model
            model_data = train_logistic_model(X, y, model_name)
            
            if model_data and model_data["performance"]["accuracy"] > 0.5:
                # Check if we should retrain
                if retrain_all or should_retrain_model(model_name, model_data["performance"]["accuracy"]):
                    if save_model_to_db(model_name, model_data):
                        results[model_name] = True
                    else:
                        results[model_name] = False
                else:
                    results[model_name] = False
            else:
                log.warning("[TRAIN] %s training failed or accuracy too low", model_name)
                results[model_name] = False
                
        except Exception as e:
            log.error("[TRAIN] Error training %s: %s", model_name, e)
            results[model_name] = False
    
    return results

def train_models(retrain_all: bool = False, days_back: int = 30) -> Dict[str, Any]:
    """
    Main training function - called from main.py
    Returns dictionary with training results
    """
    start_time = time.time()
    log.info("[TRAIN] Starting model training (retrain_all=%s, days=%d)", retrain_all, days_back)
    
    try:
        # Check database connection
        if not test_integration():
            return {
                "ok": False,
                "reason": "Database connection failed",
                "trained": {},
                "duration": int(time.time() - start_time)
            }
        
        # Load training data
        log.info("[TRAIN] Loading in-play training data...")
        inplay_df = load_training_data(days_back)
        log.info("[TRAIN] Loaded %d in-play samples", len(inplay_df))
        
        log.info("[TRAIN] Loading prematch training data...")
        prematch_df = load_prematch_training_data(max(days_back * 3, 90))
        log.info("[TRAIN] Loaded %d prematch samples", len(prematch_df))
        
        # Train models
        inplay_results = {}
        prematch_results = {}
        
        if len(inplay_df) >= 50:
            log.info("[TRAIN] Training in-play models...")
            inplay_results = train_inplay_models(inplay_df, retrain_all)
        else:
            log.warning("[TRAIN] Skipping in-play models - insufficient data")
            inplay_results = {model: False for model in FEATURE_SETS.keys()}
        
        if len(prematch_df) >= 30:
            log.info("[TRAIN] Training prematch models...")
            prematch_results = train_prematch_models(prematch_df, retrain_all)
        else:
            log.warning("[TRAIN] Skipping prematch models - insufficient data")
            prematch_results = {f"PRE_{model}": False for model in PREMATCH_FEATURE_SETS.keys()}
        
        # Combine results
        all_results = {**inplay_results, **prematch_results}
        trained_count = sum(all_results.values())
        
        duration = time.time() - start_time
        
        result = {
            "ok": True,
            "trained": all_results,
            "duration": int(duration),
            "summary": {
                "inplay_samples": len(inplay_df),
                "prematch_samples": len(prematch_df),
                "models_trained": trained_count,
                "models_skipped": len(all_results) - trained_count,
                "total_time": int(duration)
            },
            "message": f"Training completed: {trained_count} models updated"
        }
        
        log.info("[TRAIN] Training completed in %d seconds. %d models trained.", 
                duration, trained_count)
        
        return result
        
    except Exception as e:
        log.exception("[TRAIN] Training failed: %s", e)
        return {
            "ok": False,
            "reason": str(e),
            "trained": {},
            "duration": int(time.time() - start_time)
        }

def cleanup_old_data(days_to_keep: int = 90):
    """
    Clean up old training data to prevent database bloat
    """
    try:
        cutoff = int((datetime.now() - timedelta(days=days_to_keep)).timestamp())
        
        # Delete old snapshots
        queries = [
            "DELETE FROM tip_snapshots WHERE created_ts < %s",
            "DELETE FROM prematch_snapshots WHERE created_ts < %s",
            "DELETE FROM tips WHERE created_ts < %s AND suggestion = 'HARVEST'"
        ]
        
        total_deleted = 0
        for query in queries:
            result = db_execute(query, (cutoff,))
            # Note: result doesn't return rowcount in our simple db_execute
            # In production, you might want to modify db_execute to return affected rows
        
        log.info("[TRAIN] Cleaned up data older than %d days", days_to_keep)
        return True
        
    except Exception as e:
        log.error("[TRAIN] Cleanup failed: %s", e)
        return False

if __name__ == "__main__":
    """
    Standalone execution for Railway trainer service
    """
    log.info("[TRAIN] Starting standalone training session")
    
    # Set environment
    os.environ.setdefault("RUN_SCHEDULER", "0")
    
    # Run training
    result = train_models(retrain_all=False, days_back=30)
    
    # Cleanup old data (weekly)
    if datetime.now().weekday() == 0:  # Monday
        cleanup_old_data(90)
    
    # Exit with appropriate code
    if result.get("ok"):
        trained_count = sum(result.get("trained", {}).values())
        log.info("[TRAIN] Standalone training successful. %d models trained.", trained_count)
        exit(0)
    else:
        log.error("[TRAIN] Standalone training failed: %s", result.get("reason", "Unknown error"))
        exit(1)
