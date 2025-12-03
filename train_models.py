import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from pathlib import Path
import sys

# Configure logging first
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import xgboost as xgb

# Import feature columns from config
try:
    from config import FEATURE_COLUMNS
except ImportError:
    # Define fallback feature columns
    FEATURE_COLUMNS = [
        'home_team_rating', 'away_team_rating', 'home_score', 'away_score',
        'goal_difference', 'total_goals', 'minute', 'time_ratio',
        'league_rank', 'hour_of_day', 'day_of_week', 'month',
        'momentum', 'scoring_pressure', 'home_possession', 'away_possession',
        'home_shots_on_goal', 'away_shots_on_goal', 'implied_prob_home',
        'implied_prob_draw', 'implied_prob_away'
    ]

def create_training_data(n_samples=10000):
    """Create synthetic training data with consistent features"""
    np.random.seed(42)
    
    # Create feature matrix with exact feature columns
    X = pd.DataFrame(np.random.randn(n_samples, len(FEATURE_COLUMNS)), columns=FEATURE_COLUMNS)
    
    # Generate realistic targets based on feature relationships
    # Home win prediction
    y_home_win = (
        (X['home_team_rating'] * 0.3 + 
         X['away_team_rating'] * -0.2 + 
         X['home_possession'] * 0.15 + 
         X['home_shots_on_goal'] * 0.25 + 
         X['home_score'] * 0.1) > 0.1
    ).astype(int)
    
    # Over 2.5 goals prediction
    y_over_25 = (
        (X['scoring_pressure'] * 0.4 + 
         X['total_goals'] * 0.3 + 
         X['momentum'].abs() * 0.2 + 
         X['time_ratio'] * 0.1) > 0.2
    ).astype(int)
    
    # BTTS prediction
    y_btts = (
        (X['home_shots_on_goal'] * 0.25 + 
         X['away_shots_on_goal'] * 0.25 + 
         X['goal_difference'].abs() * -0.2 + 
         X['total_goals'] * 0.2 + 
         X['time_ratio'] * 0.1) > 0.1
    ).astype(int)
    
    logger.info(f"Created training data with {n_samples} samples")
    logger.info(f"Home win distribution: {pd.Series(y_home_win).value_counts(normalize=True).to_dict()}")
    logger.info(f"Over 2.5 distribution: {pd.Series(y_over_25).value_counts(normalize=True).to_dict()}")
    logger.info(f"BTTS distribution: {pd.Series(y_btts).value_counts(normalize=True).to_dict()}")
    
    return X, y_home_win, y_over_25, y_btts

def train_models():
    """Train and save models with consistent features"""
    logger.info("=" * 50)
    logger.info("Starting model training with consistent features")
    logger.info(f"Using {len(FEATURE_COLUMNS)} features")
    logger.info("=" * 50)
    
    # Create training data
    X, y_home_win, y_over_25, y_btts = create_training_data(10000)
    
    # Split data
    X_train, X_test, y_home_train, y_home_test = train_test_split(X, y_home_win, test_size=0.2, random_state=42)
    _, _, y_over_train, y_over_test = train_test_split(X, y_over_25, test_size=0.2, random_state=42)
    _, _, y_btts_train, y_btts_test = train_test_split(X, y_btts, test_size=0.2, random_state=42)
    
    models = {}
    scalers = {}
    results = {}
    
    # Train for each target
    targets = [
        ('home_win', y_home_train, y_home_test),
        ('over_2_5', y_over_train, y_over_test),
        ('btts', y_btts_train, y_btts_test)
    ]
    
    for target_name, y_train, y_test in targets:
        logger.info(f"\nTraining model for {target_name}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Create Random Forest model
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            n_jobs=-1,
            random_state=42,
            class_weight='balanced'
        )
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"ROC AUC: {roc_auc:.4f}")
        logger.info(f"Feature importance shape: {model.feature_importances_.shape}")
        
        # Store results
        models[target_name] = model
        scalers[target_name] = scaler
        results[target_name] = {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'feature_importance': dict(zip(FEATURE_COLUMNS, model.feature_importances_))
        }
    
    # Save models
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = Path(f"models/{timestamp}")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    for target, model in models.items():
        model_path = model_dir / f"{target}_ensemble.pkl"
        joblib.dump(model, model_path)
        logger.info(f"Saved {target} model to {model_path}")
    
    # Save scalers
    scaler_path = model_dir / "scalers.pkl"
    joblib.dump(scalers, scaler_path)
    
    # Update current symlink
    current_path = Path("models/current")
    if current_path.exists():
        if current_path.is_symlink():
            current_path.unlink()
        elif current_path.is_dir():
            import shutil
            shutil.rmtree(current_path)
    
    current_path.symlink_to(model_dir)
    
    # Calculate and log overall performance
    avg_accuracy = np.mean([r['accuracy'] for r in results.values()])
    avg_roc_auc = np.mean([r['roc_auc'] for r in results.values()])
    
    logger.info("=" * 50)
    logger.info("Training completed successfully!")
    logger.info(f"Models saved to: {model_dir}")
    logger.info(f"Average accuracy: {avg_accuracy:.4f}")
    logger.info(f"Average ROC AUC: {avg_roc_auc:.4f}")
    logger.info("=" * 50)
    
    return {
        'success': True,
        'model_dir': str(model_dir),
        'results': results,
        'avg_accuracy': avg_accuracy,
        'avg_roc_auc': avg_roc_auc,
        'timestamp': timestamp
    }

if __name__ == "__main__":
    # Create necessary directories
    Path("models").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    # Train models
    result = train_models()
    
    if result['success']:
        print(f"\n✅ Training successful!")
        print(f"📊 Average Accuracy: {result['avg_accuracy']:.4f}")
        print(f"🎯 Average ROC AUC: {result['avg_roc_auc']:.4f}")
        print(f"📁 Models saved to: {result['model_dir']}")
    else:
        print(f"\n❌ Training failed")
