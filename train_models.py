import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
import psycopg2
from psycopg2.extras import RealDictCursor
import joblib
import warnings
warnings.filterwarnings('ignore')
import os
from dotenv import load_dotenv

load_dotenv()

def get_db_connection():
    """Establish database connection"""
    conn = psycopg2.connect(
        host=os.getenv("SUPABASE_URL"),
        database="postgres",
        user="postgres",
        password=os.getenv("SUPABASE_KEY"),
        port=5432
    )
    return conn

def fetch_training_data(years=3):
    """Fetch historical match data for training"""
    conn = get_db_connection()
    
    query = f"""
    SELECT 
        -- Match details
        f.id, f.date, f.home_team, f.away_team, f.home_score, f.away_score,
        
        -- Derived outcomes
        CASE WHEN f.home_score > f.away_score THEN 1 ELSE 0 END as home_win,
        CASE WHEN f.home_score = f.away_score THEN 1 ELSE 0 END as draw,
        CASE WHEN f.home_score < f.away_score THEN 1 ELSE 0 END as away_win,
        CASE WHEN (f.home_score + f.away_score) > 2.5 THEN 1 ELSE 0 END as over_25,
        CASE WHEN (f.home_score + f.away_score) < 2.5 THEN 1 ELSE 0 END as under_25,
        CASE WHEN f.home_score > 0 AND f.away_score > 0 THEN 1 ELSE 0 END as btts_yes,
        
        -- Historical features (simplified - you'd expand these)
        -- Team form last 5 games
        -- Head-to-head history
        -- Injury data
        -- etc.
        
        -- Placeholder features for now
        RANDOM() as form_home_last5,
        RANDOM() as form_away_last5,
        RANDOM() as h2h_home_wins
        
    FROM fixtures f
    WHERE f.date >= NOW() - INTERVAL '{years} years'
    AND f.status = 'FT'  # Only finished matches
    AND f.home_score IS NOT NULL
    ORDER BY f.date DESC
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"Fetched {len(df)} historical matches for training")
    return df

def engineer_features(df):
    """Create advanced features for prediction models"""
    
    # This is a simplified feature engineering process
    # In production, you would create much more sophisticated features:
    # - Rolling averages of team performance
    # - Weighted recent form
    # - Head-to-head statistics
    # - Injury impact metrics
    # - Rest days between matches
    # - Travel distance
    # - Manager statistics
    
    features = pd.DataFrame()
    
    # Basic features (expand these significantly)
    features['home_win_rate'] = df.groupby('home_team')['home_win'].transform('mean').fillna(0.5)
    features['away_win_rate'] = df.groupby('away_team')['away_win'].transform('mean').fillna(0.5)
    
    # Form features (placeholder - you'd implement actual form calculation)
    features['home_form'] = np.random.rand(len(df)) * 0.5 + 0.25
    features['away_form'] = np.random.rand(len(df)) * 0.5 + 0.25
    
    # Goal scoring/conceding averages
    features['home_goals_scored_avg'] = df.groupby('home_team')['home_score'].transform('mean').fillna(1.2)
    features['away_goals_conceded_avg'] = df.groupby('away_team')['away_score'].transform('mean').fillna(1.2)
    
    # Additional placeholder features
    features['feature_1'] = np.random.randn(len(df))
    features['feature_2'] = np.random.randn(len(df))
    features['feature_3'] = np.random.randn(len(df))
    
    return features

def train_winner_model(X, y):
    """Train model to predict match winner (1X2)"""
    print("Training winner prediction model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Model selection and training
    models = {
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'logistic_regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    best_model = None
    best_score = 0
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        score = roc_auc_score(y_test, y_pred_proba)
        
        print(f"  {name}: ROC-AUC = {score:.3f}")
        
        if score > best_score:
            best_score = score
            best_model = model
    
    # Evaluate best model
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Best model: {type(best_model).__name__}")
    print(f"Accuracy: {accuracy:.3f}, ROC-AUC: {best_score:.3f}")
    
    return best_model

def train_over_under_model(X, y):
    """Train model to predict over/under 2.5 goals"""
    print("Training over/under 2.5 goals model...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Accuracy: {accuracy:.3f}, ROC-AUC: {roc_auc:.3f}")
    
    return model

def train_btts_model(X, y):
    """Train model to predict Both Teams To Score"""
    print("Training Both Teams To Score model...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=7,
        min_samples_split=10,
        random_state=42,
        class_weight='balanced'  # Important for imbalanced datasets
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Accuracy: {accuracy:.3f}, ROC-AUC: {roc_auc:.3f}")
    
    return model

def auto_tune_models():
    """Automatic hyperparameter tuning for all models"""
    print("Starting auto-tuning process...")
    
    # Fetch data
    df = fetch_training_data(years=2)
    features = engineer_features(df)
    
    # Prepare targets
    y_winner = df['home_win']
    y_over_under = df['over_25']
    y_btts = df['btts_yes']
    
    # Hyperparameter grids (simplified)
    param_grid_rf = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5, 10]
    }
    
    # Example tuning for one model
    base_model = RandomForestClassifier(random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        features, y_winner, test_size=0.2, random_state=42
    )
    
    grid_search = GridSearchCV(
        base_model,
        param_grid_rf,
        cv=3,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
    
    return grid_search.best_params_

def run_training_pipeline():
    """Complete training pipeline for all models"""
    print("=" * 50)
    print("STARTING MODEL TRAINING PIPELINE")
    print("=" * 50)
    
    # Step 1: Fetch data
    df = fetch_training_data(years=3)
    
    if len(df) < 100:
        print(f"Warning: Only {len(df)} matches available. Need more data for reliable training.")
        return {"status": "warning", "message": "Insufficient data"}
    
    # Step 2: Engineer features
    features = engineer_features(df)
    
    # Step 3: Train individual models
    print("\n--- Training Individual Models ---")
    
    # Winner model
    winner_model = train_winner_model(features, df['home_win'])
    joblib.dump(winner_model, 'models/winner_model.pkl')
    
    # Over/Under model
    over_under_model = train_over_under_model(features, df['over_25'])
    joblib.dump(over_under_model, 'models/over_under_model.pkl')
    
    # BTTS model
    btts_model = train_btts_model(features, df['btts_yes'])
    joblib.dump(btts_model, 'models/btts_model.pkl')
    
    # Step 4: Create ensemble metadata
    ensemble_info = {
        'training_date': pd.Timestamp.now().isoformat(),
        'training_samples': len(df),
        'feature_count': features.shape[1],
        'model_versions': {
            'winner_model': '1.0',
            'over_under_model': '1.0',
            'btts_model': '1.0'
        }
    }
    
    joblib.dump(ensemble_info, 'models/ensemble_info.pkl')
    
    print("\n" + "=" * 50)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 50)
    
    return {
        "status": "success",
        "models_trained": 3,
        "training_samples": len(df),
        "model_saved": True
    }

def backfill_historical_data(days=365):
    """Backfill historical data from API-Football"""
    print(f"Backfilling {days} days of historical data...")
    
    # This would connect to API-Football and fetch historical matches
    # Implement based on your API-Football subscription limits
    
    return {"status": "placeholder", "days": days, "message": "Implement API-Football historical endpoint"}

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Run the training pipeline
    result = run_training_pipeline()
    print(f"\nTraining result: {result}")
