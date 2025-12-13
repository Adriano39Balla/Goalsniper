import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import joblib
import logging
from dotenv import load_dotenv
import os

load_dotenv()

logger = logging.getLogger(__name__)

def get_db_connection():
    """Get database connection"""
    return psycopg2.connect(os.getenv('DATABASE_URL'))

def fetch_training_data(conn, days: int = 90):
    """Fetch historical data for training"""
    query = """
    WITH match_data AS (
        SELECT 
            ep.fixture_id,
            ep.league_id,
            ep.minute,
            ep.predictions,
            ms.market_type,
            ms.probability as predicted_probability,
            ms.confidence_score,
            mr.outcome,
            mr.actual_probability,
            mr.match_state,
            mr.analyzed_at
        FROM event_predictions ep
        LEFT JOIN market_suggestions ms ON ep.fixture_id = ms.fixture_id 
            AND ABS(ep.minute - EXTRACT(MINUTE FROM ms.created_at)) < 5
        LEFT JOIN market_results mr ON ms.id = mr.market_suggestion_id
        WHERE ep.created_at >= NOW() - INTERVAL '%s days'
            AND mr.outcome IS NOT NULL
    )
    SELECT * FROM match_data
    """
    
    df = pd.read_sql_query(query, conn, params=(days,))
    return df

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare features for training"""
    features = []
    
    for _, row in df.iterrows():
        try:
            predictions = row['predictions']
            if isinstance(predictions, str):
                predictions = json.loads(predictions)
            
            match_state = row['match_state']
            if isinstance(match_state, str):
                match_state = json.loads(match_state)
            
            # Extract base features
            feature = {
                'league_id': row['league_id'],
                'minute': row['minute'],
                'market_type': row['market_type'],
                'predicted_probability': row['predicted_probability'],
                'confidence_score': row['confidence_score'],
                
                # Event probabilities
                'goal_next_10min': predictions.get('goal_next_10min', 0),
                'home_goal_probability': predictions.get('home_goal_probability', 0),
                'away_goal_probability': predictions.get('away_goal_probability', 0),
                'expected_final_goals': predictions.get('expected_final_goals', 0),
                'corner_next_10min': predictions.get('corner_next_10min', 0),
                'yellow_card_next_10min': predictions.get('yellow_card_next_10min', 0),
                
                # Match state
                'score_delta': match_state.get('score_delta', 0),
                'home_possession': match_state.get('home_possession', 0.5),
                'shot_pressure': match_state.get('shot_pressure', 0),
                'corner_rate': match_state.get('corner_rate', 0),
                
                # Target
                'outcome': 1 if row['outcome'] == 'win' else 0
            }
            
            features.append(feature)
        except Exception as e:
            logger.warning(f"Error processing row: {e}")
            continue
    
    return pd.DataFrame(features)

def train_market_model(market_type: str, df: pd.DataFrame):
    """Train model for specific market type"""
    market_df = df[df['market_type'] == market_type].copy()
    
    if len(market_df) < 50:
        logger.warning(f"Insufficient data for {market_type}: {len(market_df)} samples")
        return None
    
    # Prepare features and target
    feature_cols = [
        'minute', 'predicted_probability', 'confidence_score',
        'goal_next_10min', 'home_goal_probability', 'away_goal_probability',
        'expected_final_goals', 'corner_next_10min', 'yellow_card_next_10min',
        'score_delta', 'home_possession', 'shot_pressure', 'corner_rate'
    ]
    
    X = market_df[feature_cols]
    y = market_df['outcome']
    
    # Encode league_id if needed
    if 'league_id' in X.columns:
        X = pd.get_dummies(X, columns=['league_id'], drop_first=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'train_samples': len(X_train),
        'test_samples': len(X_test)
    }
    
    logger.info(f"{market_type} model performance: {metrics}")
    
    # Save model and scaler
    joblib.dump(model, f'models/{market_type}_model.joblib')
    joblib.dump(scaler, f'models/{market_type}_scaler.joblib')
    
    return metrics

def train_event_prediction_model(df: pd.DataFrame):
    """Train model for event predictions"""
    # This would train models to predict events (goals, corners, cards)
    # For now, return placeholder
    logger.info("Event prediction model training placeholder")
    return {}

def update_league_stats(conn):
    """Update league statistics based on recent performance"""
    cur = conn.cursor()
    
    # Update reliability scores
    cur.execute("""
        UPDATE league_stats ls
        SET 
            reliability_score = COALESCE(
                (SELECT 
                    CASE 
                        WHEN COUNT(*) > 20 
                        THEN SUM(CASE WHEN mr.outcome = 'win' THEN 1 ELSE 0 END)::float / COUNT(*)
                        ELSE ls.reliability_score
                    END
                 FROM market_suggestions ms
                 JOIN market_results mr ON ms.id = mr.market_suggestion_id
                 WHERE ms.league_id = ls.league_id
                    AND mr.analyzed_at >= NOW() - INTERVAL '30 days'
                ), 
                ls.reliability_score
            ),
            last_updated = NOW()
        WHERE EXISTS (
            SELECT 1 FROM market_suggestions ms
            JOIN market_results mr ON ms.id = mr.market_suggestion_id
            WHERE ms.league_id = ls.league_id
                AND mr.analyzed_at >= NOW() - INTERVAL '30 days'
        )
    """)
    
    conn.commit()
    cur.close()
    logger.info("League stats updated")

def train_all_models(conn):
    """Train all models"""
    logger.info("Starting model training...")
    
    try:
        # Fetch training data
        df_raw = fetch_training_data(conn, days=90)
        
        if df_raw.empty:
            logger.warning("No training data available")
            return {"status": "no_data"}
        
        # Prepare features
        df = prepare_features(df_raw)
        
        if df.empty:
            logger.warning("No valid features extracted")
            return {"status": "no_features"}
        
        # Train models for each market type
        market_types = df['market_type'].unique()
        results = {}
        
        for market_type in market_types:
            logger.info(f"Training model for {market_type}")
            metrics = train_market_model(market_type, df)
            if metrics:
                results[market_type] = metrics
        
        # Train event prediction models
        event_results = train_event_prediction_model(df)
        results['event_models'] = event_results
        
        # Update league statistics
        update_league_stats(conn)
        
        logger.info(f"Model training completed: {len(results)} models trained")
        return {
            "status": "success",
            "results": results,
            "total_samples": len(df)
        }
        
    except Exception as e:
        logger.error(f"Error in model training: {e}")
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    # Standalone training script
    logging.basicConfig(level=logging.INFO)
    
    conn = get_db_connection()
    results = train_all_models(conn)
    conn.close()
    
    print(f"Training results: {results}")
