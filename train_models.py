# ============================================================
#   GOALSNIPER.AI — MODEL TRAINING ENGINE (UNIVERSAL VERSION)
# ============================================================

import os
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from joblib import dump

from supabase import create_client

from app.config import (
    MODEL_1X2,
    MODEL_OU25,
    MODEL_BTTS,
    MODEL_DIR,
    SUPABASE_URL,
    SUPABASE_KEY,
)

# ------------------------------------------------------------
# CONNECT SUPABASE
# ------------------------------------------------------------
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
print("[TRAIN] Supabase connected.")


# ------------------------------------------------------------
# UNIVERSAL DATA LOADING - WORKS WITH ANY SCHEMA
# ------------------------------------------------------------

def discover_table_schema(table_name):
    """Discover the actual schema of a table by sampling data"""
    try:
        response = supabase.table(table_name).select("*").limit(5).execute()
        if not response.data:
            return {}
        
        # Analyze the first record to understand schema
        sample_record = response.data[0]
        schema_info = {
            'columns': list(sample_record.keys()),
            'sample': sample_record,
            'has_data': True
        }
        print(f"[TRAIN] Discovered {table_name} columns: {schema_info['columns']}")
        return schema_info
    except Exception as e:
        print(f"[TRAIN] Error discovering {table_name} schema: {e}")
        return {'has_data': False, 'columns': []}


def find_column_mapping(tips_schema, results_schema):
    """Find how to map between tips and results tables"""
    mapping = {
        'tip_id': None,
        'fixture_id': None, 
        'market': None,
        'selection': None,
        'prob': None,
        'odds': None,
        'minute': None,
        'result': None
    }
    
    tip_cols = tips_schema.get('columns', [])
    result_cols = results_schema.get('columns', [])
    
    # Find common columns
    for key in mapping.keys():
        if key in tip_cols:
            mapping[key] = key
        else:
            # Try to find similar columns
            for col in tip_cols:
                if key in col.lower():
                    mapping[key] = col
                    break
    
    print(f"[TRAIN] Column mapping: {mapping}")
    return mapping


def load_training_data_universal():
    """
    Universal data loader that adapts to any database schema
    """
    print("[TRAIN] Loading training data with universal adapter...")
    
    # Discover actual schema
    tips_schema = discover_table_schema("tips")
    results_schema = discover_table_schema("tip_results")
    
    if not tips_schema.get('has_data'):
        print("[TRAIN] No tips data found")
        return pd.DataFrame([])
    
    # Get column mapping
    mapping = find_column_mapping(tips_schema, results_schema)
    
    # Load all tips
    try:
        tips_response = supabase.table("tips").select("*").execute()
        tips_data = tips_response.data or []
        print(f"[TRAIN] Loaded {len(tips_data)} tips")
    except Exception as e:
        print(f"[TRAIN] Error loading tips: {e}")
        return pd.DataFrame([])
    
    # Load results if available
    results_data = []
    if results_schema.get('has_data'):
        try:
            results_response = supabase.table("tip_results").select("*").execute()
            results_data = results_response.data or []
            print(f"[TRAIN] Loaded {len(results_data)} results")
        except Exception as e:
            print(f"[TRAIN] Error loading results: {e}")
    
    # Process data with discovered mapping
    rows = []
    for tip in tips_data:
        row = {}
        
        # Map known fields
        for field, source_col in mapping.items():
            if source_col and source_col in tip:
                row[field] = tip[source_col]
            else:
                row[field] = None
        
        # Try to find matching result
        matching_result = None
        if results_data:
            # Strategy 1: Match by tip_id if available
            if row['tip_id'] and results_data:
                matching_result = next(
                    (r for r in results_data 
                     if r.get(mapping['tip_id'] or 'tip_id') == row['tip_id']), 
                    None
                )
            
            # Strategy 2: Match by fixture_id + market + selection
            if not matching_result and row['fixture_id']:
                fixture_key = mapping['fixture_id'] or 'fixture_id'
                market_key = mapping['market'] or 'market' 
                selection_key = mapping['selection'] or 'selection'
                
                matching_result = next(
                    (r for r in results_data
                     if r.get(fixture_key) == row['fixture_id']
                     and r.get(market_key) == row.get('market')
                     and r.get(selection_key) == row.get('selection')),
                    None
                )
        
        # Add result data if found
        if matching_result:
            result_value = matching_result.get(mapping['result'] or 'result')
            if result_value in ('WIN', 'LOSS'):
                row['result'] = result_value
                row['pnl'] = matching_result.get('pnl', 0)
            else:
                continue  # Skip non-WIN/LOSS results
        else:
            continue  # Skip tips without results
        
        # Validate required fields
        if row.get('prob') is None or row.get('odds') is None:
            continue
            
        try:
            # Ensure numeric fields
            final_row = {
                "fixture_id": row.get('fixture_id', 0),
                "market": row.get('market', ''),
                "selection": row.get('selection', ''),
                "prob": float(row['prob']),
                "odds": float(row['odds']),
                "minute": row.get('minute', 0) or 0,
                "result": row['result'],
                "pnl": row.get('pnl', 0)
            }
            rows.append(final_row)
        except (ValueError, TypeError) as e:
            print(f"[TRAIN] Skipping invalid row: {e}")
            continue
    
    df = pd.DataFrame(rows)
    print(f"[TRAIN] Processed {len(df)} valid training examples")
    return df


# ------------------------------------------------------------
# FALLBACK: CREATE SYNTHETIC TRAINING DATA
# ------------------------------------------------------------

def create_synthetic_training_data():
    """
    Create synthetic training data when no real data exists
    This helps bootstrap the system
    """
    print("[TRAIN] Creating synthetic training data for bootstrapping...")
    
    synthetic_data = []
    
    # Synthetic 1X2 data
    for i in range(50):
        prob = np.random.uniform(0.4, 0.8)
        odds = np.random.uniform(1.5, 3.0)
        minute = np.random.randint(0, 90)
        # Higher prob → more likely to win
        result = "WIN" if prob > 0.55 and np.random.random() > 0.3 else "LOSS"
        
        synthetic_data.append({
            "fixture_id": 1000 + i,
            "market": "1X2",
            "selection": ["home", "draw", "away"][i % 3],
            "prob": prob,
            "odds": odds,
            "minute": minute,
            "result": result,
            "pnl": 1.0 if result == "WIN" else -1.0
        })
    
    # Synthetic OU25 data
    for i in range(50):
        prob = np.random.uniform(0.4, 0.8)
        odds = np.random.uniform(1.5, 2.5)
        minute = np.random.randint(0, 90)
        result = "WIN" if prob > 0.5 and np.random.random() > 0.4 else "LOSS"
        
        synthetic_data.append({
            "fixture_id": 2000 + i,
            "market": "OU25",
            "selection": ["over", "under"][i % 2],
            "prob": prob,
            "odds": odds,
            "minute": minute,
            "result": result,
            "pnl": 1.0 if result == "WIN" else -1.0
        })
    
    # Synthetic BTTS data
    for i in range(50):
        prob = np.random.uniform(0.3, 0.7)
        odds = np.random.uniform(1.6, 2.2)
        minute = np.random.randint(0, 90)
        result = "WIN" if prob > 0.45 and np.random.random() > 0.5 else "LOSS"
        
        synthetic_data.append({
            "fixture_id": 3000 + i,
            "market": "BTTS",
            "selection": ["yes", "no"][i % 2],
            "prob": prob,
            "odds": odds,
            "minute": minute,
            "result": result,
            "pnl": 1.0 if result == "WIN" else -1.0
        })
    
    df = pd.DataFrame(synthetic_data)
    print(f"[TRAIN] Created {len(df)} synthetic training examples")
    return df


# ------------------------------------------------------------
# MODEL TRAINING FUNCTIONS (SAME AS BEFORE)
# ------------------------------------------------------------

def prep(df, market):
    if df.empty:
        return None, None
        
    market_data = df[df["market"] == market]
    if market_data.empty:
        return None, None

    X = market_data[["prob", "minute", "odds"]].fillna(0).astype(float).values
    y = market_data["result"].map({"WIN": 1, "LOSS": 0}).astype(int).values

    return X, y


def train_model_safe(X, y, market_name):
    if len(X) < 5:  # Very low minimum for bootstrapping
        print(f"[TRAIN] Insufficient data for {market_name} ({len(X)} examples)")
        return None

    if len(set(y)) < 2:
        print(f"[TRAIN] Only one class in {market_name} data")
        return None

    try:
        base = LogisticRegression(max_iter=1000)
        # Use simpler approach for very small datasets
        cv_folds = max(2, min(3, len(X) // 5))
        model = CalibratedClassifierCV(base, cv=cv_folds)
        model.fit(X, y)
        print(f"[TRAIN] {market_name} model trained on {len(X)} examples")
        return model
    except Exception as e:
        print(f"[TRAIN] {market_name} model training failed:", e)
        return None


def make_dummy_model_safe():
    """Create a basic model that won't fail"""
    try:
        X = np.array([
            [0.5, 0, 2.0],
            [0.6, 45, 1.8], 
            [0.4, 90, 2.2],
            [0.7, 30, 1.5]
        ])
        y = np.array([0, 1, 0, 1])
        
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        return model
    except Exception as e:
        print("[TRAIN] Dummy model creation failed:", e)
        return None


# ------------------------------------------------------------
# MAIN TRAINING PIPELINE
# ------------------------------------------------------------

def main():
    print("============================================")
    print("        GOALSNIPER AI — TRAINING START")
    print("============================================")

    # Try to load real data first
    df = load_training_data_universal()
    
    # If no real data, use synthetic data for bootstrapping
    if df.empty:
        print("[TRAIN] No real training data found → using synthetic data")
        df = create_synthetic_training_data()
    
    models_trained = 0
    
    # Train models for each market
    markets = [
        ("1X2", MODEL_1X2),
        ("OU25", MODEL_OU25), 
        ("BTTS", MODEL_BTTS)
    ]

    for market_name, model_path in markets:
        X, y = prep(df, market_name)
        if X is not None:
            model = train_model_safe(X, y, market_name)
            if model:
                try:
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    dump(model, model_path)
                    print(f"[TRAIN] ✓ Saved {market_name} model")
                    models_trained += 1
                except Exception as e:
                    print(f"[TRAIN] ✗ Failed to save {market_name} model:", e)
                    # Create dummy as fallback
                    dummy = make_dummy_model_safe()
                    if dummy:
                        dump(dummy, model_path)
            else:
                print(f"[TRAIN] Using fallback model for {market_name}")
                dummy = make_dummy_model_safe()
                if dummy:
                    dump(dummy, model_path)

    print(f"[TRAIN] Training complete. {models_trained}/3 models trained successfully.")


def run_full_training():
    print("[TRAIN] Manual run triggered.")
    main()


if __name__ == "__main__":
    main()
