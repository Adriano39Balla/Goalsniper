# ============================================================
#   GOALSNIPER.AI — MODEL TRAINING ENGINE (ROBUST VERSION)
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
# DATABASE SCHEMA VALIDATION
# ------------------------------------------------------------

def validate_database_schema():
    """Check if required tables and columns exist"""
    try:
        # Test if tips table exists and has basic structure
        test_resp = supabase.table("tips").select("id").limit(1).execute()
        print("[TRAIN] Tips table exists and is accessible")
        return True
    except Exception as e:
        error_str = str(e)
        if "column tips.id does not exist" in error_str:
            print("[TRAIN] WARNING: Tips table has different structure than expected")
            return False
        elif "relation" in error_str and "tips" in error_str:
            print("[TRAIN] WARNING: Tips table doesn't exist")
            return False
        else:
            print("[TRAIN] WARNING: Unknown database error:", e)
            return False


# ------------------------------------------------------------
# FALLBACK DATA LOADING WITH SCHEMA DETECTION
# ------------------------------------------------------------

def fetch_training_dataset_robust():
    """
    Robust data loading that handles various schema issues
    """
    print("[TRAIN] Fetching training dataset with robust loader...")

    # First, validate schema
    if not validate_database_schema():
        print("[TRAIN] Schema validation failed - no training data available")
        return pd.DataFrame([])

    try:
        # Try different column names for ID
        possible_id_columns = ['id', 'tip_id', 'fixture_id']
        
        for id_col in possible_id_columns:
            try:
                # Try to get tips with this ID column
                tips_resp = supabase.table("tips").select("*").order(id_col, desc=False).limit(100).execute()
                if tips_resp.data:
                    print(f"[TRAIN] Successfully loaded tips using '{id_col}' as key")
                    tips_data = tips_resp.data
                    break
            except:
                continue
        else:
            print("[TRAIN] Could not find valid ID column in tips table")
            return pd.DataFrame([])

    except Exception as e:
        print("[TRAIN] ERROR loading tips:", e)
        return pd.DataFrame([])

    # Try to load results
    try:
        results_resp = supabase.table("tip_results").select("*").limit(100).execute()
        results_data = results_resp.data or []
        print(f"[TRAIN] Loaded {len(results_data)} results")
    except Exception as e:
        print("[TRAIN] WARNING: Could not load tip_results:", e)
        results_data = []

    # Manual joining with flexible column mapping
    rows = []
    for tip in tips_data:
        # Find matching result - try different join strategies
        matching_result = None
        
        # Strategy 1: Try tip_id in results
        tip_id = tip.get('id')
        if tip_id and results_data:
            matching_result = next((r for r in results_data if r.get('tip_id') == tip_id), None)
        
        # Strategy 2: Try fixture_id matching
        if not matching_result:
            fixture_id = tip.get('fixture_id')
            if fixture_id and results_data:
                market = tip.get('market')
                selection = tip.get('selection')
                matching_result = next(
                    (r for r in results_data 
                     if r.get('fixture_id') == fixture_id 
                     and r.get('market') == market
                     and r.get('selection') == selection), 
                    None
                )

        if not matching_result:
            continue

        # Validate required fields
        prob = tip.get("prob")
        odds = tip.get("odds")
        result = matching_result.get("result")

        if prob is None or odds is None:
            continue
        if result not in ("WIN", "LOSS"):
            continue

        try:
            rows.append({
                "fixture_id": tip.get("fixture_id"),
                "market": tip.get("market"),
                "selection": tip.get("selection"),
                "prob": float(prob),
                "odds": float(odds),
                "minute": tip.get("minute") or 0,
                "result": result,
                "pnl": matching_result.get("pnl", 0),
            })
        except (ValueError, TypeError) as e:
            print(f"[TRAIN] Skipping invalid tip data: {e}")
            continue

    df = pd.DataFrame(rows)
    print(f"[TRAIN] Successfully loaded {len(df)} valid training examples.")
    return df


# ------------------------------------------------------------
# MARKET PREPROCESSOR
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


# ------------------------------------------------------------
# ROBUST MODEL TRAINING
# ------------------------------------------------------------

def train_model_safe(X, y, market_name):
    """Train model with comprehensive error handling"""
    if len(X) < 10:  # Lowered minimum for early stages
        print(f"[TRAIN] Insufficient data for {market_name} ({len(X)} examples)")
        return None

    if len(set(y)) < 2:
        print(f"[TRAIN] Only one class in {market_name} data")
        return None

    try:
        # Use simpler approach for small datasets
        if len(X) < 50:
            base = LogisticRegression(max_iter=1000)
            model = CalibratedClassifierCV(base, cv=min(2, len(X) // 5))
        else:
            base = LogisticRegression(max_iter=1000)
            model = CalibratedClassifierCV(base, cv=min(3, len(X) // 10))
            
        model.fit(X, y)
        print(f"[TRAIN] {market_name} model trained on {len(X)} examples")
        return model

    except Exception as e:
        print(f"[TRAIN] {market_name} model training failed:", e)
        return None


# ------------------------------------------------------------
# DUMMY MODEL CREATION
# ------------------------------------------------------------

def make_dummy_model_safe():
    """Create a safe dummy model without cross-validation issues"""
    try:
        # Simple model without calibration to avoid CV issues
        X = np.array([[0.5, 0, 2.0], [0.6, 90, 1.8]])
        y = np.array([0, 1])
        
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        return model
    except Exception as e:
        print("[TRAIN] Dummy model creation failed:", e)
        return None


def train_dummies_safe():
    """Create dummy models safely"""
    print("[TRAIN] Creating safe dummy models...")

    dummy = make_dummy_model_safe()
    if dummy is None:
        print("[TRAIN] Failed to create dummy models")
        return

    try:
        os.makedirs(os.path.dirname(MODEL_1X2), exist_ok=True)
        dump(dummy, MODEL_1X2)
        dump(dummy, MODEL_OU25)
        dump(dummy, MODEL_BTTS)
        print("[TRAIN] Safe dummy models created and saved locally")
    except Exception as e:
        print("[TRAIN] Failed to save dummy models:", e)


# ------------------------------------------------------------
# MAIN TRAINING PIPELINE
# ------------------------------------------------------------

def main():
    print("============================================")
    print("        GOALSNIPER AI — TRAINING START")
    print("============================================")

    # Load training data
    df = fetch_training_dataset_robust()

    if df.empty:
        print("[TRAIN] No training data available → creating safe dummy models")
        train_dummies_safe()
        print("[TRAIN] Training pipeline completed with dummy models")
        return

    models_trained = 0

    # Train each market model
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
                    dump(model, model_path)
                    print(f"[TRAIN] Saved {market_name} model to {model_path}")
                    models_trained += 1
                except Exception as e:
                    print(f"[TRAIN] Failed to save {market_name} model:", e)
            else:
                print(f"[TRAIN] Using dummy model for {market_name}")
                dummy = make_dummy_model_safe()
                if dummy:
                    dump(dummy, model_path)

    print(f"[TRAIN] Training complete. {models_trained} models trained successfully.")


# ------------------------------------------------------------
# MANUAL TRIGGER
# ------------------------------------------------------------

def run_full_training():
    print("[TRAIN] Manual run triggered.")
    main()


if __name__ == "__main__":
    main()
