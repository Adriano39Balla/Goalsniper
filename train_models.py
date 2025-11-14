# ============================================================
#   GOALSNIPER.AI — MODEL TRAINING ENGINE (FINAL SAFE VERSION)
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
# FETCH TRAINING DATA WITH FALLBACK SUPPORT
# ------------------------------------------------------------

def fetch_training_dataset_fallback():
    """
    Fallback method when foreign key relationship doesn't exist
    """
    print("[TRAIN] Using fallback data loading method...")
    
    try:
        # Load tips and results separately
        tips_resp = supabase.table("tips").select("*").order("id", desc=False).execute()
        results_resp = supabase.table("tip_results").select("*").execute()
        
        tips_data = tips_resp.data or []
        results_data = results_resp.data or []
        
        print(f"[TRAIN] Loaded {len(tips_data)} tips and {len(results_data)} results")
        
        # Create a mapping of tip_id to result for quick lookup
        results_map = {}
        for result in results_data:
            tip_id = result.get('tip_id')
            if tip_id:
                results_map[tip_id] = result
        
        rows = []
        for tip in tips_data:
            tip_id = tip.get('id')
            result = results_map.get(tip_id)
            
            if not result:
                continue
                
            if tip["prob"] is None or tip["odds"] is None:
                continue
            if result.get("result") not in ("WIN", "LOSS"):
                continue

            rows.append({
                "fixture_id": tip["fixture_id"],
                "market": tip["market"],
                "selection": tip["selection"],
                "prob": float(tip["prob"]),
                "odds": float(tip["odds"]),
                "minute": tip.get("minute") or 0,
                "result": result["result"],
                "pnl": result.get("pnl", 0),
            })

        df = pd.DataFrame(rows)
        print(f"[TRAIN] Fallback loaded {len(df)} labeled examples.")
        return df
        
    except Exception as e:
        print("[TRAIN] ERROR in fallback loading:", e)
        return pd.DataFrame([])


def fetch_training_dataset():
    """
    Loads historical tips + results with automatic fallback
    """
    print("[TRAIN] Fetching training dataset...")

    try:
        # Try the joined query first
        resp = (
            supabase.table("tips")
            .select("*, tip_results!tip_id(*)")
            .order("id", desc=False)
            .execute()
        )
        
        data = resp.data or []
        print(f"[TRAIN] Joined query successful, found {len(data)} tips")

    except Exception as e:
        error_str = str(e)
        if "relationship" in error_str and "tip_results" in error_str:
            print("[TRAIN] Foreign key relationship missing, using fallback...")
            return fetch_training_dataset_fallback()
        else:
            print("[TRAIN] ERROR loading data:", e)
            return pd.DataFrame([])

    rows = []
    for tip in data:
        results = tip.get("tip_results") or []
        if not results:
            continue

        res = results[0]

        if tip["prob"] is None or tip["odds"] is None:
            continue
        if res.get("result") not in ("WIN", "LOSS"):
            continue

        rows.append({
            "fixture_id": tip["fixture_id"],
            "market": tip["market"],
            "selection": tip["selection"],
            "prob": float(tip["prob"]),
            "odds": float(tip["odds"]),
            "minute": tip.get("minute") or 0,
            "result": res["result"],
            "pnl": res.get("pnl", 0),
        })

    df = pd.DataFrame(rows)
    print(f"[TRAIN] Loaded {len(df)} labeled examples.")
    return df


# ------------------------------------------------------------
# MARKET PREPROCESSOR
# ------------------------------------------------------------

def prep(df, market):
    df = df[df["market"] == market]
    if df.empty:
        return None, None

    X = df[["prob", "minute", "odds"]].fillna(0).astype(float).values
    y = df["result"].map({"WIN": 1, "LOSS": 0}).astype(int).values

    return X, y


# ------------------------------------------------------------
# TRAIN A MODEL
# ------------------------------------------------------------

def train_model(X, y):
    if len(X) < 40:
        print(f"[TRAIN] Too little data ({len(X)} examples) → skipping real training.")
        return None

    if len(set(y)) < 2:
        print("[TRAIN] Only one class present → skipping.")
        return None

    try:
        base = LogisticRegression(max_iter=1000)
        model = CalibratedClassifierCV(base, cv=min(3, len(X) // 10))  # Adaptive CV
        model.fit(X, y)
        print(f"[TRAIN] Model trained successfully on {len(X)} examples")
        return model

    except Exception as e:
        print("[TRAIN] Model training failed:", e)
        return None


# ------------------------------------------------------------
# SAVE MODEL
# ------------------------------------------------------------

def save_model(model, path, storage_name):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        dump(model, path)
        print(f"[TRAIN] Saved model → {path}")

        # Try to upload to Supabase Storage
        try:
            with open(path, "rb") as f:
                supabase.storage.from_("models").upload(
                    path=storage_name,
                    file=f,
                    file_options={"content-type": "application/octet-stream"},
                    upsert=True
                )
            print(f"[TRAIN] Uploaded → Supabase Storage/{storage_name}")

        except Exception as e:
            print("[TRAIN] Upload to Supabase Storage failed:", e)

    except Exception as e:
        print(f"[TRAIN] Failed to save model {path}:", e)


# ------------------------------------------------------------
# SAFE DUMMY MODEL
# ------------------------------------------------------------

def make_dummy():
    X = np.array([[0, 0, 1], [1, 90, 10]])
    y = np.array([0, 1])

    base = LogisticRegression(max_iter=200)
    model = CalibratedClassifierCV(base, cv=2)
    model.fit(X, y)
    return model


def train_dummies():
    print("[TRAIN] Creating dummy models...")

    dummy = make_dummy()

    save_model(dummy, MODEL_1X2, "logreg_1x2.pkl")
    save_model(dummy, MODEL_OU25, "logreg_ou25.pkl")
    save_model(dummy, MODEL_BTTS, "logreg_btts.pkl")

    print("[TRAIN] Dummy models created.")


# ------------------------------------------------------------
# MAIN TRAINING PIPELINE
# ------------------------------------------------------------

def main():
    print("============================================")
    print("        GOALSNIPER AI — TRAINING START")
    print("============================================")

    df = fetch_training_dataset()

    models_trained = 0

    if df.empty:
        print("[TRAIN] No training data → dummy models only.")
        train_dummies()
        return

    # ---- 1X2 ----
    X1, y1 = prep(df, "1X2")
    if X1 is not None:
        m1 = train_model(X1, y1)
        if m1:
            save_model(m1, MODEL_1X2, "logreg_1x2.pkl")
            models_trained += 1
        else:
            print("[TRAIN] 1X2 model training failed, using dummy")
            save_model(make_dummy(), MODEL_1X2, "logreg_1x2.pkl")

    # ---- OU25 ----
    X2, y2 = prep(df, "OU25")
    if X2 is not None:
        m2 = train_model(X2, y2)
        if m2:
            save_model(m2, MODEL_OU25, "logreg_ou25.pkl")
            models_trained += 1
        else:
            print("[TRAIN] OU25 model training failed, using dummy")
            save_model(make_dummy(), MODEL_OU25, "logreg_ou25.pkl")

    # ---- BTTS ----
    X3, y3 = prep(df, "BTTS")
    if X3 is not None:
        m3 = train_model(X3, y3)
        if m3:
            save_model(m3, MODEL_BTTS, "logreg_btts.pkl")
            models_trained += 1
        else:
            print("[TRAIN] BTTS model training failed, using dummy")
            save_model(make_dummy(), MODEL_BTTS, "logreg_btts.pkl")

    print(f"[TRAIN] Training complete. {models_trained} models trained successfully.")


# ------------------------------------------------------------
# MANUAL TRIGGER
# ------------------------------------------------------------

def run_full_training():
    print("[TRAIN] Manual run triggered.")
    main()


if __name__ == "__main__":
    main()
