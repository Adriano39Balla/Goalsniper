# create_fallback_models.py
import os
import json
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression

# Create models directory
os.makedirs("models", exist_ok=True)

# Models to create
models_to_create = [
    "1X2_Home_Win", "1X2_Away_Win", "1X2_Draw",
    "BTTS_Yes", "BTTS_No",
    "Over_2_5", "Under_2_5", "Over_3_5", "Under_3_5"
]

for model_name in models_to_create:
    # Determine feature count
    if model_name.startswith("1X2"):
        feature_count = 8
        features = ['minute', 'goals_h', 'goals_a', 'xg_h', 'xg_a', 'sot_h', 'sot_a', 'pos_diff']
    elif model_name.startswith("BTTS"):
        feature_count = 9
        features = ['minute', 'goals_h', 'goals_a', 'xg_h', 'xg_a', 'sot_h', 'sot_a', 'cor_sum', 'momentum_sum']
    else:
        feature_count = 10
        features = ['minute', 'goals_sum', 'xg_sum', 'sot_sum', 'cor_sum', 'pos_diff', 
                   'momentum_h', 'momentum_a', 'pressure_index', 'action_intensity']
    
    # Create simple model
    model = LogisticRegression(random_state=42, max_iter=1000)
    X = np.random.randn(100, feature_count) * 0.1
    y = np.random.randint(0, 2, 100)
    model.fit(X, y)
    
    # Save model
    joblib.dump(model, f"models/{model_name}.joblib")
    
    # Save metadata
    metadata = {
        'model_name': model_name,
        'required_features': features,
        'is_fallback': True,
        'created_at': time.time()
    }
    with open(f"models/{model_name}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Created {model_name}")
