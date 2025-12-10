# train_config.py
"""
Configuration for model training that avoids enhanced features
"""
import os
from typing import List, Dict, Any

# Force basic feature extraction for training
FORCE_BASIC_FEATURES_FOR_TRAINING = True

# Feature selection configuration
TRAINING_FEATURE_NAMES = [
    'minute', 'goals_sum', 'goals_diff', 'xg_sum', 'xg_diff',
    'sot_sum', 'cor_sum', 'pos_diff', 'momentum_h', 'momentum_a'
]

# Minimum variance threshold
FEATURE_VARIANCE_THRESHOLD = 0.001  # Lower than default to accept more features

def get_training_features(features: Dict[str, float]) -> Dict[str, float]:
    """Extract only basic features for training"""
    basic_features = {}
    for key in TRAINING_FEATURE_NAMES:
        if key in features:
            basic_features[key] = features.get(key, 0.0)
        else:
            # Provide sensible defaults for missing features
            if key == 'minute':
                basic_features[key] = 45.0  # Mid-game default
            else:
                basic_features[key] = 0.0
    return basic_features

def should_use_basic_features() -> bool:
    """Check if we should use basic features for training"""
    return FORCE_BASIC_FEATURES_FOR_TRAINING or os.getenv(
        'ENABLE_ENHANCED_FEATURES_FOR_TRAINING', '0'
    ) not in ('1', 'true', 'True', 'yes', 'YES')
