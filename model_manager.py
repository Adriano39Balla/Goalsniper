import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from config import MODEL_DIR
from utils import logger

def train_model(df, target_column):
    features = ['home_avg_scored', 'home_avg_conceded', 'away_avg_scored', 'away_avg_conceded']
    X = df[features]
    y = df[target_column]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y_enc)

    logger.info(f"Trained model for target '{target_column}'.")
    return model, le

def save_model(model, label_encoder, filename):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    path = os.path.join(MODEL_DIR, filename)
    joblib.dump({'model': model, 'label_encoder': label_encoder}, path)
    logger.info(f"Saved model to {path}.")

def load_model(filename):
    path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(path):
        data = joblib.load(path)
        logger.info(f"Loaded model from {path}.")
        return data['model'], data['label_encoder']
    else:
        logger.info(f"Model file not found: {path}.")
        return None, None

def update_models(df):
    models_info = {
        'outcome': 'model_result.pkl',
        'over_under': 'model_over_under.pkl',
        'btts': 'model_btts.pkl'
    }
    for target, fname in models_info.items():
        model, le = train_model(df, target)
        save_model(model, le, fname)
