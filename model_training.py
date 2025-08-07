# src/model_training.py
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from src.db import get_training_data

def auto_train_model():
    data = get_training_data()
    if len(data) < 10:
        print("[ML] Not enough data to train.")
        return

    teams, leagues, tips, confidences, results = zip(*data)
    labels = [1 if r == "✅" else 0 for r in results]

    text_features = [f"{t} {l} {tip}" for t, l, tip in zip(teams, leagues, tips)]

    # Simple pipeline: text vectorizer + logistic regression
    model = make_pipeline(
        CountVectorizer(),
        LogisticRegression()
    )

    try:
        model.fit(text_features, labels)
        print("[ML] Model trained successfully.")
    except Exception as e:
        print(f"[ML] Training error: {e}")
