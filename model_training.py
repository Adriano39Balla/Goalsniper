from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from db import get_training_data
import logging

logger = logging.getLogger("uvicorn")

def auto_train_model():
    data = get_training_data()
    if len(data) < 10:
        logger.info("[ML] Not enough data to train.")
        return

    try:
        teams, leagues, tips, confidences, results = zip(*data)

        # Ensure only valid result labels
        labels = [1 if result == "✅" else 0 for result in results]

        # Concatenate features into one string
        text_features = [f"{team} {league} {tip}" for team, league, tip in zip(teams, leagues, tips)]

        model = make_pipeline(
            CountVectorizer(),
            LogisticRegression()
        )

        model.fit(text_features, labels)
        logger.info("[ML] Model trained successfully.")

    except Exception as e:
        logger.error(f"[ML] Training error: {e}")
