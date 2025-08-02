import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def train_model(data: pd.DataFrame):
    print("📊 Training AI model...")
    X = data.drop(columns=["target"])
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    print("✅ Model trained.")
    return model

def predict(model, data: pd.DataFrame):
    print("🤖 Generating predictions...")
    X = data.drop(columns=["target"])
    predictions = model.predict(X)
    return predictions

def load_model(path: str):
    return joblib.load(path)

def save_model(model, path: str):
    joblib.dump(model, path)
