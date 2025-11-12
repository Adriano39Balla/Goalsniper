import os
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib


MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def make_model_1x2():
    """
    Dummy multinomial logistic regression for 1X2.
    Input dim: 18 (from build_features_1x2).
    """
    n_features = 18
    n_samples = 800

    X = np.random.normal(size=(n_samples, n_features))

    # Home a bit stronger than away, draw in the middle
    logits_home = X[:, 0] + 0.5 * X[:, 3] - 0.3 * X[:, 4]
    logits_draw = -0.1 * X[:, 0] + 0.1 * X[:, 2]
    logits_away = -logits_home

    logits = np.vstack([logits_home, logits_draw, logits_away]).T
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)

    y = np.array([np.random.choice(3, p=p) for p in probs])

    clf = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=500,
        n_jobs=-1,
    )
    clf.fit(X, y)
    return clf


def make_model_binary(n_features: int):
    """
    Dummy binary logistic regression for OU25 / BTTS.
    """
    n_samples = 800
    X = np.random.normal(size=(n_samples, n_features))
    w = np.random.normal(size=(n_features,))
    logits = X @ w
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (np.random.rand(n_samples) < probs).astype(int)

    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=500,
        n_jobs=-1,
    )
    clf.fit(X, y)
    return clf


def main():
    print("[DUMMY] Creating neutral models in ./models ...")

    model_1x2 = make_model_1x2()
    joblib.dump(model_1x2, os.path.join(MODEL_DIR, "logreg_1x2.pkl"))
    print("[DUMMY] Saved models/logreg_1x2.pkl")

    model_ou25 = make_model_binary(10)  # OU25 has 10 features
    joblib.dump(model_ou25, os.path.join(MODEL_DIR, "logreg_ou25.pkl"))
    print("[DUMMY] Saved models/logreg_ou25.pkl")

    model_btts = make_model_binary(11)  # BTTS has 11 features
    joblib.dump(model_btts, os.path.join(MODEL_DIR, "logreg_btts.pkl"))
    print("[DUMMY] Saved models/logreg_btts.pkl")

    print("[DUMMY] Done. You can now commit the .pkl files.")


if __name__ == "__main__":
    main()
