import os
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import joblib

from app.supabase_db import fetch_training_fixtures
from app.features import (
    build_features_1x2,
    build_features_ou25,
    build_features_btts,
)
from app.config import (
    MODEL_DIR,
    MODEL_1X2,
    MODEL_OU25,
    MODEL_BTTS,
)


# ---------------------------------------------------------
# UTIL
# ---------------------------------------------------------

def ensure_model_dir():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)


# ---------------------------------------------------------
# BUILD TRAINING DATASETS
# ---------------------------------------------------------

def build_dataset_1x2(fixtures: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    X = []
    y = []

    for row in fixtures:
        try:
            fixture = {
                "fixture_id": row["id"],
                "league_id": row.get("league_id"),
                "home_team_id": row["home_team_id"],
                "away_team_id": row["away_team_id"],
                "home_goals": row["home_goals"],
                "away_goals": row["away_goals"],
                "status": "FT",
                "minute": 90,
                "timestamp": row.get("timestamp"),
            }

            stats = {
                "home_shots_on": row.get("home_shots_on", 0),
                "away_shots_on": row.get("away_shots_on", 0),
                "home_attacks": row.get("home_attacks", 0),
                "away_attacks": row.get("away_attacks", 0),
            }

            odds_1x2 = {
                "home": float(row.get("odds_home_1x2") or 0.0),
                "draw": float(row.get("odds_draw_1x2") or 0.0),
                "away": float(row.get("odds_away_1x2") or 0.0),
            }

            prematch_strength = {
                "home": float(row.get("home_strength") or 0.0),
                "away": float(row.get("away_strength") or 0.0),
            }

            # Skip if odds missing (model needs odds to learn their relation)
            if not (odds_1x2["home"] and odds_1x2["draw"] and odds_1x2["away"]):
                continue

            features, _ = build_features_1x2(
                fixture,
                stats,
                odds_1x2,
                prematch_strength,
            )

            hg = row["home_goals"]
            ag = row["away_goals"]

            if hg > ag:
                label = 0  # home
            elif hg == ag:
                label = 1  # draw
            else:
                label = 2  # away

            X.append(features)
            y.append(label)

        except KeyError:
            continue

    return np.array(X), np.array(y)


def build_dataset_ou25(fixtures: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    X = []
    y = []

    for row in fixtures:
        try:
            goals_total = (row["home_goals"] or 0) + (row["away_goals"] or 0)

            fixture = {
                "fixture_id": row["id"],
                "home_goals": row["home_goals"],
                "away_goals": row["away_goals"],
                "minute": 90,
            }

            stats = {
                "home_shots_on": row.get("home_shots_on", 0),
                "away_shots_on": row.get("away_shots_on", 0),
                "home_attacks": row.get("home_attacks", 0),
                "away_attacks": row.get("away_attacks", 0),
            }

            odds_ou = {
                "over": float(row.get("odds_over_25") or 0.0),
                "under": float(row.get("odds_under_25") or 0.0),
            }

            if not (odds_ou["over"] and odds_ou["under"]):
                continue

            prematch_goal_expectation = float(row.get("goal_expectation") or 2.6)

            features, _ = build_features_ou25(
                fixture,
                stats,
                odds_ou,
                prematch_goal_expectation,
            )

            label = 1 if goals_total > 2 else 0  # 1=over, 0=under

            X.append(features)
            y.append(label)

        except KeyError:
            continue

    return np.array(X), np.array(y)


def build_dataset_btts(fixtures: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    X = []
    y = []

    for row in fixtures:
        try:
            hg = row["home_goals"] or 0
            ag = row["away_goals"] or 0
            btts = 1 if (hg > 0 and ag > 0) else 0

            fixture = {
                "fixture_id": row["id"],
                "home_goals": hg,
                "away_goals": ag,
                "minute": 90,
            }

            stats = {
                "home_shots_on": row.get("home_shots_on", 0),
                "away_shots_on": row.get("away_shots_on", 0),
                "home_attacks": row.get("home_attacks", 0),
                "away_attacks": row.get("away_attacks", 0),
            }

            odds_btts = {
                "yes": float(row.get("odds_btts_yes") or 0.0),
                "no": float(row.get("odds_btts_no") or 0.0),
            }

            if not (odds_btts["yes"] and odds_btts["no"]):
                continue

            prematch_btts_expectation = float(row.get("btts_expectation") or 0.55)

            features, _ = build_features_btts(
                fixture,
                stats,
                odds_btts,
                prematch_btts_expectation,
            )

            X.append(features)
            y.append(btts)

        except KeyError:
            continue

    return np.array(X), np.array(y)


# ---------------------------------------------------------
# TRAIN HELPERS
# ---------------------------------------------------------

def train_logreg_multiclass(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=1000,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def train_logreg_binary(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


# ---------------------------------------------------------
# MAIN TRAIN PIPELINE
# ---------------------------------------------------------

def main():
    ensure_model_dir()

    print("[TRAIN] Fetching fixtures from Supabase...")
    fixtures = fetch_training_fixtures(limit=50000)

    if not fixtures:
        print("[TRAIN] No fixtures returned from Supabase. Aborting.")
        return

    print(f"[TRAIN] Retrieved {len(fixtures)} fixtures.")

    # 1X2
    print("[TRAIN] Building dataset for 1X2...")
    X_1x2, y_1x2 = build_dataset_1x2(fixtures)
    print(f"[TRAIN] 1X2 dataset: X={X_1x2.shape}, y={y_1x2.shape}")

    if len(y_1x2) > 1000:
        X_train, X_val, y_train, y_val = train_test_split(X_1x2, y_1x2, test_size=0.2, random_state=42)
        model_1x2 = train_logreg_multiclass(X_train, y_train)

        # Evaluate
        y_val_proba = model_1x2.predict_proba(X_val)
        ll = log_loss(y_val, y_val_proba)
        print(f"[TRAIN] 1X2 log-loss: {ll:.4f}")

        # Save
        joblib.dump(model_1x2, MODEL_1X2)
        print(f"[TRAIN] Saved 1X2 model -> {MODEL_1X2}")
    else:
        print("[TRAIN] Not enough samples for 1X2, skipping model training.")

    # OU 2.5
    print("[TRAIN] Building dataset for OU25...")
    X_ou, y_ou = build_dataset_ou25(fixtures)
    print(f"[TRAIN] OU25 dataset: X={X_ou.shape}, y={y_ou.shape}")

    if len(y_ou) > 1000:
        X_train, X_val, y_train, y_val = train_test_split(X_ou, y_ou, test_size=0.2, random_state=42)
        model_ou = train_logreg_binary(X_train, y_train)
        y_val_proba = model_ou.predict_proba(X_val)
        ll = log_loss(y_val, y_val_proba)
        print(f"[TRAIN] OU25 log-loss: {ll:.4f}")

        joblib.dump(model_ou, MODEL_OU25)
        print(f"[TRAIN] Saved OU25 model -> {MODEL_OU25}")
    else:
        print("[TRAIN] Not enough samples for OU25, skipping model training.")

    # BTTS
    print("[TRAIN] Building dataset for BTTS...")
    X_btts, y_btts = build_dataset_btts(fixtures)
    print(f"[TRAIN] BTTS dataset: X={X_btts.shape}, y={y_btts.shape}")

    if len(y_btts) > 1000:
        X_train, X_val, y_train, y_val = train_test_split(X_btts, y_btts, test_size=0.2, random_state=42)
        model_btts = train_logreg_binary(X_train, y_train)
        y_val_proba = model_btts.predict_proba(X_val)
        ll = log_loss(y_val, y_val_proba)
        print(f"[TRAIN] BTTS log-loss: {ll:.4f}")

        joblib.dump(model_btts, MODEL_BTTS)
        print(f"[TRAIN] Saved BTTS model -> {MODEL_BTTS}")
    else:
        print("[TRAIN] Not enough samples for BTTS, skipping model training.")

    print("[TRAIN] Training complete.")


if __name__ == "__main__":
    main()
