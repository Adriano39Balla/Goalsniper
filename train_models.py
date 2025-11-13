# train_models.py

import os
from typing import Dict, Any, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, accuracy_score
from joblib import dump

from app.supabase_db import get_supabase
from app.features import (
    build_features_1x2,
    build_features_ou25,
    build_features_btts,
)
from app.config import (
    MODEL_1X2,
    MODEL_OU25,
    MODEL_BTTS,
)


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def load_finished_fixtures(limit: int = 50000) -> List[Dict[str, Any]]:
    sb = get_supabase()
    resp = (
        sb.table("fixtures")
        .select("*")
        .eq("status", "FT")   # finished matches only
        .limit(limit)
        .execute()
    )
    data = resp.data or []
    print(f"[TRAIN] Loaded {len(data)} finished fixtures from Supabase.")
    return data


def load_odds_for_fixture(fixture_id: int) -> List[Dict[str, Any]]:
    sb = get_supabase()
    resp = (
        sb.table("odds")
        .select("market,selection,odd")
        .eq("fixture_id", fixture_id)
        .execute()
    )
    return resp.data or []


def build_training_rows(fixtures: List[Dict[str, Any]]):
    """
    Build training datasets for:
      - 1X2 (multiclass)
      - OU 2.5 (binary)
      - BTTS (binary)
    Using:

      - fixtures table for final score
      - odds table (markets: '1X2', 'OU25', 'BTTS')
      - zeroed in-play stats (for now; will get richer as you log more)
    """

    X_1x2, y_1x2 = [], []
    X_ou, y_ou = [], []
    X_btts, y_btts = [], []

    for fx in fixtures:
        try:
            fid = fx["fixture_id"]
            home_goals = fx.get("home_goals")
            away_goals = fx.get("away_goals")

            if home_goals is None or away_goals is None:
                continue

            # Prepare "fixture" + dummy stats (no historical in-play yet)
            fixture = {
                "fixture_id": fid,
                "home_team_id": fx.get("home_team_id"),
                "away_team_id": fx.get("away_team_id"),
                "home_goals": home_goals,
                "away_goals": away_goals,
                "minute": 0,  # prematch snapshot
            }

            stats = {
                "home_shots_on": 0,
                "away_shots_on": 0,
                "home_attacks": 0,
                "away_attacks": 0,
            }

            odds_rows = load_odds_for_fixture(fid)
            if not odds_rows:
                continue

            # Rebuild odds dicts from flat rows
            odds_1x2 = {"home": None, "draw": None, "away": None}
            odds_ou25 = {"over": None, "under": None}
            odds_btts = {"yes": None, "no": None}

            for row in odds_rows:
                market = row.get("market")
                sel = (row.get("selection") or "").upper()
                odd = _safe_float(row.get("odd"))
                if odd is None:
                    continue

                if market == "1X2":
                    if sel in ("HOME", "1"):
                        odds_1x2["home"] = odd
                    elif sel in ("DRAW", "X"):
                        odds_1x2["draw"] = odd
                    elif sel in ("AWAY", "2"):
                        odds_1x2["away"] = odd

                elif market == "OU25":
                    if sel in ("OVER",):
                        odds_ou25["over"] = odd
                    elif sel in ("UNDER",):
                        odds_ou25["under"] = odd

                elif market == "BTTS":
                    if sel in ("YES", "1"):
                        odds_btts["yes"] = odd
                    elif sel in ("NO", "2"):
                        odds_btts["no"] = odd

            # Prematch placeholders
            prematch_strength = {"home": 0.0, "away": 0.0}
            prematch_goal_exp = 2.6
            prematch_btts_exp = 0.55

            # -------------------- 1X2 --------------------
            if all(odds_1x2.values()):
                Xf, _aux = build_features_1x2(
                    fixture, stats, odds_1x2, prematch_strength
                )

                if home_goals > away_goals:
                    y_label = 0  # home
                elif home_goals == away_goals:
                    y_label = 1  # draw
                else:
                    y_label = 2  # away

                X_1x2.append(Xf)
                y_1x2.append(y_label)

            # -------------------- OU25 -------------------
            if odds_ou25.get("over") and odds_ou25.get("under"):

                Xf, _aux = build_features_ou25(
                    fixture, stats, odds_ou25, prematch_goal_exp
                )
                goals_total = home_goals + away_goals
                y_label = 1 if goals_total > 2.5 else 0

                X_ou.append(Xf)
                y_ou.append(y_label)

            # -------------------- BTTS -------------------
            if odds_btts.get("yes") and odds_btts.get("no"):

                Xf, _aux = build_features_btts(
                    fixture, stats, odds_btts, prematch_btts_exp
                )
                y_label = 1 if (home_goals > 0 and away_goals > 0) else 0

                X_btts.append(Xf)
                y_btts.append(y_label)

        except Exception as e:
            print(f"[TRAIN] Error building rows for fixture {fx.get('fixture_id')}: {e}")
            continue

    print(f"[TRAIN] 1X2 samples: {len(X_1x2)}")
    print(f"[TRAIN] OU25 samples: {len(X_ou)}")
    print(f"[TRAIN] BTTS samples: {len(X_btts)}")

    return (
        (np.array(X_1x2), np.array(y_1x2)),
        (np.array(X_ou), np.array(y_ou)),
        (np.array(X_btts), np.array(y_btts)),
    )


def train_logreg_multiclass(X, y):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = LogisticRegression(
        max_iter=1000,
        multi_class="multinomial",
        solver="lbfgs",
    )
    clf.fit(X_train, y_train)
    y_proba = clf.predict_proba(X_val)
    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"[TRAIN] 1X2 validation accuracy: {acc:.3f}")
    return clf


def train_logreg_binary(X, y, label: str):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
    )
    clf.fit(X_train, y_train)
    prob_pos = clf.predict_proba(X_val)[:, 1]
    brier = brier_score_loss(y_val, prob_pos)
    acc = accuracy_score(y_val, (prob_pos >= 0.5).astype(int))
    print(f"[TRAIN] {label} Brier: {brier:.4f} | Acc: {acc:.3f}")
    return clf


def main():
    os.makedirs(os.path.dirname(MODEL_1X2), exist_ok=True)

    fixtures = load_finished_fixtures()
    if not fixtures:
        print("[TRAIN] No finished fixtures found. Abort.")
        return

    (X1, y1), (Xou, you), (Xb, yb) = build_training_rows(fixtures)

    # 1X2
    if len(X1) >= 100:
        print("[TRAIN] Training 1X2 model...")
        model_1x2 = train_logreg_multiclass(X1, y1)
        dump(model_1x2, MODEL_1X2)
        print(f"[TRAIN] 1X2 model saved -> {MODEL_1X2}")
    else:
        print("[TRAIN] Not enough 1X2 samples to train.")

    # OU25
    if len(Xou) >= 100:
        print("[TRAIN] Training OU25 model...")
        model_ou = train_logreg_binary(Xou, you, "OU25")
        dump(model_ou, MODEL_OU25)
        print(f"[TRAIN] OU25 model saved -> {MODEL_OU25}")
    else:
        print("[TRAIN] Not enough OU25 samples to train.")

    # BTTS
    if len(Xb) >= 100:
        print("[TRAIN] Training BTTS model...")
        model_btts = train_logreg_binary(Xb, yb, "BTTS")
        dump(model_btts, MODEL_BTTS)
        print(f"[TRAIN] BTTS model saved -> {MODEL_BTTS}")
    else:
        print("[TRAIN] Not enough BTTS samples to train.")

    print("[TRAIN] Done.")


if __name__ == "__main__":
    main()
