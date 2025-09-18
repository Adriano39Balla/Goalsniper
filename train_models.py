import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import psycopg
from psycopg.rows import dict_row
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression

# ========= ENV / CONFIG =========
TZ = ZoneInfo(os.getenv("TZ", "Europe/Berlin"))
DB_URL = os.getenv("DATABASE_URL", "")
TARGET_PRECISION = float(os.getenv("TARGET_PRECISION", "0.78"))  # 75–80% sweet spot
MIN_SAMPLES_TO_TRAIN = int(os.getenv("MIN_SAMPLES_TO_TRAIN", "1200"))  # guardrail
VALID_FRAC = float(os.getenv("VALID_FRAC", "0.10"))  # last 10% for validation
TEST_FRAC  = float(os.getenv("TEST_FRAC",  "0.10"))  # last 10% for holdout

# Keep in sync with main.py
FEATURES_ORDER = [
    "home_form_gf5","home_form_ga5","away_form_gf5","away_form_ga5",
    "home_rest_days","away_rest_days","league_goal_env",
    "odds_over_implied","odds_under_implied","odds_overround",
    "odds_over_drift","odds_under_drift"
]

# ========= SQL HELPERS =========
LOAD_SQL = """
SELECT
  f.fixture_id,
  fx.kickoff,
  fx.league_name,
  f.home_form_gf5, f.home_form_ga5, f.away_form_gf5, f.away_form_ga5,
  f.home_rest_days, f.away_rest_days, f.league_goal_env,
  f.odds_over_implied, f.odds_under_implied, f.odds_overround,
  COALESCE(f.odds_over_drift, 0)  AS odds_over_drift,
  COALESCE(f.odds_under_drift, 0) AS odds_under_drift,
  (fx.goals_home + fx.goals_away) >= 3 AS y_over25
FROM features f
JOIN fixtures fx ON fx.fixture_id = f.fixture_id
WHERE fx.status = 'FT'
  AND fx.goals_home IS NOT NULL
  AND fx.goals_away IS NOT NULL
ORDER BY fx.kickoff ASC;
"""

UPSERT_MODEL_SQL = """
INSERT INTO model_registry(label, version, lib, blob, meta)
VALUES ('ou25_global', %s, 'catboost', %s, %s);
"""

UPSERT_POLICY_SQL = """
INSERT INTO model_cfg(key, value)
VALUES ('policy', %s)
ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value;
"""

# ========= DATA LOADING =========
async def load_training_dataframe():
    async with await psycopg.AsyncConnection.connect(DB_URL, row_factory=dict_row) as con:
        rows = await con.execute(LOAD_SQL)
        data = await rows.fetchall()
    if not data:
        return None
    df = pd.DataFrame(data)
    # strict cast & cleanup
    for col in FEATURES_ORDER:
        if col not in df:
            df[col] = np.nan
    df = df.dropna(subset=FEATURES_ORDER + ["kickoff", "y_over25"])
    # boolean -> int
    df["y_over25"] = df["y_over25"].astype(int)
    # sanity for implied probs (avoid div-by-zero later if you add EV here)
    df["odds_over_implied"]  = df["odds_over_implied"].fillna(0.0).clip(0, 1)
    df["odds_under_implied"] = df["odds_under_implied"].fillna(0.0).clip(0, 1)
    return df

def time_split(df: pd.DataFrame):
    # chronological split: [train | valid | test]
    n = len(df)
    if n < MIN_SAMPLES_TO_TRAIN:
        return None, None, None
    test_cut  = int(n * (1 - TEST_FRAC))
    valid_cut = int(test_cut * (1 - VALID_FRAC))
    train_df = df.iloc[:valid_cut].copy()
    valid_df = df.iloc[valid_cut:test_cut].copy()
    test_df  = df.iloc[test_cut:].copy()
    return train_df, valid_df, test_df

# ========= MODEL TRAINING =========
def fit_catboost_with_platt(train_df: pd.DataFrame, valid_df: pd.DataFrame):
    Xtr = train_df[FEATURES_ORDER].astype(float).values
    ytr = train_df["y_over25"].astype(int).values
    Xva = valid_df[FEATURES_ORDER].astype(float).values
    yva = valid_df["y_over25"].astype(int).values

    # CatBoost — good defaults for tabular, fast enough for nightly
    model = CatBoostClassifier(
        loss_function="Logloss",
        depth=6,
        learning_rate=0.06,
        l2_leaf_reg=3.0,
        iterations=800,
        random_strength=1.2,
        rsm=0.9,
        bootstrap_type="MVS",
        eval_metric="Logloss",
        verbose=False
    )
    model.fit(Pool(Xtr, ytr), eval_set=Pool(Xva, yva), verbose=False)

    # Platt calibration on validation logits
    p_va = np.clip(model.predict_proba(Xva)[:, 1], 1e-6, 1 - 1e-6)
    logits = np.log(p_va / (1 - p_va))
    lr = LogisticRegression(max_iter=1000)
    lr.fit(logits.reshape(-1, 1), yva)

    class CalibratedModel:
        def __init__(self, cat, lr):
            self.cat = cat
            self.lr = lr
        def predict_over(self, X: np.ndarray) -> np.ndarray:
            raw = np.clip(self.cat.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
            z = np.log(raw / (1 - raw)).reshape(-1, 1)
            return self.lr.predict_proba(z)[:, 1]

    calib = CalibratedModel(model, lr)
    return calib

def evaluate(model, df: pd.DataFrame):
    if df is None or df.empty:
        return {"bets": 0}
    X = df[FEATURES_ORDER].astype(float).values
    y = df["y_over25"].astype(int).values
    p = model.predict_over(X)
    # Simple quality diagnostics
    logloss = float(-np.mean(y * np.log(np.clip(p,1e-9,1)) + (1 - y) * np.log(np.clip(1 - p,1e-9,1))))
    brier   = float(np.mean((p - y) ** 2))
    return {"bets": int(len(y)), "logloss": logloss, "brier": brier}

# ========= POLICY TUNING (LEAGUE-AWARE) =========
def implied_from_dec(odds):
    return 1.0 / odds if odds and odds > 1.0 else 0.0

def derive_policy(valid_df: pd.DataFrame, model, target_precision: float):
    """
    For each league (and globally), grid-search (theta, ev_min) that
    meets/exceeds precision target and maximizes coverage.
    """
    policy = {"global": {"theta": 0.78, "ev_min": 0.04}}  # defaults
    if valid_df is None or valid_df.empty:
        return policy

    leagues = sorted(valid_df["league_name"].str.lower().unique().tolist())
    # Precompute probabilities for valid set
    Xv = valid_df[FEATURES_ORDER].astype(float).values
    p_over = model.predict_over(Xv)

    # Build odds back from implied (robust to zeros)
    eps = 1e-9
    o_over  = 1.0 / np.clip(valid_df["odds_over_implied"].values,  eps, 1.0)   # rough book odds
    o_under = 1.0 / np.clip(valid_df["odds_under_implied"].values, eps, 1.0)

    def eval_block(mask):
        idx = np.where(mask)[0]
        if len(idx) < 50:
            return None
        best = None
        for th_i in range(70, 86):   # 0.70 .. 0.85
            th = th_i / 100
            for ev_i in range(0, 11):  # 0.00 .. 0.10
                evm = ev_i / 100
                bets = wins = 0
                for i in idx:
                    p = p_over[i]; pu = 1 - p
                    # conservative blend with market priors (same as main.py)
                    pbk = (1 / o_over[i]) / ((1 / o_over[i]) + (1 / o_under[i]))
                    p_blend = 0.7 * p + 0.3 * pbk
                    p_blend_u = 1.0 - p_blend
                    ev_o = p_blend * (o_over[i] - 1) - (1 - p_blend)
                    ev_u = p_blend_u * (o_under[i] - 1) - (1 - p_blend_u)
                    # decide selection
                    if p_blend >= th and ev_o >= evm and ev_o >= ev_u:
                        pick_over = 1
                    elif p_blend_u >= th and ev_u >= evm and ev_u > ev_o:
                        pick_over = 0
                    else:
                        continue
                    bets += 1
                    y = int(valid_df.iloc[i]["y_over25"])
                    won = (pick_over == 1 and y == 1) or (pick_over == 0 and y == 0)
                    wins += 1 if won else 0
                if bets == 0:
                    continue
                precision = wins / bets
                score = (precision >= target_precision, bets, precision)
                if best is None or score > best[0]:
                    best = (score, th, evm, precision, bets)
        if best:
            _, th, evm, prec, cov = best
            return {"theta": round(th, 2), "ev_min": round(evm, 2), "precision": round(prec, 3), "coverage": cov}
        return None

    # Global
    res_g = eval_block(np.ones(len(valid_df), dtype=bool))
    if res_g:
        policy["global"] = {"theta": res_g["theta"], "ev_min": res_g["ev_min"]}

    # Per-league
    for lg in leagues:
        mask = valid_df["league_name"].str.lower() == lg
        res = eval_block(mask.values)
        if res:
            policy[lg] = {"theta": res["theta"], "ev_min": res["ev_min"]}

    return policy

# ========= PERSISTENCE =========
async def save_model_and_policy(model, meta: dict, policy: dict):
    blob = pickle.dumps(model)
    async with await psycopg.AsyncConnection.connect(DB_URL) as con:
        await con.execute(UPSERT_MODEL_SQL, (datetime.now(TZ).strftime("%Y%m%d-%H%M"), psycopg.Binary(blob), json.dumps(meta)))
        await con.execute(UPSERT_POLICY_SQL, (json.dumps(policy),))
        await con.commit()

# ========= MAIN =========
async def main():
    if not DB_URL:
        raise SystemExit("Missing DATABASE_URL")
    # Load data
    df = await load_training_dataframe()
    if df is None or len(df) < MIN_SAMPLES_TO_TRAIN:
        print(f"[TRAIN] Not enough samples ({0 if df is None else len(df)}) — harvest/backfill more history.")
        return

    # Time split
    train_df, valid_df, test_df = time_split(df)
    if train_df is None:
        print("[TRAIN] Split failed (too little data after cleaning).")
        return

    # Fit model + calibration
    model = fit_catboost_with_platt(train_df, valid_df)

    # Quick eval (diagnostic)
    metrics = {
        "train": evaluate(model, train_df),
        "valid": evaluate(model, valid_df),
        "test":  evaluate(model, test_df)
    }
    print("[TRAIN] Metrics:", json.dumps(metrics))

    # Derive league-aware policy on validation
    policy = derive_policy(valid_df, model, TARGET_PRECISION)
    print("[TRAIN] Policy:", json.dumps(policy))

    # Persist model + policy
    meta = {
        "features": FEATURES_ORDER,
        "feature_order": FEATURES_ORDER,
        "trained_at": datetime.now(TZ).isoformat(),
        "metrics": metrics
    }
    await save_model_and_policy(model, meta, policy)
    print("[TRAIN] Model saved and policy updated.")

# Allow running standalone
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
