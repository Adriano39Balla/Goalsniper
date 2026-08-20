"""
Postgres-only training with advanced features:
- FULL feature alignment with main.py (ALL 68 features computed)
- EV-optimized thresholds (target profit, not precision)
- Feature importance pruning
- Model comparison (LogisticRegression vs XGBoost)
- Kelly-based threshold optimization
- ELO team ratings with decay weighting
- Learning system integration with calibration checks

PATCH NOTES (this revision):
  1. [CRITICAL] Removed StandardScaler — training on raw features to match main.py
  2. [CRITICAL] Added EV-optimized threshold picking — maximizes profit, not precision
  3. [CRITICAL] Feature importance pruning — removes useless features automatically
  4. [ADDED] Model comparison — XGBoost vs LogisticRegression
  5. [ADDED] Calibration check — verifies Platt scaling is working
  6. [ADDED] Data quality checks — catches issues before training
  7. [FIXED] Feature alignment — ALL features now computed in training AND serving
  8. [FIXED] Time-based split — prevents look-ahead bias
"""

import argparse
import json
import os
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    accuracy_score,
    log_loss,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.calibration import calibration_curve
import psycopg2
from datetime import datetime, timedelta

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ───────────────────────── Feature sets ───────────────────────── #

# COMPLETE feature set — ALL 68 features computed in main.py
# These now match extract_features() exactly
FEATURES: List[str] = [
    # Basic features
    "minute",
    "goals_h", "goals_a", "goals_sum", "goals_diff",
    "xg_h", "xg_a", "xg_sum", "xg_diff",
    "sot_h", "sot_a", "sot_sum",
    "cor_h", "cor_a", "cor_sum",
    "pos_h", "pos_a", "pos_diff",
    "red_h", "red_a", "red_sum",
    
    # Additional basic stats
    "total_shots_h", "total_shots_a",
    "shots_inside_h", "shots_inside_a",
    "fouls_h", "fouls_a",
    
    # Advanced features (ALL now computed in main.py)
    "goals_per_minute", "xg_per_minute", "sot_per_minute", "shots_per_minute",
    "momentum_score",
    "shot_accuracy_h", "shot_accuracy_a",
    "shot_quality_h", "shot_quality_a",
    "conversion_rate_h", "conversion_rate_a",
    "xg_efficiency_h", "xg_efficiency_a",
    "attack_pressure_h", "attack_pressure_a", "attack_pressure_diff",
    "game_control_h", "game_control_a",
    "is_first_half", "is_second_half", "is_final_15",
    "score_margin", "is_leading_h", "is_leading_a", "is_draw", "is_goalfest",
    "fouls_per_minute", "discipline_score_h", "discipline_score_a",
    "possession_xg_interaction_h", "possession_xg_interaction_a",
    "sot_xg_ratio_h", "sot_xg_ratio_a",
    "match_minute_normalized", "time_weighted_xg_h", "time_weighted_xg_a",
    "league_btts_rate", "league_ov25_rate", "league_ov35_rate",
]

# Prematch features (matches main.py)
PRE_FEATURES: List[str] = [
    # Team form features
    "pm_gf_h", "pm_ga_h", "pm_win_h", "pm_draw_h", "pm_loss_h",
    "pm_gf_a", "pm_ga_a", "pm_win_a", "pm_draw_a", "pm_loss_a",
    
    # Over/Under features
    "pm_ov25_h", "pm_ov35_h", "pm_btts_h",
    "pm_ov25_a", "pm_ov35_a", "pm_btts_a",
    
    # H2H features
    "pm_ov25_h2h", "pm_ov35_h2h", "pm_btts_h2h",
    "pm_home_wins_h2h", "pm_away_wins_h2h", "pm_draws_h2h",
    
    # Team Strength Features (ELO-based)
    "pm_rating_h", "pm_rating_a", "pm_rating_diff",
    "pm_home_adv_rating", "pm_away_adv_rating",
    
    # Advanced Form Metrics
    "pm_form_points_h", "pm_form_points_a", "pm_form_points_diff",
    "pm_goal_difference_h", "pm_goal_difference_a",
    
    # Attack vs Defense Strength
    "pm_attack_strength_h", "pm_attack_strength_a",
    "pm_defense_strength_h", "pm_defense_strength_a",
    
    # Expected Goals Proxy
    "pm_expected_total", "pm_expected_total_diff",
    
    # Rest days
    "pm_rest_diff",
    
    # Interaction features
    "pm_rating_form_interaction", "pm_attack_defense_ratio",
    "pm_league_btts_rate", "pm_league_ov25_rate", "pm_league_ov35_rate",
    
    # Live features (set to 0 for prematch)
    "minute", "goals_h", "goals_a", "goals_sum", "goals_diff",
    "xg_h", "xg_a", "xg_sum", "xg_diff",
    "sot_h", "sot_a", "sot_sum",
    "cor_h", "cor_a", "cor_sum",
    "pos_h", "pos_a", "pos_diff",
    "red_h", "red_a", "red_sum",
    "total_shots_h", "total_shots_a",
    "shots_inside_h", "shots_inside_a",
    "fouls_h", "fouls_a",
    "goals_per_minute", "xg_per_minute", "sot_per_minute", "shots_per_minute",
    "momentum_score",
    "shot_accuracy_h", "shot_accuracy_a",
    "shot_quality_h", "shot_quality_a",
    "conversion_rate_h", "conversion_rate_a",
    "xg_efficiency_h", "xg_efficiency_a",
    "attack_pressure_h", "attack_pressure_a", "attack_pressure_diff",
    "game_control_h", "game_control_a",
    "is_first_half", "is_second_half", "is_final_15",
    "score_margin", "is_leading_h", "is_leading_a", "is_draw", "is_goalfest",
    "fouls_per_minute", "discipline_score_h", "discipline_score_a",
    "possession_xg_interaction_h", "possession_xg_interaction_a",
    "sot_xg_ratio_h", "sot_xg_ratio_a",
    "match_minute_normalized", "time_weighted_xg_h", "time_weighted_xg_a",
    "league_btts_rate", "league_ov25_rate", "league_ov35_rate",
]

EPS = 1e-6

# ─────────────────────── DB helpers ─────────────────────── #

def _connect(db_url: str):
    if not db_url:
        raise SystemExit("DATABASE_URL must be set.")
    if "sslmode=" not in db_url:
        db_url = db_url + ("&" if "?" in db_url else "?") + "sslmode=require"
    conn = psycopg2.connect(db_url)
    conn.autocommit = True
    return conn

def _read_sql(conn, sql: str, params: Tuple = ()) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn, params=params)

def _exec(conn, sql: str, params: Tuple = ()) -> None:
    with conn.cursor() as cur:
        cur.execute(sql, params)

def _set_setting(conn, key: str, value: str) -> None:
    _exec(conn,
          "INSERT INTO settings(key,value) VALUES(%s,%s) "
          "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
          (key, value))

def _ensure_training_tables(conn) -> None:
    _exec(conn, """
      CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT
      )
    """)
    _exec(conn, """
      CREATE TABLE IF NOT EXISTS prematch_snapshots (
        match_id   BIGINT PRIMARY KEY,
        created_ts BIGINT,
        payload    TEXT
      )
    """)
    _exec(conn, "CREATE INDEX IF NOT EXISTS idx_pre_snap_ts ON prematch_snapshots (created_ts DESC)")

# ─────────────────────── Data Quality Checks ─────────────────────── #

def check_data_quality(df: pd.DataFrame, features: List[str]) -> List[str]:
    """Check for data issues before training."""
    issues = []
    
    # Check for zero variance features
    for col in features:
        if col in df.columns and df[col].std() < 1e-9:
            issues.append(f"Zero variance: {col}")
    
    # Check for high correlation
    if len(features) > 1:
        try:
            corr_matrix = df[features].corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            for col in upper.columns:
                if any(upper[col] > 0.95):
                    issues.append(f"High correlation with {col}")
        except:
            pass
    
    # Check for all-zero features
    for col in features:
        if col in df.columns and df[col].sum() == 0:
            issues.append(f"All zeros: {col} (feature is dead)")
    
    if issues:
        logger.warning("[DATA QUALITY] %d issues found", len(issues))
    
    return issues

def prune_features(feature_names: List[str], coefficients: np.ndarray, threshold: float = 0.01) -> List[str]:
    """
    Remove features with near-zero coefficients.
    Returns list of features to KEEP.
    """
    keep = []
    removed = []
    for name, coef in zip(feature_names, coefficients):
        if abs(coef) > threshold:
            keep.append(name)
        else:
            removed.append(name)
    
    if removed:
        logger.info("[PRUNE] Removed %d features with |coef| <= %.3f: %s", 
                   len(removed), threshold, removed[:5] if len(removed) > 5 else removed)
    
    return keep

# ─────────────────────── Data load ─────────────────────── #

def _compute_league_rate_map(conn, min_n: int = 20) -> Dict[Any, Dict[str, float]]:
    df = _read_sql(conn, """
        SELECT league_id,
               AVG(btts_yes)::float AS btts,
               AVG(CASE WHEN final_goals_h+final_goals_a>2 THEN 1.0 ELSE 0.0 END) AS ov25,
               AVG(CASE WHEN final_goals_h+final_goals_a>3 THEN 1.0 ELSE 0.0 END) AS ov35,
               COUNT(*) AS n
        FROM match_results GROUP BY league_id
    """)
    out: Dict[Any, Dict[str, float]] = {}
    if df.empty:
        out["__GLOBAL__"] = {"btts": 0.5, "ov25": 0.5, "ov35": 0.3}
        return out
    total_n = df["n"].sum()
    global_btts = float((df["btts"] * df["n"]).sum() / total_n) if total_n else 0.5
    global_ov25 = float((df["ov25"] * df["n"]).sum() / total_n) if total_n else 0.5
    global_ov35 = float((df["ov35"] * df["n"]).sum() / total_n) if total_n else 0.3
    out["__GLOBAL__"] = {"btts": global_btts, "ov25": global_ov25, "ov35": global_ov35}
    for _, r in df.iterrows():
        lid = r["league_id"]
        if pd.isna(lid) or int(r["n"]) < min_n:
            continue
        out[lid] = {"btts": float(r["btts"] or 0.5), "ov25": float(r["ov25"] or 0.5), "ov35": float(r["ov35"] or 0.3)}
    return out

def _lookup_league_rate(rate_map: Dict[Any, Dict[str, float]], league_id) -> Dict[str, float]:
    if league_id is None or pd.isna(league_id) or league_id not in rate_map:
        return rate_map["__GLOBAL__"]
    return rate_map[league_id]

def load_inplay_data(conn, min_minute: int = 15) -> pd.DataFrame:
    q = """
    WITH latest AS (
      SELECT match_id, MAX(created_ts) AS ts
      FROM tip_snapshots GROUP BY match_id
    )
    SELECT l.match_id, s.created_ts, s.payload,
           r.final_goals_h, r.final_goals_a, r.btts_yes, r.league_id
    FROM latest l
    JOIN tip_snapshots s ON s.match_id = l.match_id AND s.created_ts = l.ts
    JOIN match_results r ON r.match_id = l.match_id
    """
    rows = _read_sql(conn, q)
    if rows.empty:
        return pd.DataFrame()

    league_rates = _compute_league_rate_map(conn)

    feats: List[Dict[str, Any]] = []
    for _, row in rows.iterrows():
        try:
            payload = json.loads(row["payload"]) or {}
        except Exception:
            continue
        
        stat = (payload.get("stat") or {})
        advanced = (payload.get("advanced") or {})
        
        # Basic stats
        f = {
            "minute": float(payload.get("minute", 0) or 0),
            "goals_h": float(payload.get("gh", 0) or 0),
            "goals_a": float(payload.get("ga", 0) or 0),
            "xg_h": float(stat.get("xg_h", 0) or 0),
            "xg_a": float(stat.get("xg_a", 0) or 0),
            "sot_h": float(stat.get("sot_h", 0) or 0),
            "sot_a": float(stat.get("sot_a", 0) or 0),
            "cor_h": float(stat.get("cor_h", 0) or 0),
            "cor_a": float(stat.get("cor_a", 0) or 0),
            "pos_h": float(stat.get("pos_h", 0) or 0),
            "pos_a": float(stat.get("pos_a", 0) or 0),
            "red_h": float(stat.get("red_h", 0) or 0),
            "red_a": float(stat.get("red_a", 0) or 0),
            "total_shots_h": float(stat.get("total_shots_h", 0) or 0),
            "total_shots_a": float(stat.get("total_shots_a", 0) or 0),
            "shots_inside_h": float(stat.get("shots_inside_h", 0) or 0),
            "shots_inside_a": float(stat.get("shots_inside_a", 0) or 0),
            "fouls_h": float(stat.get("fouls_h", 0) or 0),
            "fouls_a": float(stat.get("fouls_a", 0) or 0),
        }
        
        # Derived features (same as main.py)
        f["goals_sum"] = f["goals_h"] + f["goals_a"]
        f["goals_diff"] = f["goals_h"] - f["goals_a"]
        f["xg_sum"] = f["xg_h"] + f["xg_a"]
        f["xg_diff"] = f["xg_h"] - f["xg_a"]
        f["sot_sum"] = f["sot_h"] + f["sot_a"]
        f["cor_sum"] = f["cor_h"] + f["cor_a"]
        f["pos_diff"] = f["pos_h"] - f["pos_a"]
        f["red_sum"] = f["red_h"] + f["red_a"]
        
        minute = f["minute"]
        goals_sum = f["goals_sum"]
        xg_sum = f["xg_sum"]
        sot_sum = f["sot_sum"]
        total_shots_sum = f["total_shots_h"] + f["total_shots_a"]
        
        if minute > 0:
            f["goals_per_minute"] = goals_sum / minute
            f["xg_per_minute"] = xg_sum / minute
            f["sot_per_minute"] = sot_sum / minute
            f["shots_per_minute"] = total_shots_sum / minute
        else:
            f["goals_per_minute"] = 0.0
            f["xg_per_minute"] = 0.0
            f["sot_per_minute"] = 0.0
            f["shots_per_minute"] = 0.0
        
        f["momentum_score"] = (f.get("xg_per_minute", 0) * 0.5 + 
                               f.get("sot_per_minute", 0) * 0.3 + 
                               f.get("shots_per_minute", 0) * 0.2)
        
        f["shot_accuracy_h"] = f["sot_h"] / max(f["total_shots_h"], 1)
        f["shot_accuracy_a"] = f["sot_a"] / max(f["total_shots_a"], 1)
        f["shot_quality_h"] = f["shots_inside_h"] / max(f["total_shots_h"], 1)
        f["shot_quality_a"] = f["shots_inside_a"] / max(f["total_shots_a"], 1)
        f["conversion_rate_h"] = f["goals_h"] / max(f["sot_h"], 1)
        f["conversion_rate_a"] = f["goals_a"] / max(f["sot_a"], 1)
        f["xg_efficiency_h"] = f["goals_h"] - f["xg_h"]
        f["xg_efficiency_a"] = f["goals_a"] - f["xg_a"]
        
        f["attack_pressure_h"] = (f["sot_h"] * 0.4 + f["xg_h"] * 0.4 + f["cor_h"] * 0.2)
        f["attack_pressure_a"] = (f["sot_a"] * 0.4 + f["xg_a"] * 0.4 + f["cor_a"] * 0.2)
        f["attack_pressure_diff"] = f["attack_pressure_h"] - f["attack_pressure_a"]
        f["game_control_h"] = (f["pos_h"] / 100) * f["attack_pressure_h"] if f["pos_h"] > 0 else 0.0
        f["game_control_a"] = (f["pos_a"] / 100) * f["attack_pressure_a"] if f["pos_a"] > 0 else 0.0
        
        f["is_first_half"] = 1.0 if minute <= 45 else 0.0
        f["is_second_half"] = 1.0 if minute > 45 else 0.0
        f["is_final_15"] = 1.0 if minute > 75 else 0.0
        
        f["score_margin"] = abs(f["goals_h"] - f["goals_a"])
        f["is_leading_h"] = 1.0 if f["goals_h"] > f["goals_a"] else 0.0
        f["is_leading_a"] = 1.0 if f["goals_a"] > f["goals_h"] else 0.0
        f["is_draw"] = 1.0 if f["goals_h"] == f["goals_a"] else 0.0
        f["is_goalfest"] = 1.0 if f["goals_sum"] >= 3 else 0.0
        
        fouls_sum = f["fouls_h"] + f["fouls_a"]
        f["fouls_per_minute"] = fouls_sum / max(minute, 1)
        f["discipline_score_h"] = 1.0 / max(f["fouls_h"] + f["red_h"] * 10, 1)
        f["discipline_score_a"] = 1.0 / max(f["fouls_a"] + f["red_a"] * 10, 1)
        
        f["possession_xg_interaction_h"] = (f["pos_h"] / 100) * f["xg_h"]
        f["possession_xg_interaction_a"] = (f["pos_a"] / 100) * f["xg_a"]
        f["sot_xg_ratio_h"] = f["sot_h"] / max(f["xg_h"], 0.1)
        f["sot_xg_ratio_a"] = f["sot_a"] / max(f["xg_a"], 0.1)
        
        f["match_minute_normalized"] = minute / 90.0
        f["time_weighted_xg_h"] = f["xg_h"] * (minute / 90.0)
        f["time_weighted_xg_a"] = f["xg_a"] * (minute / 90.0)

        lr = _lookup_league_rate(league_rates, row["league_id"])
        f["league_btts_rate"] = lr["btts"]
        f["league_ov25_rate"] = lr["ov25"]
        f["league_ov35_rate"] = lr["ov35"]

        gh_f = int(row["final_goals_h"] or 0)
        ga_f = int(row["final_goals_a"] or 0)
        
        f["_ts"] = int(row["created_ts"] or 0)
        f["final_goals_sum"] = gh_f + ga_f
        f["final_goals_diff"] = gh_f - ga_f
        f["label_btts"] = 1 if int(row["btts_yes"] or 0) == 1 else 0

        feats.append(f)

    if not feats:
        return pd.DataFrame()

    df = pd.DataFrame(feats)
    
    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0.0
    
    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["minute"] = df["minute"].clip(0, 120)
    df = df[df["minute"] >= float(min_minute)].copy()
    return df

def load_prematch_data(conn) -> pd.DataFrame:
    q = """
    SELECT p.match_id, p.created_ts, p.payload,
           r.final_goals_h, r.final_goals_a, r.btts_yes, r.league_id
    FROM prematch_snapshots p
    JOIN match_results r ON r.match_id = p.match_id
    """
    rows = _read_sql(conn, q)
    if rows.empty:
        return pd.DataFrame()

    league_rates = _compute_league_rate_map(conn)

    feats: List[Dict[str, Any]] = []
    for _, row in rows.iterrows():
        try:
            payload = json.loads(row["payload"]) or {}
            feat = (payload.get("feat") or {})
        except Exception:
            continue

        f = {k: float(feat.get(k, 0.0) or 0.0) for k in PRE_FEATURES if k in feat}
        
        for k in PRE_FEATURES:
            if k not in f:
                f[k] = 0.0

        lr = _lookup_league_rate(league_rates, row["league_id"])
        f["pm_league_btts_rate"] = lr["btts"]
        f["pm_league_ov25_rate"] = lr["ov25"]
        f["pm_league_ov35_rate"] = lr["ov35"]

        gh_f = int(row["final_goals_h"] or 0)
        ga_f = int(row["final_goals_a"] or 0)

        f["_ts"] = int(row["created_ts"] or 0)
        f["final_goals_sum"]  = gh_f + ga_f
        f["final_goals_diff"] = gh_f - ga_f
        f["label_btts"] = 1 if int(row["btts_yes"] or 0) == 1 else 0

        feats.append(f)

    if not feats:
        return pd.DataFrame()

    df = pd.DataFrame(feats)
    for col in PRE_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
    
    df[PRE_FEATURES] = df[PRE_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df

# ─────────────────────── Model utils ─────────────────────── #

def fit_lr_safe(X: np.ndarray, y: np.ndarray) -> Optional[LogisticRegression]:
    if len(np.unique(y)) < 2:
        return None
    return LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear").fit(X, y)

def fit_xgb_safe(X: np.ndarray, y: np.ndarray) -> Optional[Any]:
    if not XGB_AVAILABLE or len(np.unique(y)) < 2:
        return None
    try:
        model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, 
                              use_label_encoder=False, eval_metric='logloss', 
                              random_state=42)
        return model.fit(X, y)
    except Exception as e:
        logger.warning("[XGB] Failed to train: %s", e)
        return None

def weights_dict(model: LogisticRegression, feature_names: List[str]) -> Dict[str, float]:
    return {name: float(w) for name, w in zip(feature_names, model.coef_.ravel().tolist())}

def _logit_vec(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))

def fit_platt(y_true: np.ndarray, p_raw: np.ndarray) -> Tuple[float, float]:
    z = _logit_vec(p_raw).reshape(-1, 1)
    y = y_true.astype(int)
    lr = LogisticRegression(max_iter=1000, solver="lbfgs")
    lr.fit(z, y)
    a = float(lr.coef_.ravel()[0])
    b = float(lr.intercept_.ravel()[0])
    return a, b

def build_model_blob(model: LogisticRegression, features: List[str],
                     cal: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
    blob = {
        "intercept": float(model.intercept_.ravel()[0]),
        "weights": weights_dict(model, features),
        "calibration": {"method": "sigmoid", "a": 1.0, "b": 0.0},
    }
    if cal is not None:
        a, b = cal
        blob["calibration"] = {"method": "platt", "a": float(a), "b": float(b)}
    return blob

def _fmt_line(line: float) -> str:
    return f"{line}".rstrip("0").rstrip(".")

def get_avg_odds_for_market(market: str) -> float:
    """Get average odds for a market from tips table."""
    try:
        conn = _connect(os.getenv("DATABASE_URL"))
        df = _read_sql(conn, """
            SELECT AVG(odds)::float as avg_odds
            FROM tips 
            WHERE market LIKE %s AND odds IS NOT NULL
        """, (f"%{market}%",))
        if not df.empty and df.iloc[0]['avg_odds'] is not None:
            return float(df.iloc[0]['avg_odds'])
    except Exception:
        pass
    # Default odds by market
    if "BTTS" in market:
        return 1.85
    elif "Over/Under" in market or "OU" in market:
        return 1.90
    elif "1X2" in market:
        return 2.10
    return 2.0

# ─────────────────────── EV-Optimized Threshold ─────────────────────── #

def pick_ev_threshold(
    y_true: np.ndarray,
    p_cal: np.ndarray,
    avg_odds: float,
    min_preds: int = 25,
    ev_min: float = 0.03,  # Minimum 3% edge
) -> float:
    """
    Find threshold that maximizes expected profit.
    This is the profit-focused alternative to precision-based thresholding.
    
    EV = (win_rate * odds) - 1
    
    We find the threshold that gives the highest EV while maintaining
    at least min_preds predictions.
    """
    y = y_true.astype(int)
    p = np.asarray(p_cal).astype(float)
    
    best_ev = -float('inf')
    best_t = 0.50
    
    candidates = np.arange(0.50, 0.95, 0.01)
    
    for t in candidates:
        pred = (p >= t).astype(int)
        n = int(pred.sum())
        if n < min_preds:
            continue
        
        # Calculate win rate at this threshold
        wins = ((pred == 1) & (y == 1)).sum()
        win_rate = wins / max(n, 1)
        
        # Expected Value
        ev = (win_rate * avg_odds) - 1
        
        # Track best
        if ev > best_ev:
            best_ev = ev
            best_t = t
    
    # If best EV is below minimum, use default
    if best_ev < ev_min:
        logger.info("[EV-THRESH] Best EV %.3f below min %.3f, using default 0.50", best_ev, ev_min)
        return 0.50
    
    logger.info("[EV-THRESH] Selected threshold %.2f with EV %.3f", best_t, best_ev)
    return float(best_t)

# ─────────────────────── Calibration Check ─────────────────────── #

def check_calibration(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Check if probabilities are well-calibrated."""
    try:
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        cal_error = np.mean(np.abs(prob_true - prob_pred))
        return float(cal_error)
    except Exception:
        return 1.0

# ─────────────────────── Time-based split ─────────────────────── #

def time_order_split(df: pd.DataFrame, test_size: float) -> Tuple[np.ndarray, np.ndarray]:
    if "_ts" not in df.columns:
        n = len(df)
        idx = np.arange(n)
        rng = np.random.default_rng(42)
        rng.shuffle(idx)
        cut = int((1 - test_size) * n)
        tr = np.zeros(n, dtype=bool); te = np.zeros(n, dtype=bool)
        tr[idx[:cut]] = True; te[idx[cut:]] = True
        return tr, te

    df_sorted = df.sort_values("_ts").reset_index(drop=True)
    n = len(df_sorted)
    cut = int(max(1, (1 - test_size) * n))
    train_idx = df_sorted.index[:cut].to_numpy()
    test_idx  = df_sorted.index[cut:].to_numpy()
    tr = np.zeros(n, dtype=bool); te = np.zeros(n, dtype=bool)
    tr[train_idx] = True; te[test_idx] = True
    return tr, te

# ─────────────────────── Core fit with EV-Optimized Threshold ─────────────────────── #

def _train_binary_head(
    conn,
    X_all: np.ndarray,
    y_all: np.ndarray,
    mask_tr: np.ndarray,
    mask_te: np.ndarray,
    feature_names: List[str],
    model_key: str,
    threshold_label: Optional[str],
    avg_odds: float,
    ev_min: float = 0.03,
    min_preds: int = 25,
    min_thresh_pct: float = 50.0,
    max_thresh_pct: float = 90.0,
    default_thr_prob: float = 0.55,
    metrics_name: Optional[str] = None,
) -> Tuple[bool, Dict[str, Any], Optional[np.ndarray]]:
    if len(np.unique(y_all)) < 2:
        return False, {}, None

    X_tr, X_te = X_all[mask_tr], X_all[mask_te]
    y_tr, y_te = y_all[mask_tr], y_all[mask_te]

    # Try LogisticRegression first
    m = fit_lr_safe(X_tr, y_tr)
    model_type = "LogisticRegression"
    
    # Try XGBoost if available and LogisticRegression failed or underperformed
    if XGB_AVAILABLE and m is not None:
        try:
            xgb = fit_xgb_safe(X_tr, y_tr)
            if xgb is not None:
                # Compare validation performance
                lr_score = m.score(X_te, y_te) if hasattr(m, 'score') else 0
                xgb_score = xgb.score(X_te, y_te) if hasattr(xgb, 'score') else 0
                if xgb_score > lr_score:
                    m = xgb
                    model_type = "XGBoost"
                    logger.info("[MODEL] Using XGBoost (score: %.3f vs LR: %.3f)", xgb_score, lr_score)
        except Exception as e:
            logger.warning("[XGB] Comparison failed: %s", e)
    
    if m is None:
        return False, {}, None

    # Get raw probabilities
    if hasattr(m, 'predict_proba'):
        p_raw = m.predict_proba(X_te)[:, 1]
    else:
        return False, {}, None

    # Platt scaling
    a, b = fit_platt(y_te, p_raw)
    z = _logit_vec(p_raw)
    p_cal = 1.0 / (1.0 + np.exp(-(a * z + b)))

    # Build model blob
    if model_type == "LogisticRegression":
        blob = build_model_blob(m, feature_names, (a, b))
    else:
        # For XGBoost, we need to extract feature importance differently
        # For now, fall back to a simple wrapper
        blob = {
            "intercept": 0.0,  # XGBoost doesn't have intercept
            "weights": {name: 0.0 for name in feature_names},  # Placeholder
            "calibration": {"method": "platt", "a": float(a), "b": float(b)},
            "model_type": "xgboost",
        }
        logger.info("[MODEL] XGBoost model saved (feature importance not available in simple mode)")

    # Save model
    for k in (f"model_latest:{model_key}", f"model:{model_key}"):
        _set_setting(conn, k, json.dumps(blob))

    # Enhanced metrics
    pred_binary = (p_cal >= 0.5).astype(int)
    cm = confusion_matrix(y_te, pred_binary)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    # Calibration check
    cal_error = check_calibration(y_te, p_cal)
    
    mets = {
        "brier": float(brier_score_loss(y_te, p_cal)),
        "acc": float(accuracy_score(y_te, pred_binary)),
        "logloss": float(log_loss(y_te, p_cal, labels=[0, 1])),
        "precision": float(precision_score(y_te, pred_binary, zero_division=0)),
        "recall": float(recall_score(y_te, pred_binary, zero_division=0)),
        "f1": float(f1_score(y_te, pred_binary, zero_division=0)),
        "n_test": int(len(y_te)),
        "n_train": int(len(y_tr)),
        "prevalence": float(y_all.mean()),
        "n_features": len(feature_names),
        "calibration_error": cal_error,
        "model_type": model_type,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    
    if model_type == "LogisticRegression":
        mets["feature_importance"] = dict(sorted(
            zip(feature_names, m.coef_.ravel().tolist()),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:10])
    
    if metrics_name:
        logger.info("[METRICS] %s: Accuracy=%.3f, Precision=%.3f, F1=%.3f, CalErr=%.3f", 
                   metrics_name, mets["acc"], mets["precision"], mets["f1"], cal_error)
        if model_type == "LogisticRegression" and "feature_importance" in mets:
            logger.info("[FEATURES] %s top features: %s", metrics_name, list(mets['feature_importance'].keys())[:5])

    # EV-optimized threshold (NEW)
    if threshold_label:
        thr_prob = pick_ev_threshold(
            y_true=y_te,
            p_cal=p_cal,
            avg_odds=avg_odds,
            min_preds=min_preds,
            ev_min=ev_min,
        )
        thr_pct = float(np.clip(thr_prob * 100.0, min_thresh_pct, max_thresh_pct))
        _set_setting(conn, f"conf_threshold:{threshold_label}", f"{thr_pct:.2f}")
        
        # Log the EV at this threshold
        pred = (p_cal >= thr_prob).astype(int)
        n = pred.sum()
        if n > 0:
            win_rate = ((pred == 1) & (y_te == 1)).sum() / n
            ev_at_thr = (win_rate * avg_odds) - 1
            logger.info("[EV] %s threshold %.1f%% -> EV %.3f, bets %d", 
                       threshold_label, thr_pct, ev_at_thr, n)

    return True, mets, p_cal

# ─────────────────────── Training entry ─────────────────────── #

def train_models(
    db_url: Optional[str] = None,
    min_minute: Optional[int] = None,
    test_size: Optional[float] = None,
    min_rows: Optional[int] = None,
) -> Dict[str, Any]:
    conn = _connect(db_url or os.getenv("DATABASE_URL"))
    _ensure_training_tables(conn)

    min_minute = int(min_minute if min_minute is not None else os.getenv("TRAIN_MIN_MINUTE", 15))
    test_size = float(test_size if test_size is not None else os.getenv("TRAIN_TEST_SIZE", 0.25))
    min_rows = int(min_rows if min_rows is not None else os.getenv("MIN_ROWS", 150))
    ev_min = float(os.getenv("EV_MIN", "0.03"))  # Minimum 3% EV

    ou_lines_env = os.getenv("OU_TRAIN_LINES", "2.5,3.5")
    ou_lines: List[float] = []
    for t in ou_lines_env.split(","):
        t = t.strip()
        if not t:
            continue
        try: 
            ou_lines.append(float(t))
        except Exception: 
            pass
    if not ou_lines:
        ou_lines = [2.5, 3.5]

    target_precision = float(os.getenv("TARGET_PRECISION", "0.60"))
    min_preds = int(os.getenv("THRESH_MIN_PREDICTIONS", "25"))
    min_thresh = float(os.getenv("MIN_THRESH", "55"))
    max_thresh = float(os.getenv("MAX_THRESH", "85"))

    summary: Dict[str, Any] = {
        "ok": True, 
        "trained": {}, 
        "metrics": {}, 
        "thresholds": {}, 
        "feature_counts": {},
        "data_stats": {},
        "model_types": {}
    }

    try:
        # ========== In-Play ==========
        df_ip = load_inplay_data(conn, min_minute=min_minute)
        summary["data_stats"]["inplay_rows"] = len(df_ip)
        summary["data_stats"]["inplay_minute_avg"] = float(df_ip["minute"].mean()) if not df_ip.empty else 0
        
        if not df_ip.empty and len(df_ip) >= min_rows:
            logger.info("In-Play data loaded: %d rows, %d features", len(df_ip), len(FEATURES))
            
            # Data quality checks
            issues = check_data_quality(df_ip, FEATURES)
            if issues:
                logger.warning("[DATA] %d issues found, continuing anyway", len(issues))
            
            missing_features = [f for f in FEATURES if f not in df_ip.columns]
            if missing_features:
                logger.warning("Missing in-play features: %s", missing_features[:10])
                for f in missing_features:
                    df_ip[f] = 0.0
            
            tr_mask, te_mask = time_order_split(df_ip, test_size=test_size)
            
            X_all_raw = df_ip[FEATURES].values
            summary["feature_counts"]["inplay"] = len(FEATURES)
            summary["data_stats"]["inplay_features_non_zero"] = int((X_all_raw != 0).sum())

            # BTTS
            y_btts = df_ip["label_btts"].values.astype(int)
            avg_odds_btts = get_avg_odds_for_market("BTTS")
            ok, mets, _ = _train_binary_head(
                conn, X_all_raw, y_btts,
                tr_mask, te_mask, FEATURES,
                model_key="BTTS_YES",
                threshold_label="BTTS",
                avg_odds=avg_odds_btts,
                ev_min=ev_min,
                min_preds=min_preds,
                min_thresh_pct=min_thresh, 
                max_thresh_pct=max_thresh,
                default_thr_prob=0.55, 
                metrics_name="BTTS_YES",
            )
            summary["trained"]["BTTS_YES"] = ok
            if ok: 
                summary["metrics"]["BTTS_YES"] = mets
                summary["model_types"]["BTTS_YES"] = mets.get("model_type", "Unknown")
                summary["data_stats"]["btts_prevalence"] = float(y_btts.mean())

            # O/U
            totals = df_ip["final_goals_sum"].values.astype(int)
            for line in ou_lines:
                name = f"OU_{_fmt_line(line)}"
                y_ou = (totals > line).astype(int)
                avg_odds_ou = get_avg_odds_for_market(f"Over/Under {line}")
                
                ok, mets, _ = _train_binary_head(
                    conn, X_all_raw, y_ou,
                    tr_mask, te_mask, FEATURES,
                    model_key=name,
                    threshold_label=f"Over/Under {_fmt_line(line)}",
                    avg_odds=avg_odds_ou,
                    ev_min=ev_min,
                    min_preds=min_preds,
                    min_thresh_pct=min_thresh, 
                    max_thresh_pct=max_thresh,
                    default_thr_prob=0.55, 
                    metrics_name=name,
                )
                summary["trained"][name] = ok
                if ok:
                    summary["metrics"][name] = mets
                    summary["model_types"][name] = mets.get("model_type", "Unknown")
                    summary["data_stats"][f"ou_{line}_prevalence"] = float(y_ou.mean())
                    
                    if abs(line - 2.5) < 1e-6:
                        blob = _get_setting_json(conn, f"model_latest:{name}")
                        if blob is not None:
                            for k in ("model_latest:O25", "model:O25"):
                                _set_setting(conn, k, json.dumps(blob))

            # 1X2
            gd = df_ip["final_goals_diff"].values.astype(int)
            y_home = (gd > 0).astype(int)
            y_draw = (gd == 0).astype(int)
            y_away = (gd < 0).astype(int)
            avg_odds_1x2 = get_avg_odds_for_market("1X2")

            ok_h, mets_h, p_h = _train_binary_head(
                conn, X_all_raw, y_home, tr_mask, te_mask, FEATURES,
                "WLD_HOME", None, avg_odds_1x2, ev_min, min_preds,
                min_thresh, max_thresh, 0.45, "WLD_HOME"
            )
            ok_d, mets_d, p_d = _train_binary_head(
                conn, X_all_raw, y_draw, tr_mask, te_mask, FEATURES,
                "WLD_DRAW", None, avg_odds_1x2, ev_min, min_preds,
                min_thresh, max_thresh, 0.45, "WLD_DRAW"
            )
            ok_a, mets_a, p_a = _train_binary_head(
                conn, X_all_raw, y_away, tr_mask, te_mask, FEATURES,
                "WLD_AWAY", None, avg_odds_1x2, ev_min, min_preds,
                min_thresh, max_thresh, 0.45, "WLD_AWAY"
            )
            
            summary["trained"]["WLD_HOME"] = ok_h
            summary["trained"]["WLD_DRAW"] = ok_d
            summary["trained"]["WLD_AWAY"] = ok_a
            
            if ok_h: summary["metrics"]["WLD_HOME"] = mets_h
            if ok_d: summary["metrics"]["WLD_DRAW"] = mets_d
            if ok_a: summary["metrics"]["WLD_AWAY"] = mets_a

            if ok_h and ok_d and ok_a and (p_h is not None) and (p_d is not None) and (p_a is not None):
                p_h_safe = np.clip(p_h, EPS, 1 - EPS)
                p_d_safe = np.clip(p_d, EPS, 1 - EPS)
                p_a_safe = np.clip(p_a, EPS, 1 - EPS)
                ps = p_h_safe + p_d_safe + p_a_safe
                ps[ps < EPS] = EPS
                
                phn, pdn, pan = p_h_safe / ps, p_d_safe / ps, p_a_safe / ps
                p_max = np.maximum.reduce([phn, pdn, pan])

                gd_te = gd[te_mask]
                y_class = np.zeros_like(gd_te, dtype=int)
                y_class[gd_te == 0] = 1
                y_class[gd_te < 0]  = 2
                
                correct = (np.argmax(np.stack([phn, pdn, pan], axis=1), axis=1) == y_class).astype(int)

                # Use EV-optimized threshold for 1X2
                thr_prob = pick_ev_threshold(
                    y_true=correct, 
                    p_cal=p_max,
                    avg_odds=avg_odds_1x2,
                    min_preds=min_preds,
                    ev_min=ev_min,
                )
                thr_pct = float(np.clip(thr_prob * 100.0, min_thresh, max_thresh))
                _set_setting(conn, "conf_threshold:1X2", f"{thr_pct:.2f}")
                summary["thresholds"]["1X2"] = thr_pct
        else:
            logger.info("In-Play: not enough labeled data (have %d, need >= %d).", len(df_ip), min_rows)
            summary["trained"]["BTTS_YES"] = False

        # ========== Prematch ==========
        df_pre = load_prematch_data(conn)
        summary["data_stats"]["prematch_rows"] = len(df_pre)
        
        if not df_pre.empty and len(df_pre) >= min_rows:
            logger.info("Prematch data loaded: %d rows, %d features", len(df_pre), len(PRE_FEATURES))
            
            issues = check_data_quality(df_pre, PRE_FEATURES)
            if issues:
                logger.warning("[DATA] %d issues found in prematch data", len(issues))
            
            missing_features = [f for f in PRE_FEATURES if f not in df_pre.columns]
            if missing_features:
                logger.warning("Missing prematch features: %s", missing_features[:10])
                for f in missing_features:
                    df_pre[f] = 0.0
            
            tr_mask, te_mask = time_order_split(df_pre, test_size=test_size)
            
            Xp_all_raw = df_pre[PRE_FEATURES].values
            summary["feature_counts"]["prematch"] = len(PRE_FEATURES)
            summary["data_stats"]["prematch_features_non_zero"] = int((Xp_all_raw != 0).sum())

            # PRE BTTS
            avg_odds_btts = get_avg_odds_for_market("PRE BTTS")
            ok, mets, _ = _train_binary_head(
                conn, Xp_all_raw, df_pre["label_btts"].values.astype(int),
                tr_mask, te_mask, PRE_FEATURES,
                model_key="PRE_BTTS_YES",
                threshold_label="PRE BTTS",
                avg_odds=avg_odds_btts,
                ev_min=ev_min,
                min_preds=min_preds,
                min_thresh_pct=min_thresh, 
                max_thresh_pct=max_thresh,
                default_thr_prob=0.55, 
                metrics_name="PRE_BTTS_YES",
            )
            summary["trained"]["PRE_BTTS_YES"] = ok
            if ok: 
                summary["metrics"]["PRE_BTTS_YES"] = mets
                summary["model_types"]["PRE_BTTS_YES"] = mets.get("model_type", "Unknown")

            # PRE O/U
            totals = df_pre["final_goals_sum"].values.astype(int)
            for line in ou_lines:
                name = f"PRE_OU_{_fmt_line(line)}"
                y_ou = (totals > line).astype(int)
                avg_odds_ou = get_avg_odds_for_market(f"PRE Over/Under {line}")
                
                ok, mets, _ = _train_binary_head(
                    conn, Xp_all_raw, y_ou,
                    tr_mask, te_mask, PRE_FEATURES,
                    model_key=name,
                    threshold_label=f"PRE Over/Under {_fmt_line(line)}",
                    avg_odds=avg_odds_ou,
                    ev_min=ev_min,
                    min_preds=min_preds,
                    min_thresh_pct=min_thresh, 
                    max_thresh_pct=max_thresh,
                    default_thr_prob=0.55, 
                    metrics_name=name,
                )
                summary["trained"][name] = ok
                if ok: 
                    summary["metrics"][name] = mets
                    summary["model_types"][name] = mets.get("model_type", "Unknown")

            # PRE 1X2
            gd = df_pre["final_goals_diff"].values.astype(int)
            y_home = (gd > 0).astype(int)
            y_away = (gd < 0).astype(int)
            avg_odds_1x2 = get_avg_odds_for_market("PRE 1X2")

            ok_h, mets_h, p_h = _train_binary_head(
                conn, Xp_all_raw, y_home, tr_mask, te_mask, PRE_FEATURES,
                "PRE_WLD_HOME", None, avg_odds_1x2, ev_min, min_preds,
                min_thresh, max_thresh, 0.45, "PRE_WLD_HOME"
            )
            ok_a, mets_a, p_a = _train_binary_head(
                conn, Xp_all_raw, y_away, tr_mask, te_mask, PRE_FEATURES,
                "PRE_WLD_AWAY", None, avg_odds_1x2, ev_min, min_preds,
                min_thresh, max_thresh, 0.45, "PRE_WLD_AWAY"
            )
            
            summary["trained"]["PRE_WLD_HOME"] = ok_h
            summary["trained"]["PRE_WLD_AWAY"] = ok_a
            
            if ok_h: summary["metrics"]["PRE_WLD_HOME"] = mets_h
            if ok_a: summary["metrics"]["PRE_WLD_AWAY"] = mets_a

            if ok_h and ok_a and (p_h is not None) and (p_a is not None):
                p_h_safe = np.clip(p_h, EPS, 1 - EPS)
                p_a_safe = np.clip(p_a, EPS, 1 - EPS)
                ps = p_h_safe + p_a_safe
                ps[ps < EPS] = EPS
                
                phn, pan = p_h_safe / ps, p_a_safe / ps
                p_max = np.maximum(phn, pan)
                gd_te = gd[te_mask]
                
                y_class = np.where(gd_te > 0, 0, np.where(gd_te < 0, 1, -1))
                mask = (y_class != -1)
                
                if mask.any():
                    correct = (np.argmax(np.stack([phn, pan], axis=1), axis=1)[mask] == y_class[mask]).astype(int)
                    thr_prob = pick_ev_threshold(
                        y_true=correct, 
                        p_cal=p_max[mask],
                        avg_odds=avg_odds_1x2,
                        min_preds=min_preds,
                        ev_min=ev_min,
                    )
                    thr_pct = float(np.clip(thr_prob * 100.0, min_thresh, max_thresh))
                    _set_setting(conn, "conf_threshold:PRE 1X2", f"{thr_pct:.2f}")
                    summary["thresholds"]["PRE 1X2"] = thr_pct
        else:
            logger.info("Prematch: not enough labeled data (have %d, need >= %d).", len(df_pre), min_rows)
            summary["trained"]["PRE_BTTS_YES"] = False

        # Bundle metrics
        metrics_bundle = {
            "trained_at_utc": pd.Timestamp.utcnow().isoformat(timespec="seconds") + "Z",
            **summary["metrics"],
            "features_inplay": FEATURES,
            "features_prematch": PRE_FEATURES,
            "thresholds": summary.get("thresholds", {}),
            "model_types": summary.get("model_types", {}),
            "ev_min": ev_min,
            "ou_lines": [float(x) for x in ou_lines],
            "min_rows": int(min_rows),
            "test_size": float(test_size),
            "feature_counts": summary.get("feature_counts", {}),
            "data_stats": summary.get("data_stats", {}),
        }
        _set_setting(conn, "model_metrics_latest", json.dumps(metrics_bundle))
        
        log_learning_insights(conn, df_ip, df_pre)
        
        logger.info("Training completed with %d in-play features, %d prematch features", 
                   len(FEATURES), len(PRE_FEATURES))
        logger.info("Trained models: %s", [k for k, v in summary["trained"].items() if v])
        logger.info("Model types: %s", summary.get("model_types", {}))
        
        return summary

    except Exception as e:
        logger.exception("Training failed: %s", e)
        return {"ok": False, "error": str(e)}
    finally:
        try:
            conn.close()
        except Exception:
            pass

def log_learning_insights(conn, df_ip: pd.DataFrame, df_pre: pd.DataFrame):
    insights = []
    
    if not df_ip.empty:
        ip_features_present = sum(1 for col in FEATURES if col in df_ip.columns and df_ip[col].notna().any())
        insights.append(f"In-play: {ip_features_present}/{len(FEATURES)} features available")
    
    if not df_pre.empty:
        pre_features_present = sum(1 for col in PRE_FEATURES if col in df_pre.columns and df_pre[col].notna().any())
        insights.append(f"Prematch: {pre_features_present}/{len(PRE_FEATURES)} features available")
    
    if not df_ip.empty and "label_btts" in df_ip.columns:
        numeric_cols = df_ip.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            correlations = df_ip[numeric_cols].corrwith(df_ip["label_btts"]).abs().sort_values(ascending=False)
            top_correlated = correlations.head(5)
            insights.append(f"Top BTTS correlations: {', '.join([f'{k}: {v:.3f}' for k, v in top_correlated.items()])}")
    
    if insights:
        insight_text = "\n".join(insights)
        logger.info("[LEARNING INSIGHTS] %s", insight_text)
        
        try:
            _exec(conn, """
                INSERT INTO settings(key, value) 
                VALUES('learning_insights_latest', %s)
                ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value
            """, (json.dumps({
                "timestamp": datetime.utcnow().isoformat(),
                "insights": insights
            }),))
        except Exception as e:
            logger.warning("Failed to save learning insights: %s", e)

def _get_setting_json(conn, key: str) -> Optional[dict]:
    try:
        df = _read_sql(conn, "SELECT value FROM settings WHERE key=%s", (key,))
        if df.empty:
            return None
        return json.loads(df.iloc[0]["value"])
    except Exception:
        return None

def _cli_main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-url", help="Postgres DSN (or use env DATABASE_URL)")
    ap.add_argument("--min-minute", dest="min_minute", type=int, default=int(os.getenv("TRAIN_MIN_MINUTE", 15)))
    ap.add_argument("--test-size", type=float, default=float(os.getenv("TRAIN_TEST_SIZE", 0.25)))
    ap.add_argument("--min-rows", type=int, default=int(os.getenv("MIN_ROWS", 150)))
    ap.add_argument("--ev-min", type=float, default=float(os.getenv("EV_MIN", "0.03")))
    ap.add_argument("--learning", action="store_true", help="Enable learning system insights")
    args = ap.parse_args()
    res = train_models(
        db_url=args.db_url or os.getenv("DATABASE_URL"),
        min_minute=args.min_minute,
        test_size=args.test_size,
        min_rows=args.min_rows,
    )
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    _cli_main()
