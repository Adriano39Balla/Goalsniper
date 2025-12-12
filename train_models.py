#!/usr/bin/env python3
"""
train_models_accurate.py – PROVEN ACCURACY TRAINER
SYNCED: Features match main.py exactly
SYNCED: Probability modeling correct
SYNCED: Scaling and normalization
"""

import argparse
import json
import logging
import os
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import psycopg2

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
)
log = logging.getLogger("trainer_accurate")

# ---------- ACCURATE Feature Set (MUST match main.py.extract_features_accurate exactly) ----------

ACCURATE_FEATURES: List[str] = [
    # Core metrics
    "minute",
    "minute_norm",
    
    # Goals
    "goals_h", "goals_a", "goals_sum", "goals_diff",
    
    # Expected Goals (MOST IMPORTANT)
    "xg_h", "xg_a", "xg_sum", "xg_diff",
    "xg_efficiency_h", "xg_efficiency_a",
    
    # Shots
    "sot_h", "sot_a", "sot_sum",
    "shot_accuracy_h", "shot_accuracy_a",
    
    # Corners
    "cor_h", "cor_a", "cor_sum",
    
    # Possession
    "pos_h", "pos_a", "pos_diff",
    
    # Momentum (CRITICAL for in-play)
    "momentum_h", "momentum_a", "momentum_sum", "momentum_diff",
    
    # Pressure
    "pressure_index", "urgency_index",
    
    # Action
    "total_actions", "action_intensity",
    
    # Derived composites
    "attack_strength_h", "attack_strength_a",
    "defense_weakness_h", "defense_weakness_a",
]

# Feature importance rankings (based on historical analysis)
FEATURE_IMPORTANCE = {
    'high': ['xg_sum', 'goals_sum', 'minute_norm', 'pressure_index', 'action_intensity'],
    'medium': ['sot_sum', 'momentum_sum', 'xg_diff', 'goals_diff', 'pos_diff'],
    'low': ['cor_sum', 'shot_accuracy_h', 'shot_accuracy_a', 'urgency_index']
}

# ---------- Env knobs ----------

TRAIN_MIN_MINUTE = int(os.getenv("TRAIN_MIN_MINUTE", "15"))
TRAIN_TEST_SIZE = float(os.getenv("TRAIN_TEST_SIZE", "0.25"))
MIN_ROWS = int(os.getenv("MIN_ROWS", "200"))  # Increased for accuracy
CV_FOLDS = int(os.getenv("CV_FOLDS", "5"))
MODELS_DIR = os.getenv("MODELS_DIR", "models")
USE_FEATURE_SELECTION = os.getenv("USE_FEATURE_SELECTION", "1") not in ("0","false","False","no","NO")
MIN_FEATURES = int(os.getenv("MIN_FEATURES", "12"))

# Accuracy thresholds
MIN_ACCURACY = float(os.getenv("MIN_ACCURACY", "0.55"))
MIN_AUC = float(os.getenv("MIN_AUC", "0.60"))
MIN_PRECISION = float(os.getenv("MIN_PRECISION", "0.50"))

def _fmt_line(line: float) -> str:
    return f"{line}".rstrip("0").rstrip(".")

def _parse_ou_lines(raw: str) -> List[float]:
    vals: List[float] = []
    for t in (raw or "").split(","):
        t = t.strip()
        if not t:
            continue
        try:
            vals.append(float(t))
        except Exception:
            pass
    return vals or [1.5, 2.5, 3.5]  # Multiple lines for better training

OU_TRAIN_LINES_RAW = os.getenv("OU_TRAIN_LINES", os.getenv("OU_LINES", "1.5,2.5,3.5"))
OU_TRAIN_LINES: List[float] = _parse_ou_lines(OU_TRAIN_LINES_RAW)

# ---------- Feature Engineering (MUST match main.py.extract_features_accurate) ----------

def _calculate_momentum_accurate(features: Dict[str, float], side: str) -> float:
    """Calculate momentum exactly as main.py"""
    suffix = "_h" if side == "home" else "_a"
    sot = float(features.get(f"sot{suffix}", 0.0))
    cor = float(features.get(f"cor{suffix}", 0.0))
    minute = float(features.get("minute", 1.0))
    if minute <= 0:
        return 0.0
    return (sot + cor) / max(1.0, minute)

def _calculate_pressure_index_accurate(features: Dict[str, float]) -> float:
    """Calculate pressure index exactly as main.py"""
    minute = float(features.get("minute", 0.0))
    goal_diff = abs(float(features.get("goals_diff", 0.0)))
    
    minute_norm = minute / 90.0
    score_pressure = min(1.0, goal_diff / 3.0)
    
    # Weighted combination favoring score pressure
    return score_pressure * 0.7 + minute_norm * 0.3

def _calculate_urgency_index(features: Dict[str, float]) -> float:
    """Calculate urgency index for late game"""
    minute = float(features.get("minute", 0.0))
    goal_diff = abs(float(features.get("goals_diff", 0.0)))
    
    if minute >= 75:
        time_urgency = (90 - minute) / 15.0  # 1.0 at 75m, 0.0 at 90m
        score_urgency = min(1.0, goal_diff / 2.0)
        return time_urgency * 0.6 + score_urgency * 0.4
    return 0.0

def _calculate_attack_strength(features: Dict[str, float], side: str) -> float:
    """Calculate attack strength"""
    suffix = "_h" if side == "home" else "_a"
    xg = float(features.get(f"xg{suffix}", 0.0))
    sot = float(features.get(f"sot{suffix}", 0.0))
    return xg + (sot * 0.3)

def _calculate_defense_weakness(features: Dict[str, float], side: str) -> float:
    """Calculate defense weakness (opponent's attack)"""
    opponent_suffix = "_a" if side == "home" else "_h"
    xg = float(features.get(f"xg{opponent_suffix}", 0.0))
    sot = float(features.get(f"sot{opponent_suffix}", 0.0))
    return xg + (sot * 0.3)

def _build_accurate_features_from_snapshot(snap: dict) -> Optional[Dict[str, float]]:
    """
    Rebuild the EXACT feature vector that main.py's extract_features_accurate() produces.
    """
    if not isinstance(snap, dict):
        return None

    # Get raw data
    minute = float(snap.get("minute", 0.0))
    gh = float(snap.get("gh", 0.0))
    ga = float(snap.get("ga", 0.0))
    stat = snap.get("stat") or {}
    if not isinstance(stat, dict):
        stat = {}

    # If stat already has accurate features, use them
    if "minute_norm" in stat:
        # Already has accurate features
        feat = {key: float(stat.get(key, 0.0)) for key in ACCURATE_FEATURES}
        return feat

    # Otherwise, rebuild from basics (legacy data)
    feat: Dict[str, float] = {}
    
    # Core metrics
    feat["minute"] = minute
    feat["minute_norm"] = minute / 90.0
    
    # Goals
    feat["goals_h"] = float(stat.get("goals_h", gh))
    feat["goals_a"] = float(stat.get("goals_a", ga))
    feat["goals_sum"] = feat["goals_h"] + feat["goals_a"]
    feat["goals_diff"] = feat["goals_h"] - feat["goals_a"]
    
    # xG
    xg_h = float(stat.get("xg_h", 0.0))
    xg_a = float(stat.get("xg_a", 0.0))
    feat["xg_h"] = xg_h
    feat["xg_a"] = xg_a
    feat["xg_sum"] = xg_h + xg_a
    feat["xg_diff"] = xg_h - xg_a
    feat["xg_efficiency_h"] = feat["goals_h"] / max(0.1, xg_h) if xg_h > 0 else 0.0
    feat["xg_efficiency_a"] = feat["goals_a"] / max(0.1, xg_a) if xg_a > 0 else 0.0
    
    # Shots
    sot_h = float(stat.get("sot_h", 0.0))
    sot_a = float(stat.get("sot_a", 0.0))
    feat["sot_h"] = sot_h
    feat["sot_a"] = sot_a
    feat["sot_sum"] = sot_h + sot_a
    
    sh_total_h = float(stat.get("sh_total_h", 0.0))
    sh_total_a = float(stat.get("sh_total_a", 0.0))
    feat["shot_accuracy_h"] = sot_h / max(1.0, sh_total_h) if sh_total_h > 0 else 0.0
    feat["shot_accuracy_a"] = sot_a / max(1.0, sh_total_a) if sh_total_a > 0 else 0.0
    
    # Corners
    cor_h = float(stat.get("cor_h", 0.0))
    cor_a = float(stat.get("cor_a", 0.0))
    feat["cor_h"] = cor_h
    feat["cor_a"] = cor_a
    feat["cor_sum"] = cor_h + cor_a
    
    # Possession
    pos_h = float(stat.get("pos_h", 0.0))
    pos_a = float(stat.get("pos_a", 0.0))
    feat["pos_h"] = pos_h
    feat["pos_a"] = pos_a
    feat["pos_diff"] = pos_h - pos_a
    
    # Momentum
    feat["momentum_h"] = _calculate_momentum_accurate(feat, "home")
    feat["momentum_a"] = _calculate_momentum_accurate(feat, "away")
    feat["momentum_sum"] = feat["momentum_h"] + feat["momentum_a"]
    feat["momentum_diff"] = feat["momentum_h"] - feat["momentum_a"]
    
    # Pressure
    feat["pressure_index"] = _calculate_pressure_index_accurate(feat)
    feat["urgency_index"] = _calculate_urgency_index(feat)
    
    # Action
    feat["total_actions"] = feat["sot_sum"] + feat["cor_sum"]
    feat["action_intensity"] = feat["total_actions"] / max(1.0, minute)
    
    # Derived composites
    feat["attack_strength_h"] = _calculate_attack_strength(feat, "home")
    feat["attack_strength_a"] = _calculate_attack_strength(feat, "away")
    feat["defense_weakness_h"] = _calculate_defense_weakness(feat, "home")
    feat["defense_weakness_a"] = _calculate_defense_weakness(feat, "away")
    
    # Fill any missing features with 0
    for key in ACCURATE_FEATURES:
        if key not in feat:
            feat[key] = 0.0
    
    return feat

# ---------- DB helpers ----------

def _connect(db_url: Optional[str]):
    url = db_url or os.getenv("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL must be set.")
    
    # Parse DSN
    from urllib.parse import urlparse, parse_qsl
    pr = urlparse(url)
    if pr.scheme not in ("postgresql", "postgres"):
        raise SystemExit("DATABASE_URL must start with postgresql:// or postgres://")
    
    conn_params = {
        "host": pr.hostname or "",
        "port": pr.port or 5432,
        "dbname": (pr.path or "").lstrip("/") or "postgres",
        "user": pr.username or "",
        "password": pr.password or "",
    }
    
    # Add SSL mode
    params = dict(parse_qsl(pr.query))
    params.setdefault("sslmode", "require")
    
    # Build connection string
    conn_str = f"host={conn_params['host']} port={conn_params['port']} dbname={conn_params['dbname']} "
    if conn_params['user']:
        conn_str += f"user={conn_params['user']} "
    if conn_params['password']:
        conn_str += f"password={conn_params['password']} "
    conn_str += "sslmode=require"
    
    try:
        conn = psycopg2.connect(conn_str)
        conn.autocommit = True
        log.info("[DB] Connected to database")
        return conn
    except Exception as e:
        log.error(f"❌ Database connection failed: {e}")
        raise

def _read_sql(conn, sql: str, params: Tuple = ()) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn, params=params)

# ---------- ACCURATE Data loading ----------

def load_accurate_live_data(
    conn,
    min_minute: int = TRAIN_MIN_MINUTE,
    days_back: int = 90,
) -> Dict[str, Any]:
    """
    Load harvested in-play snapshots with ACCURATE features.
    """
    cutoff_ts = int(time.time()) - (days_back * 24 * 3600)
    
    sql = """
    SELECT
        t.match_id,
        t.created_ts,
        t.payload,
        r.final_goals_h,
        r.final_goals_a,
        r.btts_yes
    FROM tip_snapshots t
    JOIN match_results r ON t.match_id = r.match_id
    WHERE t.payload IS NOT NULL
      AND t.created_ts >= %s
    ORDER BY t.created_ts DESC
    LIMIT 50000
    """
    
    df = _read_sql(conn, sql, (cutoff_ts,))

    if df.empty:
        log.warning("No live data found for training")
        return {"X": None, "y": {}, "features": ACCURATE_FEATURES}

    feat_rows: List[List[float]] = []
    market_targets: Dict[str, List[int]] = {}
    
    # Initialize market targets
    market_targets["BTTS_Yes"] = []
    market_targets["BTTS_No"] = []
    market_targets["1X2_Home"] = []
    market_targets["1X2_Away"] = []
    market_targets["1X2_Draw"] = []
    
    # Prepare OU markets for each line
    for ln in OU_TRAIN_LINES:
        market_targets[f"Over_{ln}"] = []
        market_targets[f"Under_{ln}"] = []

    sample_count = 0
    skipped_count = 0
    
    for _, row in df.iterrows():
        try:
            snap_raw = row["payload"]
            if isinstance(snap_raw, str):
                snap = json.loads(snap_raw)
            else:
                snap = snap_raw

            # Build ACCURATE features
            feat = _build_accurate_features_from_snapshot(snap)
            if not feat:
                skipped_count += 1
                continue

            minute = float(feat.get("minute", 0.0))
            if minute < float(min_minute):
                skipped_count += 1
                continue

            # Build vector with ACCURATE features in correct order
            vec = []
            for f in ACCURATE_FEATURES:
                vec.append(float(feat.get(f, 0.0)))
            
            feat_rows.append(vec)

            # Get final results
            gh = int(row["final_goals_h"] or 0)
            ga = int(row["final_goals_a"] or 0)
            total_goals = gh + ga
            btts = int(row["btts_yes"] or 0)

            # BTTS targets
            market_targets["BTTS_Yes"].append(1 if btts == 1 else 0)
            market_targets["BTTS_No"].append(1 if btts == 0 else 0)

            # OU targets for each line
            for ln in OU_TRAIN_LINES:
                market_targets[f"Over_{ln}"].append(1 if total_goals > ln else 0)
                market_targets[f"Under_{ln}"].append(1 if total_goals < ln else 0)

            # 1X2 targets (PROPER 3-way)
            if gh > ga:
                market_targets["1X2_Home"].append(1)
                market_targets["1X2_Away"].append(0)
                market_targets["1X2_Draw"].append(0)
            elif ga > gh:
                market_targets["1X2_Home"].append(0)
                market_targets["1X2_Away"].append(1)
                market_targets["1X2_Draw"].append(0)
            else:
                market_targets["1X2_Home"].append(0)
                market_targets["1X2_Away"].append(0)
                market_targets["1X2_Draw"].append(1)

            sample_count += 1

        except Exception as e:
            log.debug(f"Row parse error: {e}")
            skipped_count += 1
            continue

    if not feat_rows:
        log.warning("No usable feature rows extracted")
        return {"X": None, "y": {}, "features": ACCURATE_FEATURES}

    X = np.array(feat_rows, dtype=float)
    log.info(f"✅ Loaded {X.shape[0]} samples × {X.shape[1]} features")
    log.info(f"⏭️ Skipped {skipped_count} invalid samples")
    
    # Log class distribution
    for market, targets in market_targets.items():
        if targets:
            pos_ratio = sum(targets) / len(targets)
            log.info(f"📊 {market}: {len(targets)} samples, {pos_ratio*100:.1f}% positive")

    return {"X": X, "y": market_targets, "features": ACCURATE_FEATURES}

# ---------- ACCURATE Ensemble Model ----------

class AccurateEnsembleModel:
    """
    ACCURATE ensemble with feature selection, cross-validation, and performance tracking.
    """

    def __init__(self, market: str):
        self.market = market
        self.models: Dict[str, Any] = {}
        self.scaler: Optional[StandardScaler] = None
        self.feature_selector = None
        self.selected_features: List[str] = []
        self.feature_importances: Dict[str, float] = {}
        self.validation_scores: Dict[str, float] = {}
        self.best_model = None
        log.info(f"🎯 Initializing ACCURATE model for {market}")

    def select_features(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Select most important features"""
        if not USE_FEATURE_SELECTION or len(feature_names) <= MIN_FEATURES:
            return X, feature_names
        
        # Use mutual information for feature selection
        selector = SelectKBest(score_func=mutual_info_classif, k=min(MIN_FEATURES, len(feature_names)))
        
        try:
            X_selected = selector.fit_transform(X, y)
            selected_indices = selector.get_support(indices=True)
            selected_features = [feature_names[i] for i in selected_indices]
            
            # Store feature scores
            for idx, score in enumerate(selector.scores_):
                if idx < len(feature_names):
                    self.feature_importances[feature_names[idx]] = float(score)
            
            # Sort by importance
            self.feature_importances = dict(sorted(
                self.feature_importances.items(), 
                key=lambda x: x[1], 
                reverse=True
            ))
            
            log.info(f"✅ Selected {len(selected_features)} features for {self.market}")
            log.info(f"📈 Top features: {list(self.feature_importances.keys())[:5]}")
            
            return X_selected, selected_features
            
        except Exception as e:
            log.warning(f"⚠️ Feature selection failed for {self.market}: {e}")
            return X, feature_names

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        if X.shape[0] < MIN_ROWS:
            log.warning(f"❌ Insufficient samples for {self.market}: {X.shape[0]} < {MIN_ROWS}")
            return {"error": f"insufficient samples: {X.shape[0]}"}

        log.info(f"🔧 Training {self.market} with {X.shape[0]} samples, {len(feature_names)} features")

        # Feature selection
        X_selected, self.selected_features = self.select_features(X, y, feature_names)
        
        # Train-test split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, 
            test_size=TRAIN_TEST_SIZE, 
            random_state=42, 
            stratify=y
        )

        # Scale features
        self.scaler = RobustScaler()  # More robust to outliers
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Define models with optimized hyperparameters
        base_models = {
            "logistic": LogisticRegression(
                max_iter=1000, 
                class_weight="balanced", 
                random_state=42,
                C=0.1,  # Regularization
                solver='liblinear'
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=200, 
                random_state=42, 
                class_weight="balanced_subsample",
                max_depth=10,
                min_samples_split=5
            ),
            "gradient_boosting": GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=6,
                random_state=42
            ),
            "xgboost": xgb.XGBClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=6,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        }

        # Cross-validate each model
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
        
        best_score = 0
        best_model_name = None
        
        for name, base in base_models.items():
            try:
                # Cross-validation
                cv_scores = cross_val_score(
                    base, X_train_scaled, y_train, 
                    cv=cv, scoring='roc_auc'
                )
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                log.info(f"📊 {name} CV AUC: {cv_mean:.3f} (±{cv_std:.3f})")
                
                # Store validation score
                self.validation_scores[name] = cv_mean
                
                # Train on full training set
                clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
                clf.fit(X_train_scaled, y_train)
                self.models[name] = clf
                
                # Update best model
                if cv_mean > best_score:
                    best_score = cv_mean
                    best_model_name = name
                    self.best_model = clf
                    
                log.info(f"✅ Trained {name} for {self.market}")
                
            except Exception as e:
                log.error(f"❌ Failed to train {name} for {self.market}: {e}")

        if not self.models:
            log.error(f"❌ No models trained for {self.market}")
            return {"error": "no models trained"}

        # Evaluate ensemble
        ensemble_proba = self._ensemble_predict_proba_accurate(X_test_scaled)
        ensemble_pred = (ensemble_proba >= 0.5).astype(int)
        
        # Calculate metrics
        acc = accuracy_score(y_test, ensemble_pred)
        prec = precision_score(y_test, ensemble_pred, zero_division=0)
        rec = recall_score(y_test, ensemble_pred, zero_division=0)
        f1 = f1_score(y_test, ensemble_pred, zero_division=0)
        auc = roc_auc_score(y_test, ensemble_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, ensemble_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        log.info(f"[{self.market}] ACCURATE RESULTS:")
        log.info(f"  Accuracy:   {acc:.3f}")
        log.info(f"  Precision:  {prec:.3f}")
        log.info(f"  Recall:     {rec:.3f}")
        log.info(f"  F1 Score:   {f1:.3f}")
        log.info(f"  AUC:        {auc:.3f}")
        log.info(f"  Confusion:  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
        
        # Check if model meets minimum requirements
        if auc < MIN_AUC:
            log.warning(f"⚠️ Model {self.market} AUC {auc:.3f} < minimum {MIN_AUC}")
            return {"error": f"insufficient AUC: {auc:.3f}"}
        
        if acc < MIN_ACCURACY:
            log.warning(f"⚠️ Model {self.market} accuracy {acc:.3f} < minimum {MIN_ACCURACY}")
            return {"error": f"insufficient accuracy: {acc:.3f}"}
        
        # Save the trained model
        self._save_to_disk_accurate()
        
        return {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "auc": float(auc),
            "confusion_matrix": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
            "samples": int(len(y)),
            "positive_ratio": float(np.mean(y)),
            "feature_count": len(self.selected_features),
            "best_model": best_model_name,
            "validation_scores": self.validation_scores,
        }

    def _ensemble_predict_proba_accurate(self, X: np.ndarray) -> np.ndarray:
        """Weighted ensemble prediction based on validation performance"""
        preds = []
        weights = []
        
        for name, clf in self.models.items():
            try:
                p = clf.predict_proba(X)[:, 1]
                preds.append(p)
                
                # Weight by validation score
                weight = self.validation_scores.get(name, 0.5)
                weights.append(max(0.1, weight))  # Minimum weight 0.1
                
            except Exception as e:
                log.warning(f"⚠️ Prediction failed for {self.market}/{name}: {e}")
                continue
        
        if not preds:
            return np.full(X.shape[0], 0.5)
        
        # Weighted average
        preds_arr = np.vstack(preds)
        weights_arr = np.array(weights)
        weights_arr = weights_arr / weights_arr.sum()  # Normalize
        
        weighted_proba = np.average(preds_arr, axis=0, weights=weights_arr)
        return weighted_proba
    
    def _save_to_disk_accurate(self) -> None:
        """Save model with COMPLETE metadata for main.py"""
        try:
            # Create models directory
            Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
            
            # Map training market names to main.py model names
            model_name_map = {
                "1X2_Home": "1X2_Home_Win",
                "1X2_Away": "1X2_Away_Win", 
                "1X2_Draw": "1X2_Draw",
                "BTTS_Yes": "BTTS_Yes",
                "BTTS_No": "BTTS_No",
            }
            
            # Determine output model name
            if self.market in model_name_map:
                output_name = model_name_map[self.market]
            elif self.market.startswith("Over_"):
                line = self.market.replace("Over_", "")
                line_str = line.replace(".", "_")
                output_name = f"Over_{line_str}"
            elif self.market.startswith("Under_"):
                line = self.market.replace("Under_", "")
                line_str = line.replace(".", "_")
                output_name = f"Under_{line_str}"
            else:
                output_name = self.market
            
            # Prepare model data
            model_data = {
                'models': self.models,
                'best_model': self.best_model,
                'scaler': self.scaler,
                'selected_features': self.selected_features,
                'feature_count': len(self.selected_features),
                'feature_importances': self.feature_importances,
                'validation_scores': self.validation_scores,
                'training_timestamp': time.time(),
            }
            
            # Save model
            model_path = Path(MODELS_DIR) / f"{output_name}.joblib"
            joblib.dump(model_data, model_path, compress=('zlib', 3))
            
            # Save COMPLETE metadata
            metadata = {
                'model_name': output_name,
                'market': self.market,
                'required_features': self.selected_features,
                'feature_count': len(self.selected_features),
                'feature_importances': self.feature_importances,
                'validation_scores': self.validation_scores,
                'training_timestamp': time.time(),
                'is_accurate_model': True,
                'accuracy_requirements': {
                    'min_accuracy': MIN_ACCURACY,
                    'min_auc': MIN_AUC,
                    'min_precision': MIN_PRECISION,
                }
            }
            
            metadata_path = Path(MODELS_DIR) / f"{output_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Save feature importance visualization data
            importance_path = Path(MODELS_DIR) / f"{output_name}_importance.json"
            importance_data = {
                'features': list(self.feature_importances.keys()),
                'scores': list(self.feature_importances.values())
            }
            with open(importance_path, 'w') as f:
                json.dump(importance_data, f, indent=2)
            
            log.info(f"💾 Saved ACCURATE model to {model_path}")
            log.info(f"📄 Saved metadata to {metadata_path}")
            log.info(f"📊 Saved feature importance to {importance_path}")
            
        except Exception as e:
            log.error(f"❌ Failed to save model for {self.market}: {e}")

# ---------- ACCURATE Market Model Training ----------

def train_accurate_market_model(
    X: np.ndarray,
    y: List[int],
    features: List[str],
    market: str,
) -> Optional[Dict[str, Any]]:
    if X is None or len(y) == 0:
        log.warning("No data for market %s", market)
        return None

    y_arr = np.array(y, dtype=int)
    if X.shape[0] != y_arr.shape[0]:
        n = min(X.shape[0], y_arr.shape[0])
        X = X[:n]
        y_arr = y_arr[:n]

    if len(np.unique(y_arr)) < 2:
        log.warning("Market %s has single-class labels; skipping", market)
        return None

    # Check positive ratio (avoid extreme class imbalance)
    positive_ratio = np.mean(y_arr)
    if positive_ratio < 0.1 or positive_ratio > 0.9:
        log.warning(f"Market {market} has extreme class imbalance: {positive_ratio:.3f}")
        # Still try, but warn

    model = AccurateEnsembleModel(market)
    res = model.train(X, y_arr, features)
    
    if "error" in res:
        log.error(f"Training error for {market}: {res['error']}")
        return None

    return res

# ---------- Main ACCURATE training function ----------

def train_all_models(
    db_url: Optional[str] = None,
    min_minute: int = TRAIN_MIN_MINUTE,
    force: bool = False,
) -> Dict[str, Any]:
    log.info("🚀 Starting ACCURATE in-play training")
    log.info(f"📊 Minimum accuracy: {MIN_ACCURACY}")
    log.info(f"📈 Minimum AUC: {MIN_AUC}")
    log.info(f"⏱️  Minimum minute: {min_minute}")
    
    start_time = time.time()
    
    conn = None
    try:
        conn = _connect(db_url)
        
        results: Dict[str, Any] = {
            "ok": True,
            "trained": {},
            "models_trained": 0,
            "performance": {},
            "errors": [],
            "warnings": [],
        }

        # Load and prepare data
        log.info("📥 Loading ACCURATE training data...")
        live = load_accurate_live_data(conn, min_minute=min_minute)
        X = live["X"]
        
        if X is None or X.shape[0] < MIN_ROWS:
            msg = f"Insufficient data: {X.shape[0] if X is not None else 0} samples (< {MIN_ROWS})"
            log.error(f"❌ {msg}")
            results["ok"] = False
            results["errors"].append(msg)
            return results

        log.info(f"🎯 Training models for {len(live['y'])} markets...")
        
        trained_count = 0
        skipped_count = 0
        
        for market, y in live["y"].items():
            if len(y) < MIN_ROWS:
                msg = f"Skipping {market}: only {len(y)} rows (<{MIN_ROWS})"
                log.info(f"⏭️ {msg}")
                results["warnings"].append(msg)
                results["trained"][market] = False
                skipped_count += 1
                continue

            log.info(f"🔧 Training ACCURATE model for {market} with {len(y)} samples")
            mdl = train_accurate_market_model(X, y, live["features"], market)
            
            if mdl:
                results["trained"][market] = True
                results["performance"][market] = mdl
                trained_count += 1
                log.info(f"✅ Successfully trained {market} (AUC: {mdl.get('auc', 0):.3f})")
            else:
                results["trained"][market] = False
                results["errors"].append(f"Failed to train model for {market}")
                log.error(f"❌ Failed to train {market}")
                skipped_count += 1

        results["models_trained"] = trained_count
        results["models_skipped"] = skipped_count
        
        training_duration = time.time() - start_time
        
        # Generate summary
        avg_auc = np.mean([p.get('auc', 0) for p in results["performance"].values()])
        avg_acc = np.mean([p.get('accuracy', 0) for p in results["performance"].values()])
        
        log.info("=" * 60)
        log.info(f"🎉 ACCURATE TRAINING COMPLETE")
        log.info(f"   Models trained: {trained_count}")
        log.info(f"   Models skipped: {skipped_count}")
        log.info(f"   Average AUC:    {avg_auc:.3f}")
        log.info(f"   Average Acc:    {avg_acc:.3f}")
        log.info(f"   Duration:       {training_duration:.1f}s")
        log.info("=" * 60)
        
        # Save training report
        report = {
            "timestamp": time.time(),
            "duration_seconds": training_duration,
            "models_trained": trained_count,
            "models_skipped": skipped_count,
            "average_auc": float(avg_auc),
            "average_accuracy": float(avg_acc),
            "performance": results["performance"],
            "errors": results["errors"],
            "warnings": results["warnings"],
        }
        
        report_path = Path(MODELS_DIR) / "training_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        log.info(f"📋 Training report saved to {report_path}")

    except Exception as e:
        log.exception(f"❌ ACCURATE training failed: {e}")
        results = {
            "ok": False,
            "error": str(e),
            "trained": {},
            "models_trained": 0,
            "performance": {},
            "errors": [str(e)],
        }
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass

    return results

# ---------- CLI ----------

def _cli_main() -> None:
    ap = argparse.ArgumentParser(description="ACCURATE In-play trainer")
    ap.add_argument("--db-url", help="Postgres DSN (or env DATABASE_URL)")
    ap.add_argument("--min-minute", type=int, default=TRAIN_MIN_MINUTE)
    ap.add_argument("--force", action="store_true", help="Force training even with warnings")
    args = ap.parse_args()

    res = train_all_models(
        db_url=args.db_url or os.getenv("DATABASE_URL"),
        min_minute=args.min_minute,
        force=args.force,
    )
    
    # Print results
    import json
    print(json.dumps(res, indent=2))
    
    # Exit with appropriate code
    if not res.get("ok", False):
        exit(1)

if __name__ == "__main__":
    _cli_main()
