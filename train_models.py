#!/usr/bin/env python3
"""
train_models.py – simplified in-play trainer matching main.py
FIXED: Feature synchronization with main.py
FIXED: Model saving with correct feature mapping
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import psycopg2

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
)
log = logging.getLogger("trainer")

# ---------- Feature Set (MUST match main.py.extract_features exactly) ----------

BASE_FEATURES: List[str] = [
    "minute",
    "goals_h", "goals_a", "goals_sum", "goals_diff",
    "xg_h", "xg_a", "xg_sum", "xg_diff",
    "sot_h", "sot_a", "sot_sum",
    "sh_total_h", "sh_total_a",
    "cor_h", "cor_a", "cor_sum",
    "pos_h", "pos_a", "pos_diff",
    "momentum_h", "momentum_a",
    "pressure_index",
    "efficiency_h", "efficiency_a",
    "total_actions",
    "action_intensity",
]

FEATURES = BASE_FEATURES

# ---------- Env knobs ----------

TRAIN_MIN_MINUTE = int(os.getenv("TRAIN_MIN_MINUTE", "15"))
TRAIN_TEST_SIZE = float(os.getenv("TRAIN_TEST_SIZE", "0.25"))
MIN_ROWS = int(os.getenv("MIN_ROWS", "100"))

MODELS_DIR = os.getenv("MODELS_DIR", "models")

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
    return vals or [2.5, 3.5]

OU_TRAIN_LINES_RAW = os.getenv("OU_TRAIN_LINES", os.getenv("OU_LINES", "2.5,3.5"))
OU_TRAIN_LINES: List[float] = _parse_ou_lines(OU_TRAIN_LINES_RAW)

# ---------- Feature Engineering (MUST match main.py) ----------

def _calculate_momentum(features: Dict[str, float], side: str) -> float:
    suffix = "_h" if side == "home" else "_a"
    sot = float(features.get(f"sot{suffix}", 0.0))
    cor = float(features.get(f"cor{suffix}", 0.0))
    minute = float(features.get("minute", 1.0))
    if minute <= 0:
        return 0.0
    return (sot + cor) / minute

def _calculate_pressure_index(features: Dict[str, float]) -> float:
    minute = float(features.get("minute", 0.0))
    goal_diff = abs(float(features.get("goals_diff", 0.0)))
    time_pressure = minute / 90.0
    score_pressure = min(1.0, goal_diff / 3.0)
    return time_pressure * 0.5 + score_pressure * 0.3 + 0.2

def _calculate_efficiency(features: Dict[str, float], side: str) -> float:
    suffix = "_h" if side == "home" else "_a"
    goals = float(features.get(f"goals{suffix}", 0.0))
    sot = float(features.get(f"sot{suffix}", 0.0))
    if sot <= 0:
        return 0.0
    return goals / sot

def _calculate_total_actions(features: Dict[str, float]) -> float:
    return (
        float(features.get("sot_h", 0.0))
        + float(features.get("sot_a", 0.0))
        + float(features.get("cor_h", 0.0))
        + float(features.get("cor_a", 0.0))
    )

def _calculate_action_intensity(features: Dict[str, float]) -> float:
    minute = float(features.get("minute", 1.0))
    if minute <= 0:
        return 0.0
    total = _calculate_total_actions(features)
    return total / minute

def _build_features_from_snapshot(snap: dict) -> Optional[Dict[str, float]]:
    """
    Rebuild the exact feature vector that main.py's extract_features() produces.
    """
    if not isinstance(snap, dict):
        return None

    minute = float(snap.get("minute", 0.0))
    gh = float(snap.get("gh", 0.0))
    ga = float(snap.get("ga", 0.0))
    stat = snap.get("stat") or {}
    if not isinstance(stat, dict):
        stat = {}

    # Reconstruct features EXACTLY as main.py does
    feat: Dict[str, float] = {}

    # Base goals
    feat["minute"] = minute
    feat["goals_h"] = float(stat.get("goals_h", gh))
    feat["goals_a"] = float(stat.get("goals_a", ga))
    feat["goals_sum"] = float(stat.get("goals_sum", feat["goals_h"] + feat["goals_a"]))
    feat["goals_diff"] = float(stat.get("goals_diff", feat["goals_h"] - feat["goals_a"]))

    # xG
    feat["xg_h"] = float(stat.get("xg_h", 0.0))
    feat["xg_a"] = float(stat.get("xg_a", 0.0))
    feat["xg_sum"] = float(stat.get("xg_sum", feat["xg_h"] + feat["xg_a"]))
    feat["xg_diff"] = float(stat.get("xg_diff", feat["xg_h"] - feat["xg_a"]))

    # Shots
    feat["sot_h"] = float(stat.get("sot_h", 0.0))
    feat["sot_a"] = float(stat.get("sot_a", 0.0))
    feat["sot_sum"] = float(stat.get("sot_sum", feat["sot_h"] + feat["sot_a"]))
    feat["sh_total_h"] = float(stat.get("sh_total_h", 0.0))
    feat["sh_total_a"] = float(stat.get("sh_total_a", 0.0))

    # Corners
    feat["cor_h"] = float(stat.get("cor_h", 0.0))
    feat["cor_a"] = float(stat.get("cor_a", 0.0))
    feat["cor_sum"] = float(stat.get("cor_sum", feat["cor_h"] + feat["cor_a"]))

    # Possession
    feat["pos_h"] = float(stat.get("pos_h", 0.0))
    feat["pos_a"] = float(stat.get("pos_a", 0.0))
    feat["pos_diff"] = float(stat.get("pos_diff", feat["pos_h"] - feat["pos_a"]))

    # Advanced metrics - calculate EXACTLY as main.py
    feat["momentum_h"] = float(_calculate_momentum(feat, "home"))
    feat["momentum_a"] = float(_calculate_momentum(feat, "away"))
    feat["pressure_index"] = float(_calculate_pressure_index(feat))
    feat["efficiency_h"] = float(_calculate_efficiency(feat, "home"))
    feat["efficiency_a"] = float(_calculate_efficiency(feat, "away"))
    feat["total_actions"] = float(_calculate_total_actions(feat))
    feat["action_intensity"] = float(_calculate_action_intensity(feat))

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

# ---------- Data loading ----------

def load_live_data(
    conn,
    min_minute: int = TRAIN_MIN_MINUTE,
) -> Dict[str, Any]:
    """
    Load harvested in-play snapshots and join with final results.
    """
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
    ORDER BY t.created_ts DESC
    LIMIT 10000
    """
    df = _read_sql(conn, sql)

    if df.empty:
        log.warning("No live data found for training")
        return {"X": None, "y": {}, "features": FEATURES}

    feat_rows: List[List[float]] = []
    market_targets: Dict[str, List[int]] = {
        "BTTS": [],
        "1X2_HOME": [],
        "1X2_AWAY": [],
    }

    # Prepare OU markets
    ou_keys = [f"OU_{_fmt_line(ln)}" for ln in OU_TRAIN_LINES]
    for key in ou_keys:
        market_targets[key] = []

    for _, row in df.iterrows():
        try:
            snap_raw = row["payload"]
            if isinstance(snap_raw, str):
                snap = json.loads(snap_raw)
            else:
                snap = snap_raw

            feat = _build_features_from_snapshot(snap)
            if not feat:
                continue

            minute = float(feat.get("minute", 0.0))
            if minute < float(min_minute):
                continue

            # Build vector with ALL features in correct order
            vec = []
            for f in FEATURES:
                vec.append(float(feat.get(f, 0.0)))
            
            feat_rows.append(vec)

            gh = int(row["final_goals_h"] or 0)
            ga = int(row["final_goals_a"] or 0)
            total_goals = gh + ga
            btts = int(row["btts_yes"] or 0)

            # BTTS
            market_targets["BTTS"].append(1 if btts == 1 else 0)

            # OU per configured line
            for ln in OU_TRAIN_LINES:
                key = f"OU_{_fmt_line(ln)}"
                market_targets[key].append(1 if total_goals > ln else 0)

            # 1X2 markets
            if gh > ga:
                market_targets["1X2_HOME"].append(1)
                market_targets["1X2_AWAY"].append(0)
            elif ga > gh:
                market_targets["1X2_HOME"].append(0)
                market_targets["1X2_AWAY"].append(1)
            else:
                market_targets["1X2_HOME"].append(0)
                market_targets["1X2_AWAY"].append(0)

        except Exception as e:
            log.debug("Row parse error: %s", e)
            continue

    if not feat_rows:
        log.warning("No usable feature rows extracted")
        return {"X": None, "y": {}, "features": FEATURES}

    X = np.array(feat_rows, dtype=float)
    log.info("Loaded %d samples × %d features", X.shape[0], X.shape[1])

    return {"X": X, "y": market_targets, "features": FEATURES}

# ---------- Simple Ensemble Model ----------

class SimpleEnsembleModel:
    """
    Simple ensemble that saves models with correct feature metadata.
    """

    def __init__(self, market: str):
        self.market = market
        self.models: Dict[str, Any] = {}
        self.scaler: Optional[StandardScaler] = None
        self.selected_features: List[str] = []
        log.info(f"🎯 Initializing model for {market}")

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        if X.shape[0] < MIN_ROWS:
            log.warning(f"❌ Insufficient samples for {self.market}: {X.shape[0]} < {MIN_ROWS}")
            return {"error": f"insufficient samples: {X.shape[0]}"}

        log.info(f"🔧 Training {self.market} with {X.shape[0]} samples, {len(feature_names)} features")

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TRAIN_TEST_SIZE, random_state=42, stratify=y
        )

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train simple models
        base_models = {
            "logistic": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
        }

        for name, base in base_models.items():
            try:
                clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
                clf.fit(X_train_scaled, y_train)
                self.models[name] = clf
                log.info(f"✅ Trained {name} for {self.market}")
            except Exception as e:
                log.error(f"❌ Failed to train {name} for {self.market}: {e}")

        if not self.models:
            log.error(f"❌ No models trained for {self.market}")
            return {"error": "no models trained"}

        # Evaluate
        ensemble_proba = self._ensemble_predict_proba(X_test_scaled)
        ensemble_pred = (ensemble_proba >= 0.5).astype(int)

        acc = accuracy_score(y_test, ensemble_pred)
        prec = precision_score(y_test, ensemble_pred, zero_division=0)
        auc = roc_auc_score(y_test, ensemble_proba)

        log.info(f"[{self.market}] ENSEMBLE acc={acc:.3f} prec={prec:.3f} auc={auc:.3f}")

        # Save the trained model
        self._save_to_disk()
        
        return {
            "accuracy": float(acc),
            "precision": float(prec),
            "auc": float(auc),
            "samples": int(len(y)),
            "positive_ratio": float(np.mean(y)),
            "feature_count": len(feature_names),
        }

    def _ensemble_predict_proba(self, X: np.ndarray) -> np.ndarray:
        preds = []
        for name, clf in self.models.items():
            try:
                p = clf.predict_proba(X)[:, 1]
                preds.append(p)
            except Exception as e:
                log.warning(f"⚠️ Prediction failed for {self.market}/{name}: {e}")
        
        if not preds:
            return np.full(X.shape[0], 0.5)
        
        preds_arr = np.vstack(preds)
        return preds_arr.mean(axis=0)
    
    def _save_to_disk(self) -> None:
        """Save model with metadata for main.py"""
        try:
            # Create models directory
            Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
            
            # Map training market names to main.py model names
            model_name_map = {
                "1X2_HOME": "1X2_Home_Win",
                "1X2_AWAY": "1X2_Away_Win", 
                "BTTS": "BTTS",
            }
            
            # Determine output model name
            if self.market in model_name_map:
                output_name = model_name_map[self.market]
            elif self.market.startswith("OU_"):
                line = self.market.replace("OU_", "")
                line_str = line.replace(".", "_")
                output_name = f"Over_Under_{line_str}"
            else:
                output_name = self.market
            
            # Prepare model data
            model_data = {
                'models': self.models,
                'scaler': self.scaler,
                'selected_features': FEATURES,  # ALL features
                'feature_count': len(FEATURES),
                'training_timestamp': time.time(),
            }
            
            # Save model
            model_path = Path(MODELS_DIR) / f"{output_name}.joblib"
            joblib.dump(model_data, model_path, compress=('zlib', 3))
            
            # Save metadata
            metadata = {
                'model_name': output_name,
                'market': self.market,
                'required_features': FEATURES,  # ALL features
                'feature_count': len(FEATURES),
                'training_timestamp': time.time(),
                'features_example': {feature: 0.0 for feature in FEATURES}
            }
            
            metadata_path = Path(MODELS_DIR) / f"{output_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            log.info(f"💾 Saved model to {model_path} ({len(FEATURES)} features)")
            log.info(f"📄 Saved metadata to {metadata_path}")
            
        except Exception as e:
            log.error(f"❌ Failed to save model for {self.market}: {e}")

# ---------- Train market models ----------

def train_market_model(
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

    model = SimpleEnsembleModel(market)
    res = model.train(X, y_arr, features)
    
    if "error" in res:
        log.error("Training error for %s: %s", market, res["error"])
        return None

    return res

# ---------- Main training function ----------

def train_models(
    db_url: Optional[str] = None,
    min_minute: int = TRAIN_MIN_MINUTE,
) -> Dict[str, Any]:
    log.info("🚀 Starting in-play training (min_minute=%d)", min_minute)
    start_time = time.time()
    
    conn = None
    try:
        conn = _connect(db_url)
        
        results: Dict[str, Any] = {
            "ok": True,
            "trained": {},
            "models_trained": 0,
            "errors": [],
        }

        # Load and prepare data
        log.info("📥 Loading training data...")
        live = load_live_data(conn, min_minute=min_minute)
        X = live["X"]
        
        if X is not None:
            log.info(f"🎯 Training models for {len(live['y'])} markets...")
            for market, y in live["y"].items():
                if len(y) < MIN_ROWS:
                    log.info(f"⏭️ Skipping {market}: only {len(y)} rows (<{MIN_ROWS})")
                    results["trained"][market] = False
                    continue

                log.info(f"🔧 Training model for {market} with {len(y)} samples")
                mdl = train_market_model(X, y, live["features"], market)
                
                if mdl:
                    results["trained"][market] = True
                    results["models_trained"] += 1
                    log.info(f"✅ Successfully trained {market}")
                else:
                    results["trained"][market] = False
                    results["errors"].append(f"Failed to train model for {market}")
                    log.error(f"❌ Failed to train {market}")

        training_duration = time.time() - start_time
        
        log.info(f"🎉 Training completed: {results['models_trained']} models, {training_duration:.1f} seconds")
        log.info("💾 Models saved to 'models/' directory for main.py to load")

    except Exception as e:
        log.exception("❌ Training failed: %s", e)
        results = {
            "ok": False,
            "error": str(e),
            "trained": {},
            "models_trained": 0,
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
    ap = argparse.ArgumentParser(description="In-play trainer")
    ap.add_argument("--db-url", help="Postgres DSN (or env DATABASE_URL)")
    ap.add_argument("--min-minute", type=int, default=TRAIN_MIN_MINUTE)
    args = ap.parse_args()

    res = train_models(
        db_url=args.db_url or os.getenv("DATABASE_URL"),
        min_minute=args.min_minute,
    )
    
    # Print results
    import json
    print(json.dumps(res, indent=2))
    
    # Exit with appropriate code
    if not res.get("ok", False):
        exit(1)

if __name__ == "__main__":
    _cli_main()
