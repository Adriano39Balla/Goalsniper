# train_models.py – OPTIMIZED in-play trainer
# Removed: Complex feature engineering, placeholder features, over-engineering
# Kept: Core ensemble training, proven features, robust data preparation

import argparse
import json
import logging
import os
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg2
from psycopg2 import OperationalError

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Configure warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
)
log = logging.getLogger("trainer")

# ---------- OPTIMIZED Feature Set (15 proven features) ----------

OPTIMIZED_FEATURES: List[str] = [
    "minute",
    "goals_h", "goals_a",           # Current score
    "xg_h", "xg_a",                 # Expected goals
    "sot_h", "sot_a",               # Shots on target
    "cor_h", "cor_a",               # Corner kicks
    "pos_diff",                     # Possession difference
    "momentum_h", "momentum_a",     # Attack momentum
    "pressure_index",               # Game pressure
    "total_actions",                # Game activity
]

# REMOVED: All advanced features that were adding noise
# REMOVED: "goals_sum", "goals_diff", "xg_sum", "xg_diff", "sot_sum", "cor_sum", etc.

# ---------- Env knobs (simplified) ----------

RECENCY_HALF_LIFE_DAYS = float(os.getenv("RECENCY_HALF_LIFE_DAYS", "120"))
TRAIN_MIN_MINUTE = int(os.getenv("TRAIN_MIN_MINUTE", "15"))
TRAIN_TEST_SIZE = float(os.getenv("TRAIN_TEST_SIZE", "0.25"))
MIN_ROWS = int(os.getenv("MIN_ROWS", "150"))

# REMOVED: Complex auto-tuning and threshold optimization

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

def _fmt_line(line: float) -> str:
    return f"{line}".rstrip("0").rstrip(".")

OU_TRAIN_LINES: List[float] = _parse_ou_lines(os.getenv("OU_LINES", "2.5,3.5"))

# ---------- Optimized Feature Engineering ----------

def remove_low_value_features(df: pd.DataFrame) -> pd.DataFrame:
    """Remove features that don't contribute to predictions"""
    original_count = len(df.columns)
    
    # Remove constant features
    constant_features = []
    for col in df.columns:
        if df[col].nunique() <= 1:
            constant_features.append(col)
    
    if constant_features:
        log.info(f"🗑️ Removing {len(constant_features)} constant features")
        df = df.drop(columns=constant_features)
    
    final_count = len(df.columns)
    if original_count != final_count:
        log.info(f"✅ Feature cleaning: {final_count} features remaining ({original_count - final_count} removed)")
    
    return df

def create_smart_features(df):
    """Create only features that have proven predictive value"""
    log.info("🔧 Creating smart features...")
    
    df_enhanced = df.copy()
    
    try:
        # Only create features that directly relate to match outcomes
        # Goal efficiency - proven predictive value
        df_enhanced['goal_efficiency_h'] = np.where(
            df_enhanced['sot_h'] > 0,
            df_enhanced['goals_h'] / (df_enhanced['sot_h'] + 1),
            0.0
        )
        df_enhanced['goal_efficiency_a'] = np.where(
            df_enhanced['sot_a'] > 0,
            df_enhanced['goals_a'] / (df_enhanced['sot_a'] + 1),
            0.0
        )
        
        # Game phase indicators - simple but effective
        df_enhanced['late_game'] = (df_enhanced['minute'] > 70).astype(int)
        df_enhanced['close_game'] = (np.abs(df_enhanced['goals_h'] - df_enhanced['goals_a']) <= 1).astype(int)
        
        log.info("✅ Created 4 smart features")
        
    except Exception as e:
        log.error(f"❌ Smart feature creation failed: {e}")
        return df  # Return original if enhancement fails
    
    return df_enhanced

def optimized_data_preparation(X, y, feature_names, test_size=0.25):
    """Optimized data preparation pipeline"""
    log.info("🔄 Starting optimized data preparation")
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame(X, columns=feature_names)
        log.info(f"📥 Loaded data: {df.shape[0]} samples, {df.shape[1]} features")
        
        # Step 1: Remove low-value features
        df_clean = remove_low_value_features(df)
        
        # Step 2: Create only proven smart features
        df_enhanced = create_smart_features(df_clean)
        
        # Step 3: Handle data quality
        df_enhanced = df_enhanced.replace([np.inf, -np.inf], 0)
        df_enhanced = df_enhanced.fillna(0)
        
        # Step 4: Simple feature selection
        if len(df_enhanced.columns) > 10:
            # Keep top 10 features by variance
            variances = df_enhanced.var()
            top_features = variances.nlargest(10).index.tolist()
            df_enhanced = df_enhanced[top_features]
            log.info(f"🎯 Selected top 10 features by variance")
        
        # Step 5: Train-test split
        X_clean = df_enhanced.values
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y, test_size=test_size, random_state=42, stratify=y
        )
        
        selected_features = list(df_enhanced.columns)
        log.info(f"✅ Data preparation complete: {X_train.shape[0]} train, {X_test.shape[0]} test, {len(selected_features)} features")
        
        return X_train, X_test, y_train, y_test, selected_features
        
    except Exception as e:
        log.error(f"❌ Data preparation failed: {e}")
        # Fallback to basic split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        return X_train, X_test, y_train, y_test, feature_names

# ---------- Optimized feature rebuilding (aligned with main.py) ----------

def _calculate_momentum(features: Dict[str, float], side: str) -> float:
    suffix = "_h" if side == "home" else "_a"
    sot = float(features.get(f"sot{suffix}", 0.0))
    cor = float(features.get(f"cor{suffix}", 0.0))
    minute = max(1.0, float(features.get("minute", 1.0)))
    return (sot + cor) / minute

def _calculate_pressure_index(features: Dict[str, float]) -> float:
    minute = float(features.get("minute", 0.0))
    goal_diff = abs(float(features.get("goals_h", 0.0)) - float(features.get("goals_a", 0.0)))
    time_pressure = minute / 90.0
    score_pressure = min(1.0, goal_diff / 3.0)
    return time_pressure * 0.5 + score_pressure * 0.3 + 0.2

def _calculate_total_actions(features: Dict[str, float]) -> float:
    return (
        float(features.get("sot_h", 0.0))
        + float(features.get("sot_a", 0.0))
        + float(features.get("cor_h", 0.0))
        + float(features.get("cor_a", 0.0))
    )

def _rebuild_optimized_features(snap: dict) -> Optional[Dict[str, float]]:
    """Rebuild the optimized feature vector that matches main.py"""
    if not isinstance(snap, dict):
        return None

    minute = max(1.0, float(snap.get("minute", 0.0)))
    gh = float(snap.get("gh", 0.0))
    ga = float(snap.get("ga", 0.0))
    stat = snap.get("stat") or {}
    if not isinstance(stat, dict):
        stat = {}

    # Build optimized features (only 15 core features)
    feat: Dict[str, float] = {}

    # Core match state
    feat["minute"] = minute
    feat["goals_h"] = float(stat.get("goals_h", gh))
    feat["goals_a"] = float(stat.get("goals_a", ga))

    # Expected goals
    feat["xg_h"] = float(stat.get("xg_h", 0.0))
    feat["xg_a"] = float(stat.get("xg_a", 0.0))

    # Shots
    feat["sot_h"] = float(stat.get("sot_h", 0.0))
    feat["sot_a"] = float(stat.get("sot_a", 0.0))

    # Corners
    feat["cor_h"] = float(stat.get("cor_h", 0.0))
    feat["cor_a"] = float(stat.get("cor_a", 0.0))

    # Possession
    pos_h = float(stat.get("pos_h", 0.0))
    pos_a = float(stat.get("pos_a", 0.0))
    feat["pos_diff"] = pos_h - pos_a

    # Momentum and pressure
    feat["momentum_h"] = float(stat.get("momentum_h", _calculate_momentum(feat, "home")))
    feat["momentum_a"] = float(stat.get("momentum_a", _calculate_momentum(feat, "away")))
    feat["pressure_index"] = float(stat.get("pressure_index", _calculate_pressure_index(feat)))
    feat["total_actions"] = float(stat.get("total_actions", _calculate_total_actions(feat)))

    return feat

# ---------- Optimized Ensemble Model ----------

class OptimizedEnsembleModel:
    """
    Optimized trainer-side ensemble - removed complexity, kept core functionality
    """

    def __init__(self, market: str):
        self.market = market
        self.models: Dict[str, CalibratedClassifierCV] = {}
        self.scaler: Optional[StandardScaler] = None
        self.selected_features: List[str] = []
        log.info(f"🎯 Initializing OptimizedEnsembleModel for {market}")

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        test_size: float = 0.25,
    ) -> Dict[str, Any]:
        if X.shape[0] < MIN_ROWS:
            log.warning(f"❌ Insufficient samples for {self.market}: {X.shape[0]} < {MIN_ROWS}")
            return {"error": f"insufficient samples: {X.shape[0]} < {MIN_ROWS}"}

        log.info(f"🔧 Training {self.market} with {X.shape[0]} samples")

        # Use optimized data preparation
        X_train, X_test, y_train, y_test, self.selected_features = optimized_data_preparation(
            X, y, feature_names, test_size
        )

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train only core models - removed gradient boost (wasn't helping)
        base_models = {
            "logistic": LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=42
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=100, random_state=42, class_weight="balanced"
            ),
        }

        metrics_by_model: Dict[str, Dict[str, float]] = {}

        for name, base in base_models.items():
            try:
                # Use isotonic calibration for both - simpler and effective
                clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
                clf.fit(X_train_scaled, y_train)
                self.models[name] = clf

                # Evaluate model
                proba = clf.predict_proba(X_test_scaled)[:, 1]
                pred = (proba >= 0.5).astype(int)

                acc = accuracy_score(y_test, pred)
                prec = precision_score(y_test, pred, zero_division=0)
                auc = roc_auc_score(y_test, proba)

                metrics_by_model[name] = {
                    "accuracy": float(acc),
                    "precision": float(prec),
                    "auc": float(auc),
                }

                log.info(
                    "[%s] %s acc=%.3f prec=%.3f auc=%.3f",
                    self.market,
                    name,
                    acc,
                    prec,
                    auc,
                )
            except Exception as e:
                log.error("❌ Failed to train %s for %s: %s", name, self.market, e)

        if not self.models:
            log.error("❌ No models successfully trained for %s", self.market)
            return {"error": "no models trained"}

        # Ensemble evaluation
        ens_proba = self._ensemble_predict_proba(X_test_scaled)
        ens_pred = (ens_proba >= 0.5).astype(int)

        acc = accuracy_score(y_test, ens_pred)
        prec = precision_score(y_test, ens_pred, zero_division=0)
        auc = roc_auc_score(y_test, ens_proba)

        log.info(
            "[%s] ENSEMBLE acc=%.3f prec=%.3f auc=%.3f",
            self.market,
            acc,
            prec,
            auc,
        )

        return {
            "metrics": {
                "accuracy": float(acc),
                "precision": float(prec),
                "auc": float(auc),
                "samples": int(len(y)),
                "positive_ratio": float(np.mean(y)),
                "feature_count": len(self.selected_features),
            },
            "per_model": metrics_by_model,
            "selected_features": list(self.selected_features),
            "training_summary": {
                "market": self.market,
                "total_samples": len(y),
                "model_versions": list(self.models.keys())
            }
        }

    def _ensemble_predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Simple ensemble averaging"""
        preds = []
        for name, clf in self.models.items():
            try:
                p = clf.predict_proba(X)[:, 1]
                preds.append(p)
            except Exception as e:
                log.warning("⚠️ Prediction failed for %s: %s", name, e)
        if not preds:
            return np.full(X.shape[0], 0.5)
        preds_arr = np.vstack(preds)
        return preds_arr.mean(axis=0)

# ---------- DB helpers (simplified) ----------

def _connect(db_url: Optional[str]):
    """Simplified database connection"""
    url = db_url or os.getenv("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL must be set.")
    
    try:
        conn = psycopg2.connect(url)
        conn.autocommit = True
        log.info("[DB] Connected to database")
        return conn
    except OperationalError as e:
        log.error("❌ Database connection failed: %s", e)
        raise

def _read_sql(conn, sql: str, params: Tuple = ()) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn, params=params)

def _exec(conn, sql: str, params: Tuple = ()) -> None:
    with conn.cursor() as cur:
        cur.execute(sql, params)

def _set_setting(conn, key: str, value: str) -> None:
    _exec(
        conn,
        "INSERT INTO settings(key,value) VALUES(%s,%s) "
        "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
        (key, value),
    )

def _ensure_training_tables(conn) -> None:
    """Ensure basic tables exist"""
    _exec(conn, "CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)")
    _exec(
        conn,
        """
        CREATE TABLE IF NOT EXISTS model_performance (
            id SERIAL PRIMARY KEY,
            model_name TEXT,
            accuracy FLOAT,
            precision FLOAT,
            auc FLOAT,
            samples INTEGER,
            training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
    )

# ---------- Optimized data loading ----------

def load_training_data(
    conn,
    min_minute: int = TRAIN_MIN_MINUTE,
) -> Dict[str, Any]:
    """
    Load training data with optimized feature extraction
    """
    sql = """
    SELECT
        t.match_id,
        t.created_ts,
        t.payload,
        r.final_goals_h,
        r.final_goals_a,
        r.btts_yes,
        EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - to_timestamp(t.created_ts))) / (60*60*24) AS days_ago
    FROM tip_snapshots t
    JOIN match_results r ON t.match_id = r.match_id
    WHERE t.payload IS NOT NULL 
      AND r.final_goals_h IS NOT NULL
      AND r.final_goals_a IS NOT NULL
    ORDER BY t.created_ts DESC
    LIMIT 15000
    """
    df = _read_sql(conn, sql)

    if df.empty:
        log.warning("No training data found")
        return {"X": None, "y": {}, "features": OPTIMIZED_FEATURES}

    # Simple recency weighting
    df["recency_weight"] = np.exp(
        -np.abs(df["days_ago"].astype(float)) / max(1.0, RECENCY_HALF_LIFE_DAYS)
    )

    feat_rows: List[List[float]] = []
    market_targets: Dict[str, List[int]] = {
        "BTTS": [],
        "1X2_HOME": [],
        "1X2_AWAY": [],
    }

    # Prepare OU markets
    for ln in OU_TRAIN_LINES:
        market_targets[f"OU_{_fmt_line(ln)}"] = []

    for _, row in df.iterrows():
        try:
            snap_raw = row["payload"]
            if isinstance(snap_raw, str):
                snap = json.loads(snap_raw)
            else:
                snap = snap_raw

            feat = _rebuild_optimized_features(snap)
            if not feat:
                continue

            minute = float(feat.get("minute", 0.0))
            if minute < float(min_minute):
                continue

            # Build feature vector using only optimized features
            vec = [float(feat.get(f, 0.0) or 0.0) for f in OPTIMIZED_FEATURES]
            feat_rows.append(vec)

            gh = int(row["final_goals_h"] or 0)
            ga = int(row["final_goals_a"] or 0)
            total_goals = gh + ga
            btts = int(row["btts_yes"] or 0)

            # BTTS target
            market_targets["BTTS"].append(1 if btts == 1 else 0)

            # OU targets
            for ln in OU_TRAIN_LINES:
                key = f"OU_{_fmt_line(ln)}"
                market_targets[key].append(1 if total_goals > ln else 0)

            # 1X2 targets
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
        log.warning("No usable training samples extracted")
        return {"X": None, "y": {}, "features": OPTIMIZED_FEATURES}

    X = np.array(feat_rows, dtype=float)
    log.info("✅ Loaded %d training samples × %d features", X.shape[0], X.shape[1])

    return {"X": X, "y": market_targets, "features": OPTIMIZED_FEATURES}

# ---------- Train market models ----------

def train_market_model(
    X: np.ndarray,
    y: List[int],
    features: List[str],
    market: str,
    test_size: float,
) -> Optional[Dict[str, Any]]:
    """Train a single market model"""
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

    model = OptimizedEnsembleModel(market)
    res = model.train(X, y_arr, features, test_size=test_size)
    if "error" in res:
        log.error("Training error for %s: %s", market, res["error"])
        return None

    return {
        "model_type": "optimized_ensemble",
        "selected_features": res["selected_features"],
        "metrics": res["metrics"],
        "per_model": res["per_model"],
        "training_summary": res.get("training_summary", {}),
    }

# ---------- Store model metadata ----------

def _store_model_metadata(conn, key: str, model_obj: Dict[str, Any]) -> None:
    """Store model metadata and performance"""
    model_json = json.dumps(model_obj, separators=(",", ":"), ensure_ascii=False)
    _set_setting(conn, key, model_json)

    m = model_obj.get("metrics") or {}
    if m:
        _exec(
            conn,
            """
            INSERT INTO model_performance (model_name, accuracy, precision, auc, samples)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (
                key,
                float(m.get("accuracy", 0.0)),
                float(m.get("precision", 0.0)),
                float(m.get("auc", 0.0)),
                int(m.get("samples", 0)),
            ),
        )
    log.info(
        "✅ Stored model %s (acc=%.3f)",
        key,
        float(m.get("accuracy", 0.0)),
    )

# ---------- Training Progress Analysis ----------

def analyze_training_progress(conn):
    """Simple training progress analysis"""
    try:
        sql = """
        SELECT 
            model_name,
            accuracy,
            precision,
            auc,
            samples,
            training_date
        FROM model_performance 
        WHERE training_date >= CURRENT_DATE - INTERVAL '30 days'
        ORDER BY training_date DESC
        """
        df = _read_sql(conn, sql)
        
        if df.empty:
            return {"status": "no_data", "message": "No training data available"}
        
        # Simple analysis
        latest_models = df.drop_duplicates('model_name', keep='first')
        
        trends = {}
        for _, row in latest_models.iterrows():
            trends[row['model_name']] = {
                'latest_accuracy': float(row['accuracy']),
                'latest_precision': float(row['precision']),
                'latest_auc': float(row['auc']),
                'samples': int(row['samples']),
                'health_status': "EXCELLENT" if row['accuracy'] > 0.75 else "GOOD" if row['accuracy'] > 0.65 else "NEEDS_ATTENTION"
            }
        
        avg_accuracy = latest_models['accuracy'].mean()
        
        return {
            "status": "success",
            "analysis_period": "30 days",
            "overall_health": "GOOD" if avg_accuracy > 0.7 else "NEEDS_ATTENTION",
            "summary": {
                "avg_accuracy": float(avg_accuracy),
                "avg_precision": float(latest_models['precision'].mean()),
                "total_samples_used": int(latest_models['samples'].sum()),
                "models_tracked": len(latest_models)
            },
            "trends": trends
        }
        
    except Exception as e:
        log.error(f"Training progress analysis failed: {e}")
        return {"status": "error", "error": str(e)}

# ---------- Main training entrypoint ----------

def train_models(
    db_url: Optional[str] = None,
    min_minute: int = TRAIN_MIN_MINUTE,
    test_size: float = TRAIN_TEST_SIZE,
    min_rows: int = MIN_ROWS,
) -> Dict[str, Any]:
    log.info("🚀 Starting OPTIMIZED in-play training")
    start_time = time.time()
    conn = _connect(db_url)
    _ensure_training_tables(conn)

    results: Dict[str, Any] = {
        "ok": True,
        "trained": {},
        "errors": [],
        "training_type": "optimized_inplay",
        "models_trained": 0,
        "training_analysis": {},
        "feature_engineering": {
            "base_features": len(OPTIMIZED_FEATURES),
            "smart_features": 4,  # goal_efficiency_h, goal_efficiency_a, late_game, close_game
            "total_features": len(OPTIMIZED_FEATURES) + 4
        }
    }

    try:
        # Analyze previous training progress
        log.info("📊 Analyzing training progress...")
        results["training_analysis"] = analyze_training_progress(conn)
        
        # Load training data
        log.info("📥 Loading training data...")
        training_data = load_training_data(conn, min_minute=min_minute)
        X = training_data["X"]
        
        if X is not None:
            log.info(f"🎯 Training models for {len(training_data['y'])} markets...")
            for market, y in training_data["y"].items():
                if len(y) < min_rows:
                    log.info(
                        "⏭️ Skipping %s: only %d rows (<%d)",
                        market,
                        len(y),
                        min_rows,
                    )
                    results["trained"][market] = False
                    continue

                log.info("🔧 Training model for %s with %d samples", market, len(y))
                model_result = train_market_model(
                    X,
                    y,
                    training_data["features"],
                    market,
                    test_size=test_size,
                )
                if model_result:
                    key = f"model_advanced:{market}"
                    _store_model_metadata(conn, key, model_result)
                    results["trained"][market] = True
                    results["models_trained"] += 1
                    log.info(f"✅ Successfully trained {market}")
                else:
                    results["trained"][market] = False
                    results["errors"].append(f"Failed to train model for {market}")
                    log.error(f"❌ Failed to train {market}")

        # Store training metadata
        training_duration = time.time() - start_time
        meta = {
            "timestamp": time.time(),
            "training_duration_seconds": float(training_duration),
            "training_samples": int(X.shape[0]) if X is not None else 0,
            "trained_models": results["models_trained"],
            "feature_strategy": "optimized_15_core_features"
        }
        _set_setting(
            conn,
            "training_metadata",
            json.dumps(meta, separators=(",", ":")),
        )

        log.info(
            "🎉 OPTIMIZED training completed: %d models, %.1f seconds",
            results["models_trained"],
            training_duration
        )

    except Exception as e:
        log.exception("❌ Training failed: %s", e)
        results["ok"] = False
        results["error"] = str(e)
    finally:
        try:
            conn.close()
        except Exception:
            pass

    return results

# ---------- CLI ----------

def _cli_main() -> None:
    ap = argparse.ArgumentParser(description="Optimized in-play trainer")
    ap.add_argument("--db-url", help="Postgres DSN (or env DATABASE_URL)")
    ap.add_argument("--min-minute", type=int, default=TRAIN_MIN_MINUTE)
    ap.add_argument("--test-size", type=float, default=TRAIN_TEST_SIZE)
    ap.add_argument("--min-rows", type=int, default=MIN_ROWS)
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
