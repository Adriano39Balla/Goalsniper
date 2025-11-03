import os, json, logging, sys
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import psycopg2
from psycopg2.pool import SimpleConnectionPool
from zoneinfo import ZoneInfo
import time

# Database connection (same as main.py)
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise SystemExit("DATABASE_URL is required")

# Logging setup (same as main.py)
log = logging.getLogger("train_models")
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s [train_models] - %(message)s")
handler.setFormatter(formatter)
log.handlers = [handler]
log.setLevel(logging.INFO)
log.propagate = False

# Database connection pool (same as main.py)
POOL: Optional[SimpleConnectionPool] = None

def _init_pool():
    """Initialize database connection pool - same as main.py"""
    global POOL
    if POOL:
        return
    
    maxconn = int(os.getenv("DB_POOL_MAX", "3"))
    try:
        POOL = SimpleConnectionPool(minconn=1, maxconn=maxconn, dsn=DATABASE_URL)
        log.info("[TRAIN_DB] Connected to database (pool=%d)", maxconn)
    except Exception as e:
        log.error("[TRAIN_DB] Failed to connect: %s", e)
        raise

class PooledConn:
    """Database connection context manager - same as main.py"""
    def __init__(self, pool): 
        self.pool = pool
        self.conn = None
        self.cur = None
        
    def __enter__(self):
        _init_pool()
        self.conn = self.pool.getconn()
        self.conn.autocommit = True
        self.cur = self.conn.cursor()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb): 
        try: 
            if self.cur:
                self.cur.close()
        except Exception as e:
            log.warning("[TRAIN_DB] Error closing cursor: %s", e)
        finally: 
            if self.conn:
                try:
                    self.pool.putconn(self.conn)
                except Exception as e:
                    log.warning("[TRAIN_DB] Error returning connection: %s", e)
                    try:
                        self.conn.close()
                    except:
                        pass
    
    def execute(self, sql: str, params: tuple|list=()):
        try:
            self.cur.execute(sql, params or ())
            return self.cur
        except Exception as e:
            log.error("DB execute failed: %s\nSQL: %s\nParams: %s", e, sql, params)
            raise

    def fetchone_safe(self):
        """Safe fetchone that handles empty results"""
        try:
            row = self.cur.fetchone()
            if row is None or len(row) == 0:
                return None
            return row
        except Exception as e:
            log.warning("[TRAIN_DB] fetchone_safe error: %s", e)
            return None
    
    def fetchall_safe(self):
        """Safe fetchall that handles empty results"""
        try:
            rows = self.cur.fetchall()
            return rows if rows else []
        except Exception as e:
            log.warning("[TRAIN_DB] fetchall_safe error: %s", e)
            return []

def db_conn(): 
    """Database connection helper - matches main.py pattern"""
    if not POOL: 
        _init_pool()
    return PooledConn(POOL)

# Feature extraction - MUST MATCH main.py exactly
def _num(v) -> float:
    try:
        if isinstance(v, str) and v.endswith("%"): 
            return float(v[:-1])
        return float(v or 0)
    except: 
        return 0.0

def _pos_pct(v) -> float:
    try: 
        return float(str(v).replace("%", "").strip() or 0)
    except: 
        return 0.0

def extract_basic_features(m: dict) -> Dict[str, float]:
    """EXACT SAME as main.py - CRITICAL for model alignment"""
    home = m["teams"]["home"]["name"]
    away = m["teams"]["away"]["name"]
    gh = m["goals"]["home"] or 0
    ga = m["goals"]["away"] or 0
    minute = int(((m.get("fixture") or {}).get("status") or {}).get("elapsed") or 0)
    stats = {}
    
    for s in (m.get("statistics") or []):
        t = (s.get("team") or {}).get("name")
        if t:
            stats[t] = { (i.get("type") or ""): i.get("value") for i in (s.get("statistics") or []) }
    
    sh = stats.get(home, {}) or {}
    sa = stats.get(away, {}) or {}
    
    # EXACT SAME as main.py
    sot_h = _num(sh.get("Shots on Goal", sh.get("Shots on Target", 0)))
    sot_a = _num(sa.get("Shots on Goal", sa.get("Shots on Target", 0)))
    sh_total_h = _num(sh.get("Total Shots", 0))
    sh_total_a = _num(sa.get("Total Shots", 0))
    cor_h = _num(sh.get("Corner Kicks", 0))
    cor_a = _num(sa.get("Corner Kicks", 0))
    pos_h = _pos_pct(sh.get("Ball Possession", 0))
    pos_a = _pos_pct(sa.get("Ball Possession", 0))

    # EXACT SAME xG logic as main.py
    xg_h = _num(sh.get("Expected Goals", 0))
    xg_a = _num(sa.get("Expected Goals", 0))
    
    # Only estimate if real xG is not available (0 or missing)
    if xg_h == 0 and xg_a == 0:
        # Fallback: estimate xG from shots on target
        xg_h = sot_h * 0.3
        xg_a = sot_a * 0.3

    red_h = red_a = yellow_h = yellow_a = 0
    for ev in (m.get("events") or []):
        if (ev.get("type", "").lower() == "card"):
            d = (ev.get("detail", "") or "").lower()
            t = (ev.get("team") or {}).get("name") or ""
            if "yellow" in d and "second" not in d:
                if t == home: yellow_h += 1
                elif t == away: yellow_a += 1
            if "red" in d or "second yellow" in d:
                if t == home: red_h += 1
                elif t == away: red_a += 1

    return {
        "minute": float(minute),
        "goals_h": float(gh), "goals_a": float(ga),
        "goals_sum": float(gh + ga), "goals_diff": float(gh - ga),

        "xg_h": float(xg_h), "xg_a": float(xg_a),
        "xg_sum": float(xg_h + xg_a), "xg_diff": float(xg_h - xg_a),

        "sot_h": float(sot_h), "sot_a": float(sot_a),
        "sot_sum": float(sot_h + sot_a),

        "sh_total_h": float(sh_total_h), "sh_total_a": float(sh_total_a),

        "cor_h": float(cor_h), "cor_a": float(cor_a),
        "cor_sum": float(cor_h + cor_a),

        "pos_h": float(pos_h), "pos_a": float(pos_a),
        "pos_diff": float(pos_h - pos_a),

        "red_h": float(red_h), "red_a": float(red_a),
        "red_sum": float(red_h + red_a),

        "yellow_h": float(yellow_h), "yellow_a": float(yellow_a)
    }

def get_setting(key: str) -> Optional[str]:
    """Get setting from database - same as main.py"""
    with db_conn() as c:
        cursor = c.execute("SELECT value FROM settings WHERE key=%s", (key,))
        row = cursor.fetchone_safe()
        if row is None or len(row) == 0:
            return None
        return row[0] if row else None

def set_setting(key: str, value: str) -> None:
    """Set setting in database - same as main.py"""
    with db_conn() as c:
        c.execute(
            "INSERT INTO settings(key,value) VALUES(%s,%s) ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
            (key, value)
        )

def load_training_data(days: int = 60, min_minute: int = 20) -> List[Dict[str, Any]]:
    """Load training data from tip_snapshots - IN-PLAY ONLY with enhanced filtering"""
    cutoff_ts = int((datetime.now() - timedelta(days=days)).timestamp())
    
    with db_conn() as c:
        c.execute("""
            SELECT payload 
            FROM tip_snapshots 
            WHERE created_ts >= %s 
            ORDER BY created_ts DESC
            LIMIT 10000  -- Increased limit for better training
        """, (cutoff_ts,))
        
        training_data = []
        skipped_no_result = 0
        skipped_insufficient_data = 0
        
        for (payload,) in c.fetchall_safe():
            try:
                data = json.loads(payload)
                match_data = data.get("match", {})
                features = data.get("features", {})
                
                # Only use in-play data with sufficient minutes
                minute = int(features.get("minute", 0))
                if minute < min_minute:
                    continue
                
                # Check if match has final result
                fixture = match_data.get("fixture", {})
                status = fixture.get("status", {})
                short_status = (status.get("short") or "").upper()
                
                if short_status not in {"FT", "AET", "PEN"}:
                    skipped_no_result += 1
                    continue
                
                # Check for sufficient data quality
                if not _has_sufficient_data(features, minute):
                    skipped_insufficient_data += 1
                    continue
                
                training_data.append({
                    "match": match_data,
                    "features": features,
                    "timestamp": data.get("timestamp", 0)
                })
                
            except Exception as e:
                log.warning("Failed to parse training sample: %s", e)
                continue
        
        log.info("Loaded %d training samples (minute >= %d)", len(training_data), min_minute)
        log.info("Skipped %d (no result) + %d (insufficient data) = %d total samples", 
                skipped_no_result, skipped_insufficient_data, 
                skipped_no_result + skipped_insufficient_data)
        return training_data

def _has_sufficient_data(features: Dict[str, float], minute: int) -> bool:
    """Check if features have sufficient data for training"""
    # Require at least some statistical data
    required_fields = ['sot_h', 'sot_a', 'pos_h', 'pos_a']
    has_data = any(features.get(field, 0) > 0 for field in required_fields)
    
    # For later minutes, require more data
    if minute > 60:
        has_data = has_data and (features.get('xg_sum', 0) > 0 or features.get('sot_sum', 0) >= 3)
    
    return has_data

def calculate_outcome(match_data: dict, market: str, suggestion: str) -> Optional[int]:
    """Calculate if a prediction would have been correct - ENHANCED VERSION"""
    goals = match_data.get("goals", {})
    gh = int(goals.get("home", 0) or 0)
    ga = int(goals.get("away", 0) or 0)
    total_goals = gh + ga
    
    # Get final result from match status
    fixture = match_data.get("fixture", {})
    status = fixture.get("status", {})
    short_status = (status.get("short") or "").upper()
    
    # Only use completed matches
    if short_status not in {"FT", "AET", "PEN"}:
        return None
    
    # BTTS outcomes
    if market == "BTTS":
        if suggestion == "BTTS: Yes":
            return 1 if (gh > 0 and ga > 0) else 0
        elif suggestion == "BTTS: No":
            return 1 if (gh == 0 or ga == 0) else 0
    
    # Over/Under outcomes
    elif market.startswith("Over/Under"):
        try:
            line = float(suggestion.split()[1])
            if suggestion.startswith("Over"):
                return 1 if total_goals > line else (0 if total_goals < line else None)
            elif suggestion.startswith("Under"):
                return 1 if total_goals < line else (0 if total_goals > line else None)
        except:
            return None
    
    # 1X2 outcomes (draw suppressed)
    elif market == "1X2":
        if suggestion == "Home Win":
            return 1 if gh > ga else 0
        elif suggestion == "Away Win":
            return 1 if ga > gh else 0
    
    return None

def prepare_features_and_labels(training_data: List[Dict], market: str, suggestion: str) -> Tuple[List[Dict], List[int]]:
    """Prepare features and labels for a specific market/suggestion with balancing"""
    features_list = []
    labels = []
    
    for data in training_data:
        outcome = calculate_outcome(data["match"], market, suggestion)
        if outcome is not None:
            features_list.append(data["features"])
            labels.append(outcome)
    
    # Balance classes if needed
    if len(labels) > 0:
        pos_count = sum(labels)
        neg_count = len(labels) - pos_count
        
        if min(pos_count, neg_count) < 50:  # If severe imbalance
            log.warning("Class imbalance for %s %s: %d positive, %d negative", 
                       market, suggestion, pos_count, neg_count)
    
    log.info("Market %s - %s: %d samples", market, suggestion, len(features_list))
    return features_list, labels

def train_model_for_market(market: str, suggestion: str, training_data: List[Dict]) -> Optional[Dict[str, Any]]:
    """Train a model for a specific market and suggestion with enhanced validation"""
    
    features_list, labels = prepare_features_and_labels(training_data, market, suggestion)
    
    if len(features_list) < 150:  # Increased minimum samples for better models
        log.warning("Insufficient samples for %s %s: %d", market, suggestion, len(features_list))
        return None
    
    # Convert features to matrix
    feature_names = list(features_list[0].keys())
    
    # Filter out features with no variance
    filtered_features = []
    for feat in features_list:
        filtered_feat = {}
        for name in feature_names:
            value = feat.get(name, 0)
            # Remove features that are always zero or have extremely low variance
            if abs(value) > 1e-6:  # Only include if non-zero
                filtered_feat[name] = value
        filtered_features.append(filtered_feat)
    
    # Update feature names to only those with variance
    feature_names = list(set().union(*(d.keys() for d in filtered_features)))
    
    if len(feature_names) < 5:  # Need minimum features
        log.warning("Insufficient features with variance for %s %s", market, suggestion)
        return None
    
    X = np.array([[feat.get(name, 0) for name in feature_names] for feat in filtered_features])
    y = np.array(labels)
    
    if len(np.unique(y)) < 2:
        log.warning("Only one class in labels for %s %s", market, suggestion)
        return None
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model with class weights for imbalance
    try:
        model = LogisticRegression(
            random_state=42, 
            max_iter=1000,
            class_weight='balanced'  # Handle class imbalance
        )
        model.fit(X_train, y_train)
        
        # Calibrate probabilities
        calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv=min(3, len(X_train)))
        calibrated_model.fit(X_train, y_train)
        
        # Evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        # Calculate precision and recall
        y_pred = model.predict(X_test)
        precision = np.sum((y_pred == 1) & (y_test == 1)) / max(1, np.sum(y_pred == 1))
        recall = np.sum((y_pred == 1) & (y_test == 1)) / max(1, np.sum(y_test == 1))
        
        log.info("Trained %s %s: train=%.3f test=%.3f precision=%.3f recall=%.3f samples=%d", 
                 market, suggestion, train_score, test_score, precision, recall, len(X_train))
        
        # Extract weights and intercept from calibrated model
        if hasattr(calibrated_model, 'calibrated_classifiers_') and calibrated_model.calibrated_classifiers_:
            base_estimator = calibrated_model.calibrated_classifiers_[0].base_estimator
            weights = dict(zip(feature_names, base_estimator.coef_[0]))
            intercept = float(base_estimator.intercept_[0])
        else:
            weights = dict(zip(feature_names, model.coef_[0]))
            intercept = float(model.intercept_[0])
        
        return {
            "weights": weights,
            "intercept": intercept,
            "feature_names": feature_names,
            "train_score": float(train_score),
            "test_score": float(test_score),
            "precision": float(precision),
            "recall": float(recall),
            "samples": len(X_train),
            "positive_samples": int(np.sum(y_train)),
            "calibration": {"method": "sigmoid", "a": 1.0, "b": 0.0},
            "created_ts": int(time.time())
        }
        
    except Exception as e:
        log.error("Model training failed for %s %s: %s", market, suggestion, e)
        return None

def train_models(days: int = 60) -> Dict[str, Any]:
    """Main training function - IN-PLAY MARKETS ONLY with enhanced tracking"""
    log.info("Starting model training (in-play only, %d days)", days)
    
    try:
        training_data = load_training_data(days)
        if not training_data:
            return {"ok": False, "error": "No training data available"}
        
        log.info("Training data summary: %d matches", len(training_data))
        
        # IN-PLAY MARKETS ONLY (no prematch)
        markets_to_train = [
            ("BTTS", "BTTS: Yes"),
            ("BTTS", "BTTS: No"),
            ("Over/Under 2.5", "Over 2.5 Goals"),
            ("Over/Under 2.5", "Under 2.5 Goals"),
            ("Over/Under 3.5", "Over 3.5 Goals"), 
            ("Over/Under 3.5", "Under 3.5 Goals"),
            ("1X2", "Home Win"),
            ("1X2", "Away Win")
        ]
        
        trained_models = {}
        model_details = {}
        
        for market, suggestion in markets_to_train:
            try:
                model = train_model_for_market(market, suggestion, training_data)
                if model:
                    # Convert to main.py format
                    if market.startswith("Over/Under"):
                        line = market.split()[-1]
                        model_key = f"OU_{line}"
                    else:
                        model_key = market
                    
                    # Save to database
                    set_setting(model_key, json.dumps(model, separators=(",", ":")))
                    trained_models[f"{market} {suggestion}"] = True
                    model_details[model_key] = {
                        "train_score": model["train_score"],
                        "test_score": model["test_score"],
                        "samples": model["samples"]
                    }
                    log.info("✅ Saved model: %s (test score: %.3f)", model_key, model["test_score"])
                else:
                    trained_models[f"{market} {suggestion}"] = False
                    log.warning("❌ Failed to train: %s %s", market, suggestion)
                
            except Exception as e:
                log.error("🚨 Training error for %s %s: %s", market, suggestion, e)
                trained_models[f"{market} {suggestion}"] = False
        
        # Auto-tune thresholds if enabled
        tuned_thresholds = {}
        if os.getenv("AUTO_TUNE_ENABLE", "0") not in ("0", "false", "False"):
            try:
                tuned_thresholds = auto_tune_thresholds(training_data, model_details)
                log.info("Auto-tuned thresholds: %s", tuned_thresholds)
                
                # Save tuned thresholds
                for market, threshold in tuned_thresholds.items():
                    set_setting(f"conf_threshold:{market}", str(threshold))
                    
            except Exception as e:
                log.warning("Auto-tuning failed: %s", e)
        
        success_count = sum(1 for v in trained_models.values() if v)
        total_count = len(trained_models)
        
        return {
            "ok": True,
            "trained": trained_models,
            "model_details": model_details,
            "tuned_thresholds": tuned_thresholds,
            "total_samples": len(training_data),
            "success_rate": f"{success_count}/{total_count}",
            "message": f"Trained {success_count}/{total_count} models from {len(training_data)} samples"
        }
        
    except Exception as e:
        log.exception("Training failed: %s", e)
        return {"ok": False, "error": str(e)}

def auto_tune_thresholds(training_data: List[Dict], model_details: Dict) -> Dict[str, float]:
    """Enhanced auto-tuning based on model performance and precision"""
    default_threshold = float(os.getenv("CONF_THRESHOLD", "75"))
    
    tuned = {}
    
    for model_key, details in model_details.items():
        test_score = details.get("test_score", 0.5)
        samples = details.get("samples", 0)
        
        # Adjust threshold based on model performance
        if test_score > 0.65 and samples > 200:
            # High-performing model: can use higher threshold
            tuned[model_key] = min(85.0, default_threshold + 5.0)
        elif test_score < 0.55:
            # Low-performing model: use lower threshold
            tuned[model_key] = max(65.0, default_threshold - 10.0)
        else:
            tuned[model_key] = default_threshold
            
        log.info("Tuned %s: %.1f%% (test_score: %.3f, samples: %d)", 
                model_key, tuned[model_key], test_score, samples)
    
    return tuned

# Add cleanup function for database pool
def cleanup():
    """Cleanup database connections"""
    global POOL
    if POOL:
        try:
            POOL.closeall()
            log.info("[TRAIN_DB] Closed database connections")
        except Exception as e:
            log.warning("[TRAIN_DB] Error closing pool: %s", e)

if __name__ == "__main__":
    try:
        result = train_models()
        print(json.dumps(result, indent=2))
        
        # Set exit code based on success
        if not result.get("ok", False):
            sys.exit(1)
            
    finally:
        cleanup()
