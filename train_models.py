import os, json, logging, sys
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import psycopg2
from zoneinfo import ZoneInfo

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

def db_conn():
    """Database connection helper - simplified version"""
    import psycopg2
    from psycopg2.pool import SimpleConnectionPool
    
    # Simple connection without pooling for training
    return psycopg2.connect(DATABASE_URL)

def get_setting(key: str) -> Optional[str]:
    """Get setting from database - same as main.py"""
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT value FROM settings WHERE key=%s", (key,))
            row = cur.fetchone()
            return row[0] if row else None

def set_setting(key: str, value: str) -> None:
    """Set setting in database - same as main.py"""
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO settings(key,value) VALUES(%s,%s) ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
                (key, value)
            )
        conn.commit()

def load_training_data(days: int = 60, min_minute: int = 20) -> List[Dict[str, Any]]:
    """Load training data from tip_snapshots - IN-PLAY ONLY"""
    cutoff_ts = int((datetime.now() - timedelta(days=days)).timestamp())
    
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT payload 
                FROM tip_snapshots 
                WHERE created_ts >= %s 
                ORDER BY created_ts DESC
                LIMIT 5000
            """, (cutoff_ts,))
            
            training_data = []
            for (payload,) in cur.fetchall():
                try:
                    data = json.loads(payload)
                    match_data = data.get("match", {})
                    features = data.get("features", {})
                    
                    # Only use in-play data with sufficient minutes
                    minute = int(features.get("minute", 0))
                    if minute >= min_minute:
                        training_data.append({
                            "match": match_data,
                            "features": features,
                            "timestamp": data.get("timestamp", 0)
                        })
                except Exception as e:
                    log.warning("Failed to parse training sample: %s", e)
                    continue
            
            log.info("Loaded %d training samples (minute >= %d)", len(training_data), min_minute)
            return training_data

def calculate_outcome(match_data: dict, market: str, suggestion: str) -> Optional[int]:
    """Calculate if a prediction would have been correct"""
    goals = match_data.get("goals", {})
    gh = goals.get("home", 0)
    ga = goals.get("away", 0)
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
                return 1 if total_goals > line else 0
            elif suggestion.startswith("Under"):
                return 1 if total_goals < line else 0
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
    """Prepare features and labels for a specific market/suggestion"""
    features_list = []
    labels = []
    
    for data in training_data:
        outcome = calculate_outcome(data["match"], market, suggestion)
        if outcome is not None:
            features_list.append(data["features"])
            labels.append(outcome)
    
    log.info("Market %s - %s: %d samples", market, suggestion, len(features_list))
    return features_list, labels

def train_model_for_market(market: str, suggestion: str, training_data: List[Dict]) -> Optional[Dict[str, Any]]:
    """Train a model for a specific market and suggestion"""
    
    features_list, labels = prepare_features_and_labels(training_data, market, suggestion)
    
    if len(features_list) < 100:  # Minimum samples
        log.warning("Insufficient samples for %s %s: %d", market, suggestion, len(features_list))
        return None
    
    # Convert features to matrix
    feature_names = list(features_list[0].keys())
    X = np.array([[feat.get(name, 0) for name in feature_names] for feat in features_list])
    y = np.array(labels)
    
    if len(np.unique(y)) < 2:
        log.warning("Only one class in labels for %s %s", market, suggestion)
        return None
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    # Calibrate probabilities
    calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv=3)
    calibrated_model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    log.info("Trained %s %s: train=%.3f test=%.3f samples=%d", 
             market, suggestion, train_score, test_score, len(X_train))
    
    # Extract weights and intercept
    if hasattr(calibrated_model, 'calibrated_classifiers_'):
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
        "train_score": train_score,
        "test_score": test_score,
        "samples": len(X_train),
        "calibration": {"method": "sigmoid", "a": 1.0, "b": 0.0}
    }

def train_models(days: int = 60) -> Dict[str, Any]:
    """Main training function - IN-PLAY MARKETS ONLY"""
    log.info("Starting model training (in-play only, %d days)", days)
    
    try:
        training_data = load_training_data(days)
        if not training_data:
            return {"ok": False, "error": "No training data available"}
        
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
        
        for market, suggestion in markets_to_train:
            try:
                model = train_model_for_market(market, suggestion, training_data)
                if model:
                    # Save to database (same as main.py expects)
                    model_key = market
                    if market.startswith("Over/Under"):
                        # Convert to main.py format: "OU_2.5" instead of "Over/Under 2.5"
                        line = market.split()[-1]
                        model_key = f"OU_{line}"
                    
                    set_setting(model_key, json.dumps(model, separators=(",", ":")))
                    trained_models[f"{market} {suggestion}"] = True
                    log.info("Saved model: %s", model_key)
                
            except Exception as e:
                log.error("Failed to train %s %s: %s", market, suggestion, e)
                trained_models[f"{market} {suggestion}"] = False
        
        # Auto-tune thresholds if enabled
        if os.getenv("AUTO_TUNE_ENABLE", "0") not in ("0", "false", "False"):
            try:
                tuned_thresholds = auto_tune_thresholds(training_data)
                log.info("Auto-tuned thresholds: %s", tuned_thresholds)
            except Exception as e:
                log.warning("Auto-tuning failed: %s", e)
        
        return {
            "ok": True,
            "trained": trained_models,
            "total_samples": len(training_data),
            "message": f"Trained {sum(1 for v in trained_models.values() if v)}/{len(trained_models)} models"
        }
        
    except Exception as e:
        log.exception("Training failed: %s", e)
        return {"ok": False, "error": str(e)}

def auto_tune_thresholds(training_data: List[Dict]) -> Dict[str, float]:
    """Simple auto-tuning based on training data performance"""
    # This is a simplified version - you might want to expand this
    default_threshold = float(os.getenv("CONF_THRESHOLD", "75"))
    
    # For now, return defaults - you can implement proper tuning here
    return {
        "BTTS": default_threshold,
        "Over/Under 2.5": default_threshold,
        "Over/Under 3.5": default_threshold, 
        "1X2": default_threshold
    }

if __name__ == "__main__":
    result = train_models()
    print(json.dumps(result, indent=2))
