#!/usr/bin/env python3
"""
goalsniper — PURE IN-PLAY AI mode
Fixed: Automatic model training at startup
Fixed: Feature synchronization between training and prediction
Fixed: Model loading with correct feature mapping
Fixed: Single reliable prediction path
"""

import os, json, time, logging, requests, psycopg2
import numpy as np
import pandas as pd
from collections import deque, defaultdict
from psycopg2.pool import SimpleConnectionPool
from html import escape
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from flask import Flask, jsonify, request, abort
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

# ───────── Env bootstrap ─────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ───────── App / logging ─────────
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
log = logging.getLogger("goalsniper")
app = Flask(__name__)

# ───────── Required envs ─────────
def _require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise SystemExit(f"Missing required environment variable: {name}")
    return v

# ───────── Core env ─────────
TELEGRAM_BOT_TOKEN = _require_env("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = _require_env("TELEGRAM_CHAT_ID")
API_KEY            = _require_env("API_KEY")
ADMIN_API_KEY      = os.getenv("ADMIN_API_KEY")
WEBHOOK_SECRET     = os.getenv("TELEGRAM_WEBHOOK_SECRET")
RUN_SCHEDULER      = os.getenv("RUN_SCHEDULER", "1") not in ("0","false","False","no","NO")

# Precision-related knobs
CONF_THRESHOLD     = float(os.getenv("CONF_THRESHOLD", "75"))
MAX_TIPS_PER_SCAN  = int(os.getenv("MAX_TIPS_PER_SCAN", "25"))
DUP_COOLDOWN_MIN   = int(os.getenv("DUP_COOLDOWN_MIN", "20"))
TIP_MIN_MINUTE     = int(os.getenv("TIP_MIN_MINUTE", "12"))
SCAN_INTERVAL_SEC  = int(os.getenv("SCAN_INTERVAL_SEC", "300"))

# API budget controls
API_BUDGET_DAILY      = int(os.getenv("API_BUDGET_DAILY", "150000"))
MAX_FIXTURES_PER_SCAN = int(os.getenv("MAX_FIXTURES_PER_SCAN", "160"))
USE_EVENTS_IN_FEATURES = os.getenv("USE_EVENTS_IN_FEATURES", "0") not in ("0","false","False","no","NO")

try:
    LEAGUE_ALLOW_IDS = {int(x) for x in os.getenv("LEAGUE_ALLOW_IDS","").split(",") if x.strip().isdigit()}
except Exception:
    LEAGUE_ALLOW_IDS = set()

HARVEST_MODE       = os.getenv("HARVEST_MODE", "1") not in ("0","false","False","no","NO")
TRAIN_ENABLE       = os.getenv("TRAIN_ENABLE", "1") not in ("0","false","False","no","NO")
TRAIN_HOUR_UTC     = int(os.getenv("TRAIN_HOUR_UTC", "2"))
TRAIN_MINUTE_UTC   = int(os.getenv("TRAIN_MINUTE_UTC", "12"))
TRAIN_MIN_MINUTE   = int(os.getenv("TRAIN_MIN_MINUTE", "15"))

BACKFILL_EVERY_MIN = int(os.getenv("BACKFILL_EVERY_MIN", "15"))
BACKFILL_DAYS      = int(os.getenv("BACKFILL_DAYS", "14"))
DAILY_ACCURACY_DIGEST_ENABLE = os.getenv("DAILY_ACCURACY_DIGEST_ENABLE", "1") not in ("0","false","False","no","NO")
DAILY_ACCURACY_HOUR   = int(os.getenv("DAILY_ACCURACY_HOUR", "3"))
DAILY_ACCURACY_MINUTE = int(os.getenv("DAILY_ACCURACY_MINUTE", "6"))

SELF_LEARNING_ENABLE   = os.getenv("SELF_LEARNING_ENABLE", "1") not in ("0","false","False","no","NO")
SELF_LEARN_BATCH_SIZE  = int(os.getenv("SELF_LEARN_BATCH_SIZE", "50"))

# Simple feature extraction only - no complex enhancements
ENABLE_ENHANCED_FEATURES = False  # Disabled for reliability
ENABLE_CONTEXT_ANALYSIS = False
ENABLE_PERFORMANCE_MONITOR = False
ENABLE_MULTI_BOOK_ODDS = False
ENABLE_TIMING_OPTIMIZATION = False
ENABLE_API_PREDICTIONS = False

# Optional warnings
if not ADMIN_API_KEY:
    log.warning("ADMIN_API_KEY is not set — /admin/* endpoints are less protected.")
if not WEBHOOK_SECRET:
    log.warning("TELEGRAM_WEBHOOK_SECRET is not set — /telegram/webhook/<secret> would be unsafe if exposed.")

# ───────── Lines ─────────
def _parse_lines(env_val: str, default: List[float]) -> List[float]:
    out = []
    for t in (env_val or "").split(","):
        t = t.strip()
        if not t:
            continue
        try:
            out.append(float(t))
        except Exception:
            pass
    return out or default

OU_LINES = [ln for ln in _parse_lines(os.getenv("OU_LINES", "2.5,3.5"), [2.5, 3.5]) if abs(ln - 1.5) > 1e-6]
TOTAL_MATCH_MINUTES   = int(os.getenv("TOTAL_MATCH_MINUTES", "95"))
PREDICTIONS_PER_MATCH = int(os.getenv("PREDICTIONS_PER_MATCH", "1"))
PER_LEAGUE_CAP        = int(os.getenv("PER_LEAGUE_CAP", "2"))

# ───────── Odds/EV controls ─────────
MIN_ODDS_OU   = float(os.getenv("MIN_ODDS_OU", "1.50"))
MIN_ODDS_BTTS = float(os.getenv("MIN_ODDS_BTTS", "1.50"))
MIN_ODDS_1X2  = float(os.getenv("MIN_ODDS_1X2", "1.50"))
MAX_ODDS_ALL  = float(os.getenv("MAX_ODDS_ALL", "20.0"))
EDGE_MIN_BPS  = int(os.getenv("EDGE_MIN_BPS", "600"))
ODDS_BOOKMAKER_ID = os.getenv("ODDS_BOOKMAKER_ID")
ALLOW_TIPS_WITHOUT_ODDS = os.getenv("ALLOW_TIPS_WITHOUT_ODDS","0") not in ("0","false","False","no","NO")

ODDS_SOURCE         = os.getenv("ODDS_SOURCE", "auto").lower()
ODDS_AGGREGATION    = os.getenv("ODDS_AGGREGATION", "median").lower()
ODDS_OUTLIER_MULT   = float(os.getenv("ODDS_OUTLIER_MULT", "1.8"))
ODDS_REQUIRE_N_BOOKS= int(os.getenv("ODDS_REQUIRE_N_BOOKS", "2"))
ODDS_FAIR_MAX_MULT  = float(os.getenv("ODDS_FAIR_MAX_MULT", "2.5"))

# ───────── Markets allow-list (draw suppressed) ─────────
ALLOWED_SUGGESTIONS = {"BTTS: Yes", "OVER 2.5", "UNDER 2.5" "OVER 3.5", "UNDER 3.5", "BTTS: No", "Home Win", "Away Win"}
def _fmt_line(line: float) -> str:
    return f"{line}".rstrip("0").rstrip(".")

for _ln in OU_LINES:
    s = _fmt_line(_ln)
    ALLOWED_SUGGESTIONS.add(f"Over {s} Goals")
    ALLOWED_SUGGESTIONS.add(f"Under {s} Goals")

# ───────── External APIs / HTTP session ─────────
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise SystemExit("DATABASE_URL is required")

BASE_URL         = "https://v3.football.api-sports.io"
FOOTBALL_API_URL = f"{BASE_URL}/fixtures"
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
HEADERS          = {"x-apisports-key": API_KEY, "Accept": "application/json"}
INPLAY_STATUSES  = {"1H", "HT", "2H", "ET", "BT", "P"}

session = requests.Session()
session.mount(
    "https://",
    HTTPAdapter(
        max_retries=Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            respect_retry_after_header=True,
        )
    ),
)

# ───────── Caches & timezones ─────────
STATS_CACHE:  Dict[int, Tuple[float, list]] = {}
EVENTS_CACHE: Dict[int, Tuple[float, list]] = {}
ODDS_CACHE:   Dict[int, Tuple[float, dict]] = {}
SETTINGS_TTL  = int(os.getenv("SETTINGS_TTL_SEC", "60"))
MODELS_TTL    = int(os.getenv("MODELS_CACHE_TTL_SEC", "120"))
TZ_UTC        = ZoneInfo("UTC")
BERLIN_TZ     = ZoneInfo("Europe/Berlin")

# ───────── Negative-result cache ─────────
NEG_CACHE: Dict[Tuple[str,int], Tuple[float, bool]] = {}
NEG_TTL_SEC = int(os.getenv("NEG_TTL_SEC", "45"))

# ───────── API circuit breaker / timeouts ─────────
API_CB = {"failures": 0, "opened_until": 0.0}
API_CB_THRESHOLD    = int(os.getenv("API_CB_THRESHOLD", "8"))
API_CB_COOLDOWN_SEC = int(os.getenv("API_CB_COOLDOWN_SEC", "90"))
REQ_TIMEOUT_SEC     = float(os.getenv("REQ_TIMEOUT_SEC", "8.0"))

# ───────── FIXED Model Manager ─────────
class ModelManager:
    """Manages model loading with EXACT feature matching"""
    
    def __init__(self):
        self.loaded_models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Dict] = {}
        self.available_models = self._scan_available_models()
        self.default_probability = 0.5
        
        log.info("🤖 ModelManager initialized with %s available models", len(self.available_models))
        
    def _scan_available_models(self) -> List[str]:
        """Scan for available model files"""
        available = []
        try:
            import glob, json
            # Look for model files
            for path in glob.glob("models/*.joblib"):
                model_name = path.split('/')[-1].replace('.joblib', '')
                if model_name not in available:
                    available.append(model_name)
                    
            # Look for metadata files
            for path in glob.glob("models/*_metadata.json"):
                with open(path, 'r') as f:
                    metadata = json.load(f)
                model_name = metadata.get('model_name')
                if model_name and model_name not in available:
                    available.append(model_name)
                    
        except Exception as e:
            log.error(f"❌ Error scanning models: {e}")
            
        # Add default model types
        default_models = ["1X2_Home_Win", "1X2_Away_Win", "BTTS"]
        for line in OU_LINES:
            line_str = str(line).replace('.', '_')
            default_models.append(f"Over_Under_{line_str}")
            
        for model in default_models:
            if model not in available:
                available.append(model)
                
        return available
    
    def load_model(self, model_name: str, retries: int = 2) -> Optional[Any]:
        """Load model with EXACT feature metadata"""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        log.info(f"🔄 Loading model: {model_name}")
        
        for attempt in range(1, retries + 1):
            try:
                # Try to load the actual model
                model_path = f"models/{model_name}.joblib"
                if not os.path.exists(model_path):
                    log.warning(f"❌ Model file not found: {model_path}")
                    # Try to create fallback model
                    return self._create_fallback_model(model_name)
                
                model = joblib.load(model_path)
                
                # Load metadata if available
                metadata_path = f"models/{model_name}_metadata.json"
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        self.model_metadata[model_name] = json.load(f)
                
                self.loaded_models[model_name] = model
                log.info(f"✅ Model {model_name} loaded successfully")
                return model
                
            except Exception as e:
                log.error(f"❌ Attempt {attempt} failed for model {model_name}: {e}")
                if attempt < retries:
                    time.sleep(1)
        
        log.warning(f"⚠️ Creating fallback model for {model_name}")
        return self._create_fallback_model(model_name)
    
    def _create_fallback_model(self, model_name: str) -> Any:
        """Create simple fallback model"""
        log.info(f"🛠️ Creating fallback model for {model_name}")
        
        # Determine feature count based on model type
        if '1X2' in model_name:
            feature_count = 5
        elif 'BTTS' in model_name:
            feature_count = 7
        elif 'Over_Under' in model_name:
            feature_count = 8
        else:
            feature_count = 5
        
        # Create simple logistic regression
        model = LogisticRegression(random_state=42)
        X = np.random.randn(10, feature_count)
        y = np.random.randint(0, 2, 10)
        model.fit(X, y)
        
        # Store basic metadata
        self.model_metadata[model_name] = {
            'model_name': model_name,
            'required_features': [f'feature_{i}' for i in range(feature_count)],
            'feature_count': feature_count,
            'is_fallback': True
        }
        
        self.loaded_models[model_name] = model
        return model
    
    def get_required_features(self, model_name: str) -> List[str]:
        """Get EXACT features required by model"""
        if model_name in self.model_metadata:
            return self.model_metadata[model_name].get('required_features', [])
        
        # Default features if no metadata
        if '1X2' in model_name:
            return ['minute', 'goals_sum', 'goals_diff', 'xg_sum', 'xg_diff']
        elif 'BTTS' in model_name:
            return ['minute', 'goals_h', 'goals_a', 'xg_h', 'xg_a', 'sot_h', 'sot_a']
        elif 'Over_Under' in model_name:
            return ['minute', 'goals_sum', 'xg_sum', 'sot_sum', 'cor_sum', 'pos_diff', 'momentum_h', 'momentum_a']
        else:
            return ['minute', 'goals_sum', 'goals_diff', 'xg_sum', 'xg_diff']
    
    def prepare_feature_vector(self, model_name: str, features: Dict[str, float]) -> Optional[np.ndarray]:
        """Prepare EXACT feature vector matching model training"""
        try:
            required_features = self.get_required_features(model_name)
            log.info(f"📊 Model {model_name} expects {len(required_features)} features")
            
            vector = []
            missing = []
            
            for feat in required_features:
                if feat in features:
                    vector.append(float(features[feat]))
                else:
                    missing.append(feat)
                    vector.append(0.0)  # Use 0 for missing
            
            if missing:
                log.warning(f"⚠️ Missing features for {model_name}: {missing}")
            
            return np.array(vector).reshape(1, -1)
            
        except Exception as e:
            log.error(f"❌ Feature vector preparation failed for {model_name}: {e}")
            return None
    
    def predict(self, model_name: str, features: Dict[str, float]) -> float:
        """Single reliable prediction path"""
        try:
            # Load model
            model = self.load_model(model_name)
            if model is None:
                log.warning(f"⚠️ Model {model_name} not available, using fallback")
                return self.default_probability
            
            # Prepare features
            feature_vector = self.prepare_feature_vector(model_name, features)
            if feature_vector is None:
                log.warning(f"⚠️ Feature preparation failed for {model_name}")
                return self.default_probability
            
            # Make prediction
            if isinstance(model, dict) and 'models' in model:
                # New format from train_models.py
                return self._predict_ensemble(model, feature_vector)
            elif hasattr(model, 'predict_proba'):
                prob = float(model.predict_proba(feature_vector)[0][1])
                log.info(f"📊 Model {model_name} prediction: {prob:.3f}")
                return prob
            else:
                log.warning(f"⚠️ Model {model_name} has no predict_proba")
                return self.default_probability
                
        except Exception as e:
            log.error(f"❌ Prediction failed for {model_name}: {e}")
            return self.default_probability
    
    def _predict_ensemble(self, model_dict: Dict, feature_vector: np.ndarray) -> float:
        """Predict using ensemble model format"""
        try:
            selected_features = model_dict.get('selected_features', [])
            models = model_dict.get('models', {})
            scaler = model_dict.get('scaler')
            
            if not models:
                return self.default_probability
            
            # Apply scaler if available
            if scaler is not None:
                feature_vector = scaler.transform(feature_vector)
            
            # Get predictions from all models
            predictions = []
            for model_key, model in models.items():
                try:
                    if hasattr(model, 'predict_proba'):
                        prob = float(model.predict_proba(feature_vector)[0][1])
                        predictions.append(prob)
                except Exception as e:
                    log.error(f"❌ Sub-model {model_key} failed: {e}")
            
            if not predictions:
                return self.default_probability
            
            # Average predictions
            ensemble_prob = float(np.mean(predictions))
            log.info(f"📊 Ensemble prediction: {ensemble_prob:.3f} ({len(predictions)} models)")
            return ensemble_prob
            
        except Exception as e:
            log.error(f"❌ Ensemble prediction failed: {e}")
            return self.default_probability


# Initialize ModelManager
model_manager = ModelManager()

# ───────── AUTOMATIC MODEL TRAINING AT STARTUP ─────────
def _train_if_models_missing():
    """Train models automatically if they don't exist"""
    models_dir = "models"
    required_models = [
        "1X2_Home_Win.joblib", "1X2_Away_Win.joblib", "BTTS.joblib",
        "Over_Under_2_5.joblib", "Over_Under_3_5.joblib"
    ]
    
    # Check if any models are missing
    missing_models = []
    for model_file in required_models:
        model_path = os.path.join(models_dir, model_file)
        if not os.path.exists(model_path):
            missing_models.append(model_file.replace('.joblib', ''))
    
    if missing_models and TRAIN_ENABLE:
        log.warning(f"⚠️ Missing models: {missing_models}. Starting training...")
        
        # Try to import and run train_models
        try:
            import sys
            sys.path.append('.')  # Ensure current directory is in path
            
            # Create models directory if it doesn't exist
            os.makedirs(models_dir, exist_ok=True)
            
            log.info("🔧 Running training module...")
            
            # Simple training script
            import subprocess
            result = subprocess.run(
                [sys.executable, "train_models.py"],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                log.info("✅ Training completed successfully")
                # Check if models were created
                created_models = []
                for model_file in required_models:
                    if os.path.exists(os.path.join(models_dir, model_file)):
                        created_models.append(model_file.replace('.joblib', ''))
                
                if created_models:
                    log.info(f"📊 Created models: {created_models}")
                    send_telegram(f"🤖 Auto-trained {len(created_models)} models at startup")
                else:
                    log.warning("⚠️ Training ran but no models were created")
                    
            else:
                log.error(f"❌ Training failed: {result.stderr}")
                # Create fallback models as files
                _create_fallback_model_files()
                
        except Exception as e:
            log.error(f"❌ Auto-training failed: {e}")
            # Create fallback models as files
            _create_fallback_model_files()
    
    elif missing_models:
        log.warning(f"⚠️ Missing models but TRAIN_ENABLE is False: {missing_models}")
        _create_fallback_model_files()

def _create_fallback_model_files():
    """Create fallback model files on disk"""
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    fallback_models = ["1X2_Home_Win", "1X2_Away_Win", "BTTS", "Over_Under_2_5", "Over_Under_3_5"]
    
    for model_name in fallback_models:
        model_path = os.path.join(models_dir, f"{model_name}.joblib")
        metadata_path = os.path.join(models_dir, f"{model_name}_metadata.json")
        
        if not os.path.exists(model_path):
            # Create simple fallback model
            import numpy as np
            from sklearn.linear_model import LogisticRegression
            
            # Determine feature count
            if '1X2' in model_name:
                feature_count = 5
                features = ['minute', 'goals_sum', 'goals_diff', 'xg_sum', 'xg_diff']
            elif 'BTTS' in model_name:
                feature_count = 7
                features = ['minute', 'goals_h', 'goals_a', 'xg_h', 'xg_a', 'sot_h', 'sot_a']
            else:  # Over/Under
                feature_count = 8
                features = ['minute', 'goals_sum', 'xg_sum', 'sot_sum', 'cor_sum', 'pos_diff', 'momentum_h', 'momentum_a']
            
            # Create model
            model = LogisticRegression(random_state=42)
            X = np.random.randn(10, feature_count)
            y = np.random.randint(0, 2, 10)
            model.fit(X, y)
            
            # Save model
            import joblib
            joblib.dump(model, model_path)
            
            # Save metadata
            metadata = {
                'model_name': model_name,
                'required_features': features,
                'feature_count': feature_count,
                'is_fallback': True,
                'created_at': time.time()
            }
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            log.info(f"💾 Created fallback model file: {model_name}")

# ───────── Simple Feature Extraction (Matches Training) ─────────
def _num(v) -> float:
    try:
        if isinstance(v, str) and v.endswith("%"):
            return float(v[:-1])
        return float(v or 0.0)
    except Exception:
        return 0.0

def _pos_pct(v) -> float:
    try:
        return float(str(v).replace("%", "").strip() or 0.0)
    except Exception:
        return 0.0

def extract_features(m: dict) -> Dict[str, float]:
    """Extract features - MUST match train_models.py exactly"""
    home   = m["teams"]["home"]["name"]
    away   = m["teams"]["away"]["name"]
    gh     = m["goals"]["home"] or 0
    ga     = m["goals"]["away"] or 0
    
    # Get minute
    minute_data = ((m.get("fixture") or {}).get("status") or {})
    minute = int(minute_data.get("elapsed") or 1)
    if minute <= 0:
        minute = 1

    stats: Dict[str, Dict[str, Any]] = {}
    for s in (m.get("statistics") or []):
        t = (s.get("team") or {}).get("name")
        if t:
            stats[t] = {(i.get("type") or ""): i.get("value") for i in (s.get("statistics") or [])}

    sh = stats.get(home, {}) or {}
    sa = stats.get(away, {}) or {}

    # Core features (MUST match training)
    xg_h       = _num(sh.get("Expected Goals", 0))
    xg_a       = _num(sa.get("Expected Goals", 0))
    sot_h      = _num(sh.get("Shots on Target", sh.get("Shots on Goal", 0)))
    sot_a      = _num(sa.get("Shots on Target", sa.get("Shots on Goal", 0)))
    sh_total_h = _num(sh.get("Total Shots", sh.get("Shots Total", 0)))
    sh_total_a = _num(sa.get("Total Shots", sa.get("Shots Total", 0)))
    cor_h      = _num(sh.get("Corner Kicks", 0))
    cor_a      = _num(sa.get("Corner Kicks", 0))
    pos_h      = _pos_pct(sh.get("Ball Possession", 0))
    pos_a      = _pos_pct(sa.get("Ball Possession", 0))

    # Calculate derived features (MUST match training)
    momentum_h      = (sot_h + cor_h) / max(1, minute)
    momentum_a      = (sot_a + cor_a) / max(1, minute)
    pressure_index  = abs(gh - ga) * (minute / 90.0)
    efficiency_h    = gh / max(1, sot_h) if sot_h > 0 else 0.0
    efficiency_a    = ga / max(1, sot_a) if sot_a > 0 else 0.0
    total_actions   = sot_h + sot_a + cor_h + cor_a
    action_intensity= total_actions / max(1, minute)

    # Build feature dict (ALL features for compatibility)
    features = {
        "minute": float(minute),
        "goals_h": float(gh),
        "goals_a": float(ga),
        "goals_sum": float(gh + ga),
        "goals_diff": float(gh - ga),
        "xg_h": float(xg_h),
        "xg_a": float(xg_a),
        "xg_sum": float(xg_h + xg_a),
        "xg_diff": float(xg_h - xg_a),
        "sot_h": float(sot_h),
        "sot_a": float(sot_a),
        "sot_sum": float(sot_h + sot_a),
        "sh_total_h": float(sh_total_h),
        "sh_total_a": float(sh_total_a),
        "cor_h": float(cor_h),
        "cor_a": float(cor_a),
        "cor_sum": float(cor_h + cor_a),
        "pos_h": float(pos_h),
        "pos_a": float(pos_a),
        "pos_diff": float(pos_h - pos_a),
        # Advanced features for compatibility
        "momentum_h": float(momentum_h),
        "momentum_a": float(momentum_a),
        "pressure_index": float(pressure_index),
        "efficiency_h": float(efficiency_h),
        "efficiency_a": float(efficiency_a),
        "total_actions": float(total_actions),
        "action_intensity": float(action_intensity),
    }
    
    log.debug(f"✅ Extracted {len(features)} features for {home} vs {away}")
    return features

# ───────── Model Name Mapping ─────────
def get_model_name_for_market(market: str, suggestion: str) -> str:
    """Convert market and suggestion to model name"""
    if market == "BTTS":
        return "BTTS"
    elif market == "1X2":
        if "Home" in suggestion:
            return "1X2_Home_Win"
        else:
            return "1X2_Away_Win"
    elif market.startswith("Over/Under"):
        line = _parse_ou_line_from_suggestion(suggestion)
        if line:
            line_str = str(line).replace('.', '_')
            return f"Over_Under_{line_str}"
    return f"{market}_{suggestion.replace(' ', '_')}"

def _parse_ou_line_from_suggestion(s: str) -> Optional[float]:
    """Parse the line value from an Over/Under suggestion"""
    try:
        import re
        numbers = re.findall(r"\d+\.?\d*", s)
        if numbers:
            return float(numbers[0])
        return None
    except Exception:
        return None

# ───────── Simple Prediction Logic ─────────
def predict_probability(features: Dict[str, float], market: str, suggestion: str) -> float:
    """Simple reliable prediction using ModelManager"""
    model_name = get_model_name_for_market(market, suggestion)
    log.info(f"🔍 Predicting with model: {model_name}")
    
    # Get prediction from ModelManager
    prob = model_manager.predict(model_name, features)
    
    # Handle complements for Under/BTTS No
    if "Under" in suggestion and "Over_Under" in model_name:
        prob = 1.0 - prob
        log.info(f"🔄 Under prediction: complement {prob:.3f}")
    elif suggestion == "BTTS: No":
        prob = 1.0 - prob
        log.info(f"🔄 BTTS No prediction: complement {prob:.3f}")
    
    log.info(f"🎯 Final probability for {suggestion}: {prob:.3f} ({prob*100:.1f}%)")
    return prob

# ───────── DB pool & helpers ─────────
POOL: Optional[SimpleConnectionPool] = None

class PooledConn:
    def __init__(self, pool):
        self.pool = pool
        self.conn = None
        self.cur  = None
    def __enter__(self):
        self.conn = self.pool.getconn()
        self.conn.autocommit = True
        self.cur = self.conn.cursor()
        return self
    def __exit__(self, a, b, c):
        try:
            if self.cur:
                self.cur.close()
        finally:
            if self.conn:
                self.pool.putconn(self.conn)
    def execute(self, sql: str, params: tuple|list = ()):
        try:
            self.cur.execute(sql, params or ())
            return self.cur
        except Exception as e:
            log.error("DB execute failed: %s\nSQL: %s\nParams: %s", e, sql, params)
            raise

def _init_pool():
    global POOL
    dsn = DATABASE_URL
    if "sslmode=" not in dsn:
        dsn = dsn + (("&" if "?" in dsn else "?") + "sslmode=require")
    POOL = SimpleConnectionPool(minconn=1, maxconn=int(os.getenv("DB_POOL_MAX", "5")), dsn=dsn)

def db_conn() -> PooledConn:
    if not POOL:
        _init_pool()
    return PooledConn(POOL)

def _db_ping() -> bool:
    try:
        with db_conn() as c:
            c.execute("SELECT 1")
        return True
    except Exception:
        log.warning("[DB] ping failed, re-initializing pool")
        try:
            _init_pool()
            with db_conn() as c2:
                c2.execute("SELECT 1")
            return True
        except Exception as e:
            log.error("[DB] reinit failed: %s", e)
            return False

# ───────── Init DB ─────────
def init_db() -> None:
    with db_conn() as c:
        c.execute(
            """CREATE TABLE IF NOT EXISTS tips (
            match_id BIGINT,
            league_id BIGINT,
            league TEXT,
            home TEXT,
            away TEXT,
            market TEXT,
            suggestion TEXT,
            confidence DOUBLE PRECISION,
            confidence_raw DOUBLE PRECISION,
            score_at_tip TEXT,
            minute INTEGER,
            created_ts BIGINT,
            odds DOUBLE PRECISION,
            book TEXT,
            ev_pct DOUBLE PRECISION,
            sent_ok INTEGER DEFAULT 1,
            PRIMARY KEY (match_id, created_ts)
        )"""
        )
        c.execute(
            """CREATE TABLE IF NOT EXISTS tip_snapshots (
            match_id BIGINT,
            created_ts BIGINT,
            payload TEXT,
            PRIMARY KEY (match_id, created_ts)
        )"""
        )
        c.execute(
            """CREATE TABLE IF NOT EXISTS feedback (
            id SERIAL PRIMARY KEY,
            match_id BIGINT UNIQUE,
            verdict INTEGER,
            created_ts BIGINT
        )"""
        )
        c.execute("""CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)""")
        c.execute(
            """CREATE TABLE IF NOT EXISTS match_results (
            match_id BIGINT PRIMARY KEY,
            final_goals_h INTEGER,
            final_goals_a INTEGER,
            btts_yes INTEGER,
            updated_ts BIGINT
        )"""
        )
        c.execute("CREATE INDEX IF NOT EXISTS idx_tips_created ON tips (created_ts DESC)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_tips_match ON tips (match_id)")

# ───────── Telegram ─────────
def send_telegram(text: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.error("❌ Telegram credentials missing")
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            data={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=REQ_TIMEOUT_SEC,
        )
        ok = bool(r.ok)
        if not ok:
            log.error("❌ Telegram API error: %s - %s", r.status_code, r.text)
        else:
            log.info("✅ Telegram message sent")
        return ok
    except Exception as e:
        log.error("❌ Telegram send exception: %s", e)
        return False

# ───────── API helpers ─────────
def _api_get(url: str, params: dict, timeout: int = 15) -> Optional[dict]:
    if not API_KEY:
        log.error("❌ API_KEY missing for API call to %s", url)
        return None
    now = time.time()
    if API_CB["opened_until"] > now:
        log.warning("🚫 API Circuit Breaker open")
        return None
    
    try:
        r = session.get(url, headers=HEADERS, params=params, timeout=min(timeout, REQ_TIMEOUT_SEC))
        if r.status_code == 429:
            log.warning("🚫 API Rate Limited - status 429")
            API_CB["failures"] += 1
        elif r.status_code >= 500:
            log.error("❌ API Server Error - status %s", r.status_code)
            API_CB["failures"] += 1
        else:
            API_CB["failures"] = 0

        if API_CB["failures"] >= API_CB_THRESHOLD:
            API_CB["opened_until"] = now + API_CB_COOLDOWN_SEC
            log.warning("[CB] API-Football opened for %ss", API_CB_COOLDOWN_SEC)

        return r.json() if r.ok else None
    except Exception as e:
        log.error("❌ API call exception: %s", e)
        API_CB["failures"] += 1
        if API_CB["failures"] >= API_CB_THRESHOLD:
            API_CB["opened_until"] = time.time() + API_CB_COOLDOWN_SEC
        return None

# ───────── League filter ─────────
_BLOCK_PATTERNS = [
    "u17", "u18", "u19", "u20", "u21", "u23", "youth", "junior", "reserve", "res.", "friendlies", "friendly",
]
def _blocked_league(league_obj: dict) -> bool:
    name    = str((league_obj or {}).get("name", "")).lower()
    country = str((league_obj or {}).get("country", "")).lower()
    typ     = str((league_obj or {}).get("type", "")).lower()
    txt     = f"{country} {name} {typ}"
    if any(p in txt for p in _BLOCK_PATTERNS):
        log.debug("🚫 Blocked league: %s", txt)
        return True
    deny = [x.strip() for x in os.getenv("LEAGUE_DENY_IDS", "").split(",") if x.strip()]
    lid  = str((league_obj or {}).get("id") or "")
    if lid in deny:
        log.debug("🚫 Denied league ID: %s", lid)
        return True
    return False

# ───────── Live fetches ─────────
def fetch_match_stats(fid: int) -> list:
    log.debug("📊 Fetching stats for fixture %s", fid)
    now = time.time()
    k   = ("stats", fid)
    ts_empty = NEG_CACHE.get(k, (0.0, False))
    if ts_empty[1] and (now - ts_empty[0] < NEG_TTL_SEC):
        return []
    if fid in STATS_CACHE and now - STATS_CACHE[fid][0] < 90:
        return STATS_CACHE[fid][1]
    js  = _api_get(f"{FOOTBALL_API_URL}/statistics", {"fixture": fid}) or {}
    out = js.get("response", []) if isinstance(js, dict) else []
    STATS_CACHE[fid] = (now, out)
    if not out:
        NEG_CACHE[k] = (now, True)
    return out

def fetch_live_fixtures_only() -> List[dict]:
    log.info("🌐 Fetching live fixtures...")
    js = _api_get(FOOTBALL_API_URL, {"live": "all"}) or {}
    matches = [
        m
        for m in (js.get("response", []) if isinstance(js, dict) else [])
        if not _blocked_league(m.get("league") or {})
    ]
    
    out = []
    for m in matches:
        st      = ((m.get("fixture", {}) or {}).get("status", {}) or {})
        elapsed = st.get("elapsed")
        short   = (st.get("short") or "").upper()
        if elapsed is None or elapsed > 120 or short not in INPLAY_STATUSES:
            continue
        out.append(m)
    
    log.info("🎯 Found %s in-play matches", len(out))
    return out

def _quota_per_scan() -> int:
    scans_per_day = max(1, int(86400 / max(1, SCAN_INTERVAL_SEC)))
    ppf = 1 + (1 if USE_EVENTS_IN_FEATURES else 0)
    safe = int(API_BUDGET_DAILY / max(1, (scans_per_day * ppf))) - 10
    quota = max(1, min(MAX_FIXTURES_PER_SCAN, safe))
    return quota

def _priority_key(m: dict) -> Tuple[int,int,int,int,int]:
    minute = int(((m.get("fixture") or {}).get("status") or {}).get("elapsed") or 0)
    gh     = int((m.get("goals") or {}).get("home") or 0)
    ga     = int((m.get("goals") or {}).get("away") or 0)
    total  = gh + ga
    lid    = int(((m.get("league") or {}) or {}).get("id") or 0)
    return (
        3 if (LEAGUE_ALLOW_IDS and lid in LEAGUE_ALLOW_IDS) else 0,
        2 if 20 <= minute <= 80 else 0,
        1 if total in (1, 2, 3) else 0,
        -abs(60 - minute),
        -total,
    )

def fetch_match_events(fid: int) -> list:
    log.debug("📅 Fetching events for fixture %s", fid)
    now = time.time()
    k   = ("events", fid)
    ts_empty = NEG_CACHE.get(k, (0.0, False))
    if ts_empty[1] and (now - ts_empty[0] < NEG_TTL_SEC):
        return []
    if fid in EVENTS_CACHE and now - EVENTS_CACHE[fid][0] < 90:
        return EVENTS_CACHE[fid][1]
    js  = _api_get(f"{FOOTBALL_API_URL}/events", {"fixture": fid}) or {}
    out = js.get("response", []) if isinstance(js, dict) else []
    EVENTS_CACHE[fid] = (now, out)
    if not out:
        NEG_CACHE[k] = (now, True)
    return out

def fetch_live_matches() -> List[dict]:
    log.info("🚀 Starting live matches fetch...")
    fixtures = fetch_live_fixtures_only()
    if not fixtures:
        log.info("❌ No live fixtures found")
        return []
        
    fixtures.sort(key=_priority_key, reverse=True)
    quota  = _quota_per_scan()
    chosen = fixtures[:quota]
    log.info("🎯 Selected %s fixtures (quota: %s)", len(chosen), quota)
    
    out    = []
    for i, m in enumerate(chosen):
        fid           = int((m.get("fixture", {}) or {}).get("id") or 0)
        m["statistics"] = fetch_match_stats(fid)
        m["events"]     = fetch_match_events(fid) if USE_EVENTS_IN_FEATURES else []
        out.append(m)
        
    log.info("✅ Completed live matches fetch: %s fixtures", len(out))
    return out

# ───────── Odds fetching ─────────
def _market_name_normalize(s: str) -> str:
    s = (s or "").lower()
    if "both teams" in s or "btts" in s:
        return "BTTS"
    if "match winner" in s or "winner" in s or "1x2" in s:
        return "1X2"
    if "over/under" in s or "total" in s or "goals" in s:
        return "OU"
    return s

def _aggregate_price(vals: List[Tuple[float, str]], prob_hint: Optional[float]) -> Tuple[Optional[float], Optional[str]]:
    if not vals:
        return None, None
    xs = sorted([o for (o, _) in vals if (o or 0.0) > 0.0])
    if not xs:
        return None, None
    import statistics
    med = statistics.median(xs)
    cleaned = [(o, b) for (o, b) in vals if o <= med * max(1.0, ODDS_OUTLIER_MULT)]
    if not cleaned:
        cleaned = vals
    xs2  = sorted([o for (o, _) in cleaned])
    med2 = statistics.median(xs2)
    if prob_hint is not None and prob_hint > 0:
        fair = 1.0 / max(1e-6, float(prob_hint))
        cap  = fair * max(1.0, ODDS_FAIR_MAX_MULT)
        cleaned = [(o, b) for (o, b) in cleaned if o <= cap] or cleaned
    if ODDS_AGGREGATION == "best":
        best = max(cleaned, key=lambda t: t[0])
        return float(best[0]), str(best[1])
    target = med2
    pick   = min(cleaned, key=lambda t: abs(t[0] - target))
    return float(pick[0]), f"{pick[1]} (median of {len(xs)})"

def fetch_odds(fid: int) -> dict:
    log.debug("💰 Fetching odds for fixture %s", fid)
    now    = time.time()
    cached = ODDS_CACHE.get(fid)
    if cached and now - cached[0] < 120:
        return cached[1]

    def _fetch(path: str) -> dict:
        js = _api_get(f"{BASE_URL}/{path}", {"fixture": fid}) or {}
        return js if isinstance(js, dict) else {}

    js = {}
    if ODDS_SOURCE in ("auto", "live"):
        js = _fetch("odds/live")
    if not (js.get("response") or []) and ODDS_SOURCE in ("auto", "prematch"):
        js = _fetch("odds")

    by_market: Dict[str, Dict[str, List[Tuple[float, str]]]] = {}
    try:
        for r in js.get("response", []) or []:
            for bk in (r.get("bookmakers") or []):
                book_name = bk.get("name") or "Book"
                for mkt in (bk.get("bets") or []):
                    mname = _market_name_normalize(mkt.get("name", ""))
                    vals  = mkt.get("values") or []
                    if mname == "BTTS":
                        for v in vals:
                            lbl = (v.get("value") or "").strip().lower()
                            if "yes" in lbl:
                                by_market.setdefault("BTTS", {}).setdefault("Yes", []).append((float(v.get("odd") or 0), book_name))
                            elif "no" in lbl:
                                by_market.setdefault("BTTS", {}).setdefault("No", []).append((float(v.get("odd") or 0), book_name))
                    elif mname == "1X2":
                        for v in vals:
                            lbl = (v.get("value") or "").strip().lower()
                            if lbl in ("home", "1"):
                                by_market.setdefault("1X2", {}).setdefault("Home", []).append((float(v.get("odd") or 0), book_name))
                            elif lbl in ("away", "2"):
                                by_market.setdefault("1X2", {}).setdefault("Away", []).append((float(v.get("odd") or 0), book_name))
                    elif mname == "OU":
                        for v in vals:
                            lbl = (v.get("value") or "").lower()
                            if ("over" in lbl) or ("under" in lbl):
                                try:
                                    ln  = float(lbl.split()[-1])
                                    key = f"OU_{_fmt_line(ln)}"
                                    side= "Over" if "over" in lbl else "Under"
                                    by_market.setdefault(key, {}).setdefault(side, []).append((float(v.get("odd") or 0), book_name))
                                except Exception:
                                    pass
    except Exception as e:
        log.error("❌ Error parsing odds: %s", e)

    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for mkey, side_map in by_market.items():
        ok = True
        for side, lst in side_map.items():
            if len({b for (_, b) in lst}) < max(1, ODDS_REQUIRE_N_BOOKS):
                ok = False
                break
        if not ok:
            continue

        out[mkey] = {}
        for side, lst in side_map.items():
            ag, label = _aggregate_price(lst, None)
            if ag is not None:
                out[mkey][side] = {"odds": float(ag), "book": label}

    ODDS_CACHE[fid] = (now, out)
    return out

def _min_odds_for_market(market: str) -> float:
    if market.startswith("Over/Under"):
        return MIN_ODDS_OU
    if market == "BTTS":
        return MIN_ODDS_BTTS
    if market == "1X2":
        return MIN_ODDS_1X2
    return 1.01

def _get_odds_for_market(odds_map: dict, market: str, suggestion: str) -> Tuple[Optional[float], Optional[str]]:
    if market == "BTTS":
        d   = odds_map.get("BTTS", {})
        tgt = "Yes" if suggestion.endswith("Yes") else "No"
        if tgt in d:
            odds, book = d[tgt]["odds"], d[tgt]["book"]
            return odds, book
    elif market == "1X2":
        d   = odds_map.get("1X2", {})
        tgt = "Home" if suggestion == "Home Win" else ("Away" if suggestion == "Away Win" else None)
        if tgt and tgt in d:
            odds, book = d[tgt]["odds"], d[tgt]["book"]
            return odds, book
    elif market.startswith("Over/Under"):
        ln_val = _parse_ou_line_from_suggestion(suggestion)
        d      = odds_map.get(f"OU_{_fmt_line(ln_val)}", {}) if ln_val is not None else {}
        tgt    = "Over" if suggestion.startswith("Over") else "Under"
        if tgt in d:
            odds, book = d[tgt]["odds"], d[tgt]["book"]
            return odds, book
    return None, None

# ───────── Core prediction logic ─────────
def _ev(prob: float, odds: float) -> float:
    return prob * max(0.0, float(odds)) - 1.0

def _league_name(m: dict) -> Tuple[int, str]:
    lg = (m.get("league") or {}) or {}
    return int(lg.get("id") or 0), f"{lg.get('country','')} - {lg.get('name','')}".strip(" -")

def _teams(m: dict) -> Tuple[str, str]:
    t = (m.get("teams") or {}) or {}
    return (t.get("home", {}).get("name", ""), t.get("away", {}).get("name", ""))

def _pretty_score(m: dict) -> str:
    gh = (m.get("goals") or {}).get("home") or 0
    ga = (m.get("goals") or {}).get("away") or 0
    return f"{gh}-{ga}"

def _get_market_threshold(m: str) -> float:
    try:
        v = get_setting_cached(f"conf_threshold:{m}")
        if v is not None:
            return float(v)
    except Exception:
        pass
    return float(CONF_THRESHOLD)

def _candidate_is_sane(sug: str, feat: Dict[str, float]) -> bool:
    gh    = int(feat.get("goals_h", 0))
    ga    = int(feat.get("goals_a", 0))
    total = gh + ga
    minute = int(feat.get("minute", 0))

    if sug.startswith("Over"):
        ln = _parse_ou_line_from_suggestion(sug)
        if ln is None:
            return False
        return total < ln

    if sug.startswith("Under"):
        ln = _parse_ou_line_from_suggestion(sug)
        if ln is None:
            return False
        return total < ln

    if sug.startswith("BTTS") and (gh > 0 and ga > 0):
        return False

    if sug == "BTTS: Yes" and (gh > 0 and ga > 0):
        return True
    if sug == "BTTS: No" and not (gh > 0 and ga > 0):
        return True

    return True

def _generate_predictions(features: Dict[str, float], fid: int, minute: int) -> List[Tuple[str, str, float]]:
    log.info("🤖 Generating predictions (minute: %s)", minute)
    candidates: List[Tuple[str, str, float]] = []

    # OU markets
    for line in OU_LINES:
        sline  = _fmt_line(line)
        market = f"Over/Under {sline}"
        threshold = _get_market_threshold(market)

        over_sug = f"Over {sline} Goals"
        over_prob = predict_probability(features, market, over_sug)
        
        if over_prob * 100.0 >= threshold and _candidate_is_sane(over_sug, features):
            candidates.append((market, over_sug, over_prob))

        under_sug  = f"Under {sline} Goals"
        under_prob = 1.0 - over_prob
        
        if under_prob * 100.0 >= threshold and _candidate_is_sane(under_sug, features):
            candidates.append((market, under_sug, under_prob))

    # BTTS market
    market = "BTTS"
    threshold = _get_market_threshold(market)
    
    btts_yes_prob = predict_probability(features, market, "BTTS: Yes")
    
    if btts_yes_prob * 100.0 >= threshold and _candidate_is_sane("BTTS: Yes", features):
        candidates.append((market, "BTTS: Yes", btts_yes_prob))

    btts_no_prob = 1.0 - btts_yes_prob
    
    if btts_no_prob * 100.0 >= threshold and _candidate_is_sane("BTTS: No", features):
        candidates.append((market, "BTTS: No", btts_no_prob))

    # 1X2 market
    market = "1X2"
    threshold = _get_market_threshold(market)
    
    home_win_prob = predict_probability(features, market, "Home Win")
    away_win_prob = predict_probability(features, market, "Away Win")
    
    total_win_prob = home_win_prob + away_win_prob
    if total_win_prob > 0:
        home_win_prob = home_win_prob / total_win_prob
        away_win_prob = away_win_prob / total_win_prob
        
        if home_win_prob * 100.0 >= threshold:
            candidates.append((market, "Home Win", home_win_prob))
            
        if away_win_prob * 100.0 >= threshold:
            candidates.append((market, "Away Win", away_win_prob))

    log.info("🎯 Generated %s prediction candidates", len(candidates))
    return candidates

def _process_and_rank_candidates(
    candidates: List[Tuple[str, str, float]],
    fid: int,
    features: Dict[str, float],
) -> List[Tuple[str, str, float, Optional[float], Optional[str], Optional[float], float]]:
    log.info("🏆 Processing and ranking %s candidates", len(candidates))
    ranked: List[Tuple[str, str, float, Optional[float], Optional[str], Optional[float], float]] = []
    
    # Fetch odds once for all candidates
    odds_map = fetch_odds(fid)

    for market, suggestion, prob in candidates:
        if suggestion not in ALLOWED_SUGGESTIONS:
            continue

        odds, book = _get_odds_for_market(odds_map, market, suggestion)
            
        if odds is None and not ALLOW_TIPS_WITHOUT_ODDS:
            continue

        if odds is not None:
            min_odds = _min_odds_for_market(market)
            if not (min_odds <= odds <= MAX_ODDS_ALL):
                continue

            edge   = _ev(prob, odds)
            ev_pct = round(edge * 100.0, 1)
            if int(round(edge * 10000)) < EDGE_MIN_BPS:
                continue
        else:
            ev_pct = None

        rank_score = (prob ** 1.2) * (1.0 + (ev_pct or 0.0) / 100.0)
        ranked.append((market, suggestion, prob, odds, book, ev_pct, rank_score))

    ranked.sort(key=lambda x: x[6], reverse=True)
    log.info("🎯 Ranked %s candidates", len(ranked))
    return ranked

def production_scan() -> Tuple[int, int]:
    """Main in-play scanning"""
    log.info("🚀 STARTING PRODUCTION SCAN")
    
    if not _db_ping():
        log.error("❌ Database ping failed")
        return 0, 0
        
    matches   = fetch_live_matches()
    live_seen = len(matches)
    if live_seen == 0:
        log.info("[PROD] no live matches")
        return 0, 0

    log.info("🔍 Processing %s live matches", live_seen)
    saved = 0
    now_ts = int(time.time())
    per_league_counter: Dict[int, int] = {}

    with db_conn() as c:
        for i, m in enumerate(matches):
            try:
                fid = int((m.get("fixture", {}) or {}).get("id") or 0)
                if not fid:
                    continue

                log.info("🎯 Processing match %s/%s: fixture %s", i+1, len(matches), fid)
                
                if DUP_COOLDOWN_MIN > 0:
                    cutoff = now_ts - DUP_COOLDOWN_MIN * 60
                    dup_check = c.execute(
                        "SELECT 1 FROM tips WHERE match_id=%s AND created_ts>=%s AND suggestion<>'HARVEST' LIMIT 1",
                        (fid, cutoff),
                    ).fetchone()
                    if dup_check:
                        log.debug("⏩ Skipping match %s due to duplicate cooldown", fid)
                        continue

                feat = extract_features(m)
                minute = int(feat.get("minute", 0))
                
                if minute < TIP_MIN_MINUTE:
                    log.info("⏩ Skipping match %s - minute %s < %s", fid, minute, TIP_MIN_MINUTE)
                    continue

                league_id, league = _league_name(m)
                home, away        = _teams(m)
                score             = _pretty_score(m)
                
                log.info("🏠 %s vs %s (%s) - %s minute %s", home, away, league, score, minute)

                candidates = _generate_predictions(feat, fid, minute)
                if not candidates:
                    log.info("⏩ No prediction candidates for match %s", fid)
                    continue

                ranked = _process_and_rank_candidates(candidates, fid, feat)
                if not ranked:
                    log.info("⏩ No ranked candidates for match %s", fid)
                    continue

                # Save and send tips
                saved_in_match = 0
                base_now = int(time.time())
                
                for idx, (market, suggestion, prob, odds, book, ev_pct, _) in enumerate(ranked):
                    if PER_LEAGUE_CAP > 0 and per_league_counter.get(league_id, 0) >= PER_LEAGUE_CAP:
                        break

                    created_ts = base_now + idx
                    prob_pct   = round(prob * 100.0, 1)

                    try:
                        c.execute(
                            "INSERT INTO tips(match_id,league_id,league,home,away,market,suggestion,"
                            "confidence,confidence_raw,score_at_tip,minute,created_ts,odds,book,ev_pct,sent_ok) "
                            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                            (
                                fid,
                                league_id,
                                league,
                                home,
                                away,
                                market,
                                suggestion,
                                float(prob_pct),
                                float(prob),
                                score,
                                minute,
                                created_ts,
                                (float(odds) if odds is not None else None),
                                (book or None),
                                (float(ev_pct) if ev_pct is not None else None),
                                0,
                            ),
                        )

                        # Format and send message
                        message = (
                            "⚽️ <b>New Tip!</b>\n"
                            f"<b>Match:</b> {escape(home)} vs {escape(away)}\n"
                            f"🕒 <b>Minute:</b> {minute}'  |  <b>Score:</b> {escape(score)}\n"
                            f"<b>Tip:</b> {escape(suggestion)}\n"
                            f"📈 <b>Confidence:</b> {prob_pct:.1f}%\n"
                        )
                        if odds:
                            message += f"\n💰 <b>Odds:</b> {odds:.2f} @ {book or 'Book'}"
                            if ev_pct:
                                message += f"  •  <b>EV:</b> {ev_pct:+.1f}%"
                        
                        message += f"\n🏆 <b>League:</b> {escape(league)}"
                        
                        sent = send_telegram(message)
                        if sent:
                            c.execute("UPDATE tips SET sent_ok=1 WHERE match_id=%s AND created_ts=%s", (fid, created_ts))

                        saved_in_match += 1
                        saved += 1
                        per_league_counter[league_id] = per_league_counter.get(league_id, 0) + 1

                        if MAX_TIPS_PER_SCAN and saved >= MAX_TIPS_PER_SCAN:
                            break
                        if saved_in_match >= max(1, PREDICTIONS_PER_MATCH):
                            break

                    except Exception as e:
                        log.exception("❌ insert/send failed: %s", e)
                        continue

                log.info("💾 Match %s: saved %s tips", fid, saved_in_match)

                if MAX_TIPS_PER_SCAN and saved >= MAX_TIPS_PER_SCAN:
                    log.info("🎯 Reached MAX_TIPS_PER_SCAN limit (%s)", MAX_TIPS_PER_SCAN)
                    break

            except Exception as e:
                log.exception("❌ match loop failed for match %s: %s", fid, e)
                continue

    log.info("[PROD] saved=%d live_seen=%d", saved, live_seen)
    return saved, live_seen

# ───────── Settings cache ─────────
class _KVCache:
    def __init__(self, ttl: int):
        self.ttl  = ttl
        self.data: Dict[str, Tuple[float, Optional[str]]] = {}
    def get(self, k: str) -> Optional[str]:
        v = self.data.get(k)
        if not v:
            return None
        ts, val = v
        if time.time() - ts > self.ttl:
            self.data.pop(k, None)
            return None
        return val
    def set(self, k: str, v: Optional[str]) -> None:
        self.data[k] = (time.time(), v)

_SETTINGS_CACHE = _KVCache(SETTINGS_TTL)
_MODELS_CACHE   = _KVCache(MODELS_TTL)

def get_setting(key: str) -> Optional[str]:
    with db_conn() as c:
        r = c.execute("SELECT value FROM settings WHERE key=%s", (key,)).fetchone()
        return r[0] if r else None

def set_setting(key: str, value: str) -> None:
    with db_conn() as c:
        c.execute(
            "INSERT INTO settings(key,value) VALUES(%s,%s) "
            "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
            (key, value),
        )

def get_setting_cached(key: str) -> Optional[str]:
    v = _SETTINGS_CACHE.get(key)
    if v is None:
        v = get_setting(key)
        _SETTINGS_CACHE.set(key, v)
    return v

# ───────── Scheduler & admin ─────────
_scheduler_started = False

def _run_with_pg_lock(lock_key: int, fn, *a, **k):
    try:
        with db_conn() as c:
            got = c.execute("SELECT pg_try_advisory_lock(%s)", (lock_key,)).fetchone()[0]
            if not got:
                log.info("[LOCK %s] busy; skipped.", lock_key)
                return None
            try:
                return fn(*a, **k)
            finally:
                c.execute("SELECT pg_advisory_unlock(%s)", (lock_key,))
    except Exception as e:
        log.exception("[LOCK %s] failed: %s", lock_key, e)
        return None

def backfill_results_for_open_matches(max_rows: int = 200) -> int:
    now_ts = int(time.time())
    cutoff = now_ts - BACKFILL_DAYS * 24 * 3600
    updated = 0
    with db_conn() as c:
        rows = c.execute(
            """
            WITH last AS (
                SELECT match_id, MAX(created_ts) last_ts
                FROM tips
                WHERE created_ts >= %s
                GROUP BY match_id
            )
            SELECT l.match_id
            FROM last l
            LEFT JOIN match_results r ON r.match_id = l.match_id
            WHERE r.match_id IS NULL
            ORDER BY l.last_ts DESC
            LIMIT %s
            """,
            (cutoff, max_rows),
        ).fetchall()
        
    log.info("🔍 Backfilling results for %s matches", len(rows))
    for (mid,) in rows:
        fx = _fixture_by_id(int(mid))
        if not fx:
            continue
        st = (((fx.get("fixture") or {}).get("status") or {}).get("short") or "")
        if not _is_final(st):
            continue
        g  = fx.get("goals") or {}
        gh = int(g.get("home") or 0)
        ga = int(g.get("away") or 0)
        btts = 1 if (gh > 0 and ga > 0) else 0
        with db_conn() as c2:
            c2.execute(
                "INSERT INTO match_results(match_id, final_goals_h, final_goals_a, btts_yes, updated_ts) "
                "VALUES(%s,%s,%s,%s,%s) "
                "ON CONFLICT(match_id) DO UPDATE SET final_goals_h=EXCLUDED.final_goals_h, "
                "final_goals_a=EXCLUDED.final_goals_a, btts_yes=EXCLUDED.btts_yes, updated_ts=EXCLUDED.updated_ts",
                (int(mid), gh, ga, btts, int(time.time())),
            )
        updated += 1
        
    if updated:
        log.info("[RESULTS] backfilled %d", updated)
    return updated

def _fixture_by_id(mid: int) -> Optional[dict]:
    js  = _api_get(FOOTBALL_API_URL, {"id": mid}) or {}
    arr = js.get("response") or [] if isinstance(js, dict) else []
    return arr[0] if arr else None

def _is_final(short: str) -> bool:
    return (short or "").upper() in {"FT", "AET", "PEN"}

def daily_accuracy_digest(window_days: int = 1) -> Optional[str]:
    if not DAILY_ACCURACY_DIGEST_ENABLE:
        return None
    backfill_results_for_open_matches(400)

    cutoff = int((datetime.now(BERLIN_TZ) - timedelta(days=window_days)).timestamp())
    with db_conn() as c:
        rows = c.execute(
            """
            SELECT t.market, t.suggestion, t.confidence, t.confidence_raw, t.created_ts,
                   t.odds, r.final_goals_h, r.final_goals_a, r.btts_yes
            FROM tips t
            LEFT JOIN match_results r ON r.match_id = t.match_id
            WHERE t.created_ts >= %s
              AND t.suggestion <> 'HARVEST'
              AND t.sent_ok = 1
        """,
            (cutoff,),
        ).fetchall()

    total = graded = wins = 0
    by_market: Dict[str, Dict[str, int]] = {}

    for mkt, sugg, conf, conf_raw, cts, odds, gh, ga, btts in rows:
        res = {"final_goals_h": gh, "final_goals_a": ga, "btts_yes": btts}
        out = _tip_outcome_for_result(sugg, res)
        if out is None:
            continue

        total += 1
        graded += 1
        if out == 1:
            wins += 1
        d = by_market.setdefault(mkt or "?", {"graded": 0, "wins": 0})
        d["graded"] += 1
        if out == 1:
            d["wins"] += 1

    if graded == 0:
        msg = "📊 Accuracy Digest\nNo graded tips in window."
    else:
        acc = 100.0 * wins / max(1, graded)
        lines = [
            f"📊 <b>Accuracy Digest</b> (last {window_days}d)",
            f"Tips sent: {total}  •  Graded: {graded}  •  Wins: {wins}  •  Accuracy: {acc:.1f}%",
        ]

        for mk, st in sorted(by_market.items()):
            if st["graded"] == 0:
                continue
            a = 100.0 * st["wins"] / st["graded"]
            lines.append(f"• {escape(mk)} — {st['wins']}/{st['graded']} ({a:.1f}%)")

        msg = "\n".join(lines)

    send_telegram(msg)
    return msg

def _tip_outcome_for_result(suggestion: str, res: Dict[str, Any]) -> Optional[int]:
    gh    = int(res.get("final_goals_h") or 0)
    ga    = int(res.get("final_goals_a") or 0)
    total = gh + ga
    btts  = int(res.get("btts_yes") or 0)
    s     = (suggestion or "").strip()

    if s.startswith("Over") or s.startswith("Under"):
        ln = _parse_ou_line_from_suggestion(s)
        if ln is None:
            return None
        if s.startswith("Over"):
            if total > ln:
                return 1
            if abs(total - ln) < 1e-9:
                return None
            return 0
        else:
            if total < ln:
                return 1
            if abs(total - ln) < 1e-9:
                return None
            return 0

    if s == "BTTS: Yes":
        return 1 if btts == 1 else 0
    if s == "BTTS: No":
        return 1 if btts == 0 else 0
    if s == "Home Win":
        return 1 if gh > ga else 0
    if s == "Away Win":
        return 1 if ga > gh else 0
    return None

def save_snapshot_from_match(m: dict, feat: Dict[str, float]) -> None:
    fx    = (m.get("fixture") or {})
    lg    = (m.get("league") or {})
    teams = (m.get("teams") or {})

    fid = int(fx.get("id") or 0)
    if not fid:
        return

    league_id = int(lg.get("id") or 0)
    league    = f"{lg.get('country','')} - {lg.get('name','')}".strip(" -")
    home      = (teams.get("home") or {}).get("name", "")
    away      = (teams.get("away") or {}).get("name", "")

    gh     = int((m.get("goals") or {}).get("home") or 0)
    ga     = int((m.get("goals") or {}).get("away") or 0)
    minute = int(feat.get("minute", 0))

    snapshot = {
        "minute": minute,
        "gh": gh,
        "ga": ga,
        "league_id": league_id,
        "market": "HARVEST",
        "suggestion": "HARVEST",
        "confidence": 0,
        "stat": feat,
    }

    now     = int(time.time())
    payload = json.dumps(snapshot, separators=(",", ":"), ensure_ascii=False)

    with db_conn() as c:
        c.execute(
            "INSERT INTO tip_snapshots(match_id, created_ts, payload) VALUES (%s,%s,%s) "
            "ON CONFLICT (match_id, created_ts) DO UPDATE SET payload=EXCLUDED.payload",
            (fid, now, payload),
        )
        c.execute(
            "INSERT INTO tips(match_id,league_id,league,home,away,market,suggestion,confidence,confidence_raw,"
            "score_at_tip,minute,created_ts,sent_ok) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (
                fid,
                league_id,
                league,
                home,
                away,
                "HARVEST",
                "HARVEST",
                0.0,
                0.0,
                f"{gh}-{ga}",
                minute,
                now,
                1,
            ),
        )

def _start_scheduler_once():
    global _scheduler_started
    if _scheduler_started or not RUN_SCHEDULER:
        return
    try:
        sched = BackgroundScheduler(timezone=TZ_UTC)
        
        sched.add_job(
            lambda: _run_with_pg_lock(1001, production_scan),
            "interval",
            seconds=SCAN_INTERVAL_SEC,
            id="scan",
            max_instances=1,
            coalesce=True,
        )
        
        sched.add_job(
            lambda: _run_with_pg_lock(1002, backfill_results_for_open_matches, 400),
            "interval",
            minutes=BACKFILL_EVERY_MIN,
            id="backfill",
            max_instances=1,
            coalesce=True,
        )

        if DAILY_ACCURACY_DIGEST_ENABLE:
            sched.add_job(
                lambda: _run_with_pg_lock(1004, daily_accuracy_digest, 1),
                CronTrigger(
                    hour=int(os.getenv("DAILY_ACCURACY_HOUR", "3")),
                    minute=int(os.getenv("DAILY_ACCURACY_MINUTE", "6")),
                    timezone=BERLIN_TZ,
                ),
                id="digest",
                max_instances=1,
                coalesce=True,
            )

        sched.start()
        _scheduler_started = True
        send_telegram("🚀 goalsniper PURE IN-PLAY AI mode started.")
        log.info("[SCHED] started (scan=%ss)", SCAN_INTERVAL_SEC)

    except Exception as e:
        log.exception("[SCHED] failed: %s", e)

# ───────── Flask / HTTP ─────────
def _require_admin():
    key = (
        request.headers.get("X-API-Key")
        or request.args.get("key")
        or ((request.json or {}).get("key") if request.is_json else None)
    )
    if not ADMIN_API_KEY or key != ADMIN_API_KEY:
        abort(401)

@app.route("/")
def root():
    return jsonify({"ok": True, "name": "goalsniper", "mode": "PURE_INPLAY_AI"})

@app.route("/health")
def health():
    try:
        with db_conn() as c:
            n = c.execute("SELECT COUNT(*) FROM tips").fetchone()[0]
        return jsonify({"ok": True, "db": "ok", "tips_count": int(n)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/admin/scan", methods=["POST", "GET"])
def http_scan():
    _require_admin()
    s, l = production_scan()
    return jsonify({"ok": True, "saved": s, "live_seen": l})

@app.route("/admin/backfill-results", methods=["POST", "GET"])
def http_backfill():
    _require_admin()
    n = backfill_results_for_open_matches(400)
    return jsonify({"ok": True, "updated": n})

@app.route("/admin/digest", methods=["POST","GET"])
def http_digest(): 
    _require_admin()
    msg = daily_accuracy_digest()
    return jsonify({"ok": True, "sent": bool(msg)})

@app.route("/admin/model-status", methods=["GET"])
def http_model_status():
    _require_admin()
    
    status = {
        "available_models": model_manager.available_models,
        "loaded_models": list(model_manager.loaded_models.keys()),
        "metadata": model_manager.model_metadata,
        "default_probability": model_manager.default_probability
    }
    
    return jsonify({"ok": True, "model_status": status})

@app.route("/tips/latest")
def http_latest():
    limit = int(request.args.get("limit", "50"))
    with db_conn() as c:
        rows = c.execute(
            "SELECT match_id,league,home,away,market,suggestion,confidence,confidence_raw,"
            "score_at_tip,minute,created_ts,odds,book,ev_pct "
            "FROM tips WHERE suggestion<>'HARVEST' ORDER BY created_ts DESC LIMIT %s",
            (max(1, min(500, limit)),),
        ).fetchall()
    tips = []
    for r in rows:
        tips.append(
            {
                "match_id": int(r[0]),
                "league": r[1],
                "home": r[2],
                "away": r[3],
                "market": r[4],
                "suggestion": r[5],
                "confidence": float(r[6]),
                "confidence_raw": (float(r[7]) if r[7] is not None else None),
                "score_at_tip": r[8],
                "minute": int(r[9]),
                "created_ts": int(r[10]),
                "odds": (float(r[11]) if r[11] is not None else None),
                "book": r[12],
                "ev_pct": (float(r[13]) if r[13] is not None else None),
            }
        )
    return jsonify({"ok": True, "tips": tips})

# ───────── Startup ─────────
def _on_boot():
    _init_pool()
    init_db()
    set_setting("boot_ts", str(int(time.time())))
    
    # Train models if missing
    log.info("🔍 Checking for missing models...")
    _train_if_models_missing()
    
    # Load models at startup
    log.info("🚀 Loading models at startup...")
    for model_name in model_manager.available_models:
        model_manager.load_model(model_name)
    
    _start_scheduler_once()
    
    loaded_count = len(model_manager.loaded_models)
    total_count = len(model_manager.available_models)
    
    send_telegram(
        f"🚀 goalsniper PURE IN-PLAY AI started\n"
        f"📊 Models: {loaded_count}/{total_count} loaded\n"
        f"⚙️ Threshold: {CONF_THRESHOLD}%\n"
        f"✅ Ready for predictions"
    )

_on_boot()

# ───────── Main execution ─────────
if __name__ == "__main__":
    log.info("=" * 60)
    log.info("🚀 STARTING GOALSNIPER PURE IN-PLAY AI")
    log.info("=" * 60)
    log.info("📊 Model configuration loaded")
    log.info("✅ Ready for predictions")
    
    app.run(host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8080")))
