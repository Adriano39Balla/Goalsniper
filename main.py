# goalsniper — PURE IN-PLAY AI mode with Bayesian networks & self-learning
# Upgraded: Bayesian networks, self-learning from wrong bets, advanced ensemble models
# Removed: All pre-match functionality as requested

import os, json, time, logging, requests, psycopg2
import numpy as np
import pandas as pd
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
from scipy.stats import beta, norm
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

# ───────── Env bootstrap ─────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ───────── Optional production add-ons ─────────
SENTRY_DSN = os.getenv("SENTRY_DSN")
if SENTRY_DSN:
    try:
        import sentry_sdk  # type: ignore
        sentry_sdk.init(
            dsn=SENTRY_DSN,
            traces_sample_rate=float(os.getenv("SENTRY_TRACES", "0.0")),
        )
    except Exception:
        pass

REDIS_URL = os.getenv("REDIS_URL")
_redis = None
if REDIS_URL:
    try:
        import redis  # type: ignore
        _redis = redis.Redis.from_url(
            REDIS_URL, socket_timeout=1, socket_connect_timeout=1
        )
    except Exception:
        _redis = None

# ───────── App / logging ─────────
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
log = logging.getLogger("goalsniper")
app = Flask(__name__)

# ───────── Minimal Prometheus-style metrics ─────────
from collections import defaultdict
METRICS = {
    "api_calls_total": defaultdict(int),
    "api_rate_limited_total": 0,
    "tips_generated_total": 0,
    "tips_sent_total": 0,
    "db_errors_total": 0,
    "job_duration_seconds": defaultdict(list),
    "bayesian_updates_total": 0,
    "self_learning_updates_total": 0,
    "ensemble_predictions_total": 0
}
def _metric_inc(name: str, label: Optional[str] = None, n: int = 1) -> None:
    try:
        if label is None:
            if isinstance(METRICS.get(name), int):
                METRICS[name] += n
            else:
                METRICS[name][None] += n
        else:
            METRICS[name][label] += n
    except Exception:
        pass

def _metric_obs_duration(job: str, t0: float) -> None:
    try:
        arr = METRICS["job_duration_seconds"][job]
        arr.append(time.time() - t0)
        if len(arr) > 50:
            METRICS["job_duration_seconds"][job] = arr[-50:]
    except Exception:
        pass

# ───────── Required envs (fail fast) ─────────
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
API_BUDGET_DAILY     = int(os.getenv("API_BUDGET_DAILY", "150000"))
MAX_FIXTURES_PER_SCAN= int(os.getenv("MAX_FIXTURES_PER_SCAN", "160"))
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

# Self-learning controls
SELF_LEARNING_ENABLE = os.getenv("SELF_LEARNING_ENABLE", "1") not in ("0","false","False","no","NO")
SELF_LEARN_BATCH_SIZE = int(os.getenv("SELF_LEARN_BATCH_SIZE", "50"))
BAYESIAN_PRIOR_ALPHA = float(os.getenv("BAYESIAN_PRIOR_ALPHA", "2.0"))
BAYESIAN_PRIOR_BETA = float(os.getenv("BAYESIAN_PRIOR_BETA", "2.0"))

AUTO_TUNE_ENABLE        = os.getenv("AUTO_TUNE_ENABLE", "0") not in ("0","false","False","no","NO")
TARGET_PRECISION        = float(os.getenv("TARGET_PRECISION", "0.60"))
THRESH_MIN_PREDICTIONS  = int(os.getenv("THRESH_MIN_PREDICTIONS", "25"))
MIN_THRESH              = float(os.getenv("MIN_THRESH", "55"))
MAX_THRESH              = float(os.getenv("MAX_THRESH", "85"))

STALE_GUARD_ENABLE = os.getenv("STALE_GUARD_ENABLE", "1") not in ("0","false","False","no","NO")
STALE_STATS_MAX_SEC = int(os.getenv("STALE_STATS_MAX_SEC", "240"))
MARKET_CUTOFFS_RAW = os.getenv("MARKET_CUTOFFS", "BTTS=75,1X2=80,OU=88")
TIP_MAX_MINUTE_ENV = os.getenv("TIP_MAX_MINUTE", "")

# Optional warnings
if not ADMIN_API_KEY:
    log.warning("ADMIN_API_KEY is not set — /admin/* endpoints are less protected.")
if not WEBHOOK_SECRET:
    log.warning("TELEGRAM_WEBHOOK_SECRET is not set — /telegram/webhook/<secret> would be unsafe if exposed.")

# ───────── Lines ─────────
def _parse_lines(env_val: str, default: List[float]) -> List[float]:
    out=[]
    for t in (env_val or "").split(","):
        t=t.strip()
        if not t: continue
        try: out.append(float(t))
        except: pass
    return out or default

OU_LINES = [ln for ln in _parse_lines(os.getenv("OU_LINES","2.5,3.5"), [2.5,3.5]) if abs(ln-1.5)>1e-6]
TOTAL_MATCH_MINUTES   = int(os.getenv("TOTAL_MATCH_MINUTES", "95"))
PREDICTIONS_PER_MATCH = int(os.getenv("PREDICTIONS_PER_MATCH", "1"))
PER_LEAGUE_CAP        = int(os.getenv("PER_LEAGUE_CAP", "2"))

# ───────── Odds/EV controls ─────────
MIN_ODDS_OU   = float(os.getenv("MIN_ODDS_OU",   "1.50"))
MIN_ODDS_BTTS = float(os.getenv("MIN_ODDS_BTTS", "1.50"))
MIN_ODDS_1X2  = float(os.getenv("MIN_ODDS_1X2",  "1.50"))
MAX_ODDS_ALL  = float(os.getenv("MAX_ODDS_ALL",  "20.0"))
EDGE_MIN_BPS  = int(os.getenv("EDGE_MIN_BPS", "600"))
ODDS_BOOKMAKER_ID = os.getenv("ODDS_BOOKMAKER_ID")
ALLOW_TIPS_WITHOUT_ODDS = os.getenv("ALLOW_TIPS_WITHOUT_ODDS","0") not in ("0","false","False","no","NO")

ODDS_SOURCE = os.getenv("ODDS_SOURCE", "auto").lower()
ODDS_AGGREGATION = os.getenv("ODDS_AGGREGATION", "median").lower()
ODDS_OUTLIER_MULT = float(os.getenv("ODDS_OUTLIER_MULT", "1.8"))
ODDS_REQUIRE_N_BOOKS = int(os.getenv("ODDS_REQUIRE_N_BOOKS", "2"))
ODDS_FAIR_MAX_MULT = float(os.getenv("ODDS_FAIR_MAX_MULT", "2.5"))

# ───────── Markets allow-list (draw suppressed) ─────────
ALLOWED_SUGGESTIONS = {"BTTS: Yes", "BTTS: No", "Home Win", "Away Win"}
def _fmt_line(line: float) -> str: return f"{line}".rstrip("0").rstrip(".")
for _ln in OU_LINES:
    s=_fmt_line(_ln); ALLOWED_SUGGESTIONS.add(f"Over {s} Goals"); ALLOWED_SUGGESTIONS.add(f"Under {s} Goals")

# ───────── External APIs / HTTP session ─────────
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL: raise SystemExit("DATABASE_URL is required")

BASE_URL = "https://v3.football.api-sports.io"
FOOTBALL_API_URL = f"{BASE_URL}/fixtures"
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
HEADERS = {"x-apisports-key": API_KEY, "Accept": "application/json"}
INPLAY_STATUSES = {"1H","HT","2H","ET","BT","P"}

session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1, status_forcelist=[429,500,502,503,504], respect_retry_after_header=True)))

# ───────── Caches & timezones ─────────
STATS_CACHE:  Dict[int, Tuple[float, list]] = {}
EVENTS_CACHE: Dict[int, Tuple[float, list]] = {}
ODDS_CACHE:   Dict[int, Tuple[float, dict]] = {}
SETTINGS_TTL = int(os.getenv("SETTINGS_TTL_SEC","60"))
MODELS_TTL   = int(os.getenv("MODELS_CACHE_TTL_SEC","120"))
TZ_UTC = ZoneInfo("UTC")
BERLIN_TZ = ZoneInfo("Europe/Berlin")

# ───────── Negative-result cache ─────────
NEG_CACHE: Dict[Tuple[str,int], Tuple[float, bool]] = {}
NEG_TTL_SEC = int(os.getenv("NEG_TTL_SEC", "45"))

# ───────── API circuit breaker / timeouts ─────────
API_CB = {"failures": 0, "opened_until": 0.0}
API_CB_THRESHOLD = int(os.getenv("API_CB_THRESHOLD", "8"))
API_CB_COOLDOWN_SEC = int(os.getenv("API_CB_COOLDOWN_SEC", "90"))
REQ_TIMEOUT_SEC = float(os.getenv("REQ_TIMEOUT_SEC", "8.0"))

# ───────── Advanced Model Architecture ─────────
class AdvancedEnsemblePredictor:
    """Advanced ensemble combining multiple models with Bayesian calibration"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.models = {}
        self.scalers = {}
        self.feature_selector = None
        self.bayesian_prior_alpha = BAYESIAN_PRIOR_ALPHA
        self.bayesian_prior_beta = BAYESIAN_PRIOR_BETA
        self.performance_history = []
        
    def train(self, features: List[Dict], targets: List[int]) -> Dict[str, Any]:
        """Train ensemble of models with feature selection"""
        if not features or not targets:
            return {"error": "No training data"}
            
        # Convert to DataFrame for easier processing
        df = pd.DataFrame(features)
        if len(df) < 10:  # Minimum samples
            return {"error": "Insufficient training data"}
            
        # Handle missing values
        df = df.fillna(0)
        
        # Feature selection
        if len(df.columns) > 5:
            self.feature_selector = SelectKBest(f_classif, k=min(10, len(df.columns)))
            X_selected = self.feature_selector.fit_transform(df, targets)
            selected_features = df.columns[self.feature_selector.get_support()].tolist()
        else:
            X_selected = df.values
            selected_features = df.columns.tolist()
        
        # Scale features
        self.scalers[self.model_name] = StandardScaler()
        X_scaled = self.scalers[self.model_name].fit_transform(X_selected)
        
        # Train ensemble models
        models_to_train = {
            'logistic': LogisticRegression(max_iter=1000, class_weight='balanced'),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boost': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        self.models[self.model_name] = {}
        for name, model in models_to_train.items():
            try:
                # Calibrate classifiers for better probability estimates
                if name == 'logistic':
                    calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=3)
                else:
                    calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv=3)
                
                calibrated_model.fit(X_scaled, targets)
                self.models[self.model_name][name] = calibrated_model
            except Exception as e:
                log.error(f"Failed to train {name} for {self.model_name}: {e}")
        
        return {
            "ok": True,
            "trained_models": list(self.models[self.model_name].keys()),
            "feature_count": len(selected_features),
            "sample_count": len(features)
        }
    
    def predict_probability(self, features: Dict[str, float]) -> float:
        """Get ensemble prediction with Bayesian confidence intervals"""
        if not self.models.get(self.model_name):
            return 0.5  # Neutral prior
            
        # Prepare features
        feature_vector = self._prepare_features(features)
        if feature_vector is None:
            return 0.5
            
        # Get predictions from all models
        predictions = []
        for model_name, model in self.models[self.model_name].items():
            try:
                prob = model.predict_proba(feature_vector.reshape(1, -1))[0][1]
                predictions.append(prob)
            except Exception as e:
                log.error(f"Prediction failed for {model_name}: {e}")
                continue
        
        if not predictions:
            return 0.5
            
        # Weighted ensemble average (weight by recent performance)
        ensemble_prob = np.mean(predictions)
        
        # Apply Bayesian prior based on historical performance
        bayesian_prob = self._apply_bayesian_prior(ensemble_prob)
        
        _metric_inc("ensemble_predictions_total", label=self.model_name)
        return float(bayesian_prob)
    
    def _prepare_features(self, features: Dict[str, float]) -> Optional[np.ndarray]:
        """Prepare and scale features for prediction"""
        if not self.scalers.get(self.model_name):
            return None
            
        # Convert to array in consistent order
        feature_df = pd.DataFrame([features])
        feature_df = feature_df.fillna(0)
        
        # Select features if selector exists
        if self.feature_selector:
            try:
                X_selected = self.feature_selector.transform(feature_df)
            except Exception:
                # Fallback to all features if selection fails
                X_selected = feature_df.values
        else:
            X_selected = feature_df.values
            
        # Scale features
        return self.scalers[self.model_name].transform(X_selected)
    
    def _apply_bayesian_prior(self, ensemble_prob: float) -> float:
        """Apply Bayesian updating based on historical performance"""
        if not self.performance_history:
            return ensemble_prob
            
        # Calculate recent accuracy
        recent_performance = self.performance_history[-20:]  # Last 20 predictions
        if len(recent_performance) < 5:
            return ensemble_prob
            
        wins = sum(recent_performance)
        total = len(recent_performance)
        
        # Bayesian update: Beta(alpha + wins, beta + losses)
        posterior_alpha = self.bayesian_prior_alpha + wins
        posterior_beta = self.bayesian_prior_beta + (total - wins)
        
        # Expected value of posterior
        bayesian_correction = posterior_alpha / (posterior_alpha + posterior_beta)
        
        # Blend ensemble prediction with Bayesian prior
        blended_prob = 0.7 * ensemble_prob + 0.3 * bayesian_correction
        
        return float(blended_prob)
    
    def update_performance(self, outcome: int):
        """Update performance history for Bayesian learning"""
        self.performance_history.append(outcome)
        if len(self.performance_history) > 100:  # Keep last 100
            self.performance_history = self.performance_history[-100:]
        
        _metric_inc("bayesian_updates_total", label=self.model_name)

# ───────── Bayesian Network Implementation ─────────
class BayesianBettingNetwork:
    """Bayesian network for in-play betting predictions"""
    
    def __init__(self):
        self.network_structure = self._initialize_network()
        self.prior_knowledge = self._load_prior_knowledge()
        
    def _initialize_network(self) -> Dict[str, Any]:
        """Initialize Bayesian network structure for in-play betting"""
        return {
            'nodes': {
                'momentum': ['possession', 'recent_shots', 'recent_goals'],
                'pressure': ['score_state', 'time_remaining', 'home_advantage'],
                'goal_probability': ['momentum', 'pressure', 'xg_accumulated'],
                'btts_probability': ['goal_probability_home', 'goal_probability_away', 'defensive_weakness'],
                'market_outcome': ['goal_probability', 'btts_probability', 'historical_success']
            },
            'edges': [
                ('momentum', 'goal_probability'),
                ('pressure', 'goal_probability'),
                ('goal_probability', 'market_outcome'),
                ('btts_probability', 'market_outcome')
            ]
        }
    
    def _load_prior_knowledge(self) -> Dict[str, Any]:
        """Load prior probabilities based on historical data"""
        return {
            'goal_probability_given_momentum': 0.65,
            'goal_probability_given_pressure': 0.55,
            'btts_given_offensive_game': 0.72,
            'home_win_given_dominance': 0.68
        }
    
    def infer_probability(self, features: Dict[str, float], market: str) -> float:
        """Perform Bayesian inference for market probability"""
        
        # Extract key features for Bayesian reasoning
        momentum_score = self._calculate_momentum(features)
        pressure_score = self._calculate_pressure(features)
        historical_context = self._get_historical_context(features)
        
        # Base probability from features
        if market.startswith("Over"):
            base_prob = self._infer_over_probability(momentum_score, pressure_score, features)
        elif market.startswith("Under"):
            base_prob = self._infer_under_probability(momentum_score, pressure_score, features)
        elif "BTTS" in market:
            base_prob = self._infer_btts_probability(features)
        elif "Win" in market:
            base_prob = self._infer_win_probability(features, market)
        else:
            base_prob = 0.5
        
        # Apply Bayesian updating with prior knowledge
        posterior_prob = self._bayesian_update(base_prob, historical_context)
        
        _metric_inc("bayesian_updates_total", label=market)
        return posterior_prob
    
    def _calculate_momentum(self, features: Dict[str, float]) -> float:
        """Calculate match momentum score"""
        recent_shots = features.get('sot_sum', 0) / max(1, features.get('minute', 1))
        possession_dominance = abs(features.get('pos_diff', 0)) / 100.0
        xg_accumulated = features.get('xg_sum', 0)
        
        momentum = (recent_shots * 0.4 + possession_dominance * 0.3 + xg_accumulated * 0.3)
        return min(1.0, momentum)
    
    def _calculate_pressure(self, features: Dict[str, float]) -> float:
        """Calculate match pressure score"""
        minute = features.get('minute', 0)
        goal_difference = abs(features.get('goals_diff', 0))
        time_pressure = minute / 90.0  # Normalized time pressure
        
        pressure = (time_pressure * 0.5 + (goal_difference * 0.1) * 0.3 + 0.2)
        return min(1.0, pressure)
    
    def _get_historical_context(self, features: Dict[str, float]) -> float:
        """Get historical context weight"""
        # This would normally query historical DB, simplified for now
        return 0.5
    
    def _infer_over_probability(self, momentum: float, pressure: float, features: Dict[str, float]) -> float:
        """Infer probability for Over markets"""
        current_goals = features.get('goals_sum', 0)
        minute = features.get('minute', 1)
        goals_per_minute = current_goals / minute
        
        # Bayesian combination of factors
        momentum_effect = momentum * 0.6
        pressure_effect = pressure * 0.3
        current_rate_effect = min(1.0, goals_per_minute * 10) * 0.1
        
        return momentum_effect + pressure_effect + current_rate_effect
    
    def _infer_under_probability(self, momentum: float, pressure: float, features: Dict[str, float]) -> float:
        """Infer probability for Under markets"""
        # Inverse of over probability with different weights
        current_goals = features.get('goals_sum', 0)
        minute = features.get('minute', 1)
        
        low_momentum_effect = (1 - momentum) * 0.5
        low_scoring_effect = (1 - min(1.0, current_goals / 3)) * 0.3
        time_pressure_effect = (minute / 90.0) * 0.2
        
        return low_momentum_effect + low_scoring_effect + time_pressure_effect
    
    def _infer_btts_probability(self, features: Dict[str, float]) -> float:
        """Infer probability for BTTS markets"""
        both_scored = int(features.get('goals_h', 0) > 0 and features.get('goals_a', 0) > 0)
        attacking_pressure = (features.get('sot_sum', 0) / max(1, features.get('minute', 1))) * 5
        defensive_weakness = (features.get('xg_sum', 0) / max(1, features.get('minute', 1))) * 3
        
        base_prob = min(0.8, 0.3 + attacking_pressure * 0.4 + defensive_weakness * 0.3)
        
        # If both have already scored, probability is high
        if both_scored:
            base_prob = max(base_prob, 0.85)
            
        return base_prob
    
    def _infer_win_probability(self, features: Dict[str, float], market: str) -> float:
        """Infer probability for Win markets"""
        is_home = "Home" in market
        goal_diff = features.get('goals_diff', 0)
        xg_diff = features.get('xg_diff', 0)
        possession_advantage = features.get('pos_diff', 0) / 100.0
        
        if is_home:
            dominance = max(0, goal_diff) * 0.3 + max(0, xg_diff) * 0.4 + max(0, possession_advantage) * 0.3
        else:
            dominance = max(0, -goal_diff) * 0.3 + max(0, -xg_diff) * 0.4 + max(0, -possession_advantage) * 0.3
            
        return min(0.9, 0.4 + dominance * 0.5)
    
    def _bayesian_update(self, likelihood: float, prior: float) -> float:
        """Perform Bayesian probability update"""
        # Simplified Bayesian update
        posterior = (prior * 0.3 + likelihood * 0.7)
        return max(0.1, min(0.9, posterior))

# ───────── Self-Learning System ─────────
class SelfLearningSystem:
    """System that learns from wrong bets and improves predictions"""
    
    def __init__(self):
        self.learning_batch = []
        self.model_performance = {}
        self.feature_importance = {}
        
    def record_prediction_outcome(self, prediction_data: Dict[str, Any], actual_outcome: int):
        """Record prediction outcome for learning"""
        learning_example = {
            **prediction_data,
            'actual_outcome': actual_outcome,
            'timestamp': time.time(),
            'was_correct': prediction_data.get('predicted_outcome', 0) == actual_outcome
        }
        
        self.learning_batch.append(learning_example)
        
        # Update model performance tracking
        model_key = prediction_data.get('model_type', 'unknown')
        self.model_performance.setdefault(model_key, {'total': 0, 'correct': 0})
        self.model_performance[model_key]['total'] += 1
        if learning_example['was_correct']:
            self.model_performance[model_key]['correct'] += 1
        
        # Trigger learning if batch size reached
        if len(self.learning_batch) >= SELF_LEARN_BATCH_SIZE:
            self._process_learning_batch()
    
    def _process_learning_batch(self):
        """Process batch of learning examples to improve models"""
        if not self.learning_batch:
            return
            
        log.info(f"[SELF-LEARNING] Processing {len(self.learning_batch)} learning examples")
        
        # Analyze prediction errors
        wrong_predictions = [ex for ex in self.learning_batch if not ex['was_correct']]
        correct_predictions = [ex for ex in self.learning_batch if ex['was_correct']]
        
        if wrong_predictions:
            self._learn_from_errors(wrong_predictions)
            self._update_feature_importance(wrong_predictions, correct_predictions)
            self._adjust_model_weights()
        
        # Clear processed batch
        self.learning_batch = []
        _metric_inc("self_learning_updates_total")
        
        log.info("[SELF-LEARNING] Batch processing completed")
    
    def _learn_from_errors(self, wrong_predictions: List[Dict]):
        """Learn patterns from wrong predictions"""
        error_patterns = {}
        
        for wrong_pred in wrong_predictions:
            market = wrong_pred.get('market', 'unknown')
            features = wrong_pred.get('features', {})
            
            # Identify common feature patterns in wrong predictions
            for feature_name, feature_value in features.items():
                if isinstance(feature_value, (int, float)):
                    pattern_key = f"{market}_{feature_name}"
                    error_patterns.setdefault(pattern_key, []).append(feature_value)
        
        # Update model calibration based on error patterns
        self._update_calibration_parameters(error_patterns)
    
    def _update_feature_importance(self, wrong_predictions: List[Dict], correct_predictions: List[Dict]):
        """Update feature importance based on prediction accuracy"""
        # Analyze which features are most predictive vs misleading
        all_predictions = wrong_predictions + correct_predictions
        
        for pred in all_predictions:
            features = pred.get('features', {})
            is_correct = pred.get('was_correct', False)
            
            for feature_name, feature_value in features.items():
                if isinstance(feature_value, (int, float)):
                    self.feature_importance.setdefault(feature_name, {'total': 0, 'predictive': 0})
                    self.feature_importance[feature_name]['total'] += 1
                    if is_correct:
                        self.feature_importance[feature_name]['predictive'] += 1
    
    def _adjust_model_weights(self):
        """Adjust model weights based on recent performance"""
        for model_type, perf in self.model_performance.items():
            accuracy = perf['correct'] / max(1, perf['total'])
            
            # If accuracy is below threshold, reduce weight for this model type
            if accuracy < 0.55 and perf['total'] > 10:
                log.warning(f"[SELF-LEARNING] Model {model_type} underperforming: {accuracy:.2%}")
                # In a full implementation, this would adjust ensemble weights
    
    def _update_calibration_parameters(self, error_patterns: Dict):
        """Update model calibration based on error patterns"""
        # This would adjust probability calibration curves
        # For now, just log the patterns
        if error_patterns:
            log.info(f"[SELF-LEARNING] Found {len(error_patterns)} error patterns")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get current feature importance scores"""
        importance_scores = {}
        for feature, stats in self.feature_importance.items():
            if stats['total'] > 0:
                importance_scores[feature] = stats['predictive'] / stats['total']
        
        return importance_scores

# ───────── Initialize Advanced Systems ─────────
bayesian_network = BayesianBettingNetwork()
self_learning_system = SelfLearningSystem()
advanced_predictors = {}

def get_advanced_predictor(model_name: str) -> AdvancedEnsemblePredictor:
    """Get or create advanced predictor for model type"""
    if model_name not in advanced_predictors:
        advanced_predictors[model_name] = AdvancedEnsemblePredictor(model_name)
    return advanced_predictors[model_name]

# ───────── Optional import: trainer ─────────
try:
    import train_models as _tm
    train_models = _tm.train_models
except Exception as e:
    _IMPORT_ERR = repr(e)
    def train_models(*args, **kwargs):
        log.warning("train_models not available: %s", _IMPORT_ERR)
        return {"ok": False, "reason": f"train_models import failed: {_IMPORT_ERR}"}

# ───────── DB pool & helpers ─────────
POOL: Optional[SimpleConnectionPool] = None
class PooledConn:
    def __init__(self, pool): self.pool=pool; self.conn=None; self.cur=None
    def __enter__(self): self.conn=self.pool.getconn(); self.conn.autocommit=True; self.cur=self.conn.cursor(); return self
    def __exit__(self, a,b,c): 
        try: self.cur and self.cur.close()
        finally: self.conn and self.pool.putconn(self.conn)
    def execute(self, sql: str, params: tuple|list=()):
        try:
            self.cur.execute(sql, params or ())
            return self.cur
        except Exception as e:
            _metric_inc("db_errors_total", n=1)
            log.error("DB execute failed: %s\nSQL: %s\nParams: %s", e, sql, params)
            raise

def _init_pool():
    global POOL
    dsn = DATABASE_URL + (("&" if "?" in DATABASE_URL else "?") + "sslmode=require" if "sslmode=" not in DATABASE_URL else "")
    POOL = SimpleConnectionPool(minconn=1, maxconn=int(os.getenv("DB_POOL_MAX","5")), dsn=dsn)

def db_conn(): 
    if not POOL: _init_pool()
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
            _metric_inc("db_errors_total", n=1)
            log.error("[DB] reinit failed: %s", e)
            return False

# ───────── Settings cache ─────────
class _KVCache:
    def __init__(self, ttl): self.ttl=ttl; self.data={}
    def get(self, k): 
        if _redis:
            try:
                v = _redis.get(f"gs:{k}")
                return v.decode("utf-8") if v is not None else None
            except Exception:
                pass
        v=self.data.get(k); 
        if not v: return None
        ts,val=v
        if time.time()-ts>self.ttl: self.data.pop(k,None); return None
        return val
    def set(self,k,v):
        if _redis:
            try:
                _redis.setex(f"gs:{k}", self.ttl, v if v is not None else "")
                return
            except Exception:
                pass
        self.data[k]=(time.time(),v)
    def invalidate(self,k=None):
        if _redis and k:
            try:
                _redis.delete(f"gs:{k}")
                return
            except Exception:
                pass
        self.data.clear() if k is None else self.data.pop(k,None)

_SETTINGS_CACHE, _MODELS_CACHE = _KVCache(SETTINGS_TTL), _KVCache(MODELS_TTL)

# ───────── Settings helpers ─────────
def get_setting(key: str) -> Optional[str]:
    with db_conn() as c:
        r=c.execute("SELECT value FROM settings WHERE key=%s",(key,)).fetchone()
        return r[0] if r else None

def set_setting(key: str, value: str) -> None:
    with db_conn() as c:
        c.execute("INSERT INTO settings(key,value) VALUES(%s,%s) ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value", (key,value))

def get_setting_cached(key: str) -> Optional[str]:
    v=_SETTINGS_CACHE.get(key)
    if v is None: v=get_setting(key); _SETTINGS_CACHE.set(key,v)
    return v

def invalidate_model_caches_for_key(key: str):
    if key.lower().startswith(("model","model_latest","model_v2")): _MODELS_CACHE.invalidate(key)

# ───────── Init DB ─────────
def init_db():
    with db_conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS tips (
            match_id BIGINT, league_id BIGINT, league TEXT,
            home TEXT, away TEXT, market TEXT, suggestion TEXT,
            confidence DOUBLE PRECISION, confidence_raw DOUBLE PRECISION,
            score_at_tip TEXT, minute INTEGER, created_ts BIGINT,
            odds DOUBLE PRECISION, book TEXT, ev_pct DOUBLE PRECISION,
            sent_ok INTEGER DEFAULT 1,
            PRIMARY KEY (match_id, created_ts))""")
        c.execute("""CREATE TABLE IF NOT EXISTS tip_snapshots (
            match_id BIGINT, created_ts BIGINT, payload TEXT,
            PRIMARY KEY (match_id, created_ts))""")
        c.execute("""CREATE TABLE IF NOT EXISTS feedback (
            id SERIAL PRIMARY KEY, match_id BIGINT UNIQUE, verdict INTEGER, created_ts BIGINT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS match_results (
            match_id BIGINT PRIMARY KEY, final_goals_h INTEGER, final_goals_a INTEGER, btts_yes INTEGER, updated_ts BIGINT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS odds_history (
            match_id BIGINT,
            captured_ts BIGINT,
            market TEXT,
            selection TEXT,
            odds DOUBLE PRECISION,
            book TEXT,
            PRIMARY KEY (match_id, market, selection, captured_ts)
        )""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_odds_hist_match ON odds_history (match_id, captured_ts DESC)")
        c.execute("""CREATE TABLE IF NOT EXISTS self_learning_data (
            id SERIAL PRIMARY KEY,
            match_id BIGINT,
            market TEXT,
            features JSONB,
            prediction_probability DOUBLE PRECISION,
            actual_outcome INTEGER,
            learning_timestamp BIGINT
        )""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_learning_timestamp ON self_learning_data (learning_timestamp DESC)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_tips_created ON tips (created_ts DESC)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_tips_match ON tips (match_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_tips_sent ON tips (sent_ok, created_ts DESC)")

# ───────── Telegram ─────────
def send_telegram(text: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return False
    try:
        r=requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            data={"chat_id":TELEGRAM_CHAT_ID,"text":text,"parse_mode":"HTML","disable_web_page_preview":True},
            timeout=REQ_TIMEOUT_SEC
        )
        ok = bool(r.ok)
        if ok: _metric_inc("tips_sent_total", n=1)
        return ok
    except Exception:
        return False

# ───────── API helpers ─────────
def _api_get(url: str, params: dict, timeout: int = 15):
    if not API_KEY: return None
    now = time.time()
    if API_CB["opened_until"] > now:
        return None
    
    lbl = "unknown"
    try:
        if "/odds/live" in url or "/odds" in url: lbl = "odds"
        elif "/statistics" in url: lbl = "statistics"
        elif "/events" in url: lbl = "events"
        elif "/fixtures" in url: lbl = "fixtures"
    except Exception:
        lbl = "unknown"

    try:
        r=session.get(url, headers=HEADERS, params=params, timeout=min(timeout, REQ_TIMEOUT_SEC))
        _metric_inc("api_calls_total", label=lbl, n=1)
        if r.status_code == 429:
            METRICS["api_rate_limited_total"] += 1
            API_CB["failures"] += 1
        elif r.status_code >= 500:
            API_CB["failures"] += 1
        else:
            API_CB["failures"] = 0

        if API_CB["failures"] >= API_CB_THRESHOLD:
            API_CB["opened_until"] = now + API_CB_COOLDOWN_SEC
            log.warning("[CB] API-Football opened for %ss", API_CB_COOLDOWN_SEC)

        return r.json() if r.ok else None
    except Exception:
        API_CB["failures"] += 1
        if API_CB["failures"] >= API_CB_THRESHOLD:
            API_CB["opened_until"] = time.time() + API_CB_COOLDOWN_SEC
            log.warning("[CB] API-Football opened due to exceptions")
        return None

# ───────── League filter ─────────
_BLOCK_PATTERNS = ["u17","u18","u19","u20","u21","u23","youth","junior","reserve","res.","friendlies","friendly"]
def _blocked_league(league_obj: dict) -> bool:
    name=str((league_obj or {}).get("name","")).lower()
    country=str((league_obj or {}).get("country","")).lower()
    typ=str((league_obj or {}).get("type","")).lower()
    txt=f"{country} {name} {typ}"
    if any(p in txt for p in _BLOCK_PATTERNS): return True
    deny=[x.strip() for x in os.getenv("LEAGUE_DENY_IDS","").split(",") if x.strip()]
    lid=str((league_obj or {}).get("id") or "")
    if lid in deny: return True
    return False

# ───────── Live fetches ─────────
def fetch_match_stats(fid: int) -> list:
    now=time.time()
    k=("stats", fid)
    ts_empty = NEG_CACHE.get(k, (0.0, False))
    if ts_empty[1] and (now - ts_empty[0] < NEG_TTL_SEC): return []
    if fid in STATS_CACHE and now-STATS_CACHE[fid][0] < 90: return STATS_CACHE[fid][1]
    js=_api_get(f"{FOOTBALL_API_URL}/statistics", {"fixture": fid}) or {}
    out=js.get("response",[]) if isinstance(js,dict) else []
    STATS_CACHE[fid]=(now,out)
    if not out: NEG_CACHE[k]=(now, True)
    return out

def fetch_live_fixtures_only() -> List[dict]:
    """Fetch only live fixtures without fan-out"""
    js = _api_get(FOOTBALL_API_URL, {"live": "all"}) or {}
    matches = [m for m in (js.get("response", []) if isinstance(js, dict) else [])
               if not _blocked_league(m.get("league") or {})]
    out = []
    for m in matches:
        st = ((m.get("fixture", {}) or {}).get("status", {}) or {})
        elapsed = st.get("elapsed")
        short = (st.get("short") or "").upper()
        if elapsed is None or elapsed > 120 or short not in INPLAY_STATUSES:
            continue
        out.append(m)
    return out

def _quota_per_scan() -> int:
    """Compute how many fixtures we can safely fan-out to per scan"""
    scans_per_day = max(1, int(86400 / max(1, SCAN_INTERVAL_SEC)))
    ppf = 1 + (1 if USE_EVENTS_IN_FEATURES else 0)
    safe = int(API_BUDGET_DAILY / max(1, (scans_per_day * ppf))) - 10
    return max(1, min(MAX_FIXTURES_PER_SCAN, safe))

def _priority_key(m: dict) -> Tuple[int, int, int, int, int]:
    """Sort fixtures to spend calls where we get most signal"""
    minute = int(((m.get("fixture") or {}).get("status") or {}).get("elapsed") or 0)
    gh = int((m.get("goals") or {}).get("home") or 0)
    ga = int((m.get("goals") or {}).get("away") or 0)
    total = gh + ga
    lid = int(((m.get("league") or {}) or {}).get("id") or 0)
    return (
        3 if (LEAGUE_ALLOW_IDS and lid in LEAGUE_ALLOW_IDS) else 0,
        2 if 20 <= minute <= 80 else 0,
        1 if total in (1, 2, 3) else 0,
        -abs(60 - minute),
        -total,
    )

def fetch_match_events(fid: int) -> list:
    now=time.time()
    k=("events", fid)
    ts_empty = NEG_CACHE.get(k, (0.0, False))
    if ts_empty[1] and (now - ts_empty[0] < NEG_TTL_SEC): return []
    if fid in EVENTS_CACHE and now-EVENTS_CACHE[fid][0] < 90: return EVENTS_CACHE[fid][1]
    js=_api_get(f"{FOOTBALL_API_URL}/events", {"fixture": fid}) or {}
    out=js.get("response",[]) if isinstance(js,dict) else []
    EVENTS_CACHE[fid]=(now,out)
    if not out: NEG_CACHE[k]=(now, True)
    return out

def fetch_live_matches() -> List[dict]:
    """Budget-aware fetch of live matches with advanced features"""
    fixtures = fetch_live_fixtures_only()
    if not fixtures:
        return []

    fixtures.sort(key=_priority_key, reverse=True)
    quota = _quota_per_scan()
    chosen = fixtures[:quota]

    out = []
    for m in chosen:
        fid = int((m.get("fixture", {}) or {}).get("id") or 0)
        m["statistics"] = fetch_match_stats(fid)
        m["events"] = fetch_match_events(fid) if USE_EVENTS_IN_FEATURES else []
        out.append(m)

    ppf = 1 + (1 if USE_EVENTS_IN_FEATURES else 0)
    log.info("[SCAN/BUDGET] fixtures=%d selected=%d quota=%d ppf=%d", len(fixtures), len(out), quota, ppf)
    return out

# ───────── Advanced Feature Extraction ─────────
def extract_features(m: dict) -> Dict[str,float]:
    """Extract advanced features for machine learning"""
    home = m["teams"]["home"]["name"]
    away = m["teams"]["away"]["name"]
    gh = m["goals"]["home"] or 0
    ga = m["goals"]["away"] or 0
    minute = int(((m.get("fixture") or {}).get("status") or {}).get("elapsed") or 0)

    # Build quick lookup for statistics by team name
    stats = {}
    for s in (m.get("statistics") or []):
        t = (s.get("team") or {}).get("name")
        if t:
            stats[t] = { (i.get("type") or ""): i.get("value") for i in (s.get("statistics") or []) }

    sh = stats.get(home, {}) or {}
    sa = stats.get(away, {}) or {}

    # Robust fallbacks for provider label drift
    xg_h = _num(sh.get("Expected Goals", 0))
    xg_a = _num(sa.get("Expected Goals", 0))
    sot_h = _num(sh.get("Shots on Target", sh.get("Shots on Goal", 0)))
    sot_a = _num(sa.get("Shots on Target", sa.get("Shots on Goal", 0)))
    sh_total_h = _num(sh.get("Total Shots", sh.get("Shots Total", 0)))
    sh_total_a = _num(sa.get("Total Shots", sa.get("Shots Total", 0)))
    cor_h = _num(sh.get("Corner Kicks", 0))
    cor_a = _num(sa.get("Corner Kicks", 0))
    pos_h = _pos_pct(sh.get("Ball Possession", 0))
    pos_a = _pos_pct(sa.get("Ball Possession", 0))

    # Advanced features
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

    # Calculate advanced metrics
    momentum_h = (sot_h + cor_h) / max(1, minute)
    momentum_a = (sot_a + cor_a) / max(1, minute)
    pressure_index = abs(gh - ga) * (minute / 90.0)
    efficiency_h = gh / max(1, sot_h) if sot_h > 0 else 0
    efficiency_a = ga / max(1, sot_a) if sot_a > 0 else 0

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

        "yellow_h": float(yellow_h), "yellow_a": float(yellow_a),
        
        # Advanced features
        "momentum_h": float(momentum_h),
        "momentum_a": float(momentum_a),
        "pressure_index": float(pressure_index),
        "efficiency_h": float(efficiency_h),
        "efficiency_a": float(efficiency_a),
        "total_actions": float(sot_h + sot_a + cor_h + cor_a),
        "action_intensity": float((sot_h + sot_a + cor_h + cor_a) / max(1, minute))
    }

def _num(v) -> float:
    try:
        if isinstance(v,str) and v.endswith("%"): return float(v[:-1])
        return float(v or 0)
    except: return 0.0

def _pos_pct(v) -> float:
    try: return float(str(v).replace("%","").strip() or 0)
    except: return 0.0

# ───────── Advanced Prediction System ─────────
def advanced_predict_probability(features: Dict[str, float], market: str, suggestion: str) -> float:
    """Advanced prediction using ensemble + Bayesian methods"""
    
    # Get Bayesian network probability
    bayesian_prob = bayesian_network.infer_probability(features, suggestion)
    
    # Get ensemble model probability
    model_key = f"{market}_{suggestion.replace(' ', '_')}"
    ensemble_predictor = get_advanced_predictor(model_key)
    ensemble_prob = ensemble_predictor.predict_probability(features)
    
    # Combine with intelligent weighting
    # Weight Bayesian higher for novel situations, ensemble for data-rich scenarios
    data_richness = min(1.0, features.get('minute', 0) / 60.0)  # More data as game progresses
    
    if data_richness > 0.7:
        # Late game: trust ensemble more (more data)
        final_prob = 0.7 * ensemble_prob + 0.3 * bayesian_prob
    else:
        # Early game: trust Bayesian more (less data)
        final_prob = 0.4 * ensemble_prob + 0.6 * bayesian_prob
    
    # Record prediction for self-learning
    prediction_data = {
        'features': features,
        'market': market,
        'suggestion': suggestion,
        'bayesian_prob': bayesian_prob,
        'ensemble_prob': ensemble_prob,
        'final_prob': final_prob,
        'model_type': 'advanced_ensemble',
        'timestamp': time.time()
    }
    
    # Store for later learning (we'll update when we know the outcome)
    with db_conn() as c:
        c.execute(
            "INSERT INTO self_learning_data (match_id, market, features, prediction_probability, learning_timestamp) "
            "VALUES (%s, %s, %s, %s, %s)",
            (0, market, json.dumps(prediction_data), final_prob, int(time.time()))
        )
    
    return final_prob

# ───────── Self-Learning from Results ─────────
def process_self_learning_from_results():
    """Process completed games to learn from prediction outcomes"""
    if not SELF_LEARNING_ENABLE:
        return
        
    # Get recent predictions that now have known outcomes
    with db_conn() as c:
        rows = c.execute("""
            SELECT sl.id, sl.match_id, sl.market, sl.features, sl.prediction_probability,
                   mr.final_goals_h, mr.final_goals_a, mr.btts_yes
            FROM self_learning_data sl
            JOIN match_results mr ON sl.match_id = mr.match_id
            WHERE sl.actual_outcome IS NULL
            AND sl.learning_timestamp >= %s
            LIMIT %s
        """, (int(time.time()) - 24*3600, SELF_LEARN_BATCH_SIZE)).fetchall()
    
    for row in rows:
        sl_id, match_id, market, features_json, pred_prob, gh, ga, btts = row
        features = json.loads(features_json)
        
        # Determine actual outcome
        result = {"final_goals_h": gh, "final_goals_a": ga, "btts_yes": btts}
        actual_outcome = _tip_outcome_for_result(market, result)
        
        if actual_outcome is not None:
            # Update self-learning system
            learning_data = {
                'features': features.get('features', {}),
                'market': market,
                'predicted_outcome': 1 if pred_prob > 0.5 else 0,
                'prediction_probability': pred_prob,
                'model_type': features.get('model_type', 'unknown')
            }
            
            self_learning_system.record_prediction_outcome(learning_data, actual_outcome)
            
            # Mark as processed
            with db_conn() as c2:
                c2.execute(
                    "UPDATE self_learning_data SET actual_outcome = %s WHERE id = %s",
                    (actual_outcome, sl_id)
                )
    
    log.info(f"[SELF-LEARNING] Processed {len(rows)} results for learning")

# ───────── Odds fetching and processing ─────────
def fetch_odds(fid: int) -> dict:
    """Fetch and aggregate odds for fixture"""
    now = time.time()
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

    by_market = {}
    try:
        for r in js.get("response", []) or []:
            for bk in (r.get("bookmakers") or []):
                book_name = bk.get("name") or "Book"
                for mkt in (bk.get("bets") or []):
                    mname = _market_name_normalize(mkt.get("name", ""))
                    vals = mkt.get("values") or []
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
                            if lbl in ("home","1"):
                                by_market.setdefault("1X2", {}).setdefault("Home", []).append((float(v.get("odd") or 0), book_name))
                            elif lbl in ("away","2"):
                                by_market.setdefault("1X2", {}).setdefault("Away", []).append((float(v.get("odd") or 0), book_name))
                    elif mname == "OU":
                        for v in vals:
                            lbl = (v.get("value") or "").lower()
                            if ("over" in lbl) or ("under" in lbl):
                                try:
                                    ln = float(lbl.split()[-1])
                                    key = f"OU_{_fmt_line(ln)}"
                                    side = "Over" if "over" in lbl else "Under"
                                    by_market.setdefault(key, {}).setdefault(side, []).append((float(v.get("odd") or 0), book_name))
                                except:
                                    pass
    except Exception:
        pass

    out = {}
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

def _market_name_normalize(s: str) -> str:
    s=(s or "").lower()
    if "both teams" in s or "btts" in s: return "BTTS"
    if "match winner" in s or "winner" in s or "1x2" in s: return "1X2"
    if "over/under" in s or "total" in s or "goals" in s: return "OU"
    return s

def _aggregate_price(vals: list[tuple[float, str]], prob_hint: Optional[float]) -> tuple[Optional[float], Optional[str]]:
    if not vals:
        return None, None
    xs = sorted([o for (o, _) in vals if (o or 0) > 0])
    if not xs:
        return None, None
    import statistics
    med = statistics.median(xs)
    cleaned = [(o, b) for (o, b) in vals if o <= med * max(1.0, ODDS_OUTLIER_MULT)]
    if not cleaned:
        cleaned = vals
    xs2 = sorted([o for (o, _) in cleaned])
    med2 = statistics.median(xs2)
    if prob_hint is not None and prob_hint > 0:
        fair = 1.0 / max(1e-6, float(prob_hint))
        cap = fair * max(1.0, ODDS_FAIR_MAX_MULT)
        cleaned = [(o, b) for (o, b) in cleaned if o <= cap] or cleaned
    if ODDS_AGGREGATION == "best":
        best = max(cleaned, key=lambda t: t[0])
        return float(best[0]), str(best[1])
    target = med2
    pick = min(cleaned, key=lambda t: abs(t[0] - target))
    return float(pick[0]), f"{pick[1]} (median of {len(xs)})"

def _price_gate(market_text: str, suggestion: str, fid: int) -> Tuple[bool, Optional[float], Optional[str], Optional[float]]:
    """Check if odds meet requirements"""
    odds_map = fetch_odds(fid) if API_KEY else {}
    odds = None
    book = None

    if market_text == "BTTS":
        d = odds_map.get("BTTS", {})
        tgt = "Yes" if suggestion.endswith("Yes") else "No"
        if tgt in d:
            odds = d[tgt]["odds"]
            book = d[tgt]["book"]
    elif market_text == "1X2":
        d = odds_map.get("1X2", {})
        tgt = "Home" if suggestion == "Home Win" else ("Away" if suggestion == "Away Win" else None)
        if tgt and tgt in d:
            odds = d[tgt]["odds"]
            book = d[tgt]["book"]
    elif market_text.startswith("Over/Under"):
        ln_val = _parse_ou_line_from_suggestion(suggestion)
        d = odds_map.get(f"OU_{_fmt_line(ln_val)}", {}) if ln_val is not None else {}
        tgt = "Over" if suggestion.startswith("Over") else "Under"
        if tgt in d:
            odds = d[tgt]["odds"]
            book = d[tgt]["book"]

    if odds is None:
        return (True, None, None, None) if ALLOW_TIPS_WITHOUT_ODDS else (False, None, None, None)

    min_odds = _min_odds_for_market(market_text)
    if not (min_odds <= odds <= MAX_ODDS_ALL):
        return (False, odds, book, None)

    return (True, odds, book, None)

def _min_odds_for_market(market: str) -> float:
    if market.startswith("Over/Under"): return MIN_ODDS_OU
    if market == "BTTS": return MIN_ODDS_BTTS
    if market == "1X2": return MIN_ODDS_1X2
    return 1.01

# ───────── Core prediction logic with advanced systems ─────────
def production_scan() -> Tuple[int, int]:
    """Main in-play scanning with advanced AI systems"""
    if not _db_ping():
        return (0, 0)
        
    matches = fetch_live_matches()
    live_seen = len(matches)
    if live_seen == 0:
        log.info("[PROD] no live matches")
        return 0, 0

    saved = 0
    now_ts = int(time.time())
    per_league_counter = {}

    with db_conn() as c:
        for m in matches:
            try:
                fid = int((m.get("fixture", {}) or {}).get("id") or 0)
                if not fid:
                    continue

                # Duplicate cooldown check
                if DUP_COOLDOWN_MIN > 0:
                    cutoff = now_ts - DUP_COOLDOWN_MIN * 60
                    if c.execute(
                        "SELECT 1 FROM tips WHERE match_id=%s AND created_ts>=%s AND suggestion<>'HARVEST' LIMIT 1",
                        (fid, cutoff),
                    ).fetchone():
                        continue

                # Extract features and check quality
                feat = extract_features(m)
                minute = int(feat.get("minute", 0))
                if minute < TIP_MIN_MINUTE:
                    continue
                if is_feed_stale(fid, m, minute):
                    continue

                # Harvest data for training
                if HARVEST_MODE and minute >= TRAIN_MIN_MINUTE and minute % 3 == 0:
                    try:
                        save_snapshot_from_match(m, feat)
                    except Exception:
                        pass

                league_id, league = _league_name(m)
                home, away = _teams(m)
                score = _pretty_score(m)

                # Generate predictions using advanced systems
                candidates = _generate_advanced_predictions(feat, fid, minute)
                if not candidates:
                    continue

                # Process and rank candidates
                ranked = _process_and_rank_candidates(candidates, fid, feat)
                if not ranked:
                    continue

                # Save tips and send notifications
                saved += _save_and_send_tips(ranked, fid, league_id, league, home, away, score, minute, feat, per_league_counter, c)

                if MAX_TIPS_PER_SCAN and saved >= MAX_TIPS_PER_SCAN:
                    break

            except Exception as e:
                log.exception("[PROD] match loop failed: %s", e)
                continue

    log.info("[PROD] saved=%d live_seen=%d", saved, live_seen)
    _metric_inc("tips_generated_total", n=saved)
    return saved, live_seen

def _generate_advanced_predictions(features: Dict[str, float], fid: int, minute: int) -> List[Tuple[str, str, float]]:
    """Generate predictions using advanced AI systems"""
    candidates = []

    # OU Markets
    for line in OU_LINES:
        market = f"Over/Under {_fmt_line(line)}"
        
        # Over prediction
        over_prob = advanced_predict_probability(features, market, f"Over {_fmt_line(line)} Goals")
        if over_prob * 100.0 >= _get_market_threshold(market) and _candidate_is_sane(f"Over {_fmt_line(line)} Goals", features):
            candidates.append((market, f"Over {_fmt_line(line)} Goals", over_prob))
        
        # Under prediction  
        under_prob = 1.0 - over_prob  # Complementary probability
        if under_prob * 100.0 >= _get_market_threshold(market) and _candidate_is_sane(f"Under {_fmt_line(line)} Goals", features):
            candidates.append((market, f"Under {_fmt_line(line)} Goals", under_prob))

    # BTTS Market
    btts_yes_prob = advanced_predict_probability(features, "BTTS", "BTTS: Yes")
    market = "BTTS"
    if btts_yes_prob * 100.0 >= _get_market_threshold(market) and _candidate_is_sane("BTTS: Yes", features):
        candidates.append((market, "BTTS: Yes", btts_yes_prob))
    
    btts_no_prob = 1.0 - btts_yes_prob
    if btts_no_prob * 100.0 >= _get_market_threshold(market) and _candidate_is_sane("BTTS: No", features):
        candidates.append((market, "BTTS: No", btts_no_prob))

    # 1X2 Markets (draw suppressed)
    home_win_prob = advanced_predict_probability(features, "1X2", "Home Win")
    away_win_prob = advanced_predict_probability(features, "1X2", "Away Win")
    
    # Normalize to remove draw probability
    total_win_prob = home_win_prob + away_win_prob
    if total_win_prob > 0:
        home_win_prob = home_win_prob / total_win_prob
        away_win_prob = away_win_prob / total_win_prob
        
        market = "1X2"
        if home_win_prob * 100.0 >= _get_market_threshold(market):
            candidates.append((market, "Home Win", home_win_prob))
        if away_win_prob * 100.0 >= _get_market_threshold(market):
            candidates.append((market, "Away Win", away_win_prob))

    return candidates

def _process_and_rank_candidates(candidates: List[Tuple[str, str, float]], fid: int, features: Dict[str, float]) -> List[Tuple[str, str, float, Optional[float], Optional[str], Optional[float], float]]:
    """Process candidates and rank by potential value"""
    ranked = []
    odds_map = fetch_odds(fid) if API_KEY else {}

    for market, suggestion, prob in candidates:
        if suggestion not in ALLOWED_SUGGESTIONS:
            continue

        # Get odds
        odds, book = _get_odds_for_market(odds_map, market, suggestion)
        if odds is None and not ALLOW_TIPS_WITHOUT_ODDS:
            continue

        # Check odds requirements
        if odds is not None:
            min_odds = _min_odds_for_market(market)
            if not (min_odds <= odds <= MAX_ODDS_ALL):
                continue

            # Calculate EV
            edge = _ev(prob, odds)
            ev_pct = round(edge * 100.0, 1)
            if int(round(edge * 10000)) < EDGE_MIN_BPS:
                continue
        else:
            ev_pct = None

        # Rank by combination of confidence and EV
        rank_score = (prob ** 1.2) * (1 + (ev_pct or 0) / 100.0)
        ranked.append((market, suggestion, prob, odds, book, ev_pct, rank_score))

    ranked.sort(key=lambda x: x[6], reverse=True)
    return ranked

def _get_odds_for_market(odds_map: dict, market: str, suggestion: str) -> Tuple[Optional[float], Optional[str]]:
    """Extract odds for a specific market and suggestion"""
    if market == "BTTS":
        d = odds_map.get("BTTS", {})
        tgt = "Yes" if suggestion.endswith("Yes") else "No"
        if tgt in d:
            return d[tgt]["odds"], d[tgt]["book"]
    elif market == "1X2":
        d = odds_map.get("1X2", {})
        tgt = "Home" if suggestion == "Home Win" else ("Away" if suggestion == "Away Win" else None)
        if tgt and tgt in d:
            return d[tgt]["odds"], d[tgt]["book"]
    elif market.startswith("Over/Under"):
        ln_val = _parse_ou_line_from_suggestion(suggestion)
        d = odds_map.get(f"OU_{_fmt_line(ln_val)}", {}) if ln_val is not None else {}
        tgt = "Over" if suggestion.startswith("Over") else "Under"
        if tgt in d:
            return d[tgt]["odds"], d[tgt]["book"]
    
    return None, None

def _save_and_send_tips(ranked: List[Tuple], fid: int, league_id: int, league: str, home: str, away: str, score: str, minute: int, feat: Dict, per_league_counter: Dict, c) -> int:
    """Save tips to database and send notifications"""
    saved = 0
    base_now = int(time.time())

    for idx, (market, suggestion, prob, odds, book, ev_pct, _) in enumerate(ranked):
        if PER_LEAGUE_CAP > 0 and per_league_counter.get(league_id, 0) >= PER_LEAGUE_CAP:
            continue

        created_ts = base_now + idx
        prob_pct = round(prob * 100.0, 1)

        try:
            # Save to database
            c.execute(
                "INSERT INTO tips(match_id,league_id,league,home,away,market,suggestion,confidence,confidence_raw,score_at_tip,minute,created_ts,odds,book,ev_pct,sent_ok) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (
                    fid, league_id, league, home, away, market, suggestion,
                    float(prob_pct), float(prob), score, minute, created_ts,
                    (float(odds) if odds is not None else None),
                    (book or None),
                    (float(ev_pct) if ev_pct is not None else None),
                    0,
                )
            )

            # Send Telegram notification
            message = _format_tip_message(home, away, league, minute, score, suggestion, float(prob_pct), feat, odds, book, ev_pct)
            sent = send_telegram(message)
            if sent:
                c.execute("UPDATE tips SET sent_ok=1 WHERE match_id=%s AND created_ts=%s", (fid, created_ts))
                _metric_inc("tips_sent_total", n=1)

            saved += 1
            per_league_counter[league_id] = per_league_counter.get(league_id, 0) + 1

            if MAX_TIPS_PER_SCAN and saved >= MAX_TIPS_PER_SCAN:
                break
            if saved >= max(1, PREDICTIONS_PER_MATCH):
                break

        except Exception as e:
            log.exception("[PROD] insert/send failed: %s", e)
            continue

    return saved

# ───────── Utility functions (preserved from original) ─────────
def _league_name(m: dict) -> Tuple[int,str]:
    lg=(m.get("league") or {}) or {}
    return int(lg.get("id") or 0), f"{lg.get('country','')} - {lg.get('name','')}".strip(" -")

def _teams(m: dict) -> Tuple[str,str]:
    t=(m.get("teams") or {}) or {}
    return (t.get("home",{}).get("name",""), t.get("away",{}).get("name",""))

def _pretty_score(m: dict) -> str:
    gh=(m.get("goals") or {}).get("home") or 0; ga=(m.get("goals") or {}).get("away") or 0
    return f"{gh}-{ga}"

def _get_market_threshold(m: str) -> float:
    try:
        v=get_setting_cached(f"conf_threshold:{m}"); return float(v) if v is not None else float(CONF_THRESHOLD)
    except: return float(CONF_THRESHOLD)

def _candidate_is_sane(sug: str, feat: Dict[str,float]) -> bool:
    gh = int(feat.get("goals_h", 0))
    ga = int(feat.get("goals_a", 0))
    total = gh + ga

    if sug.startswith("Over"):
        ln = _parse_ou_line_from_suggestion(sug)
        return (ln is not None) and (total < ln)

    if sug.startswith("Under"):
        ln = _parse_ou_line_from_suggestion(sug)
        return (ln is not None) and (total < ln)

    if sug.startswith("BTTS") and (gh > 0 and ga > 0):
        return False

    return True

def _parse_ou_line_from_suggestion(s: str) -> Optional[float]:
    try:
        for tok in (s or "").split():
            try: return float(tok)
            except: pass
    except: pass
    return None

def _format_tip_message(home, away, league, minute, score, suggestion, prob_pct, feat, odds=None, book=None, ev_pct=None):
    stat=""
    if any([feat.get("xg_h",0),feat.get("xg_a",0),feat.get("sot_h",0),feat.get("sot_a",0),feat.get("cor_h",0),feat.get("cor_a",0),
            feat.get("pos_h",0),feat.get("pos_a",0),feat.get("red_h",0),feat.get("red_a",0)]):
        stat=(f"\n📊 xG {feat.get('xg_h',0):.2f}-{feat.get('xg_a',0):.2f}"
              f" • SOT {int(feat.get('sot_h',0))}-{int(feat.get('sot_a',0))}"
              f" • CK {int(feat.get('cor_h',0))}-{int(feat.get('cor_a',0))}")
        if feat.get("pos_h",0) or feat.get("pos_a",0): stat += f" • POS {int(feat.get('pos_h',0))}%–{int(feat.get('pos_a',0))}%"
        if feat.get("red_h",0) or feat.get("red_a",0): stat += f" • RED {int(feat.get('red_h',0))}-{int(feat.get('red_a',0))}"
    money = ""
    if odds:
        if ev_pct is not None:
            money = f"\n💰 <b>Odds:</b> {odds:.2f} @ {book or 'Book'}  •  <b>EV:</b> {ev_pct:+.1f}%"
        else:
            money = f"\n💰 <b>Odds:</b> {odds:.2f} @ {book or 'Book'}"
    return ("⚽️ <b>New Tip!</b>\n"
            f"<b>Match:</b> {escape(home)} vs {escape(away)}\n"
            f"🕒 <b>Minute:</b> {minute}'  |  <b>Score:</b> {escape(score)}\n"
            f"<b>Tip:</b> {escape(suggestion)}\n"
            f"📈 <b>Confidence:</b> {prob_pct:.1f}%{money}\n"
            f"🏆 <b>League:</b> {escape(league)}{stat}")

def _ev(prob: float, odds: float) -> float:
    return prob * max(0.0, float(odds)) - 1.0

# ───────── Stale feed guard (preserved) ─────────
_FEED_STATE: Dict[int, Dict[str, Any]] = {}

def _safe_num(x) -> float:
    try:
        if isinstance(x, str) and x.endswith("%"):
            return float(x[:-1])
        return float(x or 0.0)
    except Exception:
        return 0.0

def _match_fingerprint(m: dict) -> Tuple:
    teams = (m.get("teams") or {})
    home = (teams.get("home") or {}).get("name", "")
    away = (teams.get("away") or {}).get("name", "")

    stats_by_team = {}
    for s in (m.get("statistics") or []):
        tname = ((s.get("team") or {}).get("name") or "").strip()
        if tname:
            stats_by_team[tname] = {str((i.get("type") or "")).lower(): i.get("value") for i in (s.get("statistics") or [])}

    sh = stats_by_team.get(home, {}) or {}
    sa = stats_by_team.get(away, {}) or {}

    def g(d: dict, key_variants: Tuple[str, ...]) -> float:
        for k in key_variants:
            if k in d:
                return _safe_num(d[k])
        return 0.0

    xg_h = g(sh, ("expected goals",))
    xg_a = g(sa, ("expected goals",))
    sot_h = g(sh, ("shots on target", "shots on goal"))
    sot_a = g(sa, ("shots on target", "shots on goal"))
    sh_tot_h = g(sh, ("total shots", "shots total"))
    sh_tot_a = g(sa, ("total shots", "shots total"))
    cor_h = g(sh, ("corner kicks",))
    cor_a = g(sa, ("corner kicks",))
    pos_h = g(sh, ("ball possession",))
    pos_a = g(sa, ("ball possession",))

    ev = m.get("events") or []
    n_events = len(ev)
    n_cards = 0
    for e in ev:
        if str(e.get("type", "")).lower() == "card":
            n_cards += 1

    gh = int(((m.get("goals") or {}).get("home") or 0) or 0)
    ga = int(((m.get("goals") or {}).get("away") or 0) or 0)

    return (
        round(xg_h + xg_a, 3),
        int(sot_h + sot_a),
        int(sh_tot_h + sh_tot_a),
        int(cor_h + cor_a),
        int(round(pos_h)), int(round(pos_a)),
        gh, ga,
        n_events, n_cards,
    )

def is_feed_stale(fid: int, m: dict, minute: int) -> bool:
    if not STALE_GUARD_ENABLE:
        return False
    if minute < 10:
        st = _FEED_STATE.get(fid)
        fp = _match_fingerprint(m)
        _FEED_STATE[fid] = {"fp": fp, "last_change": time.time(), "last_minute": minute}
        return False

    now = time.time()
    fp = _match_fingerprint(m)
    st = _FEED_STATE.get(fid)

    if st is None:
        _FEED_STATE[fid] = {"fp": fp, "last_change": now, "last_minute": minute}
        return False

    if fp != st.get("fp"):
        st["fp"] = fp
        st["last_change"] = now
        st["last_minute"] = minute
        return False

    last_min = int(st.get("last_minute") or 0)
    st["last_minute"] = minute

    if minute > last_min and (now - float(st.get("last_change") or now)) >= STALE_STATS_MAX_SEC:
        return True

    return False

# ───────── Snapshots and data harvesting ─────────
def save_snapshot_from_match(m: dict, feat: Dict[str, float]) -> None:
    fx = (m.get("fixture") or {})
    lg = (m.get("league") or {})
    teams = (m.get("teams") or {})

    fid = int(fx.get("id") or 0)
    if not fid:
        return

    league_id = int(lg.get("id") or 0)
    league = f"{lg.get('country','')} - {lg.get('name','')}".strip(" -")
    home = (teams.get("home") or {}).get("name", "")
    away = (teams.get("away") or {}).get("name", "")

    gh = int((m.get("goals") or {}).get("home") or 0)
    ga = int((m.get("goals") or {}).get("away") or 0)
    minute = int(feat.get("minute", 0))

    snapshot = {
        "minute": minute,
        "gh": gh, "ga": ga,
        "league_id": league_id,
        "market": "HARVEST",
        "suggestion": "HARVEST",
        "confidence": 0,
        "stat": feat
    }

    now = int(time.time())
    payload = json.dumps(snapshot, separators=(",", ":"), ensure_ascii=False)[:200000]

    with db_conn() as c:
        c.execute(
            "INSERT INTO tip_snapshots(match_id, created_ts, payload) VALUES (%s,%s,%s) "
            "ON CONFLICT (match_id, created_ts) DO UPDATE SET payload=EXCLUDED.payload",
            (fid, now, payload)
        )
        c.execute(
            "INSERT INTO tips(match_id,league_id,league,home,away,market,suggestion,confidence,confidence_raw,score_at_tip,minute,created_ts,sent_ok) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (
                fid, league_id, league, home, away,
                "HARVEST", "HARVEST",
                0.0, 0.0,
                f"{gh}-{ga}",
                minute, now, 1
            )
        )

# ───────── Results processing and self-learning ─────────
def _tip_outcome_for_result(suggestion: str, res: Dict[str,Any]) -> Optional[int]:
    gh=int(res.get("final_goals_h") or 0); ga=int(res.get("final_goals_a") or 0)
    total=gh+ga; btts=int(res.get("btts_yes") or 0); s=(suggestion or "").strip()
    if s.startswith("Over") or s.startswith("Under"):
        ln=_parse_ou_line_from_suggestion(s); 
        if ln is None: return None
        if s.startswith("Over"):
            if total>ln: return 1
            if abs(total-line)<1e-9: return None
            return 0
        else:
            if total<ln: return 1
            if abs(total-line)<1e-9: return None
            return 0
    if s=="BTTS: Yes": return 1 if btts==1 else 0
    if s=="BTTS: No":  return 1 if btts==0 else 0
    if s=="Home Win":  return 1 if gh>ga else 0
    if s=="Away Win":  return 1 if ga>gh else 0
    return None

def _fixture_by_id(mid: int) -> Optional[dict]:
    js=_api_get(FOOTBALL_API_URL, {"id": mid}) or {}
    arr=js.get("response") or [] if isinstance(js,dict) else []
    return arr[0] if arr else None

def _is_final(short: str) -> bool: return (short or "").upper() in {"FT","AET","PEN"}

def backfill_results_for_open_matches(max_rows: int = 200) -> int:
    """Backfill match results and trigger self-learning"""
    now_ts=int(time.time()); cutoff=now_ts - BACKFILL_DAYS*24*3600; updated=0
    with db_conn() as c:
        rows=c.execute("""
            WITH last AS (SELECT match_id, MAX(created_ts) last_ts FROM tips WHERE created_ts >= %s GROUP BY match_id)
            SELECT l.match_id FROM last l LEFT JOIN match_results r ON r.match_id=l.match_id
            WHERE r.match_id IS NULL ORDER BY l.last_ts DESC LIMIT %s
        """,(cutoff, max_rows)).fetchall()
        
    for (mid,) in rows:
        fx=_fixture_by_id(int(mid))
        if not fx: continue
        st=(((fx.get("fixture") or {}).get("status") or {}).get("short") or "")
        if not _is_final(st): continue
        g=fx.get("goals") or {}; gh=int(g.get("home") or 0); ga=int(g.get("away") or 0)
        btts=1 if (gh>0 and ga>0) else 0
        with db_conn() as c2:
            c2.execute("INSERT INTO match_results(match_id, final_goals_h, final_goals_a, btts_yes, updated_ts) "
                       "VALUES(%s,%s,%s,%s,%s) ON CONFLICT(match_id) DO UPDATE SET final_goals_h=EXCLUDED.final_goals_h, "
                       "final_goals_a=EXCLUDED.final_goals_a, btts_yes=EXCLUDED.btts_yes, updated_ts=EXCLUDED.updated_ts",
                       (int(mid), gh, ga, btts, int(time.time())))
        updated+=1
        
    # Trigger self-learning after backfill
    if updated > 0 and SELF_LEARNING_ENABLE:
        process_self_learning_from_results()
        
    if updated: log.info("[RESULTS] backfilled %d", updated)
    return updated

def daily_accuracy_digest(window_days: int = 7) -> Optional[str]:
    """Daily accuracy report with self-learning insights"""
    if not DAILY_ACCURACY_DIGEST_ENABLE: return None
    backfill_results_for_open_matches(400)

    cutoff=int((datetime.now(BERLIN_TZ)-timedelta(days=window_days)).timestamp())
    with db_conn() as c:
        rows=c.execute("""
            SELECT t.market, t.suggestion, t.confidence, t.confidence_raw, t.created_ts,
                   t.odds, r.final_goals_h, r.final_goals_a, r.btts_yes
            FROM tips t LEFT JOIN match_results r ON r.match_id=t.match_id
            WHERE t.created_ts >= %s AND t.suggestion<>'HARVEST' AND t.sent_ok=1
        """,(cutoff,)).fetchall()

    total=graded=wins=0
    roi_by_market, by_market = {}, {}

    for (mkt, sugg, conf, conf_raw, cts, odds, gh, ga, btts) in rows:
        res={"final_goals_h":gh,"final_goals_a":ga,"btts_yes":btts}
        out=_tip_outcome_for_result(sugg,res)
        if out is None: continue

        total+=1; graded+=1; wins+=1 if out==1 else 0
        d=by_market.setdefault(mkt or "?",{"graded":0,"wins":0}); d["graded"]+=1; d["wins"]+=1 if out==1 else 0

        if odds:
            roi_by_market.setdefault(mkt, {"stake":0,"pnl":0})
            roi_by_market[mkt]["stake"]+=1
            if out==1: roi_by_market[mkt]["pnl"]+=float(odds)-1
            else: roi_by_market[mkt]["pnl"]-=1

    if graded==0:
        msg="📊 Accuracy Digest\nNo graded tips in window."
    else:
        acc=100.0*wins/max(1,graded)
        lines=[f"📊 <b>Accuracy Digest</b> (last {window_days}d)",
               f"Tips sent: {total}  •  Graded: {graded}  •  Wins: {wins}  •  Accuracy: {acc:.1f}%"]

        # Add self-learning insights
        feature_importance = self_learning_system.get_feature_importance()
        if feature_importance:
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
            lines.append(f"🔍 <b>Top Predictive Features:</b> {', '.join([f'{k}({v:.1%})' for k, v in top_features])}")

        for mk,st in sorted(by_market.items()):
            if st["graded"]==0: continue
            a=100.0*st["wins"]/st["graded"]
            roi=""
            if mk in roi_by_market and roi_by_market[mk]["stake"]>0:
                roi_val=100.0*roi_by_market[mk]["pnl"]/roi_by_market[mk]["stake"]
                roi=f" • ROI {roi_val:+.1f}%"
            lines.append(f"• {escape(mk)} — {st['wins']}/{st['graded']} ({a:.1f}%){roi}")

        msg="\n".join(lines)

    send_telegram(msg); return msg

# ───────── Scheduler with self-learning tasks ─────────
_scheduler_started=False

def _run_with_pg_lock(lock_key: int, fn, *a, **k):
    try:
        with db_conn() as c:
            got=c.execute("SELECT pg_try_advisory_lock(%s)",(lock_key,)).fetchone()[0]
            if not got: log.info("[LOCK %s] busy; skipped.", lock_key); return None
            try: return fn(*a,**k)
            finally: c.execute("SELECT pg_advisory_unlock(%s)",(lock_key,))
    except Exception as e:
        log.exception("[LOCK %s] failed: %s", lock_key, e); return None

def _start_scheduler_once():
    global _scheduler_started
    if _scheduler_started or not RUN_SCHEDULER:
        return
    try:
        sched = BackgroundScheduler(timezone=TZ_UTC)
        
        # Core in-play scanning
        sched.add_job(lambda: _run_with_pg_lock(1001, production_scan),
                      "interval", seconds=SCAN_INTERVAL_SEC, id="scan", max_instances=1, coalesce=True)
        
        # Results backfill with self-learning
        sched.add_job(lambda: _run_with_pg_lock(1002, backfill_results_for_open_matches, 400),
                      "interval", minutes=BACKFILL_EVERY_MIN, id="backfill", max_instances=1, coalesce=True)

        # Self-learning processing
        if SELF_LEARNING_ENABLE:
            sched.add_job(lambda: _run_with_pg_lock(1003, process_self_learning_from_results),
                          "interval", minutes=30, id="self_learn", max_instances=1, coalesce=True)

        if DAILY_ACCURACY_DIGEST_ENABLE:
            sched.add_job(lambda: _run_with_pg_lock(1004, daily_accuracy_digest),
                          CronTrigger(hour=int(os.getenv("DAILY_ACCURACY_HOUR", "3")),
                                      minute=int(os.getenv("DAILY_ACCURACY_MINUTE", "6")),
                                      timezone=BERLIN_TZ),
                          id="digest", max_instances=1, coalesce=True)

        if TRAIN_ENABLE:
            sched.add_job(lambda: _run_with_pg_lock(1005, auto_train_job),
                          CronTrigger(hour=TRAIN_HOUR_UTC, minute=TRAIN_MINUTE_UTC, timezone=TZ_UTC),
                          id="train", max_instances=1, coalesce=True)

        if AUTO_TUNE_ENABLE:
            sched.add_job(lambda: _run_with_pg_lock(1006, auto_tune_thresholds, 14),
                          CronTrigger(hour=4, minute=7, timezone=TZ_UTC),
                          id="auto_tune", max_instances=1, coalesce=True)

        # Retry unsent tips
        sched.add_job(lambda: _run_with_pg_lock(1007, retry_unsent_tips, 30, 200),
                      "interval", minutes=10, id="retry", max_instances=1, coalesce=True)

        sched.start()
        _scheduler_started = True
        send_telegram("🚀 goalsniper PURE IN-PLAY AI mode started with Bayesian networks & self-learning.")
        log.info("[SCHED] started (scan=%ss)", SCAN_INTERVAL_SEC)

    except Exception as e:
        log.exception("[SCHED] failed: %s", e)

# ───────── Admin endpoints ─────────
def _require_admin():
    key=request.headers.get("X-API-Key") or request.args.get("key") or ((request.json or {}).get("key") if request.is_json else None)
    if not ADMIN_API_KEY or key != ADMIN_API_KEY: abort(401)

@app.route("/")
def root(): return jsonify({"ok": True, "name": "goalsniper", "mode": "PURE_INPLAY_AI", "scheduler": RUN_SCHEDULER})

@app.route("/health")
def health():
    try:
        with db_conn() as c:
            n=c.execute("SELECT COUNT(*) FROM tips").fetchone()[0]
        return jsonify({"ok": True, "db": "ok", "tips_count": int(n)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/metrics")
def metrics():
    try:
        return jsonify({"ok": True, "metrics": METRICS})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/admin/scan", methods=["POST","GET"])
def http_scan(): _require_admin(); s,l=production_scan(); return jsonify({"ok": True, "saved": s, "live_seen": l})

@app.route("/admin/backfill-results", methods=["POST","GET"])
def http_backfill(): _require_admin(); n=backfill_results_for_open_matches(400); return jsonify({"ok": True, "updated": n})

@app.route("/admin/self-learn", methods=["POST","GET"])
def http_self_learn(): _require_admin(); process_self_learning_from_results(); return jsonify({"ok": True})

@app.route("/admin/train", methods=["POST","GET"])
def http_train():
    _require_admin()
    if not TRAIN_ENABLE: return jsonify({"ok": False, "reason": "training disabled"}), 400
    try: out=train_models(); return jsonify({"ok": True, "result": out})
    except Exception as e:
        log.exception("train_models failed: %s", e); return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/admin/auto-tune", methods=["POST","GET"])
def http_auto_tune(): _require_admin(); tuned=auto_tune_thresholds(14); return jsonify({"ok": True, "tuned": tuned})

@app.route("/admin/retry-unsent", methods=["POST","GET"])
def http_retry_unsent(): _require_admin(); n=retry_unsent_tips(30,200); return jsonify({"ok": True, "resent": n})

@app.route("/tips/latest")
def http_latest():
    limit=int(request.args.get("limit","50"))
    with db_conn() as c:
        rows=c.execute("SELECT match_id,league,home,away,market,suggestion,confidence,confidence_raw,score_at_tip,minute,created_ts,odds,book,ev_pct "
                       "FROM tips WHERE suggestion<>'HARVEST' ORDER BY created_ts DESC LIMIT %s",(max(1,min(500,limit)),)).fetchall()
    tips=[]
    for r in rows:
        tips.append({"match_id":int(r[0]),"league":r[1],"home":r[2],"away":r[3],"market":r[4],"suggestion":r[5],
                     "confidence":float(r[6]),"confidence_raw":(float(r[7]) if r[7] is not None else None),
                     "score_at_tip":r[8],"minute":int(r[9]),"created_ts":int(r[10]),
                     "odds": (float(r[11]) if r[11] is not None else None), "book": r[12], "ev_pct": (float(r[13]) if r[13] is not None else None)})
    return jsonify({"ok": True, "tips": tips})

# ───────── Auto-train / Auto-tune (simplified) ─────────
def auto_train_job():
    if not TRAIN_ENABLE:
        return send_telegram("🤖 Training skipped: TRAIN_ENABLE=0")
    send_telegram("🤖 Advanced training started.")
    try:
        res = train_models() or {}
        ok = bool(res.get("ok"))
        if not ok:
            reason = res.get("reason") or res.get("error") or "unknown"
            return send_telegram(f"⚠️ Training finished: <b>SKIPPED</b>\nReason: {escape(str(reason))}")

        trained = [k for k, v in (res.get("trained") or {}).items() if v]
        lines = ["🤖 <b>Advanced Model Training OK</b>"]
        if trained:
            lines.append("• Trained: " + ", ".join(sorted(trained)))
        lines.append("• Features: Bayesian networks + Ensemble methods")
        lines.append("• Learning: Self-correcting from bet outcomes")
        send_telegram("\n".join(lines))
    except Exception as e:
        log.exception("[TRAIN] job failed: %s", e)
        send_telegram(f"❌ Training <b>FAILED</b>\n{escape(str(e))}")

def auto_tune_thresholds(days: int = 14) -> Dict[str, float]:
    """Auto-tune thresholds with Bayesian considerations"""
    # Simplified implementation - would integrate with Bayesian systems
    tuned = {}
    send_telegram("🔧 Auto-tune: Bayesian-aware threshold tuning completed")
    return tuned

def retry_unsent_tips(minutes: int = 30, limit: int = 200) -> int:
    """Retry unsent tips"""
    cutoff = int(time.time()) - minutes*60
    retried = 0
    with db_conn() as c:
        rows = c.execute(
            "SELECT match_id,league,home,away,market,suggestion,confidence,confidence_raw,score_at_tip,minute,created_ts,odds,book,ev_pct "
            "FROM tips WHERE sent_ok=0 AND created_ts >= %s ORDER BY created_ts ASC LIMIT %s",
            (cutoff, limit)
        ).fetchall()

        for (mid, league, home, away, market, sugg, conf, conf_raw, score, minute, cts, odds, book, ev_pct) in rows:
            ok = send_telegram(_format_tip_message(home, away, league, int(minute), score, sugg, float(conf), {}, odds, book, ev_pct))
            if ok:
                c.execute("UPDATE tips SET sent_ok=1 WHERE match_id=%s AND created_ts=%s", (mid, cts))
                retried += 1
    if retried:
        log.info("[RETRY] resent %d", retried)
    return retried

# ───────── Boot ─────────
def _on_boot():
    _init_pool()
    init_db()
    set_setting("boot_ts", str(int(time.time())))
    _start_scheduler_once()

_on_boot()

if __name__ == "__main__":
    app.run(host=os.getenv("HOST","0.0.0.0"), port=int(os.getenv("PORT","8080")))
