# goalsniper — PURE IN-PLAY AI mode with Bayesian networks & self-learning
# Upgraded: Bayesian networks, self-learning from wrong bets, advanced ensemble models

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
    "ensemble_predictions_total": 0,
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

# Self-learning controls
SELF_LEARNING_ENABLE   = os.getenv("SELF_LEARNING_ENABLE", "1") not in ("0","false","False","no","NO")
SELF_LEARN_BATCH_SIZE  = int(os.getenv("SELF_LEARN_BATCH_SIZE", "50"))
BAYESIAN_PRIOR_ALPHA   = float(os.getenv("BAYESIAN_PRIOR_ALPHA", "2.0"))
BAYESIAN_PRIOR_BETA    = float(os.getenv("BAYESIAN_PRIOR_BETA", "2.0"))

AUTO_TUNE_ENABLE        = os.getenv("AUTO_TUNE_ENABLE", "0") not in ("0","false","False","no","NO")
TARGET_PRECISION        = float(os.getenv("TARGET_PRECISION", "0.60"))
THRESH_MIN_PREDICTIONS  = int(os.getenv("THRESH_MIN_PREDICTIONS", "25"))
MIN_THRESH              = float(os.getenv("MIN_THRESH", "55"))
MAX_THRESH              = float(os.getenv("MAX_THRESH", "85"))

STALE_GUARD_ENABLE = os.getenv("STALE_GUARD_ENABLE", "1") not in ("0","false","False","no","NO")
STALE_STATS_MAX_SEC = int(os.getenv("STALE_STATS_MAX_SEC", "240"))
MARKET_CUTOFFS_RAW  = os.getenv("MARKET_CUTOFFS", "BTTS=75,1X2=80,OU=88")
TIP_MAX_MINUTE_ENV  = os.getenv("TIP_MAX_MINUTE", "")

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
MIN_ODDS_OU   = float(os.getenv("MIN_ODDS_OU",   "1.50"))
MIN_ODDS_BTTS = float(os.getenv("MIN_ODDS_BTTS", "1.50"))
MIN_ODDS_1X2  = float(os.getenv("MIN_ODDS_1X2",  "1.50"))
MAX_ODDS_ALL  = float(os.getenv("MAX_ODDS_ALL",  "20.0"))
EDGE_MIN_BPS  = int(os.getenv("EDGE_MIN_BPS", "600"))
ODDS_BOOKMAKER_ID = os.getenv("ODDS_BOOKMAKER_ID")
ALLOW_TIPS_WITHOUT_ODDS = os.getenv("ALLOW_TIPS_WITHOUT_ODDS","0") not in ("0","false","False","no","NO")

ODDS_SOURCE         = os.getenv("ODDS_SOURCE", "auto").lower()
ODDS_AGGREGATION    = os.getenv("ODDS_AGGREGATION", "median").lower()
ODDS_OUTLIER_MULT   = float(os.getenv("ODDS_OUTLIER_MULT", "1.8"))
ODDS_REQUIRE_N_BOOKS= int(os.getenv("ODDS_REQUIRE_N_BOOKS", "2"))
ODDS_FAIR_MAX_MULT  = float(os.getenv("ODDS_FAIR_MAX_MULT", "2.5"))

# ───────── Markets allow-list (draw suppressed) ─────────
ALLOWED_SUGGESTIONS = {"BTTS: Yes", "BTTS: No", "Home Win", "Away Win"}
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

# ───────── Advanced Model Architecture ─────────
class AdvancedEnsemblePredictor:
    """Advanced ensemble combining multiple models with Bayesian calibration"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_selector: Optional[SelectKBest] = None
        self.bayesian_prior_alpha = BAYESIAN_PRIOR_ALPHA
        self.bayesian_prior_beta  = BAYESIAN_PRIOR_BETA
        self.performance_history: List[int] = []
        
    def train(self, features: List[Dict[str, Any]], targets: List[int]) -> Dict[str, Any]:
        """Train ensemble of models with feature selection"""
        if not features or not targets:
            return {"error": "No training data"}
            
        df = pd.DataFrame(features)
        if len(df) < 10:
            return {"error": "Insufficient training data"}
            
        df = df.fillna(0)
        
        # Feature selection
        if len(df.columns) > 5:
            self.feature_selector = SelectKBest(f_classif, k=min(10, len(df.columns)))
            X_selected = self.feature_selector.fit_transform(df.values, np.array(targets, dtype=int))
            selected_features = df.columns[self.feature_selector.get_support()].tolist()
        else:
            X_selected = df.values
            selected_features = df.columns.tolist()
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected)
        self.scalers[self.model_name] = scaler
        
        # Train ensemble models
        models_to_train = {
            "logistic": LogisticRegression(max_iter=1000, class_weight="balanced"),
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "gradient_boost": GradientBoostingClassifier(n_estimators=100, random_state=42),
        }
        
        self.models[self.model_name] = {}
        y = np.array(targets, dtype=int)
        for name, model in models_to_train.items():
            try:
                if name == "logistic":
                    calibrated_model = CalibratedClassifierCV(model, method="isotonic", cv=3)
                else:
                    calibrated_model = CalibratedClassifierCV(model, method="sigmoid", cv=3)
                calibrated_model.fit(X_scaled, y)
                self.models[self.model_name][name] = calibrated_model
            except Exception as e:
                log.error("Failed to train %s for %s: %s", name, self.model_name, e)
        
        return {
            "ok": True,
            "trained_models": list(self.models[self.model_name].keys()),
            "feature_count": len(selected_features),
            "sample_count": len(features),
        }
    
    def predict_probability(self, features: Dict[str, float]) -> float:
        """Get ensemble prediction with Bayesian confidence intervals"""
        if not self.models.get(self.model_name):
            return 0.5  # Neutral prior
            
        feature_vector = self._prepare_features(features)
        if feature_vector is None:
            return 0.5
            
        predictions: List[float] = []
        for model_name, model in self.models[self.model_name].items():
            try:
                prob = float(model.predict_proba(feature_vector.reshape(1, -1))[0][1])
                predictions.append(prob)
            except Exception as e:
                log.error("Prediction failed for %s: %s", model_name, e)
                continue
        
        if not predictions:
            return 0.5
            
        ensemble_prob = float(np.mean(predictions))
        bayesian_prob = self._apply_bayesian_prior(ensemble_prob)
        
        _metric_inc("ensemble_predictions_total", label=self.model_name)
        return bayesian_prob

    def calculate_tip_reliability(match_id: int, suggestion: str, probability: float) -> Dict[str, Any]:
        """Calculate reliability metrics for each tip"""
        reliability_score = 0.0
        flags = []
    
        # Score based on probability strength
        if probability >= 0.7:
            reliability_score += 0.3
            flags.append("high_confidence")
        elif probability >= 0.6:
            reliability_score += 0.2
            flags.append("medium_confidence")
        else:
            reliability_score += 0.1
            flags.append("low_confidence")
    
        # Score based on data quality (minute of match)
        with db_conn() as c:
            minute_row = c.execute(
                "SELECT minute FROM tips WHERE match_id=%s ORDER BY created_ts DESC LIMIT 1",
                (match_id,)
            ).fetchone()
        
        if minute_row and minute_row[0] >= 60:
            reliability_score += 0.3
            flags.append("late_match_data")
        elif minute_row and minute_row[0] >= 30:
            reliability_score += 0.2
            flags.append("mid_match_data")
        else:
            reliability_score += 0.1
            flags.append("early_match_data")
    
        # Score based on feature completeness
        with db_conn() as c:
            feature_row = c.execute(
                "SELECT COUNT(*) FROM self_learning_data WHERE match_id=%s",
                (match_id,)
            ).fetchone()
    
        if feature_row and feature_row[0] > 5:
            reliability_score += 0.2
            flags.append("rich_features")
        else:
            reliability_score += 0.1
            flags.append("basic_features")
    
        # Market-specific adjustments
        if "Over" in suggestion or "Under" in suggestion:
            reliability_score += 0.1
            flags.append("ou_market")
        elif "BTTS" in suggestion:
            reliability_score += 0.05
            flags.append("btts_market")
    
        return {
            "reliability_score": min(1.0, reliability_score),
            "flags": flags,
            "grade": "A" if reliability_score >= 0.8 else "B" if reliability_score >= 0.6 else "C"
        }
    
    def _prepare_features(self, features: Dict[str, float]) -> Optional[np.ndarray]:
        if self.model_name not in self.scalers:
            return None
            
        feature_df = pd.DataFrame([features]).fillna(0)
        
        if self.feature_selector:
            try:
                X_selected = self.feature_selector.transform(feature_df.values)
            except Exception:
                X_selected = feature_df.values
        else:
            X_selected = feature_df.values
            
        scaler = self.scalers[self.model_name]
        return scaler.transform(X_selected)[0]
    
    def _apply_bayesian_prior(self, ensemble_prob: float) -> float:
        if not self.performance_history:
            return ensemble_prob
            
        recent = self.performance_history[-20:]
        if len(recent) < 5:
            return ensemble_prob
            
        wins = sum(recent)
        total = len(recent)
        
        posterior_alpha = self.bayesian_prior_alpha + wins
        posterior_beta  = self.bayesian_prior_beta  + (total - wins)
        
        bayesian_correction = posterior_alpha / max(1e-9, (posterior_alpha + posterior_beta))
        blended_prob = 0.7 * ensemble_prob + 0.3 * bayesian_correction
        return float(blended_prob)
    
    def update_performance(self, outcome: int) -> None:
        """Update performance history for Bayesian learning"""
        self.performance_history.append(int(outcome))
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
        _metric_inc("bayesian_updates_total", label=self.model_name)

# ───────── Bayesian Network Implementation ─────────
class BayesianBettingNetwork:
    """Bayesian network for in-play betting predictions"""
    
    def __init__(self):
        self.network_structure = self._initialize_network()
        self.prior_knowledge   = self._load_prior_knowledge()
        
    def _initialize_network(self) -> Dict[str, Any]:
        return {
            "nodes": {
                "momentum": ["possession", "recent_shots", "recent_goals"],
                "pressure": ["score_state", "time_remaining", "home_advantage"],
                "goal_probability": ["momentum", "pressure", "xg_accumulated"],
                "btts_probability": ["goal_probability_home", "goal_probability_away", "defensive_weakness"],
                "market_outcome": ["goal_probability", "btts_probability", "historical_success"],
            },
            "edges": [
                ("momentum", "goal_probability"),
                ("pressure", "goal_probability"),
                ("goal_probability", "market_outcome"),
                ("btts_probability", "market_outcome"),
            ],
        }
    
    def _load_prior_knowledge(self) -> Dict[str, Any]:
        return {
            "goal_probability_given_momentum": 0.65,
            "goal_probability_given_pressure": 0.55,
            "btts_given_offensive_game": 0.72,
            "home_win_given_dominance": 0.68,
        }
    
    def infer_probability(self, features: Dict[str, float], market: str) -> float:
        momentum_score   = self._calculate_momentum(features)
        pressure_score   = self._calculate_pressure(features)
        historical_prior = self._get_historical_context(features)
        
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
        
        posterior_prob = self._bayesian_update(base_prob, historical_prior)
        
        _metric_inc("bayesian_updates_total", label=market)
        return posterior_prob
    
    def _calculate_momentum(self, features: Dict[str, float]) -> float:
        recent_shots         = features.get("sot_sum", 0.0) / max(1.0, features.get("minute", 1.0))
        possession_dominance = abs(features.get("pos_diff", 0.0)) / 100.0
        xg_accumulated       = features.get("xg_sum", 0.0)
        
        momentum = recent_shots * 0.4 + possession_dominance * 0.3 + xg_accumulated * 0.3
        return float(min(1.0, max(0.0, momentum)))
    
    def _calculate_pressure(self, features: Dict[str, float]) -> float:
        minute          = float(features.get("minute", 0.0))
        goal_difference = abs(float(features.get("goals_diff", 0.0)))
        time_pressure   = minute / 90.0
        pressure        = time_pressure * 0.5 + (goal_difference * 0.1) * 0.3 + 0.2
        return float(min(1.0, max(0.0, pressure)))
    
    def _get_historical_context(self, features: Dict[str, float]) -> float:
        # Placeholder: could use long-term hit-rate priors
        return 0.5
    
    def _infer_over_probability(self, momentum: float, pressure: float, features: Dict[str, float]) -> float:
        current_goals     = float(features.get("goals_sum", 0.0))
        minute            = float(features.get("minute", 1.0))
        goals_per_minute  = current_goals / max(1.0, minute)
        momentum_effect   = momentum * 0.6
        pressure_effect   = pressure * 0.3
        current_rate_eff  = min(1.0, goals_per_minute * 10.0) * 0.1
        return float(max(0.0, min(1.0, momentum_effect + pressure_effect + current_rate_eff)))
    
    def _infer_under_probability(self, momentum: float, pressure: float, features: Dict[str, float]) -> float:
        current_goals       = float(features.get("goals_sum", 0.0))
        minute              = float(features.get("minute", 1.0))
        low_momentum_effect = (1.0 - momentum) * 0.5
        low_scoring_effect  = (1.0 - min(1.0, current_goals / 3.0)) * 0.3
        time_pressure_eff   = (minute / 90.0) * 0.2
        return float(max(0.0, min(1.0, low_momentum_effect + low_scoring_effect + time_pressure_eff)))
    
    def _infer_btts_probability(self, features: Dict[str, float]) -> float:
        both_scored        = int(features.get("goals_h", 0.0) > 0 and features.get("goals_a", 0.0) > 0)
        attacking_pressure = (features.get("sot_sum", 0.0) / max(1.0, features.get("minute", 1.0))) * 5.0
        defensive_weakness = (features.get("xg_sum", 0.0) / max(1.0, features.get("minute", 1.0))) * 3.0
        base_prob = min(0.8, 0.3 + attacking_pressure * 0.4 + defensive_weakness * 0.3)
        if both_scored:
            base_prob = max(base_prob, 0.85)
        return float(max(0.0, min(1.0, base_prob)))
    
    def _infer_win_probability(self, features: Dict[str, float], market: str) -> float:
        is_home             = "Home" in market
        goal_diff           = float(features.get("goals_diff", 0.0))
        xg_diff             = float(features.get("xg_diff", 0.0))
        possession_adv      = float(features.get("pos_diff", 0.0)) / 100.0
        
        if is_home:
            dominance = max(0.0, goal_diff) * 0.3 + max(0.0, xg_diff) * 0.4 + max(0.0, possession_adv) * 0.3
        else:
            dominance = max(0.0, -goal_diff) * 0.3 + max(0.0, -xg_diff) * 0.4 + max(0.0, -possession_adv) * 0.3
        
        return float(min(0.9, max(0.0, 0.4 + dominance * 0.5)))
    
    def _bayesian_update(self, likelihood: float, prior: float) -> float:
        posterior = prior * 0.3 + likelihood * 0.7
        return float(max(0.1, min(0.9, posterior)))

# ───────── Self-Learning System ─────────
class SelfLearningSystem:
    """System that learns from wrong bets and improves predictions"""
    
    def __init__(self):
        self.learning_batch: List[Dict[str, Any]] = []
        self.model_performance: Dict[str, Dict[str, float]] = {}
        self.feature_importance: Dict[str, Dict[str, float]] = {}
        
    def record_prediction_outcome(self, prediction_data: Dict[str, Any], actual_outcome: int) -> None:
        learning_example = {
            **prediction_data,
            "actual_outcome": actual_outcome,
            "timestamp": time.time(),
            "was_correct": int(prediction_data.get("predicted_outcome", 0)) == int(actual_outcome),
        }
        self.learning_batch.append(learning_example)
        
        model_key = prediction_data.get("model_type", "unknown")
        self.model_performance.setdefault(model_key, {"total": 0, "correct": 0})
        self.model_performance[model_key]["total"]  += 1
        if learning_example["was_correct"]:
            self.model_performance[model_key]["correct"] += 1
        
        if len(self.learning_batch) >= SELF_LEARN_BATCH_SIZE:
            self._process_learning_batch()
    
    def _process_learning_batch(self) -> None:
        if not self.learning_batch:
            return
            
        log.info("[SELF-LEARNING] Processing %d learning examples", len(self.learning_batch))
        wrong_predictions   = [ex for ex in self.learning_batch if not ex["was_correct"]]
        correct_predictions = [ex for ex in self.learning_batch if ex["was_correct"]]
        
        if wrong_predictions:
            self._learn_from_errors(wrong_predictions)
            self._update_feature_importance(wrong_predictions, correct_predictions)
            self._adjust_model_weights()
        
        self.learning_batch = []
        _metric_inc("self_learning_updates_total")
        log.info("[SELF-LEARNING] Batch processing completed")
    
    def _learn_from_errors(self, wrong_predictions: List[Dict[str, Any]]) -> None:
        error_patterns: Dict[str, List[float]] = {}
        for wrong_pred in wrong_predictions:
            market   = wrong_pred.get("market", "unknown")
            features = wrong_pred.get("features", {})
            for feature_name, feature_value in features.items():
                if isinstance(feature_value, (int, float)):
                    pattern_key = f"{market}_{feature_name}"
                    error_patterns.setdefault(pattern_key, []).append(float(feature_value))
        self._update_calibration_parameters(error_patterns)
    
    def _update_feature_importance(
        self,
        wrong_predictions: List[Dict[str, Any]],
        correct_predictions: List[Dict[str, Any]],
    ) -> None:
        all_predictions = wrong_predictions + correct_predictions
        for pred in all_predictions:
            features   = pred.get("features", {})
            is_correct = bool(pred.get("was_correct", False))
            for feature_name, feature_value in features.items():
                if isinstance(feature_value, (int, float)):
                    self.feature_importance.setdefault(feature_name, {"total": 0, "predictive": 0})
                    self.feature_importance[feature_name]["total"] += 1
                    if is_correct:
                        self.feature_importance[feature_name]["predictive"] += 1
    
    def _adjust_model_weights(self) -> None:
        for model_type, perf in self.model_performance.items():
            accuracy = perf["correct"] / max(1, perf["total"])
            if accuracy < 0.55 and perf["total"] > 10:
                log.warning("[SELF-LEARNING] Model %s underperforming: %.2f%%", model_type, accuracy * 100.0)
                # Hook for adjusting ensemble weights if needed.
    
    def _update_calibration_parameters(self, error_patterns: Dict[str, List[float]]) -> None:
        if error_patterns:
            log.info("[SELF-LEARNING] Found %d error patterns", len(error_patterns))
    
    def get_feature_importance(self) -> Dict[str, float]:
        importance_scores: Dict[str, float] = {}
        for feature, stats in self.feature_importance.items():
            if stats["total"] > 0:
                importance_scores[feature] = stats["predictive"] / stats["total"]
        return importance_scores

def verify_learning_system() -> Dict[str, Any]:
    """Check if the learning system is working properly"""
    status = {}
    
    with db_conn() as c:
        # Check if we have match results to learn from
        results_count = c.execute("SELECT COUNT(*) FROM match_results").fetchone()[0]
        status['match_results_available'] = results_count
        
        # Check self-learning data
        learning_data = c.execute("SELECT COUNT(*) FROM self_learning_data").fetchone()[0]
        status['learning_records'] = learning_data
        
        # Check if learning is happening
        learned_outcomes = c.execute(
            "SELECT COUNT(*) FROM self_learning_data WHERE actual_outcome IS NOT NULL"
        ).fetchone()[0]
        status['learned_outcomes'] = learned_outcomes
        
        # Check recent tip performance
        recent_tips = c.execute("""
            SELECT COUNT(*), 
                   SUM(CASE WHEN t.suggestion <> 'HARVEST' THEN 1 ELSE 0 END) as real_tips,
                   AVG(t.confidence) as avg_confidence
            FROM tips t 
            WHERE t.created_ts >= %s
        """, (int(time.time()) - 24*3600,)).fetchone()
        status['recent_tips_24h'] = recent_tips[1] if recent_tips else 0
        status['avg_confidence'] = float(recent_tips[2]) if recent_tips and recent_tips[2] else 0
    
    return status

# ───────── Initialize Advanced Systems ─────────
bayesian_network    = BayesianBettingNetwork()
self_learning_system = SelfLearningSystem()
advanced_predictors: Dict[str, AdvancedEnsemblePredictor] = {}

def get_advanced_predictor(model_name: str) -> AdvancedEnsemblePredictor:
    if model_name not in advanced_predictors:
        advanced_predictors[model_name] = AdvancedEnsemblePredictor(model_name)
    return advanced_predictors[model_name]

# ───────── Optional import: trainer ─────────
_TRAIN_MODULE_OK = False
try:
    import train_models as _tm
    train_models = _tm.train_models
    _TRAIN_MODULE_OK = True
except Exception as e:
    _IMPORT_ERR = repr(e)
    def train_models(*args, **kwargs):
        log.warning("train_models not available: %s", _IMPORT_ERR)
        return {"ok": False, "reason": f"train_models import failed: {_IMPORT_ERR}"}

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
            _metric_inc("db_errors_total", n=1)
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
            _metric_inc("db_errors_total", n=1)
            log.error("[DB] reinit failed: %s", e)
            return False

# ───────── Settings cache ─────────
class _KVCache:
    def __init__(self, ttl: int):
        self.ttl  = ttl
        self.data: Dict[str, Tuple[float, Optional[str]]] = {}
    def get(self, k: str) -> Optional[str]:
        if _redis:
            try:
                v = _redis.get(f"gs:{k}")
                return v.decode("utf-8") if v is not None else None
            except Exception:
                pass
        v = self.data.get(k)
        if not v:
            return None
        ts, val = v
        if time.time() - ts > self.ttl:
            self.data.pop(k, None)
            return None
        return val
    def set(self, k: str, v: Optional[str]) -> None:
        if _redis:
            try:
                _redis.setex(f"gs:{k}", self.ttl, v if v is not None else "")
                return
            except Exception:
                pass
        self.data[k] = (time.time(), v)
    def invalidate(self, k: Optional[str] = None) -> None:
        if _redis and k:
            try:
                _redis.delete(f"gs:{k}")
                return
            except Exception:
                pass
        if k is None:
            self.data.clear()
        else:
            self.data.pop(k, None)

_SETTINGS_CACHE = _KVCache(SETTINGS_TTL)
_MODELS_CACHE   = _KVCache(MODELS_TTL)

# ───────── Settings helpers ─────────
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

def invalidate_model_caches_for_key(key: str) -> None:
    if key.lower().startswith(("model", "model_latest", "model_v2")):
        _MODELS_CACHE.invalidate(key)

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
        c.execute(
            """CREATE TABLE IF NOT EXISTS odds_history (
            match_id BIGINT,
            captured_ts BIGINT,
            market TEXT,
            selection TEXT,
            odds DOUBLE PRECISION,
            book TEXT,
            PRIMARY KEY (match_id, market, selection, captured_ts)
        )"""
        )
        c.execute("CREATE INDEX IF NOT EXISTS idx_odds_hist_match ON odds_history (match_id, captured_ts DESC)")
        c.execute(
            """CREATE TABLE IF NOT EXISTS self_learning_data (
            id SERIAL PRIMARY KEY,
            match_id BIGINT,
            market TEXT,
            features JSONB,
            prediction_probability DOUBLE PRECISION,
            actual_outcome INTEGER,
            learning_timestamp BIGINT
        )"""
        )
        c.execute("CREATE INDEX IF NOT EXISTS idx_learning_timestamp ON self_learning_data (learning_timestamp DESC)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_tips_created ON tips (created_ts DESC)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_tips_match ON tips (match_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_tips_sent ON tips (sent_ok, created_ts DESC)")

# ───────── Telegram ─────────
def send_telegram(text: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.error("❌ Telegram credentials missing - BOT_TOKEN: %s, CHAT_ID: %s", 
                 "SET" if TELEGRAM_BOT_TOKEN else "MISSING", 
                 "SET" if TELEGRAM_CHAT_ID else "MISSING")
        return False
    try:
        log.info("📤 Attempting to send Telegram message...")
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
            log.info("✅ Telegram message sent successfully")
            _metric_inc("tips_sent_total", n=1)
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
        log.warning("🚫 API Circuit Breaker open until %s", API_CB["opened_until"])
        return None
    
    lbl = "unknown"
    try:
        if "/odds/live" in url or "/odds" in url:
            lbl = "odds"
        elif "/statistics" in url:
            lbl = "statistics"
        elif "/events" in url:
            lbl = "events"
        elif "/fixtures" in url:
            lbl = "fixtures"
    except Exception:
        lbl = "unknown"

    try:
        log.debug("🌐 API call to %s with params %s", url, params)
        r = session.get(url, headers=HEADERS, params=params, timeout=min(timeout, REQ_TIMEOUT_SEC))
        _metric_inc("api_calls_total", label=lbl, n=1)
        if r.status_code == 429:
            log.warning("🚫 API Rate Limited - status 429")
            METRICS["api_rate_limited_total"] += 1
            API_CB["failures"] += 1
        elif r.status_code >= 500:
            log.error("❌ API Server Error - status %s", r.status_code)
            API_CB["failures"] += 1
        else:
            API_CB["failures"] = 0

        if API_CB["failures"] >= API_CB_THRESHOLD:
            API_CB["opened_until"] = now + API_CB_COOLDOWN_SEC
            log.warning("[CB] API-Football opened for %ss", API_CB_COOLDOWN_SEC)

        if r.ok:
            log.debug("✅ API call successful")
        else:
            log.error("❌ API call failed: %s - %s", r.status_code, r.text)
            
        return r.json() if r.ok else None
    except Exception as e:
        log.error("❌ API call exception: %s", e)
        API_CB["failures"] += 1
        if API_CB["failures"] >= API_CB_THRESHOLD:
            API_CB["opened_until"] = time.time() + API_CB_COOLDOWN_SEC
            log.warning("[CB] API-Football opened due to exceptions")
        return None

# ───────── League filter ─────────
_BLOCK_PATTERNS = [
    "u17",
    "u18",
    "u19",
    "u20",
    "u21",
    "u23",
    "youth",
    "junior",
    "reserve",
    "res.",
    "friendlies",
    "friendly",
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
        log.debug("📊 Using negative cache for stats %s", fid)
        return []
    if fid in STATS_CACHE and now - STATS_CACHE[fid][0] < 90:
        log.debug("📊 Using cache for stats %s", fid)
        return STATS_CACHE[fid][1]
    js  = _api_get(f"{FOOTBALL_API_URL}/statistics", {"fixture": fid}) or {}
    out = js.get("response", []) if isinstance(js, dict) else []
    STATS_CACHE[fid] = (now, out)
    if not out:
        log.debug("📊 No stats found for %s, caching negative", fid)
        NEG_CACHE[k] = (now, True)
    else:
        log.debug("📊 Found %s stats records for %s", len(out), fid)
    return out

def fetch_live_fixtures_only() -> List[dict]:
    log.info("🌐 Fetching live fixtures...")
    js = _api_get(FOOTBALL_API_URL, {"live": "all"}) or {}
    matches = [
        m
        for m in (js.get("response", []) if isinstance(js, dict) else [])
        if not _blocked_league(m.get("league") or {})
    ]
    log.info("📋 Found %s total live matches before filtering", len(matches))
    
    out = []
    for m in matches:
        st      = ((m.get("fixture", {}) or {}).get("status", {}) or {})
        elapsed = st.get("elapsed")
        short   = (st.get("short") or "").upper()
        if elapsed is None or elapsed > 120 or short not in INPLAY_STATUSES:
            log.debug("⏩ Skipping match %s - elapsed: %s, status: %s", 
                     m.get('fixture', {}).get('id'), elapsed, short)
            continue
        out.append(m)
    
    log.info("🎯 Filtered to %s in-play matches", len(out))
    return out

def _quota_per_scan() -> int:
    scans_per_day = max(1, int(86400 / max(1, SCAN_INTERVAL_SEC)))
    ppf = 1 + (1 if USE_EVENTS_IN_FEATURES else 0)
    safe = int(API_BUDGET_DAILY / max(1, (scans_per_day * ppf))) - 10
    quota = max(1, min(MAX_FIXTURES_PER_SCAN, safe))
    log.debug("💰 Quota calculation: scans_per_day=%s, ppf=%s, safe=%s, quota=%s", 
             scans_per_day, ppf, safe, quota)
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
        log.debug("📅 Using negative cache for events %s", fid)
        return []
    if fid in EVENTS_CACHE and now - EVENTS_CACHE[fid][0] < 90:
        log.debug("📅 Using cache for events %s", fid)
        return EVENTS_CACHE[fid][1]
    js  = _api_get(f"{FOOTBALL_API_URL}/events", {"fixture": fid}) or {}
    out = js.get("response", []) if isinstance(js, dict) else []
    EVENTS_CACHE[fid] = (now, out)
    if not out:
        log.debug("📅 No events found for %s, caching negative", fid)
        NEG_CACHE[k] = (now, True)
    else:
        log.debug("📅 Found %s events for %s", len(out), fid)
    return out

def fetch_live_matches() -> List[dict]:
    log.info("🚀 Starting live matches fetch...")
    fixtures = fetch_live_fixtures_only()
    if not fixtures:
        log.info("❌ No live fixtures found")
        return []
        
    log.info("📊 Sorting %s fixtures by priority...", len(fixtures))
    fixtures.sort(key=_priority_key, reverse=True)
    quota  = _quota_per_scan()
    chosen = fixtures[:quota]
    log.info("🎯 Selected %s fixtures from %s total (quota: %s)", len(chosen), len(fixtures), quota)
    
    out    = []
    for i, m in enumerate(chosen):
        fid           = int((m.get("fixture", {}) or {}).get("id") or 0)
        log.debug("🔍 Processing fixture %s (%s/%s)", fid, i+1, len(chosen))
        m["statistics"] = fetch_match_stats(fid)
        m["events"]     = fetch_match_events(fid) if USE_EVENTS_IN_FEATURES else []
        out.append(m)
        
    ppf = 1 + (1 if USE_EVENTS_IN_FEATURES else 0)
    log.info("✅ Completed live matches fetch: %s fixtures, ppf=%s", len(out), ppf)
    return out

# ───────── Advanced Feature Extraction ─────────
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
    log.debug("🔧 Extracting features for match...")
    home   = m["teams"]["home"]["name"]
    away   = m["teams"]["away"]["name"]
    gh     = m["goals"]["home"] or 0
    ga     = m["goals"]["away"] or 0
    minute = int(((m.get("fixture") or {}).get("status") or {}).get("elapsed") or 0)

    stats: Dict[str, Dict[str, Any]] = {}
    for s in (m.get("statistics") or []):
        t = (s.get("team") or {}).get("name")
        if t:
            stats[t] = {(i.get("type") or ""): i.get("value") for i in (s.get("statistics") or [])}

    sh = stats.get(home, {}) or {}
    sa = stats.get(away, {}) or {}

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

    red_h = red_a = yellow_h = yellow_a = 0
    for ev in (m.get("events") or []):
        if (ev.get("type", "") or "").lower() == "card":
            d = (ev.get("detail", "") or "").lower()
            t = (ev.get("team") or {}).get("name") or ""
            if "yellow" in d and "second" not in d:
                if t == home:
                    yellow_h += 1
                elif t == away:
                    yellow_a += 1
            if "red" in d or "second yellow" in d:
                if t == home:
                    red_h += 1
                elif t == away:
                    red_a += 1

    momentum_h      = (sot_h + cor_h) / max(1, minute)
    momentum_a      = (sot_a + cor_a) / max(1, minute)
    pressure_index  = abs(gh - ga) * (minute / 90.0)
    efficiency_h    = gh / max(1, sot_h) if sot_h > 0 else 0.0
    efficiency_a    = ga / max(1, sot_a) if sot_a > 0 else 0.0
    total_actions   = sot_h + sot_a + cor_h + cor_a
    action_intensity= total_actions / max(1, minute)

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
        "red_h": float(red_h),
        "red_a": float(red_a),
        "red_sum": float(red_h + red_a),
        "yellow_h": float(yellow_h),
        "yellow_a": float(yellow_a),
        # Advanced features
        "momentum_h": float(momentum_h),
        "momentum_a": float(momentum_a),
        "pressure_index": float(pressure_index),
        "efficiency_h": float(efficiency_h),
        "efficiency_a": float(efficiency_a),
        "total_actions": float(total_actions),
        "action_intensity": float(action_intensity),
    }
    
    log.debug("✅ Extracted %s features for %s vs %s (minute: %s)", len(features), home, away, minute)
    return features

# ───────── Advanced Prediction System ─────────
def advanced_predict_probability(
    features: Dict[str, float],
    market: str,
    suggestion: str,
    match_id: Optional[int] = None,
) -> float:
    """Advanced prediction using ensemble + Bayesian methods"""
    log.debug("🤖 Predicting probability for %s - %s (match: %s)", market, suggestion, match_id)
    
    # Bayesian network: treat `suggestion` as the 'market phrase'
    bayesian_prob = bayesian_network.infer_probability(features, suggestion)
    log.debug("📊 Bayesian probability: %.3f", bayesian_prob)
    
    # Ensemble predictor
    model_key = f"{market}_{suggestion.replace(' ', '_')}"
    ensemble_predictor = get_advanced_predictor(model_key)
    ensemble_prob = ensemble_predictor.predict_probability(features)
    log.debug("📈 Ensemble probability: %.3f", ensemble_prob)
    
    data_richness = min(1.0, float(features.get("minute", 0)) / 60.0)
    if data_richness > 0.7:
        final_prob = 0.7 * ensemble_prob + 0.3 * bayesian_prob
        log.debug("📋 Using data-rich weighting (70/30)")
    else:
        final_prob = 0.4 * ensemble_prob + 0.6 * bayesian_prob
        log.debug("📋 Using data-poor weighting (40/60)")
    
    final_prob = float(max(0.01, min(0.99, final_prob)))
    log.info("🎯 FINAL probability for %s - %s: %.1f%% (bayesian: %.1f%%, ensemble: %.1f%%)", 
             market, suggestion, final_prob*100, bayesian_prob*100, ensemble_prob*100)
    
    prediction_data = {
        "features": features,
        "market": market,
        "suggestion": suggestion,
        "bayesian_prob": bayesian_prob,
        "ensemble_prob": ensemble_prob,
        "final_prob": final_prob,
        "model_type": "advanced_ensemble",
        "model_key": model_key,
        "match_id": int(match_id or 0),
        "timestamp": time.time(),
    }
    
    # Store for later learning
    try:
        with db_conn() as c:
            c.execute(
                "INSERT INTO self_learning_data (match_id, market, features, prediction_probability, learning_timestamp) "
                "VALUES (%s,%s,%s,%s,%s)",
                (
                    int(match_id or 0),
                    market,
                    json.dumps(prediction_data, separators=(",", ":"), ensure_ascii=False),
                    float(final_prob),
                    int(time.time()),
                ),
            )
        log.debug("💾 Saved prediction data for self-learning")
    except Exception as e:
        log.debug("[SELF-LEARNING] store failed: %s", e)
    
    return final_prob

# ───────── Self-Learning from Results ─────────
def process_self_learning_from_results() -> None:
    """Process completed games to learn from prediction outcomes"""
    if not SELF_LEARNING_ENABLE:
        log.info("🤖 Self-learning disabled")
        return
        
    cutoff_ts = int(time.time()) - 24 * 3600
    with db_conn() as c:
        rows = c.execute(
            """
            SELECT sl.id, sl.match_id, sl.market, sl.features, sl.prediction_probability,
                   mr.final_goals_h, mr.final_goals_a, mr.btts_yes
            FROM self_learning_data sl
            JOIN match_results mr ON sl.match_id = mr.match_id
            WHERE sl.actual_outcome IS NULL
              AND sl.learning_timestamp >= %s
            LIMIT %s
        """,
            (cutoff_ts, SELF_LEARN_BATCH_SIZE),
        ).fetchall()
    
    log.info("🤖 Processing %s self-learning records", len(rows))
    processed = 0
    for sl_id, match_id, market, features_json, pred_prob, gh, ga, btts in rows:
        try:
            meta = json.loads(features_json)
        except Exception:
            continue
        
        suggestion = meta.get("suggestion")
        if not suggestion:
            continue
        
        result = {"final_goals_h": gh, "final_goals_a": ga, "btts_yes": btts}
        actual_outcome = _tip_outcome_for_result(suggestion, result)
        if actual_outcome is None:
            continue
        
        learning_data = {
            "features": meta.get("features", {}),
            "market": market,
            "predicted_outcome": 1 if float(pred_prob or 0.0) > 0.5 else 0,
            "prediction_probability": float(pred_prob or 0.0),
            "model_type": meta.get("model_type", "unknown"),
        }
        
        self_learning_system.record_prediction_outcome(learning_data, int(actual_outcome))
        
        # Update the corresponding advanced predictor's performance if we know the key
        model_key = meta.get("model_key")
        if model_key:
            try:
                get_advanced_predictor(model_key).update_performance(int(actual_outcome))
            except Exception:
                pass
        
        with db_conn() as c2:
            c2.execute("UPDATE self_learning_data SET actual_outcome=%s WHERE id=%s", (int(actual_outcome), int(sl_id)))
        processed += 1
    
    if processed:
        log.info("[SELF-LEARNING] Processed %d results for learning", processed)

# ───────── Odds fetching and processing ─────────
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
        log.debug("💰 Using cached odds for %s", fid)
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
            log.debug("💰 Insufficient books for market %s", mkey)
            continue

        out[mkey] = {}
        for side, lst in side_map.items():
            ag, label = _aggregate_price(lst, None)
            if ag is not None:
                out[mkey][side] = {"odds": float(ag), "book": label}
                log.debug("💰 Market %s %s: odds=%.2f, book=%s", mkey, side, ag, label)

    ODDS_CACHE[fid] = (now, out)
    log.debug("💰 Found odds for %s markets for fixture %s", len(out), fid)
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
    log.debug("🔍 Getting odds for %s - %s", market, suggestion)
    if market == "BTTS":
        d   = odds_map.get("BTTS", {})
        tgt = "Yes" if suggestion.endswith("Yes") else "No"
        if tgt in d:
            odds, book = d[tgt]["odds"], d[tgt]["book"]
            log.debug("✅ Found BTTS odds: %.2f from %s", odds, book)
            return odds, book
        else:
            log.debug("❌ No BTTS odds found for %s", tgt)
    elif market == "1X2":
        d   = odds_map.get("1X2", {})
        tgt = "Home" if suggestion == "Home Win" else ("Away" if suggestion == "Away Win" else None)
        if tgt and tgt in d:
            odds, book = d[tgt]["odds"], d[tgt]["book"]
            log.debug("✅ Found 1X2 odds: %.2f from %s", odds, book)
            return odds, book
        else:
            log.debug("❌ No 1X2 odds found for %s", tgt)
    elif market.startswith("Over/Under"):
        ln_val = _parse_ou_line_from_suggestion(suggestion)
        d      = odds_map.get(f"OU_{_fmt_line(ln_val)}", {}) if ln_val is not None else {}
        tgt    = "Over" if suggestion.startswith("Over") else "Under"
        if tgt in d:
            odds, book = d[tgt]["odds"], d[tgt]["book"]
            log.debug("✅ Found OU odds: %.2f from %s", odds, book)
            return odds, book
        else:
            log.debug("❌ No OU odds found for %s %s", tgt, ln_val)
    else:
        log.debug("❌ Unknown market type: %s", market)
    return None, None

# ───────── Core prediction logic with advanced systems ─────────
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
            threshold = float(v)
            log.info("⚙️ Using CUSTOM threshold for %s: %.1f%%", m, threshold)
            return threshold
        else:
            # Use the global CONF_THRESHOLD as default for all markets
            threshold = float(CONF_THRESHOLD)
            log.info("⚙️ Using GLOBAL threshold for %s: %.1f%%", m, threshold)
            return threshold
    except Exception as e:
        log.error("❌ Error getting threshold for %s: %s, using default: %.1f%%", m, e, float(CONF_THRESHOLD))
        return float(CONF_THRESHOLD)

def reset_market_thresholds_to_global() -> Dict[str, float]:
    """Reset all market thresholds to use the global CONF_THRESHOLD"""
    log.info("🔄 Resetting all market thresholds to global CONF_THRESHOLD: %.1f%%", CONF_THRESHOLD)
    
    markets = ["BTTS", "1X2"]
    for line in OU_LINES:
        markets.append(f"Over/Under {_fmt_line(line)}")
    
    reset_results = {}
    for market in markets:
        try:
            # Delete any custom threshold settings to fall back to global
            with db_conn() as c:
                c.execute("DELETE FROM settings WHERE key = %s", (f"conf_threshold:{market}",))
            reset_results[market] = CONF_THRESHOLD
            log.info("✅ Reset %s threshold to global: %.1f%%", market, CONF_THRESHOLD)
        except Exception as e:
            log.error("❌ Failed to reset %s threshold: %s", market, e)
            reset_results[market] = None
    
    return reset_results

def _parse_ou_line_from_suggestion(s: str) -> Optional[float]:
    try:
        for tok in (s or "").split():
            try:
                return float(tok)
            except Exception:
                continue
    except Exception:
        pass
    return None

def _candidate_is_sane(sug: str, feat: Dict[str, float]) -> bool:
    gh    = int(feat.get("goals_h", 0))
    ga    = int(feat.get("goals_a", 0))
    total = gh + ga
    minute = int(feat.get("minute", 0))

    log.debug("🔍 Sanity check for %s: score %s-%s (total: %s), minute: %s", sug, gh, ga, total, minute)

    if sug.startswith("Over"):
        ln = _parse_ou_line_from_suggestion(sug)
        if ln is None:
            log.debug("❌ Invalid Over line in suggestion: %s", sug)
            return False
        sane = (ln is not None) and (total < ln)
        if not sane:
            log.debug("❌ Insane Over suggestion: %s (total: %s, line: %s)", sug, total, ln)
        else:
            log.debug("✅ Sane Over suggestion: %s (total: %s, line: %s)", sug, total, ln)
        return sane

    if sug.startswith("Under"):
        ln = _parse_ou_line_from_suggestion(sug)
        if ln is None:
            log.debug("❌ Invalid Under line in suggestion: %s", sug)
            return False
        sane = (ln is not None) and (total < ln)
        if not sane:
            log.debug("❌ Insane Under suggestion: %s (total: %s, line: %s)", sug, total, ln)
        else:
            log.debug("✅ Sane Under suggestion: %s (total: %s, line: %s)", sug, total, ln)
        return sane

    if sug.startswith("BTTS") and (gh > 0 and ga > 0):
        log.debug("❌ Insane BTTS suggestion: %s (both already scored)", sug)
        return False

    if sug == "BTTS: Yes" and (gh > 0 and ga > 0):
        log.debug("✅ Sane BTTS Yes: both teams scored")
        return True
    if sug == "BTTS: No" and not (gh > 0 and ga > 0):
        log.debug("✅ Sane BTTS No: both teams haven't scored")
        return True

    log.debug("✅ Sane suggestion: %s", sug)
    return True

def _format_tip_message(
    home: str,
    away: str,
    league: str,
    minute: int,
    score: str,
    suggestion: str,
    prob_pct: float,
    feat: Dict[str, float],
    odds: Optional[float] = None,
    book: Optional[str] = None,
    ev_pct: Optional[float] = None,
) -> str:
    stat = ""
    if any(
        [
            feat.get("xg_h", 0),
            feat.get("xg_a", 0),
            feat.get("sot_h", 0),
            feat.get("sot_a", 0),
            feat.get("cor_h", 0),
            feat.get("cor_a", 0),
            feat.get("pos_h", 0),
            feat.get("pos_a", 0),
            feat.get("red_h", 0),
            feat.get("red_a", 0),
        ]
    ):
        stat = (
            f"\n📊 xG {feat.get('xg_h',0):.2f}-{feat.get('xg_a',0):.2f}"
            f" • SOT {int(feat.get('sot_h',0))}-{int(feat.get('sot_a',0))}"
            f" • CK {int(feat.get('cor_h',0))}-{int(feat.get('cor_a',0))}"
        )
        if feat.get("pos_h", 0) or feat.get("pos_a", 0):
            stat += f" • POS {int(feat.get('pos_h',0))}%–{int(feat.get('pos_a',0))}%"
        if feat.get("red_h", 0) or feat.get("red_a", 0):
            stat += f" • RED {int(feat.get('red_h',0))}-{int(feat.get('red_a',0))}"
    money = ""
    if odds:
        if ev_pct is not None:
            money = f"\n💰 <b>Odds:</b> {odds:.2f} @ {book or 'Book'}  •  <b>EV:</b> {ev_pct:+.1f}%"
        else:
            money = f"\n💰 <b>Odds:</b> {odds:.2f} @ {book or 'Book'}"
    return (
        "⚽️ <b>New Tip!</b>\n"
        f"<b>Match:</b> {escape(home)} vs {escape(away)}\n"
        f"🕒 <b>Minute:</b> {minute}'  |  <b>Score:</b> {escape(score)}\n"
        f"<b>Tip:</b> {escape(suggestion)}\n"
        f"📈 <b>Confidence:</b> {prob_pct:.1f}%{money}\n"
        f"🏆 <b>League:</b> {escape(league)}{stat}"
    )

def _save_and_send_tips(
    ranked: List[Tuple],
    fid: int,
    league_id: int,
    league: str,
    home: str,
    away: str,
    score: str,
    minute: int,
    feat: Dict[str, float],
    per_league_counter: Dict[int, int],
    c: PooledConn,
) -> int:
    log.info("💾 Starting to save and send tips for %s vs %s (%s tips ranked)", home, away, len(ranked))
    saved   = 0
    base_now= int(time.time())

    for idx, (market, suggestion, prob, odds, book, ev_pct, _) in enumerate(ranked):
        log.debug("💾 Processing tip %s/%s: %s - %s (prob: %.3f)", 
                 idx+1, len(ranked), market, suggestion, prob)
                 
        if PER_LEAGUE_CAP > 0 and per_league_counter.get(league_id, 0) >= PER_LEAGUE_CAP:
            log.debug("⏩ League cap reached for league %s (%s tips)", league_id, per_league_counter.get(league_id, 0))
            continue

        created_ts = base_now + idx
        prob_pct   = round(prob * 100.0, 1)
        log.debug("📊 Tip details: prob=%.1f%%, odds=%s, ev=%s", prob_pct, odds, ev_pct)

        try:
            log.info("💾 INSERTING tip into database: %s vs %s - %s (%.1f%%)", 
                    home, away, suggestion, prob_pct)
                    
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
                    0,  # sent_ok=0 initially
                ),
            )
            log.info("✅ SUCCESSFULLY inserted tip into database")

            # Count this as saved regardless of Telegram success
            saved += 1
            log.info("📈 Incremented saved counter to %s", saved)
            per_league_counter[league_id] = per_league_counter.get(league_id, 0) + 1
            log.debug("🏆 League %s counter: %s", league_id, per_league_counter[league_id])

            message = _format_tip_message(
                home, away, league, minute, score, suggestion, float(prob_pct), feat, odds, book, ev_pct
            )
            log.info("📤 Attempting to send Telegram message...")
            sent = send_telegram(message)
            if sent:
                log.info("✅ Telegram sent successfully, updating sent_ok to 1")
                c.execute("UPDATE tips SET sent_ok=1 WHERE match_id=%s AND created_ts=%s", (fid, created_ts))
                _metric_inc("tips_sent_total", n=1)
            else:
                log.error("❌ Failed to send Telegram message, but tip was saved to DB")

            if MAX_TIPS_PER_SCAN and saved >= MAX_TIPS_PER_SCAN:
                log.info("🎯 Reached MAX_TIPS_PER_SCAN limit (%s)", MAX_TIPS_PER_SCAN)
                break
            if saved >= max(1, PREDICTIONS_PER_MATCH):
                log.info("🎯 Reached PREDICTIONS_PER_MATCH limit (%s)", PREDICTIONS_PER_MATCH)
                break

        except Exception as e:
            log.exception("❌ [PROD] insert/send failed: %s", e)
            continue

    log.info("🎉 Finished saving tips: %s tips saved for %s vs %s", saved, home, away)
    return saved

def _process_and_rank_candidates(
    candidates: List[Tuple[str, str, float]],
    fid: int,
    features: Dict[str, float],
) -> List[Tuple[str, str, float, Optional[float], Optional[str], Optional[float], float]]:
    log.info("🏆 Processing and ranking %s candidates for fixture %s", len(candidates), fid)
    ranked: List[Tuple[str, str, float, Optional[float], Optional[str], Optional[float], float]] = []
    odds_map = fetch_odds(fid) if API_KEY else {}
    log.debug("💰 Odds map has %s markets", len(odds_map))

    for market, suggestion, prob in candidates:
        log.debug("🔍 Processing candidate: %s - %s (prob: %.3f)", market, suggestion, prob)
        
        if suggestion not in ALLOWED_SUGGESTIONS:
            log.debug("❌ Suggestion not in allowed list: %s", suggestion)
            continue

        odds, book = _get_odds_for_market(odds_map, market, suggestion)
        if odds is None and not ALLOW_TIPS_WITHOUT_ODDS:
            log.debug("❌ No odds found and ALLOW_TIPS_WITHOUT_ODDS=False")
            continue

        if odds is not None:
            min_odds = _min_odds_for_market(market)
            if not (min_odds <= odds <= MAX_ODDS_ALL):
                log.debug("❌ Odds outside range: %.2f (min: %.2f, max: %.2f)", odds, min_odds, MAX_ODDS_ALL)
                continue

            edge   = _ev(prob, odds)
            ev_pct = round(edge * 100.0, 1)
            if int(round(edge * 10000)) < EDGE_MIN_BPS:
                log.debug("❌ EV too low: %.1f%% < %s bps", ev_pct, EDGE_MIN_BPS)
                continue
            log.debug("✅ Good EV: %.1f%%", ev_pct)
        else:
            ev_pct = None
            log.debug("ℹ️ No odds available, skipping EV check")

        rank_score = (prob ** 1.2) * (1.0 + (ev_pct or 0.0) / 100.0)
        log.debug("📊 Rank score: %.3f", rank_score)
        ranked.append((market, suggestion, prob, odds, book, ev_pct, rank_score))

    ranked.sort(key=lambda x: x[6], reverse=True)
    log.info("🎯 Ranked %s candidates out of %s", len(ranked), len(candidates))
    for i, (mkt, sug, prob, odds, _, ev, score) in enumerate(ranked[:3]):  # Log top 3
        log.debug("🏅 Rank %s: %s - %s (prob: %.3f, odds: %s, ev: %s, score: %.3f)", 
                 i+1, mkt, sug, prob, odds, ev, score)
    return ranked

def _generate_advanced_predictions(
    features: Dict[str, float],
    fid: int,
    minute: int,
) -> List[Tuple[str, str, float]]:
    log.info("🤖 Generating advanced predictions for fixture %s (minute: %s)", fid, minute)
    candidates: List[Tuple[str, str, float]] = []

    # Log all features for debugging
    log.info("📊 Feature summary - minute: %s, goals: %s-%s, xG: %.2f-%.2f, SOT: %s-%s", 
             features.get('minute'), 
             features.get('goals_h'), features.get('goals_a'),
             features.get('xg_h', 0), features.get('xg_a', 0),
             features.get('sot_h', 0), features.get('sot_a', 0))

    # OU markets
    for line in OU_LINES:
        sline  = _fmt_line(line)
        market = f"Over/Under {sline}"
        threshold = _get_market_threshold(market)
        
        log.info("🎯 Testing OU market: %s (threshold: %.1f%%)", market, threshold)

        over_sug = f"Over {sline} Goals"
        over_prob = advanced_predict_probability(features, market, over_sug, match_id=fid)
        log.info("📈 Over %s probability: %.1f%%", sline, over_prob * 100)
        
        if over_prob * 100.0 >= threshold:
            if _candidate_is_sane(over_sug, features):
                log.info("✅ Adding Over candidate: %s (prob: %.1f%%)", over_sug, over_prob*100)
                candidates.append((market, over_sug, over_prob))
            else:
                log.info("❌ Over candidate failed sanity check: %s", over_sug)
        else:
            log.info("❌ Over probability below threshold: %.1f%% < %.1f%%", over_prob*100, threshold)

        under_sug  = f"Under {sline} Goals"
        under_prob = 1.0 - over_prob
        log.info("📈 Under %s probability: %.1f%%", sline, under_prob * 100)
        
        if under_prob * 100.0 >= threshold:
            if _candidate_is_sane(under_sug, features):
                log.info("✅ Adding Under candidate: %s (prob: %.1f%%)", under_sug, under_prob*100)
                candidates.append((market, under_sug, under_prob))
            else:
                log.info("❌ Under candidate failed sanity check: %s", under_sug)
        else:
            log.info("❌ Under probability below threshold: %.1f%% < %.1f%%", under_prob*100, threshold)

    # BTTS market
    market = "BTTS"
    threshold = _get_market_threshold(market)
    log.info("🎯 Testing BTTS market (threshold: %.1f%%)", threshold)
    
    btts_yes_prob = advanced_predict_probability(features, market, "BTTS: Yes", match_id=fid)
    log.info("📈 BTTS Yes probability: %.1f%%", btts_yes_prob * 100)
    
    if btts_yes_prob * 100.0 >= threshold:
        if _candidate_is_sane("BTTS: Yes", features):
            log.info("✅ Adding BTTS Yes candidate (prob: %.1f%%)", btts_yes_prob*100)
            candidates.append((market, "BTTS: Yes", btts_yes_prob))
        else:
            log.info("❌ BTTS Yes candidate failed sanity check")
    else:
        log.info("❌ BTTS Yes probability below threshold: %.1f%% < %.1f%%", btts_yes_prob*100, threshold)

    btts_no_prob = 1.0 - btts_yes_prob
    log.info("📈 BTTS No probability: %.1f%%", btts_no_prob * 100)
    
    if btts_no_prob * 100.0 >= threshold:
        if _candidate_is_sane("BTTS: No", features):
            log.info("✅ Adding BTTS No candidate (prob: %.1f%%)", btts_no_prob*100)
            candidates.append((market, "BTTS: No", btts_no_prob))
        else:
            log.info("❌ BTTS No candidate failed sanity check")
    else:
        log.info("❌ BTTS No probability below threshold: %.1f%% < %.1f%%", btts_no_prob*100, threshold)

    # 1X2 market
    market = "1X2"
    threshold = _get_market_threshold(market)
    log.info("🎯 Testing 1X2 market (threshold: %.1f%%)", threshold)
    
    home_win_prob = advanced_predict_probability(features, market, "Home Win", match_id=fid)
    away_win_prob = advanced_predict_probability(features, market, "Away Win", match_id=fid)
    
    log.info("📈 Home Win probability: %.1f%%", home_win_prob * 100)
    log.info("📈 Away Win probability: %.1f%%", away_win_prob * 100)

    total_win_prob = home_win_prob + away_win_prob
    if total_win_prob > 0:
        home_win_prob = home_win_prob / total_win_prob
        away_win_prob = away_win_prob / total_win_prob
        
        log.info("📊 Normalized probabilities - Home: %.1f%%, Away: %.1f%%", 
                 home_win_prob*100, away_win_prob*100)

        if home_win_prob * 100.0 >= threshold:
            log.info("✅ Adding Home Win candidate (prob: %.1f%%)", home_win_prob*100)
            candidates.append((market, "Home Win", home_win_prob))
        else:
            log.info("❌ Home Win probability below threshold: %.1f%% < %.1f%%", home_win_prob*100, threshold)
            
        if away_win_prob * 100.0 >= threshold:
            log.info("✅ Adding Away Win candidate (prob: %.1f%%)", away_win_prob*100)
            candidates.append((market, "Away Win", away_win_prob))
        else:
            log.info("❌ Away Win probability below threshold: %.1f%% < %.1f%%", away_win_prob*100, threshold)
    else:
        log.info("❌ No valid 1X2 probabilities (total: %.3f)", total_win_prob)

    log.info("🎯 Generated %s total prediction candidates", len(candidates))
    
    # Log what the confidence threshold actually is
    log.info("⚙️ Current CONF_THRESHOLD setting: %s", CONF_THRESHOLD)
    
    return candidates

def production_scan() -> Tuple[int, int]:
    """Main in-play scanning with advanced AI systems"""
    log.info("🚀 STARTING PRODUCTION SCAN")
    
    # Log threshold configuration at the start
    log.info("⚙️ GLOBAL CONF_THRESHOLD: %.1f%%", CONF_THRESHOLD)
    log.info("⚙️ Checking market thresholds...")
    
    # Test what thresholds are being used for each market
    test_markets = ["BTTS", "1X2"]
    for line in OU_LINES:
        test_markets.append(f"Over/Under {_fmt_line(line)}")
    
    for market in test_markets:
        threshold = _get_market_threshold(market)
        log.info("⚙️ Market %s will use threshold: %.1f%%", market, threshold)
    
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
                    log.debug("⏩ Skipping match without fixture ID")
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

                feat   = extract_features(m)
                minute = int(feat.get("minute", 0))
                log.info("⏱️ Match minute: %s, TIP_MIN_MINUTE: %s", minute, TIP_MIN_MINUTE)
                
                if minute < TIP_MIN_MINUTE:
                    log.info("⏩ Skipping match %s - minute %s < TIP_MIN_MINUTE %s", fid, minute, TIP_MIN_MINUTE)
                    continue
                    
                if is_feed_stale(fid, m, minute):
                    log.info("⏩ Skipping match %s - stale feed", fid)
                    continue

                if HARVEST_MODE and minute >= TRAIN_MIN_MINUTE and minute % 3 == 0:
                    try:
                        log.debug("🌱 Saving snapshot for training")
                        save_snapshot_from_match(m, feat)
                    except Exception as e:
                        log.debug("❌ Snapshot save failed: %s", e)

                league_id, league = _league_name(m)
                home, away        = _teams(m)
                score             = _pretty_score(m)
                
                log.info("🏠 %s vs %s (%s) - %s minute %s", home, away, league, score, minute)

                candidates = _generate_advanced_predictions(feat, fid, minute)
                if not candidates:
                    log.info("⏩ No prediction candidates for match %s", fid)
                    continue

                ranked = _process_and_rank_candidates(candidates, fid, feat)
                if not ranked:
                    log.info("⏩ No ranked candidates for match %s", fid)
                    continue

                match_saved = _save_and_send_tips(
                    ranked,
                    fid,
                    league_id,
                    league,
                    home,
                    away,
                    score,
                    minute,
                    feat,
                    per_league_counter,
                    c,
                )
                saved += match_saved
                log.info("💾 Match %s: saved %s tips (total saved: %s)", fid, match_saved, saved)

                if MAX_TIPS_PER_SCAN and saved >= MAX_TIPS_PER_SCAN:
                    log.info("🎯 Reached MAX_TIPS_PER_SCAN limit (%s)", MAX_TIPS_PER_SCAN)
                    break

            except Exception as e:
                log.exception("❌ [PROD] match loop failed for match %s: %s", fid, e)
                continue

    log.info("[PROD] saved=%d live_seen=%d", saved, live_seen)
    _metric_inc("tips_generated_total", n=saved)
    return saved, live_seen

# ───────── Stale feed guard ─────────
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
    home  = (teams.get("home") or {}).get("name", "")
    away  = (teams.get("away") or {}).get("name", "")

    stats_by_team: Dict[str, Dict[str, Any]] = {}
    for s in (m.get("statistics") or []):
        tname = ((s.get("team") or {}).get("name") or "").strip()
        if tname:
            stats_by_team[tname] = {
                str((i.get("type") or "")).lower(): i.get("value") for i in (s.get("statistics") or [])
            }

    sh = stats_by_team.get(home, {}) or {}
    sa = stats_by_team.get(away, {}) or {}

    def g(d: dict, key_variants: Tuple[str, ...]) -> float:
        for k in key_variants:
            if k in d:
                return _safe_num(d[k])
        return 0.0

    xg_h    = g(sh, ("expected goals",))
    xg_a    = g(sa, ("expected goals",))
    sot_h   = g(sh, ("shots on target", "shots on goal"))
    sot_a   = g(sa, ("shots on target", "shots on goal"))
    sh_tot_h= g(sh, ("total shots", "shots total"))
    sh_tot_a= g(sa, ("total shots", "shots total"))
    cor_h   = g(sh, ("corner kicks",))
    cor_a   = g(sa, ("corner kicks",))
    pos_h   = g(sh, ("ball possession",))
    pos_a   = g(sa, ("ball possession",))

    ev       = m.get("events") or []
    n_events = len(ev)
    n_cards  = 0
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
        int(round(pos_h)),
        int(round(pos_a)),
        gh,
        ga,
        n_events,
        n_cards,
    )

def is_feed_stale(fid: int, m: dict, minute: int) -> bool:
    if not STALE_GUARD_ENABLE:
        return False
    if minute < 10:
        fp = _match_fingerprint(m)
        _FEED_STATE[fid] = {"fp": fp, "last_change": time.time(), "last_minute": minute}
        return False

    now = time.time()
    fp  = _match_fingerprint(m)
    st  = _FEED_STATE.get(fid)

    if st is None:
        _FEED_STATE[fid] = {"fp": fp, "last_change": now, "last_minute": minute}
        return False

    if fp != st.get("fp"):
        st["fp"]          = fp
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
    payload = json.dumps(snapshot, separators=(",", ":"), ensure_ascii=False)[:200000]

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

# ───────── Results processing ─────────
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

def _fixture_by_id(mid: int) -> Optional[dict]:
    js  = _api_get(FOOTBALL_API_URL, {"id": mid}) or {}
    arr = js.get("response") or [] if isinstance(js, dict) else []
    return arr[0] if arr else None

def _is_final(short: str) -> bool:
    return (short or "").upper() in {"FT", "AET", "PEN"}

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
        
    if updated > 0 and SELF_LEARNING_ENABLE:
        process_self_learning_from_results()
        
    if updated:
        log.info("[RESULTS] backfilled %d", updated)
    return updated

def daily_accuracy_digest(window_days: int = 7) -> Optional[str]:
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
    roi_by_market: Dict[str, Dict[str, float]] = {}
    by_market: Dict[str, Dict[str, int]]       = {}

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

        if odds:
            roi_by_market.setdefault(mkt, {"stake": 0.0, "pnl": 0.0})
            roi_by_market[mkt]["stake"] += 1.0
            if out == 1:
                roi_by_market[mkt]["pnl"] += float(odds) - 1.0
            else:
                roi_by_market[mkt]["pnl"] -= 1.0

    if graded == 0:
        msg = "📊 Accuracy Digest\nNo graded tips in window."
    else:
        acc = 100.0 * wins / max(1, graded)
        lines = [
            f"📊 <b>Accuracy Digest</b> (last {window_days}d)",
            f"Tips sent: {total}  •  Graded: {graded}  •  Wins: {wins}  •  Accuracy: {acc:.1f}%",
        ]

        feature_importance = self_learning_system.get_feature_importance()
        if feature_importance:
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
            lines.append(
                "🔍 <b>Top Predictive Features:</b> "
                + ", ".join([f"{k}({v:.1%})" for k, v in top_features])
            )

        for mk, st in sorted(by_market.items()):
            if st["graded"] == 0:
                continue
            a = 100.0 * st["wins"] / st["graded"]
            roi = ""
            if mk in roi_by_market and roi_by_market[mk]["stake"] > 0:
                roi_val = 100.0 * roi_by_market[mk]["pnl"] / roi_by_market[mk]["stake"]
                roi = f" • ROI {roi_val:+.1f}%"
            lines.append(f"• {escape(mk)} — {st['wins']}/{st['graded']} ({a:.1f}%){roi}")

        msg = "\n".join(lines)

    send_telegram(msg)
    return msg

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

def auto_train_job():
    if not TRAIN_ENABLE:
        return send_telegram("🤖 Training skipped: TRAIN_ENABLE=0")
    send_telegram("🤖 Advanced training started.")
    try:
        res = train_models() or {}
        ok  = bool(res.get("ok"))
        if not ok:
            reason = res.get("reason") or res.get("error") or "unknown"
            return send_telegram(f"⚠️ Training finished: <b>SKIPPED</b>\nReason: {escape(str(reason))}")

        trained = [k for k, v in (res.get("trained") or {}).items() if v]
        lines   = ["🤖 <b>Advanced Model Training OK</b>"]
        if trained:
            lines.append("• Trained: " + ", ".join(sorted(trained)))
        lines.append("• Features: Bayesian networks + Ensemble methods")
        lines.append("• Learning: Self-correcting from bet outcomes")
        send_telegram("\n".join(lines))
    except Exception as e:
        log.exception("[TRAIN] job failed: %s", e)
        send_telegram(f"❌ Training <b>FAILED</b>\n{escape(str(e))}")

def auto_tune_thresholds(days: int = 14) -> Dict[str, float]:
    """Auto-tune thresholds using the advanced tuner from train_models.py if available."""
    tuned: Dict[str, float] = {}
    if not AUTO_TUNE_ENABLE:
        return tuned
    if not _TRAIN_MODULE_OK:
        send_telegram("🔧 Auto-tune skipped: trainer module not available")
        return tuned

    try:
        # Reuse train_models.py connection logic for consistency
        conn = _tm._connect(os.getenv("DATABASE_URL"))
        try:
            tuned = _tm.auto_tune_thresholds_advanced(conn, days)
        finally:
            try:
                conn.close()
            except Exception:
                pass
        send_telegram(f"🔧 Auto-tune completed: tuned {len(tuned)} markets")
        return tuned
    except Exception as e:
        log.exception("[AUTO-TUNE] failed: %s", e)
        send_telegram(f"❌ Auto-tune FAILED\n{escape(str(e))}")
        return {}

def retry_unsent_tips(minutes: int = 30, limit: int = 200) -> int:
    cutoff  = int(time.time()) - minutes * 60
    retried = 0
    with db_conn() as c:
        rows = c.execute(
            "SELECT match_id,league,home,away,market,suggestion,confidence,confidence_raw,score_at_tip,"
            "minute,created_ts,odds,book,ev_pct "
            "FROM tips WHERE sent_ok=0 AND created_ts >= %s ORDER BY created_ts ASC LIMIT %s",
            (cutoff, limit),
        ).fetchall()

        for (
            mid,
            league,
            home,
            away,
            market,
            sugg,
            conf,
            conf_raw,
            score,
            minute,
            cts,
            odds,
            book,
            ev_pct,
        ) in rows:
            ok = send_telegram(
                _format_tip_message(
                    home,
                    away,
                    league,
                    int(minute),
                    score,
                    sugg,
                    float(conf),
                    {},
                    odds,
                    book,
                    ev_pct,
                )
            )
            if ok:
                c.execute("UPDATE tips SET sent_ok=1 WHERE match_id=%s AND created_ts=%s", (mid, cts))
                retried += 1
    if retried:
        log.info("[RETRY] resent %d", retried)
    return retried

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

        if SELF_LEARNING_ENABLE:
            sched.add_job(
                lambda: _run_with_pg_lock(1003, process_self_learning_from_results),
                "interval",
                minutes=30,
                id="self_learn",
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

        if TRAIN_ENABLE:
            sched.add_job(
                lambda: _run_with_pg_lock(1005, auto_train_job),
                CronTrigger(hour=TRAIN_HOUR_UTC, minute=TRAIN_MINUTE_UTC, timezone=TZ_UTC),
                id="train",
                max_instances=1,
                coalesce=True,
            )

        if AUTO_TUNE_ENABLE:
            sched.add_job(
                lambda: _run_with_pg_lock(1006, auto_tune_thresholds, 14),
                CronTrigger(hour=4, minute=7, timezone=TZ_UTC),
                id="auto_tune",
                max_instances=1,
                coalesce=True,
            )

        sched.add_job(
            lambda: _run_with_pg_lock(1007, retry_unsent_tips, 30, 200),
            "interval",
            minutes=10,
            id="retry",
            max_instances=1,
            coalesce=True,
        )

        sched.start()
        _scheduler_started = True
        send_telegram("🚀 goalsniper PURE IN-PLAY AI mode started with Bayesian networks & self-learning.")
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
    return jsonify({"ok": True, "name": "goalsniper", "mode": "PURE_INPLAY_AI", "scheduler": RUN_SCHEDULER})

@app.route("/health")
def health():
    try:
        with db_conn() as c:
            n = c.execute("SELECT COUNT(*) FROM tips").fetchone()[0]
        return jsonify({"ok": True, "db": "ok", "tips_count": int(n)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/metrics")
def metrics():
    try:
        return jsonify({"ok": True, "metrics": METRICS})
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

@app.route("/admin/self-learn", methods=["POST", "GET"])
def http_self_learn():
    _require_admin()
    process_self_learning_from_results()
    return jsonify({"ok": True})

@app.route("/admin/digest", methods=["POST","GET"])
def http_digest(): _require_admin(); msg=daily_accuracy_digest(); return jsonify({"ok": True, "sent": bool(msg)})

@app.route("/admin/reset-thresholds", methods=["POST", "GET"])
def http_reset_thresholds():
    _require_admin()
    results = reset_market_thresholds_to_global()
    return jsonify({"ok": True, "reset_thresholds": results})

@app.route("/admin/train", methods=["POST", "GET"])
def http_train():
    _require_admin()
    if not TRAIN_ENABLE:
        return jsonify({"ok": False, "reason": "training disabled"}), 400
    try:
        out = train_models()
        return jsonify({"ok": True, "result": out})
    except Exception as e:
        log.exception("train_models failed: %s", e)
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/admin/system-status")
def http_system_status():
    _require_admin()
    learning_status = verify_learning_system()
    return jsonify({"ok": True, "system_status": learning_status})

@app.route("/admin/auto-tune", methods=["POST", "GET"])
def http_auto_tune():
    _require_admin()
    tuned = auto_tune_thresholds(14)
    return jsonify({"ok": True, "tuned": tuned})

@app.route("/admin/retry-unsent", methods=["POST", "GET"])
def http_retry_unsent():
    _require_admin()
    n = retry_unsent_tips(30, 200)
    return jsonify({"ok": True, "resent": n})

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

# ───────── Boot ─────────
def _on_boot():
    _init_pool()
    init_db()
    set_setting("boot_ts", str(int(time.time())))
    _start_scheduler_once()

_on_boot()

if __name__ == "__main__":
    app.run(host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8080")))
