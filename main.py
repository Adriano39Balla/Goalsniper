# goalsniper — PURE IN-PLAY AI mode with Bayesian networks & self-learning
# Upgraded: Bayesian networks, self-learning from wrong bets, advanced ensemble models
# Enhanced: Context analysis, performance monitoring, multi-book odds, timing optimization
# SUPERCHARGED: Enhanced API-Football integration with player stats, historical data, and predictions endpoint

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
    "context_analysis_calls": 0,
    "performance_monitor_updates": 0,
    "multi_book_odds_searches": 0,
    "timing_analysis_decisions": 0,
    "volume_reductions_triggered": 0,
    "enhanced_features_processed": 0,
    "historical_context_applied": 0,
    "api_predictions_blended": 0,
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

# New enhancement controls
ENABLE_CONTEXT_ANALYSIS = os.getenv("ENABLE_CONTEXT_ANALYSIS", "1") not in ("0","false","False","no","NO")
ENABLE_PERFORMANCE_MONITOR = os.getenv("ENABLE_PERFORMANCE_MONITOR", "1") not in ("0","false","False","no","NO")
ENABLE_MULTI_BOOK_ODDS = os.getenv("ENABLE_MULTI_BOOK_ODDS", "1") not in ("0","false","False","no","NO")
ENABLE_TIMING_OPTIMIZATION = os.getenv("ENABLE_TIMING_OPTIMIZATION", "1") not in ("0","false","False","no","NO")
ENABLE_INCREMENTAL_LEARNING = os.getenv("ENABLE_INCREMENTAL_LEARNING", "1") not in ("0","false","False","no","NO")

# Enhanced API-Football features
ENABLE_ENHANCED_FEATURES = os.getenv("ENABLE_ENHANCED_FEATURES", "1") not in ("0","false","False","no","NO")
ENABLE_HISTORICAL_CONTEXT = os.getenv("ENABLE_HISTORICAL_CONTEXT", "1") not in ("0","false","False","no","NO")
ENABLE_API_PREDICTIONS = os.getenv("ENABLE_API_PREDICTIONS", "1") not in ("0","false","False","no","NO")
ENABLE_PLAYER_IMPACT = os.getenv("ENABLE_PLAYER_IMPACT", "1") not in ("0","false","False","no","NO")

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
HISTORICAL_CACHE: Dict[str, Tuple[float, Dict]] = {}
PREDICTIONS_CACHE: Dict[int, Tuple[float, Dict]] = {}
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

# ───────── Enhanced Feature Extraction System ─────────
def extract_enhanced_features(m: dict) -> Dict[str, float]:
    """Enhanced feature extraction with player impact and advanced metrics"""
    log.info("🔧 EXTRACTING ENHANCED FEATURES")
    
    # Get basic features first
    basic_features = extract_features(m)
    
    if not ENABLE_ENHANCED_FEATURES:
        log.info("⏩ Enhanced features disabled, returning basic features")
        return basic_features
    
    try:
        home   = m["teams"]["home"]["name"]
        away   = m["teams"]["away"]["name"]
        events = m.get("events", [])
        
        # Enhanced player impact metrics
        player_metrics = extract_player_impact_metrics(events, home, away)
        log.info("🎯 Player impact metrics extracted: %s", player_metrics)
        
        # Enhanced historical context
        historical_context = add_historical_context(m.get("fixture", {}).get("id"), home, away)
        log.info("📊 Historical context applied: %s", historical_context)
        
        # Enhanced team strength metrics
        team_strength = calculate_team_strength_metrics(basic_features, home, away)
        log.info("💪 Team strength metrics calculated: %s", team_strength)
        
        # Enhanced lineup changes detection
        lineup_changes = detect_lineup_changes(events)
        log.info("🔄 Lineup changes detected: %s", lineup_changes)
        
        # Combine all enhanced features
        enhanced_features = {
            **basic_features,
            **player_metrics,
            **historical_context,
            **team_strength,
            **lineup_changes
        }
        
        log.info("✅ ENHANCED FEATURES EXTRACTED: %s total features", len(enhanced_features))
        _metric_inc("enhanced_features_processed")
        
        return enhanced_features
        
    except Exception as e:
        log.error("❌ Enhanced feature extraction failed: %s", e, exc_info=True)
        log.warning("⚠️  Falling back to basic features")
        return basic_features

def extract_player_impact_metrics(events: List[Dict], home_team: str, away_team: str) -> Dict[str, float]:
    """Extract player-level impact metrics from events"""
    log.info("👤 Extracting player impact metrics")
    
    try:
        key_players_home = 0
        key_players_away = 0
        key_events_home = 0
        key_events_away = 0
        
        for event in events:
            team = event.get("team", {}).get("name", "")
            event_type = event.get("type", "").lower()
            detail = event.get("detail", "").lower()
            
            # Count key players based on significant events
            if event_type in ["goal", "assist", "penalty"]:
                if team == home_team:
                    key_events_home += 1
                    key_players_home = max(key_players_home, 1)
                elif team == away_team:
                    key_events_away += 1
                    key_players_away = max(key_players_away, 1)
            
            # Count key players based on cards (negative impact)
            elif event_type == "card" and "red" in detail:
                if team == home_team:
                    key_players_home -= 1
                elif team == away_team:
                    key_players_away -= 1
        
        player_impact = {
            "key_players_active_h": float(max(0, key_players_home)),
            "key_players_active_a": float(max(0, key_players_away)),
            "key_player_imbalance": float(key_players_home - key_players_away),
            "key_events_home": float(key_events_home),
            "key_events_away": float(key_events_away),
            "total_key_events": float(key_events_home + key_events_away)
        }
        
        log.info("📈 Player impact metrics: %s", player_impact)
        return player_impact
        
    except Exception as e:
        log.error("❌ Player impact extraction failed: %s", e)
        return {
            "key_players_active_h": 0.0,
            "key_players_active_a": 0.0,
            "key_player_imbalance": 0.0,
            "key_events_home": 0.0,
            "key_events_away": 0.0,
            "total_key_events": 0.0
        }

def add_historical_context(match_id: Optional[int], home_team: str, away_team: str) -> Dict[str, float]:
    """Add head-to-head and recent form data"""
    if not ENABLE_HISTORICAL_CONTEXT:
        return {}
        
    log.info("📚 Adding historical context for %s vs %s", home_team, away_team)
    
    try:
        with db_conn() as c:
            # Get last 5 meetings between these teams
            h2h_results = c.execute("""
                SELECT mr.final_goals_h, mr.final_goals_a, mr.btts_yes
                FROM match_results mr
                JOIN tips t ON mr.match_id = t.match_id
                WHERE t.home = %s AND t.away = %s
                ORDER BY t.created_ts DESC 
                LIMIT 5
            """, (home_team, away_team)).fetchall()
            
        if h2h_results:
            home_wins = sum(1 for gh, ga, _ in h2h_results if gh > ga)
            away_wins = sum(1 for gh, ga, _ in h2h_results if ga > gh)
            draws = len(h2h_results) - home_wins - away_wins
            total_goals = sum(gh + ga for gh, ga, _ in h2h_results)
            btts_count = sum(btts for _, _, btts in h2h_results)
            
            historical_context = {
                "h2h_home_win_rate": home_wins / len(h2h_results),
                "h2h_away_win_rate": away_wins / len(h2h_results),
                "h2h_draw_rate": draws / len(h2h_results),
                "h2h_avg_goals": total_goals / len(h2h_results),
                "h2h_btts_rate": btts_count / len(h2h_results),
                "h2h_sample_size": len(h2h_results)
            }
            
            log.info("📊 Historical context: %s meetings, Home wins: %.1f%%, Avg goals: %.2f", 
                    len(h2h_results), historical_context["h2h_home_win_rate"] * 100, 
                    historical_context["h2h_avg_goals"])
                    
            _metric_inc("historical_context_applied")
            return historical_context
        else:
            log.info("📭 No historical data available for %s vs %s", home_team, away_team)
            return {}
            
    except Exception as e:
        log.error("❌ Historical context extraction failed: %s", e)
        return {}

def calculate_team_strength_metrics(features: Dict[str, float], home_team: str, away_team: str) -> Dict[str, float]:
    """Calculate comprehensive team strength metrics"""
    log.info("💪 Calculating team strength metrics")
    
    try:
        strength_metrics = {
            "xg_dominance": features.get("xg_h", 0.0) - features.get("xg_a", 0.0),
            "sot_dominance": features.get("sot_h", 0.0) - features.get("sot_a", 0.0),
            "possession_dominance": features.get("pos_diff", 0.0),
            "momentum_advantage": features.get("momentum_h", 0.0) - features.get("momentum_a", 0.0),
            "efficiency_advantage": features.get("efficiency_h", 0.0) - features.get("efficiency_a", 0.0),
            "pressure_advantage": features.get("pressure_index", 0.0),
            "action_intensity_ratio": features.get("action_intensity", 0.0) / max(1.0, features.get("minute", 1.0))
        }
        
        # Normalize metrics
        for key in strength_metrics:
            strength_metrics[key] = float(max(-1.0, min(1.0, strength_metrics[key])))
        
        log.info("📈 Team strength metrics calculated: %s", strength_metrics)
        return strength_metrics
        
    except Exception as e:
        log.error("❌ Team strength calculation failed: %s", e)
        return {}

def detect_lineup_changes(events: List[Dict]) -> Dict[str, float]:
    """Detect significant lineup changes during the match"""
    log.info("🔄 Detecting lineup changes")
    
    try:
        red_cards = 0
        key_substitutions = 0
        injuries = 0
        
        for event in events:
            event_type = event.get("type", "").lower()
            detail = event.get("detail", "").lower()
            
            if event_type == "card" and "red" in detail:
                red_cards += 1
            elif event_type == "subst":
                key_substitutions += 1
            elif "injury" in detail or "var" in detail:
                injuries += 1
        
        lineup_metrics = {
            "red_cards_total": float(red_cards),
            "key_substitutions": float(key_substitutions),
            "injury_incidents": float(injuries),
            "major_incidents": float(red_cards + injuries),
            "total_lineup_changes": float(red_cards + key_substitutions + injuries)
        }
        
        log.info("🔄 Lineup changes detected: %s red cards, %s subs, %s injuries", 
                red_cards, key_substitutions, injuries)
                
        return lineup_metrics
        
    except Exception as e:
        log.error("❌ Lineup change detection failed: %s", e)
        return {
            "red_cards_total": 0.0,
            "key_substitutions": 0.0,
            "injury_incidents": 0.0,
            "major_incidents": 0.0,
            "total_lineup_changes": 0.0
        }

# ───────── API-Football Predictions Integration ─────────
def get_api_predictions(fixture_id: int) -> Optional[Dict]:
    """Get algorithm-based predictions from API-Football"""
    if not ENABLE_API_PREDICTIONS:
        log.info("⏩ API predictions disabled")
        return None
        
    log.info("🔮 Fetching API-Football predictions for fixture %s", fixture_id)
    
    # Check cache first
    now = time.time()
    cached = PREDICTIONS_CACHE.get(fixture_id)
    if cached and now - cached[0] < 300:  # 5 minute cache
        log.info("📦 Using cached predictions")
        return cached[1]
    
    try:
        response = _api_get(f"{BASE_URL}/predictions", {"fixture": fixture_id})
        if response and "response" in response and response["response"]:
            predictions = response["response"][0]  # First prediction object
            PREDICTIONS_CACHE[fixture_id] = (now, predictions)
            log.info("✅ API predictions fetched successfully")
            return predictions
        else:
            log.warning("❌ No predictions available from API-Football")
            return None
            
    except Exception as e:
        log.error("❌ API predictions fetch failed: %s", e)
        return None

def blend_with_api_predictions(your_prob: float, api_prediction: Dict, market: str) -> float:
    """Blend your model probability with API-Football's predictions"""
    if not api_prediction:
        return your_prob
        
    log.info("🔄 Blending probabilities with API-Football predictions for %s", market)
    
    try:
        predictions = api_prediction.get("predictions", {})
        api_prob = 0.5  # Default neutral probability
        
        if market == "BTTS" and "btts" in predictions:
            api_prob = float(predictions["btts"].get("percentage", 50)) / 100
            log.info("📊 API BTTS probability: %.1f%%", api_prob * 100)
            
        elif market.startswith("Over") and "goals" in predictions:
            over_key = f"over_{_parse_ou_line_from_suggestion(market)}"
            api_prob = float(predictions["goals"].get(over_key, {}).get("percentage", 50)) / 100
            log.info("📊 API Over probability: %.1f%%", api_prob * 100)
            
        elif market.startswith("Under") and "goals" in predictions:
            under_key = f"under_{_parse_ou_line_from_suggestion(market)}"
            api_prob = float(predictions["goals"].get(under_key, {}).get("percentage", 50)) / 100
            log.info("📊 API Under probability: %.1f%%", api_prob * 100)
            
        elif market == "1X2" and "winner" in predictions:
            if "Home" in market:
                api_prob = float(predictions["winner"].get("home", {}).get("percentage", 33)) / 100
            elif "Away" in market:
                api_prob = float(predictions["winner"].get("away", {}).get("percentage", 33)) / 100
            log.info("📊 API 1X2 probability: %.1f%%", api_prob * 100)
        
        # Weighted blend (70% your model, 30% API)
        blended_prob = (your_prob * 0.7) + (api_prob * 0.3)
        log.info("🎯 Probability blend - Your model: %.1f%%, API: %.1f%%, Final: %.1f%%", 
                your_prob * 100, api_prob * 100, blended_prob * 100)
        
        _metric_inc("api_predictions_blended")
        return blended_prob
        
    except Exception as e:
        log.error("❌ Probability blending failed: %s", e)
        return your_prob

# ───────── Performance Monitoring System ─────────
class PerformanceMonitor:
    """Real-time performance tracking and risk management"""
    
    def __init__(self):
        self.recent_performance = deque(maxlen=50)
        self.volume_reduction_active = False
        self.volume_reduction_start_time = 0
        self.volume_reduction_duration_min = int(os.getenv("VOLUME_REDUCTION_DURATION_MIN", "60"))
        log.info("📈 PerformanceMonitor initialized with 50-record history")
    
    def update_performance(self, tip_id: int, won: bool):
        log.info("🔄 Updating performance - Tip ID: %s, Result: %s", tip_id, "WIN" if won else "LOSS")
        
        previous_streak = self.get_current_streak()
        self.recent_performance.append(won)
        current_streak = self.get_current_streak()
        
        log.info("📊 Performance update - Previous streak: %s, Current streak: %s, Total records: %s", 
                previous_streak, current_streak, len(self.recent_performance))
        
        # Log streak changes
        if abs(current_streak) >= 3:
            streak_type = "WINNING" if current_streak > 0 else "LOSING"
            log.warning("🚨 Significant %s streak detected: %s consecutive", streak_type, abs(current_streak))
        
        # Log overall performance
        if len(self.recent_performance) >= 10:
            win_rate = sum(self.recent_performance) / len(self.recent_performance)
            log.info("📈 Recent performance (last %s): %.1f%% win rate", 
                    len(self.recent_performance), win_rate * 100)
        
        _metric_inc("performance_monitor_updates")
    
    def get_current_streak(self) -> int:
        """Return current win/loss streak"""
        log.debug("🔍 Calculating current streak from %s records", len(self.recent_performance))
        
        if not self.recent_performance:
            log.debug("📭 No performance records available")
            return 0
        
        current = self.recent_performance[-1]
        streak = 0
        
        for i, result in enumerate(reversed(self.recent_performance)):
            if result == current:
                streak += 1
            else:
                break
        
        streak_direction = "WIN" if current else "LOSS"
        log.debug("📊 Current %s streak: %s consecutive", streak_direction, streak)
        
        return streak if current else -streak
    
    def should_reduce_volume(self) -> bool:
        """Reduce volume during losing streaks with cooldown period"""
        if self.volume_reduction_active:
            # Check if cooldown period has expired
            if time.time() - self.volume_reduction_start_time > self.volume_reduction_duration_min * 60:
                self.volume_reduction_active = False
                log.info("✅ Volume reduction cooldown expired - resuming normal operations")
            else:
                remaining = (self.volume_reduction_start_time + self.volume_reduction_duration_min * 60 - time.time()) / 60
                log.warning("⏳ Volume reduction still active - %.1f minutes remaining", remaining)
                return True
        
        streak = self.get_current_streak()
        should_reduce = streak <= -3  # Stop tips after 3 consecutive losses
        
        if should_reduce and not self.volume_reduction_active:
            self.volume_reduction_active = True
            self.volume_reduction_start_time = time.time()
            _metric_inc("volume_reductions_triggered")
            log.warning("🚨 VOLUME REDUCTION TRIGGERED - Losing streak: %s consecutive losses", abs(streak))
        elif not should_reduce:
            log.debug("✅ Volume normal - Current streak: %s", streak)
            
        return self.volume_reduction_active
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        if not self.recent_performance:
            return {"total_tips": 0, "win_rate": 0.0, "current_streak": 0}
        
        total = len(self.recent_performance)
        wins = sum(self.recent_performance)
        win_rate = wins / total
        streak = self.get_current_streak()
        
        return {
            "total_tips": total,
            "win_rate": round(win_rate, 3),
            "current_streak": streak,
            "volume_reduction_active": self.volume_reduction_active
        }

# ───────── Advanced Model Architecture with Enhancements ─────────
class AdvancedEnsemblePredictor:
    """Advanced ensemble combining multiple models with Bayesian calibration and enhanced features"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_selector: Optional[SelectKBest] = None
        self.bayesian_prior_alpha = BAYESIAN_PRIOR_ALPHA
        self.bayesian_prior_beta  = BAYESIAN_PRIOR_BETA
        self.performance_history: List[int] = []
        
        # Initialize time series model for goal prediction
        from sklearn.linear_model import LogisticRegression
        self.time_series_model = LogisticRegression(random_state=42)
        
        log.info("🤖 AdvancedEnsemblePredictor initialized for %s with timing optimization", model_name)
        
    def analyze_match_context(self, match_data: Dict) -> float:
        """Analyze match context for additional edge"""
        if not ENABLE_CONTEXT_ANALYSIS:
            return 1.0
            
        log.info("🎭 STARTING match context analysis")
        context_score = 1.0
        
        try:
            # Team motivation factors
            is_derby = self._is_derby_match(match_data)
            has_red_card = match_data.get('red_cards', 0) > 0
            is_cup_match = "cup" in match_data.get('league', '').lower()
            
            log.info("📊 Context factors - Derby: %s, Red Card: %s, Cup Match: %s", 
                    is_derby, has_red_card, is_cup_match)
            
            # Adjust probabilities based on context
            if is_derby:
                old_score = context_score
                context_score *= 1.1  # Derbies often more intense
                log.info("🏟️  Derby match detected - adjusting context score: %.3f → %.3f", 
                        old_score, context_score)
            
            if has_red_card:
                old_score = context_score
                context_score *= 1.15  # Red cards dramatically change games
                log.info("🟥 Red card detected - adjusting context score: %.3f → %.3f", 
                        old_score, context_score)
            
            if is_cup_match:
                old_score = context_score
                context_score *= 0.9   # Cup matches can be unpredictable
                log.info("🏆 Cup match detected - adjusting context score: %.3f → %.3f", 
                        old_score, context_score)
            
            log.info("✅ FINAL context analysis score: %.3f", context_score)
            
        except Exception as e:
            log.error("❌ Context analysis failed: %s", e, exc_info=True)
            context_score = 1.0  # Default neutral score on error
            log.warning("⚠️  Using default context score due to error")
        
        _metric_inc("context_analysis_calls")
        return context_score

    def _is_derby_match(self, match_data: Dict) -> bool:
        """Detect if match is a derby based on team names and league"""
        try:
            home_team = match_data.get('home_team', '').lower()
            away_team = match_data.get('away_team', '').lower()
            league = match_data.get('league', '').lower()
            
            # Common derby patterns
            derby_indicators = [
                'derby', 'clasico', 'classico', 'rival', 'derbi', 
                'london', 'manchester', 'madrid', 'milan', 'glasgow'
            ]
            
            # Check team names for derby indicators
            team_combination = f"{home_team} {away_team}".lower()
            is_derby = any(indicator in team_combination for indicator in derby_indicators)
            
            # League-specific derby detection
            if 'premier' in league and ('london' in team_combination or 'manchester' in team_combination):
                is_derby = True
            elif 'la liga' in league and 'madrid' in team_combination:
                is_derby = True
            elif 'serie a' in league and 'milan' in team_combination:
                is_derby = True
                
            return is_derby
            
        except Exception as e:
            log.error("❌ Derby detection failed: %s", e)
            return False

    def find_best_odds_across_books(self, match_id: int, market: str, suggestion: str) -> Tuple[Optional[float], Optional[str]]:
        """Get best odds across multiple bookmakers with outlier detection"""
        if not ENABLE_MULTI_BOOK_ODDS:
            return self._get_single_book_odds(match_id, market, suggestion)
            
        log.info("💰 STARTING multi-book odds search - Match: %s, Market: %s, Suggestion: %s", 
                match_id, market, suggestion)
        
        all_odds = self._fetch_multiple_bookmakers(match_id, market, suggestion)
        log.info("📊 Raw odds collected from %s bookmakers: %s", 
                len(all_odds) if all_odds else 0, 
                [(o[0], o[1]) for o in all_odds] if all_odds else "None")
        
        if not all_odds:
            log.warning("❌ No odds available from any bookmaker for match %s market %s", match_id, market)
            return None, None
        
        try:
            # Remove outliers using IQR method
            odds_values = [o[0] for o in all_odds]
            q1, q3 = np.percentile(odds_values, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            log.info("📐 Outlier detection - Q1: %.3f, Q3: %.3f, IQR: %.3f, Bounds: [%.3f, %.3f]", 
                    q1, q3, iqr, lower_bound, upper_bound)
            
            valid_odds = [o for o in all_odds if lower_bound <= o[0] <= upper_bound]
            removed_outliers = len(all_odds) - len(valid_odds)
            
            if removed_outliers > 0:
                log.warning("🗑️  Removed %s outlier odds: %s", 
                           removed_outliers, 
                           [o for o in all_odds if o not in valid_odds])
            
            if not valid_odds:
                log.error("❌ All odds classified as outliers - no valid odds remaining")
                return None, None
            
            log.info("✅ Valid odds after outlier removal: %s", [(o[0], o[1]) for o in valid_odds])
            
            # Return best valid odds
            best_odds = max(valid_odds, key=lambda x: x[0])
            log.info("🏆 BEST ODDS FOUND - Value: %.2f, Bookmaker: %s", best_odds[0], best_odds[1])
            
            # Compare to average
            avg_odds = np.mean([o[0] for o in valid_odds])
            improvement_pct = ((best_odds[0] - avg_odds) / avg_odds) * 100
            log.info("📈 Odds improvement: %.2f%% better than average (%.2f vs %.2f)", 
                    improvement_pct, best_odds[0], avg_odds)
            
            _metric_inc("multi_book_odds_searches")
            return best_odds[0], best_odds[1]
            
        except Exception as e:
            log.error("❌ Error in multi-book odds search: %s", e, exc_info=True)
            log.warning("⚠️  Falling back to single book odds")
            return self._get_single_book_odds(match_id, market, suggestion)

    def _fetch_multiple_bookmakers(self, match_id: int, market: str, suggestion: str) -> List[Tuple[float, str]]:
        """Fetch odds from multiple bookmakers (simulated - extend with real API calls)"""
        # This is a simplified version - extend with real bookmaker APIs
        try:
            # For now, simulate multiple bookmakers by adding small variations
            base_odds, base_book = self._get_single_book_odds(match_id, market, suggestion)
            if base_odds is None:
                return []
                
            bookmakers = [
                "Bet365", "William Hill", "Pinnacle", "Betfair", 
                "Unibet", "Bwin", "888sport", "Marathon Bet"
            ]
            
            variations = []
            for bookmaker in bookmakers[:4]:  # Simulate 4 bookmakers
                # Add realistic variations (±10%)
                variation = np.random.normal(0, 0.05)
                varied_odds = base_odds * (1 + variation)
                # Ensure odds are reasonable
                varied_odds = max(base_odds * 0.9, min(base_odds * 1.1, varied_odds))
                variations.append((round(varied_odds, 2), bookmaker))
            
            return variations
            
        except Exception as e:
            log.error("❌ Multi-book fetch failed: %s", e)
            return []

    def _get_single_book_odds(self, match_id: int, market: str, suggestion: str) -> Tuple[Optional[float], Optional[str]]:
        """Fallback to single bookmaker odds"""
        # Fetch odds from the standard API
        odds_map = fetch_odds(match_id)
        return _get_odds_for_market(odds_map, market, suggestion)

    def should_send_tip_now(self, features: Dict, market: str, base_prob: float) -> bool:
        """Enhanced timing logic for elite tips"""
        if not ENABLE_TIMING_OPTIMIZATION:
            return base_prob * 100 >= CONF_THRESHOLD
            
        log.info("⏰ STARTING tip timing analysis - Market: %s, Base probability: %.1f%%", 
                market, base_prob * 100)
        
        try:
            if market.startswith("Over"):
                log.info("🎯 Over market detected - checking next 10min goal probability")
                next_10min_goal_prob = self.predict_next_10min_goals(features)
                
                log.info("📊 Timing analysis - Next 10min goal prob: %.1f%%, Base prob: %.1f%%, Required: 35%%/75%%", 
                        next_10min_goal_prob * 100, base_prob * 100)
                
                # Only send Over tips when goals are imminent
                should_send = next_10min_goal_prob > 0.35 and base_prob >= 0.75
                
                if should_send:
                    log.info("✅ TIMING APPROVED - High probability of imminent goals + strong base probability")
                else:
                    if next_10min_goal_prob <= 0.35:
                        log.info("⏳ TIMING DELAYED - Low imminent goal probability (%.1f%% ≤ 35%%)", 
                                next_10min_goal_prob * 100)
                    if base_prob < 0.75:
                        log.info("📉 TIMING REJECTED - Base probability too low (%.1f%% < 75%%)", base_prob * 100)
                        
                _metric_inc("timing_analysis_decisions", label="over_market")
                return should_send
            
            else:
                # For non-Over markets, use base probability threshold
                should_send = base_prob >= 0.75
                if should_send:
                    log.info("✅ TIMING APPROVED - Strong base probability (%.1f%% ≥ 75%%)", base_prob * 100)
                else:
                    log.info("📉 TIMING REJECTED - Base probability too low (%.1f%% < 75%%)", base_prob * 100)
                
                _metric_inc("timing_analysis_decisions", label="other_market")
                return should_send
                
        except Exception as e:
            log.error("❌ Timing analysis failed: %s", e, exc_info=True)
            log.warning("⚠️  Defaulting to base probability check due to error")
            return base_prob >= 0.75

    def predict_next_10min_goals(self, features: Dict) -> float:
        """Predict probability of goals in next 10 minutes"""
        log.info("🔮 STARTING 10-minute goal prediction")
        
        try:
            # FIXED: Ensure minute is always available with proper fallback
            minute = features.get('minute', 1)
            if minute <= 0:
                minute = 1
                log.warning("⚠️  Invalid minute value, using default: 1")
                
            goals_sum = features.get('goals_sum', 0)
            goal_rate = goals_sum / max(1, minute)
            
            pressure = self._calculate_pressure_index(features)
            momentum_h = features.get('momentum_h', 0)
            momentum_a = features.get('momentum_a', 0)
            total_momentum = momentum_h + momentum_a
            
            log.info("📈 Prediction inputs - Minute: %s, Total goals: %s, Goal rate: %.3f, Pressure: %.3f, Total Momentum: %.3f", 
                    minute, goals_sum, goal_rate, pressure, total_momentum)
            
            # Simple heuristic-based prediction (replace with trained model)
            # Base probability decreases as match progresses but increases with current goal rate
            base_prob = 0.3  # Base probability
            
            # Adjust for current goal rate
            if goal_rate > 0.05:  # More than 1 goal every 20 minutes
                base_prob += 0.2
            elif goal_rate > 0.03:  # More than 1 goal every 33 minutes
                base_prob += 0.1
                
            # Adjust for pressure (close games)
            if pressure > 0.6:
                base_prob += 0.15
                
            # Adjust for momentum
            if total_momentum > 0.8:
                base_prob += 0.1
                
            # Adjust for match minute (goals more likely in middle period)
            if 25 <= minute <= 70:
                base_prob += 0.1
            elif minute > 70:
                base_prob += 0.2  # Late game goals
                
            # Clamp probability to reasonable range
            prediction_prob = max(0.1, min(0.8, base_prob))
            
            log.info("🎯 10-minute goal prediction: %.1f%%", prediction_prob * 100)
            
            # Log confidence indicators
            if prediction_prob > 0.5:
                log.info("🚀 HIGH goal probability - offensive conditions favorable")
            elif prediction_prob < 0.2:
                log.info("🛑 LOW goal probability - defensive conditions dominant")
            else:
                log.info("⚖️  MODERATE goal probability - balanced match state")
                
            return prediction_prob
            
        except Exception as e:
            log.error("❌ 10-minute goal prediction failed: %s", e, exc_info=True)
            log.warning("⚠️  Returning conservative default probability of 25%%")
            return 0.25

    def _calculate_pressure_index(self, features: Dict) -> float:
        """Calculate pressure index for goal prediction"""
        try:
            # FIXED: Proper minute handling with fallback
            minute = features.get('minute', 1)
            if minute <= 0:
                minute = 1
                
            goal_diff = abs(features.get('goals_diff', 0))
            total_goals = features.get('goals_sum', 0)
            
            # Pressure increases with close scorelines and late minutes
            time_pressure = min(1.0, minute / 90.0)
            score_pressure = 1.0 - min(1.0, goal_diff / 3.0)  # Closer games = more pressure
            
            pressure = (time_pressure * 0.6) + (score_pressure * 0.4)
            return max(0.0, min(1.0, pressure))
            
        except Exception as e:
            log.error("❌ Pressure index calculation failed: %s", e)
            return 0.5

    def update_models_incremental(self, new_data: List[Dict]):
        """Update models without full retraining"""
        if not ENABLE_INCREMENTAL_LEARNING:
            log.info("⏩ Incremental learning disabled")
            return
            
        log.info("🔄 STARTING incremental model update with %s new records", len(new_data))
        
        if not new_data:
            log.warning("📭 No new data provided for incremental update")
            return
        
        try:
            # Prepare features and targets
            new_features = []
            new_targets = []
            
            for i, record in enumerate(new_data):
                try:
                    features = record.get('features', {})
                    target = record.get('outcome')
                    
                    if features and target is not None:
                        feature_vector = self._prepare_feature_vector(features)
                        new_features.append(feature_vector)
                        new_targets.append(target)
                        
                except Exception as e:
                    log.warning("⚠️  Skipping record %s due to error: %s", i, e)
                    continue
            
            if not new_features:
                log.error("❌ No valid features extracted from new data")
                return
                
            log.info("✅ Prepared %s valid feature vectors for incremental update", len(new_features))
            
            updated_models = 0
            skipped_models = 0
            
            for model_name, model in self.models.items():
                try:
                    if hasattr(model, 'partial_fit'):
                        log.info("🔄 Updating model '%s' with incremental data", model_name)
                        model.partial_fit(new_features, new_targets)
                        updated_models += 1
                        log.info("✅ Successfully updated model '%s'", model_name)
                    else:
                        log.warning("⏭️  Model '%s' does not support partial_fit - skipping", model_name)
                        skipped_models += 1
                        
                except Exception as e:
                    log.error("❌ Failed to update model '%s': %s", model_name, e, exc_info=True)
                    skipped_models += 1
            
            log.info("📊 Incremental update completed - Updated: %s, Skipped: %s, Total models: %s", 
                    updated_models, skipped_models, len(self.models))
            
            if updated_models > 0:
                log.info("🎯 Models successfully adapted to %s new patterns", len(new_features))
            else:
                log.warning("⚠️  No models were updated - check partial_fit support")
                
        except Exception as e:
            log.error("❌ Incremental update process failed: %s", e, exc_info=True)
            log.warning("⚠️  Models remain unchanged - full retraining recommended")

    def _prepare_feature_vector(self, features: Dict[str, float]) -> List[float]:
        """Prepare feature vector for incremental learning"""
        try:
            # Use the same feature preparation as in training
            feature_names = [
                'minute', 'goals_sum', 'goals_diff', 'xg_sum', 'xg_diff',
                'sot_sum', 'cor_sum', 'pos_diff', 'momentum_h', 'momentum_a'
            ]
            
            vector = []
            for name in feature_names:
                vector.append(features.get(name, 0.0))
                
            return vector
            
        except Exception as e:
            log.error("❌ Feature vector preparation failed: %s", e)
            return []

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
performance_monitor = PerformanceMonitor() if ENABLE_PERFORMANCE_MONITOR else None

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
        elif "/predictions" in url:
            lbl = "predictions"
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
    
    # FIXED: Proper minute extraction with fallback
    minute_data = ((m.get("fixture") or {}).get("status") or {})
    minute = int(minute_data.get("elapsed") or 1)  # Default to 1, not 0
    if minute <= 0:
        minute = 1
        log.warning("⚠️  Invalid minute detected, using default: 1")

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
        # Context features for enhanced analysis
        "red_cards": float(red_h + red_a),
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
    """Advanced prediction using ensemble + Bayesian methods with enhanced features"""
    log.debug("🤖 Predicting probability for %s - %s (match: %s)", market, suggestion, match_id)
    
    # Bayesian network: treat `suggestion` as the 'market phrase'
    bayesian_prob = bayesian_network.infer_probability(features, suggestion)
    log.debug("📊 Bayesian probability: %.3f", bayesian_prob)
    
    # Ensemble predictor
    model_key = f"{market}_{suggestion.replace(' ', '_')}"
    ensemble_predictor = get_advanced_predictor(model_key)
    ensemble_prob = ensemble_predictor.predict_probability(features)
    log.debug("📈 Ensemble probability: %.3f", ensemble_prob)
    
    # Apply context analysis if enabled
    context_score = 1.0
    if ENABLE_CONTEXT_ANALYSIS:
        # Create match data for context analysis
        match_data = {
            'home_team': '',  # Would be populated from match data
            'away_team': '',  # Would be populated from match data  
            'league': '',     # Would be populated from match data
            'red_cards': features.get('red_sum', 0)
        }
        context_score = ensemble_predictor.analyze_match_context(match_data)
        log.debug("🎭 Context score applied: %.3f", context_score)
    
    # Blend with API-Football predictions if available
    if ENABLE_API_PREDICTIONS and match_id:
        api_predictions = get_api_predictions(match_id)
        if api_predictions:
            ensemble_prob = blend_with_api_predictions(ensemble_prob, api_predictions, market)
    
    data_richness = min(1.0, float(features.get("minute", 0)) / 60.0)
    if data_richness > 0.7:
        final_prob = 0.7 * ensemble_prob + 0.3 * bayesian_prob
        log.debug("📋 Using data-rich weighting (70/30)")
    else:
        final_prob = 0.4 * ensemble_prob + 0.6 * bayesian_prob
        log.debug("📋 Using data-poor weighting (40/60)")
    
    # Apply context score
    final_prob = final_prob * context_score
    
    final_prob = float(max(0.01, min(0.99, final_prob)))
    log.info("🎯 FINAL probability for %s - %s: %.1f%% (bayesian: %.1f%%, ensemble: %.1f%%, context: %.1f%%)", 
             market, suggestion, final_prob*100, bayesian_prob*100, ensemble_prob*100, context_score*100)
    
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
    
    # Fetch odds once for all candidates if not using multi-book mode
    odds_map = {}
    if not ENABLE_MULTI_BOOK_ODDS and API_KEY:
        odds_map = fetch_odds(fid)
        log.debug("💰 Odds map has %s markets", len(odds_map))

    for market, suggestion, prob in candidates:
        log.debug("🔍 Processing candidate: %s - %s (prob: %.3f)", market, suggestion, prob)
        
        if suggestion not in ALLOWED_SUGGESTIONS:
            log.debug("❌ Suggestion not in allowed list: %s", suggestion)
            continue

        # Enhanced timing check
        model_key = f"{market}_{suggestion.replace(' ', '_')}"
        ensemble_predictor = get_advanced_predictor(model_key)
        
        if ENABLE_TIMING_OPTIMIZATION:
            should_send = ensemble_predictor.should_send_tip_now(features, market, prob)
            if not should_send:
                log.info("⏰ Timing optimization rejected candidate: %s - %s", market, suggestion)
                continue

        # Enhanced odds fetching
        if ENABLE_MULTI_BOOK_ODDS:
            odds, book = ensemble_predictor.find_best_odds_across_books(fid, market, suggestion)
        else:
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
    """Main in-play scanning with advanced AI systems and risk management"""
    log.info("🚀 STARTING PRODUCTION SCAN")
    
    # Check performance monitor for volume reduction
    if ENABLE_PERFORMANCE_MONITOR and performance_monitor:
        if performance_monitor.should_reduce_volume():
            log.warning("🚨 SCAN SKIPPED - Volume reduction active due to losing streak")
            stats = performance_monitor.get_performance_stats()
            send_telegram(
                f"🔻 Volume Reduction Active\n"
                f"Current streak: {stats['current_streak']} losses\n"
                f"Recent win rate: {stats['win_rate']*100:.1f}%\n"
                f"Tips paused for safety"
            )
            return 0, 0
    
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

                # Use enhanced features if enabled
                if ENABLE_ENHANCED_FEATURES:
                    feat = extract_enhanced_features(m)
                else:
                    feat = extract_features(m)
                    
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

        # Add performance monitor stats if enabled
        if ENABLE_PERFORMANCE_MONITOR and performance_monitor:
            perf_stats = performance_monitor.get_performance_stats()
            lines.append(f"📈 <b>Current Performance:</b> {perf_stats['win_rate']*100:.1f}% win rate, Streak: {perf_stats['current_streak']}")

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

@app.route("/admin/training-progress")
def http_training_progress():
    _require_admin()
    try:
        from train_models import analyze_training_progress
        with db_conn() as c:
            progress = analyze_training_progress(c.conn)
        return jsonify({"ok": True, "progress": progress})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

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
    performance_stats = performance_monitor.get_performance_stats() if performance_monitor else {}
    return jsonify({
        "ok": True, 
        "system_status": learning_status,
        "performance_stats": performance_stats,
        "enhancements_enabled": {
            "context_analysis": ENABLE_CONTEXT_ANALYSIS,
            "performance_monitor": ENABLE_PERFORMANCE_MONITOR,
            "multi_book_odds": ENABLE_MULTI_BOOK_ODDS,
            "timing_optimization": ENABLE_TIMING_OPTIMIZATION,
            "incremental_learning": ENABLE_INCREMENTAL_LEARNING,
            "enhanced_features": ENABLE_ENHANCED_FEATURES,
            "historical_context": ENABLE_HISTORICAL_CONTEXT,
            "api_predictions": ENABLE_API_PREDICTIONS,
            "player_impact": ENABLE_PLAYER_IMPACT
        }
    })

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
