# goalsniper — OU 2.5 ONLY (in-play + prematch snapshot) — Railway-ready
# Enhanced with extensive logging and learning functions

import os, json, time, logging, requests, sys, signal, atexit
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import traceback

import numpy as np
import psycopg2
from psycopg2.pool import SimpleConnectionPool
from flask import Flask, jsonify, request, abort
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# ───────── Enhanced Logging Configuration ─────────
class StructuredLogger:
    """Enhanced logger with structured JSON logging for better analysis"""
    
    def __init__(self, name="goalsniper_ou25"):
        self.logger = logging.getLogger(name)
        self.logger.handlers.clear()
        
        # Console handler with structured format
        console_handler = logging.StreamHandler()
        console_format = logging.Formatter(
            "[%(asctime)s] %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # File handler for persistent logs
        log_file = os.getenv("LOG_FILE", "goalsniper.log")
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_format = logging.Formatter(
                '{"time": "%(asctime)s", "level": "%(levelname)s", '
                '"file": "%(filename)s", "line": %(lineno)d, '
                '"function": "%(funcName)s", "message": "%(message)s"}'
            )
            file_handler.setFormatter(file_format)
            self.logger.addHandler(file_handler)
        
        self.logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))
        self.logger.propagate = False
        
        # Performance tracking
        self.performance_stats = {
            "scans_completed": 0,
            "tips_generated": 0,
            "tips_sent": 0,
            "api_calls": 0,
            "errors": 0,
            "avg_processing_time": 0
        }
    
    def log_scan_start(self, scan_id: str):
        """Log scan initiation"""
        self.logger.info(f"SCAN_START id={scan_id} time={time.time()}")
    
    def log_scan_end(self, scan_id: str, saved: int, live_seen: int, duration: float):
        """Log scan completion"""
        self.logger.info(
            f"SCAN_END id={scan_id} saved={saved} live_seen={live_seen} "
            f"duration={duration:.2f}s"
        )
        self.performance_stats["scans_completed"] += 1
    
    def log_match_processing(self, match_id: int, minute: int, score: str, 
                           confidence: float, decision: str, reason: str = ""):
        """Log detailed match processing"""
        self.logger.info(
            f"MATCH_PROCESSING match_id={match_id} minute={minute} "
            f"score={score} confidence={confidence:.2f}% "
            f"decision={decision} reason={reason}"
        )
    
    def log_tip_creation(self, tip_id: str, match_id: int, suggestion: str,
                        confidence: float, odds: float, ev_pct: float):
        """Log tip creation"""
        self.logger.info(
            f"TIP_CREATED tip_id={tip_id} match_id={match_id} "
            f"suggestion={suggestion} confidence={confidence:.2f}% "
            f"odds={odds:.2f} ev={ev_pct:.2f}%"
        )
        self.performance_stats["tips_generated"] += 1
    
    def log_tip_sent(self, tip_id: str, success: bool, telegram_response: str = ""):
        """Log tip sending result"""
        level = logging.INFO if success else logging.ERROR
        self.logger.log(
            level,
            f"TIP_SENT tip_id={tip_id} success={success} "
            f"response={telegram_response}"
        )
        if success:
            self.performance_stats["tips_sent"] += 1
    
    def log_api_call(self, endpoint: str, status: str, duration: float):
        """Log API call details"""
        self.logger.debug(
            f"API_CALL endpoint={endpoint} status={status} "
            f"duration={duration:.3f}s"
        )
        self.performance_stats["api_calls"] += 1
    
    def log_error(self, context: str, error: Exception, critical: bool = False):
        """Log errors with context"""
        level = logging.ERROR if critical else logging.WARNING
        self.logger.log(
            level,
            f"ERROR context={context} error={str(error)} "
            f"traceback={traceback.format_exc()}"
        )
        self.performance_stats["errors"] += 1
    
    def log_performance(self):
        """Log performance statistics"""
        self.logger.info(
            f"PERFORMANCE_STATS {json.dumps(self.performance_stats)}"
        )
    
    def log_learning_update(self, metric: str, old_value: float, 
                           new_value: float, reason: str):
        """Log learning updates"""
        self.logger.info(
            f"LEARNING_UPDATE metric={metric} old={old_value:.4f} "
            f"new={new_value:.4f} reason={reason}"
        )

# Initialize enhanced logger
log_enhanced = StructuredLogger("goalsniper_ou25")
log = log_enhanced.logger  # Backward compatibility

app = Flask(__name__)

# ───────── Env ─────────
def _require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise SystemExit(f"Missing required environment variable: {name}")
    return v

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# Core env
TELEGRAM_BOT_TOKEN = _require_env("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = _require_env("TELEGRAM_CHAT_ID")
API_KEY            = _require_env("API_KEY")
DATABASE_URL       = _require_env("DATABASE_URL")

# Knobs (focused on OU 2.5 only)
CONF_THRESHOLD     = float(os.getenv("CONF_THRESHOLD", "72"))  # % threshold for sending tips
MAX_TIPS_PER_SCAN  = int(os.getenv("MAX_TIPS_PER_SCAN", "20"))
TIP_MIN_MINUTE     = int(os.getenv("TIP_MIN_MINUTE", "15"))
SCAN_INTERVAL_SEC  = int(os.getenv("SCAN_INTERVAL_SEC", "240"))
TOTAL_MATCH_MINUTES= int(os.getenv("TOTAL_MATCH_MINUTES", "95"))
PER_LEAGUE_CAP     = int(os.getenv("PER_LEAGUE_CAP", "3"))
RUN_SCHEDULER      = os.getenv("RUN_SCHEDULER", "1") not in ("0","false","False","no","NO")

# Odds/EV gates (only OU 2.5)
MIN_ODDS_OU        = float(os.getenv("MIN_ODDS_OU", "1.50"))
MAX_ODDS_ALL       = float(os.getenv("MAX_ODDS_ALL", "20.0"))
EDGE_MIN_BPS       = int(os.getenv("EDGE_MIN_BPS", "500"))  # +5% EV by default
ODDS_REQUIRE_N_BOOKS = int(os.getenv("ODDS_REQUIRE_N_BOOKS", "2"))
ODDS_AGGREGATION   = os.getenv("ODDS_AGGREGATION", "median").lower() # median|best
ODDS_OUTLIER_MULT  = float(os.getenv("ODDS_OUTLIER_MULT", "1.8"))
ODDS_FAIR_MAX_MULT = float(os.getenv("ODDS_FAIR_MAX_MULT", "2.5"))
ALLOW_TIPS_WITHOUT_ODDS = os.getenv("ALLOW_TIPS_WITHOUT_ODDS","0") not in ("0","false","False","no","NO")

# Data-quality guards
REQUIRE_STATS_MINUTE = int(os.getenv("REQUIRE_STATS_MINUTE", "35"))
REQUIRE_DATA_FIELDS  = int(os.getenv("REQUIRE_DATA_FIELDS", "2"))
STALE_GUARD_ENABLE   = os.getenv("STALE_GUARD_ENABLE","1") not in ("0","false","False","no","NO")
STALE_STATS_MAX_SEC  = int(os.getenv("STALE_STATS_MAX_SEC","240"))

# Learning system parameters
LEARNING_ENABLED       = os.getenv("LEARNING_ENABLED", "1") not in ("0","false","False","no","NO")
CONFIDENCE_ADJUSTMENT  = float(os.getenv("CONFIDENCE_ADJUSTMENT", "0.1"))  # % adjustment per win/loss
MIN_LEARNING_SAMPLES   = int(os.getenv("MIN_LEARNING_SAMPLES", "50"))
LEARNING_WINDOW_DAYS   = int(os.getenv("LEARNING_WINDOW_DAYS", "30"))
EV_THRESHOLD_ADJUST    = float(os.getenv("EV_THRESHOLD_ADJUST", "0.05"))  # EV threshold adjustment
ODDS_QUALITY_THRESHOLD = float(os.getenv("ODDS_QUALITY_THRESHOLD", "0.8"))  # Bookmaker agreement threshold
PERFORMANCE_TRACKING   = os.getenv("PERFORMANCE_TRACKING", "1") not in ("0","false","False","no","NO")

# Timezones
TZ_UTC = ZoneInfo("UTC")
BERLIN_TZ = ZoneInfo("Europe/Berlin")

# ───────── Learning System ─────────
class LearningSystem:
    """Self-improving system that learns from tip outcomes"""
    
    def __init__(self, db_conn_func):
        self.db_conn = db_conn_func
        self.learning_enabled = LEARNING_ENABLED
        self.confidence_adjustment = CONFIDENCE_ADJUSTMENT
        self.min_samples = MIN_LEARNING_SAMPLES
        
        # Performance tracking
        self.performance_history = {
            "accuracy": [],
            "ev": [],
            "confidence_bias": 0.0,
            "odds_bias": 0.0,
            "feature_importance": {}
        }
    
    def analyze_tip_performance(self, days_back: int = LEARNING_WINDOW_DAYS) -> Dict[str, Any]:
        """Analyze performance of recent tips to identify patterns"""
        cutoff_ts = int(time.time()) - (days_back * 24 * 3600)
        
        with self.db_conn() as c:
            # Get graded tips with outcomes
            query = """
                SELECT t.match_id, t.suggestion, t.confidence_raw, t.odds, t.ev_pct,
                       t.minute, t.score_at_tip, r.final_goals_h, r.final_goals_a
                FROM tips t
                JOIN match_results r ON t.match_id = r.match_id
                WHERE t.created_ts >= %s
                  AND t.market = 'Over/Under 2.5'
                  AND t.sent_ok = 1
                  AND r.final_goals_h IS NOT NULL
                ORDER BY t.created_ts DESC
                LIMIT 500
            """
            rows = c.execute(query, (cutoff_ts,)).fetchall()
        
        if len(rows) < self.min_samples:
            return {"status": "insufficient_data", "sample_size": len(rows)}
        
        # Calculate performance metrics
        wins = 0
        ev_sum = 0
        confidence_bias_sum = 0
        odds_discrepancy_sum = 0
        
        for row in rows:
            match_id, suggestion, confidence_raw, odds, ev_pct, minute, score, gh, ga = row
            
            # Determine if tip was correct
            total_goals = (gh or 0) + (ga or 0)
            if suggestion.startswith("Over"):
                correct = total_goals > 2.5
            else:  # Under
                correct = total_goals < 2.5
            
            if correct:
                wins += 1
            
            if ev_pct:
                ev_sum += ev_pct
            
            # Calculate confidence bias (actual vs predicted)
            outcome_prob = 1.0 if correct else 0.0
            if confidence_raw:
                confidence_bias_sum += (outcome_prob - confidence_raw)
            
            # Calculate odds discrepancy
            if odds and confidence_raw:
                fair_odds = 1.0 / max(0.01, confidence_raw)
                odds_discrepancy_sum += (odds - fair_odds)
        
        accuracy = (wins / len(rows)) * 100 if rows else 0
        avg_ev = ev_sum / len(rows) if rows and ev_sum else 0
        avg_confidence_bias = confidence_bias_sum / len(rows) if rows and confidence_bias_sum else 0
        avg_odds_discrepancy = odds_discrepancy_sum / len(rows) if rows and odds_discrepancy_sum else 0
        
        # Store in performance history
        self.performance_history["accuracy"].append(accuracy)
        self.performance_history["ev"].append(avg_ev)
        self.performance_history["confidence_bias"] = avg_confidence_bias
        self.performance_history["odds_bias"] = avg_odds_discrepancy
        
        log_enhanced.logger.info(
            f"PERFORMANCE_ANALYSIS samples={len(rows)} accuracy={accuracy:.2f}% "
            f"avg_ev={avg_ev:.2f}% confidence_bias={avg_confidence_bias:.4f} "
            f"odds_bias={avg_odds_discrepancy:.4f}"
        )
        
        return {
            "status": "success",
            "sample_size": len(rows),
            "accuracy": accuracy,
            "avg_ev": avg_ev,
            "confidence_bias": avg_confidence_bias,
            "odds_bias": avg_odds_discrepancy,
            "wins": wins,
            "total": len(rows)
        }
    
    def adjust_confidence_threshold(self, performance: Dict[str, Any]) -> float:
        """Adjust confidence threshold based on performance"""
        if not self.learning_enabled:
            return CONF_THRESHOLD
        
        if performance.get("status") != "success":
            log_enhanced.logger.info("LEARNING: Insufficient data for threshold adjustment")
            return CONF_THRESHOLD
        
        accuracy = performance.get("accuracy", 0)
        confidence_bias = performance.get("confidence_bias", 0)
        
        # Get current threshold
        current_threshold = self._get_current_threshold()
        
        # Adjust threshold based on accuracy
        target_accuracy = 65.0  # Target 65% accuracy
        
        if accuracy < target_accuracy - 5:  # Underperforming by 5%
            # Increase threshold to be more conservative
            adjustment = min(3.0, (target_accuracy - accuracy) * 0.2)
            new_threshold = current_threshold + adjustment
            reason = f"Accuracy {accuracy:.1f}% below target {target_accuracy}%"
        elif accuracy > target_accuracy + 5:  # Overperforming by 5%
            # Decrease threshold to be more aggressive
            adjustment = min(2.0, (accuracy - target_accuracy) * 0.1)
            new_threshold = max(60.0, current_threshold - adjustment)
            reason = f"Accuracy {accuracy:.1f}% above target {target_accuracy}%"
        else:
            # Adjust based on confidence bias
            if abs(confidence_bias) > 0.05:  # Significant bias
                new_threshold = current_threshold + (confidence_bias * 20)  # Scale adjustment
                reason = f"Confidence bias {confidence_bias:.3f}"
            else:
                new_threshold = current_threshold
                reason = "Performance within acceptable range"
        
        # Apply bounds
        new_threshold = max(60.0, min(85.0, new_threshold))
        
        if abs(new_threshold - current_threshold) > 0.1:
            self._save_threshold(new_threshold)
            log_enhanced.log_learning_update(
                "confidence_threshold",
                current_threshold,
                new_threshold,
                reason
            )
            return new_threshold
        
        return current_threshold
    
    def adjust_ev_threshold(self, performance: Dict[str, Any]) -> float:
        """Adjust EV threshold based on performance"""
        if not self.learning_enabled:
            return EDGE_MIN_BPS / 10000.0  # Convert bps to decimal
        
        if performance.get("status") != "success":
            return EDGE_MIN_BPS / 10000.0
        
        avg_ev = performance.get("avg_ev", 0)
        current_ev_threshold = EDGE_MIN_BPS / 10000.0
        
        # Target positive EV while maintaining tip volume
        if avg_ev < 2.0:  # Low average EV
            # Increase threshold to be more selective
            adjustment = min(0.02, (2.0 - avg_ev) * 0.01)
            new_threshold = current_ev_threshold + adjustment
            reason = f"Average EV {avg_ev:.2f}% too low"
        elif avg_ev > 8.0:  # High average EV
            # Decrease threshold to capture more value
            adjustment = min(0.01, (avg_ev - 8.0) * 0.005)
            new_threshold = max(0.01, current_ev_threshold - adjustment)
            reason = f"Average EV {avg_ev:.2f}% very high"
        else:
            new_threshold = current_ev_threshold
            reason = "EV performance satisfactory"
        
        new_threshold = max(0.01, min(0.10, new_threshold))  # Bound between 1% and 10%
        
        if abs(new_threshold - current_ev_threshold) > 0.005:
            # Save as bps
            new_bps = int(new_threshold * 10000)
            self._save_ev_threshold(new_bps)
            log_enhanced.log_learning_update(
                "ev_threshold",
                current_ev_threshold * 10000,
                new_bps,
                reason
            )
            return new_threshold
        
        return current_ev_threshold
    
    def learn_from_mistake(self, match_id: int, tip_data: Dict[str, Any], 
                          actual_outcome: Dict[str, Any]):
        """Learn from individual tip mistakes"""
        if not self.learning_enabled:
            return
        
        predicted = tip_data.get("suggestion", "")
        confidence = tip_data.get("confidence_raw", 0.5)
        minute = tip_data.get("minute", 0)
        
        # Determine if prediction was correct
        gh = actual_outcome.get("final_goals_h", 0)
        ga = actual_outcome.get("final_goals_a", 0)
        total_goals = gh + ga
        
        correct = False
        if predicted.startswith("Over"):
            correct = total_goals > 2.5
        elif predicted.startswith("Under"):
            correct = total_goals < 2.5
        
        if not correct:
            # Log the mistake for analysis
            with self.db_conn() as c:
                c.execute("""
                    INSERT INTO tip_mistakes 
                    (match_id, predicted, confidence, minute, total_goals, created_ts)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (match_id, predicted, confidence, minute, total_goals, int(time.time())))
            
            log_enhanced.logger.info(
                f"MISTAKE_LEARNED match_id={match_id} predicted={predicted} "
                f"confidence={confidence:.3f} actual_goals={total_goals} "
                f"minute={minute}"
            )
    
    def update_model_calibration(self, performance: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update model calibration based on performance data"""
        if not self.learning_enabled:
            return None
        
        if performance.get("status") != "success":
            return None
        
        confidence_bias = performance.get("confidence_bias", 0)
        
        # Get current model
        current_model = _load_ou25_model()
        if not current_model:
            return None
        
        # Adjust calibration parameters
        cal = current_model.get("calibration", {})
        a = cal.get("a", 1.0)
        b = cal.get("b", 0.0)
        
        # Adjust b based on confidence bias
        new_b = b - (confidence_bias * 0.5)  # Scale adjustment
        
        # Update calibration
        current_model["calibration"] = {
            "method": "platt",
            "a": a,
            "b": new_b
        }
        
        log_enhanced.log_learning_update(
            "model_calibration",
            b,
            new_b,
            f"Confidence bias {confidence_bias:.3f}"
        )
        
        return current_model
    
    def _get_current_threshold(self) -> float:
        """Get current confidence threshold from settings"""
        with self.db_conn() as c:
            row = c.execute(
                "SELECT value FROM settings WHERE key = %s",
                ("conf_threshold:Over/Under 2.5",)
            ).fetchone()
            if row and row[0]:
                try:
                    return float(row[0])
                except:
                    return CONF_THRESHOLD
        return CONF_THRESHOLD
    
    def _save_threshold(self, threshold: float):
        """Save new threshold to settings"""
        with self.db_conn() as c:
            c.execute("""
                INSERT INTO settings (key, value) 
                VALUES (%s, %s)
                ON CONFLICT (key) 
                DO UPDATE SET value = EXCLUDED.value
            """, ("conf_threshold:Over/Under 2.5", str(threshold)))
    
    def _save_ev_threshold(self, bps: int):
        """Save new EV threshold to settings"""
        with self.db_conn() as c:
            c.execute("""
                INSERT INTO settings (key, value) 
                VALUES (%s, %s)
                ON CONFLICT (key) 
                DO UPDATE SET value = EXCLUDED.value
            """, ("edge_min_bps", str(bps)))
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        performance = self.analyze_tip_performance(LEARNING_WINDOW_DAYS)
        
        return {
            "learning_enabled": self.learning_enabled,
            "performance": performance,
            "history_summary": {
                "accuracy_trend": self._calculate_trend(self.performance_history["accuracy"]),
                "ev_trend": self._calculate_trend(self.performance_history["ev"]),
                "confidence_bias": self.performance_history["confidence_bias"],
                "odds_bias": self.performance_history["odds_bias"]
            },
            "current_threshold": self._get_current_threshold(),
            "recommendations": self._generate_recommendations(performance)
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend from historical values"""
        if len(values) < 2:
            return "insufficient_data"
        
        recent = values[-5:] if len(values) >= 5 else values
        if len(recent) < 2:
            return "stable"
        
        first = sum(recent[:2]) / 2
        last = sum(recent[-2:]) / 2
        
        if last > first * 1.05:
            return "improving"
        elif last < first * 0.95:
            return "declining"
        else:
            return "stable"
    
    def _generate_recommendations(self, performance: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if performance.get("status") != "success":
            recommendations.append("Collect more data before making adjustments")
            return recommendations
        
        accuracy = performance.get("accuracy", 0)
        avg_ev = performance.get("avg_ev", 0)
        confidence_bias = performance.get("confidence_bias", 0)
        
        if accuracy < 60:
            recommendations.append("Consider increasing confidence threshold by 2-3%")
        
        if accuracy > 70:
            recommendations.append("Consider decreasing confidence threshold by 1-2%")
        
        if avg_ev < 1:
            recommendations.append("Increase EV threshold to be more selective")
        
        if abs(confidence_bias) > 0.1:
            if confidence_bias > 0:
                recommendations.append("Model is overconfident, consider calibration adjustment")
            else:
                recommendations.append("Model is underconfident, consider calibration adjustment")
        
        return recommendations

# Initialize learning system
learning_system = None

# ───────── External APIs ─────────
BASE_URL = "https://v3.football.api-sports.io"
FOOTBALL_API_URL = f"{BASE_URL}/fixtures"
HEADERS = {"x-apisports-key": API_KEY, "Accept": "application/json"}
INPLAY_STATUSES = {"1H","HT","2H","ET","BT","P"}

class EnhancedSession:
    """Enhanced session with detailed logging and retry logic"""
    
    def __init__(self):
        self.session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            respect_retry_after_header=True
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
    
    def get(self, url: str, params: dict = None, timeout: float = None) -> Optional[dict]:
        """Make GET request with detailed logging"""
        start_time = time.time()
        timeout = timeout or float(os.getenv("REQ_TIMEOUT_SEC", "8.0"))
        
        try:
            response = self.session.get(
                url, 
                headers=HEADERS, 
                params=params, 
                timeout=timeout
            )
            duration = time.time() - start_time
            
            if response.ok:
                log_enhanced.log_api_call(url, "success", duration)
                return response.json()
            else:
                log_enhanced.log_api_call(url, f"error_{response.status_code}", duration)
                log_enhanced.logger.warning(
                    f"API_ERROR url={url} status={response.status_code} "
                    f"duration={duration:.3f}s"
                )
                return None
        except requests.exceptions.Timeout:
            duration = time.time() - start_time
            log_enhanced.log_api_call(url, "timeout", duration)
            log_enhanced.logger.warning(f"API_TIMEOUT url={url} duration={duration:.3f}s")
            return None
        except Exception as e:
            duration = time.time() - start_time
            log_enhanced.log_api_call(url, f"exception_{type(e).__name__}", duration)
            log_enhanced.log_error(f"API call to {url}", e)
            return None

session = EnhancedSession()
REQ_TIMEOUT_SEC = float(os.getenv("REQ_TIMEOUT_SEC", "8.0"))

# ───────── Telegram ─────────
def send_telegram(text: str) -> Tuple[bool, str]:
    """Send Telegram message with detailed logging"""
    tip_id = f"tip_{int(time.time())}_{hash(text) % 10000:04d}"
    
    try:
        start_time = time.time()
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            data={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True
            },
            timeout=REQ_TIMEOUT_SEC
        )
        duration = time.time() - start_time
        
        if r.ok:
            response_data = r.json()
            log_enhanced.log_tip_sent(tip_id, True, f"duration={duration:.3f}s")
            return True, tip_id
        else:
            log_enhanced.log_tip_sent(tip_id, False, f"status={r.status_code}")
            return False, tip_id
    except Exception as e:
        log_enhanced.log_error("send_telegram", e)
        return False, tip_id

# ───────── DB Pool ─────────
POOL: Optional[SimpleConnectionPool] = None

def _normalize_dsn(url: str) -> str:
    if not url: 
        return url
    dsn = url.strip()
    if "sslmode=" not in dsn:
        dsn += ("&" if "?" in dsn else "?") + "sslmode=require"
    return dsn

def _init_pool():
    global POOL
    if POOL: 
        return
    
    try:
        POOL = SimpleConnectionPool(
            minconn=1,
            maxconn=int(os.getenv("PG_MAXCONN", "10")),
            dsn=_normalize_dsn(DATABASE_URL),
            connect_timeout=int(os.getenv("PG_CONNECT_TIMEOUT", "10")),
            application_name="goalsniper_ou25"
        )
        log.info("[DB] pool initialized successfully")
    except Exception as e:
        log_enhanced.log_error("DB pool initialization", e, critical=True)
        raise

class PooledConn:
    """Enhanced connection context manager with logging"""
    
    def __enter__(self):
        try:
            self.conn = POOL.getconn()  # type: ignore
            self.conn.autocommit = True
            self.cur = self.conn.cursor()
            return self
        except Exception as e:
            log_enhanced.log_error("DB connection acquisition", e, critical=True)
            raise
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.cur.close()
        except Exception as e:
            log_enhanced.logger.debug(f"Error closing cursor: {e}")
        
        try:
            POOL.putconn(self.conn)  # type: ignore
        except Exception as e:
            log_enhanced.logger.debug(f"Error returning connection to pool: {e}")
        
        if exc_type:
            log_enhanced.log_error("DB transaction", exc_val)
    
    def execute(self, sql: str, params: tuple | list = ()):
        """Execute SQL with logging"""
        log_enhanced.logger.debug(f"DB_EXECUTE sql={sql[:100]} params={params}")
        self.cur.execute(sql, params or ())
        return self.cur

def db_conn():
    """Get database connection with initialization"""
    if not POOL:
        _init_pool()
    return PooledConn()

def init_db():
    """Initialize database with enhanced schema"""
    try:
        with db_conn() as c:
            # Existing tables
            c.execute("""CREATE TABLE IF NOT EXISTS tips (
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
                telegram_message_id TEXT,
                learning_features JSONB,
                PRIMARY KEY (match_id, created_ts))""")
            
            c.execute("""CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY, 
                value TEXT)""")
            
            c.execute("""CREATE TABLE IF NOT EXISTS match_results (
                match_id BIGINT PRIMARY KEY, 
                final_goals_h INTEGER, 
                final_goals_a INTEGER, 
                btts_yes INTEGER, 
                updated_ts BIGINT)""")
            
            c.execute("""CREATE TABLE IF NOT EXISTS odds_history (
                match_id BIGINT,
                captured_ts BIGINT,
                market TEXT,
                selection TEXT,
                odds DOUBLE PRECISION,
                book TEXT,
                PRIMARY KEY (match_id, market, selection, captured_ts)
            )""")
            
            # New tables for learning system
            c.execute("""CREATE TABLE IF NOT EXISTS tip_mistakes (
                id BIGSERIAL PRIMARY KEY,
                match_id BIGINT,
                predicted TEXT,
                confidence DOUBLE PRECISION,
                minute INTEGER,
                total_goals INTEGER,
                features JSONB,
                created_ts BIGINT,
                analyzed BOOLEAN DEFAULT FALSE
            )""")
            
            c.execute("""CREATE TABLE IF NOT EXISTS learning_history (
                id BIGSERIAL PRIMARY KEY,
                timestamp BIGINT,
                metric TEXT,
                old_value DOUBLE PRECISION,
                new_value DOUBLE PRECISION,
                reason TEXT,
                sample_size INTEGER
            )""")
            
            c.execute("""CREATE TABLE IF NOT EXISTS performance_snapshots (
                id BIGSERIAL PRIMARY KEY,
                timestamp BIGINT,
                period_days INTEGER,
                total_tips INTEGER,
                graded_tips INTEGER,
                wins INTEGER,
                accuracy DOUBLE PRECISION,
                avg_ev DOUBLE PRECISION,
                confidence_bias DOUBLE PRECISION,
                odds_bias DOUBLE PRECISION,
                confidence_threshold DOUBLE PRECISION,
                ev_threshold_bps INTEGER
            )""")
            
            # Indexes
            c.execute("CREATE INDEX IF NOT EXISTS idx_tips_created ON tips (created_ts DESC)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_tips_match ON tips (match_id)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_odds_hist_match ON odds_history (match_id, captured_ts DESC)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_tip_mistakes_ts ON tip_mistakes (created_ts DESC)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_learning_history_ts ON learning_history (timestamp DESC)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_performance_snapshots_ts ON performance_snapshots (timestamp DESC)")
        
        log.info("[DB] schema ensured successfully")
        
        # Initialize learning system
        global learning_system
        learning_system = LearningSystem(db_conn)
        log.info("[LEARNING] System initialized")
        
    except Exception as e:
        log_enhanced.log_error("Database initialization", e, critical=True)
        raise

# ───────── Settings helpers ─────────
def get_setting(key: str) -> Optional[str]:
    with db_conn() as c:
        cur = c.execute("SELECT value FROM settings WHERE key=%s", (key,))
        row = cur.fetchone()
        return (row[0] if row else None)

def set_setting(key: str, value: str) -> None:
    with db_conn() as c:
        c.execute(
            "INSERT INTO settings(key,value) VALUES(%s,%s) "
            "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value", 
            (key, value)
        )

# ───────── API helpers ─────────
def _api_get(url: str, params: dict, timeout: int = 12) -> Optional[dict]:
    """Wrapper for API calls with logging"""
    return session.get(url, params, timeout)

def fetch_match_stats(fid: int) -> list:
    """Fetch match statistics with error handling"""
    log_enhanced.logger.debug(f"Fetching stats for fixture {fid}")
    js = _api_get(f"{FOOTBALL_API_URL}/statistics", {"fixture": fid}) or {}
    stats = js.get("response", []) if isinstance(js, dict) else []
    log_enhanced.logger.debug(f"Retrieved {len(stats)} stat entries for fixture {fid}")
    return stats

def fetch_match_events(fid: int) -> list:
    """Fetch match events with error handling"""
    log_enhanced.logger.debug(f"Fetching events for fixture {fid}")
    js = _api_get(f"{FOOTBALL_API_URL}/events", {"fixture": fid}) or {}
    events = js.get("response", []) if isinstance(js, dict) else []
    log_enhanced.logger.debug(f"Retrieved {len(events)} events for fixture {fid}")
    return events

_BLOCK_PATTERNS = ["u17","u18","u19","u20","u21","u23","youth","junior","reserve","res.","friendlies","friendly"]
def _blocked_league(league_obj: dict) -> bool:
    name=str((league_obj or {}).get("name","")).lower()
    country=str((league_obj or {}).get("country","")).lower()
    typ=str((league_obj or {}).get("type","")).lower()
    txt=f"{country} {name} {typ}"
    if any(p in txt for p in _BLOCK_PATTERNS): 
        return True
    deny=[x.strip() for x in os.getenv("LEAGUE_DENY_IDS","").split(",") if x.strip()]
    lid=str((league_obj or {}).get("id") or "")
    return lid in deny

def fetch_live_matches() -> List[dict]:
    """Fetch live matches with detailed logging"""
    log_enhanced.logger.info("Fetching live matches")
    start_time = time.time()
    
    js = _api_get(FOOTBALL_API_URL, {"live": "all"}) or {}
    matches = [m for m in (js.get("response",[]) if isinstance(js,dict) else []) 
               if not _blocked_league(m.get("league") or {})]
    
    log_enhanced.logger.info(f"Retrieved {len(matches)} total matches")
    
    out=[]
    for m in matches:
        st=((m.get("fixture") or {}).get("status") or {})
        elapsed=st.get("elapsed"); short=(st.get("short") or "").upper()
        if elapsed is None or elapsed>120 or short not in INPLAY_STATUSES: 
            continue
        
        fid=(m.get("fixture") or {}).get("id")
        log_enhanced.logger.debug(f"Processing match {fid}, minute {elapsed}")
        
        m["statistics"]=fetch_match_stats(fid); 
        m["events"]=fetch_match_events(fid)
        out.append(m)
    
    duration = time.time() - start_time
    log_enhanced.logger.info(
        f"Live matches processed: {len(out)} valid matches in {duration:.2f}s"
    )
    return out

# ───────── Feature extraction (lean) ─────────
def _num(v) -> float:
    try:
        if isinstance(v,str) and v.endswith("%"): 
            return float(v[:-1])
        return float(v or 0)
    except: 
        return 0.0

def _pos_pct(v) -> float:
    try: 
        return float(str(v).replace("%","").strip() or 0)
    except: 
        return 0.0

def extract_features(m: dict) -> Dict[str,float]:
    """Extract features with detailed logging"""
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

    red_h = red_a = 0
    for ev in (m.get("events") or []):
        if (ev.get("type","").lower()=="card"):
            d = (ev.get("detail","") or "").lower()
            t = (ev.get("team") or {}).get("name") or ""
            if "red" in d or "second yellow" in d:
                if t == home: 
                    red_h += 1
                elif t == away: 
                    red_a += 1

    features = {
        "minute": float(minute),
        "goals_h": float(gh), 
        "goals_a": float(ga),
        "goals_sum": float(gh + ga),
        "xg_h": float(xg_h), 
        "xg_a": float(xg_a),
        "xg_sum": float(xg_h + xg_a),
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
        "red_h": float(red_h), 
        "red_a": float(red_a), 
        "red_sum": float(red_h + red_a),
    }
    
    log_enhanced.logger.debug(
        f"Features extracted: match={home} vs {away}, minute={minute}, "
        f"goals={gh}-{ga}, xG={xg_h:.2f}-{xg_a:.2f}"
    )
    
    return features

# ───────── Model loading/scoring (OU 2.5 only) ─────────
EPS=1e-12
def _sigmoid(x: float) -> float:
    try:
        if x<-50: 
            return 1e-22
        if x>50:  
            return 1-1e-22
        import math; 
        return 1/(1+math.exp(-x))
    except: 
        return 0.5

def _logit(p: float) -> float:
    import math; 
    p=max(EPS,min(1-EPS,float(p))); 
    return math.log(p/(1-p))

def _linpred(feat: Dict[str,float], weights: Dict[str,float], intercept: float) -> float:
    s=float(intercept or 0.0)
    for k,w in (weights or {}).items(): 
        s += float(w or 0.0)*float(feat.get(k,0.0))
    return s

def _calibrate(p: float, cal: Dict[str,Any]) -> float:
    method=(cal or {}).get("method","sigmoid")
    a=float((cal or {}).get("a",1.0))
    b=float((cal or {}).get("b",0.0))
    
    if method.lower()=="platt": 
        return _sigmoid(a*_logit(p)+b)
    
    import math; 
    p=max(EPS,min(1-EPS,float(p)))
    z=math.log(p/(1-p))
    return _sigmoid(a*z+b)

def _score_prob(feat: Dict[str,float], mdl: Dict[str,Any]) -> float:
    """Score probability with learning system adjustments"""
    p=_sigmoid(_linpred(feat, mdl.get("weights",{}), float(mdl.get("intercept",0.0))))
    cal=mdl.get("calibration") or {}
    
    try: 
        if cal: 
            p=_calibrate(p, cal)
    except Exception as e:
        log_enhanced.log_error("Model calibration", e)
    
    # Apply learning system adjustments if available
    if learning_system and learning_system.performance_history["confidence_bias"]:
        confidence_bias = learning_system.performance_history["confidence_bias"]
        if abs(confidence_bias) > 0.05:
            # Adjust probability based on bias
            adjustment = -confidence_bias * 0.1  # Small adjustment
            p = max(0.0, min(1.0, p + adjustment))
            log_enhanced.logger.debug(
                f"Applied confidence bias adjustment: {adjustment:.4f}, "
                f"new prob: {p:.4f}"
            )
    
    return max(0.0, min(1.0, float(p)))

def _validate_model_blob(tmp: dict) -> bool:
    return isinstance(tmp, dict) and "weights" in tmp and "intercept" in tmp and isinstance(tmp.get("weights"), dict)

MODEL_KEYS_ORDER = ["model_v2:{name}", "model_latest:{name}", "model:{name}", "pre_{name}"]

def load_model_from_settings(name: str) -> Optional[Dict[str, Any]]:
    for pat in MODEL_KEYS_ORDER:
        raw=get_setting(pat.format(name=name))
        if not raw: 
            continue
        try:
            tmp=json.loads(raw)
            if _validate_model_blob(tmp):
                tmp.setdefault("intercept",0.0); 
                tmp.setdefault("weights",{})
                cal=tmp.get("calibration") or {}
                if isinstance(cal,dict):
                    cal.setdefault("method","sigmoid")
                    cal.setdefault("a",1.0)
                    cal.setdefault("b",0.0)
                    tmp["calibration"]=cal
                return tmp
        except Exception as e:
            log_enhanced.log_error(f"Loading model {name}", e)
            continue
    return None

def _load_ou25_model() -> Optional[Dict[str,Any]]:
    """Load OU 2.5 model with fallback"""
    model = load_model_from_settings("OU_2.5") or load_model_from_settings("O25")
    if model:
        log_enhanced.logger.info("OU 2.5 model loaded successfully")
    else:
        log_enhanced.logger.warning("OU 2.5 model not found")
    return model

def _ou25_live_odds_plausible(odds: Optional[float], minute: int, goals_sum: int) -> bool:
    """Quick plausibility guard for Over 2.5 prices in-play."""
    if odds is None:
        return False
    try:
        m = int(minute); g = int(goals_sum)
    except Exception:
        return True  # don't block if we can't parse

    # Very early 2–0 → price must be very short
    if g >= 2 and m <= 30:
        plausible = odds <= 1.30
        if not plausible:
            log_enhanced.logger.debug(
                f"Odds plausibility failed: 2-0 at {m}', odds {odds:.2f} > 1.30"
            )
        return plausible
    # Late first half 2–0 → even shorter
    if g >= 2 and m <= 45:
        plausible = odds <= 1.20
        if not plausible:
            log_enhanced.logger.debug(
                f"Odds plausibility failed: 2-0 at {m}', odds {odds:.2f} > 1.20"
            )
        return plausible
    # Early 1–0 → still fairly short on O2.5 in many leagues
    if g == 1 and m <= 20:
        plausible = odds <= 1.80
        if not plausible:
            log_enhanced.logger.debug(
                f"Odds plausibility failed: 1-0 at {m}', odds {odds:.2f} > 1.80"
            )
        return plausible

    return True

# ───────── Odds aggregation (OU 2.5 only) ─────────
def _market_name_normalize(s: str) -> str:
    s=(s or "").lower()
    if "over/under" in s or "total" in s or "goals" in s: 
        return "OU"
    return s

def _aggregate_price(vals: List[tuple[float,str]], prob_hint: Optional[float]) -> tuple[Optional[float], Optional[str]]:
    if not vals: 
        return None, None
    xs = sorted([o for (o,_) in vals if (o or 0) > 0])
    if not xs: 
        return None, None
    
    import statistics
    med = statistics.median(xs)
    filtered = [(o,b) for (o,b) in vals if o <= med * max(1.0, ODDS_OUTLIER_MULT)] or vals
    xs2 = sorted([o for (o,_) in filtered])
    med2 = statistics.median(xs2)
    
    if prob_hint is not None and prob_hint > 0:
        fair = 1.0 / max(1e-6, float(prob_hint))
        cap = fair * max(1.0, ODDS_FAIR_MAX_MULT)
        filtered = [(o,b) for (o,b) in filtered if o <= cap] or filtered
    
    if ODDS_AGGREGATION == "best":
        best = max(filtered, key=lambda t: t[0])
        return float(best[0]), str(best[1])
    
    target = med2
    pick = min(filtered, key=lambda t: abs(t[0] - target))
    return float(pick[0]), f"{pick[1]} (median of {len(xs)})"

def fetch_odds_ou25(fid: int, prob_hint_over: Optional[float] = None) -> Dict[str, Dict[str, Any]]:
    """Return {'Over': {'odds':x,'book':y}, 'Under': {...}} for OU 2.5"""
    log_enhanced.logger.debug(f"Fetching odds for fixture {fid}")
    
    js = _api_get(f"{BASE_URL}/odds/live", {"fixture": fid}) or {}
    if not (js.get("response") or []):
        js = _api_get(f"{BASE_URL}/odds", {"fixture": fid}) or {}
    
    by: Dict[str, List[tuple[float,str]]] = {"Over": [], "Under": []}
    book_counts = {}
    
    try:
        for r in js.get("response",[]) or []:
            for bk in (r.get("bookmakers") or []):
                book = bk.get("name") or "Book"
                book_counts[book] = book_counts.get(book, 0) + 1
                
                for mkt in (bk.get("bets") or []):
                    if _market_name_normalize(mkt.get("name","")) != "OU":
                        continue
                    
                    for v in (mkt.get("values") or []):
                        lbl = str(v.get("value") or "").lower()
                        if ("over" in lbl) or ("under" in lbl):
                            try:
                                ln = float(lbl.split()[-1])
                            except Exception:
                                continue
                            
                            if abs(ln - 2.5) > 1e-6:
                                continue
                            
                            if "over" in lbl:
                                by["Over"].append((float(v.get("odd") or 0), book))
                            elif "under" in lbl:
                                by["Under"].append((float(v.get("odd") or 0), book))
    except Exception as e:
        log_enhanced.log_error("Parsing odds", e)
    
    # require distinct books
    out: Dict[str, Dict[str, Any]] = {}
    for side, lst in by.items():
        distinct_books = len({b for (_, b) in lst})
        
        if distinct_books < max(1, ODDS_REQUIRE_N_BOOKS):
            log_enhanced.logger.debug(
                f"Odds insufficient for {side}: {distinct_books} books "
                f"(required {ODDS_REQUIRE_N_BOOKS})"
            )
            continue
        
        ag, label = _aggregate_price(
            lst, 
            prob_hint_over if side=="Over" else (1.0-(prob_hint_over or 0.0) if prob_hint_over is not None else None)
        )
        
        if ag is not None:
            out[side] = {"odds": float(ag), "book": label}
            log_enhanced.logger.debug(
                f"Odds aggregated for {side}: {ag:.2f} from {label}, "
                f"{len(lst)} quotes from {distinct_books} books"
            )
    
    # Log odds quality
    if out:
        quality_metric = len(out) * 10 + sum(len(by[s]) for s in out)
        log_enhanced.logger.info(
            f"Odds quality: fixture={fid}, quality={quality_metric}, "
            f"books_available={len(book_counts)}"
        )
    
    return out

# ───────── Helpers ─────────
def _league_name(m: dict) -> Tuple[int,str]:
    lg=(m.get("league") or {}) or {}
    return int(lg.get("id") or 0), f"{lg.get('country','')} - {lg.get('name','')}".strip(" -")

def _teams(m: dict) -> Tuple[str,str]:
    t=(m.get("teams") or {}) or {}
    return (t.get("home",{}).get("name",""), t.get("away",{}).get("name",""))

def _pretty_score(m: dict) -> str:
    gh=(m.get("goals") or {}).get("home") or 0
    ga=(m.get("goals") or {}).get("away") or 0
    return f"{gh}-{ga}"

def stats_coverage_ok(feat: Dict[str,float], minute: int) -> bool:
    if minute < REQUIRE_STATS_MINUTE:
        return True
    
    fields = [
        feat.get("xg_sum", 0.0),
        feat.get("sot_sum", 0.0),
        feat.get("cor_sum", 0.0),
        (feat.get("pos_h", 0.0) + feat.get("pos_a", 0.0)),
    ]
    nonzero = sum(1 for v in fields if (v or 0) > 0)
    
    coverage_ok = nonzero >= max(0, REQUIRE_DATA_FIELDS)
    if not coverage_ok:
        log_enhanced.logger.debug(
            f"Stats coverage insufficient: minute={minute}, "
            f"non_zero_fields={nonzero}/{len(fields)}"
        )
    
    return coverage_ok

def _parse_ou_line_from_suggestion(s: str) -> Optional[float]:
    try:
        import re
        m = re.search(r'(\d+\.?\d*)', s or "")
        return float(m.group(1)) if m else None
    except Exception:
        return None

def _ev(prob: float, odds: float) -> float:
    return prob*max(0.0, float(odds)) - 1.0

def _min_odds_for_market() -> float: 
    return MIN_ODDS_OU

def _candidate_is_sane_over(feat: Dict[str,float]) -> bool:
    """For Over 2.5: must not already be settled, avoid absurd states"""
    total = int(feat.get("goals_sum", 0))
    if total >= 3:   # already over line
        log_enhanced.logger.debug(f"Over candidate rejected: already {total} goals")
        return False
    return True

def _candidate_is_sane_under(feat: Dict[str,float], minute: int) -> bool:
    """Under 2.5: only if total <= 1 and minute sufficiently advanced"""
    total = int(feat.get("goals_sum", 0))
    if total > 1:
        log_enhanced.logger.debug(f"Under candidate rejected: {total} goals > 1")
        return False
    
    if minute < max(25, TIP_MIN_MINUTE):  # discourage too-early unders
        log_enhanced.logger.debug(
            f"Under candidate rejected: minute {minute} < {max(25, TIP_MIN_MINUTE)}"
        )
        return False
    
    # Red cards increase volatility → avoid when >=1
    if int(feat.get("red_sum", 0)) >= 1:
        log_enhanced.logger.debug("Under candidate rejected: red cards present")
        return False
    
    return True

# Stale-feed guard (lean)
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
            stats_by_team[tname] = {str((i.get("type") or "")).lower(): i.get("value") 
                                   for i in (s.get("statistics") or [])}

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
    cor_h = g(sh, ("corner kicks",))
    cor_a = g(sa, ("corner kicks",))
    gh = int(((m.get("goals") or {}).get("home") or 0) or 0)
    ga = int(((m.get("goals") or {}).get("away") or 0) or 0)
    n_events = len(m.get("events") or [])

    return (round(xg_h + xg_a, 3), int(sot_h + sot_a), int(cor_h + cor_a), gh, ga, n_events)

def is_feed_stale(fid: int, m: dict, minute: int) -> bool:
    if not STALE_GUARD_ENABLE:
        return False
    
    now = time.time()
    if minute < 10:
        _FEED_STATE[fid] = {"fp": _match_fingerprint(m), "last_change": now, "last_minute": minute}
        return False
    
    fp = _match_fingerprint(m)
    st = _FEED_STATE.get(fid)
    
    if st is None:
        _FEED_STATE[fid] = {"fp": fp, "last_change": now, "last_minute": minute}
        return False
    
    if fp != st.get("fp"):
        st["fp"] = fp
        st["last_change"] = now
        st["last_minute"] = minute
        log_enhanced.logger.debug(f"Feed updated for match {fid}")
        return False
    
    last_min = int(st.get("last_minute") or 0)
    st["last_minute"] = minute
    
    if minute > last_min and (now - float(st.get("last_change") or now)) >= STALE_STATS_MAX_SEC:
        log_enhanced.logger.warning(f"Stale feed detected for match {fid}, minute {minute}")
        return True
    
    return False

# ───────── Production scan (OU 2.5 only) ─────────
def _get_market_threshold_ou25() -> float:
    """Get confidence threshold with learning system adjustments"""
    if learning_system:
        performance = learning_system.analyze_tip_performance(7)  # Last 7 days
        threshold = learning_system.adjust_confidence_threshold(performance)
        log_enhanced.logger.info(f"Adjusted confidence threshold: {threshold:.2f}%")
        return threshold
    
    # Fallback to stored or default threshold
    v = get_setting("conf_threshold:Over/Under 2.5")
    try:
        return float(v) if v is not None else float(CONF_THRESHOLD)
    except Exception:
        return float(CONF_THRESHOLD)

def production_scan() -> Tuple[int, int]:
    """Main scan function with enhanced logging and learning"""
    scan_id = f"scan_{int(time.time())}_{hash(str(time.time())) % 10000:04d}"
    log_enhanced.log_scan_start(scan_id)
    start_time = time.time()
    
    try:
        matches = fetch_live_matches()
    except Exception as e:
        log_enhanced.log_error("fetch live matches", e)
        return (0, 0)
    
    live_seen = len(matches)
    if live_seen == 0:
        log_enhanced.log_scan_end(scan_id, 0, 0, time.time() - start_time)
        log.info("[SCAN] no live matches")
        return 0, 0

    mdl = _load_ou25_model()
    if not mdl:
        log_enhanced.log_scan_end(scan_id, 0, live_seen, time.time() - start_time)
        log.warning("[SCAN] OU 2.5 model missing in settings (keys: OU_2.5 / O25)")
        return 0, live_seen

    saved = 0
    threshold_pct = _get_market_threshold_ou25()
    now_ts = int(time.time())
    per_league_counter: dict[int,int] = {}
    
    # Get EV threshold from learning system
    if learning_system:
        performance = learning_system.analyze_tip_performance(7)
        ev_threshold = learning_system.adjust_ev_threshold(performance)
        ev_threshold_bps = int(ev_threshold * 10000)
        log_enhanced.logger.info(f"Using EV threshold: {ev_threshold_bps} bps")
    else:
        ev_threshold_bps = EDGE_MIN_BPS

    with db_conn() as c:
        for m in matches:
            try:
                fid = int((m.get("fixture") or {}).get("id") or 0)
                if not fid: 
                    continue

                feat = extract_features(m)
                minute = int(feat.get("minute", 0))

                if minute < TIP_MIN_MINUTE:
                    log_enhanced.log_match_processing(
                        fid, minute, _pretty_score(m), 0, "skipped", 
                        f"minute < {TIP_MIN_MINUTE}"
                    )
                    continue
                
                if is_feed_stale(fid, m, minute):
                    log_enhanced.log_match_processing(
                        fid, minute, _pretty_score(m), 0, "skipped", "stale feed"
                    )
                    continue
                
                if not stats_coverage_ok(feat, minute):
                    log_enhanced.log_match_processing(
                        fid, minute, _pretty_score(m), 0, "skipped", 
                        "insufficient stats"
                    )
                    continue

                # Model prob for OVER 2.5
                p_over = _score_prob(feat, mdl)
                p_under = 1.0 - p_over

                # Determine suggestion and probability
                suggestion = None
                prob = 0.0
                
                if p_over*100.0 >= threshold_pct and _candidate_is_sane_over(feat):
                    suggestion = "Over 2.5 Goals"
                    prob = p_over
                    decision_reason = f"Over confidence {p_over*100:.1f}% >= {threshold_pct}%"
                
                elif p_under*100.0 >= threshold_pct and _candidate_is_sane_under(feat, minute):
                    suggestion = "Under 2.5 Goals"
                    prob = p_under
                    decision_reason = f"Under confidence {p_under*100:.1f}% >= {threshold_pct}%"
                
                if not suggestion:
                    log_enhanced.log_match_processing(
                        fid, minute, _pretty_score(m), 
                        max(p_over, p_under)*100, "skipped", "confidence threshold"
                    )
                    continue

                # Odds/EV gate
                odds_map = fetch_odds_ou25(fid, prob_hint_over=p_over)
                side = "Over" if suggestion.startswith("Over") else "Under"
                rec = odds_map.get(side)
                
                if not rec:
                    if not ALLOW_TIPS_WITHOUT_ODDS:
                        log_enhanced.log_match_processing(
                            fid, minute, _pretty_score(m), prob*100, 
                            "skipped", "no odds available"
                        )
                        continue
                    odds = None
                    book = None
                    ev_pct = None
                else:
                    odds = float(rec["odds"])
                    book = rec["book"]
                    ev_pct = round(_ev(prob, odds)*100.0, 1)
                    
                    # Apply odds filters
                    if not (_min_odds_for_market() <= odds <= MAX_ODDS_ALL):
                        log_enhanced.log_match_processing(
                            fid, minute, _pretty_score(m), prob*100, 
                            "skipped", f"odds {odds:.2f} outside range"
                        )
                        continue
                    
                    if int(round(_ev(prob, odds)*10000)) < ev_threshold_bps:
                        log_enhanced.log_match_processing(
                            fid, minute, _pretty_score(m), prob*100, 
                            "skipped", f"EV {ev_pct:.1f}% < {ev_threshold_bps/100}%"
                        )
                        continue
                    
                    if suggestion.startswith("Over 2.5") and odds is not None:
                        if not _ou25_live_odds_plausible(float(odds), minute, int(feat.get("goals_sum", 0))):
                            log_enhanced.log_match_processing(
                                fid, minute, _pretty_score(m), prob*100, 
                                "skipped", "implausible odds"
                            )
                            continue

                league_id, league = _league_name(m)
                if PER_LEAGUE_CAP > 0 and per_league_counter.get(league_id, 0) >= PER_LEAGUE_CAP:
                    log_enhanced.log_match_processing(
                        fid, minute, _pretty_score(m), prob*100, 
                        "skipped", f"league cap {PER_LEAGUE_CAP} reached"
                    )
                    continue

                home, away = _teams(m)
                score = _pretty_score(m)
                conf_pct = round(prob*100.0, 1)
                created_ts = now_ts + saved
                tip_id = f"tip_{created_ts}_{fid}"

                # Store learning features
                learning_features = {
                    "features": feat,
                    "model_confidence_raw": prob,
                    "odds_quality": len(odds_map) if odds_map else 0,
                    "minute": minute,
                    "score": score
                }

                c.execute(
                    "INSERT INTO tips(match_id, league_id, league, home, away, market, "
                    "suggestion, confidence, confidence_raw, score_at_tip, minute, "
                    "created_ts, odds, book, ev_pct, sent_ok, learning_features) "
                    "VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 0, %s)",
                    (fid, league_id, league, home, away, "Over/Under 2.5", suggestion,
                     float(conf_pct), float(prob), score, minute, created_ts,
                     (float(odds) if odds is not None else None),
                     (book or None),
                     (float(ev_pct) if ev_pct is not None else None),
                     json.dumps(learning_features))
                )

                # Send Telegram
                message = _format_tip_message(home, away, league, minute, score, 
                                            suggestion, conf_pct, feat, odds, book, ev_pct)
                sent, telegram_tip_id = send_telegram(message)
                
                if sent:
                    c.execute("UPDATE tips SET sent_ok=1 WHERE match_id=%s AND created_ts=%s", 
                             (fid, created_ts))
                    
                    log_enhanced.log_tip_creation(
                        tip_id, fid, suggestion, conf_pct, 
                        odds if odds else 0.0, ev_pct if ev_pct else 0.0
                    )
                    
                    saved += 1
                    per_league_counter[league_id] = per_league_counter.get(league_id, 0) + 1
                    
                    log_enhanced.log_match_processing(
                        fid, minute, score, conf_pct, "tip_sent", decision_reason
                    )
                else:
                    log_enhanced.log_match_processing(
                        fid, minute, score, conf_pct, "failed", "telegram_send_failed"
                    )

                if MAX_TIPS_PER_SCAN and saved >= MAX_TIPS_PER_SCAN:
                    log_enhanced.logger.info(f"Scan tip limit reached: {saved} tips")
                    break

            except Exception as e:
                log_enhanced.log_error(f"Processing match {fid}", e)
                continue
    
    duration = time.time() - start_time
    log_enhanced.log_scan_end(scan_id, saved, live_seen, duration)
    log_enhanced.log_performance()
    
    log.info(f"[SCAN] saved={saved} live_seen={live_seen} duration={duration:.2f}s")
    
    # Trigger learning if tips were sent
    if saved > 0 and learning_system and PERFORMANCE_TRACKING:
        log_enhanced.logger.info("Triggering learning analysis")
        performance = learning_system.analyze_tip_performance(7)
        
        # Update model calibration if needed
        updated_model = learning_system.update_model_calibration(performance)
        if updated_model:
            # Save updated model to settings
            set_setting("model_latest:OU_2.5", json.dumps(updated_model))
            log_enhanced.logger.info("Model calibration updated based on recent performance")
        
        # Save performance snapshot
        with db_conn() as c:
            c.execute("""
                INSERT INTO performance_snapshots 
                (timestamp, period_days, total_tips, graded_tips, wins, accuracy,
                 avg_ev, confidence_bias, odds_bias, confidence_threshold, ev_threshold_bps)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                int(time.time()), 7,
                performance.get("total", 0),
                performance.get("sample_size", 0),
                performance.get("wins", 0),
                performance.get("accuracy", 0),
                performance.get("avg_ev", 0),
                performance.get("confidence_bias", 0),
                performance.get("odds_bias", 0),
                threshold_pct,
                ev_threshold_bps
            ))
    
    return saved, live_seen

def _format_tip_message(home, away, league, minute, score, suggestion, prob_pct, feat, odds=None, book=None, ev_pct=None):
    stat = ""
    if any([feat.get("xg_h",0),feat.get("xg_a",0),feat.get("sot_h",0),feat.get("sot_a",0),
            feat.get("cor_h",0),feat.get("cor_a",0),feat.get("pos_h",0),feat.get("pos_a",0),feat.get("red_sum",0)]):
        stat = (f"\n📊 xG {feat.get('xg_h',0):.2f}-{feat.get('xg_a',0):.2f}"
                f" • SOT {int(feat.get('sot_h',0))}-{int(feat.get('sot_a',0))}"
                f" • CK {int(feat.get('cor_h',0))}-{int(feat.get('cor_a',0))}")
        if feat.get("pos_h",0) or feat.get("pos_a",0): 
            stat += f" • POS {int(feat.get('pos_h',0))}%–{int(feat.get('pos_a',0))}%"
        if feat.get("red_sum",0):
            stat += f" • RED {int(feat.get('red_h',0))}-{int(feat.get('red_a',0))}"

    money = ""
    if odds:
        if ev_pct is not None:
            money = f"\n💰 <b>Odds:</b> {odds:.2f} @ {book or 'Book'}  •  <b>EV:</b> {ev_pct:+.1f}%"
        else:
            money = f"\n💰 <b>Odds:</b> {odds:.2f} @ {book or 'Book'}"
    
    # Add learning system insights if available
    learning_insight = ""
    if learning_system:
        performance = learning_system.get_performance_report()
        if performance.get("performance", {}).get("accuracy"):
            acc = performance["performance"]["accuracy"]
            learning_insight = f"\n🧠 <b>Recent Accuracy:</b> {acc:.1f}%"
    
    return ("⚽️ <b>OU2.5 Tip</b>\n"
            f"<b>Match:</b> {home} vs {away}\n"
            f"🕒 <b>Minute:</b> {minute}'  |  <b>Score:</b> {score}\n"
            f"<b>Tip:</b> {suggestion}\n"
            f"📈 <b>Confidence:</b> {prob_pct:.1f}%{money}{learning_insight}\n"
            f"🏆 <b>League:</b> {league}{stat}")

# ───────── Backfill results (used by digest) ─────────
def _fixture_by_id(mid: int) -> Optional[dict]:
    js=_api_get(FOOTBALL_API_URL, {"id": mid}) or {}
    arr=js.get("response") or [] if isinstance(js,dict) else []
    return arr[0] if arr else None

def _is_final(short: str) -> bool: 
    return (short or "").upper() in {"FT","AET","PEN"}

def backfill_results_for_open_matches(max_rows: int = 200) -> int:
    """Backfill results with learning integration"""
    log_enhanced.logger.info(f"Starting results backfill, max {max_rows} rows")
    
    now_ts=int(time.time())
    cutoff=now_ts - 14*24*3600
    updated=0
    
    with db_conn() as c:
        rows=c.execute("""
            WITH last AS (SELECT match_id, MAX(created_ts) last_ts FROM tips WHERE created_ts >= %s GROUP BY match_id)
            SELECT l.match_id FROM last l LEFT JOIN match_results r ON r.match_id=l.match_id
            WHERE r.match_id IS NULL ORDER BY l.last_ts DESC LIMIT %s
        """,(cutoff, max_rows)).fetchall()
    
    for (mid,) in rows:
        fx=_fixture_by_id(int(mid))
        if not fx: 
            continue
        
        st=(((fx.get("fixture") or {}).get("status") or {}).get("short") or "")
        if not _is_final(st): 
            continue
        
        g=fx.get("goals") or {}
        gh=int(g.get("home") or 0)
        ga=int(g.get("away") or 0)
        btts=1 if (gh>0 and ga>0) else 0
        
        with db_conn() as c2:
            c2.execute("""
                INSERT INTO match_results(match_id, final_goals_h, final_goals_a, btts_yes, updated_ts) 
                VALUES(%s,%s,%s,%s,%s) 
                ON CONFLICT(match_id) DO UPDATE SET 
                    final_goals_h=EXCLUDED.final_goals_h, 
                    final_goals_a=EXCLUDED.final_goals_a, 
                    btts_yes=EXCLUDED.btts_yes, 
                    updated_ts=EXCLUDED.updated_ts
            """, (int(mid), gh, ga, btts, int(time.time())))
        
        updated+=1
        
        # Trigger learning from this result
        if learning_system:
            # Get the tip for this match
            with db_conn() as c3:
                tip_row = c3.execute("""
                    SELECT suggestion, confidence_raw, minute, score_at_tip, learning_features
                    FROM tips 
                    WHERE match_id = %s 
                    ORDER BY created_ts DESC 
                    LIMIT 1
                """, (int(mid),)).fetchone()
                
                if tip_row:
                    suggestion, confidence_raw, minute, score, features_json = tip_row
                    tip_data = {
                        "suggestion": suggestion,
                        "confidence_raw": confidence_raw,
                        "minute": minute,
                        "score": score
                    }
                    
                    actual_outcome = {
                        "final_goals_h": gh,
                        "final_goals_a": ga
                    }
                    
                    learning_system.learn_from_mistake(int(mid), tip_data, actual_outcome)
    
    if updated: 
        log_enhanced.logger.info(f"[RESULTS] backfilled {updated} matches")
    
    return updated

# ───────── Daily digest (OU 2.5 focused) ─────────
def daily_accuracy_digest() -> Optional[str]:
    """Enhanced daily digest with learning insights"""
    log_enhanced.logger.info("Generating daily digest")
    
    today = datetime.now(BERLIN_TZ).date()
    start_of_day = datetime.combine(today, datetime.min.time(), tzinfo=BERLIN_TZ)
    start_ts = int(start_of_day.timestamp())

    backfill_results_for_open_matches(300)

    with db_conn() as c:
        rows = c.execute("""
            SELECT t.market, t.suggestion, t.confidence, t.confidence_raw, t.created_ts, t.odds,
                   r.final_goals_h, r.final_goals_a, r.btts_yes
            FROM tips t LEFT JOIN match_results r ON r.match_id=t.match_id
            WHERE t.created_ts >= %s AND t.market='Over/Under 2.5' AND t.sent_ok=1
            ORDER BY t.created_ts DESC
        """, (start_ts,)).fetchall()

    total = len(rows)
    graded = wins = 0
    roi_stake = 0.0
    roi_pnl = 0.0
    recent = []
    confidence_bias_sum = 0.0

    def _outcome(sugg: str, gh: int, ga: int) -> Optional[int]:
        total = gh + ga
        if sugg.startswith("Over"):
            if total > 2.5: 
                return 1
            if abs(total - 2.5) < 1e-9: 
                return None
            return 0
        else:
            if total < 2.5: 
                return 1
            if abs(total - 2.5) < 1e-9: 
                return None
            return 0

    for (mkt, sugg, conf, conf_raw, ts, odds, gh, ga, btts) in rows:
        tip_time = datetime.fromtimestamp(ts, BERLIN_TZ).strftime("%H:%M")
        recent.append(f"{sugg} ({conf:.1f}%) - {tip_time}")
        
        if gh is None or ga is None:
            continue
        
        res = _outcome(sugg, int(gh or 0), int(ga or 0))
        if res is None: 
            continue
        
        graded += 1
        if res == 1:
            wins += 1
        
        # Calculate confidence bias
        if conf_raw:
            outcome_prob = 1.0 if res == 1 else 0.0
            confidence_bias_sum += (outcome_prob - conf_raw)
        
        if odds:
            roi_stake += 1
            if res == 1:
                roi_pnl += float(odds) - 1.0
            else:
                roi_pnl -= 1.0

    # Learning system analysis
    learning_insights = ""
    if learning_system and graded >= MIN_LEARNING_SAMPLES:
        performance = learning_system.analyze_tip_performance(LEARNING_WINDOW_DAYS)
        report = learning_system.get_performance_report()
        
        # Adjust thresholds based on performance
        new_conf_threshold = learning_system.adjust_confidence_threshold(performance)
        new_ev_threshold = learning_system.adjust_ev_threshold(performance)
        
        learning_insights = (
            f"\n\n🧠 <b>Learning System Report</b>\n"
            f"• Adjusted confidence threshold: <b>{new_conf_threshold:.1f}%</b>\n"
            f"• Adjusted EV threshold: <b>{new_ev_threshold*10000:.0f} bps</b>\n"
            f"• Recent accuracy trend: <b>{report['history_summary']['accuracy_trend']}</b>\n"
            f"• Confidence bias: <b>{report['history_summary']['confidence_bias']:.3f}</b>"
        )
        
        if report.get("recommendations"):
            learning_insights += f"\n• Recommendations: {', '.join(report['recommendations'][:2])}"

    if graded == 0:
        msg = f"📊 OU2.5 Digest {today.strftime('%Y-%m-%d')}\nNo graded tips yet.{learning_insights}"
    else:
        acc = 100.0 * wins / max(1, graded)
        roi = (100.0 * roi_pnl / max(1.0, roi_stake)) if roi_stake > 0 else 0.0
        avg_confidence_bias = confidence_bias_sum / max(1, graded)
        
        msg = (f"📊 <b>OU 2.5 Digest</b> — {today.strftime('%Y-%m-%d')}\n"
               f"Tips sent: {total}  •  Graded: {graded}  •  Wins: {wins}  •  Accuracy: {acc:.1f}%\n"
               f"ROI: {roi:+.1f}%  •  Confidence Bias: {avg_confidence_bias:+.3f}")
        
        if recent:
            msg += f"\n🕒 Recent: {', '.join(recent[:5])}"
        
        msg += learning_insights
    
    send_telegram(msg)
    log_enhanced.logger.info(f"Daily digest sent: {total} tips, {graded} graded, {wins} wins")
    
    return msg

# ───────── Enhanced HTTP endpoints ─────────
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")
def _require_admin():
    key = request.headers.get("X-API-Key") or request.args.get("key") or ((request.json or {}).get("key") if request.is_json else None)
    if not ADMIN_API_KEY or key != ADMIN_API_KEY: 
        abort(401)

@app.route("/")
def root(): 
    """Enhanced root endpoint with system status"""
    status = {
        "ok": True, 
        "name": "goalsniper_ou25", 
        "only_market": "Over/Under 2.5", 
        "scheduler": RUN_SCHEDULER,
        "learning_enabled": LEARNING_ENABLED,
        "performance_tracking": PERFORMANCE_TRACKING,
        "log_level": log.getEffectiveLevel(),
        "timestamp": time.time()
    }
    return jsonify(status)

@app.route("/health")
def health():
    """Enhanced health check with detailed system status"""
    try:
        with db_conn() as c:
            n = c.execute("SELECT COUNT(*) FROM tips").fetchone()[0]
            learning_status = c.execute(
                "SELECT COUNT(*) FROM learning_history"
            ).fetchone()[0]
        
        api_ok = bool(_api_get(FOOTBALL_API_URL, {"live":"all"}))
        
        status = {
            "ok": True,
            "db": "ok",
            "tips_count": int(n),
            "learning_entries": int(learning_status),
            "api_connected": api_ok,
            "performance_stats": log_enhanced.performance_stats,
            "timestamp": time.time(),
            "memory_usage_mb": round(os.sys.getsizeof({}) / 1024 / 1024, 2)
        }
        
        if learning_system:
            status["learning_system"] = {
                "enabled": learning_system.learning_enabled,
                "confidence_bias": learning_system.performance_history["confidence_bias"],
                "performance_trend": learning_system._calculate_trend(
                    learning_system.performance_history["accuracy"]
                )
            }
        
        return jsonify(status)
    except Exception as e:
        log_enhanced.log_error("Health check", e)
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/init-db", methods=["POST"])
def http_init_db(): 
    _require_admin()
    init_db()
    return jsonify({"ok": True})

@app.route("/admin/scan", methods=["POST","GET"])
def http_scan(): 
    _require_admin()
    s,l=production_scan()
    return jsonify({"ok": True, "saved": s, "live_seen": l})

@app.route("/admin/backfill-results", methods=["POST","GET"])
def http_backfill(): 
    _require_admin()
    n=backfill_results_for_open_matches(400)
    return jsonify({"ok": True, "updated": n})

@app.route("/admin/digest", methods=["POST","GET"])
def http_digest(): 
    _require_admin()
    msg=daily_accuracy_digest()
    return jsonify({"ok": True, "sent": bool(msg)})

@app.route("/tips/latest")
def http_latest():
    limit=int(request.args.get("limit","50"))
    with db_conn() as c:
        rows=c.execute("""
            SELECT match_id,league,home,away,market,suggestion,confidence,
                   confidence_raw,score_at_tip,minute,created_ts,odds,book,ev_pct 
            FROM tips 
            WHERE market='Over/Under 2.5' 
            ORDER BY created_ts DESC 
            LIMIT %s
        """,(max(1,min(500,limit)),)).fetchall()
    
    tips=[]
    for r in rows:
        tips.append({
            "match_id":int(r[0]),
            "league":r[1],
            "home":r[2],
            "away":r[3],
            "market":r[4],
            "suggestion":r[5],
            "confidence":float(r[6]),
            "confidence_raw":(float(r[7]) if r[7] is not None else None),
            "score_at_tip":r[8],
            "minute":int(r[9]),
            "created_ts":int(r[10]),
            "odds": (float(r[11]) if r[11] is not None else None), 
            "book": r[12], 
            "ev_pct": (float(r[13]) if r[13] is not None else None)
        })
    
    return jsonify({"ok": True, "tips": tips})

@app.route("/admin/learning/status", methods=["GET"])
def http_learning_status():
    """Get learning system status"""
    _require_admin()
    
    if not learning_system:
        return jsonify({"ok": False, "error": "Learning system not initialized"}), 500
    
    report = learning_system.get_performance_report()
    
    return jsonify({
        "ok": True,
        "learning_enabled": LEARNING_ENABLED,
        "performance_report": report,
        "current_threshold": learning_system._get_current_threshold(),
        "performance_history_summary": {
            "accuracy_samples": len(learning_system.performance_history["accuracy"]),
            "ev_samples": len(learning_system.performance_history["ev"]),
            "latest_accuracy": learning_system.performance_history["accuracy"][-1] if learning_system.performance_history["accuracy"] else None,
            "latest_ev": learning_system.performance_history["ev"][-1] if learning_system.performance_history["ev"] else None
        }
    })

@app.route("/admin/learning/analyze", methods=["POST"])
def http_learning_analyze():
    """Trigger learning analysis"""
    _require_admin()
    
    if not learning_system:
        return jsonify({"ok": False, "error": "Learning system not initialized"}), 500
    
    days = request.json.get("days", LEARNING_WINDOW_DAYS) if request.is_json else LEARNING_WINDOW_DAYS
    performance = learning_system.analyze_tip_performance(days)
    
    # Apply learning adjustments
    new_conf_threshold = learning_system.adjust_confidence_threshold(performance)
    new_ev_threshold = learning_system.adjust_ev_threshold(performance)
    updated_model = learning_system.update_model_calibration(performance)
    
    result = {
        "ok": True,
        "performance_analysis": performance,
        "adjustments_made": {
            "confidence_threshold": new_conf_threshold,
            "ev_threshold": new_ev_threshold * 10000,  # Convert to bps
            "model_calibration_updated": updated_model is not None
        },
        "recommendations": learning_system._generate_recommendations(performance)
    }
    
    if updated_model:
        set_setting("model_latest:OU_2.5", json.dumps(updated_model))
        result["model_updated"] = True
    
    return jsonify(result)

@app.route("/admin/logs/level", methods=["POST"])
def http_set_log_level():
    """Set log level dynamically"""
    _require_admin()
    
    if request.is_json:
        level = request.json.get("level", "INFO").upper()
    else:
        level = request.args.get("level", "INFO").upper()
    
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if level not in valid_levels:
        return jsonify({"ok": False, "error": f"Invalid level. Must be one of: {valid_levels}"}), 400
    
    log.setLevel(getattr(logging, level))
    log_enhanced.logger.setLevel(getattr(logging, level))
    
    return jsonify({
        "ok": True, 
        "message": f"Log level set to {level}",
        "new_level": log.getEffectiveLevel()
    })

@app.route("/admin/performance", methods=["GET"])
def http_performance():
    """Get performance statistics"""
    _require_admin()
    
    with db_conn() as c:
        # Get recent performance
        recent_perf = c.execute("""
            SELECT timestamp, accuracy, avg_ev, confidence_bias, confidence_threshold
            FROM performance_snapshots 
            ORDER BY timestamp DESC 
            LIMIT 10
        """).fetchall()
    
    return jsonify({
        "ok": True,
        "system_stats": log_enhanced.performance_stats,
        "recent_performance": [
            {
                "timestamp": row[0],
                "accuracy": float(row[1]),
                "avg_ev": float(row[2]),
                "confidence_bias": float(row[3]),
                "confidence_threshold": float(row[4])
            }
            for row in recent_perf
        ]
    })

# ───────── Enhanced Scheduler ─────────
_scheduler_started=False
def _start_scheduler_once():
    global _scheduler_started
    if _scheduler_started or not RUN_SCHEDULER: 
        return
    
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        
        sched = BackgroundScheduler(timezone=TZ_UTC)
        
        # Main scan job
        sched.add_job(
            production_scan, 
            "interval", 
            seconds=SCAN_INTERVAL_SEC, 
            id="scan", 
            max_instances=1, 
            coalesce=True,
            misfire_grace_time=30
        )
        
        # Backfill job
        sched.add_job(
            lambda: backfill_results_for_open_matches(400), 
            "interval", 
            minutes=int(os.getenv("BACKFILL_EVERY_MIN", "15")), 
            id="backfill", 
            max_instances=1, 
            coalesce=True
        )
        
        # Daily digest job
        sched.add_job(
            daily_accuracy_digest, 
            "cron", 
            hour=int(os.getenv("DAILY_ACCURACY_HOUR", "3")), 
            minute=int(os.getenv("DAILY_ACCURACY_MINUTE", "6")), 
            id="digest", 
            max_instances=1, 
            coalesce=True, 
            timezone=BERLIN_TZ
        )
        
        # Learning analysis job (daily)
        if LEARNING_ENABLED:
            sched.add_job(
                lambda: learning_system.analyze_tip_performance(LEARNING_WINDOW_DAYS) if learning_system else None,
                "cron",
                hour=int(os.getenv("LEARNING_ANALYSIS_HOUR", "4")),
                minute=int(os.getenv("LEARNING_ANALYSIS_MINUTE", "0")),
                id="learning_analysis",
                max_instances=1,
                coalesce=True,
                timezone=BERLIN_TZ
            )
        
        sched.start()
        _scheduler_started = True
        
        startup_message = (
            "🚀 goalsniper OU 2.5–only mode started.\n"
            f"• Learning system: {'ENABLED' if LEARNING_ENABLED else 'DISABLED'}\n"
            f"• Scan interval: {SCAN_INTERVAL_SEC}s\n"
            f"• Confidence threshold: {CONF_THRESHOLD}%"
        )
        send_telegram(startup_message)
        
        log_enhanced.logger.info(
            f"[SCHED] started (scan={SCAN_INTERVAL_SEC}s, "
            f"learning={LEARNING_ENABLED})"
        )
        
    except Exception as e:
        log_enhanced.log_error("Scheduler startup", e, critical=True)

# ───────── Boot ─────────
def _on_boot():
    """Enhanced boot sequence"""
    log_enhanced.logger.info("Starting goalsniper OU 2.5 system...")
    
    try:
        _init_pool()
        init_db()
        
        # Initialize learning system
        global learning_system
        learning_system = LearningSystem(db_conn)
        
        _start_scheduler_once()
        
        log_enhanced.logger.info("Boot sequence completed successfully")
        
    except Exception as e:
        log_enhanced.log_error("Boot sequence", e, critical=True)
        raise

_on_boot()

if __name__ == "__main__":
    # Add signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        log_enhanced.logger.info(f"Received signal {signum}, shutting down gracefully...")
        
        # Stop scheduler if running
        global _scheduler_started
        if _scheduler_started:
            from apscheduler.schedulers.background import BackgroundScheduler
            sched = BackgroundScheduler()
            sched.shutdown()
            _scheduler_started = False
        
        # Close database pool
        if POOL:
            POOL.closeall()
        
        log_enhanced.logger.info("Shutdown complete")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run Flask app
    app.run(
        host=os.getenv("HOST", "0.0.0.0"), 
        port=int(os.getenv("PORT", "8080")),
        debug=os.getenv("FLASK_DEBUG", "0") == "1"
    )
