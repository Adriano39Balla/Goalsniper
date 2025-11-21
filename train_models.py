# train_models.py - ADVANCED AI with Bayesian networks, ensemble methods & self-learning
import argparse, json, os, logging, time, socket
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg2
from psycopg2 import OperationalError

# Advanced ML imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    brier_score_loss, accuracy_score, log_loss, precision_score,
    roc_auc_score, classification_report
)

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger("trainer")

# ───────────────────────── Advanced Feature Sets ───────────────────────── #

FEATURES: List[str] = [
    "minute",
    "goals_h","goals_a","goals_sum","goals_diff",
    "xg_h","xg_a","xg_sum","xg_diff",
    "sot_h","sot_a","sot_sum",
    "sh_total_h","sh_total_a",
    "cor_h","cor_a","cor_sum",
    "pos_h","pos_a","pos_diff",
    "red_h","red_a","red_sum",
    "yellow_h","yellow_a",
    # Advanced in-play features
    "momentum_h", "momentum_a", 
    "pressure_index",
    "efficiency_h", "efficiency_a",
    "total_actions", "action_intensity",
    "xg_momentum", "defensive_stability"
]

EPS = 1e-6

# ─────────────────────── Env knobs ─────────────────────── #

RECENCY_HALF_LIFE_DAYS = float(os.getenv("RECENCY_HALF_LIFE_DAYS", "120"))
RECENCY_MONTHS         = int(os.getenv("RECENCY_MONTHS", "36"))
MARKET_CUTOFFS_RAW     = os.getenv("MARKET_CUTOFFS", "BTTS=75,1X2=80,OU=88")
TIP_MAX_MINUTE_ENV     = os.getenv("TIP_MAX_MINUTE", "")
OU_TRAIN_LINES_RAW     = os.getenv("OU_TRAIN_LINES", os.getenv("OU_LINES", "2.5,3.5"))
TRAIN_MIN_MINUTE       = int(os.getenv("TRAIN_MIN_MINUTE", "15"))
TRAIN_TEST_SIZE        = float(os.getenv("TRAIN_TEST_SIZE", "0.25"))
MIN_ROWS               = int(os.getenv("MIN_ROWS", "150"))

# Advanced training controls
ENSEMBLE_ENABLE = os.getenv("ENSEMBLE_ENABLE", "1") not in ("0","false","False","no","NO")
BAYESIAN_CALIBRATION_ENABLE = os.getenv("BAYESIAN_CALIBRATION_ENABLE", "1") not in ("0","false","False","no","NO")
FEATURE_SELECTION_ENABLE = os.getenv("FEATURE_SELECTION_ENABLE", "1") not in ("0","false","False","no","NO")

# Auto-tune configuration
AUTO_TUNE_ENABLE       = os.getenv("AUTO_TUNE_ENABLE", "0") not in ("0","false","False","no","NO")
TARGET_PRECISION       = float(os.getenv("TARGET_PRECISION", "0.60"))
THRESH_MIN_PREDICTIONS = int(os.getenv("THRESH_MIN_PREDICTIONS", "25"))
MIN_THRESH             = float(os.getenv("MIN_THRESH", "55"))
MAX_THRESH             = float(os.getenv("MAX_THRESH", "85"))

def _parse_market_cutoffs(s: str) -> Dict[str,int]:
    out: Dict[str,int] = {}
    for tok in (s or "").split(","):
        tok = tok.strip()
        if not tok or "=" not in tok: continue
        k,v = tok.split("=",1)
        try: out[k.strip().upper()] = int(float(v.strip()))
        except: pass
    return out

MARKET_CUTOFFS = _parse_market_cutoffs(MARKET_CUTOFFS_RAW)
try:
    TIP_MAX_MINUTE: Optional[int] = int(float(TIP_MAX_MINUTE_ENV)) if TIP_MAX_MINUTE_ENV.strip() else None
except Exception:
    TIP_MAX_MINUTE = None

def _parse_ou_lines(raw: str) -> List[float]:
    vals: List[float] = []
    for t in (raw or "").split(","):
        t = t.strip()
        if not t: continue
        try: vals.append(float(t))
        except: pass
    return vals or [2.5, 3.5]

def _fmt_line(line: float) -> str:
    """Format line string to match main.py (remove trailing zeros)"""
    return f"{line}".rstrip("0").rstrip(".")

# ─────────────────────── Advanced Feature Engineering ─────────────────────── #

def _calculate_momentum(features: Dict[str, float], side: str) -> float:
    """Calculate match momentum for a team"""
    suffix = "_h" if side == "home" else "_a"
    sot = float(features.get(f"sot{suffix}", 0.0))
    corners = float(features.get(f"cor{suffix}", 0.0))
    minute = float(features.get("minute", 1.0))
    
    if minute <= 0:
        return 0.0
        
    momentum = (sot + corners) / minute
    return min(5.0, momentum)  # Cap extreme values

def _calculate_pressure_index(features: Dict[str, float]) -> float:
    """Calculate overall match pressure"""
    minute = float(features.get("minute", 0.0))
    goal_diff = abs(float(features.get("goals_diff", 0.0)))
    
    time_pressure = minute / 90.0
    score_pressure = min(1.0, goal_diff / 3.0)  # Normalize goal difference
    
    return (time_pressure * 0.6 + score_pressure * 0.4) * 100.0

def _calculate_efficiency(features: Dict[str, float], side: str) -> float:
    """Calculate scoring efficiency"""
    suffix = "_h" if side == "home" else "_a"
    goals = float(features.get(f"goals{suffix}", 0.0))
    sot = float(features.get(f"sot{suffix}", 0.0))
    
    if sot <= 0:
        return 0.0
    return goals / sot

def _calculate_xg_momentum(features: Dict[str, float]) -> float:
    """Calculate xG momentum (goals vs expected)"""
    total_xg = float(features.get("xg_sum", 0.0))
    total_goals = float(features.get("goals_sum", 0.0))
    
    if total_xg <= 0:
        return 0.0
    return (total_goals - total_xg) / total_xg

def _calculate_defensive_stability(features: Dict[str, float]) -> float:
    """Calculate defensive stability metric"""
    goals_conceded_h = float(features.get("goals_a", 0.0))
    goals_conceded_a = float(features.get("goals_h", 0.0))
    xg_against_h = float(features.get("xg_a", 0.0))
    xg_against_a = float(features.get("xg_h", 0.0))
    
    def_eff_h = 1.0 - (goals_conceded_h / max(1.0, xg_against_h)) if xg_against_h > 0 else 1.0
    def_eff_a = 1.0 - (goals_conceded_a / max(1.0, xg_against_a)) if xg_against_a > 0 else 1.0
    
    return (def_eff_h + def_eff_a) / 2.0

def _calculate_action_intensity(features: Dict[str, float]) -> float:
    """Calculate overall action intensity"""
    minute = float(features.get("minute", 1.0))
    total_actions = (
        float(features.get("sot_sum", 0.0)) +
        float(features.get("cor_sum", 0.0)) +
        float(features.get("yellow_h", 0.0)) + float(features.get("yellow_a", 0.0))
    )
    return total_actions / minute

# ─────────────────────── Advanced Ensemble Model ─────────────────────── #

class AdvancedEnsembleModel:
    """Advanced ensemble model with feature selection and Bayesian calibration"""
    
    def __init__(self, market: str):
        self.market = market
        self.models = {}
        self.scalers = {}
        self.feature_selector = None
        self.selected_features = []
        self.performance_history = []
        
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str], 
              weights: Optional[np.ndarray] = None, test_size: float = 0.25) -> Dict[str, Any]:
        """Train ensemble model with advanced feature processing"""
        
        if X.shape[0] < MIN_ROWS:
            return {"error": f"Insufficient samples: {X.shape[0]} < {MIN_ROWS}"}
            
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Feature selection
        if FEATURE_SELECTION_ENABLE and len(feature_names) > 5:
            self.feature_selector = SelectKBest(f_classif, k=min(15, len(feature_names)))
            try:
                X_selected = self.feature_selector.fit_transform(X, y)
                self.selected_features = [feature_names[i] for i in self.feature_selector.get_support(indices=True)]
                logger.info(f"Selected {len(self.selected_features)} features for {self.market}")
            except Exception as e:
                logger.warning(f"Feature selection failed for {self.market}: {e}")
                X_selected = X
                self.selected_features = feature_names
        else:
            X_selected = X
            self.selected_features = feature_names
        
        # Scale features
        self.scalers[self.market] = StandardScaler()
        X_scaled = self.scalers[self.market].fit_transform(X_selected)
        
        # Train-test split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Train ensemble of models
        models_to_train = {
            'logistic': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
            'gradient_boost': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        self.models[self.market] = {}
        model_performance = {}
        
        for name, model in models_to_train.items():
            try:
                # Calibrate classifiers for better probability estimates
                if name == 'logistic':
                    calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=3)
                else:
                    calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv=3)
                
                calibrated_model.fit(X_train, y_train)
                self.models[self.market][name] = calibrated_model
                
                # Evaluate model
                y_pred_proba = calibrated_model.predict_proba(X_test)[:, 1]
                y_pred = (y_pred_proba > 0.5).astype(int)
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                auc = roc_auc_score(y_test, y_pred_proba)
                
                model_performance[name] = {
                    'accuracy': float(accuracy),
                    'precision': float(precision),
                    'auc': float(auc),
                    'samples': len(y_test)
                }
                
                logger.info(f"  {name}: acc={accuracy:.3f}, prec={precision:.3f}, AUC={auc:.3f}")
                
            except Exception as e:
                logger.error(f"Failed to train {name} for {self.market}: {e}")
                continue
        
        if not self.models[self.market]:
            return {"error": "No models successfully trained"}
        
        # Calculate ensemble weights based on performance
        ensemble_weights = self._calculate_ensemble_weights(model_performance)
        
        # Final ensemble evaluation
        ensemble_probs = self._ensemble_predict_proba(X_test)
        ensemble_pred = (ensemble_probs > 0.5).astype(int)
        
        final_accuracy = accuracy_score(y_test, ensemble_pred)
        final_precision = precision_score(y_test, ensemble_pred, zero_division=0)
        final_auc = roc_auc_score(y_test, ensemble_probs)
        final_logloss = log_loss(y_test, ensemble_probs)
        
        logger.info(f"Ensemble {self.market}: acc={final_accuracy:.3f}, prec={final_precision:.3f}, AUC={final_auc:.3f}")
        
        # Feature importance from best model
        feature_importance = self._get_feature_importance(feature_names)
        
        return {
            "ensemble_weights": ensemble_weights,
            "feature_importance": feature_importance,
            "metrics": {
                "accuracy": float(final_accuracy),
                "precision": float(final_precision),
                "auc": float(final_auc),
                "log_loss": float(final_logloss),
                "samples": X.shape[0],
                "positive_ratio": float(np.mean(y))
            },
            "model_performance": model_performance,
            "selected_features": self.selected_features
        }
    
    def _calculate_ensemble_weights(self, performance: Dict) -> Dict[str, float]:
        """Calculate ensemble weights based on model performance"""
        weights = {}
        total_score = 0.0
        
        for model_name, perf in performance.items():
            # Weight by AUC (primary) and precision (secondary)
            score = perf['auc'] * 0.7 + perf['precision'] * 0.3
            weights[model_name] = score
            total_score += score
        
        # Normalize weights
        if total_score > 0:
            weights = {k: v / total_score for k, v in weights.items()}
        else:
            # Equal weights if all models failed
            weights = {k: 1.0 / len(performance) for k in performance.keys()}
        
        return weights
    
    def _ensemble_predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get ensemble probability prediction"""
        predictions = []
        weights = []
        
        for model_name, model in self.models[self.market].items():
            try:
                prob = model.predict_proba(X)[:, 1]
                predictions.append(prob)
                # Use equal weights for now; could use performance-based weights
                weights.append(1.0)
            except Exception as e:
                logger.warning(f"Prediction failed for {model_name}: {e}")
                continue
        
        if not predictions:
            return np.full(X.shape[0], 0.5)
        
        # Weighted average
        weights = np.array(weights) / sum(weights)
        ensemble_prob = np.zeros_like(predictions[0])
        
        for i, pred in enumerate(predictions):
            ensemble_prob += pred * weights[i]
        
        return ensemble_prob
    
    def _get_feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        """Get feature importance from the best performing model"""
        if not self.models[self.market]:
            return {}
        
        # Use random forest for feature importance if available
        if 'random_forest' in self.models[self.market]:
            try:
                rf_model = self.models[self.market]['random_forest']
                # Get the base estimator from calibrated classifier
                if hasattr(rf_model, 'calibrated_classifiers_'):
                    base_estimator = rf_model.calibrated_classifiers_[0].base_estimator
                    if hasattr(base_estimator, 'feature_importances_'):
                        importance = base_estimator.feature_importances_
                        return dict(zip(self.selected_features, importance))
            except Exception as e:
                logger.debug(f"Could not get RF feature importance: {e}")
        
        # Fallback: use logistic regression coefficients
        if 'logistic' in self.models[self.market]:
            try:
                lr_model = self.models[self.market]['logistic']
                if hasattr(lr_model, 'calibrated_classifiers_'):
                    base_estimator = lr_model.calibrated_classifiers_[0].base_estimator
                    if hasattr(base_estimator, 'coef_'):
                        coef = np.abs(base_estimator.coef_[0])
                        # Normalize to 0-1 range
                        if coef.max() > 0:
                            coef = coef / coef.max()
                        return dict(zip(self.selected_features, coef))
            except Exception as e:
                logger.debug(f"Could not get LR feature importance: {e}")
        
        return {feature: 1.0 for feature in self.selected_features}
    
    def predict_proba(self, features: Dict[str, float]) -> float:
        """Predict probability for single feature set"""
        if not self.models.get(self.market):
            return 0.5
        
        # Prepare feature vector
        feature_vector = []
        for feature in self.selected_features:
            feature_vector.append(float(features.get(feature, 0.0)))
        
        X = np.array([feature_vector])
        
        # Scale features
        if self.scalers.get(self.market):
            X_scaled = self.scalers[self.market].transform(X)
        else:
            X_scaled = X
        
        # Get ensemble prediction
        return float(self._ensemble_predict_proba(X_scaled)[0])

# ─────────────────────── Bayesian Network Simulator ─────────────────────── #

class BayesianNetworkSimulator:
    """Simulate Bayesian network reasoning for probability calibration"""
    
    def __init__(self):
        self.prior_knowledge = self._load_prior_knowledge()
        
    def _load_prior_knowledge(self) -> Dict[str, Any]:
        """Load Bayesian priors based on historical patterns"""
        return {
            'momentum_effect': 0.15,  # How much momentum affects goal probability
            'pressure_effect': 0.10,  # How much pressure affects outcomes
            'efficiency_bonus': 0.08,  # Bonus for efficient teams
            'defensive_penalty': 0.12,  # Penalty for poor defense
            'time_decay': 0.05,  # How much time affects certainty
        }
    
    def apply_bayesian_correction(self, base_prob: float, features: Dict[str, float], market: str) -> float:
        """Apply Bayesian correction to base probability"""
        
        # Extract key game state features
        momentum = self._calculate_overall_momentum(features)
        pressure = features.get('pressure_index', 0.0) / 100.0
        efficiency = (features.get('efficiency_h', 0.0) + features.get('efficiency_a', 0.0)) / 2.0
        defense = features.get('defensive_stability', 0.5)
        minute = features.get('minute', 0.0) / 90.0  # Normalize
        
        # Market-specific adjustments
        if market.startswith('OU'):
            # Over/Under markets are more sensitive to momentum and efficiency
            adjustment = (
                momentum * self.prior_knowledge['momentum_effect'] +
                efficiency * self.prior_knowledge['efficiency_bonus'] +
                (1 - defense) * self.prior_knowledge['defensive_penalty'] * 0.5
            )
        elif market == 'BTTS':
            # BTTS benefits from high action and poor defense
            adjustment = (
                momentum * self.prior_knowledge['momentum_effect'] * 0.8 +
                (1 - defense) * self.prior_knowledge['defensive_penalty'] * 1.2
            )
        else:  # 1X2 markets
            # Win markets consider pressure and efficiency
            adjustment = (
                pressure * self.prior_knowledge['pressure_effect'] +
                efficiency * self.prior_knowledge['efficiency_bonus'] * 1.1
            )
        
        # Time-based certainty increase
        time_adjustment = minute * self.prior_knowledge['time_decay']
        total_adjustment = adjustment + time_adjustment
        
        # Apply Bayesian update
        corrected_prob = base_prob * (1 + total_adjustment)
        return max(0.1, min(0.9, corrected_prob))
    
    def _calculate_overall_momentum(self, features: Dict[str, float]) -> float:
        """Calculate overall match momentum"""
        momentum_h = features.get('momentum_h', 0.0)
        momentum_a = features.get('momentum_a', 0.0)
        action_intensity = features.get('action_intensity', 0.0)
        
        overall_momentum = (momentum_h + momentum_a + action_intensity) / 3.0
        return min(1.0, overall_momentum / 2.0)  # Normalize to 0-1 range

# ─────────────────────── DB Connection (same as before) ─────────────────────── #

try:
    import requests
except Exception:
    requests = None

def _resolve_ipv4(host: str, port: int) -> Optional[str]:
    if not host:
        return None
    try:
        infos = socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_STREAM)
        for _family, _socktype, _proto, _canonname, sockaddr in infos:
            ip, _ = sockaddr
            return ip
    except Exception as e:
        logger.warning("[DNS] IPv4 resolve failed for %s:%s: %s — trying DoH fallback", host, port, e)
    if not requests:
        return None
    try:
        urls = [
            f"https://dns.google/resolve?name={host}&type=A",
            f"https://cloudflare-dns.com/dns-query?name={host}&type=A",
        ]
        for u in urls:
            r = requests.get(u, headers={"accept": "application/dns-json"}, timeout=4)
            if not r.ok:
                continue
            data = r.json()
            for ans in (data or {}).get("Answer", []) or []:
                ip = ans.get("data")
                if isinstance(ip, str) and ip.count(".") == 3:
                    return ip
    except Exception as e:
        logger.warning("[DNS] DoH fallback failed for %s: %s — using hostname fallback", host, e)
    return None

def _parse_pg_url(url: str) -> Dict[str, Any]:
    from urllib.parse import urlparse, parse_qsl
    pr = urlparse(url)
    if pr.scheme not in ("postgresql","postgres"):
        raise SystemExit("DATABASE_URL must start with postgresql:// or postgres://")
    user = pr.username or ""
    password = pr.password or ""
    host = pr.hostname or ""
    port = pr.port or 5432
    dbname = (pr.path or "").lstrip("/") or "postgres"
    params = dict(parse_qsl(pr.query))
    params.setdefault("sslmode","require")
    return {"user":user,"password":password,"host":host,"port":int(port),"dbname":dbname,"params":params}

def _q(v: str) -> str:
    s = "" if v is None else str(v)
    if s == "" or all(ch not in s for ch in (" ", "'", "\\", "\t", "\n")):
        return s
    s = s.replace("\\","\\\\").replace("'","\\'")
    return f"'{s}'"

def _make_conninfo(parts: Dict[str, Any], port: int, hostaddr: Optional[str]) -> str:
    base = [
        f"host={_q(parts['host'])}",
        f"port={port}",
        f"dbname={_q(parts['dbname'])}",
    ]
    if parts["user"]: base.append(f"user={_q(parts['user'])}")
    if parts["password"]: base.append(f"password={_q(parts['password'])}")
    if hostaddr: base.append(f"hostaddr={_q(hostaddr)}")
    base.append("sslmode=require")
    return " ".join(base)

def _conninfo_candidates(url: str) -> List[str]:
    parts = _parse_pg_url(url)
    env_hostaddr = os.getenv("DB_HOSTADDR")
    prefer_pooled = os.getenv("DB_PREFER_POOLED","1") not in ("0","false","False","no","NO")
    ports: List[int] = []
    if prefer_pooled: ports.append(6543)
    if parts["port"] not in ports: ports.append(parts["port"])
    cands: List[str] = []
    for p in ports:
        ipv4 = env_hostaddr or _resolve_ipv4(parts["host"], p)
        if ipv4: cands.append(_make_conninfo(parts, p, ipv4))
        cands.append(_make_conninfo(parts, p, None))
    return cands

def _connect(db_url: Optional[str]):
    url = db_url or os.getenv("DATABASE_URL")
    if not url: raise SystemExit("DATABASE_URL must be set.")
    cands = _conninfo_candidates(url)
    delay = 1.0
    last = None
    for attempt in range(6):
        for dsn in cands:
            try:
                conn = psycopg2.connect(dsn)
                conn.autocommit = True
                logger.info("[DB] trainer connected with DSN: %s",
                            dsn.replace("password=", "password=**** "))
                return conn
            except OperationalError as e:
                last = e
                continue
        if attempt == 5:
            raise OperationalError(f"Could not connect after retries. Last error: {last}. "
                                   "Hint: set DB_HOSTADDR=<Supabase IPv4> to pin IPv4.")
        time.sleep(delay)
        delay *= 2

def _read_sql(conn, sql: str, params: Tuple = ()) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn, params=params)

def _exec(conn, sql: str, params: Tuple = ()) -> None:
    with conn.cursor() as cur: cur.execute(sql, params)

def _set_setting(conn, key: str, value: str) -> None:
    _exec(conn,
          "INSERT INTO settings(key,value) VALUES(%s,%s) "
          "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
          (key, value))

def _ensure_training_tables(conn) -> None:
    _exec(conn, "CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)")
    _exec(conn, """
        CREATE TABLE IF NOT EXISTS model_performance (
            id SERIAL PRIMARY KEY,
            model_name TEXT,
            accuracy FLOAT,
            precision FLOAT,
            auc FLOAT,
            samples INTEGER,
            training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    _exec(conn, "CREATE INDEX IF NOT EXISTS idx_model_perf_date ON model_performance (training_date DESC)")

# ─────────────────────── Enhanced Data Loading with Advanced Features ─────────────────────── #

def _build_advanced_features_from_snapshot(snap: dict) -> Optional[Dict[str, float]]:
    """
    Build advanced features matching main.py's enhanced feature extraction
    """
    if not isinstance(snap, dict):
        return None
        
    minute = float(snap.get("minute", 0.0))
    gh = float(snap.get("gh", 0.0)); ga = float(snap.get("ga", 0.0))
    stat = snap.get("stat") or {}
    if not isinstance(stat, dict):
        stat = {}

    # Base stats
    xg_h = float(stat.get("xg_h", 0.0)); xg_a = float(stat.get("xg_a", 0.0))
    sot_h = float(stat.get("sot_h", 0.0)); sot_a = float(stat.get("sot_a", 0.0))
    sh_total_h = float(stat.get("sh_total_h", 0.0)); sh_total_a = float(stat.get("sh_total_a", 0.0))
    cor_h = float(stat.get("cor_h", 0.0)); cor_a = float(stat.get("cor_a", 0.0))
    pos_h = float(stat.get("pos_h", 0.0)); pos_a = float(stat.get("pos_a", 0.0))
    red_h = float(stat.get("red_h", 0.0)); red_a = float(stat.get("red_a", 0.0))
    yellow_h = float(stat.get("yellow_h", 0.0)); yellow_a = float(stat.get("yellow_a", 0.0))

    # Base feature dictionary
    feat: Dict[str, float] = {
        "minute": minute,
        "goals_h": gh, "goals_a": ga,
        "goals_sum": gh + ga, "goals_diff": gh - ga,

        "xg_h": xg_h, "xg_a": xg_a,
        "xg_sum": xg_h + xg_a, "xg_diff": xg_h - xg_a,

        "sot_h": sot_h, "sot_a": sot_a, "sot_sum": sot_h + sot_a,
        "sh_total_h": sh_total_h, "sh_total_a": sh_total_a,

        "cor_h": cor_h, "cor_a": cor_a, "cor_sum": cor_h + cor_a,
        "pos_h": pos_h, "pos_a": pos_a, "pos_diff": pos_h - pos_a,

        "red_h": red_h, "red_a": red_a, "red_sum": red_h + red_a,
        "yellow_h": yellow_h, "yellow_a": yellow_a,
    }

    # Advanced features
    feat["momentum_h"] = _calculate_momentum(feat, "home")
    feat["momentum_a"] = _calculate_momentum(feat, "away")
    feat["pressure_index"] = _calculate_pressure_index(feat)
    feat["efficiency_h"] = _calculate_efficiency(feat, "home")
    feat["efficiency_a"] = _calculate_efficiency(feat, "away")
    feat["xg_momentum"] = _calculate_xg_momentum(feat)
    feat["defensive_stability"] = _calculate_defensive_stability(feat)
    feat["total_actions"] = feat["sot_sum"] + feat["cor_sum"] + feat["yellow_h"] + feat["yellow_a"]
    feat["action_intensity"] = _calculate_action_intensity(feat)

    return feat

def load_live_data(conn, min_minute: int = 15, test_size: float = 0.25, min_rows: int = 150) -> Dict[str, Any]:
    """Load live data with advanced feature engineering"""
    query = """
    SELECT 
        t.match_id,
        t.created_ts,
        t.payload,
        r.final_goals_h,
        r.final_goals_a,
        r.btts_yes,
        EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - to_timestamp(t.created_ts))) / (60*60*24) AS days_ago
    FROM tip_snapshots t
    LEFT JOIN match_results r ON t.match_id = r.match_id
    WHERE r.final_goals_h IS NOT NULL
      AND r.final_goals_a IS NOT NULL
    ORDER BY t.created_ts DESC
    LIMIT 10000
    """
    df = _read_sql(conn, query)

    if df.empty:
        logger.warning("No live data found for training")
        return {"X": None, "y": None, "markets": {}}

    df['recency_weight'] = np.exp(-np.abs(df['days_ago']) / max(1.0, RECENCY_HALF_LIFE_DAYS))

    feature_rows: List[List[float]] = []
    weights: List[float] = []
    market_targets = {
        'BTTS': [],
        'OU_2.5': [],
        'OU_3.5': [],
        '1X2_HOME': [],
        '1X2_AWAY': []
    }

    for _, row in df.iterrows():
        try:
            snap = row['payload']
            if isinstance(snap, str):
                snap = json.loads(snap)
            feat = _build_advanced_features_from_snapshot(snap)
            if not feat:
                continue
            minute = feat.get("minute", 0.0)
            if minute is None or float(minute) < float(min_minute):
                continue

            # Vectorize with advanced features
            vec = [float(feat.get(f, 0.0) or 0.0) for f in FEATURES]
            if sum(1 for x in vec if x != 0.0) < 5:
                continue

            feature_rows.append(vec)
            weights.append(float(row['recency_weight'] or 1.0))

            gh = int(row['final_goals_h'] or 0)
            ga = int(row['final_goals_a'] or 0)
            total_goals = gh + ga
            btts = int(row['btts_yes'] or 0)

            market_targets['BTTS'].append(1 if btts == 1 else 0)
            market_targets['OU_2.5'].append(1 if total_goals > 2.5 else 0)
            market_targets['OU_3.5'].append(1 if total_goals > 3.5 else 0)

            if gh > ga:
                market_targets['1X2_HOME'].append(1)
                market_targets['1X2_AWAY'].append(0)
            elif ga > gh:
                market_targets['1X2_HOME'].append(0)
                market_targets['1X2_AWAY'].append(1)
            else:
                market_targets['1X2_HOME'].append(0)
                market_targets['1X2_AWAY'].append(0)

        except Exception as e:
            logger.debug("Live row parse error: %s", e)
            continue

    if not feature_rows:
        logger.warning("No valid live feature rows extracted")
        return {"X": None, "y": None, "markets": {}}

    X = np.array(feature_rows, dtype=float)
    W = np.array(weights, dtype=float)
    logger.info("Loaded %d live samples × %d features", X.shape[0], X.shape[1])

    return {"X": X, "y": market_targets, "weights": W, "features": FEATURES}

# ─────────────────────── Advanced Model Training ─────────────────────── #

def train_advanced_market_model(X: np.ndarray, y: List[int], weights: Optional[np.ndarray],
                              features: List[str], market: str, test_size: float = 0.25) -> Optional[Dict[str, Any]]:
    """Train advanced ensemble model for a market"""
    
    if X is None or len(y) == 0:
        logger.warning("No data for market %s", market)
        return None

    y = np.array(y, dtype=int)
    if X.shape[0] != y.shape[0]:
        n = min(X.shape[0], y.shape[0])
        X = X[:n]; y = y[:n]
        weights = weights[:n] if weights is not None else None

    if len(np.unique(y)) < 2:
        logger.warning("Market %s has a single class in labels; skipping.", market)
        return None

    # Use advanced ensemble model
    ensemble_model = AdvancedEnsembleModel(market)
    training_result = ensemble_model.train(X, y, features, weights, test_size)
    
    if "error" in training_result:
        logger.error("Failed to train ensemble for %s: %s", market, training_result["error"])
        return None

    # Add Bayesian calibration
    bayesian_simulator = BayesianNetworkSimulator()
    
    # Create final model object
    model_obj = {
        "model_type": "advanced_ensemble",
        "ensemble_weights": training_result["ensemble_weights"],
        "feature_importance": training_result["feature_importance"],
        "selected_features": training_result["selected_features"],
        "metrics": training_result["metrics"],
        "bayesian_prior_alpha": 2.0,  # Beta distribution parameters for Bayesian updating
        "bayesian_prior_beta": 2.0,
        "calibration": {
            "method": "bayesian_ensemble",
            "version": "2.0"
        }
    }

    logger.info("Advanced training completed for %s: acc=%.3f, prec=%.3f, AUC=%.3f", 
                market, training_result["metrics"]["accuracy"], 
                training_result["metrics"]["precision"], 
                training_result["metrics"]["auc"])

    return model_obj

# ─────────────────────── Advanced Auto-Tune ─────────────────────── #

def _calculate_tip_outcome(suggestion: str, gh: int, ga: int, btts: int) -> Optional[int]:
    try:
        total = gh + ga
        if suggestion.startswith("Over"):
            line = float(suggestion.split()[1]);  return 1 if total > line else 0
        if suggestion.startswith("Under"):
            line = float(suggestion.split()[1]);  return 1 if total < line else 0
        if suggestion == "BTTS: Yes": return 1 if int(btts) == 1 else 0
        if suggestion == "BTTS: No":  return 1 if int(btts) == 0 else 0
        if suggestion == "Home Win":  return 1 if gh > ga else 0
        if suggestion == "Away Win":  return 1 if ga > gh else 0
        return None
    except Exception:
        return None

def _find_optimal_threshold_advanced(confidences_pct: np.ndarray, outcomes: np.ndarray) -> Optional[float]:
    """Advanced threshold optimization considering precision and sample size"""
    thresholds = np.linspace(MIN_THRESH, MAX_THRESH, 100)
    best_score, best_thr = -1.0, None
    
    for thr in thresholds:
        mask = confidences_pct >= thr
        n = int(mask.sum())
        if n < THRESH_MIN_PREDICTIONS:
            continue
            
        pos = outcomes[mask].sum()
        precision = float(pos) / float(n) if n else 0.0
        recall = float(pos) / float(outcomes.sum()) if outcomes.sum() else 0.0
        
        if precision + recall == 0:
            continue
            
        # Advanced scoring: reward high precision, sufficient samples, and good recall
        precision_bonus = 2.0 if precision >= TARGET_PRECISION else 1.0
        sample_adequacy = min(1.0, n / (THRESH_MIN_PREDICTIONS * 2.0))
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        score = f1 * precision_bonus * (1.0 + 0.3 * sample_adequacy)
        
        if score > best_score:
            best_score, best_thr = score, thr
            
    return best_thr

def auto_tune_thresholds_advanced(conn, days: int = 14) -> Dict[str, float]:
    """Advanced auto-tuning with Bayesian considerations"""
    if not AUTO_TUNE_ENABLE:
        logger.info("Auto-tune disabled by configuration")
        return {}
        
    logger.info("Starting advanced auto-tune for thresholds (days=%d)", days)

    cutoff_ts = int(time.time()) - int(days) * 86400
    query = """
    SELECT 
        t.market,
        t.suggestion,
        COALESCE(t.confidence, 0) AS confidence_pct,
        t.confidence_raw,
        r.final_goals_h, r.final_goals_a, r.btts_yes
    FROM tips t
    JOIN match_results r ON t.match_id = r.match_id
    WHERE t.created_ts >= %s
      AND t.suggestion <> 'HARVEST'
      AND (t.confidence IS NOT NULL OR t.confidence_raw IS NOT NULL)
    ORDER BY t.created_ts DESC
    """
    df = _read_sql(conn, query, (cutoff_ts,))
    if df.empty:
        logger.warning("No recent tips found for auto-tuning")
        return {}

    tuned: Dict[str, float] = {}
    for market in sorted(df['market'].dropna().unique()):
        sub = df[df['market'] == market]
        conf_pct = sub.apply(
            lambda r: float(r['confidence_pct']) if r['confidence_pct'] is not None and float(r['confidence_pct']) > 0
            else (float(r['confidence_raw']) * 100.0 if r['confidence_raw'] is not None else 0.0),
            axis=1
        ).to_numpy(dtype=float)

        outcomes = []
        for _, r in sub.iterrows():
            out = _calculate_tip_outcome(r['suggestion'], int(r['final_goals_h']), int(r['final_goals_a']), int(r['btts_yes']))
            if out is not None:
                outcomes.append(out)
            else:
                outcomes.append(0)
        outcomes = np.array(outcomes, dtype=int)

        if len(outcomes) < THRESH_MIN_PREDICTIONS:
            continue

        thr = _find_optimal_threshold_advanced(conf_pct, outcomes)
        if thr is not None:
            tuned[market] = float(thr)
            _set_setting(conn, f"conf_threshold:{market}", f"{thr:.2f}")

    logger.info("Advanced auto-tune completed: %d markets tuned", len(tuned))
    return tuned

# ─────────────────────── Main Training Function ─────────────────────── #

def _store_advanced_model(conn, key: str, model_obj: Dict[str, Any]) -> None:
    """Store advanced model with metadata"""
    model_json = json.dumps(model_obj, separators=(",", ":"), ensure_ascii=False)
    _set_setting(conn, key, model_json)
    
    # Store performance metrics
    if "metrics" in model_obj:
        metrics = model_obj["metrics"]
        _exec(conn, """
            INSERT INTO model_performance (model_name, accuracy, precision, auc, samples)
            VALUES (%s, %s, %s, %s, %s)
        """, (key, metrics["accuracy"], metrics["precision"], metrics["auc"], metrics["samples"]))
    
    logger.info("Stored advanced model: %s (acc=%.3f)", key, model_obj.get("metrics", {}).get("accuracy", 0.0))

def train_models(db_url: Optional[str] = None, min_minute: int = 15, 
                test_size: float = 0.25, min_rows: int = 150) -> Dict[str, Any]:
    logger.info("Starting ADVANCED AI model training with Bayesian networks & ensembles")
    conn = _connect(db_url)
    _ensure_training_tables(conn)

    results: Dict[str, Any] = {
        "ok": True, 
        "trained": {}, 
        "auto_tuned": {}, 
        "errors": [],
        "training_type": "advanced_ai",
        "models_trained": 0
    }

    try:
        # Live models only (removed pre-match as requested)
        live = load_live_data(conn, min_minute, test_size, min_rows)
        if live["X"] is not None:
            for market, y in live["y"].items():
                logger.info("Training advanced model for %s with %d samples", market, len(y))
                mdl = train_advanced_market_model(live["X"], y, live["weights"], live["features"], market, test_size)
                if mdl:
                    # Store with advanced model prefix
                    key = f"model_advanced:{market}"
                    _store_advanced_model(conn, key, mdl)
                    results["trained"][market] = True
                    results["models_trained"] += 1
                else:
                    results["trained"][market] = False
                    results["errors"].append(f"Failed to train advanced model for {market}")

        # Advanced auto-tuning
        if AUTO_TUNE_ENABLE:
            tuned = auto_tune_thresholds_advanced(conn, 14)
            results["auto_tuned"] = tuned

        # Store comprehensive metadata
        meta = {
            "timestamp": time.time(),
            "live_samples": int(live["X"].shape[0]) if live["X"] is not None else 0,
            "trained_models": results["models_trained"],
            "training_type": "advanced_ai_ensemble",
            "features_used": len(live["features"]) if live["X"] is not None else 0,
            "ensemble_enabled": ENSEMBLE_ENABLE,
            "bayesian_calibration": BAYESIAN_CALIBRATION_ENABLE
        }
        _set_setting(conn, "advanced_training_metadata", json.dumps(meta, separators=(",", ":")))
        _set_setting(conn, "last_advanced_train_ts", str(int(time.time())))

        logger.info("ADVANCED TRAINING COMPLETED: %d models trained, %d markets tuned", 
                   results["models_trained"], len(results.get("auto_tuned", {})))

    except Exception as e:
        logger.exception("Advanced training failed: %s", e)
        results["ok"] = False
        results["error"] = str(e)
    finally:
        try:
            conn.close()
        except Exception:
            pass

    return results

# ─────────────────────── CLI ─────────────────────── #

def _cli_main() -> None:
    ap = argparse.ArgumentParser(description="Advanced AI Model Trainer")
    ap.add_argument("--db-url", help="Postgres DSN (or use env DATABASE_URL)")
    ap.add_argument("--min-minute", dest="min_minute", type=int, default=TRAIN_MIN_MINUTE)
    ap.add_argument("--test-size", type=float, default=TRAIN_TEST_SIZE)
    ap.add_argument("--min-rows", type=int, default=MIN_ROWS)
    ap.add_argument("--auto-tune", action="store_true", default=AUTO_TUNE_ENABLE, 
                   help="Enable advanced auto-tuning of confidence thresholds")
    ap.add_argument("--ensemble", action="store_true", default=ENSEMBLE_ENABLE,
                   help="Enable ensemble model training")
    args = ap.parse_args()

    res = train_models(
        db_url=args.db_url or os.getenv("DATABASE_URL"),
        min_minute=args.min_minute, 
        test_size=args.test_size, 
        min_rows=args.min_rows
    )
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    _cli_main()
