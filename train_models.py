import argparse, json, os, logging, time, socket
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg2
from psycopg2 import OperationalError

from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    brier_score_loss, accuracy_score, log_loss, precision_score,
    roc_auc_score, precision_recall_curve
)

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger("trainer")

# ───────────────────────── Feature sets (enhanced to match main.py) ───────────────────────── #

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
    # Enhanced features from main.py
    "goals_last_15", "shots_last_15", "cards_last_15",
    "pressure_home", "pressure_away", "score_advantage",
    "xg_momentum", "recent_xg_impact", "defensive_stability"
]

PRE_FEATURES: List[str] = [
    "pm_ov25_h","pm_ov35_h","pm_btts_h",
    "pm_ov25_a","pm_ov35_a","pm_btts_a",
    "pm_ov25_h2h","pm_ov35_h2h","pm_btts_h2h",
    # Enhanced pre-match features
    "home_team_strength", "away_team_strength", "form_differential", 
    "h2h_advantage", "match_importance",
    # Structure compatibility
    "minute","goals_h","goals_a","goals_sum","goals_diff",
    "xg_h","xg_a","xg_sum","xg_diff","sot_h","sot_a","sot_sum",
    "sh_total_h","sh_total_a","cor_h","cor_a","cor_sum",
    "pos_h","pos_a","pos_diff","red_h","red_a","red_sum",
    "yellow_h","yellow_a",
]

EPS = 1e-6

# ─────────────────────── Env knobs (enhanced compatibility) ─────────────────────── #

RECENCY_HALF_LIFE_DAYS = float(os.getenv("RECENCY_HALF_LIFE_DAYS", "120"))
RECENCY_MONTHS         = int(os.getenv("RECENCY_MONTHS", "36"))
MARKET_CUTOFFS_RAW     = os.getenv("MARKET_CUTOFFS", "BTTS=75,1X2=80,OU=88")
TIP_MAX_MINUTE_ENV     = os.getenv("TIP_MAX_MINUTE", "")
OU_TRAIN_LINES_RAW     = os.getenv("OU_TRAIN_LINES", os.getenv("OU_LINES", "2.5,3.5"))
TRAIN_MIN_MINUTE       = int(os.getenv("TRAIN_MIN_MINUTE", "15"))
TRAIN_TEST_SIZE        = float(os.getenv("TRAIN_TEST_SIZE", "0.25"))
MIN_ROWS               = int(os.getenv("MIN_ROWS", "150"))

# Enhanced: Auto-tune configuration
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

# Enhanced: Format line function to match main.py
def _fmt_line(line: float) -> str:
    """Format line string to match main.py (remove trailing zeros)"""
    return f"{line}".rstrip("0").rstrip(".")

# ─────────────────────── DB utils (enhanced compatibility) ─────────────────────── #

import requests  # used only for DoH fallback

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
                logger.info("[DB] train_models connected with DSN: %s", dsn.replace("password=", "password=**** "))
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
        CREATE TABLE IF NOT EXISTS prematch_snapshots (
            match_id   BIGINT PRIMARY KEY,
            created_ts BIGINT,
            payload    TEXT
        )
    """)
    _exec(conn, "CREATE INDEX IF NOT EXISTS idx_pre_snap_ts ON prematch_snapshots (created_ts DESC)")

def _get_setting_json(conn, key: str) -> Optional[dict]:
    try:
        df = _read_sql(conn, "SELECT value FROM settings WHERE key=%s", (key,))
        if df.empty: return None
        return json.loads(df.iloc[0]["value"])
    except Exception:
        return None

# ─────────────────────── Enhanced Data Loading with main.py compatibility ─────────────────────── #

def load_live_data(conn, min_minute: int = 15, test_size: float = 0.25, min_rows: int = 150) -> Dict[str, Any]:
    """Load live data with enhanced feature compatibility"""
    
    # Enhanced query to include more recent data and better filtering
    query = """
    WITH snapshots AS (
        SELECT 
            t.match_id,
            t.created_ts,
            t.payload::json->'features' as features,
            t.payload::json->'match' as match_data,
            r.final_goals_h,
            r.final_goals_a,
            r.btts_yes,
            EXTRACT(EPOCH FROM (to_timestamp(t.created_ts) - CURRENT_TIMESTAMP)) / (60*60*24) as days_ago
        FROM tip_snapshots t
        LEFT JOIN match_results r ON t.match_id = r.match_id
        WHERE t.payload::json->'features'->>'minute' IS NOT NULL
          AND (t.payload::json->'features'->>'minute')::float >= %s
          AND r.final_goals_h IS NOT NULL
          AND r.final_goals_a IS NOT NULL
        ORDER BY t.created_ts DESC
        LIMIT 10000
    )
    SELECT * FROM snapshots
    """
    
    df = _read_sql(conn, query, (min_minute,))
    
    if df.empty:
        logger.warning("No live data found for training")
        return {"X": None, "y": None, "markets": {}}
    
    # Enhanced: Calculate recency weights
    df['recency_weight'] = np.exp(-np.abs(df['days_ago']) / RECENCY_HALF_LIFE_DAYS)
    
    # Enhanced: Extract features with proper error handling
    feature_data = []
    market_targets = {
        'BTTS': [],
        'OU_2.5': [],
        'OU_3.5': [],
        '1X2_HOME': [],
        '1X2_AWAY': []
    }
    
    weights = []
    
    for _, row in df.iterrows():
        try:
            features = row['features']
            if not features:
                continue
                
            # Convert features to proper format
            feat_vec = []
            for f in FEATURES:
                value = features.get(f, 0.0)
                if value is None:
                    value = 0.0
                feat_vec.append(float(value))
            
            # Enhanced: Skip if we don't have enough valid features
            if sum(1 for x in feat_vec if x != 0) < 5:
                continue
                
            feature_data.append(feat_vec)
            weights.append(row['recency_weight'])
            
            # Enhanced: Calculate targets with better logic
            goals_h = row['final_goals_h']
            goals_a = row['final_goals_a']
            total_goals = goals_h + goals_a
            btts = row['btts_yes']
            
            # BTTS target
            market_targets['BTTS'].append(1 if btts == 1 else 0)
            
            # OU targets
            market_targets['OU_2.5'].append(1 if total_goals > 2.5 else 0)
            market_targets['OU_3.5'].append(1 if total_goals > 3.5 else 0)
            
            # 1X2 targets (draw suppressed)
            if goals_h > goals_a:
                market_targets['1X2_HOME'].append(1)
                market_targets['1X2_AWAY'].append(0)
            elif goals_a > goals_h:
                market_targets['1X2_HOME'].append(0)
                market_targets['1X2_AWAY'].append(1)
            else:
                # Skip draws for binary classification
                market_targets['1X2_HOME'].append(0)
                market_targets['1X2_AWAY'].append(0)
                
        except Exception as e:
            logger.debug("Error processing row: %s", e)
            continue
    
    if not feature_data:
        logger.warning("No valid feature data extracted")
        return {"X": None, "y": None, "markets": {}}
    
    X = np.array(feature_data)
    weights = np.array(weights)
    
    logger.info("Loaded %d live samples with %d features", X.shape[0], X.shape[1])
    
    return {
        "X": X,
        "y": market_targets,
        "weights": weights,
        "features": FEATURES
    }

def load_prematch_data(conn, min_rows: int = 150) -> Dict[str, Any]:
    """Load pre-match data with enhanced compatibility"""
    
    query = """
    WITH prematch_data AS (
        SELECT 
            p.match_id,
            p.created_ts,
            p.payload::json->'feat' as features,
            r.final_goals_h,
            r.final_goals_a,
            r.btts_yes,
            EXTRACT(EPOCH FROM (to_timestamp(p.created_ts) - CURRENT_TIMESTAMP)) / (60*60*24) as days_ago
        FROM prematch_snapshots p
        LEFT JOIN match_results r ON p.match_id = r.match_id
        WHERE p.payload::json->'feat' IS NOT NULL
          AND r.final_goals_h IS NOT NULL
          AND r.final_goals_a IS NOT NULL
        ORDER BY p.created_ts DESC
        LIMIT 5000
    )
    SELECT * FROM prematch_data
    """
    
    df = _read_sql(conn, query)
    
    if df.empty:
        logger.warning("No pre-match data found for training")
        return {"X": None, "y": None, "markets": {}}
    
    # Enhanced: Calculate recency weights
    df['recency_weight'] = np.exp(-np.abs(df['days_ago']) / RECENCY_HALF_LIFE_DAYS)
    
    feature_data = []
    market_targets = {
        'PRE_BTTS': [],
        'PRE_OU_2.5': [],
        'PRE_OU_3.5': [],
        'PRE_1X2_HOME': [],
        'PRE_1X2_AWAY': []
    }
    
    weights = []
    
    for _, row in df.iterrows():
        try:
            features = row['features']
            if not features:
                continue
                
            feat_vec = []
            for f in PRE_FEATURES:
                value = features.get(f, 0.0)
                if value is None:
                    value = 0.0
                feat_vec.append(float(value))
            
            # Enhanced: Skip if we don't have enough valid features
            if sum(1 for x in feat_vec if x != 0) < 3:
                continue
                
            feature_data.append(feat_vec)
            weights.append(row['recency_weight'])
            
            goals_h = row['final_goals_h']
            goals_a = row['final_goals_a']
            total_goals = goals_h + goals_a
            btts = row['btts_yes']
            
            # Pre-match targets
            market_targets['PRE_BTTS'].append(1 if btts == 1 else 0)
            market_targets['PRE_OU_2.5'].append(1 if total_goals > 2.5 else 0)
            market_targets['PRE_OU_3.5'].append(1 if total_goals > 3.5 else 0)
            
            # 1X2 targets (draw suppressed)
            if goals_h > goals_a:
                market_targets['PRE_1X2_HOME'].append(1)
                market_targets['PRE_1X2_AWAY'].append(0)
            elif goals_a > goals_h:
                market_targets['PRE_1X2_HOME'].append(0)
                market_targets['PRE_1X2_AWAY'].append(1)
            else:
                market_targets['PRE_1X2_HOME'].append(0)
                market_targets['PRE_1X2_AWAY'].append(0)
                
        except Exception as e:
            logger.debug("Error processing pre-match row: %s", e)
            continue
    
    if not feature_data:
        logger.warning("No valid pre-match feature data extracted")
        return {"X": None, "y": None, "markets": {}}
    
    X = np.array(feature_data)
    weights = np.array(weights)
    
    logger.info("Loaded %d pre-match samples with %d features", X.shape[0], X.shape[1])
    
    return {
        "X": X,
        "y": market_targets,
        "weights": weights,
        "features": PRE_FEATURES
    }

# ─────────────────────── Enhanced Model Training ─────────────────────── #

def train_market_model(X: np.ndarray, y: List[int], weights: np.ndarray, 
                      features: List[str], market: str, test_size: float = 0.25) -> Optional[Dict[str, Any]]:
    """Train a single market model with enhanced calibration"""
    
    if X is None or len(y) == 0:
        logger.warning("No data for market %s", market)
        return None
    
    y = np.array(y)
    
    # Enhanced: Filter out samples where target is ambiguous
    valid_indices = ~np.isnan(y)
    X_valid = X[valid_indices]
    y_valid = y[valid_indices]
    weights_valid = weights[valid_indices] if weights is not None else None
    
    if len(y_valid) < 100:
        logger.warning("Insufficient data for market %s: %d samples", market, len(y_valid))
        return None
    
    # Enhanced: Balance classes if needed
    pos_ratio = np.mean(y_valid)
    if pos_ratio < 0.1 or pos_ratio > 0.9:
        logger.info("Market %s has imbalanced classes: %.3f positive", market, pos_ratio)
    
    try:
        # Enhanced: Use class weights for imbalanced data
        class_weight = 'balanced' if pos_ratio < 0.3 or pos_ratio > 0.7 else None
        
        model = LogisticRegression(
            class_weight=class_weight,
            random_state=42,
            max_iter=1000,
            C=0.1  # Enhanced: Regularization
        )
        
        # Enhanced: Simple train/test split with weights
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            X_valid, y_valid, weights_valid, test_size=test_size, random_state=42, stratify=y_valid
        )
        
        model.fit(X_train, y_train, sample_weight=w_train)
        
        # Enhanced: Calibration using isotonic regression
        train_probs = model.predict_proba(X_train)[:, 1]
        test_probs = model.predict_proba(X_test)[:, 1]
        
        # Only calibrate if we have enough samples
        if len(y_test) > 50:
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(train_probs, y_train)
            calibrated_probs = calibrator.predict(test_probs)
            
            # Use calibration if it improves Brier score
            original_brier = brier_score_loss(y_test, test_probs)
            calibrated_brier = brier_score_loss(y_test, calibrated_probs)
            
            if calibrated_brier < original_brier:
                calibration = {
                    "method": "isotonic",
                    "a": 1.0,  # Placeholder for sigmoid params
                    "b": 0.0,
                    "calibrator": calibrator
                }
                logger.info("Market %s: calibration improved Brier score from %.4f to %.4f", 
                           market, original_brier, calibrated_brier)
            else:
                calibration = {"method": "sigmoid", "a": 1.0, "b": 0.0}
        else:
            calibration = {"method": "sigmoid", "a": 1.0, "b": 0.0}
        
        # Enhanced: Comprehensive metrics
        accuracy = accuracy_score(y_test, test_probs > 0.5)
        precision = precision_score(y_test, test_probs > 0.5, zero_division=0)
        auc = roc_auc_score(y_test, test_probs)
        logloss = log_loss(y_test, test_probs)
        
        logger.info("Market %s: accuracy=%.3f, precision=%.3f, AUC=%.3f, logloss=%.3f, samples=%d",
                   market, accuracy, precision, auc, logloss, len(y_valid))
        
        # Enhanced: Feature importance
        feature_importance = dict(zip(features, model.coef_[0]))
        
        return {
            "weights": feature_importance,
            "intercept": float(model.intercept_[0]),
            "calibration": calibration,
            "metrics": {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "auc": float(auc),
                "log_loss": float(logloss),
                "samples": len(y_valid),
                "positive_ratio": float(pos_ratio)
            }
        }
        
    except Exception as e:
        logger.error("Error training market %s: %s", market, e)
        return None

# ─────────────────────── Enhanced Auto-Tune Functionality ─────────────────────── #

def auto_tune_thresholds(conn, days: int = 14) -> Dict[str, float]:
    """Enhanced auto-tune thresholds based on recent performance"""
    
    if not AUTO_TUNE_ENABLE:
        logger.info("Auto-tune disabled by configuration")
        return {}
    
    logger.info("Starting auto-tune for thresholds (days=%d)", days)
    
    # Enhanced: Get recent tips with outcomes
    query = """
    SELECT 
        t.market,
        t.suggestion,
        t.confidence,
        r.final_goals_h,
        r.final_goals_a,
        r.btts_yes
    FROM tips t
    JOIN match_results r ON t.match_id = r.match_id
    WHERE t.created_ts >= extract(epoch from now() - interval '%s days')::bigint
      AND t.suggestion <> 'HARVEST'
      AND t.confidence IS NOT NULL
    ORDER BY t.created_ts DESC
    """
    
    df = _read_sql(conn, query, (days,))
    
    if df.empty:
        logger.warning("No recent tips found for auto-tuning")
        return {}
    
    # Enhanced: Calculate outcomes per market
    market_performance = {}
    
    for market in df['market'].unique():
        market_df = df[df['market'] == market]
        outcomes = []
        
        for _, row in market_df.iterrows():
            outcome = _calculate_tip_outcome(row)
            if outcome is not None:
                outcomes.append((row['confidence'], outcome))
        
        if len(outcomes) < THRESH_MIN_PREDICTIONS:
            continue
            
        # Enhanced: Find optimal threshold
        confidences, results = zip(*outcomes)
        best_threshold = _find_optimal_threshold(confidences, results, market)
        
        if best_threshold:
            market_performance[market] = best_threshold
            logger.info("Market %s: optimal threshold = %.1f%% (from %d samples)", 
                       market, best_threshold, len(outcomes))
    
    # Enhanced: Update settings
    tuned_markets = {}
    for market, threshold in market_performance.items():
        setting_key = f"conf_threshold:{market}"
        _set_setting(conn, setting_key, str(threshold))
        tuned_markets[market] = threshold
    
    logger.info("Auto-tune completed: tuned %d markets", len(tuned_markets))
    return tuned_markets

def _calculate_tip_outcome(row) -> Optional[int]:
    """Calculate if a tip was correct (enhanced from main.py)"""
    try:
        suggestion = row['suggestion']
        goals_h = row['final_goals_h']
        goals_a = row['final_goals_a']
        btts = row['btts_yes']
        total_goals = goals_h + goals_a
        
        if suggestion.startswith("Over"):
            line = float(suggestion.split()[1])
            return 1 if total_goals > line else 0
        elif suggestion.startswith("Under"):
            line = float(suggestion.split()[1])
            return 1 if total_goals < line else 0
        elif suggestion == "BTTS: Yes":
            return 1 if btts == 1 else 0
        elif suggestion == "BTTS: No":
            return 1 if btts == 0 else 0
        elif suggestion == "Home Win":
            return 1 if goals_h > goals_a else 0
        elif suggestion == "Away Win":
            return 1 if goals_a > goals_h else 0
        else:
            return None
    except Exception:
        return None

def _find_optimal_threshold(confidences: List[float], outcomes: List[int], market: str) -> Optional[float]:
    """Find optimal confidence threshold for a market"""
    
    confidences = np.array(confidences)
    outcomes = np.array(outcomes)
    
    # Enhanced: Try different thresholds
    thresholds = np.linspace(MIN_THRESH, MAX_THRESH, 50)
    best_score = -1
    best_threshold = None
    
    for threshold in thresholds:
        predictions = confidences >= threshold
        if np.sum(predictions) < 5:  # Need minimum predictions
            continue
            
        precision = np.mean(outcomes[predictions]) if np.sum(predictions) > 0 else 0
        recall = np.sum(outcomes[predictions]) / np.sum(outcomes) if np.sum(outcomes) > 0 else 0
        
        # Enhanced: F1-score balanced by precision importance
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
            # Prefer thresholds that meet target precision
            score = f1 * (1 + 0.5 * (precision >= TARGET_PRECISION))
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
    
    return best_threshold

# ─────────────────────── Main Training Function ─────────────────────── #

def train_models(db_url: Optional[str] = None, min_minute: int = 15, 
                test_size: float = 0.25, min_rows: int = 150) -> Dict[str, Any]:
    """Enhanced main training function with auto-tune integration"""
    
    logger.info("Starting enhanced model training (min_minute=%d, test_size=%.2f)", min_minute, test_size)
    
    conn = _connect(db_url)
    _ensure_training_tables(conn)
    
    results = {
        "ok": True,
        "trained": {},
        "auto_tuned": {},
        "errors": []
    }
    
    try:
        # Enhanced: Load both live and pre-match data
        live_data = load_live_data(conn, min_minute, test_size, min_rows)
        prematch_data = load_prematch_data(conn, min_rows)
        
        trained_models = {}
        
        # Enhanced: Train live models
        if live_data["X"] is not None:
            logger.info("Training live models...")
            for market in live_data["y"].keys():
                model_data = train_market_model(
                    live_data["X"], live_data["y"][market], 
                    live_data["weights"], live_data["features"], market, test_size
                )
                if model_data:
                    # Enhanced: Use main.py compatible model keys
                    if market.startswith("PRE_"):
                        model_key = market  # Keep pre-match keys as-is
                    else:
                        model_key = f"model_v2:{market}"
                    
                    trained_models[model_key] = model_data
                    results["trained"][market] = True
                    
                    # Enhanced: Store model in settings
                    model_json = json.dumps(model_data, indent=None)
                    _set_setting(conn, model_key, model_json)
                    logger.info("Stored model: %s", model_key)
                else:
                    results["trained"][market] = False
                    results["errors"].append(f"Failed to train {market}")
        
        # Enhanced: Train pre-match models
        if prematch_data["X"] is not None:
            logger.info("Training pre-match models...")
            for market in prematch_data["y"].keys():
                model_data = train_market_model(
                    prematch_data["X"], prematch_data["y"][market],
                    prematch_data["weights"], prematch_data["features"], market, test_size
                )
                if model_data:
                    # Enhanced: Use main.py compatible pre-match keys
                    model_key = market
                    trained_models[model_key] = model_data
                    results["trained"][market] = True
                    
                    model_json = json.dumps(model_data, indent=None)
                    _set_setting(conn, model_key, model_json)
                    logger.info("Stored pre-match model: %s", model_key)
                else:
                    results["trained"][market] = False
                    results["errors"].append(f"Failed to train pre-match {market}")
        
        # Enhanced: Auto-tune thresholds if enabled
        if AUTO_TUNE_ENABLE and trained_models:
            logger.info("Starting auto-tune process...")
            tuned_thresholds = auto_tune_thresholds(conn, 14)
            results["auto_tuned"] = tuned_thresholds
        
        # Enhanced: Store training metadata
        training_meta = {
            "timestamp": time.time(),
            "live_samples": live_data["X"].shape[0] if live_data["X"] is not None else 0,
            "prematch_samples": prematch_data["X"].shape[0] if prematch_data["X"] is not None else 0,
            "trained_models": list(trained_models.keys())
        }
        _set_setting(conn, "training_metadata", json.dumps(training_meta))
        
        logger.info("Training completed: %d models trained, %d markets auto-tuned", 
                   len(trained_models), len(results["auto_tuned"]))
        
    except Exception as e:
        logger.error("Training failed: %s", e)
        results["ok"] = False
        results["error"] = str(e)
    finally:
        conn.close()
    
    return results

# ─────────────────────── CLI ─────────────────────── #
def _cli_main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-url", help="Postgres DSN (or use env DATABASE_URL)")
    ap.add_argument("--min-minute", dest="min_minute", type=int, default=TRAIN_MIN_MINUTE)
    ap.add_argument("--test-size", type=float, default=TRAIN_TEST_SIZE)
    ap.add_argument("--min-rows", type=int, default=MIN_ROWS)
    ap.add_argument("--auto-tune", action="store_true", default=AUTO_TUNE_ENABLE, 
                   help="Enable auto-tuning of confidence thresholds")
    args = ap.parse_args()
    
    # Enhanced: Import the function directly to avoid circular import
    res = train_models(
        db_url=args.db_url or os.getenv("DATABASE_URL"),
        min_minute=args.min_minute, 
        test_size=args.test_size, 
        min_rows=args.min_rows
    )
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    _cli_main()
