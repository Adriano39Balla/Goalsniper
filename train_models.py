# train_models.py
import argparse, json, os, logging, time, socket
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg2
from psycopg2 import OperationalError

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss, accuracy_score, log_loss, precision_score,
    roc_auc_score
)

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger("trainer")

# ───────────────────────── Feature sets (match main.py live features) ───────────────────────── #

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
    # Enhanced live features your app uses in predictions:
    "goals_last_15", "shots_last_15", "cards_last_15",
    "pressure_home", "pressure_away", "score_advantage",
    "xg_momentum", "recent_xg_impact", "defensive_stability"
]

EPS = 1e-6

# ─────────────────────── Env knobs ─────────────────────── #

RECENCY_HALF_LIFE_DAYS = float(os.getenv("RECENCY_HALF_LIFE_DAYS", "120"))
RECENCY_MONTHS         = int(os.getenv("RECENCY_MONTHS", "36"))
# ✅ fixed: removed stray ')'
MARKET_CUTOFFS_RAW     = os.getenv("MARKET_CUTOFFS", "BTTS=75,1X2=80,OU=88")
TIP_MAX_MINUTE_ENV     = os.getenv("TIP_MAX_MINUTE", "")
OU_TRAIN_LINES_RAW     = os.getenv("OU_TRAIN_LINES", os.getenv("OU_LINES", "2.5,3.5"))
TRAIN_MIN_MINUTE       = int(os.getenv("TRAIN_MIN_MINUTE", "15"))
TRAIN_TEST_SIZE        = float(os.getenv("TRAIN_TEST_SIZE", "0.25"))
MIN_ROWS               = int(os.getenv("MIN_ROWS", "150"))

# Auto-tune configuration (kept compatible with main.py)
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

# ─────────────────────── Helpers mirroring main.py feature math ─────────────────────── #

def _calculate_pressure(feat: Dict[str, float], side: str) -> float:
    suffix = "_h" if side == "home" else "_a"
    possession = float(feat.get(f"pos{suffix}", 50.0))
    shots = float(feat.get(f"sot{suffix}", 0.0))
    xg = float(feat.get(f"xg{suffix}", 0.0))
    possession_norm = possession / 100.0
    shots_norm = min(shots / 10.0, 1.0)
    xg_norm = min(xg / 3.0, 1.0)
    return (possession_norm * 0.3 + shots_norm * 0.4 + xg_norm * 0.3) * 100.0

def _calculate_xg_momentum(feat: Dict[str, float]) -> float:
    total_xg = float(feat.get("xg_sum", 0.0))
    total_goals = float(feat.get("goals_sum", 0.0))
    if total_xg <= 0.0:
        return 0.0
    return (total_goals - total_xg) / max(1.0, total_xg)

def _recent_xg_impact_from(feat: Dict[str, float], minute: float) -> float:
    if minute <= 0:
        return 0.0
    xg_per_minute = float(feat.get("xg_sum", 0.0)) / minute
    return xg_per_minute * 90.0

def _defensive_stability(feat: Dict[str, float]) -> float:
    goals_conceded_h = float(feat.get("goals_a", 0.0))
    goals_conceded_a = float(feat.get("goals_h", 0.0))
    xg_against_h = float(feat.get("xg_a", 0.0))
    xg_against_a = float(feat.get("xg_h", 0.0))
    def_eff_h = 1 - (goals_conceded_h / max(1.0, xg_against_h)) if xg_against_h > 0 else 1.0
    def_eff_a = 1 - (goals_conceded_a / max(1.0, xg_against_a)) if xg_against_a > 0 else 1.0
    return (def_eff_h + def_eff_a) / 2.0

# ─────────────────────── DB utils (IPv4 pin if needed) ─────────────────────── #

try:
    import requests  # used only for DoH fallback
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
        CREATE TABLE IF NOT EXISTS prematch_snapshots (
            match_id   BIGINT PRIMARY KEY,
            created_ts BIGINT,
            payload    TEXT
        )
    """)
    _exec(conn, "CREATE INDEX IF NOT EXISTS idx_pre_snap_ts ON prematch_snapshots (created_ts DESC)")

# ─────────────────────── Enhanced Data Loading ─────────────────────── #

def _build_live_features_from_snapshot(snap: dict) -> Optional[Dict[str, float]]:
    """
    Your app saved HARVEST snapshots like:
      {
        "minute": 37,
        "gh": 1, "ga": 0,
        "stat": {
           "xg_h":..., "xg_a":..., "sot_h":..., ...,
           "pos_h":..., "pos_a":..., "red_h":..., "red_a":...,
           "sh_total_h":..., "sh_total_a":..., "yellow_h":..., "yellow_a":...
        }
      }
    Rebuild the exact feature dictionary used for training/prediction.
    """
    if not isinstance(snap, dict):
        return None
    minute = float(snap.get("minute", 0.0))
    gh = float(snap.get("gh", 0.0)); ga = float(snap.get("ga", 0.0))
    stat = snap.get("stat") or {}
    if not isinstance(stat, dict):
        stat = {}

    # base stats (default to 0.0)
    xg_h = float(stat.get("xg_h", 0.0)); xg_a = float(stat.get("xg_a", 0.0))
    sot_h = float(stat.get("sot_h", 0.0)); sot_a = float(stat.get("sot_a", 0.0))
    sh_total_h = float(stat.get("sh_total_h", 0.0)); sh_total_a = float(stat.get("sh_total_a", 0.0))
    cor_h = float(stat.get("cor_h", 0.0)); cor_a = float(stat.get("cor_a", 0.0))
    pos_h = float(stat.get("pos_h", 0.0)); pos_a = float(stat.get("pos_a", 0.0))
    red_h = float(stat.get("red_h", 0.0)); red_a = float(stat.get("red_a", 0.0))
    yellow_h = float(stat.get("yellow_h", 0.0)); yellow_a = float(stat.get("yellow_a", 0.0))

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

    # derived live features (no event stream → set last_15, cards to 0)
    feat["goals_last_15"] = 0.0
    feat["shots_last_15"] = 0.0
    feat["cards_last_15"] = 0.0
    feat["pressure_home"] = _calculate_pressure(feat, "home")
    feat["pressure_away"] = _calculate_pressure(feat, "away")
    feat["score_advantage"] = feat["goals_diff"]
    feat["xg_momentum"] = _calculate_xg_momentum(feat)
    feat["recent_xg_impact"] = _recent_xg_impact_from(feat, minute)
    feat["defensive_stability"] = _defensive_stability(feat)

    return feat

def load_live_data(conn, min_minute: int = 15, test_size: float = 0.25, min_rows: int = 150) -> Dict[str, Any]:
    """Load live data from tip_snapshots.payload and rebuild features to match main.py."""
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
            feat = _build_live_features_from_snapshot(snap)
            if not feat:
                continue
            minute = feat.get("minute", 0.0)
            if minute is None or float(minute) < float(min_minute):
                continue

            # vectorize
            vec = [float(feat.get(f, 0.0) or 0.0) for f in FEATURES]
            # need some signal
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

def load_prematch_data(conn, min_rows: int = 150) -> Dict[str, Any]:
    """
    Load pre-match data from prematch_snapshots.payload->feat.
    We dynamically build a feature list from available numeric keys so it stays compatible
    with whatever extract_prematch_features() saved in main.py.
    """
    query = """
    SELECT 
        p.match_id,
        p.created_ts,
        p.payload,
        r.final_goals_h,
        r.final_goals_a,
        r.btts_yes,
        EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - to_timestamp(p.created_ts))) / (60*60*24) AS days_ago
    FROM prematch_snapshots p
    LEFT JOIN match_results r ON p.match_id = r.match_id
    WHERE r.final_goals_h IS NOT NULL
      AND r.final_goals_a IS NOT NULL
    ORDER BY p.created_ts DESC
    LIMIT 5000
    """
    df = _read_sql(conn, query)

    if df.empty:
        logger.warning("No pre-match data found for training")
        return {"X": None, "y": None, "markets": {}}

    df['recency_weight'] = np.exp(-np.abs(df['days_ago']) / max(1.0, RECENCY_HALF_LIFE_DAYS))

    # First pass: collect all numeric keys under payload->feat
    all_keys: Dict[str, int] = {}
    feats_raw: List[Dict[str, float]] = []
    weights: List[float] = []
    y_dict = {
        'PRE_BTTS': [],
        'PRE_OU_2.5': [],
        'PRE_OU_3.5': [],
        'PRE_1X2_HOME': [],
        'PRE_1X2_AWAY': [],
    }

    for _, row in df.iterrows():
        try:
            payload = row['payload']
            if isinstance(payload, str):
                payload = json.loads(payload)
            feat = (payload or {}).get("feat") or {}
            # keep only numeric
            numeric_feat = {k: float(v) for k, v in (feat.items() if isinstance(feat, dict) else []) if isinstance(v, (int, float))}
            if not numeric_feat:
                continue
            feats_raw.append(numeric_feat)
            for k in numeric_feat.keys():
                all_keys[k] = all_keys.get(k, 0) + 1
            weights.append(float(row['recency_weight'] or 1.0))

            gh = int(row['final_goals_h'] or 0)
            ga = int(row['final_goals_a'] or 0)
            total = gh + ga
            btts = int(row['btts_yes'] or 0)

            y_dict['PRE_BTTS'].append(1 if btts == 1 else 0)
            y_dict['PRE_OU_2.5'].append(1 if total > 2.5 else 0)
            y_dict['PRE_OU_3.5'].append(1 if total > 3.5 else 0)
            if gh > ga:
                y_dict['PRE_1X2_HOME'].append(1); y_dict['PRE_1X2_AWAY'].append(0)
            elif ga > gh:
                y_dict['PRE_1X2_HOME'].append(0); y_dict['PRE_1X2_AWAY'].append(1)
            else:
                y_dict['PRE_1X2_HOME'].append(0); y_dict['PRE_1X2_AWAY'].append(0)

        except Exception as e:
            logger.debug("Prematch row parse error: %s", e)
            continue

    if not feats_raw:
        logger.warning("No valid prematch features extracted")
        return {"X": None, "y": None, "markets": {}}

    # choose stable feature order (only keys seen in at least 10 samples)
    ordered_keys = sorted([k for k, c in all_keys.items() if c >= 10]) or sorted(all_keys.keys())
    X = np.array([[float(fr.get(k, 0.0) or 0.0) for k in ordered_keys] for fr in feats_raw], dtype=float)
    W = np.array(weights, dtype=float)
    logger.info("Loaded %d prematch samples × %d features", X.shape[0], X.shape[1])

    return {"X": X, "y": y_dict, "weights": W, "features": ordered_keys}

# ─────────────────────── Model Training ─────────────────────── #

def _safe_train_test_split(X, y, W, test_size: float, stratify_ok: bool):
    from sklearn.model_selection import train_test_split
    # If single class, we cannot stratify
    y_unique = np.unique(y)
    if len(y_unique) < 2:
        return None
    if W is None:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42,
                                              stratify=y if stratify_ok else None)
        return Xtr, Xte, ytr, yte, None, None
    else:
        Xtr, Xte, ytr, yte, wtr, wte = train_test_split(X, y, W, test_size=test_size, random_state=42,
                                                        stratify=y if stratify_ok else None)
        return Xtr, Xte, ytr, yte, wtr, wte

def train_market_model(X: np.ndarray, y: List[int], weights: Optional[np.ndarray],
                      features: List[str], market: str, test_size: float = 0.25) -> Optional[Dict[str, Any]]:
    """Train a single market model with Platt (sigmoid) calibration."""
    if X is None or len(y) == 0:
        logger.warning("No data for market %s", market)
        return None

    y = np.array(y, dtype=int)
    if X.shape[0] != y.shape[0]:
        n = min(X.shape[0], y.shape[0])
        X = X[:n]; y = y[:n]
        weights = weights[:n] if weights is not None else None

    pos_ratio = float(np.mean(y))
    if len(np.unique(y)) < 2:
        logger.warning("Market %s has a single class in labels; skipping.", market)
        return None

    class_weight = 'balanced' if (pos_ratio < 0.3 or pos_ratio > 0.7) else None
    model = LogisticRegression(class_weight=class_weight, random_state=42, max_iter=1000, C=0.1)

    splits = _safe_train_test_split(X, y, weights, test_size, stratify_ok=True)
    if splits is None:
        logger.warning("Market %s: unable to split (single-class); skipping.", market)
        return None
    X_train, X_test, y_train, y_test, w_train, w_test = splits

    model.fit(X_train, y_train, sample_weight=w_train)

    train_probs = model.predict_proba(X_train)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]

    # Platt scaling
    calibration = {"method": "sigmoid", "a": 1.0, "b": 0.0}
    try:
        if len(y_test) > 50 and len(np.unique(y_train)) >= 2:
            from sklearn.linear_model import LogisticRegression as CalibrationLR
            calibrator = CalibrationLR(max_iter=1000, C=1.0)
            calibrator.fit(train_probs.reshape(-1, 1), y_train)
            calibrated = calibrator.predict_proba(test_probs.reshape(-1, 1))[:, 1]
            orig_brier = brier_score_loss(y_test, test_probs)
            cal_brier = brier_score_loss(y_test, calibrated)
            if cal_brier < orig_brier:
                calibration = {
                    "method": "sigmoid",
                    "a": float(calibrator.coef_[0][0]),
                    "b": float(calibrator.intercept_[0]),
                }
    except Exception as e:
        logger.debug("Calibration skipped for %s: %s", market, e)

    # Metrics on uncalibrated test_probs (just for reference)
    accuracy = accuracy_score(y_test, test_probs > 0.5)
    precision = precision_score(y_test, test_probs > 0.5, zero_division=0)
    auc = roc_auc_score(y_test, test_probs)
    logloss = log_loss(y_test, test_probs)

    logger.info("Market %s: acc=%.3f, prec=%.3f, AUC=%.3f, logloss=%.3f, samples=%d, pos=%.3f",
                market, accuracy, precision, auc, logloss, X.shape[0], pos_ratio)

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
            "samples": int(X.shape[0]),
            "positive_ratio": float(pos_ratio)
        }
    }

# ─────────────────────── Auto-Tune (DB-level, compatible with main.py) ─────────────────────── #

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

def _find_optimal_threshold(confidences_pct: np.ndarray, outcomes: np.ndarray) -> Optional[float]:
    thresholds = np.linspace(MIN_THRESH, MAX_THRESH, 50)
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
        f1 = 2 * precision * recall / (precision + recall)
        score = f1 * (1.0 + 0.5 * (precision >= TARGET_PRECISION))
        if score > best_score:
            best_score, best_thr = score, thr
    return best_thr

def auto_tune_thresholds(conn, days: int = 14) -> Dict[str, float]:
    if not AUTO_TUNE_ENABLE:
        logger.info("Auto-tune disabled by configuration")
        return {}
    logger.info("Starting auto-tune for thresholds (days=%d)", days)

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

        thr = _find_optimal_threshold(conf_pct, outcomes)
        if thr is not None:
            tuned[market] = float(thr)
            _set_setting(conn, f"conf_threshold:{market}", f"{thr:.2f}")

    logger.info("Auto-tune completed: %d markets tuned", len(tuned))
    return tuned

# ─────────────────────── Main Training Function ─────────────────────── #

def _store_model(conn, key: str, model_obj: Dict[str, Any]) -> None:
    model_json = json.dumps(model_obj, separators=(",", ":"), ensure_ascii=False)
    _set_setting(conn, key, model_json)
    logger.info("Stored model: %s", key)

def train_models(db_url: Optional[str] = None, min_minute: int = 15, 
                test_size: float = 0.25, min_rows: int = 150) -> Dict[str, Any]:
    logger.info("Starting enhanced model training (min_minute=%d, test_size=%.2f)", min_minute, test_size)
    conn = _connect(db_url)
    _ensure_training_tables(conn)

    results: Dict[str, Any] = {"ok": True, "trained": {}, "auto_tuned": {}, "errors": []}

    try:
        # Live models
        live = load_live_data(conn, min_minute, test_size, min_rows)
        if live["X"] is not None:
            for market, y in live["y"].items():
                mdl = train_market_model(live["X"], y, live["weights"], live["features"], market, test_size)
                if mdl:
                    # live models go under model_v2:{market}
                    key = f"model_v2:{market}"
                    _store_model(conn, key, mdl)
                    results["trained"][market] = True
                else:
                    results["trained"][market] = False
                    results["errors"].append(f"Failed to train {market}")

        # Pre-match models
        pre = load_prematch_data(conn, min_rows)
        if pre["X"] is not None:
            for market, y in pre["y"].items():
                mdl = train_market_model(pre["X"], y, pre["weights"], pre["features"], market, test_size)
                if mdl:
                    # map to keys main.py actually loads:
                    if market == "PRE_BTTS":
                        key = "PRE_BTTS_YES"
                    elif market == "PRE_OU_2.5":
                        key = "PRE_OU_" + _fmt_line(2.5)
                    elif market == "PRE_OU_3.5":
                        key = "PRE_OU_" + _fmt_line(3.5)
                    elif market == "PRE_1X2_HOME":
                        key = "PRE_WLD_HOME"
                    elif market == "PRE_1X2_AWAY":
                        key = "PRE_WLD_AWAY"
                    else:
                        key = market  # fallback
                    _store_model(conn, key, mdl)
                    results["trained"][market] = True
                else:
                    results["trained"][market] = False
                    results["errors"].append(f"Failed to train pre {market}")

        # Optional: auto-tune thresholds in DB
        if AUTO_TUNE_ENABLE:
            tuned = auto_tune_thresholds(conn, 14)
            results["auto_tuned"] = tuned

        # metadata
        meta = {
            "timestamp": time.time(),
            "live_samples": int(live["X"].shape[0]) if live["X"] is not None else 0,
            "prematch_samples": int(pre["X"].shape[0]) if pre["X"] is not None else 0,
            "trained_models": [k for k, v in results["trained"].items() if v]
        }
        _set_setting(conn, "training_metadata", json.dumps(meta, separators=(",", ":")))
        _set_setting(conn, "last_train_ts", str(int(time.time())))

        logger.info("Training done: %d live + %d prematch models trained; tuned=%d",
                    sum(1 for k in results["trained"] if not k.startswith("PRE_") and results["trained"][k]),
                    sum(1 for k in results["trained"] if k.startswith("PRE_") and results["trained"][k]),
                    len(results.get("auto_tuned", {})))

    except Exception as e:
        logger.exception("Training failed: %s", e)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-url", help="Postgres DSN (or use env DATABASE_URL)")
    ap.add_argument("--min-minute", dest="min_minute", type=int, default=TRAIN_MIN_MINUTE)
    ap.add_argument("--test-size", type=float, default=TRAIN_TEST_SIZE)
    ap.add_argument("--min-rows", type=int, default=MIN_ROWS)
    ap.add_argument("--auto-tune", action="store_true", default=AUTO_TUNE_ENABLE, 
                   help="Enable auto-tuning of confidence thresholds")
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
