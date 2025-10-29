import os, json, time, logging, argparse, socket
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss, accuracy_score, log_loss,
    precision_score, roc_auc_score
)

try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger("trainer")

# ─────────── ENV CONFIGS ─────────── #

RECENCY_HALF_LIFE_DAYS = float(os.getenv("RECENCY_HALF_LIFE_DAYS", "120"))
RECENCY_MONTHS         = int(os.getenv("RECENCY_MONTHS", "36"))
MARKET_CUTOFFS_RAW     = os.getenv("MARKET_CUTOFFS", "BTTS=75,1X2=80,OU=88")
TIP_MAX_MINUTE_ENV     = os.getenv("TIP_MAX_MINUTE", "")
OU_TRAIN_LINES_RAW     = os.getenv("OU_TRAIN_LINES", os.getenv("OU_LINES", "2.5,3.5"))
TRAIN_MIN_MINUTE       = int(os.getenv("TRAIN_MIN_MINUTE", "15"))
TRAIN_TEST_SIZE        = float(os.getenv("TRAIN_TEST_SIZE", "0.25"))
MIN_ROWS               = int(os.getenv("MIN_ROWS", "150"))

AUTO_TUNE_ENABLE       = os.getenv("AUTO_TUNE_ENABLE", "0") not in ("0","false","False","no","NO")
TARGET_PRECISION       = float(os.getenv("TARGET_PRECISION", "0.60"))
THRESH_MIN_PREDICTIONS = int(os.getenv("THRESH_MIN_PREDICTIONS", "25"))
MIN_THRESH             = float(os.getenv("MIN_THRESH", "55"))
MAX_THRESH             = float(os.getenv("MAX_THRESH", "85"))

# ─────────── FEATURES ─────────── #

FEATURES = [
    "minute", "goals_h","goals_a","goals_sum","goals_diff",
    "xg_h","xg_a","xg_sum","xg_diff",
    "sot_h","sot_a","sot_sum",
    "sh_total_h","sh_total_a",
    "cor_h","cor_a","cor_sum",
    "pos_h","pos_a","pos_diff",
    "red_h","red_a","red_sum",
    "yellow_h","yellow_a",
    "goals_last_15", "shots_last_15", "cards_last_15",
    "pressure_home", "pressure_away", "score_advantage",
    "xg_momentum", "recent_xg_impact", "defensive_stability"
]

EPS = 1e-6

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
    TIP_MAX_MINUTE = int(float(TIP_MAX_MINUTE_ENV)) if TIP_MAX_MINUTE_ENV.strip() else None
except Exception:
    TIP_MAX_MINUTE = None

def _parse_ou_lines(raw: str) -> List[float]:
    vals = []
    for t in (raw or "").split(","):
        t = t.strip()
        if not t: continue
        try: vals.append(float(t))
        except: pass
    return vals or [2.5, 3.5]

def _fmt_line(line: float) -> str:
    return f"{line}".rstrip("0").rstrip(".")

import psycopg2
from psycopg2 import OperationalError
import requests

def _parse_pg_url(url: str) -> Dict[str, Any]:
    from urllib.parse import urlparse, parse_qsl
    pr = urlparse(url)
    if pr.scheme not in ("postgresql","postgres"):
        raise SystemExit("DATABASE_URL must start with postgresql:// or postgres://")
    return {
        "user": pr.username or "",
        "password": pr.password or "",
        "host": pr.hostname or "",
        "port": pr.port or 5432,
        "dbname": (pr.path or "").lstrip("/") or "postgres",
        "params": dict(parse_qsl(pr.query))
    }

def _q(v: str) -> str:
    s = "" if v is None else str(v)
    if s == "" or all(ch not in s for ch in (" ", "'", "\\", "\t", "\n")):
        return s
    return f"'{s.replace('\\','\\\\').replace(\"'\",\"\\'\")}'"

def _resolve_ipv4(host: str, port: int) -> Optional[str]:
    try:
        infos = socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_STREAM)
        for _, _, _, _, sockaddr in infos:
            ip, _ = sockaddr
            return ip
    except:
        return None

def _make_conninfo(parts: Dict[str, Any], port: int, hostaddr: Optional[str]) -> str:
    base = [f"host={_q(parts['host'])}", f"port={port}", f"dbname={_q(parts['dbname'])}"]
    if parts["user"]: base.append(f"user={_q(parts['user'])}")
    if parts["password"]: base.append(f"password={_q(parts['password'])}")
    if hostaddr: base.append(f"hostaddr={_q(hostaddr)}")
    base.append("sslmode=require")
    return " ".join(base)

def _conninfo_candidates(url: str) -> List[str]:
    parts = _parse_pg_url(url)
    env_hostaddr = os.getenv("DB_HOSTADDR")
    prefer_pooled = os.getenv("DB_PREFER_POOLED","1") not in ("0","false","False","no","NO")
    ports = [6543] if prefer_pooled else []
    if parts["port"] not in ports: ports.append(parts["port"])
    return [_make_conninfo(parts, p, env_hostaddr or _resolve_ipv4(parts["host"], p)) for p in ports]

def _connect(db_url: Optional[str]):
    url = db_url or os.getenv("DATABASE_URL")
    if not url: raise SystemExit("DATABASE_URL must be set.")
    for attempt in range(6):
        for dsn in _conninfo_candidates(url):
            try:
                conn = psycopg2.connect(dsn); conn.autocommit = True
                logger.info("[DB] Connected to %s", dsn.split()[0])
                return conn
            except OperationalError: continue
        time.sleep(1 << attempt)
    raise OperationalError("Failed to connect to DB after retries.")

def _read_sql(conn, sql: str, params: Tuple = ()) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn, params=params)

def _exec(conn, sql: str, params: Tuple = ()) -> None:
    with conn.cursor() as cur: cur.execute(sql, params)

def _set_setting(conn, key: str, value: str) -> None:
    _exec(conn, """
        INSERT INTO settings(key,value) VALUES(%s,%s)
        ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value
    """, (key, value))

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
    if total_xg <= 0.0: return 0.0
    return (total_goals - total_xg) / max(1.0, total_xg)

def _recent_xg_impact_from(feat: Dict[str, float], minute: float) -> float:
    if minute <= 0: return 0.0
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

def _build_live_features_from_snapshot(snap: dict) -> Optional[Dict[str, float]]:
    if not isinstance(snap, dict): return None
    minute = float(snap.get("minute", 0.0))
    gh = float(snap.get("gh", 0.0)); ga = float(snap.get("ga", 0.0))
    stat = snap.get("stat") or {}

    feat = {
        "minute": minute,
        "goals_h": gh, "goals_a": ga,
        "goals_sum": gh + ga, "goals_diff": gh - ga,
        "xg_h": float(stat.get("xg_h", 0.0)),
        "xg_a": float(stat.get("xg_a", 0.0)),
        "xg_sum": float(stat.get("xg_h", 0.0)) + float(stat.get("xg_a", 0.0)),
        "xg_diff": float(stat.get("xg_h", 0.0)) - float(stat.get("xg_a", 0.0)),
        "sot_h": float(stat.get("sot_h", 0.0)),
        "sot_a": float(stat.get("sot_a", 0.0)),
        "sot_sum": float(stat.get("sot_h", 0.0)) + float(stat.get("sot_a", 0.0)),
        "sh_total_h": float(stat.get("sh_total_h", 0.0)),
        "sh_total_a": float(stat.get("sh_total_a", 0.0)),
        "cor_h": float(stat.get("cor_h", 0.0)),
        "cor_a": float(stat.get("cor_a", 0.0)),
        "cor_sum": float(stat.get("cor_h", 0.0)) + float(stat.get("cor_a", 0.0)),
        "pos_h": float(stat.get("pos_h", 0.0)),
        "pos_a": float(stat.get("pos_a", 0.0)),
        "pos_diff": float(stat.get("pos_h", 0.0)) - float(stat.get("pos_a", 0.0)),
        "red_h": float(stat.get("red_h", 0.0)),
        "red_a": float(stat.get("red_a", 0.0)),
        "red_sum": float(stat.get("red_h", 0.0)) + float(stat.get("red_a", 0.0)),
        "yellow_h": float(stat.get("yellow_h", 0.0)),
        "yellow_a": float(stat.get("yellow_a", 0.0)),
        "goals_last_15": 0.0, "shots_last_15": 0.0, "cards_last_15": 0.0
    }

    feat["pressure_home"] = _calculate_pressure(feat, "home")
    feat["pressure_away"] = _calculate_pressure(feat, "away")
    feat["score_advantage"] = feat["goals_diff"]
    feat["xg_momentum"] = _calculate_xg_momentum(feat)
    feat["recent_xg_impact"] = _recent_xg_impact_from(feat, minute)
    feat["defensive_stability"] = _defensive_stability(feat)
    return feat

def _fmt_line(line: float) -> str:
    return f"{line}".rstrip("0").rstrip(".")

def load_live_data(conn, min_minute: int = 15, test_size: float = 0.25, min_rows: int = 150) -> Dict[str, Any]:
    query = """
    SELECT 
        t.match_id, t.created_ts, t.payload, 
        r.final_goals_h, r.final_goals_a, r.btts_yes,
        EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - to_timestamp(t.created_ts))) / (60*60*24) AS days_ago
    FROM tip_snapshots t
    LEFT JOIN match_results r ON t.match_id = r.match_id
    WHERE r.final_goals_h IS NOT NULL AND r.final_goals_a IS NOT NULL
    ORDER BY t.created_ts DESC LIMIT 10000
    """
    df = _read_sql(conn, query)
    if df.empty:
        logger.warning("No live data found")
        return {"X": None, "y": None, "markets": {}}

    df["recency_weight"] = np.exp(-np.abs(df["days_ago"]) / max(1.0, RECENCY_HALF_LIFE_DAYS))
    rows, weights = [], []
    targets = {"BTTS": [], "OU_2.5": [], "OU_3.5": [], "1X2_HOME": [], "1X2_AWAY": []}

    for _, row in df.iterrows():
        try:
            snap = json.loads(row["payload"]) if isinstance(row["payload"], str) else row["payload"]
            feat = _build_live_features_from_snapshot(snap)
            if not feat: continue
            if float(feat.get("minute", 0)) < min_minute: continue
            vec = [float(feat.get(f, 0.0)) for f in FEATURES]
            if sum(1 for v in vec if v != 0.0) < 5: continue
            rows.append(vec)
            weights.append(float(row["recency_weight"]))

            gh = int(row["final_goals_h"] or 0)
            ga = int(row["final_goals_a"] or 0)
            total = gh + ga
            btts = int(row["btts_yes"] or 0)

            targets["BTTS"].append(1 if btts == 1 else 0)
            targets["OU_2.5"].append(1 if total > 2.5 else 0)
            targets["OU_3.5"].append(1 if total > 3.5 else 0)
            if gh > ga: targets["1X2_HOME"].append(1); targets["1X2_AWAY"].append(0)
            elif ga > gh: targets["1X2_HOME"].append(0); targets["1X2_AWAY"].append(1)
            else: targets["1X2_HOME"].append(0); targets["1X2_AWAY"].append(0)
        except Exception as e:
            logger.debug("Live row error: %s", e)

    if not rows:
        logger.warning("No valid live features")
        return {"X": None, "y": None, "markets": {}}

    X = np.array(rows, dtype=float)
    W = np.array(weights, dtype=float)
    logger.info("Loaded %d live samples x %d features", X.shape[0], X.shape[1])
    return {"X": X, "y": targets, "weights": W, "features": FEATURES}

def load_prematch_data(conn, min_rows: int = 150) -> Dict[str, Any]:
    query = """
    SELECT p.match_id, p.created_ts, p.payload, 
           r.final_goals_h, r.final_goals_a, r.btts_yes,
           EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - to_timestamp(p.created_ts))) / (60*60*24) AS days_ago
    FROM prematch_snapshots p
    LEFT JOIN match_results r ON p.match_id = r.match_id
    WHERE r.final_goals_h IS NOT NULL AND r.final_goals_a IS NOT NULL
    ORDER BY p.created_ts DESC LIMIT 5000
    """
    df = _read_sql(conn, query)
    if df.empty:
        logger.warning("No prematch data")
        return {"X": None, "y": None, "markets": {}}

    df['recency_weight'] = np.exp(-np.abs(df['days_ago']) / max(1.0, RECENCY_HALF_LIFE_DAYS))
    feats_raw, weights, all_keys = [], [], {}
    y = {'PRE_BTTS': [], 'PRE_OU_2.5': [], 'PRE_OU_3.5': [], 'PRE_1X2_HOME': [], 'PRE_1X2_AWAY': []}

    for _, row in df.iterrows():
        try:
            payload = json.loads(row['payload']) if isinstance(row['payload'], str) else row['payload']
            feat = payload.get("feat") if payload else {}
            num_feat = {k: float(v) for k, v in feat.items() if isinstance(v, (int, float))}
            if not num_feat: continue
            feats_raw.append(num_feat)
            weights.append(float(row['recency_weight']))
            for k in num_feat.keys():
                all_keys[k] = all_keys.get(k, 0) + 1

            gh = int(row['final_goals_h'] or 0)
            ga = int(row['final_goals_a'] or 0)
            btts = int(row['btts_yes'] or 0)
            total = gh + ga

            y['PRE_BTTS'].append(1 if btts == 1 else 0)
            y['PRE_OU_2.5'].append(1 if total > 2.5 else 0)
            y['PRE_OU_3.5'].append(1 if total > 3.5 else 0)
            if gh > ga: y['PRE_1X2_HOME'].append(1); y['PRE_1X2_AWAY'].append(0)
            elif ga > gh: y['PRE_1X2_HOME'].append(0); y['PRE_1X2_AWAY'].append(1)
            else: y['PRE_1X2_HOME'].append(0); y['PRE_1X2_AWAY'].append(0)
        except Exception as e:
            logger.debug("Prematch row error: %s", e)

    if not feats_raw:
        logger.warning("No valid prematch features")
        return {"X": None, "y": None, "markets": {}}

    ordered_keys = sorted([k for k, c in all_keys.items() if c >= 10]) or sorted(all_keys.keys())
    X = np.array([[float(fr.get(k, 0.0)) for k in ordered_keys] for fr in feats_raw])
    W = np.array(weights)
    logger.info("Loaded %d prematch samples x %d features", X.shape[0], X.shape[1])
    return {"X": X, "y": y, "weights": W, "features": ordered_keys}

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score, precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
import lightgbm as lgb

def train_market_model(X, y, weights, features, market, test_size=0.25) -> Optional[Dict[str, Any]]:
    if X is None or len(y) == 0:
        logger.warning("No data for %s", market)
        return None
    y = np.array(y, dtype=int)
    if len(np.unique(y)) < 2:
        logger.warning("%s: only one class in labels", market)
        return None
    if X.shape[0] != y.shape[0]:
        n = min(X.shape[0], y.shape[0])
        X = X[:n]; y = y[:n]; weights = weights[:n] if weights is not None else None

    # Split
    strat = y if len(np.unique(y)) > 1 else None
    Xtr, Xte, ytr, yte, wtr, wte = train_test_split(X, y, weights, test_size=test_size, stratify=strat, random_state=42)

    try:
        # LightGBM
        dtrain = lgb.Dataset(Xtr, label=ytr, weight=wtr, feature_name=features)
        dval = lgb.Dataset(Xte, label=yte, reference=dtrain, weight=wte)
        params = {
            "objective": "binary", "metric": "binary_logloss", "verbosity": -1,
            "boosting_type": "gbdt", "learning_rate": 0.05, "num_leaves": 16,
            "feature_pre_filter": False
        }
        booster = lgb.train(params, dtrain, num_boost_round=300, valid_sets=[dval],
                            early_stopping_rounds=20, verbose_eval=False)
        train_preds = booster.predict(Xtr)
        test_preds = booster.predict(Xte)
        model_info = {
            "type": "lightgbm",
            "booster": booster.dump_model(),
            "metrics": {
                "accuracy": float(accuracy_score(yte, test_preds > 0.5)),
                "precision": float(precision_score(yte, test_preds > 0.5, zero_division=0)),
                "auc": float(roc_auc_score(yte, test_preds)),
                "log_loss": float(log_loss(yte, test_preds)),
                "brier": float(brier_score_loss(yte, test_preds)),
            }
        }

    except Exception as e:
        logger.warning("LightGBM failed for %s: %s — using LogisticRegression", market, e)
        class_weight = 'balanced' if np.mean(y) < 0.3 or np.mean(y) > 0.7 else None
        model = LogisticRegression(C=0.1, max_iter=1000, class_weight=class_weight, random_state=42)
        model.fit(Xtr, ytr, sample_weight=wtr)
        train_preds = model.predict_proba(Xtr)[:, 1]
        test_preds = model.predict_proba(Xte)[:, 1]
        model_info = {
            "type": "logistic_regression",
            "weights": dict(zip(features, model.coef_[0])),
            "intercept": float(model.intercept_[0]),
            "metrics": {
                "accuracy": float(accuracy_score(yte, test_preds > 0.5)),
                "precision": float(precision_score(yte, test_preds > 0.5, zero_division=0)),
                "auc": float(roc_auc_score(yte, test_preds)),
                "log_loss": float(log_loss(yte, test_preds)),
                "brier": float(brier_score_loss(yte, test_preds)),
            }
        }

    # ─── Platt Calibration ─── #
    try:
        if len(yte) > 50 and len(np.unique(ytr)) >= 2:
            calibrator = LogisticRegression(max_iter=1000, C=1.0)
            calibrator.fit(train_preds.reshape(-1, 1), ytr)
            cal_preds = calibrator.predict_proba(test_preds.reshape(-1, 1))[:, 1]
            if brier_score_loss(yte, cal_preds) < brier_score_loss(yte, test_preds):
                model_info["calibration"] = {
                    "method": "sigmoid",
                    "a": float(calibrator.coef_[0][0]),
                    "b": float(calibrator.intercept_[0]),
                }
    except Exception as e:
        logger.debug("Platt calibration skipped for %s: %s", market, e)

    return model_info

def _set_setting(conn, key: str, value: str) -> None:
    _exec(conn,
        "INSERT INTO settings(key,value) VALUES(%s,%s) "
        "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
        (key, value)
    )

def _store_model(conn, key: str, model_obj: Dict[str, Any]) -> None:
    model_json = json.dumps(model_obj, separators=(",", ":"), ensure_ascii=False)
    _set_setting(conn, key, model_json)
    logger.info("Stored model: %s", key)

def train_models(db_url: Optional[str] = None, min_minute: int = 15, 
                 test_size: float = 0.25, min_rows: int = 150) -> Dict[str, Any]:
    logger.info("Starting full training sequence...")
    conn = _connect(db_url)
    _ensure_training_tables(conn)

    results = {"ok": True, "trained": {}, "errors": []}

    try:
        # Live
        live = load_live_data(conn, min_minute, test_size, min_rows)
        if live["X"] is not None:
            for market, y in live["y"].items():
                mdl = train_market_model(live["X"], y, live["weights"], live["features"], market, test_size)
                if mdl:
                    _store_model(conn, f"model_v2:{market}", mdl)
                    results["trained"][market] = True
                else:
                    results["errors"].append(f"Failed to train {market}")
                    results["trained"][market] = False

        # Prematch
        pre = load_prematch_data(conn, min_rows)
        if pre["X"] is not None:
            for market, y in pre["y"].items():
                mdl = train_market_model(pre["X"], y, pre["weights"], pre["features"], market, test_size)
                if mdl:
                    # Rename keys to match main.py loaders
                    if market == "PRE_BTTS":
                        key = "PRE_BTTS_YES"
                    elif market == "PRE_OU_2.5":
                        key = "PRE_OU_2.5"
                    elif market == "PRE_OU_3.5":
                        key = "PRE_OU_3.5"
                    elif market == "PRE_1X2_HOME":
                        key = "PRE_WLD_HOME"
                    elif market == "PRE_1X2_AWAY":
                        key = "PRE_WLD_AWAY"
                    else:
                        key = market
                    _store_model(conn, key, mdl)
                    results["trained"][market] = True
                else:
                    results["errors"].append(f"Failed to train pre {market}")
                    results["trained"][market] = False

        # Metadata
        meta = {
            "timestamp": time.time(),
            "live_samples": int(live["X"].shape[0]) if live["X"] is not None else 0,
            "prematch_samples": int(pre["X"].shape[0]) if pre["X"] is not None else 0,
            "trained_models": [k for k, v in results["trained"].items() if v]
        }
        _set_setting(conn, "training_metadata", json.dumps(meta))
        _set_setting(conn, "last_train_ts", str(int(time.time())))
        logger.info("Training complete. Models trained: %d", len(meta["trained_models"]))
    except Exception as e:
        logger.exception("Training failed")
        results["ok"] = False
        results["error"] = str(e)
    finally:
        try: conn.close()
        except: pass
    return results

def _cli_main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-url", help="Postgres DSN or use env DATABASE_URL")
    ap.add_argument("--min-minute", type=int, default=TRAIN_MIN_MINUTE)
    ap.add_argument("--test-size", type=float, default=TRAIN_TEST_SIZE)
    ap.add_argument("--min-rows", type=int, default=MIN_ROWS)
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
