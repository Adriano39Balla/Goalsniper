# train_models.py – advanced in-play trainer matching main.py

import argparse
import json
import logging
import os
import socket
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg2
from psycopg2 import OperationalError

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    roc_auc_score,
    log_loss,
)

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
)
log = logging.getLogger("trainer")

# ---------- Feature set (must match main.py.extract_features) ----------

FEATURES: List[str] = [
    "minute",
    "goals_h", "goals_a", "goals_sum", "goals_diff",
    "xg_h", "xg_a", "xg_sum", "xg_diff",
    "sot_h", "sot_a", "sot_sum",
    "sh_total_h", "sh_total_a",
    "cor_h", "cor_a", "cor_sum",
    "pos_h", "pos_a", "pos_diff",
    "red_h", "red_a", "red_sum",
    "yellow_h", "yellow_a",
    "momentum_h", "momentum_a",
    "pressure_index",
    "efficiency_h", "efficiency_a",
    "total_actions",
    "action_intensity",
]

EPS = 1e-6

# ---------- Env knobs (kept consistent with main.py) ----------

RECENCY_HALF_LIFE_DAYS = float(os.getenv("RECENCY_HALF_LIFE_DAYS", "120"))
RECENCY_MONTHS = int(os.getenv("RECENCY_MONTHS", "36"))

MARKET_CUTOFFS_RAW = os.getenv("MARKET_CUTOFFS", "BTTS=75,1X2=80,OU=88")
TIP_MAX_MINUTE_ENV = os.getenv("TIP_MAX_MINUTE", "")
OU_TRAIN_LINES_RAW = os.getenv("OU_TRAIN_LINES", os.getenv("OU_LINES", "2.5,3.5"))

TRAIN_MIN_MINUTE = int(os.getenv("TRAIN_MIN_MINUTE", "15"))
TRAIN_TEST_SIZE = float(os.getenv("TRAIN_TEST_SIZE", "0.25"))
MIN_ROWS = int(os.getenv("MIN_ROWS", "150"))

AUTO_TUNE_ENABLE = os.getenv("AUTO_TUNE_ENABLE", "0") not in (
    "0", "false", "False", "no", "NO"
)
TARGET_PRECISION = float(os.getenv("TARGET_PRECISION", "0.60"))
THRESH_MIN_PREDICTIONS = int(os.getenv("THRESH_MIN_PREDICTIONS", "25"))
MIN_THRESH = float(os.getenv("MIN_THRESH", "55"))
MAX_THRESH = float(os.getenv("MAX_THRESH", "85"))


def _parse_market_cutoffs(s: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for tok in (s or "").split(","):
        tok = tok.strip()
        if not tok or "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        try:
            out[k.strip().upper()] = int(float(v.strip()))
        except Exception:
            pass
    return out


MARKET_CUTOFFS = _parse_market_cutoffs(MARKET_CUTOFFS_RAW)

try:
    TIP_MAX_MINUTE: Optional[int] = (
        int(float(TIP_MAX_MINUTE_ENV)) if TIP_MAX_MINUTE_ENV.strip() else None
    )
except Exception:
    TIP_MAX_MINUTE = None


def _parse_ou_lines(raw: str) -> List[float]:
    vals: List[float] = []
    for t in (raw or "").split(","):
        t = t.strip()
        if not t:
            continue
        try:
            vals.append(float(t))
        except Exception:
            pass
    return vals or [2.5, 3.5]


def _fmt_line(line: float) -> str:
    return f"{line}".rstrip("0").rstrip(".")


OU_TRAIN_LINES: List[float] = _parse_ou_lines(OU_TRAIN_LINES_RAW)

# ---------- Advanced feature engineering (aligned with main.py) ----------


def _calculate_momentum(features: Dict[str, float], side: str) -> float:
    suffix = "_h" if side == "home" else "_a"
    sot = float(features.get(f"sot{suffix}", 0.0))
    cor = float(features.get(f"cor{suffix}", 0.0))
    minute = float(features.get("minute", 1.0))
    if minute <= 0:
        return 0.0
    return (sot + cor) / minute


def _calculate_pressure_index(features: Dict[str, float]) -> float:
    minute = float(features.get("minute", 0.0))
    goal_diff = abs(float(features.get("goals_diff", 0.0)))
    time_pressure = minute / 90.0
    score_pressure = min(1.0, goal_diff / 3.0)
    return time_pressure * 0.5 + score_pressure * 0.3 + 0.2


def _calculate_efficiency(features: Dict[str, float], side: str) -> float:
    suffix = "_h" if side == "home" else "_a"
    goals = float(features.get(f"goals{suffix}", 0.0))
    sot = float(features.get(f"sot{suffix}", 0.0))
    if sot <= 0:
        return 0.0
    return goals / sot


def _calculate_total_actions(features: Dict[str, float]) -> float:
    return (
        float(features.get("sot_h", 0.0))
        + float(features.get("sot_a", 0.0))
        + float(features.get("cor_h", 0.0))
        + float(features.get("cor_a", 0.0))
    )


def _calculate_action_intensity(features: Dict[str, float]) -> float:
    minute = float(features.get("minute", 1.0))
    if minute <= 0:
        return 0.0
    total = _calculate_total_actions(features)
    return total / minute


def _build_advanced_features_from_snapshot(snap: dict) -> Optional[Dict[str, float]]:
    """
    Rebuild the same feature vector that main.py's extract_features() produces.
    snap structure comes from save_snapshot_from_match in main.py:
    {
        "minute": minute,
        "gh": gh, "ga": ga,
        "league_id": ...,
        "market": "HARVEST",
        "suggestion": "HARVEST",
        "confidence": 0,
        "stat": feat   # feat is extract_features() dict
    }
    """
    if not isinstance(snap, dict):
        return None

    minute = float(snap.get("minute", 0.0))
    gh = float(snap.get("gh", 0.0))
    ga = float(snap.get("ga", 0.0))
    stat = snap.get("stat") or {}
    if not isinstance(stat, dict):
        stat = {}

    # If stat already holds full feature dict (from main.py), reuse it.
    # Otherwise reconstruct as best we can.
    feat: Dict[str, float] = {}

    # Base goals
    feat["minute"] = minute
    feat["goals_h"] = float(stat.get("goals_h", gh))
    feat["goals_a"] = float(stat.get("goals_a", ga))
    feat["goals_sum"] = float(stat.get("goals_sum", feat["goals_h"] + feat["goals_a"]))
    feat["goals_diff"] = float(
        stat.get("goals_diff", feat["goals_h"] - feat["goals_a"])
    )

    # xG
    feat["xg_h"] = float(stat.get("xg_h", 0.0))
    feat["xg_a"] = float(stat.get("xg_a", 0.0))
    feat["xg_sum"] = float(stat.get("xg_sum", feat["xg_h"] + feat["xg_a"]))
    feat["xg_diff"] = float(stat.get("xg_diff", feat["xg_h"] - feat["xg_a"]))

    # Shots
    feat["sot_h"] = float(stat.get("sot_h", 0.0))
    feat["sot_a"] = float(stat.get("sot_a", 0.0))
    feat["sot_sum"] = float(stat.get("sot_sum", feat["sot_h"] + feat["sot_a"]))
    feat["sh_total_h"] = float(stat.get("sh_total_h", 0.0))
    feat["sh_total_a"] = float(stat.get("sh_total_a", 0.0))

    # Corners
    feat["cor_h"] = float(stat.get("cor_h", 0.0))
    feat["cor_a"] = float(stat.get("cor_a", 0.0))
    feat["cor_sum"] = float(stat.get("cor_sum", feat["cor_h"] + feat["cor_a"]))

    # Possession
    feat["pos_h"] = float(stat.get("pos_h", 0.0))
    feat["pos_a"] = float(stat.get("pos_a", 0.0))
    feat["pos_diff"] = float(stat.get("pos_diff", feat["pos_h"] - feat["pos_a"]))

    # Cards
    feat["red_h"] = float(stat.get("red_h", 0.0))
    feat["red_a"] = float(stat.get("red_a", 0.0))
    feat["red_sum"] = float(stat.get("red_sum", feat["red_h"] + feat["red_a"]))
    feat["yellow_h"] = float(stat.get("yellow_h", 0.0))
    feat["yellow_a"] = float(stat.get("yellow_a", 0.0))

    # Advanced metrics – if present from main.py, use those, otherwise recompute
    feat["momentum_h"] = float(
        stat.get("momentum_h", _calculate_momentum(feat, "home"))
    )
    feat["momentum_a"] = float(
        stat.get("momentum_a", _calculate_momentum(feat, "away"))
    )
    feat["pressure_index"] = float(
        stat.get("pressure_index", _calculate_pressure_index(feat))
    )
    feat["efficiency_h"] = float(
        stat.get("efficiency_h", _calculate_efficiency(feat, "home"))
    )
    feat["efficiency_a"] = float(
        stat.get("efficiency_a", _calculate_efficiency(feat, "away"))
    )
    feat["total_actions"] = float(
        stat.get("total_actions", _calculate_total_actions(feat))
    )
    feat["action_intensity"] = float(
        stat.get("action_intensity", _calculate_action_intensity(feat))
    )

    return feat


# ---------- Ensemble model (trainer-side) ----------


class AdvancedEnsembleModel:
    """
    Trainer-side ensemble that mirrors the idea in main.py:
    Logistic + RandomForest + GradientBoost with calibration and feature selection.
    """

    def __init__(self, market: str):
        self.market = market
        self.models: Dict[str, CalibratedClassifierCV] = {}
        self.scaler: Optional[StandardScaler] = None
        self.selector: Optional[SelectKBest] = None
        self.selected_features: List[str] = []

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        test_size: float = 0.25,
    ) -> Dict[str, Any]:
        if X.shape[0] < MIN_ROWS:
            return {"error": f"insufficient samples: {X.shape[0]} < {MIN_ROWS}"}

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Feature selection
        if len(feature_names) > 5:
            self.selector = SelectKBest(f_classif, k=min(15, len(feature_names)))
            try:
                X_sel = self.selector.fit_transform(X, y)
                self.selected_features = [
                    feature_names[i]
                    for i in self.selector.get_support(indices=True)
                ]
            except Exception as e:
                log.warning("Feature selection failed for %s: %s", self.market, e)
                self.selector = None
                X_sel = X
                self.selected_features = list(feature_names)
        else:
            X_sel = X
            self.selected_features = list(feature_names)

        # Scale
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_sel)

        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42, stratify=y
        )

        base_models = {
            "logistic": LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=42
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=120, random_state=42, class_weight="balanced"
            ),
            "gradient_boost": GradientBoostingClassifier(
                n_estimators=150, random_state=42
            ),
        }

        metrics_by_model: Dict[str, Dict[str, float]] = {}

        for name, base in base_models.items():
            try:
                method = "isotonic" if name == "logistic" else "sigmoid"
                clf = CalibratedClassifierCV(base, method=method, cv=3)
                clf.fit(X_train, y_train)
                self.models[name] = clf

                proba = clf.predict_proba(X_test)[:, 1]
                pred = (proba >= 0.5).astype(int)

                acc = accuracy_score(y_test, pred)
                prec = precision_score(y_test, pred, zero_division=0)
                auc = roc_auc_score(y_test, proba)

                metrics_by_model[name] = {
                    "accuracy": float(acc),
                    "precision": float(prec),
                    "auc": float(auc),
                }

                log.info(
                    "[%s] %s acc=%.3f prec=%.3f auc=%.3f",
                    self.market,
                    name,
                    acc,
                    prec,
                    auc,
                )
            except Exception as e:
                log.error("Failed to train %s for %s: %s", name, self.market, e)

        if not self.models:
            return {"error": "no models trained"}

        # Ensemble evaluation
        ens_proba = self._ensemble_predict_proba(X_test)
        ens_pred = (ens_proba >= 0.5).astype(int)

        acc = accuracy_score(y_test, ens_pred)
        prec = precision_score(y_test, ens_pred, zero_division=0)
        auc = roc_auc_score(y_test, ens_proba)
        ll = log_loss(y_test, ens_proba)

        log.info(
            "[%s] ensemble acc=%.3f prec=%.3f auc=%.3f logloss=%.3f",
            self.market,
            acc,
            prec,
            auc,
            ll,
        )

        feat_imp = self._feature_importance()

        return {
            "metrics": {
                "accuracy": float(acc),
                "precision": float(prec),
                "auc": float(auc),
                "log_loss": float(ll),
                "samples": int(len(y)),
                "positive_ratio": float(np.mean(y)),
            },
            "per_model": metrics_by_model,
            "selected_features": list(self.selected_features),
            "feature_importance": feat_imp,
        }

    def _ensemble_predict_proba(self, X: np.ndarray) -> np.ndarray:
        preds = []
        for name, clf in self.models.items():
            try:
                p = clf.predict_proba(X)[:, 1]
                preds.append(p)
            except Exception as e:
                log.warning("Prediction failed for %s / %s: %s", self.market, name, e)
        if not preds:
            return np.full(X.shape[0], 0.5)
        preds_arr = np.vstack(preds)
        return preds_arr.mean(axis=0)

    def _feature_importance(self) -> Dict[str, float]:
        # Prefer RF if available
        if "random_forest" in self.models:
            clf = self.models["random_forest"]
            try:
                base = clf.calibrated_classifiers_[0].base_estimator
                if hasattr(base, "feature_importances_"):
                    imp = base.feature_importances_
                    return dict(zip(self.selected_features, map(float, imp)))
            except Exception:
                pass

        # Fallback: logistic coefficients
        if "logistic" in self.models:
            clf = self.models["logistic"]
            try:
                base = clf.calibrated_classifiers_[0].base_estimator
                if hasattr(base, "coef_"):
                    coef = np.abs(base.coef_[0])
                    if coef.max() > 0:
                        coef = coef / coef.max()
                    return dict(zip(self.selected_features, map(float, coef)))
            except Exception:
                pass

        return {f: 1.0 for f in self.selected_features}


# ---------- DB helpers (same style as in main.py) ----------

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
        log.warning("[DNS] IPv4 resolve failed for %s:%s: %s", host, port, e)

    if not requests:
        return None

    try:
        urls = [
            f"https://dns.google/resolve?name={host}&type=A",
            f"https://cloudflare-dns.com/dns-query?name={host}&type=A",
        ]
        for u in urls:
            r = requests.get(
                u, headers={"accept": "application/dns-json"}, timeout=4
            )
            if not r.ok:
                continue
            data = r.json()
            for ans in (data or {}).get("Answer", []) or []:
                ip = ans.get("data")
                if isinstance(ip, str) and ip.count(".") == 3:
                    return ip
    except Exception as e:
        log.warning("[DNS] DoH fallback failed for %s: %s", host, e)

    return None


def _parse_pg_url(url: str) -> Dict[str, Any]:
    from urllib.parse import urlparse, parse_qsl

    pr = urlparse(url)
    if pr.scheme not in ("postgresql", "postgres"):
        raise SystemExit("DATABASE_URL must start with postgresql:// or postgres://")
    user = pr.username or ""
    password = pr.password or ""
    host = pr.hostname or ""
    port = pr.port or 5432
    dbname = (pr.path or "").lstrip("/") or "postgres"
    params = dict(parse_qsl(pr.query))
    params.setdefault("sslmode", "require")
    return {
        "user": user,
        "password": password,
        "host": host,
        "port": int(port),
        "dbname": dbname,
        "params": params,
    }


def _q(v: Optional[str]) -> str:
    s = "" if v is None else str(v)
    if s == "" or all(ch not in s for ch in (" ", "'", "\\", "\t", "\n")):
        return s
    s = s.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{s}'"


def _make_conninfo(parts: Dict[str, Any], port: int, hostaddr: Optional[str]) -> str:
    base = [
        f"host={_q(parts['host'])}",
        f"port={port}",
        f"dbname={_q(parts['dbname'])}",
    ]
    if parts["user"]:
        base.append(f"user={_q(parts['user'])}")
    if parts["password"]:
        base.append(f"password={_q(parts['password'])}")
    if hostaddr:
        base.append(f"hostaddr={_q(hostaddr)}")
    base.append("sslmode=require")
    return " ".join(base)


def _conninfo_candidates(url: str) -> List[str]:
    parts = _parse_pg_url(url)
    env_hostaddr = os.getenv("DB_HOSTADDR")
    prefer_pooled = os.getenv("DB_PREFER_POOLED", "1") not in (
        "0",
        "false",
        "False",
        "no",
        "NO",
    )
    ports: List[int] = []
    if prefer_pooled:
        ports.append(6543)
    if parts["port"] not in ports:
        ports.append(parts["port"])
    cands: List[str] = []
    for p in ports:
        ipv4 = env_hostaddr or _resolve_ipv4(parts["host"], p)
        if ipv4:
            cands.append(_make_conninfo(parts, p, ipv4))
        cands.append(_make_conninfo(parts, p, None))
    return cands


def _connect(db_url: Optional[str]):
    url = db_url or os.getenv("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL must be set.")
    cands = _conninfo_candidates(url)
    delay = 1.0
    last: Optional[Exception] = None
    for attempt in range(6):
        for dsn in cands:
            try:
                conn = psycopg2.connect(dsn)
                conn.autocommit = True
                log.info(
                    "[DB] trainer connected with DSN: %s",
                    dsn.replace("password=", "password=**** "),
                )
                return conn
            except OperationalError as e:
                last = e
                continue
        if attempt == 5:
            raise OperationalError(
                f"Could not connect after retries. Last error: {last}. "
                "Hint: set DB_HOSTADDR=<Supabase IPv4> to pin IPv4."
            )
        time.sleep(delay)
        delay *= 2


def _read_sql(conn, sql: str, params: Tuple = ()) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn, params=params)


def _exec(conn, sql: str, params: Tuple = ()) -> None:
    with conn.cursor() as cur:
        cur.execute(sql, params)


def _set_setting(conn, key: str, value: str) -> None:
    _exec(
        conn,
        "INSERT INTO settings(key,value) VALUES(%s,%s) "
        "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
        (key, value),
    )


def _ensure_training_tables(conn) -> None:
    _exec(conn, "CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)")
    _exec(
        conn,
        """
        CREATE TABLE IF NOT EXISTS model_performance (
            id SERIAL PRIMARY KEY,
            model_name TEXT,
            accuracy FLOAT,
            precision FLOAT,
            auc FLOAT,
            samples INTEGER,
            training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
    )
    _exec(
        conn,
        "CREATE INDEX IF NOT EXISTS idx_model_perf_date ON model_performance (training_date DESC)",
    )


# ---------- Data loading from tip_snapshots + match_results ----------


def load_live_data(
    conn,
    min_minute: int = TRAIN_MIN_MINUTE,
) -> Dict[str, Any]:
    """
    Load harvested in-play snapshots and join with final results.
    Uses tip_snapshots.payload and match_results, exactly as main.py writes them.
    """
    sql = """
    SELECT
        t.match_id,
        t.created_ts,
        t.payload,
        r.final_goals_h,
        r.final_goals_a,
        r.btts_yes,
        EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - to_timestamp(t.created_ts))) / (60*60*24) AS days_ago
    FROM tip_snapshots t
    JOIN match_results r ON t.match_id = r.match_id
    ORDER BY t.created_ts DESC
    LIMIT 20000
    """
    df = _read_sql(conn, sql)

    if df.empty:
        log.warning("No live data found for training")
        return {"X": None, "y": {}, "weights": None, "features": FEATURES}

    df["recency_weight"] = np.exp(
        -np.abs(df["days_ago"].astype(float)) / max(1.0, RECENCY_HALF_LIFE_DAYS)
    )

    feat_rows: List[List[float]] = []
    weights: List[float] = []
    market_targets: Dict[str, List[int]] = {
        "BTTS": [],
        # OU_<line> markets will be added dynamically
        # 1X2_HOME / 1X2_AWAY
        "1X2_HOME": [],
        "1X2_AWAY": [],
    }

    # Prepare OU markets
    ou_keys = [f"OU_{_fmt_line(ln)}" for ln in OU_TRAIN_LINES]
    for key in ou_keys:
        market_targets[key] = []

    for _, row in df.iterrows():
        try:
            snap_raw = row["payload"]
            if isinstance(snap_raw, str):
                snap = json.loads(snap_raw)
            else:
                snap = snap_raw

            feat = _build_advanced_features_from_snapshot(snap)
            if not feat:
                continue

            minute = float(feat.get("minute", 0.0))
            if minute < float(min_minute):
                continue
            if TIP_MAX_MINUTE is not None and minute > TIP_MAX_MINUTE:
                continue

            vec = [float(feat.get(f, 0.0) or 0.0) for f in FEATURES]
            if sum(1 for x in vec if x != 0.0) < 5:
                continue

            feat_rows.append(vec)
            weights.append(float(row["recency_weight"] or 1.0))

            gh = int(row["final_goals_h"] or 0)
            ga = int(row["final_goals_a"] or 0)
            total_goals = gh + ga
            btts = int(row["btts_yes"] or 0)

            # BTTS
            market_targets["BTTS"].append(1 if btts == 1 else 0)

            # OU per configured line
            for ln in OU_TRAIN_LINES:
                key = f"OU_{_fmt_line(ln)}"
                market_targets[key].append(1 if total_goals > ln else 0)

            # 1X2 markets (draw gives 0 for both)
            if gh > ga:
                market_targets["1X2_HOME"].append(1)
                market_targets["1X2_AWAY"].append(0)
            elif ga > gh:
                market_targets["1X2_HOME"].append(0)
                market_targets["1X2_AWAY"].append(1)
            else:
                market_targets["1X2_HOME"].append(0)
                market_targets["1X2_AWAY"].append(0)

        except Exception as e:
            log.debug("Row parse error: %s", e)
            continue

    if not feat_rows:
        log.warning("No usable feature rows extracted")
        return {"X": None, "y": {}, "weights": None, "features": FEATURES}

    X = np.array(feat_rows, dtype=float)
    W = np.array(weights, dtype=float)
    log.info("Loaded %d samples × %d features", X.shape[0], X.shape[1])

    return {"X": X, "y": market_targets, "weights": W, "features": FEATURES}


# ---------- Train market models ----------


def train_advanced_market_model(
    X: np.ndarray,
    y: List[int],
    features: List[str],
    market: str,
    test_size: float,
) -> Optional[Dict[str, Any]]:
    if X is None or len(y) == 0:
        log.warning("No data for market %s", market)
        return None

    y_arr = np.array(y, dtype=int)
    if X.shape[0] != y_arr.shape[0]:
        n = min(X.shape[0], y_arr.shape[0])
        X = X[:n]
        y_arr = y_arr[:n]

    if len(np.unique(y_arr)) < 2:
        log.warning("Market %s has single-class labels; skipping", market)
        return None

    model = AdvancedEnsembleModel(market)
    res = model.train(X, y_arr, features, test_size=test_size)
    if "error" in res:
        log.error("Training error for %s: %s", market, res["error"])
        return None

    return {
        "model_type": "advanced_ensemble",
        "selected_features": res["selected_features"],
        "feature_importance": res["feature_importance"],
        "metrics": res["metrics"],
        "per_model": res["per_model"],
        # Note: we do not persist sklearn weights here; main.py uses its own live ensemble.
        "version": "1.0",
    }


# ---------- Auto-tune thresholds (used by main.py via /admin/auto-tune) ----------


def _calculate_tip_outcome(
    suggestion: str, gh: int, ga: int, btts: int
) -> Optional[int]:
    try:
        total = gh + ga
        s = (suggestion or "").strip()
        if s.startswith("Over"):
            line = float(s.split()[1])
            return 1 if total > line else 0
        if s.startswith("Under"):
            line = float(s.split()[1])
            return 1 if total < line else 0
        if s == "BTTS: Yes":
            return 1 if int(btts) == 1 else 0
        if s == "BTTS: No":
            return 1 if int(btts) == 0 else 0
        if s == "Home Win":
            return 1 if gh > ga else 0
        if s == "Away Win":
            return 1 if ga > gh else 0
        return None
    except Exception:
        return None


def _find_optimal_threshold_advanced(
    confidences_pct: np.ndarray, outcomes: np.ndarray
) -> Optional[float]:
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

        if precision + recall <= 0.0:
            continue

        precision_bonus = 2.0 if precision >= TARGET_PRECISION else 1.0
        sample_adequacy = min(1.0, n / (THRESH_MIN_PREDICTIONS * 2.0))

        f1 = 2 * precision * recall / (precision + recall)
        score = f1 * precision_bonus * (1.0 + 0.3 * sample_adequacy)

        if score > best_score:
            best_score, best_thr = score, thr

    return best_thr


def auto_tune_thresholds_advanced(conn, days: int = 14) -> Dict[str, float]:
    if not AUTO_TUNE_ENABLE:
        log.info("Auto-tune disabled by config")
        return {}

    log.info("Starting advanced auto-tune (days=%d)", days)
    cutoff_ts = int(time.time()) - int(days) * 86400

    sql = """
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
    df = _read_sql(conn, sql, (cutoff_ts,))
    if df.empty:
        log.warning("No tips found for auto-tuning window")
        return {}

    tuned: Dict[str, float] = {}

    for market in sorted(df["market"].dropna().unique()):
        sub = df[df["market"] == market]

        conf_pct = sub.apply(
            lambda r: float(r["confidence_pct"])
            if r["confidence_pct"] is not None and float(r["confidence_pct"]) > 0
            else (
                float(r["confidence_raw"]) * 100.0
                if r["confidence_raw"] is not None
                else 0.0
            ),
            axis=1,
        ).to_numpy(dtype=float)

        outcomes: List[int] = []
        for _, r in sub.iterrows():
            out = _calculate_tip_outcome(
                str(r["suggestion"]),
                int(r["final_goals_h"]),
                int(r["final_goals_a"]),
                int(r["btts_yes"]),
            )
            outcomes.append(0 if out is None else out)

        outcomes_arr = np.array(outcomes, dtype=int)
        if len(outcomes_arr) < THRESH_MIN_PREDICTIONS:
            continue

        thr = _find_optimal_threshold_advanced(conf_pct, outcomes_arr)
        if thr is not None:
            tuned[market] = float(thr)
            _set_setting(conn, f"conf_threshold:{market}", f"{thr:.2f}")

    log.info("Auto-tune completed: %d markets tuned", len(tuned))
    return tuned


# ---------- Store model metadata ----------


def _store_advanced_model(conn, key: str, model_obj: Dict[str, Any]) -> None:
    model_json = json.dumps(model_obj, separators=(",", ":"), ensure_ascii=False)
    _set_setting(conn, key, model_json)

    m = model_obj.get("metrics") or {}
    if m:
        _exec(
            conn,
            """
            INSERT INTO model_performance (model_name, accuracy, precision, auc, samples)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (
                key,
                float(m.get("accuracy", 0.0)),
                float(m.get("precision", 0.0)),
                float(m.get("auc", 0.0)),
                int(m.get("samples", 0)),
            ),
        )
    log.info(
        "Stored advanced model %s (acc=%.3f)",
        key,
        float(m.get("accuracy", 0.0)),
    )


# ---------- Main training entrypoint used by main.py ----------


def train_models(
    db_url: Optional[str] = None,
    min_minute: int = TRAIN_MIN_MINUTE,
    test_size: float = TRAIN_TEST_SIZE,
    min_rows: int = MIN_ROWS,
) -> Dict[str, Any]:
    log.info("Starting advanced in-play training (min_minute=%d)", min_minute)
    conn = _connect(db_url)
    _ensure_training_tables(conn)

    results: Dict[str, Any] = {
        "ok": True,
        "trained": {},
        "auto_tuned": {},
        "errors": [],
        "training_type": "advanced_inplay",
        "models_trained": 0,
    }

    try:
        live = load_live_data(conn, min_minute=min_minute)
        X = live["X"]
        if X is not None:
            for market, y in live["y"].items():
                if len(y) < min_rows:
                    log.info(
                        "Skipping %s: only %d rows (<%d)",
                        market,
                        len(y),
                        min_rows,
                    )
                    results["trained"][market] = False
                    continue

                log.info(
                    "Training model for %s with %d samples", market, len(y)
                )
                mdl = train_advanced_market_model(
                    X,
                    y,
                    live["features"],
                    market,
                    test_size=test_size,
                )
                if mdl:
                    key = f"model_advanced:{market}"
                    _store_advanced_model(conn, key, mdl)
                    results["trained"][market] = True
                    results["models_trained"] += 1
                else:
                    results["trained"][market] = False
                    results["errors"].append(
                        f"Failed to train model for {market}"
                    )

        if AUTO_TUNE_ENABLE:
            tuned = auto_tune_thresholds_advanced(conn, 14)
            results["auto_tuned"] = tuned

        meta = {
            "timestamp": time.time(),
            "live_samples": int(X.shape[0]) if X is not None else 0,
            "trained_models": results["models_trained"],
            "training_type": "advanced_inplay",
            "features_used": len(live["features"]) if X is not None else 0,
        }
        _set_setting(
            conn,
            "advanced_training_metadata",
            json.dumps(meta, separators=(",", ":")),
        )
        _set_setting(conn, "last_advanced_train_ts", str(int(time.time())))

        log.info(
            "Advanced training completed: %d models, %d markets tuned",
            results["models_trained"],
            len(results.get("auto_tuned", {})),
        )

    except Exception as e:
        log.exception("Training failed: %s", e)
        results["ok"] = False
        results["error"] = str(e)
    finally:
        try:
            conn.close()
        except Exception:
            pass

    return results


# ---------- CLI ----------


def _cli_main() -> None:
    ap = argparse.ArgumentParser(description="Advanced in-play trainer")
    ap.add_argument("--db-url", help="Postgres DSN (or env DATABASE_URL)")
    ap.add_argument(
        "--min-minute", type=int, default=TRAIN_MIN_MINUTE
    )
    ap.add_argument(
        "--test-size", type=float, default=TRAIN_TEST_SIZE
    )
    ap.add_argument(
        "--min-rows", type=int, default=MIN_ROWS
    )
    args = ap.parse_args()

    res = train_models(
        db_url=args.db_url or os.getenv("DATABASE_URL"),
        min_minute=args.min_minute,
        test_size=args.test_size,
        min_rows=args.min_rows,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    _cli_main()
