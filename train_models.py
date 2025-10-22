# train_models.py - Model training module for goalsniper (in-play focused)
import os
import json
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

# ---- logging ---------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
log = logging.getLogger("trainer")

# ---- env knobs -------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL")
TRAIN_PREMATCH_ENABLE = os.getenv("TRAIN_PREMATCH_ENABLE", "0") not in ("0", "false", "False", "no", "NO")

# ---- minimal DB helper (no side-effects import of main.py) ----------------
_conn = None
def db_execute(query: str, params: tuple = ()) -> list:
    """
    Simple autocommit connection. Avoid importing main.py to prevent
    starting the web app or scheduler during training.
    """
    global _conn
    import psycopg2
    try:
        if _conn is None or _conn.closed:
            if not DATABASE_URL:
                raise RuntimeError("DATABASE_URL not set")
            _conn = psycopg2.connect(DATABASE_URL)
            _conn.autocommit = True
        with _conn.cursor() as cur:
            cur.execute(query, params)
            if query.strip().upper().startswith("SELECT"):
                return cur.fetchall()
            else:
                try:
                    return [{"rowcount": cur.rowcount}]
                except Exception:
                    return []
    except Exception as e:
        log.error("DB query failed: %s | params=%s | err=%s", query, params, e)
        raise

# ---- feature configs (keep your sets) -------------------------------------
FEATURE_SETS = {
    "BTTS": [
        "minute", "goals_h", "goals_a", "goals_sum", "goals_diff",
        "xg_h", "xg_a", "xg_sum", "xg_diff",
        "sot_h", "sot_a", "sot_sum",
        "cor_h", "cor_a", "cor_sum",
        "pos_h", "pos_a", "pos_diff",
        "red_h", "red_a", "red_sum",
        "yellow_h", "yellow_a"
    ],
    "OU_2.5": [
        "minute", "goals_h", "goals_a", "goals_sum", "goals_diff",
        "xg_h", "xg_a", "xg_sum", "xg_diff",
        "sot_h", "sot_a", "sot_sum",
        "cor_h", "cor_a", "cor_sum",
        "pos_h", "pos_a", "pos_diff"
    ],
    "OU_3.5": [
        "minute", "goals_h", "goals_a", "goals_sum", "goals_diff",
        "xg_h", "xg_a", "xg_sum", "xg_diff",
        "sot_h", "sot_a", "sot_sum",
        "cor_h", "cor_a", "cor_sum"
    ],
    "WLD_HOME": [
        "minute", "goals_h", "goals_a", "goals_diff",
        "xg_h", "xg_a", "xg_diff",
        "sot_h", "sot_a",
        "cor_h", "cor_a",
        "pos_h", "pos_a", "pos_diff",
        "red_h", "red_a"
    ],
    "WLD_AWAY": [
        "minute", "goals_h", "goals_a", "goals_diff",
        "xg_h", "xg_a", "xg_diff",
        "sot_h", "sot_a",
        "cor_h", "cor_a",
        "pos_h", "pos_a", "pos_diff",
        "red_h", "red_a"
    ],
    "WLD_DRAW": [
        "minute", "goals_h", "goals_a", "goals_diff",
        "xg_h", "xg_a", "xg_diff",
        "sot_h", "sot_a",
        "cor_h", "cor_a",
        "pos_h", "pos_a", "pos_diff"
    ]
}

# Prematch feature sets (kept but disabled by default)
PREMATCH_FEATURE_SETS = {
    "BTTS": ["avg_goals_h", "avg_goals_a", "avg_goals_h2h", "rest_days_h", "rest_days_a"],
    "OU_2.5": ["avg_goals_h", "avg_goals_a", "avg_goals_h2h", "rest_days_h", "rest_days_a"],
    "OU_3.5": ["avg_goals_h", "avg_goals_a", "avg_goals_h2h", "rest_days_h", "rest_days_a"],
    "WLD_HOME": ["avg_goals_h", "avg_goals_a", "avg_goals_h2h", "rest_days_h", "rest_days_a"],
    "WLD_AWAY": ["avg_goals_h", "avg_goals_a", "avg_goals_h2h", "rest_days_h", "rest_days_a"],
}

# ---- integration test ------------------------------------------------------
def test_integration() -> bool:
    try:
        db_execute("SELECT 1")
        log.info("✅ trainer DB connection OK")
        return True
    except Exception as e:
        log.error("❌ trainer DB connection failed: %s", e)
        return False

# ---- data loaders ----------------------------------------------------------
def load_training_data(days: int = 30) -> pd.DataFrame:
    if not test_integration():
        return pd.DataFrame()
    cutoff = int((datetime.now() - timedelta(days=days)).timestamp())
    query = """
        SELECT ts.payload AS snapshot,
               mr.final_goals_h,
               mr.final_goals_a,
               mr.btts_yes
        FROM tip_snapshots ts
        JOIN match_results mr ON ts.match_id = mr.match_id
        WHERE ts.created_ts >= %s
          AND ts.payload IS NOT NULL
          AND ts.payload != 'null'
          AND mr.final_goals_h IS NOT NULL
          AND mr.final_goals_a IS NOT NULL
        ORDER BY ts.created_ts DESC
    """
    try:
        rows = db_execute(query, (cutoff,))
        data = []
        for snapshot, goals_h, goals_a, btts_yes in rows:
            try:
                snap = json.loads(snapshot) or {}
                stat = snap.get("stat", {}) or {}
                features = {
                    "minute": float(snap.get("minute", 0)),
                    "goals_h": int(goals_h or 0),
                    "goals_a": int(goals_a or 0),
                    "goals_sum": int((goals_h or 0) + (goals_a or 0)),
                    "goals_diff": int((goals_h or 0) - (goals_a or 0)),
                    "xg_h": float(stat.get("xg_h", 0)),
                    "xg_a": float(stat.get("xg_a", 0)),
                    "xg_sum": float(stat.get("xg_sum", 0)),
                    "xg_diff": float(stat.get("xg_diff", 0)),
                    "sot_h": float(stat.get("sot_h", 0)),
                    "sot_a": float(stat.get("sot_a", 0)),
                    "sot_sum": float(stat.get("sot_sum", 0)),
                    "cor_h": float(stat.get("cor_h", 0)),
                    "cor_a": float(stat.get("cor_a", 0)),
                    "cor_sum": float(stat.get("cor_sum", 0)),
                    "pos_h": float(stat.get("pos_h", 0)),
                    "pos_a": float(stat.get("pos_a", 0)),
                    "pos_diff": float(stat.get("pos_diff", 0)),
                    "red_h": float(stat.get("red_h", 0)),
                    "red_a": float(stat.get("red_a", 0)),
                    "red_sum": float(stat.get("red_sum", 0)),
                    "yellow_h": float(stat.get("yellow_h", 0)),
                    "yellow_a": float(stat.get("yellow_a", 0)),
                    "yellow_sum": float(stat.get("yellow_sum", 0)),
                    "btts_yes": int(btts_yes or 0),
                    "final_goals_h": int(goals_h or 0),
                    "final_goals_a": int(goals_a or 0),
                    "total_goals": int((goals_h or 0) + (goals_a or 0)),
                }
                data.append(features)
            except json.JSONDecodeError:
                continue
        return pd.DataFrame(data)
    except Exception as e:
        log.error("Failed to load in-play training data: %s", e)
        return pd.DataFrame()

def load_prematch_training_data(days: int = 90) -> pd.DataFrame:
    if not test_integration():
        return pd.DataFrame()
    cutoff = int((datetime.now() - timedelta(days=days)).timestamp())
    query = """
        SELECT ps.payload AS snapshot,
               mr.final_goals_h,
               mr.final_goals_a,
               mr.btts_yes
        FROM prematch_snapshots ps
        JOIN match_results mr ON ps.match_id = mr.match_id
        WHERE ps.created_ts >= %s
          AND ps.payload IS NOT NULL
          AND ps.payload != 'null'
          AND mr.final_goals_h IS NOT NULL
          AND mr.final_goals_a IS NOT NULL
    """
    try:
        rows = db_execute(query, (cutoff,))
        data = []
        for snapshot, goals_h, goals_a, btts_yes in rows:
            try:
                snap = json.loads(snapshot) or {}
                feat = snap.get("feat", {}) or {}
                data.append({
                    "avg_goals_h": float(feat.get("avg_goals_h", 0)),
                    "avg_goals_a": float(feat.get("avg_goals_a", 0)),
                    "avg_goals_h2h": float(feat.get("avg_goals_h2h", 0)),
                    "rest_days_h": float(feat.get("rest_days_h", 0)),
                    "rest_days_a": float(feat.get("rest_days_a", 0)),
                    "final_goals_h": int(goals_h or 0),
                    "final_goals_a": int(goals_a or 0),
                    "btts_yes": int(btts_yes or 0),
                    "total_goals": int((goals_h or 0) + (goals_a or 0)),
                })
            except json.JSONDecodeError:
                continue
        return pd.DataFrame(data)
    except Exception as e:
        log.error("Failed to load prematch training data: %s", e)
        return pd.DataFrame()

# ---- labels ---------------------------------------------------------------
def create_labels(df: pd.DataFrame, target_type: str) -> np.ndarray:
    if target_type == "BTTS":
        return (df["btts_yes"] == 1).astype(int).values
    if target_type == "OU_2.5":
        return (df["total_goals"] > 2.5).astype(int).values
    if target_type == "OU_3.5":
        return (df["total_goals"] > 3.5).astype(int).values
    if target_type == "WLD_HOME":
        return (df["final_goals_h"] > df["final_goals_a"]).astype(int).values
    if target_type == "WLD_AWAY":
        return (df["final_goals_a"] > df["final_goals_h"]).astype(int).values
    if target_type == "WLD_DRAW":
        return (df["final_goals_h"] == df["final_goals_a"]).astype(int).values
    raise ValueError(f"Unknown target type: {target_type}")

# ---- training -------------------------------------------------------------
def train_logistic_model(X: np.ndarray, y: np.ndarray, model_name: str, feature_names: List[str]) -> Optional[Dict[str, Any]]:
    if len(X) < 50:
        log.warning("[TRAIN] Insufficient data for %s: %d samples", model_name, len(X))
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    if len(np.unique(y_train)) < 2:
        log.warning("[TRAIN] Only one class for %s — skip", model_name)
        return None

    try:
        model = LogisticRegression(
            penalty="l2", C=1.0, random_state=42, max_iter=1000, class_weight="balanced"
        )
        calibrated = CalibratedClassifierCV(model, method="isotonic", cv=3)
        calibrated.fit(X_train, y_train)

        y_pred = calibrated.predict(X_test)
        y_prob = calibrated.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5

        log.info("[TRAIN] %s — acc=%.3f | prec=%.3f | auc=%.3f | n=%d",
                 model_name, acc, prec, auc, len(X))

        # extract LR parameters
        base = calibrated.calibrated_classifiers_[0].base_estimator
        coefs = base.coef_[0]
        intercept = float(base.intercept_[0])

        # map to real feature names (critical for main.predict_from_model)
        weights = {str(name): float(w) for name, w in zip(feature_names, coefs)}

        # keep simple sigmoid calibration (identity) — your main.py supports it
        calibration = {"method": "sigmoid", "a": 1.0, "b": 0.0}

        return {
            "weights": weights,
            "intercept": intercept,
            "calibration": calibration,
            "performance": {
                "accuracy": float(acc),
                "precision": float(prec),
                "auc": float(auc),
                "n_samples": int(len(X)),
            },
            "feature_names": feature_names,
        }
    except Exception as e:
        log.error("[TRAIN] Failed %s: %s", model_name, e)
        return None

def _save_one_key(setting_key: str, model_blob: Dict[str, Any]) -> None:
    payload = json.dumps({
        "weights": model_blob["weights"],
        "intercept": model_blob["intercept"],
        "calibration": model_blob["calibration"],
        "trained_at": datetime.now().isoformat(),
        "performance": model_blob.get("performance", {})
    }, ensure_ascii=False)
    db_execute(
        """
        INSERT INTO settings(key, value)
        VALUES (%s, %s)
        ON CONFLICT(key) DO UPDATE SET value = EXCLUDED.value
        """,
        (setting_key, payload)
    )
    log.info("[TRAIN] Saved %s", setting_key)

def save_model_to_db(model_name: str, model_data: Dict[str, Any], aliases: Optional[List[str]] = None) -> bool:
    """
    Save under keys that main.py actually loads:
      - model_v2:{name}
      - model:{name}          (compat)
    Also save optional aliases (e.g. 1X2_HOME for WLD_HOME)
    """
    try:
        keys = [f"model_v2:{model_name}", f"model:{model_name}"]
        aliases = aliases or []
        for a in aliases:
            keys.append(f"model_v2:{a}")
            keys.append(f"model:{a}")
        for k in keys:
            _save_one_key(k, model_data)
        return True
    except Exception as e:
        log.error("[TRAIN] Save failed for %s: %s", model_name, e)
        return False

def get_existing_model_performance(model_name: str) -> Optional[float]:
    try:
        rows = db_execute("SELECT value FROM settings WHERE key IN (%s, %s) LIMIT 1",
                          (f"model_v2:{model_name}", f"model:{model_name}"))
        if rows:
            data = json.loads(rows[0][0])
            return float((data.get("performance") or {}).get("accuracy", 0))
        return None
    except Exception:
        return None

def should_retrain_model(model_name: str, new_accuracy: float, min_improvement: float = 0.02) -> bool:
    existing = get_existing_model_performance(model_name)
    if existing is None:
        log.info("[TRAIN] No existing %s — will train", model_name)
        return True
    gain = new_accuracy - existing
    if gain >= min_improvement:
        log.info("[TRAIN] %s improved: %.3f → %.3f (Δ=%.3f) — retraining", model_name, existing, new_accuracy, gain)
        return True
    log.info("[TRAIN] %s improvement insufficient: %.3f → %.3f (Δ=%.3f) — skip", model_name, existing, new_accuracy, gain)
    return False

def train_inplay_models(df: pd.DataFrame, retrain_all: bool = False) -> Dict[str, bool]:
    results: Dict[str, bool] = {}

    train_df = df[(df["minute"] > 20) & (df["xg_sum"] > 0) & (df["sot_sum"] > 0)].copy()
    if len(train_df) < 100:
        log.warning("[TRAIN] Not enough in-play data: %d", len(train_df))
        return {name: False for name in FEATURE_SETS.keys()}

    for model_name, feature_list in FEATURE_SETS.items():
        try:
            cols = [c for c in feature_list if c in train_df.columns]
            X = train_df[cols].fillna(0).values
            y = create_labels(train_df, model_name)

            if len(np.unique(y)) < 2:
                log.warning("[TRAIN] Skipping %s — only one class", model_name)
                results[model_name] = False
                continue

            blob = train_logistic_model(X, y, model_name, cols)
            if not blob or blob["performance"]["accuracy"] <= 0.5:
                log.warning("[TRAIN] %s training failed/weak", model_name)
                results[model_name] = False
                continue

            if retrain_all or should_retrain_model(model_name, blob["performance"]["accuracy"]):
                # map WLD_* to 1X2_* so main.py's ensemble can find them
                aliases = []
                if model_name == "WLD_HOME":
                    aliases = ["1X2_HOME"]
                elif model_name == "WLD_AWAY":
                    aliases = ["1X2_AWAY"]
                elif model_name == "WLD_DRAW":
                    aliases = ["1X2_DRAW"]

                ok = save_model_to_db(model_name, blob, aliases=aliases)
                results[model_name] = bool(ok)
            else:
                results[model_name] = False
        except Exception as e:
            log.error("[TRAIN] Error training %s: %s", model_name, e)
            results[model_name] = False

    return results

def train_prematch_models(df: pd.DataFrame, retrain_all: bool = False) -> Dict[str, bool]:
    """
    Kept for completeness but gated by TRAIN_PREMATCH_ENABLE.
    Saves as PRE_* models under model_v2:PRE_{name}.
    """
    results: Dict[str, bool] = {}
    if len(df) < 50:
        log.warning("[TRAIN] Not enough prematch data: %d", len(df))
        return {f"PRE_{name}": False for name in PREMATCH_FEATURE_SETS.keys()}

    for base_name, feature_list in PREMATCH_FEATURE_SETS.items():
        model_name = f"PRE_{base_name}"
        try:
            cols = [c for c in feature_list if c in df.columns]
            X = df[cols].fillna(0).values
            y = create_labels(df, base_name)

            if len(np.unique(y)) < 2:
                log.warning("[TRAIN] Skipping %s — one class", model_name)
                results[model_name] = False
                continue

            blob = train_logistic_model(X, y, model_name, cols)
            if not blob or blob["performance"]["accuracy"] <= 0.5:
                log.warning("[TRAIN] %s training failed/weak", model_name)
                results[model_name] = False
                continue

            if retrain_all or should_retrain_model(model_name, blob["performance"]["accuracy"]):
                ok = save_model_to_db(model_name, blob)
                results[model_name] = bool(ok)
            else:
                results[model_name] = False
        except Exception as e:
            log.error("[TRAIN] Error training %s: %s", model_name, e)
            results[model_name] = False

    return results

# ---- main entry ------------------------------------------------------------
def train_models(retrain_all: bool = False, days_back: int = 30) -> Dict[str, Any]:
    t0 = time.time()
    log.info("[TRAIN] Starting (retrain_all=%s, days=%d, prematch=%s)",
             retrain_all, days_back, TRAIN_PREMATCH_ENABLE)

    try:
        if not test_integration():
            return {"ok": False, "reason": "Database connection failed", "trained": {}, "duration": int(time.time()-t0)}

        # In-play
        inplay_df = load_training_data(days_back)
        log.info("[TRAIN] In-play samples: %d", len(inplay_df))
        if len(inplay_df) >= 50:
            inplay_results = train_inplay_models(inplay_df, retrain_all)
        else:
            log.warning("[TRAIN] Skipping in-play — too little data")
            inplay_results = {name: False for name in FEATURE_SETS.keys()}

        # Prematch (optional)
        if TRAIN_PREMATCH_ENABLE:
            pm_df = load_prematch_training_data(max(days_back * 3, 90))
            log.info("[TRAIN] Prematch samples: %d", len(pm_df))
            prematch_results = train_prematch_models(pm_df, retrain_all) if len(pm_df) >= 30 else {f"PRE_{n}": False for n in PREMATCH_FEATURE_SETS.keys()}
        else:
            prematch_results = {}

        all_results = {**inplay_results, **prematch_results}
        trained_count = sum(1 for v in all_results.values() if v)
        dur = int(time.time() - t0)

        log.info("[TRAIN] Done in %ds. Trained=%d, Skipped=%d",
                 dur, trained_count, len(all_results) - trained_count)

        return {
            "ok": True,
            "trained": all_results,
            "duration": dur,
            "summary": {
                "inplay_samples": len(inplay_df),
                "prematch_samples": (len(prematch_results) > 0),
                "models_trained": trained_count,
                "models_skipped": len(all_results) - trained_count,
                "total_time": dur
            },
            "message": f"Training completed: {trained_count} models updated"
        }
    except Exception as e:
        log.exception("[TRAIN] Failed: %s", e)
        return {"ok": False, "reason": str(e), "trained": {}, "duration": int(time.time()-t0)}

# ---- housekeeping ----------------------------------------------------------
def cleanup_old_data(days_to_keep: int = 90) -> bool:
    try:
        cutoff = int((datetime.now() - timedelta(days=days_to_keep)).timestamp())
        queries = [
            "DELETE FROM tip_snapshots WHERE created_ts < %s",
            "DELETE FROM prematch_snapshots WHERE created_ts < %s",
            "DELETE FROM tips WHERE created_ts < %s AND suggestion = 'HARVEST'"
        ]
        total = 0
        for q in queries:
            res = db_execute(q, (cutoff,))
            if res and isinstance(res[0], dict):
                total += max(0, int(res[0].get("rowcount", 0)))
        log.info("[TRAIN] Cleanup OK (<%d days). Rows ~%d", days_to_keep, total)
        return True
    except Exception as e:
        log.error("[TRAIN] Cleanup failed: %s", e)
        return False

if __name__ == "__main__":
    log.info("[TRAIN] Standalone training session")
    os.environ.setdefault("RUN_SCHEDULER", "0")  # make sure trainer never starts web scheduler
    result = train_models(retrain_all=False, days_back=30)
    if datetime.now().weekday() == 0:  # Monday
        cleanup_old_data(90)
    if result.get("ok"):
        trained = sum(1 for v in (result.get("trained") or {}).values() if v)
        log.info("[TRAIN] OK — %d models trained.", trained)
        raise SystemExit(0)
    else:
        log.error("[TRAIN] FAILED — %s", result.get("reason", "Unknown"))
        raise SystemExit(1)
