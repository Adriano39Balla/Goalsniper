# train_models.py
# Clean, production-grade trainer synced with goalsniper main.py
# Trains OU / BTTS / 1X2 models using AdvancedEnsemblePredictor

import os
import json
import time
import logging
import psycopg2
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

from psycopg2.extras import RealDictCursor

# ----------------------------------------------------------------------------
# IMPORTANT: prevent main.py from starting the web app & scheduler when imported
# ----------------------------------------------------------------------------
os.environ.setdefault("GOALSNIPER_SKIP_BOOT_ON_IMPORT", "1")

# Import the *same* predictor class and constants used by main.py
from main import (
    AdvancedEnsemblePredictor,
    OU_LINES,
    _fmt_line,
    extract_features,  # not used directly, kept for parity / potential feature transforms
)

log = logging.getLogger("train_models")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")


# ---------------------------------------------------------------------
# DB Connection
# ---------------------------------------------------------------------

def _connect(url: Optional[str] = None):
    dsn = url or os.getenv("DATABASE_URL")
    if not dsn:
        raise SystemExit("DATABASE_URL missing for trainer")
    if "sslmode" not in dsn:
        dsn += (("&" if "?" in dsn else "?") + "sslmode=require")
    return psycopg2.connect(dsn, cursor_factory=RealDictCursor)


# ---------------------------------------------------------------------
# Markets to train (Option A = all)
# ---------------------------------------------------------------------

def all_training_targets() -> List[Tuple[str, str]]:
    targets: List[Tuple[str, str]] = []

    # OU markets
    for ln in OU_LINES:
        s = _fmt_line(ln)
        targets.append((f"Over/Under {s}", f"Over {s} Goals"))
        targets.append((f"Over/Under {s}", f"Under {s} Goals"))

    # BTTS
    targets.append(("BTTS", "BTTS: Yes"))
    targets.append(("BTTS", "BTTS: No"))

    # 1X2
    targets.append(("1X2", "Home Win"))
    targets.append(("1X2", "Away Win"))

    return targets


# ---------------------------------------------------------------------
# Build dataset from harvested snapshots + match results
# ---------------------------------------------------------------------

def load_training_data(conn) -> List[Dict[str, Any]]:
    """
    Pulls historical tip_snapshots + match_results and builds a training dataset.
    Returns list of dicts: {features, market, suggestion, outcome}
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT s.match_id, s.created_ts, s.payload, r.final_goals_h, r.final_goals_a, r.btts_yes
        FROM tip_snapshots s
        JOIN match_results r ON r.match_id = s.match_id
        ORDER BY s.created_ts DESC
        LIMIT 50000
    """)
    rows = cur.fetchall()

    out: List[Dict[str, Any]] = []
    for row in rows:
        try:
            snap = json.loads(row["payload"])
            features = snap.get("stat", {})
            gh = int(row["final_goals_h"] or 0)
            ga = int(row["final_goals_a"] or 0)
            total = gh + ga
            btts_yes = int(row["btts_yes"] or 0)

            # Build outcomes for each market
            for market, suggestion in all_training_targets():
                y = _compute_outcome(market, suggestion, gh, ga, total, btts_yes)
                if y is None:
                    continue
                out.append({
                    "features": features,
                    "market": market,
                    "suggestion": suggestion,
                    "outcome": int(y),
                })
        except Exception:
            continue

    return out


def _compute_outcome(market: str, suggestion: str,
                     gh: int, ga: int, total: int, btts_yes: int) -> Optional[int]:
    if market.startswith("Over/Under"):
        ln = _extract_line(suggestion)
        if ln is None:
            return None
        if suggestion.startswith("Over"):
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

    if market == "BTTS":
        if suggestion == "BTTS: Yes":
            return 1 if btts_yes == 1 else 0
        if suggestion == "BTTS: No":
            return 1 if btts_yes == 0 else 0
        return None

    if market == "1X2":
        if suggestion == "Home Win":
            return 1 if gh > ga else 0
        if suggestion == "Away Win":
            return 1 if ga > gh else 0
        return None

    return None


def _extract_line(sug: str) -> Optional[float]:
    try:
        for tok in sug.split():
            tok = tok.replace(",", "").strip()
            try:
                return float(tok)
            except Exception:
                continue
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------
# Prepare features for training
# ---------------------------------------------------------------------

def build_feature_matrix(data: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[int], List[str]]:
    """
    Builds X, y and feature_name list.
    Ensures consistent feature ordering for all markets.
    """
    if not data:
        return np.zeros((0, 1)), [], []

    df_rows = [d["features"] for d in data]
    y = [d["outcome"] for d in data]

    df = pd.DataFrame(df_rows).fillna(0)
    feat_names = df.columns.tolist()

    X = df.values.astype(float)
    return X, y, feat_names


# ---------------------------------------------------------------------
# Train a single model
# ---------------------------------------------------------------------

def train_single_market(
    conn,
    market: str,
    suggestion: str,
    all_samples: List[Dict[str, Any]],
    save_dir: str = "models"
) -> Optional[Dict[str, Any]]:
    """
    Train one market_suggestion model with AdvancedEnsemblePredictor
    """
    # Filter samples for this market/suggestion
    rows = [r for r in all_samples if r["market"] == market and r["suggestion"] == suggestion]
    if len(rows) < 40:
        log.info("[TRAIN] skipped %s/%s: too few samples (%d)", market, suggestion, len(rows))
        return None

    # Build labels just to sanity-check class balance
    X, y, feat_names = build_feature_matrix(rows)
    y = np.array(y, dtype=int)

    if len(np.unique(y)) < 2:
        log.info("[TRAIN] skipped %s/%s: single class", market, suggestion)
        return None

    model_key = f"{market}_{suggestion.replace(' ', '_')}"
    predictor = AdvancedEnsemblePredictor(model_key)

    # Train using raw feature dicts to preserve selector/scaler inside predictor
    result = predictor.train([r["features"] for r in rows], y.tolist())
    if not result.get("ok"):
        log.error("[TRAIN] training failed for %s: %s", model_key, result)
        return None

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{model_key}.pkl")

    try:
        import joblib
        joblib.dump(predictor, path)
    except Exception as e:
        log.error("[TRAIN] save failed for %s: %s", model_key, e)
        return None

    # Metadata is stored in settings later
    return {
        "market": market,
        "suggestion": suggestion,
        "model_key": model_key,
        "path": path,
        "sample_count": len(rows),
        "trained_at": int(time.time()),
        "features": feat_names,
    }


# ---------------------------------------------------------------------
# Multi-market training
# ---------------------------------------------------------------------

def train_all_markets(conn) -> Dict[str, Any]:
    samples = load_training_data(conn)
    if not samples:
        return {"ok": False, "reason": "no training data"}

    trained_info: Dict[str, Any] = {}
    for market, suggestion in all_training_targets():
        info = train_single_market(conn, market, suggestion, samples)
        if info:
            key = info["model_key"]
            trained_info[key] = info

            # Store metadata in settings
            meta = {
                "market": market,
                "suggestion": suggestion,
                "sample_count": info["sample_count"],
                "model_key": info["model_key"],
                "saved_path": info["path"],
                "trained_at": info["trained_at"],
            }
            _store_model_metadata(conn, info["model_key"], meta)

    return {"ok": True, "trained": trained_info}


# ---------------------------------------------------------------------
# Store model metadata in settings
# ---------------------------------------------------------------------

def _store_model_metadata(conn, model_key: str, meta: Dict[str, Any]) -> None:
    """
    Persist model metadata so main.py (or tools) can discover latest models.
    Stored under: settings.key = f"model_latest:{model_key}"
    """
    key = f"model_latest:{model_key}"
    val = json.dumps(meta, separators=(",", ":"), ensure_ascii=False)

    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO settings(key, value)
        VALUES (%s, %s)
        ON CONFLICT(key) DO UPDATE
        SET value = EXCLUDED.value
        """,
        (key, val),
    )
    try:
        conn.commit()
    except Exception:
        pass


# ---------------------------------------------------------------------
# Advanced auto-tuning of confidence thresholds
# ---------------------------------------------------------------------

def _compute_tip_outcome_for_row(row: Dict[str, Any]) -> Optional[int]:
    """
    Use the same semantics as _compute_outcome() but for a tip row.
    """
    market = row["market"]
    suggestion = row["suggestion"]
    gh = int(row["final_goals_h"] or 0)
    ga = int(row["final_goals_a"] or 0)
    total = gh + ga
    btts_yes = int(row["btts_yes"] or 0)
    return _compute_outcome(market, suggestion, gh, ga, total, btts_yes)


def _set_conf_threshold(conn, market: str, thresh: float) -> None:
    """
    Persist tuned threshold for a given market string, as used in main._get_market_threshold:
        key = f"conf_threshold:{market}"
    """
    key = f"conf_threshold:{market}"
    val = str(round(float(thresh), 2))

    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO settings(key, value)
        VALUES (%s, %s)
        ON CONFLICT(key) DO UPDATE
        SET value = EXCLUDED.value
        """,
        (key, val),
    )
    try:
        conn.commit()
    except Exception:
        pass


def auto_tune_thresholds_advanced(conn, days: int = 14) -> Dict[str, float]:
    """
    Auto-tune confidence thresholds per market using realized precision.

    Works on the `tips` + `match_results` tables for the last `days` days.
    Writes settings entries:
        conf_threshold:{market} -> tuned threshold (percentage)
    which main._get_market_threshold() will read.
    """
    target_precision = float(os.getenv("TARGET_PRECISION", "0.60"))
    min_preds        = int(os.getenv("THRESH_MIN_PREDICTIONS", "25"))
    min_thresh       = float(os.getenv("MIN_THRESH", "55"))
    max_thresh       = float(os.getenv("MAX_THRESH", "85"))

    now_ts  = int(time.time())
    cutoff  = now_ts - days * 24 * 3600
    cur     = conn.cursor()

    cur.execute(
        """
        SELECT
            t.market,
            t.suggestion,
            t.confidence,
            t.created_ts,
            r.final_goals_h,
            r.final_goals_a,
            r.btts_yes
        FROM tips t
        JOIN match_results r ON r.match_id = t.match_id
        WHERE t.created_ts >= %s
          AND t.suggestion <> 'HARVEST'
        """,
        (cutoff,),
    )
    rows = cur.fetchall()
    if not rows:
        log.info("[AUTO-TUNE] no graded tips in window")
        return {}

    # Group by exact market string as used in main (e.g. "BTTS", "1X2", "Over/Under 2.5")
    by_market: Dict[str, List[Tuple[float, int]]] = {}
    for row in rows:
        try:
            outcome = _compute_tip_outcome_for_row(row)
        except Exception:
            continue
        if outcome is None:
            continue

        market = row["market"]
        conf   = float(row["confidence"] or 0.0)  # percentage
        by_market.setdefault(market, []).append((conf, int(outcome)))

    tuned: Dict[str, float] = {}

    for market, lst in by_market.items():
        if len(lst) < min_preds:
            log.info("[AUTO-TUNE] %s skipped: too few graded (%d)", market, len(lst))
            continue

        # Sort by confidence desc
        lst_sorted = sorted(lst, key=lambda x: x[0], reverse=True)

        best_score     = None
        best_threshold = None

        wins_prefix = 0
        total_prefix = 0

        for i, (conf, outcome) in enumerate(lst_sorted):
            total_prefix += 1
            wins_prefix  += outcome

            if total_prefix < min_preds:
                continue

            if conf < min_thresh or conf > max_thresh:
                continue

            precision = wins_prefix / max(1, total_prefix)

            # Score: closeness to target precision, with slight bias for more samples
            diff  = precision - target_precision
            score = -diff * diff + 0.0001 * total_prefix

            if (best_score is None) or (score > best_score):
                best_score = score
                best_threshold = conf

        if best_threshold is None:
            # Fallback: use midpoint if nothing met constraints
            best_threshold = (min_thresh + max_thresh) / 2.0

        tuned[market] = float(round(best_threshold, 2))
        _set_conf_threshold(conn, market, best_threshold)
        log.info(
            "[AUTO-TUNE] market=%s threshold=%.2f (samples=%d)",
            market,
            best_threshold,
            len(lst),
        )

    return tuned


# ---------------------------------------------------------------------
# Public train_models() entrypoint used by main.py
# ---------------------------------------------------------------------

def train_models() -> Dict[str, Any]:
    """
    Main entry used from main.py:
        from train_models import train_models
        ...
        res = train_models()
    """
    conn = _connect(os.getenv("DATABASE_URL"))
    try:
        conn.autocommit = True
        res = train_all_markets(conn)
        return res
    finally:
        try:
            conn.close()
        except Exception:
            pass


# ---------------------------------------------------------------------
# CLI usage
# ---------------------------------------------------------------------

if __name__ == "__main__":
    conn = _connect(os.getenv("DATABASE_URL"))
    try:
        conn.autocommit = True
        result = train_all_markets(conn)
        print(json.dumps(result, indent=2, sort_keys=True, default=str))

        if os.getenv("AUTO_TUNE_ON_TRAIN", "1").lower() not in ("0", "false", "no"):
            tuned = auto_tune_thresholds_advanced(
                conn,
                days=int(os.getenv("AUTO_TUNE_DAYS", "14")),
            )
            print("\nTuned thresholds:")
            print(json.dumps(tuned, indent=2, sort_keys=True))
    finally:
        try:
            conn.close()
        except Exception:
            pass
