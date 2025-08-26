# file: train_models.py
"""
Train logistic models for BTTS and multiple O/U lines from harvested snapshots.
Outputs to Postgres/SQLite settings with keys read by main.py:
  - model_latest:BTTS_YES
  - model_latest:OU_{line} (e.g., OU_1.5, OU_2.5, ...)
  - (mirrors to model:{name} for compatibility)
"""

import argparse
import json
import os
import sqlite3
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, accuracy_score
from sklearn.model_selection import train_test_split

try:
    import psycopg2  # pragma: no cover
except Exception:
    psycopg2 = None

# Optional .env
try:
    from dotenv import load_dotenv  # pragma: no cover
    load_dotenv()
except Exception:
    pass

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

FEATURES: List[str] = [
    "minute",
    "goals_h", "goals_a", "goals_sum", "goals_diff",
    "xg_h", "xg_a", "xg_sum", "xg_diff",
    "sot_h", "sot_a", "sot_sum",
    "cor_h", "cor_a", "cor_sum",
    "pos_h", "pos_a", "pos_diff",
    "red_h", "red_a", "red_sum",
]


def _connect(db_url: Optional[str], db_path: Optional[str]):
    """Return (engine, connection) where engine ∈ {'pg','sqlite'}."""
    if db_url:
        if psycopg2 is None:
            raise SystemExit("psycopg2 not installed but --db-url / DATABASE_URL provided.")
        if "sslmode=" not in db_url:
            db_url = db_url + ("&" if "?" in db_url else "?") + "sslmode=require"
        conn = psycopg2.connect(db_url)
        conn.autocommit = True
        return "pg", conn
    if db_path:
        conn = sqlite3.connect(db_path)
        return "sqlite", conn
    raise SystemExit("Provide --db-url (Postgres) or --db (SQLite), or set env DATABASE_URL.")


def _read_sql(engine: str, conn, sql: str, params: Tuple = ()) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn, params=params)


def _exec(engine: str, conn, sql: str, params: Tuple) -> None:
    if engine == "pg":
        with conn.cursor() as cur:
            cur.execute(sql, params)
    else:
        cur = conn.cursor()
        cur.execute(sql, params)
        conn.commit()
        cur.close()


def load_data(engine: str, conn, min_minute: int = 15) -> pd.DataFrame:
    """
    Latest snapshot per match JOIN final result, into a single DataFrame with features + labels.
    """
    q = """
    WITH latest AS (
      SELECT match_id, MAX(created_ts) AS ts
      FROM tip_snapshots GROUP BY match_id
    )
    SELECT l.match_id, s.created_ts, s.payload,
           r.final_goals_h, r.final_goals_a, r.btts_yes
    FROM latest l
    JOIN tip_snapshots s ON s.match_id = l.match_id AND s.created_ts = l.ts
    JOIN match_results r ON r.match_id = l.match_id
    """
    rows = _read_sql(engine, conn, q)
    if rows.empty:
        return pd.DataFrame()

    feats: List[Dict[str, Any]] = []
    for _, row in rows.iterrows():
        try:
            payload = json.loads(row["payload"])
        except Exception:
            continue
        stat = (payload.get("stat") or {}) if isinstance(payload, dict) else {}

        try:
            f: Dict[str, float] = {
                "minute": float(payload.get("minute", 0)),
                "goals_h": float(payload.get("gh", 0)),
                "goals_a": float(payload.get("ga", 0)),
                "xg_h": float(stat.get("xg_h", 0)),
                "xg_a": float(stat.get("xg_a", 0)),
                "sot_h": float(stat.get("sot_h", 0)),
                "sot_a": float(stat.get("sot_a", 0)),
                "cor_h": float(stat.get("cor_h", 0)),
                "cor_a": float(stat.get("cor_a", 0)),
                "pos_h": float(stat.get("pos_h", 0)),
                "pos_a": float(stat.get("pos_a", 0)),
                "red_h": float(stat.get("red_h", 0)),
                "red_a": float(stat.get("red_a", 0)),
            }
        except Exception:
            continue

        f["goals_sum"] = f["goals_h"] + f["goals_a"]
        f["goals_diff"] = f["goals_h"] - f["goals_a"]
        f["xg_sum"] = f["xg_h"] + f["xg_a"]
        f["xg_diff"] = f["xg_h"] - f["xg_a"]
        f["sot_sum"] = f["sot_h"] + f["sot_a"]
        f["cor_sum"] = f["cor_h"] + f["cor_a"]
        f["pos_diff"] = f["pos_h"] - f["pos_a"]
        f["red_sum"] = f["red_h"] + f["red_a"]

        gh_f = int(row["final_goals_h"] or 0)
        ga_f = int(row["final_goals_a"] or 0)
        f["final_goals_sum"] = gh_f + ga_f
        f["label_btts"] = 1 if int(row["btts_yes"] or 0) == 1 else 0

        feats.append(f)

    if not feats:
        return pd.DataFrame()

    df = pd.DataFrame(feats)
    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["minute"] = df["minute"].clip(0, 120)
    df = df[df["minute"] >= float(min_minute)].copy()
    return df


def fit_lr_safe(X: np.ndarray, y: np.ndarray) -> Optional[LogisticRegression]:
    if len(np.unique(y)) < 2:
        return None
    return LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear").fit(X, y)


def weights_dict(model: LogisticRegression, feature_names: List[str]) -> Dict[str, float]:
    # Why: main.py expects feature->weight mapping
    return {name: float(w) for name, w in zip(feature_names, model.coef_.ravel().tolist())}


def build_model_blob(model: LogisticRegression, features: List[str]) -> Dict[str, Any]:
    return {
        "intercept": float(model.intercept_.ravel()[0]),
        "weights": weights_dict(model, features),
        "calibration": {"method": "sigmoid", "a": 1.0, "b": 0.0},  # identity
    }


def _fmt_line(line: float) -> str:
    s = f"{line}".rstrip("0").rstrip(".")
    return s


def train_models(
    db_url: Optional[str] = None,
    db_path: Optional[str] = None,
    min_minute: Optional[int] = None,
    test_size: Optional[float] = None,
    min_rows: int = 150,
) -> Dict[str, Any]:
    """
    Train BTTS and O/U lines (config OU_TRAIN_LINES, default: 1.5,2.5,3.5).
    Stores models in settings (model_latest:* and model:* mirrors).
    """
    db_url = db_url or os.getenv("DATABASE_URL")
    min_minute = int(min_minute if min_minute is not None else os.getenv("TRAIN_MIN_MINUTE", 15))
    test_size = float(test_size if test_size is not None else os.getenv("TRAIN_TEST_SIZE", 0.25))
    ou_lines_env = os.getenv("OU_TRAIN_LINES", "1.5,2.5,3.5")
    ou_lines: List[float] = []
    for t in ou_lines_env.split(","):
        t = t.strip()
        if not t:
            continue
        try:
            ou_lines.append(float(t))
        except Exception:
            continue
    if not ou_lines:
        ou_lines = [1.5, 2.5, 3.5]

    engine, conn = _connect(db_url, db_path)
    summary: Dict[str, Any] = {"ok": True, "trained": {}, "metrics": {}, "features": FEATURES}

    try:
        df = load_data(engine, conn, min_minute)
        if df.empty:
            msg = "Not enough labeled data yet."
            logger.info(msg)
            return {"ok": False, "reason": msg}

        # --- BTTS ---
        Xb = df[FEATURES].values
        yb = df["label_btts"].values.astype(int)
        strat_b = yb if (yb.sum() and yb.sum() != len(yb)) else None
        Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(Xb, yb, test_size=test_size, random_state=42, stratify=strat_b)
        mb = fit_lr_safe(Xb_tr, yb_tr)
        if mb is None:
            summary["trained"]["BTTS_YES"] = False
        else:
            p_te_b = mb.predict_proba(Xb_te)[:, 1]
            brier_b = float(brier_score_loss(yb_te, p_te_b))
            acc_b = float(accuracy_score(yb_te, (p_te_b >= 0.5).astype(int)))
            blob_btts = build_model_blob(mb, FEATURES)
            # Save
            for k in ("model_latest:BTTS_YES", "model:BTTS_YES"):
                _exec(engine, conn,
                      "INSERT INTO settings(key,value) VALUES(%s,%s) "
                      "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value" if engine == "pg"
                      else "INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                      (k, json.dumps(blob_btts)))
            summary["trained"]["BTTS_YES"] = True
            summary["metrics"]["BTTS_YES"] = {
                "brier": brier_b, "acc": acc_b, "n_test": int(len(yb_te)), "prevalence": float(yb.mean())
            }

        # --- O/U lines ---
        total_goals = df["final_goals_sum"].values.astype(int)
        for line in ou_lines:
            name = f"OU_{_fmt_line(line)}"
            yo = (total_goals > line).astype(int)
            if yo.sum() == 0 or yo.sum() == len(yo):
                summary["trained"][name] = False
                continue
            Xo = df[FEATURES].values
            strat_o = yo if (yo.sum() and yo.sum() != len(yo)) else None
            Xo_tr, Xo_te, yo_tr, yo_te = train_test_split(Xo, yo, test_size=test_size, random_state=42, stratify=strat_o)
            mo = fit_lr_safe(Xo_tr, yo_tr)
            if mo is None:
                summary["trained"][name] = False
                continue
            p_te_o = mo.predict_proba(Xo_te)[:, 1]
            brier_o = float(brier_score_loss(yo_te, p_te_o))
            acc_o = float(accuracy_score(yo_te, (p_te_o >= 0.5).astype(int)))
            blob_o = build_model_blob(mo, FEATURES)
            # Save
            for k in (f"model_latest:{name}", f"model:{name}"):
                _exec(engine, conn,
                      "INSERT INTO settings(key,value) VALUES(%s,%s) "
                      "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value" if engine == "pg"
                      else "INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                      (k, json.dumps(blob_o)))
            # Backward-compat alias for 2.5
            if abs(line - 2.5) < 1e-6:
                for k in ("model_latest:O25", "model:O25"):
                    _exec(engine, conn,
                          "INSERT INTO settings(key,value) VALUES(%s,%s) "
                          "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value" if engine == "pg"
                          else "INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                          (k, json.dumps(blob_o)))
            summary["trained"][name] = True
            summary["metrics"][name] = {
                "brier": brier_o, "acc": acc_o, "n_test": int(len(yo_te)), "prevalence": float(yo.mean())
            }

        metrics_bundle = {
            "trained_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            **summary["metrics"],
            "features": FEATURES,
        }
        _exec(engine, conn,
              "INSERT INTO settings(key,value) VALUES(%s,%s) "
              "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value" if engine == "pg"
              else "INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
              ("model_metrics_latest", json.dumps(metrics_bundle)))
        return summary

    except Exception as e:
        logger.exception("Training failed: %s", e)
        return {"ok": False, "error": str(e)}
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _cli_main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-url", help="Postgres DSN (preferred, used by main.py)")
    ap.add_argument("--db", help="SQLite path (optional for local dev)")
    ap.add_argument("--min-minute", dest="min_minute", type=int, default=int(os.getenv("TRAIN_MIN_MINUTE", 15)))
    ap.add_argument("--test-size", type=float, default=float(os.getenv("TRAIN_TEST_SIZE", 0.25)))
    ap.add_argument("--min-rows", type=int, default=150)
    args = ap.parse_args()
    res = train_models(
        db_url=args.db_url or os.getenv("DATABASE_URL"),
        db_path=args.db,
        min_minute=args.min_minute,
        test_size=args.test_size,
        min_rows=args.min_rows,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    _cli_main()
