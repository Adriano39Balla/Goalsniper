# path: ./analyze_precision.py
import os
import sys
import json
import math
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import psycopg2
from sklearn.metrics import (
    brier_score_loss,
    accuracy_score,
    roc_auc_score,
)

# ──────────────────────────────────────────────────────────────────────────────
# Feature order must match train_models.FEATURES
FEATURES = [
    "minute","goals_h","goals_a","goals_sum","goals_diff",
    "xg_h","xg_a","xg_sum","xg_diff",
    "sot_h","sot_a","sot_sum",
    "cor_h","cor_a","cor_sum",
    "pos_h","pos_a","pos_diff",
    "red_h","red_a","red_sum"
]

OU_THRESHOLDS = [1.5, 2.5, 3.5, 4.5]

# ──────────────────────────────────────────────────────────────────────────────
def _normalize_db_url(db_url: str) -> str:
    if not db_url:
        return db_url
    if db_url.startswith("postgres://"):
        db_url = "postgresql://" + db_url[len("postgres://"):]
    if "sslmode=" not in db_url:
        db_url = db_url + ("&" if "?" in db_url else "?") + "sslmode=require"
    return db_url

def _connect(db_url: str):
    conn = psycopg2.connect(_normalize_db_url(db_url))
    conn.autocommit = True
    return conn

def _read_sql(conn, sql: str, params: Tuple = ()):
    return pd.read_sql_query(sql, conn, params=params)

def _exec(conn, sql: str, params: Tuple):
    with conn.cursor() as cur:
        cur.execute(sql, params)

# ──────────────────────────────────────────────────────────────────────────────
def load_latest_snapshots(conn, min_minute: int = 15) -> pd.DataFrame:
    """
    Mirrors train_models.load_snapshots_with_labels, but returns *all* latest
    snapshots (one per match) with derived features and labels.
    """
    q = """
    WITH latest AS (
      SELECT match_id, MAX(created_ts) AS ts
      FROM tip_snapshots GROUP BY match_id
    )
    SELECT l.match_id, s.created_ts, s.payload,
           r.final_goals_h, r.final_goals_a, r.btts_yes
    FROM latest l
    JOIN tip_snapshots s ON s.match_id=l.match_id AND s.created_ts=l.ts
    JOIN match_results r ON r.match_id=l.match_id
    """
    rows = _read_sql(conn, q)

    feats = []
    for _, row in rows.iterrows():
        try:
            p = json.loads(row["payload"]) or {}
            stat = p.get("stat") or {}
            f = {
                "match_id": int(row["match_id"]),
                "created_ts": int(row["created_ts"]),
                "minute": float(p.get("minute", 0)),
                "goals_h": float(p.get("gh", 0)), "goals_a": float(p.get("ga", 0)),
                "xg_h": float(stat.get("xg_h", 0)), "xg_a": float(stat.get("xg_a", 0)),
                "sot_h": float(stat.get("sot_h", 0)), "sot_a": float(stat.get("sot_a", 0)),
                "cor_h": float(stat.get("cor_h", 0)), "cor_a": float(stat.get("cor_a", 0)),
                "pos_h": float(stat.get("pos_h", 0)), "pos_a": float(stat.get("pos_a", 0)),
                "red_h": float(stat.get("red_h", 0)), "red_a": float(stat.get("red_a", 0)),
            }
        except Exception:
            continue

        # derived
        f["goals_sum"] = f["goals_h"] + f["goals_a"]
        f["goals_diff"] = f["goals_h"] - f["goals_a"]
        f["xg_sum"] = f["xg_h"] + f["xg_a"]
        f["xg_diff"] = f["xg_h"] - f["xg_a"]
        f["sot_sum"] = f["sot_h"] + f["sot_a"]
        f["cor_sum"] = f["cor_h"] + f["cor_a"]
        f["pos_diff"] = f["pos_h"] - f["pos_a"]
        f["red_sum"] = f["red_h"] + f["red_a"]

        # labels
        gh_f = int(row["final_goals_h"] or 0)
        ga_f = int(row["final_goals_a"] or 0)
        total = gh_f + ga_f
        f["final_total"]     = total
        f["label_btts"]      = 1 if int(row["btts_yes"] or 0) == 1 else 0
        f["label_win_home"]  = 1 if gh_f > ga_f else 0
        f["label_draw"]      = 1 if gh_f == ga_f else 0
        f["label_win_away"]  = 1 if gh_f < ga_f else 0

        feats.append(f)

    if not feats:
        return pd.DataFrame()

    df = pd.DataFrame(feats)
    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["minute"] = df["minute"].clip(0, 120)
    df = df[df["minute"] >= min_minute].copy()
    return df

# ──────────────────────────────────────────────────────────────────────────────
def load_models(conn) -> Dict[str, Dict[str, Any]]:
    rows = _read_sql(conn, "SELECT key, value FROM settings WHERE key LIKE 'model_v2:%'", ())
    out: Dict[str, Dict[str, Any]] = {}
    for _, r in rows.iterrows():
        k = r["key"]
        try:
            code = k.split(":", 1)[1]
            out[code] = json.loads(r["value"])
        except Exception:
            continue
    return out

def score_head_row(x: pd.Series, head: Dict[str, Any]) -> float:
    """
    Compute calibrated probability p = sigmoid(a * (b0 + w·x) + b).
    """
    intercept = float(head.get("intercept", 0.0))
    weights = head.get("weights", {}) or {}
    z = intercept
    for name in FEATURES:
        z += float(weights.get(name, 0.0)) * float(x.get(name, 0.0))

    calib = (head.get("calibration") or {})
    a = float(calib.get("a", 1.0))
    b = float(calib.get("b", 0.0))
    zc = a * z + b
    zc = max(-50.0, min(50.0, zc))
    p = 1.0 / (1.0 + math.exp(-zc))
    return float(p)

def attach_labels_for_head(df: pd.DataFrame, code: str) -> np.ndarray:
    if code == "BTTS" or code == "BTTS_YES":
        return df["label_btts"].astype(int).values
    if code == "WIN_HOME":
        return df["label_win_home"].astype(int).values
    if code == "DRAW":
        return df["label_draw"].astype(int).values
    if code == "WIN_AWAY":
        return df["label_win_away"].astype(int).values
    # OU codes like O15, O25, ...
    if code.startswith("O") and len(code) == 3:
        try:
            th = int(code[1:]) / 10.0
        except Exception:
            th = 2.5
        return (df["final_total"].values >= th).astype(int)
    raise ValueError(f"Unknown head code: {code}")

# ──────────────────────────────────────────────────────────────────────────────
def bucket_minute(m: float) -> int:
    edges = [15, 30, 45, 60, 75, 90, 120]
    for i, e in enumerate(edges):
        if m <= e: return i
    return len(edges)

def game_state(row: pd.Series) -> str:
    gs = int(row["goals_sum"])
    if gs >= 3: gs = 3
    gd = int(row["goals_diff"])
    d = "H" if gd > 0 else "A" if gd < 0 else "D"
    return f"gs{gs}_{d}"

def precision_at_threshold(y_true: np.ndarray, p: np.ndarray, thr: float) -> Dict[str, Any]:
    mask = p >= thr
    n = int(mask.sum())
    if n == 0:
        return {"n": 0, "precision": None, "avg_p": None}
    prec = float(y_true[mask].mean())
    return {"n": n, "precision": prec, "avg_p": float(p[mask].mean())}

def calibration_bins(y_true: np.ndarray, p: np.ndarray, bins: int = 10) -> List[Dict[str, Any]]:
    df = pd.DataFrame({"y": y_true, "p": p})
    df["bin"] = np.clip((df["p"] * bins).astype(int), 0, bins - 1)
    out = []
    for b, g in df.groupby("bin"):
        n = int(len(g))
        if n < 20:  # skip tiny bins
            continue
        out.append({
            "bin": int(b),
            "n": n,
            "p_avg": float(g["p"].mean()),
            "y_avg": float(g["y"].mean()),
        })
    return out

def thresholds_for(code: str) -> List[float]:
    # include env-specified per-head threshold (e.g., BTTS=0.77)
    env_thr = os.getenv(code)
    thrs = [0.5, 0.6, 0.7, 0.75, 0.77, 0.8, 0.85, 0.9]
    if env_thr:
        try:
            v = float(env_thr)
            if v not in thrs:
                thrs.append(v)
        except Exception:
            pass
    thrs = sorted(set([t for t in thrs if 0.0 < t < 1.0]))
    return thrs

# ──────────────────────────────────────────────────────────────────────────────
def analyze(conn, min_minute: int = 15, heads_whitelist: Optional[List[str]] = None) -> Dict[str, Any]:
    df = load_latest_snapshots(conn, min_minute=min_minute)
    if df.empty:
        return {"ok": False, "message": "No labeled snapshots available after min_minute."}

    models = load_models(conn)
    if not models:
        return {"ok": False, "message": "No model_v2:* weights found in settings."}

    # If heads_whitelist provided, filter models
    if heads_whitelist:
        models = {k: v for k, v in models.items() if k in heads_whitelist and k in models}

    report: Dict[str, Any] = {"total_rows": int(len(df)), "heads": {}}

    for code, head in models.items():
        try:
            y = attach_labels_for_head(df, code)
        except Exception:
            continue

        # score all rows
        p = df.apply(lambda r: score_head_row(r, head), axis=1).values.astype(float)

        # core metrics
        try:
            auc = float(roc_auc_score(y, p))
        except Exception:
            auc = float("nan")
        brier = float(brier_score_loss(y, p))
        acc = float(accuracy_score(y, (p >= 0.5).astype(int)))
        prev = float(np.mean(y))

        # minute/state breakdowns
        tmp = df.copy()
        tmp["p"] = p
        tmp["y"] = y
        tmp["min_b"] = tmp["minute"].apply(bucket_minute)
        tmp["state"] = tmp.apply(game_state, axis=1)

        by_minute = (
            tmp.groupby("min_b")
               .apply(lambda g: pd.Series({
                   "n": int(len(g)),
                   "p_avg": float(g["p"].mean()),
                   "y_avg": float(g["y"].mean())
               }))
               .reset_index()
               .to_dict(orient="records")
        )

        by_state = (
            tmp.groupby("state")
               .apply(lambda g: pd.Series({
                   "n": int(len(g)),
                   "p_avg": float(g["p"].mean()),
                   "y_avg": float(g["y"].mean())
               }))
               .reset_index()
               .to_dict(orient="records")
        )

        # precision @ thresholds
        thr_list = thresholds_for(code)
        prec_table = []
        for t in thr_list:
            prec_table.append({"thr": t, **precision_at_threshold(y, p, t)})

        # calibration curve (binned)
        cal_bins = calibration_bins(y, p, bins=10)

        report["heads"][code] = {
            "n": int(len(y)),
            "prevalence": prev,
            "brier": brier,
            "auc": auc,
            "acc@0.5": acc,
            "precision_at": prec_table,
            "calibration_bins": cal_bins,
            "by_minute": by_minute,
            "by_state": by_state,
        }

    # persist in settings
    _exec(conn,
          "INSERT INTO settings(key,value) VALUES(%s,%s) "
          "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
          ("analysis:precision_v2", json.dumps(report)))
    return {"ok": True, "report": report}

# ──────────────────────────────────────────────────────────────────────────────
def main():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("DATABASE_URL is required in env.", flush=True)
        sys.exit(1)

    min_minute = int(os.getenv("ANALYZE_MIN_MINUTE", os.getenv("TRAIN_MIN_MINUTE", "15")))
    heads_env = os.getenv("ANALYZE_HEADS", "").strip()
    heads = [h.strip() for h in heads_env.split(",") if h.strip()] if heads_env else None

    conn = _connect(db_url)
    try:
        result = analyze(conn, min_minute=min_minute, heads_whitelist=heads)
        if not result.get("ok"):
            print(result.get("message", "failed"), flush=True)
            sys.exit(0)

        # Pretty console summary
        rep = result["report"]
        total = rep.get("total_rows", 0)
        print(f"Analyzed {total} latest snapshots.", flush=True)
        for code, r in rep["heads"].items():
            n = r["n"]
            auc = r["auc"]
            brier = r["brier"]
            acc = r["acc@0.5"]
            prev = r["prevalence"]
            # pick an env threshold if present, else 0.77 for BTTS else 0.7 as default display
            display_thr = None
            env_thr = os.getenv(code)
            if env_thr:
                try: display_thr = float(env_thr)
                except: pass
            if display_thr is None:
                display_thr = 0.77 if code.startswith("BTT") else 0.7
            sel = next((row for row in r["precision_at"] if abs(row["thr"] - display_thr) < 1e-9), None)

            ptxt = f"prec@{display_thr:.2f}={sel['precision']:.3f} (n={sel['n']})" if sel and sel["precision"] is not None else f"prec@{display_thr:.2f}=NA"
            print(f"• {code}: n={n} prev={prev:.3f} | AUC={auc:.3f} | Brier={brier:.4f} | Acc@0.5={acc:.3f} | {ptxt}", flush=True)

        print("Saved to settings.key = 'analysis:precision_v2'.", flush=True)
        sys.exit(0)
    except Exception as e:
        print(f"[ANALYZE] exception: {e}", flush=True)
        sys.exit(1)
    finally:
        try:
            conn.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
