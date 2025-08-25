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
    log_loss,
)

# ──────────────────────────────────────────────────────────────────────────────
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
    if not db_url: return db_url
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

        gh_f = int(row["final_goals_h"] or 0)
        ga_f = int(row["final_goals_a"] or 0)
        total = gh_f + ga_f
        f["final_total"]     = total
        f["label_btts"]      = 1 if int(row["btts_yes"] or 0) == 1 else 0
        f["label_win_home"]  = 1 if gh_f > ga_f else 0
        f["label_draw"]      = 1 if gh_f == ga_f else 0
        f["label_win_away"]  = 1 if gh_f < ga_f else 0
        feats.append(f)

    if not feats: return pd.DataFrame()
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
    if head.get("type") == "lr":
        intercept = float(head.get("intercept", 0.0))
        weights = head.get("weights", {}) or {}
        z = intercept
        for name in FEATURES:
            z += float(weights.get(name, 0.0)) * float(x.get(name, 0.0))
        calib = (head.get("calibration") or {})
        a = float(calib.get("a", 1.0)); b = float(calib.get("b", 0.0))
        zc = a*z + b
        zc = max(-50.0, min(50.0, zc))
        return 1.0/(1.0+math.exp(-zc))
    # placeholder for RF/XGB: assume not supported in prod yet
    return 0.5

def attach_labels_for_head(df: pd.DataFrame, code: str) -> np.ndarray:
    if code in ("BTTS","BTTS_YES"): return df["label_btts"].astype(int).values
    if code == "WIN_HOME": return df["label_win_home"].astype(int).values
    if code == "DRAW": return df["label_draw"].astype(int).values
    if code == "WIN_AWAY": return df["label_win_away"].astype(int).values
    if code.startswith("O") and len(code)==3:
        th = int(code[1:]) / 10.0
        return (df["final_total"].values >= th).astype(int)
    raise ValueError(f"Unknown head code: {code}")

# ──────────────────────────────────────────────────────────────────────────────
def precision_at_threshold(y_true: np.ndarray, p: np.ndarray, thr: float) -> Tuple[float,int]:
    mask = p >= thr
    n = int(mask.sum())
    if n == 0: return (0.0,0)
    prec = float(y_true[mask].mean())
    return (prec,n)

def find_best_threshold(y_true: np.ndarray, p: np.ndarray, candidates: List[float]) -> float:
    best_thr, best_prec = 0.5, 0.0
    for thr in candidates:
        prec,n = precision_at_threshold(y_true,p,thr)
        if n>=30 and prec>best_prec:  # require sample size
            best_thr,best_prec = thr,prec
    return best_thr

# ──────────────────────────────────────────────────────────────────────────────
def analyze(conn, min_minute: int = 15, heads_whitelist: Optional[List[str]] = None) -> Dict[str, Any]:
    df = load_latest_snapshots(conn, min_minute=min_minute)
    if df.empty: return {"ok": False, "message": "No labeled snapshots."}

    models = load_models(conn)
    if not models: return {"ok": False, "message": "No model_v2:* found."}
    if heads_whitelist:
        models = {k:v for k,v in models.items() if k in heads_whitelist}

    thresholds: Dict[str,float] = {}
    report: Dict[str,Any] = {"total_rows": int(len(df)), "heads": {}}

    for code, head in models.items():
        try: y = attach_labels_for_head(df, code)
        except: continue
        p = df.apply(lambda r: score_head_row(r, head), axis=1).values.astype(float)

        auc = roc_auc_score(y,p) if len(np.unique(y))>1 else float("nan")
        brier = brier_score_loss(y,p)
        acc = accuracy_score(y,(p>=0.5).astype(int))
        logloss = log_loss(y,p,labels=[0,1])

        # best threshold search
        cand = [0.5,0.6,0.65,0.7,0.75,0.77,0.8,0.85,0.9]
        best_thr = find_best_threshold(y,p,cand)
        thresholds[code] = best_thr

        report["heads"][code] = {
            "n": int(len(y)),
            "auc": auc,
            "brier": brier,
            "acc@0.5": acc,
            "logloss": logloss,
            "best_thr": best_thr
        }

    # save full report
    _exec(conn,
          "INSERT INTO settings(key,value) VALUES(%s,%s) "
          "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
          ("analysis:precision_v2", json.dumps(report)))
    # save thresholds
    _exec(conn,
          "INSERT INTO settings(key,value) VALUES(%s,%s) "
          "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
          ("policy:thresholds_v1", json.dumps(thresholds)))

    return {"ok": True, "report": report, "thresholds": thresholds}

# ──────────────────────────────────────────────────────────────────────────────
def main():
    db_url = os.getenv("DATABASE_URL")
    if not db_url: 
        print("DATABASE_URL required", flush=True); sys.exit(1)

    min_minute = int(os.getenv("ANALYZE_MIN_MINUTE", "15"))
    heads_env = os.getenv("ANALYZE_HEADS","").strip()
    heads = [h.strip() for h in heads_env.split(",") if h.strip()] if heads_env else None

    conn = _connect(db_url)
    try:
        result = analyze(conn, min_minute, heads)
        if not result.get("ok"):
            print(result.get("message","failed"), flush=True); sys.exit(0)

        rep = result["report"]; thrs = result["thresholds"]
        print(f"Analyzed {rep['total_rows']} snapshots.", flush=True)
        for code,r in rep["heads"].items():
            print(f"• {code}: AUC={r['auc']:.3f} | Brier={r['brier']:.4f} | BestThr={r['best_thr']:.2f}", flush=True)
        print("Saved analysis:precision_v2 + thresholds:policy:thresholds_v1", flush=True)
        sys.exit(0)
    except Exception as e:
        print(f"[ANALYZE] exception: {e}", flush=True); sys.exit(1)
    finally:
        try: conn.close()
        except: pass

if __name__ == "__main__":
    main()
