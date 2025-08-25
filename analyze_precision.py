import os, sys, json, math
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import psycopg2
from sklearn.metrics import brier_score_loss, accuracy_score, roc_auc_score

FEATURES = [
    "minute","goals_h","goals_a","goals_sum","goals_diff",
    "xg_h","xg_a","xg_sum","xg_diff",
    "sot_h","sot_a","sot_sum",
    "cor_h","cor_a","cor_sum",
    "pos_h","pos_a","pos_diff",
    "red_h","red_a","red_sum"
]

def _normalize_db_url(db_url: str) -> str:
    if not db_url: return db_url
    if db_url.startswith("postgres://"):
        db_url = "postgresql://" + db_url[len("postgres://"):]
    if "sslmode=" not in db_url:
        db_url = db_url + ("&" if "?" in db_url else "?") + "sslmode=require"
    return db_url

def _connect(db_url: str):
    conn = psycopg2.connect(_normalize_db_url(db_url)); conn.autocommit=True; return conn

def _read_sql(conn, sql: str, params: Tuple = ()):
    return pd.read_sql_query(sql, conn, params=params)

def _exec(conn, sql: str, params: Tuple): 
    with conn.cursor() as cur: cur.execute(sql, params)

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
            f = {"match_id": int(row["match_id"]),
                 "created_ts": int(row["created_ts"]),
                 "minute": float(p.get("minute", 0)),
                 "goals_h": float(p.get("gh", 0)), "goals_a": float(p.get("ga", 0)),
                 "xg_h": float(stat.get("xg_h", 0)), "xg_a": float(stat.get("xg_a", 0)),
                 "sot_h": float(stat.get("sot_h", 0)), "sot_a": float(stat.get("sot_a", 0)),
                 "cor_h": float(stat.get("cor_h", 0)), "cor_a": float(stat.get("cor_a", 0)),
                 "pos_h": float(stat.get("pos_h", 0)), "pos_a": float(stat.get("pos_a", 0)),
                 "red_h": float(stat.get("red_h", 0)), "red_a": float(stat.get("red_a", 0))}
        except Exception: continue
        f["goals_sum"] = f["goals_h"] + f["goals_a"]; f["goals_diff"] = f["goals_h"] - f["goals_a"]
        f["xg_sum"] = f["xg_h"] + f["xg_a"]; f["xg_diff"] = f["xg_h"] - f["xg_a"]
        f["sot_sum"] = f["sot_h"] + f["sot_a"]; f["cor_sum"] = f["cor_h"] + f["cor_a"]
        f["pos_diff"] = f["pos_h"] - f["pos_a"]; f["red_sum"] = f["red_h"] + f["red_a"]
        gh_f, ga_f = int(row["final_goals_h"] or 0), int(row["final_goals_a"] or 0)
        f["final_total"] = gh_f + ga_f
        f["label_btts"] = 1 if int(row["btts_yes"] or 0) == 1 else 0
        f["label_win_home"] = 1 if gh_f > ga_f else 0
        f["label_draw"] = 1 if gh_f == ga_f else 0
        f["label_win_away"] = 1 if gh_f < ga_f else 0
        feats.append(f)
    if not feats: return pd.DataFrame()
    df = pd.DataFrame(feats)
    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["minute"] = df["minute"].clip(0, 120)
    return df[df["minute"] >= min_minute].copy()

def load_models(conn) -> Dict[str, Dict[str, Any]]:
    rows = _read_sql(conn, "SELECT key, value FROM settings WHERE key LIKE 'model_v2:%'", ())
    out = {}
    for _, r in rows.iterrows():
        try: out[r["key"].split(":", 1)[1]] = json.loads(r["value"])
        except Exception: continue
    return out

def score_head_row(x: pd.Series, head: Dict[str, Any]) -> float:
    z = float(head.get("intercept", 0.0))
    for name in FEATURES: z += float(head.get("weights", {}).get(name, 0.0)) * float(x.get(name, 0.0))
    a, b = float(head.get("calibration", {}).get("a", 1.0)), float(head.get("calibration", {}).get("b", 0.0))
    zc = max(-50.0, min(50.0, a * z + b))
    return 1.0 / (1.0 + math.exp(-zc))

def attach_labels_for_head(df: pd.DataFrame, code: str) -> np.ndarray:
    if code in ("BTTS","BTTS_YES"): return df["label_btts"].astype(int).values
    if code == "WIN_HOME": return df["label_win_home"].astype(int).values
    if code == "DRAW": return df["label_draw"].astype(int).values
    if code == "WIN_AWAY": return df["label_win_away"].astype(int).values
    if code.startswith("O") and len(code) == 3: return (df["final_total"].values >= int(code[1:])/10.0).astype(int)
    raise ValueError(code)

def thresholds_for(code: str) -> List[float]:
    thrs = [0.5, 0.6, 0.7, 0.75, 0.77, 0.8, 0.85, 0.9]
    env_thr = os.getenv(code)
    if env_thr:
        try: thrs.append(float(env_thr))
        except: pass
    return sorted(set([t for t in thrs if 0 < t < 1]))

def analyze(conn, min_minute: int = 15, heads_whitelist: Optional[List[str]] = None) -> Dict[str, Any]:
    df = load_latest_snapshots(conn, min_minute)
    if df.empty: return {"ok": False, "message": "No labeled snapshots"}
    models = load_models(conn)
    if not models: return {"ok": False, "message": "No models"}
    if heads_whitelist: models = {k: v for k, v in models.items() if k in heads_whitelist}

    report: Dict[str, Any] = {"total_rows": int(len(df)), "heads": {}}
    thresholds: Dict[str,float] = {}
    for code, head in models.items():
        try: y = attach_labels_for_head(df, code)
        except: continue
        p = df.apply(lambda r: score_head_row(r, head), axis=1).values.astype(float)
        auc = float(roc_auc_score(y, p)); brier = float(brier_score_loss(y, p))
        acc = float(accuracy_score(y, (p >= 0.5).astype(int)))
        # pick threshold that maximizes precision
        best_thr, best_prec = 0.5, 0.0
        for t in thresholds_for(code):
            mask = p >= t
            if mask.sum() > 20:
                prec = float(y[mask].mean())
                if prec > best_prec: best_thr, best_prec = t, prec
        thresholds[code] = best_thr
        report["heads"][code] = {"n": int(len(y)), "auc": auc, "brier": brier,
                                 "acc@0.5": acc, "best_thr": best_thr, "precision@best": best_prec}
    _exec(conn, "INSERT INTO settings(key,value) VALUES(%s,%s) "
                "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
                ("analysis:precision_v2", json.dumps(report)))
    _exec(conn, "INSERT INTO settings(key,value) VALUES(%s,%s) "
                "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
                ("policy:thresholds_v1", json.dumps(thresholds)))
    return {"ok": True, "report": report}

def main():
    db_url = os.getenv("DATABASE_URL")
    if not db_url: sys.exit("DATABASE_URL required")
    conn = _connect(db_url)
    try:
        result = analyze(conn, min_minute=int(os.getenv("ANALYZE_MIN_MINUTE", "15")))
        if not result["ok"]: sys.exit(result["message"])
        print("Analysis done. Thresholds updated.", flush=True)
    finally: conn.close()

if __name__ == "__main__":
    main()
