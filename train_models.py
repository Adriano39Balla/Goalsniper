#!/usr/bin/env python3
"""
Training module for goalsniper
- Fits logistic regression models (with regularization) for OU(2.5,3.5), BTTS, 1X2(Home/Draw/Away)
- Adds a Naive Bayesian evidence overlay by learning likelihood ratios from binned in-play features
- Calibrates probabilities (Platt scaling / sigmoid)
- Saves models as JSON into the `settings` table (keys: OU_2.5, OU_3.5, BTTS_YES, WLD_HOME, WLD_DRAW, WLD_AWAY)
- Provides ROI/precision-aware auto-tuning of per-market thresholds (saved in settings as conf_threshold:*)
"""
import os, json, time, logging
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import psycopg2
from psycopg2.pool import SimpleConnectionPool

from pydantic_settings import BaseSettings, SettingsConfigDict

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
log = logging.getLogger("train_models")

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False)
    DATABASE_URL: str = ""
    MIN_SAMPLES_PER_MODEL: int = 400
    APPLY_TUNE_PREC_TOL: float = float(os.getenv("APPLY_TUNE_PREC_TOL", "0.03"))
    TARGET_PRECISION: float = float(os.getenv("TARGET_PRECISION","0.60"))
    THRESH_MIN_PREDICTIONS: int = int(os.getenv("THRESH_MIN_PREDICTIONS","25"))
    MIN_THRESH: float = float(os.getenv("MIN_THRESH","55"))
    MAX_THRESH: float = float(os.getenv("MAX_THRESH","85"))

settings = Settings()

POOL: Optional[SimpleConnectionPool] = None

def _init_pool():
    global POOL
    dsn = settings.DATABASE_URL
    if "sslmode=" not in dsn:
        dsn += ("&" if "?" in dsn else "?") + "sslmode=require"
    POOL = SimpleConnectionPool(minconn=1, maxconn=3, dsn=dsn)

def db_conn():
    if not POOL: _init_pool()
    class PooledConn:
        def __init__(self, pool): self.pool=pool; self.conn=None; self.cur=None
        def __enter__(self):
            self.conn=self.pool.getconn(); self.conn.autocommit=True; self.cur=self.conn.cursor(); return self
        def __exit__(self, a,b,c):
            try: self.cur and self.cur.close()
            finally: self.conn and self.pool.putconn(self.conn)
        def execute(self, sql, params=()):
            self.cur.execute(sql, params or ()); return self.cur
    return PooledConn(POOL)  # type: ignore

# ──────────────────────────────────────────────────────────────────────────────
# Data extraction helpers
# ──────────────────────────────────────────────────────────────────────────────
def _load_training_rows(days: int = 30) -> List[Tuple[dict, str, int, Optional[float]]]:
    """
    Pull labeled tips with odds from the past window and join with match_results.
    Returns: [(features_dict, suggestion, outcome(0/1), odds), ...]
    """
    cutoff = int(time.time()) - days * 24 * 3600
    with db_conn() as c:
        rows = c.execute(
            """
            SELECT t.market, t.suggestion, t.confidence, t.confidence_raw, t.created_ts, t.odds,
                   s.payload, r.final_goals_h, r.final_goals_a, r.btts_yes
            FROM tips t
            JOIN tip_snapshots s ON (s.match_id = t.match_id AND s.created_ts <= t.created_ts)
            JOIN match_results r ON (r.match_id = t.match_id)
            WHERE t.created_ts >= %s AND t.suggestion <> 'HARVEST' AND t.odds IS NOT NULL AND t.sent_ok=1
            """,
            (cutoff,),
        ).fetchall()
    out = []
    for (market, suggestion, conf, conf_raw, created_ts, odds, payload, gh, ga, btts) in rows:
        try:
            snap = json.loads(payload)
            feat = snap.get("stat") or {}
            if not feat:
                continue
            res={"final_goals_h":gh,"final_goals_a":ga,"btts_yes":btts}
            y = _tip_outcome_for_result(suggestion, res)
            if y is None:
                continue
            out.append((feat, suggestion, int(y), float(odds or 0.0)))
        except Exception:
            continue
    return out

def _tip_outcome_for_result(suggestion: str, res: Dict[str,Any]) -> Optional[int]:
    gh=int(res.get("final_goals_h") or 0); ga=int(res.get("final_goals_a") or 0)
    total=gh+ga; btts=int(res.get("btts_yes") or 0); s=(suggestion or "").strip()
    def _parse_line(s: str)->Optional[float]:
        for tok in (s or "").split():
            try: return float(tok)
            except: pass
        return None
    if s.startswith("Over") or s.startswith("Under"):
        line=_parse_line(s)
        if line is None: return None
        if s.startswith("Over"):
            if total>line: return 1
            if abs(total-line)<1e-9: return None
            return 0
        else:
            if total<line: return 1
            if abs(total-line)<1e-9: return None
            return 0
    if s=="BTTS: Yes": return 1 if btts==1 else 0
    if s=="BTTS: No":  return 1 if btts==0 else 0
    if s=="Home Win":  return 1 if gh>ga else 0
    if s=="Away Win":  return 1 if ga>gh else 0
    return None

# ──────────────────────────────────────────────────────────────────────────────
# Model training
# ──────────────────────────────────────────────────────────────────────────────
FEATURES = ["minute","goals_sum","xg_sum","xg_diff","sot_sum","sh_total_h","sh_total_a","cor_sum","pos_diff","red_sum","yellow_h","yellow_a"]

def _Xy_for_market(rows, market: str, selector):
    X=[]; y=[]
    for feat, sugg, label, _odds in rows:
        ok, tgt = selector(sugg)
        if not ok: continue
        xi=[float(feat.get(k,0.0)) for k in FEATURES]
        X.append(xi); y.append(int(label if tgt else (1-label)))
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)

def _fit_lr_calibrated(X, y) -> Tuple[Dict[str, float], float, Dict[str, float]]:
    if len(X) == 0: raise ValueError("no samples")
    pipe = Pipeline([("scaler", StandardScaler(with_mean=False)), ("lr", LogisticRegression(max_iter=200, C=1.5))])
    clf = CalibratedClassifierCV(pipe, method="sigmoid", cv=3)
    clf.fit(X, y)
    # derive linear surrogate
    lr = clf.base_estimator.named_steps["lr"]
    coefs = lr.coef_[0]
    intercept = float(lr.intercept_[0])
    # Platt scaling on raw proba
    from sklearn.linear_model import LogisticRegression as LR2
    p_raw = pipe.fit(X, y).predict_proba(X)[:,1]
    lr2 = LR2(max_iter=200).fit(p_raw.reshape(-1,1), y)
    a = float(lr2.coef_[0][0]); b=float(lr2.intercept_[0])

    # Export weights back to original feature scale (simple variance proxy)
    std = np.sqrt((np.asarray(X)**2).mean(axis=0) + 1e-9)
    weights = {FEATURES[i]: float(coefs[i]/max(1e-9,std[i])) for i in range(len(FEATURES))}
    return weights, float(intercept), {"method":"platt", "a": a, "b": b}

def _learn_naive_bayes_overlay(rows, selector) -> Dict[str, Any]:
    """
    Build Naive Bayes likelihood ratios per binned feature for P(Y=1|X).
    """
    bins_def = {
        "minute": [0, 15, 30, 45, 60, 75, 120],
        "goals_sum": [0, 1, 2, 3, 10],
        "xg_sum": [0, 0.4, 0.8, 1.2, 2.0, 4.0],
        "sot_sum": [0, 1, 3, 6, 10, 20],
        "cor_sum": [0, 2, 5, 9, 20],
        "pos_diff": [-100, -20, -5, 5, 20, 100],
        "red_sum": [0, 1, 2, 5],
    }
    counts = {k: np.zeros((len(bins_def[k])-1, 2), dtype=float) for k in bins_def.keys()}
    total = np.zeros(2, dtype=float)

    def bin_index(val, edges):
        for i in range(len(edges)-1):
            if edges[i] <= val < edges[i+1]: return i
        return len(edges)-2

    for feat, sugg, label, _ in rows:
        ok, tgt = selector(sugg)
        if not ok: continue
        y = int(label if tgt else (1-label))
        total[y]+=1
        for k, edges in bins_def.items():
            v = float(feat.get(k, 0.0))
            i = bin_index(v, edges)
            counts[k][i, y] += 1

    features = {}
    for k, mat in counts.items():
        y1 = mat[:,1] + 1.0
        y0 = mat[:,0] + 1.0
        lr = (y1 / max(1.0, total[1] + mat.shape[0])) / (y0 / max(1.0, total[0] + mat.shape[0]))
        features[k] = {"bins": list(map(float, bins_def[k])), "lr": [float(x) for x in lr]}
    return {"features": features}

def _save_model(key: str, weights: Dict[str,float], intercept: float, calib: Dict[str,float], bayes: Dict[str,Any]) -> None:
    payload = {"weights": weights, "intercept": float(intercept), "calibration": calib, "bayes": bayes}
    with db_conn() as c:
        c.execute("INSERT INTO settings(key,value) VALUES(%s,%s) ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
                  (key, json.dumps(payload, separators=(',',':'))))

def train_models(days: int = 30) -> Dict[str, Any]:
    rows = _load_training_rows(days)
    if len(rows) < settings.MIN_SAMPLES_PER_MODEL:
        msg = f"Not enough samples to train: {len(rows)} < {settings.MIN_SAMPLES_PER_MODEL}"
        log.warning(msg)
        return {"ok": False, "reason": msg, "samples": len(rows)}

    # OU 2.5
    def sel_ou25(s): 
        if s.startswith("Over 2.5"): return True, True
        if s.startswith("Under 2.5"): return True, False
        return False, False
    X, y = _Xy_for_market(rows, "OU_2.5", sel_ou25)
    w, b, cal = _fit_lr_calibrated(X, y)
    bayes = _learn_naive_bayes_overlay(rows, sel_ou25)
    _save_model("OU_2.5", w, b, cal, bayes)

    # OU 3.5
    def sel_ou35(s): 
        if s.startswith("Over 3.5"): return True, True
        if s.startswith("Under 3.5"): return True, False
        return False, False
    X, y = _Xy_for_market(rows, "OU_3.5", sel_ou35)
    if len(X) > 50:
        w, b, cal = _fit_lr_calibrated(X, y); bayes = _learn_naive_bayes_overlay(rows, sel_ou35); _save_model("OU_3.5", w, b, cal, bayes)

    # BTTS
    def sel_btts(s):
        if s == "BTTS: Yes": return True, True
        if s == "BTTS: No":  return True, False
        return False, False
    X, y = _Xy_for_market(rows, "BTTS", sel_btts)
    w, b, cal = _fit_lr_calibrated(X, y)
    bayes = _learn_naive_bayes_overlay(rows, sel_btts)
    _save_model("BTTS_YES", w, b, cal, bayes)

    # 1X2 (draw suppressed)
    def sel_home(s):
        if s == "Home Win": return True, True
        if s == "Away Win": return True, False
        return False, False
    def sel_away(s):
        if s == "Away Win": return True, True
        if s == "Home Win": return True, False
        return False, False
    def sel_draw(s):
        if s in ("Home Win","Away Win"): return True, False
        return False, False

    Xh, yh = _Xy_for_market(rows, "1X2_HOME", sel_home); wh, bh, calh = _fit_lr_calibrated(Xh, yh)
    Xd, yd = _Xy_for_market(rows, "1X2_DRAW", sel_draw); 
    if len(Xd)>0:
        wd, bd, cald = _fit_lr_calibrated(Xd, yd)
    else:
        wd, bd, cald = {}, 0.0, {"method":"platt","a":1.0,"b":0.0}
    Xa, ya = _Xy_for_market(rows, "1X2_AWAY", sel_away); wa, ba, cala = _fit_lr_calibrated(Xa, ya)
    bayes_wld = _learn_naive_bayes_overlay(rows, lambda s: (s in ("Home Win","Away Win"), s=="Home Win"))
    _save_model("WLD_HOME", wh, bh, calh, bayes_wld)
    _save_model("WLD_DRAW", wd, bd, cald, bayes_wld)
    _save_model("WLD_AWAY", wa, ba, cala, bayes_wld)

    return {"ok": True, "trained": {"OU_2.5": True, "OU_3.5": len(X)>50, "BTTS_YES": True, "WLD_HOME": True, "WLD_AWAY": True}}

# ──────────────────────────────────────────────────────────────────────────────
# Auto-tune thresholds (ROI-aware, with precision floor)
# ──────────────────────────────────────────────────────────────────────────────
def auto_tune_thresholds(days: int = 14) -> Dict[str, float]:
    cutoff = int(time.time()) - days * 24 * 3600
    with db_conn() as c:
        rows = c.execute(
            """
            SELECT t.market,
                   t.suggestion,
                   COALESCE(t.confidence_raw, t.confidence/100.0) AS prob,
                   t.odds,
                   r.final_goals_h, r.final_goals_a, r.btts_yes
            FROM tips t
            JOIN match_results r ON r.match_id = t.match_id
            WHERE t.created_ts >= %s
              AND t.suggestion <> 'HARVEST'
              AND t.sent_ok = 1
              AND t.odds IS NOT NULL
            """,
            (cutoff,),
        ).fetchall()
    if not rows:
        return {}

    by: dict[str, list[tuple[float, int, float]]] = {}
    for (mk, sugg, prob, odds, gh, ga, btts) in rows:
        try:
            prob = float(prob or 0.0); odds = float(odds or 0.0)
        except Exception: 
            continue
        y = _label_from_suggestion(sugg, {"final_goals_h": gh, "final_goals_a": ga, "btts_yes": btts})
        if y is None: continue
        if not (1.01 <= odds <= 20.0): continue
        by.setdefault(mk, []).append((prob, y, odds))

    tuned: Dict[str, float] = {}
    def _eval(items, thr_prob):
        sel = [(p, y, o) for (p,y,o) in items if p>=thr_prob]
        n=len(sel); 
        if n==0: return 0, 0.0, 0.0
        wins=sum(y for (_,y,_) in sel); prec=wins/max(1,n)
        roi=sum((y*(o-1.0)-(1-y)) for (_,y,o) in sel)/n
        return n, float(prec), float(roi)

    for mk, items in by.items():
        if len(items) < settings.THRESH_MIN_PREDICTIONS: continue
        best=None; feasible_any=False
        for thr_pct in np.arange(settings.MIN_THRESH, settings.MAX_THRESH+1e-9, 1.0):
            thr_prob = thr_pct/100.0
            n, prec, roi = _eval(items, thr_prob)
            if n < settings.THRESH_MIN_PREDICTIONS: continue
            if prec >= settings.TARGET_PRECISION:
                feasible_any=True
                score=(roi,prec,n)
                if (best is None) or (score>(best[0], best[1], best[2])): best=(roi,prec,n,thr_pct)
        if not feasible_any:
            for thr_pct in np.arange(settings.MIN_THRESH, settings.MAX_THRESH+1e-9, 1.0):
                thr_prob = thr_pct/100.0
                n, prec, roi = _eval(items, thr_prob)
                if n < settings.THRESH_MIN_PREDICTIONS: continue
                if (prec >= max(0.0, settings.TARGET_PRECISION - settings.APPLY_TUNE_PREC_TOL)) and (roi > 0.0):
                    score=(roi,prec,n)
                    if (best is None) or (score>(best[0], best[1], best[2])): best=(roi,prec,n,thr_pct)
        if best is None:
            continue
        tuned[mk] = float(best[3])

    with db_conn() as c:
        for mk, pct in tuned.items():
            c.execute("INSERT INTO settings(key,value) VALUES(%s,%s) ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
                      (f"conf_threshold:{mk}", f"{pct:.2f}"))
    return tuned

def _label_from_suggestion(suggestion: str, res: Dict[str,Any]) -> Optional[int]:
    gh=int(res.get("final_goals_h") or 0); ga=int(res.get("final_goals_a") or 0)
    total=gh+ga; btts=int(res.get("btts_yes") or 0); s=(suggestion or "").strip()
    def _parse_line(s: str)->Optional[float]:
        for tok in (s or "").split():
            try: return float(tok)
            except: pass
        return None
    if s.startswith("Over") or s.startswith("Under"):
        line=_parse_line(s); 
        if line is None: return None
        if s.startswith("Over"):
            if total>line: return 1
            if abs(total-line)<1e-9: return None
            return 0
        else:
            if total<line: return 1
            if abs(total-line)<1e-9: return None
            return 0
    if s=="BTTS: Yes": return 1 if btts==1 else 0
    if s=="BTTS: No":  return 1 if btts==0 else 0
    if s=="Home Win":  return 1 if gh>ga else 0
    if s=="Away Win":  return 1 if ga>gh else 0
    return None

if __name__ == "__main__":
    out = train_models()
    print(json.dumps(out, indent=2))
