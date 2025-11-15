#!/usr/bin/env python3
import os, json, time, math, logging
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

load_dotenv()
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
log = logging.getLogger("trainer")

# ── ENV ───────────────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "")
if "sslmode=" not in DATABASE_URL:
    DATABASE_URL += ("&" if "?" in DATABASE_URL else "?") + "sslmode=require"

MIN_SAMPLES_PER_MODEL = int(os.getenv("MIN_SAMPLES_PER_MODEL", "50"))  # start easy
TIP_MIN_MINUTE = int(os.getenv("TIP_MIN_MINUTE", "12"))
SNAP_MAX_MINUTE = int(os.getenv("SNAP_MAX_MINUTE", "85"))

TARGET_PRECISION = float(os.getenv("TARGET_PRECISION", "0.60"))
THRESH_MIN_PREDICTIONS = int(os.getenv("THRESH_MIN_PREDICTIONS", "25"))
MIN_THRESH = float(os.getenv("MIN_THRESH", "55"))
MAX_THRESH = float(os.getenv("MAX_THRESH", "85"))
APPLY_TUNE_PREC_TOL = float(os.getenv("APPLY_TUNE_PREC_TOL", "0.03"))
MAX_ODDS_ALL = float(os.getenv("MAX_ODDS_ALL", "20.0"))

# common feature whitelist (if we need to pick from messy payloads)
WHITELIST = {
    "minute","goals_h","goals_a","goals_sum","goals_diff",
    "xg_h","xg_a","xg_sum","xg_diff",
    "sot_h","sot_a","sot_sum",
    "sh_total_h","sh_total_a",
    "cor_h","cor_a","cor_sum",
    "pos_h","pos_a","pos_diff",
    "red_h","red_a","red_sum",
    "yellow_h","yellow_a"
}

# ── DB helpers ────────────────────────────────────────────────────────────────
def _conn():
    return psycopg2.connect(DATABASE_URL)

def get_rows(sql: str, params: tuple = ()):
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params); return cur.fetchall()

def exec_sql(sql: str, params: tuple = ()):
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)

def set_setting(key: str, value: str) -> None:
    exec_sql("INSERT INTO settings(key,value) VALUES(%s,%s) ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value", (key, value))

# ── math utils ────────────────────────────────────────────────────────────────
def _sigmoid(x: float) -> float:
    if x < -50: return 1e-22
    if x >  50: return 1-1e-22
    return 1.0/(1.0+math.exp(-x))

def _logit(p: float) -> float:
    p = max(1e-12, min(1-1e-12, float(p)))
    return math.log(p/(1.0-p))

def _fit_platt(probs: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    z = np.array([_logit(p) for p in probs]).reshape(-1, 1)
    lr = LogisticRegression(solver="liblinear", max_iter=200)
    lr.fit(z, y.astype(int))
    return float(lr.coef_[0][0]), float(lr.intercept_[0])

# ── payload parsing (robust) ──────────────────────────────────────────────────
def _to_float(v) -> Optional[float]:
    try:
        if isinstance(v, str):
            s = v.strip()
            if s.endswith("%"):
                return float(s[:-1])
            return float(s)
        if isinstance(v, (int, float)):
            return float(v)
        return None
    except Exception:
        return None

def _flatten_numeric(obj: Any, out: Dict[str, float], prefix: str = "") -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}{k}" if prefix == "" else f"{prefix}.{k}"
            if isinstance(v, (dict, list)):
                _flatten_numeric(v, out, key)
            else:
                fv = _to_float(v)
                if fv is not None:
                    leaf = key.split(".")[-1]
                    # keep either whitelisted names or anything that looks numeric
                    if leaf in WHITELIST or prefix.endswith("stat"):
                        out[leaf] = fv
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _flatten_numeric(v, out, f"{prefix}[{i}]")
    else:
        fv = _to_float(obj)
        if fv is not None and (prefix or "value") in WHITELIST:
            out[prefix or "value"] = fv

def _maybe_json(s: str) -> Any:
    try:
        js = json.loads(s)
        # double-encoded?
        if isinstance(js, str) and js.strip().startswith("{"):
            return json.loads(js)
        return js
    except Exception:
        return {}

def _features_from_payload(raw: str) -> Dict[str, float]:
    """
    Accepts many shapes:
    - {"stat": {...}}  ← preferred
    - flat numeric dict
    - nested dicts (we flatten numerical leaves)
    - double-encoded JSON
    """
    try:
        js = _maybe_json(raw or "{}")
        if not isinstance(js, dict):
            return {}
        # prefer stat if present
        source = js.get("stat") if isinstance(js.get("stat"), (dict, list)) else js
        out: Dict[str, float] = {}
        _flatten_numeric(source, out)
        # soft prune: only keep whitelisted + minute if present
        if out:
            out2 = {}
            for k, v in out.items():
                if k in WHITELIST or k == "minute":
                    out2[k] = float(v)
            if not out2:  # if pruning killed everything, keep the original numeric set
                out2 = {k: float(v) for k, v in out.items()}
            return out2
        return {}
    except Exception:
        return {}

# ── Loaders ───────────────────────────────────────────────────────────────────
def _load_training_rows_from_tips(days: int = 90) -> List[Tuple[dict, str, int]]:
    cutoff = int(time.time()) - days * 24 * 3600
    sql = """
      SELECT t.match_id, t.suggestion, r.final_goals_h, r.final_goals_a, r.btts_yes,
             (
                SELECT s.payload FROM tip_snapshots s
                WHERE s.match_id = t.match_id AND s.created_ts <= t.created_ts
                ORDER BY s.created_ts DESC LIMIT 1
             ) AS payload
      FROM tips t
      JOIN match_results r ON r.match_id = t.match_id
      WHERE t.created_ts >= %s
        AND t.suggestion <> 'HARVEST'
        AND t.odds BETWEEN 1.01 AND %s
    """
    rows = get_rows(sql, (cutoff, MAX_ODDS_ALL))
    out: List[Tuple[dict, str, int]] = []
    for (mid, sugg, gh, ga, btts, payload) in rows:
        feat = _features_from_payload(payload or "{}")
        if not feat: continue
        minute = int(float(feat.get("minute", 0)))
        if minute and minute < TIP_MIN_MINUTE:  # if we know minute, respect it
            continue
        total = int(gh or 0) + int(ga or 0)
        if sugg.startswith("Over"):
            line = None
            for tok in sugg.split():
                try: line = float(tok); break
                except: pass
            if line is None: continue
            y = 1 if total > line else (0 if total < line else None)
            if y is None: continue
            out.append((feat, f"Over {line} Goals", y))
        elif sugg == "BTTS: Yes":
            out.append((feat, "BTTS: Yes", 1 if int(btts or 0) == 1 else 0))
        elif sugg == "BTTS: No":
            out.append((feat, "BTTS: Yes", 0 if int(btts or 0) == 1 else 1))  # train YES model
        elif sugg == "Home Win":
            out.append((feat, "Home Win", 1 if int(gh or 0) > int(ga or 0) else 0))
        elif sugg == "Away Win":
            out.append((feat, "Away Win", 1 if int(ga or 0) > int(gh or 0) else 0))
    log.info("[TRAIN] rows from tips+odds: %d", len(out))
    return out

def _load_training_rows_from_snapshots(days: int = 730, per_match: str = "first_viable") -> List[Tuple[dict, str, int]]:
    cutoff = int(time.time()) - days * 24 * 3600
    # only matches with results
    mids = get_rows("""
        SELECT DISTINCT s.match_id
        FROM tip_snapshots s
        JOIN match_results r ON r.match_id = s.match_id
        WHERE s.created_ts >= %s
    """, (cutoff,))
    mids = [int(x[0]) for x in mids]
    if not mids:
        log.info("[TRAIN] no snapshot matches with results in window")
        return []
    # stream snapshots and pick one per match
    rows = get_rows(
        "SELECT match_id, created_ts, payload FROM tip_snapshots WHERE match_id = ANY(%s) ORDER BY match_id, created_ts ASC",
        (mids,)
    )
    by_mid: Dict[int, List[Tuple[int, str]]] = {}
    for (mid, cts, payload) in rows:
        by_mid.setdefault(int(mid), []).append((int(cts), payload))

    # results map
    res_map = {}
    res_rows = get_rows("SELECT match_id, final_goals_h, final_goals_a, btts_yes FROM match_results WHERE match_id = ANY(%s)", (mids,))
    for (mid, gh, ga, btts) in res_rows:
        res_map[int(mid)] = (int(gh or 0), int(ga or 0), int(btts or 0))

    out: List[Tuple[dict, str, int]] = []
    used = 0
    usable = 0

    for mid, arr in by_mid.items():
        pick_feat = None
        if per_match == "latest":
            _, payload = arr[-1]
            feat = _features_from_payload(payload)
            if feat: pick_feat = feat
        else:  # earliest viable
            for _, payload in arr:
                feat = _features_from_payload(payload)
                if not feat: continue
                minute = int(float(feat.get("minute", 0)))
                if minute == 0:  # old payloads: accept; else enforce min
                    pick_feat = feat; break
                if TIP_MIN_MINUTE <= minute <= SNAP_MAX_MINUTE:
                    pick_feat = feat; break
        if not pick_feat: 
            continue
        usable += 1
        gh, ga, btts = res_map.get(mid, (None, None, None))
        if gh is None: continue
        total = gh + ga
        out.append((pick_feat, "Over 2.5 Goals", 1 if total > 2.5 else 0))
        out.append((pick_feat, "Over 3.5 Goals", 1 if total > 3.5 else 0))
        out.append((pick_feat, "BTTS: Yes",      1 if btts == 1 else 0))
        out.append((pick_feat, "Home Win",       1 if gh > ga else 0))
        out.append((pick_feat, "Away Win",       1 if ga > gh else 0))
        used += 1

    log.info("[TRAIN] snapshots: matches=%d usable=%d rows=%d", len(by_mid), usable, len(out))
    return out

# ── model training ────────────────────────────────────────────────────────────
def _train_one(rows: List[Tuple[dict, str, int]], label_key: str) -> Optional[Dict[str, Any]]:
    X: List[List[float]] = []
    y: List[int] = []

    # union feature set
    keys_set = set()
    for feat, lab, yy in rows:
        if lab == label_key and feat:
            keys_set.update(feat.keys())
    feature_names = sorted(list(keys_set))
    if not feature_names:
        log.info("[TRAIN] %s skipped — no usable features", label_key); return None

    for feat, lab, yy in rows:
        if lab != label_key: continue
        X.append([float(feat.get(k, 0.0) or 0.0) for k in feature_names])
        y.append(int(yy))

    if len(y) < MIN_SAMPLES_PER_MODEL:
        log.info("[TRAIN] %s skipped — %d < %d samples", label_key, len(y), MIN_SAMPLES_PER_MODEL); return None

    Xn = np.asarray(X, dtype=float)
    yn = np.asarray(y, dtype=int)

    lr = LogisticRegression(solver="liblinear", max_iter=1000)
    lr.fit(Xn, yn)
    raw_scores = lr.decision_function(Xn)
    raw_probs = 1 / (1 + np.exp(-raw_scores))

    try:
        _, X_te, _, y_te, p_te = train_test_split(Xn, yn, raw_probs, test_size=0.2, random_state=42, stratify=yn)
        a, b = _fit_platt(p_te, y_te)
    except Exception:
        a, b = 1.0, 0.0

    weights = {feature_names[i]: float(lr.coef_[0][i]) for i in range(len(feature_names))}
    intercept = float(lr.intercept_[0])

    bayes = {  # simple NB overlay; harmless if features missing
        "features": {
            "sot_sum": {"bins": [0, 2, 5, 10, 99], "lr": [0.8, 1.0, 1.2, 1.4]},
            "xg_sum":  {"bins": [0, .5, 1.0, 2.0, 99], "lr": [0.8, 1.0, 1.2, 1.4]},
            "cor_sum": {"bins": [0, 2, 5,  9, 99], "lr": [0.9, 1.0, 1.1, 1.2]}
        }
    }
    return {"intercept": intercept, "weights": weights,
            "calibration": {"method": "sigmoid", "a": float(a), "b": float(b)},
            "bayes": bayes}

def _save_model(key: str, mdl: Dict[str, Any]) -> None:
    set_setting(key, json.dumps(mdl, separators=(",", ":"), ensure_ascii=False))

def train_models(days: int = 365) -> Dict[str, Any]:
    # prefer tips+odds
    rows = _load_training_rows_from_tips(days=90)
    if sum(1 for r in rows if r[1] in ("Over 2.5 Goals","Over 3.5 Goals","BTTS: Yes","Home Win","Away Win")) < MIN_SAMPLES_PER_MODEL:
        rows = _load_training_rows_from_snapshots(days=730, per_match="first_viable")

    # visibility
    counts = {}
    for key in ("Over 2.5 Goals","Over 3.5 Goals","BTTS: Yes","Home Win","Away Win"):
        counts[key] = sum(1 for r in rows if r[1] == key)
    log.info("[TRAIN] label counts: %s", counts)

    results: Dict[str, Any] = {"ok": True, "trained": {}}

    mdl = _train_one(rows, "Over 2.5 Goals")
    results["trained"]["OU_2.5"] = bool(mdl)
    if mdl: _save_model("OU_2.5", mdl)

    mdl = _train_one(rows, "Over 3.5 Goals")
    results["trained"]["OU_3.5"] = bool(mdl)
    if mdl: _save_model("OU_3.5", mdl)

    mdl = _train_one(rows, "BTTS: Yes")
    results["trained"]["BTTS_YES"] = bool(mdl)
    if mdl: _save_model("BTTS_YES", mdl)

    mdl = _train_one(rows, "Home Win")
    results["trained"]["WLD_HOME"] = bool(mdl)
    if mdl: _save_model("WLD_HOME", mdl)

    mdl = _train_one(rows, "Away Win")
    results["trained"]["WLD_AWAY"] = bool(mdl)
    if mdl: _save_model("WLD_AWAY", mdl)

    # keep a non-zero draw model for normalization
    draw_mdl = {"intercept": 0.0, "weights": {}, "calibration": {"method": "sigmoid", "a": 1.0, "b": 0.0}}
    _save_model("WLD_DRAW", draw_mdl); results["trained"]["WLD_DRAW"] = True

    if not any(v for k, v in results["trained"].items() if k != "WLD_DRAW"):
        results["ok"] = False
        results["reason"] = f"Not enough usable samples per label (counts={counts}, min={MIN_SAMPLES_PER_MODEL})"
    return results

# ── ROI-aware auto tune (unchanged) ───────────────────────────────────────────
def auto_tune_thresholds(days: int = 14) -> Dict[str, float]:
    cutoff = int(time.time()) - days * 24 * 3600
    rows = get_rows(
        """
        SELECT t.market, t.suggestion, COALESCE(t.confidence_raw, t.confidence/100.0) AS prob, t.odds,
               r.final_goals_h, r.final_goals_a, r.btts_yes
        FROM tips t
        JOIN match_results r ON r.match_id = t.match_id
        WHERE t.created_ts >= %s
          AND t.suggestion <> 'HARVEST'
          AND t.sent_ok = 1
          AND t.odds IS NOT NULL
        """, (cutoff,)
    )
    if not rows:
        set_setting("last_auto_tune", "no data"); return {}

    by: Dict[str, List[Tuple[float, int, float]]] = {}
    for (mk, sugg, prob, odds, gh, ga, btts) in rows:
        try: prob=float(prob or 0.0); odds=float(odds or 0.0)
        except: continue
        if not (1.01 <= odds <= MAX_ODDS_ALL): continue

        def _lbl(s: str) -> Optional[int]:
            total = int(gh or 0) + int(ga or 0)
            if s.startswith("Over"):
                line=None
                for tok in s.split():
                    try: line=float(tok); break
                    except: pass
                if line is None: return None
                return 1 if total>line else (0 if total<line else None)
            if s=="BTTS: Yes": return 1 if int(btts or 0)==1 else 0
            if s=="BTTS: No":  return 1 if int(btts or 0)==0 else 0
            if s=="Home Win":  return 1 if int(gh or 0)>int(ga or 0) else 0
            if s=="Away Win":  return 1 if int(ga or 0)>int(gh or 0) else 0
            return None

        y=_lbl(sugg)
        if y is None: continue
        mk_key = mk.replace("PRE ", "")
        by.setdefault(mk_key, []).append((prob, int(y), odds))

    tuned: Dict[str, float] = {}
    def _eval(items: List[Tuple[float,int,float]], thr_prob: float) -> Tuple[int,float,float]:
        sel=[(p,y,o) for (p,y,o) in items if p>=thr_prob]
        n=len(sel)
        if n==0: return 0,0.0,0.0
        wins=sum(y for (_,y,_) in sel); prec=wins/n
        roi=sum((y*(od-1.0)-(1-y)) for (_,y,od) in sel)/n
        return n,float(prec),float(roi)

    for mk, items in by.items():
        if len(items) < THRESH_MIN_PREDICTIONS: continue
        best=None; feasible=False
        for thr_pct in np.arange(MIN_THRESH, MAX_THRESH+1e-9, 1.0):
            n,prec,roi=_eval(items, thr_pct/100.0)
            if n<THRESH_MIN_PREDICTIONS: continue
            if prec>=TARGET_PRECISION:
                feasible=True
                score=(roi,prec,n)
                if (best is None) or (score>(best[0],best[1],best[2])):
                    best=(roi,prec,n,thr_pct)
        if not feasible:
            for thr_pct in np.arange(MIN_THRESH, MAX_THRESH+1e-9, 1.0):
                n,prec,roi=_eval(items, thr_pct/100.0)
                if n<THRESH_MIN_PREDICTIONS: continue
                if (prec>=max(0.0,TARGET_PRECISION-APPLY_TUNE_PREC_TOL)) and (roi>0.0):
                    score=(roi,prec,n)
                    if (best is None) or (score>(best[0],best[1],best[2])):
                        best=(roi,prec,n,thr_pct)
        if best is not None:
            tuned[mk]=float(best[3])

    for k,pct in tuned.items():
        set_setting(f"conf_threshold:{k}", f"{pct:.2f}")
    return tuned

if __name__ == "__main__":
    print(json.dumps(train_models(), indent=2))
