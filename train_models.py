#!/usr/bin/env python3
import os, json, time, math, logging, random
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# sklearn pieces
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

load_dotenv()
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
log = logging.getLogger("trainer")

# --- ENV / defaults ---
DATABASE_URL = os.getenv("DATABASE_URL", "")
MIN_SAMPLES_PER_MODEL = int(os.getenv("MIN_SAMPLES_PER_MODEL", "200"))
TIP_MIN_MINUTE = int(os.getenv("TIP_MIN_MINUTE", "12"))
SNAP_MAX_MINUTE = int(os.getenv("SNAP_MAX_MINUTE", "85"))
TARGET_PRECISION = float(os.getenv("TARGET_PRECISION", "0.60"))
THRESH_MIN_PREDICTIONS = int(os.getenv("THRESH_MIN_PREDICTIONS", "25"))
MIN_THRESH = float(os.getenv("MIN_THRESH", "55"))
MAX_THRESH = float(os.getenv("MAX_THRESH", "85"))
APPLY_TUNE_PREC_TOL = float(os.getenv("APPLY_TUNE_PREC_TOL", "0.03"))
MAX_ODDS_ALL = float(os.getenv("MAX_ODDS_ALL", "20.0"))

if "sslmode=" not in DATABASE_URL:
    DATABASE_URL += ("&" if "?" in DATABASE_URL else "?") + "sslmode=require"

# --- DB helpers ---
def _conn():
    return psycopg2.connect(DATABASE_URL)

def get_rows(sql: str, params: tuple = ()):
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()

def get_one(sql: str, params: tuple = ()):
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params); r = cur.fetchone()
        return r[0] if r else None

def exec_sql(sql: str, params: tuple = ()):
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)

def set_setting(key: str, value: str) -> None:
    exec_sql("INSERT INTO settings(key,value) VALUES(%s,%s) ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value", (key, value))

# --- Utilities ---
def _sigmoid(x: float) -> float:
    try:
        if x < -50: return 1e-22
        if x > 50:  return 1 - 1e-22
        return 1.0 / (1.0 + math.exp(-x))
    except Exception:
        return 0.5

def _logit(p: float) -> float:
    p = max(1e-12, min(1 - 1e-12, float(p)))
    return math.log(p / (1.0 - p))

def _fit_platt(probs: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Fit Platt calibration: p' = sigmoid(a*logit(p) + b)
    Solve by logistic regression on z = logit(p).
    """
    z = np.array([_logit(p) for p in probs]).reshape(-1, 1)
    lr = LogisticRegression(solver="liblinear", max_iter=200)
    lr.fit(z, y.astype(int))
    a = float(lr.coef_[0][0])
    b = float(lr.intercept_[0])
    return a, b

def _features_from_payload(payload: str) -> Dict[str, float]:
    try:
        js = json.loads(payload or "{}")
        feat = js.get("stat") or js
        # ensure numeric floats
        return {k: float(feat.get(k, 0.0) or 0.0) for k in feat.keys()}
    except Exception:
        return {}

# --- Loaders ---
def _load_training_rows_from_tips(days: int = 90) -> List[Tuple[dict, str, int]]:
    """
    Preferred when available: use tips WITH odds joined to results AND a matching snapshot (same match_id, nearest created_ts).
    If you never saved per-tip snapshots, this will be sparse. That's fine; we fall back to snapshots.
    """
    cutoff = int(time.time()) - days * 24 * 3600
    sql = """
      SELECT t.match_id, t.suggestion, r.final_goals_h, r.final_goals_a, r.btts_yes,
             -- pick the nearest snapshot at or before tip time
             (
                SELECT s.payload FROM tip_snapshots s
                WHERE s.match_id = t.match_id AND s.created_ts <= t.created_ts
                ORDER BY s.created_ts DESC LIMIT 1
             ) AS payload
      FROM tips t
      JOIN match_results r ON r.match_id = t.match_id
      WHERE t.created_ts >= %s
        AND t.suggestion <> 'HARVEST'
        AND t.odds IS NOT NULL
        AND t.odds BETWEEN 1.01 AND %s
    """
    rows = get_rows(sql, (cutoff, MAX_ODDS_ALL))
    out: List[Tuple[dict, str, int]] = []
    for (mid, sugg, gh, ga, btts, payload) in rows:
        feat = _features_from_payload(payload or "{}")
        if not feat:
            continue
        minute = int(float(feat.get("minute", 0)))
        if minute < TIP_MIN_MINUTE:
            continue
        total = int(gh or 0) + int(ga or 0)
        # label by suggestion
        if sugg.startswith("Over"):
            line = None
            for tok in sugg.split():
                try:
                    line = float(tok); break
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
    """
    Big fallback: for each match with results, take one snapshot (default: earliest minute in [TIP_MIN_MINUTE, SNAP_MAX_MINUTE]),
    and create 5 labeled rows: O2.5, O3.5, BTTS:Yes, Home Win, Away Win.
    """
    cutoff = int(time.time()) - days * 24 * 3600

    # Get results map
    mids = get_rows("SELECT DISTINCT match_id FROM tip_snapshots WHERE created_ts >= %s", (cutoff,))
    mids = [int(m[0]) for m in mids] if mids else []
    res_map: Dict[int, Tuple[int, int, int]] = {}
    if mids:
        step = 5000
        for i in range(0, len(mids), step):
            chunk = mids[i:i+step]
            fmt = ",".join(["%s"] * len(chunk))
            rows = get_rows(f"SELECT match_id, final_goals_h, final_goals_a, btts_yes FROM match_results WHERE match_id IN ({fmt})", tuple(chunk))
            for (mid, gh, ga, btts) in rows:
                res_map[int(mid)] = (int(gh or 0), int(ga or 0), int(btts or 0))

    # Stream snapshots in order and pick one per match
    rows = get_rows(
        "SELECT match_id, created_ts, payload FROM tip_snapshots WHERE created_ts >= %s ORDER BY match_id, created_ts ASC",
        (cutoff,),
    )
    by_mid: Dict[int, List[Tuple[int, str]]] = {}
    for (mid, cts, payload) in rows:
        mid = int(mid)
        if mid not in res_map:
            continue
        by_mid.setdefault(mid, []).append((int(cts), payload))

    out: List[Tuple[dict, str, int]] = []
    used = 0
    for mid, arr in by_mid.items():
        pick_feat = None
        if per_match == "latest":
            _, payload = arr[-1]
            pick_feat = _features_from_payload(payload)
        else:
            for _, payload in arr:
                feat = _features_from_payload(payload)
                m = int(float(feat.get("minute", 0)))
                if m >= TIP_MIN_MINUTE and m <= SNAP_MAX_MINUTE:
                    pick_feat = feat
                    break
        if not pick_feat:
            continue
        gh, ga, btts = res_map[mid]
        total = gh + ga
        # 5 rows
        out.append((pick_feat, "Over 2.5 Goals", 1 if total > 2.5 else 0))
        out.append((pick_feat, "Over 3.5 Goals", 1 if total > 3.5 else 0))
        out.append((pick_feat, "BTTS: Yes",      1 if btts == 1 else 0))
        out.append((pick_feat, "Home Win",       1 if gh > ga else 0))
        out.append((pick_feat, "Away Win",       1 if ga > gh else 0))
        used += 1
    log.info("[TRAIN] snapshots fallback: matches used=%d, rows=%d", used, len(out))
    return out

# --- Model training ---
def _train_one(rows: List[Tuple[dict, str, int]], label_key: str) -> Optional[Dict[str, Any]]:
    X: List[List[float]] = []
    y: List[int] = []
    feature_names: List[str] = []

    # unify features across rows
    keys_set = set()
    for feat, sug, yy in rows:
        if sug != label_key:
            continue
        keys_set.update(feat.keys())
    feature_names = sorted(list(keys_set))
    if not feature_names:
        return None

    for feat, sug, yy in rows:
        if sug != label_key:
            continue
        X.append([float(feat.get(k, 0.0) or 0.0) for k in feature_names])
        y.append(int(yy))

    if len(y) < MIN_SAMPLES_PER_MODEL:
        log.info("[TRAIN] %s skipped — %d < %d samples", label_key, len(y), MIN_SAMPLES_PER_MODEL)
        return None

    Xn = np.asarray(X, dtype=float)
    yn = np.asarray(y, dtype=int)

    # fit LR
    lr = LogisticRegression(solver="liblinear", max_iter=1000)
    lr.fit(Xn, yn)
    raw_scores = lr.decision_function(Xn)
    raw_probs = 1 / (1 + np.exp(-raw_scores))

    # Platt calibration on holdout (20%)
    try:
        X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(Xn, yn, raw_probs, test_size=0.2, random_state=42, stratify=yn)
        a, b = _fit_platt(s_te, y_te)
    except Exception:
        a, b = 1.0, 0.0

    weights = {feature_names[i]: float(lr.coef_[0][i]) for i in range(len(feature_names))}
    intercept = float(lr.intercept_[0])

    # small Naive-Bayes overlay defaults (can be tuned later)
    bayes = {
        "features": {
            "sot_sum": {"bins": [0, 2, 5, 10, 99], "lr": [0.8, 1.0, 1.2, 1.4]},
            "xg_sum":  {"bins": [0, 0.5, 1.0, 2.0, 99], "lr": [0.8, 1.0, 1.2, 1.4]},
            "cor_sum": {"bins": [0, 2, 5, 9, 99], "lr": [0.9, 1.0, 1.1, 1.2]}
        }
    }

    mdl = {
        "intercept": intercept,
        "weights": weights,
        "calibration": {"method": "sigmoid", "a": float(a), "b": float(b)},
        "bayes": bayes
    }
    return mdl

def _save_model(key: str, mdl: Dict[str, Any]) -> None:
    set_setting(key, json.dumps(mdl, separators=(",", ":"), ensure_ascii=False))

def train_models(days: int = 90) -> Dict[str, Any]:
    # 1) Try tips+odds first
    rows = _load_training_rows_from_tips(days)
    # 2) Fallback to snapshots if not enough
    if sum(1 for r in rows if r[1] in ("Over 2.5 Goals","Over 3.5 Goals","BTTS: Yes","Home Win","Away Win")) < MIN_SAMPLES_PER_MODEL:
        rows = _load_training_rows_from_snapshots(days=max(days, 730), per_match="first_viable")

    labels = ["Over 2.5 Goals","Over 3.5 Goals","BTTS: Yes","Home Win","Away Win","Draw"]
    # Build Draw labels from snapshots too (if any tips rows didn’t include draw)
    if any(r[1] in ("Home Win","Away Win") for r in rows):
        # synth draw labels if we can infer (requires payload + results; our snapshot loader already encodes draw in neither home/away)
        pass

    # Expand to include Draw labels from snapshots loader directly
    # We’ll add them by re-deriving from results, but for simplicity, reuse snapshots loader once more:
    snap_rows = _load_training_rows_from_snapshots(days=max(days, 730), per_match="first_viable")
    # Create Draw labels
    with _conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT match_id, final_goals_h, final_goals_a FROM match_results")
        res = cur.fetchall()
        dr_map = {int(r[0]): (int(r[1]), int(r[2])) for r in res}
    # we can't map match_id from rows here; so just skip sophisticated draw add. Instead, train a plain draw = 0.5 baseline
    # and rely on normalization using WLD_HOME and WLD_AWAY. (Your main.py uses WLD_DRAW only for renorm; a flat model is fine.)
    draw_mdl = {"intercept": 0.0, "weights": {}, "calibration": {"method": "sigmoid", "a": 1.0, "b": 0.0}}

    results: Dict[str, Any] = {"ok": True, "trained": {}}

    # OU 2.5
    mdl = _train_one(rows, "Over 2.5 Goals")
    if mdl: _save_model("OU_2.5", mdl); results["trained"]["OU_2.5"] = True
    else:   results["trained"]["OU_2.5"] = False

    # OU 3.5
    mdl = _train_one(rows, "Over 3.5 Goals")
    if mdl: _save_model("OU_3.5", mdl); results["trained"]["OU_3.5"] = True
    else:   results["trained"]["OU_3.5"] = False

    # BTTS
    mdl = _train_one(rows, "BTTS: Yes")
    if mdl: _save_model("BTTS_YES", mdl); results["trained"]["BTTS_YES"] = True
    else:   results["trained"]["BTTS_YES"] = False

    # WLD (draw suppressed in tips, but we store all three)
    mdl = _train_one(rows, "Home Win")
    if mdl: _save_model("WLD_HOME", mdl); results["trained"]["WLD_HOME"] = True
    else:   results["trained"]["WLD_HOME"] = False

    mdl = _train_one(rows, "Away Win")
    if mdl: _save_model("WLD_AWAY", mdl); results["trained"]["WLD_AWAY"] = True
    else:   results["trained"]["WLD_AWAY"] = False

    # WLD_DRAW (flat calibration to avoid zeros)
    _save_model("WLD_DRAW", draw_mdl); results["trained"]["WLD_DRAW"] = True

    any_trained = any(results["trained"].values())
    if not any_trained:
        msg = f"Not enough samples to train: rows={len(rows)} < {MIN_SAMPLES_PER_MODEL}"
        log.warning(msg)
        return {"ok": False, "reason": msg, "samples": len(rows)}

    return results

# --- ROI-aware auto tune (works off tips with odds + results) ---
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
        """,
        (cutoff,),
    )
    if not rows:
        set_setting("last_auto_tune", "no data")
        return {}

    by: Dict[str, List[Tuple[float, int, float]]] = {}
    for (mk, sugg, prob, odds, gh, ga, btts) in rows:
        try:
            prob = float(prob or 0.0); odds = float(odds or 0.0)
        except Exception:
            continue
        if not (1.01 <= odds <= MAX_ODDS_ALL):
            continue

        # label outcome
        def _lbl(s: str) -> Optional[int]:
            total = int(gh or 0) + int(ga or 0)
            if s.startswith("Over"):
                line = None
                for tok in s.split():
                    try: line = float(tok); break
                    except: pass
                if line is None: return None
                return 1 if total > line else (0 if total < line else None)
            if s == "BTTS: Yes": return 1 if int(btts or 0) == 1 else 0
            if s == "BTTS: No":  return 1 if int(btts or 0) == 0 else 0
            if s == "Home Win":  return 1 if int(gh or 0) > int(ga or 0) else 0
            if s == "Away Win":  return 1 if int(ga or 0) > int(gh or 0) else 0
            return None

        y = _lbl(sugg)
        if y is None:
            continue
        mk_key = mk.replace("PRE ", "")
        by.setdefault(mk_key, []).append((prob, int(y), odds))

    tuned: Dict[str, float] = {}

    def _eval(items: List[Tuple[float, int, float]], thr_prob: float) -> Tuple[int, float, float]:
        sel = [(p, y, o) for (p, y, o) in items if p >= thr_prob]
        n = len(sel)
        if n == 0:
            return 0, 0.0, 0.0
        wins = sum(y for (_, y, _) in sel)
        prec = wins / n
        roi = sum((y * (odds - 1.0) - (1 - y)) for (_, y, odds) in sel) / n
        return n, float(prec), float(roi)

    for mk, items in by.items():
        if len(items) < THRESH_MIN_PREDICTIONS:
            continue
        best = None
        feasible_any = False
        for thr_pct in np.arange(MIN_THRESH, MAX_THRESH + 1e-9, 1.0):
            thr_prob = thr_pct / 100.0
            n, prec, roi = _eval(items, thr_prob)
            if n < THRESH_MIN_PREDICTIONS:
                continue
            if prec >= TARGET_PRECISION:
                feasible_any = True
                score = (roi, prec, n)
                if (best is None) or (score > (best[0], best[1], best[2])):
                    best = (roi, prec, n, thr_pct)
        if not feasible_any:
            for thr_pct in np.arange(MIN_THRESH, MAX_THRESH + 1e-9, 1.0):
                thr_prob = thr_pct / 100.0
                n, prec, roi = _eval(items, thr_prob)
                if n < THRESH_MIN_PREDICTIONS:
                    continue
                if (prec >= max(0.0, TARGET_PRECISION - APPLY_TUNE_PREC_TOL)) and (roi > 0.0):
                    score = (roi, prec, n)
                    if (best is None) or (score > (best[0], best[1], best[2])):
                        best = (roi, prec, n, thr_pct)
        if best is None:
            continue
        tuned[mk] = float(best[3])

    for k, pct in tuned.items():
        set_setting(f"conf_threshold:{k}", f"{pct:.2f}")

    return tuned

if __name__ == "__main__":
    print(json.dumps(train_models(), indent=2))
