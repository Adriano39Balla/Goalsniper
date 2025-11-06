# file: train_models.py
import os, json, logging, sys, time, atexit, signal, warnings
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

import numpy as np
import psycopg2
from psycopg2.pool import SimpleConnectionPool
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning

# ───────────────────── Logging ─────────────────────
log = logging.getLogger("train_models")
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s [train_models] - %(message)s"))
log.handlers = [handler]
log.setLevel(logging.INFO)
log.propagate = False

# ───────────────────── DB Pool ─────────────────────
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise SystemExit("DATABASE_URL is required")

POOL: Optional[SimpleConnectionPool] = None

class ShutdownManager:
    _shutdown_requested = False
    @classmethod
    def is_shutdown_requested(cls) -> bool: return cls._shutdown_requested
    @classmethod
    def request_shutdown(cls) -> None: cls._shutdown_requested = True

def _init_pool():
    global POOL
    if POOL: return
    maxconn = int(os.getenv("DB_POOL_MAX", "3"))
    POOL = SimpleConnectionPool(minconn=1, maxconn=maxconn, dsn=DATABASE_URL)
    log.info("[TRAIN_DB] Connected (pool=%d)", maxconn)

class PooledConn:
    def __init__(self, pool): self.pool=pool; self.conn=None; self.cur=None
    def __enter__(self):
        if ShutdownManager.is_shutdown_requested():
            raise RuntimeError("Shutdown in progress")
        _init_pool()
        self.conn=self.pool.getconn(); self.conn.autocommit=True; self.cur=self.conn.cursor()
        return self
    def __exit__(self, et, ev, tb):
        try:
            if self.cur: self.cur.close()
        finally:
            if self.conn:
                try: self.pool.putconn(self.conn)
                except: 
                    try: self.conn.close()
                    except: pass
    def execute(self, sql: str, params: tuple|list=()):
        self.cur.execute(sql, params or ()); return self.cur
    def fetchone_safe(self): 
        try: row=self.cur.fetchone(); return (None if row is None or len(row)==0 else row)
        except Exception as e: log.warning("[TRAIN_DB] fetchone_safe: %s", e); return None
    def fetchall_safe(self):
        try: rows=self.cur.fetchall(); return rows or []
        except Exception as e: log.warning("[TRAIN_DB] fetchall_safe: %s", e); return []

def db_conn(): 
    if not POOL: _init_pool()
    return PooledConn(POOL)

def _db_ping() -> bool:
    try:
        with db_conn() as c:
            c.execute("SELECT 1")
        return True
    except Exception as e:
        log.error("[TRAIN_DB] ping failed: %s", e)
        return False

# ───────────────── Feature helpers (match main.py) ─────────────────
def _num(v) -> float:
    try:
        if isinstance(v,str) and v.endswith("%"): return float(v[:-1])
        return float(v or 0)
    except: return 0.0

def _pos_pct(v) -> float:
    try: return float(str(v).replace("%","").strip() or 0)
    except: return 0.0

def extract_basic_features(m: dict) -> Dict[str,float]:
    """Keep aligned with main.py extract_basic_features."""
    try:
        teams = m.get("teams") or {}
        home = (teams.get("home") or {}).get("name","")
        away = (teams.get("away") or {}).get("name","")
        goals = m.get("goals") or {}
        gh = goals.get("home") or 0
        ga = goals.get("away") or 0
        minute = int(((m.get("fixture") or {}).get("status") or {}).get("elapsed") or 0)

        stats={}
        for s in (m.get("statistics") or []):
            t=(s.get("team") or {}).get("name")
            if t: stats[t]={ (i.get("type") or ""): i.get("value") for i in (s.get("statistics") or []) }
        sh=stats.get(home, {}) or {}; sa=stats.get(away, {}) or {}

        sot_h=_num(sh.get("Shots on Goal", sh.get("Shots on Target", 0)))
        sot_a=_num(sa.get("Shots on Goal", sa.get("Shots on Target", 0)))
        sh_total_h=_num(sh.get("Total Shots", 0)); sh_total_a=_num(sa.get("Total Shots", 0))
        cor_h=_num(sh.get("Corner Kicks", 0)); cor_a=_num(sa.get("Corner Kicks", 0))
        pos_h=_pos_pct(sh.get("Ball Possession", 0)); pos_a=_pos_pct(sa.get("Ball Possession", 0))

        xg_h=_num(sh.get("Expected Goals", 0)); xg_a=_num(sa.get("Expected Goals", 0))
        if xg_h==0 and xg_a==0:
            xg_h = sot_h*0.3; xg_a = sot_a*0.3  # only when missing

        red_h=red_a=yellow_h=yellow_a=0
        for ev in (m.get("events") or []):
            if (ev.get("type","").lower()=="card"):
                d=(ev.get("detail","") or "").lower(); t=(ev.get("team") or {}).get("name") or ""
                if "yellow" in d and "second" not in d:
                    if t==home: yellow_h+=1
                    elif t==away: yellow_a+=1
                if "red" in d or "second yellow" in d:
                    if t==home: red_h+=1
                    elif t==away: red_a+=1

        return {
            "minute": float(minute),
            "goals_h": float(gh), "goals_a": float(ga),
            "goals_sum": float(gh+ga), "goals_diff": float(gh-ga),
            "xg_h": float(xg_h), "xg_a": float(xg_a),
            "xg_sum": float(xg_h+xg_a), "xg_diff": float(xg_h-xg_a),
            "sot_h": float(sot_h), "sot_a": float(sot_a), "sot_sum": float(sot_h+sot_a),
            "sh_total_h": float(sh_total_h), "sh_total_a": float(sh_total_a),
            "cor_h": float(cor_h), "cor_a": float(cor_a), "cor_sum": float(cor_h+cor_a),
            "pos_h": float(pos_h), "pos_a": float(pos_a), "pos_diff": float(pos_h-pos_a),
            "red_h": float(red_h), "red_a": float(red_a), "red_sum": float(red_h+red_a),
            "yellow_h": float(yellow_h), "yellow_a": float(yellow_a)
        }
    except Exception as e:
        log.error("Feature extraction failed: %s", e)
        return {}

# ───────────────── Settings I/O ─────────────────
def get_setting(key: str) -> Optional[str]:
    try:
        with db_conn() as c:
            cur = c.execute("SELECT value FROM settings WHERE key=%s", (key,))
            row = None
            try: row = cur.fetchone()
            except: pass
            if not row: return None
            return row[0]
    except Exception as e:
        log.error("get_setting(%s) failed: %s", key, e); return None

def set_setting(key: str, value: str) -> None:
    with db_conn() as c:
        c.execute(
            "INSERT INTO settings(key,value) VALUES(%s,%s) "
            "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
            (key, value)
        )

# ───────────────── Load training snapshots ─────────────────
def _has_sufficient_data(features: Dict[str, float], minute: int) -> bool:
    try:
        has_any = any(features.get(k,0)>0 for k in ("sot_h","sot_a","pos_h","pos_a","xg_sum","cor_sum"))
        if minute > 60:
            return has_any and (features.get("sot_sum",0)>=2 or features.get("xg_sum",0)>0)
        return has_any
    except: return False

def load_training_data(days: int = 60, min_minute: int = 20) -> List[Dict[str, Any]]:
    if not _db_ping():
        log.error("DB unavailable"); return []
    cutoff_ts = int((datetime.utcnow() - timedelta(days=days)).timestamp())
    out: List[Dict[str,Any]] = []
    with db_conn() as c:
        c.execute("""
            SELECT payload FROM tip_snapshots
            WHERE created_ts >= %s
            ORDER BY created_ts DESC
            LIMIT 20000
        """, (cutoff_ts,))
        for (payload,) in c.fetchall_safe():
            try:
                data = json.loads(payload)
                m = data.get("match",{}) or {}
                feat = data.get("features",{}) or {}
                minute = int(feat.get("minute",0))
                if minute < min_minute: continue
                # Use DB backfilled result if available
                fid = int(((m.get("fixture") or {}).get("id") or 0))
                if fid:
                    # optional: attach final score from match_results if present
                    try:
                        with db_conn() as c2:
                            r = c2.execute("SELECT final_goals_h,final_goals_a,btts_yes FROM match_results WHERE match_id=%s",(fid,)).fetchone()
                            if r:
                                g = {"home": int(r[0] or 0), "away": int(r[1] or 0)}
                                m["goals"] = g
                                (m.setdefault("fixture",{}).setdefault("status",{}))["short"]="FT"
                    except Exception as e:
                        log.debug("match_results lookup fail: %s", e)
                if not _has_sufficient_data(feat, minute): continue
                out.append({"match": m, "features": feat, "timestamp": data.get("timestamp",0)})
            except Exception as e:
                log.debug("snapshot parse skip: %s", e)
    log.info("Loaded %d training samples (>= %d')", len(out), min_minute)
    return out

# ───────────────── Labeling ─────────────────
def calculate_outcome(match_data: dict, market: str, suggestion: str) -> Optional[int]:
    try:
        goals = match_data.get("goals") or {}
        gh = int(goals.get("home") or 0); ga = int(goals.get("away") or 0)
        total = gh + ga
        # Accept if we have a final score (via FT mark above)
        short = (((match_data.get("fixture") or {}).get("status") or {}).get("short") or "").upper()
        if short not in {"FT","AET","PEN"}: return None
        if market == "BTTS":
            return 1 if ((suggestion=="BTTS: Yes" and gh>0 and ga>0) or (suggestion=="BTTS: No" and (gh==0 or ga==0))) else 0
        if market.startswith("OU_"):
            try:
                line = float(market.split("_",1)[1])
                if suggestion.startswith("Over"):
                    return 1 if total > line else (0 if total < line else None)
                else:
                    return 1 if total < line else (0 if total > line else None)
            except: return None
        if market == "1X2_HOME": return 1 if gh>ga else 0
        if market == "1X2_AWAY": return 1 if ga>gh else 0
        return None
    except: return None

# ───────────────── Feature matrix ─────────────────
def prepare_xy(training: List[Dict], market: str, suggestion: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    feats: List[Dict[str,float]] = []; labels: List[int] = []
    for row in training:
        y = calculate_outcome(row["match"], market, suggestion)
        if y is None: continue
        feats.append(row["features"]); labels.append(int(y))
    if not feats: return np.zeros((0,0)), np.zeros((0,)), []
    # gather names
    names = sorted({k for d in feats for k in d.keys()})
    X = np.array([[float(d.get(n,0.0)) for n in names] for d in feats], dtype=float)
    y = np.array(labels, dtype=int)
    # drop zero-variance columns
    var = X.var(axis=0)
    keep = var > 1e-12
    if not np.any(keep): return np.zeros((0,0)), np.zeros((0,)), []
    X = X[:, keep]
    names = [n for n, k in zip(names, keep) if k]
    return X, y, names

# ───────────────── Fit (with scaling + fallback) ─────────────────
def _fit_logreg_scaled(X: np.ndarray, y: np.ndarray, max_iter: int = 5000, solver: str = "lbfgs") -> Tuple[LogisticRegression, np.ndarray, np.ndarray]:
    # Standardize per-feature on train only
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma==0] = 1.0
    Xs = (X - mu) / sigma
    model = LogisticRegression(
        random_state=42, max_iter=max_iter, solver=solver,
        class_weight="balanced", n_jobs=None if solver!="saga" else -1
    )
    model.fit(Xs, y)
    # If solver hit cap, retry with saga
    hit_cap = False
    try:
        iters = int(np.max(model.n_iter_)) if hasattr(model, "n_iter_") else 0
        hit_cap = iters >= max_iter
    except: pass
    if hit_cap and solver != "saga":
        log.warning("lbfgs hit max_iter; retrying with saga")
        return _fit_logreg_scaled(X, y, max_iter=max_iter, solver="saga")
    return model, mu, sigma

def _bake_scaler_into_weights(model: LogisticRegression, mu: np.ndarray, sigma: np.ndarray) -> Tuple[np.ndarray, float]:
    # Convert z = (x-mu)/sigma ↦ logit = i + w·z  =>  i' = i - Σ(w*mu/sigma), w' = w/sigma
    w = model.coef_.reshape(-1)
    i = float(model.intercept_.reshape(-1)[0])
    w_prime = w / sigma
    i_prime = i - float(np.sum(w * (mu / sigma)))
    return w_prime, i_prime

# ───────────────── Train per target ─────────────────
def _pretty_name_and_suggestion(key: str) -> Tuple[str, str]:
    if key == "BTTS": return ("BTTS", "BTTS: Yes")
    if key == "OU_2.5": return ("OU_2.5", "Over 2.5 Goals")
    if key == "OU_3.5": return ("OU_3.5", "Over 3.5 Goals")
    if key == "1X2_HOME": return ("1X2_HOME", "Home Win")
    if key == "1X2_AWAY": return ("1X2_AWAY", "Away Win")
    return (key, "")

def train_one(training: List[Dict], key: str) -> Optional[Dict[str,Any]]:
    market, suggestion = _pretty_name_and_suggestion(key)
    X, y, names = prepare_xy(training, market, suggestion)
    if X.shape[0] < 150 or X.shape[1] < 3:
        log.warning("Insufficient data for %s (n=%d, d=%d)", key, X.shape[0], X.shape[1])
        return None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model, mu, sigma = _fit_logreg_scaled(X_train, y_train)
    # metrics (on standardized space)
    Xtr_s = (X_train - mu)/sigma; Xte_s = (X_test - mu)/sigma
    train_score = float(model.score(Xtr_s, y_train))
    test_score  = float(model.score(Xte_s, y_test))
    # bake scaler into weights for main.py
    w_prime, i_prime = _bake_scaler_into_weights(model, mu, sigma)
    weights = {n: float(v) for n, v in zip([*names], w_prime)}
    model_blob = {
        "weights": weights,
        "intercept": float(i_prime),
        "train_score": train_score,
        "test_score": test_score,
        "samples": int(X_train.shape[0]),
        "positive_samples": int(int(y_train.sum())),
        "created_ts": int(time.time()),
        # Keep identity calibration; main.py supports {"method":"sigmoid","a","b"} if you add Platt later.
        "calibration": {"method": "sigmoid", "a": 1.0, "b": 0.0}
    }
    log.info("✅ %s trained (n=%d, d=%d, test=%.3f)", key, X_train.shape[0], X_train.shape[1], test_score)
    return model_blob

def _save_model_under_aliases(key: str, blob: Dict[str,Any]) -> None:
    # Save as both raw name and model_v2:{name}
    set_setting(key, json.dumps(blob, separators=(",",":")))
    set_setting(f"model_v2:{key}", json.dumps(blob, separators=(",",":")))
    log.info("✅ %s saved as model_v2:%s (n=%s, d=%s, test=%.3f)",
             key, key, blob.get("samples","?"),
             len((blob.get("weights") or {})), blob.get("test_score",0.0))

# ───────────────── Auto-tune thresholds (light) ─────────────────
def auto_tune_thresholds(model_details: Dict[str, Dict[str,Any]]) -> Dict[str, float]:
    default_thr = float(os.getenv("CONF_THRESHOLD", "75"))
    tuned: Dict[str,float] = {}
    for key, det in model_details.items():
        t = float(det.get("test_score", 0.5)); n = int(det.get("samples", 0))
        if t > 0.70 and n > 300: tuned[key] = min(85.0, default_thr + 5.0)
        elif t < 0.55:          tuned[key] = max(65.0, default_thr - 10.0)
        else:                    tuned[key] = default_thr
        set_setting(f"conf_threshold:{_threshold_market_name(key)}", str(tuned[key]))
    return tuned

def _threshold_market_name(key: str) -> str:
    if key.startswith("OU_"):
        return f"Over/Under {key.split('_',1)[1]}"
    if key.startswith("1X2"): return "1X2"
    return key

# ───────────────── Orchestrator ─────────────────
def train_models(days: int = 60) -> Dict[str, Any]:
    if not _db_ping(): return {"ok": False, "error": "Database unavailable"}
    training = load_training_data(days=days, min_minute=int(os.getenv("TRAIN_MIN_MINUTE","15")))
    if not training:  return {"ok": False, "error": "No training data"}
    to_train = ["BTTS","OU_2.5","OU_3.5","1X2_HOME","1X2_AWAY"]

    trained: Dict[str,bool] = {}
    details: Dict[str,Dict[str,Any]] = {}
    for key in to_train:
        if ShutdownManager.is_shutdown_requested(): break
        try:
            blob = train_one(training, key)
            if blob:
                _save_model_under_aliases(key, blob)
                trained[key] = True
                details[key] = {
                    "samples": blob.get("samples",0),
                    "test_score": blob.get("test_score",0.0)
                }
            else:
                trained[key] = False
        except Exception as e:
            log.exception("Training error for %s: %s", key, e)
            trained[key] = False

    tuned = auto_tune_thresholds(details) if trained else {}
    ok = any(trained.values())
    return {"ok": ok, "trained": trained, "model_details": details, "tuned_thresholds": tuned}

# ───────────────── Shutdown hooks ─────────────────
def cleanup():
    global POOL
    if POOL:
        try: POOL.closeall(); log.info("[TRAIN_DB] pool closed")
        except Exception as e: log.warning("[TRAIN_DB] pool close err: %s", e)

def _shutdown(signum=None, frame=None):
    log.info("Shutdown signal received"); ShutdownManager.request_shutdown(); cleanup(); sys.exit(0)

def register_shutdown_handlers():
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    atexit.register(cleanup)

# ───────────────── Entrypoint ─────────────────
if __name__ == "__main__":
    # Convert ConvergenceWarning to log only (not exception)
    warnings.simplefilter("always", category=ConvergenceWarning)
    register_shutdown_handlers()
    try:
        log.info("Starting training...")
        res = train_models()
        print(json.dumps(res, indent=2))
        if not res.get("ok"): sys.exit(1)
        log.info("Training completed.")
    except KeyboardInterrupt:
        log.info("Interrupted"); sys.exit(1)
    except Exception as e:
        log.exception("Fatal: %s", e); sys.exit(1)
    finally:
        cleanup()
