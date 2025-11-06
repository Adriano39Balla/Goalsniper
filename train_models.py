# file: train_models.py
import os, json, logging, sys, time, re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import psycopg2
from psycopg2.pool import SimpleConnectionPool
import requests

# ──────────────────────────────────────────────────────────────────────────────
# Config / env
# ──────────────────────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise SystemExit("DATABASE_URL is required")

API_KEY = os.getenv("API_KEY")  # optional, for result backfill
BASE_URL = "https://v3.football.api-sports.io"
FOOTBALL_API_URL = f"{BASE_URL}/fixtures"
HEADERS = {"x-apisports-key": API_KEY, "Accept": "application/json"} if API_KEY else {}
REQ_TIMEOUT = float(os.getenv("REQ_TIMEOUT_SEC", "8.0"))

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
log = logging.getLogger("train_models")
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s [train_models] - %(message)s"))
log.handlers = [_handler]
log.setLevel(logging.INFO)
log.propagate = False

# ──────────────────────────────────────────────────────────────────────────────
# DB pool (simple version compatible with main)
# ──────────────────────────────────────────────────────────────────────────────
POOL: Optional[SimpleConnectionPool] = None

class ShutdownManager:
    _shutdown_requested = False
    @classmethod
    def is_shutdown_requested(cls): return cls._shutdown_requested
    @classmethod
    def request_shutdown(cls): cls._shutdown_requested = True

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
            raise Exception("DB refused: shutdown")
        _init_pool()
        self.conn=self.pool.getconn(); self.conn.autocommit=True
        self.cur=self.conn.cursor(); return self
    def __exit__(self, et, ev, tb):
        try:
            if self.cur: self.cur.close()
        finally:
            if self.conn:
                try: self.pool.putconn(self.conn)
                except:
                    try: self.conn.close()
                    except: pass
    def execute(self, sql, params=()):
        if ShutdownManager.is_shutdown_requested():
            raise Exception("DB op refused: shutdown")
        self.cur.execute(sql, params or ()); return self.cur
    def fetchone_safe(self):
        try:
            row=self.cur.fetchone()
            return None if row is None or len(row)==0 else row
        except Exception as e:
            log.warning("[TRAIN_DB] fetchone_safe: %s", e); return None
    def fetchall_safe(self):
        try:
            rows=self.cur.fetchall()
            return rows if rows else []
        except Exception as e:
            log.warning("[TRAIN_DB] fetchall_safe: %s", e); return []

def db_conn(): 
    if not POOL: _init_pool()
    return PooledConn(POOL)

def _db_ping()->bool:
    try:
        with db_conn() as c:
            c.execute("SELECT 1")
        return True
    except Exception as e:
        log.error("[TRAIN_DB] ping failed: %s", e); return False

# ──────────────────────────────────────────────────────────────────────────────
# Settings helpers (match main.py)
# ──────────────────────────────────────────────────────────────────────────────
def get_setting(key: str) -> Optional[str]:
    try:
        with db_conn() as c:
            c.execute("SELECT value FROM settings WHERE key=%s", (key,))
            row = c.fetchone_safe()  # FIX: call on wrapper, not cursor
            return row[0] if row else None
    except Exception as e:
        log.error("get_setting(%s) failed: %s", key, e); return None

def set_setting(key: str, value: str) -> None:
    with db_conn() as c:
        c.execute(
            "INSERT INTO settings(key,value) VALUES(%s,%s) "
            "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
            (key, value)
        )

# ──────────────────────────────────────────────────────────────────────────────
# Feature extraction (must mirror main.py basic features)
# ──────────────────────────────────────────────────────────────────────────────
def _num(v)->float:
    try:
        if isinstance(v,str) and v.endswith("%"): return float(v[:-1])
        return float(v or 0)
    except: return 0.0

def _pos_pct(v)->float:
    try: return float(str(v).replace("%","").strip() or 0)
    except: return 0.0

def extract_basic_features(m: dict) -> Dict[str,float]:
    try:
        home = (m.get("teams") or {}).get("home", {}).get("name", "")
        away = (m.get("teams") or {}).get("away", {}).get("name", "")
        gh = (m.get("goals") or {}).get("home") or 0
        ga = (m.get("goals") or {}).get("away") or 0
        minute = int(((m.get("fixture") or {}).get("status") or {}).get("elapsed") or 0)
        stats = {}
        for s in (m.get("statistics") or []):
            t = (s.get("team") or {}).get("name")
            if t: stats[t] = { (i.get("type") or ""): i.get("value") for i in (s.get("statistics") or []) }
        sh = stats.get(home, {}) or {}; sa = stats.get(away, {}) or {}
        sot_h = _num(sh.get("Shots on Goal", sh.get("Shots on Target", 0)))
        sot_a = _num(sa.get("Shots on Goal", sa.get("Shots on Target", 0)))
        sh_total_h = _num(sh.get("Total Shots", 0)); sh_total_a = _num(sa.get("Total Shots", 0))
        cor_h = _num(sh.get("Corner Kicks", 0)); cor_a = _num(sa.get("Corner Kicks", 0))
        pos_h = _pos_pct(sh.get("Ball Possession", 0)); pos_a = _pos_pct(sa.get("Ball Possession", 0))
        xg_h = _num(sh.get("Expected Goals", 0)); xg_a = _num(sa.get("Expected Goals", 0))
        if xg_h == 0 and xg_a == 0:
            xg_h = sot_h * 0.3; xg_a = sot_a * 0.3
        red_h = red_a = yellow_h = yellow_a = 0
        for ev in (m.get("events") or []):
            if (ev.get("type", "").lower() == "card"):
                d = (ev.get("detail", "") or "").lower()
                t = (ev.get("team") or {}).get("name") or ""
                if "yellow" in d and "second" not in d:
                    if t == home: yellow_h += 1
                    elif t == away: yellow_a += 1
                if "red" in d or "second yellow" in d:
                    if t == home: red_h += 1
                    elif t == away: red_a += 1
        return {
            "minute": float(minute),
            "goals_h": float(gh), "goals_a": float(ga),
            "goals_sum": float(gh + ga), "goals_diff": float(gh - ga),
            "xg_h": float(xg_h), "xg_a": float(xg_a),
            "xg_sum": float(xg_h + xg_a), "xg_diff": float(xg_h - xg_a),
            "sot_h": float(sot_h), "sot_a": float(sot_a), "sot_sum": float(sot_h + sot_a),
            "sh_total_h": float(sh_total_h), "sh_total_a": float(sh_total_a),
            "cor_h": float(cor_h), "cor_a": float(cor_a), "cor_sum": float(cor_h + cor_a),
            "pos_h": float(pos_h), "pos_a": float(pos_a), "pos_diff": float(pos_h - pos_a),
            "red_h": float(red_h), "red_a": float(red_a), "red_sum": float(red_h + red_a),
            "yellow_h": float(yellow_h), "yellow_a": float(yellow_a)
        }
    except Exception as e:
        log.error("Feature extraction failed: %s", e)
        return {}

# ──────────────────────────────────────────────────────────────────────────────
# Data loading & backfill
# ──────────────────────────────────────────────────────────────────────────────
def _has_sufficient_data(features: Dict[str, float], minute: int) -> bool:
    try:
        base = (features.get('sot_h',0)>0 or features.get('sot_a',0)>0 or
                features.get('pos_h',0)>0 or features.get('pos_a',0)>0 or
                features.get('cor_h',0)>0 or features.get('cor_a',0)>0)
        if minute > 60:
            base = base and (features.get('xg_sum',0)>0 or features.get('sot_sum',0)>=3)
        return bool(base)
    except Exception:
        return False

def _api_get(url: str, params: dict) -> Optional[dict]:
    if not API_KEY: return None
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=REQ_TIMEOUT)
        return r.json() if r.ok else None
    except Exception:
        return None

def backfill_results_for_snapshot_ids(match_ids: List[int]) -> int:
    if not API_KEY or not match_ids: return 0
    written = 0
    with db_conn() as c:
        for mid in match_ids[:400]:
            js = _api_get(FOOTBALL_API_URL, {"id": mid}) or {}
            arr = (js.get("response") or []) if isinstance(js, dict) else []
            if not arr: continue
            fx = arr[0]
            st = (((fx.get("fixture") or {}).get("status") or {}).get("short") or "").upper()
            if st not in {"FT","AET","PEN"}: continue
            goals = fx.get("goals") or {}
            gh = int(goals.get("home") or 0); ga = int(goals.get("away") or 0)
            btts = 1 if (gh>0 and ga>0) else 0
            c.execute(
                "INSERT INTO match_results(match_id, final_goals_h, final_goals_a, btts_yes, updated_ts) "
                "VALUES(%s,%s,%s,%s,%s) "
                "ON CONFLICT(match_id) DO UPDATE SET "
                "final_goals_h=EXCLUDED.final_goals_h, final_goals_a=EXCLUDED.final_goals_a, "
                "btts_yes=EXCLUDED.btts_yes, updated_ts=EXCLUDED.updated_ts",
                (mid, gh, ga, btts, int(time.time()))
            )
            written += 1
    if written:
        log.info("[BACKFILL] wrote %d match_results", written)
    return written

def load_training_rows(days: int = 60, min_minute: int = 20) -> List[Dict[str, Any]]:
    if not _db_ping():
        log.error("DB unavailable"); return []

    cutoff_ts = int((datetime.now() - timedelta(days=days)).timestamp())
    with db_conn() as c:
        c.execute(
            """
            SELECT s.match_id, s.created_ts, s.payload,
                   r.final_goals_h, r.final_goals_a, r.btts_yes
            FROM tip_snapshots s
            LEFT JOIN match_results r ON r.match_id = s.match_id
            WHERE s.created_ts >= %s
            ORDER BY s.created_ts DESC
            LIMIT 20000
            """,
            (cutoff_ts,)
        )
        rows = c.fetchall_safe()  # FIX

    missing = sorted({mid for (mid, *_rest) in rows if _rest[2] is None})
    if missing and API_KEY:
        backfill_results_for_snapshot_ids(missing)

    with db_conn() as c2:
        c2.execute(
            """
            SELECT s.match_id, s.created_ts, s.payload,
                   r.final_goals_h, r.final_goals_a, r.btts_yes
            FROM tip_snapshots s
            JOIN match_results r ON r.match_id = s.match_id
            WHERE s.created_ts >= %s
            ORDER BY s.created_ts DESC
            LIMIT 20000
            """,
            (cutoff_ts,)
        )
        rows = c2.fetchall_safe()  # FIX

    out: List[Dict[str, Any]] = []
    kept, skipped_minute, skipped_quality, parse_errors = 0,0,0,0

    for (mid, cts, payload, gh, ga, btts) in rows:
        try:
            data = json.loads(payload)
            features = data.get("features") or {}
            minute = int(features.get("minute", 0))
            if minute < min_minute:
                skipped_minute += 1; continue
            if not _has_sufficient_data(features, minute):
                skipped_quality += 1; continue
            out.append({
                "match_id": int(mid),
                "created_ts": int(cts),
                "features": features,
                "final_goals_h": int(gh),
                "final_goals_a": int(ga),
                "btts_yes": int(btts)
            })
            kept += 1
        except Exception:
            parse_errors += 1
            continue

    log.info("Snapshots graded: %d (skipped minute=%d, quality=%d, parse=%d)", kept, skipped_minute, skipped_quality, parse_errors)
    return out

# ──────────────────────────────────────────────────────────────────────────────
# Labeling per target
# ──────────────────────────────────────────────────────────────────────────────
def make_label(row: Dict[str, Any], target: str) -> Optional[int]:
    gh = int(row["final_goals_h"]); ga = int(row["final_goals_a"])
    total = gh + ga
    if target == "BTTS":
        return 1 if (gh>0 and ga>0) else 0
    if target.startswith("OU_"):
        try:
            line = float(target.split("_",1)[1])
        except Exception:
            return None
        return 1 if (total > line) else 0
    if target == "1X2_HOME":
        return 1 if gh > ga else 0
    if target == "1X2_AWAY":
        return 1 if ga > gh else 0
    return None

# ──────────────────────────────────────────────────────────────────────────────
# Model training
# ──────────────────────────────────────────────────────────────────────────────
def build_dataset(rows: List[Dict[str,Any]], target: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    X_list: List[Dict[str,float]] = []
    y_list: List[int] = []
    for r in rows:
        y = make_label(r, target)
        if y is None: continue
        X_list.append(r["features"])
        y_list.append(int(y))
    if not X_list: return np.zeros((0,0)), np.zeros((0,)), []

    names = sorted({k for d in X_list for k in d.keys()})
    vals = np.array([[float(d.get(n,0.0)) for n in names] for d in X_list], dtype=float)
    vari = np.var(vals, axis=0)
    mask = vari > 1e-8
    names_filtered = [n for n, m in zip(names, mask) if m]
    if not names_filtered:
        return np.zeros((0,0)), np.zeros((0,)), []
    X = vals[:, mask]
    y = np.array(y_list, dtype=int)
    return X, y, names_filtered

def train_logreg_balanced(X: np.ndarray, y: np.ndarray) -> Tuple[LogisticRegression, Dict[str,float]]:
    model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    model.fit(X, y)
    y_pred = model.predict(X)
    acc = float((y_pred == y).mean())
    pos = int(y.sum()); n = int(len(y))
    return model, {"acc_train": acc, "n": n, "pos": pos}

def to_model_blob(model: LogisticRegression, feature_names: List[str], metrics: Dict[str, float]) -> Dict[str, Any]:
    weights = {name: float(w) for name, w in zip(feature_names, model.coef_[0])}
    intercept = float(model.intercept_[0])
    return {
        "weights": weights,
        "intercept": intercept,
        "feature_names": feature_names,
        "train_acc": metrics.get("acc_train", 0.0),
        "samples": int(metrics.get("n", 0)),
        "positive_samples": int(metrics.get("pos", 0)),
        "calibration": {"method": "sigmoid", "a": 1.0, "b": 0.0},
        "created_ts": int(time.time())
    }

# ──────────────────────────────────────────────────────────────────────────────
# Threshold auto-tune (lightweight by test score buckets)
# ──────────────────────────────────────────────────────────────────────────────
def simple_tune_thresholds(test_scores: Dict[str, float], default_thr: float) -> Dict[str, float]:
    tuned: Dict[str, float] = {}
    for name, score in test_scores.items():
        if score >= 0.68: tuned[name] = min(85.0, default_thr + 5.0)
        elif score <= 0.55: tuned[name] = max(65.0, default_thr - 10.0)
        else: tuned[name] = default_thr
    return tuned

# ──────────────────────────────────────────────────────────────────────────────
# Main training entry
# ──────────────────────────────────────────────────────────────────────────────
def train_models(days: int = 60, min_minute: int = 20) -> Dict[str, Any]:
    log.info("Starting model training (days=%d, min_minute=%d)", days, min_minute)
    if not _db_ping():
        return {"ok": False, "error": "Database unavailable"}

    rows = load_training_rows(days=days, min_minute=min_minute)
    if not rows:
        return {"ok": False, "error": "No graded snapshots available (ensure match_results is populated)"}

    targets = ["BTTS", "OU_2.5", "OU_3.5", "1X2_HOME", "1X2_AWAY"]
    trained: Dict[str, bool] = {}
    details: Dict[str, Any] = {}
    test_scores: Dict[str, float] = {}

    for name in targets:
        if ShutdownManager.is_shutdown_requested():
            log.warning("Training interrupted"); break

        X, y, feat_names = build_dataset(rows, name)
        n = len(y)
        if n < 150 or X.shape[1] < 5:
            log.warning("Insufficient data for %s: n=%d, d=%d", name, n, X.shape[1] if n else 0)
            trained[name] = False
            continue

        try:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        except Exception as e:
            log.warning("Split failed for %s: %s", name, e); trained[name]=False; continue

        model, mtr = train_logreg_balanced(X_tr, y_tr)
        y_hat = model.predict(X_te)
        test_acc = float((y_hat == y_te).mean())
        test_scores[name] = test_acc

        blob = to_model_blob(model, feat_names, mtr | {"acc_test": test_acc})
        key = f"model_v2:{name}"
        try:
            set_setting(key, json.dumps(blob, separators=(",", ":")))
            trained[name] = True
            details[name] = {"n": int(n), "d": int(X.shape[1]), "test_acc": round(test_acc,3)}
            log.info("✅ %s saved as %s (n=%d, d=%d, test=%.3f)", name, key, n, X.shape[1], test_acc)
        except Exception as e:
            trained[name] = False
            log.error("Save failed for %s: %s", name, e)

    default_thr = float(os.getenv("CONF_THRESHOLD", "75"))
    tuned = simple_tune_thresholds(test_scores, default_thr)
    for mk, thr in tuned.items():
        try:
            store_key = {
                "BTTS": "BTTS",
                "OU_2.5": "Over/Under 2.5",
                "OU_3.5": "Over/Under 3.5",
                "1X2_HOME": "1X2",
                "1X2_AWAY": "1X2",
            }.get(mk, mk)
            set_setting(f"conf_threshold:{store_key}", f"{thr}")
        except Exception as e:
            log.warning("Failed to save threshold for %s: %s", mk, e)

    ok = any(trained.values())
    return {
        "ok": bool(ok),
        "trained": trained,
        "model_details": details,
        "tuned_thresholds": tuned,
        "total_rows": len(rows)
    }

# ──────────────────────────────────────────────────────────────────────────────
# CLI wrapper
# ──────────────────────────────────────────────────────────────────────────────
def cleanup():
    global POOL
    if POOL:
        try:
            POOL.closeall()
            log.info("[TRAIN_DB] Closed pool")
        except Exception as e:
            log.warning("[TRAIN_DB] Close pool: %s", e)

def shutdown_handler(signum=None, frame=None):
    log.info("Shutdown signal received; cleaning up...")
    ShutdownManager.request_shutdown()
    cleanup()
    sys.exit(0)

if __name__ == "__main__":
    import signal, atexit, json as _json
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    atexit.register(cleanup)

    log.info("Starting training process...")
    res = train_models()
    print(_json.dumps(res, indent=2))
    if not res.get("ok", False):
        log.error("Training failed/skipped: %s", res.get("error", "unknown"))
        sys.exit(1)
    log.info("Training completed OK")
