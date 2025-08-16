# train_models.py
import os, json, time, math, sqlite3, random
import requests
import numpy as np
from datetime import datetime, timezone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# from sklearn.isotonic import IsotonicRegression  # optional (off by default)

DB_PATH = "tip_performance.db"
APISPORTS_KEY = os.getenv("APISPORTS_KEY") or os.getenv("API_KEY")
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": APISPORTS_KEY, "Accept": "application/json"}

assert APISPORTS_KEY, "Set APISPORTS_KEY in env to fetch final results."

FEATURES = [
    "minute","goals_h","goals_a","goals_sum","goals_diff",
    "xg_h","xg_a","xg_sum","xg_diff",
    "sot_h","sot_a","sot_sum",
    "cor_h","cor_a","cor_sum",
]

def db():
    return sqlite3.connect(DB_PATH)

def api_get(path, params=None):
    for _ in range(3):
        try:
            r = requests.get(f"{BASE_URL}{path}", headers=HEADERS, params=params or {}, timeout=15)
            if r.ok:
                return r.json()
            time.sleep(0.5)
        except Exception:
            time.sleep(0.5)
    return None

def backfill_results(match_ids):
    """Fetch FT scores and store into match_results."""
    if not match_ids:
        return
    # API supports "ids" batch
    B = 20
    with db() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS match_results(
            match_id INTEGER PRIMARY KEY,
            final_goals_h INTEGER,
            final_goals_a INTEGER,
            btts_yes INTEGER,
            updated_ts INTEGER)""")
        for i in range(0, len(match_ids), B):
            batch = match_ids[i:i+B]
            js = api_get("/fixtures", {"ids": ",".join(str(i) for i in batch), "timezone":"UTC"})
            resp = (js or {}).get("response", [])
            for fx in resp:
                fid = (fx.get("fixture") or {}).get("id")
                status = (fx.get("fixture") or {}).get("status", {}).get("short")
                goals = fx.get("goals") or {}
                gh = int(goals.get("home") or 0)
                ga = int(goals.get("away") or 0)
                if status in ("FT","AET","PEN","PST","CANC","ABD","AWD","WO"):  # final/decided
                    conn.execute("""INSERT OR REPLACE INTO match_results(match_id,final_goals_h,final_goals_a,btts_yes,updated_ts)
                                    VALUES (?,?,?,?,?)""",
                                 (int(fid), gh, ga, 1 if (gh>0 and ga>0) else 0, int(time.time())))
        conn.commit()

def fetch_training_rows():
    """Return rows of (features, y_o25, y_btts)."""
    with db() as conn:
        # get match_ids we have snapshots for
        cur = conn.execute("SELECT DISTINCT match_id FROM tip_snapshots")
        all_ids = [r[0] for r in cur.fetchall()]
    if not all_ids:
        return [], []

    backfill_results(all_ids)

    X, y_o25, y_btts = [], [], []
    with db() as conn:
        q = """
        SELECT s.match_id, s.payload, r.final_goals_h, r.final_goals_a
        FROM tip_snapshots s
        JOIN match_results r ON r.match_id = s.match_id
        """
        for match_id, payload, gh, ga in conn.execute(q):
            try:
                snap = json.loads(payload)
            except Exception:
                continue
            # extract features with same names as main.py
            def g(name, default=0.0):
                v = snap.get("stat", {}).get(name, snap.get(name, default))
                try:
                    return float(v)
                except Exception:
                    return float(default)
            minute = float(snap.get("minute", 0))
            gh_ = float(snap.get("gh", 0)); ga_ = float(snap.get("ga", 0))
            feat = {
                "minute": minute,
                "goals_h": gh_, "goals_a": ga_,
                "goals_sum": gh_+ga_, "goals_diff": gh_-ga_,
                "xg_h": g("xg_h"), "xg_a": g("xg_a"),
                "xg_sum": g("xg_h")+g("xg_a"), "xg_diff": g("xg_h")-g("xg_a"),
                "sot_h": g("sot_h"), "sot_a": g("sot_a"), "sot_sum": g("sot_h")+g("sot_a"),
                "cor_h": g("cor_h"), "cor_a": g("cor_a"), "cor_sum": g("cor_h")+g("cor_a"),
            }
            X.append([feat[k] for k in FEATURES])
            total_ft = int(gh) + int(ga)
            y_o25.append(1 if total_ft >= 3 else 0)
            y_btts.append(1 if (int(gh)>0 and int(ga)>0) else 0)

    return (np.array(X, dtype=float), np.array(y_o25, dtype=int), np.array(y_btts, dtype=int))

def fit_and_dump(X, y, name):
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y if y.sum() and y.sum()!=len(y) else None)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
    clf.fit(Xtr, ytr)
    va_acc = clf.score(Xva, yva)
    print(f"{name}: val_acc={va_acc:.3f}")
    model = {
        "features": FEATURES,
        "coef": clf.coef_.ravel().tolist(),
        "intercept": float(clf.intercept_[0]),
        "val_acc": float(va_acc),
        # "isotonic": {"x": x_pts, "y": y_pts}  # optional later
    }
    return model

def save_models_to_settings(models_dict):
    with db() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)""")
        conn.execute("""INSERT INTO settings(key,value) VALUES(?,?)
                        ON CONFLICT(key) DO UPDATE SET value=excluded.value""",
                     ("model_coeffs", json.dumps(models_dict)))
        conn.commit()

def main():
    X, y_o25, y_btts = fetch_training_rows()
    if len(X) < 100:
        print(f"Not enough rows ({len(X)}) — gather more tips/feedback first.")
        return
    m1 = fit_and_dump(X, y_o25, "O25")
    m2 = fit_and_dump(X, y_btts, "BTTS_YES")
    models = {"O25": m1, "BTTS_YES": m2, "trained_at": int(time.time())}
    save_models_to_settings(models)
    print("Saved model_coeffs to settings.")

if __name__ == "__main__":
    main()
