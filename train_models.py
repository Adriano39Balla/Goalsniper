import os, json, time, sqlite3, requests, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

DB_PATH = "tip_performance.db"
APISPORTS_KEY = os.getenv("APISPORTS_KEY") or os.getenv("API_KEY")
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": APISPORTS_KEY, "Accept": "application/json"}

FEATURES = [
    # same names as extract_features / snapshot.stat
    "minute","goals_h","goals_a","goals_sum","goals_diff",
    "xg_h","xg_a","xg_sum","xg_diff",
    "sot_h","sot_a","sot_sum",
    "cor_h","cor_a","cor_sum",
    "pos_h","pos_a","pos_diff",
    "red_h","red_a","red_diff",
]

def db(): return sqlite3.connect(DB_PATH)

def api_get(path, params=None):
    for _ in range(3):
        try:
            r = requests.get(f"{BASE_URL}{path}", headers=HEADERS, params=params or {}, timeout=15)
            if r.ok: return r.json()
            time.sleep(0.5)
        except Exception:
            time.sleep(0.5)
    return None

def backfill_results(match_ids):
    if not match_ids: return
    with db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS match_results (
              match_id INTEGER PRIMARY KEY,
              final_goals_h INTEGER,
              final_goals_a INTEGER,
              btts_yes INTEGER,
              updated_ts INTEGER
            )""")
        B = 20
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
                if status in ("FT","AET","PEN","PST","CANC","ABD","AWD","WO"):
                    conn.execute("""
                      INSERT OR REPLACE INTO match_results(match_id, final_goals_h, final_goals_a, btts_yes, updated_ts)
                      VALUES (?,?,?,?,?)
                    """, (int(fid), gh, ga, 1 if (gh>0 and ga>0) else 0, int(time.time())))
        conn.commit()

def _g(d, key, default=0.0):
    try:
        v = d.get(key, default)
        return float(v)
    except Exception:
        return float(default)

def fetch_training_rows():
    with db() as conn:
        cur = conn.execute("SELECT DISTINCT match_id FROM tip_snapshots")
        all_ids = [r[0] for r in cur.fetchall()]
    if not all_ids:
        return None

    # ensure final results exist
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
            stat = snap.get("stat", {})
            minute = _g(snap, "minute")
            gh_s = _g(snap, "gh"); ga_s = _g(snap, "ga")
            row = {
                "minute": minute,
                "goals_h": gh_s, "goals_a": ga_s,
                "goals_sum": gh_s + ga_s, "goals_diff": gh_s - ga_s,
                "xg_h": _g(stat, "xg_h"), "xg_a": _g(stat, "xg_a"),
                "xg_sum": _g(stat, "xg_h") + _g(stat, "xg_a"),
                "xg_diff": _g(stat, "xg_h") - _g(stat, "xg_a"),
                "sot_h": _g(stat, "sot_h"), "sot_a": _g(stat, "sot_a"),
                "sot_sum": _g(stat, "sot_h") + _g(stat, "sot_a"),
                "cor_h": _g(stat, "cor_h"), "cor_a": _g(stat, "cor_a"),
                "cor_sum": _g(stat, "cor_h") + _g(stat, "cor_a"),
                "pos_h": _g(stat, "pos_h"), "pos_a": _g(stat, "pos_a"),
                "pos_diff": _g(stat, "pos_h") - _g(stat, "pos_a"),
                "red_h": _g(stat, "red_h"), "red_a": _g(stat, "red_a"),
                "red_diff": _g(stat, "red_h") - _g(stat, "red_a"),
            }
            X.append([row[k] for k in FEATURES])

            total_ft = int(gh) + int(ga)
            y_o25.append(1 if total_ft >= 3 else 0)
            y_btts.append(1 if (int(gh)>0 and int(ga)>0) else 0)

    return np.array(X, dtype=float), np.array(y_o25, dtype=int), np.array(y_btts, dtype=int)

def fit_and_dump(X, y, name):
    strat = y if (y.sum() and y.sum()!=len(y)) else None
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.25, random_state=42, stratify=strat)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
    clf.fit(Xtr, ytr)
    va_acc = clf.score(Xva, yva)
    print(f"{name}: val_acc={va_acc:.3f}")
    return {
        "features": FEATURES,
        "coef": clf.coef_.ravel().tolist(),
        "intercept": float(clf.intercept_[0]),
        "val_acc": float(va_acc),
    }

def save_models_to_settings(models_dict):
    with db() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)""")
        conn.execute("""INSERT INTO settings(key,value) VALUES(?,?)
                        ON CONFLICT(key) DO UPDATE SET value=excluded.value""",
                     ("model_coeffs", json.dumps(models_dict)))
        conn.commit()

def main():
    assert APISPORTS_KEY, "Set APISPORTS_KEY in env to fetch final results."
    out = fetch_training_rows()
    if not out:
        print("No snapshots found. Enable HARVEST_MODE=1 for a day, then rerun.")
        return
    X, y_o25, y_btts = out
    if len(X) < 200:
        print(f"Not enough rows ({len(X)}). Harvest more snapshots or wait for auto labels.")
        return
    m1 = fit_and_dump(X, y_o25, "O25")
    m2 = fit_and_dump(X, y_btts, "BTTS_YES")
    models = {"O25": m1, "BTTS_YES": m2, "trained_at": int(time.time())}
    save_models_to_settings(models)
    print("Saved model_coeffs to settings.")
    print("Tip: set ONLY_MODEL_MODE=1 to use learned models only.")

if __name__ == "__main__":
    main()
