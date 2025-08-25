# path: analyze_precision.py
import os
import re
import json
import psycopg2
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Tuple
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
OU_RE = re.compile(r"^Over\s*(\d+(?:\.\d+)?)\s*Goals$", re.I)
OUU_RE = re.compile(r"^Under\s*(\d+(?:\.\d+)?)\s*Goals$", re.I)

def _connect(db_url: str):
    if db_url.startswith("postgres://"):
        db_url = "postgresql://" + db_url[len("postgres://"):]
    if "sslmode=" not in db_url:
        db_url = db_url + ("&" if "?" in db_url else "?") + "sslmode=require"
    conn = psycopg2.connect(db_url)
    conn.autocommit = True
    return conn

def _read_sql(conn, sql: str, params=()):
    return pd.read_sql_query(sql, conn, params=params)

def _parse_tip_correct(suggestion: str, market: str, final_h: int, final_a: int) -> Optional[int]:
    """Return 1 if tip was correct, 0 if wrong, None if unknown."""
    s = (suggestion or "").strip()
    m = (market or "").strip().lower()
    total = int(final_h) + int(final_a)
    # Over/Under X.Y Goals
    mo = OU_RE.match(s)
    mu = OUU_RE.match(s)
    if mo:
        th = float(mo.group(1))
        return 1 if total > th else 0
    if mu:
        th = float(mu.group(1))
        return 1 if total < th else 0
    # BTTS
    if s.lower().startswith("btts"):
        yes = "yes" in s.lower()
        actual_yes = (int(final_h) > 0 and int(final_a) > 0)
        return 1 if (yes == actual_yes) else 0
    # 1X2
    sl = s.lower()
    if "home win" in sl:
        return 1 if final_h > final_a else 0
    if sl == "draw":
        return 1 if final_h == final_a else 0
    if "away win" in sl:
        return 1 if final_a > final_h else 0
    # unknown market -> skip
    return None

def load_tips_with_outcomes(conn, days: int = 30) -> pd.DataFrame:
    since_ts = int((datetime.utcnow() - timedelta(days=days)).timestamp())
    q = """
    SELECT t.match_id, t.market, t.suggestion, t.confidence,
           t.created_ts,
           r.final_goals_h, r.final_goals_a
    FROM tips t
    JOIN match_results r ON r.match_id = t.match_id
    WHERE t.created_ts >= %s
      AND t.sent_ok = 1
    """
    df = _read_sql(conn, q, (since_ts,))
    if df.empty:
        return df
    # confidence in your table is % (0–100) – convert to 0–1
    df["prob"] = (pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0) / 100.0).clip(0, 1)
    lab = []
    for _, row in df.iterrows():
        y = _parse_tip_correct(row["suggestion"], row["market"], row["final_goals_h"], row["final_goals_a"])
        lab.append(y)
    df["y"] = lab
    df = df.dropna(subset=["y"]).copy()
    df["y"] = df["y"].astype(int)
    return df

def precision_at_threshold(df: pd.DataFrame, thr: float) -> Tuple[int, int, float]:
    sel = df[df["prob"] >= thr]
    n_preds = len(sel)
    n_true = int(sel["y"].sum()) if n_preds else 0
    prec = (n_true / n_preds) if n_preds else 0.0
    return n_true, n_preds, prec

def find_threshold_for_precision(df: pd.DataFrame, target: float = 0.90) -> Optional[float]:
    if df.empty: return None
    # scan thresholds on unique prob values (descending)
    probs = sorted(df["prob"].unique().tolist(), reverse=True)
    best = None
    for thr in probs:
        _, n_preds, prec = precision_at_threshold(df, thr)
        if n_preds >= 30 and prec >= target:  # require some support
            best = thr
    return best

def print_precision_table(df: pd.DataFrame):
    print("\nPrecision by threshold (every 0.02):")
    print("thr\tpreds\tcorrect\tprecision")
    for thr in np.arange(0.50, 0.991, 0.02):
        n_true, n_preds, prec = precision_at_threshold(df, thr)
        print(f"{thr:0.2f}\t{n_preds}\t{n_true}\t{prec:0.3f}")

def calibration_report(df: pd.DataFrame, bins: int = 10):
    df2 = df.copy()
    df2["bin"] = pd.qcut(df2["prob"].rank(method="first"), q=bins, labels=False)
    grp = df2.groupby("bin")
    rows = []
    for b, g in grp:
        if g.empty: continue
        mean_p = g["prob"].mean()
        hit = g["y"].mean()  # empirical prob
        rows.append((b, len(g), mean_p, hit))
    cal = pd.DataFrame(rows, columns=["bin","n","mean_pred","empirical"])
    print("\nCalibration deciles (mean predicted vs empirical):")
    print(cal.to_string(index=False))
    return cal

def plot_calibration(df: pd.DataFrame, out_png: str = "calibration.png"):
    y_true = df["y"].values
    y_prob = df["prob"].values
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    plt.figure(figsize=(5,5))
    plt.plot([0,1],[0,1])
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Empirical frequency")
    plt.title("Calibration")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"\nSaved calibration plot to {out_png}")

def main():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("Set DATABASE_URL in your environment.")
        return
    days = int(os.getenv("EVAL_DAYS", "30"))
    target_precision = float(os.getenv("TARGET_PRECISION", "0.90"))

    conn = _connect(db_url)
    try:
        df = load_tips_with_outcomes(conn, days=days)
        if df.empty:
            print("No labeled tips found in the time window. Expand days or wait for matches to finish.")
            return

        # Overall snapshot
        total_preds = len(df)
        total_correct = int(df["y"].sum())
        overall_prec = total_correct / total_preds if total_preds else 0
        print(f"Loaded {total_preds} tips, correct={total_correct} (overall precision={overall_prec:0.3f})")

        # Per-market snapshot (helps diagnose BTTS etc.)
        per_mkt = df.groupby("market")["y"].agg(["count","mean"]).sort_values("mean", ascending=False)
        print("\nPer-market precision:")
        print(per_mkt.to_string())

        # Threshold search
        thr = find_threshold_for_precision(df, target=target_precision)
        print_precision_table(df)
        if thr is None:
            print(f"\nCould not find a threshold that achieves {int(target_precision*100)}% precision with ≥30 tips.")
        else:
            n_true, n_preds, prec = precision_at_threshold(df, thr)
            print(f"\nRecommended MIN_PROB threshold for ~{int(target_precision*100)}% precision: {thr:0.3f} "
                  f"(n={n_preds}, correct={n_true}, precision={prec:0.3f})")

        # Calibration deciles + plot
        cal = calibration_report(df)
        plot_calibration(df, out_png=os.getenv("CALIB_PNG","calibration.png"))

        # Optional: write a JSON summary you can read from Robi and auto‑tune
        summary = {
            "evaluated_at_utc": datetime.utcnow().isoformat()+"Z",
            "days": days,
            "overall_precision": overall_prec,
            "suggested_min_prob": thr
        }
        with open(os.getenv("CALIB_SUMMARY","calibration_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print("\nWrote calibration_summary.json")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
