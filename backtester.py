
"""
Backtester Script for Betting Model
Test Markets: BTTS, OU (Over/Under), 1X2
Assumes tip_snapshots, match_results, and model thresholds are available.
"""

import os
import json
import psycopg2
import pandas as pd
from typing import Optional, Dict
from datetime import datetime

# Set up database connection (from environment variable)
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL must be set in environment variables.")

def connect_db():
    return psycopg2.connect(DATABASE_URL)

def get_tip_snapshots(conn) -> pd.DataFrame:
    query = """
    SELECT s.match_id, s.created_ts, s.payload, r.final_goals_h, r.final_goals_a, r.btts_yes
    FROM tip_snapshots s
    JOIN match_results r ON r.match_id = s.match_id
    """
    return pd.read_sql(query, conn)

def parse_tips(row, thresholds: Dict[str, float]):
    payload = row.get("payload", {})
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except:
            return []

    tips = []
    preds = payload.get("preds") or {}
    odds = payload.get("odds") or {}

    for key, prob in preds.items():
        threshold = thresholds.get(key)
        if threshold is None or float(prob)*100 < threshold:
            continue

        if key == "BTTS_YES":
            result = int(row["btts_yes"]) == 1
        elif key.startswith("OU_"):
            line = float(key.split("_", 1)[1].replace(",", "."))
            goals = (row["final_goals_h"] or 0) + (row["final_goals_a"] or 0)
            result = goals > line
        elif key.startswith("WLD_"):
            diff = (row["final_goals_h"] or 0) - (row["final_goals_a"] or 0)
            result = (
                (key == "WLD_HOME" and diff > 0) or
                (key == "WLD_DRAW" and diff == 0) or
                (key == "WLD_AWAY" and diff < 0)
            )
        else:
            continue

        stake = 1.0
        odd = float(odds.get(key) or 0)
        profit = (odd - 1) * stake if result else -stake

        tips.append({
            "match_id": row["match_id"],
            "created_ts": row["created_ts"],
            "market": key,
            "prob": float(prob),
            "threshold": threshold,
            "odd": odd,
            "result": result,
            "profit": profit
        })
    return tips

def get_thresholds(conn) -> Dict[str, float]:
    cursor = conn.cursor()
    cursor.execute("SELECT key, value FROM settings WHERE key LIKE 'conf_threshold:%'")
    rows = cursor.fetchall()
    out = {}
    for key, value in rows:
        try:
            label = key.split(":", 1)[1].strip().replace(" ", "_").upper()
            out[label] = float(value)
        except:
            continue
    return out

def summarize(df: pd.DataFrame, market: str):
    df = df[df["market"] == market]
    if df.empty:
        return f"Market: {market}\nNo tips made.\n"

    total = len(df)
    wins = df["result"].sum()
    winrate = wins / total * 100
    avg_odds = df["odd"].mean()
    roi = df["profit"].sum() / total * 100
    profit = df["profit"].sum()

    lines = [
        f"Market: {market}",
        "-" * 30,
        f"Total Bets:     {total}",
        f"Win Rate:       {winrate:.2f}%",
        f"Avg Odds:       {avg_odds:.2f}",
        f"ROI:            {roi:.2f}%",
        f"Total Profit:   â¬{profit:.2f}",
        ""
    ]
    return "\n".join(lines)

def main():
    conn = connect_db()
    
    print("Pulling tip snapshots from the database...")
    df_raw = get_tip_snapshots(conn)
    print("Tip snapshots pulled:", len(df_raw))
    
    thresholds = get_thresholds(conn)

    all_tips = []
    for _, row in df_raw.iterrows():
        tips = parse_tips(row, thresholds)
        all_tips.extend(tips)

    df_tips = pd.DataFrame(all_tips)
    if df_tips.empty:
        print("No tips met the confidence thresholds.")
        return

    for market in sorted(df_tips["market"].unique()):
        print(summarize(df_tips, market))

if __name__ == "__main__":
    main()
