import os
import json
import time
from supabase import create_client, Client

# ============================================================
# ENV → LAZY LOAD CLIENT
# ============================================================

_supabase: Client | None = None

def get_supabase() -> Client:
    global _supabase

    if _supabase is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")

        if not url or not key:
            raise ValueError(
                f"[Supabase] Missing environment variables. "
                f"SUPABASE_URL={url}, KEY={'SET' if key else 'MISSING'}"
            )

        _supabase = create_client(url, key)
        print("[SUPABASE] Client initialized.")

    return _supabase


# ============================================================
# FIXTURES
# ============================================================

def upsert_fixture(data: dict):
    """
    Inserts/updates fixtures in the clean fixtures table.
    """
    try:
        supabase = get_supabase()

        payload = {
            "fixture_id": data["fixture_id"],
            "league_id": data.get("league_id"),
            "home_team_id": data.get("home_id"),
            "away_team_id": data.get("away_id"),
            "home_goals": data.get("home_goals"),
            "away_goals": data.get("away_goals"),
            "status": data.get("status"),
            "minute": data.get("minute"),
            "timestamp": data.get("timestamp"),
        }

        supabase.table("fixtures").upsert(
            payload, on_conflict="fixture_id"
        ).execute()

        print(f"[SUPABASE] Fixture {data['fixture_id']} upserted.")

    except Exception as e:
        print("[SUPABASE] Error upserting fixture:", e)


# ============================================================
# ODDS
# ============================================================

def insert_odds(fixture_id: int, odds_data: dict):
    """
    Inserts flattened odds rows into the new odds table.
    """
    supabase = get_supabase()
    rows = []

    for bookmaker, markets in odds_data.items():
        for market, selections in markets.items():
            for selection, odd in selections.items():
                if odd is None:
                    continue

                rows.append({
                    "fixture_id": fixture_id,
                    "bookmaker": bookmaker,
                    "market": market,
                    "selection": selection,
                    "odd": float(odd),
                })

    if not rows:
        return

    try:
        supabase.table("odds").insert(rows).execute()
        print(f"[SUPABASE] Inserted {len(rows)} odds rows for fixture {fixture_id}")

    except Exception as e:
        print("[SUPABASE] Error inserting odds:", e)


# ============================================================
# LIVE STATS
# ============================================================

def insert_stats(fixture_id: int, stats: dict):
    """
    Inserts live match stats into clean stats table.
    """
    try:
        supabase = get_supabase()

        payload = {
            "fixture_id": fixture_id,
            "minute": stats.get("minute"),
            "home_shots_on": stats.get("home_shots_on"),
            "away_shots_on": stats.get("away_shots_on"),
            "home_attacks": stats.get("home_attacks"),
            "away_attacks": stats.get("away_attacks"),
            "snapshot": stats,
            "raw": stats,
        }

        supabase.table("stats").insert(payload).execute()

        print(f"[SUPABASE] Stats inserted for fixture {fixture_id}")

    except Exception as e:
        print("[SUPABASE] Error inserting stats:", e)


# ============================================================
# TIPS — BETS SENT BY THE BOT
# ============================================================

def insert_tip(tip: dict):
    try:
        supabase = get_supabase()
        supabase.table("tips").insert(tip).execute()

        print(f"[SUPABASE] Tip inserted: {tip}")

    except Exception as e:
        print("[SUPABASE] Error inserting tip:", e)


# ============================================================
# TIP RESULTS — BET GRADING
# ============================================================

def insert_tip_result(result: dict):
    try:
        supabase = get_supabase()
        supabase.table("tip_results").insert(result).execute()

        print("[SUPABASE] Tip result saved.")

    except Exception as e:
        print("[SUPABASE] Error inserting tip result:", e)


# ============================================================
# DAILY ROI LOG
# ============================================================

def write_daily_roi(day: str, tips_count: int, pnl: float):
    roi = pnl / max(tips_count, 1)

    try:
        supabase = get_supabase()

        supabase.table("roi_log").upsert(
            {
                "day": day,
                "tips_count": tips_count,
                "pnl": pnl,
                "roi": roi,
            },
            on_conflict="day"
        ).execute()

        print(f"[SUPABASE] Daily ROI logged for {day}")

    except Exception as e:
        print("[SUPABASE] Error writing daily ROI:", e)
