import json
import time
from supabase import create_client, Client
from app.config import SUPABASE_URL, SUPABASE_KEY

# =========================================================
# INIT SUPABASE CLIENT
# =========================================================

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
print("[SUPABASE] Client initialized.")


# =========================================================
# FIXTURES
# =========================================================

def upsert_fixture(data: dict):
    """
    Upserts a fixture row using the clean table schema.
    """

    try:
        payload = {
            "fixture_id": data["fixture_id"],
            "league_id": data.get("league_id"),
            "home_team_id": data.get("home_team_id"),
            "away_team_id": data.get("away_team_id"),
            "home_goals": data.get("home_goals"),
            "away_goals": data.get("away_goals"),
            "status": data.get("status"),
            "minute": data.get("minute"),
            "timestamp": data.get("timestamp"),
        }

        supabase.table("fixtures").upsert(
            payload,
            on_conflict="fixture_id"
        ).execute()

        print(f"[SUPABASE] Fixture {data['fixture_id']} upserted.")

    except Exception as e:
        print("[SUPABASE] ERROR upserting fixture:", e)


# =========================================================
# STATS SNAPSHOT
# =========================================================

def insert_stats_snapshot(fixture_id: int, stats: dict):
    """
    Inserts a row into stats table.
    """

    try:
        payload = {
            "fixture_id": fixture_id,
            "minute": stats.get("minute"),
            "home_shots_on": stats.get("home_shots_on"),
            "away_shots_on": stats.get("away_shots_on"),
            "home_attacks": stats.get("home_attacks"),
            "away_attacks": stats.get("away_attacks"),
            "snapshot": stats,   # raw JSON
        }

        supabase.table("stats").insert(payload).execute()
        print(f"[SUPABASE] Stats inserted for fixture {fixture_id}")

    except Exception as e:
        print("[SUPABASE] ERROR inserting stats:", e)


# =========================================================
# ODDS SNAPSHOT
# =========================================================

def insert_odds_snapshot(fixture_id: int, odds: dict):
    """
    Inserts a standardized live-odds snapshot.

    odds format:
    {
        "1x2": {"home": 1.50, "draw": 4.00, "away": 6.00},
        "ou25": {"over": 2.10, "under": 1.70},
        "btts": {"yes": 1.90, "no": 1.90}
    }
    """

    try:
        payload = {
            "fixture_id": fixture_id,
            "odds_1x2": odds.get("1x2"),
            "odds_ou25": odds.get("ou25"),
            "odds_btts": odds.get("btts"),
            "snapshot": odds,   # raw JSON saved for debugging
        }

        supabase.table("odds").insert(payload).execute()
        print(f"[SUPABASE] Odds snapshot inserted for fixture {fixture_id}")

    except Exception as e:
        print("[SUPABASE] ERROR inserting odds:", e)


# =========================================================
# TIPS (bot sends)
# =========================================================

def insert_tip(tip: dict):
    """
    Inserts a prediction that passed filters (Telegram-sent tip).
    """

    try:
        supabase.table("tips").insert(tip).execute()
        print(f"[SUPABASE] Tip inserted: {tip}")

    except Exception as e:
        print("[SUPABASE] ERROR inserting tip:", e)


# =========================================================
# TIP RESULTS (for grading)
# =========================================================

def insert_tip_result(result: dict):
    """
    Saves graded results for accuracy/ROI tracking.
    """

    try:
        supabase.table("tip_results").insert(result).execute()
        print("[SUPABASE] Tip result saved.")

    except Exception as e:
        print("[SUPABASE] ERROR inserting tip result:", e)


# =========================================================
# DAILY ROI LOGGING
# =========================================================

def write_daily_roi(day: str, tips_count: int, pnl: float):
    """
    Saves a daily ROI summary row.
    """

    try:
        roi = pnl / max(tips_count, 1)

        supabase.table("roi_log").upsert(
            {
                "day": day,
                "tips_count": tips_count,
                "pnl": pnl,
                "roi": roi,
            },
            on_conflict="day"
        ).execute()

        print(f"[SUPABASE] Daily ROI logged ({day})")

    except Exception as e:
        print("[SUPABASE] ERROR writing daily ROI:", e)
