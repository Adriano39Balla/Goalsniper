import json
import time
from supabase import create_client, Client


SUPABASE_URL = "YOUR_URL"
SUPABASE_KEY = "YOUR_SERVICE_ROLE_KEY"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# =========================================================
# FIXTURES
# =========================================================

def upsert_fixture(data):
    """
    Inserts or updates the fixture record.
    Matches your clean new table structure.
    """
    try:
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

        res = (
            supabase.table("fixtures")
            .upsert(payload, on_conflict="fixture_id")
            .execute()
        )
        print(f"[SUPABASE] Fixture {data['fixture_id']} upserted.")
        return res

    except Exception as e:
        print("[SUPABASE] Error upserting fixture:", e)
        return None


# =========================================================
# ODDS
# =========================================================

def insert_odds(fixture_id: int, odds_data: dict):
    """
    Inserts odds into the new clean odds table.

    odds_data format MUST be:
    {
        "bet365": {
            "1X2": {"HOME": 1.80, "DRAW": 3.60, "AWAY": 4.20},
            "BTTS": {"YES": 1.95, "NO": 1.80},
            "OU_2.5": {"OVER": 2.10, "UNDER": 1.65}
        }
    }
    """

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


# =========================================================
# LIVE MATCH STATS
# =========================================================

def insert_stats(fixture_id: int, stats: dict):
    """
    Inserts live stats into the new stats table.

    stats format:
    {
        "minute": 55,
        "home_shots_on": 4,
        "away_shots_on": 2,
        "home_attacks": 63,
        "away_attacks": 51
    }
    """
    try:
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


# =========================================================
# TIPS
# =========================================================

def insert_tip(tip: dict):
    """
    Saves bot tips.
    These are the bets evaluated in ROI.
    """

    try:
        res = supabase.table("tips").insert(tip).execute()
        print(f"[SUPABASE] Tip inserted: {tip}")
        return res
    except Exception as e:
        print("[SUPABASE] Error inserting tip:", e)
        return None


# =========================================================
# TIP RESULTS (grading)
# =========================================================

def insert_tip_result(result: dict):
    """
    Every graded bet is saved here:
    {
        "tip_id": 123,
        "fixture_id": 1483272,
        "result": "WIN/LOSS/PUSH",
        "stake": 1,
        "odds": 1.85,
        "pnl": +0.85
    }
    """

    try:
        res = supabase.table("tip_results").insert(result).execute()
        print(f"[SUPABASE] Tip result saved.")
        return res
    except Exception as e:
        print("[SUPABASE] Error inserting tip result:", e)
        return None


# =========================================================
# DAILY ROI LOG
# =========================================================

def write_daily_roi(day: str, tips_count: int, pnl: float):
    roi = pnl / max(tips_count, 1)
    try:
        supabase.table("roi_log").upsert({
            "day": day,
            "tips_count": tips_count,
            "pnl": pnl,
            "roi": roi
        }, on_conflict="day").execute()

        print(f"[SUPABASE] Daily ROI logged for {day}")

    except Exception as e:
        print("[SUPABASE] Error writing daily ROI:", e)
