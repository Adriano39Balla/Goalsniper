from supabase import create_client, Client
from typing import Dict, Any, Optional
from datetime import datetime

from app.config import (
    SUPABASE_URL,
    SUPABASE_KEY,
    TABLE_FIXTURES,
    TABLE_STATS,
    TABLE_ODDS,
    TABLE_RESULTS,
)

# ---------------------------------------------------------
# INITIALIZE CLIENT
# ---------------------------------------------------------

def get_supabase() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)

sb = get_supabase()

def log(msg: str):
    print(f"[SUPABASE] {msg}")


# ---------------------------------------------------------
# INSERT FIXTURE SNAPSHOT
# ---------------------------------------------------------

def upsert_fixture(fix: Dict[str, Any]):
    """
    Stores each live snapshot of the fixture.
    Upsert = no duplicate rows for same fixture_id.
    """
    fix["created_at"] = datetime.utcnow().isoformat()

    try:
        sb.table(TABLE_FIXTURES).upsert(fix, on_conflict="fixture_id").execute()
        log(f"Fixture {fix['fixture_id']} upserted.")
    except Exception as e:
        log(f"Error upserting fixture: {e}")


# ---------------------------------------------------------
# INSERT STATS SNAPSHOT
# ---------------------------------------------------------

def insert_stats_snapshot(fixture_id: int, stats: Dict[str, Any]):
    """
    Save each stats snapshot. Do NOT upsert —  
    we keep historical timeline of stats for training time decay.
    """
    row = {
        "fixture_id": fixture_id,
        "snapshot": stats,
        "created_at": datetime.utcnow().isoformat(),
    }

    try:
        sb.table(TABLE_STATS).insert(row).execute()
        log(f"Stats inserted for fixture {fixture_id}.")
    except Exception as e:
        log(f"Error inserting stats: {e}")


# ---------------------------------------------------------
# INSERT ODDS SNAPSHOT
# ---------------------------------------------------------

def insert_odds_snapshot(fixture_id: int, odds: Dict[str, Any]):
    """
    Save each odds snapshot (1X2, OU25, BTTS).
    """
    row = {
        "fixture_id": fixture_id,
        "odds_1x2": odds.get("1x2"),
        "odds_ou25": odds.get("ou25"),
        "odds_btts": odds.get("btts"),
        "created_at": datetime.utcnow().isoformat(),
    }

    try:
        sb.table(TABLE_ODDS).insert(row).execute()
        log(f"Odds inserted for fixture {fixture_id}.")
    except Exception as e:
        log(f"Error inserting odds: {e}")


# ---------------------------------------------------------
# INSERT PREDICTION (ALREADY EXISTS)
# ---------------------------------------------------------

def insert_tip(t: Dict[str, Any]):
    t["created_at"] = datetime.utcnow().isoformat()
    try:
        sb.table("tips").insert(t).execute()
    except Exception as e:
        log(f"Error inserting tip: {e}")


# ---------------------------------------------------------
# FETCH FINAL MATCH RESULT & CREATE LABELS
# ---------------------------------------------------------

def create_labels_from_result(home_goals: int, away_goals: int):
    """Generate ML labels."""
    # 1X2 label
    if home_goals > away_goals:
        l1 = 0       # home
    elif home_goals == away_goals:
        l1 = 1       # draw
    else:
        l1 = 2       # away

    # OU 2.5 label
    total = home_goals + away_goals
    lou = 1 if total > 2 else 0    # over = 1, under = 0

    # BTTS label
    lbtts = 1 if (home_goals > 0 and away_goals > 0) else 0

    return l1, lou, lbtts


def insert_match_result(fixture_id: int, home_goals: int, away_goals: int):
    l1, lou, lbtts = create_labels_from_result(home_goals, away_goals)

    row = {
        "fixture_id": fixture_id,
        "final_home_goals": home_goals,
        "final_away_goals": away_goals,
        "label_1x2": l1,
        "label_ou25": lou,
        "label_btts": lbtts,
        "resolved_at": datetime.utcnow().isoformat(),
    }

    try:
        sb.table(TABLE_RESULTS).upsert(row, on_conflict="fixture_id").execute()
        log(f"Result + labels saved for fixture {fixture_id}.")
    except Exception as e:
        log(f"Error inserting result: {e}")


# ---------------------------------------------------------
# TRAINING DATA FETCHING (used by train_models.py)
# ---------------------------------------------------------

def fetch_training_fixtures(limit: int = 50000):
    resp = sb.table(TABLE_FIXTURES) \
             .select("*") \
             .order("timestamp", desc=False) \
             .limit(limit) \
             .execute()
    return resp.data or []


def fetch_training_results():
    resp = sb.table(TABLE_RESULTS).select("*").execute()
    return resp.data or []
