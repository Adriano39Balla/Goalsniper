from supabase import create_client, Client
from typing import Dict, List, Any
from datetime import datetime
from app.config import (
    SUPABASE_URL,
    SUPABASE_KEY,
    TABLE_FIXTURES,
    TABLE_ODDS,
    TABLE_STATS,
    TABLE_TIPS,
    TABLE_RESULTS
)

# ---------------------------------------------------------
# Initialize Supabase Client
# ---------------------------------------------------------

def get_supabase() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)

sb = get_supabase()


# ---------------------------------------------------------
# INSERT DATA
# ---------------------------------------------------------

def insert_fixture(f: Dict[str, Any]):
    return sb.table(TABLE_FIXTURES).insert(f).execute()


def insert_odds(o: Dict[str, Any]):
    return sb.table(TABLE_ODDS).insert(o).execute()


def insert_stats(s: Dict[str, Any]):
    return sb.table(TABLE_STATS).insert(s).execute()


def insert_tip(t: Dict[str, Any]):
    """
    Insert a prediction/tip into Supabase. Called from main.py after
    we generate predictions and pass filters.
    """
    t["created_at"] = datetime.utcnow().isoformat()
    return sb.table(TABLE_TIPS).insert(t).execute()


def insert_result(r: Dict[str, Any]):
    """
    Insert the result of a tip after the match ends.
    """
    r["resolved_at"] = datetime.utcnow().isoformat()
    return sb.table(TABLE_RESULTS).insert(r).execute()


# ---------------------------------------------------------
# QUERY DATA FOR TRAINING (train_models.py)
# ---------------------------------------------------------

def fetch_training_fixtures(limit: int = 50000):
    """
    Fetch historical fixtures for ML training.
    """
    return (
        sb.table(TABLE_FIXTURES)
        .select("*")
        .order("timestamp", desc=False)
        .limit(limit)
        .execute()
        .data
    )


def fetch_training_tips():
    """
    Fetch all tips with their results for supervised learning.
    """
    return (
        sb.table(TABLE_TIPS)
        .select("*, tip_results(*)")
        .execute()
        .data
    )


# ---------------------------------------------------------
# MATCH RESULT RESOLUTION (main.py)
# ---------------------------------------------------------

def unresolved_tips():
    """
    Return all tips that need resolution (their fixture is finished).
    """
    return (
        sb.table(TABLE_TIPS)
        .select("*")
        .is_("resolved", None)
        .execute()
        .data
    )


def mark_tip_resolved(tip_id: int):
    return (
        sb.table(TABLE_TIPS)
        .update({"resolved": True})
        .eq("id", tip_id)
        .execute()
    )


# ---------------------------------------------------------
# MODEL VERSIONING (optional but recommended)
# ---------------------------------------------------------

def record_model_version(version: str, notes: str = ""):
    return sb.table("model_versions").insert({
        "version": version,
        "notes": notes,
        "created_at": datetime.utcnow().isoformat()
    }).execute()


def get_latest_model_version():
    resp = (
        sb.table("model_versions")
        .select("*")
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    if resp.data:
        return resp.data[0]["version"]
    return None
