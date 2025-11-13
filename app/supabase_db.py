from supabase import create_client
from app.config import SUPABASE_URL, SUPABASE_KEY

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# ============================================================
# FIXTURE SNAPSHOT
# ============================================================

def upsert_fixture(fix):
    try:
        supabase.table("fixtures_live").upsert(fix).execute()
    except Exception as e:
        print("[DB] upsert_fixture error:", e)


# ============================================================
# STATS SNAPSHOT
# ============================================================

def insert_stats_snapshot(fid, stats):
    try:
        supabase.table("fixture_stats").insert({
            "fixture_id": fid,
            **stats
        }).execute()
    except Exception as e:
        print("[DB] insert_stats_snapshot error:", e)


# ============================================================
# ODDS SNAPSHOT
# ============================================================

def insert_odds_snapshot(fid, odds):
    try:
        supabase.table("fixture_odds").insert({
            "fixture_id": fid,
            "odds": odds
        }).execute()
    except Exception as e:
        print("[DB] insert_odds_snapshot error:", e)


# ============================================================
# PREDICTION STORAGE
# ============================================================

def insert_tip(tip):
    try:
        supabase.table("tips").insert(tip).execute()
    except Exception as e:
        print("[DB] insert_tip error:", e)
