from supabase import create_client
from app.config import SUPABASE_URL, SUPABASE_KEY

# ------------------------------------------------------------
# INIT CLIENT
# ------------------------------------------------------------
try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("[SUPABASE] Client initialized.")
except Exception as e:
    print("[SUPABASE] FAILED TO INITIALIZE:", e)
    supabase = None


# ============================================================
# FIXTURE SNAPSHOT
# ============================================================

def upsert_fixture(fix):
    """
    fix MUST contain:
        fixture_id, league_id, home_team_id, away_team_id,
        home_goals, away_goals, status, minute, timestamp
    """
    if not supabase:
        print("[DB] upsert_fixture skipped — client not initialized")
        return

    try:
        res = (
            supabase.table("fixtures_live")
            .upsert(fix, on_conflict="fixture_id")
            .execute()
        )
        # Debug print
        # print(f"[DB] Upsert fixture {fix.get('fixture_id')}")
        return res

    except Exception as e:
        print("[DB] upsert_fixture error:", e)
        return None


# ============================================================
# STATS SNAPSHOT
# ============================================================

def insert_stats(fid, stats):
    """
    stats MUST contain:
        minute, home_shots_on, away_shots_on,
        home_attacks, away_attacks
    """

    if not supabase:
        print("[DB] insert_stats skipped — client not initialized")
        return None

    payload = {
        "fixture_id": fid,
        **stats
    }

    try:
        res = supabase.table("fixture_stats").insert(payload).execute()
        # print(f"[DB] Stats inserted for {fid}")
        return res

    except Exception as e:
        print("[DB] insert_stats error:", e)
        print("[DB] Payload was:", payload)
        return None


# ============================================================
# ODDS SNAPSHOT
# ============================================================

def insert_odds(fid, odds):
    """
    odds MUST be structured like:
    {
        "1x2": {"home": X, "draw": Y, "away": Z},
        "ou25": {"over": X, "under": Y},
        "btts": {"yes": X, "no": Y}
    }
    """

    if not supabase:
        print("[DB] insert_odds skipped — client not initialized")
        return None

    payload = {
        "fixture_id": fid,
        "odds": odds
    }

    try:
        res = supabase.table("fixture_odds").insert(payload).execute()
        # print(f"[DB] Odds inserted for {fid}")
        return res

    except Exception as e:
        print("[DB] insert_odds error:", e)
        print("[DB] Payload was:", payload)
        return None


# ============================================================
# PREDICTION STORAGE
# ============================================================

def insert_tip(tip):
    """
    tip MUST contain:
        fixture_id, market, selection, prob, odds, ev, minute
    """

    if not supabase:
        print("[DB] insert_tip skipped — client not initialized")
        return None

    try:
        res = supabase.table("tips").insert(tip).execute()
        # print(f"[DB] Tip stored for {tip.get('fixture_id')}")
        return res

    except Exception as e:
        print("[DB] insert_tip error:", e)
        print("[DB] Payload was:", tip)
        return None
