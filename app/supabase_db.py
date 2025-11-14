from supabase import create_client
from app.config import SUPABASE_URL, SUPABASE_KEY
import datetime

# ------------------------------------------------------------
# INIT CLIENT
# ------------------------------------------------------------
try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("[SUPABASE] Client initialized.")
except Exception as e:
    print("[SUPABASE] FAILED TO INITIALIZE:", e)
    supabase = None


# ------------------------------------------------------------
# Helper: convert UNIX timestamp → ISO UTC
# ------------------------------------------------------------
def convert_timestamp(ts):
    """
    Converts UNIX timestamp to ISO-8601 format.
    If already a string or invalid → return as-is.
    """
    if ts is None:
        return None

    # Already iso format?
    if isinstance(ts, str) and "T" in ts:
        return ts

    try:
        if isinstance(ts, (int, float)):
            return datetime.datetime.utcfromtimestamp(ts).isoformat() + "Z"
    except Exception:
        pass

    return ts  # fallback


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
        # FIX UNIX TIMESTAMP
        if "timestamp" in fix:
            fix["timestamp"] = convert_timestamp(fix["timestamp"])

        res = (
            supabase.table("fixtures_live")
            .upsert(fix, on_conflict="fixture_id")
            .execute()
        )

        # Debug
        # print(f"[DB] Upserted fixture {fix.get('fixture_id')}")
        return res

    except Exception as e:
        print("[DB] upsert_fixture error:", e)
        print("[DB] Payload was:", fix)
        return None


# ============================================================
# STATS SNAPSHOT
# ============================================================

def insert_stats(fid, stats):
    payload = {"fixture_id": fid, **stats}

    if not supabase:
        print("[DB] insert_stats skipped — client not initialized")
        return

    try:
        return supabase.table("fixture_stats").insert(payload).execute()

    except Exception as e:
        print("[DB] insert_stats error:", e)
        print("[DB] Payload was:", payload)
        return None


# ============================================================
# ODDS SNAPSHOT
# ============================================================

def insert_odds(fid, odds):
    payload = {
        "fixture_id": fid,
        "odds": odds
    }

    if not supabase:
        print("[DB] insert_odds skipped — client not initialized")
        return

    try:
        return supabase.table("fixture_odds").insert(payload).execute()

    except Exception as e:
        print("[DB] insert_odds error:", e)
        print("[DB] Payload was:", payload)
        return None


# ============================================================
# PREDICTION STORAGE
# ============================================================

def insert_tip(tip):
    if not supabase:
        print("[DB] insert_tip skipped — client not initialized")
        return

    try:
        return supabase.table("tips").insert(tip).execute()

    except Exception as e:
        print("[DB] insert_tip error:", e)
        print("[DB] Payload was:", tip)
        return None
