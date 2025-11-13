import threading
from flask import Flask, jsonify
from apscheduler.schedulers.background import BackgroundScheduler

# API-Football
from app.api_football import (
    get_live_fixtures,
    get_fixture_stats,
    api_get,
    normalize_fixture,
)

# ML Prediction Engine
from app.predictor import (
    predict_fixture_1x2,
    predict_fixture_ou25,
    predict_fixture_btts,
)

# Filters
from app.filters import filter_predictions

# Telegram
from app.telegram_bot import send_predictions

# Supabase logging
from app.supabase_db import (
    upsert_fixture,
    insert_stats_snapshot,
    insert_odds_snapshot,
    insert_tip,
)

# Manual training
from train_models import run_full_training

from app.config import DEBUG_MODE, LIVE_PREDICTION_INTERVAL


app = Flask(__name__)
scheduler = BackgroundScheduler(timezone="UTC")


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "goalsniper-ai"})


@app.route("/manual/health-full")
def health_full():
    """
    Returns internal system indicators — useful for debugging.
    """
    return jsonify({
        "status": "ok",
        "fixtures_source": "API-FOOTBALL",
        "models": "loaded",
        "scheduler_running": scheduler.running,
    })


# ============================================================
# MANUAL ENDPOINTS
# ============================================================

@app.route("/manual/train")
def manual_train():
    """
    Trigger full model training manually.
    """
    run_full_training()
    return jsonify({"status": "training started"})


@app.route("/manual/scan")
def manual_scan():
    """
    Trigger a one-off scan manually.
    """
    live_prediction_cycle()
    return jsonify({"status": "scan completed"})


# ============================================================
# LIVE ODDS PARSER (SAFE)
# ============================================================

def extract_core_odds(odds_live):
    """
    Accepts raw live odds feed from /odds/live and extracts:

    • Fulltime 1X2
    • Fulltime Both Teams To Score
    • Fulltime Over/Under 2.5

    Works safely even if odds feed contains exotic markets only.
    """

    out_1x2 = {"home": None, "draw": None, "away": None}
    out_btts = {"yes": None, "no": None}
    out_ou25 = {"over": None, "under": None}

    if not odds_live:
        return out_1x2, out_ou25, out_btts

    for m in odds_live:
        name = m.get("name", "").lower()
        values = m.get("values", [])

        # -------- FULLTIME 1X2 --------
        if "fulltime result" in name or name == "match winner" or name == "1x2":
            for v in values:
                val = v.get("value", "").lower()
                odd = v.get("odd")
                if odd is None:
                    continue

                if val == "home":
                    out_1x2["home"] = float(odd)
                elif val == "draw":
                    out_1x2["draw"] = float(odd)
                elif val == "away":
                    out_1x2["away"] = float(odd)

        # -------- BTTS FT --------
        if "both teams to score" in name and "half" not in name:
            for v in values:
                val = v.get("value", "").lower()
                odd = v.get("odd")
                if odd is None:
                    continue

                if val == "yes":
                    out_btts["yes"] = float(odd)
                elif val == "no":
                    out_btts["no"] = float(odd)

        # -------- O/U 2.5 --------
        if "over/under" in name or "line" in name:
            for v in values:
                label = v.get("value", "").lower()
                handicap = str(v.get("handicap", "")).strip()
                odd = v.get("odd")

                if odd is None:
                    continue

                if handicap == "2.5":
                    if "over" in label:
                        out_ou25["over"] = float(odd)
                    elif "under" in label:
                        out_ou25["under"] = float(odd)

    return out_1x2, out_ou25, out_btts


# ============================================================
# PROCESS A FIXTURE
# ============================================================

def process_fixture(f_raw):
    """
    Takes raw fixture from API-Football and returns predictions.
    Also logs fixture + stats + odds to Supabase.
    """

    fixture = normalize_fixture(f_raw)
    fid = fixture["fixture_id"]

    # ---- FIXTURE SNAPSHOT ----
    upsert_fixture(fixture)

    # ---- STATS SNAPSHOT ----
    stats_raw = get_fixture_stats(fid)
    stats = {
        "minute": fixture.get("minute", 0),
        "home_shots_on": 0,
        "away_shots_on": 0,
        "home_attacks": 0,
        "away_attacks": 0,
    }
    if stats_raw:
        for team in stats_raw:
            tid = team.get("team", {}).get("id")
            for s in team.get("statistics", []):
                t, v = s.get("type"), s.get("value") or 0
                if t == "Shots on Goal":
                    if tid == fixture["home_team_id"]:
                        stats["home_shots_on"] = v
                    else:
                        stats["away_shots_on"] = v
                if t == "Attacks":
                    if tid == fixture["home_team_id"]:
                        stats["home_attacks"] = v
                    else:
                        stats["away_attacks"] = v

    insert_stats_snapshot(fid, stats)

    # ---- ODDS SNAPSHOT ----
    odds_live = api_get("odds/live", {"fixture": fid})

    # odds structure is: { "response": [ { ... } ] }
    odds_root = None
    try:
        odds_root = odds_live[0]["odds"]
    except Exception:
        odds_root = []

    odds_1x2, odds_ou25, odds_btts = extract_core_odds(odds_root)

    insert_odds_snapshot(fid, {
        "1x2": odds_1x2,
        "ou25": odds_ou25,
        "btts": odds_btts,
    })

    # ---- ML FEATURES ----
    prematch_strength = {"home": 0.0, "away": 0.0}
    prematch_goal_exp = 2.6
    prematch_btts_exp = 0.55

    preds = []
    preds += predict_fixture_1x2(fixture, stats, odds_1x2, prematch_strength)
    preds += predict_fixture_ou25(fixture, stats, odds_ou25, prematch_goal_exp)
    preds += predict_fixture_btts(fixture, stats, odds_btts, prematch_btts_exp)

    return preds


# ============================================================
# MAIN LIVE CYCLE
# ============================================================

def live_prediction_cycle():
    print("[GOALSNIPER] Live scan triggered...")

    fixtures_raw = get_live_fixtures()
    all_preds = []

    for fr in fixtures_raw:
        all_preds += process_fixture(fr)

    final_preds = filter_predictions(all_preds)

    if not final_preds:
        print("[GOALSNIPER] No predictions passed filters.")
        return

    # Send to Telegram
    send_predictions(final_preds)

    # Save to Supabase
    for p in final_preds:
        insert_tip({
            "fixture_id": p.fixture_id,
            "market": p.market,
            "selection": p.selection,
            "prob": p.prob,
            "odds": p.odds,
            "ev": p.ev,
            "minute": p.aux.get("minute"),
        })

    print(f"[GOALSNIPER] Predictions sent & stored.")


# ============================================================
# SCHEDULER
# ============================================================

def start_scheduler_once():
    if scheduler.running:
        return
    scheduler.add_job(live_prediction_cycle, "interval", seconds=LIVE_PREDICTION_INTERVAL)
    scheduler.start()
    print(f"[GOALSNIPER] Scheduler started ({LIVE_PREDICTION_INTERVAL}s)")


def start_async_scheduler():
    threading.Thread(target=start_scheduler_once).start()


start_async_scheduler()


# ============================================================
# FLASK ENTRY
# ============================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=DEBUG_MODE)
