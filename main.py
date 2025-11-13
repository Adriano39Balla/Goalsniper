import threading
from flask import Flask, jsonify
from apscheduler.schedulers.background import BackgroundScheduler

# API-Football
from app.api_football import (
    get_live_fixtures,
    get_fixture_stats,
    get_live_odds,
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

# Supabase (NOW INCLUDING FULL LOGGING)
from app.supabase_db import (
    upsert_fixture,
    insert_stats_snapshot,
    insert_odds_snapshot,
    insert_tip
)

from app.config import DEBUG_MODE, LIVE_PREDICTION_INTERVAL


app = Flask(__name__)
scheduler = BackgroundScheduler(timezone="UTC")


# ---------------------------------------------------------
# HEALTH ENDPOINT
# ---------------------------------------------------------

@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "goalsniper-ai"})


# ---------------------------------------------------------
# INTERNAL: PROCESS A SINGLE FIXTURE
# ---------------------------------------------------------

def process_fixture(fixture_raw):
    """
    Takes raw fixture from API-Football, returns predictions.
    Also logs fixture → Supabase.
    """

    fixture = normalize_fixture(fixture_raw)
    fid = fixture["fixture_id"]

    # ---- SAVE FIXTURE SNAPSHOT ----
    upsert_fixture(fixture)

    # ---- FETCH LIVE STATS ----
    stats_raw = get_fixture_stats(fid)

    stats = {
        "home_shots_on": 0,
        "away_shots_on": 0,
        "home_attacks": 0,
        "away_attacks": 0,
    }

    if stats_raw:
        for team_stats in stats_raw:
            tid = team_stats.get("team", {}).get("id")
            items = team_stats.get("statistics", [])

            for s in items:
                t = s.get("type")
                v = s.get("value") or 0

                # Shots on Goal
                if t == "Shots on Goal":
                    if tid == fixture["home_team_id"]:
                        stats["home_shots_on"] = v
                    else:
                        stats["away_shots_on"] = v

                # Attacks
                if t == "Attacks":
                    if tid == fixture["home_team_id"]:
                        stats["home_attacks"] = v
                    else:
                        stats["away_attacks"] = v

    # ---- SAVE STATS SNAPSHOT ----
    insert_stats_snapshot(fid, stats)

    # ---- FETCH LIVE ODDS ----
    odds_raw = get_live_odds(fid)

    odds_1x2 = {"home": None, "draw": None, "away": None}
    odds_ou25 = {"over": None, "under": None}
    odds_btts = {"yes": None, "no": None}

    if odds_raw:
        for item in odds_raw:
            markets = item.get("bets", [])

            for m in markets:
                name = m.get("name")
                values = m.get("values", [])

                # 1X2
                if name == "Match Winner":
                    for v in values:
                        if v["value"] == "Home":
                            odds_1x2["home"] = float(v["odd"])
                        if v["value"] == "Draw":
                            odds_1x2["draw"] = float(v["odd"])
                        if v["value"] == "Away":
                            odds_1x2["away"] = float(v["odd"])

                # OU 2.5
                if name == "Over/Under" and "2.5" in str(values):
                    for v in values:
                        if v["value"] == "Over 2.5":
                            odds_ou25["over"] = float(v["odd"])
                        if v["value"] == "Under 2.5":
                            odds_ou25["under"] = float(v["odd"])

                # BTTS
                if name == "Both Teams To Score":
                    for v in values:
                        if v["value"] == "Yes":
                            odds_btts["yes"] = float(v["odd"])
                        if v["value"] == "No":
                            odds_btts["no"] = float(v["odd"])

    # ---- SAVE ODDS SNAPSHOT ----
    insert_odds_snapshot(fid, {
        "1x2": odds_1x2,
        "ou25": odds_ou25,
        "btts": odds_btts,
    })

    # ---- PREMATCH PLACEHOLDER (future: use Supabase team ratings) ----
    prematch_strength = {"home": 0.0, "away": 0.0}
    prematch_goal_exp = 2.6
    prematch_btts_exp = 0.55

    # ---- ML PREDICTIONS ----
    preds_1x2 = predict_fixture_1x2(fixture, stats, odds_1x2, prematch_strength)
    preds_ou25 = predict_fixture_ou25(fixture, stats, odds_ou25, prematch_goal_exp)
    preds_btts = predict_fixture_btts(fixture, stats, odds_btts, prematch_btts_exp)

    return preds_1x2 + preds_ou25 + preds_btts


# ---------------------------------------------------------
# MAIN LIVE CYCLE
# ---------------------------------------------------------

def live_prediction_cycle():

    print("[GOALSNIPER] Live scan triggered...")

    fixtures_raw = get_live_fixtures()
    all_preds = []

    for fr in fixtures_raw:
        preds = process_fixture(fr)
        all_preds.extend(preds)

    # Filter + adaptive filters
    final_preds = filter_predictions(all_preds)

    if not final_preds:
        print("[GOALSNIPER] No predictions passed filters.")
        return

    # Send to Telegram
    send_predictions(final_preds)

    # Save Tips
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

    print(f"[GOALSNIPER] Predictions sent and stored.")


# ---------------------------------------------------------
# START SCHEDULER
# ---------------------------------------------------------

def start_scheduler_once():
    if scheduler.running:
        return
    scheduler.add_job(live_prediction_cycle, "interval", seconds=LIVE_PREDICTION_INTERVAL)
    scheduler.start()
    print(f"[GOALSNIPER] Scheduler started ({LIVE_PREDICTION_INTERVAL}s interval)")


def start_async_scheduler():
    t = threading.Thread(target=start_scheduler_once)
    t.start()


start_async_scheduler()


# ---------------------------------------------------------
# FLASK ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=DEBUG_MODE)
