import threading
import time
from flask import Flask, jsonify
from apscheduler.schedulers.background import BackgroundScheduler

# Internal modules
from app.api_football import (
    get_live_fixtures,
    get_fixture_stats,
    get_live_odds,
    normalize_fixture
)
from app.predictor import (
    predict_fixture_1x2,
    predict_fixture_ou25,
    predict_fixture_btts
)
from app.filters import filter_predictions
from app.telegram_bot import send_predictions
from app.supabase_db import insert_tip, fetch_training_fixtures
from app.config import (
    DEBUG_MODE,
    LIVE_PREDICTION_INTERVAL,
)


app = Flask(__name__)
scheduler = BackgroundScheduler(timezone="UTC")


# ---------------------------------------------------------
# HEALTH CHECK ENDPOINT
# ---------------------------------------------------------

@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "goalsniper-ai"})


# ---------------------------------------------------------
# INTERNAL: PROCESS A SINGLE FIXTURE
# ---------------------------------------------------------

def process_fixture(fixture_raw):
    """
    Takes a raw fixture object from API-Football, returns predictions (list).
    """

    fixture = normalize_fixture(fixture_raw)

    fid = fixture["fixture_id"]
    minute = fixture.get("minute")

    # Fetch stats & odds for this fixture
    stats_raw = get_fixture_stats(fid)
    odds_raw = get_live_odds(fid)

    # Convert API-Football stats into simple dicts
    stats = {
        "home_shots_on": 0,
        "away_shots_on": 0,
        "home_attacks": 0,
        "away_attacks": 0,
    }

    # Extract stats if available
    if stats_raw:
        for team_stats in stats_raw:
            team = team_stats.get("team", {}).get("name", "")
            items = team_stats.get("statistics", [])

            for s in items:
                t = s.get("type")
                v = s.get("value") or 0

                if t == "Shots on Goal" and team_stats.get("team", {}).get("id") == fixture["home_team_id"]:
                    stats["home_shots_on"] = v
                elif t == "Shots on Goal":
                    stats["away_shots_on"] = v

                elif t == "Attacks" and team_stats.get("team", {}).get("id") == fixture["home_team_id"]:
                    stats["home_attacks"] = v
                elif t == "Attacks":
                    stats["away_attacks"] = v

    # Odds normalization
    odds_1x2 = {"home": None, "draw": None, "away": None}
    odds_ou25 = {"over": None, "under": None}
    odds_btts = {"yes": None, "no": None}

    if odds_raw:
        for item in odds_raw:
            book = item.get("bookmaker")
            markets = item.get("bets", [])

            for m in markets:
                market_name = m.get("name")
                values = m.get("values", [])

                # 1X2
                if market_name == "Match Winner":
                    for v in values:
                        if v["value"] == "Home":
                            odds_1x2["home"] = float(v["odd"])
                        elif v["value"] == "Draw":
                            odds_1x2["draw"] = float(v["odd"])
                        elif v["value"] == "Away":
                            odds_1x2["away"] = float(v["odd"])

                # Over/Under 2.5
                if market_name == "Over/Under" and "2.5" in str(values):
                    for v in values:
                        if v["value"] == "Over 2.5":
                            odds_ou25["over"] = float(v["odd"])
                        if v["value"] == "Under 2.5":
                            odds_ou25["under"] = float(v["odd"])

                # BTTS
                if market_name == "Both Teams To Score":
                    for v in values:
                        if v["value"] == "Yes":
                            odds_btts["yes"] = float(v["odd"])
                        if v["value"] == "No":
                            odds_btts["no"] = float(v["odd"])

    # Prematch placeholders (will come from DB later)
    prematch_strength = {"home": 0.0, "away": 0.0}
    prematch_goal_exp = 2.6
    prematch_btts_exp = 0.55

    # Make predictions using the ML engine
    preds_1x2 = predict_fixture_1x2(fixture, stats, odds_1x2, prematch_strength)
    preds_ou25 = predict_fixture_ou25(fixture, stats, odds_ou25, prematch_goal_exp)
    preds_btts = predict_fixture_btts(fixture, stats, odds_btts, prematch_btts_exp)

    return preds_1x2 + preds_ou25 + preds_btts


# ---------------------------------------------------------
# MAIN SCAN LOOP (CALLED EVERY X SECONDS)
# ---------------------------------------------------------

def live_prediction_cycle():

    print("[GOALSNIPER] Live scan triggered...")

    fixtures_raw = get_live_fixtures()
    all_preds = []

    for fr in fixtures_raw:
        preds = process_fixture(fr)
        all_preds.extend(preds)

    # Filter predictions (adaptive)
    final_preds = filter_predictions(all_preds)

    if not final_preds:
        print("[GOALSNIPER] No predictions passed filters.")
        return

    # Send predictions to telegram
    sent = send_predictions(final_preds)

    # Store in Supabase
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

    print(f"[GOALSNIPER] Predictions sent: {sent}")


# ---------------------------------------------------------
# START THE SCHEDULER
# ---------------------------------------------------------

def start_scheduler_once():
    if scheduler.running:
        return
    scheduler.add_job(live_prediction_cycle, "interval", seconds=LIVE_PREDICTION_INTERVAL)
    scheduler.start()
    print(f"[GOALSNIPER] Scheduler started ({LIVE_PREDICTION_INTERVAL}s interval)")


# ---------------------------------------------------------
# FLASK STARTUP
# ---------------------------------------------------------

def start_async_scheduler():
    """Run scheduler in separate thread so Flask can serve."""
    t = threading.Thread(target=start_scheduler_once)
    t.start()


start_async_scheduler()


# ---------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=DEBUG_MODE)
