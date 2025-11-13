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

# Supabase (NEW CLEAN INTERFACE)
from app.supabase_db import (
    upsert_fixture,
    insert_stats,
    insert_odds,
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
    1. Normalize fixture
    2. Save fixture to Supabase
    3. Fetch + log stats
    4. Fetch + log odds
    5. Run predictions
    """

    fixture = normalize_fixture(fixture_raw)
    fid = fixture["fixture_id"]

    # ---- SAVE FIXTURE ----
    upsert_fixture(fixture)

    # =====================================================
    # FETCH STATS
    # =====================================================

    stats_raw = get_fixture_stats(fid)

    stats = {
        "minute": fixture.get("minute"),
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

    # ---- SAVE STATS ----
    insert_stats(fid, stats)

    # =====================================================
    # FETCH ODDS
    # =====================================================

    odds_raw = get_live_odds(fid)

    odds_1x2 = {"HOME": None, "DRAW": None, "AWAY": None}
    odds_ou25 = {"OVER": None, "UNDER": None}
    odds_btts = {"YES": None, "NO": None}

    structured_odds = {}

    if odds_raw:
        for item in odds_raw:
            bookmaker = item.get("bookmaker")
            markets = item.get("bets", [])

            if bookmaker not in structured_odds:
                structured_odds[bookmaker] = {}

            for m in markets:
                name = m.get("name")
                values = m.get("values", [])

                # 1X2
                if name == "Match Winner":
                    for v in values:
                        if v["value"] == "Home":
                            odds_1x2["HOME"] = float(v["odd"])
                        if v["value"] == "Draw":
                            odds_1x2["DRAW"] = float(v["odd"])
                        if v["value"] == "Away":
                            odds_1x2["AWAY"] = float(v["odd"])

                    structured_odds[bookmaker]["1X2"] = odds_1x2

                # Over/Under 2.5
                if name == "Over/Under" and "2.5" in str(values):
                    for v in values:
                        if v["value"] == "Over 2.5":
                            odds_ou25["OVER"] = float(v["odd"])
                        if v["value"] == "Under 2.5":
                            odds_ou25["UNDER"] = float(v["odd"])

                    structured_odds[bookmaker]["OU_2.5"] = odds_ou25

                # BTTS
                if name == "Both Teams To Score":
                    for v in values:
                        if v["value"] == "Yes":
                            odds_btts["YES"] = float(v["odd"])
                        if v["value"] == "No":
                            odds_btts["NO"] = float(v["odd"])

                    structured_odds[bookmaker]["BTTS"] = odds_btts

    # ---- SAVE ODDS ----
    insert_odds(fid, structured_odds)

    # =====================================================
    # PREMATCH PLACEHOLDERS
    # =====================================================

    prematch_strength = {"home": 0.0, "away": 0.0}
    prematch_goal_exp = 2.6
    prematch_btts_exp = 0.55

    # =====================================================
    # ML PREDICTIONS
    # =====================================================

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

    # FILTER
    final_preds = filter_predictions(all_preds)

    if not final_preds:
        print("[GOALSNIPER] No predictions passed filters.")
        return

    # SEND TO TELEGRAM
    send_predictions(final_preds)

    # SAVE TIPS
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
