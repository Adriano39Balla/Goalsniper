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

# Supabase
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
# HEALTH
# ---------------------------------------------------------

@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# ---------------------------------------------------------
# PROCESS FIXTURE
# ---------------------------------------------------------

def process_fixture(fixture_raw):

    fixture = normalize_fixture(fixture_raw)
    fid = fixture["fixture_id"]

    # Save fixture
    upsert_fixture(fixture)

    # =====================================================
    # FETCH + SAVE STATS
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
        for t in stats_raw:
            tid = t.get("team", {}).get("id")
            for item in t.get("statistics", []):
                tname = item.get("type")
                val = item.get("value") or 0

                if tname == "Shots on Goal":
                    if tid == fixture["home_team_id"]:
                        stats["home_shots_on"] = val
                    else:
                        stats["away_shots_on"] = val

                if tname == "Attacks":
                    if tid == fixture["home_team_id"]:
                        stats["home_attacks"] = val
                    else:
                        stats["away_attacks"] = val

    insert_stats(fid, stats)

    # =====================================================
    # FETCH + SAVE ODDS
    # =====================================================

    odds_raw = get_live_odds(fid)
    structured_odds = {}

    if odds_raw:
        for item in odds_raw:
            bookmaker_obj = item.get("bookmaker", {})
            bookmaker = bookmaker_obj.get("name") or bookmaker_obj.get("id")

            if not bookmaker:
                continue

            if bookmaker not in structured_odds:
                structured_odds[bookmaker] = {}

            for bet in item.get("bets", []):
                name = bet.get("name")

                # MATCH WINNER
                if name == "Match Winner":
                    out = {}
                    for v in bet.get("values", []):
                        if v["value"] == "Home":
                            out["HOME"] = float(v["odd"])
                        if v["value"] == "Draw":
                            out["DRAW"] = float(v["odd"])
                        if v["value"] == "Away":
                            out["AWAY"] = float(v["odd"])
                    structured_odds[bookmaker]["1X2"] = out

                # BTTS
                if name == "Both Teams To Score":
                    out = {}
                    for v in bet.get("values", []):
                        if v["value"] == "Yes":
                            out["YES"] = float(v["odd"])
                        if v["value"] == "No":
                            out["NO"] = float(v["odd"])
                    structured_odds[bookmaker]["BTTS"] = out

                # O/U ANY LINE
                if name == "Over/Under":
                    out = {}
                    for v in bet.get("values", []):
                        if "Over" in v["value"]:
                            out["OVER"] = float(v["odd"])
                        if "Under" in v["value"]:
                            out["UNDER"] = float(v["odd"])
                    structured_odds[bookmaker]["OU"] = out

    # Save to DB
    insert_odds(fid, structured_odds)

    # =====================================================
    # RUN PREDICTIONS
    # =====================================================

    # pick ANY bookmaker (priority: bet365)
    for bookmaker in ["bet365", "Bet365", "bwin", "pinnacle", *structured_odds.keys()]:
        if bookmaker in structured_odds:
            m = structured_odds[bookmaker]
            odds_1x2 = m.get("1X2", {})
            odds_ou = m.get("OU", {})
            odds_btts = m.get("BTTS", {})
            break
    else:
        odds_1x2, odds_ou, odds_btts = {}, {}, {}

    prematch_strength = {"home": 0.0, "away": 0.0}
    prematch_goal_exp = 2.6
    prematch_btts_exp = 0.55

    preds_1x2 = predict_fixture_1x2(fixture, stats, odds_1x2, prematch_strength)
    preds_ou25 = predict_fixture_ou25(fixture, stats, odds_ou, prematch_goal_exp)
    preds_btts = predict_fixture_btts(fixture, stats, odds_btts, prematch_btts_exp)

    return preds_1x2 + preds_ou25 + preds_btts


# ---------------------------------------------------------
# LIVE PREDICTION LOOP
# ---------------------------------------------------------

def live_prediction_cycle():

    print("[GOALSNIPER] Live scan triggered...")

    fixtures_raw = get_live_fixtures()
    all_preds = []

    for fr in fixtures_raw:
        preds = process_fixture(fr)
        all_preds.extend(preds)

    final_preds = filter_predictions(all_preds)

    if not final_preds:
        print("[GOALSNIPER] No predictions passed filters.")
        return

    send_predictions(final_preds)

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

    print("[GOALSNIPER] Predictions sent and stored.")


# ---------------------------------------------------------
# STARTUP
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=DEBUG_MODE)
