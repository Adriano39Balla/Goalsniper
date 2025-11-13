import threading
from flask import Flask, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
from app.models import load_all_models
from app.train_models import run_full_training

from app.api_football import (
    get_live_fixtures,
    get_fixture_stats,
    get_live_odds,
    normalize_fixture,
)
from app.predictor import (
    predict_fixture_1x2,
    predict_fixture_ou25,
    predict_fixture_btts,
)
from app.filters import filter_predictions
from app.telegram_bot import send_predictions
from app.supabase_db import (
    upsert_fixture,
    insert_stats,
    insert_odds,
    insert_tip,
)
from app.config import DEBUG_MODE, LIVE_PREDICTION_INTERVAL


app = Flask(__name__)
scheduler = BackgroundScheduler(timezone="UTC")


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


# ---------------------------------------------------------
# HEALTH
# ---------------------------------------------------------

@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "goalsniper-ai"})

@app.route("/manual/scan", methods=["GET"])
def manual_scan():
    """
    Manually trigger a live scan without waiting for scheduler.
    """
    try:
        print("[MANUAL] Manual live scan triggered")
        live_prediction_cycle()
        return jsonify({"status": "ok", "message": "Live scan completed"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/manual/train", methods=["GET"])
def manual_training():
    """
    Manually retrain all ML models.
    """
    try:
        print("[MANUAL] Manual model training started")
        run_full_training()
        return jsonify({"status": "ok", "message": "Training completed"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/manual/reload-models", methods=["GET"])
def manual_reload_models():
    """
    Reload all model .pkl files without restarting the container.
    """
    try:
        load_all_models()
        print("[MANUAL] Models reloaded.")
        return jsonify({"status": "ok", "message": "Models reloaded"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ---------------------------------------------------------
# INTERNAL: PROCESS A SINGLE FIXTURE
# ---------------------------------------------------------

def process_fixture(fixture_raw):
    """
    1. Normalize fixture
    2. Save fixture to Supabase
    3. Fetch + log stats
    4. Fetch + log odds (robust)
    5. Run predictions
    """

    fixture = normalize_fixture(fixture_raw)
    fid = fixture["fixture_id"]

    # 1) Save fixture
    upsert_fixture(fixture)

    # =====================================================
    # 2) FETCH + SAVE STATS
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
            for s in team_stats.get("statistics", []):
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

    insert_stats(fid, stats)

    # =====================================================
    # 3) FETCH + PARSE + SAVE ODDS
    # =====================================================

    odds_raw = get_live_odds(fid)
    structured_odds = {}

    if odds_raw:
        # odds_raw shape is usually a list of fixtures with "bookmakers"
        for fixt_block in odds_raw:
            bookmakers = fixt_block.get("bookmakers") or [fixt_block]

            for bm in bookmakers:
                bookmaker_obj = bm.get("bookmaker", {}) or {}
                bookmaker = (
                    bookmaker_obj.get("name")
                    or str(bookmaker_obj.get("id") or bm.get("name") or bm.get("id"))
                )

                if not bookmaker:
                    continue

                if bookmaker not in structured_odds:
                    structured_odds[bookmaker] = {}

                bets = bm.get("bets", [])
                for bet in bets:
                    bet_name = (bet.get("name") or "").lower()
                    values = bet.get("values", []) or []

                    # ---------------- 1X2 / Match Winner ----------------
                    is_1x2 = any(
                        key in bet_name
                        for key in ["match winner", "1x2", "winner", "result", "3way", "3-way"]
                    )
                    if is_1x2:
                        out = structured_odds[bookmaker].get("1X2", {})
                        for v in values:
                            val = (v.get("value") or "").lower()
                            odd = safe_float(v.get("odd"))
                            if odd is None or odd <= 1.01:
                                continue

                            if val in ("home", "1", "1 (home)"):
                                out["HOME"] = odd
                            elif val in ("draw", "x"):
                                out["DRAW"] = odd
                            elif val in ("away", "2", "2 (away)"):
                                out["AWAY"] = odd

                        if out:
                            structured_odds[bookmaker]["1X2"] = out

                    # ---------------- BTTS (full-time only) -------------
                    is_btts = any(
                        key in bet_name
                        for key in ["both teams to score", "btts", "goal/no goal", "gg/ng", "gg"]
                    )
                    is_half = any(
                        k in bet_name
                        for k in ["1st half", "first half", "2nd half", "second half", "1h", "2h"]
                    )
                    is_extra = "extra time" in bet_name or "et" in bet_name

                    if is_btts and not (is_half or is_extra):
                        out = structured_odds[bookmaker].get("BTTS", {})
                        for v in values:
                            val = (v.get("value") or "").lower()
                            odd = safe_float(v.get("odd"))
                            if odd is None or odd <= 1.01:
                                continue

                            if "yes" in val or val == "1":
                                out["YES"] = odd
                            elif "no" in val or val == "2":
                                out["NO"] = odd

                        if out:
                            structured_odds[bookmaker]["BTTS"] = out

                    # ---------------- Over/Under 2.5 --------------------
                    is_ou = any(
                        key in bet_name
                        for key in ["over/under", "total goals", "goals over", "goals under", "o/u"]
                    )
                    if is_ou:
                        out = structured_odds[bookmaker].get("OU25", {})
                        for v in values:
                            val = (v.get("value") or "").lower()
                            odd = safe_float(v.get("odd"))
                            if odd is None or odd <= 1.01:
                                continue

                            # Normalize all "Over 2.5" variants
                            if "over 2.5" in val or val == "o 2.5" or "2.5 over" in val:
                                out["OVER"] = odd
                            # Normalize all "Under 2.5" variants
                            elif "under 2.5" in val or val == "u 2.5" or "2.5 under" in val:
                                out["UNDER"] = odd

                        if out:
                            structured_odds[bookmaker]["OU25"] = out

    # Save odds to Supabase
    insert_odds(fid, structured_odds)

    # =====================================================
    # 4) BUILD ODDS FOR PREDICTOR (pick a bookmaker)
    # =====================================================

    preferred_books = ["Bet365", "bet365", "Pinnacle", "bwin", "William Hill"]
    chosen_book = None

    for b in preferred_books + list(structured_odds.keys()):
        if b in structured_odds:
            chosen_book = b
            break

    odds_1x2 = {}
    odds_ou25 = {}
    odds_btts = {}

    if chosen_book:
        book_markets = structured_odds[chosen_book]
        odds_1x2 = {
            "home": book_markets.get("1X2", {}).get("HOME"),
            "draw": book_markets.get("1X2", {}).get("DRAW"),
            "away": book_markets.get("1X2", {}).get("AWAY"),
        }
        odds_ou25 = {
            "over": book_markets.get("OU25", {}).get("OVER"),
            "under": book_markets.get("OU25", {}).get("UNDER"),
        }
        odds_btts = {
            "yes": book_markets.get("BTTS", {}).get("YES"),
            "no": book_markets.get("BTTS", {}).get("NO"),
        }

    # =====================================================
    # 5) PREMATCH PLACEHOLDERS
    # =====================================================

    prematch_strength = {"home": 0.0, "away": 0.0}
    prematch_goal_exp = 2.6
    prematch_btts_exp = 0.55

    # =====================================================
    # 6) ML PREDICTIONS
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
# SCHEDULER
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
