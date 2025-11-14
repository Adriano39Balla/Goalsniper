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
    insert_stats,
    insert_odds,
    insert_tip,
)

# Manual training (train_models.py)
from train_models import run_full_training

# Config
from app.config import DEBUG_MODE, LIVE_PREDICTION_INTERVAL


app = Flask(__name__)
scheduler = BackgroundScheduler(timezone="UTC")


# ============================================================
# HEALTH CHECKS
# ============================================================

@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "goalsniper-ai"})


@app.route("/manual/health-full")
def health_full():
    return jsonify({
        "status": "ok",
        "models_ready": True,
        "scheduler_running": scheduler.running,
        "api_source": "api-football",
    })


# ============================================================
# MANUAL ENDPOINTS (SAFE)
# ============================================================

def _background_train():
    """Run full training in a background thread."""
    try:
        print("[TRAIN] Background training started...")
        run_full_training()
        print("[TRAIN] Background training finished.")
    except Exception as e:
        print("[TRAIN] Background training ERROR:", e)


@app.route("/manual/train")
def manual_train():
    """
    Manually trigger full model training.
    Runs in background to avoid HTTP timeout on Railway.
    """
    t = threading.Thread(target=_background_train, daemon=True)
    t.start()
    return jsonify({"status": "training started (background)"})


@app.route("/manual/scan")
def manual_scan():
    """
    Manually trigger a single live prediction scan.
    Wrapped in try/except so the HTTP request always returns.
    """
    try:
        live_prediction_cycle()
        return jsonify({"status": "scan completed"})
    except Exception as e:
        print("[SCAN] Manual scan ERROR:", e)
        return jsonify({"status": "error", "detail": str(e)}), 500


# ============================================================
# UNIVERSAL ODDS PARSER
# ============================================================

def extract_core_odds(odds_list):
    """
    Extract only the markets we need from API-Football live odds:

      - Fulltime 1X2
      - Fulltime BTTS
      - Fulltime Over/Under 2.5

    `odds_list` is a list like:
      [
        {
          "id": ...,
          "name": "Fulltime Result",
          "values": [...]
        },
        ...
      ]
    """

    out_1x2 = {"home": None, "draw": None, "away": None}
    out_btts = {"yes": None, "no": None}
    out_ou25 = {"over": None, "under": None}

    if not odds_list:
        return out_1x2, out_ou25, out_btts

    for m in odds_list:
        name = (m.get("name") or "").lower()
        values = m.get("values") or []

        # -------------- FULLTIME 1X2 --------------
        if name in ["fulltime result", "match winner", "1x2"]:
            for v in values:
                val = (v.get("value") or "").lower()
                odd = v.get("odd")
                if odd is None:
                    continue
                if val == "home":
                    out_1x2["home"] = float(odd)
                elif val == "draw":
                    out_1x2["draw"] = float(odd)
                elif val == "away":
                    out_1x2["away"] = float(odd)

        # -------------- BTTS FULLTIME --------------
        if "both teams to score" in name and "half" not in name:
            for v in values:
                val = (v.get("value") or "").lower()
                odd = v.get("odd")
                if odd is None:
                    continue
                if val == "yes":
                    out_btts["yes"] = float(odd)
                elif val == "no":
                    out_btts["no"] = float(odd)

        # -------------- O/U 2.5 --------------
        if "over/under" in name or "line" in name:
            for v in values:
                label = (v.get("value") or "").lower()
                handicap = str(v.get("handicap") or "").strip()
                odd = v.get("odd")
                if odd is None:
                    continue

                if handicap == "2.5":
                    if "over" in label:
                        out_ou25["over"] = float(odd)
                    if "under" in label:
                        out_ou25["under"] = float(odd)

    return out_1x2, out_ou25, out_btts


# ============================================================
# PROCESS A SINGLE FIXTURE
# ============================================================

def process_fixture(f_raw):
    """
    Take raw fixture from API-Football, log it, fetch stats+odds,
    build features, run predictions, and return a list[Prediction].
    """

    try:
        fixture = normalize_fixture(f_raw)
        fid = fixture["fixture_id"]

        # ------------------- FIXTURE SNAPSHOT -------------------
        upsert_fixture(fixture)

        # ------------------- STATS SNAPSHOT ---------------------
        stats_row = {
            "minute": fixture.get("minute", 0),
            "home_shots_on": 0,
            "away_shots_on": 0,
            "home_attacks": 0,
            "away_attacks": 0,
        }

        stats_raw = get_fixture_stats(fid)
        if stats_raw:
            for team in stats_raw:
                tid = team.get("team", {}).get("id")
                for s in team.get("statistics", []):
                    t = s.get("type")
                    v = s.get("value") or 0

                    if t == "Shots on Goal":
                        if tid == fixture["home_team_id"]:
                            stats_row["home_shots_on"] = v
                        else:
                            stats_row["away_shots_on"] = v

                    if t == "Attacks":
                        if tid == fixture["home_team_id"]:
                            stats_row["home_attacks"] = v
                        else:
                            stats_row["away_attacks"] = v

        insert_stats(fid, stats_row)

        # ------------------- ODDS SNAPSHOT ----------------------
        # api_get returns the "response" array directly → list
        odds_rows = api_get("odds/live", {"fixture": fid})

        if odds_rows and isinstance(odds_rows, list):
            first = odds_rows[0]
            odds_list = first.get("odds", []) if isinstance(first, dict) else []
        else:
            odds_list = []

        odds_1x2, odds_ou25, odds_btts = extract_core_odds(odds_list)

        insert_odds(fid, {
            "1x2": odds_1x2,
            "ou25": odds_ou25,
            "btts": odds_btts,
        })

        # ------------------- ML PREDICTIONS ---------------------
        prematch_strength = {"home": 0.0, "away": 0.0}
        prematch_goal_exp = 2.6
        prematch_btts_exp = 0.55

        preds = []
        preds += predict_fixture_1x2(fixture, stats_row, odds_1x2, prematch_strength)
        preds += predict_fixture_ou25(fixture, stats_row, odds_ou25, prematch_goal_exp)
        preds += predict_fixture_btts(fixture, stats_row, odds_btts, prematch_btts_exp)

        return preds

    except Exception as e:
        print(f"[PROCESS] Error processing fixture: {e}")
        return []


# ============================================================
# MAIN LIVE CYCLE
# ============================================================

def live_prediction_cycle():
    print("[GOALSNIPER] Live scan triggered...")

    try:
        fixtures_raw = get_live_fixtures()
    except Exception as e:
        print("[GOALSNIPER] Error fetching live fixtures:", e)
        return

    all_preds = []

    for fr in fixtures_raw:
        preds = process_fixture(fr)
        if preds:
            all_preds.extend(preds)

    final_preds = filter_predictions(all_preds)

    if not final_preds:
        print("[GOALSNIPER] No predictions passed filters.")
        return

    # Send to Telegram
    send_predictions(final_preds)

    # Store predictions in Supabase
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

    print(f"[GOALSNIPER] Predictions sent & saved ({len(final_preds)} bets).")


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
    t = threading.Thread(target=start_scheduler_once, daemon=True)
    t.start()


start_async_scheduler()


# ============================================================
# FLASK ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=DEBUG_MODE)
