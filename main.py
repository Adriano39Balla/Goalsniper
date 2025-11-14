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

# Training engine  (IMPORTANT: must match file location)
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
        "scheduler_running": scheduler.running,
        "models_loaded": True,
        "api": "api-football",
    })


# ============================================================
# MANUAL TRAINING — NON BLOCKING
# ============================================================

def _async_train():
    """Run full model training in a background thread."""
    print("[TRAIN] Async training started...")
    run_full_training()
    print("[TRAIN] Async training finished.")


@app.route("/manual/train")
def manual_train():
    """Triggers full training without blocking the web request."""
    threading.Thread(target=_async_train).start()
    return jsonify({"status": "training started (async)"})


# ============================================================
# MANUAL SCAN
# ============================================================

@app.route("/manual/scan")
def manual_scan():
    threading.Thread(target=live_prediction_cycle).start()
    return jsonify({"status": "manual scan triggered"})


# ============================================================
# UNIVERSAL ODDS PARSER
# ============================================================

def extract_core_odds(odds_list):
    out_1x2 = {"home": None, "draw": None, "away": None}
    out_btts = {"yes": None, "no": None}
    out_ou25 = {"over": None, "under": None}

    if not odds_list:
        return out_1x2, out_ou25, out_btts

    for m in odds_list:
        name = (m.get("name") or "").lower()
        values = m.get("values") or []

        # Fulltime 1X2
        if name in ["fulltime result", "match winner", "1x2"]:
            for v in values:
                val = (v.get("value") or "").lower()
                odd = v.get("odd")
                if odd is None:
                    continue
                if val == "home": out_1x2["home"] = float(odd)
                if val == "draw": out_1x2["draw"] = float(odd)
                if val == "away": out_1x2["away"] = float(odd)

        # BTTS
        if "both teams to score" in name and "half" not in name:
            for v in values:
                val = (v.get("value") or "").lower()
                odd = v.get("odd")
                if odd is None:
                    continue
                if val == "yes": out_btts["yes"] = float(odd)
                if val == "no": out_btts["no"] = float(odd)

        # Over/Under Line
        if "over/under" in name or "line" in name:
            for v in values:
                handicap = str(v.get("handicap") or "").strip()
                label = (v.get("value") or "").lower()
                odd = v.get("odd")
                if odd is None:
                    continue
                if handicap == "2.5":
                    if "over" in label: out_ou25["over"] = float(odd)
                    if "under" in label: out_ou25["under"] = float(odd)

    return out_1x2, out_ou25, out_btts


# ============================================================
# PROCESS FIXTURE
# ============================================================

def process_fixture(f_raw):
    fixture = normalize_fixture(f_raw)
    fid = fixture["fixture_id"]

    upsert_fixture(fixture)

    # --- Stats ---
    stats = {
        "minute": fixture.get("minute") or 0,
        "home_shots_on": 0,
        "away_shots_on": 0,
        "home_attacks": 0,
        "away_attacks": 0,
    }

    stats_raw = get_fixture_stats(fid) or []
    for team in stats_raw:
        tid = team.get("team", {}).get("id")
        for s in team.get("statistics", []):
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

    # --- Odds ---
    odds_api = api_get("odds/live", {"fixture": fid})

    # Normalize dict/list response
    if isinstance(odds_api, dict):
        odds_resp = odds_api.get("response") or []
    elif isinstance(odds_api, list):
        odds_resp = odds_api[0].get("response") if odds_api and isinstance(odds_api[0], dict) else []
    else:
        odds_resp = []

    odds_list = odds_resp[0].get("odds") if odds_resp else []
    odds_1x2, odds_ou25, odds_btts = extract_core_odds(odds_list)

    insert_odds(fid, {
        "1x2": odds_1x2,
        "ou25": odds_ou25,
        "btts": odds_btts,
    })

    # --- Predictions ---
    prematch_strength = {"home": 0.0, "away": 0.0}

    preds = []
    preds += predict_fixture_1x2(fixture, stats, odds_1x2, prematch_strength)
    preds += predict_fixture_ou25(fixture, stats, odds_ou25, 2.6)
    preds += predict_fixture_btts(fixture, stats, odds_btts, 0.55)

    return preds


# ============================================================
# MAIN LIVE CYCLE
# ============================================================

def live_prediction_cycle():
    print("[GOALSNIPER] Live scan running...")

    fixtures = get_live_fixtures()
    all_preds = []

    for fr in fixtures:
        all_preds.extend(process_fixture(fr))

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


# ============================================================
# SCHEDULER
# ============================================================

def start_scheduler_once():
    if scheduler.running:
        return
    scheduler.add_job(live_prediction_cycle, "interval", seconds=LIVE_PREDICTION_INTERVAL)
    scheduler.start()


threading.Thread(target=start_scheduler_once).start()


# ============================================================
# Flask entry
# ============================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=DEBUG_MODE)
