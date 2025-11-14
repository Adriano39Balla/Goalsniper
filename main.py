import threading
import traceback
import sys
import os
from datetime import datetime
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

# Training engine
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
    return jsonify({
        "status": "ok",
        "service": "goalsniper-ai",
        "timestamp": datetime.utcnow().isoformat()
    })


@app.route("/manual/health-full")
def health_full():
    try:
        return jsonify({
            "status": "ok",
            "timestamp": datetime.utcnow().isoformat(),
            "scheduler_running": scheduler.running,
            "debug_mode": DEBUG_MODE,
            "interval_seconds": LIVE_PREDICTION_INTERVAL,
            "python_version": sys.version,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ============================================================
# MANUAL TRAINING (ASYNC SAFE)
# ============================================================

def _async_train():
    try:
        print(f"[TRAIN] Async training started {datetime.utcnow().isoformat()}")
        run_full_training()
        print(f"[TRAIN] Async training finished {datetime.utcnow().isoformat()}")
    except Exception as e:
        print(f"[TRAIN ERROR] {e}")
        traceback.print_exc()


@app.route("/manual/train")
def manual_train():
    try:
        threading.Thread(target=_async_train, daemon=True).start()
        return jsonify({
            "status": "training started (async)",
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ============================================================
# MANUAL SCAN
# ============================================================

@app.route("/manual/scan")
def manual_scan():
    try:
        threading.Thread(target=live_prediction_cycle, daemon=True).start()
        return jsonify({
            "status": "manual scan triggered",
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


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

        # --- 1X2 ---
        if name in ["fulltime result", "match winner", "1x2"]:
            for v in values:
                val = (v.get("value") or "").lower()
                odd = v.get("odd")
                if odd is None:
                    continue

                if val == "home": out_1x2["home"] = float(odd)
                elif val == "draw": out_1x2["draw"] = float(odd)
                elif val == "away": out_1x2["away"] = float(odd)

        # --- BTTS ---
        if "both teams to score" in name and "half" not in name:
            for v in values:
                val = (v.get("value") or "").lower()
                odd = v.get("odd")
                if odd is None:
                    continue
                if val == "yes": out_btts["yes"] = float(odd)
                if val == "no": out_btts["no"] = float(odd)

        # --- Over/Under 2.5 ---
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

    # Write fixture snapshot (now with timestamp conversion)
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

    if isinstance(odds_api, dict):
        odds_resp = odds_api.get("response") or []
    elif isinstance(odds_api, list) and odds_api and isinstance(odds_api[0], dict):
        odds_resp = odds_api[0].get("response") or []
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
    preds = []
    preds += predict_fixture_1x2(fixture, stats, odds_1x2, {"home": 0.0, "away": 0.0})
    preds += predict_fixture_ou25(fixture, stats, odds_ou25, 2.6)
    preds += predict_fixture_btts(fixture, stats, odds_btts, 0.55)

    return preds


# ============================================================
# MAIN LIVE CYCLE
# ============================================================

def live_prediction_cycle():
    print(f"[GOALSNIPER] Live scan started {datetime.utcnow().isoformat()}")

    try:
        fixtures = get_live_fixtures()
        if not fixtures:
            print("[GOALSNIPER] No live fixtures.")
            return

        all_preds = []
        for fr in fixtures:
            try:
                all_preds.extend(process_fixture(fr))
            except Exception as e:
                print("[ERROR] processing fixture:", e)
                traceback.print_exc()

        final_preds = filter_predictions(all_preds)

        if not final_preds:
            print("[GOALSNIPER] No predictions passed filters.")
            return

        send_predictions(final_preds)

        for p in final_preds:
            try:
                insert_tip({
                    "fixture_id": p.fixture_id,
                    "market": p.market,
                    "selection": p.selection,
                    "prob": p.prob,
                    "odds": p.odds,
                    "ev": p.ev,
                    "minute": p.aux.get("minute"),
                })
            except Exception as e:
                print("[ERROR] storing tip:", e)

        print(f"[GOALSNIPER] Live scan completed.")

    except Exception as e:
        print("[GOALSNIPER ERROR]", e)
        traceback.print_exc()


# ============================================================
# SCHEDULER
# ============================================================

def start_scheduler_once():
    if not scheduler.running:
        scheduler.add_job(live_prediction_cycle, "interval", seconds=LIVE_PREDICTION_INTERVAL)
        scheduler.start()
        print(f"[SCHEDULER] Started interval={LIVE_PREDICTION_INTERVAL}s")


threading.Thread(target=start_scheduler_once, daemon=True).start()


# ============================================================
# FLASK ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting Goalsniper AI on port {port}")

    start_scheduler_once()

    app.run(host="0.0.0.0", port=port, debug=DEBUG_MODE)
