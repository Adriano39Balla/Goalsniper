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

# Supabase logging — FIXED IMPORTS
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
    run_full_training()
    return jsonify({"status": "training started"})


@app.route("/manual/scan")
def manual_scan():
    live_prediction_cycle()
    return jsonify({"status": "scan completed"})


# ============================================================
# ODDS PARSER (UNIVERSAL)
# ============================================================

def extract_core_odds(odds_list):
    """
    Extracts FT 1X2, BTTS, O/U 2.5 from the wild API-Football feed.
    """

    out_1x2 = {"home": None, "draw": None, "away": None}
    out_btts = {"yes": None, "no": None}
    out_ou25 = {"over": None, "under": None}

    if not odds_list:
        return out_1x2, out_ou25, out_btts

    for m in odds_list:
        name = m.get("name", "").lower()
        values = m.get("values", [])

        # ---------- FULLTIME 1X2 ----------
        if name in ["fulltime result", "match winner", "1x2"]:
            for v in values:
                label = v.get("value", "").lower()
                odd = v.get("odd")
                if odd is None:
                    continue
                if label == "home":
                    out_1x2["home"] = float(odd)
                elif label == "draw":
                    out_1x2["draw"] = float(odd)
                elif label == "away":
                    out_1x2["away"] = float(odd)

        # ---------- BTTS (Fulltime only) ----------
        if "both teams to score" in name and "half" not in name:
            for v in values:
                label = v.get("value", "").lower()
                odd = v.get("odd")
                if odd is None:
                    continue
                if label == "yes":
                    out_btts["yes"] = float(odd)
                elif label == "no":
                    out_btts["no"] = float(odd)

        # ---------- OVER/UNDER 2.5 ----------
        if "over/under" in name or "line" in name:
            for v in values:
                hcap = str(v.get("handicap", "")).strip()
                label = v.get("value", "").lower()
                odd = v.get("odd")
                if odd is None:
                    continue

                if hcap == "2.5":
                    if "over" in label:
                        out_ou25["over"] = float(odd)
                    if "under" in label:
                        out_ou25["under"] = float(odd)

    return out_1x2, out_ou25, out_btts


# ============================================================
# PROCESS FIXTURE
# ============================================================

def process_fixture(f_raw):
    fixture = normalize_fixture(f_raw)
    fid = fixture["fixture_id"]

    # Save fixture snapshot
    upsert_fixture(fixture)

    # ---------------- STATS ----------------
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

    insert_stats_snapshot(fid, stats)

    # ---------------- ODDS ----------------
    odds_resp = api_get("odds/live", {"fixture": fid})
    odds_data = odds_resp.get("response", [])
    odds_list = odds_data[0].get("odds", []) if odds_data else []

    odds_1x2, odds_ou25, odds_btts = extract_core_odds(odds_list)

    insert_odds_snapshot(fid, {
        "1x2": odds_1x2,
        "ou25": odds_ou25,
        "btts": odds_btts,
    })

    # ---------------- ML PREDICTIONS ----------------
    prematch_strength = {"home": 0.0, "away": 0.0}
    prematch_goal_exp = 2.6
    prematch_btts_exp = 0.55

    preds = []
    preds += predict_fixture_1x2(fixture, stats, odds_1x2, prematch_strength)
    preds += predict_fixture_ou25(fixture, stats, odds_ou25, prematch_goal_exp)
    preds += predict_fixture_btts(fixture, stats, odds_btts, prematch_btts_exp)

    return preds


# ============================================================
# LIVE SCAN LOOP
# ============================================================

def live_prediction_cycle():
    print("[GOALSNIPER] Live scan triggered...")

    fixtures_raw = get_live_fixtures()
    all_preds = []

    for f in fixtures_raw:
        all_preds.extend(process_fixture(f))

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

    print("[GOALSNIPER] Predictions sent & stored.")


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
# FLASK ENTRY POINT
# ============================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=DEBUG_MODE)
