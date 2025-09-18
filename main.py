# file: main.py
# goalsniper AI – Opta Supercomputer style (entrypoint)

import os, time, logging
from flask import Flask, jsonify, request, abort
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from zoneinfo import ZoneInfo

from db import init_db, db_conn
from telegram_utils import send_telegram
from scan import (
    production_scan, prematch_scan_save, send_match_of_the_day,
    backfill_results_for_open_matches, daily_accuracy_digest, retry_unsent_tips
)
from train_models import train_models, auto_tune_thresholds

# ───────── Logging ─────────
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
log = logging.getLogger("goalsniper")
app = Flask(__name__)

# ───────── Env ─────────
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")
RUN_SCHEDULER = os.getenv("RUN_SCHEDULER", "1") not in ("0","false","False","no","NO")
SCAN_INTERVAL_SEC = int(os.getenv("SCAN_INTERVAL_SEC", "300"))
BACKFILL_EVERY_MIN = int(os.getenv("BACKFILL_EVERY_MIN", "15"))
TRAIN_ENABLE = os.getenv("TRAIN_ENABLE", "1") not in ("0","false","False","no","NO")
TRAIN_HOUR_UTC = int(os.getenv("TRAIN_HOUR_UTC", "2"))
TRAIN_MINUTE_UTC = int(os.getenv("TRAIN_MINUTE_UTC", "12"))
AUTO_TUNE_ENABLE = os.getenv("AUTO_TUNE_ENABLE", "0") not in ("0","false","False","no","NO"))
DAILY_ACCURACY_DIGEST_ENABLE = os.getenv("DAILY_ACCURACY_DIGEST_ENABLE", "1") not in ("0","false","False","no","NO")
DAILY_ACCURACY_HOUR = int(os.getenv("DAILY_ACCURACY_HOUR", "8"))
DAILY_ACCURACY_MINUTE = int(os.getenv("DAILY_ACCURACY_MINUTE", "0"))
MOTD_ENABLE = os.getenv("MOTD_ENABLE", "1") not in ("0","false","False","no","NO")
MOTD_HOUR = int(os.getenv("MOTD_HOUR", "10"))
MOTD_MINUTE = int(os.getenv("MOTD_MINUTE", "0"))

TZ_UTC = ZoneInfo("UTC")
BERLIN_TZ = ZoneInfo("Europe/Berlin")

# Nightly cooldown (Berlin) — default 23:00–07:00
REST_START_HOUR_BERLIN = int(os.getenv("REST_START_HOUR_BERLIN", "23"))  # inclusive
REST_END_HOUR_BERLIN   = int(os.getenv("REST_END_HOUR_BERLIN", "7"))     # exclusive

def is_rest_window_now() -> bool:
    """Return True if current Berlin local time is within the nightly rest window."""
    now = time.localtime()  # not TZ aware; we'll use ZoneInfo for Berlin hour
    h = int(__import__("datetime").datetime.now(BERLIN_TZ).hour)
    if REST_START_HOUR_BERLIN <= REST_END_HOUR_BERLIN:
        return REST_START_HOUR_BERLIN <= h < REST_END_HOUR_BERLIN
    # wrap across midnight (e.g., 23..24 & 0..6)
    return (h >= REST_START_HOUR_BERLIN) or (h < REST_END_HOUR_BERLIN)

# ───────── Admin auth ─────────
def _require_admin():
    key = request.headers.get("X-API-Key") or request.args.get("key") or (
        (request.json or {}).get("key") if request.is_json else None
    )
    if not ADMIN_API_KEY or key != ADMIN_API_KEY:
        abort(401)

# ───────── Scheduler ─────────
_scheduler_started = False

def _run_with_pg_lock(lock_key: int, fn, *a, **k):
    try:
        with db_conn() as c:
            got = c.execute("SELECT pg_try_advisory_lock(%s)", (lock_key,)).fetchone()[0]
            if not got:
                log.info("[LOCK %s] busy; skipped.", lock_key)
                return None
            try:
                return fn(*a, **k)
            finally:
                c.execute("SELECT pg_advisory_unlock(%s)", (lock_key,))
    except Exception as e:
        log.exception("[LOCK %s] failed: %s", lock_key, e)
        return None

def _maybe_skip_rest(fn_name: str):
    """Guard that turns scan jobs into no-ops in the Berlin nightly rest window."""
    if is_rest_window_now():
        log.info("[%s] rest window active (%02d:00–%02d:00 Berlin) — skipped.",
                 fn_name, REST_START_HOUR_BERLIN, REST_END_HOUR_BERLIN)
        return True
    return False

def _start_scheduler_once():
    global _scheduler_started
    if _scheduler_started or not RUN_SCHEDULER:
        return
    try:
        sched = BackgroundScheduler(timezone=TZ_UTC)

        # live in-play scanning (no-op during rest window)
        def _scan_job():
            if _maybe_skip_rest("scan"):
                return
            return _run_with_pg_lock(1001, production_scan)
        sched.add_job(_scan_job, "interval",
                      seconds=SCAN_INTERVAL_SEC, id="scan", max_instances=1, coalesce=True)

        # backfill results (can run at night — cheap and non-live)
        sched.add_job(lambda: _run_with_pg_lock(1002, backfill_results_for_open_matches, 400),
                      "interval", minutes=BACKFILL_EVERY_MIN, id="backfill", max_instances=1, coalesce=True)

        # daily digest
        if DAILY_ACCURACY_DIGEST_ENABLE:
            sched.add_job(lambda: _run_with_pg_lock(1003, daily_accuracy_digest),
                          CronTrigger(hour=DAILY_ACCURACY_HOUR, minute=DAILY_ACCURACY_MINUTE, timezone=BERLIN_TZ),
                          id="digest", max_instances=1, coalesce=True)

        # match of the day (prematch; allowed any time, but usually morning)
        if MOTD_ENABLE:
            sched.add_job(lambda: _run_with_pg_lock(1004, send_match_of_the_day),
                          CronTrigger(hour=MOTD_HOUR, minute=MOTD_MINUTE, timezone=BERLIN_TZ),
                          id="motd", max_instances=1, coalesce=True)

        # training (runs at fixed UTC time)
        if TRAIN_ENABLE:
            sched.add_job(lambda: _run_with_pg_lock(1005, train_models),
                          CronTrigger(hour=TRAIN_HOUR_UTC, minute=TRAIN_MINUTE_UTC, timezone=TZ_UTC),
                          id="train", max_instances=1, coalesce=True)

        # auto-tune thresholds (UTC; cheap)
        if AUTO_TUNE_ENABLE:
            sched.add_job(lambda: _run_with_pg_lock(1006, auto_tune_thresholds, 14),
                          CronTrigger(hour=4, minute=7, timezone=TZ_UTC),
                          id="auto_tune", max_instances=1, coalesce=True)

        # retry unsent (also cheap; allowed at night)
        sched.add_job(lambda: _run_with_pg_lock(1007, retry_unsent_tips, 30, 200),
                      "interval", minutes=10, id="retry", max_instances=1, coalesce=True)

        sched.start()
        _scheduler_started = True
        send_telegram("🚀 goalsniper AI live and scanning (night rest 23:00–07:00 Berlin)")
        log.info("[SCHED] started (scan=%ss)", SCAN_INTERVAL_SEC)

    except Exception as e:
        log.exception("[SCHED] failed: %s", e)

# ───────── Routes ─────────
@app.route("/")
def root():
    return jsonify({"ok": True, "name": "goalsniper", "mode": "FULL_AI", "scheduler": RUN_SCHEDULER})

@app.route("/health")
def health():
    try:
        with db_conn() as c:
            n = c.execute("SELECT COUNT(*) FROM tips").fetchone()[0]
        return jsonify({"ok": True, "db": "ok", "tips_count": int(n)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/stats")
def stats():
    try:
        now = int(time.time())
        day_ago = now - 24 * 3600
        week_ago = now - 7 * 24 * 3600

        with db_conn() as c:
            (t24,) = c.execute(
                "SELECT COUNT(*) FROM tips WHERE created_ts >= %s AND suggestion <> 'HARVEST'",
                (day_ago,)
            ).fetchone()
            (t7d,) = c.execute(
                "SELECT COUNT(*) FROM tips WHERE created_ts >= %s AND suggestion <> 'HARVEST'",
                (week_ago,)
            ).fetchone()
            (unsent,) = c.execute(
                "SELECT COUNT(*) FROM tips WHERE sent_ok=0 AND created_ts >= %s",
                (week_ago,)
            ).fetchone()

            rows = c.execute("""
                SELECT t.suggestion, t.odds, r.final_goals_h, r.final_goals_a, r.btts_yes
                FROM tips t
                JOIN match_results r ON r.match_id = t.match_id
                WHERE t.created_ts >= %s
                  AND t.suggestion <> 'HARVEST'
                  AND t.sent_ok = 1
            """, (week_ago,)).fetchall()

        graded = wins = 0
        stake = pnl = 0.0

        for (sugg, odds, gh, ga, btts) in rows:
            gh = int(gh or 0)
            ga = int(ga or 0)
            total = gh + ga
            result = None

            if sugg.startswith("Over") or sugg.startswith("Under"):
                line = None
                for tok in str(sugg).split():
                    try:
                        line = float(tok)
                        break
                    except:
                        pass
                if line is None:
                    continue
                result = (total > line) if sugg.startswith("Over") else (total < line)
            elif sugg == "BTTS: Yes":
                result = (gh > 0 and ga > 0)
            elif sugg == "BTTS: No":
                result = not (gh > 0 and ga > 0)
            elif sugg == "Home Win":
                result = (gh > ga)
            elif sugg == "Away Win":
                result = (ga > gh)

            if result is None:
                continue

            graded += 1
            if result:
                wins += 1

            if odds:
                stake += 1.0
                pnl += (float(odds) - 1.0) if result else -1.0

        acc = (100.0 * wins / graded) if graded else 0.0
        roi = (100.0 * pnl / stake) if stake > 0 else 0.0

        return jsonify({
            "ok": True,
            "tips_last_24h": int(t24),
            "tips_last_7d": int(t7d),
            "unsent_last_7d": int(unsent),
            "graded_last_7d": int(graded),
            "wins_last_7d": int(wins),
            "accuracy_last_7d_pct": round(acc, 1),
            "roi_last_7d_pct": round(roi, 1),
        })
    except Exception as e:
        log.exception("/stats failed: %s", e)
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/admin/scan", methods=["POST","GET"])
def http_scan():
    _require_admin()
    if is_rest_window_now():
        return jsonify({"ok": True, "saved": 0, "live_seen": 0, "skipped": "rest-window"}), 200
    s, l = production_scan()
    return jsonify({"ok": True, "saved": s, "live_seen": l})

@app.route("/admin/train", methods=["POST","GET"])
def http_train():
    _require_admin()
    if not TRAIN_ENABLE:
        return jsonify({"ok": False, "reason": "training disabled"}), 400
    res = train_models()
    return jsonify({"ok": True, "result": res})

@app.route("/admin/digest", methods=["POST","GET"])
def http_digest():
    _require_admin()
    msg = daily_accuracy_digest()
    return jsonify({"ok": True, "sent": bool(msg)})

@app.route("/admin/auto-tune", methods=["POST","GET"])
def http_auto_tune():
    _require_admin()
    tuned = auto_tune_thresholds(14)
    return jsonify({"ok": True, "tuned": tuned})

@app.route("/admin/motd", methods=["POST","GET"])
def http_motd():
    _require_admin()
    ok = send_match_of_the_day()
    return jsonify({"ok": bool(ok)})

# --- Web-browser GET endpoints (admin key via ?key=...) ---

@app.route("/admin/backfill-results", methods=["GET"])
def http_backfill_results():
    _require_admin()
    # optional: ?max=400
    try:
        max_rows = int(request.args.get("max", "400"))
    except Exception:
        max_rows = 400
    n = backfill_results_for_open_matches(max_rows)
    return jsonify({"ok": True, "updated": int(n)})

@app.route("/admin/prematch-scan", methods=["GET"])
def http_prematch_scan():
    _require_admin()
    if is_rest_window_now():
        return jsonify({"ok": True, "saved": 0, "skipped": "rest-window"}), 200
    saved = prematch_scan_save()
    return jsonify({"ok": True, "saved": int(saved)})

@app.route("/admin/retry-unsent", methods=["GET"])
def http_retry_unsent():
    _require_admin()
    # optional: ?minutes=30&limit=200
    try:
        minutes = int(request.args.get("minutes", "30"))
    except Exception:
        minutes = 30
    try:
        limit = int(request.args.get("limit", "200"))
    except Exception:
        limit = 200
    n = retry_unsent_tips(minutes=minutes, limit=limit)
    return jsonify({"ok": True, "resent": int(n)})

@app.route("/admin/live-scan", methods=["GET"])
def http_live_scan_once():
    _require_admin()
    if is_rest_window_now():
        return jsonify({"ok": True, "saved": 0, "live_seen": 0, "skipped": "rest-window"}), 200
    saved, live_seen = production_scan()
    return jsonify({"ok": True, "saved": int(saved), "live_seen": int(live_seen)})

@app.route("/admin/train", methods=["GET"])
def http_train_get():
    _require_admin()
    if not TRAIN_ENABLE:
        return jsonify({"ok": False, "reason": "training disabled"}), 400
    res = train_models()
    return jsonify({"ok": True, "result": res})

@app.route("/admin/auto-tune", methods=["GET"])
def http_auto_tune_get():
    _require_admin()
    tuned = auto_tune_thresholds(14)
    return jsonify({"ok": True, "tuned": tuned})

@app.route("/admin/daily-digest", methods=["GET"])
def http_daily_digest_get():
    _require_admin()
    msg = daily_accuracy_digest()
    return jsonify({"ok": True, "sent": bool(msg)})

@app.route("/admin/motd", methods=["GET"])
def http_motd_get():
    _require_admin()
    ok = send_match_of_the_day()
    return jsonify({"ok": bool(ok)})

# Optional: REST window config endpoint (admin)
@app.route("/admin/rest-window", methods=["GET","POST"])
def http_rest_window():
    _require_admin()
    global REST_START_HOUR_BERLIN, REST_END_HOUR_BERLIN
    if request.method == "GET":
        return jsonify({
            "ok": True,
            "start_hour": REST_START_HOUR_BERLIN,
            "end_hour": REST_END_HOUR_BERLIN,
            "active_now": bool(is_rest_window_now()),
            "tz": "Europe/Berlin"
        })
    body = request.get_json(silent=True) or {}
    try:
        if "start_hour" in body:
            REST_START_HOUR_BERLIN = int(body["start_hour"])
        if "end_hour" in body:
            REST_END_HOUR_BERLIN = int(body["end_hour"])
        return jsonify({"ok": True, "start_hour": REST_START_HOUR_BERLIN, "end_hour": REST_END_HOUR_BERLIN})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

# ───────── Boot ─────────
def _on_boot():
    init_db()
    _start_scheduler_once()

_on_boot()

if __name__ == "__main__":
    app.run(host=os.getenv("HOST","0.0.0.0"), port=int(os.getenv("PORT","8080")))
