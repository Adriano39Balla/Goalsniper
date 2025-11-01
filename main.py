# main.py

import os
import sys
import time
import signal
import atexit
import logging

from flask import Flask, jsonify, request, abort

from config import (
    TRAIN_ENABLE,
    AUTO_TUNE_ENABLE,
    RUN_SCHEDULER,
    TRAIN_HOUR_UTC,
    TRAIN_MINUTE_UTC,
    SCAN_INTERVAL_SEC,
    BACKFILL_EVERY_MIN,
    ADMIN_API_KEY,
    WEBHOOK_SECRET,
    TZ_UTC,
    BERLIN_TZ,
    METRICS,
)

from telegram import send_telegram
from db import (
    db_conn,
    init_db,
    _init_pool,
    POOL,
    _SETTINGS_CACHE,
    set_setting,
    get_setting_cached,
    invalidate_model_caches_for_key,
)
from predictions import (
    production_scan,
    prematch_scan_save,
    backfill_results_for_open_matches,
    daily_accuracy_digest,
    send_match_of_the_day,
    _api_get,
)
from train_models import train_models, auto_tune_thresholds
from scheduler import BackgroundScheduler, CronTrigger

# ───────────── Logging ───────────── #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] - %(message)s",
)
log = logging.getLogger("main")

# ───────────── Globals ───────────── #
app = Flask(__name__)
_SHUTDOWN_RAN = False
_SHUTDOWN_HANDLERS_SET = False
_SCHED = None
_scheduler_started = False

# ───────────── Admin Auth ───────────── #
def _require_admin():
    key = request.headers.get("X-API-Key") or request.args.get("key") or (
        (request.json or {}).get("key") if request.is_json else None
    )
    if not ADMIN_API_KEY or key != ADMIN_API_KEY:
        abort(401)

# ───────────── API Routes ───────────── #

@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "ok": True,
        "name": "goalsniper",
        "mode": "FULL_AI_ENHANCED",
        "scheduler": RUN_SCHEDULER,
    })

@app.route("/health", methods=["GET"])
def health():
    try:
        with db_conn() as c:
            n = c.execute("SELECT COUNT(*) FROM tips").fetchone()[0]
        api_ok = False
        try:
            test_resp = _api_get(os.getenv("FOOTBALL_API_URL"), {"live": "all"}, timeout=5)
            api_ok = test_resp is not None
        except:
            pass
        return jsonify({
            "ok": True,
            "db": "ok",
            "tips_count": int(n),
            "api_connected": api_ok,
            "scheduler_running": _scheduler_started,
            "timestamp": time.time()
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/admin/train", methods=["GET", "POST"])
def http_train():
    _require_admin()
    if not TRAIN_ENABLE:
        return jsonify({"ok": False, "reason": "training disabled"}), 400
    try:
        result = train_models()
        return jsonify({"ok": True, "result": result})
    except Exception as e:
        log.exception("train_models failed: %s", e)
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/admin/auto-tune", methods=["GET", "POST"])
def http_auto_tune():
    _require_admin()
    tuned = auto_tune_thresholds()
    return jsonify({"ok": True, "tuned": tuned})

# Add all other routes like /tips/latest, /admin/scan, /metrics, /digest, etc.
# They should call into services from predictions, db, telegram, etc.

# ───────────── Shutdown Handlers ───────────── #

def shutdown_handler(signum=None, frame=None, *, from_atexit: bool = False):
    global _SHUTDOWN_RAN
    if _SHUTDOWN_RAN:
        return
    _SHUTDOWN_RAN = True

    who = "atexit" if from_atexit else ("signal" if signum else "manual")
    log.info("Received shutdown (%s), cleaning up...", who)

    try:
        if _SCHED is not None:
            _SCHED.shutdown(wait=False)
    except Exception:
        pass

    try:
        if POOL:
            POOL.closeall()
    except Exception as e:
        log.warning("Error closing pool during shutdown: %s", e)

    if not from_atexit:
        try:
            sys.exit(0)
        except SystemExit:
            pass

def register_shutdown_handlers():
    global _SHUTDOWN_HANDLERS_SET
    if _SHUTDOWN_HANDLERS_SET:
        return
    _SHUTDOWN_HANDLERS_SET = True

    signal.signal(signal.SIGINT, lambda s, f: shutdown_handler(s, f, from_atexit=False))
    signal.signal(signal.SIGTERM, lambda s, f: shutdown_handler(s, f, from_atexit=False))
    atexit.register(lambda: shutdown_handler(from_atexit=True))

# ───────────── Scheduler ───────────── #

def _run_with_pg_lock(lock_key: int, fn, *args, **kwargs):
    try:
        with db_conn() as c:
            got = c.execute("SELECT pg_try_advisory_lock(%s)", (lock_key,)).fetchone()[0]
            if not got:
                log.info("[LOCK %s] busy; skipped.", lock_key)
                return None
            try:
                return fn(*args, **kwargs)
            finally:
                c.execute("SELECT pg_advisory_unlock(%s)", (lock_key,))
    except Exception as e:
        log.exception("[LOCK %s] failed: %s", lock_key, e)
        return None

def _start_scheduler_once():
    global _scheduler_started, _SCHED
    if _scheduler_started or not RUN_SCHEDULER:
        return
    try:
        sched = BackgroundScheduler(timezone=TZ_UTC)
        sched.add_job(lambda: _run_with_pg_lock(1001, production_scan), "interval", seconds=SCAN_INTERVAL_SEC, id="scan", max_instances=1, coalesce=True)
        sched.add_job(lambda: _run_with_pg_lock(1002, backfill_results_for_open_matches, 400), "interval", minutes=BACKFILL_EVERY_MIN, id="backfill", max_instances=1, coalesce=True)

        if AUTO_TUNE_ENABLE:
            sched.add_job(lambda: _run_with_pg_lock(1003, auto_tune_thresholds, 14),
                          CronTrigger(hour=4, minute=7, timezone=TZ_UTC),
                          id="auto_tune", max_instances=1, coalesce=True)

        if TRAIN_ENABLE:
            sched.add_job(lambda: _run_with_pg_lock(1004, train_models),
                          CronTrigger(hour=TRAIN_HOUR_UTC, minute=TRAIN_MINUTE_UTC, timezone=TZ_UTC),
                          id="train", max_instances=1, coalesce=True)

        sched.start()
        _SCHED = sched
        _scheduler_started = True
        send_telegram("🚀 Scheduler started with enhanced predictions.")
        log.info("[SCHEDULER] started (scan interval: %ss)", SCAN_INTERVAL_SEC)
    except Exception as e:
        log.exception("[SCHEDULER] failed to start: %s", e)

# ───────────── Bootstrapping ───────────── #

def _on_boot():
    register_shutdown_handlers()
    _init_pool()
    init_db()
    set_setting("boot_ts", str(int(time.time())))
    _start_scheduler_once()

_on_boot()

if __name__ == "__main__":
    app.run(host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8080")))
