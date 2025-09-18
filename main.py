# file: main.py
# goalsniper AI – Opta Supercomputer style (entrypoint)

import os, time, logging
from flask import Flask, jsonify, request, abort
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from zoneinfo import ZoneInfo

from db import init_db, db_conn
from telegram_utils import send_telegram
from scan import production_scan, prematch_scan_save, send_match_of_the_day, backfill_results_for_open_matches, daily_accuracy_digest, retry_unsent_tips
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
AUTO_TUNE_ENABLE = os.getenv("AUTO_TUNE_ENABLE", "0") not in ("0","false","False","no","NO")
DAILY_ACCURACY_DIGEST_ENABLE = os.getenv("DAILY_ACCURACY_DIGEST_ENABLE", "1") not in ("0","false","False","no","NO")
DAILY_ACCURACY_HOUR = int(os.getenv("DAILY_ACCURACY_HOUR", "8"))
DAILY_ACCURACY_MINUTE = int(os.getenv("DAILY_ACCURACY_MINUTE", "0"))
MOTD_ENABLE = os.getenv("MOTD_ENABLE", "1") not in ("0","false","False","no","NO")
MOTD_HOUR = int(os.getenv("MOTD_HOUR", "10"))
MOTD_MINUTE = int(os.getenv("MOTD_MINUTE", "0"))

TZ_UTC = ZoneInfo("UTC")
BERLIN_TZ = ZoneInfo("Europe/Berlin")

# ───────── Admin auth ─────────
def _require_admin():
    key = request.headers.get("X-API-Key") or request.args.get("key") or ((request.json or {}).get("key") if request.is_json else None)
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

def _start_scheduler_once():
    global _scheduler_started
    if _scheduler_started or not RUN_SCHEDULER:
        return
    try:
        sched = BackgroundScheduler(timezone=TZ_UTC)

        # live in-play scanning
        sched.add_job(lambda: _run_with_pg_lock(1001, production_scan),
                      "interval", seconds=SCAN_INTERVAL_SEC, id="scan", max_instances=1, coalesce=True)

        # backfill results
        sched.add_job(lambda: _run_with_pg_lock(1002, backfill_results_for_open_matches, 400),
                      "interval", minutes=BACKFILL_EVERY_MIN, id="backfill", max_instances=1, coalesce=True)

        # daily digest
        if DAILY_ACCURACY_DIGEST_ENABLE:
            sched.add_job(lambda: _run_with_pg_lock(1003, daily_accuracy_digest),
                          CronTrigger(hour=DAILY_ACCURACY_HOUR, minute=DAILY_ACCURACY_MINUTE, timezone=BERLIN_TZ),
                          id="digest", max_instances=1, coalesce=True)

        # match of the day
        if MOTD_ENABLE:
            sched.add_job(lambda: _run_with_pg_lock(1004, send_match_of_the_day),
                          CronTrigger(hour=MOTD_HOUR, minute=MOTD_MINUTE, timezone=BERLIN_TZ),
                          id="motd", max_instances=1, coalesce=True)

        # training
        if TRAIN_ENABLE:
            sched.add_job(lambda: _run_with_pg_lock(1005, train_models),
                          CronTrigger(hour=TRAIN_HOUR_UTC, minute=TRAIN_MINUTE_UTC, timezone=TZ_UTC),
                          id="train", max_instances=1, coalesce=True)

        # auto-tune thresholds
        if AUTO_TUNE_ENABLE:
            sched.add_job(lambda: _run_with_pg_lock(1006, auto_tune_thresholds, 14),
                          CronTrigger(hour=4, minute=7, timezone=TZ_UTC),
                          id="auto_tune", max_instances=1, coalesce=True)

        # retry unsent
        sched.add_job(lambda: _run_with_pg_lock(1007, retry_unsent_tips, 30, 200),
                      "interval", minutes=10, id="retry", max_instances=1, coalesce=True)

        sched.start()
        _scheduler_started = True
        send_telegram("🚀 goalsniper AI live and scanning")
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

@app.route("/admin/scan", methods=["POST","GET"])
def http_scan():
    _require_admin()
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

# ───────── Boot ─────────
def _on_boot():
    init_db()
    _start_scheduler_once()

_on_boot()

if __name__ == "__main__":
    app.run(host=os.getenv("HOST","0.0.0.0"), port=int(os.getenv("PORT","8080")))
