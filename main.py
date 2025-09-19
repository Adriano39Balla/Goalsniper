# file: main.py
# goalsniper AI – Opta Supercomputer style (entrypoint)

import os
import time
import uuid
import hmac
import logging
import signal, sys
from typing import Any, Callable, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, jsonify, request, abort, g, has_request_context
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.executors.pool import ThreadPoolExecutor as APS_ThreadPoolExecutor
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
_base_log = logging.getLogger("goalsniper")
app = Flask(__name__)

def get_logger():
    """Return a logger that includes request_id when there is a request context."""
    try:
        if has_request_context():
            rid = getattr(g, "request_id", None)
            if rid:
                return logging.LoggerAdapter(_base_log, extra={"request_id": rid})
    except Exception:
        pass
    return _base_log

# ───────── Small DB helpers (psycopg2/3 safe) ─────────
def _scalar(c, sql: str, params: tuple = ()):
    c.execute(sql, params)
    row = c.fetchone()
    return row[0] if row else None

# ───────── Env helpers ─────────
def env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() not in {"0", "false", "no", "off", ""}

def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

# ───────── Env ─────────
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")  # empty means: all admin endpoints will 401
ALLOW_QUERY_KEY = env_bool("ALLOW_QUERY_KEY", False)

RUN_SCHEDULER = env_bool("RUN_SCHEDULER", True)
SCAN_INTERVAL_SEC = env_int("SCAN_INTERVAL_SEC", 300)
BACKFILL_EVERY_MIN = env_int("BACKFILL_EVERY_MIN", 15)
TRAIN_ENABLE = env_bool("TRAIN_ENABLE", True)
TRAIN_HOUR_UTC = env_int("TRAIN_HOUR_UTC", 2)
TRAIN_MINUTE_UTC = env_int("TRAIN_MINUTE_UTC", 12)
AUTO_TUNE_ENABLE = env_bool("AUTO_TUNE_ENABLE", False)
DAILY_ACCURACY_DIGEST_ENABLE = env_bool("DAILY_ACCURACY_DIGEST_ENABLE", True)
DAILY_ACCURACY_HOUR = env_int("DAILY_ACCURACY_HOUR", 8)
DAILY_ACCURACY_MINUTE = env_int("DAILY_ACCURACY_MINUTE", 0)
MOTD_ENABLE = env_bool("MOTD_ENABLE", True)
MOTD_HOUR = env_int("MOTD_HOUR", 10)
MOTD_MINUTE = env_int("MOTD_MINUTE", 0)
SEND_BOOT_TELEGRAM = env_bool("SEND_BOOT_TELEGRAM", False)

TZ_UTC = ZoneInfo("UTC")
BERLIN_TZ = ZoneInfo("Europe/Berlin")

# Nightly cooldown (Berlin) — default 23:00–07:00
REST_START_HOUR_BERLIN = env_int("REST_START_HOUR_BERLIN", 23)  # inclusive
REST_END_HOUR_BERLIN = env_int("REST_END_HOUR_BERLIN", 7)       # exclusive

def is_rest_window_now() -> bool:
    """Return True if current Berlin local time is within the nightly rest window."""
    h = datetime.now(BERLIN_TZ).hour
    if REST_START_HOUR_BERLIN <= REST_END_HOUR_BERLIN:
        return REST_START_HOUR_BERLIN <= h < REST_END_HOUR_BERLIN
    return (h >= REST_START_HOUR_BERLIN) or (h < REST_END_HOUR_BERLIN)

# ───────── Request middleware ─────────
@app.before_request
def _assign_request_id():
    g.request_start = time.time()
    g.request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex

@app.after_request
def _inject_request_id(resp):
    resp.headers["X-Request-ID"] = getattr(g, "request_id", "")
    return resp

# ───────── Admin auth ─────────
def _constant_time_eq(a: str, b: str) -> bool:
    a = a or ""
    b = b or ""
    try:
        return hmac.compare_digest(a, b)
    except Exception:
        return a == b

def _require_admin():
    if not ADMIN_API_KEY:
        abort(401)
    key = request.headers.get("X-API-Key")
    if not key and ALLOW_QUERY_KEY:
        key = request.args.get("key") or ((request.json or {}).get("key") if request.is_json else None)
    if not key or not _constant_time_eq(key, ADMIN_API_KEY):
        abort(401)

# ───────── Scheduler ─────────
_scheduler_started = False
_scheduler_ref: Optional[BackgroundScheduler] = None

def _on_sigterm(*_):
    _base_log.info("[BOOT] SIGTERM received — shutting down (likely platform restart)")
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
    except Exception:
        pass
    sys.exit(0)

signal.signal(signal.SIGTERM, _on_sigterm)

def _run_with_pg_lock(lock_key: int, fn: Callable, *a, **k):
    log = get_logger()
    try:
        with db_conn() as c:
            # execute() may return None (psycopg2) → always fetch from cursor
            c.execute("SELECT pg_try_advisory_lock(%s)", (lock_key,))
            row = c.fetchone()
            got = bool(row and row[0])
            if not got:
                log.info("[LOCK %s] busy; skipped.", lock_key)
                return None
            try:
                return fn(*a, **k)
            finally:
                try:
                    c.execute("SELECT pg_advisory_unlock(%s)", (lock_key,))
                except Exception:
                    pass
    except Exception as e:
        log.exception("[LOCK %s] failed: %s", lock_key, e)
        return None

def _maybe_skip_rest(fn_name: str) -> bool:
    log = get_logger()
    if is_rest_window_now():
        log.info("[%s] rest window active (%02d:00–%02d:00 Berlin) — skipped.",
                 fn_name, REST_START_HOUR_BERLIN, REST_END_HOUR_BERLIN)
        return True
    return False

def _start_scheduler_once():
    global _scheduler_started, _scheduler_ref
    log = _base_log
    if _scheduler_started or not RUN_SCHEDULER:
        return
    # Avoid double-start under Flask dev reloader
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        pass
    elif os.environ.get("FLASK_ENV") == "development" and os.environ.get("WERKZEUG_RUN_MAIN") is None:
        return
    try:
        sched = BackgroundScheduler(timezone=TZ_UTC)

        def _scan_job():
            if _maybe_skip_rest("scan"):
                return
            return _run_with_pg_lock(1001, production_scan)

        sched.add_job(
            _scan_job, "interval",
            seconds=SCAN_INTERVAL_SEC, id="scan", max_instances=1, coalesce=True, misfire_grace_time=60
        )

        sched.add_job(
            lambda: _run_with_pg_lock(1002, backfill_results_for_open_matches, 400),
            "interval", minutes=BACKFILL_EVERY_MIN, id="backfill", max_instances=1, coalesce=True, misfire_grace_time=120
        )

        if DAILY_ACCURACY_DIGEST_ENABLE:
            sched.add_job(
                lambda: _run_with_pg_lock(1003, daily_accuracy_digest),
                CronTrigger(hour=DAILY_ACCURACY_HOUR, minute=DAILY_ACCURACY_MINUTE, timezone=BERLIN_TZ),
                id="digest", max_instances=1, coalesce=True, misfire_grace_time=3600
            )

        if MOTD_ENABLE:
            sched.add_job(
                lambda: _run_with_pg_lock(1004, send_match_of_the_day),
                CronTrigger(hour=MOTD_HOUR, minute=MOTD_MINUTE, timezone=BERLIN_TZ),
                id="motd", max_instances=1, coalesce=True, misfire_grace_time=3600
            )

        if TRAIN_ENABLE:
            sched.add_job(
                lambda: _run_with_pg_lock(1005, train_models),
                CronTrigger(hour=TRAIN_HOUR_UTC, minute=TRAIN_MINUTE_UTC, timezone=TZ_UTC),
                id="train", max_instances=1, coalesce=True, misfire_grace_time=3600
            )

        if AUTO_TUNE_ENABLE:
            sched.add_job(
                lambda: _run_with_pg_lock(1006, auto_tune_thresholds, 14),
                CronTrigger(hour=4, minute=7, timezone=TZ_UTC),
                id="auto_tune", max_instances=1, coalesce=True, misfire_grace_time=3600
            )

        sched.add_job(
            lambda: _run_with_pg_lock(1007, retry_unsent_tips, 30, 200),
            "interval", minutes=10, id="retry", max_instances=1, coalesce=True, misfire_grace_time=120
        )

        sched.start()
        _scheduler_started = True
        _scheduler_ref = sched

        if SEND_BOOT_TELEGRAM:
            try:
                send_telegram("🚀 goalsniper AI live and scanning (night rest 23:00–07:00 Berlin)")
            except Exception:
                log.warning("Boot telegram failed", exc_info=True)

        log.info("[SCHED] started (scan=%ss)", SCAN_INTERVAL_SEC)

    except Exception as e:
        log.exception("[SCHED] failed: %s", e)

def _stop_scheduler(*_args):
    try:
        if _scheduler_ref and _scheduler_ref.running:
            _base_log.info("[SCHED] shutting down…")
            _scheduler_ref.shutdown(wait=False)
    except Exception:
        _base_log.warning("[SCHED] shutdown error", exc_info=True)

# ───────── Routes ─────────
@app.route("/")
def root():
    return jsonify({
        "ok": True, "name": "goalsniper", "mode": "FULL_AI", "scheduler": RUN_SCHEDULER,
        "request_id": getattr(g, "request_id", None)
    })

@app.route("/health")
def health():
    """
    Readiness/diagnostic endpoint.
    - Always returns 200 so platform probes won't kill the app.
    - Includes db status; optionally returns counts when ?deep=1.
    """
    log = get_logger()
    deep = request.args.get("deep") in ("1", "true", "yes")
    resp: dict[str, Any] = {"ok": True, "service": "goalsniper", "scheduler": RUN_SCHEDULER}

    try:
        with db_conn() as c:
            c.execute("SELECT 1")
            resp["db"] = "ok"
            if deep:
                try:
                    n = _scalar(c, "SELECT COUNT(*) FROM tips") or 0
                    resp["tips_count"] = int(n)
                except Exception as e:
                    resp["tips_count_error"] = str(e)
    except Exception as e:
        resp["db"] = "down"
        resp["error"] = str(e)
        log.warning("/health degraded: %s", e)

    return jsonify(resp), 200

@app.route("/stats")
def stats():
    log = get_logger()
    try:
        now = int(time.time())
        day_ago = now - 24 * 3600
        week_ago = now - 7 * 24 * 3600
        with db_conn() as c:
            t24 = int(_scalar(
                c, "SELECT COUNT(*) FROM tips WHERE created_ts >= %s AND suggestion <> 'HARVEST'",
                (day_ago,)
            ) or 0)
            t7d = int(_scalar(
                c, "SELECT COUNT(*) FROM tips WHERE created_ts >= %s AND suggestion <> 'HARVEST'",
                (week_ago,)
            ) or 0)
            unsent = int(_scalar(
                c, "SELECT COUNT(*) FROM tips WHERE sent_ok=0 AND created_ts >= %s",
                (week_ago,)
            ) or 0)

            c.execute("""
                SELECT t.suggestion, t.odds, r.final_goals_h, r.final_goals_a, r.btts_yes
                FROM tips t
                JOIN match_results r ON r.match_id = t.match_id
                WHERE t.created_ts >= %s
                  AND t.suggestion <> 'HARVEST'
                  AND t.sent_ok = 1
            """, (week_ago,))
            rows = c.fetchall() or []

        graded = wins = 0
        stake = pnl = 0.0

        for (sugg, odds, gh, ga, btts) in rows:
            gh = int(gh or 0)
            ga = int(ga or 0)
            total = gh + ga
            result: Optional[bool] = None

            s = str(sugg or "")
            if s.startswith("Over") or s.startswith("Under"):
                line: Optional[float] = None
                for tok in s.split():
                    try:
                        line = float(tok)
                        break
                    except Exception:
                        pass
                if line is None:
                    continue
                result = (total > line) if s.startswith("Over") else (total < line)
            elif s == "BTTS: Yes":
                result = (gh > 0 and ga > 0)
            elif s == "BTTS: No":
                result = not (gh > 0 and ga > 0)
            elif s == "Home Win":
                result = (gh > ga)
            elif s == "Away Win":
                result = (ga > gh)

            if result is None:
                continue

            graded += 1
            if result:
                wins += 1

            if odds is not None:
                try:
                    o = float(odds)
                    stake += 1.0
                    pnl += (o - 1.0) if result else -1.0
                except Exception:
                    pass

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

# ── Admin endpoints ──
@app.route("/admin/scan", methods=["POST", "GET"])
def http_scan():
    _require_admin()
    if is_rest_window_now():
        return jsonify({"ok": True, "saved": 0, "live_seen": 0, "skipped": "rest-window"}), 200
    s, l = production_scan()
    return jsonify({"ok": True, "saved": s, "live_seen": l})

@app.route("/admin/train", methods=["POST", "GET"])
def http_train():
    _require_admin()
    if not TRAIN_ENABLE:
        return jsonify({"ok": False, "reason": "training disabled"}), 400
    res = train_models()
    return jsonify({"ok": True, "result": res})

@app.route("/admin/digest", methods=["POST", "GET"])
def http_digest():
    _require_admin()
    msg = daily_accuracy_digest()
    return jsonify({"ok": True, "sent": bool(msg)})

@app.route("/admin/auto-tune", methods=["POST", "GET"])
def http_auto_tune():
    _require_admin()
    tuned = auto_tune_thresholds(14)
    return jsonify({"ok": True, "tuned": tuned})

@app.route("/admin/motd", methods=["POST", "GET"])
def http_motd():
    _require_admin()
    ok = send_match_of_the_day()
    return jsonify({"ok": bool(ok)})

@app.route("/admin/backfill-results", methods=["GET"])
def http_backfill_results():
    _require_admin()
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
    try:
        init_db()
    except Exception as e:
        _base_log.exception("init_db failed (continuing to serve): %s", e)
    try:
        _start_scheduler_once()
    except Exception as e:
        _base_log.exception("scheduler start failed (continuing to serve): %s", e)

# Ensure clean scheduler shutdown on SIGTERM (Railway stop/redeploy)
signal.signal(signal.SIGTERM, _stop_scheduler)

_on_boot()

if __name__ == "__main__":
    # Use gunicorn/uwsgi in prod instead of Flask dev server.
    # Example: gunicorn -w 2 -k gthread -t 60 -b 0.0.0.0:8080 main:app
    app.run(host=os.getenv("HOST","0.0.0.0"), port=int(os.getenv("PORT","8080")))
