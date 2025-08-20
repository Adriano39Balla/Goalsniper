# app/routes.py
import json
import logging
from flask import Blueprint, jsonify, request

from app.harvest import harvest_scan
from app.backfill import backfill_results_from_snapshots
from app.training import run_training
from app.db import db_conn

routes = Blueprint("routes", __name__)


@routes.route("/harvest", methods=["POST"])
def harvest_route():
    try:
        saved = harvest_scan()
        return jsonify({"ok": True, "snapshots_saved": saved})
    except Exception as e:
        logging.exception("[ROUTE] harvest error: %s", e)
        return jsonify({"ok": False, "error": str(e)}), 500


@routes.route("/backfill", methods=["POST"])
def backfill_route():
    try:
        inserted = backfill_results_from_snapshots()
        return jsonify({"ok": True, "results_inserted": inserted})
    except Exception as e:
        logging.exception("[ROUTE] backfill error: %s", e)
        return jsonify({"ok": False, "error": str(e)}), 500


@routes.route("/train", methods=["POST"])
def train_route():
    try:
        result = run_training()
        return jsonify({"ok": True, "training_result": result})
    except Exception as e:
        logging.exception("[ROUTE] train error: %s", e)
        return jsonify({"ok": False, "error": str(e)}), 500


@routes.route("/stats", methods=["GET"])
def stats_route():
    """
    Basic stats overview from DB (v_tip_stats view).
    """
    try:
        with db_conn() as conn:
            cur = conn.execute("SELECT * FROM v_tip_stats ORDER BY n DESC LIMIT 20")
            rows = [dict(r) for r in cur.fetchall()]
        return jsonify({"ok": True, "stats": rows})
    except Exception as e:
        logging.exception("[ROUTE] stats error: %s", e)
        return jsonify({"ok": False, "error": str(e)}), 500


@routes.route("/debug/settings", methods=["GET"])
def debug_settings_route():
    """
    Dump settings key/values for inspection.
    """
    try:
        with db_conn() as conn:
            cur = conn.execute("SELECT key, value FROM settings")
            rows = [dict(r) for r in cur.fetchall()]
        return jsonify({"ok": True, "settings": rows})
    except Exception as e:
        logging.exception("[ROUTE] debug error: %s", e)
        return jsonify({"ok": False, "error": str(e)}), 500
