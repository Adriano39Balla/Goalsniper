# routes/tasks.py

from flask import Blueprint, jsonify
from services.scheduler import scheduler_service

task_bp = Blueprint("tasks", __name__, url_prefix="/tasks")

@task_bp.route("/scan-live", methods=["GET"])
def scan_live():
    scheduler_service.scan_live_matches()
    return jsonify({"status": "live scanned"})

@task_bp.route("/scan-upcoming", methods=["GET"])
def scan_upcoming():
    scheduler_service.scan_upcoming_matches()
    return jsonify({"status": "upcoming scanned"})

@task_bp.route("/daily-summary", methods=["GET"])
def summary():
    scheduler_service.send_daily_summary()
    return jsonify({"status": "summary sent"})

@task_bp.route("/cleanup", methods=["GET"])
def cleanup():
    scheduler_service.cleanup_old_data()
    return jsonify({"status": "cleanup done"})
