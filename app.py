import os
import time
import logging
from flask import Flask, jsonify, request, abort

from core.config import config
from core.database import db
from services.telegram import telegram_service
from jobs.scheduler import JobScheduler
from utils.shutdown import ShutdownManager
from utils.logging import setup_logging

# Setup logging
setup_logging()
log = logging.getLogger("goalsniper")

app = Flask(__name__)

# Global instances
scheduler = JobScheduler()
shutdown_manager = ShutdownManager()

@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "ok": True, 
        "name": "goalsniper", 
        "mode": "INPLAY_AI_ENHANCED", 
        "scheduler": config.scheduler.enabled
    })

@app.route("/health", methods=["GET"])
def health():
    try:
        with db.get_cursor() as c:
            n = c.execute("SELECT COUNT(*) FROM tips").fetchone()[0]
        
        api_ok = False
        try:
            from services.api_client import api_client
            test_resp = api_client.get_fixtures({"live": "all"})
            api_ok = test_resp is not None
        except:
            pass
        
        return jsonify({
            "ok": True, 
            "db": "ok", 
            "tips_count": int(n),
            "api_connected": api_ok,
            "scheduler_running": scheduler.is_running(),
            "timestamp": time.time()
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/admin/scan", methods=["POST", "GET"])
def http_scan():
    _require_admin()
    from jobs.scanners import ProductionScanner
    scanner = ProductionScanner()
    saved, live_seen = scanner.scan()
    return jsonify({"ok": True, "saved": saved, "live_seen": live_seen})

@app.route("/admin/train", methods=["POST", "GET"])
def http_train():
    _require_admin()
    if not config.train_enable:
        return jsonify({"ok": False, "reason": "training disabled"}), 400
    
    try:
        from services.training import TrainingService
        trainer = TrainingService()
        result = trainer.train_models()
        return jsonify({"ok": True, "result": result})
    except Exception as e:
        log.exception("train_models failed: %s", e)
        return jsonify({"ok": False, "error": str(e)}), 500

def _require_admin():
    """Require admin authentication"""
    admin_key = os.getenv("ADMIN_API_KEY")
    if not admin_key:
        abort(401)
    
    key = (request.headers.get("X-API-Key") or 
           request.args.get("key") or 
           ((request.json or {}).get("key") if request.is_json else None))
    
    if key != admin_key:
        abort(401)

def initialize_app():
    """Initialize the application"""
    log.info("Initializing GoalSniper AI...")
    
    # Validate configuration
    config.validate()
    
    # Initialize database
    db.initialize()
    db.init_schema()
    
    # Start scheduler if enabled
    if config.scheduler.enabled:
        scheduler.start()
    
    log.info("GoalSniper AI initialized successfully")

# Initialize on import
initialize_app()

if __name__ == "__main__":
    app.run(host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8080")))
