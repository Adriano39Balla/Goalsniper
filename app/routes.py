"""
routes.py

Defines Flask API routes for Goalsniper.
Includes:
- /harvest [POST] : Harvest new snapshots from API
- /backfill [POST]: Backfill historical results into DB
- /train [POST]   : Trigger model training for all configured markets
"""

from flask import Blueprint
from app.training import training_route
from app.harvest import harvest_route
from app.backfill import backfill_route

# ✅ Define blueprint
bp = Blueprint("routes", __name__)

@bp.route("/harvest", methods=["POST"])
def harvest():
    """Harvest new snapshots from API."""
    return harvest_route()

@bp.route("/backfill", methods=["POST"])
def backfill():
    """Backfill historical match results into DB."""
    return backfill_route()

@bp.route("/train", methods=["POST"])
def train():
    """Trigger model training for all markets."""
    return training_route()
