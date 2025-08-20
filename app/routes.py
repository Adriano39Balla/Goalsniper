"""
routes.py

Defines Flask API routes for Goalsniper.
Currently includes:
- /train [POST] : Trigger model training for all configured markets
"""

from flask import Blueprint
from app.training import training_route

# ✅ Define blueprint
bp = Blueprint("routes", __name__)

@bp.route("/train", methods=["POST"])
def train():
    """Trigger model training for all markets."""
    return training_route()
