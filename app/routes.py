from flask import Blueprint
from app.training import training_route

bp = Blueprint("routes", __name__)

@bp.route("/train", methods=["POST"])
def train():
    """Trigger model training for all markets."""
    return training_route()
