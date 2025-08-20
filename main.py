#!/usr/bin/env python3
"""
main.py

Flask entrypoint for Goalsniper.
- Creates the app instance
- Registers blueprints
- Runs in Railway
"""

from flask import Flask
from app.routes import bp as routes_bp  # ✅ import the blueprint correctly


def create_app():
    """Application factory for Flask."""
    app = Flask(__name__)

    # Register all blueprints
    app.register_blueprint(routes_bp)

    return app


# Railway / Gunicorn will look for "app"
app = create_app()

if __name__ == "__main__":
    # For local dev only
    app.run(host="0.0.0.0", port=5000, debug=True)
