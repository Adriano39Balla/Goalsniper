#!/usr/bin/env python3
"""
main.py

Flask entrypoint for Goalsniper.
- Serves API routes (harvest, backfill, train, stats, debug).
- Designed for Railway deployment.
"""

import os
import logging
from flask import Flask
from app.routes import routes
from app.utils import setup_logging


def create_app() -> Flask:
    setup_logging(logging.INFO)
    app = Flask(__name__)
    app.register_blueprint(routes)
    return app


app = create_app()


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    logging.info(f"[MAIN] Starting server on port {port}")
    app.run(host="0.0.0.0", port=port)
