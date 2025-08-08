import os
import logging
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from werkzeug.middleware.proxy_fix import ProxyFix

db = SQLAlchemy()

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "football-betting-assistant-2024")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///football_betting.db")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

db.init_app(app)

with app.app_context():
    import models
    db.create_all()

from routes.dashboard import dashboard_bp
from routes.tasks import task_bp
app.register_blueprint(dashboard_bp)
app.register_blueprint(task_bp)

from services.telegram_bot import init_telegram_bot
init_telegram_bot()
