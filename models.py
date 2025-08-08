# models.py

from app import db
from datetime import datetime
from sqlalchemy.types import TEXT
import json

# JSON-encoded dict type for SQLite compatibility
class JSONEncodedDict(db.TypeDecorator):
    impl = TEXT

    def process_bind_param(self, value, dialect):
        if value is None:
            return '{}'
        return json.dumps(value)

    def process_result_value(self, value, dialect):
        if not value:
            return {}
        return json.loads(value)

class Match(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    api_match_id = db.Column(db.Integer, unique=True, nullable=False)
    home_team = db.Column(db.String(100), nullable=False)
    away_team = db.Column(db.String(100), nullable=False)
    league_id = db.Column(db.Integer, nullable=False)
    league_name = db.Column(db.String(100), nullable=False)
    match_date = db.Column(db.DateTime, nullable=False)
    status = db.Column(db.String(20), nullable=False)
    home_score = db.Column(db.Integer)
    away_score = db.Column(db.Integer)
    statistics = db.Column(JSONEncodedDict)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    match_id = db.Column(db.Integer, db.ForeignKey('match.id'), nullable=False)
    prediction_type = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    reasoning = db.Column(db.Text)
    predicted_at = db.Column(db.DateTime, default=datetime.utcnow)
    actual_outcome = db.Column(db.Boolean)
    feedback_received = db.Column(db.Boolean, default=False)
    sent_to_telegram = db.Column(db.Boolean, default=False)

    match = db.relationship('Match', backref=db.backref('predictions', lazy=True))

class PredictionFeedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    prediction_id = db.Column(db.Integer, db.ForeignKey('prediction.id'), nullable=False)
    is_correct = db.Column(db.Boolean, nullable=False)
    feedback_time = db.Column(db.DateTime, default=datetime.utcnow)

    prediction = db.relationship('Prediction', backref=db.backref('feedbacks', lazy=True))

class LearnedPattern(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    pattern_type = db.Column(db.String(50), nullable=False)
    conditions = db.Column(JSONEncodedDict, nullable=False)
    success_rate = db.Column(db.Float, nullable=False)
    sample_size = db.Column(db.Integer, nullable=False)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
