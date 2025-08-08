from flask import Blueprint, render_template_string
from models import Match, Prediction
from app import db
from datetime import datetime, timedelta

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/')
def index():
    live_matches = Match.query.filter(Match.status.in_(['1H', '2H', 'LIVE', 'ET'])).order_by(Match.match_date.desc()).all()
    recent_predictions = Prediction.query.order_by(Prediction.predicted_at.desc()).limit(10).all()

    return render_template_string(TEMPLATE, matches=live_matches, predictions=recent_predictions)


TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Football Betting Assistant Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f7f7f7; }
        h2 { color: #333; }
        .section { margin-bottom: 40px; }
        table { border-collapse: collapse; width: 100%; background: white; }
        th, td { text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }
        th { background-color: #f2f2f2; }
        .timestamp { color: #999; font-size: 0.9em; }
    </style>
</head>
<body>
    <h1>📊 Football Betting Assistant Dashboard</h1>

    <div class="section">
        <h2>🟢 Live Matches</h2>
        <table>
            <tr>
                <th>Match</th>
                <th>Status</th>
                <th>Score</th>
                <th>League</th>
                <th>Time</th>
            </tr>
            {% for match in matches %}
            <tr>
                <td>{{ match.home_team }} vs {{ match.away_team }}</td>
                <td>{{ match.status }}</td>
                <td>{{ match.home_score }} - {{ match.away_score }}</td>
                <td>{{ match.league_name }}</td>
                <td class="timestamp">{{ match.match_date.strftime('%Y-%m-%d %H:%M') }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>

    <div class="section">
        <h2>📈 Recent Predictions</h2>
        <table>
            <tr>
                <th>Match</th>
                <th>Prediction</th>
                <th>Confidence</th>
                <th>Sent</th>
                <th>Outcome</th>
            </tr>
            {% for pred in predictions %}
            <tr>
                <td>{{ pred.match.home_team }} vs {{ pred.match.away_team }}</td>
                <td>{{ pred.prediction_type }}</td>
                <td>{{ "%.0f" % (pred.confidence * 100) }}%</td>
                <td class="timestamp">{{ pred.predicted_at.strftime('%Y-%m-%d %H:%M') }}</td>
                <td>{{ "✅" if pred.actual_outcome else "❌" if pred.actual_outcome == False else "Pending" }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>
</body>
</html>
"""
