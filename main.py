from flask import Flask, jsonify
import os
from datetime import datetime, timedelta
import schedule
import time
import threading
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor
import requests
import pandas as pd
import joblib
import numpy as np

# Load environment variables
load_dotenv()

app = Flask(__name__)

# ========== CONFIGURATION ==========
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY")
API_FOOTBALL_HOST = "v3.football.api-sports.io"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Database connection
def get_db_connection():
    conn = psycopg2.connect(
        host=SUPABASE_URL,
        database="postgres",
        user="postgres",
        password=SUPABASE_KEY,
        port=5432
    )
    return conn

# ========== MANUAL CONTROL ROUTES (GET) ==========
@app.route('/manual/scan', methods=['GET'])
def manual_scan():
    """Trigger a live data scan manually"""
    try:
        result = fetch_live_fixtures()
        return jsonify({"status": "success", "message": "Live scan completed", "data": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/manual/train', methods=['GET'])
def manual_train():
    """Trigger model training manually"""
    try:
        # This would call functions from train_models.py
        from train_models import run_training_pipeline
        result = run_training_pipeline()
        return jsonify({"status": "success", "message": "Training completed", "details": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/manual/digest', methods=['GET'])
def manual_digest():
    """Generate and send daily digest"""
    try:
        digest = generate_daily_digest()
        send_telegram_message(digest)
        return jsonify({"status": "success", "message": "Daily digest sent"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/manual/autotune', methods=['GET'])
def manual_autotune():
    """Trigger hyperparameter auto-tuning"""
    try:
        from train_models import auto_tune_models
        result = auto_tune_models()
        return jsonify({"status": "success", "message": "Auto-tuning completed", "best_params": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/manual/backfill', methods=['GET'])
def manual_backfill():
    """Backfill historical data"""
    try:
        days = int(request.args.get('days', 365))  # Default to 1 year
        result = backfill_historical_data(days)
        return jsonify({"status": "success", "message": f"Backfilled {days} days", "matches_added": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/manual/health', methods=['GET'])
def manual_health():
    """System health check"""
    health_status = check_system_health()
    return jsonify(health_status)

# ========== CORE DATA FUNCTIONS ==========
def fetch_live_fixtures():
    """Fetch live fixtures from API-Football"""
    headers = {
        'x-rapidapi-host': API_FOOTBALL_HOST,
        'x-rapidapi-key': API_FOOTBALL_KEY
    }
    
    # Get today's fixtures
    today = datetime.now().strftime('%Y-%m-%d')
    url = f"https://{API_FOOTBALL_HOST}/fixtures"
    params = {'date': today, 'timezone': 'Europe/London'}
    
    response = requests.get(url, headers=headers, params=params)
    data = response.json()
    
    if data['response']:
        # Process and store in database
        conn = get_db_connection()
        cur = conn.cursor()
        
        for fixture in data['response']:
            cur.execute("""
                INSERT INTO fixtures (id, home_team, away_team, league, date, status)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                status = EXCLUDED.status
            """, (
                fixture['fixture']['id'],
                fixture['teams']['home']['name'],
                fixture['teams']['away']['name'],
                fixture['league']['name'],
                fixture['fixture']['date'],
                fixture['fixture']['status']['short']
            ))
        
        conn.commit()
        cur.close()
        conn.close()
        
        return {"fixtures_fetched": len(data['response']), "date": today}
    return {"fixtures_fetched": 0}

def identify_value_bets():
    """Core function to identify value bets by comparing predictions with odds"""
    conn = get_db_connection()
    
    # Load trained models
    try:
        winner_model = joblib.load('models/winner_model.pkl')
        over_under_model = joblib.load('models/over_under_model.pkl')
        btts_model = joblib.load('models/btts_model.pkl')
    except:
        return []  # Models not trained yet
    
    # Fetch upcoming fixtures with odds
    query = """
    SELECT f.*, o.home_win_odds, o.draw_odds, o.away_win_odds, 
           o.over_25_odds, o.under_25_odds, o.btts_yes_odds, o.btts_no_odds
    FROM fixtures f
    LEFT JOIN odds o ON f.id = o.fixture_id
    WHERE f.date >= NOW() AND f.date <= NOW() + INTERVAL '3 days'
    AND f.status = 'NS'
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        return []
    
    # Feature engineering (simplified - you'd expand this)
    # This is where you'd create features from historical data
    features = create_features(df)
    
    # Get predictions from all models
    df['pred_home_win'] = winner_model.predict_proba(features)[:, 1]
    df['pred_over_25'] = over_under_model.predict_proba(features)[:, 1]
    df['pred_btts_yes'] = btts_model.predict_proba(features)[:, 1]
    
    value_bets = []
    
    # Calculate Expected Value for each market
    for _, row in df.iterrows():
        # Home win value
        if row['home_win_odds'] and row['home_win_odds'] > 0:
            implied_prob = 1 / row['home_win_odds']
            ev = (row['pred_home_win'] * (row['home_win_odds'] - 1)) - (1 - row['pred_home_win'])
            if ev > 0.05:  # 5% edge threshold
                value_bets.append({
                    'fixture_id': row['id'],
                    'market': 'HOME_WIN',
                    'predicted_prob': float(row['pred_home_win']),
                    'odds': float(row['home_win_odds']),
                    'implied_prob': float(implied_prob),
                    'edge': float(row['pred_home_win'] - implied_prob),
                    'ev': float(ev),
                    'recommended_stake': calculate_stake(ev)
                })
        
        # Over 2.5 goals value
        if row['over_25_odds'] and row['over_25_odds'] > 0:
            implied_prob = 1 / row['over_25_odds']
            ev = (row['pred_over_25'] * (row['over_25_odds'] - 1)) - (1 - row['pred_over_25'])
            if ev > 0.05:
                value_bets.append({
                    'fixture_id': row['id'],
                    'market': 'OVER_25',
                    'predicted_prob': float(row['pred_over_25']),
                    'odds': float(row['over_25_odds']),
                    'implied_prob': float(implied_prob),
                    'edge': float(row['pred_over_25'] - implied_prob),
                    'ev': float(ev),
                    'recommended_stake': calculate_stake(ev)
                })
    
    # Sort by EV and get top 3
    value_bets.sort(key=lambda x: x['ev'], reverse=True)
    
    # Send alerts for top value bets
    for bet in value_bets[:3]:
        send_bet_alert(bet)
    
    return value_bets[:5]  # Return top 5 for API response

def send_bet_alert(bet):
    """Send value bet alert to Telegram"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    
    message = f"🎯 *VALUE BET ALERT*\n\n"
    message += f"⚽ {bet.get('home_team', 'Home')} vs {bet.get('away_team', 'Away')}\n"
    message += f"📊 Market: {bet['market']}\n"
    message += f"🔢 Odds: {bet['odds']:.2f}\n"
    message += f"📈 Our Probability: {bet['predicted_prob']:.1%}\n"
    message += f"🏦 Implied Probability: {bet['implied_prob']:.1%}\n"
    message += f"✅ Edge: {bet['edge']:.2%}\n"
    message += f"💰 Expected Value: {bet['ev']:.2%}\n"
    message += f"💵 Recommended Stake: {bet['recommended_stake']} units"
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message,
        'parse_mode': 'Markdown'
    }
    
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Failed to send Telegram alert: {e}")

def calculate_stake(ev, bankroll=100, max_stake=5):
    """Calculate stake using Kelly Criterion (fractional)"""
    kelly_fraction = 0.1  # Use 10% of full Kelly for safety
    stake = (ev * bankroll * kelly_fraction) / 100
    return min(max(1, round(stake, 2)), max_stake)

def create_features(df):
    """Create features for model prediction (simplified example)"""
    # This is a placeholder - you would expand this significantly
    # with historical data, team form, injuries, etc.
    features = pd.DataFrame()
    
    # Simple placeholder features
    features['is_home'] = 1  # Would be derived from actual data
    features['form_diff'] = 0.1  # Form difference
    
    return features

def generate_daily_digest():
    """Generate daily performance digest"""
    conn = get_db_connection()
    
    query = """
    SELECT 
        COUNT(*) as total_bets,
        SUM(CASE WHEN won THEN 1 ELSE 0 END) as wins,
        SUM(CASE WHEN won THEN stake * (odds - 1) ELSE -stake END) as pnl,
        AVG(CASE WHEN won THEN odds ELSE NULL END) as avg_winning_odds
    FROM bets
    WHERE bet_date >= CURRENT_DATE - INTERVAL '7 days'
    """
    
    cur = conn.cursor()
    cur.execute(query)
    result = cur.fetchone()
    cur.close()
    conn.close()
    
    digest = f"📊 *7-Day Performance Digest*\n\n"
    digest += f"Total Bets: {result[0]}\n"
    digest += f"Win Rate: {result[1]/result[0]*100 if result[0] > 0 else 0:.1f}%\n"
    digest += f"Total P&L: {result[2]:.2f} units\n"
    digest += f"Avg Winning Odds: {result[3]:.2f if result[3] else 'N/A'}\n"
    
    return digest

def check_system_health():
    """Check health of all system components"""
    health = {
        'database': False,
        'api_football': False,
        'models_loaded': False,
        'telegram_bot': False,
        'timestamp': datetime.now().isoformat()
    }
    
    # Check database
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.close()
        conn.close()
        health['database'] = True
    except:
        health['database'] = False
    
    # Check API-Football
    try:
        headers = {'x-rapidapi-key': API_FOOTBALL_KEY, 'x-rapidapi-host': API_FOOTBALL_HOST}
        response = requests.get(f"https://{API_FOOTBALL_HOST}/status", headers=headers, timeout=5)
        health['api_football'] = response.status_code == 200
    except:
        health['api_football'] = False
    
    # Check models
    try:
        for model_file in ['winner_model.pkl', 'over_under_model.pkl', 'btts_model.pkl']:
            joblib.load(f'models/{model_file}')
        health['models_loaded'] = True
    except:
        health['models_loaded'] = False
    
    # Check Telegram bot
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        health['telegram_bot'] = True
    
    health['overall'] = all([health['database'], health['api_football'], health['models_loaded']])
    
    return health

# ========== SCHEDULED TASKS ==========
def run_scheduled_tasks():
    """Background thread for scheduled tasks"""
    
    # Schedule daily tasks
    schedule.every().day.at("08:00").do(fetch_live_fixtures)
    schedule.every().day.at("09:00").do(identify_value_bets)
    schedule.every().day.at("18:00").do(lambda: send_telegram_message(generate_daily_digest()))
    
    # Schedule weekly training
    schedule.every().sunday.at("03:00").do(manual_train)
    
    while True:
        schedule.run_pending()
        time.sleep(60)

# ========== MAIN EXECUTION ==========
if __name__ == '__main__':
    # Start background scheduler in a separate thread
    scheduler_thread = threading.Thread(target=run_scheduled_tasks, daemon=True)
    scheduler_thread.start()
    
    # Start Flask server
    print("Starting Football Prediction System...")
    print("Manual control routes available:")
    print("  GET /manual/scan     - Trigger live data scan")
    print("  GET /manual/train    - Trigger model training")
    print("  GET /manual/digest   - Generate daily digest")
    print("  GET /manual/autotune - Auto-tune models")
    print("  GET /manual/backfill - Backfill historical data")
    print("  GET /manual/health   - System health check")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
