import schedule
import time
import threading
from datetime import datetime, timedelta
from data_fetcher import fetch_fixtures, fetch_team_form
from model_manager import load_model, update_models
from predictor import generate_prediction, format_predictions
from telegram_bot import send_message
from config import LEAGUE_ID, SEASON
from utils import logger

def select_motd_match(fixtures):
    if not fixtures:
        return None
    # Customize selection here; for now first fixture
    return fixtures[0]

def send_match_of_the_day():
    today = datetime.utcnow().date()
    tomorrow = today + timedelta(days=1)

    fixtures = fetch_fixtures(status='NS', from_date=today.isoformat(), to_date=tomorrow.isoformat())

    if not fixtures:
        logger.info("No fixtures for MOTD.")
        return

    motd = select_motd_match(fixtures)
    home = motd['teams']['home']
    away = motd['teams']['away']
    fixture_date = motd['fixture']['date']

    home_stats = fetch_team_form(home['id'], fixture_date)
    away_stats = fetch_team_form(away['id'], fixture_date)
    if home_stats is None or away_stats is None:
        logger.info("No data for MOTD prediction.")
        return

    model_res, le_res = load_model('model_result.pkl')
    model_ou, le_ou = load_model('model_over_under.pkl')
    model_btts, le_btts = load_model('model_btts.pkl')

    if not all([model_res, le_res, model_ou, le_ou, model_btts, le_btts]):
        logger.info("Models missing for MOTD; retraining...")
        past_fixtures = fetch_fixtures('FT', from_date=(today - timedelta(days=365)).isoformat(), to_date=today.isoformat())
        from feature_engineering import prepare_dataset
        df = prepare_dataset(past_fixtures)
        if df.empty:
            logger.error("No data for training models.")
            return
        update_models(df)
        model_res, le_res = load_model('model_result.pkl')
        model_ou, le_ou = load_model('model_over_under.pkl')
        model_btts, le_btts = load_model('model_btts.pkl')

    preds = {
        'outcome': generate_prediction(model_res, le_res, home_stats, away_stats),
        'over_under': generate_prediction(model_ou, le_ou, home_stats, away_stats),
        'btts': generate_prediction(model_btts, le_btts, home_stats, away_stats)
    }

    message = (
        f"🏆 Match of the Day 🏆\n"
        f"{home['name']} vs {away['name']}\n"
        f"Date: {fixture_date}\n\n"
        f"{format_predictions(preds)}"
    )
    send_message(message)

def send_all_predictions():
    today = datetime.utcnow().date()
    fixtures = fetch_fixtures(status='NS', from_date=today.isoformat(), to_date=(today + timedelta(days=7)).isoformat())
    if not fixtures:
        logger.info("No upcoming fixtures for tips.")
        return

    model_res, le_res = load_model('model_result.pkl')
    model_ou, le_ou = load_model('model_over_under.pkl')
    model_btts, le_btts = load_model('model_btts.pkl')

    if not all([model_res, le_res, model_ou, le_ou, model_btts, le_btts]):
        logger.info("Models missing; retraining...")
        past_fixtures = fetch_fixtures('FT', from_date=(today - timedelta(days=365)).isoformat(), to_date=today.isoformat())
        from feature_engineering import prepare_dataset
        df = prepare_dataset(past_fixtures)
        if df.empty:
            logger.error("No data for training models.")
            return
        update_models(df)
        model_res, le_res = load_model('model_result.pkl')
        model_ou, le_ou = load_model('model_over_under.pkl')
        model_btts, le_btts = load_model('model_btts.pkl')

    for fix in fixtures:
        home = fix['teams']['home']
        away = fix['teams']['away']
        fixture_date = fix['fixture']['date']

        home_stats = fetch_team_form(home['id'], fixture_date)
        away_stats = fetch_team_form(away['id'], fixture_date)
        if home_stats is None or away_stats is None:
            continue

        preds = {
            'outcome': generate_prediction(model_res, le_res, home_stats, away_stats),
            'over_under': generate_prediction(model_ou, le_ou, home_stats, away_stats),
            'btts': generate_prediction(model_btts, le_btts, home_stats, away_stats)
        }

        message = (
            f"Fixture: {home['name']} vs {away['name']}\n"
            f"Date: {fixture_date}\n"
            f"{format_predictions(preds)}"
        )
        send_message(message)

def start_scheduler():
    schedule.every().day.at("08:00").do(send_match_of_the_day)
    schedule.every().day.at("08:10").do(send_all_predictions)  # example, 10 minutes later

    def run_loop():
        while True:
            schedule.run_pending()
            time.sleep(30)

    import threading
    threading.Thread(target=run_loop, daemon=True).start()
    logger.info("Scheduler started.")
