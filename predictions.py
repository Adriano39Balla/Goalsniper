import os
import logging
import requests
import certifi
from telegram import send_telegram_message
from datetime import date
from requests.adapters import HTTPAdapter, Retry

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

API_KEY = os.getenv("API_KEY")

def run_daily_predictions():
    """
    Runs daily football predictions using API-Football.
    Logs errors instead of killing the worker.
    """
    # Validate API key
    if not API_KEY:
        logger.error("API_KEY environment variable not set.")
        return {"success": False, "error": "API key missing", "data": None}

    try:
        logger.info("Running daily predictions...")

        # API endpoint & params
        url = "https://v3.football.api-sports.io/fixtures"
        params = {"date": date.today().isoformat()}  # Dynamic today's date
        headers = {"x-apisports-key": API_KEY}

        # Session with retry logic
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        session.mount("https://", HTTPAdapter(max_retries=retries))

        # Make the request
        response = session.get(
            url,
            headers=headers,
            params=params,
            timeout=15,
            verify=certifi.where()
        )
        response.raise_for_status()

        # Parse response
        data = response.json()

        # Log summary instead of dumping all data
        match_count = len(data.get("response", []))
        logger.info(f"Received {match_count} matches from API-Football")

        return {"success": True, "data": data}

    except requests.exceptions.SSLError as ssl_err:
        logger.error(f"SSL error during API-Football request: {ssl_err}", exc_info=True)
        return {"success": False, "error": "SSL error contacting API-Football", "data": None}
    except requests.exceptions.RequestException as req_err:
        logger.error(f"HTTP error during API-Football request: {req_err}", exc_info=True)
        return {"success": False, "error": "HTTP error contacting API-Football", "data": None}
    except Exception as e:
        logger.error(f"Unexpected error in run_daily_predictions: {e}", exc_info=True)
        return {"success": False, "error": "Unexpected error in predictions", "data": None}

@app.route("/predict", methods=["GET"])
def predict():
    result = run_daily_predictions()
    send_telegram_message(f"Daily prediction run result: {result}")
    return jsonify(result)
