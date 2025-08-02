# betting_predictions.py
import os
import logging
import requests
import certifi

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

API_KEY = os.getenv("API_KEY")

def run_daily_predictions():
    """
    Runs daily football predictions using API-Football.
    Logs errors instead of killing the worker.
    """
    try:
        logger.info("Running daily predictions...")

        url = "https://v3.football.api-sports.io/fixtures"
        params = {"date": "2025-08-02"}  # Example: today's date
        headers = {"x-apisports-key": API_KEY}

        # Use certifi to avoid SSL issues
        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=15,
            verify=certifi.where()
        )
        response.raise_for_status()

        data = response.json()
        logger.info(f"API-Football response: {data}")
        return {"success": True, "data": data}

    except requests.exceptions.SSLError as ssl_err:
        logger.error(f"SSL error during API-Football request: {ssl_err}", exc_info=True)
        return {"success": False, "error": "SSL error contacting API-Football"}
    except requests.exceptions.RequestException as req_err:
        logger.error(f"HTTP error during API-Football request: {req_err}", exc_info=True)
        return {"success": False, "error": "HTTP error contacting API-Football"}
    except Exception as e:
        logger.error(f"Unexpected error in run_daily_predictions: {e}", exc_info=True)
        return {"success": False, "error": "Unexpected error in predictions"}
