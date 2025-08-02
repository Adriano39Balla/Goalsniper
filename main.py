from flask import Flask, jsonify
from predictions import run_daily_predictions
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

@app.route("/predict", methods=["GET"])
def predict():
    """
    Runs daily football predictions and returns JSON response.
    Also sends predictions to Telegram via predictions.py.
    """
    try:
        result = run_daily_predictions()
        return jsonify(result), 200 if result.get("success") else 500
    except Exception as e:
        logging.error(f"Unexpected error in /predict endpoint: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    # Use port from Render's environment or default to 10000
    import os
    port = int(os.getenv("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
