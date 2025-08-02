from flask import Flask, jsonify
from flask_cors import CORS
from predictions import run_daily_predictions
import logging

# Flask app initialization
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests if API will be called from browsers

# Configure logging
logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)

@app.route("/predict", methods=["GET"])
def predict():
    """
    Returns daily predictions in JSON format.
    Never crashes the worker.
    """
    try:
        app.logger.info("Prediction request received")
        result = run_daily_predictions()
        return jsonify(result), 200 if result.get("success") else 500
    except Exception as e:
        app.logger.error(f"Unexpected error in /predict: {e}", exc_info=True)
        return jsonify({"success": False, "error": "Internal server error", "data": None}), 500

if __name__ == "__main__":
    # For development only. Use Gunicorn in production.
    app.run(host="0.0.0.0", port=10000, debug=False)
