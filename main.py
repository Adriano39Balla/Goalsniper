from flask import Flask, jsonify
import logging
from predictions import run_live_predictions

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return "🤖 Robi Superbrain is monitoring live matches..."

@app.route("/predict", methods=["GET"])
def predict():
    try:
        logger.info("📡 Running live predictions...")
        run_live_predictions()
        return jsonify({"status": "success", "message": "Live predictions executed."})
    except Exception as e:
        logger.error(f"❌ Error running predictions: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
