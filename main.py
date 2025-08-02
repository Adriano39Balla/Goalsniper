# main.py
from flask import Flask, jsonify
from predictions import run_daily_predictions

app = Flask(__name__)

@app.route("/predict", methods=["GET"])
def predict():
    """
    Returns daily predictions in JSON format.
    Never crashes the worker.
    """
    result = run_daily_predictions()
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
