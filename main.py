from flask import Flask, jsonify
from predictions import run_daily_predictions

app = Flask(__name__)

@app.route("/predict", methods=["GET"])
def predict():
    result = run_daily_predictions()
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
