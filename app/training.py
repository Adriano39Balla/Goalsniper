"""
app/training.py

Wrapper around train_models.py for Flask routes.
Runs training via subprocess and returns results.
"""

import subprocess
import json
from flask import jsonify

def run_training():
    """Run train_models.py as subprocess and parse JSON output."""
    try:
        result = subprocess.run(
            ["python", "train_models.py", "--db", "tip_performance.db"],
            capture_output=True, text=True, check=True
        )
        # Logs from training (stdout)
        logs = result.stdout.strip().splitlines()

        # Last line should be JSON summary
        summary = {}
        try:
            summary = json.loads(logs[-1])
        except json.JSONDecodeError:
            # fallback: training didn’t output JSON
            summary = {"ok": False, "error": "Could not parse training output"}

        return {
            "ok": True,
            "logs": logs,
            "summary": summary
        }

    except subprocess.CalledProcessError as e:
        return {
            "ok": False,
            "error": e.stderr or str(e)
        }

# Flask route handler
def training_route():
    result = run_training()
    return jsonify(result)
