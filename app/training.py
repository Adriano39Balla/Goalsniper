# app/training.py
import json
import logging
import subprocess
import sys
from datetime import datetime
from typing import Dict, Any, Optional

from app.db import db_conn


def run_training(export_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Trigger the training pipeline by calling train_models.py as a subprocess.
    Returns parsed JSON summary of the training run.
    """
    cmd = [sys.executable, "train_models.py", "--db", "tip_performance.db"]

    if export_path:
        cmd.extend(["--export-path", export_path])

    logging.info(f"[TRAIN] running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        stdout = result.stdout.strip()

        # Last line of stdout should be JSON summary (from train_models.py)
        lines = stdout.splitlines()
        last_line = lines[-1] if lines else "{}"
        try:
            summary = json.loads(last_line)
        except Exception:
            logging.warning("[TRAIN] failed to parse summary JSON, fallback to empty.")
            summary = {"ok": False, "raw": stdout}

        # Save into models table (for versioning)
        _persist_model_version(summary)

        return summary
    except subprocess.CalledProcessError as e:
        logging.error(f"[TRAIN] subprocess failed: {e.stderr}")
        return {"ok": False, "error": e.stderr}
    except Exception as e:
        logging.exception("[TRAIN] error")
        return {"ok": False, "error": str(e)}


def _persist_model_version(summary: Dict[str, Any]):
    """
    Persist training result into models table for version history.
    Keeps settings.model_coeffs as well for backward compatibility.
    """
    if not summary.get("ok"):
        return

    trained_at = summary.get("trained_at_utc", datetime.utcnow().isoformat())
    metrics = summary.get("metrics") or {}
    counts = summary.get("counts") or {}

    # Insert into models table
    with db_conn() as conn:
        try:
            conn.execute("""
                INSERT INTO models(market, trained_at, coeffs, metrics, hyperparams, active)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                "all",  # placeholder, you can later split O25 vs BTTS
                trained_at,
                json.dumps(summary, ensure_ascii=False),
                json.dumps(metrics, ensure_ascii=False),
                json.dumps({"counts": counts}, ensure_ascii=False),
                1
            ))
            conn.commit()
            logging.info(f"[TRAIN] persisted model version at {trained_at}")
        except Exception as e:
            logging.exception(f"[TRAIN] persist failed: {e}")
