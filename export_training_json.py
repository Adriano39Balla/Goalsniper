#!/usr/bin/env python3
"""
Export the full training metrics JSON that was saved during the last training run.

This script connects to your Postgres database and exports the complete
model_metrics_latest JSON that contains all validation, golden sample,
and detailed metric information.

Usage:
    python3 export_training_json.py

The full metrics JSON will be saved to: training_metrics_export.json

Key fields to verify:
- marker_anchoring.placeholder_share_pct (should be < 1%)
- <head>.deviation_from_market.p95_abs_pp (should be < 20pp)
- <head>.calibration_gap_pct (sign should be correct: predicted - actual)
- validation.findings (should be empty or minimal)
- <head>.golden_samples (re-scored samples for parity verification)
- <head>.brier_skill (should be > 0 for prematch)
"""

import json
import logging
import os
import sys
from typing import Optional

import psycopg2
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    load_dotenv()
except Exception:
    pass


def get_db_connection():
    """Connect to the database using DATABASE_URL."""
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        logger.error("DATABASE_URL environment variable not set")
        logger.error("")
        logger.error("Set it via:")
        logger.error("  export DATABASE_URL='postgresql://user:pass@host/db'")
        logger.error("")
        raise SystemExit(1)

    # Add SSL requirement if not specified
    if "sslmode=" not in db_url:
        db_url = db_url + ("&" if "?" in db_url else "?") + "sslmode=require"

    try:
        return psycopg2.connect(db_url)
    except psycopg2.Error as e:
        logger.error(f"Failed to connect to database: {e}")
        raise SystemExit(1)


def get_setting(conn, key: str) -> Optional[str]:
    """Fetch a single setting value."""
    try:
        with conn.cursor() as c:
            c.execute("SELECT value FROM settings WHERE key = %s LIMIT 1", (key,))
            row = c.fetchone()
            return row[0] if row else None
    except Exception as e:
        logger.error(f"Error fetching setting {key}: {e}")
        return None


def main():
    logger.info("Exporting training metrics...")
    logger.info("")

    conn = get_db_connection()

    try:
        # Fetch the latest training metrics
        metrics_raw = get_setting(conn, "model_metrics_latest")
        if not metrics_raw:
            logger.error("No model_metrics_latest found in database")
            logger.error("Has training been run yet?")
            return 1

        # Parse and validate JSON
        try:
            metrics = json.loads(metrics_raw)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse metrics JSON: {e}")
            return 1

        # Extract metadata
        trained_at = metrics.get("trained_at_utc", "unknown")
        logger.info(f"Training timestamp: {trained_at}")
        logger.info("")

        # Count validation findings
        validation = metrics.get("validation", {})
        n_critical = validation.get("n_critical", 0)
        n_warning = validation.get("n_warning", 0)

        logger.info(f"Validation: {n_critical} critical, {n_warning} warning")

        # Count unfit heads
        unfit_heads = validation.get("unfit_heads", {})
        if unfit_heads:
            logger.error(f"Unfit heads (blocked from betting):")
            for head, reason in sorted(unfit_heads.items()):
                logger.error(f"  {head}: {reason}")
        else:
            logger.info("No unfit heads (all models eligible to bet)")

        logger.info("")

        # Check key metrics
        anchoring = metrics.get("market_anchoring", {})
        if anchoring.get("anchored"):
            ph_pct = float(anchoring.get("placeholder_share_pct", 0))
            logger.info(f"Placeholder anchor share: {ph_pct:.2f}% (target: < 1%)")
            if ph_pct == 0:
                logger.info("  ✓ Fix confirmed: No placeholder anchors")
            elif ph_pct > 1:
                logger.error("  ⚠ Fix may not be working: High placeholder share")

        logger.info("")

        # Export to file
        output_file = "training_metrics_export.json"
        with open(output_file, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"✓ Full metrics exported to: {output_file}")
        logger.info("")
        logger.info("You can now inspect this file to verify:")
        logger.info("  1. Calibration gap signs are correct")
        logger.info("  2. Placeholder anchor share is near 0")
        logger.info("  3. Deviation from market p95 is < 20pp for anchored heads")
        logger.info("  4. No critical validation failures")
        logger.info("  5. Golden samples are recorded")
        logger.info("")

        return 0

    except Exception as e:
        logger.exception("Failed: %s", e)
        return 1

    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
