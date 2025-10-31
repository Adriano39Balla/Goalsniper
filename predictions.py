# predictions.py

import logging
import os
import time
from typing import Dict, Any, Optional, Tuple

from core.db import db_conn
from core.football_api import get_today_fixture_ids
from core.snapshots import snapshot_odds_for_fixtures
from core.results import backfill_results_for_open_matches
from core.models import predict_all_matches

log = logging.getLogger(__name__)

# Constants
SCAN_INTERVAL_SEC = int(os.getenv("SCAN_INTERVAL_SEC", "90"))
BACKFILL_EVERY_MIN = int(os.getenv("BACKFILL_EVERY_MIN", "7"))
RUN_SCHEDULER = os.getenv("RUN_SCHEDULER", "1") not in ("0", "false", "False", "no", "NO")


def production_scan() -> Tuple[int, int]:
    """Run prediction scan and save results."""
    try:
        saved, live_seen = predict_all_matches()
        log.info("[SCAN] Saved %d predictions; live seen: %d", saved, live_seen)
        return saved, live_seen
    except Exception as e:
        log.exception("[SCAN] failed: %s", e)
        return 0, 0


def prematch_scan_save() -> int:
    """Save odds snapshots for today's matches."""
    try:
        fixture_ids = get_today_fixture_ids()
        snapshot_odds_for_fixtures(fixture_ids)
        return len(fixture_ids)
    except Exception as e:
        log.exception("[PREMATCH SCAN] failed: %s", e)
        return 0


def _apply_tune_thresholds(days: int = 14) -> Dict[str, float]:
    """Private hook expected by auto_tune_thresholds()."""
    try:
        from train_models import auto_tune_thresholds
        with db_conn() as conn:
            return auto_tune_thresholds(conn, days)
    except Exception as e:
        log.warning("[AUTO-TUNE] _apply_tune_thresholds failed: %s", e)
        return {}


def auto_tune_thresholds(days: int = 14) -> Dict[str, float]:
    """Safe wrapper for applying thresholds, exposed to Flask controller."""
    try:
        return _apply_tune_thresholds(days)
    except Exception as e:
        log.exception("[AUTO-TUNE] failed: %s", e)
        return {}
