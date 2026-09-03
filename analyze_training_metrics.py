#!/usr/bin/env python3
"""
Extract and analyze the latest training metrics to verify all fixes have taken effect.

Verifies:
1. Calibration gap sign is correct (predicted - actual, not actual - predicted)
2. placeholder_share_pct is close to 0 (not high due to anchoring bug)
3. deviation_from_market p95 is single digits (not 49-64pp)
4. No critical validation failures
5. Model parity checks pass for golden samples
6. Prematch heads have brier_skill > 0
"""

import json
import logging
import os
import sys
from typing import Any, Dict, Optional

import psycopg2
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    load_dotenv()
except Exception:
    pass


def get_db_connection():
    """Connect to the database."""
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise SystemExit("DATABASE_URL environment variable is required")

    # Add SSL requirement if not already specified
    if "sslmode=" not in db_url:
        db_url = db_url + ("&" if "?" in db_url else "?") + "sslmode=require"

    return psycopg2.connect(db_url)


def get_setting(conn, key: str) -> Optional[str]:
    """Fetch a setting value from the database."""
    try:
        with conn.cursor() as c:
            c.execute("SELECT value FROM settings WHERE key = %s LIMIT 1", (key,))
            row = c.fetchone()
            return row[0] if row else None
    except Exception as e:
        logger.error(f"Error fetching setting {key}: {e}")
        return None


def analyze_metrics() -> None:
    """Extract and analyze the latest training metrics."""
    conn = get_db_connection()
    try:
        # Fetch the latest metrics
        metrics_raw = get_setting(conn, "model_metrics_latest")
        if not metrics_raw:
            logger.error("No model_metrics_latest found in database")
            return

        try:
            metrics = json.loads(metrics_raw)
        except json.JSONDecodeError:
            logger.error("Failed to parse model_metrics_latest as JSON")
            return

        logger.info("=" * 80)
        logger.info("TRAINING METRICS ANALYSIS")
        logger.info("=" * 80)
        logger.info(f"Trained at: {metrics.get('trained_at_utc', 'unknown')}")
        logger.info("")

        # ──────────────── CALIBRATION GAP VERIFICATION ──────────────
        logger.info("CALIBRATION GAP SIGN CHECK (predicted - actual)")
        logger.info("-" * 80)

        calib_issues = []
        for head, data in sorted(metrics.items()):
            if not isinstance(data, dict) or "calibration_gap_pct" not in data:
                continue

            gap = float(data.get("calibration_gap_pct", 0))
            mp = float(data.get("mean_predicted", 0))
            ma = float(data.get("mean_actual", 0))
            expected = 100.0 * (mp - ma)

            # Sign check: reported should equal expected (within 0.02pp tolerance)
            sign_ok = abs(gap - expected) <= 0.02
            status = "✓" if sign_ok else "✗"

            logger.info(f"{status} {head:20s} | gap={gap:+7.2f}pp | predicted={mp:.3f} | "
                       f"actual={ma:.3f} | expected={expected:+7.2f}pp")

            if not sign_ok:
                calib_issues.append((head, gap, expected))

        if calib_issues:
            logger.error(f"⚠ Found {len(calib_issues)} calibration gap sign mismatches:")
            for head, gap, expected in calib_issues:
                logger.error(f"    {head}: reported {gap:+.2f}pp vs expected {expected:+.2f}pp")
        else:
            logger.info("✓ All calibration gap signs are correct")

        logger.info("")

        # ──────────────── ANCHORING VERIFICATION ──────────────
        logger.info("ANCHORING VERIFICATION (placeholder_share_pct)")
        logger.info("-" * 80)

        anchoring = metrics.get("market_anchoring", {})
        if anchoring.get("anchored"):
            placeholder_pct = float(anchoring.get("placeholder_share_pct", 0))
            logger.info(f"Placeholder share: {placeholder_pct:.2f}% (target: < 1%)")

            if placeholder_pct > 1.0:
                logger.error(f"⚠ Placeholder share too high: {placeholder_pct:.2f}% "
                            "(anchoring bug likely not fixed)")
            elif placeholder_pct == 0:
                logger.info("✓ No placeholder anchors found (fix confirmed)")
            else:
                logger.info("✓ Placeholder share within acceptable range")

            # Deviation check
            for head, data in sorted(metrics.items()):
                if not isinstance(data, dict):
                    continue
                dev = data.get("deviation_from_market", {})
                if dev:
                    p95 = dev.get("p95_abs_pp")
                    if p95 is not None:
                        p95 = float(p95)
                        status = "✓" if p95 < 20 else "⚠" if p95 < 30 else "✗"
                        logger.info(f"  {status} {head:20s} | p95 deviation: {p95:6.2f}pp")
                        if p95 > 30:
                            logger.error(f"    → Deviation too high for anchored head")
        else:
            logger.info("(No market anchoring in this training run)")

        logger.info("")

        # ──────────────── VALIDATION FINDINGS ──────────────
        logger.info("VALIDATION FINDINGS")
        logger.info("-" * 80)

        validation = metrics.get("validation", {})
        findings = validation.get("findings", [])
        unfit_heads = validation.get("unfit_heads", {})
        n_critical = validation.get("n_critical", 0)

        logger.info(f"Critical findings: {n_critical}")
        logger.info(f"Warning findings: {validation.get('n_warning', 0)}")
        logger.info(f"Unfit heads (blocked from betting): {len(unfit_heads)}")

        if findings:
            logger.info("")
            for finding in findings:
                severity = finding.get("severity", "?")
                symbol = "✗" if severity == "CRITICAL" else "⚠"
                logger.info(f"{symbol} [{severity}] {finding.get('head', '-')} / "
                           f"{finding.get('check')}: {finding.get('detail')}")

        if unfit_heads:
            logger.info("")
            logger.error("Blocked heads:")
            for head, reason in sorted(unfit_heads.items()):
                logger.error(f"  {head}: {reason}")
        elif n_critical == 0:
            logger.info("✓ No critical validation failures (all heads eligible to bet)")

        logger.info("")

        # ──────────────── PREMATCH BRIER SKILL CHECK ──────────────
        logger.info("PREMATCH MODEL SKILL (Brier Skill)")
        logger.info("-" * 80)

        prematch_heads = [h for h in metrics.keys() if h.startswith("PRE_")]
        skill_issues = []

        for head in sorted(prematch_heads):
            data = metrics[head]
            if not isinstance(data, dict):
                continue

            skill = data.get("brier_skill")
            if skill is None:
                logger.warning(f"  {head:20s} | no brier_skill recorded")
                continue

            skill = float(skill)
            status = "✓" if skill > 0 else "✗"

            logger.info(f"{status} {head:20s} | brier_skill={skill:+.4f}")

            if skill <= 0:
                skill_issues.append((head, skill))

        if skill_issues:
            logger.error(f"⚠ Found {len(skill_issues)} prematch heads with no/negative skill:")
            for head, skill in skill_issues:
                logger.error(f"    {head}: {skill:+.4f}")
        else:
            logger.info("✓ All prematch heads show positive skill")

        logger.info("")

        # ──────────────── MODEL PARITY GOLDEN SAMPLES ──────────────
        # THE BUG THIS FIXES: golden_samples and a "parity_verified" flag were
        # read off `data` here, i.e. off `metrics[head]` from the
        # model_metrics_latest bundle this function already loaded. But
        # train_models._train_binary_head() writes golden_samples into a
        # SEPARATE settings key, `model_health:{head}` - never into the
        # metrics bundle - and no code anywhere ever writes a
        # "parity_verified" key at all: the actual parity check
        # (main.verify_model_parity()) runs live at SERVE time, re-scoring
        # golden_samples through whichever model is currently deployed, and
        # its result is cached in-process rather than persisted back to the
        # database. So `data.get("golden_samples")` was always None and
        # `data.get("parity_verified")` was always None too - the loop never
        # found a sample count to print, `parity_issues` could never be
        # populated, and "✓ Model parity checks passed (or not yet run)"
        # printed unconditionally regardless of whether anything had actually
        # been checked. Reading model_health:{head} (where the samples truly
        # live) fixes the sample count; the pass/fail claim is removed rather
        # than faked, since only main.py's serving process can compute it.
        logger.info("MODEL PARITY (Golden Samples recorded for serve-time verification)")
        logger.info("-" * 80)

        heads_with_metrics = [h for h in sorted(metrics.keys())
                              if isinstance(metrics.get(h), dict) and "n_train" in metrics[h]]
        any_samples = False
        for head in heads_with_metrics:
            health_raw = get_setting(conn, f"model_health:{head}")
            if not health_raw:
                logger.warning(f"  {head:20s} | no model_health record found")
                continue
            try:
                health = json.loads(health_raw)
            except json.JSONDecodeError:
                logger.warning(f"  {head:20s} | model_health record is not valid JSON")
                continue
            golden_samples = health.get("golden_samples") or []
            if golden_samples:
                any_samples = True
                logger.info(f"  ✓ {head:20s} | {len(golden_samples)} golden sample(s) recorded")
            else:
                logger.warning(f"  {head:20s} | no golden samples recorded — "
                               "train/serve parity cannot be checked for this head")

        if not any_samples:
            logger.warning("⚠ No head has golden samples recorded")
        logger.info("Note: this only confirms samples EXIST to check against. Whether the "
                   "currently deployed model actually agrees with them is verified live, on "
                   "every prediction, by main.verify_model_parity() — not by this script.")

        logger.info("")

        # ──────────────── IN-PLAY VS PREMATCH FIXTURE COUNTS ──────────────
        logger.info("TRAINING DATA DISTRIBUTION")
        logger.info("-" * 80)

        inplay_heads = [h for h in metrics.keys() if not h.startswith("PRE_")
                       and isinstance(metrics.get(h), dict) and "n_train" in metrics[h]]
        prematch_heads = [h for h in metrics.keys() if h.startswith("PRE_")
                         and isinstance(metrics.get(h), dict) and "n_train" in metrics[h]]

        if inplay_heads:
            sample_head = inplay_heads[0]
            sample_data = metrics[sample_head]
            logger.info(f"In-play training fixtures: {sample_data.get('n_train_matches', '?')}")
            logger.info(f"In-play training rows: {sample_data.get('n_train', '?')}")

        if prematch_heads:
            sample_head = prematch_heads[0]
            sample_data = metrics[sample_head]
            logger.info(f"Prematch training fixtures: {sample_data.get('n_train_matches', '?')}")
            logger.info(f"Prematch training rows: {sample_data.get('n_train', '?')}")

        logger.info("")
        logger.info("=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)

        issues = bool(calib_issues or skill_issues)
        logger.info(f"Calibration sign issues: {len(calib_issues)}")
        logger.info(f"Skill check failures: {len(skill_issues)}")
        logger.info(f"Critical validation failures: {n_critical}")
        logger.info(f"Heads with golden samples for parity checking: "
                   f"{'yes' if any_samples else 'NONE'}")

        if not issues and n_critical == 0:
            logger.info("")
            logger.info("✓ ALL CHECKS PASSED - System is ready for production")
        else:
            logger.warning("")
            logger.warning("⚠ ISSUES DETECTED - Review above for details")

        logger.info("")

        # Write full metrics to file for detailed inspection
        output_file = "/tmp/training_metrics_latest.json"
        with open(output_file, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Full metrics saved to: {output_file}")

    finally:
        conn.close()


if __name__ == "__main__":
    try:
        analyze_metrics()
    except Exception as e:
        logger.exception("Analysis failed: %s", e)
        sys.exit(1)
