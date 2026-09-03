"""
_pick_threshold() logs "[THRESHOLD] no threshold beat base rate ... SUPPRESSING
this market" whenever a market never earns a bettable threshold on the
calibration split - the same event a real training run hits for every head
with no signal. That line carried no market/head identifier at all, so an
operator (or a log-parsing script) reading it could not tell WHICH market had
just been suppressed - the only clue in the whole run. Every other gate line
in this file (e.g. [HOLDOUT]) names the head it is about; this one silently
didn't.
"""
import logging

import numpy as np

from train_models import _pick_threshold


def test_suppression_log_line_names_the_head(caplog):
    # No signal at all: labels are pure noise relative to p, so no threshold
    # can reach the (noise-floor) target precision at min_preds=5 - this always
    # falls through to the "suppressed" branch.
    rng = np.random.default_rng(0)
    y = (rng.random(200) < 0.5).astype(int)
    p = rng.random(200)

    with caplog.at_level(logging.WARNING):
        thr, diag = _pick_threshold(y, p, target_precision=0.99, min_preds=5,
                                    default_threshold=0.5, max_thresh_pct=85.0,
                                    label="PRE_OU_3.5")

    assert diag["method"] == "suppressed"
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("PRE_OU_3.5" in msg and "SUPPRESSING" in msg for msg in warnings), (
        f"suppression warning does not name the head it is about: {warnings}")


def test_label_defaults_to_a_placeholder_when_the_caller_omits_it():
    # Every real call site threads a label through (see _decide_threshold), but
    # the parameter must not be required - a caller that forgets it should not
    # crash training, only lose attribution on that one log line.
    rng = np.random.default_rng(1)
    y = (rng.random(50) < 0.5).astype(int)
    p = rng.random(50)
    thr, diag = _pick_threshold(y, p, target_precision=0.99, min_preds=5,
                                default_threshold=0.5, max_thresh_pct=85.0)
    assert diag["method"] == "suppressed"
