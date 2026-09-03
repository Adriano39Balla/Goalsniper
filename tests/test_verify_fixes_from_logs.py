"""
verify_fixes_from_logs.py parses training logs to confirm a run behaved as
expected. Three defects in the parser itself meant it could report false
confidence:

1. `if "8717" in line` gated fixture-count extraction on one past run's exact
   prematch row count - a coincidence, not a signal. Any other run silently
   produced an empty "DATA DISTRIBUTION" section.
2. `re.search(r"[THRESHOLD]...")` used an unescaped character class instead of
   the literal tag, so it matched *something* on every suppression line
   (T/H/R/E/S/O/L/D all appear in "SUPPRESSING") and reported wrong text as
   the suppressed market's name rather than failing to find one.
3. `main()` printed a "Usage: ... [log_file]" line but never read
   `sys.argv`, so a log file passed on the command line was silently ignored
   in favour of one developer's own hardcoded path.
"""
import subprocess
import sys
from pathlib import Path

from verify_fixes_from_logs import parse_training_logs

SAMPLE_LOG = """
In-play: 1500 snapshots across 500 fixtures, 56 features
[SPLIT] fixtures: train=300 cal=100 holdout=100 (rows: 900/300/300)
Prematch: 9001 rows, 25 features
[SPLIT] fixtures: train=5401 cal=1800 holdout=1800 (rows: 5401/1800/1800)
[THRESHOLD] PRE_OU_3.5: no threshold beat base rate 0.500 by 3pp with >=100 selections (best was 0.5123). SUPPRESSING this market at 85.0%.
[VALIDATE] no critical findings
[SETTINGS] committed 42 keys atomically
"""


def _write_log(tmp_path, text=SAMPLE_LOG):
    p = tmp_path / "training.log"
    p.write_text(text)
    return str(p)


def test_prematch_fixture_counts_are_read_from_a_run_with_a_different_size(tmp_path):
    # The old "8717" literal would find nothing at all here, on a log shaped
    # exactly like a real one just with different row counts.
    results = parse_training_logs(_write_log(tmp_path))
    assert results["prematch_metrics"]["fixtures"] == {
        "train": 5401, "cal": 1800, "holdout": 1800,
    }


def test_inplay_split_is_not_mistaken_for_prematch(tmp_path):
    results = parse_training_logs(_write_log(tmp_path))
    # Only the prematch split (the one following a "Prematch:" line) should
    # land in prematch_metrics; the in-play split must not overwrite it.
    assert results["prematch_metrics"]["fixtures"]["train"] == 5401


def test_threshold_suppression_captures_the_real_head_name(tmp_path):
    results = parse_training_logs(_write_log(tmp_path))
    assert results["flags"]["prematch_suppressed"] == ["PRE_OU_3.5"]
    assert "SUPPRESSING" not in " ".join(results["flags"]["prematch_suppressed"])
    assert results["flags"]["inplay_suppressed"] == []


def test_cli_reads_the_log_path_from_argv(tmp_path):
    log_path = _write_log(tmp_path)
    out = subprocess.run(
        [sys.executable, "verify_fixes_from_logs.py", log_path],
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True, text=True, timeout=30,
    )
    assert f"Analyzing logs: {log_path}" in out.stdout
    assert "TRAINING VERIFICATION REPORT" in out.stdout
