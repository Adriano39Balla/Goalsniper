"""
logging.basicConfig(), with no `stream=`, installs a single StreamHandler on
sys.stderr for every level. Railway's log pipeline - like most hosted log
collectors - tags every stderr line severity=error, so plain INFO and
WARNING lines ("[PROD] saved=0 live_seen=14 ...", the per-minute rate-limit
notice, the xG-feed warning) all arrived tagged as errors, and a genuine
ERROR could not be told apart from routine scan output in the log view.

_setup_logging() replaces that with two handlers split at ERROR: INFO/WARNING
to stdout, ERROR/CRITICAL to stderr. These tests call it directly (rather
than relying on the one call already made at import time) and restore
whatever handlers were on the root logger before each test, since root
logger state is process-global and every other test's logger propagates
through it.
"""
import io
import json
import logging
import sys

import pytest

import main


@pytest.fixture
def _isolated_root_handlers():
    root = logging.getLogger()
    saved = list(root.handlers)
    saved_level = root.level
    yield root
    for h in list(root.handlers):
        root.removeHandler(h)
    for h in saved:
        root.addHandler(h)
    root.setLevel(saved_level)


def _capture(root, monkeypatch):
    """Swap the stdout/stderr handlers' streams for in-memory buffers."""
    out_buf, err_buf = io.StringIO(), io.StringIO()
    for h in root.handlers:
        if h.stream is sys.stdout:
            h.stream = out_buf
        elif h.stream is sys.stderr:
            h.stream = err_buf
    return out_buf, err_buf


def test_info_and_warning_go_to_stdout_not_stderr(_isolated_root_handlers, monkeypatch):
    monkeypatch.delenv("LOG_JSON", raising=False)
    root = _isolated_root_handlers
    main._setup_logging()
    out_buf, err_buf = _capture(root, monkeypatch)

    log = logging.getLogger("goalsniper")
    log.info("routine scan output")
    log.warning("xg feed absent on 2 fixtures")

    assert "routine scan output" in out_buf.getvalue()
    assert "xg feed absent on 2 fixtures" in out_buf.getvalue()
    assert out_buf.getvalue().count("\n") == 2
    assert err_buf.getvalue() == ""


def test_error_goes_to_stderr_not_stdout(_isolated_root_handlers, monkeypatch):
    monkeypatch.delenv("LOG_JSON", raising=False)
    root = _isolated_root_handlers
    main._setup_logging()
    out_buf, err_buf = _capture(root, monkeypatch)

    logging.getLogger("goalsniper").error("a genuine failure")

    assert "a genuine failure" in err_buf.getvalue()
    assert "a genuine failure" not in out_buf.getvalue()


def test_only_two_handlers_are_installed_not_accumulated(_isolated_root_handlers):
    # Calling setup twice (as a test suite naturally would) must not leave
    # duplicate handlers behind - that would double-print every line.
    root = _isolated_root_handlers
    main._setup_logging()
    main._setup_logging()
    assert len(root.handlers) == 2


def test_log_json_emits_one_json_object_per_line_with_a_level_field(
        _isolated_root_handlers, monkeypatch):
    monkeypatch.setenv("LOG_JSON", "1")
    root = _isolated_root_handlers
    main._setup_logging()
    out_buf, _err_buf = _capture(root, monkeypatch)

    logging.getLogger("goalsniper").info("structured line")

    line = out_buf.getvalue().strip()
    obj = json.loads(line)  # raises if it isn't valid JSON
    assert obj["level"] == "info"
    assert obj["message"] == "structured line"
    assert "timestamp" in obj


def test_log_json_0_keeps_the_original_text_format(_isolated_root_handlers, monkeypatch):
    monkeypatch.setenv("LOG_JSON", "0")
    root = _isolated_root_handlers
    main._setup_logging()
    out_buf, _err_buf = _capture(root, monkeypatch)

    logging.getLogger("goalsniper").info("plain text line")

    line = out_buf.getvalue().strip()
    with pytest.raises(json.JSONDecodeError):
        json.loads(line)
    assert "INFO - plain text line" in line
