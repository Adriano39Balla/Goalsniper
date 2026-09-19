import os

from startup_checks import StartupConfigError, validate_runtime_environment


def test_valid_runtime_environment_sets_required_values():
    original = {k: os.environ.get(k) for k in [
        "DATABASE_URL",
        "API_KEY",
        "RUN_SCHEDULER",
        "SCAN_INTERVAL_SEC",
        "MAX_TIPS_PER_SCAN",
        "TARGET_PRECISION",
        "MIN_THRESH",
        "MAX_THRESH",
    ]}

    try:
        os.environ["DATABASE_URL"] = "postgresql://user:pass@localhost:5432/goalsniper"
        os.environ["API_KEY"] = "unit-test-api-key"
        os.environ["RUN_SCHEDULER"] = "0"
        os.environ["SCAN_INTERVAL_SEC"] = "300"
        os.environ["MAX_TIPS_PER_SCAN"] = "25"
        os.environ["TARGET_PRECISION"] = "0.6"
        os.environ["MIN_THRESH"] = "55"
        os.environ["MAX_THRESH"] = "85"

        cfg = validate_runtime_environment()
        assert cfg["database_url"].startswith("postgresql://")
        assert cfg["api_key"] == "unit-test-api-key"
        assert cfg["run_scheduler"] is False
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_missing_api_key_raises():
    original = {k: os.environ.get(k) for k in ["API_KEY", "DATABASE_URL", "RUN_SCHEDULER"]}
    try:
        os.environ["DATABASE_URL"] = "postgresql://user:pass@localhost:5432/goalsniper"
        os.environ.pop("API_KEY", None)
        os.environ["RUN_SCHEDULER"] = "0"

        try:
            validate_runtime_environment()
            raise AssertionError("Expected StartupConfigError")
        except StartupConfigError:
            pass
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_threshold_order_is_enforced():
    original = {k: os.environ.get(k) for k in ["DATABASE_URL", "API_KEY", "RUN_SCHEDULER", "MIN_THRESH", "MAX_THRESH"]}
    try:
        os.environ["DATABASE_URL"] = "postgresql://user:pass@localhost:5432/goalsniper"
        os.environ["API_KEY"] = "unit-test-api-key"
        os.environ["RUN_SCHEDULER"] = "0"
        os.environ["MIN_THRESH"] = "90"
        os.environ["MAX_THRESH"] = "85"

        try:
            validate_runtime_environment()
            raise AssertionError("Expected StartupConfigError for inverted thresholds")
        except StartupConfigError:
            pass
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
