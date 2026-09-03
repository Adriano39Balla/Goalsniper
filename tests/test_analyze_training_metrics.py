"""
analyze_training_metrics.py's "MODEL PARITY VERIFICATION" section read
golden_samples and a "parity_verified" flag off model_metrics_latest's
per-head entries - but train_models.py writes golden_samples into a SEPARATE
settings key, model_health:{head}, and no code anywhere ever writes
"parity_verified" at all (the real check runs live in main.py, at serve
time, against whatever model is currently deployed, and its result is never
persisted). So the loop here could never find a sample count, could never
populate a failure, and unconditionally printed "checks passed (or not yet
run)" regardless of whether anything had been examined - a diagnostic
script whose central claim was structurally incapable of being false.

These tests exercise analyze_metrics() against a fake DB connection to
confirm it now reads golden_samples from the settings key that actually
holds them, and no longer asserts a pass it never computed.
"""
import json

import analyze_training_metrics as atm


class _FakeCursor:
    def __init__(self, table):
        self._table = table

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=()):
        self._key = params[0] if params else None
        return self

    def fetchone(self):
        v = self._table.get(self._key)
        return (v,) if v is not None else None


class _FakeConn:
    def __init__(self, table):
        self._table = table

    def cursor(self):
        return _FakeCursor(self._table)

    def close(self):
        pass


def _settings_table(metrics_bundle, health_by_head):
    table = {"model_metrics_latest": json.dumps(metrics_bundle)}
    for head, health in health_by_head.items():
        table[f"model_health:{head}"] = json.dumps(health)
    return table


def test_golden_samples_are_read_from_model_health_not_the_metrics_bundle(monkeypatch, caplog):
    metrics_bundle = {
        "trained_at_utc": "2026-01-01T00:00:00Z",
        "BTTS_YES": {"n_train": 900, "calibration_gap_pct": 0.0,
                     "mean_predicted": 0.5, "mean_actual": 0.5},
    }
    health = {"BTTS_YES": {"golden_samples": [{"features": {}, "prob": 0.6}] * 3}}
    table = _settings_table(metrics_bundle, health)
    monkeypatch.setattr(atm, "get_db_connection", lambda: _FakeConn(table))

    with caplog.at_level("INFO"):
        atm.analyze_metrics()

    messages = "\n".join(r.getMessage() for r in caplog.records)
    assert "3 golden sample(s) recorded" in messages
    # The old always-true claim must be gone.
    assert "Model parity checks passed (or not yet run)" not in messages
    assert "parity_verified" not in messages.lower()


def test_a_head_with_no_health_record_is_reported_not_silently_skipped(monkeypatch, caplog):
    metrics_bundle = {
        "trained_at_utc": "2026-01-01T00:00:00Z",
        "PRE_BTTS_YES": {"n_train": 5000, "calibration_gap_pct": 0.0,
                         "mean_predicted": 0.5, "mean_actual": 0.5},
    }
    table = _settings_table(metrics_bundle, {})
    monkeypatch.setattr(atm, "get_db_connection", lambda: _FakeConn(table))

    with caplog.at_level("INFO"):
        atm.analyze_metrics()

    messages = "\n".join(r.getMessage() for r in caplog.records)
    assert "no model_health record found" in messages
    assert "No head has golden samples recorded" in messages
