"""
compute_price_gate_breakdown() answers "why aren't candidates becoming
tips" from the predictions table, which records every gated candidate's
decision - so the question is answerable over days and after the fact,
rather than only from whatever is still in a log buffer.

below_threshold and per_match_cap are set BEFORE the gate runs. Counting
them as gate outcomes would bury the real rejection reasons under a pile of
candidates that never reached the gate at all.
"""
import pytest

import main


def _stub(monkeypatch, rows):
    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, sql, params=()):
            self._sql = sql
            self._params = params
            return self

        def fetchall(self):
            return rows

    monkeypatch.setattr(main, "db_conn", lambda: _Conn())


def test_rejection_reasons_are_ranked_by_how_often_they_block(monkeypatch):
    _stub(monkeypatch, [
        ("live", "tipped", 5),
        ("live", "no_odds", 60),
        ("live", "too_few_books", 25),
        ("live", "ev_below_min", 10),
    ])
    live = main.compute_price_gate_breakdown(days=7)["by_phase"]["live"]
    assert live["reached_price_gate"] == 100
    assert live["tipped"] == 5
    assert live["tipped_pct"] == 5.0
    assert [b["reason"] for b in live["blocked_by"]] == [
        "no_odds", "too_few_books", "ev_below_min"]
    assert live["blocked_by"][0]["pct_of_gated"] == 60.0


def test_candidates_that_never_reached_the_gate_are_kept_separate(monkeypatch):
    # 900 below-threshold candidates would otherwise drown the 100 real
    # outcomes and make every rejection reason look negligible.
    _stub(monkeypatch, [
        ("live", "below_threshold", 900),
        ("live", "per_match_cap", 40),
        ("live", "tipped", 20),
        ("live", "no_odds", 80),
    ])
    live = main.compute_price_gate_breakdown(days=7)["by_phase"]["live"]
    assert live["reached_price_gate"] == 100
    assert live["tipped_pct"] == 20.0
    assert live["never_reached_gate"] == {"below_threshold": 900, "per_match_cap": 40}
    assert all(b["reason"] not in ("below_threshold", "per_match_cap")
               for b in live["blocked_by"])


def test_live_and_prematch_are_reported_separately(monkeypatch):
    # They have different thresholds and different market depth; averaging
    # them together hides which one is actually stuck.
    _stub(monkeypatch, [
        ("live", "no_odds", 30), ("live", "tipped", 10),
        ("prematch", "ev_below_min", 50), ("prematch", "tipped", 50),
    ])
    out = main.compute_price_gate_breakdown(days=7)["by_phase"]
    assert out["live"]["tipped_pct"] == 25.0
    assert out["prematch"]["tipped_pct"] == 50.0


def test_an_empty_window_does_not_divide_by_zero(monkeypatch):
    _stub(monkeypatch, [])
    out = main.compute_price_gate_breakdown(days=1)
    assert out["by_phase"] == {}


def test_a_phase_with_no_gated_rows_reports_zero_not_an_error(monkeypatch):
    _stub(monkeypatch, [("live", "below_threshold", 12)])
    live = main.compute_price_gate_breakdown(days=1)["by_phase"]["live"]
    assert live["reached_price_gate"] == 0
    assert live["tipped_pct"] == 0.0
    assert live["blocked_by"] == []


def test_the_sampling_bias_is_stated_not_hidden(monkeypatch):
    # tipped rows are always kept while others are trimmed, so tipped_pct is
    # an upper bound. Saying so is the difference between a diagnostic and a
    # misleading number.
    _stub(monkeypatch, [("live", "tipped", 1)])
    out = main.compute_price_gate_breakdown(days=7)
    assert "upper bound" in out["sampling_note"]
    assert "Not a census" in out["sampling_note"]


def test_phase_filter_is_passed_to_the_query(monkeypatch):
    seen = {}

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, sql, params=()):
            seen["sql"], seen["params"] = sql, params
            return self

        def fetchall(self):
            return []

    monkeypatch.setattr(main, "db_conn", lambda: _Conn())
    main.compute_price_gate_breakdown(days=3, phase="prematch")
    assert "AND phase = %s" in seen["sql"]
    assert seen["params"][-1] == "prematch"


def test_endpoint_is_admin_gated():
    r = main.app.test_client().get("/admin/diagnostics/price-gate")
    assert r.status_code == 401
