"""
A head's calibration gap has to reach the operator, because EV is computed
straight from the model probability.

If a head runs N points overconfident, every EV it produces is overstated
by roughly N x odds points. At a live price of 2.0 an 8pp gap is a 16pp
phantom edge against an EDGE_MIN_BPS of 3pp - the price gate would be
measuring the model's own error rather than the market's mistake, and
would pass exactly the bets with no real edge.

The real run that motivated this: OU_2.5 predicted 0.592 against an actual
0.504 on holdout, because the calibration window's base rate (65.5%) and
the holdout's (50.4%) were 15pp apart. Calibration fitted on one period
did not transfer to the next.
"""
import main


def _train_result(metrics):
    return {"ok": True, "trained": {"OU_2.5": True},
            "thresholds": {"Over/Under 2.5": 55.0},
            "data_stats": {"inplay_rows": 15806, "inplay_matches": 1729, "prematch_rows": 14226},
            "metrics": metrics}


def _run(monkeypatch, metrics):
    sent = []
    monkeypatch.setattr(main, "TRAIN_ENABLE", True)
    monkeypatch.setattr(main, "train_models", lambda: _train_result(metrics))
    monkeypatch.setattr(main, "send_telegram", lambda msg: sent.append(msg) or True)
    monkeypatch.setattr(main._MODELS_CACHE, "invalidate", lambda: None)
    monkeypatch.setattr(main._SETTINGS_CACHE, "invalidate", lambda: None)
    main.auto_train_job()
    return sent[-1]


def test_an_overconfident_head_is_called_out_with_its_ev_impact(monkeypatch):
    msg = _run(monkeypatch, {"OU_2.5": {"calibration_gap_pct": -8.76,
                                        "mean_predicted": 0.5918, "mean_actual": 0.5042}})
    # Reported as predicted - actual, i.e. positive means overconfident.
    msg = _run(monkeypatch, {"OU_2.5": {"calibration_gap_pct": 8.76}})
    assert "Miscalibrated heads" in msg
    assert "OU_2.5" in msg
    assert "overconfident" in msg
    assert "EV overstated" in msg


def test_an_underconfident_head_is_reported_without_a_false_ev_claim(monkeypatch):
    # Underconfidence understates EV - it costs opportunities, it does not
    # manufacture phantom edge, so it must not carry the same warning.
    msg = _run(monkeypatch, {"WLD_DRAW": {"calibration_gap_pct": -6.09}})
    assert "WLD_DRAW" in msg
    assert "underconfident" in msg
    assert "EV overstated" not in msg


def test_a_well_calibrated_run_says_nothing(monkeypatch):
    msg = _run(monkeypatch, {"OU_3.5": {"calibration_gap_pct": -1.4},
                             "WLD_HOME": {"calibration_gap_pct": 1.49}})
    assert "Miscalibrated heads" not in msg


def test_heads_are_ranked_worst_first(monkeypatch):
    msg = _run(monkeypatch, {"OU_2.5": {"calibration_gap_pct": 8.76},
                             "BTTS_YES": {"calibration_gap_pct": 3.52},
                             "WLD_AWAY": {"calibration_gap_pct": 4.22}})
    order = [msg.index(h) for h in ("OU_2.5", "WLD_AWAY", "BTTS_YES")]
    assert order == sorted(order), "the worst-calibrated head must lead"


def test_non_dict_metric_entries_do_not_break_the_summary(monkeypatch):
    # The metrics blob mixes per-head dicts with scalar diagnostics.
    msg = _run(monkeypatch, {"OU_2.5": {"calibration_gap_pct": 8.76},
                             "some_scalar_diag": 0.42, "a_note": "text"})
    assert "Miscalibrated heads" in msg


def test_the_normal_summary_is_still_sent(monkeypatch):
    msg = _run(monkeypatch, {"OU_2.5": {"calibration_gap_pct": 8.76}})
    assert "Model training OK" in msg
    assert "Trained:" in msg
    assert "in-play 15806" in msg


def test_settled_row_share_is_reported_in_the_summary(monkeypatch):
    # The measurement is useless if it only lives in a metrics blob.
    msg = _run(monkeypatch, {"OU_2.5": {"already_decided": {
        "decided_share_pct": 31.4, "base_rate_all": 0.5042,
        "base_rate_undecided": 0.3310}}})
    assert "Already-settled rows" in msg
    assert "OU_2.5" in msg and "31%" in msg
    assert "0.50" in msg and "0.33" in msg


def test_a_head_with_no_settled_rows_is_not_listed(monkeypatch):
    msg = _run(monkeypatch, {"OU_2.5": {"already_decided": {
        "decided_share_pct": 0.0, "base_rate_all": 0.5, "base_rate_undecided": 0.5}}})
    assert "Already-settled rows" not in msg


def test_heads_are_listed_worst_first(monkeypatch):
    msg = _run(monkeypatch, {
        "BTTS_YES": {"already_decided": {"decided_share_pct": 12.0, "base_rate_all": 0.52,
                                         "base_rate_undecided": 0.45}},
        "OU_2.5": {"already_decided": {"decided_share_pct": 31.4, "base_rate_all": 0.50,
                                       "base_rate_undecided": 0.33}}})
    assert msg.index("OU_2.5") < msg.index("BTTS_YES")
