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


# ───────── market anchoring ─────────
# Which regime tonight's models are in is not a detail. An unanchored head can
# wander from the market and call the distance edge; the operator has to be
# able to tell that apart from a head that is anchored and genuinely quiet.

def _run_anchor(monkeypatch, anchoring, metrics=None, fallbacks=None):
    res = _train_result(metrics or {})
    res["market_anchoring"] = anchoring
    if fallbacks:
        res["anchor_fallbacks"] = fallbacks
    sent = []
    monkeypatch.setattr(main, "TRAIN_ENABLE", True)
    monkeypatch.setattr(main, "train_models", lambda: res)
    monkeypatch.setattr(main, "send_telegram", lambda msg: sent.append(msg) or True)
    monkeypatch.setattr(main._MODELS_CACHE, "invalidate", lambda: None)
    monkeypatch.setattr(main._SETTINGS_CACHE, "invalidate", lambda: None)
    main.auto_train_job()
    return sent[-1]


def test_an_anchored_run_says_so_with_its_coverage(monkeypatch):
    msg = _run_anchor(monkeypatch, {"anchored": True, "anchored_rows": 4200,
                                    "anchored_matches": 460, "anchored_share_pct": 62.0})
    assert "Market-anchored" in msg
    assert "4200" in msg and "460" in msg and "62%" in msg


def test_an_unanchored_run_says_why_not(monkeypatch):
    # Silence here reads as "anchoring is on and found nothing", which is a
    # completely different situation from "not enough data yet".
    msg = _run_anchor(monkeypatch, {"anchored": False,
                                    "reason": "only 12 rows / 3 fixtures carry a real market price"})
    assert "Not market-anchored" in msg
    assert "only 12 rows" in msg


def test_the_deviation_from_market_is_reported(monkeypatch):
    # This is the number the price gate actually trades on, and it is also the
    # noise whose upper tail the gate selects.
    msg = _run_anchor(monkeypatch,
                      {"anchored": True, "anchored_rows": 5000, "anchored_matches": 500,
                       "anchored_share_pct": 70.0},
                      metrics={"OU_2.5": {"deviation_from_market":
                                          {"mean_abs_pp": 2.4, "p95_abs_pp": 6.1, "max_abs_pp": 9.0}},
                               "BTTS_YES": {"deviation_from_market":
                                            {"mean_abs_pp": 5.8, "p95_abs_pp": 12.2, "max_abs_pp": 18.0}}})
    assert "Deviation from market" in msg
    assert "2.4pp" in msg and "6.1pp" in msg
    # Scoped to the section: head names also appear in the "Trained:" line above.
    section = msg[msg.index("Deviation from market"):]
    assert section.index("BTTS_YES") < section.index("OU_2.5"), "the noisiest head must lead"


def test_a_silent_fallback_is_impossible(monkeypatch):
    # An anchored fit that failed and fell back looks exactly like an anchored
    # model that learned nothing, and the two need different fixes.
    msg = _run_anchor(monkeypatch,
                      {"anchored": True, "anchored_rows": 5000, "anchored_matches": 500,
                       "anchored_share_pct": 70.0},
                      fallbacks=["WLD_DRAW"])
    assert "Anchored fit failed" in msg and "WLD_DRAW" in msg


def test_a_run_with_no_anchoring_information_adds_nothing(monkeypatch):
    msg = _run_anchor(monkeypatch, {})
    assert "anchor" not in msg.lower()
