"""
A head its own holdout says is untrustworthy must not be able to send a tip.

Warning about it in the nightly digest is not containment. The scan runs every
five minutes; the digest is read once a day. A miscalibrated head bets roughly
288 times before anyone sees the warning.

The sharp edge is that EV is computed straight from the model probability, so
an N-point overconfident head overstates every EV it produces by about N x odds
points. At 2.0 a 5pp gap is a 10pp phantom edge against an EDGE_MIN_BPS of 3pp:
the gate stops measuring the market's mistake and starts measuring the model's,
and passes exactly the bets that have no edge.
"""
import json

import pytest

import main


def _health(monkeypatch, **by_head):
    store = {f"model_health:{k}": json.dumps(v) for k, v in by_head.items()}
    monkeypatch.setattr(main, "get_setting_cached", lambda k: store.get(k))


def test_an_overconfident_head_may_not_bet(monkeypatch):
    _health(monkeypatch, **{"OU_2.5": {"calibration_gap_pct": 8.8, "n_train_matches": 900}})
    ok, why = main.head_fit_to_bet("OU_2.5")
    assert ok is False
    assert "overconfident by 8.8pp" in why
    assert "EV overstated" in why


def test_an_underconfident_head_is_left_alone(monkeypatch):
    # Underconfidence costs opportunities. It does not manufacture edge, so it
    # is not a reason to stop betting. Positive is overconfident.
    _health(monkeypatch, **{"OU_2.5": {"calibration_gap_pct": -8.8, "n_train_matches": 900}})
    assert main.head_fit_to_bet("OU_2.5")[0] is True


def test_a_head_trained_on_too_few_fixtures_may_not_bet(monkeypatch):
    _health(monkeypatch, **{"BTTS_YES": {"calibration_gap_pct": 0.4, "n_train_matches": 120}})
    ok, why = main.head_fit_to_bet("BTTS_YES")
    assert ok is False and "120 fixtures" in why


def test_a_head_that_routinely_fights_the_market_may_not_bet(monkeypatch):
    _health(monkeypatch, **{"BTTS_YES": {"calibration_gap_pct": 0.1, "n_train_matches": 900,
                                         "deviation_p95_pp": 22.0}})
    ok, why = main.head_fit_to_bet("BTTS_YES")
    assert ok is False and "22.0pp at p95" in why


def test_a_head_mostly_trained_on_settled_questions_may_not_bet(monkeypatch):
    _health(monkeypatch, **{"OU_2.5": {"calibration_gap_pct": 0.2, "n_train_matches": 900,
                                       "decided_share_pct": 71.0}})
    ok, why = main.head_fit_to_bet("OU_2.5")
    assert ok is False and "71% of its training rows" in why


def test_a_healthy_head_bets(monkeypatch):
    _health(monkeypatch, **{"OU_2.5": {"calibration_gap_pct": 1.1, "n_train_matches": 900,
                                       "deviation_p95_pp": 4.2, "decided_share_pct": 18.0}})
    assert main.head_fit_to_bet("OU_2.5") == (True, None)


def test_a_head_with_no_health_record_is_not_blocked(monkeypatch):
    # Heads trained before this existed, and every prematch head, have none.
    # Refusing those would silently stop the whole system on the next deploy,
    # which is a worse failure than the one being prevented.
    _health(monkeypatch)
    assert main.head_fit_to_bet("OU_2.5") == (True, None)


def test_a_corrupt_health_record_is_not_blocked(monkeypatch):
    monkeypatch.setattr(main, "get_setting_cached", lambda k: "{not json")
    assert main.head_fit_to_bet("OU_2.5") == (True, None)


# ───────── it actually stops the bet ─────────

def _gate(monkeypatch, market, suggestion):
    monkeypatch.setattr(main, "fetch_odds", lambda fid, live: {
        "OU_2.5": {"best": {"Over": {"odds": 2.0, "book": "B"}}, "fair": {"Over": 0.58},
                   "overround": 0.05, "n_books": 5},
        "BTTS": {"best": {"No": {"odds": 2.0, "book": "B"}}, "fair": {"No": 0.58},
                 "overround": 0.05, "n_books": 5},
        "1X2": {"best": {"Home": {"odds": 2.0, "book": "B"}}, "fair": {"Home": 0.58},
                "overround": 0.05, "n_books": 5}})
    return main._price_gate(market, suggestion, fid=1, prob=0.62, live=True)


def test_the_price_gate_refuses_a_suppressed_head(monkeypatch):
    _health(monkeypatch, **{"OU_2.5": {"calibration_gap_pct": 9.0, "n_train_matches": 900}})
    res = _gate(monkeypatch, "Over/Under 2.5", "Over 2.5 Goals")
    assert res["passed"] is False
    assert res["decision"] == "head_suppressed"
    assert "overconfident" in res["suppressed_reason"]


def test_it_refuses_before_spending_an_odds_call(monkeypatch):
    # Reporting "EV too low" for a head that should not bet invites tuning
    # EDGE_MIN_BPS when the problem is the model.
    _health(monkeypatch, **{"OU_2.5": {"calibration_gap_pct": 9.0, "n_train_matches": 900}})
    calls = []
    monkeypatch.setattr(main, "fetch_odds", lambda fid, live: calls.append(fid) or {})
    res = main._price_gate("Over/Under 2.5", "Over 2.5 Goals", fid=1, prob=0.62, live=True)
    assert res["decision"] == "head_suppressed"
    assert calls == []


def test_a_healthy_head_still_reaches_a_price(monkeypatch):
    _health(monkeypatch, **{"OU_2.5": {"calibration_gap_pct": 0.5, "n_train_matches": 900}})
    assert _gate(monkeypatch, "Over/Under 2.5", "Over 2.5 Goals")["decision"] == "tipped"


def test_the_prematch_variant_of_a_market_is_gated_by_the_same_head(monkeypatch):
    _health(monkeypatch, **{"OU_2.5": {"calibration_gap_pct": 9.0, "n_train_matches": 900}})
    res = main._price_gate("PRE Over/Under 2.5", "Over 2.5 Goals", fid=1, prob=0.62, live=False)
    assert res["decision"] == "head_suppressed"


# ───────── the head a candidate came from ─────────

def test_candidates_map_back_to_the_head_that_produced_them():
    assert main.head_for_candidate("BTTS", "BTTS: No") == "BTTS_YES"
    assert main.head_for_candidate("Over/Under 2.5", "Over 2.5 Goals") == "OU_2.5"
    assert main.head_for_candidate("Over/Under 3.5", "Under 3.5 Goals") == "OU_3.5"
    assert main.head_for_candidate("PRE BTTS", "BTTS: Yes") == "BTTS_YES"


def test_the_derived_markets_are_gated_by_the_heads_they_derive_from(monkeypatch):
    # Double Chance and Draw No Bet are built from the same three WLD heads, so
    # any one of them being unfit taints all three markets.
    _health(monkeypatch, **{"WLD_DRAW": {"calibration_gap_pct": 11.0, "n_train_matches": 900}})
    for market, sug in (("1X2", "Home Win"), ("Double Chance", "Double Chance: 1X"),
                        ("Draw No Bet", "Draw No Bet: Home")):
        assert main.candidate_head_blocked(market, sug) is not None, market


def test_an_unrecognised_market_is_not_blocked_by_this(monkeypatch):
    # unmapped_market is a different decision with a different fix.
    _health(monkeypatch, **{"OU_2.5": {"calibration_gap_pct": 9.0, "n_train_matches": 900}})
    assert main.candidate_head_blocked("Corners", "Over 9.5") is None


# ───────── the prematch head gates the prematch candidate ─────────
#
# Prematch is scored by SEPARATE heads — _btts_candidates(feat, "PRE_", ...)
# loads PRE_BTTS_YES — and train_models.py writes each its own
# model_health:PRE_* record. Without a prefix the gate read the LIVE head's
# record for a prematch candidate, so every PRE_* record training writes went
# unread by the gate that exists to act on it.

def test_an_unfit_prematch_head_may_not_bet(monkeypatch):
    # The live head is in perfect health; the prematch one failed validation.
    _health(monkeypatch, **{
        "OU_2.5": {"calibration_gap_pct": 0.2, "n_train_matches": 900},
        "PRE_OU_2.5": {"validation_failed": "single-class prediction",
                       "n_train_matches": 900},
    })
    why = main.candidate_head_blocked("Over/Under 2.5", "Over 2.5 Goals", prefix="PRE_")
    assert why is not None, "a prematch head that failed validation must not bet"
    assert "single-class prediction" in why


def test_a_skill_less_prematch_head_may_not_bet(monkeypatch):
    # The exact case head_fit_to_bet() exists to catch, on the prematch side.
    _health(monkeypatch, **{
        "BTTS_YES": {"calibration_gap_pct": 0.2, "n_train_matches": 900},
        "PRE_BTTS_YES": {"calibration_gap_pct": 0.2, "n_train_matches": 900,
                         "brier_skill": -0.004},
    })
    why = main.candidate_head_blocked("BTTS", "BTTS: Yes", prefix="PRE_")
    assert why is not None and "learned nothing" in why


def test_the_prematch_scan_reaches_the_prematch_head(monkeypatch):
    """
    prematch_scan_save() calls _price_gate(mk, ...) with the market text
    UNPREFIXED — it adds "PRE " only when writing the tip row — so the gate
    cannot infer the phase from the market and takes it from live=False.
    """
    _health(monkeypatch, **{
        "OU_2.5": {"calibration_gap_pct": 0.2, "n_train_matches": 900},
        "PRE_OU_2.5": {"validation_failed": "no skill", "n_train_matches": 900},
    })
    monkeypatch.setattr(main, "fetch_odds", lambda fid, live: {
        "OU_2.5": {"best": {"Over": {"odds": 2.0, "book": "B"}},
                   "fair": {"Over": 0.58}, "overround": 0.05, "n_books": 5}})
    res = main._price_gate("Over/Under 2.5", "Over 2.5 Goals",
                           fid=1, prob=0.62, live=False)
    assert res["decision"] == "head_suppressed"
    assert "no skill" in res["suppressed_reason"]


def test_a_healthy_prematch_head_still_reaches_a_price(monkeypatch):
    _health(monkeypatch, **{
        "OU_2.5": {"calibration_gap_pct": 0.2, "n_train_matches": 900},
        "PRE_OU_2.5": {"calibration_gap_pct": 0.3, "n_train_matches": 900,
                       "brier_skill": 0.02},
    })
    monkeypatch.setattr(main, "fetch_odds", lambda fid, live: {
        "OU_2.5": {"best": {"Over": {"odds": 2.0, "book": "B"}},
                   "fair": {"Over": 0.58}, "overround": 0.05, "n_books": 5}})
    res = main._price_gate("Over/Under 2.5", "Over 2.5 Goals",
                           fid=1, prob=0.62, live=False)
    assert res["decision"] == "tipped"


def test_the_live_scan_is_unaffected_by_prematch_health(monkeypatch):
    # A broken PRE_ head must not stop the in-play side betting.
    _health(monkeypatch, **{
        "OU_2.5": {"calibration_gap_pct": 0.2, "n_train_matches": 900},
        "PRE_OU_2.5": {"validation_failed": "no skill", "n_train_matches": 900},
    })
    assert main.candidate_head_blocked("Over/Under 2.5", "Over 2.5 Goals") is None


# ───────── skill against the best possible constant ─────────
# The most basic test there is, and accuracy hides it completely: a head that
# calls every fixture Over on a 60% base rate reports 60% accuracy and has no
# skill at all. Every prematch head in the first real run scored between
# -0.005 and +0.006, while reporting accuracies of 54-79%.

def test_a_head_with_no_skill_may_not_bet(monkeypatch):
    _health(monkeypatch, **{"OU_2.5": {"calibration_gap_pct": 0.2, "n_train_matches": 900,
                                       "brier_skill": -0.005}})
    ok, why = main.head_fit_to_bet("OU_2.5")
    assert ok is False
    assert "learned nothing a single number could not do" in why


def test_skill_indistinguishable_from_zero_is_not_enough(monkeypatch):
    _health(monkeypatch, **{"OU_2.5": {"calibration_gap_pct": 0.2, "n_train_matches": 900,
                                       "brier_skill": 0.006}})
    assert main.head_fit_to_bet("OU_2.5")[0] is False


def test_a_head_with_real_skill_bets(monkeypatch):
    # The in-play heads in that same run scored +0.22 to +0.26.
    _health(monkeypatch, **{"OU_2.5": {"calibration_gap_pct": 0.2, "n_train_matches": 900,
                                       "brier_skill": 0.23, "deviation_p95_pp": 6.0}})
    assert main.head_fit_to_bet("OU_2.5") == (True, None)


def test_high_accuracy_does_not_rescue_a_skill_less_head(monkeypatch):
    # PRE_OU_2.5 reported 60.2% accuracy and 100% recall by predicting Over on
    # literally every fixture: tn=0, fn=0. Accuracy alone would wave it through.
    _health(monkeypatch, **{"OU_2.5": {"calibration_gap_pct": 0.2, "n_train_matches": 900,
                                       "brier_skill": -0.005, "acc": 0.602}})
    assert main.head_fit_to_bet("OU_2.5")[0] is False


def test_a_head_without_a_skill_number_is_judged_on_the_other_checks(monkeypatch):
    _health(monkeypatch, **{"OU_2.5": {"calibration_gap_pct": 0.2, "n_train_matches": 900}})
    assert main.head_fit_to_bet("OU_2.5") == (True, None)
