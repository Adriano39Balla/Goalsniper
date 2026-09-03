"""
A fabricated market price must never be mistaken for the market's opinion.

main._market_fair_priors() used to start from dict(NEUTRAL_MARKET_PRIORS) and
overwrite what it could resolve, so it ALWAYS returned all five market keys.
Every snapshot therefore persisted a value whether or not a price had existed,
and load_inplay_data's "did this row carry a real price?" test — value is not
None — was true for every row ever harvested.

Market anchoring then anchored to a neutral 0.5 wherever nothing had been
quoted, which is exactly what frame_anchor_mask() exists to prevent. The first
real training run is the evidence: anchored heads reported a mean deviation
from "market" of 20-24pp and a maximum of almost exactly 50pp — the
fingerprint of a model at ~0.95 measured against a placeholder at 0.50, not of
a model with an opinion worth 50 points.
"""
import json

import pytest

import main
from feature_spec import NEUTRAL_MARKET_PRIORS, RAW_INPLAY_KEYS
from train_models import _is_real_market_price


# ───────── telling a price from a placeholder ─────────

def test_a_real_price_is_real():
    assert _is_real_market_price("market_fair_over25", 0.62) is True
    assert _is_real_market_price("market_fair_home", 0.41) is True


def test_an_absent_price_is_not_real():
    assert _is_real_market_price("market_fair_over25", None) is False


def test_a_value_sitting_exactly_on_its_neutral_prior_is_treated_as_absent():
    # Rows already in the database carry the placeholder as a number. A
    # genuinely de-vigged price landing on exactly 0.5000000000 is vanishingly
    # unlikely, and discarding the odd real row that does costs nothing next to
    # anchoring to a number no bookmaker ever quoted.
    assert _is_real_market_price("market_fair_over25", 0.5) is False
    assert _is_real_market_price("market_fair_btts_yes", 0.5) is False
    assert _is_real_market_price("market_fair_home", 1.0 / 3.0) is False


def test_a_price_merely_near_the_neutral_prior_is_still_real():
    # A market genuinely pricing a coin flip must not be thrown away.
    assert _is_real_market_price("market_fair_over25", 0.5001) is True
    assert _is_real_market_price("market_fair_over25", 0.4999) is True


def test_a_key_with_no_neutral_prior_is_judged_on_presence_alone():
    assert _is_real_market_price("something_else", 0.5) is True


# ───────── absence has to survive persistence ─────────

def _saved_payload(monkeypatch, raw):
    written = {}

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, sql, params=()):
            written["payload"] = json.loads(params[2])
            return self

    monkeypatch.setattr(main, "db_conn", lambda: _Conn())
    main.save_snapshot_from_match(
        {"fixture": {"id": 5}, "league": {"id": 39}, "teams": {}}, raw)
    return written["payload"]["raw"]


def test_an_unquoted_market_is_persisted_as_null(monkeypatch):
    saved = _saved_payload(monkeypatch, {"minute": 60.0, "sot_h": 3})
    for k in NEUTRAL_MARKET_PRIORS:
        assert saved[k] is None, f"{k} must stay absent, not become a number"


def test_it_is_not_persisted_as_zero(monkeypatch):
    # Every other raw key is a count where missing and zero mean the same
    # thing. A probability is the opposite: 0.0 reads as "the market says
    # impossible", the most confident wrong number in the row.
    saved = _saved_payload(monkeypatch, {"minute": 60.0})
    assert saved["market_fair_over25"] != 0.0


def test_a_quoted_market_is_persisted_as_its_price(monkeypatch):
    saved = _saved_payload(monkeypatch, {"minute": 60.0, "market_fair_over25": 0.62})
    assert saved["market_fair_over25"] == pytest.approx(0.62)


def test_ordinary_counts_still_default_to_zero(monkeypatch):
    # The null treatment is for probabilities only.
    saved = _saved_payload(monkeypatch, {"minute": 60.0})
    assert saved["sot_h"] == 0.0
    assert saved["cor_a"] == 0.0
    assert set(saved) == set(RAW_INPLAY_KEYS)


def test_a_persisted_null_round_trips_to_not_real():
    # The end-to-end contract: what save writes, training must read as absent.
    assert _is_real_market_price("market_fair_over25", None) is False
