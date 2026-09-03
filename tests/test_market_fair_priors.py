"""
_market_fair_priors() feeds the market's own de-vigged read to every model
head as an input feature (not just the post-hoc price gate).

TWO invariants, and they pull in opposite directions:

  1. The FEATURE VECTOR must degrade to feature_spec.NEUTRAL_MARKET_PRIORS,
     never to 0.0. These are probabilities, and a literal 0.0 reads as "the
     market says this is impossible" - the most confident wrong number
     available.

  2. The RAW dict must leave an unquoted market ABSENT. Market anchoring can
     only exclude a row that has no real price if absence survives to
     training, and this function used to start from dict(NEUTRAL_MARKET_PRIORS)
     and so always returned all five keys. Every snapshot ever harvested
     therefore looked like it carried a price, and anchoring treated a
     fabricated 0.5 as the market's opinion. The first real run showed heads
     deviating from "market" by a mean of 20-24pp with a maximum of almost
     exactly 50pp - a model at ~0.95 against a placeholder at 0.50.

So the neutral fill moved downstream into build_inplay_features(), which is
where the feature vector is actually assembled.
"""
import main
from feature_spec import NEUTRAL_MARKET_PRIORS, build_inplay_features, DEFAULT_LEAGUE_RATES


def _features(raw_extra):
    raw = {"minute": 60.0, "goals_h": 1, "goals_a": 0}
    raw.update(raw_extra)
    return build_inplay_features(raw, DEFAULT_LEAGUE_RATES)


def test_fixture_with_no_id_never_calls_fetch_odds(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("fetch_odds must not be called for fid=0")

    monkeypatch.setattr(main, "fetch_odds", _boom)
    assert main._market_fair_priors(0, live=True) == {}


def test_no_odds_available_returns_nothing_rather_than_placeholders(monkeypatch):
    # Absence has to stay absent, or the row cannot be excluded from an
    # anchored fit later.
    monkeypatch.setattr(main, "fetch_odds", lambda fid, live: {})
    assert main._market_fair_priors(123, live=True) == {}


def test_the_feature_vector_still_degrades_to_neutral_not_zero(monkeypatch):
    # The invariant that matters for scoring, now enforced downstream.
    monkeypatch.setattr(main, "fetch_odds", lambda fid, live: {})
    feat = _features(main._market_fair_priors(123, live=True))
    for k, neutral in NEUTRAL_MARKET_PRIORS.items():
        assert feat[k] == neutral, k
        assert feat[k] != 0.0


def test_full_odds_coverage_uses_real_devigged_probabilities(monkeypatch):
    monkeypatch.setattr(main, "fetch_odds", lambda fid, live: {
        "1X2": {"fair": {"Home": 0.55, "Draw": 0.25, "Away": 0.20}},
        "OU_2.5": {"fair": {"Over": 0.62, "Under": 0.38}},
        "BTTS": {"fair": {"Yes": 0.58, "No": 0.42}},
    })
    out = main._market_fair_priors(123, live=True)
    assert out["market_fair_home"] == 0.55
    assert out["market_fair_draw"] == 0.25
    assert out["market_fair_away"] == 0.20
    assert out["market_fair_over25"] == 0.62
    assert out["market_fair_btts_yes"] == 0.58


def test_partial_coverage_reports_only_what_resolved(monkeypatch):
    # 1X2 incomplete (missing Draw) -> absent; OU_2.5 present -> real value.
    monkeypatch.setattr(main, "fetch_odds", lambda fid, live: {
        "1X2": {"fair": {"Home": 0.55, "Away": 0.20}},
        "OU_2.5": {"fair": {"Over": 0.62}},
    })
    out = main._market_fair_priors(123, live=True)
    assert out == {"market_fair_over25": 0.62}
    # ...and the vector still fills the rest neutrally.
    feat = _features(out)
    assert feat["market_fair_over25"] == 0.62
    assert feat["market_fair_home"] == NEUTRAL_MARKET_PRIORS["market_fair_home"]
    assert feat["market_fair_btts_yes"] == NEUTRAL_MARKET_PRIORS["market_fair_btts_yes"]


def test_live_flag_is_forwarded_to_fetch_odds(monkeypatch):
    seen = {}
    monkeypatch.setattr(main, "fetch_odds", lambda fid, live: seen.update(fid=fid, live=live) or {})
    main._market_fair_priors(999, live=False)
    assert seen == {"fid": 999, "live": False}
