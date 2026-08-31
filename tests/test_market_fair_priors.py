"""
_market_fair_priors() feeds the market's own de-vigged read to every model
head as an input feature (not just the post-hoc price gate). It must
degrade to feature_spec.NEUTRAL_MARKET_PRIORS - not 0.0 - whenever odds are
unavailable, since these are probabilities: a literal 0.0 would falsely read
as "the market says this is impossible."
"""
import main
from feature_spec import NEUTRAL_MARKET_PRIORS


def test_fixture_with_no_id_never_calls_fetch_odds_and_uses_neutral_priors(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("fetch_odds must not be called for fid=0")

    monkeypatch.setattr(main, "fetch_odds", _boom)
    assert main._market_fair_priors(0, live=True) == NEUTRAL_MARKET_PRIORS


def test_no_odds_available_falls_back_to_neutral_priors(monkeypatch):
    monkeypatch.setattr(main, "fetch_odds", lambda fid, live: {})
    assert main._market_fair_priors(123, live=True) == NEUTRAL_MARKET_PRIORS


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


def test_partial_coverage_only_overrides_available_markets(monkeypatch):
    # 1X2 incomplete (missing Draw) -> stays neutral; OU_2.5 present -> real value.
    monkeypatch.setattr(main, "fetch_odds", lambda fid, live: {
        "1X2": {"fair": {"Home": 0.55, "Away": 0.20}},
        "OU_2.5": {"fair": {"Over": 0.62}},
    })
    out = main._market_fair_priors(123, live=True)
    assert out["market_fair_home"] == NEUTRAL_MARKET_PRIORS["market_fair_home"]
    assert out["market_fair_over25"] == 0.62
    assert out["market_fair_btts_yes"] == NEUTRAL_MARKET_PRIORS["market_fair_btts_yes"]


def test_live_flag_is_forwarded_to_fetch_odds(monkeypatch):
    seen = {}
    monkeypatch.setattr(main, "fetch_odds", lambda fid, live: seen.update(fid=fid, live=live) or {})
    main._market_fair_priors(999, live=False)
    assert seen == {"fid": 999, "live": False}
