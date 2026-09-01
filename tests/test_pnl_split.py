"""
P&L must not report bets priced off contaminated market data as a track
record.

Before the market-mapping fix, team totals and half markets were folded
into the full-match markets and fetch_odds kept the best price across that
mix - so the recorded price was one that was never available for the
selection. Grading against it produced a 116.8% ROI on PRE Over/Under 2.5
and a headline of +492u. Those rows are unrecoverable (the true price at
tip time was never stored), so they are reported separately and excluded
from every headline figure rather than deleted or silently counted.
"""
import pytest

import main
from feature_spec import ODDS_TRUSTED_FROM_TS as CUTOFF


def _row(created_ts, odds=2.0, gh=2, ga=0, market="1X2", sugg="Home Win",
         stake_units=1.0, clv=1.0):
    # compute_pnl's SELECT order: market, suggestion, odds, created_ts,
    # stake_units, clv_pct, final_goals_h, final_goals_a, btts_yes
    return (market, sugg, odds, created_ts, stake_units, clv, gh, ga,
            1 if (gh > 0 and ga > 0) else 0)


def _stub_rows(monkeypatch, rows):
    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, sql, params=()):
            return self

        def fetchall(self):
            return rows

    monkeypatch.setattr(main, "db_conn", lambda: _Conn())


def test_bets_priced_before_the_fix_are_excluded_from_the_headline(monkeypatch):
    _stub_rows(monkeypatch, [_row(CUTOFF - 100), _row(CUTOFF - 50)])
    pnl = main.compute_pnl()
    assert pnl["n_bets"] == 0
    assert pnl["total_profit"] == 0.0
    assert pnl["excluded_unreliable_pricing"]["n_bets"] == 2


def test_bets_priced_after_the_fix_count_normally(monkeypatch):
    _stub_rows(monkeypatch, [_row(CUTOFF), _row(CUTOFF + 100)])
    pnl = main.compute_pnl()
    assert pnl["n_bets"] == 2
    assert pnl["excluded_unreliable_pricing"]["n_bets"] == 0
    # Two winners at 2.0 on 1u each.
    assert pnl["total_profit"] == pytest.approx(2.0)


def test_the_two_periods_are_never_mixed(monkeypatch):
    # A fabricated winner before the cutoff must not prop up a real loss after.
    _stub_rows(monkeypatch, [
        _row(CUTOFF - 10, odds=9.0, gh=2, ga=0),   # contaminated "win"
        _row(CUTOFF + 10, odds=2.0, gh=0, ga=1),   # genuine loss
    ])
    pnl = main.compute_pnl()
    assert pnl["n_bets"] == 1
    assert pnl["total_profit"] == pytest.approx(-1.0)
    assert pnl["roi_pct"] == pytest.approx(-100.0)
    assert pnl["excluded_unreliable_pricing"]["total_profit"] == pytest.approx(8.0)


def test_excluded_bets_are_reported_not_hidden(monkeypatch):
    _stub_rows(monkeypatch, [_row(CUTOFF - 10, odds=5.0, gh=1, ga=0),
                             _row(CUTOFF - 20, odds=5.0, gh=0, ga=1)])
    stale = main.compute_pnl()["excluded_unreliable_pricing"]
    assert stale["n_bets"] == 2
    assert stale["win_rate_pct"] == 50.0
    assert stale["total_profit"] == pytest.approx(3.0)
    assert "never available" in stale["note"]


def test_by_market_covers_only_trustworthy_bets(monkeypatch):
    _stub_rows(monkeypatch, [
        _row(CUTOFF - 10, market="PRE Over/Under 2.5", sugg="Over 2.5 Goals", odds=4.1, gh=2, ga=1),
        _row(CUTOFF + 10, market="1X2", odds=2.0, gh=2, ga=0),
    ])
    by_market = main.compute_pnl()["by_market"]
    assert "PRE Over/Under 2.5" not in by_market
    assert by_market["1X2"]["bets"] == 1


def test_equity_curve_starts_at_the_fix(monkeypatch):
    # An equity curve splicing fabricated profit onto real profit is worse
    # than no curve at all.
    _stub_rows(monkeypatch, [_row(CUTOFF - 10, odds=9.0), _row(CUTOFF + 10, odds=2.0)])
    curve = main.compute_pnl()["equity_curve"]
    assert [p["ts"] for p in curve] == [CUTOFF + 10]


def test_clv_is_measured_only_on_trustworthy_prices(monkeypatch):
    # CLV compares tip odds against closing odds; both were contaminated.
    _stub_rows(monkeypatch, [_row(CUTOFF - 10, clv=99.0), _row(CUTOFF + 10, clv=1.0)])
    assert main.compute_pnl()["mean_clv_pct"] == pytest.approx(1.0)


def test_pushes_are_still_excluded_from_both_periods(monkeypatch):
    # Draw No Bet on a draw voids; it is not a loss on either side of the line.
    _stub_rows(monkeypatch, [_row(CUTOFF + 10, market="Draw No Bet",
                                  sugg="Draw No Bet: Home", gh=1, ga=1)])
    pnl = main.compute_pnl()
    assert pnl["n_bets"] == 0
    assert pnl["n_pushes_excluded"] == 1
    assert pnl["excluded_unreliable_pricing"]["n_bets"] == 0


def test_the_cutoff_is_shared_with_training(monkeypatch):
    # main.py and train_models.py must agree on where the line falls.
    import train_models
    assert train_models.MARKET_FAIR_TRUSTED_FROM_TS == CUTOFF
    _stub_rows(monkeypatch, [])
    assert main.compute_pnl()["odds_trusted_from_ts"] == CUTOFF
