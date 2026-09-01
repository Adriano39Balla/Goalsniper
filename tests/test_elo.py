"""
elo_update() is the single implementation of the rating update.

It was written out twice in main.py - once on the live result path, once in
the historical backfill - with nothing keeping them in step. A K-factor or
home-advantage change applied to one copy would have made a team's rating
depend on which path happened to observe the match first, and nothing would
have failed loudly.
"""
import pytest

from feature_spec import ELO_DEFAULT, ELO_HOME_ADV, ELO_K, elo_expected_home, elo_update


def test_expected_scores_sum_to_one():
    assert elo_expected_home(1500, 1600) + elo_expected_home(1600, 1500 + 2 * ELO_HOME_ADV) \
        == pytest.approx(1.0, abs=1e-9)


def test_home_advantage_makes_equal_teams_favourites_at_home():
    assert elo_expected_home(1500, 1500) > 0.5


def test_home_advantage_is_worth_exactly_its_rating_points():
    # Folding the advantage into the home rating is the whole definition -
    # a home side rated ELO_HOME_ADV below its opponent is an even match.
    assert elo_expected_home(1500 - ELO_HOME_ADV, 1500) == pytest.approx(0.5)


def test_rating_points_are_conserved():
    # Elo is zero-sum: whatever the home side gains, the away side loses.
    for gh, ga in [(2, 0), (1, 1), (0, 3)]:
        rh, ra = elo_update(1500, 1500, gh, ga)
        assert (rh - 1500) + (ra - 1500) == pytest.approx(0.0, abs=1e-9)


def test_a_home_win_raises_the_home_rating_and_a_loss_lowers_it():
    won, _ = elo_update(1500, 1500, 2, 0)
    lost, _ = elo_update(1500, 1500, 0, 2)
    assert won > 1500 > lost


def test_a_draw_costs_the_home_favourite_points():
    # Equal ratings but home advantage means a draw is an under-performance.
    rh, ra = elo_update(1500, 1500, 1, 1)
    assert rh < 1500
    assert ra > 1500


def test_margin_of_victory_is_deliberately_ignored():
    # This K-factor update grades win/draw/loss only. Pinning it so a change
    # to goal-difference weighting has to be a deliberate edit, not a drift.
    assert elo_update(1500, 1500, 1, 0) == elo_update(1500, 1500, 5, 0)


def test_beating_a_stronger_opponent_gains_more_than_beating_a_weaker_one():
    upset, _ = elo_update(1400, 1800, 1, 0)
    expected, _ = elo_update(1800, 1400, 1, 0)
    assert (upset - 1400) > (expected - 1800)


def test_a_single_update_cannot_move_a_rating_more_than_k():
    for rh, ra, gh, ga in [(1500, 1500, 3, 0), (1000, 2000, 1, 0), (2000, 1000, 0, 1)]:
        new_h, new_a = elo_update(rh, ra, gh, ga)
        assert abs(new_h - rh) <= ELO_K + 1e-9
        assert abs(new_a - ra) <= ELO_K + 1e-9


def test_both_call_sites_now_share_one_implementation():
    # The live path and the backfill path must be indistinguishable. Both go
    # through elo_update, so this pins the contract rather than the copies.
    import main
    assert main.elo_update is elo_update
    src = open("main.py").read()
    assert "ELO_HOME_ADV" not in src, "main.py is computing Elo itself again"


def test_default_rating_is_the_neutral_starting_point():
    rh, ra = elo_update(ELO_DEFAULT, ELO_DEFAULT, 1, 1)
    assert rh != ELO_DEFAULT and ra != ELO_DEFAULT
