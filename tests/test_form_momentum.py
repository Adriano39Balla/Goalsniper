"""
Test form momentum features added to prematch models.

Recent form (last 5 games) is NOT in market prices yet - prices are set
based on season-long records. Form momentum captures short-term trends
that create an edge.
"""
import pytest

from feature_spec import (
    assemble_prematch_features,
    recent_form_momentum,
    team_form_stats,
)


TEAM_H = 123
TEAM_A = 456


def sample_fixture(team_id, goals_for, goals_against, is_home=True, day=1):
    """
    A finished match for testing.

    `day` sets the kickoff date, so a window can express real recency. This
    matters: recent_form_momentum() selects its window by kickoff timestamp
    rather than by list position, because the live path and the historical
    backfill hand it lists ordered opposite ways. Fixtures built without
    distinct dates cannot distinguish "newest N" from "oldest N" and so cannot
    test that selection at all.
    """
    return {
        "fixture": {"id": 999, "status": {"short": "FT"},
                    "date": f"2026-09-{day:02d}T15:00:00Z"},
        "teams": {
            "home": {"id": team_id if is_home else TEAM_A},
            "away": {"id": TEAM_A if is_home else team_id},
        },
        "goals": {
            "home": goals_for if is_home else goals_against,
            "away": goals_against if is_home else goals_for,
        },
        "date": f"2026-09-{day:02d}T15:00:00Z",
    }


def test_recent_form_momentum_all_wins():
    """Recent momentum captures a team on a hot streak."""
    last_5 = [
        sample_fixture(TEAM_H, 3, 1),  # Win
        sample_fixture(TEAM_H, 2, 0),  # Win
        sample_fixture(TEAM_H, 2, 1),  # Win
        sample_fixture(TEAM_H, 1, 0),  # Win
        sample_fixture(TEAM_H, 2, 0),  # Win
    ]
    momentum = recent_form_momentum(TEAM_H, last_5, window_size=5)

    assert momentum["momentum"] == 1.0  # 5/5 wins
    assert momentum["goals_for"] == 2.0  # (3+2+2+1+2)/5
    assert momentum["goals_against"] == 0.4  # (1+0+1+0+0)/5
    assert momentum["played"] == 5


def test_recent_form_momentum_mixed_record():
    """Momentum reflects actual recent form, not season-long stats."""
    # End of season: 2 recent losses despite season record
    last_5 = [
        sample_fixture(TEAM_H, 1, 2),  # Loss
        sample_fixture(TEAM_H, 0, 1),  # Loss
        sample_fixture(TEAM_H, 1, 0),  # Win (earlier)
        sample_fixture(TEAM_H, 3, 1),  # Win (earlier)
        sample_fixture(TEAM_H, 2, 2),  # Draw (earlier)
    ]
    momentum = recent_form_momentum(TEAM_H, last_5, window_size=5)

    # Last 2 games: both losses
    assert momentum["momentum"] == 0.4  # 2/5 wins
    assert momentum["goals_for"] == 1.4  # (1+0+1+3+2)/5
    assert momentum["goals_against"] == 1.2


def test_recent_form_momentum_fewer_games():
    """Momentum works with < 5 games available."""
    last_3 = [
        sample_fixture(TEAM_H, 2, 1),  # Win
        sample_fixture(TEAM_H, 1, 0),  # Win
        sample_fixture(TEAM_H, 3, 2),  # Win
    ]
    momentum = recent_form_momentum(TEAM_H, last_3, window_size=5)

    assert momentum["momentum"] == 1.0  # 3/3 wins
    assert momentum["played"] == 3  # Only 3 games available
    assert momentum["goals_for"] == 2.0  # (2+1+3)/3


def test_form_momentum_vs_season_form():
    """Season form (team_form_stats) and momentum tell different stories."""
    # Season form uses decay weights (recent games count more)
    # Recent momentum uses uniform weights (all recent games count equally)
    # This shows that recent_form_momentum captures pure recency
    all_season = [
        sample_fixture(TEAM_H, 1, 2, day=1),  # Loss (old)
        sample_fixture(TEAM_H, 0, 1, day=2),  # Loss (old)
        sample_fixture(TEAM_H, 1, 1, day=3),  # Draw (old)
        sample_fixture(TEAM_H, 2, 1, day=4),  # Win (recent)
        sample_fixture(TEAM_H, 1, 0, day=5),  # Win (recent)
    ]

    # Season form: uses decay weights, so recent games matter more
    season = team_form_stats(TEAM_H, all_season)
    # With decay, the 2 recent wins get higher weight than the 3 old games
    # Expected: ~0.4-0.5 depending on decay factor
    season_win_rate = season["win"]

    # Recent momentum: unweighted, just last N games
    momentum = recent_form_momentum(TEAM_H, all_season, window_size=5)
    # 2 wins out of 5 games = 0.4
    assert momentum["momentum"] == 0.4

    # 3-game window: last 2 wins + 1 draw
    momentum_3 = recent_form_momentum(TEAM_H, all_season, window_size=3)
    assert momentum_3["momentum"] == (2.0 / 3.0)  # 2/3 wins in last 3
    assert momentum_3["momentum"] > momentum["momentum"]  # Hot streak in last 3


def test_form_momentum_empty_games():
    """Momentum returns sensible defaults with no games."""
    momentum = recent_form_momentum(TEAM_H, [], window_size=5)

    assert momentum["momentum"] == 0.0
    assert momentum["goals_for"] == 0.0
    assert momentum["goals_against"] == 0.0
    assert momentum["played"] == 0


def test_assemble_prematch_includes_momentum_features():
    """Momentum features are included in the final feature vector."""
    # Home team: 2-0 win, 1-1 draw = 1 win, 1 draw
    last_h = [
        sample_fixture(TEAM_H, 2, 0, is_home=True),  # Home win
        sample_fixture(TEAM_H, 1, 1, is_home=True),  # Home draw
    ]
    # Away team: 2-1 loss (as away), 1-0 win (as away)
    # When TEAM_A is away, sample_fixture(TEAM_A, gf, ga, is_home=False)
    last_a = [
        sample_fixture(TEAM_A, 1, 2, is_home=False),  # Away loss (1 scored, 2 conceded)
        sample_fixture(TEAM_A, 1, 0, is_home=False),  # Away win (1 scored, 0 conceded)
    ]
    h2h = [sample_fixture(TEAM_H, 2, 1, is_home=True)]

    features = assemble_prematch_features(
        home_id=TEAM_H,
        away_id=TEAM_A,
        last_h=last_h,
        last_a=last_a,
        h2h=h2h,
        kickoff_ts=1700000000,
        rating_h=1600.0,
        rating_a=1500.0,
        league_rates={"btts": 0.5, "ov25": 0.5, "ov35": 0.3},
    )

    # New form momentum features should be present
    assert "pm_form_momentum_h" in features
    assert "pm_form_momentum_a" in features
    assert "pm_recent_gf_h" in features
    assert "pm_recent_ga_h" in features
    assert "pm_recent_gf_a" in features
    assert "pm_recent_ga_a" in features
    assert "pm_home_form_h" in features
    assert "pm_away_form_a" in features

    # Home team: 1 win, 1 draw in last 2 = 0.5 win rate
    assert features["pm_form_momentum_h"] == 0.5
    # Away team: 1 loss, 1 win in last 2 = 0.5 win rate
    assert features["pm_form_momentum_a"] == 0.5


def test_form_momentum_helps_distinguish_teams():
    """Momentum captures form differences not visible in season stats."""
    # Team A: 1-1 in last 2 (just stabilized)
    team_a_recent = [
        sample_fixture(TEAM_H, 1, 0),  # Win
        sample_fixture(TEAM_H, 0, 1),  # Loss
    ]

    # Team B: 0-2 in last 2 (sliding)
    team_b_recent = [
        sample_fixture(TEAM_A, 0, 1),  # Loss
        sample_fixture(TEAM_A, 1, 2),  # Loss
    ]

    momentum_a = recent_form_momentum(TEAM_H, team_a_recent, window_size=2)
    momentum_b = recent_form_momentum(TEAM_A, team_b_recent, window_size=2)

    # Team A has momentum (0.5 win rate), Team B doesn't (0.0)
    assert momentum_a["momentum"] > momentum_b["momentum"]


# ───────── regressions ─────────

def test_window_is_chosen_by_date_not_by_list_position():
    """
    The live path and the historical backfill order their windows opposite
    ways: API-Football's /fixtures?last= returns most-recent-first, while the
    backfill slices [-5:] off an ascending history. Selecting the window with
    finished[-window_size:] therefore meant "newest N" while training and
    "oldest N" while serving — the same fixture scored two different ways.
    Both orderings must now produce the same numbers.
    """
    oldest_first = [
        sample_fixture(TEAM_H, 0, 3, day=1),  # heavy loss, oldest
        sample_fixture(TEAM_H, 0, 2, day=2),  # loss
        sample_fixture(TEAM_H, 2, 0, day=3),  # win
        sample_fixture(TEAM_H, 3, 0, day=4),  # win
        sample_fixture(TEAM_H, 1, 0, day=5),  # win, newest
    ]
    newest_first = list(reversed(oldest_first))

    backfill = recent_form_momentum(TEAM_H, oldest_first, window_size=3)
    live = recent_form_momentum(TEAM_H, newest_first, window_size=3)

    assert backfill == live, "train and serve disagree on the same fixture"
    # ...and it is genuinely the three most RECENT games (3 wins), not the oldest.
    assert backfill["momentum"] == 1.0
    assert backfill["goals_against"] == 0.0


def test_played_counts_only_this_teams_games():
    """
    A fixture the team did not play in is skipped, so counting it in the
    denominator would dilute every average toward zero. team_form_stats()
    already counts this way.
    """
    OTHER_A, OTHER_B = 777, 888
    window = [
        sample_fixture(TEAM_H, 2, 0, day=3),  # this team: a win
        {  # two strangers, no TEAM_H involvement
            "fixture": {"id": 1, "status": {"short": "FT"},
                        "date": "2026-09-02T15:00:00Z"},
            "teams": {"home": {"id": OTHER_A}, "away": {"id": OTHER_B}},
            "goals": {"home": 1, "away": 1},
            "date": "2026-09-02T15:00:00Z",
        },
    ]
    m = recent_form_momentum(TEAM_H, window, window_size=5)

    assert m["played"] == 1          # not 2
    assert m["momentum"] == 1.0      # 1 win / 1 game, not 1/2
    assert m["goals_for"] == 2.0     # not 1.0


def test_prematch_vector_has_no_duplicated_columns():
    """
    pm_goals_momentum_h/a were assigned the same expression as
    pm_recent_gf_h/a, so the matrix carried two pairs of identical columns.
    Under L2 that splits one effect arbitrarily between them and makes both
    coefficients — and feature_importance — meaningless. This is the defect
    the feature_spec header records as having been purged from the old lists.
    """
    last_h = [sample_fixture(TEAM_H, 3, 1, day=d) for d in (1, 2, 3)]
    last_a = [sample_fixture(TEAM_A, 0, 2, is_home=False, day=d) for d in (1, 2)]

    features = assemble_prematch_features(
        home_id=TEAM_H, away_id=TEAM_A,
        last_h=last_h, last_a=last_a, h2h=[],
        kickoff_ts=1700000000, rating_h=1600.0, rating_a=1500.0,
        league_rates={"btts": 0.5, "ov25": 0.5, "ov35": 0.3},
    )

    dupes = {}
    for name, value in features.items():
        dupes.setdefault(value, []).append(name)
    collisions = {v: names for v, names in dupes.items() if len(names) > 1}

    # Distinct features may coincide numerically by luck; the specific pair
    # that was literally the same expression must not both be present.
    assert not ("pm_goals_momentum_h" in features and "pm_recent_gf_h" in features), \
        f"duplicated column pair is back: {collisions}"
    assert "pm_goals_momentum_h" not in features
    assert "pm_goals_momentum_a" not in features
