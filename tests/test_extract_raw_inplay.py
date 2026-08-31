"""
extract_raw_inplay() pulls RAW_INPLAY_KEYS out of a live /fixtures?live=all
match object. These tests cover the newly-added fields (yellow cards,
goalkeeper saves, passes) to confirm they're read from the exact API stat
type strings ("Yellow Cards", "Goalkeeper Saves", "Total passes",
"Passes accurate") API-Football actually returns in that same response.
"""
import main


def _match(home_stats, away_stats):
    return {
        "fixture": {"status": {"elapsed": 60}},
        "teams": {"home": {"name": "Home FC"}, "away": {"name": "Away FC"}},
        "goals": {"home": 1, "away": 0},
        "events": [],
        "statistics": [
            {"team": {"name": "Home FC"}, "statistics": [
                {"type": k, "value": v} for k, v in home_stats.items()]},
            {"team": {"name": "Away FC"}, "statistics": [
                {"type": k, "value": v} for k, v in away_stats.items()]},
        ],
    }


def test_yellow_cards_saves_and_passes_are_extracted():
    m = _match(
        {"Yellow Cards": 2, "Goalkeeper Saves": 3, "Total passes": 400, "Passes accurate": 350},
        {"Yellow Cards": 1, "Goalkeeper Saves": 5, "Total passes": 300, "Passes accurate": 240},
    )
    raw = main.extract_raw_inplay(m)
    assert raw["yellow_h"] == 2.0
    assert raw["yellow_a"] == 1.0
    assert raw["saves_h"] == 3.0
    assert raw["saves_a"] == 5.0
    assert raw["passes_h"] == 400.0
    assert raw["passes_a"] == 300.0
    assert raw["passes_acc_h"] == 350.0
    assert raw["passes_acc_a"] == 240.0


def test_missing_new_stat_types_default_to_zero():
    # A stats block that doesn't include these types (e.g. a provider outage
    # for that fixture) must not raise - it should read as zero, same as the
    # existing fields already handle a missing "Fouls"/"Corner Kicks".
    m = _match({}, {})
    raw = main.extract_raw_inplay(m)
    assert raw["yellow_h"] == raw["yellow_a"] == 0.0
    assert raw["saves_h"] == raw["saves_a"] == 0.0
    assert raw["passes_h"] == raw["passes_acc_h"] == 0.0
