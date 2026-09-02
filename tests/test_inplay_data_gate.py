"""
Enough observation to record is not enough observation to bet.

Two live tips went out on a dead statistics feed. The clearest was Macara vs
Manta: tipped BTTS: No at minute 37 with "xG 0.00-0.00 • SOT 3-0". Three shots
on target and exactly zero expected goals is not a cagey game, it is an absent
xG channel - and extract_raw_inplay() defaults a missing "Expected Goals" to
0.0, so the model could not tell the two apart. It read 0.00-0.00 as strong
evidence that neither side was threatening, priced No at 68.3%, and both teams
scored.

That is the failure mode worth gating: a missing feature arriving as a
confident negative observation. stats_coverage_ok() did not catch it because
possession and corners were present, and possession is nonzero from the first
minute of every match - close to a free pass.
"""
import main
from main import _xg_feed_is_dead, inplay_data_gate


def _raw(**kw):
    base = {"minute": 40.0, "xg_h": 0.0, "xg_a": 0.0, "sot_h": 0.0, "sot_a": 0.0,
            "cor_h": 0.0, "cor_a": 0.0, "pos_h": 50.0, "pos_a": 50.0,
            "total_shots_h": 0.0, "total_shots_a": 0.0}
    base.update(kw)
    return base


# ───────── the xG channel ─────────

def test_shots_on_target_with_zero_xg_proves_the_feed_is_absent():
    # No shot on target carries zero expected goals. This state cannot occur.
    assert _xg_feed_is_dead(_raw(sot_h=3.0)) is True


def test_shots_off_target_with_zero_xg_is_caught_too():
    # A feed carrying shot counts but no xG fails the same way.
    assert _xg_feed_is_dead(_raw(total_shots_h=6.0, total_shots_a=2.0)) is True


def test_a_genuinely_chanceless_match_is_not_flagged():
    # No shots and no xG is consistent, and it is real football. Refusing it
    # here would be inventing a fault to explain an ordinary dull half.
    assert _xg_feed_is_dead(_raw()) is False


def test_any_xg_at_all_means_the_channel_is_live():
    assert _xg_feed_is_dead(_raw(sot_h=3.0, xg_h=0.31)) is False
    # Even a trivial value: the question is presence, not magnitude.
    assert _xg_feed_is_dead(_raw(sot_h=3.0, xg_a=0.01)) is False


def test_the_tip_that_lost_would_not_have_been_sent():
    # Macara vs Manta, minute 37: xG 0.00-0.00, SOT 3-0, CK 1-0, POS 59-41.
    macara = _raw(minute=37.0, sot_h=3.0, cor_h=1.0, pos_h=59.0, pos_a=41.0)
    assert main.stats_coverage_ok(macara, 37) is True, "coverage saw 3 of 4 fields"
    assert inplay_data_gate(macara, 37) == "xg_feed_dead"


# ───────── the shot channel ─────────

def test_no_shot_at_all_late_on_is_a_dead_feed(monkeypatch):
    monkeypatch.setattr(main, "SHOT_DATA_MIN_MINUTE", 25)
    # Possession and corners present, entire shot channel missing.
    assert inplay_data_gate(_raw(minute=60.0, cor_h=4.0, pos_h=61.0), 60) == "no_shot_data"


def test_early_on_an_empty_shot_channel_is_still_plausible(monkeypatch):
    monkeypatch.setattr(main, "SHOT_DATA_MIN_MINUTE", 25)
    monkeypatch.setattr(main, "LIVE_TIP_MIN_MINUTE", 15)
    assert inplay_data_gate(_raw(minute=18.0), 18) is None


def test_one_recorded_shot_is_enough_to_call_the_channel_live(monkeypatch):
    monkeypatch.setattr(main, "SHOT_DATA_MIN_MINUTE", 25)
    assert inplay_data_gate(_raw(minute=70.0, total_shots_a=1.0, xg_a=0.04), 70) is None


# ───────── the minute floor ─────────

def test_a_fixture_is_not_bettable_before_the_model_has_seen_its_like(monkeypatch):
    monkeypatch.setattr(main, "LIVE_TIP_MIN_MINUTE", 15)
    assert inplay_data_gate(_raw(minute=13.0, sot_h=1.0, xg_h=0.2), 13) == "too_early"
    assert inplay_data_gate(_raw(minute=15.0, sot_h=1.0, xg_h=0.2), 15) is None


def test_the_floor_defaults_to_where_training_data_starts():
    # Snapshots are harvested from TRAIN_MIN_MINUTE onward, so tipping earlier
    # asks the model to extrapolate outside its own training distribution.
    # Tying the two together is why this is not a hand-picked number.
    assert main.LIVE_TIP_MIN_MINUTE == main.TRAIN_MIN_MINUTE


def test_the_floor_is_above_the_eligibility_minute():
    # TIP_MIN_MINUTE also gates harvesting, so it must stay low. This one is
    # about betting and can be strict without starving the training set.
    assert main.LIVE_TIP_MIN_MINUTE > main.TIP_MIN_MINUTE


def test_the_xg_requirement_can_be_turned_off_without_touching_the_rest(monkeypatch):
    monkeypatch.setattr(main, "REQUIRE_XG_FEED", False)
    monkeypatch.setattr(main, "SHOT_DATA_MIN_MINUTE", 25)
    assert inplay_data_gate(_raw(minute=37.0, sot_h=3.0), 37) is None
    # The shot-channel check is independent and still bites.
    assert inplay_data_gate(_raw(minute=60.0), 60) == "no_shot_data"


def test_a_healthy_fixture_passes_everything(monkeypatch):
    good = _raw(minute=52.0, xg_h=1.12, xg_a=0.43, sot_h=4.0, sot_a=2.0,
                total_shots_h=11.0, total_shots_a=6.0, cor_h=5.0, pos_h=57.0, pos_a=43.0)
    assert inplay_data_gate(good, 52) is None
