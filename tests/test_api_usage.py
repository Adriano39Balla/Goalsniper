"""
_track_api_call() backs the /admin/status "api_usage" counter that makes 429
rate-limiting visible (previously logged at DEBUG, invisible under the app's
default INFO level). These tests exercise the counting/reset logic directly,
without going through a real HTTP call.
"""
import main


def _reset_stats():
    with main._api_call_lock:
        main._api_call_stats.update(day=None, total=0, rate_limited=0)


def test_successful_call_increments_total_but_not_rate_limited():
    _reset_stats()
    main._track_api_call(None)
    main._track_api_call(None)
    snap = main._api_call_stats_snapshot()
    assert snap["total"] == 2
    assert snap["rate_limited"] == 0


def test_429_increments_both_total_and_rate_limited():
    _reset_stats()
    main._track_api_call(None)
    main._track_api_call(429)
    snap = main._api_call_stats_snapshot()
    assert snap["total"] == 2
    assert snap["rate_limited"] == 1


def test_non_429_error_increments_total_but_not_rate_limited():
    _reset_stats()
    main._track_api_call(500)
    snap = main._api_call_stats_snapshot()
    assert snap["total"] == 1
    assert snap["rate_limited"] == 0


def test_counters_reset_when_the_calendar_day_changes():
    _reset_stats()
    main._track_api_call(429)
    assert main._api_call_stats_snapshot()["total"] == 1

    # Force a stale day so the next call is treated as a new day.
    with main._api_call_lock:
        main._api_call_stats["day"] = "2000-01-01"
    main._track_api_call(None)

    snap = main._api_call_stats_snapshot()
    assert snap["total"] == 1  # yesterday's call was not carried over
    assert snap["rate_limited"] == 0
