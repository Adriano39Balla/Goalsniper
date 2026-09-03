"""
Where the daily API quota actually goes.

The live scan and the prematch scan compete for one budget, and only one of
them produces a closing line to measure against. Spending most of the quota on
the unmeasurable half is a decision worth making on purpose rather than by
default — and it cannot be made at all without the split.
"""
import main
from main import _endpoint_of


def _reset():
    main._api_call_stats.update(day=None, total=0, rate_limited=0, api_errors=0,
                                by_endpoint={})


def test_endpoints_are_labelled_by_resource():
    base = "https://v3.football.api-sports.io"
    assert _endpoint_of(f"{base}/fixtures/statistics") == "/fixtures/statistics"
    assert _endpoint_of(f"{base}/fixtures/events") == "/fixtures/events"
    assert _endpoint_of(f"{base}/odds/live") == "/odds/live"
    assert _endpoint_of(f"{base}/odds") == "/odds"
    assert _endpoint_of(f"{base}/fixtures") == "/fixtures"


def test_the_more_specific_path_wins():
    # /fixtures/statistics must not be filed under /fixtures, or the live
    # scan's real cost disappears into the fixture listing.
    base = "https://v3.football.api-sports.io"
    assert _endpoint_of(f"{base}/fixtures/statistics") != "/fixtures"
    assert _endpoint_of(f"{base}/odds/live") != "/odds"


def test_a_query_string_does_not_create_a_new_bucket():
    base = "https://v3.football.api-sports.io/fixtures"
    assert _endpoint_of(f"{base}?live=all") == _endpoint_of(f"{base}?date=2026-09-03")


def test_calls_are_counted_per_endpoint():
    _reset()
    for _ in range(3):
        main._track_api_call(None, url="https://x/fixtures/statistics")
    main._track_api_call(None, url="https://x/odds")
    snap = main._api_call_stats_snapshot()
    assert snap["total"] == 4
    assert snap["by_endpoint"]["/fixtures/statistics"] == 3
    assert snap["by_endpoint"]["/odds"] == 1


def test_the_biggest_consumer_is_listed_first():
    _reset()
    main._track_api_call(None, url="https://x/odds")
    for _ in range(5):
        main._track_api_call(None, url="https://x/fixtures/events")
    assert list(main._api_call_stats_snapshot()["by_endpoint"])[0] == "/fixtures/events"


def test_the_live_share_is_computed_from_the_live_endpoints():
    # The three the live scan spends on, and only those.
    _reset()
    for url in ("https://x/fixtures/statistics", "https://x/fixtures/events",
                "https://x/odds/live"):
        main._track_api_call(None, url=url)
    main._track_api_call(None, url="https://x/odds")
    snap = main._api_call_stats_snapshot()
    assert snap["live_scan_share_pct"] == 75.0


def test_prematch_only_traffic_reports_a_zero_live_share():
    _reset()
    for _ in range(4):
        main._track_api_call(None, url="https://x/odds")
    assert main._api_call_stats_snapshot()["live_scan_share_pct"] == 0.0


def test_percentages_are_reported_alongside_counts():
    _reset()
    for _ in range(3):
        main._track_api_call(None, url="https://x/odds")
    main._track_api_call(None, url="https://x/fixtures")
    assert main._api_call_stats_snapshot()["by_endpoint_pct"]["/odds"] == 75.0


def test_an_untracked_call_still_counts_toward_the_total():
    # Calls made before this existed, or without a url, must not vanish.
    _reset()
    main._track_api_call(None)
    snap = main._api_call_stats_snapshot()
    assert snap["total"] == 1
    assert snap["by_endpoint"] == {}


def test_the_day_rollover_clears_the_breakdown_too():
    _reset()
    main._track_api_call(None, url="https://x/odds")
    main._api_call_stats["day"] = "1999-01-01"
    main._track_api_call(None, url="https://x/fixtures")
    snap = main._api_call_stats_snapshot()
    assert snap["total"] == 1
    assert snap["by_endpoint"] == {"/fixtures": 1}, "yesterday's split must not carry over"
