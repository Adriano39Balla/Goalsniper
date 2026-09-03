"""
API-Football reports quota, plan and auth failures with HTTP 200.

    {"errors": {"requests": "You have reached the request limit for the day"},
     "results": 0, "response": []}

Read only for `response`, that is indistinguishable from "no fixtures are
live" or "nothing is priced". The system keeps running, quietly blind, and
every downstream diagnostic points somewhere else — the same shape of failure
as the in-play odds parser reading only `bookmakers`, which cost a long hunt
that one line of logging would have ended.
"""
import pytest

import main
from main import api_response_error


class _Resp:
    def __init__(self, js, ok=True, status=200):
        self._js, self.ok, self.status_code, self.text = js, ok, status, str(js)

    def json(self):
        return self._js


def _get(monkeypatch, js, ok=True, status=200):
    monkeypatch.setattr(main, "API_KEY", "k")
    monkeypatch.setattr(main.session, "get",
                        lambda *a, **k: _Resp(js, ok=ok, status=status))
    return main._api_get("https://x/fixtures", {})


# ───────── recognising the error ─────────

def test_a_quota_error_is_recognised():
    assert "request limit" in api_response_error(
        {"errors": {"requests": "You have reached the request limit for the day"},
         "response": []})


def test_a_plan_error_is_recognised():
    assert "plan" in api_response_error(
        {"errors": {"plan": "Your plan does not have access to this endpoint"}}).lower()


def test_an_empty_error_list_is_not_an_error():
    # The success shape: errors is an empty LIST, not a dict.
    assert api_response_error({"errors": [], "results": 3, "response": [1, 2, 3]}) is None


def test_an_error_list_is_read_too():
    assert "bad" in api_response_error({"errors": ["bad token"], "response": []})


def test_a_missing_errors_key_is_not_an_error():
    assert api_response_error({"response": []}) is None


def test_a_non_dict_body_is_not_an_error():
    assert api_response_error(None) is None
    assert api_response_error([1, 2]) is None


# ───────── how the caller sees it ─────────

def test_an_errored_200_is_reported_as_a_failed_call(monkeypatch, caplog):
    with caplog.at_level("WARNING"):
        out = _get(monkeypatch, {"errors": {"requests": "limit reached"}, "response": []})
    assert out is None, "an error must not be handed back as data"
    msg = " ".join(r.getMessage() for r in caplog.records)
    assert "limit reached" in msg
    assert "not 'nothing was live'" in msg


def test_a_clean_response_passes_through(monkeypatch):
    js = {"errors": [], "results": 1, "response": [{"fixture": {"id": 1}}]}
    assert _get(monkeypatch, js) == js


def test_errored_calls_are_counted_separately_from_rate_limits(monkeypatch):
    main._api_call_stats.update(day=None, total=0, rate_limited=0, api_errors=0)
    _get(monkeypatch, {"errors": {"token": "invalid"}, "response": []})
    snap = main._api_call_stats_snapshot()
    assert snap["api_errors"] == 1
    assert snap["rate_limited"] == 0, "this is not a 429 and must not read as one"


# ───────── it must not poison the cache ─────────

def test_a_failed_odds_call_is_not_cached_as_no_odds(monkeypatch):
    # Caching {} here would extend a transient outage for the whole TTL and
    # make every candidate in that window read as no_odds, which looks like a
    # pricing problem rather than an outage.
    main.ODDS_CACHE.invalidate()
    monkeypatch.setattr(main, "_api_get", lambda url, params, timeout=15: None)
    assert main.fetch_odds(42, live=True) == {}

    good = {"response": [{"fixture": {"id": 42}, "odds": [
        {"name": "Goals Over/Under",
         "values": [{"value": "Over 2.5", "odd": "2.05"},
                    {"value": "Under 2.5", "odd": "1.80"}]}]}]}
    monkeypatch.setattr(main, "_api_get", lambda url, params, timeout=15: good)
    out = main.fetch_odds(42, live=True)
    assert "OU_2.5" in out, "the next call must not be served a cached failure"


def test_a_genuinely_empty_response_is_still_cached(monkeypatch):
    # No fixtures priced yet is a real answer and worth caching.
    main.ODDS_CACHE.invalidate()
    calls = []
    def _api(url, params, timeout=15):
        calls.append(1)
        return {"errors": [], "response": []}
    monkeypatch.setattr(main, "_api_get", _api)
    assert main.fetch_odds(7, live=True) == {}
    assert main.fetch_odds(7, live=True) == {}
    assert len(calls) == 1, "an empty result should be served from cache"


# ───────── the same rule, in the fetchers that were still breaking it ─────────
#
# fetch_odds() has guarded this since the odds outage that motivated it. The
# other four fetchers still did `_api_get(...) or {}` and cached the resulting
# [], which pins a FAILED call as "this match has no data" for the whole TTL:
# 90s for stats/events, and 30 MINUTES for team form.

@pytest.mark.parametrize("fetch, cache, endpoint", [
    (lambda: main.fetch_match_stats(42), lambda: main.STATS_CACHE, "statistics"),
    (lambda: main.fetch_match_events(42), lambda: main.EVENTS_CACHE, "events"),
])
def test_a_failed_live_fetch_is_not_cached_as_no_data(monkeypatch, fetch, cache, endpoint):
    cache().invalidate()
    monkeypatch.setattr(main, "_api_get", lambda url, params, timeout=15: None)
    assert fetch() == [], "a failed call yields nothing to work with"

    real = [{"team": {"id": 1}, "statistics": [{"type": "Shots on Goal", "value": 4}]}]
    monkeypatch.setattr(main, "_api_get", lambda url, params, timeout=15: {"response": real})
    assert fetch() == real, f"the next {endpoint} call must not be served a cached failure"


def test_a_failed_team_form_fetch_is_not_cached_for_half_an_hour(monkeypatch):
    # TEAM_FORM_CACHE has a 1800s TTL. Caching [] here makes
    # assemble_prematch_features() derive every form feature as 0.0 — as though
    # neither side had ever played a match — and keep doing so for 30 minutes.
    main.TEAM_FORM_CACHE.invalidate()
    monkeypatch.setattr(main, "_api_get", lambda url, params, timeout=15: None)
    assert main._api_last_fixtures(99, 5) == []

    real = [{"fixture": {"id": 5, "status": {"short": "FT"}}}]
    monkeypatch.setattr(main, "_api_get", lambda url, params, timeout=15: {"response": real})
    assert main._api_last_fixtures(99, 5) == real, "a failure must not stick for the TTL"


def test_a_failed_h2h_fetch_is_not_cached(monkeypatch):
    main.TEAM_FORM_CACHE.invalidate()
    monkeypatch.setattr(main, "_api_get", lambda url, params, timeout=15: None)
    assert main._api_h2h(1, 2, 5) == []

    real = [{"fixture": {"id": 9, "status": {"short": "FT"}}}]
    monkeypatch.setattr(main, "_api_get", lambda url, params, timeout=15: {"response": real})
    assert main._api_h2h(1, 2, 5) == real


def test_an_empty_team_form_window_is_still_cached(monkeypatch):
    # A team with genuinely no recent fixtures is a real answer.
    main.TEAM_FORM_CACHE.invalidate()
    calls = []
    def _api(url, params, timeout=15):
        calls.append(1)
        return {"errors": [], "response": []}
    monkeypatch.setattr(main, "_api_get", _api)
    assert main._api_last_fixtures(123, 5) == []
    assert main._api_last_fixtures(123, 5) == []
    assert len(calls) == 1, "an empty window should be served from cache"


# ───────── the per-minute limit ─────────
#
# API-Football reports it as HTTP 200 with the reason in `errors`, so the
# urllib3 Retry mounted on the session (status_forcelist=[429, ...]) never sees
# it and never backs off. Nothing else throttled, and the callers are
# deliberately concurrent, so the first refusal was followed immediately by the
# rest of the burst — production logs show 8 refusals inside 140ms.

PER_MINUTE = {"errors": {"rateLimit":
              "Too many requests. You have exceeded the limit of requests per "
              "minute of your subscription."}, "response": []}


def test_the_per_minute_limit_starts_a_cooldown(monkeypatch):
    main._rate_limit_until = 0.0
    assert _get(monkeypatch, PER_MINUTE) is None
    assert main._rate_limit_cooling_down(), "a refusal must stop the burst"


def test_calls_during_the_cooldown_spend_no_requests(monkeypatch):
    main._rate_limit_until = 0.0
    monkeypatch.setattr(main, "API_KEY", "k")
    sent = []
    monkeypatch.setattr(main.session, "get",
                        lambda *a, **k: sent.append(1) or _Resp(PER_MINUTE))

    assert main._api_get("https://x/fixtures", {}) is None   # the one real refusal
    for _ in range(20):                                       # the rest of the burst
        assert main._api_get("https://x/fixtures", {}) is None

    assert len(sent) == 1, (
        f"sent {len(sent)} requests to be refused; the burst is what exhausts "
        "the minute window in the first place")


def test_the_cooldown_lifts_when_the_window_rolls_over(monkeypatch):
    main._rate_limit_until = 0.0
    assert _get(monkeypatch, PER_MINUTE) is None
    assert main._rate_limit_cooling_down()

    main._rate_limit_until = 0.0  # window rolled over
    good = {"errors": [], "results": 1, "response": [{"fixture": {"id": 1}}]}
    assert _get(monkeypatch, good) == good, "the client must resume on its own"


def test_an_account_fault_is_not_swallowed_as_a_throttle(monkeypatch):
    # An expired plan or bad key is not fixed by waiting, and must stay loud
    # rather than being quietly absorbed into a cooldown.
    main._rate_limit_until = 0.0
    assert _get(monkeypatch, {"errors": {"token": "invalid"}, "response": []}) is None
    assert not main._rate_limit_cooling_down()
