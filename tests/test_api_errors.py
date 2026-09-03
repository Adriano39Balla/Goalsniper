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
