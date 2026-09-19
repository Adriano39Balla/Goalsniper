"""
A wrong league ID in PREMATCH_LEAGUE_IDS fails silently and forever.

_int_list() parses "999999" as happily as "39". _collect_todays_prematch_fixtures()
compares it against the day's card, never matches, and moves on. No exception,
no warning, no counter. The league is simply never scanned again, and the only
symptom is a card that looks like a slow day.

That matters more than usual right now because the list is about to be set from
a catalogue of remembered IDs rather than read off the API, and because
PREMATCH_LEAGUE_IDS is the main lever on API spend: with it unset the scanner
walks every fixture on earth at ~3 calls each, which is what pins the
per-minute limit and starves the in-play scans sharing it.

/admin/diagnostics/league-filter resolves the configured IDs against the API's
own catalogue in one call and replays the filters that run BEFORE the league
list is applied.
"""
import main


def _league(lid, name, country="England", ltype="League", season=2026):
    return {"league": {"id": lid, "name": name, "type": ltype},
            "country": {"name": country},
            "seasons": [{"year": season, "current": True}]}


def _stub_api(monkeypatch, response):
    seen = {}

    def _get(url, params, timeout=15):
        seen["url"], seen["params"] = url, params
        return response

    monkeypatch.setattr(main, "_api_get", _get)
    return seen


def _configure(monkeypatch, ids, raw=None):
    monkeypatch.setattr(main, "PREMATCH_LEAGUE_IDS", ids)
    monkeypatch.setenv("PREMATCH_LEAGUE_IDS", raw if raw is not None else
                       ",".join(str(i) for i in ids))


# ───────────────── the thing it exists to catch ─────────────────

def test_an_id_that_names_no_real_league_is_reported(monkeypatch):
    _configure(monkeypatch, [39, 999999])
    _stub_api(monkeypatch, {"response": [_league(39, "Premier League")]})

    out = main.check_league_filter()

    assert out["unresolved_ids"] == [999999]
    assert [r["id"] for r in out["resolved"]] == [39]
    assert "999999" in out["verdict"]
    assert "silently never scanned" in out["verdict"]


def test_a_fully_valid_list_says_nothing_is_being_dropped(monkeypatch):
    _configure(monkeypatch, [39, 140])
    _stub_api(monkeypatch, {"response": [_league(39, "Premier League"),
                                         _league(140, "La Liga", country="Spain")]})

    out = main.check_league_filter()

    assert out["unresolved_ids"] == []
    assert out["blocked_ids"] == []
    assert "Nothing is being silently dropped" in out["verdict"]
    assert {r["country"] for r in out["resolved"]} == {"England", "Spain"}


def test_a_non_numeric_entry_is_reported_rather_than_silently_dropped(monkeypatch):
    """
    _int_list() keeps only entries that parse as integers and discards the rest
    without a word, so `PREMATCH_LEAGUE_IDS=39,EPL,140` configures two leagues
    while reading as three.
    """
    _configure(monkeypatch, [39, 140], raw="39, EPL, 140")
    _stub_api(monkeypatch, {"response": [_league(39, "Premier League"),
                                         _league(140, "La Liga", country="Spain")]})

    out = main.check_league_filter()

    assert out["unparsed_entries"] == ["EPL"]
    assert "not numbers" in out["verdict"]


# ───────────────── filters that run earlier ─────────────────

def test_an_id_the_hard_allowlist_already_drops_is_reported(monkeypatch):
    """
    _blocked_league() runs BEFORE the PREMATCH_LEAGUE_IDS filter, and
    LEAGUE_ALLOW_IDS is absolute: anything not on it is dropped no matter what
    else is configured. Adding a league to PREMATCH_LEAGUE_IDS while
    LEAGUE_ALLOW_IDS omits it does nothing at all, which is impossible to see
    from either variable on its own.
    """
    _configure(monkeypatch, [39, 140])
    monkeypatch.setenv("LEAGUE_ALLOW_IDS", "39")
    _stub_api(monkeypatch, {"response": [_league(39, "Premier League"),
                                         _league(140, "La Liga", country="Spain")]})

    out = main.check_league_filter()

    assert out["blocked_ids"] == [140]
    assert "BEFORE this filter runs" in out["verdict"]
    blocked = [r for r in out["resolved"] if r["id"] == 140][0]
    assert blocked["blocked_before_this_filter"] is True


def test_a_league_caught_by_the_name_block_patterns_is_reported(monkeypatch):
    _configure(monkeypatch, [4001])
    monkeypatch.delenv("LEAGUE_ALLOW_IDS", raising=False)
    _stub_api(monkeypatch, {"response": [_league(4001, "Premier League U21")]})

    out = main.check_league_filter()

    assert out["blocked_ids"] == [4001]


# ───────────────── refusing to guess ─────────────────

def test_a_failed_catalogue_call_is_not_reported_as_a_list_of_bad_ids(monkeypatch):
    """
    _api_get() returns None when the call failed or was refused — which, on a
    rate-limited key, is exactly when someone is most likely to be poking at
    diagnostics. Treating that as "no league matched" would condemn every
    configured ID on the strength of a call that never happened. Same rule the
    fetch layer already states: a failed call is not an empty result.
    """
    _configure(monkeypatch, [39, 140])
    _stub_api(monkeypatch, None)

    out = main.check_league_filter()

    assert out["catalogue_reachable"] is False
    assert "unresolved_ids" not in out
    assert "not evidence" in out["verdict"]


def test_an_unreachable_catalogue_answers_503_not_200(monkeypatch):
    _configure(monkeypatch, [39])
    _stub_api(monkeypatch, None)
    monkeypatch.setattr(main, "_require_admin", lambda: None)

    r = main.app.test_client().get("/admin/diagnostics/league-filter")

    assert r.status_code == 503
    assert r.get_json()["ok"] is False


# ───────────────── the unset case ─────────────────

def test_an_empty_list_names_the_cost_rather_than_reporting_success(monkeypatch):
    _configure(monkeypatch, [], raw="")
    calls = []
    monkeypatch.setattr(main, "_api_get", lambda *a, **k: calls.append(a) or None)

    out = main.check_league_filter()

    assert calls == [], "an unset list needs no API call to diagnose"
    assert "every fixture" in out["verdict"]
    assert "per-minute limit" in out["verdict"]


# ───────────────── plumbing ─────────────────

def test_one_api_call_covers_the_whole_list(monkeypatch):
    """
    Resolving IDs one at a time would cost a call each — a diagnostic for API
    overspend must not itself be expensive. `current=true` also keeps the
    response to one season per league instead of every season ever played.
    """
    _configure(monkeypatch, [39, 140, 78])
    seen = _stub_api(monkeypatch, {"response": []})

    main.check_league_filter()

    assert seen["url"].endswith("/leagues")
    assert seen["params"] == {"current": "true"}


def test_the_endpoint_is_admin_gated():
    assert main.app.test_client().get("/admin/diagnostics/league-filter").status_code == 401
