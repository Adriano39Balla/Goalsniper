"""
railway.json points its healthcheck at /health with a 30s timeout, and
restartPolicyType ON_FAILURE. If a build boots but /health doesn't answer
200, Railway fails the deploy and silently keeps serving the previous
build - which looks exactly like "my push never deployed".

These are cheap guards against shipping that: the app object imports, the
healthcheck path answers, and the routes the dashboard calls are actually
registered.
"""
import json

import main


def _healthcheck_path():
    with open("railway.json") as fh:
        return json.load(fh)["deploy"]["healthcheckPath"]


class _CountingConn:
    """A cursor that answers the health probe's COUNT(*) like a real DB."""

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=()):
        return self

    def fetchone(self):
        return (42,)


def test_healthcheck_path_answers_200_against_a_working_db(monkeypatch):
    # conftest's fake cursor returns None for everything, so /health's
    # fetchone()[0] would raise there - that's the fake, not the handler.
    # Give it a real-shaped row and the success path must come back 200.
    monkeypatch.setattr(main, "db_conn", lambda: _CountingConn())
    r = main.app.test_client().get(_healthcheck_path())
    assert r.status_code == 200
    assert r.get_json()["ok"] is True


def test_healthcheck_path_in_railway_json_is_actually_routed():
    # A healthcheckPath pointing at a route that doesn't exist would fail
    # every deploy while the previous build kept serving.
    assert _healthcheck_path() in {str(r) for r in main.app.url_map.iter_rules()}


def test_dashboard_routes_the_frontend_calls_are_registered():
    rules = {str(r) for r in main.app.url_map.iter_rules()}
    for path in ("/dashboard", "/dashboard/data", "/dashboard/live",
                 "/dashboard/live/refresh", "/dashboard/match/<int:fid>/form"):
        assert path in rules, f"{path} is not registered"


# ───────── which commit is actually running ─────────
#
# "Did my push deploy?" was unanswerable from outside the Railway dashboard,
# and getting it wrong wastes real time: a push that never deployed looks
# exactly like a change that didn't work.

def test_build_info_reports_the_deployed_commit(monkeypatch):
    monkeypatch.setenv("RAILWAY_GIT_COMMIT_SHA", "1a8aa03fcb2cb3ffb0e6297c4959bebfb6cce9bd")
    monkeypatch.setenv("RAILWAY_GIT_BRANCH", "main")
    info = main.build_info()
    assert info["commit"] == "1a8aa03"
    assert info["commit_full"].startswith("1a8aa03")
    assert info["branch"] == "main"


def test_build_info_degrades_when_not_running_on_railway(monkeypatch):
    monkeypatch.delenv("RAILWAY_GIT_COMMIT_SHA", raising=False)
    monkeypatch.delenv("RAILWAY_GIT_BRANCH", raising=False)
    info = main.build_info()
    assert info["commit"] == "unknown"
    assert info["commit_full"] is None
    assert info["started_ts"] > 0


def test_dashboard_payload_carries_the_build(monkeypatch):
    monkeypatch.setenv("RAILWAY_GIT_COMMIT_SHA", "deadbeefcafe")
    monkeypatch.setattr(main, "_dashboard_authed", lambda: True)
    monkeypatch.setattr(main, "DASHBOARD_ENABLED", True)
    monkeypatch.setattr(main, "compute_pnl", lambda **k: {"n_bets": 0})
    r = main.app.test_client().get("/dashboard/data")
    assert r.status_code == 200
    assert r.get_json()["build"]["commit"] == "deadbee"
