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
