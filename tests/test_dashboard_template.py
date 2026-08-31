"""
dashboard.html references url_for("dashboard_live") and
url_for("dashboard_live_refresh") directly - a typo in either the route name
or the template would only surface at request time in production (Jinja
doesn't check url_for targets at import time). This test renders the real
template through Flask so that coupling is checked here.
"""
import main


def test_dashboard_template_renders_and_references_live_endpoints():
    with main.app.test_request_context("/dashboard"):
        html = main.render_template("dashboard.html", refresh_sec=30)
    assert "/dashboard/live" in html
    assert "/dashboard/live/refresh" in html
    assert "/dashboard/data" in html
