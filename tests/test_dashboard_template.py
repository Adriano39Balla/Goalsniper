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


# Team names, league/country names and bookmaker names come from
# API-Football, not from goalsniper, and the feed cards build their HTML with
# JS template literals assigned via innerHTML. An unescaped one is a stored
# XSS hole the moment a feed ever carries a hostile string - the exact class
# of bug main.py's own escape() calls around home/away/league/suggestion
# (before they reach Telegram's HTML parse mode) already guard against on the
# bot side. These checks make sure the dashboard's JS applies the same guard
# rather than interpolating that data straight into innerHTML.
def _dashboard_js():
    with main.app.test_request_context("/dashboard"):
        return main.render_template("dashboard.html", refresh_sec=30)


def test_dashboard_defines_an_html_escaper():
    assert "function escapeHtml(" in _dashboard_js()


def test_external_team_and_league_fields_are_escaped_before_interpolation():
    html = _dashboard_js()
    # Every place a live/tip item's team names, or a league/country split off
    # the API's own league string, reach a template literal, they must go
    # through escapeHtml() rather than being interpolated raw.
    for unescaped in ("${m.home}", "${m.away}", "${t.home}", "${t.away}",
                      "${c.team}", "${name}", "${country}"):
        assert unescaped not in html, (
            f"{unescaped} is interpolated without escapeHtml() - "
            f"external API data would be injected into innerHTML unescaped")
    for escaped in ("${escapeHtml(m.home)}", "${escapeHtml(m.away)}",
                    "${escapeHtml(t.home)}", "${escapeHtml(t.away)}",
                    "${escapeHtml(c.team)}", "${escapeHtml(name)}",
                    "${escapeHtml(country)}"):
        assert escaped in html, f"expected {escaped} in the rendered template"


def test_bookmaker_name_is_escaped_before_interpolation():
    html = _dashboard_js()
    assert "escapeHtml(t.book)" in html
