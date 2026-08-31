"""
Goalsniper Desktop - a native window wrapper around your existing Railway
dashboard. It doesn't run any scanning or scoring itself; it just opens
/dashboard in its own window instead of a browser tab, so opening it feels
like launching a program rather than navigating to a page.

The "Refresh now" button on the Live matches section (already part of the
dashboard) triggers an on-demand scan of whatever's live right now, scored
against your real trained models - that's what actually answers "what's
live and what does the model think" each time you open this.

One-time setup:
    pip install pywebview

Run:
    python goalsniper_desktop.py

First launch shows the normal dashboard login - enter your ADMIN_API_KEY
once. The session cookie is stored in this window's own local profile
(separate from any browser), so you generally won't need to log in again on
later launches unless the cookie expires or you sign out.
"""
import webview

# Your Railway deployment's base URL.
GOALSNIPER_URL = "https://goalsniper-production-85b3.up.railway.app"

if __name__ == "__main__":
    webview.create_window(
        "Goalsniper",
        f"{GOALSNIPER_URL}/dashboard",
        width=1100,
        height=820,
        min_size=(720, 560),
    )
    webview.start()
