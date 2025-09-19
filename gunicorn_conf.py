# file: gunicorn_conf.py
import os

# Bind to the platform-provided PORT (Railway/Heroku style)
bind = f"0.0.0.0:{os.environ.get('PORT', '8080')}"

# ───────── Workers & Threads ─────────
# You run APScheduler inside the web process. Keep a single worker so the scheduler
# only lives once here (you already gate with SCHEDULER_LEADER too).
workers = int(os.environ.get("WEB_CONCURRENCY", "1"))

# gthread is a good fit for I/O-bound Flask apps (DB, HTTP calls).
worker_class = "gthread"
threads = int(os.environ.get("GTHREADS", "4"))   # adjust per RAM/CPU

# ───────── Timeouts ─────────
# Long external API calls or DB retries? Keep a generous timeout, but not too big.
timeout = int(os.environ.get("GUNICORN_TIMEOUT", "120"))
graceful_timeout = int(os.environ.get("GUNICORN_GRACEFUL_TIMEOUT", "30"))
# Keep connections alive a bit to reuse sockets
keepalive = int(os.environ.get("GUNICORN_KEEPALIVE", "5"))

# ───────── Memory Leak Safety ─────────
# If anything leaks, this recycles workers gradually.
max_requests = int(os.environ.get("GUNICORN_MAX_REQUESTS", "1000"))
max_requests_jitter = int(os.environ.get("GUNICORN_MAX_REQUESTS_JITTER", "200"))

# ───────── Logging ─────────
loglevel = os.environ.get("GUNICORN_LOGLEVEL", "info")
accesslog = "-"   # stdout
errorlog  = "-"   # stderr
capture_output = True

# ───────── Misc

```python
# file: gunicorn_conf.py (continued)

# Preload can reduce RSS slightly, but if you ever scale workers >1, preloading can
# double-run module import side effects. You have guards in main.py; keep this False.
preload_app = False

# Limit header size a bit (defense-in-depth; optional)
limit_request_line = int(os.environ.get("GUNICORN_LIMIT_REQUEST_LINE", "4094"))
limit_request_fields = int(os.environ.get("GUNICORN_LIMIT_REQUEST_FIELDS", "100"))
limit_request_field_size = int(os.environ.get("GUNICORN_LIMIT_REQUEST_FIELD_SIZE", "8190"))

# If behind a proxy (Railway), trust X-Forwarded-* headers via a proxy downstream.
# If you run your own proxy, consider setting 'forwarded_allow_ips' or using proxy_protocol.
# forwarded_allow_ips = "*"
