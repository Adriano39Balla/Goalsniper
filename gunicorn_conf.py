# file: gunicorn_conf.py
import os

# Bind to platform port
bind = f"0.0.0.0:{os.environ.get('PORT', '8080')}"

# Workers & threads
workers = int(os.environ.get("WEB_CONCURRENCY", "1"))
worker_class = "gthread"
threads = int(os.environ.get("GTHREADS", "4"))

# Timeouts
timeout = int(os.environ.get("GUNICORN_TIMEOUT", "120"))
graceful_timeout = int(os.environ.get("GUNICORN_GRACEFUL_TIMEOUT", "30"))
keepalive = int(os.environ.get("GUNICORN_KEEPALIVE", "5"))

# Memory leak safety
max_requests = int(os.environ.get("GUNICORN_MAX_REQUESTS", "1000"))
max_requests_jitter = int(os.environ.get("GUNICORN_MAX_REQUESTS_JITTER", "200"))

# Logging
loglevel = os.environ.get("GUNICORN_LOGLEVEL", "info")
accesslog = "-"   # stdout
errorlog = "-"    # stderr
capture_output = True

# Safer restarts / imports
preload_app = False  # avoid double-import side effects with scheduler

# Request limits (optional hardening)
limit_request_line = int(os.environ.get("GUNICORN_LIMIT_REQUEST_LINE", "4094"))
limit_request_fields = int(os.environ.get("GUNICORN_LIMIT_REQUEST_FIELDS", "100"))
limit_request_field_size = int(os.environ.get("GUNICORN_LIMIT_REQUEST_FIELD_SIZE", "8190"))
