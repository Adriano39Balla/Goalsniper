# file: gunicorn_conf.py
# Hardened Gunicorn configuration for goalsniper (Railway/production)

import os
import multiprocessing

# Bind to platform port (Railway injects PORT)
bind = f"0.0.0.0:{os.environ.get('PORT', '8080')}"

# ───────── Workers & threads ─────────
# Auto-scale workers to available CPUs unless explicitly set
_default_workers = max(1, multiprocessing.cpu_count() // 2)
workers = int(os.environ.get("WEB_CONCURRENCY", str(_default_workers)))
worker_class = "gthread"
threads = int(os.environ.get("GTHREADS", "4"))

# ───────── Timeouts ─────────
timeout = int(os.environ.get("GUNICORN_TIMEOUT", "120"))
graceful_timeout = int(os.environ.get("GUNICORN_GRACEFUL_TIMEOUT", "30"))
keepalive = int(os.environ.get("GUNICORN_KEEPALIVE", "5"))

# ───────── Memory leak / resilience ─────────
max_requests = int(os.environ.get("GUNICORN_MAX_REQUESTS", "1000"))
max_requests_jitter = int(os.environ.get("GUNICORN_MAX_REQUESTS_JITTER", "200"))

# ───────── Logging ─────────
loglevel = os.environ.get("GUNICORN_LOGLEVEL", "info")
accesslog = "-"   # stdout (needed for Railway logs)
errorlog = "-"    # stderr
capture_output = True

# ───────── Safer restarts / imports ─────────
# preload_app=False avoids double-import side effects with APScheduler
preload_app = False

# ───────── Request limits (DoS guardrails) ─────────
limit_request_line = int(os.environ.get("GUNICORN_LIMIT_REQUEST_LINE", "4094"))
limit_request_fields = int(os.environ.get("GUNICORN_LIMIT_REQUEST_FIELDS", "100"))
limit_request_field_size = int(os.environ.get("GUNICORN_LIMIT_REQUEST_FIELD_SIZE", "8190"))

# ───────── Hooks (optional observability) ─────────
def post_fork(server, worker):
    server.log.info(f"Worker spawned (pid={worker.pid}) with {threads} threads.")

def worker_exit(server, worker):
    server.log.info(f"Worker exited (pid={worker.pid})")

def pre_exec(server):
    server.log.info("Forked child, re-executing.")

def when_ready(server):
    server.log.info("Gunicorn server is ready. Spawning workers...")
