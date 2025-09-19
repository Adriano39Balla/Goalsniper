# gunicorn_conf.py
import os
bind = f"0.0.0.0:{os.environ.get('PORT','8080')}"
workers = 1
worker_class = "gthread"
threads = 4          # lower to reduce RAM
timeout = 120
graceful_timeout = 30
loglevel = "info"
accesslog = "-"
errorlog = "-"
