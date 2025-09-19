# file: gunicorn_conf.py
bind = "0.0.0.0:{}".format(__import__("os").environ.get("PORT", "8080"))
workers = 1                  # keep 1 so scheduler runs once
worker_class = "gthread"
threads = 8
timeout = 180
graceful_timeout = 30
loglevel = "info"
accesslog = "-"
errorlog = "-"
