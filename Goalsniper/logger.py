from datetime import datetime, timezone

def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()

def log(*args):
    print(_ts(), "-", *args, flush=True)

def warn(*args):
    print(_ts(), "[warn]", *args, flush=True)

def error(*args):
    print(_ts(), "[error]", *args, flush=True)
