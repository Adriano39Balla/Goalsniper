from __future__ import annotations

import os
import importlib
import asyncio
from datetime import datetime, timezone
from typing import Optional, Any, Dict, Tuple, List

import httpx
from fastapi import FastAPI, Request, HTTPException, Query, Response

app = FastAPI(title="Goalsniper", version="1.6.3")

# -------------------------
# env helpers
# -------------------------
def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default)

def _run_token() -> str:
    tok = _env("RUN_TOKEN", "")
    if not tok:
        raise HTTPException(status_code=500, detail="RUN_TOKEN not set")
    return tok

def _telegram_webhook_token() -> str:
    return _env("TELEGRAM_WEBHOOK_TOKEN", "")

def _safe_import(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"import error: {module_name}: {e}")

# optional logger (defensive import)
try:
    _logger = importlib.import_module("goalsniper.logger")
    log = getattr(_logger, "log", print)
    warn = getattr(_logger, "warn", print)
except Exception:
    def log(*a, **k):  # noqa: N802
        print(*a)
    warn = log

def _auth_header(request: Request):
    auth = request
