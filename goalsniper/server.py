# goalsniper/server.py
from __future__ import annotations

import os
import importlib
import asyncio
from datetime import datetime, timezone
from typing import Optional, Any, Dict

import httpx
from fastapi import FastAPI, Request, HTTPException, Query, Response

app = FastAPI(title="Goalsniper", version="1.5.0")

# -------------------------
# env helpers
# -------------------------
def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default)

def _run_token() -> str:
    tok = _env("RUN_TOKEN", "")
    if not tok:
        # don't crash process; surface clearly
        raise HTTPException(status_code=500, detail="RUN_TOKEN not set")
    return tok

def _telegram_webhook_token() -> str:
    return _env("TELEGRAM_WEBHOOK_TOKEN", "")

def _safe_import(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"import error: {module_name}: {e}")

def _auth_header(request: Request):
    auth = request.headers.get("authorization") or ""
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    token = auth[len("Bearer "):].strip()
    if token != _run_token():
        raise HTTPException(status_code=403, detail="Forbidden")

def _auth_qs(token: str):
    if not token or token != _run_token():
        raise HTTPException(status_code=401, detail="Unauthorized")

# -------------------------
# Health & root (HEAD-safe)
# -------------------------
@app.api_route("/health", methods=["GET", "HEAD"])
async def health(request: Request):
    if request.method == "HEAD":
        return Response(status_code=200)
    return {"ok": True, "name": "Goalsniper", "time": os.getenv("TZ", "UTC")}

@app.api_route("/", methods=["GET", "HEAD"])
async def root(request: Request):
    if request.method == "HEAD":
        return Response(status_code=200)
    return {"ok": True, "name": "Goalsniper", "time": os.getenv("TZ", "UTC")}

# -------------------------
# Scan orchestration (non‑blocking)
# -------------------------
_scan_lock = asyncio.Lock()
_scan_task: Optional[asyncio.Task] = None
_last_run_started: Optional[str] = None   # ISO time
_last_run_finished: Optional[str] = None  # ISO time
_last_run_result: Optional[Dict[str, Any]] = None
_last_run_error: Optional[str] = None

async def _do_scan():
    """Run one scan and update status snapshots."""
    global _last_run_started, _last_run_finished, _last_run_result, _last_run_error
    scanner = importlib.import_module("goalsniper.scanner")
    _last_run_error = None
    _last_run_started = datetime.now(timezone.utc).isoformat()
    try:
        res = await scanner.run_scan_and_send()
        _last_run_result = {
            "fixturesChecked": res.get("fixturesChecked", 0),
            "tipsSent": res.get("tipsSent", 0),
            "elapsedSeconds": res.get("elapsedSeconds", 0),
            "debug": res.get("debug", {}),
        }
    except Exception as e:
        _last_run_error = str(e)
        _last_run_result = None
    finally:
        _last_run_finished = datetime.now(timezone.utc).isoformat()

def _kick_off_background_scan() -> bool:
    """Start a background scan if none is running. Returns True if started, False if already running."""
    global _scan_task
    if _scan_task and not _scan_task.done():
        return False
    # create a fresh task
    _scan_task = asyncio.create_task(_do_scan())
    return True

# -----------------------------------------
# /run (non‑blocking)  — GET ?token=...  or  POST with Authorization: Bearer
# -----------------------------------------
@app.post("/run")
async def run_post(request: Request):
    _auth_header(request)
    started = _kick_off_background_scan()
    return Response(
        content=(b'{"accepted":true,"started":' + (b"true" if started else b"false") + b"}"),
        status_code=202,
        media_type="application/json",
    )

@app.api_route("/run", methods=["GET", "HEAD"])
async def run_get(request: Request, token: str = Query("")):
    _auth_qs(token)
    if request.method == "HEAD":
        return Response(status_code=200)
    started = _kick_off_background_scan()
    return Response(
        content=(b'{"accepted":true,"started":' + (b"true" if started else b"false") + b"}"),
        status_code=202,
        media_type="application/json",
    )

# -----------------------------------------
# /run/wait — runs scan and waits (blocking)
# -----------------------------------------
@app.post("/run/wait")
async def run_wait_post(request: Request):
    _auth_header(request)
    return await _run_wait_impl()

@app.get("/run/wait")
async def run_wait_get(token: str = Query("")):
    _auth_qs(token)
    return await _run_wait_impl()

async def _run_wait_impl():
    async with _scan_lock:
        # ensure no overlap; if a background one is going, wait for it
        global _scan_task
        if _scan_task and not _scan_task.done():
            await _scan_task
            _scan_task = None
        # now run a dedicated scan
        await _do_scan()
        return {
            "status": "ok",
            "startedAt": _last_run_started,
            "finishedAt": _last_run_finished,
            "result": _last_run_result,
            "error": _last_run_error,
        }

# -----------------------------------------
# /run/status — inspect last/background state
# -----------------------------------------
@app.get("/run/status")
async def run_status(token: str = Query("")):
    _auth_qs(token)
    running = bool(_scan_task and not _scan_task.done())
    return {
        "running": running,
        "startedAt": _last_run_started,
        "finishedAt": _last_run_finished,
        "result": _last_run_result,
        "error": _last_run_error,
    }

# -------------------------
# Telegram webhook (feedback)
# -------------------------
@app.post("/telegram/webhook")
@app.post("/telegram/webhook/{token}")
async def telegram_webhook(request: Request, token: Optional[str] = None):
    secret = _telegram_webhook_token()
    if secret and token != secret:
        raise HTTPException(status_code=403, detail="Forbidden")

    payload = await request.json()
    cq = payload.get("callback_query")
    if not cq:
        return {"ok": True}

    data = (cq.get("data") or "").strip()
    # expected: fb:<tip_id>:<1|0>
    if not data.startswith("fb:"):
        return {"ok": True}

    try:
        _, tip_id_s, outcome_s = data.split(":")
        tip_id = int(tip_id_s)
        outcome = 1 if outcome_s == "1" else 0

        storage = _safe_import("goalsniper.storage")
        learning = _safe_import("goalsniper.learning")
        logger = _safe_import("goalsniper.logger")

        await storage.set_outcome(tip_id, outcome)

        try:
            info = await learning.on_feedback_update(tip_id, outcome)
            if info.get("ok"):
                logger.log(
                    f"[learn] tip={tip_id} market={info.get('market')} "
                    f"p_cal={info.get('calibratedProb'):.3f} conf={info.get('confidence'):.3f} "
                    f"thr={info.get('marketThreshold'):.3f} outcome={outcome}"
                )
        except Exception as le:
            logger.log(f"[learn] update failed for tip_id={tip_id}: {le}")

        logger.log(f"Feedback received tip_id={tip_id} outcome={outcome}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"bad data: {str(e)}")

    return {"ok": True}

# -------------------------
# Daily digest
# -------------------------
@app.api_route("/digest", methods=["GET", "HEAD"])
async def digest_get(
    request: Request,
    token: str = Query(""),
    date: str | None = Query(None, description="YYYY-MM-DD in UTC (optional)"),
    push: int = Query(1, description="1=send to Telegram, 0=JSON only"),
):
    _auth_qs(token)

    if request.method == "HEAD":
        return Response(status_code=200)

    storage = _safe_import("goalsniper.storage")
    telegram = _safe_import("goalsniper.telegram")

    try:
        day = (
            datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            if date else datetime.now(timezone.utc)
        )
    except Exception:
        raise HTTPException(status_code=400, detail="Bad date format, use YYYY-MM-DD")

    stats = await storage.daily_counts_for(day)
    tot = await storage.totals()

    sent = stats["sent"]; good = stats["good"]; bad = stats["bad"]; pending = stats["pending"]
    acc = (good / sent * 100.0) if sent else 0.0

    day_str = day.strftime("%Y-%m-%d")
    text = (
        f"📅 <b>Daily Goalsniper Digest — {day_str}</b>\n"
        f"Sent: <b>{sent}</b>  |  👍 <b>{good}</b>  |  👎 <b>{bad}</b>  |  ⏳ <b>{pending}</b>\n"
        f"Accuracy: <b>{acc:.1f}%</b>\n"
        f"\n<i>All‑time</i> — Sent {tot['sent']}, 👍 {tot['good']}, 👎 {tot['bad']}"
    )

    result = {
        "date": day_str,
        "sent": sent,
        "good": good,
        "bad": bad,
        "pending": pending,
        "accuracy": round(acc, 1),
        "pushed": False,
    }

    if push:
        async with httpx.AsyncClient(timeout=20) as client:
            try:
                await telegram.send_text(client, text)
                result["pushed"] = True
            except Exception as e:
                result["error"] = f"telegram: {e}"

    return result
