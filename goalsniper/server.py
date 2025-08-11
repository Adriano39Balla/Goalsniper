from __future__ import annotations

import os
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, Request, HTTPException, Query, Response

# Keep boot ultra‑light: only stdlib + FastAPI here.
# Do **not** import storage/scanner/telegram/learning at module import time.

app = FastAPI(title="Goalsniper", version="1.4.3")


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def _run_token() -> str:
    # Pull on demand so server can start even if .env is missing at first.
    return _env("RUN_TOKEN", "change_me_to_a_long_random_string")


def _telegram_webhook_token() -> str:
    return _env("TELEGRAM_WEBHOOK_TOKEN", "")


def _safe_import(path: str):
    """
    Import a module by dotted path and return it (raise with helpful message if it fails).
    """
    try:
        __import__(path)
        return globals()[path.split(".")[0]] if "." not in path else __import__(path, fromlist=["*"])
    except Exception as e:
        # Surface the *real* reason in logs and HTTP responses.
        raise HTTPException(status_code=500, detail=f"import error: {path}: {e}")


def _auth_header(request: Request):
    auth = request.headers.get("authorization") or ""
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    token = auth[len("Bearer "):].strip()
    if token != _run_token():
        raise HTTPException(status_code=403, detail="Forbidden")


def _auth_qs(token: str):
    if token != _run_token():
        raise HTTPException(status_code=401, detail="Unauthorized")


# --- Health & root (HEAD-safe) ------------------------------------------------
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


# --- Run scan (POST header auth, GET query auth; HEAD is a no-op) -------------
@app.post("/run")
async def run_post(request: Request):
    _auth_header(request)
    # Lazy import scanner at call time
    scanner = _safe_import("goalsniper.scanner")
    # run
    result = await scanner.run_scan_and_send()
    return {"status": "ok", **result}

@app.api_route("/run", methods=["GET", "HEAD"])
async def run_get_or_head(request: Request, token: str = Query("")):
    _auth_qs(token)
    if request.method == "HEAD":
        return Response(status_code=200)
    scanner = _safe_import("goalsniper.scanner")
    result = await scanner.run_scan_and_send()
    return {"status": "ok", **result}


# --- Telegram webhook (feedback) ----------------------------------------------
@app.post("/telegram/webhook")
@app.post("/telegram/webhook/{token}")
async def telegram_webhook(request: Request, token: str | None = None):
    if _telegram_webhook_token() and token != _telegram_webhook_token():
        raise HTTPException(status_code=403, detail="Forbidden")

    payload = await request.json()
    cq = payload.get("callback_query")
    if not cq:
        return {"ok": True}

    data = (cq.get("data") or "").strip()
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


# --- Daily digest (GET/HEAD) --------------------------------------------------
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

    # Parse day
    try:
        day = (
            datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            if date else datetime.now(timezone.utc)
        )
    except Exception:
        raise HTTPException(status_code=400, detail="Bad date format, use YYYY-MM-DD")

    stats = await storage.daily_counts_for(day)
    tot = await storage.totals()

    sent = stats["sent"]
    good = stats["good"]
    bad = stats["bad"]
    pending = stats["pending"]
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
