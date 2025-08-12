# goalsniper/server.py
from __future__ import annotations

import os
import importlib
from datetime import datetime, timezone
from typing import Optional

import httpx
from fastapi import FastAPI, Request, HTTPException, Query, Response

app = FastAPI(title="Goalsniper", version="1.4.6")


# -------------------------
# env helpers (no defaults for secrets in code)
# -------------------------
def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def _telegram_webhook_token() -> str:
    return _env("TELEGRAM_WEBHOOK_TOKEN", "")


def _safe_import(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"import error: {module_name}: {e}")


# -------------------------
# unified auth (header OR query)
# -------------------------
def _get_bearer_from_header(request: Request) -> str:
    auth = (request.headers.get("authorization") or "").strip()
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return ""


def _auth_any(request: Request, qs_token: Optional[str] = None):
    provided = (qs_token or "").strip() or _get_bearer_from_header(request)
    expected = (_env("RUN_TOKEN", "") or "").strip()

    if not expected:
        raise HTTPException(status_code=500, detail="RUN_TOKEN not set")
    if not provided or provided != expected:
        # Use 401 to avoid hinting anything about token validity
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


# -----------------------------------------
# Run scan (POST header auth, GET/HEAD header or query auth)
# -----------------------------------------
@app.post("/run")
async def run_post(request: Request):
    _auth_any(request)  # header Bearer token
    scanner = _safe_import("goalsniper.scanner")
    result = await scanner.run_scan_and_send()
    return {"status": "ok", **result}


@app.api_route("/run", methods=["GET", "HEAD"])
async def run_get_or_head(request: Request, token: str = Query("")):
    _auth_any(request, token)  # query ?token=... OR header Bearer ...
    if request.method == "HEAD":
        return Response(status_code=200)
    scanner = _safe_import("goalsniper.scanner")
    result = await scanner.run_scan_and_send()
    return {"status": "ok", **result}


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
        # ignore non-callback updates
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

        # Optional learning telemetry (never breaks webhook)
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
# Daily digest (GET/HEAD)
# -------------------------
@app.api_route("/digest", methods=["GET", "HEAD"])
async def digest_get(
    request: Request,
    token: str = Query(""),
    date: str | None = Query(None, description="YYYY-MM-DD in UTC (optional)"),
    push: int = Query(1, description="1=send to Telegram, 0=JSON only"),
):
    _auth_any(request, token)

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
