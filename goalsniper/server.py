from __future__ import annotations

import os
from datetime import datetime, timezone
from fastapi import FastAPI, Request, HTTPException, Query, Response
import httpx

# only light/safe imports at module import time
from .config import RUN_TOKEN, TELEGRAM_WEBHOOK_TOKEN

app = FastAPI(title="Goalsniper", version="1.4.3")

# ---------- health ----------
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

# ---------- helpers ----------
def _load_scanner():
    try:
        # lazy import so server import never fails
        from . import scanner  # type: ignore
        return scanner, None
    except Exception as e:
        return None, e

def _auth_header(request: Request):
    auth = request.headers.get("authorization") or ""
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    token = auth[len("Bearer "):].strip()
    if token != RUN_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")

# ---------- run endpoints ----------
@app.post("/run")
async def run_post(request: Request):
    _auth_header(request)
    scanner, err = _load_scanner()
    if err:
        raise HTTPException(status_code=500, detail=f"scanner import failed: {err}")
    return {"status": "ok", **(await scanner.run_scan_and_send())}

@app.api_route("/run", methods=["GET", "HEAD"])
async def run_get_or_head(request: Request, token: str = Query("")):
    if token != RUN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if request.method == "HEAD":
        return Response(status_code=200)
    scanner, err = _load_scanner()
    if err:
        raise HTTPException(status_code=500, detail=f"scanner import failed: {err}")
    result = await scanner.run_scan_and_send()
    return {"status": "ok", **result}

# ---------- telegram webhook ----------
@app.post("/telegram/webhook")
@app.post("/telegram/webhook/{token}")
async def telegram_webhook(request: Request, token: str | None = None):
    if TELEGRAM_WEBHOOK_TOKEN and token != TELEGRAM_WEBHOOK_TOKEN:
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

        # lazy imports to avoid import-time failures
        from .storage import set_outcome
        from . import learning
        from .logger import log

        await set_outcome(tip_id, outcome)

        try:
            info = await learning.on_feedback_update(tip_id, outcome)
            log(
                f"[learn] tip={tip_id} market={info.get('market')} "
                f"p_cal={info.get('calibratedProb'):.3f} conf={info.get('confidence'):.3f} "
                f"thr={info.get('marketThreshold'):.3f} outcome={outcome}"
            )
        except Exception as le:
            log(f"[learn] update failed for tip_id={tip_id}: {le}")

        log(f"Feedback received tip_id={tip_id} outcome={outcome}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"bad data: {str(e)}")

    return {"ok": True}

# ---------- daily digest ----------
@app.api_route("/digest", methods=["GET", "HEAD"])
async def digest_get(
    request: Request,
    token: str = Query(""),
    date: str | None = Query(None, description="YYYY-MM-DD in UTC (optional)"),
    push: int = Query(1, description="1=send to Telegram, 0=JSON only"),
):
    if token != RUN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if request.method == "HEAD":
        return Response(status_code=200)

    from .storage import daily_counts_for, totals
    from . import telegram as tg

    try:
        day = (
            datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            if date else datetime.now(timezone.utc)
        )
    except Exception:
        raise HTTPException(status_code=400, detail="Bad date format, use YYYY-MM-DD")

    stats = await daily_counts_for(day)
    tot = await totals()

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
                await tg.send_text(client, text)
                result["pushed"] = True
            except Exception as e:
                result["error"] = f"telegram: {e}"

    return result

# explicit export helps some loaders
__all__ = ["app"]
