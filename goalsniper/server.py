from fastapi import FastAPI, Request, HTTPException, Query, Response
import os
from .config import RUN_TOKEN, TELEGRAM_WEBHOOK_TOKEN
from .storage import set_outcome
from .logger import log

app = FastAPI(title="Goalsniper", version="1.3.3")


@app.get("/health")
async def health():
    return {"ok": True, "name": "Goalsniper", "time": os.getenv("TZ", "UTC")}


def _load_scanner():
    # Lazy import so startup never crashes; we return exceptions for debugging
    try:
        from . import scanner
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


@app.post("/run")
async def run_post(request: Request):
    _auth_header(request)
    scanner, err = _load_scanner()
    if err:
        # Show the real import error
        raise HTTPException(status_code=500, detail=f"scanner import failed: {err}")
    return {"status": "ok", **(await scanner.run_scan_and_send())}


# UPDATED: accept both GET and HEAD so UptimeRobot free (HEAD) can trigger scans.
@app.api_route("/run", methods=["GET", "HEAD"])
async def run_get_or_head(request: Request, token: str = Query("")):
    if token != RUN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    scanner, err = _load_scanner()
    if err:
        raise HTTPException(status_code=500, detail=f"scanner import failed: {err}")

    # Trigger the scan for both GET and HEAD
    result = await scanner.run_scan_and_send()

    # HEAD must not include a body; return 200 if scan executed
    if request.method == "HEAD":
        return Response(status_code=200)

    return {"status": "ok", **result}


@app.post("/telegram/webhook")
@app.post("/telegram/webhook/{token}")
async def telegram_webhook(request: Request, token: str | None = None):
    if TELEGRAM_WEBHOOK_TOKEN and token != TELEGRAM_WEBHOOK_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")

    payload = await request.json()
    cq = payload.get("callback_query")
    if not cq:
        return {"ok": True}

    data = cq.get("data") or ""
    if not data.startswith("fb:"):
        return {"ok": True}

    try:
        _, tip_id_s, outcome_s = data.split(":")
        tip_id = int(tip_id_s)
        outcome = 1 if outcome_s == "1" else 0
        await set_outcome(tip_id, outcome)
        log(f"Feedback received tip_id={tip_id} outcome={outcome}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"bad data: {str(e)}")

    return {"ok": True}
