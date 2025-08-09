import os
from fastapi import FastAPI, Request, HTTPException
from .scanner import run_scan_and_send
from .config import RUN_TOKEN, TELEGRAM_WEBHOOK_TOKEN
from .storage import set_outcome
from .logger import log

app = FastAPI(title="Goalsniper", version="1.3.0")

@app.get("/health")
async def health():
    return {"ok": True, "name": "Goalsniper", "time": os.getenv("TZ", "UTC")}

def _auth(request: Request):
    auth = request.headers.get("authorization") or ""
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    token = auth[len("Bearer "):].strip()
    if token != RUN_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")

@app.post("/run")
async def run(request: Request):
    _auth(request)
    return {"status": "ok", **(await run_scan_and_send())}

# Telegram webhook: set to https://<your-app>/telegram/webhook[/<token>]
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
