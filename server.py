import os
import logging
import pytz
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, status
from analyzer import analyze_matches
from telegram import send_tip_message
from db import store_tip, store_feedback
from model_training import auto_train_model

import requests

load_dotenv()
app = FastAPI()
logger = logging.getLogger("uvicorn")

API_KEY = os.getenv("API_KEY")
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
AUTH_HEADER = os.getenv("AUTH_HEADER", "secure-token-123")  # Default fallback
API_URL = "https://v3.football.api-sports.io/fixtures"
HEADERS = {"x-apisports-key": API_KEY}

def verify_auth(x_token: str = Header(...)):
    if x_token != AUTH_HEADER:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


@app.get("/")
def root():
    return {"status": "Robi_Superbrain v10 ready"}


@app.get("/match-alert")
def match_alert(x_token: str = Header(...)):
    verify_auth(x_token)
    try:
        now = datetime.utcnow().replace(tzinfo=pytz.UTC)
        res = requests.get(API_URL, headers=HEADERS, params={"live": "all"})
        data = res.json()
        matches = data.get("response", [])
        sent_count = 0

        for match in matches:
            tip = analyze_matches(match)
            if tip:
                was_stored = store_tip(tip)
                if was_stored:
                    send_tip_message(tip, BOT_TOKEN, CHAT_ID)
                    sent_count += 1

        return {"result": "ok", "tips_sent": sent_count}

    except Exception as e:
        logger.error(f"[MatchAlert] Error: {e}")
        return {"error": str(e)}


@app.post("/feedback")
def feedback_endpoint(payload: dict, x_token: str = Header(...)):
    verify_auth(x_token)
    try:
        match_id = int(payload.get("match_id"))
        result = str(payload.get("result"))
        if result not in ["✅", "❌"]:
            raise ValueError("Invalid result format")
        store_feedback(match_id, result)
        auto_train_model()
        return {"status": "Feedback received"}
    except Exception as e:
        logger.error(f"[Feedback] Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
