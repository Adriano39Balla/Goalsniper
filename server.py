import os
import requests
import logging
import pytz
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI
from src.analyzer import analyze_matches
from src.telegram import send_tip_message
from src.db import store_tip
from src.model_training import auto_train_model

load_dotenv()

app = FastAPI()
logger = logging.getLogger("uvicorn")

API_KEY = os.getenv("API_KEY")
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
API_URL = "https://v3.football.api-sports.io/fixtures"

HEADERS = {"x-apisports-key": API_KEY}

@app.get("/")
def root():
    return {"status": "Robi_Superbrain v10 ready"}

@app.get("/match-alert")
def match_alert():
    try:
        now = datetime.utcnow().replace(tzinfo=pytz.UTC)
        params = {"live": "all"}
        res = requests.get(API_URL, headers=HEADERS, params=params)
        data = res.json()

        matches = data.get("response", [])
        for match in matches:
            tip = analyze_matches(match)
            if tip:
                store_tip(tip)
                send_tip_message(tip, BOT_TOKEN, CHAT_ID)
        return {"result": "ok", "count": len(matches)}
    except Exception as e:
        logger.error(f"Error in /match-alert: {e}")
        return {"error": str(e)}

@app.post("/feedback")
def feedback_endpoint(match_id: int, result: str):
    from src.db import store_feedback
    store_feedback(match_id, result)
    auto_train_model()
    return {"status": "Feedback received"}
