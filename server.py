import os
import sqlite3
import requests
import time
import threading
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()

PRESSURE_THRESHOLD = 5
CONFIDENCE_THRESHOLD = 70 
DB_PATH = os.getenv("DB_PATH", "tips.db")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
API_FOOTBALL_KEY = os.getenv("API_KEY")

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    raise RuntimeError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set in .env")

if not API_FOOTBALL_KEY:
    raise RuntimeError("API_KEY must be set in .env")

API_BASE = "https://v3.football.api-sports.io"

logger = logging.getLogger("tips_logger")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

def safe_int(val):
    try:
        v = int(str(val).replace('%', '').strip())
        return max(v, 0)
    except:
        return 0

def safe_float(val):
    try:
        return float(val)
    except:
        return 0.0

def high_pressure_tip(home, away, shots_home, shots_away, corners_home, corners_away, poss_home, poss_away):
    if shots_home >= PRESSURE_THRESHOLD or corners_home >= PRESSURE_THRESHOLD or poss_home >= 60:
        return ("HIGH_PRESSURE", f"📈 High attacking pressure by <b>{home}</b> → Next Goal")
    if shots_away >= PRESSURE_THRESHOLD or corners_away >= PRESSURE_THRESHOLD or poss_away >= 60:
        return ("HIGH_PRESSURE", f"📈 High attacking pressure by <b>{away}</b> → Next Goal")
    return None

def over_goals_tip(xg_home, xg_away, score_home, score_away):
    if (xg_home + xg_away >= 2.5) and (score_home + score_away <= 2):
        return ("OVER_GOALS", "🔥 High xG → Over 2.5 Goals looks promising")
    return None

def btts_tip(xg_home, xg_away, shots_home, shots_away):
    if (xg_home >= 1 and xg_away >= 1) and (shots_home >= 3 and shots_away >= 3):
        return ("BTTS", "⚡ Both teams creating chances → BTTS looks strong")
    return None

def comeback_tip(home, away, score_home, score_away, xg_home, xg_away, shots_home, shots_away):
    losing_team = home if score_home < score_away else (away if score_away < score_home else None)
    if losing_team:
        losing_xg = xg_home if losing_team == home else xg_away
        losing_shots = shots_home if losing_team == home else shots_away
        if losing_xg >= 1.2 and losing_shots >= 4:
            return ("COMEBACK", f"💥 {losing_team} creating chances despite trailing → Possible comeback goal")
    return None

def calculate_confidence(s_home: dict, s_away: dict) -> int:
    shots_home = safe_int(s_home.get("Shots on Target", 0))
    shots_away = safe_int(s_away.get("Shots on Target", 0))
    corners_home = safe_int(s_home.get("Corner Kicks", 0))
    corners_away = safe_int(s_away.get("Corner Kicks", 0))
    xg_home = safe_float(s_home.get("Expected Goals", 0))
    xg_away = safe_float(s_away.get("Expected Goals", 0))
    score = (shots_home + shots_away) * 5 + (corners_home + corners_away) * 3 + (xg_home + xg_away) * 10
    return min(int(score), 100)

def send_telegram_message(text: str) -> Optional[int]:
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}
        r = requests.post(url, json=payload)
        if r.status_code == 200:
            return r.json().get("result", {}).get("message_id")
    except Exception as e:
        logger.error(f"Telegram send error: {e}")
    return None

def edit_telegram_message(message_id: int, text: str) -> bool:
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/editMessageText"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "message_id": message_id, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}
        r = requests.post(url, json=payload)
        return r.status_code == 200
    except Exception as e:
        logger.error(f"Telegram edit error: {e}")
        return False

def generate_tip(match: Dict[str, Any]) -> List[Tuple[str, str]]:
    home = match["teams"]["home"]["name"]
    away = match["teams"]["away"]["name"]

    s_home = match["statistics"][home]
    s_away = match["statistics"][away]

    confidence = calculate_confidence(s_home, s_away)
    if confidence < CONFIDENCE_THRESHOLD:
        return []

    # ✅ Fixed away stats extraction
    xg_home = safe_float(s_home.get("Expected Goals", 0))
    xg_away = safe_float(s_away.get("Expected Goals", 0))
    shots_home = safe_int(s_home.get("Shots on Target", 0))
    shots_away = safe_int(s_away.get("Shots on Target", 0))
    corners_home = safe_int(s_home.get("Corner Kicks", 0))
    corners_away = safe_int(s_away.get("Corner Kicks", 0))
    poss_home = safe_int(s_home.get("Ball Possession", 0))
    poss_away = safe_int(s_away.get("Ball Possession", 0))

    tips = [
        high_pressure_tip(home, away, shots_home, shots_away, corners_home, corners_away, poss_home, poss_away),
        over_goals_tip(xg_home, xg_away, match["goals"]["home"], match["goals"]["away"]),
        btts_tip(xg_home, xg_away, shots_home, shots_away),
        comeback_tip(home, away, match["goals"]["home"], match["goals"]["away"], xg_home, xg_away, shots_home, shots_away)
    ]

    tips = [t for t in tips if t]
    if not tips:
        tips.append(("BALANCED", "📌 Stats suggest a balanced game."))

    return tips

def get_live_matches() -> List[Dict[str, Any]]:
    headers = {"x-rapidapi-key": API_KEY, "x-rapidapi-host": "v3.football.api-sports.io"}
    r = requests.get(f"{API_BASE}/fixtures", headers=headers, params={"live": "all"})
    if r.status_code != 200:
        return []
    live_data = r.json().get("response", [])
    matches = []
    for m in live_data:
        fixture_id = m["fixture"]["id"]
        s = requests.get(f"{API_BASE}/fixtures/statistics", headers=headers, params={"fixture": fixture_id})
        if s.status_code != 200:
            continue
        stats_data = s.json().get("response", [])
        stat_dict = {team_stats["team"]["name"]: {item["type"]: item["value"] for item in team_stats["statistics"]} for team_stats in stats_data}
        matches.append({"fixture": m["fixture"], "teams": m["teams"], "goals": m["goals"], "statistics": stat_dict})
    return matches

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tip_stats (
                match_id INTEGER,
                tip_type TEXT,
                tip TEXT,
                score TEXT,
                correct BOOLEAN,
                telegram_msg_id INTEGER,
                last_message TEXT,
                PRIMARY KEY (match_id, tip_type)
            );
        """)

def get_db_tip(match_id: int, tip_type: str):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("SELECT telegram_msg_id, last_message FROM tip_stats WHERE match_id=? AND tip_type=?", (match_id, tip_type))
        return cur.fetchone()

def save_or_update_tip(match_id: int, tip_type: str, tip: str, score: str, msg_id: Optional[int]):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT OR REPLACE INTO tip_stats (match_id, tip_type, tip, score, correct, telegram_msg_id, last_message)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (match_id, tip_type, tip, score, None, msg_id, tip))
        conn.commit()

def close_tip(match_id: int, tip_type: str, msg_id: int, last_message: str):
    closed_text = f"{last_message}\n\n✅ Match Finished"
    if edit_telegram_message(msg_id, closed_text):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                UPDATE tip_stats
                SET last_message = ?, correct = 1
                WHERE match_id = ? AND tip_type = ?
            """, (closed_text, match_id, tip_type))
            conn.commit()
        logger.info(f"[CLOSED] Match {match_id} tip {tip_type} marked as finished.")

def run_live_tips():
    init_db()
    logger.info("[SYSTEM] Starting live match tips loop...")

    msg_template = (
        "⚽️ {home} vs {away}\n"
        "⏱️ {elapsed}'\n"
        "🔢 Score: {score_home} - {score_away}\n"
        "📊 Confidence: {confidence}%\n\n"
        "🔍 Tip: {tip_type}\n"
        "{tip_text}"
    )

    while True:
        try:
            matches = get_live_matches()
            if matches:
                for match in matches:
                    match_id = match["fixture"]["id"]
                    status = match["fixture"]["status"]["short"]
                    home = match["teams"]["home"]["name"]
                    away = match["teams"]["away"]["name"]
                    elapsed = match["fixture"]["status"]["elapsed"]
                    s_home = match["statistics"][home]
                    s_away = match["statistics"][away]
                    confidence = calculate_confidence(s_home, s_away)

                    if status == "FT":
                        for tip_type in ["HIGH_PRESSURE", "OVER_GOALS", "BTTS", "COMEBACK", "BALANCED"]:
                            existing = get_db_tip(match_id, tip_type)
                            if existing:
                                msg_id, last_msg = existing
                                if last_msg and not last_msg.endswith("✅ Match Finished"):
                                    close_tip(match_id, tip_type, msg_id, last_msg)
                        continue

                    tips = generate_tip(match)
                    for tip_type, tip_text in tips:
                        score_str = f"{match['goals']['home']} - {match['goals']['away']}"
                        existing = get_db_tip(match_id, tip_type)
                        tip_msg = msg_template.format(
                            home=home,
                            away=away,
                            elapsed=elapsed,
                            score_home=match['goals']['home'],
                            score_away=match['goals']['away'],
                            confidence=confidence,
                            tip_type=tip_type,
                            tip_text=tip_text
                        )

                        if not existing:
                            msg_id = send_telegram_message(tip_msg)
                            if msg_id:
                                save_or_update_tip(match_id, tip_type, tip_text, score_str, msg_id)
                                logger.info(f"[NEW] Sent tip {tip_type} for {home} vs {away}")
                        else:
                            msg_id, last_msg = existing
                            if tip_text != last_msg:
                                if edit_telegram_message(msg_id, tip_msg):
                                    save_or_update_tip(match_id, tip_type, tip_text, score_str, msg_id)
                                    logger.info(f"[UPDATE] Edited tip {tip_type} for {home} vs {away}")
            else:
                logger.info("[SYSTEM] No live matches.")
        except Exception as e:
            logger.error(f"[ERROR] Loop failed: {e}")
        time.sleep(90)

app = FastAPI()

@app.on_event("startup")
def startup_event():
    threading.Thread(target=run_live_tips, daemon=True).start()

@app.get("/")
def root():
    return {"status": "ok", "message": "Live Match Tips API running"}

@app.get("/tips")
def get_tips():
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("SELECT * FROM tip_stats").fetchall()
        return {"tips": rows}

@app.post("/send-test")
def send_test_message():
    msg_id = send_telegram_message("Test message from Live Match Tips server ✅")
    if msg_id:
        return {"status": "sent", "message_id": msg_id}
    return JSONResponse({"status": "error"}, status_code=500)
