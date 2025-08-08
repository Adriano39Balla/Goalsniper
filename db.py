import sqlite3
from datetime import datetime
import os

DB_PATH = os.path.join(os.getcwd(), "tips.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS tips (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        match_id INTEGER UNIQUE,
        team TEXT,
        league TEXT,
        tip TEXT,
        confidence INTEGER,
        created_at TEXT,
        result TEXT
    )
    """)
    conn.commit()
    conn.close()

def store_tip(tip: dict) -> bool:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("""
            INSERT INTO tips (match_id, team, league, tip, confidence, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            tip["match_id"],
            tip["team"],
            tip["league"],
            tip["tip"],
            tip["confidence"],
            datetime.utcnow().isoformat()
        ))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        # Match already exists
        return False
    finally:
        conn.close()

def store_feedback(match_id: int, result: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        UPDATE tips SET result = ? WHERE match_id = ?
    """, (result, match_id))
    conn.commit()
    conn.close()

def get_training_data():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT team, league, tip, confidence, result FROM tips
        WHERE result IS NOT NULL
    """)
    rows = c.fetchall()
    conn.close()
    return rows

# Initialize DB on import
init_db()
