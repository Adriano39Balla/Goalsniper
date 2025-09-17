import os, asyncio, json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import psycopg
from psycopg.rows import dict_row

TZ = ZoneInfo(os.getenv("TZ", "Europe/Berlin"))
DB_URL = os.getenv("DATABASE_URL", "")
TARGET_PRECISION = float(os.getenv("TARGET_PRECISION", "0.78"))  # aim 75–80%
COOLDOWN_START = os.getenv("COOLDOWN_START","23:00")
COOLDOWN_END   = os.getenv("COOLDOWN_END","07:00")

def within_cooldown():
    h1, m1 = map(int, COOLDOWN_START.split(":"))
    h2, m2 = map(int, COOLDOWN_END.split(":"))
    s = datetime.now(TZ).replace(hour=h1, minute=m1, second=0, microsecond=0).time()
    e = datetime.now(TZ).replace(hour=h2, minute=m2, second=0, microsecond=0).time()
    n = datetime.now(TZ).time()
    return n >= s or n < e

async def tune_thresholds():
    if not within_cooldown(): 
        print("Not in cooldown; skip."); return

    end = datetime.now(TZ).date()
    start = end - timedelta(days=14)
    async with await psycopg.AsyncConnection.connect(DB_URL, row_factory=dict_row) as con:
        rows = await con.execute(
            """
            SELECT p.*, r.goals_home, r.goals_away
            FROM predictions p
            JOIN results r ON r.fixture_id = p.fixture_id
            WHERE (p.sent_at AT TIME ZONE 'Europe/Berlin')::date BETWEEN %s::date AND %s::date
            """, (start.isoformat(), end.isoformat())
        )
        data = await rows.fetchall()

    if not data:
        print("No historical data to tune; keeping existing."); return

    best = None
    for th_i in range(65, 91):
        th = th_i / 100
        for ev_i in range(0, 11):
            evmin = ev_i / 100
            bets = wins = 0
            for d in data:
                over25 = (d["goals_home"] + d["goals_away"]) >= 3
                conf = float(d["confidence"]); ev = float(d["ev"])
                if conf >= th and ev >= evmin:
                    bets += 1
                    won = (d["pick"] == "over" and over25) or (d["pick"] == "under" and not over25)
                    wins += 1 if won else 0
            if bets == 0: continue
            precision = wins / bets
            score = (precision >= TARGET_PRECISION, bets, precision)
            if best is None or score > best[0]:
                best = (score, th, evmin, precision, bets)

    if best:
        _, th, evmin, prec, cov = best
        policy = {"theta": round(th, 2), "ev_min": round(evmin, 2), "target_precision": TARGET_PRECISION}
        async with await psycopg.AsyncConnection.connect(DB_URL) as con:
            await con.execute(
                "INSERT INTO model_cfg(key,value) VALUES ('policy', %s) "
                "ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value",
                (json.dumps(policy),)
            ); await con.commit()
        print(f"Tuned θ={th:.2f} EVmin={evmin:.2f} | precision={prec:.2f} coverage={cov}")
    else:
        print("No viable thresholds found; keeping previous.")

async def main():
    if not DB_URL: 
        raise SystemExit("Missing DATABASE_URL")
    await tune_thresholds()

if __name__ == "__main__":
    asyncio.run(main())
