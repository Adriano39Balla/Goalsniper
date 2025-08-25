import os
import re
import json
import time
import math
import heapq
import logging
import subprocess
from typing import List, Dict, Any, Optional, Tuple

import requests
import psycopg2
from flask import Flask, jsonify, request, abort
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from zoneinfo import ZoneInfo

# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
app = Flask(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Env
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")
API_KEY            = os.getenv("API_KEY")
ADMIN_API_KEY      = os.getenv("ADMIN_API_KEY")

HARVEST_MODE       = os.getenv("HARVEST_MODE", "0") not in ("0","false","False","no","NO")
MAX_TIPS_PER_SCAN  = int(os.getenv("MAX_TIPS_PER_SCAN", "25"))
DUP_COOLDOWN_MIN   = int(os.getenv("DUP_COOLDOWN_MIN", "20"))

# Model-only mode knobs
ONLY_MODEL_MODE    = True
MIN_PROB           = max(0.0, min(0.999, float(os.getenv("MIN_PROB", "0.90"))))
TOP_K_PER_MATCH    = int(os.getenv("TOP_K_PER_MATCH", "2"))

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise SystemExit("DATABASE_URL is required (Postgres).")

BASE_URL         = "https://v3.football.api-sports.io"
FOOTBALL_API_URL = f"{BASE_URL}/fixtures"
HEADERS          = {"x-apisports-key": API_KEY, "Accept": "application/json"}
INPLAY_STATUSES  = {"1H","HT","2H","ET","BT","P"}

session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], respect_retry_after_header=True)
session.mount("https://", HTTPAdapter(max_retries=retries))
session.mount("http://", HTTPAdapter(max_retries=retries))

# ──────────────────────────────────────────────────────────────────────────────
# DB helpers
class PgCursor:
    def __init__(self, cur): self.cur = cur
    def fetchone(self): return self.cur.fetchone()
    def fetchall(self): return self.cur.fetchall()

class PgConn:
    def __init__(self, dsn: str): self.dsn = dsn; self.conn=None; self.cur=None
    def __enter__(self):
        dsn = self.dsn
        if "sslmode=" not in dsn:
            dsn = dsn + ("&" if "?" in dsn else "?") + "sslmode=require"
        self.conn = psycopg2.connect(dsn)
        self.conn.autocommit = True
        self.cur = self.conn.cursor()
        return self
    def __exit__(self, exc_type, exc, tb):
        try:
            if self.cur: self.cur.close()
        finally:
            if self.conn: self.conn.close()
    def execute(self, sql: str, params: tuple|list=()):
        self.cur.execute(sql, params or ())
        return PgCursor(self.cur)

def db_conn() -> PgConn: return PgConn(DATABASE_URL)

def set_setting(key: str, value: str):
    with db_conn() as conn:
        conn.execute("""INSERT INTO settings(key,value) VALUES(%s,%s)
                        ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value""", (key, value))

def get_setting(key: str, default: Optional[str] = None) -> Optional[str]:
    with db_conn() as conn:
        row = conn.execute("SELECT value FROM settings WHERE key=%s", (key,)).fetchone()
        return row[0] if row else default

def init_db():
    with db_conn() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS tips (
            match_id   BIGINT,
            league_id  BIGINT,
            league     TEXT,
            home       TEXT,
            away       TEXT,
            market     TEXT,
            suggestion TEXT,
            confidence DOUBLE PRECISION,
            score_at_tip TEXT,
            minute     INTEGER,
            created_ts BIGINT,
            sent_ok    INTEGER DEFAULT 1,
            PRIMARY KEY (match_id, created_ts)
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS tip_snapshots (match_id BIGINT, created_ts BIGINT, payload TEXT, PRIMARY KEY (match_id, created_ts))""")
        conn.execute("""CREATE TABLE IF NOT EXISTS feedback (id SERIAL PRIMARY KEY, match_id BIGINT UNIQUE, verdict INTEGER, created_ts BIGINT)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS match_results (match_id BIGINT PRIMARY KEY, final_goals_h INTEGER, final_goals_a INTEGER, btts_yes INTEGER, updated_ts BIGINT)""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tip_snaps_created ON tip_snapshots(created_ts)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tips_match ON tips(match_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tips_recent ON tips(created_ts)")
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_tips_dedupe
            ON tips(match_id, market, suggestion, created_ts)
        """)

# ──────────────────────────────────────────────────────────────────────────────
# Dynamic thresholds
def get_dynamic_threshold(code: str, default: float = MIN_PROB) -> float:
    raw = get_setting("policy:thresholds_v1", "{}")
    try:
        thresholds = json.loads(raw)
        return float(thresholds.get(code, default))
    except Exception:
        return default

# ──────────────────────────────────────────────────────────────────────────────
# Feature extraction + model scoring + Telegram helpers
# (keep your existing extract_features, _mk_suggestions_for_code, etc. from before)

# ──────────────────────────────────────────────────────────────────────────────
# Smarter tip selection
def _select_best_suggestions(raw: List[Tuple[str, str, float, str]], top_k: int) -> List[Tuple[str,str,float,str]]:
    selected = []
    for market, suggestion, p, head in raw:
        thr = get_dynamic_threshold(head, MIN_PROB)
        if p >= thr:
            heapq.heappush(selected, (-p, (market, suggestion, p, head)))
    out = []
    while selected and len(out) < top_k:
        _, s = heapq.heappop(selected)
        out.append(s)
    return out

# ──────────────────────────────────────────────────────────────────────────────
# Production scan
def production_scan() -> Tuple[int,int]:
    matches = fetch_live_matches()
    live_seen = len(matches)
    if live_seen == 0:
        logging.info("[PROD] no live matches"); return 0, 0

    models = _list_models("model_v2:")
    if not models:
        logging.warning("[PROD] no models found in settings (model_v2:*)")
        return 0, live_seen

    saved = 0
    now_ts = int(time.time())

    with db_conn() as conn:
        for m in matches:
            try:
                fid = int((m.get("fixture", {}) or {}).get("id") or 0)
                if not fid: continue
                feat = extract_features(m)

                raw: List[Tuple[str,str,float,str]] = []
                for code, mdl in models.items():
                    raw.extend(_mk_suggestions_for_code(code, mdl, feat))

                filtered = _select_best_suggestions(raw, TOP_K_PER_MATCH)
                if not filtered: continue

                league_id, league = _league_name(m)
                home, away = _teams(m)
                score_txt = _pretty_score(m)

                for market, suggestion, prob, head in filtered:
                    cutoff = now_ts - (DUP_COOLDOWN_MIN * 60)
                    dup = conn.execute(
                        "SELECT 1 FROM tips WHERE match_id=%s AND market=%s AND suggestion=%s AND created_ts>=%s LIMIT 1",
                        (fid, market, suggestion, cutoff)
                    ).fetchone()
                    if dup: continue

                    now = int(time.time())
                    conn.execute("""
                        INSERT INTO tips(match_id,league_id,league,home,away,market,suggestion,confidence,score_at_tip,minute,created_ts,sent_ok)
                        VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,1)
                    """, (fid, league_id, league, home, away, market, suggestion, float(min(99.9, prob*100.0)), score_txt, int(feat.get("minute",0)), now))

                    msg = _format_tip_message(
                        league=league, home=home, away=away,
                        minute=int(feat.get("minute",0)), score_txt=score_txt,
                        suggestion=suggestion, prob=prob,
                        rationale=_rationale_from_feat(feat)
                    )
                    send_telegram(msg)
                    saved += 1
                    if MAX_TIPS_PER_SCAN and saved >= MAX_TIPS_PER_SCAN: break

                if MAX_TIPS_PER_SCAN and saved >= MAX_TIPS_PER_SCAN: break
            except Exception as e:
                logging.exception(f"[PROD] failure on match: {e}")
                continue

    logging.info(f"[PROD] saved={saved} live_seen={live_seen}")
    return saved, live_seen

# ──────────────────────────────────────────────────────────────────────────────
# Nightly digest + Match of the Day
def _counts_since(ts_from: int) -> Dict[str, int]:
    with db_conn() as conn:
        snap_total = conn.execute("SELECT COUNT(*) FROM tip_snapshots").fetchone()[0]
        tips_total = conn.execute("SELECT COUNT(*) FROM tips").fetchone()[0]
        res_total  = conn.execute("SELECT COUNT(*) FROM match_results").fetchone()[0]
        unlabeled  = conn.execute("""
            SELECT COUNT(DISTINCT s.match_id)
            FROM tip_snapshots s
            LEFT JOIN match_results r ON r.match_id = s.match_id
            WHERE r.match_id IS NULL
        """).fetchone()[0]
        snap_24h = conn.execute("SELECT COUNT(*) FROM tip_snapshots WHERE created_ts>=%s", (ts_from,)).fetchone()[0]
        res_24h  = conn.execute("SELECT COUNT(*) FROM match_results WHERE updated_ts>=%s", (ts_from,)).fetchone()[0]
    return {"snap_total": int(snap_total), "tips_total": int(tips_total), "res_total": int(res_total),
            "unlabeled": int(unlabeled), "snap_24h": int(snap_24h), "res_24h": int(res_24h)}

def nightly_digest_job():
    try:
        now = int(time.time()); day_ago = now - 24*3600
        c = _counts_since(day_ago)
        msg = (
            "📊 <b>Robi Nightly Digest</b>\n"
            f"Snapshots: {c['snap_total']} (+ {c['snap_24h']} last 24h)\n"
            f"Finals: {c['res_total']} (+ {c['res_24h']} last 24h)\n"
            f"Unlabeled match_ids: {c['unlabeled']}\n"
            "Models retrain at 03:00 CEST daily."
        )
        logging.info("[DIGEST]\n%s", msg.replace("<b>","").replace("</b>",""))
        send_telegram(msg)
    except Exception as e:
        logging.exception("[DIGEST] failed: %s", e)

def match_of_the_day_job():
    try:
        now = int(time.time()); day_ago = now - 24*3600
        with db_conn() as conn:
            row = conn.execute(
                """
                SELECT league, home, away, market, suggestion, confidence, score_at_tip, minute, created_ts
                FROM tips
                WHERE created_ts >= %s AND sent_ok=1
                ORDER BY confidence DESC, created_ts DESC
                LIMIT 1
                """, (day_ago,)
            ).fetchone()
        if not row:
            send_telegram("🏅 Match of the Day: (no tips in the last 24h)")
            return
        league, home, away, market, suggestion, conf, score_at_tip, minute, ts = row
        when = time.strftime("%Y-%m-%d %H:%M", time.gmtime(ts)) + " UTC"
        msg = (
            "🏅 <b>Match of the Day</b>\n"
            f"{home} vs {away}\n"
            f"⏰ Minute {minute}' | Score then: {score_at_tip} | {when}\n"
            f"Market: {market}\n"
            f"Tip: {suggestion}\n"
            f"📈 Model {conf:.1f}%\n"
            f"🏆 League: {league}"
        )
        send_telegram(msg)
    except Exception as e:
        logging.exception("[MOTD] failed: %s", e)

# ──────────────────────────────────────────────────────────────────────────────
# Retrain + Analyze
def retrain_and_analyze_job():
    try:
        logging.info("[RETRAIN] Starting nightly retrain & analyze job...")
        result_train = subprocess.run(["python", "train_models.py"], capture_output=True, text=True)
        logging.info("[RETRAIN] train_models.py output:\n%s", result_train.stdout)

        result_analyze = subprocess.run(["python", "analyze_precision.py"], capture_output=True, text=True)
        logging.info("[RETRAIN] analyze_precision.py output:\n%s", result_analyze.stdout)

        thresholds = get_setting("policy:thresholds_v1", "{}")
        msg = (
            "🤖 <b>Nightly Retrain Completed</b>\n"
            f"Models & thresholds updated at {time.strftime('%Y-%m-%d %H:%M', time.gmtime())} UTC\n"
            f"Dynamic thresholds: <code>{thresholds}</code>"
        )
        send_telegram(msg)
    except Exception as e:
        logging.exception("[RETRAIN] failed: %s", e)

# ──────────────────────────────────────────────────────────────────────────────
# Entrypoint
if __name__ == "__main__":
    if not API_KEY: logging.error("API_KEY is not set — live fetch will return 0 matches.")
    if not ADMIN_API_KEY: logging.error("ADMIN_API_KEY is not set — admin endpoints will 401.")
    init_db()

    scheduler = BackgroundScheduler()

    # production scan
    scheduler.add_job(production_scan, CronTrigger(minute="*/5", timezone=ZoneInfo("Europe/Berlin")),
                      id="production_scan", replace_existing=True)

    # nightly digest
    scheduler.add_job(nightly_digest_job, CronTrigger(hour=3, minute=2, timezone=ZoneInfo("Europe/Berlin")),
                      id="digest", replace_existing=True, misfire_grace_time=3600, coalesce=True)

    # Match of the Day
    scheduler.add_job(match_of_the_day_job, CronTrigger(hour=9, minute=5, timezone=ZoneInfo("Europe/Berlin")),
                      id="motd", replace_existing=True, misfire_grace_time=3600, coalesce=True)

    # retrain loop
    scheduler.add_job(retrain_and_analyze_job, CronTrigger(hour=3, minute=0, timezone=ZoneInfo("Europe/Berlin")),
                      id="retrain_loop", replace_existing=True, misfire_grace_time=3600, coalesce=True)

    scheduler.start()
    logging.info("⏱️ Scheduler started (model-only, MIN_PROB=%.2f, TOP_K=%d)", MIN_PROB, TOP_K_PER_MATCH)

    port = int(os.getenv("PORT", 5000))
    logging.info("✅ Robi Superbrain started.")
    app.run(host="0.0.0.0", port=port)
