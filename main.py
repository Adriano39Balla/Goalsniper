import os
import json
import time
import math
import logging
import requests
import sqlite3
import subprocess, shlex
from html import escape as html_escape
from zoneinfo import ZoneInfo
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from flask import Flask, jsonify, request, abort
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
app = Flask(__name__)

# ── Env ───────────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")
API_KEY            = os.getenv("API_KEY")  # API-Football key only
ADMIN_API_KEY      = os.getenv("ADMIN_API_KEY")  # separate admin key for our endpoints
PUBLIC_BASE_URL    = os.getenv("PUBLIC_BASE_URL")
BET_URL_TMPL       = os.getenv("BET_URL")
WATCH_URL_TMPL     = os.getenv("WATCH_URL")
WEBHOOK_SECRET     = os.getenv("TELEGRAM_WEBHOOK_SECRET")
HEARTBEAT_ENABLE   = os.getenv("HEARTBEAT_ENABLE", "1") not in ("0","false","False","no","NO")

# Mode switch
HARVEST_MODE       = os.getenv("HARVEST_MODE", "1") not in ("0","false","False","no","NO")

# Core knobs (used if you switch off harvest)
CONF_THRESHOLD     = int(os.getenv("CONF_THRESHOLD", "60"))
MAX_TIPS_PER_SCAN  = int(os.getenv("MAX_TIPS_PER_SCAN", "25"))
DUP_COOLDOWN_MIN   = int(os.getenv("DUP_COOLDOWN_MIN", "20"))

# Guards
O25_LATE_MINUTE     = int(os.getenv("O25_LATE_MINUTE", "88"))
O25_LATE_MIN_GOALS  = int(os.getenv("O25_LATE_MIN_GOALS", "2"))
BTTS_LATE_MINUTE    = int(os.getenv("BTTS_LATE_MINUTE", "88"))
UNDER_SUPPRESS_AFTER_MIN = int(os.getenv("UNDER_SUPPRESS_AFTER_MIN", "82"))

# Data sufficiency for harvesting/training
ONLY_MODEL_MODE          = os.getenv("ONLY_MODEL_MODE", "0") not in ("0","false","False","no","NO")
REQUIRE_STATS_MINUTE     = int(os.getenv("REQUIRE_STATS_MINUTE", "35"))
REQUIRE_DATA_FIELDS      = int(os.getenv("REQUIRE_DATA_FIELDS", "2"))  # need >=2 among {xG,SOT,CK,POS}

# MOTD (kept, but harmless in harvest)
LEAGUE_PRIORITY_IDS = [int(x) for x in (os.getenv("MOTD_LEAGUE_IDS", "39,140,135,78,61,2").split(",")) if x.strip().isdigit()]
MOTD_PREDICT        = os.getenv("MOTD_PREDICT", "1") not in ("0","false","False","no","NO")
MOTD_MIN_SAMPLES    = int(os.getenv("MOTD_MIN_SAMPLES", "30"))
MOTD_CONF_MIN       = int(os.getenv("MOTD_CONF_MIN", "65"))

ALLOWED_SUGGESTIONS = {"Over 2.5 Goals", "Under 2.5 Goals", "BTTS: Yes", "BTTS: No"}

# ── Training env (nightly) ────────────────────────────────────────────────────
TRAIN_ENABLE       = os.getenv("TRAIN_ENABLE", "1") not in ("0","false","False","no","NO")
TRAIN_MIN_MINUTE   = int(os.getenv("TRAIN_MIN_MINUTE", "15"))
TRAIN_TEST_SIZE    = float(os.getenv("TRAIN_TEST_SIZE", "0.25"))

# ── External APIs ─────────────────────────────────────────────────────────────
BASE_URL         = "https://v3.football.api-sports.io"
FOOTBALL_API_URL = f"{BASE_URL}/fixtures"
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
HEADERS          = {"x-apisports-key": API_KEY, "Accept": "application/json"}  # API-Football

# ── HTTP session ──────────────────────────────────────────────────────────────
session = requests.Session()
retries = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    respect_retry_after_header=True,
)
session.mount("https://", HTTPAdapter(max_retries=retries))

# ── In-memory caches ──────────────────────────────────────────────────────────
STATS_CACHE: Dict[int, Tuple[float, list]] = {}
EVENTS_CACHE: Dict[int, Tuple[float, list]] = {}
CAL_CACHE: Dict[str, Any] = {"ts": 0, "bins": []}

# ── DB ────────────────────────────────────────────────────────────────────────
DB_PATH = "tip_performance.db"

def db_conn():
    # Safer SQLite settings for concurrent scheduler + HTTP
    conn = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None)  # autocommit mode
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn

def init_db():
    with db_conn() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS tips (
            match_id INTEGER,
            league_id INTEGER,
            league   TEXT,
            home     TEXT,
            away     TEXT,
            market   TEXT,
            suggestion TEXT,
            confidence REAL,
            score_at_tip TEXT,
            minute    INTEGER,
            created_ts INTEGER,
            sent_ok   INTEGER DEFAULT 1,
            PRIMARY KEY (match_id, created_ts)
        )""")
        # backward-compat add
        try:
            conn.execute("ALTER TABLE tips ADD COLUMN sent_ok INTEGER DEFAULT 1")
        except Exception:
            pass

        conn.execute("""
        CREATE TABLE IF NOT EXISTS tip_snapshots (
            match_id INTEGER,
            created_ts INTEGER,
            payload TEXT,
            PRIMARY KEY (match_id, created_ts)
        )""")
        conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id INTEGER UNIQUE,
            verdict  INTEGER CHECK (verdict IN (0,1)),
            created_ts INTEGER
        )""")
        conn.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )""")
        conn.execute("""
        CREATE TABLE IF NOT EXISTS match_results (
            match_id   INTEGER PRIMARY KEY,
            final_goals_h INTEGER,
            final_goals_a INTEGER,
            btts_yes      INTEGER,
            updated_ts    INTEGER
        )""")

        # Indices & views
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tip_snaps_created ON tip_snapshots(created_ts)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tips_match ON tips(match_id)")
        conn.execute("DROP VIEW IF EXISTS v_tip_stats")
        conn.execute("""
        CREATE VIEW v_tip_stats AS
        SELECT t.market, t.suggestion,
               AVG(f.verdict) AS hit_rate,
               COUNT(DISTINCT t.match_id) AS n
        FROM (
          SELECT match_id, market, suggestion, MAX(created_ts) AS last_ts
          FROM tips GROUP BY match_id, market, suggestion
        ) lt
        JOIN tips t ON t.match_id=lt.match_id AND t.created_ts=lt.last_ts
        JOIN feedback f ON f.match_id = t.match_id
        GROUP BY t.market, t.suggestion
        """)
        conn.commit()

def set_setting(key: str, value: str):
    with db_conn() as conn:
        conn.execute(
            "INSERT INTO settings(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        conn.commit()

def get_setting(key: str, default: Optional[str] = None) -> Optional[str]:
    with db_conn() as conn:
        cur = conn.execute("SELECT value FROM settings WHERE key=?", (key,))
        row = cur.fetchone()
        return row[0] if row else default

# ── Telegram (unused in harvest, but kept) ────────────────────────────────────
def send_telegram(message: str, inline_keyboard: Optional[list] = None) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logging.error("Missing Telegram credentials")
        return False
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": html_escape(message), "parse_mode": "HTML"}
    if inline_keyboard:
        payload["reply_markup"] = json.dumps({"inline_keyboard": inline_keyboard})
    try:
        res = session.post(f"{TELEGRAM_API_URL}/sendMessage", data=payload, timeout=10)
        if not res.ok:
            logging.error(f"[Telegram] sendMessage FAILED status={res.status_code} body={res.text[:300]}")
        else:
            logging.info("[Telegram] sendMessage OK")
        return res.ok
    except Exception as e:
        logging.exception(f"[Telegram] Exception: {e}")
        return False

def answer_callback(callback_id: str, text: str):
    try:
        session.post(f"{TELEGRAM_API_URL}/answerCallbackQuery",
                     data={"callback_query_id": callback_id, "text": text, "show_alert": False}, timeout=10)
    except Exception:
        pass

# ── API helpers ───────────────────────────────────────────────────────────────
def _api_get(url: str, params: dict, timeout: int = 15):
    if not API_KEY:
        logging.error("[API] API_KEY not set; skipping request to %s", url)
        return None
    try:
        res = session.get(url, headers=HEADERS, params=params, timeout=timeout)
        if not res.ok:
            logging.error(f"[API] {url} status={res.status_code} body={res.text[:300]}")
            return None

        # Optional: rate limit telemetry
        h = {k.lower(): v for k, v in res.headers.items()}
        try:
            rem = int(h.get("x-ratelimit-remaining", "-1"))
            if 0 <= rem <= 2:
                reset = h.get("x-ratelimit-reset") or h.get("ratelimit-reset")
                logging.warning(f"[API] Low remaining quota: {rem} (reset={reset})")
        except Exception:
            pass

        return res.json()
    except Exception as e:
        logging.exception(f"[API] error {url}: {e}")
        return None

def fetch_match_stats(fixture_id: int) -> Optional[List[Dict[str, Any]]]:
    now = time.time()
    if fixture_id in STATS_CACHE:
        ts, data = STATS_CACHE[fixture_id]
        if now - ts < 60:
            return data
    js = _api_get(f"{FOOTBALL_API_URL}/statistics", {"fixture": fixture_id})
    stats = js.get("response", []) if isinstance(js, dict) else None
    STATS_CACHE[fixture_id] = (now, stats or [])
    return stats

def fetch_match_events(fixture_id: int) -> Optional[List[Dict[str, Any]]]:
    now = time.time()
    if fixture_id in EVENTS_CACHE:
        ts, data = EVENTS_CACHE[fixture_id]
        if now - ts < 60:
            return data
    js = _api_get(f"{FOOTBALL_API_URL}/events", {"fixture": fixture_id})
    evs = js.get("response", []) if isinstance(js, dict) else None
    EVENTS_CACHE[fixture_id] = (now, evs or [])
    return evs

def fetch_live_matches() -> List[Dict[str, Any]]:
    js = _api_get(FOOTBALL_API_URL, {"live": "all"})
    if not isinstance(js, dict): return []
    matches = js.get("response", []) or []
    out = []
    for m in matches:
        status = (m.get("fixture", {}) or {}).get("status", {}) or {}
        elapsed = status.get("elapsed")
        if elapsed is None or elapsed > 90:
            continue
        fid = (m.get("fixture", {}) or {}).get("id")
        try:
            m["statistics"] = fetch_match_stats(fid) or []
        except Exception:
            m["statistics"] = []
        try:
            m["events"] = fetch_match_events(fid) or []
        except Exception:
            m["events"] = []
        out.append(m)
    logging.info(f"[FETCH] live={len(matches)} kept={len(out)}")
    return out

def fetch_fixtures_by_ids(ids: List[int]) -> Dict[int, Dict[str, Any]]:
    if not ids: return {}
    js = _api_get(FOOTBALL_API_URL, {"ids": ",".join(str(i) for i in ids), "timezone": "UTC"})
    resp = js.get("response", []) if isinstance(js, dict) else []
    return { (fx.get("fixture") or {}).get("id"): fx for fx in resp }

# ── Features / snapshots ─────────────────────────────────────────────────────
def _num(v) -> float:
    try:
        if isinstance(v, str) and v.endswith('%'):
            return float(v[:-1])
        return float(v or 0)
    except Exception:
        return 0.0

def _pos_pct(v) -> float:
    try:
        return float(str(v).replace('%', '').strip() or 0)
    except Exception:
        return 0.0

def _get_stat(stats_dict: Dict[str, Any], *keys: str, default=0) -> Any:
    """Accept common variants and case-insensitive matches."""
    for k in keys:
        if k in stats_dict:
            return stats_dict[k]
    lower = {str(k).lower(): v for k, v in stats_dict.items()}
    for k in keys:
        lk = str(k).lower()
        if lk in lower:
            return lower[lk]
    return default

def extract_features(match: Dict[str, Any]) -> Dict[str, float]:
    home_name = match["teams"]["home"]["name"]
    away_name = match["teams"]["away"]["name"]
    gh = match["goals"]["home"] or 0
    ga = match["goals"]["away"] or 0
    minute = int((match.get("fixture", {}).get("status", {}) or {}).get("elapsed") or 0)
    stats_blocks = match.get("statistics") or []
    stats: Dict[str, Dict[str, Any]] = {}
    for s in stats_blocks:
        tname = (s.get("team") or {}).get("name")
        if tname:
            stats[tname] = {i["type"]: i["value"] for i in (s.get("statistics") or [])}
    sh = stats.get(home_name, {}); sa = stats.get(away_name, {})

    xg_h = _num(_get_stat(sh, "Expected Goals", "xG", "expected_goals"))
    xg_a = _num(_get_stat(sa, "Expected Goals", "xG", "expected_goals"))

    sot_h = _num(_get_stat(sh, "Shots on Target", "shots_on_target", "SOT"))
    sot_a = _num(_get_stat(sa, "Shots on Target", "shots_on_target", "SOT"))

    cor_h = _num(_get_stat(sh, "Corner Kicks", "Corners", "corner_kicks"))
    cor_a = _num(_get_stat(sa, "Corner Kicks", "Corners", "corner_kicks"))

    pos_h = _pos_pct(_get_stat(sh, "Ball Possession", "Possession", "ball_possession"))
    pos_a = _pos_pct(_get_stat(sa, "Ball Possession", "Possession", "ball_possession"))

    # count red cards from events
    red_h = 0; red_a = 0
    for ev in (match.get("events") or []):
        try:
            etype = (ev.get("type") or "").lower()
            edetail = (ev.get("detail") or "").lower()
            tname = (ev.get("team") or {}).get("name") or ""
            if etype == "card" and "red" in edetail:
                if tname == home_name: red_h += 1
                elif tname == away_name: red_a += 1
        except Exception:
            pass

    return {
        "minute": float(minute),
        "goals_h": float(gh), "goals_a": float(ga),
        "goals_sum": float(gh + ga), "goals_diff": float(gh - ga),
        "xg_h": float(xg_h), "xg_a": float(xg_a), "xg_sum": float(xg_h + xg_a),
        "sot_h": float(sot_h), "sot_a": float(sot_a), "sot_sum": float(sot_h + sot_a),
        "cor_h": float(cor_h), "cor_a": float(cor_a), "cor_sum": float(cor_h + cor_a),
        "pos_h": float(pos_h), "pos_a": float(pos_a),
        "red_h": float(red_h), "red_a": float(red_a),
    }

def stats_coverage_ok(feat: Dict[str, float], minute: int) -> bool:
    if minute < REQUIRE_STATS_MINUTE:
        return True
    fields = [
        feat.get("xg_sum", 0.0),
        feat.get("sot_sum", 0.0),
        feat.get("cor_sum", 0.0),
        max(feat.get("pos_h", 0.0), feat.get("pos_a", 0.0)),
    ]
    nonzero = sum(1 for v in fields if (v or 0) > 0)
    return nonzero >= max(0, REQUIRE_DATA_FIELDS)

def save_snapshot_from_match(m: Dict[str, Any], feat: Dict[str, float]) -> None:
    fx = m.get("fixture", {}) or {}
    lg = m.get("league", {}) or {}
    fid = int(fx.get("id"))
    league_id = int(lg.get("id") or 0)
    league = f"{lg.get('country','')} - {lg.get('name','')}".strip(" -")
    home = (m.get("teams") or {}).get("home", {}).get("name", "")
    away = (m.get("teams") or {}).get("away", {}).get("name", "")
    gh = (m.get("goals") or {}).get("home") or 0
    ga = (m.get("goals") or {}).get("away") or 0
    minute = int(feat.get("minute", 0))

    snapshot = {
        "minute": minute, "gh": gh, "ga": ga,
        "league_id": league_id, "market": "HARVEST", "suggestion": "HARVEST",
        "confidence": 0,
        "stat": {
            "xg_h": feat.get("xg_h",0), "xg_a": feat.get("xg_a",0),
            "sot_h": feat.get("sot_h",0), "sot_a": feat.get("sot_a",0),
            "cor_h": feat.get("cor_h",0), "cor_a": feat.get("cor_a",0),
            "pos_h": feat.get("pos_h",0), "pos_a": feat.get("pos_a",0),
            "red_h": feat.get("red_h",0), "red_a": feat.get("red_a",0),
        }
    }

    now = int(time.time())
    with db_conn() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO tip_snapshots(match_id, created_ts, payload)
            VALUES (?,?,?)
        """, (fid, now, json.dumps(snapshot)[:200000]))
        # also record a tips row for bookkeeping (sent_ok=0)
        conn.execute("""
            INSERT INTO tips(match_id,league_id,league,home,away,market,suggestion,confidence,score_at_tip,minute,created_ts,sent_ok)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, (fid, league_id, league, home, away, "HARVEST", "HARVEST", 0.0, f"{gh}-{ga}", minute, now, 0))
        conn.commit()

# ── Harvest scan (no sends) ───────────────────────────────────────────────────
def harvest_scan() -> Tuple[int, int]:
    matches = fetch_live_matches()
    saved = 0
    for m in matches:
        feat = extract_features(m)
        if not stats_coverage_ok(feat, int(feat.get("minute", 0))):
            continue
        save_snapshot_from_match(m, feat)
        saved += 1
    set_setting("last_harvest_ts", str(int(time.time())))
    set_setting("last_harvest_saved", str(saved))
    set_setting("last_harvest_live_seen", str(len(matches)))
    logging.info(f"[HARVEST] snapshots_saved={saved} live={len(matches)}")
    return saved, len(matches)

# ── Backfill final results for harvested match_ids ────────────────────────────
def backfill_results_from_snapshots(hours: int = 48) -> Tuple[int, int]:
    """
    Returns: (updated, checked)
    """
    hours = max(1, min(int(hours), 168))  # clamp to 1..168h
    since = int(time.time()) - int(hours) * 3600
    with db_conn() as conn:
        cur = conn.execute("""
            SELECT DISTINCT match_id FROM tip_snapshots
            WHERE created_ts >= ?
              AND match_id NOT IN (SELECT match_id FROM match_results)
        """, (since,))
        ids = [r[0] for r in cur.fetchall()]

    if not ids:
        logging.info("[RESULTS] nothing to backfill in the last %sh", hours)
        set_setting("last_backfill_ts", str(int(time.time())))
        set_setting("last_backfill_updated", "0")
        set_setting("last_backfill_checked", "0")
        return 0, 0

    updated = 0
    B = 25
    finished_status = {"FT","AET","PEN"}  # only finalize truly finished games
    for i in range(0, len(ids), B):
        batch = ids[i:i+B]
        fx = fetch_fixtures_by_ids(batch)
        with db_conn() as conn:
            for fid in batch:
                m = fx.get(fid)
                if not m:
                    continue
                status = ((m.get("fixture") or {}).get("status") or {}).get("short")
                if status not in finished_status:
                    continue
                gh = (m.get("goals") or {}).get("home") or 0
                ga = (m.get("goals") or {}).get("away") or 0
                conn.execute("""
                    INSERT OR REPLACE INTO match_results(match_id, final_goals_h, final_goals_a, btts_yes, updated_ts)
                    VALUES (?,?,?,?,?)
                """, (int(fid), int(gh), int(ga), 1 if (int(gh)>0 and int(ga)>0) else 0, int(time.time())))
                updated += 1
            conn.commit()
        time.sleep(0.25)  # be nice to the API
    logging.info(f"[RESULTS] backfilled updated={updated} checked={len(ids)} (window={hours}h)")
    set_setting("last_backfill_ts", str(int(time.time())))
    set_setting("last_backfill_updated", str(updated))
    set_setting("last_backfill_checked", str(len(ids)))
    return updated, len(ids)

# ── Nightly training job ──────────────────────────────────────────────────────
def retrain_models_job():
    if not TRAIN_ENABLE:
        logging.info("[TRAIN] skipped (TRAIN_ENABLE=0)")
        return {"ok": False, "skipped": True, "reason": "TRAIN_ENABLE=0"}

    cmd = (
        f"python -u train_models.py "
        f"--db {DB_PATH} "
        f"--min-minute {TRAIN_MIN_MINUTE} "
        f"--test-size {TRAIN_TEST_SIZE}"
    )
    logging.info(f"[TRAIN] starting: {cmd}")
    try:
        proc = subprocess.run(
            shlex.split(cmd),
            capture_output=True,
            text=True,
            timeout=900  # 15 min safety
        )
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        logging.info(f"[TRAIN] returncode={proc.returncode}\nstdout:\n{out}\nstderr:\n{err}")

        try:
            set_setting("last_train_ts", str(int(time.time())))
            set_setting("last_train_rc", str(proc.returncode))
        except Exception:
            pass

        # Optional short ping to Telegram
        try:
            if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                summary = "✅ Nightly training OK" if proc.returncode == 0 else "❌ Nightly training failed"
                tail = "\n".join(out.splitlines()[-3:]) if out else ""
                send_telegram(f"{summary}\n{tail[:900]}")
        except Exception:
            pass

        return {"ok": proc.returncode == 0, "code": proc.returncode, "stdout": out[-2000:], "stderr": err[-1000:]}
    except subprocess.TimeoutExpired:
        logging.error("[TRAIN] timed out (15 min)")
        try:
            set_setting("last_train_ts", str(int(time.time())))
            set_setting("last_train_rc", "124")  # timeout
        except Exception:
            pass
        try: send_telegram("❌ Nightly training timed out after 15 min.")
        except Exception: pass
        return {"ok": False, "timeout": True}
    except Exception as e:
        logging.exception(f"[TRAIN] exception: {e}")
        try:
            set_setting("last_train_ts", str(int(time.time())))
            set_setting("last_train_rc", "1")
        except Exception:
            pass
        try: send_telegram(f"❌ Nightly training crashed: {e}")
        except Exception: pass
        return {"ok": False, "error": str(e)}

# ── Routes ────────────────────────────────────────────────────────────────────
@app.after_request
def add_security_headers(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["Cache-Control"] = "no-store"
    return resp

@app.route("/")
def home():
    mode = "HARVEST" if HARVEST_MODE else "PRODUCTION"
    return f"🤖 Robi Superbrain is active ({mode})."

@app.route("/harvest")
def harvest_route():
    _require_api_key()  # protect; remove if you want public
    saved, live_seen = harvest_scan()
    return jsonify({"ok": True, "live_seen": live_seen, "snapshots_saved": saved})

@app.route("/backfill")
def backfill_route():
    _require_api_key()
    try:
        hours = int(request.args.get("hours", 48))
    except Exception:
        hours = 48
    hours = max(1, min(hours, 168))
    updated, checked = backfill_results_from_snapshots(hours=hours)
    return jsonify({"ok": True, "hours": hours, "checked": checked, "updated": updated})

@app.route("/train", methods=["POST", "GET"])
def train_route():
    _require_api_key()
    result = retrain_models_job()
    return jsonify(result)

@app.route("/stats/snapshots_count")
def snapshots_count():
    _require_api_key()
    with db_conn() as conn:
        snap = conn.execute("SELECT COUNT(*) FROM tip_snapshots").fetchone()[0]
        tips = conn.execute("SELECT COUNT(*) FROM tips").fetchone()[0]
        res  = conn.execute("SELECT COUNT(*) FROM match_results").fetchone()[0]
        unlabeled = conn.execute("""
            SELECT COUNT(DISTINCT s.match_id)
            FROM tip_snapshots s
            LEFT JOIN match_results r ON r.match_id = s.match_id
            WHERE r.match_id IS NULL
        """).fetchone()[0]
    return jsonify({
        "tip_snapshots": int(snap),
        "tips_rows": int(tips),           # includes HARVEST rows with sent_ok=0
        "match_results": int(res),
        "unlabeled_match_ids": int(unlabeled)
    })

@app.route("/debug/env")
def debug_env():
    _require_api_key()
    def mark(val): return {"set": bool(val), "len": len(val) if val else 0}
    return jsonify({
        "API_KEY":            mark(API_KEY),
        "ADMIN_API_KEY":      {"set": bool(ADMIN_API_KEY)},  # don't expose length
        "TELEGRAM_BOT_TOKEN": mark(TELEGRAM_BOT_TOKEN),
        "TELEGRAM_CHAT_ID":   mark(TELEGRAM_CHAT_ID),
        "HARVEST_MODE":       HARVEST_MODE,
        "TRAIN_ENABLE":       TRAIN_ENABLE,
        "TRAIN_MIN_MINUTE":   TRAIN_MIN_MINUTE,
        "TRAIN_TEST_SIZE":    TRAIN_TEST_SIZE,
    })

def _require_api_key():
    key = request.headers.get("X-API-Key") or request.args.get("key")
    if not ADMIN_API_KEY or key != ADMIN_API_KEY:
        abort(401)

@app.route("/stats/config")
def stats_config():
    _require_api_key()
    return jsonify({
        "HARVEST_MODE": HARVEST_MODE,
        "CONF_THRESHOLD": CONF_THRESHOLD,
        "MAX_TIPS_PER_SCAN": MAX_TIPS_PER_SCAN,
        "DUP_COOLDOWN_MIN": DUP_COOLDOWN_MIN,
        "REQUIRE_STATS_MINUTE": REQUIRE_STATS_MINUTE,
        "REQUIRE_DATA_FIELDS": REQUIRE_DATA_FIELDS,
        "UNDER_SUPPRESS_AFTER_MIN": UNDER_SUPPRESS_AFTER_MIN,
        "O25_LATE_MINUTE": O25_LATE_MINUTE,
        "O25_LATE_MIN_GOALS": O25_LATE_MIN_GOALS,
        "BTTS_LATE_MINUTE": BTTS_LATE_MINUTE,
        "ONLY_MODEL_MODE": ONLY_MODEL_MODE,
        "API_KEY_set": bool(API_KEY),
        "ADMIN_API_KEY_set": bool(ADMIN_API_KEY),
        "TELEGRAM_CHAT_ID_set": bool(TELEGRAM_CHAT_ID),
        "TRAIN_ENABLE": TRAIN_ENABLE,
        "TRAIN_MIN_MINUTE": TRAIN_MIN_MINUTE,
        "TRAIN_TEST_SIZE": TRAIN_TEST_SIZE,
    })

@app.route("/healthz")
def healthz():
    # Very light health check: DB reachable
    try:
        with db_conn() as conn:
            conn.execute("SELECT 1")
        return jsonify({"ok": True}), 200
    except Exception as e:
        logging.exception("healthz failed")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/readyz")
def readyz():
    # Readiness: show last scheduler runs
    def to_int(name, default=0):
        try:
            return int(get_setting(name, str(default)) or default)
        except Exception:
            return default
    return jsonify({
        "harvest": {"last_ts": to_int("last_harvest_ts"), "last_saved": to_int("last_harvest_saved"), "last_live_seen": to_int("last_harvest_live_seen")},
        "backfill": {"last_ts": to_int("last_backfill_ts"), "last_updated": to_int("last_backfill_updated"), "last_checked": to_int("last_backfill_checked")},
        "train": {"last_ts": to_int("last_train_ts"), "last_rc": to_int("last_train_rc", -999)},
    })

# ── Entrypoint / Scheduler ────────────────────────────────────────────────────
if __name__ == "__main__":
    if not API_KEY:
        logging.error("API_KEY is not set — live fetch will return 0 matches.")
    if not ADMIN_API_KEY:
        logging.error("ADMIN_API_KEY is not set — admin endpoints will return 401.")
    init_db()

    scheduler = BackgroundScheduler()

    if HARVEST_MODE:
        # Sun–Thu, 09:00–21:59 Europe/Berlin every 2 min (no night pings)
        scheduler.add_job(
            harvest_scan,
            CronTrigger(
                day_of_week="sun,mon,tue,wed,thu",
                hour="9-21",
                minute="*/2",
                timezone=ZoneInfo("Europe/Berlin"),
            ),
            id="harvest",
            replace_existing=True,
        )

        # Backfill results every 15 min
        scheduler.add_job(
            backfill_results_from_snapshots,
            "interval",
            minutes=15,
            id="backfill",
            replace_existing=True,
        )

    # Nightly training every day at 03:00 Europe/Berlin (works in both modes)
    scheduler.add_job(
        retrain_models_job,
        CronTrigger(hour=3, minute=0, timezone=ZoneInfo("Europe/Berlin")),
        id="train",
        replace_existing=True,
        misfire_grace_time=3600,
        coalesce=True,
    )

    try:
        scheduler.start()
        logging.info("⏱️ Scheduler started (HARVEST_MODE=%s)", HARVEST_MODE)

        port = int(os.getenv("PORT", 5000))
        logging.info("✅ Robi Superbrain started.")
        app.run(host="0.0.0.0", port=port, use_reloader=False)
    finally:
        try:
            scheduler.shutdown(wait=False)
            logging.info("🛑 Scheduler stopped")
        except Exception:
            pass
