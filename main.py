import os
import json
import time
import logging
import requests
import psycopg2
import subprocess, shlex
from html import escape
from zoneinfo import ZoneInfo
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from flask import Flask, jsonify, request, abort
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

# ──────────────────────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
app = Flask(__name__)

# ── Env ───────────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")
API_KEY            = os.getenv("API_KEY")
ADMIN_API_KEY      = os.getenv("ADMIN_API_KEY")
PUBLIC_BASE_URL    = (os.getenv("PUBLIC_BASE_URL") or "").rstrip("/")
WEBHOOK_SECRET     = os.getenv("TELEGRAM_WEBHOOK_SECRET")
HEARTBEAT_ENABLE   = os.getenv("HEARTBEAT_ENABLE", "1") not in ("0","false","False","no","NO")

HARVEST_MODE       = os.getenv("HARVEST_MODE", "1") not in ("0","false","False","no","NO")
CONF_THRESHOLD     = int(os.getenv("CONF_THRESHOLD", "60"))
MAX_TIPS_PER_SCAN  = int(os.getenv("MAX_TIPS_PER_SCAN", "25"))
DUP_COOLDOWN_MIN   = int(os.getenv("DUP_COOLDOWN_MIN", "20"))

O25_LATE_MINUTE     = int(os.getenv("O25_LATE_MINUTE", "88"))
O25_LATE_MIN_GOALS  = int(os.getenv("O25_LATE_MIN_GOALS", "2"))
BTTS_LATE_MINUTE    = int(os.getenv("BTTS_LATE_MINUTE", "88"))
UNDER_SUPPRESS_AFTER_MIN = int(os.getenv("UNDER_SUPPRESS_AFTER_MIN", "82"))

ONLY_MODEL_MODE          = os.getenv("ONLY_MODEL_MODE", "0") not in ("0","false","False","no","NO")
REQUIRE_STATS_MINUTE     = int(os.getenv("REQUIRE_STATS_MINUTE", "35"))
REQUIRE_DATA_FIELDS      = int(os.getenv("REQUIRE_DATA_FIELDS", "2"))

LEAGUE_PRIORITY_IDS = [int(x) for x in (os.getenv("MOTD_LEAGUE_IDS", "39,140,135,78,61,2").split(",")) if x.strip().isdigit()]
MOTD_PREDICT        = os.getenv("MOTD_PREDICT", "1") not in ("0","false","False","no","NO")
MOTD_MIN_SAMPLES    = int(os.getenv("MOTD_MIN_SAMPLES", "30"))
MOTD_CONF_MIN       = int(os.getenv("MOTD_CONF_MIN", "65"))

ALLOWED_SUGGESTIONS = {"Over 2.5 Goals", "Under 2.5 Goals", "BTTS: Yes", "BTTS: No"}

TRAIN_ENABLE       = os.getenv("TRAIN_ENABLE", "1") not in ("0","false","False","no","NO")
TRAIN_MIN_MINUTE   = int(os.getenv("TRAIN_MIN_MINUTE", "15"))
TRAIN_TEST_SIZE    = float(os.getenv("TRAIN_TEST_SIZE", "0.25"))

# Postgres (Supabase) — REQUIRED
DATABASE_URL = os.getenv("DATABASE_URL")  # postgres://... ?sslmode=require
if not DATABASE_URL:
    raise SystemExit("DATABASE_URL is required (Supabase Postgres).")

# ── External APIs ─────────────────────────────────────────────────────────────
BASE_URL         = "https://v3.football.api-sports.io"
FOOTBALL_API_URL = f"{BASE_URL}/fixtures"
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
HEADERS          = {"x-apisports-key": API_KEY, "Accept": "application/json"}
INPLAY_STATUSES  = ["1H","HT","2H","ET","BT","P"]

# ── HTTP session ──────────────────────────────────────────────────────────────
session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], respect_retry_after_header=True)
session.mount("https://", HTTPAdapter(max_retries=retries))

# ── Caches ────────────────────────────────────────────────────────────────────
STATS_CACHE: Dict[int, Tuple[float, list]] = {}
EVENTS_CACHE: Dict[int, Tuple[float, list]] = {}
CAL_CACHE: Dict[str, Any] = {"ts": 0, "bins": []}

# ──────────────────────────────────────────────────────────────────────────────
# Postgres adapter (tiny convenience wrapper)
# ──────────────────────────────────────────────────────────────────────────────
class PgCursor:
    def __init__(self, cur): self.cur = cur
    def fetchone(self): return self.cur.fetchone()
    def fetchall(self): return self.cur.fetchall()

class PgConn:
    def __init__(self, dsn: str): self.dsn = dsn; self.conn=None; self.cur=None
    def __enter__(self):
        if "sslmode=" not in self.dsn:
            self.dsn = self.dsn + ("&" if "?" in self.dsn else "?") + "sslmode=require"
        self.conn = psycopg2.connect(self.dsn)
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

def db_conn() -> PgConn:
    return PgConn(DATABASE_URL)

# ── Init DB ───────────────────────────────────────────────────────────────────
def init_db():
    with db_conn() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS tips (
            match_id BIGINT,
            league_id BIGINT,
            league   TEXT,
            home     TEXT,
            away     TEXT,
            market   TEXT,
            suggestion TEXT,
            confidence DOUBLE PRECISION,
            score_at_tip TEXT,
            minute    INTEGER,
            created_ts BIGINT,
            sent_ok   INTEGER DEFAULT 1,
            PRIMARY KEY (match_id, created_ts)
        )""")

        conn.execute("""
        CREATE TABLE IF NOT EXISTS tip_snapshots (
            match_id BIGINT,
            created_ts BIGINT,
            payload TEXT,
            PRIMARY KEY (match_id, created_ts)
        )""")

        conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id SERIAL PRIMARY KEY,
            match_id BIGINT UNIQUE,
            verdict  INTEGER,
            created_ts BIGINT
        )""")

        conn.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )""")

        conn.execute("""
        CREATE TABLE IF NOT EXISTS match_results (
            match_id   BIGINT PRIMARY KEY,
            final_goals_h INTEGER,
            final_goals_a INTEGER,
            btts_yes      INTEGER,
            updated_ts    BIGINT
        )""")

        conn.execute("CREATE INDEX IF NOT EXISTS idx_tip_snaps_created ON tip_snapshots(created_ts)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tips_match ON tips(match_id)")

        try:
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
        except Exception:
            pass

def set_setting(key: str, value: str):
    with db_conn() as conn:
        conn.execute("""
            INSERT INTO settings(key,value) VALUES(%s,%s)
            ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value
        """, (key, value))

def get_setting(key: str, default: Optional[str] = None) -> Optional[str]:
    with db_conn() as conn:
        row = conn.execute("SELECT value FROM settings WHERE key=%s", (key,)).fetchone()
        return row[0] if row else default

# ── Telegram ──────────────────────────────────────────────────────────────────
def send_telegram(message: str, inline_keyboard: Optional[list] = None) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": escape(message), "parse_mode": "HTML"}
    if inline_keyboard:
        payload["reply_markup"] = json.dumps({"inline_keyboard": inline_keyboard})
    try:
        res = session.post(f"{TELEGRAM_API_URL}/sendMessage", data=payload, timeout=10)
        return res.ok
    except Exception:
        return False

# ── API helpers ───────────────────────────────────────────────────────────────
def _api_get(url: str, params: dict, timeout: int = 15):
    if not API_KEY:
        return None
    try:
        res = session.get(url, headers=HEADERS, params=params, timeout=timeout)
        if not res.ok:
            return None
        return res.json()
    except Exception:
        return None

def fetch_match_stats(fixture_id: int) -> Optional[List[Dict[str, Any]]]:
    now = time.time()
    if fixture_id in STATS_CACHE:
        ts, data = STATS_CACHE[fixture_id]
        if now - ts < 90:
            return data
    js = _api_get(f"{FOOTBALL_API_URL}/statistics", {"fixture": fixture_id})
    stats = js.get("response", []) if isinstance(js, dict) else None
    STATS_CACHE[fixture_id] = (now, stats or [])
    return stats

def fetch_match_events(fixture_id: int) -> Optional[List[Dict[str, Any]]]:
    now = time.time()
    if fixture_id in EVENTS_CACHE:
        ts, data = EVENTS_CACHE[fixture_id]
        if now - ts < 90:
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
        m["statistics"] = fetch_match_stats(fid) or []
        m["events"] = fetch_match_events(fid) or []
        out.append(m)
    return out

# ── Fixtures fetch ────────────────────────────────────────────────────────────
MAX_IDS_PER_REQ = 20

def fetch_fixtures_by_ids(ids: List[int]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    if not ids: return out
    for i in range(0, len(ids), MAX_IDS_PER_REQ):
        chunk = ids[i:i+MAX_IDS_PER_REQ]
        js = _api_get(FOOTBALL_API_URL, {"ids": ",".join(map(str,chunk)), "timezone": "UTC"})
        resp = js.get("response", []) if isinstance(js, dict) else []
        for fx in resp:
            fid = (fx.get("fixture") or {}).get("id")
            if fid: out[int(fid)] = fx
        missing = [fid for fid in chunk if fid not in out]
        for fid in missing:
            js1 = _api_get(FOOTBALL_API_URL, {"id": fid, "timezone": "UTC"})
            r1 = (js1.get("response") if isinstance(js1, dict) else []) or []
            if r1: out[int(fid)] = r1[0]
        time.sleep(0.15)
    return out

def fetch_fixtures_by_ids_hyphen(ids: List[int]) -> Dict[int, Dict[str, Any]]:
    if not ids: return {}
    out: Dict[int, Dict[str, Any]] = {}
    for i in range(0, len(ids), MAX_IDS_PER_REQ):
        chunk = ids[i:i+MAX_IDS_PER_REQ]
        js = _api_get(FOOTBALL_API_URL, {"ids": "-".join(map(str,chunk)), "timezone": "UTC"})
        resp = js.get("response", []) if isinstance(js, dict) else []
        for fx in resp:
            fid = (fx.get("fixture") or {}).get("id")
            if fid: out[int(fid)] = fx
        time.sleep(0.15)
    return out

# ── Features & snapshots ──────────────────────────────────────────────────────
def _num(v) -> float:
    try:
        if isinstance(v, str) and v.endswith('%'): return float(v[:-1])
        return float(v or 0)
    except Exception:
        return 0.0

def _pos_pct(v) -> float:
    try:
        return float(str(v).replace('%','').strip() or 0)
    except Exception:
        return 0.0

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
    xg_h = _num(sh.get("Expected Goals", 0)); xg_a = _num(sa.get("Expected Goals", 0))
    sot_h = _num(sh.get("Shots on Target", 0)); sot_a = _num(sa.get("Shots on Target", 0))
    cor_h = _num(sh.get("Corner Kicks", 0));   cor_a = _num(sa.get("Corner Kicks", 0))
    pos_h = _pos_pct(sh.get("Ball Possession", 0)); pos_a = _pos_pct(sa.get("Ball Possession", 0))

    red_h = 0; red_a = 0
    for ev in (match.get("events") or []):
        try:
            if (ev.get("type","").lower() == "card") and ("red" in (ev.get("detail","").lower())):
                tname = (ev.get("team") or {}).get("name") or ""
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
    if minute < REQUIRE_STATS_MINUTE: return True
    fields = [feat.get("xg_sum",0.0), feat.get("sot_sum",0.0), feat.get("cor_sum",0.0), max(feat.get("pos_h",0.0), feat.get("pos_a",0.0))]
    nonzero = sum(1 for v in fields if (v or 0) > 0)
    return nonzero >= max(0, REQUIRE_DATA_FIELDS)

def save_snapshot_from_match(m: Dict[str, Any], feat: Dict[str, float]) -> None:
    fx = m.get("fixture", {}) or {}; lg = m.get("league", {}) or {}
    fid = int(fx.get("id")); league_id = int(lg.get("id") or 0)
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
            INSERT INTO tip_snapshots(match_id, created_ts, payload)
            VALUES (%s,%s,%s)
            ON CONFLICT (match_id, created_ts) DO UPDATE SET payload=EXCLUDED.payload
        """, (fid, now, json.dumps(snapshot)[:200000]))

        conn.execute("""
            INSERT INTO tips(match_id,league_id,league,home,away,market,suggestion,confidence,score_at_tip,minute,created_ts,sent_ok)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (fid, league_id, league, home, away, "HARVEST", "HARVEST", 0.0, f"{gh}-{ga}", minute, now, 0))

# ── League helpers / backfill ─────────────────────────────────────────────────
def _chunk(seq: List[int], size: int) -> List[List[int]]:
    size = max(1, int(size)); return [seq[i:i+size] for i in range(0, len(seq), size)]

def get_fixture_ids_for_league(league_id: int, season: int, statuses: List[str]) -> List[int]:
    status_param = "-".join(statuses) if statuses else "FT-AET-PEN"
    js = _api_get(FOOTBALL_API_URL, {"league": league_id, "season": season, "status": status_param, "timezone": "UTC"})
    if not isinstance(js, dict): return []
    ids: List[int] = []
    for item in (js.get("response") or []):
        fx = item.get("fixture") or {}; fid = fx.get("id")
        if fid: ids.append(int(fid))
    return ids

def _ensure_events_stats_present(m: Dict[str, Any], fid: int) -> Dict[str, Any]:
    if not m.get("statistics"): m["statistics"] = fetch_match_stats(fid) or []
    if not m.get("events"):     m["events"] = fetch_match_events(fid) or []
    return m

def create_synthetic_snapshots_for_league(league_id:int, season:int, include_inplay:bool=False,
                                          extra_statuses:Optional[List[str]]=None, max_per_run:Optional[int]=None,
                                          sleep_ms_between_chunks:int=1000) -> Dict[str, Any]:
    base_statuses = ["FT","AET","PEN"]
    if include_inplay: base_statuses += INPLAY_STATUSES
    if extra_statuses:
        for s in extra_statuses:
            if s not in base_statuses: base_statuses.append(s)
    ids = get_fixture_ids_for_league(league_id, season, base_statuses)
    if not ids: return {"ok": True, "league": league_id, "season": season, "requested": 0, "processed": 0, "saved": 0}
    if max_per_run and max_per_run > 0: ids = ids[:max_per_run]

    saved = 0; processed = 0
    for group in _chunk(ids, MAX_IDS_PER_REQ):
        bulk = fetch_fixtures_by_ids_hyphen(group)
        for fid in group:
            m = bulk.get(fid)
            if not m:
                single = fetch_fixtures_by_ids([fid]).get(fid)
                if not single:
                    logging.warning("[SYNTH] fixture %s not returned by API (bulk+single)", fid)
                    continue
                m = single
            processed += 1
            m = _ensure_events_stats_present(m, fid)
            feat = extract_features(m)
            minute = int(((m.get("fixture", {}) or {}).get("status", {}) or {}).get("elapsed") or 90)
            feat["minute"] = float(min(120, max(0, minute)))
            save_snapshot_from_match(m, feat)
            saved += 1
        time.sleep(max(0, sleep_ms_between_chunks) / 1000.0)

    return {"ok": True, "league": league_id, "season": season, "requested": len(ids), "processed": processed, "saved": saved}

def backfill_results_from_snapshots(hours: int = 48) -> Tuple[int, int]:
    hours = max(1, min(int(hours), 168))
    since = int(time.time()) - int(hours) * 3600
    with db_conn() as conn:
        rows = conn.execute("""
            SELECT DISTINCT match_id FROM tip_snapshots
            WHERE created_ts >= %s
              AND match_id NOT IN (SELECT match_id FROM match_results)
        """, (since,)).fetchall()
        ids = [r[0] for r in rows]

    if not ids: return (0, 0)

    updated = 0
    finished_status = {"FT","AET","PEN"}
    for group in _chunk(ids, 20):
        fx = fetch_fixtures_by_ids(group)
        with db_conn() as conn:
            for fid in group:
                m = fx.get(fid)
                if not m:
                    logging.warning("[RESULTS] fixture %s not returned by API", fid)
                    continue
                status = ((m.get("fixture") or {}).get("status") or {}).get("short")
                if status not in finished_status:
                    continue
                gh = (m.get("goals") or {}).get("home") or 0
                ga = (m.get("goals") or {}).get("away") or 0
                conn.execute("""
                    INSERT INTO match_results(match_id, final_goals_h, final_goals_a, btts_yes, updated_ts)
                    VALUES (%s,%s,%s,%s,%s)
                    ON CONFLICT (match_id) DO UPDATE SET
                        final_goals_h=EXCLUDED.final_goals_h,
                        final_goals_a=EXCLUDED.final_goals_a,
                        btts_yes=EXCLUDED.btts_yes,
                        updated_ts=EXCLUDED.updated_ts
                """, (int(fid), int(gh), int(ga), 1 if (int(gh)>0 and int(ga)>0) else 0, int(time.time())))
                updated += 1
        time.sleep(0.25)
    return updated, len(ids)

def _list_unlabeled_ids(limit: Optional[int] = None) -> List[int]:
    with db_conn() as conn:
        sql = """
            SELECT DISTINCT s.match_id
            FROM tip_snapshots s
            LEFT JOIN match_results r ON r.match_id = s.match_id
            WHERE r.match_id IS NULL
            ORDER BY s.match_id
        """
        rows = conn.execute(sql + (" LIMIT %s" if limit else ""), ((limit,) if limit else ())).fetchall()
    return [r[0] for r in rows]

def _backfill_for_ids(ids: List[int]) -> Tuple[int, int]:
    if not ids: return (0,0)
    updated = 0
    finished_status = {"FT","AET","PEN"}
    for group in _chunk(ids, MAX_IDS_PER_REQ):
        fx = fetch_fixtures_by_ids(group)
        with db_conn() as conn:
            for fid in group:
                m = fx.get(fid)
                if not m:
                    logging.warning("[RESULTS] fixture %s not returned by API", fid)
                    continue
                status = ((m.get("fixture") or {}).get("status") or {}).get("short")
                if status not in finished_status: continue
                gh = (m.get("goals") or {}).get("home") or 0
                ga = (m.get("goals") or {}).get("away") or 0
                conn.execute("""
                    INSERT INTO match_results(match_id, final_goals_h, final_goals_a, btts_yes, updated_ts)
                    VALUES (%s,%s,%s,%s,%s)
                    ON CONFLICT (match_id) DO UPDATE SET
                        final_goals_h=EXCLUDED.final_goals_h,
                        final_goals_a=EXCLUDED.final_goals_a,
                        btts_yes=EXCLUDED.btts_yes,
                        updated_ts=EXCLUDED.updated_ts
                """, (int(fid), int(gh), int(ga), 1 if (int(gh)>0 and int(ga)>0) else 0, int(time.time())))
                updated += 1
        time.sleep(0.25)
    return updated, len(ids)

# ── Training job (calls external script) ──────────────────────────────────────
def retrain_models_job():
    if not TRAIN_ENABLE:
        logging.info("[TRAIN] skipped (TRAIN_ENABLE=0)")
        return {"ok": False, "skipped": True, "reason": "TRAIN_ENABLE=0"}

    cmd = (
        f"python -u train_models.py "
        f"--db-url \"{os.getenv('DATABASE_URL','')}\" "
        f"--min-minute {TRAIN_MIN_MINUTE} "
        f"--test-size {TRAIN_TEST_SIZE}"
    )
    logging.info(f"[TRAIN] starting: {cmd}")
    try:
        proc = subprocess.run(
            shlex.split(cmd),
            capture_output=True,
            text=True,
            timeout=900
        )
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        logging.info(f"[TRAIN] returncode={proc.returncode}\nstdout:\n{out}\nstderr:\n{err}")

        try:
            if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                summary = "✅ Nightly training OK" if proc.returncode == 0 else "❌ Nightly training failed"
                tail = "\n".join(out.splitlines()[-3:]) if out else ""
                session.post(f"{TELEGRAM_API_URL}/sendMessage",
                             data={"chat_id": TELEGRAM_CHAT_ID, "text": summary + "\n" + tail[:900]},
                             timeout=10)
        except Exception:
            pass

        return {"ok": proc.returncode == 0, "code": proc.returncode,
                "stdout": out[-2000:], "stderr": err[-1000:]}
    except subprocess.TimeoutExpired:
        logging.error("[TRAIN] timed out (15 min)")
        return {"ok": False, "timeout": True}
    except Exception as e:
        logging.exception(f"[TRAIN] exception: {e}")
        return {"ok": False, "error": str(e)}

# ── Simple stats & digest ─────────────────────────────────────────────────────
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
    return {
        "snap_total": int(snap_total),
        "tips_total": int(tips_total),
        "res_total": int(res_total),
        "unlabeled": int(unlabeled),
        "snap_24h": int(snap_24h),
        "res_24h": int(res_24h),
    }

def nightly_digest_job():
    try:
        now = int(time.time())
        day_ago = now - 24*3600
        c = _counts_since(day_ago)
        msg = (
            "📊 <b>Robi Nightly Digest</b>\n"
            f"Snapshots: {c['snap_total']} (＋{c['snap_24h']} last 24h)\n"
            f"Finals: {c['res_total']} (＋{c['res_24h']} last 24h)\n"
            f"Unlabeled match_ids: {c['unlabeled']}\n"
            "Models retrain at 03:00 CEST daily."
        )
        logging.info("[DIGEST]\n%s", msg.replace("<b>","").replace("</b>",""))
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            send_telegram(msg)
        return True
    except Exception as e:
        logging.exception("[DIGEST] failed: %s", e)
        return False

# ── Harvest scan (final, single version) ──────────────────────────────────────
def harvest_scan() -> Tuple[int, int]:
    matches = fetch_live_matches()
    live_seen = len(matches)
    if live_seen == 0:
        logging.info("[HARVEST] no live matches")
        return 0, 0

    saved = 0
    now_ts = int(time.time())

    with db_conn() as conn:
        for m in matches:
            try:
                fid = int((m.get("fixture", {}) or {}).get("id") or 0)
                if not fid:
                    continue

                # cooldown – avoid writing multiple rows per match too frequently
                if DUP_COOLDOWN_MIN > 0:
                    cutoff = now_ts - (DUP_COOLDOWN_MIN * 60)
                    dup = conn.execute(
                        "SELECT 1 FROM tips WHERE match_id=%s AND created_ts>=%s LIMIT 1",
                        (fid, cutoff)
                    ).fetchone()
                    if dup:
                        continue

                feat = extract_features(m)
                minute = int(feat.get("minute", 0))

                # require some live stats after a given minute (configurable)
                if not stats_coverage_ok(feat, minute):
                    continue

                save_snapshot_from_match(m, feat)
                saved += 1

                if MAX_TIPS_PER_SCAN and saved >= MAX_TIPS_PER_SCAN:
                    break

            except Exception as e:
                logging.exception(f"[HARVEST] failure on match: {e}")
                continue

    logging.info(f"[HARVEST] saved={saved} live_seen={live_seen}")
    return saved, live_seen

# ── Routes ────────────────────────────────────────────────────────────────────
@app.after_request
def add_security_headers(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["Cache-Control"] = "no-store"
    return resp

@app.route("/harvest/bulk_leagues", methods=["POST"])
def harvest_bulk_leagues_post():
    _require_api_key()
    body = request.get_json(silent=True) or {}
    leagues = body.get("leagues") or []
    seasons = body.get("seasons") or [2024]
    return jsonify(_run_bulk_leagues(leagues, seasons))

# /harvest/bulk_leagues?key=...&seasons=2023,2024
@app.route("/harvest/bulk_leagues", methods=["GET"])
def harvest_bulk_leagues_get():
    _require_api_key()
    seasons_q = (request.args.get("seasons") or "").strip()
    seasons = [int(s) for s in seasons_q.split(",") if s.strip().isdigit()] or [2024]
    leagues = []  # will use curated defaults
    return jsonify(_run_bulk_leagues(leagues, seasons))

@app.route("/")
def home():
    mode = "HARVEST" if HARVEST_MODE else "PRODUCTION"
    return f"🤖 Robi Superbrain is active ({mode}) · DB=Postgres"

def _require_api_key():
    key = request.headers.get("X-API-Key") or request.args.get("key")
    if not ADMIN_API_KEY or key != ADMIN_API_KEY:
        abort(401)

@app.route("/harvest")
def harvest_route():
    _require_api_key()
    saved, live_seen = harvest_scan()
    return jsonify({"ok": True, "live_seen": live_seen, "snapshots_saved": saved})

@app.route("/harvest/league_ids")
def harvest_league_ids_route():
    _require_api_key()
    try:
        league = int(request.args.get("league")); season = int(request.args.get("season"))
    except Exception:
        abort(400)
    statuses_param = request.args.get("statuses", "")
    include_inplay = request.args.get("include_inplay", "0") not in ("0","false","False","no","NO")
    statuses = [s for s in statuses_param.split("-") if s] if statuses_param else ["FT","AET","PEN"]
    if include_inplay:
        for s in INPLAY_STATUSES:
            if s not in statuses: statuses.append(s)
    ids = get_fixture_ids_for_league(league, season, statuses)
    return jsonify({"ok": True, "league": league, "season": season, "statuses": "-".join(statuses), "count": len(ids), "ids": ids})

@app.route("/harvest/league_snapshots")
def harvest_league_snapshots_route():
    _require_api_key()
    try:
        league = int(request.args.get("league")); season = int(request.args.get("season"))
    except Exception:
        abort(400)
    include_inplay = request.args.get("include_inplay", "0") not in ("0","false","False","no","NO")
    statuses_param = request.args.get("statuses", "")
    extra_statuses = [s for s in statuses_param.split("-") if s] if statuses_param else None
    limit_param = request.args.get("limit")
    limit = int(limit_param) if (limit_param and str(limit_param).isdigit()) else None
    delay_ms = int(request.args.get("delay_ms", 1000))
    summary = create_synthetic_snapshots_for_league(league, season, include_inplay, extra_statuses, limit, delay_ms)
    return jsonify(summary)

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

@app.route("/backfill/all")
def backfill_all_unlabeled_route():
    _require_api_key()
    limit = request.args.get("limit")
    ids = _list_unlabeled_ids(int(limit) if (limit and str(limit).isdigit()) else None)
    updated, checked = _backfill_for_ids(ids)
    return jsonify({"ok": True, "mode": "all_unlabeled", "checked": checked, "updated": updated})

@app.route("/debug/backfill_preview")
def debug_backfill_preview():
    _require_api_key()
    hours = request.args.get("hours")
    if hours is not None:
        try:
            h = max(1, min(int(hours), 168))
        except Exception:
            abort(400)
        since = int(time.time()) - h * 3600
        with db_conn() as conn:
            rows = conn.execute("""
                SELECT DISTINCT match_id FROM tip_snapshots
                WHERE created_ts >= %s
                  AND match_id NOT IN (SELECT match_id FROM match_results)
                ORDER BY match_id
            """, (since,)).fetchall()
            ids = [r[0] for r in rows]
    else:
        ids = _list_unlabeled_ids(int(request.args.get("limit", "0")) or None)
    fx = fetch_fixtures_by_ids(ids)
    finished_status = {"FT","AET","PEN"}
    out = []
    for fid in ids:
        m = fx.get(fid)
        if not m:
            out.append({"match_id": fid, "fetched": False}); continue
        st = ((m.get("fixture") or {}).get("status") or {}).get("short")
        gh = (m.get("goals") or {}).get("home"); ga = (m.get("goals") or {}).get("away")
        out.append({"match_id": fid, "fetched": True, "status": st, "goals_h": gh, "goals_a": ga, "will_update": st in finished_status})
    return jsonify({"count": len(ids), "candidates": out})

@app.route("/train", methods=["POST", "GET"])
def train_route():
    _require_api_key()
    return jsonify(retrain_models_job())

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
    return jsonify({"tip_snapshots": int(snap), "tips_rows": int(tips), "match_results": int(res), "unlabeled_match_ids": int(unlabeled)})

@app.route("/stats/progress")
def stats_progress():
    _require_api_key()
    now = int(time.time())
    day_ago = now - 24*3600
    week_ago = now - 7*24*3600
    c24 = _counts_since(day_ago)
    c7  = _counts_since(week_ago)
    return jsonify({
        "window_24h": {"snapshots_added": c24["snap_24h"], "results_added": c24["res_24h"]},
        "totals": {"tip_snapshots": c24["snap_total"], "tips_rows": c24["tips_total"], "match_results": c24["res_total"], "unlabeled_match_ids": c24["unlabeled"]},
        "window_7d": {"snapshots_added_est": c7["snap_24h"], "results_added_est": c7["res_24h"]},
    })

@app.route("/debug/env")
def debug_env():
    _require_api_key()
    def mark(val): return {"set": bool(val), "len": (len(val) if val else 0)}
    return jsonify({
        "API_KEY":            mark(API_KEY),
        "ADMIN_API_KEY":      {"set": bool(ADMIN_API_KEY)},
        "TELEGRAM_BOT_TOKEN": mark(TELEGRAM_BOT_TOKEN),
        "TELEGRAM_CHAT_ID":   mark(TELEGRAM_CHAT_ID),
        "HARVEST_MODE":       HARVEST_MODE,
        "TRAIN_ENABLE":       TRAIN_ENABLE,
        "TRAIN_MIN_MINUTE":   TRAIN_MIN_MINUTE,
        "TRAIN_TEST_SIZE":    TRAIN_TEST_SIZE,
        "DB": "Postgres",
    })

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
        "TELEGRAM_CHAT_ID_set": bool(TELEGRAM_CHAT_ID),
        "TRAIN_ENABLE": TRAIN_ENABLE,
        "TRAIN_MIN_MINUTE": TRAIN_MIN_MINUTE,
        "TRAIN_TEST_SIZE": TRAIN_TEST_SIZE,
    })

# ── Entrypoint / Scheduler ────────────────────────────────────────────────────
if __name__ == "__main__":
    if not API_KEY: logging.error("API_KEY is not set — live fetch will return 0 matches.")
    if not ADMIN_API_KEY: logging.error("ADMIN_API_KEY is not set — admin endpoints will 401.")
    init_db()

    scheduler = BackgroundScheduler()
    if HARVEST_MODE:
        # every 2 min during daytime, Europe/Berlin
        scheduler.add_job(
            harvest_scan,
            CronTrigger(day_of_week="sun,mon,tue,wed,thu", hour="9-21", minute="*/2", timezone=ZoneInfo("Europe/Berlin")),
            id="harvest", replace_existing=True,
        )
        scheduler.add_job(backfill_results_from_snapshots, "interval", minutes=15, id="backfill", replace_existing=True)

    scheduler.add_job(
        retrain_models_job,
        CronTrigger(hour=3, minute=0, timezone=ZoneInfo("Europe/Berlin")),
        id="train", replace_existing=True, misfire_grace_time=3600, coalesce=True,
    )
    scheduler.add_job(
        nightly_digest_job,
        CronTrigger(hour=3, minute=2, timezone=ZoneInfo("Europe/Berlin")),
        id="digest", replace_existing=True, misfire_grace_time=3600, coalesce=True,
    )

    scheduler.start()
    logging.info("⏱️ Scheduler started (HARVEST_MODE=%s)", HARVEST_MODE)
    port = int(os.getenv("PORT", 5000))
    logging.info("✅ Robi Superbrain started.")
    app.run(host="0.0.0.0", port=port)
