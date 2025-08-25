# path: main.py
import os
import re
import json
import time
import math
import logging
import requests
import psycopg2
from typing import List, Dict, Any, Optional, Tuple
from flask import Flask, jsonify, request, abort
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from zoneinfo import ZoneInfo

# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
app = Flask(__name__)

# Env
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")
API_KEY            = os.getenv("API_KEY")
ADMIN_API_KEY      = os.getenv("ADMIN_API_KEY")

HARVEST_MODE       = os.getenv("HARVEST_MODE", "1") not in ("0","false","False","no","NO")
MAX_TIPS_PER_SCAN  = int(os.getenv("MAX_TIPS_PER_SCAN", "25"))
DUP_COOLDOWN_MIN   = int(os.getenv("DUP_COOLDOWN_MIN", "20"))

# Pure‑model thresholds (no odds/filters): global + per head overrides
ONLY_MODEL_MODE    = True  # force pure model mode
MIN_PROB           = max(0.0, min(0.99, float(os.getenv("MIN_PROB", "0.55"))))

HEAD_MINP = {
    "BTTS":      float(os.getenv("MINP_BTTS",      str(MIN_PROB))),
    "WIN_HOME":  float(os.getenv("MINP_WIN_HOME",  str(MIN_PROB))),
    "DRAW":      float(os.getenv("MINP_DRAW",      str(MIN_PROB))),
    "WIN_AWAY":  float(os.getenv("MINP_WIN_AWAY",  str(MIN_PROB))),
    "O15":       float(os.getenv("MINP_O15",       str(MIN_PROB))),
    "O25":       float(os.getenv("MINP_O25",       str(MIN_PROB))),
    "O35":       float(os.getenv("MINP_O35",       str(MIN_PROB))),
    "O45":       float(os.getenv("MINP_O45",       str(MIN_PROB))),
}

# MOTD (Match of the Day)
MOTD_ENABLE        = os.getenv("MOTD_ENABLE", "1") not in ("0","false","False","no","NO")
MOTD_HOUR_LOCAL    = int(os.getenv("MOTD_HOUR", "10"))  # 10:00 local (Europe/Berlin)
MOTD_LOOKBACK_HRS  = int(os.getenv("MOTD_LOOKBACK_HRS", "24"))

# Training toggles (unchanged)
TRAIN_ENABLE       = os.getenv("TRAIN_ENABLE", "1") not in ("0","false","False","no","NO")
TRAIN_MIN_MINUTE   = int(os.getenv("TRAIN_MIN_MINUTE", "15"))

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

STATS_CACHE: Dict[int, Tuple[float, list]] = {}
EVENTS_CACHE: Dict[int, Tuple[float, list]] = {}

# ──────────────────────────────────────────────────────────────────────────────
# DB helpers (same schema as before)
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
        conn.execute("""CREATE TABLE IF NOT EXISTS tip_snapshots (match_id BIGINT, created_ts BIGINT, payload TEXT, PRIMARY KEY (match_id, created_ts))""")
        conn.execute("""CREATE TABLE IF NOT EXISTS feedback (id SERIAL PRIMARY KEY, match_id BIGINT UNIQUE, verdict INTEGER, created_ts BIGINT)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS match_results (match_id BIGINT PRIMARY KEY, final_goals_h INTEGER, final_goals_a INTEGER, btts_yes INTEGER, updated_ts BIGINT)""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tip_snaps_created ON tip_snapshots(created_ts)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tips_match ON tips(match_id)")
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_tips_dedupe
            ON tips(match_id, market, suggestion, created_ts)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_tips_recent
            ON tips(match_id, created_ts)
        """)

# ──────────────────────────────────────────────────────────────────────────────
# API wrappers
def _api_get(url: str, params: dict, timeout: int = 15):
    if not API_KEY: return None
    try:
        res = session.get(url, headers=HEADERS, params=params, timeout=timeout)
        if not res.ok:
            logging.warning("[API] %s %s -> %s", url, params, res.status_code)
            return None
        return res.json()
    except Exception as e:
        logging.exception("[API] error: %s", e)
        return None

def fetch_match_stats(fixture_id: int):
    now = time.time()
    if fixture_id in STATS_CACHE:
        ts, data = STATS_CACHE[fixture_id]
        if now - ts < 90: return data
    js = _api_get(f"{FOOTBALL_API_URL}/statistics", {"fixture": fixture_id})
    stats = js.get("response", []) if isinstance(js, dict) else None
    STATS_CACHE[fixture_id] = (now, stats or [])
    return stats

def fetch_match_events(fixture_id: int):
    now = time.time()
    if fixture_id in EVENTS_CACHE:
        ts, data = EVENTS_CACHE[fixture_id]
        if now - ts < 90: return data
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
        elapsed = status.get("elapsed"); short = (status.get("short") or "").upper()
        if elapsed is None or elapsed > 90:
            continue
        if short not in INPLAY_STATUSES:
            continue
        fid = (m.get("fixture", {}) or {}).get("id")
        m["statistics"] = fetch_match_stats(fid) or []
        m["events"] = fetch_match_events(fid) or []
        out.append(m)
    return out

# ──────────────────────────────────────────────────────────────────────────────
# Feature extraction
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
    home = (match.get("teams") or {}).get("home", {}).get("name", "")
    away = (match.get("teams") or {}).get("away", {}).get("name", "")
    gh = (match.get("goals") or {}).get("home") or 0
    ga = (match.get("goals") or {}).get("away") or 0
    minute = int((match.get("fixture", {}).get("status", {}) or {}).get("elapsed") or 0)
    stats_blocks = match.get("statistics") or []
    stats: Dict[str, Dict[str, Any]] = {}
    for s in stats_blocks:
        tname = (s.get("team") or {}).get("name")
        if tname:
            stats[tname] = {i["type"]: i["value"] for i in (s.get("statistics") or [])}
    sh = stats.get(home, {}); sa = stats.get(away, {})
    xg_h = _num(sh.get("Expected Goals", 0)); xg_a = _num(sa.get("Expected Goals", 0))
    sot_h = _num(sh.get("Shots on Target", 0)); sot_a = _num(sa.get("Shots on Target", 0))
    cor_h = _num(sh.get("Corner Kicks", 0));   cor_a = _num(sa.get("Corner Kicks", 0))
    pos_h = _pos_pct(sh.get("Ball Possession", 0)); pos_a = _pos_pct(sa.get("Ball Possession", 0))

    red_h = 0; red_a = 0
    for ev in (match.get("events") or []):
        try:
            if (ev.get("type","").lower() == "card"):
                detail = (ev.get("detail","") or "").lower()
                if ("red" in detail) or ("second yellow" in detail):
                    tname = (ev.get("team") or {}).get("name") or ""
                    if tname == home: red_h += 1
                    elif tname == away: red_a += 1
        except Exception:
            pass

    return {
        "minute": float(minute),
        "goals_h": float(gh), "goals_a": float(ga),
        "goals_sum": float(gh + ga), "goals_diff": float(gh - ga),
        "xg_h": float(xg_h), "xg_a": float(xg_a), "xg_sum": float(xg_h + xg_a), "xg_diff": float(xg_h - xg_a),
        "sot_h": float(sot_h), "sot_a": float(sot_a), "sot_sum": float(sot_h + sot_a),
        "cor_h": float(cor_h), "cor_a": float(cor_a), "cor_sum": float(cor_h + cor_a),
        "pos_h": float(pos_h), "pos_a": float(pos_a), "pos_diff": float(pos_h - pos_a),
        "red_h": float(red_h), "red_a": float(red_a), "red_sum": float(red_h + red_a),
    }

# ──────────────────────────────────────────────────────────────────────────────
# Models & scoring (pure)
OU_RE = re.compile(r"^O(\d{2})$")

def _sigmoid(x: float) -> float:
    if x < -50: return 1e-22
    if x > 50: return 1.0 - 1e-22
    return 1.0 / (1.0 + math.exp(-x))

def _linpred(feat: Dict[str,float], weights: Dict[str,float], intercept: float) -> float:
    s = float(intercept or 0.0)
    for k, w in (weights or {}).items():
        s += float(w or 0.0) * float(feat.get(k, 0.0))
    return s

def _calibrate_prob(p: float, cal: Dict[str,Any]) -> float:
    method = (cal or {}).get("method", "sigmoid").lower()
    a = float((cal or {}).get("a", 1.0)); b = float((cal or {}).get("b", 0.0))
    p = max(1e-12, min(1-1e-12, float(p)))
    if method in ("platt","sigmoid"):
        logit = math.log(p / (1 - p))
        return 1.0 / (1.0 + math.exp(-(a * logit + b)))
    return p

def _score_prob(feat: Dict[str,float], mdl: Dict[str,Any]) -> float:
    lp = _linpred(feat, mdl.get("weights", {}), float(mdl.get("intercept", 0.0)))
    p = _sigmoid(lp)
    cal = mdl.get("calibration") or {}
    try:
        p = _calibrate_prob(p, cal) if cal else p
    except Exception:
        pass
    return max(0.0, min(1.0, float(p)))

def _list_models(prefix: str = "model_v2:") -> Dict[str, Dict[str,Any]]:
    with db_conn() as conn:
        rows = conn.execute("SELECT key, value FROM settings WHERE key LIKE %s", (prefix + "%",)).fetchall()
    out = {}
    for k, v in rows:
        try:
            code = k.split(":",1)[1]
            if code.lower() in ("o05","o5"):
                continue
            out[code] = json.loads(v)
        except Exception:
            continue
    return out

def _ou_label(code: str) -> Optional[float]:
    m = OU_RE.match(code)
    if not m: return None
    return int(m.group(1)) / 10.0

def _mk_suggestions_for_code(code: str, mdl: Dict[str,Any], feat: Dict[str,float]) -> List[Tuple[str,str,float,str]]:
    out: List[Tuple[str,str,float,str]] = []
    th = _ou_label(code)
    if th is not None:
        p_over = _score_prob(feat, mdl)
        th_txt = f"{th:.1f}"; market = f"Over/Under {th_txt}"
        out.append((market, f"Over {th_txt} Goals", p_over, code))
        out.append((market, f"Under {th_txt} Goals", 1.0 - p_over, code))
        return out
    if code in ("BTTS","BTTS_YES","BTTS_NO"):
        p = _score_prob(feat, mdl)
        out.append(("BTTS", "BTTS: Yes", p, "BTTS"))
        out.append(("BTTS", "BTTS: No", 1.0 - p, "BTTS"))
        return out
    if code == "WIN_HOME":
        out.append(("Match Result 1X2", "Home Win", _score_prob(feat, mdl), code)); return out
    if code == "DRAW":
        out.append(("Match Result 1X2", "Draw", _score_prob(feat, mdl), code)); return out
    if code == "WIN_AWAY":
        out.append(("Match Result 1X2", "Away Win", _score_prob(feat, mdl), code)); return out
    p = _score_prob(feat, mdl)
    out.append((code, f"{code}: Yes", p, code))
    out.append((code, f"{code}: No", 1.0 - p, code))
    return out

def _min_prob_for_head(head: str) -> float:
    return float(HEAD_MINP.get(head, MIN_PROB))

# ──────────────────────────────────────────────────────────────────────────────
# Telegram & formatting
def tg_escape(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def send_telegram(message: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return False
    try:
        res = session.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT_ID, "text": tg_escape(message), "parse_mode": "HTML", "disable_web_page_preview": True},
            timeout=10
        )
        return res.ok
    except Exception:
        return False

def _rationale_from_feat(feat: Dict[str, float]) -> str:
    parts = []
    gd = int(feat.get("goals_diff", 0))
    parts.append("home leading" if gd > 0 else ("away leading" if gd < 0 else "level"))
    parts.append(f"xG_sum {feat.get('xg_sum',0):.2f}")
    sot = int(feat.get('sot_sum',0));  cor = int(feat.get('cor_sum',0)); red = int(feat.get('red_sum',0))
    if sot: parts.append(f"SOT {sot}")
    if cor: parts.append(f"corners {cor}")
    if red: parts.append(f"reds {red}")
    parts.append(f"minute {int(feat.get('minute',0))}")
    return ", ".join(parts)

def _format_tip_message_pure(
    league: str, home: str, away: str, minute: int, score_txt: str,
    suggestion: str, prob: float, rationale: str
) -> str:
    prob_pct = min(99.0, max(0.0, prob * 100.0))
    parts = [
        "⚽️ New Tip!",
        f"Match: {home} vs {away}",
        f"⏰ Minute: {minute}' | Score: {score_txt}",
        f"Tip: {suggestion}",
        f"📈 Model {prob_pct:.1f}%",
        f"🔎 Why: {rationale}",
        f"🏆 League: {league}",
    ]
    return "\n".join(parts)

def _pretty_score(m: Dict[str,Any]) -> str:
    gh = (m.get("goals") or {}).get("home") or 0
    ga = (m.get("goals") or {}).get("away") or 0
    return f"{gh}-{ga}"

def _league_name(m: Dict[str,Any]) -> Tuple[int,str]:
    lg = (m.get("league") or {}) or {}
    league_id = int(lg.get("id") or 0)
    league = f"{lg.get('country','')} - {lg.get('name','')}".strip(" -")
    return league_id, league

def _teams(m: Dict[str,Any]) -> Tuple[str,str]:
    t = (m.get("teams") or {}) or {}
    return (t.get("home",{}).get("name",""), t.get("away",{}).get("name",""))

# ──────────────────────────────────────────────────────────────────────────────
# Production scan (PURE MODEL, no odds filters)
def production_scan() -> Tuple[int,int]:
    matches = fetch_live_matches()
    live_seen = len(matches)
    if live_seen == 0:
        logging.info("[PROD] no live matches")
        return 0, 0

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
                minute = int(feat.get("minute", 0))

                # score all heads
                raw: List[Tuple[str,str,float,str]] = []
                for code, mdl in models.items():
                    raw.extend(_mk_suggestions_for_code(code, mdl, feat))

                # filter only by per‑head probability
                filtered = []
                for market, suggestion, p, head in raw:
                    if p >= _min_prob_for_head(head):
                        filtered.append((market, suggestion, p, head))

                if not filtered:
                    continue

                # pick top‑K (by prob)
                filtered.sort(key=lambda x: x[2], reverse=True)
                filtered = filtered[:max(1, int(2 if MAX_TIPS_PER_SCAN else 2))]

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
                    """, (fid, league_id, league, home, away, market, suggestion, float(min(99.0, prob*100.0)), score_txt, minute, now))

                    rationale = _rationale_from_feat(feat)
                    msg = _format_tip_message_pure(
                        league=league, home=home, away=away,
                        minute=minute, score_txt=score_txt,
                        suggestion=suggestion, prob=prob, rationale=rationale
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
# Harvest
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
        "stat": {"xg_h": feat.get("xg_h",0), "xg_a": feat.get("xg_a",0),
                 "sot_h": feat.get("sot_h",0), "sot_a": feat.get("sot_a",0),
                 "cor_h": feat.get("cor_h",0), "cor_a": feat.get("cor_a",0),
                 "pos_h": feat.get("pos_h",0), "pos_a": feat.get("pos_a",0),
                 "red_h": feat.get("red_h",0), "red_a": feat.get("red_a",0)}
    }
    now = int(time.time())
    with db_conn() as conn:
        conn.execute("INSERT INTO tip_snapshots(match_id, created_ts, payload) VALUES (%s,%s,%s) ON CONFLICT (match_id, created_ts) DO UPDATE SET payload=EXCLUDED.payload",
                     (fid, now, json.dumps(snapshot)[:200000]))
        conn.execute("""
            INSERT INTO tips(match_id,league_id,league,home,away,market,suggestion,confidence,score_at_tip,minute,created_ts,sent_ok)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (fid, league_id, league, home, away, "HARVEST", "HARVEST", 0.0, f"{gh}-{ga}", minute, now, 0))

def harvest_scan() -> Tuple[int, int]:
    matches = fetch_live_matches()
    live_seen = len(matches)
    if live_seen == 0:
        logging.info("[HARVEST] no live matches"); return 0, 0
    saved = 0; now_ts = int(time.time())
    with db_conn() as conn:
        for m in matches:
            try:
                fid = int((m.get("fixture", {}) or {}).get("id") or 0)
                if not fid: continue
                cutoff = now_ts - (DUP_COOLDOWN_MIN * 60)
                dup = conn.execute("SELECT 1 FROM tips WHERE match_id=%s AND created_ts>=%s LIMIT 1", (fid, cutoff)).fetchone()
                if dup: continue
                feat = extract_features(m)
                save_snapshot_from_match(m, feat)
                saved += 1
                if MAX_TIPS_PER_SCAN and saved >= MAX_TIPS_PER_SCAN: break
            except Exception as e:
                logging.exception(f"[HARVEST] failure on match: {e}")
                continue
    logging.info(f"[HARVEST] saved={saved} live_seen={live_seen}")
    return saved, live_seen

# ──────────────────────────────────────────────────────────────────────────────
# MOTD (Match of the Day)
def _motd_row(since_ts: int) -> Optional[Tuple]:
    with db_conn() as conn:
        row = conn.execute("""
            SELECT match_id, league_id, league, home, away, market, suggestion, confidence, score_at_tip, minute, created_ts
            FROM tips
            WHERE created_ts >= %s AND sent_ok=1 AND suggestion <> 'HARVEST'
            ORDER BY confidence DESC, created_ts DESC
            LIMIT 1
        """, (since_ts,)).fetchone()
        return row

def send_motd_if_needed() -> Optional[Dict[str,Any]]:
    if not MOTD_ENABLE:
        return None
    # avoid duplicate within a day (Europe/Berlin date)
    now = time.time()
    berlin = ZoneInfo("Europe/Berlin")
    ymd = time.strftime("%Y-%m-%d", time.localtime(now))
    last = get_setting("motd_last_ymd", "")
    if last == ymd:
        return None

    since_ts = int(now - MOTD_LOOKBACK_HRS*3600)
    row = _motd_row(since_ts)
    if not row:
        return None

    (match_id, league_id, league, home, away, market, suggestion, confidence, score_at_tip, minute, created_ts) = row
    msg = (
        "🏅 Match of the Day\n"
        f"Match: {home} vs {away}\n"
        f"⏰ Minute at tip: {minute}' | Score: {score_at_tip}\n"
        f"Tip: {suggestion}\n"
        f"📈 Model {float(confidence):.1f}%\n"
        f"🏆 League: {league}"
    )
    send_telegram(msg)
    set_setting("motd_last_ymd", ymd)
    return {
        "match_id": int(match_id), "league": league, "home": home, "away": away,
        "suggestion": suggestion, "confidence": float(confidence), "created_ts": int(created_ts)
    }

# ──────────────────────────────────────────────────────────────────────────────
# Training + digest (unchanged logic; calls external script)
def nightly_digest_job():
    try:
        now = int(time.time()); day_ago = now - 24*3600
        with db_conn() as conn:
            snap_total = conn.execute("SELECT COUNT(*) FROM tip_snapshots").fetchone()[0]
            tips_total = conn.execute("SELECT COUNT(*) FROM tips").fetchone()[0]
            res_total  = conn.execute("SELECT COUNT(*) FROM match_results").fetchone()[0]
        msg = (
            "📊 <b>Robi Nightly Digest</b>\n"
            f"Snapshots: {snap_total}\nTips: {tips_total}\nFinals: {res_total}\n"
            "Models retrain at 03:00 CEST daily."
        )
        logging.info("[DIGEST]\n%s", msg.replace("<b>","").replace("</b>",""))
        send_telegram(msg)
        return True
    except Exception as e:
        logging.exception("[DIGEST] failed: %s", e)
        return False

def retrain_models_job():
    if not TRAIN_ENABLE:
        logging.info("[TRAIN] skipped (TRAIN_ENABLE=0)")
        return {"ok": False, "skipped": True}
    import subprocess, os
    env = os.environ.copy()
    env["DATABASE_URL"] = DATABASE_URL
    cmd = ["python", "-u", "train_models.py", "--min-minute", str(TRAIN_MIN_MINUTE)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)
        out = (proc.stdout or "").strip(); err = (proc.stderr or "").strip()
        logging.info(f"[TRAIN] returncode={proc.returncode}\nstdout:\n{out}\nstderr:\n{err}")
        ok = (proc.returncode == 0)
        send_telegram(("✅" if ok else "❌") + " Nightly training " + ("OK" if ok else "failed"))
        return {"ok": ok, "code": proc.returncode, "stdout": out[-2000:], "stderr": err[-1000:]}
    except subprocess.TimeoutExpired:
        logging.error("[TRAIN] timed out")
        send_telegram("❌ Nightly training timed out.")
        return {"ok": False, "timeout": True}
    except Exception as e:
        logging.exception(f"[TRAIN] exception: {e}")
        send_telegram(f"❌ Nightly training exception: {e}")
        return {"ok": False, "error": str(e)}

# ──────────────────────────────────────────────────────────────────────────────
# Routes
def _require_api_key():
    key = request.headers.get("X-API-Key") or request.args.get("key")
    if not ADMIN_API_KEY or key != ADMIN_API_KEY: abort(401)

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
    return f"🤖 Robi Superbrain is active ({mode}, pure-model) · DB=Postgres · MIN_PROB={MIN_PROB:.2f}"

@app.route("/healthz")
def healthz():
    db_ok = True
    try:
        with db_conn() as conn:
            conn.execute("SELECT 1")
    except Exception:
        db_ok = False
    return jsonify({"ok": True, "mode": ("HARVEST" if HARVEST_MODE else "PRODUCTION"), "ai_only": True, "db_ok": db_ok})

@app.route("/predict/models")
def predict_models_route():
    _require_api_key()
    models = _list_models("model_v2:")
    return jsonify({"count": len(models), "codes": sorted(models.keys())})

@app.route("/predict/scan")
def predict_scan_route():
    _require_api_key()
    saved, live = production_scan()
    return jsonify({"ok": True, "live_seen": live, "tips_saved": saved, "min_prob": MIN_PROB})

@app.route("/train", methods=["POST", "GET"])
def train_route():
    _require_api_key()
    return jsonify(retrain_models_job())

@app.route("/motd")
def motd_route():
    # no admin key required for convenience; add if you prefer
    lookback = int(request.args.get("hours", MOTD_LOOKBACK_HRS))
    since_ts = int(time.time()) - lookback*3600
    row = _motd_row(since_ts)
    if not row:
        return jsonify({"ok": True, "motd": None})
    (match_id, league_id, league, home, away, market, suggestion, confidence, score_at_tip, minute, created_ts) = row
    return jsonify({
        "ok": True,
        "motd": {
            "match_id": int(match_id), "league": league, "home": home, "away": away,
            "market": market, "suggestion": suggestion, "confidence": float(confidence),
            "score_at_tip": score_at_tip, "minute": int(minute), "created_ts": int(created_ts)
        }
    })

# ──────────────────────────────────────────────────────────────────────────────
# Entrypoint / Scheduler
if __name__ == "__main__":
    if not API_KEY: logging.error("API_KEY is not set — live fetch will return 0 matches.")
    if not ADMIN_API_KEY: logging.error("ADMIN_API_KEY is not set — admin endpoints will 401.")
    init_db()

    scheduler = BackgroundScheduler()

    if HARVEST_MODE:
        scheduler.add_job(harvest_scan, CronTrigger(minute="*/10", timezone=ZoneInfo("Europe/Berlin")), id="harvest", replace_existing=True)
        logging.info("⛏️  Running in HARVEST mode.")
    else:
        scheduler.add_job(production_scan, CronTrigger(minute="*/5", timezone=ZoneInfo("Europe/Berlin")), id="production_scan", replace_existing=True)
        logging.info("🎯 Running in PRODUCTION mode (pure model).")

    scheduler.add_job(retrain_models_job, CronTrigger(hour=3, minute=0, timezone=ZoneInfo("Europe/Berlin")),
                      id="train", replace_existing=True, misfire_grace_time=3600, coalesce=True)
    scheduler.add_job(nightly_digest_job, CronTrigger(hour=3, minute=2, timezone=ZoneInfo("Europe/Berlin")),
                      id="digest", replace_existing=True, misfire_grace_time=3600, coalesce=True)

    if MOTD_ENABLE:
        scheduler.add_job(send_motd_if_needed, CronTrigger(hour=MOTD_HOUR_LOCAL, minute=0, timezone=ZoneInfo("Europe/Berlin")),
                          id="motd", replace_existing=True, misfire_grace_time=3600, coalesce=True)
        logging.info("🏅 MOTD scheduled at %02d:00 Europe/Berlin", MOTD_HOUR_LOCAL)

    scheduler.start()
    logging.info("⏱️ Scheduler started (HARVEST_MODE=%s)", HARVEST_MODE)

    port = int(os.getenv("PORT", 5000))
    logging.info("✅ Robi Superbrain started.")
    app.run(host="0.0.0.0", port=port)
