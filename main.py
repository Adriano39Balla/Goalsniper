# path: main.py
import os
import re
import json
import time
import math
import logging
import requests
import psycopg2
from psycopg2 import pool
import subprocess, shlex
from zoneinfo import ZoneInfo
from typing import List, Dict, Any, Optional, Tuple
from flask import Flask, jsonify, request, abort
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
app = Flask(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Env
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")
API_KEY            = os.getenv("API_KEY")
ADMIN_API_KEY      = os.getenv("ADMIN_API_KEY")

HARVEST_MODE       = os.getenv("HARVEST_MODE", "1") not in ("0","false","False","no","NO")
MAX_TIPS_PER_SCAN  = int(os.getenv("MAX_TIPS_PER_SCAN", "25"))
DUP_COOLDOWN_MIN   = int(os.getenv("DUP_COOLDOWN_MIN", "20"))

# AI / policy
ONLY_MODEL_MODE    = os.getenv("ONLY_MODEL_MODE", "1") not in ("0","false","False","no","NO")
MIN_PROB_DEFAULT   = max(0.0, min(0.99, float(os.getenv("MIN_PROB", "0.55"))))
MIN_QUOTA          = float(os.getenv("MIN_QUOTA", "1.05"))

# Per‑head overrides (so BTTS can punch through even when global MIN_PROB is high)
BTTS_MIN_PROB      = float(os.getenv("BTTS_MIN_PROB", str(MIN_PROB_DEFAULT)))

# Training
TRAIN_ENABLE       = os.getenv("TRAIN_ENABLE", "1") not in ("0","false","False","no","NO")
TRAIN_MIN_MINUTE   = int(os.getenv("TRAIN_MIN_MINUTE", "15"))

# Stats requirements
REQUIRE_STATS_AFTER_MINUTE = int(os.getenv("REQUIRE_STATS_AFTER_MINUTE", "35"))
REQUIRE_DATA_FIELDS        = int(os.getenv("REQUIRE_DATA_FIELDS", "2"))

# Odds gates
MIN_DEC_ODDS       = float(os.getenv("MIN_DEC_ODDS", "1.5"))
MINUTE_MIN         = int(os.getenv("MINUTE_MIN", "10"))
MINUTE_MAX         = int(os.getenv("MINUTE_MAX", "85"))
REQ_AFTER_MIN      = int(os.getenv("REQ_AFTER_MIN", str(REQUIRE_STATS_AFTER_MINUTE)))
REQ_MIN_FIELDS     = int(os.getenv("REQ_MIN_FIELDS", str(REQUIRE_DATA_FIELDS)))
REQ_MIN_SOT_SUM    = int(os.getenv("REQ_MIN_SOT_SUM", "3"))
REQ_MIN_XG_SUM     = float(os.getenv("REQ_MIN_XG_SUM", "1.1"))

# Odds availability rule:
# require odds for non‑BTTS markets to reduce noise, but allow BTTS without odds so BTTS tips actually appear.
REQUIRE_ODDS_FOR_NON_BTTS = os.getenv("REQUIRE_ODDS_FOR_NON_BTTS", "1") not in ("0","false","False","no","NO")

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise SystemExit("DATABASE_URL is required (Postgres).")

# ──────────────────────────────────────────────────────────────────────────────
# Requests session
session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], respect_retry_after_header=True)
session.mount("https://", HTTPAdapter(max_retries=retries))
session.mount("http://", HTTPAdapter(max_retries=retries))

BASE_URL         = "https://v3.football.api-sports.io"
FOOTBALL_API_URL = f"{BASE_URL}/fixtures"
HEADERS          = {"x-apisports-key": API_KEY, "Accept": "application/json"}
INPLAY_STATUSES  = {"1H","HT","2H","ET","BT","P"}

STATS_CACHE: Dict[int, Tuple[float, list]] = {}
EVENTS_CACHE: Dict[int, Tuple[float, list]] = {}
ODDS_CACHE: Dict[int, Tuple[float, Dict[str, Any]]] = {}

# ──────────────────────────────────────────────────────────────────────────────
# DB pooling + helpers
def _normalize_db_url(db_url: str) -> str:
    if db_url.startswith("postgres://"):
        db_url = "postgresql://" + db_url[len("postgres://"):]
    if "sslmode=" not in db_url:
        db_url = db_url + ("&" if "?" in db_url else "?") + "sslmode=require"
    return db_url

pg_pool: Optional[pool.SimpleConnectionPool] = None

class PgConn:
    def __enter__(self):
        assert pg_pool is not None, "pg_pool not initialized"
        self.conn = pg_pool.getconn()
        self.conn.autocommit = True
        self.cur = self.conn.cursor()
        return self
    def __exit__(self, exc_type, exc, tb):
        try:
            if self.cur: self.cur.close()
        finally:
            if self.conn and pg_pool: pg_pool.putconn(self.conn)
    def execute(self, sql: str, params: tuple|list=()):
        self.cur.execute(sql, params or ())
        return self.cur

def db_conn() -> PgConn: return PgConn()

def init_pool_and_db():
    global pg_pool
    if pg_pool is None:
        dsn = _normalize_db_url(DATABASE_URL)
        pg_pool = pool.SimpleConnectionPool(minconn=1, maxconn=10, dsn=dsn)
    init_db()

def init_db():
    # tables + indexes + non‑breaking migrations
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
            -- new analytics cols (nullable for backward compat)
            head      TEXT,
            baseline  DOUBLE PRECISION,
            lift      DOUBLE PRECISION,
            quota     DOUBLE PRECISION,
            offered_odds DOUBLE PRECISION,
            PRIMARY KEY (match_id, created_ts)
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS tip_snapshots (match_id BIGINT, created_ts BIGINT, payload TEXT, PRIMARY KEY (match_id, created_ts))""")
        conn.execute("""CREATE TABLE IF NOT EXISTS feedback (id SERIAL PRIMARY KEY, match_id BIGINT UNIQUE, verdict INTEGER, created_ts BIGINT)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS match_results (match_id BIGINT PRIMARY KEY, final_goals_h INTEGER, final_goals_a INTEGER, btts_yes INTEGER, updated_ts BIGINT)""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tip_snaps_created ON tip_snapshots(created_ts)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tips_match ON tips(match_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tips_recent ON tips(created_ts)")
        # Defensive migrations if old tips table exists without new cols
        for col, ddl in [
            ("head",         "ALTER TABLE tips ADD COLUMN IF NOT EXISTS head TEXT"),
            ("baseline",     "ALTER TABLE tips ADD COLUMN IF NOT EXISTS baseline DOUBLE PRECISION"),
            ("lift",         "ALTER TABLE tips ADD COLUMN IF NOT EXISTS lift DOUBLE PRECISION"),
            ("quota",        "ALTER TABLE tips ADD COLUMN IF NOT EXISTS quota DOUBLE PRECISION"),
            ("offered_odds", "ALTER TABLE tips ADD COLUMN IF NOT EXISTS offered_odds DOUBLE PRECISION"),
        ]:
            conn.execute(ddl)

def set_setting(key: str, value: str):
    with db_conn() as conn:
        conn.execute("""INSERT INTO settings(key,value) VALUES(%s,%s)
                        ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value""", (key, value))

def get_setting(key: str, default: Optional[str] = None) -> Optional[str]:
    with db_conn() as conn:
        row = conn.execute("SELECT value FROM settings WHERE key=%s", (key,)).fetchone()
        return row[0] if row else default

# ──────────────────────────────────────────────────────────────────────────────
# API calls
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
    if fixture_id in STATS_CACHE and now - STATS_CACHE[fixture_id][0] < 90:
        return STATS_CACHE[fixture_id][1]
    js = _api_get(f"{FOOTBALL_API_URL}/statistics", {"fixture": fixture_id})
    stats = js.get("response", []) if isinstance(js, dict) else []
    STATS_CACHE[fixture_id] = (now, stats)
    return stats

def fetch_match_events(fixture_id: int):
    now = time.time()
    if fixture_id in EVENTS_CACHE and now - EVENTS_CACHE[fixture_id][0] < 90:
        return EVENTS_CACHE[fixture_id][1]
    js = _api_get(f"{FOOTBALL_API_URL}/events", {"fixture": fixture_id})
    evs = js.get("response", []) if isinstance(js, dict) else []
    EVENTS_CACHE[fixture_id] = (now, evs)
    return evs

def fetch_live_matches() -> List[Dict[str, Any]]:
    js = _api_get(FOOTBALL_API_URL, {"live": "all"})
    if not isinstance(js, dict): return []
    matches = js.get("response", []) or []
    out = []
    for m in matches:
        status = (m.get("fixture", {}) or {}).get("status", {}) or {}
        elapsed = status.get("elapsed")
        short = (status.get("short") or "").upper()
        if elapsed is None or elapsed > 90:  # cap at 90; ET will show via short
            continue
        if short not in INPLAY_STATUSES:
            continue
        fid = (m.get("fixture", {}) or {}).get("id")
        m["statistics"] = fetch_match_stats(fid)
        m["events"] = fetch_match_events(fid)
        out.append(m)
    return out

# ──────────────────────────────────────────────────────────────────────────────
# Odds parsing
def _odds_to_prob(odd: Any) -> float:
    try:
        o = float(odd)
        return 1.0 / o if o > 0 else 0.0
    except Exception:
        return 0.0

def _normalize_three(p1: float, p2: float, p3: float) -> Tuple[float,float,float]:
    s = max(1e-9, p1 + p2 + p3)
    return p1/s, p2/s, p3/s

def _parse_odds_payload(js: Dict[str, Any]) -> Dict[str, Any]:
    books = (js or {}).get("response", []) or []
    if not books: return {}
    node = books[0]
    bks = node.get("bookmakers", books) or []
    if not bks: return {}
    bets = (bks[0] or {}).get("bets", []) or []
    out = {"hda": {}, "hda_odds": {}, "ou": {}, "ou_odds": {}}

    def _to_float(x):
        try: return float(x)
        except Exception: return None

    # 1X2
    mw = next((b for b in bets if "match winner" in (b.get("name","").lower())), None)
    if mw:
        vals = { (v.get("value","") or "").strip().lower(): v for v in (mw.get("values") or []) }
        raw_home = _to_float((vals.get("home") or {}).get("odd"))
        raw_draw = _to_float((vals.get("draw") or {}).get("odd"))
        raw_away = _to_float((vals.get("away") or {}).get("odd"))
        pH = _odds_to_prob(raw_home) if raw_home else 0.0
        pD = _odds_to_prob(raw_draw) if raw_draw else 0.0
        pA = _odds_to_prob(raw_away) if raw_away else 0.0
        pH, pD, pA = _normalize_three(pH, pD, pA)
        out["hda"] = {"home": pH, "draw": pD, "away": pA}
        out["hda_odds"] = {"home": raw_home, "draw": raw_draw, "away": raw_away}

    # Over/Under
    ou = next((b for b in bets if "over/under" in (b.get("name","").lower())), None)
    if ou:
        for v in (ou.get("values") or []):
            val = (v.get("value") or "").strip().lower()
            odd = _to_float(v.get("odd"))
            if not odd: continue
            is_over = val.startswith("over ")
            is_under = val.startswith("under ")
            if not (is_over or is_under): continue
            try:
                th = float(val.split()[1])
            except Exception:
                continue
            d = out["ou"].setdefault(th, {"over": 0.0, "under": 0.0})
            d2 = out["ou_odds"].setdefault(th, {"over": None, "under": None})
            if is_over:
                d["over"] = _odds_to_prob(odd); d2["over"] = odd
            else:
                d["under"] = _odds_to_prob(odd); d2["under"] = odd
        # normalize each threshold
        for th, d in list(out["ou"].items()):
            s = max(1e-9, d.get("over",0.0) + d.get("under",0.0))
            out["ou"][th] = {"over": d.get("over",0.0)/s, "under": d.get("under",0.0)/s}

    return out

def fetch_fixture_odds(fixture_id: int) -> Dict[str, Any]:
    now = time.time()
    if fixture_id in ODDS_CACHE and now - ODDS_CACHE[fixture_id][0] < 90:
        return ODDS_CACHE[fixture_id][1]
    js1 = _api_get(f"{BASE_URL}/odds", {"fixture": fixture_id})
    odds = _parse_odds_payload(js1 if isinstance(js1, dict) else {})
    if not odds:
        js2 = _api_get(f"{BASE_URL}/odds/live", {"fixture": fixture_id})
        odds = _parse_odds_payload(js2 if isinstance(js2, dict) else {})
    ODDS_CACHE[fixture_id] = (now, odds or {})
    return odds or {}

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
# Models & policy
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
            if code.lower() in ("o05","o5"):  # never use O0.5
                continue
            out[code] = json.loads(v)
        except Exception:
            continue
    return out

# Baselines from odds (fallback to empirical prevalence)
def _ou_label(code: str) -> Optional[float]:
    m = OU_RE.match(code)
    if not m: return None
    return int(m.group(1)) / 10.0

def _head_prevalence(head: str) -> float:
    raw = get_setting("model_coeffs")
    if not raw:
        return 0.5
    try:
        js = json.loads(raw)
        m = (js.get("metrics") or {}).get(head) or {}
        p = float(m.get("prevalence", 0.5))
        return max(0.05, min(0.95, p))
    except Exception:
        return 0.5

def _baseline_from_odds(head: str, suggestion: str, odds: Dict[str,Any]) -> Optional[float]:
    s = (suggestion or "").lower()
    hda = (odds or {}).get("hda") or {}
    ou  = (odds or {}).get("ou") or {}
    if head in ("WIN_HOME","DRAW","WIN_AWAY") and hda:
        if "home" in hda and "home win" in s: return float(hda["home"])
        if "draw" in hda and s == "draw": return float(hda["draw"])
        if "away" in hda and "away win" in s: return float(hda["away"])
    th = _ou_label(head)
    if th is not None and th in ou:
        d = ou[th]
        if s.startswith("over "): return float(d.get("over", 0.0))
        if s.startswith("under "): return float(d.get("under", 0.0))
    if head == "BTTS" and hda:
        return 0.5  # weak fallback; better than none
    return None

def _clamp_prob(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))

def _safe_odds(p: float) -> float:
    p = _clamp_prob(p, 1e-6, 1-1e-6)
    return p / (1.0 - p)

def _quota(p: float, b: float) -> float:
    p = _clamp_prob(p, 0.03, 0.97)
    b = _clamp_prob(b, 0.05, 0.95)
    return _safe_odds(p) / _safe_odds(b)

def _stats_coverage_ok(feat: Dict[str, float], min_fields: int) -> bool:
    fields = [
        feat.get("xg_sum", 0.0),
        feat.get("sot_sum", 0.0),
        feat.get("cor_sum", 0.0),
        max(feat.get("pos_h", 0.0), feat.get("pos_a", 0.0)),
    ]
    nonzero = sum(1 for v in fields if (v or 0) > 0)
    return nonzero >= max(0, int(min_fields))

def _offered_odds(head: str, suggestion: str, odds: Dict[str,Any]) -> Optional[float]:
    s = (suggestion or "").strip().lower()
    if head in ("WIN_HOME","DRAW","WIN_AWAY"):
        m = (odds or {}).get("hda_odds") or {}
        if "home win" in s: return m.get("home")
        if s == "draw":     return m.get("draw")
        if "away win" in s: return m.get("away")
        return None
    th = _ou_label(head)
    if th is not None:
        m = (odds or {}).get("ou_odds") or {}
        d = m.get(th) or {}
        if s.startswith("over "):  return d.get("over")
        if s.startswith("under "): return d.get("under")
        return None
    if head == "BTTS":
        return None
    return None

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

def _quality_gate(feat: Dict[str,float]) -> bool:
    minute = int(feat.get("minute", 0))
    if minute < MINUTE_MIN or minute > MINUTE_MAX:
        return False
    if minute >= REQ_AFTER_MIN:
        if not _stats_coverage_ok(feat, REQ_MIN_FIELDS):
            return False
        if (feat.get("sot_sum", 0.0) or 0.0) < REQ_MIN_SOT_SUM and (feat.get("xg_sum", 0.0) or 0.0) < REQ_MIN_XG_SUM:
            return False
    return True

def _min_prob_for_head(head: str) -> float:
    if head == "BTTS":
        return BTTS_MIN_PROB
    return MIN_PROB_DEFAULT

def _apply_filter_with_odds(
    suggestions: List[Tuple[str,str,float,str]],
    minute: int,
    feat: Dict[str,float],
    odds: Dict[str,Any],
    min_quota: float,
    top_k: int = 2,
    min_fields_after_minute: Tuple[int,int] = (REQUIRE_STATS_AFTER_MINUTE, REQUIRE_DATA_FIELDS)
) -> List[Tuple[str,str,float,str,float,float,float,Optional[float]]]:
    """
    Returns: (market, suggestion, p, head, base, lift, edge_quota, offered_odds)
    """
    out = []
    req_minute, req_fields = min_fields_after_minute

    for market, sugg, p, head in suggestions:
        # Stats coverage gate after a given minute
        if req_minute and minute >= req_minute and not _stats_coverage_ok(feat, req_fields):
            continue

        offered = _offered_odds(head, sugg, odds)
        # If we have odds quoted, enforce a minimum price
        if offered is not None and offered < MIN_DEC_ODDS:
            continue

        # If odds are required for non‑BTTS, enforce presence
        if REQUIRE_ODDS_FOR_NON_BTTS and head != "BTTS" and offered is None:
            continue

        base = _baseline_from_odds(head, sugg, odds)
        if base is None:
            base = _head_prevalence(head)

        lift   = p - base
        edge_q = _quota(p, base)

        # Per‑head probability floors (BTTS override lives here)
        min_prob = _min_prob_for_head(head)

        if p >= min_prob and edge_q >= min_quota:
            out.append((market, sugg, p, head, base, lift, edge_q, offered))

    out.sort(key=lambda x: (x[6], x[5], x[2]), reverse=True)  # quota, lift, prob
    return out[:max(1, int(top_k))]

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

def _format_tip_message(
    league: str, home: str, away: str, minute: int, score_txt: str,
    suggestion: str, prob: float, baseline: float, lift: float, quota: float, rationale: str,
    offered_odds: Optional[float] = None
) -> str:
    prob_pct = min(99.0, max(0.0, prob * 100.0))
    base_pct = min(99.0, max(0.0, baseline * 100.0))
    parts = [
        "⚽️ New Tip!",
        f"Match: {home} vs {away}",
        f"⏰ Minute: {minute}' | Score: {score_txt}",
        f"Tip: {suggestion}",
        f"📈 Model {prob_pct:.1f}% | Odds‑implied {base_pct:.1f}% | Edge +{lift*100:.1f} pp | Quota ×{quota:.2f}",
    ]
    if offered_odds:
        parts.append(f"💸 Odds {offered_odds:.2f}")
    parts.append(f"🔎 Why: {rationale}")
    parts.append(f"🏆 League: {league}")
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
# Nightly training + digest + MOTD
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
        return True
    except Exception as e:
        logging.exception("[DIGEST] failed: %s", e)
        return False

def _top_feature_strings(model: Dict[str,Any], k: int = 8) -> str:
    try:
        ws = model.get("weights", {}) or {}
        items = sorted(ws.items(), key=lambda kv: abs(float(kv[1] or 0.0)), reverse=True)[:k]
        parts = [f"{name}={'+' if float(w)>=0 else ''}{float(w):.3f}" for name,w in items]
        icpt = float(model.get("intercept", 0.0))
        parts.append(f"intercept={'+' if icpt>=0 else ''}{icpt:.3f}")
        return " | ".join(parts)
    except Exception:
        return "(no weights)"

def _metrics_blob_for_telegram() -> str:
    raw = get_setting("model_coeffs")
    if not raw: return "{}"
    try:
        js = json.loads(raw)
        keep = {"ok": True, "trained_at_utc": js.get("trained_at_utc"), "metrics": js.get("metrics", {})}
        return json.dumps(keep, ensure_ascii=False)
    except Exception:
        return raw[:900]

def retrain_models_job():
    if not TRAIN_ENABLE:
        logging.info("[TRAIN] skipped (TRAIN_ENABLE=0)")
        return {"ok": False, "skipped": True, "reason": "TRAIN_ENABLE=0"}

    env = os.environ.copy()
    env["DATABASE_URL"] = DATABASE_URL
    cmd = ["python", "-u", "train_models.py", "--min-minute", str(TRAIN_MIN_MINUTE)]
    logging.info("[TRAIN] starting train_models.py")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        logging.info(f"[TRAIN] returncode={proc.returncode}\nstdout:\n{out}\nstderr:\n{err}")

        models = _list_models("model_v2:")
        chosen = None
        for key in ("BTTS","O25","O15","WIN_HOME","DRAW","WIN_AWAY"):
            if key in models: chosen = key; break
        top_line = f"[{chosen}] top | {_top_feature_strings(models[chosen], 8)}" if chosen else ""
        metrics_json = _metrics_blob_for_telegram()
        set_setting("model_metrics_latest", metrics_json)

        ok = (proc.returncode == 0)
        emoji = "✅" if ok else "❌"
        msg = f"{emoji} Nightly training {'OK' if ok else 'failed'} (rc={proc.returncode})\n{top_line}\nSaved model_metrics_latest in settings.\n{metrics_json}"
        send_telegram(msg)

        return {"ok": ok, "code": proc.returncode, "stdout": out[-2000:], "stderr": err[-1000:]}
    except subprocess.TimeoutExpired:
        logging.error("[TRAIN] timed out")
        send_telegram("❌ Nightly training timed out.")
        return {"ok": False, "timeout": True}
    except Exception as e:
        logging.exception(f"[TRAIN] exception: {e}")
        send_telegram(f"❌ Nightly training exception: {e}")
        return {"ok": False, "error": str(e)}

def motd_job():
    """Pick the single best tip from the last 24h by quota, then by confidence, and send it."""
    try:
        now = int(time.time()); from_ts = now - 24*3600
        with db_conn() as conn:
            rows = conn.execute("""
                SELECT league, home, away, market, suggestion, confidence, minute, score_at_tip,
                       COALESCE(quota, 0) AS quota, COALESCE(lift, 0) AS lift, COALESCE(baseline, 0) AS baseline,
                       COALESCE(offered_odds, NULL) AS offered_odds, created_ts
                FROM tips
                WHERE created_ts >= %s AND suggestion <> 'HARVEST'
                ORDER BY quota DESC NULLS LAST, confidence DESC NULLS LAST, created_ts DESC
                LIMIT 1
            """, (from_ts,)).fetchone()
        if not rows:
            send_telegram("🏆 Match of the Day: no qualifying tips in the last 24h.")
            return False

        league, home, away, market, suggestion, conf, minute, score, quota, lift, baseline, offered, _ = rows
        msg = (
            "🏆 <b>Match of the Day</b>\n"
            f"Match: {home} vs {away}\n"
            f"Tip: {suggestion} ({market})\n"
            f"⏰ At {minute}' | Score then: {score}\n"
            f"📈 Model {float(conf or 0):.1f}% | Odds‑implied {baseline*100:.1f}% | Edge +{lift*100:.1f} pp | Quota ×{float(quota or 0):.2f}"
        )
        if offered:
            msg += f"\n💸 Odds {float(offered):.2f}"
        msg += f"\n🏁 League: {league}"
        send_telegram(msg)
        return True
    except Exception as e:
        logging.exception("[MOTD] failed: %s", e)
        return False

# ──────────────────────────────────────────────────────────────────────────────
# Production scan
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

                if not _quality_gate(feat):
                    continue

                minute = int(feat.get("minute", 0))
                odds = fetch_fixture_odds(fid)

                raw: List[Tuple[str,str,float,str]] = []
                for code, mdl in models.items():
                    raw.extend(_mk_suggestions_for_code(code, mdl, feat))

                filtered = _apply_filter_with_odds(
                    raw, minute, feat, odds,
                    min_quota=MIN_QUOTA, top_k=2
                )
                if not filtered: continue

                league_id, league = _league_name(m)
                home, away = _teams(m)
                score_txt = _pretty_score(m)

                for market, suggestion, prob, head, base, lift, q, offered in filtered:
                    cutoff = now_ts - (DUP_COOLDOWN_MIN * 60)
                    dup = conn.execute(
                        "SELECT 1 FROM tips WHERE match_id=%s AND market=%s AND suggestion=%s AND created_ts>=%s LIMIT 1",
                        (fid, market, suggestion, cutoff)
                    ).fetchone()
                    if dup: continue

                    now = int(time.time())
                    conn.execute("""
                        INSERT INTO tips(match_id,league_id,league,home,away,market,suggestion,confidence,score_at_tip,minute,created_ts,sent_ok,head,baseline,lift,quota,offered_odds)
                        VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,1,%s,%s,%s,%s,%s)
                    """, (fid, league_id, league, home, away, market, suggestion,
                          float(min(99.0, prob*100.0)), score_txt, minute, now,
                          head, float(base), float(lift), float(q), float(offered) if (offered is not None) else None))

                    rationale = _rationale_from_feat(feat)
                    msg = _format_tip_message(
                        league=league, home=home, away=away,
                        minute=minute, score_txt=score_txt,
                        suggestion=suggestion, prob=prob, baseline=base, lift=lift, quota=q,
                        rationale=rationale, offered_odds=offered
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
# Harvest (data collection)
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
    ai_tag = "AI‑only" if ONLY_MODEL_MODE else "legacy"
    return f"🤖 Robi Superbrain is active ({mode}, {ai_tag}) · DB=Postgres · MIN_PROB={MIN_PROB_DEFAULT:.2f} · MIN_DEC_ODDS={MIN_DEC_ODDS:.2f} · BTTS_MIN={BTTS_MIN_PROB:.2f}"

@app.route("/healthz")
def healthz():
    db_ok = True
    try:
        with db_conn() as conn:
            conn.execute("SELECT 1")
    except Exception:
        db_ok = False
    return jsonify({"ok": True, "mode": ("HARVEST" if HARVEST_MODE else "PRODUCTION"), "ai_only": ONLY_MODEL_MODE, "db_ok": db_ok})

@app.route("/predict/models")
def predict_models_route():
    _require_api_key()
    models = _list_models("model_v2:")
    return jsonify({"count": len(models), "codes": sorted(models.keys())})

@app.route("/predict/scan")
def predict_scan_route():
    _require_api_key()
    saved, live = production_scan()
    return jsonify({"ok": True, "live_seen": live, "tips_saved": saved, "min_prob": MIN_PROB_DEFAULT, "btts_min_prob": BTTS_MIN_PROB, "min_dec_odds": MIN_DEC_ODDS})

@app.route("/train", methods=["POST", "GET"])
def train_route():
    _require_api_key()
    return jsonify(retrain_models_job())

# ──────────────────────────────────────────────────────────────────────────────
# Entrypoint / Scheduler
def start_scheduler():
    scheduler = BackgroundScheduler()
    if HARVEST_MODE:
        scheduler.add_job(harvest_scan, CronTrigger(minute="*/10", timezone=ZoneInfo("Europe/Berlin")), id="harvest", replace_existing=True)
        logging.info("⛏️  Running in HARVEST mode.")
    else:
        scheduler.add_job(production_scan, CronTrigger(minute="*/5", timezone=ZoneInfo("Europe/Berlin")), id="production_scan", replace_existing=True)
        logging.info("🎯 Running in PRODUCTION mode (AI-only).")

    scheduler.add_job(retrain_models_job, CronTrigger(hour=3, minute=0, timezone=ZoneInfo("Europe/Berlin")),
                      id="train", replace_existing=True, misfire_grace_time=3600, coalesce=True)
    scheduler.add_job(nightly_digest_job, CronTrigger(hour=3, minute=2, timezone=ZoneInfo("Europe/Berlin")),
                      id="digest", replace_existing=True, misfire_grace_time=3600, coalesce=True)
    # MOTD every morning 09:00 CEST
    scheduler.add_job(motd_job, CronTrigger(hour=9, minute=0, timezone=ZoneInfo("Europe/Berlin")),
                      id="motd", replace_existing=True, misfire_grace_time=3600, coalesce=True)

    scheduler.start()
    logging.info("⏱️ Scheduler started (HARVEST_MODE=%s)", HARVEST_MODE)

# Initialize pool + DB at import so routes work under Gunicorn
init_pool_and_db()

if __name__ == "__main__":
    start_scheduler()
    port = int(os.getenv("PORT", 5000))
    logging.info("✅ Robi Superbrain started.")
    app.run(host="0.0.0.0", port=port)
