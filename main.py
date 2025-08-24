# main.py
import os
import re
import json
import time
import math
import logging
import requests
import psycopg2
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

# Env
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")
API_KEY            = os.getenv("API_KEY")
ADMIN_API_KEY      = os.getenv("ADMIN_API_KEY")

HARVEST_MODE       = os.getenv("HARVEST_MODE", "1") not in ("0","false","False","no","NO")
MAX_TIPS_PER_SCAN  = int(os.getenv("MAX_TIPS_PER_SCAN", "25"))
DUP_COOLDOWN_MIN   = int(os.getenv("DUP_COOLDOWN_MIN", "20"))

# AI mode
ONLY_MODEL_MODE    = os.getenv("ONLY_MODEL_MODE", "1") not in ("0","false","False","no","NO")
MIN_PROB           = max(0.0, min(0.99, float(os.getenv("MIN_PROB", "0.55"))))
MIN_QUOTA          = float(os.getenv("MIN_QUOTA", "1.05"))  # why: realistic vs odds baseline

TRAIN_ENABLE       = os.getenv("TRAIN_ENABLE", "1") not in ("0","false","False","no","NO")
TRAIN_MIN_MINUTE   = int(os.getenv("TRAIN_MIN_MINUTE", "15"))

REQUIRE_STATS_AFTER_MINUTE = int(os.getenv("REQUIRE_STATS_AFTER_MINUTE", "35"))
REQUIRE_DATA_FIELDS        = int(os.getenv("REQUIRE_DATA_FIELDS", "2"))

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

STATS_CACHE: Dict[int, Tuple[float, list]] = {}
EVENTS_CACHE: Dict[int, Tuple[float, list]] = {}
ODDS_CACHE: Dict[int, Tuple[float, Dict[str, Any]]] = {}  # {fixture_id: (ts, odds_blob)}

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

# ──────────────────────────────────────────────────────────────────────────────
# API + features
def _api_get(url: str, params: dict, timeout: int = 15):
    if not API_KEY: return None
    try:
        res = session.get(url, headers=HEADERS, params=params, timeout=timeout)
        if not res.ok: return None
        return res.json()
    except Exception:
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
# Odds: parse & implied baselines
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
    books = []
    try:
        books = (js or {}).get("response", []) or []
    except Exception:
        pass
    if not books:
        return {}
    # API returns list on fixture or bookmakers list per fixture; handle both
    node = books[0]
    if "bookmakers" in node:
        bks = node["bookmakers"] or []
    else:
        bks = books  # already a bookmaker list
    if not bks:
        return {}
    bets = (bks[0] or {}).get("bets", []) or []
    out = {"hda": {}, "ou": {}}
    # 1X2
    mw = next((b for b in bets if "match winner" in (b.get("name","").lower())), None)
    if mw:
        vals = {v.get("value",""): v.get("odd") for v in (mw.get("values") or [])}
        pH, pD, pA = _odds_to_prob(vals.get("Home")), _odds_to_prob(vals.get("Draw")), _odds_to_prob(vals.get("Away"))
        pH, pD, pA = _normalize_three(pH, pD, pA)
        out["hda"] = {"home": pH, "draw": pD, "away": pA}
    # OU
    ou = next((b for b in bets if "over/under" in (b.get("name","").lower())), None)
    if ou:
        for v in (ou.get("values") or []):
            val = (v.get("value") or "")
            odd = _odds_to_prob(v.get("odd"))
            if val.lower().startswith("over "):
                try:
                    th = float(val.split()[1])
                except Exception:
                    continue
                d = out["ou"].setdefault(th, {"over": 0.0, "under": 0.0})
                d["over"] = odd
            elif val.lower().startswith("under "):
                try:
                    th = float(val.split()[1])
                except Exception:
                    continue
                d = out["ou"].setdefault(th, {"over": 0.0, "under": 0.0})
                d["under"] = odd
        # normalize pairs
        for th, d in list(out["ou"].items()):
            s = max(1e-9, d.get("over",0.0) + d.get("under",0.0))
            out["ou"][th] = {"over": d.get("over",0.0)/s, "under": d.get("under",0.0)/s}
    return out

def fetch_fixture_odds(fixture_id: int) -> Dict[str, Any]:
    now = time.time()
    if fixture_id in ODDS_CACHE:
        ts, data = ODDS_CACHE[fixture_id]
        if now - ts < 90: return data
    # try prematch endpoint first; fallback to live
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
            if (ev.get("type","").lower() == "card") and ("red" in (ev.get("detail","").lower())):
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
            if code.lower() in ("o05","o5"):
                continue
            out[code] = json.loads(v)
        except Exception:
            continue
    return out

# ──────────────────────────────────────────────────────────────────────────────
# Baseline from odds (fallback: empirical)
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
    if head == "BTTS" and hda:  # weak fallback: derive from price symmetry if no direct BTTS
        return 0.5
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

# coverage gate
def _stats_coverage_ok(feat: Dict[str, float], min_fields: int) -> bool:
    fields = [
        feat.get("xg_sum", 0.0),
        feat.get("sot_sum", 0.0),
        feat.get("cor_sum", 0.0),
        max(feat.get("pos_h", 0.0), feat.get("pos_a", 0.0)),
    ]
    nonzero = sum(1 for v in fields if (v or 0) > 0)
    return nonzero >= max(0, int(min_fields))

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
    # generic yes/no
    p = _score_prob(feat, mdl)
    out.append((code, f"{code}: Yes", p, code))
    out.append((code, f"{code}: No", 1.0 - p, code))
    return out

def _apply_filter_with_odds(
    suggestions: List[Tuple[str,str,float,str]],
    minute: int,
    feat: Dict[str,float],
    odds: Dict[str,Any],
    min_prob: float,
    min_quota: float,
    top_k: int = 2,
    min_fields_after_minute: Tuple[int,int] = (REQUIRE_STATS_AFTER_MINUTE, REQUIRE_DATA_FIELDS)
) -> List[Tuple[str,str,float,str,float,float,float]]:
    out = []
    req_minute, req_fields = min_fields_after_minute
    for market, sugg, p, head in suggestions:
        if req_minute and minute >= req_minute and not _stats_coverage_ok(feat, req_fields):
            continue
        base = _baseline_from_odds(head, sugg, odds)
        if base is None:
            base = _head_prevalence(head)  # last resort
        lift = p - base
        q = _quota(p, base)
        if p >= min_prob and q >= min_quota:
            out.append((market, sugg, p, head, base, lift, q))
    out.sort(key=lambda x: (x[6], x[5], x[2]), reverse=True)
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
    suggestion: str, prob: float, baseline: float, lift: float, quota: float, rationale: str
) -> str:
    prob_pct = min(99.0, max(0.0, prob * 100.0))
    base_pct = min(99.0, max(0.0, baseline * 100.0))
    display_quota = min(quota, 10.0)
    return (
        "⚽️ New Tip!\n"
        f"Match: {home} vs {away}\n"
        f"⏰ Minute: {minute}' | Score: {score_txt}\n"
        f"Tip: {suggestion}\n"
        f"📈 Model {prob_pct:.1f}% | Odds-implied {base_pct:.1f}% | Edge +{lift*100:.1f} pp | Quota ×{display_quota:.2f}\n"
        f"🔎 Why: {rationale}\n"
        f"🏆 League: {league}"
    )

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
# Nightly training + digest (unchanged)
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
    cmd = f"python -u train_models.py --db-url \"{os.getenv('DATABASE_URL','')}\" --min-minute {TRAIN_MIN_MINUTE}"
    logging.info(f"[TRAIN] starting: {cmd}")
    try:
        proc = subprocess.run(shlex.split(cmd), capture_output=True, text=True, timeout=900)
        out = (proc.stdout or "").strip(); err = (proc.stderr or "").strip()
        logging.info(f"[TRAIN] returncode={proc.returncode}\nstdout:\n{out}\nstderr:\n{err}")

        models = _list_models("model_v2:")
        chosen = None
        for key in ("BTTS","O25","O15","WIN_HOME","DRAW","WIN_AWAY"):
            if key in models: chosen = key; break
        top_line = f"[{chosen}] top | {_top_feature_strings(models[chosen], 8)}" if chosen else ""
        metrics_json = _metrics_blob_for_telegram()
        set_setting("model_metrics_latest", metrics_json)

        emoji = "✅" if proc.returncode == 0 else "❌"
        msg = f"{emoji} Nightly training {'OK' if proc.returncode == 0 else 'failed'}\n{top_line}\nSaved model_metrics_latest in settings.\n{metrics_json}"
        send_telegram(msg)

        return {"ok": proc.returncode == 0, "code": proc.returncode, "stdout": out[-2000:], "stderr": err[-1000:]}
    except subprocess.TimeoutExpired:
        logging.error("[TRAIN] timed out")
        send_telegram("❌ Nightly training timed out.")
        return {"ok": False, "timeout": True}
    except Exception as e:
        logging.exception(f"[TRAIN] exception: {e}")
        send_telegram(f"❌ Nightly training exception: {e}")
        return {"ok": False, "error": str(e)}

# ──────────────────────────────────────────────────────────────────────────────
# Production scan: model vs odds baseline
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
                minute = int(feat.get("minute", 0))

                # odds baseline
                odds = fetch_fixture_odds(fid)

                raw: List[Tuple[str,str,float,str]] = []
                for code, mdl in models.items():
                    raw.extend(_mk_suggestions_for_code(code, mdl, feat))

                filtered = _apply_filter_with_odds(
                    raw, minute, feat, odds,
                    min_prob=MIN_PROB, min_quota=MIN_QUOTA, top_k=2
                )
                if not filtered: continue

                league_id, league = _league_name(m)
                home, away = _teams(m)
                score_txt = _pretty_score(m)

                for market, suggestion, prob, head, base, lift, q in filtered:
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
                    msg = _format_tip_message(
                        league=league, home=home, away=away,
                        minute=minute, score_txt=score_txt,
                        suggestion=suggestion, prob=prob, baseline=base, lift=lift, quota=q,
                        rationale=rationale
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
# Harvest (unchanged)
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
    return f"🤖 Robi Superbrain is active ({mode}, {ai_tag}) · DB=Postgres · MIN_PROB={MIN_PROB:.2f} · MIN_QUOTA={MIN_QUOTA:.2f}"

@app.route("/healthz")
def healthz():
    return jsonify({"ok": True, "mode": ("HARVEST" if HARVEST_MODE else "PRODUCTION"), "ai_only": ONLY_MODEL_MODE})

@app.route("/predict/models")
def predict_models_route():
    _require_api_key()
    models = _list_models("model_v2:")
    return jsonify({"count": len(models), "codes": sorted(models.keys())})

@app.route("/predict/scan")
def predict_scan_route():
    _require_api_key()
    saved, live = production_scan()
    return jsonify({"ok": True, "live_seen": live, "tips_saved": saved, "min_prob": MIN_PROB, "min_quota": MIN_QUOTA})

@app.route("/train", methods=["POST", "GET"])
def train_route():
    _require_api_key()
    return jsonify(retrain_models_job())

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
        logging.info("🎯 Running in PRODUCTION mode (AI-only).")

    # Nightly jobs
    scheduler.add_job(retrain_models_job, CronTrigger(hour=3, minute=0, timezone=ZoneInfo("Europe/Berlin")),
                      id="train", replace_existing=True, misfire_grace_time=3600, coalesce=True)
    scheduler.add_job(nightly_digest_job, CronTrigger(hour=3, minute=2, timezone=ZoneInfo("Europe/Berlin")),
                      id="digest", replace_existing=True, misfire_grace_time=3600, coalesce=True)

    scheduler.start()
    logging.info("⏱️ Scheduler started (HARVEST_MODE=%s)", HARVEST_MODE)

    port = int(os.getenv("PORT", 5000))
    logging.info("✅ Robi Superbrain started.")
    app.run(host="0.0.0.0", port=port)
