import os, json, time, logging, sys, atexit, signal
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests
import numpy as np
import psycopg2
from psycopg2.pool import SimpleConnectionPool
from flask import Flask, jsonify, request, abort
from html import escape

# ──────────────────────────────────────────────────────────────────────────────
# Basic setup
# ──────────────────────────────────────────────────────────────────────────────

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

BERLIN_TZ = ZoneInfo("Europe/Berlin")
UTC_TZ = ZoneInfo("UTC")

# Logging
class _Formatter(logging.Formatter):
    def format(self, record):
        if not hasattr(record, "job_id"):
            record.job_id = "main"
        return super().format(record)

log = logging.getLogger("gs_ou25")
handler = logging.StreamHandler()
handler.setFormatter(_Formatter("[%(asctime)s] %(levelname)s [%(job_id)s] - %(message)s"))
log.handlers = [handler]
log.setLevel(logging.INFO)
log.propagate = False

app = Flask(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Environment / knobs
# ──────────────────────────────────────────────────────────────────────────────

def _required(k: str) -> str:
    v = os.getenv(k)
    if not v:
        raise SystemExit(f"Missing required environment variable: {k}")
    return v

TELEGRAM_BOT_TOKEN = _required("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = _required("TELEGRAM_CHAT_ID")
API_KEY            = _required("API_KEY")               # API-Football
DATABASE_URL       = _required("DATABASE_URL")

ADMIN_API_KEY      = os.getenv("ADMIN_API_KEY", "")
RUN_SCHEDULER      = os.getenv("RUN_SCHEDULER", "1").lower() not in ("0","false","no")

# Thresholds (singles). Scanner will read a live override from settings if present
CONF_THRESHOLD_PREMATCH_DEFAULT = float(os.getenv("CONF_THRESHOLD_PRE_OU25", "62"))  # %
CONF_THRESHOLD_LIVE             = float(os.getenv("CONF_THRESHOLD_LIVE_OU25", "68")) # % (not used unless you add live)

# Auto-tune knobs
TARGET_PRECISION        = float(os.getenv("TARGET_PRECISION", "0.60"))
THRESH_MIN_PREDICTIONS  = int(os.getenv("THRESH_MIN_PREDICTIONS", "25"))
MIN_THRESH              = float(os.getenv("MIN_THRESH", "55"))
MAX_THRESH              = float(os.getenv("MAX_THRESH", "85"))

# Odds / EV filters for singles
MIN_ODDS_OU       = float(os.getenv("MIN_ODDS_OU", "1.50"))
MAX_ODDS_ALL      = float(os.getenv("MAX_ODDS_ALL", "20.0"))
EDGE_MIN_BPS      = int(os.getenv("EDGE_MIN_BPS", "800"))   # +8.00% min EV per leg for singles

# Scanner pacing
SCAN_INTERVAL_LIVE_SEC  = int(os.getenv("SCAN_INTERVAL_LIVE_SEC", "180"))  # live scan cadence (unused if no live)
PREDICTIONS_PER_MATCH   = int(os.getenv("PREDICTIONS_PER_MATCH", "1"))
MAX_TIPS_PER_SCAN       = int(os.getenv("MAX_TIPS_PER_SCAN", "25"))

# OU line (locked to 2.5)
OU_LINE = 2.5

# MOTD (prematch OU 2.5 only)
MOTD_ENABLE       = os.getenv("MOTD_ENABLE", "1").lower() not in ("0","false","no")
MOTD_HOUR         = int(os.getenv("MOTD_HOUR", "10"))     # 10:15 Berlin
MOTD_MINUTE       = int(os.getenv("MOTD_MINUTE", "15"))
MOTD_CONF_MIN     = float(os.getenv("MOTD_CONF_MIN", "65"))  # %

# Auto-train (optional)
TRAIN_ENABLE      = os.getenv("TRAIN_ENABLE", "0").lower() not in ("0","false","no")
TRAIN_HOUR_UTC    = int(os.getenv("TRAIN_HOUR_UTC", "2"))
TRAIN_MINUTE_UTC  = int(os.getenv("TRAIN_MINUTE_UTC", "12"))

# Prematch parlay controls (OU 2.5 only)
PARLAY_ENABLE               = os.getenv("PARLAY_ENABLE", "1").lower() not in ("0","false","no")
PARLAY_LEGS_MIN             = int(os.getenv("PARLAY_LEGS_MIN", "2"))   # 2–3
PARLAY_LEGS_MAX             = int(os.getenv("PARLAY_LEGS_MAX", "3"))
PARLAY_MAX_PER_DAY          = int(os.getenv("PARLAY_MAX_PER_DAY", "2"))
PARLAY_PROB_MIN             = float(os.getenv("PARLAY_PROB_MIN", "0.56"))  # each leg
PARLAY_PROB_MAX             = float(os.getenv("PARLAY_PROB_MAX", "0.70"))
PARLAY_MIN_LEG_EV_BPS       = int(os.getenv("PARLAY_MIN_LEG_EV_BPS", "1000"))  # +10% per leg
PARLAY_MIN_COMBINED_EV_BPS  = int(os.getenv("PARLAY_MIN_COMBINED_EV_BPS", "800")) # +8% combined
PARLAY_LEAGUE_DIVERSITY     = os.getenv("PARLAY_LEAGUE_DIVERSITY", "1").lower() not in ("0","false","no")
PARLAY_HOUR                 = int(os.getenv("PARLAY_HOUR", "9"))   # build parlays once daily
PARLAY_MINUTE               = int(os.getenv("PARLAY_MINUTE", "45"))
PARLAY_INCLUDE_UNDER        = os.getenv("PARLAY_INCLUDE_UNDER", "1").lower() not in ("0","false","no")

# ──────────────────────────────────────────────────────────────────────────────
# API-Football
# ──────────────────────────────────────────────────────────────────────────────

BASE_URL = "https://v3.football.api-sports.io"
FOOTBALL_API_URL = f"{BASE_URL}/fixtures"
HEADERS = {"x-apisports-key": API_KEY, "Accept": "application/json"}
REQ_TIMEOUT = float(os.getenv("REQ_TIMEOUT_SEC", "8.0"))

session = requests.Session()

def _api_get(url: str, params: dict) -> Optional[dict]:
    try:
        r = session.get(url, headers=HEADERS, params=params, timeout=REQ_TIMEOUT)
        if r.ok:
            return r.json()
        return None
    except Exception as e:
        log.warning("API error: %s", e)
        return None

# ──────────────────────────────────────────────────────────────────────────────
# DB (Railway Postgres)
# ──────────────────────────────────────────────────────────────────────────────

def _normalize_dsn(url: str) -> str:
    dsn = url.strip()
    if "sslmode=" not in dsn:
        dsn += ("&" if "?" in dsn else "?") + "sslmode=require"
    return dsn

POOL: Optional[SimpleConnectionPool] = None

def _init_pool():
    global POOL
    if POOL:
        return
    POOL = SimpleConnectionPool(
        minconn=1,
        maxconn=int(os.getenv("PG_MAXCONN", "10")),
        dsn=_normalize_dsn(DATABASE_URL),
        connect_timeout=int(os.getenv("PG_CONNECT_TIMEOUT", "10")),
        application_name="gs-ou25"
    )
    log.info("[DB] pool initialized")

class Pooled:
    def __enter__(self):
        self.conn = POOL.getconn()
        self.conn.autocommit = True
        self.cur = self.conn.cursor()
        return self
    def __exit__(self, a, b, c):
        try:
            self.cur.close()
        except Exception:
            pass
        try:
            POOL.putconn(self.conn)
        except Exception:
            try:
                self.conn.close()
            except Exception:
                pass
    def execute(self, sql: str, params: tuple|list=()):
        self.cur.execute(sql, params or ())
        return self.cur

def init_db():
    with Pooled() as p:
        # Singles (tips)
        p.execute("""
        CREATE TABLE IF NOT EXISTS tips (
            match_id BIGINT,
            league_id BIGINT,
            league TEXT,
            home TEXT,
            away TEXT,
            market TEXT,
            suggestion TEXT,
            confidence DOUBLE PRECISION,
            confidence_raw DOUBLE PRECISION,
            score_at_tip TEXT,
            minute INTEGER,
            created_ts BIGINT,
            odds DOUBLE PRECISION,
            book TEXT,
            ev_pct DOUBLE PRECISION,
            sent_ok INTEGER DEFAULT 0,
            PRIMARY KEY (match_id, created_ts)
        )""")
        p.execute("CREATE INDEX IF NOT EXISTS idx_tips_created ON tips (created_ts DESC)")
        p.execute("CREATE INDEX IF NOT EXISTS idx_tips_market ON tips (market, created_ts DESC)")

        # Settings / results
        p.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)")
        p.execute("""
        CREATE TABLE IF NOT EXISTS match_results (
            match_id BIGINT PRIMARY KEY,
            final_goals_h INTEGER,
            final_goals_a INTEGER,
            btts_yes INTEGER,
            updated_ts BIGINT
        )""")

        # Prematch snapshots (for training)
        p.execute("""
        CREATE TABLE IF NOT EXISTS prematch_snapshots (
            match_id BIGINT PRIMARY KEY,
            created_ts BIGINT,
            payload TEXT
        )""")

        # Odds history (optional)
        p.execute("""
        CREATE TABLE IF NOT EXISTS odds_history (
            match_id BIGINT,
            captured_ts BIGINT,
            market TEXT,
            selection TEXT,
            odds DOUBLE PRECISION,
            book TEXT,
            PRIMARY KEY (match_id, market, selection, captured_ts)
        )""")
        p.execute("CREATE INDEX IF NOT EXISTS idx_odds_hist_match ON odds_history (match_id, captured_ts DESC)")

        # Parlays (prematch only)
        p.execute("""
        CREATE TABLE IF NOT EXISTS parlays (
            id BIGSERIAL PRIMARY KEY,
            created_ts BIGINT,
            legs JSONB,
            legs_count INTEGER,
            combined_odds DOUBLE PRECISION,
            combined_prob DOUBLE PRECISION,
            ev_pct DOUBLE PRECISION,
            sent_ok INTEGER DEFAULT 0,
            type TEXT DEFAULT 'PREMATCH'
        )""")

# ──────────────────────────────────────────────────────────────────────────────
# Settings helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_setting(key: str) -> Optional[str]:
    with Pooled() as p:
        r = p.execute("SELECT value FROM settings WHERE key=%s", (key,)).fetchone()
        return r[0] if r else None

def set_setting(key: str, value: str) -> None:
    with Pooled() as p:
        p.execute(
            "INSERT INTO settings(key,value) VALUES(%s,%s) "
            "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
            (key, value)
        )

def get_conf_threshold_prematch() -> float:
    # live override from settings if present
    k = "conf_threshold:PRE Over/Under 2.5"
    v = get_setting(k)
    try:
        if v is not None:
            return float(v)
    except Exception:
        pass
    return CONF_THRESHOLD_PREMATCH_DEFAULT

# ──────────────────────────────────────────────────────────────────────────────
# Telegram
# ──────────────────────────────────────────────────────────────────────────────

def send_telegram(text: str) -> bool:
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode":"HTML", "disable_web_page_preview":True},
            timeout=REQ_TIMEOUT
        )
        return bool(r.ok)
    except Exception as e:
        log.warning("TG error: %s", e)
        return False

# ──────────────────────────────────────────────────────────────────────────────
# Models (OU 2.5 only)
# ──────────────────────────────────────────────────────────────────────────────

MODEL_KEYS_ORDER = [
    "model_v2:{name}", "model_latest:{name}", "model:{name}"
]

def _fmt_line(x: float) -> str:
    s = f"{x}"
    return s.rstrip("0").rstrip(".")

def _validate_model_blob(tmp: dict) -> bool:
    return isinstance(tmp, dict) and isinstance(tmp.get("weights"), dict) and "intercept" in tmp

def load_model_from_settings(name: str) -> Optional[Dict[str, Any]]:
    for kpat in MODEL_KEYS_ORDER:
        k = kpat.format(name=name)
        raw = get_setting(k)
        if not raw:
            continue
        try:
            blob = json.loads(raw)
            if _validate_model_blob(blob):
                cal = blob.get("calibration") or {}
                if isinstance(cal, dict):
                    cal.setdefault("method", "sigmoid")
                    cal.setdefault("a", 1.0)
                    cal.setdefault("b", 0.0)
                blob["calibration"] = cal
                return blob
        except Exception as e:
            log.warning("model parse %s failed: %s", k, e)
    return None

def _sigmoid(z: float) -> float:
    if z < -50: return 1e-22
    if z >  50: return 1-1e-22
    import math
    return 1 / (1 + math.exp(-z))

def _logit(p: float) -> float:
    import math
    p = max(1e-12, min(1-1e-12, float(p)))
    return math.log(p/(1-p))

def _score_linear(feat: Dict[str, float], weights: Dict[str, float], intercept: float) -> float:
    s = float(intercept or 0.0)
    for k, w in (weights or {}).items():
        s += float(w or 0.0) * float(feat.get(k, 0.0) or 0.0)
    return s

def _calibrate(p: float, cal: Dict[str, Any]) -> float:
    m = (cal or {}).get("method", "sigmoid").lower()
    a = float((cal or {}).get("a", 1.0)); b = float((cal or {}).get("b", 0.0))
    return _sigmoid(a*_logit(p) + b)

def score_prob(mdl: Dict[str, Any], feat: Dict[str, float]) -> float:
    p = _sigmoid(_score_linear(feat, mdl.get("weights", {}), float(mdl.get("intercept", 0.0))))
    try:
        cal = mdl.get("calibration") or {}
        p = _calibrate(p, cal)
    except Exception:
        pass
    return float(max(0.0, min(1.0, p)))

# ──────────────────────────────────────────────────────────────────────────────
# Feature extraction (prematch + minimal live)
# ──────────────────────────────────────────────────────────────────────────────

def _pos_pct(v) -> float:
    try: return float(str(v).replace("%","").strip() or 0.0)
    except: return 0.0

def extract_prematch_features(fx: dict) -> Dict[str, float]:
    # Simple placeholder (your real prematch features can be used here)
    return {"fid": float(((fx.get("fixture") or {}).get("id") or 0))}

def extract_live_features(m: dict) -> Dict[str, float]:
    feat: Dict[str, float] = {}
    minute = int((((m.get("fixture") or {}).get("status") or {}).get("elapsed") or 0) or 0)
    feat["minute"] = float(minute)

    home = ((m.get("teams") or {}).get("home") or {}).get("name", "")
    away = ((m.get("teams") or {}).get("away") or {}).get("name", "")

    stats_by = {}
    for s in (m.get("statistics") or []):
        t = ((s.get("team") or {}).get("name") or "").strip()
        stats_by[t] = { (i.get("type") or ""): i.get("value") for i in (s.get("statistics") or []) }

    sh = stats_by.get(home, {}) or {}
    sa = stats_by.get(away, {}) or {}

    xg_h = float(sh.get("Expected Goals", 0) or 0.0)
    xg_a = float(sa.get("Expected Goals", 0) or 0.0)
    feat["xg_sum"] = float(xg_h + xg_a)
    feat["sot_sum"] = float((sh.get("Shots on Target", sh.get("Shots on Goal", 0)) or 0) +
                             (sa.get("Shots on Target", sa.get("Shots on Goal", 0)) or 0))
    feat["cor_sum"] = float((sh.get("Corner Kicks", 0) or 0) + (sa.get("Corner Kicks", 0) or 0))
    feat["pos_h"] = _pos_pct(sh.get("Ball Possession", 0) or 0)
    feat["pos_a"] = _pos_pct(sa.get("Ball Possession", 0) or 0)

    gh = int(((m.get("goals") or {}).get("home") or 0) or 0)
    ga = int(((m.get("goals") or {}).get("away") or 0) or 0)
    feat["goals_sum"] = float(gh + ga)

    return feat

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures & odds
# ──────────────────────────────────────────────────────────────────────────────

def _today_utc_dates() -> List[str]:
    today_local = datetime.now(BERLIN_TZ).date()
    return [today_local.strftime("%Y-%m-%d")]

def _collect_todays_prematch_fixtures() -> List[dict]:
    out = []
    for d in _today_utc_dates():
        js = _api_get(FOOTBALL_API_URL, {"date": d}) or {}
        for r in (js.get("response") or []):
            st = (((r.get("fixture") or {}).get("status") or {}).get("short") or "").upper()
            if st == "NS":
                out.append(r)
    return out

def fetch_odds_prematch(fid: int) -> dict:
    """
    Return aggregated OU 2.5 odds structure:
    { "OU_2.5": {"Over": {"odds": x, "book":"..."}, "Under": {...}} }
    """
    js = _api_get(f"{BASE_URL}/odds", {"fixture": fid}) or {}
    by_market: dict[str, dict[str, List[Tuple[float,str]]]] = {}
    try:
        for r in (js.get("response") or []):
            for bk in (r.get("bookmakers") or []):
                book = bk.get("name") or "Book"
                for bet in (bk.get("bets") or []):
                    name = (bet.get("name") or "").lower()
                    if "over/under" in name or "total" in name or "goals" in name:
                        for v in (bet.get("values") or []):
                            lbl = (v.get("value") or "").lower().strip()  # e.g., "Over 2.5"
                            try:
                                side = "Over" if "over" in lbl else ("Under" if "under" in lbl else None)
                                ln = float(lbl.split()[-1])
                            except Exception:
                                side = None; ln = None
                            if side and abs(ln - 2.5) < 1e-6:
                                by_market.setdefault("OU_2.5", {}).setdefault(side, []).append((float(v.get("odd") or 0), book))
    except Exception:
        pass

    out: dict = {}
    if "OU_2.5" in by_market:
        out["OU_2.5"] = {}
        for side, lst in by_market["OU_2.5"].items():
            xs = [o for (o, _) in lst if o > 0]
            if not xs:
                continue
            import statistics
            med = statistics.median(xs)
            pick = min(lst, key=lambda t: abs(t[0]-med))  # median quote
            out["OU_2.5"][side] = {"odds": float(pick[0]), "book": f"{pick[1]} (median {len(xs)})"}
    return out

# ──────────────────────────────────────────────────────────────────────────────
# Math helpers
# ──────────────────────────────────────────────────────────────────────────────

def ev(prob: float, odds: float) -> float:
    """Return EV as decimal (e.g., 0.08 = +8%)."""
    return float(prob) * float(max(0.0, odds)) - 1.0

# ──────────────────────────────────────────────────────────────────────────────
# Singles: PREMATCH OU 2.5
# ──────────────────────────────────────────────────────────────────────────────

def _load_model_pre_ou25() -> Optional[dict]:
    return load_model_from_settings("PRE_OU_2.5") or load_model_from_settings("OU_2.5")

def _format_tip_msg_prematch(home, away, league, suggestion, pct, odds, book, ev_pct):
    return (
        "⚽️ <b>Prematch Tip</b>\n"
        f"<b>Match:</b> {escape(home)} vs {escape(away)}\n"
        f"<b>Market:</b> Over/Under 2.5\n"
        f"<b>Tip:</b> {escape(suggestion)}\n"
        f"📈 <b>Confidence:</b> {pct:.1f}%\n"
        f"💰 <b>Odds:</b> {odds:.2f} @ {escape(book or 'Book')}  •  <b>EV:</b> {ev_pct:+.1f}%\n"
        f"🏆 <b>League:</b> {escape(league)}"
    )

def prematch_scan_ou25() -> int:
    """Score all today's 'NS' fixtures for OU 2.5, send singles meeting gates."""
    mdl = _load_model_pre_ou25()
    if not mdl:
        log.warning("[PRE] model PRE_OU_2.5 not found")
        return 0

    thr_pct = get_conf_threshold_prematch()
    fixtures = _collect_todays_prematch_fixtures()
    if not fixtures:
        return 0

    saved = 0
    now = int(time.time())
    for fx in fixtures:
        try:
            fixture = fx.get("fixture") or {}
            fid = int(fixture.get("id") or 0)
            if not fid:
                continue

            teams = fx.get("teams") or {}
            lg = fx.get("league") or {}
            home = (teams.get("home") or {}).get("name","")
            away = (teams.get("away") or {}).get("name","")
            league_id = int(lg.get("id") or 0)
            league = f"{lg.get('country','')} - {lg.get('name','')}".strip(" -")

            feat = extract_prematch_features(fx)
            p_over = score_prob(mdl, feat)
            q_under = 1.0 - p_over

            odds_map = fetch_odds_prematch(fid).get("OU_2.5") or {}
            over_odds = (odds_map.get("Over") or {}).get("odds")
            under_odds = (odds_map.get("Under") or {}).get("odds")
            over_book = (odds_map.get("Over") or {}).get("book")
            under_book = (odds_map.get("Under") or {}).get("book")

            def _maybe_send(suggestion: str, prob: float, odds: Optional[float], book: Optional[str]) -> bool:
                if prob * 100.0 < thr_pct:
                    return False
                if not odds or not (MIN_ODDS_OU <= odds <= MAX_ODDS_ALL):
                    return False
                edge = ev(prob, odds)
                if int(round(edge * 10000)) < EDGE_MIN_BPS:
                    return False

                pct = round(prob * 100.0, 1)
                ev_pct = round(edge * 100.0, 1)
                with Pooled() as p:
                    p.execute("""
                        INSERT INTO tips(match_id,league_id,league,home,away,market,suggestion,
                                         confidence,confidence_raw,score_at_tip,minute,created_ts,
                                         odds,book,ev_pct,sent_ok)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,'0-0',0,%s,%s,%s,%s,0)
                    """, (fid, league_id, league, home, away, "PRE Over/Under 2.5", suggestion,
                          pct, prob, now, float(odds), book or None, ev_pct))
                ok = send_telegram(_format_tip_msg_prematch(home, away, league, suggestion, pct, odds, book, ev_pct))
                if ok:
                    with Pooled() as p:
                        p.execute("UPDATE tips SET sent_ok=1 WHERE match_id=%s AND created_ts=%s", (fid, now))
                return ok

            sent_any = False
            over_first = (p_over >= q_under)
            if over_first:
                sent_any |= _maybe_send("Over 2.5 Goals", p_over, over_odds, over_book)
                if not sent_any and PARLAY_INCLUDE_UNDER:
                    sent_any |= _maybe_send("Under 2.5 Goals", q_under, under_odds, under_book)
            else:
                if PARLAY_INCLUDE_UNDER:
                    sent_any |= _maybe_send("Under 2.5 Goals", q_under, under_odds, under_book)
                if not sent_any:
                    sent_any |= _maybe_send("Over 2.5 Goals", p_over, over_odds, over_book)

            if sent_any:
                saved += 1
                if saved >= MAX_TIPS_PER_SCAN:
                    break
        except Exception as e:
            log.warning("[PRE] error: %s", e)
            continue
    return saved

# ──────────────────────────────────────────────────────────────────────────────
# MOTD (Prematch OU 2.5) at 10:15 Berlin
# ──────────────────────────────────────────────────────────────────────────────

def send_motd() -> bool:
    if not MOTD_ENABLE:
        return False
    mdl = _load_model_pre_ou25()
    if not mdl:
        return False
    fixtures = _collect_todays_prematch_fixtures()
    best = None
    for fx in fixtures:
        try:
            fid = int((fx.get("fixture") or {}).get("id") or 0)
            if not fid: 
                continue
            teams = fx.get("teams") or {}
            lg = fx.get("league") or {}
            home = (teams.get("home") or {}).get("name","")
            away = (teams.get("away") or {}).get("name","")
            league = f"{lg.get('country','')} - {lg.get('name','')}".strip(" -")
            kickoff_iso = (fx.get("fixture") or {}).get("date") or ""
            kickoff_txt = "TBD"
            try:
                dt = datetime.fromisoformat(kickoff_iso.replace("Z","+00:00")).astimezone(BERLIN_TZ)
                kickoff_txt = dt.strftime("%H:%M")
            except Exception:
                pass
            feat = extract_prematch_features(fx)
            p_over = score_prob(mdl, feat)
            odds = (fetch_odds_prematch(fid).get("OU_2.5") or {}).get("Over") or {}
            o = odds.get("odds")
            if not o or o < MIN_ODDS_OU:
                continue
            edge = ev(p_over, o)
            pct = p_over*100.0
            if pct < MOTD_CONF_MIN:
                continue
            score = pct + (edge*100.0)
            cand = (score, pct, edge*100.0, home, away, league, kickoff_txt, o, odds.get("book"))
            if (best is None) or (cand > best):
                best = cand
        except Exception:
            continue

    if not best:
        return send_telegram("🏅 MOTD: no OU 2.5 pick met thresholds today.")
    score, pct, ev_pct, home, away, league, kickoff_txt, odds, book = best
    msg = (
        "🏅 <b>Match of the Day</b>\n"
        f"<b>Match:</b> {escape(home)} vs {escape(away)}\n"
        f"<b>Market:</b> Over/Under 2.5\n"
        f"<b>Tip:</b> Over 2.5 Goals\n"
        f"📈 <b>Confidence:</b> {pct:.1f}%\n"
        f"💰 <b>Odds:</b> {odds:.2f} @ {escape(book or 'Book')}  •  <b>EV:</b> {ev_pct:+.1f}%\n"
        f"⏰ <b>Kickoff (Berlin):</b> {kickoff_txt}\n"
        f"🏆 <b>League:</b> {escape(league)}"
    )
    return send_telegram(msg)

# ──────────────────────────────────────────────────────────────────────────────
# PREMATCH PARLAYS: OU 2.5 (2–3 legs)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Leg:
    match_id: int
    league_id: int
    league: str
    home: str
    away: str
    kickoff: str
    suggestion: str   # "Over 2.5 Goals" or "Under 2.5 Goals"
    prob: float
    odds: float
    book: str
    ev_pct: float

def _gather_parlay_legs() -> List[Leg]:
    mdl = _load_model_pre_ou25()
    if not mdl:
        return []
    fixtures = _collect_todays_prematch_fixtures()
    legs: List[Leg] = []
    for fx in fixtures:
        try:
            fixture = fx.get("fixture") or {}
            fid = int(fixture.get("id") or 0)
            if not fid: 
                continue
            dt_iso = fixture.get("date") or ""
            try:
                dt_local = datetime.fromisoformat(dt_iso.replace("Z","+00:00")).astimezone(BERLIN_TZ)
                kickoff_txt = dt_local.strftime("%H:%M")
            except Exception:
                kickoff_txt = "TBD"
            teams = fx.get("teams") or {}
            lg = fx.get("league") or {}
            home = (teams.get("home") or {}).get("name","")
            away = (teams.get("away") or {}).get("name","")
            league_id = int(lg.get("id") or 0)
            league = f"{lg.get('country','')} - {lg.get('name','')}".strip(" -")

            feat = extract_prematch_features(fx)
            p_over = score_prob(mdl, feat)
            p_under = 1.0 - p_over

            odds_map = fetch_odds_prematch(fid).get("OU_2.5") or {}
            over_odds = (odds_map.get("Over") or {}).get("odds")
            under_odds = (odds_map.get("Under") or {}).get("odds")
            over_book = (odds_map.get("Over") or {}).get("book")
            under_book = (odds_map.get("Under") or {}).get("book")

            def _push(side: str, prob: float, odds: Optional[float], book: Optional[str]):
                if odds is None or odds < MIN_ODDS_OU or odds > MAX_ODDS_ALL:
                    return
                if not (PARLAY_PROB_MIN <= prob <= PARLAY_PROB_MAX):
                    return
                edge = ev(prob, float(odds))
                if int(round(edge * 10000)) < PARLAY_MIN_LEG_EV_BPS:
                    return
                legs.append(Leg(
                    match_id=fid, league_id=league_id, league=league,
                    home=home, away=away, kickoff=kickoff_txt,
                    suggestion=f"{side} 2.5 Goals", prob=float(prob),
                    odds=float(odds), book=str(book or "Book"),
                    ev_pct=round(edge*100.0, 1)
                ))

            _push("Over", p_over, over_odds, over_book)
            if PARLAY_INCLUDE_UNDER:
                _push("Under", p_under, under_odds, under_book)

        except Exception as e:
            log.debug("parlay leg error: %s", e)
            continue
    # Deduplicate per match (keep best EV)
    best_by_match: Dict[int, Leg] = {}
    for leg in legs:
        cur = best_by_match.get(leg.match_id)
        if (cur is None) or (leg.ev_pct > cur.ev_pct):
            best_by_match[leg.match_id] = leg
    legs = list(best_by_match.values())
    legs.sort(key=lambda L: L.ev_pct, reverse=True)
    return legs

def _combine_legs(legs: List[Leg], k: int) -> List[List[Leg]]:
    from itertools import combinations
    combos = []
    for comb in combinations(legs, k):
        # league diversity if enabled
        if PARLAY_LEAGUE_DIVERSITY:
            leagues = {c.league_id for c in comb}
            if len(leagues) < len(comb):
                continue
        combos.append(list(comb))
    return combos

def _score_parlay(legs: List[Leg]) -> Tuple[float, float, float]:
    """Return (combined_odds, combined_prob, ev_pct)."""
    odds = 1.0
    prob = 1.0
    for l in legs:
        odds *= float(l.odds)
        prob *= float(l.prob)
    edge = ev(prob, odds)
    return odds, prob, round(edge*100.0, 1)

def build_and_send_prematch_parlays() -> int:
    if not PARLAY_ENABLE:
        return 0
    legs = _gather_parlay_legs()
    if not legs:
        log.info("[PARLAY] no eligible legs")
        return 0

    best_parlays: List[Tuple[List[Leg], float, float, float]] = []  # legs, odds, prob, ev_pct

    for n_legs in range(max(2, PARLAY_LEGS_MIN), min(3, PARLAY_LEGS_MAX)+1):
        combos = _combine_legs(legs, n_legs)
        for comb in combos:
            co, cp, ev_pct = _score_parlay(comb)
            if int(round(ev_pct*100)) < PARLAY_MIN_COMBINED_EV_BPS:
                continue
            best_parlays.append((comb, co, cp, ev_pct))

    if not best_parlays:
        log.info("[PARLAY] no combos met combined EV")
        return 0

    best_parlays.sort(key=lambda x: (x[3], x[1]), reverse=True)
    picked = best_parlays[:max(1, PARLAY_MAX_PER_DAY)]

    sent = 0
    now = int(time.time())
    for legs_list, co, cp, ev_pct in picked:
        try:
            legs_payload = []
            lines = []
            for l in legs_list:
                legs_payload.append({
                    "match_id": l.match_id, "league_id": l.league_id, "league": l.league,
                    "home": l.home, "away": l.away, "kickoff": l.kickoff,
                    "suggestion": l.suggestion, "prob": l.prob,
                    "odds": l.odds, "book": l.book, "ev_pct": l.ev_pct
                })
                lines.append(f"• {escape(l.home)} vs {escape(l.away)} — {escape(l.suggestion)} — "
                             f"{l.prob*100:.1f}%  |  {l.odds:.2f} @ {escape(l.book)}  |  EV {l.ev_pct:+.1f}%")

            with Pooled() as p:
                p.execute("""
                    INSERT INTO parlays(created_ts, legs, legs_count, combined_odds, combined_prob, ev_pct, sent_ok, type)
                    VALUES (%s,%s,%s,%s,%s,%s,0,'PREMATCH')
                """, (now, json.dumps(legs_payload, separators=(",",":")), len(legs_list), float(co), float(cp), float(ev_pct)))

            msg = (
                "🎯 <b>Prematch Parlay (OU 2.5)</b>\n"
                f"🧩 Legs: {len(legs_list)}\n"
                + "\n".join(lines) + "\n\n"
                f"🔗 <b>Combined Odds:</b> {co:.2f}\n"
                f"📈 <b>Combined Confidence:</b> {cp*100:.1f}%\n"
                f"💰 <b>Combined EV:</b> {ev_pct:+.1f}%"
            )
            ok = send_telegram(msg)
            if ok:
                with Pooled() as p:
                    p.execute(
                        "UPDATE parlays SET sent_ok=1 WHERE created_ts=%s AND combined_odds=%s",
                        (now, float(co))
                    )
                sent += 1
        except Exception as e:
            log.warning("[PARLAY] send error: %s", e)
            continue

    log.info("[PARLAY] sent=%d", sent)
    return sent

# ──────────────────────────────────────────────────────────────────────────────
# Outcomes / backfill / digest
# ──────────────────────────────────────────────────────────────────────────────

def _tip_outcome_for_result(suggestion: str, gh: int, ga: int) -> Optional[int]:
    total = int(gh) + int(ga)
    if suggestion == "Over 2.5 Goals":
        if total > 2.5: return 1
        if abs(total - 2.5) < 1e-9: return None
        return 0
    if suggestion == "Under 2.5 Goals":
        if total < 2.5: return 1
        if abs(total - 2.5) < 1e-9: return None
        return 0
    return None

def backfill_results_for_open_matches(max_rows: int = 300) -> int:
    now_ts = int(time.time())
    cutoff = now_ts - 14*24*3600
    with Pooled() as p:
        rows = p.execute("""
            WITH last AS (SELECT match_id, MAX(created_ts) last_ts FROM tips WHERE created_ts >= %s GROUP BY match_id)
            SELECT l.match_id FROM last l
            LEFT JOIN match_results r ON r.match_id=l.match_id
            WHERE r.match_id IS NULL
            ORDER BY l.last_ts DESC
            LIMIT %s
        """, (cutoff, max_rows)).fetchall()
    upd = 0
    for (mid,) in rows:
        try:
            js = _api_get(FOOTBALL_API_URL, {"id": int(mid)}) or {}
            arr = (js.get("response") or [])
            if not arr: 
                continue
            fx = arr[0]
            short = ((((fx.get("fixture") or {}).get("status") or {}).get("short") or "").upper())
            if short not in {"FT","AET","PEN"}:
                continue
            g = (fx.get("goals") or {})
            gh = int(g.get("home") or 0); ga = int(g.get("away") or 0)
            with Pooled() as p:
                p.execute("""
                    INSERT INTO match_results(match_id, final_goals_h, final_goals_a, btts_yes, updated_ts)
                    VALUES (%s,%s,%s,%s,%s)
                    ON CONFLICT(match_id) DO UPDATE
                    SET final_goals_h=EXCLUDED.final_goals_h,
                        final_goals_a=EXCLUDED.final_goals_a,
                        btts_yes=EXCLUDED.btts_yes,
                        updated_ts=EXCLUDED.updated_ts
                """, (int(mid), gh, ga, 1 if (gh>0 and ga>0) else 0, int(time.time())))
            upd += 1
        except Exception:
            continue
    return upd

def daily_accuracy_digest() -> Optional[str]:
    today = datetime.now(BERLIN_TZ).date()
    start_of_day = datetime.combine(today, datetime.min.time(), tzinfo=BERLIN_TZ)
    start_ts = int(start_of_day.timestamp())

    backfill_results_for_open_matches(400)

    with Pooled() as p:
        rows = p.execute("""
            SELECT t.market, t.suggestion, t.confidence, t.confidence_raw, t.created_ts,
                   t.odds, r.final_goals_h, r.final_goals_a
            FROM tips t LEFT JOIN match_results r ON r.match_id = t.match_id
            WHERE t.created_ts >= %s AND t.sent_ok=1
            ORDER BY t.created_ts DESC
        """, (start_ts,)).fetchall()

    total = graded = wins = 0
    for (mkt, sugg, conf, conf_raw, cts, odds, gh, ga) in rows:
        if gh is None or ga is None:
            continue
        out = _tip_outcome_for_result(sugg, int(gh), int(ga))
        if out is None:
            continue
        graded += 1
        wins += int(out == 1)
        total += 1

    with Pooled() as p:
        pr = p.execute("""
            SELECT legs_count, combined_odds, combined_prob, ev_pct
            FROM parlays WHERE created_ts >= %s AND sent_ok=1
            ORDER BY created_ts DESC
        """, (start_ts,)).fetchall()

    acc = (100.0 * wins / max(1, graded)) if graded else 0.0
    lines = [
        f"📊 <b>Daily Accuracy Digest</b> - {today.strftime('%Y-%m-%d')}",
        f"Tips sent: {total}  •  Graded: {graded}  •  Wins: {wins}  •  Accuracy: {acc:.1f}%"
    ]
    if pr:
        lines.append("\n🧮 Parlays sent today:")
        for (lc, co, cp, evp) in pr[:5]:
            lines.append(f"• {lc}-leg — Odds {float(co):.2f} • Conf {float(cp)*100:.1f}% • EV {float(evp):+.1f}%")
    msg = "\n".join(lines)
    send_telegram(msg)
    return msg

# ──────────────────────────────────────────────────────────────────────────────
# Retry unsent tips (manual)
# ──────────────────────────────────────────────────────────────────────────────

def retry_unsent_tips(minutes: int = 30, limit: int = 200) -> int:
    cutoff = int(time.time()) - minutes*60
    retried = 0
    with Pooled() as p:
        rows = p.execute(
            "SELECT match_id,league,home,away,market,suggestion,confidence,confidence_raw,score_at_tip,minute,created_ts,odds,book,ev_pct "
            "FROM tips WHERE sent_ok=0 AND created_ts >= %s ORDER BY created_ts ASC LIMIT %s",
            (cutoff, limit)
        ).fetchall()

    for (mid, league, home, away, market, sugg, conf, conf_raw, score, minute, cts, odds, book, ev_pct) in rows:
        try:
            pct = float(conf or 0.0)
            msg = _format_tip_msg_prematch(home, away, league, sugg, pct, float(odds or 0.0), book or "Book", float(ev_pct or 0.0))
            ok = send_telegram(msg)
            if ok:
                with Pooled() as p:
                    p.execute("UPDATE tips SET sent_ok=1 WHERE match_id=%s AND created_ts=%s", (mid, cts))
                retried += 1
        except Exception:
            continue
    if retried:
        log.info("[RETRY] resent %d", retried)
    return retried

# ──────────────────────────────────────────────────────────────────────────────
# ROI-aware Auto-tune (prematch OU 2.5)
# ──────────────────────────────────────────────────────────────────────────────

def auto_tune_thresholds(days: int = 14) -> Dict[str, float]:
    cutoff = int(time.time()) - days * 24 * 3600
    with Pooled() as p:
        rows = p.execute(
            """
            SELECT t.suggestion,
                   COALESCE(t.confidence_raw, t.confidence/100.0) AS prob,
                   t.odds,
                   r.final_goals_h, r.final_goals_a
            FROM tips t
            JOIN match_results r ON r.match_id = t.match_id
            WHERE t.created_ts >= %s
              AND t.market = 'PRE Over/Under 2.5'
              AND t.odds IS NOT NULL
              AND t.sent_ok = 1
            """,
            (cutoff,),
        ).fetchall()

    items: List[Tuple[float, int, float]] = []
    for (sugg, prob, odds, gh, ga) in rows:
        try:
            prob = float(prob or 0.0); odds = float(odds or 0.0)
        except Exception:
            continue
        if not (1.01 <= odds <= MAX_ODDS_ALL):
            continue
        out = _tip_outcome_for_result(sugg, int(gh), int(ga))
        if out is None:
            continue
        items.append((prob, int(out), odds))

    if len(items) < THRESH_MIN_PREDICTIONS:
        send_telegram("🔧 Auto-tune: not enough labeled prematch OU 2.5 tips.")
        return {}

    def _eval(thr_prob: float) -> Tuple[int, float, float]:
        sel = [(p, y, o) for (p, y, o) in items if p >= thr_prob]
        n = len(sel)
        if n == 0: return 0, 0.0, 0.0
        wins = sum(y for (_, y, _) in sel); prec = wins / n
        roi = sum((y * (odds - 1.0) - (1 - y)) for (_, y, odds) in sel) / n
        return n, float(prec), float(roi)

    best = None
    feasible_any = False
    candidates_pct = list(np.arange(MIN_THRESH, MAX_THRESH + 1e-9, 1.0))
    for thr_pct in candidates_pct:
        thr_prob = float(thr_pct / 100.0)
        n, prec, roi = _eval(thr_prob)
        if n < THRESH_MIN_PREDICTIONS:
            continue
        if prec >= TARGET_PRECISION:
            feasible_any = True
            score = (roi, prec, n)
            if (best is None) or (score > (best[0], best[1], best[2])):
                best = (roi, prec, n, thr_pct)

    if not feasible_any:
        # fallback: allow precision within -0.03 and positive ROI
        for thr_pct in candidates_pct:
            thr_prob = float(thr_pct / 100.0)
            n, prec, roi = _eval(thr_prob)
            if n < THRESH_MIN_PREDICTIONS:
                continue
            if (prec >= max(0.0, TARGET_PRECISION - 0.03)) and (roi > 0.0):
                score = (roi, prec, n)
                if (best is None) or (score > (best[0], best[1], best[2])):
                    best = (roi, prec, n, thr_pct)

    if best is None:
        # last resort: highest precision then volume
        fallback = None
        for thr_pct in candidates_pct:
            thr_prob = float(thr_pct / 100.0)
            n, prec, roi = _eval(thr_prob)
            if n < THRESH_MIN_PREDICTIONS:
                continue
            score = (prec, n, roi)
            if (fallback is None) or (score > (fallback[0], fallback[1], fallback[2])):
                fallback = (prec, n, roi, thr_pct)
        if fallback is not None:
            tuned_pct = float(fallback[3])
        else:
            send_telegram("🔧 Auto-tune: unable to select a threshold.")
            return {}
    else:
        tuned_pct = float(best[3])

    set_setting("conf_threshold:PRE Over/Under 2.5", f"{tuned_pct:.2f}")
    send_telegram(f"🔧 Auto-tune updated: PRE Over/Under 2.5 → {tuned_pct:.1f}%")
    return {"PRE Over/Under 2.5": tuned_pct}

# ──────────────────────────────────────────────────────────────────────────────
# Scheduler
# ──────────────────────────────────────────────────────────────────────────────

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

_s = None

def start_scheduler():
    global _s
    if _s or not RUN_SCHEDULER:
        return
    _s = BackgroundScheduler(timezone=UTC_TZ)

    # Prematch singles: run every 20 min
    _s.add_job(prematch_scan_ou25, "interval", minutes=20, id="pre_singles", max_instances=1, coalesce=True)

    # MOTD at 10:15 Berlin
    if MOTD_ENABLE:
        _s.add_job(
            send_motd,
            CronTrigger(hour=MOTD_HOUR, minute=MOTD_MINUTE, timezone=BERLIN_TZ),
            id="motd", max_instances=1, coalesce=True
        )

    # Auto-train (optional)
    if TRAIN_ENABLE:
        _s.add_job(
            lambda: train_models_hook(),
            CronTrigger(hour=TRAIN_HOUR_UTC, minute=TRAIN_MINUTE_UTC, timezone=UTC_TZ),
            id="train", max_instances=1, coalesce=True
        )

    # Build + send prematch parlays once daily
    if PARLAY_ENABLE:
        _s.add_job(
            build_and_send_prematch_parlays,
            CronTrigger(hour=PARLAY_HOUR, minute=PARLAY_MINUTE, timezone=BERLIN_TZ),
            id="parlays", max_instances=1, coalesce=True
        )

    _s.start()
    send_telegram("🚀 OU 2.5 engine started (prematch singles + MOTD + prematch parlays).")
    log.info("[SCHED] started")

# ──────────────────────────────────────────────────────────────────────────────
# Optional training hook
# ──────────────────────────────────────────────────────────────────────────────

def train_models_hook():
    if not TRAIN_ENABLE:
        return send_telegram("🤖 Training skipped: TRAIN_ENABLE=0")
    try:
        import train_models as _tm  # your own module
        res = _tm.train_models() or {}
        ok = bool(res.get("ok"))
        if ok:
            send_telegram("🤖 Model training OK")
        else:
            send_telegram(f"⚠️ Training skipped: {escape(str(res))}")
    except Exception as e:
        send_telegram(f"❌ Training failed: {escape(str(e))}")

# ──────────────────────────────────────────────────────────────────────────────
# Flask endpoints (admin + status)
# ──────────────────────────────────────────────────────────────────────────────

def _require_admin():
    key = request.headers.get("X-API-Key") or request.args.get("key") or ((request.json or {}).get("key") if request.is_json else None)
    if not ADMIN_API_KEY or key != ADMIN_API_KEY:
        abort(401)

@app.route("/")
def root():
    return jsonify({"ok": True, "name": "gs-ou25", "parlays": PARLAY_ENABLE, "scheduler": RUN_SCHEDULER})

@app.route("/init-db", methods=["POST","GET"])
def http_init_db():
    _require_admin()
    init_db()
    return jsonify({"ok": True})

# Manual controls (GET or POST)

@app.route("/admin/prematch-scan", methods=["POST","GET"])
def http_pre_scan():
    _require_admin()
    n = prematch_scan_ou25()
    return jsonify({"ok": True, "saved": int(n)})

@app.route("/admin/backfill-results", methods=["POST","GET"])
def http_backfill():
    _require_admin()
    n = backfill_results_for_open_matches(400)
    return jsonify({"ok": True, "updated": int(n)})

@app.route("/admin/retry-unsent", methods=["POST","GET"])
def http_retry_unsent():
    _require_admin()
    try:
        minutes = int(request.args.get("minutes", "30"))
        limit = int(request.args.get("limit", "200"))
    except Exception:
        minutes, limit = 30, 200
    n = retry_unsent_tips(minutes, limit)
    return jsonify({"ok": True, "resent": int(n)})

@app.route("/admin/train", methods=["POST","GET"])
def http_train():
    _require_admin()
    train_models_hook()
    return jsonify({"ok": True})

@app.route("/admin/auto-tune", methods=["POST","GET"])
def http_auto_tune():
    _require_admin()
    tuned = auto_tune_thresholds(14)
    return jsonify({"ok": True, "tuned": tuned})

@app.route("/admin/parlays", methods=["POST","GET"])
def http_parlays():
    _require_admin()
    n = build_and_send_prematch_parlays()
    return jsonify({"ok": True, "sent": int(n)})

@app.route("/admin/motd", methods=["POST","GET"])
def http_motd():
    _require_admin()
    ok = send_motd()
    return jsonify({"ok": bool(ok)})

@app.route("/admin/digest", methods=["POST","GET"])
def http_digest():
    _require_admin()
    msg = daily_accuracy_digest()
    return jsonify({"ok": True, "sent": bool(msg)})

@app.route("/health")
def health():
    try:
        with Pooled() as p:
            n = p.execute("SELECT COUNT(*) FROM tips").fetchone()[0]
        api_ok = bool(_api_get(FOOTBALL_API_URL, {"live": "all"}) is not None)
        return jsonify({"ok": True, "db":"ok", "tips_count": int(n), "api_connected": api_ok, "scheduler_running": bool(_s)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# ──────────────────────────────────────────────────────────────────────────────
# Boot / Shutdown
# ──────────────────────────────────────────────────────────────────────────────

def _on_boot():
    _init_pool()
    init_db()
    start_scheduler()

def _shutdown(*args, **kwargs):
    try:
        if _s:
            _s.shutdown(wait=False)
    except Exception:
        pass
    try:
        if POOL:
            POOL.closeall()
    except Exception:
        pass

signal.signal(signal.SIGINT, lambda s,f: (_shutdown(), sys.exit(0)))
signal.signal(signal.SIGTERM, lambda s,f: (_shutdown(), sys.exit(0)))
atexit.register(_shutdown)

_on_boot()

if __name__ == "__main__":
    app.run(host=os.getenv("HOST","0.0.0.0"), port=int(os.getenv("PORT","8080")))
