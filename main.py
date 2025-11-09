# main.py — goalsniper OU 2.5 only + MOTD + auto-train (Railway-ready, no sleep gating)

import os, sys, time, json, logging, requests, atexit, signal
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import psycopg2
from psycopg2.pool import SimpleConnectionPool
from flask import Flask, jsonify, request, abort
from zoneinfo import ZoneInfo
from html import escape

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ─────────────────────────── Env bootstrap ───────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Optional: Sentry
SENTRY_DSN = os.getenv("SENTRY_DSN")
if SENTRY_DSN:
    try:
        import sentry_sdk  # type: ignore
        sentry_sdk.init(
            dsn=SENTRY_DSN,
            traces_sample_rate=float(os.getenv("SENTRY_TRACES", "0.0")),
        )
    except Exception:
        pass  # optional

# ─────────────────────────── Required env (fail fast) ───────────────────────────
def _require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise SystemExit(f"Missing required environment variable: {name}")
    return v

# Core secrets
TELEGRAM_BOT_TOKEN = _require_env("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = _require_env("TELEGRAM_CHAT_ID")
API_KEY            = _require_env("API_KEY")  # API-Football
DATABASE_URL       = _require_env("DATABASE_URL")  # e.g., Railway Postgres

# Optional admin & webhooks
ADMIN_API_KEY  = os.getenv("ADMIN_API_KEY")
WEBHOOK_SECRET = os.getenv("TELEGRAM_WEBHOOK_SECRET")

if not ADMIN_API_KEY:
    print("[WARN] ADMIN_API_KEY is not set — /admin/* endpoints are less protected.", file=sys.stderr)
if not WEBHOOK_SECRET:
    print("[WARN] TELEGRAM_WEBHOOK_SECRET is not set — /telegram/webhook/<secret> would be unsafe.", file=sys.stderr)

# ─────────────────────────── App / logging ───────────────────────────
class CustomFormatter(logging.Formatter):
    def format(self, record):
        if not hasattr(record, "job_id"):
            record.job_id = "main"
        return super().format(record)

handler = logging.StreamHandler()
formatter = CustomFormatter("[%(asctime)s] %(levelname)s [%(job_id)s] - %(message)s")
handler.setFormatter(formatter)
log = logging.getLogger("goalsniper")
log.handlers = [handler]
log.setLevel(logging.INFO)
log.propagate = False

app = Flask(__name__)

# ─────────────────────────── Minimal metrics ───────────────────────────
METRICS = {
    "api_calls_total": defaultdict(int),
    "api_rate_limited_total": 0,
    "tips_generated_total": 0,
    "tips_sent_total": 0,
    "db_errors_total": 0,
    "job_duration_seconds": defaultdict(list)
}
def _metric_inc(name: str, label: Optional[str] = None, n: int = 1):
    try:
        if label is None:
            if isinstance(METRICS.get(name), int):
                METRICS[name] += n  # type: ignore
            else:
                METRICS[name][None] += n
        else:
            METRICS[name][label] += n
    except Exception:
        pass

def _metric_obs_duration(job: str, t0: float):
    try:
        arr = METRICS["job_duration_seconds"][job]
        arr.append(time.time() - t0)
        if len(arr) > 50:
            METRICS["job_duration_seconds"][job] = arr[-50:]
    except Exception:
        pass

# ─────────────────────────── Config knobs (OU 2.5 only) ───────────────────────────
RUN_SCHEDULER     = os.getenv("RUN_SCHEDULER", "1") not in ("0","false","False","no","NO")
CONF_THRESHOLD    = float(os.getenv("CONF_THRESHOLD", "72"))   # fallback threshold %
EDGE_MIN_BPS      = int(os.getenv("EDGE_MIN_BPS", "600"))      # 600 bps = +6% EV
MAX_TIPS_PER_SCAN = int(os.getenv("MAX_TIPS_PER_SCAN", "30"))
DUP_COOLDOWN_MIN  = int(os.getenv("DUP_COOLDOWN_MIN", "20"))
TIP_MIN_MINUTE    = int(os.getenv("TIP_MIN_MINUTE", "12"))
TIP_MAX_MINUTE_ENV= os.getenv("TIP_MAX_MINUTE", "")
try:
    TIP_MAX_MINUTE = int(float(TIP_MAX_MINUTE_ENV)) if TIP_MAX_MINUTE_ENV.strip() else None
except Exception:
    TIP_MAX_MINUTE = None

PREDICTIONS_PER_MATCH = int(os.getenv("PREDICTIONS_PER_MATCH", "1"))
PER_LEAGUE_CAP        = int(os.getenv("PER_LEAGUE_CAP", "2"))
TOTAL_MATCH_MINUTES   = int(os.getenv("TOTAL_MATCH_MINUTES", "95"))
SCAN_INTERVAL_SEC     = int(os.getenv("SCAN_INTERVAL_SEC", "300"))

# Odds gates
MIN_ODDS_OU    = float(os.getenv("MIN_ODDS_OU", "1.50"))
MAX_ODDS_ALL   = float(os.getenv("MAX_ODDS_ALL", "20.0"))
ALLOW_TIPS_WITHOUT_ODDS = os.getenv("ALLOW_TIPS_WITHOUT_ODDS","0") not in ("0","false","False","no","NO")

# Odds aggregation
ODDS_SOURCE           = os.getenv("ODDS_SOURCE","auto").lower()     # auto|live|prematch
ODDS_AGGREGATION      = os.getenv("ODDS_AGGREGATION","median")       # median|best
ODDS_OUTLIER_MULT     = float(os.getenv("ODDS_OUTLIER_MULT","1.8"))
ODDS_REQUIRE_N_BOOKS  = int(os.getenv("ODDS_REQUIRE_N_BOOKS","2"))
ODDS_FAIR_MAX_MULT    = float(os.getenv("ODDS_FAIR_MAX_MULT","2.5"))

# Stale & quality guards
STALE_GUARD_ENABLE    = os.getenv("STALE_GUARD_ENABLE","1") not in ("0","false","False","no","NO")
STALE_STATS_MAX_SEC   = int(os.getenv("STALE_STATS_MAX_SEC","240"))
REQUIRE_STATS_MINUTE  = int(os.getenv("REQUIRE_STATS_MINUTE","35"))
REQUIRE_DATA_FIELDS   = int(os.getenv("REQUIRE_DATA_FIELDS","2"))
LEAGUE_DENY_IDS       = os.getenv("LEAGUE_DENY_IDS","")

# Scheduler extras
BACKFILL_EVERY_MIN    = int(os.getenv("BACKFILL_EVERY_MIN","15"))
DAILY_ACCURACY_DIGEST_ENABLE = os.getenv("DAILY_ACCURACY_DIGEST_ENABLE","1") not in ("0","false","False","no","NO")
DAILY_ACCURACY_HOUR   = int(os.getenv("DAILY_ACCURACY_HOUR","3"))
DAILY_ACCURACY_MINUTE = int(os.getenv("DAILY_ACCURACY_MINUTE","6"))

# Auto-train
TRAIN_ENABLE     = os.getenv("TRAIN_ENABLE","1") not in ("0","false","False","no","NO")
TRAIN_HOUR_UTC   = int(os.getenv("TRAIN_HOUR_UTC","2"))
TRAIN_MINUTE_UTC = int(os.getenv("TRAIN_MINUTE_UTC","12"))

# MOTD time (Berlin timezone) — 10:15 by default
MOTD_HOUR   = int(os.getenv("MOTD_HOUR","10"))
MOTD_MINUTE = int(os.getenv("MOTD_MINUTE","15"))

# Timezones
TZ_UTC   = ZoneInfo("UTC")
BERLIN_TZ = ZoneInfo("Europe/Berlin")

# ─────────────────────────── External APIs ───────────────────────────
BASE_URL         = "https://v3.football.api-sports.io"
FOOTBALL_API_URL = f"{BASE_URL}/fixtures"
HEADERS          = {"x-apisports-key": API_KEY, "Accept": "application/json"}
INPLAY_STATUSES  = {"1H","HT","2H","ET","BT","P"}  # live-ish

session = requests.Session()
session.mount(
    "https://",
    HTTPAdapter(
        max_retries=Retry(total=3, backoff_factor=1, status_forcelist=[429,500,502,503,504], respect_retry_after_header=True)
    )
)

REQ_TIMEOUT_SEC  = float(os.getenv("REQ_TIMEOUT_SEC","8.0"))

# ─────────────────────────── DB pool ───────────────────────────
POOL: Optional[SimpleConnectionPool] = None

def _normalize_dsn(url: str) -> str:
    if not url:
        return url
    dsn = url.strip()
    if "sslmode=" not in dsn:
        dsn += ("&" if "?" in dsn else "?") + "sslmode=require"
    return dsn

def _init_pool():
    global POOL
    if POOL:
        return
    try:
        maxconn = int(os.getenv("PG_MAXCONN","10"))
        timeout = int(os.getenv("PG_CONNECT_TIMEOUT","10"))
        dsn = _normalize_dsn(DATABASE_URL)
        POOL = SimpleConnectionPool(minconn=1, maxconn=maxconn, dsn=dsn, connect_timeout=timeout, application_name="goalsniper_ou25")
        log.info("[DB] pool ready (maxconn=%s, timeout=%ss)", maxconn, timeout)
    except Exception as e:
        log.error("[DB] pool init failed: %s", e)
        raise

class PooledConn:
    def __init__(self, pool): self.pool=pool; self.conn=None; self.cur=None
    def __enter__(self):
        self.conn=self.pool.getconn(); self.conn.autocommit=True; self.cur=self.conn.cursor(); return self
    def __exit__(self, exc_type, exc, tb):
        try:
            if self.cur: self.cur.close()
        finally:
            if self.conn:
                try: self.pool.putconn(self.conn)
                except Exception:
                    try: self.conn.close()
                    except: pass
    def execute(self, sql: str, params: tuple|list=()):
        try:
            self.cur.execute(sql, params or ())
            return self.cur
        except Exception as e:
            _metric_inc("db_errors_total", n=1)
            log.error("DB execute failed: %s\nSQL: %s\nParams: %s", e, sql, params)
            raise

def db_conn(): 
    global POOL
    if not POOL: _init_pool()
    return PooledConn(POOL)  # type: ignore

def _db_ping() -> bool:
    try:
        with db_conn() as c:
            c.execute("SELECT 1")
            return True
    except Exception as e:
        log.warning("[DB] ping failed: %s", e)
        try:
            _init_pool()
            with db_conn() as c2:
                c2.execute("SELECT 1")
                return True
        except Exception as e2:
            log.error("[DB] reinit failed: %s", e2)
            return False

# ─────────────────────────── Tables ───────────────────────────
def init_db():
    with db_conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS tips (
            match_id BIGINT, league_id BIGINT, league TEXT,
            home TEXT, away TEXT, market TEXT, suggestion TEXT,
            confidence DOUBLE PRECISION, confidence_raw DOUBLE PRECISION,
            score_at_tip TEXT, minute INTEGER, created_ts BIGINT,
            odds DOUBLE PRECISION, book TEXT, ev_pct DOUBLE PRECISION,
            sent_ok INTEGER DEFAULT 1,
            PRIMARY KEY (match_id, created_ts))""")
        c.execute("""CREATE TABLE IF NOT EXISTS tip_snapshots (
            match_id BIGINT, created_ts BIGINT, payload TEXT,
            PRIMARY KEY (match_id, created_ts))""")
        c.execute("""CREATE TABLE IF NOT EXISTS feedback (
            id SERIAL PRIMARY KEY, match_id BIGINT UNIQUE, verdict INTEGER, created_ts BIGINT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS match_results (
            match_id BIGINT PRIMARY KEY, final_goals_h INTEGER, final_goals_a INTEGER, btts_yes INTEGER, updated_ts BIGINT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS odds_history (
            match_id BIGINT,
            captured_ts BIGINT,
            market TEXT,
            selection TEXT,
            odds DOUBLE PRECISION,
            book TEXT,
            PRIMARY KEY (match_id, market, selection, captured_ts)
        )""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_tips_created ON tips (created_ts DESC)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_tips_match ON tips (match_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_tips_sent ON tips (sent_ok, created_ts DESC)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_results_updated ON match_results (updated_ts DESC)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_odds_hist_match ON odds_history (match_id, captured_ts DESC)")

# ─────────────────────────── Settings cache ───────────────────────────
class _KVCache:
    def __init__(self, ttl: int): self.ttl=ttl; self.data={}
    def get(self, k): 
        v=self.data.get(k)
        if not v: return None
        ts,val=v
        if time.time()-ts>self.ttl: self.data.pop(k,None); return None
        return val
    def set(self,k,v): self.data[k]=(time.time(),v)
    def invalidate(self,k=None): self.data.clear() if k is None else self.data.pop(k,None)

_SETTINGS_CACHE, _MODELS_CACHE = _KVCache(60), _KVCache(120)

def set_setting(key: str, value: str) -> None:
    with db_conn() as c:
        c.execute("INSERT INTO settings(key,value) VALUES(%s,%s) ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value", (key,value))
        _SETTINGS_CACHE.invalidate(key)

def get_setting(key: str) -> Optional[str]:
    with db_conn() as c:
        row = c.execute("SELECT value FROM settings WHERE key=%s", (key,)).fetchone()
    return (row[0] if row and len(row)>0 else None)

def get_setting_cached(key: str) -> Optional[str]:
    v=_SETTINGS_CACHE.get(key)
    if v is None:
        v=get_setting(key)
        _SETTINGS_CACHE.set(key,v)
    return v

def invalidate_model_caches_for_key(key: str):
    if key.lower().startswith(("model","model_latest","model_v2","pre_")):
        _MODELS_CACHE.invalidate(key)

# ─────────────────────────── Helpers (API/HTTP) ───────────────────────────
def _api_get(url: str, params: dict, timeout: int = 15):
    lbl = "unknown"
    if "/odds" in url: lbl="odds"
    elif "/statistics" in url: lbl="statistics"
    elif "/events" in url: lbl="events"
    elif "/fixtures" in url: lbl="fixtures"
    try:
        r = session.get(url, headers=HEADERS, params=params, timeout=min(timeout, REQ_TIMEOUT_SEC))
        _metric_inc("api_calls_total", label=lbl, n=1)
        if not r.ok:
            if r.status_code == 429:
                METRICS["api_rate_limited_total"] += 1
            return None
        return r.json()
    except Exception:
        return None

# ─────────────────────────── Odds ───────────────────────────
ODDS_CACHE: Dict[int, Tuple[float, dict]] = {}
def _market_name_normalize(s: str) -> str:
    s=(s or "").lower()
    if "over/under" in s or "total" in s or "goals" in s: return "OU"
    return s

def _aggregate_price(vals: List[Tuple[float, str]], prob_hint: Optional[float]) -> Tuple[Optional[float], Optional[str]]:
    if not vals: return None, None
    xs = sorted([o for (o,_) in vals if (o or 0) > 0])
    if not xs: return None, None
    import statistics
    med = statistics.median(xs)
    filtered = [(o,b) for (o,b) in vals if o <= med * max(1.0, ODDS_OUTLIER_MULT)] or vals
    xs2 = sorted([o for (o,_) in filtered])
    med2 = statistics.median(xs2)
    if prob_hint is not None and prob_hint > 0:
        fair = 1.0 / max(1e-6, float(prob_hint))
        cap  = fair * max(1.0, ODDS_FAIR_MAX_MULT)
        filtered = [(o,b) for (o,b) in filtered if o <= cap] or filtered
    if ODDS_AGGREGATION == "best":
        best = max(filtered, key=lambda t: t[0])
        return float(best[0]), str(best[1])
    target = med2
    pick = min(filtered, key=lambda t: abs(t[0]-target))
    return float(pick[0]), f"{pick[1]} (median of {len(xs)})"

def fetch_odds_ou25(fid: int, prob_hint_over: Optional[float] = None) -> Dict[str, Dict[str, Any]]:
    now=time.time()
    rec=ODDS_CACHE.get(fid)
    if rec and now-rec[0] < 120:
        return rec[1]
    # Prefer live odds, fallback to prematch
    payload = {}
    for path in ("odds/live","odds"):
        js = _api_get(f"{BASE_URL}/{path}", {"fixture": fid}) or {}
        for r in (js.get("response") or []):
            for bk in (r.get("bookmakers") or []):
                bname = bk.get("name") or "Book"
                for mkt in (bk.get("bets") or []):
                    if _market_name_normalize(mkt.get("name","")) != "ou":
                        continue
                    for v in (mkt.get("values") or []):
                        lbl = (v.get("value") or "").lower()
                        if ("over" in lbl or "under" in lbl):
                            try:
                                ln = float(lbl.split()[-1])
                                if abs(ln-2.5)<1e-6:
                                    side = "Over" if "over" in lbl else "Under"
                                    payload.setdefault("OU_2.5",{}).setdefault(side,[]).append((float(v.get("odd") or 0), bname))
                            except:
                                pass
        if payload:
            break

    out={}
    sides = (payload.get("OU_2.5") or {})
    if len({b for lst in sides.values() for (_,b) in lst}) >= max(1, ODDS_REQUIRE_N_BOOKS):
        out["OU_2.5"]={}
        for side,lst in sides.items():
            hint = (prob_hint_over if side=="Over" else (1.0 - (prob_hint_over or 0.0))) if prob_hint_over is not None else None
            agg, book = _aggregate_price(lst, hint)
            if agg is not None:
                out["OU_2.5"][side]={"odds": float(agg), "book": book}
    ODDS_CACHE[fid]=(time.time(), out)
    return out

# ─────────────────────────── Features (live + prematch) ───────────────────────────
def _num(v) -> float:
    try:
        if isinstance(v,str) and v.endswith("%"): return float(v[:-1])
        return float(v or 0)
    except: return 0.0

def _pos_pct(v) -> float:
    try: return float(str(v).replace("%","").strip() or 0)
    except: return 0.0

def extract_basic_features(m: dict) -> Dict[str,float]:
    home = (m.get("teams") or {}).get("home",{}).get("name","")
    away = (m.get("teams") or {}).get("away",{}).get("name","")
    gh = int((m.get("goals") or {}).get("home") or 0)
    ga = int((m.get("goals") or {}).get("away") or 0)
    minute = int(((m.get("fixture") or {}).get("status") or {}).get("elapsed") or 0)

    stats_by_team={}
    for s in (m.get("statistics") or []):
        t=(s.get("team") or {}).get("name")
        if t: stats_by_team[t] = { (i.get("type") or ""): i.get("value") for i in (s.get("statistics") or []) }
    sh = stats_by_team.get(home, {}) or {}
    sa = stats_by_team.get(away, {}) or {}

    xg_h = _num(sh.get("Expected Goals", 0))
    xg_a = _num(sa.get("Expected Goals", 0))
    sot_h = _num(sh.get("Shots on Target", sh.get("Shots on Goal", 0)))
    sot_a = _num(sa.get("Shots on Target", sa.get("Shots on Goal", 0)))
    sh_total_h = _num(sh.get("Total Shots", sh.get("Shots Total", 0)))
    sh_total_a = _num(sa.get("Total Shots", sa.get("Shots Total", 0)))
    cor_h = _num(sh.get("Corner Kicks", 0))
    cor_a = _num(sa.get("Corner Kicks", 0))
    pos_h = _pos_pct(sh.get("Ball Possession", 0))
    pos_a = _pos_pct(sa.get("Ball Possession", 0))

    # Cards from events
    red_h=red_a=yellow_h=yellow_a=0
    for ev in (m.get("events") or []):
        if str(ev.get("type","")).lower()=="card":
            d=(ev.get("detail","") or "").lower()
            t=(ev.get("team") or {}).get("name") or ""
            if "yellow" in d and "second" not in d:
                if t==home: yellow_h+=1
                elif t==away: yellow_a+=1
            if "red" in d or "second yellow" in d:
                if t==home: red_h+=1
                elif t==away: red_a+=1

    return {
        "minute": float(minute),
        "goals_h": float(gh), "goals_a": float(ga),
        "goals_sum": float(gh+ga), "goals_diff": float(gh-ga),
        "xg_h": float(xg_h), "xg_a": float(xg_a),
        "xg_sum": float(xg_h+xg_a), "xg_diff": float(xg_h-xg_a),
        "sot_h": float(sot_h), "sot_a": float(sot_a), "sot_sum": float(sot_h+sot_a),
        "sh_total_h": float(sh_total_h), "sh_total_a": float(sh_total_a),
        "cor_h": float(cor_h), "cor_a": float(cor_a), "cor_sum": float(cor_h+cor_a),
        "pos_h": float(pos_h), "pos_a": float(pos_a), "pos_diff": float(pos_h-pos_a),
        "red_h": float(red_h), "red_a": float(red_a), "red_sum": float(red_h+red_a),
        "yellow_h": float(yellow_h), "yellow_a": float(yellow_a)
    }

def stats_coverage_ok(feat: Dict[str,float], minute: int) -> bool:
    if minute < REQUIRE_STATS_MINUTE: return True
    fields = [
        feat.get("xg_sum", 0.0),
        feat.get("sot_sum", 0.0),
        feat.get("cor_sum", 0.0),
        max(feat.get("pos_h",0.0), feat.get("pos_a",0.0)),
        (feat.get("sh_total_h",0.0) + feat.get("sh_total_a",0.0)),
        (feat.get("yellow_h",0.0) + feat.get("yellow_a",0.0)),
    ]
    nonzero = sum(1 for v in fields if (v or 0) > 0)
    return nonzero >= max(0, REQUIRE_DATA_FIELDS)

# ─────────────────────────── Models & scoring ───────────────────────────
EPS=1e-12
def _sigmoid(x: float) -> float:
    try:
        if x<-50: return 1e-22
        if x>50:  return 1-1e-22
        import math; return 1/(1+math.exp(-x))
    except: return 0.5

def _logit(p: float) -> float:
    import math
    p=max(EPS,min(1-EPS,float(p)))
    return math.log(p/(1-p))

def _linpred(feat: Dict[str,float], weights: Dict[str,float], intercept: float) -> float:
    s=float(intercept or 0.0)
    for k,w in (weights or {}).items(): s += float(w or 0.0)*float(feat.get(k,0.0))
    return s

def _calibrate(p: float, cal: Dict[str,Any]) -> float:
    method=(cal or {}).get("method","sigmoid"); a=float((cal or {}).get("a",1.0)); b=float((cal or {}).get("b",0.0))
    if method.lower()=="platt": return _sigmoid(a*_logit(p)+b)
    import math; p=max(EPS,min(1-EPS,float(p))); z=math.log(p/(1-p)); return _sigmoid(a*z+b)

def _score_prob(feat: Dict[str,float], mdl: Dict[str,Any]) -> float:
    p=_sigmoid(_linpred(feat, mdl.get("weights",{}), float(mdl.get("intercept",0.0))))
    cal=mdl.get("calibration") or {}
    try:
        if cal: p=_calibrate(p, cal)
    except: pass
    return max(0.0, min(1.0, float(p)))

MODEL_KEYS_ORDER=["model_v2:{name}","model_latest:{name}","model:{name}","pre_{name}"]
def _validate_model_blob(tmp: dict) -> bool:
    return isinstance(tmp, dict) and isinstance(tmp.get("weights"), dict) and ("intercept" in tmp)

def load_model_from_settings(name: str) -> Optional[Dict[str,Any]]:
    cached=_MODELS_CACHE.get(name)
    if cached is not None: return cached
    mdl=None
    for pat in MODEL_KEYS_ORDER:
        raw=get_setting_cached(pat.format(name=name))
        if not raw: continue
        try:
            tmp=json.loads(raw)
            if not _validate_model_blob(tmp): 
                log.warning("[MODEL] invalid schema for %s", name)
                continue
            tmp.setdefault("intercept",0.0)
            tmp.setdefault("weights",{})
            cal=tmp.get("calibration") or {}
            if isinstance(cal,dict):
                cal.setdefault("method","sigmoid"); cal.setdefault("a",1.0); cal.setdefault("b",0.0)
                tmp["calibration"]=cal
            mdl=tmp; break
        except Exception as e:
            log.warning("[MODEL] parse %s failed: %s", name, e)
    if mdl is not None:
        _MODELS_CACHE.set(name, mdl)
    return mdl

# ─────────────────────────── Live scan (OU 2.5 only) ───────────────────────────
STATS_CACHE: Dict[int, Tuple[float, list]] = {}
EVENTS_CACHE: Dict[int, Tuple[float, list]] = {}
NEG_CACHE: Dict[Tuple[str,int], Tuple[float, bool]] = {}
NEG_TTL_SEC = int(os.getenv("NEG_TTL_SEC","45"))

def fetch_match_stats(fid: int) -> list:
    now=time.time()
    k=("stats", fid)
    ts_empty = NEG_CACHE.get(k, (0.0, False))
    if ts_empty[1] and (now - ts_empty[0] < NEG_TTL_SEC): return []
    if fid in STATS_CACHE and now-STATS_CACHE[fid][0] < 90: return STATS_CACHE[fid][1]
    js=_api_get(f"{FOOTBALL_API_URL}/statistics", {"fixture": fid}) or {}
    out=js.get("response",[]) if isinstance(js,dict) else []
    STATS_CACHE[fid]=(now,out)
    if not out: NEG_CACHE[k]=(now, True)
    return out

def fetch_match_events(fid: int) -> list:
    now=time.time()
    k=("events", fid)
    ts_empty = NEG_CACHE.get(k, (0.0, False))
    if ts_empty[1] and (now - ts_empty[0] < NEG_TTL_SEC): return []
    if fid in EVENTS_CACHE and now-EVENTS_CACHE[fid][0] < 90: return EVENTS_CACHE[fid][1]
    js=_api_get(f"{FOOTBALL_API_URL}/events", {"fixture": fid}) or {}
    out=js.get("response",[]) if isinstance(js,dict) else []
    EVENTS_CACHE[fid]=(now,out)
    if not out: NEG_CACHE[k]=(now, True)
    return out

def _blocked_league(league_obj: dict) -> bool:
    name=str((league_obj or {}).get("name","")).lower()
    country=str((league_obj or {}).get("country","")).lower()
    typ=str((league_obj or {}).get("type","")).lower()
    txt=f"{country} {name} {typ}"
    patterns=["u17","u18","u19","u20","u21","u23","youth","junior","reserve","friendlies","friendly"]
    if any(p in txt for p in patterns): return True
    deny=[x.strip() for x in LEAGUE_DENY_IDS.split(",") if x.strip()]
    lid=str((league_obj or {}).get("id") or "")
    return lid in deny

def fetch_live_matches() -> List[dict]:
    js=_api_get(FOOTBALL_API_URL, {"live":"all"}) or {}
    matches=[m for m in (js.get("response",[]) if isinstance(js,dict) else []) if not _blocked_league(m.get("league") or {})]
    out=[]
    for m in matches:
        st=((m.get("fixture") or {}).get("status") or {})
        elapsed=st.get("elapsed"); short=(st.get("short") or "").upper()
        if elapsed is None or elapsed>120 or short not in INPLAY_STATUSES: 
            continue
        fid=(m.get("fixture") or {}).get("id")
        m["statistics"]=fetch_match_stats(fid); m["events"]=fetch_match_events(fid)
        out.append(m)
    return out

def _format_score(m: dict) -> str:
    gh=(m.get("goals") or {}).get("home") or 0
    ga=(m.get("goals") or {}).get("away") or 0
    return f"{gh}-{ga}"

def _league_name(m: dict) -> Tuple[int,str]:
    lg=(m.get("league") or {}) or {}
    return int(lg.get("id") or 0), f"{lg.get('country','')} - {lg.get('name','')}".strip(" -")

def _teams(m: dict) -> Tuple[str,str]:
    t=(m.get("teams") or {}) or {}
    return (t.get("home",{}).get("name",""), t.get("away",{}).get("name",""))

def _candidate_is_sane_ou25(suggestion: str, feat: Dict[str,float]) -> bool:
    # Never suggest Under/Over 2.5 if total goals already >= 3 (market settled)
    total = int(feat.get("goals_sum",0))
    if total >= 3:
        return False
    # Both Over and Under are theoretically valid for totals 0,1,2
    return True

def _ev(prob: float, odds: float) -> float:
    return prob*max(0.0,float(odds)) - 1.0

def _min_odds_for_market() -> float:
    return MIN_ODDS_OU

def _price_gate_ou25(fid: int, prob_over: float) -> Tuple[bool, Optional[float], Optional[str], Optional[float]]:
    odds_map = fetch_odds_ou25(fid, prob_hint_over=prob_over)
    odds=None; book=None
    if "OU_2.5" in odds_map:
        # Choose the better side based on prob
        side = "Over" if prob_over >= 0.5 else "Under"
        entry = odds_map["OU_2.5"].get(side)
        if entry:
            odds=float(entry["odds"]); book=str(entry["book"])
    if odds is None:
        return (True, None, None, None) if ALLOW_TIPS_WITHOUT_ODDS else (False, None, None, None)
    if not (_min_odds_for_market() <= odds <= MAX_ODDS_ALL):
        return (False, odds, book, None)
    return (True, odds, book, None)

def _get_market_threshold_ou25() -> float:
    v = get_setting_cached("conf_threshold:Over/Under 2.5")
    try:
        return float(v) if v is not None else float(CONF_THRESHOLD)
    except Exception:
        return float(CONF_THRESHOLD)

def send_telegram(text: str) -> bool:
    try:
        r=requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            data={"chat_id":TELEGRAM_CHAT_ID,"text":text,"parse_mode":"HTML","disable_web_page_preview":True},
            timeout=REQ_TIMEOUT_SEC
        )
        ok=bool(r.ok)
        if ok: _metric_inc("tips_sent_total", n=1)
        return ok
    except Exception:
        return False

_FEED_STATE: Dict[int, Dict[str, Any]] = {}

def _safe_num(x) -> float:
    try:
        if isinstance(x, str) and x.endswith("%"):
            return float(x[:-1])
        return float(x or 0.0)
    except Exception:
        return 0.0

def _match_fingerprint(m: dict) -> Tuple:
    teams = (m.get("teams") or {})
    home = (teams.get("home") or {}).get("name", "")
    away = (teams.get("away") or {}).get("name", "")

    stats_by_team = {}
    for s in (m.get("statistics") or []):
        tname = ((s.get("team") or {}).get("name") or "").strip()
        if tname:
            stats_by_team[tname] = {str((i.get("type") or "")).lower(): i.get("value") for i in (s.get("statistics") or [])}

    sh = stats_by_team.get(home, {}) or {}
    sa = stats_by_team.get(away, {}) or {}

    def g(d: dict, key_variants: Tuple[str, ...]) -> float:
        for k in key_variants:
            if k in d: return _safe_num(d[k])
        return 0.0

    xg_h = g(sh, ("expected goals",))
    xg_a = g(sa, ("expected goals",))
    sot_h = g(sh, ("shots on target","shots on goal"))
    sot_a = g(sa, ("shots on target","shots on goal"))
    sh_tot_h = g(sh, ("total shots","shots total"))
    sh_tot_a = g(sa, ("total shots","shots total"))
    cor_h = g(sh, ("corner kicks",))
    cor_a = g(sa, ("corner kicks",))
    pos_h = g(sh, ("ball possession",))
    pos_a = g(sa, ("ball possession",))

    ev = m.get("events") or []
    n_events = len(ev)
    n_cards = sum(1 for e in ev if str(e.get("type","")).lower()=="card")

    gh = int(((m.get("goals") or {}).get("home") or 0) or 0)
    ga = int(((m.get("goals") or {}).get("away") or 0) or 0)

    return (
        round(xg_h + xg_a, 3),
        int(sot_h + sot_a),
        int(sh_tot_h + sh_tot_a),
        int(cor_h + cor_a),
        int(round(pos_h)), int(round(pos_a)),
        gh, ga,
        n_events, n_cards,
    )

def is_feed_stale(fid: int, m: dict, minute: int) -> bool:
    if not STALE_GUARD_ENABLE:
        return False
    now=time.time()
    if minute < 10:
        _FEED_STATE[fid] = {"fp": _match_fingerprint(m), "last_change": now, "last_minute": minute}
        return False
    st=_FEED_STATE.get(fid)
    fp=_match_fingerprint(m)
    if st is None:
        _FEED_STATE[fid] = {"fp": fp, "last_change": now, "last_minute": minute}
        return False
    if fp != st.get("fp"):
        st["fp"]=fp; st["last_change"]=now; st["last_minute"]=minute
        return False
    last_min=int(st.get("last_minute") or 0)
    st["last_minute"]=minute
    if minute > last_min and (now - float(st.get("last_change") or now)) >= STALE_STATS_MAX_SEC:
        return True
    return False

def production_scan_ou25() -> Tuple[int,int]:
    t0=time.time()
    if not _db_ping():
        log.error("[SCAN] DB unavailable")
        return (0,0)
    try:
        matches = fetch_live_matches()
    except Exception as e:
        log.error("[SCAN] fetch live failed: %s", e)
        return (0,0)

    live_seen=len(matches)
    saved=0
    now_ts=int(time.time())

    with db_conn() as c:
        for m in matches:
            try:
                fid=int(((m.get("fixture") or {}).get("id") or 0))
                if not fid: continue

                # duplicate cooldown
                if DUP_COOLDOWN_MIN>0:
                    cutoff=now_ts - DUP_COOLDOWN_MIN*60
                    row = c.execute(
                        "SELECT 1 FROM tips WHERE match_id=%s AND created_ts>=%s AND market='Over/Under 2.5' LIMIT 1",
                        (fid, cutoff)
                    ).fetchone()
                    if row is not None and len(row)>0:
                        continue

                feat = extract_basic_features(m)
                minute = int(feat.get("minute",0))
                if minute < TIP_MIN_MINUTE:
                    continue
                if TIP_MAX_MINUTE is not None and minute > TIP_MAX_MINUTE:
                    continue
                if is_feed_stale(fid, m, minute):
                    continue
                if not stats_coverage_ok(feat, minute):
                    continue

                mdl = load_model_from_settings("OU_2.5")
                if not mdl:
                    log.debug("[SCAN] model OU_2.5 missing")
                    continue
                p_over = _score_prob(feat, mdl)
                # Build two candidates
                candidates = [
                    ("Over/Under 2.5", "Over 2.5 Goals", p_over),
                    ("Over/Under 2.5", "Under 2.5 Goals", 1.0 - p_over),
                ]

                # threshold
                threshold = _get_market_threshold_ou25()

                ranked=[]
                for market, sugg, prob in candidates:
                    if prob*100.0 < threshold:
                        continue
                    if not _candidate_is_sane_ou25(sugg, feat):
                        continue

                    # price + EV gate
                    pass_odds, odds, book, _ = _price_gate_ou25(fid, p_over)
                    if not pass_odds:
                        continue

                    ev_pct=None
                    if odds is not None:
                        edge=_ev(prob, odds)
                        ev_pct=round(edge*100.0, 1)
                        if int(round(edge*10000)) < EDGE_MIN_BPS:
                            continue
                    elif not ALLOW_TIPS_WITHOUT_ODDS:
                        continue

                    # rank = prob * (1+EV) * minute factor
                    rank = (prob**1.2) * (1 + ((ev_pct or 0)/100.0)) * (0.8 + 0.2*min(1.0, minute/60.0))
                    ranked.append((market, sugg, prob, odds, book, ev_pct, rank))

                if not ranked:
                    continue
                ranked.sort(key=lambda x: x[6], reverse=True)

                league_id, league = _league_name(m)
                home, away = _teams(m)
                score = _format_score(m)

                per_match = 0
                base_now = int(time.time())
                for idx,(market,sugg,prob,odds,book,ev_pct,_rank) in enumerate(ranked):
                    if per_match >= max(1, PREDICTIONS_PER_MATCH): break

                    created_ts = base_now + idx
                    pct = round(prob*100.0, 1)

                    # insert + send
                    try:
                        c.execute(
                            "INSERT INTO tips(match_id,league_id,league,home,away,market,suggestion,confidence,confidence_raw,score_at_tip,minute,created_ts,odds,book,ev_pct,sent_ok) "
                            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,0)",
                            (fid, league_id, league, home, away,
                             market, sugg, float(pct), float(prob), score, minute, created_ts,
                             (float(odds) if odds is not None else None),
                             (book or None),
                             (float(ev_pct) if ev_pct is not None else None))
                        )
                    except Exception as e:
                        log.warning("[SCAN] insert failed: %s", e)
                        continue

                    msg=_format_tip_message(home, away, league, minute, score, sugg, pct, feat, odds, book, ev_pct)
                    sent = send_telegram(msg)
                    if sent:
                        c.execute("UPDATE tips SET sent_ok=1 WHERE match_id=%s AND created_ts=%s", (fid, created_ts))
                        _metric_inc("tips_sent_total", n=1)

                    saved += 1
                    per_match += 1
                    if MAX_TIPS_PER_SCAN and saved >= MAX_TIPS_PER_SCAN:
                        break
                if MAX_TIPS_PER_SCAN and saved >= MAX_TIPS_PER_SCAN:
                    break

            except Exception as e:
                log.exception("[SCAN] match loop error: %s", e)
                continue

    log.info("[SCAN] ou2.5 saved=%d live_seen=%d", saved, live_seen)
    _metric_inc("tips_generated_total", n=saved)
    _metric_obs_duration("scan", t0)
    return saved, live_seen

def _format_tip_message(home, away, league, minute, score, suggestion, prob_pct, feat, odds=None, book=None, ev_pct=None):
    stat=""
    if any([feat.get("xg_h",0),feat.get("xg_a",0),feat.get("sot_h",0),feat.get("sot_a",0),
            feat.get("cor_h",0),feat.get("cor_a",0),feat.get("pos_h",0),feat.get("pos_a",0),
            feat.get("red_h",0),feat.get("red_a",0)]):
        stat=(f"\n📊 xG {feat.get('xg_h',0):.2f}-{feat.get('xg_a',0):.2f}"
              f" • SOT {int(feat.get('sot_h',0))}-{int(feat.get('sot_a',0))}"
              f" • CK {int(feat.get('cor_h',0))}-{int(feat.get('cor_a',0))}")
        if feat.get("pos_h",0) or feat.get("pos_a",0):
            stat += f" • POS {int(feat.get('pos_h',0))}%–{int(feat.get('pos_a',0))}%"
        if feat.get("red_h",0) or feat.get("red_a",0):
            stat += f" • RED {int(feat.get('red_h',0))}-{int(feat.get('red_a',0))}"
    money=""
    if odds:
        if ev_pct is not None:
            money = f"\n💰 <b>Odds:</b> {odds:.2f} @ {book or 'Book'}  •  <b>EV:</b> {ev_pct:+.1f}%"
        else:
            money = f"\n💰 <b>Odds:</b> {odds:.2f} @ {book or 'Book'}"
    return ("⚽️ <b>OU 2.5 Tip</b>\n"
            f"<b>Match:</b> {escape(home)} vs {escape(away)}\n"
            f"🕒 <b>Minute:</b> {minute}'  |  <b>Score:</b> {escape(score)}\n"
            f"<b>Tip:</b> {escape(suggestion)}\n"
            f"📈 <b>Confidence:</b> {prob_pct:.1f}%{money}\n"
            f"🏆 <b>League:</b> {escape(league)}{stat}")

# ─────────────────────────── Results backfill & digest ───────────────────────────
def _fixture_by_id(mid: int) -> Optional[dict]:
    js=_api_get(FOOTBALL_API_URL, {"id": mid}) or {}
    arr=js.get("response") or [] if isinstance(js,dict) else []
    return arr[0] if arr else None

def _is_final(short: str) -> bool: return (short or "").upper() in {"FT","AET","PEN"}

def backfill_results_for_open_matches(max_rows: int = 200) -> int:
    now_ts=int(time.time()); cutoff=now_ts - 14*24*3600; updated=0
    with db_conn() as c:
        rows=c.execute("""
            WITH last AS (SELECT match_id, MAX(created_ts) last_ts FROM tips WHERE created_ts >= %s GROUP BY match_id)
            SELECT l.match_id FROM last l LEFT JOIN match_results r ON r.match_id=l.match_id
            WHERE r.match_id IS NULL ORDER BY l.last_ts DESC LIMIT %s
        """,(cutoff, max_rows)).fetchall()
    for (mid,) in rows:
        fx=_fixture_by_id(int(mid))
        if not fx: continue
        st=(((fx.get("fixture") or {}).get("status") or {}).get("short") or "")
        if not _is_final(st): continue
        g=fx.get("goals") or {}; gh=int(g.get("home") or 0); ga=int(g.get("away") or 0)
        btts=1 if (gh>0 and ga>0) else 0
        with db_conn() as c2:
            c2.execute("INSERT INTO match_results(match_id, final_goals_h, final_goals_a, btts_yes, updated_ts) "
                       "VALUES(%s,%s,%s,%s,%s) ON CONFLICT(match_id) DO UPDATE SET final_goals_h=EXCLUDED.final_goals_h, "
                       "final_goals_a=EXCLUDED.final_goals_a, btts_yes=EXCLUDED.btts_yes, updated_ts=EXCLUDED.updated_ts",
                       (int(mid), gh, ga, btts, int(time.time())))
        updated+=1
    if updated: log.info("[RESULTS] backfilled %d", updated)
    return updated

def _tip_outcome_for_result_ou25(suggestion: str, gh: int, ga: int) -> Optional[int]:
    total = int(gh) + int(ga)
    if "Over 2.5" in suggestion:
        if total > 2.5: return 1
        if abs(total-2.5)<1e-9: return None
        return 0
    if "Under 2.5" in suggestion:
        if total < 2.5: return 1
        if abs(total-2.5)<1e-9: return None
        return 0
    return None

def daily_accuracy_digest() -> Optional[str]:
    if not DAILY_ACCURACY_DIGEST_ENABLE:
        return None
    today = datetime.now(BERLIN_TZ).date()
    start_ts = int(datetime.combine(today, datetime.min.time(), tzinfo=BERLIN_TZ).timestamp())

    backfill_results_for_open_matches(200)

    with db_conn() as c:
        rows = c.execute("""
            SELECT t.market, t.suggestion, t.confidence, t.confidence_raw, t.created_ts,
                   t.odds, r.final_goals_h, r.final_goals_a
            FROM tips t LEFT JOIN match_results r ON r.match_id=t.match_id
            WHERE t.created_ts >= %s 
            AND t.suggestion<>'HARVEST' 
            AND t.sent_ok=1
            ORDER BY t.created_ts DESC
        """, (start_ts,)).fetchall()

    total=graded=wins=0
    roi = {"stake":0.0,"pnl":0.0}
    recent=[]

    for (mkt, sugg, conf, conf_raw, cts, odds, gh, ga) in rows:
        gh=int(gh or 0); ga=int(ga or 0)
        out=_tip_outcome_for_result_ou25(sugg, gh, ga)
        tm=datetime.fromtimestamp(int(cts), BERLIN_TZ).strftime("%H:%M")
        recent.append(f"{sugg} ({float(conf):.1f}%) - {tm}")

        if out is None: 
            continue
        total += 1; graded += 1; wins += 1 if out==1 else 0
        if odds:
            roi["stake"] += 1
            if out==1: roi["pnl"] += float(odds) - 1
            else: roi["pnl"] -= 1

    if graded==0:
        msg=f"📊 Daily Accuracy Digest - {today.isoformat()}\nNo graded tips today."
    else:
        acc=100.0 * wins/max(1,graded)
        roi_txt=""
        if roi["stake"]>0:
            roi_val=100.0 * roi["pnl"]/roi["stake"]
            roi_txt=f" • ROI {roi_val:+.1f}%"
        msg=(f"📊 Daily Accuracy Digest - {today.isoformat()}\n"
             f"Tips sent: {total}  •  Graded: {graded}  •  Wins: {wins}  •  Accuracy: {acc:.1f}%{roi_txt}")
        if recent:
            msg += f"\n🕒 Recent tips: {', '.join(recent[:3])}"

    send_telegram(msg)
    return msg

# ─────────────────────────── Prematch features & MOTD (OU 2.5) ───────────────────────────
def _collect_todays_prematch_fixtures() -> List[dict]:
    today_local=datetime.now(BERLIN_TZ).date()
    start_local=datetime.combine(today_local, datetime.min.time(), tzinfo=BERLIN_TZ)
    end_local=start_local+timedelta(days=1)
    dates_utc={start_local.astimezone(ZoneInfo("UTC")).date(), (end_local - timedelta(seconds=1)).astimezone(ZoneInfo("UTC")).date()}
    fixtures=[]
    for d in sorted(dates_utc):
        js=_api_get(FOOTBALL_API_URL, {"date": d.strftime("%Y-%m-%d")}) or {}
        for r in js.get("response",[]) if isinstance(js,dict) else []:
            if (((r.get("fixture") or {}).get("status") or {}).get("short") or "").upper() == "NS":
                if not _blocked_league(r.get("league") or {}):
                    fixtures.append(r)
    return fixtures

def extract_prematch_features(fx: dict) -> Dict[str, float]:
    # Minimal prematch: last x goals averages & rest days & h2h goals avg
    feat = {}
    home_id = ((fx.get("teams") or {}).get("home") or {}).get("id")
    away_id = ((fx.get("teams") or {}).get("away") or {}).get("id")

    def _last(team_id: int, n: int=5) -> List[dict]:
        js=_api_get(f"{BASE_URL}/fixtures", {"team":team_id,"last":n}) or {}
        return js.get("response",[]) if isinstance(js,dict) else []

    def _h2h(hid:int, aid:int, n:int=5) -> List[dict]:
        js=_api_get(f"{BASE_URL}/fixtures/headtohead", {"h2h":f"{hid}-{aid}","last":n}) or {}
        return js.get("response",[]) if isinstance(js,dict) else []

    recent_h=_last(int(home_id or 0), 5) if home_id else []
    recent_a=_last(int(away_id or 0), 5) if away_id else []
    h2h = _h2h(int(home_id or 0), int(away_id or 0), 5) if (home_id and away_id) else []

    if recent_h:
        feat["avg_goals_h"] = np.mean([(m.get("goals") or {}).get("home",0) for m in recent_h])
    if recent_a:
        feat["avg_goals_a"] = np.mean([(m.get("goals") or {}).get("away",0) for m in recent_a])
    if h2h:
        feat["avg_goals_h2h"] = np.mean([(m.get("goals") or {}).get("home",0)+(m.get("goals") or {}).get("away",0) for m in h2h])

    try:
        dts_h=[datetime.fromisoformat((m.get("fixture") or {}).get("date","")) for m in recent_h]
        dts_a=[datetime.fromisoformat((m.get("fixture") or {}).get("date","")) for m in recent_a]
        if dts_h: feat["rest_days_h"] = (datetime.now(tz=TZ_UTC) - max(dts_h).astimezone(TZ_UTC)).days
        if dts_a: feat["rest_days_a"] = (datetime.now(tz=TZ_UTC) - max(dts_a).astimezone(TZ_UTC)).days
    except Exception:
        pass

    # Fixture id for odds snapshot
    feat["fid"] = float(((fx.get("fixture") or {}).get("id") or 0) or 0)
    return feat

def _kickoff_berlin(utc_iso: str|None) -> str:
    try:
        if not utc_iso: return "TBD"
        dt=datetime.fromisoformat(utc_iso.replace("Z","+00:00"))
        return dt.astimezone(BERLIN_TZ).strftime("%H:%M")
    except: return "TBD"

def _format_motd_message(home, away, league, kickoff_txt, suggestion, prob_pct, odds=None, book=None, ev_pct=None):
    money=""
    if odds:
        if ev_pct is not None:
            money=f"\n💰 <b>Odds:</b> {odds:.2f} @ {book or 'Book'}  •  <b>EV:</b> {ev_pct:+.1f}%"
        else:
            money=f"\n💰 <b>Odds:</b> {odds:.2f} @ {book or 'Book'}"
    return ("🏅 <b>Match of the Day (OU 2.5)</b>\n"
            f"<b>Match:</b> {escape(home)} vs {escape(away)}\n"
            f"🏆 <b>League:</b> {escape(league)}\n"
            f"⏰ <b>Kickoff (Berlin):</b> {kickoff_txt}\n"
            f"<b>Tip:</b> {escape(suggestion)}\n"
            f"📈 <b>Confidence:</b> {prob_pct:.1f}%{money}")

def send_match_of_the_day() -> bool:
    fixtures = _collect_todays_prematch_fixtures()
    if not fixtures:
        return send_telegram("🏅 Match of the Day: no fixtures today.")

    best=None; best_score=-1e9
    threshold = _get_market_threshold_ou25()

    for fx in fixtures:
        try:
            fixture=fx.get("fixture") or {}
            lg      =fx.get("league")  or {}
            teams   =fx.get("teams")   or {}
            fid=int((fixture.get("id") or 0))
            home=(teams.get("home") or {}).get("name","")
            away=(teams.get("away") or {}).get("name","")
            league=f"{lg.get('country','')} - {lg.get('name','')}".strip(" -")
            kickoff_txt=_kickoff_berlin((fixture.get("date") or ""))

            feat=extract_prematch_features(fx)
            mdl_pre = load_model_from_settings("PRE_OU_2.5")
            if mdl_pre:
                p_over = _score_prob(feat, mdl_pre)
            else:
                # If no prematch model, skip (we only do prematch at 10:15)
                continue

            # Build both sides, apply threshold
            cand=[]
            if p_over*100.0 >= threshold:
                cand.append(("Over 2.5 Goals", p_over))
            if (1.0-p_over)*100.0 >= threshold:
                cand.append(("Under 2.5 Goals", 1.0 - p_over))
            if not cand: 
                continue

            # Odds + EV
            odds_map = fetch_odds_ou25(fid, prob_hint_over=p_over)
            for sugg,prob in cand:
                entry = (odds_map.get("OU_2.5") or {}).get("Over" if "Over" in sugg else "Under")
                if not entry:
                    if not ALLOW_TIPS_WITHOUT_ODDS:
                        continue
                    odds=None; book=None; ev_pct=None
                    score = prob*100.0  # rank by confidence only
                else:
                    odds=float(entry["odds"]); book=str(entry["book"])
                    if not (MIN_ODDS_OU <= odds <= MAX_ODDS_ALL):
                        continue
                    edge=_ev(prob, odds); ev_pct=round(edge*100.0,1)
                    if int(round(edge*10000)) < EDGE_MIN_BPS:
                        continue
                    score = (prob*100.0) + ev_pct

                if score > best_score:
                    best_score=score; best=(home,away,league,kickoff_txt,sugg,prob*100.0,odds,book, (ev_pct if entry else None))

        except Exception as e:
            log.warning("[MOTD] fixture eval failed: %s", e)
            continue

    if not best:
        return send_telegram("🏅 Match of the Day: no prematch pick met thresholds today.")

    home,away,league,kickoff_txt,sugg,prob_pct,odds,book,ev_pct = best
    message=_format_motd_message(home,away,league,kickoff_txt,sugg,prob_pct,odds,book,ev_pct)
    ok = send_telegram(message)
    return ok

# ─────────────────────────── Auto-train (optional) ───────────────────────────
try:
    import train_models as _tm
    train_models = _tm.train_models  # expose fn signature
except Exception as e:
    _IMPORT_ERR = repr(e)
    def train_models(*args, **kwargs):
        log.warning("train_models not available: %s", _IMPORT_ERR)
        return {"ok": False, "reason": f"train_models import failed: {_IMPORT_ERR}"}

def auto_train_job():
    if not TRAIN_ENABLE:
        return send_telegram("🤖 Training skipped: TRAIN_ENABLE=0")
    send_telegram("🤖 Training started.")
    try:
        res = train_models() or {}
        ok  = bool(res.get("ok"))
        if not ok:
            reason = res.get("reason") or res.get("error") or "unknown"
            return send_telegram(f"⚠️ Training finished: <b>SKIPPED</b>\nReason: {escape(str(reason))}")

        trained = [k for k, v in (res.get("trained") or {}).items() if v]
        lines = ["🤖 <b>Model training OK</b>"]
        if trained:
            lines.append("• Trained: " + ", ".join(sorted(trained)))
        # Echo OU2.5 threshold for visibility
        try:
            thr = get_setting_cached("conf_threshold:Over/Under 2.5")
            if thr is not None:
                lines.append(f"• OU 2.5 threshold: {float(thr):.1f}%")
        except: pass
        send_telegram("\n".join(lines))
    except Exception as e:
        log.exception("[TRAIN] job failed: %s", e)
        send_telegram(f"❌ Training <b>FAILED</b>\n{escape(str(e))}")

# ─────────────────────────── HTTP endpoints ───────────────────────────
def _require_admin():
    key=request.headers.get("X-API-Key") or request.args.get("key") or ((request.json or {}).get("key") if request.is_json else None)
    if not ADMIN_API_KEY or key != ADMIN_API_KEY: abort(401)

@app.route("/")
def root():
    return jsonify({"ok": True, "name": "goalsniper", "mode": "OU_2.5_ONLY", "scheduler": RUN_SCHEDULER})

@app.route("/health")
def health():
    try:
        with db_conn() as c:
            n = c.execute("SELECT COUNT(*) FROM tips").fetchone()[0]
        api_ok=False
        try:
            test_resp=_api_get(FOOTBALL_API_URL, {"live":"all"})
            api_ok = test_resp is not None
        except: pass
        return jsonify({"ok": True,"db":"ok","tips_count":int(n),"api_connected":api_ok,"scheduler_running":_scheduler_started,"timestamp": time.time()})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/metrics")
def metrics():
    try:
        return jsonify({"ok": True, "metrics": METRICS})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/init-db", methods=["POST"])
def http_init_db(): 
    _require_admin(); init_db(); return jsonify({"ok": True})

@app.route("/settings/<key>", methods=["GET","POST"])
def http_settings(key: str):
    _require_admin()
    if request.method=="GET":
        val=get_setting_cached(key); return jsonify({"ok": True, "key": key, "value": val})
    val=(request.get_json(silent=True) or {}).get("value")
    if val is None: abort(400)
    set_setting(key, str(val)); invalidate_model_caches_for_key(key)
    return jsonify({"ok": True})

@app.route("/admin/scan", methods=["POST","GET"])
def http_scan(): 
    _require_admin(); s,l=production_scan_ou25(); return jsonify({"ok": True, "saved": s, "live_seen": l})

@app.route("/admin/backfill-results", methods=["POST","GET"])
def http_backfill(): 
    _require_admin(); n=backfill_results_for_open_matches(400); return jsonify({"ok": True, "updated": n})

@app.route("/admin/digest", methods=["POST","GET"])
def http_digest(): 
    _require_admin(); msg=daily_accuracy_digest(); return jsonify({"ok": True, "sent": bool(msg)})

@app.route("/admin/train", methods=["POST","GET"])
def http_train():
    _require_admin()
    if not TRAIN_ENABLE:
        return jsonify({"ok": False, "reason": "training disabled"}), 400
    try:
        out=train_models()
        return jsonify({"ok": True, "result": out})
    except Exception as e:
        log.exception("train_models failed: %s", e)
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/admin/motd", methods=["POST","GET"])
def http_motd():
    _require_admin(); ok=send_match_of_the_day(); return jsonify({"ok": bool(ok)})

@app.route("/tips/latest")
def http_latest():
    limit=int(request.args.get("limit","50"))
    with db_conn() as c:
        rows=c.execute("SELECT match_id,league,home,away,market,suggestion,confidence,confidence_raw,score_at_tip,minute,created_ts,odds,book,ev_pct "
                       "FROM tips WHERE suggestion<>'HARVEST' ORDER BY created_ts DESC LIMIT %s",(max(1,min(500,limit)),)).fetchall()
    tips=[]
    for r in rows:
        tips.append({"match_id":int(r[0]),"league":r[1],"home":r[2],"away":r[3],"market":r[4],"suggestion":r[5],
                     "confidence":float(r[6]),"confidence_raw":(float(r[7]) if r[7] is not None else None),
                     "score_at_tip":r[8],"minute":int(r[9]),"created_ts":int(r[10]),
                     "odds": (float(r[11]) if r[11] is not None else None), "book": r[12], "ev_pct": (float(r[13]) if r[13] is not None else None)})
    return jsonify({"ok": True, "tips": tips})

@app.route("/telegram/webhook/<secret>", methods=["POST"])
def telegram_webhook(secret: str):
    if (WEBHOOK_SECRET or "") != secret: abort(403)
    update=request.get_json(silent=True) or {}
    try:
        msg=(update.get("message") or {}).get("text") or ""
        if msg.startswith("/start"): send_telegram("👋 goalsniper OU 2.5 bot is online.")
        elif msg.startswith("/digest"): daily_accuracy_digest()
        elif msg.startswith("/motd"): send_match_of_the_day()
        elif msg.startswith("/scan"):
            parts=msg.split()
            if len(parts)>1 and ADMIN_API_KEY and parts[1]==ADMIN_API_KEY:
                s,l=production_scan_ou25(); send_telegram(f"🔁 Scan done. Saved: {s}, Live seen: {l}")
            else: send_telegram("🔒 Admin key required.")
    except Exception as e:
        log.warning("telegram webhook parse error: %s", e)
    return jsonify({"ok": True})

# ─────────────────────────── Scheduler ───────────────────────────
_scheduler_started=False

def _run_with_pg_lock(lock_key: int, fn, *a, **k):
    try:
        with db_conn() as c:
            got=c.execute("SELECT pg_try_advisory_lock(%s)",(lock_key,)).fetchone()[0]
            if not got: log.info("[LOCK %s] busy; skipped.", lock_key); return None
            try: return fn(*a,**k)
            finally: c.execute("SELECT pg_advisory_unlock(%s)",(lock_key,))
    except Exception as e:
        log.exception("[LOCK %s] failed: %s", lock_key, e); return None

def _today_fixture_ids() -> List[int]:
    fixtures = _collect_todays_prematch_fixtures()
    return [int(((fx.get('fixture') or {}).get('id') or 0)) for fx in fixtures if int(((fx.get('fixture') or {}).get('id') or 0))]

def snapshot_odds_for_fixtures(fixtures: List[int]) -> int:
    wrote=0; now=int(time.time())
    for fid in fixtures:
        try:
            od = fetch_odds_ou25(fid)
            rows=[]
            for mk, sides in (od or {}).items():
                for sel,p in (sides or {}).items():
                    o=float((p or {}).get("odds") or 0); b=(p or {}).get("book") or "Book"
                    if o <= 0: continue
                    rows.append((fid, now, mk, sel, o, b))
            if not rows: continue
            with db_conn() as c:
                c.cur.executemany(
                    "INSERT INTO odds_history(match_id,captured_ts,market,selection,odds,book) "
                    "VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING", rows
                )
                wrote += len(rows)
        except Exception:
            continue
    return wrote

def _start_scheduler_once():
    global _scheduler_started
    if _scheduler_started or not RUN_SCHEDULER:
        return
    try:
        sched=BackgroundScheduler(timezone=TZ_UTC)

        # live scan
        sched.add_job(lambda: _run_with_pg_lock(2001, production_scan_ou25),
                      "interval", seconds=SCAN_INTERVAL_SEC, id="scan", max_instances=1, coalesce=True)

        # backfill
        sched.add_job(lambda: _run_with_pg_lock(2002, backfill_results_for_open_matches, 400),
                      "interval", minutes=BACKFILL_EVERY_MIN, id="backfill", max_instances=1, coalesce=True)

        # digest (daily)
        if DAILY_ACCURACY_DIGEST_ENABLE:
            sched.add_job(lambda: _run_with_pg_lock(2003, daily_accuracy_digest),
                          CronTrigger(hour=DAILY_ACCURACY_HOUR, minute=DAILY_ACCURACY_MINUTE, timezone=BERLIN_TZ),
                          id="digest", max_instances=1, coalesce=True)

        # MOTD (Berlin 10:15 by default)
        sched.add_job(lambda: _run_with_pg_lock(2004, send_match_of_the_day),
                      CronTrigger(hour=MOTD_HOUR, minute=MOTD_MINUTE, timezone=BERLIN_TZ),
                      id="motd", max_instances=1, coalesce=True)

        # training (UTC cron)
        if TRAIN_ENABLE:
            sched.add_job(lambda: _run_with_pg_lock(2005, auto_train_job),
                          CronTrigger(hour=TRAIN_HOUR_UTC, minute=TRAIN_MINUTE_UTC, timezone=TZ_UTC),
                          id="train", max_instances=1, coalesce=True)

        # odds snapshot loop for today's fixtures (helps prematch EV quality)
        sched.add_job(lambda: _run_with_pg_lock(2006, lambda: snapshot_odds_for_fixtures(_today_fixture_ids())),
                      "interval", seconds=180, id="odds_snap", max_instances=1, coalesce=True)

        sched.start()
        _scheduler_started=True
        send_telegram("🚀 goalsniper OU 2.5 mode started (live + MOTD + auto-train).")
        log.info("[SCHED] started (scan=%ss)", SCAN_INTERVAL_SEC)
    except Exception as e:
        log.exception("[SCHED] failed: %s", e)

# ─────────────────────────── Boot / shutdown ───────────────────────────
def validate_config():
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID or not API_KEY or not DATABASE_URL:
        raise SystemExit("Missing required config.")

def shutdown_handler(signum=None, frame=None, *, allow_exit: bool = True):
    log.info("Received shutdown signal, cleaning up...")
    try:
        if POOL:
            try: POOL.closeall()
            except Exception as e: log.warning("Error closing pool: %s", e)
    except Exception as e:
        log.debug("Shutdown cleanup skipped: %s", e)
    if allow_exit and signum is not None:
        try: sys.exit(0)
        except SystemExit: pass

def register_shutdown_handlers():
    signal.signal(signal.SIGINT,  lambda s,f: shutdown_handler(s,f,allow_exit=True))
    signal.signal(signal.SIGTERM, lambda s,f: shutdown_handler(s,f,allow_exit=True))
    atexit.register(lambda: shutdown_handler(None,None,allow_exit=False))

def _on_boot():
    register_shutdown_handlers()
    validate_config()
    _init_pool()
    init_db()
    set_setting("boot_ts", str(int(time.time())))
    _start_scheduler_once()

_on_boot()

if __name__ == "__main__":
    app.run(host=os.getenv("HOST","0.0.0.0"), port=int(os.getenv("PORT","8080")))
