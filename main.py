# main.py — Opta supercomputer style (merged)
# Robust infra (current) + live/prematch intelligence (old)
from __future__ import annotations

import os, re, sys, atexit, time, json, math, html, hmac, uuid, signal, random, statistics, threading
from typing import Any, Dict, List, Tuple, Optional, Iterable, Callable
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from contextlib import contextmanager

# 3rd party
from flask import Flask, jsonify, request, abort, g, has_request_context
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from cachetools import TTLCache
import psycopg2
from psycopg2.pool import SimpleConnectionPool
from psycopg2 import OperationalError, InterfaceError
from psycopg2.extras import DictCursor, execute_values

# training/tuning helpers (guarded import to avoid boot crash if symbols move)
import logging
try:
    from train_models import train_models as _train_models, auto_tune_thresholds as _auto_tune_thresholds, load_thresholds_from_settings as _load_thresholds_from_settings
    _TRAIN_MODULE_AVAILABLE = True
except Exception as _imp_err:
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("goalsniper").warning("[import] train_models import problem: %s", _imp_err)
    _TRAIN_MODULE_AVAILABLE = False
    # Fallbacks so app still boots; admin endpoints/scheduler will gate on availability.
    def _train_models(*_a, **_k):
        raise RuntimeError("train_models not available")
    def _auto_tune_thresholds(*_a, **_k):
        return {"ok": False, "reason": "auto_tune_thresholds not available"}
    def _load_thresholds_from_settings():
        return {
            "CONF_MIN": float(os.getenv("CONF_MIN", "0.75")),
            "EV_MIN": float(os.getenv("EV_MIN", "0.00")),
            "MOTD_CONF_MIN": float(os.getenv("MOTD_CONF_MIN", "0.78")),
            "MOTD_EV_MIN": float(os.getenv("MOTD_EV_MIN", "0.05")),
        }

# ───────────────────────────────────────────────────────────────────────────────
# Logging with request-id injection
# ───────────────────────────────────────────────────────────────────────────────
def _record_factory_with_request_id():
    base_factory = logging.getLogRecordFactory()
    def factory(*args, **kwargs):
        record = base_factory(*args, **kwargs)
        try:
            if not hasattr(record, "request_id"):
                if has_request_context():
                    rid = getattr(g, "request_id", None)
                    record.request_id = rid or "-"
                else:
                    record.request_id = "-"
        except Exception:
            record.request_id = "-"
        return record
    return factory

import logging
logging.setLogRecordFactory(_record_factory_with_request_id())
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(request_id)s - %(message)s")
log = logging.getLogger("goalsniper")

# ───────────────────────────────────────────────────────────────────────────────
# Env helpers
# ───────────────────────────────────────────────────────────────────────────────
def env_bool(name: str, default: bool=False) -> bool:
    v = os.getenv(name)
    if v is None: return default
    return str(v).strip().lower() not in {"0","false","no","off",""}

def env_int(name: str, default: int) -> int:
    try: return int(os.getenv(name, str(default)))
    except: return default

def env_float(name: str, default: float) -> float:
    try: return float(os.getenv(name, str(default)))
    except: return default

TZ_UTC = ZoneInfo("UTC")
BERLIN_TZ = ZoneInfo("Europe/Berlin")

# ───────────────────────────────────────────────────────────────────────────────
# Telegram utils
# ───────────────────────────────────────────────────────────────────────────────
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN","")
CHAT_IDS = [c.strip() for c in os.getenv("TELEGRAM_CHAT_ID","").split(",") if c.strip()]
THREAD_ID = os.getenv("TELEGRAM_THREAD_ID")
SEND_MESSAGES = env_bool("SEND_MESSAGES", True)
PARSE_MODE = os.getenv("TELEGRAM_PARSE_MODE","HTML").strip()
ALLOW_HTML_TAGS = env_bool("TELEGRAM_ALLOW_HTML_TAGS", False)
API_BASE_TG = f"https://api.telegram.org/bot{BOT_TOKEN}" if BOT_TOKEN else ""
TG_HARD_MAX=4096
MAX_LEN_TG=min(env_int("TELEGRAM_MAX_LEN", TG_HARD_MAX), TG_HARD_MAX)
FAIL_COOLDOWN_SEC=env_int("TELEGRAM_FAIL_COOLDOWN_SEC", 60)
_last_fail_ts_tg=0

# Thread-local sessions (why: requests.Session is not guaranteed thread-safe)
_thread_local = threading.local()

def _build_session(kind: str) -> requests.Session:
    s = requests.Session()
    if kind == "tg":
        s.mount("https://", HTTPAdapter(max_retries=Retry(total=3, connect=3, read=3, backoff_factor=0.7,
                        status_forcelist=[429,500,502,503,504], allowed_methods=frozenset(["POST"]),
                        respect_retry_after_header=True, raise_on_status=False), pool_connections=32, pool_maxsize=64))
        s.mount("http://", HTTPAdapter())
    else:
        s.mount("https://", HTTPAdapter(max_retries=Retry(total=3,connect=3,read=3,backoff_factor=0.6,
                        status_forcelist=[429,500,502,503,504],allowed_methods=frozenset(["GET"]),
                        respect_retry_after_header=True,raise_on_status=False), pool_connections=64, pool_maxsize=128))
        s.mount("http://", HTTPAdapter())
    return s

def _get_tg_session() -> requests.Session:
    s = getattr(_thread_local, "tg_session", None)
    if s is None:
        s = _build_session("tg"); _thread_local.tg_session = s
    return s

def _get_http_session() -> requests.Session:
    s = getattr(_thread_local, "http_session", None)
    if s is None:
        s = _build_session("http"); _thread_local.http_session = s
    return s

# Legacy singletons kept for compatibility (not used directly anymore)
_session_tg=_build_session("tg")
_session_http=_build_session("http")

def _tg_prepare(text:str)->str:
    t=str(text or "")
    if PARSE_MODE.lower()=="html":
        return (t if ALLOW_HTML_TAGS else html.escape(t))[:MAX_LEN_TG]
    return t[:MAX_LEN_TG]

def _tg_split(s:str,limit:int)->List[str]:
    if len(s)<=limit: return [s]
    out=[]; cur=s
    while len(cur)>limit:
        cut=cur.rfind("\n\n",0,limit)
        if cut==-1: cut=cur.rfind("\n",0,limit)
        if cut==-1: cut=cur.rfind(" ",0,limit)
        if cut==-1: cut=limit
        out.append(cur[:cut].rstrip()); cur=cur[cut:].lstrip()
    if cur: out.append(cur)
    return out

def send_telegram(text:str, disable_preview:bool=True)->bool:
    # why: per-thread session eliminates rare concurrent post bugs
    sess = _get_tg_session()
    global _last_fail_ts_tg
    if not (BOT_TOKEN and CHAT_IDS and SEND_MESSAGES): return False
    if _last_fail_ts_tg and time.time()-_last_fail_ts_tg<FAIL_COOLDOWN_SEC: return False
    url=f"{API_BASE_TG}/sendMessage"
    body=_tg_prepare(text); parts=_tg_split(body, MAX_LEN_TG)
    ok_any=False
    for chat_id in CHAT_IDS:
        for idx, part in enumerate(parts,1):
            payload={"chat_id":chat_id,"text":part,"disable_web_page_preview":disable_preview}
            if PARSE_MODE.lower() in ("html","markdown","markdownv2"): payload["parse_mode"]=PARSE_MODE
            if THREAD_ID:
                try: payload["message_thread_id"]=int(THREAD_ID)
                except: pass
            backoff=1.5
            for _ in range(3):
                try:
                    r=sess.post(url, json=payload, timeout=(3,10))
                    if r.status_code==200: ok_any=True; break
                    if r.status_code==429:
                        try: time.sleep(int(r.json().get("parameters",{}).get("retry_after",5)))
                        except: time.sleep(5)
                        continue
                    time.sleep(backoff+random.uniform(0,0.4)); backoff*=2
                except: time.sleep(backoff+random.uniform(0,0.4)); backoff*=2
            if idx<len(parts): time.sleep(0.25+random.uniform(0,0.15))
    if not ok_any: _last_fail_ts_tg=time.time()
    return ok_any

def send_telegram_safe(text:str, **kw)->bool:
    try: return send_telegram(text, **kw)
    except Exception as e: log.warning("[TELEGRAM] send fail: %s", e); return False

# ───────────────────────────────────────────────────────────────────────────────
# Postgres pool + schema/bootstrap
# ───────────────────────────────────────────────────────────────────────────────
DB_URL=os.getenv("DATABASE_URL"); 
if not DB_URL: raise SystemExit("DATABASE_URL is required")
if os.getenv("DB_SSLMODE_REQUIRE","1").lower() not in {"0","false","no",""} and "sslmode=" not in DB_URL:
    DB_URL += (("&" if "?" in DB_URL else "?") + "sslmode=require")

STMT_TIMEOUT_MS=env_int("PG_STATEMENT_TIMEOUT_MS",15000)
LOCK_TIMEOUT_MS=env_int("PG_LOCK_TIMEOUT_MS",2000)
IDLE_TX_TIMEOUT_MS=env_int("PG_IDLE_TX_TIMEOUT_MS",30000)
FORCE_UTC=env_bool("PG_FORCE_UTC", True)

POOL: Optional[SimpleConnectionPool]=None
def _apply_session_settings(conn)->None:
    cur=conn.cursor()
    try:
        if FORCE_UTC: cur.execute("SET TIME ZONE 'UTC'")
        cur.execute("SET statement_timeout = %s",(STMT_TIMEOUT_MS,))
        cur.execute("SET lock_timeout = %s",(LOCK_TIMEOUT_MS,))
        cur.execute("SET idle_in_transaction_session_timeout = %s",(IDLE_TX_TIMEOUT_MS,))
    finally:
        cur.close()

def _init_pool():
    global POOL
    if POOL is None:
        minconn=env_int("DB_POOL_MIN",1); maxconn=env_int("DB_POOL_MAX",4)
        POOL=SimpleConnectionPool(minconn=minconn, maxconn=maxconn, dsn=DB_URL)
        log.info("[DB] pool created (min=%d max=%d)", minconn, maxconn)
        atexit.register(_close_pool)

def _close_pool():
    global POOL
    if POOL is not None:
        try: POOL.closeall(); log.info("[DB] pool closed")
        except: log.warning("[DB] pool close failed", exc_info=True)
        POOL=None

MAX_RETRIES=env_int("DB_MAX_RETRIES",3); BASE_BACKOFF_MS=env_int("DB_BASE_BACKOFF_MS",150)

class _PooledCursor:
    def __init__(self, pool:SimpleConnectionPool, dict_rows:bool=False):
        self.pool=pool; self.conn=None; self.cur=None; self.dict_rows=dict_rows
    def __enter__(self): self._acquire(); return self
    def __exit__(self,a,b,c):
        try:
            if self.cur: 
                try: self.cur.close()
                except: pass
        finally:
            if self.conn:
                try: self.pool.putconn(self.conn)
                except: pass
        self.conn=self.cur=None
    def __getattr__(self,name):
        if name.startswith("_"): raise AttributeError(name)
        if self.cur is None: raise AttributeError(name)
        return getattr(self.cur,name)
    def _acquire(self):
        _init_pool(); self.pool=POOL  # type: ignore
        attempts=0
        while True:
            try:
                self.conn=self.pool.getconn()  # type: ignore
                self.conn.autocommit=True; _apply_session_settings(self.conn)
                self.cur=self.conn.cursor(cursor_factory=DictCursor if self.dict_rows else None)
                self.cur.execute("SELECT 1"); self.cur.fetchone(); return
            except (OperationalError, InterfaceError) as e:
                try:
                    if self.cur: 
                        try: self.cur.close()
                        except: pass
                    if self.conn: self.pool.putconn(self.conn, close=True)  # type: ignore
                except: pass
                self.conn=self.cur=None
                if attempts>=MAX_RETRIES: log.warning("[DB] acquire failed: %s", e); raise
                backoff=min(2000, BASE_BACKOFF_MS*(2**attempts)); time.sleep(backoff/1000.0); attempts+=1
    def _reset(self):
        try:
            if self.cur: 
                try: self.cur.close()
                except: pass
        finally:
            try:
                if self.conn: self.pool.putconn(self.conn, close=True)  # type: ignore
            except: pass
        self.conn=self.cur=None; self._acquire()
    def _retry_loop(self, fn, *a, **k):
        attempts=0
        while True:
            try: return fn(*a,**k)
            except (OperationalError, InterfaceError) as e:
                if attempts>=MAX_RETRIES: log.warning("[DB] giving up: %s", e); raise
                self._reset(); backoff=min(2000, BASE_BACKOFF_MS*(2**attempts)); time.sleep(backoff/1000.0); attempts+=1
    def execute(self, sql_text:str, params:Tuple|list=()):
        def _call(): self.cur.execute(sql_text, params or ()); return self.cur
        return self._retry_loop(_call)
    def executemany(self, sql_text:str, rows:Iterable[Tuple]):
        rows=list(rows) or []
        def _call(): self.cur.executemany(sql_text, rows); return self.cur
        return self._retry_loop(_call)
    def execute_values(self, sql_text:str, rows:Iterable[Tuple], page_size:int=200):
        # why: wrap psycopg2.extras.execute_values with the same retry logic
        rows = list(rows) or []
        def _call():
            execute_values(self.cur, sql_text, rows, page_size=page_size)
            return self.cur
        return self._retry_loop(_call)
    def fetchone(self): return self.cur.fetchone() if self.cur else None
    def fetchall(self): return self.cur.fetchall() if self.cur else []

def db_conn(dict_rows:bool=False)->_PooledCursor:
    _init_pool(); return _PooledCursor(POOL, dict_rows=dict_rows)  # type: ignore

@contextmanager
def tx(dict_rows:bool=False):
    _init_pool(); conn=None; cur=None; attempts=0
    while True:
        try:
            conn=POOL.getconn()  # type: ignore
            conn.autocommit=False; _apply_session_settings(conn)
            cur=conn.cursor(cursor_factory=DictCursor if dict_rows else None)
            cur.execute("SELECT 1"); cur.fetchone(); break
        except (OperationalError, InterfaceError):
            if attempts>=MAX_RETRIES: raise
            if conn:
                try: POOL.putconn(conn, close=True)  # type: ignore
                except: pass
            attempts+=1; time.sleep(min(2000, BASE_BACKOFF_MS*(2**attempts))/1000.0)
    try:
        yield cur; conn.commit()
    except Exception:
        try: conn.rollback()
        except: pass
        raise
    finally:
        try:
            if cur: cur.close()
        finally:
            if conn:
                try: POOL.putconn(conn)  # type: ignore
                except: pass

# ───────────────────────────────────────────────────────────────────────────────
# Schema / settings helpers
# ───────────────────────────────────────────────────────────────────────────────
SCHEMA_TABLES_SQL = [
    """CREATE TABLE IF NOT EXISTS tips(
        match_id BIGINT, league_id BIGINT, league TEXT, home TEXT, away TEXT,
        market TEXT, suggestion TEXT, confidence DOUBLE PRECISION, confidence_raw DOUBLE PRECISION,
        score_at_tip TEXT, minute INTEGER, created_ts BIGINT, odds DOUBLE PRECISION, book TEXT,
        ev_pct DOUBLE PRECISION, sent_ok INTEGER DEFAULT 1, PRIMARY KEY(match_id, created_ts))""",
    """CREATE TABLE IF NOT EXISTS tip_snapshots(
        match_id BIGINT, created_ts BIGINT, payload TEXT, PRIMARY KEY(match_id, created_ts))""",
    """CREATE TABLE IF NOT EXISTS prematch_snapshots(
        match_id BIGINT PRIMARY KEY, created_ts BIGINT, payload TEXT)""",
    """CREATE TABLE IF NOT EXISTS feedback(id SERIAL PRIMARY KEY, match_id BIGINT UNIQUE, verdict INTEGER, created_ts BIGINT)""",
    """CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT)""",
    """CREATE TABLE IF NOT EXISTS match_results(
        match_id BIGINT PRIMARY KEY, final_goals_h INTEGER, final_goals_a INTEGER, btts_yes INTEGER, updated_ts BIGINT)""",
    """CREATE TABLE IF NOT EXISTS odds_history(
        match_id BIGINT, captured_ts BIGINT, market TEXT, selection TEXT, odds DOUBLE PRECISION, book TEXT,
        PRIMARY KEY(match_id, market, selection, captured_ts))""",
    """CREATE TABLE IF NOT EXISTS lineups(match_id BIGINT PRIMARY KEY, created_ts BIGINT, payload TEXT)""",
    """CREATE TABLE IF NOT EXISTS fixtures(
        fixture_id BIGINT PRIMARY KEY, league_name TEXT, home TEXT, away TEXT,
        kickoff TIMESTAMPTZ, last_update TIMESTAMPTZ, status TEXT)"""
]
MIGRATIONS_SQL = [
    "ALTER TABLE tips ADD COLUMN IF NOT EXISTS odds DOUBLE PRECISION",
    "ALTER TABLE tips ADD COLUMN IF NOT EXISTS book TEXT",
    "ALTER TABLE tips ADD COLUMN IF NOT EXISTS ev_pct DOUBLE PRECISION",
    "ALTER TABLE tips ADD COLUMN IF NOT EXISTS confidence_raw DOUBLE PRECISION",
    "ALTER TABLE tips ADD COLUMN IF NOT EXISTS sent_ok INTEGER DEFAULT 1",
    "ALTER TABLE fixtures ADD COLUMN IF NOT EXISTS last_update TIMESTAMPTZ",
    "ALTER TABLE fixtures ADD COLUMN IF NOT EXISTS status TEXT",
]
INDEX_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_tips_created ON tips(created_ts DESC)",
    "CREATE INDEX IF NOT EXISTS idx_tips_match ON tips(match_id)",
    "CREATE INDEX IF NOT EXISTS idx_tips_sent ON tips(sent_ok, created_ts DESC)",
    "CREATE INDEX IF NOT EXISTS idx_snap_by_match ON tip_snapshots(match_id, created_ts DESC)",
    "CREATE INDEX IF NOT EXISTS idx_results_updated ON match_results(updated_ts DESC)",
    "CREATE INDEX IF NOT EXISTS idx_odds_hist_match ON odds_history(match_id, captured_ts DESC)",
    "CREATE INDEX IF NOT EXISTS idx_fixtures_status_update ON fixtures(status, last_update DESC)",
]
def init_db()->None:
    _init_pool()
    with db_conn() as c:
        for s in SCHEMA_TABLES_SQL: c.execute(s)
        for s in MIGRATIONS_SQL:
            try: c.execute(s)
            except Exception as e: log.debug("[DB] migration skipped: %s -> %s", s, e)
        for s in INDEX_SQL:
            try: c.execute(s)
            except Exception as e: log.debug("[DB] index skipped: %s -> %s", s, e)
    log.info("[DB] schema ready")

def get_setting(key:str)->Optional[str]:
    with db_conn() as c:
        row=c.execute("SELECT value FROM settings WHERE key=%s",(key,)).fetchone()
        return (row[0] if row else None)

def set_setting(key:str, value:str)->None:
    with db_conn() as c:
        c.execute("INSERT INTO settings(key,value) VALUES(%s,%s) ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value", (key,value))

def get_setting_json(key:str)->Optional[dict]:
    try:
        raw=get_setting(key); return json.loads(raw) if raw else None
    except Exception as e:
        log.warning("[DB] get_setting_json failed key=%s: %s", key, e); return None

# ───────────────────────────────────────────────────────────────────────────────
# API-Football odds client + EV gating
# ───────────────────────────────────────────────────────────────────────────────
APIFOOTBALL_KEY=os.getenv("APIFOOTBALL_KEY","") or os.getenv("API_KEY","")
APISPORTS_BASE_URL=os.getenv("APISPORTS_BASE_URL","https://v3.football.api-sports.io").rstrip("/")
ODDS_HEADERS={"x-apisports-key":APIFOOTBALL_KEY,"Accept":"application/json","User-Agent":os.getenv("HTTP_USER_AGENT","goalsniper/1.0 (+odds)")}
ODDS_SOURCE=os.getenv("ODDS_SOURCE","auto").lower()  # auto|live|prematch
ODDS_AGGREGATION=os.getenv("ODDS_AGGREGATION","median").lower()
ODDS_OUTLIER_MULT=env_float("ODDS_OUTLIER_MULT",1.8)
ODDS_REQUIRE_N_BOOKS=env_int("ODDS_REQUIRE_N_BOOKS",2)
ODDS_FAIR_MAX_MULT=env_float("ODDS_FAIR_MAX_MULT",2.5)
MAX_ODDS_ALL=env_float("MAX_ODDS_ALL",20.0)
FALLBACK_TO_PREMATCH_ON_EMPTY_LIVE=env_bool("FALLBACK_TO_PREMATCH_ON_EMPTY_LIVE", True)
BOOK_WHITELIST={b.strip().lower() for b in os.getenv("ODDS_BOOK_WHITELIST","").split(",") if b.strip()}
BOOK_BLACKLIST={b.strip().lower() for b in os.getenv("ODDS_BOOK_BLACKLIST","").split(",") if b.strip()}
MIN_ODDS_OU=env_float("MIN_ODDS_OU",1.50)
MIN_ODDS_BTTS=env_float("MIN_ODDS_BTTS",1.50)
MIN_ODDS_1X2=env_float("MIN_ODDS_1X2",1.50)
ALLOW_TIPS_WITHOUT_ODDS=env_bool("ALLOW_TIPS_WITHOUT_ODDS", False)
HTTP_CONNECT_TIMEOUT=env_float("HTTP_CONNECT_TIMEOUT",3.0)
HTTP_READ_TIMEOUT=env_float("HTTP_READ_TIMEOUT",10.0)
ODDS_CACHE_TTL_SEC=env_int("ODDS_CACHE_TTL_SEC",120)
ODDS_CACHE_MAX_ITEMS=env_int("ODDS_CACHE_MAX_ITEMS",2000)

# Thread-safe cache access
_ODDS_CACHE: TTLCache[int, dict] = TTLCache(maxsize=ODDS_CACHE_MAX_ITEMS, ttl=ODDS_CACHE_TTL_SEC)
_odds_cache_lock = threading.RLock()

def _book_ok(name:str)->bool:
    n=(name or "").strip().lower()
    if BOOK_WHITELIST and n not in BOOK_WHITELIST: return False
    if n in BOOK_BLACKLIST: return False
    return True

def _market_name_normalize(s:str)->str:
    s=(s or "").strip().lower()
    if "both teams" in s or "btts" in s: return "BTTS"
    if "winner" in s or "1x2" in s: return "1X2"
    if "over/under" in s or "total" in s or "goals" in s: return "OU"
    return s.upper()

def _api_get(path:str, params:dict)->Optional[dict]:
    if not APIFOOTBALL_KEY: return None
    url=f"{APISPORTS_BASE_URL}/{path.lstrip('/')}"
    sess = _get_http_session()
    try:
        r=sess.get(url, headers=ODDS_HEADERS, params=params, timeout=(HTTP_CONNECT_TIMEOUT,HTTP_READ_TIMEOUT))
        if not r.ok: return None
        js=r.json(); return js if isinstance(js,dict) else None
    except: return None

def fetch_odds(fid:int, prob_hints:Optional[Dict[str,float]]=None)->dict:
    with _odds_cache_lock:
        cached=_ODDS_CACHE.get(fid)
    if cached is not None: return cached
    js={}
    if ODDS_SOURCE in ("auto","live"):
        tmp=_api_get("odds/live",{"fixture":fid}) or {}
        if tmp.get("response"): js=tmp
        elif ODDS_SOURCE=="auto" and FALLBACK_TO_PREMATCH_ON_EMPTY_LIVE:
            js=_api_get("odds",{"fixture":fid}) or {}
    if not js and ODDS_SOURCE=="prematch": js=_api_get("odds",{"fixture":fid}) or {}
    by_market={}
    try:
        for r in (js.get("response") or []):
            for bk in (r.get("bookmakers") or []):
                book_name=(bk.get("name") or "Book").strip()
                if not _book_ok(book_name): continue
                for mkt in (bk.get("bets") or []):
                    mname=_market_name_normalize(mkt.get("name",""))
                    vals=(mkt.get("values") or [])
                    if mname=="BTTS":
                        for v in vals:
                            lbl=(v.get("value") or "").lower(); odd=float(v.get("odd") or 0)
                            if "yes" in lbl: by_market.setdefault("BTTS",{}).setdefault("Yes",[]).append((odd,book_name))
                            elif "no" in lbl: by_market.setdefault("BTTS",{}).setdefault("No",[]).append((odd,book_name))
                    elif mname=="1X2":
                        for v in vals:
                            lbl=(v.get("value") or "").lower(); odd=float(v.get("odd") or 0)
                            if lbl in ("home","1"): by_market.setdefault("1X2",{}).setdefault("Home",[]).append((odd,book_name))
                            elif lbl in ("away","2"): by_market.setdefault("1X2",{}).setdefault("Away",[]).append((odd,book_name))
                    elif mname=="OU":
                        for v in vals:
                            lbl=(v.get("value") or "").lower()
                            if "over" in lbl or "under" in lbl:
                                try: ln=float(lbl.split()[-1])
                                except: continue
                                key=f"OU_{ln}"; side="Over" if "over" in lbl else "Under"
                                odd=float(v.get("odd") or 0)
                                by_market.setdefault(key,{}).setdefault(side,[]).append((odd,book_name))
    except Exception as e: log.debug("[odds] parse fail fid=%s: %s", fid, e)
    with _odds_cache_lock:
        _ODDS_CACHE[fid]=by_market
    return by_market

def price_gate(market:str,suggestion:str,fid:int,prob:Optional[float]=None):
    odds_map=fetch_odds(fid); odds=None; book=None
    if market=="BTTS":
        d=odds_map.get("BTTS",{}); tgt="Yes" if suggestion.endswith("Yes") else "No"
        if tgt in d: odds,book=d[tgt][0][0], d[tgt][0][1]
    elif market=="1X2":
        d=odds_map.get("1X2",{}); tgt="Home" if suggestion=="Home Win" else "Away" if suggestion=="Away Win" else None
        if tgt and tgt in d: odds,book=d[tgt][0][0], d[tgt][0][1]
    elif market.startswith("Over/Under"):
        try:
            ln=float(suggestion.split()[1]); d=odds_map.get(f"OU_{ln}",{})
            tgt="Over" if suggestion.startswith("Over") else "Under"
            if tgt in d: odds,book=d[tgt][0][0], d[tgt][0][1]
        except: pass
    if odds is None: return (ALLOW_TIPS_WITHOUT_ODDS, None, None, None)
    if not (1.01<=float(odds)<=MAX_ODDS_ALL): return (False, odds, book, None)
    ev_pct=None
    if prob is not None:
        edge=prob*float(odds)-1.0; ev_pct=round(edge*100,1)
        if edge<0: return (False, odds, book, ev_pct)
    return (True, float(odds), book, ev_pct)


# ───────────────────────────────────────────────────────────────────────────────
# Predictor hook (supports JSON/pickle; caches per-fixture briefly)
# ───────────────────────────────────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")
MODEL_KIND = os.getenv("MODEL_KIND", "").strip().lower()  # "", "pickle", "json"
PREDICTOR_STRICT = env_bool("PREDICTOR_STRICT", False)
PREDICTION_CACHE_TTL = env_int("PREDICTION_CACHE_TTL", 0)

_model_obj: Any = None
_model_loaded = False
_model_lock = threading.RLock()
_pred_cache: Dict[int, tuple[float, Dict[str, float]]] = {}
_pred_cache_lock = threading.RLock()

def _now_ts() -> float:
    return time.time()

def _load_model() -> Optional[Any]:
    global _model_obj, _model_loaded
    if _model_loaded:
        return _model_obj
    with _model_lock:
        if _model_loaded:
            return _model_obj
        try:
            if not os.path.exists(MODEL_PATH):
                _model_obj = None
                _model_loaded = True
                log.info("[predictor] MODEL_PATH not found (%s) — fallback to odds de-vig", MODEL_PATH)
                return None
            kind = MODEL_KIND or ("json" if MODEL_PATH.endswith(".json") else "pickle")
            if kind == "json":
                with open(MODEL_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    data = {}
                _model_obj = {"kind": "json", "data": data}
                log.info("[predictor] loaded JSON model: %s", MODEL_PATH)
            else:
                import pickle
                with open(MODEL_PATH, "rb") as f:
                    _model_obj = pickle.load(f)
                log.info("[predictor] loaded PICKLE model: %s (%s)", MODEL_PATH, type(_model_obj).__name__)
            _model_loaded = True
        except Exception as e:
            _model_obj = None
            _model_loaded = True
            log.warning("[predictor] load failed: %s", e)
        return _model_obj

def warm_model() -> bool:
    return _load_model() is not None

def _predict_with_json_model(model_obj: dict, fid: int) -> Dict[str, float]:
    data = model_obj.get("data") or {}
    if not isinstance(data, dict):
        return {}
    out = {}
    for k, v in data.items():
        try:
            x = float(v)
            if 0.0 <= x <= 1.0:
                out[str(k)] = x
        except Exception:
            continue
    return out

def _predict_with_pickle(model: Any, fid: int) -> Dict[str, float]:
    try:
        return {}
    except Exception as e:
        log.warning("[predictor] pickle predict failed fid=%s: %s", fid, e)
        return {}

def _get_cached_prediction(fid: int) -> Optional[Dict[str, float]]:
    if PREDICTION_CACHE_TTL <= 0:
        return None
    with _pred_cache_lock:
        entry = _pred_cache.get(fid)
        if not entry:
            return None
        ts, val = entry
        if (_now_ts() - ts) <= PREDICTION_CACHE_TTL:
            return val
        _pred_cache.pop(fid, None)
        return None

def _put_cached_prediction(fid: int, val: Dict[str, float]) -> None:
    if PREDICTION_CACHE_TTL <= 0:
        return
    with _pred_cache_lock:
        if len(_pred_cache) >= 200:
            try:
                _pred_cache.pop(next(iter(_pred_cache)))
            except Exception:
                _pred_cache.clear()
        _pred_cache[fid] = (_now_ts(), val)

def predict_for_fixture(fid: int) -> Dict[str, float]:
    cached = _get_cached_prediction(fid)
    if cached is not None:
        return cached
    model = _load_model()
    if model is None:
        if PREDICTOR_STRICT:
            log.debug("[predictor] strict mode: model missing (fid=%s)", fid)
        return {}
    probs: Dict[str, float]
    if isinstance(model, dict) and model.get("kind") == "json":
        probs = _predict_with_json_model(model, fid)
    else:
        probs = _predict_with_pickle(model, fid)
    out: Dict[str, float] = {}
    for k, v in (probs or {}).items():
        try:
            x = float(v)
            if 0.0 <= x <= 1.0:
                out[str(k)] = x
        except Exception:
            continue
    _put_cached_prediction(fid, out)
    return out

# ───────────────────────────────────────────────────────────────────────────────
# Results provider (final scores / BTTS) — via /fixtures endpoint
# ───────────────────────────────────────────────────────────────────────────────
RESULTS_HEADERS = {
    "x-apisports-key": APIFOOTBALL_KEY or "",
    "Accept": "application/json",
    "User-Agent": os.getenv("HTTP_USER_AGENT", "goalsniper/1.0 (+results)"),
}
HTTP_RESULTS_TIMEOUT = env_float("HTTP_RESULTS_TIMEOUT", 10.0)

def fetch_results_for_fixtures(fixture_ids: Iterable[int]) -> List[Tuple[int, int, int, int]]:
    out: List[Tuple[int, int, int, int]] = []
    for fid in fixture_ids:
        try:
            url = f"{APISPORTS_BASE_URL}/fixtures"
            sess = _get_http_session()
            resp = sess.get(url, headers=RESULTS_HEADERS, params={"id": fid}, timeout=HTTP_RESULTS_TIMEOUT)
            if not resp.ok:
                continue
            js = resp.json()
            data = (js.get("response") or [])
            if not data:
                continue
            fix = data[0]
            goals = fix.get("goals") or {}
            gh = int(goals.get("home") or 0)
            ga = int(goals.get("away") or 0)
            btts = int(gh > 0 and ga > 0)
            out.append((fid, gh, ga, btts))
        except Exception:
            continue
    return out

def update_match_results(rows: Iterable[Tuple[int, int, int, int]]) -> int:
    now_ts = int(time.time())
    n = 0
    with db_conn() as c:
        for mid, gh, ga, btts in rows:
            c.execute(
                """
                INSERT INTO match_results(match_id, final_goals_h, final_goals_a, btts_yes, updated_ts)
                VALUES (%s,%s,%s,%s,%s)
                ON CONFLICT (match_id) DO UPDATE SET
                  final_goals_h=EXCLUDED.final_goals_h,
                  final_goals_a=EXCLUDED.final_goals_a,
                  btts_yes=EXCLUDED.btts_yes,
                  updated_ts=EXCLUDED.updated_ts
                """,
                (mid, gh, ga, int(bool(btts)), now_ts),
            )
            n += 1
    return n

# ───────────────────────────────────────────────────────────────────────────────
# Scanning logic (in-play & prematch), digest, MOTD, retry, backfill
# ───────────────────────────────────────────────────────────────────────────────
CONF_MIN = env_float("CONF_MIN", 0.75)    # min probability (0..1)
EV_MIN   = env_float("EV_MIN",   0.00)    # min EV (>= 0 -> no negative-EV)
MOTD_CONF_MIN = env_float("MOTD_CONF_MIN", 0.78)
MOTD_EV_MIN   = env_float("MOTD_EV_MIN",   0.05)
FEED_STALE_SEC = env_int("FEED_STALE_SEC", 300)
MAX_TELEGRAM_PER_SCAN = env_int("MAX_TELEGRAM_PER_SCAN", 5)

def _now_dt() -> datetime:
    return datetime.now(TZ_UTC)

def _is_feed_stale(last_update: Optional[datetime]) -> bool:
    if not last_update:
        return True
    if last_update.tzinfo is None:
        last_update = last_update.replace(tzinfo=TZ_UTC)
    return (_now_dt() - last_update).total_seconds() > FEED_STALE_SEC

def _fmt_tip_message(match: dict, market: str, suggestion: str,
                     conf: float, odds: Optional[float], book: Optional[str], ev_pct: Optional[float]) -> str:
    ko_dt = match.get("kickoff")
    if ko_dt and getattr(ko_dt, "tzinfo", None) is None:
        ko_dt = ko_dt.replace(tzinfo=TZ_UTC)
    kickoff = (ko_dt or _now_dt()).astimezone(BERLIN_TZ).strftime("%Y-%m-%d %H:%M")
    home, away = match.get("home"), match.get("away")
    league = match.get("league")
    odds_str = "-" if odds is None else f"{float(odds):.2f}"
    book_str = (book or "best")
    msg = (
        f"⚽️ <b>{html.escape(league or '')}</b>\n"
        f"{html.escape(home or '')} vs {html.escape(away or '')}\n"
        f"🕒 Kickoff: {kickoff} Berlin\n"
        f"🎯 <b>Tip:</b> {html.escape(suggestion)}\n"
        f"📊 <b>Confidence:</b> {conf*100:.1f}%\n"
        f"💰 <b>Odds:</b> {odds_str} @ {html.escape(book_str)}"
    )
    if ev_pct is not None:
        msg += f" • <b>EV:</b> {ev_pct:+.1f}%"
    return msg

def _implied_prob(odds: float) -> float:
    try:
        o = max(1e-6, float(odds))
        return 1.0 / o
    except Exception:
        return 0.0

def _normalize_pair(a: float, b: float) -> Tuple[float, float]:
    s = a + b
    if s <= 0:
        return 0.0, 0.0
    return a / s, b / s

def _prob_hints_from_model_or_odds(fid: int, odds_map: dict) -> Dict[str, float]:
    try:
        probs = predict_for_fixture(fid)
        if isinstance(probs, dict) and probs:
            return {str(k): float(v) for k, v in probs.items() if v is not None}
    except Exception as e:
        log.warning("[model] predict failed fid=%s: %s", fid, e)

    hints: Dict[str, float] = {}

    d = odds_map.get("BTTS", {})
    if isinstance(d, dict) and d:
        yes = d.get("Yes") or []
        no  = d.get("No")  or []
        p_yes = _implied_prob((yes[0][0] if yes else 0)) if yes else 0.0
        p_no  = _implied_prob((no[0][0]  if no  else 0)) if no  else 0.0
        p_yes, p_no = _normalize_pair(p_yes, p_no)
        hints["BTTS: Yes"] = p_yes
        hints["BTTS: No"]  = p_no

    d = odds_map.get("1X2", {})
    if isinstance(d, dict) and d:
        h = d.get("Home") or []
        a = d.get("Away") or []
        p_h = _implied_prob((h[0][0] if h else 0)) if h else 0.0
        p_a = _implied_prob((a[0][0] if a else 0)) if a else 0.0
        p_h, p_a = _normalize_pair(p_h, p_a)
        hints["Home Win"] = p_h
        hints["Away Win"] = p_a

    for mk, sides in odds_map.items():
        if not str(mk).startswith("OU_"):
            continue
        try:
            ln = mk.split("_", 1)[1]
        except Exception:
            continue
        over = (sides.get("Over") or [])
        under = (sides.get("Under") or [])
        p_over = _implied_prob((over[0][0] if over else 0)) if over else 0.0
        p_under= _implied_prob((under[0][0] if under else 0)) if under else 0.0
        p_over, p_under = _normalize_pair(p_over, p_under)
        hints[f"Over {ln} Goals"]  = p_over
        hints[f"Under {ln} Goals"] = p_under

    return hints

def _best_candidate_for_fixture(fid: int) -> Optional[Tuple[str, str, float, Optional[float], Optional[str], Optional[float]]]:
    odds_map = fetch_odds(fid)
    if not odds_map:
        return None

    prob_hints = _prob_hints_from_model_or_odds(fid, odds_map)
    candidates: List[Tuple[str, str, float]] = []

    if "BTTS: Yes" in prob_hints:
        candidates.append(("BTTS", "BTTS: Yes", prob_hints["BTTS: Yes"]))
    if "BTTS: No" in prob_hints:
        candidates.append(("BTTS", "BTTS: No", prob_hints["BTTS: No"]))

    if "Home Win" in prob_hints:
        candidates.append(("1X2", "Home Win", prob_hints["Home Win"]))
    if "Away Win" in prob_hints:
        candidates.append(("1X2", "Away Win", prob_hints["Away Win"]))

    for mk in list(odds_map.keys()):
        if not str(mk).startswith("OU_"):
            continue
        ln = str(mk).split("_", 1)[1]
        over_key = f"Over {ln} Goals"
        under_key = f"Under {ln} Goals"
        if over_key in prob_hints:
            candidates.append((f"Over/Under {ln}", over_key, prob_hints[over_key]))
        if under_key in prob_hints:
            candidates.append((f"Over/Under {ln}", under_key, prob_hints[under_key]))

    best = None
    best_ev = float("-inf")

    for market, suggestion, prob in candidates:
        ok, odds, book, ev_pct = price_gate(market, suggestion, fid, prob=prob)
        if not ok:
            continue
        if prob < CONF_MIN:
            continue
        if (ev_pct or 0.0) < (EV_MIN * 100.0):
            continue
        edge = (ev_pct or 0.0)
        if edge > best_ev:
            best_ev = edge
            best = (market, suggestion, prob,
                    (float(odds) if odds is not None else None),
                    (str(book) if book is not None else None),
                    (float(ev_pct) if ev_pct is not None else None))
    return best

def production_scan() -> Tuple[int, int]:
    matches = fetch_live_matches()
    live_seen = len(matches)
    if live_seen == 0:
        log.info("[PROD] no live")
        return 0, 0

    saved = 0
    now_ts = int(time.time())
    per_league_counter: dict[int, int] = {}

    with db_conn() as c:
        for m in matches:
            try:
                fid = int((m.get("fixture", {}) or {}).get("id") or 0)
                if not fid:
                    continue

                if DUP_COOLDOWN_MIN > 0:
                    cutoff = now_ts - DUP_COOLDOWN_MIN * 60
                    if c.execute(
                        "SELECT 1 FROM tips WHERE match_id=%s AND created_ts>=%s AND suggestion<>'HARVEST' LIMIT 1",
                        (fid, cutoff),
                    ).fetchone():
                        continue

                feat = extract_features(m)
                minute = int(feat.get("minute", 0))
                if not stats_coverage_ok(feat, minute):
                    continue
                if minute < TIP_MIN_MINUTE:
                    continue
                if is_feed_stale(fid, m, minute):
                    continue

                if HARVEST_MODE and minute >= TRAIN_MIN_MINUTE and minute % 3 == 0:
                    try:
                        save_snapshot_from_match(m, feat)
                    except Exception:
                        pass

                league_id, league = _league_name(m)
                home, away = _teams(m)
                score = _pretty_score(m)

                candidates: List[Tuple[str, str, float]] = []

                # OU
                for line in OU_LINES:
                    mdl = _load_ou_model_for_line(line)
                    if not mdl:
                        continue
                    mk = f"Over/Under {_fmt_line(line)}"
                    thr = _get_market_threshold(mk)

                    p_over = _score_prob(feat, mdl)
                    sug_over = f"Over {_fmt_line(line)} Goals"
                    if (
                        p_over * 100.0 >= thr
                        and _candidate_is_sane(sug_over, feat)
                        and market_cutoff_ok(minute, mk, sug_over)
                    ):
                        candidates.append((mk, sug_over, p_over))

                    p_under = 1.0 - p_over
                    sug_under = f"Under {_fmt_line(line)} Goals"
                    if (
                        p_under * 100.0 >= thr
                        and _candidate_is_sane(sug_under, feat)
                        and market_cutoff_ok(minute, mk, sug_under)
                    ):
                        candidates.append((mk, sug_under, p_under))

                # BTTS
                mdl_btts = load_model_from_settings("BTTS_YES")
                if mdl_btts:
                    mk = "BTTS"
                    thr = _get_market_threshold(mk)

                    p_yes = _score_prob(feat, mdl_btts)
                    if (
                        p_yes * 100.0 >= thr
                        and _candidate_is_sane("BTTS: Yes", feat)
                        and market_cutoff_ok(minute, mk, "BTTS: Yes")
                    ):
                        candidates.append((mk, "BTTS: Yes", p_yes))

                    p_no = 1.0 - p_yes
                    if (
                        p_no * 100.0 >= thr
                        and _candidate_is_sane("BTTS: No", feat)
                        and market_cutoff_ok(minute, mk, "BTTS: No")
                    ):
                        candidates.append((mk, "BTTS: No", p_no))

                # 1X2 (draw suppressed)
                mh, md, ma = _load_wld_models()
                if mh and md and ma:
                    mk = "1X2"
                    thr = _get_market_threshold(mk)
                    ph = _score_prob(feat, mh)
                    pd = _score_prob(feat, md)
                    pa = _score_prob(feat, ma)
                    s = max(EPS, ph + pd + pa)
                    ph, pa = ph / s, pa / s

                    if ph * 100.0 >= thr and market_cutoff_ok(minute, mk, "Home Win"):
                        candidates.append((mk, "Home Win", ph))
                    if pa * 100.0 >= thr and market_cutoff_ok(minute, mk, "Away Win"):
                        candidates.append((mk, "Away Win", pa))

                if not candidates:
                    continue

                odds_map = fetch_odds(fid) if API_KEY else {}
                ranked: List[Tuple[str, str, float, Optional[float], Optional[str], Optional[float], float]] = []

                for mk, sug, prob in candidates:
                    if sug not in ALLOWED_SUGGESTIONS:
                        continue

                    # odds lookup
                    odds = None
                    book = None
                    if mk == "BTTS":
                        d = odds_map.get("BTTS", {})
                        tgt = "Yes" if sug.endswith("Yes") else "No"
                        if tgt in d:
                            odds, book = d[tgt]["odds"], d[tgt]["book"]
                    elif mk == "1X2":
                        d = odds_map.get("1X2", {})
                        tgt = "Home" if sug == "Home Win" else ("Away" if sug == "Away Win" else None)
                        if tgt and tgt in d:
                            odds, book = d[tgt]["odds"], d[tgt]["book"]
                    elif mk.startswith("Over/Under"):
                        ln = _parse_ou_line_from_suggestion(sug)
                        d = odds_map.get(f"OU_{_fmt_line(ln)}", {}) if ln is not None else {}
                        tgt = "Over" if sug.startswith("Over") else "Under"
                        if tgt in d:
                            odds, book = d[tgt]["odds"], d[tgt]["book"]

                    # gate on presence (and floor/ceiling) + compute EV
                    pass_odds, odds2, book2, _ = _price_gate(mk, sug, fid)
                    if not pass_odds:
                        continue
                    if odds is None:
                        odds = odds2
                        book = book2

                    ev_pct = None
                    if odds is not None:
                        edge = _ev(prob, float(odds))
                        ev_pct = round(edge * 100.0, 1)
                        if int(round(edge * 10000)) < EDGE_MIN_BPS:
                            continue
                    else:
                        # odds are mandatory by default (ALLOW_TIPS_WITHOUT_ODDS=0); skip
                        continue

                    rank_score = (prob ** 1.2) * (1 + (ev_pct or 0) / 100.0)
                    ranked.append((mk, sug, prob, odds, book, ev_pct, rank_score))

                if not ranked:
                    continue

                ranked.sort(key=lambda x: x[6], reverse=True)

                per_match = 0
                base_now = int(time.time())

                for idx, (market_txt, suggestion, prob, odds, book, ev_pct, _rank) in enumerate(ranked):
                    if PER_LEAGUE_CAP > 0 and per_league_counter.get(league_id, 0) >= PER_LEAGUE_CAP:
                        continue

                    created_ts = base_now + idx
                    raw = float(prob)
                    prob_pct = round(raw * 100.0, 1)

                    try:
                        with db_conn() as c2:
                            c2.execute(
                                "INSERT INTO tips("
                                "match_id,league_id,league,home,away,market,suggestion,"
                                "confidence,confidence_raw,score_at_tip,minute,created_ts,"
                                "odds,book,ev_pct,sent_ok"
                                ") VALUES ("
                                "%s,%s,%s,%s,%s,%s,%s,"
                                "%s,%s,%s,%s,%s,"
                                "%s,%s,%s,%s"
                                ")",
                                (
                                    fid, league_id, league, home, away,
                                    market_txt, suggestion,
                                    float(prob_pct), raw, score, minute, created_ts,
                                    (float(odds) if odds is not None else None),
                                    (book or None),
                                    (float(ev_pct) if ev_pct is not None else None),
                                    0,
                                ),
                             )

                            sent = send_telegram(_format_tip_message(home, away, league, minute, score, suggestion, float(prob_pct), feat, odds, book, ev_pct))
                            if sent:
                                c2.execute("UPDATE tips SET sent_ok=1 WHERE match_id=%s AND created_ts=%s", (fid, created_ts))
                    except Exception as e:
                        log.exception("[PROD] insert/send failed: %s", e)
                        continue

                    saved += 1
                    per_match += 1
                    per_league_counter[league_id] = per_league_counter.get(league_id, 0) + 1

                    if MAX_TIPS_PER_SCAN and saved >= MAX_TIPS_PER_SCAN:
                        break
                    if per_match >= max(1, PREDICTIONS_PER_MATCH):
                        break

                if MAX_TIPS_PER_SCAN and saved >= MAX_TIPS_PER_SCAN:
                    break

            except Exception as e:
                log.exception("[PROD] match loop failed: %s", e)
                continue

    log.info("[PROD] saved=%d live_seen=%d", saved, live_seen)
    return saved, live_seen

def prematch_scan_save() -> int:
    saved = 0
    to_insert: List[Tuple] = []
    try:
        with db_conn() as c:
            c.execute("""
                SELECT fixture_id, league_name, home, away, kickoff
                FROM fixtures WHERE status IN ('NS','TBD')
            """)
            rows = c.fetchall()

        for fid, league, home, away, kickoff in rows:
            best = _best_candidate_for_fixture(fid)
            if not best:
                continue
            market, suggestion, prob, odds, book, ev_pct = best
            now_ts = int(time.time())
            to_insert.append((
                fid, league, home, away,
                market, suggestion,
                prob * 100.0, prob, now_ts, odds, book, ev_pct, 1
            ))

        if to_insert:
            with db_conn() as c:
                c.execute_values(
                    """
                    INSERT INTO tips(
                        match_id, league, home, away, market, suggestion,
                        confidence, confidence_raw, created_ts, odds, book, ev_pct, sent_ok
                    ) VALUES %s
                    ON CONFLICT DO NOTHING
                    """,
                    to_insert, page_size=200
                )
            saved = len(to_insert)
    except Exception as e:
        log.exception("[prematch] failed: %s", e)
    return saved

def daily_accuracy_digest() -> Optional[str]:
    today = _now_dt().astimezone(BERLIN_TZ).date()
    import datetime as _dt
    yesterday = today - _dt.timedelta(days=1)
    msg = None
    try:
        with db_conn() as c:
            c.execute("""
                SELECT t.suggestion, t.odds,
                       r.final_goals_h, r.final_goals_a, r.btts_yes
                FROM tips t
                JOIN match_results r ON t.match_id = r.match_id
                WHERE (to_timestamp(t.created_ts) AT TIME ZONE 'Europe/Berlin')::date = %s::date
                  AND t.sent_ok = 1
            """, (yesterday,))
            rows = c.fetchall()

        if not rows:
            return None

        wins, pnl = 0, 0.0
        for sug, odds, gh, ga, btts_yes in rows:
            s = str(sug or "")
            gh = int(gh or 0); ga = int(ga or 0)
            total = gh + ga
            win = False
            if s.startswith("Over") or s.startswith("Under"):
                try:
                    line = float(s.split()[1])
                except Exception:
                    line = 2.5
                win = (total > line) if s.startswith("Over") else (total < line)
            elif s == "BTTS: Yes":
                win = bool(btts_yes)
            elif s == "BTTS: No":
                win = not bool(btts_yes)
            elif s == "Home Win":
                win = gh > ga
            elif s == "Away Win":
                win = ga > gh

            if win:
                wins += 1
            try:
                o = float(odds)
                pnl += (o - 1.0) if win else -1.0
            except Exception:
                pass

        hit = wins / len(rows) * 100.0
        msg = f"📊 Digest {yesterday} — {len(rows)} bets | Hit {hit:.1f}% | ROI {pnl:+.2f}u"
        send_telegram(msg)
    except Exception as e:
        log.exception("[digest] failed: %s", e)
    return msg

def send_match_of_the_day() -> bool:
    candidates: List[Tuple[float, dict, str, str, float, Optional[float], Optional[str], Optional[float]]] = []
    try:
        with db_conn() as c:
            c.execute("""
                SELECT fixture_id, league_name, home, away, kickoff
                FROM fixtures WHERE status IN ('NS','TBD')
            """)
            rows = c.fetchall()

        for fid, league, home, away, kickoff in rows:
            best = _best_candidate_for_fixture(fid)
            if not best:
                continue
            market, suggestion, prob, odds, book, ev_pct = best
            score = prob + ((ev_pct or 0.0) / 100.0)
            if prob >= MOTD_CONF_MIN and (ev_pct or 0.0) >= (MOTD_EV_MIN * 100.0):
                candidates.append((score, {"league": league, "home": home, "away": away, "kickoff": kickoff},
                                   market, suggestion, prob, odds, book, ev_pct))

        if not candidates:
            send_telegram("🌟 MOTD — no high-confidence pick today.")
            return False

        candidates.sort(key=lambda x: x[0], reverse=True)
        _, match, market, suggestion, prob, odds, book, ev_pct = candidates[0]
        msg = "🌟 <b>Match of the Day</b>\n" + _fmt_tip_message(match, market, suggestion, prob, odds, book, ev_pct)
        send_telegram(msg)
        return True
    except Exception as e:
        log.exception("[motd] failed: %s", e)
        return False

def retry_unsent_tips(minutes: int = 30, limit: int = 200) -> int:
    cutoff = int(time.time()) - minutes * 60
    retried = 0
    try:
        with db_conn() as c:
            c.execute("""
                SELECT match_id, league, home, away, market, suggestion,
                       confidence, confidence_raw, score_at_tip, minute,
                       created_ts, odds, book, ev_pct
                FROM tips
                WHERE sent_ok = 0 AND created_ts >= %s
                ORDER BY created_ts ASC
                LIMIT %s
            """, (cutoff, limit))
            rows = c.fetchall()

        delivered: List[Tuple[int, int]] = []
        for (
            mid, league, home, away, market, sugg,
            conf_pct, conf_raw, score, minute, cts,
            odds, book, ev_pct
        ) in rows:
            pct = float(conf_pct if conf_pct is not None else (100.0 * float(conf_raw or 0.0)))
            odds_str = "-" if odds is None else f"{float(odds):.2f}"
            msg = f"♻️ RETRY\n{league}: {home} vs {away}\nTip: {sugg}\nConf: {pct:.1f}%\nOdds: {odds_str}"
            ok = send_telegram(msg)
            if ok:
                delivered.append((mid, cts))

        if delivered:
            with db_conn() as c2:
                c2.execute_values(
                    """
                    UPDATE tips AS t SET sent_ok=1
                    FROM (VALUES %s) AS v(match_id, created_ts)
                    WHERE t.match_id=v.match_id AND t.created_ts=v.created_ts
                    """,
                    delivered
                )
            retried = len(delivered)
            log.info("[retry] resent %d", retried)
    except Exception as e:
        log.exception("[retry] failed: %s", e)
    return retried

def backfill_results_for_open_matches(limit=300) -> int:
    try:
        with db_conn() as c:
            c.execute(
                "SELECT fixture_id FROM fixtures WHERE status NOT IN ('FT','AET','PEN') "
                "ORDER BY last_update ASC LIMIT %s",
                (limit,),
            )
            ids = [r[0] for r in c.fetchall()]
        if not ids:
            return 0
        rows = list(fetch_results_for_fixtures(ids))
        if not rows:
            return 0
        return update_match_results(rows)
    except Exception as e:
        log.exception("[backfill] failed: %s", e)
        return 0

# ───────────────────────────────────────────────────────────────────────────────
# Flask app, admin auth, request-id middleware, thresholds application
# ───────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)

def get_logger() -> logging.Logger:
    return log

# Small DB helper
def _scalar(c, sql: str, params: tuple = ()):
    c.execute(sql, params)
    row = c.fetchone()
    return row[0] if row else None

# ───────── Admin auth helpers ─────────
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")
ALLOW_QUERY_KEY = os.getenv("ALLOW_QUERY_KEY", "0").lower() in {"1","true","yes"}

def _constant_time_eq(a: str, b: str) -> bool:
    a = a or ""
    b = b or ""
    try:
        return hmac.compare_digest(a, b)
    except Exception:
        return a == b

def _require_admin():
    if not ADMIN_API_KEY:
        abort(401)
    key = request.headers.get("X-API-Key")
    if not key and ALLOW_QUERY_KEY:
        key = request.args.get("key") or ((request.json or {}).get("key") if request.is_json else None)
    if not key or not _constant_time_eq(key, ADMIN_API_KEY):
        abort(401)

# ───────── Request middleware ─────────
@app.before_request
def _assign_request_id():
    g.request_start = time.time()
    g.request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex

@app.after_request
def _inject_request_id(resp):
    resp.headers["X-Request-ID"] = getattr(g, "request_id", "")
    return resp

# ───────── Night rest window (Berlin) ─────────
REST_START_HOUR_BERLIN = env_int("REST_START_HOUR_BERLIN", 23)
REST_END_HOUR_BERLIN   = env_int("REST_END_HOUR_BERLIN", 7)

def is_rest_window_now() -> bool:
    h = datetime.now(BERLIN_TZ).hour
    if REST_START_HOUR_BERLIN <= REST_END_HOUR_BERLIN:
        return REST_START_HOUR_BERLIN <= h < REST_END_HOUR_BERLIN
    return (h >= REST_START_HOUR_BERLIN) or (h < REST_END_HOUR_BERLIN)

# ───────── Threshold application (runtime) ─────────
def _apply_tuned_thresholds():
    """
    Load CONF_MIN / EV_MIN / MOTD_* from DB settings and apply to module globals.
    """
    try:
        th = _load_thresholds_from_settings()
        globals()["CONF_MIN"] = float(th["CONF_MIN"])
        globals()["EV_MIN"] = float(th["EV_MIN"])
        globals()["MOTD_CONF_MIN"] = float(th["MOTD_CONF_MIN"])
        globals()["MOTD_EV_MIN"] = float(th["MOTD_EV_MIN"])
        log.info(
            "[THRESH] applied: CONF_MIN=%.2f EV_MIN=%.2f MOTD_CONF_MIN=%.2f MOTD_EV_MIN=%.2f",
            CONF_MIN, EV_MIN, MOTD_CONF_MIN, MOTD_EV_MIN
        )
    except Exception as e:
        log.warning("[THRESH] failed to apply: %s", e, exc_info=True)

# ───────────────────────────────────────────────────────────────────────────────
# Scheduler (leader-controlled), jobs & locking
# ───────────────────────────────────────────────────────────────────────────────
RUN_SCHEDULER = os.getenv("RUN_SCHEDULER", "1").lower() in {"1","true","yes"}
SCHEDULER_LEADER = os.getenv("SCHEDULER_LEADER", "1").lower() in {"1","true","yes"}

SCAN_INTERVAL_SEC   = env_int("SCAN_INTERVAL_SEC", 300)
BACKFILL_EVERY_MIN  = env_int("BACKFILL_EVERY_MIN", 15)

TRAIN_ENABLE     = os.getenv("TRAIN_ENABLE", "1").lower() in {"1","true","yes"}
TRAIN_HOUR_UTC   = env_int("TRAIN_HOUR_UTC", 2)
TRAIN_MINUTE_UTC = env_int("TRAIN_MINUTE_UTC", 12)

AUTO_TUNE_ENABLE = os.getenv("AUTO_TUNE_ENABLE", "0").lower() in {"1","true","yes"}

DAILY_ACCURACY_DIGEST_ENABLE = os.getenv("DAILY_ACCURACY_DIGEST_ENABLE", "1").lower() in {"1","true","yes"}
DAILY_ACCURACY_HOUR   = env_int("DAILY_ACCURACY_HOUR", 8)
DAILY_ACCURACY_MINUTE = env_int("DAILY_ACCURACY_MINUTE", 0)

MOTD_ENABLE = os.getenv("MOTD_ENABLE", "1").lower() in {"1","true","yes"}
MOTD_HOUR   = env_int("MOTD_HOUR", 10)
MOTD_MINUTE = env_int("MOTD_MINUTE", 0)

SEND_BOOT_TELEGRAM = os.getenv("SEND_BOOT_TELEGRAM", "0").lower() in {"1","true","yes"}

_scheduler_started = False
_scheduler_ref: Optional[BackgroundScheduler] = None

def _run_with_pg_lock(lock_key: int, fn: Callable, *a, **k):
    lg = get_logger()
    try:
        with db_conn() as c:
            c.execute("SELECT pg_try_advisory_lock(%s)", (lock_key,))
            row = c.fetchone()
            got = bool(row and row[0])
            if not got:
                lg.info("[LOCK %s] busy; skipped.", lock_key)
                return None
            try:
                return fn(*a, **k)
            finally:
                try:
                    c.execute("SELECT pg_advisory_unlock(%s)", (lock_key,))
                except Exception:
                    pass
    except Exception as e:
        lg.exception("[LOCK %s] failed: %s", lock_key, e)
        return None

def _start_scheduler_once():
    """
    Start BackgroundScheduler exactly once per process if this instance is leader.
    """
    global _scheduler_started, _scheduler_ref
    if _scheduler_started or not RUN_SCHEDULER or not SCHEDULER_LEADER:
        if not SCHEDULER_LEADER:
            log.info("[SCHED] disabled for this process (SCHEDULER_LEADER=false)")
        return

    # Avoid double-start under Flask reloader
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        pass
    elif os.environ.get("FLASK_ENV") == "development" and os.environ.get("WERKZEUG_RUN_MAIN") is None:
        return

    try:
        sched = BackgroundScheduler(timezone=TZ_UTC, job_defaults={"coalesce": True, "max_instances": 1})

        def _scan_job():
            if is_rest_window_now():
                log.info("[scan] rest window active — skipped.")
                return
            return _run_with_pg_lock(1001, production_scan)

        # Live/prematch scan
        sched.add_job(
            _scan_job, "interval",
            seconds=SCAN_INTERVAL_SEC, id="scan", max_instances=1, coalesce=True, misfire_grace_time=60
        )

        # Results backfill
        sched.add_job(
            lambda: _run_with_pg_lock(1002, backfill_results_for_open_matches, 400),
            "interval", minutes=BACKFILL_EVERY_MIN, id="backfill", max_instances=1, coalesce=True, misfire_grace_time=120
        )

        # Daily digest
        if DAILY_ACCURACY_DIGEST_ENABLE:
            sched.add_job(
                lambda: _run_with_pg_lock(1003, daily_accuracy_digest),
                CronTrigger(hour=DAILY_ACCURACY_HOUR, minute=DAILY_ACCURACY_MINUTE, timezone=BERLIN_TZ),
                id="digest", max_instances=1, coalesce=True, misfire_grace_time=3600
            )

        # MOTD
        if MOTD_ENABLE:
            sched.add_job(
                lambda: _run_with_pg_lock(1004, send_match_of_the_day),
                CronTrigger(hour=MOTD_HOUR, minute=MOTD_MINUTE, timezone=BERLIN_TZ),
                id="motd", max_instances=1, coalesce=True, misfire_grace_time=3600
            )

        # Training
        if TRAIN_ENABLE and _TRAIN_MODULE_AVAILABLE:
            sched.add_job(
                lambda: _run_with_pg_lock(1005, _train_models),
                CronTrigger(hour=TRAIN_HOUR_UTC, minute=TRAIN_MINUTE_UTC, timezone=TZ_UTC),
                id="train", max_instances=1, coalesce=True, misfire_grace_time=3600
            )
        elif TRAIN_ENABLE and not _TRAIN_MODULE_AVAILABLE:
            log.warning("[SCHED] TRAIN_ENABLE=1 but train_models module unavailable; skipping train job.")

        # Auto-tune
        if AUTO_TUNE_ENABLE and _TRAIN_MODULE_AVAILABLE:
            sched.add_job(
                lambda: _run_with_pg_lock(1006, _auto_tune_thresholds, 14),
                CronTrigger(hour=4, minute=7, timezone=TZ_UTC),
                id="auto_tune", max_instances=1, coalesce=True, misfire_grace_time=3600
            )
        elif AUTO_TUNE_ENABLE and not _TRAIN_MODULE_AVAILABLE:
            log.warning("[SCHED] AUTO_TUNE_ENABLE=1 but training module unavailable; skipping auto_tune job.")

        # Retry unsent
        sched.add_job(
            lambda: _run_with_pg_lock(1007, retry_unsent_tips, 30, 200),
            "interval", minutes=10, id="retry", max_instances=1, coalesce=True, misfire_grace_time=120
        )

        sched.start()
        _scheduler_started = True
        _scheduler_ref = sched

        if SEND_BOOT_TELEGRAM:
            try:
                send_telegram("🚀 goalsniper AI live and scanning (night rest 23:00–07:00 Berlin)")
            except Exception:
                log.warning("Boot telegram failed", exc_info=True)

        log.info("[SCHED] started (scan=%ss)", SCAN_INTERVAL_SEC)

    except Exception as e:
        log.exception("[SCHED] failed: %s", e)

def _stop_scheduler(*_args):
    try:
        if _scheduler_ref and _scheduler_ref.running:
            log.info("[SCHED] shutting down…")
            _scheduler_ref.shutdown(wait=False)
    except Exception:
        log.warning("[SCHED] shutdown error", exc_info=True)

# ───────────────────────────────────────────────────────────────────────────────
# Routes: health, stats, admin actions
# ───────────────────────────────────────────────────────────────────────────────

@app.route("/")
def root():
    return jsonify({
        "ok": True, "name": "goalsniper", "mode": "FULL_AI", "scheduler": RUN_SCHEDULER,
        "request_id": getattr(g, "request_id", None)
    })

@app.route("/health", methods=["GET", "HEAD"])
@app.route("/healthz", methods=["GET", "HEAD"])
@app.route("/_health", methods=["GET", "HEAD"])
def health():
    """
    Readiness endpoint:
      - Fast 200 by default (no DB)
      - ?db=1 adds a DB ping (still 200 even if DB is down)
      - ?deep=1 adds counts (for humans)
    """
    want_db = request.args.get("db") in ("1", "true", "yes")
    deep = request.args.get("deep") in ("1", "true", "yes")

    resp: dict[str, Any] = {"ok": True, "service": "goalsniper", "scheduler": RUN_SCHEDULER}

    # Fast path for platform probes
    if not want_db and not deep:
        return jsonify(resp), 200

    # DB ping (best-effort)
    try:
        t0 = time.time()
        with db_conn() as c:
            c.execute("SELECT 1")
        resp["db"] = "ok"
        resp["db_ms"] = int((time.time() - t0) * 1000)
    except Exception as e:
        resp["db"] = "down"
        resp["error"] = str(e)
        return jsonify(resp), 200

    if deep:
        try:
            with db_conn() as c2:
                (n,) = c2.execute("SELECT COUNT(*) FROM tips").fetchone()
            resp["tips_count"] = int(n)
        except Exception as e:
            resp["tips_count_error"] = str(e)

    return jsonify(resp), 200

@app.route("/stats")
def stats():
    try:
        now = int(time.time())
        day_ago = now - 24 * 3600
        week_ago = now - 7 * 24 * 3600
        with db_conn() as c:
            t24 = int(_scalar(
                c, "SELECT COUNT(*) FROM tips WHERE created_ts >= %s AND suggestion <> 'HARVEST'",
                (day_ago,)
            ) or 0)
            t7d = int(_scalar(
                c, "SELECT COUNT(*) FROM tips WHERE created_ts >= %s AND suggestion <> 'HARVEST'",
                (week_ago,)
            ) or 0)
            unsent = int(_scalar(
                c, "SELECT COUNT(*) FROM tips WHERE sent_ok=0 AND created_ts >= %s",
                (week_ago,)
            ) or 0)

            c.execute("""
                SELECT t.suggestion, t.odds, r.final_goals_h, r.final_goals_a, r.btts_yes
                FROM tips t
                JOIN match_results r ON r.match_id = t.match_id
                WHERE t.created_ts >= %s
                  AND t.suggestion <> 'HARVEST'
                  AND t.sent_ok = 1
            """, (week_ago,))
            rows = c.fetchall() or []

        graded = wins = 0
        stake = pnl = 0.0

        for (sugg, odds, gh, ga, btts) in rows:
            gh = int(gh or 0)
            ga = int(ga or 0)
            total = gh + ga
            result: Optional[bool] = None

            s = str(sugg or "")
            if s.startswith("Over") or s.startswith("Under"):
                line: Optional[float] = None
                for tok in s.split():
                    try:
                        line = float(tok)
                        break
                    except Exception:
                        pass
                if line is None:
                    continue
                result = (total > line) if s.startswith("Over") else (total < line)
            elif s == "BTTS: Yes":
                result = (gh > 0 and ga > 0)
            elif s == "BTTS: No":
                result = not (gh > 0 and ga > 0)
            elif s == "Home Win":
                result = (gh > ga)
            elif s == "Away Win":
                result = (ga > gh)

            if result is None:
                continue

            graded += 1
            if result:
                wins += 1

            if odds is not None:
                try:
                    o = float(odds)
                    stake += 1.0
                    pnl += (o - 1.0) if result else -1.0
                except Exception:
                    pass

        acc = (100.0 * wins / graded) if graded else 0.0
        roi = (100.0 * pnl / stake) if stake > 0 else 0.0

        return jsonify({
            "ok": True,
            "tips_last_24h": int(t24),
            "tips_last_7d": int(t7d),
            "unsent_last_7d": int(unsent),
            "graded_last_7d": int(graded),
            "wins_last_7d": int(wins),
            "accuracy_last_7d_pct": round(acc, 1),
            "roi_last_7d_pct": round(roi, 1),
        })
    except Exception as e:
        log.exception("/stats failed: %s", e)
        return jsonify({"ok": False, "error": str(e)}), 500

# ── Admin endpoints ──
@app.route("/admin/scan", methods=["POST", "GET"])
def http_scan():
    _require_admin()
    if is_rest_window_now():
        return jsonify({"ok": True, "saved": 0, "live_seen": 0, "skipped": "rest-window"}), 200
    s, l = production_scan()
    return jsonify({"ok": True, "saved": s, "live_seen": l})

@app.route("/admin/train", methods=["POST", "GET"])
def http_train():
    _require_admin()
    if not TRAIN_ENABLE:
        return jsonify({"ok": False, "reason": "training disabled"}), 400
    if not _TRAIN_MODULE_AVAILABLE:
        return jsonify({"ok": False, "reason": "training module not available"}), 400
    res = _train_models()
    return jsonify({"ok": True, "result": res})

@app.route("/admin/digest", methods=["POST", "GET"])
def http_digest():
    _require_admin()
    msg = daily_accuracy_digest()
    return jsonify({"ok": True, "sent": bool(msg)})

@app.route("/admin/auto-tune", methods=["POST", "GET"])
def http_auto_tune():
    _require_admin()
    if not _TRAIN_MODULE_AVAILABLE:
        return jsonify({"ok": False, "reason": "auto_tune not available"}), 400
    tuned = _auto_tune_thresholds(14)
    try:
        if tuned.get("ok"):
            _apply_tuned_thresholds()
    except Exception:
        log.warning("[THRESH] failed to apply after auto-tune", exc_info=True)
    return jsonify({"ok": True, "tuned": tuned})

@app.route("/admin/apply-thresholds", methods=["POST"])
def http_apply_thresholds():
    _require_admin()
    try:
        _apply_tuned_thresholds()
        return jsonify({"ok": True})
    except Exception as e:
        log.exception("apply-thresholds failed: %s", e)
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/admin/motd", methods=["POST", "GET"])
def http_motd():
    _require_admin()
    ok = send_match_of_the_day()
    return jsonify({"ok": bool(ok)})

@app.route("/admin/backfill-results", methods=["GET"])
def http_backfill_results():
    _require_admin()
    try:
        max_rows = int(request.args.get("max", "400"))
    except Exception:
        max_rows = 400
    n = backfill_results_for_open_matches(max_rows)
    return jsonify({"ok": True, "updated": int(n)})

@app.route("/admin/prematch-scan", methods=["GET"])
def http_prematch_scan():
    _require_admin()
    if is_rest_window_now():
        return jsonify({"ok": True, "saved": 0, "skipped": "rest-window"}), 200
    saved = prematch_scan_save()
    return jsonify({"ok": True, "saved": int(saved)})

@app.route("/admin/retry-unsent", methods=["GET"])
def http_retry_unsent():
    _require_admin()
    try:
        minutes = int(request.args.get("minutes", "30"))
    except Exception:
        minutes = 30
    try:
        limit = int(request.args.get("limit", "200"))
    except Exception:
        limit = 200
    n = retry_unsent_tips(minutes=minutes, limit=limit)
    return jsonify({"ok": True, "resent": int(n)})

@app.route("/admin/rest-window", methods=["GET","POST"])
def http_rest_window():
    _require_admin()
    global REST_START_HOUR_BERLIN, REST_END_HOUR_BERLIN
    if request.method == "GET":
        return jsonify({
            "ok": True,
            "start_hour": REST_START_HOUR_BERLIN,
            "end_hour": REST_END_HOUR_BERLIN,
            "active_now": bool(is_rest_window_now()),
            "tz": "Europe/Berlin"
        })
    body = request.get_json(silent=True) or {}
    try:
        if "start_hour" in body:
            REST_START_HOUR_BERLIN = int(body["start_hour"])
        if "end_hour" in body:
            REST_END_HOUR_BERLIN = int(body["end_hour"])
        return jsonify({"ok": True, "start_hour": REST_START_HOUR_BERLIN, "end_hour": REST_END_HOUR_BERLIN})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

# ───────────────────────────────────────────────────────────────────────────────
# Boot, signal handlers, and Flask entrypoint
# ───────────────────────────────────────────────────────────────────────────────

def _handle_sigterm(*_):
    log.info("[BOOT] SIGTERM received — shutting down gracefully")
    _stop_scheduler()
    sys.exit(0)

def _handle_sigint(*_):
    log.info("[BOOT] SIGINT received — shutting down gracefully")
    _stop_scheduler()
    sys.exit(0)

def _on_boot():
    try:
        init_db()
    except Exception as e:
        log.exception("init_db failed (continuing to serve): %s", e)
    try:
        _apply_tuned_thresholds()   # apply thresholds from settings at boot
    except Exception as e:
        log.warning("[THRESH] apply at boot failed (continuing): %s", e, exc_info=True)
    try:
        _start_scheduler_once()
    except Exception as e:
        log.exception("scheduler start failed (continuing to serve): %s", e)
    try:
        warm_model()
    except Exception:
        pass

def _on_boot():
    _init_pool(); init_db(); set_setting("boot_ts", str(int(time.time())))

# Ensure clean scheduler shutdown on signals
signal.signal(signal.SIGTERM, _handle_sigterm)
signal.signal(signal.SIGINT, _handle_sigint)

# Run boot sequence
_on_boot()

if __name__ == "__main__":
    # For local/dev usage (Railway uses startCommand)
    app.run(host=os.getenv("HOST","0.0.0.0"), port=int(os.getenv("PORT","8080")))
