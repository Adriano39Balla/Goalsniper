# main.py
# goalsniper — single-file app (Flask API, scheduler, DB, odds, predictor, scan, results)
# NOTE: training/auto-tune lives in separate train_models.py as requested.

from __future__ import annotations

import os
import re
import sys
import time
import uuid
import hmac
import json
import html
import math
import random
import signal
import atexit
import logging
import threading
import statistics
from typing import Any, Dict, List, Tuple, Optional, Iterable, Callable
from datetime import datetime
from zoneinfo import ZoneInfo

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
from psycopg2.extras import DictCursor
from psycopg2.extras import execute_values

# training lives as a separate file, per your 4-file plan
from train_models import (
    train_models, auto_tune_thresholds, load_thresholds_from_settings
)

# ───────────────────────────────────────────────────────────────────────────────
# Logging with request-id injection (safe across threads)
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

logging.setLogRecordFactory(_record_factory_with_request_id())
logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s %(request_id)s - %(message)s")
log = logging.getLogger("goalsniper")

# ───────────────────────────────────────────────────────────────────────────────
# Env helpers
# ───────────────────────────────────────────────────────────────────────────────

def env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() not in {"0", "false", "no", "off", ""}

def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default

TZ_UTC = ZoneInfo("UTC")
BERLIN_TZ = ZoneInfo("Europe/Berlin")

# ───────────────────────────────────────────────────────────────────────────────
# Telegram utils (inline so main.py is self-contained)
# ───────────────────────────────────────────────────────────────────────────────

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_IDS = [c.strip() for c in os.getenv("TELEGRAM_CHAT_ID", "").split(",") if c.strip()]
THREAD_ID = os.getenv("TELEGRAM_THREAD_ID")
SEND_MESSAGES = os.getenv("SEND_MESSAGES", "true").lower() in ("1", "true", "yes")

PARSE_MODE = os.getenv("TELEGRAM_PARSE_MODE", "HTML").strip()
ALLOW_HTML_TAGS = os.getenv("TELEGRAM_ALLOW_HTML_TAGS", "0").lower() in ("1", "true", "yes")

API_BASE_TG = f"https://api.telegram.org/bot{BOT_TOKEN}" if BOT_TOKEN else ""
TG_HARD_MAX = 4096
MAX_LEN_TG = min(int(os.getenv("TELEGRAM_MAX_LEN", TG_HARD_MAX)), TG_HARD_MAX)
FAIL_COOLDOWN_SEC = int(os.getenv("TELEGRAM_FAIL_COOLDOWN_SEC", "60"))
_last_fail_ts_tg = 0

_session_tg = requests.Session()
_retry_tg = Retry(
    total=3, connect=3, read=3, backoff_factor=0.7,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=frozenset(["POST"]),
    respect_retry_after_header=True, raise_on_status=False,
)
_adapter_tg = HTTPAdapter(max_retries=_retry_tg, pool_connections=32, pool_maxsize=64)
_session_tg.mount("https://", _adapter_tg)
_session_tg.mount("http://", _adapter_tg)

_MD_V2_CHARS = r'[_*[\]()~`>#+\-=|{}.!]'

def _escape_md(text: str) -> str:
    return re.sub(r'([_*`\[\]])', r'\\\1', text)

def _escape_md_v2(text: str) -> str:
    return re.sub(r'(' + _MD_V2_CHARS + r')', r'\\\1', text)

def _tg_prepare(text: str) -> str:
    t = str(text or "")
    if PARSE_MODE.lower() == "html":
        return (t if ALLOW_HTML_TAGS else html.escape(t))[:MAX_LEN_TG]
    if PARSE_MODE.lower() == "markdownv2":
        return _escape_md_v2(t)[:MAX_LEN_TG]
    if PARSE_MODE.lower() == "markdown":
        return _escape_md(t)[:MAX_LEN_TG]
    return t[:MAX_LEN_TG]

def _tg_split(s: str, limit: int) -> List[str]:
    if len(s) <= limit:
        return [s]
    out: List[str] = []
    cur = s
    while len(cur) > limit:
        cut = cur.rfind("\n\n", 0, limit)
        if cut == -1: cut = cur.rfind("\n", 0, limit)
        if cut == -1: cut = cur.rfind(" ", 0, limit)
        if cut == -1: cut = limit
        out.append(cur[:cut].rstrip())
        cur = cur[cut:].lstrip()
    if cur:
        out.append(cur)
    return out

def send_telegram(text: str, disable_preview: bool = True) -> bool:
    global _last_fail_ts_tg
    if not (BOT_TOKEN and CHAT_IDS and SEND_MESSAGES):
        return False
    if _last_fail_ts_tg and time.time() - _last_fail_ts_tg < FAIL_COOLDOWN_SEC:
        return False

    url = f"{API_BASE_TG}/sendMessage"
    body = _tg_prepare(text)
    parts = _tg_split(body, MAX_LEN_TG)

    ok_any = False
    for chat_id in CHAT_IDS:
        for idx, part in enumerate(parts, 1):
            payload = {"chat_id": chat_id, "text": part, "disable_web_page_preview": disable_preview}
            if PARSE_MODE.lower() in ("html", "markdown", "markdownv2"):
                payload["parse_mode"] = "MarkdownV2" if PARSE_MODE.lower() == "markdownv2" else PARSE_MODE
            if THREAD_ID:
                try:
                    payload["message_thread_id"] = int(THREAD_ID)
                except Exception:
                    pass

            backoff = 1.5
            for attempt in range(3):
                try:
                    r = _session_tg.post(url, json=payload, timeout=(3, 10))
                    if r.status_code == 200:
                        ok_any = True
                        break
                    if r.status_code == 429:
                        try:
                            retry_after = int(r.json().get("parameters", {}).get("retry_after", 5))
                        except Exception:
                            retry_after = 5
                        time.sleep(retry_after)
                        continue
                    time.sleep(backoff + random.uniform(0, 0.4))
                    backoff *= 2
                except Exception:
                    time.sleep(backoff + random.uniform(0, 0.4))
                    backoff *= 2
            if idx < len(parts):
                time.sleep(0.25 + random.uniform(0, 0.15))

    if not ok_any:
        _last_fail_ts_tg = time.time()
    return ok_any

def send_telegram_safe(text: str, **kwargs) -> bool:
    try:
        return send_telegram(text, **kwargs)
    except Exception as e:
        log.warning("[TELEGRAM] send_telegram_safe failed: %s", e)
        return False

# ───────────────────────────────────────────────────────────────────────────────
# Postgres — connection pool, session settings, schema/bootstrap
# ───────────────────────────────────────────────────────────────────────────────

DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise SystemExit("DATABASE_URL is required")

def _should_force_ssl(url: str) -> bool:
    if not url.startswith(("postgres://", "postgresql://")):
        return False
    v = os.getenv("DB_SSLMODE_REQUIRE", "1").strip().lower()
    return v not in {"0", "false", "no", ""}

if _should_force_ssl(DB_URL) and "sslmode=" not in DB_URL:
    DB_URL = DB_URL + (("&" if "?" in DB_URL else "?") + "sslmode=require")

STMT_TIMEOUT_MS = env_int("PG_STATEMENT_TIMEOUT_MS", 15000)
LOCK_TIMEOUT_MS = env_int("PG_LOCK_TIMEOUT_MS", 2000)
IDLE_TX_TIMEOUT_MS = env_int("PG_IDLE_TX_TIMEOUT_MS", 30000)
FORCE_UTC = os.getenv("PG_FORCE_UTC", "1").strip().lower() not in {"0", "false", "no", ""}

POOL: Optional[SimpleConnectionPool] = None

def _apply_session_settings(conn) -> None:
    cur = conn.cursor()
    try:
        if FORCE_UTC:
            cur.execute("SET TIME ZONE 'UTC'")
        cur.execute("SET statement_timeout = %s", (STMT_TIMEOUT_MS,))
        cur.execute("SET lock_timeout = %s", (LOCK_TIMEOUT_MS,))
        cur.execute("SET idle_in_transaction_session_timeout = %s", (IDLE_TX_TIMEOUT_MS,))
    finally:
        cur.close()

def _init_pool():
    global POOL
    if POOL is None:
        minconn = int(os.getenv("DB_POOL_MIN", "1"))
        maxconn = int(os.getenv("DB_POOL_MAX", "4"))
        POOL = SimpleConnectionPool(minconn=minconn, maxconn=maxconn, dsn=DB_URL)
        log.info("[DB] pool created (min=%d, max=%d)", minconn, maxconn)
        atexit.register(_close_pool)

def _close_pool():
    global POOL
    if POOL is not None:
        try:
            POOL.closeall()
            log.info("[DB] pool closed")
        except Exception:
            log.warning("[DB] pool close failed", exc_info=True)
        POOL = None

MAX_RETRIES = int(os.getenv("DB_MAX_RETRIES", "3"))
BASE_BACKOFF_MS = int(os.getenv("DB_BASE_BACKOFF_MS", "150"))

class _PooledCursor:
    def __init__(self, pool: SimpleConnectionPool, dict_rows: bool = False):
        self.pool = pool
        self.conn = None
        self.cur = None
        self.dict_rows = dict_rows

    def __enter__(self):
        self._acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if self.cur is not None:
                try:
                    self.cur.close()
                except Exception:
                    pass
        finally:
            if self.conn is not None:
                try:
                    self.pool.putconn(self.conn)
                except Exception:
                    pass
            self.conn = None
            self.cur = None

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if self.cur is None:
            raise AttributeError(name)
        return getattr(self.cur, name)

    def execute(self, sql_text: str, params: Tuple | list = ()):
        def _call():
            self.cur.execute(sql_text, params or ())
            return self.cur
        return self._retry_loop(_call)

    def executemany(self, sql_text: str, rows: Iterable[Tuple]):
        rows = list(rows) or []
        def _call():
            self.cur.executemany(sql_text, rows)
            return self.cur
        return self._retry_loop(_call)

    def fetchone(self):
        return self.cur.fetchone() if self.cur else None

    def fetchall(self):
        return self.cur.fetchall() if self.cur else []

    def _acquire(self):
        _init_pool()
        self.pool = POOL  # type: ignore
        attempts = 0
        while True:
            try:
                self.conn = self.pool.getconn()  # type: ignore
                self.conn.autocommit = True
                _apply_session_settings(self.conn)
                self.cur = self.conn.cursor(cursor_factory=DictCursor if self.dict_rows else None)
                self.cur.execute("SELECT 1")
                _ = self.cur.fetchone()
                return
            except (OperationalError, InterfaceError) as e:
                try:
                    if self.cur:
                        try:
                            self.cur.close()
                        except Exception:
                            pass
                    if self.conn:
                        self.pool.putconn(self.conn, close=True)  # type: ignore
                except Exception:
                    pass
                self.conn = None
                self.cur = None
                if attempts >= MAX_RETRIES:
                    log.warning("[DB] acquire failed after %d retries: %s", attempts, e)
                    raise
                backoff = min(2000, BASE_BACKOFF_MS * (2 ** attempts))
                log.warning("[DB] acquire failed, retrying in %dms: %s", backoff, e)
                time.sleep(backoff / 1000.0)
                attempts += 1

    def _reset(self):
        try:
            if self.cur:
                try:
                    self.cur.close()
                except Exception:
                    pass
        finally:
            try:
                if self.conn:
                    self.pool.putconn(self.conn, close=True)  # type: ignore
            except Exception:
                pass
        self.conn = None
        self.cur = None
        self._acquire()

    def _retry_loop(self, fn, *args, **kwargs):
        attempts = 0
        while True:
            try:
                return fn(*args, **kwargs)
            except (OperationalError, InterfaceError) as e:
                if attempts >= MAX_RETRIES:
                    log.warning("[DB] giving up after %d retries: %s", attempts, e)
                    raise
                backoff = min(2000, BASE_BACKOFF_MS * (2 ** attempts))
                log.warning("[DB] transient error, retrying in %dms: %s", backoff, e)
                self._reset()
                time.sleep(backoff / 1000.0)
                attempts += 1

def db_conn(dict_rows: bool = False) -> _PooledCursor:
    _init_pool()
    return _PooledCursor(POOL, dict_rows=dict_rows)  # type: ignore

from contextlib import contextmanager

@contextmanager
def tx(dict_rows: bool = False):
    _init_pool()
    conn = None
    cur = None
    attempts = 0
    while True:
        try:
            conn = POOL.getconn()  # type: ignore
            conn.autocommit = False
            _apply_session_settings(conn)
            cur = conn.cursor(cursor_factory=DictCursor if dict_rows else None)
            cur.execute("SELECT 1")
            cur.fetchone()
            break
        except (OperationalError, InterfaceError):
            if attempts >= MAX_RETRIES:
                raise
            if conn:
                try:
                    POOL.putconn(conn, close=True)  # type: ignore
                except Exception:
                    pass
            attempts += 1
            time.sleep(min(2000, BASE_BACKOFF_MS * (2 ** attempts)) / 1000.0)

    try:
        yield cur
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        try:
            if cur:
                cur.close()
        finally:
            if conn:
                try:
                    POOL.putconn(conn)  # type: ignore
                except Exception:
                    pass

# ───────────────────────────────────────────────────────────────────────────────
# Schema & bootstrap
# ───────────────────────────────────────────────────────────────────────────────

SCHEMA_TABLES_SQL = [
    """
    CREATE TABLE IF NOT EXISTS tips (
        match_id        BIGINT,
        league_id       BIGINT,
        league          TEXT,
        home            TEXT,
        away            TEXT,
        market          TEXT,
        suggestion      TEXT,
        confidence      DOUBLE PRECISION,
        confidence_raw  DOUBLE PRECISION,
        score_at_tip    TEXT,
        minute          INTEGER,
        created_ts      BIGINT,
        odds            DOUBLE PRECISION,
        book            TEXT,
        ev_pct          DOUBLE PRECISION,
        sent_ok         INTEGER DEFAULT 1,
        PRIMARY KEY (match_id, created_ts)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS tip_snapshots (
        match_id   BIGINT,
        created_ts BIGINT,
        payload    TEXT,
        PRIMARY KEY (match_id, created_ts)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS prematch_snapshots (
        match_id   BIGINT PRIMARY KEY,
        created_ts BIGINT,
        payload    TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS feedback (
        id SERIAL PRIMARY KEY,
        match_id BIGINT UNIQUE,
        verdict  INTEGER,
        created_ts BIGINT
    )
    """,
    """CREATE TABLE IF NOT EXISTS settings ( key TEXT PRIMARY KEY, value TEXT )""",
    """
    CREATE TABLE IF NOT EXISTS match_results (
        match_id       BIGINT PRIMARY KEY,
        final_goals_h  INTEGER,
        final_goals_a  INTEGER,
        btts_yes       INTEGER,
        updated_ts     BIGINT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS odds_history (
        match_id    BIGINT,
        captured_ts BIGINT,
        market      TEXT,
        selection   TEXT,
        odds        DOUBLE PRECISION,
        book        TEXT,
        PRIMARY KEY (match_id, market, selection, captured_ts)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS lineups (
        match_id   BIGINT PRIMARY KEY,
        created_ts BIGINT,
        payload    TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS fixtures (
        fixture_id   BIGINT PRIMARY KEY,
        league_name  TEXT,
        home         TEXT,
        away         TEXT,
        kickoff      TIMESTAMPTZ,
        last_update  TIMESTAMPTZ,
        status       TEXT
    )
    """,
]

MIGRATIONS_SQL = [
    "ALTER TABLE tips ADD COLUMN IF NOT EXISTS odds           DOUBLE PRECISION",
    "ALTER TABLE tips ADD COLUMN IF NOT EXISTS book           TEXT",
    "ALTER TABLE tips ADD COLUMN IF NOT EXISTS ev_pct         DOUBLE PRECISION",
    "ALTER TABLE tips ADD COLUMN IF NOT EXISTS confidence_raw DOUBLE PRECISION",
    "ALTER TABLE tips ADD COLUMN IF NOT EXISTS sent_ok        INTEGER DEFAULT 1",
    "ALTER TABLE fixtures ADD COLUMN IF NOT EXISTS last_update TIMESTAMPTZ",
    "ALTER TABLE fixtures ADD COLUMN IF NOT EXISTS status TEXT",
]

INDEX_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_tips_created           ON tips(created_ts DESC)",
    "CREATE INDEX IF NOT EXISTS idx_tips_match             ON tips(match_id)",
    "CREATE INDEX IF NOT EXISTS idx_tips_sent              ON tips(sent_ok, created_ts DESC)",
    "CREATE INDEX IF NOT EXISTS idx_snap_by_match          ON tip_snapshots(match_id, created_ts DESC)",
    "CREATE INDEX IF NOT EXISTS idx_results_updated        ON match_results(updated_ts DESC)",
    "CREATE INDEX IF NOT EXISTS idx_odds_hist_match        ON odds_history(match_id, captured_ts DESC)",
    "CREATE INDEX IF NOT EXISTS idx_fixtures_status_update ON fixtures(status, last_update DESC)",
]

def init_db() -> None:
    _init_pool()
    with db_conn() as c:
        for sql_stmt in SCHEMA_TABLES_SQL:
            c.execute(sql_stmt)
        for sql_stmt in MIGRATIONS_SQL:
            try:
                c.execute(sql_stmt)
            except Exception as e:
                log.debug("[DB] migration skipped: %s -> %s", sql_stmt, e)
        for sql_stmt in INDEX_SQL:
            try:
                c.execute(sql_stmt)
            except Exception as e:
                log.debug("[DB] index skipped: %s -> %s", sql_stmt, e)
    log.info("[DB] schema ready")

# Small settings helpers
def get_setting(key: str) -> Optional[str]:
    with db_conn() as c:
        row = c.execute("SELECT value FROM settings WHERE key=%s", (key,)).fetchone()
        return (row[0] if row else None)

def set_setting(key: str, value: str) -> None:
    with db_conn() as c:
        c.execute(
            "INSERT INTO settings(key,value) VALUES(%s,%s) "
            "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
            (key, value),
        )

def get_setting_json(key: str) -> Optional[dict]:
    try:
        raw = get_setting(key)
        return json.loads(raw) if raw else None
    except Exception as e:
        log.warning("[DB] get_setting_json failed for key=%s: %s", key, e)
        return None

# ───────────────────────────────────────────────────────────────────────────────
# API-Football odds client + EV gating
# ───────────────────────────────────────────────────────────────────────────────

APIFOOTBALL_KEY = os.getenv("APIFOOTBALL_KEY", "")
APISPORTS_BASE_URL = os.getenv("APISPORTS_BASE_URL", "https://v3.football.api-sports.io").rstrip("/")

ODDS_HEADERS = {
    "x-apisports-key": APIFOOTBALL_KEY or "",
    "Accept": "application/json",
    "User-Agent": os.getenv("HTTP_USER_AGENT", "goalsniper/1.0 (+odds)"),
}

ODDS_SOURCE = os.getenv("ODDS_SOURCE", "auto").strip().lower()  # auto|live|prematch
ODDS_AGGREGATION = os.getenv("ODDS_AGGREGATION", "median").strip().lower()  # median|best
ODDS_OUTLIER_MULT = env_float("ODDS_OUTLIER_MULT", 1.8)
ODDS_REQUIRE_N_BOOKS = env_int("ODDS_REQUIRE_N_BOOKS", 2)
ODDS_FAIR_MAX_MULT = env_float("ODDS_FAIR_MAX_MULT", 2.5)
MAX_ODDS_ALL = env_float("MAX_ODDS_ALL", 20.0)
FALLBACK_TO_PREMATCH_ON_EMPTY_LIVE = os.getenv("FALLBACK_TO_PREMATCH_ON_EMPTY_LIVE", "1").lower() not in {"0","false","no"}

BOOK_WHITELIST = {b.strip().lower() for b in os.getenv("ODDS_BOOK_WHITELIST", "").split(",") if b.strip()}
BOOK_BLACKLIST = {b.strip().lower() for b in os.getenv("ODDS_BOOK_BLACKLIST", "").split(",") if b.strip()}

MIN_ODDS_OU = env_float("MIN_ODDS_OU", 1.50)
MIN_ODDS_BTTS = env_float("MIN_ODDS_BTTS", 1.50)
MIN_ODDS_1X2 = env_float("MIN_ODDS_1X2", 1.50)
ALLOW_TIPS_WITHOUT_ODDS = os.getenv("ALLOW_TIPS_WITHOUT_ODDS", "0").lower() not in {"0","false","no"}

HTTP_CONNECT_TIMEOUT = env_float("HTTP_CONNECT_TIMEOUT", 3.0)
HTTP_READ_TIMEOUT = env_float("HTTP_READ_TIMEOUT", 10.0)

ODDS_CACHE_TTL_SEC = env_int("ODDS_CACHE_TTL_SEC", 120)
ODDS_CACHE_MAX_ITEMS = env_int("ODDS_CACHE_MAX_ITEMS", 2000)

_session_http = requests.Session()
_retry_http = Retry(
    total=3, connect=3, read=3, backoff_factor=0.6,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=frozenset(["GET"]),
    respect_retry_after_header=True, raise_on_status=False,
)
_adapter_http = HTTPAdapter(max_retries=_retry_http, pool_connections=64, pool_maxsize=128)
_session_http.mount("https://", _adapter_http)
_session_http.mount("http://", _adapter_http)

_ODDS_CACHE: TTLCache[int, dict] = TTLCache(maxsize=ODDS_CACHE_MAX_ITEMS, ttl=ODDS_CACHE_TTL_SEC)

def _book_ok(name: str) -> bool:
    n = (name or "").strip().lower()
    if BOOK_WHITELIST and n not in BOOK_WHITELIST:
        return False
    if n in BOOK_BLACKLIST:
        return False
    return True

def _fmt_line(line: float) -> str:
    return f"{line}".rstrip("0").rstrip(".")

def _market_name_normalize(s: str) -> str:
    s = (s or "").strip().lower()
    if "both teams" in s or "btts" in s:
        return "BTTS"
    if "match winner" in s or "winner" in s or "1x2" in s:
        return "1X2"
    if "over/under" in s or "total" in s or "goals" in s:
        return "OU"
    return s.upper()

def _api_get(path: str, params: dict) -> Optional[dict]:
    if not APIFOOTBALL_KEY:
        log.debug("[odds] API key missing; skip")
        return None
    url = f"{APISPORTS_BASE_URL}/{path.lstrip('/')}"
    try:
        r = _session_http.get(url, headers=ODDS_HEADERS, params=params,
                              timeout=(HTTP_CONNECT_TIMEOUT, HTTP_READ_TIMEOUT))
        if not r.ok:
            code = r.status_code
            if code in (429, 500, 502, 503, 504):
                log.debug("[odds] non-200 %s path=%s params=%s", code, path, params)
            else:
                log.warning("[odds] %s path=%s params=%s body=%s", code, path, params, r.text[:160])
            return None
        try:
            js = r.json()
        except ValueError as e:
            log.warning("[odds] invalid json from %s: %s", path, e)
            return None
        return js if isinstance(js, dict) else None
    except Exception as e:
        log.warning("[odds] request error for %s: %s", path, e)
        return None

def _aggregate_price(vals: List[Tuple[float, str]], prob_hint: Optional[float]):
    xs = [(float(o or 0.0), str(b)) for (o, b) in vals if (o or 0) > 0 and _book_ok(b)]
    if not xs:
        return None, None

    med = statistics.median([o for (o, _) in xs])
    cap_outlier = med * max(1.0, ODDS_OUTLIER_MULT)
    trimmed = [(o, b) for (o, b) in xs if o <= cap_outlier] or xs

    if prob_hint is not None and prob_hint > 0:
        fair = 1.0 / max(1e-6, float(prob_hint))
        cap_fair = fair * max(1.0, ODDS_FAIR_MAX_MULT)
        trimmed = [(o, b) for (o, b) in trimmed if o <= cap_fair] or trimmed

    if ODDS_AGGREGATION == "best":
        best = max(trimmed, key=lambda t: t[0])
        return float(best[0]), str(best[1])

    med2 = statistics.median([o for (o, _) in trimmed])
    pick = min(trimmed, key=lambda t: abs(t[0] - med2))
    distinct_books = len({b for _, b in trimmed})
    return float(pick[0]), f"{pick[1]} (median of {distinct_books})"

def _min_odds_for_market(market: str) -> float:
    if market.startswith("Over/Under"):
        return MIN_ODDS_OU
    if market == "BTTS":
        return MIN_ODDS_BTTS
    if market == "1X2":
        return MIN_ODDS_1X2
    return 1.01

def _ev(prob: float, odds: float) -> float:
    return prob * max(0.0, float(odds)) - 1.0

def fetch_odds(fid: int, prob_hints: Optional[Dict[str, float]] = None) -> dict:
    cached = _ODDS_CACHE.get(fid)
    if cached is not None:
        return cached

    js: dict = {}
    if ODDS_SOURCE in ("auto", "live"):
        tmp = _api_get("odds/live", {"fixture": fid}) or {}
        if tmp.get("response"):
            js = tmp
        elif ODDS_SOURCE == "auto" and FALLBACK_TO_PREMATCH_ON_EMPTY_LIVE:
            js = _api_get("odds", {"fixture": fid}) or {}
    if not js and ODDS_SOURCE == "prematch":
        js = _api_get("odds", {"fixture": fid}) or {}

    if not js:
        log.debug("[odds] no odds for fid=%s source=%s", fid, ODDS_SOURCE)

    by_market: Dict[str, Dict[str, List[Tuple[float, str]]]] = {}
    try:
        for r in (js.get("response") or []):
            for bk in (r.get("bookmakers") or []):
                book_name = (bk.get("name") or "Book").strip()
                if not _book_ok(book_name):
                    continue
                for mkt in (bk.get("bets") or []):
                    mname = _market_name_normalize(mkt.get("name", ""))
                    vals = (mkt.get("values") or [])
                    if mname == "BTTS":
                        for v in vals:
                            lbl = (v.get("value") or "").strip().lower()
                            odd = float(v.get("odd") or 0)
                            if "yes" in lbl:
                                by_market.setdefault("BTTS", {}).setdefault("Yes", []).append((odd, book_name))
                            elif "no" in lbl:
                                by_market.setdefault("BTTS", {}).setdefault("No", []).append((odd, book_name))
                    elif mname == "1X2":
                        for v in vals:
                            lbl = (v.get("value") or "").strip().lower()
                            odd = float(v.get("odd") or 0)
                            if lbl in ("home", "1"):
                                by_market.setdefault("1X2", {}).setdefault("Home", []).append((odd, book_name))
                            elif lbl in ("away", "2"):
                                by_market.setdefault("1X2", {}).setdefault("Away", []).append((odd, book_name))
                            # draw ignored
                    elif mname == "OU":
                        for v in vals:
                            lbl = (v.get("value") or "").strip().lower()
                            if "over" in lbl or "under" in lbl:
                                parts = lbl.split()
                                try:
                                    ln = float(parts[-1])
                                except Exception:
                                    continue
                                key = f"OU_{_fmt_line(ln)}"
                                side = "Over" if "over" in lbl else "Under"
                                odd = float(v.get("odd") or 0)
                                by_market.setdefault(key, {}).setdefault(side, []).append((odd, book_name))
    except Exception as e:
        log.debug("[odds] parse failed fid=%s: %s", fid, e)

    out: Dict[str, Dict[str, dict]] = {}
    for mkey, side_map in by_market.items():
        def _distinct_count(lst: List[Tuple[float, str]]) -> int:
            return len({b for _, b in lst})
        if not side_map:
            continue
        if not all(_distinct_count(lst) >= ODDS_REQUIRE_N_BOOKS for lst in side_map.values()):
            continue

        out[mkey] = {}
        for side, lst in side_map.items():
            hint: Optional[float] = None
            if prob_hints:
                if mkey == "BTTS":
                    if side == "Yes":
                        hint = prob_hints.get("BTTS: Yes")
                    else:
                        yes = prob_hints.get("BTTS: Yes")
                        hint = (1.0 - yes) if yes is not None else None
                elif mkey == "1X2":
                    if side == "Home":
                        hint = prob_hints.get("Home Win")
                    elif side == "Away":
                        hint = prob_hints.get("Away Win")
                elif mkey.startswith("OU_"):
                    try:
                        ln = float(mkey.split("_", 1)[1])
                        over_key = f"Over {_fmt_line(ln)} Goals"
                        if side == "Over":
                            hint = prob_hints.get(over_key)
                        else:
                            ov = prob_hints.get(over_key)
                            hint = (1.0 - ov) if ov is not None else None
                    except Exception:
                        hint = None
            ag, label = _aggregate_price(lst, hint)
            if ag is not None:
                out[mkey][side] = {"odds": float(ag), "book": label}

    _ODDS_CACHE[fid] = out
    return out

def price_gate(market: str, suggestion: str, fid: int, prob: Optional[float] = None):
    """
    Return (pass, odds, book, ev_pct).
    """
    odds_map = fetch_odds(fid)
    odds: Optional[float] = None
    book: Optional[str] = None

    if market == "BTTS":
        d = odds_map.get("BTTS", {})
        tgt = "Yes" if str(suggestion).endswith("Yes") else "No"
        if tgt in d:
            odds, book = d[tgt]["odds"], d[tgt]["book"]
    elif market == "1X2":
        d = odds_map.get("1X2", {})
        tgt = "Home" if suggestion == "Home Win" else ("Away" if suggestion == "Away Win" else None)
        if tgt and tgt in d:
            odds, book = d[tgt]["odds"], d[tgt]["book"]
    elif market.startswith("Over/Under"):
        try:
            parts = str(suggestion).split()
            ln = float(parts[1])
            d = odds_map.get(f"OU_{_fmt_line(ln)}", {})
            tgt = "Over" if str(suggestion).startswith("Over") else "Under"
            if tgt in d:
                odds, book = d[tgt]["odds"], d[tgt]["book"]
        except Exception:
            pass

    if odds is None:
        return (ALLOW_TIPS_WITHOUT_ODDS, odds, book, None)

    if not (_min_odds_for_market(market) <= float(odds) <= MAX_ODDS_ALL):
        return (False, odds, book, None)

    ev_pct = None
    if prob is not None:
        edge = _ev(prob, float(odds))
        ev_pct = round(edge * 100.0, 1)
        if edge < 0:
            return (False, odds, book, ev_pct)

    return (True, float(odds), book, ev_pct)

# ───────────────────────────────────────────────────────────────────────────────
# Predictor hook (lightweight; falls back to odds if empty)
# ───────────────────────────────────────────────────────────────────────────────

MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")
MODEL_KIND = os.getenv("MODEL_KIND", "").strip().lower()  # "", "pickle", "json"
PREDICTOR_STRICT = os.getenv("PREDICTOR_STRICT", "0").lower() not in {"0", "false", "no", ""}
PREDICTION_CACHE_TTL = env_int("PREDICTION_CACHE_TTL", 0)

_model: Any = None
_model_loaded = False
_model_lock = threading.RLock()
_pred_cache: Dict[int, tuple[float, Dict[str, float]]] = {}
_pred_cache_lock = threading.RLock()

def _now() -> float:
    return time.time()

def _load_model() -> Optional[Any]:
    global _model, _model_loaded
    if _model_loaded:
        return _model
    with _model_lock:
        if _model_loaded:
            return _model
        try:
            if not os.path.exists(MODEL_PATH):
                _model = None
                _model_loaded = True
                log.info("[predictor] MODEL_PATH not found: %s (fallback to odds de-vig)", MODEL_PATH)
                return _model
            kind = MODEL_KIND or ("json" if MODEL_PATH.endswith(".json") else "pickle")
            if kind == "json":
                with open(MODEL_PATH, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                if not isinstance(obj, dict):
                    obj = {}
                _model = {"kind": "json", "data": obj}
                log.info("[predictor] loaded JSON: %s", MODEL_PATH)
            else:
                import pickle
                with open(MODEL_PATH, "rb") as f:
                    _model = pickle.load(f)
                log.info("[predictor] loaded PICKLE: %s (%s)", MODEL_PATH, type(_model).__name__)
            _model_loaded = True
        except Exception as e:
            _model = None
            _model_loaded = True
            log.warning("[predictor] load failed: %s", e)
        return _model

def warm_model() -> bool:
    return _load_model() is not None

def _predict_with_json_model(model_obj: dict, fid: int) -> Dict[str, float]:
    data = model_obj.get("data", {})
    if isinstance(data, dict):
        return {str(k): float(v) for k, v in data.items() if v is not None}
    return {}

def _predict_with_pickle(model: Any, fid: int) -> Dict[str, float]:
    try:
        # plug your real model here (features_for(fid) etc.). For now, fallback.
        return {}
    except Exception as e:
        log.warning("[predictor] prediction failed fid=%s: %s", fid, e)
        return {}

def _get_cached_prediction(fid: int) -> Optional[Dict[str, float]]:
    if PREDICTION_CACHE_TTL <= 0:
        return None
    with _pred_cache_lock:
        entry = _pred_cache.get(fid)
        if not entry: return None
        ts, val = entry
        if (_now() - ts) <= PREDICTION_CACHE_TTL:
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
        _pred_cache[fid] = (_now(), val)

def predict_for_fixture(fid: int) -> Dict[str, float]:
    cached = _get_cached_prediction(fid)
    if cached is not None:
        return cached
    model = _load_model()
    if model is None:
        if PREDICTOR_STRICT:
            log.debug("[predictor] strict mode: model missing (fid=%s)", fid)
        return {}
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
# Results provider (final scores / BTTS) — /fixtures endpoint
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
            resp = _session_http.get(url, headers=RESULTS_HEADERS, params={"id": fid}, timeout=HTTP_RESULTS_TIMEOUT)
            if not resp.ok:
                log.debug("[results] fixture %s fetch failed (%s)", fid, resp.status_code)
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
        except Exception as e:
            log.debug("[results] fixture %s error: %s", fid, e)
    return out

def update_match_results(rows: Iterable[Tuple[int,int,int,int]]) -> int:
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
# Scanning logic (adaptive): production in-play + prematch + MOTD + retry + backfill
# ───────────────────────────────────────────────────────────────────────────────

CONF_MIN = env_float("CONF_MIN", 0.75)          # min model prob to consider (0..1)
EV_MIN = env_float("EV_MIN", 0.00)              # min EV gate (>=0)
MOTD_CONF_MIN = env_float("MOTD_CONF_MIN", 0.78)
MOTD_EV_MIN = env_float("MOTD_EV_MIN", 0.05)
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

def _fmt_tip_message(match, market, suggestion, conf, odds, book, ev_pct):
    ko_dt = match.get("kickoff")
    if ko_dt and ko_dt.tzinfo is None:
        ko_dt = ko_dt.replace(tzinfo=TZ_UTC)
    kickoff = (ko_dt or _now_dt()).astimezone(BERLIN_TZ).strftime("%Y-%m-%d %H:%M")
    home, away = match.get("home"), match.get("away")
    league = match.get("league")

    odds_str = "-" if odds is None else f"{float(odds):.2f}"
    book_str = book or "best"

    msg = (
        f"⚽️ *{league}*\n"
        f"{home} vs {away}\n"
        f"🕒 Kickoff: {kickoff} Berlin\n"
        f"🎯 *Tip:* {suggestion}\n"
        f"📊 *Confidence:* {conf*100:.1f}%\n"
        f"💰 *Odds:* {odds_str} @ {book_str}"
    )
    if ev_pct is not None:
        msg += f" • *EV:* {ev_pct:+.1f}%"
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
    # 1) try model
    try:
        probs = predict_for_fixture(fid)
        if isinstance(probs, dict) and probs:
            return {str(k): float(v) for k, v in probs.items() if v is not None}
    except Exception as e:
        log.warning("[model] predict failed fid=%s: %s", fid, e)

    # 2) fallback from odds-implied (two-way de-vig)
    hints: Dict[str, float] = {}

    # BTTS Yes/No
    d = odds_map.get("BTTS", {})
    if d:
        p_yes = _implied_prob(d.get("Yes", {}).get("odds")) if "Yes" in d else 0.0
        p_no  = _implied_prob(d.get("No",  {}).get("odds")) if "No"  in d else 0.0
        p_yes, p_no = _normalize_pair(p_yes, p_no)
        hints["BTTS: Yes"] = p_yes
        hints["BTTS: No"]  = p_no

    # 1X2 Home/Away (ignore Draw)
    d = odds_map.get("1X2", {})
    if d:
        p_h = _implied_prob(d.get("Home", {}).get("odds")) if "Home" in d else 0.0
        p_a = _implied_prob(d.get("Away", {}).get("odds")) if "Away" in d else 0.0
        p_h, p_a = _normalize_pair(p_h, p_a)
        hints["Home Win"] = p_h
        hints["Away Win"] = p_a

    # OU lines
    for k, sides in odds_map.items():
        if not k.startswith("OU_"):
            continue
        line = k.split("_", 1)[1]
        p_over = _implied_prob(sides.get("Over", {}).get("odds")) if "Over" in sides else 0.0
        p_under= _implied_prob(sides.get("Under",{}).get("odds")) if "Under" in sides else 0.0
        p_over, p_under = _normalize_pair(p_over, p_under)
        hints[f"Over {line} Goals"]  = p_over
        hints[f"Under {line} Goals"] = p_under

    return hints

def _best_candidate_for_fixture(fid: int) -> Optional[Tuple[str, str, float, Optional[float], Optional[str], Optional[float]]]:
    """
    Returns: (market, suggestion, prob, odds, book, ev_pct) if passes gates; else None.
    """
    odds_map = fetch_odds(fid)
    if not odds_map:
        return None

    prob_hints = _prob_hints_from_model_or_odds(fid, odds_map)
    candidates: List[Tuple[str, str, float]] = []

    # BTTS
    if "BTTS: Yes" in prob_hints:
        candidates.append(("BTTS", "BTTS: Yes", prob_hints["BTTS: Yes"]))
    if "BTTS: No" in prob_hints:
        candidates.append(("BTTS", "BTTS: No", prob_hints["BTTS: No"]))

    # 1X2 (H/A only)
    if "Home Win" in prob_hints:
        candidates.append(("1X2", "Home Win", prob_hints["Home Win"]))
    if "Away Win" in prob_hints:
        candidates.append(("1X2", "Away Win", prob_hints["Away Win"]))

    # OU lines
    for k in sorted([kk for kk in odds_map.keys() if str(kk).startswith("OU_")]):
        ln = k.split("_", 1)[1]
        over_key, under_key = f"Over {ln} Goals", f"Under {ln} Goals"
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
    """
    Pull candidates per live/soon fixture, save best passing pick per fixture, notify up to N.
    Returns: (saved_count, live_seen)
    """
    saved, live_seen = 0, 0
    to_insert: List[Tuple] = []
    to_notify: List[Tuple[str, dict]] = []

    try:
        with db_conn() as c:
            rows = c.execute(
                """
                SELECT fixture_id, league_name, home, away, kickoff, last_update, status
                FROM fixtures
                WHERE
                  status IN ('1H','HT','2H','ET','P','LIVE')
                  OR (status IN ('NS','TBD') AND kickoff >= now() AND kickoff <= now() + interval '60 minutes')
                """
            ).fetchall()

        for fid, league, home, away, kickoff, last_update, status in rows:
            live_seen += 1
            if _is_feed_stale(last_update):
                continue
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

            match = {"league": league, "home": home, "away": away, "kickoff": kickoff}
            msg = _fmt_tip_message(match, market, suggestion, prob, odds, book, ev_pct)
            to_notify.append((msg, match))

        if to_insert:
            with db_conn() as c:
                execute_values(
                    c.cur,
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

        sent = 0
        for msg, _ in to_notify[:MAX_TELEGRAM_PER_SCAN]:
            time.sleep(random.uniform(0.05, 0.2))
            send_telegram(msg)
            sent += 1

        if sent and len(to_notify) > sent:
            log.info("[scan] sent %d of %d tips (cap=%d)", sent, len(to_notify), MAX_TELEGRAM_PER_SCAN)

    except Exception as e:
        log.exception("[scan] failed: %s", e)

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
                execute_values(
                    c.cur,
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
    """Summarize yesterday’s accuracy and ROI (proper grading for BTTS/OU/1X2)."""
    today = _now_dt().astimezone(BERLIN_TZ).date()
    yesterday = today - datetime.timedelta(days=1)
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

            is_win = False
            if s.startswith("Over") or s.startswith("Under"):
                try:
                    line = float(s.split()[1])
                except Exception:
                    line = 2.5
                is_win = (total > line) if s.startswith("Over") else (total < line)
            elif s == "BTTS: Yes":
                is_win = bool(btts_yes)
            elif s == "BTTS: No":
                is_win = not bool(btts_yes)
            elif s == "Home Win":
                is_win = gh > ga
            elif s == "Away Win":
                is_win = ga > gh

            if is_win:
                wins += 1
            try:
                o = float(odds)
                pnl += (o - 1.0) if is_win else -1.0
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
        msg = "🌟 *Match of the Day*\n" + _fmt_tip_message(match, market, suggestion, prob, odds, book, ev_pct)
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
                execute_values(
                    c2.cur,
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
# Flask app, logging, scheduler, routes
# ───────────────────────────────────────────────────────────────────────────────

# Import training/tuning helpers from the separate module
from train_models import train_models, auto_tune_thresholds, load_thresholds_from_settings  # noqa: E402

# ───────── Logging (global request_id injection) ─────────
def _record_factory_with_request_id():
    base_factory = logging.getLogRecordFactory()
    def factory(*args, **kwargs):
        record = base_factory(*args, **kwargs)
        try:
            # Attach request_id safely for ALL loggers/threads
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

logging.setLogRecordFactory(_record_factory_with_request_id())
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(request_id)s - %(message)s"
)
log = logging.getLogger("goalsniper")

app = Flask(__name__)

def get_logger() -> logging.Logger:
    return log

# ───────── Small DB helpers ─────────
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

# ───────── Threshold application (runtime) ─────────
def _apply_tuned_thresholds():
    """
    Load CONF_MIN / EV_MIN / MOTD_* from DB settings and apply to module globals.
    """
    try:
        th = load_thresholds_from_settings()
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

# ───────── Scheduler ─────────
RUN_SCHEDULER = os.getenv("RUN_SCHEDULER", "1").lower() in {"1","true","yes"}
SCHEDULER_LEADER = os.getenv("SCHEDULER_LEADER", "1").lower() in {"1","true","yes"}
SCAN_INTERVAL_SEC = env_int("SCAN_INTERVAL_SEC", 300)
BACKFILL_EVERY_MIN = env_int("BACKFILL_EVERY_MIN", 15)
TRAIN_ENABLE = os.getenv("TRAIN_ENABLE", "1").lower() in {"1","true","yes"}
TRAIN_HOUR_UTC = env_int("TRAIN_HOUR_UTC", 2)
TRAIN_MINUTE_UTC = env_int("TRAIN_MINUTE_UTC", 12)
AUTO_TUNE_ENABLE = os.getenv("AUTO_TUNE_ENABLE", "0").lower() in {"1","true","yes"}
DAILY_ACCURACY_DIGEST_ENABLE = os.getenv("DAILY_ACCURACY_DIGEST_ENABLE", "1").lower() in {"1","true","yes"}
DAILY_ACCURACY_HOUR = env_int("DAILY_ACCURACY_HOUR", 8)
DAILY_ACCURACY_MINUTE = env_int("DAILY_ACCURACY_MINUTE", 0)
MOTD_ENABLE = os.getenv("MOTD_ENABLE", "1").lower() in {"1","true","yes"}
MOTD_HOUR = env_int("MOTD_HOUR", 10)
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
    Starts BackgroundScheduler exactly once per process, and only if:
      - RUN_SCHEDULER = true
      - SCHEDULER_LEADER = true (use this to ensure only one process runs the scheduler)
    """
    global _scheduler_started, _scheduler_ref
    if _scheduler_started or not RUN_SCHEDULER or not SCHEDULER_LEADER:
        if not SCHEDULER_LEADER:
            log.info("[SCHED] disabled in this process (SCHEDULER_LEADER=false)")
        return

    # Avoid double-start under Flask dev reloader
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

        sched.add_job(
            _scan_job, "interval",
            seconds=SCAN_INTERVAL_SEC, id="scan", max_instances=1, coalesce=True, misfire_grace_time=60
        )

        sched.add_job(
            lambda: _run_with_pg_lock(1002, backfill_results_for_open_matches, 400),
            "interval", minutes=BACKFILL_EVERY_MIN, id="backfill", max_instances=1, coalesce=True, misfire_grace_time=120
        )

        if DAILY_ACCURACY_DIGEST_ENABLE:
            sched.add_job(
                lambda: _run_with_pg_lock(1003, daily_accuracy_digest),
                CronTrigger(hour=DAILY_ACCURACY_HOUR, minute=DAILY_ACCURACY_MINUTE, timezone=BERLIN_TZ),
                id="digest", max_instances=1, coalesce=True, misfire_grace_time=3600
            )

        if MOTD_ENABLE:
            sched.add_job(
                lambda: _run_with_pg_lock(1004, send_match_of_the_day),
                CronTrigger(hour=MOTD_HOUR, minute=MOTD_MINUTE, timezone=BERLIN_TZ),
                id="motd", max_instances=1, coalesce=True, misfire_grace_time=3600
            )

        if TRAIN_ENABLE:
            sched.add_job(
                lambda: _run_with_pg_lock(1005, train_models),
                CronTrigger(hour=TRAIN_HOUR_UTC, minute=TRAIN_MINUTE_UTC, timezone=TZ_UTC),
                id="train", max_instances=1, coalesce=True, misfire_grace_time=3600
            )

        if AUTO_TUNE_ENABLE:
            sched.add_job(
                lambda: _run_with_pg_lock(1006, auto_tune_thresholds, 14),
                CronTrigger(hour=4, minute=7, timezone=TZ_UTC),
                id="auto_tune", max_instances=1, coalesce=True, misfire_grace_time=3600
            )

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

def _handle_sigterm(*_):
    log.info("[BOOT] SIGTERM received — shutting down gracefully")
    _stop_scheduler()
    sys.exit(0)

def _handle_sigint(*_):
    log.info("[BOOT] SIGINT received — shutting down gracefully")
    _stop_scheduler()
    sys.exit(0)

# ───────── Routes ─────────
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

    # Optional deep count in a NEW context (don't reuse a closed one)
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
    lg = get_logger()
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
        lg.exception("/stats failed: %s", e)
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
    res = train_models()
    return jsonify({"ok": True, "result": res})

@app.route("/admin/digest", methods=["POST", "GET"])
def http_digest():
    _require_admin()
    msg = daily_accuracy_digest()
    return jsonify({"ok": True, "sent": bool(msg)})

@app.route("/admin/auto-tune", methods=["POST", "GET"])
def http_auto_tune():
    _require_admin()
    tuned = auto_tune_thresholds(14)
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

# ───────── Boot ─────────
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

# Ensure clean scheduler shutdown on signals
signal.signal(signal.SIGTERM, _handle_sigterm)
signal.signal(signal.SIGINT, _handle_sigint)

_on_boot()

if __name__ == "__main__":
    # In production, prefer gunicorn. For local/dev, Flask's server is fine.
    app.run(host=os.getenv("HOST","0.0.0.0"), port=int(os.getenv("PORT","8080")))
