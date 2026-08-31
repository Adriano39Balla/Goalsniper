"""
goalsniper — in-play + prematch football tipping service.

Headline corrections, in the order they matter:

  T0.1  Feature building lives in feature_spec.py, imported by BOTH this file
        and train_models.py. Train/serve drift is structurally impossible.
  T0.3  1X2 normalises over Home+Draw+Away, producing a real P(Home) that
        matches how the bet is priced and graded. The previous Home/(Home+Away)
        normalisation was a Draw-No-Bet probability priced against 1X2 odds,
        inflating every 1X2 confidence by ~1.30-1.35x.
  T0.4  In-play prices come from /odds/live. /odds is prematch-only.
  T0.5  HARVEST rows live in tip_snapshots, not `tips`.
  T0.6  ALLOW_TIPS_WITHOUT_ODDS defaults to 0.
  T1.2  Models carry their own StandardScaler (mean/scale) in the blob.
  NEW   De-vigged fair prices, a fair-edge gate, a model-sanity edge cap,
        closing-line-value capture, fractional-Kelly staking, and a
        `predictions` log recording EVERY candidate so calibration can be
        measured without selection bias.

THIS REVISION FIXES THREE DEFECTS FOUND IN THE DOUBLE CHANCE / DRAW NO BET AND
DASHBOARD ADDITIONS:

  1. Double Chance de-vig was normalising to 1.0. DC selections are not
     mutually exclusive — 1X, X2 and 12 each cover two of three outcomes, so
     their true probabilities sum to 2.0. Every DC fair price came out at
     exactly half its true value, which read as a ~36 percentage-point edge and
     tripped MAX_MODEL_EDGE_BPS on every DC candidate. Now uses
     feature_spec.MARKET_PROBABILITY_TOTAL.
  2. Double Chance and Draw No Bet had no trained threshold, so
     _get_market_threshold() fell through to CONF_THRESHOLD (70) while the 1X2
     heads they derive from sat suppressed at 85. train_models.py now trains and
     holdout-verifies both, and this file refuses to serve a derived market
     whose threshold has never been written.
  3. The dashboard generated a random SECRET_KEY when the env var was unset.
     Under multiple workers each worker signs cookies with a different key, so
     logins fail at random. The dashboard now refuses to start without an
     explicit SECRET_KEY, and the login endpoint is rate-limited.

Requires DATABASE_URL and API_KEY.
"""
from __future__ import annotations

import hmac
import json
import logging
import math
import os
import random
import signal
import sys
import threading
import time
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from html import escape
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import psycopg2
import requests
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from flask import (
    Flask, abort, jsonify, redirect, render_template, request, url_for,
)
# Aliased: this module already has a module-level `session` (a requests.Session
# for outbound HTTP, defined below) that would otherwise shadow Flask's session
# proxy the moment that line executes.
from flask import session as flask_session
from psycopg2.pool import ThreadedConnectionPool
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from feature_spec import (
    ELO_DEFAULT, ELO_HOME_ADV, ELO_K,
    DEFAULT_LEAGUE_RATES, MARKET_PROBABILITY_TOTAL, RAW_INPLAY_KEYS,
    assemble_prematch_features, build_inplay_features, derive_dc_dnb,
    devig, ev as _ev, fixture_ts as _fixture_ts, kelly_fraction,
    enforce_ou_monotonicity,
)

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
log = logging.getLogger("goalsniper")
app = Flask(__name__)


def _env_flag(name: str, default: str) -> bool:
    return os.getenv(name, default) not in ("0", "false", "False", "no", "NO")


# ───────── Dashboard session security ─────────
# FIX: the previous code fell back to os.urandom() when SECRET_KEY was unset.
# That is not merely "everyone gets logged out on restart" — with more than one
# gunicorn worker each worker generates its OWN key, so a cookie signed by
# worker A is rejected by worker B and login fails at random. There is no safe
# automatic fallback for a multi-process signing key, so the dashboard is
# disabled instead of being silently broken.
SECRET_KEY = os.getenv("SECRET_KEY")
DASHBOARD_ENABLED = bool(SECRET_KEY)
app.secret_key = SECRET_KEY or os.urandom(32).hex()
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=_env_flag("SESSION_COOKIE_SECURE", "1"),
    PERMANENT_SESSION_LIFETIME=timedelta(days=int(os.getenv("DASHBOARD_SESSION_DAYS", "7"))),
)
if not DASHBOARD_ENABLED:
    log.warning("[DASHBOARD] SECRET_KEY is not set — /dashboard is DISABLED. Generate one with "
                "`python -c \"import secrets; print(secrets.token_hex(32))\"` and set it as an "
                "env var. Everything else runs normally.")


# ───────── Core env ─────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
API_KEY = os.getenv("API_KEY")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")
WEBHOOK_SECRET = os.getenv("TELEGRAM_WEBHOOK_SECRET")
RUN_SCHEDULER = _env_flag("RUN_SCHEDULER", "1")

CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "70"))
MAX_TIPS_PER_SCAN = int(os.getenv("MAX_TIPS_PER_SCAN", "25"))
DUP_COOLDOWN_MIN = int(os.getenv("DUP_COOLDOWN_MIN", "20"))
TIP_MIN_MINUTE = int(os.getenv("TIP_MIN_MINUTE", "8"))
SCAN_INTERVAL_SEC = int(os.getenv("SCAN_INTERVAL_SEC", "300"))

PREDICTIONS_PER_MATCH = int(os.getenv("PREDICTIONS_PER_MATCH", "1"))
CORRELATED_EXTRA_EV_BPS = int(os.getenv("CORRELATED_EXTRA_EV_BPS", "400"))

HARVEST_MODE = _env_flag("HARVEST_MODE", "1")
TRAIN_ENABLE = _env_flag("TRAIN_ENABLE", "1")
TRAIN_HOUR_UTC = int(os.getenv("TRAIN_HOUR_UTC", "2"))
TRAIN_MINUTE_UTC = int(os.getenv("TRAIN_MINUTE_UTC", "12"))
TRAIN_MIN_MINUTE = int(os.getenv("TRAIN_MIN_MINUTE", "15"))
HARVEST_EVERY_MINUTES = int(os.getenv("HARVEST_EVERY_MINUTES", "3"))

BACKFILL_EVERY_MIN = int(os.getenv("BACKFILL_EVERY_MIN", "15"))
BACKFILL_DAYS = int(os.getenv("BACKFILL_DAYS", "14"))
DAILY_ACCURACY_DIGEST_ENABLE = _env_flag("DAILY_ACCURACY_DIGEST_ENABLE", "1")
DAILY_ACCURACY_HOUR = int(os.getenv("DAILY_ACCURACY_HOUR", "3"))
DAILY_ACCURACY_MINUTE = int(os.getenv("DAILY_ACCURACY_MINUTE", "6"))

PREMATCH_SCAN_ENABLE = _env_flag("PREMATCH_SCAN_ENABLE", "1")
PREMATCH_SCAN_INTERVAL_MIN = int(os.getenv("PREMATCH_SCAN_INTERVAL_MIN", "180"))
PREMATCH_SNAPSHOT_TTL_SEC = int(os.getenv("PREMATCH_SNAPSHOT_TTL_SEC", "21600"))
PREMATCH_DEDUP_ENABLE = _env_flag("PREMATCH_DEDUP_ENABLE", "1")
MAX_PREMATCH_TIPS_PER_SCAN = int(os.getenv("MAX_PREMATCH_TIPS_PER_SCAN", "40"))


def _int_list(env_val: str) -> List[int]:
    out = []
    for x in (env_val or "").split(","):
        x = x.strip()
        if x.lstrip("-").isdigit():
            out.append(int(x))
    return out


PREMATCH_LEAGUE_IDS = _int_list(os.getenv("PREMATCH_LEAGUE_IDS", ""))
MOTD_LEAGUE_IDS = _int_list(os.getenv("MOTD_LEAGUE_IDS", ""))

AUTO_TUNE_ENABLE = _env_flag("AUTO_TUNE_ENABLE", "0")
TARGET_PRECISION = float(os.getenv("TARGET_PRECISION", "0.60"))
THRESH_MIN_PREDICTIONS = int(os.getenv("THRESH_MIN_PREDICTIONS", "100"))
MIN_THRESH = float(os.getenv("MIN_THRESH", "55"))
MAX_THRESH = float(os.getenv("MAX_THRESH", "85"))

MOTD_PREDICT = _env_flag("MOTD_PREDICT", "1")
MOTD_HOUR = int(os.getenv("MOTD_HOUR", "19"))
MOTD_MINUTE = int(os.getenv("MOTD_MINUTE", "15"))
MOTD_CONF_MIN = float(os.getenv("MOTD_CONF_MIN", "70"))


def _parse_lines(env_val: str, default: List[float]) -> List[float]:
    out = []
    for t in (env_val or "").split(","):
        t = t.strip()
        if not t:
            continue
        try:
            out.append(float(t))
        except Exception:
            pass
    return out or default


OU_LINES = [ln for ln in _parse_lines(os.getenv("OU_LINES", "2.5,3.5"), [2.5, 3.5]) if abs(ln - 1.5) > 1e-6]

# ───────── Odds / EV controls ─────────
MIN_ODDS_OU = float(os.getenv("MIN_ODDS_OU", "1.30"))
MIN_ODDS_BTTS = float(os.getenv("MIN_ODDS_BTTS", "1.30"))
MIN_ODDS_1X2 = float(os.getenv("MIN_ODDS_1X2", "1.30"))
MIN_ODDS_DC = float(os.getenv("MIN_ODDS_DC", "1.15"))
MIN_ODDS_DNB = float(os.getenv("MIN_ODDS_DNB", "1.20"))
MAX_ODDS_ALL = float(os.getenv("MAX_ODDS_ALL", "20.0"))

EDGE_MIN_BPS = int(os.getenv("EDGE_MIN_BPS", "300"))
FAIR_EDGE_MIN_BPS = int(os.getenv("FAIR_EDGE_MIN_BPS", "200"))
# Sanity cap, tightened from 1500. Live Double Chance tips went out claiming
# 10.2 and 12.9 percentage-point disagreements with a de-vigged consensus. In a
# market that liquid a double-digit edge is model error every time, and the old
# cap was loose enough to wave both through.
MAX_MODEL_EDGE_BPS = int(os.getenv("MAX_MODEL_EDGE_BPS", "800"))
REQUIRE_FAIR_PRICE = _env_flag("REQUIRE_FAIR_PRICE", "1")
# A "consensus" fair price built from one bookmaker is not a consensus. Those
# live tips came from Danish 2. Division and Swedish Division 2 — thin markets
# where a single stale quote can both set the best price and define "fair",
# manufacturing an overlay that does not exist (best 1.53 against a "fair" 1.41).
# Best price is still taken across every book; this governs whether the FAIR side
# is trustworthy enough to bet against.
MIN_BOOKS_FOR_FAIR = int(os.getenv("MIN_BOOKS_FOR_FAIR", "3"))
ODDS_BOOKMAKER_ID = os.getenv("ODDS_BOOKMAKER_ID")
ALLOW_TIPS_WITHOUT_ODDS = _env_flag("ALLOW_TIPS_WITHOUT_ODDS", "0")

BANKROLL_UNITS = float(os.getenv("BANKROLL_UNITS", "100"))
KELLY_FRACTION = float(os.getenv("KELLY_FRACTION", "0.25"))
MAX_STAKE_PCT = float(os.getenv("MAX_STAKE_PCT", "2.0"))

CLV_ENABLE = _env_flag("CLV_ENABLE", "1")
CLV_CAPTURE_EVERY_MIN = int(os.getenv("CLV_CAPTURE_EVERY_MIN", "5"))
CLV_MAX_AGE_MIN = int(os.getenv("CLV_MAX_AGE_MIN", "90"))

PREDICTION_LOG_ENABLE = _env_flag("PREDICTION_LOG_ENABLE", "1")
PREDICTION_LOG_MIN_PROB = float(os.getenv("PREDICTION_LOG_MIN_PROB", "0.35"))

# Markets with no model of their own — they are algebraic transforms of the 1X2
# heads. They must never fall back to a default threshold: see
# _get_market_threshold().
DERIVED_MARKETS = {"Double Chance", "Draw No Bet"}

ALLOWED_SUGGESTIONS = {
    "BTTS: Yes", "BTTS: No", "Home Win", "Away Win",
    "Double Chance: 1X", "Double Chance: X2", "Double Chance: 12",
    "Draw No Bet: Home", "Draw No Bet: Away",
}


def _fmt_line(line: float) -> str:
    return f"{line}".rstrip("0").rstrip(".")


for _ln in OU_LINES:
    _s = _fmt_line(_ln)
    ALLOWED_SUGGESTIONS.add(f"Over {_s} Goals")
    ALLOWED_SUGGESTIONS.add(f"Under {_s} Goals")

_GOALS_UP = {"BTTS: Yes"} | {f"Over {_fmt_line(l)} Goals" for l in OU_LINES}
_GOALS_DOWN = {"BTTS: No"} | {f"Under {_fmt_line(l)} Goals" for l in OU_LINES}
# Double Chance: 12 excludes the draw either way, so it correlates with both sides.
_HOME_SIDE = {"Home Win", "Double Chance: 1X", "Double Chance: 12", "Draw No Bet: Home"}
_AWAY_SIDE = {"Away Win", "Double Chance: X2", "Double Chance: 12", "Draw No Bet: Away"}
_CORRELATION_FAMILIES = (_GOALS_UP, _GOALS_DOWN, _HOME_SIDE, _AWAY_SIDE)

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise SystemExit("DATABASE_URL is required")

BASE_URL = "https://v3.football.api-sports.io"
FOOTBALL_API_URL = f"{BASE_URL}/fixtures"
ODDS_PREMATCH_URL = f"{BASE_URL}/odds"
ODDS_LIVE_URL = f"{BASE_URL}/odds/live"
HEADERS = {"x-apisports-key": API_KEY, "Accept": "application/json"}
INPLAY_STATUSES = {"1H", "HT", "2H", "ET", "BT", "P"}
FINAL_STATUSES = {"FT", "AET", "PEN"}

session = requests.Session()
HTTP_POOL_MAXSIZE = int(os.getenv("HTTP_POOL_MAXSIZE", "30"))
session.mount("https://", HTTPAdapter(
    max_retries=Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504],
                      respect_retry_after_header=True),
    pool_connections=HTTP_POOL_MAXSIZE, pool_maxsize=HTTP_POOL_MAXSIZE))

TZ_UTC, BERLIN_TZ = ZoneInfo("UTC"), ZoneInfo("Europe/Berlin")
EPS = 1e-12


def _safe_compare(a: Any, b: Any) -> bool:
    """
    Constant-time comparison that cannot raise.

    hmac.compare_digest() raises TypeError when handed a str containing
    non-ASCII characters, which would turn a mistyped key into a 500 rather than
    a clean 401. Comparing the UTF-8 bytes keeps it constant-time and total.
    """
    try:
        return hmac.compare_digest(str(a).encode("utf-8"), str(b).encode("utf-8"))
    except Exception:
        return False


# ───────── Thread-safe bounded TTL cache ─────────
_MISS = object()


class _TTLCache:
    """
    Thread-safe, size-bounded TTL cache. get() returns a caller-supplied default
    on miss, so a cached None is distinguishable from an absent key.
    """

    def __init__(self, ttl: float, maxsize: int = 5000):
        self.ttl = ttl
        self.maxsize = max(1, int(maxsize))
        self._data: "OrderedDict[Any, Tuple[float, Any]]" = OrderedDict()
        self._lock = threading.RLock()

    def get(self, k, default=None):
        with self._lock:
            v = self._data.get(k, _MISS)
            if v is _MISS:
                return default
            ts, val = v
            if time.time() - ts > self.ttl:
                self._data.pop(k, None)
                return default
            self._data.move_to_end(k)
            return val

    def set(self, k, v):
        with self._lock:
            self._data[k] = (time.time(), v)
            self._data.move_to_end(k)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def invalidate(self, k=None):
        with self._lock:
            if k is None:
                self._data.clear()
            else:
                self._data.pop(k, None)


TEAM_FORM_TTL = int(os.getenv("TEAM_FORM_CACHE_TTL_SEC", "1800"))
STATS_CACHE = _TTLCache(ttl=90, maxsize=int(os.getenv("STATS_CACHE_MAXSIZE", "1000")))
EVENTS_CACHE = _TTLCache(ttl=90, maxsize=int(os.getenv("EVENTS_CACHE_MAXSIZE", "1000")))
ODDS_CACHE = _TTLCache(ttl=int(os.getenv("ODDS_CACHE_TTL_SEC", "45")),
                       maxsize=int(os.getenv("ODDS_CACHE_MAXSIZE", "2000")))
TEAM_FORM_CACHE = _TTLCache(ttl=TEAM_FORM_TTL, maxsize=int(os.getenv("TEAM_FORM_CACHE_MAXSIZE", "8000")))

SETTINGS_TTL = int(os.getenv("SETTINGS_TTL_SEC", "60"))
MODELS_TTL = int(os.getenv("MODELS_CACHE_TTL_SEC", "120"))
_SETTINGS_CACHE = _TTLCache(SETTINGS_TTL)
_MODELS_CACHE = _TTLCache(MODELS_TTL)
LEAGUE_RATE_TTL = int(os.getenv("LEAGUE_RATE_TTL_SEC", "21600"))
LEAGUE_RATE_MIN_N = int(os.getenv("LEAGUE_RATE_MIN_N", "20"))
_LEAGUE_RATE_CACHE = _TTLCache(LEAGUE_RATE_TTL)

try:
    from train_models import train_models
except Exception as e:  # pragma: no cover
    _IMPORT_ERR = repr(e)

    def train_models(*args, **kwargs):  # type: ignore
        log.warning("train_models not available: %s", _IMPORT_ERR)
        return {"ok": False, "reason": f"train_models import failed: {_IMPORT_ERR}"}


# ───────── DB pool ─────────
POOL: Optional[ThreadedConnectionPool] = None


class PooledConn:
    def __init__(self, pool):
        self.pool = pool
        self.conn = None
        self.cur = None

    def __enter__(self):
        last_err = None
        for attempt in range(5):
            try:
                self.conn = self.pool.getconn()
                self.conn.autocommit = True
                self.cur = self.conn.cursor()
                return self
            except psycopg2.pool.PoolError as e:
                last_err = e
                time.sleep(0.2 * (attempt + 1))
        raise last_err

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self.cur:
                self.cur.close()
        except Exception:
            pass
        finally:
            if self.conn is not None:
                broken = exc_type is not None and issubclass(
                    exc_type, (psycopg2.OperationalError, psycopg2.InterfaceError))
                try:
                    self.pool.putconn(self.conn, close=broken)
                except Exception:
                    try:
                        self.conn.close()
                    except Exception:
                        pass

    def execute(self, sql: str, params=()):
        self.cur.execute(sql, params or ())
        return self.cur

    def executemany(self, sql: str, seq):
        if not seq:
            return self.cur
        self.cur.executemany(sql, seq)
        return self.cur


def _init_pool():
    global POOL
    dsn = DATABASE_URL + (("&" if "?" in DATABASE_URL else "?") + "sslmode=require"
                          if "sslmode=" not in DATABASE_URL else "")
    POOL = ThreadedConnectionPool(minconn=1, maxconn=int(os.getenv("DB_POOL_MAX", "20")), dsn=dsn)


def db_conn():
    if not POOL:
        _init_pool()
    return PooledConn(POOL)  # type: ignore


# ───────── Settings ─────────
def get_setting(key: str) -> Optional[str]:
    with db_conn() as c:
        r = c.execute("SELECT value FROM settings WHERE key=%s", (key,)).fetchone()
        return r[0] if r else None


def set_setting(key: str, value: str) -> None:
    with db_conn() as c:
        c.execute("INSERT INTO settings(key,value) VALUES(%s,%s) "
                  "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value", (key, value))


def get_setting_cached(key: str) -> Optional[str]:
    v = _SETTINGS_CACHE.get(key, _MISS)
    if v is _MISS:
        v = get_setting(key)
        _SETTINGS_CACHE.set(key, v)
    return v


def invalidate_model_caches_for_key(key: str):
    if key.lower().startswith(("model", "pre_")):
        _MODELS_CACHE.invalidate()


# ───────── Schema ─────────
def init_db():
    with db_conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)""")
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
        c.execute("""CREATE TABLE IF NOT EXISTS match_results (
            match_id BIGINT PRIMARY KEY, final_goals_h INTEGER, final_goals_a INTEGER,
            btts_yes INTEGER, updated_ts BIGINT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS team_ratings (
            team_id BIGINT PRIMARY KEY, rating DOUBLE PRECISION NOT NULL DEFAULT 1500.0,
            updated_ts BIGINT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS prematch_snapshots (
            match_id BIGINT PRIMARY KEY, created_ts BIGINT, payload TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS predictions (
            id BIGSERIAL PRIMARY KEY,
            match_id BIGINT, league_id BIGINT, kickoff_ts BIGINT,
            created_ts BIGINT, phase TEXT, minute INTEGER,
            market TEXT, suggestion TEXT,
            prob DOUBLE PRECISION, threshold_pct DOUBLE PRECISION,
            odds DOUBLE PRECISION, fair_prob DOUBLE PRECISION,
            ev_pct DOUBLE PRECISION, decision TEXT)""")

        for stmt in [
            "ALTER TABLE match_results ADD COLUMN IF NOT EXISTS league_id BIGINT",
            "ALTER TABLE match_results ADD COLUMN IF NOT EXISTS kickoff_ts BIGINT",
            "ALTER TABLE tips ADD COLUMN IF NOT EXISTS odds DOUBLE PRECISION",
            "ALTER TABLE tips ADD COLUMN IF NOT EXISTS book TEXT",
            "ALTER TABLE tips ADD COLUMN IF NOT EXISTS ev_pct DOUBLE PRECISION",
            "ALTER TABLE tips ADD COLUMN IF NOT EXISTS confidence_raw DOUBLE PRECISION",
            "ALTER TABLE tips ADD COLUMN IF NOT EXISTS fair_prob DOUBLE PRECISION",
            "ALTER TABLE tips ADD COLUMN IF NOT EXISTS kickoff_ts BIGINT",
            "ALTER TABLE tips ADD COLUMN IF NOT EXISTS is_prematch INTEGER DEFAULT 0",
            "ALTER TABLE tips ADD COLUMN IF NOT EXISTS stake_units DOUBLE PRECISION",
            "ALTER TABLE tips ADD COLUMN IF NOT EXISTS closing_odds DOUBLE PRECISION",
            "ALTER TABLE tips ADD COLUMN IF NOT EXISTS clv_pct DOUBLE PRECISION",
            "ALTER TABLE tip_snapshots ADD COLUMN IF NOT EXISTS kickoff_ts BIGINT",
            "ALTER TABLE prematch_snapshots ADD COLUMN IF NOT EXISTS kickoff_ts BIGINT",
        ]:
            try:
                c.execute(stmt)
            except Exception as e:
                log.warning("[SCHEMA] %s -> %s", stmt, e)

        for stmt in [
            "CREATE INDEX IF NOT EXISTS idx_results_league ON match_results (league_id)",
            "CREATE INDEX IF NOT EXISTS idx_results_kickoff ON match_results (kickoff_ts)",
            "CREATE INDEX IF NOT EXISTS idx_results_updated ON match_results (updated_ts DESC)",
            "CREATE INDEX IF NOT EXISTS idx_tips_created ON tips (created_ts DESC)",
            "CREATE INDEX IF NOT EXISTS idx_tips_match ON tips (match_id)",
            "CREATE INDEX IF NOT EXISTS idx_tips_sent ON tips (sent_ok, created_ts DESC)",
            "CREATE INDEX IF NOT EXISTS idx_tips_clv ON tips (is_prematch, closing_odds, kickoff_ts)",
            "CREATE INDEX IF NOT EXISTS idx_snap_by_match ON tip_snapshots (match_id, created_ts DESC)",
            "CREATE INDEX IF NOT EXISTS idx_snap_kickoff ON tip_snapshots (kickoff_ts)",
            "CREATE INDEX IF NOT EXISTS idx_pre_snap_ts ON prematch_snapshots (created_ts DESC)",
            "CREATE INDEX IF NOT EXISTS idx_pre_snap_kickoff ON prematch_snapshots (kickoff_ts)",
            "CREATE INDEX IF NOT EXISTS idx_pred_match ON predictions (match_id)",
            "CREATE INDEX IF NOT EXISTS idx_pred_created ON predictions (created_ts DESC)",
        ]:
            try:
                c.execute(stmt)
            except Exception as e:
                log.warning("[SCHEMA] %s -> %s", stmt, e)

        if _env_flag("PURGE_LEGACY_HARVEST_TIPS", "1"):
            try:
                n = c.execute("DELETE FROM tips WHERE suggestion='HARVEST'").rowcount
                if n:
                    log.info("[SCHEMA] removed %d legacy HARVEST rows from tips", n)
            except Exception as e:
                log.warning("[SCHEMA] HARVEST purge failed: %s", e)


# ───────── Telegram ─────────
def send_telegram(text: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("[TELEGRAM] not sent — TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is unset")
        return False
    try:
        r = session.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                         data={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML",
                               "disable_web_page_preview": True}, timeout=10)
        if not r.ok:
            log.warning("[TELEGRAM] send failed: HTTP %s — %s", r.status_code, r.text[:300])
        return r.ok
    except Exception as e:
        log.warning("[TELEGRAM] send raised: %s", e)
        return False


# ───────── API ─────────
# In-process, resets when the UTC calendar day rolls over. This is visibility
# only (single gunicorn worker per railway.json, so one counter is accurate) -
# 429s used to be logged at DEBUG, i.e. invisible under the default INFO
# level, so a plan running over its daily request cap failed silently with no
# symptom beyond fewer tips. See /admin/status -> api_usage.
_api_call_lock = threading.Lock()
_api_call_stats = {"day": None, "total": 0, "rate_limited": 0}


def _track_api_call(status_code: Optional[int]) -> Dict[str, Any]:
    today = datetime.now(TZ_UTC).strftime("%Y-%m-%d")
    with _api_call_lock:
        if _api_call_stats["day"] != today:
            _api_call_stats.update(day=today, total=0, rate_limited=0)
        _api_call_stats["total"] += 1
        if status_code == 429:
            _api_call_stats["rate_limited"] += 1
        return dict(_api_call_stats)


def _api_call_stats_snapshot() -> Dict[str, Any]:
    with _api_call_lock:
        return dict(_api_call_stats)


def _api_get(url: str, params: dict, timeout: int = 15):
    if not API_KEY:
        return None
    try:
        r = session.get(url, headers=HEADERS, params=params, timeout=timeout)
        if r.ok:
            _track_api_call(None)
            return r.json()
        stats = _track_api_call(r.status_code)
        if r.status_code == 429:
            log.warning("[API] 429 rate-limited on %s (today: %d calls, %d rate-limited)",
                        url, stats["total"], stats["rate_limited"])
        else:
            log.debug("[API] HTTP %s for %s — %s", r.status_code, url, r.text[:200])
        return None
    except Exception as e:
        log.debug("[API] request raised for %s: %s", url, e)
        return None


_BLOCK_PATTERNS = ["u17", "u18", "u19", "u20", "u21", "u23", "youth", "junior",
                   "reserve", "res.", "friendlies", "friendly"]


def _blocked_league(league_obj: dict) -> bool:
    """
    LEAGUE_ALLOW_IDS, when set, is a hard allowlist: only those league IDs are
    scanned and everything else is blocked, _BLOCK_PATTERNS/LEAGUE_DENY_IDS
    included. There is no reliable "division tier" signal to pattern-match a
    league name against (naming conventions vary per country - "Championship"
    is England's 2nd tier, "Segunda División" is Spain's, neither looks
    "lower" by name), so an opt-in allowlist of leagues actually worth their
    API cost is the only safe way to cut the rest. Unset (the default),
    behaviour is unchanged: block by name pattern, then by LEAGUE_DENY_IDS.
    """
    lg = league_obj or {}
    league_id = str(lg.get("id") or "")
    allow = [x.strip() for x in os.getenv("LEAGUE_ALLOW_IDS", "").split(",") if x.strip()]
    if allow:
        return league_id not in allow
    txt = f"{lg.get('country','')} {lg.get('name','')} {lg.get('type','')}".lower()
    if any(p in txt for p in _BLOCK_PATTERNS):
        return True
    deny = [x.strip() for x in os.getenv("LEAGUE_DENY_IDS", "").split(",") if x.strip()]
    return league_id in deny


def _kickoff_ts_of(fx: dict) -> int:
    return int(_fixture_ts(fx) or 0)


# ───────── Live fetches ─────────
def fetch_match_stats(fid: int) -> list:
    cached = STATS_CACHE.get(fid, _MISS)
    if cached is not _MISS:
        return cached
    js = _api_get(f"{FOOTBALL_API_URL}/statistics", {"fixture": fid}) or {}
    out = js.get("response", []) if isinstance(js, dict) else []
    STATS_CACHE.set(fid, out)
    return out


def fetch_match_events(fid: int) -> list:
    cached = EVENTS_CACHE.get(fid, _MISS)
    if cached is not _MISS:
        return cached
    js = _api_get(f"{FOOTBALL_API_URL}/events", {"fixture": fid}) or {}
    out = js.get("response", []) if isinstance(js, dict) else []
    EVENTS_CACHE.set(fid, out)
    return out


def fetch_live_matches() -> List[dict]:
    js = _api_get(FOOTBALL_API_URL, {"live": "all"}) or {}
    matches = [m for m in (js.get("response", []) if isinstance(js, dict) else [])
               if not _blocked_league(m.get("league") or {})]
    eligible = []
    for m in matches:
        st = ((m.get("fixture", {}) or {}).get("status", {}) or {})
        elapsed = st.get("elapsed")
        short = (st.get("short") or "").upper()
        if elapsed is None or elapsed > 120 or short not in INPLAY_STATUSES:
            continue
        eligible.append(m)

    def _hydrate(m: dict) -> dict:
        fid = (m.get("fixture", {}) or {}).get("id")
        try:
            with ThreadPoolExecutor(max_workers=2) as ex:
                fs = ex.submit(fetch_match_stats, fid)
                fe = ex.submit(fetch_match_events, fid)
                stats, events = fs.result(), fe.result()
        except Exception as e:
            log.warning("[LIVE] stats/events fetch failed for fixture %s: %s", fid, e)
            stats, events = [], []
        m["statistics"] = stats
        m["events"] = events
        return m

    if not eligible:
        return []
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(eligible)))) as ex:
        return list(ex.map(_hydrate, eligible))


# ───────── League base rates ─────────
def _global_rates() -> Dict[str, float]:
    cached = _LEAGUE_RATE_CACHE.get("__GLOBAL__", _MISS)
    if cached is not _MISS:
        return cached
    with db_conn() as c:
        row = c.execute("""
            SELECT AVG(btts_yes)::float,
                   AVG(CASE WHEN final_goals_h+final_goals_a>2 THEN 1.0 ELSE 0.0 END)::float,
                   AVG(CASE WHEN final_goals_h+final_goals_a>3 THEN 1.0 ELSE 0.0 END)::float,
                   COUNT(*)::bigint
            FROM match_results""").fetchone()
    out = {"btts": float(row[0] if row[0] is not None else DEFAULT_LEAGUE_RATES["btts"]),
           "ov25": float(row[1] if row[1] is not None else DEFAULT_LEAGUE_RATES["ov25"]),
           "ov35": float(row[2] if row[2] is not None else DEFAULT_LEAGUE_RATES["ov35"]),
           "n": int(row[3] or 0)}
    _LEAGUE_RATE_CACHE.set("__GLOBAL__", out)
    return out


def get_league_rates(league_id: Optional[int]) -> Dict[str, float]:
    if not league_id:
        return _global_rates()
    key = f"L{league_id}"
    cached = _LEAGUE_RATE_CACHE.get(key, _MISS)
    if cached is not _MISS:
        return cached
    with db_conn() as c:
        row = c.execute("""
            SELECT AVG(btts_yes)::float,
                   AVG(CASE WHEN final_goals_h+final_goals_a>2 THEN 1.0 ELSE 0.0 END)::float,
                   AVG(CASE WHEN final_goals_h+final_goals_a>3 THEN 1.0 ELSE 0.0 END)::float,
                   COUNT(*)::bigint
            FROM match_results WHERE league_id=%s""", (league_id,)).fetchone()
    n = int(row[3] or 0)
    out = _global_rates() if n < LEAGUE_RATE_MIN_N else {
        "btts": float(row[0] if row[0] is not None else DEFAULT_LEAGUE_RATES["btts"]),
        "ov25": float(row[1] if row[1] is not None else DEFAULT_LEAGUE_RATES["ov25"]),
        "ov35": float(row[2] if row[2] is not None else DEFAULT_LEAGUE_RATES["ov35"]),
        "n": n}
    _LEAGUE_RATE_CACHE.set(key, out)
    return out


# ───────── Raw in-play extraction ─────────
def _num(v) -> float:
    try:
        if isinstance(v, str) and v.strip().endswith("%"):
            return float(v.strip()[:-1])
        return float(v or 0)
    except Exception:
        return 0.0


def extract_raw_inplay(m: dict) -> Dict[str, float]:
    """Pull the RAW_INPLAY_KEYS out of an API fixture object. Nothing derived."""
    home = m["teams"]["home"]["name"]
    away = m["teams"]["away"]["name"]
    stats: Dict[str, Dict[str, Any]] = {}
    for s in (m.get("statistics") or []):
        t = (s.get("team") or {}).get("name")
        if t:
            stats[t] = {(i.get("type") or ""): i.get("value") for i in (s.get("statistics") or [])}
    sh = stats.get(home, {}) or {}
    sa = stats.get(away, {}) or {}

    red_h = red_a = 0
    for ev_ in (m.get("events") or []):
        if (ev_.get("type", "") or "").lower() == "card":
            d = (ev_.get("detail", "") or "").lower()
            if "red" in d or "second yellow" in d:
                t = (ev_.get("team") or {}).get("name") or ""
                if t == home:
                    red_h += 1
                elif t == away:
                    red_a += 1

    return {
        "minute": float(((m.get("fixture") or {}).get("status") or {}).get("elapsed") or 0),
        "goals_h": _num((m.get("goals") or {}).get("home")),
        "goals_a": _num((m.get("goals") or {}).get("away")),
        "xg_h": _num(sh.get("Expected Goals", sh.get("expected_goals", 0))),
        "xg_a": _num(sa.get("Expected Goals", sa.get("expected_goals", 0))),
        "sot_h": _num(sh.get("Shots on Goal", 0)),
        "sot_a": _num(sa.get("Shots on Goal", 0)),
        "cor_h": _num(sh.get("Corner Kicks", 0)),
        "cor_a": _num(sa.get("Corner Kicks", 0)),
        "pos_h": _num(sh.get("Ball Possession", 0)),
        "pos_a": _num(sa.get("Ball Possession", 0)),
        "red_h": float(red_h), "red_a": float(red_a),
        "total_shots_h": _num(sh.get("Total Shots", 0)),
        "total_shots_a": _num(sa.get("Total Shots", 0)),
        "shots_inside_h": _num(sh.get("Shots insidebox", 0)),
        "shots_inside_a": _num(sa.get("Shots insidebox", 0)),
        "fouls_h": _num(sh.get("Fouls", 0)),
        "fouls_a": _num(sa.get("Fouls", 0)),
        "yellow_h": _num(sh.get("Yellow Cards", 0)),
        "yellow_a": _num(sa.get("Yellow Cards", 0)),
        "saves_h": _num(sh.get("Goalkeeper Saves", 0)),
        "saves_a": _num(sa.get("Goalkeeper Saves", 0)),
        "passes_h": _num(sh.get("Total passes", 0)),
        "passes_a": _num(sa.get("Total passes", 0)),
        "passes_acc_h": _num(sh.get("Passes accurate", 0)),
        "passes_acc_a": _num(sa.get("Passes accurate", 0)),
    }


def extract_features(m: dict) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Returns (raw, features). Features come from feature_spec, shared with training."""
    raw = extract_raw_inplay(m)
    league_id = ((m.get("league") or {}).get("id"))
    lr = get_league_rates(int(league_id) if league_id else None)
    return raw, build_inplay_features(raw, lr)


def stats_coverage_ok(raw: Dict[str, float], minute: int) -> bool:
    """Coverage is required from TIP_MIN_MINUTE onward: an all-zero stats vector
    makes the model output sigmoid(intercept), which carries no match info."""
    require_from = int(os.getenv("REQUIRE_STATS_MINUTE", str(TIP_MIN_MINUTE)))
    require_fields = int(os.getenv("REQUIRE_DATA_FIELDS", "2"))
    if minute < require_from:
        return False
    fields = [raw.get("xg_h", 0) + raw.get("xg_a", 0),
              raw.get("sot_h", 0) + raw.get("sot_a", 0),
              raw.get("cor_h", 0) + raw.get("cor_a", 0),
              max(raw.get("pos_h", 0), raw.get("pos_a", 0))]
    return sum(1 for v in fields if (v or 0) > 0) >= max(0, require_fields)


def _league_name(m: dict) -> Tuple[int, str]:
    lg = (m.get("league") or {}) or {}
    return int(lg.get("id") or 0), f"{lg.get('country','')} - {lg.get('name','')}".strip(" -")


def _teams(m: dict) -> Tuple[str, str]:
    t = (m.get("teams") or {}) or {}
    return t.get("home", {}).get("name", ""), t.get("away", {}).get("name", "")


def _pretty_score(m: dict) -> str:
    g = m.get("goals") or {}
    return f"{g.get('home') or 0}-{g.get('away') or 0}"


# ───────── Models ─────────
MODEL_KEYS_ORDER = ["model_latest:{name}", "model:{name}"]


def _sigmoid(x: float) -> float:
    if x < -50:
        return 1e-22
    if x > 50:
        return 1 - 1e-22
    return 1 / (1 + math.exp(-x))


def _logit(p: float) -> float:
    p = max(EPS, min(1 - EPS, float(p)))
    return math.log(p / (1 - p))


def load_model_from_settings(name: str) -> Optional[Dict[str, Any]]:
    cached = _MODELS_CACHE.get(name, _MISS)
    if cached is not _MISS:
        return cached
    mdl = None
    for pat in MODEL_KEYS_ORDER:
        raw = get_setting_cached(pat.format(name=name))
        if not raw:
            continue
        try:
            tmp = json.loads(raw)
            tmp.setdefault("intercept", 0.0)
            tmp.setdefault("weights", {})
            cal = tmp.get("calibration") or {}
            if isinstance(cal, dict):
                cal.setdefault("method", "sigmoid")
                cal.setdefault("a", 1.0)
                cal.setdefault("b", 0.0)
                tmp["calibration"] = cal
            mdl = tmp
            break
        except Exception as e:
            log.warning("[MODEL] parse %s failed: %s", name, e)
    _MODELS_CACHE.set(name, mdl)
    return mdl


def _linpred(feat: Dict[str, float], mdl: Dict[str, Any]) -> float:
    """
    Apply the model's persisted StandardScaler before the dot product. Training
    fits on standardized features (so L2 penalises every feature on a comparable
    scale) and ships mean/scale inside the blob, so serving reproduces the
    transform exactly. Blobs without a scaler are treated as raw.
    """
    scaler = mdl.get("scaler") or {}
    mean = scaler.get("mean") or {}
    scale = scaler.get("scale") or {}
    s = float(mdl.get("intercept") or 0.0)
    for k, w in (mdl.get("weights") or {}).items():
        x = float(feat.get(k, 0.0))
        if k in mean:
            sc = float(scale.get(k, 1.0)) or 1.0
            x = (x - float(mean[k])) / sc
        s += float(w or 0.0) * x
    return s


def _calibrate(p: float, cal: Dict[str, Any]) -> float:
    a = float((cal or {}).get("a", 1.0))
    b = float((cal or {}).get("b", 0.0))
    return _sigmoid(a * _logit(p) + b)


def _score_prob(feat: Dict[str, float], mdl: Dict[str, Any]) -> float:
    p = _sigmoid(_linpred(feat, mdl))
    cal = mdl.get("calibration") or {}
    if cal:
        try:
            p = _calibrate(p, cal)
        except Exception:
            pass
    return max(0.0, min(1.0, float(p)))


def _load_ou_model_for_line(line: float, prefix: str = "") -> Optional[Dict[str, Any]]:
    name = f"{prefix}OU_{_fmt_line(line)}"
    mdl = load_model_from_settings(name)
    if mdl is None and not prefix and abs(line - 2.5) < 1e-6:
        mdl = load_model_from_settings("O25")
    return mdl


# ───────── Odds ─────────
def _min_odds_for_market(market: str) -> float:
    if market.startswith("Over/Under"):
        return MIN_ODDS_OU
    if market == "BTTS":
        return MIN_ODDS_BTTS
    if market == "1X2":
        return MIN_ODDS_1X2
    if market == "Double Chance":
        return MIN_ODDS_DC
    if market == "Draw No Bet":
        return MIN_ODDS_DNB
    return 1.01


def _txt(v: Any) -> str:
    """
    Coerce an odds-feed field to a string.

    THE BUG THIS FIXES: 530 occurrences of
        [ODDS] parse failed ... 'int' object has no attribute 'lower'
    in six hours. The parser did `(v.get("value") or "").strip().lower()` and
    `(mkt.get("name","")).lower()`. When the feed returns a NUMBER rather than a
    string — which it does for some bookmakers' market names and for Asian-style
    Over/Under values — `int or ""` evaluates to the int (it's truthy), which is
    then handed straight to .lower()/.strip(). None, ints and floats all become
    strings here instead.
    """
    if v is None:
        return ""
    return v if isinstance(v, str) else str(v)


def _market_name_normalize(s: Any) -> str:
    s = _txt(s).lower()
    if "both teams" in s or "btts" in s:
        return "BTTS"
    if "double chance" in s:
        return "DC"
    if "draw no bet" in s:
        return "DNB"
    if "match winner" in s or "winner" in s or "1x2" in s:
        return "1X2"
    if "over/under" in s or "total" in s or "goals" in s:
        return "OU"
    return s


def _odd_value(v: dict) -> float:
    """Parse a price, tolerating strings, commas and nulls. 0.0 means unusable."""
    try:
        raw = v.get("odd")
        if raw is None:
            return 0.0
        return float(_txt(raw).replace(",", "."))
    except Exception:
        return 0.0


def _parse_book_market(mkt: dict) -> Optional[Tuple[str, Dict[str, float]]]:
    """Parse one bookmaker's one market into {market_key: {selection: odds}}."""
    mname = _market_name_normalize(mkt.get("name"))
    vals = mkt.get("values") or []
    if mname == "BTTS":
        d = {}
        for v in vals:
            lbl = _txt(v.get("value")).strip().lower()
            o = _odd_value(v)
            if o <= 1.0:
                continue
            if lbl.startswith("yes"):
                d["Yes"] = o
            elif lbl.startswith("no"):
                d["No"] = o
        return ("BTTS", d) if d else None
    if mname == "1X2":
        d = {}
        for v in vals:
            lbl = _txt(v.get("value")).strip().lower()
            o = _odd_value(v)
            if o <= 1.0:
                continue
            if lbl in ("home", "1"):
                d["Home"] = o
            elif lbl in ("draw", "x"):
                d["Draw"] = o
            elif lbl in ("away", "2"):
                d["Away"] = o
        return ("1X2", d) if d else None
    if mname == "DC":
        d = {}
        for v in vals:
            lbl = _txt(v.get("value")).strip().lower().replace(" ", "")
            o = _odd_value(v)
            if o <= 1.0:
                continue
            if lbl in ("home/draw", "1x", "homeordraw"):
                d["1X"] = o
            elif lbl in ("draw/away", "x2", "draworaway"):
                d["X2"] = o
            elif lbl in ("home/away", "12", "homeoraway"):
                d["12"] = o
        return ("DC", d) if d else None
    if mname == "DNB":
        d = {}
        for v in vals:
            lbl = _txt(v.get("value")).strip().lower()
            o = _odd_value(v)
            if o <= 1.0:
                continue
            if lbl in ("home", "1"):
                d["Home"] = o
            elif lbl in ("away", "2"):
                d["Away"] = o
        return ("DNB", d) if d else None
    if mname == "OU":
        by_line: Dict[str, Dict[str, float]] = {}
        for v in vals:
            lbl = _txt(v.get("value")).strip().lower()
            if "over" not in lbl and "under" not in lbl:
                continue
            o = _odd_value(v)
            if o <= 1.0:
                continue
            try:
                ln = float(lbl.split()[-1].replace(",", "."))
            except Exception:
                continue
            key = f"OU_{_fmt_line(ln)}"
            side = "Over" if "over" in lbl else "Under"
            by_line.setdefault(key, {})[side] = o
        return ("OU_MULTI", by_line) if by_line else None
    return None


# Selections required before a market can be de-vigged, and what its true
# probabilities sum to. OU lines are keyed OU_<line> at runtime and default to
# 2 selections / total 1.0.
_MARKET_SELECTION_COUNT = {"BTTS": 2, "1X2": 3, "DC": 3, "DNB": 2}


def fetch_odds(fid: int, live: bool) -> Dict[str, Any]:
    """
    Returns, per market key:
      {"best": {selection: {"odds": float, "book": str}},
       "fair": {selection: float},          # consensus de-vigged probability
       "n_books": int}

    De-vigging happens WITHIN each bookmaker's complete market (de-vigging
    across best-of-many-books prices would produce a fake sub-1.0 overround and
    a systematically optimistic fair price), then averages across books. The
    best available price across all books is used separately for EV.

    FIX: the market total is now looked up per market rather than assumed to be
    1.0. Double Chance sums to 2.0 — see feature_spec.devig().
    """
    key = (fid, bool(live))
    cached = ODDS_CACHE.get(key, _MISS)
    if cached is not _MISS:
        return cached

    params: Dict[str, Any] = {"fixture": fid}
    if ODDS_BOOKMAKER_ID:
        params["bookmaker"] = ODDS_BOOKMAKER_ID
    js = _api_get(ODDS_LIVE_URL if live else ODDS_PREMATCH_URL, params) or {}

    best: Dict[str, Dict[str, Dict[str, Any]]] = {}
    fair_acc: Dict[str, Dict[str, List[float]]] = {}
    books_seen: Dict[str, set] = {}
    parse_errors = 0

    # FIX: the try/except used to wrap the ENTIRE response, and its handler
    # reset best/fair/books to {}. So a single malformed value, in a single
    # market, from a single bookmaker, threw away every price for that fixture —
    # all markets, all books. That is why 530 parse failures produced zero
    # priced candidates rather than merely degraded ones. Failures are now
    # isolated to the market that caused them; everything else survives.
    for r in (js.get("response", []) if isinstance(js, dict) else []):
        for bk in (r.get("bookmakers") or []):
            book_name = _txt(bk.get("name")) or "Book"
            per_market: Dict[str, Dict[str, float]] = {}
            for mkt in (bk.get("bets") or []):
                try:
                    parsed = _parse_book_market(mkt)
                except Exception as e:
                    parse_errors += 1
                    log.debug("[ODDS] fixture %s book %s market %r unparseable: %s",
                              fid, book_name, mkt.get("name"), e)
                    continue
                if not parsed:
                    continue
                mkey, payload = parsed
                if mkey == "OU_MULTI":
                    for k, sel in payload.items():
                        per_market.setdefault(k, {}).update(sel)
                else:
                    per_market.setdefault(mkey, {}).update(payload)

            for mkey, sel in per_market.items():
                try:
                    books_seen.setdefault(mkey, set()).add(book_name)
                    for name, o in sel.items():
                        cur = best.setdefault(mkey, {}).get(name)
                        if cur is None or o > cur["odds"]:
                            best[mkey][name] = {"odds": float(o), "book": book_name}
                    # Only de-vig a COMPLETE market, and normalise to the total
                    # that market's true probabilities actually sum to.
                    needed = _MARKET_SELECTION_COUNT.get(mkey, 2)
                    if len(sel) >= needed:
                        total = MARKET_PROBABILITY_TOTAL.get(mkey, 1.0)
                        implied = {k: 1.0 / v for k, v in sel.items() if v > 1.0}
                        for k, p in devig(implied, market_total=total).items():
                            fair_acc.setdefault(mkey, {}).setdefault(k, []).append(p)
                except Exception as e:
                    parse_errors += 1
                    log.debug("[ODDS] fixture %s market %s aggregation failed: %s", fid, mkey, e)

    if parse_errors:
        log.debug("[ODDS] fixture %s (live=%s): %d market(s) unparseable, %d market(s) usable",
                  fid, live, parse_errors, len(best))

    out: Dict[str, Any] = {}
    for mkey, sels in best.items():
        fair = {k: (sum(v) / len(v)) for k, v in (fair_acc.get(mkey) or {}).items() if v}
        out[mkey] = {"best": sels, "fair": fair, "n_books": len(books_seen.get(mkey, ()))}
    ODDS_CACHE.set(key, out)
    return out


def _market_key_and_selection(market_text: str, suggestion: str) -> Tuple[Optional[str], Optional[str]]:
    mt = market_text.replace("PRE ", "")
    if mt == "BTTS":
        return "BTTS", ("Yes" if suggestion.endswith("Yes") else "No")
    if mt == "1X2":
        if suggestion == "Home Win":
            return "1X2", "Home"
        if suggestion == "Away Win":
            return "1X2", "Away"
        return None, None
    if mt == "Double Chance":
        if suggestion.endswith("1X"):
            return "DC", "1X"
        if suggestion.endswith("X2"):
            return "DC", "X2"
        if suggestion.endswith("12"):
            return "DC", "12"
        return None, None
    if mt == "Draw No Bet":
        if suggestion.endswith("Home"):
            return "DNB", "Home"
        if suggestion.endswith("Away"):
            return "DNB", "Away"
        return None, None
    if mt.startswith("Over/Under"):
        try:
            ln = _fmt_line(float(suggestion.split()[1]))
        except Exception:
            return None, None
        return f"OU_{ln}", ("Over" if suggestion.startswith("Over") else "Under")
    return None, None


class PriceCheck(dict):
    """Result of _price_gate. Dict so it serialises straight into the log row."""


def _price_gate(market_text: str, suggestion: str, fid: int, prob: float, live: bool) -> PriceCheck:
    """
    Single place where a candidate meets the market.

    Gates, in order:
      1. odds exist (unless ALLOW_TIPS_WITHOUT_ODDS)
      2. odds within [min_for_market, MAX_ODDS_ALL]
      3. a de-vigged fair price is computable (unless REQUIRE_FAIR_PRICE=0)
      4. EV at the available price >= EDGE_MIN_BPS
      5. edge over the fair price >= FAIR_EDGE_MIN_BPS
      6. edge over the fair price <= MAX_MODEL_EDGE_BPS  (model-sanity cap)
    """
    res = PriceCheck(passed=False, odds=None, book=None, fair_prob=None,
                     ev_pct=None, decision="no_odds", n_books=0)
    mkey, sel = _market_key_and_selection(market_text, suggestion)
    if not mkey or not sel:
        res["decision"] = "unmapped_market"
        return res

    odds_map = fetch_odds(fid, live=live) if API_KEY else {}
    entry = odds_map.get(mkey) or {}
    best = (entry.get("best") or {}).get(sel)
    res["n_books"] = int(entry.get("n_books") or 0)

    if not best:
        res["decision"] = "no_odds"
        res["passed"] = bool(ALLOW_TIPS_WITHOUT_ODDS)
        return res

    odds = float(best["odds"])
    res["odds"] = odds
    res["book"] = best.get("book")

    if not (_min_odds_for_market(market_text.replace("PRE ", "")) <= odds <= MAX_ODDS_ALL):
        res["decision"] = "odds_out_of_range"
        return res

    fair = (entry.get("fair") or {}).get(sel)
    if fair is None:
        res["decision"] = "no_fair_price"
        if REQUIRE_FAIR_PRICE:
            return res
    elif res["n_books"] < MIN_BOOKS_FOR_FAIR:
        res["decision"] = "too_few_books"
        res["fair_prob"] = float(fair)
        if REQUIRE_FAIR_PRICE:
            return res
    else:
        res["fair_prob"] = float(fair)

    edge_ev = _ev(prob, odds)
    res["ev_pct"] = round(edge_ev * 100.0, 2)
    if int(round(edge_ev * 10000)) < EDGE_MIN_BPS:
        res["decision"] = "ev_below_min"
        return res

    if fair is not None:
        fair_edge = prob - float(fair)
        res["fair_edge_pct"] = round(fair_edge * 100.0, 2)
        if int(round(fair_edge * 10000)) < FAIR_EDGE_MIN_BPS:
            res["decision"] = "fair_edge_below_min"
            return res
        if int(round(fair_edge * 10000)) > MAX_MODEL_EDGE_BPS:
            # The model claims to be enormously smarter than a liquid market.
            # That is a model failure, not an opportunity.
            res["decision"] = "edge_implausible"
            log.warning("[SANITY] fixture %s %s: model %.1f%% vs fair %.1f%% — suppressed",
                        fid, suggestion, prob * 100, float(fair) * 100)
            return res

    res["passed"] = True
    res["decision"] = "tipped"
    return res


def _stake_units(prob: float, odds: Optional[float]) -> Optional[float]:
    """Fractional Kelly on the model probability, capped."""
    if not odds:
        return None
    f = kelly_fraction(prob, odds) * KELLY_FRACTION
    f = max(0.0, min(f, MAX_STAKE_PCT / 100.0))
    return round(BANKROLL_UNITS * f, 2)


# ───────── Prediction log ─────────
# Cap on prediction-log rows per fixture. The logs showed 15,590 candidate rows
# from ONE prematch scan (1,297 fixtures x ~12 candidates). At 8 scans a day that
# is ~124k rows/day of which the overwhelming majority are far below any
# threshold and carry no calibration information. Keeping the highest-probability
# few per fixture preserves everything the calibration curve actually needs.
PREDICTION_LOG_MAX_PER_FIXTURE = int(os.getenv("PREDICTION_LOG_MAX_PER_FIXTURE", "4"))


def _trim_fixture_predictions(rows: List[tuple]) -> List[tuple]:
    """Keep every tipped candidate plus the top-N remaining by probability."""
    if len(rows) <= PREDICTION_LOG_MAX_PER_FIXTURE:
        return rows
    tipped = [r for r in rows if r[13] == "tipped"]
    rest = sorted((r for r in rows if r[13] != "tipped"), key=lambda r: r[8], reverse=True)
    keep = max(0, PREDICTION_LOG_MAX_PER_FIXTURE - len(tipped))
    return tipped + rest[:keep]


_PRED_SQL = ("INSERT INTO predictions(match_id,league_id,kickoff_ts,created_ts,phase,minute,"
             "market,suggestion,prob,threshold_pct,odds,fair_prob,ev_pct,decision) "
             "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)")


def _log_predictions(rows: List[tuple]) -> None:
    if not PREDICTION_LOG_ENABLE or not rows:
        return
    try:
        with db_conn() as c:
            c.executemany(_PRED_SQL, rows)
    except Exception as e:
        log.warning("[PRED-LOG] insert failed: %s", e)


# ───────── Elo ─────────
def get_team_ratings_bulk(team_ids: List[int]) -> Dict[int, float]:
    ids = [t for t in set(team_ids) if t]
    if not ids:
        return {}
    with db_conn() as c:
        rows = c.execute("SELECT team_id, rating FROM team_ratings WHERE team_id = ANY(%s)", (ids,)).fetchall()
    out = {int(t): ELO_DEFAULT for t in ids}
    for tid, rating in rows:
        out[int(tid)] = float(rating)
    return out


def update_team_ratings(home_id: int, away_id: int, gh: int, ga: int) -> None:
    if not home_id or not away_id:
        return
    ratings = get_team_ratings_bulk([home_id, away_id])
    rh = ratings.get(home_id, ELO_DEFAULT)
    ra = ratings.get(away_id, ELO_DEFAULT)
    exp_h = 1.0 / (1.0 + 10 ** ((ra - (rh + ELO_HOME_ADV)) / 400.0))
    score_h = 1.0 if gh > ga else (0.5 if gh == ga else 0.0)
    new_rh = rh + ELO_K * (score_h - exp_h)
    new_ra = ra + ELO_K * ((1.0 - score_h) - (1.0 - exp_h))
    now = int(time.time())
    with db_conn() as c:
        c.executemany(
            "INSERT INTO team_ratings(team_id,rating,updated_ts) VALUES(%s,%s,%s) "
            "ON CONFLICT(team_id) DO UPDATE SET rating=EXCLUDED.rating, updated_ts=EXCLUDED.updated_ts",
            [(home_id, float(new_rh), now), (away_id, float(new_ra), now)])


# ───────── Snapshots ─────────
def save_snapshot_from_match(m: dict, raw: Dict[str, float]) -> None:
    """
    Writes ONLY to tip_snapshots, in the RAW field set that
    feature_spec.build_inplay_features() consumes — so training reconstructs the
    identical vector.
    """
    fx = m.get("fixture", {}) or {}
    lg = m.get("league", {}) or {}
    fid = int(fx.get("id"))
    payload = {
        "raw": {k: float(raw.get(k, 0.0)) for k in RAW_INPLAY_KEYS},
        "league_id": int(lg.get("id") or 0),
        "kickoff_ts": _kickoff_ts_of(m),
        "schema": 2,
    }
    with db_conn() as c:
        c.execute("INSERT INTO tip_snapshots(match_id, created_ts, payload, kickoff_ts) "
                  "VALUES (%s,%s,%s,%s) ON CONFLICT (match_id, created_ts) "
                  "DO UPDATE SET payload=EXCLUDED.payload, kickoff_ts=EXCLUDED.kickoff_ts",
                  (fid, int(time.time()), json.dumps(payload)[:200000], payload["kickoff_ts"]))


def save_prematch_snapshot(fid: int, feat: Dict[str, float], kickoff_ts: int) -> None:
    payload = {"feat": {k: v for k, v in feat.items() if not k.startswith("_")},
               "kickoff_ts": int(kickoff_ts), "schema": 2}
    with db_conn() as c:
        c.execute("INSERT INTO prematch_snapshots(match_id, created_ts, payload, kickoff_ts) "
                  "VALUES (%s,%s,%s,%s) ON CONFLICT (match_id) DO UPDATE SET "
                  "created_ts=EXCLUDED.created_ts, payload=EXCLUDED.payload, kickoff_ts=EXCLUDED.kickoff_ts",
                  (fid, int(time.time()), json.dumps(payload)[:200000], int(kickoff_ts)))


# ───────── Grading ─────────
def _parse_ou_line_from_suggestion(s: str) -> Optional[float]:
    for tok in (s or "").split():
        try:
            return float(tok)
        except Exception:
            continue
    return None


def _tip_outcome_for_result(suggestion: str, res: Dict[str, Any]) -> Optional[int]:
    """1 = win, 0 = loss, None = push/void or ungradeable."""
    gh = int(res.get("final_goals_h") or 0)
    ga = int(res.get("final_goals_a") or 0)
    total = gh + ga
    btts = int(res.get("btts_yes") or 0)
    s = (suggestion or "").strip()
    if s.startswith("Over") or s.startswith("Under"):
        line = _parse_ou_line_from_suggestion(s)
        if line is None:
            return None
        if abs(total - line) < 1e-9:
            return None
        return int(total > line) if s.startswith("Over") else int(total < line)
    if s == "BTTS: Yes":
        return 1 if btts == 1 else 0
    if s == "BTTS: No":
        return 1 if btts == 0 else 0
    if s == "Home Win":
        return 1 if gh > ga else 0
    if s == "Away Win":
        return 1 if ga > gh else 0
    if s == "Double Chance: 1X":
        return 1 if gh >= ga else 0
    if s == "Double Chance: X2":
        return 1 if ga >= gh else 0
    if s == "Double Chance: 12":
        return 1 if gh != ga else 0
    # Draw No Bet voids on a draw — stake returned, not a loss.
    if s == "Draw No Bet: Home":
        return None if gh == ga else int(gh > ga)
    if s == "Draw No Bet: Away":
        return None if gh == ga else int(ga > gh)
    return None


def _fixture_by_id(mid: int) -> Optional[dict]:
    js = _api_get(FOOTBALL_API_URL, {"id": mid}) or {}
    arr = (js.get("response") or []) if isinstance(js, dict) else []
    return arr[0] if arr else None


def backfill_results_for_open_matches(max_rows: int = 400) -> int:
    """
    Covers fixtures that only ever produced a snapshot, not just fixtures that
    produced a tip — otherwise Elo only advances for matches you happened to
    tip, leaving pm_rating_diff at exactly 0 for most fixtures.
    """
    now_ts = int(time.time())
    cutoff = now_ts - BACKFILL_DAYS * 24 * 3600
    with db_conn() as c:
        rows = c.execute("""
            WITH seen AS (
              SELECT match_id, MAX(created_ts) AS last_ts FROM tips
              WHERE created_ts >= %s GROUP BY match_id
              UNION ALL
              SELECT match_id, MAX(created_ts) FROM tip_snapshots
              WHERE created_ts >= %s GROUP BY match_id
              UNION ALL
              SELECT match_id, MAX(created_ts) FROM prematch_snapshots
              WHERE created_ts >= %s GROUP BY match_id
            ), agg AS (
              SELECT match_id, MAX(last_ts) AS last_ts FROM seen GROUP BY match_id
            )
            SELECT a.match_id FROM agg a
            LEFT JOIN match_results r ON r.match_id = a.match_id
            WHERE r.match_id IS NULL ORDER BY a.last_ts DESC LIMIT %s
        """, (cutoff, cutoff, cutoff, max_rows)).fetchall()

    updated = 0
    for (mid,) in rows:
        fx = _fixture_by_id(int(mid))
        if not fx:
            continue
        st = (((fx.get("fixture") or {}).get("status") or {}).get("short") or "").upper()
        if st not in FINAL_STATUSES:
            continue
        g = fx.get("goals") or {}
        gh, ga = int(g.get("home") or 0), int(g.get("away") or 0)
        league_id = int(((fx.get("league") or {}).get("id")) or 0) or None
        with db_conn() as c2:
            c2.execute(
                "INSERT INTO match_results(match_id, final_goals_h, final_goals_a, btts_yes, "
                "updated_ts, league_id, kickoff_ts) VALUES(%s,%s,%s,%s,%s,%s,%s) "
                "ON CONFLICT(match_id) DO UPDATE SET final_goals_h=EXCLUDED.final_goals_h, "
                "final_goals_a=EXCLUDED.final_goals_a, btts_yes=EXCLUDED.btts_yes, "
                "updated_ts=EXCLUDED.updated_ts, league_id=EXCLUDED.league_id, "
                "kickoff_ts=EXCLUDED.kickoff_ts",
                (int(mid), gh, ga, 1 if (gh > 0 and ga > 0) else 0, int(time.time()),
                 league_id, _kickoff_ts_of(fx)))
        try:
            th = ((fx.get("teams") or {}).get("home") or {}).get("id")
            ta = ((fx.get("teams") or {}).get("away") or {}).get("id")
            if th and ta:
                update_team_ratings(int(th), int(ta), gh, ga)
        except Exception as e:
            log.warning("[ELO] rating update failed for match %s: %s", mid, e)
        updated += 1
    if updated:
        log.info("[RESULTS] backfilled %d", updated)
    return updated


# ───────── Closing line value ─────────
def capture_closing_lines(limit: int = 200) -> int:
    """
    Records, for every prematch tip, the best available price at (or just after)
    kickoff, and stores tip_odds/closing_odds - 1.

    PREMATCH ONLY. An in-play bet has no well-defined closing line — the market
    for "Over 2.5 at minute 62" ceases to exist the moment the state changes.
    """
    if not CLV_ENABLE:
        return 0
    now = int(time.time())
    with db_conn() as c:
        rows = c.execute("""
            SELECT match_id, created_ts, market, suggestion, odds
            FROM tips
            WHERE is_prematch=1 AND closing_odds IS NULL AND odds IS NOT NULL
              AND kickoff_ts IS NOT NULL
              AND kickoff_ts <= %s AND kickoff_ts >= %s
            ORDER BY kickoff_ts DESC LIMIT %s
        """, (now, now - CLV_MAX_AGE_MIN * 60, limit)).fetchall()

    n = 0
    for (mid, cts, market, sugg, tip_odds) in rows:
        odds_map = fetch_odds(int(mid), live=False)
        mkey, sel = _market_key_and_selection(market or "", sugg or "")
        best = ((odds_map.get(mkey) or {}).get("best") or {}).get(sel) if mkey else None
        if not best:
            continue
        closing = float(best["odds"])
        if closing <= 1.0:
            continue
        clv = (float(tip_odds) / closing - 1.0) * 100.0
        with db_conn() as c2:
            c2.execute("UPDATE tips SET closing_odds=%s, clv_pct=%s WHERE match_id=%s AND created_ts=%s",
                       (closing, round(clv, 3), mid, cts))
        n += 1
    if n:
        log.info("[CLV] captured closing prices for %d tips", n)
    return n


def compute_clv(days: Optional[int] = None) -> Dict[str, Any]:
    cutoff = int(time.time()) - days * 86400 if days else 0
    with db_conn() as c:
        rows = c.execute("""
            SELECT market, clv_pct FROM tips
            WHERE clv_pct IS NOT NULL AND created_ts >= %s
        """, (cutoff,)).fetchall()
    if not rows:
        return {"n": 0, "note": "No closing prices captured yet. CLV is prematch-only "
                                "and needs at least one full kickoff cycle."}
    by: Dict[str, List[float]] = {}
    allv: List[float] = []
    for mkt, clv in rows:
        by.setdefault(mkt or "?", []).append(float(clv))
        allv.append(float(clv))
    allv.sort()

    def _summary(v: List[float]) -> Dict[str, Any]:
        return {"n": len(v), "mean_clv_pct": round(sum(v) / len(v), 2),
                "median_clv_pct": round(sorted(v)[len(v) // 2], 2),
                "beat_close_pct": round(100.0 * sum(1 for x in v if x > 0) / len(v), 1)}

    return {"overall": _summary(allv),
            "by_market": {k: _summary(v) for k, v in by.items() if v},
            "note": "mean_clv_pct > 0 sustained over a few hundred prematch bets is the "
                    "strongest available evidence of a real edge. Negative CLV with positive "
                    "ROI means you have been lucky, not right. Prematch only."}


# ───────── Message formatting ─────────
def _format_tip_message(home, away, league, minute, score, suggestion, prob_pct,
                        raw=None, odds=None, book=None, ev_pct=None, fair_prob=None,
                        stake=None, kickoff_txt=None, prematch=False):
    raw = raw or {}
    stat = ""
    if not prematch and any(raw.get(k, 0) for k in ("xg_h", "xg_a", "sot_h", "sot_a", "cor_h", "cor_a",
                                                    "pos_h", "pos_a", "red_h", "red_a")):
        stat = (f"\n📊 xG {raw.get('xg_h',0):.2f}-{raw.get('xg_a',0):.2f}"
                f" • SOT {int(raw.get('sot_h',0))}-{int(raw.get('sot_a',0))}"
                f" • CK {int(raw.get('cor_h',0))}-{int(raw.get('cor_a',0))}")
        if raw.get("pos_h", 0) or raw.get("pos_a", 0):
            stat += f" • POS {int(raw.get('pos_h',0))}%–{int(raw.get('pos_a',0))}%"
        if raw.get("red_h", 0) or raw.get("red_a", 0):
            stat += f" • RED {int(raw.get('red_h',0))}-{int(raw.get('red_a',0))}"

    money = ""
    if odds:
        money = f"\n💰 <b>Odds:</b> {odds:.2f} @ {book or 'Book'}"
        if fair_prob is not None:
            money += f"  •  <b>Fair:</b> {1.0/max(fair_prob,1e-9):.2f} ({fair_prob*100:.1f}%)"
        if ev_pct is not None:
            money += f"\n📐 <b>EV:</b> {ev_pct:+.1f}%"
        if stake:
            money += f"  •  <b>Stake:</b> {stake:.2f}u"

    header = "🏅 <b>Prematch Tip</b>" if prematch else "⚽️ <b>New Tip!</b>"
    when = (f"⏰ <b>Kickoff (Berlin):</b> {kickoff_txt}" if prematch
            else f"🕒 <b>Minute:</b> {minute}'  |  <b>Score:</b> {escape(score)}")
    return (f"{header}\n"
            f"<b>Match:</b> {escape(home)} vs {escape(away)}\n"
            f"{when}\n"
            f"<b>Tip:</b> {escape(suggestion)}\n"
            f"📈 <b>Confidence:</b> {prob_pct:.1f}%{money}\n"
            f"🏆 <b>League:</b> {escape(league)}{stat}")


def _kickoff_berlin(utc_iso: Optional[str]) -> str:
    try:
        if not utc_iso:
            return "TBD"
        dt = datetime.fromisoformat(utc_iso.replace("Z", "+00:00"))
        return dt.astimezone(BERLIN_TZ).strftime("%H:%M")
    except Exception:
        return "TBD"


# ───────── Thresholds ─────────
def _get_market_threshold(m: str) -> float:
    """
    Confidence threshold for a market.

    FIX: markets in DERIVED_MARKETS (Double Chance, Draw No Bet) have no model
    of their own — they are transforms of the 1X2 heads. Previously training
    never wrote a threshold for them, so this function fell through to
    CONF_THRESHOLD (70) while the 1X2 heads they derive from could be suppressed
    at 85 for failing their holdout. Double Chance then fired at 70 on any
    fixture with a competent home side.

    train_models.py now writes and holdout-verifies both. If a derived market's
    threshold is still absent (e.g. training has not run since this release), it
    is treated as SUPPRESSED rather than defaulted — an unverified derived
    market must not trade.
    """
    base = m.replace("PRE ", "")
    try:
        v = get_setting_cached(f"conf_threshold:{m}")
        if v is not None:
            return float(v)
    except Exception:
        pass
    if base in DERIVED_MARKETS:
        log.debug("[THRESHOLD] %s has no verified threshold — suppressed at %.1f%%", m, MAX_THRESH + 100)
        return MAX_THRESH + 100.0  # unreachable: never fires
    return float(CONF_THRESHOLD)


def _get_market_threshold_pre(m: str) -> float:
    return _get_market_threshold(f"PRE {m}")


def _is_threshold_locked(m: str) -> bool:
    try:
        v = get_setting_cached(f"conf_threshold_locked:{m}")
        return v is not None and str(v).strip() == "1"
    except Exception:
        return False


# ───────── Candidate generation ─────────
def _candidate_is_sane(sug: str, feat: Dict[str, float]) -> bool:
    """
    Reject selections already decided by the current score.

    Over/Under and BTTS can settle mid-match (three goals in means Over 2.5 has
    already won and Under 2.5 has already lost), so those are checked here.
    Double Chance, Draw No Bet and 1X2 cannot settle before full time — a
    two-goal lead at minute 88 is near-certain but not decided — so they have no
    branch. Near-certain cases are filtered by the per-market minimum odds
    (MIN_ODDS_DC / MIN_ODDS_DNB), which a 1.01 price cannot clear.
    """
    goals_sum = feat.get("goals_sum", 0.0)
    goals_h = (feat.get("goals_sum", 0.0) + feat.get("goals_diff", 0.0)) / 2.0
    goals_a = (feat.get("goals_sum", 0.0) - feat.get("goals_diff", 0.0)) / 2.0
    if sug.startswith("Over"):
        ln = _parse_ou_line_from_suggestion(sug)
        return ln is not None and goals_sum <= ln - 1e-9
    if sug.startswith("Under"):
        ln = _parse_ou_line_from_suggestion(sug)
        return ln is not None and goals_sum < ln - 1e-9
    if sug.startswith("BTTS"):
        return not (goals_h > 0 and goals_a > 0)
    return True


def _ou_candidates(feat: Dict[str, float], prefix: str, thr_fn) -> List[Tuple[str, str, float, float]]:
    raw_probs: List[Tuple[float, float]] = []
    for line in OU_LINES:
        mdl = _load_ou_model_for_line(line, prefix=prefix)
        if not mdl:
            continue
        raw_probs.append((line, _score_prob(feat, mdl)))
    if not raw_probs:
        return []
    # Independent per-line heads can emit P(Over 3.5) > P(Over 2.5), which is
    # impossible — Over 3.5 is a strict subset of Over 2.5. Project onto the
    # non-increasing-in-line constraint. No-op when already coherent.
    coherent = enforce_ou_monotonicity(raw_probs) if len(raw_probs) > 1 else dict(raw_probs)
    out = []
    for line in sorted(coherent):
        p_over = coherent[line]
        mk = f"Over/Under {_fmt_line(line)}"
        thr = thr_fn(mk)
        for sug, p in ((f"Over {_fmt_line(line)} Goals", p_over),
                       (f"Under {_fmt_line(line)} Goals", 1.0 - p_over)):
            out.append((mk, sug, p, thr))
    return out


def _btts_candidates(feat: Dict[str, float], prefix: str, thr_fn) -> List[Tuple[str, str, float, float]]:
    mdl = load_model_from_settings(f"{prefix}BTTS_YES")
    if not mdl:
        return []
    p = _score_prob(feat, mdl)
    thr = thr_fn("BTTS")
    return [("BTTS", "BTTS: Yes", p, thr), ("BTTS", "BTTS: No", 1.0 - p, thr)]


def _wld_probs(feat: Dict[str, float], prefix: str) -> Optional[Tuple[float, float, float]]:
    """
    Normalised (p_home, p_draw, p_away) summing to 1, or None if heads missing.

    The old code did `s = ph + pa; ph, pa = ph/s, pa/s`, which yields
    P(Home | not a draw): a Draw-No-Bet probability. But the suggestion is
    graded as a LOSS on a draw and priced against 1X2 Home odds, both full 1X2
    semantics. That inflated every 1X2 probability by roughly 1/(1 - P(draw)) ≈
    1.30-1.35x. The draw head is trained and used, so the normalisation is over
    all three outcomes.
    """
    mh = load_model_from_settings(f"{prefix}WLD_HOME")
    md = load_model_from_settings(f"{prefix}WLD_DRAW")
    ma = load_model_from_settings(f"{prefix}WLD_AWAY")
    if not (mh and ma):
        return None
    ph = _score_prob(feat, mh)
    pa = _score_prob(feat, ma)
    if md:
        pd_ = _score_prob(feat, md)
    else:
        # Rather than silently reverting to the DNB error, fall back to an
        # empirical draw prior so the denominator is still a full 1X2 one.
        pd_ = float(os.getenv("FALLBACK_DRAW_PROB", "0.26"))
        log.warning("[1X2] %sWLD_DRAW model missing — using fallback draw prior %.2f", prefix, pd_)
    s = max(EPS, ph + pd_ + pa)
    return ph / s, pd_ / s, pa / s


def _wld_candidates(feat: Dict[str, float], prefix: str, thr_fn) -> List[Tuple[str, str, float, float]]:
    """1X2. The draw is suppressed from OUTPUT (we don't tip draws) but not from
    the DENOMINATOR — see _wld_probs."""
    probs = _wld_probs(feat, prefix)
    if probs is None:
        return []
    ph, _pd, pa = probs
    thr = thr_fn("1X2")
    return [("1X2", "Home Win", ph, thr), ("1X2", "Away Win", pa, thr)]


def _dc_dnb_candidates(feat: Dict[str, float], prefix: str, thr_fn) -> List[Tuple[str, str, float, float]]:
    """
    Double Chance and Draw No Bet — algebraic transforms of the same
    (p_home, p_draw, p_away) the 1X2 heads produce, via the shared
    feature_spec.derive_dc_dnb() that training also uses.

    These have no model of their own, so they used to have no threshold either.
    They are now trained and holdout-verified by train_models.py exactly like
    every other market, and _get_market_threshold() suppresses them outright if
    that verification has never run.
    """
    probs = _wld_probs(feat, prefix)
    if probs is None:
        return []
    d = derive_dc_dnb(*probs)
    dc_thr = thr_fn("Double Chance")
    dnb_thr = thr_fn("Draw No Bet")
    return [
        ("Double Chance", "Double Chance: 1X", d["1X"], dc_thr),
        ("Double Chance", "Double Chance: X2", d["X2"], dc_thr),
        ("Double Chance", "Double Chance: 12", d["12"], dc_thr),
        ("Draw No Bet", "Draw No Bet: Home", d["DNB_Home"], dnb_thr),
        ("Draw No Bet", "Draw No Bet: Away", d["DNB_Away"], dnb_thr),
    ]


def _correlation_blocked(suggestion: str, taken: List[str]) -> bool:
    for fam in _CORRELATION_FAMILIES:
        if suggestion in fam and any(t in fam for t in taken):
            return True
    return False


# ───────── In-play scan ─────────
def _last_snapshot_ts_bulk(fids: List[int]) -> Dict[int, int]:
    """
    Most recent tip_snapshots.created_ts per fixture, in ONE query per scan.

    Reading from the table rather than an in-process dict means the harvest
    cadence survives restarts and stays correct with more than one instance
    scanning. On failure this returns {}, which makes every fixture look
    un-harvested and therefore harvests on this scan — the safe direction.
    """
    ids = [int(f) for f in set(fids) if f]
    if not ids:
        return {}
    try:
        with db_conn() as c:
            rows = c.execute(
                "SELECT match_id, MAX(created_ts) FROM tip_snapshots "
                "WHERE match_id = ANY(%s) GROUP BY match_id", (ids,)).fetchall()
        return {int(mid): int(ts or 0) for mid, ts in rows}
    except Exception as e:
        log.warning("[HARVEST] last-snapshot lookup failed (harvesting anyway): %s", e)
        return {}


# In-memory snapshot of every live match's FULL market breakdown (every
# candidate production_scan() evaluates, not just the ones that clear the
# tipping bar). production_scan() already computes this every 5 minutes for
# every live fixture regardless of whether anything gets tipped, so exposing
# it costs zero extra API calls - it's the same numbers the tipping logic
# already has, just not thrown away. Backs GET /dashboard/live.
_live_snapshot_lock = threading.Lock()
_live_snapshot: Dict[str, Any] = {"updated_ts": 0, "matches": []}


def _build_live_match_entry(fid: int, league: str, league_id: int, home: str, away: str,
                            score: str, minute: int,
                            candidates: List[Tuple[str, str, float, float]]) -> Dict[str, Any]:
    markets = [
        {"market": mt, "suggestion": sg, "prob_pct": round(float(pr) * 100.0, 1),
         "threshold_pct": round(float(thr), 1)}
        for mt, sg, pr, thr in candidates
    ]
    return {
        "fixture_id": fid, "league": league, "league_id": league_id,
        "home": home, "away": away, "score": score, "minute": minute,
        "markets": markets,
        # Count of candidates clearing their own threshold - lets the
        # dashboard flag "worth a look" matches without re-deriving it from
        # every row client-side.
        "hits": sum(1 for m in markets if m["prob_pct"] >= m["threshold_pct"]),
    }


def _set_live_snapshot(matches: List[Dict[str, Any]]) -> None:
    with _live_snapshot_lock:
        _live_snapshot["updated_ts"] = int(time.time())
        _live_snapshot["matches"] = matches


def _get_live_snapshot() -> Dict[str, Any]:
    with _live_snapshot_lock:
        return {"updated_ts": _live_snapshot["updated_ts"], "matches": list(_live_snapshot["matches"])}


def production_scan() -> Tuple[int, int]:
    matches = fetch_live_matches()
    live_seen = len(matches)
    if live_seen == 0:
        log.info("[PROD] no live")
        _set_live_snapshot([])
        return 0, 0

    saved = 0
    now_ts = int(time.time())
    pred_rows: List[tuple] = []
    live_snapshot_matches: List[Dict[str, Any]] = []
    harvested = 0
    no_coverage = 0
    last_snap: Dict[int, int] = {}
    if HARVEST_MODE:
        try:
            last_snap = _last_snapshot_ts_bulk(
                [int((m.get("fixture") or {}).get("id") or 0) for m in matches])
        except Exception as e:
            log.warning("[HARVEST] cadence lookup failed (harvesting anyway): %s", e)

    for m in matches:
        try:
            fid = int((m.get("fixture", {}) or {}).get("id") or 0)
            if not fid:
                continue

            raw, feat = extract_features(m)
            minute = int(feat.get("minute", 0))
            if minute < TIP_MIN_MINUTE:
                continue

            # Harvest BEFORE the coverage and duplicate gates. Three separate
            # concerns, previously collapsed into one:
            #   - the cooldown stops us TIPPING a fixture repeatedly
            #   - the coverage gate stops us TIPPING on an all-zero stats vector
            #   - harvesting is DATA COLLECTION and should be blocked by neither
            #
            # THE REGRESSION THIS FIXES: I moved stats_coverage_ok() ahead of
            # the harvest block and tightened it from minute 35 to minute 8. Any
            # fixture whose /fixtures/statistics feed was empty then hit
            # `continue` before harvesting. Six hours of logs: 33 scans,
            # 316 live fixtures seen, harvested=1. In-play data collection was
            # effectively dead, which is also why candidates_logged was 0 —
            # live predictions are logged on the harvest tick.
            #
            # The cadence is stated in TIME, not "the elapsed minute happens to
            # be divisible by 3", because nothing aligns the scan schedule with
            # that arithmetic.
            is_harvest_tick = (
                HARVEST_MODE
                and minute >= TRAIN_MIN_MINUTE
                and (now_ts - last_snap.get(fid, 0)) >= HARVEST_EVERY_MINUTES * 60
            )
            if is_harvest_tick:
                try:
                    save_snapshot_from_match(m, raw)
                    last_snap[fid] = now_ts
                    harvested += 1
                except Exception as e:
                    log.warning("[HARVEST] snapshot failed for %s: %s", fid, e)

            # Coverage governs TIPPING only. An all-zero stats vector makes the
            # model output sigmoid(intercept), which carries no match
            # information — fine to record, not fine to bet on.
            if not stats_coverage_ok(raw, minute):
                no_coverage += 1
                continue

            if DUP_COOLDOWN_MIN > 0:
                with db_conn() as c:
                    dup = c.execute(
                        "SELECT 1 FROM tips WHERE match_id=%s AND created_ts>=%s "
                        "AND suggestion<>'HARVEST' LIMIT 1",
                        (fid, now_ts - DUP_COOLDOWN_MIN * 60)).fetchone()
                if dup:
                    continue

            league_id, league = _league_name(m)
            home, away = _teams(m)
            score = _pretty_score(m)
            kickoff = _kickoff_ts_of(m)

            candidates = (_ou_candidates(feat, "", _get_market_threshold)
                          + _btts_candidates(feat, "", _get_market_threshold)
                          + _wld_candidates(feat, "", _get_market_threshold)
                          + _dc_dnb_candidates(feat, "", _get_market_threshold))
            candidates = [c for c in candidates
                          if c[1] in ALLOWED_SUGGESTIONS and _candidate_is_sane(c[1], feat)]
            candidates.sort(key=lambda x: x[2], reverse=True)

            # Full breakdown for the dashboard, independent of whether any of
            # these candidates go on to clear a threshold or the price gate.
            live_snapshot_matches.append(_build_live_match_entry(
                fid, league, league_id, home, away, score, minute, candidates))

            per_match = 0
            taken: List[str] = []
            base_now = int(time.time())
            fixture_preds: List[tuple] = []

            for idx, (market_txt, suggestion, prob, thr) in enumerate(candidates):
                below = prob * 100.0 < thr
                capped = per_match >= max(1, PREDICTIONS_PER_MATCH)
                pc = PriceCheck(passed=False, odds=None, book=None, fair_prob=None, ev_pct=None,
                                decision="below_threshold" if below else "per_match_cap")
                if not below and not capped:
                    pc = _price_gate(market_txt, suggestion, fid, prob, live=True)
                    if pc["passed"] and _correlation_blocked(suggestion, taken):
                        extra = int(round((pc.get("ev_pct") or 0) * 100)) - EDGE_MIN_BPS
                        if extra < CORRELATED_EXTRA_EV_BPS:
                            pc["passed"] = False
                            pc["decision"] = "correlated_with_existing_tip"

                if PREDICTION_LOG_ENABLE and (is_harvest_tick or pc["passed"]) and prob >= PREDICTION_LOG_MIN_PROB:
                    fixture_preds.append((fid, league_id, kickoff, base_now, "live", minute,
                                          market_txt, suggestion, float(prob), float(thr),
                                          pc.get("odds"), pc.get("fair_prob"), pc.get("ev_pct"),
                                          pc["decision"]))

                if not pc["passed"]:
                    continue

                created_ts = base_now + idx
                prob_pct = round(float(prob) * 100.0, 1)
                stake = _stake_units(prob, pc.get("odds"))

                with db_conn() as c:
                    c.execute(
                        "INSERT INTO tips(match_id,league_id,league,home,away,market,suggestion,"
                        "confidence,confidence_raw,score_at_tip,minute,created_ts,odds,book,ev_pct,"
                        "fair_prob,kickoff_ts,is_prematch,stake_units,sent_ok) "
                        "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,0,%s,0) "
                        "ON CONFLICT (match_id, created_ts) DO NOTHING",
                        (fid, league_id, league, home, away, market_txt, suggestion,
                         float(prob_pct), float(prob), score, minute, created_ts,
                         pc.get("odds"), pc.get("book"), pc.get("ev_pct"), pc.get("fair_prob"),
                         kickoff, stake))

                sent = send_telegram(_format_tip_message(
                    home, away, league, minute, score, suggestion, prob_pct, raw,
                    pc.get("odds"), pc.get("book"), pc.get("ev_pct"), pc.get("fair_prob"), stake))
                if sent:
                    with db_conn() as c:
                        c.execute("UPDATE tips SET sent_ok=1 WHERE match_id=%s AND created_ts=%s",
                                  (fid, created_ts))

                saved += 1
                per_match += 1
                taken.append(suggestion)
                if per_match >= max(1, PREDICTIONS_PER_MATCH):
                    break
                if MAX_TIPS_PER_SCAN and saved >= MAX_TIPS_PER_SCAN:
                    break
            pred_rows.extend(_trim_fixture_predictions(fixture_preds))
            if MAX_TIPS_PER_SCAN and saved >= MAX_TIPS_PER_SCAN:
                break
        except Exception as e:
            log.exception("[PROD] failure: %s", e)
            continue

    _log_predictions(pred_rows)
    _set_live_snapshot(live_snapshot_matches)
    log.info("[PROD] saved=%d live_seen=%d candidates_logged=%d harvested=%d no_coverage=%d",
             saved, live_seen, len(pred_rows), harvested, no_coverage)
    if live_seen and no_coverage >= live_seen:
        # Every live fixture lacked usable statistics. Harvesting still runs, but
        # the rows are goals/minute only and nothing can be tipped. Usually means
        # the API plan does not include /fixtures/statistics.
        log.warning("[PROD] no fixture had usable statistics this scan (%d live). "
                    "Check that your API-Football plan includes /fixtures/statistics — "
                    "without it the in-play model has nothing to score.", live_seen)
    return saved, live_seen


def score_live_matches_now() -> Tuple[List[Dict[str, Any]], int]:
    """
    Read-only, on-demand equivalent of production_scan()'s live-scoring step:
    fetches whatever is live RIGHT NOW and scores every market for every
    fixture with usable stats coverage. Deliberately does NOT write to
    tips/predictions, harvest snapshots, or send Telegram - it exists purely
    to answer "what does the model see right now" for a human looking at the
    dashboard, e.g. via /dashboard/live/refresh.

    Deliberately duplicates production_scan()'s candidate-building step
    rather than sharing it, so a bug in this read-only path can never affect
    what the live tipping bot actually does.
    """
    matches = fetch_live_matches()
    live_seen = len(matches)
    out: List[Dict[str, Any]] = []
    for m in matches:
        try:
            fid = int((m.get("fixture", {}) or {}).get("id") or 0)
            if not fid:
                continue
            raw, feat = extract_features(m)
            minute = int(feat.get("minute", 0))
            if minute < TIP_MIN_MINUTE or not stats_coverage_ok(raw, minute):
                continue

            league_id, league = _league_name(m)
            home, away = _teams(m)
            score = _pretty_score(m)

            candidates = (_ou_candidates(feat, "", _get_market_threshold)
                          + _btts_candidates(feat, "", _get_market_threshold)
                          + _wld_candidates(feat, "", _get_market_threshold)
                          + _dc_dnb_candidates(feat, "", _get_market_threshold))
            candidates = [c for c in candidates
                          if c[1] in ALLOWED_SUGGESTIONS and _candidate_is_sane(c[1], feat)]
            candidates.sort(key=lambda x: x[2], reverse=True)

            out.append(_build_live_match_entry(fid, league, league_id, home, away, score,
                                               minute, candidates))
        except Exception as e:
            log.warning("[LIVE-SCORE] failed for a fixture: %s", e)
            continue
    return out, live_seen


# ───────── Prematch data ─────────
def _api_last_fixtures(team_id: int, n: int = 5) -> List[dict]:
    key = ("last", team_id, n)
    cached = TEAM_FORM_CACHE.get(key, _MISS)
    if cached is not _MISS:
        return cached
    js = _api_get(FOOTBALL_API_URL, {"team": team_id, "last": n}) or {}
    out = js.get("response", []) if isinstance(js, dict) else []
    TEAM_FORM_CACHE.set(key, out)
    return out


def _api_h2h(home_id: int, away_id: int, n: int = 5) -> List[dict]:
    key = ("h2h", home_id, away_id, n)
    cached = TEAM_FORM_CACHE.get(key, _MISS)
    if cached is not _MISS:
        return cached
    js = _api_get(f"{FOOTBALL_API_URL}/headtohead", {"h2h": f"{home_id}-{away_id}", "last": n}) or {}
    out = js.get("response", []) if isinstance(js, dict) else []
    TEAM_FORM_CACHE.set(key, out)
    return out


def _collect_todays_prematch_fixtures() -> List[dict]:
    today_local = datetime.now(BERLIN_TZ).date()
    start_local = datetime.combine(today_local, datetime.min.time(), tzinfo=BERLIN_TZ)
    end_local = start_local + timedelta(days=1)
    dates_utc = {start_local.astimezone(TZ_UTC).date(),
                 (end_local - timedelta(seconds=1)).astimezone(TZ_UTC).date()}
    fixtures = []
    for d in sorted(dates_utc):
        js = _api_get(FOOTBALL_API_URL, {"date": d.strftime("%Y-%m-%d")}) or {}
        for r in (js.get("response", []) if isinstance(js, dict) else []):
            if (((r.get("fixture") or {}).get("status") or {}).get("short") or "").upper() == "NS":
                fixtures.append(r)
    fixtures = [f for f in fixtures if not _blocked_league(f.get("league") or {})]
    if PREMATCH_LEAGUE_IDS:
        fixtures = [f for f in fixtures
                    if int(((f.get("league") or {}).get("id") or 0)) in PREMATCH_LEAGUE_IDS]
    else:
        log.warning("[PREMATCH] PREMATCH_LEAGUE_IDS is empty — scanning %d fixtures worldwide. "
                    "This will consume ~%d API calls on a cold cache.",
                    len(fixtures), len(fixtures) * 3)
    return fixtures


def extract_prematch_features(fx: dict) -> Dict[str, float]:
    teams = fx.get("teams") or {}
    th = (teams.get("home") or {}).get("id")
    ta = (teams.get("away") or {}).get("id")
    if not th or not ta:
        return {}
    with ThreadPoolExecutor(max_workers=3) as ex:
        f_h = ex.submit(_api_last_fixtures, th, 5)
        f_a = ex.submit(_api_last_fixtures, ta, 5)
        f_x = ex.submit(_api_h2h, th, ta, 5)
        last_h, last_a, h2h = f_h.result(), f_a.result(), f_x.result()
    ratings = get_team_ratings_bulk([th, ta])
    league_id = ((fx.get("league") or {}).get("id"))
    lr = get_league_rates(int(league_id) if league_id else None)
    kickoff = _fixture_ts(fx) or time.time()
    return assemble_prematch_features(th, ta, last_h, last_a, h2h, kickoff,
                                      ratings.get(th, ELO_DEFAULT), ratings.get(ta, ELO_DEFAULT), lr)


def _safe_extract_prematch_features(fx: dict) -> Dict[str, float]:
    try:
        return extract_prematch_features(fx)
    except Exception as e:
        log.warning("[PREMATCH] feature extraction failed for fixture %s: %s",
                    ((fx.get("fixture") or {}).get("id")), e)
        return {}


def _load_fresh_snapshot_feats(fids: List[int], now_ts: int) -> Dict[int, Dict[str, float]]:
    if not fids:
        return {}
    with db_conn() as c:
        rows = c.execute(
            "SELECT match_id, payload FROM prematch_snapshots "
            "WHERE match_id = ANY(%s) AND created_ts >= %s",
            (fids, now_ts - PREMATCH_SNAPSHOT_TTL_SEC)).fetchall()
    out = {}
    for mid, payload in rows:
        try:
            feat = (json.loads(payload) or {}).get("feat") or {}
            if feat:
                out[int(mid)] = feat
        except Exception:
            continue
    return out


def _get_prematch_features_bulk(fixtures: List[dict]) -> Tuple[Dict[int, Dict[str, float]], Dict[int, Dict[str, float]]]:
    now_ts = int(time.time())
    fid_map = {int((fx.get("fixture") or {}).get("id")): fx
               for fx in fixtures if (fx.get("fixture") or {}).get("id")}
    cached = _load_fresh_snapshot_feats(list(fid_map.keys()), now_ts)
    need = [fx for fid, fx in fid_map.items() if fid not in cached]
    fetched: Dict[int, Dict[str, float]] = {}
    if need:
        with ThreadPoolExecutor(max_workers=8) as ex:
            feats = list(ex.map(_safe_extract_prematch_features, need))
        for fx, feat in zip(need, feats):
            fid = (fx.get("fixture") or {}).get("id")
            if fid and feat:
                fetched[int(fid)] = feat
    log.info("[PREMATCH] features: %d reused, %d fetched (%d fixtures)",
             len(cached), len(fetched), len(fid_map))
    out = dict(cached)
    out.update(fetched)
    return out, fetched


def prematch_scan_save() -> int:
    fixtures = _collect_todays_prematch_fixtures()
    if not fixtures:
        return 0
    feats_by_fid, freshly_fetched = _get_prematch_features_bulk(fixtures)
    saved = 0
    pred_rows: List[tuple] = []

    for fx in fixtures:
        fixture = fx.get("fixture") or {}
        lg = fx.get("league") or {}
        teams = fx.get("teams") or {}
        fid = int(fixture.get("id") or 0)
        feat = feats_by_fid.get(fid)
        if not fid or not feat:
            continue

        home = (teams.get("home") or {}).get("name", "")
        away = (teams.get("away") or {}).get("name", "")
        league_id = int(lg.get("id") or 0)
        league = f"{lg.get('country','')} - {lg.get('name','')}".strip(" -")
        kickoff = _kickoff_ts_of(fx)
        kickoff_txt = _kickoff_berlin(fixture.get("date"))

        if fid in freshly_fetched:
            try:
                save_prematch_snapshot(fid, feat, kickoff)
            except Exception as e:
                log.warning("[PREMATCH] snapshot save failed for %s: %s", fid, e)

        if PREMATCH_DEDUP_ENABLE:
            with db_conn() as c:
                dup = c.execute("SELECT 1 FROM tips WHERE match_id=%s AND is_prematch=1 "
                                "AND suggestion<>'HARVEST' LIMIT 1", (fid,)).fetchone()
            if dup:
                continue

        if MAX_PREMATCH_TIPS_PER_SCAN and saved >= MAX_PREMATCH_TIPS_PER_SCAN:
            break

        candidates = (_ou_candidates(feat, "PRE_", _get_market_threshold_pre)
                      + _btts_candidates(feat, "PRE_", _get_market_threshold_pre)
                      + _wld_candidates(feat, "PRE_", _get_market_threshold_pre)
                      + _dc_dnb_candidates(feat, "PRE_", _get_market_threshold_pre))
        candidates = [c for c in candidates if c[1] in ALLOWED_SUGGESTIONS]
        candidates.sort(key=lambda x: x[2], reverse=True)

        per_match = 0
        taken: List[str] = []
        base_now = int(time.time())
        fixture_preds: List[tuple] = []

        for idx, (mk, sug, prob, thr) in enumerate(candidates):
            below = prob * 100.0 < thr
            capped = per_match >= max(1, PREDICTIONS_PER_MATCH)
            pc = PriceCheck(passed=False, odds=None, book=None, fair_prob=None, ev_pct=None,
                            decision="below_threshold" if below else "per_match_cap")
            if not below and not capped:
                pc = _price_gate(mk, sug, fid, prob, live=False)
                if pc["passed"] and _correlation_blocked(sug, taken):
                    extra = int(round((pc.get("ev_pct") or 0) * 100)) - EDGE_MIN_BPS
                    if extra < CORRELATED_EXTRA_EV_BPS:
                        pc["passed"] = False
                        pc["decision"] = "correlated_with_existing_tip"

            if PREDICTION_LOG_ENABLE and prob >= PREDICTION_LOG_MIN_PROB:
                fixture_preds.append((fid, league_id, kickoff, base_now, "prematch", 0,
                                      f"PRE {mk}", sug, float(prob), float(thr),
                                      pc.get("odds"), pc.get("fair_prob"), pc.get("ev_pct"),
                                      pc["decision"]))

            if not pc["passed"]:
                continue

            created_ts = base_now + idx
            pct = round(float(prob) * 100.0, 1)
            stake = _stake_units(prob, pc.get("odds"))

            with db_conn() as c2:
                c2.execute(
                    "INSERT INTO tips(match_id,league_id,league,home,away,market,suggestion,"
                    "confidence,confidence_raw,score_at_tip,minute,created_ts,odds,book,ev_pct,"
                    "fair_prob,kickoff_ts,is_prematch,stake_units,sent_ok) "
                    "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,NULL,NULL,%s,%s,%s,%s,%s,%s,1,%s,0) "
                    "ON CONFLICT (match_id, created_ts) DO NOTHING",
                    (fid, league_id, league, home, away, f"PRE {mk}", sug,
                     float(pct), float(prob), created_ts, pc.get("odds"), pc.get("book"),
                     pc.get("ev_pct"), pc.get("fair_prob"), kickoff, stake))

            sent = send_telegram(_format_tip_message(
                home, away, league, 0, "", sug, pct, None,
                pc.get("odds"), pc.get("book"), pc.get("ev_pct"), pc.get("fair_prob"),
                stake, kickoff_txt=kickoff_txt, prematch=True))
            if sent:
                with db_conn() as c2:
                    c2.execute("UPDATE tips SET sent_ok=1 WHERE match_id=%s AND created_ts=%s",
                               (fid, created_ts))

            saved += 1
            per_match += 1
            taken.append(sug)
            if per_match >= max(1, PREDICTIONS_PER_MATCH):
                break

        pred_rows.extend(_trim_fixture_predictions(fixture_preds))

    _log_predictions(pred_rows)
    log.info("[PREMATCH] saved=%d candidates_logged=%d", saved, len(pred_rows))
    return saved


def send_match_of_the_day() -> bool:
    fixtures = _collect_todays_prematch_fixtures()
    if MOTD_LEAGUE_IDS:
        fixtures = [f for f in fixtures
                    if int(((f.get("league") or {}).get("id") or 0)) in MOTD_LEAGUE_IDS]
    if not fixtures:
        return send_telegram("🏅 Match of the Day: no eligible fixtures today.")

    feats_by_fid, freshly_fetched = _get_prematch_features_bulk(fixtures)
    for fx in fixtures:
        fid = int((fx.get("fixture") or {}).get("id") or 0)
        if fid in freshly_fetched:
            try:
                save_prematch_snapshot(fid, freshly_fetched[fid], _kickoff_ts_of(fx))
            except Exception:
                pass

    best = None
    for fx in fixtures:
        fixture = fx.get("fixture") or {}
        lg = fx.get("league") or {}
        teams = fx.get("teams") or {}
        fid = int(fixture.get("id") or 0)
        feat = feats_by_fid.get(fid)
        if not feat:
            continue

        candidates = (_ou_candidates(feat, "PRE_", _get_market_threshold_pre)
                      + _btts_candidates(feat, "PRE_", _get_market_threshold_pre)
                      + _wld_candidates(feat, "PRE_", _get_market_threshold_pre)
                      + _dc_dnb_candidates(feat, "PRE_", _get_market_threshold_pre))
        candidates = [c for c in candidates if c[1] in ALLOWED_SUGGESTIONS and c[2] * 100.0 >= c[3]]
        if not candidates:
            continue
        candidates.sort(key=lambda x: x[2], reverse=True)
        mk, sug, prob, _thr = candidates[0]
        if prob * 100.0 < MOTD_CONF_MIN:
            continue

        pc = _price_gate(mk, sug, fid, prob, live=False)
        if not pc["passed"]:
            continue

        item = (prob * 100.0, sug, (teams.get("home") or {}).get("name", ""),
                (teams.get("away") or {}).get("name", ""),
                f"{lg.get('country','')} - {lg.get('name','')}".strip(" -"),
                _kickoff_berlin(fixture.get("date")), pc.get("odds"), pc.get("book"),
                pc.get("ev_pct"), pc.get("fair_prob"), _stake_units(prob, pc.get("odds")))
        if best is None or item[0] > best[0]:
            best = item

    if not best:
        return send_telegram("🏅 Match of the Day: no prematch pick met thresholds.")
    pct, sug, home, away, league, kickoff_txt, odds, book, ev_pct, fair, stake = best
    msg = _format_tip_message(home, away, league, 0, "", sug, pct, None, odds, book,
                              ev_pct, fair, stake, kickoff_txt=kickoff_txt, prematch=True)
    return send_telegram(msg.replace("🏅 <b>Prematch Tip</b>", "🏅 <b>Match of the Day</b>"))


# ───────── Historical backfill ─────────
def _api_fixtures_by_league_season(league_id: int, season: int) -> Tuple[List[dict], dict]:
    js = _api_get(FOOTBALL_API_URL, {"league": league_id, "season": season}) or {}
    diag = {"errors": js.get("errors") if isinstance(js, dict) else "no response",
            "results": js.get("results") if isinstance(js, dict) else None}
    return (js.get("response", []) if isinstance(js, dict) else []), diag


def backfill_historical_prematch(league_id: int, seasons: List[int]) -> Dict[str, int]:
    """
    Reconstructs prematch training data for past seasons of one league using
    ~1 bulk API call per season. Snapshots and results are stamped with the
    fixture's KICKOFF timestamp, not time.time().
    """
    all_fx: Dict[int, dict] = {}
    diags: Dict[str, dict] = {}
    for s in seasons:
        fxs, diag = _api_fixtures_by_league_season(league_id, s)
        diags[str(s)] = diag
        for fx in fxs:
            fid = (fx.get("fixture") or {}).get("id")
            if fid:
                all_fx[fid] = fx
    fixtures = sorted(all_fx.values(), key=_fixture_ts)

    team_history: Dict[int, List[dict]] = {}
    for fx in fixtures:
        st = (((fx.get("fixture") or {}).get("status") or {}).get("short") or "").upper()
        if st not in FINAL_STATUSES:
            continue
        th = ((fx.get("teams") or {}).get("home") or {}).get("id")
        ta = ((fx.get("teams") or {}).get("away") or {}).get("id")
        if th:
            team_history.setdefault(th, []).append(fx)
        if ta:
            team_history.setdefault(ta, []).append(fx)
    for tid in team_history:
        team_history[tid].sort(key=_fixture_ts)

    lr = get_league_rates(league_id)
    elo_local: Dict[int, float] = {}
    snapshots_saved = results_saved = 0
    last_ts = 0.0

    for fx in fixtures:
        st = (((fx.get("fixture") or {}).get("status") or {}).get("short") or "").upper()
        if st not in FINAL_STATUSES:
            continue
        fid = (fx.get("fixture") or {}).get("id")
        th = ((fx.get("teams") or {}).get("home") or {}).get("id")
        ta = ((fx.get("teams") or {}).get("away") or {}).get("id")
        if not fid or not th or not ta:
            continue
        cutoff = _fixture_ts(fx)
        last_ts = max(last_ts, cutoff)

        last_h = [g for g in team_history.get(th, []) if _fixture_ts(g) < cutoff][-5:]
        last_a = [g for g in team_history.get(ta, []) if _fixture_ts(g) < cutoff][-5:]

        def _involves_both(g):
            hh = ((g.get("teams") or {}).get("home") or {}).get("id")
            aa = ((g.get("teams") or {}).get("away") or {}).get("id")
            return {hh, aa} == {th, ta}

        h2h = [g for g in team_history.get(th, [])
               if _fixture_ts(g) < cutoff and _involves_both(g)][-5:]

        rating_h = elo_local.get(th, ELO_DEFAULT)
        rating_a = elo_local.get(ta, ELO_DEFAULT)
        feat = assemble_prematch_features(th, ta, last_h, last_a, h2h, cutoff,
                                          rating_h, rating_a, lr)

        try:
            save_prematch_snapshot(int(fid), feat, int(cutoff))
            snapshots_saved += 1
        except Exception as e:
            log.warning("[HIST-PRE] snapshot save failed for %s: %s", fid, e)

        gh = int((fx.get("goals") or {}).get("home") or 0)
        ga = int((fx.get("goals") or {}).get("away") or 0)
        try:
            with db_conn() as c:
                c.execute(
                    "INSERT INTO match_results(match_id, final_goals_h, final_goals_a, btts_yes, "
                    "updated_ts, league_id, kickoff_ts) VALUES(%s,%s,%s,%s,%s,%s,%s) "
                    "ON CONFLICT(match_id) DO UPDATE SET final_goals_h=EXCLUDED.final_goals_h, "
                    "final_goals_a=EXCLUDED.final_goals_a, btts_yes=EXCLUDED.btts_yes, "
                    "updated_ts=EXCLUDED.updated_ts, league_id=EXCLUDED.league_id, "
                    "kickoff_ts=EXCLUDED.kickoff_ts",
                    (int(fid), gh, ga, 1 if (gh > 0 and ga > 0) else 0,
                     int(time.time()), int(league_id), int(cutoff)))
            results_saved += 1
        except Exception as e:
            log.warning("[HIST-PRE] result save failed for %s: %s", fid, e)

        exp_h = 1.0 / (1.0 + 10 ** ((rating_a - (rating_h + ELO_HOME_ADV)) / 400.0))
        score_h = 1.0 if gh > ga else (0.5 if gh == ga else 0.0)
        elo_local[th] = rating_h + ELO_K * (score_h - exp_h)
        elo_local[ta] = rating_a + ELO_K * ((1.0 - score_h) - (1.0 - exp_h))

    with db_conn() as c:
        for tid, rating in elo_local.items():
            row = c.execute("SELECT updated_ts FROM team_ratings WHERE team_id=%s", (tid,)).fetchone()
            if row and row[0] and int(row[0]) > int(last_ts):
                continue
            c.execute("INSERT INTO team_ratings(team_id,rating,updated_ts) VALUES(%s,%s,%s) "
                      "ON CONFLICT(team_id) DO UPDATE SET rating=EXCLUDED.rating, "
                      "updated_ts=EXCLUDED.updated_ts", (tid, float(rating), int(last_ts)))

    _LEAGUE_RATE_CACHE.invalidate()
    return {"fixtures_seen": len(fixtures), "snapshots_saved": snapshots_saved,
            "results_saved": results_saved, "api_diagnostics_per_season": diags}


# ───────── Analytics ─────────
def _norm_cdf(x: float) -> float:
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def compute_pnl(days: Optional[int] = None, stake: float = 1.0, use_kelly: bool = False) -> Dict[str, Any]:
    cutoff = int(time.time()) - days * 86400 if days else 0
    with db_conn() as c:
        rows = c.execute("""
            SELECT t.market, t.suggestion, t.odds, t.created_ts, t.stake_units, t.clv_pct,
                   r.final_goals_h, r.final_goals_a, r.btts_yes
            FROM tips t JOIN match_results r ON r.match_id = t.match_id
            WHERE t.suggestion<>'HARVEST' AND t.odds IS NOT NULL AND t.created_ts >= %s
            ORDER BY t.created_ts ASC
        """, (cutoff,)).fetchall()

    total_staked = total_profit = 0.0
    n_bets = n_wins = n_push = 0
    by_market: Dict[str, Dict[str, float]] = {}
    equity: List[Dict[str, Any]] = []
    running = 0.0
    clvs: List[float] = []

    for (mkt, sugg, odds, cts, stake_units, clv, gh, ga, btts) in rows:
        outcome = _tip_outcome_for_result(sugg, {"final_goals_h": gh, "final_goals_a": ga, "btts_yes": btts})
        if outcome is None:
            n_push += 1
            continue
        s = float(stake_units) if (use_kelly and stake_units) else float(stake)
        if s <= 0:
            continue
        n_bets += 1
        total_staked += s
        profit = s * (float(odds) - 1.0) if outcome == 1 else -s
        if outcome == 1:
            n_wins += 1
        total_profit += profit
        running += profit
        equity.append({"ts": int(cts), "bankroll": round(running, 2)})
        if clv is not None:
            clvs.append(float(clv))
        d = by_market.setdefault(mkt or "?", {"bets": 0, "wins": 0, "staked": 0.0, "profit": 0.0})
        d["bets"] += 1
        d["wins"] += 1 if outcome == 1 else 0
        d["staked"] += s
        d["profit"] += profit

    roi = (total_profit / total_staked * 100.0) if total_staked > 0 else 0.0
    market_summary = {
        mkt: {"bets": d["bets"], "wins": d["wins"],
              "win_rate_pct": round(100.0 * d["wins"] / d["bets"], 1) if d["bets"] else 0.0,
              "staked": round(d["staked"], 2), "profit": round(d["profit"], 2),
              "roi_pct": round(d["profit"] / d["staked"] * 100.0, 1) if d["staked"] > 0 else 0.0}
        for mkt, d in by_market.items()}

    return {
        "n_bets": n_bets, "n_wins": n_wins, "n_pushes_excluded": n_push,
        "win_rate_pct": round(100.0 * n_wins / n_bets, 1) if n_bets else 0.0,
        "staking": "fractional_kelly" if use_kelly else f"flat {stake}u",
        "total_staked": round(total_staked, 2), "total_profit": round(total_profit, 2),
        "roi_pct": round(roi, 2),
        "mean_clv_pct": round(sum(clvs) / len(clvs), 2) if clvs else None,
        "by_market": market_summary,
        "equity_curve": equity,
        "note": ("Real odds captured at tip time, never synthetic. Tips sent without odds are "
                 "excluded — there is no price to grade them against. Draw No Bet pushes on a "
                 "draw and is excluded rather than counted as a loss. If mean_clv_pct is "
                 "negative while roi_pct is positive, treat the ROI as variance, not edge."),
    }


def compute_calibration(days: Optional[int] = None, phase: Optional[str] = None,
                        min_n: int = 20) -> Dict[str, Any]:
    """
    Reads from `predictions`, which records every candidate evaluated, not from
    `tips`, which by construction only contains candidates that already cleared
    the threshold.
    """
    cutoff = int(time.time()) - days * 86400 if days else 0
    q = """
        SELECT p.market, p.suggestion, p.prob,
               r.final_goals_h, r.final_goals_a, r.btts_yes
        FROM predictions p JOIN match_results r ON r.match_id = p.match_id
        WHERE p.created_ts >= %s
    """
    params: List[Any] = [cutoff]
    if phase:
        q += " AND p.phase = %s"
        params.append(phase)
    with db_conn() as c:
        rows = c.execute(q, tuple(params)).fetchall()

    buckets = [(lo / 100.0, (lo + 5) / 100.0) for lo in range(30, 100, 5)]
    acc: Dict[Tuple[float, float], List[Tuple[float, int]]] = {b: [] for b in buckets}
    for (mkt, sugg, prob, gh, ga, btts) in rows:
        if prob is None:
            continue
        p = float(prob)
        outcome = _tip_outcome_for_result(sugg, {"final_goals_h": gh, "final_goals_a": ga, "btts_yes": btts})
        if outcome is None:
            continue
        for b in buckets:
            if b[0] <= p < b[1]:
                acc[b].append((p, outcome))
                break

    out = {}
    total_n = 0
    weighted_gap = 0.0
    for (lo, hi), arr in acc.items():
        if len(arr) < min_n:
            continue
        n = len(arr)
        expected = 100.0 * sum(p for p, _ in arr) / n
        actual = 100.0 * sum(y for _, y in arr) / n
        out[f"{lo*100:.0f}-{hi*100:.0f}%"] = {
            "n": n, "expected_win_rate_pct": round(expected, 1),
            "actual_win_rate_pct": round(actual, 1), "gap_pct": round(actual - expected, 1)}
        total_n += n
        weighted_gap += (actual - expected) * n

    return {"buckets": out,
            "n_graded": total_n,
            "overall_gap_pct": round(weighted_gap / total_n, 2) if total_n else None,
            "source": "predictions (all evaluated candidates, unfiltered)",
            "note": "A large negative gap means overconfidence in that band. Bands BELOW your "
                    "live threshold are visible too — which is where miscalibration starts."}


def compute_market_significance(days: Optional[int] = None, min_n: int = 50) -> Dict[str, Any]:
    """
    Benchmark is the DE-VIGGED fair probability stored on the tip, and the
    variance is Poisson-binomial because the bets have different probabilities.
    """
    cutoff = int(time.time()) - days * 86400 if days else 0
    with db_conn() as c:
        rows = c.execute("""
            SELECT t.market, t.suggestion, t.odds, t.fair_prob, t.is_prematch,
                   r.final_goals_h, r.final_goals_a, r.btts_yes
            FROM tips t JOIN match_results r ON r.match_id = t.match_id
            WHERE t.suggestion<>'HARVEST' AND t.odds IS NOT NULL AND t.created_ts >= %s
        """, (cutoff,)).fetchall()

    by: Dict[str, List[Tuple[float, int]]] = {}
    skipped_no_fair_pre = 0
    skipped_no_fair_live = 0
    for (mkt, sugg, odds, fair, is_prematch, gh, ga, btts) in rows:
        outcome = _tip_outcome_for_result(sugg, {"final_goals_h": gh, "final_goals_a": ga, "btts_yes": btts})
        if outcome is None:
            continue
        if fair is None:
            if is_prematch:
                skipped_no_fair_pre += 1
            else:
                skipped_no_fair_live += 1
            continue
        by.setdefault(mkt or "?", []).append((float(fair), outcome))

    out = {}
    for mkt, arr in by.items():
        n = len(arr)
        if n < min_n:
            continue
        wins = sum(y for _, y in arr)
        exp_wins = sum(p for p, _ in arr)
        var = sum(p * (1 - p) for p, _ in arr)
        se = math.sqrt(var) if var > 0 else 0.0
        z = (wins - exp_wins) / se if se > 0 else 0.0
        out[mkt] = {
            "n": n,
            "actual_win_rate_pct": round(100.0 * wins / n, 1),
            "fair_market_win_rate_pct": round(100.0 * exp_wins / n, 1),
            "z_score": round(z, 2),
            "p_value": round(2 * (1 - _norm_cdf(abs(z))), 4),
            "statistically_significant": bool(abs(z) > 1.96),
        }
    return {"by_market": out, "min_n_required": min_n,
            "tips_without_fair_price_skipped": {
                "pre": skipped_no_fair_pre, "live": skipped_no_fair_live,
                "total": skipped_no_fair_pre + skipped_no_fair_live,
            },
            "note": "Benchmark is the de-vigged fair probability, not 1/odds."}


def monte_carlo_bankroll(days: Optional[int], initial_bankroll: float, stake_pct: float,
                         simulations: int = 5000, ruin_pct: float = 20.0) -> Dict[str, Any]:
    """
    Bootstrap (with replacement) over real graded history, resampled at the
    FIXTURE level so correlated same-match bets move together. Ruin is a
    drawdown threshold, because with percentage staking a bankroll approaches
    zero asymptotically and never reaches it.
    """
    simulations = max(1, int(simulations))
    cutoff = int(time.time()) - days * 86400 if days else 0
    with db_conn() as c:
        rows = c.execute("""
            SELECT t.match_id, t.suggestion, t.odds,
                   r.final_goals_h, r.final_goals_a, r.btts_yes
            FROM tips t JOIN match_results r ON r.match_id = t.match_id
            WHERE t.suggestion<>'HARVEST' AND t.odds IS NOT NULL AND t.created_ts >= %s
        """, (cutoff,)).fetchall()

    by_match: Dict[int, List[Tuple[float, int]]] = {}
    n_bets = 0
    for (mid, sugg, odds, gh, ga, btts) in rows:
        o = _tip_outcome_for_result(sugg, {"final_goals_h": gh, "final_goals_a": ga, "btts_yes": btts})
        if o is None:
            continue
        by_match.setdefault(int(mid), []).append((float(odds), o))
        n_bets += 1

    groups = list(by_match.values())
    if len(groups) < 30:
        return {"error": f"only {len(groups)} graded fixtures with real odds — need at least 30 "
                         f"for a simulation that means anything"}

    ruin_level = initial_bankroll * (ruin_pct / 100.0)
    finals: List[float] = []
    ruin_count = 0
    max_dds: List[float] = []
    n_draws = len(groups)

    for _ in range(simulations):
        bankroll = initial_bankroll
        peak = bankroll
        max_dd = 0.0
        ruined = False
        for _i in range(n_draws):
            grp = random.choice(groups)
            for odds, outcome in grp:
                s = bankroll * (stake_pct / 100.0)
                bankroll += s * (odds - 1.0) if outcome == 1 else -s
                if bankroll > peak:
                    peak = bankroll
                if peak > 0:
                    max_dd = max(max_dd, (peak - bankroll) / peak * 100.0)
            if bankroll <= ruin_level:
                ruined = True
                break
        if ruined:
            ruin_count += 1
        finals.append(max(0.0, bankroll))
        max_dds.append(max_dd)

    finals.sort()
    n = len(finals)
    return {
        "initial_bankroll": initial_bankroll, "stake_pct": stake_pct,
        "simulations": simulations, "graded_fixtures_used": len(groups), "graded_bets_used": n_bets,
        "ruin_defined_as_bankroll_below_pct": ruin_pct,
        "probability_of_ruin_pct": round(100.0 * ruin_count / simulations, 2),
        "median_final_bankroll": round(finals[n // 2], 2),
        "worst_10pct_final_bankroll": round(finals[int(n * 0.1)], 2),
        "best_10pct_final_bankroll": round(finals[min(n - 1, int(n * 0.9))], 2),
        "avg_max_drawdown_pct": round(sum(max_dds) / len(max_dds), 1),
    }


def compute_league_breakdown(market: Optional[str] = None, days: Optional[int] = None,
                             min_n: int = 20) -> Dict[str, Any]:
    cutoff = int(time.time()) - days * 86400 if days else 0
    q = """
        SELECT t.league, t.market, t.suggestion, t.odds,
               r.final_goals_h, r.final_goals_a, r.btts_yes
        FROM tips t JOIN match_results r ON r.match_id = t.match_id
        WHERE t.suggestion<>'HARVEST' AND t.created_ts >= %s
    """
    params: List[Any] = [cutoff]
    if market:
        q += " AND t.market = %s"
        params.append(market)
    with db_conn() as c:
        rows = c.execute(q, tuple(params)).fetchall()

    by: Dict[str, Dict[str, float]] = {}
    for (league, mkt, sugg, odds, gh, ga, btts) in rows:
        outcome = _tip_outcome_for_result(sugg, {"final_goals_h": gh, "final_goals_a": ga, "btts_yes": btts})
        if outcome is None:
            continue
        d = by.setdefault(league or "?", {"n": 0, "wins": 0, "profit": 0.0, "staked": 0.0})
        d["n"] += 1
        d["wins"] += 1 if outcome == 1 else 0
        if odds:
            d["staked"] += 1.0
            d["profit"] += (float(odds) - 1.0) if outcome == 1 else -1.0

    out = {k: {"n": int(d["n"]), "wins": int(d["wins"]),
               "win_rate_pct": round(100.0 * d["wins"] / d["n"], 1),
               "roi_pct": round(d["profit"] / d["staked"] * 100.0, 1) if d["staked"] > 0 else None}
           for k, d in by.items() if d["n"] >= min_n}
    ranked = sorted(out.items(), key=lambda kv: kv[1]["win_rate_pct"])
    return {"market_filter": market or "ALL", "min_n": min_n, "by_league": out,
            "worst_5": ranked[:5], "best_5": ranked[-5:],
            "note": "Read-only. Nothing here changes a threshold automatically."}


def daily_accuracy_digest() -> Optional[str]:
    if not DAILY_ACCURACY_DIGEST_ENABLE:
        return None
    now_local = datetime.now(BERLIN_TZ)
    y0 = (now_local - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    y1 = y0 + timedelta(days=1)
    backfill_results_for_open_matches(400)
    capture_closing_lines(200)

    with db_conn() as c:
        rows = c.execute("""
            SELECT t.market, t.suggestion, r.final_goals_h, r.final_goals_a, r.btts_yes
            FROM tips t LEFT JOIN match_results r ON r.match_id=t.match_id
            WHERE t.created_ts >= %s AND t.created_ts < %s
              AND t.suggestion<>'HARVEST' AND t.sent_ok=1
        """, (int(y0.timestamp()), int(y1.timestamp()))).fetchall()

    total = graded = wins = pushes = 0
    by: Dict[str, Dict[str, int]] = {}
    for (mkt, sugg, gh, ga, btts) in rows:
        total += 1  # counted before the grading guard, so "Sent" != "Graded"
        if gh is None:
            continue
        out = _tip_outcome_for_result(sugg, {"final_goals_h": gh, "final_goals_a": ga, "btts_yes": btts})
        if out is None:
            pushes += 1
            continue
        graded += 1
        wins += 1 if out == 1 else 0
        d = by.setdefault(mkt or "?", {"graded": 0, "wins": 0})
        d["graded"] += 1
        d["wins"] += 1 if out == 1 else 0

    if total == 0:
        msg = "📊 <b>Daily Digest</b>\nNo tips sent yesterday."
    else:
        lines = ["📊 <b>Daily Digest</b> (yesterday, Berlin time)",
                 f"Sent: {total}  •  Graded: {graded}  •  Pushed: {pushes}  •  "
                 f"Pending: {total - graded - pushes}"]
        if graded:
            lines.append(f"Wins: {wins}  •  Accuracy: {100.0*wins/graded:.1f}%")
            for mk, st in sorted(by.items()):
                if st["graded"]:
                    lines.append(f"• {escape(mk)} — {st['wins']}/{st['graded']} "
                                 f"({100.0*st['wins']/st['graded']:.1f}%)")
        try:
            pnl = compute_pnl(days=1, stake=1.0)
            if pnl["n_bets"] > 0:
                sign = "+" if pnl["total_profit"] >= 0 else ""
                lines.append(f"💰 P&L (1u): {sign}{pnl['total_profit']:.2f}u  •  "
                             f"ROI: {pnl['roi_pct']:+.1f}%")
        except Exception:
            pass
        try:
            clv = compute_clv(days=7)
            ov = clv.get("overall")
            if ov:
                lines.append(f"📉 CLV (7d, prematch): {ov['mean_clv_pct']:+.2f}%  •  "
                             f"beat close {ov['beat_close_pct']:.0f}% of the time")
        except Exception:
            pass
        msg = "\n".join(lines)

    send_telegram(msg)
    return msg


# ───────── Training / tuning jobs ─────────
def auto_train_job():
    if not TRAIN_ENABLE:
        send_telegram("🤖 Training skipped: TRAIN_ENABLE=0")
        return
    send_telegram("🤖 Training started.")
    try:
        res = train_models() or {}
        if not res.get("ok"):
            reason = res.get("reason") or res.get("error") or "unknown"
            send_telegram(f"⚠️ Training finished: <b>SKIPPED</b>\nReason: {escape(str(reason))}")
            return
        _MODELS_CACHE.invalidate()
        _SETTINGS_CACHE.invalidate()
        trained = [k for k, v in (res.get("trained") or {}).items() if v]
        thr = res.get("thresholds") or {}
        lines = ["🤖 <b>Model training OK</b>"]
        if trained:
            lines.append("• Trained: " + ", ".join(sorted(trained)))
        if thr:
            lines.append("• Thresholds: " + "  |  ".join(
                f"{escape(str(k))}: {float(v):.1f}%" for k, v in sorted(thr.items())))
        ds = res.get("data_stats") or {}
        lines.append(f"• Rows: in-play {ds.get('inplay_rows', 0)} "
                     f"({ds.get('inplay_matches', 0)} matches), prematch {ds.get('prematch_rows', 0)}")
        send_telegram("\n".join(lines))
    except Exception as e:
        log.exception("[TRAIN] job failed: %s", e)
        send_telegram(f"❌ Training <b>FAILED</b>\n{escape(str(e))}")


def auto_tune_thresholds(days: int = 30) -> Dict[str, float]:
    """Reads from `predictions` rather than `tips`, so it can observe the region
    below the current threshold rather than ratcheting in one direction."""
    if not AUTO_TUNE_ENABLE:
        return {}
    cutoff = int(time.time()) - days * 86400
    with db_conn() as c:
        rows = c.execute("""
            SELECT p.market, p.suggestion, p.prob,
                   r.final_goals_h, r.final_goals_a, r.btts_yes
            FROM predictions p JOIN match_results r ON r.match_id = p.match_id
            WHERE p.created_ts >= %s AND p.prob IS NOT NULL
        """, (cutoff,)).fetchall()

    by: Dict[str, List[Tuple[float, int]]] = {}
    for (mk, sugg, prob, gh, ga, btts) in rows:
        out = _tip_outcome_for_result(sugg, {"final_goals_h": gh, "final_goals_a": ga, "btts_yes": btts})
        if out is None:
            continue
        by.setdefault(mk, []).append((float(prob), int(out)))

    tuned = {}
    for mk, arr in by.items():
        if len(arr) < THRESH_MIN_PREDICTIONS:
            continue
        if _is_threshold_locked(mk):
            log.warning("[AUTO-TUNE] %s is locked — skipping", mk)
            continue
        best = None
        for t_pct in [MIN_THRESH + i for i in range(int(MAX_THRESH - MIN_THRESH) + 1)]:
            t = t_pct / 100.0
            sel = [y for (p, y) in arr if p >= t]
            if len(sel) < THRESH_MIN_PREDICTIONS:
                continue
            prec = sum(sel) / len(sel)
            if prec >= TARGET_PRECISION:
                best = float(t_pct)
                break
        if best is None:
            continue
        set_setting(f"conf_threshold:{mk}", f"{best:.2f}")
        _SETTINGS_CACHE.invalidate(f"conf_threshold:{mk}")
        tuned[mk] = best
    if tuned:
        send_telegram("🔧 Auto-tune updated thresholds:\n" +
                      "\n".join(f"• {k}: {v:.1f}%" for k, v in tuned.items()))
    else:
        send_telegram("🔧 Auto-tune: no updates (insufficient data).")
    return tuned


def retry_unsent_tips(minutes: int = 120, limit: int = 200) -> int:
    """Both scan paths send inline; this only catches Telegram outages."""
    cutoff = int(time.time()) - minutes * 60
    with db_conn() as c:
        rows = c.execute(
            "SELECT match_id,league,home,away,market,suggestion,confidence,score_at_tip,minute,"
            "created_ts,odds,book,ev_pct,fair_prob,stake_units,is_prematch,kickoff_ts "
            "FROM tips WHERE sent_ok=0 AND created_ts >= %s ORDER BY created_ts ASC LIMIT %s",
            (cutoff, limit)).fetchall()

    retried = 0
    for (mid, league, home, away, market, sugg, conf, score, minute, cts, odds, book,
         ev_pct, fair, stake, is_pre, kickoff) in rows:
        kickoff_txt = "TBD"
        if kickoff:
            kickoff_txt = datetime.fromtimestamp(int(kickoff), TZ_UTC).astimezone(BERLIN_TZ).strftime("%H:%M")
        ok = send_telegram(_format_tip_message(
            home, away, league, int(minute or 0), score or "", sugg, float(conf), None,
            odds, book, ev_pct, fair, stake, kickoff_txt=kickoff_txt, prematch=bool(is_pre)))
        if ok:
            with db_conn() as c2:
                c2.execute("UPDATE tips SET sent_ok=1 WHERE match_id=%s AND created_ts=%s", (mid, cts))
            retried += 1
    if retried:
        log.info("[RETRY] resent %d", retried)
    return retried


# ───────── Scheduler ─────────
_SCHED: Optional[BackgroundScheduler] = None
_scheduler_started = False


def _run_with_pg_lock(lock_key: int, fn, *a, **k):
    try:
        with db_conn() as c:
            got = c.execute("SELECT pg_try_advisory_lock(%s)", (lock_key,)).fetchone()[0]
            if not got:
                log.info("[LOCK %s] busy; skipped.", lock_key)
                return None
            try:
                return fn(*a, **k)
            finally:
                c.execute("SELECT pg_advisory_unlock(%s)", (lock_key,))
    except Exception as e:
        log.exception("[LOCK %s] failed: %s", lock_key, e)
        return None


def _start_scheduler_once():
    global _scheduler_started, _SCHED
    if _scheduler_started or not RUN_SCHEDULER:
        return
    try:
        sched = BackgroundScheduler(timezone=TZ_UTC)
        sched.add_job(lambda: _run_with_pg_lock(1001, production_scan), "interval",
                      seconds=SCAN_INTERVAL_SEC, id="scan", max_instances=1, coalesce=True)
        sched.add_job(lambda: _run_with_pg_lock(1002, backfill_results_for_open_matches, 400),
                      "interval", minutes=BACKFILL_EVERY_MIN, id="backfill",
                      max_instances=1, coalesce=True)
        if PREMATCH_SCAN_ENABLE:
            sched.add_job(lambda: _run_with_pg_lock(1008, prematch_scan_save), "interval",
                          minutes=PREMATCH_SCAN_INTERVAL_MIN, id="prematch_scan",
                          max_instances=1, coalesce=True)
        if CLV_ENABLE:
            sched.add_job(lambda: _run_with_pg_lock(1009, capture_closing_lines, 200), "interval",
                          minutes=CLV_CAPTURE_EVERY_MIN, id="clv", max_instances=1, coalesce=True)
        if DAILY_ACCURACY_DIGEST_ENABLE:
            sched.add_job(lambda: _run_with_pg_lock(1003, daily_accuracy_digest),
                          CronTrigger(hour=DAILY_ACCURACY_HOUR, minute=DAILY_ACCURACY_MINUTE,
                                      timezone=BERLIN_TZ),
                          id="digest", max_instances=1, coalesce=True)
        if MOTD_PREDICT:
            sched.add_job(lambda: _run_with_pg_lock(1004, send_match_of_the_day),
                          CronTrigger(hour=MOTD_HOUR, minute=MOTD_MINUTE, timezone=BERLIN_TZ),
                          id="motd", max_instances=1, coalesce=True)
        if TRAIN_ENABLE:
            sched.add_job(lambda: _run_with_pg_lock(1005, auto_train_job),
                          CronTrigger(hour=TRAIN_HOUR_UTC, minute=TRAIN_MINUTE_UTC, timezone=TZ_UTC),
                          id="train", max_instances=1, coalesce=True)
        if AUTO_TUNE_ENABLE:
            sched.add_job(lambda: _run_with_pg_lock(1006, auto_tune_thresholds, 30),
                          CronTrigger(hour=4, minute=7, timezone=TZ_UTC),
                          id="auto_tune", max_instances=1, coalesce=True)
        sched.add_job(lambda: _run_with_pg_lock(1007, retry_unsent_tips, 120, 200), "interval",
                      minutes=10, id="retry", max_instances=1, coalesce=True)
        sched.start()
        _SCHED = sched
        _scheduler_started = True
        send_telegram("🚀 goalsniper started (market-aware pricing, CLV tracking on).")
        log.info("[SCHED] started (scan=%ss)", SCAN_INTERVAL_SEC)
    except Exception as e:
        log.exception("[SCHED] failed: %s", e)


def _shutdown(signum=None, frame=None):
    """
    Installing a SIGTERM handler REPLACES the default terminate behaviour, so a
    handler that only releases resources and returns leaves the process running
    with a dead connection pool until the platform escalates to SIGKILL. This
    stops the scheduler, releases resources, and exits.
    """
    log.info("[SHUTDOWN] signal %s received", signum)
    try:
        if _SCHED is not None:
            _SCHED.shutdown(wait=False)
    except Exception as e:
        log.warning("[SHUTDOWN] scheduler stop failed: %s", e)
    try:
        if POOL:
            POOL.closeall()
    except Exception as e:
        log.warning("[SHUTDOWN] pool close failed: %s", e)
    try:
        session.close()
    except Exception as e:
        log.warning("[SHUTDOWN] session close failed: %s", e)
    log.info("[SHUTDOWN] complete")
    sys.exit(0)


try:
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
except Exception as e:
    log.warning("[SHUTDOWN] could not register signal handlers: %s", e)


# ───────── Auth ─────────
def _require_admin():
    body = request.get_json(silent=True) if request.is_json else None
    key = (request.headers.get("X-API-Key") or request.args.get("key")
           or ((body or {}).get("key") if body else None))
    if not ADMIN_API_KEY or not key or not _safe_compare(key, ADMIN_API_KEY):
        abort(401)


def _arg_int(name: str, default=None):
    v = request.args.get(name)
    try:
        return int(v) if v not in (None, "") else default
    except Exception:
        return default


def _arg_float(name: str, default: float) -> float:
    try:
        return float(request.args.get(name, default))
    except Exception:
        return default


# ───────── HTTP ─────────
@app.route("/")
def root():
    return jsonify({"ok": True, "name": "goalsniper", "scheduler": RUN_SCHEDULER})


@app.route("/health")
def health():
    try:
        with db_conn() as c:
            n = c.execute("SELECT COUNT(*) FROM tips WHERE suggestion<>'HARVEST'").fetchone()[0]
        return jsonify({"ok": True, "db": "ok", "tips_count": int(n)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/init-db", methods=["POST"])
def http_init_db():
    _require_admin()
    init_db()
    return jsonify({"ok": True})


@app.route("/admin/scan", methods=["POST", "GET"])
def http_scan():
    _require_admin()
    s, l = production_scan()
    return jsonify({"ok": True, "saved": s, "live_seen": l})


@app.route("/admin/backfill-results", methods=["POST", "GET"])
def http_backfill():
    _require_admin()
    return jsonify({"ok": True, "updated": backfill_results_for_open_matches(400)})


@app.route("/admin/train", methods=["POST", "GET"])
def http_train():
    _require_admin()
    if not TRAIN_ENABLE:
        return jsonify({"ok": False, "reason": "training disabled"}), 400
    try:
        out = train_models()
        _MODELS_CACHE.invalidate()
        _SETTINGS_CACHE.invalidate()
        return jsonify({"ok": True, "result": out})
    except Exception as e:
        log.exception("train_models failed: %s", e)
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/admin/train-notify", methods=["POST", "GET"])
def http_train_notify():
    _require_admin()
    auto_train_job()
    return jsonify({"ok": True})


@app.route("/admin/digest", methods=["POST", "GET"])
def http_digest():
    _require_admin()
    return jsonify({"ok": True, "sent": bool(daily_accuracy_digest())})


@app.route("/admin/auto-tune", methods=["POST", "GET"])
def http_auto_tune():
    _require_admin()
    return jsonify({"ok": True, "tuned": auto_tune_thresholds(30)})


@app.route("/admin/retry-unsent", methods=["POST", "GET"])
def http_retry_unsent():
    _require_admin()
    return jsonify({"ok": True, "resent": retry_unsent_tips(120, 200)})


@app.route("/admin/prematch-scan", methods=["POST", "GET"])
def http_prematch_scan():
    _require_admin()
    return jsonify({"ok": True, "saved": int(prematch_scan_save())})


@app.route("/admin/motd", methods=["POST", "GET"])
def http_motd():
    _require_admin()
    return jsonify({"ok": bool(send_match_of_the_day())})


@app.route("/admin/capture-clv", methods=["POST", "GET"])
def http_capture_clv():
    _require_admin()
    return jsonify({"ok": True, "captured": capture_closing_lines(500)})


@app.route("/admin/backfill-prematch-history", methods=["POST", "GET"])
def http_backfill_prematch_history():
    """/admin/backfill-prematch-history?league=39&seasons=2023,2024,2025&key=..."""
    _require_admin()
    league_id = _arg_int("league", 0) or 0
    if not league_id:
        return jsonify({"ok": False, "error": "missing ?league=<API-Football league id>"}), 400
    try:
        seasons = [int(s.strip()) for s in request.args.get("seasons", "").split(",") if s.strip()]
    except Exception:
        seasons = []
    if not seasons:
        return jsonify({"ok": False, "error": "missing ?seasons=2023,2024,2025"}), 400
    return jsonify({"ok": True, "league": league_id, "seasons": seasons,
                    **backfill_historical_prematch(league_id, seasons)})


@app.route("/admin/leagues", methods=["GET"])
def http_leagues():
    _require_admin()
    params = {}
    if request.args.get("search", "").strip():
        params["search"] = request.args["search"].strip()
    if request.args.get("country", "").strip():
        params["country"] = request.args["country"].strip()
    if not params:
        return jsonify({"ok": False, "error": "provide ?search=<3+ chars> and/or ?country=<name>"}), 400
    js = _api_get(f"{BASE_URL}/leagues", params) or {}
    out = []
    for item in (js.get("response", []) if isinstance(js, dict) else []):
        lg = item.get("league") or {}
        out.append({"id": lg.get("id"), "name": lg.get("name"), "type": lg.get("type"),
                    "country": (item.get("country") or {}).get("name"),
                    "available_seasons": sorted(s.get("year") for s in (item.get("seasons") or [])
                                                if s.get("year"))})
    return jsonify({"ok": True, "count": len(out), "leagues": out})


@app.route("/admin/pnl", methods=["GET"])
def http_pnl():
    _require_admin()
    return jsonify({"ok": True, "pnl": compute_pnl(
        days=_arg_int("days"), stake=_arg_float("stake", 1.0),
        use_kelly=request.args.get("kelly") in ("1", "true", "yes"))})


@app.route("/admin/diagnostics/clv", methods=["GET"])
def http_clv():
    _require_admin()
    return jsonify({"ok": True, "clv": compute_clv(days=_arg_int("days"))})


@app.route("/admin/diagnostics/calibration", methods=["GET"])
def http_calibration():
    _require_admin()
    return jsonify({"ok": True, "calibration": compute_calibration(
        days=_arg_int("days"), phase=request.args.get("phase"),
        min_n=_arg_int("min_n", 20))})


@app.route("/admin/diagnostics/significance", methods=["GET"])
def http_significance():
    _require_admin()
    return jsonify({"ok": True, "significance": compute_market_significance(
        days=_arg_int("days"), min_n=_arg_int("min_n", 50))})


@app.route("/admin/diagnostics/monte-carlo", methods=["GET"])
def http_monte_carlo():
    _require_admin()
    return jsonify({"ok": True, "simulation": monte_carlo_bankroll(
        _arg_int("days"), _arg_float("bankroll", 1000.0), _arg_float("stake_pct", 2.0),
        _arg_int("simulations", 5000), _arg_float("ruin_pct", 20.0))})


@app.route("/admin/diagnostics/league-breakdown", methods=["GET"])
def http_league_breakdown():
    _require_admin()
    return jsonify({"ok": True, "breakdown": compute_league_breakdown(
        market=request.args.get("market"), days=_arg_int("days"), min_n=_arg_int("min_n", 20))})


@app.route("/admin/thresholds", methods=["GET"])
def http_thresholds():
    """
    Every market's live threshold in one view, with derived markets flagged.

    A derived market showing "SUPPRESSED (never verified)" means training has
    not written a threshold for it — it will not fire, which is correct until
    it has passed a holdout.
    """
    _require_admin()
    markets = ["BTTS", "1X2", "Double Chance", "Draw No Bet"] + \
              [f"Over/Under {_fmt_line(l)}" for l in OU_LINES]
    out = {}
    for phase_prefix in ("", "PRE "):
        for mk in markets:
            label = f"{phase_prefix}{mk}"
            raw = get_setting_cached(f"conf_threshold:{label}")
            effective = _get_market_threshold(label)
            out[label] = {
                "stored": float(raw) if raw is not None else None,
                "effective_pct": round(effective, 2),
                "derived_market": mk in DERIVED_MARKETS,
                "locked": _is_threshold_locked(label),
                "status": ("SUPPRESSED (never verified)" if raw is None and mk in DERIVED_MARKETS
                           else "suppressed" if effective >= MAX_THRESH
                           else "active" if raw is not None
                           else "default (untrained)"),
            }
    return jsonify({"ok": True, "max_thresh": MAX_THRESH, "thresholds": out})


@app.route("/admin/status", methods=["GET"])
def http_status():
    _require_admin()
    with db_conn() as c:
        n_tip_snap = c.execute("SELECT COUNT(*) FROM tip_snapshots").fetchone()[0]
        n_snap_matches = c.execute("SELECT COUNT(DISTINCT match_id) FROM tip_snapshots").fetchone()[0]
        n_pre_snap = c.execute("SELECT COUNT(*) FROM prematch_snapshots").fetchone()[0]
        n_results = c.execute("SELECT COUNT(*) FROM match_results").fetchone()[0]
        n_tips = c.execute("SELECT COUNT(*) FROM tips WHERE suggestion<>'HARVEST'").fetchone()[0]
        n_unsent = c.execute("SELECT COUNT(*) FROM tips WHERE sent_ok=0").fetchone()[0]
        n_preds = c.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        n_clv = c.execute("SELECT COUNT(*) FROM tips WHERE clv_pct IS NOT NULL").fetchone()[0]
        n_has_league = c.execute("SELECT COUNT(*) FROM match_results WHERE league_id IS NOT NULL").fetchone()[0]
        n_leagues = c.execute("SELECT COUNT(DISTINCT league_id) FROM match_results "
                              "WHERE league_id IS NOT NULL").fetchone()[0]
        n_rated = c.execute("SELECT COUNT(*) FROM team_ratings").fetchone()[0]
    metrics_raw = get_setting_cached("model_metrics_latest")
    try:
        metrics = json.loads(metrics_raw) if metrics_raw else None
    except Exception:
        metrics = None
    snap_ratio = (float(n_tip_snap) / n_snap_matches) if n_snap_matches else 0.0
    return jsonify({
        "ok": True,
        "harvest": {"tip_snapshots": int(n_tip_snap), "distinct_matches_snapshotted": int(n_snap_matches),
                    "snapshots_per_match": round(snap_ratio, 2),
                    "prematch_snapshots": int(n_pre_snap), "match_results_resolved": int(n_results),
                    "match_results_with_league_id": int(n_has_league),
                    "distinct_leagues": int(n_leagues), "teams_rated": int(n_rated)},
        "tips": {"total": int(n_tips), "unsent": int(n_unsent), "with_closing_price": int(n_clv)},
        "predictions_logged": int(n_preds),
        "dashboard_enabled": DASHBOARD_ENABLED,
        "last_training_run": metrics,
        "api_usage": _api_call_stats_snapshot(),
    })


@app.route("/settings/<path:key>", methods=["GET", "POST"])
def http_settings(key: str):
    _require_admin()
    if request.method == "GET":
        qval = request.args.get("value")
        if qval is not None:
            set_setting(key, str(qval))
            _SETTINGS_CACHE.invalidate(key)
            invalidate_model_caches_for_key(key)
            return jsonify({"ok": True, "key": key, "value": str(qval), "wrote_via": "GET ?value="})
        return jsonify({"ok": True, "key": key, "value": get_setting_cached(key)})
    val = (request.get_json(silent=True) or {}).get("value")
    if val is None:
        abort(400)
    set_setting(key, str(val))
    _SETTINGS_CACHE.invalidate(key)
    invalidate_model_caches_for_key(key)
    return jsonify({"ok": True})


# ───────── Web dashboard ─────────
# Read-only browser view. Auth is session-based: the admin key is checked once
# at /dashboard/login and a signed HttpOnly cookie is set, so the raw key is
# never stored client-side or re-sent on every page load the way a bookmarked
# ?key=... URL would be. Nothing here can train, scan, or change settings.
DASHBOARD_REFRESH_SEC = int(os.getenv("DASHBOARD_REFRESH_SEC", "60"))
LOGIN_MAX_ATTEMPTS = int(os.getenv("LOGIN_MAX_ATTEMPTS", "8"))
LOGIN_WINDOW_SEC = int(os.getenv("LOGIN_WINDOW_SEC", "900"))
_login_attempts: Dict[str, List[float]] = defaultdict(list)
_login_lock = threading.Lock()


def _login_rate_limited(ip: str) -> bool:
    """
    Throttle /dashboard/login. Without this the endpoint is an unthrottled
    oracle for the admin key.

    Per-process, so with N workers the effective limit is N x
    LOGIN_MAX_ATTEMPTS. That is still a hard ceiling on guessing rate and needs
    no shared state; a long random ADMIN_API_KEY remains the real defence.
    """
    now = time.time()
    with _login_lock:
        hits = [t for t in _login_attempts[ip] if now - t < LOGIN_WINDOW_SEC]
        _login_attempts[ip] = hits
        if len(_login_attempts) > 10000:      # bound memory
            _login_attempts.clear()
        return len(hits) >= LOGIN_MAX_ATTEMPTS


def _login_record_failure(ip: str) -> None:
    with _login_lock:
        _login_attempts[ip].append(time.time())


def _dashboard_authed() -> bool:
    return DASHBOARD_ENABLED and bool(flask_session.get("dash_authed"))


def _dashboard_unavailable():
    return jsonify({
        "ok": False,
        "error": "dashboard disabled",
        "reason": "SECRET_KEY is not set. Without a fixed signing key each worker process "
                  "signs session cookies differently, so logins fail at random.",
        "fix": "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\" "
               "and set it as the SECRET_KEY environment variable.",
    }), 503


@app.route("/dashboard/login", methods=["GET", "POST"])
def dashboard_login():
    if not DASHBOARD_ENABLED:
        return _dashboard_unavailable()
    ip = request.headers.get("X-Forwarded-For", request.remote_addr or "?").split(",")[0].strip()
    if request.method == "POST":
        if _login_rate_limited(ip):
            log.warning("[DASHBOARD] login rate-limited for %s", ip)
            return render_template("dashboard_login.html",
                                   error="Too many attempts. Try again later."), 429
        if ADMIN_API_KEY and _safe_compare(request.form.get("key", ""), ADMIN_API_KEY):
            flask_session.clear()
            flask_session["dash_authed"] = True
            flask_session.permanent = True
            return redirect(url_for("dashboard"))
        _login_record_failure(ip)
        return render_template("dashboard_login.html", error="Incorrect key."), 401
    if _dashboard_authed():
        return redirect(url_for("dashboard"))
    return render_template("dashboard_login.html", error=None)


@app.route("/dashboard/logout", methods=["GET", "POST"])
def dashboard_logout():
    flask_session.clear()
    return redirect(url_for("dashboard_login"))


@app.route("/dashboard")
def dashboard():
    if not DASHBOARD_ENABLED:
        return _dashboard_unavailable()
    if not _dashboard_authed():
        return redirect(url_for("dashboard_login"))
    return render_template("dashboard.html", refresh_sec=DASHBOARD_REFRESH_SEC)


@app.route("/dashboard/data")
def dashboard_data():
    if not DASHBOARD_ENABLED:
        return _dashboard_unavailable()
    if not _dashboard_authed():
        abort(401)
    limit = max(1, min(200, _arg_int("limit", 50) or 50))
    days = _arg_int("days")
    with db_conn() as c:
        rows = c.execute(
            "SELECT match_id,league,home,away,market,suggestion,confidence,"
            "score_at_tip,minute,created_ts,odds,book,ev_pct,fair_prob,stake_units,"
            "clv_pct,is_prematch,sent_ok "
            "FROM tips WHERE suggestion<>'HARVEST' ORDER BY created_ts DESC LIMIT %s", (limit,)
        ).fetchall()
    keys = ["match_id", "league", "home", "away", "market", "suggestion", "confidence",
            "score_at_tip", "minute", "created_ts", "odds", "book", "ev_pct", "fair_prob",
            "stake_units", "clv_pct", "is_prematch", "sent_ok"]
    tips = [dict(zip(keys, r)) for r in rows]
    try:
        pnl = compute_pnl(days=days, stake=1.0)
    except Exception as e:
        log.warning("[DASHBOARD] pnl computation failed: %s", e)
        pnl = {"error": str(e)}
    return jsonify({"ok": True, "tips": tips, "pnl": pnl, "server_ts": int(time.time())})


@app.route("/dashboard/live")
def dashboard_live():
    """
    Every currently-live match with usable stats, and the FULL set of market
    probabilities production_scan() computed for it - not just whichever
    candidate cleared the tipping threshold and price gate. Backed by an
    in-memory snapshot refreshed on every scan (see _set_live_snapshot),
    so this costs zero extra API-Football requests.
    """
    if not DASHBOARD_ENABLED:
        return _dashboard_unavailable()
    if not _dashboard_authed():
        abort(401)
    return jsonify({"ok": True, **_get_live_snapshot(), "server_ts": int(time.time())})


@app.route("/dashboard/live/refresh", methods=["POST"])
def dashboard_live_refresh():
    """
    On-demand version of the same snapshot: scores whatever is live RIGHT NOW
    instead of waiting for the next scheduled scan (up to SCAN_INTERVAL_SEC
    away). Costs real API-Football requests each time it's called - this is
    for a human clicking "refresh now", not something to poll automatically.
    """
    if not DASHBOARD_ENABLED:
        return _dashboard_unavailable()
    if not _dashboard_authed():
        abort(401)
    matches, live_seen = score_live_matches_now()
    _set_live_snapshot(matches)
    return jsonify({"ok": True, **_get_live_snapshot(), "live_seen": live_seen,
                    "server_ts": int(time.time())})


@app.route("/tips/latest")
def http_latest():
    _require_admin()
    limit = max(1, min(500, _arg_int("limit", 50) or 50))
    with db_conn() as c:
        rows = c.execute(
            "SELECT match_id,league,home,away,market,suggestion,confidence,confidence_raw,"
            "score_at_tip,minute,created_ts,odds,book,ev_pct,fair_prob,stake_units,clv_pct,is_prematch "
            "FROM tips WHERE suggestion<>'HARVEST' ORDER BY created_ts DESC LIMIT %s", (limit,)).fetchall()
    keys = ["match_id", "league", "home", "away", "market", "suggestion", "confidence",
            "confidence_raw", "score_at_tip", "minute", "created_ts", "odds", "book",
            "ev_pct", "fair_prob", "stake_units", "clv_pct", "is_prematch"]
    return jsonify({"ok": True, "tips": [dict(zip(keys, r)) for r in rows]})


@app.route("/telegram/webhook/<secret>", methods=["POST"])
def telegram_webhook(secret: str):
    if not WEBHOOK_SECRET or not _safe_compare(WEBHOOK_SECRET, secret):
        abort(403)
    update = request.get_json(silent=True) or {}
    try:
        msg = (update.get("message") or {}).get("text") or ""
        if msg.startswith("/start"):
            send_telegram("👋 goalsniper is online.")
        elif msg.startswith("/digest"):
            daily_accuracy_digest()
        elif msg.startswith("/motd"):
            send_match_of_the_day()
        elif msg.startswith("/clv"):
            send_telegram(f"<pre>{escape(json.dumps(compute_clv(days=30), indent=2)[:3500])}</pre>")
        elif msg.startswith("/scan"):
            parts = msg.split()
            if len(parts) > 1 and ADMIN_API_KEY and _safe_compare(parts[1], ADMIN_API_KEY):
                s, l = production_scan()
                send_telegram(f"🔁 Scan done. Saved: {s}, Live seen: {l}")
            else:
                send_telegram("🔒 Admin key required.")
    except Exception as e:
        log.warning("telegram webhook parse error: %s", e)
    return jsonify({"ok": True})


# ───────── Boot ─────────
def _on_boot():
    _init_pool()
    init_db()
    set_setting("boot_ts", str(int(time.time())))


# Order matters: the schema must exist before any scheduled job can run.
_on_boot()
_start_scheduler_once()

if __name__ == "__main__":
    app.run(host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8080")))
