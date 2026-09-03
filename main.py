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
    ELO_DEFAULT,
    DEFAULT_LEAGUE_RATES, MARKET_PROBABILITY_TOTAL, NEUTRAL_MARKET_PRIORS,
    MARKET_ANCHOR, anchor_logit,
    ODDS_TRUSTED_FROM_TS, RAW_INPLAY_KEYS,
    assemble_prematch_features, build_inplay_features, derive_dc_dnb,
    devig, elo_update, ev as _ev, fixture_ts as _fixture_ts, kelly_fraction,
    enforce_ou_monotonicity, venue_form_stats,
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

# ───────── Minimum information before a live tip ─────────
# TIP_MIN_MINUTE governs when a fixture becomes ELIGIBLE - it also gates
# harvesting, so lowering it feeds the training set. These three govern when a
# fixture is worth BETTING, which is a stricter question, and they deliberately
# do not touch data collection.
#
# 1. Do not tip a game state the model was never trained on. Snapshots are
#    harvested from TRAIN_MIN_MINUTE onward, so scoring at minute 8 asks the
#    model to extrapolate outside its own training distribution. Defaulting to
#    TRAIN_MIN_MINUTE ties the two together rather than picking a number.
LIVE_TIP_MIN_MINUTE = int(os.getenv("LIVE_TIP_MIN_MINUTE", str(TRAIN_MIN_MINUTE)))
# 2. A match with no shot recorded at all by this minute is a dead statistics
#    feed, not a cagey game. Real football produces a shot long before this.
SHOT_DATA_MIN_MINUTE = int(os.getenv("SHOT_DATA_MIN_MINUTE", "25"))
# The top of the training range. load_inplay_data() harvests minute 15-90, so
# a fixture scored at minute 95 is being asked of a model that never saw one.
LIVE_TIP_MAX_MINUTE = int(os.getenv("LIVE_TIP_MAX_MINUTE", "90"))
# 3. The xG channel specifically. See _xg_feed_is_dead() for why a zero here
#    is worse than a missing value.
REQUIRE_XG_FEED = _env_flag("REQUIRE_XG_FEED", "1")
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
# The in-play feed is ONE aggregated source, not a panel of books, so it can
# never reach MIN_BOOKS_FOR_FAIR - live candidates would sit at
# too_few_books forever. This is a separate knob rather than a lower global
# value because prematch genuinely does have multiple books and should keep
# demanding a consensus. Defaults to the strict value so nothing loosens by
# itself: set it to 1 to accept the in-play feed's own de-vigged price, in
# full knowledge that a single source's overround is not a consensus.
MIN_BOOKS_FOR_FAIR_LIVE = int(os.getenv("MIN_BOOKS_FOR_FAIR_LIVE",
                                        str(MIN_BOOKS_FOR_FAIR)))
# ───────── Market width (overround) ─────────
# The overround is what the book's prices sum to above 100%: quote Yes at 1.53
# and No at 2.28 and the implied probabilities total 109.3%, so the overround
# is 9.3%. devig() strips it by scaling every selection down PROPORTIONALLY,
# and that is where the danger sits.
#
# Books do not load vig proportionally. Favourite-longshot bias means the
# longshot side carries more of the margin than its share of probability, so
# proportional de-vig takes too much off the FAVOURITE and leaves it with a
# fair probability that is too low - i.e. a fair price that looks too long,
# i.e. an edge on the favourite side that is partly de-vig method error rather
# than a market mistake. The wider the overround, the larger that error, and
# the more of any measured "edge over fair" is our own arithmetic.
#
# So a wide market is not merely expensive, it makes the fair price we gate on
# untrustworthy. Above the cap the candidate is refused rather than priced.
# Three-way markets carry a mechanically larger margin than two-way ones (the
# book prices one more outcome), hence the separate cap - a single number would
# either wave 1X2 through or strangle BTTS.
#
# These defaults reject clearly untrustworthy quotes rather than merely
# expensive ones; the in-play feed has been running around 9% on two-way
# markets, so expect this to bind rarely at first. The number is on every tip
# message now - tighten it once you can see the distribution you actually get.
# Set either to 0 to disable that cap.
MAX_OVERROUND_BPS = int(os.getenv("MAX_OVERROUND_BPS", "1200"))
MAX_OVERROUND_BPS_3WAY = int(os.getenv("MAX_OVERROUND_BPS_3WAY", "1800"))

ODDS_BOOKMAKER_ID = os.getenv("ODDS_BOOKMAKER_ID")
ALLOW_TIPS_WITHOUT_ODDS = _env_flag("ALLOW_TIPS_WITHOUT_ODDS", "0")

BANKROLL_UNITS = float(os.getenv("BANKROLL_UNITS", "100"))
KELLY_FRACTION = float(os.getenv("KELLY_FRACTION", "0.25"))
MAX_STAKE_PCT = float(os.getenv("MAX_STAKE_PCT", "2.0"))

CLV_ENABLE = _env_flag("CLV_ENABLE", "1")
CLV_CAPTURE_EVERY_MIN = int(os.getenv("CLV_CAPTURE_EVERY_MIN", "5"))
# How long before kickoff to start treating a fixture as "closing" - the
# prematch market is still open in this window, unlike after kickoff. See
# capture_closing_lines() for why this must be BEFORE kickoff, not after.
CLV_CAPTURE_LEAD_MIN = int(os.getenv("CLV_CAPTURE_LEAD_MIN", "15"))
# Below this many captured closing prices, a CLV figure is noise dressed as a
# verdict - "beat close 0% of the time" means nothing at n=3.
CLV_MIN_SAMPLE_FOR_VERDICT = int(os.getenv("CLV_MIN_SAMPLE_FOR_VERDICT", "100"))
# Holdout |predicted - actual|, in percentage points, past which a head's
# probabilities are called out. EV is computed directly from those
# probabilities, so the gap propagates into every EV the gate evaluates.
CALIBRATION_GAP_WARN_PP = float(os.getenv("CALIBRATION_GAP_WARN_PP", "3.0"))

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
# Statuses in which a FULL-TIME market is still open. Every market Goalsniper
# prices settles on 90 minutes plus stoppage, so once a tie reaches extra time
# (ET), the break before it (BT) or a shootout (P), the bet has already been
# decided - scoring those fixtures is pricing a settled question, and doing it
# off a scoreline that now includes extra-time goals. Harvesting still sees the
# wider set; this governs betting.
FULLTIME_MARKET_OPEN_STATUSES = {"1H", "HT", "2H"}
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
        """
        A connection handed out here but never bound into a completed __enter__
        is invisible to __exit__, because __exit__ does not run when __enter__
        raises. It therefore has to be given back inside this loop or the pool
        loses that slot permanently.

        This matters because getconn() happily returns a connection the server
        has since closed (Postgres restart, idle timeout, network blip); the
        failure then surfaces on `.autocommit =` or `.cursor()` as
        InterfaceError/OperationalError, neither of which the retry previously
        caught. So a dead connection both failed its caller AND leaked a slot,
        and ~DB_POOL_MAX of them left the app unable to reach the database at
        all until it was restarted.
        """
        last_err = None
        for attempt in range(5):
            conn = None
            try:
                conn = self.pool.getconn()
                conn.autocommit = True
                self.cur = conn.cursor()
                self.conn = conn
                return self
            except (psycopg2.pool.PoolError, psycopg2.OperationalError,
                    psycopg2.InterfaceError) as e:
                last_err = e
                if conn is not None:
                    # close=True: this connection is suspect, don't recycle it.
                    try:
                        self.pool.putconn(conn, close=True)
                    except Exception:
                        try:
                            conn.close()
                        except Exception:
                            pass
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
            # Seconds before kickoff at which the closing price was taken. The
            # line sharpens as kickoff approaches, so a price captured 15
            # minutes out is a weaker benchmark than one taken at the bell -
            # and being measured against a weaker benchmark FLATTERS CLV, the
            # one number meant to decide whether the edge is real. Recorded so
            # the series can be judged, not just read.
            "ALTER TABLE tips ADD COLUMN IF NOT EXISTS closing_lead_sec INTEGER",
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
_api_call_stats = {"day": None, "total": 0, "rate_limited": 0, "api_errors": 0,
                   "by_endpoint": {}}


def _endpoint_of(url: str) -> str:
    """Coarse label for quota accounting: the API-Football resource being hit."""
    u = (url or "").split("?", 1)[0].rstrip("/")
    for tail in ("/fixtures/statistics", "/fixtures/events", "/fixtures/lineups",
                 "/odds/live", "/odds", "/fixtures", "/teams", "/standings"):
        if u.endswith(tail):
            return tail
    return u.rsplit("/", 1)[-1] or "other"


def _track_api_call(status_code: Optional[int], api_error: bool = False,
                    url: str = "") -> Dict[str, Any]:
    today = datetime.now(TZ_UTC).strftime("%Y-%m-%d")
    with _api_call_lock:
        if _api_call_stats["day"] != today:
            _api_call_stats.update(day=today, total=0, rate_limited=0, api_errors=0,
                                   by_endpoint={})
        _api_call_stats["total"] += 1
        if url:
            # WHERE the quota goes, not just how much of it. The live scan and
            # the prematch scan compete for one budget, and only one of them
            # produces a closing line to measure against - so knowing the split
            # is what makes the trade-off decidable instead of guessed at.
            ep = _endpoint_of(url)
            book = _api_call_stats.setdefault("by_endpoint", {})
            book[ep] = book.get(ep, 0) + 1
        if status_code == 429:
            _api_call_stats["rate_limited"] += 1
        if api_error:
            _api_call_stats["api_errors"] = _api_call_stats.get("api_errors", 0) + 1
        return dict(_api_call_stats)


def api_response_error(js: Any) -> Optional[str]:
    """
    The error API-Football reports INSIDE a 200 response, or None.

    This is the trap in this API: quota exhaustion, an expired or insufficient
    plan, and a bad key all come back as HTTP 200 with an empty `response` and
    the reason in `errors`:

        {"errors": {"requests": "You have reached the request limit for the day"},
         "results": 0, "response": []}

    Read only for `response`, that is indistinguishable from "no fixtures are
    live" or "nothing is priced" - the system keeps running, quietly blind, and
    every downstream diagnostic points somewhere else. This is the same shape
    of failure as the in-play odds parser reading only `bookmakers`, which cost
    a long hunt that one line of logging would have ended.

    `errors` is a LIST (usually empty) when there is nothing wrong and a DICT
    when there is, so both shapes have to be handled.
    """
    if not isinstance(js, dict):
        return None
    err = js.get("errors")
    if isinstance(err, dict) and err:
        return "; ".join(f"{k}: {v}" for k, v in err.items())
    if isinstance(err, list) and err:
        return "; ".join(str(e) for e in err)
    if isinstance(err, str) and err.strip():
        return err.strip()
    return None


def _api_call_stats_snapshot() -> Dict[str, Any]:
    with _api_call_lock:
        snap = dict(_api_call_stats)
    ep = dict(snap.get("by_endpoint") or {})
    snap["by_endpoint"] = dict(sorted(ep.items(), key=lambda kv: kv[1], reverse=True))
    total = snap.get("total") or 0
    if total and ep:
        snap["by_endpoint_pct"] = {k: round(100.0 * v / total, 1)
                                   for k, v in snap["by_endpoint"].items()}
        # The live branch cannot be measured against a closing line; the
        # prematch branch can. Spending most of one budget on the unmeasurable
        # half is a decision worth making on purpose rather than by default.
        live_side = sum(v for k, v in ep.items()
                        if k in ("/fixtures/statistics", "/fixtures/events", "/odds/live"))
        snap["live_scan_share_pct"] = round(100.0 * live_side / total, 1)
    return snap


# ───────── Per-minute rate limit ─────────
#
# API-Football reports the PER-MINUTE limit the same way it reports quota
# exhaustion: HTTP 200, empty `response`, and the reason in `errors`. The
# urllib3 Retry mounted on `session` keys off HTTP STATUS CODES
# (status_forcelist=[429, ...]), so it never sees this and never backs off.
#
# Nothing else throttled either, and the callers are deliberately concurrent —
# fetch_live_matches() hydrates 8 fixtures at a time, each spawning 2 more
# requests, and the prematch scan runs 8 fixtures x 3 requests. So the first
# refusal was followed immediately by the rest of the burst, every one of them
# spending a request to be refused again. Production logs show 8 such refusals
# inside 140ms and 20 within a single scan.
#
# A refusal means the minute window is full, and the only useful response is to
# stop asking until it rolls over. This records when that happened; _api_get()
# refuses locally (spending nothing) until the window clears.
_RATE_LIMIT_COOLDOWN_SEC = float(os.getenv("API_RATE_LIMIT_COOLDOWN_SEC", "60"))
_rate_limit_until = 0.0

# Matched against the message API-Football puts in `errors`. Kept narrow so a
# genuine account fault (expired plan, bad key) is NOT silently swallowed as a
# throttle — those need to stay loud and are not fixed by waiting.
_RATE_LIMIT_MARKERS = ("requests per minute", "per minute of your subscription",
                       "too many requests", "rate limit")


def _note_rate_limit_if_present(problem: str) -> None:
    """Start a cooldown if `problem` is the per-minute limit, else do nothing."""
    global _rate_limit_until
    msg = (problem or "").lower()
    if not any(m in msg for m in _RATE_LIMIT_MARKERS):
        return
    with _api_call_lock:
        was_cooling = _rate_limit_until > time.time()
        _rate_limit_until = time.time() + _RATE_LIMIT_COOLDOWN_SEC
    if not was_cooling:
        log.warning("[API] per-minute rate limit hit — pausing outbound API calls "
                    "for %.0fs instead of spending requests to be refused again.",
                    _RATE_LIMIT_COOLDOWN_SEC)


def _rate_limit_cooling_down() -> bool:
    with _api_call_lock:
        return _rate_limit_until > time.time()


def _api_get(url: str, params: dict, timeout: int = 15):
    if not API_KEY:
        return None
    # A per-minute limit already reported by the API is still in force: spending
    # another request to be told so again is what turns one refusal into the
    # twenty-in-a-row bursts seen in production. Refuse locally instead.
    if _rate_limit_cooling_down():
        return None
    try:
        r = session.get(url, headers=HEADERS, params=params, timeout=timeout)
        if r.ok:
            js = r.json()
            problem = api_response_error(js)
            if problem:
                # Treated as a FAILED call, not as an empty result, so callers
                # cannot cache it as "nothing was available". Logged at warning
                # because it is almost always an account-level fault that stops
                # the whole system until someone acts on it.
                _note_rate_limit_if_present(problem)
                stats = _track_api_call(None, api_error=True, url=url)
                log.warning("[API] %s returned an error inside a 200 response: %s "
                            "(today: %d calls, %d such errors). No data was received — "
                            "this is not 'nothing was live'.",
                            url, problem, stats["total"], stats.get("api_errors", 0))
                return None
            _track_api_call(None, url=url)
            return js
        stats = _track_api_call(r.status_code, url=url)
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
    js = _api_get(f"{FOOTBALL_API_URL}/statistics", {"fixture": fid})
    if js is None:
        # The call FAILED (rate limit, network, or an error reported inside a
        # 200). Not cached, for the reason fetch_odds() already states: caching
        # [] here pins "this match has no statistics" for the whole 90s TTL, and
        # every feature derived from stats — xg, shots, possession, corners —
        # scores as 0.0 for a match that is in fact being played. That is not a
        # neutral input: against a market anchor it reads as a confident
        # deviation and produces a tip from data that was never received.
        return []
    out = js.get("response", []) if isinstance(js, dict) else []
    STATS_CACHE.set(fid, out)
    return out


def fetch_match_events(fid: int) -> list:
    cached = EVENTS_CACHE.get(fid, _MISS)
    if cached is not _MISS:
        return cached
    js = _api_get(f"{FOOTBALL_API_URL}/events", {"fixture": fid})
    if js is None:
        return []  # failed, not empty — see fetch_match_stats()
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


# Only used when match_results holds too little to say anything about a
# league - the long-run cross-league split of home wins / away wins. Kept
# here rather than in feature_spec because nothing trains on it: it is a
# presentation baseline for the dashboard's form cards, not a model input.
DEFAULT_VENUE_RATES: Dict[str, float] = {"home_win": 0.45, "away_win": 0.29}


def _global_venue_rates() -> Dict[str, float]:
    cached = _LEAGUE_RATE_CACHE.get("V__GLOBAL__", _MISS)
    if cached is not _MISS:
        return cached
    with db_conn() as c:
        row = c.execute("""
            SELECT AVG(CASE WHEN final_goals_h>final_goals_a THEN 1.0 ELSE 0.0 END)::float,
                   AVG(CASE WHEN final_goals_a>final_goals_h THEN 1.0 ELSE 0.0 END)::float,
                   COUNT(*)::bigint
            FROM match_results""").fetchone()
    n = int((row[2] if row else 0) or 0)
    out = {"home_win": float(row[0]) if n and row[0] is not None else DEFAULT_VENUE_RATES["home_win"],
           "away_win": float(row[1]) if n and row[1] is not None else DEFAULT_VENUE_RATES["away_win"],
           "n": n}
    _LEAGUE_RATE_CACHE.set("V__GLOBAL__", out)
    return out


def get_league_venue_rates(league_id: Optional[int]) -> Dict[str, float]:
    """
    How often the home side and the away side actually win in this league -
    the baseline a team's own venue form is judged against ("above the
    league's usual"). Same shape and same thin-sample fallback as
    get_league_rates(): under LEAGUE_RATE_MIN_N finished matches the league
    tells us nothing, so the global split is used instead.
    """
    if not league_id:
        return _global_venue_rates()
    key = f"VL{league_id}"
    cached = _LEAGUE_RATE_CACHE.get(key, _MISS)
    if cached is not _MISS:
        return cached
    with db_conn() as c:
        row = c.execute("""
            SELECT AVG(CASE WHEN final_goals_h>final_goals_a THEN 1.0 ELSE 0.0 END)::float,
                   AVG(CASE WHEN final_goals_a>final_goals_h THEN 1.0 ELSE 0.0 END)::float,
                   COUNT(*)::bigint
            FROM match_results WHERE league_id=%s""", (league_id,)).fetchone()
    n = int((row[2] if row else 0) or 0)
    out = _global_venue_rates() if n < LEAGUE_RATE_MIN_N else {
        "home_win": float(row[0]) if row[0] is not None else DEFAULT_VENUE_RATES["home_win"],
        "away_win": float(row[1]) if row[1] is not None else DEFAULT_VENUE_RATES["away_win"],
        "n": n}
    _LEAGUE_RATE_CACHE.set(key, out)
    return out


# ───────── Raw in-play extraction ─────────
def _possession(sh: Dict[str, Any], sa: Dict[str, Any]) -> Dict[str, float]:
    ph, pa = _num(sh.get("Ball Possession", 0)), _num(sa.get("Ball Possession", 0))
    if ph <= 0 and pa <= 0:
        return {"pos_h": 50.0, "pos_a": 50.0}
    # One side quoted is enough: the other is its complement.
    if ph <= 0:
        ph = max(0.0, 100.0 - pa)
    elif pa <= 0:
        pa = max(0.0, 100.0 - ph)
    return {"pos_h": ph, "pos_a": pa}


def _num(v) -> float:
    try:
        if isinstance(v, str) and v.strip().endswith("%"):
            return float(v.strip()[:-1])
        return float(v or 0)
    except Exception:
        return 0.0


def _side_of(team: Any, home_id: int, away_id: int, home: str, away: str) -> Optional[str]:
    """
    Which side an API team object refers to: "h", "a", or None.

    Matches on ID first. /fixtures, /fixtures/statistics and /fixtures/events
    all carry team.id, and it is the field that cannot drift - a name arrives
    as free text and differs between endpoints often enough to matter
    (punctuation, accents, "FC" prefixes, a mid-season rename cached on one
    endpoint and not the other). A name mismatch does not raise: it silently
    yields an EMPTY statistics dict for that side, which then reads as a
    genuine 0 for every shot, corner and card the team has. Name matching is
    kept only as a fallback for feeds that omit the id.
    """
    t = team or {}
    tid = t.get("id")
    if tid is not None:
        try:
            tid = int(tid)
        except (TypeError, ValueError):
            tid = None
        if tid is not None and (tid == home_id or tid == away_id):
            return "h" if tid == home_id else "a"
        if tid is not None:
            return None
    name = _txt(t.get("name"))
    if name and name == home:
        return "h"
    if name and name == away:
        return "a"
    return None


def extract_raw_inplay(m: dict) -> Dict[str, float]:
    """Pull the RAW_INPLAY_KEYS out of an API fixture object. Nothing derived."""
    teams = (m.get("teams") or {})
    home = _txt((teams.get("home") or {}).get("name"))
    away = _txt((teams.get("away") or {}).get("name"))
    home_id, away_id = _team_ids(m)

    sh: Dict[str, Any] = {}
    sa: Dict[str, Any] = {}
    for s in (m.get("statistics") or []):
        side = _side_of(s.get("team"), home_id, away_id, home, away)
        if side is None:
            continue
        parsed = {(i.get("type") or ""): i.get("value") for i in (s.get("statistics") or [])}
        (sh if side == "h" else sa).update(parsed)

    red_h = red_a = 0
    for ev_ in (m.get("events") or []):
        if (ev_.get("type", "") or "").lower() == "card":
            d = (ev_.get("detail", "") or "").lower()
            if "red" in d or "second yellow" in d:
                side = _side_of(ev_.get("team"), home_id, away_id, home, away)
                if side == "h":
                    red_h += 1
                elif side == "a":
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
        # Possession is the one statistic whose absence is NOT plausibly zero:
        # the two sides always sum to 100. A missing feed therefore arrives as
        # 0/0, which is not merely uninformative - it zeroes game_control_h/a
        # (both are possession-weighted) and hands the model a false "neither
        # side had the ball". Substituting an even split says "unknown".
        **_possession(sh, sa),
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
    fid = int((m.get("fixture") or {}).get("id") or 0)
    raw.update(_market_fair_priors(fid, live=True))
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
    # Possession is deliberately NOT one of these. It was always close to a
    # free pass - nonzero from the first minute of every match - and since
    # _possession() substitutes an even split for a missing feed it is now
    # literally always populated, so counting it would mean this check could
    # be satisfied by data that never arrived.
    fields = [raw.get("xg_h", 0) + raw.get("xg_a", 0),
              raw.get("sot_h", 0) + raw.get("sot_a", 0),
              raw.get("cor_h", 0) + raw.get("cor_a", 0),
              raw.get("total_shots_h", 0) + raw.get("total_shots_a", 0)]
    return sum(1 for v in fields if (v or 0) > 0) >= max(0, require_fields)


def _xg_feed_is_dead(raw: Dict[str, float]) -> bool:
    """
    A shot on target ALWAYS carries positive expected goals. There is no
    football state in which SOT > 0 and total xG is exactly 0.00, so that
    combination is proof the xG channel is absent - not proof that no chances
    were created.

    This matters more than an ordinary missing feature because the model cannot
    tell the difference. extract_raw_inplay() defaults a missing "Expected
    Goals" to 0.0, so an absent feed and a genuinely chanceless half arrive at
    build_inplay_features() as the same vector. The model reads 0.00-0.00 as
    strong evidence that neither side is threatening and prices the unders and
    the No side accordingly - confidently, off a fact that was never observed.

    Total shots are checked as well as shots on target so a feed that carries
    shot counts but no xG is caught either way.
    """
    if (raw.get("xg_h", 0) or 0) + (raw.get("xg_a", 0) or 0) > 0:
        return False
    shots = ((raw.get("sot_h", 0) or 0) + (raw.get("sot_a", 0) or 0)
             + (raw.get("total_shots_h", 0) or 0) + (raw.get("total_shots_a", 0) or 0))
    return shots > 0


def fulltime_market_open(m: dict) -> bool:
    """Is the 90-minute market this fixture would be tipped on still live?"""
    st = (((m.get("fixture") or {}).get("status") or {}).get("short") or "").upper()
    return st in FULLTIME_MARKET_OPEN_STATUSES


def inplay_data_gate(raw: Dict[str, float], minute: int,
                     match: Optional[dict] = None) -> Optional[str]:
    """
    Is there enough real observation here to BET on, as opposed to enough to
    record? Returns None when the fixture is bettable, else the reason.

    stats_coverage_ok() answers "did any statistics arrive at all" and is
    deliberately coarse. This is the stricter question, and it is separate for
    two reasons: it must not block harvesting, and a fixture it blocks must
    still appear on the dashboard with the reason attached - a gate you cannot
    see is indistinguishable from a bug.
    """
    if match is not None and not fulltime_market_open(match):
        # Extra time, the break before it, or a shootout. The full-time market
        # settled at 90 minutes; anything quoted now is a different bet.
        return "market_already_settled"
    if minute < LIVE_TIP_MIN_MINUTE:
        return "too_early"
    if minute > LIVE_TIP_MAX_MINUTE:
        # Snapshots are harvested up to TRAIN_MAX_MINUTE, so scoring past it
        # asks the model to extrapolate off the top of its training range -
        # the same argument as LIVE_TIP_MIN_MINUTE, at the other end.
        return "too_late"
    if REQUIRE_XG_FEED and _xg_feed_is_dead(raw):
        return "xg_feed_dead"
    if minute >= SHOT_DATA_MIN_MINUTE:
        shots = ((raw.get("sot_h", 0) or 0) + (raw.get("sot_a", 0) or 0)
                 + (raw.get("total_shots_h", 0) or 0) + (raw.get("total_shots_a", 0) or 0))
        if shots <= 0:
            # Possession and corners alone pass stats_coverage_ok() while the
            # entire shot channel is missing. Possession is nonzero from the
            # first minute of every match, so it is close to a free pass.
            return "no_shot_data"
    return None


def prematch_data_gate(feat: Dict[str, float]) -> Optional[str]:
    """
    The prematch counterpart of inplay_data_gate(): is there enough real
    history here to BET on? Returns None when the fixture is bettable, else
    the reason.

    The in-play path refuses an all-zero observation twice, in
    stats_coverage_ok() and again in inplay_data_gate(). The prematch path had
    no equivalent at all — its only check was `if not feat`, and
    assemble_prematch_features() ends with

        return {k: float(f.get(k, 0.0)) for k in PRE_FEATURES}

    so it ALWAYS returns a fully-populated dict and `feat` is never falsy. A
    fixture whose team-form fetches all failed therefore arrived here as a
    complete vector of zeros and was scored and tipped like any other. Worse,
    every fixture in such an outage gets the SAME vector — two nameless 1500-Elo
    sides with no history — so the model returns the same probability for all of
    them and the scan can emit a burst of identical tips off data it never
    received.

    A team that genuinely played matches cannot have gf, ga, win and draw all
    exactly zero: every finished game lands in exactly one of the three
    outcomes, and a win moves `win`, a draw moves `draw`, and a defeat concedes
    at least one goal and so moves `ga`. All four at zero therefore means the
    window was empty — which after ODDS/form fetch failures is the common case,
    not a rare one.
    """
    for side, tag in (("h", "home"), ("a", "away")):
        observed = (abs(float(feat.get(f"pm_gf_{side}", 0.0)))
                    + abs(float(feat.get(f"pm_ga_{side}", 0.0)))
                    + abs(float(feat.get(f"pm_win_{side}", 0.0)))
                    + abs(float(feat.get(f"pm_draw_{side}", 0.0))))
        if observed <= 0.0:
            return f"no_form_data_{tag}"
    return None


def _league_name(m: dict) -> Tuple[int, str]:
    lg = (m.get("league") or {}) or {}
    return int(lg.get("id") or 0), f"{lg.get('country','')} - {lg.get('name','')}".strip(" -")


def _teams(m: dict) -> Tuple[str, str]:
    t = (m.get("teams") or {}) or {}
    return t.get("home", {}).get("name", ""), t.get("away", {}).get("name", "")


def _team_ids(m: dict) -> Tuple[int, int]:
    """(home_id, away_id), 0 where the feed didn't carry one."""
    t = (m.get("teams") or {}) or {}
    return (int((t.get("home") or {}).get("id") or 0),
            int((t.get("away") or {}).get("id") or 0))


def fulltime_goals(fx: dict) -> Tuple[int, int]:
    """
    The 90-minute score, which is what every market here settles on.

    THE BUG THIS FIXES. API-Football's top-level `goals` object is the CURRENT
    total and keeps counting through extra time, while `score.fulltime` is the
    90-minute score. Results were being recorded from `goals` for every fixture
    in FINAL_STATUSES - which includes AET and PEN - so any knockout tie that
    went to extra time was stored with its 120-minute score.

    Every market Goalsniper prices (Over/Under, BTTS, 1X2, Double Chance, Draw
    No Bet) is a FULL-TIME market: bookmakers grade it on 90 minutes plus
    stoppage and ignore extra time entirely. So a tie level at 1-1 after 90 and
    finishing 3-2 after extra time was being stored as Over 2.5 = yes and
    BTTS = yes, when the bets actually settled Under 2.5 and BTTS = yes at
    1-1. That is a wrong TRAINING LABEL and a wrong P&L grade from the same
    line of code.

    Falls back to `goals` when score.fulltime is absent - it is null for
    fixtures that have not finished, and for those `goals` is the live score,
    which is what the caller wants.
    """
    ft = ((fx.get("score") or {}).get("fulltime") or {})
    h, a = ft.get("home"), ft.get("away")
    if h is None or a is None:
        g = fx.get("goals") or {}
        h, a = g.get("home"), g.get("away")
    try:
        return int(h or 0), int(a or 0)
    except (TypeError, ValueError):
        return 0, 0


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


def _anchor_offset(feat: Dict[str, float], mdl: Dict[str, Any]) -> float:
    """
    The market's log-odds for a market-anchored model, 0.0 for an ordinary one.

    Reproduced here exactly as training built it - same feature, same clip -
    via the shared feature_spec.anchor_logit(). A second copy of that clip
    would be one more pair of constants to drift apart.

    A missing market falls back to the NEUTRAL prior rather than to 0.0:
    anchor_logit(0.0) is -13.8, which would read as "the market says
    impossible" and drag every prediction to the floor.
    """
    anchor = mdl.get("market_anchor")
    if not anchor:
        return 0.0
    return anchor_logit(feat.get(anchor, NEUTRAL_MARKET_PRIORS.get(anchor, 0.5)))


def _score_prob(feat: Dict[str, float], mdl: Dict[str, Any]) -> float:
    """
    logit(p) = market_offset + a * (intercept + w.x) + b

    Platt scaling is applied to the model's DEVIATION only, never to the
    market offset. Calibrating the sum would multiply the market's log-odds by
    `a` as well, which un-pins the very coefficient anchoring exists to fix -
    an a of 1.1 would quietly restore the market as a fitted feature, and a
    noisy `b` would shift every prediction off the market by a constant.

    For an unanchored model the offset is 0 and this reduces to exactly the
    previous sigmoid(a * linpred + b), since logit(sigmoid(z)) == z.
    """
    dev = _linpred(feat, mdl)
    offset = _anchor_offset(feat, mdl)
    cal = mdl.get("calibration") or {}
    try:
        a = float(cal.get("a", 1.0)) if cal else 1.0
        b = float(cal.get("b", 0.0)) if cal else 0.0
    except Exception as e:
        # Falling back to the UNCALIBRATED probability changes what the number
        # means while it still gets compared against the same threshold, so
        # leave a trace rather than swallowing it whole.
        log.debug("[SCORE] calibration unusable (%s) — scoring uncalibrated", e)
        a, b = 1.0, 0.0
    return max(0.0, min(1.0, _sigmoid(offset + a * dev + b)))


# ───────── Serving-side circuit breaker ─────────
# A head whose own holdout says it should not be trusted must not be able to
# send a tip. Warning about it in the nightly digest is not containment: the
# scan runs every five minutes and the digest is read once a day, so a
# miscalibrated head bets ~288 times before anyone sees the warning.
#
# EV is computed straight from the model's probability, so an N-point
# overconfident head overstates every EV it produces by roughly N x odds
# points. At a live price of 2.0 a 5pp gap is a 10pp phantom edge against an
# EDGE_MIN_BPS of 3pp - the gate would be measuring the model's error rather
# than the market's, and would pass exactly the bets with no real edge.
MODEL_HEADS = ["BTTS_YES", "WLD_HOME", "WLD_DRAW", "WLD_AWAY"] + [
    f"OU_{_fmt_line(ln)}" for ln in OU_LINES]
CALIBRATION_GAP_SUPPRESS_PP = float(os.getenv("CALIBRATION_GAP_SUPPRESS_PP", "5.0"))
# Below this many independent fixtures a holdout number is not an estimate.
MIN_TRAIN_MATCHES_TO_BET = int(os.getenv("MIN_TRAIN_MATCHES_TO_BET", "300"))
# A head that routinely disagrees with a priced market by this much is noisy,
# not smart. Only meaningful for market-anchored heads, which is where it is
# measured.
MAX_DEVIATION_P95_PP = float(os.getenv("MAX_DEVIATION_P95_PP", "15.0"))
# Share of training rows whose outcome the scoreline had already settled. Past
# this the head's apparent skill is mostly answering decided questions, none of
# which is bettable.
MAX_DECIDED_SHARE_PCT = float(os.getenv("MAX_DECIDED_SHARE_PCT", "60.0"))
# Brier skill against the best possible constant (always predict the base
# rate). At or below zero the head has learned nothing a single number could
# not do. This is the most basic test there is, and accuracy hides it
# completely: a head calling every fixture Over on a 60% base rate reports 60%
# accuracy and has no skill whatsoever. Every prematch head in the first real
# run scored between -0.005 and +0.006 — noise around zero.
MIN_BRIER_SKILL = float(os.getenv("MIN_BRIER_SKILL", "0.02"))


def _head_health(name: str) -> Dict[str, Any]:
    raw = get_setting_cached(f"model_health:{name}")
    if not raw:
        return {}
    try:
        out = json.loads(raw)
        return out if isinstance(out, dict) else {}
    except Exception:
        return {}


def head_fit_to_bet(name: str) -> Tuple[bool, Optional[str]]:
    """
    (ok, reason). False means SCORE but never TIP this head.

    Absent health data is not a failure: heads trained before this existed, and
    every prematch head, have none. Refusing to bet those would silently stop
    the whole system on the next deploy, which is a worse failure than the one
    being prevented. They pass, and the digest reports how many did.
    """
    h = _head_health(name)
    if not h:
        return True, None

    # A validation finding is the training run refusing its own output. It is
    # checked first because it means a number elsewhere in this record cannot
    # be trusted to judge anything.
    failed = h.get("validation_failed")
    if failed:
        return False, f"failed post-training validation — {failed}"

    ok_parity, why_parity = verify_model_parity(name)
    if not ok_parity:
        return False, why_parity

    gap = h.get("calibration_gap_pct")
    if gap is not None and float(gap) > CALIBRATION_GAP_SUPPRESS_PP:
        # Positive is OVERconfident: predicted minus actual.
        return False, (f"overconfident by {float(gap):.1f}pp on holdout "
                       f"(EV overstated ~{abs(float(gap)) * 2:.0f}pp at odds 2.0)")

    skill = h.get("brier_skill")
    if skill is not None and float(skill) < MIN_BRIER_SKILL:
        return False, (f"Brier skill {float(skill):+.3f} vs a constant — it has learned "
                       f"nothing a single number could not do")

    n = h.get("n_train_matches")
    if n is not None and int(n) < MIN_TRAIN_MATCHES_TO_BET:
        return False, f"trained on {int(n)} fixtures, needs {MIN_TRAIN_MATCHES_TO_BET}"

    p95 = h.get("deviation_p95_pp")
    if p95 is not None and float(p95) > MAX_DEVIATION_P95_PP:
        return False, (f"disagrees with the market by {float(p95):.1f}pp at p95 — "
                       f"noise before it is edge")

    dec = h.get("decided_share_pct")
    if dec is not None and float(dec) > MAX_DECIDED_SHARE_PCT:
        return False, (f"{float(dec):.0f}% of its training rows were already "
                       f"settled when harvested")
    return True, None


PARITY_TOLERANCE = float(os.getenv("PARITY_TOLERANCE", "1e-6"))
_PARITY_CACHE = _TTLCache(ttl=int(os.getenv("PARITY_CACHE_TTL_SEC", "300")))


def verify_model_parity(name: str) -> Tuple[bool, Optional[str]]:
    """
    Does the deployed model score its own training samples the way training did?

    Training records a few real holdout rows with the probability it produced.
    This re-scores them through the serving path and compares.

    It is the only check that can catch train/serve DRIFT, and drift is the
    failure mode neither side's own tests can see: each is internally
    consistent, and the bug exists only in their disagreement. Every scaler,
    calibration or market-offset change is a chance to introduce it — the
    market anchor alone put the offset in three places that all had to agree.
    """
    cached = _PARITY_CACHE.get(name, _MISS)
    if cached is not _MISS:
        return cached

    samples = (_head_health(name) or {}).get("golden_samples") or []
    mdl = load_model_from_settings(name)
    result: Tuple[bool, Optional[str]] = (True, None)
    if samples and mdl:
        worst = 0.0
        for s_ in samples:
            try:
                got = _score_prob(dict(s_.get("features") or {}), mdl)
                worst = max(worst, abs(got - float(s_.get("prob"))))
            except Exception as e:
                result = (False, f"could not re-score a training sample: {e}")
                break
        else:
            if worst > PARITY_TOLERANCE:
                result = (False, (f"scores its own training samples {worst:.4f} differently "
                                  f"here than training did — the serving path and the "
                                  f"fitted model disagree"))
    _PARITY_CACHE.set(name, result)
    return result


def _suppressed_heads() -> Dict[str, str]:
    """Every head currently refused a bet, with the reason. For the dashboard."""
    out: Dict[str, str] = {}
    for name in MODEL_HEADS:
        ok, why = head_fit_to_bet(name)
        if not ok and why:
            out[name] = why
    return out


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


# Bet names that mention "total"/"goals" but are NOT the match over/under.
# API-Football's catalogue also carries team totals ("Total - Home"), halves
# ("Goals Over/Under First Half"), corners, cards, exact-score and odd/even
# markets - and every one of them quotes a plain "Over 2.5" label, so they
# used to be folded into OU_2.5 alongside the real match total.
#
# That was not cosmetic. fetch_odds keeps the BEST price per selection, and a
# single team scoring 3+ prices around 4.0-9.0 against ~1.9 for the match
# total, so the wrong price won the comparison every time. The inflated price
# then flowed into the EV gate (tipping bets whose real price never existed)
# and into P&L - which is what produced a 116.8% ROI on 310 PRE Over/Under
# 2.5 bets at a 52.9% win rate, implying average winning odds of ~4.1 on a
# market that trades between about 1.4 and 3.0.
#
# Asian/handicap lines are excluded deliberately too: their quarter lines
# (2.25, 2.75) settle half-win/half-loss, and _tip_outcome_for_result grades
# a straight win or loss, so pricing off them would misgrade the bet.
# Scope qualifiers that make a bet something other than the FULL-MATCH
# market, whichever family it otherwise names. This is not an Over/Under
# quirk - every family had the same hole:
#
#   "Both Teams To Score - First Half"  -> BTTS
#   "First Half Winner"                 -> 1X2
#   "Double Chance - First Half"        -> DC
#   "Draw No Bet (1st Half)"            -> DNB
#
# A half is a shorter sample than a match, so its decisive outcomes always
# price LONGER than the full-time equivalent (half-time BTTS ~3.5 against
# ~1.9, half-time home ~2.5 against ~1.8). fetch_odds keeps the BEST price
# per selection, so the half price won every comparison and became the
# recorded price for a full-match bet - inflating EV, the tip decision, and
# the P&L, in every market rather than just Over/Under.
_NOT_FULL_MATCH_SCOPE = (
    "half", "halves", "1st", "2nd", "first", "second", "quarter", "period",
    "minute", "extra", "overtime", "incl", "penalt", "shootout",
    "corner", "card", "booking", "offside", "foul", "shot", "save",
    "player", "exact", "odd", "even", "handicap", "asian",
)

# Additionally for the goals total: a TEAM's total is not the MATCH total.
# "Total - Home" quotes a plain "Over 2.5" priced ~4.0-9.0 (that team
# scoring 3+) against ~1.9 for the match. Kept separate from the list above
# because "Both Teams To Score" legitimately contains "team".
_OU_NOT_MATCH_TOTAL = ("home", "away", "team")


def _market_name_normalize(s: Any) -> str:
    s = _txt(s).lower()
    # Returning the raw name leaves it unmapped, so _parse_book_market skips
    # it rather than pricing one market's selection off another's.
    if any(bad in s for bad in _NOT_FULL_MATCH_SCOPE):
        return s
    if "both teams" in s or "btts" in s:
        return "BTTS"
    if "double chance" in s:
        return "DC"
    if "draw no bet" in s:
        return "DNB"
    if "match winner" in s or "winner" in s or "1x2" in s:
        return "1X2"
    if "over/under" in s or "total" in s or "goals" in s:
        if any(bad in s for bad in _OU_NOT_MATCH_TOTAL):
            return s
        return "OU"
    return s


# The in-play feed is one aggregated source rather than a panel of books, so
# it gets a stable name of its own: n_books is then honestly 1 for live,
# instead of borrowing the credibility of a multi-book consensus.
LIVE_FEED_BOOK = "API-Football (in-play)"


def _iter_price_sources(r: dict) -> List[Tuple[str, List[dict]]]:
    """
    (book_name, bets) pairs, for BOTH shapes the odds API returns.

    /odds (prematch) nests markets under a list of "bookmakers".
    /odds/live returns a single aggregated in-play feed with the markets
    directly under "odds" and no bookmaker layer at all.

    Only the first was handled, so every live fixture parsed to zero markets
    no matter what the feed contained - which is why in-play candidates came
    back no_odds 100% of the time, on every scan, while prematch priced
    normally off the same code. Nothing downstream was wrong; the prices
    never arrived.
    """
    books = r.get("bookmakers")
    if isinstance(books, list) and books:
        return [(_txt(bk.get("name")) or "Book", bk.get("bets") or []) for bk in books]
    live_odds = r.get("odds")
    if isinstance(live_odds, list) and live_odds:
        return [(LIVE_FEED_BOOK, live_odds)]
    return []


def _odd_value(v: dict) -> float:
    """Parse a price, tolerating strings, commas and nulls. 0.0 means unusable."""
    try:
        # In-play selections carry a suspended flag while the market is
        # frozen. A suspended price cannot be taken, so it is not a price.
        if v.get("suspended") is True:
            return 0.0
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
    js = _api_get(ODDS_LIVE_URL if live else ODDS_PREMATCH_URL, params)
    if js is None:
        # The call failed (network, HTTP error, or an error reported inside a
        # 200). Deliberately NOT cached: caching {} here would extend a
        # transient outage for the whole TTL and make every candidate in that
        # window read as no_odds, which looks like a pricing problem.
        return {}

    best: Dict[str, Dict[str, Dict[str, Any]]] = {}
    by_book: Dict[str, Dict[str, Dict[str, float]]] = {}
    fair_acc: Dict[str, Dict[str, List[float]]] = {}
    # Per-book market width, averaged the same way the fair price is. Measured
    # per book on a COMPLETE market, never across the best-of-many-books
    # prices - those sum to less than the truth and would report a negative
    # overround, i.e. the book paying you to bet.
    overround_acc: Dict[str, List[float]] = {}
    books_seen: Dict[str, set] = {}
    parse_errors = 0

    # FIX: the try/except used to wrap the ENTIRE response, and its handler
    # reset best/fair/books to {}. So a single malformed value, in a single
    # market, from a single bookmaker, threw away every price for that fixture —
    # all markets, all books. That is why 530 parse failures produced zero
    # priced candidates rather than merely degraded ones. Failures are now
    # isolated to the market that caused them; everything else survives.
    response = js.get("response", []) if isinstance(js, dict) else []
    for r in response:
        for book_name, bets in _iter_price_sources(r):
            per_market: Dict[str, Dict[str, float]] = {}
            for mkt in bets:
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
                        # Per-book prices, so a price can be compared against
                        # the SAME book later. "best" is a maximum over
                        # whichever books happened to be quoting, and that set
                        # grows towards kickoff - comparing one max against a
                        # larger max measures book coverage, not line movement.
                        by_book.setdefault(mkey, {}).setdefault(name, {})[book_name] = float(o)
                    # Only de-vig a COMPLETE market, and normalise to the total
                    # that market's true probabilities actually sum to.
                    needed = _MARKET_SELECTION_COUNT.get(mkey, 2)
                    if len(sel) >= needed:
                        total = MARKET_PROBABILITY_TOTAL.get(mkey, 1.0)
                        implied = {k: 1.0 / v for k, v in sel.items() if v > 1.0}
                        book_sum = sum(implied.values())
                        if book_sum > 0:
                            # Relative to what this market's true probabilities
                            # sum to, so Double Chance (total 2.0) is not
                            # reported as a 100% margin.
                            overround_acc.setdefault(mkey, []).append(book_sum / total - 1.0)
                        for k, p in devig(implied, market_total=total).items():
                            fair_acc.setdefault(mkey, {}).setdefault(k, []).append(p)
                except Exception as e:
                    parse_errors += 1
                    log.debug("[ODDS] fixture %s market %s aggregation failed: %s", fid, mkey, e)

    if parse_errors:
        log.debug("[ODDS] fixture %s (live=%s): %d market(s) unparseable, %d market(s) usable",
                  fid, live, parse_errors, len(best))

    if response and not best:
        # A response arrived but nothing priced. Say what shape it had: the
        # in-play feed going unparsed for its entire history cost a long hunt
        # that one line of this would have ended immediately.
        # Name the markets that were on offer, not just the envelope. The
        # shape question is settled; what matters now is whether the feed
        # only quoted markets we deliberately refuse (asian/quarter lines,
        # halves, corners) or whether the exclusion list is over-rejecting
        # something that is genuinely the full-match market.
        offered = []
        for r in response[:3]:
            if not isinstance(r, dict):
                continue
            for _bk_name, _bets in _iter_price_sources(r):
                offered += [_txt(b.get("name")) for b in _bets if isinstance(b, dict)]
        log.warning("[ODDS] fixture %s (live=%s): %d response item(s) but no usable markets. "
                    "Top-level keys: %s. Markets offered: %s",
                    fid, live, len(response),
                    sorted(response[0].keys()) if isinstance(response[0], dict) else type(response[0]),
                    sorted(set(offered)) or "none")

    out: Dict[str, Any] = {}
    for mkey, sels in best.items():
        fair = {k: (sum(v) / len(v)) for k, v in (fair_acc.get(mkey) or {}).items() if v}
        ovr = overround_acc.get(mkey) or []
        out[mkey] = {"best": sels, "fair": fair, "n_books": len(books_seen.get(mkey, ())),
                     "overround": (sum(ovr) / len(ovr)) if ovr else None,
                     "by_book": by_book.get(mkey, {})}
    ODDS_CACHE.set(key, out)
    return out


def _market_fair_priors(fid: int, live: bool) -> Dict[str, float]:
    """
    De-vigged consensus market probabilities as a MODEL INPUT feature, not
    just the post-hoc price/EV gate _price_gate() already uses them for.
    Feeding the market's own read to every model head is one of the
    best-established calibration aids in sports modeling, and this reuses
    the exact same fetch_odds()/devig() machinery already paid for - the
    only change is fetching it before scoring instead of only after a
    candidate already cleared its confidence threshold.

    Returns ONLY the markets that actually resolved to a de-vigged price.
    build_inplay_features() fills anything absent from NEUTRAL_MARKET_PRIORS,
    so the feature vector is unchanged either way.

    THE BUG THIS FIXES. This used to start from dict(NEUTRAL_MARKET_PRIORS)
    and overwrite what it could resolve, so it ALWAYS returned all five keys.
    Snapshots therefore always persisted a market_fair_* value, and
    load_inplay_data's "did this row carry a real market price?" test -
    raw.get(k) is not None - was true for every row ever harvested.

    Market anchoring then anchored to a neutral 0.5 wherever no price had
    existed, which is precisely the outcome frame_anchor_mask() was written to
    prevent. The damage is visible in the first real run: heads reported a mean
    deviation from "market" of 20-24pp and a maximum of almost exactly 50pp -
    the signature of a model saying ~0.95 against a fabricated market of 0.50,
    not of a model with an opinion.

    Absence has to stay absent for the row to be excludable later.
    """
    out: Dict[str, float] = {}
    if not fid:
        return out
    odds_map = fetch_odds(fid, live=live) if API_KEY else {}
    wld = (odds_map.get("1X2") or {}).get("fair") or {}
    if all(k in wld for k in ("Home", "Draw", "Away")):
        out["market_fair_home"] = float(wld["Home"])
        out["market_fair_draw"] = float(wld["Draw"])
        out["market_fair_away"] = float(wld["Away"])
    ou25 = (odds_map.get("OU_2.5") or {}).get("fair") or {}
    if "Over" in ou25:
        out["market_fair_over25"] = float(ou25["Over"])
    btts = (odds_map.get("BTTS") or {}).get("fair") or {}
    if "Yes" in btts:
        out["market_fair_btts_yes"] = float(btts["Yes"])
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


def head_for_candidate(market_text: str, suggestion: str) -> Optional[str]:
    """Which model head produced this candidate, so its health can gate it."""
    mt = market_text.replace("PRE ", "")
    if mt == "BTTS":
        return "BTTS_YES"
    if mt.startswith("Over/Under"):
        try:
            return f"OU_{_fmt_line(float(suggestion.split()[1]))}"
        except (ValueError, IndexError):
            return None
    # 1X2, Double Chance and Draw No Bet are all derived from the same three
    # WLD heads, so any one of them being unfit taints the lot.
    if mt in ("1X2", "Double Chance", "Draw No Bet"):
        return "WLD_HOME"
    return None


def candidate_head_blocked(market_text: str, suggestion: str) -> Optional[str]:
    """The reason this candidate's head may not bet, or None."""
    head = head_for_candidate(market_text, suggestion)
    if not head:
        return None
    heads = (["WLD_HOME", "WLD_DRAW", "WLD_AWAY"]
             if head == "WLD_HOME" and market_text.replace("PRE ", "") != "BTTS"
             else [head])
    for h in heads:
        ok, why = head_fit_to_bet(h)
        if not ok:
            return why
    return None


class PriceCheck(dict):
    """Result of _price_gate. Dict so it serialises straight into the log row."""


def _max_overround_bps(mkey: str) -> int:
    """Width cap for this market, in basis points. See MAX_OVERROUND_BPS."""
    return (MAX_OVERROUND_BPS_3WAY if _MARKET_SELECTION_COUNT.get(mkey, 2) >= 3
            else MAX_OVERROUND_BPS)


def _price_gate(market_text: str, suggestion: str, fid: int, prob: float, live: bool) -> PriceCheck:
    """
    Single place where a candidate meets the market.

    Gates, in order:
      1. odds exist (unless ALLOW_TIPS_WITHOUT_ODDS)
      2. odds within [min_for_market, MAX_ODDS_ALL]
      3. a de-vigged fair price is computable (unless REQUIRE_FAIR_PRICE=0)
      3b. the market is not so wide that its fair price is untrustworthy
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

    # Placed FIRST, ahead of any odds work. A head its own holdout says is
    # untrustworthy should not reach a price at all: spending a fetch on it and
    # then reporting "EV too low" invites tuning EDGE_MIN_BPS when the problem
    # is the model. Every tipping path runs through here, so this is the one
    # place that cannot be bypassed.
    blocked = candidate_head_blocked(market_text, suggestion)
    if blocked:
        res["decision"] = "head_suppressed"
        res["suppressed_reason"] = blocked
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
    elif res["n_books"] < (MIN_BOOKS_FOR_FAIR_LIVE if live else MIN_BOOKS_FOR_FAIR):
        res["decision"] = "too_few_books"
        res["fair_prob"] = float(fair)
        if REQUIRE_FAIR_PRICE:
            return res
    else:
        res["fair_prob"] = float(fair)

    # Market width. Reported on every candidate that has a complete market,
    # whether or not it gates - a number you can see for a week is worth more
    # than a threshold picked before seeing any.
    overround = entry.get("overround")
    if overround is not None:
        res["overround_pct"] = round(float(overround) * 100.0, 2)
        cap = _max_overround_bps(mkey)
        if fair is not None and cap > 0 and int(round(float(overround) * 10000)) > cap:
            # Refused BEFORE measuring edge, because at this width the fair
            # price is the thing in doubt. Measuring an edge against it would
            # be quantifying our own de-vig error. See MAX_OVERROUND_BPS.
            res["decision"] = "overround_too_wide"
            return res

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
    new_rh, new_ra = elo_update(ratings.get(home_id, ELO_DEFAULT),
                                ratings.get(away_id, ELO_DEFAULT), gh, ga)
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
        # A market probability that was never quoted is persisted as NULL, not
        # as 0.0. Every other raw key is a count where a missing value and zero
        # mean the same thing; a probability is the opposite — 0.0 reads as
        # "the market says impossible" and would be the most confident wrong
        # number in the row. NULL is also what lets training tell a real price
        # from an absent one later, which is the whole basis of anchoring.
        "raw": {k: (None if (k in NEUTRAL_MARKET_PRIORS and raw.get(k) is None)
                    else float(raw.get(k, 0.0)))
                for k in RAW_INPLAY_KEYS},
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
        # 90 minutes, not 120 - see fulltime_goals().
        gh, ga = fulltime_goals(fx)
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
    Records, for every prematch tip approaching kickoff, the best available
    price as the closing line, and stores tip_odds/closing_odds - 1.

    FIX: this used to query AFTER kickoff (kickoff_ts <= now) and call
    fetch_odds(mid, live=False) - the PREMATCH endpoint. Once a fixture goes
    live its prematch market closes (live prices move to /odds/live instead,
    per this file's own T0.4 note that /odds is prematch-only), so that query
    was asking for a prematch price on a market that had already closed by
    the time it asked. Across weeks of real tips this captured a closing
    price for exactly zero of them. "Closing line" also just means the last
    price BEFORE the event starts, industry-wide - fetching it after the fact
    was never the right definition, independent of the API's behaviour.

    Now queries fixtures approaching kickoff (within CLV_CAPTURE_LEAD_MIN)
    while the prematch market is still open, so it can actually succeed.

    PREMATCH ONLY. An in-play bet has no well-defined closing line — the market
    for "Over 2.5 at minute 62" ceases to exist the moment the state changes.
    """
    if not CLV_ENABLE:
        return 0
    now = int(time.time())
    with db_conn() as c:
        rows = c.execute("""
            SELECT match_id, created_ts, market, suggestion, odds, book, kickoff_ts,
                   closing_lead_sec
            FROM tips
            WHERE is_prematch=1 AND odds IS NOT NULL
              AND kickoff_ts IS NOT NULL
              AND kickoff_ts > %s AND kickoff_ts <= %s
            ORDER BY kickoff_ts ASC LIMIT %s
        """, (now, now + CLV_CAPTURE_LEAD_MIN * 60, limit)).fetchall()

    n = 0
    no_same_book = 0
    for (mid, cts, market, sugg, tip_odds, book, kickoff, prev_lead) in rows:
        # RE-CAPTURED on every pass, keeping the price closest to kickoff.
        #
        # This used to filter on `closing_odds IS NULL`, so the FIRST
        # successful capture won - the earliest one, up to CLV_CAPTURE_LEAD_MIN
        # before kickoff. That is not a closing line. The market sharpens as
        # kickoff approaches, so scoring a tip against a T-15 price rather than
        # a T-0 one systematically FLATTERS CLV, and CLV is the one instrument
        # meant to tell us whether the edge is real before the P&L can. An
        # error in the flattering direction is the worst kind here.
        lead = max(0, int(kickoff) - now)
        if prev_lead is not None and int(prev_lead) <= lead:
            # Already hold a price taken at least as close to kickoff.
            continue
        odds_map = fetch_odds(int(mid), live=False)
        mkey, sel = _market_key_and_selection(market or "", sugg or "")
        if not mkey or not book:
            continue
        # SAME BOOK, both ends. The tip price is a maximum across whichever
        # books were quoting at the time, and that set grows towards kickoff -
        # so comparing it against a closing maximum over MORE books measures
        # book coverage, not line movement, and is biased negative by
        # construction. That is what produced "beat close 0% of the time":
        # a larger maximum is almost always the bigger number, whatever the
        # market did. Comparing one book against itself removes the bias.
        closing_prices = ((odds_map.get(mkey) or {}).get("by_book") or {}).get(sel) or {}
        same = closing_prices.get(book)
        if same is None:
            # Better a smaller honest sample than a large biased one: this is
            # the metric that decides whether the edge is real.
            no_same_book += 1
            continue
        closing = float(same)
        if closing <= 1.0:
            continue
        clv = (float(tip_odds) / closing - 1.0) * 100.0
        with db_conn() as c2:
            c2.execute("UPDATE tips SET closing_odds=%s, clv_pct=%s, closing_lead_sec=%s "
                       "WHERE match_id=%s AND created_ts=%s",
                       (closing, round(clv, 3), lead, mid, cts))
        n += 1
    if n or no_same_book:
        log.info("[CLV] captured/refreshed %d closing prices (%d skipped: the book that priced "
                 "the tip was not quoting at close)", n, no_same_book)
    return n


def compute_price_gate_breakdown(days: Optional[int] = None,
                                 phase: Optional[str] = None) -> Dict[str, Any]:
    """
    Why candidates are not becoming tips, read from the `predictions` table
    rather than scraped out of a log buffer.

    Every candidate that reaches _price_gate() has its decision recorded per
    row, so the same question the per-scan "[PROD] price_gate:" line answers
    is answerable over days, filterable, and after the fact.

    SAMPLING, because these counts are not a census: a live candidate is
    logged only on a harvest tick or when it passed, prematch only when it
    was scored, both only above PREDICTION_LOG_MIN_PROB, and then trimmed to
    PREDICTION_LOG_MAX_PER_FIXTURE rows per fixture with every tipped
    candidate kept. So "tipped" is over-represented relative to reality and
    the absolute numbers understate volume. The MIX of rejection reasons is
    what this is for - that is not distorted by keeping extra winners.
    """
    cutoff = int(time.time()) - days * 86400 if days else 0
    sql = "SELECT phase, decision, COUNT(*) FROM predictions WHERE created_ts >= %s"
    params: List[Any] = [cutoff]
    if phase:
        sql += " AND phase = %s"
        params.append(phase)
    sql += " GROUP BY phase, decision"
    with db_conn() as c:
        rows = c.execute(sql, tuple(params)).fetchall()

    by_phase: Dict[str, Dict[str, int]] = {}
    for ph, decision, n in rows:
        by_phase.setdefault(ph or "?", {})[decision or "?"] = int(n)

    # These two are set before the gate runs, so they are not gate outcomes:
    # they mean the candidate never got that far.
    not_gate_outcomes = {"below_threshold", "per_match_cap"}
    summary: Dict[str, Any] = {}
    for ph, counts in by_phase.items():
        gate = {k: v for k, v in counts.items() if k not in not_gate_outcomes}
        reached = sum(gate.values())
        blockers = sorted(((k, v) for k, v in gate.items() if k != "tipped"),
                          key=lambda kv: kv[1], reverse=True)
        summary[ph] = {
            "reached_price_gate": reached,
            "tipped": gate.get("tipped", 0),
            "tipped_pct": round(100.0 * gate.get("tipped", 0) / reached, 1) if reached else 0.0,
            "blocked_by": [{"reason": k, "n": v,
                            "pct_of_gated": round(100.0 * v / reached, 1) if reached else 0.0}
                           for k, v in blockers],
            "never_reached_gate": {k: counts.get(k, 0) for k in not_gate_outcomes
                                   if counts.get(k)},
        }

    return {
        "window_days": days,
        "by_phase": summary,
        "sampling_note": ("Not a census. Live candidates are logged on harvest ticks or when "
                          "they passed, prematch when scored, both only above "
                          f"PREDICTION_LOG_MIN_PROB={PREDICTION_LOG_MIN_PROB}, then trimmed to "
                          f"{PREDICTION_LOG_MAX_PER_FIXTURE}/fixture keeping every tipped one. "
                          "Read the MIX of blocked_by reasons, not the absolute counts, and "
                          "treat tipped_pct as an upper bound."),
        "reading_note": ("no_odds / too_few_books dominating means the market has no usable "
                         "depth for these fixtures — the answer is league scope, not looser "
                         "EV gates. ev_below_min / fair_edge_below_min dominating means the "
                         "model agrees with the market and there is genuinely no edge to bet."),
    }


def compute_clv(days: Optional[int] = None) -> Dict[str, Any]:
    cutoff = int(time.time()) - days * 86400 if days else 0
    with db_conn() as c:
        rows = c.execute("""
            SELECT market, clv_pct, closing_lead_sec FROM tips
            WHERE clv_pct IS NOT NULL AND created_ts >= %s
        """, (cutoff,)).fetchall()
    if not rows:
        return {"n": 0, "note": "No closing prices captured yet. CLV is prematch-only "
                                "and needs at least one full kickoff cycle."}
    by: Dict[str, List[float]] = {}
    allv: List[float] = []
    leads: List[int] = []
    for mkt, clv, lead in rows:
        by.setdefault(mkt or "?", []).append(float(clv))
        allv.append(float(clv))
        if lead is not None:
            leads.append(int(lead))
    allv.sort()

    def _summary(v: List[float]) -> Dict[str, Any]:
        return {"n": len(v), "mean_clv_pct": round(sum(v) / len(v), 2),
                "median_clv_pct": round(sorted(v)[len(v) // 2], 2),
                "beat_close_pct": round(100.0 * sum(1 for x in v if x > 0) / len(v), 1)}

    # How good the benchmark itself was. A line taken 15 minutes out is
    # softer than one at the bell, and being scored against a softer line
    # FLATTERS CLV - so a series with a large mean lead should be read with
    # more suspicion than its number alone suggests.
    bench: Dict[str, Any] = {}
    if leads:
        leads_sorted = sorted(leads)
        bench = {
            "n_with_lead": len(leads),
            "median_sec_before_kickoff": leads_sorted[len(leads_sorted) // 2],
            "mean_sec_before_kickoff": round(sum(leads) / len(leads)),
            "worst_sec_before_kickoff": leads_sorted[-1],
            "note": ("Seconds before kickoff at which the closing price was taken. "
                     "Smaller is a sharper benchmark and a more honest CLV. Prices "
                     "recorded before this field existed are excluded."),
        }

    return {"overall": _summary(allv),
            "by_market": {k: _summary(v) for k, v in by.items() if v},
            "benchmark_quality": bench,
            "note": "mean_clv_pct > 0 sustained over a few hundred prematch bets is the "
                    "strongest available evidence of a real edge. Negative CLV with positive "
                    "ROI means you have been lucky, not right. Prematch only."}


# ───────── Message formatting ─────────
def _format_tip_message(home, away, league, minute, score, suggestion, prob_pct,
                        raw=None, odds=None, book=None, ev_pct=None, fair_prob=None,
                        stake=None, kickoff_txt=None, prematch=False, overround_pct=None):
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
        if overround_pct is not None:
            # The width of the market the fair price above was derived from.
            # Read it as a confidence interval on that fair price, not as a
            # cost: proportional de-vig flatters the favourite by roughly the
            # margin it has to redistribute, so a wide market means part of
            # the EV shown is arithmetic rather than edge.
            money += f"  •  <b>Overround:</b> {overround_pct:.1f}%"
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
                            candidates: List[Tuple[str, str, float, float]],
                            kickoff_ts: int = 0, raw: Optional[Dict[str, float]] = None,
                            home_id: int = 0, away_id: int = 0,
                            data_block: Optional[str] = None) -> Dict[str, Any]:
    # For every candidate that clears its own threshold, run it through the
    # same _price_gate() production_scan() uses to decide whether it would
    # actually get tipped, and surface *why* when it wouldn't - "high
    # confidence, nothing on Telegram" is otherwise unexplainable from the
    # dashboard alone. Candidates nowhere near threshold skip the gate
    # entirely: no point spending a fetch_odds() call to explain a market
    # nobody was going to look at.
    #
    # Known simplification: this evaluates each candidate in isolation, so it
    # does not replicate _correlation_blocked() or the sequential
    # PREDICTIONS_PER_MATCH/MAX_TIPS_PER_SCAN caps from the real tipping loop
    # below. On a match with several qualifying candidates the displayed
    # "tipped" status can therefore disagree with what actually got sent -
    # but it is accurate for the odds/EV/fair-price/sanity gates that explain
    # the overwhelming majority of "why wasn't this tipped" questions.
    markets = []
    for mt, sg, pr, thr in candidates:
        prob_pct = round(float(pr) * 100.0, 1)
        thr_pct = round(float(thr), 1)
        row = {"market": mt, "suggestion": sg, "prob_pct": prob_pct, "threshold_pct": thr_pct}
        if prob_pct >= thr_pct and data_block:
            # The fixture is not bettable at all, so no market on it is. Said
            # once per market rather than silently: "68% confidence, nothing
            # sent" is exactly the question the dashboard exists to answer.
            # Also skips the fetch_odds() call, which would be spent explaining
            # a price we were never going to take.
            row["decision"] = data_block
            row["odds"] = None
            row["ev_pct"] = None
        elif prob_pct >= thr_pct:
            pc = _price_gate(mt, sg, fid, pr, live=True)
            row["decision"] = pc["decision"]
            row["odds"] = pc.get("odds")
            row["ev_pct"] = pc.get("ev_pct")
            row["overround_pct"] = pc.get("overround_pct")
        else:
            row["decision"] = "below_threshold"
            row["odds"] = None
            row["ev_pct"] = None
        markets.append(row)

    # A few of the raw in-play numbers we already fetched, for the dashboard's
    # per-match overview panel - not new API cost, just surfacing what
    # extract_features() already pulled out of /fixtures/statistics.
    stats = None
    if raw:
        stats = {
            "sot_h": raw.get("sot_h", 0.0), "sot_a": raw.get("sot_a", 0.0),
            "cor_h": raw.get("cor_h", 0.0), "cor_a": raw.get("cor_a", 0.0),
            "pos_h": raw.get("pos_h", 0.0), "pos_a": raw.get("pos_a", 0.0),
            "yellow_h": raw.get("yellow_h", 0.0), "yellow_a": raw.get("yellow_a", 0.0),
            # Carried so the xG-feed diagnostic can answer "is the channel
            # alive, and where" from the snapshot the scan already built,
            # without spending a single extra API call.
            "xg_h": raw.get("xg_h", 0.0), "xg_a": raw.get("xg_a", 0.0),
            "total_shots_h": raw.get("total_shots_h", 0.0),
            "total_shots_a": raw.get("total_shots_a", 0.0),
        }

    return {
        "fixture_id": fid, "league": league, "league_id": league_id,
        "home": home, "away": away, "score": score, "minute": minute,
        # Team ids let /dashboard/match/<fid>/form resolve who to look up
        # from the snapshot, instead of trusting ids from the query string.
        "home_id": int(home_id or 0), "away_id": int(away_id or 0),
        "kickoff_ts": int(kickoff_ts or 0), "stats": stats,
        # Why nothing on this fixture is bettable yet, or None when it is.
        "data_block": data_block,
        "markets": markets,
        # Count of candidates that would actually be tipped (passed the full
        # price gate), not just candidates with high raw confidence - this is
        # what "worth a look" should mean on the dashboard.
        "hits": sum(1 for m in markets if m["decision"] == "tipped"),
    }


def _set_live_snapshot(matches: List[Dict[str, Any]], live_seen: Optional[int] = None,
                       no_coverage: Optional[int] = None) -> None:
    # live_seen/no_coverage travel with the matches so the dashboard can tell
    # "nothing is being played right now" apart from "plenty is being played,
    # none of it has usable stats yet" - an empty list on its own can't.
    with _live_snapshot_lock:
        _live_snapshot["updated_ts"] = int(time.time())
        _live_snapshot["matches"] = matches
        _live_snapshot["live_seen"] = live_seen
        _live_snapshot["no_coverage"] = no_coverage


def _get_live_snapshot() -> Dict[str, Any]:
    with _live_snapshot_lock:
        return {"updated_ts": _live_snapshot["updated_ts"],
                "matches": list(_live_snapshot["matches"]),
                "live_seen": _live_snapshot.get("live_seen"),
                "no_coverage": _live_snapshot.get("no_coverage")}


def production_scan() -> Tuple[int, int]:
    matches = fetch_live_matches()
    live_seen = len(matches)
    if live_seen == 0:
        log.info("[PROD] no live")
        _set_live_snapshot([], live_seen=0, no_coverage=0)
        return 0, 0

    saved = 0
    now_ts = int(time.time())
    pred_rows: List[tuple] = []
    live_snapshot_matches: List[Dict[str, Any]] = []
    harvested = 0
    no_coverage = 0
    # Tally of _price_gate() outcomes across the whole scan - saved=0 with a
    # healthy live_seen count is otherwise unexplainable from this log line
    # alone (was it no odds? too few books? edge implausible?).
    gate_decisions: Dict[str, int] = {}
    # Tally of the information gate, kept apart from the price gate: "nothing
    # was tipped" has two completely different causes and one number cannot
    # tell them apart.
    data_gate: Dict[str, int] = {}
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

            # The cooldown stops a fixture being TIPPED twice in quick
            # succession. It is not a reason to hide the match from the
            # dashboard: doing that made every fixture disappear from the
            # live view for DUP_COOLDOWN_MIN minutes starting the moment it
            # produced a tip - i.e. the matches most worth looking at were
            # exactly the ones missing. Same split as the harvest block
            # above: this decides tipping, not what gets displayed.
            cooling_down = False
            if DUP_COOLDOWN_MIN > 0:
                with db_conn() as c:
                    cooling_down = bool(c.execute(
                        "SELECT 1 FROM tips WHERE match_id=%s AND created_ts>=%s "
                        "AND suggestion<>'HARVEST' LIMIT 1",
                        (fid, now_ts - DUP_COOLDOWN_MIN * 60)).fetchone())

            # Is there enough real observation to bet on? Evaluated here, and
            # carried as a flag rather than a `continue`, for the same reason
            # the cooldown is: a fixture we refuse to bet is still a fixture
            # worth SEEING, with the refusal shown next to it. Blocking the
            # snapshot instead would hide exactly the matches being asked about.
            data_block = inplay_data_gate(raw, minute, match=m)
            if data_block:
                data_gate[data_block] = data_gate.get(data_block, 0) + 1

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
            home_id, away_id = _team_ids(m)
            live_snapshot_matches.append(_build_live_match_entry(
                fid, league, league_id, home, away, score, minute, candidates,
                kickoff_ts=kickoff, raw=raw, home_id=home_id, away_id=away_id,
                data_block=data_block))

            # Displayed above, just not re-tipped yet.
            if cooling_down or data_block:
                continue

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
                    gate_decisions[pc["decision"]] = gate_decisions.get(pc["decision"], 0) + 1

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
                    pc.get("odds"), pc.get("book"), pc.get("ev_pct"), pc.get("fair_prob"), stake,
                    overround_pct=pc.get("overround_pct")))
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
    _set_live_snapshot(live_snapshot_matches, live_seen=live_seen, no_coverage=no_coverage)
    log.info("[PROD] saved=%d live_seen=%d candidates_logged=%d harvested=%d no_coverage=%d",
             saved, live_seen, len(pred_rows), harvested, no_coverage)
    if gate_decisions:
        log.info("[PROD] price_gate: %s", gate_decisions)
    if data_gate:
        log.info("[PROD] data_gate: %s", data_gate)
    if gate_decisions.get("head_suppressed"):
        log.warning("[PROD] %d candidate(s) refused because their model head is not fit "
                    "to bet: %s", gate_decisions["head_suppressed"],
                    _suppressed_heads() or "see model_health:*")
    if data_gate.get("xg_feed_dead"):
        # Worth its own line: this one is provable (shots recorded, xG exactly
        # zero) and it silently poisons every model head that reads xG.
        log.warning("[PROD] xG feed absent on %d fixture(s) that had shots recorded — "
                    "those fixtures were scored but not tipped. If this is every "
                    "fixture, the API plan's statistics feed is not carrying "
                    "Expected Goals.", data_gate["xg_feed_dead"])
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
            kickoff = _kickoff_ts_of(m)

            candidates = (_ou_candidates(feat, "", _get_market_threshold)
                          + _btts_candidates(feat, "", _get_market_threshold)
                          + _wld_candidates(feat, "", _get_market_threshold)
                          + _dc_dnb_candidates(feat, "", _get_market_threshold))
            candidates = [c for c in candidates
                          if c[1] in ALLOWED_SUGGESTIONS and _candidate_is_sane(c[1], feat)]
            candidates.sort(key=lambda x: x[2], reverse=True)

            home_id, away_id = _team_ids(m)
            out.append(_build_live_match_entry(fid, league, league_id, home, away, score,
                                               minute, candidates, kickoff_ts=kickoff, raw=raw,
                                               home_id=home_id, away_id=away_id,
                                               data_block=inplay_data_gate(raw, minute, match=m)))
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
    js = _api_get(FOOTBALL_API_URL, {"team": team_id, "last": n})
    if js is None:
        # Failed, not empty. This cache has a 30-MINUTE TTL, so caching [] here
        # is the most damaging instance of the bug fetch_odds() guards against:
        # assemble_prematch_features() would then see an empty window and derive
        # every form feature as 0.0 — gf, ga, win, draw, the momentum block, the
        # venue splits — i.e. score the fixture as though neither side had ever
        # played, and keep doing so for half an hour.
        return []
    out = js.get("response", []) if isinstance(js, dict) else []
    TEAM_FORM_CACHE.set(key, out)
    return out


def _api_h2h(home_id: int, away_id: int, n: int = 5) -> List[dict]:
    key = ("h2h", home_id, away_id, n)
    cached = TEAM_FORM_CACHE.get(key, _MISS)
    if cached is not _MISS:
        return cached
    js = _api_get(f"{FOOTBALL_API_URL}/headtohead", {"h2h": f"{home_id}-{away_id}", "last": n})
    if js is None:
        return []  # failed, not empty — see _api_last_fixtures()
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
    data_gate: Dict[str, int] = {}

    for fx in fixtures:
        fixture = fx.get("fixture") or {}
        lg = fx.get("league") or {}
        teams = fx.get("teams") or {}
        fid = int(fixture.get("id") or 0)
        feat = feats_by_fid.get(fid)
        if not fid or not feat:
            continue

        # The snapshot is still saved below — a fixture we cannot bet is still
        # worth harvesting — but nothing about it may reach a threshold or
        # Telegram. Counted rather than skipped silently, for the reason the
        # in-play gate states: a gate you cannot see is indistinguishable from
        # a bug.
        data_block = prematch_data_gate(feat)

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

        # Placed AFTER the harvest for the reason production_scan() states at
        # length: harvesting is data collection and a gate on BETTING must not
        # also switch off data collection. Same split, same order.
        if data_block:
            data_gate[data_block] = data_gate.get(data_block, 0) + 1
            continue

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
                stake, kickoff_txt=kickoff_txt, prematch=True,
                overround_pct=pc.get("overround_pct")))
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
    log.info("[PREMATCH] saved=%d candidates_logged=%d%s", saved, len(pred_rows),
             f" data_gate={data_gate}" if data_gate else "")
    if data_gate:
        # Loud, because the overwhelmingly likely cause is that the team-form
        # fetches failed rather than that these teams have never played: an
        # empty window is what an API outage looks like from in here.
        log.warning("[PREMATCH] %d fixture(s) had no usable form data and were not "
                    "bet: %s. If this is most of the card, the team-form fetches "
                    "are failing — check [API] warnings above.",
                    sum(data_gate.values()), data_gate)
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
                pc.get("ev_pct"), pc.get("fair_prob"), _stake_units(prob, pc.get("odds")),
                pc.get("overround_pct"))
        if best is None or item[0] > best[0]:
            best = item

    if not best:
        return send_telegram("🏅 Match of the Day: no prematch pick met thresholds.")
    pct, sug, home, away, league, kickoff_txt, odds, book, ev_pct, fair, stake, ovr = best
    msg = _format_tip_message(home, away, league, 0, "", sug, pct, None, odds, book,
                              ev_pct, fair, stake, kickoff_txt=kickoff_txt, prematch=True,
                              overround_pct=ovr)
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

        gh, ga = fulltime_goals(fx)
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

        elo_local[th], elo_local[ta] = elo_update(rating_h, rating_a, gh, ga)

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
    # Bets whose recorded price came from the contaminated market mapping.
    # They are graded the same way and reported separately, never mixed into
    # the headline: their prices were never available for the selection, so
    # counting them as a track record reports fiction as edge.
    stale = {"n_bets": 0, "n_wins": 0, "staked": 0.0, "profit": 0.0}

    for (mkt, sugg, odds, cts, stake_units, clv, gh, ga, btts) in rows:
        outcome = _tip_outcome_for_result(sugg, {"final_goals_h": gh, "final_goals_a": ga, "btts_yes": btts})
        if outcome is None:
            n_push += 1
            continue
        s = float(stake_units) if (use_kelly and stake_units) else float(stake)
        if s <= 0:
            continue
        profit = s * (float(odds) - 1.0) if outcome == 1 else -s

        if int(cts or 0) < ODDS_TRUSTED_FROM_TS:
            stale["n_bets"] += 1
            stale["n_wins"] += 1 if outcome == 1 else 0
            stale["staked"] += s
            stale["profit"] += profit
            continue

        n_bets += 1
        total_staked += s
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
        "odds_trusted_from_ts": ODDS_TRUSTED_FROM_TS,
        "excluded_unreliable_pricing": {
            "n_bets": stale["n_bets"],
            "win_rate_pct": (round(100.0 * stale["n_wins"] / stale["n_bets"], 1)
                             if stale["n_bets"] else 0.0),
            "total_profit": round(stale["profit"], 2),
            "roi_pct": (round(stale["profit"] / stale["staked"] * 100.0, 2)
                        if stale["staked"] > 0 else 0.0),
            "note": ("Graded at prices that were never available for the selection: team "
                     "totals and half markets were being folded into the full-match markets, "
                     "and the best price across that mix won. Reported for completeness, "
                     "excluded from every figure above. Not recoverable — the true price at "
                     "tip time was never recorded."),
        },
        "note": ("Real odds captured at tip time, never synthetic. Tips sent without odds are "
                 "excluded — there is no price to grade them against. Draw No Bet pushes on a "
                 "draw and is excluded rather than counted as a loss. If mean_clv_pct is "
                 "negative while roi_pct is positive, treat the ROI as variance, not edge. "
                 "Figures cover bets priced after the market-mapping fix only; see "
                 "excluded_unreliable_pricing for what came before."),
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
                # Always with n. "beat close 0% of the time" reads as a damning
                # verdict and as noise depending entirely on whether it is 3
                # bets or 300, and the percentage alone cannot be told apart.
                n_clv = int(ov.get("n") or 0)
                line = (f"📉 CLV (7d, prematch, n={n_clv}): {ov['mean_clv_pct']:+.2f}%  •  "
                        f"beat close {ov['beat_close_pct']:.0f}% of the time")
                if n_clv < CLV_MIN_SAMPLE_FOR_VERDICT:
                    line += f"\n   ⚠️ too few to read as edge (need ~{CLV_MIN_SAMPLE_FOR_VERDICT})"
                lines.append(line)
        except Exception:
            pass
        msg = "\n".join(lines)

    # Two failures that otherwise look exactly like "a quiet day".
    tail = []
    try:
        api = _api_call_stats_snapshot()
        if api.get("api_errors"):
            tail.append(f"🚨 API-Football returned {api['api_errors']} error(s) inside "
                        f"200 responses today (quota, plan or key). Those calls got NO "
                        f"data — a quiet scan may mean blindness, not calm.")
        if api.get("rate_limited"):
            tail.append(f"⏳ Rate-limited {api['rate_limited']}x of {api.get('total', 0)} calls.")
        sup = _suppressed_heads()
        if sup:
            tail.append("⛔ Not betting: " + ", ".join(
                f"{escape(h)} ({escape(w)})" for h, w in sorted(sup.items())))
    except Exception as e:
        log.debug("[DIGEST] health tail failed: %s", e)
    if tail:
        msg = msg + "\n" + "\n".join(tail)

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
        rows_line = (f"• Rows: in-play {ds.get('inplay_rows', 0)} "
                     f"({ds.get('inplay_matches', 0)} matches), "
                     f"prematch {ds.get('prematch_rows', 0)}")
        # An anchored run trains on the subset carrying real market prices, so
        # the harvested total is not what the model saw.
        trained = ds.get("inplay_rows_trained")
        if trained is not None and trained != ds.get("inplay_rows"):
            rows_line += (f"\n   ↳ in-play trained on {trained} rows "
                          f"({ds.get('inplay_matches_trained', 0)} matches)")
        lines.append(rows_line)

        # Calibration is what makes a probability mean anything, and EV is
        # computed straight from it - so a head that runs N points
        # overconfident overstates every EV it produces by roughly N x odds
        # points. At a live price of 2.0 an 8pp gap is a 16pp phantom edge
        # against an EDGE_MIN_BPS of 3pp: the gate would be measuring the
        # model's error, not the market's. Surfaced here because it was
        # otherwise buried in a metrics blob nobody reads nightly.
        # How much of each head's apparent skill is answering questions the
        # scoreline had already settled. Those rows are free accuracy and
        # cannot be bet, so a high share means the headline precision is
        # measuring an easier problem than the one being staked.
        settled = []
        for name, m in sorted((res.get("metrics") or {}).items()):
            if not isinstance(m, dict):
                continue
            dd = m.get("already_decided")
            if isinstance(dd, dict) and dd.get("decided_share_pct"):
                settled.append((name, dd))
        if settled:
            lines.append("📐 <b>Already-settled rows</b> (free accuracy, not bettable):")
            for name, dd in sorted(settled, key=lambda kv: kv[1]["decided_share_pct"],
                                   reverse=True):
                und = dd.get("base_rate_undecided")
                lines.append(f"   • {escape(name)}: {dd['decided_share_pct']:.0f}% of rows"
                             + (f" · base rate {dd['base_rate_all']:.2f} → {und:.2f} undecided"
                                if und is not None else ""))

        # The run refusing its own output. First, because a critical finding
        # means other numbers in the same run cannot be trusted to judge
        # anything.
        val = res.get("validation") or {}
        if val.get("n_critical"):
            lines.append(f"🛑 <b>Validation: {val['n_critical']} critical finding(s)</b> — "
                         f"{len(val.get('unfit_heads') or {})} head(s) refused a bet")
            for f in [f for f in val.get("findings", [])
                      if f.get("severity") == "CRITICAL"][:6]:
                lines.append(f"   • {escape(str(f.get('head')))} "
                             f"[{escape(str(f.get('check')))}]: {escape(str(f.get('detail')))}")
        elif val:
            lines.append(f"✅ Validation clean ({val.get('n_warning', 0)} warning(s))")

        # Skill against the best possible constant, per head. Ranked worst
        # first because a negative number is not a weak model, it is no model.
        skills = []
        for name, m in sorted((res.get("metrics") or {}).items()):
            if isinstance(m, dict) and m.get("brier_skill") is not None:
                skills.append((name, float(m["brier_skill"])))
        if skills:
            weak = [(n, v) for n, v in skills if v < MIN_BRIER_SKILL]
            lines.append(f"🎯 <b>Skill vs a constant</b>: {len(skills) - len(weak)}/"
                         f"{len(skills)} heads beat always-predict-the-base-rate")
            for n, v in sorted(weak, key=lambda kv: kv[1])[:6]:
                lines.append(f"   • {escape(n)}: {v:+.3f} — learned nothing a single "
                             f"number could not do")

        drifted = []
        for name, m in sorted((res.get("metrics") or {}).items()):
            if not isinstance(m, dict):
                continue
            gap = m.get("calibration_gap_pct")
            if gap is not None and abs(float(gap)) >= CALIBRATION_GAP_WARN_PP:
                drifted.append((name, float(gap)))
        if drifted:
            lines.append("⚠️ <b>Miscalibrated heads</b> (holdout predicted − actual):")
            for name, gap in sorted(drifted, key=lambda kv: abs(kv[1]), reverse=True):
                direction = "over" if gap > 0 else "under"
                lines.append(f"   • {escape(name)}: {gap:+.1f}pp {direction}confident "
                             f"→ EV overstated ~{abs(gap) * 2:.0f}pp at odds 2.0"
                             if gap > 0 else
                             f"   • {escape(name)}: {gap:+.1f}pp {direction}confident")
            lines.append("   Treat their EV as unproven until the gap closes.")

        # Whether the in-play heads are anchored to the market, and if not, how
        # far off that is. An unanchored model is free to wander away from the
        # market price and call the distance an edge - and the price gate then
        # selects whichever candidates wandered furthest in the profitable
        # direction, i.e. the model's own largest errors. This line says which
        # regime tonight's models are in.
        anc = res.get("market_anchoring") or {}
        if anc:
            if anc.get("anchored"):
                lines.append(f"⚓ <b>Market-anchored</b>: {anc.get('anchored_rows', 0)} rows / "
                             f"{anc.get('anchored_matches', 0)} fixtures "
                             f"({anc.get('anchored_share_pct', 0):.0f}% of the set). "
                             f"Heads predict deviation FROM the market price.")
                devs = []
                for name, m in sorted((res.get("metrics") or {}).items()):
                    if not isinstance(m, dict):
                        continue
                    d = m.get("deviation_from_market")
                    if isinstance(d, dict) and d.get("mean_abs_pp") is not None:
                        devs.append((name, d))
                if devs:
                    lines.append("   Deviation from market on holdout (mean / p95):")
                    for name, d in sorted(devs, key=lambda kv: kv[1]["mean_abs_pp"],
                                          reverse=True):
                        lines.append(f"   • {escape(name)}: {d['mean_abs_pp']:.1f}pp / "
                                     f"{d['p95_abs_pp']:.1f}pp")
                    lines.append("   This is what the price gate trades on. Large values are "
                                 "model noise before they are edge.")
            else:
                lines.append(f"⚓ <b>Not market-anchored</b> — {escape(str(anc.get('reason', '')))}")
        # Which feature set each head was fitted on, and whether the reduced
        # one earned the swap. CORE_FEATURES is a hypothesis about which
        # columns are collinear restatements; this is where it gets judged.
        fsel = res.get("feature_selection") or {}
        picks = [(h, d) for h, d in sorted(fsel.items()) if isinstance(d, dict)]
        if picks:
            core = [h for h, d in picks if d.get("chosen") == "core"]
            lines.append(f"🧮 <b>Feature set</b>: reduced on {len(core)}/{len(picks)} heads"
                         + (f" ({', '.join(escape(h) for h in core)})" if core else ""))
            gains = [(h, d["improvement_vs_full"]) for h, d in picks
                     if d.get("improvement_vs_full")]
            if gains:
                best = max(gains, key=lambda kv: kv[1])
                lines.append(f"   Best cal-logloss gain: {escape(best[0])} "
                             f"{best[1]:+.4f} (selected on cal, so believe the holdout)")
        col = (res.get("collinearity") or {}).get("full") or {}
        if col.get("pairs_above_0.95") is not None:
            lines.append(f"   Collinear pairs (|r| ≥ 0.95) in the full set: "
                         f"{col['pairs_above_0.95']}, worst |r| {col.get('max_abs_corr')}")
        rw = (res.get("data_stats") or {}).get("inplay_row_weighting")
        if rw and rw != "per_row":
            lines.append("⚖️ <b>Row weighting</b>: one fixture, one observation "
                         "(snapshots share an outcome, so counting them as independent "
                         "told the fit it had ~9× the sample it has).")
        # Containment, not just a warning: these heads are scored for the
        # dashboard but cannot send a tip until their next retrain clears them.
        try:
            _MODELS_CACHE.invalidate()
            _SETTINGS_CACHE.invalidate()
            sup = _suppressed_heads()
        except Exception:
            sup = {}
        if sup:
            lines.append("⛔ <b>Heads suppressed from betting</b> (still scored, not tipped):")
            for h, why in sorted(sup.items()):
                lines.append(f"   • {escape(h)}: {escape(why)}")
        if res.get("anchor_fallbacks"):
            lines.append("⚠️ Anchored fit failed, fell back for: "
                         + ", ".join(escape(str(h)) for h in res["anchor_fallbacks"]))

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
_PROCESS_STARTED_TS = int(time.time())
_SCHED: Optional[BackgroundScheduler] = None
_scheduler_started = False


def build_info() -> Dict[str, Any]:
    """
    Which commit is actually running.

    "Did my push deploy?" is otherwise unanswerable from outside the Railway
    dashboard, and answering it wrongly wastes real time - a push can sit
    undeployed while the previous build keeps serving, which is
    indistinguishable from the change not working. Railway injects these for
    a GitHub-connected service; they are absent when running locally.
    """
    sha = os.getenv("RAILWAY_GIT_COMMIT_SHA") or ""
    return {
        "commit": sha[:7] or "unknown",
        "commit_full": sha or None,
        "branch": os.getenv("RAILWAY_GIT_BRANCH") or None,
        "deployed_at": os.getenv("RAILWAY_DEPLOYMENT_CREATED_AT") or None,
        "started_ts": _PROCESS_STARTED_TS,
    }


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
        # The gates that decide whether anything gets tipped, as this process
        # actually resolved them. A setting changed in the wrong place - a
        # local .env, which is gitignored and never reaches the container,
        # rather than the platform's own variables - is otherwise invisible
        # until you infer it from behaviour hours later.
        log.info("[CONFIG] price gate: min_books=%d (live=%d) require_fair=%s "
                 "allow_no_odds=%s edge_min=%dbps fair_edge_min=%dbps max_edge=%dbps "
                 "max_overround=%dbps (3way=%dbps) odds_book_filter=%s",
                 MIN_BOOKS_FOR_FAIR, MIN_BOOKS_FOR_FAIR_LIVE, bool(REQUIRE_FAIR_PRICE),
                 bool(ALLOW_TIPS_WITHOUT_ODDS), EDGE_MIN_BPS, FAIR_EDGE_MIN_BPS,
                 MAX_MODEL_EDGE_BPS, MAX_OVERROUND_BPS, MAX_OVERROUND_BPS_3WAY,
                 ODDS_BOOKMAKER_ID or "none")
        log.info("[CONFIG] live data gate: minute %d-%d (tip_min=%d, train_min=%d) "
                 "shot_data_from=%d require_xg=%s",
                 LIVE_TIP_MIN_MINUTE, LIVE_TIP_MAX_MINUTE, TIP_MIN_MINUTE,
                 TRAIN_MIN_MINUTE, SHOT_DATA_MIN_MINUTE, bool(REQUIRE_XG_FEED))
        sup = _suppressed_heads()
        if sup:
            log.warning("[CONFIG] model heads NOT fit to bet: %s", sup)
        else:
            log.info("[CONFIG] all model heads pass their health check")
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


def xg_feed_report(matches: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Is the Expected Goals channel actually arriving, and where is it not?

    A shot on target always carries positive xG, so `shots recorded AND total
    xG exactly 0.00` is proof the channel is absent rather than proof the game
    is quiet. That single test is what separates "nothing is happening" from
    "we are blind", and it is the difference between a model reading a real
    goalless grind and one reading a false one.

    Broken down by league because coverage is a per-competition property of the
    data plan: a handful of dead leagues is a filtering decision, every league
    dead is an account-level fault worth more than any model change.
    """
    total = live = dead = no_shots_yet = 0
    by_league: Dict[str, Dict[str, int]] = {}
    dead_examples: List[Dict[str, Any]] = []
    for m in matches or []:
        st = m.get("stats") or {}
        xg = float(st.get("xg_h") or 0) + float(st.get("xg_a") or 0)
        shots = (float(st.get("sot_h") or 0) + float(st.get("sot_a") or 0)
                 + float(st.get("total_shots_h") or 0) + float(st.get("total_shots_a") or 0))
        total += 1
        lg = _txt(m.get("league")) or "unknown"
        b = by_league.setdefault(lg, {"fixtures": 0, "xg_live": 0, "xg_dead": 0,
                                      "no_shots_yet": 0})
        b["fixtures"] += 1
        if xg > 0:
            live += 1
            b["xg_live"] += 1
        elif shots > 0:
            dead += 1
            b["xg_dead"] += 1
            if len(dead_examples) < 10:
                dead_examples.append({
                    "fixture_id": m.get("fixture_id"), "league": lg,
                    "match": f"{m.get('home')} vs {m.get('away')}",
                    "minute": m.get("minute"), "shots": shots, "xg": xg})
        else:
            # No shots and no xG is consistent and is real football.
            no_shots_yet += 1
            b["no_shots_yet"] += 1

    decided = total - no_shots_yet
    pct = round(100.0 * live / decided, 1) if decided else None
    if not total:
        verdict = "No live fixtures in the snapshot — nothing to judge yet."
    elif decided == 0:
        verdict = ("Every live fixture is still shotless, so the xG channel cannot "
                   "be judged yet. Re-check when matches are further along.")
    elif dead == 0:
        verdict = "xG is arriving on every fixture that has had a shot."
    elif live == 0:
        verdict = ("xG is absent on EVERY fixture that has had a shot. This is an "
                   "account-level fault, not a per-league gap: the API plan is not "
                   "carrying Expected Goals. No live tip can fire while this holds, "
                   "and fixing it is worth more than any model change.")
    else:
        verdict = (f"xG is arriving on {pct}% of fixtures that have had a shot. "
                   f"The dead ones are listed by league below — if they cluster in "
                   f"a few competitions, that is a coverage gap and those leagues "
                   f"are candidates for LEAGUE_DENY_IDS.")

    return {
        "fixtures_in_snapshot": total,
        "xg_live": live,
        "xg_dead_but_shots_recorded": dead,
        "no_shots_yet_undecidable": no_shots_yet,
        "xg_coverage_pct_of_decidable": pct,
        "verdict": verdict,
        "by_league": dict(sorted(by_league.items(),
                                 key=lambda kv: kv[1]["xg_dead"], reverse=True)),
        "dead_examples": dead_examples,
        "method": "A shot on target always carries positive xG, so shots > 0 with "
                  "total xG exactly 0.00 proves the channel is absent rather than "
                  "the game being quiet.",
    }


@app.route("/admin/diagnostics/xg-feed", methods=["GET"])
def http_xg_feed():
    """
    Whether Expected Goals is actually arriving, and on which leagues.

    Reads the in-memory snapshot the scan already built, so it costs NO API
    calls and reflects the most recent scan. Pass ?live=1 to force a fresh
    scoring pass instead, which does spend calls.
    """
    _require_admin()
    if _arg_int("live", 0):
        matches, live_seen = score_live_matches_now()
        rep = xg_feed_report(matches)
        rep["source"] = f"fresh scan ({live_seen} live fixtures)"
        return jsonify({"ok": True, "xg_feed": rep})
    snap = _get_live_snapshot()
    rep = xg_feed_report(snap.get("matches") or [])
    rep["source"] = "last scan snapshot"
    rep["snapshot_age_sec"] = max(0, int(time.time()) - int(snap.get("updated_ts") or 0))
    rep["live_seen_last_scan"] = snap.get("live_seen")
    rep["blocked_from_betting"] = {}
    for m in (snap.get("matches") or []):
        if m.get("data_block"):
            rep["blocked_from_betting"][m["data_block"]] = \
                rep["blocked_from_betting"].get(m["data_block"], 0) + 1
    return jsonify({"ok": True, "xg_feed": rep})


@app.route("/admin/repair/fulltime-results", methods=["POST"])
def http_repair_fulltime_results():
    """
    Re-grade stored results that were recorded from the extra-time score.

    Results used to be read from API-Football's top-level `goals`, which keeps
    counting through extra time, for every fixture in FINAL_STATUSES - and that
    set includes AET and PEN. Every market here settles on 90 minutes, so a tie
    level at 1-1 after 90 and finishing 3-2 after extra time was stored as
    Over 2.5 = yes: a wrong training label and a wrong P&L grade from one line.

    Writing is fixed; these are the rows written before it. Only fixtures whose
    90-minute score actually differs are touched, so this is safe to re-run and
    reports 0 once clean. Bounded by `limit` because each row costs one API
    call, and ordered oldest-first so repeated runs make progress.
    """
    _require_admin()
    limit = max(1, min(_arg_int("limit", 100) or 100, 500))
    dry_run = bool(_arg_int("dry_run", 0))
    with db_conn() as c:
        rows = c.execute(
            "SELECT match_id, final_goals_h, final_goals_a FROM match_results "
            "ORDER BY updated_ts ASC LIMIT %s", (limit,)).fetchall()

    checked = fixed = 0
    changes = []
    for (mid, old_h, old_a) in rows:
        fx = _fixture_by_id(int(mid))
        if not fx:
            continue
        st = (((fx.get("fixture") or {}).get("status") or {}).get("short") or "").upper()
        if st not in FINAL_STATUSES:
            continue
        checked += 1
        gh, ga = fulltime_goals(fx)
        if (gh, ga) == (int(old_h or 0), int(old_a or 0)):
            continue
        if not dry_run:
            with db_conn() as c2:
                c2.execute(
                    "UPDATE match_results SET final_goals_h=%s, final_goals_a=%s, btts_yes=%s, "
                    "updated_ts=%s WHERE match_id=%s",
                    (gh, ga, 1 if (gh > 0 and ga > 0) else 0, int(time.time()), int(mid)))
        fixed += 1
        changes.append({"match_id": int(mid), "was": f"{old_h}-{old_a}",
                        "now": f"{gh}-{ga}", "status": st})
    log.info("[REPAIR] full-time results: checked=%d %s=%d",
             checked, "would_fix" if dry_run else "fixed", fixed)
    return jsonify({"ok": True, "dry_run": dry_run, "checked": checked,
                    "would_fix" if dry_run else "fixed": fixed,
                    "changes": changes[:50],
                    "note": "Re-run until fixed=0. Retrain afterwards so the corrected "
                            "labels reach the models."})


@app.route("/admin/diagnostics/price-gate", methods=["GET"])
def http_price_gate():
    """Why candidates are not becoming tips, over a window rather than per scan."""
    _require_admin()
    return jsonify({"ok": True,
                    "price_gate": compute_price_gate_breakdown(
                        days=_arg_int("days", 7), phase=request.args.get("phase"))})


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
        "build": build_info(),
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
    return jsonify({"ok": True, "tips": tips, "pnl": pnl, "build": build_info(),
                    "server_ts": int(time.time())})


# How many recent fixtures to pull per team. The window is split by venue
# afterwards, so 10 mixed games is what leaves a usable home-only or
# away-only sample; it costs exactly the same one API call as asking for 5.
FORM_WINDOW_GAMES = 10
# Below this many games AT THE VENUE there is nothing to compare - the card
# still shows the number, it just doesn't dress one or two matches up as a
# trend by calling the team "well below the league's usual".
VENUE_FORM_MIN_GAMES = 3


def _venue_verdict(team_rate: float, league_rate: float, played: int) -> Optional[Dict[str, str]]:
    """Where a team's venue win rate sits against its league's, in pp."""
    if played < VENUE_FORM_MIN_GAMES:
        return None
    gap_pp = (team_rate - league_rate) * 100.0
    if gap_pp >= 15.0:
        return {"text": "well above the league's usual", "tone": "good"}
    if gap_pp >= 5.0:
        return {"text": "above the league's usual", "tone": "good"}
    if gap_pp <= -15.0:
        return {"text": "well below the league's usual", "tone": "bad"}
    if gap_pp <= -5.0:
        return {"text": "below the league's usual", "tone": "bad"}
    return {"text": "about the league's usual", "tone": "neutral"}


def _team_form_card(team_id: int, team_name: str, venue: str, league_rate: float) -> Dict[str, Any]:
    # win/gf/ga come back recency-weighted (feature_spec.decay_weights), so
    # the most recent game at this venue counts for more than the oldest -
    # the frontend says "recency-weighted" next to the sample size rather
    # than passing this off as a plain count.
    games = _api_last_fixtures(team_id, FORM_WINDOW_GAMES)
    st = venue_form_stats(team_id, games, venue)
    played = int(st.get("played") or 0)
    win_rate = float(st.get("win") or 0.0)
    return {
        "team": team_name, "venue": venue, "played": played,
        "win_pct": round(win_rate * 100.0, 1),
        "goals_for": round(float(st.get("gf") or 0.0), 2),
        "goals_against": round(float(st.get("ga") or 0.0), 2),
        "league_win_pct": round(league_rate * 100.0, 1),
        "verdict": _venue_verdict(win_rate, league_rate, played),
    }


def build_match_form(entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Home side's home form and away side's away form, each judged against how
    often that league's home/away teams actually win.

    Costs two /fixtures?last= calls per fixture on a cold TEAM_FORM_CACHE,
    which is why nothing calls this during a scan - it runs only when a human
    opens a specific match on the dashboard.
    """
    home_id = int(entry.get("home_id") or 0)
    away_id = int(entry.get("away_id") or 0)
    if not home_id or not away_id:
        return {"available": False, "reason": "this fixture's feed carried no team ids"}
    lvr = get_league_venue_rates(entry.get("league_id"))
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_h = ex.submit(_team_form_card, home_id, entry.get("home") or "", "home", lvr["home_win"])
        f_a = ex.submit(_team_form_card, away_id, entry.get("away") or "", "away", lvr["away_win"])
        home_card, away_card = f_h.result(), f_a.result()
    return {"available": True, "home": home_card, "away": away_card,
            "league_sample": int(lvr.get("n") or 0)}


@app.route("/dashboard/match/<int:fid>/form")
def dashboard_match_form(fid: int):
    """
    Form & momentum for one live fixture, fetched on demand.

    The fixture must be in the current live snapshot: the team ids come from
    there rather than the query string, so this can't be pointed at arbitrary
    teams to burn API quota.
    """
    if not DASHBOARD_ENABLED:
        return _dashboard_unavailable()
    if not _dashboard_authed():
        abort(401)
    entry = next((m for m in _get_live_snapshot()["matches"]
                  if int(m.get("fixture_id") or 0) == fid), None)
    if not entry:
        return jsonify({"ok": False, "error": "fixture is not in the current live snapshot"}), 404
    try:
        form = build_match_form(entry)
    except Exception as e:
        log.warning("[FORM] lookup failed for fixture %s: %s", fid, e)
        return jsonify({"ok": False, "error": "form lookup failed"}), 502
    return jsonify({"ok": True, "fixture_id": fid, **form})


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
    _set_live_snapshot(matches, live_seen=live_seen)
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
