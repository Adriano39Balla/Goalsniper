"""GoalSniper service orchestration.

Research-first release: no delivery before a frozen 500-fixture sharp-close
trial passes. SHADOW_ONLY remains a separate manual stop. Execution quotes
are Tipico only, the prematch benchmark is Pinnacle, and live delivery stays
blocked without a same-state reference. Read README.md before deployment.

Modules: feature_spec (shared model inputs), train_models (classifier fitting),
odds_parser (full-match parsing), risk_policy (shrinkage / CLV inference),
research_store (frozen trials / fills), grading, and xg_history.
"""
from __future__ import annotations

import hmac
import secrets
import json
import logging
import math
import os
import random
import re
import signal
import sys
import threading
import time
from collections import OrderedDict, defaultdict
from contextlib import contextmanager, nullcontext
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from html import escape
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import uuid
import hashlib
import sys
from pathlib import Path
from risk_policy import EXECUTION_BOOK, SHARP_BOOK, MODEL_WEIGHT, shrink_probability, source_timestamp
from research_store import ResearchStore, init_research_schema
from xg_history import historical_xg_features
import psycopg2
import requests
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from flask import (
    Flask, abort, g, jsonify, redirect, render_template, render_template_string, request, url_for,
)
# Aliased: this module already has a module-level `session` (a requests.Session
# for outbound HTTP, defined below) that would otherwise shadow Flask's session
# proxy the moment that line executes.
from werkzeug.exceptions import HTTPException
from flask import session as flask_session
from psycopg2.pool import ThreadedConnectionPool
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from dotenv import load_dotenv
except ModuleNotFoundError as exc:
    if exc.name != "dotenv":
        raise
else:
    load_dotenv()

from feature_spec import (
    ELO_DEFAULT, FEATURE_SCHEMA_VERSION,
    DEFAULT_LEAGUE_RATES, MARKET_PROBABILITY_TOTAL, NEUTRAL_MARKET_PRIORS,
    ODDS_TRUSTED_FROM_TS, RAW_INPLAY_KEYS,
    assemble_prematch_features, build_inplay_features, derive_dc_dnb,
    devig, elo_update, ev as _ev, fixture_ts as _fixture_ts, kelly_fraction,
    enforce_ou_monotonicity, venue_form_stats,
)


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
LIVE_SCAN_ENABLE = _env_flag("LIVE_SCAN_ENABLE", "1")

CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "70"))
MAX_TIPS_PER_SCAN = int(os.getenv("MAX_TIPS_PER_SCAN", "25"))
DUP_COOLDOWN_MIN = int(os.getenv("DUP_COOLDOWN_MIN", "20"))
TIP_MIN_MINUTE = int(os.getenv("TIP_MIN_MINUTE", "8"))
SCAN_INTERVAL_SEC = int(os.getenv("SCAN_INTERVAL_SEC", "300"))
LIVE_MAX_INPUT_AGE_SEC = max(1, int(os.getenv("LIVE_MAX_INPUT_AGE_SEC", "120")))
LIVE_MAX_ODDS_AGE_SEC = max(1, int(os.getenv("LIVE_MAX_ODDS_AGE_SEC", "90")))

PREDICTIONS_PER_MATCH = int(os.getenv("PREDICTIONS_PER_MATCH", "1"))
CORRELATED_EXTRA_EV_BPS = int(os.getenv("CORRELATED_EXTRA_EV_BPS", "400"))

HARVEST_MODE = _env_flag("HARVEST_MODE", "1")
TRAIN_ENABLE = _env_flag("TRAIN_ENABLE", "1")
AUTO_TRAIN_ENABLE = _env_flag("AUTO_TRAIN_ENABLE", "0")
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
# The old collector queried only the current Berlin calendar day.  That made
# the evening scan return zero fixtures as soon as tomorrow's schedule was
# the next available slate.  Keep the window explicit and configurable while
# retaining the prematch/status checks below.
PREMATCH_LOOKAHEAD_HOURS = max(1, int(os.getenv("PREMATCH_LOOKAHEAD_HOURS", "48")))
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
SUPPRESSED_THRESHOLD_PCT = float(os.getenv("SUPPRESSED_THRESHOLD_PCT", "101.0"))
if SUPPRESSED_THRESHOLD_PCT <= 100.0:
    raise SystemExit("SUPPRESSED_THRESHOLD_PCT must be greater than 100")

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
MAX_PREMATCH_ODDS_ALL = float(os.getenv("MAX_PREMATCH_ODDS_ALL", "6.0"))
# Reject an executable quote that is wildly longer than the same market's
# de-vigged consensus. This catches repeated bad mappings that a simple
# best-versus-second-best comparison cannot detect.
MAX_PRICE_VS_FAIR_PCT = float(os.getenv("MAX_PRICE_VS_FAIR_PCT", "20.0"))

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
 # The release benchmark is Pinnacle's de-vigged close, not a blended
 # consensus.  The external adapter currently supplies Tipico + Pinnacle;
 # requiring three books therefore made the default evaluation path
 # deterministically reject every otherwise valid candidate.
MIN_BOOKS_FOR_FAIR = int(os.getenv("MIN_BOOKS_FOR_FAIR", "1"))
# The in-play feed is ONE aggregated source, not a panel of books, so it can
# never reach MIN_BOOKS_FOR_FAIR - live candidates would sit at
# too_few_books forever. This is a separate knob rather than a lower global
# value because prematch genuinely does have multiple books and should keep
# demanding a consensus. Defaults to the strict value so nothing loosens by
# itself: set it to 1 to accept the in-play feed's own de-vigged price, in
# full knowledge that a single source's overround is not a consensus.
MIN_BOOKS_FOR_FAIR_LIVE = int(os.getenv("MIN_BOOKS_FOR_FAIR_LIVE",
                                        str(MIN_BOOKS_FOR_FAIR)))

if min(MIN_BOOKS_FOR_FAIR, MIN_BOOKS_FOR_FAIR_LIVE) < 1:
    raise SystemExit("Fair-price source counts must be positive")
if REQUIRE_FAIR_PRICE and MIN_BOOKS_FOR_FAIR_LIVE > 1:
    log.warning("[CONFIG] live tips blocked by fair-price source count: "
                "the feed has one source, MIN_BOOKS_FOR_FAIR_LIVE=%d. "
                "Set 1 explicitly to accept that source.", MIN_BOOKS_FOR_FAIR_LIVE)

# ───────── Execution realism ─────────
# MIN_BOOKS_FOR_FAIR governs whether a price is trustworthy enough to call
# "the market's fair read" (the EV/edge benchmark). It says nothing about
# whether the BEST price itself — the number a bet actually settles at — was
# real and takeable. fetch_odds() keeps a running max() over whichever books
# happen to be quoting a selection; a single book spiking a stale or fat-
# fingered quote wins that max unconditionally and becomes both the recorded
# tip price and the P&L settlement price for that bet. This is the same
# failure class as the team-totals contamination already fixed in this file
# (_market_name_normalize / _OU_NOT_MATCH_TOTAL) — a wrong price winning a
# comparison it should never have been eligible for — just one level down:
# there the wrong MARKET won the max, here a single uncorroborated BOOK does.
#
# Two independent checks, both applied to the specific selection being
# priced (not the market as a whole) — see fetch_odds()'s "executability"
# block and _price_gate()'s use of it:
#   1. CORROBORATION: at least MIN_BOOKS_FOR_EXECUTION books must be quoting
#      that exact selection. A price only one book is offering is a price
#      nobody else in the market agrees exists.
#   2. OUTLIER SPREAD: the best price cannot exceed the SECOND-best price by
#      more than MAX_EXECUTION_OUTLIER_PCT. A price 40% above the next-best
#      quote is far more likely to be a stale or broken feed than a genuine
#      standout offer — real cross-book spreads on liquid markets are a few
#      percent, not tens of percent.
# A selection failing either check is not un-bettable — the fair-price
# benchmark and EV gate above are untouched — it just cannot be the price
# EXECUTED AND GRADED AT.
MIN_BOOKS_FOR_EXECUTION = int(os.getenv("MIN_BOOKS_FOR_EXECUTION", "2"))
# Same in-play caveat as MIN_BOOKS_FOR_FAIR_LIVE, and it bites immediately:
# the live feed is ONE aggregated source (see LIVE_FEED_BOOK below), so every
# in-play selection has exactly one book by construction and will fail this
# at a multi-book value. The live default is therefore 1: accept the
# aggregated feed's own price, while the outlier comparison is naturally
# unavailable. Prematch remains at the stricter multi-book default above.
MIN_BOOKS_FOR_EXECUTION_LIVE = int(os.getenv("MIN_BOOKS_FOR_EXECUTION_LIVE", "1"))
MAX_EXECUTION_OUTLIER_PCT = float(os.getenv("MAX_EXECUTION_OUTLIER_PCT", "15.0"))
# Mirrors REQUIRE_FAIR_PRICE's shape: 1 means a candidate whose price fails
# corroboration or the outlier check is rejected outright (the decision
# string reflects which); 0 means the check still runs and is logged but
# never blocks a tip.
REQUIRE_EXECUTABLE_PRICE = _env_flag("REQUIRE_EXECUTABLE_PRICE", "1")

ODDS_BOOKMAKER_ID = os.getenv("ODDS_BOOKMAKER_ID")
ALLOW_TIPS_WITHOUT_ODDS = _env_flag("ALLOW_TIPS_WITHOUT_ODDS", "0")

BANKROLL_UNITS = float(os.getenv("BANKROLL_UNITS", "100"))
KELLY_FRACTION = float(os.getenv("KELLY_FRACTION", "0.25"))
MAX_STAKE_PCT = float(os.getenv("MAX_STAKE_PCT", "2.0"))

CLV_ENABLE = _env_flag("CLV_ENABLE", "1")
CLV_CAPTURE_EVERY_MIN = 1  # Sharp-close window is two minutes; do not widen cadence.
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
SHADOW_ENABLE = _env_flag("SHADOW_ENABLE", "1")
SHADOW_ONLY = _env_flag("SHADOW_ONLY", "1")
if SHADOW_ONLY and not SHADOW_ENABLE:
    raise SystemExit("SHADOW_ONLY=1 requires SHADOW_ENABLE=1")

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

# ───────── Concentration mode ─────────
# A solo operator spreading pre+in-play x 7 markets x every league in the
# world thins the data behind every individual head. CONCENTRATION_MODE lets
# the operator explicitly commit to a subset of markets, enforced at every
# candidate-generation call site (see _market_active()) rather than only
# documented as a recommendation nobody re-checks later. It does NOT choose
# which LEAGUES to keep — that needs real numbers, not a guess (see
# compute_league_density() / GET /admin/diagnostics/league-density) — but it
# does refuse to boot if concentration is on while league scope
# (LEAGUE_ALLOW_IDS / PREMATCH_LEAGUE_IDS) is left unbounded, since "I turned
# on concentration" while still scanning every league worldwide is exactly
# the failure mode this exists to prevent. See _enforce_concentration_scope(),
# called once at boot.
CONCENTRATION_MODE = _env_flag("CONCENTRATION_MODE", "1")

# Canonical bare market names — no "PRE " prefix, since that is a phase tag
# concatenated only at the DB-insert call sites in prematch_scan_save()
# (f"PRE {mk}") and in Telegram/dashboard formatting; it is never part of a
# candidate tuple's market_text (see _ou_candidates/_btts_candidates/
# _wld_candidates/_dc_dnb_candidates, which all emit bare names in both
# phases). _market_family() strips it defensively anyway, for any caller
# that passes an already phase-tagged string from tips/predictions.
#
# Mirrors http_thresholds()'s local `markets` list below — kept as an
# independent literal rather than shared, so a future change to one cannot
# silently reorder the other's JSON output.
_ALL_MARKET_FAMILIES = (
    {"BTTS", "1X2", "Double Chance", "Draw No Bet"}
    | {f"Over/Under {_fmt_line(l)}" for l in OU_LINES}
)

_DEFAULT_ACTIVE_MARKETS = "BTTS,Over/Under 2.5"


def _parse_active_markets(env_val: str) -> set:
    names = {m.strip() for m in (env_val or "").split(",") if m.strip()}
    unknown = names - _ALL_MARKET_FAMILIES
    if unknown:
        raise SystemExit(
            f"ACTIVE_MARKETS contains unrecognised market(s): {sorted(unknown)}. "
            f"Valid values: {sorted(_ALL_MARKET_FAMILIES)}")
    return names


# Only consulted when CONCENTRATION_MODE=1. Comma-separated bare market names
# from _ALL_MARKET_FAMILIES, e.g. "BTTS,Over/Under 2.5,1X2". Deliberately
# narrow by default (two markets) — concentration mode exists to force a real
# choice, not to default to "everything" under a new setting's name.
ACTIVE_MARKETS = (_parse_active_markets(os.getenv("ACTIVE_MARKETS", _DEFAULT_ACTIVE_MARKETS))
                  if CONCENTRATION_MODE else set(_ALL_MARKET_FAMILIES))
DISABLED_MARKETS = _parse_active_markets(os.getenv("DISABLED_MARKETS", ""))
DISABLED_LEAGUE_IDS = set(_int_list(os.getenv("DISABLED_LEAGUE_IDS", "")))

# Upper bound on distinct league IDs when concentration is on, enforced at
# boot against LEAGUE_ALLOW_IDS / PREMATCH_LEAGUE_IDS (see
# _enforce_concentration_scope()). A hard SystemExit rather than a log
# warning, because a boot-time check nobody re-reads after a Railway restart
# is not a check.
MAX_ACTIVE_LEAGUES = min(5, int(os.getenv("MAX_ACTIVE_LEAGUES", "5")))


def _market_family(market_text: str) -> str:
    """Bare market name from a candidate/tip/prediction market string."""
    return market_text[4:] if market_text.startswith("PRE ") else market_text


def _market_active(market_text: str) -> bool:
    """Explicit blocks override the active-market set."""
    family = _market_family(market_text)
    return family in ACTIVE_MARKETS and family not in DISABLED_MARKETS


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
# A postponed/abandoned fixture id is not a future match.  Keeping it pending
# forever freezes the prospective cohort.  SUSP is only voided after a grace
# period below, so a same-day interruption can still be settled if resumed.
VOID_STATUSES = {"CANC", "AWD", "WO", "PST", "ABD"}
SUSPENDED_VOID_AFTER_SEC = 48 * 3600


def _fixture_status_void(status: str, fixture: Optional[dict], now: Optional[int] = None) -> bool:
    status = (status or "").upper()
    if status in VOID_STATUSES:
        return True
    if status != "SUSP":
        return False
    now = int(time.time()) if now is None else int(now)
    kickoff = _fixture_ts(fixture or {}) if fixture else 0
    return bool(kickoff and now - kickoff >= SUSPENDED_VOID_AFTER_SEC)

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
ODDS_DIAGNOSTICS = _TTLCache(ttl=900, maxsize=2000)
TEAM_FORM_CACHE = _TTLCache(ttl=TEAM_FORM_TTL, maxsize=int(os.getenv("TEAM_FORM_CACHE_MAXSIZE", "8000")))

SETTINGS_TTL = int(os.getenv("SETTINGS_TTL_SEC", "60"))
MODELS_TTL = int(os.getenv("MODELS_CACHE_TTL_SEC", "120"))
_SETTINGS_CACHE = _TTLCache(SETTINGS_TTL)
_MODELS_CACHE = _TTLCache(MODELS_TTL)
LEAGUE_RATE_TTL = int(os.getenv("LEAGUE_RATE_TTL_SEC", "21600"))
LEAGUE_RATE_MIN_N = int(os.getenv("LEAGUE_RATE_MIN_N", "20"))
_LEAGUE_RATE_CACHE = _TTLCache(LEAGUE_RATE_TTL)
# Original observation times survive cache hits, allowing tip audits to show
# how old the inputs really were when the decision was made.
_STATS_FETCHED_TS: Dict[int, int] = {}

try:
    from train_models import train_models
except ImportError as e:  # pragma: no cover
    # Do not boot a service that looks healthy while its training endpoint is a
    # hidden stub because a deployment dependency is missing. Railway should
    # fail the release and show the real package/import error in the logs.
    raise RuntimeError(
        "train_models could not be imported. Install the runtime dependencies "
        "required by train_models before starting goalsniper."
    ) from e
except Exception as e:  # pragma: no cover
    raise RuntimeError("train_models failed during import; deployment is not safe") from e


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


_SCAN_LOCAL = threading.local()


def get_setting_cached(key: str) -> Optional[str]:
    audit = getattr(_SCAN_LOCAL, 'audit', None)
    if audit is not None and key.startswith(('model', 'research_threshold:')):
        return audit.settings.get(key)
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
    with _tip_transaction() as c:
        c.execute('SELECT pg_advisory_xact_lock(%s)', (19017,))
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
        c.execute("""CREATE TABLE IF NOT EXISTS scan_funnel (
            id BIGSERIAL PRIMARY KEY, created_ts BIGINT NOT NULL,
            phase TEXT NOT NULL, counts TEXT NOT NULL)""")
        c.execute("""CREATE TABLE IF NOT EXISTS shadow_picks (
            id BIGSERIAL PRIMARY KEY, match_id BIGINT NOT NULL,
            league_id BIGINT, league TEXT, kickoff_ts BIGINT,
            created_ts BIGINT NOT NULL, phase TEXT NOT NULL, minute INTEGER,
            market TEXT NOT NULL, suggestion TEXT NOT NULL,
            prob DOUBLE PRECISION NOT NULL, threshold_pct DOUBLE PRECISION,
            odds DOUBLE PRECISION NOT NULL, book TEXT, fair_prob DOUBLE PRECISION,
            ev_pct DOUBLE PRECISION, stake_units DOUBLE PRECISION,
            model_version TEXT NOT NULL, closing_odds DOUBLE PRECISION,
            clv_pct DOUBLE PRECISION,
            UNIQUE(match_id, phase, suggestion))""")

        c.execute("""CREATE TABLE IF NOT EXISTS scan_runs (
            scan_id TEXT PRIMARY KEY, phase TEXT NOT NULL,
            started_ts BIGINT NOT NULL, finished_ts BIGINT,
            status TEXT NOT NULL, model_version TEXT NOT NULL, policy_version TEXT NOT NULL)""")
        c.execute("""CREATE TABLE IF NOT EXISTS scan_decisions (
            id BIGSERIAL PRIMARY KEY, scan_id TEXT NOT NULL REFERENCES scan_runs(scan_id),
            created_ts BIGINT NOT NULL, match_id BIGINT, league_id BIGINT, kickoff_ts BIGINT,
            stage TEXT NOT NULL, market TEXT, suggestion TEXT, reason TEXT NOT NULL,
            prob DOUBLE PRECISION, threshold_pct DOUBLE PRECISION,
            odds DOUBLE PRECISION, fair_prob DOUBLE PRECISION, ev_pct DOUBLE PRECISION,
            price_decision TEXT, qualified BOOLEAN NOT NULL DEFAULT FALSE,
            detail TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS fixture_voids (
            match_id BIGINT PRIMARY KEY, status TEXT NOT NULL, updated_ts BIGINT NOT NULL)""")
        c.execute("""CREATE TABLE IF NOT EXISTS settlement_checks (
            match_id BIGINT PRIMARY KEY, last_checked_ts BIGINT NOT NULL)""")
        for stmt in [
            "ALTER TABLE shadow_picks ADD COLUMN IF NOT EXISTS policy_version TEXT NOT NULL DEFAULT 'legacy'",
            "ALTER TABLE shadow_picks ADD COLUMN IF NOT EXISTS closing_ts BIGINT",
            "ALTER TABLE tips ADD COLUMN IF NOT EXISTS closing_ts BIGINT",
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
            "ALTER TABLE tips ADD COLUMN IF NOT EXISTS decision_ts BIGINT",
            "ALTER TABLE tips ADD COLUMN IF NOT EXISTS stats_fetched_ts BIGINT",
            "ALTER TABLE tips ADD COLUMN IF NOT EXISTS odds_fetched_ts BIGINT",
            "ALTER TABLE tips ADD COLUMN IF NOT EXISTS telegram_sent_ts BIGINT",
            "ALTER TABLE tips ADD COLUMN IF NOT EXISTS price_verified INTEGER DEFAULT 0",
            "ALTER TABLE tips ADD COLUMN IF NOT EXISTS audit_json TEXT",
            "ALTER TABLE tip_snapshots ADD COLUMN IF NOT EXISTS kickoff_ts BIGINT",
            "ALTER TABLE prematch_snapshots ADD COLUMN IF NOT EXISTS kickoff_ts BIGINT",
        ]:
            try:
                c.execute(stmt)
            except Exception as e:
                log.error("[SCHEMA] %s -> %s", stmt, e)
                raise

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
            "CREATE INDEX IF NOT EXISTS idx_decisions_scan ON scan_decisions(scan_id)",
            "CREATE INDEX IF NOT EXISTS idx_decisions_created ON scan_decisions(created_ts)",
            "CREATE INDEX IF NOT EXISTS idx_decisions_match ON scan_decisions(match_id)",
            "CREATE INDEX IF NOT EXISTS idx_scan_funnel_created ON scan_funnel (created_ts DESC)",
            "CREATE INDEX IF NOT EXISTS idx_shadow_created ON shadow_picks (created_ts DESC)",
            "CREATE INDEX IF NOT EXISTS idx_shadow_close ON shadow_picks (phase,kickoff_ts) WHERE closing_odds IS NULL",
        ]:
            try:
                c.execute(stmt)
            except Exception as e:
                log.error("[SCHEMA] %s -> %s", stmt, e)
                raise

        init_research_schema(c)

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
        return bool(r.ok and isinstance(r.json(), dict) and r.json().get('ok') is True)
    except Exception as e:
        log.warning("[TELEGRAM] send raised: %s", type(e).__name__)
        return False


# ───────── API ─────────
# In-process, resets when the UTC calendar day rolls over. This is visibility
# only (single gunicorn worker per railway.json, so one counter is accurate) -
# 429s used to be logged at DEBUG, i.e. invisible under the default INFO
# level, so a plan running over its daily request cap failed silently with no
# symptom beyond fewer tips. See /admin/status -> api_usage.
_api_call_lock = threading.Lock()
_api_pace_lock = threading.Lock()
_api_next_request_ts = 0.0
API_MIN_REQUEST_INTERVAL_SEC = max(
    0.0, float(os.getenv("API_MIN_REQUEST_INTERVAL_SEC", "0.15")))
_api_call_stats = {
    "day": None, "total": 0, "rate_limited": 0,
    "daily_limit": None, "daily_remaining": None,
    "minute_limit": None, "minute_remaining": None,
    "cooldown_until_ts": 0,
}


def _track_api_call(status_code: Optional[int], headers: Optional[Any] = None) -> Dict[str, Any]:
    today = datetime.now(TZ_UTC).strftime("%Y-%m-%d")
    with _api_call_lock:
        if _api_call_stats["day"] != today:
            _api_call_stats.update(
                day=today, total=0, rate_limited=0,
                daily_limit=None, daily_remaining=None,
                minute_limit=None, minute_remaining=None,
                cooldown_until_ts=0,
            )
        _api_call_stats["total"] += 1
        if status_code == 429:
            _api_call_stats["rate_limited"] += 1
        if headers is not None:
            quota_headers = {
                "daily_limit": "x-ratelimit-requests-limit",
                "daily_remaining": "x-ratelimit-requests-remaining",
                "minute_limit": "x-ratelimit-limit",
                "minute_remaining": "x-ratelimit-remaining",
            }
            for field, header in quota_headers.items():
                try:
                    value = headers.get(header)
                    if value not in (None, ""):
                        _api_call_stats[field] = int(value)
                except (TypeError, ValueError, AttributeError):
                    continue
        return dict(_api_call_stats)


def _api_call_stats_snapshot() -> Dict[str, Any]:
    with _api_call_lock:
        out = dict(_api_call_stats)
        out["cooldown_remaining_sec"] = max(
            0, int(float(out.get("cooldown_until_ts") or 0) - time.time()))
        return out


def _api_rate_limit_cooldown(count: bool = True) -> None:
    """Stop a rate-limit response from becoming hundreds more failed calls."""
    with _api_call_lock:
        if count:
            _api_call_stats["rate_limited"] += 1
        remaining = _api_call_stats.get("minute_remaining")
        # A body-level rateLimit error while the response header still reports
        # substantial minute capacity is a burst throttle, not exhaustion of
        # the whole minute allowance. Back off briefly. Only wait across a
        # minute boundary when the provider says no capacity remains (or did
        # not supply enough information to distinguish the two cases).
        cooldown_sec = 3.0 if isinstance(remaining, int) and remaining > 0 else 65.0
        _api_call_stats["cooldown_until_ts"] = max(
            float(_api_call_stats.get("cooldown_until_ts") or 0), time.time() + cooldown_sec)


def _api_in_cooldown() -> bool:
    with _api_call_lock:
        return time.time() < float(_api_call_stats.get("cooldown_until_ts") or 0)


def _is_rate_limit_error(errors: Any) -> bool:
    if errors in (None, {}, [], ""):
        return False
    return "ratelimit" in str(errors).replace("_", "").replace(" ", "").lower() \
        or "too many requests" in str(errors).lower()


def _pace_api_request() -> bool:
    """Reserve a request-start slot so worker threads cannot hit the API in a burst."""
    global _api_next_request_ts
    if _api_in_cooldown():
        return False
    with _api_pace_lock:
        now = time.time()
        with _api_call_lock:
            minute_limit = _api_call_stats.get("minute_limit")
        dynamic_interval = (60.0 / float(minute_limit) * 1.10
                            if isinstance(minute_limit, int) and minute_limit > 0 else 0.0)
        interval = max(API_MIN_REQUEST_INTERVAL_SEC, dynamic_interval)
        slot = max(now, _api_next_request_ts)
        _api_next_request_ts = slot + interval
    delay = slot - now
    if delay > 0:
        time.sleep(delay)
    # A request already in flight may have activated cooldown while this
    # thread waited for its paced slot.
    return not _api_in_cooldown()


def _api_get(url: str, params: dict, timeout: int = 15):
    if not API_KEY:
        return None
    if not _pace_api_request():
        return None
    try:
        r = session.get(url, headers=HEADERS, params=params, timeout=timeout)
        stats = _track_api_call(r.status_code, r.headers)
        if r.ok:
            payload = r.json()
            api_errors = payload.get("errors") if isinstance(payload, dict) else None
            if api_errors not in (None, {}, [], ""):
                log.warning("[API] response errors on %s params=%s: %s",
                            url, sorted(params), api_errors)
                if _is_rate_limit_error(api_errors):
                    _api_rate_limit_cooldown(count=True)
            return payload
        if r.status_code == 429:
            _api_rate_limit_cooldown(count=False)  # already counted by _track_api_call
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

# Module-level values are used by the boot-time concentration check and status
# output. _blocked_league() re-reads the environment so administrative tests,
# workers that inject configuration before a scan, and long-lived processes do
# not retain a stale allow/deny decision.
LEAGUE_ALLOW_IDS = [x.strip() for x in os.getenv("LEAGUE_ALLOW_IDS", "").split(",") if x.strip()]
LEAGUE_DENY_IDS = [x.strip() for x in os.getenv("LEAGUE_DENY_IDS", "").split(",") if x.strip()]


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
    if league_id.isdigit() and int(league_id) in DISABLED_LEAGUE_IDS:
        return True
    allow_ids = LEAGUE_ALLOW_IDS
    if CONCENTRATION_MODE and not allow_ids:
        return True
    deny_ids = [x.strip() for x in os.getenv("LEAGUE_DENY_IDS", "").split(",") if x.strip()]
    if allow_ids:
        return league_id not in allow_ids
    txt = f"{lg.get('country','')} {lg.get('name','')} {lg.get('type','')}".lower()
    if any(p in txt for p in _BLOCK_PATTERNS):
        return True
    return league_id in deny_ids


def _enforce_concentration_scope() -> None:
    if not CONCENTRATION_MODE:
        raise SystemExit('CONCENTRATION_MODE must stay on for the frozen trial')
    scopes = [set(map(str, LEAGUE_ALLOW_IDS)), set(map(str, PREMATCH_LEAGUE_IDS))]
    if any(scopes) and (scopes[0] != scopes[1] or not 3 <= len(scopes[0]) <= MAX_ACTIVE_LEAGUES):
        raise SystemExit('Set identical 3–5 league IDs for live and prematch, or leave both empty for density selection')
    if not 1 <= len(ACTIVE_MARKETS) <= 2 or not ACTIVE_MARKETS <= {'BTTS', 'Over/Under 2.5'}:
        raise SystemExit('Choose one or both: BTTS,Over/Under 2.5')


def _select_density_scope():
    """Persist one choice from actual settled volume; never rotate on P&L."""
    global LEAGUE_ALLOW_IDS, PREMATCH_LEAGUE_IDS
    if LEAGUE_ALLOW_IDS:
        return
    saved = get_setting('research:league_scope')
    if saved:
        chosen = json.loads(saved)
    else:
        # Restricted top-flight pool; density, not historical returns, ranks it.
        pool = {39, 140, 135, 78, 61, 88, 94} - DISABLED_LEAGUE_IDS
        density = compute_league_density(days=365, min_n=20)
        chosen = [r['league_id'] for r in density['leagues']
                  if r['league_id'] in pool and not r['below_min_n']][:MAX_ACTIVE_LEAGUES]
        if len(chosen) < 3:
            log.warning('[RESEARCH] Need density for at least three leagues; scans remain blocked')
            return
        with db_conn() as c:
            c.execute("INSERT INTO settings(key,value) VALUES('research:league_scope',%s) ON CONFLICT(key) DO NOTHING",
                      (json.dumps(chosen),))
        chosen = json.loads(get_setting('research:league_scope'))
    LEAGUE_ALLOW_IDS = [str(x) for x in chosen]
    PREMATCH_LEAGUE_IDS = [int(x) for x in chosen]
    _enforce_concentration_scope()


def _research():
    return ResearchStore(sys.modules[__name__])


def _delivery_allowed(phase):
    try:
        return _research().delivery_allowed(phase)
    except Exception:
        log.exception('[RESEARCH] Delivery gate unavailable; blocking')
        return False


def _kickoff_ts_of(fx: dict) -> int:
    return int(_fixture_ts(fx) or 0)


# ───────── Live fetches ─────────
def fetch_match_stats(fid: int) -> Optional[list]:
    cached = STATS_CACHE.get(fid, _MISS)
    if cached is not _MISS:
        return cached
    js = _api_get(f"{FOOTBALL_API_URL}/statistics", {"fixture": fid})
    if not isinstance(js, dict):
        log.warning("[STATS] fixture %s fetch failed; not cached", fid)
        return None
    api_errors = js.get("errors")
    if api_errors not in (None, {}, [], ""):
        log.warning("[STATS] fixture %s returned API errors; not cached: %s", fid, api_errors)
        return None
    out = js.get("response", [])
    if not isinstance(out, list):
        log.warning("[STATS] fixture %s returned malformed payload; not cached", fid)
        return None
    # Empty is a successful no-coverage response and may be cached briefly by
    # the normal stats cache; transport/API failures are represented by None.
    STATS_CACHE.set(fid, out)
    _STATS_FETCHED_TS[int(fid)] = int(time.time())
    return out


def fetch_match_events(fid: int) -> list:
    cached = EVENTS_CACHE.get(fid, _MISS)
    if cached is not _MISS:
        return cached
    js = _api_get(f"{FOOTBALL_API_URL}/events", {"fixture": fid})
    if not isinstance(js, dict):
        return []
    api_errors = js.get("errors")
    out = js.get("response", []) if isinstance(js, dict) else []
    if not api_errors:
        EVENTS_CACHE.set(fid, out)
    return out


def fetch_live_matches(audit=None) -> List[dict]:
    js = _api_get(FOOTBALL_API_URL, {"live": "all"})
    if not isinstance(js, dict) or not isinstance(js.get("response"), list) or js.get("errors"):
        raise RuntimeError("fixture_feed_unavailable")
    observed_ts = int(time.time())
    matches = js['response']
    eligible = []
    for m in matches:
        fid = (m.get('fixture') or {}).get('id')
        if _blocked_league(m.get('league') or {}):
            if audit:
                audit.record(fid, 'fixture', 'league_excluded')
            continue
        st = ((m.get("fixture", {}) or {}).get("status", {}) or {})
        elapsed = st.get("elapsed")
        short = (st.get("short") or "").upper()
        if elapsed is None or elapsed > 90 or short not in {'1H', 'HT', '2H'}:
            if audit:
                audit.record(fid, 'fixture', 'not_regulation_live')
            continue
        m["_fixture_observed_ts"] = observed_ts
        eligible.append(m)

    def _hydrate(m: dict) -> dict:
        fid = (m.get("fixture", {}) or {}).get("id")
        try:
            # Statistics decide whether the fixture can be scored. Fetch them
            # first; events are only needed for a fixture whose stats payload
            # exists. Previously every worldwide fixture consumed both calls,
            # allowing low-coverage leagues to starve useful matches.
            stats = fetch_match_stats(fid)
            events = fetch_match_events(fid) if stats else []
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
def _num(v) -> float:
    try:
        if isinstance(v, str) and v.strip().endswith("%"):
            return float(v.strip()[:-1])
        return float(v or 0)
    except Exception:
        return 0.0


_STATS_COVERAGE_GROUPS: Dict[str, Tuple[str, ...]] = {
    "xg": ("Expected Goals", "expected_goals"),
    "shots_on_goal": ("Shots on Goal",),
    "corners": ("Corner Kicks",),
    "possession": ("Ball Possession",),
    "total_shots": ("Total Shots",),
    "shots_inside": ("Shots insidebox",),
    "passes": ("Total passes",),
    "accurate_passes": ("Passes accurate",),
    "fouls": ("Fouls",),
    "saves": ("Goalkeeper Saves",),
}


def _stat_is_present(stats: Dict[str, Any], aliases: Tuple[str, ...]) -> bool:
    """Presence is not positivity: a supplied value of zero is valid data."""
    return any(name in stats and stats.get(name) not in (None, "") for name in aliases)


def extract_raw_inplay(m: dict) -> Dict[str, Any]:
    """Pull the RAW_INPLAY_KEYS out of an API fixture object. Nothing derived."""
    home_obj = (m.get("teams") or {}).get("home") or {}
    away_obj = (m.get("teams") or {}).get("away") or {}
    home = str(home_obj.get("name") or "")
    away = str(away_obj.get("name") or "")
    home_id = int(home_obj.get("id") or 0)
    away_id = int(away_obj.get("id") or 0)
    stats_by_id: Dict[int, Dict[str, Any]] = {}
    stats_by_name: Dict[str, Dict[str, Any]] = {}
    for s in (m.get("statistics") or []):
        team = s.get("team") or {}
        values = {(i.get("type") or ""): i.get("value")
                  for i in (s.get("statistics") or []) if isinstance(i, dict) and i.get("type")}
        team_id = int(team.get("id") or 0)
        team_name = str(team.get("name") or "").strip().casefold()
        if team_id:
            stats_by_id[team_id] = values
        if team_name:
            stats_by_name[team_name] = values

    # Stable IDs are authoritative. Names remain as a compatibility fallback
    # for old/test payloads that do not carry IDs.
    sh = stats_by_id.get(home_id) if home_id else None
    sa = stats_by_id.get(away_id) if away_id else None
    sh = sh if sh is not None else stats_by_name.get(home.strip().casefold(), {})
    sa = sa if sa is not None else stats_by_name.get(away.strip().casefold(), {})
    sh = sh or {}
    sa = sa or {}

    paired_groups = [name for name, aliases in _STATS_COVERAGE_GROUPS.items()
                     if _stat_is_present(sh, aliases) and _stat_is_present(sa, aliases)]
    returned_groups = [name for name, aliases in _STATS_COVERAGE_GROUPS.items()
                       if _stat_is_present(sh, aliases) or _stat_is_present(sa, aliases)]
    xg_h_available = _stat_is_present(sh, _STATS_COVERAGE_GROUPS["xg"])
    xg_a_available = _stat_is_present(sa, _STATS_COVERAGE_GROUPS["xg"])

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
        # Serving-only metadata. build_inplay_features() ignores keys outside
        # RAW_INPLAY_KEYS, so this improves coverage decisions without changing
        # the trained feature vector or breaking train/serve parity.
        "_stats_home_found": bool(sh),
        "_stats_away_found": bool(sa),
        "_stats_response_teams": len(m.get("statistics") or []),
        "_stats_paired_fields": len(paired_groups),
        "_stats_paired_field_names": paired_groups,
        "_stats_returned_fields": len(returned_groups),
        "_stats_returned_field_names": returned_groups,
        "_stats_home_field_names": sorted(sh),
        "_stats_away_field_names": sorted(sa),
        "_xg_h_available": xg_h_available,
        "_xg_a_available": xg_a_available,
        "_fixture_observed_ts": m.get("_fixture_observed_ts"),
    }


def _complete_live_features(
    m: dict, raw: Dict[str, Any], *, fetch_market_price: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """Finish a raw statistics row without wasting odds calls on unusable games."""
    fid = int((m.get("fixture") or {}).get("id") or 0)
    # Market prices are execution inputs only; never classifier features.
    league_id = ((m.get("league") or {}).get("id"))
    lr = get_league_rates(int(league_id) if league_id else None)
    return raw, build_inplay_features(raw, lr)


def extract_features(m: dict) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Returns (raw, features). Features come from feature_spec, shared with training."""
    return _complete_live_features(m, extract_raw_inplay(m))


def stats_coverage_ok(raw: Dict[str, Any], minute: int) -> bool:
    """Require real statistic fields, while accepting legitimate zero values."""
    require_from = int(os.getenv("REQUIRE_STATS_MINUTE", str(TIP_MIN_MINUTE)))
    require_fields = int(os.getenv("REQUIRE_DATA_FIELDS", "2"))
    if minute < require_from:
        return False
    if "_stats_paired_fields" in raw:
        return (bool(raw.get("_stats_home_found"))
                and bool(raw.get("_stats_away_found"))
                and int(raw.get("_stats_returned_fields") or 0) >= max(0, require_fields))
    # Compatibility for historical/test dictionaries without presence
    # metadata. New live payloads always use the branch above.
    fields = [raw.get("xg_h", 0) + raw.get("xg_a", 0),
              raw.get("sot_h", 0) + raw.get("sot_a", 0),
              raw.get("cor_h", 0) + raw.get("cor_a", 0),
              max(raw.get("pos_h", 0), raw.get("pos_a", 0))]
    return sum(1 for v in fields if (v or 0) > 0) >= max(0, require_fields)


def _stats_coverage_details(raw: Dict[str, Any], minute: int) -> Dict[str, Any]:
    require_from = int(os.getenv("REQUIRE_STATS_MINUTE", str(TIP_MIN_MINUTE)))
    require_fields = int(os.getenv("REQUIRE_DATA_FIELDS", "2"))
    covered = stats_coverage_ok(raw, minute)
    if minute < require_from:
        reason = "before_required_minute"
    elif not raw.get("_stats_home_found") or not raw.get("_stats_away_found"):
        reason = "missing_team_statistics"
    elif int(raw.get("_stats_returned_fields") or 0) < max(0, require_fields):
        reason = "too_few_returned_fields"
    else:
        reason = "usable"
    return {
        "covered": bool(covered), "reason": reason,
        "response_teams": int(raw.get("_stats_response_teams") or 0),
        "returned_field_count": int(raw.get("_stats_returned_fields") or 0),
        "returned_fields": list(raw.get("_stats_returned_field_names") or []),
        "paired_field_count": int(raw.get("_stats_paired_fields") or 0),
        "paired_fields": list(raw.get("_stats_paired_field_names") or []),
        "home_fields": list(raw.get("_stats_home_field_names") or []),
        "away_fields": list(raw.get("_stats_away_field_names") or []),
        "xg_home_available": bool(raw.get("_xg_h_available")),
        "xg_away_available": bool(raw.get("_xg_a_available")),
    }


def prematch_data_gate(feat: Dict[str, float]) -> Optional[str]:
    """
    The prematch counterpart of stats_coverage_ok(): did any form data arrive?
    Returns None when the fixture was genuinely observed, else the reason.

    The in-play path refuses an empty observation. The prematch path has no
    equivalent — its only check is `if not feat`, and
    assemble_prematch_features() ends with

        out = {k: float(f.get(k, 0.0)) for k in PRE_FEATURES}

    so it ALWAYS returns a fully-populated dict and `feat` is never falsy. A
    fixture whose team-form fetches all failed therefore arrives as a complete
    vector of zeros, is written to prematch_snapshots, and is read back by
    load_prematch_data() as a real observation.

    Two things make that worse than one bad row. Every fixture in the same
    outage gets the SAME vector — two nameless 1500-Elo sides with no history —
    so a fit sees a large block of identical inputs carrying whatever label each
    fixture happened to produce, which is unlearnable noise. And
    prematch_snapshots upserts on match_id, so a rescan during a rate-limit
    cooldown CLOBBERS a good snapshot that was already there.

    A team that genuinely played matches cannot have gf, ga, win and draw all
    exactly zero: every finished game lands in exactly one of the three
    outcomes, and a win moves `win`, a draw moves `draw`, and a defeat concedes
    at least one goal and so moves `ga`. All four at zero therefore means the
    window was empty. That is a test for the ABSENCE of an observation, not for
    a thin one — which is why, unlike a betting gate, it is allowed to stop a
    harvest.
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
    p = max(1e-6, min(1 - 1e-6, float(p)))  # Matches training Platt clipping.
    return math.log(p / (1 - p))


def load_model_from_settings(name: str) -> Optional[Dict[str, Any]]:
    audit = getattr(_SCAN_LOCAL, 'audit', None)
    cached = audit.models.get(name, _MISS) if audit is not None else _MODELS_CACHE.get(name, _MISS)
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
            if tmp.get('feature_schema_version') != FEATURE_SCHEMA_VERSION or any(
                    'market_fair_' in k for k in tmp.get('weights', {})):
                log.warning('[MODEL] %s needs retraining with price-free schema', name)
                continue
            mdl = tmp
            break
        except Exception as e:
            log.warning("[MODEL] parse %s failed: %s", name, e)
    if audit is not None:
        audit.models[name] = mdl
    else:
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
    linear = _linpred(feat, mdl)
    if not math.isfinite(linear):
        raise ValueError("non-finite model score")
    p = _sigmoid(linear)
    cal = mdl.get("calibration") or {}
    if cal:
        try:
            p = _calibrate(p, cal)
        except Exception as e:
            # Falling back to the UNCALIBRATED probability changes what the
            # number means while it still gets compared against the same
            # threshold, so leave a trace rather than swallowing it whole.
            raise ValueError("model calibration failed") from e
    if not math.isfinite(p):
        raise ValueError("non-finite calibrated probability")
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


from odds_parser import (
    _txt, _market_name_normalize, _iter_price_sources, _odd_value,
    _price_suspended, LIVE_FEED_BOOK, _MARKET_SELECTION_COUNT, parse_book_market,
)


def _parse_book_market(mkt):
    return parse_book_market(mkt, ou_lines=OU_LINES)


def _selection_executability(book_prices: Dict[str, float], min_books: int) -> Dict[str, Any]:
    """
    Whether a selection's BEST price is real and takeable, as distinct from
    whether it is a good price to bet against (that's the fair-price/EV gate
    below).

    fetch_odds() keeps a running max() over whichever books happen to be
    quoting a given selection. A single stale or fat-fingered quote wins that
    max unconditionally, and the winning number becomes both the tip's
    recorded price and — via compute_pnl/compute_market_significance/
    monte_carlo_bankroll — the price it is graded against. That is the same
    failure class as the team-totals contamination fixed elsewhere in this
    file (a wrong candidate winning a max() it should never have been
    eligible for), one level down: there the wrong MARKET won: here a single
    uncorroborated BOOK does.

    Two independent, cheap checks:
      - corroboration: at least min_books distinct books quote this EXACT
        selection (not just the market — a book can quote Home/Away in a
        1X2 market while never quoting Draw at all).
      - outlier spread: the best price does not exceed the second-best price
        by more than MAX_EXECUTION_OUTLIER_PCT. With fewer than two prices
        there is nothing to compare, so this check is silently skipped
        (not failed) — corroboration alone governs that case.
    """
    n = len(book_prices)
    if n == 0:
        return {"n_books": 0, "best_odds": None, "second_best_odds": None,
                "outlier_pct": None, "executable": False}
    sorted_odds = sorted(book_prices.values(), reverse=True)
    best_o = sorted_odds[0]
    second_o = sorted_odds[1] if n >= 2 else None
    outlier_pct = None
    outlier_ok = True
    if second_o is not None and second_o > 0:
        outlier_pct = round((best_o / second_o - 1.0) * 100.0, 2)
        outlier_ok = outlier_pct <= MAX_EXECUTION_OUTLIER_PCT
    executable = (n >= min_books) and outlier_ok
    return {"n_books": n, "best_odds": round(best_o, 4),
            "second_best_odds": round(second_o, 4) if second_o is not None else None,
            "outlier_pct": outlier_pct, "executable": executable}


# The second provider is opt-in; evaluation makes requests only via admin POST.
from the_odds_feed import OddsFeed
THE_ODDS_API_MODE = os.getenv("THE_ODDS_API_MODE", "evaluation").strip().lower()
if THE_ODDS_API_MODE not in ("off", "evaluation", "supplement"):
    raise SystemExit("THE_ODDS_API_MODE must be off, evaluation or supplement")
THE_ODDS_API_DAILY_CREDITS = max(0, int(os.getenv("THE_ODDS_API_DAILY_CREDITS", "10")))
THE_ODDS_API_MONTHLY_CREDITS = max(0, int(os.getenv("THE_ODDS_API_MONTHLY_CREDITS", "100")))


def _reserve_odds_credits(cost):
    # Count attempted paid requests conservatively, even failures. Calendar UTC
    # budgets survive restarts and share a lock across workers/replicas.
    day = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    month = day[:7]
    with _tip_transaction() as c:
        c.execute('SELECT pg_advisory_xact_lock(19022)')
        row = c.execute("SELECT value FROM settings WHERE key='odds_api:budget'").fetchone()
        budget = json.loads(row[0]) if row else {}
        daily = int(budget.get('daily', 0)) if budget.get('day') == day else 0
        monthly = int(budget.get('monthly', 0)) if budget.get('month') == month else 0
        if daily + cost > THE_ODDS_API_DAILY_CREDITS or monthly + cost > THE_ODDS_API_MONTHLY_CREDITS:
            return False
        value = json.dumps(dict(day=day, month=month, daily=daily+cost, monthly=monthly+cost))
        c.execute("INSERT INTO settings(key,value) VALUES('odds_api:budget',%s) "
                  "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value", (value,))
    return True


_THE_ODDS_FEED = OddsFeed(os.getenv('THE_ODDS_API_KEY', '').strip(), _reserve_odds_credits)


def _external_odds_rows(fid):
    try:
        js = _api_get(FOOTBALL_API_URL, {'id': int(fid)})
        fixtures = js.get('response', []) if isinstance(js, dict) else []
        if len(fixtures) != 1 or fixtures[0].get('fixture', {}).get('id') != int(fid):
            _THE_ODDS_FEED.status.update(status='fixture_lookup_failed', fixture_id=int(fid),
                                         football_api_results=len(fixtures))
            return []
        if str(fixtures[0].get('league', {}).get('id')) not in set(map(str, PREMATCH_LEAGUE_IDS)):
            _THE_ODDS_FEED.status.update(status='league_outside_scope', fixture_id=int(fid),
                                         fixture_league_id=fixtures[0].get('league', {}).get('id'))
            return []
        _THE_ODDS_FEED.status['fixture_id'] = int(fid)
        return _THE_ODDS_FEED.rows(fixtures[0])
    except Exception:
        _THE_ODDS_FEED.status.update(status='fixture_lookup_failed', fixture_id=int(fid))
        log.exception('[THE_ODDS_API] fixture lookup failed; external quotes ignored')
        return []


def fetch_odds(fid: int, live: bool) -> Dict[str, Any]:
    """
    Returns, per market key:
      {"best": {selection: {"odds": float, "book": str}},
       "fair": {selection: float},          # consensus de-vigged probability
       "n_books": int,
       "by_book": {selection: {book_name: odds}},
       "executability": {selection: {                    # NEW
           "n_books": int, "best_odds": float, "second_best_odds": float|None,
           "outlier_pct": float|None, "executable": bool}}}

    De-vigging happens WITHIN each bookmaker's complete market (de-vigging
    across best-of-many-books prices would produce a fake sub-1.0 overround and
    a systematically optimistic fair price), then averages across books. The
    best available price across all books is used separately for EV.

    FIX: the market total is now looked up per market rather than assumed to be
    1.0. Double Chance sums to 2.0 — see feature_spec.devig().

    "executability" governs whether the price in "best" is corroborated and
    non-outlying enough to actually bet at and grade against — separate from
    whether "fair"/n_books says it's a good price to bet against. See
    _selection_executability() and _price_gate()'s use of this block.
    """
    key = (fid, bool(live))
    cached = ODDS_CACHE.get(key, _MISS)
    if cached is not _MISS:
        return cached

    params: Dict[str, Any] = {"fixture": fid}
    # Fetch all sources: execution and sharp benchmark are separate books.
    js = _api_get(ODDS_LIVE_URL if live else ODDS_PREMATCH_URL, params)
    external_rows = []
    if not live and THE_ODDS_API_MODE == 'supplement':
        external_rows = _external_odds_rows(fid)
        if external_rows:
            original = js.get('response', []) if isinstance(js, dict) and not js.get('errors') else []
            js = {'response': external_rows + (original if isinstance(original, list) else [])}
    fetched_ts = int(time.time())
    diagnostics = {"fetched_ts": fetched_ts, "status": "ok", "markets": []}
    diagnostics['external_provider'] = dict(_THE_ODDS_FEED.status) if not live and THE_ODDS_API_MODE == 'supplement' else {'status': 'not_used'}
    ODDS_DIAGNOSTICS.set(key, diagnostics)
    if not isinstance(js, dict):
        diagnostics["status"] = "api_unavailable"
        return {}
    api_errors = js.get("errors")
    if api_errors not in (None, {}, [], ""):
        diagnostics["status"] = "api_error"
        return {}

    best: Dict[str, Dict[str, Dict[str, Any]]] = {}
    by_book: Dict[str, Dict[str, Dict[str, float]]] = {}
    fair_acc: Dict[str, Dict[str, List[float]]] = {}
    books_seen: Dict[str, set] = {}
    fair_books_seen: Dict[str, set] = {}
    parse_errors = 0
    book_updates = {}

    # FIX: the try/except used to wrap the ENTIRE response, and its handler
    # reset best/fair/books to {}. So a single malformed value, in a single
    # market, from a single bookmaker, threw away every price for that fixture —
    # all markets, all books. That is why 530 parse failures produced zero
    # priced candidates rather than merely degraded ones. Failures are now
    # isolated to the market that caused them; everything else survives.
    response = js.get("response", []) if isinstance(js, dict) else []
    if not isinstance(response, list):
        diagnostics["status"] = "malformed_response"
        return {}
    if not response:
        diagnostics["status"] = "no_markets_returned"
    for r in response:
        if not isinstance(r, dict):
            continue
        status = r.get("status") or {}
        if live and isinstance(status, dict) and _price_suspended(status):
            diagnostics["status"] = "fixture_odds_suspended"
            log.info("[ODDS] fixture %s: feed-level suspension; prices ignored", fid)
            continue
        for book_name, bets in _iter_price_sources(r):
            per_market: Dict[str, Dict[str, float]] = {}
            for mkt in bets:
                if not isinstance(mkt, dict):
                    continue
                try:
                    parsed = _parse_book_market(mkt)
                except Exception as e:
                    parse_errors += 1
                    log.debug("[ODDS] fixture %s book %s market %r unparseable: %s",
                              fid, book_name, mkt.get("name"), e)
                    continue
                if len(diagnostics["markets"]) < 80:
                    normalized = _market_name_normalize(mkt.get("name"))
                    values = mkt.get("values")
                    diagnostics["markets"].append({
                        "name": _txt(mkt.get("name")), "book": book_name,
                        "normalized": normalized or None,
                        "reason": ("accepted" if parsed else "suspended" if _price_suspended(mkt)
                                   else "unsupported_market" if not normalized
                                   else "incomplete_or_invalid_selections"),
                        "values": [{k: v.get(k) for k in ("value", "odd", "handicap", "suspended")}
                                   for v in (values[:8] if isinstance(values, list) else [])
                                   if isinstance(v, dict)],
                    })
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
                    if book_name in books_seen.get(mkey, set()):
                        continue  # A repeated source must not get two consensus votes.
                    books_seen.setdefault(mkey, set()).add(book_name)
                    book_updates.setdefault(mkey, {})[book_name] = r.get("update")
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
                    source_ts = source_timestamp(r.get('update'))
                    fair_max_age = LIVE_MAX_ODDS_AGE_SEC if live else 300
                    # Fair price is the sharp benchmark used by release
                    # evidence.  Do not average the execution book into it:
                    # doing so mixes Tipico margin with Pinnacle skill and
                    # makes the benchmark depend on feed composition.
                    if (book_name == SHARP_BOOK and len(sel) >= needed and
                            source_ts is not None and 0 <= fetched_ts - source_ts <= fair_max_age):
                        fair_books_seen.setdefault(mkey, set()).add(book_name)
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
        diagnostic_samples = []
        for r in response[:3]:
            if not isinstance(r, dict):
                continue
            for _bk_name, _bets in _iter_price_sources(r):
                for b in _bets:
                    if not isinstance(b, dict):
                        continue
                    raw_name = _txt(b.get("name"))
                    offered.append(raw_name)
                    normalized = _market_name_normalize(raw_name)
                    if normalized not in ("1X2", "OU", "BTTS", "DC", "DNB"):
                        continue
                    if len(diagnostic_samples) >= 4:
                        continue
                    values_sample = []
                    vals = b.get("values")
                    if isinstance(vals, list):
                        for v in vals[:6]:
                            if not isinstance(v, dict):
                                values_sample.append({"type": type(v).__name__})
                                continue
                            values_sample.append({
                                k: v.get(k)
                                for k in ("value", "odd", "handicap", "main", "suspended")
                                if k in v
                            })
                    else:
                        values_sample.append({"values_type": type(vals).__name__})
                    diagnostic_samples.append({
                        "name": raw_name,
                        "normalized": normalized,
                        "values": values_sample,
                    })
        log.warning("[ODDS] fixture %s (live=%s): %d response item(s) but no usable markets. "
                    "Top-level keys: %s. Markets offered: %s. Unparsed full-match samples: %s",
                    fid, live, len(response),
                    sorted(response[0].keys()) if isinstance(response[0], dict) else type(response[0]),
                    sorted(set(offered)) or "none", diagnostic_samples or "none")

    min_books_exec = MIN_BOOKS_FOR_EXECUTION_LIVE if live else MIN_BOOKS_FOR_EXECUTION
    out: Dict[str, Any] = {}
    for mkey, sels in best.items():
        fair = {k: (sum(v) / len(v)) for k, v in (fair_acc.get(mkey) or {}).items() if v}
        by_book_mkey = by_book.get(mkey, {})
        fresh_books = fair_books_seen.get(mkey, set())
        executability = {name: _selection_executability(
            {book: price for book, price in by_book_mkey.get(name, {}).items() if book in fresh_books}, min_books_exec)
                         for name in sels}
        source_updates = [_txt(r.get("update")) for r in response
                          if isinstance(r, dict) and r.get("update") not in (None, "")]
        out[mkey] = {"best": sels, "fair": fair, "n_books": len(books_seen.get(mkey, ())),
                     "n_fair_books": len(fair_books_seen.get(mkey, ())),
                     "by_book": by_book_mkey, "executability": executability,
                     "book_updates": book_updates.get(mkey, {}),
                     "fetched_ts": fetched_ts,
                     "source_update": source_updates[0] if source_updates else None}
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

    Falls back to NEUTRAL_MARKET_PRIORS per-market wherever odds are
    unavailable or too thin to devig (see that constant for why).
    """
    out = dict(NEUTRAL_MARKET_PRIORS)
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


class PriceCheck(dict):
    """Result of _price_gate. Dict so it serialises straight into the log row."""


def _price_gate(market_text: str, suggestion: str, fid: int, prob: float, live: bool) -> PriceCheck:
    """
    Single place where a candidate meets the market.

    Gates, in order:
      1. odds exist (unless ALLOW_TIPS_WITHOUT_ODDS)
      2. odds within [min_for_market, MAX_ODDS_ALL]
      3. a de-vigged fair price is computable (unless REQUIRE_FAIR_PRICE=0)
      4. the best price is EXECUTABLE — corroborated by enough books and not
         an outlier spread against the next-best quote (unless
         REQUIRE_EXECUTABLE_PRICE=0). See fetch_odds()'s "executability"
         block and the MIN_BOOKS_FOR_EXECUTION(_LIVE) / MAX_EXECUTION_OUTLIER_PCT
         module comment for why this is separate from the fair-price check.
      5. EV at the available price >= EDGE_MIN_BPS
      6. edge over the fair price >= FAIR_EDGE_MIN_BPS
      7. edge over the fair price <= MAX_MODEL_EDGE_BPS  (model-sanity cap)
    """
    res = PriceCheck(passed=False, odds=None, book=None, fair_prob=None,
                     ev_pct=None, decision="no_odds", n_books=0)
    if not math.isfinite(float(prob)) or not 0.0 <= prob <= 1.0:
        res["decision"] = "invalid_probability"
        return res
    mkey, sel = _market_key_and_selection(market_text, suggestion)
    if not mkey or not sel:
        res["decision"] = "unmapped_market"
        return res

    odds_map = fetch_odds(fid, live=live) if API_KEY else {}
    entry = odds_map.get(mkey) or {}
    execution_odds = ((entry.get('by_book') or {}).get(sel) or {}).get(EXECUTION_BOOK)
    best = {'odds': execution_odds, 'book': EXECUTION_BOOK} if execution_odds else None
    res["n_books"] = int(entry.get("n_fair_books", entry.get("n_books")) or 0)

    if not best:
        res["decision"] = "no_odds"
        res["passed"] = bool(ALLOW_TIPS_WITHOUT_ODDS)
        return res

    odds = float(best["odds"])
    if not math.isfinite(odds) or odds <= 1.0:
        res["decision"] = "invalid_odds"
        return res
    res["odds"] = odds
    res["book"] = best.get("book")
    res["odds_fetched_ts"] = entry.get("fetched_ts")
    res["odds_source_update"] = (entry.get("book_updates") or {}).get(EXECUTION_BOOK)
    res["model_prob"] = float(prob)
    now = time.time()
    fetched = entry.get('fetched_ts')
    max_age = LIVE_MAX_ODDS_AGE_SEC if live else 300
    if fetched is None or not 0 <= now - float(fetched) <= max_age:
        res['decision'] = 'stale_odds'
        return res
    updated = source_timestamp(res['odds_source_update'])
    if updated is None or not 0 <= now - updated <= max_age:
        res['decision'] = 'missing_or_stale_book_timestamp'
        return res

    if not (_min_odds_for_market(market_text.replace("PRE ", "")) <= odds <= MAX_ODDS_ALL):
        res["decision"] = "odds_out_of_range"
        return res
    if not live and odds > MAX_PREMATCH_ODDS_ALL:
        res["decision"] = "prematch_odds_implausible"
        return res

    fair = (entry.get("fair") or {}).get(sel)
    if fair is not None and (not math.isfinite(float(fair)) or not 0.0 < float(fair) < 1.0):
        res["decision"] = "invalid_fair_probability"
        return res
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

    if fair is not None and float(fair) > 0:
        fair_odds = 1.0 / float(fair)
        price_vs_fair_pct = (odds / fair_odds - 1.0) * 100.0
        res["price_vs_fair_pct"] = round(price_vs_fair_pct, 2)
        if price_vs_fair_pct > MAX_PRICE_VS_FAIR_PCT:
            res["decision"] = "price_vs_fair_outlier"
            log.warning("[PRICE] fixture %s %s: %.3f is %.1f%% above fair %.3f — suppressed",
                        fid, suggestion, odds, price_vs_fair_pct, fair_odds)
            return res

    # Current fetch_odds() entries always carry selection-level execution
    # metadata. Tolerate legacy/injected entries without it so a rolling
    # deploy cannot strand already-cached prices; new API results still pass
    # through both execution checks below.
    exec_info = (entry.get("executability") or {}).get(sel)
    if exec_info is not None:
        min_books_exec = MIN_BOOKS_FOR_EXECUTION_LIVE if live else MIN_BOOKS_FOR_EXECUTION
        res["execution_n_books"] = int(exec_info.get("n_books") or 0)
        res["execution_outlier_pct"] = exec_info.get("outlier_pct")
        if not exec_info.get("executable", False):
            res["decision"] = ("too_few_books_for_execution"
                               if res["execution_n_books"] < min_books_exec
                               else "execution_price_outlier")
            if REQUIRE_EXECUTABLE_PRICE:
                return res

    if fair is None:
        res['decision'] = 'market_probability_required_for_shrinkage'
        return res
    prob = shrink_probability(prob, float(fair))
    res['effective_prob'] = prob
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


def _stake_units(prob: float, odds: Optional[float], fair: Optional[float] = None) -> Optional[float]:
    """Market-shrunk fractional Kelly; missing market means no stake."""
    if not odds or fair is None:
        return None
    f = kelly_fraction(shrink_probability(prob, fair), odds) * KELLY_FRACTION
    f = max(0.0, min(f, MAX_STAKE_PCT / 100.0))
    return round(BANKROLL_UNITS * f, 2)


def _live_delivery_check(fid: int, raw: Dict[str, Any], pc: PriceCheck) -> PriceCheck:
    """Final score/clock check. No synthetic prices or predictions are substituted."""
    now = time.time()
    for timestamp in (raw.get("_fixture_observed_ts"), _STATS_FETCHED_TS.get(fid)):
        if timestamp is None or not 0 <= now - float(timestamp) <= LIVE_MAX_INPUT_AGE_SEC:
            return PriceCheck(**{**pc, "passed": False, "decision": "stale_live_inputs"})
    current = _fixture_by_id(fid)
    if not current:
        return PriceCheck(**{**pc, "passed": False, "decision": "fixture_recheck_failed"})
    status = (current.get("fixture") or {}).get("status") or {}
    goals = current.get("goals") or {}
    elapsed = status.get("elapsed")
    if (status.get("short") not in {"1H", "HT", "2H"} or elapsed is None
            or not 0 <= float(elapsed) - float(raw["minute"]) <= 1
            or float(elapsed) > 90
            or goals.get("home") is None or goals.get("away") is None
            or float(goals["home"]) != float(raw["goals_h"])
            or float(goals["away"]) != float(raw["goals_a"])):
        return PriceCheck(**{**pc, "passed": False, "decision": "fixture_state_changed"})
    if any(time.time() - float(ts) > LIVE_MAX_INPUT_AGE_SEC for ts in
           (raw["_fixture_observed_ts"], _STATS_FETCHED_TS[fid])):
        return PriceCheck(**{**pc, "passed": False, "decision": "stale_live_inputs"})
    if time.time() - float(pc.get("odds_fetched_ts") or 0) > LIVE_MAX_ODDS_AGE_SEC:
        return PriceCheck(**{**pc, "passed": False, "decision": "stale_odds"})
    pc["fixture_rechecked_ts"] = int(time.time())
    return pc


def _tip_audit_json(fid: int, phase: str, feat: Dict[str, float],
                    raw: Optional[Dict[str, float]],
                    candidates: List[Tuple[str, str, float, float]],
                    pc: PriceCheck, decision_ts: int) -> str:
    """Permanent evidence for reconstructing exactly why a tip was emitted."""
    market_probabilities = {
        suggestion: {"market": market, "prob": round(float(prob), 8),
                     "threshold_pct": round(float(threshold), 4)}
        for market, suggestion, prob, threshold in candidates
    }
    prefix = "PRE_" if phase == "prematch" else ""
    wld = _wld_probs(feat, prefix)
    if wld is not None:
        market_probabilities["1X2 full vector"] = {
            "home": round(wld[0], 8), "draw": round(wld[1], 8),
            "away": round(wld[2], 8)}
    payload = {
        "schema_version": 1,
        "fixture_id": int(fid),
        "phase": phase,
        "decision_ts": int(decision_ts),
        "stats_fetched_ts": _STATS_FETCHED_TS.get(int(fid)) if phase == "live" else None,
        "odds_fetched_ts": pc.get("odds_fetched_ts"),
        "odds_source_update": pc.get("odds_source_update"),
        "price_check": dict(pc),
        "market_probabilities": market_probabilities,
        "features": {k: float(v) for k, v in feat.items()},
        "raw_stats": dict(raw) if raw else None,
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True, allow_nan=False)


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


def _model_version() -> str:
    """Training timestamp plus deployed commit, frozen with each prospective pick."""
    raw = get_setting_cached("model_metrics_latest")
    try:
        trained = (json.loads(raw) if raw else {}).get("trained_at_utc") or "unknown"
    except (ValueError, TypeError):
        trained = "unknown"
    return f"{trained}|{os.getenv('RAILWAY_GIT_COMMIT_SHA', 'unknown')[:12]}"


def _save_funnel(phase: str, counts: Dict[str, int]) -> None:
    try:
        with db_conn() as c:
            c.execute("INSERT INTO scan_funnel(created_ts,phase,counts) VALUES(%s,%s,%s)",
                      (int(time.time()), phase, json.dumps(counts, sort_keys=True)))
    except Exception as e:
        log.warning("[FUNNEL] save failed: %s", e)
    log.info("[FUNNEL] %s %s", phase, counts)


def _shadow_record(fid: int, league_id: int, league: str, kickoff: int,
                   phase: str, minute: int, market: str, suggestion: str,
                   prob: float, threshold: float, pc: PriceCheck,
                   model_version: Optional[str] = None, policy_version: str = "legacy") -> bool:
    """One immutable first qualifying quote per fixture/phase/selection."""
    if not SHADOW_ENABLE or not pc.get("passed") or not pc.get("odds"):
        return False
    with db_conn() as c:
        inserted = c.execute("""
            INSERT INTO shadow_picks(match_id,league_id,league,kickoff_ts,created_ts,
                phase,minute,market,suggestion,prob,threshold_pct,odds,book,fair_prob,
                ev_pct,stake_units,model_version,policy_version,model_prob)
            VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT(match_id,phase,suggestion) DO NOTHING RETURNING id
        """, (fid, league_id, league, kickoff, int(time.time()), phase, minute,
              market, suggestion, float(prob), float(threshold), float(pc["odds"]),
              pc.get("book"), pc.get("fair_prob"), pc.get("ev_pct"),
              _stake_units(pc.get("model_prob", prob), pc.get("odds"), pc.get("fair_prob")), model_version or _model_version(), policy_version, pc.get("model_prob", prob))).fetchone()
    return bool(inserted)


class ScanAudit:
    """Durable terminal decisions. A crashed scan remains visibly incomplete."""
    def __init__(self, phase: str, connection=None):
        self.phase = phase
        self.scan_id = uuid.uuid4().hex
        with (nullcontext(connection) if connection is not None else db_conn()) as c:
            self.settings = dict(c.execute("SELECT key,value FROM settings WHERE key LIKE %s OR key LIKE %s "
                                           "OR key LIKE %s OR key=%s",
                                           ('model_latest:%', 'model:%', 'research_threshold:%', 'model_metrics_latest')).fetchall())
        self.models = {}
        try:
            metrics = json.loads(self.settings.get('model_metrics_latest') or '{}')
            trained = metrics.get('trained_at_utc') or 'unknown'
        except (ValueError, TypeError, AttributeError):
            trained = 'unknown'
        model_hash = hashlib.sha256(json.dumps({k:v for k,v in self.settings.items()
            if k.startswith(('model:', 'model_latest:'))}, sort_keys=True).encode()).hexdigest()[:16]
        self.model_version = f"{trained}|{os.getenv('RAILWAY_GIT_COMMIT_SHA', 'unknown')[:12]}|{model_hash}"
        policy = {k: globals().get(k) for k in (
            'EDGE_MIN_BPS', 'FAIR_EDGE_MIN_BPS', 'MAX_MODEL_EDGE_BPS',
            'MIN_BOOKS_FOR_FAIR', 'MIN_BOOKS_FOR_FAIR_LIVE',
            'EXECUTION_BOOK', 'SHARP_BOOK', 'MODEL_WEIGHT',
            'MAX_TIPS_PER_SCAN', 'MAX_PREMATCH_TIPS_PER_SCAN', 'PREDICTIONS_PER_MATCH',
            'MIN_ODDS_OU', 'MIN_ODDS_BTTS', 'MIN_ODDS_1X2', 'MIN_ODDS_DC', 'MIN_ODDS_DNB',
            'MAX_ODDS_ALL', 'MAX_PREMATCH_ODDS_ALL', 'MAX_PRICE_VS_FAIR_PCT',
            'MIN_BOOKS_FOR_EXECUTION', 'MIN_BOOKS_FOR_EXECUTION_LIVE',
            'MAX_EXECUTION_OUTLIER_PCT', 'REQUIRE_EXECUTABLE_PRICE', 'REQUIRE_FAIR_PRICE',
            'LIVE_MAX_INPUT_AGE_SEC', 'LIVE_MAX_ODDS_AGE_SEC', 'TIP_MIN_MINUTE',
            'CORRELATED_EXTRA_EV_BPS', 'DUP_COOLDOWN_MIN', 'PREMATCH_DEDUP_ENABLE',
            'LEAGUE_ALLOW_IDS', 'LEAGUE_DENY_IDS', 'PREMATCH_LEAGUE_IDS',
            'THE_ODDS_API_MODE', 'THE_ODDS_API_DAILY_CREDITS', 'THE_ODDS_API_MONTHLY_CREDITS')}
        # Source-file hashes are deliberately excluded.  A bug fix or logging
        # change must not invalidate an already-frozen prospective cohort.
        # Decision constants/settings below are the policy identity; the
        # deployed git commit is retained separately in model_version/build
        # info for audit and rollback.
        policy['policy_schema'] = 'price-policy-v2'
        policy['thresholds'] = {k:v for k,v in self.settings.items() if k.startswith('research_threshold:')}
        policy['active'] = sorted(ACTIVE_MARKETS - DISABLED_MARKETS)
        policy['disabled_leagues'] = sorted(DISABLED_LEAGUE_IDS)
        self.policy_version = hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()[:16]
        self.counts = defaultdict(int)
        self.errors = 0

    def __enter__(self):
        with db_conn() as c:
            c.execute("INSERT INTO scan_runs(scan_id,phase,started_ts,status,model_version,policy_version) "
                      "VALUES(%s,%s,%s,'running',%s,%s)",
                      (self.scan_id, self.phase, int(time.time()), self.model_version, self.policy_version))
        self.previous_audit = getattr(_SCAN_LOCAL, 'audit', None)
        _SCAN_LOCAL.audit = self
        return self

    def record(self, fid, stage, reason, *, league_id=None, kickoff=None,
               market=None, suggestion=None, prob=None, threshold=None,
               pc=None, qualified=False, detail=None):
        pc = pc or {}
        with db_conn() as c:
            c.execute("""INSERT INTO scan_decisions(scan_id,created_ts,match_id,league_id,kickoff_ts,
                stage,market,suggestion,reason,prob,threshold_pct,odds,fair_prob,ev_pct,
                price_decision,qualified,detail)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (self.scan_id, int(time.time()), fid, league_id, kickoff, stage, market,
                 suggestion, reason, prob if prob is None or math.isfinite(prob) else None,
                 threshold if threshold is None or math.isfinite(threshold) else None,
                 pc.get('odds'), pc.get('fair_prob'), pc.get('ev_pct'),
                 'qualified' if pc.get('passed') else pc.get('decision'), bool(qualified), detail))
        self.counts[f'{stage}:{reason}'] += 1
        if reason in ('fixture_error', 'candidate_error', 'scan_error'):
            self.errors += 1

    def __exit__(self, typ, error, tb):
        try:
            status = 'failed' if typ else 'completed_with_errors' if self.errors else 'completed'
            if typ:
                log.error('[AUDIT] scan %s failed: %s', self.scan_id, typ.__name__)
                self.record(None, 'scan', 'scan_error', detail=typ.__name__)
            with db_conn() as c:
                c.execute("UPDATE scan_runs SET finished_ts=%s,status=%s WHERE scan_id=%s",
                          (int(time.time()), status, self.scan_id))
            _save_funnel(self.phase, dict(self.counts))
        finally:
            _SCAN_LOCAL.audit = self.previous_audit
        return False


def _send_reserved_tip(fid, created_ts, text):
    """Serialize inline and retry deliveries for one stored tip.

    Telegram has no idempotency key: a crash after remote acceptance but
    before the local commit remains an ambiguous-delivery case.
    """
    with db_conn() as check_conn:
        check = check_conn.execute('SELECT is_prematch FROM tips WHERE match_id=%s AND created_ts=%s', (fid, created_ts)).fetchone()
    if check is None or not _delivery_allowed('prematch' if check[0] else 'live'):
        return False
    with _tip_transaction() as c:
        c.execute('SELECT pg_advisory_xact_lock(%s,%s)', (19019, fid))
        row = c.execute('SELECT sent_ok,is_prematch FROM tips WHERE match_id=%s AND created_ts=%s',
                        (fid, created_ts)).fetchone()
        if row is None or int(row[0]) != 0:
            return bool(row and int(row[0]) == 1)
        try:
            sent = send_telegram(text)
        except Exception:
            log.exception('[DELIVERY] Telegram call failed for fixture %s', fid)
            sent = False
        if sent:
            c.execute('UPDATE tips SET sent_ok=1,telegram_sent_ts=%s WHERE match_id=%s AND created_ts=%s',
                      (int(time.time()), fid, created_ts))
        return bool(sent)


def _save_candidate_tip(fid, league_id, league, home, away, kickoff, phase, minute,
                        score, raw, feat, candidates, market, suggestion, prob, pc):
    """Reserve and insert atomically; only the worker owning the insert can send."""
    if not _delivery_allowed(phase):
        return 'research_release_blocked', False
    now = int(time.time())
    pct = round(prob * 100.0, 1)
    stake = _stake_units(pc.get('model_prob', prob), pc.get('odds'), pc.get('fair_prob'))
    audit_json = _tip_audit_json(fid, phase, feat, raw, candidates, pc, now)
    with _tip_transaction() as c:
        if not _reserve_fixture_selection(c, fid, suggestion):
            return 'duplicate_or_conflicting_selection', False
        # The primary key is per-fixture/second: obtain a free second while the
        # fixture lock is held, rather than sending after an INSERT did nothing.
        previous = c.execute("SELECT MAX(created_ts) FROM tips WHERE match_id=%s", (fid,)).fetchone()[0]
        created = max(now, int(previous or 0) + 1)
        inserted = c.execute("""INSERT INTO tips(match_id,league_id,league,home,away,market,suggestion,
            confidence,confidence_raw,score_at_tip,minute,created_ts,odds,book,ev_pct,
            fair_prob,kickoff_ts,is_prematch,stake_units,sent_ok,decision_ts,
            stats_fetched_ts,odds_fetched_ts,telegram_sent_ts,price_verified,audit_json)
            VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,0,%s,%s,%s,NULL,%s,%s)
            ON CONFLICT(match_id,created_ts) DO NOTHING RETURNING created_ts""",
            (fid, league_id, league, home, away, market, suggestion, pct, pc.get("model_prob", prob),
             score if phase == 'live' else None, minute if phase == 'live' else None, created,
             pc.get('odds'), pc.get('book'), pc.get('ev_pct'), pc.get('fair_prob'), kickoff,
             int(phase == 'prematch'), stake, now, _STATS_FETCHED_TS.get(fid) if phase == 'live' else None,
             pc.get('odds_fetched_ts'), int(bool(pc.get('odds'))), audit_json)).fetchone()
        if not inserted:
            return 'insert_conflict', False
    kickoff_txt = datetime.fromtimestamp(kickoff, TZ_UTC).astimezone(BERLIN_TZ).strftime('%H:%M') if kickoff else 'TBD'
    message = _format_tip_message(home, away, league, minute, score, suggestion, pct, raw,
        pc.get('odds'), pc.get('book'), pc.get('ev_pct'), pc.get('fair_prob'), stake,
        kickoff_txt=kickoff_txt, prematch=phase == 'prematch')
    sent = _send_reserved_tip(fid, created, message)
    return ('telegram_sent' if sent else 'telegram_failed'), True


def _evaluate_candidates(audit, fx, feat, raw, candidates, saved, cooling_down=False):
    """Evaluate every candidate, recording shadow independently of delivery limits."""
    phase = audit.phase
    live = phase == 'live'
    fid = int((fx.get('fixture') or {}).get('id') or 0)
    league_id, league = _league_name(fx)
    home, away = _teams(fx)
    kickoff = _kickoff_ts_of(fx)
    minute = int((raw or {}).get('minute') or 0)
    score = _pretty_score(fx) if live else ''
    taken = _fixture_tip_history(fid)
    # Cooldowns and quotas control delivery only; never truncate research.
    already_pre = False
    if not live and PREMATCH_DEDUP_ENABLE:
        with db_conn() as c:
            already_pre = bool(c.execute("SELECT 1 FROM tips WHERE match_id=%s AND is_prematch=1 LIMIT 1", (fid,)).fetchone())
    for missing in sorted(_ALL_MARKET_FAMILIES - {row[0] for row in candidates}):
        audit.record(fid, 'model', 'no_candidates', league_id=league_id, market=missing)
    per_match = 0
    pred_rows = []
    for market, suggestion, prob, thr in sorted(candidates, key=lambda c: c[2] if math.isfinite(c[2]) else -1, reverse=True):
        label = market if live else f'PRE {market}'
        reason = None
        qualified = False
        pc = PriceCheck(passed=False, decision='not_priced')
        try:
            if suggestion not in ALLOWED_SUGGESTIONS:
                reason = 'unsupported_selection'
            elif not math.isfinite(prob) or not 0 <= prob <= 1 or not math.isfinite(thr):
                reason = 'invalid_probability_or_threshold'
            elif live and not _candidate_is_sane(suggestion, feat):
                reason = 'already_decided'
            else:
                # Retain priced candidates below the current threshold so
                # later threshold research is not limited to accepted picks.
                pc = _price_gate(market, suggestion, fid, prob, live=live)
                if pc.get('passed') and live:
                    ODDS_CACHE.invalidate((fid, True))
                    pc = _price_gate(market, suggestion, fid, prob, live=True)
                    if pc.get('passed'):
                        pc = _live_delivery_check(fid, raw, pc)
                if pc.get('passed') and not pc.get('odds'):
                    pc = PriceCheck(passed=False, decision='unpriced_candidate')
                if not live and (not kickoff or kickoff <= time.time()):
                    pc = PriceCheck(**{**pc, 'passed': False, 'decision': 'kickoff_passed'})
                prob = pc.get('effective_prob', prob)
                qualified = bool(pc.get('passed') and prob * 100 >= thr)
                if qualified:
                    _shadow_record(fid, league_id, league, kickoff, phase, minute,
                                   label, suggestion, prob, thr, pc,
                                   model_version=audit.model_version, policy_version=audit.policy_version)
                if not _market_active(market):
                    reason = 'market_disabled'
                elif not pc.get('passed'):
                    reason = pc['decision']
                elif prob * 100 < thr:
                    reason = 'model_suppressed' if thr > 100 else 'below_threshold'
                elif SHADOW_ONLY:
                    reason = 'shadow_only'
                elif not _delivery_allowed(phase):
                    reason = 'research_release_blocked'
                elif cooling_down:
                    reason = 'fixture_cooldown'
                elif already_pre:
                    reason = 'fixture_already_tipped'
                elif per_match >= max(1, PREDICTIONS_PER_MATCH):
                    reason = 'per_match_cap'
                elif (MAX_TIPS_PER_SCAN if live else MAX_PREMATCH_TIPS_PER_SCAN) and saved >= (MAX_TIPS_PER_SCAN if live else MAX_PREMATCH_TIPS_PER_SCAN):
                    reason = 'scan_cap'
                else:
                    reason = _history_rejection(suggestion, taken)
                    if not reason and _correlation_blocked(suggestion, taken):
                        extra = int(round((pc.get('ev_pct') or 0) * 100)) - EDGE_MIN_BPS
                        if extra < CORRELATED_EXTRA_EV_BPS:
                            reason = 'correlated_with_existing_tip'
                    if not reason:
                        reason, inserted = _save_candidate_tip(fid, league_id, league, home, away,
                            kickoff, phase, minute, score, raw, feat, candidates, label, suggestion, prob, pc)
                        if inserted:
                            saved += 1
                            per_match += 1
                            taken.append(suggestion)
        except Exception as exc:
            log.exception('[CANDIDATE] fixture %s evaluation failed', fid)
            audit.record(fid, 'candidate', 'candidate_error', league_id=league_id, kickoff=kickoff,
                         market=label, suggestion=suggestion, prob=prob, threshold=thr, pc=pc,
                         qualified=qualified, detail=type(exc).__name__)
            continue
        audit.record(fid, 'candidate', reason, league_id=league_id, kickoff=kickoff,
                     market=label, suggestion=suggestion, prob=prob, threshold=thr, pc=pc, qualified=qualified)
        if math.isfinite(prob) and 0 <= prob <= 1 and math.isfinite(thr):
            pred_rows.append((fid, league_id, kickoff, int(time.time()), phase, minute,
                              label, suggestion, prob, thr, pc.get('odds'), pc.get('fair_prob'), pc.get('ev_pct'), reason))
    _log_predictions(pred_rows)
    return saved


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
from grading import _parse_ou_line_from_suggestion, _tip_outcome_for_result


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
              UNION ALL
              SELECT match_id, MAX(created_ts) FROM shadow_picks GROUP BY match_id
              UNION ALL
              SELECT match_id, MAX(created_ts) FROM scan_decisions
              WHERE created_ts >= %s AND stage='candidate' AND match_id IS NOT NULL GROUP BY match_id
            ), agg AS (
              SELECT match_id, MAX(last_ts) AS last_ts FROM seen GROUP BY match_id
            )
            SELECT a.match_id FROM agg a
            LEFT JOIN match_results r ON r.match_id = a.match_id
            LEFT JOIN fixture_voids v ON v.match_id = a.match_id
            LEFT JOIN settlement_checks q ON q.match_id = a.match_id
            WHERE r.match_id IS NULL AND v.match_id IS NULL
            ORDER BY COALESCE(q.last_checked_ts,0) ASC,a.last_ts ASC LIMIT %s
        """, (cutoff, cutoff, cutoff, cutoff, max_rows)).fetchall()

    updated = 0
    for (mid,) in rows:
        with db_conn() as c:
            c.execute("INSERT INTO settlement_checks(match_id,last_checked_ts) VALUES(%s,%s) "
                      "ON CONFLICT(match_id) DO UPDATE SET last_checked_ts=EXCLUDED.last_checked_ts",
                      (mid, int(time.time())))
        fx = _fixture_by_id(int(mid))
        if not fx:
            continue
        st = (((fx.get("fixture") or {}).get("status") or {}).get("short") or "").upper()
        if _fixture_status_void(st, fx):
            with db_conn() as c:
                c.execute("INSERT INTO fixture_voids(match_id,status,updated_ts) VALUES(%s,%s,%s) "
                          "ON CONFLICT(match_id) DO NOTHING", (mid, st, int(time.time())))
            updated += 1
            continue
        if st not in FINAL_STATUSES:
            continue
        # Full-match 1X2/totals/BTTS settle on regulation time, including added
        # time, not extra time or penalties. Missing scores must stay pending.
        g = ((fx.get('score') or {}).get('fulltime') or {})
        if st == 'FT' and (g.get('home') is None or g.get('away') is None):
            g = fx.get('goals') or {}
        if g.get('home') is None or g.get('away') is None:
            continue
        gh, ga = int(g['home']), int(g['away'])
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
    """Refresh the latest available same-book sample before kickoff.

    This is a sampled pre-kickoff line, not proof of the exact final tick.
    Its timestamp is retained so coverage and staleness are visible.
    """
    if not CLV_ENABLE:
        return 0
    now = int(time.time())
    n = _research().capture(limit)
    for table in ('tips', 'shadow_picks'):
        if table == 'shadow_picks' and not SHADOW_ENABLE:
            continue
        condition = "is_prematch=1 AND COALESCE(price_verified,0)=1" if table == 'tips' else "phase='prematch'"
        with db_conn() as c:
            rows = c.execute(f"""SELECT match_id,created_ts,market,suggestion,odds,book,kickoff_ts
                FROM {table} WHERE {condition} AND odds IS NOT NULL
                AND kickoff_ts > %s AND kickoff_ts <= %s
                ORDER BY closing_ts ASC NULLS FIRST,kickoff_ts LIMIT %s""",
                (now, now + CLV_CAPTURE_LEAD_MIN * 60, limit)).fetchall()
        for mid, created, market, suggestion, entry_odds, book, kickoff in rows:
            if not book:
                continue
            mkey, sel = _market_key_and_selection(market, suggestion)
            if not mkey or not sel:
                continue
            entry = fetch_odds(int(mid), live=False).get(mkey) or {}
            observed = entry.get('fetched_ts')
            if observed is None or not 0 <= time.time() - float(observed) <= 90 or float(observed) >= kickoff:
                continue
            source = entry.get('source_update')
            if source:
                try:
                    stamp = datetime.fromisoformat(str(source).replace('Z', '+00:00'))
                    if stamp.tzinfo is None or not -30 <= time.time() - stamp.timestamp() <= 300:
                        continue
                except (ValueError, TypeError, OverflowError):
                    continue
            prices = (entry.get('by_book') or {}).get(sel) or {}
            close = prices.get(book)
            if close is None or not math.isfinite(float(close)) or float(close) <= 1.0:
                continue
            # Check time again after the API call: never label a post-kickoff
            # response as a prematch close.
            if time.time() >= kickoff:
                continue
            clv = (float(entry_odds) / float(close) - 1.0) * 100.0
            with db_conn() as c:
                changed = c.execute(f"""UPDATE {table} SET closing_odds=%s,clv_pct=%s,closing_ts=%s
                    WHERE match_id=%s AND created_ts=%s AND suggestion=%s
                    AND (closing_ts IS NULL OR closing_ts < %s)""",
                    (float(close), round(clv, 3), int(observed), mid, created, suggestion, int(observed))).rowcount
            n += changed
    return n


def compute_price_gate_breakdown(days: Optional[int] = None, phase: Optional[str] = None) -> Dict[str, Any]:
    cutoff = int(time.time()) - int(days) * 86400 if days else 0
    sql = """SELECT r.phase,d.price_decision,d.reason,COUNT(*) FROM scan_decisions d
             JOIN scan_runs r ON r.scan_id=d.scan_id WHERE d.stage='candidate' AND d.created_ts >= %s"""
    params = [cutoff]
    if phase:
        sql += ' AND r.phase=%s'
        params.append(phase)
    sql += ' GROUP BY r.phase,d.price_decision,d.reason'
    with db_conn() as c:
        rows = c.execute(sql, tuple(params)).fetchall()
    result = {}
    for ph, price, reason, n in rows:
        d = result.setdefault(ph, {'price_decisions': {}, 'terminal_decisions': {}})
        d['price_decisions'][price or 'not_priced'] = d['price_decisions'].get(price or 'not_priced', 0) + n
        d['terminal_decisions'][reason] = d['terminal_decisions'].get(reason, 0) + n
    return {'window_days': days, 'by_phase': result,
            'note': 'Complete candidate decisions since this audit build; earlier sampled predictions excluded.'}


def compute_clv(days: Optional[int] = None) -> Dict[str, Any]:
    cutoff = int(time.time()) - days * 86400 if days else 0
    with db_conn() as c:
        rows = c.execute("""
            SELECT market, clv_pct FROM tips
            WHERE clv_pct IS NOT NULL AND created_ts >= %s
              AND COALESCE(price_verified,0)=1
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


def compute_clv_breakdown(days: Optional[int] = None, min_n: int = 20) -> Dict[str, Any]:
    """
    CLV per (market, league), with a 95% CI on the mean — the same clv_pct
    values compute_clv() already pools, sliced finely enough to say WHICH
    combination has edge instead of one aggregate that can hide a strong
    pocket and a bleeding one inside a mean of zero. clv_pct is only ever set
    by capture_closing_lines(), which is prematch-only by construction, so no
    is_prematch filter is needed here.
    """
    cutoff = int(time.time()) - days * 86400 if days else 0
    with db_conn() as c:
        rows = c.execute("""
            SELECT market, league, clv_pct FROM tips
            WHERE clv_pct IS NOT NULL AND created_ts >= %s
              AND COALESCE(price_verified,0)=1
        """, (cutoff,)).fetchall()

    if not rows:
        return {"n": 0, "note": "No closing prices captured yet."}

    by: Dict[Tuple[str, str], List[float]] = {}
    for mkt, league, clv in rows:
        by.setdefault((mkt or "?", league or "?"), []).append(float(clv))

    def _ci95(v: List[float]) -> Tuple[float, float]:
        n = len(v)
        mean = sum(v) / n
        if n < 2:
            return (round(mean, 2), round(mean, 2))
        var = sum((x - mean) ** 2 for x in v) / (n - 1)
        se = math.sqrt(var / n)
        return (round(mean - 1.96 * se, 2), round(mean + 1.96 * se, 2))

    out = []
    for (mkt, league), vals in by.items():
        n = len(vals)
        mean = sum(vals) / n
        lo, hi = _ci95(vals)
        out.append({
            "market": mkt, "league": league, "n": n,
            "mean_clv_pct": round(mean, 2),
            "ci95_low": lo, "ci95_high": hi,
            "beat_close_pct": round(100.0 * sum(1 for x in vals if x > 0) / n, 1),
            "below_min_n": n < min_n,
        })
    out.sort(key=lambda d: (d["below_min_n"], -d["mean_clv_pct"]))

    return {
        "window_days": days, "min_n": min_n, "n_combinations": len(out),
        "breakdown": out,
        "note": ("mean_clv_pct is the number to act on: sustained positive CLV is evidence of "
                 "real edge; sustained negative CLV means retire that market/league even if its "
                 "ROI looks fine — ROI without positive CLV is variance, not skill. "
                 "below_min_n=true rows are too thin to trust regardless of what the CI says."),
    }


# ───────── Message formatting ─────────
def _format_tip_message(home, away, league, minute, score, suggestion, prob_pct,
                        raw=None, odds=None, book=None, ev_pct=None, fair_prob=None,
                        stake=None, kickoff_txt=None, prematch=False):
    raw = raw or {}
    stat = ""
    if not prematch and any(raw.get(k, 0) for k in ("xg_h", "xg_a", "sot_h", "sot_a", "cor_h", "cor_a",
                                                    "pos_h", "pos_a", "red_h", "red_a")):
        xg_h_known = bool(raw.get("_xg_h_available"))
        xg_a_known = bool(raw.get("_xg_a_available"))
        xg_h = f"{raw.get('xg_h', 0):.2f}" if xg_h_known else "N/A"
        xg_a = f"{raw.get('xg_a', 0):.2f}" if xg_a_known else "N/A"
        stat = (f"\n📊 xG {xg_h}-{xg_a}"
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
    """Fixed research baseline, never inherited from a precision-picked threshold."""
    base = m.replace('PRE ', '')
    if base not in ('BTTS Yes', 'BTTS No', 'Over 2.5', 'Under 2.5'):
        return SUPPRESSED_THRESHOLD_PCT
    value = get_setting_cached(f'research_threshold:{m}')
    return float(value) if value is not None else 55.0


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
        if f"Over/Under {_fmt_line(line)}" not in ACTIVE_MARKETS:
            continue
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
        line_txt = _fmt_line(line)
        mk = f"Over/Under {line_txt}"
        over_thr = thr_fn(f"Over {line_txt}")
        under_thr = thr_fn(f"Under {line_txt}")
        out.append((mk, f"Over {line_txt} Goals", p_over, over_thr))
        out.append((mk, f"Under {line_txt} Goals", 1.0 - p_over, under_thr))
    return out


def _btts_candidates(feat: Dict[str, float], prefix: str, thr_fn) -> List[Tuple[str, str, float, float]]:
    if "BTTS" not in ACTIVE_MARKETS:
        return []
    mdl = load_model_from_settings(f"{prefix}BTTS_YES")
    if not mdl:
        return []
    p = _score_prob(feat, mdl)
    yes_thr = thr_fn("BTTS Yes")
    no_thr = thr_fn("BTTS No")
    return [("BTTS", "BTTS: Yes", p, yes_thr),
            ("BTTS", "BTTS: No", 1.0 - p, no_thr)]


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
    details = _wld_details(feat, prefix)
    if details is None:
        return None
    p = details["normalized"]
    return p["home"], p["draw"], p["away"]


def _wld_details(feat: Dict[str, float], prefix: str) -> Optional[Dict[str, Any]]:
    if not ACTIVE_MARKETS & {"1X2", "Double Chance", "Draw No Bet"}:
        return None
    mh = load_model_from_settings(f"{prefix}WLD_HOME")
    md = load_model_from_settings(f"{prefix}WLD_DRAW")
    ma = load_model_from_settings(f"{prefix}WLD_AWAY")
    # A complete 1X2 vector is required. A hand-written draw fallback makes
    # the denominator look complete while the model is actually missing a
    # head, distorting 1X2, Double Chance and DNB together.
    if not (mh and md and ma):
        log.warning("[1X2] %sWLD model set incomplete — suppressing derived markets", prefix)
        return None
    ph = _score_prob(feat, mh)
    pa = _score_prob(feat, ma)
    pd_ = _score_prob(feat, md)
    s = ph + pd_ + pa
    if not math.isfinite(s) or s <= EPS:
        raise ValueError("invalid 1X2 normalization sum")
    return {"before_normalization": {"home": ph, "draw": pd_, "away": pa},
            "normalization_sum": s,
            "normalized": {"home": ph / s, "draw": pd_ / s, "away": pa / s},
            "draw_fallback_used": False}


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


def _fixture_tip_history(fid: int) -> List[str]:
    # Include queued tips as well as delivered tips: retries must not create
    # a second position. The restriction persists beyond the scan cooldown.
    with db_conn() as c:
        return [row[0] for row in c.execute(
            "SELECT suggestion FROM tips WHERE match_id=%s AND suggestion<>'HARVEST'",
            (fid,)).fetchall()]


def _history_rejection(suggestion: str, taken: List[str]) -> Optional[str]:
    if suggestion in taken:
        return "duplicate_fixture_selection"
    opposite = {"BTTS: Yes": "BTTS: No", "BTTS: No": "BTTS: Yes",
                "Home Win": "Away Win", "Away Win": "Home Win",
                "Draw No Bet: Home": "Draw No Bet: Away",
                "Draw No Bet: Away": "Draw No Bet: Home"}.get(suggestion)
    if opposite in taken:
        return "opposing_fixture_selection"
    match = re.fullmatch(r"(Over|Under) (\d+(?:\.\d+)?) Goals", suggestion)
    if match:
        opposite = ("Under" if match[1] == "Over" else "Over") + " " + match[2] + " Goals"
        if opposite in taken:
            return "opposing_fixture_selection"
    return None


def _directional_market(market: str, suggestion: str) -> str:
    if re.fullmatch(r"(?:Over|Under) \d+(?:\.\d+)? Goals", suggestion):
        return ("PRE " if (market or "").startswith("PRE ") else "") + suggestion.removesuffix(" Goals")
    return market or "?"


@contextmanager
def _tip_transaction():
    with db_conn() as c:
        # Ordinary pooled queries use autocommit; this section needs one
        # transaction spanning the advisory lock, history read and insertion.
        c.conn.autocommit = False
        try:
            yield c
            c.conn.commit()
        except BaseException:
            c.conn.rollback()
            raise
        finally:
            c.conn.autocommit = True


def _reserve_fixture_selection(c, fid: int, suggestion: str) -> bool:
    # Lock and recheck inside the INSERT transaction so simultaneous workers
    # cannot both accept the same selection from an earlier history snapshot.
    c.execute("SELECT pg_advisory_xact_lock(%s,%s)", (19018, fid))
    previous = [row[0] for row in c.execute(
        "SELECT suggestion FROM tips WHERE match_id=%s AND suggestion<>'HARVEST'",
        (fid,)).fetchall()]
    reason = _history_rejection(suggestion, previous)
    if reason:
        log.info("[HISTORY] fixture %s %s: %s", fid, suggestion, reason)
    return reason is None


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
_live_snapshot: Dict[str, Any] = {"updated_ts": 0, "matches": [], "stats_diagnostics": []}


def _build_live_match_entry(fid: int, league: str, league_id: int, home: str, away: str,
                            score: str, minute: int,
                            candidates: List[Tuple[str, str, float, float]],
                            kickoff_ts: int = 0, raw: Optional[Dict[str, float]] = None,
                            home_id: int = 0, away_id: int = 0,
                            feat: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
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
        if float(pr) * 100.0 >= float(thr):
            pc = _price_gate(mt, sg, fid, pr, live=True)
            row["decision"] = pc["decision"]
            row["odds"] = pc.get("odds")
            row["ev_pct"] = pc.get("ev_pct")
            for key in ("fair_prob", "fair_edge_pct", "n_books", "book",
                        "odds_fetched_ts", "odds_source_update", "price_vs_fair_pct"):
                row[key] = pc.get(key)
            row["gate_passed"] = bool(pc["passed"])
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
        }

    return {
        "fixture_id": fid, "league": league, "league_id": league_id,
        "home": home, "away": away, "score": score, "minute": minute,
        # Team ids let /dashboard/match/<fid>/form resolve who to look up
        # from the snapshot, instead of trusting ids from the query string.
        "home_id": int(home_id or 0), "away_id": int(away_id or 0),
        "kickoff_ts": int(kickoff_ts or 0), "stats": stats,
        "markets": markets,
        "wld_diagnostics": _wld_details(feat, "") if feat is not None else None,
        "odds_diagnostics": ODDS_DIAGNOSTICS.get((fid, True)),
        "gate_limits": {"max_model_edge_pp": MAX_MODEL_EDGE_BPS / 100.0,
                        "min_ev_pct": EDGE_MIN_BPS / 100.0,
                        "min_fair_edge_pp": FAIR_EDGE_MIN_BPS / 100.0},
        "decision_note": "tipped means price gate passed; delivery and fixture limits are checked separately",
        # Count of candidates that would actually be tipped (passed the full
        # price gate), not just candidates with high raw confidence - this is
        # what "worth a look" should mean on the dashboard.
        "hits": sum(1 for m in markets if m["decision"] == "tipped"),
    }


def _set_live_snapshot(matches: List[Dict[str, Any]], live_seen: Optional[int] = None,
                       no_coverage: Optional[int] = None,
                       stats_diagnostics: Optional[List[Dict[str, Any]]] = None) -> None:
    # live_seen/no_coverage travel with the matches so the dashboard can tell
    # "nothing is being played right now" apart from "plenty is being played,
    # none of it has usable stats yet" - an empty list on its own can't.
    with _live_snapshot_lock:
        _live_snapshot["updated_ts"] = int(time.time())
        _live_snapshot["matches"] = matches
        _live_snapshot["live_seen"] = live_seen
        _live_snapshot["no_coverage"] = no_coverage
        _live_snapshot["stats_diagnostics"] = list(stats_diagnostics or [])


def _get_live_snapshot() -> Dict[str, Any]:
    with _live_snapshot_lock:
        return {"updated_ts": _live_snapshot["updated_ts"],
                "matches": list(_live_snapshot["matches"]),
                "live_seen": _live_snapshot.get("live_seen"),
                "no_coverage": _live_snapshot.get("no_coverage"),
                "stats_diagnostics": list(_live_snapshot.get("stats_diagnostics") or [])}


def _live_stats_diagnostic_payload() -> Dict[str, Any]:
    """Sanitized explanation of statistics coverage from the latest scan."""
    snap = _get_live_snapshot()
    rows = snap.get("stats_diagnostics") or []
    reasons: Dict[str, int] = {}
    for row in rows:
        reason = str(row.get("reason") or "unknown")
        reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "updated_ts": snap.get("updated_ts"),
        "live_seen": snap.get("live_seen"),
        "diagnosed": len(rows),
        "usable": sum(1 for row in rows if row.get("covered")),
        "xg_available_both_teams": sum(
            1 for row in rows
            if row.get("xg_home_available") and row.get("xg_away_available")),
        "reasons": reasons,
        "fixtures": rows,
        "api_usage": _api_call_stats_snapshot(),
        "note": ("Zero is treated as a valid statistic. missing_team_statistics means the API "
                 "did not return resolvable blocks for both team IDs; too_few_returned_fields "
                 "means both teams were present but too few recognised statistic groups had values."),
    }


def production_scan() -> Tuple[int, int]:
    if not LIVE_SCAN_ENABLE:
        log.info("[SCAN] live scan disabled (LIVE_SCAN_ENABLE=0); prematch research remains active")
        return (0, 0)
    with ScanAudit('live') as audit:
        matches = fetch_live_matches(audit=audit)
        saved = 0
        snapshots = []
        diagnostics = []
        no_coverage = 0
        last_snap = _last_snapshot_ts_bulk([int((m.get('fixture') or {}).get('id') or 0) for m in matches]) if HARVEST_MODE else {}
        for fx in matches:
            fid = int((fx.get('fixture') or {}).get('id') or 0)
            try:
                if not fid:
                    audit.record(None, 'fixture', 'invalid_fixture_id')
                    continue
                raw = extract_raw_inplay(fx)
                minute = int(raw.get('minute', 0))
                league_id, league = _league_name(fx)
                home, away = _teams(fx)
                diag = _stats_coverage_details(raw, minute)
                diag.update(fixture_id=fid, league_id=league_id, league=league, home=home, away=away, minute=minute)
                diagnostics.append(diag)
                if minute < TIP_MIN_MINUTE:
                    audit.record(fid, 'fixture', 'before_minute', league_id=league_id)
                    continue
                if not stats_coverage_ok(raw, minute):
                    no_coverage += 1
                    audit.record(fid, 'fixture', diag.get('reason') or 'statistics_unusable', league_id=league_id)
                    continue
                raw, feat = _complete_live_features(fx, raw, fetch_market_price=True)
                if (HARVEST_MODE and minute >= TRAIN_MIN_MINUTE
                        and time.time() - last_snap.get(fid, 0) >= HARVEST_EVERY_MINUTES * 60):
                    save_snapshot_from_match(fx, raw)
                candidates = (_ou_candidates(feat, '', _get_market_threshold)
                            + _btts_candidates(feat, '', _get_market_threshold)
                            + _wld_candidates(feat, '', _get_market_threshold)
                            + _dc_dnb_candidates(feat, '', _get_market_threshold))
                cooling = False
                if DUP_COOLDOWN_MIN > 0:
                    with db_conn() as c:
                        cooling = bool(c.execute("SELECT 1 FROM tips WHERE match_id=%s AND created_ts >= %s LIMIT 1",
                            (fid, int(time.time()) - DUP_COOLDOWN_MIN * 60)).fetchone())
                saved = _evaluate_candidates(audit, fx, feat, raw, candidates, saved, cooling)
                audit.record(fid, 'fixture', 'evaluated' if candidates else 'no_models', league_id=league_id)
                # The dashboard reads the same fixture without governing any decision.
                try:
                    home_id, away_id = _team_ids(fx)
                    visible = [c for c in candidates if c[1] in ALLOWED_SUGGESTIONS
                               and _candidate_is_sane(c[1], feat) and _market_active(c[0])]
                    snapshots.append(_build_live_match_entry(fid, league, league_id, home, away,
                        _pretty_score(fx), minute, visible, kickoff_ts=_kickoff_ts_of(fx), raw=raw,
                        home_id=home_id, away_id=away_id, feat=feat))
                except Exception:
                    log.exception('[DASHBOARD] snapshot failed for %s', fid)
            except Exception as exc:
                log.exception('[PROD] fixture %s failed', fid)
                audit.record(fid, 'fixture', 'fixture_error', detail=type(exc).__name__)
        _set_live_snapshot(snapshots, live_seen=len(matches), no_coverage=no_coverage, stats_diagnostics=diagnostics)
        log.info('[PROD] saved=%d live_seen=%d no_coverage=%d scan_id=%s', saved, len(matches), no_coverage, audit.scan_id)
        return saved, len(matches)


def score_live_matches_now(
    stats_diagnostics_out: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], int]:
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
            raw = extract_raw_inplay(m)
            minute = int(raw.get("minute", 0))
            if stats_diagnostics_out is not None:
                league_id, league = _league_name(m)
                home, away = _teams(m)
                stat_diag = _stats_coverage_details(raw, minute)
                stat_diag.update({"fixture_id": fid, "league_id": league_id, "league": league,
                                  "home": home, "away": away, "minute": minute})
                stats_diagnostics_out.append(stat_diag)
            if minute < TIP_MIN_MINUTE or not stats_coverage_ok(raw, minute):
                continue
            raw, feat = _complete_live_features(m, raw, fetch_market_price=True)

            league_id, league = _league_name(m)
            home, away = _teams(m)
            score = _pretty_score(m)
            kickoff = _kickoff_ts_of(m)

            candidates = (_ou_candidates(feat, "", _get_market_threshold)
                          + _btts_candidates(feat, "", _get_market_threshold)
                          + _wld_candidates(feat, "", _get_market_threshold)
                          + _dc_dnb_candidates(feat, "", _get_market_threshold))
            candidates = [c for c in candidates
                          if c[1] in ALLOWED_SUGGESTIONS and _candidate_is_sane(c[1], feat)
                          and _market_active(c[0])]
            candidates.sort(key=lambda x: x[2], reverse=True)

            home_id, away_id = _team_ids(m)
            out.append(_build_live_match_entry(fid, league, league_id, home, away, score,
                                               minute, candidates, kickoff_ts=kickoff, raw=raw,
                                               home_id=home_id, away_id=away_id, feat=feat))
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


def _collect_todays_prematch_fixtures(audit=None) -> List[dict]:
    # Despite the historical function name, this is now a forward-looking
    # prematch window.  Query every UTC calendar date touched by the local
    # window so Berlin midnight/DST transitions cannot silently drop fixtures.
    now_utc = datetime.now(TZ_UTC)
    now_ts = now_utc.timestamp()
    start_local = now_utc.astimezone(BERLIN_TZ)
    end_utc = now_utc + timedelta(hours=PREMATCH_LOOKAHEAD_HOURS)
    end_local = end_utc.astimezone(BERLIN_TZ)
    start_utc_date = start_local.astimezone(TZ_UTC).date()
    end_utc_date = end_local.astimezone(TZ_UTC).date()
    dates_utc = []
    cursor = start_utc_date
    while cursor <= end_utc_date:
        dates_utc.append(cursor)
        cursor += timedelta(days=1)
    fixtures = []
    seen = set()
    for d in dates_utc:
        js = _api_get(FOOTBALL_API_URL, {'date': d.strftime('%Y-%m-%d')})
        if not isinstance(js, dict) or not isinstance(js.get('response'), list) or js.get('errors'):
            raise RuntimeError('fixture_feed_unavailable')
        for fx in js['response']:
            fid = (fx.get('fixture') or {}).get('id')
            if fid in seen:
                continue
            seen.add(fid)
            kickoff = _kickoff_ts_of(fx)
            reason = None
            if not kickoff or not now_ts < kickoff <= end_local.timestamp():
                reason = 'outside_prematch_horizon'
            elif (((fx.get('fixture') or {}).get('status') or {}).get('short') or '').upper() != 'NS':
                reason = 'not_prematch'
            elif _blocked_league(fx.get('league') or {}):
                reason = 'league_excluded'
            elif PREMATCH_LEAGUE_IDS and int((fx.get('league') or {}).get('id') or 0) not in PREMATCH_LEAGUE_IDS:
                reason = 'prematch_league_excluded'
            if reason:
                if audit:
                    audit.record(fid, 'fixture', reason)
            else:
                fixtures.append(fx)
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
    fid = int((fx.get("fixture") or {}).get("id") or 0)
    feat = assemble_prematch_features(th, ta, last_h, last_a, h2h, kickoff,
                                      ratings.get(th, ELO_DEFAULT), ratings.get(ta, ELO_DEFAULT), lr)
    # Current observations only. Historical backfill does not call this path.
    feat.update(historical_xg_features(th, ta, last_h, last_a, min(kickoff, time.time()),
                                       _fetch_historical_xg_stats))
    return feat


_XG_HISTORY_CACHE = _TTLCache(86400)


def _fetch_historical_xg_stats(fid):
    cached = _XG_HISTORY_CACHE.get(fid, _MISS)
    if cached is not _MISS:
        return cached
    value = fetch_match_stats(fid)
    if value is None:
        log.warning("[XG] fixture %s historical stats unavailable; retrying next extraction", fid)
        return None
    # Cache only a successful response. Empty means the provider answered and
    # this fixture has no xG coverage; it is different from a failed fetch.
    _XG_HISTORY_CACHE.set(fid, value)
    return value


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
    with ScanAudit('prematch') as audit:
        fixtures = _collect_todays_prematch_fixtures(audit=audit)
        feats, fresh = _get_prematch_features_bulk(fixtures)
        saved = 0
        for fx in fixtures:
            fid = int((fx.get('fixture') or {}).get('id') or 0)
            try:
                if not fid:
                    audit.record(None, 'fixture', 'invalid_fixture_id')
                    continue
                league_id, _ = _league_name(fx)
                feat = feats.get(fid)
                if not feat:
                    audit.record(fid, 'fixture', 'features_missing', league_id=league_id)
                    continue
                block = prematch_data_gate(feat)
                if block:
                    audit.record(fid, 'fixture', 'form_unusable', league_id=league_id, detail=str(block))
                    continue
                kickoff = _kickoff_ts_of(fx)
                if not kickoff or kickoff <= time.time():
                    audit.record(fid, 'fixture', 'kickoff_passed', league_id=league_id)
                    continue
                if fid in fresh:
                    save_prematch_snapshot(fid, feat, kickoff)
                candidates = (_ou_candidates(feat, 'PRE_', _get_market_threshold_pre)
                            + _btts_candidates(feat, 'PRE_', _get_market_threshold_pre)
                            + _wld_candidates(feat, 'PRE_', _get_market_threshold_pre)
                            + _dc_dnb_candidates(feat, 'PRE_', _get_market_threshold_pre))
                saved = _evaluate_candidates(audit, fx, feat, None, candidates, saved)
                audit.record(fid, 'fixture', 'evaluated' if candidates else 'no_models', league_id=league_id)
            except Exception as exc:
                log.exception('[PREMATCH] fixture %s failed', fid)
                audit.record(fid, 'fixture', 'fixture_error', detail=type(exc).__name__)
        log.info('[PREMATCH] saved=%d fixtures=%d scan_id=%s', saved, len(fixtures), audit.scan_id)
        return saved


def send_match_of_the_day() -> bool:
    """Recap an already recorded/delivered pick; never create an untracked pick."""
    if SHADOW_ONLY or not _delivery_allowed('prematch'):
        return False
    now = int(time.time())
    with db_conn() as c:
        row = c.execute("""SELECT home,away,league,suggestion,confidence,odds,book,ev_pct,
            fair_prob,stake_units,kickoff_ts FROM tips WHERE is_prematch=1 AND sent_ok=1
            AND kickoff_ts > %s AND created_ts >= %s ORDER BY confidence DESC LIMIT 1""",
            (now, now - 86400)).fetchone()
    if not row:
        return False
    home, away, league, sug, pct, odds, book, ev_pct, fair, stake, kickoff = row
    clock = datetime.fromtimestamp(kickoff, TZ_UTC).astimezone(BERLIN_TZ).strftime('%H:%M')
    msg = _format_tip_message(home, away, league, 0, '', sug, pct, None,
        odds, book, ev_pct, fair, stake, kickoff_txt=clock, prematch=True)
    return send_telegram('🏅 Previously sent pick — original price may have changed\n' + msg)


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
    snapshots_saved = results_saved = hist_no_form = 0
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

        # Backfill has its own way of producing a blind vector: a side with no
        # fixtures in team_history before the cutoff — the opening rounds of a
        # season, a promoted club, a gap in what was fetched — assembles to all
        # zeros just like a failed live fetch. The result below is still
        # recorded; only the unusable feature row is skipped.
        if prematch_data_gate(feat):
            hist_no_form += 1
        else:
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
            "results_saved": results_saved, "skipped_no_form": hist_no_form,
            "api_diagnostics_per_season": diags}


# ───────── Analytics ─────────
def _norm_cdf(x: float) -> float:
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def compute_shadow_report(days: Optional[int] = 90) -> Dict[str, Any]:
    """Prospective one-unit returns; pushes excluded from ROI denominator."""
    cutoff = int(time.time()) - int(days) * 86400 if days else 0
    with db_conn() as c:
        rows = c.execute("""
            SELECT s.match_id,s.league_id,s.league,s.phase,s.market,s.suggestion,
                   s.prob,s.odds,s.clv_pct,s.model_version,
                   r.final_goals_h,r.final_goals_a,r.btts_yes,v.status,s.policy_version
            FROM shadow_picks s LEFT JOIN match_results r ON r.match_id=s.match_id
            LEFT JOIN fixture_voids v ON v.match_id=s.match_id
            WHERE s.created_ts >= %s ORDER BY s.created_ts
        """, (cutoff,)).fetchall()
    groups: Dict[str, Dict[str, Dict[str, Any]]] = {
        dim: {} for dim in ("market", "league", "probability_bucket", "odds_bucket", "model_version", "policy_version", "phase")}
    overall: Dict[str, Any] = {}

    def add(bucket: Dict[str, Any], outcome: Optional[int], pending: bool,
            odds: float, prob: float, clv: Optional[float], fid: int) -> None:
        bucket["recorded"] = bucket.get("recorded", 0) + 1
        bucket.setdefault("fixtures", set()).add(fid)
        if pending:
            bucket["pending"] = bucket.get("pending", 0) + 1
        elif outcome is None:
            bucket["void"] = bucket.get("void", 0) + 1
        else:
            bucket["graded"] = bucket.get("graded", 0) + 1
            bucket["wins"] = bucket.get("wins", 0) + outcome
            bucket["profit_units"] = bucket.get("profit_units", 0.0) + (odds - 1 if outcome else -1)
            bucket["predicted_probability_sum"] = bucket.get("predicted_probability_sum", 0.0) + prob
        if clv is not None:
            bucket["clv_n"] = bucket.get("clv_n", 0) + 1
            bucket["clv_sum"] = bucket.get("clv_sum", 0.0) + float(clv)

    for fid, league_id, league, phase, market, suggestion, prob, odds, clv, version, gh, ga, btts, void_status, policy in rows:
        pending = not void_status and (gh is None or ga is None)
        outcome = None if pending or void_status else _tip_outcome_for_result(
            suggestion, {"final_goals_h": gh, "final_goals_a": ga, "btts_yes": btts})
        p = float(prob)
        o = float(odds)
        pb = f"{min(int(p * 20) * 5, 95):02d}-{min(int(p * 20) * 5 + 5, 100):02d}%"
        ob = "<1.50" if o < 1.5 else "1.50-1.99" if o < 2 else "2.00-2.99" if o < 3 else "3.00+"
        labels = {"market": market, "league": f"{league_id}: {league}",
                  "probability_bucket": pb, "odds_bucket": ob,
                  "model_version": version, "policy_version": policy, "phase": phase}
        add(overall, outcome, pending, o, p, clv, int(fid))
        for dimension, label in labels.items():
            add(groups[dimension].setdefault(label, {}), outcome, pending, o, p, clv, int(fid))

    def finish(b: Dict[str, Any]) -> Dict[str, Any]:
        n = b.get("graded", 0)
        c = b.get("clv_n", 0)
        profit = b.get("profit_units", 0.0)
        return {"recorded": b.get("recorded", 0), "unique_fixtures": len(b.get("fixtures", ())),
                "pending": b.get("pending", 0), "void": b.get("void", 0),
                "graded": n, "wins": b.get("wins", 0),
                "profit_units": round(profit, 3), "roi_pct": round(100 * profit / n, 2) if n else None,
                "hit_rate_pct": round(100 * b.get("wins", 0) / n, 2) if n else None,
                "mean_model_prob_pct": round(100 * b.get("predicted_probability_sum", 0) / n, 2) if n else None,
                "clv_n": c, "mean_clv_pct": round(b.get("clv_sum", 0) / c, 3) if c else None}

    completed = {dim: {name: finish(b) for name, b in buckets.items()}
                 for dim, buckets in groups.items()}
    # A review flag is not an automatic disable. Repeated slices and market
    # selection on the same history otherwise manufacture a winning backtest.
    review = []
    for dim in ("market", "league"):
        for name, b in completed[dim].items():
            if b["graded"] >= 100 and b["roi_pct"] is not None and b["roi_pct"] < 0:
                review.append({"dimension": dim, "name": name,
                               "reason": "negative prospective ROI over at least 100 graded picks",
                               "prematch_clv_available": b["clv_n"]})
    return {"days": days, "overall": finish(overall), "by": completed,
            'sharp_close_release': _research().report('release'),
            "forward_target": {"graded_target": 500, "graded": overall.get('graded', 0),
                               "remaining": max(0, 500 - overall.get('graded', 0)),
                               "commercially_validated": False},
            "review_for_disabling": [],
            "note": "One unit per first qualifying fixture/phase/selection quote; live CLV is undefined. "
                    "Legacy CLV above is same-book and is NOT release evidence; see sharp_close_release. Repeated bets on one match "
                    "are correlated. Review flags do not change production filters."}


def compute_scan_funnel(days: int = 7) -> Dict[str, Any]:
    cutoff = int(time.time()) - max(1, int(days)) * 86400
    with db_conn() as c:
        runs = c.execute("SELECT phase,status,COUNT(*) FROM scan_runs WHERE started_ts >= %s GROUP BY phase,status", (cutoff,)).fetchall()
        rows = c.execute("""SELECT r.phase,d.stage,d.reason,COUNT(*) FROM scan_decisions d
            JOIN scan_runs r ON r.scan_id=d.scan_id WHERE r.started_ts >= %s
            GROUP BY r.phase,d.stage,d.reason""", (cutoff,)).fetchall()
        stale = c.execute("SELECT scan_id FROM scan_runs WHERE status='running' AND started_ts < %s",
                          (int(time.time()) - 3600,)).fetchall()
    out = {}
    for phase, status, n in runs:
        out.setdefault(phase, {'scan_status': {}, 'decisions': {}})['scan_status'][status] = n
    for phase, stage, reason, n in rows:
        out.setdefault(phase, {'scan_status': {}, 'decisions': {}})['decisions'][f'{stage}:{reason}'] = n
    return {'days': days, 'by_phase': out, 'unfinished_over_one_hour': [r[0] for r in stale],
            'note': 'One terminal record per examined fixture and candidate; repeated scans remain separate.'}


def compute_pnl(days: Optional[int] = None, stake: float = 1.0, use_kelly: bool = False) -> Dict[str, Any]:
    cutoff = int(time.time()) - days * 86400 if days else 0
    with db_conn() as c:
        rows = c.execute("""
            SELECT t.market, t.suggestion, t.odds, t.created_ts, t.stake_units, t.clv_pct,
                   COALESCE(t.price_verified,0), r.final_goals_h, r.final_goals_a, r.btts_yes, t.match_id
            FROM tips t JOIN match_results r ON r.match_id = t.match_id
            WHERE t.suggestion<>'HARVEST' AND t.odds IS NOT NULL AND t.created_ts >= %s
            ORDER BY t.created_ts ASC
        """, (cutoff,)).fetchall()

    total_staked = total_profit = 0.0
    n_bets = n_wins = n_push = 0
    by_market: Dict[str, Dict[str, float]] = {}
    by_selection = {}
    fixture_ids = set()
    equity: List[Dict[str, Any]] = []
    running = 0.0
    clvs: List[float] = []
    # Bets whose recorded price came from the contaminated market mapping.
    # They are graded the same way and reported separately, never mixed into
    # the headline: their prices were never available for the selection, so
    # counting them as a track record reports fiction as edge.
    stale = {"n_bets": 0, "n_wins": 0, "staked": 0.0, "profit": 0.0}

    for (mkt, sugg, odds, cts, stake_units, clv, price_verified, gh, ga, btts, fid) in rows:
        outcome = _tip_outcome_for_result(sugg, {"final_goals_h": gh, "final_goals_a": ga, "btts_yes": btts})
        if outcome is None:
            if int(price_verified or 0) and int(cts or 0) >= ODDS_TRUSTED_FROM_TS:
                n_push += 1
            continue
        s = float(stake_units) if (use_kelly and stake_units) else float(stake)
        if s <= 0:
            continue
        profit = s * (float(odds) - 1.0) if outcome == 1 else -s

        # Old records lack proof that their quote represented the suggested
        # market. Preserve them for audit, but never mix them into headline ROI.
        if not int(price_verified or 0) or int(cts or 0) < ODDS_TRUSTED_FROM_TS:
            stale["n_bets"] += 1
            stale["n_wins"] += 1 if outcome == 1 else 0
            stale["staked"] += s
            stale["profit"] += profit
            continue

        n_bets += 1
        fixture_ids.add(fid)
        directional = _directional_market(mkt, sugg)
        ds = by_selection.setdefault(directional, {"bets": 0, "wins": 0, "profit": 0.0,
                                                   "staked": 0.0, "fixtures": set()})
        ds["bets"] += 1
        ds["wins"] += int(outcome == 1)
        ds["profit"] += profit
        ds["staked"] += s
        ds["fixtures"].add(fid)
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
        "n_fixtures": len(fixture_ids),
        "by_selection": {key: {"bets": d["bets"], "wins": d["wins"],
            "n_fixtures": len(d["fixtures"]), "profit": round(d["profit"], 2),
            "staked": round(d["staked"], 2),
            "roi_pct": round(100 * d["profit"] / d["staked"], 2) if d["staked"] else 0.0}
            for key, d in by_selection.items()},
        "equity_curve": equity,
        "odds_trusted_from_ts": ODDS_TRUSTED_FROM_TS,
        "excluded_unreliable_pricing": {
            "n_bets": stale["n_bets"],
            "win_rate_pct": (round(100.0 * stale["n_wins"] / stale["n_bets"], 1)
                             if stale["n_bets"] else 0.0),
            "total_profit": round(stale["profit"], 2),
            "roi_pct": (round(stale["profit"] / stale["staked"] * 100.0, 2)
                        if stale["staked"] > 0 else 0.0),
            "note": ("Legacy or unverified prices recorded before the strict market-and-line "
                     "audit boundary. Preserved for inspection but excluded from every headline "
                     "figure because the true executable selection price cannot be proven."),
        },
        "note": ("Real odds captured at tip time, never synthetic. Tips sent without odds are "
                 "excluded — there is no price to grade them against. Draw No Bet pushes on a "
                 "draw and is excluded rather than counted as a loss. If mean_clv_pct is "
                 "negative while roi_pct is positive, treat the ROI as variance, not edge. "
                 "Headline figures contain only price_verified=1 records created by the strict "
                 "market-and-line parser; see excluded_unreliable_pricing for older records."),
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
        FROM (SELECT DISTINCT ON (match_id,phase,suggestion) * FROM predictions
              ORDER BY match_id,phase,suggestion,created_ts,id) p
        JOIN match_results r ON r.match_id = p.match_id
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
              AND COALESCE(t.price_verified,0)=1
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
              AND COALESCE(t.price_verified,0)=1
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
          AND COALESCE(t.price_verified,0)=1
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


def compute_league_density(days: Optional[int] = None, min_n: int = 20) -> Dict[str, Any]:
    """
    Settled-fixture volume per league, for choosing CONCENTRATION_MODE's
    league list from real numbers instead of a guess. A league can accumulate
    plenty of TIPS while producing few RESOLVED fixtures (match_results), and
    settled volume — not tip count or scan frequency — is the actual
    constraint on training a model or verifying a threshold on holdout data.

    days filters by match_results.updated_ts (when the result was recorded),
    matching every other days= filter in this file. league_id is API-Football's
    own id — cross-reference names via GET /admin/leagues.
    """
    cutoff = int(time.time()) - days * 86400 if days else 0
    with db_conn() as c:
        rows = c.execute("""
            SELECT r.league_id, COUNT(DISTINCT r.match_id)::bigint AS n_results,
                   COUNT(DISTINCT t.match_id)::bigint AS n_tipped_matches
            FROM match_results r
            LEFT JOIN tips t ON t.match_id = r.match_id AND t.suggestion <> 'HARVEST'
            WHERE r.updated_ts >= %s AND r.league_id IS NOT NULL
            GROUP BY r.league_id
        """, (cutoff,)).fetchall()

    out = []
    for league_id, n_results, n_tipped in rows:
        out.append({
            "league_id": int(league_id),
            "n_settled_fixtures": int(n_results),
            "n_tipped_matches": int(n_tipped),
            "below_min_n": int(n_results) < min_n,
        })
    out.sort(key=lambda d: d["n_settled_fixtures"], reverse=True)

    return {
        "window_days": days, "min_n": min_n, "n_leagues": len(out),
        "leagues": out, "selected_scope": list(LEAGUE_ALLOW_IDS),
        "note": ("Ranked by settled-fixture volume (match_results), the real constraint on "
                 "training and holdout verification — not scan frequency or tip count. Use "
                 "this to pick CONCENTRATION_MODE's LEAGUE_ALLOW_IDS / PREMATCH_LEAGUE_IDS: "
                 "leagues below min_n cannot support a verified threshold regardless of how "
                 "often they are scanned. Cross-reference league_id against GET /admin/leagues "
                 "for names."),
    }


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
            SELECT t.market, t.suggestion, r.final_goals_h, r.final_goals_a, r.btts_yes,
                   t.odds, t.match_id
            FROM tips t LEFT JOIN match_results r ON r.match_id=t.match_id
            WHERE t.created_ts >= %s AND t.created_ts < %s
              AND t.suggestion<>'HARVEST' AND t.sent_ok=1
              AND COALESCE(t.price_verified,0)=1
        """, (int(y0.timestamp()), int(y1.timestamp()))).fetchall()

    total = graded = wins = pushes = 0
    profit = 0.0
    priced = 0
    fixtures = set()
    by: Dict[str, Dict[str, int]] = {}
    for (mkt, sugg, gh, ga, btts, odds, fid) in rows:
        total += 1  # counted before the grading guard, so "Sent" != "Graded"
        fixtures.add(fid)
        if gh is None:
            continue
        out = _tip_outcome_for_result(sugg, {"final_goals_h": gh, "final_goals_a": ga, "btts_yes": btts})
        if out is None:
            pushes += 1
            continue
        graded += 1
        wins += 1 if out == 1 else 0
        d = by.setdefault(_directional_market(mkt, sugg), {"graded": 0, "wins": 0,
                            "profit": 0.0, "priced": 0, "fixtures": set()})
        d["fixtures"].add(fid)
        d["graded"] += 1
        d["wins"] += 1 if out == 1 else 0
        if odds is not None and math.isfinite(float(odds)) and float(odds) > 1.0:
            result_profit = float(odds) - 1.0 if out == 1 else -1.0
            profit += result_profit
            priced += 1
            d["profit"] += result_profit
            d["priced"] += 1

    if total == 0:
        msg = "📊 <b>Daily Digest</b>\nNo tips sent yesterday."
    else:
        lines = ["📊 <b>Daily Digest</b> (yesterday, Berlin time)",
                 f"Sent: {total}  •  Graded: {graded}  •  Pushed: {pushes}  •  "
                 f"Pending: {total - graded - pushes}"]
        lines.append(f"Fixtures: {len(fixtures)}")
        if graded:
            lines.append(f"Wins: {wins}  •  Accuracy: {100.0*wins/graded:.1f}%")
            for mk, st in sorted(by.items()):
                if st["graded"]:
                    lines.append(f"• {escape(mk)} — {st['wins']}/{st['graded']} "
                                 f"({100.0*st['wins']/st['graded']:.1f}%)"
                                 f" • {len(st['fixtures'])} fixtures"
                                 f" • {st['profit']:+.2f}u ({st['priced']} priced)")
        if priced:
            lines.append(f"💰 P&L (1u): {profit:+.2f}u  •  ROI: {100.0*profit/priced:+.1f}%"
                         f" • {priced} priced bets")
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
                # calibration_gap_pct = predicted - actual. Positive means
                # predicted > actual (overconfidence); negative means underconfidence.
                direction = "over" if gap > 0 else "under"
                lines.append(f"   • {escape(name)}: {gap:+.1f}pp {direction}confident "
                             f"→ EV overstated ~{abs(gap) * 2:.0f}pp at odds 2.0"
                             if gap > 0 else
                             f"   • {escape(name)}: {gap:+.1f}pp {direction}confident")
            lines.append("   Treat their EV as unproven until the gap closes.")

        send_telegram("\n".join(lines))
    except Exception as e:
        log.exception("[TRAIN] job failed: %s", e)
        send_telegram(f"❌ Training <b>FAILED</b>\n{escape(str(e))}")


def auto_tune_thresholds(days: int = 30) -> Dict[str, float]:
    """Keep threshold writes inside the train/calibration/holdout pipeline."""
    if not AUTO_TUNE_ENABLE:
        return {}
    log.warning("[AUTO-TUNE] write path disabled: thresholds are controlled by "
                "train_models.py holdout verification. No settings changed.")
    send_telegram("🔒 Auto-tune made no changes: threshold writes are restricted to the "
                  "holdout-verified training pipeline.")
    return {}


def retry_unsent_tips(minutes: int = 120, limit: int = 200) -> int:
    """Both scan paths send inline; this only catches Telegram outages."""
    if SHADOW_ONLY or not _delivery_allowed('prematch'):
        return 0
    cutoff = int(time.time()) - minutes * 60
    with db_conn() as c:
        rows = c.execute(
            "SELECT match_id,league,home,away,market,suggestion,confidence,score_at_tip,minute,"
            "created_ts,odds,book,ev_pct,fair_prob,stake_units,is_prematch,kickoff_ts,confidence_raw "
            "FROM tips WHERE sent_ok=0 AND created_ts >= %s ORDER BY created_ts ASC LIMIT %s",
            (cutoff, limit)).fetchall()

    retried = 0
    for (mid, league, home, away, market, sugg, conf, score, minute, cts, odds, book,
         ev_pct, fair, stake, is_pre, kickoff, model_prob) in rows:
        # A delayed message must not advertise an old, unavailable entry price.
        max_age = 300 if is_pre else LIVE_MAX_ODDS_AGE_SEC
        expired = (model_prob is None or time.time() - cts > max_age or (is_pre and (not kickoff or time.time() >= kickoff)))
        if not expired:
            current = _fixture_by_id(int(mid))
            status = ((current or {}).get('fixture') or {}).get('status') or {}
            expired = (not current or (status.get('short') != 'NS' if is_pre else
                       status.get('short') not in {'1H', 'HT', '2H'} or _pretty_score(current) != score
                       or status.get('elapsed') is None or int(status['elapsed']) - int(minute or 0) not in (0, 1)))
        if not expired:
            key, sel = _market_key_and_selection(market, sugg)
            ODDS_CACHE.invalidate((int(mid), not bool(is_pre)))
            entry = fetch_odds(int(mid), live=not bool(is_pre)).get(key) or {}
            gate = _price_gate(market, sugg, int(mid), float(model_prob), live=not bool(is_pre))
            quote = ((entry.get('by_book') or {}).get(sel) or {}).get(book)
            observed = entry.get('fetched_ts')
            expired = (not gate.get('passed') or quote is None or odds is None or float(quote) < float(odds)
                       or observed is None or not 0 <= time.time() - float(observed) <= LIVE_MAX_ODDS_AGE_SEC)
        if expired:
            with db_conn() as c:
                c.execute("UPDATE tips SET sent_ok=-1 WHERE match_id=%s AND created_ts=%s AND sent_ok=0", (mid, cts))
            continue
        kickoff_txt = "TBD"
        if kickoff:
            kickoff_txt = datetime.fromtimestamp(int(kickoff), TZ_UTC).astimezone(BERLIN_TZ).strftime("%H:%M")
        ok = _send_reserved_tip(mid, cts, _format_tip_message(
            home, away, league, int(minute or 0), score or "", sugg, float(conf), None,
            odds, book, ev_pct, fair, stake, kickoff_txt=kickoff_txt, prematch=bool(is_pre)))
        if ok:
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
        if LIVE_SCAN_ENABLE:
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
        if TRAIN_ENABLE and AUTO_TRAIN_ENABLE:
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
                 "odds_book_filter=%s exec_min_books=%d (live=%d) exec_max_outlier=%.1f%% "
                 "require_executable=%s concentration_mode=%s",
                 MIN_BOOKS_FOR_FAIR, MIN_BOOKS_FOR_FAIR_LIVE, bool(REQUIRE_FAIR_PRICE),
                 bool(ALLOW_TIPS_WITHOUT_ODDS), EDGE_MIN_BPS, FAIR_EDGE_MIN_BPS,
                 MAX_MODEL_EDGE_BPS, ODDS_BOOKMAKER_ID or "none",
                 MIN_BOOKS_FOR_EXECUTION, MIN_BOOKS_FOR_EXECUTION_LIVE, MAX_EXECUTION_OUTLIER_PCT,
                 bool(REQUIRE_EXECUTABLE_PRICE), bool(CONCENTRATION_MODE))
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
# Direct report access is read-only. Actions use the CSRF-protected control form.
_BROWSER_REPORTS = frozenset({
    'http_clv', 'http_clv_breakdown', 'http_price_gate', 'http_scan_funnel',
    'http_scan_decisions', 'http_shadow_report', 'http_calibration',
    'http_significance', 'http_league_breakdown', 'http_league_density',
    'http_odds_provider', 'http_thresholds', 'http_execution_receipt',
})


@app.before_request
def _browser_report_login():
    if (request.method == 'GET' and request.endpoint in _BROWSER_REPORTS
            and request.accept_mimetypes.best == 'text/html'
            and not request.headers.get('X-API-Key') and not request.is_json
            and not _dashboard_authed()):
        return redirect(url_for('dashboard_login', next='diagnostics'))


@app.after_request
def _private_report_headers(response):
    if request.endpoint in _BROWSER_REPORTS or request.path.startswith('/dashboard'):
        response.headers['Cache-Control'] = 'no-store'
        response.headers['Referrer-Policy'] = 'no-referrer'
    return response


def _require_admin():
    if getattr(g, "browser_control_endpoint", None) == request.endpoint:
        return
    if request.method == 'GET' and request.endpoint in _BROWSER_REPORTS and _dashboard_authed():
        return
    body = request.get_json(silent=True) if request.is_json else None
    key = (request.headers.get("X-API-Key")
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
    result = _run_with_pg_lock(1001, production_scan)
    if result is None:
        return jsonify({"ok": False, "error": "scan_already_running"}), 409
    s, l = result
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
        if not out.get("ok"):
            return jsonify({"ok": False, "result": out}), 500
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


@app.route("/admin/diagnostics/clv-breakdown", methods=["GET"])
def http_clv_breakdown():
    """CLV sliced by (market, league) with a 95% CI — which pocket has edge,
    not just whether the pooled average does."""
    _require_admin()
    return jsonify({"ok": True, "clv_breakdown": compute_clv_breakdown(
        days=_arg_int("days"), min_n=_arg_int("min_n", 20))})


@app.route("/admin/diagnostics/price-gate", methods=["GET"])
def http_price_gate():
    """Why candidates are not becoming tips, over a window rather than per scan."""
    _require_admin()
    return jsonify({"ok": True,
                    "price_gate": compute_price_gate_breakdown(
                        days=_arg_int("days", 7), phase=request.args.get("phase"))})


@app.route("/admin/diagnostics/funnel", methods=["GET"])
def http_scan_funnel():
    _require_admin()
    return jsonify({"ok": True, "funnel": compute_scan_funnel(days=_arg_int("days", 7))})


def compute_threshold_review(days=365):
    return _research().report('threshold')


@app.route('/admin/research/<kind>/start', methods=['POST'])
def http_start_research(kind):
    _require_admin()
    try:
        return jsonify({'ok': True, 'trial': _research().start(kind)})
    except ValueError as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 409


@app.route('/admin/diagnostics/release', methods=['GET'])
def http_release_status():
    _require_admin()
    return jsonify(_research().report('release'))


@app.route('/admin/research/threshold/adopt', methods=['POST'])
def http_adopt_thresholds():
    _require_admin()
    try:
        return jsonify(_research().adopt_thresholds())
    except ValueError as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 409


@app.route('/admin/execution-receipts', methods=['GET', 'POST'])
def http_execution_receipt():
    _require_admin()
    if request.method == 'GET':
        return jsonify(_research().execution_report())
    try:
        return jsonify(_research().record_receipt(request.get_json(silent=True) or {}))
    except (ValueError, TypeError, OverflowError) as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 400


@app.route('/admin/diagnostics/threshold-review', methods=['GET'])
def http_threshold_review():
    _require_admin()
    return jsonify({'ok': True, 'review': compute_threshold_review(_arg_int('days', 365))})


@app.route('/admin/diagnostics/decisions', methods=['GET'])
def http_scan_decisions():
    _require_admin()
    limit = min(1000, max(1, _arg_int('limit', 100)))
    sql = """SELECT d.id,d.scan_id,r.phase,d.created_ts,d.match_id,d.stage,d.market,d.suggestion,
        d.reason,d.prob,d.threshold_pct,d.odds,d.price_decision,d.qualified,d.detail
        FROM scan_decisions d JOIN scan_runs r ON r.scan_id=d.scan_id WHERE d.id > %s"""
    params = [max(0, _arg_int('after_id', 0))]
    for key, col in (('scan_id', 'd.scan_id'), ('match_id', 'd.match_id')):
        value = request.args.get(key)
        if value:
            if key == 'match_id' and not value.isdigit():
                return jsonify({'ok': False, 'error': 'match_id must be an integer'}), 400
            sql += f' AND {col}=%s'
            params.append(int(value) if key == 'match_id' else value)
    sql += ' ORDER BY d.id LIMIT %s'
    params.append(limit)
    with db_conn() as c:
        rows = c.execute(sql, tuple(params)).fetchall()
    names = ('id','scan_id','phase','created_ts','match_id','stage','market','suggestion',
             'reason','prob','threshold_pct','odds','price_decision','qualified','detail')
    return jsonify({'ok': True, 'decisions': [dict(zip(names, row)) for row in rows],
                    'next_after_id': rows[-1][0] if rows else None})


@app.route("/admin/diagnostics/shadow", methods=["GET"])
def http_shadow_report():
    _require_admin()
    return jsonify({"ok": True, "shadow": compute_shadow_report(days=_arg_int("days", 90))})


@app.route("/admin/diagnostics/live-stats", methods=["GET"])
def http_live_stats_diagnostic():
    """Why each fixture did or did not pass live-statistics coverage."""
    _require_admin()
    return jsonify({"ok": True, "live_stats": _live_stats_diagnostic_payload()})


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


@app.route('/admin/diagnostics/fixtures', methods=['GET'])
def http_fixture_lookup():
    _require_admin()
    day = request.args.get('date', datetime.now(BERLIN_TZ).date().isoformat())
    try:
        if datetime.strptime(day, '%Y-%m-%d').strftime('%Y-%m-%d') != day:
            raise ValueError()
    except ValueError:
        return jsonify({'ok': False, 'error': 'date must be YYYY-MM-DD'}), 400
    team = request.args.get('team', '').strip().casefold()
    js = _api_get(FOOTBALL_API_URL, {'date': day, 'timezone': 'Europe/Berlin'})
    if not isinstance(js, dict) or js.get('errors') or not isinstance(js.get('response'), list):
        return jsonify({'ok': False, 'error': 'fixture_feed_unavailable'}), 502
    fixtures = []
    for fx in js['response']:
        info, teams, league = fx.get('fixture') or {}, fx.get('teams') or {}, fx.get('league') or {}
        home, away = (teams.get('home') or {}).get('name', ''), (teams.get('away') or {}).get('name', '')
        if team and team not in home.casefold() and team not in away.casefold():
            continue
        fixtures.append({'fixture_id': info.get('id'), 'home': home, 'away': away,
                         'kickoff': info.get('date'), 'kickoff_ts': info.get('timestamp'),
                         'status': (info.get('status') or {}).get('short'),
                         'league_id': league.get('id'), 'league': league.get('name')})
    return jsonify({'ok': True, 'date': day, 'timezone': 'Europe/Berlin',
                    'count': len(fixtures), 'fixtures': fixtures})


@app.route("/admin/diagnostics/odds-provider", methods=["GET", "POST"])
def http_odds_provider():
    _require_admin()
    # GET with ?fixture_id=... is intentionally read-only so the same
    # fixture-level trace is usable from a phone browser; POST remains
    # supported for scripted checks.
    if request.method == 'POST' or request.args.get('fixture_id'):
        if THE_ODDS_API_MODE == 'off':
            return jsonify({'ok': False, 'error': 'provider_disabled'}), 409
        body = request.get_json(silent=True) or {}
        fixture_arg = body.get('fixture_id', request.args.get('fixture_id'))
        if fixture_arg is not None:
            try:
                fid = int(fixture_arg)
                if fid <= 0:
                    raise ValueError()
            except (TypeError, ValueError):
                return jsonify({'ok': False, 'error': 'invalid_fixture_id'}), 400
            rows = _external_odds_rows(fid)
            return jsonify({'ok': bool(rows), 'mode': THE_ODDS_API_MODE,
                            'provider': _THE_ODDS_FEED.status, 'quotes': rows})
        result = _THE_ODDS_FEED.check()
        return jsonify({'ok': result['status'] == 'ok', 'mode': THE_ODDS_API_MODE, 'provider': result})
    with db_conn() as c:
        row = c.execute("SELECT value FROM settings WHERE key='odds_api:budget'").fetchone()
    return jsonify({'mode': THE_ODDS_API_MODE, 'provider': _THE_ODDS_FEED.status,
                    'reserved_budget': json.loads(row[0]) if row else {},
                    'daily_limit': THE_ODDS_API_DAILY_CREDITS,
                    'monthly_limit': THE_ODDS_API_MONTHLY_CREDITS})


@app.route("/admin/diagnostics/league-density", methods=["GET"])
def http_league_density():
    """Settled-fixture volume per league — pick CONCENTRATION_MODE's league
    list from this, not a guess."""
    _require_admin()
    return jsonify({"ok": True, "league_density": compute_league_density(
        days=_arg_int("days"), min_n=_arg_int("min_n", 20))})


@app.route("/admin/thresholds", methods=["GET"])
def http_thresholds():
    """
    Every market's live threshold in one view, with derived markets flagged.

    A derived market showing "SUPPRESSED (never verified)" means training has
    not written a threshold for it — it will not fire, which is correct until
    it has passed a holdout.
    """
    _require_admin()
    markets = ["BTTS Yes", "BTTS No", "1X2", "Double Chance", "Draw No Bet"] + \
              [x for l in OU_LINES for x in (f"Over {_fmt_line(l)}", f"Under {_fmt_line(l)}")]
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
                "status": ("SUPPRESSED (never verified)"
                           if raw is None and (mk in DERIVED_MARKETS or mk in ("BTTS Yes", "BTTS No")
                                               or mk.startswith("Over ") or mk.startswith("Under "))
                           else "suppressed" if effective >= SUPPRESSED_THRESHOLD_PCT
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
        "concentration": {
            "enabled": bool(CONCENTRATION_MODE),
            "active_markets": sorted(ACTIVE_MARKETS),
            "league_allow_ids": LEAGUE_ALLOW_IDS,
            "prematch_league_ids": PREMATCH_LEAGUE_IDS,
            "max_active_leagues": MAX_ACTIVE_LEAGUES,
        },
        "prematch_collection": {
            "enabled": bool(PREMATCH_SCAN_ENABLE),
            "lookahead_hours": PREMATCH_LOOKAHEAD_HOURS,
            "scan_interval_min": PREMATCH_SCAN_INTERVAL_MIN,
        },
        "fair_price": {
            "required": REQUIRE_FAIR_PRICE,
            "min_books": MIN_BOOKS_FOR_FAIR,
            "min_books_live": MIN_BOOKS_FOR_FAIR_LIVE,
            "live_feed_sources": 1,
            "live_blocked_by_source_count": REQUIRE_FAIR_PRICE and MIN_BOOKS_FOR_FAIR_LIVE > 1,
        },
        "execution_realism": {
            "min_books": MIN_BOOKS_FOR_EXECUTION,
            "min_books_live": MIN_BOOKS_FOR_EXECUTION_LIVE,
            "max_outlier_pct": MAX_EXECUTION_OUTLIER_PCT,
            "require_executable_price": bool(REQUIRE_EXECUTABLE_PRICE),
        },
        "last_training_run": metrics,
        "api_usage": _api_call_stats_snapshot(),
    })


@app.route("/settings/<path:key>", methods=["GET", "POST"])
def http_settings(key: str):
    _require_admin()
    if (request.method != 'GET' or request.args.get('value') is not None):
        if key.startswith(('research:', 'research_threshold:', 'model', 'conf_threshold')) and (
                _research().trial('release') or _research().trial('threshold')):
            return jsonify({'ok': False, 'error': 'settings frozen by research trial'}), 409
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
# Browser reports and manual controls. The admin key is checked once
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
            return redirect(url_for({"diagnostics": "dashboard_diagnostics", "controls": "dashboard_controls"}.get(request.args.get("next"), "dashboard")))
        _login_record_failure(ip)
        return render_template("dashboard_login.html", error="Incorrect key."), 401
    if _dashboard_authed():
        return redirect(url_for({"diagnostics": "dashboard_diagnostics", "controls": "dashboard_controls"}.get(request.args.get("next"), "dashboard")))
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


# Explicitly authenticated admin routes only; never proxy arbitrary URLs.
def _browser_control_routes():
    return {r.rule: r for r in app.url_map.iter_rules()
            if (r.rule.startswith('/admin/') or r.rule in ('/init-db', '/settings/<path:key>'))
            and r.endpoint.startswith('http_')}


@app.route('/dashboard/controls', methods=['GET', 'POST'])
def dashboard_controls():
    if not DASHBOARD_ENABLED:
        return _dashboard_unavailable()
    if not _dashboard_authed():
        return redirect(url_for('dashboard_login', next='controls'))
    routes = _browser_control_routes()
    result = None
    status = 200
    if request.method == 'POST':
        token = flask_session.get('control_csrf', '')
        if not token or not _safe_compare(request.form.get('csrf', ''), token):
            abort(403)
        # Rotate before dispatch; a refreshed result page cannot repeat the action.
        flask_session['control_csrf'] = secrets.token_urlsafe(32)
        rule = routes.get(request.form.get('route', ''))
        method = request.form.get('method', '')
        if rule is None or method not in ('GET', 'POST') or method not in rule.methods:
            abort(400)
        try:
            params = json.loads(request.form.get('parameters', '{}') or '{}')
            query = json.loads(request.form.get('query', '{}') or '{}')
            body = json.loads(request.form.get('body', '{}') or '{}')
            if not all(isinstance(v, dict) for v in (params, query, body)):
                raise ValueError('Inputs must be JSON objects.')
            if set(params) != set(rule.arguments):
                raise ValueError('Supply exactly these path parameters: ' + ', '.join(sorted(rule.arguments)))
            # No arbitrary endpoint, host, header, URL or credentials can be supplied.
            target = url_for(rule.endpoint, **params)
            adapter = app.url_map.bind('localhost')
            matched, values = adapter.match(target, method=method)
            if matched != rule.endpoint:
                raise ValueError('Path parameters do not match the selected endpoint.')
            with app.app_context(), app.test_request_context(target, method=method, query_string=query, json=body):
                g.browser_control_endpoint = matched
                response = app.make_response(app.view_functions[matched](**values))
                status = response.status_code
                result = response.get_json(silent=True)
                if result is None:
                    result = {'status': status, 'message': 'Endpoint returned a non-JSON response.'}
        except HTTPException as exc:
            status, result = exc.code, {'ok': False, 'error': exc.description}
        except (ValueError, TypeError) as exc:
            status, result = 400, {'ok': False, 'error': str(exc)}
        except Exception:
            log.exception('[BROWSER CONTROL] action failed')
            status, result = 500, {'ok': False, 'error': 'Action failed. Check the application logs.'}
    if 'control_csrf' not in flask_session:
        flask_session['control_csrf'] = secrets.token_urlsafe(32)
    response = app.make_response((render_template_string("""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>GoalSniper manual controls</title><style>
body{font:16px system-ui;background:#101820;color:#edf3f8;max-width:800px;margin:auto;padding:18px}
a{color:#9dd9ff}details,section{background:#1c2936;border-radius:12px;padding:16px;margin:14px 0}
button,textarea,select{font:inherit;box-sizing:border-box;width:100%;padding:12px;margin:8px 0}
button{background:#9dd9ff;color:#101820;border:0;border-radius:8px;font-weight:bold}
pre{white-space:pre-wrap;overflow-wrap:anywhere}summary{overflow-wrap:anywhere;cursor:pointer}label{display:block}
</style></head><body><h1>Manual controls</h1>
<p><a href="{{ url_for('dashboard_diagnostics') }}">Diagnostics</a> · <a href="{{ url_for('dashboard_controls') }}">Fresh controls page</a></p>
<p>Opening this page runs nothing. Each Run button executes the selected endpoint with your existing permissions and research safeguards.</p>
<p>Shadow only: <strong>{{ shadow }}</strong>. Training replaces compatible models only if the research rules permit it.</p>
{% if result is not none %}<section><h2>Result — HTTP {{ status }}</h2><pre>{{ result }}</pre></section>{% endif %}
<p>For training, open <strong>/admin/train</strong> and tap Run. Leave the JSON fields unchanged. Wait for the result; a timeout does not prove the job stopped. Check Status or logs before retrying.</p>
<p>Scans and backfills can use provider credits. Train-notify, digest, retry-unsent and MOTD can send Telegram messages. Starting a research trial freezes its scope; adopting thresholds and writing settings change configuration.</p>
{% for path, rule in routes %}<details {% if path == '/admin/train' %}open{% endif %}><summary>{{ path }}</summary>
<form method="post" action="{{ url_for('dashboard_controls') }}">
<input type="hidden" name="csrf" value="{{ csrf }}"><input type="hidden" name="route" value="{{ path }}">
<label>Method<select name="method">{% for method in ['POST','GET'] if method in rule.methods %}<option>{{ method }}</option>{% endfor %}</select></label>
{% if rule.arguments %}<label>Path parameters (JSON)<textarea name="parameters">{{ examples.get(path, {})|tojson }}</textarea></label>{% else %}<input type="hidden" name="parameters" value="{}">{% endif %}
<label>Query parameters (JSON)<textarea name="query">{}</textarea></label>
<label>Request body (JSON)<textarea name="body">{}</textarea></label>
<button type="submit">Run {{ path }}</button></form></details>{% endfor %}
</body></html>""", routes=sorted(routes.items()), csrf=flask_session['control_csrf'],
        examples={'/admin/research/<kind>/start': {'kind': 'release'},
                  '/settings/<path:key>': {'key': 'YOUR_SETTING_NAME'},
                  '/admin/tip-audit/<int:fid>': {'fid': 0}},
        result=json.dumps(result, indent=2, default=str) if result is not None else None,
        status=status, shadow=SHADOW_ONLY), status))
    response.headers['Content-Security-Policy'] = "default-src 'none'; style-src 'unsafe-inline'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response


@app.route("/dashboard/diagnostics")
def dashboard_diagnostics():
    """Read-only mobile view. Never exposes credentials or changes trial scope."""
    if not DASHBOARD_ENABLED:
        return _dashboard_unavailable()
    if not _dashboard_authed():
        return redirect(url_for('dashboard_login', next='diagnostics'))
    density = compute_league_density(days=365)
    with db_conn() as c:
        saved = c.execute("SELECT value FROM settings WHERE key='research:league_scope'").fetchone()
        trials = c.execute('SELECT kind,created_ts FROM research_trials ORDER BY kind').fetchall()
    try:
        saved_scope = json.loads(saved[0]) if saved else []
    except (ValueError, TypeError):
        saved_scope = 'Invalid saved value; inspect configuration'
    labels = {5: 'UEFA Nations League', 39: 'Premier League', 140: 'La Liga',
              135: 'Serie A', 78: 'Bundesliga', 61: 'Ligue 1',
              88: 'Eredivisie', 94: 'Primeira Liga'}
    payload = {'selected_scope': list(LEAGUE_ALLOW_IDS),
               'prematch_scope': list(PREMATCH_LEAGUE_IDS), 'saved_scope': saved_scope,
               'environment_scope_empty': not bool(os.getenv('LEAGUE_ALLOW_IDS', '').strip())
                   and not bool(os.getenv('PREMATCH_LEAGUE_IDS', '').strip()),
               'max_active_leagues': MAX_ACTIVE_LEAGUES,
               'shadow_only': SHADOW_ONLY, 'odds_mode': THE_ODDS_API_MODE,
               'frozen_trials': [{'kind': r[0], 'created_ts': r[1]} for r in trials],
               'league_density': density['leagues']}
    response = app.make_response(render_template_string("""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>GoalSniper diagnostics</title><style>
body{font:16px system-ui,sans-serif;background:#101820;color:#edf3f8;margin:0;padding:20px;max-width:780px;margin-inline:auto}
h1{font-size:26px}h2{font-size:20px}section{background:#1c2936;border-radius:12px;padding:18px;margin:16px 0}
a{color:#9dd9ff}li{margin:10px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}
td,th{text-align:left;padding:10px 8px;border-bottom:1px solid #425261}table{width:100%;border-collapse:collapse}
.note{color:#c4d1dd;line-height:1.5}</style></head><body>
<h1>GoalSniper diagnostics</h1><p class="note">Read-only. Opening this page does not change leagues or make provider requests.</p>
<p><a href="{{ url_for('dashboard_controls') }}">Manual controls — training, scans and all admin endpoints</a></p>
<section><h2>Browser checks</h2><ul>
<li><a href="{{ url_for('http_league_density', days=365) }}">League density (JSON)</a></li>
<li><a href="{{ url_for('http_scan_funnel', days=7) }}">Scan and rejection counts</a></li>
<li><a href="{{ url_for('http_shadow_report', days=90) }}">Shadow predictions</a></li>
<li><a href="{{ url_for('http_odds_provider') }}">Odds provider status</a></li>
<li><a href="{{ url_for('http_thresholds') }}">Selection thresholds</a></li>
</ul></section><section><h2>Active leagues</h2><ul>{% for lid in payload.selected_scope %}
<li>{{ lid }} — {{ labels.get(lid|int, 'Other competition') }}</li>
{% else %}<li>No active leagues selected. Scans remain blocked.</li>{% endfor %}</ul>
<p>Prematch IDs: {{ payload.prematch_scope|join(', ') or 'None' }}</p>
<p>Maximum: {{ payload.max_active_leagues }} competitions.</p>
<p>Both Railway league variables empty: {{ 'Yes' if payload.environment_scope_empty else 'No' }}</p>
<p>Database selection: {{ payload.saved_scope }}</p></section>
<section><h2>Research status</h2><p>Shadow only: {{ 'Yes' if payload.shadow_only else 'No' }}</p>
<p>Odds provider mode: {{ payload.odds_mode }}</p>
<p>Frozen trials: {% for trial in payload.frozen_trials %}{{ trial.kind }}{% if not loop.last %}, {% endif %}{% else %}None started{% endfor %}</p>
<p class="note">Nations League has provider mapping support. It is only active if ID 5 appears above. A change of scope must preserve existing research records.</p></section>
<section><h2>Stored results, past 365 days</h2><p class="note">Counts use the date a result was recorded. They do not prove model or odds coverage.</p>
<table><thead><tr><th>Competition</th><th>Settled fixtures</th></tr></thead><tbody>
{% for row in payload.league_density %}<tr><td>{{ row.league_id }} — {{ labels.get(row.league_id, 'Other') }}</td><td>{{ row.n_settled_fixtures }}</td></tr>{% endfor %}
</tbody></table></section><details><summary>Diagnostic JSON</summary><pre>{{ diagnostic_json }}</pre></details>
<p><a href="{{ url_for('dashboard_diagnostics') }}">Refresh</a> · <a href="{{ url_for('dashboard') }}">Dashboard</a></p>
</body></html>""", payload=payload, labels=labels, diagnostic_json=json.dumps(payload, indent=2)))
    response.headers['Cache-Control'] = 'no-store'
    response.headers['Referrer-Policy'] = 'no-referrer'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Content-Security-Policy'] = "default-src 'none'; style-src 'unsafe-inline'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
    return response


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
            "clv_pct,is_prematch,sent_ok,decision_ts,stats_fetched_ts,odds_fetched_ts,"
            "telegram_sent_ts,price_verified "
            "FROM tips WHERE suggestion<>'HARVEST' ORDER BY created_ts DESC LIMIT %s", (limit,)
        ).fetchall()
    keys = ["match_id", "league", "home", "away", "market", "suggestion", "confidence",
            "score_at_tip", "minute", "created_ts", "odds", "book", "ev_pct", "fair_prob",
            "stake_units", "clv_pct", "is_prematch", "sent_ok", "decision_ts",
            "stats_fetched_ts", "odds_fetched_ts", "telegram_sent_ts", "price_verified"]
    tips = [dict(zip(keys, r)) for r in rows]
    try:
        pnl = compute_pnl(days=days, stake=1.0)
    except Exception as e:
        log.warning("[DASHBOARD] pnl computation failed: %s", e)
        pnl = {"error": str(e)}
    return jsonify({"ok": True, "tips": tips, "pnl": pnl, "build": build_info(),
                    "server_ts": int(time.time())})


def _tip_audit_payload(fid: int) -> Tuple[Dict[str, Any], int]:
    with db_conn() as c:
        row = c.execute(
            "SELECT match_id,home,away,market,suggestion,created_ts,decision_ts,"
            "stats_fetched_ts,odds_fetched_ts,telegram_sent_ts,price_verified,audit_json "
            "FROM tips WHERE match_id=%s AND suggestion<>'HARVEST' "
            "ORDER BY created_ts DESC LIMIT 1", (fid,)).fetchone()
    if not row:
        return {"ok": False, "error": "tip_not_found", "match_id": fid}, 404
    keys = ["match_id", "home", "away", "market", "suggestion", "created_ts",
            "decision_ts", "stats_fetched_ts", "odds_fetched_ts", "telegram_sent_ts",
            "price_verified", "audit"]
    out = dict(zip(keys, row))
    raw_audit = out.get("audit")
    if raw_audit:
        try:
            out["audit"] = json.loads(raw_audit)
        except Exception:
            out["audit"] = {"error": "stored audit JSON is unreadable"}
    return {"ok": True, "tip": out, "server_ts": int(time.time())}, 200


@app.route("/dashboard/tip-audit/<int:fid>")
def dashboard_tip_audit(fid: int):
    if not DASHBOARD_ENABLED:
        return _dashboard_unavailable()
    if not _dashboard_authed():
        abort(401)
    payload, status = _tip_audit_payload(fid)
    return jsonify(payload), status


@app.route("/admin/tip-audit/<int:fid>")
def admin_tip_audit(fid: int):
    _require_admin()
    payload, status = _tip_audit_payload(fid)
    return jsonify(payload), status


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


@app.route("/dashboard/live-stats")
def dashboard_live_stats():
    """Phone-friendly statistics diagnostics using the dashboard session."""
    if not DASHBOARD_ENABLED:
        return _dashboard_unavailable()
    if not _dashboard_authed():
        abort(401)
    return jsonify({"ok": True, "live_stats": _live_stats_diagnostic_payload(),
                    "server_ts": int(time.time())})


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
    stats_diagnostics: List[Dict[str, Any]] = []
    result = _run_with_pg_lock(1001, score_live_matches_now, stats_diagnostics)
    if result is None:
        return jsonify({"ok": False, "error": "scan_already_running"}), 409
    matches, live_seen = result
    no_coverage = sum(1 for row in stats_diagnostics
                      if row.get("reason") not in ("usable", "before_required_minute"))
    _set_live_snapshot(matches, live_seen=live_seen, no_coverage=no_coverage,
                       stats_diagnostics=stats_diagnostics)
    return jsonify({"ok": True, **_get_live_snapshot(), "live_seen": live_seen,
                    "server_ts": int(time.time())})


@app.route("/tips/latest")
def http_latest():
    _require_admin()
    limit = max(1, min(500, _arg_int("limit", 50) or 50))
    with db_conn() as c:
        rows = c.execute(
            "SELECT match_id,league,home,away,market,suggestion,confidence,confidence_raw,"
            "score_at_tip,minute,created_ts,odds,book,ev_pct,fair_prob,stake_units,clv_pct,is_prematch,"
            "decision_ts,stats_fetched_ts,odds_fetched_ts,telegram_sent_ts,price_verified "
            "FROM tips WHERE suggestion<>'HARVEST' ORDER BY created_ts DESC LIMIT %s", (limit,)).fetchall()
    keys = ["match_id", "league", "home", "away", "market", "suggestion", "confidence",
            "confidence_raw", "score_at_tip", "minute", "created_ts", "odds", "book",
            "ev_pct", "fair_prob", "stake_units", "clv_pct", "is_prematch", "decision_ts",
            "stats_fetched_ts", "odds_fetched_ts", "telegram_sent_ts", "price_verified"]
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
                result = _run_with_pg_lock(1001, production_scan)
                if result is None:
                    send_telegram("⏳ A scan is already running.")
                else:
                    s, l = result
                    send_telegram(f"🔁 Scan done. Saved: {s}, Live seen: {l}")
            else:
                send_telegram("🔒 Admin key required.")
    except Exception as e:
        log.warning("telegram webhook parse error: %s", e)
    return jsonify({"ok": True})


# ───────── Boot ─────────
def _on_boot():
    # Runs first, before any DB connection is opened: a misconfigured
    # CONCENTRATION_MODE should fail fast and cheap, not after the pool and
    # schema are already up.
    _enforce_concentration_scope()
    _init_pool()
    init_db()
    _select_density_scope()
    set_setting("boot_ts", str(int(time.time())))


# Order matters: the schema must exist before any scheduled job can run.
_on_boot()
_start_scheduler_once()

if __name__ == "__main__":
    app.run(host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8080")))
