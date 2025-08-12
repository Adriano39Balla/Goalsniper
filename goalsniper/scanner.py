from __future__ import annotations

import os
import asyncio
import inspect
from datetime import datetime, timezone
from typing import Dict, List, Callable, Optional, Tuple

import httpx

from .logger import log, warn
from . import api_football as api
from . import tips as tip_engine
from .storage import (
    insert_tip_return_id,
    fixture_ever_sent,
    count_sent_today,
    has_fixture_tip_recent,   # recent-window dedupe
)
from .config import (
    MIN_CONFIDENCE_TO_SEND,
    MAX_TIPS_PER_RUN,
    LIVE_ENABLED,
    LIVE_MINUTE_MIN,
    LIVE_MINUTE_MAX,
    LIVE_MAX_FIXTURES,
    SCAN_DAYS,                  # kept for info
    DAILY_TIP_CAP,
    DUPLICATE_SUPPRESS_FOREVER,
    STATS_REQUEST_DELAY_MS,
    DUPLICATE_SUPPRESS_MINUTES, # for has_fixture_tip_recent
)
from .learning import calibrate_probability, dynamic_conf_threshold

# -------------------- additional caps (env-driven) --------------------
MAX_TODAY_FIXTURES = max(0, int(os.getenv("MAX_TODAY_FIXTURES", "24")))
BUILD_FIXTURE_LIMIT = max(0, int(os.getenv("BUILD_FIXTURE_LIMIT", "20")))

# ---------- status sets ----------
FINISHED_STATES = {"FT", "AET", "PEN", "CANC", "PST", "ABD", "AWD", "WO"}
LIVE_STATES     = {"1H", "2H", "HT", "ET"}
NS_STATES       = {"NS"}

# ---------- report which functions we actually use ----------
FOUND_BUILDER_NAME: Optional[str] = None
FOUND_LIVE_API_NAME: Optional[str] = None
FOUND_DATE_API_NAME: Optional[str] = None

# ---------- flexible resolver helpers ----------
def _resolve_fn_with_name(mod, candidates, required=True):
    for name in candidates:
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn, name
    if required:
        warn(f"Missing functions. Tried: {candidates} in module {mod.__name__}")
    return None, None

def _get_live_api() -> Optional[Callable]:
    global FOUND_LIVE_API_NAME
    fn, nm = _resolve_fn_with_name(
        api,
        ["get_live_fixtures", "fetch_live_fixtures", "live_fixtures", "get_fixtures_live"],
        required=False,
    )
    FOUND_LIVE_API_NAME = nm
    return fn

def _get_by_date_api() -> Optional[Callable]:
    global FOUND_DATE_API_NAME
    fn, nm = _resolve_fn_with_name(
        api,
        ["get_fixtures_by_date", "fixtures_by_date", "get_fixtures_date", "fetch_fixtures_by_date"],
        required=False,
    )
    FOUND_DATE_API_NAME = nm
    return fn

def _get_build_tips_fn() -> Optional[Callable]:
    global FOUND_BUILDER_NAME
    candidates = [
        "build_tips_for_fixture", "build_for_fixture",
        "generate_tips_for_fixture", "generate_tips",
        "build_tips", "make_tips", "predict_fixture",
        "predict_tips", "tips_for_fixture", "create_tips_for_fixture",
    ]
    fn, nm = _resolve_fn_with_name(tip_engine, candidates, required=False)
    FOUND_BUILDER_NAME = nm
    return fn

# ---------- async call helpers ----------
async def _maybe_await(x):
    return await x if inspect.isawaitable(x) else x

async def _call_fn(fn, *args, **kwargs):
    return await _maybe_await(fn(*args, **kwargs))

# ---------- small utils ----------
def _status(fx: Dict) -> str:
    return (fx.get("fixture", {}).get("status", {}) or {}).get("short") or ""

def _elapsed_minute(fx: Dict) -> int:
    return int((fx.get("fixture", {}).get("status", {}) or {}).get("elapsed") or 0)

def _fixture_id(fx_or_tip: Dict) -> int:
    if "fixtureId" in fx_or_tip:
        return int(fx_or_tip["fixtureId"])
    return int((fx_or_tip.get("fixture", {}) or {}).get("id") or 0)

def _league_id(fx: Dict) -> int:
    return int((fx.get("league", {}) or {}).get("id") or 0)

def _season(fx: Dict) -> int:
    return int((fx.get("league", {}) or {}).get("season") or 0)

# ---------- league filtering (reuse api_football env policy) ----------
def _flag_to_iso2(flag: str) -> str:
    cps = [ord(c) - 0x1F1E6 for c in flag if 0x1F1E6 <= ord(c) <= 0x1F1FF]
    if len(cps) == 2:
        return chr(cps[0] + 65) + chr(cps[1] + 65)
    return flag.upper()

def _normalize_country_flags(flags_csv: str) -> set[str]:
    out = set()
    for chunk in (flags_csv or "").split(","):
        s = chunk.strip()
        if not s:
            continue
        if any(0x1F1E6 <= ord(c) <= 0x1F1FF for c in s):
            out.add(_flag_to_iso2(s))
        else:
            out.add(s.upper())
    return out

def _csv_upper(s: str) -> list[str]:
    return [x.strip().upper() for x in (s or "").split(",") if x.strip()]

_ALLOW_COUNTRIES = _normalize_country_flags(getattr(api, "COUNTRY_FLAGS_ALLOW", ""))
_ALLOW_KEYS = _csv_upper(getattr(api, "LEAGUE_ALLOW_KEYWORDS", ""))
_EX_KEYS    = _csv_upper(getattr(api, "EXCLUDE_KEYWORDS", ""))

_COUNTRY_TO_ISO = {
    "GERMANY":"DE","ENGLAND":"GB","FRANCE":"FR","SPAIN":"ES","NETHERLANDS":"NL","HOLLAND":"NL",
    "EGYPT":"EG","BELGIUM":"BE","CHINA":"CN","AUSTRALIA":"AU","ITALY":"IT","CROATIA":"HR",
    "AUSTRIA":"AT","PORTUGAL":"PT","ROMANIA":"RO","SCOTLAND":"GB","SWEDEN":"SE","SWITZERLAND":"CH",
    "TURKEY":"TR","USA":"US","UNITED STATES":"US"
}

def _league_row_allowed(row: Dict) -> bool:
    lname = (row.get("leagueName") or "").upper()
    country = (row.get("country") or "").upper()
    if not lname:
        return False
    for bad in _EX_KEYS:
        if bad and bad in lname:
            return False
    if _ALLOW_KEYS and not any(k in lname for k in _ALLOW_KEYS):
        return False
    iso = _COUNTRY_TO_ISO.get(country, country)
    if _ALLOW_COUNTRIES and iso not in _ALLOW_COUNTRIES:
        return False
    return True

# ---------- fetchers ----------
async def _fetch_live(client: httpx.AsyncClient) -> List[Dict]:
    fn = _get_live_api()
    if not fn:
        warn("No live fixtures API function available; skipping live scan")
        return []
    data = await _call_fn(fn, client)
    out = []
    for f in data[: LIVE_MAX_FIXTURES]:
        st = _status(f)
        if st in FINISHED_STATES or st not in LIVE_STATES:
            continue
        minute = _elapsed_minute(f)
        if minute < LIVE_MINUTE_MIN or minute > LIVE_MINUTE_MAX:
            continue
        out.append(f)
    return out

async def _fetch_today(client: httpx.AsyncClient) -> List[Dict]:
    """Fetch current leagues; then today’s fixtures per allowed league."""
    if MAX_TODAY_FIXTURES <= 0:
        return []
    get_leagues = getattr(api, "get_current_leagues", None)
    by_date_fn  = _get_by_date_api()
    if not get_leagues or not by_date_fn:
        warn("No fixtures-by-date path available; skipping scheduled scan")
        return []

    leagues = await _call_fn(get_leagues, client)
    if not leagues:
        return []

    leagues = [row for row in leagues if _league_row_allowed(row)]
    if not leagues:
        return []

    out: List[Dict] = []
    day = datetime.now(timezone.utc).date().isoformat()

    for row in leagues:
        if len(out) >= MAX_TODAY_FIXTURES:   # HARD CAP
            break
        try:
            lid = int(row.get("leagueId") or 0)
            season = int(row.get("season") or 0)
            if not lid or not season:
                continue
            res = await _call_fn(by_date_fn, client, lid, season, day)
            for f in (res or []):
                if len(out) >= MAX_TODAY_FIXTURES:
                    break
                st = _status(f)
                if st in FINISHED_STATES:
                    continue
                if st not in NS_STATES:
                    continue
                out.append(f)
        except Exception as e:
            warn("fixtures_by_date failed:", str(e))
        await asyncio.sleep(max(0, STATS_REQUEST_DELAY_MS) / 1000.0)

    return out

# ---------- tip generation ----------
def _dedupe_in_run(candidates: List[Dict]) -> Tuple[List[Dict], int]:
    seen = set()
    out: List[Dict] = []
    removed = 0
    for t in candidates:
        fid = _fixture_id(t)
        if fid in seen:
            removed += 1
            continue
        seen.add(fid)
        out.append(t)
    return out, removed

async def _invoke_builder(build_fn: Callable, client: httpx.AsyncClient, fixture: Dict) -> Optional[List[Dict]]:
    params = list(inspect.signature(build_fn).parameters.keys())
    kwargs = {}
    if "league_id" in params:
        kwargs["league_id"] = _league_id(fixture)
    if "season" in params:
        kwargs["season"] = _season(fixture)
    if "fixture_id" in params:
        kwargs["fixture_id"] = _fixture_id(fixture)

    return await _call_fn(build_fn, client, fixture, **kwargs)

async def _build_tips_for_fixtures(client: httpx.AsyncClient, fixtures: List[Dict]) -> Tuple[List[Dict], Dict[str, int]]:
    stats = {"raw_candidates": 0, "low_conf": 0, "inrun_dedup_removed": 0}
    build_fn = _get_build_tips_fn()
    if not build_fn:
        warn("No tip builder function found in tips module; skipping tip generation")
        return [], stats
    if BUILD_FIXTURE_LIMIT <= 0:
        return [], stats

    raw: List[Dict] = []
    for f in fixtures[:BUILD_FIXTURE_LIMIT]:
        try:
            res = await _invoke_builder(build_fn, client, f)
            if res:
                for t in res:
                    t.setdefault("fixtureId", _fixture_id(f))
                    t.setdefault("leagueId", _league_id(f))
                    t.setdefault("season", _season(f))
                raw.extend(res)
        except Exception as e:
            warn("Tip build failed for fixture", _fixture_id(f), str(e))
        await asyncio.sleep(max(0, STATS_REQUEST_DELAY_MS) / 1000.0)

    deduped, removed = _dedupe_in_run(raw)
    stats["raw_candidates"] = len(deduped)
    stats["inrun_dedup_removed"] = removed
    return deduped, stats

# ---------- learning gate ----------
async def _calibrate_and_filter(t: dict) -> Optional[dict]:
    """Calibrate using history and enforce a dynamic per‑market threshold."""
    try:
        p_raw = float(t.get("probability", 0.5))
        market = str(t.get("market", ""))
        league_id = t.get("leagueId")
        p_cal = await calibrate_probability(market, p_raw, league_id)
        conf = abs(p_cal - 0.5) * 2.0
        thr = await dynamic_conf_threshold(market)
        if conf < max(thr, MIN_CONFIDENCE_TO_SEND):
            return None
        out = dict(t)
        out["probability"] = round(p_cal, 3)
        out["confidence"] = round(conf, 3)
        return out
    except Exception as e:
        warn("learning gate failed:", str(e))
        return None

# ---------- sending ----------
async def _send_one(client: httpx.AsyncClient, tip: Dict) -> bool:
    try:
        from . import telegram as tg
    except Exception as e:
        warn("telegram module import failed:", str(e))
        return False

    msg_id = await tg.send_tip_plain(client, tip)
    tip_id = await insert_tip_return_id({**tip, "messageId": msg_id}, msg_id)

    try:
        await tg.attach_feedback_buttons(client, msg_id, tip_id)
    except Exception as e:
        warn("attach feedback buttons failed:", str(e))

    log(
        f"Sent tip id={tip_id} fixture={tip['fixtureId']} "
        f"{tip['market']} {tip['selection']} conf={tip['confidence']:.2f}"
    )
    return True

async def _recently_sent(fid: int) -> bool:
    """Optional short-window dedupe if FOREVER suppression is off."""
    if DUPLICATE_SUPPRESS_FOREVER:
        return False
    if DUPLICATE_SUPPRESS_MINUTES <= 0:
        return False
    return await has_fixture_tip_recent(fid, DUPLICATE_SUPPRESS_MINUTES)

async def _send_tips(client: httpx.AsyncClient, tips: List[Dict], start_count: Optional[int] = None) -> Tuple[int, Dict[str, int]]:
    stats = {"ever_sent_skipped": 0, "daily_cap_hits": 0, "low_conf": 0}
    sent = 0
    if start_count is None:
        used_today = await count_sent_today()
    else:
        used_today = int(start_count)

    for t in tips:
        if used_today >= DAILY_TIP_CAP:
            stats["daily_cap_hits"] += 1
            log(f"Daily cap reached ({DAILY_TIP_CAP}). Stopping.")
            break

        fid = _fixture_id(t)
        if DUPLICATE_SUPPRESS_FOREVER and fid and await fixture_ever_sent(fid):
            stats["ever_sent_skipped"] += 1
            continue
        if fid and await _recently_sent(fid):
            stats["ever_sent_skipped"] += 1
            continue

        tt = await _calibrate_and_filter(t)
        if not tt:
            stats["low_conf"] += 1
            continue

        try:
            if await _send_one(client, tt):
                sent += 1
                used_today += 1
                if sent >= MAX_TIPS_PER_RUN:
                    break
        except Exception as e:
            warn("Send failed:", str(e))

    return sent, stats

# ---------- public entrypoint ----------
async def run_scan_and_send() -> Dict[str, int]:
    started = datetime.now(timezone.utc)

    live_fixtures: List[Dict] = []
    today_fixtures: List[Dict] = []

    async with httpx.AsyncClient(timeout=30) as client:
        if LIVE_ENABLED:
            try:
                live_fixtures = await _fetch_live(client)
            except Exception as e:
                warn("Live fetch failed:", str(e))

        try:
            today_fixtures = await _fetch_today(client)
        except Exception as e:
            warn("Today fetch failed:", str(e))

        fixtures_checked = len(live_fixtures) + len(today_fixtures)

        live_tips: List[Dict] = []
        today_tips: List[Dict] = []
        live_stats: Dict[str, int] = {}
        today_stats: Dict[str, int] = {}

        if live_fixtures:
            live_tips, live_stats = await _build_tips_for_fixtures(client, live_fixtures)

        if today_fixtures:
            today_tips, today_stats = await _build_tips_for_fixtures(client, today_fixtures)

        # cross‑bucket dedupe
        combined = live_tips + today_tips
        combined, cross_removed = _dedupe_in_run(combined)
        today_stats["inrun_dedup_removed"] = today_stats.get("inrun_dedup_removed", 0) + cross_removed

        start_count = await count_sent_today()
        sent_total, send_stats = await _send_tips(client, combined, start_count)

        live_stats.update({f"send_{k}": v for k, v in send_stats.items()})
        today_stats.update({f"send_{k}": 0 for k in send_stats})  # keep keys symmetrical

    elapsed = (datetime.now(timezone.utc) - started).total_seconds()

    total_candidates = live_stats.get("raw_candidates", 0) + today_stats.get("raw_candidates", 0)
    total_low_conf   = (
        live_stats.get("low_conf", 0)
        + today_stats.get("low_conf", 0)
        + live_stats.get("send_low_conf", 0)
        + today_stats.get("send_low_conf", 0)
    )
    total_inrun_dup  = live_stats.get("inrun_dedup_removed", 0) + today_stats.get("inrun_dedup_removed", 0)
    total_ever_sent  = live_stats.get("send_ever_sent_skipped", 0) + today_stats.get("send_ever_sent_skipped", 0)
    total_cap_hits   = live_stats.get("send_daily_cap_hits", 0) + today_stats.get("send_daily_cap_hits", 0)

    log(
        "DEBUG scan: "
        f"fixtures(live={len(live_fixtures)},today={len(today_fixtures)})  "
        f"candidates={total_candidates}  low_conf={total_low_conf}  "
        f"inrun_dedup={total_inrun_dup}  ever_sent={total_ever_sent}  "
        f"sent={sent_total}  daily_cap_hits={total_cap_hits}"
    )

    return {
        "status": "ok",
        "fixturesChecked": fixtures_checked,
        "tipsSent": sent_total,
        "elapsedSeconds": round(elapsed),
        "debug": {
            "liveFixtures": len(live_fixtures),
            "todayFixtures": len(today_fixtures),
            "candidates": total_candidates,
            "lowConfidenceFiltered": total_low_conf,
            "inRunDuplicatesRemoved": total_inrun_dup,
            "everSentDuplicatesSkipped": total_ever_sent,
            "dailyCapHits": total_cap_hits,
            "builderFn": FOUND_BUILDER_NAME,
            "liveApiFn": FOUND_LIVE_API_NAME,
            "dateApiFn": FOUND_DATE_API_NAME,
            "config": {
                "MIN_CONFIDENCE_TO_SEND": MIN_CONFIDENCE_TO_SEND,
                "MAX_TIPS_PER_RUN": MAX_TIPS_PER_RUN,
                "DAILY_TIP_CAP": DAILY_TIP_CAP,
                "LIVE_MINUTE_MIN": LIVE_MINUTE_MIN,
                "LIVE_MINUTE_MAX": LIVE_MINUTE_MAX,
                "LIVE_MAX_FIXTURES": LIVE_MAX_FIXTURES,
                "MAX_TODAY_FIXTURES": MAX_TODAY_FIXTURES,
                "BUILD_FIXTURE_LIMIT": BUILD_FIXTURE_LIMIT,
            },
        },
    }
