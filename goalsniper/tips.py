from __future__ import annotations

import asyncio
import inspect
from datetime import datetime, timezone, date
from typing import Dict, List, Callable, Optional, Tuple

import httpx

from .logger import log, warn
from . import api_football as api
from . import tips as tip_engine
from .storage import (
    insert_tip_return_id,
    fixture_ever_sent,
    count_sent_today,
)
from .config import (
    MIN_CONFIDENCE_TO_SEND,
    MAX_TIPS_PER_RUN,
    LIVE_ENABLED,
    LIVE_MINUTE_MIN,
    LIVE_MINUTE_MAX,
    LIVE_MAX_FIXTURES,
    SCAN_DAYS,  # kept for compatibility/info
    DAILY_TIP_CAP,
    DUPLICATE_SUPPRESS_FOREVER,
    STATS_REQUEST_DELAY_MS,
)

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

# ---------- league filtering (reuse your emoji + keyword policy) ----------
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

# Pull the same allowlists used inside api_football.py
_ALLOW_COUNTRIES = _normalize_country_flags(getattr(api, "COUNTRY_FLAGS_ALLOW", ""))

_ALLOW_KEYS = [k.upper() for k in getattr(api, "LEAGUE_ALLOW_KEYWORDS", [])]
_EX_KEYS    = [k.upper() for k in getattr(api, "EXCLUDE_KEYWORDS", [])]

_COUNTRY_TO_ISO = {
    "GERMANY":"DE","ENGLAND":"GB","FRANCE":"FR","SPAIN":"ES","NETHERLANDS":"NL","HOLLAND":"NL",
    "EGYPT":"EG","BELGIUM":"BE","CHINA":"CN","AUSTRALIA":"AU","ITALY":"IT","CROATIA":"HR",
    "AUSTRIA":"AT","PORTUGAL":"PT","ROMANIA":"RO","SCOTLAND":"GB","SWEDEN":"SE","SWITZERLAND":"CH",
    "TURKEY":"TR","USA":"US","UNITED STATES":"US"
}

def _league_row_allowed(row: Dict) -> bool:
    """Filter get_current_leagues rows before we call fixtures-by-date."""
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
    data = await (fn(client) if asyncio.iscoroutinefunction(fn) else fn(client))
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
    """Fetch current leagues once; then fetch today fixtures per allowed league+season."""
    get_leagues = getattr(api, "get_current_leagues", None)
    by_date_fn  = _get_by_date_api()
    if not get_leagues or not by_date_fn:
        warn("No fixtures-by-date path available; skipping scheduled scan")
        return []

    leagues = await (get_leagues(client) if asyncio.iscoroutinefunction(get_leagues) else get_leagues(client))
    if not leagues:
        return []

    leagues = [row for row in leagues if _league_row_allowed(row)]
    if not leagues:
        return []

    out: List[Dict] = []
    day = date.today().isoformat()

    for row in leagues:
        try:
            lid = int(row.get("leagueId") or 0)
            season = int(row.get("season") or 0)
            if not lid or not season:
                continue
            if asyncio.iscoroutinefunction(by_date_fn):
                res = await by_date_fn(client, lid, season, day)
            else:
                res = by_date_fn(client, lid, season, day)
            for f in (res or []):
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

# ---------- tip generation & sending with debug stats ----------
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

    if asyncio.iscoroutinefunction(build_fn):
        return await build_fn(client, fixture, **kwargs)
    return build_fn(client, fixture, **kwargs)

async def _build_tips_for_fixtures(client: httpx.AsyncClient, fixtures: List[Dict]) -> Tuple[List[Dict], Dict[str, int]]:
    stats = {"raw_candidates": 0, "low_conf": 0, "inrun_dedup_removed": 0}
    build_fn = _get_build_tips_fn()
    if not build_fn:
        warn("No tip builder function found in tips module; skipping tip generation")
        return [], stats

    raw: List[Dict] = []
    for f in fixtures:
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

    stats["raw_candidates"] = len(raw)
    conf_filtered = [t for t in raw if float(t.get("confidence", 0.0)) >= MIN_CONFIDENCE_TO_SEND]
    stats["low_conf"] = len(raw) - len(conf_filtered)

    deduped, removed = _dedupe_in_run(conf_filtered)
    stats["inrun_dedup_removed"] = removed

    return deduped, stats

# ---------- sending ----------
async def _send_one(client: httpx.AsyncClient, tip: Dict) -> bool:
    # lazy import telegram so scanner import can’t fail if TG module has an issue
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

async def _send_tips(client: httpx.AsyncClient, tips: List[Dict]) -> Tuple[int, Dict[str, int]]:
    stats = {"ever_sent_skipped": 0, "daily_cap_hits": 0}
    sent = 0

    for t in tips:
        if (await count_sent_today()) >= DAILY_TIP_CAP:
            stats["daily_cap_hits"] += 1
            log(f"Daily cap reached ({DAILY_TIP_CAP}). Stopping.")
            break

        fid = _fixture_id(t)
        if DUPLICATE_SUPPRESS_FOREVER and fid and await fixture_ever_sent(fid):
            stats["ever_sent_skipped"] += 1
            continue

        try:
            if await _send_one(client, t):
                sent += 1
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

        live_stats: Dict[str, int] = {}
        today_stats: Dict[str, int] = {}
        total_sent = 0

        if live_fixtures:
            live_tips, live_stats = await _build_tips_for_fixtures(client, live_fixtures)
            sent_live, send_live_stats = await _send_tips(client, live_tips)
            total_sent += sent_live
            live_stats.update({f"send_{k}": v for k, v in send_live_stats.items()})

        if today_fixtures:
            today_tips, today_stats = await _build_tips_for_fixtures(client, today_fixtures)
            sent_today, send_today_stats = await _send_tips(client, today_tips)
            total_sent += sent_today
            today_stats.update({f"send_{k}": v for k, v in send_today_stats.items()})

    elapsed = (datetime.now(timezone.utc) - started).total_seconds()

    total_candidates = live_stats.get("raw_candidates", 0) + today_stats.get("raw_candidates", 0)
    total_low_conf   = live_stats.get("low_conf", 0) + today_stats.get("low_conf", 0)
    total_inrun_dup  = live_stats.get("inrun_dedup_removed", 0) + today_stats.get("inrun_dedup_removed", 0)
    total_ever_sent  = live_stats.get("send_ever_sent_skipped", 0) + today_stats.get("send_ever_sent_skipped", 0)
    total_cap_hits   = live_stats.get("send_daily_cap_hits", 0) + today_stats.get("send_daily_cap_hits", 0)

    log(
        "DEBUG scan: "
        f"fixtures(live={len(live_fixtures)},today={len(today_fixtures)})  "
        f"candidates={total_candidates}  low_conf={total_low_conf}  "
        f"inrun_dedup={total_inrun_dup}  ever_sent={total_ever_sent}  "
        f"sent={total_sent}  daily_cap_hits={total_cap_hits}"
    )

    return {
        "status": "ok",
        "fixturesChecked": fixtures_checked,
        "tipsSent": total_sent,
        "elapsedSeconds": int(elapsed),
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
            },
        },
    }
