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
    has_fixture_tip_recent,
)
from .config import (
    MIN_CONFIDENCE_TO_SEND,
    MAX_TIPS_PER_RUN,
    LIVE_ENABLED,
    LIVE_MINUTE_MIN,
    LIVE_MINUTE_MAX,
    LIVE_MAX_FIXTURES,
    SCAN_DAYS,
    DAILY_TIP_CAP,
    DUPLICATE_SUPPRESS_FOREVER,
    STATS_REQUEST_DELAY_MS,
    DUPLICATE_SUPPRESS_MINUTES,
)
from .learning import calibrate_probability, dynamic_conf_threshold

# ---------- Limits from .env ----------
MAX_TODAY_FIXTURES = max(0, int(os.getenv("MAX_TODAY_FIXTURES", "24")))
BUILD_FIXTURE_LIMIT = max(0, int(os.getenv("BUILD_FIXTURE_LIMIT", "20")))

# ---------- Status sets ----------
FINISHED_STATES = {"FT", "AET", "PEN", "CANC", "PST", "ABD", "AWD", "WO"}
LIVE_STATES = {"1H", "2H", "HT", "ET"}
NS_STATES = {"NS"}

FOUND_BUILDER_NAME: Optional[str] = None
FOUND_LIVE_API_NAME: Optional[str] = None
FOUND_DATE_API_NAME: Optional[str] = None


# ---------- Resolve helpers ----------
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
    fn, nm = _resolve_fn_with_name(
        tip_engine,
        [
            "generate_tips_for_fixture", "build_tips_for_fixture",
            "predict_fixture", "create_tips_for_fixture"
        ],
        required=False,
    )
    FOUND_BUILDER_NAME = nm
    return fn


# ---------- Async call helpers ----------
async def _maybe_await(x):
    return await x if inspect.isawaitable(x) else x


async def _call_fn(fn, *args, **kwargs):
    return await _maybe_await(fn(*args, **kwargs))


# ---------- Fixture utils ----------
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


# ---------- League filters ----------
def _csv_upper(s: str) -> list[str]:
    return [x.strip().upper() for x in (s or "").split(",") if x.strip()]


_ALLOW_COUNTRIES = _csv_upper(getattr(api, "COUNTRY_ALLOW", ""))
_ALLOW_KEYS = _csv_upper(getattr(api, "LEAGUE_ALLOW_KEYWORDS", ""))
_EX_KEYS = _csv_upper(getattr(api, "EXCLUDE_KEYWORDS", ""))


def _league_row_allowed(row: Dict) -> bool:
    lname = (row.get("leagueName") or "").upper()
    country = (row.get("country") or "").upper()
    if not lname:
        return False
    for bad in _EX_KEYS:
        if bad in lname:
            return False
    if _ALLOW_KEYS and not any(k in lname for k in _ALLOW_KEYS):
        return False
    if _ALLOW_COUNTRIES and country not in _ALLOW_COUNTRIES:
        return False
    return True


# ---------- Fetchers ----------
async def _fetch_live(client: httpx.AsyncClient) -> List[Dict]:
    fn = _get_live_api()
    if not fn:
        warn("No live fixtures API function available; skipping live scan")
        return []
    data = await _call_fn(fn, client)
    out = []
    for f in data[:LIVE_MAX_FIXTURES]:
        st = _status(f)
        if st in FINISHED_STATES or st not in LIVE_STATES:
            continue
        minute = _elapsed_minute(f)
        if minute < LIVE_MINUTE_MIN or minute > LIVE_MINUTE_MAX:
            continue
        out.append(f)
    return out


async def _fetch_today(client: httpx.AsyncClient) -> List[Dict]:
    if MAX_TODAY_FIXTURES <= 0:
        return []
    get_leagues = getattr(api, "get_current_leagues", None)
    by_date_fn = _get_by_date_api()
    if not get_leagues or not by_date_fn:
        warn("No fixtures-by-date path available; skipping scheduled scan")
        return []

    leagues = await _call_fn(get_leagues, client)
    leagues = [row for row in leagues if _league_row_allowed(row)]
    log(f"[scan] allowed_leagues={len(leagues)}")

    if not leagues:
        return []

    out: List[Dict] = []
    day = datetime.now(timezone.utc).date().isoformat()
    for row in leagues:
        if len(out) >= MAX_TODAY_FIXTURES:
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
                if _status(f) in NS_STATES:
                    out.append(f)
        except Exception as e:
            warn("fixtures_by_date failed:", str(e))
        await asyncio.sleep(STATS_REQUEST_DELAY_MS / 1000.0)

    return out


# ---------- Tip generation ----------
def _dedupe_in_run(candidates: List[Dict]) -> Tuple[List[Dict], int]:
    seen = set()
    out = []
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
    return await _call_fn(build_fn, client, fixture, **kwargs)


async def _build_tips_for_fixtures(client: httpx.AsyncClient, fixtures: List[Dict]) -> Tuple[List[Dict], Dict[str, int]]:
    stats = {"raw_candidates": 0, "low_conf": 0, "inrun_dedup_removed": 0}
    build_fn = _get_build_tips_fn()
    if not build_fn or BUILD_FIXTURE_LIMIT <= 0:
        return [], stats

    raw: List[Dict] = []
    for f in fixtures[:BUILD_FIXTURE_LIMIT]:
        try:
            res = await _invoke_builder(build_fn, client, f)
            if res:
                raw.extend(res)
        except Exception as e:
            warn("Tip build failed for fixture", _fixture_id(f), str(e))
        await asyncio.sleep(STATS_REQUEST_DELAY_MS / 1000.0)

    deduped, removed = _dedupe_in_run(raw)
    stats["raw_candidates"] = len(deduped)
    stats["inrun_dedup_removed"] = removed
    return deduped, stats


# ---------- Learning gate ----------
async def _calibrate_and_filter(t: dict) -> Optional[dict]:
    try:
        p_raw = float(t.get("probability", 0.5))
        market = str(t.get("market", ""))
        league_id = t.get("leagueId")
        p_cal = await calibrate_probability(market, p_raw, league_id)
        conf = abs(p_cal - 0.5) * 2.0
        thr = await dynamic_conf_threshold(market)
        if conf < max(thr, MIN_CONFIDENCE_TO_SEND):
            return None
        t["probability"] = round(p_cal, 3)
        t["confidence"] = round(conf, 3)
        return t
    except Exception as e:
        warn("learning gate failed:", str(e))
        return None


# ---------- Sending ----------
async def _send_one(client: httpx.AsyncClient, tip: Dict) -> bool:
    from . import telegram as tg
    msg_id = await tg.send_tip_plain(client, tip)
    tip_id = await insert_tip_return_id({**tip, "messageId": msg_id}, msg_id)
    await tg.attach_feedback_buttons(client, msg_id, tip_id)
    log(f"Sent tip id={tip_id} fixture={tip['fixtureId']} {tip['market']} {tip['selection']} conf={tip['confidence']:.2f}")
    return True


async def _recently_sent(fid: int) -> bool:
    if DUPLICATE_SUPPRESS_FOREVER:
        return False
    if DUPLICATE_SUPPRESS_MINUTES <= 0:
        return False
    return await has_fixture_tip_recent(fid, DUPLICATE_SUPPRESS_MINUTES)


async def _send_tips(client: httpx.AsyncClient, tips: List[Dict], start_count: Optional[int] = None) -> Tuple[int, Dict[str, int]]:
    stats = {"ever_sent_skipped": 0, "daily_cap_hits": 0, "low_conf": 0}
    sent = 0
    used_today = await count_sent_today() if start_count is None else int(start_count)

    for t in tips:
        if used_today >= DAILY_TIP_CAP:
            stats["daily_cap_hits"] += 1
            break
        fid = _fixture_id(t)
        if DUPLICATE_SUPPRESS_FOREVER and await fixture_ever_sent(fid):
            stats["ever_sent_skipped"] += 1
            continue
        if await _recently_sent(fid):
            stats["ever_sent_skipped"] += 1
            continue
        tt = await _calibrate_and_filter(t)
        if not tt:
            stats["low_conf"] += 1
            continue
        if await _send_one(client, tt):
            sent += 1
            used_today += 1
            if sent >= MAX_TIPS_PER_RUN:
                break

    return sent, stats


# ---------- Entrypoint ----------
async def run_scan_and_send() -> Dict[str, int]:
    started = datetime.now(timezone.utc)
    async with httpx.AsyncClient(timeout=30) as client:
        live_fixtures = await _fetch_live(client) if LIVE_ENABLED else []
        today_fixtures = await _fetch_today(client)
        live_tips, live_stats = await _build_tips_for_fixtures(client, live_fixtures) if live_fixtures else ([], {})
        today_tips, today_stats = await _build_tips_for_fixtures(client, today_fixtures) if today_fixtures else ([], {})

        combined, _ = _dedupe_in_run(live_tips + today_tips)
        sent_total, send_stats = await _send_tips(client, combined, await count_sent_today())

    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    log(f"DEBUG scan: fixtures={len(live_fixtures)+len(today_fixtures)} candidates={len(combined)} sent={sent_total}")

    return {
        "status": "ok",
        "fixturesChecked": len(live_fixtures) + len(today_fixtures),
        "tipsSent": sent_total,
        "elapsedSeconds": round(elapsed),
        "debug": {
            "liveFixtures": len(live_fixtures),
            "todayFixtures": len(today_fixtures),
            "builderFn": FOUND_BUILDER_NAME,
            "liveApiFn": FOUND_LIVE_API_NAME,
            "dateApiFn": FOUND_DATE_API_NAME,
        },
    }
