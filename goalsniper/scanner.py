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
from . import filters as cfg_filters
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

# ---------- Debug toggle ----------
SCANNER_DEBUG = (os.getenv("SCANNER_DEBUG", "0").strip() in ("1", "true", "yes", "on"))

def _dlog(*args):
    if SCANNER_DEBUG:
        log("[scan-debug]", *args)

# ---------- Limits from .env ----------
MAX_TODAY_FIXTURES = max(0, int(os.getenv("MAX_TODAY_FIXTURES", "24")))
BUILD_FIXTURE_LIMIT = max(0, int(os.getenv("BUILD_FIXTURE_LIMIT", "20")))
LEAGUE_SCAN_CONCURRENCY = max(1, int(os.getenv("LEAGUE_SCAN_CONCURRENCY", "6")))
BUILD_CONCURRENCY = max(1, int(os.getenv("BUILD_CONCURRENCY", "6")))
MAX_LEAGUES_PER_RUN = max(1, int(os.getenv("MAX_LEAGUES_PER_RUN", "24")))

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
            "predict_fixture", "create_tips_for_fixture",
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

def _team_name(side: Dict) -> str:
    return (side or {}).get("name") or "?"

def _fx_label(fx: Dict) -> str:
    lg = fx.get("league") or {}
    t  = fx.get("teams") or {}
    f  = fx.get("fixture") or {}
    return (
        f"{_team_name(t.get('home'))} vs {_team_name(t.get('away'))} | "
        f"{lg.get('country','')} - {lg.get('name','')} | "
        f"id={f.get('id')} st={(_status(fx) or '-')}/{_elapsed_minute(fx)}'"
    )

# ---------- League filters (DB/env unified) ----------
async def _league_row_allowed(row: Dict) -> bool:
    filters = await cfg_filters.get_filters()
    lname = (row.get("leagueName") or "").upper()
    country = (row.get("country") or "").upper()
    if not lname:
        return False
    if filters["excludeKeywords"] and any(bad in lname for bad in filters["excludeKeywords"]):
        return False
    if filters["allowLeagueKeywords"] and not any(k in lname for k in filters["allowLeagueKeywords"]):
        return False
    if filters["allowCountries"] and country and (country not in filters["allowCountries"]):
        return False
    return True

# ---------- Fetchers ----------
async def _fetch_live(client: httpx.AsyncClient) -> List[Dict]:
    fn = _get_live_api()
    if not fn:
        warn("No live fixtures API function available; skipping live scan")
        return []
    data = await _call_fn(fn, client)  # already passed API-level filters
    out: List[Dict] = []
    for f in data[:LIVE_MAX_FIXTURES]:
        st = _status(f)
        minute = _elapsed_minute(f)

        if st in FINISHED_STATES:
            _dlog("skip live (finished):", _fx_label(f))
            continue
        if st not in LIVE_STATES:
            _dlog("skip live (not in LIVE_STATES):", _fx_label(f))
            continue
        if minute < LIVE_MINUTE_MIN or minute > LIVE_MINUTE_MAX:
            _dlog(f"skip live (minute {minute} not in [{LIVE_MINUTE_MIN},{LIVE_MINUTE_MAX}]):", _fx_label(f))
            continue

        _dlog("keep live:", _fx_label(f))
        out.append(f)
    log(f"[scan] live fixtures after minute-window: {len(out)}")
    return out

async def _fetch_today(client: httpx.AsyncClient) -> List[Dict]:
    if MAX_TODAY_FIXTURES <= 0:
        return []
    get_leagues = getattr(api, "get_current_leagues", None)
    by_date_fn  = _get_by_date_api()
    if not get_leagues or not by_date_fn:
        warn("No fixtures-by-date path available; skipping scheduled scan")
        return []

    leagues_all = await _call_fn(get_leagues, client)
    leagues = []
    for r in leagues_all:
        ok = await _league_row_allowed(r)
        if ok:
            leagues.append(r)
            _dlog("allow league:", r.get("country"), "-", r.get("leagueName"))
        else:
            _dlog("block league:", r.get("country"), "-", r.get("leagueName"))

    total_allowed = len(leagues)
    if total_allowed > MAX_LEAGUES_PER_RUN:
        leagues = leagues[:MAX_LEAGUES_PER_RUN]
        log(f"[scan] allowed_leagues={total_allowed} capped_to={MAX_LEAGUES_PER_RUN}")
    else:
        log(f"[scan] allowed_leagues={total_allowed}")

    if not leagues:
        return []

    out: List[Dict] = []
    day = datetime.now(timezone.utc).date().isoformat()
    sem = asyncio.Semaphore(LEAGUE_SCAN_CONCURRENCY)

    async def _pull_league(row: Dict):
        async with sem:
            lid = int(row.get("leagueId") or 0)
            season = int(row.get("season") or 0)
            if not lid or not season:
                return []
            try:
                res = await _call_fn(by_date_fn, client, lid, season, day)
                adds = [f for f in (res or []) if _status(f) in NS_STATES]
                _dlog(f"date {row.get('country')} - {row.get('leagueName')}: "
                      f"NS={len(adds)} of {len(res or [])}")
                return adds
            except Exception as e:
                warn("fixtures_by_date failed:", f"league={lid} season={season}", str(e))
                return []
            finally:
                await asyncio.sleep(STATS_REQUEST_DELAY_MS / 1000.0)

    tasks = [asyncio.create_task(_pull_league(row)) for row in leagues]
    for fut in asyncio.as_completed(tasks):
        if len(out) >= MAX_TODAY_FIXTURES:
            break
        chunk = await fut
        for f in (chunk or []):
            if len(out) >= MAX_TODAY_FIXTURES:
                break
            out.append(f)

    log(f"[scan] scheduled fixtures (NS): {len(out)}")
    return out

# ---------- Tip generation ----------
def _dedupe_in_run(candidates: List[Dict]) -> Tuple[List[Dict], int]:
    seen = set()
    out: List[Dict] = []
    removed = 0
    for t in candidates:
        fid = _fixture_id(t)
        if fid in seen:
            removed += 1
            _dlog("dedupe-inrun: drop duplicate fixture", fid)
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

    sem = asyncio.Semaphore(BUILD_CONCURRENCY)
    raw: List[Dict] = []

    async def _build_one(fx):
        async with sem:
            fid = _fixture_id(fx)
            try:
                _dlog("build: start", _fx_label(fx))
                res = await _invoke_builder(build_fn, client, fx)
                n = len(res or [])
                _dlog(f"build: fixture {fid} produced {n} raw tips")
                # normalize IDs on each tip
                if res:
                    for t in res:
                        t.setdefault("fixtureId", fid)
                        t.setdefault("leagueId", _league_id(fx))
                        t.setdefault("season", _season(fx))
                return res
            except Exception as e:
                warn("Tip build failed for fixture", fid, str(e))
                return None
            finally:
                await asyncio.sleep(STATS_REQUEST_DELAY_MS / 1000.0)

    tasks = [asyncio.create_task(_build_one(f)) for f in fixtures[:BUILD_FIXTURE_LIMIT]]
    for fut in asyncio.as_completed(tasks):
        res = await fut
        if res:
            raw.extend(res)

    deduped, removed = _dedupe_in_run(raw)
    stats["raw_candidates"] = len(deduped)
    stats["inrun_dedup_removed"] = removed
    log(f"[scan] raw tips={len(raw)} deduped={len(deduped)} removed={removed}")
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
        min_thr = max(thr, MIN_CONFIDENCE_TO_SEND)
        if conf < min_thr:
            _dlog(
                f"gate: drop low_conf market={market} "
                f"fixture={t.get('fixtureId')} conf={conf:.3f} thr={min_thr:.3f}"
            )
            return None
        out = dict(t)
        out["probability"] = round(p_cal, 3)
        out["confidence"] = round(conf, 3)
        return out
    except Exception as e:
        warn("learning gate failed:", str(e))
        return None

# ---------- Sending ----------
async def _send_one(client: httpx.AsyncClient, tip: Dict) -> bool:
    from . import telegram as tg

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
        fid = _fixture_id(t)

        if used_today >= DAILY_TIP_CAP:
            stats["daily_cap_hits"] += 1
            log(f"Daily cap reached ({DAILY_TIP_CAP}). Stopping.")
            break

        if DUPLICATE_SUPPRESS_FOREVER and await fixture_ever_sent(fid):
            stats["ever_sent_skipped"] += 1
            _dlog("send: skip (ever_sent) fixture", fid)
            continue

        if await _recently_sent(fid):
            stats["ever_sent_skipped"] += 1
            _dlog("send: skip (recently_sent) fixture", fid)
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

# ---------- Entrypoint ----------
async def run_scan_and_send() -> Dict[str, int]:
    started = datetime.now(timezone.utc)

    async with httpx.AsyncClient(timeout=30) as client:
        # LIVE
        live_fixtures: List[Dict] = []
        if LIVE_ENABLED:
            try:
                live_fixtures = await _fetch_live(client)
            except Exception as e:
                warn("Live fetch failed:", str(e))

        # TODAY (NS)
        today_fixtures: List[Dict] = []
        try:
            today_fixtures = await _fetch_today(client)
        except Exception as e:
            warn("Today fetch failed:", str(e))

        fixtures_checked = len(live_fixtures) + len(today_fixtures)
        _dlog(f"fixtures_checked live={len(live_fixtures)} today={len(today_fixtures)}")

        # BUILD
        live_tips: List[Dict] = []
        today_tips: List[Dict] = []
        if live_fixtures:
            live_tips, _ = await _build_tips_for_fixtures(client, live_fixtures)
        if today_fixtures:
            today_tips, _ = await _build_tips_for_fixtures(client, today_fixtures)

        combined, cross_removed = _dedupe_in_run(live_tips + today_tips)
        if cross_removed:
            _dlog("cross-bucket dedupe removed:", cross_removed)

        start_count = await count_sent_today()
        sent_total, send_stats = await _send_tips(client, combined, start_count)

    elapsed = (datetime.now(timezone.utc) - started).total_seconds()

    log(
        "DEBUG scan: "
        f"fixtures(live={len(live_fixtures)},today={len(today_fixtures)})  "
        f"candidates={len(combined)}  sent={sent_total}  "
        f"ever_sent_skipped={send_stats.get('ever_sent_skipped',0)} "
        f"low_conf={send_stats.get('low_conf',0)} "
        f"cap_hits={send_stats.get('daily_cap_hits',0)}"
    )

    return {
        "status": "ok",
        "fixturesChecked": fixtures_checked,
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
