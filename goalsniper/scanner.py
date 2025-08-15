from __future__ import annotations

import os
import asyncio
import inspect
from datetime import datetime, timezone, timedelta
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
    get_config_bulk,   # MOTD state
    set_config,        # MOTD state
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
LEAGUE_SCAN_CONCURRENCY = max(1, int(os.getenv("LEAGUE_SCAN_CONCURRENCY", "6")))
BUILD_CONCURRENCY = max(1, int(os.getenv("BUILD_CONCURRENCY", "6")))
MAX_LEAGUES_PER_RUN = max(1, int(os.getenv("MAX_LEAGUES_PER_RUN", "24")))

# Soft window (optional): allow a few tips slightly below the hard minimum
SOFT_MIN_CONF = float(os.getenv("SOFT_MIN_CONF", "0.0"))           # 0.00 disables
SOFT_TIPS_PER_RUN = max(0, int(os.getenv("SOFT_TIPS_PER_RUN", "0")))
SCANNER_DEBUG = (os.getenv("SCANNER_DEBUG", "0").strip().lower() in ("1", "true", "yes", "on"))

# ---------- MOTD options ----------
ENABLE_MOTD = (os.getenv("ENABLE_MOTD", "true").strip().lower() in ("1","true","yes","on"))
MOTD_MIN_CONF = float(os.getenv("MOTD_MIN_CONF", "0.70"))
MOTD_MARKET_ORDER = [s.strip().upper() for s in os.getenv(
    "MOTD_MARKET_ORDER", "OVER_UNDER_2.5,BTTS,1X2,1ST_HALF_OU"
).split(",") if s.strip()]

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

def _short_fixture_label(fx: Dict) -> str:
    lg = fx.get("league") or {}
    t  = fx.get("teams") or {}
    h  = (t.get("home") or {}).get("name") or "Home"
    a  = (t.get("away") or {}).get("name") or "Away"
    lid = int((lg.get("id") or 0))
    seas = int((lg.get("season") or 0))
    return f"{h} vs {a} | {lg.get('country') or ''} {lg.get('name') or ''} (L{lid}/S{seas})"

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
    if SCANNER_DEBUG:
        log(f"[scan] live fixtures after minute-window: {len(out)}")
    return out

async def _fetch_today(client: httpx.AsyncClient) -> List[Dict]:
    """
    Pulls fixtures across SCAN_DAYS (today .. today+SCAN_DAYS-1), respecting filters and caps.
    """
    if MAX_TODAY_FIXTURES <= 0:
        return []
    get_leagues = getattr(api, "get_current_leagues", None)
    by_date_fn  = _get_by_date_api()
    if not get_leagues or not by_date_fn:
        warn("No fixtures-by-date path available; skipping scheduled scan")
        return []

    eff = await cfg_filters.get_filters()
    allow_keys = eff.get("allowLeagueKeywords", [])
    ex_keys    = eff.get("excludeKeywords", [])
    allow_c    = eff.get("allowCountries", set())

    def allowed(row: Dict) -> bool:
        lname = (row.get("leagueName") or "").upper()
        if not lname:
            return False
        if ex_keys and any(bad in lname for bad in ex_keys):
            return False
        if allow_keys and not any(k in lname for k in allow_keys):
            return False
        country = (row.get("country") or "").upper()
        if allow_c and country and country not in allow_c:
            return False
        return True

    leagues_all = await _call_fn(get_leagues, client)
    leagues = [row for row in leagues_all if allowed(row)]

    total_allowed = len(leagues)
    if total_allowed > MAX_LEAGUES_PER_RUN:
        leagues = leagues[:MAX_LEAGUES_PER_RUN]
        log(f"[scan] allowed_leagues={total_allowed} capped_to={MAX_LEAGUES_PER_RUN}")
    else:
        log(f"[scan] allowed_leagues={total_allowed}")

    if not leagues:
        return []

    out: List[Dict] = []
    start_day = datetime.now(timezone.utc).date()
    days = [start_day + timedelta(days=offset) for offset in range(max(1, int(SCAN_DAYS)))]

    sem = asyncio.Semaphore(LEAGUE_SCAN_CONCURRENCY)

    async def _pull_league_day(row: Dict, day_iso: str):
        async with sem:
            try:
                lid = int(row.get("leagueId") or 0)
                season = int(row.get("season") or 0)
                if not lid or not season:
                    return []
                res = await _call_fn(by_date_fn, client, lid, season, day_iso)
                return [f for f in (res or []) if _status(f) in NS_STATES]
            except Exception as e:
                warn("fixtures_by_date failed:", str(e))
                return []
            finally:
                await asyncio.sleep(STATS_REQUEST_DELAY_MS / 1000.0)

    tasks = []
    for day in days:
        day_iso = day.isoformat()
        for row in leagues:
            tasks.append(asyncio.create_task(_pull_league_day(row, day_iso)))

    for fut in asyncio.as_completed(tasks):
        if len(out) >= MAX_TODAY_FIXTURES:
            break
        chunk = await fut
        for f in chunk:
            if len(out) >= MAX_TODAY_FIXTURES:
                break
            out.append(f)

    if SCANNER_DEBUG:
        log(f"[scan] scheduled fixtures (NS) across {len(days)} day(s): {len(out)}")
        # List a sample of fixtures being considered
        for f in out[:min(12, len(out))]:
            log("   ↳", _short_fixture_label(f))

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
    if not build_fn or BUILD_FIXTURE_LIMIT <= 0:
        return [], stats

    sem = asyncio.Semaphore(BUILD_CONCURRENCY)

    async def _build_one(f):
        async with sem:
            try:
                return await _invoke_builder(build_fn, client, f)
            except Exception as e:
                warn("Tip build failed for fixture", _fixture_id(f), str(e))
                return None
            finally:
                await asyncio.sleep(STATS_REQUEST_DELAY_MS / 1000.0)

    raw: List[Dict] = []
    tasks = [asyncio.create_task(_build_one(f)) for f in fixtures[:BUILD_FIXTURE_LIMIT]]
    for fut in asyncio.as_completed(tasks):
        res = await fut
        if res:
            raw.extend(res)

    deduped, removed = _dedupe_in_run(raw)
    stats["raw_candidates"] = len(deduped)
    stats["inrun_dedup_removed"] = removed

    if SCANNER_DEBUG:
        log(f"[scan] raw tips={len(raw)} deduped={len(deduped)} removed_dupes={removed}")
        # Show the first few raw (pre-calibration) tips
        for t in deduped[:min(10, len(deduped))]:
            log(f"   ↳ cand fixture={t.get('fixtureId')} {t.get('home')} vs {t.get('away')} | "
                f"{t.get('market')} {t.get('selection')} p_raw={t.get('probability')}")

    return deduped, stats

# ---------- Learning diagnostics ----------
async def _analyze_tip(t: dict) -> Tuple[float, float, float, float]:
    """
    Returns (p_cal, conf, dyn_thr, hard_thr) for diagnostics.
    """
    p_raw = float(t.get("probability", 0.5))
    market = str(t.get("market", ""))
    league_id = t.get("leagueId")
    p_cal = await calibrate_probability(market, p_raw, league_id)
    conf = abs(p_cal - 0.5) * 2.0
    dyn_thr = await dynamic_conf_threshold(market)
    hard_thr = max(dyn_thr, MIN_CONFIDENCE_TO_SEND)
    return float(p_cal), float(conf), float(dyn_thr), float(hard_thr)

# ---------- Learning gate ----------
async def _calibrate_and_filter(t: dict) -> Optional[dict]:
    """
    Returns:
      - dict(t) with updated probability/confidence (+ 'soft': False/True)  OR
      - None if rejected.
    """
    try:
        p_cal, conf, dyn_thr, hard_thr = await _analyze_tip(t)

        # Decide
        if conf >= hard_thr:
            t["probability"] = round(p_cal, 3)
            t["confidence"] = round(conf, 3)
            t["soft"] = False
            if SCANNER_DEBUG:
                log(f"[gate] OK  fixture={t.get('fixtureId')} {t.get('market')} {t.get('selection')} "
                    f"p_cal={p_cal:.3f} conf={conf:.3f} thr_dyn={dyn_thr:.3f} thr_hard={hard_thr:.3f}")
            return t

        # Soft window
        if SOFT_TIPS_PER_RUN > 0 and SOFT_MIN_CONF > 0.0 and conf >= SOFT_MIN_CONF:
            t["probability"] = round(p_cal, 3)
            t["confidence"] = round(conf, 3)
            t["soft"] = True
            if SCANNER_DEBUG:
                log(f"[gate] SOFT fixture={t.get('fixtureId')} {t.get('market')} {t.get('selection')} "
                    f"p_cal={p_cal:.3f} conf={conf:.3f} < hard={hard_thr:.3f} but >= soft={SOFT_MIN_CONF:.3f}")
            return t

        if SCANNER_DEBUG:
            log(f"[gate] REJ fixture={t.get('fixtureId')} {t.get('market')} {t.get('selection')} "
                f"p_cal={p_cal:.3f} conf={conf:.3f} < hard={hard_thr:.3f} "
                + (f"(soft={SOFT_MIN_CONF:.3f} not met)" if SOFT_MIN_CONF > 0 else ""))
        return None
    except Exception as e:
        warn("learning gate failed:", str(e))
        return None

# ---------- MOTD helpers ----------
def _market_priority(market: str) -> int:
    m = (market or "").upper()
    try:
        return MOTD_MARKET_ORDER.index(m)
    except ValueError:
        return len(MOTD_MARKET_ORDER) + 1

async def _choose_motd(client: httpx.AsyncClient, tips: List[Dict]) -> Optional[Dict]:
    candidates: List[Dict] = []
    for t in tips:
        if t.get("live"):
            continue
        tt = await _calibrate_and_filter(dict(t))  # copy, calibrate, add conf
        if not tt:
            continue
        if float(tt.get("confidence", 0.0)) < MOTD_MIN_CONF:
            continue
        candidates.append(tt)

    if not candidates:
        if SCANNER_DEBUG:
            log("[motd] no candidate met the MOTD_MIN_CONF threshold")
        return None

    candidates.sort(
        key=lambda x: (
            float(x.get("confidence", 0.0)),
            -_market_priority(x.get("market")),
            float(x.get("expectedGoals") or 0.0),
        ),
        reverse=True,
    )
    top = candidates[0]
    if SCANNER_DEBUG:
        log(f"[motd] candidate fixture={top.get('fixtureId')} {top.get('home')} vs {top.get('away')} | "
            f"{top.get('market')} {top.get('selection')} conf={top.get('confidence')}")
    return top

async def _maybe_send_motd(client: httpx.AsyncClient, today_tips: List[Dict]) -> Optional[int]:
    if not ENABLE_MOTD:
        return None

    today_utc = datetime.now(timezone.utc).date().isoformat()
    cfg = await get_config_bulk(["MOTD_LAST_DATE"])
    last = (cfg.get("MOTD_LAST_DATE") or "").strip()
    if last == today_utc:
        return None  # already sent today

    motd = await _choose_motd(client, today_tips or [])
    if not motd:
        return None

    from .telegram import send_motd
    msg_id = await send_motd(client, motd)
    if msg_id:
        await set_config("MOTD_LAST_DATE", today_utc)
        log(f"[motd] sent message_id={msg_id} fixture={motd.get('fixtureId')} {motd.get('market')} {motd.get('selection')}")
    return msg_id or None

# ---------- Sending ----------
async def _send_one(client: httpx.AsyncClient, tip: Dict) -> bool:
    from . import telegram as tg
    msg_id = await tg.send_tip_plain(client, tip)
    tip_id = await insert_tip_return_id({**tip, "messageId": msg_id}, msg_id)
    await tg.attach_feedback_buttons(client, msg_id, tip_id)
    log(f"Sent tip id={tip_id} fixture={tip['fixtureId']} {tip['market']} {tip['selection']} conf={tip['confidence']:.2f}"
        + (" [soft]" if tip.get("soft") else ""))
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
    soft_used = 0
    used_today = await count_sent_today() if start_count is None else int(start_count)

    for t in tips:
        if used_today >= DAILY_TIP_CAP:
            stats["daily_cap_hits"] += 1
            if SCANNER_DEBUG:
                log(f"[gate] stop: daily cap reached ({DAILY_TIP_CAP})")
            break
        fid = _fixture_id(t)
        if DUPLICATE_SUPPRESS_FOREVER and await fixture_ever_sent(fid):
            stats["ever_sent_skipped"] += 1
            if SCANNER_DEBUG:
                log(f"[gate] skip: fixture {fid} was sent before (ever)")
            continue
        if await _recently_sent(fid):
            stats["ever_sent_skipped"] += 1
            if SCANNER_DEBUG:
                log(f"[gate] skip: fixture {fid} was sent recently")
            continue

        tt = await _calibrate_and_filter(t)
        if not tt:
            stats["low_conf"] += 1
            continue

        if tt.get("soft"):
            if soft_used >= SOFT_TIPS_PER_RUN:
                if SCANNER_DEBUG:
                    log("[gate] soft budget exhausted; skipping a soft tip")
                stats["low_conf"] += 1
                continue
            soft_used += 1

        if await _send_one(client, tt):
            sent += 1
            used_today += 1
            if sent >= MAX_TIPS_PER_RUN:
                if SCANNER_DEBUG:
                    log(f"[gate] stop: run cap reached ({MAX_TIPS_PER_RUN})")
                break

    return sent, stats

# ---------- Entrypoint ----------
async def run_scan_and_send() -> Dict[str, int]:
    started = datetime.now(timezone.utc)

    async with httpx.AsyncClient(timeout=30) as client:
        live_fixtures: List[Dict] = await _fetch_live(client) if LIVE_ENABLED else []
        today_fixtures: List[Dict] = await _fetch_today(client)

        live_tips: List[Dict] = []
        today_tips: List[Dict] = []
        if live_fixtures:
            live_tips, _ = await _build_tips_for_fixtures(client, live_fixtures)
        if today_fixtures:
            today_tips, _ = await _build_tips_for_fixtures(client, today_fixtures)

        # Try sending MOTD highlight once per day (non-blocking on failure)
        if today_tips:
            try:
                await _maybe_send_motd(client, today_tips)
            except Exception as e:
                warn("MOTD send failed:", str(e))

        combined, _ = _dedupe_in_run(live_tips + today_tips)
        sent_total, send_stats = await _send_tips(client, combined, await count_sent_today())

    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    if SCANNER_DEBUG:
        log(f"DEBUG scan: fixtures(live={len(live_fixtures)},today={len(today_fixtures)})  "
            f"candidates={len(combined)}  sent={sent_total}  "
            f"ever_sent_skipped={send_stats.get('ever_sent_skipped',0)}  "
            f"low_conf={send_stats.get('low_conf',0)}  cap_hits={send_stats.get('daily_cap_hits',0)}")

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
