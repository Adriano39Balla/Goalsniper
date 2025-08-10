import asyncio
from datetime import datetime, timezone, date
from typing import Dict, List, Callable, Optional, Tuple

import httpx

from .logger import log, warn
from . import api_football as api
from . import tips as tip_engine
from . import telegram as tg  # resolve sender dynamically
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


# ---------- flexible resolver helpers ----------

FOUND_BUILDER_NAME = None
FOUND_LIVE_API_NAME = None
FOUND_DATE_API_NAME = None

def _resolve_fn_with_name(mod, candidates, required=True):
    global FOUND_BUILDER_NAME, FOUND_LIVE_API_NAME, FOUND_DATE_API_NAME
    for name in candidates:
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn, name
    if required:
        warn(f"Missing functions. Tried: {candidates} in module {mod.__name__}")
    return None, None

def _get_live_api():
    global FOUND_LIVE_API_NAME
    fn, nm = _resolve_fn_with_name(
        api,
        ["get_live_fixtures","fetch_live_fixtures","live_fixtures","get_fixtures_live"],
        required=False
    )
    FOUND_LIVE_API_NAME = nm
    return fn

def _get_by_date_api():
    global FOUND_DATE_API_NAME
    fn, nm = _resolve_fn_with_name(
        api,
        ["get_fixtures_by_date","fixtures_by_date","get_fixtures_date","fetch_fixtures_by_date"],
        required=False
    )
    FOUND_DATE_API_NAME = nm
    return fn

def _get_build_tips_fn():
    global FOUND_BUILDER_NAME
    # Try a wider set of common names
    candidates = [
        "build_tips_for_fixture", "build_for_fixture",
        "generate_tips_for_fixture", "generate_tips",
        "build_tips", "make_tips", "predict_fixture", "predict_tips",
        "tips_for_fixture", "create_tips_for_fixture"
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


# ---------- fetchers (1–2 calls per run) ----------

async def _fetch_live(client: httpx.AsyncClient) -> List[Dict]:
    fn = _get_live_api()
    if not fn:
        warn("No live fixtures API function available; skipping live scan")
        return []
    data = await fn(client)  # list of fixtures
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
    fn = _get_by_date_api()
    if not fn:
        warn("No fixtures-by-date API function available; skipping scheduled scan")
        return []
    day = date.today().isoformat()
    fixtures = await fn(client, day)
    out = []
    for f in fixtures:
        st = _status(f)
        if st in FINISHED_STATES:
            continue
        if st not in NS_STATES:
            continue
        out.append(f)
    return out


# ---------- tip generation & sending with debug stats ----------

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

async def _build_tips_for_fixtures(client: httpx.AsyncClient, fixtures: List[Dict]) -> Tuple[List[Dict], Dict[str, int]]:
    """
    Returns (tips_after_filters, stats)
    stats keys:
      raw_candidates, low_conf, inrun_dedup_removed
    """
    stats = {"raw_candidates": 0, "low_conf": 0, "inrun_dedup_removed": 0}
    build_fn = _get_build_tips_fn()
    if not build_fn:
        warn("No tip builder function found in tips module; skipping tip generation")
        return [], stats

    raw: List[Dict] = []
    for f in fixtures:
        try:
            res = await build_fn(client, f) if asyncio.iscoroutinefunction(build_fn) else build_fn(client, f)
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
    # confidence filter
    conf_filtered = [t for t in raw if float(t.get("confidence", 0.0)) >= MIN_CONFIDENCE_TO_SEND]
    stats["low_conf"] = len(raw) - len(conf_filtered)

    # in-run dedupe per fixture
    deduped, removed = _dedupe_in_run(conf_filtered)
    stats["inrun_dedup_removed"] = removed

    return deduped, stats

async def _send_one(client: httpx.AsyncClient, tip: Dict) -> bool:
    send_fn = _get_send_fn()
    if not send_fn:
        raise RuntimeError(
            "No send function found in goalsniper.telegram "
            "(expected one of: send_tip_message, send_tip, send_message_with_feedback, send_message, push_tip, send)"
        )
    msg_id = await send_fn(client, tip) if asyncio.iscoroutinefunction(send_fn) else send_fn(client, tip)
    tip_id = await insert_tip_return_id({**tip, "messageId": msg_id}, msg_id)
    log(f"Sent tip id={tip_id} fixture={tip['fixtureId']} {tip['market']} {tip['selection']} conf={tip['confidence']:.2f}")
    return True

async def _send_tips(client: httpx.AsyncClient, tips: List[Dict]) -> Tuple[int, Dict[str, int]]:
    """
    Returns (sent_count, stats)
    stats keys:
      ever_sent_skipped, daily_cap_hits
    """
    stats = {"ever_sent_skipped": 0, "daily_cap_hits": 0}
    sent = 0

    for t in tips:
        # daily cap
        if (await count_sent_today()) >= DAILY_TIP_CAP:
            stats["daily_cap_hits"] += 1
            log(f"Daily cap reached ({DAILY_TIP_CAP}). Stopping.")
            break

        # forever dedupe
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
    """
    Single run:
      - fetch live fixtures (1 call) and today's fixtures (1 call)
      - generate tips (in-memory)
      - enforce per-run cap, daily cap, dedupe per fixture, skip finished
      - return detailed debug stats in JSON
    """
    started = datetime.now(timezone.utc)

    live_fixtures: List[Dict] = []
    today_fixtures: List[Dict] = []

    async with httpx.AsyncClient(timeout=30) as client:
        # Fetch fixtures
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

        # Build tips + send (LIVE)
        live_stats, today_stats = {}, {}
        total_sent = 0

        if live_fixtures:
            live_tips, live_stats = await _build_tips_for_fixtures(client, live_fixtures)
            sent_live, send_live_stats = await _send_tips(client, live_tips)
            total_sent += sent_live
            live_stats.update({f"send_{k}": v for k, v in send_live_stats.items()})

        # Build tips + send (TODAY)
        if today_fixtures:
            today_tips, today_stats = await _build_tips_for_fixtures(client, today_fixtures)
            sent_today, send_today_stats = await _send_tips(client, today_tips)
            total_sent += sent_today
            today_stats.update({f"send_{k}": v for k, v in send_today_stats.items()})

    elapsed = (datetime.now(timezone.utc) - started).total_seconds()

    # aggregate debug totals
    total_candidates = live_stats.get("raw_candidates", 0) + today_stats.get("raw_candidates", 0)
    total_low_conf   = live_stats.get("low_conf", 0) + today_stats.get("low_conf", 0)
    total_inrun_dup  = live_stats.get("inrun_dedup_removed", 0) + today_stats.get("inrun_dedup_removed", 0)
    total_ever_sent  = live_stats.get("send_ever_sent_skipped", 0) + today_stats.get("send_ever_sent_skipped", 0)
    total_cap_hits   = live_stats.get("send_daily_cap_hits", 0) + today_stats.get("send_daily_cap_hits", 0)

    # keep the compact log, but the main info will be returned below
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
            "config": {
                "MIN_CONFIDENCE_TO_SEND": MIN_CONFIDENCE_TO_SEND,
                "MAX_TIPS_PER_RUN": MAX_TIPS_PER_RUN,
                "DAILY_TIP_CAP": DAILY_TIP_CAP,
                "LIVE_MINUTE_MIN": LIVE_MINUTE_MIN,
                "LIVE_MINUTE_MAX": LIVE_MINUTE_MAX,
                "LIVE_MAX_FIXTURES": LIVE_MAX_FIXTURES,
            }
        }
    }
