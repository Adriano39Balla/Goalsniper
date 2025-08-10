import asyncio
from datetime import datetime, timezone, date
from typing import Dict, List, Callable, Optional

import httpx

from .logger import log, warn
from . import api_football as api
from . import tips as tip_engine
from .telegram import send_tip_message
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
)

# ---------- status sets ----------
FINISHED_STATES = {"FT", "AET", "PEN", "CANC", "PST", "ABD", "AWD", "WO"}
LIVE_STATES     = {"1H", "2H", "HT", "ET"}
NS_STATES       = {"NS"}

# ---------- flexible resolver helpers ----------

def _resolve_fn(mod, candidates: List[str], required: bool = True) -> Optional[Callable]:
    for name in candidates:
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn
    if required:
        # We don't raise here to avoid boot-time crash; we log and return None.
        warn(f"Missing functions. Tried: {candidates} in module {mod.__name__}")
    return None

# We resolve lazily inside functions so import-time never crashes,
# even if the API module/tips module doesn't have the expected names yet.
def _get_live_api() -> Optional[Callable]:
    return _resolve_fn(api, [
        "get_live_fixtures", "fetch_live_fixtures", "live_fixtures", "get_fixtures_live"
    ])

def _get_by_date_api() -> Optional[Callable]:
    return _resolve_fn(api, [
        "get_fixtures_by_date", "fixtures_by_date", "get_fixtures_date", "fetch_fixtures_by_date"
    ])

def _get_build_tips_fn() -> Optional[Callable]:
    return _resolve_fn(tip_engine, [
        "build_tips_for_fixture", "build_for_fixture", "generate_tips_for_fixture", "generate_tips"
    ])

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
    data = await fn(client)  # should return list of fixtures
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
    today = date.today().isoformat()
    fixtures = await fn(client, today)  # list
    out = []
    for f in fixtures:
        st = _status(f)
        if st in FINISHED_STATES:
            continue
        if st not in NS_STATES:
            continue
        out.append(f)
    return out

# ---------- tip generation & sending ----------

def _dedupe_in_run(candidates: List[Dict]) -> List[Dict]:
    seen = set()
    out = []
    for t in candidates:
        fid = _fixture_id(t)
        if fid and fid not in seen:
            seen.add(fid)
            out.append(t)
    return out

async def _build_tips_for_fixtures(client: httpx.AsyncClient, fixtures: List[Dict]) -> List[Dict]:
    build_fn = _get_build_tips_fn()
    if not build_fn:
        warn("No tip builder function found in tips module; skipping tip generation")
        return []

    candidates: List[Dict] = []
    for f in fixtures:
        try:
            lst = await build_fn(client, f) if asyncio.iscoroutinefunction(build_fn) else build_fn(client, f)
            if lst:
                # Normalize minimal fields if builder omitted them
                for t in lst:
                    t.setdefault("fixtureId", _fixture_id(f))
                    t.setdefault("leagueId", _league_id(f))
                    t.setdefault("season", _season(f))
                candidates.extend(lst)
        except Exception as e:
            warn("Tip build failed for fixture", _fixture_id(f), str(e))
        await asyncio.sleep(max(0, STATS_REQUEST_DELAY_MS) / 1000.0)

    # Confidence filter and in-run dedupe
    filtered = [t for t in candidates if float(t.get("confidence", 0.0)) >= MIN_CONFIDENCE_TO_SEND]
    return _dedupe_in_run(filtered)

async def _send_one(client: httpx.AsyncClient, tip: Dict) -> bool:
    msg_id = await send_tip_message(client, tip)
    tip_id = await insert_tip_return_id({**tip, "messageId": msg_id}, msg_id)
    log(f"Sent tip id={tip_id} fixture={tip['fixtureId']} {tip['market']} {tip['selection']} conf={tip['confidence']:.2f}")
    return True

async def _send_tips(client: httpx.AsyncClient, tips: List[Dict]) -> int:
    sent = 0
    for t in tips:
        # daily cap
        if (await count_sent_today()) >= DAILY_TIP_CAP:
            log(f"Daily cap reached ({DAILY_TIP_CAP}). Stopping.")
            break

        # suppress duplicates forever
        fid = _fixture_id(t)
        if DUPLICATE_SUPPRESS_FOREVER and fid and await fixture_ever_sent(fid):
            continue

        try:
            if await _send_one(client, t):
                sent += 1
                if sent >= MAX_TIPS_PER_RUN:
                    break
        except Exception as e:
            warn("Send failed:", str(e))
    return sent

# ---------- public entrypoint ----------

async def run_scan_and_send() -> Dict[str, int]:
    started = datetime.now(timezone.utc)
    fixtures_checked = 0
    total_sent = 0

    async with httpx.AsyncClient(timeout=30) as client:
        if LIVE_ENABLED:
            try:
                live = await _fetch_live(client)
                fixtures_checked += len(live)
                live_tips = await _build_tips_for_fixtures(client, live)
                total_sent += await _send_tips(client, live_tips)
            except Exception as e:
                warn("Live scan failed:", str(e))

        try:
            todays = await _fetch_today(client)
            fixtures_checked += len(todays)
            today_tips = await _build_tips_for_fixtures(client, todays)
            total_sent += await _send_tips(client, today_tips)
        except Exception as e:
            warn("Today scan failed:", str(e))

    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    log(f"Scan complete: fixtures={fixtures_checked}, tips sent={total_sent}, elapsed={elapsed:.1f}s")
    return {"fixturesChecked": fixtures_checked, "tipsSent": total_sent, "elapsedSeconds": int(elapsed)}
