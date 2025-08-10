import asyncio
from datetime import datetime, timezone, date
from typing import Dict, List

import httpx

from .logger import log, warn
from . import tips as tip_engine
from . import api_football as api
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

# Hard guard: never process these fixture states
FINISHED_STATES = {"FT", "AET", "PEN", "CANC", "PST", "ABD", "AWD", "WO"}
LIVE_STATES     = {"1H", "2H", "HT", "ET"}   # adjust if API uses others
NS_STATES       = {"NS"}                     # not started

# --------- fixture fetchers (1–2 calls per run) ---------

async def _fetch_live(client: httpx.AsyncClient) -> List[Dict]:
    # Single call to get all live fixtures
    data = await api.get_live_fixtures(client)  # must return list of fixtures (API wrapper you already have)
    out = []
    for f in data[: LIVE_MAX_FIXTURES]:
        st = f["fixture"]["status"]["short"]
        if st in FINISHED_STATES:
            continue
        if st not in LIVE_STATES:
            continue
        minute = (f["fixture"].get("status", {}) or {}).get("elapsed") or 0
        if minute < LIVE_MINUTE_MIN or minute > LIVE_MINUTE_MAX:
            continue
        out.append(f)
    return out

async def _fetch_today(client: httpx.AsyncClient) -> List[Dict]:
    # Single call for today
    today = date.today().isoformat()
    fixtures = await api.get_fixtures_by_date(client, today)  # must return list
    out = []
    for f in fixtures:
        st = f["fixture"]["status"]["short"]
        if st in FINISHED_STATES:
            continue
        if st not in NS_STATES:
            continue
        out.append(f)
    return out

# --------- tip generation + sending ---------

def _dedupe_in_run(candidates: List[Dict]) -> List[Dict]:
    seen = set()
    deduped = []
    for t in candidates:
        fid = int(t["fixtureId"])
        if fid in seen:
            continue
        seen.add(fid)
        deduped.append(t)
    return deduped

async def _build_tips_for_fixtures(client: httpx.AsyncClient, fixtures: List[Dict]) -> List[Dict]:
    # tip_engine should provide a function that builds candidate tips for a fixture
    # We keep everything local, then filter by confidence and dedupe by fixture.
    candidates: List[Dict] = []
    for f in fixtures:
        try:
            # build tips for this single fixture; the function should not hit API again heavily
            lst = await tip_engine.build_tips_for_fixture(client, f)
            if lst:
                candidates.extend(lst)
        except Exception as e:
            warn("tip build failed for fixture", f.get("fixture", {}).get("id"), str(e))
        # tiny delay to be gentle to API
        await asyncio.sleep(max(0, STATS_REQUEST_DELAY_MS) / 1000.0)
    # filter by confidence
    filtered = [t for t in candidates if float(t.get("confidence", 0.0)) >= MIN_CONFIDENCE_TO_SEND]
    # one tip per fixture (in-run)
    return _dedupe_in_run(filtered)

async def _send_calibrated(client: httpx.AsyncClient, tip: Dict) -> bool:
    # Telegram push + DB record
    msg_id = await send_tip_message(client, tip)  # should return Telegram message_id
    tip_id = await insert_tip_return_id({**tip, "messageId": msg_id}, msg_id)
    log(f"Sent tip id={tip_id} fixture={tip['fixtureId']} {tip['market']} {tip['selection']} conf={tip['confidence']:.2f}")
    return True

async def _send_tips(client: httpx.AsyncClient, tips: List[Dict]) -> int:
    sent = 0
    for t in tips:
        # daily cap
        today_count = await count_sent_today()
        if today_count >= DAILY_TIP_CAP:
            log(f"Daily cap reached ({DAILY_TIP_CAP}). Stopping.")
            break

        # forever dedupe: if any tip for this fixture was ever sent, skip
        fid = int(t["fixtureId"])
        if DUPLICATE_SUPPRESS_FOREVER and await fixture_ever_sent(fid):
            continue

        try:
            ok = await _send_calibrated(client, t)
            if ok:
                sent += 1
                if sent >= MAX_TIPS_PER_RUN:
                    break
        except Exception as e:
            warn("Send/calibrate failed:", str(e))
    return sent

# --------- public entrypoint ---------

async def run_scan_and_send() -> Dict[str, int]:
    """
    Single run:
      - fetch live fixtures (1 call) and today fixtures (1 call)
      - generate tips (in-memory)
      - enforce per-run cap, daily cap, dedupe per fixture, skip finished
    """
    started = datetime.now(timezone.utc)
    fixtures_checked = 0
    total_sent = 0

    async with httpx.AsyncClient(timeout=30) as client:
        # LIVE
        if LIVE_ENABLED:
            try:
                live = await _fetch_live(client)
                fixtures_checked += len(live)
                live_tips = await _build_tips_for_fixtures(client, live)
                if live_tips:
                    total_sent += await _send_tips(client, live_tips)
            except Exception as e:
                warn("live scan failed:", str(e))

        # TODAY (scheduled)
        try:
            todays = await _fetch_today(client)
            fixtures_checked += len(todays)
            today_tips = await _build_tips_for_fixtures(client, todays)
            if today_tips:
                total_sent += await _send_tips(client, today_tips)
        except Exception as e:
            warn("today scan failed:", str(e))

    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    log(f"Scan complete: fixtures={fixtures_checked}, tips sent={total_sent}, elapsed={elapsed:.1f}s")
    return {"fixturesChecked": fixtures_checked, "tipsSent": total_sent, "elapsedSeconds": int(elapsed)}
