import asyncio
from datetime import datetime, timedelta, timezone
from typing import List, Dict
import httpx

from .api_football import get_current_leagues, get_fixtures_by_date, get_live_fixtures
from .tips import generate_tips_for_fixture
from .telegram import send_telegram_message, format_tip_message, edit_message_markup
from .config import (
    SCAN_DAYS, MAX_TIPS_PER_RUN, LEAGUE_EXCLUDE_TYPES,
    LIVE_ENABLED, LIVE_MINUTE_MIN, LIVE_MINUTE_MAX, LIVE_MAX_FIXTURES
)
from .learning import calibrate_probability, dynamic_conf_threshold, register_sent_tip
from .logger import log, warn

def _is_league_excluded(typ: str) -> bool:
    if not typ:
        return False
    return any(t.lower() == str(typ).lower() for t in LEAGUE_EXCLUDE_TYPES)

def _buttons_payload(internal_tip_id: int) -> dict:
    return {
        "inline_keyboard": [[
            {"text": "👍 Correct", "callback_data": f"fb:{internal_tip_id}:1"},
            {"text": "👎 Wrong",   "callback_data": f"fb:{internal_tip_id}:0"},
        ]]
    }

async def _send_calibrated(client: httpx.AsyncClient, raw_tip: Dict) -> bool:
    p_adj = await calibrate_probability(raw_tip["market"], float(raw_tip["probability"]), league_id=raw_tip.get("leagueId"))
    conf_adj = abs(p_adj - 0.5) * 2.0
    thr = await dynamic_conf_threshold(raw_tip["market"])

    tip = dict(raw_tip)
    tip["probability"] = round(p_adj, 3)
    tip["confidence"] = round(conf_adj, 3)

    if conf_adj < thr:
        return False

    msg_text = format_tip_message(tip)
    message_id = await send_telegram_message(client, msg_text, reply_markup=_buttons_payload(0))
    internal_id = await register_sent_tip(tip, message_id)
    await edit_message_markup(client, message_id, _buttons_payload(internal_id))
    return True

async def _send_tips(client: httpx.AsyncClient, tips: List[Dict]) -> int:
    sent = 0
    for t in tips:
        try:
            ok = await _send_calibrated(client, t)
            if ok:
                sent += 1
        except Exception as e:
            warn("Send/calibrate failed:", str(e))
    return sent

async def run_scan_and_send() -> Dict[str, int]:
    started = datetime.now(timezone.utc)
    tips_sent = 0
    fixtures_checked = 0

    async with httpx.AsyncClient() as client:
        # LIVE
        if LIVE_ENABLED:
            live = await get_live_fixtures(client)
            eligible = []
            for f in live:
                minute = ((f.get("fixture", {}) or {}).get("status", {}) or {}).get("elapsed") or 0
                if LIVE_MINUTE_MIN <= minute <= LIVE_MINUTE_MAX:
                    eligible.append(f)
            if len(eligible) > LIVE_MAX_FIXTURES:
                eligible = eligible[:LIVE_MAX_FIXTURES]
            log(f"Live fixtures eligible: {len(eligible)}")
            for f in eligible:
                league_id = ((f.get("league") or {}).get("id"))
                season = ((f.get("league") or {}).get("season"))
                if not league_id or not season:
                    continue
                try:
                    tips = await generate_tips_for_fixture(client, f, league_id, season)
                    fixtures_checked += 1
                    if tips:
                        tips_sent += await _send_tips(client, tips)
                        if tips_sent >= MAX_TIPS_PER_RUN:
                            break
                except Exception as e:
                    warn("Live tip error:", str(e))

        # SCHEDULED
        leagues = [l for l in (await get_current_leagues(client)) if not _is_league_excluded(l.get("type"))]
        log(f"Leagues (current, filtered): {len(leagues)}")
        today = datetime.now(timezone.utc).date()
        dates = [(today + timedelta(days=i)).isoformat() for i in range(SCAN_DAYS)]

        for date_iso in dates:
            log(f"Scanning date {date_iso} across {len(leagues)} leagues...")
            tasks = [get_fixtures_by_date(client, l["leagueId"], l["season"], date_iso) for l in leagues]
            per_league = await asyncio.gather(*tasks, return_exceptions=True)

            fixtures: List[Dict] = []
            for l, res in zip(leagues, per_league):
                if isinstance(res, Exception):
                    warn("Fixtures fetch failed:", str(res))
                    continue
                fixtures.extend([{"f": f, "meta": l} for f in res])

            log(f"Found {len(fixtures)} fixtures on {date_iso}")

            for item in fixtures:
                if tips_sent >= MAX_TIPS_PER_RUN:
                    break
                f, meta = item["f"], item["meta"]
                try:
                    tips = await generate_tips_for_fixture(client, f, meta["leagueId"], meta["season"])
                    fixtures_checked += 1
                    if tips:
                        tips_sent += await _send_tips(client, tips)
                except Exception as e:
                    warn(f"Tip generation failed for fixture {((f.get('fixture') or {}).get('id'))}: {e}")

    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    log(f"Scan complete: fixtures checked={fixtures_checked}, tips sent={tips_sent}, elapsed={elapsed:.1f}s")
    return {"fixturesChecked": fixtures_checked, "tipsSent": tips_sent, "elapsedSeconds": int(elapsed)}
