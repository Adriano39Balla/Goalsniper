# ... existing imports ...
import os
# (keep the rest of your imports as-is)

# NEW: hard caps driven by env so you don’t need to edit config.py
MAX_TODAY_FIXTURES = int(os.getenv("MAX_TODAY_FIXTURES", "24"))   # how many NS fixtures to consider per run
BUILD_FIXTURE_LIMIT = int(os.getenv("BUILD_FIXTURE_LIMIT", "20")) # how many fixtures we pass to the tip builder

# (keep everything else unchanged until the fetchers)

async def _fetch_today(client: httpx.AsyncClient) -> List[Dict]:
    # ... unchanged prelude ...
    out: List[Dict] = []
    day = date.today().isoformat()

    for row in leagues:
        # stop if we already collected enough fixtures
        if len(out) >= MAX_TODAY_FIXTURES:
            break
        try:
            lid = int(row.get("leagueId") or 0)
            season = int(row.get("season") or 0)
            if not lid or not season:
                continue
            res = await by_date_fn(client, lid, season, day) if asyncio.iscoroutinefunction(by_date_fn) else by_date_fn(client, lid, season, day)
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

async def _build_tips_for_fixtures(client: httpx.AsyncClient, fixtures: List[Dict]) -> Tuple[List[Dict], Dict[str, int]]:
    stats = {"raw_candidates": 0, "low_conf": 0, "inrun_dedup_removed": 0}
    build_fn = _get_build_tips_fn()
    if not build_fn:
        warn("No tip builder function found in tips module; skipping tip generation")
        return [], stats

    raw: List[Dict] = []
    # *** only process the first N fixtures to limit /teams/statistics calls ***
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

    # ... rest of function unchanged ...
    stats["raw_candidates"] = len(raw)
    conf_filtered = [t for t in raw if float(t.get("confidence", 0.0)) >= MIN_CONFIDENCE_TO_SEND]
    stats["low_conf"] = len(raw) - len(conf_filtered)
    deduped, removed = _dedupe_in_run(conf_filtered)
    stats["inrun_dedup_removed"] = removed
    return deduped, stats
