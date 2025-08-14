# goalsniper/server.py
from __future__ import annotations

import os
import importlib
import asyncio
from datetime import datetime, timezone
from typing import Optional, Any, Dict, Tuple, List

import httpx
from fastapi import FastAPI, Request, HTTPException, Query, Response

app = FastAPI(title="Goalsniper", version="1.6.4")

# -------------------------
# env helpers
# -------------------------
def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default)

def _run_token() -> str:
    tok = _env("RUN_TOKEN", "")
    if not tok:
        raise HTTPException(status_code=500, detail="RUN_TOKEN not set")
    return tok

def _telegram_webhook_token() -> str:
    return _env("TELEGRAM_WEBHOOK_TOKEN", "")

def _safe_import(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"import error: {module_name}: {e}")

# optional logger (defensive import)
try:
    _logger = importlib.import_module("goalsniper.logger")
    log = getattr(_logger, "log", print)
    warn = getattr(_logger, "warn", print)
except Exception:
    def log(*a, **k):  # noqa: N802
        print(*a)
    warn = log

def _auth_header(request: Request):
    auth = request.headers.get("authorization") or ""
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    token = auth[len("Bearer "):].strip()
    if token != _run_token():
        raise HTTPException(status_code=403, detail="Forbidden")

def _auth_qs(token: str):
    if not token or token != _run_token():
        raise HTTPException(status_code=401, detail="Unauthorized")

# -------------------------
# Health & root (HEAD-safe)
# -------------------------
@app.api_route("/health", methods=["GET", "HEAD"])
async def health(request: Request):
    if request.method == "HEAD":
        return Response(status_code=200)
    return {"ok": True, "name": "Goalsniper", "time": os.getenv("TZ", "UTC")}

@app.api_route("/", methods=["GET", "HEAD"])
async def root(request: Request):
    if request.method == "HEAD":
        return Response(status_code=200)
    return {"ok": True, "name": "Goalsniper", "time": os.getenv("TZ", "UTC")}

# -------------------------
# Scan orchestration (non‑blocking)
# -------------------------
_scan_lock = asyncio.Lock()
_scan_task: Optional[asyncio.Task] = None
_last_run_started: Optional[str] = None
_last_run_finished: Optional[str] = None
_last_run_result: Optional[Dict[str, Any]] = None
_last_run_error: Optional[str] = None

async def _do_scan():
    """Run one scan and update status snapshots."""
    global _last_run_started, _last_run_finished, _last_run_result, _last_run_error
    scanner = importlib.import_module("goalsniper.scanner")
    _last_run_error = None
    _last_run_started = datetime.now(timezone.utc).isoformat()
    log(f"[server] scan starting at {_last_run_started}")
    try:
        res = await scanner.run_scan_and_send()
        _last_run_result = {
            "fixturesChecked": res.get("fixturesChecked", 0),
            "tipsSent": res.get("tipsSent", 0),
            "elapsedSeconds": res.get("elapsedSeconds", 0),
            "debug": res.get("debug", {}),
        }
        log(f"[server] scan finished: tipsSent={_last_run_result['tipsSent']} "
            f"fixturesChecked={_last_run_result['fixturesChecked']}")
    except Exception as e:
        _last_run_error = str(e)
        _last_run_result = None
        warn(f"[server] scan error: {e}")
    finally:
        _last_run_finished = datetime.now(timezone.utc).isoformat()

def _kick_off_background_scan() -> Tuple[bool, str]:
    """Start a background scan if none is running. Returns (started, reason)."""
    global _scan_task
    if _scan_task and not _scan_task.done():
        return False, "already_running"
    _scan_task = asyncio.create_task(_do_scan())
    return True, ""

# -----------------------------------------
# /run (non‑blocking)
# -----------------------------------------
@app.post("/run")
async def run_post(request: Request):
    _auth_header(request)
    started, reason = _kick_off_background_scan()
    if started:
        log("[server] /run accepted -> started")
    else:
        log(f"[server] /run accepted -> not started (reason={reason})")
    return Response(
        content=(b'{"accepted":true,"started":' +
                 (b"true" if started else b"false") +
                 (b',"reason":"' + reason.encode("utf-8") + b'"}' if not started else b"}")),
        status_code=202,
        media_type="application/json",
    )

@app.api_route("/run", methods=["GET", "HEAD"])
async def run_get(request: Request, token: str = Query("")):
    _auth_qs(token)
    if request.method == "HEAD":
        return Response(status_code=200)
    started, reason = _kick_off_background_scan()
    if started:
        log("[server] /run (GET) accepted -> started")
    else:
        log(f"[server] /run (GET) accepted -> not started (reason={reason})")
    return {"accepted": True, "started": bool(started), **({"reason": reason} if not started else {})}

# -----------------------------------------
# /run/wait — blocking one‑shot
# -----------------------------------------
@app.post("/run/wait")
async def run_wait_post(request: Request):
    _auth_header(request)
    return await _run_wait_impl()

@app.get("/run/wait")
async def run_wait_get(token: str = Query("")):
    _auth_qs(token)
    return await _run_wait_impl()

async def _run_wait_impl():
    async with _scan_lock:
        global _scan_task
        if _scan_task and not _scan_task.done():
            log("[server] /run/wait — awaiting running scan")
            await _scan_task
            _scan_task = None
        await _do_scan()
        return {
            "status": "ok",
            "startedAt": _last_run_started,
            "finishedAt": _last_run_finished,
            "result": _last_run_result,
            "error": _last_run_error,
        }

# -----------------------------------------
# /run/status — inspect last/background state
# -----------------------------------------
@app.get("/run/status")
async def run_status(token: str = Query("")):
    _auth_qs(token)
    running = bool(_scan_task and not _scan_task.done())
    return {"running": running, "startedAt": _last_run_started, "finishedAt": _last_run_finished,
            "result": _last_run_result, "error": _last_run_error}

# -------------------------
# Telegram webhook (feedback)
# -------------------------
@app.post("/telegram/webhook")
@app.post("/telegram/webhook/{token}")
async def telegram_webhook(request: Request, token: Optional[str] = None):
    secret = _telegram_webhook_token()
    if secret and token != secret:
        raise HTTPException(status_code=403, detail="Forbidden")

    payload = await request.json()
    cq = payload.get("callback_query")
    if not cq:
        return {"ok": True}

    data = (cq.get("data") or "").strip()
    if not data.startswith("fb:"):
        return {"ok": True}

    try:
        _, tip_id_s, outcome_s = data.split(":")
        tip_id = int(tip_id_s)
        outcome = 1 if outcome_s == "1" else 0

        storage = _safe_import("goalsniper.storage")
        learning = _safe_import("goalsniper.learning")
        logger = _safe_import("goalsniper.logger")
        telegram = _safe_import("goalsniper.telegram")

        await storage.set_outcome(tip_id, outcome)

        try:
            info = await learning.on_feedback_update(tip_id, outcome)
            if info.get("ok"):
                logger.log(
                    f"[learn] tip={tip_id} market={info.get('market')} "
                    f"p_cal={info.get('calibratedProb'):.3f} conf={info.get('confidence'):.3f} "
                    f"thr={info.get('marketThreshold'):.3f} outcome={outcome}"
                )
        except Exception as le:
            logger.log(f"[learn] update failed for tip_id={tip_id}: {le}")

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                await telegram.answer_callback_query(client, cq.get("id"), "Recorded ✅")
        except Exception as e:
            logger.log(f"answerCallbackQuery failed: {e}")

        logger.log(f"Feedback received tip_id={tip_id} outcome={outcome}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"bad data: {str(e)}")

    return {"ok": True}

# -------------------------
# Daily digest
# -------------------------
@app.api_route("/digest", methods=["GET", "HEAD"])
async def digest_get(
    request: Request,
    token: str = Query(""),
    date: str | None = Query(None, description="YYYY-MM-DD in UTC (optional)"),
    push: int = Query(1, description="1=send to Telegram, 0=JSON only"),
):
    _auth_qs(token)

    if request.method == "HEAD":
        return Response(status_code=200)

    storage = _safe_import("goalsniper.storage")
    telegram = _safe_import("goalsniper.telegram")

    try:
        day = (datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
               if date else datetime.now(timezone.utc))
    except Exception:
        raise HTTPException(status_code=400, detail="Bad date format, use YYYY-MM-DD")

    stats = await storage.daily_counts_for(day)
    tot = await storage.totals()

    sent = stats["sent"]; good = stats["good"]; bad = stats["bad"]; pending = stats["pending"]
    acc = (good / sent * 100.0) if sent else 0.0

    day_str = day.strftime("%Y-%m-%d")
    text = (
        f"📅 <b>Daily Goalsniper Digest — {day_str}</b>\n"
        f"Sent: <b>{sent}</b>  |  👍 <b>{good}</b>  |  👎 <b>{bad}</b>  |  ⏳ <b>{pending}</b>\n"
        f"Accuracy: <b>{acc:.1f}%</b>\n"
        f"\n<i>All‑time</i> — Sent {tot['sent']}, 👍 {tot['good']}, 👎 {tot['bad']}"
    )

    result = {"date": day_str, "sent": sent, "good": good, "bad": bad,
              "pending": pending, "accuracy": round(acc, 1), "pushed": False}

    if push:
        async with httpx.AsyncClient(timeout=20) as client:
            try:
                await telegram.send_text(client, text)
                result["pushed"] = True
            except Exception as e:
                result["error"] = f"telegram: {e}"

    return result

# -------------------------
# Debug endpoints (browser‑friendly)
# -------------------------
@app.get("/debug/functions")
async def debug_functions(token: str = Query(""), warm: int = Query(1)):
    _auth_qs(token)
    scanner = _safe_import("goalsniper.scanner")
    api     = _safe_import("goalsniper.api_football")
    tips    = _safe_import("goalsniper.tips")

    if warm:
        try: getattr(scanner, "_get_live_api")()
        except Exception: pass
        try: getattr(scanner, "_get_by_date_api")()
        except Exception: pass
        try: getattr(scanner, "_get_build_tips_fn")()
        except Exception: pass

    info = {
        "scanner.builderFn": getattr(scanner, "FOUND_BUILDER_NAME", None),
        "scanner.liveApiFn": getattr(scanner, "FOUND_LIVE_API_NAME", None),
        "scanner.dateApiFn": getattr(scanner, "FOUND_DATE_API_NAME", None),
        "api_has.get_current_leagues": callable(getattr(api, "get_current_leagues", None)),
        "api_has.get_fixtures_by_date": callable(getattr(api, "get_fixtures_by_date", None)),
        "api_has.get_live_fixtures": callable(getattr(api, "get_live_fixtures", None)),
        "tips_has.generate_tips_for_fixture": callable(getattr(tips, "generate_tips_for_fixture", None)),
    }
    return info

@app.get("/debug/filters")
async def debug_filters(token: str = Query("")):
    _auth_qs(token)
    filters = _safe_import("goalsniper.filters")
    eff = await filters.get_filters()
    return {
        "allowCountries": sorted(list(eff.get("allowCountries", set()))),
        "allowLeagueKeywords": eff.get("allowLeagueKeywords", []),
        "excludeKeywords": eff.get("excludeKeywords", []),
    }

@app.get("/debug/live-count")
async def debug_live_count(token: str = Query("")):
    _auth_qs(token)
    api = _safe_import("goalsniper.api_football")
    async with httpx.AsyncClient(timeout=20) as client:
        try:
            data = await api.get_live_fixtures(client)
            return {"liveFixturesPassingFilters": len(data)}
        except Exception as e:
            return {"error": str(e)}

@app.get("/debug/live-fixtures")
async def debug_live_fixtures(token: str = Query(""), limit: int = Query(15)):
    _auth_qs(token)
    api = _safe_import("goalsniper.api_football")
    async with httpx.AsyncClient(timeout=25) as client:
        try:
            fixtures = await api.get_live_fixtures(client)
            out = []
            for fx in fixtures[:max(1, int(limit))]:
                lg = fx.get("league") or {}
                out.append({
                    "league": lg.get("name"),
                    "country": lg.get("country"),
                    "round": (lg.get("round") or ""),
                    "fixtureId": (fx.get("fixture") or {}).get("id"),
                    "status": ((fx.get("fixture") or {}).get("status") or {}).get("short"),
                })
            return {"total": len(fixtures), "sample": out}
        except Exception as e:
            return {"error": str(e)}

@app.get("/debug/leagues-today")
async def debug_leagues_today(token: str = Query(""), cap: int = Query(50)):
    _auth_qs(token)
    api = _safe_import("goalsniper.api_football")
    filters = _safe_import("goalsniper.filters")

    async with httpx.AsyncClient(timeout=30) as client:
        leagues = await api.get_current_leagues(client)
        eff = await filters.get_filters()

        def allowed(row: Dict) -> bool:
            lname = (row.get("leagueName") or "").upper()
            if not lname:
                return False
            if eff["excludeKeywords"] and any(bad in lname for bad in eff["excludeKeywords"]):
                return False
            if eff["allowLeagueKeywords"] and not any(k in lname for k in eff["allowLeagueKeywords"]):
                return False
            c = (row.get("country") or "").upper()
            if eff["allowCountries"] and c and c not in eff["allowCountries"]:
                return False
            return True

        leagues_ok = [r for r in leagues if allowed(r)]
        day = datetime.now(timezone.utc).date().isoformat()

        out: List[Dict[str, Any]] = []
        for row in leagues_ok[:max(1, int(cap))]:
            lid = int(row.get("leagueId") or 0)
            season = int(row.get("season") or 0)
            try:
                fixes = await api.get_fixtures_by_date(client, lid, season, day)
                out.append({
                    "leagueId": lid,
                    "leagueName": row.get("leagueName"),
                    "country": row.get("country"),
                    "season": season,
                    "fixturesToday": len(fixes),
                })
            except Exception as e:
                out.append({
                    "leagueId": lid,
                    "leagueName": row.get("leagueName"),
                    "country": row.get("country"),
                    "season": season,
                    "error": str(e),
                })
        return {"allowedLeagues": len(leagues_ok), "sampled": len(out), "day": day, "data": out}

@app.get("/debug/diagnostics")
async def debug_diagnostics(token: str = Query(""), liveLimit: int = Query(10)):
    _auth_qs(token)
    funcs = await debug_functions(token, warm=1)
    filts = await debug_filters(token)
    livec = await debug_live_count(token)
    livef = await debug_live_fixtures(token, limit=liveLimit)
    return {"functions": funcs, "filters": filts, "liveCount": livec, "liveSample": livef}

# --- startup log: prove routes are present ---
@app.on_event("startup")
async def _startup_log():
    log(f"[server] startup: {len(app.router.routes)} routes registered")
