# app/api_client.py
import os
import time
import logging
import requests
from typing import Dict, Any, List, Optional
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# ── Environment ─────────────────────────────
API_KEY = os.getenv("API_KEY")
BASE_URL = "https://v3.football.api-sports.io"
FOOTBALL_API_URL = f"{BASE_URL}/fixtures"

HEADERS = {"x-apisports-key": API_KEY, "Accept": "application/json"}

# Live/in-play statuses (per API-FOOTBALL)
INPLAY_STATUSES = ["1H", "HT", "2H", "ET", "BT", "P"]

# ── Session with retry ──────────────────────
session = requests.Session()
retries = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    respect_retry_after_header=True,
)
session.mount("https://", HTTPAdapter(max_retries=retries))

# ── In-memory caches ────────────────────────
STATS_CACHE: Dict[int, tuple] = {}
EVENTS_CACHE: Dict[int, tuple] = {}
MAX_IDS_PER_REQ = 20


# ── Core API helper ─────────────────────────
def _api_get(url: str, params: dict, timeout: int = 15) -> Optional[dict]:
    if not API_KEY:
        logging.error("[API] API_KEY not set; skipping request to %s", url)
        return None
    try:
        res = session.get(url, headers=HEADERS, params=params, timeout=timeout)
        if not res.ok:
            logging.error(f"[API] {url} status={res.status_code} body={res.text[:300]}")
            return None
        # warn if near rate limit
        try:
            rem = int(res.headers.get("X-RateLimit-Remaining", "-1"))
            if 0 <= rem <= 2:
                logging.warning(f"[API] Low remaining quota: {rem}")
        except Exception:
            pass
        return res.json()
    except Exception as e:
        logging.exception(f"[API] error %s: %s", url, e)
        return None


# ── Fetch functions ─────────────────────────
def fetch_match_stats(fixture_id: int) -> Optional[List[Dict[str, Any]]]:
    now = time.time()
    if fixture_id in STATS_CACHE:
        ts, data = STATS_CACHE[fixture_id]
        if now - ts < 90:
            return data
    js = _api_get(f"{FOOTBALL_API_URL}/statistics", {"fixture": fixture_id})
    stats = js.get("response", []) if isinstance(js, dict) else None
    STATS_CACHE[fixture_id] = (now, stats or [])
    return stats


def fetch_match_events(fixture_id: int) -> Optional[List[Dict[str, Any]]]:
    now = time.time()
    if fixture_id in EVENTS_CACHE:
        ts, data = EVENTS_CACHE[fixture_id]
        if now - ts < 90:
            return data
    js = _api_get(f"{FOOTBALL_API_URL}/events", {"fixture": fixture_id})
    evs = js.get("response", []) if isinstance(js, dict) else None
    EVENTS_CACHE[fixture_id] = (now, evs or [])
    return evs


def fetch_live_matches() -> List[Dict[str, Any]]:
    js = _api_get(FOOTBALL_API_URL, {"live": "all"})
    if not isinstance(js, dict):
        return []
    matches = js.get("response", []) or []
    out = []
    for m in matches:
        status = (m.get("fixture", {}) or {}).get("status", {}) or {}
        elapsed = status.get("elapsed")
        if elapsed is None or elapsed > 90:
            continue
        fid = (m.get("fixture", {}) or {}).get("id")
        try:
            m["statistics"] = fetch_match_stats(fid) or []
        except Exception:
            m["statistics"] = []
        try:
            m["events"] = fetch_match_events(fid) or []
        except Exception:
            m["events"] = []
        out.append(m)
    logging.info(f"[FETCH] live={len(matches)} kept={len(out)}")
    return out


def fetch_fixtures_by_ids(ids: List[int]) -> Dict[int, Dict[str, Any]]:
    """
    Comma-joined bulk (max 20) with one retry and per-ID fallback.
    """
    out: Dict[int, Dict[str, Any]] = {}
    if not ids:
        return out

    for i in range(0, len(ids), MAX_IDS_PER_REQ):
        chunk = ids[i:i + MAX_IDS_PER_REQ]

        # Try bulk up to 2 attempts
        resp_ok = False
        for _ in range(2):
            js = _api_get(FOOTBALL_API_URL, {"ids": ",".join(str(x) for x in chunk), "timezone": "UTC"})
            resp = js.get("response", []) if isinstance(js, dict) else []
            for fx in resp:
                fid = (fx.get("fixture") or {}).get("id")
                if fid:
                    out[int(fid)] = fx
            missing = [fid for fid in chunk if fid not in out]
            if not missing:
                resp_ok = True
                break
            time.sleep(0.35)

        # Per-ID fallback
        if not resp_ok:
            for fid in missing:
                js1 = _api_get(FOOTBALL_API_URL, {"id": fid, "timezone": "UTC"})
                r1 = (js1.get("response") if isinstance(js1, dict) else []) or []
                if r1:
                    out[int(fid)] = r1[0]
                else:
                    logging.warning("[FETCH] single-id fallback still missing fixture %s", fid)

        time.sleep(0.15)

    return out


def fetch_fixtures_by_ids_hyphen(ids: List[int]) -> Dict[int, Dict[str, Any]]:
    """
    Robust bulk fetch (hyphen-joined, max 20) with fallback.
    """
    if not ids:
        return {}

    out: Dict[int, Dict[str, Any]] = {}
    for i in range(0, len(ids), MAX_IDS_PER_REQ):
        chunk = ids[i:i + MAX_IDS_PER_REQ]

        # Try bulk up to 2 attempts
        resp_ok = False
        for _ in range(2):
            js = _api_get(
                FOOTBALL_API_URL,
                {"ids": "-".join(str(x) for x in chunk), "timezone": "UTC"}
            )
            resp = js.get("response", []) if isinstance(js, dict) else []
            for fx in resp:
                fid = (fx.get("fixture") or {}).get("id")
                if fid:
                    out[int(fid)] = fx
            missing = [fid for fid in chunk if fid not in out]
            if not missing:
                resp_ok = True
                break
            time.sleep(0.35)

        # Per-ID fallback
        if not resp_ok:
            for fid in missing:
                js1 = _api_get(FOOTBALL_API_URL, {"id": fid, "timezone": "UTC"})
                r1 = (js1.get("response") if isinstance(js1, dict) else []) or []
                if r1:
                    out[int(fid)] = r1[0]
                else:
                    logging.warning("[FETCH] single-id fallback still missing fixture %s", fid)

        time.sleep(0.15)

    return out
