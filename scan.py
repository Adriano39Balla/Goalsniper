# file: scan.py
# adaptive scanning logic for in-play + prematch + MOTD (selects best market per match)

from __future__ import annotations

import os
import time
import random
import datetime
import logging
from typing import List, Tuple, Optional, Dict
from zoneinfo import ZoneInfo

from db import db_conn
try:
    from db import tx as db_tx  # transactional context from db.py
except Exception:
    db_tx = None

from psycopg2.extras import execute_values
from telegram_utils import send_telegram
from results_provider import fetch_results_for_fixtures, update_match_results
from odds import fetch_odds, price_gate

log = logging.getLogger("goalsniper")

TZ_UTC = ZoneInfo("UTC")
BERLIN_TZ = ZoneInfo("Europe/Berlin")

# ───────── ENV ─────────
CONF_MIN = float(os.getenv("CONF_MIN", "0.75"))          # min probability to consider (0..1)
EV_MIN = float(os.getenv("EV_MIN", "0.0"))               # min EV (0 = non-negative)
MOTD_CONF_MIN = float(os.getenv("MOTD_CONF_MIN", "0.78"))
MOTD_EV_MIN = float(os.getenv("MOTD_EV_MIN", "0.05"))
FEED_STALE_SEC = int(os.getenv("FEED_STALE_SEC", "300"))
MAX_TELEGRAM_PER_SCAN = int(os.getenv("MAX_TELEGRAM_PER_SCAN", "5"))

# ───────── Optional model hook ─────────
# Provide predictor.py with:
#   def predict_for_fixture(fid: int) -> dict:  # keys match suggestion labels
#       return {"Over 2.5 Goals": 0.61, "BTTS: Yes": 0.58, "Home Win": 0.44, ...}
try:
    from predictor import predict_for_fixture  # type: ignore
except Exception:
    predict_for_fixture = None  # type: ignore

# ───────── Helpers ─────────
def _now() -> datetime.datetime:
    return datetime.datetime.now(TZ_UTC)

def _is_feed_stale(last_update: Optional[datetime.datetime]) -> bool:
    if not last_update:
        return True
    if last_update.tzinfo is None:
        last_update = last_update.replace(tzinfo=TZ_UTC)
    return (_now() - last_update).total_seconds() > FEED_STALE_SEC

def _fmt_tip_message(match, market, suggestion, conf, odds, book, ev_pct):
    ko_dt = match.get("kickoff")
    if ko_dt and ko_dt.tzinfo is None:
        ko_dt = ko_dt.replace(tzinfo=TZ_UTC)
    kickoff = (ko_dt or _now()).astimezone(BERLIN_TZ).strftime("%Y-%m-%d %H:%M")
    home, away = match.get("home"), match.get("away")
    league = match.get("league")

    msg = (
        f"⚽️ *{league}*\n"
        f"{home} vs {away}\n"
        f"🕒 Kickoff: {kickoff} Berlin\n"
        f"🎯 *Tip:* {suggestion}\n"
        f"📊 *Confidence:* {conf*100:.1f}%\n"
        f"💰 *Odds:* {odds:.2f} @ {book or 'best'}"
    )
    if ev_pct is not None:
        msg += f" • *EV:* {ev_pct:+.1f}%"
    return msg

def _bulk_insert_tips(rows: List[Tuple]) -> int:
    if not rows:
        return 0
    sql = """
        INSERT INTO tips(
            match_id, league, home, away, market, suggestion,
            confidence, confidence_raw, created_ts, odds, book, ev_pct, sent_ok
        ) VALUES %s
        ON CONFLICT DO NOTHING
    """
    if db_tx:
        with db_tx() as cur:
            execute_values(cur, sql, rows, page_size=200)
    else:
        with db_conn() as c:
            execute_values(c.cur, sql, rows, page_size=200)  # wrapper exposes .cur
    return len(rows)

# ───────── Probability helpers ─────────
def _implied_prob(odds: float) -> float:
    try:
        o = max(1e-6, float(odds))
        return 1.0 / o
    except Exception:
        return 0.0

def _normalize_pair(a: float, b: float) -> Tuple[float, float]:
    s = a + b
    if s <= 0:
        return 0.0, 0.0
    return a / s, b / s

def _prob_hints_from_model_or_odds(fid: int, odds_map: dict) -> Dict[str, float]:
    # 1) Model override if available
    if predict_for_fixture:
        try:
            probs = predict_for_fixture(fid)  # type: ignore
            if isinstance(probs, dict) and probs:
                return {str(k): float(v) for k, v in probs.items() if v is not None}
        except Exception as e:
            log.warning("[MODEL] predict_for_fixture failed for %s: %s", fid, e)

    # 2) Fallback: odds-implied probabilities with simple de-vig per two-way market
    hints: Dict[str, float] = {}

    # BTTS Yes/No
    d = odds_map.get("BTTS", {})
    if d:
        p_yes = _implied_prob(d.get("Yes", {}).get("odds")) if "Yes" in d else 0.0
        p_no  = _implied_prob(d.get("No",  {}).get("odds")) if "No"  in d else 0.0
        p_yes, p_no = _normalize_pair(p_yes, p_no)
        hints["BTTS: Yes"] = p_yes
        hints["BTTS: No"]  = p_no

    # 1X2 Home/Away (ignore Draw; renormalize H/A)
    d = odds_map.get("1X2", {})
    if d:
        p_h = _implied_prob(d.get("Home", {}).get("odds")) if "Home" in d else 0.0
        p_a = _implied_prob(d.get("Away", {}).get("odds")) if "Away" in d else 0.0
        p_h, p_a = _normalize_pair(p_h, p_a)
        hints["Home Win"] = p_h
        hints["Away Win"] = p_a

    # OU lines
    for k, sides in odds_map.items():
        if not k.startswith("OU_"):
            continue
        line = k.split("_", 1)[1]
        p_over = _implied_prob(sides.get("Over", {}).get("odds")) if "Over" in sides else 0.0
        p_under= _implied_prob(sides.get("Under",{}).get("odds")) if "Under" in sides else 0.0
        p_over, p_under = _normalize_pair(p_over, p_under)
        hints[f"Over {line} Goals"]  = p_over
        hints[f"Under {line} Goals"] = p_under

    return hints

# ───────── Adaptive candidate selection ─────────
def _best_candidate_for_fixture(fid: int) -> Optional[Tuple[str, str, float, float, str, Optional[float]]]:
    """
    Returns best passing candidate:
    (market, suggestion, prob, odds, book, ev_pct) or None.
    """
    odds_map = fetch_odds(fid)
    if not odds_map:
        return None

    prob_hints = _prob_hints_from_model_or_odds(fid, odds_map)

    candidates: List[Tuple[str, str, float]] = []

    # BTTS
    if "BTTS: Yes" in prob_hints:
        candidates.append(("BTTS", "BTTS: Yes", prob_hints["BTTS: Yes"]))
    if "BTTS: No" in prob_hints:
        candidates.append(("BTTS", "BTTS: No", prob_hints["BTTS: No"]))

    # 1X2 (Home/Away only)
    if "Home Win" in prob_hints:
        candidates.append(("1X2", "Home Win", prob_hints["Home Win"]))
    if "Away Win" in prob_hints:
        candidates.append(("1X2", "Away Win", prob_hints["Away Win"]))

    # OU lines
    for k in sorted([kk for kk in odds_map.keys() if str(kk).startswith("OU_")]):
        ln = k.split("_", 1)[1]
        over_key, under_key = f"Over {ln} Goals", f"Under {ln} Goals"
        if over_key in prob_hints:
            candidates.append((f"Over/Under {ln}", over_key, prob_hints[over_key]))
        if under_key in prob_hints:
            candidates.append((f"Over/Under {ln}", under_key, prob_hints[under_key]))

    best = None
    best_ev = float("-inf")

    for market, suggestion, prob in candidates:
        ok, odds, book, ev_pct = price_gate(market, suggestion, fid, prob=prob)
        if not ok:
            continue
        if prob < CONF_MIN:
            continue
        if (ev_pct or 0.0) < (EV_MIN * 100.0):
            continue
        edge = (ev_pct or 0.0)
        if edge > best_ev:
            best_ev = edge
            best = (market, suggestion, prob, float(odds), str(book or "best"),
                    float(ev_pct) if ev_pct is not None else None)

    return best

# ───────── Core Scan (adaptive) ─────────
def production_scan():
    """
    Pulls candidates per fixture, computes EV from model or odds-implied,
    saves only the best passing pick per fixture, and sends up to N messages.
    """
    saved, live_seen = 0, 0
    to_insert: List[Tuple] = []
    to_notify: List[Tuple[str, dict]] = []

    try:
        with db_conn() as c:
            rows = c.execute(
                """
                SELECT fixture_id, league_name, home, away, kickoff, last_update, status
                FROM fixtures
                WHERE
                  status IN ('1H','HT','2H','ET','P','LIVE')
                  OR (status IN ('NS','TBD') AND kickoff >= now() AND kickoff <= now() + interval '60 minutes')
                """
            ).fetchall()

        for fid, league, home, away, kickoff, last_update, status in rows:
            live_seen += 1
            if _is_feed_stale(last_update):
                continue

            best = _best_candidate_for_fixture(fid)
            if not best:
                continue

            market, suggestion, prob, odds, book, ev_pct = best
            now_ts = int(time.time())
            to_insert.append((
                fid, league, home, away,
                market, suggestion,
                prob * 100.0, prob, now_ts, odds, book, ev_pct, 1
            ))

            match = {"league": league, "home": home, "away": away, "kickoff": kickoff}
            msg = _fmt_tip_message(match, market, suggestion, prob, odds, book, ev_pct)
            to_notify.append((msg, match))

        saved = _bulk_insert_tips(to_insert)

        sent = 0
        for msg, _ in to_notify[:MAX_TELEGRAM_PER_SCAN]:
            time.sleep(random.uniform(0.05, 0.2))
            send_telegram(msg)
            sent += 1

        if sent and len(to_notify) > sent:
            log.info("[SCAN] sent %d of %d tips (cap=%d)", sent, len(to_notify), MAX_TELEGRAM_PER_SCAN)

    except Exception as e:
        log.exception("[SCAN] failed: %s", e)

    return saved, live_seen

# ───────── Prematch (adaptive) ─────────
def prematch_scan_save():
    saved = 0
    to_insert: List[Tuple] = []
    try:
        with db_conn() as c:
            c.execute("""
                SELECT fixture_id, league_name, home, away, kickoff
                FROM fixtures WHERE status IN ('NS','TBD')
            """)
            rows = c.fetchall()

        for fid, league, home, away, kickoff in rows:
            best = _best_candidate_for_fixture(fid)
            if not best:
                continue
            market, suggestion, prob, odds, book, ev_pct = best
            now_ts = int(time.time())
            to_insert.append((
                fid, league, home, away,
                market, suggestion,
                prob * 100.0, prob, now_ts, odds, book, ev_pct, 1
            ))

        saved = _bulk_insert_tips(to_insert)
    except Exception as e:
        log.exception("[PREMATCH] failed: %s", e)
    return saved

# ───────── Digest (uses stored results) ─────────
def daily_accuracy_digest():
    """Summarize yesterday’s accuracy and ROI (proper grading for BTTS/OU/1X2)."""
    today = _now().astimezone(BERLIN_TZ).date()
    yesterday = today - datetime.timedelta(days=1)
    msg = None
    try:
        with db_conn() as c:
            c.execute("""
                SELECT t.suggestion, t.odds,
                       r.final_goals_h, r.final_goals_a, r.btts_yes
                FROM tips t
                JOIN match_results r ON t.match_id = r.match_id
                WHERE (to_timestamp(t.created_ts) AT TIME ZONE 'Europe/Berlin')::date = %s::date
                  AND t.sent_ok = 1
            """, (yesterday,))
            rows = c.fetchall()

        if not rows:
            return None

        wins, pnl = 0, 0.0
        for sug, odds, gh, ga, btts_yes in rows:
            s = str(sug or "")
            gh = int(gh or 0); ga = int(ga or 0)
            total = gh + ga

            is_win = False
            if s.startswith("Over") or s.startswith("Under"):
                # parse line from suggestion "Over X.Y Goals"
                try:
                    line = float(s.split()[1])
                except Exception:
                    line = 2.5
                is_win = (total > line) if s.startswith("Over") else (total < line)
            elif s == "BTTS: Yes":
                is_win = bool(btts_yes)
            elif s == "BTTS: No":
                is_win = not bool(btts_yes)
            elif s == "Home Win":
                is_win = gh > ga
            elif s == "Away Win":
                is_win = ga > gh

            if is_win:
                wins += 1
            try:
                o = float(odds)
                pnl += (o - 1.0) if is_win else -1.0
            except Exception:
                pass

        hit = wins / len(rows) * 100.0
        msg = f"📊 Digest {yesterday} — {len(rows)} bets | Hit {hit:.1f}% | ROI {pnl:+.2f}u"
        send_telegram(msg)
    except Exception as e:
        log.exception("[DIGEST] failed: %s", e)
    return msg

# ───────── MOTD (adaptive) ─────────
def send_match_of_the_day():
    candidates: List[Tuple[float, dict, str, str, float, float, str, Optional[float]]] = []
    try:
        with db_conn() as c:
            c.execute("""
                SELECT fixture_id, league_name, home, away, kickoff
                FROM fixtures WHERE status IN ('NS','TBD')
            """)
            rows = c.fetchall()

        for fid, league, home, away, kickoff in rows:
            best = _best_candidate_for_fixture(fid)
            if not best:
                continue
            market, suggestion, prob, odds, book, ev_pct = best
            score = prob + ((ev_pct or 0.0) / 100.0)
            if prob >= MOTD_CONF_MIN and (ev_pct or 0.0) >= (MOTD_EV_MIN * 100.0):
                candidates.append((score, {"league": league, "home": home, "away": away, "kickoff": kickoff},
                                   market, suggestion, prob, odds, book, ev_pct))

        if not candidates:
            send_telegram("🌟 MOTD — no high-confidence pick today.")
            return False

        candidates.sort(key=lambda x: x[0], reverse=True)
        _, match, market, suggestion, prob, odds, book, ev_pct = candidates[0]
        msg = "🌟 *Match of the Day*\n" + _fmt_tip_message(match, market, suggestion, prob, odds, book, ev_pct)
        send_telegram(msg)
        return True

    except Exception as e:
        log.exception("[MOTD] failed: %s", e)
        return False

# ───────── Retry unsent ─────────
def retry_unsent_tips(minutes: int = 30, limit: int = 200) -> int:
    cutoff = int(time.time()) - minutes * 60
    retried = 0
    try:
        with db_conn() as c:
            c.execute("""
                SELECT match_id, league, home, away, market, suggestion,
                       confidence, confidence_raw, score_at_tip, minute,
                       created_ts, odds, book, ev_pct
                FROM tips
                WHERE sent_ok = 0 AND created_ts >= %s
                ORDER BY created_ts ASC
                LIMIT %s
            """, (cutoff, limit))
            rows = c.fetchall()

        delivered_keys: List[Tuple[int, int]] = []
        for (
            mid, league, home, away, market, sugg,
            conf_pct, conf_raw, score, minute, cts,
            odds, book, ev_pct
        ) in rows:
            pct = float(conf_pct if conf_pct is not None else (100.0 * float(conf_raw or 0.0)))
            msg = f"♻️ RETRY\n{league}: {home} vs {away}\nTip: {sugg}\nConf: {pct:.1f}%\nOdds: {odds or '-'}"
            ok = send_telegram(msg)
            if ok:
                delivered_keys.append((mid, cts))

        if delivered_keys:
            if db_tx:
                with db_tx() as cur:
                    execute_values(cur,
                        "UPDATE tips AS t SET sent_ok=1 FROM (VALUES %s) AS v(match_id, created_ts) "
                        "WHERE t.match_id=v.match_id AND t.created_ts=v.created_ts",
                        delivered_keys
                    )
            else:
                with db_conn() as c2:
                    for mid, cts in delivered_keys:
                        c2.execute("UPDATE tips SET sent_ok=1 WHERE match_id=%s AND created_ts=%s", (mid, cts))
            retried = len(delivered_keys)
            log.info("[RETRY] resent %d", retried)
    except Exception as e:
        log.exception("[RETRY] failed: %s", e)
    return retried

# ───────── Backfill ─────────
def backfill_results_for_open_matches(limit=300):
    """Fetch results for fixtures still open and update DB."""
    try:
        with db_conn() as c:
            c.execute(
                "SELECT fixture_id FROM fixtures WHERE status NOT IN ('FT','AET','PEN') ORDER BY last_update ASC LIMIT %s",
                (limit,),
            )
            ids = [r[0] for r in c.fetchall()]
        if not ids:
            return 0
        rows = list(fetch_results_for_fixtures(ids))
        if not rows:
            return 0
        return update_match_results(rows)
    except Exception as e:
        log.exception("[BACKFILL] failed: %s", e)
        return 0
