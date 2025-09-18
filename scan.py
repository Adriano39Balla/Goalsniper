# file: scan.py
# scanning logic for in-play + prematch + MOTD

from __future__ import annotations

import os
import time
import random
import datetime
import logging
from typing import List, Tuple, Optional
from zoneinfo import ZoneInfo

from db import db_conn
try:
    # Optional: use transactional context if available in db.py (from my improved version)
    from db import tx as db_tx  # type: ignore
except Exception:
    db_tx = None  # fallback to autocommit path

from psycopg2.extras import execute_values  # safe even if not used on fallback
from telegram_utils import send_telegram
from odds import fetch_odds, price_gate

log = logging.getLogger("goalsniper")

TZ_UTC = ZoneInfo("UTC")
BERLIN_TZ = ZoneInfo("Europe/Berlin")

# ───────── ENV ─────────
CONF_MIN = float(os.getenv("CONF_MIN", "0.75"))
EV_MIN = float(os.getenv("EV_MIN", "0.0"))  # require non-negative EV by default
MOTD_CONF_MIN = float(os.getenv("MOTD_CONF_MIN", "0.78"))
MOTD_EV_MIN = float(os.getenv("MOTD_EV_MIN", "0.05"))
FEED_STALE_SEC = int(os.getenv("FEED_STALE_SEC", "300"))  # skip stale matches
MAX_TELEGRAM_PER_SCAN = int(os.getenv("MAX_TELEGRAM_PER_SCAN", "5"))  # anti-spam

# ───────── Helpers ─────────
def _now() -> datetime.datetime:
    return datetime.datetime.now(TZ_UTC)

def _is_feed_stale(last_update: Optional[datetime.datetime]) -> bool:
    if not last_update:
        return True
    # normalize naive to UTC if needed
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

    pick_line = f"🎯 *Tip:* {suggestion}"
    conf_line = f"📊 *Confidence:* {conf*100:.1f}%"
    odds_line = f"💰 *Odds:* {odds:.2f} @ {book or 'best'}"
    if ev_pct is not None:
        odds_line += f" • *EV:* {ev_pct:+.1f}%"

    msg = (
        f"⚽️ *{league}*\n"
        f"{home} vs {away}\n"
        f"🕒 Kickoff: {kickoff} Berlin\n"
        f"{pick_line}\n"
        f"{conf_line}\n"
        f"{odds_line}"
    )
    return msg

def _bulk_insert_tips(rows: List[Tuple]) -> int:
    """
    rows tuple layout (match_id, league, home, away, market, suggestion,
                       confidence, confidence_raw, created_ts, odds, book, ev_pct, sent_ok)
    """
    if not rows:
        return 0

    sql = """
        INSERT INTO tips(
            match_id, league, home, away, market, suggestion,
            confidence, confidence_raw, created_ts, odds, book, ev_pct, sent_ok
        ) VALUES %s
        ON CONFLICT DO NOTHING
    """
    # Prefer a transactional context if available
    if db_tx:
        with db_tx() as cur:  # type: ignore
            execute_values(cur, sql, rows, page_size=200)
        return len(rows)
    else:
        with db_conn() as c:
            execute_values(c.cur, sql, rows, page_size=200)  # type: ignore[attr-defined]
        return len(rows)

# ───────── Core Scan ─────────
def production_scan():
    """
    Live scan (currently prematch OU 2.5 as example).
    - Skips stale fixtures
    - Gathers candidates, bulk-inserts accepted tips
    - Sends up to MAX_TELEGRAM_PER_SCAN messages per run
    """
    saved, live_seen = 0, 0
    to_insert: List[Tuple] = []
    to_notify: List[Tuple[str, dict]] = []  # (message, match)

    try:
        with db_conn() as c:
            rows = c.execute(
                """
                SELECT fixture_id, league_name, home, away, kickoff, last_update, status
                FROM fixtures
                WHERE status IN ('NS','TBD')
                """
            ).fetchall()

        for r in rows:
            fid = r[0]
            match = {
                "fid": fid, "league": r[1], "home": r[2], "away": r[3],
                "kickoff": r[4], "last_update": r[5]
            }
            live_seen += 1

            if _is_feed_stale(r[5]):
                continue

            # Example market: OU 2.5 Over (placeholder; replace with model output)
            market, suggestion, conf = "Over/Under 2.5", "Over 2.5 Goals", 0.80

            ok, odds, book, ev_pct = price_gate(market, suggestion, fid, prob=conf)
            if not ok or conf < CONF_MIN or (ev_pct or 0.0) < EV_MIN:
                continue

            now_ts = int(time.time())
            to_insert.append((
                fid, match["league"], match["home"], match["away"],
                market, suggestion,
                conf * 100.0, conf, now_ts, odds, book, ev_pct, 1
            ))

            # queue notification (throttled later)
            msg = _fmt_tip_message(match, market, suggestion, conf, odds, book, ev_pct)
            to_notify.append((msg, match))

        # Bulk insert tips
        saved = _bulk_insert_tips(to_insert)

        # Send up to N messages to avoid spam
        sent = 0
        for msg, _match in to_notify[:MAX_TELEGRAM_PER_SCAN]:
            # small jitter to avoid rate-limits when many
            time.sleep(random.uniform(0.05, 0.2))
            send_telegram(msg)
            sent += 1

        if sent and len(to_notify) > sent:
            log.info("[SCAN] sent %d of %d tips (capped by MAX_TELEGRAM_PER_SCAN=%d)",
                     sent, len(to_notify), MAX_TELEGRAM_PER_SCAN)

    except Exception as e:
        log.exception("[SCAN] failed: %s", e)

    return saved, live_seen

def prematch_scan_save():
    """Save prematch candidates into DB (no Telegram)."""
    saved = 0
    to_insert: List[Tuple] = []
    try:
        with db_conn() as c:
            rows = c.execute(
                "SELECT fixture_id, league_name, home, away, kickoff "
                "FROM fixtures WHERE status IN ('NS','TBD')"
            ).fetchall()

        for r in rows:
            fid = r[0]
            match = {"fid": fid, "league": r[1], "home": r[2], "away": r[3], "kickoff": r[4]}
            market, suggestion, conf = "Over/Under 2.5", "Over 2.5 Goals", 0.78

            ok, odds, book, ev_pct = price_gate(market, suggestion, fid, prob=conf)
            if not ok or conf < CONF_MIN or (ev_pct or 0.0) < EV_MIN:
                continue

            now_ts = int(time.time())
            to_insert.append((
                fid, match["league"], match["home"], match["away"],
                market, suggestion,
                conf * 100.0, conf, now_ts, odds, book, ev_pct, 1
            ))

        saved = _bulk_insert_tips(to_insert)

    except Exception as e:
        log.exception("[PREMATCH] failed: %s", e)
    return saved

# ───────── Digest ─────────
def daily_accuracy_digest():
    """Summarize yesterday’s accuracy and ROI."""
    today = _now().astimezone(BERLIN_TZ).date()
    yesterday = today - datetime.timedelta(days=1)
    msg = None
    try:
        with db_conn() as c:
            rows = c.execute(
                """
                SELECT t.suggestion, t.odds, (r.final_goals_h + r.final_goals_a) AS goals
                FROM tips t
                JOIN match_results r ON t.match_id = r.match_id
                WHERE (to_timestamp(t.created_ts) AT TIME ZONE 'Europe/Berlin')::date = %s::date
                  AND t.sent_ok = 1
                """,
                (yesterday,),
            ).fetchall()

        bets = len(rows)
        if bets == 0:
            return None

        wins = 0
        pnl = 0.0
        for sug, odds, goals in rows:
            s = str(sug or "")
            is_win = False
            if s.startswith("Over"):
                # Using 2.5 shortcut — optional: parse numeric like we do in /stats
                is_win = (int(goals or 0) >= 3)
            elif s.startswith("Under"):
                is_win = (int(goals or 0) <= 2)
            # Extend with BTTS / 1X2 if you use them in prematch

            wins += 1 if is_win else 0
            try:
                o = float(odds)
                pnl += (o - 1.0) if is_win else -1.0
            except Exception:
                # If no odds, treat as push-zero PnL (or skip)
                pass

        hit = wins / bets * 100.0
        msg = f"📊 Digest {yesterday} — {bets} bets | Hit {hit:.1f}% | ROI {pnl:+.2f}u"
        send_telegram(msg)
    except Exception as e:
        log.exception("[DIGEST] failed: %s", e)
    return msg

# ───────── MOTD ─────────
def send_match_of_the_day():
    """Pick best prematch tip for today."""
    candidates: List[Tuple[float, dict, str, str, float, float, str, Optional[float]]] = []
    try:
        with db_conn() as c:
            rows = c.execute(
                "SELECT fixture_id, league_name, home, away, kickoff "
                "FROM fixtures WHERE status IN ('NS','TBD')"
            ).fetchall()

        for r in rows:
            fid = r[0]
            match = {"fid": fid, "league": r[1], "home": r[2], "away": r[3], "kickoff": r[4]}
            market, suggestion, conf = "Over/Under 2.5", "Over 2.5 Goals", 0.80

            ok, odds, book, ev_pct = price_gate(market, suggestion, fid, prob=conf)
            if not ok or conf < MOTD_CONF_MIN or (ev_pct or 0.0) < MOTD_EV_MIN:
                continue

            score = conf + (ev_pct or 0.0) / 100.0
            candidates.append((score, match, market, suggestion, conf, odds, book, ev_pct))

        if not candidates:
            send_telegram("🌟 MOTD — no high-confidence pick today.")
            return False

        candidates.sort(key=lambda x: x[0], reverse=True)
        _, match, market, suggestion, conf, odds, book, ev_pct = candidates[0]
        msg = "🌟 *Match of the Day*\n" + _fmt_tip_message(match, market, suggestion, conf, odds, book, ev_pct)
        send_telegram(msg)
        return True

    except Exception as e:
        log.exception("[MOTD] failed: %s", e)
        return False

# ───────── Retry unsent ─────────
def retry_unsent_tips(minutes: int = 30, limit: int = 200) -> int:
    """
    Re-send tips that failed to deliver to Telegram.
    """
    cutoff = int(time.time()) - minutes * 60
    retried = 0
    rows = []

    with db_conn() as c:
        rows = c.execute(
            """
            SELECT
                match_id,
                league,
                home,
                away,
                market,
                suggestion,
                confidence,
                confidence_raw,
                score_at_tip,
                minute,
                created_ts,
                odds,
                book,
                ev_pct
            FROM tips
            WHERE sent_ok = 0
              AND created_ts >= %s
            ORDER BY created_ts ASC
            LIMIT %s
            """,
            (cutoff, limit),
        ).fetchall()

    # resend and gather keys to mark delivered
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
            with db_tx() as cur:  # type: ignore
                execute_values(cur,
                    "UPDATE tips AS t SET sent_ok=1 FROM (VALUES %s) AS v(match_id, created_ts) "
                    "WHERE t.match_id=v.match_id AND t.created_ts=v.created_ts",
                    delivered_keys
                )
        else:
            with db_conn() as c2:
                # In lack of tx(), fall back to individual updates (less ideal)
                for mid, cts in delivered_keys:
                    c2.execute("UPDATE tips SET sent_ok=1 WHERE match_id=%s AND created_ts=%s", (mid, cts))
        retried = len(delivered_keys)
        log.info("[RETRY] resent %d", retried)

    return retried

# ───────── Backfill ─────────
def backfill_results_for_open_matches(limit=300):
    """Fetch results for fixtures still open and update DB (placeholder)."""
    log.info("[BACKFILL] placeholder run (limit=%s)", limit)
    return 0
