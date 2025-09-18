# scanning logic for in-play + prematch + MOTD

import os, time, datetime, logging
from zoneinfo import ZoneInfo

from db import db_conn
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

# ───────── Helpers ─────────
def _now(): return datetime.datetime.now(TZ_UTC)

def _is_feed_stale(last_update: datetime.datetime) -> bool:
    return (_now() - last_update).total_seconds() > FEED_STALE_SEC

def _fmt_tip_message(match, market, suggestion, conf, odds, book, ev_pct):
    kickoff = match.get("kickoff").astimezone(BERLIN_TZ).strftime("%Y-%m-%d %H:%M")
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

# ───────── Core Scan ─────────
def production_scan():
    """Live scan (placeholder: uses prematch odds for now)."""
    saved, live_seen = 0, 0
    try:
        with db_conn() as c:
            rows = c.execute("SELECT fixture_id, league_name, home, away, kickoff, last_update, status "
                             "FROM fixtures WHERE status IN ('NS','TBD')").fetchall()
            for r in rows:
                fid = r[0]
                match = {
                    "fid": fid, "league": r[1], "home": r[2], "away": r[3],
                    "kickoff": r[4], "last_update": r[5]
                }
                live_seen += 1
                if _is_feed_stale(r[5]): 
                    continue
                # Example market: OU 2.5 Over
                market, suggestion, conf = "Over/Under 2.5", "Over 2.5 Goals", 0.80
                ok, odds, book, ev_pct = price_gate(market, suggestion, fid, prob=conf)
                if ok and conf >= CONF_MIN and (ev_pct or 0) >= EV_MIN:
                    msg = _fmt_tip_message(match, market, suggestion, conf, odds, book, ev_pct)
                    send_telegram(msg)
                    c.execute("INSERT INTO tips(match_id, league, home, away, market, suggestion, confidence, confidence_raw, created_ts, odds, book, ev_pct, sent_ok) "
                              "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,1)",
                              (fid, match["league"], match["home"], match["away"], market, suggestion,
                               conf*100.0, conf, int(time.time()), odds, book, ev_pct))
                    saved += 1
            c.commit()
    except Exception as e:
        log.exception("[SCAN] failed: %s", e)
    return saved, live_seen

def prematch_scan_save():
    """Save prematch candidates into DB (no Telegram)."""
    saved = 0
    try:
        with db_conn() as c:
            rows = c.execute("SELECT fixture_id, league_name, home, away, kickoff "
                             "FROM fixtures WHERE status IN ('NS','TBD')").fetchall()
            for r in rows:
                fid = r[0]
                match = {"fid": fid, "league": r[1], "home": r[2], "away": r[3], "kickoff": r[4]}
                # Example suggestion
                market, suggestion, conf = "Over/Under 2.5", "Over 2.5 Goals", 0.78
                ok, odds, book, ev_pct = price_gate(market, suggestion, fid, prob=conf)
                if ok and conf >= CONF_MIN and (ev_pct or 0) >= EV_MIN:
                    c.execute("INSERT INTO tips(match_id, league, home, away, market, suggestion, confidence, confidence_raw, created_ts, odds, book, ev_pct, sent_ok) "
                              "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,1)",
                              (fid, match["league"], match["home"], match["away"], market, suggestion,
                               conf*100.0, conf, int(time.time()), odds, book, ev_pct))
                    saved += 1
            c.commit()
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
                "SELECT t.suggestion, t.odds, r.final_goals_h+r.final_goals_a AS goals "
                "FROM tips t JOIN match_results r ON t.match_id=r.match_id "
                "WHERE to_timestamp(t.created_ts) AT TIME ZONE 'Europe/Berlin'::text::date=%s::date", (yesterday,)
            ).fetchall()
        bets = len(rows)
        if bets == 0: return None
        wins = sum(1 for sug, odds, g in rows if (("Over" in sug and g>=3) or ("Under" in sug and g<=2)))
        roi = sum((float(odds)-1) if (("Over" in sug and g>=3) or ("Under" in sug and g<=2)) else -1.0 for sug, odds, g in rows)
        hit = wins/bets*100
        msg = f"📊 Digest {yesterday} — {bets} bets | Hit {hit:.1f}% | ROI {roi:+.2f}u"
        send_telegram(msg)
    except Exception as e:
        log.exception("[DIGEST] failed: %s", e)
    return msg

# ───────── MOTD ─────────
def send_match_of_the_day():
    """Pick best prematch tip for today."""
    today = _now().astimezone(BERLIN_TZ).date()
    candidates = []
    try:
        with db_conn() as c:
            rows = c.execute("SELECT fixture_id, league_name, home, away, kickoff "
                             "FROM fixtures WHERE status IN ('NS','TBD')").fetchall()
            for r in rows:
                fid = r[0]
                match = {"fid": fid, "league": r[1], "home": r[2], "away": r[3], "kickoff": r[4]}
                market, suggestion, conf = "Over/Under 2.5", "Over 2.5 Goals", 0.80
                ok, odds, book, ev_pct = price_gate(market, suggestion, fid, prob=conf)
                if ok and conf >= MOTD_CONF_MIN and (ev_pct or 0) >= MOTD_EV_MIN:
                    score = conf + (ev_pct or 0)/100.0
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

    for (
        mid, league, home, away, market, sugg,
        conf_pct, conf_raw, score, minute, cts,
        odds, book, ev_pct
    ) in rows:
        pct = float(conf_pct if conf_pct is not None else (100.0 * float(conf_raw or 0.0)))
        msg = f"♻️ RETRY\n{league}: {home} vs {away}\nTip: {sugg}\nConf: {pct:.1f}%\nOdds: {odds or '-'}"
        ok = send_telegram(msg)
        if ok:
            with db_conn() as c2:
                c2.execute("UPDATE tips SET sent_ok=1 WHERE match_id=%s AND created_ts=%s", (mid, cts))
            retried += 1

    if retried:
        log.info("[RETRY] resent %d", retried)
    return retried

# ───────── Backfill ─────────
def backfill_results_for_open_matches(limit=300):
    """Fetch results for fixtures still open and update DB."""
    log.info("[BACKFILL] placeholder run (limit=%s)", limit)
    return 0
