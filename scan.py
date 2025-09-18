# file: scan.py
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
                    c.execute("INSERT INTO tips(fixture_id, market, suggestion, confidence, odds, ev_pct, sent_at) "
                              "VALUES (%s,%s,%s,%s,%s,%s,now())",
                              (fid, market, suggestion, conf, odds, ev_pct))
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
                    c.execute("INSERT INTO tips(fixture_id, market, suggestion, confidence, odds, ev_pct, sent_at) "
                              "VALUES (%s,%s,%s,%s,%s,%s,now())",
                              (fid, market, suggestion, conf, odds, ev_pct))
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
                "SELECT t.suggestion, t.odds, r.goals_home+r.goals_away AS goals "
                "FROM tips t JOIN results r ON t.fixture_id=r.fixture_id "
                "WHERE (t.sent_at AT TIME ZONE 'Europe/Berlin')::date=%s::date", (yesterday,)
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
def retry_unsent_tips(limit=30, batch=200):
    """Retry tips not sent to Telegram."""
    retried = 0
    try:
        with db_conn() as c:
            rows = c.execute("SELECT id, fixture_id, market, suggestion, confidence, odds, ev_pct "
                             "FROM tips WHERE sent_at IS NULL ORDER BY id ASC LIMIT %s", (batch,)).fetchall()
            for r in rows:
                id, fid, market, suggestion, conf, odds, ev_pct = r
                match = {"fid": fid, "league": "", "home": "?", "away": "?", "kickoff": _now()}
                msg = _fmt_tip_message(match, market, suggestion, conf, odds, "retry", ev_pct)
                send_telegram(msg)
                c.execute("UPDATE tips SET sent_at=now() WHERE id=%s", (id,))
                retried += 1
            c.commit()
    except Exception as e:
        log.exception("[RETRY] failed: %s", e)
    return retried

# ───────── Backfill ─────────
def backfill_results_for_open_matches(limit=300):
    """Fetch results for fixtures still open and update DB."""
    # (placeholder — would query API-football /fixtures and update results table)
    log.info("[BACKFILL] placeholder run (limit=%s)", limit)
    return 0
