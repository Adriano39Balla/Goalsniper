import os
import json
import time
import logging
import requests
import sqlite3
from html import escape
from datetime import datetime, timezone, date
from flask import Flask, jsonify, request, abort
from typing import List, Dict, Any, Optional, Tuple
from requests.adapters import HTTPAdapter, Retry
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

app = Flask(__name__)

# ── Env ───────────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")

# API keys
API_KEY        = os.getenv("API_KEY")                # admin endpoints
APISPORTS_KEY  = os.getenv("APISPORTS_KEY", API_KEY) # used for API-Football

PUBLIC_BASE_URL    = os.getenv("PUBLIC_BASE_URL")
BET_URL_TMPL       = os.getenv("BET_URL")
WATCH_URL_TMPL     = os.getenv("WATCH_URL")
WEBHOOK_SECRET     = os.getenv("TELEGRAM_WEBHOOK_SECRET")

# Knobs
CONF_THRESHOLD     = int(os.getenv("CONF_THRESHOLD", "55"))     # send only if >= 55%
MAX_TIPS_PER_SCAN  = int(os.getenv("MAX_TIPS_PER_SCAN", "8"))   # soft cap per scan (set 0 to disable)
DUP_COOLDOWN_MIN   = int(os.getenv("DUP_COOLDOWN_MIN", "20"))   # don't resend same fixture within N min

# League priority for Match-of-the-Day (can override via env)
LEAGUE_PRIORITY_IDS = [
    int(x) for x in (os.getenv("MOTD_LEAGUE_IDS", "39,140,135,78,61,2").split(","))
    if x.strip().isdigit()
]

# Allow these final suggestion texts (includes BTTS)
ALLOWED_SUGGESTIONS = {
    "Over 1.5 Goals", "Over 2.5 Goals", "Under 2.5 Goals", "BTTS: Yes", "BTTS: No"
}

# ── External APIs ─────────────────────────────────────────────────────────────
FOOTBALL_API_URL = "https://v3.football.api-sports.io/fixtures"
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
HEADERS          = {"x-apisports-key": APISPORTS_KEY, "Accept": "application/json"}

# ── DB ────────────────────────────────────────────────────────────────────────
DB_PATH = "tip_performance.db"

# ── HTTP session ──────────────────────────────────────────────────────────────
session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))

# ───────────────────────────── DB helpers ─────────────────────────────────────
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tips (
                match_id INTEGER,
                league_id INTEGER,
                league   TEXT,
                home     TEXT,
                away     TEXT,
                market   TEXT,
                suggestion TEXT,
                confidence REAL,
                score_at_tip TEXT,
                minute    INTEGER,
                created_ts INTEGER,
                PRIMARY KEY (match_id, created_ts)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id INTEGER,
                verdict  INTEGER CHECK (verdict IN (0,1)),
                created_ts INTEGER
            )
        """)
        conn.execute("""
            CREATE VIEW IF NOT EXISTS v_tip_stats AS
            SELECT market, suggestion, AVG(verdict) AS hit_rate, COUNT(*) AS n
            FROM tips t JOIN feedback f ON f.match_id = t.match_id
            GROUP BY market, suggestion
        """)
        conn.commit()

def save_tip(match_id: int, league_id: int, league: str, home: str, away: str,
             market: str, suggestion: str, confidence: float,
             score_at_tip: str, minute: int):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO tips(match_id,league_id,league,home,away,market,suggestion,confidence,score_at_tip,minute,created_ts)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, (match_id, league_id, league, home, away, market, suggestion, confidence, score_at_tip, minute, int(time.time())))
            conn.commit()
    except sqlite3.OperationalError:
        logging.exception("[DB] save_tip failed — attempting init and retry")
        init_db()
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO tips(match_id,league_id,league,home,away,market,suggestion,confidence,score_at_tip,minute,created_ts)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, (match_id, league_id, league, home, away, market, suggestion, confidence, score_at_tip, minute, int(time.time())))
            conn.commit()

def recent_tip_exists(match_id: int, cooldown_min: int) -> bool:
    cutoff = int(time.time()) - cooldown_min * 60
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("SELECT 1 FROM tips WHERE match_id=? AND created_ts>=? LIMIT 1", (match_id, cutoff))
        return cur.fetchone() is not None

def record_feedback(match_id: int, verdict: int):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("INSERT INTO feedback(match_id, verdict, created_ts) VALUES (?,?,?)",
                     (match_id, verdict, int(time.time())))
        conn.commit()

def get_historic_hit_rate(market: str, suggestion: str) -> Tuple[float, int]:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "SELECT COALESCE(hit_rate,0.5), COALESCE(n,0) FROM v_tip_stats WHERE market=? AND suggestion=?",
            (market, suggestion),
        )
        row = cur.fetchone()
        return (row[0], row[1]) if row else (0.5, 0)

# ─────────────────────────── Telegram helpers ─────────────────────────────────
def send_telegram(message: str, inline_keyboard: Optional[list] = None) -> bool:
    if not TELEGRAM_BOT_TOKEN:
        logging.error("Missing TELEGRAM_BOT_TOKEN")
        return False
    if not TELEGRAM_CHAT_ID:
        logging.error("Missing TELEGRAM_CHAT_ID")
        return False
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
    }
    if inline_keyboard:
        payload["reply_markup"] = json.dumps({"inline_keyboard": inline_keyboard})
    url = f"{TELEGRAM_API_URL}/sendMessage"
    try:
        res = session.post(url, data=payload, timeout=10)
        if not res.ok:
            logging.error(f"[Telegram] sendMessage FAILED status={res.status_code} body={res.text}")
        else:
            logging.info("[Telegram] sendMessage OK")
        return res.ok
    except Exception as e:
        logging.exception(f"[Telegram] Exception during sendMessage: {e}")
        return False

def answer_callback(callback_id: str, text: str):
    try:
        session.post(f"{TELEGRAM_API_URL}/answerCallbackQuery",
                     data={"callback_query_id": callback_id, "text": text, "show_alert": False}, timeout=10)
    except Exception:
        pass

# ───────────────────────── Football API ───────────────────────────────────────
def fetch_match_stats(fixture_id: int) -> Optional[List[Dict[str, Any]]]:
    try:
        res = session.get(
            f"{FOOTBALL_API_URL}/statistics",
            headers=HEADERS,
            params={"fixture": fixture_id},
            timeout=10
        )
        if not res.ok:
            logging.warning(f"[API] stats fixture={fixture_id} status={res.status_code} body={res.text[:200]}")
            return None
        return res.json().get("response", [])
    except Exception as e:
        logging.warning(f"[SKIP] Stats fetch failed for {fixture_id}: {e}")
        return None

def fetch_live_matches() -> List[Dict[str, Any]]:
    try:
        res = session.get(FOOTBALL_API_URL, headers=HEADERS, params={"live": "all"}, timeout=10)
        if not res.ok:
            logging.error(f"[API] fixtures status={res.status_code} body={res.text[:300]}")
            return []
        data = res.json()
        matches = data.get("response", []) if isinstance(data, dict) else []
        filtered = []
        for m in matches:
            status = (m.get("fixture", {}) or {}).get("status", {}) or {}
            elapsed = status.get("elapsed")
            if elapsed is None or elapsed > 90:
                continue
            # Try stats (non-blocking)
            fid = (m.get("fixture", {}) or {}).get("id")
            try:
                stats = fetch_match_stats(fid)
                m["statistics"] = stats or []
            except Exception:
                m["statistics"] = []
            filtered.append(m)
        logging.info(f"[FETCH] live={len(matches)} kept={len(filtered)}")
        return filtered
    except Exception as e:
        logging.exception("API error while fetching live fixtures")
        return []

def fetch_today_fixtures_utc() -> List[Dict[str, Any]]:
    """Today's fixtures (UTC). Useful for 08:00 UTC 'Match of the Day'."""
    try:
        today = date.today().isoformat()  # UTC on Railway
        res = session.get(FOOTBALL_API_URL, headers=HEADERS, params={"date": today, "timezone": "UTC"}, timeout=10)
        if not res.ok:
            logging.error(f"[API] today fixtures status={res.status_code} body={res.text[:300]}")
            return []
        data = res.json()
        return data.get("response", []) if isinstance(data, dict) else []
    except Exception as e:
        logging.exception("API error while fetching today's fixtures")
        return []

# ───────────────────────── Tip logic ──────────────────────────────────────────
def _num(v) -> float:
    try:
        if isinstance(v, str) and v.endswith('%'):
            return float(v[:-1])
        return float(v or 0)
    except Exception:
        return 0.0

def decide_market(match: Dict[str, Any]) -> Tuple[str, str, float, Dict[str, float]]:
    """
    Returns (market, suggestion, confidence, statline)
    Allowed suggestions include: Over 1.5, Over 2.5, Under 2.5, BTTS: Yes, BTTS: No
    """
    home = match["teams"]["home"]["name"]
    away = match["teams"]["away"]["name"]
    gh = match["goals"]["home"] or 0
    ga = match["goals"]["away"] or 0
    minute = int((match.get("fixture", {}).get("status", {}) or {}).get("elapsed") or 0)

    # Build stats (may be missing)
    stats_blocks = match.get("statistics") or []
    stats: Dict[str, Dict[str, Any]] = {}
    for s in stats_blocks:
        tname = (s.get("team") or {}).get("name")
        if not tname:
            continue
        stats[tname] = {i["type"]: i["value"] for i in (s.get("statistics") or [])}

    # Defaults for statline
    sh = stats.get(home, {}); sa = stats.get(away, {})
    xg_h = _num(sh.get("Expected Goals", 0)); xg_a = _num(sa.get("Expected Goals", 0))
    sot_h = _num(sh.get("Shots on Target", 0)); sot_a = _num(sa.get("Shots on Target", 0))
    cor_h = _num(sh.get("Corner Kicks", 0));   cor_a = _num(sa.get("Corner Kicks", 0))

    total_goals = gh + ga

    # Fallbacks when no detailed stats
    if not stats:
        if minute >= 70 and total_goals <= 1:
            return "Over/Under 2.5 Goals", "Under 2.5 Goals", 60, {"xg_h":0,"xg_a":0,"sot_h":0,"sot_a":0,"cor_h":0,"cor_a":0}
        if minute >= 30 and total_goals == 0:
            return "Over/Under 2.5 Goals", "Over 1.5 Goals", 58, {"xg_h":0,"xg_a":0,"sot_h":0,"sot_a":0,"cor_h":0,"cor_a":0}
        return "Over/Under 2.5 Goals", "Over 1.5 Goals", 55, {"xg_h":0,"xg_a":0,"sot_h":0,"sot_a":0,"cor_h":0,"cor_a":0}

    xg_sum   = xg_h + xg_a
    sot_sum  = sot_h + sot_a
    corners  = cor_h + cor_a

    # Start with Over 2.5 as default
    market, suggestion, score = "Over/Under 2.5 Goals", "Over 2.5 Goals", 50.0

    if xg_sum >= 2.4 and total_goals <= 2 and minute >= 25:
        score += 18
    if sot_sum >= 6:
        score += 10
    if corners >= 10:
        score += 5
    if minute < 20:
        score -= 8

    # BTTS conditions
    if minute >= 25 and xg_h >= 0.9 and xg_a >= 0.9 and sot_h >= 3 and sot_a >= 3 and total_goals <= 3:
        market, suggestion, score = "BTTS", "Yes", max(score, 62)
    # Defensive vibe → BTTS: No (or Under)
    if xg_sum < 1.4 and sot_sum < 4 and total_goals == 0 and minute > 35:
        market, suggestion, score = "BTTS", "No", 62

    # Late & still low → Under 2.5
    if minute > 70 and total_goals <= 1 and suggestion != "BTTS: Yes":
        market, suggestion = "Over/Under 2.5 Goals", "Under 2.5 Goals"
        score = max(score, 62)

    # High tempo but still low score → Over 1.5 safer angle
    if minute >= 30 and total_goals == 0 and (xg_sum >= 1.6 or sot_sum >= 6) and suggestion not in ("BTTS: Yes", "Under 2.5 Goals"):
        market, suggestion, score = "Over/Under 2.5 Goals", "Over 1.5 Goals", max(score, 60)

    # Clamp and history bump
    score = max(35.0, min(90.0, score))
    hist_rate, n = get_historic_hit_rate(market, suggestion)
    bump = (hist_rate - 0.5) * 14.0 * min(1.0, n / 50.0)  # cap ±7%
    confidence = round(max(35.0, min(95.0, score + bump)))

    statline = {"xg_h": xg_h, "xg_a": xg_a, "sot_h": sot_h, "sot_a": sot_a, "cor_h": cor_h, "cor_a": cor_a}
    return market, suggestion, confidence, statline

def format_human_tip(match: Dict[str, Any],
                     chosen: Optional[Tuple[str,str,float,Dict[str,float]]] = None) -> Tuple[str, list, Dict[str, Any]]:
    fixture = match["fixture"]
    league_block = match.get("league") or {}
    league_id = league_block.get("id") or 0
    league = escape(f"{league_block.get('country','')} - {league_block.get('name','')}".strip(" -"))
    minute = fixture["status"].get("elapsed") or 0
    match_id = fixture["id"]
    home = escape(match["teams"]["home"]["name"])
    away = escape(match["teams"]["away"]["name"])
    g_home = match["goals"]["home"] or 0
    g_away = match["goals"]["away"] or 0

    market, suggestion, confidence, stat = chosen if chosen else decide_market(match)

    # Gate: allowed suggestion + confidence threshold
    if suggestion not in ALLOWED_SUGGESTIONS or confidence < CONF_THRESHOLD:
        return "", [], {"match_id": match_id, "skip": True, "confidence": confidence}

    # Persist
    save_tip(
        match_id=match_id,
        league_id=int(league_id),
        league=league,
        home=home,
        away=away,
        market=market,
        suggestion=suggestion,
        confidence=float(confidence),
        score_at_tip=f"{g_home}-{g_away}",
        minute=int(minute),
    )

    # Stats line (only if present)
    stat_line = ""
    if any(stat.values()):
        stat_line = f"\n📊 xG H {stat['xg_h']:.2f} / A {stat['xg_a']:.2f} • SOT {int(stat['sot_h'])}–{int(stat['sot_a'])} • CK {int(stat['cor_h'])}–{int(stat['cor_a'])}"

    # ⏱ Minute + live score added here
    lines = [
        "⚽️ <b>New Tip!</b>",
        f"<b>Match:</b> {home} vs {away}",
        f"⏱ <b>Minute:</b> {minute}'  |  <b>Score:</b> {g_home}–{g_away}",
        f"<b>Tip:</b> {escape(suggestion)}",
        f"📈 <b>Confidence:</b> {int(confidence)}%",
        f"🏆 <b>League:</b> {league}{stat_line}",
    ]
    msg = "\n".join(lines)

    # Buttons
    kb = [[
        {"text": "👍 Correct", "callback_data": json.dumps({"t": "correct", "id": match_id})},
        {"text": "👎 Wrong",   "callback_data": json.dumps({"t": "wrong",   "id": match_id})},
    ]]
    link_row = []
    if BET_URL_TMPL:
        try:
            link_row.append({"text": "💰 Bet Now", "url": BET_URL_TMPL.format(home=home, away=away, fixture_id=match_id)})
        except Exception: pass
    if WATCH_URL_TMPL:
        try:
            link_row.append({"text": "📺 Watch Match", "url": WATCH_URL_TMPL.format(home=home, away=away, fixture_id=match_id)})
        except Exception: pass
    if link_row: kb.append(link_row)

    meta = {"match_id": match_id, "league_id": int(league_id), "confidence": confidence}
    return msg, kb, meta

# ───────────────────────── Scheduler / Orchestration ──────────────────────────
last_heartbeat = 0
def maybe_send_heartbeat():
    global last_heartbeat
    if time.time() - last_heartbeat > 1200:
        send_telegram("✅ Robi Superbrain is online and scanning…")
        last_heartbeat = time.time()

def match_alert():
    logging.info("🔍 Scanning live matches…")
    matches = fetch_live_matches()
    if not matches:
        logging.warning("[SCAN] No matches returned – check APISPORTS_KEY/plan or time of day.")
        return

    sent = 0
    for match in matches:
        if MAX_TIPS_PER_SCAN and sent >= MAX_TIPS_PER_SCAN:
            break

        fid = (match.get("fixture", {}) or {}).get("id")
        if recent_tip_exists(fid, DUP_COOLDOWN_MIN):
            continue

        chosen = decide_market(match)
        msg, kb, meta = format_human_tip(match, chosen)
        if meta.get("skip"):
            continue

        logging.info(f"[TIP] sending fixture={fid} conf={meta['confidence']} suggestion={chosen[1]}")
        if send_telegram(msg, kb):
            sent += 1

    logging.info(f"[TIP] Sent={sent}")
    maybe_send_heartbeat()

def pick_match_of_the_day(fixtures: List[Dict[str,Any]]) -> Optional[Dict[str,Any]]:
    """Choose one fixture for MOTD: prioritize by league id list, otherwise earliest kickoff."""
    pri = {lid:i for i,lid in enumerate(LEAGUE_PRIORITY_IDS)}
    def key_fn(f):
        lg = (f.get("league") or {})
        lid = lg.get("id", 10**9)
        idx = pri.get(lid, len(LEAGUE_PRIORITY_IDS)+1)
        kick = (f.get("fixture") or {}).get("timestamp") or 10**12
        return (idx, kick)
    fixtures = [f for f in fixtures if (f.get("fixture") or {}).get("status",{}).get("short") in ("NS","TBD")]
    if not fixtures:
        return None
    fixtures.sort(key=key_fn)
    return fixtures[0]

def send_match_of_the_day():
    fixtures = fetch_today_fixtures_utc()
    motd = pick_match_of_the_day(fixtures)
    if not motd:
        logging.info("[MOTD] No suitable fixture found for today.")
        return

    lg = motd.get("league") or {}
    fx = motd.get("fixture") or {}
    home = escape((motd.get("teams") or {}).get("home",{}).get("name",""))
    away = escape((motd.get("teams") or {}).get("away",{}).get("name",""))
    league = escape(f"{lg.get('country','')} - {lg.get('name','')}".strip(" -"))
    ts = fx.get("timestamp")
    when = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC") if ts else "TBD"
    fid = fx.get("id")

    lines = [
        "🌟 <b>Match of the Day</b>",
        f"<b>Match:</b> {home} vs {away}",
        f"🗓 <b>Kickoff:</b> {when}",
        f"🏆 <b>League:</b> {league}",
    ]
    msg = "\n".join(lines)

    kb = []
    link_row = []
    if BET_URL_TMPL:
        try:
            link_row.append({"text": "💰 Bet Now", "url": BET_URL_TMPL.format(home=home, away=away, fixture_id=fid)})
        except Exception: pass
    if WATCH_URL_TMPL:
        try:
            link_row.append({"text": "📺 Watch (when live)", "url": WATCH_URL_TMPL.format(home=home, away=away, fixture_id=fid)})
        except Exception: pass
    if link_row: kb.append(link_row)

    send_telegram(msg, kb)
    logging.info(f"[MOTD] Sent for fixture={fid}")

# ───────────────────────── Routes ─────────────────────────────────────────────
@app.route("/")
def home():
    return "🤖 Robi Superbrain is active and learning."

@app.route("/match-alert")
def manual_match_alert():
    match_alert()
    return jsonify({"status": "ok"})

# Manual trigger for MOTD
@app.route("/motd")
def motd_route():
    send_match_of_the_day()
    return jsonify({"status": "ok"})

# Debug routes
@app.route("/debug/scan")
def debug_scan():
    data = fetch_live_matches()
    return jsonify({
        "count": len(data),
        "fixture_ids": [ (m.get("fixture",{}) or {}).get("id") for m in data ],
        "has_stats": [ bool(m.get("statistics")) for m in data ],
    })

@app.route("/debug/format")
def debug_format():
    matches = fetch_live_matches()
    if not matches:
        return jsonify({"error": "no matches"}), 200
    i = 0
    try: i = int(request.args.get("i", 0))
    except Exception: pass
    i = max(0, min(i, len(matches)-1))
    chosen = decide_market(matches[i])
    msg, kb, meta = format_human_tip(matches[i], chosen)
    return jsonify({"fixture_id": meta.get("match_id"), "confidence": meta.get("confidence"), "message": msg, "inline_keyboard": kb})

@app.route("/debug/send")
def debug_send():
    limit = int(request.args.get("n", 3))
    sent = 0
    matches = fetch_live_matches()
    for m in matches[:limit]:
        chosen = decide_market(m)
        msg, kb, meta = format_human_tip(m, chosen)
        if meta.get("skip"):
            continue
        ok = send_telegram(msg, kb)
        logging.info(f"[DEBUG/SEND] fixture={m.get('fixture',{}).get('id')} ok={ok}")
        if ok: sent += 1
    return jsonify({"requested": limit, "sent": sent, "total_matches": len(matches)})

# Telegram webhook for callback buttons
@app.route("/telegram-webhook", methods=["POST"])
def telegram_webhook():
    if WEBHOOK_SECRET and request.headers.get("X-Telegram-Bot-Api-Secret-Token") != WEBHOOK_SECRET:
        return "forbidden", 403
    update = request.get_json(force=True, silent=True) or {}
    cq = update.get("callback_query")
    if not cq:
        return "ok"
    try:
        payload = json.loads(cq.get("data") or "{}")
        verdict = 1 if payload.get("t") == "correct" else 0
        match_id = int(payload.get("id"))
        record_feedback(match_id, verdict)
        answer_callback(cq["id"], "Thanks! Feedback recorded ✅" if verdict else "Got it. Marked as wrong ❌")
    except Exception:
        logging.exception("webhook error")
        try: answer_callback(cq.get("id",""), "Sorry, couldn’t record feedback.")
        except Exception: pass
    return "ok"

# Admin: hit-rate and config (optional)
def _require_api_key():
    key = request.headers.get("X-API-Key") or request.args.get("key")
    if not API_KEY or key != API_KEY:
        abort(401)

@app.route("/stats/hitrate")
def stats_hitrate():
    _require_api_key()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("SELECT market, suggestion, ROUND(hit_rate*100,1), n FROM v_tip_stats ORDER BY n DESC, hit_rate DESC")
        rows = [{"market": r[0], "suggestion": r[1], "hit_rate_pct": r[2], "n": r[3]} for r in cur.fetchall()]
    return jsonify({"rows": rows})

@app.route("/stats/config")
def stats_config():
    _require_api_key()
    return jsonify({
        "CONF_THRESHOLD": CONF_THRESHOLD,
        "MAX_TIPS_PER_SCAN": MAX_TIPS_PER_SCAN,
        "DUP_COOLDOWN_MIN": DUP_COOLDOWN_MIN,
        "MOTD_LEAGUE_IDS": LEAGUE_PRIORITY_IDS,
        "APISPORTS_KEY_set": bool(APISPORTS_KEY),
        "TELEGRAM_CHAT_ID_set": bool(TELEGRAM_CHAT_ID),
    })

# ───────────────────────── Entrypoint ─────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    scheduler = BackgroundScheduler()
    # Live scan every 5 minutes
    scheduler.add_job(match_alert, "interval", minutes=5, id="scan", replace_existing=True)
    # Match of the Day at 08:00 UTC daily
    scheduler.add_job(send_match_of_the_day, CronTrigger(hour=8, minute=0, timezone="UTC"),
                      id="motd", replace_existing=True)
    scheduler.start()
    logging.info("⏱️ Scheduler started (scan=5min, MOTD=08:00 UTC)")
    port = int(os.getenv("PORT", 5000))
    logging.info("✅ Robi Superbrain started.")
    app.run(host="0.0.0.0", port=port)
