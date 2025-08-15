import os
import json
import time
import logging
import sqlite3
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple

import requests
from flask import Flask, jsonify, request, abort
from requests.adapters import HTTPAdapter, Retry
from apscheduler.schedulers.background import BackgroundScheduler

# ──────────────────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
app = Flask(__name__)

# Env configuration (no placeholders)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
API_KEY = os.getenv("API_KEY")  # used to protect /feedback
BET_URL = os.getenv("BET_URL", "https://example.com/bet")
WATCH_URL = os.getenv("WATCH_URL", "https://example.com/watch")

FOOTBALL_API_URL = "https://v3.football.api-sports.io/fixtures"
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}" if TELEGRAM_BOT_TOKEN else None
HEADERS = {"x-apisports-key": API_KEY, "Accept": "application/json"} if API_KEY else {"Accept": "application/json"}
DB_PATH = os.getenv("DB_PATH", "tip_performance.db")

session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))

# ──────────────────────────────────────────────────────────────────────────────
# DB
# ──────────────────────────────────────────────────────────────────────────────
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        # Tips table stores every tip sent (features included)
        c.execute("""
            CREATE TABLE IF NOT EXISTS tips (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id TEXT,
                rule TEXT,
                sent_at TEXT,
                minute INTEGER,
                home TEXT,
                away TEXT,
                score_home INTEGER,
                score_away INTEGER,
                xg_home REAL,
                xg_away REAL,
                shots_home INTEGER,
                shots_away INTEGER,
                corners_home INTEGER,
                corners_away INTEGER,
                poss_home INTEGER,
                poss_away INTEGER,
                won INTEGER,         -- null until feedback arrives (1/0)
                UNIQUE(match_id, rule, minute) ON CONFLICT IGNORE
            )
        """)
        # Online learner state (simple Beta-Bernoulli per rule + thresholds)
        c.execute("""
            CREATE TABLE IF NOT EXISTS model_state (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        # Initialize defaults if not present
        defaults = {
            "pressure_threshold": "5",
            "poss_threshold": "60",
            "over_xg_trigger": "2.5",
            # Thompson/Beta params per rule
            "beta_pressure_a": "1", "beta_pressure_b": "1",
            "beta_over_a": "1", "beta_over_b": "1"
        }
        for k, v in defaults.items():
            c.execute("INSERT OR IGNORE INTO model_state(key, value) VALUES(?,?)", (k, v))
        conn.commit()

def _get_state(keys: List[str]) -> Dict[str, float]:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        q = "SELECT key, value FROM model_state WHERE key IN ({})".format(",".join("?"*len(keys)))
        cur.execute(q, keys)
        out = {k: float(v) for k, v in cur.fetchall()}
    return out

def _set_state(updates: Dict[str, float]) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.executemany(
            "INSERT INTO model_state(key, value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            [(k, str(v)) for k, v in updates.items()]
        )
        conn.commit()

def log_tip(row: Dict[str, Any]) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT OR IGNORE INTO tips
            (match_id, rule, sent_at, minute, home, away,
             score_home, score_away, xg_home, xg_away,
             shots_home, shots_away, corners_home, corners_away,
             poss_home, poss_away, won)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,NULL)
        """, (
            row["match_id"], row["rule"], row["sent_at"], row["minute"], row["home"], row["away"],
            row["score_home"], row["score_away"], row["xg_home"], row["xg_away"],
            row["shots_home"], row["shots_away"], row["corners_home"], row["corners_away"],
            row["poss_home"], row["poss_away"]
        ))
        conn.commit()

def record_feedback(match_id: str, rule: str, minute: int, won: bool) -> bool:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE tips SET won=? WHERE match_id=? AND rule=? AND minute=?",
            (1 if won else 0, match_id, rule, minute)
        )
        conn.commit()
        return cur.rowcount > 0

# ──────────────────────────────────────────────────────────────────────────────
# Messaging
# ──────────────────────────────────────────────────────────────────────────────
def send_telegram(message: str) -> bool:
    if not (TELEGRAM_CHAT_ID and TELEGRAM_API_URL):
        logging.error("Missing Telegram credentials")
        return False

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "reply_markup": json.dumps({
            "inline_keyboard": [
                [{"text": "💰 Bet Now", "url": BET_URL}],
                [{"text": "📺 Watch Match", "url": WATCH_URL}]
            ]
        })
    }

    try:
        res = session.post(f"{TELEGRAM_API_URL}/sendMessage", data=payload, timeout=10)
        if not res.ok:
            logging.error(f"[Telegram] Failed: {res.status_code} - {res.text}")
        return res.ok
    except Exception as e:
        logging.error(f"[Telegram] Exception: {e}")
        return False

# ──────────────────────────────────────────────────────────────────────────────
# Data fetching
# ──────────────────────────────────────────────────────────────────────────────
def fetch_match_stats(fixture_id: int) -> Optional[List[Dict[str, Any]]]:
    try:
        res = session.get(
            "https://v3.football.api-sports.io/fixtures/statistics",
            headers=HEADERS,
            params={"fixture": fixture_id},
            timeout=10
        )
        res.raise_for_status()
        return res.json().get("response", [])
    except Exception as e:
        logging.warning(f"[SKIP] Stats fetch failed for {fixture_id}: {e}")
        return None

def fetch_live_matches() -> List[Dict[str, Any]]:
    try:
        res = session.get(FOOTBALL_API_URL, headers=HEADERS, params={"live": "all"}, timeout=10)
        res.raise_for_status()
        matches = res.json().get("response", [])
        filtered = []
        for m in matches:
            elapsed = m.get("fixture", {}).get("status", {}).get("elapsed")
            if elapsed is None or elapsed > 90 or elapsed < 5:
                continue
            fid = m.get("fixture", {}).get("id")
            m["statistics"] = fetch_match_stats(fid) or []
            if m["statistics"]:
                filtered.append(m)
        return filtered
    except Exception as e:
        logging.error(f"API error: {e}")
        return []

# ──────────────────────────────────────────────────────────────────────────────
# Feature extraction & online rules
# ──────────────────────────────────────────────────────────────────────────────
def _coerce_int(val, default=0) -> int:
    try:
        if isinstance(val, str) and val.endswith("%"):
            val = val[:-1]
        return int(float(val))
    except Exception:
        return default

def _coerce_float(val, default=0.0) -> float:
    try:
        return float(val)
    except Exception:
        return default

def extract_features(match: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Returns (team_features, meta) where team_features[team] = per-team stats."""
    fid = match["fixture"]["id"]
    home = match["teams"]["home"]["name"]
    away = match["teams"]["away"]["name"]
    score_home = match["goals"]["home"] or 0
    score_away = match["goals"]["away"] or 0
    minute = match["fixture"]["status"]["elapsed"] or 0
    stats = match["statistics"]

    data = {s["team"]["name"]: {i["type"]: i["value"] for i in s["statistics"]} for s in stats}
    if home not in data or away not in data:
        return {}, {}

    s_home = data[home]; s_away = data[away]
    feats = {
        home: {
            "xg": _coerce_float(s_home.get("Expected Goals", 0)),
            "shots": _coerce_int(s_home.get("Shots on Target", 0)),
            "corners": _coerce_int(s_home.get("Corner Kicks", 0)),
            "poss": _coerce_int(s_home.get("Ball Possession", 0)),
        },
        away: {
            "xg": _coerce_float(s_away.get("Expected Goals", 0)),
            "shots": _coerce_int(s_away.get("Shots on Target", 0)),
            "corners": _coerce_int(s_away.get("Corner Kicks", 0)),
            "poss": _coerce_int(s_away.get("Ball Possession", 0)),
        }
    }
    meta = {
        "match_id": fid,
        "home": home,
        "away": away,
        "minute": minute,
        "score_home": score_home,
        "score_away": score_away
    }
    return feats, meta

def get_thresholds() -> Dict[str, float]:
    st = _get_state(["pressure_threshold", "poss_threshold", "over_xg_trigger"])
    return {
        "pressure": int(st.get("pressure_threshold", 5)),
        "poss": int(st.get("poss_threshold", 60)),
        "over_xg": float(st.get("over_xg_trigger", 2.5)),
    }

def update_online(rule: str, won: bool):
    """Simple Beta-Bernoulli update per rule + gentle threshold nudging."""
    keys = ("beta_pressure_a", "beta_pressure_b") if rule == "pressure" else ("beta_over_a", "beta_over_b")
    st = _get_state(list(keys) + ["pressure_threshold", "poss_threshold", "over_xg_trigger"])
    a, b = st.get(keys[0], 1.0), st.get(keys[1], 1.0)
    a = a + (1 if won else 0); b = b + (0 if won else 1)

    updates = {keys[0]: a, keys[1]: b}

    # Nudge thresholds: if losing 3 of last 4 for a rule, tighten a bit
    # (this is intentionally conservative; we’re online-learning without extra deps)
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("SELECT won FROM tips WHERE rule=? ORDER BY id DESC LIMIT 4", (rule,))
        rows = [r[0] for r in cur.fetchall()]
    if len(rows) >= 4 and sum(1 for r in rows if r == 0) >= 3:
        if rule == "pressure":
            updates["pressure_threshold"] = max(3, st.get("pressure_threshold", 5) + 1)
            updates["poss_threshold"] = min(80, st.get("poss_threshold", 60) + 2)
        else:
            updates["over_xg_trigger"] = min(4.0, st.get("over_xg_trigger", 2.5) + 0.2)
    _set_state(updates)

def format_tip(meta: Dict[str, Any], rule: str, line: str) -> str:
    return (
        f"⚽️ {meta['home']} vs {meta['away']}\n"
        f"⏱️ {meta['minute']}'  🔢 {meta['score_home']}–{meta['score_away']}\n\n"
        f"{line}\n\n"
        f"🤖 Rule: {rule} • Adaptive"
    )

def maybe_rules(team: str, feats: Dict[str, Any], other: Dict[str, Any], thresholds: Dict[str, float], total_xg: float) -> List[Tuple[str, str]]:
    rules = []
    pressure_hit = feats["shots"] >= thresholds["pressure"] or feats["corners"] >= thresholds["pressure"] or feats["poss"] >= thresholds["poss"]
    if pressure_hit:
        rules.append((
            "pressure",
            f"📈 High attacking pressure by <b>{team}</b>\n"
            f"📊 Possession {feats['poss']}%, corners {feats['corners']}, shots on target {feats['shots']}"
        ))
    if total_xg >= thresholds["over_xg"]:
        rules.append((
            "over",
            f"🔥 Elevated chance of another goal soon\n"
            f"🧮 xG total {total_xg:.2f} (H {feats['xg']:.2f} / A {other['xg']:.2f})"
        ))
    return rules

def generate_tips(match: Dict[str, Any]) -> List[Dict[str, Any]]:
    feats_by_team, meta = extract_features(match)
    if not feats_by_team:
        return []

    home = meta["home"]; away = meta["away"]
    fh, fa = feats_by_team[home], feats_by_team[away]
    thresholds = get_thresholds()
    total_xg = (fh["xg"] + fa["xg"])

    tips: List[Dict[str, Any]] = []
    for team, feats, other in [(home, fh, fa), (away, fa, fh)]:
        for rule, line in maybe_rules(team, feats, other, thresholds, total_xg):
            tips.append({
                "rule": rule,
                "message": format_tip(meta, rule, line),
                "features": {
                    "xg_home": fh["xg"], "xg_away": fa["xg"],
                    "shots_home": fh["shots"], "shots_away": fa["shots"],
                    "corners_home": fh["corners"], "corners_away": fa["corners"],
                    "poss_home": fh["poss"], "poss_away": fa["poss"],
                },
                "meta": meta
            })
    if not tips:
        tips.append({
            "rule": "balanced",
            "message": format_tip(meta, "balanced", "📌 Stats suggest a balanced game in progress."),
            "features": {
                "xg_home": fh["xg"], "xg_away": fa["xg"],
                "shots_home": fh["shots"], "shots_away": fa["shots"],
                "corners_home": fh["corners"], "corners_away": fa["corners"],
                "poss_home": fh["poss"], "poss_away": fa["poss"],
            },
            "meta": meta
        })
    return tips

# ──────────────────────────────────────────────────────────────────────────────
# Orchestration
# ──────────────────────────────────────────────────────────────────────────────
last_heartbeat = 0

def maybe_send_heartbeat():
    global last_heartbeat
    if time.time() - last_heartbeat > 1200:  # ~20 mins
        send_telegram("✅ Robi Superbrain online — adaptive engine learning from feedback.")
        last_heartbeat = time.time()

def match_alert():
    logging.info("🔍 Scanning live matches...")
    matches = fetch_live_matches()
    logging.info(f"[SCAN] {len(matches)} matches after filtering")

    sent = 0
    for match in matches:
        for tip in generate_tips(match):
            meta = tip["meta"]; feats = tip["features"]; rule = tip["rule"]
            # De-dup per match/rule/minute (UNIQUE constraint in DB)
            row = {
                "match_id": str(meta["match_id"]),
                "rule": rule,
                "sent_at": datetime.now(timezone.utc).isoformat(),
                "minute": int(meta["minute"]),
                "home": meta["home"],
                "away": meta["away"],
                "score_home": int(meta["score_home"]),
                "score_away": int(meta["score_away"]),
                **feats
            }
            log_tip(row)
            if send_telegram(tip["message"]):
                sent += 1

    logging.info(f"[TIP] Sent {sent} tips")
    maybe_send_heartbeat()

# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return "🤖 Robi Superbrain is active, adaptive, and watching the game."

@app.route("/match-alert")
def manual_match_alert():
    match_alert()
    return jsonify({"status": "ok"})

@app.route("/feedback", methods=["POST"])
def feedback():
    # Protect with API_KEY (send as header X-API-Key or ?key=)
    key = request.headers.get("X-API-Key") or request.args.get("key")
    if not API_KEY or key != API_KEY:
        abort(401)

    data = request.get_json(force=True, silent=True) or {}
    # expected: {match_id, rule, minute, won: true/false}
    try:
        match_id = str(data["match_id"])
        rule = str(data["rule"])
        minute = int(data["minute"])
        won = bool(data["won"])
    except Exception:
        abort(400, "Provide match_id, rule, minute, won")

    ok = record_feedback(match_id, rule, minute, won)
    if ok:
        # update online learner for rules we adapt (ignore 'balanced')
        if rule in ("pressure", "over"):
            update_online("pressure" if rule == "pressure" else "over", won)
        return jsonify({"updated": True})
    return jsonify({"updated": False})

# ──────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    scheduler = BackgroundScheduler()
    scheduler.add_job(match_alert, "interval", minutes=5)
    scheduler.start()
    port = int(os.getenv("PORT", 5000))
    logging.info("✅ Robi Superbrain (adaptive) started — scanning every 5 minutes.")
    app.run(host="0.0.0.0", port=port)
