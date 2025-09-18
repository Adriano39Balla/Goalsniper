import os, asyncio, contextlib, json, math, pickle, re
from dataclasses import dataclass
from datetime import datetime, timedelta, time as dtime, date
from zoneinfo import ZoneInfo

import numpy as np
import httpx
import psycopg
from psycopg.rows import dict_row

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from aiohttp import web

# ========= ENV / CONFIG =========
TZ = ZoneInfo(os.getenv("TZ", "Europe/Berlin"))
API_KEY = os.getenv("APIFOOTBALL_KEY", "")
BOT = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT = os.getenv("TELEGRAM_CHAT_ID", "")
DB_URL = os.getenv("DATABASE_URL", "")

COOLDOWN_START = os.getenv("COOLDOWN_START", "23:00")
COOLDOWN_END   = os.getenv("COOLDOWN_END", "07:00")
SEND_MESSAGES  = os.getenv("SEND_MESSAGES", "true").lower() == "true"

LOOKBACK_MATCHES = int(os.getenv("LOOKBACK_MATCHES", "10"))
API_BASE = "https://v3.football.api-sports.io"
HEADERS  = {"x-apisports-key": API_KEY}
HTTP_PORT = int(os.getenv("PORT", "8000"))

# ========= UTILITIES =========
def now_local() -> datetime: return datetime.now(TZ)
def parse_hhmm(s: str) -> dtime: h, m = map(int, s.split(":")); return dtime(h, m)
def within_cooldown() -> bool:
    n = now_local().time()
    s, e = parse_hhmm(COOLDOWN_START), parse_hhmm(COOLDOWN_END)
    return (n >= s) or (n < e)
def is_exact_local_time(hhmm: str) -> bool:
    t = parse_hhmm(hhmm); n = now_local()
    return n.hour == t.hour and n.minute == t.minute

def league_excluded(name: str, type_: str) -> bool:
    if type_ not in ("League", "Cup"): return True
    return re.search(r"(U-?\d{2}|U\d{2}|Youth|Reserves?|Academy|U19|U21|U23|B$|II$)", name, re.I) is not None

async def send_telegram(text: str):
    if not (BOT and CHAT and SEND_MESSAGES): return
    url = f"https://api.telegram.org/bot{BOT}/sendMessage"
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.post(url, json={"chat_id": CHAT, "text": text, "parse_mode": "Markdown", "disable_web_page_preview": True})
        r.raise_for_status()

def implied_prob(odds: float) -> float: return 0.0 if not odds or odds <= 1 else 1.0 / odds
def normalize_overround(p_over: float, p_under: float):
    s = max(p_over + p_under, 1e-6); return p_over / s, p_under / s
def expected_value(p: float, dec: float): return p * (dec - 1.0) - (1.0 - p)

def poisson_p_total_ge_k(lmb: float, k: int) -> float:
    cdf = 0.0
    for i in range(0, k):
        cdf += math.exp(-lmb) * lmb**i / math.factorial(i)
    return max(0.0, min(1.0, 1.0 - cdf))

# ========= DB SCHEMA =========
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS fixtures (
  fixture_id INT PRIMARY KEY,
  date DATE,
  league_id INT,
  league_name TEXT,
  season INT,
  home_id INT,
  away_id INT,
  home TEXT,
  away TEXT,
  kickoff TIMESTAMPTZ,
  status TEXT,
  goals_home SMALLINT,
  goals_away SMALLINT
);

CREATE TABLE IF NOT EXISTS odds_snapshots (
  fixture_id INT,
  ts TIMESTAMPTZ DEFAULT now(),
  over25 REAL,
  under25 REAL,
  PRIMARY KEY (fixture_id, ts)
);

CREATE TABLE IF NOT EXISTS features (
  fixture_id INT PRIMARY KEY,
  computed_at TIMESTAMPTZ DEFAULT now(),
  home_form_gf5 REAL, home_form_ga5 REAL,
  away_form_gf5 REAL, away_form_ga5 REAL,
  home_rest_days REAL, away_rest_days REAL,
  league_goal_env REAL,
  odds_over_implied REAL, odds_under_implied REAL,
  odds_overround REAL, odds_over_drift REAL, odds_under_drift REAL
);

CREATE TABLE IF NOT EXISTS predictions (
  id BIGSERIAL PRIMARY KEY,
  fixture_id INT,
  league_name TEXT,
  home TEXT, away TEXT,
  kickoff TIMESTAMPTZ,
  market TEXT,
  pick TEXT,
  p_over REAL, p_under REAL,
  odds_over REAL, odds_under REAL,
  confidence REAL, ev REAL,
  policy_id TEXT DEFAULT 'global',
  explain JSONB,
  sent_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS results (
  fixture_id INT PRIMARY KEY,
  goals_home SMALLINT, goals_away SMALLINT,
  settled_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS model_cfg (
  key TEXT PRIMARY KEY,
  value JSONB
);

CREATE TABLE IF NOT EXISTS model_registry (
  model_id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  created_at TIMESTAMPTZ DEFAULT now(),
  label TEXT, version TEXT, lib TEXT,
  blob BYTEA, meta JSONB
);

CREATE EXTENSION IF NOT EXISTS pgcrypto;
"""

async def ensure_schema():
    async with await psycopg.AsyncConnection.connect(DB_URL) as con:
        await con.execute(SCHEMA_SQL); await con.commit()

# ---- startup migration (idempotent) ----
async def get_cfg(key: str):
    async with await psycopg.AsyncConnection.connect(DB_URL, row_factory=dict_row) as con:
        row = await con.execute("SELECT value FROM model_cfg WHERE key=%s", (key,))
        r = await row.fetchone(); return r["value"] if r else None

async def set_cfg(key: str, value):
    async with await psycopg.AsyncConnection.connect(DB_URL) as con:
        await con.execute("INSERT INTO model_cfg(key,value) VALUES (%s,%s) ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value",
                          (key, json.dumps(value))); await con.commit()

async def run_migrations_once():
    key = "migration:supercomputer_v2"
    if await get_cfg(key): return
    MIGRATION_SQL = """
    CREATE EXTENSION IF NOT EXISTS pgcrypto;

    DO $$
    BEGIN
      IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='fixtures' AND column_name='league_id') THEN
        ALTER TABLE fixtures ADD COLUMN league_id INT;
      END IF;
      IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='fixtures' AND column_name='season') THEN
        ALTER TABLE fixtures ADD COLUMN season INT;
      END IF;
      IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='fixtures' AND column_name='home_id') THEN
        ALTER TABLE fixtures ADD COLUMN home_id INT;
      END IF;
      IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='fixtures' AND column_name='away_id') THEN
        ALTER TABLE fixtures ADD COLUMN away_id INT;
      END IF;
      IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='fixtures' AND column_name='goals_home') THEN
        ALTER TABLE fixtures ADD COLUMN goals_home SMALLINT;
      END IF;
      IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='fixtures' AND column_name='goals_away') THEN
        ALTER TABLE fixtures ADD COLUMN goals_away SMALLINT;
      END IF;
    END$$;

    CREATE TABLE IF NOT EXISTS odds_snapshots (
      fixture_id INT, ts TIMESTAMPTZ DEFAULT now(), over25 REAL, under25 REAL,
      PRIMARY KEY (fixture_id, ts)
    );

    CREATE TABLE IF NOT EXISTS features (
      fixture_id INT PRIMARY KEY,
      computed_at TIMESTAMPTZ DEFAULT now(),
      home_form_gf5 REAL, home_form_ga5 REAL,
      away_form_gf5 REAL, away_form_ga5 REAL,
      home_rest_days REAL, away_rest_days REAL,
      league_goal_env REAL,
      odds_over_implied REAL, odds_under_implied REAL,
      odds_overround REAL, odds_over_drift REAL, odds_under_drift REAL
    );

    CREATE TABLE IF NOT EXISTS results (
      fixture_id INT PRIMARY KEY,
      goals_home SMALLINT, goals_away SMALLINT, settled_at TIMESTAMPTZ
    );

    CREATE TABLE IF NOT EXISTS model_registry (
      model_id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
      created_at TIMESTAMPTZ DEFAULT now(),
      label TEXT, version TEXT, lib TEXT,
      blob BYTEA, meta JSONB
    );

    DO $$
    BEGIN
      IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='predictions' AND column_name='policy_id') THEN
        ALTER TABLE predictions ADD COLUMN policy_id TEXT DEFAULT 'global';
      END IF;
      IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='predictions' AND column_name='explain') THEN
        ALTER TABLE predictions ADD COLUMN explain JSONB;
      END IF;
    END$$;

    DO $$
    BEGIN
      IF NOT EXISTS (SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
                     WHERE c.relname='idx_fixtures_kickoff' AND n.nspname='public') THEN
        CREATE INDEX idx_fixtures_kickoff ON fixtures(kickoff);
      END IF;
      IF NOT EXISTS (SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
                     WHERE c.relname='idx_predictions_sent_at' AND n.nspname='public') THEN
        CREATE INDEX idx_predictions_sent_at ON predictions(sent_at);
      END IF;
    END$$;
    """
    async with await psycopg.AsyncConnection.connect(DB_URL) as con:
        await con.execute(MIGRATION_SQL); await con.commit()
    await set_cfg(key, {"ran_at": now_local().isoformat()})
    print("[MIGRATION] Completed supercomputer_v2 migration.")

# ========= API-FOOTBALL =========
async def api_get(path: str, params: dict):
    async with httpx.AsyncClient(headers=HEADERS, timeout=30) as c:
        r = await c.get(f"{API_BASE}/{path}", params=params); r.raise_for_status()
        return r.json()

async def get_fixtures_by_date(date_iso: str):
    data = await api_get("fixtures", {"date": date_iso})
    out=[]
    for r in data.get("response", []):
        lg = r.get("league", {}) or {}
        if league_excluded(lg.get("name",""), lg.get("type","League")): continue
        out.append(r)
    return out

async def get_team_last_matches(team_id: int, n: int = LOOKBACK_MATCHES):
    data = await api_get("fixtures", {"team": team_id, "last": n})
    return data.get("response", [])

async def get_odds_for_fixture(fixture_id: int):
    """Return best over/under 2.5 odds with bookmaker names."""
    data = await api_get("odds", {"fixture": fixture_id})
    best = {"over": None, "under": None, "over_bm": None, "under_bm": None}
    for resp in data.get("response", []):
        for bk in resp.get("bookmakers", []):
            bk_name = bk.get("name") or "Bookmaker"
            for market in bk.get("bets", []):
                if str(market.get("name","")).lower() in ("over/under","goals over/under"):
                    for v in market.get("values", []):
                        label = (v.get("value") or "").strip()
                        odd = v.get("odd"); 
                        if not odd: continue
                        odd = float(odd)
                        if label in ("Over 2.5","Over 2,5"):
                            if (best["over"] or 0) < odd: best["over"], best["over_bm"] = odd, bk_name
                        elif label in ("Under 2.5","Under 2,5"):
                            if (best["under"] or 0) < odd: best["under"], best["under_bm"] = odd, bk_name
    return best

# ========= UPSERTS =========
async def upsert_fixture_row(f):
    fix=f["fixture"]; teams=f["teams"]; league=f["league"]; goals=f.get("goals") or {}
    async with await psycopg.AsyncConnection.connect(DB_URL) as con:
        await con.execute("""
            INSERT INTO fixtures(
                fixture_id, date, league_id, league_name, season,
                home_id, away_id, home, away,
                kickoff, status, goals_home, goals_away
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (fixture_id) DO UPDATE SET
                date=EXCLUDED.date,
                league_id=EXCLUDED.league_id,
                league_name=EXCLUDED.league_name,
                season=EXCLUDED.season,
                home_id=EXCLUDED.home_id,
                away_id=EXCLUDED.away_id,
                home=EXCLUDED.home,
                away=EXCLUDED.away,
                kickoff=EXCLUDED.kickoff,
                status=EXCLUDED.status,
                goals_home=EXCLUDED.goals_home,
                goals_away=EXCLUDED.goals_away
        """, (
            fix["id"],
            datetime.fromtimestamp(fix["timestamp"], tz=TZ).date(),
            league.get("id"), league.get("name"), league.get("season"),
            teams["home"]["id"], teams["away"]["id"],
            teams["home"]["name"], teams["away"]["name"],
            datetime.fromtimestamp(fix["timestamp"], tz=TZ),
            fix["status"]["short"],
            (goals.get("home") or None), (goals.get("away") or None)
        ))
        await con.commit()

# ========= FEATURES =========
@dataclass
class FixtureInfo:
    fixture_id:int; league:str; kickoff:datetime; home_id:int; away_id:int; home:str; away:str

async def team_recent_stats(team_id:int, limit:int=LOOKBACK_MATCHES):
    # DB-first
    async with await psycopg.AsyncConnection.connect(DB_URL, row_factory=dict_row) as con:
        rows = await con.execute("""
            SELECT goals_home, goals_away, home_id, away_id, kickoff
            FROM fixtures
            WHERE (home_id=%s OR away_id=%s) AND status='FT'
            ORDER BY kickoff DESC LIMIT %s
        """, (team_id, team_id, limit))
        data = await rows.fetchall()
    if data:
        gf=ga=0; last_dt=None
        for r in data:
            if r["home_id"]==team_id: gf+=(r["goals_home"] or 0); ga+=(r["goals_away"] or 0); last_dt = last_dt or r["kickoff"]
            else: gf+=(r["goals_away"] or 0); ga+=(r["goals_home"] or 0); last_dt = last_dt or r["kickoff"]
        cnt=len(data); rest=(now_local()-last_dt).days if last_dt else 7.0
        return (gf/cnt if cnt else 1.2, ga/cnt if cnt else 1.1, rest)
    # API fallback
    last = await get_team_last_matches(team_id, limit)
    gf=ga=cnt=0; last_ts=None
    for m in last:
        t_home=m["teams"]["home"]["id"]; gh=m["goals"]["home"] or 0; ga_=m["goals"]["away"] or 0
        ts=datetime.fromtimestamp(m["fixture"]["timestamp"], tz=TZ)
        if t_home==team_id: gf+=gh; ga+=ga_; cnt+=1
        else: gf+=ga_; ga+=gh; cnt+=1
        last_ts = last_ts or ts
    rest=(now_local()-last_ts).days if last_ts else 7.0
    return (gf/cnt if cnt else 1.2, ga/cnt if cnt else 1.1, rest)

async def insert_odds_snapshot(fid:int, over:float, under:float):
    async with await psycopg.AsyncConnection.connect(DB_URL) as con:
        await con.execute("INSERT INTO odds_snapshots(fixture_id,over25,under25) VALUES (%s,%s,%s)", (fid, over, under))
        await con.commit()

async def compute_and_upsert_features(fixture):
    fix=fixture["fixture"]; league=fixture["league"]; teams=fixture["teams"]
    fid=fix["id"]; home_id=teams["home"]["id"]; away_id=teams["away"]["id"]

    odds = await get_odds_for_fixture(fid)
    over, under = odds.get("over"), odds.get("under")
    if over and under: await insert_odds_snapshot(fid, over, under)

    # odds drift
    async with await psycopg.AsyncConnection.connect(DB_URL, row_factory=dict_row) as con:
        rows = await con.execute("SELECT * FROM odds_snapshots WHERE fixture_id=%s ORDER BY ts ASC", (fid,))
        snaps = await rows.fetchall()
    drift_over = drift_under = None
    if len(snaps) >= 2:
        drift_over  = (snaps[-1]["over25"] or 0) - (snaps[0]["over25"] or 0)
        drift_under = (snaps[-1]["under25"] or 0) - (snaps[0]["under25"] or 0)

    # rolling form
    h_gf,h_ga,h_rest = await team_recent_stats(home_id, LOOKBACK_MATCHES)
    a_gf,a_ga,a_rest = await team_recent_stats(away_id, LOOKBACK_MATCHES)

    # league env
    async with await psycopg.AsyncConnection.connect(DB_URL) as con:
        r = await con.execute("""
            SELECT AVG((goals_home+goals_away)::float) FROM fixtures
            WHERE league_id=%s AND status='FT' AND goals_home IS NOT NULL AND goals_away IS NOT NULL
        """, (league.get("id"),))
        env = (await r.fetchone())[0] or 2.6

    p_over_bk = implied_prob(over) if over else None
    p_under_bk = implied_prob(under) if under else None
    overround = (p_over_bk or 0) + (p_under_bk or 0)

    async with await psycopg.AsyncConnection.connect(DB_URL) as con:
        await con.execute("""
            INSERT INTO features(
                fixture_id, home_form_gf5,home_form_ga5,away_form_gf5,away_form_ga5,
                home_rest_days,away_rest_days,league_goal_env,
                odds_over_implied,odds_under_implied,odds_overround,odds_over_drift,odds_under_drift
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (fixture_id) DO UPDATE SET
              computed_at=now(),
              home_form_gf5=EXCLUDED.home_form_gf5, home_form_ga5=EXCLUDED.home_form_ga5,
              away_form_gf5=EXCLUDED.away_form_gf5, away_form_ga5=EXCLUDED.away_form_ga5,
              home_rest_days=EXCLUDED.home_rest_days, away_rest_days=EXCLUDED.away_rest_days,
              league_goal_env=EXCLUDED.league_goal_env,
              odds_over_implied=EXCLUDED.odds_over_implied, odds_under_implied=EXCLUDED.odds_under_implied,
              odds_overround=EXCLUDED.odds_overround, odds_over_drift=EXCLUDED.odds_over_drift, odds_under_drift=EXCLUDED.odds_under_drift
        """, (fid,h_gf,h_ga,a_gf,a_ga,h_rest,a_rest,env,p_over_bk,p_under_bk,overround,drift_over,drift_under))
        await con.commit()

    return odds

# ========= MODEL LOADING =========
FEATURES_ORDER = [
    "home_form_gf5","home_form_ga5","away_form_gf5","away_form_ga5",
    "home_rest_days","away_rest_days","league_goal_env",
    "odds_over_implied","odds_under_implied","odds_overround",
    "odds_over_drift","odds_under_drift"
]

class CalibratedModel:
    def __init__(self, cat, lr): self.cat=cat; self.lr=lr
    def predict_over(self, X: np.ndarray) -> np.ndarray:
        raw = np.clip(self.cat.predict_proba(X)[:,1], 1e-6, 1-1e-6)
        logits = np.log(raw/(1-raw))
        coef = float(self.lr.coef_.ravel()[0]); intercept = float(self.lr.intercept_.ravel()[0])
        z = coef*logits + intercept
        return 1/(1+np.exp(-z))

async def load_latest_model():
    async with await psycopg.AsyncConnection.connect(DB_URL, row_factory=dict_row) as con:
        row = await con.execute("""
            SELECT blob, meta FROM model_registry
            WHERE label='ou25_global' ORDER BY created_at DESC LIMIT 1
        """); r = await row.fetchone()
        if not r: return None, None
        model = pickle.loads(bytes(r["blob"]))
        meta  = r["meta"] or {}
        return model, meta

# ========= POLICY =========
async def get_policy_for_league(league_name: str):
    pol = await get_cfg("policy") or {}
    g = pol.get("global", {"theta":0.78,"ev_min":0.04})
    theta_g, ev_g = float(g.get("theta",0.78)), float(g.get("ev_min",0.04))
    ln = (league_name or "").lower()
    for k,v in pol.items():
        if k=="global": continue
        if k in ln:
            return float(v.get("theta",theta_g)), float(v.get("ev_min",ev_g))
    return theta_g, ev_g

# ========= PICK LOGIC =========
def choose_pick(p_model, odds_over, odds_under, theta, ev_min):
    if not (odds_over and odds_under): return None
    p_over = float(p_model); p_under = 1.0 - p_over
    p_over_bk, _ = normalize_overround(implied_prob(odds_over), implied_prob(odds_under))
    p_over_blend = 0.7*p_over + 0.3*p_over_bk
    p_under_blend = 1.0 - p_over_blend
    ev_over  = expected_value(p_over_blend, odds_over)
    ev_under = expected_value(p_under_blend, odds_under)
    if p_over_blend>=theta and ev_over>=ev_min and ev_over>=ev_under: return ("over", p_over_blend, ev_over)
    if p_under_blend>=theta and ev_under>=ev_min and ev_under>ev_over: return ("under", p_under_blend, ev_under)
    return None

# ========= HEARTBEAT =========
async def maybe_send_daily_heartbeat():
    if not is_exact_local_time("08:00"): return
    key = f"heartbeat:{now_local().strftime('%Y-%m-%d')}"
    if not await get_cfg(key):
        await send_telegram("Goalsniper AI live and scanning ✅")
        await set_cfg(key, {"sent_at": now_local().isoformat()})

# ========= CORE: PREDICT =========
async def predict_and_send_for_today():
    if within_cooldown():
        print("[INFO] Cooldown active; no outbound predictions."); return

    model, meta = await load_latest_model()
    feature_order = FEATURES_ORDER if not (meta and isinstance(meta, dict)) else (meta.get("feature_order") or FEATURES_ORDER)

    today = now_local().date().isoformat()
    fixtures = await get_fixtures_by_date(today)

    for f in fixtures:
        await upsert_fixture_row(f)
        fix=f["fixture"]; teams=f["teams"]; league=f["league"]
        if fix["status"]["short"] not in ("NS","TBD"): continue

        info = FixtureInfo(
            fixture_id=fix["id"], league=league["name"],
            kickoff=datetime.fromtimestamp(fix["timestamp"], tz=TZ),
            home_id=teams["home"]["id"], away_id=teams["away"]["id"],
            home=teams["home"]["name"], away=teams["away"]["name"]
        )

        odds = await compute_and_upsert_features(f)
        if not odds or not odds.get("over") or not odds.get("under"): continue

        # load features row
        async with await psycopg.AsyncConnection.connect(DB_URL, row_factory=dict_row) as con:
            r=await con.execute("SELECT * FROM features WHERE fixture_id=%s",(info.fixture_id,))
            feat=await r.fetchone()
        if not feat: continue

        X = np.array([[float(feat.get(k) or 0.0) for k in feature_order]])
        # probability
        if model:
            try:
                p_over = float(model.predict_over(X).reshape(-1)[0])
            except Exception:
                p_over = normalize_overround(implied_prob(odds["over"]), implied_prob(odds["under"]))[0]
        else:
            p_over = normalize_overround(implied_prob(odds["over"]), implied_prob(odds["under"]))[0]

        theta, ev_min = await get_policy_for_league(info.league)
        pick = choose_pick(p_over, odds["over"], odds["under"], theta, ev_min)
        if not pick: continue
        sel, conf, ev = pick

        # pretty tip message
        chosen_odds = odds["over"] if sel == "over" else odds["under"]
        chosen_bm   = odds["over_bm"] if sel == "over" else odds["under_bm"]
        pick_text   = "Over 2.5 Goals" if sel == "over" else "Under 2.5 Goals"
        msg = (
            "⚽️ *New Tip!*\n"
            f"*Match:* {info.home} vs {info.away}\n"
            f"🕒 *Kickoff:* {info.kickoff.strftime('%Y-%m-%d %H:%M')} (Berlin)\n"
            f"🎯 *Tip:* {pick_text}\n"
            f"📊 *Confidence:* {conf*100:.1f}%\n"
            f"💰 *Odds:* {chosen_odds:.2f} @ {chosen_bm or 'best available'} (best)  •  *EV:* {ev*100:+.1f}%\n"
            f"🏆 *League:* {info.league}"
        )
        await send_telegram(msg)

        async with await psycopg.AsyncConnection.connect(DB_URL) as con:
            await con.execute("""
                INSERT INTO predictions(fixture_id,league_name,home,away,kickoff,market,pick,p_over,p_under,odds_over,odds_under,confidence,ev,explain)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (info.fixture_id, info.league, info.home, info.away, info.kickoff, "ou_2_5", sel,
                  p_over, 1-p_over, odds["over"], odds["under"], conf, ev, json.dumps({"bm_over":odds["over_bm"],"bm_under":odds["under_bm"]})))
            await con.commit()

# ========= MOTD =========
async def motd_at_10():
    if not is_exact_local_time("10:00"): return
    model, meta = await load_latest_model()
    feature_order = FEATURES_ORDER if not (meta and isinstance(meta, dict)) else (meta.get("feature_order") or FEATURES_ORDER)
    today = now_local().date().isoformat()
    fixtures = await get_fixtures_by_date(today)
    cands=[]
    for f in fixtures:
        await upsert_fixture_row(f)
        if f["fixture"]["status"]["short"] not in ("NS","TBD"): continue
        odds = await compute_and_upsert_features(f)
        if not odds or not odds.get("over") or not odds.get("under"): continue
        fid=f["fixture"]["id"]; league=f["league"]; teams=f["teams"]
        async with await psycopg.AsyncConnection.connect(DB_URL, row_factory=dict_row) as con:
            r=await con.execute("SELECT * FROM features WHERE fixture_id=%s",(fid,))
            feat=await r.fetchone()
        if not feat: continue
        X = np.array([[float(feat.get(k) or 0.0) for k in feature_order]])
        p_over = (float(model.predict_over(X).reshape(-1)[0]) if model else
                  normalize_overround(implied_prob(odds["over"]), implied_prob(odds["under"]))[0])
        theta, ev_min = await get_policy_for_league(league["name"])
        pick = choose_pick(p_over, odds["over"], odds["under"], theta, ev_min)
        if not pick: continue
        sel, conf, ev = pick
        score = 2.0*ev + 0.5*conf
        cands.append((score, league, teams, f["fixture"], sel, conf, ev, odds))
    if not cands:
        await send_telegram("🌟 MOTD — no high-confidence O/U 2.5 pre-match tip today."); return
    cands.sort(key=lambda x:x[0], reverse=True)
    _, league, teams, fix, sel, conf, ev, odds = cands[0]
    chosen_odds = odds["over"] if sel == "over" else odds["under"]
    chosen_bm   = odds["over_bm"] if sel == "over" else odds["under_bm"]
    pick_text   = "Over 2.5 Goals" if sel == "over" else "Under 2.5 Goals"
    text = (
        "🌟 *MOTD — Best Pre-Match Tip*\n"
        f"*Match:* {teams['home']['name']} vs {teams['away']['name']}\n"
        f"🕒 *Kickoff:* {datetime.fromtimestamp(fix['timestamp'], tz=TZ).strftime('%Y-%m-%d %H:%M')} (Berlin)\n"
        f"🎯 *Tip:* {pick_text}\n"
        f"📊 *Confidence:* {conf*100:.1f}%  •  *EV:* {ev*100:+.1f}%\n"
        f"💰 *Odds:* {chosen_odds:.2f} @ {chosen_bm or 'best available'} (best)\n"
        f"🏆 *League:* {league['name']}"
    )
    await send_telegram(text)

# ========= DIGEST =========
async def settle_for_date(d: date, send_digest: bool):
    fixtures = await get_fixtures_by_date(d.isoformat())
    results_map={}
    for f in fixtures:
        await upsert_fixture_row(f)
        if f["fixture"]["status"]["short"]=="FT":
            gh=f["goals"]["home"] or 0; ga=f["goals"]["away"] or 0
            results_map[f["fixture"]["id"]] = (gh,ga)

    wins=loss=push=0; bets=0; roi=0.0; avg_odds=0.0
    async with await psycopg.AsyncConnection.connect(DB_URL, row_factory=dict_row) as con:
        rows = await con.execute(
            "SELECT * FROM predictions WHERE (sent_at AT TIME ZONE 'Europe/Berlin')::date = %s::date", (d.isoformat(),)
        ); preds = await rows.fetchall()
        for p in preds:
            if p["fixture_id"] not in results_map: continue
            gh,ga = results_map[p["fixture_id"]]
            over25=(gh+ga)>=3
            won = (p["pick"]=="over" and over25) or (p["pick"]=="under" and not over25)
            odds = p["odds_over"] if p["pick"]=="over" else p["odds_under"]
            if won: wins+=1; roi+=(odds-1.0)
            else:   loss+=1; roi-=1.0
            avg_odds+=odds; bets+=1
        if bets: avg_odds/=bets
        if send_digest:
            msg=f"{d.isoformat()} — Bets: {bets} | W-L-P: {wins}-{loss}-{push} | Hit: {(wins/bets*100 if bets else 0):.1f}% | ROI: {roi:+.2f}u | Avg Odds: {avg_odds:.2f}"
            await send_telegram(msg)
        for fid,(gh,ga) in results_map.items():
            await con.execute(
                "INSERT INTO results(fixture_id,goals_home,goals_away,settled_at) VALUES (%s,%s,%s,now()) ON CONFLICT (fixture_id) DO NOTHING",
                (fid, gh, ga)
            ); await con.commit()

async def settle_yesterday_and_digest_at_0830():
    if not is_exact_local_time("08:30"): return
    await settle_for_date(now_local().date()-timedelta(days=1), send_digest=True)

# ========= HARVEST / BACKFILL (manual use) =========
@contextlib.asynccontextmanager
async def advisory_lock(lock_id: int):
    async with await psycopg.AsyncConnection.connect(DB_URL) as con:
        await con.execute("SELECT pg_advisory_lock(%s)", (lock_id,))
        try: yield
        finally: await con.execute("SELECT pg_advisory_unlock(%s)", (lock_id,))

def daterange(d1: date, d2: date):
    cur=d1
    while cur<=d2: yield cur; cur+=timedelta(days=1)

async def harvest_range(start: str, end: str):
    d1=datetime.fromisoformat(start).date(); d2=datetime.fromisoformat(end).date()
    for d in daterange(d1,d2):
        fixtures = await get_fixtures_by_date(d.isoformat())
        for f in fixtures: await upsert_fixture_row(f)
        print(f"[HARVEST] {d} fixtures={len(fixtures)}")

async def backfill_days(days:int, send_digest:bool):
    y=now_local().date()-timedelta(days=1); start=y-timedelta(days=days-1)
    for d in daterange(start,y): await settle_for_date(d, send_digest); print(f"[BACKFILL] settled {d}")

# ========= HTTP SERVER =========
async def http_health(_): return web.json_response({"ok": True, "time": now_local().isoformat(), "cooldown": within_cooldown()})
async def http_stats(_):
    out={"now": now_local().isoformat()}
    async with await psycopg.AsyncConnection.connect(DB_URL, row_factory=dict_row) as con:
        rows = await con.execute("""
            SELECT p.fixture_id,p.pick,p.odds_over,p.odds_under,p.sent_at,r.goals_home,r.goals_away
            FROM predictions p LEFT JOIN results r ON r.fixture_id=p.fixture_id
            WHERE p.sent_at >= now() - interval '24 hours'
        """); d24=await rows.fetchall()
        bets24=len(d24); wins24=0
        for r in d24:
            if r["goals_home"] is None or r["goals_away"] is None: continue
            over25=(r["goals_home"]+r["goals_away"])>=3
            won=(r["pick"]=="over" and over25) or (r["pick"]=="under" and not over25)
            wins24+=1 if won else 0
        out["last_24h"]={"bets":bets24,"wins":wins24,"hit_rate":(wins24/bets24) if bets24 else None}

        rows = await con.execute("""
            SELECT p.fixture_id,p.pick,p.odds_over,p.odds_under,p.sent_at,r.goals_home,r.goals_away
            FROM predictions p LEFT JOIN results r ON r.fixture_id=p.fixture_id
            WHERE p.sent_at >= now() - interval '7 days'
        """); d7=await rows.fetchall()
        bets7=len(d7); wins7=0
        for r in d7:
            if r["goals_home"] is None or r["goals_away"] is None: continue
            over25=(r["goals_home"]+r["goals_away"])>=3
            won=(r["pick"]=="over" and over25) or (r["pick"]=="under" and not over25)
            wins7+=1 if won else 0
        out["last_7d"]={"bets":bets7,"wins":wins7,"hit_rate":(wins7/bets7) if bets7 else None}

        pol = await con.execute("SELECT value FROM model_cfg WHERE key='policy'"); pol_row=await pol.fetchone()
        out["policy"] = pol_row["value"] if pol_row else None
        mdl = await con.execute("SELECT label,version,created_at FROM model_registry ORDER BY created_at DESC LIMIT 1"); mdl_row=await mdl.fetchone()
        out["model"] = dict(mdl_row) if mdl_row else None
    return web.json_response(out)

async def start_http_server():
    app=web.Application()
    app.router.add_get("/", lambda _: web.json_response({"name":"goalsniper","status":"running"}))
    app.router.add_get("/health", http_health)
    app.router.add_get("/stats",  http_stats)
    runner=web.AppRunner(app); await runner.setup()
    site=web.TCPSite(runner,"0.0.0.0",HTTP_PORT); await site.start()
    print(f"[HTTP] listening on 0.0.0.0:{HTTP_PORT}")

# ========= SCHEDULER =========
async def scheduler_main():
    if not (API_KEY and BOT and CHAT and DB_URL):
        raise SystemExit("Missing required env (APIFOOTBALL_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, DATABASE_URL).")
    await ensure_schema()
    await run_migrations_once()
    await start_http_server()

    sched = AsyncIOScheduler(timezone=str(TZ))

    async def job_predict():
        async with advisory_lock(1001): await predict_and_send_for_today()
    sched.add_job(job_predict, IntervalTrigger(minutes=15), id="predict-interval", coalesce=True, max_instances=1)

    async def job_motd():
        async with advisory_lock(1002): await motd_at_10()
    sched.add_job(job_motd, CronTrigger(hour=10, minute=0), id="motd-10", coalesce=True, max_instances=1)

    async def job_digest():
        async with advisory_lock(1003): await settle_yesterday_and_digest_at_0830()
    sched.add_job(job_digest, CronTrigger(hour=8, minute=30), id="digest-0830", coalesce=True, max_instances=1)

    async def job_heartbeat():
        async with advisory_lock(1004): await maybe_send_daily_heartbeat()
    sched.add_job(job_heartbeat, CronTrigger(hour=8, minute=0), id="heartbeat-0800", coalesce=True, max_instances=1)

    # Nightly trainer hook (works with the train_models.py I sent)
    try:
        from train_models import main as train_main
        async def job_train():
            async with advisory_lock(1005): await train_main()
        sched.add_job(job_train, CronTrigger(hour=23, minute=30), id="train-2330", coalesce=True, max_instances=1)
    except Exception:
        print("[WARN] train_models.py not importable; skipping nightly training.")

    sched.start()
    try:
        while True: await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass

# ========= BOOT =========
if __name__ == "__main__":
    asyncio.run(scheduler_main())
