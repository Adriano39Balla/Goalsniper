import os, asyncio, httpx, math, json, re
from dataclasses import dataclass
from datetime import datetime, timedelta, time as dtime
from zoneinfo import ZoneInfo
import psycopg
from psycopg.rows import dict_row

# ---- ENV / CONFIG ----
TZ = ZoneInfo(os.getenv("TZ", "Europe/Berlin"))
API_KEY = os.getenv("APIFOOTBALL_KEY", "")
BOT = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT = os.getenv("TELEGRAM_CHAT_ID", "")
DB_URL = os.getenv("DATABASE_URL", "")
COOLDOWN_START = os.getenv("COOLDOWN_START", "23:00")
COOLDOWN_END   = os.getenv("COOLDOWN_END", "07:00")

# Fallbacks (will be overridden by DB policy if present)
THRESHOLD_FALLBACK = float(os.getenv("THRESHOLD", "0.75"))   # aim 75–80% precision
EV_MIN_FALLBACK    = float(os.getenv("EV_MIN", "0.05"))      # minimum expected value
MAX_LOOKBACK = int(os.getenv("LOOKBACK_MATCHES", "10"))      # recent matches for Poisson

API_BASE = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}

# ---- UTILITIES ----
def now_local(): return datetime.now(TZ)
def parse_hhmm(s):
    h, m = map(int, s.split(":")); return dtime(h, m)
def within_cooldown():
    n = now_local().time()
    s, e = parse_hhmm(COOLDOWN_START), parse_hhmm(COOLDOWN_END)
    return (n >= s) or (n < e)

def is_exact_local_time(hhmm: str):
    t = parse_hhmm(hhmm); n = now_local()
    return n.hour == t.hour and n.minute == t.minute

def league_excluded(name: str, type_: str) -> bool:
    if type_ not in ("League", "Cup"): return True
    return re.search(r"(U-?\d{2}|U\d{2}|Youth|Reserves?|Academy|U19|U21|U23|B$|II$)", name, re.I) is not None

async def send_telegram(text: str):
    if not (BOT and CHAT): return
    url = f"https://api.telegram.org/bot{BOT}/sendMessage"
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.post(url, json={"chat_id": CHAT, "text": text, "parse_mode": "Markdown", "disable_web_page_preview": True})
        r.raise_for_status()

def implied_prob(decimal_odds: float) -> float:
    if not decimal_odds or decimal_odds <= 1.0: return 0.0
    return 1.0 / decimal_odds

def normalize_overround(p_over: float, p_under: float):
    s = max(p_over + p_under, 1e-6)
    return p_over / s, p_under / s

def expected_value(prob: float, dec_odds: float):
    # EV per 1u stake
    return prob * (dec_odds - 1.0) - (1.0 - prob)

def poisson_p_total_ge_k(lambda_total: float, k: int) -> float:
    # P(X >= k) for Poisson; 1 - CDF(k-1)
    cdf = 0.0
    for i in range(0, k):
        cdf += math.exp(-lambda_total) * lambda_total**i / math.factorial(i)
    return max(0.0, min(1.0, 1.0 - cdf))

# ---- DB ----
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
  id BIGSERIAL PRIMARY KEY,
  fixture_id INT,
  league_name TEXT,
  home TEXT, away TEXT,
  kickoff TIMESTAMPTZ,
  market TEXT,
  pick TEXT,
  p_over REAL,
  p_under REAL,
  odds_over REAL,
  odds_under REAL,
  confidence REAL,
  ev REAL,
  sent_at TIMESTAMPTZ DEFAULT now()
);
CREATE TABLE IF NOT EXISTS results (
  fixture_id INT PRIMARY KEY,
  goals_home SMALLINT,
  goals_away SMALLINT,
  settled_at TIMESTAMPTZ
);
CREATE TABLE IF NOT EXISTS model_cfg (
  key TEXT PRIMARY KEY,
  value JSONB
);
"""

async def ensure_schema():
    async with await psycopg.AsyncConnection.connect(DB_URL) as con:
        await con.execute(SCHEMA_SQL)
        await con.commit()

async def get_policy():
    # Reads {'theta': float, 'ev_min': float} from model_cfg.policy; falls back to env defaults
    theta, ev_min = THRESHOLD_FALLBACK, EV_MIN_FALLBACK
    async with await psycopg.AsyncConnection.connect(DB_URL, row_factory=dict_row) as con:
        row = await con.execute("SELECT value FROM model_cfg WHERE key='policy'")
        r = await row.fetchone()
        if r and isinstance(r["value"], dict):
            theta = float(r["value"].get("theta", theta))
            ev_min = float(r["value"].get("ev_min", ev_min))
    return theta, ev_min

# ---- API-FOOTBALL CALLS ----
async def api_get(path: str, params: dict):
    async with httpx.AsyncClient(headers=HEADERS, timeout=30) as c:
        r = await c.get(f"{API_BASE}/{path}", params=params); r.raise_for_status()
        return r.json()

async def get_fixtures_by_date(date_iso: str):
    data = await api_get("fixtures", {"date": date_iso})
    # filter out youth/reserve cups/leagues if API returns 'league' info here
    out = []
    for r in data.get("response", []):
        lg = r.get("league", {}) or {}
        lgname = lg.get("name", "")
        lgtype = lg.get("type", "League")
        if league_excluded(lgname, lgtype): 
            continue
        out.append(r)
    return out

async def get_team_last_matches(team_id: int, n: int = MAX_LOOKBACK):
    data = await api_get("fixtures", {"team": team_id, "last": n})
    return data.get("response", [])

async def get_odds_for_fixture(fixture_id: int):
    data = await api_get("odds", {"fixture": fixture_id})
    # find O/U 2.5 market
    best = {"over": None, "under": None}
    for resp in data.get("response", []):
        for bk in resp.get("bookmakers", []):
            for market in bk.get("bets", []):
                if str(market.get("name", "")).lower() in ("over/under", "goals over/under"):
                    for v in market.get("values", []):
                        label = v.get("value", "").strip()
                        odd = v.get("odd", None)
                        if not odd: 
                            continue
                        if label in ("Over 2.5", "Over 2,5"):
                            best["over"] = max(best["over"] or 0.0, float(odd))
                        elif label in ("Under 2.5", "Under 2,5"):
                            best["under"] = max(best["under"] or 0.0, float(odd))
    return best

# ---- FEATURES via SIMPLE POISSON (fast baseline) ----
@dataclass
class FixtureInfo:
    fixture_id: int
    league: str
    kickoff: datetime
    home_id: int
    away_id: int
    home: str
    away: str

async def compute_poisson_over25(home_id: int, away_id: int):
    # compute λ_home_for, λ_away_for as recent averages; total lambda is sum of both teams' avg goals per match
    last_home = await get_team_last_matches(home_id, MAX_LOOKBACK)
    last_away = await get_team_last_matches(away_id, MAX_LOOKBACK)

    def avg_goals(matches, as_team_id):
        gf = ga = cnt = 0
        for m in matches:
            t_home = m["teams"]["home"]["id"]; t_away = m["teams"]["away"]["id"]
            gh = m["goals"]["home"] or 0; ga_ = m["goals"]["away"] or 0
            if as_team_id == t_home:
                gf += gh; ga += ga_; cnt += 1
            elif as_team_id == t_away:
                gf += ga_; ga += gh; cnt += 1
        return (gf / cnt if cnt else 1.2), (ga / cnt if cnt else 1.2)

    gf_h, _ = avg_goals(last_home, home_id)
    gf_a, _ = avg_goals(last_away, away_id)
    lam_total = max(0.2, gf_h + gf_a)  # quick proxy
    p_over = poisson_p_total_ge_k(lam_total, 3)
    p_under = 1.0 - p_over
    return p_over, p_under, lam_total

# ---- SELECTION POLICY ----
def choose_pick(p_over, odds_over, p_under, odds_under, threshold, ev_min):
    if not odds_over or not odds_under: return None
    p_over_adj, p_under_adj = normalize_overround(implied_prob(odds_over), implied_prob(odds_under))  # book priors
    # conservative blend with market priors
    p_over_blend = 0.6 * p_over + 0.4 * p_over_adj
    p_under_blend = 1.0 - p_over_blend

    ev_over = expected_value(p_over_blend, odds_over)
    ev_under = expected_value(p_under_blend, odds_under)

    if p_over_blend >= threshold and ev_over >= ev_min and ev_over >= ev_under:
        return ("over", p_over_blend, ev_over)
    if p_under_blend >= threshold and ev_under >= ev_min and ev_under > ev_over:
        return ("under", p_under_blend, ev_under)
    return None

# ---- CORE TASKS ----
async def predict_and_send_for_today():
    if within_cooldown():
        print("[INFO] Cooldown active; no outbound predictions.")
        return

    theta, ev_min = await get_policy()
    today = now_local().date().isoformat()
    fixtures = await get_fixtures_by_date(today)
    results = []
    for f in fixtures:
        fix = f["fixture"]; teams = f["teams"]; league = f["league"]
        status = fix["status"]["short"]
        if status not in ("NS","TBD"):  # only not started
            continue
        info = FixtureInfo(
            fixture_id=fix["id"],
            league=league["name"],
            kickoff=datetime.fromtimestamp(fix["timestamp"], tz=TZ),
            home_id=teams["home"]["id"], away_id=teams["away"]["id"],
            home=teams["home"]["name"], away=teams["away"]["name"]
        )
        odds = await get_odds_for_fixture(info.fixture_id)
        if not odds or (not odds["over"] or not odds["under"]): continue

        p_over, p_under, lam = await compute_poisson_over25(info.home_id, info.away_id)
        pick = choose_pick(p_over, odds["over"], p_under, odds["under"], theta, ev_min)
        if not pick: continue

        sel, conf, ev = pick
        msg = (f"*{info.league}* {info.kickoff.strftime('%Y-%m-%d %H:%M')}\n"
               f"{info.home} vs {info.away}\n"
               f"Pick: *{sel.upper()} 2.5* | P={conf:.2f} | Odds(O/U)={odds['over']:.2f}/{odds['under']:.2f} | EV={ev:+.2f}\n"
               f"λ≈{lam:.2f} | θ={theta:.2f} EVmin={ev_min:.2f}")
        await send_telegram(msg)
        # log
        async with await psycopg.AsyncConnection.connect(DB_URL) as con:
            await con.execute(
                """INSERT INTO predictions(fixture_id, league_name, home, away, kickoff, market, pick, p_over, p_under, odds_over, odds_under, confidence, ev)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (info.fixture_id, info.league, info.home, info.away, info.kickoff, "ou_2_5", sel, p_over, p_under, odds["over"], odds["under"], conf, ev)
            ); await con.commit()
        results.append((info.fixture_id, sel, conf, ev))

    if not results:
        print("[INFO] No qualifying picks found (threshold/EV too strict or no odds).")

async def motd_at_10():
    if not is_exact_local_time("10:00"): 
        return
    theta, ev_min = await get_policy()
    today = now_local().date().isoformat()
    fixtures = await get_fixtures_by_date(today)
    candidates = []
    for f in fixtures:
        fix = f["fixture"]; teams = f["teams"]; league = f["league"]
        if fix["status"]["short"] not in ("NS","TBD"): continue
        info = FixtureInfo(
            fixture_id=fix["id"], league=league["name"],
            kickoff=datetime.fromtimestamp(fix["timestamp"], tz=TZ),
            home_id=teams["home"]["id"], away_id=teams["away"]["id"],
            home=teams["home"]["name"], away=teams["away"]["name"]
        )
        odds = await get_odds_for_fixture(info.fixture_id)
        if not odds or (not odds["over"] or not odds["under"]): continue
        p_over, p_under, lam = await compute_poisson_over25(info.home_id, info.away_id)
        pick = choose_pick(p_over, odds["over"], p_under, odds["under"], theta, ev_min)
        if pick:
            sel, conf, ev = pick
            weight = 1.0 + (0.1 if any(x in info.league.lower() for x in ["premier","bundesliga","laliga","serie a","ligue 1","uefa"]) else 0)
            score = ev * 2.0 + conf * 0.5
            candidates.append((score*weight, info, sel, conf, ev, lam, odds))
    if not candidates:
        await send_telegram("MOTD — no high-confidence O/U 2.5 pick today.")
        return
    candidates.sort(key=lambda x: x[0], reverse=True)
    _, info, sel, conf, ev, lam, odds = candidates[0]
    text = (f"*MOTD — Best O/U 2.5*\n{info.league} {info.kickoff.strftime('%Y-%m-%d %H:%M')}\n"
            f"{info.home} vs {info.away}\n"
            f"Pick: *{sel.UPPER()} 2.5* | P={conf:.2f} | Odds={odds['over']:.2f}/{odds['under']:.2f} | EV={ev:+.2f} | λ≈{lam:.2f}")
    await send_telegram(text)

async def settle_yesterday_and_digest_at_0830():
    if not is_exact_local_time("08:30"): 
        return
    y = now_local().date() - timedelta(days=1)
    fixtures = await get_fixtures_by_date(y.isoformat())
    # map fixture_id -> goals
    results = {}
    for f in fixtures:
        fix = f["fixture"]; goals = f["goals"]
        if fix["status"]["short"] == "FT":
            results[fix["id"]] = (goals["home"] or 0, goals["away"] or 0)
    wins=loss=push=0; bets=0; roi=0.0; avg_odds=0.0
    async with await psycopg.AsyncConnection.connect(DB_URL, row_factory=dict_row) as con:
        rows = await con.execute(
            "SELECT * FROM predictions WHERE (sent_at AT TIME ZONE 'Europe/Berlin')::date = %s::date",
            (y.isoformat(),)
        ); preds = await rows.fetchall()
        for p in preds:
            if p["fixture_id"] not in results: continue
            gh, ga = results[p["fixture_id"]]
            over25 = (gh + ga) >= 3
            won = (p["pick"] == "over" and over25) or (p["pick"] == "under" and not over25)
            odds = p["odds_over"] if p["pick"]=="over" else p["odds_under"]
            if won:
                wins += 1
                roi += (odds - 1.0)
            else:
                loss += 1
                roi -= 1.0
            avg_odds += odds; bets += 1
        if bets:
            avg_odds /= bets
        msg = f"{y.isoformat()} — Bets: {bets} | W-L-P: {wins}-{loss}-{push} | Hit: {(wins/bets*100 if bets else 0):.1f}% | ROI: {roi:+.2f}u | Avg Odds: {avg_odds:.2f}"
        await send_telegram(msg)
        # store results
        for fid,(gh,ga) in results.items():
            await con.execute(
                "INSERT INTO results(fixture_id,goals_home,goals_away,settled_at) VALUES (%s,%s,%s,now()) ON CONFLICT (fixture_id) DO NOTHING",
                (fid, gh, ga)
            ); await con.commit()

# ---- ENTRYPOINT ----
async def main():
    if not (API_KEY and BOT and CHAT and DB_URL):
        raise SystemExit("Missing required env (APIFOOTBALL_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, DATABASE_URL).")
    await ensure_schema()

    task = os.getenv("TASK", "predict")  # predict | motd | digest
    if task == "digest":
        await settle_yesterday_and_digest_at_0830()
    elif task == "motd":
        await motd_at_10()
    else:
        await predict_and_send_for_today()

if __name__ == "__main__":
    asyncio.run(main())
