import asyncio
import math
from typing import Any, Dict, List, Optional, Tuple
from .api_football import get_team_statistics, get_odds_for_fixture
from .config import STATS_REQUEST_DELAY_MS, MIN_CONFIDENCE_TO_SEND
from .logger import warn

def _form_to_points(form: Optional[str]) -> int:
    if not form:
        return 0
    pts = 0
    for ch in (ch for ch in form if ch in ("W","D","L")):
        pts += 3 if ch == "W" else (1 if ch == "D" else 0)
    return pts

def _build_team_rating(stats: Optional[Dict[str, Any]], home_away: str):
    if not stats:
        return dict(rating=0.5, gf=1.2, ga=1.2, form_pts=0, games=0)

    fixtures = stats.get("fixtures", {}) or {}
    wins = (fixtures.get("wins", {}) or {}).get("total", 0) or 0
    draws = (fixtures.get("draws", {}) or {}).get("total", 0) or 0
    loses = (fixtures.get("loses", {}) or {}).get("total", 0) or 0
    games = (wins + draws + loses) or 1

    gf = ((stats.get("goals", {}) or {}).get("for", {}) or {}).get("total", {}).get("total", 0) or 0
    ga = ((stats.get("goals", {}) or {}).get("against", {}) or {}).get("total", {}).get("total", 0) or 0
    gf_per = gf / games
    ga_per = ga / games

    form_pts = _form_to_points(stats.get("form"))
    win_rate = wins / games

    played_home = ((fixtures.get("played", {}) or {}).get("home", 0)) or 1
    played_away = ((fixtures.get("played", {}) or {}).get("away", 0)) or 1
    home_perf = ((fixtures.get("wins", {}) or {}).get("home", 0)) / played_home
    away_perf = ((fixtures.get("wins", {}) or {}).get("away", 0)) / played_away
    side_perf = home_perf if home_away == "home" else away_perf

    gd_per = (gf - ga) / games
    rating = 0.5 + 0.5 * (win_rate - 0.5) * 2 * 0.5 + 0.3 * math.tanh(gd_per) + 0.2 * (side_perf - 0.5) * 2
    rating += min(0.08, (form_pts / (5 * 3)) * 0.08)
    rating = max(0.1, min(0.9, rating))

    return dict(rating=rating, gf=max(0.05, gf_per), ga=max(0.05, ga_per), form_pts=form_pts, games=games)

def _prob_to_confidence(p: float) -> float:
    return abs(p - 0.5) * 2.0

def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def _poisson_zero_prob(lmbd: float) -> float:
    return math.exp(-max(0.01, lmbd))

def _btts_prob(lambda_home: float, lambda_away: float) -> float:
    return (1 - _poisson_zero_prob(lambda_home)) * (1 - _poisson_zero_prob(lambda_away))

def _asian_line_from_rating_diff(diff: float) -> Tuple[str, float]:
    if diff >= 0.55:   return ("HOME -1.0",  -1.0)
    if diff >= 0.40:   return ("HOME -0.75", -0.75)
    if diff >= 0.25:   return ("HOME -0.5",  -0.5)
    if diff >= 0.12:   return ("HOME -0.25", -0.25)
    if diff <= -0.55:  return ("AWAY -1.0",  +1.0)
    if diff <= -0.40:  return ("AWAY -0.75", +0.75)
    if diff <= -0.25:  return ("AWAY -0.5",  +0.5)
    if diff <= -0.12:  return ("AWAY -0.25", +0.25)
    return ("DRAW NO BET (AH 0)", 0.0)

def _asian_win_prob_from_scores(h_score: float, a_score: float, line: float) -> float:
    adv = h_score - a_score
    base = 1 / (1 + math.exp(-3.5 * adv))
    adj = base + (0.12 if line in (-0.25, +0.25) else 0.18 if line in (-0.5, +0.5) else 0.24 if abs(line) == 0.75 else 0.30 if abs(line) == 1.0 else 0.0)
    return _clamp01(adj if line <= 0 else 1 - adj)

async def _fetch_stats_pair(client, league_id: int, season: int, home_id: int, away_id: int):
    h = asyncio.create_task(get_team_statistics(client, league_id, season, home_id))
    await asyncio.sleep(STATS_REQUEST_DELAY_MS / 1000.0)
    a = asyncio.create_task(get_team_statistics(client, league_id, season, away_id))
    return await asyncio.gather(h, a)

def _lambda_by_side(homeR, awayR) -> Tuple[float, float]:
    lh = 0.6 * homeR["gf"] + 0.4 * awayR["ga"]
    la = 0.6 * awayR["gf"] + 0.4 * homeR["ga"]
    return (max(0.05, lh), max(0.05, la))

def _adjust_for_live(lambda_home: float, lambda_away: float, minute: Optional[int], goals_h: int, goals_a: int) -> Tuple[float, float]:
    if minute is None or minute <= 0:
        return lambda_home, lambda_away
    remain = max(0, 90 - min(90, minute))
    factor = remain / 90.0
    lh = lambda_home * factor
    la = lambda_away * factor
    if goals_h < goals_a:
        lh *= 1.10
    elif goals_a < goals_h:
        la *= 1.10
    return (lh, la)

async def generate_tips_for_fixture(client, fixture: Dict[str, Any], league_id: int, season: int) -> List[Dict[str, Any]]:
    tips: List[Dict[str, Any]] = []
    fixture_id = ((fixture.get("fixture", {}) or {}).get("id"))
    home_id = ((fixture.get("teams", {}) or {}).get("home", {}) or {}).get("id")
    away_id = ((fixture.get("teams", {}) or {}).get("away", {}) or {}).get("id")
    if not fixture_id or not home_id or not away_id:
        return tips

    minute = ((fixture.get("fixture", {}) or {}).get("status", {}) or {}).get("elapsed")
    goals_h = ((fixture.get("goals", {}) or {}).get("home")) or 0
    goals_a = ((fixture.get("goals", {}) or {}).get("away")) or 0

    home_stats, away_stats = await _fetch_stats_pair(client, league_id, season, home_id, away_id)
    homeR = _build_team_rating(home_stats, "home")
    awayR = _build_team_rating(away_stats, "away")

    lambda_home, lambda_away = _lambda_by_side(homeR, awayR)
    lambda_home_live, lambda_away_live = _adjust_for_live(lambda_home, lambda_away, minute, goals_h, goals_a)
    expected_goals = lambda_home + lambda_away
    expected_goals_live = lambda_home_live + lambda_away_live

    home_score = homeR["rating"] + 0.06
    away_score = awayR["rating"]
    draw_base = 0.25 + max(0.0, 0.1 - abs(home_score - away_score))
    expA, expB, expD = math.exp(home_score), math.exp(away_score), math.exp(draw_base)
    s = expA + expB + expD
    p_home, p_draw, p_away = expA / s, expD / s, expB / s

    try:
        odds = await get_odds_for_fixture(client, fixture_id)
        if odds:
            bmk = (odds[0].get("bookmakers") or [])
            if bmk:
                bets = bmk[0].get("bets") or []
                mw = next((b for b in bets if b.get("name") == "Match Winner"), None)
                if mw and (mw.get("values") or []):
                    values = mw["values"]
                    def _grab(lbl):
                        for v in values:
                            val = (v.get("value") or "").lower()
                            if lbl == "home" and "home" in val: return float(v.get("odd"))
                            if lbl == "draw" and "draw" in val: return float(v.get("odd"))
                            if lbl == "away" and "away" in val: return float(v.get("odd"))
                        return None
                    oh, od, oa = _grab("home"), _grab("draw"), _grab("away")
                    if oh and od and oa:
                        ih, id_, ia = 1/oh, 1/od, 1/oa
                        isum = ih + id_ + ia
                        bh, bd, ba = ih/isum, id_/isum, ia/isum
                        p_home = _clamp01(0.8*p_home + 0.2*bh)
                        p_draw = _clamp01(0.8*p_draw + 0.2*bd)
                        p_away = _clamp01(0.8*p_away + 0.2*ba)
    except Exception as e:
        warn("odds fetch failed:", str(e))

    picks: List[Tuple[str, str, float, float]] = []
    diff = abs(p_home - p_away)
    if diff < 0.08:
        p_over = _clamp01((expected_goals - 2.5) * 0.25 + 0.5)
        p_under = 1.0 - p_over
        if p_over >= p_under:
            picks.append(("OVER_UNDER_2.5", "OVER 2.5", p_over, expected_goals))
        else:
            picks.append(("OVER_UNDER_2.5", "UNDER 2.5", p_under, expected_goals))
    else:
        if p_home > max(p_draw, p_away):
            picks.append(("1X2", "HOME", p_home, expected_goals))
        elif p_away > max(p_draw, p_home):
            picks.append(("1X2", "AWAY", p_away, expected_goals))
        else:
            picks.append(("1X2", "DRAW", p_draw, expected_goals))

    p_btts = _btts_prob(lambda_home, lambda_away)
    p_btts_live = _btts_prob(lambda_home_live, lambda_away_live)
    p_btts_use = p_btts_live if minute else p_btts
    if _prob_to_confidence(p_btts_use) >= MIN_CONFIDENCE_TO_SEND:
        picks.append(("BTTS", "YES" if p_btts_use >= 0.5 else "NO", p_btts_use, expected_goals_live if minute else expected_goals))

    line_label, line = _asian_line_from_rating_diff(home_score - away_score)
    p_ah = _asian_win_prob_from_scores(home_score, away_score, line)
    if _prob_to_confidence(p_ah) >= MIN_CONFIDENCE_TO_SEND:
        picks.append(("ASIAN_HANDICAP", line_label, p_ah, expected_goals))

    for market, selection, prob, xg in picks:
        conf = _prob_to_confidence(prob)
        if conf < MIN_CONFIDENCE_TO_SEND:
            continue
        tips.append({
            "leagueId": league_id,
            "season": season,
            "fixtureId": fixture_id,
            "kickOff": ((fixture.get("fixture") or {}).get("date")),
            "leagueName": ((fixture.get("league") or {}).get("name") or ""),
            "country": ((fixture.get("league") or {}).get("country") or ""),
            "home": ((fixture.get("teams") or {}).get("home") or {}).get("name") or "Home",
            "away": ((fixture.get("teams") or {}).get("away") or {}).get("name") or "Away",
            "market": market,
            "selection": selection,
            "probability": round(float(prob), 3),
            "confidence": round(float(conf), 3),
            "expectedGoals": round(float(xg), 2),
            "note": ("LIVE" if minute else None),
            "live": bool(minute),
            "minute": minute or 0,
            "score": f"{goals_h}-{goals_a}",
        })

    return tips
