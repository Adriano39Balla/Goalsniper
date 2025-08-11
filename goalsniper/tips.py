from __future__ import annotations

import asyncio
import math
from typing import Any, Dict, List, Optional, Tuple

from .api_football import get_team_statistics
from .config import STATS_REQUEST_DELAY_MS, MIN_CONFIDENCE_TO_SEND
from .logger import warn

# ---------- helpers ----------
def _form_to_points(form: Optional[str]) -> int:
    if not form:
        return 0
    pts = 0
    for ch in (c for c in form if c in ("W", "D", "L")):
        pts += 3 if ch == "W" else (1 if ch == "D" else 0)
    return pts

def _build_team_rating(stats: Optional[Dict[str, Any]], home_away: str) -> Dict[str, Any]:
    if not stats:
        return {"rating": 0.5, "gf": 1.2, "ga": 1.2, "form_pts": 0, "games": 0}

    fixtures = stats.get("fixtures", {}) or {}
    wins  = (fixtures.get("wins",  {}) or {}).get("total", 0) or 0
    draws = (fixtures.get("draws", {}) or {}).get("total", 0) or 0
    loses = (fixtures.get("loses", {}) or {}).get("total", 0) or 0
    games = (wins + draws + loses) or 1

    goals = (stats.get("goals") or {})
    gf = ((goals.get("for")     or {}).get("total") or {}).get("total", 0) or 0
    ga = ((goals.get("against") or {}).get("total") or {}).get("total", 0) or 0
    gf_per = gf / games
    ga_per = ga / games

    form_pts = _form_to_points(stats.get("form"))
    win_rate = wins / games

    played = (fixtures.get("played") or {})
    wins_s = (fixtures.get("wins") or {})
    played_home = (played.get("home") or 0) or 1
    played_away = (played.get("away") or 0) or 1
    home_perf = (wins_s.get("home") or 0) / played_home
    away_perf = (wins_s.get("away") or 0) / played_away
    side_perf = home_perf if home_away == "home" else away_perf

    gd_per = (gf - ga) / games
    rating = (
        0.5
        + 0.5 * (win_rate - 0.5) * 2 * 0.5
        + 0.3 * math.tanh(gd_per)
        + 0.2 * (side_perf - 0.5) * 2
    )
    rating += min(0.08, (form_pts / 15.0) * 0.08)
    rating = max(0.1, min(0.9, rating))

    return {"rating": rating, "gf": max(0.05, gf_per), "ga": max(0.05, ga_per), "form_pts": form_pts, "games": games}

def _prob_to_confidence(p: float) -> float:
    return abs(p - 0.5) * 2.0

def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def _poisson_zero_prob(lmbd: float) -> float:
    return math.exp(-max(0.01, lmbd))

def _btts_prob(lambda_home: float, lambda_away: float) -> float:
    return (1 - _poisson_zero_prob(lambda_home)) * (1 - _poisson_zero_prob(lambda_away))

async def _fetch_stats_pair(client, league_id: int, season: int, home_id: int, away_id: int):
    # staggered to respect API pacing
    h = asyncio.create_task(get_team_statistics(client, int(league_id), int(season), int(home_id)))
    await asyncio.sleep(max(0, int(STATS_REQUEST_DELAY_MS)) / 1000.0)
    a = asyncio.create_task(get_team_statistics(client, int(league_id), int(season), int(away_id)))
    return await asyncio.gather(h, a)

def _lambda_by_side(homeR: Dict[str, float], awayR: Dict[str, float]) -> Tuple[float, float]:
    lh = 0.6 * float(homeR["gf"]) + 0.4 * float(awayR["ga"])
    la = 0.6 * float(awayR["gf"]) + 0.4 * float(homeR["ga"])
    return (max(0.05, lh), max(0.05, la))

def _adjust_for_live(lambda_home: float, lambda_away: float, minute: Optional[int], goals_h: int, goals_a: int) -> Tuple[float, float]:
    if not minute or int(minute) <= 0:
        return lambda_home, lambda_away
    minute = min(90, max(0, int(minute)))
    remain = 90 - minute
    factor = remain / 90.0
    lh = lambda_home * factor
    la = lambda_away * factor
    if goals_h < goals_a:
        lh *= 1.10
    elif goals_a < goals_h:
        la *= 1.10
    return (lh, la)

# ---------- 1st-half helpers ----------
_FIRST_HALF_SHARE = 0.56  # rough share of goals in first half

def _first_half_xg(total_xg: float) -> float:
    return max(0.01, total_xg * _FIRST_HALF_SHARE)

def _first_half_remaining_factor(minute: Optional[int]) -> float:
    if not minute or int(minute) <= 0:
        return 1.0
    m = int(minute)
    if m >= 45:
        return 0.0
    return (45 - m) / 45.0

# ---------- main entry ----------
async def generate_tips_for_fixture(
    client,
    fixture: Dict[str, Any],
    league_id: int,
    season: int
) -> List[Dict[str, Any]]:
    """
    Markets:
      - 1X2
      - OVER/UNDER 2.5 (FT)
      - BTTS
      - 1st Half Over/Under (1.0 or 1.5, pick stronger)
    """
    tips: List[Dict[str, Any]] = []

    fx = fixture or {}
    fx_info = fx.get("fixture") or {}
    teams = fx.get("teams") or {}

    fixture_id = fx_info.get("id")
    home_id = (teams.get("home") or {}).get("id")
    away_id = (teams.get("away") or {}).get("id")
    if not fixture_id or not home_id or not away_id:
        return tips

    status  = fx_info.get("status") or {}
    minute  = status.get("elapsed")
    goals   = fx.get("goals") or {}
    goals_h = goals.get("home") or 0
    goals_a = goals.get("away") or 0

    try:
        home_stats, away_stats = await _fetch_stats_pair(client, league_id, season, int(home_id), int(away_id))
    except Exception as e:
        warn("team statistics fetch failed:", str(e))
        return tips

    homeR = _build_team_rating(home_stats, "home")
    awayR = _build_team_rating(away_stats, "away")

    # expected goals (full match)
    lambda_home, lambda_away = _lambda_by_side(homeR, awayR)
    lambda_home_live, lambda_away_live = _adjust_for_live(lambda_home, lambda_away, minute, int(goals_h), int(goals_a))
    expected_goals = lambda_home + lambda_away
    expected_goals_live = lambda_home_live + lambda_away_live

    # 1X2 baseline using ratings (+ small home adv)
    home_score = homeR["rating"] + 0.06
    away_score = awayR["rating"]
    draw_base  = 0.25 + max(0.0, 0.1 - abs(home_score - away_score))
    expA, expB, expD = math.exp(home_score), math.exp(away_score), math.exp(draw_base)
    s = expA + expB + expD
    p_home, p_draw, p_away = expA / s, expD / s, expB / s

    picks: List[Tuple[str, str, float, float]] = []

    # --- 1X2 ---
    gap = max(p_home, p_draw, p_away) - sorted([p_home, p_draw, p_away])[1]
    if gap >= 0.02:
        if p_home >= p_draw and p_home >= p_away:
            picks.append(("1X2", "HOME", p_home, expected_goals))
        elif p_away >= p_home and p_away >= p_draw:
            picks.append(("1X2", "AWAY", p_away, expected_goals))
        else:
            picks.append(("1X2", "DRAW", p_draw, expected_goals))

    # --- Over/Under 2.5 (FT) ---
    p_over  = _clamp01((expected_goals - 2.5) * 0.25 + 0.5)
    p_under = 1.0 - p_over
    best_ou_prob = max(p_over, p_under)
    if _prob_to_confidence(best_ou_prob) >= MIN_CONFIDENCE_TO_SEND:
        picks.append((
            "OVER_UNDER_2.5",
            "OVER 2.5" if p_over >= p_under else "UNDER 2.5",
            best_ou_prob,
            expected_goals,
        ))

    # --- BTTS ---
    p_btts      = _btts_prob(lambda_home, lambda_away)
    p_btts_live = _btts_prob(lambda_home_live, lambda_away_live)
    p_btts_use  = p_btts_live if minute else p_btts
    if _prob_to_confidence(p_btts_use) >= MIN_CONFIDENCE_TO_SEND:
        picks.append(("BTTS", "YES" if p_btts_use >= 0.5 else "NO",
                      p_btts_use, expected_goals_live if minute else expected_goals))

    # --- 1st Half Over/Under (1.0 or 1.5) ---
    if not minute or int(minute) < 45:
        fh_factor   = _first_half_remaining_factor(minute)
        fh_xg_total = _first_half_xg(expected_goals) * fh_factor

        def _ou(line: float) -> Tuple[str, float]:
            p_over_fh  = _clamp01((fh_xg_total - line) * 0.60 + 0.5)  # steeper than FT
            p_under_fh = 1.0 - p_over_fh
            return (f"OVER {line}", p_over_fh) if p_over_fh >= p_under_fh else (f"UNDER {line}", p_under_fh)

        sel10, prob10 = _ou(1.0)
        sel15, prob15 = _ou(1.5)
        best_sel, best_prob = (sel10, prob10) if prob10 >= prob15 else (sel15, prob15)
        if _prob_to_confidence(best_prob) >= MIN_CONFIDENCE_TO_SEND:
            picks.append(("OVER_UNDER_1H", best_sel, best_prob, fh_xg_total))

    # Format tips
    league = fixture.get("league") or {}
    tms    = fixture.get("teams")  or {}
    for market, selection, prob, xg in picks:
        conf = _prob_to_confidence(prob)
        if conf < MIN_CONFIDENCE_TO_SEND:
            continue
        tips.append({
            "leagueId": int(league_id),
            "season": int(season),
            "fixtureId": int(fixture_id),
            "kickOff": (fixture.get("fixture") or {}).get("date"),
            "leagueName": (league.get("name") or ""),
            "country": (league.get("country") or ""),
            "home": (tms.get("home") or {}).get("name") or "Home",
            "away": (tms.get("away") or {}).get("name") or "Away",
            "market": market,
            "selection": selection,
            "probability": round(float(prob), 3),
            "confidence": round(float(conf), 3),
            "expectedGoals": round(float(xg), 2),
            "note": ("LIVE" if minute else None),
            "live": bool(minute),
            "minute": int(minute or 0),
            "score": f"{int(goals_h)}-{int(goals_a)}",
        })

    return tips
