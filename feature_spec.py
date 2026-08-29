"""
goalsniper — shared feature specification.

WHY THIS FILE EXISTS
--------------------
Previously main.py's extract_features() and train_models.py's load_inplay_data()
each contained their own copy of the same ~40 derivations. Any divergence between
them silently breaks train/serve parity, which is exactly the class of bug that
produced the "shots on target key never matched" and "weights multiplied by 0.0"
failures. Both paths now call the SAME functions in this module, so drift is
structurally impossible rather than merely discouraged.

DESIGN NOTES ON THE FEATURE LISTS
---------------------------------
The old FEATURES (64) / PRE_FEATURES (~130) lists contained many exactly
collinear or exactly duplicated columns. Under L2 regularization, perfectly
collinear features split their shared effect arbitrarily, which makes individual
coefficients (and therefore `feature_importance`) meaningless. Removed:

  - pm_away_adv_rating       == pm_rating_a                     (exact duplicate)
  - pm_attack_strength_h/a   == pm_gf_h / pm_gf_a               (exact duplicate)
  - pm_defense_strength_h/a  == pm_ga_h / pm_ga_a               (exact duplicate)
  - pm_home_adv_rating       == pm_rating_h + const             (collinear)
  - pm_expected_total        == (gf_h+gf_a+ga_h+ga_a)/2         (linear combo)
  - pm_expected_total_diff   == (gf_h+ga_a-gf_a-ga_h)/2         (linear combo)
  - pm_form_points_diff      == 3(win_h-win_a)+(draw_h-draw_a)  (linear combo)
  - pm_goal_difference_h/a   == gf - ga                         (linear combo)
  - pm_loss_h/a              == 1 - win - draw                  (linear combo)
  - momentum_score           == .5*xg_pm + .3*sot_pm + .2*sh_pm (linear combo)
  - attack_pressure_h/a/diff == .4*sot + .4*xg + .2*cor         (linear combo)
  - xg_efficiency_h/a        == goals - xg                      (linear combo)
  - match_minute_normalized  == minute/90                       (linear combo)
  - is_first_half            == 1 - is_second_half              (linear combo)
  - is_draw                  == 1 - is_leading_h - is_leading_a (linear combo)
  - the ~65 hardcoded-zero live features inside PRE_FEATURES     (constant columns)

Nonlinear derivations (ratios, products, indicators, absolute values) are kept:
those carry information a linear model cannot recover from the components.

Result: 45 in-play features and 25 prematch features, all of which vary and none
of which is a linear function of the others.

SCALING
-------
Values here are RAW and on wildly different natural scales (minute 0-120,
pm_rating_diff -400..400, is_* flags 0/1). train_models.py fits a StandardScaler
and PERSISTS mean/scale inside the model blob; main.py applies the identical
transform before scoring. Do not add clipping or rescaling here unless it is
applied to both paths, which — since both paths call this module — it now
automatically would be.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# ───────── Constants shared by serving and training ─────────
ELO_DEFAULT = float(os.getenv("ELO_DEFAULT", "1500.0"))
ELO_K = float(os.getenv("ELO_K", "20.0"))
ELO_HOME_ADV = float(os.getenv("ELO_HOME_ADV", "60.0"))
FORM_DECAY_RATE = float(os.getenv("FORM_DECAY_RATE", "0.8"))

MIN_COUNT_DENOM = 1.0
MIN_XG_DENOM = 0.1
FINISHED_STATUSES = {"FT", "AET", "PEN"}

DEFAULT_LEAGUE_RATES: Dict[str, float] = {"btts": 0.5, "ov25": 0.5, "ov35": 0.3}

# How much the true probabilities of a market's selections sum to.
#
# This is NOT always 1.0, and getting it wrong silently halves every fair price.
# Double Chance selections are not mutually exclusive — 1X, X2 and 12 each cover
# two of the three outcomes, so P(1X) + P(X2) + P(12) = 2(P(H)+P(D)+P(A)) = 2.0.
# See devig().
MARKET_PROBABILITY_TOTAL: Dict[str, float] = {
    "1X2": 1.0,
    "BTTS": 1.0,
    "DNB": 1.0,   # two mutually exclusive outcomes, draw is void
    "DC": 2.0,    # three selections, each covering two of three outcomes
}


# ───────── In-play ─────────

# The raw fields a tip_snapshot MUST persist for build_inplay_features() to be
# reproducible at training time. main.py writes exactly these keys.
RAW_INPLAY_KEYS: List[str] = [
    "minute",
    "goals_h", "goals_a",
    "xg_h", "xg_a",
    "sot_h", "sot_a",
    "cor_h", "cor_a",
    "pos_h", "pos_a",
    "red_h", "red_a",
    "total_shots_h", "total_shots_a",
    "shots_inside_h", "shots_inside_a",
    "fouls_h", "fouls_a",
]

FEATURES: List[str] = [
    "minute",
    "goals_sum", "goals_diff",
    "xg_sum", "xg_diff",
    "sot_sum", "sot_diff",
    "cor_sum", "cor_diff",
    "pos_diff",
    "red_sum", "red_diff",
    "total_shots_sum", "total_shots_diff",
    "shots_inside_sum", "shots_inside_diff",
    "fouls_sum",
    "goals_per_minute", "xg_per_minute", "sot_per_minute", "shots_per_minute",
    "shot_accuracy_h", "shot_accuracy_a",
    "shot_quality_h", "shot_quality_a",
    "conversion_rate_h", "conversion_rate_a",
    "game_control_h", "game_control_a",
    "is_second_half", "is_final_15",
    "score_margin", "is_leading_h", "is_leading_a", "is_goalfest",
    "fouls_per_minute",
    "discipline_score_h", "discipline_score_a",
    "possession_xg_interaction_h", "possession_xg_interaction_a",
    "sot_xg_ratio_h", "sot_xg_ratio_a",
    "league_btts_rate", "league_ov25_rate", "league_ov35_rate",
]

PRE_FEATURES: List[str] = [
    "pm_gf_h", "pm_ga_h", "pm_gf_a", "pm_ga_a",
    "pm_win_h", "pm_draw_h", "pm_win_a", "pm_draw_a",
    "pm_ov25_h", "pm_ov35_h", "pm_btts_h",
    "pm_ov25_a", "pm_ov35_a", "pm_btts_a",
    "pm_ov25_h2h", "pm_btts_h2h",
    "pm_home_wins_h2h", "pm_away_wins_h2h",
    "pm_rating_diff", "pm_rating_mean",
    "pm_rest_diff",
    "pm_attack_defense_ratio",
    "pm_league_btts_rate", "pm_league_ov25_rate", "pm_league_ov35_rate",
]

# Which feature holds each league base rate, per phase. Used by the training
# loaders to overwrite whatever was stored at harvest time with a rate computed
# from TRAINING ROWS ONLY (see train_models._compute_league_rate_map).
LEAGUE_RATE_FIELDS_INPLAY = {
    "btts": "league_btts_rate", "ov25": "league_ov25_rate", "ov35": "league_ov35_rate",
}
LEAGUE_RATE_FIELDS_PREMATCH = {
    "btts": "pm_league_btts_rate", "ov25": "pm_league_ov25_rate", "ov35": "pm_league_ov35_rate",
}


def _f(v: Any) -> float:
    """Coerce an API value to float, tolerating '54%' and None."""
    try:
        if isinstance(v, str):
            v = v.strip()
            if v.endswith("%"):
                v = v[:-1]
        return float(v or 0)
    except Exception:
        return 0.0


def build_inplay_features(raw: Dict[str, Any], league_rates: Dict[str, float]) -> Dict[str, float]:
    """
    Build the full in-play feature vector from RAW_INPLAY_KEYS.

    Called by main.py.extract_features() (raw pulled live from the API) and by
    train_models.load_inplay_data() (raw pulled from the stored snapshot). Same
    code, same numbers, by construction.
    """
    r = {k: _f(raw.get(k)) for k in RAW_INPLAY_KEYS}
    f: Dict[str, float] = {}

    minute = r["minute"]
    f["minute"] = minute

    f["goals_sum"] = r["goals_h"] + r["goals_a"]
    f["goals_diff"] = r["goals_h"] - r["goals_a"]
    f["xg_sum"] = r["xg_h"] + r["xg_a"]
    f["xg_diff"] = r["xg_h"] - r["xg_a"]
    f["sot_sum"] = r["sot_h"] + r["sot_a"]
    f["sot_diff"] = r["sot_h"] - r["sot_a"]
    f["cor_sum"] = r["cor_h"] + r["cor_a"]
    f["cor_diff"] = r["cor_h"] - r["cor_a"]
    f["pos_diff"] = r["pos_h"] - r["pos_a"]
    f["red_sum"] = r["red_h"] + r["red_a"]
    f["red_diff"] = r["red_h"] - r["red_a"]
    f["total_shots_sum"] = r["total_shots_h"] + r["total_shots_a"]
    f["total_shots_diff"] = r["total_shots_h"] - r["total_shots_a"]
    f["shots_inside_sum"] = r["shots_inside_h"] + r["shots_inside_a"]
    f["shots_inside_diff"] = r["shots_inside_h"] - r["shots_inside_a"]
    f["fouls_sum"] = r["fouls_h"] + r["fouls_a"]

    if minute > 0:
        f["goals_per_minute"] = f["goals_sum"] / minute
        f["xg_per_minute"] = f["xg_sum"] / minute
        f["sot_per_minute"] = f["sot_sum"] / minute
        f["shots_per_minute"] = f["total_shots_sum"] / minute
    else:
        f["goals_per_minute"] = f["xg_per_minute"] = 0.0
        f["sot_per_minute"] = f["shots_per_minute"] = 0.0

    f["shot_accuracy_h"] = r["sot_h"] / max(r["total_shots_h"], MIN_COUNT_DENOM)
    f["shot_accuracy_a"] = r["sot_a"] / max(r["total_shots_a"], MIN_COUNT_DENOM)
    f["shot_quality_h"] = r["shots_inside_h"] / max(r["total_shots_h"], MIN_COUNT_DENOM)
    f["shot_quality_a"] = r["shots_inside_a"] / max(r["total_shots_a"], MIN_COUNT_DENOM)
    f["conversion_rate_h"] = r["goals_h"] / max(r["sot_h"], MIN_COUNT_DENOM)
    f["conversion_rate_a"] = r["goals_a"] / max(r["sot_a"], MIN_COUNT_DENOM)

    # Nonlinear: possession x attacking output. Kept because a linear model
    # cannot construct a product from the components.
    ap_h = 0.4 * r["sot_h"] + 0.4 * r["xg_h"] + 0.2 * r["cor_h"]
    ap_a = 0.4 * r["sot_a"] + 0.4 * r["xg_a"] + 0.2 * r["cor_a"]
    f["game_control_h"] = (r["pos_h"] / 100.0) * ap_h
    f["game_control_a"] = (r["pos_a"] / 100.0) * ap_a

    f["is_second_half"] = 1.0 if minute > 45 else 0.0
    f["is_final_15"] = 1.0 if minute > 75 else 0.0

    f["score_margin"] = abs(f["goals_diff"])
    f["is_leading_h"] = 1.0 if r["goals_h"] > r["goals_a"] else 0.0
    f["is_leading_a"] = 1.0 if r["goals_a"] > r["goals_h"] else 0.0
    f["is_goalfest"] = 1.0 if f["goals_sum"] >= 3 else 0.0

    f["fouls_per_minute"] = f["fouls_sum"] / max(minute, MIN_COUNT_DENOM)
    f["discipline_score_h"] = 1.0 / max(r["fouls_h"] + r["red_h"] * 10.0, MIN_COUNT_DENOM)
    f["discipline_score_a"] = 1.0 / max(r["fouls_a"] + r["red_a"] * 10.0, MIN_COUNT_DENOM)

    f["possession_xg_interaction_h"] = (r["pos_h"] / 100.0) * r["xg_h"]
    f["possession_xg_interaction_a"] = (r["pos_a"] / 100.0) * r["xg_a"]
    f["sot_xg_ratio_h"] = r["sot_h"] / max(r["xg_h"], MIN_XG_DENOM)
    f["sot_xg_ratio_a"] = r["sot_a"] / max(r["xg_a"], MIN_XG_DENOM)

    lr = league_rates or DEFAULT_LEAGUE_RATES
    f["league_btts_rate"] = float(lr.get("btts", DEFAULT_LEAGUE_RATES["btts"]))
    f["league_ov25_rate"] = float(lr.get("ov25", DEFAULT_LEAGUE_RATES["ov25"]))
    f["league_ov35_rate"] = float(lr.get("ov35", DEFAULT_LEAGUE_RATES["ov35"]))

    return {k: float(f.get(k, 0.0)) for k in FEATURES}


# ───────── Prematch form helpers ─────────

def fixture_ts(fx: dict) -> float:
    try:
        d = (fx.get("fixture") or {}).get("date")
        return datetime.fromisoformat((d or "").replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0


def _is_finished(g: dict) -> bool:
    st = (((g.get("fixture") or {}).get("status") or {}).get("short") or "").upper()
    return st in FINISHED_STATUSES


def decay_weights(games: List[dict]) -> Dict[int, float]:
    """
    Exponential recency weights keyed by id(game).

    FIX: ranks only FINISHED fixtures. Previously an abandoned or postponed
    fixture inside the last-5 window consumed the weight-1.0 slot and demoted
    the most recent real result to 0.8.

    Ranks by actual kickoff timestamp rather than list position, because the
    live path returns most-recent-first while the historical backfill builds its
    windows oldest-first.
    """
    finished = [g for g in games if _is_finished(g)]
    dated = sorted(((g, fixture_ts(g)) for g in finished), key=lambda x: x[1], reverse=True)
    return {id(g): FORM_DECAY_RATE ** i for i, (g, _) in enumerate(dated)}


def team_form_stats(team_id: int, games: List[dict]) -> Dict[str, Any]:
    w_map = decay_weights(games)
    gf = ga = win = draw = 0.0
    total_w = 0.0
    played = 0
    last_ts: Optional[float] = None

    for g in games:
        if not _is_finished(g):
            continue
        th = ((g.get("teams") or {}).get("home") or {}).get("id")
        ta = ((g.get("teams") or {}).get("away") or {}).get("id")
        gh = int((g.get("goals") or {}).get("home") or 0)
        ga_ = int((g.get("goals") or {}).get("away") or 0)
        if team_id == th:
            my, opp = gh, ga_
        elif team_id == ta:
            my, opp = ga_, gh
        else:
            continue
        w = w_map.get(id(g), 1.0)
        gf += my * w
        ga += opp * w
        total_w += w
        played += 1
        if my > opp:
            win += w
        elif my == opp:
            draw += w
        ts = fixture_ts(g)
        if ts and (last_ts is None or ts > last_ts):
            last_ts = ts

    if played == 0 or total_w <= 0:
        return {"gf": 0.0, "ga": 0.0, "win": 0.0, "draw": 0.0, "played": 0, "last_ts": None}
    return {"gf": gf / total_w, "ga": ga / total_w, "win": win / total_w,
            "draw": draw / total_w, "played": played, "last_ts": last_ts}


def rate_totals(games: List[dict]) -> Tuple[float, float, float]:
    """Recency-weighted Over 2.5 / Over 3.5 / BTTS rates for a fixture window."""
    w_map = decay_weights(games)
    ov25 = ov35 = btts = 0.0
    total_w = 0.0
    for g in games:
        if not _is_finished(g):
            continue
        gh = int((g.get("goals") or {}).get("home") or 0)
        ga = int((g.get("goals") or {}).get("away") or 0)
        w = w_map.get(id(g), 1.0)
        total_w += w
        if gh + ga > 2:
            ov25 += w
        if gh + ga > 3:
            ov35 += w
        if gh > 0 and ga > 0:
            btts += w
    if total_w <= 0:
        return 0.0, 0.0, 0.0
    return ov25 / total_w, ov35 / total_w, btts / total_w


def h2h_counts(h2h: List[dict], home_id: int, away_id: int) -> Tuple[float, float, float]:
    w_map = decay_weights(h2h)
    hw = aw = dr = 0.0
    total_w = 0.0
    for g in h2h:
        if not _is_finished(g):
            continue
        th = ((g.get("teams") or {}).get("home") or {}).get("id")
        ta = ((g.get("teams") or {}).get("away") or {}).get("id")
        gh = int((g.get("goals") or {}).get("home") or 0)
        ga = int((g.get("goals") or {}).get("away") or 0)
        w = w_map.get(id(g), 1.0)
        total_w += w
        if gh == ga:
            dr += w
        else:
            winner_id = th if gh > ga else ta
            if winner_id == home_id:
                hw += w
            elif winner_id == away_id:
                aw += w
    if total_w <= 0:
        return 0.0, 0.0, 0.0
    return hw / total_w, aw / total_w, dr / total_w


def assemble_prematch_features(
    home_id: int,
    away_id: int,
    last_h: List[dict],
    last_a: List[dict],
    h2h: List[dict],
    kickoff_ts: float,
    rating_h: float,
    rating_a: float,
    league_rates: Dict[str, float],
) -> Dict[str, float]:
    """
    Build the prematch feature vector. Single implementation shared by the live
    prematch scan, Match of the Day, and the historical season backfill.
    """
    ov25_h, ov35_h, btts_h = rate_totals(last_h)
    ov25_a, ov35_a, btts_a = rate_totals(last_a)
    ov25_h2h, _ov35_h2h, btts_h2h = rate_totals(h2h)
    hw_h2h, aw_h2h, _dr_h2h = h2h_counts(h2h, home_id, away_id)

    form_h = team_form_stats(home_id, last_h)
    form_a = team_form_stats(away_id, last_a)

    rest_h = rest_a = 3.0  # neutral default when the last fixture date is unknown
    if form_h["last_ts"]:
        rest_h = max(0.0, (kickoff_ts - form_h["last_ts"]) / 86400.0)
    if form_a["last_ts"]:
        rest_a = max(0.0, (kickoff_ts - form_a["last_ts"]) / 86400.0)

    lr = league_rates or DEFAULT_LEAGUE_RATES

    f: Dict[str, float] = {
        "pm_gf_h": form_h["gf"], "pm_ga_h": form_h["ga"],
        "pm_gf_a": form_a["gf"], "pm_ga_a": form_a["ga"],
        "pm_win_h": form_h["win"], "pm_draw_h": form_h["draw"],
        "pm_win_a": form_a["win"], "pm_draw_a": form_a["draw"],
        "pm_ov25_h": ov25_h, "pm_ov35_h": ov35_h, "pm_btts_h": btts_h,
        "pm_ov25_a": ov25_a, "pm_ov35_a": ov35_a, "pm_btts_a": btts_a,
        "pm_ov25_h2h": ov25_h2h, "pm_btts_h2h": btts_h2h,
        "pm_home_wins_h2h": hw_h2h, "pm_away_wins_h2h": aw_h2h,
        # Home advantage is a constant and is absorbed by the intercept, so it
        # is deliberately NOT added into the rating diff (that was the old
        # pm_home_adv_rating duplicate).
        "pm_rating_diff": rating_h - rating_a,
        "pm_rating_mean": (rating_h + rating_a) / 2.0,
        "pm_rest_diff": rest_h - rest_a,
        "pm_attack_defense_ratio": (form_h["gf"] + form_a["gf"]) / max(form_h["ga"] + form_a["ga"], MIN_XG_DENOM),
        "pm_league_btts_rate": float(lr.get("btts", DEFAULT_LEAGUE_RATES["btts"])),
        "pm_league_ov25_rate": float(lr.get("ov25", DEFAULT_LEAGUE_RATES["ov25"])),
        "pm_league_ov35_rate": float(lr.get("ov35", DEFAULT_LEAGUE_RATES["ov35"])),
    }
    return {k: float(f.get(k, 0.0)) for k in PRE_FEATURES}


# ───────── Derived 1X2 markets ─────────

def derive_dc_dnb(p_home: float, p_draw: float, p_away: float) -> Dict[str, float]:
    """
    Double Chance and Draw No Bet probabilities from a normalised 1X2 triple.

    Single implementation so serving and training compute these identically —
    the same reason build_inplay_features() lives here.

    Draw No Bet is conditional on the draw not happening (the stake is returned
    on a draw), hence the ph+pa denominator. Double Chance is not conditional:
    1X simply covers two outcomes, so its probability is the plain sum.
    """
    s = max(p_home + p_draw + p_away, 1e-12)
    ph, pd, pa = p_home / s, p_draw / s, p_away / s
    dnb_s = max(ph + pa, 1e-12)
    return {
        "1X": ph + pd,
        "X2": pd + pa,
        "12": ph + pa,
        "DNB_Home": ph / dnb_s,
        "DNB_Away": pa / dnb_s,
    }


# ───────── Market maths (de-vig, EV, Kelly) ─────────

def devig(probs: Dict[str, float], market_total: float = 1.0) -> Dict[str, float]:
    """
    Multiplicative de-vig: rescale implied probabilities so the market sums to
    its true total, removing the bookmaker's overround.

    THE market_total ARGUMENT IS NOT COSMETIC. It defaults to 1.0, which is
    right for mutually exclusive markets (1X2, BTTS, Over/Under, Draw No Bet).
    It is WRONG for Double Chance: 1X, X2 and 12 each cover two of three
    outcomes, so their true probabilities sum to 2.0. Normalising DC to 1.0
    halves every fair price — a fair P(1X) of 0.72 comes out as 0.36, which
    then reads as a +36 percentage-point "edge" and trips the model-sanity cap
    on every single Double Chance candidate.

    See MARKET_PROBABILITY_TOTAL for the per-market values.

    This is the minimum correct treatment. Shin's method or the power method
    handle favourite-longshot bias better and are worth revisiting once you have
    enough closing-line history to test which fits your books.
    """
    s = sum(v for v in probs.values() if v and v > 0)
    if s <= 0:
        return {}
    scale = float(market_total) / s
    return {k: (v * scale) for k, v in probs.items() if v and v > 0}


def ev(prob: float, odds: float) -> float:
    """Expected value per unit staked, as a decimal (0.05 = +5%)."""
    return float(prob) * max(0.0, float(odds)) - 1.0


def kelly_fraction(prob: float, odds: float) -> float:
    """
    Full-Kelly fraction of bankroll. Negative means no bet.
    b = odds - 1;  f* = (p*b - (1-p)) / b
    """
    b = float(odds) - 1.0
    if b <= 0:
        return 0.0
    p = float(prob)
    return (p * b - (1.0 - p)) / b


def enforce_ou_monotonicity(line_probs: List[Tuple[float, float]]) -> Dict[float, float]:
    """
    Project independently-scored Over/Under lines onto the one constraint they
    must obey: P(Over line) is non-increasing as the line rises. Over 3.5 can
    never be more likely than Over 2.5 — it is a strict subset of it — but
    OU_2.5 and OU_3.5 are separate logistic heads with no such constraint
    baked in, and in practice they do occasionally cross.

    This is pool-adjacent-violators (isotonic regression) for a non-increasing
    sequence: the least-squares projection of the raw probabilities onto the
    monotone cone, sorted by line. When the inputs are already coherent this
    is a no-op; only genuine crossings get pulled toward each other.
    """
    items = sorted(line_probs, key=lambda x: x[0])
    if len(items) <= 1:
        return dict(items)
    blocks: List[List[float]] = []  # each: [sum_of_probs, count]
    for _line, p in items:
        blocks.append([p, 1.0])
        while len(blocks) >= 2 and (blocks[-2][0] / blocks[-2][1]) < (blocks[-1][0] / blocks[-1][1]):
            b2 = blocks.pop()
            b1 = blocks.pop()
            blocks.append([b1[0] + b2[0], b1[1] + b2[1]])
    out_vals: List[float] = []
    for total, count in blocks:
        out_vals.extend([total / count] * int(count))
    return {line: v for (line, _), v in zip(items, out_vals)}
