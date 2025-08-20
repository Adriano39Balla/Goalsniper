# app/features.py
import math
from typing import Dict, Any, List


def _num(v) -> float:
    """
    Convert to float safely. Supports strings like '54%'.
    """
    try:
        if isinstance(v, str) and v.endswith('%'):
            return float(v[:-1])
        return float(v or 0)
    except Exception:
        return 0.0


def _pos_pct(v) -> float:
    """
    Normalize possession % value to float.
    """
    try:
        return float(str(v).replace('%', '').strip() or 0)
    except Exception:
        return 0.0


def extract_features(match: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract numeric features from a match snapshot:
    - goals
    - xG
    - shots on target
    - corners
    - possession
    - red cards
    """
    home_name = match["teams"]["home"]["name"]
    away_name = match["teams"]["away"]["name"]
    gh = match["goals"]["home"] or 0
    ga = match["goals"]["away"] or 0
    minute = int((match.get("fixture", {}).get("status", {}) or {}).get("elapsed") or 0)

    # collect stats into a team -> dict
    stats_blocks = match.get("statistics") or []
    stats: Dict[str, Dict[str, Any]] = {}
    for s in stats_blocks:
        tname = (s.get("team") or {}).get("name")
        if tname:
            stats[tname] = {i["type"]: i["value"] for i in (s.get("statistics") or [])}

    sh = stats.get(home_name, {})
    sa = stats.get(away_name, {})

    # basic stats
    xg_h = _num(sh.get("Expected Goals", 0))
    xg_a = _num(sa.get("Expected Goals", 0))
    sot_h = _num(sh.get("Shots on Target", 0))
    sot_a = _num(sa.get("Shots on Target", 0))
    cor_h = _num(sh.get("Corner Kicks", 0))
    cor_a = _num(sa.get("Corner Kicks", 0))
    pos_h = _pos_pct(sh.get("Ball Possession", 0))
    pos_a = _pos_pct(sa.get("Ball Possession", 0))

    # red cards from events
    red_h = 0
    red_a = 0
    for ev in (match.get("events") or []):
        try:
            etype = (ev.get("type") or "").lower()
            edetail = (ev.get("detail") or "").lower()
            tname = (ev.get("team") or {}).get("name") or ""
            if etype == "card" and "red" in edetail:
                if tname == home_name:
                    red_h += 1
                elif tname == away_name:
                    red_a += 1
        except Exception:
            pass

    return {
        "minute": float(minute),
        "goals_h": float(gh), "goals_a": float(ga),
        "goals_sum": float(gh + ga), "goals_diff": float(gh - ga),
        "xg_h": float(xg_h), "xg_a": float(xg_a), "xg_sum": float(xg_h + xg_a),
        "sot_h": float(sot_h), "sot_a": float(sot_a), "sot_sum": float(sot_h + sot_a),
        "cor_h": float(cor_h), "cor_a": float(cor_a), "cor_sum": float(cor_h + cor_a),
        "pos_h": float(pos_h), "pos_a": float(pos_a),
        "red_h": float(red_h), "red_a": float(red_a),
    }


def stats_coverage_ok(feat: Dict[str, float], minute: int, require_stats_minute: int, require_data_fields: int) -> bool:
    """
    Validate whether a snapshot has enough live stats to be usable.
    - Before `require_stats_minute` → always True
    - After → need at least `require_data_fields` nonzero among xG, SoT, Corners, Possession
    """
    if minute < require_stats_minute:
        return True

    fields = [
        feat.get("xg_sum", 0.0),
        feat.get("sot_sum", 0.0),
        feat.get("cor_sum", 0.0),
        max(feat.get("pos_h", 0.0), feat.get("pos_a", 0.0)),
    ]
    nonzero = sum(1 for v in fields if (v or 0) > 0)
    return nonzero >= max(0, require_data_fields)
