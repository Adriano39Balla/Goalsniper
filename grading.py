"""Pure regulation-time market grading. Missing results never become 0–0."""
from typing import Any, Dict, Optional

def _parse_ou_line_from_suggestion(s: str) -> Optional[float]:
    for tok in (s or "").split():
        try:
            return float(tok)
        except Exception:
            continue
    return None


def _tip_outcome_for_result(suggestion: str, res: Dict[str, Any]) -> Optional[int]:
    """1 = win, 0 = loss, None = push/void or ungradeable."""
    if res.get('final_goals_h') is None or res.get('final_goals_a') is None:
        return None
    gh = int(res['final_goals_h'])
    ga = int(res['final_goals_a'])
    if gh < 0 or ga < 0:
        return None
    total = gh + ga
    btts = int(gh > 0 and ga > 0)
    s = (suggestion or "").strip()
    if s.startswith("Over") or s.startswith("Under"):
        line = _parse_ou_line_from_suggestion(s)
        if line is None:
            return None
        if abs(total - line) < 1e-9:
            return None
        return int(total > line) if s.startswith("Over") else int(total < line)
    if s == "BTTS: Yes":
        return 1 if btts == 1 else 0
    if s == "BTTS: No":
        return 1 if btts == 0 else 0
    if s == "Home Win":
        return 1 if gh > ga else 0
    if s == "Away Win":
        return 1 if ga > gh else 0
    if s == "Double Chance: 1X":
        return 1 if gh >= ga else 0
    if s == "Double Chance: X2":
        return 1 if ga >= gh else 0
    if s == "Double Chance: 12":
        return 1 if gh != ga else 0
    # Draw No Bet voids on a draw — stake returned, not a loss.
    if s == "Draw No Bet: Home":
        return None if gh == ga else int(gh > ga)
    if s == "Draw No Bet: Away":
        return None if gh == ga else int(ga > gh)
    return None
