from db import get_training_data
import logging

logger = logging.getLogger("uvicorn")

def generate_kombi_tips(max_tips: int = 3) -> list[dict]:
    data = get_training_data()
    if not data:
        logger.info("[Combi] No training data available.")
        return []

    valid = [
        {
            "team": team,
            "league": league,
            "tip": tip,
            "confidence": confidence
        }
        for team, league, tip, confidence, result in data
        if result == "✅" and confidence >= 75
    ]

    if len(valid) < max_tips:
        logger.info(f"[Combi] Not enough valid tips (found {len(valid)}, need {max_tips})")
        return []

    # Try to ensure different leagues
    seen_leagues = set()
    combo = []

    for item in valid:
        if item["league"] in seen_leagues:
            continue
        combo.append(item)
        seen_leagues.add(item["league"])
        if len(combo) == max_tips:
            break

    logger.info(f"[Combi] Returning {len(combo)} kombi tips")
    return combo
