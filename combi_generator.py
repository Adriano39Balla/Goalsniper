from src.db import get_training_data

def generate_kombi_tips(max_tips: int = 3) -> list[dict]:
    data = get_training_data()
    if not data:
        return []

    # Filter only ✅ tips with high confidence
    valid = [
        {
            "team": t,
            "league": l,
            "tip": tip,
            "confidence": conf
        }
        for t, l, tip, conf, res in data
        if res == "✅" and conf >= 75
    ]

    if len(valid) < max_tips:
        return []

    # Select diverse leagues or teams
    seen_leagues = set()
    combo = []

    for item in valid:
        if item["league"] in seen_leagues:
            continue
        combo.append(item)
        seen_leagues.add(item["league"])
        if len(combo) == max_tips:
            break

    return combo
