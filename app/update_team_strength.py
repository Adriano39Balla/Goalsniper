from datetime import datetime, timedelta
from collections import defaultdict

from app.supabase_db import get_supabase

sb = get_supabase()


def log(msg: str):
    print(f"[STRENGTH] {msg}")


def main():
    # get recent fixtures with final scores
    resp = (
        sb.table("fixtures")
        .select("fixture_id, home_team_id, away_team_id, home_goals, away_goals")
        .not_.is_("home_goals", None)
        .not_.is_("away_goals", None)
        .order("timestamp", desc=True)
        .limit(2000)
        .execute()
    )
    fixtures = resp.data or []
    if not fixtures:
        log("No fixtures with results.")
        return

    team_stats = defaultdict(lambda: {"gf": 0, "ga": 0, "matches": 0})

    total_goals = 0
    total_matches = 0

    for fx in fixtures:
        hg = fx["home_goals"]
        ag = fx["away_goals"]
        ht = fx["home_team_id"]
        at = fx["away_team_id"]

        # update team stats
        team_stats[ht]["gf"] += hg
        team_stats[ht]["ga"] += ag
        team_stats[ht]["matches"] += 1

        team_stats[at]["gf"] += ag
        team_stats[at]["ga"] += hg
        team_stats[at]["matches"] += 1

        total_goals += (hg + ag)
        total_matches += 1

    if total_matches == 0:
        log("No valid matches.")
        return

    league_avg_goals = total_goals / total_matches
    if league_avg_goals <= 0:
        league_avg_goals = 2.6

    log(f"League avg goals: {league_avg_goals:.2f}")

    # upsert strengths
    for team_id, s in team_stats.items():
        m = s["matches"]
        if m < 5:
            continue  # not enough data yet

        gf_avg = s["gf"] / m
        ga_avg = s["ga"] / m

        attack_rating = gf_avg / league_avg_goals
        defense_rating = league_avg_goals / max(ga_avg, 0.1)
        btts_rating = (gf_avg + ga_avg) / (2 * league_avg_goals)

        sb.table("team_strengths").upsert({
            "team_id": team_id,
            "attack_rating": attack_rating,
            "defense_rating": defense_rating,
            "btts_rating": btts_rating,
            "matches": m,
            "last_updated": datetime.utcnow().isoformat()
        }, on_conflict="team_id").execute()

    log(f"Updated strengths for {len(team_stats)} teams.")


if __name__ == "__main__":
    main()
