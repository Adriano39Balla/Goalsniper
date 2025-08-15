import pandas as pd
from data_fetcher import fetch_team_form
from utils import logger

def prepare_dataset(fixtures):
    rows = []
    logger.info(f"Preparing dataset from {len(fixtures)} fixtures")

    for fixture in fixtures:
        try:
            goals_home = fixture['goals']['home']
            goals_away = fixture['goals']['away']
            if goals_home is None or goals_away is None:
                continue

            home_team = fixture['teams']['home']
            away_team = fixture['teams']['away']

            # Targets
            outcome = 'H' if goals_home > goals_away else 'D' if goals_home == goals_away else 'A'
            total_goals = goals_home + goals_away
            over_under = 'Over' if total_goals > 2.5 else 'Under'
            btts = 'Yes' if (goals_home > 0 and goals_away > 0) else 'No'

            fixture_date = fixture['fixture']['date'][:10]

            home_stats = fetch_team_form(home_team['id'], fixture_date)
            away_stats = fetch_team_form(away_team['id'], fixture_date)
            if home_stats is None or away_stats is None:
                continue

            row = {
                'home_team_id': home_team['id'],
                'away_team_id': away_team['id'],
                'home_avg_scored': home_stats['avg_scored'],
                'home_avg_conceded': home_stats['avg_conceded'],
                'away_avg_scored': away_stats['avg_scored'],
                'away_avg_conceded': away_stats['avg_conceded'],
                'outcome': outcome,
                'over_under': over_under,
                'btts': btts
            }
            rows.append(row)
        except Exception as e:
            logger.error(f"Error preparing row: {e}")
            continue
    return pd.DataFrame(rows)
