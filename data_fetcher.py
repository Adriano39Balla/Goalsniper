import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from config import API_FOOTBALL_API_KEY, LEAGUE_ID, SEASON, DATA_DIR
from utils import logger

HEADERS = {'x-apisports-key': API_FOOTBALL_API_KEY}
BASE_URL = 'https://v3.football.api-sports.io/'

def fetch_fixtures(status=None, from_date=None, to_date=None, league_id=LEAGUE_ID, season=SEASON):
    url = f"{BASE_URL}fixtures"
    params = {'league': league_id, 'season': season}
    if status:
        params['status'] = status
    if from_date:
        params['from'] = from_date
    if to_date:
        params['to'] = to_date

    try:
        resp = requests.get(url, headers=HEADERS, params=params)
        resp.raise_for_status()
        data = resp.json()
        fixtures = data.get('response', [])
        logger.info(f"Fetched {len(fixtures)} fixtures (status={status}).")
        return fixtures
    except Exception as e:
        logger.error(f"Error fetching fixtures: {e}")
        return []

def fetch_team_form(team_id, before_date, matches=5, season=SEASON):
    url = f"{BASE_URL}fixtures"
    params = {
        'team': team_id,
        'season': season,
        'to': before_date,
        'last': matches
    }
    try:
        resp = requests.get(url, headers=HEADERS, params=params)
        resp.raise_for_status()
        data = resp.json()
        fixtures = data.get('response', [])
        if not fixtures:
            return None

        total_scored = 0
        total_conceded = 0
        count = 0
        for match in fixtures:
            goals = match['goals']
            teams = match['teams']
            if goals['home'] is None or goals['away'] is None:
                continue
            if teams['home']['id'] == team_id:
                scored = goals['home']
                conceded = goals['away']
            else:
                scored = goals['away']
                conceded = goals['home']
            total_scored += scored
            total_conceded += conceded
            count += 1
        if count == 0:
            return None
        return {'avg_scored': total_scored / count, 'avg_conceded': total_conceded / count}
    except Exception as e:
        logger.error(f"Error fetching team form for {team_id}: {e}")
        return None

# Placeholder for fetching odds from API-Football or other source
def fetch_odds(fixture_id):
    # To implement: fetch odds for fixture_id and return dict
    return {}
