import requests
import time
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from config import Config

logger = logging.getLogger(__name__)

class APIFootballService:
    def __init__(self):
        self.base_url = Config.API_FOOTBALL_BASE_URL
        self.api_key = Config.API_FOOTBALL_KEY
        self.headers = {
            'x-apisports-key': self.api_key
        }
        self.last_request_time = 0

    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Make a rate-limited request to the API"""
        try:
            # Rate limiting
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < Config.REQUEST_DELAY:
                time.sleep(Config.REQUEST_DELAY - time_since_last)

            url = f"{self.base_url}/{endpoint}"
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            self.last_request_time = time.time()

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API request failed: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            logger.error(f"Error making API request: {str(e)}")
            return None

    def get_live_matches(self) -> List[Dict]:
        """Get all live matches"""
        try:
            data = self._make_request('fixtures', {'live': 'all'})
            if data and 'response' in data:
                return [
                    {
                        'id': fixture['fixture']['id'],
                        'home_team': fixture['teams']['home']['name'],
                        'away_team': fixture['teams']['away']['name'],
                        'league': fixture['league']['name'],
                        'league_id': fixture['league']['id'],
                        'status': fixture['fixture']['status']['short'],
                        'elapsed': fixture['fixture']['status']['elapsed'],
                        'home_score': fixture['goals']['home'],
                        'away_score': fixture['goals']['away'],
                        'date': fixture['fixture']['date']
                    }
                    for fixture in data['response']
                    if fixture['league']['id'] in Config.MAJOR_LEAGUES
                ]
            return []
        except Exception as e:
            logger.error(f"Error fetching live matches: {str(e)}")
            return []

    def get_match_statistics(self, fixture_id: int) -> Optional[Dict]:
        """Get detailed statistics for a specific match"""
        try:
            data = self._make_request('fixtures/statistics', {'fixture': fixture_id})
            if data and 'response' in data:
                stats = {}
                for team_stats in data['response']:
                    team_name = team_stats['team']['name']
                    team_data = {
                        stat['type']: self._parse_stat_value(stat['value'])
                        for stat in team_stats['statistics']
                    }
                    stats[team_name] = team_data
                return stats
            return None
        except Exception as e:
            logger.error(f"Error fetching match statistics: {str(e)}")
            return None

    def get_upcoming_matches(self, days_ahead: int = 1) -> List[Dict]:
        """Get upcoming matches for the next N days"""
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            end_date = (datetime.now() + timedelta(days=days_ahead)).strftime('%Y-%m-%d')

            upcoming_matches = []
            for league_id in Config.MAJOR_LEAGUES:
                data = self._make_request('fixtures', {
                    'league': league_id,
                    'from': today,
                    'to': end_date
                })

                if data and 'response' in data:
                    for fixture in data['response']:
                        if fixture['fixture']['status']['short'] == 'NS':
                            upcoming_matches.append({
                                'id': fixture['fixture']['id'],
                                'home_team': fixture['teams']['home']['name'],
                                'away_team': fixture['teams']['away']['name'],
                                'league': fixture['league']['name'],
                                'league_id': fixture['league']['id'],
                                'date': fixture['fixture']['date'],
                                'timestamp': fixture['fixture']['timestamp']
                            })

            return upcoming_matches
        except Exception as e:
            logger.error(f"Error fetching upcoming matches: {str(e)}")
            return []

    def _parse_stat_value(self, value):
        """Convert percentage strings to float, or return raw value"""
        if isinstance(value, str) and '%' in value:
            try:
                return float(value.replace('%', ''))
            except:
                return value
        return value
