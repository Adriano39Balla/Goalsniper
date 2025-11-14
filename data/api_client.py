import requests
import aiohttp
import asyncio
from typing import List, Dict, Any, Optional
import logging
from config.settings import settings

logger = logging.getLogger(__name__)

class APIFootballClient:
    def __init__(self):
        self.base_url = settings.API_FOOTBALL_URL
        self.headers = {
            'x-rapidapi-key': settings.API_FOOTBALL_KEY,
            'x-rapidapi-host': 'api-football-v1.p.rapidapi.com'
        }
        self.session = None
    
    async def get_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession(headers=self.headers)
        return self.session
    
    async def make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make async request to API-Football"""
        session = await self.get_session()
        try:
            async with session.get(f"{self.base_url}/{endpoint}", params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('response', [])
                else:
                    logger.error(f"API request failed: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Request error: {e}")
            return None
    
    async def get_live_fixtures(self) -> List[Dict]:
        """Get live fixtures"""
        params = {'live': 'all'}
        data = await self.make_request('fixtures', params)
        return data or []
    
    async def get_fixture_stats(self, fixture_id: int) -> Dict:
        """Get detailed fixture statistics"""
        params = {'fixture': fixture_id}
        data = await self.make_request('fixtures/statistics', params)
        return data[0] if data else {}
    
    async def get_fixture_events(self, fixture_id: int) -> List[Dict]:
        """Get fixture events (goals, cards, etc.)"""
        params = {'fixture': fixture_id}
        data = await self.make_request('fixtures/events', params)
        return data or []
    
    async def get_historical_fixtures(self, league_id: int, season: int, page: int = 1) -> List[Dict]:
        """Get historical fixtures for training"""
        params = {
            'league': league_id,
            'season': season,
            'page': page
        }
        data = await self.make_request('fixtures', params)
        return data or []
    
    async def get_league_standings(self, league_id: int, season: int) -> Dict:
        """Get league standings"""
        params = {
            'league': league_id,
            'season': season
        }
        data = await self.make_request('standings', params)
        return data[0] if data else {}

api_client = APIFootballClient()
