import aiohttp
import asyncio
from typing import Dict, List, Optional
import os
from datetime import datetime
from .logger import logger

class APIFootballClient:
    """API Football client with rate limiting and error handling"""
    
    def __init__(self):
        self.api_key = os.getenv('API_FOOTBALL_KEY')
        self.base_url = f"https://{os.getenv('API_FOOTBALL_HOST', 'v3.football.api-sports.io')}"
        self.headers = {
            'x-rapidapi-host': os.getenv('API_FOOTBALL_HOST', 'v3.football.api-sports.io'),
            'x-rapidapi-key': self.api_key
        }
        self.session = None
        self.rate_limit_remaining = 100
        self.rate_limit_reset = datetime.now()
    
    async def get_session(self):
        """Get or create aiohttp session"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(
                headers=self.headers,
                timeout=timeout
            )
        return self.session
    
    async def make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make API request with rate limiting"""
        
        await self._check_rate_limit()
        
        try:
            session = await self.get_session()
            url = f"{self.base_url}{endpoint}"
            
            async with session.get(url, params=params) as response:
                # Update rate limit headers
                self._update_rate_limits(response.headers)
                
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    logger.error(f"API request failed: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"API request error: {e}")
            return None
    
    async def get_live_matches(self, league_ids: Optional[List[int]] = None) -> List[Dict]:
        """Get currently live matches"""
        
        endpoint = "/fixtures"
        params = {'live': 'all'}
        
        if league_ids:
            params['league'] = ','.join(map(str, league_ids))
        
        response = await self.make_request(endpoint, params)
        
        if response and response.get('response'):
            return response['response']
        
        return []
    
    async def get_match_statistics(self, match_id: int) -> Optional[Dict]:
        """Get detailed match statistics"""
        
        endpoint = f"/fixtures/statistics"
        params = {'fixture': match_id}
        
        response = await self.make_request(endpoint, params)
        
        if response and response.get('response'):
            return response['response'][0] if response['response'] else None
        
        return None
    
    async def get_match_events(self, match_id: int) -> List[Dict]:
        """Get match events (goals, cards, substitutions)"""
        
        endpoint = f"/fixtures/events"
        params = {'fixture': match_id}
        
        response = await self.make_request(endpoint, params)
        
        if response and response.get('response'):
            return response['response']
        
        return []
    
    async def get_team_statistics(self, team_id: int, league_id: int, season: int = 2024) -> Dict:
        """Get team statistics for a season"""
        
        endpoint = f"/teams/statistics"
        params = {
            'team': team_id,
            'league': league_id,
            'season': season
        }
        
        response = await self.make_request(endpoint, params)
        
        if response and response.get('response'):
            return response['response']
        
        return {}
    
    def _update_rate_limits(self, headers: Dict):
        """Update rate limit tracking from headers"""
        
        remaining = headers.get('x-ratelimit-requests-remaining')
        reset = headers.get('x-ratelimit-requests-reset')
        
        if remaining:
            self.rate_limit_remaining = int(remaining)
        
        if reset:
            self.rate_limit_reset = datetime.fromtimestamp(int(reset))
    
    async def _check_rate_limit(self):
        """Check and respect rate limits"""
        
        if self.rate_limit_remaining < 5:
            wait_time = (self.rate_limit_reset - datetime.now()).total_seconds()
            if wait_time > 0:
                logger.warning(f"Rate limit low. Waiting {wait_time:.0f} seconds...")
                await asyncio.sleep(wait_time)
    
    async def close(self):
        """Close the session"""
        if self.session and not self.session.closed:
            await self.session.close()
