"""
Football Data Pipeline
High-performance data ingestion from API-Football with focus on live in-play matches
"""

import aiohttp
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json

from loguru import logger
from config import settings


@dataclass
class LiveMatchData:
    """Structured live match data"""
    fixture_id: int
    league_id: int
    league_name: str
    home_team: str
    away_team: str
    elapsed_minutes: int
    status: str
    
    # Score
    home_goals: int
    away_goals: int
    
    # Statistics
    home_shots_total: Optional[int] = None
    away_shots_total: Optional[int] = None
    home_shots_on_target: Optional[int] = None
    away_shots_on_target: Optional[int] = None
    home_possession: Optional[int] = None
    away_possession: Optional[int] = None
    home_attacks: Optional[int] = None
    away_attacks: Optional[int] = None
    home_dangerous_attacks: Optional[int] = None
    away_dangerous_attacks: Optional[int] = None
    home_corners: Optional[int] = None
    away_corners: Optional[int] = None
    home_yellow_cards: Optional[int] = None
    away_yellow_cards: Optional[int] = None
    home_red_cards: Optional[int] = None
    away_red_cards: Optional[int] = None
    home_fouls: Optional[int] = None
    away_fouls: Optional[int] = None
    home_offsides: Optional[int] = None
    away_offsides: Optional[int] = None
    
    # Metadata
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for database storage"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class FootballDataPipeline:
    """
    High-performance data pipeline for fetching and processing live football data
    """
    
    def __init__(self):
        self.base_url = settings.API_FOOTBALL_BASE_URL
        self.api_key = settings.API_FOOTBALL_KEY
        self.headers = {
            'x-apisports-key': self.api_key
        }
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Cache for reducing API calls
        self.cache: Dict[str, Any] = {}
        self.cache_ttl = 30  # seconds
        
        # Rate limiting
        self.request_count = 0
        self.request_limit = 100  # per minute
        self.rate_limit_reset = datetime.now()
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(headers=self.headers)
        return self.session
    
    async def _check_rate_limit(self):
        """Check and enforce rate limiting"""
        now = datetime.now()
        
        if now > self.rate_limit_reset:
            self.request_count = 0
            self.rate_limit_reset = now + timedelta(minutes=1)
        
        if self.request_count >= self.request_limit:
            wait_time = (self.rate_limit_reset - now).total_seconds()
            logger.warning(f"Rate limit reached, waiting {wait_time:.1f}s")
            await asyncio.sleep(wait_time)
            self.request_count = 0
            self.rate_limit_reset = datetime.now() + timedelta(minutes=1)
        
        self.request_count += 1
    
    async def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make API request with error handling and caching"""
        cache_key = f"{endpoint}_{json.dumps(params, sort_keys=True)}"
        
        # Check cache
        if cache_key in self.cache:
            cached_data, cached_time = self.cache[cache_key]
            if (datetime.now() - cached_time).total_seconds() < self.cache_ttl:
                return cached_data
        
        # Rate limiting
        await self._check_rate_limit()
        
        url = f"{self.base_url}/{endpoint}"
        session = await self._get_session()
        
        try:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Cache the response
                    self.cache[cache_key] = (data, datetime.now())
                    
                    return data
                elif response.status == 429:
                    logger.error("API rate limit exceeded")
                    await asyncio.sleep(60)
                    return None
                else:
                    logger.error(f"API request failed: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Request error for {endpoint}: {e}")
            return None
    
    async def fetch_live_matches(self, date: Optional[str] = None) -> List[LiveMatchData]:
        """
        Fetch all live in-play matches
        """
        params = {}
        
        if date:
            params['date'] = date
        else:
            params['live'] = 'all'  # Only live matches
        
        data = await self._make_request('fixtures', params)
        
        if not data or 'response' not in data:
            logger.warning("No live matches data received")
            return []
        
        matches = []
        
        for fixture in data['response']:
            try:
                # Only process live matches (not pre-match)
                status = fixture['fixture']['status']['short']
                if status not in ['1H', '2H', 'HT', 'ET', 'BT', 'P']:
                    continue
                
                fixture_id = fixture['fixture']['id']
                
                # Fetch detailed statistics for this match
                stats = await self.fetch_match_statistics(fixture_id)
                
                match_data = self._parse_match_data(fixture, stats)
                if match_data:
                    matches.append(match_data)
                
            except Exception as e:
                logger.error(f"Error parsing match data: {e}")
        
        logger.info(f"Fetched {len(matches)} live matches")
        return matches
    
    async def fetch_match_statistics(self, fixture_id: int) -> Optional[Dict]:
        """
        Fetch detailed statistics for a specific match
        """
        data = await self._make_request('fixtures/statistics', {'fixture': fixture_id})
        
        if not data or 'response' not in data:
            return None
        
        return data['response']
    
    def _parse_match_data(self, fixture: Dict, stats: Optional[Dict]) -> Optional[LiveMatchData]:
        """
        Parse raw API data into structured LiveMatchData
        """
        try:
            fixture_info = fixture['fixture']
            teams = fixture['teams']
            goals = fixture['goals']
            score = fixture['score']
            
            # Basic match info
            match_data = LiveMatchData(
                fixture_id=fixture_info['id'],
                league_id=fixture['league']['id'],
                league_name=fixture['league']['name'],
                home_team=teams['home']['name'],
                away_team=teams['away']['name'],
                elapsed_minutes=fixture_info['status']['elapsed'] or 0,
                status=fixture_info['status']['short'],
                home_goals=goals['home'] or 0,
                away_goals=goals['away'] or 0
            )
            
            # Parse statistics if available
            if stats:
                stats_dict = self._parse_statistics(stats)
                
                # Update match data with statistics
                for key, value in stats_dict.items():
                    if hasattr(match_data, key):
                        setattr(match_data, key, value)
            
            return match_data
            
        except Exception as e:
            logger.error(f"Error parsing match data: {e}")
            return None
    
    def _parse_statistics(self, stats: List[Dict]) -> Dict[str, int]:
        """
        Parse statistics into flat dictionary
        """
        stats_dict = {}
        
        # Statistics mapping
        stat_mapping = {
            'Shots on Goal': ('shots_on_target', 'shots_on_target'),
            'Shots off Goal': ('shots_off_target', 'shots_off_target'),
            'Total Shots': ('shots_total', 'shots_total'),
            'Ball Possession': ('possession', 'possession'),
            'Corner Kicks': ('corners', 'corners'),
            'Fouls': ('fouls', 'fouls'),
            'Yellow Cards': ('yellow_cards', 'yellow_cards'),
            'Red Cards': ('red_cards', 'red_cards'),
            'Offsides': ('offsides', 'offsides'),
            'Total passes': ('passes_total', 'passes_total'),
            'Passes accurate': ('passes_accurate', 'passes_accurate'),
            'Passes %': ('passes_percentage', 'passes_percentage'),
        }
        
        for team_stats in stats:
            team_name = team_stats['team']['name']
            is_home = team_stats['team']['id'] == stats[0]['team']['id']
            prefix = 'home' if is_home else 'away'
            
            for stat in team_stats['statistics']:
                stat_type = stat['type']
                stat_value = stat['value']
                
                if stat_type in stat_mapping:
                    _, field_name = stat_mapping[stat_type]
                    key = f"{prefix}_{field_name}"
                    
                    # Convert percentage strings to integers
                    if isinstance(stat_value, str) and '%' in stat_value:
                        stat_value = int(stat_value.replace('%', ''))
                    elif stat_value is None:
                        stat_value = 0
                    
                    stats_dict[key] = stat_value
        
        return stats_dict
    
    async def fetch_league_fixtures(self, league_id: int, season: int = 2024) -> List[Dict]:
        """
        Fetch all fixtures for a specific league
        """
        data = await self._make_request('fixtures', {
            'league': league_id,
            'season': season
        })
        
        if not data or 'response' not in data:
            return []
        
        return data['response']
    
    async def fetch_team_statistics(self, team_id: int, league_id: int, season: int = 2024) -> Optional[Dict]:
        """
        Fetch team statistics for a season
        """
        data = await self._make_request('teams/statistics', {
            'team': team_id,
            'league': league_id,
            'season': season
        })
        
        if not data or 'response' not in data:
            return None
        
        return data['response']
    
    async def get_top_leagues(self) -> List[int]:
        """
        Get list of top football leagues to monitor
        """
        # Top leagues with most live betting activity
        return [
            39,   # Premier League (England)
            140,  # La Liga (Spain)
            78,   # Bundesliga (Germany)
            135,  # Serie A (Italy)
            61,   # Ligue 1 (France)
            94,   # Primeira Liga (Portugal)
            88,   # Eredivisie (Netherlands)
            203,  # Super Lig (Turkey)
            2,    # UEFA Champions League
            3,    # UEFA Europa League
        ]
    
    async def monitor_live_matches(self, callback, interval: int = 30):
        """
        Continuously monitor live matches and call callback with new data
        """
        logger.info(f"Starting live match monitoring (interval: {interval}s)")
        
        while True:
            try:
                matches = await self.fetch_live_matches()
                
                if matches:
                    await callback(matches)
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in live monitoring: {e}")
                await asyncio.sleep(interval)
    
    async def close(self):
        """Close the session"""
        if self.session and not self.session.closed:
            await self.session.close()


async def test_pipeline():
    """Test the data pipeline"""
    logger.add("logs/data_pipeline_{time}.log")
    
    pipeline = FootballDataPipeline()
    
    try:
        logger.info("Testing live matches fetch...")
        matches = await pipeline.fetch_live_matches()
        
        if matches:
            logger.info(f"Found {len(matches)} live matches")
            for match in matches[:3]:
                logger.info(f"\n{match.home_team} vs {match.away_team}")
                logger.info(f"Score: {match.home_goals}-{match.away_goals}")
                logger.info(f"Elapsed: {match.elapsed_minutes} min")
                logger.info(f"Possession: {match.home_possession}% - {match.away_possession}%")
        else:
            logger.info("No live matches currently")
        
    finally:
        await pipeline.close()


if __name__ == "__main__":
    asyncio.run(test_pipeline())
