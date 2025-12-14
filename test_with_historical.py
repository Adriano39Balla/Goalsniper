# test_with_historical.py
import os
import sys
import json
from dotenv import load_dotenv

load_dotenv()

print("🧪 Testing Prediction Engine with Historical Data")
print("=" * 60)

# Sample historical match data (Portugal vs Spain, 2022 World Cup)
HISTORICAL_MATCH = {
    "fixture": {
        "id": 592872,
        "status": {"short": "FT", "elapsed": 90}
    },
    "league": {
        "id": 1,
        "name": "World Cup",
        "country": "International"
    },
    "teams": {
        "home": {
            "id": 33,
            "name": "Portugal",
            "logo": "https://media.api-sports.io/football/teams/33.png"
        },
        "away": {
            "id": 36,
            "name": "Spain",
            "logo": "https://media.api-sports.io/football/teams/36.png"
        }
    },
    "goals": {
        "home": 2,
        "away": 2
    },
    "score": {
        "halftime": {"home": 1, "away": 1},
        "fulltime": {"home": 2, "away": 2},
        "extratime": {"home": None, "away": None},
        "penalty": {"home": None, "away": None}
    }
}

# Sample statistics at minute 60 (2-2 draw)
HISTORICAL_STATS = {
    33: {  # Portugal stats
        "Shots on Goal": 8,
        "Shots off Goal": 4,
        "Total Shots": 12,
        "Blocked Shots": 2,
        "Shots insidebox": 10,
        "Shots outsidebox": 2,
        "Fouls": 12,
        "Corner Kicks": 6,
        "Offsides": 1,
        "Ball Possession": "58%",
        "Yellow Cards": 2,
        "Red Cards": 0,
        "Goalkeeper Saves": 3,
        "Total passes": 512,
        "Passes accurate": 468,
        "Passes %": "91%",
        "expected_goals": "2.1"
    },
    36: {  # Spain stats
        "Shots on Goal": 6,
        "Shots off Goal": 3,
        "Total Shots": 9,
        "Blocked Shots": 1,
        "Shots insidebox": 7,
        "Shots outsidebox": 2,
        "Fouls": 10,
        "Corner Kicks": 4,
        "Offsides": 2,
        "Ball Possession": "42%",
        "Yellow Cards": 1,
        "Red Cards": 0,
        "Goalkeeper Saves": 6,
        "Total passes": 412,
        "Passes accurate": 378,
        "Passes %": "92%",
        "expected_goals": "1.8"
    }
}

def test_engine():
    """Test the engine with historical data"""
    print("\n📊 Historical Match: Portugal 2-2 Spain (Minute 60)")
    print("League: World Cup 2022")
    
    # Import the actual engine components
    sys.path.append('.')
    
    try:
        # Import your actual classes
        from main import EventProbabilityEngine, MarketOpportunityGenerator, ValueConfidenceFilter
        
        # Initialize engines
        prob_engine = EventProbabilityEngine()
        market_gen = MarketOpportunityGenerator()
        value_filter = ValueConfidenceFilter()
        
        # Prepare match data
        match_data = HISTORICAL_MATCH.copy()
        match_data['statistics'] = HISTORICAL_STATS
        
        # Extract features at minute 60
        print("\n🔍 Extracting features...")
        features = prob_engine.extract_features(match_data, 60)
        
        print(f"✅ Features extracted: {len(features)} metrics")
        print(f"  - Score: {features.get('home_score', 0)}-{features.get('away_score', 0)}")
        print(f"  - xG: {features.get('home_xg', 0):.2f} - {features.get('away_xg', 0):.2f}")
        print(f"  - Shots on target: {features.get('home_shots_on', 0)}-{features.get('away_shots_on', 0)}")
        print(f"  - Corners: {features.get('home_corners', 0)}-{features.get('away_corners', 0)}")
        print(f"  - Possession: {features.get('home_possession', 0):.1%}")
        
        # Calculate probabilities
        print("\n🧮 Calculating probabilities...")
        probabilities = prob_engine.calculate_probabilities(features)
        
        print("📈 Key Probabilities:")
        for key, value in probabilities.items():
            if 'next' in key or 'probability' in key or 'expected' in key:
                print(f"  - {key}: {value:.1%}")
        
        # Generate opportunities
        print("\n🎯 Generating market opportunities...")
        opportunities = market_gen.generate_opportunities(probabilities, features)
        
        if opportunities:
            print(f"✅ Found {len(opportunities)} opportunities:")
            for opp in opportunities:
                # Calculate confidence
                confidence = value_filter.calculate_confidence_score(
                    opp, 
                    match_data['league']['id'], 
                    60, 
                    probabilities
                )
                
                print(f"\n  Market: {opp['description']}")
                print(f"  Probability: {opp['probability']:.1%}")
                print(f"  Confidence: {confidence['confidence_score']:.1%}")
                print(f"  Expected Value: {confidence['expected_value']:.2f}")
                print(f"  Recommendation: {'✅ VALUE' if confidence['final_decision'] else '⚠️ Monitor'}")
        else:
            print("❌ No opportunities generated")
            print("\nPossible reasons:")
            print("1. Probability thresholds too high")
            print("2. Match state doesn't trigger any market conditions")
            print("3. Feature extraction issue")
            
            # Show what thresholds we're missing
            print("\n📊 Missing thresholds:")
            thresholds = {
                'Next Goal': probabilities.get('goal_next_10min', 0) > 0.35,
                'Over 2.5': probabilities.get('expected_final_goals', 0) >= 2.5,
                'Next Corner': probabilities.get('corner_next_10min', 0) > 0.4,
                'Home to Score': probabilities.get('home_goal_probability', 0) > 0.6,
                'Away to Score': probabilities.get('away_goal_probability', 0) > 0.6,
            }
            
            for market, condition in thresholds.items():
                status = "✅ Met" if condition else "❌ Not met"
                print(f"  {market}: {status}")
                
    except Exception as e:
        print(f"❌ Error testing engine: {e}")
        import traceback
        traceback.print_exc()

def test_live_api():
    """Test the live API to see what data is available"""
    print("\n" + "=" * 60)
    print("🌐 Testing Live API Data Availability")
    print("=" * 60)
    
    import requests
    
    API_KEY = os.getenv('API_FOOTBALL_KEY')
    headers = {
        'x-apisports-key': API_KEY,
        'x-rapidapi-host': 'v3.football.api-sports.io'
    }
    
    # Get live matches
    url = "https://v3.football.api-sports.io/fixtures"
    params = {'live': 'all'}
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            matches = data.get('response', [])
            
            print(f"📊 Found {len(matches)} live matches")
            
            for i, match in enumerate(matches[:3]):  # Check first 3
                fixture_id = match.get('fixture', {}).get('id')
                home = match.get('teams', {}).get('home', {}).get('name')
                away = match.get('teams', {}).get('away', {}).get('name')
                minute = match.get('fixture', {}).get('status', {}).get('elapsed')
                
                print(f"\nMatch {i+1}: {home} vs {away}")
                print(f"  ID: {fixture_id}, Minute: {minute}")
                
                # Check if statistics endpoint returns data
                stats_url = "https://v3.football.api-sports.io/fixtures/statistics"
                stats_params = {'fixture': fixture_id}
                
                stats_response = requests.get(stats_url, headers=headers, params=stats_params, timeout=30)
                
                if stats_response.status_code == 200:
                    stats_data = stats_response.json()
                    if stats_data.get('response'):
                        print(f"  ✅ Statistics available")
                        # Show available stats
                        team_stats = stats_data['response'][0].get('statistics', [])
                        stat_types = [stat.get('type') for stat in team_stats[:5]]
                        print(f"  Available stats: {', '.join(stat_types[:3])}...")
                    else:
                        print(f"  ❌ No statistics data (empty response)")
                else:
                    print(f"  ❌ Statistics endpoint error: {stats_response.status_code}")
                
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error testing API: {e}")

def adjust_thresholds():
    """Create a version with adjusted thresholds"""
    print("\n" + "=" * 60)
    print("🔄 Creating Adjusted Thresholds Version")
    print("=" * 60)
    
    adjusted_code = '''
# ADJUSTED THRESHOLDS FOR TESTING
# Add this to your MarketOpportunityGenerator class:

class MarketOpportunityGenerator:
    def __init__(self):
        # LOWERED THRESHOLDS FOR TESTING
        self.market_definitions = {
            'next_goal': {
                'condition': lambda p: p.get('goal_next_10min', 0) > 0.25,  # Lowered from 0.35
                'probability_field': 'goal_next_10min',
                'description': 'Next Goal'
            },
            'over_2.5': {
                'condition': lambda p: p.get('expected_final_goals', 0) >= 2.0,  # Lowered from 2.5
                'probability_field': 'expected_final_goals',
                'description': 'Over 2.5 Goals'
            },
            'next_corner': {
                'condition': lambda p: p.get('corner_next_10min', 0) > 0.3,  # Lowered from 0.4
                'probability_field': 'corner_next_10min',
                'description': 'Next Corner'
            },
            'home_to_score': {
                'condition': lambda p: p.get('home_goal_probability', 0) > 0.4,  # Lowered from 0.6
                'probability_field': 'home_goal_probability',
                'description': 'Home Team to Score'
            },
            'away_to_score': {
                'condition': lambda p: p.get('away_goal_probability', 0) > 0.4,  # Lowered from 0.6
                'probability_field': 'away_goal_probability',
                'description': 'Away Team to Score'
            },
            'yellow_card': {
                'condition': lambda p: p.get('yellow_card_next_10min', 0) > 0.3,  # Lowered from 0.5
                'probability_field': 'yellow_card_next_10min',
                'description': 'Yellow Card Next 10min'
            },
            'both_teams_score': {
                'condition': lambda p: p.get('home_goal_probability', 0) > 0.3 and  # Lowered
                                      p.get('away_goal_probability', 0) > 0.3,
                'probability_field': lambda p: min(p.get('home_goal_probability', 0), 
                                                  p.get('away_goal_probability', 0)),
                'description': 'Both Teams to Score'
            }
        }
    
    # Also adjust ValueConfidenceFilter:
    class ValueConfidenceFilter:
        def __init__(self):
            self.min_confidence = 0.4  # Lowered from 0.65
            self.min_value = 1.0  # Lowered from 1.1
    '''
    
    print("✅ Copy this adjusted code into your main.py to lower thresholds")
    print("\n📝 Instructions:")
    print("1. Replace MarketOpportunityGenerator.__init__ with the adjusted version")
    print("2. Replace ValueConfidenceFilter.__init__ with adjusted thresholds")
    print("3. Restart the engine")
    print("4. You should see more opportunities (but they might be lower quality)")

if __name__ == "__main__":
    print("🧪 MARKET-AGNOSTIC ENGINE DIAGNOSTIC TOOL")
    print("=" * 60)
    
    # Check environment
    required = ['API_FOOTBALL_KEY', 'TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID']
    missing = [var for var in required if not os.getenv(var)]
    
    if missing:
        print(f"❌ Missing environment variables: {missing}")
        print("Please check your .env file")
    else:
        print("✅ Environment variables OK")
        
        # Run tests
        test_engine()
        test_live_api()
        adjust_thresholds()
        
        print("\n" + "=" * 60)
        print("🎯 RECOMMENDED ACTIONS:")
        print("1. Run the test above to see if engine works with historical data")
        print("2. Check if live matches have statistics available")
        print("3. Lower thresholds temporarily to get some test predictions")
        print("4. Once working, gradually increase thresholds for quality")
        print("=" * 60)
