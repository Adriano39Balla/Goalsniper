import pandas as pd
import requests
from config import API_KEY

API_URL = "https://v3.football.api-sports.io/fixtures"

def fetch_live_data():
    print("📡 Fetching live match data...")
    headers = {"x-apisports-key": API_KEY}
    params = {"live": "all"}  # Example for live matches

    response = requests.get(API_URL, headers=headers, params=params)
    if response.status_code != 200:
        print(f"⚠️ Failed to fetch data: {response.status_code}")
        return None

    json_data = response.json()
    if "response" not in json_data:
        return None

    # Convert API response into a dataframe your model can use
    matches = []
    for match in json_data["response"]:
        matches.append({
            "home_team": match["teams"]["home"]["name"],
            "away_team": match["teams"]["away"]["name"],
            "target": 1  # Dummy for now — update if you have real labels
        })

    df = pd.DataFrame(matches)
    return df
