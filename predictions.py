import os
import logging
import requests
import certifi

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment Variables
API_KEY = os.getenv("API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# API URLs
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}
TELEGRAM_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

def send_to_telegram(message: str):
    """Send formatted message to Telegram."""
    try:
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML"
        }
        res = requests.post(TELEGRAM_URL, json=payload, timeout=10, verify=certifi.where())
        res.raise_for_status()
        logger.info("✅ Tip sent to Telegram")
    except Exception as e:
        logger.error(f"❌ Telegram send failed: {e}")

def fetch_live_matches():
    """Fetch all currently live matches."""
    try:
        res = requests.get(
            f"{BASE_URL}/fixtures",
            headers=HEADERS,
            params={"live": "all"},
            timeout=10,
            verify=certifi.where()
        )
        res.raise_for_status()
        return res.json().get("response", [])
    except Exception as e:
        logger.error(f"❌ Failed to fetch live matches: {e}")
        return []

def fetch_match_stats(fixture_id: int):
    """Fetch statistics for a specific live fixture."""
    try:
        res = requests.get(
            f"{BASE_URL}/fixtures/statistics",
            headers=HEADERS,
            params={"fixture": fixture_id},
            timeout=10,
            verify=certifi.where()
        )
        res.raise_for_status()
        return res.json().get("response", [])
    except Exception as e:
        logger.warning(f"⚠️ No stats for fixture {fixture_id}: {e}")
        return []

def generate_tip(match):
    """Generate betting tip based on live stats."""
    fixture_id = match["fixture"]["id"]
    home = match["teams"]["home"]["name"]
    away = match["teams"]["away"]["name"]
    score_home = match["goals"]["home"]
    score_away = match["goals"]["away"]
    elapsed = match["fixture"]["status"]["elapsed"]

    if not elapsed or elapsed > 90:
        return None

    stats_data = fetch_match_stats(fixture_id)
    if not stats_data:
        return None

    # Structure stats into a dictionary
    stats_dict = {
        s["team"]["name"]: {i["type"]: i["value"] for i in s["statistics"]}
        for s in stats_data
    }
    if home not in stats_dict or away not in stats_dict:
        return None

    s_home = stats_dict[home]
    s_away = stats_dict[away]

    # Extract relevant metrics
    shots_home = int(s_home.get("Shots on Target", 0) or 0)
    shots_away = int(s_away.get("Shots on Target", 0) or 0)
    corners_home = int(s_home.get("Corner Kicks", 0) or 0)
    corners_away = int(s_away.get("Corner Kicks", 0) or 0)
    possession_home = int(str(s_home.get("Ball Possession", "0")).replace('%', '') or 0)
    possession_away = int(str(s_away.get("Ball Possession", "0")).replace('%', '') or 0)

    tip_lines = []
    total_shots = shots_home + shots_away
    total_corners = corners_home + corners_away

    # Pressure detection
    if shots_home >= 5 or corners_home >= 5 or possession_home >= 60:
        tip_lines.append(f"🔥 <b>{home}</b> showing strong pressure!")

    if shots_away >= 5 or corners_away >= 5 or possession_away >= 60:
        tip_lines.append(f"🔥 <b>{away}</b> showing strong pressure!")

    if not tip_lines:
        tip_lines.append("📌 Balanced match so far.")

    return (
        f"⚽ <b>{home}</b> vs <b>{away}</b>\n"
        f"⏱️ {elapsed}'\n"
        f"🔢 Score: {score_home}-{score_away}\n\n" +
        "\n".join(tip_lines)
    )

def run_live_predictions():
    """Run the in-play match checker and send tips."""
    matches = fetch_live_matches()
    if not matches:
        logger.info("📭 No live matches found.")
        return

    for match in matches:
        tip = generate_tip(match)
        if tip:
            send_to_telegram(tip)

if __name__ == "__main__":
    run_live_predictions()
