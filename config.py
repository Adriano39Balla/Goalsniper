import os
from dotenv import load_dotenv

load_dotenv()

API_FOOTBALL_API_KEY = os.getenv('API_KEY')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

LEAGUE_ID = int(os.getenv('LEAGUE_ID', 39))  # Default EPL
SEASON = int(os.getenv('SEASON', 2023))

# Other configs
MODEL_DIR = 'models'
DATA_DIR = 'data'
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
