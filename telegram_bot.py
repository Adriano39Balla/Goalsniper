telegram_bot.py




import logging
import asyncio
import threading
from datetime import datetime
from typing import Dict, List
from telegram import Bot
from telegram.error import TelegramError
from config import Config
from utils.data_storage import storage

logger = logging.getLogger(__name__)

class TelegramBotService:
    def __init__(self):
        self.bot_token = Config.TELEGRAM_BOT_TOKEN
        self.chat_id = Config.TELEGRAM_CHAT_ID
        self.bot = None
        self.loop = None
        self.thread = None
        
    def initialize(self):
        """Initialize the Telegram bot"""
        try:
            self.bot = Bot(token=self.bot_token)
            logger.info("Telegram bot initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Telegram bot: {str(e)}")
            return False
    
    def send_prediction(self, prediction: Dict) -> bool:
        """Send a prediction to Telegram"""
        try:
            if not self.bot:
                if not self.initialize():
                    return False
            
            message = self._format_prediction_message(prediction)
            
            # Run in new event loop since we're in a different thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                if self.bot:  # Type guard
                    loop.run_until_complete(
                        self.bot.send_message(chat_id=self.chat_id, text=message, parse_mode='HTML')
                    )
                    logger.info(f"Sent prediction to Telegram: {prediction['type']} for {prediction['home_team']} vs {prediction['away_team']}")
                    return True
                else:
                    return False
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Failed to send prediction to Telegram: {str(e)}")
            return False
    
    def send_daily_summary(self, predictions: List[Dict], performance: Dict) -> bool:
        """Send daily summary of predictions and performance"""
        try:
            if not self.bot:
                if not self.initialize():
                    return False
            
            message = self._format_daily_summary(predictions, performance)
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                if self.bot:  # Type guard
                    loop.run_until_complete(
                        self.bot.send_message(chat_id=self.chat_id, text=message, parse_mode='HTML')
                    )
                    logger.info("Sent daily summary to Telegram")
                    return True
                else:
                    return False
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Failed to send daily summary to Telegram: {str(e)}")
            return False
    
    def send_feedback_request(self, prediction: Dict) -> bool:
        """Send feedback request for a completed prediction"""
        try:
            if not self.bot:
                if not self.initialize():
                    return False
            
            message = self._format_feedback_message(prediction)
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                if self.bot:  # Type guard
                    loop.run_until_complete(
                        self.bot.send_message(chat_id=self.chat_id, text=message, parse_mode='HTML')
                    )
                    logger.info(f"Sent feedback request for prediction {prediction['id']}")
                    return True
                else:
                    return False
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Failed to send feedback request to Telegram: {str(e)}")
            return False
    
    def _format_prediction_message(self, prediction: Dict) -> str:
        """Format prediction for Telegram message"""
        confidence_emoji = "🔥" if prediction['confidence'] > 0.8 else "⚡" if prediction['confidence'] > 0.7 else "💡"
        
        prediction_type_map = {
            'over_2.5_goals': 'Over 2.5 Goals',
            'btts': 'Both Teams to Score',
            'home_win': 'Home Win',
            'away_win': 'Away Win',
            'draw': 'Draw',
            'over_1.5_goals': 'Over 1.5 Goals',
            'under_2.5_goals': 'Under 2.5 Goals'
        }
        
        prediction_text = prediction_type_map.get(prediction['type'], prediction['type'])
        
        message = f"""
{confidence_emoji} <b>Football Prediction</b> {confidence_emoji}

🏆 <b>{prediction['home_team']} vs {prediction['away_team']}</b>

📊 <b>Prediction:</b> {prediction_text}
🎯 <b>Confidence:</b> {prediction['confidence']:.0%}

💡 <b>Reasoning:</b>
{prediction['reasoning']}

⏰ <b>Time:</b> {datetime.now().strftime('%H:%M')}
📅 <b>Date:</b> {datetime.now().strftime('%d/%m/%Y')}

🤖 <i>Generated by AI Football Assistant</i>
        """
        
        return message.strip()
    
    def _format_daily_summary(self, predictions: List[Dict], performance: Dict) -> str:
        """Format daily summary message"""
        total_predictions = len(predictions)
        
        if total_predictions == 0:
            return """
📊 <b>Daily Summary</b>

No predictions generated today.

🤖 <i>AI Football Assistant</i>
            """.strip()
        
        high_confidence = len([p for p in predictions if p['confidence'] > 0.8])
        
        message = f"""
📊 <b>Daily Summary</b>

📈 <b>Today's Predictions:</b> {total_predictions}
🔥 <b>High Confidence (>80%):</b> {high_confidence}

📊 <b>Overall Performance:</b>
✅ <b>Accuracy:</b> {performance.get('accuracy', 0):.1%}
📊 <b>Total Predictions:</b> {performance.get('total_predictions', 0)}
✅ <b>Correct:</b> {performance.get('correct_predictions', 0)}

🤖 <i>AI Football Assistant</i>
        """
        
        return message.strip()
    
    def _format_feedback_message(self, prediction: Dict) -> str:
        """Format feedback request message"""
        prediction_type_map = {
            'over_2.5_goals': 'Over 2.5 Goals',
            'btts': 'Both Teams to Score',
            'home_win': 'Home Win',
            'away_win': 'Away Win',
            'draw': 'Draw',
            'over_1.5_goals': 'Over 1.5 Goals',
            'under_2.5_goals': 'Under 2.5 Goals'
        }
        
        prediction_text = prediction_type_map.get(prediction['type'], prediction['type'])
        
        message = f"""
🔄 <b>Feedback Request</b>

Match: <b>{prediction['home_team']} vs {prediction['away_team']}</b>
Prediction: <b>{prediction_text}</b>

Was this prediction correct?
Reply with ✅ for correct or ❌ for incorrect

<i>Prediction ID: {prediction['id']}</i>
        """
        
        return message.strip()

# Global bot service instance
bot_service = TelegramBotService()

def init_telegram_bot():
    """Initialize the Telegram bot service"""
    try:
        bot_service.initialize()
        logger.info("Telegram bot service initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Telegram bot service: {str(e)}")
