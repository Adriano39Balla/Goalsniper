"""
Telegram Notification System
Sends betting predictions and daily digests to Telegram
"""

from typing import Dict, List, Optional
from datetime import datetime
import asyncio

from telegram import Bot
from telegram.constants import ParseMode
from loguru import logger

from config import settings
from ml_engine import PredictionResult
from data_pipeline import LiveMatchData


class TelegramNotifier:
    """
    High-quality Telegram notification system for betting predictions
    """
    
    def __init__(self):
        self.bot = Bot(token=settings.TELEGRAM_BOT_TOKEN)
        self.chat_id = settings.TELEGRAM_CHAT_ID
        self.sent_predictions = set()  # Avoid duplicate notifications
    
    async def send_prediction(self, match: LiveMatchData, prediction: PredictionResult):
        """
        Send betting prediction to Telegram with rich formatting
        """
        # Check if already sent
        prediction_key = f"{prediction.fixture_id}_{prediction.market}_{prediction.prediction}"
        if prediction_key in self.sent_predictions:
            return
        
        # Build message
        message = self._format_prediction_message(match, prediction)
        
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True
            )
            
            self.sent_predictions.add(prediction_key)
            logger.info(f"Sent prediction to Telegram: {match.home_team} vs {match.away_team} - {prediction.market}")
            
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")
    
    def _format_prediction_message(self, match: LiveMatchData, prediction: PredictionResult) -> str:
        """
        Format prediction message with HTML formatting
        """
        # Confidence emoji
        if prediction.confidence_score >= 0.8:
            confidence_emoji = "ð¥"
        elif prediction.confidence_score >= 0.7:
            confidence_emoji = "â­"
        else:
            confidence_emoji = "â"
        
        # Market emoji
        market_emojis = {
            'over_under_goals': 'â½',
            'btts': 'ð¯',
            'next_goal': 'ð¥',
            'total_cards': 'ð¨',
            'total_corners': 'ð©',
            'match_winner': 'ð'
        }
        market_emoji = market_emojis.get(prediction.market, 'ð')
        
        # Format market name
        market_display = prediction.market.replace('_', ' ').title()
        
        # Build message
        message = f"""
{confidence_emoji} <b>NEW BETTING TIP</b> {confidence_emoji}

<b>Match:</b> {match.home_team} vs {match.away_team}
<b>League:</b> {match.league_name}
<b>Current Score:</b> {match.home_goals} - {match.away_goals}
<b>Time:</b> {match.elapsed_minutes}' â±ï¸

{market_emoji} <b>Market:</b> {market_display}
<b>Prediction:</b> {prediction.prediction}
<b>Probability:</b> {prediction.calibrated_probability * 100:.1f}%
<b>Confidence:</b> {prediction.confidence_score * 100:.1f}%
<b>Expected Value:</b> {prediction.expected_value:.3f}

<b>Current Stats:</b>
â¢ Possession: {match.home_possession}% - {match.away_possession}%
â¢ Shots: {match.home_shots_total or 0} - {match.away_shots_total or 0}
â¢ Corners: {match.home_corners or 0} - {match.away_corners or 0}
â¢ Cards: {(match.home_yellow_cards or 0) + (match.home_red_cards or 0)} - {(match.away_yellow_cards or 0) + (match.away_red_cards or 0)}

<i>Generated at {prediction.timestamp.strftime('%H:%M:%S')}</i>
"""
        
        return message.strip()
    
    async def send_daily_digest(self, stats: Dict):
        """
        Send daily performance digest
        """
        message = self._format_daily_digest(stats)
        
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True
            )
            
            logger.info("Sent daily digest to Telegram")
            
        except Exception as e:
            logger.error(f"Error sending daily digest: {e}")
    
    def _format_daily_digest(self, stats: Dict) -> str:
        """
        Format daily digest message
        """
        total_predictions = stats.get('total_predictions', 0)
        market_stats = stats.get('market_statistics', [])
        daily_performance = stats.get('daily_performance', [])
        
        # Calculate overall metrics
        total_correct = sum(m.get('correct', 0) for m in market_stats)
        overall_accuracy = (total_correct / total_predictions * 100) if total_predictions > 0 else 0
        total_profit = sum(m.get('total_profit', 0) for m in market_stats)
        
        # Build message
        message = f"""
ð <b>DAILY PERFORMANCE DIGEST</b> ð

<b>Overall Statistics</b>
â¢ Total Predictions: {total_predictions}
â¢ Overall Accuracy: {overall_accuracy:.1f}%
â¢ Total Profit: {total_profit:+.2f} units

<b>Performance by Market</b>
"""
        
        # Add market statistics
        for market in market_stats[:5]:  # Top 5 markets
            market_name = market.get('market', 'Unknown').replace('_', ' ').title()
            accuracy = market.get('accuracy', 0) * 100
            profit = market.get('total_profit', 0)
            total = market.get('total', 0)
            
            message += f"\n{market_name}:\n"
            message += f"  â¢ Predictions: {total}\n"
            message += f"  â¢ Accuracy: {accuracy:.1f}%\n"
            message += f"  â¢ Profit: {profit:+.2f} units\n"
        
        # Add recent daily performance
        if daily_performance:
            message += "\n<b>Last 7 Days</b>\n"
            for day in daily_performance[:7]:
                date = day.get('date', 'Unknown')
                predictions = day.get('predictions', 0)
                accuracy = day.get('accuracy', 0) * 100
                profit = day.get('profit', 0)
                
                message += f"\n{date}: {predictions} tips, {accuracy:.1f}% accuracy, {profit:+.2f} units"
        
        message += f"\n\n<i>Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>"
        
        return message.strip()
    
    async def send_alert(self, message: str, level: str = "INFO"):
        """
        Send system alert to Telegram
        """
        emoji_map = {
            "INFO": "â¹ï¸",
            "WARNING": "â ï¸",
            "ERROR": "â",
            "SUCCESS": "â"
        }
        
        emoji = emoji_map.get(level, "â¹ï¸")
        formatted_message = f"{emoji} <b>{level}</b>\n\n{message}"
        
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=formatted_message,
                parse_mode=ParseMode.HTML
            )
            
        except Exception as e:
            logger.error(f"Error sending alert: {e}")
    
    async def send_training_complete(self, performance: Dict):
        """
        Send notification when model training completes
        """
        message = f"""
ð <b>MODEL TRAINING COMPLETE</b> ð

<b>Markets Trained:</b> {len(performance)}

<b>Performance Summary:</b>
"""
        
        for market, perf in performance.items():
            market_name = market.replace('_', ' ').title()
            message += f"\n{market_name}:\n"
            message += f"  â¢ Accuracy: {perf.accuracy * 100:.1f}%\n"
            message += f"  â¢ AUC-ROC: {perf.auc_roc:.3f}\n"
            message += f"  â¢ Samples: {perf.total_predictions}\n"
        
        message += f"\n<i>Training completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>"
        
        await self.send_alert(message, "SUCCESS")
    
    async def test_connection(self):
        """
        Test Telegram bot connection
        """
        try:
            me = await self.bot.get_me()
            logger.info(f"Telegram bot connected: @{me.username}")
            
            await self.send_alert("Telegram bot connection test successful!", "SUCCESS")
            return True
            
        except Exception as e:
            logger.error(f"Telegram connection test failed: {e}")
            return False


async def test_telegram():
    """Test Telegram notifications"""
    logger.add("logs/telegram_{time}.log")
    
    notifier = TelegramNotifier()
    
    # Test connection
    success = await notifier.test_connection()
    
    if success:
        logger.info("Telegram test successful")
    else:
        logger.error("Telegram test failed")


if __name__ == "__main__":
    asyncio.run(test_telegram())
