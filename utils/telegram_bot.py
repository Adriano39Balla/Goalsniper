import asyncio
from typing import Dict, Any
import os
from .logger import logger

class TelegramBot:
    """Telegram bot for sending predictions"""
    
    def __init__(self):
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        
    async def send_message(self, text: str, parse_mode: str = 'HTML') -> bool:
        """Send message to Telegram"""
        
        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram credentials not configured")
            return False
        
        import aiohttp
        
        url = f"{self.base_url}/sendMessage"
        payload = {
            'chat_id': self.chat_id,
            'text': text,
            'parse_mode': parse_mode,
            'disable_web_page_preview': True
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        logger.info("Telegram message sent successfully")
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"Telegram error: {error_text}")
                        return False
        except Exception as e:
            logger.error(f"Telegram send error: {e}")
            return False
    
    async def send_prediction(self, prediction: Dict[str, Any]) -> bool:
        """Format and send prediction alert"""
        
        match_info = prediction.get('match_info', {})
        predictions = prediction.get('predictions', {})
        
        # Format message
        message = self._format_prediction_message(match_info, predictions)
        
        # Send message
        return await self.send_message(message)
    
    def _format_prediction_message(self, match_info: Dict, predictions: Dict) -> str:
        """Format prediction message for Telegram"""
        
        home_team = match_info.get('home_team', 'Home')
        away_team = match_info.get('away_team', 'Away')
        score = match_info.get('current_score', '0-0')
        minute = match_info.get('minute', 0)
        
        # Create message header
        message = f"⚽ <b>LIVE BETTING ALERT</b> ⚽\n\n"
        message += f"🏆 <b>{home_team} vs {away_team}</b>\n"
        message += f"📊 Score: {score} ({minute}')\n"
        message += f"🕒 Time: {match_info.get('match_time', 'Live')}\n\n"
        
        # Add predictions
        message += "🔮 <b>PREDICTIONS:</b>\n"
        
        for pred_type, pred_data in predictions.items():
            if pred_type == '1X2':
                message += f"\n🎯 <b>1X2</b>:\n"
                message += f"   • {pred_data.get('prediction', 'N/A').upper()}\n"
                message += f"   • Confidence: {pred_data.get('confidence', 0)*100:.1f}%\n"
                message += f"   • Probability: {pred_data.get('probability', 0)*100:.1f}%\n"
            
            elif pred_type == 'over_under':
                message += f"\n📈 <b>OVER/UNDER 2.5</b>:\n"
                message += f"   • {pred_data.get('prediction', 'N/A').upper()}\n"
                message += f"   • Confidence: {pred_data.get('confidence', 0)*100:.1f}%\n"
                message += f"   • Probability: {pred_data.get('probability', 0)*100:.1f}%\n"
            
            elif pred_type == 'btts':
                message += f"\n🎪 <b>BOTH TEAMS TO SCORE</b>:\n"
                message += f"   • {pred_data.get('prediction', 'N/A').upper()}\n"
                message += f"   • Confidence: {pred_data.get('confidence', 0)*100:.1f}%\n"
                message += f"   • Probability: {pred_data.get('probability', 0)*100:.1f}%\n"
        
        # Add risk warning
        message += "\n⚠️ <i>Bet responsibly. Past performance doesn't guarantee future results.</i>"
        
        return message
