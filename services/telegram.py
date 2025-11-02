import logging
import requests
from typing import Optional
from html import escape

from core.config import config

log = logging.getLogger("goalsniper.telegram")

class TelegramService:
    """Telegram notification service"""
    
    def __init__(self):
        self.bot_token = config.telegram.bot_token
        self.chat_id = config.telegram.chat_id
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
    
    def send_message(self, text: str) -> bool:
        """Send message to Telegram"""
        if not self.bot_token or not self.chat_id:
            log.warning("Telegram not configured - missing bot_token or chat_id")
            return False
        
        try:
            response = requests.post(
                f"{self.base_url}/sendMessage",
                data={
                    "chat_id": self.chat_id,
                    "text": text,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True
                },
                timeout=10
            )
            
            success = response.ok
            if success:
                log.info("Telegram message sent successfully")
            else:
                log.warning(f"Telegram send failed: {response.status_code} - {response.text}")
            
            return success
            
        except Exception as e:
            log.error(f"Telegram send error: {e}")
            return False
    
    def format_tip_message(self, tip_data: dict) -> str:
        """Format a tip message for Telegram"""
        home = escape(tip_data['home'])
        away = escape(tip_data['away'])
        league = escape(tip_data['league'])
        minute = tip_data['minute']
        score = escape(tip_data['score'])
        suggestion = escape(tip_data['suggestion'])
        confidence = tip_data['confidence']
        features = tip_data.get('features', {})
        odds = tip_data.get('odds')
        book = tip_data.get('book')
        ev_pct = tip_data.get('ev_pct')
        
        # Build statistics section
        stats = ""
        if any([
            features.get("xg_h", 0), features.get("xg_a", 0),
            features.get("sot_h", 0), features.get("sot_a", 0),
            features.get("cor_h", 0), features.get("cor_a", 0),
            features.get("pos_h", 0), features.get("pos_a", 0)
        ]):
            stats = (
                f"\n📊 xG {features.get('xg_h', 0):.2f}-{features.get('xg_a', 0):.2f}"
                f" • SOT {int(features.get('sot_h', 0))}-{int(features.get('sot_a', 0))}"
                f" • CK {int(features.get('cor_h', 0))}-{int(features.get('cor_a', 0))}"
            )
            if features.get("pos_h", 0) or features.get("pos_a", 0):
                stats += f" • POS {int(features.get('pos_h', 0))}%–{int(features.get('pos_a', 0))}%"
        
        # Build odds section
        money = ""
        if odds:
            if ev_pct is not None:
                money = f"\n💰 <b>Odds:</b> {odds:.2f} @ {book or 'Book'}  •  <b>EV:</b> {ev_pct:+.1f}%"
            else:
                money = f"\n💰 <b>Odds:</b> {odds:.2f} @ {book or 'Book'}"
        
        # AI confidence info
        ai_info = ""
        confidence_raw = tip_data.get('confidence_raw')
        if confidence_raw is not None:
            confidence_level = "🟢 HIGH" if confidence_raw > 0.8 else "🟡 MEDIUM" if confidence_raw > 0.6 else "🔴 LOW"
            ai_info = f"\n🤖 <b>AI Confidence:</b> {confidence_level} ({confidence_raw:.1%})"
        
        # Context note for current state predictions
        context_note = ""
        current_score = score.split('-')
        if len(current_score) == 2:
            home_goals, away_goals = int(current_score[0]), int(current_score[1])
            is_current_state = (
                (suggestion == "Home Win" and home_goals > away_goals) or
                (suggestion == "Away Win" and away_goals > home_goals) or
                (suggestion == "BTTS: Yes" and home_goals > 0 and away_goals > 0) or
                (suggestion == "BTTS: No" and (home_goals == 0 or away_goals == 0))
            )
            
            if is_current_state:
                context_note = f"\n⚠️ <i>Note: This reflects current match probability based on score and time</i>"
        
        return (
            "⚽️ <b>🤖 AI ENHANCED TIP!</b>\n"
            f"<b>Match:</b> {home} vs {away}\n"
            f"🕒 <b>Minute:</b> {minute}'  |  <b>Score:</b> {score}\n"
            f"<b>Tip:</b> {suggestion}\n"
            f"📈 <b>Confidence:</b> {confidence:.1f}%{ai_info}{money}{context_note}\n"
            f"🏆 <b>League:</b> {league}{stats}"
        )
    
    def send_tip(self, tip_data: dict) -> bool:
        """Send a formatted tip message"""
        message = self.format_tip_message(tip_data)
        return self.send_message(message)
    
    def send_system_message(self, message: str) -> bool:
        """Send a system message (no formatting)"""
        return self.send_message(f"🤖 {message}")

# Global telegram service instance
telegram_service = TelegramService()
