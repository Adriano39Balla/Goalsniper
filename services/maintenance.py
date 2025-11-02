import os
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from zoneinfo import ZoneInfo

from core.config import config
from core.database import db
from services.api_client import api_client

log = logging.getLogger("goalsniper.maintenance")

# Add these missing maintenance-related functions
def cleanup_caches():
    """Clean up expired cache entries from global caches"""
    now = time.time()
    
    # Import caches from api_client
    from services.api_client import STATS_CACHE, EVENTS_CACHE, ODDS_CACHE, NEG_CACHE
    
    # Clean STATS_CACHE
    expired = [fid for fid, (ts, _) in STATS_CACHE.items() if now - ts > 300]
    for fid in expired:
        STATS_CACHE.pop(fid, None)
    
    # Clean EVENTS_CACHE
    expired = [fid for fid, (ts, _) in EVENTS_CACHE.items() if now - ts > 300]
    for fid in expired:
        EVENTS_CACHE.pop(fid, None)
        
    # Clean ODDS_CACHE
    expired = [fid for fid, (ts, _) in ODDS_CACHE.items() if now - ts > 300]
    for fid in expired:
        ODDS_CACHE.pop(fid, None)
        
    # Clean NEG_CACHE
    expired = [key for key, (ts, _) in NEG_CACHE.items() if now - ts > NEG_TTL_SEC]
    for key in expired:
        NEG_CACHE.pop(key, None)

def retry_unsent_tips(minutes: int = 30, limit: int = 200) -> int:
    """Retry sending unsent tips (standalone function)"""
    cutoff = int(time.time()) - minutes * 60
    retried = 0
    
    with db.get_cursor() as c:
        c.execute("""
            SELECT match_id, league, home, away, market, suggestion, confidence, 
                   confidence_raw, score_at_tip, minute, created_ts, odds, book, ev_pct 
            FROM tips 
            WHERE sent_ok = 0 AND created_ts >= %s 
            ORDER BY created_ts ASC 
            LIMIT %s
        """, (cutoff, limit))
        
        rows = c.fetchall()
        
        for row in rows:
            (mid, league, home, away, market, sugg, conf, conf_raw, score, minute, 
             cts, odds, book, ev_pct) = row
            
            # Import telegram service here to avoid circular imports
            from services.telegram import telegram_service
            
            # Format and send message
            message = _format_tip_message(
                home, away, league, int(minute), score, sugg, float(conf), 
                {}, odds, book, ev_pct
            )
            
            ok = telegram_service.send_message(message)
            if ok:
                c.execute(
                    "UPDATE tips SET sent_ok = 1 WHERE match_id = %s AND created_ts = %s",
                    (mid, cts)
                )
                retried += 1
    
    if retried:
        log.info("[RETRY] Resent %d unsent tips", retried)
    
    return retried

def _format_tip_message(home, away, league, minute, score, suggestion, confidence, 
                       feat, odds=None, book=None, ev_pct=None):
    """Format a tip message for Telegram"""
    stat = ""
    if any([feat.get("xg_h",0), feat.get("xg_a",0), feat.get("sot_h",0), feat.get("sot_a",0),
            feat.get("cor_h",0), feat.get("cor_a",0), feat.get("pos_h",0), feat.get("pos_a",0)]):
        stat = (f"\n📊 xG {feat.get('xg_h',0):.2f}-{feat.get('xg_a',0):.2f}"
                f" • SOT {int(feat.get('sot_h',0))}-{int(feat.get('sot_a',0))}"
                f" • CK {int(feat.get('cor_h',0))}-{int(feat.get('cor_a',0))}")
        if feat.get("pos_h",0) or feat.get("pos_a",0): 
            stat += f" • POS {int(feat.get('pos_h',0))}%–{int(feat.get('pos_a',0))}%"
    
    money = ""
    if odds:
        if ev_pct is not None:
            money = f"\n💰 <b>Odds:</b> {odds:.2f} @ {book or 'Book'}  •  <b>EV:</b> {ev_pct:+.1f}%"
        else:
            money = f"\n💰 <b>Odds:</b> {odds:.2f} @ {book or 'Book'}"
            
    from html import escape
    return ("⚽️ <b>AI TIP!</b>\n"
            f"<b>Match:</b> {escape(home)} vs {escape(away)}\n"
            f"🕒 <b>Minute:</b> {minute}'  |  <b>Score:</b> {escape(score)}\n"
            f"<b>Tip:</b> {escape(suggestion)}\n"
            f"📈 <b>Confidence:</b> {confidence:.1f}%{money}\n"
            f"🏆 <b>League:</b> {escape(league)}{stat}")

def _is_final(short: str) -> bool:
    """Check if match status is final (standalone function)"""
    return (short or "").upper() in {"FT", "AET", "PEN"}

def _fixture_by_id(mid: int) -> Optional[dict]:
    """Get fixture by ID (standalone function)"""
    fixtures = api_client.get_fixtures({"id": mid})
    return fixtures[0] if fixtures else None

def backfill_results_for_open_matches(max_rows: int = 200) -> int:
    """Backfill results for open matches (standalone function)"""
    return maintenance_service.backfill_results(max_rows)

class MaintenanceService:
    """Maintenance and cleanup operations"""
    
    def __init__(self):
        self.berlin_tz = ZoneInfo("Europe/Berlin")
    
    def backfill_results(self, max_rows: int = 200) -> int:
        """Backfill results for open matches"""
        updated = 0
        cutoff_ts = int((datetime.now() - timedelta(days=config.scheduler.backfill_days)).timestamp())
        
        with db.get_cursor() as c:
            c.execute("""
                WITH last_tips AS (
                    SELECT match_id, MAX(created_ts) as last_ts
                    FROM tips 
                    WHERE created_ts >= %s
                    GROUP BY match_id
                )
                SELECT lt.match_id
                FROM last_tips lt
                LEFT JOIN match_results mr ON mr.match_id = lt.match_id
                WHERE mr.match_id IS NULL
                ORDER BY lt.last_ts DESC
                LIMIT %s
            """, (cutoff_ts, max_rows))
            
            match_ids = [row[0] for row in c.fetchall()]
        
        for match_id in match_ids:
            if self._backfill_single_result(match_id):
                updated += 1
        
        if updated:
            log.info("[RESULTS] Backfilled %d match results", updated)
        
        return updated
    
    def _backfill_single_result(self, match_id: int) -> bool:
        """Backfill result for a single match"""
        try:
            fixtures = api_client.get_fixtures({"id": match_id})
            if not fixtures:
                return False
            
            fixture_data = fixtures[0]
            fixture_info = fixture_data.get("fixture", {})
            status = fixture_info.get("status", {})
            short_status = (status.get("short") or "").upper()
            
            # Only process final matches
            if not self._is_final_status(short_status):
                return False
            
            goals = fixture_data.get("goals", {})
            home_goals = goals.get("home", 0) or 0
            away_goals = goals.get("away", 0) or 0
            btts_yes = 1 if (home_goals > 0 and away_goals > 0) else 0
            
            with db.get_cursor() as c:
                c.execute("""
                    INSERT INTO match_results (match_id, final_goals_h, final_goals_a, btts_yes, updated_ts)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (match_id) DO UPDATE SET
                    final_goals_h = EXCLUDED.final_goals_h,
                    final_goals_a = EXCLUDED.final_goals_a,
                    btts_yes = EXCLUDED.btts_yes,
                    updated_ts = EXCLUDED.updated_ts
                """, (match_id, home_goals, away_goals, btts_yes, int(time.time())))
            
            return True
            
        except Exception as e:
            log.warning("[RESULTS] Failed to backfill match %s: %s", match_id, e)
            return False
    
    def _is_final_status(self, status: str) -> bool:
        """Check if match status is final"""
        return _is_final(status)
    
    def cleanup_caches(self):
        """Clean up expired cache entries"""
        cleanup_caches()
    
    def retry_unsent_tips(self, minutes: int = 30, limit: int = 200) -> int:
        """Retry sending unsent tips"""
        return retry_unsent_tips(minutes, limit)

# Global maintenance service instance
maintenance_service = MaintenanceService()

# Export for compatibility
__all__ = [
    'maintenance_service',
    'cleanup_caches',
    'retry_unsent_tips', 
    'backfill_results_for_open_matches',
    '_is_final',
    '_fixture_by_id'
]
