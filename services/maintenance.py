import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

from core.config import config
from core.database import db
from services.api_client import api_client

log = logging.getLogger("goalsniper.maintenance")

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
        return status in {"FT", "AET", "PEN"}
    
    def cleanup_caches(self):
        """Clean up expired cache entries"""
        # Implementation of cache cleanup
        # [Your existing cache cleanup logic]
        pass
    
    def retry_unsent_tips(self, minutes: int = 30, limit: int = 200) -> int:
        """Retry sending unsent tips"""
        # Implementation of tip retry logic
        # [Your existing retry logic]
        return 0

# Global maintenance service instance
maintenance_service = MaintenanceService()
