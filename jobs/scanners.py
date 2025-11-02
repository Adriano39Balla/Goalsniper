import time, os
import logging
from typing import Dict, Any, List, Tuple, Optional
from html import escape

from core.config import config
from core.database import db
from core.features import feature_engineer
from core.predictors import market_predictor
from services.api_client import api_client
from services.telegram import telegram_service

log = logging.getLogger("goalsniper.scanner")

class ProductionScanner:
    """Enhanced production scanner with AI systems"""
    
    def __init__(self):
        self.inplay_statuses = {"1H", "HT", "2H", "ET", "BT", "P"}
        self.allowed_suggestions = self._initialize_allowed_suggestions()
        self.league_block_patterns = [
            "u17", "u18", "u19", "u20", "u21", "u23", "youth", 
            "junior", "reserve", "res.", "friendlies", "friendly"
        ]
    
    def _initialize_allowed_suggestions(self) -> set:
        """Initialize allowed betting suggestions"""
        suggestions = {"BTTS: Yes", "BTTS: No", "Home Win", "Away Win"}
        
        for line in config.ou_lines:
            line_str = f"{line}".rstrip("0").rstrip(".")
            suggestions.add(f"Over {line_str} Goals")
            suggestions.add(f"Under {line_str} Goals")
        
        return suggestions
    
    def scan(self) -> Tuple[int, int]:
        """
        Enhanced scan with per-gate debug logging
        Returns: (tips_saved, matches_seen)
        """
        try:
            matches = self._fetch_live_matches()
            if not matches:
                log.info("[SCAN] No live matches found")
                return 0, 0
            
            saved_tips = 0
            now_ts = int(time.time())
            per_league_counter = {}
            
            for match in matches:
                try:
                    tips_saved = self._process_match(match, now_ts, per_league_counter)
                    saved_tips += tips_saved
                    
                    if config.models.max_tips_per_scan and saved_tips >= config.models.max_tips_per_scan:
                        break
                        
                except Exception as e:
                    log.exception("[SCAN] Error processing match: %s", e)
                    continue
            
            log.info("[SCAN] Saved %d tips from %d matches", saved_tips, len(matches))
            return saved_tips, len(matches)
            
        except Exception as e:
            log.exception("[SCAN] Global scan error: %s", e)
            return 0, 0
    
    def _fetch_live_matches(self) -> List[Dict[str, Any]]:
        """Fetch and filter live matches"""
        matches = api_client.get_live_matches()
        viable_matches = []
        
        for match in matches:
            if self._is_match_viable(match):
                # Enhance match with additional data
                fixture_id = match["fixture"]["id"]
                match["statistics"] = api_client.get_fixture_statistics(fixture_id)
                match["events"] = api_client.get_fixture_events(fixture_id)
                viable_matches.append(match)
        
        return viable_matches
    
    def _is_match_viable(self, match: Dict[str, Any]) -> bool:
        """Check if match is viable for scanning"""
        # Check league
        if self._is_league_blocked(match.get("league", {})):
            return False
        
        # Check match status
        fixture = match.get("fixture", {})
        status = fixture.get("status", {})
        elapsed = status.get("elapsed")
        short_status = (status.get("short") or "").upper()
        
        if elapsed is None or elapsed > 120 or short_status not in self.inplay_statuses:
            return False
        
        return True
    
    def _is_league_blocked(self, league_obj: Dict[str, Any]) -> bool:
        """Check if league should be blocked"""
        name = str(league_obj.get("name", "")).lower()
        country = str(league_obj.get("country", "")).lower()
        league_type = str(league_obj.get("type", "")).lower()
        
        text = f"{country} {name} {league_type}"
        if any(pattern in text for pattern in self.league_block_patterns):
            return True
        
        # Check denied IDs
        denied_ids = [x.strip() for x in os.getenv("LEAGUE_DENY_IDS", "").split(",") if x.strip()]
        league_id = str(league_obj.get("id", ""))
        if league_id in denied_ids:
            return True
        
        return False
    
    def _process_match(self, match: Dict[str, Any], timestamp: int, 
                      per_league_counter: Dict[int, int]) -> int:
        """Process a single match and return number of tips saved"""
        fixture_id = match["fixture"]["id"]
        
        # Check duplicate cooldown
        if self._is_in_cooldown(fixture_id, timestamp):
            return 0
        
        # Extract features
        try:
            features = feature_engineer.extract_advanced_features(match)
        except Exception as e:
            log.warning("[SCAN] Feature extraction failed for match %s: %s", fixture_id, e)
            return 0
        
        minute = int(features.get("minute", 0))
        
        # Basic viability checks
        if not self._is_match_data_viable(features, minute, fixture_id, match):
            return 0
        
        # Harvest snapshot if enabled
        if config.harvest_mode and minute >= config.models.train_min_minute and minute % 3 == 0:
            self._save_snapshot(match, features)
        
        # Generate predictions
        candidates = self._generate_predictions(match, features, minute)
        if not candidates:
            return 0
        
        # Process candidates with odds checking
        saved_tips = self._process_candidates(
            match, features, candidates, timestamp, per_league_counter
        )
        
        return saved_tips
    
    def _is_in_cooldown(self, fixture_id: int, timestamp: int) -> bool:
        """Check if match is in duplicate cooldown"""
        if config.models.dup_cooldown_min <= 0:
            return False
        
        cutoff = timestamp - config.models.dup_cooldown_min * 60
        with db.get_cursor() as c:
            c.execute(
                "SELECT 1 FROM tips WHERE match_id = %s AND created_ts >= %s AND suggestion <> 'HARVEST' LIMIT 1",
                (fixture_id, cutoff)
            )
            return c.fetchone() is not None
    
    def _is_match_data_viable(self, features: Dict[str, float], minute: int, 
                             fixture_id: int, match: Dict[str, Any]) -> bool:
        """Check if match data is viable for predictions"""
        # Coverage check
        if not self._stats_coverage_ok(features, minute):
            log.debug("[SCAN] Poor stats coverage for match %s", fixture_id)
            return False
        
        # Minute check
        if minute < config.models.tip_min_minute:
            log.debug("[SCAN] Minute too low for match %s: %d", fixture_id, minute)
            return False
        
        # Stale feed check
        if self._is_feed_stale(fixture_id, match, minute):
            log.debug("[SCAN] Feed stale for match %s", fixture_id)
            return False
        
        return True
    
    def _stats_coverage_ok(self, features: Dict[str, float], minute: int) -> bool:
        """Check if stats coverage is sufficient"""
        # Nuclear option - accept virtually everything
        has_any_data = any([
            features.get('pos_h', 0) > 0, features.get('pos_a', 0) > 0,
            features.get('sot_h', 0) > 0, features.get('sot_a', 0) > 0,
            features.get('goals_h', 0) > 0, features.get('goals_a', 0) > 0,
            features.get('cor_h', 0) > 0, features.get('cor_a', 0) > 0,
            features.get('sh_total_h', 0) > 0, features.get('sh_total_a', 0) > 0
        ])
        
        # After 30 minutes, require at least SOME data
        if minute > 30:
            return has_any_data
        
        # Before 30 minutes, be very lenient
        return True
    
    def _is_feed_stale(self, fixture_id: int, match: Dict[str, Any], minute: int) -> bool:
        """Check if data feed is stale"""
        if not config.stale_guard_enable:
            return False
        
        # Check events recency
        events = match.get("events", [])
        if events:
            latest_minute = max([ev.get('time', {}).get('elapsed', 0) for ev in events], default=0)
            if minute - latest_minute > 10 and minute > 30:
                return True
        
        return False
    
    def _save_snapshot(self, match: Dict[str, Any], features: Dict[str, float]):
        """Save snapshot for training data"""
        try:
            fixture_id = match["fixture"]["id"]
            snapshot = {
                "match": match,
                "features": features,
                "timestamp": int(time.time())
            }
            
            with db.get_cursor() as c:
                c.execute(
                    "INSERT INTO tip_snapshots (match_id, created_ts, payload) "
                    "VALUES (%s, %s, %s) "
                    "ON CONFLICT (match_id, created_ts) DO UPDATE SET payload = EXCLUDED.payload",
                    (fixture_id, int(time.time()), json.dumps(snapshot))
                )
        except Exception as e:
            log.warning("[SNAPSHOT] Failed to save snapshot: %s", e)
    
    def _generate_predictions(self, match: Dict[str, Any], features: Dict[str, float], 
                            minute: int) -> List[Tuple]:
        """Generate predictions for all markets"""
        candidates = []
        
        # BTTS predictions
        btts_candidates = self._generate_btts_predictions(features, minute)
        candidates.extend(btts_candidates)
        
        # Over/Under predictions
        ou_candidates = self._generate_ou_predictions(features, minute)
        candidates.extend(ou_candidates)
        
        # 1X2 predictions
        one_x_two_candidates = self._generate_1x2_predictions(features, minute)
        candidates.extend(one_x_two_candidates)
        
        return candidates
    
    def _generate_btts_predictions(self, features: Dict[str, float], minute: int) -> List[Tuple]:
        """Generate BTTS predictions"""
        candidates = []
        try:
            prob, confidence = market_predictor.predict_for_market(features, "BTTS", minute)
            if prob > 0 and confidence > 0.5:
                threshold = self._get_market_threshold("BTTS")
                
                for suggestion in ["BTTS: Yes", "BTTS: No"]:
                    adjusted_prob = prob if suggestion == "BTTS: Yes" else max(0.0, 1 - prob)
                    
                    if (self._is_prediction_viable("BTTS", suggestion, minute, features) and
                        adjusted_prob * 100 >= threshold):
                        candidates.append(("BTTS", suggestion, adjusted_prob, confidence))
                        
        except Exception as e:
            log.warning("[BTTS] Prediction error: %s", e)
        
        return candidates
    
    def _generate_ou_predictions(self, features: Dict[str, float], minute: int) -> List[Tuple]:
        """Generate Over/Under predictions"""
        candidates = []
        
        for line in config.ou_lines:
            market_key = f"OU_{line}"
            try:
                prob, confidence = market_predictor.predict_for_market(features, market_key, minute)
                if prob > 0 and confidence > 0.5:
                    threshold = self._get_market_threshold(f"Over/Under {line}")
                    
                    for suggestion in [f"Over {line} Goals", f"Under {line} Goals"]:
                        adjusted_prob = prob if "Over" in suggestion else max(0.0, 1 - prob)
                        
                        if (self._is_prediction_viable(f"Over/Under {line}", suggestion, minute, features) and
                            adjusted_prob * 100 >= threshold):
                            candidates.append((f"Over/Under {line}", suggestion, adjusted_prob, confidence))
                            
            except Exception as e:
                log.warning("[OU] Prediction error for line %s: %s", line, e)
        
        return candidates
    
    def _generate_1x2_predictions(self, features: Dict[str, float], minute: int) -> List[Tuple]:
        """Generate 1X2 predictions (draw suppressed)"""
        candidates = []
        # Implementation of 1X2 prediction logic
        # [Your existing 1X2 logic here]
        return candidates
    
    def _is_prediction_viable(self, market: str, suggestion: str, minute: int, 
                             features: Dict[str, float]) -> bool:
        """Check if prediction is viable"""
        # Market cutoff check
        if not self._market_cutoff_ok(minute, market, suggestion):
            return False
        
        # Sanity check
        if not self._prediction_sane(suggestion, features):
            return False
        
        return True
    
    def _market_cutoff_ok(self, minute: int, market: str, suggestion: str) -> bool:
        """Check if market is within minute cutoff"""
        # Implementation of market cutoff logic
        # [Your existing market cutoff logic]
        return True
    
    def _prediction_sane(self, suggestion: str, features: Dict[str, float]) -> bool:
        """Check if prediction makes sense given current match state"""
        # Implementation of sanity checks
        # [Your existing sanity check logic]
        return True
    
    def _get_market_threshold(self, market: str) -> float:
        """Get confidence threshold for a specific market"""
        # Try market-specific threshold first
        # [Your existing threshold logic]
        return config.models.confidence_threshold
    
    def _process_candidates(self, match: Dict[str, Any], features: Dict[str, float],
                          candidates: List[Tuple], timestamp: int,
                          per_league_counter: Dict[int, int]) -> int:
        """Process viable candidates and save tips"""
        # Implementation of candidate processing with odds checking
        # [Your existing candidate processing logic]
        return 0

# Global scanner instance
production_scanner = ProductionScanner()
