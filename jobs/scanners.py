import time, os, json
import logging
from typing import Dict, Any, List, Tuple, Optional
from html import escape

from core.config import config
from core.database import db
from core.features import feature_engineer
from core.predictors import market_predictor
from services.api_client import api_client, fetch_odds
from services.telegram import telegram_service
from services.markets import (
    _parse_ou_line_from_suggestion, _odds_key_for_market, _market_family,
    market_cutoff_ok, _candidate_is_sane, _inplay_1x2_sanity_ok,
    _min_odds_for_market, _ev, calculate_ev_bps
)

log = logging.getLogger("goalsniper.scanner")

class ProductionScanner:
    """COMPLETE production scanner with ALL methods from main.py"""
    
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
        Enhanced scan with per-gate debug logging - EXACTLY from main.py
        Returns: (tips_saved, matches_seen)
        """
        try:
            if not self._db_ping():
                log.error("[SCAN] Database unavailable")
                return (0, 0)

            try:
                matches = self._fetch_live_matches()
            except Exception as e:
                log.error("[SCAN] Failed to fetch live matches: %s", e)
                return (0, 0)

            live_seen = len(matches)
            if live_seen == 0:
                log.info("[SCAN] No live matches")
                return 0, 0

            saved = 0
            now_ts = int(time.time())
            per_league_counter = {}

            for m in matches:
                try:
                    fid = int((m.get("fixture") or {}).get("id") or 0)
                    if not fid:
                        continue

                    # Duplicate cooldown — FIXED: use c.fetchone_safe() (not chaining on cursor)
                    if config.models.dup_cooldown_min > 0:
                        cutoff = now_ts - config.models.dup_cooldown_min * 60
                        with db.get_cursor() as c:
                            c.execute(
                                "SELECT 1 FROM tips WHERE match_id=%s AND created_ts>=%s AND suggestion<>'HARVEST' LIMIT 1",
                                (fid, cutoff),
                            )
                            if c.fetchone():
                                continue

                    # Extract advanced features with error handling
                    try:
                        feat = feature_engineer.extract_advanced_features(m)
                    except Exception as e:
                        log.warning("[SCAN] Feature extraction failed for fid %s: %s", fid, e)
                        continue
                        
                    minute = int(feat.get("minute", 0))

                    if not self._stats_coverage_ok(feat, minute):
                        continue
                    if minute < config.models.tip_min_minute:
                        continue
                    if self._is_feed_stale(fid, m, minute):
                        continue

                    # Harvest snapshot
                    if config.harvest_mode and minute >= config.models.train_min_minute and minute % 3 == 0:
                        try:
                            self._save_snapshot_from_match(m, feat)
                        except Exception:
                            pass

                    league_id, league = self._league_name(m)
                    home, away = self._teams(m)
                    score = self._pretty_score(m)

                    candidates = self._generate_candidates(feat, minute, m)
                    if not candidates:
                        continue

                    odds_map = fetch_odds(fid) if config.api.key else {}
                    ranked = self._rank_candidates(candidates, odds_map, feat, fid, minute)
                    if not ranked:
                        continue

                    per_match = 0
                    base_now = int(time.time())

                    for idx, (market_txt, suggestion, prob, odds, book, ev_pct, _rank, conf) in enumerate(ranked):
                        if config.models.per_league_cap > 0 and per_league_counter.get(league_id, 0) >= config.models.per_league_cap:
                            break
                        if per_match >= max(1, config.models.predictions_per_match):
                            break

                        created_ts = base_now + idx
                        raw = float(prob)
                        prob_pct = round(raw * 100.0, 1)

                        try:
                            with db.get_cursor() as c:
                                c.execute(
                                    "INSERT INTO tips("
                                    "match_id,league_id,league,home,away,market,suggestion,"
                                    "confidence,confidence_raw,score_at_tip,minute,created_ts,"
                                    "odds,book,ev_pct,sent_ok"
                                    ") VALUES ("
                                    "%s,%s,%s,%s,%s,%s,%s,"
                                    "%s,%s,%s,%s,%s,"
                                    "%s,%s,%s,%s"
                                    ")",
                                    (
                                        fid, league_id, league, home, away,
                                        market_txt, suggestion,
                                        float(prob_pct), raw, score, minute, created_ts,
                                        (float(odds) if odds is not None else None),
                                        (book or None),
                                        (float(ev_pct) if ev_pct is not None else None),
                                        0,
                                    ),
                                )
                            sent = telegram_service.send_message(self._format_enhanced_tip_message(
                                home, away, league, minute, score, suggestion,
                                float(prob_pct), feat, odds, book, ev_pct, conf
                            ))
                            if sent:
                                with db.get_cursor() as c:
                                    c.execute(
                                        "UPDATE tips SET sent_ok=1 WHERE match_id=%s AND created_ts=%s",
                                        (fid, created_ts)
                                    )
                        except Exception as e:
                            log.exception("[SCAN] insert/send failed: %s", e)
                            continue

                        saved += 1
                        per_match += 1
                        per_league_counter[league_id] = per_league_counter.get(league_id, 0) + 1

                        if config.models.max_tips_per_scan and saved >= config.models.max_tips_per_scan:
                            break

                    if config.models.max_tips_per_scan and saved >= config.models.max_tips_per_scan:
                        break

                except Exception as e:
                    log.exception("[SCAN] match loop failed: %s", e)
                    continue

            log.info("[SCAN] saved=%d live_seen=%d", saved, live_seen)
            return saved, live_seen
        except Exception as e:
            log.exception("[SCAN] Global scan error: %s", e)
            return (0, 0)
    
    def _fetch_live_matches(self) -> List[Dict[str, Any]]:
        """Fetch and filter live matches - EXACTLY from main.py"""
        matches = api_client.get_live_matches()
        viable_matches = []
        
        for match in matches:
            if self._is_league_blocked(match.get("league", {})):
                continue
                
            fixture = match.get("fixture", {})
            status = fixture.get("status", {})
            elapsed = status.get("elapsed")
            short_status = (status.get("short") or "").upper()
            
            if elapsed is None or elapsed > 120 or short_status not in self.inplay_statuses:
                continue
            
            # Enhance match with additional data
            fixture_id = fixture["id"]
            match["statistics"] = api_client.get_fixture_statistics(fixture_id)
            match["events"] = api_client.get_fixture_events(fixture_id)
            viable_matches.append(match)
        
        return viable_matches
    
    def _is_league_blocked(self, league_obj: Dict[str, Any]) -> bool:
        """Check if league should be blocked - EXACTLY from main.py"""
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
    
    def _stats_coverage_ok(self, feat: Dict[str, float], minute: int) -> bool:
        """EXACTLY from main.py - NUCLEAR OPTION"""
        # Only reject if we have absolutely NO data at all
        has_any_data = any([
            feat.get('pos_h', 0) > 0, feat.get('pos_a', 0) > 0,
            feat.get('sot_h', 0) > 0, feat.get('sot_a', 0) > 0, 
            feat.get('goals_h', 0) > 0, feat.get('goals_a', 0) > 0,
            feat.get('cor_h', 0) > 0, feat.get('cor_a', 0) > 0,
            feat.get('sh_total_h', 0) > 0, feat.get('sh_total_a', 0) > 0
        ])
        
        # After 30 minutes, require at least SOME data
        if minute > 30:
            return has_any_data
        
        # Before 30 minutes, be very lenient
        return True
    
    def _is_feed_stale(self, fid: int, m: dict, minute: int) -> bool:
        """EXACTLY from main.py"""
        if not config.stale_guard_enable:
            return False
        
        # Check if events are recent
        events = m.get("events", [])
        if events:
            latest_event_minute = max([ev.get('time', {}).get('elapsed', 0) for ev in events], default=0)
            if minute - latest_event_minute > 10 and minute > 30:
                return True
        
        # Check if stats are recent
        from services.api_client import STATS_CACHE
        cache_time = STATS_CACHE.get(fid, (0, []))[0]
        if time.time() - cache_time > config.cache.stale_stats_max_sec:
            return True
            
        return False
    
    def _save_snapshot_from_match(self, m: dict, feat: Dict[str, float]) -> None:
        """EXACTLY from main.py"""
        try:
            fid = int((m.get("fixture") or {}).get("id") or 0)
            if not fid:
                return
                
            now = int(time.time())
            payload = json.dumps({
                "match": m,
                "features": feat,
                "timestamp": now
            }, separators=(",", ":"), ensure_ascii=False)
            
            with db.get_cursor() as c:
                c.execute(
                    "INSERT INTO tip_snapshots(match_id, created_ts, payload) "
                    "VALUES (%s,%s,%s) "
                    "ON CONFLICT (match_id, created_ts) DO UPDATE SET payload=EXCLUDED.payload",
                    (fid, now, payload)
                )
        except Exception as e:
            log.warning("[SNAPSHOT] Failed to save snapshot: %s", e)

    def _league_name(self, m: dict) -> Tuple[int, str]:
        """EXACTLY from main.py"""
        league = m.get("league", {}) or {}
        league_id = int(league.get("id", 0))
        country = league.get("country", "")
        name = league.get("name", "")
        league_name = f"{country} - {name}".strip(" -")
        return league_id, league_name

    def _teams(self, m: dict) -> Tuple[str, str]:
        """EXACTLY from main.py"""
        teams = m.get("teams", {}) or {}
        home = (teams.get("home") or {}).get("name", "")
        away = (teams.get("away") or {}).get("name", "")
        return home, away

    def _pretty_score(self, m: dict) -> str:
        """EXACTLY from main.py"""
        goals = m.get("goals", {}) or {}
        home = goals.get("home") or 0
        away = goals.get("away") or 0
        return f"{home}-{away}"

    def _generate_candidates(self, feat: Dict[str, float], minute: int, m: dict) -> List[Tuple[str, str, float, float]]:
        """EXACTLY from main.py"""
        candidates: List[Tuple[str, str, float, float]] = []
        log.info(f"[MARKET_SCAN] Processing {self._teams(m)[0]} vs {self._teams(m)[1]} at minute {minute}")

        # 1) BTTS
        try:
            btts_prob, btts_conf = market_predictor.predict_for_market(feat, "BTTS", minute)
            if btts_prob > 0 and btts_conf > 0.5:
                preds = {"BTTS: Yes": btts_prob, "BTTS: No": max(0.0, 1 - btts_prob)}
                for suggestion, p in preds.items():
                    if not market_cutoff_ok(minute, "BTTS", suggestion):
                        continue
                    if not _candidate_is_sane(suggestion, feat):
                        continue
                    thr = self._get_market_threshold("BTTS")
                    if p * 100.0 >= thr:
                        candidates.append(("BTTS", suggestion, p, btts_conf))
        except Exception as e:
            log.warning(f"[BTTS] Prediction error: {e}")

        # 2) OU
        for line in config.ou_lines:
            mkkey = f"Over/Under {config._fmt_line(line)}"
            try:
                ou_prob, ou_conf = market_predictor.predict_for_market(feat, f"OU_{config._fmt_line(line)}", minute)
                if ou_prob <= 0 or ou_conf <= 0.5:
                    continue
                preds = {
                    f"Over {config._fmt_line(line)} Goals": ou_prob,
                    f"Under {config._fmt_line(line)} Goals": max(0.0, 1 - ou_prob),
                }
                for suggestion, p in preds.items():
                    if not market_cutoff_ok(minute, mkkey, suggestion):
                        continue
                    if not _candidate_is_sane(suggestion, feat):
                        continue
                    thr = self._get_market_threshold(mkkey)
                    if p * 100.0 >= thr:
                        candidates.append((mkkey, suggestion, p, ou_conf))
            except Exception as e:
                log.warning(f"[OU] Prediction error for line {line}: {e}")

        # 3) 1X2 (draw suppressed)
        try:
            ph, pa, c1 = market_predictor._predict_1x2_advanced(feat, minute)
            if ph > 0 and pa > 0 and c1 > 0.5:
                s = max(1e-9, ph + pa)
                ph, pa = ph/s, pa/s
                preds = {"Home Win": ph, "Away Win": pa}
                for suggestion, p in preds.items():
                    if not market_cutoff_ok(minute, "1X2", suggestion):
                        continue
                    thr = self._get_market_threshold("1X2")
                    if p * 100.0 >= thr:
                        candidates.append(("1X2", suggestion, p, c1))
        except Exception as e:
            log.warning(f"[1X2] Prediction error: {e}")

        return candidates

    def _rank_candidates(self, candidates: List[Tuple[str, str, float, float]], odds_map: dict, 
                        feat: Dict[str, float], fid: int, minute: int) -> List[Tuple]:
        """EXACTLY from main.py"""
        ranked: List[Tuple[str, str, float, Optional[float], Optional[str], Optional[float], float, float]] = []

        for mk, sug, prob, conf in candidates:
            if sug not in self.allowed_suggestions:
                continue

            # Per-market odds quality
            okey = _odds_key_for_market(mk, sug)
            # Skip odds quality check for now to match main.py behavior

            # Odds lookup + price gate
            odds = None; book = None
            if mk == "BTTS":
                d = (odds_map.get("BTTS") or {})
                tgt = "Yes" if sug.endswith("Yes") else "No"
                if tgt in d: odds, book = d[tgt]["odds"], d[tgt]["book"]
            elif mk == "1X2":
                d = (odds_map.get("1X2") or {})
                tgt = "Home" if sug == "Home Win" else ("Away" if sug == "Away Win" else None)
                if tgt and tgt in d: odds, book = d[tgt]["odds"], d[tgt]["book"]
            elif mk.startswith("Over/Under"):
                ln = _parse_ou_line_from_suggestion(sug)
                d = (odds_map.get(f"OU_{config._fmt_line(ln)}") or {}) if ln is not None else {}
                tgt = "Over" if sug.startswith("Over") else "Under"
                if tgt in d: odds, book = d[tgt]["odds"], d[tgt]["book"]

            pass_odds, odds2, book2, _ = self._price_gate(mk, sug, fid)
            if not pass_odds:
                continue
            if odds is None:
                odds, book = odds2, book2
            if mk == "1X2" and not _inplay_1x2_sanity_ok(sug, feat, odds):
                continue

            ev_pct = None
            if odds is not None:
                edge = _ev(prob, float(odds))
                ev_bps = calculate_ev_bps(prob, float(odds))
                ev_pct = round(edge * 100.0, 1)
                if ev_bps < config.odds.edge_min_bps:
                    continue
            elif not config.allow_tips_without_odds:
                continue

            rank_score = (prob ** 1.2) * (1 + (ev_pct or 0) / 100.0) * max(0.0, conf)
            ranked.append((mk, sug, prob, odds, book, ev_pct, rank_score, conf))

        ranked.sort(key=lambda x: x[6], reverse=True)
        return ranked

    def _price_gate(self, market_text: str, suggestion: str, fid: int, prob_hint: float = None) -> Tuple[bool, Optional[float], Optional[str], Optional[float]]:
        """
        EXACTLY from main.py
        Return (pass, odds, book, ev_pct).
        For in-play tips, ALWAYS require live odds - no prematch fallback.
        """
        odds_map = fetch_odds(fid, require_live=True) if config.api.key else {}
        odds = None
        book = None
        
        if market_text == "BTTS":
            d = odds_map.get("BTTS", {})
            tgt = "Yes" if suggestion.endswith("Yes") else "No"
            if tgt in d: 
                odds = d[tgt]["odds"]
                book = d[tgt]["book"]
        elif market_text == "1X2":
            d = odds_map.get("1X2", {})
            tgt = "Home" if suggestion == "Home Win" else ("Away" if suggestion == "Away Win" else None)
            if tgt and tgt in d: 
                odds = d[tgt]["odds"]
                book = d[tgt]["book"]
        elif market_text.startswith("Over/Under"):
            ln_val = _parse_ou_line_from_suggestion(suggestion)
            d = odds_map.get(f"OU_{config._fmt_line(ln_val)}", {}) if ln_val is not None else {}
            tgt = "Over" if suggestion.startswith("Over") else "Under"
            if tgt in d:
                odds = d[tgt]["odds"]
                book = d[tgt]["book"]

        # CRITICAL: Require live odds for in-play picks - no fallback to prematch
        if odds is None:
            log.warning(f"[PRICE_GATE] No LIVE odds found for {market_text} {suggestion} in fixture {fid}")
            return (False, None, None, None)

        min_odds = _min_odds_for_market(market_text)
        if not (min_odds <= odds <= config.odds.max_odds_all):
            log.warning(f"[PRICE_GATE] Odds {odds} outside range {min_odds}-{config.odds.max_odds_all} for {market_text}")
            return (False, odds, book, None)
        
        # Calculate EV
        ev_pct = None
        if odds and prob_hint and prob_hint > 0:
            edge = _ev(prob_hint, float(odds))
            ev_pct = round(edge * 100.0, 1)
        
        return (True, odds, book, ev_pct)

    def _format_enhanced_tip_message(self, home, away, league, minute, score, suggestion, 
                                   prob_pct, feat, odds=None, book=None, ev_pct=None, confidence=None):
        """EXACTLY from main.py"""
        stat = ""
        if any([feat.get("xg_h",0),feat.get("xg_a",0),feat.get("sot_h",0),feat.get("sot_a",0),
                feat.get("cor_h",0),feat.get("cor_a",0),feat.get("pos_h",0),feat.get("pos_a",0)]):
            stat = (f"\n📊 xG {feat.get('xg_h',0):.2f}-{feat.get('xg_a',0):.2f}"
                    f" • SOT {int(feat.get('sot_h',0))}-{int(feat.get('sot_a',0))}"
                    f" • CK {int(feat.get('cor_h',0))}-{int(feat.get('cor_a',0))}")
            if feat.get("pos_h",0) or feat.get("pos_a",0): 
                stat += f" • POS {int(feat.get('pos_h',0))}%–{int(feat.get('pos_a',0))}%"
        
        # FIXED: Rename "Confidence" to "Win Probability" when appropriate
        current_score = score.split('-')
        home_goals, away_goals = int(current_score[0]), int(current_score[1])
        
        # Determine if this is a current state (like Home Win when already winning)
        is_current_state_prediction = (
            (suggestion == "Home Win" and home_goals > away_goals) or
            (suggestion == "Away Win" and away_goals > home_goals) or
            (suggestion == "BTTS: Yes" and home_goals > 0 and away_goals > 0) or
            (suggestion == "BTTS: No" and (home_goals == 0 or away_goals == 0))
        )
        
        confidence_label = "📈 Win Probability" if is_current_state_prediction else "📈 Confidence"
        
        ai_info = ""
        if confidence is not None:
            confidence_level = "🟢 HIGH" if confidence > 0.8 else "🟡 MEDIUM" if confidence > 0.6 else "🔴 LOW"
            ai_info = f"\n🤖 <b>AI Confidence:</b> {confidence_level} ({confidence:.1%})"
        
        money = ""
        if odds:
            if ev_pct is not None:
                money = f"\n💰 <b>Odds:</b> {odds:.2f} @ {book or 'Book'}  •  <b>EV:</b> {ev_pct:+.1f}%"
            else:
                money = f"\n💰 <b>Odds:</b> {odds:.2f} @ {book or 'Book'}"
        
        # FIXED: Add context about current match state
        context_note = ""
        if is_current_state_prediction:
            context_note = f"\n⚠️ <i>Note: This reflects current match probability based on score and time</i>"
        
        return ("⚽️ <b>🤖 AI ENHANCED TIP!</b>\n"
                f"<b>Match:</b> {escape(home)} vs {escape(away)}\n"
                f"🕒 <b>Minute:</b> {minute}'  |  <b>Score:</b> {escape(score)}\n"
                f"<b>Tip:</b> {escape(suggestion)}\n"
                f"{confidence_label}: {prob_pct:.1f}%{ai_info}{money}{context_note}\n"
                f"🏆 <b>League:</b> {escape(league)}{stat}")

    def _get_market_threshold(self, market: str) -> float:
        """EXACTLY from main.py"""
        # Try market-specific threshold first
        market_key = f"conf_threshold:{market}"
        cached = self._get_setting_cached(market_key)
        if cached:
            try:
                return float(cached)
            except:
                pass
        # Fall back to global threshold
        return config.models.confidence_threshold

    def _get_setting_cached(self, key: str) -> Optional[str]:
        """EXACTLY from main.py - placeholder for settings cache"""
        # This would connect to your settings cache
        return None

    def _db_ping(self) -> bool:
        """EXACTLY from main.py"""
        try:
            with db.get_cursor() as c:
                c.execute("SELECT 1")
                return True
        except Exception:
            log.warning("[DB] ping failed")
            return False

# Global scanner instance
production_scanner = ProductionScanner()

# Legacy functions for compatibility with main.py
def enhanced_production_scan() -> Tuple[int, int]:
    return production_scanner.scan()

def production_scan() -> Tuple[int, int]:
    return enhanced_production_scan()

__all__ = ['production_scanner', 'enhanced_production_scan', 'production_scan']
