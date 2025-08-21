# app/harvest.py
import json
import logging
import time
from typing import Dict, Any, List
from flask import Blueprint, jsonify

from app.db import db_conn
from app.features import extract_features, stats_coverage_ok
from app.api_client import fetch_live_matches, fetch_fixtures_by_ids

bp = Blueprint("harvest", __name__)

def save_snapshot_from_match(match: Dict[str, Any], min_minute: int = 15, require_stats_minute: int = 10, require_data_fields: int = 2) -> bool:
    try:
        fixture = match.get("fixture") or {}
        fid = int(fixture.get("id"))
        minute = int((fixture.get("status") or {}).get("elapsed") or 0)

        feats = extract_features(match)

        if not stats_coverage_ok(feats, minute, require_stats_minute, require_data_fields):
            logging.debug(f"[SNAP] skipping match {fid}: insufficient stats coverage")
            return False

        payload = {
            "minute": feats.get("minute"),
            "gh": feats.get("goals_h"),
            "ga": feats.get("goals_a"),
            "stat": {
                "xg_h": feats.get("xg_h"),
                "xg_a": feats.get("xg_a"),
                "sot_h": feats.get("sot_h"),
                "sot_a": feats.get("sot_a"),
                "cor_h": feats.get("cor_h"),
                "cor_a": feats.get("cor_a"),
                "pos_h": feats.get("pos_h"),
                "pos_a": feats.get("pos_a"),
                "red_h": feats.get("red_h"),
                "red_a": feats.get("red_a"),
            }
        }

        ts = int(time.time())
        with db_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO tip_snapshots(match_id, created_ts, payload)
                VALUES (?, ?, ?)
            """, (fid, ts, json.dumps(payload)))
            conn.commit()

        logging.info(f"[SNAP] saved snapshot for match {fid} at minute {minute}")
        return True

    except Exception as e:
        logging.exception(f"[SNAP] error saving snapshot: {e}")
        return False


def harvest_scan(min_minute: int = 15) -> int:
    matches = fetch_live_matches()
    saved = 0
    for m in matches:
        ok = save_snapshot_from_match(m, min_minute=min_minute)
        if ok:
            saved += 1
    logging.info(f"[HARVEST] total snapshots saved: {saved}")
    return saved


def create_synthetic_snapshots_for_league(league_id: int, season: int, match_ids: List[int]) -> int:
    if not match_ids:
        return 0

    fixtures = fetch_fixtures_by_ids(match_ids)
    saved = 0
    for fid, match in fixtures.items():
        try:
            feats = extract_features(match)
            payload = {
                "minute": 90,
                "gh": feats.get("goals_h"),
                "ga": feats.get("goals_a"),
                "stat": {
                    "xg_h": feats.get("xg_h"),
                    "xg_a": feats.get("xg_a"),
                    "sot_h": feats.get("sot_h"),
                    "sot_a": feats.get("sot_a"),
                    "cor_h": feats.get("cor_h"),
                    "cor_a": feats.get("cor_a"),
                    "pos_h": feats.get("pos_h"),
                    "pos_a": feats.get("pos_a"),
                    "red_h": feats.get("red_h"),
                    "red_a": feats.get("red_a"),
                }
            }
            ts = int(time.time())
            with db_conn() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO tip_snapshots(match_id, created_ts, payload)
                    VALUES (?, ?, ?)
                """, (fid, ts, json.dumps(payload)))
                conn.commit()
            saved += 1
        except Exception as e:
            logging.exception(f"[HARVEST] failed synthetic snapshot for {fid}: {e}")

    logging.info(f"[HARVEST] synthetic snapshots saved={saved}")
    return saved


# ✅ Route for API + Scheduler
def harvest_route():
    saved = harvest_scan()
    return jsonify({"ok": True, "snapshots_saved": saved})


@bp.route("/harvest", methods=["POST"])
def harvest_api():
    return harvest_route()
