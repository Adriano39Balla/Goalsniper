from datetime import datetime
from app.api_football import api_get
from app.supabase_db import get_supabase
from app.config import TABLE_TIPS, TABLE_RESULTS

sb = get_supabase()


def log(msg: str):
    print(f"[RESOLVE] {msg}")


def fetch_unresolved_tips():
    resp = (
        sb.table(TABLE_TIPS)
        .select("*")
        .eq("resolved", False)
        .execute()
    )
    return resp.data or []


def fetch_fixture_final(fixture_id: int):
    resp = api_get("fixtures", {"id": fixture_id})
    if not resp:
        return None

    fx = resp[0]
    goals = fx.get("goals", {})
    status = fx.get("fixture", {}).get("status", {}).get("short")

    return {
        "status": status,
        "home_goals": goals.get("home") or 0,
        "away_goals": goals.get("away") or 0,
    }


def settle_tip(tip, home_goals: int, away_goals: int):
    market = tip["market"]
    sel = tip["selection"]
    odds = float(tip["odds"])
    stake = 1.0

    # basic result
    # 1X2
    if market == "1X2":
        if home_goals > away_goals:
            actual = "home"
        elif home_goals == away_goals:
            actual = "draw"
        else:
            actual = "away"
        won = (sel == actual)

    # OU25
    elif market == "OU25":
        total = home_goals + away_goals
        actual = "over" if total > 2 else "under"
        won = (sel == actual)

    # BTTS
    elif market == "BTTS":
        btts_yes = (home_goals > 0 and away_goals > 0)
        actual = "yes" if btts_yes else "no"
        won = (sel == actual)

    else:
        won = False

    if won:
        pnl = stake * (odds - 1.0)
        result = "win"
    else:
        pnl = -stake
        result = "lose"

    return result, pnl, stake, odds


def main():
    tips = fetch_unresolved_tips()
    if not tips:
        log("No unresolved tips.")
        return

    log(f"Found {len(tips)} unresolved tips.")

    for tip in tips:
        fid = tip["fixture_id"]
        fx = fetch_fixture_final(fid)
        if not fx:
            continue

        status = fx["status"]
        if status not in ("FT", "AET", "PEN"):
            # match not fully finished
            continue

        home_goals = fx["home_goals"]
        away_goals = fx["away_goals"]

        result, pnl, stake, odds = settle_tip(tip, home_goals, away_goals)

        # insert into tip_results
        sb.table(TABLE_RESULTS).insert({
            "tip_id": tip["id"],
            "fixture_id": fid,
            "result": result,
            "pnl": pnl,
            "stake": stake,
            "odds": odds,
        }).execute()

        # mark tip as resolved
        sb.table(TABLE_TIPS).update({"resolved": True}).eq("id", tip["id"]).execute()

        log(f"Tip {tip['id']} settled as {result} (PnL={pnl:.2f}).")


if __name__ == "__main__":
    main()
