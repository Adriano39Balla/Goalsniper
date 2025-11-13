# log_roi.py
from datetime import datetime, timedelta
from app.supabase_db import get_supabase

sb = get_supabase()


def log(msg: str):
    print(f"[ROI] {msg}")


def main():
    # ROI for yesterday
    utc_today = datetime.utcnow().date()
    day = utc_today - timedelta(days=1)

    start = f"{day} 00:00:00+00"
    end   = f"{day} 23:59:59+00"

    # -------------------------------------------------------
    # Step 1: Fetch tip_results resolved yesterday
    # -------------------------------------------------------
    resp = (
        sb.table("tip_results")
        .select("*, tips(sent_ok)")
        .gte("resolved_at", start)
        .lte("resolved_at", end)
        .execute()
    )
    rows = resp.data or []

    if not rows:
        log(f"No bets resolved on {day}.")
        return

    # -------------------------------------------------------
    # Step 2: Only include bets that the bot actually sent
    # -------------------------------------------------------
    real_bets = [r for r in rows if r.get("tips", {}).get("sent_ok") == 1]

    if not real_bets:
        log(f"No real tips (sent_ok=1) resolved on {day}.")
        return

    # -------------------------------------------------------
    # Step 3: Compute PnL + ROI
    # -------------------------------------------------------
    tips_count = len(real_bets)
    pnl = sum(float(r.get("pnl", 0.0)) for r in real_bets)
    roi = pnl / tips_count

    # -------------------------------------------------------
    # Step 4: Store into roi_log
    # -------------------------------------------------------
    sb.table("roi_log").upsert(
        {
            "day": day.isoformat(),
            "tips_count": tips_count,
            "pnl": pnl,
            "roi": roi,
        },
        on_conflict="day",
    ).execute()

    log(f"Logged ROI for {day} → tips={tips_count}, pnl={pnl:.2f}, roi={roi:.3f}")


if __name__ == "__main__":
    main()
