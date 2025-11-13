from datetime import datetime, timedelta
from app.supabase_db import get_supabase

sb = get_supabase()


def log(msg: str):
    print(f"[ROI] {msg}")


def main():
    # yesterday UTC
    today = datetime.utcnow().date()
    day = today - timedelta(days=1)
    day_str = day.isoformat()

    # fetch yesterday's tip_results
    resp = (
        sb.table("tip_results")
        .select("*")
        .gte("resolved_at", f"{day_str} 00:00:00+00")
        .lte("resolved_at", f"{day_str} 23:59:59+00")
        .execute()
    )
    rows = resp.data or []
    if not rows:
        log(f"No results for {day_str}.")
        return

    tips_count = len(rows)
    pnl = sum(float(r["pnl"]) for r in rows)
    roi = pnl / tips_count

    sb.table("roi_log").upsert({
        "day": day_str,
        "tips_count": tips_count,
        "pnl": pnl,
        "roi": roi,
    }, on_conflict="day").execute()

    log(f"Logged ROI for {day_str}: tips={tips_count}, pnl={pnl:.2f}, roi={roi:.3f}")


if __name__ == "__main__":
    main()
