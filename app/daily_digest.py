from datetime import datetime, timedelta
from app.supabase_db import get_supabase
from app.telegram_bot import send_telegram

sb = get_supabase()


def log(msg):
    print(f"[DIGEST] {msg}")


def fetch_yesterday_range():
    today = datetime.utcnow().date()
    day = today - timedelta(days=1)
    start = f"{day} 00:00:00+00"
    end = f"{day} 23:59:59+00"
    return day, start, end


def fetch_tips_sent(start, end):
    """Fetch all tips SENT (sent_ok = 1) yesterday."""
    resp = (
        sb.table("tips")
        .select("*")
        .gte("created_ts", start)
        .lte("created_ts", end)
        .eq("sent_ok", 1)
        .execute()
    )
    return resp.data or []


def fetch_resolved_tips(start, end):
    """Fetch settlements from tip_results."""
    resp = (
        sb.table("tip_results")
        .select("*, tips(match_id, market, suggestion)")
        .gte("resolved_at", start)
        .lte("resolved_at", end)
        .execute()
    )
    return resp.data or []


def compute_market_stats(resolved_rows):
    """
    Build market-by-market:
      wins / total (accuracy%) and ROI
    """
    markets = {}

    for row in resolved_rows:
        tip = row["tips"] or {}
        market = tip.get("market")
        if not market:
            continue

        if market not in markets:
            markets[market] = {
                "wins": 0,
                "count": 0,
                "pnl": 0.0,
            }

        markets[market]["count"] += 1
        markets[market]["pnl"] += float(row.get("pnl", 0))

        if row.get("result") == "win":
            markets[market]["wins"] += 1

    # compute accuracy and ROI for each market
    out = []
    for m, v in markets.items():
        wins = v["wins"]
        total = v["count"]
        pnl = v["pnl"]
        accuracy = wins / total if total else 0
        roi = pnl / total if total else 0

        out.append(
            f"• <b>{m}</b> — {wins}/{total} ({accuracy*100:.1f}%) • ROI {roi*100:+.1f}%"
        )

    return "\n".join(out)


def format_recent_tips(resolved_rows):
    """Last 3 resolved tips by time."""
    rows = sorted(
        resolved_rows,
        key=lambda r: r.get("resolved_at", ""),
        reverse=True,
    )[:3]

    out = []
    for r in rows:
        tip = r["tips"] or {}
        prob = float(tip.get("confidence", 0)) * 100
        sel = tip.get("suggestion", "").upper()
        mkt = tip.get("market", "")
        minute = tip.get("minute") or 0
        out.append(f"{mkt}: {sel} ({prob:.1f}%) - {minute:02d}'")

    return " • ".join(out)


def send_daily_digest():
    day, start, end = fetch_yesterday_range()

    tips_sent = fetch_tips_sent(start, end)
    resolved = fetch_resolved_tips(start, end)

    sent_count = len(tips_sent)
    resolved_count = len(resolved)
    wins = sum(1 for r in resolved if r.get("result") == "win")
    accuracy = wins / resolved_count if resolved_count else 0

    pnl = sum(float(r.get("pnl", 0)) for r in resolved)
    roi = pnl / resolved_count if resolved_count else 0

    market_stats = compute_market_stats(resolved)
    recent = format_recent_tips(resolved)

    msg = (
        f"📊 <b>Daily Accuracy Digest — {day}</b>\n"
        f"Tips sent: <b>{sent_count}</b>  •  Graded: <b>{resolved_count}</b>  •  Wins: <b>{wins}</b>  •  Accuracy: <b>{accuracy*100:.1f}%</b>\n\n"
        f"🕒 <b>Recent tips:</b> {recent}\n\n"
        f"{market_stats}\n"
    )

    send_telegram(msg)
    log("Daily accuracy digest sent.")


if __name__ == "__main__":
    send_daily_digest()
