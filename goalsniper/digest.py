import sqlite3
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import httpx

from .telegram import send_telegram_message
from .config import TELEGRAM_CHAT_ID  # just to ensure config is loaded
from . import storage as st  # to reuse DB path

def _today_utc_start_iso() -> str:
    return datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    ).isoformat()

def _fetch_today_rows() -> List[sqlite3.Row]:
    conn = sqlite3.connect(st.DB_PATH, timeout=30, isolation_level=None)
    try:
        conn.row_factory = sqlite3.Row
        start_iso = _today_utc_start_iso()
        cur = conn.execute(
            """
            SELECT market, outcome
            FROM tips
            WHERE sent_at >= ?
            """,
            (start_iso,),
        )
        return cur.fetchall()
    finally:
        conn.close()

def _summarize(rows: List[sqlite3.Row]) -> Dict:
    total = len(rows)
    wins  = sum(1 for r in rows if r["outcome"] == 1)
    loss  = sum(1 for r in rows if r["outcome"] == 0)
    pend  = sum(1 for r in rows if r["outcome"] is None)

    by_market: Dict[str, Dict[str, int]] = {}
    for r in rows:
        m = (r["market"] or "UNKNOWN").upper()
        d = by_market.setdefault(m, {"wins": 0, "loss": 0, "pend": 0, "total": 0})
        d["total"] += 1
        if r["outcome"] == 1:
            d["wins"] += 1
        elif r["outcome"] == 0:
            d["loss"] += 1
        else:
            d["pend"] += 1

    # best market of the day (≥ 2 resolved)
    best = None
    best_acc = -1.0
    for m, d in by_market.items():
        resolved = d["wins"] + d["loss"]
        if resolved >= 2:
            acc = d["wins"] / resolved if resolved else 0.0
            if acc > best_acc:
                best_acc = acc
                best = (m, d, acc)

    return {
        "total": total,
        "wins": wins,
        "loss": loss,
        "pend": pend,
        "by_market": by_market,
        "best": best,
    }

def _format_digest(summary: Dict) -> str:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    total, wins, loss, pend = (
        summary["total"], summary["wins"], summary["loss"], summary["pend"]
    )
    resolved = wins + loss
    acc = (wins / resolved * 100.0) if resolved else 0.0

    lines = [
        f"📊 <b>Daily Digest — {today} (UTC)</b>",
        "",
        f"Total tips sent: <b>{total}</b>",
        f"Resolved: <b>{resolved}</b>  👍 {wins}  👎 {loss}  ⏳ Pending: {pend}",
        f"Accuracy (resolved): <b>{acc:.0f}%</b>",
        "",
        "<b>Markets today</b>:",
    ]

    # per‑market lines
    if summary["by_market"]:
        for m, d in sorted(summary["by_market"].items()):
            res = d["wins"] + d["loss"]
            acc_m = (d["wins"] / res * 100.0) if res else 0.0
            lines.append(
                f"• {m}: total {d['total']} | "
                f"resolved {res} (👍{d['wins']} / 👎{d['loss']}) | "
                f"acc {acc_m:.0f}%"
            )
    else:
        lines.append("• No tips today.")

    # best market
    if summary["best"]:
        m, d, acc_b = summary["best"]
        lines += [
            "",
            f"🏅 Best (≥2 resolved): <b>{m}</b> — {int(acc_b*100)}% (👍{d['wins']} / 👎{d['loss']})"
        ]

    return "\n".join(lines)

async def send_daily_digest():
    """Collect today’s stats and send a digest message to Telegram."""
    rows = _fetch_today_rows()
    summary = _summarize(rows)
    text = _format_digest(summary)
    async with httpx.AsyncClient(timeout=30) as client:
        await send_telegram_message(client, text)
    return {
        "ok": True,
        "sent": True,
        "total": summary["total"],
        "wins": summary["wins"],
        "loss": summary["loss"],
        "pending": summary["pend"],
    }
