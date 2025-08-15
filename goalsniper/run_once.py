# goalsniper/run_once.py

from __future__ import annotations

import argparse
import asyncio
from typing import Optional

from .logger import log, warn


async def _run_scan() -> dict:
    from .scanner import run_scan_and_send
    res = await run_scan_and_send()
    log(
        f"[run_once] scan done: tipsSent={res.get('tipsSent', 0)} "
        f"fixturesChecked={res.get('fixturesChecked', 0)} "
        f"elapsed={res.get('elapsedSeconds', 0)}s"
    )
    return res


async def _send_digest() -> Optional[dict]:
    try:
        from .digest import send_daily_digest
        stats = await send_daily_digest()
        log(
            f"[run_once] daily digest sent "
            f"(sent={stats.get('sent')} wins={stats.get('wins')} "
            f"loss={stats.get('loss')} pending={stats.get('pending')})"
        )
        return stats
    except Exception as e:
        warn(f"[run_once] daily digest failed: {e}")
        return None


async def main():
    parser = argparse.ArgumentParser(description="Run a single Goalsniper scan (and optional daily digest).")
    parser.add_argument(
        "--daily",
        action="store_true",
        help="Run a scan (so MOTD can be selected) and then send the daily digest.",
    )
    args = parser.parse_args()

    # Always run a scan first (this also lets the MOTD logic fire inside the scanner).
    await _run_scan()

    # If --daily, follow up with the summary message.
    if args.daily:
        await _send_digest()


if __name__ == "__main__":
    asyncio.run(main())
