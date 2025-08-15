# goalsniper/run_once.py
import asyncio
from .scanner import run_scan_and_send

if __name__ == "__main__":
    asyncio.run(run_scan_and_send())
