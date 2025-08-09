import asyncio
from scanner import run_scan_and_send

def main():
    result = asyncio.run(run_scan_and_send())
    print("Run finished", result, flush=True)

if __name__ == "__main__":
    main()
