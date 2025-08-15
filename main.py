import time
from scheduler import start_scheduler
from utils import logger

def main():
    logger.info("Starting Football Betting Tips System")
    start_scheduler()

    # Keep running
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("Exiting...")

if __name__ == '__main__':
    main()
