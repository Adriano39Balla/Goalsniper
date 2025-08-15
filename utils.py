import logging
from config import LOG_LEVEL

def setup_logger():
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger()

logger = setup_logger()
