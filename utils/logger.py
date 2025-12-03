import logging
import sys
from datetime import datetime
from loguru import logger
import json
from pathlib import Path

class InterceptHandler(logging.Handler):
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )

def setup_logger():
    """Setup comprehensive logging system"""
    
    # Remove default handler
    logger.remove()
    
    # Custom format with colors
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    
    # Console logging
    logger.add(
        sys.stdout,
        format=log_format,
        level="INFO",
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    
    # File logging - Detailed
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logger.add(
        log_dir / "betting_{time:YYYY-MM-DD}.log",
        format=log_format,
        level="DEBUG",
        rotation="00:00",
        retention="30 days",
        compression="zip",
        backtrace=True,
        diagnose=True
    )
    
    # JSON logging for structured analysis
    logger.add(
        log_dir / "structured_{time:YYYY-MM-DD}.jsonl",
        format=lambda record: json.dumps({
            "time": record["time"].isoformat(),
            "level": record["level"].name,
            "message": record["message"],
            "module": record["name"],
            "function": record["function"],
            "line": record["line"],
            "extra": record["extra"]
        }),
        level="INFO",
        rotation="00:00",
        retention="30 days",
        serialize=True
    )
    
    # Intercept standard logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    return logger

# Global logger instance
logger = setup_logger()
