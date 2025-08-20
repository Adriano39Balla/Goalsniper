# app/utils.py
import os
import logging
from typing import List, Iterable


def chunk_list(seq: List, size: int) -> Iterable[List]:
    """
    Yield successive chunks of size `size` from list `seq`.
    """
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def require_env_var(name: str) -> str:
    """
    Ensure an environment variable is set, else raise.
    """
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val


def setup_logging(level: int = logging.INFO):
    """
    Configure global logging with timestamp + level.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def pct_str(v: float, digits: int = 1) -> str:
    """
    Format a float as percentage string.
    """
    try:
        return f"{100.0 * float(v):.{digits}f}%"
    except Exception:
        return "0%"


def safe_float(x, default: float = 0.0) -> float:
    """
    Convert value to float safely.
    """
    try:
        return float(x)
    except Exception:
        return float(default)
