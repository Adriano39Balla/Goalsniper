import logging
from typing import Optional, Dict, Any
import requests

# ───────────────────────── Logging ───────────────────────── #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] - %(message)s"
)
log = logging.getLogger(__name__)

# ───────────────────────── API Wrapper ───────────────────────── #

def api_get(url: str, params: Dict[str, Any], timeout: int = 5, headers: Optional[Dict[str, str]] = None) -> Optional[dict]:
    """
    Makes a GET request to the specified URL with given parameters and headers.
    Logs and returns parsed JSON if response is 200; else returns None.
    """
    try:
        response = requests.get(url, params=params, headers=headers, timeout=timeout)
        if response.status_code == 200:
            return response.json()
        else:
            log.warning("[HTTP] Non-200 response: %s | URL: %s | Params: %s", response.status_code, url, params)
    except Exception as e:
        log.exception("[HTTP] GET request failed: %s | URL: %s | Params: %s", e, url, params)

    return None
