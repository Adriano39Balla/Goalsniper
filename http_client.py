import requests
from typing import Optional, Dict

def api_get(url: str, params: Dict, timeout: int = 5) -> Optional[dict]:
    try:
        response = requests.get(url, params=params, timeout=timeout)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None
