from __future__ import annotations

import time
from typing import Dict, List, Set, Tuple

from . import storage

_TTL = 60
_cache: Tuple[float, Dict[str, object]] | None = None

def _flag_to_iso2(flag: str) -> str:
    cps = [ord(c) - 0x1F1E6 for c in flag if 0x1F1E6 <= ord(c) <= 0x1F1FF]
    if len(cps) == 2:
        return chr(cps[0] + 65) + chr(cps[1] + 65)
    return flag.upper()

def _norm_csv(csv: str) -> List[str]:
    return [s.strip() for s in (csv or "").split(",") if s.strip()]

def _normalize_countries(raw: List[str]) -> Set[str]:
    out: Set[str] = set()
    for s in raw:
        if any(0x1F1E6 <= ord(c) <= 0x1F1FF for c in s):
            out.add(_flag_to_iso2(s))
        else:
            out.add(s.upper())
    return out

async def get_filters() -> Dict[str, object]:
    """
    Returns:
      {
        "allowCountries": set[str],
        "allowLeagueKeywords": list[str] (UPPER),
        "excludeKeywords": list[str] (UPPER),
      }
    """
    global _cache
    now = time.time()
    if _cache and (now - _cache[0]) < _TTL:
        return _cache[1]

    cfg = await storage.get_config_bulk([
        "COUNTRY_FLAGS_ALLOW",
        "LEAGUE_ALLOW_KEYWORDS",
        "EXCLUDE_KEYWORDS",
    ])

    countries = _normalize_countries(_norm_csv(cfg.get("COUNTRY_FLAGS_ALLOW", "")))
    leagues   = [s.upper() for s in _norm_csv(cfg.get("LEAGUE_ALLOW_KEYWORDS", ""))]
    exclude   = [s.upper() for s in _norm_csv(cfg.get("EXCLUDE_KEYWORDS", ""))]

    filters = {
        "allowCountries": countries,
        "allowLeagueKeywords": leagues,
        "excludeKeywords": exclude,
    }
    _cache = (now, filters)
    return filters
