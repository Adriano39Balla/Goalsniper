from __future__ import annotations

import os
import time
from typing import Dict, List, Set, Tuple

from . import storage
from .config import (
    COUNTRY_FLAGS_ALLOW as ENV_COUNTRY_FLAGS_ALLOW,
    LEAGUE_ALLOW_KEYWORDS as ENV_LEAGUE_ALLOW_KEYWORDS,
    EXCLUDE_KEYWORDS as ENV_EXCLUDE_KEYWORDS,
)

# Small cache to avoid hammering the DB every call; tunable via env
_TTL = int(os.getenv("FILTERS_CACHE_TTL", "60"))
_cache: Tuple[float, Dict[str, object]] | None = None

def _flag_to_iso2(flag: str) -> str:
    cps = [ord(c) - 0x1F1E6 for c in flag if 0x1F1E6 <= ord(c) <= 0x1F1FF]
    if len(cps) == 2:
        return chr(cps[0] + 65) + chr(cps[1] + 65)
    return (flag or "").upper()

def _norm_csv(csv: str) -> List[str]:
    return [s.strip() for s in (csv or "").split(",") if s.strip()]

def _normalize_countries(raw: List[str]) -> Set[str]:
    out: Set[str] = set()
    for s in raw:
        u = _flag_to_iso2(s)
        # basic aliases
        alias_map = {
            "USA": {"USA", "UNITED STATES", "US"},
            "UNITED STATES": {"USA", "UNITED STATES", "US"},
            "HOLLAND": {"NETHERLANDS", "HOLLAND"},
            "NETHERLANDS": {"NETHERLANDS", "HOLLAND"},
            "ENGLAND": {"ENGLAND", "GB", "UNITED KINGDOM", "UK"},
            "SCOTLAND": {"SCOTLAND", "GB", "UNITED KINGDOM", "UK"},
            "WALES": {"WALES", "GB", "UNITED KINGDOM", "UK"},
            "NORTHERN IRELAND": {"NORTHERN IRELAND", "GB", "UNITED KINGDOM", "UK"},
        }
        added = False
        for key, aliases in alias_map.items():
            if u == key:
                out.update(aliases)
                added = True
                break
        if not added:
            out.add(u)
    return out

def _expand_league_keywords(leagues: List[str]) -> List[str]:
    base = {s.upper() for s in leagues}
    # keep UEFA synonyms if user adds any, otherwise empty means allow-all
    if any(k in base for k in ("UEFA CHAMPIONS LEAGUE", "CHAMPIONS LEAGUE", "UCL")):
        base.update({"UEFA CHAMPIONS LEAGUE", "CHAMPIONS LEAGUE", "UCL"})
    if any(k in base for k in ("UEFA EUROPA LEAGUE", "EUROPA LEAGUE", "UEL")):
        base.update({"UEFA EUROPA LEAGUE", "EUROPA LEAGUE", "UEL"})
    if any(k in base for k in ("UEFA EUROPA CONFERENCE", "EUROPA CONFERENCE", "UEFA CONFERENCE LEAGUE", "UECL")):
        base.update({"UEFA EUROPA CONFERENCE", "EUROPA CONFERENCE", "UEFA CONFERENCE LEAGUE", "UECL"})
    if base & {"UCL","UEL","UECL","UEFA CHAMPIONS LEAGUE","UEFA EUROPA LEAGUE","UEFA EUROPA CONFERENCE"}:
        base.update({"QUALIFICATION","QUALIFIERS","PLAYOFF","PLAY-OFF","PRELIMINARY","GROUP"})
    return sorted(base)

def _fallback_env_or_default(db_value: str, env_value: str) -> str:
    db_value = (db_value or "").strip()
    return db_value if db_value else (env_value or "").strip()

async def get_filters() -> Dict[str, object]:
    """
    Returns:
      {
        "allowCountries": set[str],       # empty => allow all
        "allowLeagueKeywords": list[str], # empty => allow all
        "excludeKeywords": list[str],     # applied to league names (UPPER)
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

    countries_raw = _fallback_env_or_default(cfg.get("COUNTRY_FLAGS_ALLOW", ""), ENV_COUNTRY_FLAGS_ALLOW)
    leagues_raw   = _fallback_env_or_default(cfg.get("LEAGUE_ALLOW_KEYWORDS", ""), ENV_LEAGUE_ALLOW_KEYWORDS)
    exclude_raw   = _fallback_env_or_default(cfg.get("EXCLUDE_KEYWORDS", ""), ENV_EXCLUDE_KEYWORDS)

    countries = _normalize_countries(_norm_csv(countries_raw))
    leagues   = _expand_league_keywords(_norm_csv(leagues_raw))
    exclude   = [s.upper() for s in _norm_csv(exclude_raw)]

    filters = {
        "allowCountries": countries,            # set[str]; empty => allow all
        "allowLeagueKeywords": leagues,         # list[str]; empty => allow all
        "excludeKeywords": exclude,             # list[str]
    }
    _cache = (now, filters)
    return filters
