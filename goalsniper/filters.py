# goalsniper/filters.py
from __future__ import annotations

import time
from typing import Dict, List, Set, Tuple

from . import storage

_TTL = 60
_cache: Tuple[float, Dict[str, object]] | None = None


def _flag_to_iso2(flag: str) -> str:
    """Convert emoji regional-indicator flags to ISO2; otherwise return uppercased input.

    Note: some flags (e.g., Scotland 🏴) are tag-sequences (not RI pairs),
    so this function will return the emoji unchanged; we handle those below.
    """
    cps = [ord(c) - 0x1F1E6 for c in flag if 0x1F1E6 <= ord(c) <= 0x1F1FF]
    if len(cps) == 2:
        return chr(cps[0] + 65) + chr(cps[1] + 65)
    return flag.upper()


def _norm_csv(csv: str) -> List[str]:
    return [s.strip() for s in (csv or "").split(",") if s.strip()]


def _normalize_countries(raw: List[str]) -> Set[str]:
    """Normalize countries to what API-Football returns in `league.country`."""
    out: Set[str] = set()
    for s in raw:
        # Emoji → ISO2 (if applicable) or keep emoji
        v = _flag_to_iso2(s)
        # Canonical upper text
        u = v.upper()

        # Handle Scotland flag & aliases
        if s in ("🏴", "SCOTLAND", "GB-SCT", "UK-SCOTLAND"):
            out.update({"SCOTLAND", "GB", "UNITED KINGDOM", "UK"})
            continue

        # Common textual aliases
        alias_map = {
            "ENGLAND": {"ENGLAND", "GB", "UNITED KINGDOM", "UK"},
            "SCOTLAND": {"SCOTLAND", "GB", "UNITED KINGDOM", "UK"},
            "WALES": {"WALES", "GB", "UNITED KINGDOM", "UK"},
            "NORTHERN IRELAND": {"NORTHERN IRELAND", "GB", "UNITED KINGDOM", "UK"},
            "USA": {"USA", "UNITED STATES", "US"},
            "UNITED STATES": {"USA", "UNITED STATES", "US"},
            "HOLLAND": {"NETHERLANDS", "HOLLAND"},
            "NETHERLANDS": {"NETHERLANDS", "HOLLAND"},
        }
        added = False
        for key, aliases in alias_map.items():
            if u == key:
                out.update(aliases)
                added = True
                break
        if added:
            continue

        # UEFA / continental comps: if EU flag or EUROPE/EU appears, allow EUROPE/WORLD too
        if s in ("🇪🇺", "EU", "EUROPE") or u in ("EU", "EUROPE", "UEFA"):
            out.update({"EUROPE", "UEFA", "WORLD"})  # API sometimes uses "Europe" or "World"
            continue

        # Default
        out.add(u)
    return out


def _expand_league_keywords(leagues: List[str]) -> List[str]:
    """Add robust synonyms for UEFA comps & playoff/qualifier stages (uppercased)."""
    base = {s.upper() for s in leagues}

    # If user asked for CL/EL/Conf in any wording, include wide synonyms
    if any(k in base for k in ("UEFA CHAMPIONS LEAGUE", "CHAMPIONS LEAGUE", "UCL")):
        base.update({"UEFA CHAMPIONS LEAGUE", "CHAMPIONS LEAGUE", "UCL"})

    if any(k in base for k in ("UEFA EUROPA LEAGUE", "EUROPA LEAGUE", "UEL")):
        base.update({"UEFA EUROPA LEAGUE", "EUROPA LEAGUE", "UEL"})

    if any(k in base for k in ("UEFA EUROPA CONFERENCE", "EUROPA CONFERENCE", "UEFA CONFERENCE LEAGUE", "UECL")):
        base.update({"UEFA EUROPA CONFERENCE", "EUROPA CONFERENCE", "UEFA CONFERENCE LEAGUE", "UECL"})

    # If any UEFA comp was requested at all, include generic tokens that often appear in names
    if base & {"UEFA CHAMPIONS LEAGUE", "CHAMPIONS LEAGUE", "UCL",
               "UEFA EUROPA LEAGUE", "EUROPA LEAGUE", "UEL",
               "UEFA EUROPA CONFERENCE", "EUROPA CONFERENCE", "UEFA CONFERENCE LEAGUE", "UECL"}:
        base.update({"QUALIFICATION", "QUALIFIERS", "PLAYOFF", "PLAY-OFF", "PRELIMINARY", "GROUP"})

    return sorted(base)


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
    leagues_raw = _norm_csv(cfg.get("LEAGUE_ALLOW_KEYWORDS", ""))
    leagues = _expand_league_keywords(leagues_raw)
    exclude = [s.upper() for s in _norm_csv(cfg.get("EXCLUDE_KEYWORDS", ""))]

    filters = {
        "allowCountries": countries,             # set[str]
        "allowLeagueKeywords": leagues,          # list[str]
        "excludeKeywords": exclude,              # list[str]
    }
    _cache = (now, filters)
    return filters
