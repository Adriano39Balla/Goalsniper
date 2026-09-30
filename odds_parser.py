"""Pure full-match odds normalization; independent of Flask and database state."""
import math
import re
from typing import Any, Dict, List, Optional, Tuple

def _fmt_line(line):
    return str(float(line)).rstrip("0").rstrip(".")

def _txt(v: Any) -> str:
    """
    Coerce an odds-feed field to a string.

    THE BUG THIS FIXES: 530 occurrences of
        [ODDS] parse failed ... 'int' object has no attribute 'lower'
    in six hours. The parser did `(v.get("value") or "").strip().lower()` and
    `(mkt.get("name","")).lower()`. When the feed returns a NUMBER rather than a
    string — which it does for some bookmakers' market names and for Asian-style
    Over/Under values — `int or ""` evaluates to the int (it's truthy), which is
    then handed straight to .lower()/.strip(). None, ints and floats all become
    strings here instead.
    """
    if v is None:
        return ""
    return v if isinstance(v, str) else str(v)


# Bet names that mention "total"/"goals" but are NOT the match over/under.
# API-Football's catalogue also carries team totals ("Total - Home"), halves
# ("Goals Over/Under First Half"), corners, cards, exact-score and odd/even
# markets - and every one of them quotes a plain "Over 2.5" label, so they
# used to be folded into OU_2.5 alongside the real match total.
#
# That was not cosmetic. fetch_odds keeps the BEST price per selection, and a
# single team scoring 3+ prices around 4.0-9.0 against ~1.9 for the match
# total, so the wrong price won the comparison every time. The inflated price
# then flowed into the EV gate (tipping bets whose real price never existed)
# and into P&L - which is what produced a 116.8% ROI on 310 PRE Over/Under
# 2.5 bets at a 52.9% win rate, implying average winning odds of ~4.1 on a
# market that trades between about 1.4 and 3.0.
#
# Asian/handicap lines are excluded deliberately too: their quarter lines
# (2.25, 2.75) settle half-win/half-loss, and _tip_outcome_for_result grades
# a straight win or loss, so pricing off them would misgrade the bet.
# Scope qualifiers that make a bet something other than the FULL-MATCH
# market, whichever family it otherwise names. This is not an Over/Under
# quirk - every family had the same hole:
#
#   "Both Teams To Score - First Half"  -> BTTS
#   "First Half Winner"                 -> 1X2
#   "Double Chance - First Half"        -> DC
#   "Draw No Bet (1st Half)"            -> DNB
#
# A half is a shorter sample than a match, so its decisive outcomes always
# price LONGER than the full-time equivalent (half-time BTTS ~3.5 against
# ~1.9, half-time home ~2.5 against ~1.8). fetch_odds keeps the BEST price
# per selection, so the half price won every comparison and became the
# recorded price for a full-match bet - inflating EV, the tip decision, and
# the P&L, in every market rather than just Over/Under.
_NOT_FULL_MATCH_SCOPE = (
    "half", "halves", "1st", "2nd", "first", "second", "quarter", "period",
    "minute", "extra", "overtime", "incl", "penalt", "shootout",
    "corner", "card", "booking", "offside", "foul", "shot", "save",
    "player", "exact", "odd", "even", "handicap", "asian",
)

# Additionally for the goals total: a TEAM's total is not the MATCH total.
# "Total - Home" quotes a plain "Over 2.5" priced ~4.0-9.0 (that team
# scoring 3+) against ~1.9 for the match. Kept separate from the list above
# because "Both Teams To Score" legitimately contains "team".
_OU_NOT_MATCH_TOTAL = ("home", "away", "team")


def _market_name_normalize(s: Any) -> str:
    # Exact aliases only. A substring such as "winner" also matches combined
    # winner/total markets, which must never price a straight match outcome.
    name = " ".join(_txt(s).strip().casefold().replace("-", " ").split())
    aliases = {
        "1X2": {"match winner", "fulltime result", "full time result", "1x2", "winner"},
        "BTTS": {"both teams score", "both teams to score", "btts"},
        "DC": {"double chance"},
        "DNB": {"draw no bet"},
        "OU": {"goals over/under", "over/under", "over/under line", "match goals",
               "total goals", "total goals over/under", "match total goals"},
    }
    return next((key for key, names in aliases.items() if name in names), "")


# The in-play feed is one aggregated source rather than a panel of books, so
# it gets a stable name of its own: n_books is then honestly 1 for live,
# instead of borrowing the credibility of a multi-book consensus.
LIVE_FEED_BOOK = "API-Football (in-play)"


def _iter_price_sources(r: dict) -> List[Tuple[str, List[dict]]]:
    """
    (book_name, bets) pairs, for BOTH shapes the odds API returns.

    /odds (prematch) nests markets under a list of "bookmakers".
    /odds/live returns a single aggregated in-play feed with the markets
    directly under "odds" and no bookmaker layer at all.

    Only the first was handled, so every live fixture parsed to zero markets
    no matter what the feed contained - which is why in-play candidates came
    back no_odds 100% of the time, on every scan, while prematch priced
    normally off the same code. Nothing downstream was wrong; the prices
    never arrived.
    """
    books = r.get("bookmakers")
    if isinstance(books, list) and books:
        return [(_txt(bk.get("name")) or "Book", bk.get("bets") or []) for bk in books]
    live_odds = r.get("odds")
    if isinstance(live_odds, list) and live_odds:
        return [(LIVE_FEED_BOOK, live_odds)]
    return []


def _odd_value(v: dict) -> float:
    """Parse a price, tolerating strings, commas and nulls. 0.0 means unusable."""
    try:
        # In-play selections carry a suspended flag while the market is
        # frozen. A suspended price cannot be taken, so it is not a price.
        if not isinstance(v, dict) or _price_suspended(v):
            return 0.0
        raw = v.get("odd")
        if raw is None:
            return 0.0
        value = float(_txt(raw).replace(",", "."))
        return value if math.isfinite(value) and value > 1.0 else 0.0
    except Exception:
        return 0.0


def _price_suspended(obj: dict) -> bool:
    return any(str(obj.get(k, "")).strip().lower() in ("true", "1", "yes")
               for k in ("suspended", "blocked", "stopped"))


def parse_book_market(mkt: dict, ou_lines=(2.5, 3.5)) -> Optional[Tuple[str, Dict[str, float]]]:
    """Parse one bookmaker's one market into {market_key: {selection: odds}}."""
    if not isinstance(mkt, dict) or _price_suspended(mkt):
        return None
    mname = _market_name_normalize(mkt.get("name"))
    values = mkt.get("values")
    vals = [v for v in values if isinstance(v, dict)] if isinstance(values, list) else []
    if mname == "BTTS":
        d = {}
        for v in vals:
            lbl = _txt(v.get("value")).strip().lower()
            o = _odd_value(v)
            if o <= 1.0:
                continue
            if lbl == "yes":
                d["Yes"] = o
            elif lbl == "no":
                d["No"] = o
        return ("BTTS", d) if set(d) == {"Yes", "No"} else None
    if mname == "1X2":
        d = {}
        for v in vals:
            lbl = _txt(v.get("value")).strip().lower()
            o = _odd_value(v)
            if o <= 1.0:
                continue
            if lbl in ("home", "1"):
                d["Home"] = o
            elif lbl in ("draw", "x"):
                d["Draw"] = o
            elif lbl in ("away", "2"):
                d["Away"] = o
        return ("1X2", d) if set(d) == {"Home", "Draw", "Away"} else None
    if mname == "DC":
        d = {}
        for v in vals:
            lbl = _txt(v.get("value")).strip().lower().replace(" ", "")
            o = _odd_value(v)
            if o <= 1.0:
                continue
            if lbl in ("home/draw", "1x", "homeordraw"):
                d["1X"] = o
            elif lbl in ("draw/away", "away/draw", "x2", "draworaway", "awayordraw"):
                d["X2"] = o
            elif lbl in ("home/away", "12", "homeoraway"):
                d["12"] = o
        return ("DC", d) if set(d) == {"1X", "X2", "12"} else None
    if mname == "DNB":
        d = {}
        for v in vals:
            lbl = _txt(v.get("value")).strip().lower()
            o = _odd_value(v)
            if o <= 1.0:
                continue
            if lbl in ("home", "1"):
                d["Home"] = o
            elif lbl in ("away", "2"):
                d["Away"] = o
        return ("DNB", d) if set(d) == {"Home", "Away"} else None
    if mname == "OU":
        by_line: Dict[str, Dict[str, float]] = {}
        for v in vals:
            lbl = _txt(v.get("value")).strip().lower()
            # Validate the selection as well as the enclosing market. Generic
            # market names can contain team/period selections which must not
            # be folded into a full-match total.
            if any(bad in lbl for bad in _NOT_FULL_MATCH_SCOPE + _OU_NOT_MATCH_TOTAL):
                continue
            side_match = re.fullmatch(r"(over|under)(?:\s+(\d+(?:[.,]\d+)?))?", lbl)
            if not side_match:
                continue
            o = _odd_value(v)
            if o <= 1.0:
                continue
            label_line = float(side_match.group(2).replace(",", ".")) if side_match.group(2) else None
            handicap = v.get("handicap")
            try:
                handicap_line = (float(_txt(handicap).replace(",", "."))
                                  if handicap not in (None, "") else None)
            except (TypeError, ValueError):
                continue
            # Prematch normally embeds the line in `value`; live odds place it
            # in `handicap`. If both exist, they must describe the same line.
            if (label_line is not None and handicap_line is not None
                    and abs(label_line - handicap_line) > 1e-6):
                continue
            ln = handicap_line if handicap_line is not None else label_line
            if ln is None or not math.isfinite(ln) or ln % 1 != 0.5:
                continue
            # Only exact lines the service trains and grades are eligible;
            # quarter/integer Asian lines settle differently.
            if not any(abs(ln - configured) <= 1e-6 for configured in ou_lines):
                continue
            key = f"OU_{_fmt_line(ln)}"
            side = "Over" if side_match.group(1) == "over" else "Under"
            by_line.setdefault(key, {})[side] = o
        complete = {k: sides for k, sides in by_line.items()
                    if set(sides) == {"Over", "Under"}}
        return ("OU_MULTI", complete) if complete else None
    return None


# Selections required before a market can be de-vigged, and what its true
# probabilities sum to. OU lines are keyed OU_<line> at runtime and default to
# 2 selections / total 1.0.
_MARKET_SELECTION_COUNT = {"BTTS": 2, "1X2": 3, "DC": 3, "DNB": 2}
