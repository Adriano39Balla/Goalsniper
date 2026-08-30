"""
_blocked_league() decides whether a fixture's league is scanned at all (both
live and prematch). Default behaviour blocks by name pattern (youth/reserve/
friendly) or an ID denylist; LEAGUE_ALLOW_IDS, when set, overrides both with
a hard allowlist so only those league IDs are ever scanned.
"""
import main


def _league(id_=1, name="Premier League", country="England", type_="League"):
    return {"id": id_, "name": name, "country": country, "type": type_}


def test_youth_leagues_are_blocked_by_default(monkeypatch):
    monkeypatch.delenv("LEAGUE_ALLOW_IDS", raising=False)
    monkeypatch.delenv("LEAGUE_DENY_IDS", raising=False)
    for name in ["Premier League U21", "UEFA U19 Championship", "U17 World Cup"]:
        assert main._blocked_league(_league(name=name)) is True


def test_reserve_and_friendly_leagues_are_blocked_by_default(monkeypatch):
    monkeypatch.delenv("LEAGUE_ALLOW_IDS", raising=False)
    monkeypatch.delenv("LEAGUE_DENY_IDS", raising=False)
    assert main._blocked_league(_league(name="Bundesliga II Reserve")) is True
    assert main._blocked_league(_league(name="Club Friendlies")) is True


def test_normal_league_is_not_blocked_by_default(monkeypatch):
    monkeypatch.delenv("LEAGUE_ALLOW_IDS", raising=False)
    monkeypatch.delenv("LEAGUE_DENY_IDS", raising=False)
    assert main._blocked_league(_league()) is False


def test_league_deny_ids_blocks_by_id_when_no_allowlist(monkeypatch):
    monkeypatch.delenv("LEAGUE_ALLOW_IDS", raising=False)
    monkeypatch.setenv("LEAGUE_DENY_IDS", "39,140")
    assert main._blocked_league(_league(id_=39)) is True
    assert main._blocked_league(_league(id_=61)) is False


def test_allow_ids_set_blocks_everything_not_listed(monkeypatch):
    # A normal top-flight league not on the allowlist is now blocked too.
    monkeypatch.setenv("LEAGUE_ALLOW_IDS", "39,140")
    assert main._blocked_league(_league(id_=39)) is False
    assert main._blocked_league(_league(id_=140)) is False
    assert main._blocked_league(_league(id_=61, name="Ligue 1")) is True


def test_allow_ids_overrides_name_patterns_and_deny_list(monkeypatch):
    # The allowlist is a hard override: even a name that would normally be
    # blocked passes if its id is explicitly allowed, and LEAGUE_DENY_IDS is
    # ignored entirely once an allowlist is set.
    monkeypatch.setenv("LEAGUE_ALLOW_IDS", "999")
    monkeypatch.setenv("LEAGUE_DENY_IDS", "999")
    assert main._blocked_league(_league(id_=999, name="Some U21 Friendly Reserve Cup")) is False
