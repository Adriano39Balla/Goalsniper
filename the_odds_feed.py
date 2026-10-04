"""The Odds API prematch adapter. Secrets are environment-only.

Exact team names (case/punctuation normalized), oriented home/away, sport and
kickoff are required. Ambiguous matches fail closed; no fuzzy name matching.
"""
import math
import re
import time
import unicodedata
from datetime import datetime
from threading import RLock
import requests

SPORTS = {5: 'soccer_uefa_nations_league', 39: 'soccer_epl', 140: 'soccer_spain_la_liga',
          135: 'soccer_italy_serie_a', 78: 'soccer_germany_bundesliga',
          61: 'soccer_france_ligue_one', 88: 'soccer_netherlands_eredivisie',
          94: 'soccer_portugal_primeira_liga'}
BOOKS = {'tipico_de': 'Tipico', 'pinnacle': 'Pinnacle'}


def timestamp(value):
    try:
        d = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        return d.timestamp() if d.tzinfo else None
    except (ValueError, TypeError):
        return None


def name(value):
    return re.sub(r'[^a-z0-9]', '', unicodedata.normalize('NFKD', str(value)).encode('ascii', 'ignore').decode().lower())


def same_event(event, fixture, sport):
    try:
        kickoff = timestamp(event['commence_time'])
        return (event.get('sport_key') == sport and kickoff is not None
                and abs(kickoff - int(fixture['fixture']['timestamp'])) <= 60
                and name(event['home_team']) == name(fixture['teams']['home']['name'])
                and name(event['away_team']) == name(fixture['teams']['away']['name'])
                and bool(name(event['home_team'])) and bool(name(event['away_team'])))
    except (KeyError, TypeError, ValueError):
        return False


def normalize(event, now):
    """One row per complete book/market, retaining its market timestamp."""
    rows = []
    kickoff = timestamp(event.get('commence_time'))
    if kickoff is None or kickoff <= now:
        return rows
    for book in event.get('bookmakers') or []:
        if not isinstance(book, dict) or book.get('key') not in BOOKS:
            continue
        for market in book.get('markets') or []:
            if not isinstance(market, dict) or market.get('key') not in ('btts', 'totals', 'alternate_totals'):
                continue
            updated = timestamp(market.get('last_update'))
            # Never substitute fetch time or a different market/book timestamp.
            if updated is None or not 0 <= now - updated <= 300:
                continue
            sides = {}
            bad = False
            for outcome in market.get('outcomes') or []:
                try:
                    side = outcome['name']
                    if market['key'] != 'btts' and float(outcome.get('point', -1)) != 2.5:
                        continue
                    allowed = ('Yes', 'No') if market['key'] == 'btts' else ('Over', 'Under')
                    price = float(outcome['price'])
                    if side not in allowed or side in sides or not math.isfinite(price) or price <= 1:
                        bad = True
                        break
                    sides[side] = price
                except (KeyError, TypeError, ValueError):
                    bad = True
                    break
            expected = {'Yes', 'No'} if market['key'] == 'btts' else {'Over', 'Under'}
            if bad or set(sides) != expected:
                continue
            values = [{'value': s if market['key'] == 'btts' else s + ' 2.5', 'odd': p}
                      for s, p in sides.items()]
            rows.append({'update': market['last_update'], 'provider': 'the_odds_api',
                         'bookmakers': [{'name': BOOKS[book['key']], 'bets': [{
                             'name': 'Both Teams Score' if market['key'] == 'btts' else 'Goals Over/Under',
                             'values': values}]}]})
    return rows


class OddsFeed:
    def __init__(self, key, reserve, session=None, reconcile=None):
        self.key = key
        self.reserve = reserve  # durable, cross-worker budget reservation
        self.reconcile = reconcile
        self.session = session or requests.Session()
        self.lock = RLock()
        self.cache = {}
        self.status = {'status': 'not_checked', 'configured': bool(key)}
        self.blocked_until = 0

    def get(self, path, cost=0, **params):
        self.status.update(http_status=None, last_cost=None)
        if not self.key:
            self.status['status'] = 'missing_key'
            return None
        if time.time() < self.blocked_until:
            self.status['status'] = 'provider_cooldown'
            return None
        reservation = self.reserve(cost) if cost else None
        if cost and not reservation:
            self.status['status'] = 'local_budget_exhausted'
            return None
        try:
            r = self.session.get('https://api.the-odds-api.com/v4/' + path,
                params={'apiKey': self.key, **params}, timeout=(5, 12), allow_redirects=False)
            self.status.update(http_status=r.status_code,
                remaining=r.headers.get('x-requests-remaining'),
                used=r.headers.get('x-requests-used'), last_cost=r.headers.get('x-requests-last'))
            # Reconcile only an explicit provider charge. A timeout or missing
            # header keeps the reservation; never assume an unknown cost is zero.
            billed = r.headers.get('x-requests-last')
            if cost and self.reconcile and str(billed).isdigit():
                self.reconcile(reservation, int(billed))
            if r.status_code != 200:
                self.status['status'] = 'provider_http_error'
                self.blocked_until = time.time() + (3600 if r.status_code in (401, 403, 422) else 60)
                return None
            data = r.json()
            self.status['status'] = 'ok'
            return data
        except (requests.RequestException, ValueError):
            # Exception text and response bodies can contain the credential URL.
            self.status['status'] = 'network_or_json_error'
            self.blocked_until = time.time() + 60
            return None

    def check(self):
        with self.lock:
            # A successful catalogue request says nothing about the last
            # fixture's prices. Keep these two diagnostic scopes independent.
            fixture_status = self.status
            self.status = {'status': 'not_checked', 'configured': bool(self.key)}
            try:
                data = self.get('sports')
                return {**self.status, 'sports_available': len(data) if isinstance(data, list) else None}
            finally:
                self.status = fixture_status

    def rows(self, fixture):
        with self.lock:
            now = time.time()
            sport = SPORTS.get(int(fixture.get('league', {}).get('id', 0)))
            kickoff = int(fixture.get('fixture', {}).get('timestamp', 0))
            home = (fixture.get('teams', {}).get('home') or {}).get('name')
            away = (fixture.get('teams', {}).get('away') or {}).get('name')
            self.status.update(sport_key=sport, fixture_kickoff=kickoff,
                               fixture_id=fixture.get('fixture', {}).get('id'),
                               fixture_home_team=home, fixture_away_team=away,
                               event_id=None, complete_book_markets=0, event_samples=[],
                               http_status=None, last_cost=None,
                               event_count=None, matched_event_count=0,
                               matched_event_id=None, bookmakers=[], markets=[],
                               quote_age_seconds=None)
            if not sport or kickoff <= now:
                self.status['status'] = 'unsupported_or_started_fixture'
                return []
            if fixture.get('fixture', {}).get('status', {}).get('short') != 'NS':
                self.status['status'] = 'fixture_not_scheduled'
                return []
            cache_key = ('events', sport)
            cached = self.cache.get(cache_key)
            if cached is None or now - cached[0] >= 300:
                events = self.get('sports/' + sport + '/events')
                if not isinstance(events, list):
                    self.status['status'] = 'events_request_failed'
                    return []
                self.cache[cache_key] = (now, events)
            else:
                events = cached[1]
            self.status['event_count'] = len(events)
            matches = [e for e in events if isinstance(e, dict) and same_event(e, fixture, sport)]
            self.status['matched_event_count'] = len(matches)
            if len(matches) != 1 or not re.fullmatch('[a-zA-Z0-9_-]+', str(matches[0].get('id', ''))):
                # Keep a redacted sample so the admin endpoint explains name,
                # kickoff, or sport-key mismatches without exposing quotes or
                # credentials.
                # Rank all events by team identity first, then kickoff proximity.
                # The first five catalogue entries need not resemble this fixture.
                candidates = sorted(
                    (e for e in events if isinstance(e, dict)),
                    key=lambda e: (
                        -(int(bool(home) and name(e.get('home_team', '')) == name(home))
                          + int(bool(away) and name(e.get('away_team', '')) == name(away))),
                        abs(timestamp(e.get('commence_time')) - kickoff)
                        if timestamp(e.get('commence_time')) is not None else float('inf')))
                self.status['event_samples'] = [
                    {'id': e.get('id'), 'sport_key': e.get('sport_key'),
                     'home_team': e.get('home_team'), 'away_team': e.get('away_team'),
                     'commence_time': e.get('commence_time'),
                     'home_name_matches': bool(home) and name(e.get('home_team', '')) == name(home),
                     'away_name_matches': bool(away) and name(e.get('away_team', '')) == name(away),
                     'sport_matches': e.get('sport_key') == sport,
                     'kickoff_delta_seconds': (timestamp(e.get('commence_time')) - kickoff)
                         if timestamp(e.get('commence_time')) is not None else None}
                    for e in candidates[:5]]
                self.status['status'] = 'unmatched_or_ambiguous_fixture'
                return []
            event_id = matches[0]['id']
            self.status['matched_event_id'] = event_id
            cache_key = ('odds', event_id)
            cached = self.cache.get(cache_key)
            if cached is None or now - cached[0] >= 20:
                data = self.get('sports/' + sport + '/events/' + event_id + '/odds', cost=3,
                    bookmakers='tipico_de,pinnacle', markets='btts,totals,alternate_totals', oddsFormat='decimal', dateFormat='iso')
                if not isinstance(data, dict) or data.get('id') != event_id or not same_event(data, fixture, sport):
                    if self.status['status'] == 'ok':
                        self.status['status'] = 'invalid_event_response'
                    return []
                self.cache[cache_key] = (now, data)
            else:
                data = cached[1]
            books = data.get('bookmakers') or []
            self.status['bookmakers'] = [b.get('key') for b in books if isinstance(b, dict)]
            self.status['markets'] = sorted({m.get('key') for b in books if isinstance(b, dict)
                                             for m in (b.get('markets') or [])
                                             if isinstance(m, dict) and m.get('key')})
            rows = normalize(data, time.time())
            if rows:
                ages = [time.time() - (timestamp(r.get('update')) or time.time()) for r in rows]
                self.status['quote_age_seconds'] = round(max(ages), 1)
            self.status.update(status='quotes_available' if rows else 'no_complete_fresh_markets',
                               event_id=event_id, complete_book_markets=len(rows))
            return rows
