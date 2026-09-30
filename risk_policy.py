"""Pure probability, closing-price and uncertainty policy. No network or database."""
import math
import random
from collections import defaultdict
from datetime import datetime

EXECUTION_BOOK = 'Tipico'
SHARP_BOOK = 'Pinnacle'
MODEL_WEIGHT = 0.5
MIN_GRADED = 500
MIN_DAYS = 30
MAX_CLOSE_LEAD_SEC = 120
MAX_SOURCE_AGE_SEC = 60


def shrink_probability(model, market, weight=MODEL_WEIGHT):
    if not all(math.isfinite(float(x)) for x in (model, market, weight)):
        raise ValueError('non-finite probability or weight')
    if not 0 <= model <= 1 or not 0 < market < 1 or not 0 <= weight <= 1:
        raise ValueError('invalid probability or weight')
    return weight * model + (1 - weight) * market


def source_timestamp(value):
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        return dt.timestamp() if dt.tzinfo is not None else None
    except (ValueError, TypeError, OverflowError):
        return None


def sharp_close(entry, selection, kickoff, now, devig):
    """A complete, time-stamped Pinnacle market within two minutes of kickoff.

    A sampled close is explicitly not claimed to be the exchange's final tick.
    Missing provider timestamps never become current merely by being fetched now.
    """
    observed = entry.get('fetched_ts')
    source = source_timestamp((entry.get('book_updates') or {}).get(SHARP_BOOK))
    if observed is None or source is None:
        return None
    if not 0 < kickoff - now <= MAX_CLOSE_LEAD_SEC:
        return None
    if not 0 <= now - observed <= MAX_SOURCE_AGE_SEC:
        return None
    if not 0 <= observed - source <= MAX_SOURCE_AGE_SEC or source >= kickoff:
        return None
    prices = {s: books.get(SHARP_BOOK) for s, books in entry.get('by_book', {}).items()}
    # Release scope is binary BTTS / OU2.5. Other markets need explicit policy.
    if set(prices) not in ({'Yes', 'No'}, {'Over', 'Under'}):
        return None
    if any(o is None or not math.isfinite(float(o)) or o <= 1 for o in prices.values()):
        return None
    fair = devig({s: 1 / o for s, o in prices.items()})
    p = fair.get(selection)
    if p is None or not 0 < p < 1:
        return None
    return {'odds': prices[selection], 'fair_prob': p, 'observed_ts': int(observed),
            'source_ts': int(source), 'book': SHARP_BOOK, 'market_prices': prices}


def block_interval(rows, value='close_ev', resamples=5000):
    """Fixed-seed day-cluster percentile bootstrap; all same-day bets stay together.

    Report only at a precommitted final sample. A CI is not a guarantee of edge.
    """
    groups = defaultdict(list)
    for r in rows:
        v = float(r[value])
        if not math.isfinite(v):
            raise ValueError('non-finite evidence')
        groups[int(r['kickoff_ts']) // 86400].append(v)
    blocks = list(groups.values())
    n = sum(map(len, blocks))
    if len(blocks) < MIN_DAYS or not n:
        return {'n': n, 'days': len(blocks), 'mean': None, 'ci95': None}
    sums = [sum(b) for b in blocks]
    counts = [len(b) for b in blocks]
    rng = random.Random(741921)
    samples = []
    for _ in range(resamples):
        indices = [rng.randrange(len(blocks)) for _ in blocks]
        samples.append(sum(sums[i] for i in indices) / sum(counts[i] for i in indices))
    samples.sort()
    return {'n': n, 'days': len(blocks), 'mean': sum(sums) / n,
            'ci95': [samples[int(.025 * resamples)], samples[int(.975 * resamples)]],
            'method': 'UTC-day cluster percentile bootstrap, 5000 resamples, frozen endpoint'}


def release_evidence(rows):
    """First 500 unique fixtures, retaining every qualified candidate on each.

    Selection uses creation order, never result or close availability. No later
    winners can replace missing or losing evidence. Voids are terminal but do
    not count toward 500 graded bets.
    """
    ordered = sorted(rows, key=lambda r: (r['created_ts'], r['id']))
    fixture_ids = list(dict.fromkeys(r['match_id'] for r in ordered))[:MIN_GRADED]
    cohort = [r for r in ordered if r['match_id'] in set(fixture_ids)]
    base = {'ready': False, 'passed': False, 'fixture_count': len(fixture_ids),
            'candidate_count': len(cohort), 'target_fixtures': MIN_GRADED,
            'graded': sum(bool(r['graded']) for r in cohort),
            'sharp_close_count': sum(r.get('close_ev') is not None for r in cohort),
            'pending_candidates': sum(not r['terminal'] for r in cohort)}
    if len(fixture_ids) < MIN_GRADED or any(not r['terminal'] for r in cohort):
        return {**base, 'reason': 'collecting_frozen_cohort'}
    graded = [r for r in cohort if r['graded']]
    base.update(ready=True, graded=len(graded))
    if len(graded) < MIN_GRADED:
        return {**base, 'reason': 'fewer_than_500_graded'}
    missing = sum(r.get('close_ev') is None for r in graded)
    if missing:
        return {**base, 'reason': 'incomplete_sharp_close_coverage', 'missing_closes': missing}
    ci = block_interval(graded)
    passed = ci['ci95'] is not None and ci['ci95'][0] > 0
    return {**base, 'passed': passed, 'reason': 'passed' if passed else 'clv_not_confirmed',
            'clv': ci, 'metric': 'entry_odds * no_vig_sharp_close_probability - 1'}
