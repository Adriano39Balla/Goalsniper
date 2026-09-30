"""Frozen trials, sharp-close evidence, and actual execution receipts.

No bet placement. Quotes, Telegram deliveries, and bookmaker fills are distinct.
"""
import json
import math
import time
from risk_policy import (EXECUTION_BOOK, SHARP_BOOK, release_evidence,
                         block_interval, sharp_close)


def init_research_schema(c):
    c.execute('''CREATE TABLE IF NOT EXISTS research_trials (
        kind TEXT PRIMARY KEY, created_ts BIGINT NOT NULL, config TEXT NOT NULL,
        result TEXT, evaluated_ts BIGINT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS sharp_closes (
        match_id BIGINT NOT NULL, market_key TEXT NOT NULL, selection TEXT NOT NULL,
        kickoff_ts BIGINT NOT NULL, book TEXT NOT NULL, odds DOUBLE PRECISION NOT NULL,
        fair_prob DOUBLE PRECISION NOT NULL, observed_ts BIGINT NOT NULL,
        source_ts BIGINT NOT NULL, market_prices TEXT NOT NULL,
        PRIMARY KEY(match_id,market_key,selection))''')
    c.execute('''CREATE TABLE IF NOT EXISTS execution_receipts (
        receipt_id TEXT PRIMARY KEY, shadow_id BIGINT NOT NULL REFERENCES shadow_picks(id),
        recorded_ts BIGINT NOT NULL, event_ts BIGINT NOT NULL, book TEXT NOT NULL,
        status TEXT NOT NULL CHECK(status IN ('filled','partial','rejected')),
        requested_stake DOUBLE PRECISION NOT NULL, filled_stake DOUBLE PRECISION NOT NULL,
        filled_odds DOUBLE PRECISION, source TEXT NOT NULL,
        CHECK(requested_stake > 0 AND filled_stake >= 0 AND filled_stake <= requested_stake))''')
    c.execute('ALTER TABLE shadow_picks ADD COLUMN IF NOT EXISTS model_prob DOUBLE PRECISION')
    c.execute('ALTER TABLE research_trials ADD COLUMN IF NOT EXISTS adopted_ts BIGINT')


class ResearchStore:
    def __init__(self, app):
        self.g = app

    def signature(self, connection=None):
        audit = self.g.ScanAudit('prematch', connection=connection)
        return {'model_version': audit.model_version, 'policy_version': audit.policy_version}

    def trial(self, kind):
        with self.g.db_conn() as c:
            row = c.execute('SELECT created_ts,config,result FROM research_trials WHERE kind=%s', (kind,)).fetchone()
        return None if row is None else {'created_ts': row[0], 'config': json.loads(row[1]),
                                         'result': json.loads(row[2]) if row[2] else None}

    def start(self, kind):
        if kind not in ('release', 'threshold'):
            raise ValueError('unknown trial')
        if self.trial(kind):
            raise ValueError('trial already exists; its holdout cannot be reset or reused')
        if kind == 'release':
            with self.g.db_conn() as c:
                prior = c.execute("SELECT adopted_ts FROM research_trials WHERE kind='threshold'").fetchone()
            if prior and prior[0] is None:
                raise ValueError('finish and adopt the frozen threshold trial before release validation')
        if not self.g.CONCENTRATION_MODE or not 3 <= len(self.g.LEAGUE_ALLOW_IDS) <= 5:
            raise ValueError('choose 3–5 leagues from league-density first')
        if set(map(str, self.g.PREMATCH_LEAGUE_IDS)) != set(self.g.LEAGUE_ALLOW_IDS):
            raise ValueError('prematch and live league scopes must match')
        if not 1 <= len(self.g.ACTIVE_MARKETS) <= 2 or not self.g.ACTIVE_MARKETS <= {'BTTS', 'Over/Under 2.5'}:
            raise ValueError('release scope supports BTTS and OU2.5 only')
        self.g._MODELS_CACHE.invalidate()
        self.g._SETTINGS_CACHE.invalidate()
        for family, head in [('BTTS', 'PRE_BTTS_YES'), ('Over/Under 2.5', 'PRE_OU_2.5')]:
            if family in self.g.ACTIVE_MARKETS and not self.g.load_model_from_settings(head):
                raise ValueError('retrain clean price-free models before starting')
        active = self.g.ACTIVE_MARKETS - self.g.DISABLED_MARKETS
        if not active:
            raise ValueError('no enabled research markets')
        config = {**self.signature(), 'leagues': sorted(map(int, self.g.LEAGUE_ALLOW_IDS)),
                  'markets': sorted(active), 'execution_book': EXECUTION_BOOK,
            'sharp_book': SHARP_BOOK, 'phase': 'prematch', 'min_fixtures': 500,
                  'created_ts': int(time.time()), 'alpha': .05}
        if kind == 'threshold':
            config['strategies'] = self._select_thresholds(config)
            if not config['strategies']:
                raise ValueError('need 100 priced, sharply closed calibration fixtures per strategy')
            # Future one-day embargo, persisted once. Never a moving 70/30 split.
            config['holdout_from'] = int(time.time()) + 86400
        with self.g._tip_transaction() as c:
            c.execute('SELECT pg_advisory_xact_lock(19021)')
            # Shares the training-promotion lock; do not bind a stale model.
            current = self.signature(c)
            if any(config[k] != current[k] for k in current):
                raise ValueError('model changed while preparing trial; retry with the new model')
            config['shadow_floor'] = c.execute('SELECT COALESCE(MAX(id),0) FROM shadow_picks').fetchone()[0]
            config['decision_floor'] = c.execute('SELECT COALESCE(MAX(id),0) FROM scan_decisions').fetchone()[0]
            inserted = c.execute('INSERT INTO research_trials(kind,created_ts,config) VALUES(%s,%s,%s) ON CONFLICT(kind) DO NOTHING RETURNING kind',
                      (kind, config['created_ts'], json.dumps(config, sort_keys=True))).fetchone()
            if inserted is None:
                raise ValueError('trial already exists')
        return self.trial(kind)

    def _rows(self, config, qualified=True):
        """First quote per fixture/selection, bound to one model and policy."""
        with self.g.db_conn() as c:
            if qualified:
                rows = c.execute('''SELECT s.id,s.match_id,s.created_ts,s.kickoff_ts,s.market,
                    s.suggestion,s.prob,s.odds,s.book,m.final_goals_h,m.final_goals_a,v.status
                    FROM shadow_picks s LEFT JOIN match_results m ON m.match_id=s.match_id
                    LEFT JOIN fixture_voids v ON v.match_id=s.match_id
                    WHERE s.phase='prematch' AND s.created_ts >= %s AND s.id > %s
                    AND s.model_version=%s AND s.policy_version=%s AND s.book=%s
                    ORDER BY s.created_ts,s.id''', (config['created_ts'], config.get('shadow_floor', 0), config['model_version'],
                                                    config['policy_version'], EXECUTION_BOOK)).fetchall()
            else:
                rows = c.execute('''SELECT DISTINCT ON (d.match_id,d.suggestion)
                    d.id,d.match_id,d.created_ts,d.kickoff_ts,d.market,d.suggestion,d.prob,d.odds,
                    %s,m.final_goals_h,m.final_goals_a,v.status
                    FROM scan_decisions d JOIN scan_runs r ON r.scan_id=d.scan_id
                    LEFT JOIN match_results m ON m.match_id=d.match_id
                    LEFT JOIN fixture_voids v ON v.match_id=d.match_id
                    WHERE r.phase='prematch' AND r.model_version=%s AND r.policy_version=%s
                    AND d.stage='candidate' AND d.price_decision='qualified'
                    ORDER BY d.match_id,d.suggestion,d.created_ts,d.id''',
                    (EXECUTION_BOOK, config['model_version'], config['policy_version'])).fetchall()
            closes = c.execute('SELECT match_id,market_key,selection,fair_prob FROM sharp_closes WHERE book=%s',
                               (SHARP_BOOK,)).fetchall()
        price_map = {(r[0], r[1], r[2]): r[3] for r in closes}
        output = []
        for rid, fid, created, kickoff, market, suggestion, prob, odds, book, gh, ga, void in rows:
            if not kickoff or created >= kickoff or market.replace('PRE ', '') not in config['markets']:
                continue
            key, selection = self.g._market_key_and_selection(market, suggestion)
            p_close = price_map.get((fid, key, selection))
            result = None if gh is None or ga is None or void else self.g._tip_outcome_for_result(
                suggestion, {'final_goals_h': gh, 'final_goals_a': ga, 'btts_yes': int(gh > 0 and ga > 0)})
            output.append({'id': rid, 'match_id': fid, 'created_ts': created, 'kickoff_ts': kickoff,
                           'market': market, 'suggestion': suggestion, 'prob': float(prob), 'odds': float(odds),
                           'terminal': bool(void or (gh is not None and ga is not None)),
                           'graded': result is not None,
                           'close_ev': float(odds) * p_close - 1 if p_close is not None else None})
        return output

    def report(self, kind='release'):
        trial = self.trial(kind)
        if trial is None:
            return {'passed': False, 'reason': 'trial_not_started', 'kind': kind}
        config = trial['config']
        signature = self.signature()
        if any(config[k] != signature[k] for k in signature):
            return {'passed': False, 'reason': 'model_or_policy_changed', 'kind': kind}
        if trial['result'] is not None:
            return trial['result']
        rows = self._rows(config, qualified=kind == 'release')
        result = release_evidence(rows) if kind == 'release' else self._threshold_holdout(rows, config)
        result['kind'] = kind
        if result.get('ready'):
            # Compare-and-set: simultaneous observers get the same frozen verdict.
            with self.g.db_conn() as c:
                c.execute('UPDATE research_trials SET result=%s,evaluated_ts=%s WHERE kind=%s AND result IS NULL',
                          (json.dumps(result), int(time.time()), kind))
            return self.trial(kind)['result']
        return result

    def delivery_allowed(self, phase):
        if self.g.SHADOW_ONLY or phase != 'prematch':
            return False
        trial = self.trial('release')
        if trial is None or not (trial['result'] or {}).get('passed'):
            return False
        current = self.signature()
        return all(trial['config'][k] == current[k] for k in current)

    def capture(self, limit=200):
        now = int(time.time())
        # Includes below-confidence-threshold priced candidates for EV research.
        with self.g.db_conn() as c:
            rows = c.execute('''SELECT DISTINCT match_id,kickoff_ts,market,suggestion FROM (
                SELECT match_id,kickoff_ts,market,suggestion FROM shadow_picks WHERE phase='prematch'
                UNION SELECT d.match_id,d.kickoff_ts,d.market,d.suggestion FROM scan_decisions d
                JOIN scan_runs r ON r.scan_id=d.scan_id WHERE r.phase='prematch'
                AND d.stage='candidate' AND d.price_decision='qualified') q
                WHERE kickoff_ts > %s AND kickoff_ts <= %s
                ORDER BY kickoff_ts,match_id,market,suggestion LIMIT %s''', (now, now + 120, limit)).fetchall()
        maps, count = {}, 0
        for fid, kickoff, market, suggestion in rows:
            key, selection = self.g._market_key_and_selection(market, suggestion)
            if key not in ('BTTS', 'OU_2.5'):
                continue
            if fid not in maps:
                self.g.ODDS_CACHE.invalidate((fid, False))
                maps[fid] = self.g.fetch_odds(fid, live=False)
            close = sharp_close(maps[fid].get(key) or {}, selection, kickoff, time.time(), self.g.devig)
            if close is None:
                continue
            with self.g.db_conn() as c:
                changed = c.execute('''INSERT INTO sharp_closes VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT(match_id,market_key,selection) DO UPDATE SET
                    odds=EXCLUDED.odds,fair_prob=EXCLUDED.fair_prob,observed_ts=EXCLUDED.observed_ts,
                    source_ts=EXCLUDED.source_ts,market_prices=EXCLUDED.market_prices
                    WHERE sharp_closes.source_ts < EXCLUDED.source_ts
                    AND sharp_closes.kickoff_ts=EXCLUDED.kickoff_ts''',
                    (fid,key,selection,kickoff,SHARP_BOOK,close['odds'],close['fair_prob'],
                     close['observed_ts'],close['source_ts'],json.dumps(close['market_prices']))).rowcount
            count += changed
        return count

    def _select_thresholds(self, config):
        rows = [r for r in self._rows(config, qualified=False)
                if r['terminal'] and r['close_ev'] is not None and r['kickoff_ts'] < config['created_ts']]
        strategies = {}
        for label in sorted({r['suggestion'] for r in rows}):
            data = [r for r in rows if r['suggestion'] == label]
            if len({r['match_id'] for r in data}) < 100:
                continue
            trials = []
            for t in range(50, 86, 5):
                selected = [r for r in data if r['prob'] >= t / 100]
                if len(selected) >= 50:
                    trials.append((sum(r['close_ev'] for r in selected) / len(selected), t, len(selected)))
            if trials:
                mean, threshold, n = max(trials, key=lambda x: (x[0], -x[1]))
                strategies[label] = {'threshold_pct': threshold, 'calibration_close_ev': mean, 'calibration_n': n}
        return strategies

    def _threshold_holdout(self, rows, config):
        # One frozen combined policy, not multiple p-value searches on holdout.
        future = sorted([r for r in rows if r['created_ts'] >= config['holdout_from']],
                        key=lambda r: (r['created_ts'], r['id']))
        ids = list(dict.fromkeys(r['match_id'] for r in future))[:500]
        cohort = [r for r in future if r['match_id'] in set(ids)]
        if len(ids) < 500 or any(not r['terminal'] for r in cohort):
            return {'ready': False, 'passed': False, 'reason': 'collecting_frozen_holdout', 'fixtures': len(ids)}
        selected = [r for r in cohort if r['suggestion'] in config['strategies']
                    and r['prob'] >= config['strategies'][r['suggestion']]['threshold_pct'] / 100]
        if len(selected) < 50 or any(r['close_ev'] is None for r in selected):
            return {'ready': True, 'passed': False, 'reason': 'insufficient_complete_holdout_evidence'}
        ci = block_interval(selected)
        return {'ready': True, 'passed': bool(ci['ci95'] and ci['ci95'][0] > 0),
                'objective': 'recorded entry price EV against no-vig sharp close', 'holdout': ci,
                'strategies': config['strategies'], 'settings_changed': False,
                'note': 'Frozen combined-policy test. Threshold adoption requires a new release trial.'}

    def adopt_thresholds(self):
        """One adoption after a passed holdout, before the release trial starts."""
        report = self.report('threshold')
        if not report.get('passed'):
            raise ValueError('frozen threshold holdout has not passed')
        labels = {'BTTS: Yes': 'BTTS Yes', 'BTTS: No': 'BTTS No',
                  'Over 2.5 Goals': 'Over 2.5', 'Under 2.5 Goals': 'Under 2.5'}
        chosen = report['strategies']
        with self.g._tip_transaction() as c:
            c.execute('SELECT pg_advisory_xact_lock(19021)')
            if c.execute("SELECT 1 FROM research_trials WHERE kind='release'").fetchone():
                raise ValueError('release trial has already frozen its thresholds')
            row = c.execute("SELECT config,adopted_ts FROM research_trials WHERE kind='threshold' FOR UPDATE").fetchone()
            config = json.loads(row[0])
            if row[1] is not None:
                raise ValueError('threshold policy already adopted')
            current = self.signature(c)
            if any(config[k] != current[k] for k in current):
                raise ValueError('model or policy changed')
            for suggestion, label in labels.items():
                threshold = chosen.get(suggestion, {}).get('threshold_pct', 101)
                c.execute('''INSERT INTO settings(key,value) VALUES(%s,%s)
                    ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value''',
                    ('research_threshold:PRE '+label, str(threshold)))
            c.execute("UPDATE research_trials SET adopted_ts=%s WHERE kind='threshold'", (int(time.time()),))
        self.g._SETTINGS_CACHE.invalidate()
        return {'adopted': True, 'live_delivery': False, 'next': 'start a new release trial'}

    def record_receipt(self, data):
        required = ('receipt_id','shadow_id','event_ts','status','requested_stake','filled_stake')
        if any(k not in data for k in required):
            raise ValueError('missing receipt fields')
        if data['status'] not in ('filled','partial','rejected'):
            raise ValueError('invalid execution status')
        requested, filled = float(data['requested_stake']), float(data['filled_stake'])
        odds = float(data['filled_odds']) if data.get('filled_odds') is not None else None
        if not all(math.isfinite(x) for x in (requested, filled)) or not 0 <= filled <= requested or requested <= 0:
            raise ValueError('invalid stakes')
        status = data['status']
        if (status == 'rejected' and (filled != 0 or odds is not None)
            or status == 'filled' and filled != requested
            or status == 'partial' and not 0 < filled < requested
            or filled > 0 and (odds is None or not math.isfinite(odds) or odds <= 1)):
            raise ValueError('status, odds and stakes disagree')
        event_ts = int(data['event_ts'])
        with self.g.db_conn() as c:
            row = c.execute('SELECT created_ts,kickoff_ts,book,phase FROM shadow_picks WHERE id=%s',
                            (int(data['shadow_id']),)).fetchone()
            if row is None or row[2] != EXECUTION_BOOK or row[3] != 'prematch':
                raise ValueError('receipt requires a prematch Tipico candidate')
            if not row[0] <= event_ts < row[1] or event_ts > time.time():
                raise ValueError('receipt timestamp outside candidate execution window')
            rid = str(data['receipt_id']).strip()
            if not rid or len(rid) > 160:
                raise ValueError('invalid receipt id')
            previous = c.execute('''SELECT shadow_id,event_ts,status,requested_stake,filled_stake,filled_odds
                FROM execution_receipts WHERE receipt_id=%s''', (rid,)).fetchone()
            expected = (int(data['shadow_id']),event_ts,status,requested,filled,odds)
            if previous is not None and tuple(previous) != expected:
                raise ValueError('receipt id already used with different details')
            inserted = c.execute('''INSERT INTO execution_receipts VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT(receipt_id) DO NOTHING RETURNING receipt_id''',
                (rid,int(data['shadow_id']),int(time.time()),event_ts,EXECUTION_BOOK,status,
                 requested,filled,odds,'manual_bookmaker_receipt')).fetchone()
        return {'inserted': bool(inserted), 'source': 'manual_bookmaker_receipt',
                'note': 'User-supplied receipt, not independently verified or automatically placed.'}

    def execution_report(self):
        with self.g.db_conn() as c:
            rows = c.execute('''SELECT e.status,e.requested_stake,e.filled_stake,e.filled_odds,
                s.odds,s.suggestion,m.final_goals_h,m.final_goals_a,v.status
                FROM execution_receipts e JOIN shadow_picks s ON s.id=e.shadow_id
                LEFT JOIN match_results m ON m.match_id=s.match_id
                LEFT JOIN fixture_voids v ON v.match_id=s.match_id''').fetchall()
        result = {'book': EXECUTION_BOOK, 'receipts': len(rows), 'filled': 0, 'partial': 0, 'rejected': 0,
                  'requested_stake_eur': 0., 'filled_stake_eur': 0., 'settled_stake_eur': 0.,
                  'profit_eur_before_fees': 0., 'pending_receipts': 0, 'mean_slippage_pct': None}
        slippage = []
        for status, requested, filled, odds, quoted, selection, gh, ga, void in rows:
            result[status] += 1
            result['requested_stake_eur'] += requested
            result['filled_stake_eur'] += filled
            if filled <= 0:
                continue
            slippage.append((odds / quoted - 1) * 100)
            if not void and (gh is None or ga is None):
                result['pending_receipts'] += 1
                continue
            outcome = None if void else self.g._tip_outcome_for_result(selection,
                {'final_goals_h': gh, 'final_goals_a': ga})
            if outcome is not None:
                result['settled_stake_eur'] += filled
                result['profit_eur_before_fees'] += filled * (odds - 1 if outcome else -1)
        result['mean_slippage_pct'] = sum(slippage) / len(slippage) if slippage else None
        result['roi_pct_before_fees'] = (100 * result['profit_eur_before_fees'] / result['settled_stake_eur']
                                         if result['settled_stake_eur'] else None)
        result['source'] = 'manual receipts, not independently verified; stakes must be EUR'
        return result
