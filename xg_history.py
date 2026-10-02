"""Prospectively observed historical xG features; never backfilled into old decisions."""
import math


class XGHistoryFetchError(RuntimeError):
    """A historical xG request failed; never convert this into zero features."""

XG_FEATURES = [f'pm_xg_{metric}_{side}' for side in ('h', 'a')
               for metric in ('for', 'against', 'n')]


def historical_xg_features(home_id, away_id, fixtures_h, fixtures_a, cutoff, fetch_stats):
    result = dict.fromkeys(XG_FEATURES, 0.0)
    for side, team_id, fixtures in (('h', home_id, fixtures_h), ('a', away_id, fixtures_a)):
        values = []
        seen = set()
        for fx in fixtures:
            info = fx.get('fixture') or {}
            fid, stamp = info.get('id'), info.get('timestamp')
            # FT only: AET totals are a different exposure, PEN may omit FT xG.
            if not fid or fid in seen or not stamp or stamp >= cutoff or (info.get('status') or {}).get('short') != 'FT':
                continue
            seen.add(fid)
            parsed = {}
            stats = fetch_stats(fid)
            if stats is None:
                raise XGHistoryFetchError(f"historical xG fetch failed for fixture {fid}")
            for row in stats:
                tid = (row.get('team') or {}).get('id')
                for stat in row.get('statistics') or []:
                    if str(stat.get('type', '')).lower() not in ('expected_goals', 'expected goals', 'xg'):
                        continue
                    try:
                        val = float(stat['value'])
                        if math.isfinite(val) and 0 <= val <= 15:
                            parsed[tid] = val
                    except (TypeError, ValueError, KeyError):
                        pass
            if team_id in parsed and len(parsed) == 2:
                values.append((parsed[team_id], next(v for k, v in parsed.items() if k != team_id)))
        if values:
            result[f'pm_xg_for_{side}'] = sum(x[0] for x in values) / len(values)
            result[f'pm_xg_against_{side}'] = sum(x[1] for x in values) / len(values)
            result[f'pm_xg_n_{side}'] = float(len(values))
    return result
