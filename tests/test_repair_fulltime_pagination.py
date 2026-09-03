"""
http_repair_fulltime_results() re-grades match_results rows whose stored
score came from the extra-time score instead of the 90-minute one, and is
documented as safe to re-run "ordered oldest-first so repeated runs make
progress".

THE BUG: a row found already correct (API score matches the stored score)
was left completely untouched, updated_ts included. Since the candidate
query is `ORDER BY updated_ts ASC LIMIT %s`, an already-correct row never
moves out of "oldest" and is re-selected — and re-verified with a fresh API
call — on every subsequent call, forever. On a real table, where only a
small minority of fixtures ever go to extra time, this pins the scan to its
first `limit` rows and the remaining rows are never reached no matter how
many times the endpoint is called.

The fix touches updated_ts on a verified-correct row too (skipped in
dry_run, which must not write), so it moves to the back of the queue exactly
like a fixed one and a second call advances to the next batch.
"""
import main


class _Cursor:
    def __init__(self, rows):
        self.rows = rows

    def execute(self, sql, params=()):
        self.sql, self.params = sql, params
        return self

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return None


class _Conn:
    """One shared cursor, so every `with db_conn() as c:` in the endpoint
    (the initial SELECT and each per-row UPDATE) hits the same call log."""

    def __init__(self, select_rows):
        self.cursor = _Cursor(select_rows)
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=()):
        self.calls.append((sql, params))
        return self.cursor.execute(sql, params)

    def fetchall(self):
        return self.cursor.fetchall()

    def fetchone(self):
        return self.cursor.fetchone()


def _fixture(status, h, a):
    return {"fixture": {"status": {"short": status}},
            "score": {"fulltime": {"home": h, "away": a}},
            "goals": {"home": h, "away": a}}


def test_verified_correct_row_gets_updated_ts_touched(monkeypatch):
    # match 1 is already stored correctly (2-1, API agrees); match 2 needs a
    # real fix (stored 1-1, API says 90-minute score was 0-0 - e.g. an AET
    # fixture recorded from the wrong field).
    conn = _Conn(select_rows=[(1, 2, 1), (2, 1, 1)])
    monkeypatch.setattr(main, "db_conn", lambda: conn)
    monkeypatch.setattr(main, "_require_admin", lambda: None)
    monkeypatch.setattr(main.time, "time", lambda: 5_000_000.0)

    fixtures = {1: _fixture("FT", 2, 1), 2: _fixture("AET", 0, 0)}
    monkeypatch.setattr(main, "_fixture_by_id", lambda mid: fixtures.get(mid))

    with main.app.test_request_context("/admin/repair/fulltime-results?limit=2"):
        resp = main.http_repair_fulltime_results()
    body = resp.get_json() if hasattr(resp, "get_json") else resp[0].get_json()

    assert body["checked"] == 2
    assert body["fixed"] == 1

    # Every UPDATE issued against match_results, keyed by match_id.
    updates = {params[-1]: (sql, params) for sql, params in conn.calls
               if sql.startswith("UPDATE match_results")}

    # The already-correct row (1) must still have been touched - a plain
    # updated_ts bump, not a score rewrite - or it stays "oldest" forever.
    assert 1 in updates, (
        "a verified-correct row's updated_ts was never touched, so "
        "ORDER BY updated_ts ASC would keep re-selecting it on every call "
        "instead of advancing to unchecked rows")
    sql1, params1 = updates[1]
    assert "final_goals_h" not in sql1, "a correct row's score must not be rewritten"
    assert params1 == (5_000_000, 1)

    # The genuinely wrong row (2) gets its score corrected AND updated_ts
    # bumped, same as before this fix.
    sql2, params2 = updates[2]
    assert "final_goals_h" in sql2
    assert params2[0] == 0 and params2[1] == 0  # gh, ga


def test_dry_run_never_writes(monkeypatch):
    conn = _Conn(select_rows=[(1, 2, 1)])
    monkeypatch.setattr(main, "db_conn", lambda: conn)
    monkeypatch.setattr(main, "_require_admin", lambda: None)

    monkeypatch.setattr(main, "_fixture_by_id", lambda mid: _fixture("FT", 2, 1))

    with main.app.test_request_context("/admin/repair/fulltime-results?dry_run=1"):
        main.http_repair_fulltime_results()

    assert not any(sql.startswith("UPDATE") for sql, _ in conn.calls)


def test_repeated_calls_advance_past_already_correct_rows(monkeypatch):
    """
    The whole point of touching updated_ts: simulate two calls against the
    same in-memory table and check the second call sees a DIFFERENT row,
    not the same one re-verified again.
    """
    table = {1: [2, 1, 0], 2: [9, 9, 0]}  # match_id -> [h, a, updated_ts]
    fixtures = {1: _fixture("FT", 2, 1), 2: _fixture("FT", 9, 9)}
    seen_per_call = []

    class _LiveConn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, sql, params=()):
            if sql.startswith("SELECT"):
                (limit,) = params
                ordered = sorted(table.items(), key=lambda kv: kv[1][2])[:limit]
                self._rows = [(mid, v[0], v[1]) for mid, v in ordered]
                seen_per_call.append([mid for mid, _v in ordered])
            elif sql.startswith("UPDATE match_results SET updated_ts"):
                ts, mid = params
                table[mid][2] = ts
            elif sql.startswith("UPDATE match_results SET final_goals_h"):
                gh, ga, _btts, ts, mid = params
                table[mid][0], table[mid][1], table[mid][2] = gh, ga, ts
            return self

        def fetchall(self):
            return self._rows

    monkeypatch.setattr(main, "db_conn", lambda: _LiveConn())
    monkeypatch.setattr(main, "_require_admin", lambda: None)
    monkeypatch.setattr(main, "_fixture_by_id", lambda mid: fixtures.get(mid))

    clock = [100]
    monkeypatch.setattr(main.time, "time", lambda: clock[0])

    with main.app.test_request_context("/admin/repair/fulltime-results?limit=1"):
        main.http_repair_fulltime_results()
    clock[0] = 200
    with main.app.test_request_context("/admin/repair/fulltime-results?limit=1"):
        main.http_repair_fulltime_results()

    assert seen_per_call[0] != seen_per_call[1], (
        "the second call re-selected the same row the first call already "
        "verified as correct, instead of advancing to the next one")
