"""
/admin/diagnostics/blind-snapshots measures damage already in the tables.

The write-side bug is fixed (a failed API call is no longer persisted as
training data — see test_empty_observation_not_harvested.py), but every row
written before that fix is still sitting in tip_snapshots and
prematch_snapshots, and the loaders have no filter that would drop them:
load_inplay_data() has no coverage test at all, and load_prematch_data()'s
only filter is `if not feat`, which never fires on a vector that is
populated-with-zeros. So the fix stops the bleeding and changes nothing about
what is already there.

This endpoint answers the one question that decides what to do next: how much
of the CURRENT training set records no observation at all. The answer is the
difference between "retrain and move on" and "every head fitted on this was
fitted on noise, so no conclusion drawn from it stands".

Counting it wrong in either direction is expensive — overstating it argues for
throwing away good data, understating it leaves a poisoned fit in place — so
the classification rules are pinned here.

The SQL itself was also run against a real PostgreSQL instance holding a
deliberately mixed set of rows (legacy schema-1 with stats, schema-2 with
stats, schema-2 blind inside the training window, schema-2 blind outside it,
prematch blind on one side only) and classified every one of them correctly.
Set BLIND_SNAPSHOT_TEST_DSN to re-run the part of that which needs a live
server; the rest is pinned below without one.
"""
import os

import pytest

import main
from feature_spec import PRE_FEATURES


class _Conn:
    """
    Answers each COUNT by what the query asks for rather than by call order,
    so the assertions below stay about classification and not about the order
    the function happens to issue its queries in.
    """

    def __init__(self, ip, pre):
        self.ip, self.pre = ip, pre
        self.calls = []
        self._n = 0

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=()):
        self.calls.append((sql, params))
        counts = self.pre if "prematch_snapshots" in sql else self.ip
        training = "match_results" in sql
        blind = "= 0" in sql
        key = ("blind" if blind else "total") + ("_training" if training else "")
        self._n = counts[key]
        return self

    def fetchone(self):
        return (self._n,)


def _counts(total, blind, total_training, blind_training):
    return {"total": total, "blind": blind,
            "total_training": total_training, "blind_training": blind_training}


def _run(monkeypatch, ip, pre):
    conn = _Conn(ip, pre)
    monkeypatch.setattr(main, "db_conn", lambda: conn)
    return main.count_blind_snapshots(), conn


# ───────────────── what counts as an observation ─────────────────

def test_legacy_rows_are_read_from_stat_as_well_as_raw():
    """
    Schema-1 rows keep the statistics under `stat`; schema-2 moved them to
    `raw`. load_inplay_data()'s compatibility shim reads both, and there are
    still hundreds of schema-1 rows in the table.

    Reading only `raw` would report every one of them as blind — an endpoint
    built to decide whether to purge the training set would then recommend
    purging rows that are perfectly good data.
    """
    sql = main._inplay_observed_sum_sql()
    for field in main._OBSERVED_INPLAY_FIELDS:
        assert f"'raw'->>'{field}'" in sql, field
        assert f"'stat'->>'{field}'" in sql, field


def test_possession_is_not_evidence_that_anything_was_observed():
    """
    _possession() substitutes an even 50/50 split when the feed is absent, so
    pos_h/pos_a are non-zero on a row where nothing whatsoever arrived.
    Including them would classify every blind row as observed and report zero
    damage on a table full of it. stats_coverage_ok() excludes possession for
    exactly this reason; so does this.
    """
    assert "pos_h" not in main._OBSERVED_INPLAY_FIELDS
    assert "pos_a" not in main._OBSERVED_INPLAY_FIELDS
    assert "pos" not in main._inplay_observed_sum_sql()


def test_prematch_blindness_is_judged_per_side():
    """
    Mirrors prematch_data_gate(): one side's form fetch can fail while the
    other succeeds, and the resulting half-zero vector is just as unusable.
    The predicate must be OR across the two sides, not AND.
    """
    home = main._prematch_observed_sum_sql(main._OBSERVED_PREMATCH_HOME)
    away = main._prematch_observed_sum_sql(main._OBSERVED_PREMATCH_AWAY)
    assert home != away
    for field in main._OBSERVED_PREMATCH_HOME + main._OBSERVED_PREMATCH_AWAY:
        assert field in PRE_FEATURES, f"{field} is not a real prematch feature"


def test_the_prematch_predicate_flags_a_fixture_blind_on_one_side_only(monkeypatch):
    _, conn = _run(monkeypatch, _counts(0, 0, 0, 0), _counts(0, 0, 0, 0))
    blind_sql = [s for s, _ in conn.calls if "prematch_snapshots" in s and "= 0" in s]
    assert blind_sql, "no blind-row query was issued against prematch_snapshots"
    for sql in blind_sql:
        assert " OR " in sql, (
            "home and away are ANDed — a fixture whose home form failed but "
            "whose away form arrived would be counted as observed")


# ───────────────── which rows are actually in the fit ─────────────────

def test_in_play_damage_is_confined_to_the_window_the_loader_reads(monkeypatch):
    """
    A blind row at minute 5 is real, but load_inplay_data() never reads it, so
    it is not in any fit and counting it as damage overstates the case for a
    purge. The bound has to come from the same constants the loader uses, not
    from numbers typed in twice.
    """
    _, conn = _run(monkeypatch, _counts(0, 0, 0, 0), _counts(0, 0, 0, 0))
    windowed = [p for s, p in conn.calls
                if "tip_snapshots" in s and "match_results" in s]
    assert windowed, "no training-set query was issued against tip_snapshots"
    for params in windowed:
        assert params == (main.TRAIN_MIN_MINUTE, main.LIVE_TIP_MAX_MINUTE)


def test_a_row_whose_fixture_has_no_result_yet_is_not_counted_as_damage(monkeypatch):
    """
    Both loaders join to match_results; an ungraded fixture is in neither fit.
    It still shows up in rows_blind so the table total stays honest, but the
    number the verdict is drawn from must exclude it.
    """
    out, _ = _run(monkeypatch,
                  ip=_counts(total=1000, blind=100, total_training=500, blind_training=10),
                  pre=_counts(total=0, blind=0, total_training=0, blind_training=0))
    assert out["in_play"]["rows_blind"] == 100
    assert out["in_play"]["blind_pct_of_table"] == 10.0
    assert out["in_play"]["blind_in_training_set"] == 10
    assert out["in_play"]["blind_pct_of_training_set"] == 2.0


def test_percentages_never_divide_by_zero_on_an_empty_table(monkeypatch):
    out, _ = _run(monkeypatch, _counts(0, 0, 0, 0), _counts(0, 0, 0, 0))
    assert out["in_play"]["blind_pct_of_training_set"] == 0.0
    assert out["prematch"]["blind_pct_of_training_set"] == 0.0


# ───────────────── the verdict ─────────────────

def test_a_clean_training_set_says_there_is_nothing_to_purge(monkeypatch):
    out, _ = _run(monkeypatch,
                  ip=_counts(900, 0, 900, 0), pre=_counts(900, 0, 900, 0))
    assert "Nothing to purge" in out["verdict"]


def test_the_verdict_follows_the_worse_of_the_two_training_sets(monkeypatch):
    """
    In-play and prematch are fitted separately. A clean in-play set does not
    make a ruined prematch set safe to train on, so the verdict has to track
    whichever is worse rather than an average of the two.
    """
    out, _ = _run(monkeypatch,
                  ip=_counts(1000, 0, 1000, 0),
                  pre=_counts(1000, 600, 1000, 600))
    assert out["in_play"]["blind_pct_of_training_set"] == 0.0
    assert out["prematch"]["blind_pct_of_training_set"] == 60.0
    assert "60.0%" in out["verdict"]
    assert "Purge and retrain" in out["verdict"]


def test_a_dominated_training_set_says_no_conclusion_from_it_stands(monkeypatch):
    """
    This is the case the endpoint exists for. Every prematch head has been
    reporting single_class_prediction/no_skill for weeks and the standing
    read of that was "prematch has no skill". If most of what it was fitted
    on is identical all-zero vectors carrying whatever label the fixture
    happened to produce, that read was never tested — and the verdict has to
    say so, because the number alone will be read as confirmation.
    """
    out, _ = _run(monkeypatch,
                  ip=_counts(1000, 0, 1000, 0),
                  pre=_counts(1000, 400, 1000, 400))
    assert "before drawing any conclusion" in out["verdict"]


def test_a_handful_of_blind_rows_is_not_reported_as_a_crisis(monkeypatch):
    out, _ = _run(monkeypatch,
                  ip=_counts(1000, 5, 1000, 5), pre=_counts(1000, 0, 1000, 0))
    assert "Small enough to leave alone" in out["verdict"]
    assert "Purge" not in out["verdict"]


# ───────────────── the endpoint ─────────────────

def test_the_endpoint_is_admin_gated():
    assert main.app.test_client().get("/admin/diagnostics/blind-snapshots").status_code == 401


def test_the_endpoint_reports_a_failure_instead_of_an_empty_result(monkeypatch):
    """
    A diagnostic that swallows its own error and returns zeros would be read
    as "no damage found", which is the most expensive wrong answer it could
    give.
    """
    def _boom():
        raise RuntimeError("relation \"tip_snapshots\" does not exist")

    monkeypatch.setattr(main, "_require_admin", lambda: None)
    monkeypatch.setattr(main, "count_blind_snapshots", _boom)

    r = main.app.test_client().get("/admin/diagnostics/blind-snapshots")

    assert r.status_code == 500
    assert r.get_json()["ok"] is False


# ───────────────── against a real server ─────────────────

@pytest.mark.skipif(not os.getenv("BLIND_SNAPSHOT_TEST_DSN"),
                    reason="set BLIND_SNAPSHOT_TEST_DSN to run against a live PostgreSQL")
def test_the_queries_run_on_a_real_postgres(monkeypatch):
    """
    The fakes above cannot catch a SQL syntax error, a bad cast, or a json
    operator applied to the wrong type — all of which would only surface the
    first time the endpoint is hit in production, which is precisely when
    someone is trying to answer an urgent question with it. Read-only.
    """
    import psycopg2

    dsn = os.environ["BLIND_SNAPSHOT_TEST_DSN"]

    class _RealConn:
        def __enter__(self):
            self.conn = psycopg2.connect(dsn)
            self.cur = self.conn.cursor()
            return self

        def __exit__(self, *a):
            self.cur.close()
            self.conn.close()
            return False

        def execute(self, sql, params=()):
            self.cur.execute(sql, params)
            return self

        def fetchone(self):
            return self.cur.fetchone()

    monkeypatch.setattr(main, "db_conn", _RealConn)

    out = main.count_blind_snapshots()

    for block in (out["in_play"], out["prematch"]):
        assert block["rows_blind"] <= block["rows_total"]
        assert block["rows_in_training_set"] <= block["rows_total"]
        assert block["blind_in_training_set"] <= block["rows_blind"]
        assert block["blind_in_training_set"] <= block["rows_in_training_set"]
