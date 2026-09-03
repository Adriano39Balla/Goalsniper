"""
/dashboard/login is throttled because, unthrottled, it is an oracle for the
admin key.

The bound on the attempts table used to be a .clear() of the whole table once
it passed 10,000 addresses. The key is taken from X-Forwarded-For, which the
client sets, so fabricating distinct values cost an attacker nothing and reset
their own counter along with everyone else's.
"""
import time

import pytest

import main


@pytest.fixture(autouse=True)
def _clean_table():
    main._login_attempts.clear()
    yield
    main._login_attempts.clear()


def _fail(ip, n):
    for _ in range(n):
        main._login_rate_limited(ip)
        main._login_record_failure(ip)


def test_an_address_is_throttled_after_the_limit():
    _fail("1.2.3.4", main.LOGIN_MAX_ATTEMPTS)
    assert main._login_rate_limited("1.2.3.4") is True


def test_an_address_below_the_limit_is_not_throttled():
    _fail("1.2.3.4", main.LOGIN_MAX_ATTEMPTS - 1)
    assert main._login_rate_limited("1.2.3.4") is False


def test_flooding_the_table_cannot_clear_a_live_counter(monkeypatch):
    # The attack: X-Forwarded-For is client-controlled, so distinct fabricated
    # values are free. Overflowing the table must not reset the attacker.
    monkeypatch.setattr(main, "LOGIN_ATTEMPTS_MAX_IPS", 100)
    _fail("1.2.3.4", main.LOGIN_MAX_ATTEMPTS)
    assert main._login_rate_limited("1.2.3.4") is True

    for i in range(500):
        main._login_rate_limited(f"10.0.{i // 256}.{i % 256}")

    assert main._login_rate_limited("1.2.3.4") is True, \
        "the throttle was flushed by traffic the attacker controls"


def test_merely_loading_the_form_does_not_occupy_the_table():
    # A read used to insert via defaultdict, so every visitor grew the table
    # and brought the old .clear() closer for free.
    for i in range(50):
        main._login_rate_limited(f"9.9.9.{i}")
    assert main._login_attempts == {}, "only recorded failures belong in the table"


def test_the_table_stays_bounded(monkeypatch):
    monkeypatch.setattr(main, "LOGIN_ATTEMPTS_MAX_IPS", 100)
    for i in range(1000):
        ip = f"172.16.{i // 256}.{i % 256}"
        main._login_rate_limited(ip)
        main._login_record_failure(ip)
    assert len(main._login_attempts) <= 101, "memory must still be bounded"


def test_attempts_outside_the_window_stop_counting(monkeypatch):
    ip = "5.6.7.8"
    old = time.time() - main.LOGIN_WINDOW_SEC - 1
    main._login_attempts[ip] = [old] * (main.LOGIN_MAX_ATTEMPTS + 5)
    assert main._login_rate_limited(ip) is False
    assert ip not in main._login_attempts, "a fully expired address is dropped"
