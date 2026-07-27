"""Tests for SessionPurgeMiddleware (issue #133).

The middleware is the only thing purging expired sessions while the worker and
beat services are stopped, so these cover the guarantees that matter: it runs,
it runs at most once per interval, it survives a broken cache or database, and
it never delays the response path more than once per interval.
"""

from __future__ import annotations

from datetime import timedelta
from unittest import mock

from django.contrib.sessions.models import Session
from django.core.cache import cache
from django.http import HttpResponse
from django.test import TestCase, override_settings
from django.utils import timezone

from analyzer.middleware import SessionPurgeMiddleware

PURGE_PATH = "analyzer.tasks.maintenance_tasks.purge_expired_sessions_now"


def _make_session(expired: bool) -> Session:
    offset = timedelta(hours=-1) if expired else timedelta(hours=1)
    return Session.objects.create(
        session_key=f"{'x' if expired else 'y'}{Session.objects.count():031d}",
        session_data="e30=",
        expire_date=timezone.now() + offset,
    )


@override_settings(
    SESSION_PURGE_IN_REQUEST=True,
    SESSION_PURGE_INTERVAL_SECONDS=86400,
    CACHES={"default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"}},
)
class SessionPurgeMiddlewareTests(TestCase):
    def setUp(self):
        cache.clear()
        Session.objects.all().delete()
        self.response = HttpResponse("ok")
        self.calls = []

    def _middleware(self):
        def get_response(request):
            self.calls.append(request)
            return self.response

        return SessionPurgeMiddleware(get_response)

    def test_expired_sessions_are_deleted_on_a_request(self):
        _make_session(expired=True)
        _make_session(expired=True)
        live = _make_session(expired=False)

        mw = self._middleware()
        response = mw(mock.Mock())

        self.assertIs(response, self.response)
        self.assertEqual(
            list(Session.objects.values_list("session_key", flat=True)),
            [live.session_key],
        )

    def test_purge_runs_at_most_once_per_interval(self):
        mw = self._middleware()
        with mock.patch(PURGE_PATH, return_value={"status": "success"}) as purge:
            for _ in range(5):
                mw(mock.Mock())
        self.assertEqual(purge.call_count, 1)

    def test_a_second_process_does_not_repeat_the_purge_in_the_same_interval(self):
        """The in-process deadline is per instance, so the cache lock is what
        stops a second worker (or a restarted one) purging again."""
        with mock.patch(PURGE_PATH, return_value={"status": "success"}) as purge:
            self._middleware()(mock.Mock())
            self._middleware()(mock.Mock())
        self.assertEqual(purge.call_count, 1)

    def test_purge_runs_again_once_the_interval_has_passed(self):
        mw = self._middleware()
        with mock.patch(PURGE_PATH, return_value={"status": "success"}) as purge:
            mw(mock.Mock())
            # Expire both gates: the in-process deadline and the cache lock.
            mw._next_attempt = 0.0
            cache.delete("session_purge_lock")
            mw(mock.Mock())
        self.assertEqual(purge.call_count, 2)

    def test_response_is_returned_when_the_purge_raises(self):
        _make_session(expired=True)
        mw = self._middleware()
        with mock.patch(PURGE_PATH, side_effect=RuntimeError("db is down")):
            response = mw(mock.Mock())
        self.assertIs(response, self.response)
        self.assertEqual(len(self.calls), 1)

    def test_response_is_returned_when_the_cache_is_unreachable(self):
        mw = self._middleware()
        with mock.patch(
            "django.core.cache.cache.add", side_effect=RuntimeError("redis is down")
        ):
            response = mw(mock.Mock())
        self.assertIs(response, self.response)

    def test_a_failed_attempt_does_not_retry_on_the_next_request(self):
        """The deadline is pushed forward before the attempt, so a failing
        purge costs one try per interval and not one per request."""
        mw = self._middleware()
        with mock.patch(PURGE_PATH, side_effect=RuntimeError("boom")) as purge:
            for _ in range(4):
                mw(mock.Mock())
        self.assertEqual(purge.call_count, 1)

    @override_settings(SESSION_PURGE_IN_REQUEST=False)
    def test_disabled_by_setting(self):
        _make_session(expired=True)
        mw = self._middleware()
        with mock.patch(PURGE_PATH) as purge:
            mw(mock.Mock())
        purge.assert_not_called()
        self.assertEqual(Session.objects.count(), 1)

    @override_settings(SESSION_PURGE_INTERVAL_SECONDS=0)
    def test_zero_interval_disables_rather_than_running_every_request(self):
        mw = self._middleware()
        with mock.patch(PURGE_PATH) as purge:
            mw(mock.Mock())
            mw(mock.Mock())
        purge.assert_not_called()

    def test_a_quiet_interval_touches_the_cache_only_once(self):
        """The in-process deadline exists to keep the common path free of I/O.

        The cache lock alone would already stop a second purge, so nothing
        else here can tell whether the deadline check is present. What it buys
        is that a request which is not going to purge does no cache round-trip
        at all - on a Redis-backed cache that is a network hop per request.
        """
        mw = self._middleware()
        with mock.patch(PURGE_PATH, return_value={"status": "success"}), mock.patch(
            "django.core.cache.cache.add", wraps=cache.add
        ) as add:
            for _ in range(10):
                mw(mock.Mock())
        self.assertEqual(add.call_count, 1)

    def test_a_failing_purge_does_not_touch_the_cache_on_every_request(self):
        """Same rule on the failure path: the deadline is pushed forward
        before the attempt, so a purge that keeps raising still costs one
        cache round-trip per interval rather than one per request."""
        mw = self._middleware()
        with mock.patch(PURGE_PATH, side_effect=RuntimeError("boom")), mock.patch(
            "django.core.cache.cache.add", wraps=cache.add
        ) as add:
            for _ in range(10):
                mw(mock.Mock())
        self.assertEqual(add.call_count, 1)

    def test_first_request_after_a_restart_is_eligible(self):
        """A fresh instance must not start its interval in the future, or a
        service restarting more often than the interval would never purge."""
        self.assertEqual(self._middleware()._next_attempt, 0.0)


@override_settings(
    SESSION_PURGE_IN_REQUEST=True,
    SESSION_PURGE_INTERVAL_SECONDS=86400,
    RATELIMIT_ENABLE=False,
    CACHES={"default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"}},
)
class SessionPurgeThroughTheRequestStackTests(TestCase):
    """Exercise the purge through a real request.

    Everything above builds the middleware by hand, so all of it passes even
    if SessionPurgeMiddleware is never added to MIDDLEWARE - which is the one
    mistake that would silently disable retention entirely. This covers the
    wiring rather than the class.
    """

    def setUp(self):
        cache.clear()
        Session.objects.all().delete()

    def test_middleware_is_registered(self):
        from django.conf import settings

        self.assertIn("analyzer.middleware.SessionPurgeMiddleware", settings.MIDDLEWARE)

    def test_an_ordinary_request_purges_expired_sessions(self):
        expired = _make_session(expired=True)
        live = _make_session(expired=False)

        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        keys = set(Session.objects.values_list("session_key", flat=True))
        # The request may create a session row of its own, so assert on the
        # two rows under test rather than the whole table.
        self.assertNotIn(expired.session_key, keys)
        self.assertIn(live.session_key, keys)


class PurgeExpiredSessionsFunctionTests(TestCase):
    """The Celery task and the middleware must share one implementation."""

    def setUp(self):
        Session.objects.all().delete()

    def test_task_delegates_to_the_shared_function(self):
        from analyzer.tasks.maintenance_tasks import purge_expired_sessions

        with mock.patch(PURGE_PATH, return_value={"status": "success"}) as purge:
            result = purge_expired_sessions()
        purge.assert_called_once_with()
        self.assertEqual(result, {"status": "success"})

    def test_function_reports_what_it_removed(self):
        from analyzer.tasks.maintenance_tasks import purge_expired_sessions_now

        _make_session(expired=True)
        _make_session(expired=False)

        result = purge_expired_sessions_now()

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["purged"], 1)
        self.assertEqual(result["remaining"], 1)
