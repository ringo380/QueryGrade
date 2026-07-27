"""
Tests for the retention/maintenance tasks.

These guard the two properties that actually matter for storage growth:
expired session rows are deleted, and live session rows are not.
"""

from datetime import timedelta

from django.contrib.sessions.backends.db import SessionStore
from django.contrib.sessions.models import Session
from django.test import TestCase
from django.utils import timezone

from analyzer.tasks import purge_expired_sessions


def _make_session(expire_offset):
    """Create a real session row whose expire_date is now + expire_offset."""
    store = SessionStore()
    store["k"] = "v"
    store.create()
    Session.objects.filter(session_key=store.session_key).update(
        expire_date=timezone.now() + expire_offset
    )
    return store.session_key


class PurgeExpiredSessionsTests(TestCase):
    def test_deletes_expired_and_keeps_live_sessions(self):
        expired = [_make_session(timedelta(hours=-2)) for _ in range(3)]
        live = [_make_session(timedelta(hours=2)) for _ in range(2)]
        self.assertEqual(Session.objects.count(), 5)

        result = purge_expired_sessions()

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["purged"], 3)
        self.assertEqual(result["remaining"], 2)

        # Assert the delta, not just the count: the right rows survived.
        surviving = set(Session.objects.values_list("session_key", flat=True))
        self.assertEqual(surviving, set(live))
        for key in expired:
            self.assertFalse(Session.objects.filter(session_key=key).exists())

    def test_noop_when_nothing_expired(self):
        live = [_make_session(timedelta(hours=2)) for _ in range(2)]

        result = purge_expired_sessions()

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["purged"], 0)
        self.assertEqual(Session.objects.count(), 2)
        self.assertEqual(
            set(Session.objects.values_list("session_key", flat=True)), set(live)
        )


class TaskRoutingTests(TestCase):
    """
    A task routed to a queue no worker consumes is silently never executed -
    Celery accepts it, writes it to Redis, and it sits there. This asserts the
    worker's -Q list covers every queue settings.CELERY_TASK_ROUTES names.
    """

    def test_worker_consumes_every_routed_queue(self):
        import tomllib

        from django.conf import settings

        with open("railway.worker.toml", "rb") as fh:
            start_command = tomllib.load(fh)["deploy"]["startCommand"]

        self.assertIn("-Q ", start_command, "worker must pin its queue list")
        declared = set(start_command.split("-Q ")[1].split()[0].split(","))

        needed = {r["queue"] for r in settings.CELERY_TASK_ROUTES.values()}
        needed.add("celery")  # default queue: unrouted tasks land here

        self.assertEqual(
            needed - declared,
            set(),
            "CELERY_TASK_ROUTES names a queue the worker does not consume",
        )
