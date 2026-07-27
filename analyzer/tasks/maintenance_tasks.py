"""
System maintenance tasks.

This module contains Celery tasks for periodic system maintenance,
cleanup operations, and housekeeping functions.
"""

import logging
import os
import tempfile
from typing import Any, Dict

from celery import shared_task
from django.contrib.sessions.models import Session
from django.core.management import call_command
from django.utils import timezone

logger = logging.getLogger(__name__)


def purge_expired_sessions_now():
    """
    Delete django_session rows whose expire_date has passed.

    Django never removes expired session rows on its own - the cookie stops
    being honored at SESSION_COOKIE_AGE (1 hour here), but the row stays
    forever. Every anonymous visitor that touches a view writing to the
    session leaves one behind, so the table is the only unbounded-growth
    surface in this database (4,905 rows / 1.8 MB of a 12 MB database when
    this was added, against zero registered users). Observed accumulation is
    roughly 100-150 rows a day from crawler traffic alone.

    Delegates to the `clearsessions` management command so this stays
    correct if SESSION_ENGINE ever moves off the DB backend.

    This is a plain function, not only a Celery task, because it has two
    callers. The beat schedule runs it daily when the worker is up, and
    SessionPurgeMiddleware runs it from the web process when it is not
    (issue #133 - the worker and beat services are stopped for cost while
    QueryGrade has no users). Never assume it runs on a worker: it must stay
    safe to call inline in a request.
    """
    try:
        expired = Session.objects.filter(expire_date__lt=timezone.now()).count()
        call_command("clearsessions")
        remaining = Session.objects.count()

        logger.info(
            f"Session purge completed: removed {expired} expired sessions, "
            f"{remaining} remaining"
        )

        return {
            "status": "success",
            "purged": expired,
            "remaining": remaining,
            "timestamp": timezone.now().isoformat(),
        }

    except Exception as exc:
        logger.error(f"Error in session purge task: {str(exc)}")
        return {
            "status": "error",
            "error": str(exc),
            "timestamp": timezone.now().isoformat(),
        }


@shared_task(name="analyzer.tasks.purge_expired_sessions")
def purge_expired_sessions():
    """Celery entry point for :func:`purge_expired_sessions_now`."""
    return purge_expired_sessions_now()


@shared_task(name="analyzer.tasks.cleanup_temp_files")
def cleanup_temp_files():
    """
    Periodic task to clean up temporary files and expired cache entries.
    Should be run via celery beat every hour.
    """
    try:
        temp_dir = tempfile.gettempdir()
        cleaned_files = 0

        # Clean up old temporary files (older than 2 hours)
        import time

        current_time = time.time()

        for filename in os.listdir(temp_dir):
            if filename.startswith("tmp") and filename.endswith(".log"):
                file_path = os.path.join(temp_dir, filename)
                if os.path.isfile(file_path):
                    file_age = current_time - os.path.getmtime(file_path)
                    if file_age > 7200:  # 2 hours
                        try:
                            os.remove(file_path)
                            cleaned_files += 1
                        except OSError:
                            pass

        logger.info(f"Cleanup task completed: removed {cleaned_files} temporary files")

        return {
            "status": "success",
            "cleaned_files": cleaned_files,
            "timestamp": timezone.now().isoformat(),
        }

    except Exception as exc:
        logger.error(f"Error in cleanup task: {str(exc)}")
        return {
            "status": "error",
            "error": str(exc),
            "timestamp": timezone.now().isoformat(),
        }
