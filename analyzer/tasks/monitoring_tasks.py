"""
Celery tasks for periodic ML monitoring.

These are thin wrappers around the management-command logic so the same
code runs whether invoked by Beat, manually via `manage.py`, or imported
directly from a view (e.g., a future "run evaluation now" button).
"""

import logging
from typing import Any, Dict

from celery import shared_task
from django.utils import timezone

logger = logging.getLogger(__name__)


@shared_task(name="analyzer.tasks.monitor_ml_models")
def monitor_ml_models() -> Dict[str, Any]:
    """Periodic monitoring task. Wired into Celery Beat in querygrade/celery.py."""
    from analyzer.ml.monitoring.alert_evaluator import run_evaluation

    try:
        created, skipped = run_evaluation()
        logger.info("monitor_ml_models: created=%d skipped=%d", len(created), skipped)
        return {
            "status": "success",
            "created": len(created),
            "skipped": skipped,
            "created_ids": [a.id for a in created],
            "timestamp": timezone.now().isoformat(),
        }
    except Exception as exc:
        # Never let monitoring failures take down the beat worker — log and
        # return a structured error. Persistent failures will themselves
        # eventually surface as ops-visible behavior (e.g., zero alerts over
        # many days), and the worker logs carry the trace.
        logger.exception("monitor_ml_models failed: %s", exc)
        return {
            "status": "error",
            "error": str(exc),
            "timestamp": timezone.now().isoformat(),
        }
