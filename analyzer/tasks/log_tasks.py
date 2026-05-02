"""
Log file processing tasks.

This module contains Celery tasks for asynchronously processing
uploaded SQL log files (slow query logs and general query logs).
"""

import logging
import os
from typing import Any, Dict

from celery import shared_task
from django.contrib.auth.models import User
from django.core.cache import caches
from django.utils import timezone

from ..analytics import send_ga4_event, synthetic_client_id
from ..parser import process_general_log, process_slow_log

logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=3, name="analyzer.tasks.process_log_file_async")
def process_log_file_async(
    self, file_path: str, log_type: str, user_id: int
) -> Dict[str, Any]:
    """
    Asynchronously process uploaded log files.

    Args:
        file_path: Path to the uploaded log file
        log_type: Type of log ('slow' or 'general')
        user_id: ID of the user who uploaded the file

    Returns:
        Dict containing processing results and metadata
    """
    try:
        user = User.objects.get(id=user_id)

        logger.info(
            f"Starting async log processing for user {user.username}, file: {file_path}"
        )

        # Process the log file based on type
        if log_type == "slow":
            results = process_slow_log(file_path)
        elif log_type == "general":
            results = process_general_log(file_path)
        else:
            raise ValueError(f"Invalid log type: {log_type}")

        # Cache the results using user-specific cache key
        cache = caches["process_cache"]
        cache_key = f"log_results_{user_id}_{self.request.id}"
        cache.set(cache_key, results, timeout=3600)  # 1 hour

        # Clean up the temporary file
        if os.path.exists(file_path):
            os.remove(file_path)

        logger.info(f"Successfully processed log file for user {user.username}")

        total_queries = len(results.get("anomalies", []))
        anomaly_count = len(
            [r for r in results.get("anomalies", []) if r.get("is_anomaly", False)]
        )
        processing_time = results.get("processing_time", 0)

        send_ga4_event(
            client_id=synthetic_client_id(user_id),
            event_name="log_analysis_completed",
            params={
                "log_type": log_type,
                "total_queries": total_queries,
                "anomaly_count": anomaly_count,
                "processing_time_ms": int(processing_time * 1000),
            },
            user_id=user_id,
        )

        return {
            "status": "success",
            "cache_key": cache_key,
            "total_queries": total_queries,
            "anomaly_count": anomaly_count,
            "processing_time": processing_time,
            "timestamp": timezone.now().isoformat(),
        }

    except Exception as exc:
        logger.error(f"Error processing log file: {str(exc)}")

        # Clean up file on error
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass

        # Retry logic
        if self.request.retries < self.max_retries:
            logger.info(
                f"Retrying task in 60 seconds (attempt {self.request.retries + 1})"
            )
            raise self.retry(countdown=60, exc=exc)

        send_ga4_event(
            client_id=synthetic_client_id(user_id),
            event_name="log_analysis_failed",
            params={"log_type": log_type, "error_class": type(exc).__name__},
            user_id=user_id,
        )

        return {
            "status": "error",
            "error": str(exc),
            "timestamp": timezone.now().isoformat(),
        }
