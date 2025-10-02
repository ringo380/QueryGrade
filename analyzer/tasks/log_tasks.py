"""
Log file processing tasks.

This module contains Celery tasks for asynchronously processing
uploaded SQL log files (slow query logs and general query logs).
"""
import os
import logging
from typing import Dict, Any
from celery import shared_task
from django.core.cache import caches
from django.contrib.auth.models import User
from django.utils import timezone

from ..parser import process_slow_log, process_general_log

logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=3, name='analyzer.tasks.process_log_file_async')
def process_log_file_async(self, file_path: str, log_type: str, user_id: int) -> Dict[str, Any]:
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

        logger.info(f"Starting async log processing for user {user.username}, file: {file_path}")

        # Process the log file based on type
        if log_type == 'slow':
            results = process_slow_log(file_path)
        elif log_type == 'general':
            results = process_general_log(file_path)
        else:
            raise ValueError(f"Invalid log type: {log_type}")

        # Cache the results using user-specific cache key
        cache = caches['process_cache']
        cache_key = f"log_results_{user_id}_{self.request.id}"
        cache.set(cache_key, results, timeout=3600)  # 1 hour

        # Clean up the temporary file
        if os.path.exists(file_path):
            os.remove(file_path)

        logger.info(f"Successfully processed log file for user {user.username}")

        return {
            'status': 'success',
            'cache_key': cache_key,
            'total_queries': len(results.get('anomalies', [])),
            'anomaly_count': len([r for r in results.get('anomalies', []) if r.get('is_anomaly', False)]),
            'processing_time': results.get('processing_time', 0),
            'timestamp': timezone.now().isoformat()
        }

    except Exception as exc:
        logger.error(f"Error processing log file: {str(exc)}")

        # Clean up file on error
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass

        # Retry logic
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying task in 60 seconds (attempt {self.request.retries + 1})")
            raise self.retry(countdown=60, exc=exc)

        return {
            'status': 'error',
            'error': str(exc),
            'timestamp': timezone.now().isoformat()
        }
