"""
System maintenance tasks.

This module contains Celery tasks for periodic system maintenance,
cleanup operations, and housekeeping functions.
"""

import logging
import os
import tempfile

from celery import shared_task
from django.utils import timezone

logger = logging.getLogger(__name__)


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
