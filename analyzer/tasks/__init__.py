"""
QueryGrade Celery tasks package.

This package provides asynchronous task processing for QueryGrade using Celery.
Tasks are organized by domain:
- log_tasks: Log file processing
- query_tasks: Batch query analysis
- schema_tasks: Database schema introspection
- maintenance_tasks: System cleanup and housekeeping
- report_tasks: Report generation

All tasks are exported at the package level for backward compatibility
with existing Celery task names.

CRITICAL: Task names must remain stable for Celery to find them correctly.
Do not rename task functions as this will break existing queued tasks.
"""

# Log processing tasks
from .log_tasks import process_log_file_async

# Maintenance tasks
from .maintenance_tasks import cleanup_temp_files, purge_expired_sessions

# ML monitoring tasks
from .monitoring_tasks import monitor_ml_models

# Query analysis tasks
from .query_tasks import batch_analyze_queries

# Report generation tasks
from .report_tasks import generate_performance_report

# Schema analysis tasks
from .schema_tasks import analyze_database_schema_async

__all__ = [
    # Log tasks
    "process_log_file_async",
    # Query tasks
    "batch_analyze_queries",
    # Schema tasks
    "analyze_database_schema_async",
    # Maintenance tasks
    "cleanup_temp_files",
    "purge_expired_sessions",
    # Report tasks
    "generate_performance_report",
    # ML monitoring tasks
    "monitor_ml_models",
]
