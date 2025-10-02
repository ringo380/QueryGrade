"""
Database schema analysis tasks.

This module contains Celery tasks for asynchronous database schema
introspection and analysis operations.
"""
import logging
from typing import Dict, Any
from celery import shared_task
from django.core.cache import caches
from django.contrib.auth.models import User
from django.utils import timezone

from ..database_introspector import DatabaseIntrospector

logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=2, name='analyzer.tasks.analyze_database_schema_async')
def analyze_database_schema_async(self, db_config: Dict, user_id: int) -> Dict[str, Any]:
    """
    Perform database schema analysis asynchronously for large databases.

    Args:
        db_config: Database connection configuration
        user_id: ID of the user requesting analysis

    Returns:
        Dict containing schema analysis results
    """
    try:
        user = User.objects.get(id=user_id)

        logger.info(f"Starting async database schema analysis for user {user.username}")

        # Create database introspector
        introspector = DatabaseIntrospector(db_config)

        if not introspector.connect():
            raise Exception("Failed to connect to database")

        # Get all tables
        tables = introspector.get_tables()

        # Perform comprehensive schema analysis
        from ..views import analyze_database_schema
        schema_analysis = analyze_database_schema(tables, db_config)

        # Add table details to analysis
        table_details = []
        for table in tables:
            table_details.append({
                'name': table.name,
                'row_count': table.row_count,
                'size_mb': table.size_mb,
                'column_count': len(table.columns),
                'index_count': len(table.indexes),
                'foreign_key_count': len(table.foreign_keys),
                'columns': table.columns,
                'indexes': table.indexes,
                'foreign_keys': table.foreign_keys
            })

        results = {
            'schema_analysis': schema_analysis,
            'table_details': table_details,
            'database_info': {
                'engine': db_config['engine'],
                'total_tables': len(tables),
                'total_size_mb': sum(table.size_mb for table in tables),
                'total_rows': sum(table.row_count for table in tables)
            }
        }

        # Cache results
        cache = caches['process_cache']
        cache_key = f"schema_analysis_{user_id}_{self.request.id}"
        cache.set(cache_key, results, timeout=7200)  # 2 hours for schema analysis

        logger.info(f"Completed database schema analysis for user {user.username}")

        return {
            'status': 'success',
            'cache_key': cache_key,
            'total_tables': len(tables),
            'analysis_summary': {
                'recommendations_count': len(schema_analysis['recommendations']),
                'issues_count': len(schema_analysis['issues']),
                'performance_notes_count': len(schema_analysis['performance_notes'])
            },
            'timestamp': timezone.now().isoformat()
        }

    except Exception as exc:
        logger.error(f"Error in database schema analysis: {str(exc)}")

        # Retry logic
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying schema analysis in 120 seconds (attempt {self.request.retries + 1})")
            raise self.retry(countdown=120, exc=exc)

        return {
            'status': 'error',
            'error': str(exc),
            'timestamp': timezone.now().isoformat()
        }
