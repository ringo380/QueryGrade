"""
Query analysis tasks.

This module contains Celery tasks for batch query analysis operations.
"""
import logging
from typing import Dict, List, Any
from celery import shared_task
from django.core.cache import caches
from django.contrib.auth.models import User
from django.utils import timezone

logger = logging.getLogger(__name__)


@shared_task(bind=True, name='analyzer.tasks.batch_analyze_queries')
def batch_analyze_queries(self, query_data_list: List[Dict], user_id: int) -> Dict[str, Any]:
    """
    Analyze multiple queries in batch for improved performance.

    Args:
        query_data_list: List of query data dictionaries
        user_id: ID of the user requesting analysis

    Returns:
        Dict containing batch analysis results
    """
    try:
        from ..query_analyzer import analyze_query

        user = User.objects.get(id=user_id)
        results = []

        logger.info(f"Starting batch analysis of {len(query_data_list)} queries for user {user.username}")

        for i, query_data in enumerate(query_data_list):
            try:
                # Create query analysis
                query, analysis = analyze_query(
                    sql_query=query_data['sql_query'],
                    user=user,
                    database_version=query_data.get('database_version', ''),
                    use_case_notes=query_data.get('use_case_notes', ''),
                    expected_performance=query_data.get('expected_performance', 'medium')
                )

                results.append({
                    'index': i,
                    'query_id': query.id,
                    'analysis_id': analysis.id,
                    'grade': analysis.grade,
                    'status': 'success'
                })

            except Exception as e:
                logger.error(f"Error analyzing query {i}: {str(e)}")
                results.append({
                    'index': i,
                    'status': 'error',
                    'error': str(e)
                })

        # Cache batch results
        cache = caches['process_cache']
        cache_key = f"batch_results_{user_id}_{self.request.id}"
        cache.set(cache_key, results, timeout=3600)

        success_count = len([r for r in results if r['status'] == 'success'])

        logger.info(f"Completed batch analysis: {success_count}/{len(query_data_list)} successful")

        return {
            'status': 'success',
            'cache_key': cache_key,
            'total_queries': len(query_data_list),
            'successful_analyses': success_count,
            'failed_analyses': len(query_data_list) - success_count,
            'results': results,
            'timestamp': timezone.now().isoformat()
        }

    except Exception as exc:
        logger.error(f"Error in batch analysis: {str(exc)}")
        return {
            'status': 'error',
            'error': str(exc),
            'timestamp': timezone.now().isoformat()
        }
