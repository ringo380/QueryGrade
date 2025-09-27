import os
import tempfile
from typing import Dict, List, Any, Optional
from celery import shared_task
from django.core.cache import caches
from django.contrib.auth.models import User
from django.utils import timezone
import logging

from .parser import process_slow_log, process_general_log
from .models import Query, QueryAnalysis
from .database_introspector import DatabaseIntrospector

logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=3)
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


@shared_task(bind=True)
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
        from .query_analyzer import analyze_query

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


@shared_task(bind=True, max_retries=2)
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
        from .views import analyze_database_schema
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


@shared_task
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
            if filename.startswith('tmp') and filename.endswith('.log'):
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
            'status': 'success',
            'cleaned_files': cleaned_files,
            'timestamp': timezone.now().isoformat()
        }

    except Exception as exc:
        logger.error(f"Error in cleanup task: {str(exc)}")
        return {
            'status': 'error',
            'error': str(exc),
            'timestamp': timezone.now().isoformat()
        }


@shared_task(bind=True)
def generate_performance_report(self, user_id: int, date_range_days: int = 30) -> Dict[str, Any]:
    """
    Generate comprehensive performance report for user's query history.

    Args:
        user_id: ID of the user
        date_range_days: Number of days to include in report

    Returns:
        Dict containing performance report data
    """
    try:
        from datetime import timedelta
        from django.db.models import Avg, Count, Q

        user = User.objects.get(id=user_id)
        end_date = timezone.now()
        start_date = end_date - timedelta(days=date_range_days)

        logger.info(f"Generating performance report for user {user.username}")

        # Get query analyses in date range
        analyses = QueryAnalysis.objects.filter(
            query__user=user,
            created_at__gte=start_date,
            created_at__lte=end_date
        )

        # Calculate statistics
        total_queries = analyses.count()
        avg_grade = analyses.aggregate(avg_grade=Avg('grade'))['avg_grade'] or 0

        grade_distribution = {}
        for grade in ['A', 'B', 'C', 'D', 'F']:
            count = analyses.filter(grade=grade).count()
            grade_distribution[grade] = {
                'count': count,
                'percentage': (count / total_queries * 100) if total_queries > 0 else 0
            }

        # Performance trends
        improvement_queries = analyses.filter(
            feedback_provided=True
        ).count()

        # Most common issues
        issue_keywords = ['index', 'join', 'subquery', 'performance', 'optimization']
        common_issues = {}
        for keyword in issue_keywords:
            count = analyses.filter(
                Q(feedback__icontains=keyword) | Q(recommendations__icontains=keyword)
            ).count()
            if count > 0:
                common_issues[keyword] = count

        report_data = {
            'user_info': {
                'username': user.username,
                'report_period': f"{start_date.date()} to {end_date.date()}"
            },
            'summary': {
                'total_queries_analyzed': total_queries,
                'average_grade': round(avg_grade, 2),
                'improvement_queries': improvement_queries,
                'grade_distribution': grade_distribution
            },
            'insights': {
                'common_issues': common_issues,
                'recommendations': [
                    "Focus on query optimization techniques",
                    "Review indexing strategies",
                    "Consider query refactoring for better performance"
                ]
            }
        }

        # Cache report
        cache = caches['process_cache']
        cache_key = f"performance_report_{user_id}_{self.request.id}"
        cache.set(cache_key, report_data, timeout=86400)  # 24 hours

        logger.info(f"Generated performance report for user {user.username}")

        return {
            'status': 'success',
            'cache_key': cache_key,
            'report_summary': {
                'total_queries': total_queries,
                'average_grade': round(avg_grade, 2),
                'date_range': date_range_days
            },
            'timestamp': timezone.now().isoformat()
        }

    except Exception as exc:
        logger.error(f"Error generating performance report: {str(exc)}")
        return {
            'status': 'error',
            'error': str(exc),
            'timestamp': timezone.now().isoformat()
        }