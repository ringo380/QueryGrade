"""
Performance optimization utilities for QueryGrade.
Includes caching, query optimization, and monitoring tools.
"""

import time
import hashlib
import pickle
import logging
from functools import wraps
from typing import Any, Callable, Optional, Dict, List
from django.core.cache import caches
from django.conf import settings
from django.db import connection
from django.utils.decorators import method_decorator
from django.views.decorators.cache import cache_page
from django.views.decorators.vary import vary_on_headers


logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    Performance monitoring utility for tracking query performance
    and identifying bottlenecks.
    """

    @staticmethod
    def time_function(func_name: str = None):
        """
        Decorator to time function execution and log slow operations.
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                function_name = func_name or f"{func.__module__}.{func.__name__}"

                try:
                    result = func(*args, **kwargs)
                    execution_time = time.time() - start_time

                    # Log slow operations
                    if execution_time > getattr(settings, 'SLOW_QUERY_THRESHOLD', 1.0):
                        logger.warning(
                            f"Slow operation detected: {function_name} "
                            f"took {execution_time:.2f} seconds"
                        )
                    elif getattr(settings, 'PERFORMANCE_MONITORING_ENABLED', False):
                        logger.info(f"{function_name} completed in {execution_time:.3f}s")

                    return result

                except Exception as e:
                    execution_time = time.time() - start_time
                    logger.error(
                        f"Function {function_name} failed after {execution_time:.2f}s: {e}"
                    )
                    raise

            return wrapper
        return decorator

    @staticmethod
    def monitor_database_queries(func):
        """
        Decorator to monitor database query count and execution time.
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not getattr(settings, 'PERFORMANCE_MONITORING_ENABLED', False):
                return func(*args, **kwargs)

            # Reset query count
            initial_queries = len(connection.queries)
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                query_count = len(connection.queries) - initial_queries

                if query_count > 10:  # Flag functions with many queries
                    logger.warning(
                        f"High query count in {func.__name__}: "
                        f"{query_count} queries in {execution_time:.3f}s"
                    )

                return result

            except Exception as e:
                execution_time = time.time() - start_time
                query_count = len(connection.queries) - initial_queries
                logger.error(
                    f"Function {func.__name__} failed with {query_count} queries "
                    f"in {execution_time:.3f}s: {e}"
                )
                raise

        return wrapper


class QueryCache:
    """
    Advanced caching utility for query analysis results.
    """

    def __init__(self, cache_name: str = 'query_analysis_cache'):
        self.cache = caches[cache_name]
        self.default_timeout = getattr(settings, 'QUERY_ANALYSIS_CACHE_TIMEOUT', 7200)

    def generate_cache_key(self, sql_text: str, database_type: str = '',
                          version: str = '1.0') -> str:
        """
        Generate a consistent cache key for SQL query analysis.
        """
        # Normalize the SQL for consistent caching
        normalized_sql = ' '.join(sql_text.strip().split())
        key_data = f"{normalized_sql}|{database_type}|{version}"
        return hashlib.md5(key_data.encode('utf-8')).hexdigest()

    def get_analysis(self, sql_text: str, database_type: str = '') -> Optional[Any]:
        """
        Retrieve cached analysis result.
        """
        cache_key = self.generate_cache_key(sql_text, database_type)
        try:
            return self.cache.get(cache_key)
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
            return None

    def set_analysis(self, sql_text: str, result: Any, database_type: str = '',
                    timeout: Optional[int] = None) -> bool:
        """
        Store analysis result in cache.
        """
        cache_key = self.generate_cache_key(sql_text, database_type)
        timeout = timeout or self.default_timeout

        try:
            self.cache.set(cache_key, result, timeout)
            return True
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")
            return False

    def invalidate_analysis(self, sql_text: str, database_type: str = '') -> bool:
        """
        Invalidate cached analysis result.
        """
        cache_key = self.generate_cache_key(sql_text, database_type)
        try:
            self.cache.delete(cache_key)
            return True
        except Exception as e:
            logger.warning(f"Cache invalidation failed: {e}")
            return False

    def clear_all(self) -> bool:
        """
        Clear all cached analysis results.
        """
        try:
            self.cache.clear()
            return True
        except Exception as e:
            logger.error(f"Cache clear failed: {e}")
            return False


class DatabaseOptimizer:
    """
    Database optimization utilities.
    """

    @staticmethod
    def optimize_queryset(queryset, select_related: List[str] = None,
                         prefetch_related: List[str] = None):
        """
        Optimize Django queryset with select_related and prefetch_related.
        """
        if select_related:
            queryset = queryset.select_related(*select_related)

        if prefetch_related:
            queryset = queryset.prefetch_related(*prefetch_related)

        return queryset

    @staticmethod
    def bulk_create_optimized(model_class, objects: List[Any],
                            batch_size: int = 1000) -> int:
        """
        Optimized bulk create with batching.
        """
        total_created = 0
        for i in range(0, len(objects), batch_size):
            batch = objects[i:i + batch_size]
            created_objects = model_class.objects.bulk_create(
                batch,
                ignore_conflicts=True,
                batch_size=batch_size
            )
            total_created += len(created_objects)

        return total_created


def cached_analysis(timeout: int = 3600):
    """
    Decorator for caching analysis function results.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key from function arguments
            cache_key = f"analysis_{func.__name__}_{hashlib.md5(str(args + tuple(kwargs.items())).encode()).hexdigest()}"

            cache = caches['query_analysis_cache']
            result = cache.get(cache_key)

            if result is None:
                result = func(*args, **kwargs)
                cache.set(cache_key, result, timeout)

            return result
        return wrapper
    return decorator


def optimize_view_performance(cache_timeout: int = 300):
    """
    Decorator to optimize view performance with caching and headers.
    """
    def decorator(view_func):
        @cache_page(cache_timeout, cache='template_cache')
        @vary_on_headers('User-Agent', 'Accept-Language')
        @PerformanceMonitor.time_function()
        @PerformanceMonitor.monitor_database_queries
        def wrapper(*args, **kwargs):
            return view_func(*args, **kwargs)
        return wrapper
    return decorator


class MemoryOptimizer:
    """
    Memory optimization utilities.
    """

    @staticmethod
    def paginate_large_queryset(queryset, batch_size: int = 1000):
        """
        Generator function to paginate large querysets for memory efficiency.
        """
        count = queryset.count()
        for offset in range(0, count, batch_size):
            yield queryset[offset:offset + batch_size]

    @staticmethod
    def chunked_processing(iterable, chunk_size: int = 1000):
        """
        Process large iterables in chunks to save memory.
        """
        chunk = []
        for item in iterable:
            chunk.append(item)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


class AsyncOptimizer:
    """
    Optimization utilities for async task processing.
    """

    @staticmethod
    def should_use_async(data_size: int, complexity_score: int = 1) -> bool:
        """
        Determine if a task should be processed asynchronously based on size and complexity.
        """
        # Threshold calculations
        size_threshold = 10 * 1024 * 1024  # 10MB
        complexity_threshold = 5

        return (data_size > size_threshold or
                complexity_score > complexity_threshold)

    @staticmethod
    def estimate_processing_time(data_size: int, query_count: int = 1) -> float:
        """
        Estimate processing time based on data size and query complexity.
        """
        # Base processing time per MB and per query
        base_time_per_mb = 0.1  # seconds
        base_time_per_query = 0.05  # seconds

        size_mb = data_size / (1024 * 1024)
        estimated_time = (size_mb * base_time_per_mb) + (query_count * base_time_per_query)

        return max(estimated_time, 1.0)  # Minimum 1 second


# Global instances for easy import
#
# ⚠️ TESTING NOTE: These singletons are instantiated at module import time,
# BEFORE Django's @override_settings decorator can apply test settings.
# This means query_cache will capture the production cache backend even in tests.
#
# Solution: In test setUp(), reinitialize the cache backend:
#   from analyzer.performance import query_cache
#   from django.core.cache import caches
#   query_cache.cache = caches['query_analysis_cache']
#
# See TESTING.md and analyzer/test_integration_refactored.py for details.
query_cache = QueryCache()
performance_monitor = PerformanceMonitor()
db_optimizer = DatabaseOptimizer()
memory_optimizer = MemoryOptimizer()
async_optimizer = AsyncOptimizer()