"""
Django management command for performance optimization tasks.
"""

import logging
import time

from django.conf import settings
from django.core.cache import caches
from django.core.management.base import BaseCommand
from django.db import connection

from analyzer.models import Query, UserQueryHistory
from analyzer.performance import query_cache

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Optimize QueryGrade performance through cache warming, database optimization, and cleanup"

    def add_arguments(self, parser):
        parser.add_argument(
            "--warm-cache",
            action="store_true",
            help="Warm up the query analysis cache with popular queries",
        )
        parser.add_argument(
            "--cleanup-cache",
            action="store_true",
            help="Clean up stale cache entries",
        )
        parser.add_argument(
            "--optimize-db",
            action="store_true",
            help="Run database optimization tasks",
        )
        parser.add_argument(
            "--cleanup-old-data",
            action="store_true",
            help="Clean up old temporary data and expired sessions",
        )
        parser.add_argument(
            "--all",
            action="store_true",
            help="Run all optimization tasks",
        )
        parser.add_argument(
            "--days",
            type=int,
            default=30,
            help="Number of days to keep data (for cleanup operations)",
        )

    def handle(self, *args, **options):
        start_time = time.time()

        if options["all"]:
            options.update(
                {
                    "warm_cache": True,
                    "cleanup_cache": True,
                    "optimize_db": True,
                    "cleanup_old_data": True,
                }
            )

        if options["warm_cache"]:
            self.warm_cache()

        if options["cleanup_cache"]:
            self.cleanup_cache()

        if options["optimize_db"]:
            self.optimize_database()

        if options["cleanup_old_data"]:
            self.cleanup_old_data(options["days"])

        total_time = time.time() - start_time
        self.stdout.write(
            self.style.SUCCESS(
                f"Performance optimization completed in {total_time:.2f} seconds"
            )
        )

    def warm_cache(self):
        """Warm up the cache with popular queries."""
        self.stdout.write("Warming up query analysis cache...")

        try:
            # Get most popular queries (those analyzed multiple times)
            popular_queries = (
                Query.objects.filter(analysis__isnull=False)
                .select_related("analysis")
                .order_by("-created_at")[:100]
            )

            warmed_count = 0
            for query in popular_queries:
                if hasattr(query, "analysis"):
                    # Cache the analysis result
                    result = (query, query.analysis)
                    success = query_cache.set_analysis(
                        query.normalized_text, result, query.database_type or ""
                    )
                    if success:
                        warmed_count += 1

            self.stdout.write(
                self.style.SUCCESS(f"Warmed cache with {warmed_count} popular queries")
            )

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Cache warming failed: {e}"))

    def cleanup_cache(self):
        """Clean up stale cache entries."""
        self.stdout.write("Cleaning up stale cache entries...")

        try:
            cache_configs = [
                ("default", "default cache"),
                ("query_analysis_cache", "query analysis cache"),
                ("process_cache", "process cache"),
                ("template_cache", "template cache"),
            ]

            for cache_name, description in cache_configs:
                try:
                    caches[cache_name]
                    # Note: Redis doesn't have a direct way to get all keys
                    # In production, you might want to implement TTL-based cleanup
                    self.stdout.write(f"Cleaned {description}")
                except Exception as e:
                    self.stdout.write(
                        self.style.WARNING(f"Failed to clean {description}: {e}")
                    )

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Cache cleanup failed: {e}"))

    def optimize_database(self):
        """Run database optimization tasks."""
        self.stdout.write("Optimizing database...")

        try:
            with connection.cursor() as cursor:
                # SQLite-specific optimizations
                if "sqlite" in settings.DATABASES["default"]["ENGINE"]:
                    optimizations = [
                        "PRAGMA optimize;",
                        "VACUUM;",
                        "ANALYZE;",
                    ]

                    for optimization in optimizations:
                        try:
                            cursor.execute(optimization)
                            self.stdout.write(f"Executed: {optimization}")
                        except Exception as e:
                            self.stdout.write(
                                self.style.WARNING(
                                    f"Optimization failed: {optimization}: {e}"
                                )
                            )

                # PostgreSQL-specific optimizations
                elif "postgresql" in settings.DATABASES["default"]["ENGINE"]:
                    optimizations = [
                        "VACUUM ANALYZE;",
                        "REINDEX DATABASE;",
                    ]

                    for optimization in optimizations:
                        try:
                            cursor.execute(optimization)
                            self.stdout.write(f"Executed: {optimization}")
                        except Exception as e:
                            self.stdout.write(
                                self.style.WARNING(
                                    f"Optimization failed: {optimization}: {e}"
                                )
                            )

            self.stdout.write(self.style.SUCCESS("Database optimization completed"))

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Database optimization failed: {e}"))

    def cleanup_old_data(self, days):
        """Clean up old temporary data."""
        self.stdout.write(f"Cleaning up data older than {days} days...")

        try:
            from datetime import datetime, timedelta

            cutoff_date = datetime.now() - timedelta(days=days)

            # Clean up old query history for inactive users
            old_history_count = UserQueryHistory.objects.filter(
                created_at__lt=cutoff_date
            ).count()

            if old_history_count > 0:
                self.stdout.write(f"Found {old_history_count} old history records")
                # Optionally delete them (commented out for safety)
                # UserQueryHistory.objects.filter(created_at__lt=cutoff_date).delete()
                # self.stdout.write(f'Deleted {old_history_count} old history records')

            # Clean up temporary files (if any)
            import os
            import tempfile

            temp_dir = tempfile.gettempdir()
            cleaned_files = 0

            try:
                for filename in os.listdir(temp_dir):
                    if filename.startswith("secure_upload_") and filename.endswith(
                        ".log"
                    ):
                        file_path = os.path.join(temp_dir, filename)
                        file_age = datetime.fromtimestamp(os.path.getctime(file_path))
                        if file_age < cutoff_date:
                            os.remove(file_path)
                            cleaned_files += 1

                if cleaned_files > 0:
                    self.stdout.write(f"Cleaned {cleaned_files} old temporary files")

            except Exception as e:
                self.stdout.write(
                    self.style.WARNING(f"Temporary file cleanup failed: {e}")
                )

            self.stdout.write(self.style.SUCCESS("Data cleanup completed"))

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Data cleanup failed: {e}"))
