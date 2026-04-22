"""
Report generation tasks.

This module contains Celery tasks for generating user performance
reports and analytics summaries.
"""

import logging
from datetime import timedelta
from typing import Any, Dict

from celery import shared_task
from django.contrib.auth.models import User
from django.core.cache import caches
from django.db.models import Avg, Count, Q
from django.utils import timezone

from ..models import QueryAnalysis

logger = logging.getLogger(__name__)


@shared_task(bind=True, name="analyzer.tasks.generate_performance_report")
def generate_performance_report(
    self, user_id: int, date_range_days: int = 30
) -> Dict[str, Any]:
    """
    Generate comprehensive performance report for user's query history.

    Args:
        user_id: ID of the user
        date_range_days: Number of days to include in report

    Returns:
        Dict containing performance report data
    """
    try:
        user = User.objects.get(id=user_id)
        end_date = timezone.now()
        start_date = end_date - timedelta(days=date_range_days)

        logger.info(f"Generating performance report for user {user.username}")

        # Get query analyses in date range
        analyses = QueryAnalysis.objects.filter(
            query__user=user, created_at__gte=start_date, created_at__lte=end_date
        )

        # Calculate statistics
        total_queries = analyses.count()
        avg_grade = analyses.aggregate(avg_grade=Avg("grade"))["avg_grade"] or 0

        grade_distribution = {}
        for grade in ["A", "B", "C", "D", "F"]:
            count = analyses.filter(grade=grade).count()
            grade_distribution[grade] = {
                "count": count,
                "percentage": (count / total_queries * 100) if total_queries > 0 else 0,
            }

        # Performance trends
        improvement_queries = analyses.filter(feedback_provided=True).count()

        # Most common issues
        issue_keywords = ["index", "join", "subquery", "performance", "optimization"]
        common_issues = {}
        for keyword in issue_keywords:
            count = analyses.filter(
                Q(feedback__icontains=keyword) | Q(recommendations__icontains=keyword)
            ).count()
            if count > 0:
                common_issues[keyword] = count

        report_data = {
            "user_info": {
                "username": user.username,
                "report_period": f"{start_date.date()} to {end_date.date()}",
            },
            "summary": {
                "total_queries_analyzed": total_queries,
                "average_grade": round(avg_grade, 2),
                "improvement_queries": improvement_queries,
                "grade_distribution": grade_distribution,
            },
            "insights": {
                "common_issues": common_issues,
                "recommendations": [
                    "Focus on query optimization techniques",
                    "Review indexing strategies",
                    "Consider query refactoring for better performance",
                ],
            },
        }

        # Cache report
        cache = caches["process_cache"]
        cache_key = f"performance_report_{user_id}_{self.request.id}"
        cache.set(cache_key, report_data, timeout=86400)  # 24 hours

        logger.info(f"Generated performance report for user {user.username}")

        return {
            "status": "success",
            "cache_key": cache_key,
            "report_summary": {
                "total_queries": total_queries,
                "average_grade": round(avg_grade, 2),
                "date_range": date_range_days,
            },
            "timestamp": timezone.now().isoformat(),
        }

    except Exception as exc:
        logger.error(f"Error generating performance report: {str(exc)}")
        return {
            "status": "error",
            "error": str(exc),
            "timestamp": timezone.now().isoformat(),
        }
