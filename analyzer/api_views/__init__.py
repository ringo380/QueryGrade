"""
API views package for QueryGrade analyzer.

This package contains all REST API views organized by functionality:
- pagination: QueryGradingPagination
- query_grading_api: grade_query_api, batch_analysis_api
- history_api: QueryHistoryListAPIView, QueryAnalysisDetailAPIView, delete_query_history
- feedback_api: submit_feedback_api
- analytics_api: user_stats_api
- health_api: api_health

All views and classes are exported from this module for backward compatibility
with existing imports: `from analyzer.api_views import grade_query_api`
"""

from .analytics_api import user_stats_api
from .feedback_api import submit_feedback_api
from .health_api import api_health
from .history_api import (
    QueryAnalysisDetailAPIView,
    QueryHistoryListAPIView,
    delete_query_history,
)
from .pagination import QueryGradingPagination
from .query_grading_api import batch_analysis_api, grade_query_api

__all__ = [
    # Pagination
    "QueryGradingPagination",
    # Query grading
    "grade_query_api",
    "batch_analysis_api",
    # History
    "QueryHistoryListAPIView",
    "QueryAnalysisDetailAPIView",
    "delete_query_history",
    # Feedback
    "submit_feedback_api",
    # Analytics
    "user_stats_api",
    # Health
    "api_health",
]
