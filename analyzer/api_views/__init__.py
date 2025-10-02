"""
API views package for QueryGrade analyzer.

This package contains all REST API views organized by functionality:
- pagination: QueryGradingPagination
- query_grading_api: grade_query_api, batch_analysis_api
- history_api: QueryHistoryListAPIView, QueryAnalysisDetailAPIView
- feedback_api: submit_feedback_api
- analytics_api: user_stats_api
- health_api: api_health

All views and classes are exported from this module for backward compatibility
with existing imports: `from analyzer.api_views import grade_query_api`
"""

from .pagination import QueryGradingPagination
from .query_grading_api import grade_query_api, batch_analysis_api
from .history_api import QueryHistoryListAPIView, QueryAnalysisDetailAPIView
from .feedback_api import submit_feedback_api
from .analytics_api import user_stats_api
from .health_api import api_health

__all__ = [
    # Pagination
    'QueryGradingPagination',
    # Query grading
    'grade_query_api',
    'batch_analysis_api',
    # History
    'QueryHistoryListAPIView',
    'QueryAnalysisDetailAPIView',
    # Feedback
    'submit_feedback_api',
    # Analytics
    'user_stats_api',
    # Health
    'api_health',
]
