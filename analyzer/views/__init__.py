"""
Views package for QueryGrade analyzer.

This package organizes views into logical modules for better maintainability:
- auth_views: Authentication and user account management
- query_grading_views: Core query analysis and grading
- feedback_views: User feedback collection and analytics
- history_views: User query history browsing
- upload_views: File upload and processing
- comparison_views: Query comparison and batch analysis
- database_views: Database introspection and context-aware analysis
- async_views: Async processing, task status, and API endpoints

All views are re-exported here for backward compatibility with existing URL configurations.
"""

# Async processing and API views
from .async_views import (api_unified_query_analysis, batch_analysis_view,
                          performance_report_view)
# Authentication views
from .auth_views import (account_view, login_view, logout_view,
                         password_change, password_reset_confirm,
                         password_reset_request, register_view)
# Comparison and batch analysis views
from .comparison_views import batch_analysis, compare_results, query_compare
# Database introspection views
from .database_views import (contextualized_results, database_analyze,
                             database_schema, query_with_context)
# Feedback views
from .feedback_views import feedback_analytics, quick_feedback, submit_feedback
# History views
from .history_views import query_history
# Query grading views
from .query_grading_views import (batch_grade_queries, batch_results,
                                  compare_queries, enhanced_grade_results,
                                  grade_query, grade_results)
# Upload and async processing views
from .upload_views import (analyze, async_processing_status, async_results,
                           check_task_status, index)
# Utility functions
from .utils import csrf_failure, get_client_ip

# Alias for compatibility
user_query_history = query_history

__all__ = [
    # Auth
    "login_view",
    "logout_view",
    "register_view",
    "password_reset_request",
    "password_reset_confirm",
    "password_change",
    "account_view",
    # Grading
    "grade_query",
    "grade_results",
    "enhanced_grade_results",
    "compare_queries",
    "batch_grade_queries",
    "batch_results",
    # History
    "query_history",
    "user_query_history",
    # Feedback
    "submit_feedback",
    "quick_feedback",
    "feedback_analytics",
    # Upload & Async
    "index",
    "analyze",
    "check_task_status",
    "async_processing_status",
    "async_results",
    # Comparison
    "query_compare",
    "compare_results",
    "batch_analysis",
    # Database
    "database_analyze",
    "database_schema",
    "query_with_context",
    "contextualized_results",
    # Async API
    "batch_analysis_view",
    "performance_report_view",
    "api_unified_query_analysis",
    # Utils
    "get_client_ip",
    "csrf_failure",
]
