from django.urls import path
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
    TokenVerifyView,
)

from . import api_views

app_name = "api"

urlpatterns = [
    # Authentication endpoints
    path("auth/token/", TokenObtainPairView.as_view(), name="token_obtain_pair"),
    path("auth/token/refresh/", TokenRefreshView.as_view(), name="token_refresh"),
    path("auth/token/verify/", TokenVerifyView.as_view(), name="token_verify"),
    # Query grading endpoints
    path("grade-query/", api_views.grade_query_api, name="grade_query"),
    path("batch-analysis/", api_views.batch_analysis_api, name="batch_analysis"),
    # Query history and analysis
    path(
        "query-history/",
        api_views.QueryHistoryListAPIView.as_view(),
        name="query_history",
    ),
    path(
        "query-history/delete/",
        api_views.delete_query_history,
        name="delete_query_history",
    ),
    path(
        "analysis/<int:pk>/",
        api_views.QueryAnalysisDetailAPIView.as_view(),
        name="analysis_detail",
    ),
    # Feedback endpoints
    path(
        "feedback/<int:analysis_id>/",
        api_views.submit_feedback_api,
        name="submit_feedback",
    ),
    # User statistics
    path("user-stats/", api_views.user_stats_api, name="user_stats"),
    # Health check
    path("health/", api_views.api_health, name="health"),
]
