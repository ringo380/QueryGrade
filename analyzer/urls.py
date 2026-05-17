"""
URL Configuration for analyzer application.

All views are now imported from the modular views package structure.
Legacy views_legacy.py has been fully migrated.
"""

from django.urls import path

# ML Dashboard views (separate module)
from .ml import dashboard_views
from .views import ml_alert_views

# Import from modular views package
from .views import (  # Authentication views; Query grading views; Comparison views; Batch analysis views; History and feedback views; Upload views; Database introspection views; Async processing views; API views; Saved connection views
    account_view,
    analyze,
    api_unified_query_analysis,
    async_processing_status,
    async_results,
    batch_analysis,
    batch_analysis_view,
    batch_results,
    check_task_status,
    compare_results,
    connection_create,
    connection_delete,
    connection_edit,
    connection_test,
    connections_list,
    contextualized_results,
    database_analyze,
    database_schema,
    enhanced_grade_results,
    feedback_analytics,
    grade_query,
    grade_query_ajax,
    grade_results,
    index,
    login_view,
    logout_view,
    password_change,
    password_reset_confirm,
    password_reset_request,
    performance_report_view,
    query_compare,
    query_history,
    query_with_context,
    quick_feedback,
    register_view,
    submit_feedback,
)

urlpatterns = [
    # Home and file upload
    path("", index, name="index"),
    path("analyze/", analyze, name="analyze"),
    # Query grading
    path("grade/", grade_query, name="grade_query"),
    path("grade/ajax/", grade_query_ajax, name="grade_query_ajax"),
    path("grade/results/<int:analysis_id>/", grade_results, name="grade_results"),
    path(
        "grade/enhanced/<int:analysis_id>/",
        enhanced_grade_results,
        name="enhanced_grade_results",
    ),
    # Query comparison
    path("compare/", query_compare, name="query_compare"),
    path("compare/results/", compare_results, name="compare_results"),
    # Batch analysis
    path("batch/", batch_analysis, name="batch_analysis"),
    path("batch/results/", batch_results, name="batch_results"),
    # User history and feedback
    path("history/", query_history, name="query_history"),
    path("feedback/<int:analysis_id>/", submit_feedback, name="submit_feedback"),
    path("feedback/quick/<int:analysis_id>/", quick_feedback, name="quick_feedback"),
    path("feedback/analytics/", feedback_analytics, name="feedback_analytics"),
    # Saved DB connections (live schema for index recommendations)
    path("connections/", connections_list, name="connections_list"),
    path("connections/new/", connection_create, name="connection_create"),
    path(
        "connections/<int:connection_id>/edit/",
        connection_edit,
        name="connection_edit",
    ),
    path(
        "connections/<int:connection_id>/delete/",
        connection_delete,
        name="connection_delete",
    ),
    path(
        "connections/<int:connection_id>/test/",
        connection_test,
        name="connection_test",
    ),
    # Database introspection
    path("database/", database_analyze, name="database_analyze"),
    path("database/schema/", database_schema, name="database_schema"),
    path("database/query/", query_with_context, name="query_with_context"),
    path(
        "database/results/<int:analysis_id>/",
        contextualized_results,
        name="contextualized_results",
    ),
    # Async processing
    path("async/status/", async_processing_status, name="async_processing_status"),
    path("async/check/<str:task_id>/", check_task_status, name="check_task_status"),
    path("async/results/", async_results, name="async_results"),
    path("async/batch/", batch_analysis_view, name="batch_analysis_async"),
    path("async/report/", performance_report_view, name="performance_report"),
    # ML Dashboard
    path("ml/dashboard/", dashboard_views.ml_dashboard, name="ml_dashboard"),
    path(
        "ml/api/overview/",
        dashboard_views.dashboard_api_overview,
        name="ml_api_overview",
    ),
    path("ml/api/models/", dashboard_views.dashboard_api_models, name="ml_api_models"),
    path(
        "ml/api/satisfaction/",
        dashboard_views.dashboard_api_satisfaction,
        name="ml_api_satisfaction",
    ),
    path(
        "ml/api/training/",
        dashboard_views.dashboard_api_training,
        name="ml_api_training",
    ),
    path(
        "ml/api/realtime/",
        dashboard_views.dashboard_api_realtime,
        name="ml_api_realtime",
    ),
    path(
        "ml/api/feature-importance/",
        dashboard_views.dashboard_api_feature_importance,
        name="ml_api_feature_importance",
    ),
    path(
        "ml/api/trigger-training/",
        dashboard_views.dashboard_api_trigger_training,
        name="ml_api_trigger_training",
    ),
    # ML alert triage (issue #5). List view ships in PR 5 of the stack.
    path(
        "ml/alerts/<int:alert_id>/ack/",
        ml_alert_views.ack_alert,
        name="ml_alert_ack",
    ),
    path(
        "ml/alerts/<int:alert_id>/dismiss/",
        ml_alert_views.dismiss_alert,
        name="ml_alert_dismiss",
    ),
    path(
        "ml/alerts/<int:alert_id>/false-positive/",
        ml_alert_views.mark_false_positive,
        name="ml_alert_false_positive",
    ),
    path(
        "ml/models/<int:model_id>/rollback/",
        ml_alert_views.rollback_model,
        name="ml_model_rollback",
    ),
    # API endpoints
    path(
        "api/unified-analysis/", api_unified_query_analysis, name="api_unified_analysis"
    ),
    # Authentication
    path("login/", login_view, name="login"),
    path("logout/", logout_view, name="logout"),
    path("register/", register_view, name="register"),
    path("password-reset/", password_reset_request, name="password_reset"),
    path(
        "password-reset-confirm/<uidb64>/<token>/",
        password_reset_confirm,
        name="password_reset_confirm",
    ),
    path("password-change/", password_change, name="password_change"),
    path("account/", account_view, name="account"),
]
