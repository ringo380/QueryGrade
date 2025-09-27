from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('analyze/', views.analyze, name='analyze'),
    path('grade/', views.grade_query, name='grade_query'),
    path('grade/results/<int:analysis_id>/', views.grade_results, name='grade_results'),
    path('compare/', views.query_compare, name='query_compare'),
    path('compare/results/', views.compare_results, name='compare_results'),
    path('batch/', views.batch_analysis, name='batch_analysis'),
    path('batch/results/', views.batch_results, name='batch_results'),
    path('history/', views.user_query_history, name='query_history'),
    path('feedback/<int:analysis_id>/', views.submit_feedback, name='submit_feedback'),
    path('feedback/analytics/', views.feedback_analytics, name='feedback_analytics'),
    path('database/', views.database_analyze, name='database_analyze'),
    path('database/schema/', views.database_schema, name='database_schema'),
    path('database/query/', views.query_with_context, name='query_with_context'),
    path('database/results/<int:analysis_id>/', views.contextualized_results, name='contextualized_results'),
    # Async processing URLs
    path('async/status/', views.async_processing_status, name='async_processing_status'),
    path('async/check/<str:task_id>/', views.check_task_status, name='check_task_status'),
    path('async/results/', views.async_results, name='async_results'),
    path('async/batch/', views.batch_analysis_view, name='batch_analysis_async'),
    path('async/report/', views.performance_report_view, name='performance_report'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('register/', views.register_view, name='register'),
]
