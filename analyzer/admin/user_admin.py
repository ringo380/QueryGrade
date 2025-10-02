"""
Django admin configuration for user interaction models.

This module provides admin interfaces for UserQueryHistory and QueryFeedback
models with custom display, filtering, and search capabilities.
"""
from django.contrib import admin
from django.contrib.admin import DateFieldListFilter

from ..models import QueryFeedback, UserQueryHistory


@admin.register(QueryFeedback)
class QueryFeedbackAdmin(admin.ModelAdmin):
    """Admin interface for QueryFeedback model."""
    list_display = ('user_history_display', 'accuracy_rating', 'usefulness_rating', 'clarity_rating', 'created_at')
    list_filter = ('accuracy_rating', 'usefulness_rating', ('created_at', DateFieldListFilter))
    search_fields = ('user_history__user__username', 'improvement_suggestions', 'additional_comments')
    readonly_fields = ('created_at',)
    ordering = ('-created_at',)

    def user_history_display(self, obj):
        return f"{obj.user_history.user.username} - {obj.user_history.query.query_hash[:8]}"
    user_history_display.short_description = 'User/Query'

    def get_queryset(self, request):
        return super().get_queryset(request).select_related('user_history__user', 'user_history__query')


@admin.register(UserQueryHistory)
class UserQueryHistoryAdmin(admin.ModelAdmin):
    """Admin interface for UserQueryHistory model."""
    list_display = ('user', 'query_display', 'database_type', 'was_helpful', 'submitted_at')
    list_filter = ('database_type', 'was_helpful', ('submitted_at', DateFieldListFilter))
    search_fields = ('user__username', 'user__email', 'use_case_notes')
    readonly_fields = ('submitted_at',)
    ordering = ('-submitted_at',)

    def query_display(self, obj):
        return f"{obj.query.query_type} - {obj.query.query_hash[:8]}"
    query_display.short_description = 'Query'

    def get_queryset(self, request):
        return super().get_queryset(request).select_related('user', 'query')
