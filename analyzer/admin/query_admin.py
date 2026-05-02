"""
Django admin configuration for core query models.

This module provides admin interfaces for Query and QueryAnalysis models
with custom display, filtering, and search capabilities.
"""

from django.contrib import admin
from django.contrib.admin import DateFieldListFilter

from ..models import Query


@admin.register(Query)
class QueryAdmin(admin.ModelAdmin):
    """Admin interface for Query model."""

    list_display = (
        "query_hash",
        "query_type",
        "estimated_complexity",
        "table_count",
        "join_count",
        "created_at",
    )
    list_filter = (
        "query_type",
        "table_count",
        "join_count",
        ("created_at", DateFieldListFilter),
    )
    search_fields = ("sql_text", "query_hash")
    readonly_fields = ("query_hash", "created_at", "updated_at")
    ordering = ("-created_at",)

    fieldsets = (
        ("Basic Information", {"fields": ("query_hash", "sql_text", "query_type")}),
        (
            "Complexity Metrics",
            {
                "fields": (
                    "estimated_complexity",
                    "table_count",
                    "join_count",
                    "where_conditions",
                    "subquery_count",
                )
            },
        ),
        (
            "Timestamps",
            {"fields": ("created_at", "updated_at"), "classes": ("collapse",)},
        ),
    )

    def get_queryset(self, request):
        return super().get_queryset(request).select_related()
