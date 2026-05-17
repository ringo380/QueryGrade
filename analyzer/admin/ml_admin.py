"""
Django admin configuration for machine learning models.

This module provides admin interfaces for ML-related models including
MLModel, TrainingData, LearningMetrics, and FeedbackLearning.
"""

import json

from django.contrib import admin
from django.contrib.admin import DateFieldListFilter
from django.utils.html import format_html
from django.utils.safestring import mark_safe

from ..models import FeedbackLearning, LearningMetrics, MLAlert, MLModel, TrainingData


@admin.register(MLModel)
class MLModelAdmin(admin.ModelAdmin):
    """Admin interface for MLModel model."""

    list_display = (
        "name",
        "model_type",
        "version",
        "status",
        "performance_display",
        "created_at",
    )
    list_filter = ("model_type", "status", ("created_at", DateFieldListFilter))
    search_fields = ("name", "version", "description")
    readonly_fields = ("created_at", "performance_chart")
    ordering = ("-created_at",)

    fieldsets = (
        (
            "Model Information",
            {"fields": ("name", "model_type", "version", "description", "status")},
        ),
        ("Model Files", {"fields": ("file_path", "file_size_bytes", "checksum")}),
        (
            "Performance Metrics",
            {
                "fields": (
                    "training_accuracy",
                    "validation_accuracy",
                    "performance_chart",
                ),
                "classes": ("wide",),
            },
        ),
        (
            "Training Information",
            {"fields": ("training_samples",), "classes": ("collapse",)},
        ),
        ("Timestamps", {"fields": ("created_at",), "classes": ("collapse",)}),
    )

    def performance_display(self, obj):
        if obj.training_accuracy:
            accuracy = obj.training_accuracy
            color = "green" if accuracy > 0.8 else "orange" if accuracy > 0.6 else "red"
            return format_html(
                '<span style="color: {};">{:.1%}</span>', color, accuracy
            )
        return "N/A"

    performance_display.short_description = "Accuracy"

    def performance_chart(self, obj):
        metrics = {
            "Training Accuracy": obj.training_accuracy,
            "Validation Accuracy": obj.validation_accuracy,
        }
        metrics_json = json.dumps(metrics, indent=2)
        return format_html(
            '<pre style="background: #f8f8f8; padding: 10px;">{}</pre>', metrics_json
        )

    performance_chart.short_description = "Performance Chart"

    actions = ["activate_model", "deactivate_model"]

    def activate_model(self, request, queryset):
        # Deactivate all models of the same type first
        for obj in queryset:
            MLModel.objects.filter(model_type=obj.model_type).update(
                status="DEPRECATED"
            )
            obj.status = "ACTIVE"
            obj.save()
        self.message_user(request, f"Activated {queryset.count()} models")

    activate_model.short_description = "Activate selected models"

    def deactivate_model(self, request, queryset):
        queryset.update(status="DEPRECATED")
        self.message_user(request, f"Deprecated {queryset.count()} models")

    deactivate_model.short_description = "Deprecate selected models"


@admin.register(TrainingData)
class TrainingDataAdmin(admin.ModelAdmin):
    """Admin interface for TrainingData model."""

    list_display = (
        "query_display",
        "user_grade_avg",
        "user_grade_count",
        "system_grade",
        "is_validated",
        "created_at",
    )
    list_filter = (
        "is_validated",
        "system_grade",
        "database_type",
        ("created_at", DateFieldListFilter),
    )
    search_fields = ("query__query_hash", "query__sql_text", "validation_source")
    readonly_fields = ("created_at", "updated_at")
    ordering = ("-updated_at",)

    fieldsets = (
        (
            "Query Information",
            {
                "fields": (
                    "query",
                    "database_type",
                    "query_complexity",
                    "table_count",
                    "join_count",
                )
            },
        ),
        (
            "Grade Information",
            {
                "fields": (
                    "user_grade_avg",
                    "user_grade_count",
                    "user_grade_stddev",
                    "system_grade",
                    "system_score",
                )
            },
        ),
        (
            "Feedback Ratings",
            {
                "fields": (
                    "accuracy_rating_avg",
                    "usefulness_rating_avg",
                    "clarity_rating_avg",
                ),
                "classes": ("collapse",),
            },
        ),
        ("Validation", {"fields": ("is_validated", "validation_source")}),
        (
            "Timestamps",
            {"fields": ("created_at", "updated_at"), "classes": ("collapse",)},
        ),
    )

    def query_display(self, obj):
        return f"{obj.query.query_type} - {obj.query.query_hash[:8]}"

    query_display.short_description = "Query"

    def get_queryset(self, request):
        return super().get_queryset(request).select_related("query")


@admin.register(LearningMetrics)
class LearningMetricsAdmin(admin.ModelAdmin):
    """Admin interface for LearningMetrics model."""

    list_display = (
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "avg_user_rating",
        "created_at",
    )
    list_filter = ("model", ("created_at", DateFieldListFilter))
    search_fields = ("model__name", "model__version")
    readonly_fields = ("created_at", "metrics_chart")
    ordering = ("-created_at",)

    fieldsets = (
        ("Model Information", {"fields": ("model",)}),
        (
            "Performance Metrics",
            {
                "fields": ("accuracy", "precision", "recall", "f1_score"),
                "classes": ("wide",),
            },
        ),
        (
            "User Metrics",
            {
                "fields": (
                    "user_agreement_rate",
                    "avg_user_rating",
                    "prediction_count",
                ),
            },
        ),
        (
            "Operational Metrics",
            {
                "fields": ("avg_prediction_time_ms", "error_rate"),
                "classes": ("collapse",),
            },
        ),
        (
            "Time Period",
            {
                "fields": ("measurement_period_start", "measurement_period_end"),
                "classes": ("collapse",),
            },
        ),
        ("Visualization", {"fields": ("metrics_chart",), "classes": ("wide",)}),
        ("Metadata", {"fields": ("created_at",), "classes": ("collapse",)}),
    )

    def metrics_chart(self, obj):
        metrics = {
            "Accuracy": obj.accuracy,
            "Precision": obj.precision,
            "Recall": obj.recall,
            "F1 Score": obj.f1_score,
            "User Agreement": obj.user_agreement_rate,
            "Avg User Rating": obj.avg_user_rating / 5.0,  # Normalize to 0-1
        }

        chart_html = '<div style="margin: 10px 0;">'
        for metric, value in metrics.items():
            if value is not None:
                percentage = value * 100 if value <= 1 else value
                color = "green" if value > 0.8 else "orange" if value > 0.6 else "red"
                chart_html += f"""
                <div style="margin: 5px 0;">
                    <strong>{metric}:</strong>
                    <div style="display: inline-block; width: 200px; height: 15px; background: #eee; margin-left: 10px; position: relative;">
                        <div style="width: {min(percentage, 100)}%; height: 100%; background: {color};"></div>
                        <span style="position: absolute; right: 5px; top: 0; font-size: 10px; line-height: 15px;">{percentage:.1f}%</span>
                    </div>
                </div>
                """
        chart_html += "</div>"

        return mark_safe(chart_html)

    metrics_chart.short_description = "Performance Chart"

    def get_queryset(self, request):
        return super().get_queryset(request).select_related("model")


@admin.register(FeedbackLearning)
class FeedbackLearningAdmin(admin.ModelAdmin):
    """Admin interface for FeedbackLearning model."""

    list_display = (
        "user_history_display",
        "original_grade",
        "original_score",
        "feedback_grade_equivalent",
        "grade_difference",
        "feedback_weight",
        "was_used_for_training",
        "created_at",
    )
    list_filter = (
        "was_used_for_training",
        "original_grade",
        ("created_at", DateFieldListFilter),
    )
    search_fields = (
        "user_history__user__username",
        "user_history__query__query_hash",
        "training_session",
    )
    readonly_fields = ("created_at", "processed_at")
    ordering = ("-created_at",)

    fieldsets = (
        ("User History", {"fields": ("user_history",)}),
        (
            "Original Prediction",
            {"fields": ("original_grade", "original_score", "original_confidence")},
        ),
        (
            "Feedback Impact",
            {
                "fields": (
                    "feedback_grade_equivalent",
                    "grade_difference",
                    "feedback_weight",
                )
            },
        ),
        (
            "Learning Application",
            {"fields": ("was_used_for_training", "training_session")},
        ),
        (
            "Meta-learning",
            {
                "fields": ("user_reliability_score", "context_similarity_score"),
                "classes": ("collapse",),
            },
        ),
        (
            "Timestamps",
            {"fields": ("created_at", "processed_at"), "classes": ("collapse",)},
        ),
    )

    def user_history_display(self, obj):
        return f"{obj.user_history.user.username} - {obj.user_history.query.query_hash[:8]}"

    user_history_display.short_description = "User/Query"

    def get_queryset(self, request):
        return (
            super()
            .get_queryset(request)
            .select_related("user_history__user", "user_history__query")
        )


@admin.register(MLAlert)
class MLAlertAdmin(admin.ModelAdmin):
    """Admin interface for MLAlert — monitoring signals raised by the
    periodic monitor_models task. Acknowledge / dismiss / mark false
    positive here for ad-hoc triage; the dashboard UI offers the same
    actions in a friendlier surface."""

    list_display = (
        "created_at",
        "severity",
        "alert_type",
        "status",
        "model",
        "short_message",
    )
    list_filter = (
        "severity",
        "status",
        "alert_type",
        ("created_at", DateFieldListFilter),
    )
    search_fields = ("message", "resolution", "model__name", "model__version")
    readonly_fields = ("created_at", "model", "alert_type", "payload_pretty")
    ordering = ("-created_at",)

    fieldsets = (
        (
            "Alert",
            {
                "fields": (
                    "model",
                    "alert_type",
                    "severity",
                    "status",
                    "message",
                    "payload_pretty",
                )
            },
        ),
        (
            "Triage",
            {
                "fields": (
                    "acknowledged_by",
                    "acknowledged_at",
                    "resolution",
                ),
            },
        ),
        ("Timestamps", {"fields": ("created_at",), "classes": ("collapse",)}),
    )

    def short_message(self, obj):
        return (obj.message[:60] + "…") if len(obj.message) > 60 else obj.message

    short_message.short_description = "Message"

    def payload_pretty(self, obj):
        if not obj.payload:
            return "—"
        return format_html("<pre>{}</pre>", json.dumps(obj.payload, indent=2))

    payload_pretty.short_description = "Payload"

    def get_queryset(self, request):
        return super().get_queryset(request).select_related("model", "acknowledged_by")
