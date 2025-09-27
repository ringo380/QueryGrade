from django.contrib import admin
from django.utils.html import format_html
from django.utils.safestring import mark_safe
from django.urls import reverse
from django.contrib.admin import DateFieldListFilter
import json

from .models import (
    Query, QueryFeedback, UserQueryHistory,
    MLModel, TrainingData, LearningMetrics, FeedbackLearning
)


@admin.register(Query)
class QueryAdmin(admin.ModelAdmin):
    """Admin interface for Query model."""
    list_display = ('query_hash', 'query_type', 'estimated_complexity', 'table_count', 'join_count', 'created_at')
    list_filter = ('query_type', 'table_count', 'join_count', ('created_at', DateFieldListFilter))
    search_fields = ('sql_text', 'query_hash')
    readonly_fields = ('query_hash', 'created_at', 'updated_at')
    ordering = ('-created_at',)

    fieldsets = (
        ('Basic Information', {
            'fields': ('query_hash', 'sql_text', 'query_type')
        }),
        ('Complexity Metrics', {
            'fields': ('estimated_complexity', 'table_count', 'join_count', 'where_conditions', 'subquery_count')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        })
    )

    def get_queryset(self, request):
        return super().get_queryset(request).select_related()


@admin.register(QueryFeedback)
class QueryFeedbackAdmin(admin.ModelAdmin):
    """Admin interface for QueryFeedback model."""
    list_display = ('query_display', 'user', 'is_helpful', 'score_agreement', 'created_at')
    list_filter = ('is_helpful', 'score_agreement', ('created_at', DateFieldListFilter))
    search_fields = ('user__username', 'user__email', 'comments')
    readonly_fields = ('created_at',)
    ordering = ('-created_at',)

    def query_display(self, obj):
        return f"{obj.query.query_type} - {obj.query.query_hash[:8]}"
    query_display.short_description = 'Query'

    def get_queryset(self, request):
        return super().get_queryset(request).select_related('user', 'query')


@admin.register(UserQueryHistory)
class UserQueryHistoryAdmin(admin.ModelAdmin):
    """Admin interface for UserQueryHistory model."""
    list_display = ('user', 'query_display', 'execution_time', 'analysis_score', 'created_at')
    list_filter = ('execution_time', ('created_at', DateFieldListFilter))
    search_fields = ('user__username', 'user__email')
    readonly_fields = ('created_at',)
    ordering = ('-created_at',)

    def query_display(self, obj):
        return f"{obj.query.query_type} - {obj.query.query_hash[:8]}"
    query_display.short_description = 'Query'

    def analysis_score(self, obj):
        if obj.analysis_result and 'score' in obj.analysis_result:
            return f"{obj.analysis_result['score']}"
        return 'N/A'
    analysis_score.short_description = 'Score'

    def get_queryset(self, request):
        return super().get_queryset(request).select_related('user', 'query')


@admin.register(MLModel)
class MLModelAdmin(admin.ModelAdmin):
    """Admin interface for MLModel model."""
    list_display = ('name', 'model_type', 'version', 'is_active', 'performance_display', 'created_at')
    list_filter = ('model_type', 'is_active', ('created_at', DateFieldListFilter))
    search_fields = ('name', 'version', 'description')
    readonly_fields = ('created_at', 'updated_at', 'performance_chart')
    ordering = ('-created_at',)

    fieldsets = (
        ('Model Information', {
            'fields': ('name', 'model_type', 'version', 'description', 'is_active')
        }),
        ('Model Files', {
            'fields': ('file_path', 'file_size')
        }),
        ('Performance Metrics', {
            'fields': ('performance_metrics', 'performance_chart'),
            'classes': ('wide',)
        }),
        ('Training Information', {
            'fields': ('training_data_count', 'last_trained'),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        })
    )

    def performance_display(self, obj):
        if obj.performance_metrics:
            accuracy = obj.performance_metrics.get('accuracy', 'N/A')
            if accuracy != 'N/A':
                color = 'green' if accuracy > 0.8 else 'orange' if accuracy > 0.6 else 'red'
                return format_html(
                    '<span style="color: {};">{:.1%}</span>',
                    color, accuracy
                )
        return 'N/A'
    performance_display.short_description = 'Accuracy'

    def performance_chart(self, obj):
        if obj.performance_metrics:
            metrics_json = json.dumps(obj.performance_metrics, indent=2)
            return format_html('<pre style="background: #f8f8f8; padding: 10px;">{}</pre>', metrics_json)
        return 'No metrics available'
    performance_chart.short_description = 'Performance Chart'

    actions = ['activate_model', 'deactivate_model', 'trigger_training']

    def activate_model(self, request, queryset):
        # Deactivate all models of the same type first
        for obj in queryset:
            MLModel.objects.filter(model_type=obj.model_type).update(is_active=False)
            obj.is_active = True
            obj.save()
        self.message_user(request, f"Activated {queryset.count()} models")
    activate_model.short_description = "Activate selected models"

    def deactivate_model(self, request, queryset):
        queryset.update(is_active=False)
        self.message_user(request, f"Deactivated {queryset.count()} models")
    deactivate_model.short_description = "Deactivate selected models"

    def trigger_training(self, request, queryset):
        # This would trigger model training - for now just a message
        self.message_user(request, f"Training triggered for {queryset.count()} models (feature not implemented)")
    trigger_training.short_description = "Trigger model training"


@admin.register(TrainingData)
class TrainingDataAdmin(admin.ModelAdmin):
    """Admin interface for TrainingData model."""
    list_display = ('query_display', 'target_score', 'feedback_weight', 'user_reliability_score', 'created_date')
    list_filter = ('target_score', 'feedback_weight', ('created_date', DateFieldListFilter))
    search_fields = ('query__query_hash', 'query__sql_text')
    readonly_fields = ('created_date', 'feature_display')
    ordering = ('-created_date',)

    fieldsets = (
        ('Training Information', {
            'fields': ('query', 'target_score', 'feedback_weight', 'user_reliability_score')
        }),
        ('Features', {
            'fields': ('features_json', 'feature_display'),
            'classes': ('wide',)
        }),
        ('Metadata', {
            'fields': ('created_date',),
            'classes': ('collapse',)
        })
    )

    def query_display(self, obj):
        return f"{obj.query.query_type} - {obj.query.query_hash[:8]}"
    query_display.short_description = 'Query'

    def feature_display(self, obj):
        if obj.features_json:
            # Display first 10 features for preview
            features = obj.features_json[:10]
            preview = ', '.join([f'{f:.2f}' for f in features])
            if len(obj.features_json) > 10:
                preview += f' ... ({len(obj.features_json)} total features)'
            return format_html('<code>{}</code>', preview)
        return 'No features'
    feature_display.short_description = 'Feature Preview'

    def get_queryset(self, request):
        return super().get_queryset(request).select_related('query')


@admin.register(LearningMetrics)
class LearningMetricsAdmin(admin.ModelAdmin):
    """Admin interface for LearningMetrics model."""
    list_display = ('model_version', 'training_accuracy', 'validation_accuracy', 'feedback_correlation', 'user_satisfaction_avg', 'created_date')
    list_filter = ('model_version', ('created_date', DateFieldListFilter))
    search_fields = ('model_version',)
    readonly_fields = ('created_date', 'metrics_chart')
    ordering = ('-created_date',)

    fieldsets = (
        ('Model Information', {
            'fields': ('model_version',)
        }),
        ('Performance Metrics', {
            'fields': ('training_accuracy', 'validation_accuracy', 'feedback_correlation'),
            'classes': ('wide',)
        }),
        ('User Metrics', {
            'fields': ('user_satisfaction_avg', 'total_feedback_count'),
        }),
        ('Visualization', {
            'fields': ('metrics_chart',),
            'classes': ('wide',)
        }),
        ('Metadata', {
            'fields': ('created_date',),
            'classes': ('collapse',)
        })
    )

    def metrics_chart(self, obj):
        metrics = {
            'Training Accuracy': obj.training_accuracy,
            'Validation Accuracy': obj.validation_accuracy,
            'Feedback Correlation': obj.feedback_correlation,
            'User Satisfaction': obj.user_satisfaction_avg / 5.0 if obj.user_satisfaction_avg else 0,  # Normalize to 0-1
        }

        chart_html = '<div style="margin: 10px 0;">'
        for metric, value in metrics.items():
            if value is not None:
                percentage = value * 100 if value <= 1 else value
                color = 'green' if value > 0.8 else 'orange' if value > 0.6 else 'red'
                chart_html += f'''
                <div style="margin: 5px 0;">
                    <strong>{metric}:</strong>
                    <div style="display: inline-block; width: 200px; height: 15px; background: #eee; margin-left: 10px; position: relative;">
                        <div style="width: {min(percentage, 100)}%; height: 100%; background: {color};"></div>
                        <span style="position: absolute; right: 5px; top: 0; font-size: 10px; line-height: 15px;">{percentage:.1f}%</span>
                    </div>
                </div>
                '''
        chart_html += '</div>'

        return mark_safe(chart_html)
    metrics_chart.short_description = 'Performance Chart'


@admin.register(FeedbackLearning)
class FeedbackLearningAdmin(admin.ModelAdmin):
    """Admin interface for FeedbackLearning model."""
    list_display = ('query_display', 'user', 'original_score', 'user_feedback_score', 'agreement_level', 'learning_weight', 'model_version', 'created_at')
    list_filter = ('agreement_level', 'model_version', ('created_at', DateFieldListFilter))
    search_fields = ('user__username', 'query__query_hash', 'model_version')
    readonly_fields = ('created_at',)
    ordering = ('-created_at',)

    fieldsets = (
        ('Learning Information', {
            'fields': ('query', 'user', 'model_version')
        }),
        ('Scores and Agreement', {
            'fields': ('original_score', 'user_feedback_score', 'agreement_level', 'learning_weight')
        }),
        ('Metadata', {
            'fields': ('created_at',),
            'classes': ('collapse',)
        })
    )

    def query_display(self, obj):
        return f"{obj.query.query_type} - {obj.query.query_hash[:8]}"
    query_display.short_description = 'Query'

    def get_queryset(self, request):
        return super().get_queryset(request).select_related('user', 'query')


# Custom admin site configuration
admin.site.site_header = 'QueryGrade ML Administration'
admin.site.site_title = 'QueryGrade ML Admin'
admin.site.index_title = 'Machine Learning Dashboard'
