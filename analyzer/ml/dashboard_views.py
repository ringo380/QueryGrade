"""
ML Performance Monitoring Dashboard Views

This module provides views for the ML performance monitoring dashboard,
displaying real-time metrics, model performance, and system health.
"""

import json
from datetime import datetime, timedelta
from typing import Any, Dict, List

from django.contrib.auth.decorators import login_required, user_passes_test
from django.db.models import Avg, Count, Q
from django.http import JsonResponse
from django.shortcuts import render
from django.utils import timezone
from django.views.decorators.cache import cache_page
from django.views.decorators.http import require_http_methods

from analyzer.models import (FeedbackLearning, LearningMetrics, MLModel, Query,
                             QueryFeedback, TrainingData, UserQueryHistory)

from .core.hybrid_grader import HybridQueryGrader
from .core.training_pipeline import TrainingPipelineManager


def is_staff_or_superuser(user):
    """Check if user is staff or superuser."""
    return user.is_staff or user.is_superuser


@login_required
@user_passes_test(is_staff_or_superuser)
def ml_dashboard(request):
    """Main ML dashboard view."""
    context = {
        "page_title": "ML Performance Dashboard",
        "dashboard_sections": [
            "system_overview",
            "model_performance",
            "user_satisfaction",
            "training_pipeline",
            "real_time_metrics",
        ],
    }
    return render(request, "analyzer/ml_dashboard.html", context)


@login_required
@user_passes_test(is_staff_or_superuser)
@cache_page(60 * 5)  # Cache for 5 minutes
def dashboard_api_overview(request):
    """API endpoint for dashboard overview data."""
    try:
        # System metrics
        total_queries = Query.objects.count()
        total_feedback = QueryFeedback.objects.count()
        total_training_data = TrainingData.objects.count()
        active_models = MLModel.objects.filter(status="ACTIVE").count()

        # Recent activity (last 7 days)
        seven_days_ago = timezone.now() - timedelta(days=7)
        recent_queries = Query.objects.filter(created_at__gte=seven_days_ago).count()
        recent_feedback = QueryFeedback.objects.filter(
            created_at__gte=seven_days_ago
        ).count()

        # User engagement
        active_users = (
            QueryFeedback.objects.filter(created_at__gte=seven_days_ago)
            .values("user")
            .distinct()
            .count()
        )

        # Feedback rate
        feedback_rate = (
            (total_feedback / total_queries * 100) if total_queries > 0 else 0
        )

        # Average satisfaction
        avg_satisfaction = (
            QueryFeedback.objects.aggregate(avg=Avg("score_agreement"))["avg"] or 0
        )

        # Model performance
        latest_model = MLModel.objects.filter(status="ACTIVE").first()
        model_accuracy = (latest_model.validation_accuracy or 0) if latest_model else 0

        return JsonResponse(
            {
                "success": True,
                "data": {
                    "system_metrics": {
                        "total_queries": total_queries,
                        "total_feedback": total_feedback,
                        "total_training_data": total_training_data,
                        "active_models": active_models,
                        "feedback_rate": round(feedback_rate, 1),
                    },
                    "recent_activity": {
                        "recent_queries": recent_queries,
                        "recent_feedback": recent_feedback,
                        "active_users": active_users,
                    },
                    "performance_metrics": {
                        "avg_satisfaction": round(avg_satisfaction, 2),
                        "model_accuracy": (
                            round(model_accuracy * 100, 1) if model_accuracy else 0
                        ),
                    },
                    "timestamp": timezone.now().isoformat(),
                },
            }
        )

    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)}, status=500)


@login_required
@user_passes_test(is_staff_or_superuser)
def dashboard_api_models(request):
    """API endpoint for model performance data."""
    try:
        models_data = []

        for model in MLModel.objects.all().order_by("-created_at"):
            # Get recent feedback for this model
            recent_feedback = FeedbackLearning.objects.filter(
                created_at__gte=timezone.now() - timedelta(days=30)
            )

            user_satisfaction = (
                recent_feedback.aggregate(avg=Avg("user_feedback_score"))["avg"] or 0
            )

            models_data.append(
                {
                    "version": model.version,
                    "name": model.name,
                    "type": model.model_type,
                    "status": model.status,
                    "is_active": model.status == "ACTIVE",
                    "created_at": model.created_at.isoformat(),
                    "training_accuracy": model.training_accuracy or 0,
                    "validation_accuracy": model.validation_accuracy or 0,
                    "user_satisfaction": round(user_satisfaction, 2),
                    "training_samples": model.training_samples or 0,
                    "file_size_mb": round(
                        (model.file_size_bytes or 0) / (1024 * 1024), 2
                    ),
                }
            )

        return JsonResponse(
            {
                "success": True,
                "data": {
                    "models": models_data,
                    "total_models": len(models_data),
                    "active_models": len([m for m in models_data if m["is_active"]]),
                },
            }
        )

    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)}, status=500)


@login_required
@user_passes_test(is_staff_or_superuser)
def dashboard_api_satisfaction(request):
    """API endpoint for user satisfaction data."""
    try:
        # Overall satisfaction distribution
        satisfaction_distribution = {}
        for score in range(1, 6):
            count = QueryFeedback.objects.filter(score_agreement=score).count()
            satisfaction_distribution[str(score)] = count

        # Satisfaction by query type
        satisfaction_by_type = {}
        query_types = Query.objects.values_list("query_type", flat=True).distinct()

        for query_type in query_types:
            avg_satisfaction = QueryFeedback.objects.filter(
                query__query_type=query_type
            ).aggregate(avg=Avg("score_agreement"))["avg"]

            if avg_satisfaction:
                satisfaction_by_type[query_type] = round(avg_satisfaction, 2)

        # Satisfaction trend (last 30 days)
        thirty_days_ago = timezone.now() - timedelta(days=30)
        satisfaction_trend = []

        for i in range(30):
            date = thirty_days_ago + timedelta(days=i)
            daily_satisfaction = QueryFeedback.objects.filter(
                created_at__date=date.date()
            ).aggregate(avg=Avg("score_agreement"))["avg"]

            satisfaction_trend.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "satisfaction": (
                        round(daily_satisfaction, 2) if daily_satisfaction else None
                    ),
                }
            )

        # Top contributors
        top_contributors = (
            QueryFeedback.objects.values("user__username")
            .annotate(
                feedback_count=Count("id"), avg_satisfaction=Avg("score_agreement")
            )
            .order_by("-feedback_count")[:10]
        )

        return JsonResponse(
            {
                "success": True,
                "data": {
                    "satisfaction_distribution": satisfaction_distribution,
                    "satisfaction_by_type": satisfaction_by_type,
                    "satisfaction_trend": satisfaction_trend,
                    "top_contributors": list(top_contributors),
                    "overall_average": QueryFeedback.objects.aggregate(
                        avg=Avg("score_agreement")
                    )["avg"]
                    or 0,
                },
            }
        )

    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)}, status=500)


@login_required
@user_passes_test(is_staff_or_superuser)
def dashboard_api_training(request):
    """API endpoint for training pipeline data."""
    try:
        # Training pipeline status
        pipeline_manager = TrainingPipelineManager()
        training_status = pipeline_manager.get_training_status()

        # Recent training data creation
        thirty_days_ago = timezone.now() - timedelta(days=30)
        recent_training_data = TrainingData.objects.filter(
            created_at__gte=thirty_days_ago
        ).count()

        # Training data quality metrics
        training_data_quality = {
            "total_samples": TrainingData.objects.count(),
            "avg_feedback_weight": TrainingData.objects.aggregate(
                avg=Avg("feedback_weight")
            )["avg"]
            or 0,
            "avg_user_reliability": TrainingData.objects.aggregate(
                avg=Avg("user_reliability_score")
            )["avg"]
            or 0,
            "recent_samples": recent_training_data,
        }

        # Learning metrics history
        learning_metrics = LearningMetrics.objects.select_related("model").order_by(
            "-created_at"
        )[:10]
        metrics_history = []

        for metric in learning_metrics:
            metrics_history.append(
                {
                    "model_version": metric.model.version,
                    "date": metric.created_at.isoformat(),
                    "accuracy": metric.accuracy,
                    "precision": metric.precision,
                    "recall": metric.recall,
                    "f1_score": metric.f1_score,
                    "user_agreement_rate": metric.user_agreement_rate,
                    "avg_user_rating": metric.avg_user_rating,
                    "prediction_count": metric.prediction_count,
                }
            )

        return JsonResponse(
            {
                "success": True,
                "data": {
                    "pipeline_status": training_status,
                    "training_data_quality": training_data_quality,
                    "metrics_history": metrics_history,
                    "recommendations": _get_training_recommendations(training_status),
                },
            }
        )

    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)}, status=500)


@login_required
@user_passes_test(is_staff_or_superuser)
def dashboard_api_realtime(request):
    """API endpoint for real-time metrics."""
    try:
        # Metrics for the last hour
        one_hour_ago = timezone.now() - timedelta(hours=1)

        # Recent activity
        recent_activity = {
            "queries_last_hour": Query.objects.filter(
                created_at__gte=one_hour_ago
            ).count(),
            "feedback_last_hour": QueryFeedback.objects.filter(
                created_at__gte=one_hour_ago
            ).count(),
            "active_users_last_hour": QueryFeedback.objects.filter(
                created_at__gte=one_hour_ago
            )
            .values("user")
            .distinct()
            .count(),
        }

        # System health indicators
        system_health = {
            "ml_system_active": MLModel.objects.filter(status="ACTIVE").exists(),
            "recent_errors": 0,  # This would come from error logging
            "cache_hit_rate": 0.95,  # This would come from cache metrics
            "avg_response_time": 0.250,  # This would come from performance monitoring
        }

        # Performance alerts
        alerts = []

        # Check for performance issues
        latest_model = MLModel.objects.filter(status="ACTIVE").first()
        if latest_model:
            accuracy = latest_model.validation_accuracy or 0
            if accuracy < 0.7:
                alerts.append(
                    {
                        "type": "warning",
                        "message": f"Model accuracy below threshold: {accuracy:.1%}",
                        "severity": "medium",
                    }
                )

        # Check for low feedback volume
        if recent_activity["feedback_last_hour"] == 0:
            alerts.append(
                {
                    "type": "info",
                    "message": "No feedback received in the last hour",
                    "severity": "low",
                }
            )

        return JsonResponse(
            {
                "success": True,
                "data": {
                    "recent_activity": recent_activity,
                    "system_health": system_health,
                    "alerts": alerts,
                    "timestamp": timezone.now().isoformat(),
                },
            }
        )

    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)}, status=500)


@login_required
@user_passes_test(is_staff_or_superuser)
@require_http_methods(["POST"])
def dashboard_api_trigger_training(request):
    """API endpoint to trigger model training."""
    try:
        force_retrain = request.POST.get("force", "false").lower() == "true"

        pipeline_manager = TrainingPipelineManager()
        result = pipeline_manager.run_training_pipeline(force_retrain=force_retrain)

        return JsonResponse(
            {
                "success": result.success,
                "data": {
                    "model_version": result.model_version,
                    "training_accuracy": result.training_accuracy,
                    "validation_accuracy": result.validation_accuracy,
                    "training_time": result.training_time,
                    "error_message": result.error_message,
                },
            }
        )

    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)}, status=500)


@login_required
@user_passes_test(is_staff_or_superuser)
def dashboard_api_feature_importance(request):
    """API endpoint for feature importance data."""
    try:
        # Feature importance is no longer persisted on MLModel directly.
        active_model = MLModel.objects.filter(status="ACTIVE").first()

        if not active_model:
            return JsonResponse({"success": False, "error": "No active model found"})

        return JsonResponse(
            {
                "success": False,
                "error": "Feature importance not persisted in current schema; load from model bundle to surface.",
            }
        )

        feature_importance = {}

        # Sort by importance
        sorted_features = sorted(
            feature_importance.items(), key=lambda x: x[1], reverse=True
        )

        # Group features by category
        feature_categories = {
            "Basic Structure": [
                "query_length",
                "token_count",
                "keyword_count",
                "identifier_count",
            ],
            "Complexity": [
                "table_count",
                "join_count",
                "where_conditions",
                "subquery_count",
            ],
            "Query Type": ["is_select", "is_insert", "is_update", "is_delete"],
            "Performance": [
                "select_star_present",
                "missing_where_clause",
                "cartesian_product_risk",
            ],
            "Advanced": ["nesting_level", "function_call_count", "union_count"],
        }

        category_importance = {}
        for category, features in feature_categories.items():
            total_importance = sum(
                importance
                for feature, importance in sorted_features
                if feature in features
            )
            category_importance[category] = total_importance

        return JsonResponse(
            {
                "success": True,
                "data": {
                    "feature_importance": dict(sorted_features),
                    "top_features": sorted_features[:15],  # Top 15 features
                    "category_importance": category_importance,
                    "model_version": active_model.version,
                },
            }
        )

    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)}, status=500)


def _get_training_recommendations(
    training_status: Dict[str, Any],
) -> List[Dict[str, str]]:
    """Generate training recommendations based on current status."""
    recommendations = []

    # Check training data volume
    if not training_status["training_data"]["ready_for_training"]:
        recommendations.append(
            {
                "type": "warning",
                "title": "Insufficient Training Data",
                "message": "Collect more user feedback to improve model training.",
                "action": "Encourage users to provide feedback on query results",
            }
        )

    # Check recent activity
    if training_status["recent_activity"]["feedback_last_week"] < 10:
        recommendations.append(
            {
                "type": "info",
                "title": "Low Recent Activity",
                "message": "Consider running targeted campaigns to collect more feedback.",
                "action": "Implement feedback collection improvements",
            }
        )

    # Check model performance
    latest_model = training_status["latest_model"]
    if latest_model["performance"]:
        accuracy = latest_model["performance"].get("validation_accuracy", 0)
        if accuracy < 0.7:
            recommendations.append(
                {
                    "type": "warning",
                    "title": "Low Model Accuracy",
                    "message": f"Current model accuracy is {accuracy:.1%}. Consider retraining.",
                    "action": "Trigger model retraining with updated data",
                }
            )

    # Check model age
    if latest_model["created_at"]:
        model_age = (
            timezone.now() - datetime.fromisoformat(latest_model["created_at"])
        ).days
        if model_age > 30:
            recommendations.append(
                {
                    "type": "info",
                    "title": "Model Age",
                    "message": f"Current model is {model_age} days old.",
                    "action": "Consider scheduled retraining",
                }
            )

    return recommendations
