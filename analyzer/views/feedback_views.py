"""
Feedback collection views for query analysis.
"""

import json
import logging

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.db.models import Avg, Count
from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.http import require_POST

from ..forms import QueryFeedbackForm
from ..models import FeedbackLearning, QueryFeedback, UserQueryHistory

logger = logging.getLogger(__name__)


@login_required
def submit_feedback(request, analysis_id):
    """Submit detailed feedback for a query analysis."""
    try:
        user_history = UserQueryHistory.objects.get(
            query__analysis__id=analysis_id, user=request.user
        )
    except UserQueryHistory.DoesNotExist:
        logger.warning(
            f"User {request.user.username} attempted to provide feedback for non-existent analysis {analysis_id}"
        )
        messages.error(
            request,
            "Analysis not found or you don't have permission to provide feedback.",
        )
        return redirect("query_history")

    # Check if feedback already exists
    existing_feedback = QueryFeedback.objects.filter(user_history=user_history).first()

    if request.method == "POST":
        form = QueryFeedbackForm(request.POST)
        if form.is_valid():
            if existing_feedback:
                # Update existing feedback
                existing_feedback.accuracy_rating = int(
                    form.cleaned_data["accuracy_rating"]
                )
                existing_feedback.usefulness_rating = int(
                    form.cleaned_data["usefulness_rating"]
                )
                existing_feedback.clarity_rating = int(
                    form.cleaned_data["clarity_rating"]
                )
                existing_feedback.suggestions = form.cleaned_data["suggestions"]
                existing_feedback.would_recommend = form.cleaned_data["would_recommend"]
                existing_feedback.save()

                logger.info(
                    f"User {request.user.username} updated feedback for analysis {analysis_id}"
                )
                messages.success(request, "Thank you! Your feedback has been updated.")
            else:
                # Create new feedback
                feedback = QueryFeedback.objects.create(
                    user_history=user_history,
                    accuracy_rating=int(form.cleaned_data["accuracy_rating"]),
                    usefulness_rating=int(form.cleaned_data["usefulness_rating"]),
                    clarity_rating=int(form.cleaned_data["clarity_rating"]),
                    suggestions=form.cleaned_data["suggestions"],
                    would_recommend=form.cleaned_data["would_recommend"],
                )

                logger.info(
                    f"User {request.user.username} submitted feedback for analysis {analysis_id}"
                )
                messages.success(
                    request,
                    "Thank you for your feedback! It helps us improve QueryGrade.",
                )

            # Update user history feedback flags
            user_history.was_helpful = True
            user_history.feedback_comments = form.cleaned_data["suggestions"]
            user_history.save()

            return redirect("grade_results", analysis_id=analysis_id)
        else:
            messages.error(request, "Please correct the errors in the feedback form.")
    else:
        # Pre-populate form if feedback exists
        initial_data = {}
        if existing_feedback:
            initial_data = {
                "accuracy_rating": existing_feedback.accuracy_rating,
                "usefulness_rating": existing_feedback.usefulness_rating,
                "clarity_rating": existing_feedback.clarity_rating,
                "suggestions": existing_feedback.suggestions,
                "would_recommend": existing_feedback.would_recommend,
            }
        form = QueryFeedbackForm(initial=initial_data)

    context = {
        "form": form,
        "user_history": user_history,
        "analysis": user_history.query.analysis,
        "existing_feedback": existing_feedback,
    }

    return render(request, "analyzer/feedback_form.html", context)


@login_required
@require_POST
def quick_feedback(request, analysis_id):
    """Handle quick thumbs up/down feedback via AJAX."""
    try:
        # Get the user's query history for this analysis
        user_history = UserQueryHistory.objects.get(
            query__analysis__id=analysis_id, user=request.user
        )
    except UserQueryHistory.DoesNotExist:
        logger.warning(
            f"User {request.user.username} attempted quick feedback for non-existent analysis {analysis_id}"
        )
        return JsonResponse(
            {
                "success": False,
                "error": "Analysis not found or you don't have permission to provide feedback.",
            },
            status=404,
        )

    try:
        # Parse JSON data
        data = json.loads(request.body)
        was_helpful = data.get("was_helpful")

        if was_helpful is None:
            return JsonResponse(
                {"success": False, "error": "Missing feedback data"}, status=400
            )

        # Update the user history with quick feedback
        user_history.was_helpful = bool(was_helpful)
        user_history.save()

        # Try to create or update the FeedbackLearning record for ML training
        try:
            # Convert thumbs up/down to grade equivalent (1-5 scale)
            feedback_grade = 4.0 if was_helpful else 2.0
            feedback_score = (feedback_grade - 1) * 25  # Convert to 0-100 scale

            analysis = user_history.query.analysis
            grade_difference = feedback_score - analysis.score

            # Create or update learning record
            learning_record, created = FeedbackLearning.objects.update_or_create(
                user_history=user_history,
                defaults={
                    "original_grade": analysis.grade,
                    "original_score": analysis.score,
                    "original_confidence": 0.8,  # Default system confidence
                    "feedback_grade_equivalent": feedback_score,
                    "grade_difference": grade_difference,
                    "feedback_weight": 0.7,  # Medium weight for quick feedback
                    "user_reliability_score": 0.5,  # Default for new feedback
                    "context_similarity_score": 0.0,
                },
            )

            logger.info(
                f"{'Created' if created else 'Updated'} ML learning record for analysis {analysis_id}"
            )

        except Exception as ml_error:
            # ML processing failed but don't fail the entire request
            logger.warning(f"ML processing failed for quick feedback: {str(ml_error)}")

        logger.info(
            f"User {request.user.username} submitted quick feedback ({'helpful' if was_helpful else 'not helpful'}) for analysis {analysis_id}"
        )

        return JsonResponse(
            {
                "success": True,
                "message": f"Thank you for your feedback! This helps us improve QueryGrade.",
                "feedback_type": "helpful" if was_helpful else "not_helpful",
            }
        )

    except json.JSONDecodeError:
        return JsonResponse(
            {"success": False, "error": "Invalid JSON data"}, status=400
        )

    except Exception as e:
        logger.error(
            f"Error processing quick feedback for analysis {analysis_id}: {str(e)}"
        )
        return JsonResponse(
            {
                "success": False,
                "error": "An error occurred while processing your feedback. Please try again.",
            },
            status=500,
        )


@login_required
def feedback_analytics(request):
    """View feedback analytics (admin only)."""
    if not request.user.is_staff:
        messages.error(request, "You don't have permission to view feedback analytics.")
        return redirect("grade_query")

    # Get feedback statistics
    feedback_data = QueryFeedback.objects.all()

    statistics = {
        "total_feedback": feedback_data.count(),
        "avg_accuracy": feedback_data.aggregate(Avg("accuracy_rating"))[
            "accuracy_rating__avg"
        ]
        or 0,
        "avg_usefulness": feedback_data.aggregate(Avg("usefulness_rating"))[
            "usefulness_rating__avg"
        ]
        or 0,
        "avg_clarity": feedback_data.aggregate(Avg("clarity_rating"))[
            "clarity_rating__avg"
        ]
        or 0,
        "would_recommend_count": feedback_data.filter(would_recommend=True).count(),
        "rating_distribution": feedback_data.values("accuracy_rating").annotate(
            count=Count("id")
        ),
    }

    context = {
        "statistics": statistics,
        "recent_feedback": feedback_data.select_related(
            "user_history", "user_history__user"
        ).order_by("-created_at")[:10],
    }

    return render(request, "analyzer/feedback_analytics.html", context)
